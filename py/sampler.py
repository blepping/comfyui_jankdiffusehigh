from __future__ import annotations

import gc
import random
from typing import Any

import torch
from tqdm import tqdm
from tqdm.auto import trange

from .config import Config
from .schedule import Schedule
from .tensor_image_ops import (
    blend_wavelets,
    scale_wavelets,
)
from .utils import ensure_model, fallback


class DiffuseHighSampler:
    def __init__(
        self,
        model: object,
        initial_x: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        callback: None | callable,
        extra_args: None | dict,
        disable_pbar: None | bool,
        highres_sigmas: None | torch.Tensor = None,
        guidance_sampler_opt: None | object = None,
        reference_sampler_opt: None | object = None,
        reference_image_opt: None | object = None,
        vae_opt: None | object = None,
        upscale_model_opt: None | object = None,
        **kwargs: dict[str, Any],
    ):
        self.s_in = initial_x.new_ones((initial_x.shape[0],))
        self.initial_x = initial_x
        self.callback = callback
        self.disable_pbar = disable_pbar
        self.sigmas = sigmas
        self.extra_args = fallback(extra_args, {})
        self.model = model
        self.latent_format = model.inner_model.inner_model.latent_format
        self.model_sampling = model.inner_model.inner_model.model_sampling
        self.config = self.base_config = Config(
            initial_x.device,
            initial_x.dtype,
            self.latent_format,
            guidance_sampler=guidance_sampler_opt,
            reference_sampler=reference_sampler_opt,
            vae=vae_opt,
            upscale_model=upscale_model_opt,
            **kwargs,
        )
        if highres_sigmas is not None:
            self.highres_sigmas_input = highres_sigmas.detach().clone().to(sigmas)
        else:
            self.highres_sigmas_input = (
                Schedule(
                    self.model_sampling,
                    self.latent_format,
                    "karras",
                    steps=50,
                )
                .sigmas[-16:]
                .to(sigmas)
            )
        self.highres_sigmas: None | torch.FloatTensor = None
        self.reference_image: None | torch.FloatTensor = reference_image_opt
        self.guidance_waves = None
        if self.config.seed_rng:
            seed = self.extra_args.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                random.seed(seed)
                for _ in range(self.config.seed_rng_offset):
                    _ = random.random()  # noqa: S311
                    _ = torch.randn_like(initial_x)

    def __getattr__(self, key: str) -> Any:  # noqa: ANN401
        return getattr(self.config, key)

    def gc(self) -> None:
        if (
            self.enable_cache_clearing
            and hasattr(torch, "cuda")
            and hasattr(torch.cuda, "empty_cache")
        ):
            torch.cuda.empty_cache()
        if self.enable_gc:
            gc.collect()

    def apply_guidance(self, idx: int, denoised: torch.Tensor) -> torch.Tensor:
        if self.guidance_waves is None or idx >= self.guidance_steps:
            return denoised
        mix_scale = (
            self.guidance_factor
            - ((self.guidance_factor / self.guidance_steps) * idx) * self.fadeout_factor
        )
        if mix_scale == 0:
            return denoised
        if self.guidance_mode not in {"image", "latent"}:
            raise ValueError("Bad guidance mode")
        if self.guidance_mode == "image":
            dn_img = (
                self.vae.decode(denoised, disable_pbar=self.disable_pbar)
                .to(denoised)
                .movedim(-1, 1)
            )
            denoised_waves = self.dwt(dn_img)
        elif self.guidance_mode == "latent":
            denoised_waves = self.dwt(denoised)
        denoised_waves_orig = denoised_waves
        if self.denoised_wavelet_multiplier != 1:
            denoised_waves = scale_wavelets(self.denoised_wavelet_multiplier)
        coeffs = (
            (self.guidance_waves[0], denoised_waves[1])
            if not self.dwt_flip_filters
            else (denoised_waves[0], self.guidance_waves[1])
        )
        if self.blend_by_mode == "wavelet" or (
            self.blend_by_mode == "image" and self.guidance_mode != "image"
        ):
            coeffs = blend_wavelets(
                denoised_waves_orig,
                coeffs,
                mix_scale,
                self.blend_function,
            )
        result = self.idwt(coeffs)
        if self.guidance_mode == "image":
            if self.blend_by_mode == "image":
                result = self.blend_function(
                    dn_img,
                    result.to(dn_img),
                    dn_img.new_full((1,), mix_scale),
                ).clamp_(0, 1)
            result = self.vae.encode(
                result.cpu(),
                fix_dims=True,
                disable_pbar=self.disable_pbar,
            )
        if self.blend_by_mode != "latent":
            return result.to(denoised)
        return self.blend_function(
            denoised,
            result.to(denoised),
            denoised.new_full((1,), mix_scale),
        )

    def set_schedule(self) -> None:
        if self.schedule_override is None:
            self.highres_sigmas = self.highres_sigmas_input
            return
        self.highres_sigmas = Schedule(
            self.model_sampling,
            self.latent_format,
            **self.schedule_override,
        ).sigmas.to(self.highres_sigmas_input)

    @classmethod
    def add_restart_noise(
        cls,
        x: torch.Tensor,
        sigma_min: float | torch.Tensor,
        sigma_max: float | torch.Tensor,
        *,
        s_noise: float = 1.0,
    ) -> torch.Tensor:
        noise_factor = (sigma_max**2 - sigma_min**2) ** 0.5
        return x + torch.randn_like(x).mul_(noise_factor * s_noise)

    def run_steps(
        self,
        x: torch.Tensor,
        *,
        sigmas: None | torch.Tensor = None,
    ) -> torch.Tensor:
        sigmas = self.highres_sigmas if sigmas is None else sigmas
        soffset = (
            self.sigma_offset
            if self.sigma_offset >= 0
            else len(sigmas) + self.sigma_offset
        )
        guidance_sigmas = sigmas[soffset : soffset + self.guidance_steps + 1]
        normal_sigmas = sigmas[soffset + self.guidance_steps :]
        step_idx = 0
        model = self.model

        def model_wrapper(x, sigma, **extra_args: dict):
            nonlocal step_idx
            ensure_model(model)
            denoised = model(x, sigma, **extra_args)
            return self.apply_guidance(step_idx, denoised)

        for k in (
            "inner_model",
            "sigmas",
        ):
            if hasattr(model, k):
                setattr(model_wrapper, k, getattr(model, k))

        for repidx in trange(
            self.guidance_restart + 1,
            initial=1,
            disable=self.guidance_restart < 1 or self.disable_pbar,
            desc="guidance steps iteration",
        ):
            if repidx > 0:
                x = self.add_restart_noise(
                    x,
                    guidance_sigmas[-1],
                    guidance_sigmas[0],
                    s_noise=self.guidance_restart_s_noise,
                )
            guidance_steps = len(guidance_sigmas) - 1
            for idx in trange(
                guidance_steps,
                initial=1,
                disable=self.disable_pbar,
                desc="guidance step",
            ):
                step_idx = idx
                x = self.run_sampler(
                    x,
                    guidance_sigmas[idx : idx + 2],
                    model=model_wrapper,
                    sampler=self.guidance_sampler,
                    disable_pbar=True,
                )
        if len(normal_sigmas) > 1:
            ensure_model(model)
            with tqdm(disable=self.disable_pbar, total=1, desc="normal steps") as pbar:
                x = self.run_sampler(x, normal_sigmas)
                pbar.update()
        return x

    def run_sampler(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        model: None | object = None,
        sampler: None | object = None,
        disable_pbar: bool = False,
    ):
        sampler = fallback(sampler, self.sampler)
        return sampler.sampler_function(
            fallback(model, self.model),
            x,
            sigmas,
            callback=self.callback if not self.skip_callback else None,
            extra_args=self.extra_args.copy(),
            disable=disable_pbar or self.disable_pbar,
            **sampler.extra_options,
        )

    def __call__(self) -> torch.Tensor:
        self.config = self.base_config.get_iteration_config("reference")
        if self.reference_image is None:
            with tqdm(disable=self.disable_pbar, desc="reference steps"):
                x_lr = self.run_sampler(
                    self.initial_x,
                    self.sigmas,
                    sampler=self.reference_sampler,
                )
            if self.iterations < 1:
                return x_lr
            self.reference_image = self.vae.decode(x_lr, disable_pbar=self.disable_pbar)
        elif self.iterations < 1:
            return self.vae.encode(self.reference_image, disable_pbar=self.disable_pbar)
        self.config = self.base_config
        x_new = None
        for iteration in trange(
            self.iterations,
            disable=self.disable_pbar,
            initial=1,
            desc="DiffuseHigh iteration",
        ):
            del x_new
            self.config = self.base_config.get_iteration_config(iteration)
            self.set_schedule()
            self.gc()
            if self.config.sigma_offset >= len(self.highres_sigmas) - 1:
                raise ValueError(
                    "Bad sigma_offset: points to sigma past penultimate sigma",
                )
            if self.config.sigma_offset < 0 and (
                self.config_sigma_offset == -1
                or abs(self.config.sigma_offset) >= len(self.highres_sigmas)
            ):
                raise ValueError(
                    "Negative sigma_offset can't point to last sigma or to sigma less than index 0",
                )
            with tqdm(disable=self.disable_pbar, total=1, desc="upscale") as pbar:
                img_hr = self.upscale(
                    self.reference_image,
                    self.scale_factor,
                    use_upscale_model=self.use_upscale_model,
                    pbar=pbar,
                )
                pbar.update()
            self.reference_image = self.sharpen(img_hr, fix_dims=True)
            x_new = self.vae.encode(
                self.reference_image,
                disable_pbar=self.disable_pbar,
            ).to(self.initial_x)
            if self.guidance_mode == "image":
                self.guidance_waves = self.dwt(
                    self.reference_image.clone().movedim(-1, 1).to(self.initial_x),
                )
            elif self.guidance_mode == "latent":
                self.guidance_waves = self.dwt(x_new)
                if self.reference_wavelet_multiplier != 1:
                    self.guidance_waves = scale_wavelets(
                        self.guidance_waves,
                        self.reference_wavelet_multiplier,
                    )
            else:
                raise ValueError("Bad guidance_mode")
            x_noise = torch.randn_like(x_new)
            x_new = x_new + x_noise * (
                self.highres_sigmas[self.sigma_offset] * self.renoise_factor
            )
            del x_noise
            # x_new = self.model.inner_model.inner_model.model_sampling.noise_scaling(
            #     self.highres_sigmas[0] * self.renoise_factor,
            #     x_noise,
            #     x_new,
            # )
            self.gc()
            x_new = self.run_steps(x_new, sigmas=self.highres_sigmas)
            if iteration == self.iterations - 1:
                break
            self.reference_image = self.vae.decode(
                x_new,
                disable_pbar=self.disable_pbar,
            )
        return x_new


def diffusehigh_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    diffusehigh_options: dict[str, Any],
    disable: None | bool = None,
    extra_args: None | dict[str, Any] = None,
    callback: None | callable = None,
) -> torch.Tensor:
    return DiffuseHighSampler(
        model,
        x,
        sigmas,
        disable_pbar=disable,
        callback=callback,
        extra_args=extra_args,
        **diffusehigh_options,
    )()
