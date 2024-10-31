from __future__ import annotations

import gc
import random
import sys
from typing import Any

import torch
from tqdm import tqdm
from tqdm.auto import trange

from .config import Config
from .guided_model import GuidedModel
from .schedule import Schedule
from .tensor_image_ops import (
    blend_wavelets,
    scale_wavelets,
)
from .utils import fallback


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
        _params,
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
        self._params = _params
        self.cfg = self.base_config = Config(
            initial_x.device,
            initial_x.dtype,
            self.latent_format,
            _params=_params,
            **kwargs,
        )
        self.highres_sigmas: None | torch.FloatTensor = None
        self.reference_image: None | torch.FloatTensor = None
        self.guidance_waves = None
        self.seed_offset = 0
        self.seed = self.extra_args.get("seed")
        if self.cfg.seed_rng:
            seed = self.seed
            if seed is not None:
                torch.manual_seed(seed)
                random.seed(seed)
                for _ in range(self.cfg.seed_rng_offset):
                    _ = random.random()  # noqa: S311
                    _ = torch.randn_like(initial_x)
                    self.seed_offset += 1
        if self.seed is None:
            self.seed = 0

    def __getattr__(self, key: str) -> Any:  # noqa: ANN401
        return getattr(self.cfg, key)

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

    def add_restart_noise(
        self,
        x: torch.Tensor,
        sigma_min: float | torch.Tensor,
        sigma_max: float | torch.Tensor,
        *,
        s_noise: float = 1.0,
    ) -> torch.Tensor:
        noise_factor = (sigma_max**2 - sigma_min**2) ** 0.5
        return self.add_noise(
            x,
            noise_factor,
            factor=s_noise,
            noise_sampler=self.restart_noise_sampler,
        )

    def add_noise(
        self,
        latent: torch.Tensor,
        sigma: float | torch.Tensor,
        *,
        sigma_next: None | float | torch.Tensor = None,
        factor=1.0,
        allow_max_denoise=True,
        noise_sampler: None | callable = None,
    ) -> torch.Tensor:
        self.seed_offset += 1
        sigma_next = fallback(sigma_next, sigma)
        noise_sampler = fallback(noise_sampler, self.noise_sampler)
        noise = noise_sampler(sigma, sigma_next)
        if factor != 1:
            noise *= factor
        return self.model_sampling.noise_scaling(
            sigma,
            noise,
            latent,
            max_denoise=allow_max_denoise
            and sigma >= self.model_sampling.sigma_max - 1e-05,
        )

    def run_sampler_with_pbar(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        pbar_title: str,
        *args: list,
        **kwargs: dict,
    ):
        with tqdm(
            disable=self.disable_pbar,
            total=1,
            desc=f"{pbar_title} ({len(sigmas) - 1}) {float(sigmas[0]):>2.03f} ... {float(sigmas[-1]):>2.03f}",
        ) as pbar:
            x = self.run_sampler(x, sigmas, *args, **kwargs)
            pbar.update()
        return x

    def run_steps(
        self,
        x: torch.Tensor,
        *,
        sigmas: None | torch.Tensor = None,
    ) -> torch.Tensor:
        sigmas = self.highres_sigmas if sigmas is None else sigmas
        sigmas_len = len(sigmas)
        if sigmas_len < 2:
            return x
        guidance_steps = max(0, min(self.guidance_steps, sigmas_len - 1))
        guidance_sigmas = sigmas[: guidance_steps + 1]
        guidance_sigmas_len = len(guidance_sigmas)
        normal_sigmas = sigmas[guidance_steps:]
        normal_sigmas_len = len(normal_sigmas)
        if guidance_sigmas_len < 2 and normal_sigmas_len < 2:
            return x
        guided_model = GuidedModel(self, guidance_sigmas, guidance_steps)
        model_wrapper = guided_model.make_wrapper()

        same_samplers = self.sampler == self.guidance_sampler

        if self.guidance_restart == 0 and same_samplers and self.chunked_sampling:
            # No guidance restarts and the guidance sampler is the same as the
            # # normal one and chunked mode enabled - we can sample all the sigmas at once.
            return self.run_sampler_with_pbar(
                x,
                sigmas,
                model=model_wrapper,
                pbar_title="combined steps",
            )

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
            if (
                same_samplers
                and self.chunked_sampling
                and repidx == self.guidance_restart
            ):
                # On the last guidance restart iteration, we can sample all the sigmas at once
                # as long as the guidance sampler is the same as the normal one and we're in
                # chunked mode.
                return self.run_sampler_with_pbar(
                    x,
                    sigmas,
                    model=model_wrapper,
                    pbar_title="combined steps",
                )
            if guidance_sigmas_len < 2:
                continue
            if self.chunked_sampling:
                guided_model.force_guidance = -1
                x = self.run_sampler_with_pbar(
                    x,
                    guidance_sigmas,
                    model=model_wrapper,
                    sampler=self.guidance_sampler,
                    pbar_title="guidance steps",
                )
                guided_model.force_guidance = None
                continue
            with trange(
                guidance_steps,
                initial=1,
                disable=self.disable_pbar,
                desc="guidance step",
            ) as pbar:
                for idx in pbar:
                    guided_model.force_guidance = idx
                    step_sigmas = guidance_sigmas[idx : idx + 2]
                    if step_sigmas[-1] > step_sigmas[0]:
                        raise ValueError(
                            "Hit out-of-order sigma, likely due to restart sigmas in guidance step range",
                        )
                    pbar.set_description(
                        f"guidance step {float(step_sigmas[0]):>2.03f} -> {float(step_sigmas[-1]):>2.03f}",
                    )
                    x = self.run_sampler(
                        x,
                        step_sigmas,
                        model=model_wrapper,
                        sampler=self.guidance_sampler,
                        disable_pbar=True,
                    )
            guided_model.force_guidance = None
        if normal_sigmas_len >= 2:
            guided_model.allow_guidance = False
            x = self.run_sampler_with_pbar(
                x,
                normal_sigmas,
                model=model_wrapper,
                pbar_title="normal steps",
            )
        return x

    @classmethod
    def unbork_brownian_noise(cls):
        kds = sys.modules.get("comfy.k_diffusion.sampling")
        if kds is None:
            return
        btns = getattr(kds, "BrownianTreeNoiseSampler", None)
        if btns is None:
            return
        pc_reset = getattr(btns, "pc_reset", None)
        if pc_reset is not None:
            # Curse you, prompt-control!
            pc_reset()

    def get_extra_args(self):
        if self.seed is None:
            return self.extra_args.copy()
        return self.extra_args | {"seed": self.seed + self.seed_offset}

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
        self.unbork_brownian_noise()
        return sampler.sampler_function(
            fallback(model, self.model),
            x,
            sigmas,
            callback=self.callback if not self.skip_callback else None,
            extra_args=self.get_extra_args(),
            disable=disable_pbar or self.disable_pbar,
            **sampler.extra_options,
        )

    def set_schedule(self) -> None:
        if self.schedule_override is not None:
            self.highres_sigmas = Schedule(
                self.model_sampling,
                self.latent_format,
                **self.schedule_override,
            ).sigmas.to(self.sigmas)
            return
        highres_sigmas = self._params.get_item(
            "sigmas",
            name=self.cfg.highres_sigmas_name,
        )
        if highres_sigmas is None:
            self.highres_sigmas = Schedule(
                self.model_sampling,
                self.latent_format,
                "karras",
                steps=15,
                denoise=0.3,
            ).sigmas.to(self.sigmas)
            return
        self.highres_sigmas = highres_sigmas.detach().clone().to(self.sigmas)

    def update_iteration_config(self, iteration: int | str) -> None:
        self.cfg = self.base_config.get_iteration_config(iteration)
        self.set_schedule()
        if iteration == "reference":
            self.reference_image = self._params.get_item(
                "image",
                name=self.cfg.reference_image_name,
            )

    def get_noise_sampler(self, x, sigmas, *, name=""):
        custom_noise = self._params.get_item("custom_noise", name=name)
        if custom_noise is None:
            return lambda _s, _sn: torch.randn_like(x)
        custom_noise_params = self._params.get_item(
            "custom_noise",
            name=name,
            param_mode=True,
            default={},
        )
        custom_noise_params = {
            "normalized": True,
            "seed": self.seed + self.seed_offset,
            "cpu": True,
        } | custom_noise_params
        self.seed_offset += 1
        return custom_noise.make_noise_sampler(
            x,
            sigmas.min(),
            sigmas.max(),
            **custom_noise_params,
        )

    def update_noise_samplers(self, x):
        self.noise_sampler = self.get_noise_sampler(
            x,
            self.highres_sigmas,
            name=self.cfg.custom_noise_name,
        )
        if self.guidance_restart == 0:
            self.restart_noise_sampler = None
            return
        self.restart_noise_sampler = self.get_noise_sampler(
            x,
            self.highres_sigmas,
            name=self.cfg.restart_custom_noise_name,
        )

    def __call__(self) -> torch.Tensor:
        self.update_iteration_config("reference")
        if self.reference_image is None:
            if self.sigmas[-1] != 0:
                raise ValueError(
                    "Initial reference sigmas must end at 0 (full denoise)",
                )
            normal_step_count = len(self.sigmas) - 1
            with tqdm(
                disable=self.disable_pbar,
                desc=f"normal steps ({normal_step_count}) {float(self.sigmas[0]):>2.03f} ... {float(self.sigmas[-1]):>2.03f}",
            ):
                x_lr = self.run_sampler(
                    self.initial_x,
                    self.sigmas,
                    sampler=self.reference_sampler,
                )
                self.seed_offset += normal_step_count
            if self.iterations < 1:
                return x_lr
            self.reference_image = self.vae.decode(x_lr, disable_pbar=self.disable_pbar)
        elif self.iterations < 1:
            return self.vae.encode(self.reference_image, disable_pbar=self.disable_pbar)
        self.cfg = self.base_config
        x_new = None
        for iteration in trange(
            self.iterations,
            disable=self.disable_pbar,
            initial=1,
            desc="DiffuseHigh iteration",
        ):
            self.update_iteration_config(iteration)
            del x_new
            if self.highres_sigmas[-1] != 0:
                raise ValueError(
                    "Highres sigmas (including schedule overrides) must end at 0 (full denoise)",
                )
            self.gc()
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
            self.update_noise_samplers(x_new)
            x_new = self.add_noise(
                x_new,
                self.highres_sigmas[0],
                sigma_next=self.highres_sigmas[1],
                factor=self.renoise_factor,
            )
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
