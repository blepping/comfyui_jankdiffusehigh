from __future__ import annotations

import torch
from tqdm import tqdm
from tqdm.auto import trange

from .config import Config
from .tensor_image_ops import (
    blend_wavelets,
    scale_wavelets,
)
from .utils import ensure_model, fallback


class DiffuseHighSampler:
    def __init__(
        self,
        model,
        initial_x,
        sigmas,
        *,
        callback,
        extra_args,
        disable_pbar,
        highres_sigmas,
        guidance_sampler_opt=None,
        reference_sampler_opt=None,
        reference_image_opt=None,
        vae_opt=None,
        upscale_model_opt=None,
        **kwargs: dict,
    ):
        self.s_in = initial_x.new_ones((initial_x.shape[0],))
        self.initial_x = initial_x
        self.callback = callback
        self.disable_pbar = disable_pbar
        self.sigmas = sigmas
        self.extra_args = fallback(extra_args, {})
        self.model = model
        self.latent_format = model.inner_model.inner_model.latent_format
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
        self.highres_sigmas = highres_sigmas.detach().clone().to(sigmas)
        self.reference_image = reference_image_opt
        self.guidance_waves = None

    def __getattr__(self, key):
        return getattr(self.config, key)

    def apply_guidance(self, idx, denoised):
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
        # tqdm.write(str(("GUIDE OUT", denoised.shape, result.shape, mix_scale)))
        if self.blend_by_mode != "latent":
            return result.to(denoised)
        return self.blend_function(
            denoised,
            result.to(denoised),
            denoised.new_full((1,), mix_scale),
        )

    def run_steps(self, *, x=None, sigmas=None):
        x = self.initial_x if x is None else x
        sigmas = self.sigmas if sigmas is None else sigmas
        soffset = self.sigma_offset
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
                noise_factor = (
                    guidance_sigmas[0] ** 2 - guidance_sigmas[-1] ** 2
                ) ** 0.5
                x = x + torch.randn_like(x) * (
                    noise_factor * self.guidance_restart_s_noise
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

    def run_sampler(self, x, sigmas, *, model=None, sampler=None, disable_pbar=False):
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

    def __call__(self):
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
        for iteration in trange(
            self.iterations,
            disable=self.disable_pbar,
            initial=1,
            desc="DiffuseHigh iteration",
        ):
            self.config = self.base_config.get_iteration_config(iteration)
            if self.config.sigma_offset >= len(self.highres_sigmas) - 1:
                raise ValueError(
                    "Bad sigma_offset: posts to sigma past penultimate sigma",
                )
            with tqdm(disable=self.disable_pbar, total=1, desc="upscale") as pbar:
                img_hr = self.upscale(
                    self.reference_image,
                    self.scale_factor,
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
                self.guidance_waves = self.dwt(x_new.clone())
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
            result = self.run_steps(x=x_new, sigmas=self.highres_sigmas)
            if iteration == self.iterations - 1:
                break
            self.reference_image = self.vae.decode(
                result,
                disable_pbar=self.disable_pbar,
            )
        return result


def diffusehigh_sampler(
    model,
    x,
    sigmas,
    *,
    diffusehigh_options,
    disable=None,
    extra_args=None,
    callback=None,
):
    sampler = DiffuseHighSampler(
        model,
        x,
        sigmas,
        disable_pbar=disable,
        callback=callback,
        extra_args=extra_args,
        **diffusehigh_options,
    )
    return sampler()
