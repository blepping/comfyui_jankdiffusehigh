from __future__ import annotations

import math

import PIL.Image as PILImage
import torch
import torchvision
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse
from tqdm.auto import trange

from .utils import (
    ensure_model,
    pilimgbatch_to_torch,
    torch_to_pilimgbatch,
)
from .vae import VAEHelper


def gaussian_blur_image_sharpening(image, kernel_size=3, sigma=(0.1, 2.0), alpha=1):
    gaussian_blur = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size,
        sigma=sigma,
    )
    image_blurred = gaussian_blur(image)
    return (alpha + 1) * image - alpha * image_blurred


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
        sampler,
        guidance_steps=5,
        guidance_mode="image",
        guidance_factor=1.0,
        guidance_restart=0,
        guidance_restart_s_noise=1.0,
        fadeout_factor=0.0,
        scale_factor=2.0,
        renoise_factor=1.0,
        iterations=1,
        vae_mode="normal",
        dwt_level=1,
        dwt_wave="db4",
        dwt_mode="symmetric",
        dwt_flip_filters=False,
        dtcwt_mode=False,
        dtcwt_biort="near_sym_a",
        dtcwt_qshift="qshift_a",
        reference_wavelet_multiplier=1.0,
        denoised_wavelet_multiplier=1.0,
        sharpen_reference=True,
        sharpen_kernel_size=3,
        sharpen_sigma=(0.1, 2.0),
        sharpen_alpha=1.0,
        resample_mode="bicubic",
        rescale_increment=64,
        guidance_sampler_opt=None,
        reference_sampler_opt=None,
        reference_image_opt=None,
        vae_opt=None,
        upscale_model_opt=None,
    ):
        self.s_in = initial_x.new_ones((initial_x.shape[0],))
        self.initial_x = initial_x
        self.callback = callback
        self.disable_pbar = disable_pbar
        self.sigmas = sigmas
        self.extra_args = extra_args if extra_args is not None else {}
        self.model = model
        self.latent_format = model.inner_model.inner_model.latent_format
        self.fadeout_factor = fadeout_factor
        self.scale_factor = scale_factor
        self.guidance_factor = guidance_factor
        self.renoise_factor = renoise_factor
        self.iterations = iterations
        self.highres_sigmas = highres_sigmas.clone().to(sigmas)
        self.guidance_steps = guidance_steps
        self.guidance_mode = guidance_mode
        self.guidance_restart = guidance_restart
        self.guidance_restart_s_noise = guidance_restart_s_noise
        self.sampler = sampler
        self.guidance_sampler = guidance_sampler_opt or sampler
        self.reference_sampler = reference_sampler_opt or sampler
        self.vae = VAEHelper(
            vae_mode,
            self.latent_format,
            device=initial_x.device,
            dtype=initial_x.dtype,
            vae=vae_opt,
        )
        self.reference_image = reference_image_opt
        self.sharpen_reference = sharpen_reference
        self.sharpen_kernel_size = sharpen_kernel_size
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_alpha = sharpen_alpha
        self.resample_mode = getattr(PILImage, resample_mode.upper())
        self.rescale_increment = self.scale_dim(
            max(8, rescale_increment),
            1,
            increment=8,
        )
        if dtcwt_mode:
            self.dwt = DTCWTForward(
                J=dwt_level,
                mode=dwt_mode,
                biort=dtcwt_biort,
                qshift=dtcwt_qshift,
            ).to(
                initial_x.device,
            )
            self.idwt = DTCWTInverse(
                mode=dwt_mode,
                biort=dtcwt_biort,
                qshift=dtcwt_qshift,
            ).to(initial_x.device)
        else:
            self.dwt = DWTForward(J=dwt_level, wave=dwt_wave, mode=dwt_mode).to(
                initial_x.device,
            )
            self.idwt = DWTInverse(wave=dwt_wave, mode=dwt_mode).to(initial_x.device)
        self.dwt_flip_filters = dwt_flip_filters
        self.reference_wavelet_multiplier = reference_wavelet_multiplier
        self.denoised_wavelet_multiplier = denoised_wavelet_multiplier
        self.guidance_waves = None
        self.guidance_latent = None
        self.upscale_model = upscale_model_opt

    def call_model(self, x, sigma):
        return self.model(x, sigma * self.s_in, **self.extra_args)

    def do_callback(self, idx, x, sigma, denoised):
        if self.callback is None:
            return
        self.callback({
            "i": idx,
            "x": x,
            "sigma": sigma,
            "sigma_hat": sigma,
            "denoised": denoised,
        })

    def apply_guidance(self, idx, denoised):
        if self.guidance_waves is None or idx >= self.guidance_steps:
            return denoised
        mix_scale = (
            self.guidance_factor
            - ((self.guidance_factor / (self.guidance_steps + 1)) * idx)
            * self.fadeout_factor
        )
        if mix_scale == 0:
            return denoised
        print("GUIDANCE APPLY", idx)
        if self.guidance_mode not in {"image", "latent"}:
            raise ValueError("ohno")
        if self.guidance_mode == "image":
            dn_img = self.vae.decode(denoised).to(denoised).movedim(-1, 1)
            print("DN_IMG", dn_img.shape)
            denoised_waves = self.dwt(dn_img)
            del dn_img
        elif self.guidance_mode == "latent":
            denoised_waves = self.dwt(denoised)
        if self.denoised_wavelet_multiplier != 1:
            denoised_waves = (
                denoised_waves[0] * self.denoised_wavelet_multiplier,
                tuple(t * self.denoised_wavelet_multiplier for t in denoised_waves[1]),
            )
        coeffs = (
            (self.guidance_waves[0], denoised_waves[1])
            if not self.dwt_flip_filters
            else (denoised_waves[0], self.guidance_waves[1])
        )
        result = self.idwt(coeffs)
        if self.guidance_mode == "image":
            result = self.vae.encode(result.cpu(), fix_dims=True)

        print("GUIDE OUT", denoised.shape, result.shape, mix_scale)
        return torch.lerp(denoised, result.to(denoised), mix_scale)

    def run_steps(self, *, x=None, sigmas=None):
        x = self.initial_x if x is None else x
        sigmas = self.sigmas if sigmas is None else sigmas
        guidance_sigmas = sigmas[: self.guidance_steps + 1]
        normal_sigmas = sigmas[self.guidance_steps :]
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

        for repidx in range(self.guidance_restart + 1):
            if repidx > 0:
                noise_factor = (
                    guidance_sigmas[0] ** 2 - guidance_sigmas[-1] ** 2
                ) ** 0.5
                x = x + torch.randn_like(x) * (
                    noise_factor * self.guidance_restart_s_noise
                )
            for idx in range(len(guidance_sigmas) - 1):
                step_idx = idx
                x = self.run_sampler(
                    x,
                    guidance_sigmas[idx : idx + 2],
                    model=model_wrapper,
                    sampler=self.guidance_sampler,
                )
        if len(normal_sigmas) > 1:
            ensure_model(model)
            x = self.run_sampler(x, normal_sigmas)
        return x

    @staticmethod
    def scale_dim(n, factor, *, increment=64) -> int:
        return math.ceil((n * factor) / increment) * increment

    def upscale(self, imgbatch):
        _batch, height, width, _channels = imgbatch.shape
        target_height = self.scale_dim(
            height,
            self.scale_factor,
            increment=self.rescale_increment,
        )
        target_width = self.scale_dim(
            width,
            self.scale_factor,
            increment=self.rescale_increment,
        )
        print(f">> UPSCALE: {width}x{height} -> {target_width}x{target_height}")
        if (target_height, target_width) == (height, width):
            return imgbatch
        if self.upscale_model is not None:
            print("** Upscaling with model")
            imgbatch = ImageUpscaleWithModel().upscale(self.upscale_model, imgbatch)[0]
            if imgbatch.shape[1:3] == (target_height, target_width):
                return imgbatch
        print(
            f"** PIL upscale {imgbatch.shape[2]}x{imgbatch.shape[1]} -> {target_width}x{target_height}",
        )
        ref_imgbatch = torch_to_pilimgbatch(self.reference_image)
        return pilimgbatch_to_torch(
            tuple(
                i.resize((target_width, target_height), resample=self.resample_mode)
                for i in ref_imgbatch
            ),
        )

    def run_sampler(self, x, sigmas, *, model=None, sampler=None):
        model = model or self.model
        sampler = sampler or self.sampler
        return sampler.sampler_function(
            model,
            x,
            sigmas,
            callback=self.callback,
            extra_args=self.extra_args.copy(),
            disable=self.disable_pbar,
            **sampler.extra_options,
        )

    def __call__(self):
        if self.reference_image is None:
            x_lr = self.run_sampler(
                self.initial_x,
                self.sigmas,
                sampler=self.reference_sampler,
            )
            if self.iterations < 1:
                return x_lr
            self.reference_image = self.vae.decode(x_lr)
        elif self.iterations < 1:
            return self.vae.encode(self.reference_image)
        for iteration in trange(self.iterations, disable=self.disable_pbar):
            print(
                f"\nIT({iteration}): shp={self.reference_image.shape}, min={self.reference_image.min()}, max={self.reference_image.max()}",
            )
            img_hr = self.upscale(self.reference_image)
            print("IMG_HR", img_hr.shape)
            if self.sharpen_reference:
                img_hr = gaussian_blur_image_sharpening(
                    img_hr.movedim(-1, 1),
                    kernel_size=self.sharpen_kernel_size,
                    sigma=self.sharpen_sigma,
                    alpha=self.sharpen_alpha,
                ).movedim(1, -1)
            self.reference_image = img_hr
            x_new = self.vae.encode(self.reference_image).to(self.initial_x)
            self.guidance_latent = x_new.clone()
            if self.guidance_mode == "image":
                print("REF IMG", self.reference_image.shape)
                self.guidance_waves = self.dwt(
                    self.reference_image.clone().movedim(-1, 1).to(self.initial_x),
                )
            elif self.guidance_mode == "latent":
                self.guidance_waves = self.dwt(self.guidance_latent)
                if self.reference_wavelet_multiplier != 1:
                    self.guidance_waves = (
                        self.guidance_waves[0] * self.reference_wavelet_multiplier,
                        tuple(
                            t * self.reference_wavelet_multiplier
                            for t in self.guidance_waves[1]
                        ),
                    )
            else:
                raise ValueError("ohno")
            x_noise = torch.randn_like(x_new)
            x_new = x_new + x_noise * (self.highres_sigmas[0] * self.renoise_factor)
            # x_new = self.model.inner_model.inner_model.model_sampling.noise_scaling(
            #     self.highres_sigmas[0] * self.renoise_factor,
            #     x_noise,
            #     x_new,
            # )
            result = self.run_steps(x=x_new, sigmas=self.highres_sigmas)
            if iteration == self.iterations - 1:
                break
            self.reference_image = self.vae.decode(result)
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
