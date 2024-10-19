from __future__ import annotations

from comfy.samplers import ksampler
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse

from .tensor_image_ops import (
    BLENDING_MODES,
    Sharpen,
)
from .upscale import Upscale
from .utils import fallback
from .vae import VAEHelper


class Config:
    _overridable_fields = {  # noqa: RUF012
        "blend_by_mode",
        "blend_mode",
        "denoised_wavelet_multiplier",
        "dtcwt_biort",
        "dtcwt_mode",
        "dtcwt_qshift",
        "dwt_flip_filters",
        "dwt_level",
        "dwt_mode",
        "dwt_wave",
        "enable_cache_clearing",
        "enable_gc",
        "fadeout_factor",
        "guidance_factor",
        "guidance_mode",
        "guidance_restart_s_noise",
        "guidance_restart",
        "guidance_steps",
        "reference_wavelet_multiplier",
        "renoise_factor",
        "resample_mode",
        "rescale_increment",
        "scale_factor",
        "schedule_override",
        "sharpen_gaussian_kernel_size",
        "sharpen_gaussian_sigma",
        "sharpen_mode",
        "sharpen_reference",
        "sharpen_strength",
        "skip_callback",
        "sigma_offset",
        "use_upscale_model",
        "vae_decode_kwargs",
        "vae_encode_kwargs",
        "vae_mode",
    }

    _dict_exclude_keys = {  # noqa: RUF012
        "as_dict",
        "blend_function",
        "dwt",
        "get_iteration_config",
        "idwt",
        "iteration_override",
        "sharpen",
        "upscale",
        "vae",
    }

    def __init__(
        self,
        device,
        dtype,
        latent_format,
        *,
        blend_mode="lerp",
        blend_by_mode="image",
        denoised_wavelet_multiplier=1.0,
        dtcwt_biort="near_sym_a",
        dtcwt_mode=False,
        dtcwt_qshift="qshift_a",
        dwt_flip_filters=False,
        dwt_level=1,
        dwt_mode="symmetric",
        dwt_wave="db4",
        enable_gc=True,
        enable_cache_clearing=True,
        fadeout_factor=0.0,
        guidance_factor=1.0,
        guidance_mode="image",
        guidance_restart_s_noise=1.0,
        guidance_restart=0,
        guidance_sampler=None,
        guidance_steps=5,
        iteration_override=None,
        iterations=1,
        reference_sampler=None,
        reference_wavelet_multiplier=1.0,
        renoise_factor=1.0,
        resample_mode="bicubic",
        rescale_increment=64,
        sampler=None,
        scale_factor=2.0,
        schedule_override=None,
        seed_rng=True,
        seed_rng_offset=1,
        sharpen_gaussian_kernel_size=3,
        sharpen_gaussian_sigma=(0.1, 2.0),
        sharpen_mode="gaussian",
        sharpen_reference=True,
        sharpen_strength=1.0,
        skip_callback=False,
        sigma_offset=0,
        upscale_model=None,
        use_upscale_model=True,
        vae_decode_kwargs=None,
        vae_encode_kwargs=None,
        vae_mode="normal",
        vae=None,
    ):
        sampler = fallback(
            sampler,
            lambda: ksampler("euler"),
            default_is_fun=True,
        )
        self.seed_rng = seed_rng
        self.seed_rng_offset = seed_rng_offset
        self.sigma_offset = sigma_offset
        self.skip_callback = skip_callback
        self.fadeout_factor = fadeout_factor
        self.scale_factor = scale_factor
        self.guidance_factor = guidance_factor
        self.renoise_factor = renoise_factor
        self.iterations = iterations
        self.guidance_steps = guidance_steps
        self.guidance_mode = guidance_mode
        self.guidance_restart = guidance_restart
        self.guidance_restart_s_noise = guidance_restart_s_noise
        self.sampler = sampler
        self.guidance_sampler = fallback(guidance_sampler, sampler)
        self.reference_sampler = fallback(reference_sampler, sampler)
        self.vae = VAEHelper(
            vae_mode,
            latent_format,
            device=device,
            dtype=dtype,
            vae=vae,
            encode_kwargs=fallback(vae_encode_kwargs, {}),
            decode_kwargs=fallback(vae_decode_kwargs, {}),
        )
        self.sharpen = Sharpen(
            mode=sharpen_mode,
            strength=sharpen_strength if sharpen_reference else 0,
            gaussian_kernel_size=sharpen_gaussian_kernel_size,
            gaussian_sigma=sharpen_gaussian_sigma,
        )
        self.upscale = Upscale(
            resample_mode=resample_mode,
            rescale_increment=rescale_increment,
            upscale_model=upscale_model,
        )
        if schedule_override is not None and not isinstance(schedule_override, dict):
            raise TypeError("Bad type for schedule_override: must be null or object")
        self.schedule_override = schedule_override
        self.use_upscale_model = use_upscale_model
        self.dwt_mode = dwt_mode
        self.dwt_level = dwt_level
        self.dwt_wave = dwt_wave
        self.dtcwt_mode = dtcwt_mode
        self.dtcwt_biort = dtcwt_biort
        self.dtcwt_qshift = dtcwt_qshift
        if dtcwt_mode:
            self.dwt = DTCWTForward(
                J=dwt_level,
                mode=dwt_mode,
                biort=dtcwt_biort,
                qshift=dtcwt_qshift,
            ).to(device)
            self.idwt = DTCWTInverse(
                mode=dwt_mode,
                biort=dtcwt_biort,
                qshift=dtcwt_qshift,
            ).to(device)
        else:
            self.dwt = DWTForward(J=dwt_level, wave=dwt_wave, mode=dwt_mode).to(device)
            self.idwt = DWTInverse(wave=dwt_wave, mode=dwt_mode).to(device)
        self.dwt_flip_filters = dwt_flip_filters
        self.reference_wavelet_multiplier = reference_wavelet_multiplier
        self.denoised_wavelet_multiplier = denoised_wavelet_multiplier
        self.blend_mode = blend_mode
        if blend_by_mode not in {"image", "latent", "wavelet"}:
            raise ValueError("Bad blend_by_mode: must be one of image, latent, wavelet")
        self.blend_by_mode = blend_by_mode
        self.blend_function = BLENDING_MODES[blend_mode]
        self.enable_gc = enable_gc
        self.enable_cache_clearing = enable_cache_clearing
        self.iteration_override = {}
        if iteration_override is None or iteration_override == {}:
            return
        if not isinstance(iteration_override, dict):
            raise TypeError("Iteration override must be an object")
        selfdict = self.as_dict()
        overrides = self.iteration_override
        for k, v in iteration_override.items():
            if not isinstance(k, (int, str)) or not isinstance(v, dict):
                raise TypeError(
                    "Bad type for override item: key must be integer or string, value must be an object",
                )
            okwargs = selfdict | {
                ok: ov for ok, ov in v.items() if ok in self._overridable_fields
            }
            overrides[k] = self.__class__(device, dtype, latent_format, **okwargs)

    def as_dict(self) -> dict:
        result = {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and k not in self._dict_exclude_keys
        }
        result["vae_mode"] = self.vae.mode.name.lower()
        result["vae"] = self.vae.vae
        result["vae_encode_kwargs"] = self.vae.encode_kwargs
        result["vae_decode_kwargs"] = self.vae.decode_kwargs
        result["sharpen_reference"] = self.sharpen.strength != 0
        result["sharpen_strength"] = self.sharpen.strength
        result["sharpen_gaussian_kernel_size"] = self.sharpen.gaussian_kernel_size
        result["sharpen_gaussian_sigma"] = self.sharpen.gaussian_sigma
        result["resample_mode"] = self.upscale.resample_mode
        result["rescale_increment"] = self.upscale.rescale_increment
        result["upscale_model"] = self.upscale.upscale_model
        return result

    def get_iteration_config(self, iteration):
        override = self.iteration_override.get(iteration)
        return override.get_iteration_config(iteration) if override else self
