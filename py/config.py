from __future__ import annotations

from typing import Any

from comfy.samplers import ksampler
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse

from .tensor_image_ops import Sharpen
from .upscale import Upscale
from .utils import fallback
from .vae import VAEHelper


class Config:
    _overridable_fields = {  # noqa: RUF012
        "blend_by_mode",
        "blend_mode",
        "chunked_sampling",
        "custom_noise_name",
        "custom_noise_params",
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
        "force_upscale_model",
        "guidance_mask_blend_mode",
        "guidance_mask_name",
        "guidance_factor",
        "guidance_mode",
        "guidance_restart_s_noise",
        "guidance_restart",
        "guidance_sampler_name",
        "guidance_steps",
        "highres_sigmas_name",
        "mask_blend_mode",
        "mask_name",
        "reference_image_name",
        "reference_sampler_name",
        "reference_wavelet_multiplier",
        "renoise_factor",
        "resample_mode",
        "rescale_increment",
        "restart_custom_noise_name",
        "sampler_name",
        "scale_factor",
        "schedule_override",
        "sharpen_gaussian_kernel_size",
        "sharpen_gaussian_sigma",
        "sharpen_mode",
        "sharpen_reference",
        "sharpen_strength",
        "sigma_dishonesty_factor_guidance",
        "sigma_dishonesty_factor",
        "skip",
        "skip_callback",
        "upscale_model_name",
        "use_upscale_model",
        "vae_decode_kwargs",
        "vae_encode_kwargs",
        "vae_mode",
        "vae_name",
    }

    _dict_exclude_keys = {  # noqa: RUF012
        "as_dict",
        "dwt",
        "get_iteration_config",
        "guidance_sampler",
        "highres_sigmas",
        "idwt",
        "iteration_override",
        "reference_sampler",
        "sampler",
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
        _params,
        blend_by_mode="image",
        blend_mode="lerp",
        chunked_sampling=True,
        custom_noise_name="",
        custom_noise_params=None,
        denoised_wavelet_multiplier=1.0,
        dtcwt_biort="near_sym_a",
        dtcwt_mode=False,
        dtcwt_qshift="qshift_a",
        dwt_flip_filters=False,
        dwt_level=1,
        dwt_mode="symmetric",
        dwt_wave="db4",
        enable_cache_clearing=True,
        enable_gc=True,
        fadeout_factor=0.0,
        force_upscale_model=False,
        guidance_factor=1.0,
        guidance_mask_blend_mode="lerp",
        guidance_mask_name="guidance",
        guidance_mode="image",
        guidance_restart_s_noise=1.0,
        guidance_restart=0,
        guidance_sampler_name="guidance",
        guidance_steps=5,
        highres_sigmas_name="highres",
        iteration_override=None,
        iterations=1,
        mask_blend_mode="lerp",
        mask_name="",
        reference_image_name="reference",
        reference_sampler_name="reference",
        reference_wavelet_multiplier=1.0,
        renoise_factor=1.0,
        resample_mode="bicubic",
        rescale_increment=64,
        restart_custom_noise_name="restart",
        sampler_name="",
        scale_factor=2.0,
        schedule_override=None,
        seed_rng_offset=1,
        seed_rng=True,
        sharpen_gaussian_kernel_size=3,
        sharpen_gaussian_sigma=(0.1, 2.0),
        sharpen_mode="gaussian",
        sharpen_reference=True,
        sharpen_strength=1.0,
        sigma_dishonesty_factor_guidance: float | None = None,
        sigma_dishonesty_factor=0.0,
        skip_callback=False,
        skip=False,
        upscale_model_name="",
        use_upscale_model=True,
        vae_decode_kwargs=None,
        vae_encode_kwargs=None,
        vae_mode="normal",
        vae_name="",
        ensure_model_mode: str | None = True,
    ):
        self.skip = skip
        if ensure_model_mode not in {
            None,
            "disable",
            "normal",
            "normal_unload",
            "lowvram",
            "novram",
        }:
            raise ValueError(
                "Bad ensure_model_mode - must null or one of disable, normal, normal_unload, lowvram, novram",
            )
        self.ensure_model_mode = ensure_model_mode

        self.vae_name = vae_name
        self.upscale_model_name = upscale_model_name
        self.highres_sigmas_name = highres_sigmas_name
        self.reference_image_name = reference_image_name
        self.reference_sampler_name = reference_sampler_name
        self.guidance_sampler_name = guidance_sampler_name
        self.sampler_name = sampler_name
        self.custom_noise_name = custom_noise_name
        self.custom_noise_params = (
            {} if not isinstance(custom_noise_params, dict) else custom_noise_params
        )
        self.restart_custom_noise_name = restart_custom_noise_name
        self.mask_name = mask_name
        self.guidance_mask_name = guidance_mask_name

        sampler = _params.get_item("sampler", name=sampler_name)
        guidance_sampler = _params.get_item("sampler", name=guidance_sampler_name)
        reference_sampler = _params.get_item("sampler", name=reference_sampler_name)
        upscale_model = _params.get_item("upscale_model", name=upscale_model_name)
        vae = _params.get_item("vae", name=vae_name)
        highres_sigmas = _params.get_item("sigmas", name=highres_sigmas_name)
        self.highres_sigmas = (
            None if highres_sigmas is None else highres_sigmas.detach().clone()
        )

        self.mask_blend_mode = mask_blend_mode
        self.guidance_mask_blend_mode = guidance_mask_blend_mode

        sampler = fallback(
            sampler,
            lambda: ksampler("euler"),
            default_is_fun=True,
        )
        self.seed_rng = seed_rng
        self.seed_rng_offset = seed_rng_offset
        self.sigma_dishonesty_factor = sigma_dishonesty_factor
        self.sigma_dishonesty_factor_guidance = fallback(
            sigma_dishonesty_factor_guidance,
            sigma_dishonesty_factor,
        )
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
            force_upscale_model=force_upscale_model,
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
        self.enable_gc = enable_gc
        self.enable_cache_clearing = enable_cache_clearing
        self.chunked_sampling = chunked_sampling
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
            overrides[k] = self.__class__(
                device,
                dtype,
                latent_format,
                _params=_params,
                **okwargs,
            )

    def as_dict(self) -> dict:
        result = {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and k not in self._dict_exclude_keys
        }
        result["vae_mode"] = self.vae.mode.name.lower()
        result["vae_encode_kwargs"] = self.vae.encode_kwargs
        result["vae_decode_kwargs"] = self.vae.decode_kwargs
        result["sharpen_reference"] = self.sharpen.strength != 0
        result["sharpen_strength"] = self.sharpen.strength
        result["sharpen_gaussian_kernel_size"] = self.sharpen.gaussian_kernel_size
        result["sharpen_gaussian_sigma"] = self.sharpen.gaussian_sigma
        result["resample_mode"] = self.upscale.resample_mode
        result["rescale_increment"] = self.upscale.rescale_increment
        result["force_upscale_model"] = self.upscale.force_upscale_model
        return result

    def get_iteration_config(self, iteration):
        override = self.iteration_override.get(iteration)
        return override.get_iteration_config(iteration) if override else self


class ParamGroup:
    def __init__(self, items=None):
        self.items = {} if items is None else items

    def clone(self):
        return self.__class__(items=self.items.copy())

    def append(self, item):
        self.items.append(item)
        return item

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()

    def get_item(
        self,
        type_name: str,
        *,
        name: str | None = "",
        param_mode: bool = False,
        default: Any | None = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        name = name if name is not None else ""
        key = (type_name, name) if not param_mode else (type_name, name, "params")
        return self.items.get(key, default)
