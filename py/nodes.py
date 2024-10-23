from __future__ import annotations

import yaml
from comfy.samplers import KSAMPLER

from .external import init_integrations
from .sampler import diffusehigh_sampler
from .vae import VAEMode


class DiffuseHighSamplerNode:
    DESCRIPTION = "Jank DiffuseHigh sampler node, used for generating directly to resolutions higher than what the model was trained for. Can be connected to a SamplerCustom or other sampler node that supports a SAMPLER input."
    OUTPUT_TOOLTIPS = (
        "SAMPLER that can be connected to a SamplerCustom or other sampler node that supports a SAMPLER input.",
    )
    CATEGORY = "sampling/custom_sampling/JankDiffuseHigh"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "guidance_steps": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "tooltip": "Number of guidance steps after an upscale.",
                    },
                ),
                "guidance_mode": (
                    (
                        "image",
                        "latent",
                    ),
                    {
                        "default": "image",
                        "tooltip": "The original implementation uses image guidance. This requires a VAE encode/decode per guidance step. Alternatively, you can try using guidance via the latent instead which is much faster.",
                    },
                ),
                "guidance_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Mix factor used on guidance steps. 1.0 means use 100% DiffuseHigh guidance for those steps (like the original implementation).",
                    },
                ),
                "fadeout_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "Can be enabled to fade out guidance_factor. For example, if guidance_factor is 1 and guidance_steps is 4 then fadeout_factor would use these guidance_factors for the guidance steps: 1.00, 0.75, 0.50, 0.25",
                    },
                ),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "tooltip": "Upscale factor per iteration.",
                    },
                ),
                "renoise_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Strength of noise added at the start of each iteration. The default of 1.0 (100%) is the normal amount, but you can increase this slightly to add more detail.",
                    },
                ),
                "iterations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "tooltip": "Number of upscale iterations to run. Be careful, this can add up fast - if you start at 512x512 with a 2.0 scale factor then 3 iterations will get you to 4096x4096.",
                    },
                ),
                "vae_mode": (
                    tuple(vm.name.lower() for vm in VAEMode),
                    {
                        "default": "normal",
                        "tooltip": "Mode used for encoding/decoding images. TAESD is fast/low VRAM but may reduce quality (you will also need the TAESD encoders installed). Normal will just use the normal VAE node, tiled with use the tiled VAE node. Alternatively, if you have ComfyUI-TiledDiffusion installed you can use tiled_diffusion here.",
                    },
                ),
            },
            "optional": {
                "highres_sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "Sigmas used for steps after upscaling. Generally should be around 0.3-0.5 denoise. NOTE: I do not recommend plugging in raw 1.0 denoise sigmas here.",
                    },
                ),
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": "Default sampler used for steps. If not specified the sampler will default to non-ancestral Euler.",
                    },
                ),
                "reference_image_opt": (
                    "IMAGE",
                    {
                        "tooltip": "Optional: Image used for the initial pass. If not connected, a low-res initial reference will be generated using the schedule from the normal sigmas.",
                    },
                ),
                "guidance_sampler_opt": (
                    "SAMPLER",
                    {
                        "tooltip": "Optional: Sampler used for guidance steps. If not specified, will fallback to the base sampler. Note: The sampler is called on individual steps, samplers that keep history will not work well here.",
                    },
                ),
                "reference_sampler_opt": (
                    "SAMPLER",
                    {
                        "tooltip": "Optional: Sampler used to generate the initial low-resolution reference. Only used if reference_image_opt is not connected.",
                    },
                ),
                "vae_opt": (
                    "VAE",
                    {
                        "tooltip": "Optional when vae_mode is set to `taesd`, otherwise this is the VAE that will be used for encoding/decoding images.",
                    },
                ),
                "upscale_model_opt": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": "Optional: Model used for upscaling. When not attached, simple image scaling will be used. Regardless, the image will be scaled to match the size expected based on scale_factor. For example, if you use scale_factor 2 and a 4x upscale model, the image will get scaled down after the upscale model runs.",
                    },
                ),
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. You can also override any of the normal parameters by key. This input can be converted into a multiline text widget. Note: When specifying paramaters this way, there is very little error checking.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    @classmethod
    def go(cls, yaml_parameters: None | str = None, **kwargs: dict) -> tuple[KSAMPLER]:
        init_integrations()
        if yaml_parameters:
            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "DiffuseHighSampler: yaml_parameters must either be null or an object",
                )
            else:
                kwargs |= extra_params
        return (
            KSAMPLER(
                diffusehigh_sampler,
                extra_options={
                    "diffusehigh_options": kwargs,
                },
            ),
        )
