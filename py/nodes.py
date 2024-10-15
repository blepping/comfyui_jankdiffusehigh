from __future__ import annotations

import yaml
from comfy.samplers import KSAMPLER

from .sampler import diffusehigh_sampler


class DiffuseHighSamplerNode:
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "highres_sigmas": ("SIGMAS",),
                "guidance_steps": ("INT", {"default": 5, "min": 0}),
                "guidance_mode": (
                    (
                        "image",
                        "latent",
                    ),
                ),
                "guidance_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                    },
                ),
                "fadeout_factor": ("FLOAT", {"default": 0.0}),
                "scale_factor": ("FLOAT", {"default": 2.0}),
                "renoise_factor": ("FLOAT", {"default": 1.0}),
                "iterations": ("INT", {"default": 1, "min": 0}),
                "sampler": ("SAMPLER",),
                "vae_mode": (
                    (
                        "taesd",
                        "normal",
                        "tiled",
                        "tiled_diffusion",
                    ),
                ),
            },
            "optional": {
                "reference_image_opt": ("IMAGE",),
                "guidance_sampler_opt": ("SAMPLER",),
                "reference_sampler_opt": ("SAMPLER",),
                "vae_opt": ("VAE",),
                "upscale_model_opt": ("UPSCALE_MODEL",),
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
