from __future__ import annotations

import torch
import yaml
from comfy.samplers import KSAMPLER

from .config import ParamGroup
from .external import init_integrations
from .sampler import diffusehigh_sampler
from .vae import VAEMode

try:
    from comfy_execution import validation as comfy_validation

    if not hasattr(comfy_validation, "validate_node_input"):
        raise NotImplementedError  # noqa: TRY301
    HAVE_COMFY_UNION_TYPE = comfy_validation.validate_node_input("B", "A,B")
except (ImportError, NotImplementedError):
    HAVE_COMFY_UNION_TYPE = False
except Exception as exc:  # noqa: BLE001
    HAVE_COMFY_UNION_TYPE = False
    print(
        f"** jankdiffusehigh: Warning, caught unexpected exception trying to detect ComfyUI union type support. Disabling. Exception: {exc}",
    )

PARAM_TYPES = frozenset((
    "IMAGE",
    "MASK",
    "OCS_NOISE",
    "SAMPLER",
    "SIGMAS",
    "SONAR_CUSTOM_NOISE",
    "UPSCALE_MODEL",
    "VAE",
))

if not HAVE_COMFY_UNION_TYPE:

    class Wildcard(str):  # noqa: FURB189
        __slots__ = ("whitelist",)

        @classmethod
        def __new__(cls, s, *args: list, whitelist=None, **kwargs: dict):
            result = super().__new__(s, *args, **kwargs)
            result.whitelist = whitelist
            return result

        def __ne__(self, other):
            return False if self.whitelist is None else other not in self.whitelist

    WILDCARD_PARAM = Wildcard("*", whitelist=PARAM_TYPES)
else:
    WILDCARD_PARAM = ",".join(PARAM_TYPES)


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
                "input_params_opt": (
                    "DIFFUSEHIGH_PARAMS",
                    {
                        "tooltip": "Optional: You can use a DiffuseHighParam node to specify additional parameters such as VAEs or upscale models.",
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
    def go(
        cls,
        *,
        input_params_opt: ParamGroup | None = None,
        yaml_parameters: str | None = None,
        **kwargs: dict,
    ) -> tuple[KSAMPLER]:
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
        params_opt = (
            ParamGroup(items={})
            if input_params_opt is None
            else input_params_opt.clone()
        )
        pg = ParamGroup(items={})
        for key, pgkey in (
            ("guidance_sampler_opt", ("sampler", "guidance")),
            ("highres_sigmas", ("sigmas", "highres")),
            ("reference_sampler_opt", ("sampler", "reference")),
            ("reference_image_opt", ("image", "reference")),
            ("sampler", ("sampler", "")),
            ("upscale_model_opt", ("upscale_model", "")),
            ("vae_opt", ("vae", "")),
        ):
            val = kwargs.pop(key, None)
            if val is None:
                continue
            pg[pgkey] = val
        clashed = tuple(k for k in params_opt if k in pg)
        if clashed:
            clashedstr = ", ".join(
                f"{k[0]}" if not k[1] else f"{k[0]} ({k[1]})" for k in clashed
            )
            errstr = f"Extra param names conflict with sampler node inputs. Please rename the conflicting extra params: {clashedstr}"
            raise ValueError(errstr)
        pg.items |= params_opt.items
        kwargs["_params"] = pg
        return (
            KSAMPLER(
                diffusehigh_sampler,
                extra_options={
                    "diffusehigh_options": kwargs,
                },
            ),
        )


class DiffuseHighParamNode:
    RETURN_TYPES = ("DIFFUSEHIGH_PARAMS",)
    CATEGORY = "sampling/custom_sampling/JankDiffuseHigh"
    DESCRIPTION = "Jank DiffuseHigh parameter definition node. Used to set parameters like custom noise types that require an input."
    OUTPUT_TYPES = (
        "Can be connected to another DiffuseHigh Param or DiffuseHigh sampler node.",
    )

    FUNCTION = "go"

    PARAM_TYPES = {  # noqa: RUF012
        "vae": lambda _v: True,
        "sampler": lambda v: hasattr(v, "sampler_function"),
        "upscale_model": lambda _v: True,
        "image": lambda v: isinstance(v, torch.Tensor) and v.ndim == 4,
        "sigmas": lambda v: isinstance(v, torch.Tensor) and v.ndim == 1 and len(v) >= 2,
        "custom_noise": lambda v: hasattr(v, "make_noise_sampler"),
        "mask": lambda v: isinstance(v, torch.Tensor) and v.ndim in {2, 3},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (
                    tuple(cls.PARAM_TYPES.keys()),
                    {
                        "tooltip": "Used to set the type of custom parameter.",
                    },
                ),
                "value": (
                    WILDCARD_PARAM,
                    {
                        "tooltip": f"Connect the type of value expected by the key. Allows connecting output from any type of node HOWEVER if it is the wrong type expected by the key you will get an error when you run the workflow.\nThe following input types are supported: {', '.join(PARAM_TYPES)}",
                    },
                ),
            },
            "optional": {
                "name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "You can name a parameter input here (alphanumeric or underscore only) to allow using multiple parameters of the same type.",
                    },
                ),
                "params_opt": (
                    "DIFFUSEHIGH_PARAMS",
                    {
                        "tooltip": "You may optionally connect the output from another jank DiffuseHigh param node here to set multiple parameters.",
                    },
                ),
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters for an input via YAML. This input can be converted into a multiline text widget. Note: When specifying paramaters this way, there is very little error checking.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    @classmethod
    def get_renamed_key(cls, key, *, name=None):
        if not isinstance(name, str) or not name:
            return (key, "")
        if not isinstance(name, str):
            raise TypeError("Param name key must be a string if set")
        name = name.strip()
        if not name or not all(c == "_" or c.isalnum() for c in name):
            raise ValueError(
                "Param name keys must consist of one or more alphanumeric or underscore characters",
            )
        return (key, name)

    def go(self, *, input_type, value, name="", params_opt=None, yaml_parameters=""):
        if not self.PARAM_TYPES[input_type](value):
            errstr = f"DiffuseHighParam: Bad value type for input_type {input_type}"
            raise TypeError(errstr)
        if yaml_parameters:
            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is not None and not isinstance(extra_params, dict):
                raise ValueError("Parameters must be a JSON or YAML object")
        else:
            extra_params = None
        key = self.get_renamed_key(input_type, name=name)
        params = ParamGroup(items={}) if params_opt is None else params_opt.clone()
        params[key] = value
        if extra_params is not None:
            params[(*key, "params")] = extra_params
        return (params,)
