from __future__ import annotations

from enum import Enum, auto

import folder_paths
import torch
from comfy.taesd.taesd import TAESD

from .external import EXTERNAL

tiled_diffusion = EXTERNAL.get("tiled_diffusion")


class VAEMode(Enum):
    TAESD = auto()
    NORMAL = auto()
    TILED = auto()
    TILED_DIFFUSION = auto()


class VAEHelper:
    def __init__(
        self,
        mode: VAEMode | str,
        latent_format,
        *,
        device=None,
        dtype=None,
        vae=None,
        vae_encode_kwargs=None,
        vae_decode_kwargs=None,
    ):
        if isinstance(mode, str):
            mode = VAEMode.__members__[mode.upper()]
        if mode == VAEMode.TILED_DIFFUSION:
            if tiled_diffusion is None:
                raise ValueError(
                    "Cannot use tiled_diffusion VAE mode without ComfyUI-TiledDiffusion!",
                )
            self.td_encode_default_kwargs, self.td_decode_default_kwargs = (
                {
                    k: v[1]["default"]
                    for k, v in td_node.INPUT_TYPES()
                    .get(
                        "required",
                        {},
                    )
                    .items()
                    if k not in {"pixels", "samples", "vae"}
                    and len(v) == 2
                    and isinstance(v[1], dict)
                    and "default" in v[1]
                }
                for td_node in (
                    tiled_diffusion.tiled_vae.VAEEncodeTiled_TiledDiffusion,
                    tiled_diffusion.tiled_vae.VAEDecodeTiled_TiledDiffusion,
                )
            )
        if mode != VAEMode.TAESD and vae is None:
            raise ValueError("Must pass a VAE when using non-TAESD VAE modes!")
        self.mode = mode
        self.latent_format = latent_format
        self.device = device
        self.dtype = dtype
        self.vae = vae
        self.vae_encode_kwargs = {} if vae_encode_kwargs is None else vae_encode_kwargs
        self.vae_decode_kwargs = {} if vae_decode_kwargs is None else vae_decode_kwargs
        vae_handlers = {
            VAEMode.TAESD: (self.encode_taesd, self.decode_taesd),
            VAEMode.NORMAL: (self.encode_vae, self.decode_vae),
            VAEMode.TILED: (
                self.encode_vae_tiled,
                self.decode_vae_tiled,
            ),
            VAEMode.TILED_DIFFUSION: (
                self.encode_vae_tiled_diffusion,
                self.decode_vae_tiled_diffusion,
            ),
        }
        self.encode_fun, self.decode_fun = vae_handlers[mode]

    def encode(self, imgbatch, *, fix_dims=False):
        if fix_dims:
            imgbatch = imgbatch.moveaxis(1, -1)
        # print("ENCODING", imgbatch.min(), imgbatch.max())
        result = self.encode_fun(imgbatch[..., :3])
        if self.mode != VAEMode.TAESD:
            # print("ENCODED(raw):", result.min(), result.max())
            result = self.latent_format.process_in(result)
        # print("ENCODED", result.shape, result.min(), result.max())
        return result

    def decode(self, latent, *, skip_process_out=False):
        if self.mode != VAEMode.TAESD and not skip_process_out:
            latent = self.latent_format.process_out(latent)
        # print("DECODING", latent.min(), latent.max())
        return self.decode_fun(latent)
        # print("DECODED", result.shape, result.min(), result.max())

    def encode_taesd(self, imgbatch):
        dummy = torch.zeros((), device=self.device, dtype=self.dtype)
        return OCSTAESD.encode(self.latent_format, imgbatch, dummy)

    def decode_taesd(self, latent):
        return OCSTAESD.decode(self.latent_format, latent)

    def encode_vae(self, imgbatch):
        # print("VAE ENC", imgbatch.shape)
        return self.vae.encode(imgbatch, **self.vae_encode_kwargs)

    def decode_vae(self, latent):
        return self.vae.decode(latent, **self.vae_decode_kwargs)

    def encode_vae_tiled(self, imgbatch):
        return self.vae.encode_tiled(imgbatch, **self.vae_encode_kwargs)

    def decode_vae_tiled(self, latent):
        return self.vae.decode_tiled(latent, **self.vae_decode_kwargs)

    def encode_vae_tiled_diffusion(self, imgbatch):
        kwargs = self.td_encode_default_kwargs | self.vae_encode_kwargs
        return tiled_diffusion.tiled_vae.VAEEncodeTiled_TiledDiffusion().process(
            pixels=imgbatch,
            vae=self.vae,
            **kwargs,
        )[0]["samples"]

    def decode_vae_tiled_diffusion(self, latent):
        kwargs = self.td_decode_default_kwargs | self.vae_decode_kwargs
        return tiled_diffusion.tiled_vae.VAEDecodeTiled_TiledDiffusion().process(
            samples={"samples": latent},
            vae=self.vae,
            **kwargs,
        )[0]


class OCSTAESD:
    @classmethod
    def get_encoder_name(cls, latent_format):
        result = latent_format.taesd_decoder_name
        if not result.endswith("_decoder"):
            msg = f"Could not determine TAESD encoder name from {result!r}"
            raise RuntimeError(
                msg,
            )
        return f"{result[:-7]}encoder"

    @classmethod
    def get_taesd_path(cls, name):
        taesd_path = next(
            (
                fn
                for fn in folder_paths.get_filename_list("vae_approx")
                if fn.startswith(name)
            ),
            "",
        )
        if not taesd_path:
            msg = f"Could not get TAESD path for {name!r}"
            raise RuntimeError(msg)
        return folder_paths.get_full_path("vae_approx", taesd_path)

    @classmethod
    def decode(cls, latent_format, latent):
        filename = cls.get_taesd_path(latent_format.taesd_decoder_name)
        model = TAESD(
            decoder_path=filename,
            latent_channels=latent_format.latent_channels,
        ).to(latent.device)
        return (
            model.taesd_decoder(
                (latent - model.vae_shift).mul_(model.vae_scale),
            )
            .clamp_(0, 1)
            .movedim(1, -1)
        )

    @classmethod
    def encode(cls, latent_format, imgbatch, latent) -> torch.Tensor:
        filename = cls.get_taesd_path(cls.get_encoder_name(latent_format))
        model = TAESD(
            encoder_path=filename,
            latent_channels=latent_format.latent_channels,
        ).to(device=latent.device)
        return (
            model.taesd_encoder(imgbatch.to(latent.device).moveaxis(-1, 1))
            .div_(model.vae_scale)
            .add_(model.vae_shift)
        )
