from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import folder_paths
from comfy.model_management import device_supports_non_blocking
from comfy.taesd import taesd
from tqdm import tqdm

from .external import MODULES as EXT
from .utils import fallback

if TYPE_CHECKING:
    import torch

tiled_diffusion = None


def init_integrations(integrations):
    global tiled_diffusion  # noqa: PLW0603
    tiled_diffusion = integrations.tiled_diffusion


EXT.register_init_handler(init_integrations)


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
        encode_kwargs=None,
        decode_kwargs=None,
    ):
        if isinstance(mode, str):
            mode = VAEMode.__members__[mode.upper()]
        if mode == VAEMode.TAESD:
            vae = TAESD(latent_format, device=device)
        elif vae is None:
            raise ValueError("Must pass a VAE when using non-TAESD VAE modes!")
        elif mode == VAEMode.TILED_DIFFUSION:
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
        self.mode = mode
        self.latent_format = latent_format
        self.device = device
        self.dtype = dtype
        self.vae = vae
        self.encode_kwargs = fallback(encode_kwargs, {})
        self.decode_kwargs = fallback(decode_kwargs, {})
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

    def encode(self, imgbatch, *, fix_dims=False, disable_pbar=None):
        if fix_dims:
            imgbatch = imgbatch.moveaxis(1, -1)
        with tqdm(disable=disable_pbar, total=1, desc="VAE encode") as pbar:
            result = self.encode_fun(imgbatch[..., :3])
            pbar.update()
        if self.mode != VAEMode.TAESD:
            result = self.latent_format.process_in(result)
        return result

    def decode(self, latent, *, skip_process_out=False, disable_pbar=None):
        if self.mode != VAEMode.TAESD and not skip_process_out:
            latent = self.latent_format.process_out(latent)
        with tqdm(disable=disable_pbar, total=1, desc="VAE decode") as pbar:
            result = self.decode_fun(latent)
            pbar.update()
        return result

    def encode_taesd(self, imgbatch):
        return self.vae.encode(imgbatch, device=self.device)

    def decode_taesd(self, latent):
        return self.vae.decode(latent)

    def encode_vae(self, imgbatch):
        return self.vae.encode(imgbatch, **self.encode_kwargs)

    def decode_vae(self, latent):
        return self.vae.decode(latent, **self.decode_kwargs)

    def encode_vae_tiled(self, imgbatch):
        return self.vae.encode_tiled(imgbatch, **self.encode_kwargs)

    def decode_vae_tiled(self, latent):
        return self.vae.decode_tiled(latent, **self.decode_kwargs)

    def encode_vae_tiled_diffusion(self, imgbatch):
        kwargs = self.td_encode_default_kwargs | self.encode_kwargs
        return tiled_diffusion.tiled_vae.VAEEncodeTiled_TiledDiffusion().process(
            pixels=imgbatch,
            vae=self.vae,
            **kwargs,
        )[0]["samples"]

    def decode_vae_tiled_diffusion(self, latent):
        kwargs = self.td_decode_default_kwargs | self.decode_kwargs
        return tiled_diffusion.tiled_vae.VAEDecodeTiled_TiledDiffusion().process(
            samples={"samples": latent},
            vae=self.vae,
            **kwargs,
        )[0]


class TAESD:
    def __init__(self, latent_format, *, device=None):
        self.encoder = None
        self.decoder = None
        self.latent_format = latent_format
        self.device = device

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

    def ensure_encoder(self, device=None):
        if self.encoder is None:
            filename = self.get_taesd_path(self.get_encoder_name(self.latent_format))
            self.encoder = taesd.TAESD(
                encoder_path=filename,
                latent_channels=self.latent_format.latent_channels,
            )
        device = fallback(device, self.device)
        if device is not None:
            self.encoder.to(device, non_blocking=device_supports_non_blocking(device))
        return self.encoder

    def ensure_decoder(self, device=None):
        if self.decoder is None:
            filename = self.get_taesd_path(self.latent_format.taesd_decoder_name)
            self.decoder = taesd.TAESD(
                decoder_path=filename,
                latent_channels=self.latent_format.latent_channels,
            )
        device = fallback(device, self.device)
        if device is not None:
            self.decoder.to(device, non_blocking=device_supports_non_blocking(device))
        return self.decoder

    def decode(self, latent):
        model = self.ensure_decoder(device=latent.device)
        return (
            model.taesd_decoder(
                (latent - model.vae_shift).mul_(model.vae_scale),
            )
            .clamp_(0, 1)
            .movedim(1, -1)
        )

    def encode(self, imgbatch, device=None) -> torch.Tensor:
        model = self.ensure_encoder(device=device)
        if device is not None:
            imgbatch = (
                imgbatch.detach()
                .clone()
                .to(device, non_blocking=device_supports_non_blocking(device))
            )
        return (
            model.taesd_encoder(imgbatch.moveaxis(-1, 1))
            .div_(model.vae_scale)
            .add_(model.vae_shift)
        )
