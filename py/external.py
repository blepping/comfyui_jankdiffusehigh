import contextlib
import importlib

EXTERNAL = {}

with contextlib.suppress(ImportError):
    EXTERNAL["tiled_diffusion"] = importlib.import_module(
        "custom_nodes.ComfyUI-TiledDiffusion",
    )

with contextlib.suppress(ImportError, NotImplementedError):
    bleh = importlib.import_module("custom_nodes.ComfyUI-bleh")
    bleh_version = getattr(bleh, "BLEH_VERSION", -1)
    if bleh_version < 1:
        raise NotImplementedError
    EXTERNAL["bleh"] = bleh.py
