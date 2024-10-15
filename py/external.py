import contextlib
import importlib

EXTERNAL = {}

with contextlib.suppress(ImportError):
    EXTERNAL["tiled_diffusion"] = importlib.import_module(
        "custom_nodes.ComfyUI-TiledDiffusion",
    )
