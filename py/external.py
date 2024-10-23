import contextlib
import importlib.util
import sys

EXTERNAL = {}
INITIALIZED = False


def get_custom_node(name):
    module_key = f"custom_nodes.{name}"
    try:
        spec = importlib.util.find_spec(module_key)
        if spec is None:
            raise ModuleNotFoundError(module_key)
        module = next(
            v
            for v in sys.modules.copy().values()
            if hasattr(v, "__spec__")
            and v.__spec__ is not None
            and v.__spec__.origin == spec.origin
        )
    except StopIteration:
        raise ModuleNotFoundError(module_key) from None
    return module


def init_integrations() -> None:
    global INITIALIZED  # noqa: PLW0603
    if INITIALIZED:
        return
    INITIALIZED = True

    with contextlib.suppress(ModuleNotFoundError):
        EXTERNAL["tiled_diffusion"] = get_custom_node("ComfyUI-TiledDiffusion")

    with contextlib.suppress(ModuleNotFoundError, NotImplementedError):
        bleh = get_custom_node("ComfyUI-bleh")
        bleh_version = getattr(bleh, "BLEH_VERSION", -1)
        if bleh_version < 1:
            raise NotImplementedError
        EXTERNAL["bleh"] = bleh.py

    from . import tensor_image_ops, vae  # noqa: PLC0415

    tensor_image_ops.init_integrations()
    vae.init_integrations()
