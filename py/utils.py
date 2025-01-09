from __future__ import annotations

import math

from comfy import model_management


def ensure_model(model, *, mode: str | None = None):
    if not mode or mode == "disable":
        # Ensure model mode disabled.
        return
    mp = model.inner_model.model_patcher
    lm = model_management.LoadedModel(mp)
    found_lm = None
    for list_lm in model_management.current_loaded_models:
        if lm == list_lm:
            found_lm = list_lm
            break
    if getattr(found_lm, "currently_used", False):
        # Model already exists and appears to be loaded.
        return
    if found_lm is not None:
        lm = found_lm
        mp = lm.model
    if mode.startswith("normal"):
        if mode == "normal_unload":
            model_management.unload_all_models()
        model_management.load_models_gpu((mp,))
        return
    model_management.unload_all_models()
    if mode == "lowvram":
        # Logic from comfy.model_management.load_models_gpu
        min_inference_memory = model_management.minimum_inference_memory()
        minimum_memory_required = max(
            min_inference_memory,
            model_management.extra_reserved_memory(),
        )
        loaded_memory = lm.model_loaded_memory()
        current_free_mem = model_management.get_free_memory(lm.device) + loaded_memory

        lowvram_model_memory = max(
            64 * 1024 * 1024,
            (current_free_mem - minimum_memory_required),
            min(
                current_free_mem * model_management.MIN_WEIGHT_MEMORY_RATIO,
                current_free_mem - model_management.minimum_inference_memory(),
            ),
        )
        lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)
    elif mode == "novram":
        lowvram_model_memory = 0.1
    else:
        raise ValueError("Bad ensure_model_mode")
    lm.model_load(lowvram_model_memory)
    model_management.current_loaded_models.insert(0, lm)


def fallback(val, default, *, exclude=None, default_is_fun=False):
    return val if val is not exclude else (default() if default_is_fun else default)


def scale_dim(n, factor=1.0, *, increment=64) -> int:
    return math.ceil((n * factor) / increment) * increment


def sigma_to_float(sigma):
    return sigma.detach().cpu().max().item()
