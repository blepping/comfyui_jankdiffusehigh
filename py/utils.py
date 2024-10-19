from __future__ import annotations

import math

from comfy import model_management


def ensure_model(model):
    mp = model.inner_model.model_patcher
    if model_management.LoadedModel(mp) in model_management.current_loaded_models:
        return
    model_management.load_models_gpu((mp,))


def fallback(val, default, *, exclude=None, default_is_fun=False):
    return val if val is not exclude else (default() if default_is_fun else default)


def scale_dim(n, factor=1.0, *, increment=64) -> int:
    return math.ceil((n * factor) / increment) * increment
