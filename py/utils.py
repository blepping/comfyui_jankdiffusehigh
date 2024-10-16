from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from comfy import model_management
from PIL import Image as PILImage


def pilimgbatch_to_torch(
    imgbatch: Sequence[PILImage, ...] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(imgbatch, torch.Tensor):
        print("pibtt: skip", imgbatch.shape, imgbatch.min(), imgbatch.max())
        return imgbatch
    npi = np.stack(
        tuple(np.array(i).astype(np.float32) / 255.0 for i in imgbatch),
        axis=0,
    )
    return torch.from_numpy(npi)
    return torch.from_numpy(npi.transpose(0, 3, 1, 2))


def torch_to_pilimgbatch(t: torch.Tensor) -> tuple[PILImage, ...]:
    return tuple(
        PILImage.fromarray(
            np.clip((255.0 * i).cpu().numpy(), 0, 255).astype(np.uint8),
        )
        for i in t
    )


def ensure_model(model):
    mp = model.inner_model.model_patcher
    if model_management.LoadedModel(mp) in model_management.current_loaded_models:
        return
    model_management.load_models_gpu((mp,))


def fallback(val, default, *, exclude=None, default_is_fun=False):
    return val if val is not exclude else (default() if default_is_fun else default)


def scale_dim(n, factor=1.0, *, increment=64) -> int:
    return math.ceil((n * factor) / increment) * increment
