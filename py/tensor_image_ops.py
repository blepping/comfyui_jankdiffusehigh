from __future__ import annotations

from enum import Enum, auto
from typing import Sequence

import numpy as np
import torch
import torchvision
from PIL import Image as PILImage

from .external import EXTERNAL

F = torch.nn.functional

BLENDING_MODES = {
    "lerp": torch.lerp,
}


def init_integrations():
    global BLENDING_MODES  # noqa: PLW0603
    ext_bleh = EXTERNAL.get("bleh")
    if ext_bleh is not None:
        BLENDING_MODES.clear()
        BLENDING_MODES |= ext_bleh.latent_utils.BLENDING_MODES


class SharpenMode(Enum):
    GAUSSIAN = auto()
    CONTRAST_ADAPTIVE = auto()


def pilimgbatch_to_torch(
    imgbatch: Sequence[PILImage, ...] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(imgbatch, torch.Tensor):
        return imgbatch
    npi = np.stack(
        tuple(np.array(i).astype(np.float32) / 255.0 for i in imgbatch),
        axis=0,
    )
    return torch.from_numpy(npi)


def torch_to_pilimgbatch(t: torch.Tensor) -> tuple[PILImage, ...]:
    return tuple(
        PILImage.fromarray(
            np.clip((255.0 * i).cpu().numpy(), 0, 255).astype(np.uint8),
        )
        for i in t
    )


def scale_wavelets(waves, factor=1.0):
    if factor == 1:
        return waves
    return (waves[0] * factor, tuple(t * factor for t in waves[1]))


def blend_wavelets(a, b, factor, blend_function):
    if not isinstance(factor, torch.Tensor):
        factor = a[0].new_full((1,), factor)
    return (
        blend_function(a[0], b[0], factor),
        tuple(blend_function(ta, tb, factor) for ta, tb in zip(a[1], b[1])),
    )


class Sharpen:
    def __init__(
        self,
        mode="gaussian",
        strength=1.0,
        gaussian_kernel_size=3,
        gaussian_sigma=(0.1, 2.0),
    ):
        self.mode = getattr(SharpenMode, mode.upper(), None)
        if self.mode is None:
            raise ValueError("Bad sharpen mode")
        self.strength = strength
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma

    def __call__(self, t, *, fix_dims=False):
        if self.strength == 0:
            return t
        if fix_dims:
            t = t.movedim(-1, 1)
        if self.mode == SharpenMode.GAUSSIAN:
            result = gaussian_blur_image_sharpening(
                t,
                kernel_size=self.gaussian_kernel_size,
                sigma=self.gaussian_sigma,
                alpha=self.strength,
            )
        elif self.mode == SharpenMode.CONTRAST_ADAPTIVE:
            result = contrast_adaptive_sharpening(t, amount=self.strength)
        if fix_dims:
            result = result.movedim(1, -1)
        return result


def gaussian_blur_image_sharpening(image, kernel_size=3, sigma=(0.1, 2.0), alpha=1):
    gaussian_blur = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size,
        sigma=sigma,
    )
    image_blurred = gaussian_blur(image)
    return (alpha + 1) * image - alpha * image_blurred


# Improvements added by https://github.com/Clybius
# The following is modified to work with latent images of ~0 mean from https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/tree/main.
def contrast_adaptive_sharpening(x, amount=0.8, *, epsilon=1e-06):  # noqa: D417, PLR0914
    """Performs contrast adaptive sharpening on the batch of images x.

    The algorithm is directly implemented from FidelityFX's source code,
    that can be found here
    https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h.

    Parameters
    ----------
    x : Tensor
        Image or stack of images, of shape [batch, channels, ny, nx].
        Batch and channel dimensions can be ommited.
    amount : int [0, 1]
        Amount of sharpening to do, 0 being minimum and 1 maximum

    Returns
    -------
    Tensor
        Processed stack of images.

    """  # noqa: D401

    def on_abs_stacked(tensor_list, f, *args: list, **kwargs: dict):
        return f(torch.abs(torch.stack(tensor_list)), *args, **kwargs)[0]

    x_padded = F.pad(x, pad=(1, 1, 1, 1))
    x_padded = torch.complex(x_padded, torch.zeros_like(x_padded))
    # each side gets padded with 1 pixel
    # padding = same by default

    # Extracting the 3x3 neighborhood around each pixel
    # a b c
    # d e f
    # g h i

    a = x_padded[..., :-2, :-2]
    b = x_padded[..., :-2, 1:-1]
    c = x_padded[..., :-2, 2:]
    d = x_padded[..., 1:-1, :-2]
    e = x_padded[..., 1:-1, 1:-1]
    f = x_padded[..., 1:-1, 2:]
    g = x_padded[..., 2:, :-2]
    h = x_padded[..., 2:, 1:-1]
    i = x_padded[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = on_abs_stacked(cross, torch.min, axis=0)
    mx = on_abs_stacked(cross, torch.max, axis=0)

    diag = (a, c, g, i)
    mn2 = on_abs_stacked(diag, torch.min, axis=0)
    mx2 = on_abs_stacked(diag, torch.max, axis=0)

    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx + epsilon)  # 1/mx

    amp = inv_mx * mn

    # scaling
    amp = torch.sqrt(amp)

    w = -amp * (amount * (1 / 5 - 1 / 8) + 1 / 8)
    # w scales from 0 when amp=0 to K for amp=1
    # K scales from -1/5 when amount=1 to -1/8 for amount=0

    # The local conv filter is
    # 0 w 0
    # w 1 w
    # 0 w 0
    div = torch.reciprocal(1 + 4 * w)
    output = ((b + d + f + h) * w + e) * div

    return output.real.clamp(x.min(), x.max())
