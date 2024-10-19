from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from PIL import Image

from .tensor_image_ops import (
    pilimgbatch_to_torch,
    torch_to_pilimgbatch,
)
from .utils import scale_dim


class Upscale:
    def __init__(
        self,
        *,
        resample_mode="bicubic",
        rescale_increment=64,
        upscale_model=None,
    ):
        self.resample_mode = resample_mode
        self.rescale_increment = scale_dim(max(8, rescale_increment), increment=8)
        self.upscale_model = upscale_model

    def __call__(self, imgbatch, scale_factor, *, pbar=None, use_upscale_model=True):
        if scale_factor == 1.0:
            return imgbatch
        _batch, height, width, _channels = imgbatch.shape
        target_height = scale_dim(
            height,
            scale_factor,
            increment=self.rescale_increment,
        )
        target_width = scale_dim(
            width,
            scale_factor,
            increment=self.rescale_increment,
        )
        # tqdm.write(f">> UPSCALE: {width}x{height} -> {target_width}x{target_height}")
        if (target_height, target_width) == (height, width):
            return imgbatch
        if use_upscale_model and self.upscale_model is not None:
            if pbar is not None:
                pbar.set_description(
                    f"upscale with model: {width}x{height} -> {target_width}x{target_height}",
                )
            # tqdm.write("** Upscaling with model")
            imgbatch = ImageUpscaleWithModel().upscale(self.upscale_model, imgbatch)[0]
            if imgbatch.shape[1:3] == (target_height, target_width):
                return imgbatch
        # tqdm.write(
        #     f"** PIL upscale {imgbatch.shape[2]}x{imgbatch.shape[1]} -> {target_width}x{target_height}",
        # )
        if pbar is not None and self.upscale_model is None:
            pbar.set_description(
                f"upscale (simple): {imgbatch.shape[2]}x{imgbatch.shape[1]} -> {target_width}x{target_height}",
            )
        return pilimgbatch_to_torch(
            tuple(
                i.resize(
                    (target_width, target_height),
                    resample=getattr(Image.Resampling, self.resample_mode.upper()),
                )
                for i in torch_to_pilimgbatch(imgbatch)
            ),
        )
