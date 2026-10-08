# SPDX-License-Identifier: Apache-2.0
"""Frame interpolation and upscaling support for fLLiper diffusion pipelines."""

from flliper.multimodal_gen.runtime.postprocess.realesrgan_upscaler import (
    ImageUpscaler,
    batch_upscale_frames,
    upscale_frames,
)
from flliper.multimodal_gen.runtime.postprocess.rife_interpolator import (
    FrameInterpolator,
    interpolate_video_frames,
)

__all__ = [
    "FrameInterpolator",
    "interpolate_video_frames",
    "ImageUpscaler",
    "batch_upscale_frames",
    "upscale_frames",
]
