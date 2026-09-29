# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from flliper.multimodal_gen.configs.models.vaes.dac import DacVAEConfig
from flliper.multimodal_gen.configs.models.vaes.hunyuan3d import Hunyuan3DVAEConfig
from flliper.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from flliper.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from flliper.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

__all__ = [
    "DacVAEConfig",
    "HunyuanVAEConfig",
    "StableDiffusion3VAEConfig",
    "WanVAEConfig",
    "Hunyuan3DVAEConfig",
]
