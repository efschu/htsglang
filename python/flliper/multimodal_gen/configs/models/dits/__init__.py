# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from flliper.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from flliper.multimodal_gen.configs.models.dits.helios import HeliosConfig
from flliper.multimodal_gen.configs.models.dits.hunyuan3d import Hunyuan3DDiTConfig
from flliper.multimodal_gen.configs.models.dits.hunyuanvideo import HunyuanVideoConfig
from flliper.multimodal_gen.configs.models.dits.ideogram import Ideogram4DiTConfig
from flliper.multimodal_gen.configs.models.dits.lingbot_world import (
    LingBotWorldVideoConfig,
)
from flliper.multimodal_gen.configs.models.dits.mova_audio import MOVAAudioConfig
from flliper.multimodal_gen.configs.models.dits.mova_video import MOVAVideoConfig
from flliper.multimodal_gen.configs.models.dits.stablediffusion3 import (
    StableDiffusion3TransformerConfig,
)
from flliper.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig

__all__ = [
    "Cosmos3VideoConfig",
    "HeliosConfig",
    "HunyuanVideoConfig",
    "Ideogram4DiTConfig",
    "LingBotWorldVideoConfig",
    "WanVideoConfig",
    "Hunyuan3DDiTConfig",
    "MOVAAudioConfig",
    "MOVAVideoConfig",
    "StableDiffusion3TransformerConfig",
]
