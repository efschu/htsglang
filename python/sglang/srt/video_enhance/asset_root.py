# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Where the video-enhance model artifacts live (#251).

Every checkpoint, engine-cache and SR-weight default in this package was a
separate literal pointing at one rig's directory layout. Those literals are
correct for that rig and wrong everywhere else, and a released image has no
way to move them all without touching five call sites.

This module holds the one place that decides. The fallback is the same string
those literals carried, so an environment that sets nothing behaves exactly as
it did before this module existed.

Stdlib only: it is imported from dataclass field defaults, which must stay
importable without torch, TensorRT or a CUDA context.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The rig layout this subsystem was developed against.
RIG_MODEL_ROOT = "/spinning/llm_stuff/k3-models"

#: Override for the whole artifact tree.
MODEL_ROOT_ENV = "SGLANG_VIDEO_MODEL_ROOT"


def default_model_root() -> Path:
    """Root of the video-enhance artifact tree. Env override, rig fallback."""
    return Path(os.environ.get(MODEL_ROOT_ENV) or RIG_MODEL_ROOT)


def default_engine_cache_dir() -> Path:
    """Built-engine cache. ``<root>/engines``, as before."""
    return default_model_root() / "engines"


def default_sr_model_dir() -> Path:
    """Super-resolution weights. ``<root>/sr``, as before."""
    return default_model_root() / "sr"
