# Copyright 2026 SGLang Team
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
"""Model reference resolution for the offline planner.

Turns ``--model <path|hf-id>`` into a path ``PerfCostModel`` can size
WITHOUT loading weights: a directory with ``config.json`` (safetensors
checkpoints — the byte model anchors on the on-disk file sizes) or a single
``.gguf`` file (the cost model reads dims + per-family quant bytes straight
from the GGUF header via ``_gguf_config_and_families``). For an HF hub id,
only ``config.json`` is downloaded (a few KiB) — never weights.
"""

from __future__ import annotations

import glob
import os

__all__ = ["resolve_model_ref"]


def resolve_model_ref(model_ref: str) -> str:
    """Resolve a model reference to the path the cost model consumes.

    Accepted forms:
      * a directory containing ``config.json``  (HF checkpoint layout);
      * a ``.gguf`` file path;
      * a directory containing exactly one ``*.gguf`` and no ``config.json``
        -> resolved to that file (a bare GGUF download dir);
      * an HF hub id -> ``config.json`` is fetched into the HF cache
        (config only; no weights, no tokenizer).

    Note: a GGUF dir that ALSO carries a config.json (e.g. Unsloth exports)
    resolves to the ``.gguf`` file, matching how the server is launched on a
    GGUF checkpoint (the byte model must come from the ggml tensor types,
    not from the BF16-shaped config.json).
    """
    if os.path.isfile(model_ref):
        return model_ref

    if os.path.isdir(model_ref):
        ggufs = sorted(glob.glob(os.path.join(model_ref, "*.gguf")))
        # Exclude importance-matrix side files which are not checkpoints.
        ggufs = [g for g in ggufs if "imatrix" not in os.path.basename(g).lower()]
        if ggufs:
            if len(ggufs) > 1:
                raise ValueError(
                    f"{model_ref} contains {len(ggufs)} .gguf files "
                    f"({[os.path.basename(g) for g in ggufs]}); point --model "
                    "at the specific .gguf file to plan."
                )
            return ggufs[0]
        if os.path.isfile(os.path.join(model_ref, "config.json")):
            return model_ref
        raise ValueError(
            f"{model_ref} is a directory with neither config.json nor a "
            ".gguf file — not a checkpoint the planner can size."
        )

    # Not a local path: treat as an HF hub id and fetch ONLY config.json.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ValueError(
            f"--model {model_ref!r} is not a local path, and huggingface_hub "
            "is not installed to resolve it as a hub id. Pass a local "
            "checkpoint directory or .gguf file."
        ) from e
    try:
        cfg_path = hf_hub_download(repo_id=model_ref, filename="config.json")
    except Exception as e:
        raise ValueError(
            f"Could not resolve --model {model_ref!r}: not a local path, and "
            f"fetching config.json from the HF hub failed ({e})."
        ) from e
    # PerfCostModel reads <dir>/config.json; the snapshot dir holds it.
    # Checkpoint-size anchoring is unavailable (no weights on disk) — the
    # byte model then clamps bytes/param conservatively, which the CLI warns
    # about.
    return os.path.dirname(cfg_path)
