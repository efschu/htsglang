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
from typing import List, Optional

__all__ = ["resolve_model_ref", "list_gguf_options", "list_gguf_options_any"]


def _is_sidecar_gguf(basename: str) -> bool:
    """A .gguf that is NOT a checkpoint to serve: the importance-matrix file
    (``imatrix``/``*_imatrix*``) and the vision projector (``mmproj-*``) that
    ships beside a multimodal GGUF. Neither is a text checkpoint the planner
    sizes, so both are excluded from the selectable list."""
    b = basename.lower()
    return "imatrix" in b or b.startswith("mmproj") or "mmproj" in b


def list_gguf_options(model_ref: str) -> List[str]:
    """The selectable checkpoint ``.gguf`` files in a directory (sidecars
    excluded), as bare filenames sorted for a stable dropdown. Empty when
    ``model_ref`` is not a directory or holds no checkpoint .gguf. Used by the
    web UI to offer a quant picker for an Unsloth-style multi-gguf export."""
    if not os.path.isdir(model_ref):
        return []
    ggufs = sorted(
        os.path.basename(g) for g in glob.glob(os.path.join(model_ref, "*.gguf"))
    )
    return [g for g in ggufs if not _is_sidecar_gguf(g)]


def list_gguf_options_any(model_ref: str) -> List[str]:
    """Selectable ``.gguf`` checkpoint filenames for the quant dropdown, for a
    LOCAL directory OR an HF-hub GGUF repo id.

    Local dirs go through :func:`list_gguf_options`. For anything that is not a
    local path we ask the HF hub for the repo's file list (``list_repo_files``
    — metadata only, NO weight download) and keep the checkpoint ``.gguf``
    files, excluding the ``mmproj-*`` / ``imatrix`` sidecars. Returns an empty
    list (never raises) when the ref is neither a multi-gguf dir nor a
    resolvable GGUF repo, so the UI simply shows no dropdown."""
    if os.path.isdir(model_ref):
        return list_gguf_options(model_ref)
    if os.path.isfile(model_ref):
        return []
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(model_ref)
    except Exception:
        return []
    ggufs = sorted(
        os.path.basename(f) for f in files if f.lower().endswith(".gguf")
    )
    return [g for g in ggufs if not _is_sidecar_gguf(g)]


def resolve_model_ref(model_ref: str, gguf_choice: Optional[str] = None) -> str:
    """Resolve a model reference to the path the cost model consumes.

    Accepted forms:
      * a directory containing ``config.json``  (HF checkpoint layout);
      * a ``.gguf`` file path;
      * a directory containing ``*.gguf`` files (a GGUF download dir, with or
        WITHOUT a config.json beside them) -> resolved to a .gguf file: the
        single checkpoint if there is exactly one, or the ``gguf_choice`` the
        caller selected when there are several;
      * an HF hub id -> ``config.json`` is fetched into the HF cache
        (config only; no weights, no tokenizer).

    A GGUF dir that ALSO carries a config.json (e.g. Unsloth exports) resolves
    to the ``.gguf`` file, NOT the directory: the config.json in those exports
    is the BF16-shaped source config (hidden_size/num_layers of the unquantized
    model), so sizing from it would over-count the weights ~4x ("no space" when
    there is space). The real quant bytes must come from the ggml tensor types
    in the GGUF header, exactly as the server loads it.

    ``gguf_choice`` (bare filename) selects one file from a multi-gguf dir;
    sidecar files (``imatrix``, ``mmproj-*``) are never selectable.
    """
    if os.path.isfile(model_ref):
        return model_ref

    if os.path.isdir(model_ref):
        options = list_gguf_options(model_ref)
        has_config = os.path.isfile(os.path.join(model_ref, "config.json"))
        if options:
            if gguf_choice is not None:
                choice = os.path.basename(gguf_choice)
                if choice not in options:
                    raise ValueError(
                        f"gguf_choice {gguf_choice!r} is not one of the "
                        f"checkpoint .gguf files in {model_ref}: {options}."
                    )
                return os.path.join(model_ref, choice)
            if len(options) == 1:
                return os.path.join(model_ref, options[0])
            raise ValueError(
                f"{model_ref} contains {len(options)} .gguf checkpoints "
                f"({options}); pick one (--model <dir>/<file.gguf>, or use the "
                "quant dropdown in the web UI)."
            )
        # No .gguf checkpoint -> a plain HF/safetensors dir.
        if has_config:
            return model_ref
        raise ValueError(
            f"{model_ref} is a directory with neither config.json nor a "
            ".gguf file — not a checkpoint the planner can size."
        )

    # Not a local path: treat as an HF hub id.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ValueError(
            f"--model {model_ref!r} is not a local path, and huggingface_hub "
            "is not installed to resolve it as a hub id. Pass a local "
            "checkpoint directory or .gguf file."
        ) from e

    # A GGUF repo id: the caller picked one quant from the dropdown (an
    # unpicked multi-gguf repo is reported so the UI can prompt). Sizing a GGUF
    # needs its header, so the CHOSEN .gguf is fetched (only that shard, not the
    # whole repo); config-only is not enough for GGUF sizing.
    if gguf_choice is not None:
        try:
            return hf_hub_download(
                repo_id=model_ref, filename=os.path.basename(gguf_choice)
            )
        except Exception as e:
            raise ValueError(
                f"Could not fetch GGUF {gguf_choice!r} from repo "
                f"{model_ref!r}: {e}."
            ) from e
    hub_ggufs = list_gguf_options_any(model_ref)
    if len(hub_ggufs) == 1:
        return hf_hub_download(repo_id=model_ref, filename=hub_ggufs[0])
    if len(hub_ggufs) > 1:
        raise ValueError(
            f"{model_ref} (HF repo) holds {len(hub_ggufs)} .gguf checkpoints "
            f"({hub_ggufs}); pick one (quant dropdown in the web UI, or "
            "--model <repo>/<file.gguf>)."
        )

    # A safetensors repo: fetch ONLY config.json (config-authoritative sizing
    # kicks in when the weight shards are absent -- see PerfCostModel).
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
