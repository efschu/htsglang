"""SGLANG_NAN_GUARD=1: name the first layer and forward in which a hidden
state stops being finite (Task #49, 19.09.: '!!!' answers = token 0 = NaN
logits at 259k tokens, in some boots and not in others). One reduction per
guarded tensor per forward, skipped under graph capture; the first hit per
layer is logged with the forward mode, row count and context depth, later
hits are only counted. Off = zero cost."""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

_STATE = {"on": None, "hits": {}, "total": 0}
MAX_LOGGED_PER_LAYER = 1


def nan_guard_on() -> bool:
    if _STATE["on"] is None:
        _STATE["on"] = os.environ.get("SGLANG_NAN_GUARD", "0").strip() not in ("", "0")
    return bool(_STATE["on"])


def _context_depth(forward_batch) -> int:
    for attr in ("seq_lens_cpu", "seq_lens"):
        v = getattr(forward_batch, attr, None)
        if v is not None:
            try:
                return int(v.max().item())
            except Exception:  # noqa: BLE001
                pass
    return -1


def check(tag: str, tensor: Optional[torch.Tensor], layer_id, forward_batch=None) -> bool:
    """Returns True when the tensor is finite (or the guard is off/skipped)."""
    if not nan_guard_on() or tensor is None or tensor.numel() == 0:
        return True
    try:
        if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            return True
    except Exception:  # noqa: BLE001
        pass
    finite = bool(torch.isfinite(tensor).all().item())
    if finite:
        return True
    _STATE["total"] += 1
    key = (tag, int(layer_id) if layer_id is not None else -1)
    n = _STATE["hits"].get(key, 0) + 1
    _STATE["hits"][key] = n
    if n <= MAX_LOGGED_PER_LAYER:
        mode = getattr(forward_batch, "forward_mode", None)
        mode_s = getattr(mode, "name", str(mode)) if mode is not None else "?"
        bad = (~torch.isfinite(tensor)).sum().item()
        logger.error(
            "[nan-guard] FIRST non-finite %s at layer %s: %d of %d elements, forward_mode=%s, "
            "rows=%d, context depth=%d (total hits so far %d)",
            tag, key[1], bad, tensor.numel(), mode_s, int(tensor.shape[0]),
            _context_depth(forward_batch), _STATE["total"],
        )
    return False


def hits() -> Dict:
    return dict(_STATE["hits"])


def _reset_for_tests(on: bool):
    _STATE.update({"on": on, "hits": {}, "total": 0})
