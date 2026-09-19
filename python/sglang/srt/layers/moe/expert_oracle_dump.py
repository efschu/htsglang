"""Expert-oracle dump (Task #45, 19.09.): what a predictor would have to guess.

SGLANG_EXPERT_ORACLE_DUMP=<dir> records, on TP rank 0 only and only for
small (decode/verify) forwards, per MoE layer the MoE input hidden states
and the router's GLOBAL top-k expert ids, and per draft (MTP) forward the
draft's input, fused input and output hidden states. Offline
(weg2/expert_oracle_precision.py) that answers, with the checkpoint's gate
weights, three precisions against the 59 % break-even:
(a) raw gate of layer L+1 on the state after L (today's prefetch),
(b) the draft's hidden state pushed through the target's routers (the
    "MTP head as expert oracle" idea), (c) the ceiling for a learned map.
Eager only: the recorder copies to host, which a captured graph cannot.
"""
from __future__ import annotations

import atexit
import logging
import os
import re
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_STATE = {"dir": None, "checked": False, "rank_ok": None}
_TARGET: List[dict] = []
_DRAFT: List[dict] = []
_COUNTS = {"target": 0, "draft": 0, "target_files": 0, "draft_files": 0}
MAX_TOKENS = 8
FLUSH_TARGET = 48 * 64  # 64 forwards of 48 layers
FLUSH_DRAFT = 256


def oracle_dir() -> Optional[str]:
    if not _STATE["checked"]:
        _STATE["checked"] = True
        d = os.environ.get("SGLANG_EXPERT_ORACLE_DUMP", "").strip()
        _STATE["dir"] = d or None
        if d:
            os.makedirs(d, exist_ok=True)
            atexit.register(flush)
    return _STATE["dir"]


def _rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 -- hermetic / no group
        return 0


def active(force_rank: Optional[int] = None) -> bool:
    if oracle_dir() is None:
        return False
    if _STATE["rank_ok"] is None:
        _STATE["rank_ok"] = (_rank() if force_rank is None else force_rank) == 0
    return bool(_STATE["rank_ok"])


def layer_index(prefix: str, layer_id) -> int:
    if layer_id is not None:
        try:
            return int(layer_id)
        except (TypeError, ValueError):
            pass
    m = re.search(r"layers\.(\d+)\.", prefix or "")
    return int(m.group(1)) if m else -1


def record_target(prefix: str, layer_id, hidden: torch.Tensor, topk_ids: torch.Tensor) -> bool:
    """One MoE forward: hidden [T, H] and global ids [T, k]. Skipped when the
    forward is bigger than MAX_TOKENS rows (prefill)."""
    if not active() or hidden is None or topk_ids is None:
        return False
    if hidden.shape[0] > MAX_TOKENS:
        return False
    _TARGET.append(
        {
            "t": time.monotonic(),
            "layer": layer_index(prefix, layer_id),
            "hidden": hidden.detach().to(torch.bfloat16).cpu(),
            "ids": topk_ids.detach().to(torch.int16).cpu(),
        }
    )
    _COUNTS["target"] += 1
    if len(_TARGET) >= FLUSH_TARGET:
        _flush_target()
    return True


def record_draft(kind: str, hidden: torch.Tensor) -> bool:
    if not active() or hidden is None or hidden.shape[0] > MAX_TOKENS:
        return False
    _DRAFT.append(
        {"t": time.monotonic(), "kind": kind, "hidden": hidden.detach().to(torch.bfloat16).cpu()}
    )
    _COUNTS["draft"] += 1
    if len(_DRAFT) >= FLUSH_DRAFT:
        _flush_draft()
    return True


def _flush_target():
    if not _TARGET:
        return
    d = oracle_dir()
    n = _COUNTS["target_files"]
    torch.save(_TARGET[:], os.path.join(d, f"target_{n:05d}.pt"))
    _COUNTS["target_files"] = n + 1
    _TARGET.clear()


def _flush_draft():
    if not _DRAFT:
        return
    d = oracle_dir()
    n = _COUNTS["draft_files"]
    torch.save(_DRAFT[:], os.path.join(d, f"draft_{n:05d}.pt"))
    _COUNTS["draft_files"] = n + 1
    _DRAFT.clear()


def flush():
    if oracle_dir() is None:
        return
    _flush_target()
    _flush_draft()
    logger.info(
        "[expert-oracle] flushed: %d target records in %d files, %d draft records in %d files -> %s",
        _COUNTS["target"], _COUNTS["target_files"], _COUNTS["draft"], _COUNTS["draft_files"], oracle_dir(),
    )


def _reset_for_tests(dir_: Optional[str], rank: int = 0):
    flush() if _STATE["dir"] else None
    _STATE.update({"dir": dir_, "checked": True, "rank_ok": rank == 0})
    _TARGET.clear(); _DRAFT.clear()
    for k in _COUNTS:
        _COUNTS[k] = 0
    if dir_:
        os.makedirs(dir_, exist_ok=True)
