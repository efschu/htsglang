"""23.09. (fnFL2x43): D's first decode round dies on its host rank (TP0, the
5090) with an asynchronous 'illegal memory access' that surfaces one round
LATE -- at the next round's ``forward_stream.wait_stream`` -- and no GPU
coredump is written although the coredump environment reached the rank.
A 1-token health decode in the same process survives; a 655-token request
(ten full pages, radix insert with mamba tracking) dies. The traceback names
the place the error was REPORTED, never the stage that faulted.

``SGLANG_SPEC_STAGE_SYNC=N`` synchronizes the stream a stage ran on after the
first N passes through each stage and logs one line per checkpoint with the
time the sync waited. An asynchronous CUDA error is sticky, so the FIRST
checkpoint whose sync raises bounds the fault: it lies between that stage and
the stage named ``last_ok``. The error is logged and re-raised, never
swallowed. Off by default; skipped during stream capture.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

ENV = "SGLANG_SPEC_STAGE_SYNC"
_SEEN: Dict[str, int] = {}
_STATE = {"budget": None, "last_ok": "none", "checked_ct": 0}


def _budget() -> int:
    if _STATE["budget"] is None:
        try:
            _STATE["budget"] = max(0, int(os.environ.get(ENV, "0") or 0))
        except ValueError:
            _STATE["budget"] = 0
    return _STATE["budget"]


def checkpoint(stage: str, stream: Optional["torch.cuda.Stream"] = None) -> None:
    """Sync ``stream`` (default: the current one) after ``stage``; log it."""
    budget = _budget()
    if budget <= 0:
        return
    n = _SEEN.get(stage, 0)
    if n >= budget or torch.cuda.is_current_stream_capturing():
        return
    _SEEN[stage] = n + 1
    _STATE["checked_ct"] += 1
    target = stream if stream is not None else torch.cuda.current_stream()
    t0 = time.perf_counter()
    try:
        target.synchronize()
    except Exception as exc:
        logger.error(
            "SPEC-STAGE-SYNC FAULT stage=%s n=%d checked=%d last_ok=%s "
            "sync_ms=%.1f error=%s: %s -- the fault lies between last_ok and "
            "this stage",
            stage, n + 1, _STATE["checked_ct"], _STATE["last_ok"],
            (time.perf_counter() - t0) * 1e3, type(exc).__name__,
            str(exc).splitlines()[0] if str(exc) else "",
        )
        raise
    _STATE["last_ok"] = f"{stage}#{n + 1}"
    logger.info(
        "SPEC-STAGE-SYNC ok stage=%s n=%d checked=%d sync_ms=%.1f",
        stage, n + 1, _STATE["checked_ct"], (time.perf_counter() - t0) * 1e3,
    )
