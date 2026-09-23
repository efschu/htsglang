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

import contextlib
import logging
import os
import re
import time
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

ENV = "SGLANG_SPEC_STAGE_SYNC"
_SEEN: Dict[str, int] = {}
_STATE = {"budget": None, "last_ok": "none", "checked_ct": 0}

# x44 bounded the fault to the verify FORWARD on TP0 (last_ok=verify-prepare,
# 4.3 s in the sync). ``SGLANG_SPEC_EAGER_VERIFY=N`` runs the first N verify
# rounds eager instead of replaying the captured graph, with a stream sync
# after every decoder layer and every direct child of one: a fault in round
# 1 names its module; a clean eager round 1 followed by a dying graph round 2
# makes the fault graph-specific. Rank-uniform: every rank's verify() asks
# once per round, in the same order, so all ranks go eager together.
EAGER_ENV = "SGLANG_SPEC_EAGER_VERIFY"
_EAGER = {"budget": None, "round_ct": 0}
_MODULE = {"active": False, "models": set(), "last_ok": "none", "checked_ct": 0}
#: A module whose sync waited at least this long gets its own line.
SLOW_MODULE_MS = 50.0
_HOOKED = re.compile(r"(^|\.)layers\.\d+(\.[^.]+)?$|^model\.[^.]+$|^[^.]+$")


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


def eager_verify_round() -> bool:
    """True while this verify round is among the first N (``SGLANG_SPEC_EAGER_VERIFY=N``)."""
    if _EAGER["budget"] is None:
        try:
            _EAGER["budget"] = max(0, int(os.environ.get(EAGER_ENV, "0") or 0))
        except ValueError:
            _EAGER["budget"] = 0
    if _EAGER["budget"] <= 0:
        return False
    _EAGER["round_ct"] += 1
    return _EAGER["round_ct"] <= _EAGER["budget"]


def _module_hook(name: str):
    def hook(_module, _args, _output):
        if not _MODULE["active"] or torch.cuda.is_current_stream_capturing():
            return
        _MODULE["checked_ct"] += 1
        t0 = time.perf_counter()
        try:
            torch.cuda.current_stream().synchronize()
        except Exception as exc:
            logger.error(
                "SPEC-MODULE-SYNC FAULT module=%s checked=%d last_ok=%s "
                "sync_ms=%.1f error=%s: %s -- the fault lies in this module's "
                "forward, after last_ok",
                name, _MODULE["checked_ct"], _MODULE["last_ok"],
                (time.perf_counter() - t0) * 1e3, type(exc).__name__,
                str(exc).splitlines()[0] if str(exc) else "",
            )
            raise
        ms = (time.perf_counter() - t0) * 1e3
        _MODULE["last_ok"] = name
        if ms >= SLOW_MODULE_MS:
            logger.info("SPEC-MODULE-SYNC slow module=%s sync_ms=%.1f", name, ms)

    return hook


def _install(model) -> int:
    if model is None:
        logger.warning("SPEC-MODULE-SYNC no model on this rank: no hooks")
        return 0
    if id(model) in _MODULE["models"]:
        return 0
    _MODULE["models"].add(id(model))
    hooked = 0
    containers = (torch.nn.ModuleList, torch.nn.ModuleDict)
    for name, module in model.named_modules():
        # A container is never called, so a hook on it could never fire.
        if name and _HOOKED.search(name) and not isinstance(module, containers):
            module.register_forward_hook(_module_hook(name))
            hooked += 1
    logger.info("SPEC-MODULE-SYNC installed hooks=%d", hooked)
    return hooked


@contextlib.contextmanager
def module_sync_window(model, active: bool):
    """Sync after every hooked module's forward while ``active``; one summary
    line per clean window (its denominator), the FAULT line otherwise."""
    if not active:
        yield
        return
    _install(model)
    start_ct = _MODULE["checked_ct"]
    _MODULE["active"] = True
    try:
        yield
    except BaseException:
        _MODULE["active"] = False
        raise
    _MODULE["active"] = False
    logger.info(
        "SPEC-MODULE-SYNC window ok checked=%d last_ok=%s",
        _MODULE["checked_ct"] - start_ct, _MODULE["last_ok"],
    )
