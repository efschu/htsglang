# SPDX-License-Identifier: Apache-2.0
"""EM-PREFETCH-SM (21.10.): Marlin leaves the prefetch copy its SMs while a side-stream copy is in flight.

WHY. A Marlin GEMM (dense ``gptq_marlin`` and MoE ``moe_wna16_marlin``) launches ``sms x blocks_per_sm`` blocks.
With ``blocks_per_sm == 1`` (every prefill-sized M: ``thread_m_blocks > 1``) each block asks for the whole opt-in
shared memory (``cudaDevAttrMaxSharedMemoryPerBlockOptin`` = 99 KB on sm_86 and sm_120) plus the 1 KB the driver
reserves per block = the SM's entire 100 KB. Any other block resident on an SM -- the EM-PREFETCH row copy is a
persistent Triton kernel, 4 warps, 72 registers, 0 B dynamic shared memory but its own 1 KB reserved -- therefore
shuts that SM to Marlin, and because Marlin's grid is exactly one wave of ``sms`` blocks, ONE such SM already pushes
a block into a second wave: the GEMM takes ~2x. rc31b D (``..._1021_202608``) shows exactly that on the MoE apply
(T 512-1023: 295 ms with >= 1000 prefetched rows against 196 with < 100; rc30a 174), while the small-T apply, where
Marlin splits the shared memory over up to 4 blocks per SM and a copy block still fits, did not move.

WHAT. While an EM-PREFETCH copy is in flight on this rank (``open_window`` .. ``close_window``, host order: from the
moment the side stream was forked behind layer L's last row copy until layer L+1 joins it), every Marlin launch takes
``sms - reserve`` as its grid, ``reserve`` = the copy's program count: one full wave on the SMs the copy does not hold.
The narrower grid is the path the existing ``SGLANG_MARLIN_SMS_OVERRIDE`` already takes (only DOWNWARD: the lock
workspace and ``c_tmp`` were sized from the hardware count). Outside a window, under graph capture, and with the
switch ``FLLIPER_MOE_POOL_EM_PREFETCH_SM_SHARE`` off the reserve is 0 and every launch is the one before.

No host read, no collective, rank-local (a rank whose layer skipped its prefetch has no window). Marlin's k-split
depends on the grid, so a GEMM inside a window may round differently in the last bf16 bit than outside -- the same
rank-local property a different card already has; nothing cross-rank compares these partials.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_STATE = {"open": 0, "reserve": 0, "on": None, "leaks": 0, "logged": False}


def enabled() -> bool:
    """FLLIPER_MOE_POOL_EM_PREFETCH_SM_SHARE (default ON), read once per process."""
    if _STATE["on"] is None:
        try:
            from flliper.srt.environ import envs

            _STATE["on"] = bool(envs.FLLIPER_MOE_POOL_EM_PREFETCH_SM_SHARE.get())
        except Exception:  # noqa: BLE001 -- no environ (bare harness): the default
            _STATE["on"] = True
    return bool(_STATE["on"])


def reset_for_tests() -> None:
    _STATE.update({"open": 0, "reserve": 0, "on": None, "leaks": 0, "logged": False})


def open_window(reserve: int) -> None:
    """A side-stream copy of ``reserve`` persistent programs was just issued (host order)."""
    if reserve <= 0 or not enabled():
        return
    _STATE["open"] += 1
    _STATE["reserve"] = max(int(_STATE["reserve"]), int(reserve))
    if not _STATE["logged"]:
        _STATE["logged"] = True
        logger.info(
            "EM-PREFETCH-SM: Marlin GEMMs inside an EM-PREFETCH window launch sms-%d blocks (one wave beside the "
            "copy's %d programs; FLLIPER_MOE_POOL_EM_PREFETCH_SM_SHARE)", int(reserve), int(reserve))


def close_window() -> None:
    """The window's target joined the copy (host order)."""
    if _STATE["open"] > 0:
        _STATE["open"] -= 1
    if _STATE["open"] == 0:
        _STATE["reserve"] = 0


def forward_start() -> None:
    """The scheduler funnel at a forward's start: every window of the previous forward was joined inside it (the
    join is the target layer's first act). One still open is a leak -- close it, name it once."""
    if _STATE["open"] > 0:
        _STATE["leaks"] += 1
        if _STATE["leaks"] == 1:
            logger.warning(
                "EM-PREFETCH-SM: %d prefetch window(s) still open at a forward start -- closed (Marlin back to the "
                "full grid); report it", int(_STATE["open"]))
        _STATE["open"] = 0
        _STATE["reserve"] = 0


def _capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 -- no CUDA: nothing is captured
        return False


def sm_reserve() -> int:
    """The SMs the next Marlin launch on this rank leaves free: the open window's copy programs, else 0. Never
    under graph capture (a captured grid would outlive the window)."""
    if _STATE["open"] <= 0 or not enabled():
        return 0
    if _capturing():
        return 0
    return int(_STATE["reserve"])
