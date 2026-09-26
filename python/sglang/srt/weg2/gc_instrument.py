# SPDX-License-Identifier: Apache-2.0
"""GC instrument and optional gc.freeze() for the scheduler (rank) processes.

Agent FR (26.09.) found three single-rank D-round stalls of ~0.5 s; candidate:
a Python generation-2 collection on that rank's scheduler.  Upstream arms
``--gc-warning-threshold-secs`` only in the TokenizerManager process
(``managers/tokenizer_manager.py``), i.e. not where a D round can stall.

* ``server_args.gc_warning_threshold_secs > 0``: the same warning callback
  (``utils.common.configure_gc_warning``) is armed in every scheduler once the
  scheduler is up (boot-time collections during weight load are not the
  question).  Default 0.0 -> nothing, as upstream.
* ``SGLANG_WEG2_GC_FREEZE=1`` (launcher ``--d-gc-freeze on``, group D): one
  ``gc.freeze()`` at the same point -- after every runner in the process has
  captured its CUDA graphs -- so later gen-2 passes no longer walk the boot
  objects.  Unset (default) -> nothing.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

FREEZE_ENV = "SGLANG_WEG2_GC_FREEZE"


def freeze_requested(env: Optional[Mapping[str, str]] = None) -> bool:
    return str((env if env is not None else os.environ).get(FREEZE_ENV, "") or "").strip() == "1"


def arm_after_boot(server_args, tp_rank: int, env: Optional[Mapping[str, str]] = None) -> dict:
    """Called once per scheduler process right after the Scheduler exists.
    Returns what was armed ({'warn': secs|None, 'freeze': bool}) for tests."""
    out = {"warn": None, "freeze": False}
    thr = float(getattr(server_args, "gc_warning_threshold_secs", 0.0) or 0.0)
    if thr > 0.0:
        from sglang.srt.utils.common import configure_gc_warning

        configure_gc_warning(thr)
        out["warn"] = thr
        logger.info("WEG2-GC warn armed in scheduler TP%d: collections > %.3f s are logged", tp_rank, thr)
    if freeze_requested(env):
        from sglang.srt.utils.common import freeze_gc

        freeze_gc(f"scheduler TP{tp_rank} (after boot, {FREEZE_ENV}=1)")
        out["freeze"] = True
    return out
