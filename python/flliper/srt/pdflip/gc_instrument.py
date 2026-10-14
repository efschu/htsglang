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
* ``FLLIPER_PDFLIP_GC_FREEZE=1`` (launcher ``--d-gc-freeze on`` / ``--p-gc-freeze
  on``, and the nextflash and qwen27b rows' group switch defaults for P and D): one
  ``gc.freeze()`` at the same point -- after every runner in the process has
  captured its CUDA graphs -- so later gen-2 passes no longer walk the boot
  objects.  Unset (default) -> nothing.

NF 13.10. (boot b60520f01d, 11 of 455 flips over 6 s, max 9.2 s): a gen-2 pass
over the ~1.12 M never-frozen boot objects took 4.4-5.4 s on PP0/PP1 once the
cold heap had been swapped out (host pswpin 270-565 MiB in those seconds); the
pass holds the GIL, so the P leg stood still and D waited in its collect
(``done/nf-flip-langsam-analyse-1013.md``).  The freeze takes the boot heap out
of every later pass; the marker below names what was frozen per rank.

TWO TRAPS, both covered by tests:

* CPython 3.12 parks its immortal objects in the permanent generation on every
  collection (375 measured on a bare interpreter after ``gc.collect()``), so
  ``gc.get_freeze_count() > 0`` is NOT "someone froze" (b25e417304).  A freeze
  counts as present only from :data:`FOREIGN_FREEZE_MIN` objects (a real freeze
  of a scheduler process is ~800k); a rank that already carries one is left
  alone and the marker says so.
* a later CUDA-graph capture (``base_cuda_graph_runner.freeze_gc``) used to end
  in ``gc.unfreeze()`` -- which hands the WHOLE permanent generation back, this
  freeze included, followed by a full collection over 1.1 M objects, i.e. the
  stall itself, inside a flip.  The runner now asks
  :func:`permanent_generation_populated` and leaves a populated permanent
  generation alone.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

FREEZE_ENV = "FLLIPER_PDFLIP_GC_FREEZE"

#: the per-rank line of the after-boot freeze (the metal criterion: ``frozen``
#: ~ 1.1 M, ``gen2_after`` a few thousand, and no LONG GARBAGE COLLECTION line
#: over a second afterwards).
FREEZE_MARKER = "GC-FREEZE"

#: Above this many objects the permanent generation holds a real freeze;
#: below it, only the immortals CPython 3.12 parks there itself (375 measured
#: after a bare ``gc.collect()``).  Same value as ``model_loader/load_gc.py``
#: ``_FOREIGN_FREEZE_MIN`` (a test pins the two together).
FOREIGN_FREEZE_MIN = 20000


#: 30.09. (NF y3z ep52, PP0 5.4 s process-wide still): the group env that
#: arms the warning without the CLI flag -- the launcher sets it for P.
WARN_ENV = "FLLIPER_PDFLIP_GC_WARN_SECS"
P_GC_WARN_DEFAULT_S = 0.5


def warn_secs_from_env(env: Optional[Mapping[str, str]] = None) -> float:
    """``FLLIPER_PDFLIP_GC_WARN_SECS`` as seconds (0.0 = off or unparsable)."""
    if env is None:
        from flliper.srt.environ import envs

        try:
            v = float(envs.FLLIPER_PDFLIP_GC_WARN_SECS.get() or 0.0)
        except (TypeError, ValueError):
            return 0.0
    else:
        try:
            v = float(str(env.get(WARN_ENV, "") or "0").strip() or 0.0)
        except ValueError:
            return 0.0
    return v if v > 0.0 else 0.0


def freeze_requested(env: Optional[Mapping[str, str]] = None) -> bool:
    return str((env if env is not None else os.environ).get(FREEZE_ENV, "") or "").strip() == "1"


def permanent_generation_populated() -> bool:
    """True when the permanent generation holds a REAL freeze (>=
    :data:`FOREIGN_FREEZE_MIN` objects) -- not the 3.12 immortals."""
    try:
        return gc.get_freeze_count() >= FOREIGN_FREEZE_MIN
    except Exception:  # noqa: BLE001 - an unreadable count is "nobody froze"
        return False


def rank_label(tp_rank: int, pp_rank: Optional[int] = None) -> str:
    return f"PP{pp_rank}TP{tp_rank}" if pp_rank is not None else f"TP{tp_rank}"


def freeze_after_boot(tp_rank: int, pp_rank: Optional[int] = None) -> bool:
    """One ``gc.collect()`` (nothing already garbage gets frozen) and one
    ``gc.freeze()``, then the ``GC-FREEZE frozen=<n> gen2_after=<m> rank=<r>``
    line.  Returns True when it froze; a rank whose permanent generation
    already holds a real freeze is left alone (False, and the line says so)."""
    label = rank_label(tp_rank, pp_rank)
    if permanent_generation_populated():
        logger.info(
            "%s skipped rank=%s: %d objects already frozen (a second freeze adds nothing "
            "and gc.unfreeze() could not give back only one part)",
            FREEZE_MARKER, label, gc.get_freeze_count(),
        )
        return False
    t0 = time.perf_counter()
    gc.collect()
    collect_ms = (time.perf_counter() - t0) * 1000.0
    gen2_before = len(gc.get_objects(2))
    gc.freeze()
    frozen = gc.get_freeze_count()
    gen2_after = len(gc.get_objects(2))
    logger.info(
        "%s frozen=%d gen2_after=%d rank=%s gen2_before=%d collect_ms=%.0f",
        FREEZE_MARKER, frozen, gen2_after, label, gen2_before, collect_ms,
    )
    return True


def arm_after_boot(server_args, tp_rank: int, env: Optional[Mapping[str, str]] = None,
                   pp_rank: Optional[int] = None) -> dict:
    """Called once per scheduler process right after the Scheduler exists.
    Returns what was armed ({'warn': secs|None, 'freeze': bool}) for tests."""
    out = {"warn": None, "freeze": False}
    thr = float(getattr(server_args, "gc_warning_threshold_secs", 0.0) or 0.0)
    if thr <= 0.0:
        thr = warn_secs_from_env(env)  # 30.09.: P's default via the launcher
    if thr > 0.0:
        from flliper.srt.utils.common import configure_gc_warning

        configure_gc_warning(thr)
        out["warn"] = thr
        logger.info("PDFLIP-GC warn armed in scheduler TP%d: collections > %.3f s are logged", tp_rank, thr)
    if freeze_requested(env):
        out["freeze"] = freeze_after_boot(tp_rank, pp_rank)
    return out
