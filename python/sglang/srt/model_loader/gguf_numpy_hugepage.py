# SPDX-License-Identifier: Apache-2.0
"""numpy's MADV_HUGEPAGE hint is switched off while a GGUF checkpoint streams in.

WHY (boot weg2rc7gg, RC7b bb086e1120, 2026-09-25; weg2rc5gg/weg2rc6gg had the
same shape).  Group D's GGUF producer took 700-730 s per rank against ~50 s on
group P -- same code, same file, same tensors.  The loading thread's rusage
(#89 STEP-0 ``cpu[thread]``) put the difference in the KERNEL: user 27.3 s on
both groups, sys 20.4 s (P) against 670.7 s (D), minor faults +2.5 M, major
faults equal.  All three D ranks sat in gguf-py's ``_apply_over_grouped_rows``
at ``np.concatenate(..., out=out)``: the first touch of the float32 ``out``
array of the out_proj dequantization (IQ4_XS / K-quant block 256 against
head_v_dim 128 -> dequantized on the fly, 48 GDN layers x 120 MiB per rank).

numpy (>= 1.20, Linux >= 4.6) ``madvise(MADV_HUGEPAGE)``s every array of
4 MiB or more.  With the host's THP ``defrag=madvise``, each first-touch fault
on such an array allocates a 2 MiB page SYNCHRONOUSLY, with direct compaction
and direct reclaim, in the faulting thread.  That is cheap on unfragmented
memory and ruinous behind pinned pages, which compaction can neither use nor
move (host counters over its uptime: compact_stall 2.14 M against
compact_success 9.7 k, pgmigrate_fail 2e11; Normal zone free blocks of order
>= 9: one).

Why P was not hit: group P loads FIRST, then pre-pins its HiCache arena
(#1436: 22 GiB KV + 8.2 GiB mamba of 4 KiB shmem pages, logged 09:57:56-
09:58:13, after P's load had ended at 09:57:51).  Its 2 MiB faults found or
made free blocks: P's load_weights took 8.5-8.8 M page faults (minor + major)
with the hint against 9.9-10.1 M without it (weg2rc7gg2) -- the ~1.4 M gap is
the 5.6 GiB of out_proj dequant buffers (48 x 120 MiB) arriving as 2 MiB pages
instead of 4 KiB ones.  P's ~20 s of sys is the file-fault cost every rank pays
(1.5-3 M major faults of the mmap'd GGUF), not compaction.  Group D starts at
10:00:45 behind the pinned arena: every 2 MiB fault paid a failing compaction
scan, then fell back to 4 KiB (D TP2: 10.8 M faults, the no-hint count) -- the
container's PSI memory stall read 60-70 % for the whole D load, 5 % during P's.

Metal A/B, weg2rc7gg2 (same tree, numpy hint off by NUMPY_MADVISE_HUGEPAGE=0):
D load_weights 39-50 s, sys 10.5-18.2 s; host thp_fault_alloc/fallback and
compact_stall +0 over the D load, PSI full +0.1 s in 54 s.

The dequantized values do not depend on the page size behind them: switching
the hint off changes the fault path only (4 KiB pages, no compaction).  numpy's
previous setting is restored when the load is done, so arrays the server
allocates later keep numpy's default.  ``SGLANG_GGUF_NUMPY_HUGEPAGE=1`` keeps
numpy's own setting during the load (the pre-fix behaviour, an A/B arm).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

_THP_SYSFS = "/sys/kernel/mm/transparent_hugepage"


def numpy_madvise_hugepage_switch() -> Optional[Callable[[bool], bool]]:
    """numpy's process-global hugepage setter, or None if this numpy has none.

    ``_set_madvise_hugepage(enabled)`` returns the PREVIOUS setting.  It lives
    in ``numpy._core.multiarray`` since numpy 2 (``numpy.core`` is a deprecated
    alias there) and in ``numpy.core.multiarray`` before.
    """
    try:
        from numpy._core import multiarray as ma
    except ImportError:  # numpy 1.x
        try:
            from numpy.core import multiarray as ma
        except ImportError:
            return None
    return getattr(ma, "_set_madvise_hugepage", None)


def thp_mode(knob: str) -> str:
    """The active value of a THP sysfs knob (``enabled``/``defrag``), or '?'."""
    try:
        with open(f"{_THP_SYSFS}/{knob}") as f:
            text = f.read()
    except OSError:
        return "?"
    start, end = text.find("["), text.find("]")
    if 0 <= start < end:
        return text[start + 1 : end]
    return text.strip() or "?"


def gguf_numpy_hugepage_kept() -> bool:
    """``SGLANG_GGUF_NUMPY_HUGEPAGE=1``: leave numpy's setting alone."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_GGUF_NUMPY_HUGEPAGE.get())


@contextlib.contextmanager
def numpy_hugepage_off_for_gguf_load(
    rank: object = None,
    switch: Optional[Callable[[bool], bool]] = None,
) -> Iterator[bool]:
    """Scope of the GGUF weight stream in which numpy does not hint MADV_HUGEPAGE.

    Yields True when it switched the hint off, False when it left numpy alone
    (opt-out env, or a numpy without the switch).  The previous setting comes
    back on exit, also when the load raises.  ``switch`` is injectable for
    tests; by default it is numpy's own setter.
    """
    thp = f"host THP enabled={thp_mode('enabled')} defrag={thp_mode('defrag')}"
    if gguf_numpy_hugepage_kept():
        logger.info(
            "[GGUF-NUMPY-THP] rank %s: SGLANG_GGUF_NUMPY_HUGEPAGE=1 -- numpy keeps "
            "its own MADV_HUGEPAGE setting during the GGUF weight stream "
            "(pre-fix behaviour; %s).",
            rank,
            thp,
        )
        yield False
        return
    if switch is None:
        switch = numpy_madvise_hugepage_switch()
    if switch is None:
        logger.info(
            "[GGUF-NUMPY-THP] rank %s: this numpy has no _set_madvise_hugepage, "
            "nothing to switch (%s).",
            rank,
            thp,
        )
        yield False
        return
    previous = bool(switch(False))
    logger.info(
        "[GGUF-NUMPY-THP] rank %s: numpy MADV_HUGEPAGE OFF for the GGUF weight "
        "stream (was %s; %s) -- dequant buffers fault 4 KiB pages instead of "
        "compacting for 2 MiB ones. SGLANG_GGUF_NUMPY_HUGEPAGE=1 keeps numpy's "
        "setting.",
        rank,
        "on" if previous else "off",
        thp,
    )
    try:
        yield True
    finally:
        switch(previous)


def numpy_hugepage_off_during_load(load_model: Callable) -> Callable:
    """Decorator for a loader's ``load_model(self, ...)``: the whole call runs
    inside :func:`numpy_hugepage_off_for_gguf_load`, in whatever process calls
    it -- the rank process -- so no environment variable has to travel from a
    launcher to the rank.  The rank in the log line is
    ``self.load_config.tp_rank``."""

    @functools.wraps(load_model)
    def wrapper(self, *args, **kwargs):
        rank = getattr(getattr(self, "load_config", None), "tp_rank", None)
        with numpy_hugepage_off_for_gguf_load(rank):
            return load_model(self, *args, **kwargs)

    return wrapper
