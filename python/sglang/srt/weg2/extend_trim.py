"""WEG2-EXTEND-CACHE-TRIM: empty the caching allocator before a D extend when the card is short (rc12e).

What rc12d showed (27.09., D-TP0 on the 5090). Every extend window of the
target worker (``WEG2-VRAM-PEAK phase=chunk``, and the draft extend that
follows it inside the next ``phase=round``) grew ``reserved`` by more than its
transient in 20 of 42 windows: 02:33:10 transient 621 MiB, reserved +1174 MiB,
card free 619 -- while 983 MiB sat in the cache (reserved 28740 - allocated
27757) that the extend could not reuse (blocks of other size classes / other
streams; ``expandable_segments`` is off on the saver's pools). Only a sleep
returns them. Two awake phases in a row then took an allocator retry
(02:40:00 transient 980, 02:41:14 transient 911, card free 729 / 589) -- the
rc12c death pattern, which ended in 79 retries and an OOM.

The fix is the retry the allocator does anyway, but controlled and early:
before the extend, when ``card_free < floor + booked activation`` (the D
form's planner terms, per rank, from the launcher), ``synchronize`` +
``empty_cache`` hands the unused cached segments back to the card, so the
extend's transient comes from free card bytes instead of fresh segments on
top of an unusable cache. One line per trim:

    WEG2-EXTEND-CACHE-TRIM rank=0 ms=.. released=.. card_free_before=..
      card_free_after=.. threshold=..

COST. Off (no ``SGLANG_WEG2_EXTEND_TRIM_MIB``): nothing -- the threshold is
parsed once per process and the hook returns before any CUDA call. On: one
``cudaMemGetInfo`` per target extend (driver call, no sync); a trim adds a
device sync and one ``cudaFree`` per released segment (the ``ms`` field).

WHAT IT CANNOT DO. ``empty_cache`` returns only fully free segments of the
default pool. Split segments holding one live block and private pools (CUDA
graphs) stay -- ``released`` says what it got. Never under graph capture.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

MARKER = "WEG2-EXTEND-CACHE-TRIM"
MIB = float(1 << 20)

_UNSET = object()
_CACHE = {"thresholds": _UNSET}


def parse_thresholds(text: Optional[str]) -> Optional[List[float]]:
    """``"1791,1300,1300"`` -> per-rank MiB; ``None`` for unset/empty/garbage
    (a malformed value must not trim on a guess)."""
    if text is None or not str(text).strip():
        return None
    try:
        vals = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        logger.warning("%s: unreadable SGLANG_WEG2_EXTEND_TRIM_MIB=%r, trim off", MARKER, text)
        return None
    return vals or None


def thresholds() -> Optional[List[float]]:
    """The process's thresholds, read from the env once."""
    if _CACHE["thresholds"] is _UNSET:
        try:
            from sglang.srt.environ import envs

            _CACHE["thresholds"] = parse_thresholds(envs.SGLANG_WEG2_EXTEND_TRIM_MIB.get())
        except Exception:  # noqa: BLE001 -- a guard never kills a forward
            _CACHE["thresholds"] = None
    return _CACHE["thresholds"]  # type: ignore[return-value]


def reset_for_tests() -> None:
    _CACHE["thresholds"] = _UNSET


def threshold_for(rank: int, values: Optional[Sequence[float]]) -> Optional[float]:
    """A rank's threshold; one value covers every rank."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(values[rank]) if 0 <= rank < len(values) else None


def maybe_trim(cuda, rank: int, threshold_mib: Optional[float],
               clock=time.perf_counter) -> Optional[str]:
    """Trim when the card's free bytes are under ``threshold_mib``. Returns the
    line it logged, ``None`` when it did nothing."""
    if threshold_mib is None:
        return None
    try:
        capturing = bool(cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        capturing = False
    if capturing:
        return None
    try:
        free0, _total = cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s skipped: %s", MARKER, exc)
        return None
    if free0 / MIB >= threshold_mib:
        return None
    r0 = int(cuda.memory_reserved())
    t = clock()
    cuda.synchronize()
    cuda.empty_cache()
    ms = (clock() - t) * 1000.0
    r1 = int(cuda.memory_reserved())
    try:
        free1, _ = cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        free1 = -1
    line = (
        f"{MARKER} rank={rank} ms={ms:.1f} released={(r0 - r1) / MIB:.0f} "
        f"card_free_before={free0 / MIB:.0f} "
        f"card_free_after={'na' if free1 < 0 else f'{free1 / MIB:.0f}'} "
        f"threshold={threshold_mib:.0f}"
    )
    logger.info(line)
    return line


def before_extend(worker, batch) -> Optional[str]:
    """Hook for ``TpModelWorker.forward_batch_generation``: the target
    worker's extend batches only (the draft extend that follows shares the
    allocator and profits from the same trim)."""
    values = thresholds()
    if values is None or batch is None:
        return None
    if getattr(worker, "is_draft_worker", False) and not getattr(worker, "is_phase_flip_tp_stack", False):
        return None
    try:
        mode = batch.forward_mode
        if not mode.is_extend() or mode.is_target_verify():
            return None
        import torch

        rank = int(getattr(worker, "tp_rank", 0) or 0)
        return maybe_trim(torch.cuda, rank, threshold_for(rank, values))
    except Exception as exc:  # noqa: BLE001 -- a guard never kills a forward
        logger.debug("%s skipped: %s", MARKER, exc)
        return None


def launcher_thresholds(floor_mib: Sequence[float], activation_mib: Sequence[float],
                        growth_mib: Optional[Sequence[Optional[float]]] = None) -> str:
    """The env value the launcher writes: floor + booked activation per rank --
    rc12f: floor + max(activation, measured extend growth) where the profile
    measured the growth (``D_EXTEND_GROWTH_MIB``)."""
    g = list(growth_mib) if growth_mib is not None else [None] * len(floor_mib)
    return ",".join(
        str(int(round(float(f) + max(float(a), float(x) if x is not None else 0.0))))
        for f, a, x in zip(floor_mib, activation_mib, g)
    )


#: rc12f: only extends of at least this many rows price the growth record.
GROWTH_MIN_ROWS = 512


def extend_growth_mib(windows) -> Optional[int]:
    """rc12f: the ``D_EXTEND_GROWTH_MIB`` measurement -- max over extend
    windows ``(rows, transient_mib, reserved_start_mib, peak_reserved_mib)``
    with ``rows >= GROWTH_MIN_ROWS`` of ``peak_reserved - reserved_start``.
    A pure function of the windows: no threshold enters it (no ratchet)."""
    best = None
    for rows, transient, start, peak in windows:
        if int(rows) < GROWTH_MIN_ROWS or int(transient) <= 0:
            continue
        g = int(peak) - int(start)
        best = g if best is None else max(best, g)
    return best
