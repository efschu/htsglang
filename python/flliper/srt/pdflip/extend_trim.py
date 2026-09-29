"""PDFLIP-EXTEND-CACHE-TRIM: empty the caching allocator before a D extend when the card is short (rc12e).

What rc12d showed (27.09., D-TP0 on the 5090). Every extend window of the
target worker (``PDFLIP-VRAM-PEAK phase=chunk``, and the draft extend that
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

    PDFLIP-EXTEND-CACHE-TRIM rank=0 ms=.. released=.. card_free_before=..
      card_free_after=.. threshold=..

COST. Off (no ``FLLIPER_PDFLIP_EXTEND_TRIM_MIB``): nothing -- the threshold is
parsed once per process and the hook returns before any CUDA call. On: one
``cudaMemGetInfo`` per target extend (driver call, no sync); a trim adds a
device sync and one ``cudaFree`` per released segment (the ``ms`` field).

WHAT IT CANNOT DO. ``empty_cache`` returns only fully free segments of the
default pool. Split segments holding one live block and private pools (CUDA
graphs) stay -- ``released`` says what it got. Never under graph capture.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

MARKER = "PDFLIP-EXTEND-CACHE-TRIM"
MIB = float(1 << 20)

_UNSET = object()
_CACHE = {"thresholds": _UNSET, "rates": _UNSET, "armed": False, "logged_cap": None}


def parse_thresholds(text: Optional[str]) -> Optional[List[float]]:
    """``"1791,1300,1300"`` -> per-rank MiB; ``None`` for unset/empty/garbage
    (a malformed value must not trim on a guess)."""
    if text is None or not str(text).strip():
        return None
    try:
        vals = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        logger.warning("%s: unreadable FLLIPER_PDFLIP_EXTEND_TRIM_MIB=%r, trim off", MARKER, text)
        return None
    return vals or None


def thresholds() -> Optional[List[float]]:
    """The process's thresholds, read from the env once."""
    if _CACHE["thresholds"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            _CACHE["thresholds"] = parse_thresholds(envs.FLLIPER_PDFLIP_EXTEND_TRIM_MIB.get())
        except Exception:  # noqa: BLE001 -- a guard never kills a forward
            _CACHE["thresholds"] = None
    return _CACHE["thresholds"]  # type: ignore[return-value]


def rates() -> Optional[List[float]]:
    """rc12g: the process's per-rank extend growth per row (MiB/row), read once."""
    if _CACHE["rates"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            _CACHE["rates"] = parse_thresholds(envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.get())
        except Exception:  # noqa: BLE001 -- a guard never kills a pass
            _CACHE["rates"] = None
    return _CACHE["rates"]  # type: ignore[return-value]


def reset_for_tests() -> None:
    _CACHE["thresholds"] = _UNSET
    _CACHE["rates"] = _UNSET
    _CACHE["armed"] = False
    _CACHE["logged_cap"] = None


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

        # rc12g: an extend is running -- the scheduler may trim again for the next one
        _CACHE["armed"] = False
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


# --------------------------------------------------------------------------
# rc12g PDFLIP-EXTEND-STUECKELUNG: the chunk follows the card, not the other way.
#
# rc12f at the metal (27.09. 04:27:45 / 04:27:50, D-TP0): the trim raised
# card_free to 2385 / 2325 MiB -- empty_cache returns cache, not the untagged
# rest, the draft and the KV fill -- and the next extend chunks (4096 / 4029
# rows) grew reserved by 1992 / 2116 MiB: 393 and then 207 MiB were left, under
# the 300 MiB line. No threshold fixes that: the trim never brings a card back
# above ~2730. What does is the chunk width: before the chunk is formed, cap it
# to what the card holds after the trim,
#
#     rows_cap = floor((card_free_post - 300) / growth_per_row)
#
# rounded down to the page. ``growth_per_row`` is the measured record
# ``D_EXTEND_GROWTH_PER_ROW_MIB`` (maximum of reserved growth / rows). The cap is
# this rank's VOTE in the scheduler's existing packed MIN reduce (#794 corridor
# width, `_local_corridor_width_ceiling`), so every rank cuts to the same width
# and no collective is added. One line per new cap:
#
#     PDFLIP-EXTEND-STUECKELUNG rank=0 geplant=4096 cap=3648 post=2385 rate=0.5625
#
# Off (no ``FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB``, or no rate for this rank):
# no vote, byte-identical. Only while prefill work is pending does it read the
# card; it trims at most once per extend (``armed`` until an extend runs), so a
# queue that waits for seats does not synchronize every decode round.
# --------------------------------------------------------------------------

CHUNKING_MARKER = "PDFLIP-EXTEND-STUECKELUNG"
#: the card line the cap keeps free after the chunk's growth (the operator's HALT line)
CHUNKING_FLOOR_MIB = 300.0
#: a per-row rate is only a per-row quantity where the rows dominate the
#: window: a 603-row extend in rc12b grew the window by 776 MiB (1.29/row, the
#: decode and draft rounds inside the same window). The cap binds only for
#: chunks of ~3000+ rows, so the rate is priced where it binds.
GROWTH_PER_ROW_MIN_ROWS = 2048


def extend_growth_per_row_mib(windows) -> Optional[float]:
    """rc12g: the ``D_EXTEND_GROWTH_PER_ROW_MIB`` measurement -- max over extend
    windows ``(rows, growth_mib)`` with ``rows >= GROWTH_PER_ROW_MIN_ROWS`` of
    ``growth / rows``, rounded UP to 4 decimals. A pure function of the windows:
    a cut chunk has fewer rows AND proportionally less growth, so the rate does
    not move with the cut (fixpoint)."""
    best = None
    for rows, growth in windows:
        rows = int(rows)
        if rows < GROWTH_PER_ROW_MIN_ROWS or float(growth) <= 0:
            continue
        r = float(growth) / rows
        best = r if best is None else max(best, r)
    if best is None:
        return None
    return math.ceil(best * 10000.0 - 1e-9) / 10000.0


def rows_cap(post_mib: float, rate: float, page_size: int) -> int:
    """The widest chunk the card funds: ``floor((post - 300) / rate)``, down to
    the page, never below one page (a chunk must make progress)."""
    page = max(1, int(page_size or 1))
    if rate <= 0:
        return 1 << 30
    rows = int(math.floor((float(post_mib) - CHUNKING_FLOOR_MIB) / float(rate)))
    return max(page, (rows // page) * page)


def width_vote(cuda, rank: int, configured: int, page_size: int, pending: bool,
               clock=time.perf_counter) -> Optional[int]:
    """This rank's vote for the group's chunk width, ``None`` = no vote.

    Called once per scheduler iteration from the packed MIN reduce; must never
    raise. With prefill work pending and a rate for this rank: trim once (if
    the card is under the trim threshold and no trim is armed yet), read the
    card, return ``rows_cap`` when it is narrower than ``configured``."""
    try:
        vals = rates()
        rate = threshold_for(rank, vals)
        if rate is None or rate <= 0 or not pending or int(configured) <= 0:
            return None
        if bool(cuda.is_current_stream_capturing()):
            return None
        if not _CACHE["armed"]:
            if maybe_trim(cuda, rank, threshold_for(rank, thresholds()), clock) is not None:
                _CACHE["armed"] = True
        post, _total = cuda.mem_get_info()
        post_mib = post / MIB
        cap = rows_cap(post_mib, rate, page_size)
        if cap >= int(configured):
            _CACHE["logged_cap"] = None
            return None
        if cap != _CACHE["logged_cap"]:
            _CACHE["logged_cap"] = cap
            logger.info(
                f"{CHUNKING_MARKER} rank={rank} geplant={int(configured)} cap={cap} "
                f"post={post_mib:.0f} rate={rate:.4f} floor={CHUNKING_FLOOR_MIB:.0f}"
            )
        return cap
    except Exception as exc:  # noqa: BLE001 -- a vote that cannot price abstains
        logger.debug("%s skipped: %s", CHUNKING_MARKER, exc)
        return None


def launcher_rates(rate_mib: Sequence[Optional[float]]) -> str:
    """The env value the launcher writes; '' when no rank has a rate. A rank
    without a measurement votes nothing (0 is read as 'no rate')."""
    if not rate_mib or all(r is None for r in rate_mib):
        return ""
    return ",".join("0" if r is None else f"{float(r):.4f}" for r in rate_mib)
