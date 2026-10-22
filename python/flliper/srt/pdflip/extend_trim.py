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
_CACHE = {"thresholds": _UNSET, "rates": _UNSET, "armed": False, "logged_cap": None,
          # Q-694b EXTEND-RATE: measurement switch + its rank list, this rank's
          # measured rate (MiB/row, ratchet up), the open extend's start reading
          "measure": _UNSET, "measure_ranks": _UNSET, "measured": None, "pending": None,
          # LEND-COST (21.10.): the switch (per rank, read once), this rank's measured card cost of a
          # target+draft extend per row class (MiB, ratchet up) and the extend whose draft tail is open
          "lend_cost_on": _UNSET, "lend_cost": None, "lend_cost_tail": None,
          # EXTEND-KORRIDOR (22.10.): this card's corridor line (read once) and the per-row card cost of this
          # rank's own wide extends (LEND-COST reading / rows, rows >= GROWTH_PER_ROW_MIN_ROWS, ratchet up)
          "line": _UNSET, "lend_rate": None,
          # review 22.10. (planner): this rank's START of the wide-extend per-row card cost (profile record
          # D_EXTEND_LEND_RATE_MIB via FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB, read once; vote only)
          "lend_rate_start": _UNSET}


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
    _CACHE["measure"] = _UNSET
    _CACHE["measure_ranks"] = _UNSET
    _CACHE["measured"] = None
    _CACHE["pending"] = None
    _CACHE["lend_cost_on"] = _UNSET
    _CACHE["lend_cost"] = None
    _CACHE["lend_cost_tail"] = None
    _CACHE["line"] = _UNSET
    _CACHE["lend_rate"] = None
    _CACHE["lend_rate_start"] = _UNSET


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
    allocator and profits from the same trim). Q-694b: after the trim, the rate
    measurement opens its reading (only with ``FLLIPER_PDFLIP_EXTEND_RATE_MEASURE``
    on this rank; off = byte-identical)."""
    line = _before_extend_trim(worker, batch)
    rank = getattr(worker, "tp_rank", None)
    if batch is not None and (measure_armed(rank) or lend_cost_armed(rank)):
        measure_open(worker, batch)
    return line


def _before_extend_trim(worker, batch) -> Optional[str]:
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


# --------------------------------------------------------------------------
# EXTEND-KORRIDOR (NF rc33 22.10., desk/nf-rc33-extendkorridor-1022): the vote keeps the corridor LAW.
#
# Metal (NF rc33 D ``..._fbb680d6d3_1022_060647``, TP0 = 5090, 06:17:04-06:17:14): post-trim card 2071 MiB,
# the vote ``floor((2071 - 300) / 0.5871) = 3008`` let the 2705+5-row mixed extend run whole; its reserved grew
# 2024 MiB (live transient 1416, 0.747 MiB/row), the card ended at 47 MiB (HALT 46, CORRIDOR LAW BREACHED 44 MiB
# under the 767 MiB law). Two terms were wrong, and both are fixed here:
#
#   * the LINE: the vote funded down to its private 300 MiB HALT line, not to the corridor law the breach
#     instrument grades against (per card, 767 / 700 / 701 MiB on this rig). :func:`corridor_line_mib` reads the
#     law through the guard's ONE rank-local door (``corridor_guard.corridor_floor_for_current_device`` -- the
#     same derivation ``corridor_trace`` grades with), its verdict floor, never under 300. No reserve on top.
#   * the RATE: TP0 is a start-rate rank (Q-694b EXTEND-RATE is armed on TP1/TP2 only), its start 0.5871 sits
#     under its own card cost (wide extends 0.61-0.75 MiB/row: 2375 rows -> 1788, 2710 -> 2024). The LEND-COST
#     reading -- armed on every D rank with a lend floor -- already measures that cost (target + draft extend);
#     :func:`lend_cost_settle` now also ratchets ``cost / rows`` over the extends of at least
#     ``GROWTH_PER_ROW_MIN_ROWS`` rows, and :func:`effective_rate` takes it into the max (no safety factor).
#
# Torch's cache is NOT added to the funds (the 27B rc29 double count, a4f4b8fec1 -- that change never reached
# this line). The rate is a reserved growth, i.e. a driver-free loss.
#
# REVIEW 22.10. (planner decision on the review's Auflagen): the SHORT-EXTEND price is NOT part of this -- A4 of
# 21.10. stands (d_transient_lend.short_extend_need_mib keeps the 300 MiB HALT line and the rc33 rate, byte for
# byte): every one of the 28 rc33 corridor breaches was a WIDE extend. The wide-extend cost therefore prices the
# VOTE only (:func:`vote_rate`), never :func:`effective_rate` / :func:`chunk_need_mib`. A rank may start the boot
# with a recorded wide-extend cost (``D_EXTEND_LEND_RATE_MIB``, NF TP0 0.7547: rc33 06:23:24, 2629 rows, 1984 MiB),
# so the first wide extend after the boot is priced too (rc33 06:14:15: the start 0.5871 left ~593 MiB).
#
# And what the extend leaves in the cache goes back at once: at 06:17:08 the extend's 2024 MiB stood in torch's
# cache (alloc_cache 2618 MiB) for six seconds of decode at 45 MiB driver free until the NEXT extend's trim
# released 1970 MiB. :func:`lend_cost_settle` (the next scheduler pass, before any lend move) trims when the
# card is under the line -- the bytes go back to the driver, and the lend can hand them to expert rows.
# --------------------------------------------------------------------------

LINE_MARKER = "EXTEND-CORRIDOR-LINE"


def _read_corridor_line() -> "tuple[float, str]":
    """(line MiB, source) -- the guard's derived floor for this card (its verdict floor), else the stated law."""
    try:
        from flliper.srt.managers.corridor_guard import corridor_floor_for_current_device

        fl = corridor_floor_for_current_device()
        return float(fl.verdict_floor_mib), f"{fl.source}:{int(fl.mib)}"
    except Exception:  # noqa: BLE001 -- no derived floor: the stated law, graded like the guard grades it
        pass
    try:
        from flliper.srt.managers.corridor_guard import corridor_law_mib, verdict_floor_for_mib

        law = int(corridor_law_mib())
        return float(verdict_floor_for_mib(law, "STATED-LAW")), f"stated-law:{law}"
    except Exception:  # noqa: BLE001 -- no reader at all: the HALT line
        return float(CHUNKING_FLOOR_MIB), "halt-line"


def corridor_line_mib(rank: Optional[int] = None) -> float:
    """The card line an extend must leave free: this card's corridor law (``max(300, verdict floor)``), read once
    per process and named once. The width vote, the HALT instrument and the after-extend trim keep this ONE line
    (the short-extend price keeps the 300 MiB HALT line, A4 21.10.). No reserve: the law is the corridor."""
    line = _CACHE["line"]
    if line is _UNSET:
        try:
            mib, src = _read_corridor_line()
        except Exception:  # noqa: BLE001
            mib, src = float(CHUNKING_FLOOR_MIB), "halt-line"
        line = _CACHE["line"] = max(float(CHUNKING_FLOOR_MIB), float(mib))
        logger.info(f"{LINE_MARKER} rank={'?' if rank is None else int(rank)} line_mib={line:.0f} source={src} "
                    f"(width vote, HALT and after-extend trim keep this line; the short-extend price keeps 300 (A4); "
                    f"no reserve)")
    return float(line)  # type: ignore[arg-type]


def rows_cap(post_mib: float, rate: float, page_size: int, line_mib: Optional[float] = None) -> int:
    """The widest chunk the card funds: ``floor((post - line) / rate)`` (``line`` = :func:`corridor_line_mib` in the
    vote; the 300 MiB HALT line when given none), down to the page, never below one page (a chunk must make
    progress)."""
    page = max(1, int(page_size or 1))
    if rate <= 0:
        return 1 << 30
    line = float(CHUNKING_FLOOR_MIB if line_mib is None else line_mib)
    rows = int(math.floor((float(post_mib) - line) / float(rate)))
    return max(page, (rows // page) * page)


def chunk_need_mib(rank: int, rows: int, line_mib: Optional[float] = None) -> Optional[float]:
    """S78 KEEP-SHORT: what an extend of ``rows`` rows costs the card by the rc12g
    chunk model: ``line + rate x rows`` MiB with the rc33 rate of this rank
    (:func:`effective_rate`: start or Q-694b x RATE_SAFETY -- NOT the vote's wide-extend
    cost, review 22.10.) and ``line`` the 300 MiB HALT line unless given. ``None`` when this
    rank has no rate (no vote, no price: nothing is guessed)."""
    try:
        rate = threshold_for(int(rank), rates())
        if rate is None or rate <= 0:
            return None
        rate = effective_rate(rate, int(rank))
        line = float(CHUNKING_FLOOR_MIB if line_mib is None else line_mib)
        return line + float(rate) * max(0, int(rows))
    except Exception:  # noqa: BLE001 -- an unpriced extend is priced by nobody
        return None


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
        # Q-694b: the start rate (record / derived) or this rank's measured
        # rate x RATE_SAFETY, whichever is larger; off: the start rate itself
        start_rate = rate
        rate = vote_rate(rate, rank)
        if bool(cuda.is_current_stream_capturing()):
            return None
        if not _CACHE["armed"]:
            if maybe_trim(cuda, rank, threshold_for(rank, thresholds()), clock) is not None:
                _CACHE["armed"] = True
        post, _total = cuda.mem_get_info()
        post_mib = post / MIB
        # EXTEND-KORRIDOR (22.10.): fund the chunk down to the corridor law, not to the 300 MiB HALT line
        line_mib = corridor_line_mib(rank)
        cap = rows_cap(post_mib, rate, page_size, line_mib)
        src_class = None
        if cap < int(configured):
            # CLASS-VOTE (review 22.10.): a width whose row class this rank has MEASURED (LEND-COST, rows <= the
            # class's widest reading) is priced by that reading instead of the linear worst-case rate
            wide = class_rows_cap(post_mib, line_mib, page_size, int(configured), rank)
            if wide is not None and wide[0] > cap:
                cap, src_class = wide
        if cap >= int(configured):
            _CACHE["logged_cap"] = None
            return None
        if cap != _CACHE["logged_cap"]:
            _CACHE["logged_cap"] = cap
            logger.info(
                f"{CHUNKING_MARKER} rank={rank} geplant={int(configured)} cap={cap} "
                f"post={post_mib:.0f} rate={rate:.4f} floor={line_mib:.0f}"
                + ("" if src_class is None else f" class={src_class[0]} class_cost_mib={src_class[1]:.0f}")
                + ("" if not (measure_armed(rank) or lend_rate(rank) is not None) else
                   f" rate_src={rate_source(start_rate, rank)} start_rate={start_rate:.4f}")
            )
        return cap
    except Exception as exc:  # noqa: BLE001 -- a vote that cannot price abstains
        logger.debug("%s skipped: %s", CHUNKING_MARKER, exc)
        return None


# --------------------------------------------------------------------------
# CLASS-VOTE (review 22.10., planner item 3): the linear vote ``floor((post - line) / rate)`` prices every width
# with the WORST per-row cost seen on a wide extend (TP0 rc33: 0.7547 at 2629 rows), but the card cost is not
# linear in the rows: rc33 TP0 extends of 1000-2000 rows cost 0.54-0.63 MiB/row (1956 rows at 89856 cached:
# 1192 MiB; 1944 at 98560: 1186; 2024: 1248), the 2100+-row ones 0.67-0.755. In the rc33 replay the linear vote
# split 59 of 70 wide TP0 extends, 31 of them needlessly (the old run ended above the 767 law); the class vote
# splits 45, letting none of the 28 breaches through.
#
# The LEND-COST table already holds, per row class, the MAXIMUM measured card cost and the WIDEST extend measured
# in it (``lend_cost_mib``: a width is priced only inside its class and only up to that widest reading -- never
# extrapolated). The vote may therefore WIDEN its linear cap to the largest page-aligned width ``w`` (<= the
# configured chunk) with ``line + lend_cost_mib(w) <= post``. It never narrows (the linear cap stays the floor),
# never prices an unmeasured width, and the result is still this rank's vote in the group MIN (R11). Same
# depth caveat as ``lend_rate``: a prefix deeper than any measured one can cost more than the class maximum.
# --------------------------------------------------------------------------


def class_rows_cap(post_mib: float, line_mib: float, page_size: int, configured: int,
                   rank: Optional[int] = None) -> "Optional[tuple]":
    """``(width, (class, cost_mib))`` -- the widest page-aligned width up to ``configured`` whose row class this
    rank measured (LEND-COST, ``rows <= widest``) and whose measured cost leaves ``line`` free; ``None`` when no
    measured width fits or LEND-COST is not armed on this rank. Never raises."""
    try:
        if not lend_cost_armed(rank):
            return None
        table = _CACHE["lend_cost"]
        if not table:
            return None
        page = max(1, int(page_size or 1))
        best = None
        for cls, (_cost, widest) in list(table.items()):
            w = (min(int(widest), int(configured)) // page) * page
            if w <= 0:
                continue
            c = lend_cost_mib(w)   # the class maximum of w's OWN class, None outside a measured reach
            if c is None or float(line_mib) + float(c) > float(post_mib):
                continue
            if best is None or w > best[0]:
                best = (w, (lend_cost_class(w), float(c)))
        return best
    except Exception:  # noqa: BLE001 -- a vote that cannot price abstains from widening
        return None


def launcher_rates(rate_mib: Sequence[Optional[float]]) -> str:
    """The env value the launcher writes; '' when no rank has a rate. A rank
    without a measurement votes nothing (0 is read as 'no rate')."""
    if not rate_mib or all(r is None for r in rate_mib):
        return ""
    return ",".join("0" if r is None else f"{float(r):.4f}" for r in rate_mib)


# --------------------------------------------------------------------------
# Q-694b EXTEND-RATE: the chunk-cap rate is MEASURED per rank at run time.
#
# Q-694 armed the rc12g vote on the 27B flip line from the profile record
# ``D_EXTEND_CAP_PER_ROW_MIB`` -- a number measured on ONE rig for ONE
# checkpoint. Another card mix, another model or a deeper prefix than the
# record's boots saw is then either uncapped (no record: no vote, the y8va OOM
# class) or capped by a number that describes other hardware. So the rate the
# vote uses is now
#
#     effective = max(start, measured x RATE_SAFETY)
#
# ``start`` is the env rate the launcher wrote (the record where the profile
# has one, else :func:`derived_rate_mib` from the model geometry); ``measured``
# is this rank's own maximum, over its target extends of at least
# ``GROWTH_PER_ROW_MIN_ROWS`` rows, of
#
#     max(peak_allocated - allocated_at_start, reserved_after - reserved_at_start) / rows
#
# read around the extend forward (``before_extend`` .. the forward-end hook
# ``vram_family_census.maybe_log_vram_peak``) without resetting the
# allocator's peak counter (PDFLIP-VRAM-PEAK owns those re-bases): the peak term
# counts only when the extend RAISED the counter, else only the reserved
# growth does. Process-local, so every rank keeps its own (a 3080 and a 5090
# differ). Ratchet up only. Rows under the minimum are not priced: a fixed part
# of the transient would inflate their per-row rate, narrow the next chunk,
# inflate again -- the cut must not feed its own measurement. One line per new
# maximum:
#
#     EXTEND-RATE source=measured rank=0 rows=4096 transient_mib=.. reserved_growth_mib=..
#       rate=.. prev=.. effective=.. start=.. safety=1.15
# --------------------------------------------------------------------------

RATE_MARKER = "EXTEND-RATE"
#: the vote's margin over the measured maximum: the record carries the same
#: ratio over the maximum of its own measurement (0.3091 / 0.2690 = 1.149) --
#: a deeper prefix than any seen so far draws more attention workspace
RATE_SAFETY = 1.15


def _ceil4(x: float) -> float:
    return math.ceil(float(x) * 10000.0 - 1e-9) / 10000.0


def measure_ranks() -> Optional[List[int]]:
    """``FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS`` (CSV of TP ranks), read once;
    ``None`` = the measurement applies to every rank."""
    if _CACHE["measure_ranks"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            raw = envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS.get()
            vals = [int(x) for x in str(raw).split(",") if x.strip()] if raw else []
            _CACHE["measure_ranks"] = vals or None
        except Exception:  # noqa: BLE001 -- an instrument never kills a forward
            _CACHE["measure_ranks"] = None
    return _CACHE["measure_ranks"]  # type: ignore[return-value]


def measure_armed(rank: Optional[int] = None) -> bool:
    """``FLLIPER_PDFLIP_EXTEND_RATE_MEASURE`` (read once per process), narrowed to
    ``FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS`` when that names ranks and the
    caller knows its own (NF port: TP0 keeps its recorded rate, TP1/TP2 --
    whose record is null -- measure)."""
    if _CACHE["measure"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            _CACHE["measure"] = bool(envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.get())
        except Exception:  # noqa: BLE001 -- an instrument never kills a forward
            _CACHE["measure"] = False
    if not _CACHE["measure"]:
        return False
    ranks = measure_ranks()
    return ranks is None or rank is None or int(rank) in ranks


def measured_rate() -> Optional[float]:
    """This rank's measured extend rate (MiB/row), ``None`` before the first."""
    return _CACHE["measured"]  # type: ignore[return-value]


def effective_rate(start: float, rank: Optional[int] = None) -> float:
    """The rate the vote uses: ``max(start, measured x RATE_SAFETY)`` while the
    measurement is armed (on this rank) and holds a value, else ``start``.
    Review 22.10.: this is also the short-extend price's rate (:func:`chunk_need_mib`),
    byte-identical to rc33 -- the wide-extend cost enters the VOTE only (:func:`vote_rate`)."""
    m = _CACHE["measured"] if measure_armed(rank) else None
    if m is None:
        return float(start)
    return max(float(start), _ceil4(float(m) * RATE_SAFETY))


def vote_rate(start: float, rank: Optional[int] = None) -> float:
    """EXTEND-KORRIDOR: the width vote's rate = ``max(effective_rate, lend_rate)`` -- the LEND-COST per-row
    card cost of this rank's wide extends (its recorded start and its own readings, no factor) on top of
    the rc33 rate. The short-extend price never reads it (A4 21.10. stands)."""
    r = effective_rate(start, rank)
    lr = lend_rate(rank)
    if lr is not None:
        r = max(r, float(lr))
    return r


def lend_rate_start(rank: Optional[int] = None) -> Optional[float]:
    """Review 22.10.: this rank's recorded START of the wide-extend per-row card cost
    (``FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB``, CSV per rank, "0" = none; the launcher writes it from
    the profile record ``D_EXTEND_LEND_RATE_MIB``), read once. ``None`` = no start (the boot's own reading)."""
    if _CACHE["lend_rate_start"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            _CACHE["lend_rate_start"] = parse_thresholds(envs.FLLIPER_PDFLIP_EXTEND_LEND_RATE_START_MIB.get())
        except Exception:  # noqa: BLE001 -- a guard never kills a pass
            _CACHE["lend_rate_start"] = None
    v = threshold_for(int(rank or 0), _CACHE["lend_rate_start"])  # type: ignore[arg-type]
    return None if v is None or v <= 0 else float(v)


def lend_rate(rank: Optional[int] = None) -> Optional[float]:
    """EXTEND-KORRIDOR: the card cost per row of this rank's wide extends -- ``max(recorded start, own LEND-COST
    readings)`` (:func:`lend_rate_start`, ratchet up) -- ``None`` before any of the two or where LEND-COST is not
    armed on this rank."""
    try:
        if not lend_cost_armed(rank):
            return None
    except Exception:  # noqa: BLE001
        return None
    lr = _CACHE["lend_rate"]
    st = lend_rate_start(rank)
    vals = [float(v) for v in (lr, st) if v is not None]
    return max(vals) if vals else None


def rate_source(start: float, rank: Optional[int] = None) -> str:
    """Which term set :func:`vote_rate` (for the vote's line): ``lend`` / ``measured`` / ``start``."""
    eff = vote_rate(start, rank)
    lr = lend_rate(rank)
    if lr is not None and float(lr) >= eff and eff > float(start):
        return "lend"
    return "measured" if eff > float(start) else "start"


def _alloc_stats(cuda):
    """(peak_allocated, allocated, reserved) in bytes, ONE ``memory_stats()``."""
    st = cuda.memory_stats()
    return (int(st.get("allocated_bytes.all.peak", 0)),
            int(st.get("allocated_bytes.all.current", 0)),
            int(st.get("reserved_bytes.all.current", 0)))


def measure_open(worker, batch, cuda=None) -> None:
    """Start reading of a target extend (after the trim). Never raises."""
    try:
        _CACHE["pending"] = None
        if getattr(worker, "is_draft_worker", False):
            return
        mode = getattr(batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or mode.is_target_verify():
            return
        if cuda is None:
            import torch

            cuda = torch.cuda
        if bool(cuda.is_current_stream_capturing()):
            return
        pa, a, r = _alloc_stats(cuda)
        # LEND-COST A1 (review 21.10.): the counter still holds the decode rounds since the last PDFLIP-VRAM-PEAK
        # record (TP0 rc30: up to 411 MiB) -- an extend that stays under it would read transient 0. Re-base it
        # at the extend's start (``pa_lend`` = the allocation now); where that is impossible the inherited peak
        # stays the BOUND. Q-694b keeps comparing against the raw ``pa`` (its rate byte for byte).
        pa_lend = pa
        rank = getattr(worker, "tp_rank", None)
        if lend_cost_armed(rank):
            from flliper.srt.model_executor import vram_peak_window

            if vram_peak_window.open_extend_window(cuda, int(rank or 0)):
                pa_lend = a
        runner = getattr(worker, "model_runner", None)
        _CACHE["pending"] = (None if runner is None else id(runner), pa, a, r, pa_lend)
    except Exception as exc:  # noqa: BLE001
        _CACHE["pending"] = None
        logger.debug("%s open skipped: %s", RATE_MARKER, exc)


def measure_close(runner, forward_batch, cuda, rank: Optional[int] = None) -> Optional[str]:
    """Forward-end reading of the extend :func:`measure_open` opened; raises
    this rank's measured rate when the extend's per-row cost is a new maximum.
    Returns the line it logged. Never raises."""
    pending = _CACHE["pending"]
    if pending is None:
        return None
    try:
        rate_on = measure_armed(rank)
        # LEND-COST (21.10.): the same reading prices the D lend per row class (every row count)
        cost_on = lend_cost_armed(rank)
        if not rate_on and not cost_on:
            _CACHE["pending"] = None
            return None
        if getattr(runner, "is_draft_worker", False) or getattr(runner, "is_draft_model_runner", False):
            return None
        owner, pa0, a0, r0 = pending[:4]
        pa0_lend = pending[4] if len(pending) > 4 else pa0
        if owner is not None and owner != id(runner):
            return None
        _CACHE["pending"] = None
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or mode.is_target_verify():
            return None
        ids = getattr(forward_batch, "input_ids", None)
        rows = (int(ids.shape[0]) if ids is not None
                else int(getattr(forward_batch, "extend_num_tokens", 0) or 0))
        if rows < GROWTH_PER_ROW_MIN_ROWS and not cost_on:
            return None
        pa1, _a1, r1 = _alloc_stats(cuda)
        # the peak counter is this extend's only if the extend raised it
        transient = (pa1 - a0) if pa1 > pa0 else 0
        growth = max(0, r1 - r0)
        cost = max(transient, growth)
        if cost_on and rows > 0:
            # A1 (review 21.10.): a peak the extend did not raise is not a measured 0 -- its transient is unknown
            # up to the inherited peak, so that bound is booked (0 only where the open re-based the counter)
            lend_transient = (pa1 - a0) if pa1 > pa0_lend else max(0, pa0_lend - a0)
            # the draft extend that follows grows ``reserved`` on top: the tail closes at the next scheduler pass
            _CACHE["lend_cost_tail"] = (rows, int(lend_transient), int(growth), int(r1))
        if not rate_on or rows < GROWTH_PER_ROW_MIN_ROWS:
            return None
        if cost <= 0:
            return None
        rate = _ceil4(cost / MIB / rows)
        prev = _CACHE["measured"]
        if prev is not None and rate <= float(prev):
            return None
        _CACHE["measured"] = rate
        start = threshold_for(int(rank) if rank is not None else 0, rates())
        eff = None if start is None else effective_rate(start, rank)
        line = (
            f"{RATE_MARKER} source=measured rank={'?' if rank is None else int(rank)} rows={rows} "
            f"transient_mib={transient / MIB:.0f} reserved_growth_mib={growth / MIB:.0f} "
            f"rate={rate:.4f} prev={'none' if prev is None else f'{float(prev):.4f}'} "
            f"effective={'none' if eff is None else f'{eff:.4f}'} "
            f"start={'none' if start is None else f'{float(start):.4f}'} safety={RATE_SAFETY}"
        )
        logger.info(line)
        return line
    except Exception as exc:  # noqa: BLE001
        _CACHE["pending"] = None
        logger.debug("%s close skipped: %s", RATE_MARKER, exc)
        return None


# --------------------------------------------------------------------------
# LEND-COST (21.10., nf-rc30-tp0lend-1021): what a D extend of a row class MEASURABLY costs the card,
# so the D-TRANSIENT-LEND short-extend price follows the measurement instead of the start rate.
#
# Metal (NF rc30 D ``..._1021_063252``): TP0 -- the 5090, a start-rate rank (its profile record is
# kept, Q-694b above is armed on TP1/TP2 only) -- priced every short extend with the S78 SUM
# ``767 + 300 + 0.5871 x rows`` (1069 MiB for 5 rows) and gave expert rows back 1223x in 2 h 13 min
# (133.9 s host, each one a decode pause of the group), TP1/TP2 (measured rate, max price) 287 / 277x;
# the target extends of <= 64 rows on TP0 grew the allocator by at most 449 MiB (PDFLIP-VRAM-PEAK).
#
# The reading is Q-694b's own (``measure_open`` after the EXTEND-CACHE-TRIM .. the target forward's
# end hook, BEFORE PDFLIP-VRAM-PEAK re-bases the peak counter):
#
#     forward = max(peak_allocated - allocated_at_start, reserved_after - reserved_at_start)
#
# A1 (review 21.10.): the open re-bases the allocator peak counter first
# (``vram_peak_window.open_extend_window``: the open PDFLIP round window is closed and printed, else the
# counter is folded into the shadow), so ``peak_allocated`` is the extend's own; on a backend without a
# peak reset an extend that does not raise the inherited peak books that peak as its BOUND, never 0.
#
# plus the reserved growth of the draft extend that follows inside the same batch (rc12d / review A
# method: the draft extend's reserved growth after it), closed at the next scheduler pass before any
# lend move (``lend_cost_settle``):
#
#     cost = forward + max(0, reserved_at_next_pass - reserved_after)
#
# Kept per ROW CLASS (``lend_cost_class``: <= 64, then powers of two): the MAXIMUM cost of the class
# and the WIDEST extend measured in it, both ratchet up only, process-local (the amount of a lend is
# rank-local; nothing structural reads it). An extend prices from its class only up to the widest
# measured one (rows <= widest): a class doubles its rows, so a 4000-row extend is never priced by a
# 2100-row reading. A class without a reading, or wider than its widest, prices nothing (``None``):
# the caller keeps its old price. No safety factor, no reserve: the price adds the card's 300 MiB
# HALT line to the measured cost (``d_transient_lend.short_extend_need_mib``; A4 21.10. stands, review
# 22.10.). One line per new
# class maximum:
#
#     EXTEND-LEND-COST rank=0 rows=5 class=64 cost_mib=.. forward_mib=.. transient_mib=..
#       reserved_growth_mib=.. draft_growth_mib=.. prev=.. widest=.. card_free_mib=..
#
# and one HALT line whenever the card ends an extend batch under the line (EXTEND-KORRIDOR 22.10.: the corridor
# law, :func:`corridor_line_mib`; it was the 300 MiB HALT line) -- then the cache is trimmed at once (the instrument of
# the stop criterion: ``lend_open`` says whether lent rows sat on the card through it):
#
#     EXTEND-LEND-COST HALT rank=0 rows=.. class=.. card_free_mib=.. cost_mib=.. lend_open=1 rows_on=..
#
# Armed with ``FLLIPER_PDFLIP_D_LEND_MEASURED_PRICE`` (default ON) on a rank that HAS a lend floor
# (``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB``, written for the D group only); off = byte-identical.
# --------------------------------------------------------------------------

LEND_COST_MARKER = "EXTEND-LEND-COST"
#: the smallest row class: every extend of up to this many rows is one class
LEND_COST_MIN_CLASS = 64


def lend_cost_class(rows: int) -> int:
    """The row class of an extend: ``64`` for <= 64 rows, else the next power of two (65 -> 128)."""
    rows = int(rows)
    if rows <= LEND_COST_MIN_CLASS:
        return LEND_COST_MIN_CLASS
    return 1 << (rows - 1).bit_length()


def lend_cost_armed(rank: Optional[int] = None) -> bool:
    """LEND-COST switch (``FLLIPER_PDFLIP_D_LEND_MEASURED_PRICE``, default ON) AND a lend floor for
    this rank (the D group's ``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB``): read once per process."""
    seen = _CACHE["lend_cost_on"]
    if seen is _UNSET:
        seen = _CACHE["lend_cost_on"] = {}
    key = None if rank is None else int(rank)
    if key not in seen:
        on = False
        try:
            from flliper.srt.environ import envs

            if bool(envs.FLLIPER_PDFLIP_D_LEND_MEASURED_PRICE.get()):
                from flliper.srt.pdflip import d_seat_vram as dsv

                on = dsv.lend_floor_mib_for_rank(int(key or 0)) is not None
        except Exception:  # noqa: BLE001 -- an instrument never kills a forward
            on = False
        seen[key] = on
    return bool(seen[key])


def lend_cost_mib(rows: int) -> Optional[float]:
    """The measured card cost (MiB) of an extend of ``rows`` rows: the maximum of its row class on
    this rank, ``None`` when the class has no reading yet or ``rows`` is wider than the widest extend
    measured in it (nothing is guessed, nothing extrapolated)."""
    table = _CACHE["lend_cost"]
    if not table:
        return None
    v = table.get(lend_cost_class(int(rows)))
    if v is None or int(rows) > int(v[1]):
        return None
    return float(v[0])


def lend_cost_settle(rank: Optional[int] = None, cuda=None, lend_open: bool = False,
                     rows_on: Optional[int] = None) -> Optional[str]:
    """Close the open tail at the next scheduler pass (before any lend move): add the draft extend's
    reserved growth, ratchet the class maximum, name a new maximum and an extend batch that left the card
    under the corridor line (EXTEND-KORRIDOR: and trim the cache the extend left, so the driver gets it back now).
    Ratchets this rank's wide-extend per-row cost (:func:`lend_rate`). Returns the line logged last. Never raises."""
    tail = _CACHE["lend_cost_tail"]
    if tail is None:
        return None
    _CACHE["lend_cost_tail"] = None
    try:
        if cuda is None:
            import torch

            cuda = torch.cuda
        rows, transient, growth, r1 = tail
        _pa, _a, r2 = _alloc_stats(cuda)
        draft = max(0, int(r2) - int(r1))
        forward = max(int(transient), int(growth))
        cost = _ceil_mib(forward + draft)
        try:
            free_mib = int(cuda.mem_get_info()[0] // MIB)
        except Exception:  # noqa: BLE001 -- no reading: no HALT verdict
            free_mib = None
        cls = lend_cost_class(rows)
        table = _CACHE["lend_cost"]
        if table is None:
            table = _CACHE["lend_cost"] = {}
        old = table.get(cls)
        prev = None if old is None else int(old[0])
        widest = int(rows) if old is None else max(int(old[1]), int(rows))
        line = None
        rk = "?" if rank is None else int(rank)
        if prev is None or cost > prev:
            table[cls] = (cost, widest)
            line = (f"{LEND_COST_MARKER} rank={rk} rows={int(rows)} class={cls} cost_mib={cost} "
                    f"forward_mib={_ceil_mib(forward)} transient_mib={_ceil_mib(transient)} "
                    f"reserved_growth_mib={_ceil_mib(growth)} draft_growth_mib={_ceil_mib(draft)} "
                    f"prev={'none' if prev is None else prev} widest={widest} "
                    f"card_free_mib={'na' if free_mib is None else free_mib}")
            logger.info(line)
        elif widest > int(old[1]):
            table[cls] = (prev, widest)  # a wider extend at or under the class maximum: the reach grows, no line
        # EXTEND-KORRIDOR (22.10.): the per-row card cost of a WIDE extend prices the width vote (ratchet up, no
        # factor); narrower extends carry a fixed part that would inflate a per-row rate (rc12g)
        if int(rows) >= GROWTH_PER_ROW_MIN_ROWS and cost > 0:
            lr = _ceil4(float(cost) / float(rows))
            prev_lr = _CACHE["lend_rate"]
            if prev_lr is None or lr > float(prev_lr):
                _CACHE["lend_rate"] = lr
                logger.info(f"{LEND_COST_MARKER} RATE rank={rk} rows={int(rows)} cost_mib={cost} rate={lr:.4f} "
                            f"prev={'none' if prev_lr is None else f'{float(prev_lr):.4f}'} (the width vote uses "
                            f"max(start, measured x {RATE_SAFETY}, this, recorded start); the short-extend price "
                            f"does not)")
        line_mib = corridor_line_mib(rank)
        if free_mib is not None and free_mib < line_mib:
            # the extend's growth stands in torch's cache: give it back to the driver now (the lend moves next and
            # can hand it to expert rows) -- not at the next extend's trim, seconds of decode later
            trim = None
            try:
                trim = maybe_trim(cuda, int(rank or 0), line_mib)
            except Exception as exc:  # noqa: BLE001 -- a trim that cannot run leaves the HALT verdict standing
                logger.debug("%s after-extend trim skipped: %s", LEND_COST_MARKER, exc)
            after = None
            if trim is not None:
                try:
                    after = int(cuda.mem_get_info()[0] // MIB)
                except Exception:  # noqa: BLE001
                    after = None
            line = (f"{LEND_COST_MARKER} HALT rank={rk} rows={int(rows)} class={cls} card_free_mib={free_mib} "
                    f"cost_mib={cost} class_max_mib={table[cls][0]} lend_open={int(bool(lend_open))} "
                    f"rows_on={'-' if rows_on is None else int(rows_on)} trim_after_mib={'-' if after is None else after}"
                    f" -- the extend batch ended under the {line_mib:.0f} MiB line")
            logger.warning(line)
        return line
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a pass
        logger.debug("%s settle skipped: %s", LEND_COST_MARKER, exc)
        return None


def _ceil_mib(nbytes: float) -> int:
    """Bytes -> whole MiB, rounded UP (a price is never under its reading)."""
    return int(math.ceil(float(nbytes) / MIB - 1e-9))


# Q-694b: the start rate where the profile has no record -- the per-row element
# count of ONE layer's live activations (layers run one after another), with
# no TP division: the start cannot see the rank's share, the measurement
# narrows it per rank later.
#: residual, layer input, normed copy, mixer/MLP output, all-reduce in + out
DERIVED_HIDDEN_COPIES = 6
#: gate_up (2 x intermediate) + activation (1 x intermediate)
DERIVED_MLP_COPIES = 3
#: mixer projection output + mixer output before o_proj
DERIVED_MIXER_COPIES = 2
#: routed MoE: the top-k permuted hidden copies per row (dispatch in, expert out)
DERIVED_MOE_HIDDEN_COPIES = 2
#: every element priced at least at fp32: bf16/fp16 kernels upcast norms, GDN
#: chunk states, softmax and quant scales to fp32
DERIVED_MIN_ELEM_BYTES = 4

_DTYPE_BYTES = {"float64": 8, "double": 8, "float32": 4, "float": 4,
                "bfloat16": 2, "float16": 2, "half": 2}


def derived_rate_mib(cfg: dict) -> Optional[float]:
    """Q-694b: a conservative start rate (MiB per extend row) from the model
    geometry alone -- no card, no rig measurement::

        elems/row = 6 x hidden + 3 x intermediate + 2 x mixer_width
                    [+ 2 x top_k x hidden + num_experts   for a routed MoE]
        bytes/row = elems/row x max(4, dtype bytes)

    ``intermediate`` is the widest MLP a row passes (dense
    ``intermediate_size``, or ``moe_intermediate_size x num_experts_per_tok``
    plus a shared expert; 4 x hidden when the config names neither);
    ``mixer_width`` the widest of full attention (q, its output gate under
    ``attn_output_gate``, k, v), linear attention (q, k, v, z, the per-head
    a/b) and hidden. ``None`` without ``hidden_size``: nothing derived, and the
    launcher names that the cap is not armed."""
    try:
        t = cfg.get("text_config") or cfg
        hidden = int(t.get("hidden_size") or 0)
        if hidden <= 0:
            return None
        dense_i = int(t.get("intermediate_size") or 0)
        moe_i = int(t.get("moe_intermediate_size") or 0) * int(t.get("num_experts_per_tok") or 0)
        moe_i += int(t.get("shared_expert_intermediate_size") or 0) if moe_i else 0
        inter = max(dense_i, moe_i) or 4 * hidden
        heads = int(t.get("num_attention_heads") or 0)
        kv_heads = int(t.get("num_key_value_heads") or heads)
        head_dim = int(t.get("head_dim") or (hidden // heads if heads else 0))
        full = heads * head_dim * (2 if t.get("attn_output_gate") else 1) + 2 * kv_heads * head_dim
        lk = int(t.get("linear_num_key_heads") or 0) * int(t.get("linear_key_head_dim") or 0)
        lv_heads = int(t.get("linear_num_value_heads") or 0)
        lv = lv_heads * int(t.get("linear_value_head_dim") or 0)
        linear = 2 * lk + 2 * lv + 2 * lv_heads
        mixer = max(full, linear, hidden)
        # a routed MoE layer also holds the row's top-k permuted copies of
        # hidden (dispatch in, expert out) and the router logits
        top_k = int(t.get("num_experts_per_tok") or 0) if moe_i else 0
        moe_extra = (DERIVED_MOE_HIDDEN_COPIES * top_k * hidden + int(t.get("num_experts") or 0)
                     if top_k else 0)
        dtype = str(t.get("torch_dtype") or t.get("dtype") or cfg.get("torch_dtype")
                    or cfg.get("dtype") or "bfloat16").replace("torch.", "")
        elem = max(DERIVED_MIN_ELEM_BYTES, _DTYPE_BYTES.get(dtype, 2))
        elems = (DERIVED_HIDDEN_COPIES * hidden + DERIVED_MLP_COPIES * inter
                 + DERIVED_MIXER_COPIES * mixer + moe_extra)
        return _ceil4(elems * elem / MIB)
    except Exception:  # noqa: BLE001 -- an unreadable geometry derives nothing
        return None


# --------------------------------------------------------------------------
# NF port 1003 (order 980 item 2): the ranks whose record is NULL.
#
# Next Flash carries D_EXTEND_GROWTH_PER_ROW_MIB = [0.5871, null, null]: only
# TP0 (5090) was priced, so TP1/TP2 (3080) read "0" = no rate = no vote
# (Bericht 780 section A.4: a gap). The two 3080 ranks then never cap the
# extend chunk although they have LESS room than the 5090 (z30x2: card_free
# 16 / 78 MiB under the floors with the old trim threshold). A start rate for a
# null rank is DERIVED, never a rig number written into the code:
#
#   ratio    = rate[anchor] x growth[r] / growth[anchor]
#              the anchor is the first ranked with a recorded rate; growth is
#              the profile's per-rank ``D_EXTEND_GROWTH_MIB`` (what one D extend
#              added to torch reserved on each rank, the same extends)
#   geometry = derived_rate_mib(config.json): the model's own per-row activation
#              bound, no card, no rig. It is the Q-694b formula and it is NOT
#              a bound on the reserved growth NF measures: 0.4622 on the NF
#              geometry against the measured 0.5871 (report 960, -21 %; on the
#              27B geometry it is conservative, 0.4422 against 0.3091)
#   floor    = the recorded rank's own measured rate: no rank of the profile
#              starts under the highest MEASURED value of its record until
#              the formula knows the NF terms (coordinator, report 960) or the
#              3080 ranks are measured (the run-time measurement below only
#              ratchets UP, so the floor is also the 3080 ranks' resting value)
#
# and the start is the LARGEST of the candidates that exist (the conservative
# direction); the run-time measurement above then raises it per rank
# (``FLLIPER_PDFLIP_EXTEND_RATE_MEASURE_RANKS`` names exactly these ranks, so a
# recorded rank keeps its recorded rate byte for byte).
# --------------------------------------------------------------------------


def fill_null_rates(rate_mib: Sequence[Optional[float]],
                    growth_mib: Optional[Sequence[Optional[float]]] = None,
                    geometry_mib: Optional[float] = None,
                    floor_to_record: bool = True) -> "tuple[List[Optional[float]], List[str]]":
    """``(rates, sources)``: ``rate_mib`` with every ``None`` rank replaced by
    the largest of its derivable candidates (``ratio``, ``geometry``, and with
    ``floor_to_record`` the highest recorded rate ``record-floor``), or left
    ``None`` when it has none. ``sources[r]`` is ``record`` for a recorded
    rank, else the winning candidate or ``none``."""
    rates = [None if r is None or float(r) <= 0 else float(r) for r in rate_mib]
    anchor = next((i for i, r in enumerate(rates) if r is not None), None)
    growth = list(growth_mib) if growth_mib is not None and len(growth_mib) == len(rates) else None
    top = max((r for r in rates if r is not None), default=None)
    out: List[Optional[float]] = []
    src: List[str] = []
    for i, r in enumerate(rates):
        if r is not None:
            out.append(r)
            src.append("record")
            continue
        cands = []
        if floor_to_record and top is not None:
            cands.append(("record-floor", top))
        if (anchor is not None and growth is not None
                and growth[i] is not None and growth[anchor] is not None
                and float(growth[i]) > 0 and float(growth[anchor]) > 0):
            cands.append(("ratio", rates[anchor] * float(growth[i]) / float(growth[anchor])))
        if geometry_mib is not None and float(geometry_mib) > 0:
            cands.append(("geometry", float(geometry_mib)))
        if not cands:
            out.append(None)
            src.append("none")
            continue
        name, val = max(cands, key=lambda c: c[1])
        out.append(_ceil4(val))
        src.append(name)
    return out, src
