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
import os
import time
from typing import List, Optional, Sequence

import msgspec

logger = logging.getLogger(__name__)

MARKER = "PDFLIP-EXTEND-CACHE-TRIM"
MIB = float(1 << 20)

_UNSET = object()
_CACHE = {"thresholds": _UNSET, "rates": _UNSET, "armed": False, "logged_cap": None, "cap_trimmed": False,
          # Q-694b EXTEND-RATE: measurement switch + its rank list, this rank's
          # measured rate (MiB/row, ratchet up), the open extend's start reading
          "measure": _UNSET, "measure_ranks": _UNSET, "measured": None, "pending": None,
          # Q-710 EXTEND-CAP-FLOOR: the arming switch, the last logged floor line
          "cap_floor": _UNSET, "logged_floor": None,
          # NACHT2 (port of NF rc33 bc9dcc7520): this card's corridor line, read once per process
          "line": _UNSET,
          # VOTE-LINE-SINGLE (24.10.): the measured non-row part per rank (env, read once), and whether
          # this rank's line is the single line (near-OOM + that part) -- set with "line"
          "vote_fixed": _UNSET, "single": False,
          # TRIM-LAW (#201): the static threshold was checked against this rank's own line; "law_line" is the line
          # of the last check (re-checked when the line rises live, VOTE-FIXED-LIVE)
          "law_lifted": False, "law_line": None,
          # ADMIT-TRIM (desk/27b-admit-trim-1024): the width-vote pass counter, the pass whose vote trimmed, the pass
          # whose admission trimmed
          "vote_seq": 0, "vote_trim_seq": -1, "admit_seq": -1,
          # EXTEND-CAP-SOFT (27.10.): the switch (env, read once), this rank's soft measurement (max cost per row over
          # extends >= SOFT_MIN_ROWS, and the widest such extend), the card's effective user reserve (read once), the
          # last logged soft width, whether the soft path already fell back once
          "soft": _UNSET, "soft_measured": None, "soft_rows": 0, "reserve": _UNSET, "logged_soft": None,
          "soft_failed": False}


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
    _live_reset()   # VOTE-FIXED-LIVE state (its own block below)
    _CACHE["thresholds"] = _UNSET
    _CACHE["rates"] = _UNSET
    _CACHE["armed"] = False
    _CACHE["logged_cap"] = None
    _CACHE["cap_trimmed"] = False
    _CACHE["measure"] = _UNSET
    _CACHE["measure_ranks"] = _UNSET
    _CACHE["measured"] = None
    _CACHE["pending"] = None
    _CACHE["cap_floor"] = _UNSET
    _CACHE["logged_floor"] = None
    _CACHE["line"] = _UNSET
    _CACHE["vote_fixed"] = _UNSET
    _CACHE["single"] = False
    _CACHE["law_lifted"] = False
    _CACHE["law_line"] = None
    _CACHE["vote_seq"] = 0
    _CACHE["vote_trim_seq"] = -1
    _CACHE["admit_seq"] = -1
    _CACHE["soft"] = _UNSET
    _CACHE["soft_measured"] = None
    _CACHE["soft_rows"] = 0
    _CACHE["reserve"] = _UNSET
    _CACHE["logged_soft"] = None
    _CACHE["soft_failed"] = False


def threshold_for(rank: int, values: Optional[Sequence[float]]) -> Optional[float]:
    """A rank's threshold; one value covers every rank."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(values[rank]) if 0 <= rank < len(values) else None


def maybe_trim(cuda, rank: int, threshold_mib: Optional[float],
               clock=time.perf_counter, site: str = "") -> Optional[str]:
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
        + (f" site={site}" if site else "")
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
    # EXTEND-CAP-SOFT: a rank with a start rate also reads its own cost per row (extends >= SOFT_MIN_ROWS rows)
    if batch is not None and (measure_armed(rank) or soft_measure_on(rank)):
        measure_open(worker, batch)
    return line


def _batch_rows(batch) -> int:
    """The new rows of an extend batch (``extend_num_tokens``, else the input ids); 0 when unknown."""
    n = getattr(batch, "extend_num_tokens", None)
    if n is not None:
        return int(n or 0)
    ids = getattr(batch, "input_ids", None)
    try:
        return int(ids.shape[0]) if ids is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def _before_extend_trim(worker, batch) -> Optional[str]:
    values = thresholds()
    # VOTE-LINE-SINGLE: a rank voting with the single line trims under its forward's need even without a static
    # threshold (the line prices only what the trim cannot return); vote_fixed() unset = the old early exit
    if batch is None or (values is None and not _CACHE["cap_trimmed"] and vote_fixed() is None):
        return None
    if getattr(worker, "is_draft_worker", False) and not getattr(worker, "is_phase_flip_tp_stack", False):
        return None
    try:
        mode = batch.forward_mode
        if not mode.is_extend() or mode.is_target_verify():
            return None
        # PDFLIP-EXTEND-CAP (29.09.): the vote's cap trim is spent by this extend too
        _CACHE["cap_trimmed"] = False
        rank = int(getattr(worker, "tp_rank", 0) or 0)
        thr = threshold_for(rank, values)
        need = need_trim_threshold(rank, _batch_rows(batch)) if vote_fixed() is not None else None
        if values is None and need is None:
            return None
        if need is not None:
            thr = need if thr is None else max(float(thr), float(need))
        import torch

        # rc12g: an extend is running -- the scheduler may trim again for the next one
        _CACHE["armed"] = False
        line = maybe_trim(torch.cuda, rank, thr)
        if line is not None:
            live_note_trim()   # VOTE-FIXED-LIVE: the pass's forward ran after a trim (the reading's basis)
        return line
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


LINE_MARKER = "EXTEND-CORRIDOR-LINE"


def _read_corridor_line() -> "tuple[float, str]":
    """(line MiB, source) -- the guard's derived floor for THIS card: its TARGET (``CorridorFloor.mib``, the same
    number ``corridor_trace`` measures its margin to). On a MEASURED card that is the measured transient peak of the
    awake group and equals the verdict floor the trace grades a breach against (with VOTE-LINE-SINGLE the vote holds
    near-OOM + the measured fixed part instead, the trace keeps grading this floor) (no band under a measured peak); else
    the stated law. Port of NF rc33 bc9dcc7520 / review 91e56eeec1 (``_read_corridor_line``).

    NACHT2 review 22.10.: the TARGET, not the verdict floor. For an UNMEASURED-FALLBACK / ENV-OVERRIDE / stated card
    the verdict floor is the -20 % band edge (819 at 1024): a vote aimed at it spends the band's whole tolerance on a
    number nobody measured (corridor_guard: the fallback is VERDICT-ONLY and "a mechanism that aims at the edge of its
    own tolerance has none"). Measured cards keep the port's gain (target == verdict floor, 700/701/767 on the 27B D);
    an unmeasured card keeps the pre-port line (the stated law), never a wider one."""
    try:
        from flliper.srt.managers.corridor_guard import corridor_floor_for_current_device

        fl = corridor_floor_for_current_device()
        return float(fl.mib), f"{fl.source}:{int(fl.mib)}"
    except Exception:  # noqa: BLE001 -- no derived floor: the stated law itself
        pass
    try:
        from flliper.srt.managers.corridor_guard import corridor_law_mib

        law = int(corridor_law_mib())
        return float(law), f"stated-law:{law}"
    except Exception as exc:  # noqa: BLE001 -- no reader at all: the HALT line
        # review R5 (#201): the everyday gap of SOLL 1 -- the line is "read" but silently the halt line, so the
        # TRIM-LAW-LIFT-UNCHECKED warning never fires. Named here (once per process: the line is read once); the
        # behaviour is unchanged (the halt line is still the line).
        logger.warning(f"{LINE_FALLBACK_MARKER} line_mib={float(CHUNKING_FLOOR_MIB):.0f} source=halt-line: neither the "
                       f"card's derived corridor floor nor the stated law could be read ({type(exc).__name__}: {exc}) "
                       f"-- this rank's line is the halt line, the TRIM-LAW lift prices the threshold on it "
                       f"(unchanged behaviour, named)")
        return float(CHUNKING_FLOOR_MIB), "halt-line"


# --------------------------------------------------------------------------
# VOTE-LINE-SINGLE (24.10., desk/27b-vote-line-single-1024, root ticket S4 of the VOTE-LINE-BOOK review): ONE honest
# line for the width vote -- the near-OOM edge plus only the posts that rate x rows does NOT already pay.
#
# THE DOUBLE COUNT. On a MEASURED card the NACHT2 line is the corridor floor, and that floor is
# ``phase_footprint-a191a0712717-cc0ac5caf91e.json``: ``allocated peak - post-capture floor`` of a 4096-token PREFILL
# (767 / 701 / 700 MiB on the 27B D) -- the extend activation. ``rate x rows`` prices exactly that activation already
# (D_EXTEND_CAP_PER_ROW_MIB 0.3091 = the deep-prefix maximum of the ALLOCATED transient per row, 1266 MiB at 4096 rows
# >= 701). Kept as line AND priced per row, it was counted twice; with PDFLIP-VOTE-LINE-BOOK (b130e9dab4) the KV pool
# paid for the second copy (delta 472 / 408 / 400 MiB).
#
# WHAT AN EXTEND REALLY NEEDS ABOVE rate x rows, measured (every logged vote paired with its chunk --
# budget_rest.extend_passes; six INT8-drei boots whose vote logged driver free as post, 618 passes with a forward
# trim; report deskq/done/vote-line-single-1024.md):
#
#   post            the vote's driver free (PDFLIP-EXTEND-STUECKELUNG post=, after the vote's trim)
#   admission peak  A = post - card_free_before of the FORWARD trim: the admission/load-back of the pass (Mamba
#                   restore, KV load-back staging of a post-wake handback) -- up to ~510 MiB, in post-wake passes.
#                   Released by the forward trim BEFORE the forward; it does not stack onto the rows and no width
#                   removes it (one 1-row handback carries it): NOT a line post. Its condition post - A >= 400 is NOT
#                   structurally secured (the admission runs before the forward-need trim; only the vote's STATIC trim
#                   precedes it: TP2 800 - 468 = 332 structurally). Measured: INT8 drei min 542 (six boots); int4ar
#                   376 / 390 -- violated, pre-existing, own ticket.
#   forward phase   post - (card free at the chunk's end): torch reserved is monotone between trims, so the chunk's
#                   end is the forward phase's driver-free reading (a boundary read, not a continuous minimum; the
#                   draft extend of the same pass falls into the next window -- A1 checks both on the 100 ms trace).
#                   Above rate x rows it is the admission REST the trim could not return (fragmented segments; it grows
#                   with the handback's request count, bs <= 6) plus the forward's own fixed part: max 238 MiB (boot
#                   _1022_003738 TP1 00:42:45, 6-request handback, 61 rows) -> D_EXTEND_VOTE_FIXED_MIB = 238.
#                   IN-SAMPLE (the max over the very passes the truth test checks): a regression lock, not a bound.
#   the edge        corridor_guard.NEAR_OOM_MIB = 400 (one allocation from OOM): the fixed rest.
#
#   line = NEAR_OOM_MIB + D_EXTEND_VOTE_FIXED_MIB = 400 + 238 = 638 MiB      (was the floor 767 / 701 / 700)
#
# The split holds only while the admission is released before the forward. :func:`need_trim_threshold` makes that
# true by construction: before the forward the rank trims when the card is under ``line + rate x rows`` (the forward's
# own need), not only under the hand-set static threshold (27b.env 1200 / 1100 / 800 -- TP2's 800 let a post-wake
# handback's admission sit in the cache over the forward; the SAME passes trimmed on TP0 / TP1 stay under the record).
# Over the need nothing changes (the card holds the forward anyway).
#
# Named limit (review MX3): under ``line + rate x floor_rows`` driver free the Q-710 cap floor raises the width above
# the card's funding (EXTEND-CAP-FLOOR, pre-existing): there the forward can end under the edge whatever the line.
#
# Only on a MEASURED floor (the double-count case) AND a form of the record's boots (launcher form gate, review M1:
# int4ar / NVFP4 duo measured 223-252 -- the floor line stays there); an unmeasured card keeps its stated law (NACHT2
# rule: never a wider vote on a number nobody measured), and without the env (other profiles, NF forms, other forms)
# the floor line holds, byte for byte. Nothing counted twice, no reserve (keine-korridor-reserve-nie): the edge is the
# law's own near-OOM number, the fixed part a measurement, the rows the rate.
# --------------------------------------------------------------------------

SINGLE_LINE_MARKER = "VOTE-LINE-SINGLE"


def near_oom_mib() -> float:
    """The fixed rest of the single line: ``corridor_guard.NEAR_OOM_MIB`` (one source, never a copy)."""
    from flliper.srt.managers.corridor_guard import NEAR_OOM_MIB

    return float(NEAR_OOM_MIB)


def line_for_floor_mib(floor_mib: float, fixed_mib: Optional[float] = None) -> float:
    """THE vote line, one formula for the rank (:func:`corridor_line_mib`) and the launcher's budget
    (PDFLIP-VOTE-LINE-BOOK, ``launcher.booked_rest_vote_line_deltas``), so the line the budget books and the line
    the vote keeps cannot drift apart.

    ``fixed_mib`` given (VOTE-LINE-SINGLE: a MEASURED floor and the profile's ``D_EXTEND_VOTE_FIXED_MIB``):
    ``max(300, near-OOM 400 + fixed)`` -- the floor is NOT in it (it is the extend activation rate x rows prices).
    Else the NACHT2 line ``max(300, floor)``."""
    if fixed_mib is not None:
        return max(float(CHUNKING_FLOOR_MIB), near_oom_mib() + max(0.0, float(fixed_mib)))
    return max(float(CHUNKING_FLOOR_MIB), float(floor_mib))


def vote_fixed() -> Optional[List[float]]:
    """VOTE-LINE-SINGLE: the per-rank measured non-row part (``FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB``), read once."""
    if _CACHE["vote_fixed"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            _CACHE["vote_fixed"] = parse_thresholds(envs.FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB.get())
        except Exception:  # noqa: BLE001 -- unreadable: the floor line (the old vote)
            _CACHE["vote_fixed"] = None
    return _CACHE["vote_fixed"]  # type: ignore[return-value]


def _source_measured(src: str) -> bool:
    """``corridor_guard.CorridorFloor.measured`` on the line reader's source text (``MEASURED-D:701``)."""
    return str(src or "").startswith("MEASURED-")


def corridor_line_mib(rank: Optional[int] = None) -> float:
    """The card line the chunk's growth must leave free, read once per process and named once
    (``EXTEND-CORRIDOR-LINE``); no reserve on top.

    VOTE-LINE-SINGLE (24.10.): on a MEASURED card with the record ``D_EXTEND_VOTE_FIXED_MIB`` (env
    ``FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB``) the line is ``near-OOM 400 + fixed`` (:func:`line_for_floor_mib`); the
    measured floor -- the 4096-row activation ``rate x rows`` prices -- is no longer held a second time.

    NACHT2 22.10. (port of NF rc33 bc9dcc7520) otherwise: THIS card's corridor law (``max(300, floor target)``).
    27B INT8 rc29b metal 21.10. (786fe038dc): D TP1 (3080) breached the law 17x (minimum 3-7 MiB) when the vote funded
    a 4096-row chunk down to 300 MiB; the line became the stated law ``corridor_law_mib()`` (1024 by default), then
    (NACHT2) the card's DERIVED corridor floor through the guard's one rank-local door (round 3c, boot 1022_015348,
    a4f4b8fec1: 700 / 701 / 767 against a vote at 1024 that mostly ran the 256-row floor chunk). Funding stays the
    DRIVER alone (a4f4b8fec1), the rate stays the measured one."""
    base = _CACHE["line_base"]
    if base is _UNSET:
        try:
            mib, src = _read_corridor_line()
        except Exception:  # noqa: BLE001
            mib, src = float(CHUNKING_FLOOR_MIB), "halt-line"
        fixed = None
        try:
            fx = threshold_for(int(rank or 0), vote_fixed())
            if fx is not None and fx >= 0 and _source_measured(src):
                fixed = float(fx)
        except Exception:  # noqa: BLE001 -- unpriced: the floor line
            fixed = None
        base = _CACHE["line_base"] = (float(mib), src, fixed)
        line = _CACHE["line"] = line_for_floor_mib(mib, fixed)
        _CACHE["single"] = fixed is not None
        if fixed is not None:
            logger.info(f"{LINE_MARKER} rank={'?' if rank is None else int(rank)} line_mib={line:.0f} "
                        f"source={SINGLE_LINE_MARKER}:near-oom {near_oom_mib():.0f} + fixed {fixed:.0f} "
                        f"(D_EXTEND_VOTE_FIXED_MIB: admission rest + forward fixed part above rate x rows, measured) "
                        f"replaces floor {float(mib):.0f} ({src}: the 4096-row activation rate x rows already prices); "
                        f"the forward trims under line + rate x rows; funds = driver free; no reserve"
                        + ("" if not live_on() else f"; {LIVE_MARKER}: the rank's own readings and the group max "
                           f"raise the fixed part (never lower it)"))
        else:
            logger.info(f"{LINE_MARKER} rank={'?' if rank is None else int(rank)} line_mib={line:.0f} source={src} "
                        f"(the width vote keeps this card's corridor law; funds = driver free; no reserve)")
    mib, _src, fixed = base  # type: ignore[misc]
    if fixed is None:
        return float(_CACHE["line"])  # type: ignore[arg-type]
    # VOTE-FIXED-LIVE (24.10.): the seed raised by this rank's measured readings and the group max (raise only)
    return line_for_floor_mib(mib, fixed_effective(rank, fixed))


def single_line_active(rank: Optional[int] = None) -> bool:
    """VOTE-LINE-SINGLE: this rank's vote line is ``near-OOM + measured fixed part`` (reads the line once)."""
    corridor_line_mib(rank)
    return bool(_CACHE["single"])


# --------------------------------------------------------------------------
# VOTE-FIXED-LIVE (24.10., desk/27b-vote-fixed-live-1024; pdflip/vote_fixed_live.py has the design): the rank measures
# F itself -- post (the vote's driver free) - end (driver free at the target extend's end, lowered by the draft forward
# of the same pass) - rate x rows over its trimmed, non-first passes -- and the line becomes ``400 + max(seed, own F,
# group F)``. The seed is the launcher's FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB (static record of the form, raised by the
# form's persisted readings; written only once the form is measured -- rule 3, cold start = floor line). Within the
# boot F only rises (rule 1). F is priced at the START rate (the vote never prices below it: the larger, safe F; one
# rate for every rank's payload). Every hook is a no-op without a rate on this rank (no vote = nothing to pair) and
# never raises.
# --------------------------------------------------------------------------

LIVE_MARKER = "VOTE-FIXED-LIVE"

#: VOTE-FIXED-LIVE state, kept in ``_CACHE`` but declared here (one block, merge-friendly): the line's base (floor
#: MiB, source, seed F or None), read once; the switch; this rank's meter (vote_fixed_live.Meter); the group max of F
#: from the packed reduce (ratchet up); the last F named; the store target (root, form, boot, cards, proc) or None;
#: whether a store write failed already; whether the exit flush is registered
_LIVE_DEFAULTS = {"line_base": _UNSET, "live_on": _UNSET, "live": None, "live_group": 0.0, "live_logged": None,
                  "live_store": _UNSET, "live_store_err": False, "live_atexit": False}


def _live_reset() -> None:
    _CACHE.update(_LIVE_DEFAULTS)


_live_reset()


def live_on() -> bool:
    """``FLLIPER_PDFLIP_VOTE_FIXED_LIVE`` (default ON), read once per process."""
    on = _CACHE["live_on"]
    if on is _UNSET:
        try:
            from flliper.srt.pdflip import vote_fixed_live as _vfl

            on = bool(_vfl.switch_on())
        except Exception:  # noqa: BLE001 -- unreadable switch: the seed alone (VOTE-LINE-SINGLE)
            on = False
        _CACHE["live_on"] = on
    return bool(on)


def _live_meter(rank: Optional[int] = None):
    m = _CACHE["live"]
    if m is None:
        from flliper.srt.pdflip import vote_fixed_live as _vfl

        m = _CACHE["live"] = _vfl.Meter(rank=int(rank or 0))
        if not _CACHE["live_atexit"]:
            # review NIT: a boot that ends between two persist marks keeps its last counts (best effort; a SIGKILL
            # skips it -- the marks and every rise of F were written already)
            import atexit

            atexit.register(_live_flush)
            _CACHE["live_atexit"] = True
    return m


def _live_flush() -> None:
    """Exit flush: write the meter's doc if it counted passes since the last write. Never raises."""
    try:
        m = _CACHE["live"]
        if m is not None and m.counted > m.persisted_counted:
            _live_write(m)
    except Exception:  # noqa: BLE001
        return


def _live_start_rate(rank: Optional[int]) -> Optional[float]:
    try:
        r = threshold_for(int(rank or 0), rates())
        return None if r is None or r <= 0 else float(r)
    except Exception:  # noqa: BLE001
        return None


def live_fixed_local(rank: Optional[int] = None) -> Optional[float]:
    """This rank's own measured F at its start rate (MiB), ``None`` without a counted reading or with the switch off."""
    if not live_on() or _CACHE["live"] is None:
        return None
    rate = _live_start_rate(rank if rank is not None else _CACHE["live"].rank)
    if rate is None:
        return None
    return _CACHE["live"].fixed_mib(rate)


def live_fixed_payload() -> int:
    """The packed-reduce slot (MIN): ``-ceil(own F)``, 0 without a reading -- contributed by every rank on every
    iteration, so the payload width never depends on a rank's state. Never raises."""
    try:
        f = live_fixed_local()
        return 0 if f is None else -int(math.ceil(float(f) - 1e-9))
    except Exception:  # noqa: BLE001
        return 0


def live_fixed_group(reduced_min: int) -> None:
    """Read back the reduced slot: the group max of F (ratchet up; rule 4). Never raises."""
    try:
        g = max(0.0, -float(reduced_min))
        if g > float(_CACHE["live_group"]):
            _CACHE["live_group"] = g
    except Exception:  # noqa: BLE001
        return


def fixed_effective(rank: Optional[int], seed: float) -> float:
    """The single line's fixed part: ``max(seed, own F, group F)`` with the switch on, else the seed (9a25171a59)."""
    f = float(seed)
    if not live_on():
        return f
    own = live_fixed_local(rank)
    eff = max(f, float(_CACHE["live_group"]), 0.0 if own is None else math.ceil(float(own) - 1e-9))
    if eff > f and (_CACHE["live_logged"] is None or eff > float(_CACHE["live_logged"])):
        _CACHE["live_logged"] = eff
        m = _CACHE["live"]
        logger.info(f"{LIVE_MARKER} rank={'?' if rank is None else int(rank)} fixed_mib={eff:.0f} "
                    f"line_mib={near_oom_mib() + eff:.0f} seed={f:.0f} own={'-' if own is None else f'{own:.0f}'} "
                    f"group={float(_CACHE['live_group']):.0f} counted={0 if m is None else m.counted} "
                    f"worst={None if m is None else m.worst} (rule 1: the measured fixed part raises the line, never "
                    f"lowers it)")
    return eff


def _live_store():
    st = _CACHE["live_store"]
    if st is _UNSET:
        st = None
        try:
            from flliper.srt.pdflip import vote_fixed_live as _vfl

            form = os.environ.get(_vfl.ENV_FORM, "").strip()
            root = os.environ.get(_vfl.ENV_DIR, "").strip()
            if form and root:
                # review M1: the boot name the launcher hands over (full boot stem) -- else this process's core cwd
                # (<EVIDENCE_DIR>/core/<boot stem>, launcher.core_cwd_for); the file also carries pid-start, so a
                # relaunch under one tag or a rank restart never writes over another process's readings
                boot = os.environ.get(_vfl.ENV_BOOT, "").strip() or os.path.basename(os.getcwd())
                st = (root, form, boot,
                      [x.strip() for x in os.environ.get(_vfl.ENV_CARDS, "").split(",") if x.strip()],
                      f"{os.getpid()}-{int(time.time() * 1000)}")
        except Exception:  # noqa: BLE001 -- no store: the meter still measures
            st = None
        _CACHE["live_store"] = st
    return st


def _live_write(m) -> bool:
    """Write this rank's doc (merged, its own file). False without a store target or a rate."""
    from flliper.srt.pdflip import vote_fixed_live as _vfl

    st = _live_store()
    rate = _live_start_rate(m.rank)
    if st is None or rate is None:
        return False
    root, form, boot, cards, proc = st
    card = cards[m.rank] if 0 <= m.rank < len(cards) else "?"
    _vfl.write_doc(_vfl.doc_path(root, form, boot, m.rank, proc),
                   m.snapshot(rate, form=form, boot=boot, card=card, source="rank", proc=proc))
    m.persisted_counted = m.counted
    return True


def _live_after(rank: Optional[int], msg: Optional[str]) -> None:
    """A counted reading: persist when this rank's F rose, a reading went to the quarantine, or a milestone count is
    reached. Never raises."""
    if msg is None:
        return
    m = _CACHE["live"]
    try:
        from flliper.srt.pdflip import vote_fixed_live as _vfl

        rate = _live_start_rate(m.rank)
        f = None if rate is None else m.fixed_mib(rate)
        quarantined = msg.startswith("QUARANTINED")
        prev = m.persisted_f
        rose = f is not None and (prev is None or f > prev + 0.5)
        if quarantined:
            # review M2: loud -- it raises THIS boot's line (OOM-safe) but never seeds a later boot
            logger.warning(f"{LIVE_MARKER} rank={m.rank} reading {msg}: raises this boot's line only, persisted as "
                           f"quarantined (no seed; drop with python -m flliper.srt.pdflip.vote_fixed_live --drop)")
        if not (rose or quarantined or m.counted in (1, _vfl.MIN_PASSES_PER_BOOT)
                or m.counted % _vfl.PERSIST_EVERY == 0):
            return
        if rose and not quarantined:
            logger.info(f"{LIVE_MARKER} rank={m.rank} reading {msg} own_fixed_mib={f:.0f} counted={m.counted} "
                        f"untrimmed={m.untrimmed} (post - end - start rate x rows over trimmed, non-first passes)")
        _live_write(m)
        m.persisted_f = f
    except Exception as exc:  # noqa: BLE001 -- a store write never kills a pass; named once
        if not _CACHE["live_store_err"]:
            _CACHE["live_store_err"] = True
            logger.warning(f"{LIVE_MARKER} store write failed ({type(exc).__name__}: {exc}); the meter keeps "
                           f"measuring in this boot, the next boot does not see these readings")


def live_note_vote(rank: int, post_mib: float, rate: float) -> None:
    """width_vote read the card: a pending pass (the previous reading commits). Never raises."""
    try:
        if not live_on():
            return
        _live_after(rank, _live_meter(rank).on_vote(float(post_mib), float(rate)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s vote skipped: %s", LIVE_MARKER, exc)


def live_note_trim() -> None:
    """The forward trim fired for the pending pass. Never raises."""
    try:
        if live_on() and _CACHE["live"] is not None:
            _CACHE["live"].on_trim()
    except Exception:  # noqa: BLE001
        return


def live_note_window(phase: str, rows: int, free_mib: float) -> None:
    """vram_peak_window closed a window: ``chunk`` pairs the pending vote; ``round`` commits; ``flip`` / ``idle``
    commit and unpair. Never raises."""
    try:
        if not live_on() or _CACHE["live"] is None:
            return
        m = _CACHE["live"]
        if str(phase) == "chunk":
            _live_after(m.rank, m.on_chunk(int(rows), float(free_mib)))
        else:
            _live_after(m.rank, m.on_window(str(phase)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s window skipped: %s", LIVE_MARKER, exc)


def live_note_target_decode() -> None:
    """A target decode / verify forward ended: commit, and a pending vote is no longer this pass's. Never raises."""
    try:
        if live_on() and _CACHE["live"] is not None:
            m = _CACHE["live"]
            _live_after(m.rank, m.on_target_decode())
    except Exception:  # noqa: BLE001
        return


def live_note_draft(cuda) -> None:
    """A draft forward ended while a reading is open (the draft extend of the same pass): its driver free lowers the
    reading's end. One ``cudaMemGetInfo`` (no sync), only with an open reading. Never raises."""
    try:
        if not live_on() or _CACHE["live"] is None or _CACHE["live"]._open is None:
            return
        free, _total = cuda.mem_get_info()
        _CACHE["live"].on_draft(float(free) / MIB)
    except Exception:  # noqa: BLE001
        return


def need_trim_threshold(rank: int, rows: int) -> Optional[float]:
    """VOTE-LINE-SINGLE: the driver free the coming forward needs -- ``line + vote rate x rows`` -- under which the
    rank trims before it, so the admission's released cache never sits under the forward (the single line prices
    only what the trim cannot return). ``None`` = not the single line on this rank, no rate, or no rows (the static
    threshold alone, byte for byte)."""
    try:
        if int(rows) <= 0:
            return None
        start = threshold_for(int(rank), rates())
        if start is None or start <= 0:
            return None
        if not single_line_active(int(rank)):
            return None
        return corridor_line_mib(int(rank)) + float(effective_rate(float(start), int(rank))) * int(rows)
    except Exception:  # noqa: BLE001 -- unpriced need: the static threshold
        return None


def rows_cap(post_mib: float, rate: float, page_size: int, line_mib: Optional[float] = None) -> int:
    """The widest chunk the card funds: ``floor((post - line) / rate)`` (``line`` = the corridor law, see
    :func:`corridor_line_mib`; the 300 MiB halt line when given none), down to the page, never below one page (a chunk
    must make progress)."""
    page = max(1, int(page_size or 1))
    if rate <= 0:
        return 1 << 30
    line = float(CHUNKING_FLOOR_MIB if line_mib is None else line_mib)
    rows = int(math.floor((float(post_mib) - line) / float(rate)))
    return max(page, (rows // page) * page)


def chunk_need_mib(rank: int, rows: int) -> Optional[float]:
    """S78 KEEP-SHORT: what an extend of ``rows`` rows costs the card by the rc12g
    chunk model -- the inverse of :func:`rows_cap`: ``300 + rate x rows`` MiB with
    the rate the width vote uses on this rank (start rate or the measured one x
    RATE_SAFETY, :func:`effective_rate`). ``None`` when this rank has no rate (no
    vote, no price: nothing is guessed)."""
    try:
        rate = threshold_for(int(rank), rates())
        if rate is None or rate <= 0:
            return None
        rate = effective_rate(rate, int(rank))
        return CHUNKING_FLOOR_MIB + float(rate) * max(0, int(rows))
    except Exception:  # noqa: BLE001 -- an unpriced extend is priced by nobody
        return None


# --------------------------------------------------------------------------
# EXTEND-CAP-SOFT (27.10., user rule 24.10.: the ~1 GB line is a PLAN target, not a run-time gate; only the physical
# limit -- near-OOM 400 MiB, an allocation that does not fit -- and the user's reserve are hard).
#
# THE DEFECT (boot dkr27bbar1drei10272118_fdd27a84d4_1027_211839, D group TP3 INT8, configured 16384; the 1841 boot the
# same with 8192 -> 512): the vote below was ``floor((post - line) / rate)`` with line = 1024 (UNMEASURED-FALLBACK, the
# stated law) and rate = the START rate 0.3091 for ever (``rate_src=start``: the measurement only counts extends of
# >= 2048 rows, which a 1024-row chunk never reaches). TP2 (3080): post 1062-1184 MiB -> cap 122-516, TP1 cap 853-1247;
# the Q-710 floor (configured / CAP_FLOOR_DIV = 1024) raised it, ``GROUP-NARROWED 16384 -> 1024``. What those chunks
# really cost (PDFLIP-VRAM-PEAK phase=chunk TP2, rows=1024): transient 138-139 MiB = 0.136 MiB/row, card free
# 904 -> 846 (-58 MiB), i.e. less than half the start rate; the card ended every chunk at ~850 MiB, far above the
# 400 MiB edge. The line, not the card, bound.
#
# THE RULE, per rank, off with ``FLLIPER_PDFLIP_EXTEND_SOFT_LINE=0`` (default ON):
#
#   line  = max(user reserve, min(old line, near-OOM 400 + F))       F = max(238, seed, live F, group F)
#   rate  = max(soft base rate, 1.15 x the >= 2048-row maximum)      soft base = max(1.25 x own measured cost per
#           row (rows >= 512), 0.5 x start), never over the start rate; no measurement yet = the start rate
#   cap   = floor((post - line) / rate), page aligned; with a measured soft rate at most 2 x the widest extend this
#           rank measured (an extrapolation guard) but never under what the OLD rule funded
#
# * near-OOM 400 is ``corridor_guard.NEAR_OOM_MIB``; F is the measured non-row part of an extend above rate x rows
#   (admission rest the trim cannot return + the forward's fixed part: ``D_EXTEND_VOTE_FIXED_MIB`` = 238 over six
#   INT8-drei boots, the same number the single line (VOTE-LINE-SINGLE) already books). With a launcher seed or the
#   live meter the line follows them (raise only). Boot 1027_211839 TP2, 1024 rows: vote post 1108 -> chunk end 846 =
#   262 MiB = admission 204 + forward 58; priced soft 0.17 x 1024 + 238 = 412. The meter's own F is priced at the START
#   rate (vote_fixed_live), so it does not rise in the soft regime; the 238 default is the margin there (named).
# * the user reserve is the rank's EFFECTIVE one (``corridor_guard.reserve_for_card``, the one door the floor reader
#   uses): the launcher masks it to 0 on a card whose measured awake rest is booked (``launcher.effective_reserve_by_card``:
#   "the reserve was never free VRAM - the awake D consumed it"; this boot: stated 1800/1400/1400, effective 0/0/0 on D,
#   the booked rest in ``budgets_from_dc`` replaces floor + reserve). Where it is NOT masked the chunk may not take the
#   card under it. Never added on top of the edge: the reserve is a free-column floor, as near-OOM is.
# * the reserved-but-not-allocated cache is NOT credited as funds: the 22.10. metal (D TP1, 2162 rows: reserved +534 MiB
#   for a 282 MiB transient with 860 MiB cache present) refuted it, and the vote already reads ``post`` after its trim.
#   The measured cost per row is max(allocated transient, reserved growth) without netting the cache, so the cache that
#   does serve a chunk (driver free fell 58 MiB for a 139 MiB transient) is the unlisted margin, not a counted one.
# * the Q-710 floor (CAP_FLOOR_DIV) stays the crawl guard and only ever RAISES the soft width (:func:`floored_width`).
#
# One line per new width (rank-local, like GROUP-NARROWED):
#
#   EXTEND-CAP-SOFT rank=2 free=892 hard_floor=638 rate_used=0.1700 rate_src=measured-soft cap=1494 width=1494 ...
# --------------------------------------------------------------------------

SOFT_MARKER = "EXTEND-CAP-SOFT"
#: the soft rate's margin over this rank's measured cost per row (the >= 2048-row path keeps its 1.15): the soft
#: measurement starts at 512 rows, where the per-row figure carries a share of the forward's fixed part (conservative)
#: but not yet the deep-prefix growth of a wide chunk
SOFT_RATE_SAFETY = 1.25
#: the soft base rate never goes under this fraction of the start rate (one boot's measurement is one boot's depth)
SOFT_RATE_FLOOR_FRAC = 0.5
#: only extends of at least this many rows enter the soft measurement (= GROWTH_MIN_ROWS, rc12f)
SOFT_MIN_ROWS = GROWTH_MIN_ROWS
#: a measured soft rate holds up to this multiple of the widest extend measured (linear extrapolation guard)
SOFT_RAMP_FACTOR = 2
#: F without a seed: ``D_EXTEND_VOTE_FIXED_MIB``, the measured max of admission rest + forward fixed part (INT8 drei)
SOFT_FIXED_DEFAULT_MIB = 238.0


class SoftTerms(msgspec.Struct, frozen=True):
    """One rank's soft vote terms: the rate and line the cap is computed with, and what they came from."""

    rate: float
    rate_src: str
    line: float
    reserve: float
    fixed: float
    old_line: float
    cap_old: int


def soft_on() -> bool:
    """``FLLIPER_PDFLIP_EXTEND_SOFT_LINE`` (default ON), read once per process."""
    if _CACHE["soft"] is _UNSET:
        from flliper.srt.environ import envs

        _CACHE["soft"] = bool(envs.FLLIPER_PDFLIP_EXTEND_SOFT_LINE.get())
    return bool(_CACHE["soft"])


def soft_measure_on(rank: Optional[int]) -> bool:
    """This rank measures its cost per row for the soft vote: the switch is on and the rank has a start rate."""
    if rank is None or not soft_on():
        return False
    start = threshold_for(int(rank), rates())
    return start is not None and start > 0


def soft_base_rate(start: float, measured: Optional[float]) -> "tuple[float, str]":
    """``(rate, source)``: ``max(1.25 x measured, 0.5 x start)`` capped at ``start`` once the rank measured a cost per
    row (``measured-soft``), else the start rate (``start``). Pure."""
    if measured is None:
        return float(start), "start"
    soft = max(_ceil4(float(measured) * SOFT_RATE_SAFETY), _ceil4(float(start) * SOFT_RATE_FLOOR_FRAC))
    if soft >= float(start):
        return float(start), "start"
    return soft, "measured-soft"


def soft_line_mib(old_line: float, reserve_mib: float, fixed_mib: float) -> float:
    """The soft vote line: the user reserve, or the near-OOM edge plus the fixed part when that is lower than the old
    line -- never the old line when it is higher (that is the plan target the rule softens). Pure."""
    edge = near_oom_mib() + max(0.0, float(fixed_mib))
    return max(float(reserve_mib), min(float(old_line), edge))


def soft_ramp_bound(cap_old: int, widest_rows: int, page_size: int) -> int:
    """The widest chunk a MEASURED soft rate may extrapolate to: ``SOFT_RAMP_FACTOR x`` the widest extend measured,
    page aligned, never under what the old rule funded (the soft rule never narrows below it). Pure."""
    page = max(1, int(page_size or 1))
    return max(int(cap_old), (int(widest_rows) * SOFT_RAMP_FACTOR // page) * page)


def _user_reserve_mib() -> float:
    """This card's EFFECTIVE user reserve (launcher-masked to 0 on a booked-rest card), read once per process."""
    if _CACHE["reserve"] is _UNSET:
        from flliper.srt.managers.corridor_guard import reserve_for_card

        try:
            from flliper.srt.registry import nvml as registry_nvml

            uuid = registry_nvml.current_device_uuid() or ""
        except Exception as exc:  # noqa: BLE001 -- no NVML: the floor reader's own answer (no card, no mapped reserve)
            logger.debug("%s: current device uuid unavailable for the reserve: %s", SOFT_MARKER, exc)
            uuid = ""
        # the one function every rank reader of the floor uses (#236 SOLL-2): the launcher's EFFECTIVE reserve
        _CACHE["reserve"] = float(reserve_for_card(uuid, 0)[0])
    return float(_CACHE["reserve"])  # type: ignore[arg-type]


def soft_terms(rank: int, post_mib: float, start_rate: float, page_size: int, old_line: float) -> SoftTerms:
    """The soft vote's rate and line for this rank (reads the once-per-process reserve, the live F and the soft and
    >= 2048-row measurements)."""
    seed = threshold_for(int(rank), vote_fixed())
    fixed = fixed_effective(rank, max(SOFT_FIXED_DEFAULT_MIB, 0.0 if seed is None else float(seed)))
    reserve = _user_reserve_mib()
    base, src = soft_base_rate(start_rate, _CACHE["soft_measured"])
    rate = effective_rate(base, rank)
    if rate > base:
        src = "measured"
    return SoftTerms(rate=rate, rate_src=src, line=soft_line_mib(old_line, reserve, fixed), reserve=reserve,
                     fixed=fixed, old_line=float(old_line),
                     cap_old=rows_cap(post_mib, effective_rate(start_rate, rank), page_size, old_line))


def soft_cap(post_mib: float, page_size: int, terms: SoftTerms) -> int:
    """``rows_cap`` on the soft line and rate; a measured soft rate holds up to :func:`soft_ramp_bound`."""
    cap = rows_cap(post_mib, terms.rate, page_size, terms.line)
    if terms.rate_src != "measured-soft":
        return cap
    return min(cap, soft_ramp_bound(terms.cap_old, int(_CACHE["soft_rows"]), page_size))


def _soft_terms_or_none(rank: int, post_mib: float, start_rate: float, page_size: int,
                        old_line: float) -> Optional[SoftTerms]:
    """The soft terms, or ``None`` = the old rule: switch off, or the soft path could not be priced (named once, at
    WARNING -- an unpriced soft vote must fall back to the old line, not abstain from the vote)."""
    if not soft_on():
        return None
    try:
        return soft_terms(rank, post_mib, start_rate, page_size, old_line)
    except Exception as exc:  # noqa: BLE001 -- fall back to the old, narrower rule
        if not _CACHE["soft_failed"]:
            _CACHE["soft_failed"] = True
            logger.warning(f"{SOFT_MARKER} rank={rank} could not price the soft line ({type(exc).__name__}: {exc}); "
                           f"this rank votes with the old line and rate")
        return None


def _log_soft(rank: int, terms: SoftTerms, post_mib: float, cap: int, width: int, configured: int) -> None:
    """One line per new soft width of this rank (edge-triggered like GROUP-NARROWED)."""
    if width == _CACHE["logged_soft"]:
        return
    _CACHE["logged_soft"] = width
    measured = _CACHE["soft_measured"]
    logger.info(
        f"{SOFT_MARKER} rank={rank} free={post_mib:.0f} hard_floor={terms.line:.0f} rate_used={terms.rate:.4f} "
        f"rate_src={terms.rate_src} cap={cap} width={width} configured={configured} reserve={terms.reserve:.0f} "
        f"fixed={terms.fixed:.0f} old_line={terms.old_line:.0f} cap_old={terms.cap_old} "
        f"soft_measured={'none' if measured is None else f'{float(measured):.4f}'} soft_rows={int(_CACHE['soft_rows'])}"
    )


def _note_soft_measure(rank: Optional[int], rows: int, rate: float, transient: int, growth: int) -> Optional[str]:
    """One extend of >= SOFT_MIN_ROWS rows: the widest one bounds the ramp, the highest cost per row is this rank's
    soft measurement (ratchet up). Returns the line it logged."""
    prev = _CACHE["soft_measured"]
    if rows > int(_CACHE["soft_rows"]):
        _CACHE["soft_rows"] = rows
    if prev is not None and rate <= float(prev):
        return None
    _CACHE["soft_measured"] = rate
    line = (f"{RATE_MARKER} source=measured-soft rank={'?' if rank is None else int(rank)} rows={rows} "
            f"transient_mib={transient / MIB:.0f} reserved_growth_mib={growth / MIB:.0f} rate={rate:.4f} "
            f"prev={'none' if prev is None else f'{float(prev):.4f}'} soft_rows={int(_CACHE['soft_rows'])} "
            f"safety={SOFT_RATE_SAFETY}")
    logger.info(line)
    return line


def width_vote(cuda, rank: int, configured: int, page_size: int, pending: bool,
               clock=time.perf_counter) -> Optional[int]:
    """This rank's vote for the group's chunk width, ``None`` = no vote.

    Called once per scheduler iteration from the packed MIN reduce; must never
    raise. With prefill work pending and a rate for this rank: trim once (if
    the card is under the trim threshold and no trim is armed yet), read the
    card, return ``rows_cap`` when it is narrower than ``configured``."""
    # ADMIT-TRIM: one vote = one pass (before_admission keys on it); counted before any abstention
    _CACHE["vote_seq"] = int(_CACHE["vote_seq"]) + 1
    try:
        vals = rates()
        rate = threshold_for(rank, vals)
        if rate is None or rate <= 0 or not pending or int(configured) <= 0:
            return None
        # Q-694b: the start rate (record / derived) or this rank's measured
        # rate x RATE_SAFETY, whichever is larger; off: the start rate itself
        start_rate = rate
        rate = effective_rate(rate, rank)
        if bool(cuda.is_current_stream_capturing()):
            return None
        # TRIM-LAW (#201): the static threshold never under this card's own line + floor chunk (once per process)
        thr = lift_threshold_to_law(rank, start_rate, int(configured), page_size)
        if not _CACHE["armed"]:
            if maybe_trim(cuda, rank, thr, clock) is not None:
                _CACHE["armed"] = True
                _CACHE["vote_trim_seq"] = _CACHE["vote_seq"]
        post, _total = cuda.mem_get_info()
        post_mib = post / MIB
        # PDFLIP-EXTEND-CAP: under the P0 torch cache cap the allocator's own line
        # binds before the card does (p0-nopin 17:24:06Z: card_free 134 MiB,
        # the cap refused 40 MiB). What the chunk may grow is then
        # cap - reserved; the cache above the live tensors is emptied once per
        # extend first -- torch would do it at the line anyway, only too late
        # for a chunk that is already formed.
        torch_cap = _torch_cap_mib(rank)
        cap_post = None
        if torch_cap is not None:
            cap_post = torch_cap - int(cuda.memory_reserved()) / MIB
            if (rows_cap(min(post_mib, cap_post), rate, page_size) < int(configured)
                    and not _CACHE["cap_trimmed"]):
                cuda.synchronize()
                cuda.empty_cache()
                _CACHE["cap_trimmed"] = True
                post_mib = cuda.mem_get_info()[0] / MIB
                cap_post = torch_cap - int(cuda.memory_reserved()) / MIB
            _live_driver_post = post_mib   # VOTE-FIXED-LIVE (review S2): the meter reads the DRIVER post
            post_mib = min(post_mib, cap_post)
        # 27B INT8 night 22.10. (rounds 1+2, D TP1 3080, driver-free minimum 359.8 / 371.8 MiB under a 1024 MiB line): the
        # chunk is funded by the DRIVER's free bytes alone. The rate is the measured RESERVED growth per row
        # (EXTEND-RATE: reserved_after - reserved_at_start, read AFTER the trim), and for this process a reserved growth
        # IS a driver-free loss -- whatever torch's own cache could serve is already netted out of that rate. Adding the
        # cache on top (rc29 121e4bf3f2) counted it twice: at 23:43:54Z the vote read driver 882 + cache 860 = 1724 MiB,
        # funded 2263 rows, and the 2162-row extend grew reserved by 534 MiB (driver 882 -> 372) while its live
        # transient was 282 MiB -- not one MiB came out of the 860 MiB cache. That cache is the trim's remainder: free
        # blocks inside segments that still hold live blocks (alloc_cache bound 853.7 MiB across every trim), which
        # empty_cache cannot return and the extend's activations do not fit. All 80 TP1 extends >= 1024 rows of both
        # rounds: driver drop <= 0.3091 x rows (max 0.253 MiB/row) with 807-1175 MiB of cache present, so the rate holds
        # and the cache funded nothing. The cache is read for the line only (no sync).
        # The rc29 hang (driver 19.8 MiB, vote 63, Q-710 refused) stays closed by its own fix: under the central decider
        # (SOLL R7) the floor never refuses, D prefills at the narrowed width (cap_floor_crawls_under_soll_r7).
        try:
            cache_free_mib = max(0.0, (float(cuda.memory_reserved()) - float(cuda.memory_allocated())) / MIB)
        except Exception:  # noqa: BLE001 -- the line only
            cache_free_mib = 0.0
        # VOTE-FIXED-LIVE: this read opens the pass the meter pairs with the next chunk (the previous reading commits
        # first, so a raised F is in the line below)
        # review S2: under the torch cache cap the vote prices the capped post, the meter the driver post (its end is
        # the driver free; a capped post would make the reading smaller by the gap)
        live_note_vote(rank, post_mib if cap_post is None else _live_driver_post, rate)
        line_mib = corridor_line_mib(rank)
        # EXTEND-CAP-SOFT: the hard floor and the measured rate instead of the plan line and the start rate
        soft = _soft_terms_or_none(rank, post_mib, start_rate, page_size, line_mib)
        if soft is None:
            cap = rows_cap(post_mib, rate, page_size, line_mib)
        else:
            rate, line_mib = soft.rate, soft.line
            cap = soft_cap(post_mib, page_size, soft)
        if cap >= int(configured):
            _CACHE["logged_cap"] = None
            _CACHE["logged_floor"] = None
            _CACHE["logged_soft"] = None
            return None
        if cap != _CACHE["logged_cap"]:
            _CACHE["logged_cap"] = cap
            logger.info(
                f"{CHUNKING_MARKER} rank={rank} geplant={int(configured)} cap={cap} "
                f"post={post_mib:.0f} cache_free={cache_free_mib:.0f} funds=driver rate={rate:.4f} floor={line_mib:.0f}"
                + ("" if cap_post is None else f" torch_cap={torch_cap:.0f} cap_post={cap_post:.0f}")
                + (f" rate_src={soft.rate_src} start_rate={start_rate:.4f}" if soft is not None else
                   "" if not measure_armed(rank) else
                   f" rate_src={'measured' if rate > start_rate else 'start'} start_rate={start_rate:.4f}")
            )
        width = _apply_cap_floor(rank, cap, post_mib, rate, int(configured), page_size) if cap_floor_armed() else cap
        if soft is not None:
            _log_soft(rank, soft, post_mib, cap, width, int(configured))
        return width
    except Exception as exc:  # noqa: BLE001 -- a vote that cannot price abstains
        logger.debug("%s skipped: %s", CHUNKING_MARKER, exc)
        return None


# --------------------------------------------------------------------------
# ADMIT-TRIM (desk/27b-admit-trim-1024, review M3 24.10.): the trim right before the admission when the vote of this
# pass did not trim.
#
# The vote trims once and then stays ``armed`` until an extend runs (``_before_extend_trim``). When the admission is
# REFUSED (D-ADMIT-REFUSED), no extend runs, ``armed`` stays set, the following votes do not trim, decode rounds
# refill the allocator cache -- and the handback that is finally admitted lands on an untrimmed card. int4ar
# ``_1023_223241`` 23:19:34: after 568 refusals in 174 s TP1 admitted at card_free_before=32 MiB, the forward trim then
# returned 630 MiB (TP0 222, released 578); 127 int4ar passes ran their admission without a vote trim, 10 of them
# under the near-OOM edge.
#
# The scheduler calls :func:`before_admission` where a request has passed the X gate and is about to be added (its
# Mamba restore and KV load-back allocate in ``add_one_req``). Once per pass and only when this pass's vote did not
# trim, the rank trims under its static threshold -- the same ``maybe_trim`` as the vote, logged with ``site=admit``.
# Rank-local memory care (SOLL R11): no verdict, no width, no collective moves. No static threshold = nothing, byte
# for byte. Cost: one ``cudaMemGetInfo`` per admitting pass, a sync + ``empty_cache`` only when the card is short.
# --------------------------------------------------------------------------

ADMIT_SITE = "admit"


def before_admission(cuda, rank: Optional[int]) -> Optional[str]:
    """Trim before this pass's admission when this pass's vote did not trim (``armed`` left over from a pass whose
    admission was refused). Once per pass; ``None`` when nothing was trimmed. Never raises."""
    try:
        if rank is None:
            return None
        seq = int(_CACHE["vote_seq"])
        if _CACHE["admit_seq"] == seq or _CACHE["vote_trim_seq"] == seq:
            return None
        _CACHE["admit_seq"] = seq
        thr = threshold_for(int(rank), thresholds())
        if thr is None:
            return None
        return maybe_trim(cuda, int(rank), thr, site=ADMIT_SITE)
    except Exception as exc:  # noqa: BLE001 -- a guard never kills an admission
        logger.debug("%s admit skipped: %s", MARKER, exc)
        return None


# --------------------------------------------------------------------------
# Q-710 EXTEND-CAP-FLOOR: the vote never crawls.
#
# INT8 y8vb (03.10. 19:34-19:51Z, D group on the flip line): 3905 target
# extends of ONE token per rank in 17 minutes of serving (76 requests; six
# of them 190-1006 forwards each), ~152 ms of group wall each, against a
# 45-58 ms decode round. Root: the rc12g vote above is
# ``floor((card_free_post - 300) / rate)`` and :func:`rows_cap` never goes
# below one page. The D-TP1 3080 held 72-290 MiB free in the D phase (CORRIDOR
# verdict BELOW all boot long), so its vote was 1 row, the MIN reduce cut the
# whole group to 1 (``#794 GROUP-NARROWED ... from 4096 to 1`` x11), and a
# 5-token hand-off tail took five forwards, a 1546-token SHORT 860. An extend
# is eager (``cuda graph: False``): 129 tp.all_reduce + 16 dcp.all_gather per
# forward make it latency-bound, the same ~150 ms at one row as at a hundred.
# While it crawled, prefill ran before decode (mixed chunk off): five running
# streams got no token for 133 s and 147 s.
#
# The 300 MiB line is the operator's HALT line for the chunk's growth; it is
# not an allocation limit. Under the flag the vote therefore has a floor
# chunk, derived from the run-time configuration and the card, never a
# constant of one rig:
#
#     floor_rows = chunked_prefill_size / CAP_FLOOR_DIV   (page aligned)
#     width      = max(cap, min(floor_rows, floor(card_free_post / rate)))
#
# ``card_free_post / rate`` is what the whole free card funds at the priced
# rate (measured maximum x 1.15, or the start): the floor chunk is only ever
# raised to what physically fits, so the raise cannot turn into the OOM of
# y8va. A card that cannot fund even that keeps the narrow vote -- and the
# X gate (:func:`refuses_over_width`) then refuses a fresh D-direct request
# longer than the group width, so the front routes it through P instead of
# crawling. Hand-off tails (a few rows) always fit one chunk.
#
# One line per new width:
#
#     EXTEND-CAP-FLOOR rank=1 cap=1 width=256 floor=256 post=108 rate=0.3091 configured=4096
#
# Off (no ``FLLIPER_PDFLIP_EXTEND_CAP_FLOOR``): nothing here runs, byte-identical.
# --------------------------------------------------------------------------

CAP_FLOOR_MARKER = "EXTEND-CAP-FLOOR"
#: the floor chunk is this fraction of the configured width: the width below
#: which an eager extend forward is dominated by its fixed per-forward latency
CAP_FLOOR_DIV = 16


def cap_floor_armed() -> bool:
    """``FLLIPER_PDFLIP_EXTEND_CAP_FLOOR``, read once per process."""
    if _CACHE["cap_floor"] is _UNSET:
        try:
            from flliper.srt.environ import envs

            _CACHE["cap_floor"] = bool(envs.FLLIPER_PDFLIP_EXTEND_CAP_FLOOR.get())
        except Exception:  # noqa: BLE001 -- a guard never kills a pass
            _CACHE["cap_floor"] = False
    return bool(_CACHE["cap_floor"])


def cap_floor_crawls_under_soll_r7() -> bool:
    """27B NACHT 21.10. (SOLL R7): True while the central decider (``FLLIPER_PDFLIP_DECIDER``, SOLL v2) runs -- then a
    request the X gate admitted (u <= X) is never refused by the floor: P takes no flip for it (R7), so the refusal
    deadlocked D (27B INT8 rc29 00:26:49Z). D prefills it at the narrowed group width instead. Launch env, read on
    every rank alike (rank-uniform); off = the Q-710 refusal as before."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_DECIDER.get())
    except Exception:  # noqa: BLE001 -- unknown = the refusal as before
        return False


def floor_rows(configured: int, page_size: int) -> int:
    """The floor chunk: ``configured / CAP_FLOOR_DIV`` rows, page aligned, at least one page."""
    page = max(1, int(page_size or 1))
    return max(page, (int(configured) // CAP_FLOOR_DIV // page) * page)


def flip_trim_thresholds(rates_text: str, configured: int, page_size: int, line_mib) -> str:
    """27B int4ar 23.10. (D TP2 2x3080, boot ..._a45112428e_1023_140032): the trim threshold the flip arm writes
    when the profile names none -- per rank ``ceil(line + floor_rows(configured) x rate)``, i.e. the driver free a
    card needs for the vote to fund at least the floor chunk above the corridor line. Under it the trim hands the
    fully free segments of the allocator cache back to the driver before the extend (``maybe_trim``), so the vote
    reads what the card really has. A rank without a rate (``0``) gets ``0`` = never trims.

    TRIM-LAW (#201, 24.10.): ``line_mib`` is a scalar (every rank) or one line PER RANK -- the line of that rank's
    own card (launcher :func:`d_card_vote_lines`, the same line the rank's vote keeps). A per-rank vector whose length
    does not match the rates is never zipped short: every rank then gets the HIGHEST of the lines (a threshold under a
    card's line is the OOM direction, one over it only trims earlier).

    Why the flip arm needs it: the vote funds from DRIVER free alone (a4f4b8fec1), so without a trim the cache that
    decode and extend leave behind is never returned and the vote starves. Metal: the int4ar profile (derived from
    the duo profile, which never carried the INT8 ``27b.env`` line ``--env-d FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,
    1100,800``) ran 0 trims; D TP1 sat at 40-170 MiB driver free from 14:11Z on, reserved 18.4 GiB against 17.2 GiB
    allocated (~1.2 GiB idle cache), the vote was 1 row and the floor raised it to 256 -- or to 128-135 at 40 MiB."""
    vals = [x.strip() for x in str(rates_text or "").split(",") if x.strip()]
    if not vals:
        return ""
    if isinstance(line_mib, (int, float)):
        lines = [float(line_mib)] * len(vals)
    else:
        lv = [float(x) for x in line_mib]
        if not lv:
            raise ValueError("flip_trim_thresholds: an empty line vector prices no rank")
        lines = lv if len(lv) == len(vals) else [max(lv)] * len(vals)
    rows = floor_rows(configured, page_size)
    out = []
    for v, ln in zip(vals, lines):
        try:
            r = float(v)
        except ValueError:
            r = 0.0
        out.append("0" if r <= 0 else str(int(math.ceil(ln + rows * r))))
    return ",".join(out)


# --------------------------------------------------------------------------
# TRIM-LAW (#201, Unify-RISK (c), 24.10.): the static trim threshold never sits under this rank's OWN line.
#
# The flip arm's threshold is ``ceil(line + floor_rows x rate)`` -- the driver free at which the vote funds at least
# the floor chunk above the line the vote keeps (:func:`corridor_line_mib`). The launcher now prices it with each D
# card's own line (``launcher.d_card_vote_lines``); before it priced every rank with the stated law 1024. The launcher
# reads the published floor pointer BEFORE the D process exists, the rank reads it at its first vote -- after a D
# process whose own footprint exists republished it -- so the two can still name different lines. Over the rank's
# line a threshold only trims earlier; UNDER it the card sits between ``threshold`` and ``line + floor chunk`` with an
# untrimmed cache, the vote funds less than the floor chunk above its line, the Q-710 floor raises it to what the
# whole free card funds, and the forward ends under the card's law (the danger direction of the ticket).
#
# So the rank lifts its own entry once, at its first vote (where the configured width and page are known), to
# ``ceil(corridor_line_mib(rank) + floor_rows(configured, page) x start rate)`` -- the launcher's formula on the
# rank's own terms -- and says so (``TRIM-LAW-LIFT``). Never lowered (a higher hand value, e.g. the INT8 profile's
# ``1200,1100,800``, stays byte for byte), never armed where there is none (no env value, ``0`` = never trims, no
# rate): nothing in this process trims that did not trim before, it only trims no LATER than its own line requires.
# The lift writes into the parsed threshold list itself, so every reader of :func:`thresholds` (the vote's trim, the
# forward trim, an admission trim) reads one value. Rank-local memory care: no verdict, no width rule, no collective
# (SOLL R11 untouched; the width stays the group's MIN). No reserve: line, floor chunk and rate are the vote's own.
# --------------------------------------------------------------------------

LAW_LIFT_MARKER = "TRIM-LAW-LIFT"
#: review SOLL 1: the lift check could not run (the line was unreadable) -- WARNING, so metal sees the gap
LAW_LIFT_UNCHECKED_MARKER = "TRIM-LAW-LIFT-UNCHECKED"
#: review R4 (#201): the live re-lift (the line rose after the first check, VOTE-FIXED-LIVE) and its no-op twin
LAW_RELIFT_MARKER = "TRIM-LAW-RELIFT"
LAW_RELIFT_HOLDS_MARKER = "TRIM-LAW-RELIFT-HOLDS"
#: review R5 (#201): the line reader fell back to the halt line without a word (named now, behaviour unchanged)
LINE_FALLBACK_MARKER = "TRIM-LAW-LINE-FALLBACK"


def law_bound_threshold(rank: int, start_rate: float, configured: int, page_size: int) -> Optional[float]:
    """The lowest static threshold this rank's own line allows: ``ceil(corridor_line_mib(rank) +
    floor_rows(configured, page_size) x start_rate)``; ``None`` without a rate or a width. Pure in its inputs
    but for the once-per-process line read."""
    if start_rate is None or float(start_rate) <= 0 or int(configured) <= 0:
        return None
    return float(math.ceil(corridor_line_mib(int(rank)) + floor_rows(configured, page_size) * float(start_rate)))


def lift_threshold_to_law(rank: int, start_rate: float, configured: int, page_size: int) -> Optional[float]:
    """Raise this rank's entry of the parsed static threshold to :func:`law_bound_threshold` when it is under it, log
    the lift; returns the threshold in force for the rank afterwards (``None`` = no static threshold). Never raises
    (an unreadable line lifts nothing).

    Checked at the first vote, and AGAIN whenever the rank's line is higher than at the last check (review R4, formal
    OOM direction of VOTE-FIXED-LIVE: the line is ``400 + fixed_effective`` and the fixed part rises live, so the
    once-per-process check priced a line the card no longer keeps): same rule, raise only -- never lowered, no
    collective, rank-local memory care (R11 untouched). Named ``TRIM-LAW-RELIFT`` (lifted) / ``TRIM-LAW-RELIFT-HOLDS``
    (the threshold already covers the higher line); the unchanged line costs one cached read per vote."""
    first = not _CACHE["law_lifted"]
    _CACHE["law_lifted"] = True
    vals = thresholds()
    thr = threshold_for(int(rank), vals)
    if thr is None or thr <= 0:
        return thr
    try:
        line_now = float(corridor_line_mib(int(rank)))
    except Exception as exc:  # noqa: BLE001 -- no line, no lift (the launcher's value stands)
        if first:
            # review SOLL 1: WARNING, named -- metal check "TRIM-LAW-LIFT soll 0" must also see a check that never ran
            logger.warning(f"{LAW_LIFT_UNCHECKED_MARKER} rank={int(rank)} threshold={thr:.0f} stays: this card's line "
                           f"is unreadable ({type(exc).__name__}: {exc}) -- no lift, the launcher's value stands "
                           f"unchecked")
        return thr
    last = _CACHE["law_line"]
    if not first and last is not None and line_now <= float(last):
        return thr  # the line did not rise since the last check: the check stands
    _CACHE["law_line"] = line_now
    relift = last is not None  # a later check of a line that rose (review R4)
    try:
        need = law_bound_threshold(int(rank), start_rate, configured, page_size)
    except Exception as exc:  # noqa: BLE001 -- no bound, no lift (the launcher's value stands)
        logger.warning(f"{LAW_LIFT_UNCHECKED_MARKER} rank={int(rank)} threshold={thr:.0f} stays: this card's line is "
                       f"unreadable ({type(exc).__name__}: {exc}) -- no lift, the launcher's value stands unchecked")
        return thr
    if need is None or thr >= need:
        if relift:
            logger.info(f"{LAW_RELIFT_HOLDS_MARKER} rank={int(rank)} threshold={thr:.0f} holds: the line rose "
                        f"{float(last):.0f} -> {line_now:.0f} live and the threshold still covers line + floor chunk "
                        f"({'-' if need is None else f'{need:.0f}'})")
        return thr
    new = list(vals or [])
    if len(new) == 1:
        new = [float(need)]  # one process = one rank: a single value covers only this process
    elif 0 <= int(rank) < len(new):
        new[int(rank)] = float(need)
    else:
        return thr
    _CACHE["thresholds"] = new
    logger.info(f"{LAW_RELIFT_MARKER if relift else LAW_LIFT_MARKER} rank={int(rank)} threshold={thr:.0f} -> "
                f"{need:.0f} (= ceil(line {line_now:.0f} + floor chunk {floor_rows(configured, page_size)} x rate "
                f"{float(start_rate):.4f})"
                + (f"; the line rose {float(last):.0f} -> {line_now:.0f} live (VOTE-FIXED-LIVE) after the last check"
                   if relift else "; the launcher priced a line under this card's own")
                + "; under it the vote could not fund the floor chunk above the card's law with the cache untrimmed; "
                  "raise only, never lowered, no reserve)")
    return float(need)


def floored_width(cap: int, post_mib: float, rate: float, configured: int, page_size: int) -> int:
    """``max(cap, min(floor_rows, floor(post / rate)))``, page aligned: the vote
    with its floor. A pure function of the card reading, the priced rate and
    the configured width."""
    page = max(1, int(page_size or 1))
    fl = floor_rows(configured, page)
    if int(cap) >= fl or rate <= 0:
        return int(cap)
    phys = (int(math.floor(max(0.0, float(post_mib)) / float(rate))) // page) * page
    return max(int(cap), min(fl, phys))


def _apply_cap_floor(rank: int, cap: int, post_mib: float, rate: float, configured: int,
                     page_size: int) -> int:
    width = floored_width(cap, post_mib, rate, configured, page_size)
    if width != cap and width != _CACHE["logged_floor"]:
        _CACHE["logged_floor"] = width
        logger.info(
            f"{CAP_FLOOR_MARKER} rank={rank} cap={cap} width={width} "
            f"floor={floor_rows(configured, page_size)} post={post_mib:.0f} rate={rate:.4f} "
            f"configured={configured} (the line-respecting cap is under the floor chunk; "
            f"the vote is raised to what the free card funds at the priced rate)"
        )
    return width


def refuses_over_width(group_width: Optional[int], uncached: int, configured: int,
                       page_size: int, already_admitted: bool) -> bool:
    """The X gate's EXTEND-CAP-FLOOR verdict for one request: refuse (the front
    routes it through P) a FRESH D-direct request longer than the group's chunk
    width while that width is under the floor chunk -- the card cannot fund even
    a floor chunk, so the request would crawl. Every term is the group's (the
    MIN-reduced width, the group-priced extent, a per-rid admit mark set on the
    same pass on every rank). Never for a request this gate already admitted
    (a chunk continuation), and never for a tail that fits one chunk.
    ``False`` when armed off, without a group width, or without a configured width."""
    if not cap_floor_armed() or already_admitted:
        return False
    if group_width is None or int(configured) <= 0:
        return False
    w = int(group_width)
    if w <= 0 or w >= floor_rows(configured, page_size):
        return False
    return int(uncached) > w


def _torch_cap_mib(rank: int) -> Optional[float]:
    """This rank's P0 torch cache cap in MiB while it is armed, else ``None``
    (no cap: the vote reads the card alone, byte-identical)."""
    from flliper.srt.pdflip import torch_cache_cap as _tcc

    if not _tcc.enabled():
        return None
    return _tcc.cap_mib_for(int(rank))


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
    measurement is armed (on this rank) and holds a value, else ``start``."""
    m = _CACHE["measured"] if measure_armed(rank) else None
    if m is None:
        return float(start)
    return max(float(start), _ceil4(float(m) * RATE_SAFETY))


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
        runner = getattr(worker, "model_runner", None)
        _CACHE["pending"] = (None if runner is None else id(runner), pa, a, r)
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
        armed = measure_armed(rank)
        soft = soft_measure_on(rank)
        if not (armed or soft):
            _CACHE["pending"] = None
            return None
        if getattr(runner, "is_draft_worker", False) or getattr(runner, "is_draft_model_runner", False):
            return None
        owner, pa0, a0, r0 = pending
        if owner is not None and owner != id(runner):
            return None
        _CACHE["pending"] = None
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or mode.is_target_verify():
            return None
        ids = getattr(forward_batch, "input_ids", None)
        rows = (int(ids.shape[0]) if ids is not None
                else int(getattr(forward_batch, "extend_num_tokens", 0) or 0))
        if rows < (SOFT_MIN_ROWS if soft else GROWTH_PER_ROW_MIN_ROWS):
            return None
        pa1, _a1, r1 = _alloc_stats(cuda)
        # the peak counter is this extend's only if the extend raised it
        transient = (pa1 - a0) if pa1 > pa0 else 0
        growth = max(0, r1 - r0)
        cost = max(transient, growth)
        if cost <= 0:
            return None
        rate = _ceil4(cost / MIB / rows)
        soft_line = _note_soft_measure(rank, rows, rate, transient, growth) if soft else None
        if not armed or rows < GROWTH_PER_ROW_MIN_ROWS:
            return soft_line
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
