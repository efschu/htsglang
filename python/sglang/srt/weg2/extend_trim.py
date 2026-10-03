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
import math
import time
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

MARKER = "WEG2-EXTEND-CACHE-TRIM"
MIB = float(1 << 20)

_UNSET = object()
_CACHE = {"thresholds": _UNSET, "rates": _UNSET, "armed": False, "logged_cap": None, "cap_trimmed": False,
          # Q-694b EXTEND-RATE: measurement switch, this rank's measured rate
          # (MiB/row, ratchet up), the open extend's start reading
          "measure": _UNSET, "measured": None, "pending": None}


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


def rates() -> Optional[List[float]]:
    """rc12g: the process's per-rank extend growth per row (MiB/row), read once."""
    if _CACHE["rates"] is _UNSET:
        try:
            from sglang.srt.environ import envs

            _CACHE["rates"] = parse_thresholds(envs.SGLANG_WEG2_EXTEND_GROWTH_PER_ROW_MIB.get())
        except Exception:  # noqa: BLE001 -- a guard never kills a pass
            _CACHE["rates"] = None
    return _CACHE["rates"]  # type: ignore[return-value]


def reset_for_tests() -> None:
    _CACHE["thresholds"] = _UNSET
    _CACHE["rates"] = _UNSET
    _CACHE["armed"] = False
    _CACHE["logged_cap"] = None
    _CACHE["cap_trimmed"] = False
    _CACHE["measure"] = _UNSET
    _CACHE["measured"] = None
    _CACHE["pending"] = None


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
    allocator and profits from the same trim). Q-694b: after the trim, the
    rate measurement opens its reading (only with
    ``SGLANG_WEG2_EXTEND_RATE_MEASURE``; off = byte-identical)."""
    line = _before_extend_trim(worker, batch)
    if batch is not None and measure_armed():
        measure_open(worker, batch)
    return line


def _before_extend_trim(worker, batch) -> Optional[str]:
    values = thresholds()
    if batch is None or (values is None and not _CACHE["cap_trimmed"]):
        return None
    if getattr(worker, "is_draft_worker", False) and not getattr(worker, "is_phase_flip_tp_stack", False):
        return None
    try:
        mode = batch.forward_mode
        if not mode.is_extend() or mode.is_target_verify():
            return None
        # WEG2-EXTEND-CAP (29.09.): the vote's cap trim is spent by this extend too
        _CACHE["cap_trimmed"] = False
        if values is None:
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
# rc12g WEG2-EXTEND-STUECKELUNG: the chunk follows the card, not the other way.
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
#     WEG2-EXTEND-STUECKELUNG rank=0 geplant=4096 cap=3648 post=2385 rate=0.5625
#
# Off (no ``SGLANG_WEG2_EXTEND_GROWTH_PER_ROW_MIB``, or no rate for this rank):
# no vote, byte-identical. Only while prefill work is pending does it read the
# card; it trims at most once per extend (``armed`` until an extend runs), so a
# queue that waits for seats does not synchronize every decode round.
# --------------------------------------------------------------------------

STUECKELUNG_MARKER = "WEG2-EXTEND-STUECKELUNG"
#: the card line the cap keeps free after the chunk's growth (the operator's HALT line)
STUECKELUNG_FLOOR_MIB = 300.0
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
    rows = int(math.floor((float(post_mib) - STUECKELUNG_FLOOR_MIB) / float(rate)))
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
        # Q-694b: the start rate (record / derived) or this rank's measured
        # rate x RATE_SAFETY, whichever is larger; off: the start rate itself
        start_rate = rate
        rate = effective_rate(rate)
        if bool(cuda.is_current_stream_capturing()):
            return None
        if not _CACHE["armed"]:
            if maybe_trim(cuda, rank, threshold_for(rank, thresholds()), clock) is not None:
                _CACHE["armed"] = True
        post, _total = cuda.mem_get_info()
        post_mib = post / MIB
        # WEG2-EXTEND-CAP: under the P0 torch cache cap the allocator's own line
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
            post_mib = min(post_mib, cap_post)
        cap = rows_cap(post_mib, rate, page_size)
        if cap >= int(configured):
            _CACHE["logged_cap"] = None
            return None
        if cap != _CACHE["logged_cap"]:
            _CACHE["logged_cap"] = cap
            logger.info(
                f"{STUECKELUNG_MARKER} rank={rank} geplant={int(configured)} cap={cap} "
                f"post={post_mib:.0f} rate={rate:.4f} floor={STUECKELUNG_FLOOR_MIB:.0f}"
                + ("" if cap_post is None else f" torch_cap={torch_cap:.0f} cap_post={cap_post:.0f}")
                + ("" if not measure_armed() else
                   f" rate_src={'measured' if rate > start_rate else 'start'} start_rate={start_rate:.4f}")
            )
        return cap
    except Exception as exc:  # noqa: BLE001 -- a vote that cannot price abstains
        logger.debug("%s skipped: %s", STUECKELUNG_MARKER, exc)
        return None


def _torch_cap_mib(rank: int) -> Optional[float]:
    """This rank's P0 torch cache cap in MiB while it is armed, else ``None``
    (no cap: the vote reads the card alone, byte-identical)."""
    from sglang.srt.weg2 import torch_cache_cap as _tcc

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
# allocator's peak counter (WEG2-VRAM-PEAK owns those re-bases): the peak term
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


def measure_armed() -> bool:
    """``SGLANG_WEG2_EXTEND_RATE_MEASURE``, read once per process."""
    if _CACHE["measure"] is _UNSET:
        try:
            from sglang.srt.environ import envs

            _CACHE["measure"] = bool(envs.SGLANG_WEG2_EXTEND_RATE_MEASURE.get())
        except Exception:  # noqa: BLE001 -- an instrument never kills a forward
            _CACHE["measure"] = False
    return bool(_CACHE["measure"])


def measured_rate() -> Optional[float]:
    """This rank's measured extend rate (MiB/row), ``None`` before the first."""
    return _CACHE["measured"]  # type: ignore[return-value]


def effective_rate(start: float) -> float:
    """The rate the vote uses: ``max(start, measured x RATE_SAFETY)`` while the
    measurement is armed and holds a value, else ``start``."""
    m = _CACHE["measured"] if measure_armed() else None
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
        if rows < GROWTH_PER_ROW_MIN_ROWS:
            return None
        pa1, _a1, r1 = _alloc_stats(cuda)
        # the peak counter is this extend's only if the extend raised it
        transient = (pa1 - a0) if pa1 > pa0 else 0
        growth = max(0, r1 - r0)
        cost = max(transient, growth)
        if cost <= 0:
            return None
        rate = _ceil4(cost / MIB / rows)
        prev = _CACHE["measured"]
        if prev is not None and rate <= float(prev):
            return None
        _CACHE["measured"] = rate
        start = threshold_for(int(rank) if rank is not None else 0, rates())
        eff = None if start is None else effective_rate(start)
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
