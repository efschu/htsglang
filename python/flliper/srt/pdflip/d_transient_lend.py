"""D-TRANSIENT-LEND: D's booked transient is expert rows between two extends (01.10.).

User law (29.09. ~10:20Z): "Ungenutzter Speicher wird, bis er genutzt werden
müsste, durch MoE-Experten befüllt. IMMER. Falls durch irgendwas mehr VRAM
gebraucht wird, werden MoE-Experten in den Systemram verlagert. IMMER."

What the metal showed (fqnsdm, 01.10. 14:00-14:39Z, 1-s NVML sampler): D is
awake ~95 % of the time and holds 1.7-2.0 GiB NVML-free per card plus 650-870
MiB of allocator cache. Nothing in that is a reserve -- it is the planner's
statically booked transient: the corridor floor above the near-OOM edge
(700-767), awake_overshoot (404), the D-extend activation (1024/1104), the KV
share the current stage does not map. All of it is spent only at the peak of
a D extend (+ a KV-stage move). D-ELASTIK-EXPERTEN-0929 struck "refill after
the extend" as covered by EXTEND-TRIM; EXTEND-TRIM only empties the cache, it
never turns a row ON -- that is the gap closed here.

THE ONE MECHANISM. No second residency: the rows are the bank's seat rows
(``d_seat_vram.SeatVram``), turned ON / OFF by the same live table + span
moves as every phase move. Expert bytes live 1:1 in the store (SlotLedger), so
a returned row writes nothing back and a lent row fills lazily on its next
miss (the ordinary LRU fetch).

ORDER (two hooks in the scheduler's batch funnel, both before the batch runs;
every rank of D sees the same queue and builds the same batch, so the VERDICT
is group-uniform without a collective -- the AMOUNT is rank-local, each rank
lends its own card):

* round start (``round_start``, before the #794 width vote reads the card):
  extend work is pending (a chunked request or a waiting one -- the same
  predicate as rc12g's ``width_vote``): RETURN now, so the chunk width and
  the EXTEND-TRIM see the card without the lend;
* the built batch is an extend (or anything that is not decode/idle): RETURN
  if anything is still lent -- rows OFF coldest first, sync, the lattice
  cells above the phase's k released -- then the extend allocates its
  activation on the freed bytes;
* ``SETTLE`` decode/idle rounds in a row with nothing pending (a burst of
  extends or a waiting queue lends nothing):
  LEND -- ``card_free - floor`` (cudaMemGetInfo, no sync, no empty_cache: the
  allocator cache stays for the decode's own reuse) becomes the largest lattice
  k above the phase's rows whose pages fit; map first, then ON;
* a stage move (D-MEM-SCHED), a re-seat or a wake re-plans from the phase cell
  and ends the lend in the same apply (``lent_from = None``): every lend ends on
  the cut lattice, so those applies release whole cells only (S1-Wisch) --
  S106 STAGE-KEEP: a live stage move and a live re-seat SHRINK keep the lent
  rows the card holds beside what the move maps (``stage_keep_need_bytes``: the
  lend floor + a pending short extend's KEEP-SHORT price, measured card), and
  the lend stays open above the new phase rows (``lent_from`` = the new cell's
  k). rc22-rc26 (20.10.): the old end-and-re-lend pair re-mapped the same cells
  in >= 90 % of the cases (rc26 TP0 134 re-lends, 133 back at or above the
  dropped rows), each pair a decode pause (rc26 group 82.4 s / 0.44 h);
* a sleep pauses the whole bank; the wake's apply re-plans from the cell.

No reserve: the floor is the card ledger's near-OOM edge (the launcher writes
``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB`` beside ``FLLIPER_PDFLIP_EXTEND_TRIM_MIB``); without
it nothing is lent and nothing is guessed. Uneven distribution is untouched:
only the rank-local LRU capacity changes.

LINE (rank-local, one per lend / return):

    PDFLIP D-TRANSIENT-LEND rank=1 lend rows_on 15->23 card_free_mib=1850
      floor_mib=700 budget_mib=1150 lent_total=.. returned_total=..

S78 KEEP-SHORT (19.10., ``FLLIPER_PDFLIP_D_LEND_KEEP_SHORT``, default ON). rc17 D
(``..._1019_171636``): 381 lends and 312 returns in 45 min -- one cycle per short
request. The return before a short mixed extend left 2.0-2.2 GiB free (p50) at the
forward's start on every rank, while the forward grew the allocator by <= 65-437 MiB
(T < 1024, ``PDFLIP-VRAM-PEAK phase=chunk``): ~1.3-1.5 GiB of expert rows per card
sat empty through the forward the experts were needed in, and the re-lend after it
paused the decodes ~300 ms (``D-DECODE-GAP`` tails with a lend, p50 312 ms). Now a
pending extend that is SHORT by the front's own R2 verdict (every waiting request
carries halbX and its priced uncached is under it, no chunked request; replicated, so
the verdict is group-uniform) gives back only the cells above the largest lattice k
that leaves ``lend floor + 300 + rate x rows`` MiB free -- the rc12g chunk model's
price (``extend_trim.chunk_need_mib``), so the width vote still funds the whole
request in one chunk. Before the built extend runs the same price is taken with the
batch's exact rows. A big or unpriced extend, or a rank without a chunk rate, returns
everything as before. After a partial return the next lend verdict tops the lend up
again (one ``lend`` line, cells mapped only for the difference). Amount rank-local as
before; the switch off = the full return byte for byte. Lines::

    PDFLIP D-TRANSIENT-LEND rank=0 keep rows_on 15 (short extend pending rows=312
      need_mib=1250 card_free_mib=1380)
    PDFLIP D-TRANSIENT-LEND rank=0 return rows_on 15->11 (short extend pending ...)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

MARK = "PDFLIP D-TRANSIENT-LEND"
GATE_ATTR = "_pdflip_d_lend_gate"
MIB = 1 << 20


class LendGate:
    """REPLICATED: the decode/idle streak of the D group (the batch kind is
    the same on every rank). ``step`` answers return / lend / hold.
    ``topup`` (S78, rank-local): a short extend gave back only part of the
    lend -- the next lend verdict may map the difference again."""

    __slots__ = ("settle", "streak", "topup", "last_keep")

    def __init__(self, settle: int):
        self.settle = max(1, int(settle))
        self.streak = 0
        self.topup = False
        # S78 review A2: the last logged ``keep`` (rows_on, rows, need_mib) -- a short request waiting for a
        # seat re-prices every iteration; only a change is a line (any move clears it)
        self.last_keep = None

    def step(self, kind: str) -> str:
        if kind == "extend":
            self.streak = 0
            return "return"
        self.streak += 1
        return "lend" if self.streak >= self.settle else "hold"


def batch_kind(batch) -> str:
    """``extend`` for anything that allocates an extend's activation (prefill,
    mixed, a chunk), ``decode`` for a decode / verify round, ``idle`` for no
    batch."""
    if batch is None:
        return "idle"
    mode = batch.forward_mode
    if mode.is_decode() or mode.is_target_verify() or mode.is_idle():
        return "decode"
    return "extend"


def lend_budget_bytes(free_bytes: int, floor_mib: float) -> int:
    """The card bytes above the near-OOM edge (never negative)."""
    return max(0, int(free_bytes) - int(float(floor_mib) * MIB))


def _gate(sched) -> LendGate:
    gate = getattr(sched, GATE_ATTR, None)
    if gate is None:
        from flliper.srt.environ import envs

        gate = LendGate(int(envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.get()))
        setattr(sched, GATE_ATTR, gate)
    return gate


def _card_free_bytes() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception as exc:  # noqa: BLE001 -- no driver: nothing to lend
        logger.debug("%s skipped: %s", MARK, exc)
        return None


#: S93: the admission's own park verdict of the last pass (``note_gate``) and the running rids it was taken on
GATE_SEEN_ATTR = "_pdflip_d_lend_gate_seen"
#: S93: the last logged hold census (waiting, held, reasons) -- a line only on a change
HOLD_SEEN_ATTR = "_pdflip_d_lend_hold_seen"


def hold_aware_on() -> bool:
    """S93 switch (default ON): only a TAKEABLE waiting request is pending extend work."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_D_LEND_HOLD_AWARE.get())
    except Exception:  # noqa: BLE001
        return False


def _running_rids(sched, running_batch=None) -> frozenset:
    rb = running_batch if running_batch is not None else getattr(sched, "running_batch", None)
    try:
        return frozenset(str(getattr(r, "rid", "")) for r in (getattr(rb, "reqs", None) or ()))
    except Exception:  # noqa: BLE001
        return frozenset()


def note_gate(sched, gate, running_batch) -> None:
    """S93: D's admission hands its park verdict of this pass here (``Scheduler._pdflip_d_park_admission``).
    ``gate`` is ``d_seats.AdmissionGate`` (None = no barrier stands: the gate holds nobody). Kept with the
    running rids it was taken on -- a change of the running set voids it (an older one may have finished)."""
    try:
        setattr(sched, GATE_SEEN_ATTR, (gate, _running_rids(sched, running_batch)))
    except Exception:  # noqa: BLE001 -- an instrument input never stops the admission
        pass


def _dqh_holds_queue(sched) -> bool:
    """S93: DQH (pdflip/d_head_hold.py) holds the queue -- armed by a pass that built nothing while requests
    waited, and nothing its verdict reads has changed (the scheduler's own fingerprint, replicated values).
    Read only: ``should_hold`` (which ticks) is never called here."""
    dqh = getattr(sched, "pdflip_d_head_hold", None)
    if dqh is None:
        return False
    try:
        st = dqh.state
        if not dqh.enabled or st.fingerprint is None:
            return False
        fp_of = getattr(sched, "_pdflip_d_head_fingerprint", None)
        if not callable(fp_of):
            return False
        fp = fp_of(getattr(sched, "running_batch", None), sched.__dict__.get("_pass_prefetch_verdicts"))
        return fp is not None and fp == st.fingerprint
    except Exception:  # noqa: BLE001 -- unreadable: not held (the conservative side: the lend goes back)
        return False


def held_reason(sched, req, dqh: Optional[bool] = None) -> Optional[str]:
    """S93: why D's admission will NOT take ``req`` (its own verdicts, no second one), else None (takeable).

    * ``pdflip_d_head_hold``: DQH holds the queue (covers the group's NO_TOKEN / D-WALL-HEAD refusal of the
      head: DQH is armed exactly by a pass that built nothing, on replicated values);
    * the park gate's census key (``AdmissionGate.skip``): ``pdflip_d_park_older_live`` (older_live and
      held_for_older), ``pdflip_d_park_first`` (the barrier for a newcomer), ``pdflip_d_park_decode_first``.

    The gate counts only while the running set is the one it was taken on. A rank-local NO_TOKEN of a single
    rank is NOT read (it is rank-local; the group's verdict on it is DQH's)."""
    if (_dqh_holds_queue(sched) if dqh is None else dqh):
        return "pdflip_d_head_hold"
    seen = getattr(sched, GATE_SEEN_ATTR, None)
    if not seen:
        return None
    gate, running = seen
    if gate is None or running != _running_rids(sched):
        return None
    try:
        return gate.skip(req, admitted=())
    except Exception:  # noqa: BLE001 -- unreadable: takeable (the lend goes back)
        return None


def takeable_waiting(sched) -> list:
    """S93: the waiting requests D's admission can take (all of them with the switch off)."""
    queue = list(getattr(sched, "waiting_queue", None) or ())
    if not queue or not hold_aware_on():
        return queue
    dqh = _dqh_holds_queue(sched)
    return [r for r in queue if held_reason(sched, r, dqh) is None]


def _note_hold(sched, queue, takeable) -> None:
    """S93 line: a non-empty queue that lends anyway, named once per change (waiting, held, reasons)."""
    if len(takeable) == len(queue):
        setattr(sched, HOLD_SEEN_ATTR, None)
        return
    dqh = _dqh_holds_queue(sched)
    reasons = {}
    for r in queue:
        k = held_reason(sched, r, dqh)
        if k is not None:
            reasons[k] = reasons.get(k, 0) + 1
    key = (len(queue), len(queue) - len(takeable), tuple(sorted(reasons.items())))
    if key == getattr(sched, HOLD_SEEN_ATTR, None):
        return
    setattr(sched, HOLD_SEEN_ATTR, key)
    sched.__dict__["_pdflip_d_lend_hold_n"] = int(sched.__dict__.get("_pdflip_d_lend_hold_n", 0) or 0) + 1
    logger.info("%s rank=%d hold-aware waiting=%d held=%d reasons=%s first_held=%s -- D's admission holds "
                "them out this pass (its own verdicts): no extend is coming, the expert rows stay lent (S93)",
                MARK, _rank(sched), len(queue), len(queue) - len(takeable), dict(sorted(reasons.items())),
                next((str(getattr(r, "rid", "?")) for r in queue if id(r) not in {id(t) for t in takeable}), "-"))


def pending_trigger(sched) -> Optional[str]:
    """S93: what makes extend work pending -- ``chunked`` or the rid of the first takeable waiting request;
    None = nothing pending (an empty queue, or every waiting request held by the admission's verdicts)."""
    if getattr(sched, "chunked_req", None) is not None:
        return "chunked"
    queue = list(getattr(sched, "waiting_queue", None) or ())
    if not queue:
        return None
    takeable = takeable_waiting(sched)
    if hold_aware_on():
        _note_hold(sched, queue, takeable)
    if not takeable:
        return None
    return "rid=%s" % getattr(takeable[0], "rid", "?")


def extend_pending(sched) -> bool:
    """Extend work the next batches may build: a chunked request or a waiting
    one D's admission can take (rc12g ``width_vote``'s predicate, S93: minus the
    waiting requests the admission's own verdicts hold out -- park gate, DQH;
    the queue, the running set, the gate's inputs and DQH's fingerprint are
    replicated over D's ranks)."""
    return pending_trigger(sched) is not None


def _controller(sched):
    """The applied seat controller of an awake, armed D rank, else None."""
    from flliper.srt.pdflip import d_seat_vram as dsv

    if not dsv.armed() or not dsv.lend_armed() or getattr(sched, "pdflip_dormant", False):
        return None
    st = getattr(sched, dsv.PHASE_ATTR, None)
    if st is None or not st.done:
        return None
    ctl = dsv.controller(sched)
    if ctl is None or ctl.applied is None:
        return None
    return ctl


def _rank(sched) -> int:
    # the rank lives on the parallel state (rc12h: the Scheduler has no tp_rank)
    rank = getattr(getattr(sched, "ps", None), "tp_rank", None)
    if rank is None:
        rank = getattr(sched, "tp_rank", 0)
    return int(rank or 0)


def _gap_note(sched, cause: str, ms: Optional[float] = None) -> None:
    """S78: name the lend move in the decode pause it may cause (D-DECODE-GAP, read-only). S106: with its
    own measured host wall (``ms``: the move's call, its syncs included) when the caller timed it."""
    try:
        from flliper.srt.pdflip import d_decode_gap

        if ms is None:
            d_decode_gap.note(sched, cause)
        else:
            d_decode_gap.note_ms(sched, cause, ms)
    except Exception:  # noqa: BLE001 -- an instrument never stops a lend
        pass


def _timed(fn, *args):
    """S106: ``(result, host wall ms)`` of one lend move."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1000.0


def _give_back(sched, ctl, why: str) -> Optional[str]:
    if ctl.lent_from is None:
        return None
    before = int(ctl.rows_on)
    n, ms = _timed(ctl.unlend)
    _gate(sched).topup = False
    _gate(sched).last_keep = None
    if n:
        _gap_note(sched, "lend_return", ms)
        logger.info("%s rank=%d return rows_on %d->%d (%s) lent_total=%d returned_total=%d ms=%.0f",
                    MARK, _rank(sched), before, int(ctl.rows_on), why, ctl.lent_rows,
                    ctl.returned_rows, ms)
    return "return" if n else None


def keep_short_on() -> bool:
    """S78 KEEP-SHORT switch (default ON)."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.get())
    except Exception:  # noqa: BLE001
        return False


def _decode_rows(sched) -> int:
    """The running decodes a mixed batch folds in (one row each)."""
    try:
        return len(getattr(getattr(sched, "running_batch", None), "reqs", None) or ())
    except Exception:  # noqa: BLE001
        return 0


def short_pending_rows(sched) -> Optional[int]:
    """S78: the rows the pending extend work may need when ALL of it is SHORT by
    the front's R2 verdict -- every waiting request carries the front's halbX and
    priced uncached with ``priced < halbX`` (first admission only, the same
    ``front_x.request_front_x`` the X gate reads), no chunked request -- summed,
    plus the running decodes a mixed batch folds in. ``None`` = big, unpriced
    or nothing pending: the full return. Replicated inputs only (the queue and
    the carried headers are the same on every D rank)."""
    if getattr(sched, "chunked_req", None) is not None:
        return None
    queue = takeable_waiting(sched)  # S93: the held ones build no extend this pass
    if not queue:
        return None
    try:
        from flliper.srt.pdflip import front_x

        total = 0
        for req in queue:
            hx = getattr(req, front_x.ATTR_HALFX, None)
            _x, priced = front_x.request_front_x(req)
            if hx is None or priced is None or int(hx) <= 0 or int(priced) >= int(hx):
                return None
            total += max(0, int(priced))
    except Exception:  # noqa: BLE001 -- unreadable = unpriced = the full return
        return None
    return total + _decode_rows(sched)


def batch_rows(batch) -> Optional[int]:
    """The exact rows of a built extend batch (prefill tokens + folded decode rows)."""
    try:
        if batch is None or not batch.forward_mode.is_extend():
            return None
        return int(batch.extend_num_tokens)
    except Exception:  # noqa: BLE001
        return None


def _keep_short(sched, ctl, rows: int, why: str) -> Optional[str]:
    """S78 KEEP-SHORT: give back only the lent cells the extend's price needs.

    Price = this rank's lend floor + the rc12g chunk model's ``300 + rate x rows``
    (``extend_trim.chunk_need_mib``); no floor, no rate or no card reading = the
    full return (nothing is guessed). Returns ``return`` when rows moved,
    ``keep`` when the card already holds the price, else None."""
    from flliper.srt.pdflip import d_seat_vram as dsv
    from flliper.srt.pdflip import extend_trim

    rank = _rank(sched)
    floor = dsv.lend_floor_mib_for_rank(rank)
    need = extend_trim.chunk_need_mib(rank, int(rows))
    free = _card_free_bytes()
    if floor is None or need is None or free is None:
        return _give_back(sched, ctl, why + " (unpriced)")
    need_mib = float(floor) + float(need)
    need_bytes = int(need_mib * MIB)
    before = int(ctl.rows_on)
    k = ctl.unlend_target(need_bytes, free)
    gate = _gate(sched)
    if k >= before:
        key = (before, int(rows), int(need_mib))
        if key != gate.last_keep:
            gate.last_keep = key
            logger.info("%s rank=%d keep rows_on %d (%s rows=%d need_mib=%d card_free_mib=%d)",
                        MARK, rank, before, why, int(rows), int(need_mib), free // MIB)
        # TOPUP (20.10.): a keep moves nothing -- it must not cancel a top-up a partial return armed (rc27:
        # the extend-time keep cleared it after every KEEP-SHORT return, the lend never grew back)
        return "keep"
    gate.last_keep = None
    n, ms = _timed(ctl.unlend_to, k)
    gate.topup = ctl.lent_from is not None
    if n:
        _gap_note(sched, "lend_return", ms)
        logger.info("%s rank=%d return rows_on %d->%d (%s rows=%d need_mib=%d card_free_mib=%d) "
                    "lent_total=%d returned_total=%d ms=%.0f", MARK, rank, before, int(ctl.rows_on), why,
                    int(rows), int(need_mib), free // MIB, ctl.lent_rows, ctl.returned_rows, ms)
    return "return" if n else None


def stage_keep_need_bytes(sched, ctl=None) -> Optional[int]:
    """S106 STAGE-KEEP: the card bytes a live phase move (D-MEM-SCHED stage, re-seat shrink) must
    leave free when it keeps an open lend -- the lend's OWN price, nothing new:

    * the lend floor of this rank (the card ledger's near-OOM edge, ``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB``);
    * plus, when extend work is pending and ALL of it is short (S78's R2 verdict, replicated), the
      KEEP-SHORT price ``extend_trim.chunk_need_mib(rank, rows)`` -- the same price ``round_start``
      already left free before this move; the move must not eat it.

    ``None`` (the lend ends with the move, the old behaviour): nothing lent, the lend not armed, no
    floor, a chunked request, a big or unpriced pending extend (``round_start`` returned it in full),
    no rate for this rank. The VERDICT inputs (queue, chunked request, carried headers) are replicated;
    the amount -- like every lend amount -- is rank-local."""
    try:
        if ctl is None:
            ctl = _controller(sched)
        if ctl is None or getattr(ctl, "lent_from", None) is None:
            return None
        from flliper.srt.pdflip import d_seat_vram as dsv

        rank = _rank(sched)
        floor = dsv.lend_floor_mib_for_rank(rank)
        if floor is None:
            return None
        if getattr(sched, "chunked_req", None) is not None:
            return None
        need_mib = float(floor)
        if takeable_waiting(sched):
            rows = short_pending_rows(sched) if keep_short_on() else None
            if rows is None:
                return None
            from flliper.srt.pdflip import extend_trim

            price = extend_trim.chunk_need_mib(rank, int(rows))
            if price is None:
                return None
            need_mib += float(price)
        return int(need_mib * MIB)
    except Exception:  # noqa: BLE001 -- unpriced: the lend ends with the move (the old behaviour)
        return None


def after_phase_move(sched, ctl, rows_before: int) -> None:
    """S106: a live phase move that KEPT an open lend arms a top-up (``topup``): the move changed what the
    card holds (a stage shrink frees KV bytes, a partial keep gave rows back) -- the next lend verdict maps
    whatever difference now fits (TOPUP 20.10.: before, a whole keep left the lend as it was; with S106 no
    stage move dropped the lend any more, so the lend only ever shrank -- rc27 TP0 bs2 rows_on p50 13 against
    rc26 24). A move that ended the lend clears it (the full lend follows the settle)."""
    try:
        gate = _gate(sched)
        if ctl is None or getattr(ctl, "lent_from", None) is None:
            gate.topup = False
            gate.last_keep = None
            return
        gate.topup = True
        if int(ctl.rows_on) < int(rows_before):
            gate.last_keep = None
    except Exception:  # noqa: BLE001 -- the gate is advice; the move already ran
        pass


#: F3: the lend's map on the helper thread (None = auto: on with a CUDA context; CPU desk ranks map inline)
ASYNC_LEND: Optional[bool] = None
#: F3: the gate's view of an in-flight lend (card_free, floor, budget) for its line at the finish
PENDING_ATTR = "_pdflip_d_lend_pending"


def async_lend_on() -> bool:
    if ASYNC_LEND is not None:
        return bool(ASYNC_LEND)
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001
        return False


def _log_lend_done(sched, ctl, done: Optional[dict], how: str) -> None:
    """F3: the finished lend's line (the S78/S93 shape plus map/join/on ms) and its gap note: what the
    SCHEDULER thread spent (join + rows ON), not the map, which ran beside the decode."""
    if not done:
        return
    pend = getattr(sched, PENDING_ATTR, None) or (0, 0, 0)
    try:
        setattr(sched, PENDING_ATTR, None)
    except Exception:  # noqa: BLE001
        pass
    if int(done.get("rows", 0)) <= 0:
        return
    host_ms = float(done.get("join_ms", 0.0)) + float(done.get("on_ms", 0.0))
    _gate(sched).last_keep = None
    _gap_note(sched, "lend", host_ms)
    logger.info("%s rank=%d lend rows_on %d->%d card_free_mib=%d floor_mib=%d budget_mib=%d "
                "lent_total=%d returned_total=%d ms=%.0f map_ms=%.0f join_ms=%.0f (%s; F3: the map ran on the "
                "helper thread, the rows went ON after it)", MARK, _rank(sched), int(done["before"]),
                int(done["after"]), int(pend[0]) // MIB, int(pend[1]), int(pend[2]) // MIB, ctl.lent_rows,
                ctl.returned_rows, host_ms, float(done.get("map_ms", 0.0)), float(done.get("join_ms", 0.0)), how)


def _register_streams(sched) -> None:
    """F3: the scheduler's forward stream reads the bank rows (the overlap loop's forward_stream)."""
    try:
        from flliper.srt.pdflip.d_row_fence import register_row_stream

        register_row_stream(getattr(sched, "forward_stream", None))
        # F3 review A1: FLLIPER_KVSO_DECOUPLE runs a second full forward per iteration on spill_stream
        register_row_stream(getattr(sched, "spill_stream", None))
    except Exception:  # noqa: BLE001
        pass


def poll(sched, ctl=None) -> None:
    """F3, once per pass (top of ``round_start``): a lend whose map finished goes ON, the deferred unmaps
    whose ROW-FENCE completed go, the warm copies that landed are committed -- event queries and a done
    future only, never a wait."""
    try:
        _register_streams(sched)
        if ctl is None:
            ctl = _controller(sched)
        if ctl is None:
            return
        done = getattr(ctl, "lend_done", None)
        if done is not None:
            ctl.lend_done = None
            _log_lend_done(sched, ctl, done, "landed in a move")
    except Exception:  # noqa: BLE001 -- the poll's bookkeeping is advice; the next move settles
        logger.debug("%s poll skipped", MARK, exc_info=True)
        return
    if ctl is None:
        return
    # F3 review A2: a lend whose map landed -- a REFUSED map (finish_lend re-raises the helper thread's
    # PdFlipDSeatVram* error) is named at WARNING and stops the scheduler by name, as before F3; never DEBUG
    if getattr(ctl, "lend_job", None) is not None:
        t0 = time.perf_counter()
        try:
            done = ctl.finish_lend(block=False)
        except Exception as exc:
            logger.warning("%s rank=%d lend map REFUSED on the helper thread: %s: %s -- stopping by name "
                           "(F3 review A2)", MARK, _rank(sched), type(exc).__name__, exc)
            raise
        if done is not None:
            done["on_ms"] = (time.perf_counter() - t0) * 1000.0
            _log_lend_done(sched, ctl, done, "map done")
    # a refused unmap (tms rc) or a failed warm commit stops by name, as every bank move did before F3
    if getattr(ctl, "pending_releases", None) or getattr(ctl, "release_jobs", None):
        ctl.drain_releases(block=False)
    from flliper.srt.pdflip.d_row_fence import poll_warm

    poll_warm()


def _bytes_now(sched, ctl, why: str) -> None:
    """F3: an extend (or its width vote) needs the returned bytes in THIS pass -- the deferred unmaps go now,
    waiting on their ROW-FENCE's E1 alone (the rows' last readers: the forward in flight). The wait overlaps
    that forward on the device; the line names it."""
    # F3 review B1: an unmap already handed to the map thread (release_jobs) is joined too -- the width
    # vote and the extend must read the card with the cells back
    if not (getattr(ctl, "pending_releases", None) or getattr(ctl, "release_jobs", None)):
        return
    ms = ctl.drain_releases(block=True)
    _gap_note(sched, "lend_return", ms)
    if ms >= 1.0:
        logger.info("%s rank=%d fence-wait ms=%.1f (%s: E1 of the return's ROW-FENCE only -- never device-wide)",
                    MARK, _rank(sched), ms, why)


def quiesce_for_sleep(sched) -> None:
    """F3: before D sleeps (the bank pauses): an in-flight lend's map is joined and landed, every deferred
    unmap goes (E1), every warm copy lands (its own event) -- nothing of the bank moves under the pause.
    Rank-local; the sleep is the group's verdict."""
    from flliper.srt.pdflip import d_seat_vram as dsv
    from flliper.srt.pdflip.d_row_fence import quiesce_warm

    try:
        ctl = dsv.controller(sched)
    except Exception:  # noqa: BLE001
        ctl = None
    try:
        if ctl is not None:
            ctl.settle()
            ctl.lend_done = None
    finally:
        # F3 review A3: the warm jobs never outlive the sleep, whatever settle raised -- a job committed after
        # the wake would route an expert to a row whose bytes the pause dropped
        quiesce_warm()


def round_start(sched) -> Optional[str]:
    """Top of ``get_next_batch_to_run``: extend work pending -> the lend goes
    back BEFORE the round's width vote and EXTEND-TRIM read the card (S78: only
    the short extend's price when all pending work is short). F3: first the pass's poll; with extend work
    pending an in-flight lend is joined (its map thread) and landed, and the returned cells go before the
    width vote reads the card (``_bytes_now``: E1 only)."""
    ctl = _controller(sched)
    poll(sched, ctl)
    if ctl is None:
        return None
    trigger = pending_trigger(sched)
    if trigger is None:
        return None
    _gate(sched).streak = 0
    if getattr(ctl, "lend_job", None) is not None:
        t0 = time.perf_counter()
        done = ctl.finish_lend(block=True)
        if done is not None:
            done["on_ms"] = (time.perf_counter() - t0) * 1000.0 - float(done.get("join_ms", 0.0))
            _log_lend_done(sched, ctl, done, "joined: extend pending")
    if ctl.lent_from is None:
        return None
    rows = short_pending_rows(sched) if keep_short_on() else None
    if rows is not None:
        out = _keep_short(sched, ctl, rows, "short extend pending %s" % trigger)
    else:
        out = _give_back(sched, ctl, "extend pending %s" % trigger)
    _bytes_now(sched, ctl, "the pending extend's width vote")
    return out


def on_batch(sched, batch) -> Optional[str]:
    """End of the scheduler's batch funnel (``get_next_batch_to_run``), every
    rank of an awake D, before the batch runs. Returns the verdict that moved
    rows (``lend`` / ``return``), else None."""
    from flliper.srt.pdflip import d_seat_vram as dsv

    ctl = _controller(sched)
    if ctl is None:
        return None
    kind = batch_kind(batch)
    if kind != "extend" and extend_pending(sched):
        kind = "extend"  # a waiting queue lends nothing (its extend is next)
    gate = _gate(sched)
    verdict = gate.step(kind)
    rank = _rank(sched)
    if verdict == "return":
        if getattr(ctl, "lend_job", None) is not None:
            done = ctl.finish_lend(block=True)  # F3: the extend's bytes -- the map thread lands first
            _log_lend_done(sched, ctl, done, "joined: before the extend")
        out, priced = None, False
        if keep_short_on() and ctl.lent_from is not None:
            # S78: the built extend's exact rows price it; a decode batch with a
            # waiting queue reprices the pending work (big/unpriced: full return)
            rows = batch_rows(batch)
            if rows is None:
                rows = short_pending_rows(sched)
            if rows is not None:
                priced = True
                out = _keep_short(sched, ctl, rows, "before the short extend")
        if not priced:
            out = _give_back(sched, ctl, "before the extend")
        if batch_kind(batch) == "extend":
            _bytes_now(sched, ctl, "the built extend allocates next")
        return out
    if verdict != "lend":
        return None
    if getattr(ctl, "lend_job", None) is not None:
        return None  # F3: one lend in flight (its map thread runs; poll lands it)
    topup = ctl.lent_from is not None
    if topup and not gate.topup and gate.streak != gate.settle:
        # one lend per settle: an open lend is topped up when a move armed it, or once at the settle of every
        # decode streak (after each extend) -- whatever the card frees in between becomes rows again
        # (TOPUP 20.10., user law: free VRAM = experts); the map covers only the difference
        return None
    gate.topup = False
    floor = dsv.lend_floor_mib_for_rank(rank)
    free = _card_free_bytes()
    if floor is None or free is None:
        return None
    budget = lend_budget_bytes(free, floor)
    before = int(ctl.rows_on)
    if async_lend_on() and hasattr(ctl, "start_lend"):
        # F3: the map (cuMem* per cell) runs on the helper thread; the rows go ON at a later pass's poll
        aim, ms = _timed(ctl.start_lend, budget)
        if aim <= 0:
            return None
        try:
            setattr(sched, PENDING_ATTR, (free, floor, budget))
        except Exception:  # noqa: BLE001
            pass
        logger.info("%s rank=%d lend-start rows_on %d->%d (map on the helper thread) card_free_mib=%d "
                    "floor_mib=%d budget_mib=%d ms=%.0f", MARK, rank, before, before + int(aim), free // MIB,
                    int(floor), budget // MIB, ms)
        return "lend"
    n, ms = _timed(ctl.lend, budget)
    if n:
        gate.last_keep = None
        _gap_note(sched, "lend", ms)
        logger.info("%s rank=%d lend rows_on %d->%d card_free_mib=%d floor_mib=%d budget_mib=%d "
                    "lent_total=%d returned_total=%d ms=%.0f", MARK, rank, before, int(ctl.rows_on),
                    free // MIB, int(floor), budget // MIB, ctl.lent_rows, ctl.returned_rows, ms)
    return "lend" if n else None
