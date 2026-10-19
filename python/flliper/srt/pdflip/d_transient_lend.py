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
  the cut lattice, so those applies release whole cells only (S1-Wisch);
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

    __slots__ = ("settle", "streak", "topup")

    def __init__(self, settle: int):
        self.settle = max(1, int(settle))
        self.streak = 0
        self.topup = False

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


def extend_pending(sched) -> bool:
    """Extend work the next batches may build: a chunked request or a waiting
    one (rc12g ``width_vote``'s predicate; the queue is replicated over D's
    ranks)."""
    return getattr(sched, "chunked_req", None) is not None or bool(
        getattr(sched, "waiting_queue", None))


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


def _gap_note(sched, cause: str) -> None:
    """S78: name the lend move in the decode pause it may cause (D-DECODE-GAP, read-only)."""
    try:
        from flliper.srt.pdflip import d_decode_gap

        d_decode_gap.note(sched, cause)
    except Exception:  # noqa: BLE001 -- an instrument never stops a lend
        pass


def _give_back(sched, ctl, why: str) -> Optional[str]:
    if ctl.lent_from is None:
        return None
    before = int(ctl.rows_on)
    n = ctl.unlend()
    _gate(sched).topup = False
    if n:
        _gap_note(sched, "lend_return")
        logger.info("%s rank=%d return rows_on %d->%d (%s) lent_total=%d returned_total=%d",
                    MARK, _rank(sched), before, int(ctl.rows_on), why, ctl.lent_rows,
                    ctl.returned_rows)
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
    queue = list(getattr(sched, "waiting_queue", None) or ())
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
        logger.info("%s rank=%d keep rows_on %d (%s rows=%d need_mib=%d card_free_mib=%d)",
                    MARK, rank, before, why, int(rows), int(need_mib), free // MIB)
        gate.topup = False
        return "keep"
    n = ctl.unlend_to(k)
    gate.topup = ctl.lent_from is not None
    if n:
        _gap_note(sched, "lend_return")
        logger.info("%s rank=%d return rows_on %d->%d (%s rows=%d need_mib=%d card_free_mib=%d) "
                    "lent_total=%d returned_total=%d", MARK, rank, before, int(ctl.rows_on), why,
                    int(rows), int(need_mib), free // MIB, ctl.lent_rows, ctl.returned_rows)
    return "return" if n else None


def round_start(sched) -> Optional[str]:
    """Top of ``get_next_batch_to_run``: extend work pending -> the lend goes
    back BEFORE the round's width vote and EXTEND-TRIM read the card (S78: only
    the short extend's price when all pending work is short)."""
    ctl = _controller(sched)
    if ctl is None or not extend_pending(sched):
        return None
    _gate(sched).streak = 0
    if ctl.lent_from is None:
        return None
    rows = short_pending_rows(sched) if keep_short_on() else None
    if rows is not None:
        return _keep_short(sched, ctl, rows, "short extend pending")
    return _give_back(sched, ctl, "extend pending")


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
        if keep_short_on() and ctl.lent_from is not None:
            # S78: the built extend's exact rows price it; a decode batch with a
            # waiting queue reprices the pending work (big/unpriced: full return)
            rows = batch_rows(batch)
            if rows is None:
                rows = short_pending_rows(sched)
            if rows is not None:
                return _keep_short(sched, ctl, rows, "before the short extend")
        return _give_back(sched, ctl, "before the extend")
    if verdict != "lend":
        return None
    topup = ctl.lent_from is not None
    if topup and not gate.topup:
        return None  # one lend per settle (a full lend stays as it is)
    gate.topup = False
    floor = dsv.lend_floor_mib_for_rank(rank)
    free = _card_free_bytes()
    if floor is None or free is None:
        return None
    budget = lend_budget_bytes(free, floor)
    before = int(ctl.rows_on)
    n = ctl.lend(budget)
    if n:
        _gap_note(sched, "lend")
        logger.info("%s rank=%d lend rows_on %d->%d card_free_mib=%d floor_mib=%d budget_mib=%d "
                    "lent_total=%d returned_total=%d", MARK, rank, before, int(ctl.rows_on),
                    free // MIB, int(floor), budget // MIB, ctl.lent_rows, ctl.returned_rows)
    return "lend" if n else None
