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
    the same on every rank). ``step`` answers return / lend / hold."""

    __slots__ = ("settle", "streak")

    def __init__(self, settle: int):
        self.settle = max(1, int(settle))
        self.streak = 0

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


def _give_back(sched, ctl, why: str) -> Optional[str]:
    if ctl.lent_from is None:
        return None
    before = int(ctl.rows_on)
    n = ctl.unlend()
    if n:
        logger.info("%s rank=%d return rows_on %d->%d (%s) lent_total=%d returned_total=%d",
                    MARK, _rank(sched), before, int(ctl.rows_on), why, ctl.lent_rows,
                    ctl.returned_rows)
    return "return" if n else None


def round_start(sched) -> Optional[str]:
    """Top of ``get_next_batch_to_run``: extend work pending -> the lend goes
    back BEFORE the round's width vote and EXTEND-TRIM read the card."""
    ctl = _controller(sched)
    if ctl is None or not extend_pending(sched):
        return None
    _gate(sched).streak = 0
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
    verdict = _gate(sched).step(kind)
    rank = _rank(sched)
    if verdict == "return":
        return _give_back(sched, ctl, "before the extend")
    if verdict != "lend" or ctl.lent_from is not None:
        return None
    floor = dsv.lend_floor_mib_for_rank(rank)
    free = _card_free_bytes()
    if floor is None or free is None:
        return None
    budget = lend_budget_bytes(free, floor)
    before = int(ctl.rows_on)
    n = ctl.lend(budget)
    if n:
        logger.info("%s rank=%d lend rows_on %d->%d card_free_mib=%d floor_mib=%d budget_mib=%d "
                    "lent_total=%d returned_total=%d", MARK, rank, before, int(ctl.rows_on),
                    free // MIB, int(floor), budget // MIB, ctl.lent_rows, ctl.returned_rows)
    return "lend" if n else None
