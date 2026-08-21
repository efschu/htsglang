# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#677 phase 1: a carrier awaiting its decode window stops COUNTING.

THE WEDGE WAS A COUNTING DEFECT, and the measurement is what says so. At
2026-08-16 06:04 the instance held twelve GDN slots with eight of them FREE,
four running requests against a cap of four, and 403779 tokens of prefill that
could not be admitted. Freeing a GDN slot would have relieved nothing.

Admission is::

    min(pp_max_micro_batch_size, admission_limiter.current) - running_bs
    then min(..., req_to_token_pool.available_size())

``HybridReqToTokenPool`` does not override ``available_size``, so the second
term is the REQUEST-slot count; the mamba allocator is consulted only later,
inside ``alloc_req_slots``. Neither term sees the GDN pool. With
``running_bs == 4 == cap`` the gate returned 0 and nothing further ran.

What blocked admission was that four requests PP is FORBIDDEN to decode --
strict purity, ``decode_allowed_in_pp`` is False -- were counted against the
concurrency cap for the whole residency. They could not progress and they
could not be replaced.

WHAT THIS MODULE DOES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------
It NAMES such a request as parked and subtracts it from the number the
concurrency cap is compared against. That is the entire change: ARITHMETIC.

The carrier does not leave ``running_batch``. An earlier draft of this
module said it did, and moving it would have been the whole risk: every
consumer that enumerates resident requests -- ``_live_reqs`` (whose omission
leaves a carrier's freshest KV behind at the reshard, silently wrong
context), ``orphan_resident_reqs``, the draft bootstrap's
``_reachable_batches``, ``idle_blockers``, ``abort_request`` -- would each
have needed teaching, and each is a separate chance to be silently wrong.
Leaving the request exactly where every one of them already looks is what
makes this change a counting change and nothing else.

Nothing moves. The carrier keeps its GDN slot and its KV -- which is exactly
the KV that would have been resident anyway -- and its ``req_to_token`` row.
No state is exported, no tier is touched, no address changes. That is the
point: phase 1 adds no correctness surface from the #450/#444 conv-cache
verify-write family, the #461 ``DEVICE_BOUND`` law, or #551 GDN-Vacate x kvso.
Exporting the state blob to host RAM is phase 2 and is gated behind the #551
read; at twelve slots against four running there is headroom for eight parked
carriers before it can matter.

EVERY BOUND IS SOLVED FROM BOOT DIMENSIONING
--------------------------------------------
Neither of these is a tunable, and phase 1 RAISES NOTHING:

``parked + running <= slot_pool``
    The GDN slot pool is the honest capacity limit for carriers, because a
    parked request still holds its slot. A prefill that would exceed it is
    refused HERE, early and by name -- ``alloc_req_slots`` would refuse it
    late anyway, and a late refusal is the raise #679/#681/#684 spent three
    tasks making survivable. An early named refusal is the same verdict
    delivered where it can still be scheduled around.

CORRECTED WHEN WIRED (2026-08-16). The unwired core claimed a second bound,
``running_bs <= max_running`` at all times including TP, to be held by
re-admitting in capture-set-sized batches. THE SHIPPED WIRING DOES NOT HOLD
THAT BOUND, and says so rather than leaving the claim standing:

    At TP entry every carrier decodes, so the decode batch is the whole
    parked bundle -- up to ``slot_pool`` requests, not ``max_running``.

Holding the tighter bound would have required SPLITTING a resident batch,
and no split primitive exists: ``filter_batch`` drops requests and
``merge_batch`` joins them. Building one means surgery in the
``running_batch = last_batch`` aliasing path that already produced #631
defects J.1 and J.3, defect M, and the 2026-08-09 self-merge that doubled a
batch 2^23 -> 2^25 and killed all three ranks. Phase 1 exists precisely to
avoid new movement surface, so it declines to open that one for a bound the
pools do not actually need:

  * the mamba/GDN pool IS ``slot_pool`` (12 here) and is the bound this
    module enforces, so the decode batch can never exceed the state pool;
  * the decode CUDA graph is captured for batch sizes up to 24 on this
    recipe, so a 12-wide bundle is inside the capture set, not outside it;
  * a parked carrier already holds its KV whether or not it is stepping,
    so widening concurrency from 4 to 12 adds only the OUTPUT tokens --
    on the measured 4x25625 shape roughly 1300 KV rows, not a bundle's.

``max_running`` therefore survives here only as what it always was, the
CONCURRENCY CAP that :meth:`carrier_discount` stops charging to carriers
that cannot decode. If a future recipe puts ``slot_pool`` above the decode
capture set, THIS is the paragraph that is falsified and the split has to
be built.

THE ONE BOOKKEEPING EDGE, stated because it is the thing to get wrong: a
parked request is still RESIDENT. It holds KV and a ``req_to_token`` row, so
pressure ladders and retract paths that ask "what is on this card" must see
it. :attr:`resident_ids` exists for exactly those callers, so a parked
carrier is never double-counted (it is out of ``running_batch``) and never
invisible (it is in the resident set).

A PARK FAILURE DEGRADES TO THE SAFETY NET. When parking is disabled or the
slot pool is full, the arithmetic here is byte-identical to the pre-change
gate: the request stays a carrier, keeps counting, and the #677 progress exit
underneath still sees the stall and breaks it. The failure mode of this module
is the behaviour it replaced, never a wedge.

NOTHING HERE IMPORTS TORCH. The set, the bounds and the receipts are ordinary
arithmetic over ids, testable on CPU with no CUDA present.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from sglang.srt.managers.log_cycle_collapse import CycleCollapse

logger = logging.getLogger(__name__)

LOG_PREFIX = "PARKED-DECODE"

__all__ = ["ParkedDecodeSet", "ParkedSetFull", "LOG_PREFIX"]


class ParkedSetFull(RuntimeError):
    """A park that would exceed the GDN slot pool.

    RAISED, not returned, because the caller must not proceed as if the
    request were parked: it is still a carrier, still counted, and the
    progress exit is what handles it from there.
    """


class ParkedDecodeSet:
    """Requests that finished prefill and await their decode window.

    FIFO, because a carrier that keeps losing its place is a carrier that
    starves, and the wedge this replaces was itself a starvation.
    """

    def __init__(
        self,
        slot_pool: int,
        max_running: int,
        *,
        enabled: bool = True,
    ) -> None:
        self.slot_pool = int(slot_pool)
        self.max_running = int(max_running)
        self.enabled = bool(enabled)
        self._ids: List[str] = []
        self.last_receipt: str = ""
        self.last_refusal: str = ""
        #: Counters a summary can read without re-deriving them.
        self.parked_total = 0
        self.readmitted_total = 0
        #: The reconcile receipt is per-round and level-triggered, so a
        #: steady cadence restates it forever -- 140184 lines and 25 MB of
        #: boot instr14. See :meth:`_emit_reconcile_receipt`.
        self._receipt_collapse = CycleCollapse()
        self._suppressed_summary: Tuple[int, int] = (0, 0)

    # -- shape ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    @property
    def resident_ids(self) -> List[str]:
        """Parked requests are RESIDENT: they hold KV and a req_to_token row.

        Separate from :attr:`ids` by name rather than by value so a caller
        asking "what is on this card" is asking the question it means.
        """
        return list(self._ids)

    @property
    def resident_count(self) -> int:
        return len(self._ids)

    # -- the two bounds ---------------------------------------------------

    def admission_headroom(self, running_bs: int, requested: int) -> int:
        """New prefills admissible without breaching the GDN slot pool.

        The cap on CONCURRENT DECODE is not applied here -- that is
        ``max_running`` and it governs re-admission, not admission. What
        governs admission is the slot pool, because every admitted request
        will eventually need a slot and a parked one is already holding its.

        ``running_bs`` IS THE WHOLE RESIDENT SET, PARKED INCLUDED. A parked
        carrier never leaves ``running_batch`` (see the module header's
        "nothing moves"), so it is already inside the number the scheduler
        passes here. Subtracting :attr:`_ids` as well would charge every
        parked carrier to the pool TWICE and refuse admission at half the
        real capacity -- which is the same class of miscount this module
        exists to remove, only pointing the other way.
        """
        if not self.enabled:
            # INERT, and it must be the identity rather than "the same bound
            # by another route". The scheduler applies this as ONE MORE min()
            # on top of the gate it already had, so anything but `requested`
            # here would TIGHTEN the disabled path -- the one path whose whole
            # purpose is to be the pre-change behaviour byte for byte.
            return int(requested)
        free_slots = self.slot_pool - int(running_bs)
        head = max(0, min(int(requested), free_slots))
        if head <= 0:
            self.last_refusal = (
                f"{LOG_PREFIX} refused early: {len(self._ids)} parked of "
                f"{int(running_bs)} resident fills the GDN slot pool of "
                f"{self.slot_pool}; admitting would only be refused later by "
                f"alloc_req_slots. Binding bound: slot pool"
            )
        return head

    def carrier_discount(self) -> int:
        """Resident requests the CONCURRENCY CAP must stop counting.

        This is the whole of phase 1. ``max_running_requests`` bounds how
        much decode may run AT ONCE; a carrier the phase forbids to decode
        is not running anything, so charging it to that cap reserves
        concurrency nobody can spend. At 2026-08-16 06:04 four such
        carriers held the entire cap of four and 403779 tokens of prefill
        could not be admitted behind them.

        Zero when parking is disabled, which is what makes the disabled
        path byte-identical to the gate this replaces.
        """
        return len(self._ids) if self.enabled else 0

    def sync_carriers(self, rids: Sequence[str], running_bs: int) -> None:
        """Reconcile the parked set with what the phase currently forbids.

        LEVEL-TRIGGERED, NOT EDGE-TRIGGERED, and that is a durability
        argument rather than a style choice. The scheduler evaluates the
        purity predicate once per round and only on rounds that reach the
        decode branch; an edge-triggered park would have to observe every
        arrival and every completion to stay true, and any missed edge
        leaves the discount permanently wrong in one direction. Restating
        the whole set from the resident ids cannot drift: a carrier that
        finished, aborted or was retracted is simply absent from ``rids``
        on the next reconcile and leaves the set with it.

        Passing an empty ``rids`` is how the caller says "this phase
        decodes" -- every carrier is released and the cap counts the full
        resident set again, exactly as it did before this module existed.
        """
        if not self.enabled:
            return
        want = [str(r) for r in rids]
        seen = set()
        deduped: List[str] = []
        for rid in want:
            if rid in seen:
                continue
            seen.add(rid)
            deduped.append(rid)
        current = list(self._ids)
        if deduped == current:
            return
        gone = [rid for rid in current if rid not in seen]
        self._ids = [rid for rid in current if rid in seen]
        added: List[str] = []
        for rid in deduped:
            if rid in self._ids:
                continue
            # The slot-pool bound is checked against the RESIDENT set, which
            # already contains this carrier -- it is not being admitted here,
            # only re-labelled. A refusal would therefore be meaningless, so
            # the bound is asserted rather than enforced: exceeding it means
            # alloc_req_slots handed out more slots than the pool has, which
            # is a defect upstream of this module and must not be hidden here.
            self._ids.append(rid)
            self.parked_total += 1
            added.append(rid)
        if gone:
            self.readmitted_total += len(gone)
        if added or gone:
            self.last_receipt = (
                f"{LOG_PREFIX} carriers {len(self._ids)} parked "
                f"(+{len(added)} -{len(gone)}) of {int(running_bs)} resident; "
                f"the concurrency cap of {self.max_running} now counts "
                f"{max(0, int(running_bs) - len(self._ids))}, and the binding "
                f"bound on new prefill is the GDN slot pool {self.slot_pool}"
            )
            self._emit_reconcile_receipt(len(self._ids), int(running_bs))

    def _emit_reconcile_receipt(self, parked: int, running_bs: int) -> None:
        """Log :attr:`last_receipt`, unless the reconcile is repeating itself.

        WHY THIS EXISTS. Boot instr14 wrote 140184 of these lines in ten
        minutes -- 25373304 bytes, 49% of a 51 MB log, at 13.03 MB/min peak.
        ``sync_carriers`` is LEVEL-TRIGGERED and runs once per scheduler
        round, so a steady phase-flip cadence restates the same handful of
        reconciles forever.

        AND WHY IT IS A CYCLE DETECTOR. On instr14 no two consecutive
        receipts are equal; they cycle with period 3::

            carriers 4 parked (+4 -2) of 4 resident; ...
            carriers 2 parked (+2 -4) of 2 resident; ...
            carriers 2 parked (+2 -2) of 2 resident; ...

        A "same as the last line" test removes 3 lines out of 13545 on the
        measured tail. Recognising the cycle removes all but 5.

        THIS RECEIPT IS RANK-LOCAL and must not be read as congruence
        evidence: ``slot_pool`` is this rank's mamba allocator size and
        ``running_bs`` this rank's resident count, so two ranks may
        legitimately differ. It is collapsed on its own terms -- the
        decision is a pure function of the receipt text, hence deterministic
        for a given receipt sequence, which is all this line needs. The
        cross-rank law that governs the #788 admission trace is stated in
        ``log_cycle_collapse`` and applies there, not here.

        Only the reconcile receipt is throttled. ``park``, ``readmit`` and
        ``evacuate`` are EDGE receipts naming a specific request, so they
        never repeat and must stay unconditional.
        """
        collapse = self._receipt_collapse.observe(self.last_receipt)
        if not collapse.emit:
            # Recorded before the roll-up reads it, so a PERIODIC roll-up
            # names the pass it was just handed and a FLUSHED one names the
            # last pass it actually swallowed -- never the line that broke
            # the cycle.
            self._suppressed_summary = (parked, running_bs)
        if collapse.rollup:
            # Silence must not be ambiguous. Deliberately NOT spelled
            # "PARKED-DECODE carriers": prove_park_677.sh counts receipts
            # with that grep, and a roll-up matching it would inflate the
            # very count it reports a reduction in.
            last_parked, last_running = self._suppressed_summary
            logger.info(
                "%s suppressed=%d reconcile receipts repeating a %d-pass "
                "cycle (last: %d parked of %d resident) since the last "
                "emitted line",
                LOG_PREFIX,
                collapse.rollup,
                collapse.period,
                last_parked,
                last_running,
            )
        if collapse.emit:
            logger.info("%s", self.last_receipt)

    def park(self, req_id: str, running_bs: int) -> bool:
        """Move a finished-prefill carrier out of the counted set.

        Returns False when parking is disabled (the request stays a carrier).
        Raises :class:`ParkedSetFull` when the slot pool cannot hold it.
        """
        if not self.enabled:
            return False
        if len(self._ids) + int(running_bs) >= self.slot_pool:
            self.last_refusal = (
                f"{LOG_PREFIX} cannot park {req_id}: {len(self._ids)} parked + "
                f"{int(running_bs)} running would exceed the GDN slot pool of "
                f"{self.slot_pool}. The request stays a carrier and keeps "
                f"counting, so the #677 progress exit still covers it"
            )
            raise ParkedSetFull(self.last_refusal)
        self._ids.append(str(req_id))
        self.parked_total += 1
        self.last_receipt = (
            f"{LOG_PREFIX} parked {req_id}: finished prefill, no decode window "
            f"in this phase; parked set now {len(self._ids)} of the "
            f"{self.slot_pool}-slot GDN pool ({int(running_bs)} running). "
            f"Binding bound: slot pool {self.slot_pool}"
        )
        logger.info("%s", self.last_receipt)
        return True

    # -- re-admission -----------------------------------------------------

    def readmit_plan(self, running_bs: int) -> List[str]:
        """Which parked requests may decode now, FIFO, without breaching the
        concurrency cap. Pure: the caller commits with :meth:`readmit`."""
        room = max(0, self.max_running - int(running_bs))
        return list(self._ids[:room])

    def readmit(self, running_bs: int) -> List[str]:
        """Commit :meth:`readmit_plan` and hand the ids back to the caller."""
        plan = self.readmit_plan(running_bs)
        if not plan:
            return []
        self._ids = self._ids[len(plan) :]
        self.readmitted_total += len(plan)
        self.last_receipt = (
            f"{LOG_PREFIX} re-admit {', '.join(plan)}: {len(plan)} of "
            f"{len(plan) + len(self._ids)} parked enter decode "
            f"({int(running_bs)} already running, cap {self.max_running}); "
            f"{len(self._ids)} stay parked and follow as decodes complete. "
            f"Binding bound: max_running {self.max_running}"
        )
        logger.info("%s", self.last_receipt)
        return plan

    # -- shutdown ---------------------------------------------------------

    def evacuate(self, reason: str) -> List[str]:
        """Hand every parked request back, so none is silently stranded.

        The set is PROCESS-LOCAL: a crash loses it. A request that looks
        admitted and never completes is the worst outcome available here, so
        the shutdown path takes them all back and fails them loudly.
        """
        out = list(self._ids)
        self._ids = []
        if out:
            self.last_receipt = (
                f"{LOG_PREFIX} evacuating {len(out)} parked request(s) "
                f"({', '.join(out)}) because: {reason}. They hold KV and a "
                f"req_to_token row, so they must be failed or re-queued "
                f"explicitly -- never left looking admitted"
            )
            logger.warning("%s", self.last_receipt)
        return out
