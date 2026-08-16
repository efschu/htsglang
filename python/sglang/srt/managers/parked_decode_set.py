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
It moves such a request out of ``running_batch`` and into a parked set that
``running_bs`` does not include. That is the entire change: ARITHMETIC.

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

``running_bs <= max_running``
    At all times, TP included. At TP entry the parked set re-admits in
    capture-set-sized batches and the remainder stays parked, re-admitting as
    decodes complete. Every pool therefore stays inside the dimensioning it
    was built for; this module never lets more decode run at once than the
    capture set and the pools were sized for.

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
from typing import List

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
        """
        if not self.enabled:
            # Byte-identical to the pre-change gate: the caller's own
            # `limit - running_bs` still decides, and this adds nothing.
            return max(0, min(int(requested), self.max_running - int(running_bs)))
        free_slots = self.slot_pool - len(self._ids) - int(running_bs)
        head = max(0, min(int(requested), free_slots))
        if head <= 0:
            self.last_refusal = (
                f"{LOG_PREFIX} refused early: {len(self._ids)} parked + "
                f"{int(running_bs)} running fills the GDN slot pool of "
                f"{self.slot_pool}; admitting would only be refused later by "
                f"alloc_req_slots. Binding bound: slot pool"
            )
        return head

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
