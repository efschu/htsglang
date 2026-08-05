# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side driver of the GDN resident-slot ladder (#364 slice 2).

One object per scheduler. Each between-tick boundary it turns the scheduler's
own request bookkeeping into the four id lists ``vacate_plan`` wants, runs the
plan through :class:`GdnSlotExecutor` inside the armed window, and logs what
actually moved.

WHY THE INVENTORY IS BUILT HERE AND NOT IN THE PLANNER
------------------------------------------------------
``vacate_plan`` is deliberately free of scheduler types -- that is what makes
it testable without a server. The mapping from "this fork's request objects"
to "sessions, active, resumed, parked" is exactly the part that is specific to
this scheduler, so it lives on this side of the seam:

* RESIDENTS are the requests that actually hold a state slot, plus the parked
  ones resuming this tick (they are about to need one back). Only residents
  can vacate -- a queued session holds nothing to export.
* ACTIVE is the running batch. Never a victim, whatever the cap says.
* RESUMED is a parked session that is queued to run again: its state must be
  back before its next decode tick, so restores are unconditional.
* PARKED is the executor's own blob inventory.
* NEWCOMERS -- queued sessions with neither a slot nor a blob -- are DEMAND,
  not residents. They are expressed by lowering the cap handed to the planner
  by their count, which is what "make room for k waiting sessions" means in
  the planner's own arithmetic. Passing them as sessions instead would let the
  youngest of them be selected as its own vacate victim.

WHERE THE IDLE SLOT HOLDERS COME FROM (the honest population)
--------------------------------------------------------------
A request in the running batch is by definition active, and a request in the
waiting queue usually holds no slot yet -- so on a bare server there is almost
no idle GDN slot holder and this ladder correctly does nothing. The standing
population it exists for is the SPILLED SESSION SET of the #236/#224 KV
offload: that path moves a session's KV tail to the host and deliberately
leaves its mamba slot resident (stated as an invariant in
kv_session_offload's destination module), so a spilled session is exactly a
live session holding a state slot with no work in the batch. The driver
therefore reads ``kv_session_offload.spills`` as a third inventory source.
Without ``--enable-kv-session-offload`` the ladder still works -- multi-turn
sessions between turns qualify too -- but that is the set that makes a capped
pool pay off, and it is why the two features are tested together.

RANK UNIFORMITY
---------------
Unlike the #287 ladder this driver adds NO collective and needs none: every
input is already rank-replicated by construction (the running batch and the
waiting queue are the same on every TP rank -- the property the admission
limiter's sample relies on), and the pool geometry is identical on every rank
because ``--gdn-resident-state-slots`` is a boot-time cap applied from the
same server args. Two ranks therefore build the same plan from the same
inputs without being asked to agree at runtime. What this driver must never
do is make the plan depend on anything rank-local (free-slot ORDER, device
occupancy, timing); it does not, and the slot each rank picks is allowed to
differ because a state blob is restorable into ANY free slot by design.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "GDN-SLOT-LADDER"


def _rid(req) -> Optional[str]:
    sid = getattr(req, "session_id", None) or getattr(req, "rid", None)
    return str(sid) if sid is not None else None


class GdnSlotRuntime:
    """Drives one scheduler's slot ladder. Constructed lazily, once."""

    def __init__(
        self,
        *,
        mamba_pool,
        mamba_allocator,
        req_to_token_pool,
        resident_slots: int,
        blob_store=None,
        idle_holders_fn=None,
    ):
        from sglang.srt.mem_cache.gdn_slot_executor import GdnSlotExecutor

        self._pool = req_to_token_pool
        self._resident_slots = int(resident_slots)
        self._idle_holders_fn = idle_holders_fn
        self._by_id: Dict[str, object] = {}
        self.rounds = 0
        # The executor's adapters are this object's methods, so it is built
        # here rather than handed in: a caller assembling the pair would have
        # to resolve the same cycle with a mutable attribute.
        self._executor = GdnSlotExecutor(
            mamba_pool=mamba_pool,
            mamba_allocator=mamba_allocator,
            slot_of=self.slot_of,
            bind=self.bind,
            unbind=self.unbind,
            blob_store=blob_store,
        )
        logger.info(
            "%s armed: %d resident state slots, idle-only vacate, blobs stay "
            "off the radix route (#212)",
            LOG_PREFIX,
            self._resident_slots,
        )

    # -- the scheduler-side adapters the executor was given ------------------
    def slot_of(self, session_id: str) -> Optional[int]:
        req = self._by_id.get(session_id)
        idx = getattr(req, "mamba_pool_idx", None) if req is not None else None
        if idx is None:
            return None
        return int(idx)

    def bind(self, session_id: str, slot: int) -> None:
        req = self._by_id.get(session_id)
        if req is None:
            return
        req.mamba_pool_idx = slot
        # Keep the req-index -> mamba-index table the batch builder reads in
        # step with the rebind. Skipping this would leave the request pointing
        # at its OLD slot for one forward -- another session's state, read as
        # if it were this one's.
        pool_idx = getattr(req, "req_pool_idx", None)
        mapping = getattr(self._pool, "req_index_to_mamba_index_mapping", None)
        if pool_idx is not None and mapping is not None:
            mapping[pool_idx] = slot

    def unbind(self, session_id: str) -> None:
        req = self._by_id.get(session_id)
        if req is not None:
            req.mamba_pool_idx = None

    def _idle_holders(self) -> List[object]:
        """Live sessions that hold a state slot outside the running batch."""
        if self._idle_holders_fn is None:
            return []
        try:
            return list(self._idle_holders_fn() or [])
        except Exception as exc:  # noqa: BLE001 -- inventory must never kill a tick
            logger.warning(
                "%s: idle-holder inventory failed (%r); this round sees only "
                "the batch and the queue",
                LOG_PREFIX,
                exc,
            )
            return []

    # -- one between-tick boundary -------------------------------------------
    def on_round(self, running_batch, waiting_queue) -> None:
        from sglang.srt.mem_cache.gdn_slot_ladder import vacate_plan

        # Fast path, and it is the common one: with nothing queued, nothing
        # parked and no idle slot holder there is no demand and no candidate,
        # so the plan is empty by construction. Returning here keeps a
        # steady-state decode loop at one attribute read plus two emptiness
        # checks per iteration instead of rebuilding the inventory.
        waiting = list(waiting_queue or [])
        idle_holders = list(self._idle_holders())
        if not waiting and not idle_holders and not self._executor.parked_ids:
            return
        running = list(getattr(running_batch, "reqs", None) or [])

        self._by_id = {}
        for req in running + waiting + idle_holders:
            sid = _rid(req)
            if sid is not None:
                self._by_id.setdefault(sid, req)

        active_ids = [s for s in (_rid(r) for r in running) if s is not None]
        parked_ids = self._executor.parked_ids
        parked_set = set(parked_ids)

        # A parked session that is queued again is resuming THIS tick.
        resumed_ids = [
            sid
            for sid in (_rid(r) for r in waiting)
            if sid is not None and sid in parked_set
        ]
        resumed_set = set(resumed_ids)

        residents, newcomers = [], 0
        for sid, req in self._by_id.items():
            if getattr(req, "mamba_pool_idx", None) is not None:
                residents.append(req)
            elif sid in resumed_set:
                residents.append(req)
            elif sid not in parked_set:
                newcomers += 1
        if not residents:
            return

        # A parked session that died while parked leaves a blob nobody will
        # ever ask for; drop it here rather than grow the store forever.
        for sid in parked_ids:
            if sid not in self._by_id:
                self._executor.forget(sid)

        # Demand lowers the cap, it does not join the resident set. Floor of 1:
        # the planner requires at least one resident slot, and a server that
        # queued more sessions than it has slots must still run one of them.
        effective_cap = max(1, self._resident_slots - newcomers)
        plan = vacate_plan(
            resident_slots=effective_cap,
            sessions=residents,
            active_ids=active_ids,
            resumed_ids=resumed_ids,
            parked_ids=self._executor.parked_ids,
        )
        if plan.is_empty:
            return

        self.rounds += 1
        with self._executor.between_ticks():
            result = self._executor.run(plan)
        if result.failed:
            logger.warning(
                "%s round %d could not complete every move: %s (skipped by "
                "the planner: %s)",
                LOG_PREFIX,
                self.rounds,
                result.failed,
                plan.skipped or "-",
            )


def build_gdn_slot_executor(scheduler) -> Optional[GdnSlotRuntime]:
    """Construct the driver for one scheduler, or ``None`` when this server
    has no GDN state pool to cap (the flag is validated at argument time, so
    reaching here without a hybrid pool means a non-mamba model: a named
    no-op, not an error)."""
    pool = scheduler.req_to_token_pool
    mamba_pool = getattr(pool, "mamba_pool", None)
    allocator = getattr(pool, "mamba_allocator", None)
    if mamba_pool is None or allocator is None:
        logger.warning(
            "%s: --gdn-resident-state-slots is set but this model has no "
            "GDN/Mamba state pool; the ladder stays off.",
            LOG_PREFIX,
        )
        return None

    def idle_holders():
        # The spilled-session set of #236/#224: live sessions whose KV tail is
        # on the host while their mamba slot stayed resident. Read fresh every
        # round -- the manager is built late in the scheduler's __init__ and
        # sessions enter and leave it constantly.
        #
        # `live_offload_reqs()` and NOT `manager.spills`: a #224-PARKED session
        # is popped out of `spills` at park commit while staying alive and
        # keeping its state slot. Reading `spills` made such a session invisible
        # here, and `on_round` treats an id it cannot find in the inventory as a
        # DEAD session and calls `forget(sid)` on it -- discarding the exported
        # GDN state blob of a session that is coming back. The session then
        # unparks, resumes, and continues with recurrent state that no longer
        # exists: wrong output, no crash, nothing logged.
        manager = getattr(scheduler, "kv_session_offload", None)
        if manager is None:
            return []
        getter = getattr(manager, "live_offload_reqs", None)
        if getter is None:
            return []
        return [req for req in getter() if req is not None]

    return GdnSlotRuntime(
        mamba_pool=mamba_pool,
        mamba_allocator=allocator,
        req_to_token_pool=pool,
        resident_slots=int(mamba_pool.size),
        idle_holders_fn=idle_holders,
    )
