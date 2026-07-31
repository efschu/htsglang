# SPDX-License-Identifier: Apache-2.0
"""Rank-uniform runtime driver of the KV pressure ladder (#287).

``kv_pressure_ladder`` (DESIGN_201 Erg. 9/9b) is a planner: sensor + table
-> plan, no side effects beyond its own bookkeeping. THIS module is the
piece that runs it inside the scheduler loop of every TP rank without ever
committing the bug class behind #94/#194/#259/#312: a ladder transition is
a RANK-UNIFORM state change in the middle of the collective family, so a
rank-local condition in front of a group collective -- or two ranks
honestly disagreeing about the rung -- is a hang, not a slowdown.

THE THREE RULES, IN CODE:

1. REPLICATED INPUTS ONLY. The occupancy sample is the admission limiter's
   sample (``replicated_pool_usage`` rationale, scheduler.py): tokens held
   by live requests and the group-agreed pool capacity, both identical on
   every rank by construction. The sensor, the ladder and the operating-
   point selection are pure functions, so every rank computes the same
   local verdict without a collective.

2. CONSENSUS BOUNDARY, UNCONDITIONAL CADENCE. Transitions are committed
   only every ``consensus_interval``-th round -- gated by the REPLICATED
   round counter, never by the local verdict -- and at that boundary every
   rank enters ONE small MIN-reduction over the CPU gloo group,
   unconditionally, carrying ``[v, -v]`` pairs of its proposal. The
   reduction yields min and max of every field at once: equal = commit,
   unequal = a real desync, and then EVERY rank raises the same loud
   ``KvLadderError`` naming both sides (rule 3). The collective itself is
   the #312 bounded primitive (``bounded_collective``), so a dead peer is
   a named ``PeerLostError`` within the probe interval, not a silent hang.

3. LOUD, NEVER SILENT. A desync raises; a flip logs at WARNING with the
   ``KV-PRESSURE-LADDER`` prefix (the card proof greps for it); an
   actuator that is only PLANNED in Stage 1 says so in the same breath.

ACTUATION (honest inventory). ``admission_cap`` and ``session_offload``
are REAL actuators (the floating admission limiter and the #236 spill
manager exist and are driven here). ``dcp_ratio`` -- the KV-vector flip,
first rung of the relief order -- is REAL when ``--kv-reshard-vectors``
declares a ceiling set covering the rung's operating grid: the flip then
ARMS a #297 phase-boundary reshard (managers/kv_reshard.py) that
physically moves the existing KV and refreshes every owner-bounds snapshot
at the next group-wide idle consensus boundary. Without the flag (or with
an uncovered grid) it stays PLANNED-ONLY exactly as in Stage 1: the flip
selects and logs the target vector, no byte moves. ``kv_spill`` and
``weightless_rank`` rungs remain planned-only. The wired/planned split is
validated at construction and printed at boot, so a ladder never quietly
promises relief it cannot deliver.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from sglang.srt.model_executor.kv_pressure_ladder import (
    HANDOVER_NONE,
    OP_PHASES,
    PHASE_ABORT_STAGE,
    PHASE_DESCEND,
    PHASE_FLIP,
    PHASE_PRE_STAGE,
    KvLadderError,
    KvPressureLadder,
    LadderPlan,
    OccupancySample,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-PRESSURE-LADDER"

#: Consensus payload layout: each field is packed as (value, -value) so ONE
#: MIN-reduction returns (min, -max) per field. Fields, in order: plan phase
#: code, target rung, current rung, ladder epoch.
_PLAN_PHASE_CODES = {
    "none": 0,
    PHASE_PRE_STAGE: 1,
    PHASE_ABORT_STAGE: 2,
    PHASE_FLIP: 3,
    PHASE_DESCEND: 4,
}
_CONSENSUS_FIELDS = ("plan_phase", "target_rung", "current_rung", "epoch")

#: Stage-1 wired actuators. Everything else a relief rung may reference is
#: planned-only and says so at boot and on every flip.
WIRED_RELIEFS = ("admission_cap", "session_offload")


def _encode_proposal(
    phase_code: int, target: int, current: int, epoch: int
) -> List[int]:
    vals: List[int] = []
    for v in (phase_code, target, current, epoch):
        vals.append(int(v))
        vals.append(-int(v))
    return vals


class KvPressureRuntime:
    """Drives one group's pressure ladder from the scheduler loop.

    ``collective_min`` is the injectable consensus channel: it takes the
    packed int payload and returns the element-wise MIN across the TP
    group. Production wiring passes :func:`default_collective_min` bound to
    the scheduler's ``tp_cpu_group``; hermetic tests inject mocks -- the
    desync falsifier injects one that merges DIFFERENT ranks' payloads and
    proves the mismatch raises loudly instead of hanging.
    """

    def __init__(
        self,
        ladder: KvPressureLadder,
        *,
        consensus_interval: int = 8,
        tp_size: int = 1,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        admission_limiter=None,
        spill_fn: Optional[Callable[[int], bool]] = None,
        reshard_arm: Optional[Callable[[tuple], tuple]] = None,
    ):
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if tp_size > 1 and collective_min is None:
            raise ValueError(
                "a multi-rank group needs a consensus channel: pass "
                "collective_min (production: default_collective_min over the "
                "TP CPU group). Running multi-rank without one would turn "
                "the first honest divergence into a collective hang instead "
                "of a loud error."
            )
        self._ladder = ladder
        self._interval = int(consensus_interval)
        self._tp_size = int(tp_size)
        self._collective_min = collective_min
        self._limiter = admission_limiter
        self._spill_fn = spill_fn
        # #297: when the phase-boundary reshard runtime is armed for this
        # group AND its ceiling set covers the ladder's dcp_ratio grid, the
        # builder passes its arm() here and the dcp_ratio rung becomes a
        # REAL actuator (the flip arms a reshard that commits at the next
        # group-wide idle boundary). None = Stage-1 behavior, planned-only.
        self._reshard_arm = reshard_arm
        self._round = 0
        self._pending: List[OccupancySample] = []
        self._epoch = 0
        self._last_noop_reason: Optional[str] = None
        self.desync_checks = 0
        self.transitions = 0

        # Fail fast (backward-compat rule: invalid mappings fail at
        # construction, not at the first pressure episode) on rungs whose
        # WIRED actuator is missing, and inventory the planned-only ones.
        planned_only: List[str] = []
        for step in ladder.table.steps:
            feature = step.relief_feature
            if feature is None:
                continue
            if feature == "admission_cap":
                if admission_limiter is None or not admission_limiter.auto:
                    raise KvLadderError(
                        "ladder rung 'admission_cap' needs the ARMED floating "
                        "admission limiter: start the server with "
                        "--max-running-requests-ceiling (the ceiling arms the "
                        "auto controller the rung actuates). Refusing at boot "
                        "instead of at the first pressure episode."
                    )
            elif feature == "session_offload":
                if spill_fn is None:
                    raise KvLadderError(
                        "ladder rung 'session_offload' needs the KV session "
                        "offload manager: start the server with "
                        "--enable-kv-session-offload. Refusing at boot "
                        "instead of at the first pressure episode."
                    )
            elif feature == "dcp_ratio" and reshard_arm is not None:
                # #297 wired: the flip arms a physical reshard.
                pass
            else:
                planned_only.append(feature)
        self.planned_only_reliefs = tuple(planned_only)
        logger.info(
            "%s armed: %d rungs (%s), consensus every %d rounds over %d " "rank(s)%s",
            LOG_PREFIX,
            len(ladder.table),
            ", ".join(s.name for s in ladder.table.steps),
            self._interval,
            self._tp_size,
            (
                "; PLANNED-ONLY reliefs (Stage 1, no actuator wired): "
                + ", ".join(planned_only)
                if planned_only
                else ""
            ),
        )

    # -- state ---------------------------------------------------------------
    @property
    def ladder(self) -> KvPressureLadder:
        return self._ladder

    @property
    def round_index(self) -> int:
        return self._round

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- the per-round hook --------------------------------------------------
    def on_round(
        self,
        held_tokens: int,
        capacity_tokens: int,
        running_bs: int,
        phase: str,
    ) -> Optional[LadderPlan]:
        """One scheduler round. EVERY argument must be replicated across the
        TP group (held tokens of live requests, group-agreed capacity, batch
        size, derived phase) -- the caller's obligation, stated here because
        it is the whole uniformity argument of rule 1.

        Samples every round; commits only at consensus boundaries. Returns
        the committed plan at a boundary (``None`` between boundaries).
        """
        if phase not in OP_PHASES:
            raise ValueError(
                f"unknown execution phase {phase!r}; known: {', '.join(OP_PHASES)}"
            )
        self._round += 1
        if capacity_tokens > 0:
            self._pending.append(
                OccupancySample(
                    round_index=self._round,
                    used_tokens=max(0, int(held_tokens)),
                    total_tokens=int(capacity_tokens),
                )
            )
        # Rule 2: the cadence gate is the REPLICATED round counter -- never
        # the local verdict, never the occupancy. Every rank passes or skips
        # this line in the same round.
        if self._round % self._interval != 0:
            return None

        plan = self._ladder.on_pressure_boundary(
            occupancy_series=self._pending, at_round_boundary=True
        )
        self._pending = []
        self._consensus_check(plan)
        return self._commit(plan, running_bs, phase)

    # -- rule 2: the consensus channel ---------------------------------------
    def _consensus_check(self, plan: LadderPlan) -> None:
        """MIN-reduce the local proposal; equal fields commit, unequal fields
        raise the SAME loud error on every rank (all ranks see the same
        reduced values, so the failure itself is rank-uniform)."""
        if self._tp_size <= 1 or self._collective_min is None:
            return
        self.desync_checks += 1
        phase_code = _PLAN_PHASE_CODES.get(plan.phase, 0)
        vals = _encode_proposal(
            phase_code, plan.target_rung, plan.current_rung, self._epoch
        )
        reduced = self._collective_min(vals)
        if len(reduced) != len(vals):
            raise KvLadderError(
                f"consensus channel returned {len(reduced)} values for a "
                f"{len(vals)}-value payload; the channel contract is "
                f"element-wise MIN of the packed proposal."
            )
        mismatches = []
        for i, name in enumerate(_CONSENSUS_FIELDS):
            lo = reduced[2 * i]
            hi = -reduced[2 * i + 1]
            if lo != hi:
                mismatches.append(f"{name}: min={lo} max={hi}")
        if mismatches:
            raise KvLadderError(
                f"{LOG_PREFIX} DESYNC at round {self._round}: the TP ranks "
                f"proposed different ladder transitions ({'; '.join(mismatches)}; "
                f"this rank: phase={plan.phase!r} target={plan.target_rung} "
                f"current={plan.current_rung} epoch={self._epoch}). A ladder "
                f"that disagrees across ranks must fail loudly HERE, before "
                f"any rank touches a group collective under the wrong "
                f"geometry -- continuing would be the #94/#194/#259 hang."
            )

    # -- rule 3 + actuation ---------------------------------------------------
    def _commit(self, plan: LadderPlan, running_bs: int, phase: str) -> LadderPlan:
        if plan.blocked is not None or plan.is_noop:
            reason = plan.blocked or plan.reason
            if reason != self._last_noop_reason:
                logger.info("%s hold: %s", LOG_PREFIX, reason)
                self._last_noop_reason = reason
            return plan
        self._last_noop_reason = None
        self._ladder.apply(plan)
        self._epoch += 1
        self.transitions += 1
        fill = (
            plan.reading.occupancy
            if plan.reading is not None and plan.reading.occupancy is not None
            else 0.0
        )
        if plan.phase == PHASE_FLIP:
            self._actuate_flip(plan, running_bs, phase, fill)
        elif plan.phase == PHASE_DESCEND:
            logger.warning(
                "%s DESCEND rung %d -> %d (epoch %d): %s",
                LOG_PREFIX,
                plan.current_rung,
                plan.target_rung,
                self._epoch,
                plan.reason,
            )
            self._release_admission(plan)
        elif plan.phase in (PHASE_PRE_STAGE, PHASE_ABORT_STAGE):
            # Erg. 9b, Stage 1: bookkeeping only. The shadow copy itself is
            # GPU-phase work; PRE-STAGING MUST NOT TOUCH THE ACTIVE PATH
            # (byte-identity gate), and a log line plus ladder bookkeeping
            # is exactly that.
            logger.warning(
                "%s %s (epoch %d): %s [Stage 1: bookkeeping only, no shadow "
                "bytes move]",
                LOG_PREFIX,
                plan.phase.upper(),
                self._epoch,
                plan.reason,
            )
        return plan

    def _actuate_flip(
        self, plan: LadderPlan, running_bs: int, phase: str, fill: float
    ) -> None:
        step = self._ladder.table[plan.target_rung]
        detail = ""
        grid = step.operating_grid
        if grid is not None and phase in grid.phases:
            point = grid.select(phase, min(max(fill, 0.0), 1.0))
            detail = (
                f"; operating point ({phase}, fill {fill:.2f}): kv vector "
                f"{list(point.kv_vector)} at layer split "
                f"{list(point.layer_split)} [{point.provenance}]"
            )
        wired: str
        if (
            step.relief_feature == "dcp_ratio"
            and self._reshard_arm is not None
            and grid is not None
            and phase in grid.phases
        ):
            point = grid.select(phase, min(max(fill, 0.0), 1.0))
            ok, msg = self._reshard_arm(tuple(point.kv_vector))
            wired = (
                f"actuator WIRED (#297): reshard to {list(point.kv_vector)} "
                f"{'armed' if ok else 'REFUSED'} -- {msg}"
            )
        elif step.relief_feature == "admission_cap" and self._limiter is not None:
            changed = self._limiter.throttle(running_bs)
            wired = (
                f"actuator WIRED: admission limit -> {self._limiter.current} "
                f"(changed={changed})"
            )
        elif step.relief_feature == "session_offload" and self._spill_fn is not None:
            spilled = bool(self._spill_fn(running_bs))
            wired = f"actuator WIRED: session spill requested (spilled={spilled})"
        elif step.relief_feature is not None:
            wired = (
                f"actuator PLANNED-ONLY (Stage 1): relief "
                f"{step.relief_feature!r} names its target but moves nothing "
                f"-- the physical KV-vector/spill handover is #297"
            )
        elif step.handover != HANDOVER_NONE:
            wired = (
                f"actuator PLANNED-ONLY (Stage 1): geometry/external rung "
                f"with handover {step.handover!r} is a named GPU-phase item"
            )
        else:
            wired = "no actuator declared"
        logger.warning(
            "%s FLIP rung %d -> %d (%s, epoch %d, occupancy %.3f): %s%s. %s",
            LOG_PREFIX,
            plan.current_rung,
            plan.target_rung,
            step.name,
            self._epoch,
            fill,
            plan.reason,
            detail,
            wired,
        )

    def _release_admission(self, plan: LadderPlan) -> None:
        """Descending below the admission rung gives the float back
        geometrically (mirror of the limiter's own release step), never in
        one jump."""
        if self._limiter is None or not self._limiter.auto:
            return
        table = self._ladder.table
        admission_rung = next(
            (
                i
                for i, s in enumerate(table.steps)
                if s.relief_feature == "admission_cap"
            ),
            None,
        )
        if admission_rung is None or plan.target_rung >= admission_rung:
            return
        lim = self._limiter
        if lim.current < lim.start:
            step_up = max(1, lim.current // 8)
            lim.set_limit(min(lim.start, lim.current + step_up))
            logger.info(
                "%s admission release on descend: limit -> %d (start %d)",
                LOG_PREFIX,
                lim.current,
                lim.start,
            )


def default_collective_min(tp_cpu_group) -> Callable[[List[int]], List[int]]:
    """The production consensus channel: one bounded MIN all-reduce of the
    packed proposal over the TP CPU (gloo) group.

    Uses the #312 bounded primitive when the HTCCL liveness table is
    published (dead peer -> loud ``PeerLostError`` instead of a hang); the
    primitive itself degrades to the plain blocking call when liveness is
    off, which is the fork's pre-#312 behavior."""
    import torch
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.htccl_liveness import (
        bounded_collective,
    )

    def _reduce(vals: List[int]) -> List[int]:
        t = torch.tensor(vals, dtype=torch.int64)
        bounded_collective(
            lambda: dist.all_reduce(
                t, op=dist.ReduceOp.MIN, group=tp_cpu_group, async_op=True
            ),
            "kv_pressure_ladder.consensus",
        )
        return [int(v) for v in t.tolist()]

    return _reduce


def build_kv_pressure_runtime(scheduler) -> Optional[KvPressureRuntime]:
    """Construct the runtime for one scheduler group, or ``None`` when the
    ladder flag is unset (today's behavior: nothing is constructed, no
    sample is taken, no collective is added)."""
    from sglang.srt.model_executor.kv_pressure_ladder import (
        build_ladder_from_server_args,
    )

    server_args = scheduler.server_args
    ladder = build_ladder_from_server_args(server_args)
    if ladder is None:
        return None
    tp_size = getattr(server_args, "tp_size", 1)
    spill_fn = None
    if scheduler.kv_session_offload is not None:

        def spill_fn(_running_bs: int) -> bool:
            batch = scheduler.running_batch
            if batch is None or batch.is_empty():
                return False
            return bool(scheduler.kv_session_offload.try_spill(batch))

    collective = default_collective_min(scheduler.tp_cpu_group) if tp_size > 1 else None

    # #297: bind the dcp_ratio rung to the phase-boundary reshard runtime,
    # but only when the declared ceiling set covers EVERY vector the rung's
    # operating grid can select -- an actuator that would refuse its own
    # targets at flip time must not be inventoried as wired.
    reshard_arm = None
    reshard_rt = getattr(scheduler, "kv_reshard_runtime", None)
    if reshard_rt is not None:
        grid_vectors = set()
        for step in ladder.table.steps:
            if step.relief_feature == "dcp_ratio" and step.operating_grid is not None:
                for point in step.operating_grid.points:
                    grid_vectors.add(tuple(point.kv_vector))
        uncovered = sorted(
            v for v in grid_vectors if v not in set(reshard_rt.allowed_vectors)
        )
        if uncovered:
            logger.warning(
                "%s dcp_ratio rung stays PLANNED-ONLY despite "
                "--kv-reshard-vectors: operating-grid vectors %s are not in "
                "the declared ceiling set %s (add them and reboot to wire "
                "the rung).",
                LOG_PREFIX,
                uncovered,
                list(reshard_rt.allowed_vectors),
            )
        else:

            def reshard_arm(vec, _rt=reshard_rt):
                return _rt.arm(vec, source="pressure_ladder")

    return KvPressureRuntime(
        ladder,
        consensus_interval=int(
            getattr(server_args, "kv_pressure_consensus_interval", 8)
        ),
        tp_size=tp_size,
        collective_min=collective,
        admission_limiter=scheduler.admission_limiter,
        spill_fn=spill_fn,
        reshard_arm=reshard_arm,
    )
