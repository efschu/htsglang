# SPDX-License-Identifier: Apache-2.0
"""Runtime per-card VRAM budget dial + KV capacity re-raise (#330).

Two halves, one mechanism (DESIGN_330_vram_dial.md):

(a) BUDGET DIAL. A running server can be told to lower one card's VRAM
    budget (absolute MiB, release MiB, or a fraction of the dialable span)
    and later raise it back. The dialable portion is exactly the VMM-backed
    KV tail (target full-attention pool + draft pool); weights, graphs, the
    GDN state pool and workspaces are the PINNED FLOOR, measured once per
    rank and gathered so every rank knows every floor. Released pages are
    decommitted at the driver level (cuMemUnmap/cuMemRelease) -- visible in
    nvidia-smi and allocatable by another process -- and the freed budget is
    reported into the #305-M1 VRAM ledger so external tenants can claim it
    honestly.

(b) C RE-RAISE (C-Wiederanhebung). #297 resharding moves KV bytes between
    vectors under a FIXED ceiling; the capacity a better vector could fund
    (#320: 6.18x on 2,11,10) stays stranded. This runtime recomputes the
    achievable group ceiling from (budgets, floors, current vector) at every
    consensus boundary and GROWS max_total_num_tokens automatically when the
    vector allows more: physical pages are committed into the boot VA
    reservation, the allocator's slot space is extended in place, and every
    capacity snapshot is refreshed. No tensor moves, no CUDA-graph
    re-capture.

THE #287/#297 DISCIPLINE, RESTATED FOR CAPACITY:

1. REPLICATED INPUTS ONLY. Budgets arrive via the broadcast RPC, floors and
   per-rank byte coefficients via a one-time gather at build, the vector via
   the process-global ``get_cp_token_ratios()``. ``C_target`` is a pure
   function of these, so every rank computes the same value locally.
2. CONSENSUS BOUNDARY, UNCONDITIONAL CADENCE. Every
   ``consensus_interval``-th round, gated by the replicated round counter,
   every rank enters ONE bounded MIN-reduction (#312) with
   ``(armed, ready, epoch, op_seq, C_target)``. ``epoch`` is
   equality-checked always; ``op_seq`` skew is LEGAL (uniform hold);
   ``C_target`` is equality-checked once every rank is armed at the same
   ``op_seq`` -- a mismatch raises the same :class:`KvCapacityError` on
   every rank, never a hang.
3. LOUD, NEVER SILENT. Commits log ``VRAM-DIAL`` records with the exact
   numbers; refusals carry the floor arithmetic; a shrink is never
   spontaneous (it flushes the radix cache and only an explicit dial
   authorizes that).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "VRAM-DIAL"
MIB = 1024 * 1024


def corridor_law_floor_bytes() -> int:
    """The user's corridor law in bytes, read from ``CorridorGuard``.

    #658, register C18: THE DIAL AND THE GUARD MUST NOT CARRY TWO FLOORS.
    Before this, ``_measure_local_floor_bytes`` derived a card-level floor of
    its own (NVML used minus our VMM-backed KV) and every capacity
    computation spent against ``budget_bytes - floor_bytes``, so a dial could
    plan a KV ceiling that consumed the last free byte on the card. The guard
    would then have to claw those bytes back at an allocation, which is the
    C17 shape: two mechanisms, one resource, different floors.

    IMPORTED FROM THE GUARD RATHER THAN RESTATED. A second literal ``1024``
    in this module would be a second source of truth, and the one thing this
    chain has repeatedly proved is that two constants that must agree
    eventually do not.
    """
    from sglang.srt.managers.corridor_guard import DEFAULT_FLOOR_MIB

    return int(DEFAULT_FLOOR_MIB) * MIB


class KvCapacityError(RuntimeError):
    """Loud, rank-uniform failure of a capacity operation."""


# ---------------------------------------------------------------------------
# Boot capacity plan + participant registry (process-local).
#
# _apply_token_constraints (target runner) computes the per-vector achievable
# ceilings C_v with the #297 min-reduce; pool construction (target AND draft
# runners, later in the same process) needs them to size the VA reservation;
# the scheduler-side runtime needs them again. A module singleton is the
# handshake -- same pattern as the process-global CP token ratios.
# ---------------------------------------------------------------------------


@dataclass
class BootCapacityPlan:
    """Per-vector achievable ceilings from the boot min-reduce."""

    vectors: Tuple[Tuple[int, ...], ...]
    caps: Dict[Tuple[int, ...], int]

    @property
    def max_cap(self) -> int:
        return max(self.caps.values())

    def reserve_rows_for_rank(self, rank: int) -> int:
        """VA rows rank ``rank`` must reserve to reach any vector's ceiling."""
        from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

        rows = 0
        for vec in self.vectors:
            rows = max(
                rows,
                dcp_compact_pool_rows(self.caps[tuple(vec)], sum(vec), vec[rank]),
            )
        return rows


@dataclass
class DialParticipant:
    """One VMM-backed KV pool taking part in capacity changes."""

    runner: object  # ModelRunner (target or draft)
    pool: object  # HybridLinearKVPool with post_capture_active
    is_target: bool
    reserved_tokens: int  # VA reservation: rows (target) / tokens (draft)


_BOOT_PLAN: Optional[BootCapacityPlan] = None
_PARTICIPANTS: List[DialParticipant] = []


def set_boot_capacity_plan(
    vectors: Sequence[Sequence[int]], caps: Dict[Tuple[int, ...], int]
) -> None:
    global _BOOT_PLAN
    _BOOT_PLAN = BootCapacityPlan(
        vectors=tuple(tuple(int(x) for x in v) for v in vectors),
        caps={tuple(int(x) for x in k): int(v) for k, v in caps.items()},
    )
    logger.info(
        "%s boot capacity plan: %s",
        LOG_PREFIX,
        {k: v for k, v in _BOOT_PLAN.caps.items()},
    )


def get_boot_capacity_plan() -> Optional[BootCapacityPlan]:
    return _BOOT_PLAN


def register_dial_participant(
    runner, pool, *, is_target: bool, reserved_tokens: int
) -> None:
    _PARTICIPANTS.append(
        DialParticipant(
            runner=runner,
            pool=pool,
            is_target=is_target,
            reserved_tokens=int(reserved_tokens),
        )
    )


def get_dial_participants() -> List[DialParticipant]:
    return list(_PARTICIPANTS)


def _pool_row_nbytes(pool) -> int:
    """Per-row (target) / per-token (draft) K+V bytes over all full layers."""
    full = pool.full_kv_pool
    total = 0
    for k, v in zip(full.k_buffer, full.v_buffer):
        total += int(k[0].numel()) * k.element_size()
        total += int(v[0].numel()) * v.element_size()
    return total


# ---------------------------------------------------------------------------
# The runtime.
# ---------------------------------------------------------------------------

#: Consensus payload fields (packed as (x, -x) pairs; one MIN yields min and
#: max per field). armed/ready are MIN-semantics; epoch always equality;
#: op_seq skew = legal hold; C_target equality once armed at a uniform op_seq.
_FIELDS = ("armed", "ready", "epoch", "op_seq", "c_target")


def _encode(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    for v in values:
        out.append(int(v))
        out.append(-int(v))
    return out


@dataclass
class RankState:
    """Replicated per-rank capacity inputs (gathered once at build)."""

    floor_bytes: int
    budget_bytes: int
    target_row_bytes: int  # trb_r: bytes per physical target-pool row
    draft_token_bytes: int  # drb_r: bytes per token over ALL draft pools
    target_reserve_rows: int  # VA rows reserved by the target pool
    draft_reserve_tokens: int  # min over draft pools of reserved tokens
    boot_backed_rows: int  # fitted-ceiling rows physically backed at boot
    card_uuid: str = ""
    device_index: int = -1
    backed_rows: int = 0  # replicated backing model (init: boot rows)


class KvCapacityRuntime:
    """Drives one TP group's runtime KV capacity from the scheduler loop.

    The runtime reconciles a DESIRED STATE -- (group ceiling C, per-rank
    physical backing rows) -- against the current state at every consensus
    boundary. Backing targets include a pre-provision for a PENDING #297
    reshard vector when the budgets fund it, so "reshard back to a
    row-heavier vector after a trim" self-heals instead of deadlocking on
    the fit guard.

    Injectables mirror the #287/#297 runtimes so hermetic tests drive real
    state machines through mock channels: ``collective_min`` is the packed
    MIN consensus channel; ``ready_fn`` the fully-idle predicate;
    ``current_vector_fn`` the live token vector; ``pending_reshard_fn`` the
    replicated pending reshard target (or None);
    ``commit_fn(c_new, backing_rows, mode)`` performs the actual
    pool/allocator/scheduler mutation for mode in {"grow", "shrink",
    "adjust"} and returns bytes released; ``ledger_report_fn(budget_bytes)``
    mirrors the budget into the VRAM ledger.
    """

    def __init__(
        self,
        *,
        n_ranks: int,
        my_rank: int,
        ranks: List[RankState],
        page_size: int,
        current_c: int,
        user_cap: Optional[int] = None,
        vector_caps: Optional[Dict[Tuple[int, ...], int]] = None,
        consensus_interval: int = 8,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        ready_fn: Callable[[], bool] = lambda: True,
        current_vector_fn: Callable[[], Tuple[int, ...]] = lambda: (1,),
        pending_reshard_fn: Callable[[], Optional[Tuple[int, ...]]] = lambda: None,
        commit_fn: Optional[Callable[[int, int, str], int]] = None,
        ledger_report_fn: Optional[Callable[[int], None]] = None,
        relief_fn: Optional[Callable[..., int]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if n_ranks < 1:
            raise KvCapacityError(f"n_ranks must be >= 1, got {n_ranks}")
        if len(ranks) != n_ranks:
            raise KvCapacityError(
                f"got {len(ranks)} rank states for n_ranks={n_ranks}"
            )
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if n_ranks > 1 and collective_min is None:
            raise KvCapacityError(
                "a multi-rank group needs a consensus channel (collective_min); "
                "running without one would turn the first honest divergence "
                "into a collective hang instead of a loud error."
            )
        self._n = int(n_ranks)
        self._rank = int(my_rank)
        self._ranks = ranks
        self._page = int(page_size)
        self._c_cur = int(current_c)
        self._user_cap = None if user_cap is None else int(user_cap)
        # Boot per-vector achievable ceilings (the #297 min-reduce, clamped
        # by the user cap AND the hybrid mamba physical ceiling). Growth must
        # never exceed the current vector's boot prognosis: the runtime's
        # byte math knows budgets and VA, but only the boot sizing knows the
        # mamba ceiling and the profiled physical capacity.
        self._vector_caps = {
            tuple(int(x) for x in k): int(v)
            for k, v in (vector_caps or {}).items()
        }
        self._interval = int(consensus_interval)
        self._collective_min = collective_min
        self._ready_fn = ready_fn
        self._vector_fn = current_vector_fn
        self._pending_reshard_fn = pending_reshard_fn
        self._commit = commit_fn
        self._ledger_report = ledger_report_fn
        # #658: the CorridorGuard's relief ladder, injected rather than
        # imported, so this class stays a pure-math object with testable
        # seams (the same shape as ``commit_fn`` and ``ledger_report_fn``).
        # ``None`` is a supported state: a rank whose guard failed to build
        # loses an optimisation, never a dial.
        self._relief = relief_fn
        self._clock = clock

        for st in ranks:
            if st.backed_rows == 0:
                st.backed_rows = st.boot_backed_rows

        self._round = 0
        self._epoch = 0
        self._op_seq = 0
        self._shrink_authorized = False
        self._last_hold: Optional[str] = None
        self.desync_checks = 0
        self.completed = 0
        self.last_stats: Optional[dict] = None
        # Report the boot budgets once so the ledger reflects reality from
        # the first boundary even without any dial.
        self._ledger_dirty = True

        logger.info(
            "%s armed at boot: rank %d/%d, C=%d, budgets %s MiB, floors %s "
            "MiB, consensus every %d rounds",
            LOG_PREFIX,
            self._rank,
            self._n,
            self._c_cur,
            [st.budget_bytes // MIB for st in ranks],
            [st.floor_bytes // MIB for st in ranks],
            self._interval,
        )

    # -- pure capacity math ----------------------------------------------------

    def _unit_cap_for_rank(self, r: int, vec: Tuple[int, ...]) -> int:
        """Largest owner-block unit rank ``r`` can fund and address for
        ``vec``: min of the budget bound and the VA-reservation bound."""
        st = self._ranks[r]
        v_r = int(vec[r])
        s = sum(vec)
        trb, drb = st.target_row_bytes, st.draft_token_bytes
        kv_budget = st.budget_bytes - st.floor_bytes
        denom = v_r * trb + s * drb
        # rows(unit) = (unit + 1) * v_r; draft tokens = unit * s + page.
        by_budget = (kv_budget - v_r * trb - self._page * drb) // denom
        by_target_va = st.target_reserve_rows // v_r - 1
        by_draft_va = (
            st.draft_reserve_tokens // s if drb > 0 else by_target_va
        )
        return min(by_budget, by_target_va, by_draft_va)

    def compute_target_c(self, vec: Optional[Tuple[int, ...]] = None) -> int:
        """The group ceiling fundable under the current budgets -- a pure
        function of replicated state, identical on every rank."""
        vec = tuple(self._vector_fn()) if vec is None else tuple(vec)
        s = sum(vec)
        unit = min(self._unit_cap_for_rank(r, vec) for r in range(self._n))
        if self._user_cap is not None:
            unit = min(unit, self._user_cap // s)
        cap = self._vector_caps.get(vec)
        if cap is None and self._vector_caps:
            # A vector outside the declared plan (should not happen: reshard
            # only accepts declared vectors) is bounded by the plan maximum.
            cap = max(self._vector_caps.values())
        if cap is not None:
            unit = min(unit, cap // s)
        # Keep C a multiple of page_size (unit granularity: page / gcd).
        step = self._page // math.gcd(s, self._page)
        unit = (unit // step) * step
        return max(unit, 0) * s

    def rows_for_rank(self, r: int, c: int, vec: Tuple[int, ...]) -> int:
        from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

        return dcp_compact_pool_rows(int(c), sum(vec), int(vec[r]))

    def _budget_rows(self, r: int, c: int) -> int:
        """Max target-pool rows rank ``r``'s budget funds at ceiling ``c``."""
        st = self._ranks[r]
        left = (
            st.budget_bytes
            - st.floor_bytes
            - (int(c) + self._page) * st.draft_token_bytes
        )
        return max(left, 0) // max(st.target_row_bytes, 1)

    def min_viable_budget_bytes(self, r: int) -> int:
        """Smallest budget for rank ``r`` that still funds one owner block."""
        vec = tuple(self._vector_fn())
        st = self._ranks[r]
        v_r, s = int(vec[r]), sum(vec)
        return (
            st.floor_bytes
            + 2 * v_r * st.target_row_bytes
            + (s + self._page) * st.draft_token_bytes
        )

    def effective_budget_ceiling_bytes(self, r: int) -> int:
        """Budget beyond which raising further has no effect; used to clamp
        stored budgets for honest ledger entries. Card-run finding: the
        VA-derived bound under the CURRENT vector can lie BELOW the rank's
        actual backing (e.g. after resharding to a vector with a small ratio
        on that rank), and clamping there both falsified the ledger and
        blocked the pre-provision escape -- so the ceiling is never below
        floor + the rank's backed bytes, and never below the best declared
        vector's funding needs (pre-provisioning a pending reshard must stay
        fundable)."""
        st = self._ranks[r]
        best = st.floor_bytes + st.backed_rows * st.target_row_bytes + (
            self._c_cur + self._page
        ) * st.draft_token_bytes
        for vec in self._vector_caps or {tuple(self._vector_fn()): None}:
            v_r, s = int(vec[r]), sum(vec)
            unit_va = st.target_reserve_rows // v_r - 1
            if st.draft_token_bytes > 0:
                unit_va = min(unit_va, st.draft_reserve_tokens // s)
            kv = (unit_va + 1) * v_r * st.target_row_bytes + (
                unit_va * s + self._page
            ) * st.draft_token_bytes
            best = max(best, st.floor_bytes + kv)
        return best

    # -- state accessors -------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def current_c(self) -> int:
        return self._c_cur

    @property
    def op_seq(self) -> int:
        return self._op_seq

    def status(self) -> dict:
        vec = tuple(self._vector_fn())
        return {
            "epoch": self._epoch,
            "op_seq": self._op_seq,
            "max_total_num_tokens": self._c_cur,
            "target_tokens": self.compute_target_c(),
            "vector": list(vec),
            "page_size": self._page,
            "ranks": [
                {
                    "rank": r,
                    "card_uuid": st.card_uuid,
                    "device_index": st.device_index,
                    "budget_mib": st.budget_bytes // MIB,
                    "floor_mib": st.floor_bytes // MIB,
                    "backed_rows": st.backed_rows,
                    "backed_kv_mib": (
                        st.backed_rows * st.target_row_bytes
                        + (self._c_cur + self._page) * st.draft_token_bytes
                    )
                    // MIB,
                    "min_viable_budget_mib": self.min_viable_budget_bytes(r)
                    // MIB,
                }
                for r, st in enumerate(self._ranks)
            ],
        }

    # -- #297 interplay --------------------------------------------------------

    def reshard_fit_check(self, target_vec: Sequence[int]) -> Tuple[bool, str]:
        """Would resharding to ``target_vec`` at the CURRENT ceiling fit every
        rank's physically-backed rows? Pure function of the replicated backing
        model (conservative: model <= actual backing)."""
        vec = tuple(int(x) for x in target_vec)
        for r, st in enumerate(self._ranks):
            needed = self.rows_for_rank(r, self._c_cur, vec)
            if needed > st.backed_rows:
                # Largest C that fits every rank's backing under target_vec.
                s = sum(vec)
                unit_fit = min(
                    self._ranks[q].backed_rows // int(vec[q]) - 1
                    for q in range(self._n)
                )
                return False, (
                    f"vector {list(vec)} at C={self._c_cur} needs "
                    f"{needed} rows on rank {r} but only {st.backed_rows} are "
                    f"backed; dial the capacity down to C<={max(unit_fit, 0) * s} "
                    f"(POST /vram_budget) or raise that rank's budget first"
                )
        return True, ""

    # -- arming (replicated callers: broadcast RPC) ----------------------------

    def apply_budget_request(
        self,
        *,
        device: str,
        budget_mib: Optional[int] = None,
        release_mib: Optional[int] = None,
        release_fraction: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Apply one dial request to the replicated budget vector.

        Replicated call: every rank receives the identical request via the
        broadcast pipe and computes the identical new budgets, so the
        consensus round observes uniform state (op_seq skew across a
        boundary is absorbed by the hold path). Returns (ok, message);
        rejections carry the exact floor arithmetic and change nothing.
        """
        modes = [
            m
            for m, v in (
                ("budget_mib", budget_mib),
                ("release_mib", release_mib),
                ("release_fraction", release_fraction),
            )
            if v is not None
        ]
        if len(modes) != 1:
            return False, (
                f"exactly one of budget_mib / release_mib / release_fraction "
                f"is required, got {modes or 'none'}"
            )
        try:
            targets = self._resolve_targets(device)
        except KvCapacityError as e:
            return False, str(e)

        # Compute all new budgets first, validate all, then commit all --
        # a multi-rank dial must not half-apply.
        new_budgets: Dict[int, int] = {}
        for r in targets:
            st = self._ranks[r]
            if budget_mib is not None:
                nb = int(budget_mib) * MIB
            elif release_mib is not None:
                nb = st.budget_bytes - int(release_mib) * MIB
            else:
                f = float(release_fraction)
                if not (-1.0 <= f <= 1.0):
                    return False, (
                        f"release_fraction must be in [-1, 1], got {f}"
                    )
                dialable = st.budget_bytes - st.floor_bytes
                nb = st.budget_bytes - int(f * dialable)
            floor_min = self.min_viable_budget_bytes(r)
            if nb < floor_min:
                st_floor = st.floor_bytes
                return False, (
                    f"budget {nb // MIB} MiB for rank {r} "
                    f"(device {st.card_uuid or st.device_index}) is below the "
                    f"pinned floor: weights+graphs+state floor is "
                    f"{st_floor // MIB} MiB and the minimum viable budget "
                    f"(floor + one KV owner block) is {floor_min // MIB} MiB. "
                    f"Nothing was changed."
                )
            ceil = self.effective_budget_ceiling_bytes(r)
            if nb > ceil:
                logger.info(
                    "%s budget %d MiB for rank %d exceeds the VA-reservation "
                    "ceiling; clamping to the effective %d MiB",
                    LOG_PREFIX,
                    nb // MIB,
                    r,
                    ceil // MIB,
                )
                nb = ceil
            new_budgets[r] = nb

        my_reduction = 0
        for r, nb in new_budgets.items():
            old = self._ranks[r].budget_bytes
            self._ranks[r].budget_bytes = nb
            if r == self._rank:
                my_reduction = max(0, old - nb)
            logger.warning(
                "%s budget rank %d: %d MiB -> %d MiB (floor %d MiB)",
                LOG_PREFIX,
                r,
                old // MIB,
                nb // MIB,
                self._ranks[r].floor_bytes // MIB,
            )
        # #658: A BUDGET REDUCTION IS CORRIDOR PRESSURE, AND THE LADDER
        # DECIDES WHAT YIELDS.
        #
        # Committed only after the whole request has validated, so a rejected
        # dial never pays a restore cost for a change that did not happen.
        if my_reduction > 0:
            self._relieve_for_reduction(my_reduction)
        self._op_seq += 1
        self._ledger_dirty = True
        c_target = self.compute_target_c()
        if c_target < self._c_cur:
            # Only an explicit dial authorizes the (cache-flushing) shrink.
            self._shrink_authorized = True
        msg = (
            f"budgets updated (op {self._op_seq}): target ceiling "
            f"{c_target} tokens vs current {self._c_cur} -- "
            + (
                "shrink armed (flushes the radix cache), commits at the next "
                "group-idle consensus boundary"
                if c_target < self._c_cur
                else (
                    "growth commits at the next group-idle consensus boundary"
                    if c_target > self._c_cur
                    else "no capacity change needed"
                )
            )
        )
        return True, msg

    def _relieve_for_reduction(self, nbytes: int) -> int:
        """Spend the corridor relief ladder for an external budget cut.

        WHY THE LADDER AND NOT THIS RUNTIME'S OWN SHRINK. The dial already
        has an actuator for a lowered budget -- ``compute_target_c`` falls,
        ``_shrink_authorized`` arms, and the next group-idle boundary cuts KV
        capacity. That path is correct and it stays; what it is NOT is
        relief, for two measured reasons:

        * it commits only at a FULLY IDLE group boundary, so under load an
          external tenant waits an unbounded time for its bytes, and
        * a shrink flushes the radix cache (``_commit_on_scheduler``), i.e.
          the default answer to "give back memory" was to throw away the
          prefix cache and shrink the pool -- "a smaller pool as the fix",
          which the standing rule forbids as a FIRST resort.

        The ladder answers the same ask with the cheapest thing that can
        yield instead: torch's unowned cached blocks (RELIEF_LOCAL), then a
        cold payload parked in a peer card's surplus (RELIEF_PARK), then host
        RAM (RELIEF_HOST) once the fleet is level. The REBALANCE tier is
        empty by standing verdict -- three candidate mechanisms falsified
        with receipts -- so in practice the ladder runs local -> park ->
        host. Whatever the ladder cannot fund is left to the dial's capacity
        arithmetic as the residual, which is the honest division of labour:
        the ladder is fast and reversible, the capacity cut is slow and
        flushes, so the slow one funds only what the fast one could not.

        RANK-LOCAL SPEND, REPLICATED DECISION -- AND WHY THAT IS NOT N39's
        TRAP. Register law 11/12's lesson from the steer was that a
        group-uniform DECISION applied on a rank-local CLOCK desynchronises
        replicated state. Nothing here is on a clock: the decision arrives
        already replicated (the dial request is broadcast to every scheduler
        and every rank computes identical budgets), and what the ladder
        spends is rank-local PAYLOAD -- the allocator's cache, a carrier's
        pages -- never replicated scheduler state. The KV pool, which IS
        replicated because it moves ``available_size()`` and therefore
        admission, is untouched here and continues to change only through the
        existing MIN-reduced consensus boundary. That split is the same one
        ``CorridorGuard`` already draws between its rank-local providers and
        ``collective_kv_backing_relief``.

        NEVER FATAL. A relief ladder that raised would turn a budget request
        -- an external tenant politely asking for its own memory -- into a
        failed dial. It is an optimisation; it is logged and swallowed.
        """
        if self._relief is None:
            return 0
        reason = (
            f"#330 external budget reduction of {nbytes // MIB} MiB on rank "
            f"{self._rank} (device "
            f"{self._ranks[self._rank].card_uuid or self._ranks[self._rank].device_index})"
        )
        try:
            got = int(self._relief(nbytes, reason=reason))
        except Exception as e:
            logger.warning(
                "%s relief ladder raised while funding a %d MiB budget "
                "reduction (%s); the reduction still stands and the capacity "
                "arithmetic funds the residual at the next boundary",
                LOG_PREFIX,
                nbytes // MIB,
                e,
            )
            return 0
        logger.info(
            "%s budget reduction of %d MiB on rank %d: the corridor relief "
            "ladder returned %d MiB now; the residual is funded by the "
            "capacity arithmetic at the next group-idle boundary. The guard "
            "decides WHAT yields (local cache -> park -> host); this runtime "
            "no longer answers a budget cut by flushing the radix cache "
            "first.",
            LOG_PREFIX,
            nbytes // MIB,
            self._rank,
            got // MIB,
        )
        return got

    def _resolve_targets(self, device: str) -> List[int]:
        spec = str(device).strip()
        if spec.lower() == "all":
            return list(range(self._n))
        if spec.lower().startswith("rank:"):
            r = int(spec.split(":", 1)[1])
            if not (0 <= r < self._n):
                raise KvCapacityError(
                    f"rank {r} out of range 0..{self._n - 1}"
                )
            return [r]
        if spec.lower().startswith("cuda:"):
            idx = int(spec.split(":", 1)[1])
            hits = [
                r for r, st in enumerate(self._ranks) if st.device_index == idx
            ]
            if not hits:
                raise KvCapacityError(
                    f"no rank runs on cuda:{idx} (rank devices: "
                    f"{[st.device_index for st in self._ranks]})"
                )
            return hits
        if spec.startswith("GPU-") or spec.startswith("MIG-"):
            hits = [
                r for r, st in enumerate(self._ranks) if st.card_uuid == spec
            ]
            if not hits:
                raise KvCapacityError(
                    f"no rank runs on card {spec} (rank cards: "
                    f"{[st.card_uuid for st in self._ranks]})"
                )
            return hits
        raise KvCapacityError(
            f"unrecognized device spec {spec!r}; use 'rank:N', 'cuda:N', a "
            f"GPU-... NVML UUID, or 'all'"
        )

    # -- the per-round hook ----------------------------------------------------

    def _desired_state(
        self, vec: Tuple[int, ...]
    ) -> Tuple[int, List[int], str]:
        """(C_new, per-rank backing targets, mode) -- a pure function of
        replicated state (budgets, floors, vector, pending reshard target,
        the backing model, the authorization flag)."""
        c_target = self.compute_target_c(vec)
        # Grow deadband: the byte->unit inversion carries ceil/page rounding
        # slack of a few owner blocks, so a natural boot can look like a
        # tiny grow (card run finding: +epsilon at the first boundary). A
        # growth below two owner blocks is noise, not capacity.
        if self._c_cur < c_target < self._c_cur + 2 * sum(vec):
            c_target = self._c_cur
        # No growth before the first dial (card run finding): the natural
        # boot budgets include the chunk-rounding slack of the ceiling
        # backing, which funded a spontaneous grow past the fitted ceiling
        # and broke declared-vector resharding until a dial-down. Growth is
        # armed only once an operator has touched the budgets (op_seq > 0);
        # from then on it is automatic (including the post-reshard re-raise).
        if c_target > self._c_cur and self._op_seq == 0:
            self._hold(
                f"growth to {c_target} tokens is funded but no dial has "
                f"authorized headroom yet (POST /vram_budget); holding the "
                f"boot ceiling {self._c_cur}"
            )
            c_target = self._c_cur
        if c_target >= self._c_cur or self._shrink_authorized:
            c_new = c_target
        else:
            c_new = self._c_cur
            self._hold(
                f"target ceiling {c_target} < current {self._c_cur} but no "
                f"dial authorized a shrink; holding capacity"
            )
        base = [self.rows_for_rank(r, c_new, vec) for r in range(self._n)]
        # Pre-provision a pending #297 reshard target when budgets fund it:
        # rows for the pending vector at C_new, on every rank, so the fit
        # guard opens instead of deadlocking after an earlier trim.
        pending = self._pending_reshard_fn()
        if pending is not None and len(pending) == self._n:
            pend = [
                max(base[r], self.rows_for_rank(r, c_new, tuple(pending)))
                for r in range(self._n)
            ]
            if all(
                self._ranks[r].floor_bytes
                + pend[r] * self._ranks[r].target_row_bytes
                + (c_new + self._page) * self._ranks[r].draft_token_bytes
                <= self._ranks[r].budget_bytes
                for r in range(self._n)
            ) and all(
                pend[r] <= self._ranks[r].target_reserve_rows
                for r in range(self._n)
            ):
                base = pend
        if self._shrink_authorized:
            # Release only where the budget binds: a rank whose budget still
            # covers its slack keeps it (instant re-raise there); the dialed
            # rank is trimmed to what its budget funds. Pure function of the
            # replicated budgets, identical on every rank.
            base = [
                max(
                    base[r],
                    min(
                        max(base[r], self._ranks[r].backed_rows),
                        self._budget_rows(r, c_new),
                        self._ranks[r].target_reserve_rows,
                    ),
                )
                for r in range(self._n)
            ]
        else:
            # Never release spontaneously: keep the boot fitted-ceiling slack
            # (it is what lets #297 move between declared vectors unaided).
            base = [
                max(base[r], self._ranks[r].backed_rows) for r in range(self._n)
            ]
        if c_new < self._c_cur:
            mode = "shrink"
        elif c_new > self._c_cur:
            mode = "grow"
        else:
            mode = "adjust"
        return c_new, base, mode

    def pending_work(self) -> bool:
        """True while the desired state differs from the current state --
        pure function of replicated inputs (the scheduler uses it to keep
        the loop ticking toward a consensus boundary, uniformly)."""
        vec = tuple(self._vector_fn())
        c_new, backing, _ = self._desired_state(vec)
        return c_new != self._c_cur or any(
            backing[r] != self._ranks[r].backed_rows for r in range(self._n)
        )

    def on_round(self) -> Optional[dict]:
        """One scheduler round. Cadence-gated by the REPLICATED round counter;
        at a boundary every rank enters ONE bounded MIN-reduction, armed or
        not. Returns commit stats when a capacity change executed."""
        self._round += 1
        if self._round % self._interval != 0:
            return None

        vec = tuple(self._vector_fn())
        c_new, backing, mode = self._desired_state(vec)
        armed = 1 if (
            c_new != self._c_cur
            or any(
                backing[r] != self._ranks[r].backed_rows for r in range(self._n)
            )
        ) else 0
        ready = 1 if (armed and self._ready_fn()) else 0
        payload = _encode(
            [armed, ready, self._epoch, self._op_seq, c_new if armed else 0]
        )
        if self._n > 1:
            self.desync_checks += 1
            reduced = self._collective_min(payload)
            if len(reduced) != len(payload):
                raise KvCapacityError(
                    f"consensus channel returned {len(reduced)} values for a "
                    f"{len(payload)}-value payload; the channel contract is "
                    f"element-wise MIN of the packed proposal."
                )
            lo = {f: reduced[2 * i] for i, f in enumerate(_FIELDS)}
            hi = {f: -reduced[2 * i + 1] for i, f in enumerate(_FIELDS)}
            if lo["epoch"] != hi["epoch"]:
                raise KvCapacityError(
                    f"{LOG_PREFIX} DESYNC at round {self._round}: epoch "
                    f"min={lo['epoch']} max={hi['epoch']} (this rank "
                    f"{self._epoch}). The ranks committed different capacity "
                    f"histories -- failing loudly HERE, before any rank "
                    f"resizes under the wrong ceiling."
                )
            if lo["op_seq"] != hi["op_seq"]:
                # Legal delivery skew of a dial RPC: wait uniformly.
                self._hold(
                    f"waiting for budget delivery (op_seq min "
                    f"{lo['op_seq']} max {hi['op_seq']})"
                )
                return None
            if lo["armed"] == 0:
                if hi["armed"] == 1:
                    self._hold("waiting for every rank to arm (skew)")
                self._maybe_report_ledger()
                return None
            if lo["c_target"] != hi["c_target"]:
                raise KvCapacityError(
                    f"{LOG_PREFIX} DESYNC at round {self._round}: the ranks "
                    f"computed different target ceilings (min "
                    f"{lo['c_target']}, max {hi['c_target']}; this rank "
                    f"{c_new}) at identical op_seq {self._op_seq}. "
                    f"Capacity math must be a pure function of replicated "
                    f"state -- failing loudly instead of resizing into a "
                    f"mixed-ceiling group."
                )
            if lo["ready"] == 0:
                self._hold(
                    f"armed ({mode} -> C={c_new}), waiting for a group-wide "
                    f"idle boundary"
                )
                return None
        elif not armed:
            self._maybe_report_ledger()
            return None
        elif not ready:
            self._hold(f"armed ({mode} -> C={c_new}), waiting for idle")
            return None

        self._last_hold = None
        return self._execute(c_new, backing, vec, mode)

    def _hold(self, reason: str) -> None:
        if reason != self._last_hold:
            logger.info("%s hold: %s", LOG_PREFIX, reason)
            self._last_hold = reason

    # -- commit ----------------------------------------------------------------

    def _execute(
        self, c_new: int, backing: List[int], vec: Tuple[int, ...], mode: str
    ) -> dict:
        if self._commit is None:
            raise KvCapacityError("commit_fn is not wired")
        t0 = self._clock()
        c_old = self._c_cur
        released = int(self._commit(c_new, backing[self._rank], mode))
        # Replicated backing model: every rank applies the same formula.
        for r, st in enumerate(self._ranks):
            st.backed_rows = backing[r]
        self._c_cur = c_new
        self._shrink_authorized = False
        self._epoch += 1
        self.completed += 1
        self._ledger_dirty = True
        self._maybe_report_ledger()
        total_ms = (self._clock() - t0) * 1000.0
        stats = {
            "kind": mode,
            "old_tokens": c_old,
            "new_tokens": c_new,
            "vector": list(vec),
            "epoch": self._epoch,
            "backing_rows": backing[self._rank],
            "released_bytes": released,
            "total_ms": total_ms,
        }
        self.last_stats = stats
        logger.warning(
            "%s DONE %s: max_total_num_tokens %d -> %d (vector %s, epoch %d, "
            "backing %d rows) in %.1f ms; released %.1f MiB to the driver on "
            "this rank",
            LOG_PREFIX,
            mode.upper(),
            c_old,
            c_new,
            list(vec),
            self._epoch,
            backing[self._rank],
            total_ms,
            released / MIB,
        )
        return stats

    def _maybe_report_ledger(self) -> None:
        if self._ledger_report is None:
            return
        now = self._clock()
        if self._ledger_dirty:
            try:
                self._ledger_report(self._ranks[self._rank].budget_bytes)
                self._ledger_dirty = False
                self._last_heartbeat = now
            except Exception as e:
                # Ledger I/O must never take serving down; stay dirty, retry.
                logger.warning("%s ledger report failed: %s", LOG_PREFIX, e)
            return
        # Lease upkeep, time-throttled (the gpu-arb holder convention).
        hb = getattr(self._ledger_report, "heartbeat", None)
        if hb is not None and now - getattr(self, "_last_heartbeat", 0.0) >= 30.0:
            try:
                hb()
                self._last_heartbeat = now
            except Exception as e:
                logger.warning("%s ledger heartbeat failed: %s", LOG_PREFIX, e)


# ---------------------------------------------------------------------------
# Production wiring (scheduler side).
# ---------------------------------------------------------------------------


def _measure_local_floor_bytes(participants: List[DialParticipant]) -> tuple:
    """(floor_bytes, card_uuid, device_index): CARD-level NVML used bytes on
    this rank's card minus the VMM-backed KV bytes.

    Card-level on purpose (first card run findings): per-process NVML pids
    are namespace-shifted inside the LXC container, and sibling rank
    processes leave small CUDA contexts on foreign cards -- pid matching is
    ambiguous, the card total is not. The arb window guarantees exclusivity
    at boot, so everything used on the card minus our KV backing IS this
    rank's pinned floor (plus sibling context slivers, which honestly do
    reduce what a dial can release on this card). The device index comes
    from the runner's own gpu_id, never from the scheduler thread's
    current-device (which is unset there -- also a card-run finding)."""
    import torch

    from sglang.srt.registry import nvml

    device_index = int(getattr(participants[0].runner, "gpu_id", 0))
    props = torch.cuda.get_device_properties(device_index)
    uuid = f"GPU-{props.uuid}"
    try:
        info = nvml.memory_info_for_uuid(uuid)
    except Exception as e:
        raise KvCapacityError(
            f"cannot resolve NVML memory info for card {uuid} (cuda:"
            f"{device_index}): {e}; --enable-vram-dial needs the card-level "
            f"usage to establish the pinned floor."
        ) from e
    used = int(info.total_bytes - info.free_bytes)
    backed = sum(int(p.pool.vmm_backed_bytes) for p in participants)
    # #658, register C18: THE CORRIDOR LAW IS A TERM IN THIS FLOOR.
    #
    # Everything downstream spends ``budget_bytes - floor_bytes`` on KV, so a
    # floor that stopped at "what is pinned on this card" let an absolute
    # budget plan a ceiling that consumed the card's last free byte -- and the
    # guard would then have to claw it back at an allocation. Adding the law
    # here reserves the user's 1024 MiB free in ONE place, and every consumer
    # (the unit cap, the budget rows, the minimum viable budget, the effective
    # ceiling, the below-floor refusal) inherits it without knowing it exists.
    #
    # THE BOOT STATE IS BIT-IDENTICAL. ``build_kv_capacity_runtime`` derives
    # the natural budget as ``floor + backed``, so raising the floor raises
    # the natural budget by exactly the same amount and the spendable
    # ``budget - floor`` does not move. The term bites only when a caller
    # names an ABSOLUTE budget, which is precisely the moment an external
    # tenant is taking bytes off this card and the law is what protects
    # serving.
    law = corridor_law_floor_bytes()
    floor = used - backed + law
    if floor <= 0:
        raise KvCapacityError(
            f"floor measurement came out non-positive ({floor} bytes = "
            f"{used} used - {backed} KV-backed + {law} corridor law) on card "
            f"{uuid}; refusing to build a capacity model on nonsense numbers."
        )
    return floor, uuid, device_index


def _ledger_reporter(server_args, card_uuid: str, rank: int):
    """Per-rank ledger mirror: one HOT entry per (tenant, rank) on this
    rank's card, reserved_bytes = the rank's budget. A dial-down re-acquires
    with the smaller budget so external tenants see the freed bytes via
    available_bytes() immediately."""
    from sglang.srt.registry.ledger import ReservationStore, TenantState

    base = os.environ.get("HTSGLANG_TENANT_ID") or f"srt-{server_args.port}"
    tenant_id = f"{base}:rank{rank}"
    store = ReservationStore()

    def _report(budget_bytes: int) -> None:
        store.acquire(
            card_uuid=card_uuid,
            tenant_id=tenant_id,
            klass=1,
            reserved_bytes=int(budget_bytes),
            state=TenantState.HOT,
            posts={"vram_dial": 1},
        )

    def _heartbeat() -> None:
        store.heartbeat(card_uuid, tenant_id)

    _report.heartbeat = _heartbeat
    return _report


def _corridor_relief(scheduler):
    """Adapt ``CorridorGuard`` to the dial's relief hook, or ``None``.

    #658: this is the join that makes the guard the ONE authority for
    co-residency on a card (the #584 line). An external tenant lowering this
    instance's budget now spends the SAME relief ladder a flip seam or a
    prefill admission spends, in the same tier order, with the same refusal
    semantics -- instead of the dial's private answer of "flush the radix
    cache and shrink the pool at the next idle boundary".

    ``ensure_headroom`` IS THE RIGHT CALL, and the equivalence is exact: an
    external budget reduction of D bytes means someone else is about to take
    D bytes of THIS card, which is precisely the question the gate answers --
    "make D bytes allocatable without breaching the floor". It is not
    ``lend_to_level``: that one is bounded by the water-fill objective and
    capped below the host tier, which is right for levelling and wrong for a
    tenant that has already been promised the bytes.

    ``refusal_is_fatal`` is FALSE here, deliberately. Unlike the pp->tp leg,
    a budget reduction has a survivable refusal path: the capacity
    arithmetic funds the residual at the next boundary. So this caller does
    not get to force host RAM onto an unlevel fleet, and item 16's
    preference stands.

    Returns ``None`` when no guard can be built, which the runtime treats as
    "no ladder" rather than as an error.
    """
    try:
        from sglang.srt.managers.phase_flip_spill import get_corridor_guard

        guard = get_corridor_guard(scheduler)
    except Exception as e:
        logger.warning(
            "%s corridor guard unavailable (%s); a budget reduction will be "
            "funded by the capacity arithmetic alone",
            LOG_PREFIX,
            e,
        )
        return None
    if guard is None:
        return None

    def _relief(want_bytes: int, reason: str = "") -> int:
        return int(guard.ensure_headroom(int(want_bytes), reason=reason).reclaimed)

    return _relief


def _refresh_capacity_snapshots(scheduler, c_new: int) -> None:
    """Push the new ceiling into every boot-time snapshot holder (see
    DESIGN_330 section 6). Live readers (admission math off
    scheduler.max_total_num_tokens) pick it up by assignment."""
    scheduler.max_total_num_tokens = c_new
    for p in get_dial_participants():
        runner = p.runner
        runner.max_total_num_tokens = c_new
        cfg = getattr(runner, "memory_pool_config", None)
        if cfg is not None:
            cfg.max_total_num_tokens = c_new
    for attr in (
        "pool_stats_observer",
        "invariant_checker",
        "load_inquirer",
        "kv_events_publisher",
    ):
        comp = getattr(scheduler, attr, None)
        if comp is not None and hasattr(comp, "max_total_num_tokens"):
            try:
                comp.max_total_num_tokens = c_new
            except Exception:
                # Several scheduler components are frozen dataclasses (card
                # run finding); the refresh is still the right semantics.
                object.__setattr__(comp, "max_total_num_tokens", c_new)


def _commit_on_scheduler(
    scheduler, c_new: int, my_backing_rows: int, mode: str
) -> int:
    """All-ranks synchronous capacity commit at an idle boundary.

    ``grow``: map pages (zeroed) up to the backing target, extend the
    allocator slot space in place -- live radix content and its slot ids
    stay valid, nothing is flushed. ``shrink``: the slot space contracts, so
    the radix cache is flushed first (live slot ids are not guaranteed <
    C_new), then the tail pages are decommitted back to the driver.
    ``adjust``: backing-only change (trim slack / pre-provision a pending
    reshard vector) with the ceiling untouched -- rows outside the owner
    rule's range are unreachable, so no flush is needed. Returns bytes
    released on this rank."""
    if mode == "shrink":

        class _SizeCfg:
            max_total_num_tokens = c_new

        scheduler.tree_cache.reset()
        scheduler.req_to_token_pool.clear()
        scheduler.token_to_kv_pool_allocator.resize(_SizeCfg)
        if scheduler.draft_worker is not None:
            scheduler.draft_worker.clear_cache_pool()
    elif mode == "grow":
        scheduler.token_to_kv_pool_allocator.grow_size(c_new)
    released = 0
    for p in get_dial_participants():
        target_backing = my_backing_rows if p.is_target else c_new
        released += int(p.pool.runtime_set_backing_rows(target_backing))
        verify_pool_reached_capacity(p, target_backing)
    if mode in ("grow", "shrink"):
        _refresh_capacity_snapshots(scheduler, c_new)
    return released


def reclaimable_bytes_for(participant, floor_rows: Optional[int]) -> Optional[int]:
    """#553 Cut 2: bytes this participant could return, or None if unknown.

    A LIVE READ, not a plan number: ``full_pool_backed_rows`` is the bound
    eager launches actually pass (the same quantity
    ``verify_pool_reached_capacity`` checks a commit against), and
    ``_pool_row_nbytes`` measures the pool's real per-row K+V bytes.

    ``floor_rows`` is REQUIRED and is not derived here. There is no per-pool
    floor authority in this module -- the dial's floor is a card-level NVML
    measurement (:func:`_measure_local_floor_bytes`) taken at boot, and
    inventing a per-pool one would create a second authority for a number the
    #584 rule says has exactly one. A caller that cannot supply it gets None.

    NONE, NEVER ZERO, when anything is unreadable. Zero would read as "this
    pool is at its floor" -- a measurement -- when the truth is "nobody
    measured". That is the #606 defaulted-measurement defect, and here it
    would quietly remove a real source from an elastic plan.
    """
    if floor_rows is None:
        return None
    pool = getattr(participant, "pool", None)
    if pool is None:
        return None
    backed = getattr(pool, "full_pool_backed_rows", None)
    if backed is None:
        return None
    try:
        row_nbytes = _pool_row_nbytes(pool)
    except Exception:
        # A pool that cannot report its row width is unmeasured, not empty.
        return None
    spare_rows = int(backed) - int(floor_rows)
    if spare_rows <= 0:
        return 0
    return spare_rows * int(row_nbytes)


def verify_pool_reached_capacity(participant, target_backing: int) -> None:
    """Post-commit check that a participant pool REALLY carries the new
    capacity -- both bounds, not just the live one (#352).

    #345 and #352 are the same defect twice: the pool grew and one consumer of
    its capacity did not. The consumer is not always a Python attribute -- the
    second time it was a number a CUDA graph had baked into a kernel launch,
    which no amount of re-assigning ``pool.size`` reaches. So a commit checks
    both quantities a store has to satisfy afterwards:

    * ``full_pool_backed_rows`` -- the LIVE bound eager launches pass, which
      must be exactly the rows this rank was told to back;
    * ``store_bound_rows`` -- the LIFETIME bound a captured graph may be
      replaying (see ``graph_safe_store_bound``), which must still cover the
      new span, padding slot included.

    A pool that cannot satisfy the second one structurally cannot host the
    growth, and saying so here -- loudly, with the numbers -- is the point:
    silent partial growth is the bug class itself.
    """
    pool = participant.pool
    target_backing = int(target_backing)
    kind = "target" if participant.is_target else "draft"
    backed = getattr(pool, "full_pool_backed_rows", None)
    if backed is not None and int(backed) != target_backing:
        raise KvCapacityError(
            f"{LOG_PREFIX} capacity commit did not reach "
            f"{type(pool).__name__} ({kind} pool): backed rows {int(backed)} "
            f"!= requested {target_backing}. The pool's live store bound "
            f"would keep rejecting slot ids the allocator now hands out."
        )
    bound = getattr(pool, "store_bound_rows", None)
    if bound is None:
        return
    page = int(getattr(pool, "page_size", 1))
    needed = target_backing + page
    if int(bound) < needed:
        raise KvCapacityError(
            f"{LOG_PREFIX} {type(pool).__name__} ({kind} pool) cannot host "
            f"capacity {target_backing}: its KV buffers address {int(bound)} "
            f"rows but {needed} are needed ({target_backing} slots + {page} "
            f"padding). CUDA graphs captured over these buffers bake this "
            f"bound into their store_kvcache launches, so growing past it "
            f"would fail as a device assert on legal slot ids (#352) rather "
            f"than here."
        )


def validate_vram_dial_compat(server_args) -> None:
    """Boot-time refusals (validate early, fail fast). Each names its reason;
    these are Stage-1 scope guards, not physical impossibilities."""
    problems = []
    if server_args.device != "cuda":
        problems.append("device != cuda (the dial is CUDA-VMM based)")
    if getattr(server_args, "enable_memory_saver", False):
        problems.append("--enable-memory-saver (competing physical-backing owner)")
    if getattr(server_args, "disaggregation_mode", "null") != "null":
        problems.append("PD disaggregation (KV transfer registers boot spans)")
    if getattr(server_args, "hicache_storage_backend", None):
        problems.append("hierarchical cache storage (host pool sized from boot C)")
    if getattr(server_args, "enable_kv_session_offload", False):
        problems.append("kv-session-offload (host rows sized from boot C)")
    if getattr(server_args, "dual_group_lane", None):
        problems.append("dual-group lane (rank-local sizing contract)")
    if getattr(server_args, "dp_size", 1) > 1:
        problems.append("data parallelism (DP controller caches the boot C)")
    if getattr(server_args, "kv_canary", "none") not in (None, "none"):
        problems.append("kv-canary (its launch capacities snapshot the boot C)")
    if getattr(server_args, "enable_hisparse", False):
        problems.append(
            "hisparse (its allocator's size-derived state has no grow path)"
        )
    if getattr(server_args, "weightless_kv_fastlane", False):
        problems.append("weightless-KV fast lane (rank-local KV sizing contract)")
    # #352 audit: DFLASH's solo draft pool takes num_draft_slots from the BOOT
    # max_total_num_tokens and builds its free list and index tables from it
    # (speculative/dflash_solo_pool.py). Nothing refreshes those at a capacity
    # commit, so a grown ceiling would hand the draft path slot ids its own
    # tables cannot address. Refused loudly here rather than discovered as a
    # device assert later -- which is exactly the failure #352 was.
    _spec = " ".join(
        str(getattr(server_args, name, "") or "").lower()
        for name in ("speculative_algorithm", "speculative_drafter_policy")
    )
    if "dflash" in _spec:
        problems.append(
            "DFLASH speculative lane (its solo draft pool's num_draft_slots "
            "and index tables are frozen at the boot ceiling)"
        )
    if problems:
        raise ValueError(
            "--enable-vram-dial is refused with: "
            + "; ".join(problems)
            + ". Each is a named Stage-1 follow-up (DESIGN_330 section 7)."
        )


def build_kv_capacity_runtime(scheduler) -> KvCapacityRuntime:
    """Construct the runtime for one scheduler rank, or raise loudly.

    Called lazily on the first scheduler iteration when --enable-vram-dial is
    set (mirrors #287/#297: every dependency exists, no request admitted).
    Flag unset = never called = byte-identical behavior.
    """
    import torch.distributed as dist

    from sglang.srt.distributed.utils import (
        get_cp_token_ratios,
        uneven_dcp_active,
    )
    from sglang.srt.managers.kv_pressure_runtime import default_collective_min
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.runtime_context import get_parallel

    server_args = scheduler.server_args
    validate_vram_dial_compat(server_args)
    dcp_size = int(get_parallel().attn_dcp_size)
    if not uneven_dcp_active(dcp_size):
        raise KvCapacityError(
            "--enable-vram-dial Stage 1 requires WEIGHTED uneven DCP "
            "(--rank-kv-ratio with SGLANG_UNEVEN_DCP=1 "
            "SGLANG_UNEVEN_DCP_WEIGHTED=1) -- the same lane as #297. The "
            "plain-MHA dial lane is a named follow-up."
        )
    participants = get_dial_participants()
    if not participants:
        raise KvCapacityError(
            "--enable-vram-dial is set but no VMM-backed KV pool registered "
            "at construction; the pool lane did not engage (wrong pool "
            "family?). Expected HybridLinearKVPool with post_capture_active."
        )
    for p in participants:
        if not isinstance(p.pool, HybridLinearKVPool) or not getattr(
            p.pool, "post_capture_active", False
        ):
            raise KvCapacityError(
                f"dial participant pool {type(p.pool).__name__} is not a "
                f"VMM-backed HybridLinearKVPool"
            )
        if getattr(p.pool, "use_mla", False):
            # DESIGN_330 section 7 names MLA as refused; until now it was only
            # blocked indirectly. Its write path carries a different pool
            # geometry than the one the capacity model measures.
            raise KvCapacityError(
                "--enable-vram-dial Stage 1 does not support MLA KV pools "
                "(DESIGN_330 section 7); the capacity model measures the "
                "MHA row geometry."
            )
    targets = [p for p in participants if p.is_target]
    drafts = [p for p in participants if not p.is_target]
    if len(targets) != 1:
        raise KvCapacityError(
            f"expected exactly one target dial participant, got {len(targets)}"
        )
    target = targets[0]

    floor, uuid, device_index = _measure_local_floor_bytes(participants)
    trb = _pool_row_nbytes(target.pool)
    drb = sum(_pool_row_nbytes(p.pool) for p in drafts)
    boot_rows = int(target.pool.full_pool_backed_rows)
    backed = sum(int(p.pool.vmm_backed_bytes) for p in participants)

    budget_flag = (
        server_args.parsed_vram_budget_mib()
        if hasattr(server_args, "parsed_vram_budget_mib")
        else None
    )
    my_rank = int(get_parallel().attn_dcp_rank)
    tp_size = int(server_args.tp_size)
    natural_budget = floor + backed
    if budget_flag:
        if len(budget_flag) != tp_size:
            raise KvCapacityError(
                f"--vram-budget-mib has {len(budget_flag)} entries for "
                f"tp_size={tp_size}"
            )
        my_budget = int(budget_flag[my_rank]) * MIB
    else:
        my_budget = natural_budget

    local = {
        "floor": floor,
        "budget": my_budget,
        "natural": natural_budget,
        "trb": trb,
        "drb": drb,
        "reserve_rows": int(target.reserved_tokens),
        "draft_reserve": min(
            (int(p.reserved_tokens) for p in drafts), default=0
        ),
        "boot_rows": boot_rows,
        "uuid": uuid,
        "device_index": device_index,
    }
    if tp_size > 1:
        gathered: List[Optional[dict]] = [None] * tp_size
        dist.all_gather_object(gathered, local, group=scheduler.tp_cpu_group)
    else:
        gathered = [local]
    ranks = [
        RankState(
            floor_bytes=g["floor"],
            budget_bytes=g["budget"],
            target_row_bytes=g["trb"],
            draft_token_bytes=g["drb"],
            target_reserve_rows=g["reserve_rows"],
            draft_reserve_tokens=g["draft_reserve"],
            boot_backed_rows=g["boot_rows"],
            card_uuid=g["uuid"],
            device_index=g["device_index"],
        )
        for g in gathered
    ]

    plan = get_boot_capacity_plan()
    rt = KvCapacityRuntime(
        n_ranks=tp_size,
        my_rank=my_rank,
        ranks=ranks,
        page_size=int(server_args.page_size or 1),
        current_c=int(scheduler.max_total_num_tokens),
        user_cap=getattr(server_args, "max_total_tokens", None),
        vector_caps=plan.caps if plan is not None else None,
        consensus_interval=int(
            getattr(server_args, "vram_dial_consensus_interval", 8)
        ),
        collective_min=(
            default_collective_min(scheduler.tp_cpu_group) if tp_size > 1 else None
        ),
        ready_fn=lambda: scheduler.is_fully_idle(),
        current_vector_fn=lambda: tuple(get_cp_token_ratios()),
        pending_reshard_fn=lambda: (
            scheduler.kv_reshard_runtime.pending
            if getattr(scheduler, "kv_reshard_runtime", None) is not None
            else None
        ),
        commit_fn=lambda c, rows, mode: _commit_on_scheduler(
            scheduler, c, rows, mode
        ),
        ledger_report_fn=_ledger_reporter(server_args, uuid, my_rank),
        relief_fn=_corridor_relief(scheduler),
    )
    # An initial --vram-budget-mib below the natural boot budget authorizes
    # the first-boundary reconcile shrink (the cache is empty seconds after
    # boot; the flush is free). RANK-UNIFORM on purpose: the verdict is
    # computed from the GATHERED budgets/naturals, not from this rank's own
    # pair -- a rank-local authorization would split armed across ranks.
    any_initial_dial = any(g["budget"] < g["natural"] for g in gathered)
    if any_initial_dial and rt.compute_target_c() < rt.current_c:
        rt._shrink_authorized = True
    return rt
