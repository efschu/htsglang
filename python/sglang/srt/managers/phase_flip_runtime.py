# SPDX-License-Identifier: Apache-2.0
"""Phase-flip KV mover for #631 Route A (PP=3 prefill <-> TP=3 decode).

Moves the full-attention paged KV between the PP layout (stage owns whole
layers, pool row = global slot id) and the TP layout (rank owns tokens
under the weighted DCP vector, pool row = compact row), on the #297
envelope, carried over LITERALLY from ``managers/kv_reshard.py``:

* CONSENSUS FIRST, BYTES SECOND: every ``consensus_interval``-th round --
  gated by the replicated round counter, never local state -- every rank
  enters ONE bounded MIN-reduction with
  ``(armed, ready, epoch, direction, config_fp, vector...)``. ``armed``
  and ``ready`` are MIN-semantics (skew is legal and uniformly resolves
  to "wait"); ``epoch``, ``direction`` (once armed), the layer-map/vector
  fingerprint (ALWAYS -- it is boot config, divergence is fatal armed or
  not) and the vector are equality-checked with the same loud
  :class:`KvReshardError` on every rank.
* PACK -> EXCHANGE -> CHECKSUM -> WRITE with the pool untouched through
  pack, exchange and checksum verification; only the write phase is the
  no-return region. Source and destination are DIFFERENT pools here (the
  PP pool and the TP pool coexist), so the #297 aliasing hazard cannot
  arise inside one buffer -- the write order (local first, then incoming,
  disjoint injective targets) is kept anyway.
* Pools are pre-sized at boot for BOTH layouts: no growth, no address
  change, no CUDA-graph recapture. Bounds are checked loudly before any
  byte moves.

Payload layout per (stage s, dcp rank r) pair, identical on both ends by
convention (a checksum trailer keeps it falsifiable at runtime): layer
ordinals ascending, slots ascending within a layer, K bytes then V; one
row list per pair, reused for every layer (token ownership is
layer-independent). The receiver derives the expected byte count from ITS
OWN pool's per-layer row width -- a sender whose row format diverges is a
loud size/checksum error, which is the runtime pin of the "PP and TP rows
are byte-compatible" claim.

Weights-arena refill and GDN state movement are separate steps of the
flip protocol (DESIGN_631 section 3.6); ``pre_cutover_fns`` is their
injection seam so the scheduler wiring can order them inside the same
no-return region.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.phase_flip_plan import (
    PP_TO_TP,
    TP_TO_PP,
    PhaseFlipTransition,
    build_phase_flip_transition,
    validate_layer_map,
)
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.model_executor.weights_arena import uint8_checksum
from sglang.srt.managers.kv_reshard import (
    _CHECKSUM_BYTES,
    KvPoolView,
    _checksum,
    _encode,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP"

PHASE_PP = "pp"
PHASE_TP = "tp"

_DIR_ID = {PP_TO_TP: 1, TP_TO_PP: 2}
_DIR_OF_PHASE = {PHASE_PP: PP_TO_TP, PHASE_TP: TP_TO_PP}

#: How long an ARMED flip may wait for a group-wide quiescent boundary
#: before it gives up -- seconds, wall clock, measured on whichever rank is
#: still unparked.
#:
#: An armed flip withholds new work so the in-flight state drains; that is
#: what makes the flip interposable BETWEEN a request's prefill and its
#: decode instead of only after every stream has finished. The cost is that
#: a rank which never reaches quiescence withholds work forever, and the
#: requests it is holding never resume. This deadline bounds that: when it
#: expires the FLIP is abandoned, loudly, and serving continues. The user's
#: requests are never aborted -- they are the thing being protected.
#:
#: 30 s is chosen against the legitimate worst case: a drain is a handful of
#: iterations plus, at most, the continuation of one already-half-written
#: chunked prefill (exempt from parking, because a chunk that stops mid-way
#: could never satisfy the quiescence predicate at all).
DEFAULT_PARK_DEADLINE_S = 30.0

#: Env override for the above. Non-positive disables the deadline, which
#: restores the old unbounded wait -- available deliberately for debugging a
#: slow drain, and named so a reader sees that "no deadline" is a choice.
ENV_PARK_DEADLINE = "SGLANG_PHASE_FLIP_PARK_DEADLINE_S"


def park_deadline_s() -> float:
    try:
        return float(os.environ.get(ENV_PARK_DEADLINE, DEFAULT_PARK_DEADLINE_S))
    except ValueError:
        return DEFAULT_PARK_DEADLINE_S
_PHASE_AFTER = {PP_TO_TP: PHASE_TP, TP_TO_PP: PHASE_PP}


def _config_fingerprint(
    layer_map: Tuple[Tuple[int, ...], ...], vector: Tuple[int, ...]
) -> int:
    """31-bit stable fingerprint of the replicated flip configuration.

    Folded into every consensus payload and equality-checked ALWAYS: a
    rank booted with a different layer map or vector must die loudly at
    the first consensus round, not at the first wrong byte."""
    acc = 0
    for s, layers in enumerate(layer_map):
        for f in layers:
            acc = (acc * 1_000_003 + (s + 1) * 8191 + f * 131) % (2**31 - 1)
    for v in vector:
        acc = (acc * 1_000_003 + v * 65_537) % (2**31 - 1)
    return acc


class AbortDeferralWindow:
    """Pin 4 (DESIGN_631 3.6a): client disconnects during a flip.

    A parked request whose client vanishes mid-flip must not mutate the
    live slot set between the plan derivation and the write phase -- an
    abort applied on one rank before its peers diverges the replicated
    live set, which the runtime can only answer with a LOUD size/desync
    error (clean abort of the attempt, but a lost flip). Deferral makes
    the window airtight instead: while a flip is pending or executing,
    abort work is QUEUED; it drains in the first round after cutover (or
    after disarm). The queue preserves order. Slots are never leaked --
    the deferred abort frees them under the NEW layout, which is
    equivalent by the global-slot-id property (metadata never rewrites).
    """

    def __init__(self):
        self._deferred: List[Callable[[], None]] = []
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def deferred_count(self) -> int:
        return len(self._deferred)

    def activate(self) -> None:
        self._active = True

    def submit(self, work: Callable[[], None]) -> bool:
        """Run ``work`` now (returns False) or defer it (returns True)."""
        if self._active:
            self._deferred.append(work)
            return True
        work()
        return False

    def deactivate_and_drain(self) -> int:
        """Close the window and run everything deferred, in order."""
        self._active = False
        drained = 0
        while self._deferred:
            work = self._deferred.pop(0)
            work()
            drained += 1
        return drained


def build_flip_quiescence_fn(scheduler) -> Callable[[], bool]:
    """The flip ready predicate (DESIGN_631 3.5) -- NOT #297 fully-idle.

    True when no forward is in flight and no chunk is half-written, with
    requests PARKED: no partial chunk, previous batch drained, overlap
    result queue empty, PP micro-batches drained. Deliberately does NOT
    require an empty waiting queue or an empty running batch -- the flip
    exists to run between a request's prefill and its decode."""

    def _ready() -> bool:
        if getattr(scheduler, "chunked_req", None) is not None:
            return False
        last_batch = getattr(scheduler, "last_batch", None)
        if last_batch is not None and not last_batch.is_empty():
            return False
        result_queue = getattr(scheduler, "result_queue", None)
        if result_queue is not None and len(result_queue) > 0:
            return False
        drained = getattr(scheduler, "_pp_microbatches_drained", None)
        if drained is not None and not drained():
            return False
        return True

    return _ready


def build_flip_live_slots_fn(scheduler) -> Callable[[], torch.Tensor]:
    """Live slots = radix tree values UNION parked requests' rows.

    #297 Stage A enumerates the tree only, correct at fully-idle. The
    flip runs with requests parked, whose KV rows live in req_to_token
    and are NOT all in the tree yet -- omitting them would silently drop
    the freshest prefix KV at the flip (DESIGN_631 3.5). Replicated: the
    tree and the batch state are rank-replicated between rounds."""

    def _live() -> torch.Tensor:
        parts: List[torch.Tensor] = []
        values = scheduler.tree_cache.all_values_flatten()
        if values is not None and values.numel():
            parts.append(values.detach().to("cpu", torch.int64))
        running = getattr(scheduler, "running_batch", None)
        reqs = list(getattr(running, "reqs", []) or []) if running else []
        req_to_token = scheduler.req_to_token_pool.req_to_token
        for req in reqs:
            n = int(req.seqlen)
            if n <= 0:
                continue
            rows = req_to_token[req.req_pool_idx, :n]
            parts.append(rows.detach().to("cpu", torch.int64))
        if not parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.unique(torch.cat(parts))

    return _live


def flip_blocking_guards(scheduler) -> List[str]:
    """Features that refuse flip arming (DESIGN_631 3.7). Mirrors the
    #297 Stage-A guard shape, plus the #630 PP x disk-HiCache wedge."""
    guards: List[str] = []
    server_args = scheduler.server_args
    try:
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if scheduler.disaggregation_mode != DisaggregationMode.NULL:
            guards.append("PD disaggregation")
    except ImportError:
        pass
    if getattr(server_args, "enable_hierarchical_cache", False):
        guards.append(
            "hierarchical cache (#630: PP x disk HiCache wedges at warmup)"
        )
    if getattr(scheduler, "kv_session_offload", None) is not None:
        guards.append("kv-session-offload")
    if getattr(scheduler, "is_dual_group_lane", False) or getattr(
        server_args, "dual_group_lane", None
    ):
        guards.append("dual-group lane")
    if not hasattr(scheduler.tree_cache, "all_values_flatten"):
        guards.append(
            f"tree cache {type(scheduler.tree_cache).__name__} (no "
            f"all_values_flatten enumeration)"
        )
    return guards


class PhaseFlipLoopExit(Exception):
    """Control-flow signal: a flip COMMITTED this round; the current event
    loop must exit to the re-dispatching wrapper (dispatch_event_loop picks
    its loop ONCE from pp_size, so a changed topology needs a fresh
    dispatch). Raised by the scheduler's on_round hook AFTER
    PhaseFlipRuntime.on_round returned commit stats -- never from inside
    the runtime, whose epoch/phase bookkeeping must complete first. The
    quiescence predicate guarantees the loop holds no half-processed batch
    state when this propagates."""

    def __init__(self, direction: str):
        super().__init__(direction)
        self.direction = direction


def derive_pp_full_attn_layer_map(
    full_attention_layer_ids: Sequence[int],
    num_hidden_layers: int,
    pp_size: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Per-stage FULL-ATTENTION ORDINALS from the global layer geometry.

    A pure replicated function of (the model's global full-attention layer
    ids, the layer count, the PP stage split) -- every rank derives the
    same map, which the consensus fingerprint then pins at runtime. The
    stage split comes from get_pp_indices, the SAME function the PP model
    build used (env-uniform SGLANG_PP_LAYER_PARTITION included), so the
    map cannot drift from the actual stage windows.

    IMPORTANT SOURCE RULE: ``full_attention_layer_ids`` must be the
    UNMUTATED global list (e.g. from the TP stack's model_config, whose
    pp_size=1 adjust is the identity) -- the PP stack's model_config was
    rewritten in place to its stage-local slice
    (adjust_hybrid_swa_layers_for_pp)."""
    from sglang.srt.distributed.utils import get_pp_indices

    ids = [int(x) for x in full_attention_layer_ids]
    if ids != sorted(set(ids)):
        raise KvReshardError(
            f"full_attention_layer_ids must be strictly ascending, got {ids}"
        )
    if ids and not (0 <= ids[0] and ids[-1] < num_hidden_layers):
        raise KvReshardError(
            f"full_attention_layer_ids {ids} outside [0, {num_hidden_layers})"
        )
    bounds = [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]
    flat = [b for pair in bounds for b in pair]
    if flat != sorted(flat) or bounds[0][0] != 0 or bounds[-1][1] != num_hidden_layers:
        raise KvReshardError(
            f"PP stage bounds {bounds} do not partition [0, {num_hidden_layers})"
        )
    layer_map = []
    for start, end in bounds:
        layer_map.append(
            tuple(i for i, gid in enumerate(ids) if start <= gid < end)
        )
    covered = sorted(o for stage in layer_map for o in stage)
    if covered != list(range(len(ids))):
        raise KvReshardError(
            f"stage map {layer_map} does not cover every full-attention "
            f"ordinal exactly once (bounds {bounds}, ids {ids})"
        )
    return tuple(layer_map)


def build_gdn_flip_guard(scheduler) -> Callable[[str], None]:
    """5.3 PLACEHOLDER for the GDN state mover, honest by refusal.

    The full mover (layer-axis -> head-axis re-shard of conv/ssm state via
    MambaPool blobs, DESIGN_631 3.4) lands as slice 5.3b. Until then a
    flip with LIVE linear-attention state must refuse LOUDLY inside the
    no-return region's first step -- before any pool byte moved -- never
    proceed and silently truncate GDN state (the #212 Store-Route lesson).
    The 5.5 validation ladder's first rung (flip empty -> flip back) is
    exactly what this permits."""

    def _guard(direction: str) -> None:
        running = getattr(scheduler, "running_batch", None)
        reqs = list(getattr(running, "reqs", []) or []) if running else []
        if reqs:
            raise KvReshardError(
                f"{LOG_PREFIX} flip {direction} refused: {len(reqs)} live "
                f"request(s) hold GDN conv/ssm state and the GDN state "
                f"mover is not wired yet (slice 5.3b); flipping now would "
                f"silently truncate linear-attention state. Drain or wait."
            )

    return _guard


def build_production_flip_cutover(scheduler) -> Callable[[str], None]:
    """The cutover leg (DESIGN_631 3.6 step 5): everything the scheduler
    snapshotted from the boot topology is rebuilt for the target phase.
    Runs inside PhaseFlipRuntime._execute after KV/GDN/arena moves; the
    loop exit is raised LATER by the on_round hook (the runtime's
    epoch/phase bookkeeping must finish first)."""
    import dataclasses as _dc

    # Boot-phase snapshot for the return trip, taken ONCE at build time
    # (the scheduler's ps still holds the boot topology then).
    boot_ps = scheduler.ps
    boot_model_worker = scheduler.tp_worker

    def _cutover(direction: str) -> None:
        from sglang.srt.distributed import parallel_state as _ps
        from sglang.srt.distributed.utils import set_cp_token_ratios
        from sglang.srt.layers.dcp.owner import refresh_all_owner_bounds
        from sglang.srt.runtime_context import get_server_args

        stacks = scheduler.phase_flip_stacks
        tp_phase = direction == PP_TO_TP
        n = len(stacks.vector)
        world_rank = _ps.get_world_group().rank_in_group

        # 1. Module-level group routing (forward collectives resolve
        # through the parallel_state getters; see phase_flip_boot).
        _ps.set_phase_flip_tp_active(tp_phase)

        # 2. Owner rule: the vector is boot-constant; refresh the bounds
        # consumers so the TP backends read the (re)installed vector.
        set_cp_token_ratios(list(stacks.vector))
        refresh_all_owner_bounds()

        # 3. Scheduler topology snapshot (frozen dataclass -> new instance).
        if tp_phase:
            scheduler.ps = _dc.replace(
                boot_ps,
                tp_rank=world_rank,
                tp_size=n,
                pp_rank=0,
                pp_size=1,
                attn_tp_rank=world_rank,
                attn_tp_size=n,
            )
        else:
            scheduler.ps = boot_ps

        # 4. Cached group handles, re-derived through the ROUTED getters.
        scheduler.tp_group = _ps.get_tp_group()
        scheduler.tp_cpu_group = scheduler.tp_group.cpu_group
        scheduler.attn_tp_group = _ps.get_attn_tp_group()
        scheduler.attn_tp_cpu_group = scheduler.attn_tp_group.cpu_group
        scheduler.pp_group = _ps.get_pp_group()
        # dp-attention is a flip arming guard; the dp routing group is tp.
        scheduler.dp_tp_group = scheduler.tp_group

        # 4b. Scheduler COMPONENTS holding ps / group snapshots (found on
        # the first post-flip serving attempt, 2026-08-08): the request
        # receiver kept the boot ps and relayed requests PP-chain-style
        # while rank 0 ran TP semantics -- one rank in the pool-budget
        # all_reduce, another in the relay's point_to_point recv, wedge.
        # The output streamer's stale ps mis-gated the detokenizer send
        # (heartbeat loss). Both are plain dataclasses over ps + group
        # handles; rebuild them against the freshly-routed handles. The
        # completeness self-check (step 9) pins each one.
        import dataclasses as _dc2

        scheduler.request_receiver = _dc2.replace(
            scheduler.request_receiver,
            ps=scheduler.ps,
            tp_group=scheduler.tp_group,
            tp_cpu_group=scheduler.tp_cpu_group,
            attn_tp_group=scheduler.attn_tp_group,
            attn_tp_cpu_group=scheduler.attn_tp_cpu_group,
        )
        scheduler.output_streamer = _dc2.replace(
            scheduler.output_streamer, ps=scheduler.ps
        )
        if getattr(scheduler, "load_inquirer", None) is not None:
            scheduler.load_inquirer = _dc2.replace(
                scheduler.load_inquirer, ps=scheduler.ps
            )
        # 4c. Census round realign: the detector's cadence counter drifted
        # per-rank under the pp loop; the cutover is group-aligned, so
        # re-zero here or the post-flip detector fires its gloo
        # all_gather_object at per-rank rounds and mispairs with the
        # request broadcasts on the same group FIFO (measured wedge,
        # window-2 boot 13). See CollectiveCensus.realign_round.
        from sglang.srt.distributed.collective_census import census as _census

        _census().realign_round()

        # 5. pp_max_micro_batch_size for the new pp_size (boot formula).
        get_server_args().override(
            "phase_flip.pp_max_micro_batch_size",
            pp_max_micro_batch_size=max(
                scheduler.max_running_requests // scheduler.ps.pp_size, 1
            ),
        )

        # 6. PP loop arrays: re-initialized clean for the new topology
        # (idempotent pure reassignment; reads the NEW ps.pp_size).
        scheduler.init_pp_loop_state()

        # 7. Active stack swap: the forward path follows model_worker.
        scheduler.model_worker = (
            stacks.tp_worker if tp_phase else boot_model_worker
        )
        scheduler.phase_flip_active_stack = PHASE_TP if tp_phase else PHASE_PP

        # 8. Deferred aborts drain in the first post-flip round.
        window = getattr(scheduler, "phase_flip_abort_window", None)
        if window is not None and window.active:
            drained = window.deactivate_and_drain()
            if drained:
                logger.info(
                    "%s drained %d deferred abort(s) after cutover",
                    LOG_PREFIX,
                    drained,
                )

        # 9. Completeness self-check: every snapshot the rebuild list names
        # is verified against the routed source of truth, HERE, before the
        # first post-flip round can touch a stale handle. A missed rebuild
        # is a loud KvReshardError, never later corruption.
        verify_flip_cutover(scheduler, tp_phase)
        logger.warning(
            "%s cutover complete: active stack %s, ps tp=%d pp=%d",
            LOG_PREFIX,
            scheduler.phase_flip_active_stack,
            scheduler.ps.tp_size,
            scheduler.ps.pp_size,
        )

    return _cutover


def verify_flip_cutover(scheduler, tp_phase: bool) -> None:
    """Post-cutover invariants (the coordinator's completeness pin): every
    scheduler snapshot on the 5.3 rebuild list must AGREE with the routed
    source of truth for the now-active phase. Any single stale reference
    -- a cached group handle still pointing at the other phase's group, a
    ps that kept the old topology, a model_worker from the wrong stack --
    fails HERE, loudly, before any round runs on it."""
    from sglang.srt.distributed import parallel_state as _ps

    stale = []
    if _ps.phase_flip_tp_routing_active() != tp_phase:
        stale.append(
            f"module routing active={_ps.phase_flip_tp_routing_active()} "
            f"but tp_phase={tp_phase}"
        )
    expect_tp = _ps.get_tp_group()
    expect_attn = _ps.get_attn_tp_group()
    expect_pp = _ps.get_pp_group()
    if scheduler.tp_group is not expect_tp:
        stale.append("tp_group")
    if scheduler.tp_cpu_group is not expect_tp.cpu_group:
        stale.append("tp_cpu_group")
    if scheduler.attn_tp_group is not expect_attn:
        stale.append("attn_tp_group")
    if scheduler.attn_tp_cpu_group is not expect_attn.cpu_group:
        stale.append("attn_tp_cpu_group")
    if scheduler.pp_group is not expect_pp:
        stale.append("pp_group")
    if scheduler.dp_tp_group is not scheduler.tp_group:
        stale.append("dp_tp_group")
    stacks = scheduler.phase_flip_stacks
    n = len(stacks.vector)
    want_tp_size = n if tp_phase else 1
    want_pp_size = 1 if tp_phase else n
    if scheduler.ps.tp_size != want_tp_size or scheduler.ps.pp_size != want_pp_size:
        stale.append(
            f"ps topology (tp={scheduler.ps.tp_size}, "
            f"pp={scheduler.ps.pp_size}; want tp={want_tp_size}, "
            f"pp={want_pp_size})"
        )
    if scheduler.ps.attn_tp_size != want_tp_size:
        stale.append(f"ps.attn_tp_size ({scheduler.ps.attn_tp_size})")
    want_worker = stacks.tp_worker if tp_phase else scheduler.tp_worker
    if scheduler.model_worker is not want_worker:
        stale.append("model_worker (wrong stack)")
    # Component ps/group snapshots (step 4b): each holder rebuilt at
    # cutover must reference the CURRENT ps object and routed groups --
    # a stale receiver relays requests in the other phase's topology
    # (measured wedge, first post-flip serving attempt 2026-08-08).
    receiver = getattr(scheduler, "request_receiver", None)
    if receiver is not None:
        if receiver.ps is not scheduler.ps:
            stale.append("request_receiver.ps")
        if receiver.attn_tp_group is not scheduler.attn_tp_group:
            stale.append("request_receiver.attn_tp_group")
        if receiver.tp_cpu_group is not scheduler.tp_cpu_group:
            stale.append("request_receiver.tp_cpu_group")
    streamer = getattr(scheduler, "output_streamer", None)
    if streamer is not None and streamer.ps is not scheduler.ps:
        stale.append("output_streamer.ps")
    inquirer = getattr(scheduler, "load_inquirer", None)
    if inquirer is not None and inquirer.ps is not scheduler.ps:
        stale.append("load_inquirer.ps")
    window = getattr(scheduler, "phase_flip_abort_window", None)
    if window is not None and window.active:
        stale.append("abort window still active (drain missed)")
    if stale:
        raise KvReshardError(
            f"{LOG_PREFIX} CUTOVER INCOMPLETE ({'tp' if tp_phase else 'pp'} "
            f"phase): stale after rebuild: {', '.join(stale)}. A stale "
            f"snapshot surviving cutover is the silent-corruption class "
            f"this check exists to catch -- refusing to run a round on it."
        )


def build_phase_flip_runtime(scheduler) -> "PhaseFlipRuntime":
    """Factory mirroring build_kv_reshard_runtime (kv_reshard.py): wires
    the scheduler's real state into PhaseFlipRuntime. Called lazily from
    the first scheduler round (house pattern); by then the boot builder
    has installed scheduler.phase_flip_stacks."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        get_world_group,
    )
    from sglang.srt.managers.kv_pressure_runtime import default_collective_min
    from sglang.srt.managers.kv_reshard import _dist_exchange

    stacks = scheduler.phase_flip_stacks
    if stacks is None:
        raise KvReshardError(
            "build_phase_flip_runtime before build_phase_flip_tp_stack "
            "(the boot builder owns pools, arena and images)"
        )
    server_args = scheduler.server_args
    flip_tp = get_phase_flip_group("tp")
    world = get_world_group()

    pp_pool = scheduler.tp_worker.model_runner.token_to_kv_pool
    tp_pool = stacks.tp_worker.model_runner.token_to_kv_pool
    for name, pool in (("PP", pp_pool), ("TP", tp_pool)):
        if not hasattr(pool, "full_kv_pool"):
            raise KvReshardError(
                f"the {name} stack's pool {type(pool).__name__} has no "
                f"full_kv_pool; the flip moves hybrid-model full-attention "
                f"KV only (DESIGN_631 scope)"
            )
    pp_full = pp_pool.full_kv_pool
    tp_full = tp_pool.full_kv_pool
    pp_view = KvPoolView(pp_full.k_buffer, pp_full.v_buffer)
    tp_view = KvPoolView(tp_full.k_buffer, tp_full.v_buffer)

    # Global full-attention geometry from the TP stack's config (pp=1 ->
    # unmutated; the PP stack's was rewritten to its stage-local slice).
    # full_attention_layer_ids is a property of the HYBRID HF text config
    # (Qwen3NextConfig etc.), not of sglang's ModelConfig wrapper -- the
    # attention registry reads it via runner.mambaish_config, mirror that
    # (first real-metal flip boot, 2026-08-08).
    tp_model_config = stacks.tp_worker.model_config
    full_ids = list(tp_model_config.hf_text_config.full_attention_layer_ids)
    layer_map = derive_pp_full_attn_layer_map(
        full_ids,
        int(tp_model_config.num_hidden_layers),
        int(server_args.pp_size),
    )

    return PhaseFlipRuntime(
        n_ranks=world.world_size,
        rank=world.rank_in_group,
        layer_map=layer_map,
        n_layers=len(full_ids),
        tp_vector=stacks.vector,
        boot_phase=PHASE_PP,
        consensus_interval=int(
            getattr(server_args, "kv_reshard_consensus_interval", 8)
        ),
        park_deadline_s=park_deadline_s(),
        collective_min=default_collective_min(flip_tp.cpu_group),
        exchange=_dist_exchange(flip_tp.device_group, pp_view.device),
        pp_pool_view=pp_view,
        tp_pool_view=tp_view,
        live_slots_fn=build_flip_live_slots_fn(scheduler),
        ready_fn=build_flip_quiescence_fn(scheduler),
        cutover_fn=build_production_flip_cutover(scheduler),
        # DESIGN_631 3.6 order inside the no-return region: GDN state move
        # (5.3b mover -- its preconditions re-validate on every flip and
        # refuse loudly, the reachable-refusal contract), then the arena
        # refill. The full-attn KV move ran before these by the runtime.
        pre_cutover_fns=(
            _build_gdn_leg(scheduler),
            stacks.refill,
        ),
        guards=flip_blocking_guards(scheduler),
    )


def _build_gdn_leg(scheduler) -> Callable[[str], None]:
    from sglang.srt.managers.gdn_flip_mover import build_gdn_flip_mover

    return build_gdn_flip_mover(scheduler)


class PhaseFlipRuntime:
    """Drives one group's PP<->TP KV layout flip at a quiescent boundary.

    Injectables mirror ``KvReshardRuntime`` so the hermetic tests drive
    REAL threads through mock channels: ``collective_min`` is the
    consensus channel, ``exchange`` the pairwise byte channel,
    ``pp_pool_view``/``tp_pool_view`` the two resident pools (PP view
    layers = this stage's ordinals ascending; TP view layers = ALL
    ordinals ascending), ``live_slots_fn`` the replicated live slot
    enumeration (tree values UNION parked requests' rows -- DESIGN_631
    section 3.5), ``ready_fn`` the flip quiescence predicate,
    ``cutover_fn(direction)`` the snapshot-cache installer,
    ``pre_cutover_fns`` the ordered extra movers (weights arena, GDN
    state) executed inside the no-return region before cutover.
    """

    def __init__(
        self,
        *,
        n_ranks: int,
        rank: int,
        layer_map: Sequence[Sequence[int]],
        n_layers: int,
        tp_vector: Sequence[int],
        boot_phase: str = PHASE_PP,
        consensus_interval: int = 8,
        park_deadline_s: float = DEFAULT_PARK_DEADLINE_S,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        exchange: Optional[
            Callable[[Dict[int, torch.Tensor], Dict[int, int]], Dict[int, torch.Tensor]]
        ] = None,
        pp_pool_view: Optional[KvPoolView] = None,
        tp_pool_view: Optional[KvPoolView] = None,
        live_slots_fn: Optional[Callable[[], torch.Tensor]] = None,
        ready_fn: Optional[Callable[[], bool]] = None,
        cutover_fn: Optional[Callable[[str], None]] = None,
        pre_cutover_fns: Sequence[Callable[[str], None]] = (),
        guards: Sequence[str] = (),
        clock: Callable[[], float] = time.perf_counter,
    ):
        if n_ranks < 2:
            raise KvReshardError(
                f"a phase flip needs a multi-rank group, got n_ranks={n_ranks}"
            )
        if not (0 <= int(rank) < n_ranks):
            raise KvReshardError(f"rank {rank} out of range for {n_ranks} ranks")
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if collective_min is None or exchange is None:
            raise KvReshardError(
                "a phase flip needs both a consensus channel (collective_min) "
                "and a pairwise byte channel (exchange); running without them "
                "would turn the first honest divergence into a hang instead "
                "of a loud error."
            )
        missing = [
            name
            for fn, name in (
                (pp_pool_view, "pp_pool_view"),
                (tp_pool_view, "tp_pool_view"),
                (live_slots_fn, "live_slots_fn"),
                (ready_fn, "ready_fn"),
                (cutover_fn, "cutover_fn"),
            )
            if fn is None
        ]
        if missing:
            raise KvReshardError(f"PhaseFlipRuntime needs {', '.join(missing)}")
        if boot_phase not in (PHASE_PP, PHASE_TP):
            raise KvReshardError(f"unknown boot phase {boot_phase!r}")

        self._n = int(n_ranks)
        self._rank = int(rank)
        self._map = validate_layer_map(layer_map, n_layers)
        self._n_layers = int(n_layers)
        self._vec = tuple(int(x) for x in tp_vector)
        if len(self._map) != self._n or len(self._vec) != self._n:
            raise KvReshardError(
                f"layer map has {len(self._map)} stages and the vector "
                f"{self._vec} has {len(self._vec)} entries, but the group "
                f"has {self._n} ranks -- the flip reuses the SAME ranks"
            )
        my_layers = self._map[self._rank]
        if pp_pool_view.num_layers != len(my_layers):
            raise KvReshardError(
                f"PP pool view has {pp_pool_view.num_layers} layers but "
                f"stage {self._rank} owns {len(my_layers)} "
                f"({my_layers}); the view must cover exactly this stage's "
                f"ordinals, ascending"
            )
        if tp_pool_view.num_layers != self._n_layers:
            raise KvReshardError(
                f"TP pool view has {tp_pool_view.num_layers} layers but the "
                f"model has {self._n_layers} full-attention layers; the TP "
                f"layout holds every ordinal on every rank"
            )
        self._fp = _config_fingerprint(self._map, self._vec)
        self._phase = boot_phase
        self._interval = int(consensus_interval)
        self._collective_min = collective_min
        self._exchange = exchange
        self._pp = pp_pool_view
        self._tp = tp_pool_view
        self._live_slots_fn = live_slots_fn
        self._ready_fn = ready_fn
        self._cutover_fn = cutover_fn
        self._pre_cutover_fns = tuple(pre_cutover_fns)
        self.blocking_guards = tuple(guards)
        self._clock = clock

        self._round = 0
        self._epoch = 0
        self._pending: Optional[str] = None
        self._last_hold_reason: Optional[str] = None
        self.desync_checks = 0
        self.completed = 0
        self.last_stats: Optional[dict] = None
        #: Wall-clock bound on the parked wait; see DEFAULT_PARK_DEADLINE_S.
        self._park_deadline_s = float(park_deadline_s)
        #: Clock reading of the moment this rank armed, or None when idle.
        self._armed_at: Optional[float] = None
        #: Flips abandoned because the park deadline expired. A counter, so
        #: "this never happens in practice" stops being an assumption.
        self.park_deadline_aborts = 0

        logger.info(
            "%s armed at boot: rank %d/%d, phase %s, layer map %s, vector "
            "%s, consensus every %d rounds%s",
            LOG_PREFIX,
            self._rank,
            self._n,
            self._phase,
            self._map,
            self._vec,
            self._interval,
            (
                "; guards BLOCKING arming: " + ", ".join(self.blocking_guards)
                if self.blocking_guards
                else ""
            ),
        )

    # -- state ---------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def pending(self) -> Optional[str]:
        return self._pending

    # -- arming (replicated callers) -----------------------------------------
    def arm(self, direction: str, source: str) -> Tuple[bool, str]:
        """Arm a flip. Replicated call; the consensus round commits it once
        every rank is armed AND ready. Returns (ok, msg)."""
        if self.blocking_guards:
            msg = (
                f"phase flip refused (guards): "
                f"{', '.join(self.blocking_guards)}"
            )
            logger.warning("%s %s", LOG_PREFIX, msg)
            return False, msg
        if direction not in _DIR_ID:
            return False, f"unknown flip direction {direction!r}"
        want = _DIR_OF_PHASE[self._phase]
        if direction != want:
            return False, (
                f"flip {direction} refused: current phase is {self._phase}, "
                f"the only legal transition is {want}"
            )
        if self._pending is not None and self._pending != direction:
            logger.warning(
                "%s re-arming %s -> %s (source %s)",
                LOG_PREFIX,
                self._pending,
                direction,
                source,
            )
        self._pending = direction
        # The park clock starts at ARMING, not at the first unparked round:
        # the deadline bounds how long the requests are held, and they are
        # held from the moment this rank starts withholding work.
        self._armed_at = self._clock()
        msg = (
            f"phase flip armed: {direction} (source {source}); commits at "
            f"the next consensus boundary where every rank is quiescent, or "
            f"is abandoned after {self._park_deadline_s:g}s parked"
        )
        logger.warning("%s %s", LOG_PREFIX, msg)
        return True, msg

    # -- the per-round hook ---------------------------------------------------
    def on_round(self, require_armed_and_parked: bool = False) -> Optional[dict]:
        """One scheduler round; see KvReshardRuntime.on_round. Returns move
        stats when a flip executed this round, else ``None``.

        ``require_armed_and_parked`` is the PP-phase entry gate (measured
        wedges 2026-08-08, boots 9 and 10): under event_loop_pp the local
        round counters of the ranks diverge in ABSOLUTE value (pipeline
        fill, conditional per-slot ops), so ANY blocking reduction entered
        at a local cadence can pair with a peer blocked in a pipeline recv
        whose satisfying send sits behind this rank's reduction -- moving
        the hook inside the iteration only moved the wedge. With the gate,
        an UNARMED rank performs NO collective at all (there is nothing to
        agree on; arming state arrives via the broadcast RPC on every
        rank), and an armed rank enters only once it is locally PARKED
        (ready_fn: drained microbatches, no partial chunk) -- a parked
        rank owes no pipeline send, so no recv/reduction cycle can close.
        Peers converge on their own arm+drain, MIN-skew is legal, and the
        liveness bound turns a lost peer into a loud error. A flip under
        continuous load needs the posted-async two-phase consensus -- a
        named follow-up, not this gate.

        The wait is BOUNDED (see DEFAULT_PARK_DEADLINE_S): a rank armed
        past the deadline without parking joins the reduction anyway
        carrying ``expired``, and every participating rank abandons the
        flip on the reduced maximum. Abandoning the flip is the whole
        point -- the parked requests are never abandoned."""
        self._round += 1
        armed = 1 if self._pending is not None else 0
        ready = 1 if (armed and self._ready_fn()) else 0
        expired = 1 if self._park_expired(armed, ready) else 0
        # The PP-phase entry gate, widened by the deadline: an armed rank
        # enters once it is PARKED, or -- if it has been armed past the
        # deadline without ever parking -- to carry that fact into the
        # consensus. Entering unparked is what makes the abandonment
        # GROUP-AGREED: the peers are already blocked in this reduction
        # waiting for exactly this rank, so the flag reaches them, every
        # rank abandons the same flip in the same round, and nobody is left
        # armed against a disarmed peer. It is safe here because an armed
        # rank has been withholding new work for the whole deadline, so it
        # owes no fresh pipeline send.
        if require_armed_and_parked and not (armed and (ready or expired)):
            return None
        if not require_armed_and_parked and self._round % self._interval != 0:
            return None
        dir_id = _DIR_ID[self._pending] if self._pending is not None else 0
        payload = _encode(
            [armed, ready, expired, self._epoch, dir_id, self._fp, *self._vec]
        )
        self.desync_checks += 1
        reduced = self._collective_min(payload)
        if len(reduced) != len(payload):
            raise KvReshardError(
                f"consensus channel returned {len(reduced)} values for a "
                f"{len(payload)}-value payload; the channel contract is "
                f"element-wise MIN of the packed proposal."
            )
        fields = [
            "armed",
            "ready",
            "expired",
            "epoch",
            "direction",
            "config_fp",
        ] + [f"vector[{i}]" for i in range(self._n)]
        lo = {f: reduced[2 * i] for i, f in enumerate(fields)}
        hi = {f: -reduced[2 * i + 1] for i, f in enumerate(fields)}

        # Equality family: epoch + config fingerprint + vector ALWAYS
        # (boot config); direction once every rank is armed.
        eq_checked = ["epoch", "config_fp"] + [
            f"vector[{i}]" for i in range(self._n)
        ]
        if lo["armed"] == 1:
            eq_checked.append("direction")
        mismatches = [
            f"{f}: min={lo[f]} max={hi[f]}" for f in eq_checked if lo[f] != hi[f]
        ]
        if mismatches:
            raise KvReshardError(
                f"{LOG_PREFIX} DESYNC at round {self._round}: the ranks "
                f"disagree on the flip state ({'; '.join(mismatches)}; this "
                f"rank: armed={armed} pending={self._pending} "
                f"epoch={self._epoch} phase={self._phase}). A flip that "
                f"disagrees across ranks must fail loudly HERE, before any "
                f"rank moves a byte under the wrong layout."
            )
        # Park deadline, decided on the MAX: one rank out of time is enough
        # to abandon the flip, and every rank in this reduction reads the
        # same max, so the abandonment is unanimous by construction.
        # Checked before the armed/ready holds -- those are the states the
        # deadline exists to stop waiting in.
        if hi["expired"] == 1:
            return self._abandon_parked_flip(ready)

        if lo["armed"] == 0:
            if hi["armed"] == 1:
                self._hold("waiting for every rank to arm (delivery skew)")
            return None
        if lo["ready"] == 0:
            self._hold(
                f"armed ({self._pending}), waiting for a group-wide "
                f"quiescent boundary (this rank ready={ready})"
            )
            return None
        self._last_hold_reason = None
        return self._execute()

    def _park_expired(self, armed: int, ready: int) -> bool:
        """Has this rank been armed-but-unparked past the deadline?

        Wall clock, not a round count: rounds are what the PP loop makes
        incomparable across ranks in the first place, and the quantity the
        operator cares about is how long a request may be held. The reading
        is rank-local and does NOT need to be replicated -- one rank
        raising the flag is enough, because the DECISION to abandon is
        taken from the reduced maximum in on_round, which every
        participating rank reads identically.
        """
        if not armed or ready or self._park_deadline_s <= 0:
            return False
        if self._armed_at is None:
            return False
        return (self._clock() - self._armed_at) >= self._park_deadline_s

    def _abandon_parked_flip(self, ready: int) -> None:
        """Give up on an armed flip that never reached quiescence.

        Disarms and returns to serving. Deliberately NOT an exception: the
        flip is the optional thing here, the requests are not. A raise
        would climb into the event loop and take the instance down with it,
        which is precisely the outcome this deadline exists to prevent --
        the parked requests would die with it.
        """
        waited = (
            self._clock() - self._armed_at if self._armed_at is not None else float("nan")
        )
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._last_hold_reason = None
        self.park_deadline_aborts += 1
        logger.error(
            "%s FLIP ABANDONED: %s was armed for %.1fs without the group "
            "reaching a quiescent boundary (deadline %gs; this rank "
            "ready=%d). The requests are NOT affected -- they were parked, "
            "not aborted, and serving resumes on the %s stack now. A rank "
            "that cannot park is holding work that never drains: look for a "
            "microbatch or a chunked prefill that never completes. Re-arm "
            "to try again.",
            LOG_PREFIX,
            direction,
            waited,
            self._park_deadline_s,
            ready,
            self._phase,
        )
        return None

    def _hold(self, reason: str) -> None:
        if reason != self._last_hold_reason:
            logger.info("%s hold: %s", LOG_PREFIX, reason)
            self._last_hold_reason = reason

    # -- pool/layer adapters --------------------------------------------------
    def _src_dst(self, direction: str) -> Tuple[KvPoolView, KvPoolView]:
        return (self._pp, self._tp) if direction == PP_TO_TP else (self._tp, self._pp)

    def _src_layer_idx(self, direction: str, ordinal: int) -> int:
        """Pool-local layer index of a global ordinal in MY sending pool."""
        if direction == PP_TO_TP:
            return self._map[self._rank].index(ordinal)
        return ordinal

    def _dst_layer_idx(self, direction: str, ordinal: int) -> int:
        if direction == PP_TO_TP:
            return ordinal
        return self._map[self._rank].index(ordinal)

    # -- the move -------------------------------------------------------------
    def _execute(self) -> dict:
        direction = self._pending
        assert direction is not None
        t0 = self._clock()
        slots = self._live_slots_fn()
        slots = torch.unique(slots.detach().to("cpu", torch.int64))
        tr: PhaseFlipTransition = build_phase_flip_transition(
            slots, self._map, self._n_layers, self._vec, self._rank, direction
        )

        src, dst = self._src_dst(direction)
        # Bounds BEFORE any byte moves: both layouts' pools are pre-sized
        # at boot; an overflow here is a sizing bug, not a runtime state.
        if tr.max_pp_row() >= self._pp.num_rows:
            raise KvReshardError(
                f"{LOG_PREFIX} flip needs PP row {tr.max_pp_row()} but the "
                f"PP pool holds {self._pp.num_rows} rows (sizing bug: the "
                f"PP pool must cover every live global slot id)"
            )
        if tr.max_tp_row() >= self._tp.num_rows:
            raise KvReshardError(
                f"{LOG_PREFIX} flip needs TP row {tr.max_tp_row()} but the "
                f"TP pool holds {self._tp.num_rows} rows (sizing bug: the "
                f"TP pool must cover the compact rows of vector {self._vec})"
            )

        # PACK (reads only): per peer, layers ascending, one row list.
        t_read0 = self._clock()
        outgoing_payloads: Dict[int, torch.Tensor] = {}
        for peer in tr.send_layers:
            # read_rows returns [n, row_nbytes] uint8, K bytes then V.
            parts = [
                src.read_rows(
                    self._src_layer_idx(direction, f), tr.send_rows[peer]
                ).reshape(-1)
                for f in tr.send_layers[peer]
            ]
            flat = torch.cat(parts)
            outgoing_payloads[peer] = torch.cat([flat, _checksum(flat)])
        read_ms = (self._clock() - t_read0) * 1000.0

        # Expected incoming sizes from MY OWN pool's row widths -- the
        # runtime pin of row byte-compatibility across layouts.
        incoming_nbytes: Dict[int, int] = {}
        for peer in tr.recv_layers:
            n = int(tr.recv_rows[peer].numel())
            nbytes = sum(
                dst.row_nbytes(self._dst_layer_idx(direction, f)) * n
                for f in tr.recv_layers[peer]
            )
            incoming_nbytes[peer] = nbytes + _CHECKSUM_BYTES

        # EXCHANGE (pools still untouched): failure up to and including
        # checksum verification aborts with both pools byte-identical.
        t_xfer0 = self._clock()
        received = self._exchange(outgoing_payloads, incoming_nbytes)
        xfer_ms = (self._clock() - t_xfer0) * 1000.0
        incoming_data: Dict[int, torch.Tensor] = {}
        for peer, rows in tr.recv_rows.items():
            payload = received.get(peer)
            if payload is None or payload.numel() != incoming_nbytes[peer]:
                got = 0 if payload is None else payload.numel()
                raise KvReshardError(
                    f"{LOG_PREFIX} exchange returned {got} bytes from peer "
                    f"{peer}, expected {incoming_nbytes[peer]} -- size "
                    f"mismatch means the layouts' row formats or the "
                    f"payload convention diverged"
                )
            data = payload[:-_CHECKSUM_BYTES]
            want = int(payload[-_CHECKSUM_BYTES:].clone().view(torch.int64).item())
            have = uint8_checksum(data)
            if want != have:
                raise KvReshardError(
                    f"{LOG_PREFIX} payload checksum mismatch from peer "
                    f"{peer}: sender {want}, receiver {have} -- refusing to "
                    f"scatter."
                )
            incoming_data[peer] = data

        # WRITE (no-return region): local first, then incoming. Source and
        # destination are different pools; targets are disjoint (injective
        # row map), so order is free -- kept deterministic anyway.
        t_write0 = self._clock()
        local_src = (
            tr.local_pp_rows if direction == PP_TO_TP else tr.local_tp_rows
        )
        local_dst = (
            tr.local_tp_rows if direction == PP_TO_TP else tr.local_pp_rows
        )
        for f in tr.local_layers:
            data = src.read_rows(self._src_layer_idx(direction, f), local_src)
            dst.write_rows(self._dst_layer_idx(direction, f), local_dst, data)
        for peer, rows in tr.recv_rows.items():
            n = int(rows.numel())
            offset = 0
            for f in tr.recv_layers[peer]:
                li = self._dst_layer_idx(direction, f)
                width = dst.row_nbytes(li)
                chunk = incoming_data[peer][offset : offset + n * width]
                dst.write_rows(li, rows, chunk.view(n, width))
                offset += n * width
        write_ms = (self._clock() - t_write0) * 1000.0

        # EXTRA MOVERS (weights arena, GDN state) then CUTOVER.
        for fn in self._pre_cutover_fns:
            fn(direction)
        self._cutover_fn(direction)
        self._phase = _PHASE_AFTER[direction]
        self._pending = None
        self._armed_at = None
        self._epoch += 1
        self.completed += 1
        total_ms = (self._clock() - t0) * 1000.0
        stats = {
            "direction": direction,
            "phase": self._phase,
            "epoch": self._epoch,
            "live_slots": tr.total_slots,
            "outgoing_cells": tr.outgoing_cells,
            "incoming_cells": tr.incoming_cells,
            "sent_bytes": sum(int(t.numel()) for t in outgoing_payloads.values()),
            "received_bytes": sum(incoming_nbytes.values()),
            "read_ms": read_ms,
            "exchange_ms": xfer_ms,
            "write_ms": write_ms,
            "total_ms": total_ms,
        }
        self.last_stats = stats
        logger.warning(
            "%s DONE %s (epoch %d) in %.1f ms: %d live slots, sent %d "
            "cells / %.2f MiB, received %d cells / %.2f MiB (read %.1f ms, "
            "exchange %.1f ms, write %.1f ms)",
            LOG_PREFIX,
            direction,
            self._epoch,
            total_ms,
            tr.total_slots,
            tr.outgoing_cells,
            stats["sent_bytes"] / 1048576.0,
            tr.incoming_cells,
            stats["received_bytes"] / 1048576.0,
            read_ms,
            xfer_ms,
            write_ms,
        )
        return stats
