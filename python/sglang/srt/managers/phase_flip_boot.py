# SPDX-License-Identifier: Apache-2.0
"""Boot-time construction of the #631 phase flip's TP decode stack.

The server boots as the PP topology (pp_size ranks, tp=1) through the
normal path. This module then builds, in the SAME process on every rank,
the SECONDARY stack that serves the TP decode phase:

1. install the uneven-TP shard plan and the weighted-DCP token vector
   (both derived from --phase-flip-tp-vector, process-global, inert for
   the already-constructed PP stack whose layers cached tp_size=1);
2. snapshot the PP stack's checkpoint weights to a host image and free
   their device storages (the boot-order enabler: PP originals + TP
   originals + arena would not fit the 5090 together);
3. build a TP-shaped TpModelWorker under the flip group routing
   (``set_phase_flip_tp_active``) plus the ``get_parallel().override``
   geometry scope -- the double mechanism is load-bearing: the contextvar
   feeds construction-time caching (weight sharding), the module routing
   feeds forward-time collectives (``tensor_model_parallel_all_reduce``
   reaches groups through ``get_tp_group()``, NOT the contextvar; without
   the routing the TP stack would all-reduce over the primary tp=1 group,
   a silent no-op);
4. pack the TP weights into the boot-allocated shared arena (fixed
   addresses for process life) and image them to host;
5. allocate the TP stack's pools, backends and decode CUDA graphs while
   the TP bytes are live in the arena (graphs bake the arena addresses --
   pin 2: ONLY this stack captures decode graphs);
6. assert full-attention KV row byte-compatibility between the two
   resident pools (pin 3, also pinned hermetically);
7. rebind the PP parameters to their arena views and refill the arena
   from the PP host image -- the boot phase (PP) is live again.

A flip is then: KV/GDN state move on the #297 envelope + ONE contiguous
arena refill from the other layout's image + cutover (PhaseFlipRuntime).

Everything here runs rank-uniformly at boot, gated on
``--enable-phase-flip``; the default path never imports this module.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.model_executor.weights_arena import (
    ArenaLayout,
    allocate_arena,
    arena_refill,
    bind_arena_views,
    image_from_tensors,
    pack_into_arena,
    plan_arena_layout,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-BOOT"

# Non-checkpoint device parameters excluded from the arena (they persist
# outside it for process life; DESIGN_631 3.3). Matched as substrings of
# the parameter name, the marlin workspace family being the known member.
NON_CHECKPOINT_NAME_PATTERNS: Tuple[str, ...] = ("workspace",)


class PhaseFlipBootError(KvReshardError):
    """Loud failure of the phase-flip boot family (#631)."""


def parse_flip_vector(server_args) -> List[int]:
    vec = [int(x) for x in server_args.phase_flip_tp_vector.split(",")]
    if len(vec) != server_args.pp_size:
        raise PhaseFlipBootError(
            f"flip vector {vec} has {len(vec)} entries but pp_size is "
            f"{server_args.pp_size}"
        )
    return vec


# Fields the TP-stack copy overrides, and ONLY these (the #470 REACH
# discipline: every context reader inside the build sees exactly these
# values changed and nothing else). Pinned by test.
TP_STACK_OVERRIDDEN_FIELDS: Tuple[str, ...] = (
    "tp_size",
    "pp_size",
    "rank_tp_ratio",
    "dcp_size",
    "pp_layer_ratio",
    "pp_stage_ratio",
    "enable_phase_flip",
    "phase_flip_tp_vector",
)


def derive_tp_stack_server_args(server_args):
    """The TP decode phase's ServerArgs: a deepcopy of the boot args with
    the geometry rotated from (tp=1, pp=N) to (tp=N, dcp=N, pp=1).

    V1 rule: ONE vector drives both the weight shard plan (rank_tp_ratio)
    and the token owner rule (the DCP vector) -- the runbook's validated
    ``--rank-tp-ratio 30,17,17`` uneven-TP lane, reached exactly as a
    production TP boot would reach it. The copy clears the flip flags: it
    DESCRIBES the TP-shaped stack, it does not enable a nested flip."""
    import copy

    vec = parse_flip_vector(server_args)
    tp_args = copy.deepcopy(server_args)
    tp_args.tp_size = len(vec)
    tp_args.pp_size = 1
    tp_args.rank_tp_ratio = list(vec)
    tp_args.dcp_size = len(vec)
    tp_args.pp_layer_ratio = None
    tp_args.pp_stage_ratio = None
    tp_args.enable_phase_flip = False
    tp_args.phase_flip_tp_vector = None
    return tp_args


@contextmanager
def phase_flip_tp_scope(world_rank: int, n: int):
    """Geometry scope for building/running the TP stack.

    Contextvar override for construction-time caching + module routing for
    forward-time collectives (see module docstring). ``world_rank`` is this
    process's flat world rank, which under the primary (tp=1, pp=N)
    topology equals pp_rank and IS the flip-TP rank."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        set_phase_flip_tp_active,
    )
    from sglang.srt.runtime_context import get_parallel

    flip_tp = get_phase_flip_group("tp")
    flip_dcp = get_phase_flip_group("dcp")
    flip_pp = get_phase_flip_group("pp")
    set_phase_flip_tp_active(True)
    # --pp-layer-ratio exports SGLANG_PP_LAYER_PARTITION process-wide; the
    # TP stack builds with pp_size=1 and get_pp_indices would refuse a
    # 3-way partition (found on the first real-metal flip boot,
    # 2026-08-08). Mask it for the build only; the PP stack read it at
    # primary init and never re-reads it at forward time.
    pp_partition_env = os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
    try:
        with get_parallel().override(
            tp_size=n,
            tp_rank=world_rank,
            moe_tp_size=n,
            moe_tp_rank=world_rank,
            moe_ep_size=1,
            moe_ep_rank=0,
            attn_tp_size=n,
            attn_tp_rank=world_rank,
            dcp_enabled=True,
            dcp_size=n,
            dcp_rank=world_rank,
            attn_dcp_size=n,
            attn_dcp_rank=world_rank,
            pp_size=1,
            pp_rank=0,
            tp_group=flip_tp,
            dcp_group=flip_dcp,
            pp_group=flip_pp,
            attn_tp_group=flip_tp,
        ):
            yield
    finally:
        set_phase_flip_tp_active(False)
        if pp_partition_env is not None:
            os.environ["SGLANG_PP_LAYER_PARTITION"] = pp_partition_env


def checkpoint_param_dict(model) -> Dict[str, torch.nn.Parameter]:
    """Named checkpoint parameters of a model, non-checkpoint families
    excluded by name (NON_CHECKPOINT_NAME_PATTERNS)."""
    return {
        name: p
        for name, p in model.named_parameters()
        if not any(pat in name for pat in NON_CHECKPOINT_NAME_PATTERNS)
    }


def snapshot_and_free(
    named: Dict[str, torch.nn.Parameter], layout: ArenaLayout, pin: bool
) -> torch.Tensor:
    """Host image of ``named`` under ``layout``, then free every device
    original by rebinding ``param.data`` to an empty placeholder.

    Between this call and the later ``bind_arena_views`` + ``arena_refill``
    the parameters are DEAD -- any forward would fail loudly on the 0-sized
    placeholder, which is the wanted behavior (nothing may run mid-boot)."""
    image = image_from_tensors(named, layout, pin=pin)
    freed = set()
    for name, param in named.items():
        key = param.data.untyped_storage().data_ptr()
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        freed.add(key)
    logger.info(
        "%s snapshotted %d params (%d storages, %.2f MiB image) and freed "
        "their device storages",
        LOG_PREFIX,
        len(named),
        len(freed),
        image.numel() / 1048576.0,
    )
    return image


def _full_attn_row_schema(pool) -> Tuple[int, int, int, str]:
    """(head_num, head_dim, elem_bytes, dtype_str) of a hybrid pool's
    full-attention KV rows -- the byte schema of one token's K (and V)
    row in one layer."""
    full = pool.full_kv_pool
    k0 = full.k_buffer[0]
    # [rows, head_num, head_dim] (page_size folded into rows at size 1).
    head_num = int(k0.shape[-2])
    head_dim = int(k0.shape[-1])
    return head_num, head_dim, k0.element_size(), str(k0.dtype)


def assert_row_schema_compatible(pp_pool, tp_pool) -> None:
    """Operator pin 3 (DESIGN_631 3.6a): the flip's KV move is a pure row
    redistribution ONLY IF both layouts' full-attention rows are the same
    bytes. That holds because the fork's weighted DCP REPLICATES KV heads
    (uneven_dcp_kv_replicated -> get_total_num_kv_heads); it breaks the
    moment the TP pool is head-sharded. Checked on the real pools at boot,
    and hermetically from the real config in the pin-3 test."""
    pp_schema = _full_attn_row_schema(pp_pool)
    tp_schema = _full_attn_row_schema(tp_pool)
    if pp_schema != tp_schema:
        raise PhaseFlipBootError(
            f"full-attention KV row schemas DIVERGE between the phase "
            f"layouts: PP (head_num, head_dim, elem_bytes, dtype) = "
            f"{pp_schema}, TP = {tp_schema}. The flip's KV move is only a "
            f"row redistribution when both layouts store identical row "
            f"bytes (weighted-DCP head replication); refusing to boot a "
            f"flip that would scatter mis-shaped rows."
        )


@dataclass
class PhaseFlipStacks:
    """Everything the scheduler flip protocol needs from the boot build."""

    tp_worker: object
    arena: torch.Tensor
    layout_pp: ArenaLayout
    layout_tp: ArenaLayout
    image_pp: torch.Tensor
    image_tp: torch.Tensor
    vector: Tuple[int, ...]

    def refill(self, direction: str) -> None:
        """The weights leg of a flip: one contiguous H2D refill of the
        arena with the TARGET phase's image (pre_cutover seam of
        PhaseFlipRuntime; direction is phase_flip_plan.PP_TO_TP or
        TP_TO_PP)."""
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP

        if direction == PP_TO_TP:
            arena_refill(self.arena, self.layout_tp, self.image_tp)
        elif direction == TP_TO_PP:
            arena_refill(self.arena, self.layout_pp, self.image_pp)
        else:
            raise PhaseFlipBootError(f"unknown flip direction {direction!r}")


def build_phase_flip_tp_stack(scheduler) -> PhaseFlipStacks:
    """Build the TP decode stack beside the fully-constructed PP stack.

    Called from Scheduler.init_model_worker AFTER the primary worker's
    pools, backends and (eager-only, pin 2) graph init, BEFORE the
    post-capture pool resize -- the resize must see the TP stack's VRAM
    as taken, not as free. Rank-uniform: every collective below runs on
    every rank in the same order."""
    from sglang.srt.distributed.parallel_state import (
        get_world_group,
        phase_flip_groups_initialized,
    )
    from sglang.srt.distributed.utils import (
        set_cp_token_ratios,
        set_tp_partition_ratios,
    )
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.runtime_context import get_context, get_server_args

    server_args = scheduler.server_args
    if not server_args.enable_phase_flip:
        raise PhaseFlipBootError("build_phase_flip_tp_stack without the flag")
    if not phase_flip_groups_initialized():
        raise PhaseFlipBootError(
            "phase-flip secondary groups were not built at primary init "
            "(initialize_phase_flip_secondary_groups)"
        )
    if not server_args.uneven_weighted_dcp_enabled():
        raise PhaseFlipBootError(
            "the flip's TP phase token-shards KV under the WEIGHTED owner "
            "rule; set SGLANG_UNEVEN_DCP=1 and SGLANG_UNEVEN_DCP_WEIGHTED=1 "
            "(the rig runbook's uneven-DCP env pair) -- refusing to build a "
            "stack whose owner rule would silently fall back to even-modulo"
        )
    vec = parse_flip_vector(server_args)
    n = len(vec)
    world_rank = get_world_group().rank_in_group
    primary_runner = scheduler.tp_worker.model_runner
    device = primary_runner.device

    # 1. Process-global uneven plan + token vector. Installed only NOW --
    # the PP stack was built with both absent (byte-identical primary
    # path) and its runtime reads are gated on its own cached tp_size=1 /
    # dcp_size=1. Step-6 watch item: any PP-phase code path that consults
    # the global plan with a non-rank-local size would be a bug HERE.
    set_tp_partition_ratios(list(vec), families=None)
    set_cp_token_ratios(list(vec))

    # 2. Snapshot the PP checkpoint weights to host, free device originals
    # (VRAM ledger: PP originals + TP originals + arena never coexist).
    pp_named = checkpoint_param_dict(primary_runner.model)
    layout_pp = plan_arena_layout(pp_named)
    image_pp = snapshot_and_free(pp_named, layout_pp, pin=True)
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Build the TP-shaped worker under the flip scope. The server-args
    # copy is PUBLISHED for the whole build (the #470 lesson: context
    # readers must see the stack's own geometry, not the target's).
    tp_args = derive_tp_stack_server_args(server_args)
    ctx = get_context()
    saved_args = get_server_args()
    ctx.set_server_args(tp_args)
    try:
        with phase_flip_tp_scope(world_rank, n):
            # NO pool sharing at construction: the TP stack builds its OWN
            # HybridReqToTokenPool so its mamba pool gets the TP geometry
            # (ALL linear layers, head-sharded) through the normal path --
            # the shared PP req pool carries a PP-shaped mamba pool (stage
            # layers, full heads), which would silently mis-shape every
            # linear-state access. Request-mapping SHARING happens below by
            # tensor rebind instead.
            tp_worker = TpModelWorker(
                server_args=tp_args,
                gpu_id=scheduler.ps.gpu_id,
                tp_rank=world_rank,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=scheduler.tp_worker.attn_cp_rank,
                moe_dp_rank=scheduler.tp_worker.moe_dp_rank,
                dp_rank=scheduler.tp_worker.dp_rank,
                nccl_port=scheduler.tp_worker.nccl_port,
                is_draft_worker=True,
                is_phase_flip_tp_stack=True,
            )

            # 4. Arena: sized max(both layouts), fixed for process life.
            tp_named = checkpoint_param_dict(tp_worker.model_runner.model)
            layout_tp = plan_arena_layout(tp_named)
            arena = allocate_arena(
                max(layout_pp.total_bytes, layout_tp.total_bytes),
                tp_worker.model_runner.device,
            )
            pack_into_arena(
                tp_named, layout_tp, arena, rebind=list(tp_named.items())
            )
            if device == "cuda":
                torch.cuda.empty_cache()
            from sglang.srt.model_executor.weights_arena import arena_image

            image_tp = arena_image(arena, layout_tp, pin=True)

            # 5. TP pools + backends + decode graphs, with the TP arena
            # bytes live (graphs bake the fixed arena addresses; pin 2:
            # ONLY this stack captures decode graphs).
            tp_worker.alloc_memory_pool()

            # 5a. Share the REQUEST MAPPINGS by tensor rebind: both stacks
            # must read the same request->token rows and request->mamba
            # slot mapping (the scheduler writes them ONCE, into the
            # primary pool's tensors; slot ids are the cross-layout keys).
            # The pools themselves stay layout-specific; only the mapping
            # tensors alias. The slot SPACES agree by construction (same
            # max_num_reqs / max_mamba_cache_size in the args copy) --
            # asserted here, loudly.
            pp_req_pool = primary_runner.req_to_token_pool
            tp_req_pool = tp_worker.model_runner.req_to_token_pool
            if tp_req_pool.req_to_token.shape != pp_req_pool.req_to_token.shape:
                raise PhaseFlipBootError(
                    f"req_to_token shapes diverge between stacks: PP "
                    f"{tuple(pp_req_pool.req_to_token.shape)} vs TP "
                    f"{tuple(tp_req_pool.req_to_token.shape)}"
                )
            tp_req_pool.req_to_token = pp_req_pool.req_to_token
            if hasattr(pp_req_pool, "req_index_to_mamba_index_mapping"):
                pp_map = pp_req_pool.req_index_to_mamba_index_mapping
                tp_map = tp_req_pool.req_index_to_mamba_index_mapping
                if tp_map.shape != pp_map.shape:
                    raise PhaseFlipBootError(
                        f"mamba index mapping shapes diverge: PP "
                        f"{tuple(pp_map.shape)} vs TP {tuple(tp_map.shape)}"
                    )
                tp_req_pool.req_index_to_mamba_index_mapping = pp_map
                if pp_req_pool.mamba_pool.size != tp_req_pool.mamba_pool.size:
                    raise PhaseFlipBootError(
                        f"mamba slot spaces diverge: PP "
                        f"{pp_req_pool.mamba_pool.size} vs TP "
                        f"{tp_req_pool.mamba_pool.size} -- the flip relies "
                        f"on slot-id identity across layouts"
                    )

            tp_worker.init_attention_backends()
            tp_worker.init_cuda_graphs()
    finally:
        ctx.set_server_args(saved_args)

    # 6. Pin 3 on the real pools.
    assert_row_schema_compatible(
        primary_runner.token_to_kv_pool,
        tp_worker.model_runner.token_to_kv_pool,
    )

    # 7. PP is the boot phase: rebind its params to arena views and refill
    # the arena with the PP image (one contiguous H2D).
    bind_arena_views(layout_pp, arena, rebind=list(pp_named.items()))
    arena_refill(arena, layout_pp, image_pp)

    logger.info(
        "%s TP stack built: vector %s, arena %.2f MiB (pp %.2f / tp %.2f), "
        "images pinned %.2f MiB host",
        LOG_PREFIX,
        vec,
        arena.numel() / 1048576.0,
        layout_pp.total_bytes / 1048576.0,
        layout_tp.total_bytes / 1048576.0,
        (image_pp.numel() + image_tp.numel()) / 1048576.0,
    )
    return PhaseFlipStacks(
        tp_worker=tp_worker,
        arena=arena,
        layout_pp=layout_pp,
        layout_tp=layout_tp,
        image_pp=image_pp,
        image_tp=image_tp,
        vector=tuple(vec),
    )
