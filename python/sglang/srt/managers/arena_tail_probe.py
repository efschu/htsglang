# SPDX-License-Identifier: Apache-2.0
"""#785: the flip seam's arena tail, derived at KV-sizing time.

THE ORDERING THIS OVERTURNS. A phase-flip boot holds ONE weights arena sized
``max(layout_pp.total_bytes, layout_tp.total_bytes)`` and the difference -- the
arena TAIL -- must stay off-limits to the KV pool, because the driver hands it
back at the next flip commit inside the no-return region. So the tail is a
first-class post in the pool solve. But ``layout_tp`` is built in
``phase_flip_boot.build_phase_flip_tp_stack`` from a real TP-shaped worker,
which the scheduler constructs AFTER the pool has been sized
(``scheduler.py:1421`` init_tp_model_worker -> ``:1425`` init_memory_pools ->
``:1442`` build_phase_flip_tp_stack). ``phase_flip_seam_reserve`` states the
consequence verbatim: "neither exists until the TP stack is built -- which
happens AFTER the PP pool is sized. There is no ordering that makes them
knowable in time, so they are not computed here at all."

That sentence is what this module retires, and the cost of leaving it standing
is not theoretical. The tail has been carried between boots in an on-disk seam
record instead (``read_seam_reserve``), which defaults to 0 on a COLD boot. A
first boot therefore prices rank 2's arming floor at 1523 MiB instead of 3226,
sizes the pool against the missing 1703 MiB, and dies when the NEXTN draft
weights land -- the two-boot cliff behind the #678 OOM class. Every later boot
inherits a number measured by its predecessor, which is also why a
configuration change silently sizes against the previous configuration.

WHY IT CAN BE DERIVED WITHOUT BUILDING THE STACK. A layout needs ``shape``,
``stride``, ``dtype``, ``storage_offset`` and the storage SIZE. All five are
correct on a META tensor, so a meta-device parameter set answers
``total_bytes`` exactly -- same alignment, same ordering, same code path, no
allocation. Measured against the boot that motivated this
(boot_735_default791b, vector 32,16,16): the meta derivation reproduces the
logged ``layout_tp`` on all three ranks byte for byte -- 15925.8 / 8573.78 /
8573.78 MiB -- in about 0.1 s per rank.

THE ONE INPUT THAT IS NOT CORRECT ON META is ``data_ptr()``, which is 0 for
every meta tensor. That is precisely what ``plan_arena_layout`` uses to detect
aliasing, so fed meta tensors it would fold the whole model into one slot and
return a total that is wrong by orders of magnitude while looking perfectly
well-formed -- straight into a memory budget. ``plan_arena_layout`` therefore
refuses meta input unless the alias relation is supplied, and supplying it is
this module's job (``storage_alias_relation``).
"""

from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)

LOG_PREFIX = "ARENA-TAIL-PROBE"

#: The bar the derivation is graded at, in MiB, against a ~16 GiB layout.
#: A layout total is subtracted from a memory budget, so the tolerance is set
#: by what a difference COSTS, not by what is easy to hit: 1 MiB of rank 2's
#: bracket is ~100 tokens of pool at this rig's cell size.
VERDICT_BAR_MIB = 1.0


class ArenaTailProbeError(RuntimeError):
    """The tail could not be derived. Never downgraded to a guess."""


def storage_alias_relation(named: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """``{alias_name: canonical_name}`` by STORAGE identity, meta included.

    ``data_ptr()`` is 0 on meta and cannot separate "these two tensors share a
    storage" from "neither tensor has one". The StorageImpl handle behind
    ``untyped_storage()._cdata`` distinguishes them on both devices: it is
    stable across repeated calls on one tensor, equal for two tensors that
    share a storage, and different for two independent ones -- on meta exactly
    as on a real device, where it also agrees with ``data_ptr()`` grouping.

    Canonical is the first name in sorted order, which is the same slot owner
    ``plan_arena_layout`` picks when it infers the relation itself. Pointing an
    alias anywhere else would leave the owning tensor unslotted and undersize
    the arena, and ``plan_arena_layout`` refuses a chained relation for that
    reason.
    """
    by_storage: Dict[int, list] = {}
    for name, t in named.items():
        by_storage.setdefault(t.untyped_storage()._cdata, []).append(name)
    relation: Dict[str, str] = {}
    for names in by_storage.values():
        if len(names) < 2:
            continue
        names = sorted(names)
        for alias in names[1:]:
            relation[alias] = names[0]
    return relation


def plan_meta_layout(named: Dict[str, torch.Tensor]):
    """``plan_arena_layout`` for a set that may live on the meta device."""
    from sglang.srt.model_executor.weights_arena import plan_arena_layout

    return plan_arena_layout(named, alias_of=storage_alias_relation(named))


def arena_tail_bytes(layout_pp_bytes: int, layout_tp_bytes: int) -> int:
    """The tail: what rung 3 must re-commit when the process re-enters PP.

    ``refill_high_water_bytes`` is ``max(pp, tp)`` and the active prefix in the
    TP phase is ``tp``, so the tail is the difference. Clamped at zero because
    a rank whose PP layout is the SMALLER one has nothing to commit on this
    leg -- treating "PP is always larger" as structural was tried on
    2026-08-11 and produced cudaErrorInvalidValue on all three ranks.
    """
    return max(0, int(layout_pp_bytes) - int(layout_tp_bytes))


def flip_draft_model_config(tp_args):
    """The ModelConfig the flip's draft worker is built from.

    Mirrors ``TpModelWorker._init_model_config`` for the draft case: the draft
    path, or the target path when the speculative method keeps its extra
    layers in the target checkpoint (NEXTN/MTP does).
    """
    from sglang.srt.configs.model_config import ModelConfig

    path = getattr(tp_args, "speculative_draft_model_path", None) or tp_args.model_path
    return ModelConfig.from_server_args(tp_args, model_path=path, is_draft_model=True)


def build_meta_tp_model(model_config, load_config, server_args, vector, world_rank):
    """A TP-shaped model on the meta device, for layout only.

    ``model_config=None`` means the flip's DRAFT model, whose config can only
    be built once the TP server args are published (see inside).

    Built under the SAME two scopes the real TP stack is built under, because
    a layout derived under a different geometry is a different layout:

    * ``phase_flip_tp_scope(world_rank, n)`` rotates the parallel context from
      (tp=1, pp=N) to (tp=N, dcp=N, pp=1) and routes the flip groups. It is
      legal here: the flip groups are created in ``init_tp_model_worker``
      (model_runner.py:2001), which runs before ``init_memory_pools``.
    * ``scoped_tp_partition_ratios(vector)`` installs the uneven shard plan as
      a context-local overlay rather than the process global the TP stack
      build uses, so the scheduler thread is left unchanged.

    The published server args are the same ``derive_tp_stack_server_args`` copy
    the real build publishes -- module construction reads it (the vision tower
    reads ``mm_enable_dp_encoder`` during ``__init__``) -- minus the pool
    sizing, which is what this boot is still solving for and which no
    parameter shape depends on.

    NOT PURE META, AND THAT IS THE ARCHITECTURE'S CHOICE, NOT THIS PROBE'S.
    ``Qwen3_5GatedDeltaNet`` builds its gated RMSNorm with an explicit
    ``device=torch.get_device_module().current_device()`` (qwen3_5.py:403),
    which overrides the ambient meta context for one small norm weight per
    linear-attention layer. That is kilobytes, it is freed with the throwaway
    model, and the layout is unaffected because a layout reads sizes rather
    than addresses -- but it is the reason this function drops the model and
    empties the cache before returning rather than relying on scope exit.
    """
    from sglang.srt.distributed.utils import scoped_tp_partition_ratios, tp_plan_active
    from sglang.srt.managers.phase_flip_boot import (
        derive_tp_stack_server_args,
        phase_flip_tp_scope,
    )
    from sglang.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
        set_default_torch_dtype,
    )
    from sglang.srt.runtime_context import get_context, get_server_args

    n = len(vector)
    ctx = get_context()
    saved_args = get_server_args()
    tp_args = derive_tp_stack_server_args(server_args)
    ctx.set_server_args(tp_args)
    try:
        with phase_flip_tp_scope(world_rank, n):
            with scoped_tp_partition_ratios(list(vector)):
                # STRICT ON THIS PATH ONLY. An installed vector whose length
                # does not match the group size does not raise -- it falls back
                # to the even split and returns a plausible layout for the
                # wrong geometry. That fallback is load-bearing across the
                # tree (a blanket refusal fails 41 distributed tests), so it is
                # not touched; it is simply not tolerated HERE, where the
                # answer is charged into a memory budget.
                if not tp_plan_active(n):
                    raise ArenaTailProbeError(
                        f"the {n}-entry flip vector {list(vector)} is not "
                        f"active for a group of size {n}; the layout would be "
                        f"an EVEN split of the wrong geometry, and it would "
                        f"look entirely plausible"
                    )
                if model_config is None:
                    # The DRAFT config has to be built with tp_args PUBLISHED,
                    # because that is the geometry the real draft worker is
                    # constructed under (build_flip_draft_worker).
                    model_config = flip_draft_model_config(tp_args)
                quant_config = _get_quantization_config(model_config, load_config)
                with set_default_torch_dtype(model_config.dtype):
                    with torch.device("meta"):
                        return _initialize_model(
                            model_config, load_config, quant_config
                        )
    finally:
        ctx.set_server_args(saved_args)


def derive_layout_tp_bytes(
    model_config, load_config, server_args, vector, world_rank
) -> int:
    """``layout_tp.total_bytes`` for THIS rank, without allocating weights."""
    from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict

    model = build_meta_tp_model(
        model_config, load_config, server_args, vector, world_rank
    )
    try:
        named = checkpoint_param_dict(model)
        if not named:
            raise ArenaTailProbeError(
                "the meta TP model exposed no checkpoint parameters; an empty "
                "layout would price the tail at the full PP layout"
            )
        return int(plan_meta_layout(named).total_bytes)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def derive_flip_draft_bytes(load_config, server_args, vector, world_rank) -> int:
    """This rank's FLIP DRAFT weight footprint, derived rather than met.

    THE SECOND HALF OF THE SAME ORDERING DEFECT, and the one that actually
    kills boots. ``build_flip_draft_worker`` (phase_flip_boot.py:614) loads a
    whole second model -- the NEXTN/MTP draft -- and it runs at
    scheduler.py:1442, long after the KV pool was sized at :1425. So the pool
    is solved against a budget that does not know a 1.4 GiB model is still
    coming.

    That is survivable only while the pool is small enough to leave accidental
    slack, which is exactly why it went unnoticed: on the shipped cut the
    binding rank held 7027 MiB free and the draft fit by luck. Raise the pool
    and the luck runs out -- measured on boot 735-full785 at commit
    6c18085383, which solved 764512 tokens and then died in
    ``ct_embedding.create_weights`` trying to allocate 406 MiB with 227.75 MiB
    free on rank 0. Same shape as the #678 OOM, same cause: a post that the
    solve cannot see because it is created later.

    Derived here at 0.01 s and no allocated weights. On this rig:
    1441.14 / 1352.35 / 1352.35 MiB.

    THIS IS ALSO THE PLANNER'S MISSING INPUT. ``planner/pp_cut.py:412-432``
    makes ``RankResources.draft_residency_mib`` MANDATORY and refuses to solve
    a cut without it; until now nothing could supply it before a boot.
    """
    from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict

    model = build_meta_tp_model(None, load_config, server_args, vector, world_rank)
    try:
        named = checkpoint_param_dict(model)
        if not named:
            raise ArenaTailProbeError(
                "the meta flip-draft model exposed no checkpoint parameters; "
                "charging 0 would reproduce the OOM this term exists to stop"
            )
        return int(plan_meta_layout(named).total_bytes)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


#: MiB of post-sizing stack demand that is NOT the arena delta and NOT the
#: draft weights: the draft's flashinfer workspaces, the decode CUDA graphs and
#: the stack's own bookkeeping. Measured, per rank, on the one boot that
#: SURVIVED the whole build -- 735-tail785 -- as the net VRAM consumed between
#: "Memory pool end" and the end of build_phase_flip_tp_stack, minus the two
#: terms above:
#:
#:     rank 0   10.60 -> 6.95 GB = 3.65 net, minus 0 arena delta, minus 1.407
#:              draft                                   -> 2294 MiB
#:     rank 1    8.94 -> 5.91 GB = 3.03 net, minus 0.551 arena delta, minus
#:              1.320 draft                             -> 1188 MiB
#:     rank 2    5.92 -> 3.99 GB = 1.93 net, minus 0 arena delta, minus 1.320
#:              draft                                   ->  625 MiB
#:
#: The TP stack's own KV pools cost nothing here and that is checked rather
#: than assumed: the boot logs "Memory pool end" nine times, and the second and
#: third readings on each rank are identical to the first (PP0 7.69 / 7.69), so
#: those pools REBIND the existing allocation instead of allocating a second
#: one. Charging them would have been charging the pool against itself.
STACK_RESIDUAL_MIB = (2294, 1188, 625)


def post_sizing_stack_bytes(
    rank: int,
    layout_pp_bytes: int,
    layout_tp_bytes: int,
    draft_bytes: int,
    cold_stack_deferred: bool = False,
) -> int:
    """ONE post for everything ``build_phase_flip_tp_stack`` creates.

    WHY THIS IS ONE TERM AND NOT THREE. Everything that build creates is
    allocated at scheduler.py:1442, after the pool is solved at :1425 -- the
    arena, the flip draft's weights, its attention-backend workspaces, its
    CUDA graphs. Each was invisible to the solve for the same reason, and each
    only became fatal once the pool grew enough to remove the accidental slack
    that had been paying for it. Charging them one at a time produced two
    boots and two different OOMs at two different lines (735-full785 in
    ct_embedding.create_weights, 735-draft785 in the flashinfer prefill
    backend), which is the evidence that the shape was wrong rather than the
    numbers.

    The first two components are DERIVED exactly for this boot, so they track
    the cut and the vector: the arena exceeds the PP weights by
    ``max(0, layout_tp - layout_pp)`` (the PP originals are snapshotted to
    host during the build, so what stays is the arena), and the draft weights
    come from the same meta probe. Only the remainder is a measured constant,
    and it is measured on a boot that lived through the build rather than
    inferred from one that died in it.

    ``cold_stack_deferred`` DROPS THE RESIDUAL, and it is not free money. The
    residual is the flip draft's attention-backend workspaces and its decode
    CUDA graphs -- the two posts that are PHASE-COLD, since the PP phase that
    sizes this pool never executes a draft forward. Deferring them until the
    first pp->tp cutover is the literal reading of spill-before-alloc: the
    memory is not allocated during the phase that cannot use it.

    THE PRICE IS REAL AND IT MOVES, IT DOES NOT VANISH. The bytes must exist
    again at the cutover, and by then the pool has grown into them. That is
    payable only because the KV pool is a relief provider at the seam
    (``kv_backing_relief`` / ``recover_kv_backing``, #656 spec item 12): the
    rung returns backing on the pp->tp leg, which is the same leg that has to
    fund the restore. So the credit here is valid ONLY when the restore is
    charged to the flip budget, never as a standalone discount.

    AND IT OVERTURNS A RECORDED DECISION, DELIBERATELY. ``phase_flip_spill``
    refused rung 4 with the reasoning that it "buys a phase-local spill of
    something the next TP phase must re-capture, which is a different and much
    worse trade than rung 2". That reasoning was about a RUNTIME, per-flip
    spill, where every flip pays a recapture. This is a BOOT-time deferral
    paid once, and the arithmetic that changed the trade is measured: rank 0
    binds the pool at 525462 tokens and needs 2242 MiB more to fund the 669k
    reference, against a residual on that same rank of 2294 MiB. The refusal
    is left standing in ``phase_flip_spill`` until the park/restore path is
    wired and exercised in a real flip cycle; this function only makes the
    sizer able to express the credit.
    """
    arena_over_weights = max(0, int(layout_tp_bytes) - int(layout_pp_bytes))
    if cold_stack_deferred:
        residual = 0
    else:
        residual = (
            STACK_RESIDUAL_MIB[rank] if 0 <= rank < len(STACK_RESIDUAL_MIB) else 0
        )
    return arena_over_weights + int(draft_bytes) + int(residual) * 1048576


def derive_arena_tail_bytes(
    model_config,
    load_config,
    server_args,
    vector,
    world_rank,
    layout_pp_bytes: int,
) -> int:
    """This rank's arena tail, derived from its own boot rather than a record.

    ``layout_pp_bytes`` is asked for rather than derived: the PP model is
    already loaded when the pool is sized, so its layout is a measurement of
    this boot and outranks a second derivation of the same quantity.
    """
    layout_tp = derive_layout_tp_bytes(
        model_config, load_config, server_args, vector, world_rank
    )
    return arena_tail_bytes(layout_pp_bytes, layout_tp)


def log_derivation(
    rank: int, layout_pp_bytes: int, layout_tp_bytes: int, elapsed_s: float
) -> None:
    """The boot instrument. Its numbers are checked against the SAME boot.

    ``phase_flip_boot`` prints both layout totals when it builds the TP stack,
    later in the same boot. Emitting the derived pair in a parseable form here
    makes the check an identity on one boot's own numbers rather than a
    comparison against a remembered rig.
    """
    mib = 1048576.0
    logger.info(
        "%s (rank %d): DERIVED layout_pp %.2f MiB, layout_tp %.2f MiB, arena "
        "tail %.2f MiB in %.2f s, from a meta-device TP model -- no weights "
        "allocated and no previous boot consulted. The TP stack build later in "
        "this same boot prints both totals again; the two must agree.",
        LOG_PREFIX,
        rank,
        layout_pp_bytes / mib,
        layout_tp_bytes / mib,
        arena_tail_bytes(layout_pp_bytes, layout_tp_bytes) / mib,
        elapsed_s,
    )


def grade_derivation(
    rank: int,
    derived_pp_bytes: int,
    derived_tp_bytes: int,
    measured_pp_bytes: int,
    measured_tp_bytes: int,
) -> bool:
    """THE DECIDING GATE: the derivation against the SAME boot's own totals.

    A desk run has to pin one GPU capability, and the compressed-tensors
    linear scheme is chosen from ``torch.cuda.get_device_capability()`` at
    construction time -- so on a mixed rig a desk agreement is evidence and
    not proof. This grades the derivation where the question actually lives:
    inside the boot, on the numbers that boot measured minutes later, on the
    card the rank is really running.

    Returns the verdict rather than raising it. An instrument that can abort a
    boot stops being an instrument -- and this one is not yet load-bearing:
    nothing consumes the derived number until the sizing path is wired to it,
    which is the change this gate exists to authorize.
    """
    mib = 1048576.0
    d_pp = derived_pp_bytes / mib
    d_tp = derived_tp_bytes / mib
    m_pp = measured_pp_bytes / mib
    m_tp = measured_tp_bytes / mib
    err_pp = abs(d_pp - m_pp)
    err_tp = abs(d_tp - m_tp)
    passed = err_pp <= VERDICT_BAR_MIB and err_tp <= VERDICT_BAR_MIB
    (logger.info if passed else logger.error)(
        "%s (rank %d): VERDICT %s at a %.1f MiB bar. layout_pp derived %.2f "
        "vs measured %.2f MiB (error %.2f); layout_tp derived %.2f vs measured "
        "%.2f MiB (error %.2f). Arena tail derived %.2f vs measured %.2f MiB. "
        "Both measurements are from THIS boot, so this is an identity check "
        "rather than a comparison against a remembered rig.",
        LOG_PREFIX,
        rank,
        "PASS" if passed else "FAIL",
        VERDICT_BAR_MIB,
        d_pp,
        m_pp,
        err_pp,
        d_tp,
        m_tp,
        err_tp,
        arena_tail_bytes(derived_pp_bytes, derived_tp_bytes) / mib,
        arena_tail_bytes(measured_pp_bytes, measured_tp_bytes) / mib,
    )
    return passed
