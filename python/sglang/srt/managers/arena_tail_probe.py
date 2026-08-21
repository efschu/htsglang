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


def build_meta_tp_model(model_config, load_config, server_args, vector, world_rank):
    """A TP-shaped model on the meta device, for layout only.

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
