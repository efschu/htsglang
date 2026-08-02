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
"""The ``--kv-pressure-ladder auto`` table source (#421 F1).

``model_executor/kv_pressure_ladder.build_ladder_from_server_args`` takes the
auto table through an injected ``table_fn`` so that module never imports the
planner. Nothing injected one, so ``auto`` -- advertised in the flag's help
text -- was a guaranteed ``ValueError`` at scheduler construction. This module
is the injection: it turns the running server's own configuration into the
:class:`~sglang.srt.planner.kv_ladder_table.RigModelProfile` the #272 planner
consumes, and hands ``build_ladder_table`` to the ladder builder.

Three properties are load-bearing and are the reason the bridge lives HERE
rather than as a one-line import at the call site.

**Rank-uniformity.** The ladder's rung index is min-reduced across the TP
group every ``--kv-pressure-consensus-interval`` rounds. Two ranks whose
tables differ would be agreeing on an integer that means different things, so
every input this module reads must be identical on every rank *by
construction*: ``server_args`` (the launcher hands the same object to every
scheduler), the launcher-published rank -> card-UUID vector
(:mod:`sglang.srt.registry.rank_cards`, an environment read), and NVML
physical facts of the node. Nothing rank-local, nothing measured per process,
and no collective -- the profile is built inside the scheduler constructor,
which is exactly the rank-local-before-group window where a collective hangs
the group with no diagnosis.

**Card identity by UUID, never by ordinal.** ``CUDA_VISIBLE_DEVICES``
narrowing in each child makes the CUDA ordinal a process-local name (#392,
#397). The rank -> card vector is UUIDs for that reason, and the profile's
card indices are NVML indices resolved from those UUIDs. When the vector is
absent the mapping is only recoverable on a rig whose cards are
indistinguishable; on a heterogeneous rig without it this refuses and names
the two remedies instead of guessing an enumeration order.

**Only rungs whose actuator exists.** ``KvPressureRuntime.__init__`` refuses a
ladder naming ``admission_cap`` without an armed limiter, or
``session_offload`` without the session manager -- and ``server_args``'
argument-time dependency checks apply to an explicit tuple spec only, so
``auto`` has no equivalent gate upstream of it. The profile therefore
inventories a relief feature only when this configuration actually wires its
actuator; the auto table can then never name a rung the runtime would reject.

Not supplied, deliberately: ``kv_bytes_per_token`` and ``weight_bytes_total``.
Both are per-rank runtime facts under uneven DCP / uneven TP, so deriving a
model-wide figure from one rank's pool would be the one input capable of
making two ranks' tables disagree. Their absence is what the planner's
``placeholder`` provenance is for -- every rung then says in its own ``source``
field that its capacity is unmeasured, which is the honest state, and the
measured figures are a follow-up that has to come off the ms/round chain
rather than out of an estimate here.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "auto_ladder_table_fn",
    "build_auto_ladder_profile",
    "wired_relief_features",
]

#: Geometry key of the rung the server boots in. A single-geometry profile is
#: the honest state at boot: the OTHER geometries of the nesting family are
#: the #297 reshard targets, and those are declared by
#: ``--kv-reshard-vectors`` as KV vectors, not as weight geometries.
BOOT_GEOMETRY_KEY = "boot"


def wired_relief_features(server_args) -> Tuple[str, ...]:
    """The relief rungs whose actuator this configuration actually wires.

    Mirrors the runtime's own inventory (``kv_pressure_runtime.WIRED_RELIEFS``
    plus the #297 reshard arm) at the level of the flags that produce those
    actuators, because the profile is built before the scheduler exists.

    ``kv_spill`` and ``weightless_rank`` are absent on purpose: the runtime
    documents both as planned-only, so an auto table naming them would
    inventory a rung that moves nothing.
    """
    reliefs: List[str] = []
    # Cheapest first is the planner's job (CHEAPNESS_ORDER); this only says
    # which features exist at all, so the order here is immaterial.
    if getattr(server_args, "kv_reshard_vectors", None) is not None:
        # #297 phase-boundary KV-vector flip. The runtime additionally checks
        # that the declared ceiling set covers the rung's operating grid and
        # downgrades the rung to planned-only when it does not; that check
        # needs the built table and stays where it is.
        reliefs.append("dcp_ratio")
    if getattr(server_args, "max_running_requests_ceiling", None) is not None:
        reliefs.append("admission_cap")
    if getattr(server_args, "enable_kv_session_offload", False):
        reliefs.append("session_offload")
    return tuple(reliefs)


def _rank_gpu_ids(server_args) -> Tuple[int, ...]:
    """The CUDA device index of every TP rank, from the launcher's own
    formula (``--rank-gpu-id`` wins, else base_gpu_id/gpu_id_step)."""
    tp_size = int(getattr(server_args, "tp_size", 1) or 1)
    return tuple(
        int(server_args.gpu_id_for_rank(0, tp_rank, 1, tp_size))
        for tp_rank in range(tp_size)
    )


def _budget_mib_for_rank(server_args, tp_rank: int) -> Optional[int]:
    """``--rank-gpu-memory-mib`` for one rank: scalar applies to every rank,
    a list is indexed per rank. Unset -> None (the card's total times the
    profile's budget fraction is used instead)."""
    budgets = getattr(server_args, "rank_gpu_memory_mib", None)
    if budgets is None:
        return None
    if isinstance(budgets, (list, tuple)):
        if tp_rank >= len(budgets):
            return None
        return int(budgets[tp_rank])
    return int(budgets)


def _cards_from_uuid_vector(server_args, uuids, gpu_ids):
    """Cards keyed by NVML index, resolved from the launcher's UUID vector."""
    from sglang.srt.planner.kv_ladder_table import CardSpec
    from sglang.srt.registry import nvml

    per_rank_index: List[int] = []
    by_index = {}
    for tp_rank, uuid in enumerate(uuids):
        info = nvml.device_by_uuid(uuid)
        per_rank_index.append(int(info.index))
        budget = _budget_mib_for_rank(server_args, tp_rank)
        existing = by_index.get(int(info.index))
        if existing is not None:
            # Co-located ranks: the budget is PER RANK (the solver charges it
            # once per rank on the card), so two different budgets on one card
            # cannot both be expressed. Refuse rather than silently keep one.
            if existing.budget_mib != budget:
                raise ValueError(
                    "--kv-pressure-ladder auto: ranks co-located on card "
                    f"{info.index} ({info.name}) declare different "
                    f"--rank-gpu-memory-mib budgets ({existing.budget_mib} vs "
                    f"{budget}); the auto table's capacity model carries one "
                    "budget per card. Give the co-located ranks the same "
                    "budget, or use an explicit --kv-pressure-ladder spec."
                )
            continue
        by_index[int(info.index)] = CardSpec(
            index=int(info.index),
            name=str(info.name),
            total_mib=int(info.total_mib),
            budget_mib=budget,
        )
    del gpu_ids  # the UUID vector is authoritative; ordinals are process-local
    return tuple(by_index[i] for i in sorted(by_index)), tuple(per_rank_index)


def _cards_from_homogeneous_node(server_args, gpu_ids):
    """Fallback when the launcher published no rank -> card vector.

    Legal only on a node whose cards are indistinguishable (same model, same
    total): then any ordinal -> card permutation yields the same profile, so
    using the CUDA ordinal as the card index cannot encode a wrong rig. On a
    mixed node it cannot, and refusing is the only honest answer.
    """
    from sglang.srt.planner.kv_ladder_table import CardSpec
    from sglang.srt.registry import nvml

    devices = nvml.list_devices()
    if not devices:
        raise ValueError(
            "--kv-pressure-ladder auto needs the rig profile, and NVML "
            "reported no devices. Use an explicit --kv-pressure-ladder step "
            "spec on a rig where NVML cannot answer."
        )
    names = {(d.name, int(d.total_mib)) for d in devices}
    if len(names) > 1:
        raise ValueError(
            "--kv-pressure-ladder auto cannot map ranks to cards on this "
            f"rig: the node holds different card models ({sorted(names)}) "
            "and no rank -> card vector was published, so a CUDA ordinal "
            "does not identify a card (CUDA_VISIBLE_DEVICES narrowing, "
            "#392/#397). Remedies: pass --rank-gpu-id (the launcher then "
            "publishes the vector), set SGLANG_RANK_CARD_PROBE_CUDA=1, or "
            "give an explicit --kv-pressure-ladder step spec."
        )
    name, total_mib = next(iter(names))
    cards = {}
    for tp_rank, gpu in enumerate(gpu_ids):
        budget = _budget_mib_for_rank(server_args, tp_rank)
        existing = cards.get(int(gpu))
        if existing is not None:
            if existing.budget_mib != budget:
                raise ValueError(
                    "--kv-pressure-ladder auto: ranks co-located on device "
                    f"{gpu} declare different --rank-gpu-memory-mib budgets "
                    f"({existing.budget_mib} vs {budget}); the auto table's "
                    "capacity model carries one budget per card."
                )
            continue
        cards[int(gpu)] = CardSpec(
            index=int(gpu),
            name=str(name),
            total_mib=int(total_mib),
            budget_mib=budget,
        )
    return tuple(cards[i] for i in sorted(cards)), tuple(int(g) for g in gpu_ids)


def build_auto_ladder_profile(server_args):
    """The rig/model profile the ``auto`` table is computed from.

    Rank-uniform by construction -- see the module docstring for why that is
    the binding constraint and which inputs are therefore admissible.
    """
    from sglang.srt.planner.kv_ladder_table import GeometryRungSpec, RigModelProfile
    from sglang.srt.registry.rank_cards import rank_card_uuids

    tp_size = int(getattr(server_args, "tp_size", 1) or 1)
    gpu_ids = _rank_gpu_ids(server_args)

    uuids = rank_card_uuids(tp_size)
    if uuids:
        cards, per_rank_card = _cards_from_uuid_vector(server_args, uuids, gpu_ids)
    else:
        cards, per_rank_card = _cards_from_homogeneous_node(server_args, gpu_ids)

    ratio = getattr(server_args, "rank_tp_ratio", None)
    if isinstance(ratio, (list, tuple)) and len(ratio) == tp_size:
        rank_ratio = tuple(int(u) for u in ratio)
    else:
        # Even TP (or --rank-tp-ratio auto not yet resolved to a vector): every
        # rank carries the same weight share, which is what (1, ..., 1) means.
        rank_ratio = tuple([1] * tp_size)

    geometry = GeometryRungSpec(
        key=BOOT_GEOMETRY_KEY,
        ratio=rank_ratio,
        gpus=per_rank_card,
        graphs_precaptured=True,
    )

    fraction = getattr(server_args, "mem_fraction_static", None)
    kwargs = {}
    if isinstance(fraction, float) and 0.0 < fraction <= 1.0:
        # Only used for cards WITHOUT an explicit --rank-gpu-memory-mib
        # budget; with one, the operator's MiB figure is the whole budget.
        kwargs["budget_fraction"] = float(fraction)

    return RigModelProfile(
        cards=cards,
        geometries=(geometry,),
        reliefs=wired_relief_features(server_args),
        **kwargs,
    )


def auto_ladder_table_fn(server_args) -> Callable[[], object]:
    """The ``table_fn`` for ``build_ladder_from_server_args``.

    Returns a THUNK: the profile is built, and NVML is touched, only if the
    spec actually is ``auto``. On every other spec -- including the default
    unset one -- calling this costs one closure allocation and the default
    path stays byte-identical.
    """

    def _table():
        from sglang.srt.model_executor.kv_pressure_ladder import (
            DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
        )
        from sglang.srt.planner.kv_ladder_table import build_ladder_table

        profile = build_auto_ladder_profile(server_args)
        table = build_ladder_table(
            profile,
            external_min_hysteresis_rounds=int(
                getattr(
                    server_args,
                    "kv_pressure_external_hysteresis_rounds",
                    DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
                )
            ),
        )
        logger.info(
            "[kv-pressure] --kv-pressure-ladder auto: table computed from the "
            "rig profile -- %d cards %s, geometry ratio %s, wired reliefs %s, "
            "%d rungs.",
            len(profile.cards),
            [f"{c.index}:{c.name}" for c in profile.cards],
            list(profile.geometries[0].ratio),
            list(profile.reliefs),
            len(table.steps),
        )
        return table

    return _table
