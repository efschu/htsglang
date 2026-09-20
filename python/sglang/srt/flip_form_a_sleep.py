# SPDX-License-Identifier: Apache-2.0
"""Form A's missing sleep/wake hook -- which tag every D-side allocation wears.

WHY THIS IS A PRECONDITION AND NOT A REFINEMENT. The two layouts of the
Next-Flash flip cannot be resident on ANY card at once. Measured (design
§5.1): PP3 holds 22303 / 13076 / 10957 MiB while Form A reserves 29.6 GiB on
the 5090 alone under decode, and under prefill-extend Form A leaves
``card free 0.24 GiB`` on the 5090 and ``0.08 GiB`` on 3080-a. So one side
must SLEEP. The mechanism exists for the 27B flip -- three device tags, a
deposit-before-pause and credit-before-resume order
(``weight_updater.py:1248-1317``, ``:6498-6503``) -- but Form A does not use
it: a grep for ``pause`` / ``resume`` / ``memory_saver`` over all six
``form_a_*.py`` plus ``rank_role.py`` returns ZERO. That is risk R7 and the
largest hole in the plan.

This module is the DESK half of closing it: which tag each Form-A allocation
wears, per role, and in which order the tags are paused and resumed. It
allocates nothing and touches no device -- it is the table the runtime hook
reads, and the refusal that fires when an allocation is not in it.

THE ROLE ASYMMETRY IS THE WHOLE POINT. Under Form A the host rank holds every
dense weight, the WHOLE KV and the solo draft; the workers hold expert weights
and nothing else of substance (fnFA19: ``Mamba Cache ... conv_state 0.01 GB,
ssm_state 0.11 GB`` on TP0 and ``0.00`` on TP1/TP2; ``Draft-solo KV planning:
rank 1/2 draft-KV cell term 64 -> 0 B/token``). So the host wears all three
tags and the workers wear two -- a worker with a ``kv_cache`` tag would be
pausing an empty region and grading its sleep against a denominator that is
not there.

AND THE EXPERT POOL WEARS NO DEVICE TAG AT ALL. ``torch_memory_saver`` manages
DEVICE memory only (``constants.py:2-4``: ``kv_cache``, ``weights``,
``cuda_graph``); there is no host tag. The worker experts live in the shared
page-locked HOST pool (slice 3) and a sleeping group keeps that pool in full
-- which is exactly why slice 3 exists and why tagging the pool here would be
a lie that makes the sleep look bigger than it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.flip_nextflash_plan import FlipInfeasible

__all__ = [
    "ROLE_HOST",
    "ROLE_WORKER",
    "Weg2FlipFormAUntagged",
    "FormAAllocation",
    "FORM_A_ALLOCATIONS",
    "tags_for_role",
    "plan_form_a_sleep",
    "sleep_tag_order",
    "wake_tag_order",
    "SleepHookPlan",
]

ROLE_HOST = "host"
ROLE_WORKER = "worker"

#: The host tag set: all three device tags. The draft tag is deliberately NOT
#: here -- ``GPU_MEMORY_TYPE_WEIGHTS_DRAFT`` is outside ``GPU_MEMORY_ALL_TYPES``
#: on purpose (constants.py:16-21: "Adding this tag there would pause the
#: drafter -- the exact opposite of the purpose"). Under Form A the drafter is
#: SOLO on the host and stays resident across the flip.
_HOST_TAGS = frozenset(
    (GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_CUDA_GRAPH)
)
_WORKER_TAGS = frozenset((GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_CUDA_GRAPH))


class Weg2FlipFormAUntagged(FlipInfeasible):
    """W118 -- a Form-A device allocation has no memory-saver tag.

    An untagged device allocation survives the sleep. On a rig whose tightest
    card has 0.08 GiB free under extend, "survives the sleep" is the same
    sentence as "the other layout OOMs on resume" -- and it fails as an
    allocation failure in the WAKING group, pointing at the wrong side.
    Hence a refusal at the desk, on the table, rather than a discovery on
    metal.

    Not the same as ``W4 Weg2WakeRefused`` (weg2_memory_saver.py:390) or
    ``W25 Weg2DormantRefused`` (:2787): those grade a sleep/wake that RAN.
    This one says the sleep was never going to be complete.
    """


@dataclass(frozen=True)
class FormAAllocation:
    """One named device allocation of the Form-A layout.

    ``roles`` is which ranks actually hold it; ``tag`` is the memory-saver tag
    it must wear; ``why`` is the log line or file:line the claim rests on, so
    a later reader can check the table rather than trust it.
    """

    name: str
    roles: FrozenSet[str]
    tag: str
    why: str


#: The Form-A device allocations, each with the evidence for its role set.
#: A new Form-A allocation that is not added here is refused by
#: :func:`plan_form_a_sleep` -- that refusal IS the maintenance mechanism.
FORM_A_ALLOCATIONS: Tuple[FormAAllocation, ...] = (
    FormAAllocation(
        name="host_kv_pool",
        roles=frozenset((ROLE_HOST,)),
        tag=GPU_MEMORY_TYPE_KV_CACHE,
        why=(
            "fnFA19:1000 KV pool sizing cell_size=14143 on rank 0 vs 768 on "
            "1/2; 'KV Cache is allocated ... K size: 1.50 GB' on the host only"
        ),
    ),
    FormAAllocation(
        name="mamba_gdn_state",
        roles=frozenset((ROLE_HOST,)),
        tag=GPU_MEMORY_TYPE_KV_CACHE,
        why=(
            "fnFA19:1029-1031 'Mamba Cache is allocated ... ssm_state 0.11 GB' "
            "on TP0, 0.00 on TP1/TP2. The mamba/GDN anchors live under the KV "
            "tag -- there is no separate mamba tag (memory_pool.py:1017)"
        ),
    ),
    FormAAllocation(
        name="dense_weights",
        roles=frozenset((ROLE_HOST,)),
        tag=GPU_MEMORY_TYPE_WEIGHTS,
        why="design §5.1: 13.48 GiB model tensors on the host, draft 2.75 GiB",
    ),
    FormAAllocation(
        name="expert_weights_resident",
        roles=frozenset((ROLE_WORKER,)),
        tag=GPU_MEMORY_TYPE_WEIGHTS,
        why=(
            "design §5.1: 13.71 of 13.89 GiB of a worker's model tensors are "
            "experts. The SPILLED part is host-side and wears no device tag"
        ),
    ),
    FormAAllocation(
        name="solo_draft_weights",
        roles=frozenset((ROLE_HOST,)),
        tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
        why=(
            "--speculative-draft-placement solo; worker_keeps_parameter "
            "(rank_role.py:590-628) vetoes every mtp. tensor on a worker. "
            "Resident across the flip: the draft tag is outside "
            "GPU_MEMORY_ALL_TYPES on purpose (constants.py:16-21)"
        ),
    ),
    FormAAllocation(
        name="role_graphs",
        roles=frozenset((ROLE_HOST, ROLE_WORKER)),
        tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
        why=(
            "launch_fnFA19.sh --cuda-graph-bs-decode 1 2 with GRAPHS=pool; "
            "both roles capture, the shapes differ per role"
        ),
    ),
)


def tags_for_role(role: str) -> FrozenSet[str]:
    """The tags a Form-A rank of this role pauses on sleep.

    The draft tag is excluded by construction: it is resident across the flip
    (constants.py:16-21). A worker gets no ``kv_cache`` tag -- pausing an
    empty region would grade its sleep against a denominator that is not
    there (``SleepAcceptanceCensus``, weg2_memory_saver.py:1495-1612).
    """
    if role == ROLE_HOST:
        return _HOST_TAGS
    if role == ROLE_WORKER:
        return _WORKER_TAGS
    raise Weg2FlipFormAUntagged(
        f"W118 Weg2FlipFormAUntagged -- {role!r} is not a Form-A role; "
        f"expected {ROLE_HOST!r} or {ROLE_WORKER!r} (rank_role.py:459-493)"
    )


def sleep_tag_order(role: str, weight_chunk_tags: Sequence[str] = ()) -> List[str]:
    """The order a Form-A rank PAUSES its tags in.

    Mirrors the 27B path exactly, because the hazards are the same ones:

      1. ``kv_cache`` FIRST, and ``flush_cache()`` before it
         (``weight_updater.py:6020-6021``, a MUST_FIX for mamba-pool
         correctness: the GDN anchors sit under this tag).
      2. the weights family, CHUNKS first and the base tag LAST
         (``weights_family_tags``, weg2_memory_saver.py:2324-2343 -- the base
         tag closes the sleep and ``derive_waves`` reads ``tags[-1]``).
      3. ``cuda_graph`` last: the graph tag's private pool can only be
         released once nothing it references is still being paused.

    The DEPOSIT of any exchanged weight bytes happens before step 2
    (``_weg2_xchg_deposit_before_sleep``, :1248-1317): "a deposit after the
    pause would read pages this rank no longer owns". That ordering is
    enforced by call-site position, not by a lock, which is why it is stated
    here rather than assumed.
    """
    tags = tags_for_role(role)
    order: List[str] = []
    if GPU_MEMORY_TYPE_KV_CACHE in tags:
        order.append(GPU_MEMORY_TYPE_KV_CACHE)
    if GPU_MEMORY_TYPE_WEIGHTS in tags:
        order.extend(list(weight_chunk_tags))
        order.append(GPU_MEMORY_TYPE_WEIGHTS)
    if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
        order.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
    return order


def wake_tag_order(role: str, weight_chunk_tags: Sequence[str] = ()) -> List[str]:
    """The order a Form-A rank RESUMES its tags in -- the mirror, not the
    reverse-by-accident.

    ``weight_updater.resume_memory_occupation`` (:6368) does graph FIRST
    (:6408), then weights per tag (:6503), then KV (:6893). The CREDIT wait
    (``_weg2_await_vram_credit``, :6498-6500) sits immediately before the
    weights resume: the other group's release must have landed before this
    group maps fresh physical pages over it. That is the Wand-8 ordering
    (Memory ``XCHG-LANE-ORDNUNG``: deposit-before-pause x credit-before-resume)
    and inverting it is the deadlock it is named after.
    """
    tags = tags_for_role(role)
    order: List[str] = []
    if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
        order.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
    if GPU_MEMORY_TYPE_WEIGHTS in tags:
        order.extend(list(weight_chunk_tags))
        order.append(GPU_MEMORY_TYPE_WEIGHTS)
    if GPU_MEMORY_TYPE_KV_CACHE in tags:
        order.append(GPU_MEMORY_TYPE_KV_CACHE)
    return order


@dataclass(frozen=True)
class SleepHookPlan:
    role: str
    tags: Tuple[str, ...]
    sleep_order: Tuple[str, ...]
    wake_order: Tuple[str, ...]
    tagged: Tuple[Tuple[str, str], ...]
    resident: Tuple[str, ...]
    #: Memory-saver pause/resume must run with the barlink abort-gate
    #: watchdog quiesced, for the same reason a CUDA-graph capture does
    #: (``parallel_state.py:3404``): the watchdog READS device memory from
    #: ANOTHER THREAD, and between pause and resume those pages are unmapped.
    #: ``barlink.graph_capture_running()`` cannot serve -- it answers for the
    #: calling thread's stream and the watchdog is not that thread
    #: (``barlink_abort_gate.py:391-400``).
    requires_pause_polling: bool = True

    def report(self) -> str:
        lines = [
            f"FORM-A SLEEP HOOK role={self.role}",
            f"  tags        {sorted(self.tags)}",
            f"  sleep order {list(self.sleep_order)}",
            f"  wake order  {list(self.wake_order)}",
            f"  resident    {list(self.resident)} (never paused)",
            f"  under barlink_abort_gate.pause_polling(): "
            f"{self.requires_pause_polling}",
        ]
        for name, tag in self.tagged:
            lines.append(f"    {name:<26} -> {tag}")
        return "\n".join(lines)


def plan_form_a_sleep(
    role: str,
    live_allocations: Iterable[str],
    weight_chunk_tags: Sequence[str] = (),
    table: Sequence[FormAAllocation] = FORM_A_ALLOCATIONS,
) -> SleepHookPlan:
    """Tag every live Form-A allocation of this role, or refuse by name.

    ``live_allocations`` is what the runner actually built -- NOT a hand list.
    Feeding it the runner's real pool inventory is what makes a newly added
    allocation fail here (W118) instead of silently surviving the sleep;
    the same discipline the design asks for in slice 5 for the state carry
    (risk R5, the QSA raw-key ring).
    """
    known = {a.name: a for a in table}
    live = [str(n) for n in live_allocations]
    role_tags = tags_for_role(role)

    unknown = sorted({n for n in live if n not in known})
    if unknown:
        raise Weg2FlipFormAUntagged(
            f"W118 Weg2FlipFormAUntagged -- Form-A role {role!r} holds device "
            f"allocation(s) {unknown} that wear no memory-saver tag. An "
            f"untagged allocation SURVIVES the sleep, and on the tightest card "
            f"of this rig (0.08 GiB free under Form-A extend, fnFA19:2194) "
            f"that is the same sentence as 'the other layout OOMs on resume' "
            f"-- reported as an allocation failure in the WAKING group, "
            f"pointing at the wrong side. Add each to FORM_A_ALLOCATIONS with "
            f"its role set and the log line that proves it, or prove it is "
            f"host-side and therefore correctly untagged (the shared expert "
            f"pool is the one legitimate case: torch_memory_saver manages "
            f"DEVICE memory only, constants.py:2-4)."
        )

    misplaced = sorted(n for n in live if role not in known[n].roles)
    if misplaced:
        detail = "; ".join(
            f"{n} belongs to {sorted(known[n].roles)}" for n in misplaced
        )
        raise Weg2FlipFormAUntagged(
            f"W118 Weg2FlipFormAUntagged -- role {role!r} holds allocation(s) "
            f"that the Form-A plan gives to another role ({detail}). Under "
            f"Form A the host holds the whole KV and every dense weight and "
            f"the workers hold neither (rank_role.py:459-493); an allocation "
            f"on the wrong side is a layout bug, not a tagging one."
        )

    tagged = tuple((n, known[n].tag) for n in sorted(live))
    resident = tuple(
        n for n, t in tagged if t not in role_tags
    )  # the solo draft: tagged, but outside the pause population
    return SleepHookPlan(
        role=role,
        tags=tuple(sorted(role_tags)),
        sleep_order=tuple(sleep_tag_order(role, weight_chunk_tags)),
        wake_order=tuple(wake_tag_order(role, weight_chunk_tags)),
        tagged=tagged,
        resident=resident,
    )
