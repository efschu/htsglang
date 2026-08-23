# Copyright 2023-2024 SGLang Team
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
"""#823: FORCE a rank-uniform TP batch-formation decision, don't just detect it.

WHAT ALREADY EXISTS, AND WHY IT IS NOT ENOUGH
---------------------------------------------
``prefetch_ballot`` (#791b) carries a CRC digest of the first
``PREFETCH_BALLOT_SLOTS`` waiting-queue rids on the packed MIN-reduce, and
``unpack_prefetch_ballot`` returns None when the group's min and max digest
differ. That is a DETECTOR, and a good one -- #823 already gave it onset,
persistence and recovery-edge logging (scheduler.py:5061). But its own
docstring states the limit exactly:

    "On mismatch the ballot is void for the pass and the caller falls back
    to the rank-local verdict -- the status quo ante -- after saying so
    once, loudly: a divergent queue head is a deeper breakage this module
    must surface, not paper over."

Surface, then fall back to rank-local. Nothing makes the ranks agree.

WHAT THAT COSTS, measured. Specimen /spinning/evidence-816-18f/
wedge_0823_055757 (boot 0516, 2026-08-23): the ranks logged the digest
mismatch at 05:55:38; at 05:56:18 they each built a DIFFERENT prefill batch
(#new-seq 1 vs 3, #cached-token 0 vs 16384, #queue-req 6 vs 3); by 05:57:57
py-spy had two ranks in the spec VERIFY arm and one in the EXTEND arm of
``eagle_worker_v2.forward_batch_generation``, three GPUs pinned at 100%
with frozen stacks, until an external SIGTERM at 06:00:43.

WHERE THE DIVERGENCE IS BORN. ``SchedulePolicy.calc_priority``
(schedule_policy.py:197) orders ``waiting_queue`` by
``req.num_matched_prefix_tokens`` under a CacheAwarePolicy
(``_sort_by_longest_prefix``, :229-232). That number is read from the RANK-
LOCAL radix tree, and each TP rank's prefix cache evolves independently --
the #616B family. Same queue, same policy, different ORDER. The order is the
decision, and the decision is what has to be uniform.

THE RULE, transplanted from #791
--------------------------------
#791 made PP admission uniform with an asymmetric local/told rule: a rank
may only ever truncate a locally-computed value TOWARD what it was told,
never use its own longer local match to extend it. The TP sibling of that
rule is this module: the group's match length is the MIN across ranks, and
every rank sorts by the GROUP number rather than its own.

MIN IS THE SAFE DIRECTION, and for the same reason it is in #616B's evict
floor ("min <= local, so every rank evicts at least as often ... under-
eviction is arithmetically impossible") and in the ballot itself ("MIN == AND
... the ballot only ever DELAYS an admission, never forces one"). A rank is
never told it has a longer prefix than it really has, because the group
minimum is <= its own value. The worst case is that a rank re-computes a
prefix it already had cached: slower, never wrong.

THE CIRCULARITY, and how it is broken
-------------------------------------
Per-rid values cannot be reduced by QUEUE POSITION, because the positions
are precisely what diverge -- slot i means a different request on different
ranks, and a MIN over that is meaningless. So the slots are indexed by a
CANONICAL rid order (``sorted`` of the rid strings) instead, which depends
only on the rid SET and on nothing rank-local. The set is the replicated
part; the order is the part that drifts.

A rank that does not hold a rid at all contributes ``_ABSENT_MATCH`` (-1),
which MIN-reduces to -1 if ANY rank is missing it. The group then drops that
rid from the uniform head entirely rather than admitting a request a peer
cannot form -- delay, never force, exactly the ballot's own safety property.

``sorted`` on the rid strings is deterministic across processes; ``hash()``
is NOT (PYTHONHASHSEED) and must never touch this path, the same rule the
ballot's digest lives under.

PURE ON PURPOSE. Everything here is a function of its arguments, so a test
drives the real decision instead of grepping the source for a branch. That
is #823's own lesson, recorded in uniform_floor_scope.py:45 -- "with this
logic inline behind a real all_reduce, the only thing a test could check is
whether the source still mentions a transition branch" -- and it is why a
mutant that disabled the recovery edge once survived a whole suite.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

#: Head depth the enforcer agrees on. Matches PREFETCH_BALLOT_SLOTS so the
#: enforcer covers exactly the window the detector already watches; a rid
#: deeper than this is handled conservatively by the caller (it waits until
#: it reaches the head), which can only delay it, never split the group.
TP_HEAD_SLOTS = 32

#: "This rank does not hold this rid." MIN-reduces to itself, so one rank
#: missing a request removes it from the group's head.
_ABSENT_MATCH = -1


def canonical_head_rids(rids: Sequence[str], slots: int = TP_HEAD_SLOTS) -> List[str]:
    """The slot->rid mapping, derived from the rid SET alone.

    Deliberately NOT queue order: queue order is the diverging quantity, so
    indexing the reduce by it would compare different requests across ranks.
    ``sorted`` is stable, total, and identical in every process.
    """
    return sorted(set(rids))[:slots]


def build_head_order_payload(
    canonical: Sequence[str],
    local_match_lens: Dict[str, int],
    slots: int = TP_HEAD_SLOTS,
) -> List[int]:
    """This rank's vote: its own match length for each canonical slot.

    Unheld rids and unused slots both ride as ``_ABSENT_MATCH``, so an
    absent request and an empty slot reduce identically -- there is nothing
    for the group to admit in either case.
    """
    payload = [_ABSENT_MATCH] * slots
    for i, rid in enumerate(canonical[:slots]):
        payload[i] = int(local_match_lens.get(rid, _ABSENT_MATCH))
    return payload


def uniform_head_order(
    canonical: Sequence[str],
    group_match_lens: Sequence[int],
    slots: int = TP_HEAD_SLOTS,
) -> List[str]:
    """The order EVERY rank must form its batch in.

    ``group_match_lens`` is the MIN-reduced payload, so entry i is the
    smallest match any rank has for ``canonical[i]``.

    Sorted by descending group match length -- the same intent as
    ``_sort_by_longest_prefix``, computed from the group's number instead of
    this rank's -- with the rid string as the tiebreak. The tiebreak is not
    decoration: equal match lengths are the common case on a cold cache, and
    a tiebreak that depended on anything rank-local would put the divergence
    straight back in.
    """
    ranked = [
        (rid, int(group_match_lens[i]))
        for i, rid in enumerate(canonical[:slots])
        if i < len(group_match_lens) and int(group_match_lens[i]) > _ABSENT_MATCH
    ]
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return [rid for rid, _ in ranked]


def local_head_order(
    rids: Sequence[str], local_match_lens: Dict[str, int]
) -> List[str]:
    """TODAY'S rule, as a pure function: sort by this rank's own match.

    A faithful transcription of ``_sort_by_longest_prefix``'s intent, kept
    here so a test can show the two rules SIDE BY SIDE on the same inputs
    and demonstrate that this one diverges while the group rule does not.
    It is not called by production.
    """
    ordered = list(rids)
    ordered.sort(key=lambda rid: (-int(local_match_lens.get(rid, 0)), rid))
    return ordered


#: What produced the head this pass. Reported so the caller can log it and
#: a test can assert WHICH rule ran, not merely what it returned.
SOURCE_GROUP = "group"
SOURCE_RANK_LOCAL = "rank-local"


def head_decision(
    canonical: Sequence[str],
    group_match_lens: Sequence[int],
    local_rids: Sequence[str],
    local_match_lens: Dict[str, int],
    digest_agreed: bool,
    enforcer_enabled: bool,
    slots: int = TP_HEAD_SLOTS,
):
    """THE SECOND BEHAVIOUR CHANGE, and the one easiest to leave implicit.

    Returns ``(order, source)``.

    Today, a digest mismatch voids the ballot and admission falls back to
    the RANK-LOCAL verdict (scheduler.py:5111-5124) -- every rank then
    orders by its own prefix matches, which is the divergence itself. The
    enforcer must not merely improve the agreeing case; it has to replace
    that fallback, because the mismatch case IS the wedge case.

    THE MISMATCH IS NOT FATAL TO THIS RULE, which is what makes replacing
    the fallback possible at all. The group order is derived from the
    canonical rid SET and the MIN-reduced match lengths, neither of which
    depends on any rank's local ORDER -- so it is still computable in
    exactly the pass where the digest says the orders disagree. The digest
    stays a first-class signal (it is what tells an operator the ranks
    drifted, with onset and recovery), it simply stops being the thing that
    decides which rule runs.

    ``enforcer_enabled`` is the kill switch and the mutant lever: turn it
    off and both branches return the rank-local order, i.e. today's
    behaviour, and the divergence goes silent again.
    """
    if not enforcer_enabled:
        return local_head_order(local_rids, local_match_lens), SOURCE_RANK_LOCAL
    return (
        uniform_head_order(canonical, group_match_lens, slots=slots),
        SOURCE_GROUP,
    )


#: "This rank cannot price its own limit." Rides as a large sentinel so a
#: rank with nothing to say cannot pull the group's MIN down to zero.
_UNPRICED_LIMIT = 1 << 30


def build_admit_limit_payload(local_limit: Optional[int]) -> List[int]:
    """This rank's vote for HOW MANY of the group head may be admitted.

    One slot, MIN-reduced beside the order slots. ``None`` means this rank
    has no opinion (no allocator to ask) and rides as the sentinel, exactly
    as the host and mamba tiers do in the existing reduce.
    """
    if local_limit is None:
        return [_UNPRICED_LIMIT]
    return [max(0, int(local_limit))]


def admit_limit_decision(
    local_limit: Optional[int],
    group_limit: Optional[int],
    enforcer_enabled: bool,
):
    """THE SECOND VARIABLE. Same defect, same rule, different quantity.

    Returns ``(limit, source)``.

    Making the ORDER uniform is not sufficient: the candidate loop stops on
    a RANK-LOCAL count (scheduler.py:7542 ``get_num_allocatable_reqs`` and
    :7547 ``req_to_token_pool.available_size()``), and neither rides the
    #610/#616g uniform floor that already covers ``PrefillAdder``'s token
    budget. Equal order with unequal count is still unequal batches -- it is
    what puts "#new-seq 1 vs 3" in the 0516 specimen alongside the
    "#cached-token 0 vs 16384" that the order arm explains.

    MIN, and for the third time in this file it is the safe direction: the
    group limit is <= every rank's own, so no rank is ever asked to admit
    more than it can seat. It can only ADMIT FEWER than it would have --
    delay, never force. The rank-local capacities stay the INPUTS; only the
    decision built from them becomes uniform, which is the
    kein-bindender-rang line: a binding rank shortens this pass, it does not
    own a permanent share.

    An unpriced group reading (every rank sentinel) leaves the local limit
    untouched, so a configuration with no allocator behaves exactly as it
    does today.
    """
    if not enforcer_enabled or group_limit is None:
        return local_limit, SOURCE_RANK_LOCAL
    group_limit = int(group_limit)
    if group_limit >= _UNPRICED_LIMIT:
        return local_limit, SOURCE_RANK_LOCAL
    return group_limit, SOURCE_GROUP


def batch_decision(
    canonical: Sequence[str],
    group_match_lens: Sequence[int],
    local_rids: Sequence[str],
    local_match_lens: Dict[str, int],
    local_limit: Optional[int],
    group_limit: Optional[int],
    digest_agreed: bool,
    enforcer_enabled: bool,
    slots: int = TP_HEAD_SLOTS,
):
    """The whole decision: WHICH requests, in WHICH order, HOW MANY.

    Returns ``(admitted, order_source, limit_source)``. Both halves must be
    uniform for the batch to be uniform, which is why they are decided in
    one place rather than in two call sites that could drift apart.
    """
    order, order_source = head_decision(
        canonical,
        group_match_lens,
        local_rids,
        local_match_lens,
        digest_agreed=digest_agreed,
        enforcer_enabled=enforcer_enabled,
        slots=slots,
    )
    limit, limit_source = admit_limit_decision(
        local_limit, group_limit, enforcer_enabled
    )
    if limit is None:
        return list(order), order_source, limit_source
    return list(order)[: max(0, int(limit))], order_source, limit_source


# ---------------------------------------------------------------------------
# W9b: the decision has to REACH the batch, and its absence has to be LOUD.
# ---------------------------------------------------------------------------
#
# Everything above decides correctly and was proven to, fifteen cases green,
# while the boot formed rank-locally 105 times out of 105. What failed was
# the hand-over: the pass destroyed the memo before the consumer read it
# (scheduler.py:7429 pops `_uniform_prefetch_ballot`, :5314 read it), and
# the COUNT arm's own hand-over failed SILENTLY through a
# `getattr(..., None)` that turned a structural defect into a plausible
# number -- the #606 family.
#
# So this block carries two things and no new decision: a VALUE that holds
# the whole group verdict for one pass (a parameter cannot be absent, which
# is the only guarantee stronger than a default), and the pure arithmetic of
# saying so when it is missing.


@dataclasses.dataclass(frozen=True)
class UniformHeadInputs:
    """One pass's group verdict, as a value rather than as four attributes.

    FOUR ATTRIBUTES WERE THE BUG. They were published at one point in the
    pass, consumed at another, and one of the four was destroyed in between
    -- a lifecycle with four independent chances to be wrong, and the boot
    took one of them on every pass. Bundled, they are taken once and handed
    down; there is no window in which three of them are live and the fourth
    is a hole.

    ``digest_agreed`` rides along because it is REPORTED, not because it
    decides: ``head_decision`` deliberately does not branch on it (see its
    docstring -- the group order is computable in exactly the pass where the
    digests disagree, which is the point of the whole ticket). It was
    nonetheless the field whose direct read raised, so it is carried here
    where reading it is free rather than left to be fetched from a memo the
    pass has already consumed.
    """

    canonical: Tuple[str, ...]
    group_match_lens: Tuple[int, ...]
    admit_limit: Optional[int]
    digest_agreed: bool


def build_uniform_head_inputs(
    canonical: Sequence[str],
    group_match_lens: Sequence[int],
    admit_limit: Optional[int],
    digest_agreed: bool,
) -> UniformHeadInputs:
    """Freeze this pass's reduce results into the value the pass hands down."""
    return UniformHeadInputs(
        canonical=tuple(canonical or ()),
        group_match_lens=tuple(int(v) for v in (group_match_lens or ())),
        admit_limit=None if admit_limit is None else int(admit_limit),
        digest_agreed=bool(digest_agreed),
    )


#: Why the enforcer is or is not in force this pass. Named strings rather
#: than a bare bool, because "inert" and "correctly gated off" read
#: identically in a log that reports neither -- and that ambiguity is what
#: cost W9 a GPU window: the boot could not say whether the enforcer was
#: broken or simply out of scope.
GATE_ON = "on"
GATE_OFF_KILL_SWITCH = "off:kill-switch"
GATE_OFF_NO_PARALLEL_STATE = "off:no-parallel-state"
GATE_OFF_TP_WORLD_OF_ONE = "off:tp-world-of-one"


@dataclasses.dataclass(frozen=True)
class GateVerdict:
    enabled: bool
    reason: str
    #: Human-readable, for the one log line this produces per transition.
    detail: str


def enforcer_gate(
    kill_switch_on: bool,
    tp_size: Optional[int],
    pp_size: Optional[int] = None,
) -> GateVerdict:
    """Is the group's batch-formation decision in force, and if not, WHY.

    THE TP WORLD OF ONE IS NOT A WIDENING OPPORTUNITY, and this is the
    decision W9's window left open. Under ``--tp-size 1 --pp-size 3`` this
    predicate returns off, 45 of that boot's 51 prefill batches ran in that
    phase, and the obvious reading was "so widen the gate to cover PP".

    Three reasons in the code say otherwise, and they are why this function
    reports a REASON instead of growing a `pp_size > 1` branch:

    1. The enforcer's numbers come from a MIN-reduce over ``tp_cpu_group``,
       and under tp=1 that group has exactly ONE member per rank
       (parallel_state.py:3166-3188 chunks the world into ``world_size //
       tp_size`` groups of size 1). A MIN over a singleton is this rank's
       own vote. Enforcing it would impose a rank-local verdict while
       calling it the group's -- worse than off, because it would also look
       right.
    2. Supplying the missing collective where the decision is consumed is
       forbidden, with a casualty: #737 (scheduler.py:7402-7419) records
       that the PP stages sit at different microbatch offsets by design and
       that the HiCache ack-count reduction deadlocked there on 2026-08-17.
    3. The PP phase already has this actuator and a stronger one. #791
       forwards the first rank's committed decision around the ring;
       downstream ranks may not drop a named request or add an unnamed one
       (``PPScheduleRefused``, scheduler.py:7903-7941) and are re-ordered
       into the forwarded order by ``order_batch_by_schedule``. This
       module's own rule is a transplant of that one (see "THE RULE,
       transplanted from #791" above).

    ``pp_size`` is therefore taken and REPORTED, never branched on: the
    reason string distinguishes "no TP group to agree with, and #791 covers
    this phase" from "no TP group at all", which a bare `False` cannot.
    """
    if not kill_switch_on:
        return GateVerdict(
            False,
            GATE_OFF_KILL_SWITCH,
            "SGLANG_TP_HEAD_CONGRUENCE=0; pre-#823 rank-local formation restored",
        )
    if tp_size is None:
        return GateVerdict(
            False,
            GATE_OFF_NO_PARALLEL_STATE,
            "no ParallelState on this scheduler; nothing to agree with",
        )
    if int(tp_size) > 1:
        return GateVerdict(True, GATE_ON, f"TP world of {int(tp_size)}")
    covered = (
        " -- this phase's formation congruence is #791's forwarded PP "
        "admission decision, not this enforcer"
        if pp_size is not None and int(pp_size) > 1
        else ""
    )
    return GateVerdict(
        False,
        GATE_OFF_TP_WORLD_OF_ONE,
        f"TP world of 1 (pp_size={pp_size}): the reduce group has one member, "
        f"so its MIN is this rank's own vote{covered}",
    )


def advance_gate_report(previous_reason: Optional[str], verdict: GateVerdict):
    """Report the TRANSITION, not the first sighting.

    Same rule and same reason as #824's ``uniform_floor_scope.report_scope``
    (scheduler.py:4877-4901): under ``--enable-phase-flip`` this gate is not
    a boot constant -- ``phase_flip_runtime`` rebuilds the TP group per phase
    (``want_tp_size = n if tp_phase else 1``), so the enforcer is ON through
    the TP decode phase and OFF through the PP prefill phase and switches at
    every cutover. A once-per-process latch made "off for the whole run" and
    "off for the prefill half of every cutover" read identically, and
    coverage coming BACK was never reported at all.

    Returns ``(reason, message_or_None)``.
    """
    if previous_reason == verdict.reason:
        return verdict.reason, None
    return verdict.reason, verdict.detail


#: Which half of the batch decision fell back. Named so the counter and the
#: log can say WHICH, instead of leaving a reader to guess -- the count arm
#: had no name and no line at all, which is why its inertness was invisible.
ARM_ORDER = "order"
ARM_COUNT = "count"


def degradation_is_a_defect(gate_enabled: bool, source: str) -> bool:
    """With the enforcer in force, a rank-local outcome is a DEFECT.

    Not a tuning knob and not a soft fallback: the enforcer being enabled is
    the operator saying the ranks must agree, so a pass that forms
    rank-locally anyway has silently reverted to the behaviour the 0516
    wedge came out of. It is counted and logged on that basis.

    With the gate off, rank-local IS the contract, and reporting it would be
    noise on every pass of every single-rank boot.
    """
    return bool(gate_enabled) and source != SOURCE_GROUP


def head_order_is_uniform(orders: Sequence[Sequence[str]]) -> bool:
    """Did every rank end up with the same decision?"""
    if not orders:
        return True
    first = list(orders[0])
    return all(list(o) == first for o in orders)


__all__ = [
    "TP_HEAD_SLOTS",
    "SOURCE_GROUP",
    "SOURCE_RANK_LOCAL",
    "ARM_ORDER",
    "ARM_COUNT",
    "GATE_ON",
    "GATE_OFF_KILL_SWITCH",
    "GATE_OFF_NO_PARALLEL_STATE",
    "GATE_OFF_TP_WORLD_OF_ONE",
    "GateVerdict",
    "UniformHeadInputs",
    "advance_gate_report",
    "build_uniform_head_inputs",
    "degradation_is_a_defect",
    "enforcer_gate",
    "head_decision",
    "build_admit_limit_payload",
    "admit_limit_decision",
    "batch_decision",
    "canonical_head_rids",
    "build_head_order_payload",
    "uniform_head_order",
    "local_head_order",
    "head_order_is_uniform",
]
