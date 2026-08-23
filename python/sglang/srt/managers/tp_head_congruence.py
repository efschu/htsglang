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

from typing import Dict, List, Optional, Sequence

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
