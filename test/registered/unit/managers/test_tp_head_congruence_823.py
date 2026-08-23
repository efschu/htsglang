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
"""#823: the TP ranks must FORM THE SAME BATCH, not merely notice they didn't.

THE MEASURED CHAIN (specimen /spinning/evidence-816-18f/wedge_0823_055757,
boot 0516, 2026-08-23):

    05:55:38  the three ranks log the prefetch-ballot digest mismatch
    05:56:18  they each build a DIFFERENT prefill batch
              (#new-seq 1 vs 3, #cached-token 0 vs 16384, #queue-req 6 vs 3)
    05:57:57  py-spy: two ranks in the spec VERIFY arm, one in EXTEND, all
              three GPUs at 100% with frozen stacks
    06:00:43  external SIGTERM

The digest mismatch at 05:55:38 is the earliest evidence in that chain, and
today it leads to a log line and a fallback to the RANK-LOCAL verdict --
which is the divergence itself. #823 is about closing that.

TWO BEHAVIOUR CHANGES ARE PINNED HERE, each with its own can-fail arm,
because the second is the one that is easy to leave implicit:

  1. THE SORT KEY becomes the GROUP's match length (MIN across ranks)
     instead of this rank's own, so ranks with different prefix caches
     still order the head identically.
  2. THE MISMATCH BRANCH stops falling back to rank-local. The group order
     is derived from the canonical rid SET and the MIN-reduced lengths,
     neither of which depends on any rank's local ORDER -- so it is still
     computable in exactly the pass where the digest says the orders
     disagree. Improving only the agreeing case would leave the wedge case
     untouched.

WHY THE SLOTS ARE NOT QUEUE POSITIONS. Reducing per-rid values by queue
position is meaningless when the positions are what diverge: slot i is a
different request on different ranks. The canonical order (sorted rids)
depends only on the rid SET, which is the replicated part.

CPU-only: real gloo process groups, real MIN all_reduce, no CUDA.
"""

import multiprocessing as mp
import os
import socket
import sys

import pytest

RIDS = ["r-alpha", "r-bravo", "r-charlie", "r-delta"]

#: Rank-local prefix-match lengths. This is the #616B family made concrete:
#: the same four requests, three independently evolved radix caches. Rank 0
#: has a hot cache for charlie, rank 1 for alpha, rank 2 almost nothing --
#: so today's "sort by MY longest prefix" gives three different orders.
LOCAL_MATCHES = {
    0: {"r-alpha": 16, "r-bravo": 4, "r-charlie": 4096, "r-delta": 0},
    1: {"r-alpha": 8192, "r-bravo": 4, "r-charlie": 16, "r-delta": 0},
    2: {"r-alpha": 16, "r-bravo": 4, "r-charlie": 16, "r-delta": 0},
}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Part 1: the decision, as pure functions.
# ---------------------------------------------------------------------------


def _group_min(payloads):
    """What the MIN all_reduce computes, for the pure-function tests."""
    return [min(col) for col in zip(*payloads)]


def _rank_queue(rank):
    """The order this rank's waiting_queue is REALLY in.

    Every rank must feed the enforcer its own diverged order, not a shared
    fixture list -- otherwise the canonical-order step is never exercised
    and the slot indexing could be queue-position based without any test
    noticing.
    """
    from sglang.srt.managers import tp_head_congruence as thc

    return thc.local_head_order(RIDS, LOCAL_MATCHES[rank])


def test_todays_local_rule_really_does_diverge():
    """THE PREMISE. If this ever stops being true the rest is pointless, so
    it is asserted rather than assumed."""
    from sglang.srt.managers import tp_head_congruence as thc

    orders = [thc.local_head_order(RIDS, LOCAL_MATCHES[r]) for r in range(3)]

    assert not thc.head_order_is_uniform(orders), (
        "the rank-local rule agreed by accident; the fixture no longer "
        f"models divergent prefix caches: {orders}"
    )
    # Concretely: rank 0 leads with charlie, rank 1 with alpha.
    assert orders[0][0] == "r-charlie"
    assert orders[1][0] == "r-alpha"


def test_the_group_rule_is_uniform_across_ranks():
    """CHANGE 1: the sort key is the group's number, so the order agrees."""
    from sglang.srt.managers import tp_head_congruence as thc

    # Each rank derives the canonical slots from ITS OWN queue order.
    canonicals = [thc.canonical_head_rids(_rank_queue(r)) for r in range(3)]
    assert len({tuple(c) for c in canonicals}) == 1, (
        f"the canonical slot mapping is not rank-independent: {canonicals}"
    )
    payloads = [
        thc.build_head_order_payload(canonicals[r], LOCAL_MATCHES[r]) for r in range(3)
    ]
    reduced = _group_min(payloads)

    orders = [thc.uniform_head_order(canonicals[r], reduced) for r in range(3)]
    assert thc.head_order_is_uniform(orders), orders
    # MIN over the three caches: alpha 16, bravo 4, charlie 16, delta 0.
    # Descending, rid as tiebreak -> alpha, charlie, bravo, delta.
    assert orders[0] == ["r-alpha", "r-charlie", "r-bravo", "r-delta"]


def test_the_group_never_claims_a_prefix_a_rank_lacks():
    """MIN is the safe direction: the agreed length is <= every rank's own."""
    from sglang.srt.managers import tp_head_congruence as thc

    canonical = thc.canonical_head_rids(_rank_queue(0))
    payloads = [
        thc.build_head_order_payload(
            thc.canonical_head_rids(_rank_queue(r)), LOCAL_MATCHES[r]
        )
        for r in range(3)
    ]
    reduced = _group_min(payloads)

    for rank in range(3):
        for i, rid in enumerate(canonical):
            assert reduced[i] <= LOCAL_MATCHES[rank][rid], (
                f"group claimed {reduced[i]} for {rid} but rank {rank} only "
                f"has {LOCAL_MATCHES[rank][rid]} -- a rank would be told to "
                "reuse a prefix it does not hold"
            )


def test_a_rid_missing_on_one_rank_is_dropped_by_the_group():
    """Delay, never force -- the ballot's own safety property.

    A request one rank has not got cannot be admitted by the others, so it
    leaves the group's head rather than splitting the batch.
    """
    from sglang.srt.managers import tp_head_congruence as thc

    canonical = thc.canonical_head_rids(_rank_queue(0))
    holders = [dict(LOCAL_MATCHES[r]) for r in range(3)]
    del holders[2]["r-charlie"]  # rank 2 never received it

    reduced = _group_min([thc.build_head_order_payload(canonical, h) for h in holders])
    order = thc.uniform_head_order(canonical, reduced)

    assert "r-charlie" not in order, order
    assert order == ["r-alpha", "r-bravo", "r-delta"]


# ---------------------------------------------------------------------------
# Part 2: the mismatch branch -- CHANGE 2, with its own can-fail.
# ---------------------------------------------------------------------------


def _decide(rank, digest_agreed, enforcer_enabled):
    from sglang.srt.managers import tp_head_congruence as thc

    canonical = thc.canonical_head_rids(_rank_queue(rank))
    reduced = _group_min(
        [
            thc.build_head_order_payload(
                thc.canonical_head_rids(_rank_queue(r)), LOCAL_MATCHES[r]
            )
            for r in range(3)
        ]
    )
    # Each rank presents its OWN queue order, as it really would.
    local_rids = _rank_queue(rank)
    return thc.head_decision(
        canonical,
        reduced,
        local_rids,
        LOCAL_MATCHES[rank],
        digest_agreed=digest_agreed,
        enforcer_enabled=enforcer_enabled,
    )


def test_a_digest_mismatch_no_longer_falls_back_to_rank_local():
    """CHANGE 2. This is the wedge case: today a mismatch voids the ballot
    and admission uses the rank-local verdict."""
    from sglang.srt.managers import tp_head_congruence as thc

    results = [_decide(r, digest_agreed=False, enforcer_enabled=True) for r in range(3)]
    orders = [o for o, _ in results]
    sources = {s for _, s in results}

    assert sources == {thc.SOURCE_GROUP}, (
        f"a diverged pass still decided rank-locally: {sources}. That is the "
        "fallback #823 exists to replace -- the mismatch case IS the wedge case"
    )
    assert thc.head_order_is_uniform(orders), orders


def test_can_fail_with_the_enforcer_off_the_divergence_goes_silent():
    """THE MUTANT LEVER, as a test rather than an edit.

    With the enforcer disabled every rank returns its own order and the
    ranks disagree while nothing objects -- exactly today's behaviour, and
    exactly what this suite must be able to see.
    """
    from sglang.srt.managers import tp_head_congruence as thc

    results = [
        _decide(r, digest_agreed=False, enforcer_enabled=False) for r in range(3)
    ]
    orders = [o for o, _ in results]
    sources = {s for _, s in results}

    assert sources == {thc.SOURCE_RANK_LOCAL}
    assert not thc.head_order_is_uniform(orders), (
        "with the enforcer off the ranks agreed anyway, so this suite could "
        "not tell an enforced pass from an unenforced one"
    )


def test_the_agreeing_case_is_enforced_too():
    """Uniformity must not be conditional on the digest: a pass that agrees
    today can diverge on the next one, and the rule may not change under it."""
    from sglang.srt.managers import tp_head_congruence as thc

    agreed = [_decide(r, digest_agreed=True, enforcer_enabled=True) for r in range(3)]
    diverged = [_decide(r, digest_agreed=False, enforcer_enabled=True) for r in range(3)]

    assert [o for o, _ in agreed] == [o for o, _ in diverged], (
        "the head depends on the digest verdict, so the rule changes shape "
        "exactly when the ranks are already drifting"
    )


# ---------------------------------------------------------------------------
# Part 3: the same decision over a REAL gloo MIN all_reduce.
# ---------------------------------------------------------------------------


def _worker(rank, world, port, out):
    try:
        import torch
        import torch.distributed as dist

        from sglang.srt.managers import tp_head_congruence as thc

        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
        )
        # THIS RANK'S OWN diverged queue order goes in, exactly as
        # production's waiting_queue would.
        local_order = thc.local_head_order(RIDS, LOCAL_MATCHES[rank])
        canonical = thc.canonical_head_rids(local_order)
        payload = thc.build_head_order_payload(canonical, LOCAL_MATCHES[rank])
        t = torch.tensor(payload, dtype=torch.int64)
        # The production shape: one MIN reduce, no new collective.
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        order, source = thc.head_decision(
            canonical,
            t.tolist(),
            local_order,
            LOCAL_MATCHES[rank],
            digest_agreed=False,
            enforcer_enabled=True,
        )
        out[rank] = (order, source)
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        out[rank] = ("error", repr(exc)[:300])


def _run(world):
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, port, out)) for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=10)
    return dict(out)


@pytest.mark.parametrize("world", [2, 3])
def test_real_gloo_ranks_form_the_same_head(world):
    from sglang.srt.managers import tp_head_congruence as thc

    got = _run(world)

    assert len(got) == world, f"a rank produced nothing: {got}"
    for rank, (order, source) in got.items():
        assert order != "error", f"rank {rank} failed: {source}"
        assert source == thc.SOURCE_GROUP

    orders = [got[r][0] for r in range(world)]
    assert thc.head_order_is_uniform(orders), (
        f"{world} real gloo ranks formed different heads: {orders}"
    )


# ---------------------------------------------------------------------------
# Part 4: the COUNT arm -- the same defect in the second variable.
# ---------------------------------------------------------------------------
#
# Making the ORDER uniform is not sufficient. The candidate loop stops on a
# RANK-LOCAL count (scheduler.py:7542 get_num_allocatable_reqs, :7547
# req_to_token_pool.available_size()), and neither rides the #610/#616g
# uniform floor that already covers PrefillAdder's token budget. Equal order
# with unequal count is still unequal batches -- it is what puts
# "#new-seq 1 vs 3" in the 0516 specimen next to the "#cached-token 0 vs
# 16384" the order arm explains.

#: Same order on both ranks (identical matches), different free pool. Rank 0
#: can seat three requests, rank 1 only one. These are the specimen's numbers.
SAME_MATCHES = {"r-alpha": 16, "r-bravo": 16, "r-charlie": 16, "r-delta": 16}
LOCAL_LIMITS = {0: 3, 1: 1}


def _count_decide(rank, enforcer_enabled, limits=None):
    from sglang.srt.managers import tp_head_congruence as thc

    limits = LOCAL_LIMITS if limits is None else limits
    canonical = thc.canonical_head_rids(RIDS)
    reduced = _group_min(
        [thc.build_head_order_payload(canonical, SAME_MATCHES) for _ in limits]
    )
    group_limit = _group_min(
        [thc.build_admit_limit_payload(limits[r]) for r in sorted(limits)]
    )[0]
    return thc.batch_decision(
        canonical,
        reduced,
        thc.local_head_order(RIDS, SAME_MATCHES),
        SAME_MATCHES,
        local_limit=limits[rank],
        group_limit=group_limit,
        digest_agreed=True,
        enforcer_enabled=enforcer_enabled,
    )


def test_todays_local_count_really_does_diverge():
    """THE PREMISE for this arm: same ORDER, different COUNT, so the batches
    still differ -- the case an order-only fix would leave wedged."""
    admitted = [_count_decide(r, enforcer_enabled=False)[0] for r in (0, 1)]

    assert len(admitted[0]) == 3 and len(admitted[1]) == 1, admitted
    assert admitted[0] != admitted[1], (
        "the count fixture no longer models divergent pool pressure"
    )


def test_the_group_count_is_uniform_across_ranks():
    """After the fix both ranks admit the SAME requests, MIN many."""
    from sglang.srt.managers import tp_head_congruence as thc

    results = [_count_decide(r, enforcer_enabled=True) for r in (0, 1)]
    admitted = [a for a, _, _ in results]
    limit_sources = {ls for _, _, ls in results}

    assert limit_sources == {thc.SOURCE_GROUP}, limit_sources
    assert admitted[0] == admitted[1], admitted
    assert len(admitted[0]) == 1, (
        f"the group admitted more than the binding rank can seat: {admitted}"
    )


def test_the_group_never_asks_a_rank_to_seat_more_than_it_can():
    """MIN, delay never force: the agreed count is <= every rank's own."""
    for rank in (0, 1):
        admitted, _, _ = _count_decide(rank, enforcer_enabled=True)
        assert len(admitted) <= LOCAL_LIMITS[rank], (
            f"rank {rank} was told to admit {len(admitted)} with room for "
            f"{LOCAL_LIMITS[rank]}"
        )


def test_can_fail_with_the_count_enforcer_off_the_divergence_goes_silent():
    """THE MUTANT LEVER for this arm, as a test."""
    from sglang.srt.managers import tp_head_congruence as thc

    results = [_count_decide(r, enforcer_enabled=False) for r in (0, 1)]
    admitted = [a for a, _, _ in results]
    limit_sources = {ls for _, _, ls in results}

    assert limit_sources == {thc.SOURCE_RANK_LOCAL}
    assert admitted[0] != admitted[1], (
        "with the count enforcer off the ranks agreed anyway, so this suite "
        "could not tell an enforced pass from an unenforced one"
    )


def test_an_unpriced_group_leaves_the_local_limit_untouched():
    """A configuration with no allocator to ask must behave exactly as it
    does today, not collapse to a zero-sized batch."""
    from sglang.srt.managers import tp_head_congruence as thc

    limit, source = thc.admit_limit_decision(
        local_limit=5,
        group_limit=thc.build_admit_limit_payload(None)[0],
        enforcer_enabled=True,
    )
    assert limit == 5 and source == thc.SOURCE_RANK_LOCAL


def test_both_arms_are_required_for_a_uniform_batch():
    """ORDER alone and COUNT alone each leave a divergent batch, which is why
    W9 is only green when both are enforced."""
    from sglang.srt.managers import tp_head_congruence as thc

    canonical = thc.canonical_head_rids(RIDS)
    reduced = _group_min(
        [thc.build_head_order_payload(canonical, SAME_MATCHES) for _ in (0, 1)]
    )
    order_only = [
        thc.batch_decision(
            canonical,
            reduced,
            thc.local_head_order(RIDS, SAME_MATCHES),
            SAME_MATCHES,
            local_limit=LOCAL_LIMITS[r],
            group_limit=None,
            digest_agreed=True,
            enforcer_enabled=True,
        )[0]
        for r in (0, 1)
    ]
    assert order_only[0] != order_only[1], (
        "an order-only fix looked sufficient; the count arm would then be "
        "untestable and #new-seq 1 vs 3 would survive it"
    )
