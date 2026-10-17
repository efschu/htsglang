"""K21 V1 (nf-nachreview2-h16v-1017 point 1, S11; law raenge-nie-uneins-crash-stop): the write ack of
a direct write is rank-LOCAL and ASYNCHRONOUS (finish_event.query() per rank; metal D.log
1017_132112: TP0 27 ms, TP1/2 2 ms, one decode pass ~87 ms).  A worker therefore pairs, per content
key and in FIFO order, its OWN ack finding with TP0's (K21 event with orphaned=0|1 at EVERY direct
ack) and judges only when BOTH exist: no timer, no handover count, no collective.

Host + two workers, the real ``_k12_direct_l3_verdict`` ack code, the real R12 broadcast/apply.
Steps: ("ack", rank) = that rank's ack of the write, ("bc",) = one broadcast.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import test_form_a_host_shadow_r12 as T  # noqa: E402
from test_k21_both_directions_1017 import _wire  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

m = T.m


def _run(orphans, steps, release_fails=False):
    """Returns the index of the step that raised K21RankDivergence, or None."""
    ranks = T._ranks()
    leaves = [_wire(r, o, release_fails=(release_fails and i == 0))
              for i, (r, o) in enumerate(zip(ranks, orphans))]
    if release_fails:
        # TP0 cannot give the host layer back: NO STATE is published (the "no_state" form)
        # (the release is a no-op and the host layer stays => ``released`` False => no record_state)
        leaves[0].component_data[T.FULL].host_value = T.torch.arange(0, T.NODE)
    with T._switch(True):
        for i, st in enumerate(steps):
            try:
                if st[0] == "ack":
                    r = ranks[st[1]]
                    with T._as_rank(r):
                        U._k12_direct_l3_verdict(r, leaves[st[1]], 1 if orphans[st[1]] else 0)
                else:
                    T._broadcast(ranks)
            except m.K21RankDivergence:
                return i
    return None


ORDERS = {
    "tp0_first": [("ack", 0), ("bc",), ("ack", 1), ("ack", 2), ("bc",), ("bc",)],
    "workers_first": [("ack", 1), ("ack", 2), ("bc",), ("ack", 0), ("bc",), ("bc",)],
    "interleaved": [("ack", 1), ("bc",), ("ack", 0), ("bc",), ("ack", 2), ("bc",), ("bc",)],
    "late_worker_many_handovers": [("ack", 0), ("bc",), ("bc",), ("bc",), ("ack", 1), ("ack", 2), ("bc",)],
    "lockstep": [("ack", 0), ("ack", 1), ("ack", 2), ("bc",), ("bc",)],
}


@pytest.mark.parametrize("release_fails", [False, True], ids=["with_state", "no_state"])
@pytest.mark.parametrize("order", sorted(ORDERS))
def test_both_orphan_staggered_either_direction_never_stops(order, release_fails):
    assert _run([True, True, True], ORDERS[order], release_fails=release_fails) is None


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_clean_acks_staggered_never_stop(order):
    assert _run([False, False, False], ORDERS[order]) is None


def test_only_tp0_orphan_stops_when_the_workers_clean_finding_arrives():
    steps = [("ack", 0), ("bc",), ("bc",), ("ack", 1), ("ack", 2), ("bc",)]
    assert _run([True, False, False], steps, release_fails=True) == 5


def test_only_tp0_orphan_workers_first_stops_at_the_verdict():
    steps = [("ack", 1), ("ack", 2), ("ack", 0), ("bc",)]
    assert _run([True, False, False], steps, release_fails=True) == 3


def test_only_worker_orphan_stops_when_tp0s_clean_verdict_arrives():
    steps = [("ack", 1), ("bc",), ("bc",), ("ack", 0), ("bc",)]
    assert _run([False, True, False], steps) == 4


def test_only_worker_orphan_tp0_first_stops_at_the_next_consume():
    steps = [("ack", 0), ("bc",), ("bc",), ("ack", 1), ("bc",)]
    assert _run([False, True, False], steps) == 4


def test_worker_orphan_without_any_tp0_finding_does_not_stop_early():
    # the old karenz stopped one handover after the mark; the pairing waits for TP0's finding
    steps = [("ack", 1), ("bc",), ("bc",), ("bc",)]
    assert _run([False, True, False], steps) is None


def test_fifo_per_key_two_acks_of_the_same_key_pair_in_order():
    ranks = T._ranks()
    leaves = [_wire(r, True) for r in ranks]
    with T._switch(True):
        with T._as_rank(ranks[0]):
            U._k12_direct_l3_verdict(ranks[0], leaves[0], 1)
            U._k12_direct_l3_verdict(ranks[0], leaves[0], 0)
        T._broadcast(ranks)
        for i in (1, 2):
            with T._as_rank(ranks[i]):
                U._k12_direct_l3_verdict(ranks[i], leaves[i], 1)
                U._k12_direct_l3_verdict(ranks[i], leaves[i], 0)
        T._broadcast(ranks)
        T._broadcast(ranks)


def test_reset_clears_the_pairing_buffers_on_every_rank():
    ranks = T._ranks()
    leaves = [_wire(r, True) for r in ranks]
    with T._switch(True):
        with T._as_rank(ranks[1]):
            U._k12_direct_l3_verdict(ranks[1], leaves[1], 1)
            assert m._S.k21_own
            m.on_tree_reset(ranks[1])
            assert not m._S.k21_own and not m._S.k21_tp0


def test_fifo_per_key_two_own_acks_wait_for_tp0_without_loss():
    # the workers ack write 1 (orphan) and write 2 (clean) BEFORE TP0's verdicts arrive: both own
    # findings must queue, and pair in order with TP0's later (1, 0) -- no stop, no lost finding
    ranks = T._ranks()
    leaves = [_wire(r, True) for r in ranks]
    with T._switch(True):
        for i in (1, 2):
            with T._as_rank(ranks[i]):
                U._k12_direct_l3_verdict(ranks[i], leaves[i], 1)
                U._k12_direct_l3_verdict(ranks[i], leaves[i], 0)
                assert m._S.k21_own[(m.node_key(leaves[i]))] == [1, 0]
        with T._as_rank(ranks[0]):
            U._k12_direct_l3_verdict(ranks[0], leaves[0], 1)
            U._k12_direct_l3_verdict(ranks[0], leaves[0], 0)
        T._broadcast(ranks)
        T._broadcast(ranks)
        for i in (1, 2):
            with T._as_rank(ranks[i]):
                assert not m._S.k21_own and not m._S.k21_tp0
