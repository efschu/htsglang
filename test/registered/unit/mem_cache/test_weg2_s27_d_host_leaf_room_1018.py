# SPDX-License-Identifier: Apache-2.0
"""S27 (18.10., fLLiper rc8 = h16v6 7d80245b52, D TP0 06:58:37-47): the READ-ROOM-D spill skipped HOST-ONLY leaves.

Metal: ``READ-ROOM SERVE need=2 deficit=2 released_pages=0`` for 10 s (334 rounds), ``D-SPILL candidates=0
tree_nodes=56 kept=no_host:28,anchor:5,...,host_only:22``, arena 6478/6485 pinned by D's own tree, then
``STORE READ CAPACITY-CUT rid=pdflip-8-119 delivered=13376 deliverable=135488`` (the read was served later with
d_compute=0, so the cost was 45 s of latency, not tokens). ``form_a_spill`` takes nodes that still carry a device copy;
a host-only H-leaf with a provable L3 copy is the same kind of cache line, and ``_w3_arena_spill`` (K23) already drops
such leaves on TP0 of a Form A group -- READ-ROOM-D never called that round.

Fix: ``form_a_spill`` runs the K23 host-only round when the device round does not cover the deficit; every dropped leaf
is sent to the workers as a node STATE(exists=0) (R11/R12: TP0 decides, the group follows).
"""
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402

from sglang.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402
from sglang.srt.weg2 import read_room as rr  # noqa: E402

from test_w3_spill_host_leaves_int22_1008 import SPAN, _AnchorPool, _death_shape  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    R12._reset_state_for_tests()
    rr._reset_for_tests() if hasattr(rr, "_reset_for_tests") else None
    yield
    R12._reset_state_for_tests()


def _spill(cache, pool, need, role="host", **kw):
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
    with mock.patch.object(R12, "role", return_value=role), mock.patch.object(rr, "form_a_role", return_value=role):
        return rr.form_a_spill(cache, pool, need, **kw)


def _sent(exists):
    return {(e[1], int(e[2])) for e in R12._S.ledger if e[0] == R12.STATE and e[3] == exists}


def test_host_only_leaves_give_the_room_the_device_round_cannot():
    """RED before: released=0 (host_only skipped, no other candidate). GREEN: leaf, then its parent, leave L2 with
    their L3 copy; TP0's tree and the verdicts to the workers agree."""
    cache, top, mid, leaf = _death_shape()
    keys = {R12.node_key(leaf), R12.node_key(mid)}
    pool = _AnchorPool()
    out = []
    got = _spill(cache, pool, 2, released_out=out)
    assert got == 2 * SPAN, "both host-only nodes of the chain are released"
    assert pool.secured == [SPAN, SPAN], "every page got its L3 copy first"
    assert leaf not in cache._collect_all_nodes() and mid not in cache._collect_all_nodes()
    assert _sent(0) == keys, "the workers are told: STATE exists=0 per dropped leaf"


def test_a_leaf_without_a_complete_l3_copy_stays():
    """The danger edge: the host copy leaves L2 only with its L3 copy (lost != 0 keeps the leaf)."""
    cache, top, mid, leaf = _death_shape()
    pool = _AnchorPool(lose={100})          # the leaf's rows start at 100
    got = _spill(cache, pool, 2)
    assert got == 0
    assert leaf in cache._collect_all_nodes() and mid in cache._collect_all_nodes()
    assert _sent(0) == set()


def test_a_host_locked_leaf_stays():
    cache, top, mid, leaf = _death_shape(host_lock=1)
    got = _spill(cache, _AnchorPool(), 2)
    assert got == 0 and leaf in cache._collect_all_nodes()
    assert _sent(0) == set()


def test_a_worker_never_drops_on_its_own():
    """R11: only TP0 (role host) runs the round; a worker's form_a_spill answers 0 and touches nothing."""
    cache, top, mid, leaf = _death_shape()
    got = _spill(cache, _AnchorPool(), 2, role="worker")
    assert got == 0 and leaf in cache._collect_all_nodes() and _sent(0) == set()


def test_the_round_is_not_entered_when_the_device_round_covers_the_deficit(monkeypatch):
    """Danger edge: no host-only drop when the deficit is already covered (a cache line is not spent for nothing)."""
    cache, top, mid, leaf = _death_shape()
    from sglang.srt.weg2 import host_only_spill as HOS

    calls = []
    monkeypatch.setattr(HOS, "spill_host_only", lambda *a, **k: calls.append(1) or {"released": 0, "leaves": 0})
    # a device-resident candidate that covers need: give `top` a host copy + l3 state is heavy; instead need=0
    got = _spill(cache, _AnchorPool(), 0)
    assert calls == [] and got == 0


def test_rec_is_restored_when_the_round_raises():
    cache, top, mid, leaf = _death_shape()
    real = cache._evict_host_leaf
    sentinel = ["outer"]
    cache._r12_rec = sentinel

    def boom(node, tracker):
        real(node, tracker)
        raise RuntimeError("after first drop")

    with mock.patch.object(cache, "_evict_host_leaf", side_effect=boom):
        with pytest.raises(RuntimeError):
            _spill(cache, _AnchorPool(), 2)
    assert cache._r12_rec is sentinel
    assert _sent(0) == {R12.node_key(leaf)}, "the node TP0 already dropped reaches the workers"
