"""Review KVLEITER-1022 F2 (09de38ffbd) + I1 (3ea132ca14): the passes after a mixed plan, slot collisions, and what
the census calls a mixed segment that waits behind an un-backed child. CPU only, desk trees of test_s77.

* RF1 progress: pass 1 demotes the L2 tail and hands the store prefix to S39; TP0's S39 flush accepts the prefix
  (its only child is the tail, host-only, backed, ``l3_present`` -- the K12 direct-ack verdict on a complete page);
  after the drop the next pass is no staged repeat of the same digest.
* RF2 the dependency named: the same prefix with a tail WITHOUT ``l3_present`` (cut page not complete yet) is KEPT
  by S39 (``child_not_in_store``) -- no drop without a copy, the plan repeats until the verdict lands.
* RF3 collisions: different mixed content in the same vote slot on different ranks, or two mixed segments in one
  slot on one rank -- every rank refuses alike (no plan anywhere).
* RF4 (xfail strict, Auflage I1-A): a mixed segment whose device child is UN-BACKED is counted ``_mixed`` and its
  blocker is not named; it is child-blocked (the class F1 removes), and BELOW/blocked_by must say so.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402,F401

from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import forma_l3_subtree as S39  # noqa: E402

from test_s77_kv_compact_s9_hold_1019 import FULL, Tree, _alloc, _node, _take  # noqa: E402

P, NP = 4, 32


def _store(n):
    n.component_data[FULL].host_value = None
    n.l3_present = True
    return n


def _chain(pages0=17, cut=(2, 2), tail_l3=True):
    alloc = _alloc(NP, P)
    tree = Tree(alloc)
    _take(alloc, list(range(1, 17)))
    pre = list(range(pages0, pages0 + cut[0]))
    tail = list(range(pages0 + cut[0], pages0 + cut[0] + cut[1]))
    a = _store(_node(tree, alloc, pre, P))
    b = _node(tree, alloc, tail, P, parent=a)
    b.l3_present = bool(tail_l3)
    return alloc, tree, a, b


def _s39_drop(tree, alloc, a, order):
    """What TP0's flush does to an accepted leaf (``_execute``): the host-only subtree leaves the tree, then the
    store-backed leaf's device rows go."""
    for d in order:
        d.parent.children = {k: v for k, v in d.parent.children.items() if v is not d}
    alloc.free(a.component_data[FULL].value)
    a.component_data[FULL].value = None
    a.parent.children = {k: v for k, v in a.parent.children.items() if v is not a}


def test_rf1_a_mixed_plan_progresses_through_the_s39_drop():
    alloc, tree, a, b = _chain()
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan, why = KC.plan_for(sv, 16)
    assert plan is not None and plan.staged, why
    d1 = KC.plan_digest(5, plan)
    KC.execute_evicts(tree, plan)                          # the L2 tail now (host copy stays)
    assert b.component_data[FULL].value is None and b.component_data[FULL].host_value is not None
    order, refused = S39.subtree(tree, a, tp0=True)
    assert refused is None and order == [b], refused
    _s39_drop(tree, alloc, a, order)
    sv2 = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan2, why2 = KC.plan_for(sv2, 16)
    assert plan2 is not None, why2
    assert not plan2.staged and KC.plan_digest(5, plan2) != d1, "no staged repeat after the drop landed"


def test_rf2_a_tail_without_its_store_verdict_keeps_the_prefix():
    alloc, tree, a, b = _chain(tail_l3=False)
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan, why = KC.plan_for(sv, 16)
    assert plan is not None and plan.staged, why
    KC.execute_evicts(tree, plan)
    order, refused = S39.subtree(tree, a, tp0=True)
    assert order is None and refused[0] == "child_not_in_store"
    assert a.component_data[FULL].value is not None, "nothing dropped without a copy"


def _plans(worlds, floor=16):
    vecs = []
    for alloc, tree in worlds:
        sv = KC.survey(tree, alloc, P, floor, store_ok=True)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, floor, NP, [floor])))
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    return [dsv._kv_compact_plans_from_vote(sv, group, floor, NP, [floor])[0][0] for sv, _v in vecs]


def test_rf3_other_mixed_content_in_the_same_slot_refuses_alike(monkeypatch):
    monkeypatch.setattr(KC, "BITMAP", 1)                   # every key collides in slot 0
    w_a = _chain(pages0=17)[:2]
    w_b = _chain(pages0=21)[:2]                            # other tokens: other content key, other profile
    assert _plans([w_a, w_b, w_b]) == [None, None, None]


def test_rf3_two_mixed_segments_in_one_slot_on_one_rank_refuse_alike(monkeypatch):
    monkeypatch.setattr(KC, "BITMAP", 1)

    def two():
        alloc = _alloc(NP, P)
        tree = Tree(alloc)
        _take(alloc, list(range(1, 17)))
        for p0 in (17, 21):
            a = _store(_node(tree, alloc, [p0, p0 + 1], P))
            b = _node(tree, alloc, [p0 + 2, p0 + 3], P, parent=a)
            b.l3_present = True
        return alloc, tree

    assert _plans([two(), _chain(pages0=17)[:2], _chain(pages0=17)[:2]]) == [None, None, None]


def test_rf4_a_mixed_segment_behind_an_unbacked_child_is_child_blocked():
    alloc, tree, a, b = _chain()
    _node(tree, alloc, [25], P, parent=b, backed=False)    # the fresh un-backed tail (rc32 under load)
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    cen = KC.holder_census(sv, 16)
    _below, blocked = KC.below_census(sv, 16)
    assert cen.get("store_child_blocked") == 2 and cen.get("l2_child_blocked") == 2, cen
    assert blocked.get("above:unbacked") == 4, blocked
