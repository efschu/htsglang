"""KVLEITER-1022 F2 + I1: a MIXED segment (store-backed prefix, L2-backed tail in one chain) is room when its
profile is the same on every rank; the refusal names what lies below the boundary and which child blocks.

THE METAL (NF rc32 ded00bd5a7, boot ..._1022_030836 D). A wake loads a conversation's prefix from the store
(``PDFLIP-LOAD-DEVICE ... matched=0 loaded=133312``: ``l3_present``, no host copy); the next turn end backs the tail
into L2 (S108). KVPLAN-1021 cut chains only where SECURED changes, so prefix and tail are ONE segment of mixed kind,
and mixed was never room: 03:27:32 ``HOLDERS S5 ... store_mixed=555 l2_mixed=123``; idle (used=0) 03:37:03
``HOLDERS S0 ... l2_mixed=2 store_mixed=5`` held S1, 03:37:59 ``store_child_blocked=98`` behind 47 mixed pages.

THE RULE. A mixed segment's PROFILE (content key, the kind of every run, the tokens of every run but the bottom
one) rides in the existing vote as MIN and as negated MIN (= MAX) -- no new collective. Agreed = MIN == MAX == its
own profile: every token depth leaves by the same path on every rank. Then run by run, bottom-up: L2 runs demoted
now (``_evict_to_host``: the host copy stays), store runs handed to the S39 drop (TP0 decides at its broadcast),
a run above a waiting run waits for a later pass. Never a store node through ``_evict_to_host`` (no host copy:
token loss); never a plan that one rank holds and another does not (ranks uneins).
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402,F401

from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

from test_s77_kv_compact_s9_hold_1019 import FULL, Tree, _alloc, _node, _take  # noqa: E402

P, NP = 4, 32


def _store(n):
    n.component_data[FULL].host_value = None
    n.l3_present = True
    return n


def _world():
    alloc = _alloc(NP, P)
    return alloc, Tree(alloc)


def _rc32_chain(cut=(2, 2)):
    """A store-loaded prefix (pages 17.., ``cut[0]`` pages) and its L2 tail (``cut[1]`` pages), nothing free below
    boundary 16. ``cut`` changes only where the store/L2 boundary lies (same tokens)."""
    alloc, tree = _world()
    _take(alloc, list(range(1, 17)))
    pre = list(range(17, 17 + cut[0]))
    tail = list(range(17 + cut[0], 17 + cut[0] + cut[1]))
    a = _store(_node(tree, alloc, pre, P))
    b = _node(tree, alloc, tail, P, parent=a)
    return alloc, tree, a, b


def _no_store_demoted(plan):
    bad = [m for m in plan.evicts if m.node.component_data[FULL].host_value is None]
    assert not bad, "a node without a host copy would go through _evict_to_host: token loss"


def test_rc32_a_store_prefix_with_an_l2_tail_is_room_tail_now_prefix_to_s39():
    alloc, tree, a, b = _rc32_chain()
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    (g,) = [x for x in sv.segments if x.kind == 3]
    assert [r[0] for r in g.runs] == [2, 1] and g.profile > 0
    plan, why = KC.plan_for(sv, 16)
    assert plan is not None, why
    assert [m.node for m in plan.evicts] == [b] and [m.node for m in plan.handoff] == [a]
    assert plan.staged and not plan.waits and not plan.pairs
    _no_store_demoted(plan)


def test_l2_above_a_store_tail_hands_off_the_tail_and_the_l2_head_waits():
    """Leaf first: the store tail leaves at TP0's broadcast, the L2 head is no device leaf before that -- it waits
    (the next pass sees it as a pure L2 leaf)."""
    alloc, tree = _world()
    _take(alloc, list(range(1, 17)))
    a = _node(tree, alloc, [17, 18], P)
    b = _store(_node(tree, alloc, [19, 20], P, parent=a))
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan, why = KC.plan_for(sv, 16)
    assert plan is not None, why
    assert plan.evicts == [] and [m.node for m in plan.handoff] == [b] and plan.waits


def test_census_counts_an_agreeable_mixed_segment_as_ok():
    alloc, tree, a, b = _rc32_chain()
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    cen = KC.holder_census(sv, 16)
    assert cen.get("store_ok") == 2 and cen.get("l2_ok") == 2 and "store_mixed" not in cen, cen


# ---- the group: one vote, MIN and MAX of the profile ----------------------------------------------

def _plans(worlds, floor=16, mutate=None):
    vecs = []
    for alloc, tree in worlds:
        sv = KC.survey(tree, alloc, P, floor, store_ok=True)
        vecs.append((sv, dsv._kv_compact_vote(sv, None, floor, NP, [floor])))
    assert len({len(v) for _s, v in vecs}) == 1
    group = [min(c) for c in zip(*[v for _s, v in vecs])]
    return [dsv._kv_compact_plans_from_vote(sv, group, floor, NP, [floor])[0][0] for sv, _v in vecs]


def test_three_ranks_with_one_profile_take_the_same_plan():
    worlds = [_rc32_chain()[:2] for _ in range(3)]
    plans = _plans(worlds)
    assert all(p is not None for p in plans), plans
    assert len({KC.plan_digest(5, p) for p in plans}) == 1
    for p in plans:
        assert len(p.evicts) == 1 and len(p.handoff) == 1
        _no_store_demoted(p)


def test_ranks_whose_store_l2_boundary_differs_refuse_alike():
    """TP0's store/L2 boundary 2 pages down, the workers' 3: an L2 page on TP0 is store-only on the workers. No
    rank may take it -- every rank answers 'no plan' alike."""
    plans = _plans([_rc32_chain((2, 2))[:2], _rc32_chain((3, 1))[:2], _rc32_chain((3, 1))[:2]])
    assert plans == [None, None, None], plans


def test_mutant_profile_min_only_parts_the_ranks(monkeypatch):
    """Danger direction 'Ränge uneins': the agreement reads only the MIN of the profile -- the rank(s) holding the
    smaller profile take the mixed segment, the others do not: the plans differ between TP0 and the workers."""
    real = KC.Agreement.agreed

    def min_only(self, seg):
        if seg.kind == 3 and self.mixed_lo is not None:
            return int(self.mixed_lo[seg.key % KC.BITMAP]) == int(seg.profile) and seg.profile > 0
        return real(self, seg)
    monkeypatch.setattr(KC.Agreement, "agreed", min_only)
    plans = _plans([_rc32_chain((2, 2))[:2], _rc32_chain((3, 1))[:2], _rc32_chain((3, 1))[:2]])
    got = [p is not None for p in plans]
    assert got[0] != got[1] and got[1] == got[2], got


def test_a_deeper_leaf_on_one_rank_keeps_the_profile():
    """Form A: TP0's leaf one page deeper, same kind -- only the bottom run is longer; the profile is the same."""
    a = _rc32_chain((2, 2))
    b = _rc32_chain((2, 3))
    pa = [g.profile for g in KC.survey(a[1], a[0], P, 16, store_ok=True).segments if g.kind == 3]
    pb = [g.profile for g in KC.survey(b[1], b[0], P, 16, store_ok=True).segments if g.kind == 3]
    assert pa == pb and pa[0] > 0


def test_mutant_store_run_demoted_now_is_a_token_loss(monkeypatch):
    """Danger direction 'Token-Verlust': every run of a mixed segment taken 'now' -- the store prefix (no host
    copy) would go through _evict_to_host."""
    monkeypatch.setattr(KC, "mixed_run_states", lambda g, kids: [1] * len(g.runs))
    alloc, tree, a, b = _rc32_chain()
    plan, _w = KC.plan_for(KC.survey(tree, alloc, P, 16, store_ok=True), 16)
    bad = [m for m in plan.evicts if m.node.component_data[FULL].host_value is None]
    assert bad, "the mutant must demote the store prefix"
