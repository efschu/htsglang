"""KVAGREED-1022: what ``blocked_by ... not_agreed`` in the KV-COMPACT BELOW line means, and what it must mean.

THE METAL (NF rc33 fbb680d6d3, boot ..._1022_060647 D). 06:17:30 TP0 ``BELOW S7 ... store_child_blocked=804
l2_child_blocked=47 ... | blocked_by ... below:not_agreed=851`` -- yet not one page on TP0 was ``*_not_agreed`` in
that HOLDERS / BELOW pair (over the whole boot TP0 printed ``store_not_agreed`` 0 times, TP1/TP2 0 times too). A
``not_agreed`` blocker without a ``_not_agreed`` page is no peer disagreement: ``_blocker`` stopped at the first
secured child that ``Agreement.agreed`` refused, and ``agreed`` refuses every child that is not ``sub_ok`` HERE
(its slot is 0 in this rank's own vote maps, so the group MIN is 0 -- on a single rank as well). The real blocker
sat one or more segments deeper (an un-backed tail, a write in flight, a pin).

THE RULE (instrument only, rank-local, log only; no decision input changes):
* ``not_agreed`` / ``mixed`` name only a child that is secured with its whole device subtree HERE and that the
  group did not agree on (a peer holds it otherwise);
* a secured child whose own subtree is not secured here is walked into, and the class that blocks IT is named.

Danger directions guarded (the fix must not leak into the agreement): "Token-Verlust" -- a secured segment above an
un-backed tail chosen as room; "Ränge uneins" -- one rank planning an eviction its peers refuse.
"""
from __future__ import annotations

import os

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


def _branch_world(tail_backed=False, prefix_store=True, outside=0, third_unbacked=False):
    """A secured shared prefix (pages 11..14) with two secured branches; the first branch (pages 15, 16) carries a
    fresh tail (page 20, above boundary 16), the second (17, 18) is whole. ``tail_backed`` False: the tail is
    un-backed (the rc33 metal: the turn end deferred it). ``third_unbacked``: a third branch (page 22) un-backed on
    this rank -- the prefix is then child-blocked on every rank, whatever the first branch is."""
    alloc = _alloc(NP, P)
    tree = Tree(alloc)
    if outside:
        _take(alloc, list(range(1, 1 + outside)))
    s = _node(tree, alloc, [11, 12, 13, 14], P)
    if prefix_store:
        _store(s)
    c1 = _node(tree, alloc, [15, 16], P, parent=s)
    t = _node(tree, alloc, [20], P, parent=c1, backed=tail_backed)
    c2 = _node(tree, alloc, [17, 18], P, parent=s)
    if third_unbacked:
        _node(tree, alloc, [22], P, parent=s, backed=False)
    return alloc, tree, s, c1, t, c2


def _vote_worlds(worlds, floor=16):
    out = []
    for alloc, tree in worlds:
        sv = KC.survey(tree, alloc, P, floor, store_ok=True)
        out.append((sv, dsv._kv_compact_vote(sv, None, floor, NP, [floor])))
    group = [min(c) for c in zip(*[v for _s, v in out])]
    return [(sv, group) for sv, _v in out]


# ---- the instrument -------------------------------------------------------------------------------------

def test_a_deeper_unbacked_tail_is_named_not_called_not_agreed_single_rank():
    """One rank, no group: the prefix waits behind branch c1, c1 behind its un-backed tail. Both wait for the
    TAIL -- 'not_agreed' is impossible on one rank."""
    alloc, tree, *_ = _branch_world()
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    below, blocked = KC.below_census(sv, 16)
    assert below.get("store_child_blocked") == 4 and below.get("l2_child_blocked") == 2, below
    assert "below:not_agreed" not in blocked, blocked
    assert blocked == {"below:unbacked": 6}, blocked


def test_rc33_three_ranks_without_any_peer_disagreement_print_no_not_agreed():
    """Three identical ranks (the 06:17:30 shape: no ``_not_agreed`` page anywhere): the group-agreed census
    names the un-backed tail on every rank."""
    worlds = [_branch_world()[:2] for _ in range(3)]
    for sv, group in _vote_worlds(worlds):
        ag = dsv._kv_compact_agreement(group, NP, [16])
        cen = KC.holder_census(sv, 16, ag)
        below, blocked = KC.below_census(sv, 16, ag)
        assert not any(k.endswith("_not_agreed") for k in list(cen) + list(below)), (cen, below)
        assert not any(k.endswith(":not_agreed") for k in blocked), blocked
        assert blocked == {"below:unbacked": 6}, blocked


def test_a_true_peer_disagreement_is_still_named_not_agreed():
    """TP0's tail un-backed, the workers' backed (Form A, the 06:23:58 shape: TP0 unbacked 44 below, TP1/TP2
    ``l2_not_agreed=44``). On a WORKER the branch c1 (+ its tail, one segment there) is secured with its subtree
    here and the group refuses it: that IS a peer disagreement -- its pages are ``_not_agreed``, and the prefix
    (child-blocked by a third, un-backed branch on every rank) names c1 ``not_agreed`` first. On TP0 c1 is not
    ``sub_ok`` (its tail is un-backed HERE): the prefix and c1 wait for that tail."""
    tp0 = _branch_world(tail_backed=False, third_unbacked=True)[:2]
    wk = _branch_world(tail_backed=True, third_unbacked=True)[:2]
    (sv0, g), (sv1, _g1), _ = _vote_worlds([tp0, wk, wk])
    ag = dsv._kv_compact_agreement(g, NP, [16])
    _b0, blk0 = KC.below_census(sv0, 16, ag)
    assert blk0 == {"below:unbacked": 6}, blk0
    cen1 = KC.holder_census(sv1, 16, ag)
    below1, blk1 = KC.below_census(sv1, 16, ag)
    assert below1.get("l2_not_agreed") == 2 and cen1.get("l2_not_agreed") == 1, (below1, cen1)
    assert blk1 == {"below:not_agreed": 4}, blk1


def test_a_mixed_child_behind_an_unbacked_tail_names_the_tail():
    """The same for a MIXED child (store prefix + L2 run) whose own subtree is not secured here: the tail is named,
    never 'mixed' (the old label, ``above:mixed=54`` at 06:17:30 with no ``_mixed`` page on TP0)."""
    alloc = _alloc(NP, P)
    tree = Tree(alloc)
    s = _store(_node(tree, alloc, [11, 12], P))
    m = _store(_node(tree, alloc, [13, 14], P, parent=s))      # branch 1: store run ...
    m2 = _node(tree, alloc, [15], P, parent=m)                  # ... + L2 run = one mixed segment
    _node(tree, alloc, [20], P, parent=m2, backed=False)        # its un-backed tail
    _node(tree, alloc, [17, 18], P, parent=s)                   # branch 2, whole
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    assert any(g.kind == 3 for g in sv.segments)
    _below, blocked = KC.below_census(sv, 16)
    assert blocked == {"below:unbacked": 5}, blocked


# ---- the fix touches no decision: the danger directions stay guarded ------------------------------------

def test_guard_token_loss_a_prefix_above_an_unbacked_tail_is_never_room():
    """Danger direction 'Token-Verlust': with the room needed (outside pages fill everything below), the plan may
    take the whole branch c2, never the prefix or c1 (the un-backed tail below them would have to go first)."""
    alloc, tree, s, c1, t, c2 = _branch_world(outside=10)
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan, _why = KC.plan_for(sv, 16)
    taken = set() if plan is None else {id(m.node) for m in plan.evicts + plan.handoff}
    assert id(s) not in taken and id(c1) not in taken and id(t) not in taken, taken


def test_mutant_secured_counts_as_room_behind_an_unbacked_tail_is_a_token_loss(monkeypatch):
    """The mutant the ordered fix ("secured pages count as room") becomes when it skips the subtree: every secured
    segment is room (leaf-first dropped in the choice) and leaves at once (no 'waits' in the split). The guard above
    catches it -- the store prefix is handed to S39 with the un-backed tail still below it (no copy of the tail
    anywhere: token loss). Both halves are needed: the choice alone is still stopped by the split's 'waits'."""
    def loose_choice(s_, ag, gap):
        avail = sorted((g for g in s_.segments if g.evictable and g.kind in (1, 2) and g.key >= 0),
                       key=lambda g: (-g.size, g.key))
        chosen, rest = [], int(gap)
        for g in avail:
            if rest <= 0:
                break
            chosen.append(g)
            rest -= g.size
        return (chosen or None), ""

    def loose_split(segs, ag):
        return [(g, 1 if ag.now(g) else 2) for g in segs]
    monkeypatch.setattr(KC, "choose_evictions", loose_choice)
    monkeypatch.setattr(KC, "split_evictions", loose_split)
    alloc, tree, s, c1, t, c2 = _branch_world(outside=10)
    sv = KC.survey(tree, alloc, P, 16, store_ok=True)
    plan, _why = KC.plan_for(sv, 16)
    taken = set() if plan is None else {id(m.node) for m in plan.evicts + plan.handoff}
    assert id(s) in taken, "the mutant must hand the prefix above the un-backed tail to S39"
    assert id(t) not in taken


def test_guard_ranks_agree_the_same_plan_digest_with_a_peer_disagreement():
    """Danger direction 'Ränge uneins': TP0 un-backed tail, workers backed -- every rank plans the same content
    (or none), one digest; the census change does not enter the plan."""
    tp0 = _branch_world(tail_backed=False, outside=10)[:2]
    wk = _branch_world(tail_backed=True, outside=10)[:2]
    plans = [dsv._kv_compact_plans_from_vote(sv, g, 16, NP, [16])[0][0] for sv, g in _vote_worlds([tp0, wk, wk])]
    got = {(-1 if p is None else KC.plan_digest(5, p)) for p in plans}
    assert len(got) == 1, plans


def test_mutant_rank_local_agreement_parts_the_ranks(monkeypatch):
    """The mutant 'agreed from this rank's own survey' (what the census asks, answered rank-locally inside the
    decision: own sub_ok, own sizes, only the gaps from the group): the workers take branch c1 with its tail,
    TP0 cannot -- the plans part (the plan check would then refuse, every pass, with room on two ranks)."""
    from flliper.srt.pdflip import d_kv_compact as KC_

    def local(group, top_page, boundaries):
        return KC_.Agreement(None, {int(b): -int(group[2 + i]) for i, b in enumerate(boundaries)})
    monkeypatch.setattr(dsv, "_kv_compact_agreement", local)
    tp0 = _branch_world(tail_backed=False, outside=10)[:2]
    wk = _branch_world(tail_backed=True, outside=10)[:2]
    plans = [dsv._kv_compact_plans_from_vote(sv, g, 16, NP, [16])[0][0] for sv, g in _vote_worlds([tp0, wk, wk])]
    got = [(-1 if p is None else KC.plan_digest(5, p)) for p in plans]
    assert got[0] != got[1] and got[1] == got[2], got
