"""S62 (19.10.) 4a: the absolute told is max(store-anchored, PP0's tree-resumable).

NF rc14 ``..._92dd662831_1019_010742.P.log``, rid pdflip-1-30 (15552 tokens recomputed):

  05:20:14 PP0  PDFLIP TURN-ANCHOR INSERT rid=pdflip-1-28 at=18624            (PP1/PP2: 05:20:15, before their acks)
  05:20:15 PP0  PDFLIP RETAIN-PUBLISH rid=pdflip-1-28 {... 'issued': 0, 'arena_refused': 12 ...}
  05:20:15 PP0  MATCH-CENSUS-DEEP rid=pdflip-1-30 reached=18624 accepted=18624
  05:20:15 PP0  #TW TWIN-REGISTER rid=pdflip-1-30 twin=False verdict=declined:store_absent head=18624 span=485
  05:20:15 PP0  #1416 STORE-TOLD ANCHOR-CLAMP rid=pdflip-1-30 completed=18624 anchored=3072
  05:20:15 PP1  PF TOLD-ACK SENT ... ('pdflip-1-30', 3072, ())        (early read at intake 05:20:11: own 3072)
  05:20:17 PP0  #969 EXTENT ... ('pdflip-1-30', 3072, 19110, 3072, 16038)

The ring double (``_told_ring_pf``) runs the shipped told module on three stages: PP0's registration matches
the head on its device tree, the store anchor clamp answers 3072, every follower's early read recorded 3072 and its
told-limited registration finds the head LOCALLY. Danger directions pinned: a told deeper than the tree resumes
(wrong KV/state), rank disagreement (one plan per stage, PP0's record = the wire told), a follower short of the tree
told (floor / told=0 for every rank, never a split), a hang (complete acks decide at once, no busy wait), and the
store clamp kept where no follower could report a shortfall (no PF stream).
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from flliper.srt.managers import pdflip_store_told as m  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402
from flliper.srt.managers import pdflip_told_fidelity as tf  # noqa: E402

RID = "pdflip-1-30"
HEAD = 18624        # MATCH-CENSUS-DEEP reached=18624 accepted=18624 (PP0), the sibling's turn anchor
STORE = 3072        # #1416 ANCHOR-CLAMP anchored=3072
STALE = 3072        # the followers' early read at intake (before 1-28 inserted 18624)


def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_TOLD") or k in (
            "FLLIPER_PDFLIP_P_TWIN_DEFER", m.ENV_FOLLOWER_EARLY_READ, m.ENV_SETTLE_ACK_NOW,
            "FLLIPER_PDFLIP_DUAL_SHARE", "FLLIPER_PDFLIP_DUAL_LAYOUT",
        ):
            monkeypatch.delenv(k, raising=False)


def _ring(monkeypatch, tree_reach, pf=True, b5=None, **env):
    """``tree_reach[rank]`` = the depth that rank's tree resumes pdflip-1-30 from (``rank_resumable``)."""
    _clean(monkeypatch)
    if pf:
        monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    else:
        monkeypatch.setenv(fb.ENV_FALLBACK, "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_TOLD_ABSOLUTE", "1")
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # the store answers the anchored question with 3072 (the chain of 1-28 never landed)
    monkeypatch.setattr(m, "_anchor_clamp", lambda s, r, told: min(int(told), STORE))

    def resumable(scheduler, req, told):
        return min(int(told), int(tree_reach[int(scheduler.ps.pp_rank)]))

    monkeypatch.setattr(tf, "rank_resumable", resumable)
    if b5 is not None:
        monkeypatch.setattr(fb, "ack_anchors", lambda s, req, rid, own, told: list(b5.get(int(s.ps.pp_rank), ())))
    reads = {r: {RID: 0.1} for r in range(3)}
    ring = R.Ring(m, monkeypatch, {RID: 0}, reads)
    pp0 = ring.stages[0]

    def pp0_read(req, rematch=True, limit_tokens=None):
        req._prefetch_registered_prefix_len = HEAD  # the device head, span beyond it absent in the store
        return "declined:store_absent"

    pp0._prefetch_kvcache = pp0_read
    for s in ring.stages[1:]:
        tree = s.tree_cache

        def fol_read(req, rematch=True, limit_tokens=None, tree=tree):
            if limit_tokens is None:  # the early read at intake, before the sibling's insert
                tree.start_read(req.rid, STALE, 0.0)
                return "issued"
            return "declined:too_short"  # the told-limited registration: nothing beyond the local head

        s._prefetch_kvcache = fol_read
    ring.arrive(RID)
    for s in ring.stages[1:]:
        for r in s.waiting_queue:
            # what this stage's device match holds when the told comes (05:20:15: the insert ran before the ack)
            r.host_hit_length = int(tree_reach[int(s.ps.pp_rank)])
    return ring


def _uniform(ring):
    plans = ring.plans(RID)
    assert all(len(p) == 1 for p in plans), plans
    assert plans[0] == plans[1] == plans[2], plans  # RANK EQUALITY: one plan, every stage
    return plans[0][0][2]


def _admits(ring):
    return [(o.told, getattr(o, fb.WIRE_FALLBACK, None), getattr(o, fb.WIRE_FLOOR, None))
            for o in ring.wire_objs() if type(o).__name__ == "PdFlipStoreAdmit"]


def _told_on_wire(ring):
    return [o.told for o in ring.wire_objs() if type(o).__name__ in ("PdFlipStoreTold", "PdFlipStoreToldTwin")]


def _cap(told):
    return told + 0  # LedgerTree is unigram: the #1419 cap equals told keys


# ---------------------------------------------------------------------------
# the metal case
# ---------------------------------------------------------------------------


def test_rc14_1_30_reproduced_with_the_switch_off(monkeypatch):
    """RED side (= the rc14 code): the told is the store anchor, every rank prefills from 3072."""
    ring = _ring(monkeypatch, {0: HEAD, 1: HEAD, 2: HEAD}, **{m.ENV_TOLD_TREE_ANCHOR: "0"})
    ring.run(12)
    assert _told_on_wire(ring) == [STORE]
    assert _uniform(ring) == _cap(STORE)


def test_rc14_1_30_every_rank_admits_the_tree_head(monkeypatch):
    ring = _ring(monkeypatch, {0: HEAD, 1: HEAD, 2: HEAD})
    ring.run(12)
    assert _told_on_wire(ring) == [HEAD]
    assert _uniform(ring) == _cap(HEAD)
    assert _admits(ring) == [(HEAD, None, None)]           # admitted on the acks: no fallback, no floor
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert all(not s.tree_cache.released for s in ring.stages)
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}         # nobody waited (no arena, no store publish)
    # PP0's own record is the told it published (#1416b): its admission compares head + 0 with it
    assert ring.stages[0]._s62_tree_told_n == 1


def test_pp0_record_is_the_wire_told(monkeypatch):
    """Rank agreement on PP0 itself: the record its admission compares (#1416b) is the tree told, not the store
    anchor -- a record left at 3072 against a wire told of 18624 is a start split on PP0."""
    ring = _ring(monkeypatch, {0: HEAD, 1: HEAD, 2: HEAD})
    ring.run(1)
    assert ring.stages[0].tree_cache._prefetch_completed_tokens[RID] == HEAD


# ---------------------------------------------------------------------------
# danger directions
# ---------------------------------------------------------------------------


def test_never_deeper_than_pp0s_tree_resumes(monkeypatch):
    """Too deep = wrong KV/state: PP0's KV head is 18624 but its recurrent state sits at 12032 -- the told is
    12032 (the tree's resumable depth), never the KV head."""
    reach = {0: 12032, 1: HEAD, 2: HEAD}
    ring = _ring(monkeypatch, reach)
    ring.run(12)
    assert _told_on_wire(ring) == [12032]
    assert _uniform(ring) == 12032
    assert _admits(ring) == [(12032, None, None)]


def test_tree_below_the_store_anchor_keeps_the_store_anchor(monkeypatch):
    ring = _ring(monkeypatch, {0: 1024, 1: HEAD, 2: HEAD})
    ring.run(12)
    assert _told_on_wire(ring) == [STORE]


def test_follower_without_the_anchor_floors_every_rank_at_the_common_depth(monkeypatch):
    """PP2 had no free slot for 1-28's turn anchor: its tree resumes from 3072 only. Its ack says 3072, PP1's
    B5 anchors carry 3072 (its path), PP0's GROUP-FLOOR admits 3072 for EVERY rank -- the rc14 depth, never a
    split, nothing released, no wait."""
    ring = _ring(monkeypatch, {0: HEAD, 1: HEAD, 2: STORE}, b5={1: (STORE,), 2: ()})
    ring.run(16)
    assert _told_on_wire(ring) == [HEAD]
    assert _uniform(ring) == STORE
    assert _admits(ring) == [(STORE, None, 1)]
    assert all(not s.tree_cache.released for s in ring.stages)
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}


def test_follower_without_the_anchor_and_no_common_depth_is_told_zero_everywhere(monkeypatch):
    """No B5 anchors: no common depth -- told=0 for every rank (uniform), decided on the complete acks."""
    ring = _ring(monkeypatch, {0: HEAD, 1: HEAD, 2: STORE}, b5={1: (), 2: ()})
    ring.run(16)
    assert _uniform(ring) == 0
    assert _admits(ring) == [(0, 1, None)]
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}


def test_no_hang_the_verdict_comes_within_the_ack_passes(monkeypatch):
    """Hang direction: the twin is decided on the acks (pp_size passes), never held for a store publish."""
    for reach in ({0: HEAD, 1: HEAD, 2: HEAD}, {0: HEAD, 1: HEAD, 2: STORE}):
        ring = _ring(monkeypatch, reach, b5={1: (STORE,), 2: ()})
        k_told = None
        for k in range(16):
            ring.step()
            if k_told is None and _told_on_wire(ring):
                k_told = k
            if _admits(ring):
                break
        assert k_told is not None and _admits(ring), reach
        assert k - k_told <= 4, (reach, k_told, k)


def test_without_the_pf_stream_the_store_clamp_stays(monkeypatch):
    """No ack stream: a follower short of a deeper told could only refuse by name (the #1416 rank exit), so
    the told stays the store anchor -- every rank admits 3072."""
    ring = _ring(monkeypatch, {0: HEAD, 1: HEAD, 2: HEAD}, pf=False)
    ring.run(12)
    assert _told_on_wire(ring) == [STORE]
    assert _uniform(ring) == _cap(STORE)


# ---------------------------------------------------------------------------
# the unit: _tree_anchored
# ---------------------------------------------------------------------------


class _S:
    def __init__(self, pf=True):
        from types import SimpleNamespace

        self.ps = SimpleNamespace(pp_rank=0, pp_size=3)
        self._pdflip_told_fallback_on = pf


def test_unit_tree_anchored(monkeypatch):
    _clean(monkeypatch)
    req = object()
    monkeypatch.setattr(tf, "rank_resumable", lambda s, r, told: min(int(told), HEAD))
    assert m._tree_anchored(_S(), req, RID, HEAD, STORE) == HEAD
    assert m._tree_anchored(_S(), req, RID, 12000, STORE) == 12000          # never past told
    assert m._tree_anchored(_S(), req, RID, HEAD, HEAD) == HEAD             # clamp is told already
    assert m._tree_anchored(_S(), req, RID, 0, 0) == 0
    assert m._tree_anchored(_S(pf=False), req, RID, HEAD, STORE) == STORE   # no PF stream
    monkeypatch.setenv(tf.ENV, "0")
    assert m._tree_anchored(_S(), req, RID, HEAD, STORE) == STORE           # fidelity off: no tree verdict
    monkeypatch.delenv(tf.ENV)
    monkeypatch.setenv(m.ENV_TOLD_TREE_ANCHOR, "0")
    assert m._tree_anchored(_S(), req, RID, HEAD, STORE) == STORE           # switch off
    monkeypatch.delenv(m.ENV_TOLD_TREE_ANCHOR)
    monkeypatch.setattr(tf, "rank_resumable", lambda s, r, told: None)
    assert m._tree_anchored(_S(), req, RID, HEAD, STORE) == STORE           # no probe
    monkeypatch.setattr(tf, "rank_resumable", lambda s, r, told: 1 // 0)
    assert m._tree_anchored(_S(), req, RID, HEAD, STORE) == STORE           # probe raised
    monkeypatch.setattr(tf, "rank_resumable", lambda s, r, told: 2048)
    assert m._tree_anchored(_S(), req, RID, HEAD, STORE) == STORE           # tree below the store anchor


def test_switch_default_on():
    assert m.told_tree_anchor_on({}) is True
    assert m.told_tree_anchor_on({m.ENV_TOLD_TREE_ANCHOR: "off"}) is False


def test_the_raise_is_pp0s_alone():
    """R11/#1004: only PP0's told path calls the tree raise; the follower side has no told decision (it acks
    its tree's reach and follows the Admit / floor marker)."""
    import inspect

    assert "_tree_anchored" in inspect.getsource(m._twin_pp0_told)
    for fn in (m._follower_admit, m._follower_absorb_impl, m.follower_early_settle, m._follower_register,
               m.admission, m.follower_reach_told):
        assert "_tree_anchored" not in inspect.getsource(fn), fn.__name__
