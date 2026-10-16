"""LANE-ANCHOR-RANK (NF h13 84e787ec96, D TP0 14:35:34, pdflip-24-181).

The LANE-PREEMPT park of six lane-0 requests spilled the arriving lane-1
request's own resume anchor (MAMBA-ARENA DISPLACE why=park_first
victim_depth=148992) to give the lane-0 park chains a slot; D refused the
lane-1 request (#904 MambaComponent absent, X-GATE W31) and P prefilled it
in full. A mamba-arena claim may displace only anchors of a lane at or below
its own, the lowest lane first; the lane rule off = the old order.
"""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs
from flliper.srt.pdflip import mamba_arena_displace as mad


def _anchor(node_id, depth, lane=None, slots=(1,), end=False):
    n = types.SimpleNamespace(id=node_id, pdflip_park_end=end)
    if lane is not None:
        n.pdflip_lane = lane
    return mad.OwnedAnchor(node=n, depth=depth, rid="", slots=slots)


def test_a_lane0_park_never_spills_a_lane1_anchor():
    lane1_end = _anchor(673, 148992, lane=1)
    assert mad.pick_park_victim([lane1_end], chain_ids=set(), claimer_is_end=False,
                                claimer_lane=0) is None
    assert mad.pick_last_resort_victim([lane1_end], chain_ids=set(), claimer_lane=0) is None


def test_lowest_lane_first_then_shallowest():
    lane1_shallow = _anchor(10, 100, lane=1)
    lane0_deep = _anchor(11, 90000)
    got = mad.pick_park_victim([lane1_shallow, lane0_deep], chain_ids=set(),
                               claimer_is_end=False, claimer_lane=1)
    assert got is lane0_deep, "a lane-1 claimer takes the lane-0 anchor before the shallower lane-1 one"
    assert mad.pick_last_resort_victim([lane1_shallow, lane0_deep], chain_ids=set(),
                                       claimer_lane=1) is lane0_deep


def test_rule_off_keeps_the_old_order():
    lane1_shallow = _anchor(10, 100, lane=1)
    lane0_deep = _anchor(11, 90000)
    assert mad.pick_park_victim([lane1_shallow, lane0_deep], chain_ids=set(),
                                claimer_is_end=False) is lane1_shallow
    assert mad.pick_last_resort_victim([lane1_shallow, lane0_deep], chain_ids=set()) is lane1_shallow


def test_lane_filter_and_untagged_is_lane0():
    a, b = _anchor(1, 10), _anchor(2, 20, lane=2)
    assert mad.lane_of_node(a.node) == 0 and mad.lane_of_node(b.node) == 2
    assert mad.lane_filter([a, b], 1) == [a]
    assert mad.lane_filter([a, b], None) == [a, b]


def _tree_stub():
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    t = object.__new__(UnifiedRadixCache)
    return t


def test_finish_stamps_the_chain_with_the_request_lane_and_claims_use_it(monkeypatch):
    t = _tree_stub()
    n1, n2 = types.SimpleNamespace(), types.SimpleNamespace(pdflip_lane=3)
    t._pdflip_chain_nodes_strict = lambda key: [n1, n2]
    req = types.SimpleNamespace(priority=1)
    with envs.FLLIPER_PDFLIP_LANE_ANCHOR_RANK.override(True):
        t._pdflip_stamp_lane(req, None)
        assert n1.pdflip_lane == 1 and n2.pdflip_lane == 3     # never lowered
        assert t._pdflip_claim_lane(n1) == 1
        n1.pdflip_park_lane = 0                              # a lane-0 park's chain
        assert t._pdflip_claim_lane(n1, park=True) == 0      # the park claims with ITS lane
        assert t._pdflip_claim_lane(n1) == 1
    with envs.FLLIPER_PDFLIP_LANE_ANCHOR_RANK.override(False):
        assert t._pdflip_claim_lane(n1, park=True) is None


def test_lane0_finish_stamps_nothing():
    t = _tree_stub()
    called = []
    t._pdflip_chain_nodes_strict = lambda key: called.append(key) or []
    with envs.FLLIPER_PDFLIP_LANE_ANCHOR_RANK.override(True):
        t._pdflip_stamp_lane(types.SimpleNamespace(priority=None), "k")
    assert called == []
