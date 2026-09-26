"""H96 (rc9l, boot dkrnfbar1rc9l09260540, rid weg2-21-21): the ABOVE-GROUP band
of the RU usable-match floor must be ACTED ON, not only counted.

MEASURED (D log lines 58506-58519, 05:55:54): TP0 matched 19712 on a
host-backed anchor (``MAMBA-HOST-RESUME ... depth=19712``), TP1/TP2 matched
16384; the MIN-reduced group usable match was 16384. TP0 logged
``RU FLOOR ABOVE-GROUP local_match=19712 group_usable=16384 (n=3): counted,
not acted on`` and then ``#1042 EXTENT ... extent=19712`` + ``#988 LOADBACK
... prefix moved to 19712`` while TP1/TP2 took 16384 -- different extends,
TP0 JIT-built alone, the group stood until the watchdog.

These tests drive the REAL ``MambaComponent.finalize_match_result`` on the
three ranks' match results: every rank must admit the SAME depth. RED on
6cda2aa44c (TP0 admits 19712), GREEN with the fix.
"""

from __future__ import annotations

import types
import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)

FLOOR_ATTR = "_tp_match_floor_group"
RID = "weg2-21-21"
LOCAL = {0: 19712, 1: 16384, 2: 16384}
GROUP = min(LOCAL.values())


def _mamba_data(value=None, host_value=None):
    return types.SimpleNamespace(value=value, host_value=host_value)


class _Tree:
    """A rank's tree: its match reaches ``depth`` on a host-backed anchor, and
    every page-aligned shallower depth down to the group depth carries one too
    (TP0's host coverage is a superset of TP1/TP2's on this form)."""

    def __init__(self, depth: int, anchors_down_to: int = GROUP):
        self.depth = depth
        self.anchors_down_to = anchors_down_to
        self.root_node = types.SimpleNamespace(name="root")
        self.cache_controller = object()
        self.is_chunk_cache = lambda: False
        self.supports_mamba = lambda: True
        self.rematch_keys = []

    def node(self):
        data = [None, None, _mamba_data(host_value=torch.tensor([3]))]
        return types.SimpleNamespace(name="anchor", component_data=data)

    def result(self, n: int) -> MatchResult:
        node = self.node()
        return MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=self.root_node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=n,
        )

    def match_prefix(self, params):
        # The nested match: the key it gets is the cut key. It reaches the cut
        # depth when this rank carries an anchor there, else the next one down
        # (modelled as 0: no anchor between).
        n = min(len(params.key), self.depth)
        self.rematch_keys.append(len(params.key))
        if n < self.anchors_down_to:
            n = 0
        comp = _component(self)
        return comp.finalize_match_result(
            result=self.result(n), params=params, value_chunks=[torch.zeros(1)], best_value_len=1
        )


def _component(tree):
    comp = types.SimpleNamespace(
        component_type=ComponentType.MAMBA,
        mamba_checkpoint_interval=None,
        mamba_ckpt_strict_resume=False,
        cache=tree,
        _stateless_resume_refusals=0,
        _foreign_pool_resume_refusals=0,
    )
    comp.finalize_match_result = types.MethodType(MambaComponent.finalize_match_result, comp)
    return comp


def _admit(rank: int, planted, tree=None):
    tree = tree or _Tree(LOCAL[rank])
    setattr(tree, FLOOR_ATTR, planted)
    req = types.SimpleNamespace(rid=RID, mamba_pool_idx=0)
    key = RadixKey(list(range(LOCAL[rank] + 100)))
    params = MatchPrefixParams(key=key, cow_mamba=False, req=req)
    # cow_mamba=True is the admission form; the nested match must see it too.
    params = MatchPrefixParams(key=key, cow_mamba=True, req=req)
    out = _component(tree).finalize_match_result(
        result=tree.result(LOCAL[rank]), params=params, value_chunks=[torch.zeros(1)], best_value_len=1
    )
    return len(out.device_indices) + int(out.host_hit_length), tree


class TestRc9lAboveGroupSplit(unittest.TestCase):
    def test_every_rank_admits_the_group_depth(self):
        planted = {RID: GROUP}
        geometry = {r: _admit(r, planted)[0] for r in LOCAL}
        self.assertEqual(
            set(geometry.values()),
            {GROUP},
            f"rc9l weg2-21-21: the ranks admitted {geometry} -- TP0 extends from "
            "its deeper host anchor while TP1/TP2 extend from the group depth",
        )

    def test_the_deeper_rank_rematches_at_exactly_the_group_depth(self):
        _, tree = _admit(0, {RID: GROUP})
        self.assertEqual(tree.rematch_keys, [GROUP])

    def test_agreeing_ranks_do_not_rematch(self):
        for r in (1, 2):
            _, tree = _admit(r, {RID: GROUP})
            self.assertEqual(tree.rematch_keys, [])

    def test_no_floor_no_change(self):
        # Outside the plan call nothing is planted: TP0 keeps its deeper resume.
        self.assertEqual(_admit(0, None)[0], LOCAL[0])

    def test_zero_group_still_zeroes(self):
        self.assertEqual(_admit(0, {RID: 0})[0], 0)

    def test_unmaterializable_group_depth_stops_loudly(self):
        from sglang.srt.managers.tp_match_floor import RankFloorCapMiss

        tree = _Tree(LOCAL[0], anchors_down_to=GROUP + 64)
        with self.assertRaises(RankFloorCapMiss) as ctx:
            _admit(0, {RID: GROUP}, tree=tree)
        self.assertIn("CAP-MISS", str(ctx.exception))


class TestCapVerdictPure(unittest.TestCase):
    def test_cap_only_in_the_above_group_band(self):
        from sglang.srt.managers.tp_match_floor import group_floor_cap

        tree = types.SimpleNamespace()
        req = types.SimpleNamespace(rid=RID)

        def res(n):
            return types.SimpleNamespace(device_indices=torch.empty(0), host_hit_length=n)

        setattr(tree, FLOOR_ATTR, {RID: GROUP})
        self.assertEqual(group_floor_cap(tree, req, res(19712)), GROUP)
        self.assertIsNone(group_floor_cap(tree, req, res(GROUP)))
        self.assertIsNone(group_floor_cap(tree, req, res(100)))
        setattr(tree, FLOOR_ATTR, {RID: 0})
        self.assertIsNone(group_floor_cap(tree, req, res(19712)))  # zero verdict owns it
        setattr(tree, FLOOR_ATTR, None)
        self.assertIsNone(group_floor_cap(tree, req, res(19712)))


if __name__ == "__main__":
    unittest.main()
