"""H97 (rc9m, boot dkrnfbar1rc9m09260642, rid weg2-18-15): the group must
admit a depth EVERY rank can realize.

MEASURED (D log lines 53969-54036, 06:57:28): TP0 voted 18112 (host anchor
there), TP1/TP2 15552; group MIN 15552. H96 cut TP0's key to 15552 -- TP0
holds NO recurrent anchor at 15552 (its host-row release differs from
TP1/TP2's, so the ranks' anchors sit at different depths, not in a superset):
``H96 RU FLOOR CAP-MISS ... capped_match=0``, D dead.

Driven through the real pure helpers every rank runs (MAX arm, skew, the
realizability round) and the real ``MambaComponent.finalize_match_result``
at admission. RED on 68d4bca032 (rc2.1i): TP0 dies. GREEN with the fix: the
group plants 0 and every rank re-prefills from 0.
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
RID = "weg2-18-15"
SLOTS = 8


class _Tree:
    """A rank whose recurrent anchors sit at exactly ``anchors``; a match on a
    key of length L reaches the deepest anchor <= min(L, depth)."""

    def __init__(self, depth, anchors):
        self.depth = depth
        self.anchors = sorted(anchors)
        self.root_node = types.SimpleNamespace(name="root")
        self.cache_controller = object()
        self.is_eagle = False
        self.is_chunk_cache = lambda: False
        self.supports_mamba = lambda: True

    def _reach(self, n):
        best = 0
        for a in self.anchors:
            if a <= n:
                best = a
        return best

    def result(self, n):
        data = [None, None, types.SimpleNamespace(value=None, host_value=torch.tensor([3]) if n else None)]
        node = types.SimpleNamespace(name=f"n{n}", component_data=data) if n else self.root_node
        return MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=self.root_node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=n,
        )

    def match_prefix(self, params):
        n = self._reach(min(len(params.key), self.depth))
        if params.cow_mamba:
            return _component(self).finalize_match_result(
                result=self.result(n), params=params, value_chunks=[torch.zeros(1)], best_value_len=1
            )
        return self.result(n)


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


def _req():
    return types.SimpleNamespace(
        rid=RID, origin_input_ids=list(range(20000)), output_ids=[], extra_key=None, mamba_pool_idx=0
    )


def _group(trees):
    """What the fixed scheduler plants: MIN arm, MAX arm, and -- on skew -- the
    realizability MIN, each reduced over the three ranks."""
    from sglang.srt.managers import tp_match_floor as m

    canonical = [RID]
    local = {r: {RID: t.depth} for r, t in trees.items()}
    red = lambda rows: [min(col) for col in zip(*rows)]  # noqa: E731
    g = m.decode_group_usable(
        canonical, red([m.build_usable_match_payload(canonical, local[r], SLOTS) for r in trees])
    )
    gmax = m.decode_group_max(
        canonical, red([m.build_usable_max_payload(canonical, local[r], SLOTS) for r in trees])
    )
    skew = m.skewed_rids(g, gmax)
    if skew:
        flags = red(
            [
                m.build_realize_payload(canonical, skew, local[r], trees[r], {RID: _req()}, SLOTS)
                for r in trees
            ]
        )
        g = m.apply_realize_verdict(g, canonical, skew, flags)
    return g


def _admit(tree, planted):
    setattr(tree, FLOOR_ATTR, planted)
    params = MatchPrefixParams(key=RadixKey(list(range(20000))), cow_mamba=True, req=_req())
    out = _component(tree).finalize_match_result(
        result=tree.result(tree.depth), params=params, value_chunks=[torch.zeros(1)], best_value_len=1
    )
    return len(out.device_indices) + int(out.host_hit_length)


class TestRc9mSkewNeverKills(unittest.TestCase):
    def test_rc9m_every_rank_admits_the_same_depth(self):
        # TP0: anchors at 2560 and 18112, none at 15552; TP1/TP2 at 15552.
        trees = {0: _Tree(18112, [2560, 18112]), 1: _Tree(15552, [2560, 15552]), 2: _Tree(15552, [2560, 15552])}
        planted = _group(trees)
        self.assertEqual(planted, {RID: 0}, "no common anchor at 15552 -> the group re-prefills")
        geometry = {r: _admit(t, planted) for r, t in trees.items()}
        self.assertEqual(set(geometry.values()), {0}, f"rc9m: {geometry}")

    def test_superset_case_keeps_the_group_depth(self):
        # rc9l shape: TP0 also holds the anchor at the group depth -> H96 cap.
        trees = {0: _Tree(19712, [16384, 19712]), 1: _Tree(16384, [16384]), 2: _Tree(16384, [16384])}
        planted = _group(trees)
        self.assertEqual(planted, {RID: 16384})
        self.assertEqual({_admit(t, planted) for t in trees.values()}, {16384})

    def test_no_skew_no_second_round(self):
        from sglang.srt.managers import tp_match_floor as m

        self.assertEqual(m.skewed_rids({RID: 16384}, {RID: 16384}), {})
        self.assertEqual(m.skewed_rids({RID: 0}, {RID: 19712}), {})  # zero verdict owns it

    def test_max_arm_is_min_neutral_for_absent(self):
        from sglang.srt.managers import tp_match_floor as m

        p = m.build_usable_max_payload([RID, "other"], {RID: 100}, 4)
        self.assertEqual(p[0], -100)
        self.assertTrue(all(v > 0 for v in p[1:]))

    def test_scheduler_runs_the_round_only_on_skew(self):
        # Source pin: the second collective sits behind the skew test, so a
        # pass without skew keeps the old single-reduce shape on every rank.
        import inspect

        from sglang.srt.managers import scheduler as s

        src = inspect.getsource(s)
        i = src.index("_usable_skew = tp_match_floor.skewed_rids(")
        j = src.index("if _usable_skew:", i)
        k = src.index("torch.distributed.all_reduce(", j)
        self.assertLess(j, k)
        self.assertIn("build_usable_max_payload", src)


if __name__ == "__main__":
    unittest.main()
