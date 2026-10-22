"""W65-REGROUP-FIXPOINT (27B INT8 round 3, boot 1022_003738 on a4f4b8fec1, D
00:47:28, rid pdflip-4-73): the D group died with W65 PdFlipPrefetchSpanSplit
(min=13632 max=14464) on the regroup's single retry.

MEASURED (D log, one rid, one intake, one second):
  * intake match: TP0 51712, TP1 52416, TP2 53120 (``W65 REGROUP ... local_match``,
    key_match_depth=65024 on all three: the KV path agreed, the mamba anchors did not);
  * REGROUP group_start=51712 (the shallowest start);
  * capped at 51712, TP1 and TP2 had NO anchor there ("ANCHOR-ONLY-READ depth=51712
    anchor_was=50880 how=branch_point_never_anchored") and matched 50880, TP0 51712
    -> spans 13632 / 14464 / 14464 -> the retry was never armed -> W65 on all ranks,
    then W17 GroupDead.

The ranks' anchors differ per node because the mamba arena slot is one blob
shared by the ranks and a spill frees it only at the last releaser (K15-TP,
test_k15_tp_shared_slot_spill_1022.py). The shallowest start is then not
always a common anchor; a split that survives the cap has a strictly
shallower start and the group regroups again there, bounded.

Driven through the REAL ``Scheduler._prefetch_kvcache`` and the REAL
``UnifiedRadixCache.prefetch_from_storage`` on three simulated ranks joined by
a mock gloo group (the H99 harness, via the 06.10. regroup test's pieces).

RED on a4f4b8fec1: W65 on every rank for the live shape. GREEN with the
fixpoint: one registered range at 50880 on every rank.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_nf_form_a_prefetch_span_h99 as h99  # noqa: E402  (the 3-rank gloo harness)
import test_w65_intake_span_regroup_1006 as r1006  # noqa: E402

from flliper.srt.managers import tp_match_floor as tmf  # noqa: E402
from flliper.srt.managers.scheduler import Scheduler  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import HiCacheCollectiveDesyncError  # noqa: E402

WORLD = h99.WORLD
MATCH_END = 65344  # local_start + local_span on every rank (51712 + 13632)
N = MATCH_END + 6
DEVICE = 40960  # "device 0 + host 10752" at anchor 51712: the host chain starts here
# the anchors every rank holds; 50880 is the deepest common one ("anchor_was=50880")
COMMON = [DEVICE + 256 * i for i in range(1, 40) if DEVICE + 256 * i < 50880] + [50880]
# the ranks' anchors around the split, as the D log's REGROUP chain_tail lines
LIVE = {
    0: COMMON + [51712],  # TP0: anchor at 51712 (node 307 kept, FLUSH-SPILL dropped=1)
    1: COMMON + [52416],  # TP1: 307/308 tombstoned, 309 kept
    2: COMMON + [53120],  # TP2: 307..313 without anchor, 314 kept
}


def _node(depth):
    return types.SimpleNamespace(
        key=None, backuped=True, parent=None, depth=depth,
        get_last_hash_value=lambda: f"h{depth}",
        get_prefix_hash_values=lambda parent: None,
    )


class _Req:
    """One rank's request: its match is the deepest anchor <= the cap."""

    def __init__(self, anchors, rid="pdflip-4-73", match_end=MATCH_END):
        self.rid = rid
        self.anchors = list(anchors)
        self.match_end = match_end
        self.full_untruncated_fill_ids = list(range(match_end + 6))
        self.matches = []
        self.init_next_round_input(None)

    def init_next_round_input(self, tree_cache, cow_mamba=False):
        cap = getattr(self, "_pdflip_prefix_cap", None)
        ok = [a for a in self.anchors if cap is None or a <= int(cap)]
        m = max(ok) if ok else DEVICE
        self.matches.append((cap, m))
        self.prefix_indices = list(range(DEVICE))
        self.host_hit_length = m - DEVICE
        self.state_anchor_depth = m
        self.key_match_depth = self.match_end
        self.last_host_node = _node(m)

    def _compute_max_prefix_len(self, n):
        return self.match_end


def _intake(anchors_by_rank, rid="pdflip-4-73", match_end=MATCH_END):
    group = h99.MockGlooGroup()
    trees, reqs = {}, {}

    def _rank(r):
        trees[r] = r1006._tree(r, group)
        reqs[r] = _Req(anchors_by_rank[r], rid=rid, match_end=match_end)
        s = r1006._Sched(trees[r])
        verdict = s._prefetch_kvcache(reqs[r])
        return verdict, h99._registered_len(trees[r], rid)

    with r1006._Env():
        results, errors = h99.run_ranks(_rank)
    return trees, reqs, results, errors, group


class TestLiveShapeConverges(unittest.TestCase):
    def test_no_w65_one_range_at_the_deepest_common_anchor(self):
        _t, reqs, results, errors, group = _intake(LIVE)
        self.assertEqual(errors, {}, f"W65 on the 00:47:28 shape: {errors}")
        want = MATCH_END - 50880  # 14464, the span the second vote saw on TP1/TP2
        self.assertEqual({r: v[1] for r, v in results.items()}, {r: (want, want) for r in range(3)})
        self.assertEqual({r: q._prefetch_registered_prefix_len for r, q in reqs.items()},
                         {0: 50880, 1: 50880, 2: 50880})
        self.assertEqual(group.errors, [])

    def test_every_rank_ran_three_votes_and_the_same_collectives(self):
        _t, _r, _res, errors, group = _intake(LIVE)
        self.assertEqual(errors, {})
        logs = [group.log[r] for r in range(WORLD)]
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[1], logs[2])
        # intake split (51712), capped split (50880), capped agreement
        self.assertEqual([x[0] for x in logs[0]].count("prefetch_participation_vote"), 3)

    def test_caps_walk_down_and_the_request_leaves_uncapped(self):
        _t, reqs, _res, errors, _g = _intake(LIVE)
        self.assertEqual(errors, {})
        # TP2: constructor, intake, cap 51712 (-> 50880), cap 50880, ONE uncapped restore by the
        # outermost round (an inner round never re-matches under the outer cap)
        caps = [c for c, _m in reqs[2].matches]
        self.assertEqual(caps, [None, None, 51712, 50880, None])
        for r, q in reqs.items():
            self.assertFalse(hasattr(q, "_pdflip_prefix_cap"), "the cap is the retry's, never kept")
            self.assertEqual(q.state_anchor_depth, max(LIVE[r]), "left matched as the unarmed path leaves it")

    def test_marks_and_arm_are_consumed(self):
        trees, _r, _res, errors, _g = _intake(LIVE)
        self.assertEqual(errors, {})
        for t in trees.values():
            self.assertEqual(getattr(t, tmf.SPAN_REGROUP_ATTR, {}), {})
            self.assertFalse(getattr(t, tmf.SPAN_REGROUP_ARMED_ATTR, False))


class TestTheBoundKeepsTheStop(unittest.TestCase):
    def _staircase(self, steps):
        # each round finds the min rank's anchor missing on the other rank one step lower
        a = [DEVICE + 64 * (2 * i) for i in range(1, steps + 1)]
        b = [DEVICE + 64 * (2 * i + 1) for i in range(1, steps + 1)]
        end = max(a + b) + 640
        return {0: a, 1: b, 2: a}, end

    def test_a_walk_longer_than_the_bound_is_w65(self):
        shape, end = self._staircase(tmf.W65_REGROUP_MAX_ROUNDS + 4)
        _t, reqs, _res, errors, group = _intake(shape, match_end=end)
        self.assertEqual(set(errors), {0, 1, 2}, errors)
        self.assertTrue(all(isinstance(e, HiCacheCollectiveDesyncError) and "W65" in str(e)
                            for e in errors.values()), errors)
        votes = [x[0] for x in group.log[0]].count("prefetch_participation_vote")
        self.assertEqual(votes, tmf.W65_REGROUP_MAX_ROUNDS + 1)
        for q in reqs.values():
            self.assertFalse(hasattr(q, "_pdflip_prefix_cap"))

    def test_a_walk_inside_the_bound_converges(self):
        shape, end = self._staircase(3)
        _t, _r, results, errors, _g = _intake(shape, match_end=end)
        self.assertEqual(errors, {}, errors)
        spans = {v[1] for v in results.values()}
        self.assertEqual(len(spans), 1, results)

    def test_no_progress_is_a_named_stop(self):
        # a capped re-match that matches DEEPER than its cap (a broken cap hook)
        class _Liar(_Req):
            def init_next_round_input(self, tree_cache, cow_mamba=False):
                cap = getattr(self, "_pdflip_prefix_cap", None)
                super().init_next_round_input(tree_cache, cow_mamba)
                if cap is not None:
                    m = max(self.anchors)
                    self.host_hit_length = m - DEVICE
                    self.state_anchor_depth = m
                    self.last_host_node = _node(m)

        group = h99.MockGlooGroup()

        def _rank(r):
            t = r1006._tree(r, group)
            q = (_Liar if r == 2 else _Req)(LIVE[r])
            return r1006._Sched(t)._prefetch_kvcache(q)

        with r1006._Env():
            _res, errors = h99.run_ranks(_rank)
        self.assertEqual(set(errors), {0, 1, 2}, errors)
        self.assertTrue(all("W65" in str(e) for e in errors.values()), errors)
        self.assertTrue(any("REGROUP-NO-PROGRESS" in str(e) for e in errors.values()), errors)


class TestPieces(unittest.TestCase):
    def test_bound_is_a_small_positive_int(self):
        self.assertIsInstance(tmf.W65_REGROUP_MAX_ROUNDS, int)
        self.assertGreaterEqual(tmf.W65_REGROUP_MAX_ROUNDS, 2, "22.10. needed two rounds")

    def test_scheduler_signature_carries_the_round(self):
        import inspect

        p = inspect.signature(Scheduler._prefetch_kvcache).parameters
        self.assertIn("_regroup_prev_start", p)


if __name__ == "__main__":
    unittest.main()
