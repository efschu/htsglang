"""W65-REGROUP (27B INT8 decode matrix, boot 113226 on 173161c595, D 11:47:15,
rid weg2-22-115): the intake prefetch of one warm stream died with W65
Weg2PrefetchSpanSplit (min=2958 max=3448) because the ranks had matched
DIFFERENT prefix depths of the same prompt -- TP0/TP2 at the 99776 anchor, TP1
one 490-token node deeper at 100266 (``FA PREFETCH-FROM-ANCHOR`` x3, ``#1042
EXTENT 2898/3388/2898``); match end 103225 on every rank.

Driven through the REAL ``Scheduler._prefetch_kvcache`` and the REAL
``UnifiedRadixCache.prefetch_from_storage`` on three simulated ranks joined by
a mock gloo group (the H99 harness). Only the radix match is simulated: each
rank's request carries the anchors its tree holds and answers
``init_next_round_input`` with the deepest anchor at or below the
``_weg2_prefix_cap`` the scheduler sets (the real #1419 hook).

RED on 173161c595: W65 on every rank. GREEN with the regroup: nothing is
registered on the split pass, every rank re-matches capped at the group's
start and registers the same range; a split that survives the cap, an unarmed
call and Form A keep the stop.
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

from sglang.srt.managers import tp_match_floor as tmf  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache import match_refusal_census as mrc  # noqa: E402
from sglang.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    HiCacheCollectiveDesyncError,
    UnifiedRadixCache,
)

WORLD = h99.WORLD
N = 103231
MATCH_END = 103225
DEVICE = 96878
# the host chain's anchors, 490 apart, as the D log's MAMBA-HOST-RESUME lines
CHAIN = [97326, 97816, 98306, 98796, 99286, 99776, 100266]
TP02 = [d for d in CHAIN if d <= 99776]  # TP0 / TP2
TP1 = list(CHAIN)  # TP1: one node deeper


def _node(depth):
    return types.SimpleNamespace(
        key=None, backuped=True, parent=None, depth=depth,
        get_last_hash_value=lambda: f"h{depth}",
        get_prefix_hash_values=lambda parent: None,
    )


class _Req:
    """One rank's request: its match is the deepest anchor <= the cap."""

    rid = "rid-w65"

    def __init__(self, anchors, rid="rid-w65"):
        self.rid = rid
        self.anchors = list(anchors)
        self.full_untruncated_fill_ids = list(range(N))
        self.matches = []  # (cap, matched) of every re-match
        self.init_next_round_input(None)

    def init_next_round_input(self, tree_cache, cow_mamba=False):
        cap = getattr(self, "_weg2_prefix_cap", None)
        ok = [a for a in self.anchors if cap is None or a <= int(cap)]
        m = max(ok) if ok else DEVICE
        self.matches.append((cap, m))
        self.prefix_indices = list(range(DEVICE))
        self.host_hit_length = m - DEVICE
        self.state_anchor_depth = m
        self.key_match_depth = m
        self.last_host_node = _node(m)

    def _compute_max_prefix_len(self, n):
        return MATCH_END


class _Sched:
    """The real `_prefetch_kvcache` on a TP rank of the 27B D group."""

    _prefetch_kvcache = Scheduler._prefetch_kvcache
    # absent on 173161c595: the tests then fail on W65 (red), not at import
    _weg2_prefetch_span_regroup = getattr(Scheduler, "_weg2_prefetch_span_regroup", None)
    _note_prefetch_unregistered = Scheduler._note_prefetch_unregistered

    def __init__(self, tree):
        self.enable_hicache_storage = True
        self.page_size = 1
        self.tree_cache = tree


_REGROUP_ATTR = getattr(tmf, "SPAN_REGROUP_ATTR", "_weg2_span_regroup")
_ARMED_ATTR = getattr(tmf, "SPAN_REGROUP_ARMED_ATTR", "_weg2_span_regroup_armed")


def _tree(rank, group):
    c = h99._carrier(rank, group)
    c.root_node = object()
    c.hicache_storage_pass_prefix_keys = False
    c.prefetch_participation_is_collective = lambda: True
    return c


class _Env:
    """A classic 27B D form: the #580 vote in force (asymmetric host tier), no
    Form A plan."""

    def __init__(self, follow=False):
        self.follow = follow
        self.stack = []

    def __enter__(self):
        for p in (
            mock.patch("sglang.srt.runtime_context.get_server_args", return_value=types.SimpleNamespace(rank_tp_ratio=None)),
            mock.patch.object(urc, "uneven_dcp_active", return_value=True),
            mock.patch.object(tmf, "form_a_follow_active", return_value=self.follow),
        ):
            p.__enter__()
            self.stack.append(p)
        return self

    def __exit__(self, *a):
        for p in reversed(self.stack):
            p.__exit__(*a)


def _intake(anchors_by_rank, rid="rid-w65", call=None, follow=False):
    """One intake on every rank; returns (trees, reqs, results, errors, group)."""
    group = h99.MockGlooGroup()
    trees, reqs = {}, {}

    def _rank(r):
        trees[r] = _tree(r, group)
        reqs[r] = _Req(anchors_by_rank[r], rid=rid)
        s = _Sched(trees[r])
        verdict = call(s, reqs[r]) if call else s._prefetch_kvcache(reqs[r])
        return verdict, h99._registered_len(trees[r], rid)

    with _Env(follow=follow):
        results, errors = h99.run_ranks(_rank)
    return trees, reqs, results, errors, group


RC12Z30 = {0: TP02, 1: TP1, 2: TP02}


class TestRc12z30SplitIsRegrouped(unittest.TestCase):
    def test_no_w65_and_one_registered_range(self):
        _t, _r, results, errors, group = _intake(RC12Z30)
        self.assertEqual(errors, {}, f"W65 on the 11:47:15 shape: {errors}")
        # span = MATCH_END - 99776 - 0 (is_eagle False): the SHALLOWEST start, on every rank
        want = MATCH_END - 99776
        self.assertEqual({r: v[1] for r, v in results.items()}, {r: (want, want) for r in range(3)})
        self.assertEqual(group.errors, [])

    def test_every_rank_ran_the_same_collectives(self):
        _t, _r, _res, errors, group = _intake(RC12Z30)
        self.assertEqual(errors, {})
        logs = [group.log[r] for r in range(WORLD)]
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[1], logs[2])
        # the split pass voted once, the capped pass voted once, then the registered
        # prefetch's own completion reduces would follow (not run here)
        self.assertEqual([x[0] for x in logs[0]].count("prefetch_participation_vote"), 2)

    def test_the_split_pass_registers_nothing_and_returns_its_rows(self):
        trees, _r, _res, errors, _g = _intake(RC12Z30)
        self.assertEqual(errors, {})
        # TP0/TP2 asked 3448 rows twice (split pass released them), TP1 2958 then 3448
        self.assertEqual(trees[0].cache_controller.released, MATCH_END - 99776)
        self.assertEqual(trees[2].cache_controller.released, MATCH_END - 99776)
        self.assertEqual(trees[1].cache_controller.released, MATCH_END - 100266)
        for t in trees.values():
            self.assertEqual(t.cache_controller.prefetch_tokens_occupied, MATCH_END - 99776)
            self.assertEqual(getattr(t, _REGROUP_ATTR, {}), {}, "the mark is consumed")
            self.assertFalse(getattr(t, _ARMED_ATTR, False), "the arm is cleared")

    def test_the_deeper_rank_rematched_capped_then_uncapped(self):
        _t, reqs, _res, errors, _g = _intake(RC12Z30)
        self.assertEqual(errors, {})
        # constructor match, intake match, capped retry, uncapped restore
        self.assertEqual([c for c, _m in reqs[1].matches], [None, None, 99776, None])
        self.assertEqual([m for _c, m in reqs[1].matches], [100266, 100266, 99776, 100266])
        # the request leaves matched exactly as the unarmed path leaves it
        self.assertEqual(reqs[1].state_anchor_depth, 100266)
        self.assertEqual(reqs[0].state_anchor_depth, 99776)
        for r in reqs.values():
            self.assertFalse(hasattr(r, "_weg2_prefix_cap"), "the cap is the retry's, never kept")

    def test_registered_prefix_stamp_is_the_group_start_on_every_rank(self):
        _t, reqs, _res, errors, _g = _intake(RC12Z30)
        self.assertEqual(errors, {})
        self.assertEqual({r: q._prefetch_registered_prefix_len for r, q in reqs.items()},
                         {0: 99776, 1: 99776, 2: 99776})

    def test_census_names_the_regroup_and_keeps_one_exit_per_call(self):
        before = dict(mrc.PREFETCH_GATE_COUNTS)
        _t, _r, _res, errors, _g = _intake(RC12Z30)
        self.assertEqual(errors, {})
        d = {k: mrc.PREFETCH_GATE_COUNTS.get(k, 0) - before.get(k, 0)
             for k in ("intake", "span_regroup", "vote_negative", "issued")}
        # 3 ranks x (split pass + capped pass) = 6 intakes; 3 split exits, 3 issued
        self.assertEqual(d, {"intake": 6, "span_regroup": 3, "vote_negative": 3, "issued": 3})


class TestAgreementIsUntouched(unittest.TestCase):
    def test_equal_matches_vote_once_and_never_arm_a_cap(self):
        _t, reqs, results, errors, group = _intake({0: TP02, 1: TP02, 2: TP02})
        self.assertEqual(errors, {})
        want = MATCH_END - 99776
        self.assertEqual({r: v[1] for r, v in results.items()}, {r: (want, want) for r in range(3)})
        self.assertEqual([x[0] for x in group.log[0]].count("prefetch_participation_vote"), 1)
        for r in reqs.values():
            self.assertEqual([c for c, _m in r.matches], [None, None], "one intake match, no cap")

    def test_a_shallower_rank_alone_is_a_split_too_and_regroups_at_its_depth(self):
        _t, _r, results, errors, _g = _intake({0: TP1, 1: TP1, 2: TP02})
        self.assertEqual(errors, {})
        want = MATCH_END - 99776
        self.assertEqual({r: v[1] for r, v in results.items()}, {r: (want, want) for r in range(3)})


class TestTheStopSurvivesWhereTheRegroupCannot(unittest.TestCase):
    def test_a_rank_that_cannot_realize_the_group_start_stops_by_name(self):
        # TP1 holds only the deeper anchor: capped at 99776 it falls back to the
        # device prefix, its span differs again -> the second vote is W65.
        _t, reqs, results, errors, _g = _intake({0: TP02, 1: [100266], 2: TP02})
        self.assertEqual(set(errors), {0, 1, 2}, (results, errors))
        self.assertTrue(all(isinstance(e, HiCacheCollectiveDesyncError) and "W65" in str(e)
                            for e in errors.values()), errors)
        # the cap never outlives the retry, even on the stop
        for r in reqs.values():
            self.assertFalse(hasattr(r, "_weg2_prefix_cap"))

    def test_unarmed_rematch_false_keeps_w65(self):
        _t, _r, _res, errors, _g = _intake(
            RC12Z30, call=lambda s, q: s._prefetch_kvcache(q, rematch=False))
        self.assertEqual(set(errors), {0, 1, 2}, errors)
        self.assertTrue(all("W65" in str(e) for e in errors.values()), errors)

    def test_a_told_limit_keeps_w65(self):
        _t, _r, _res, errors, _g = _intake(
            RC12Z30, call=lambda s, q: s._prefetch_kvcache(q, limit_tokens=MATCH_END))
        self.assertEqual(set(errors), {0, 1, 2}, errors)
        self.assertTrue(all("W65" in str(e) for e in errors.values()), errors)

    def test_a_scheduler_that_does_not_arm_keeps_w65(self):
        # Form A follow makes `span_regroup_allowed` False (TestPieces); here the
        # scheduler's own call goes through that predicate: not armed -> the stop.
        with mock.patch.object(tmf, "span_regroup_allowed", return_value=False, create=True):
            _t, _r, _res, errors, _g = _intake(RC12Z30)
        self.assertEqual(set(errors), {0, 1, 2}, errors)
        self.assertTrue(all("W65" in str(e) for e in errors.values()), errors)

    def test_the_tree_alone_without_the_arm_raises(self):
        # the tree's default (every caller but the scheduler's intake): the stop
        group = h99.MockGlooGroup()
        spans = {0: 99776, 1: 100266, 2: 99776}

        def _rank(r):
            c = _tree(r, group)
            toks = list(range(spans[r], MATCH_END))
            c.prefetch_from_storage("rid-x", _node(spans[r]), toks, last_hash=None,
                                    prefix_keys=None, span_base=spans[r], key_base=spans[r])

        with _Env():
            _res, errors = h99.run_ranks(_rank)
        self.assertEqual(set(errors), {0, 1, 2})
        self.assertTrue(all("W65" in str(e) for e in errors.values()), errors)


class TestPieces(unittest.TestCase):
    def test_raw_cap_units(self):
        self.assertEqual(tmf.regroup_raw_cap(99776, False), 99776)
        self.assertEqual(tmf.regroup_raw_cap(99776, True), 99777)  # N bigrams = N+1 raw tokens

    def test_allowed_only_for_an_intake_style_call(self):
        with mock.patch.object(tmf, "form_a_follow_active", return_value=False):
            ok = dict(rematch=True, limit_tokens=None, group_decides=True)
            self.assertTrue(tmf.span_regroup_allowed(**ok))
            self.assertFalse(tmf.span_regroup_allowed(**{**ok, "rematch": False}))
            self.assertFalse(tmf.span_regroup_allowed(**{**ok, "limit_tokens": 1}))
            self.assertFalse(tmf.span_regroup_allowed(**{**ok, "group_decides": False}))
        with mock.patch.object(tmf, "form_a_follow_active", return_value=True):
            self.assertFalse(tmf.span_regroup_allowed(rematch=True, limit_tokens=None, group_decides=True))

    def test_chain_tail_instrument_prints_the_tail_and_never_raises(self):
        from sglang.srt.managers.scheduler import _weg2_regroup_chain_tail as tail
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        root = types.SimpleNamespace(parent=None, key=None)
        a = types.SimpleNamespace(parent=root, key=list(range(490)), backuped=True,
                                  component_data={ComponentType.MAMBA: types.SimpleNamespace(value=None, host_value=3)})
        b = types.SimpleNamespace(parent=a, key=list(range(490)), backuped=False,
                                  component_data={ComponentType.MAMBA: types.SimpleNamespace(value=1, host_value=None)})
        self.assertEqual(tail(b), "490:0/1/0,490:1/0/1")
        self.assertEqual(tail(None), "-")
        self.assertTrue(tail(types.SimpleNamespace(parent=root, key=[1], backuped=True, component_data=5)).startswith("n/a("))

    def test_take_is_consumed_once(self):
        t = types.SimpleNamespace()
        self.assertIsNone(tmf.take_span_regroup(t, "r"))
        setattr(t, tmf.SPAN_REGROUP_ATTR, {"r": (1, 2, 3, 4)})
        self.assertEqual(tmf.take_span_regroup(t, "r"), (1, 2, 3, 4))
        self.assertIsNone(tmf.take_span_regroup(t, "r"))


if __name__ == "__main__":
    unittest.main()
