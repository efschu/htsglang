# SPDX-License-Identifier: Apache-2.0
"""Fix A (27.09.2026): the P pass budget is sized from rank-identical data only.

Boot dkr27breleasedraftbar1w109270737 (2d680cbe66), P group death 08:14:59Z: PP1 refused
``#1233 W27 PP WIDTH DIVERGENCE: 512 row(s) for a batch of 1024 token(s)``. Both ranks admitted
weg2-50-185 in pass fwd_ct 563; the budget came from the queue head weg2-50-184 (a released TW twin,
read still running, admissible nowhere), whose ``len(prefix_indices)`` was 21246 on PP0 (the twin's
registration match) and 0 on the followers -> PP0 replanned to 512, PP1 took the plan's 1024.

Two scheduler stubs (PP0 with the registered head, a follower without) must size the SAME budget.
The mutant (the pre-fix position) must make them disagree -- that is what pins the test.
"""

import logging
import os
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import p_budget_head as PBH  # noqa: E402
from sglang.srt.weg2 import p_chunk_policy as P  # noqa: E402

END = 29181   # weg2-50-184's fill
HEAD = 21246  # its TW registration match on PP0


class ReplanLikeThePlanner:
    """next_width as the specimen behaved: the plan at 0 opens with 1024; a
    position off the plan replans without ramp and opens with 512."""

    def __init__(self):
        self.calls = []

    def next_width(self, key, pos, end, queued=False):
        self.calls.append((key, pos, end, queued))
        return 1024 if pos == 0 else 512

    class spec:  # forward_budget reads spec.limits.fixed_tokens on a finishing width
        class limits:
            fixed_tokens = 512


def _req(rid, end, prefix):
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(end)), origin_input_ids=[],
                           output_ids=[], prefix_indices=list(range(prefix)))


def _sched(pp_rank, head_prefix, told_map=None, chunked=None, planner=None, static=2048):
    from sglang.srt.managers.scheduler import Scheduler

    class _S:
        _p_chunk_policy_width = Scheduler._p_chunk_policy_width
        _p_layer_split_width = Scheduler._p_layer_split_width

    s = _S()
    s.pp_rank = pp_rank
    s.chunked_prefill_size = static
    s.chunked_req = chunked
    twin = _req("weg2-50-184-0000", END, head_prefix)
    nxt = _req("weg2-50-185-0000", 110413, 0)
    s.waiting_queue = [twin, nxt]
    s._p_chunk_planner = planner if planner is not None else ReplanLikeThePlanner()
    if told_map is not None:
        s._weg2_store_told = told_map
    return s


class TheQueuedHeadSizesTheSameBudgetOnEveryRank(CustomTestCase):
    def test_pp0_with_the_twin_registration_and_a_follower_without_agree(self):
        pp0, pp1 = _sched(0, HEAD), _sched(1, 0)
        w0, w1 = pp0._p_chunk_policy_width(), pp1._p_chunk_policy_width()
        self.assertEqual(w0, w1, "PP0 and the follower sized different budgets for the same pass")
        self.assertEqual(w0, 1024)
        self.assertEqual(pp0._p_chunk_planner.calls[0][1], 0)  # position 0, not the local 21246
        self.assertTrue(pp0._p_chunk_planner.calls[0][3])      # flagged as a queued head

    def test_the_mutant_pre_fix_position_disagrees(self):
        """The pre-fix rule (len(prefix_indices) of the queued head) on the same
        two stubs gives 512 vs 1024 -- the specimen. If this ever passes as equal,
        the fixture no longer discriminates and the test above proves nothing."""
        orig = PBH.budget_head

        def pre_fix(scheduler):
            h = orig(scheduler)
            if h is None or h.src == "chunked":
                return h
            return PBH.BudgetHead(h.req, h.local_prefix, h.end, "mutant", h.local_prefix)

        PBH.budget_head = pre_fix
        try:
            w0, w1 = _sched(0, HEAD)._p_chunk_policy_width(), _sched(1, 0)._p_chunk_policy_width()
        finally:
            PBH.budget_head = orig
        self.assertEqual((w0, w1), (512, 1024))

    def test_a_published_told_is_the_position_on_every_rank(self):
        told = {"weg2-50-184-0000": 23678}
        pp0, pp1 = _sched(0, HEAD, dict(told)), _sched(1, 0, dict(told))
        self.assertEqual(pp0._p_chunk_policy_width(), pp1._p_chunk_policy_width())
        self.assertEqual(pp0._p_chunk_planner.calls[0][1], 23678)
        self.assertEqual(pp1._p_chunk_planner.calls[0][1], 23678)

    def test_a_told_beyond_the_fill_is_clamped(self):
        s = _sched(0, 0, {"weg2-50-184-0000": END + 999})
        s._p_chunk_policy_width()
        self.assertEqual(s._p_chunk_planner.calls[0][1], END - 1)

    def test_the_chunked_request_keeps_its_own_prefix(self):
        chunked = _req("weg2-50-185-0000", 110413, 2560)
        s = _sched(0, HEAD, chunked=chunked)
        s._p_chunk_policy_width()
        key, pos, end, queued = s._p_chunk_planner.calls[0]
        self.assertEqual((key, pos, end, queued), ("weg2-50-185-0000", 2560, 110413, False))


class TheLayerSplitWidthUsesTheSamePosition(CustomTestCase):
    def test_leader_width_gets_the_rank_identical_position(self):
        seen = []
        rt = SimpleNamespace(leader_width=lambda rid, pos, end: seen.append((rid, pos, end)) or 1024)
        for rank, prefix in ((0, HEAD), (1, 0)):
            s = _sched(rank, prefix)
            self.assertEqual(s._p_layer_split_width(rt), 1024)
        self.assertEqual([p for _r, p, _e in seen], [0, 0])


class TheBudgetLine(CustomTestCase):
    def test_one_line_per_new_head_with_the_local_prefix_beside(self):
        s = _sched(0, HEAD)
        with self.assertLogs(PBH.logger, level="INFO") as cm:
            s._p_chunk_policy_width()
            s._p_chunk_policy_width()  # same head: no second line
            s.waiting_queue.pop(0)     # the next head
            s._p_chunk_policy_width()
        lines = [l for l in cm.output if "P-CHUNK-BUDGET" in l]
        self.assertEqual(len(lines), 2)
        self.assertIn("head=weg2-50-184", lines[0])
        self.assertIn("pos=0 pos_src=zero", lines[0])
        self.assertIn("local_prefix=21246", lines[0])
        self.assertIn("where=chunk_policy pp_rank=0", lines[0])
        self.assertIn("head=weg2-50-185", lines[1])

    def test_no_line_for_the_chunked_request(self):
        s = _sched(0, 0, chunked=_req("c", 9000, 512))
        logging.getLogger(PBH.__name__).setLevel(logging.INFO)
        with self.assertNoLogs(PBH.logger, level="INFO"):
            s._p_chunk_policy_width()


def _spec():
    st = P.StageModel(((512, 40.0), (1024, 70.0), (2048, 150.0)), 0.002, 60.0)
    return P.PolicySpec((st,) * 3, P.ChunkLimits(2048, 512, 512, graph_buckets=(512,)), "unit")


class QueuedReplansAreNeverRateLimited(CustomTestCase):
    def test_the_17th_replan_is_logged_only_when_queued(self):
        lines = []
        pl = P.planner_from_env({P.POLICY_ENV: "dynamic", P.SPEC_ENV: _spec().to_json()}, log=lines.append)
        for i in range(16):  # 16 replans: logged as before
            pl.next_width(f"r{i}", 0, 30000)
            pl.next_width(f"r{i}", 777, 30000)
        base = len(lines)
        pl.next_width("x", 0, 30000)
        pl.next_width("x", 777, 30000)            # 17th replan, in-flight: rate-limited
        self.assertFalse(any("key=x " in l and "replan=1" in l for l in lines[base:]))
        pl.next_width("y", 0, 30000)
        pl.next_width("y", 777, 30000, queued=True)  # 18th replan, queued head: logged
        q = [l for l in lines[base:] if "key=y " in l and "replan=1" in l]
        self.assertEqual(len(q), 1)
        self.assertIn("queued=1", q[0])

    def test_forward_budget_passes_the_flag(self):
        pl = ReplanLikeThePlanner()
        P.forward_budget(pl, "k", 0, 9000, queued=True)
        P.forward_budget(pl, "k", 1024, 9000)
        self.assertEqual([c[3] for c in pl.calls], [True, False])


class TheGuardGeometryNamesTwelveCharacters(CustomTestCase):
    def test_999_geom_rid_is_12_chars(self):
        from sglang.srt.managers import scheduler_pp_mixin as M

        r = SimpleNamespace(rid="weg2-50-185-abcdef", extend_range=SimpleNamespace(start=0, end=512))
        s = SimpleNamespace(mbs=[SimpleNamespace(reqs=[r])])
        self.assertEqual(M._999_geom(s, 0), ("weg2-50-185-", 0, 512))


if __name__ == "__main__":
    unittest.main()
