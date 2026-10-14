# SPDX-License-Identifier: Apache-2.0
"""ARENA-CHAIN fallback (NF 14.10. 07:29:33Z, P PP0, ``#1424 CHAIN MISMATCH node=460 page=174 of 848``): a refused load-back
chain on group P is a group-wide VOID PASS, not a rank death.

Why a rank died: ``verify_load_chain`` is "the LAST latch" -- the admission vote ``provable_pages`` (``proof_cut``) is meant to cut the
match before it, but it runs in the TP group D only (``tp_match_floor``); on P nobody calls it, and the exception left
``_get_new_batch_prefill_raw`` uncaught (only ``AgreedRoomShort`` and ``PPScheduleRefused`` are caught there).

Now: ``ArenaChainMismatch`` carries the page / node; the scheduler catches it next to ``AgreedRoomShort`` and
``prefill_oom_retreat.retreat_chain_mismatch`` takes it like a load-back refusal (``load_back`` raised before any device
allocation, the host lock unwound): the request is skipped for the pass, the pass is voided on EVERY stage through the void frame
(cause ``chain_mismatch``), the N-then-503 verdict stays PP0's alone.

PORT onto the h16 line (desk/nf-h18port-1014): the h18 tree cut ``weg2_cut_unproven_chain`` (the raising rank cut the node out of
ITS host tier alone) is left out -- a rank-local repair that moves one stage's host reach against its peers (NF-RANG-AUDIT S11,
raenge-nie-uneins). No rank's tree is touched; the next read re-proves the page (#1424 ARENA-REF RECHECK), a page that stays
unprovable ends in PP0's 503. And while PP0's distributed LB-VERDICT for the rid is released on the stage, the chain void is NOT
taken (the S4 shape of ``lb_verdict_short``): the group stops by name.

Driven through the real ``pp_void_pass`` hooks and the stage rig of the 1013 void-pass test.
"""
from __future__ import annotations

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402,F401

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.weg2 import pp_loadback_verdict as LBV  # noqa: E402
from sglang.srt.weg2 import pp_void_pass as VP  # noqa: E402
from sglang.srt.weg2 import prefill_oom_retreat as R  # noqa: E402
from sglang.srt.weg2.intake_stall import is_intake_stall  # noqa: E402
from test_weg2_pp_void_pass_1013 import PP, Group, Stage, _Base  # noqa: E402
from test_weg2_prefill_oom_retreat_1013 import _Req, _fails  # noqa: E402

MSG = "#1424 CHAIN MISMATCH node=460 page=174 of 848 bad=256: key slot -1 (state 0) -- the load is refused"


def _exc(node="n460", page=174, node_page=174):
    return ap.ArenaChainMismatch(MSG, page=page, node=node, node_page=node_page)


class _Cutter:
    """A spy where the h18 cut sat: on this line nobody may call it."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"cut": True, "why": "", "nodes": 1, "tokens": 54272}

    def __call__(self, node, page=None):
        self.calls.append((node, page))
        return self.result


def plan_chain(st, mb, refused, *, admitted=(), raises=True, exc=None, cutter=None):
    """The admission loop of this stage meets ``ArenaChainMismatch`` at ``refused`` (``load_back`` raised); the loop ends
    (``NO_TOKEN``) and the pass is built from what was admitted before. ``raises=False``: this stage's chain is proven (a follower)."""
    s = st.sched
    VP.snapshot_queue(s)
    if raises:
        s.tree_cache.weg2_cut_unproven_chain = cutter if cutter is not None else _Cutter()
        e = exc or _exc()
        verdict = R.retreat_chain_mismatch(s, e, refused, types.SimpleNamespace(can_run_list=list(admitted), new_anchor_tails=None))
        if verdict is None:
            raise e
        members = list(admitted)
    else:
        st.alloc(refused.rid, 92288)  # the load-back lands
        members = list(admitted) + [refused]
    st._admit([r for r in members if raises is False or r is not refused])
    if members:
        s.mbs[mb] = types.SimpleNamespace(reqs=members)
        for r in admitted:
            st.alloc(r.rid, 4096)
    else:
        s.mbs[mb] = None


class ChainVoidTest(_Base):
    def setUp(self):
        super().setUp()
        self._chain_env = os.environ.pop(R.ENV_CHAIN, None)

    def tearDown(self):
        if self._chain_env is None:
            os.environ.pop(R.ENV_CHAIN, None)
        else:
            os.environ[R.ENV_CHAIN] = self._chain_env
        super().tearDown()

    def _pass(self, g, *, cutter=None):
        req = [_Req("weg2-44-221", prefix=0, total=94288, inflight=0) for _ in g.st]
        for st, r in zip(g.st, req):
            st.sched.waiting_queue = [r]
        plan_chain(g.st[0], 1, req[0], cutter=cutter)
        for st, r in zip(g.st[1:], req[1:]):
            plan_chain(st, 1, r, raises=False)
        return req

    def test_the_whole_pass_is_discarded_on_every_stage_and_the_rank_lives(self):
        """PP0's chain is refused; PP1/PP2 plan from their own queue and load the request back. Without the group void they launch a
        pass PP0 never sends a frame for -- and without the retreat PP0 dies (the 07:29:33Z death)."""
        g = Group(frees=(200000, 200000, 200000))
        with self.assertLogs(VP.logger, "WARNING") as logs:
            req = self._pass(g)
            g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual([st.free for st in g.st], [st.free0 for st in g.st], "the followers' load-back rows are back")
        self.assertEqual([[q.rid for q in st.sched.waiting_queue] for st in g.st], [["weg2-44-221"]] * PP,
                         "the request stays queued on every rank")
        lines = [m for m in logs.output if "PP-VOID-PASS rank=" in m]
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("cause=chain_mismatch rid=weg2-44-221" in m for m in lines), lines)

    def test_no_rank_cuts_its_host_tier(self):
        """raenge-nie-uneins (S11): the raising rank's tree is NOT cut -- every stage keeps the same host reach."""
        g = Group(frees=(200000, 200000, 200000))
        cutter = _Cutter()
        self._pass(g, cutter=cutter)
        g.run(1)
        g.assert_aligned(self, launched=[])
        self.assertEqual(cutter.calls, [], "a rank-local cut would move this stage's host reach against its peers")

    def test_the_wall_that_stands_ends_in_a_503(self):
        g = Group(frees=(200000, 200000, 200000))
        os.environ[R.ENV_HOLD] = "0"
        for _ in range(R.RETRIES_DEFAULT):
            self._pass(g)
            g.run(1)
            g.assert_aligned(self, launched=[])
            aborts = g.st[0].sched._weg2_intake_origin_aborts.take()
        self.assertEqual([a.rid for a in aborts], ["weg2-44-221"])
        msg = aborts[0].finished_reason["message"]
        self.assertTrue(is_intake_stall(msg), "the front requeues exactly this request")
        self.assertIn("ARENA-CHAIN-VOID", msg)
        self.assertEqual(g.st[0].sched.waiting_queue, [])

    def test_a_follower_that_is_refused_first_unwinds_and_takes_its_frame(self):
        """PP1's chain is refused while PP0's pass is void as well (the usual order): consistent, nobody dies."""
        g = Group(frees=(200000, 200000, 200000))
        req = [_Req("weg2-44-221", prefix=0, total=94288, inflight=0) for _ in g.st]
        for st, r in zip(g.st, req):
            st.sched.waiting_queue = [r]
        plan_chain(g.st[0], 1, req[0])
        plan_chain(g.st[1], 1, req[1])
        plan_chain(g.st[2], 1, req[2], raises=False)
        g.run(1)
        g.assert_aligned(self, launched=[])

    def test_the_switch_off_is_the_raise_of_before(self):
        os.environ[R.ENV_CHAIN] = "0"
        st = Stage(0, 200000)
        r = _Req("r", prefix=0, inflight=0)
        with self.assertRaises(ap.ArenaChainMismatch):
            plan_chain(st, 1, r)
        self.assertIsNone(getattr(st.sched, VP.AFTER_PLAN_ATTR, None))

    def test_out_of_scope_it_still_stops_the_rank(self):
        """A TP group (D) or a non-P group: a rank-local void would be a split -- the proof_cut vote is that group's protection."""
        st = Stage(0, 200000)
        st.sched.ps.tp_size = 2
        with self.assertRaises(ap.ArenaChainMismatch):
            plan_chain(st, 1, _Req("r", prefix=0, inflight=0))
        os.environ[R.GROUP_ENV] = "D"
        st2 = Stage(0, 200000)
        with self.assertRaises(ap.ArenaChainMismatch):
            plan_chain(st2, 1, _Req("r", prefix=0, inflight=0))

    def test_a_retreat_that_cannot_complete_leaves_no_pending_state(self):
        s = Group().st[0].sched
        s.tree_cache.weg2_cut_unproven_chain = _Cutter()
        with mock.patch.object(R, "_answer_abort", side_effect=ValueError("boom")), \
                mock.patch.object(R, "verdict", lambda **k: (R.ABORT, "mutant")):
            self.assertIsNone(R.retreat_chain_mismatch(
                s, _exc(), _Req("r", prefix=0, inflight=0), types.SimpleNamespace(can_run_list=[], new_anchor_tails=None)))
        self.assertIsNone(getattr(s, VP.AFTER_PLAN_ATTR, None))

    def test_the_scheduler_catches_it_where_it_catches_agreed_room_short(self):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py").read_text()
        i_ars = src.index("except AgreedRoomShort as _ars:")
        i_acm = src.index("except ArenaChainMismatch as _acm:")
        i_refused = src.index("except PPScheduleRefused as exc:")
        self.assertLess(i_ars, i_acm)
        self.assertLess(i_acm, i_refused)
        self.assertIn("retreat_chain_mismatch(self, _acm, req, adder)", src[i_acm:i_refused])
        self.assertIn("raise", src[i_acm:i_refused], "not taken -> the exception stops the rank as before")


# ---------------------------------------------------------------------------
# a released LB-VERDICT: the chain void is not taken (S4 shape)
# ---------------------------------------------------------------------------

RID = "weg2-44-221"


def _armed_group_released():
    """Three stages of the carrierless P with PP0's LB-VERDICT for RID absorbed on every stage (released this pass)."""
    g = Group(frees=(200000, 200000, 200000))
    for st in g.st:
        assert LBV.arm(st.sched)
        LBV.absorb(st.sched.tree_cache, LBV.Weg2PpLoadbackVerdict(rid=RID))
    return g


def _chain_pass_released(g):
    """PP0's chain is refused; the followers' chains are proven -- they load the released rid back and spend the verdict
    (``load_back``: ``_lbv.consume``). Returns PP0's retreat verdict (``None`` = not taken)."""
    req = [_Req(RID, prefix=0, total=94288, inflight=0) for _ in g.st]
    for st, r in zip(g.st, req):
        st.sched.waiting_queue = [r]
    VP.snapshot_queue(g.st[0].sched)
    v = R.retreat_chain_mismatch(g.st[0].sched, _exc(), req[0], types.SimpleNamespace(can_run_list=[], new_anchor_tails=None))
    for st in g.st[1:]:
        LBV.consume(st.sched.tree_cache, RID)
    return v


class ReleasedVerdictTest(_Base):
    def setUp(self):
        super().setUp()
        # NOT ``self._env``: that is _Base's saved environment (GROUP/PARK) -- overwriting it leaked GROUP=P / PARK=1
        self._rv_env = {k: os.environ.pop(k, None) for k in (R.ENV_CHAIN, R.ENV_LBV_SHORT_STOP, LBV.ENV, LBV.ROW_YIELD_ENV)}

    def tearDown(self):
        for k, v in self._rv_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()

    def test_a_released_verdict_is_a_named_stop(self):
        g = _armed_group_released()
        with self.assertLogs(R.logger, "ERROR") as logs:
            self.assertIsNone(_chain_pass_released(g), "not taken: the scheduler re-raises, the group stops by name")
        self.assertTrue(any("lb_verdict_released" in m for m in logs.output), logs.output)
        self.assertIsNone(getattr(g.st[0].sched, VP.AFTER_PLAN_ATTR, None), "no void frame is put on the wire")

    def test_without_a_released_verdict_the_void_stays(self):
        g = Group(frees=(200000, 200000, 200000))
        for st in g.st:
            assert LBV.arm(st.sched)
        req = [_Req(RID, prefix=0, total=94288, inflight=0) for _ in g.st]
        for st, r in zip(g.st, req):
            st.sched.waiting_queue = [r]
        VP.snapshot_queue(g.st[0].sched)
        self.assertEqual(R.retreat_chain_mismatch(g.st[0].sched, _exc(), req[0],
                                                  types.SimpleNamespace(can_run_list=[], new_anchor_tails=None)), R.RETRY)

    def test_rank_equality_mutant_the_void_splits_the_verdict(self):
        """Mutant: the chain void ignores the released verdict (the h18 code). PP0 voids (RETRY) and keeps the rid released,
        the followers have spent it loading back -> [True, False, False] after the pass. The real rule: PP0 does not take it
        (the stop is uniform)."""
        g = _armed_group_released()
        with mock.patch.object(R, "_lb_verdict_released", lambda s, r: False):
            v = _chain_pass_released(g)
        self.assertEqual(v, R.RETRY)
        state = [LBV.released(st.sched.tree_cache, RID) for st in g.st]
        self.assertEqual(state, [True, False, False], "the mutant leaves the stages split")
        g2 = _armed_group_released()
        self.assertIsNone(_chain_pass_released(g2))

    def test_the_switch_of_the_lbv_stop_governs_it_too(self):
        os.environ[R.ENV_LBV_SHORT_STOP] = "0"
        g = _armed_group_released()
        self.assertEqual(_chain_pass_released(g), R.RETRY)


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------


class MutantsTest(unittest.TestCase):
    def _killed(self, case, names, patches):
        self.assertFalse(_fails(case, names, []), "the unmutated tests must pass")
        self.assertTrue(_fails(case, names, patches), "the mutant survived")

    def test_c1_the_pass_is_not_voided_as_a_whole(self):
        self._killed(ChainVoidTest, ["test_the_whole_pass_is_discarded_on_every_stage_and_the_rank_lives"],
                     [mock.patch.object(VP, "after_plan", lambda *a, **k: False)])

    def test_c2_rank_local_cut_is_back(self):
        """Rank-equality mutant (S11): the h18 cut returns -- the raising rank cuts its own host tier, its peers do not."""
        orig = R.retreat_loadback

        def _with_cut(s, e, r, a, chain=False):
            if chain:
                s.tree_cache.weg2_cut_unproven_chain(getattr(e, "node", None), getattr(e, "node_page", None))
            return orig(s, e, r, a, chain=chain)

        self._killed(ChainVoidTest, ["test_no_rank_cuts_its_host_tier"],
                     [mock.patch.object(R, "retreat_loadback", _with_cut)])

    def test_c3_the_exception_is_not_taken_at_all(self):
        self._killed(ChainVoidTest, ["test_the_whole_pass_is_discarded_on_every_stage_and_the_rank_lives"],
                     [mock.patch.object(R, "chain_void_enabled", lambda env=None: False)])

    def test_c4_the_wall_never_ends_in_a_503(self):
        self._killed(ChainVoidTest, ["test_the_wall_that_stands_ends_in_a_503"],
                     [mock.patch.object(R, "max_retries", lambda env=None: 1000)])

    def test_c5_the_released_verdict_is_ignored(self):
        self._killed(ReleasedVerdictTest, ["test_a_released_verdict_is_a_named_stop"],
                     [mock.patch.object(R, "_lb_verdict_released", lambda s, r: False)])


if __name__ == "__main__":
    unittest.main()
