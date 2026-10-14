# SPDX-License-Identifier: Apache-2.0
"""ARENA-CHAIN fallback (NF 14.10. 07:29:33Z, P PP0, ``#1424 CHAIN MISMATCH node=460 page=174 of 848``): a refused load-back
chain on group P is a group-wide VOID PASS, not a rank death.

Why a rank died: ``verify_load_chain`` is "the LAST latch" -- the admission vote ``provable_pages`` (``proof_cut``) is meant to cut the
match before it, but it runs in the TP group D only (``tp_match_floor``); on P nobody calls it, and the exception left
``_get_new_batch_prefill_raw`` uncaught (only ``AgreedRoomShort`` and ``PPScheduleRefused`` are caught there).

Now: ``ArenaChainMismatch`` carries the page / node; the scheduler catches it next to ``AgreedRoomShort`` and
``prefill_oom_retreat.retreat_chain_mismatch`` takes it like a load-back refusal (``load_back`` raised before any device
allocation, the host lock unwound): the request is skipped for the pass, the pass is voided on EVERY stage through the void frame
(cause ``chain_mismatch``), the N-then-503 verdict stays PP0's alone, and the page nothing proves leaves the raising rank's host
tier so the next pass does not meet it again.

Driven through the real ``pp_void_pass`` hooks and the stage rig of the 1013 void-pass test; the tree cut through the real
``UnifiedRadixCache.weg2_cut_unproven_chain`` / ``_ud_clear_host_children`` / ``_is_host_leaf`` (I/O steps stubbed, as in the
UD-H test).
"""
from __future__ import annotations

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402,F401

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import BASE_COMPONENT_TYPE as FULL  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.weg2 import pp_void_pass as VP  # noqa: E402
from sglang.srt.weg2 import prefill_oom_retreat as R  # noqa: E402
from sglang.srt.weg2.intake_stall import is_intake_stall  # noqa: E402
from test_weg2_pp_void_pass_1013 import PP, Group, Stage, _Base  # noqa: E402
from test_weg2_prefill_oom_retreat_1013 import _Req, _fails  # noqa: E402

MSG = "#1424 CHAIN MISMATCH node=460 page=174 of 848 bad=256: key slot -1 (state 0) -- the load is refused"


def _exc(node="n460", page=174, node_page=174):
    return ap.ArenaChainMismatch(MSG, page=page, node=node, node_page=node_page)


class _Cutter:
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

    def test_the_unproven_page_leaves_the_raising_ranks_host_tier(self):
        g = Group(frees=(200000, 200000, 200000))
        cutter = _Cutter()
        self._pass(g, cutter=cutter)
        self.assertEqual(cutter.calls, [("n460", 174)], "the node and the page the proof named, on the rank that was refused")

    def test_a_cut_the_tree_refuses_does_not_stop_the_void_and_the_wall_that_stands_ends_in_a_503(self):
        g = Group(frees=(200000, 200000, 200000))
        os.environ[R.ENV_HOLD] = "0"
        stuck = _Cutter({"cut": False, "why": "children:child_host_locked:7", "nodes": 0, "tokens": 0})
        for _ in range(R.RETRIES_DEFAULT):
            self._pass(g, cutter=stuck)
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
# the cut in the real tree
# ---------------------------------------------------------------------------


class _CD:
    def __init__(self, host_value=None, host_lock=0, lock=0, value=None):
        self.value, self.host_value, self.lock_ref, self.host_lock_ref = value, host_value, lock, host_lock


class _Node:
    _n = 0

    def __init__(self, parent, *, host=0, evicted=True, backuped=True, host_lock=0, device=0):
        _Node._n += 1
        self.id = _Node._n
        self.parent = parent
        self.children = {}
        self.evicted, self.backuped = evicted, backuped
        self.component_data = [_CD(list(range(host)) if host else None, host_lock, value=list(range(device)) if device else None),
                               _CD(), _CD()]
        if parent is not None:
            parent.children[self.id] = self


class _Tree:
    weg2_cut_unproven_chain = UnifiedRadixCache.weg2_cut_unproven_chain
    _is_host_leaf = UnifiedRadixCache._is_host_leaf
    _ud_clear_host_children = UnifiedRadixCache._ud_clear_host_children

    def __init__(self):
        self.root_node = _Node(None, evicted=False, backuped=False)
        self.tree_components = (FULL,)
        self.host_evicted = []

    def _evict_host_leaf(self, node, tracker):
        assert self._is_host_leaf(node)
        node.parent.children.pop(node.id)
        tracker[FULL] += len(node.component_data[FULL].host_value)
        self.host_evicted.append(node.id)


class CutTest(unittest.TestCase):
    def test_the_node_and_its_host_only_subtree_leave_the_host_tier_children_first(self):
        t = _Tree()
        up = _Node(t.root_node, host=100)
        bad = _Node(up, host=848 * 64)
        c1 = _Node(bad, host=640)
        c2 = _Node(c1, host=320)
        res = t.weg2_cut_unproven_chain(bad, 174)
        self.assertEqual((res["cut"], res["nodes"], res["why"]), (True, 3, ""))
        self.assertEqual(t.host_evicted, [c2.id, c1.id, bad.id], "children before their parent")
        self.assertIn(up.id, t.root_node.children, "the proven ancestor stays")
        self.assertEqual(up.children, {})
        self.assertEqual(res["tokens"], 848 * 64)

    def test_a_host_lock_below_or_on_the_node_refuses_the_whole_cut(self):
        for where in ("child", "node"):
            t = _Tree()
            bad = _Node(t.root_node, host=64, host_lock=1 if where == "node" else 0)
            _Node(bad, host=64, host_lock=1 if where == "child" else 0)
            res = t.weg2_cut_unproven_chain(bad, 3)
            self.assertFalse(res["cut"], where)
            self.assertEqual(t.host_evicted, [], "all or nothing")
            self.assertIn(bad.id, t.root_node.children)

    def test_a_child_on_the_device_refuses_the_cut(self):
        t = _Tree()
        bad = _Node(t.root_node, host=64)
        _Node(bad, evicted=False, backuped=False, device=64)
        res = t.weg2_cut_unproven_chain(bad)
        self.assertFalse(res["cut"])
        self.assertTrue(res["why"].startswith("children:child_on_device"), res)

    def test_a_node_that_is_gone_or_has_a_device_copy_is_not_cut(self):
        t = _Tree()
        gone = _Node(t.root_node, host=64)
        t.root_node.children.pop(gone.id)
        self.assertEqual(t.weg2_cut_unproven_chain(gone)["why"], "already_gone")
        dev = _Node(t.root_node, host=64, evicted=False, backuped=True, device=64)
        self.assertEqual(t.weg2_cut_unproven_chain(dev)["why"], "not_host_only")
        self.assertEqual(t.weg2_cut_unproven_chain(None)["why"], "no_node")
        self.assertEqual(t.host_evicted, [])


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

    def test_c2_the_unproven_page_stays_in_the_tree(self):
        """The retreat runs as a plain load-back retreat: the tree is never asked to cut."""
        orig = R.retreat_loadback
        self._killed(ChainVoidTest, ["test_the_unproven_page_leaves_the_raising_ranks_host_tier"],
                     [mock.patch.object(R, "retreat_loadback", lambda s, e, r, a, chain=False: orig(s, e, r, a, chain=False))])

    def test_c3_the_exception_is_not_taken_at_all(self):
        self._killed(ChainVoidTest, ["test_the_whole_pass_is_discarded_on_every_stage_and_the_rank_lives"],
                     [mock.patch.object(R, "chain_void_enabled", lambda env=None: False)])

    def test_c4_the_wall_never_ends_in_a_503(self):
        self._killed(ChainVoidTest, ["test_a_cut_the_tree_refuses_does_not_stop_the_void_and_the_wall_that_stands_ends_in_a_503"],
                     [mock.patch.object(R, "max_retries", lambda env=None: 1000)])

    def test_c5_the_cut_removes_a_locked_node(self):
        self._killed(CutTest, ["test_a_host_lock_below_or_on_the_node_refuses_the_whole_cut"],
                     [mock.patch.object(_Tree, "_is_host_leaf", lambda self, n: True),
                      mock.patch.object(_Tree, "_ud_clear_host_children", lambda self, n: True)])


if __name__ == "__main__":
    unittest.main()
