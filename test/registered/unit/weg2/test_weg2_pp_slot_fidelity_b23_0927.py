# SPDX-License-Identifier: Apache-2.0
"""SF slot fidelity (27B rc12k27 b23, 27.09. 10:27:56Z, rid weg2-68-235).

P group death ``#1004 SLOT DISAGREEMENT``: PP1 launched weg2-68-235 in slot 0 with extend=512,
PP0's proxy named slot 1 with rows=1024, extent (40958, 41982). Two terms:

(a) PP0's load-back room verdict was rank-local -- the "uniform floor" is the tp_cpu_group MIN and
    that group has one member on TP=1/PP=3 (``#788 ... floors OFF``). PP0 (floor 13035 < 27095)
    refused this pass, PP1/PP2 loaded: pass skew.
(b) PP0's second visit sized the pass from position 0 (the told was popped at the first visit, H91
    KEPT it, Fix A's budget head did not look there): plan start=0 -> 1024 on an extent #988 moved
    to 40958, where the plan says 512.

Every test drives the real code (``UnifiedRadixCache.load_back``, ``Scheduler`` methods, the budget
head) on stand-ins; the switch-off cases are the metal behaviour and must disagree.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import p_budget_head as PBH  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

KV_TOKENS = 27095    # WEG2-ARENA-LOAD rows=27095 on every rank
PP0_FLOOR = 13035    # WEG2-LOADBACK-EVICT floor=13035 (PP0 only)
PP0_EVICTABLE = 218004
TOLD = 40958
END = 61395          # weg2-68-235's fill


def _env(on: bool):
    return mock.patch.dict(os.environ, {SF.ENV: "1" if on else "0"})


# ---------------------------------------------------------------------------
# (a) the real load_back on a stand-in tree
# ---------------------------------------------------------------------------


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail


class _Params:
    def to_dec_params(self):
        return None


class _LockRes(_Params):
    delta = 0


class _KvXfer:
    def __init__(self, n):
        self.host_indices = list(range(n))
        self.nodes_to_load = ()
        self.device_indices = None


class _BaseComp:
    def __init__(self, n):
        self.n = n
        self.committed = 0

    def build_hicache_transfers(self, node, phase, req=None):
        return [_KvXfer(self.n)]

    def commit_hicache_transfer(self, node, phase, xfers):
        self.committed += 1


class _Controller:
    def __init__(self, tree):
        self.tree = tree
        self.loads = 0

    def load(self, host_indices, node_id, extra_pools=None):
        n = len(host_indices)
        if self.tree.token_to_kv_pool_allocator.avail < n:
            return None
        self.tree.token_to_kv_pool_allocator.avail -= n
        self.loads += 1
        return list(range(n))


def _tree(avail, evictable, floor, local_pp=None):
    from sglang.srt.mem_cache import unified_radix_cache as URC

    class _T:
        load_back = URC.UnifiedRadixCache.load_back

        def _1424_verify_load_chain(self, kv_xfer, req=None):
            # #1424b-d (after this test was written): load_back proves the chain
            # before queueing it. These stand-in trees have no arena chain.
            return None

        def __init__(self):
            self.token_to_kv_pool_allocator = _Alloc(avail)
            self._evictable = evictable
            self.uniform_avail_floor = floor
            self.cache_controller = _Controller(self)
            self.components = {URC.BASE_COMPONENT_TYPE: _BaseComp(KV_TOKENS)}
            self._components_tuple = ()
            self.load_back_threshold = 10
            self.ongoing_load_back = {}
            self.metrics_collector = None
            self.evicted = []
            if local_pp is not None:
                setattr(self, SF.FLOOR_LOCAL_PP_ATTR, local_pp)

        def inc_host_lock_ref(self, node):
            return _Params()

        def inc_lock_ref(self, node):
            return _LockRes()

        def dec_lock_ref(self, node, params=None):
            pass

        def dec_host_lock_ref(self, node, params=None):
            pass

        def _build_sidecar_transfers(self, phase, kv, comp):
            return []

        def evictable_size(self):
            return self._evictable

        def evict(self, params):
            n = min(int(params.num_tokens), self._evictable)
            self._evictable -= n
            self.token_to_kv_pool_allocator.avail += n
            self.evicted.append(n)
            return SimpleNamespace(num_tokens_evicted=n)

        def _record_store_event(self, node, medium=None):
            pass

        def _update_evictable_leaf_sets(self, node):
            pass

    return _T()


def _node():
    return SimpleNamespace(id=7)


def _req():
    return SimpleNamespace(rid="weg2-68-235")


def _pp_ranks(local_pp):
    """The b23 shape: PP0 short by 14060 with 218004 evictable, the followers roomy."""
    pp0 = _tree(PP0_FLOOR, PP0_EVICTABLE, PP0_FLOOR, local_pp)
    pp1 = _tree(60000, 150000, 60000, local_pp)
    pp2 = _tree(58000, 150000, 58000, local_pp)
    return pp0, pp1, pp2


class TheLoadBackAdmitsInTheSamePassOnEveryRank(CustomTestCase):
    def test_b23_every_rank_loads_in_this_pass(self):
        with _env(True):
            ranks = _pp_ranks(True)
            verdicts = [t.load_back(_node(), req=_req()) for t in ranks]
        self.assertEqual(verdicts, [True, True, True], "a stage refused the pass its peers loaded in")
        pp0 = ranks[0]
        self.assertEqual(pp0.evicted, [KV_TOKENS - PP0_FLOOR])  # the shortfall, not the whole set
        self.assertEqual(pp0.cache_controller.loads, 1)
        self.assertIn(7, pp0.ongoing_load_back)

    def test_the_metal_behaviour_switch_off_splits_the_pass(self):
        with _env(False):
            ranks = _pp_ranks(True)
            verdicts = [t.load_back(_node(), req=_req()) for t in ranks]
        self.assertEqual(verdicts, [False, True, True])  # PP0 refuses, PP1/PP2 load: #1004
        self.assertEqual(ranks[0].evicted, [PP0_EVICTABLE])  # the xsn285 drain, then refusal

    def test_switch_off_writes_no_mark(self):
        with _env(False):
            t = _tree(1, 1, 1)
            SF.mark_floor_scope(t, True)
        self.assertFalse(hasattr(t, SF.FLOOR_LOCAL_PP_ATTR))

    def test_a_group_floor_keeps_the_refuse_and_retry(self):
        # TP group (the floor is a real MIN): unchanged xsn285 path.
        with _env(True):
            t = _tree(PP0_FLOOR, PP0_EVICTABLE, PP0_FLOOR, local_pp=False)
            self.assertFalse(t.load_back(_node(), req=_req()))
        self.assertEqual(t.evicted, [PP0_EVICTABLE])

    def test_the_residual_refuses_and_names_itself(self):
        with _env(True), self.assertLogs(SF.logger, level="WARNING") as cap:
            t = _tree(1000, 5000, 1000, local_pp=True)
            self.assertFalse(t.load_back(_node(), req=_req()))
        self.assertEqual(sum(t.evicted), 5000)
        self.assertEqual(t.cache_controller.loads, 0)
        self.assertIn("SF LOADBACK-ROOM PP-RESIDUAL rid=weg2-68-235 kv_tokens=27095", cap.output[0])
        self.assertIn("#1004", cap.output[0])

    def test_the_same_pass_line(self):
        with _env(True), self.assertLogs(SF.logger, level="INFO") as cap:
            _pp_ranks(True)[0].load_back(_node(), req=_req())
        line = [l for l in cap.output if "SAME-PASS" in l][0]
        self.assertIn("kv_tokens=27095 floor=13035 avail=13035 evictable=218004 evicted=14060 "
                      "avail_after=27095", line)

    def test_a_live_room_without_eviction(self):
        # published floor stale-low, the pool freed since: no eviction needed.
        with _env(True):
            t = _tree(30000, 100, PP0_FLOOR, local_pp=True)
            self.assertTrue(t.load_back(_node(), req=_req()))
        self.assertEqual(t.evicted, [])


# ---------------------------------------------------------------------------
# (a) the scheduler marks the floor's scope
# ---------------------------------------------------------------------------


def _fake_sched(avail, pp_size, tree):
    from sglang.srt.managers.scheduler import Scheduler

    class _S:
        _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
        _publish_uniform_evict_floor = Scheduler._publish_uniform_evict_floor
        _publish_uniform_host_floor = Scheduler._publish_uniform_host_floor
        _publish_uniform_mamba_floor = Scheduler._publish_uniform_mamba_floor
        _HOST_AVAIL_ABSENT = Scheduler._HOST_AVAIL_ABSENT
        _local_host_avail = Scheduler._local_host_avail
        _MAMBA_AVAIL_ABSENT = Scheduler._MAMBA_AVAIL_ABSENT
        _local_mamba_avail = Scheduler._local_mamba_avail

    s = _S()
    s.kv_session_offload = None
    s.token_to_kv_pool_allocator = _Alloc(avail)
    s.tp_cpu_group = None
    s.tree_cache = tree
    s.server_args = SimpleNamespace(pp_size=pp_size, tp_size=1, dcp_size=1)
    return s


class TheSchedulerMarksALocalFloorOnPP(CustomTestCase):
    def test_tp1_pp3_marks_local(self):
        tree = SimpleNamespace()
        with _env(True):
            _fake_sched(PP0_FLOOR, 3, tree)._update_uniform_pool_budget()
        self.assertEqual(tree.uniform_avail_floor, PP0_FLOOR)
        self.assertTrue(getattr(tree, SF.FLOOR_LOCAL_PP_ATTR))

    def test_single_gpu_pp1_is_not_marked(self):
        tree = SimpleNamespace()
        with _env(True):
            _fake_sched(PP0_FLOOR, 1, tree)._update_uniform_pool_budget()
        self.assertFalse(getattr(tree, SF.FLOOR_LOCAL_PP_ATTR))

    def test_a_group_publish_clears_the_mark(self):
        # a flip into the TP phase publishes a group MIN: the P phase's mark goes.
        tree = SimpleNamespace()
        setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, True)
        with _env(True):
            _fake_sched(1, 3, tree)._publish_uniform_evict_floor(5000, max_avail=9000)
        self.assertFalse(getattr(tree, SF.FLOOR_LOCAL_PP_ATTR))

    def test_switch_off_leaves_the_tree_untouched(self):
        tree = SimpleNamespace()
        with _env(False):
            _fake_sched(PP0_FLOOR, 3, tree)._update_uniform_pool_budget()
        self.assertFalse(hasattr(tree, SF.FLOOR_LOCAL_PP_ATTR))


# ---------------------------------------------------------------------------
# (b1) the budget head reads the KEPT told
# ---------------------------------------------------------------------------


class PlannerLikeTheSpecimen:
    """plan at 0 opens with 1024 (``widths=[1024x59,979]``), a plan at 40958 is fixed 512."""

    def __init__(self):
        self.calls = []

    def next_width(self, key, pos, end, queued=False):
        self.calls.append((key, pos, end, queued))
        return 1024 if pos == 0 else 512

    class spec:
        class limits:
            fixed_tokens = 512


def _breq(rid="weg2-68-235"):
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(END)), origin_input_ids=[],
                           output_ids=[], prefix_indices=[])


def _bsched(req, told_map=None, kept=None, static=2048):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.weg2.p_intake import _Kept

    class _S:
        _p_chunk_policy_width = Scheduler._p_chunk_policy_width

    s = _S()
    s.pp_rank = 0
    s.chunked_prefill_size = static
    s.chunked_req = None
    s.waiting_queue = [req]
    s._p_chunk_planner = PlannerLikeTheSpecimen()
    s._weg2_store_told = dict(told_map or {})
    if kept is not None:
        s._weg2_told_kept = {req.rid: _Kept(req=kept, told=TOLD, credit=KV_TOKENS)}
    return s


class TheSecondVisitSizesFromTheKeptTold(CustomTestCase):
    def test_pp0_second_visit_and_follower_first_visit_agree(self):
        r0, r1 = _breq(), _breq()
        pp0 = _bsched(r0, told_map={}, kept=r0)          # told popped at visit 1, H91 kept it
        pp1 = _bsched(r1, told_map={r1.rid: TOLD})       # first visit, told on the map
        with _env(True):
            w0, w1 = pp0._p_chunk_policy_width(), pp1._p_chunk_policy_width()
            h0 = PBH.budget_head(pp0)
        self.assertEqual((w0, w1), (512, 512))
        self.assertEqual((h0.pos, h0.src), (TOLD, "kept"))
        self.assertEqual(pp0._p_chunk_planner.calls[-1][1], TOLD)

    def test_the_metal_behaviour_switch_off_sizes_from_zero(self):
        r0 = _breq()
        pp0 = _bsched(r0, told_map={}, kept=r0)
        with _env(False):
            self.assertEqual(pp0._p_chunk_policy_width(), 1024)  # plan start=0 -> 1024
            self.assertEqual(PBH.budget_head(pp0).src, "zero")

    def test_a_kept_verdict_of_another_request_object_is_not_used(self):
        r0 = _breq()
        pp0 = _bsched(r0, told_map={}, kept=_breq())  # re-intake: a different object
        with _env(True):
            self.assertEqual(PBH.budget_head(pp0).src, "zero")

    def test_the_told_map_wins_over_the_kept(self):
        r0 = _breq()
        pp0 = _bsched(r0, told_map={r0.rid: 16383}, kept=r0)
        with _env(True):
            h = PBH.budget_head(pp0)
        self.assertEqual((h.pos, h.src), (16383, "told"))


# ---------------------------------------------------------------------------
# (b2) a #988 prefix move replans the pass budget
# ---------------------------------------------------------------------------


def _adder(rem, can_run=()):
    return SimpleNamespace(rem_chunk_tokens=rem, can_run_list=list(can_run))


def _planned(pos, width=None, told_map=None, kept=None, static=2048):
    r = _breq()
    s = _bsched(r, told_map=told_map if told_map is not None else ({r.rid: pos} if pos else {}),
                kept=kept, static=static)
    w = s._p_chunk_policy_width()
    return s, r, w


class APrefixMoveReplansTheBudget(CustomTestCase):
    def test_planned_at_zero_moved_to_the_told_narrows_to_the_plan_there(self):
        with _env(True), self.assertLogs(SF.logger, level="INFO") as cap:
            s, r, w = _planned(0)
            self.assertEqual(w, 1024)
            a = _adder(w)
            hook = SF.replan_hook(s, a)
            self.assertEqual(hook(r, TOLD), 512)
        self.assertEqual(a.rem_chunk_tokens, 512)
        self.assertEqual(s._p_chunk_planner.calls[-1], (r.rid, TOLD, END, True))
        self.assertIn("SF P-CHUNK REPLAN-AT-MOVE rid=weg2-68-235 planned_start=0 (src=zero) "
                      "moved_to=40958 width 1024 -> 512", cap.output[-1])

    def test_a_move_to_the_planned_start_changes_nothing(self):
        with _env(True):
            s, r, w = _planned(TOLD)
            a = _adder(w)
            n_calls = len(s._p_chunk_planner.calls)
            self.assertIsNone(SF.replan_hook(s, a)(r, TOLD))
        self.assertEqual(a.rem_chunk_tokens, 512)
        self.assertEqual(len(s._p_chunk_planner.calls), n_calls)

    def test_never_widened(self):
        # planned at 40958 (512), the move lands at 0 where the plan would give 1024
        with _env(True):
            s, r, w = _planned(TOLD)
            a = _adder(w)
            SF.replan_hook(s, a)(r, 0)
        self.assertEqual(a.rem_chunk_tokens, 512)

    def test_a_corridor_narrowed_budget_stays_the_ceiling(self):
        with _env(True):
            s, r, w = _planned(0)
            a = _adder(256)  # the corridor granted 256 of the 1024
            SF.replan_hook(s, a)(r, TOLD)
        self.assertEqual(a.rem_chunk_tokens, 256)

    def test_a_budget_already_spent_is_not_replanned(self):
        with _env(True), self.assertLogs(SF.logger, level="WARNING") as cap:
            s, r, w = _planned(0)
            a = _adder(w, can_run=[object()])
            self.assertIsNone(SF.replan_hook(s, a)(r, TOLD))
        self.assertEqual(a.rem_chunk_tokens, 1024)
        self.assertIn("SKIPPED", cap.output[0])

    def test_another_request_is_not_replanned(self):
        with _env(True):
            s, r, w = _planned(0)
            a = _adder(w)
            self.assertIsNone(SF.replan_hook(s, a)(_breq("weg2-69-236"), TOLD))
        self.assertEqual(a.rem_chunk_tokens, 1024)

    def test_the_plan_serves_one_adder_and_one_pass(self):
        with _env(True):
            s, r, w = _planned(0)
            self.assertIsNotNone(SF.replan_hook(s, _adder(w)))
            self.assertIsNone(SF.replan_hook(s, _adder(w)))  # consumed
            SF.note_budget(s, PBH.budget_head(s), w)
            SF.clear_budget(s)                               # next pass's sizing starts clean
            self.assertIsNone(SF.replan_hook(s, _adder(w)))

    def test_switch_off_installs_nothing(self):
        with _env(False):
            s, r, w = _planned(0)
            self.assertIsNone(SF.replan_hook(s, _adder(w)))
            self.assertIsNone(getattr(s, SF.PLAN_ATTR, None))

    def test_ranks_agree_after_the_move(self):
        # PP0 (second visit, planned from the kept told) and a follower (first visit,
        # planned from the told) both move to 40958: same budget, same planner calls.
        with _env(True):
            r0, r1 = _breq(), _breq()
            pp0 = _bsched(r0, told_map={}, kept=r0)
            pp1 = _bsched(r1, told_map={r1.rid: TOLD})
            a0, a1 = _adder(pp0._p_chunk_policy_width()), _adder(pp1._p_chunk_policy_width())
            for s, a, r in ((pp0, a0, r0), (pp1, a1, r1)):
                h = SF.replan_hook(s, a)
                if h is not None:
                    h(r, TOLD)
        self.assertEqual(a0.rem_chunk_tokens, a1.rem_chunk_tokens)
        self.assertEqual(pp0._p_chunk_planner.calls, pp1._p_chunk_planner.calls)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


class Wiring(CustomTestCase):
    def test_the_move_hook_runs_right_after_the_988_instrument(self):
        from sglang.srt.managers import schedule_policy as sp

        src = open(sp.__file__).read()
        i = src.index("_note_988_loadback(req, prefix_len)")
        blk = src[i:i + 600]
        self.assertIn('_sf_hook = getattr(self, "sf_replan_after_move", None)', blk)
        self.assertIn("_sf_hook(req, prefix_len)", blk)
        self.assertLess(blk.index("_sf_hook(req, prefix_len)"), blk.index("input_tokens = "))

    def test_the_scheduler_installs_the_hook_and_clears_the_plan(self):
        from sglang.srt.managers import scheduler as sch

        src = open(sch.__file__).read()
        i = src.index("adder.form_a_admission_follow = self._form_a_admission_follow_fn()")
        self.assertIn("_sf.replan_hook(self, adder)", src[i:i + 500])
        j = src.index("def dynamic_chunked_prefill_size(self)")
        k = src.index("_pls = _pls_rt.active()", j)
        self.assertIn("_sf.clear_budget(self)", src[j:k])

    def test_the_load_back_asks_before_the_old_floor_check(self):
        from sglang.srt.mem_cache import unified_radix_cache as urc

        src = open(urc.__file__).read()
        i = src.index("if floor < kv_tokens:")
        self.assertIn("_sf.local_pp_room(self, kv_tokens, floor", src[i - 700:i])


if __name__ == "__main__":
    unittest.main()
