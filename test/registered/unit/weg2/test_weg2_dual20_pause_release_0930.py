# SPDX-License-Identifier: Apache-2.0
"""Metal replay dual20 (qs8pr9, ...09301417 @6085673956): the first P-PAUSE
worked, and then:

(1) The paused leg came back from P as 200 with prompt_tokens=0. The front
    took it as served and handed B to D -> W50 -> 503 after 557 s. No
    P-PAUSED requeue.
(2) PP1/PP2 kept B as chunked_req. Their recorded abort waits for "PP0's
    forwarded schedule stops naming it", and PP0 sent no further frame. The
    followers were never idle and never released, so the pressure stayed
    [0, 855638016, 570425344] for 4 min. The ledger kept a high-water mark,
    and D had no way to take its pressure back once its seats were gone. No
    P pass started; a user's LONG request starved.
(3) Head-of-line blocking on D: a 358-token request waited 126 s behind a
    112k head that waited for KV growth (NO_TOKEN -> break). User rule: age
    has precedence, younger requests backfill, and displacement happens only
    when it is ENOUGH for the older one.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import types
import unittest.mock as mock
import uuid

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import d_park_runtime as DP
from sglang.srt.weg2 import dual_d_kv_stage as D
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import hol_overtake as H
from sglang.srt.weg2.d_seat_vram import AllocInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20


class PausedLegIsRequeued(CustomTestCase):
    def test_an_aborted_leg1_body_is_named(self):
        self.assertTrue(F.leg1_aborted({"choices": [{"finish_reason": {"type": "abort"}}]}, 129185))
        self.assertTrue(F.leg1_aborted({"meta_info": {"finish_reason": {"type": "abort"}}}, 5))
        self.assertTrue(F.leg1_aborted({}, 0))                       # dual20: prompt_tokens=0
        self.assertFalse(F.leg1_aborted({"choices": [{"finish_reason": "length"}]}, 129185))

    def test_the_drain_requeues_a_paused_leg_that_came_back_aborted(self):
        src = inspect.getsource(F)
        i = src.index('if getattr(p, "dual_pause", False):\n                        if getattr(p, "leg1_aborted", False):')
        self.assertIn("self._dual_requeue_paused(p)", src[i:i + 900])
        self.assertIn("p.leg1_aborted = leg1_aborted(js, pt)", src)


class FakeSpans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=264 * 32, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


class PressureFallsWithDemand(CustomTestCase):
    def _card(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkv20"), "card1")
        d, p = K.CardKvLedger(path, "D"), K.CardKvLedger(path, "P")
        d.contribute(4000 * MIB, committed=4000 * MIB)
        p.contribute(1000 * MIB)
        d.release(2000 * MIB)                                       # D holds 2000, budget 5000
        p.request(816 * MIB)                                        # P (B on PP1) holds ~855638016 B
        return path, d, p

    def test_the_pressure_is_the_current_shortfall_not_a_high_water_mark(self):
        path, d, p = self._card()
        d.request(3000 * MIB)                                       # D short -> pressure on P
        self.assertGreater(K.peek(path).pressure["P"], 0)
        d.release(K.peek(path).committed["D"] - 2000 * MIB)
        p.release(816 * MIB)                                        # P gave everything back
        d.request(100 * MIB)                                        # D asks again: granted in full
        self.assertEqual(K.peek(path).pressure["P"], 0, "a granted request left the old pressure standing")

    def test_d_takes_its_pressure_back_when_its_demand_fits(self):
        path, d, p = self._card()
        d.request(3000 * MIB)
        self.assertGreater(K.peek(path).pressure["P"], 0)
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        alloc = TokenToKVPoolAllocator(200, torch.float16, "cpu", None, False)
        geom = S._geom_for(torch.zeros(264, 8), 256, 1, "k", 264 * 32)
        a = D.DKvStage([(1, geom)], d, allocator=alloc, pools=[], page_size=1, granule=32, top_tokens=192,
                       spans=FakeSpans(), step=16, gmin=lambda v: v)
        a.mapped_tokens, a._committed = 96, a.bytes_for(96) - a.bytes_for(0)
        a._engage_cap(alloc, 96, 1)
        sched = types.SimpleNamespace(
            running_batch=types.SimpleNamespace(reqs=[]), chunked_req=None, waiting_queue=[], tree_cache=None,
            server_args=types.SimpleNamespace(chunked_prefill_size=16, speculative_num_draft_tokens=1),
            tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=a)),
            _weg2_group_min_ints=lambda v: list(v))
        D.tick(sched)                                               # the L seats were aborted: no demand
        self.assertEqual(K.peek(path).pressure["P"], 0, "D's pressure outlived its demand (dual20)")


class FollowerAppliesItsAbortWhenPP0IsIdle(CustomTestCase):
    def test_dual20_followers_release_b_after_pp0_went_idle(self):
        tag = "t-%s" % uuid.uuid4().hex[:8]
        env = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_KV_TAG": tag}
        applied = []

        def process():
            applied.append(bool(getattr(f, "_791c_pp0_drained", False)))
            f._pending_chunked_abort_req = None

        b = types.SimpleNamespace(rid="weg2-0-8")
        f = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1), _pending_chunked_abort_req=b,
                                  process_pending_chunked_abort=process,
                                  _pp_microbatches_drained=lambda: True)
        pp0 = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0))
        try:
            with mock.patch.dict(os.environ, env):
                self.assertFalse(S.follower_release_aborted_chunk(f, now=100.0))   # sees the abort
                S.mark_pp0_idle(pp0, now=99.0)                                     # idle BEFORE: no
                self.assertFalse(S.follower_release_aborted_chunk(f, now=101.0))
                S.mark_pp0_idle(pp0, now=102.0)                                    # PP0 idle after it
                self.assertTrue(S.follower_release_aborted_chunk(f, now=103.0))
            self.assertEqual(applied, [True], "the follower kept B's chunk (dual20)")
            self.assertIsNone(f._pending_chunked_abort_req)
        finally:
            try:
                os.unlink(S._idle_marker(tag))
            except OSError:
                pass

    def test_off_the_dual_p_follower_nothing(self):
        f = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1), _pending_chunked_abort_req=object())
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_LAYOUT": "", "SGLANG_WEG2_GROUP": "P"}):
            self.assertFalse(S.follower_release_aborted_chunk(f))

    def test_wired_into_on_idle(self):
        from sglang.srt.managers import scheduler as SC

        src = inspect.getsource(SC.Scheduler.on_idle)
        self.assertLess(src.index("follower_release_aborted_chunk(self)"), src.index("if not self.is_fully_idle():"))


class DAdmissionAgeBackfillDisplace(CustomTestCase):
    def test_displacement_only_when_it_is_enough(self):
        # the user's three cases (30.09.)
        self.assertEqual(DP.victims_needed(1000, 1500, [800, 900]), 0)       # fits free: nobody leaves
        self.assertEqual(DP.victims_needed(1000, 400, [700, 900]), 1)        # exactly one young seat
        self.assertIsNone(DP.victims_needed(5000, 400, [700, 900]))           # not even all: backfill stays
        self.assertEqual(DP.victims_needed(1500, 400, [700, 900]), 2)

    def test_kv_verdict_reads_the_younger_seats_only(self):
        def req(rid, n):
            return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(n)), output_ids=[],
                                         prefix_indices=[])

        older = req("weg2-0-6", 1000)
        sched = types.SimpleNamespace(
            waiting_queue=[older],
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 400),
            tree_cache=types.SimpleNamespace(evictable_size=lambda: 0))
        running_old_big = [req("weg2-0-5", 5000)]                              # older than it: never a victim
        self.assertFalse(DP.kv_displace_would_fit(sched, "weg2-0-6", running_old_big))
        self.assertTrue(DP.kv_displace_would_fit(sched, "weg2-0-6", running_old_big + [req("weg2-0-10", 700)]))

    def test_z30y6_tensor_fields_are_counted_not_truth_tested(self):
        # metal z30y6 16:47:04 (all D ranks, W17): prefix_indices is a torch tensor on D,
        # `len(x or ())` raised "Boolean value of Tensor with no values is ambiguous"
        older = types.SimpleNamespace(rid="weg2-0-6", origin_input_ids=list(range(1000)), output_ids=[],
                                      prefix_indices=torch.empty(0, dtype=torch.int64))
        young = types.SimpleNamespace(rid="weg2-0-10", origin_input_ids=torch.zeros(700, dtype=torch.int64),
                                      output_ids=[], prefix_indices=torch.arange(3))
        sched = types.SimpleNamespace(
            waiting_queue=[older],
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 400),
            tree_cache=types.SimpleNamespace(evictable_size=lambda: 0))
        self.assertTrue(DP.kv_displace_would_fit(sched, "weg2-0-6", [young]))
        self.assertTrue(DP.kv_displace_would_fit(sched, "weg2-0-6", [young], seat=True))
        older.prefix_indices = torch.arange(900)                     # a non-empty tensor: 100 left, fits free
        self.assertFalse(DP.kv_displace_would_fit(sched, "weg2-0-6", [young]))

    def test_backfill_gate_group_d_only_both_modes(self):
        s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1))
        for layout in ("1", ""):                                               # dual and flip form
            with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D", "SGLANG_WEG2_DUAL_LAYOUT": layout}):
                self.assertTrue(H.enabled(s))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "P"}):
            self.assertFalse(H.enabled(s))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D", H.ENV: "0"}):
            self.assertFalse(H.enabled(s))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            self.assertFalse(H.enabled(types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3))))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(H.enabled(s))                                     # classic boot: unchanged

    def test_a_no_token_head_no_longer_ends_the_round(self):
        s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            hp = H.HolPass(s)
            head = types.SimpleNamespace(rid="weg2-0-7")                      # the 112k head waiting for growth
            self.assertTrue(hp.may_overtake(head))                            # dual20: this was `break`
            self.assertEqual(hp.head, "weg2-0-7")
            for i in range(H.MAX_SCAN - 1):
                self.assertTrue(hp.may_overtake(types.SimpleNamespace(rid="x%d" % i)))
            self.assertFalse(hp.may_overtake(types.SimpleNamespace(rid="y")))  # bounded scan
        from sglang.srt.managers import scheduler as SC

        src = inspect.getsource(SC.Scheduler._get_new_batch_prefill_raw)
        i = src.index('_note_skip(f"add_result_{res.name}", req.rid)')
        self.assertIn("if _hol_go_on:\n                    continue\n                break", src[i:i + 200])
        j = src.index("_hol_go_on = _hol.may_overtake(req)")
        self.assertLess(j, src.index("elif self.enable_hierarchical_cache:", j))   # no batch_is_full on overtake


if __name__ == "__main__":
    import unittest

    unittest.main()
