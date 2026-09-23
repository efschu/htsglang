"""23.09. (fnFL2x36): the first P->D flip that ran to the end -- and D served
zero tokens, then died in the next cycle.

rid weg2-0-1 (8815 tokens) was HELD on dormant D during the flip; at the
wake its store read was one page short (8704 of 8768 deliverable: P
publishes the tail last), the top-up was ``need=64 < threshold=256`` and
refused as ``too_short`` on every re-issue.  The standstill pass counter
had climbed through the sleep, so the first observation after the wake was
terminal: W88 answered 503 with zero tokens at 08:27:37 -- but the request
stayed in the post-wake settle list, whose 20-s bound queued it "as it is"
at 08:27:57 into a group that had gone dormant again at 08:27:52.  The
extend ran on paused pools: Triton 'cpu tensor?' on TP1/TP2, illegal
memory access on TP0, D dead.

Four guards, one per link of that chain.
"""
import inspect
import os
import re
import time
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

S = sched_mod.Scheduler


def _holder(states=None):
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.weg2_post_wake_settle = []
    h.weg2_dormant = False
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(weg2_hold_rids=set()))
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    for n in ("_weg2_post_wake_settle_tick", "_weg2_group_min_flags", "_weg2_forget_held",
              "_weg2_note_prefetch_progress"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._weg2_refetch_one = lambda req, now, allow_reissue=True: (states or {})[req.rid]
    h._weg2_prefetch_stall_passes = lambda: 2
    return h


class SettleNeverReleasesIntoADormantGroup(unittest.TestCase):
    def test_a_lapsed_parked_request_stays_parked_while_dormant(self):
        r = types.SimpleNamespace(rid="b", _1471_since=0.0)  # lapsed long ago
        h = _holder({"b": "wait"})
        h.weg2_post_wake_settle = [r]
        h.weg2_dormant = True
        self.assertEqual(h._weg2_post_wake_settle_tick(), 0)
        self.assertEqual([x.rid for x in h.weg2_post_wake_settle], ["b"])
        self.assertEqual(h.waiting_queue, [])
        h.weg2_dormant = False  # the wake: the lapse releases as before (#1471)
        self.assertEqual(h._weg2_post_wake_settle_tick(), 1)
        self.assertEqual([x.rid for x in h.waiting_queue], ["b"])


class TheW88AnswerEndsTheRequestEverywhere(unittest.TestCase):
    def test_forget_held_purges_hold_settle_and_the_claim_set(self):
        r, other = types.SimpleNamespace(rid="a"), types.SimpleNamespace(rid="z")
        h = _holder()
        h.weg2_dormant_hold = [other, r]
        h.weg2_post_wake_settle = [r]
        h.tree_cache.cache_controller.weg2_hold_rids = {"a", "z"}
        h._weg2_forget_held(r, site="t")
        self.assertEqual([x.rid for x in h.weg2_dormant_hold], ["z"])
        self.assertEqual(h.weg2_post_wake_settle, [])
        self.assertEqual(h.tree_cache.cache_controller.weg2_hold_rids, {"z"})
        self.assertTrue(r._weg2_terminal)
        h._weg2_forget_held(r, site="t")  # idempotent

    def test_the_terminal_exit_calls_it_before_answering(self):
        src = inspect.getsource(S._weg2_store_load_terminal)
        self.assertIn("Scheduler._weg2_forget_held", src)
        self.assertLess(src.index("_forget(req, site="), src.index("send_output(abort_req"))


class TheStandstillCountStartsAtTheWake(unittest.TestCase):
    def _req(self):
        return types.SimpleNamespace(rid="a", _weg2_progress_terms=(0,) * 8,
                                     _weg2_best_delivered=0, _weg2_no_progress_passes=0)

    def test_passes_accrued_in_the_sleep_are_dropped(self):
        h = _holder()
        h._weg2_prefetch_progress_terms = lambda req: (0,) * 8
        r = self._req()
        r._weg2_no_progress_passes = 63
        h.weg2_dormant = True
        self.assertEqual(h._weg2_note_prefetch_progress(r), "stalled")
        self.assertEqual(r._weg2_no_progress_passes, 0)
        h.weg2_dormant = False
        h._weg2_last_wake_t = None
        # from the wake on the pass bound (2 here) applies again
        self.assertEqual(h._weg2_note_prefetch_progress(r), "stalled")
        self.assertEqual(h._weg2_note_prefetch_progress(r), "terminal")

    def test_the_wake_grace_holds_the_pass_bound(self):
        h = _holder()
        h._weg2_prefetch_progress_terms = lambda req: (0,) * 8
        r = self._req()
        h._weg2_last_wake_t = time.perf_counter()
        for _ in range(5):
            self.assertEqual(h._weg2_note_prefetch_progress(r), "stalled")
        h._weg2_last_wake_t = time.perf_counter() - 10_000.0  # grace long over
        self.assertEqual(h._weg2_note_prefetch_progress(r), "terminal")

    def test_the_resume_stamps_the_wake(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        src = inspect.getsource(wu)
        i = src.index("scheduler.weg2_dormant = False")
        self.assertIn("scheduler._weg2_last_wake_t = time.perf_counter()", src[i:i + 600])


class TheTopUpOfAHeldExtentIgnoresTheThreshold(unittest.TestCase):
    def test_held_rids_top_up_others_do_not(self):
        c = types.SimpleNamespace(cache_controller=types.SimpleNamespace(weg2_hold_rids={"weg2-0-1"}))
        f = types.MethodType(urc.UnifiedRadixCache._weg2_extent_topup, c)
        self.assertTrue(f("weg2-0-1"))
        self.assertFalse(f("weg2-2-2"))
        c.cache_controller.weg2_hold_rids = None
        self.assertFalse(f("weg2-0-1"))

    def test_the_gate_consults_it_on_the_too_short_term_only(self):
        src = inspect.getsource(urc.UnifiedRadixCache.prefetch_from_storage)
        self.assertRegex(src, r"elif prefetch_length < self\.prefetch_threshold and not _topup:")
        self.assertRegex(src, r'if not locally_eligible:\s*reason = "anchor"')


class AShortTailIsRecomputedNotReRead(unittest.TestCase):
    """fnFL2x38: the store never received the last page; every 2-s re-read
    answered zero and the request sat out the 20-s settle bound before the
    extend computed the 111-token tail in one pass."""

    def _holder(self, have, reason="zero-answer"):
        h = _holder()
        h.tree_cache = types.SimpleNamespace(
            cache_controller=types.SimpleNamespace(weg2_hold_rids=set()),
            check_prefetch_progress=lambda rid: True,
            prefetch_loaded_tokens_by_reqid={"a": have},
            ongoing_prefetch={})
        h.WEG2_TAIL_RECOMPUTE_TOKENS = S.WEG2_TAIL_RECOMPUTE_TOKENS
        h._weg2_note_store_shortfall = lambda req: reason
        h._prefetch_kvcache = lambda req: "issued"
        h._weg2_refetch_one = types.MethodType(S._weg2_refetch_one, h)
        return h

    def test_after_one_zero_re_read_a_page_tail_is_released(self):
        h = self._holder(have=8704)
        r = types.SimpleNamespace(rid="a", _prefetch_span_tokens=8768, _1456_n=1, _1471_short=True, _1456_last=0.0)
        self.assertEqual(h._weg2_refetch_one(r, now=100.0), "complete")
        self.assertFalse(r._1471_short)

    def test_before_the_first_re_read_and_for_a_long_tail_it_is_re_read(self):
        h = self._holder(have=8704)
        r = types.SimpleNamespace(rid="a", _prefetch_span_tokens=8768, _1456_n=0, _1471_short=True, _1456_last=0.0)
        self.assertEqual(h._weg2_refetch_one(r, now=100.0), "reissued")   # n was 0: read once first
        h2 = self._holder(have=4096)
        r2 = types.SimpleNamespace(rid="a", _prefetch_span_tokens=8768, _1456_n=3, _1471_short=True, _1456_last=0.0)
        self.assertEqual(h2._weg2_refetch_one(r2, now=100.0), "reissued")  # 4672 tokens: a real read


class TheWakeRearmsEveryModelOfTheRank(unittest.TestCase):
    """fnFL2x38: the rearm ran on `_weg2_model_for_group(group)`, which for D
    is the DRAFT; the MoE layers live in the TARGET. D woke with 0 rearmed
    layers (P: 3) and TP2 died of an illegal memory access in its first
    forward on P's expert rows."""

    def test_wake_models_are_target_then_draft_deduplicated(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        tgt, drf = object(), object()
        h = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=tgt)))
        h._weg2_model_for_group = lambda g: drf
        h._weg2_wake_models = types.MethodType(wu.SchedulerWeightUpdaterManager._weg2_wake_models, h)
        self.assertEqual(h._weg2_wake_models(), [tgt, drf])
        h._weg2_model_for_group = lambda g: tgt          # shared runner: once
        self.assertEqual(h._weg2_wake_models(), [tgt])
        h.tp_worker = None
        self.assertEqual(h._weg2_wake_models(), [tgt])

    def test_rearm_and_scratch_zero_iterate_the_wake_models(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        src = inspect.getsource(wu)
        i = src.index("_wake_models = self._weg2_wake_models()")
        blk = src[i:i + 3000]
        self.assertIn("for _m in _wake_models:\n                    _scratch.extend(zero_local_scratch(_m))", blk)
        self.assertIn("for _m in _wake_models:\n                _l, _z = rearm_expert_offload_after_wake(_m)", blk)
        self.assertNotIn("_m = self._weg2_model_for_group(self._weg2_group_name())", blk)


class ADormantGroupBuildsNoPrefillBatch(unittest.TestCase):
    def test_the_gate_is_the_first_thing_get_new_batch_prefill_does(self):
        src = inspect.getsource(S.get_new_batch_prefill)
        m = re.search(r'if getattr\(self, "weg2_dormant", False\) and self\.waiting_queue:', src)
        self.assertIsNotNone(m)
        self.assertLess(m.start(), src.index("_get_new_batch_prefill_raw"))
        self.assertIn("NextBatchPlan(batch_to_run=None, running_batch=running_batch)", src[m.start():m.start() + 900])


if __name__ == "__main__":
    unittest.main()
