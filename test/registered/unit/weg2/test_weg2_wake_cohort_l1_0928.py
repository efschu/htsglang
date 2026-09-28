"""L1 (28.09.): the post-wake cohort -- the reads of one wake join one extend.

NF rc12z26 D 18:06:01: six wake reads (0.1-0.7 s); the first one finished ran
its 2002 ms expert pass alone and the five others waited it out (harvest
2054-2167 ms). DANGER DIRECTIONS guarded here:
* a hold never outlasts HALF the measured extend price, nor 1.0 s;
* a wake of one, a cheap member (E2 skip / < 8 new tokens), an unknown record,
  or a sibling without a read in flight never waits;
* the vote rides the settle tick's ONE MIN with a rank-uniform element count,
  and any rank voting release releases the group (#580/#791).
"""
import inspect
import os
import time
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.srt.weg2 import wake_cohort as wc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

S = sched_mod.Scheduler


def _req(rid, n_ids, since=None):
    r = types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n_ids)))
    if since is not None:
        r._1471_since = since
    return r


def _records(**mat):
    return {rid: PrefetchOutcome(m, matched=0) for rid, m in mat.items()}


class _PriceReset(unittest.TestCase):
    def setUp(self):
        self._saved = dict(wc._PRICE)
        wc._PRICE.update(ms=None, n=0)

    def tearDown(self):
        wc._PRICE.clear()
        wc._PRICE.update(self._saved)


class TestHoldVote(_PriceReset):
    def test_expensive_ready_member_waits_for_a_sibling_in_flight(self):
        a, b = _req("a", 12829), _req("b", 19017)
        rec = _records(a=12800)                    # a: 29 new tokens -> the expensive pass
        self.assertEqual(wc.hold_vote([(a, "complete", True), (b, "reading", False)], 0.0, 0.2, rec), 1)

    def test_no_sibling_in_flight_or_a_writer_wait_releases(self):
        a, b = _req("a", 12829), _req("b", 19017)
        rec = _records(a=12800)
        self.assertEqual(wc.hold_vote([(a, "complete", True), (b, "complete", True)], 0.0, 0.2, rec), 0)
        self.assertEqual(wc.hold_vote([(a, "complete", True), (b, "wait", False)], 0.0, 0.2, rec), 0)

    def test_cheap_or_unknown_member_never_waits(self):
        a, b = _req("a", 12803), _req("b", 19017)
        self.assertEqual(wc.hold_vote([(a, "complete", True), (b, "reading", False)], 0.0, 0.2,
                                      _records(a=12800)), 0)        # 3 new tokens: a 12 ms pass
        self.assertEqual(wc.hold_vote([(a, "complete", True), (b, "reading", False)], 0.0, 0.2, {}), 0)

    def test_hold_never_outlasts_half_the_measured_price(self):
        a, b = _req("a", 12829), _req("b", 19017)
        m = [(a, "complete", True), (b, "reading", False)]
        rec = _records(a=12800)
        self.assertAlmostEqual(wc.cap_s(env={}), 0.75)              # unmeasured: half of 1.5 s
        wc.note_extend(3, 12.0)                                     # a cheap pass says nothing
        self.assertIsNone(wc._PRICE["ms"])
        wc.note_extend(29, 900.0)                                   # NF's LRU-aware eager plan
        self.assertAlmostEqual(wc.cap_s(env={}), 0.45)
        self.assertEqual(wc.hold_vote(m, 0.0, 0.44, rec, env={}), 1)
        self.assertEqual(wc.hold_vote(m, 0.0, 0.46, rec, env={}), 0)
        wc.note_extend(40, 5000.0)                                  # a slow pass: the 1.0 s cap holds
        self.assertLessEqual(wc.cap_s(env={}), 1.0)

    def test_a_wake_of_one_is_never_asked(self):
        self.assertFalse(wc.asks([_req("a", 10)], env={}))
        self.assertTrue(wc.asks([_req("a", 10), _req("b", 10)], env={}))
        self.assertFalse(wc.asks([_req("a", 10), _req("b", 10)], env={wc.ENV: "0"}))


def _holder(states, records, gmin=None):
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(weg2_hold_rids=set()),
                                         prefetch_loaded_tokens_by_reqid=records)
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    for n in ("_weg2_post_wake_settle_tick", "_weg2_group_min_flags", "_weg2_release_dormant_hold"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._weg2_refetch_one = lambda req, now, allow_reissue=True: states[req.rid]
    h.calls = []
    base = h._weg2_group_min_flags

    def spy(flags):
        h.calls.append(list(flags))
        out = base(flags)
        return gmin(out) if gmin else out
    h._weg2_group_min_flags = spy
    return h


class TestSettleTick(_PriceReset):
    def test_ready_member_joins_its_sibling_then_both_release_together(self):
        now = time.monotonic()
        a, b = _req("a", 12829, now), _req("b", 19017, now)
        states = {"a": "complete", "b": "reading"}
        h = _holder(states, _records(a=12800, b=18944))
        h.weg2_post_wake_settle = [a, b]
        self.assertEqual(h._weg2_post_wake_settle_tick(), 0)        # a held for b
        self.assertEqual([r.rid for r in h.weg2_post_wake_settle], ["a", "b"])
        states["b"] = "complete"
        self.assertEqual(h._weg2_post_wake_settle_tick(), 2)        # one cohort, one extend
        self.assertEqual([r.rid for r in h.waiting_queue], ["a", "b"])

    def test_vote_rides_the_same_min_with_a_uniform_count(self):
        now = time.monotonic()
        a, b = _req("a", 12829, now), _req("b", 19017, now)
        h = _holder({"a": "complete", "b": "reading"}, _records(a=12800))
        h.weg2_post_wake_settle = [a, b]
        h._weg2_post_wake_settle_tick()
        # the writer-veto MIN ([due..., decide...]) and ONE release MIN of 2 flags + the hold vote
        self.assertEqual([len(c) for c in h.calls], [4, 3])
        single = _holder({"a": "complete"}, _records(a=12800))
        single.weg2_post_wake_settle = [_req("a", 12829, now)]
        self.assertEqual(single._weg2_post_wake_settle_tick(), 1)   # a wake of one never waits
        self.assertEqual([len(c) for c in single.calls], [2, 1])

    def test_any_rank_voting_release_releases_the_group(self):
        now = time.monotonic()
        a, b = _req("a", 12829, now), _req("b", 19017, now)
        # a peer rank's clock lapsed: the MIN turns the hold element to 0
        h = _holder({"a": "complete", "b": "reading"}, _records(a=12800),
                    gmin=lambda v: v[:-1] + [0] if len(v) == 3 else v)
        h.weg2_post_wake_settle = [a, b]
        self.assertEqual(h._weg2_post_wake_settle_tick(), 1)
        self.assertEqual([r.rid for r in h.waiting_queue], ["a"])

    def test_a_raising_vote_releases_and_the_rank_lives(self):
        now = time.monotonic()
        a, b = _req("a", 12829, now), _req("b", 19017, now)
        h = _holder({"a": "complete", "b": "reading"}, _records(a=12800))
        h.weg2_post_wake_settle = [a, b]
        plain = wc.hold_vote

        def boom(*_a, **_k):
            raise RuntimeError("records unreadable")
        wc.hold_vote = boom
        try:
            with self.assertLogs(sched_mod.logger, level="WARNING") as cm:
                self.assertEqual(h._weg2_post_wake_settle_tick(), 1)   # released, not held, no raise
            self.assertEqual(sum("WEG2-WAKE-COHORT VOTE-ERROR" in m for m in cm.output), 1)
            self.assertEqual([len(c) for c in h.calls], [4, 3])        # the vote element still on the wire
            self.assertEqual([r.rid for r in h.waiting_queue], ["a"])
            # once per rid: the next tick raises again but names nobody new
            h.calls.clear()
            h._weg2_post_wake_settle_tick()
            self.assertEqual(sorted(h._weg2_cohort_vote_err_rids), ["a", "b"])
        finally:
            wc.hold_vote = plain

    def test_lapsed_cohort_releases_the_ready_member(self):
        a, b = _req("a", 12829, time.monotonic() - 5.0), _req("b", 19017, time.monotonic() - 5.0)
        h = _holder({"a": "complete", "b": "reading"}, _records(a=12800))
        h.weg2_post_wake_settle = [a, b]
        self.assertEqual(h._weg2_post_wake_settle_tick(), 1)

    def test_switch_off_releases_as_before(self):
        now = time.monotonic()
        a, b = _req("a", 12829, now), _req("b", 19017, now)
        h = _holder({"a": "complete", "b": "reading"}, _records(a=12800))
        h.weg2_post_wake_settle = [a, b]
        os.environ[wc.ENV] = "0"
        try:
            self.assertEqual(h._weg2_post_wake_settle_tick(), 1)
            self.assertEqual([len(c) for c in h.calls], [4, 2])
        finally:
            del os.environ[wc.ENV]


class TestWakeLedger(unittest.TestCase):
    def test_one_line_when_the_last_member_decodes(self):
        led = wc.WakeLedger()
        led.arm(7, ["a", "b"], 100.0)
        led.note_hold(100.2, True)
        led.note_hold(100.5, False)
        self.assertIsNone(led.decode_seen(["a"], 101.0, True))
        line = led.decode_seen(["b", "a"], 102.5, True)
        self.assertIn("wake=7 n=2 wake_to_last_decode_ms=2500 hold_ms=300 holds=1 cohort=on", line)
        self.assertIsNone(led.decode_seen(["a", "b"], 103.0, True))  # once per wake

    def test_wake_arms_the_ledger_with_the_whole_hold(self):
        a, b = _req("a", 10), _req("b", 10)
        h = _holder({"a": "complete", "b": "reading"}, {})
        h.weg2_dormant_hold = [a, b]
        h._weg2_release_dormant_hold()
        self.assertEqual(h._weg2_wake_ledger.rids, ["a", "b"])
        self.assertTrue(h._weg2_wake_ledger.open)


class TestWiring(unittest.TestCase):
    def test_price_is_fed_by_the_rank_batch_line(self):
        from sglang.srt.managers.scheduler_components import metrics_reporter as mr

        self.assertIn("_wc.note_extend(new_tokens, gpu_s * 1000.0)", inspect.getsource(mr))

    def test_decode_hook_and_partial_admission_census(self):
        src = inspect.getsource(sched_mod)
        self.assertIn("_weg2_wake_cohort_note(self, batch)", src)
        self.assertIn("admit_partial=%s", inspect.getsource(S._weg2_post_wake_pass_log))
        self.assertIn("self._admission_partial_note = ", src)


if __name__ == "__main__":
    unittest.main()
