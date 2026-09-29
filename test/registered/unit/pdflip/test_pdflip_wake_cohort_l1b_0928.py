"""L1b (28.09.): the post-wake cohort never held on the metal.

rc12z29b (f833fcbb2d) D TP0 from 20:11: ``holds=0`` in all 11 PDFLIP-WAKE-COHORT
lines. Flip ep18 (20:21:52): six wake reads, queue+read 0.3-1.3 s; pdflip-12-37
was ready 0.9 s after the wake and ran its 95-token pass alone (run_ms 2911),
the five others -- their reads done ~1 s after the wake -- were reaped only
after that pass (harvest 2967-3219 ms) and ran a second one (2482 ms):
wake_to_first_decode 7650 ms. Three reasons, one per assertion:

* the cap was measured FROM THE WAKE, so the ~1 s reads were charged to the
  hold (cap 0.75 s on the Form A workers, whose rank-batch line carries no
  gpu-ms and never measures the price) -- the hold lapsed before it began;
* a request P handed over and D never ran has an EMPTY
  ``full_untruncated_fill_ids``: its pass was priced at 0 tokens, cheap;
* a sibling seen short with its re-read due (``wait``) was not "in flight".
"""
import os
import time
import types
import unittest
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome
from flliper.srt.pdflip import wake_cohort as wc
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

S = sched_mod.Scheduler


def _records(**mat):
    return {rid: PrefetchOutcome(m, matched=0) for rid, m in mat.items()}


def _handoff(rid, n_prompt, since=None):
    """A request as P hands it over: the prompt, no output yet, no fill ids."""
    r = types.SimpleNamespace(rid=rid, origin_input_ids=list(range(n_prompt)), output_ids=[],
                              full_untruncated_fill_ids=array("q"))
    if since is not None:
        r._1471_since = since
    return r


def _holder(states, records):
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.pdflip_dormant_hold = []
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(pdflip_hold_rids=set()),
                                         prefetch_loaded_tokens_by_reqid=records)
    h.PDFLIP_POST_WAKE_SETTLE_S = S.PDFLIP_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    for n in ("_pdflip_post_wake_settle_tick", "_pdflip_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._pdflip_refetch_one = lambda req, now, allow_reissue=True: states[req.rid]
    return h


class TestL1b(unittest.TestCase):
    def setUp(self):
        self._saved = dict(wc._PRICE)
        wc._PRICE.update(ms=None, n=0)                             # a Form A worker: never measured

    def tearDown(self):
        wc._PRICE.clear()
        wc._PRICE.update(self._saved)

    def test_a_handoff_request_is_priced_by_its_prompt_not_its_empty_fill_ids(self):
        r = _handoff("pdflip-12-37", 20063)
        self.assertEqual(wc.new_tokens(r, _records(**{"pdflip-12-37": 19968})), 95)

    def test_a_sibling_whose_reread_is_due_is_in_flight(self):
        a, b = _handoff("a", 20063), _handoff("b", 26626)
        m = [(a, "complete", True), (b, "wait", False)]
        self.assertEqual(wc.hold_vote(m, 0.0, 0.2, _records(a=19968)), 1)

    def test_the_hold_clock_starts_when_the_first_member_is_ready_not_at_the_wake(self):
        wake = time.monotonic() - 0.9                               # ep18: ready 0.9 s after the wake
        a, b = _handoff("a", 20063, wake), _handoff("b", 26626, wake)
        states = {"a": "complete", "b": "reading"}
        h = _holder(states, _records(a=19968))
        h.pdflip_post_wake_settle = [a, b]
        self.assertEqual(h._pdflip_post_wake_settle_tick(), 0, "a must wait for its sibling's read")
        states["b"] = "complete"
        self.assertEqual(h._pdflip_post_wake_settle_tick(), 2)       # one cohort, one extend
        self.assertEqual([r.rid for r in h.waiting_queue], ["a", "b"])

    def test_the_hold_still_lapses_after_the_cap_from_the_first_ready(self):
        wake = time.monotonic() - 0.9
        a, b = _handoff("a", 20063, wake), _handoff("b", 26626, wake)
        h = _holder({"a": "complete", "b": "reading"}, _records(a=19968))
        h.pdflip_post_wake_settle = [a, b]
        self.assertEqual(h._pdflip_post_wake_settle_tick(), 0)
        h._pdflip_cohort_ready_at = (None, time.monotonic() - wc.cap_s() - 0.01)
        self.assertEqual(h._pdflip_post_wake_settle_tick(), 1)       # the cap bounds the hold


if __name__ == "__main__":
    unittest.main()
