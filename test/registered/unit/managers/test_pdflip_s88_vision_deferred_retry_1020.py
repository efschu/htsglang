"""S88 (NF rc19 death 20.10. 07:18:44Z, DEADMAN_PROGRESS_STALL): a vision
request held told-pending whose intake store prefetch was deferred (#1068
A12.2) must stay inside the deferred-prefetch retry scan while the vision
stage parks it out of the waiting queue for the admission pass.

Metal (P log ..._10e9b2b492_1019_230333): pdflip-262-3338
  07:16:00 #1068 PREFETCH DEFERRED reason=rate_limited site=intake (P dormant)
  07:16:03 W102 PROBE-WAIT src=told-pending
  07:16:48 H91 ... waiting=1 decline=gate=batch_full_or_empty_queue(queue=0)
  -> no #1068 / LANDED / STALLED / DROPPED line for it ever again, the
     DEADMAN stopped the group.
Cycle: get_new_batch_prefill parks held vision requests (vision_rank_pass /
follower_pass) BEFORE `_get_new_batch_prefill_raw`, whose head runs
`_retry_deferred_prefetches` over `self.waiting_queue` only; pp0_publish skips
a marked rid ("PP0 will re-issue"), so no told ever came and the told-pending
wait never ended.

F1: the retry scans waiting_queue + the requests parked for this pass.
F2: S88 DEFERRED-UNREACHABLE -- a #1400-held marked request outside the scan
(and outside the dormant hold / settle) in two consecutive scans is a named
ERROR, once per rid, counted.
"""

import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_store_told as pst  # noqa: E402
from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.managers.scheduler import (  # noqa: E402
    _DEFER_REASON_RATE,
    _DEFER_REASON_SHORTFALL,
    _DEFER_REASON_STORE_SHORT,
    Scheduler,
)
from flliper.srt.mem_cache.match_refusal_census import PREFETCH_GATE_COUNTS  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

from test_prefetch_deferral_1068 import _Intake, _Occupancy, _Req  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

LOG = "flliper.srt.managers.scheduler"
RID = "pdflip-262-3338"


def _issued(req):
    return "issued"


def _mark(req, reason):
    """A standing #1068 mark as the intake leaves it."""
    import time

    req.prefetch_deferred = reason
    req.prefetch_defer_since = time.monotonic()
    req.prefetch_defer_attempts = 1
    return req


class _Pass(_Intake):
    """The real `get_new_batch_prefill` around a stand-in admission whose
    head runs the real retry (as `_get_new_batch_prefill_raw` does)."""

    get_new_batch_prefill = Scheduler.get_new_batch_prefill

    def __init__(self, verdict, **kw):
        super().__init__(verdict, **kw)
        self.pdflip_dormant = False
        self.prefill_delayer = None
        self.seen_in_pass = None
        self.queue_in_pass = None

    def _get_new_batch_prefill_raw(self, prefill_delayer_single_pass=None, running_batch=None):
        self.queue_in_pass = [r.rid for r in self.waiting_queue]
        if self.enable_hicache_storage:
            self._retry_deferred_prefetches()
        self.seen_in_pass = list(self.prefetch_calls)
        return None, running_batch


def _park_rids(*rids):
    """A vision park (follower_pass shape): pops the named rids out of the
    waiting queue and returns them as (index, req), like the real stage."""

    def _pass(scheduler):
        wq = scheduler.waiting_queue
        parked = [(i, r) for i, r in enumerate(wq) if r.rid in rids]
        for i, _ in reversed(parked):
            wq.pop(i)
        return parked

    return _pass


class _Clean(CustomTestCase):
    def setUp(self):
        PREFETCH_GATE_COUNTS.clear()

    def tearDown(self):
        PREFETCH_GATE_COUNTS.clear()


class TestParkedDeferredIsRetried(_Clean):
    """F1, one case per deferral reason (planer: rate_limited /
    host_pool_shortfall / store_short)."""

    def _one(self, reason):
        s = _Intake(_issued)
        req = _mark(_Req(RID, seq=7), reason)
        s.waiting_queue = []  # the vision stage holds it out of the queue
        s._pdflip_vision_parked_reqs = [req]
        n = s._retry_deferred_prefetches()
        self.assertEqual(
            s.prefetch_calls, [RID],
            f"a {reason} mark on a vision-parked request must be re-issued "
            "(S88: the retry scans the queue as it stood before the park)",
        )
        self.assertEqual(n, 1)
        return req

    def test_rate_limited(self):
        req = self._one(_DEFER_REASON_RATE)
        self.assertIsNone(getattr(req, "prefetch_deferred", None), "issued -> the mark is gone")

    def test_host_pool_shortfall(self):
        self._one(_DEFER_REASON_SHORTFALL)

    def test_store_short(self):
        self._one(_DEFER_REASON_STORE_SHORT)

    def test_rate_limited_from_a_real_intake_deferral(self):
        # the 3338 path end to end on the double: the intake defers on a
        # full budget, the budget frees, the request is parked by vision --
        # the next retry still lands it
        occ = _Occupancy(limit=10)
        s = _Intake(occ)
        req = _Req(RID, seq=1, span=20)
        s._add_request_to_queue(req)
        self.assertEqual(getattr(req, "prefetch_deferred", None), _DEFER_REASON_RATE)
        occ.limit = 1000
        s._pdflip_vision_parked_reqs = [s.waiting_queue.pop(0)]
        s._retry_deferred_prefetches()
        self.assertIsNone(getattr(req, "prefetch_deferred", None))
        self.assertEqual(req._969c_verdict, "issued")


class TestThroughTheRealAdmissionPass(_Clean):
    """F1 at the call site: the real `get_new_batch_prefill` parks, the retry
    inside the pass sees the parked request, the queue is restored and the
    park list does not outlive the pass."""

    def test_follower_park_is_scanned_and_restored(self):
        s = _Pass(_issued)
        a = _Req("pdflip-1-1", seq=1)
        v = _mark(_Req(RID, seq=2), _DEFER_REASON_RATE)
        b = _Req("pdflip-1-3", seq=3)
        s.waiting_queue = [a, v, b]
        s._pdflip_vision_follower_gate = True
        with mock.patch("flliper.srt.pdflip.vision_verdict.follower_pass", _park_rids(RID)):
            s.get_new_batch_prefill(running_batch=None)
        self.assertNotIn(RID, s.queue_in_pass, "the stand-in saw the park (precondition)")
        self.assertEqual(s.seen_in_pass, [RID], "the retry inside the pass reached the parked rid")
        self.assertEqual([r.rid for r in s.waiting_queue], ["pdflip-1-1", RID, "pdflip-1-3"])
        self.assertEqual(s._pdflip_vision_parked_reqs, [], "the park list lives for one pass")

    def test_rank_pass_park_is_scanned(self):
        s = _Pass(_issued)
        v = _mark(_Req(RID, seq=2), _DEFER_REASON_RATE)
        s.waiting_queue = [v]
        s._pdflip_vision_rank_stage = True
        with mock.patch("flliper.srt.pdflip.vision_rank_runner.vision_rank_pass", _park_rids(RID)):
            s.get_new_batch_prefill(running_batch=None)
        self.assertEqual(s.queue_in_pass, [])
        self.assertEqual(s.seen_in_pass, [RID])
        self.assertEqual([r.rid for r in s.waiting_queue], [RID])

    def test_park_list_cleared_when_the_admission_raises(self):
        s = _Pass(_issued)
        v = _mark(_Req(RID, seq=2), _DEFER_REASON_RATE)
        s.waiting_queue = [v]
        s._pdflip_vision_follower_gate = True

        def _boom(**kw):
            raise RuntimeError("admission raised")

        s._get_new_batch_prefill_raw = _boom
        with mock.patch("flliper.srt.pdflip.vision_verdict.follower_pass", _park_rids(RID)):
            with self.assertRaises(RuntimeError):
                s.get_new_batch_prefill(running_batch=None)
        self.assertEqual(s._pdflip_vision_parked_reqs, [])
        self.assertEqual([r.rid for r in s.waiting_queue], [RID])

    def test_no_double_retry_after_unpark(self):
        # a retry outside the pass (the DQH path) after the unpark scans the
        # queue once -- the stale park list must not add the rid a second time
        s = _Pass(lambda r: "declined:rate_limited")
        v = _mark(_Req(RID, seq=2), _DEFER_REASON_RATE)
        s.waiting_queue = [v]
        s._pdflip_vision_follower_gate = True
        with mock.patch("flliper.srt.pdflip.vision_verdict.follower_pass", _park_rids(RID)):
            s.get_new_batch_prefill(running_batch=None)
        s.prefetch_calls.clear()
        s._retry_deferred_prefetches()
        self.assertEqual(s.prefetch_calls, [RID])


class TestLiveRc19Boot072658TextLandsVisionToo(_Clean):
    """Planer's second case, live on boot ..._10e9b2b492_1020_072658 (P PP0):
    07:34:44 #1068 PREFETCH DEFERRED rate_limited for pdflip-6-33 (text),
    pdflip-6-36 and pdflip-6-42 (vision, W102 told-pending); 07:34:46
    #1443 DORMANT-RELEASE, then `#1068 PREFETCH LANDED rid=pdflip-6-33
    after_passes=1` -- the vision rids got no retry for 3.6 min and left only
    through the FRONT's salvage: 07:38:10/17 P-LEG1-STALL -> P-INTAKE-STALL
    /abort_request on P + requeue (bound 180 s), double flip, `#1037 REQ
    RE-CONSTRUCTED instance=2`, a fresh intake read. That abort is NOT the
    exit under test: here the same pass that lands the text neighbour must
    land the vision one, on the same Req object, without any abort."""

    def test_text_and_vision_neighbours_land_in_the_same_pass(self):
        occ = _Occupancy(limit=10)
        s = _Pass(occ)
        text = _Req("pdflip-6-33", seq=33, span=20)
        v36 = _Req("pdflip-6-36", seq=36, span=20)
        v42 = _Req("pdflip-6-42", seq=42, span=20)
        for r in (text, v36, v42):
            s._add_request_to_queue(r)
            self.assertEqual(getattr(r, "prefetch_deferred", None), _DEFER_REASON_RATE)
        occ.limit = 1000  # the budget frees (6-33 landed after 2.2 s live)
        s.prefetch_calls.clear()  # the intake calls; the pass's retries follow
        s._pdflip_vision_follower_gate = True
        ids = (id(v36), id(v42))
        with mock.patch(
            "flliper.srt.pdflip.vision_verdict.follower_pass",
            _park_rids("pdflip-6-36", "pdflip-6-42"),
        ):
            s.get_new_batch_prefill(running_batch=None)
        self.assertEqual(s.queue_in_pass, ["pdflip-6-33"], "the two vision rids were parked (precondition)")
        self.assertIsNone(getattr(text, "prefetch_deferred", None), "the text neighbour lands (live: LANDED)")
        self.assertIsNone(getattr(v36, "prefetch_deferred", None), "the vision neighbour lands in the same pass")
        self.assertIsNone(getattr(v42, "prefetch_deferred", None), "the vision neighbour lands in the same pass")
        self.assertEqual(s.seen_in_pass, ["pdflip-6-33", "pdflip-6-36", "pdflip-6-42"])
        self.assertEqual((id(s.waiting_queue[1]), id(s.waiting_queue[2])), ids,
                         "same Req objects, no abort / re-construct")


class TestOrderAndSymmetry(_Clean):
    def test_kv_arrival_order_across_queue_and_park(self):
        s = _Intake(lambda r: "declined:rate_limited")
        q = _mark(_Req("pdflip-9-5", seq=5), _DEFER_REASON_RATE)
        p = _mark(_Req(RID, seq=2), _DEFER_REASON_RATE)
        s.waiting_queue = [q]
        s._pdflip_vision_parked_reqs = [p]
        s._retry_deferred_prefetches()
        self.assertEqual(s.prefetch_calls, [RID, "pdflip-9-5"])

    def test_pp0_and_follower_scan_the_same_set(self):
        # PP0 parks the told-pending vision rid, the follower's gate does
        # not: both must walk the same retry set (rank-identical by
        # construction, the property the retry's own comment relies on)
        def _rank(parked_rids):
            s = _Intake(lambda r: "declined:rate_limited")
            reqs = [_mark(_Req(f"pdflip-2-{i}", seq=i), _DEFER_REASON_RATE) for i in range(3)]
            s.waiting_queue = [r for r in reqs if r.rid not in parked_rids]
            s._pdflip_vision_parked_reqs = [r for r in reqs if r.rid in parked_rids]
            s._retry_deferred_prefetches()
            return s.prefetch_calls

        self.assertEqual(_rank({"pdflip-2-1"}), _rank(set()))


class TestDeferredUnreachableInvariant(_Clean):
    """F2: the named riegel. No self-heal, nothing decided by it."""

    def _sched(self, req, in_queue=False, dormant=False):
        s = _Intake(lambda r: "declined:rate_limited")
        s._pdflip_store_held = {req.rid: req}
        s.waiting_queue = [req] if in_queue else []
        if dormant:
            s.pdflip_dormant_hold = [req]
        return s

    def _lines(self, caught):
        return [ln for ln in caught.output if "S88 DEFERRED-UNREACHABLE" in ln]

    def test_two_consecutive_scans_raise_one_named_error(self):
        req = _mark(_Req(RID, seq=1), _DEFER_REASON_RATE)
        req.finished_reason = None
        s = self._sched(req)
        with self.assertLogs(LOG, level="INFO") as caught:
            sched_mod.logger.info("probe")
            s._retry_deferred_prefetches()
            self.assertEqual(self._lines(caught), [], "one scan alone is the abort window")
            s._retry_deferred_prefetches()
            s._retry_deferred_prefetches()
        lines = self._lines(caught)
        self.assertEqual(len(lines), 1, caught.output)
        self.assertIn("ERROR", lines[0])
        self.assertIn(f"rid={RID}", lines[0])
        self.assertIn("reason=rate_limited", lines[0])
        self.assertEqual(s._s88_deferred_unreachable, 1)
        self.assertEqual(req.prefetch_deferred, _DEFER_REASON_RATE, "no self-heal")

    def test_silent_when_reachable_or_parked_by_the_hold(self):
        for kw in ({"in_queue": True}, {"dormant": True}):
            req = _mark(_Req(RID, seq=1), _DEFER_REASON_RATE)
            s = self._sched(req, **kw)
            with self.assertLogs(LOG, level="INFO") as caught:
                sched_mod.logger.info("probe")
                for _ in range(3):
                    s._retry_deferred_prefetches()
            self.assertEqual(self._lines(caught), [], kw)

    def test_silent_for_a_vision_parked_request(self):
        req = _mark(_Req(RID, seq=1), _DEFER_REASON_RATE)
        s = self._sched(req)
        with self.assertLogs(LOG, level="INFO") as caught:
            sched_mod.logger.info("probe")
            for _ in range(3):
                s._pdflip_vision_parked_reqs = [req]
                s._retry_deferred_prefetches()
        self.assertEqual(self._lines(caught), [])

    def test_silent_for_finished_or_unmarked(self):
        fin = _mark(_Req(RID, seq=1), _DEFER_REASON_RATE)
        fin.finished_reason = {"type": "abort"}
        plain = _Req("pdflip-0-2", seq=2)
        for req in (fin, plain):
            s = self._sched(req)
            with self.assertLogs(LOG, level="INFO") as caught:
                sched_mod.logger.info("probe")
                for _ in range(3):
                    s._retry_deferred_prefetches()
            self.assertEqual(self._lines(caught), [], req.rid)


class TestLandedMarkLetsTheToldOut(_Clean):
    """The cycle's other half: pp0_publish skips a marked held rid and
    publishes the told once the retry has landed the read."""

    def _publish(self, s):
        twin = types.SimpleNamespace(
            tick=lambda sch: None,
            release_due=lambda sch, rids: [],
            is_deferred=lambda sch, rid: False,
            take_pp0_twin=lambda sch, rid: None,
        )
        with mock.patch.object(pst, "_twin", twin), \
                mock.patch.object(pst, "_minifwd_hold", lambda *a: None), \
                mock.patch.object(pst, "_commit_park", lambda *a: None), \
                mock.patch.object(pst, "_p_held_short_reread", lambda *a: False), \
                mock.patch.object(pst, "_pp0_told_any", lambda *a: (55884, True)), \
                mock.patch.object(pst, "_ple_admit_at_told", lambda *a: None), \
                mock.patch.object(pst, "_pp0_keys_digest", lambda *a: ""), \
                mock.patch.object(pst, "_pp0_fork", lambda *a: None), \
                mock.patch.object(pst, "_with_fork", lambda obj, fork: obj):
            return pst.pp0_publish(s, [])

    def test_marked_no_told_then_landed_told(self):
        s = _Intake(_issued)
        req = _mark(_Req(RID, seq=1), _DEFER_REASON_RATE)
        s.waiting_queue = [req]
        s._pdflip_store_held = {RID: req}
        s._pdflip_store_told = {}
        s.tree_cache.check_prefetch_progress = lambda rid: True
        self.assertEqual(self._publish(s), [], "a marked rid gets no told (PP0 will re-issue)")
        # the pass parks it (vision told-pending) and the S88 scan re-issues
        s._pdflip_vision_parked_reqs = [s.waiting_queue.pop(0)]
        s._retry_deferred_prefetches()
        s._pdflip_vision_parked_reqs = []
        s.waiting_queue = [req]  # vision_unpark
        self.assertIsNone(getattr(req, "prefetch_deferred", None))
        out = self._publish(s)
        self.assertEqual([getattr(o, "rid", None) for o in out], [RID])
        self.assertEqual(s._pdflip_store_told.get(RID), 55884)


if __name__ == "__main__":
    unittest.main()
