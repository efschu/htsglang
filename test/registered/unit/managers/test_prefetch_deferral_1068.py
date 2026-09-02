"""#1068 slice 3, spec A12.2 (T21-T24) and the #969C L6 intake line (4.3).

THE RULE (A12.2): a `rate_limited` prefetch verdict is NEVER a decline. The
request keeps a pending-prefetch mark, stays queued in `kv_arrival_seq` order,
is NOT admitted to prefill while marked, and is retried by the scheduler on
every scheduling pass. Two bounded, NAMED exits:
  (1) '#1068 PREFETCH UNDEFERRABLE' -- the request's own span exceeds the
      budget, it can never land: falls to the ledger-cap degradation
      (recompute, counted);
  (2) '#1068 PREFETCH DEFER EXPIRED' -- waited longer than the length-priced
      prefetch timeout of #968/#1065 for its own span: admitted with
      reason=rate_expired (counted), so a stuck budget cannot hold the queue
      forever (wedge-freedom law).

RED on 846c6797b9 (parent): none of `_apply_prefetch_deferral`,
`_retry_deferred_prefetches`, `_admission_held_for_deferred_prefetch`,
`_deferred_prefetch_bound_s` exist on `Scheduler`; the 9th span is a plain
`declined:rate_limited` with no mark; the #969C line has no population=.

The REAL `Scheduler._add_request_to_queue` (and the slice-3 helpers) are
bound to a stand-in that supplies what they read. `_prefetch_kvcache` is a
stub with a fake occupancy counter, because the storage prefetch itself is
not under test here (the #915 gate has its own file).
"""

import inspect
import types
import unittest
from unittest import mock

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers.phase_purity import SEAM_READMIT_ATTR
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.match_refusal_census import PREFETCH_GATE_COUNTS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SPAN = 39365
LIMIT = 329589  # 0.9 x 366211 (spec section 5)
LOG = "sglang.srt.managers.scheduler"


def _method(name):
    fn = getattr(Scheduler, name, None)
    if fn is not None:
        return fn

    def _missing(self, *a, **k):
        raise AssertionError(f"Scheduler.{name} does not exist (slice 3 not built)")

    return _missing


class _HostPool:
    def __init__(self, size=366211, avail=366211):
        self.size = size
        self._avail = avail

    def available_size(self):
        return self._avail


class _CC:
    def __init__(self, pool, limit):
        self.mem_pool_host = pool
        self._limit = limit
        self.prefetch_tokens_occupied = 0

    @property
    def prefetch_capacity_limit(self):
        return self._limit


class _Tree:
    def __init__(self, limit=LIMIT, page_size=1, base=2.0, per_page=1.0 / 1024):
        self.cache_controller = _CC(_HostPool(), limit)
        self.ongoing_prefetch = {}
        self.page_size = page_size
        # the #968/#1065 length-priced timeout the tree carries
        self.prefetch_timeout_base = base
        self.prefetch_timeout_per_page = per_page


class _Req:
    def __init__(self, rid, seq=None, span=SPAN, stamped=False, population=None):
        self.rid = rid
        self.kv_arrival_seq = seq
        self._prefetch_span_tokens = span
        self.time_stats = types.SimpleNamespace(
            set_wait_queue_entry_time=lambda: None,
            set_retract_time=lambda: None,
        )
        self.host_hit_length = 0
        self.prefix_indices = []
        self.storage_hit_length = 0
        self.last_host_node = None
        if stamped:
            setattr(self, SEAM_READMIT_ATTR, 7)
        if population is not None:
            self._969c_population = population

    def finished(self):
        return False


class _Occupancy:
    """A fake budget: issue while occupied + span fits, else rate-limit."""

    def __init__(self, limit=LIMIT):
        self.limit = limit
        self.occupied = 0

    def __call__(self, req):
        span = req._prefetch_span_tokens
        if self.occupied + span <= self.limit:
            self.occupied += span
            return "issued"
        return "declined:rate_limited"

    def complete_one(self, span=SPAN):
        self.occupied -= span


class _Intake:
    _add_request_to_queue = Scheduler._add_request_to_queue
    readmit_seam_residents = Scheduler.readmit_seam_residents
    _apply_prefetch_deferral = _method("_apply_prefetch_deferral")
    _retry_deferred_prefetches = _method("_retry_deferred_prefetches")
    _deferred_prefetch_bound_s = _method("_deferred_prefetch_bound_s")
    _prefetch_capacity_limit_or_none = _method("_prefetch_capacity_limit_or_none")
    _admission_held_for_deferred_prefetch = _method("_admission_held_for_deferred_prefetch")
    _prefetch_deferral_enabled = _method("_prefetch_deferral_enabled")
    _host_pool_available_size = _method("_host_pool_available_size")
    _host_pool_identity = _method("_host_pool_identity")

    def __init__(self, verdict, limit=LIMIT, pp_size=1, pp_rank=0):
        self.waiting_queue = []
        self._kv_arrival_ct = 0
        self.disaggregation_mode = DisaggregationMode.NULL
        self.max_queued_requests = None
        self.enable_priority_scheduling = False
        self.enable_hicache_storage = True
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1)
        self.forward_ct = 3
        self.tree_cache = _Tree(limit=limit)
        self.retract_sites = []
        self.prefetch_calls = []
        self._verdict = verdict

    def _set_or_validate_priority(self, req):
        return True

    def _abort_on_queued_limit(self, req):
        return False

    def _prefetch_kvcache(self, req):
        self.prefetch_calls.append(req.rid)
        return self._verdict(req)

    def _969ad_note_retract(self, req, site):
        self.retract_sites.append((req.rid, site))


class _Clean(CustomTestCase):
    def setUp(self):
        PREFETCH_GATE_COUNTS.clear()

    def tearDown(self):
        PREFETCH_GATE_COUNTS.clear()


class TestRateLimitedIsDeferredNotDeclined(_Clean):
    def test_the_ninth_span_is_deferred_and_lands_after_one_completion(self):
        # T21: nine spans of 39365 against limit 329589 = 8.37 spans.
        occ = _Occupancy()
        s = _Intake(occ)
        reqs = [_Req(f"r{i}", seq=i) for i in range(9)]
        with self.assertLogs(LOG, level="INFO") as caught:
            for r in reqs:
                s._add_request_to_queue(r)
        for r in reqs[:8]:
            self.assertEqual(r._969c_verdict, "issued")
            self.assertIsNone(getattr(r, "prefetch_deferred", None))
        ninth = reqs[8]
        self.assertEqual(
            getattr(ninth, "prefetch_deferred", None),
            "rate_limited",
            "a rate_limited verdict is NEVER a decline: the request must "
            "carry the pending-prefetch mark",
        )
        self.assertEqual(ninth.prefetch_defer_attempts, 1)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("deferred", 0), 1)
        deferred_lines = [ln for ln in caught.output if "#1068 PREFETCH DEFERRED" in ln]
        self.assertEqual(len(deferred_lines), 1, caught.output)
        self.assertIn("rid=r8", deferred_lines[0])
        self.assertIn("reason=rate_limited", deferred_lines[0])
        self.assertIn("limit=329589", deferred_lines[0])
        self.assertIn("attempt=1", deferred_lines[0])
        self.assertEqual(len(s.waiting_queue), 9, "it stays queued")

        # a retry pass with the budget still full: still deferred, no second
        # DEFERRED line (edge-triggered), attempts advance
        with self.assertLogs(LOG, level="DEBUG") as caught2:
            sched_mod.logger.debug("probe: retry pass")
            n = s._retry_deferred_prefetches()
        self.assertEqual(n, 1)
        self.assertEqual(ninth.prefetch_deferred, "rate_limited")
        self.assertEqual(ninth.prefetch_defer_attempts, 2)
        self.assertEqual([ln for ln in caught2.output if "PREFETCH DEFERRED" in ln], [])

        # one span completes -> the next pass registers the ninth
        occ.complete_one()
        with self.assertLogs(LOG, level="INFO") as caught3:
            s._retry_deferred_prefetches()
        self.assertIsNone(ninth.prefetch_deferred)
        self.assertEqual(ninth._969c_verdict, "issued")
        self.assertEqual(PREFETCH_GATE_COUNTS.get("landed", 0), 1)
        landed = [ln for ln in caught3.output if "#1068 PREFETCH LANDED" in ln]
        self.assertEqual(len(landed), 1, caught3.output)
        self.assertIn("rid=r8", landed[0])
        self.assertIn("after_passes=2", landed[0])
        self.assertIn("waited_s=", landed[0])
        # the retry visited only the marked occupant, once per pass
        self.assertEqual(s.prefetch_calls[9:], ["r8", "r8"])

    def test_retries_run_in_arrival_order(self):
        s = _Intake(lambda req: "declined:rate_limited")
        for rid, seq in (("b", 5), ("a", 1), ("c", 9)):
            s._add_request_to_queue(_Req(rid, seq=seq))
        s.prefetch_calls.clear()
        s._retry_deferred_prefetches()
        self.assertEqual(s.prefetch_calls, ["a", "b", "c"])

    def test_a_non_rate_reason_on_retry_releases_the_mark_by_name(self):
        # the gate can move from rate_limited to another term (too_short /
        # anchor / host_pool_exhausted): the mark clears through a NAMED
        # exit, never silently, and the request proceeds on the normal path.
        verdicts = iter(["declined:rate_limited", "declined:store_absent"])
        s = _Intake(lambda req: next(verdicts))
        r = _Req("x", seq=1)
        s._add_request_to_queue(r)
        self.assertEqual(r.prefetch_deferred, "rate_limited")
        with self.assertLogs(LOG, level="WARNING") as caught:
            s._retry_deferred_prefetches()
        self.assertIsNone(r.prefetch_deferred)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_released", 0), 1)
        self.assertTrue(any("#1068 PREFETCH DEFER RELEASED" in ln for ln in caught.output))

    def test_the_readmit_wave_defers_the_ninth_too(self):
        # the cutover re-issue goes through the same intake: the 9th
        # re-issued request is deferred, not declined-into-recompute.
        occ = _Occupancy()
        s = _Intake(occ)
        residents = [_Req(f"r{i}", seq=i, stamped=True) for i in range(5)]
        s.waiting_queue = [_Req(f"q{i}", seq=10 + i) for i in range(4)]
        s.readmit_seam_residents(residents, requeue_waiting=True)
        marked = [r.rid for r in s.waiting_queue if getattr(r, "prefetch_deferred", None)]
        self.assertEqual(marked, ["q3"])
        self.assertEqual(s.last_seam_readmit["verdicts"], {"issued": 8, "declined:rate_limited": 1})


class TestTheHold(_Clean):
    def test_a_marked_request_is_held_from_prefill(self):
        # T22 (predicate half). The hold is PP0's / the single rank's: a
        # follower never withholds admission for its own prefetch (#969Z).
        s = _Intake(lambda req: "declined:rate_limited")
        r = _Req("x", seq=1)
        s._add_request_to_queue(r)
        self.assertTrue(s._admission_held_for_deferred_prefetch(r))
        plain = _Req("y", seq=2)
        self.assertFalse(s._admission_held_for_deferred_prefetch(plain))
        pp0 = _Intake(lambda req: "declined:rate_limited", pp_size=3, pp_rank=0)
        pp0._add_request_to_queue(r2 := _Req("z", seq=3))
        self.assertTrue(pp0._admission_held_for_deferred_prefetch(r2))
        follower = _Intake(lambda req: "declined:rate_limited", pp_size=3, pp_rank=1)
        follower._add_request_to_queue(r3 := _Req("w", seq=4))
        self.assertFalse(follower._admission_held_for_deferred_prefetch(r3))

    def test_the_admission_loop_consults_the_hold_before_admitting(self):
        # T22 (wiring half, can-fail: drop the hold from the loop -> red).
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        hold_at = src.find("_admission_held_for_deferred_prefetch(")
        admit_at = src.find("adder.add_one_req(")
        self.assertGreater(hold_at, -1, "the loop must consult the hold")
        self.assertGreater(admit_at, -1)
        self.assertLess(hold_at, admit_at, "held BEFORE add_one_req commits")
        retry_at = src.find("_retry_deferred_prefetches(")
        drain_at = src.find("_drain_prefetch_progress(")
        self.assertGreater(retry_at, -1, "every pass retries the marked occupants")
        self.assertLess(retry_at, drain_at, "retry before the progress drain")


class TestTheNamedExits(_Clean):
    def test_an_undeferrable_span_exits_by_name(self):
        # T23: span > limit can never land; no mark, counted, one line.
        s = _Intake(lambda req: "declined:rate_limited")
        big = _Req("big", seq=1, span=LIMIT + 1)
        with self.assertLogs(LOG, level="WARNING") as caught:
            s._add_request_to_queue(big)
        self.assertIsNone(getattr(big, "prefetch_deferred", None))
        self.assertEqual(PREFETCH_GATE_COUNTS.get("undeferrable", 0), 1)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("deferred", 0), 0)
        lines = [ln for ln in caught.output if "#1068 PREFETCH UNDEFERRABLE" in ln]
        self.assertEqual(len(lines), 1, caught.output)
        self.assertIn("rid=big", lines[0])
        self.assertIn(f"span={LIMIT + 1}", lines[0])
        self.assertIn(f"limit={LIMIT}", lines[0])
        self.assertEqual(len(s.waiting_queue), 1, "it is queued, admissible, recompute-bound")

    def test_expiry_exits_by_name_and_admits_with_rate_expired(self):
        # T24: the same bound #1065 uses: base + pages x per_page.
        clock = [1000.0]
        s = _Intake(lambda req: "declined:rate_limited")
        r = _Req("slow", seq=1)
        bound = s._deferred_prefetch_bound_s(SPAN)
        self.assertAlmostEqual(bound, 2.0 + SPAN * (1.0 / 1024), places=6)
        with mock.patch.object(sched_mod.time, "monotonic", lambda: clock[0]):
            s._add_request_to_queue(r)
            self.assertEqual(r.prefetch_deferred, "rate_limited")
            clock[0] += bound / 2
            s._retry_deferred_prefetches()
            self.assertEqual(r.prefetch_deferred, "rate_limited", "inside the bound: still held")
            clock[0] += bound / 2 + 1.0
            with self.assertLogs(LOG, level="WARNING") as caught:
                s._retry_deferred_prefetches()
        self.assertIsNone(r.prefetch_deferred)
        self.assertEqual(r.prefetch_defer_reason, "rate_expired")
        self.assertFalse(s._admission_held_for_deferred_prefetch(r))
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_expired", 0), 1)
        lines = [ln for ln in caught.output if "#1068 PREFETCH DEFER EXPIRED" in ln]
        self.assertEqual(len(lines), 1, caught.output)
        self.assertIn("rid=slow", lines[0])
        self.assertIn("waited_s=", lines[0])
        self.assertIn(f"bound_s={bound:.1f}", lines[0])
        self.assertIn("attempts=3", lines[0])

    def test_the_counters_partition(self):
        # count(DEFERRED) == LANDED + UNDEFERRABLE + DEFER_EXPIRED + DEFER_RELEASED
        # over the three exits above plus a landing.
        occ = _Occupancy()
        s = _Intake(occ)
        for i in range(9):
            s._add_request_to_queue(_Req(f"r{i}", seq=i))
        occ.complete_one()
        s._retry_deferred_prefetches()
        c = PREFETCH_GATE_COUNTS
        self.assertEqual(
            c.get("deferred", 0),
            c.get("landed", 0) + c.get("undeferrable", 0) + c.get("defer_expired", 0) + c.get("defer_released", 0),
        )
        self.assertEqual(c.get("landed", 0), 1)


class TestThe969CIntakeLine(_Clean):
    def test_l6_names_population_generation_pool_and_availability(self):
        s = _Intake(lambda req: "issued")
        resident = _Req("res", seq=1, stamped=True, population="retract")
        occupant = _Req("occ", seq=2, population="queue")
        plain = _Req("new", seq=3)
        with self.assertLogs(LOG, level="WARNING") as caught:
            sched_mod.logger.warning("probe: intake driven")
            s._add_request_to_queue(resident, is_retracted=True)
            s._add_request_to_queue(occupant, is_retracted=False)
            s._add_request_to_queue(plain)
        lines = [ln for ln in caught.output if "#969C READMIT-PREFETCH" in ln]
        self.assertEqual(len(lines), 2, caught.output)
        self.assertIn("population=retract", lines[0])
        self.assertIn("rid=res", lines[0])
        self.assertIn("verdict=issued", lines[0])
        self.assertIn("generation=", lines[0])
        self.assertIn("pool_id=", lines[0])
        self.assertIn("available_before=366211", lines[0])
        self.assertIn("readmit_epoch=7", lines[0])
        self.assertIn("population=queue", lines[1])
        self.assertIn("rid=occ", lines[1])
        self.assertEqual(resident._969ac_site, "retract-intake")
        self.assertEqual(occupant._969ac_site, "cutover-requeue")
        self.assertEqual(plain._969ac_site, "intake")
        # the population marker is consumed by the intake: a later ordinary
        # retraction of the same request must not print population=queue
        self.assertIsNone(occupant._969c_population)


if __name__ == "__main__":
    unittest.main()
