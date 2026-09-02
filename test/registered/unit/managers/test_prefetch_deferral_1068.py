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

SLICE 3 FIX (review round 1 on 0ad85647cb), two blocking findings:
  (1) the mark had NO clearer at the cutover and the retry was gated only
      on enable_hicache_storage: a mark set in the PP phase survived the
      pp_to_tp readmit into the SYMMETRIC TP phase, where the deferral is
      refused, and the retry then called _prefetch_kvcache on the marked
      rank alone -- a rank-local walk into the #580 group vote.
      `TestTheMarkDoesNotSurviveTheCutover` is the reviewer's probe as a
      test: RED on 0ad85647cb (tp1 retried=1, tp0 retried=0).
  (2) the hold was the first deliberately rank-asymmetric gate in the
      admission loop, with the mark itself rank-local on followers.
      `TestFollowersCarryNoMark`: followers refuse the deferral BY NAME
      (reason=pp_follower), carry no mark, retry nothing, and the join that
      carries PP0's hold to them (#631 row authority) is pinned alive.
"""

import copy
import inspect
import os
import time
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
    def __init__(
        self,
        limit=LIMIT,
        page_size=1,
        base=2.0,
        per_page=1.0 / 1024,
        symmetric=False,
        priced=True,
    ):
        self.cache_controller = _CC(_HostPool(), limit)
        self.ongoing_prefetch = {}
        self.page_size = page_size
        # the #968/#1065 length-priced timeout the tree carries; a tree
        # without it (priced=False) is the HiRadixCache shape, which keeps
        # the terms in a config object under other names
        if priced:
            self.prefetch_timeout_base = base
            self.prefetch_timeout_per_page = per_page
        # the #580 participation mode: True = tp_world_size>1 under uneven
        # DCP, registration is a group vote (the TP phase of the flip boot)
        self._symmetric = symmetric

    def _hicache_prefetch_symmetric(self):
        return bool(self._symmetric)


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
    _prefetch_deferral_refusal_reason = _method("_prefetch_deferral_refusal_reason")
    _clear_prefetch_deferral_fields = _method("_clear_prefetch_deferral_fields")
    _drop_prefetch_deferral = _method("_drop_prefetch_deferral")
    _clear_prefetch_deferral_for_reissue = _method("_clear_prefetch_deferral_for_reissue")
    _host_pool_available_size = _method("_host_pool_available_size")
    _host_pool_identity = _method("_host_pool_identity")

    def __init__(
        self,
        verdict,
        limit=LIMIT,
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        symmetric=False,
        priced=True,
    ):
        self.waiting_queue = []
        self._kv_arrival_ct = 0
        self.disaggregation_mode = DisaggregationMode.NULL
        self.max_queued_requests = None
        self.enable_priority_scheduling = False
        self.enable_hicache_storage = True
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=tp_size)
        self.forward_ct = 3
        self.tree_cache = _Tree(limit=limit, symmetric=symmetric, priced=priced)
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
        # Every MARK has exactly one exit:
        #   count(DEFERRED) == LANDED + DEFER_EXPIRED + DEFER_RELEASED
        #                      + DEFER_DROPPED + DEFER_CLEARED_CUTOVER
        # UNDEFERRABLE is taken BEFORE a mark exists (the span alone exceeds
        # the limit) and therefore sits OUTSIDE the partition; it is counted,
        # never a member of it. (Slice-3 fix: the spec's A12.3 identity listed
        # UNDEFERRABLE inside the sum; as built that is impossible -- an
        # undeferrable request prints no DEFERRED line. Spec correction
        # line added.)
        occ = _Occupancy()
        s = _Intake(occ)
        for i in range(9):
            s._add_request_to_queue(_Req(f"r{i}", seq=i))
        occ.complete_one()
        s._retry_deferred_prefetches()
        c = PREFETCH_GATE_COUNTS

        def _exits():
            return (
                c.get("landed", 0)
                + c.get("defer_expired", 0)
                + c.get("defer_released", 0)
                + c.get("defer_dropped", 0)
                + c.get("defer_cleared_cutover", 0)
            )

        self.assertEqual(c.get("deferred", 0), _exits())
        self.assertEqual(c.get("landed", 0), 1)
        big = _Req("big", seq=99, span=LIMIT + 1)
        s._add_request_to_queue(big)
        self.assertEqual(c.get("undeferrable", 0), 1)
        self.assertEqual(c.get("deferred", 0), 1, "undeferrable never marked")
        self.assertEqual(c.get("deferred", 0), _exits())


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


def _plant_mark(req):
    """The fields exactly as `_apply_prefetch_deferral` leaves them after
    the first deferral, planted by hand where a test needs a mark on a rank
    that would never set one itself."""
    req.prefetch_deferred = "rate_limited"
    req.prefetch_defer_attempts = 1
    req.prefetch_defer_passes = 0
    req.prefetch_defer_since = time.monotonic()
    req.prefetch_defer_reason = None


def _assert_unmarked(tc, req):
    tc.assertIsNone(getattr(req, "prefetch_deferred", None))
    tc.assertIsNone(getattr(req, "prefetch_defer_since", None))
    tc.assertIsNone(getattr(req, "prefetch_defer_attempts", None))
    tc.assertIsNone(getattr(req, "prefetch_defer_passes", None))
    tc.assertFalse(getattr(req, "_prefetch_landed_hold_once", False))


class TestTheMarkDoesNotSurviveTheCutover(_Clean):
    """Review finding 1: the mark's LIFECYCLE across the cutover.

    WRITER: `_apply_prefetch_deferral` at intake/retry (PP phase here).
    SEPARATING EVENT: the pp_to_tp cutover -- `_reset_full` kills the
    operation that guarded the mark, `readmit_seam_residents` re-issues the
    queue on the TP stack whose tree is SYMMETRIC (#580 group vote).
    CLEARER (the fix): the readmit clears every deferral field of every
    population member BEFORE the re-issue; the fresh intake verdict decides
    anew. RETRY GATE (the fix): a mark that meets a refused deferral at the
    retry is DROPPED by name, never walked rank-locally into
    `prefetch_from_storage`.
    """

    def _pp_marks(self, rid="x"):
        pp = _Intake(lambda r: "declined:rate_limited", symmetric=False)
        req = _Req(rid, seq=1)
        pp._add_request_to_queue(req)
        self.assertEqual(req.prefetch_deferred, "rate_limited", "PP mark set")
        return req

    def test_the_readmit_clears_the_mark_on_every_rank_and_the_retry_calls_nobody(self):
        # The reviewer's probe (probe_mark_survives_cutover.py) as a test.
        # Two TP ranks see the SAME request object state (mark set) and reach
        # DIFFERENT local verdicts in the vote: rate_limited on the rank with
        # the small host pool, attempted_but_unregistered on the other.
        x = self._pp_marks()
        x_tp0 = copy.copy(x)
        tp1 = _Intake(lambda r: "declined:rate_limited", symmetric=True, tp_size=3)
        tp0 = _Intake(lambda r: "declined:attempted_but_unregistered", symmetric=True, tp_size=3)
        tp1.waiting_queue = [x]
        tp0.waiting_queue = [x_tp0]
        with self.assertLogs(LOG, level="INFO") as caught:
            tp1.readmit_seam_residents([], requeue_waiting=True)
            tp0.readmit_seam_residents([], requeue_waiting=True)
        for r in (x, x_tp0):
            _assert_unmarked(self, r)
        # the CLEARER ran, not the retry's drop gate: the census tells them apart
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_cleared_cutover", 0), 2)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_dropped", 0), 0)
        self.assertEqual(tp1.last_seam_readmit["deferral_cleared"], 1)
        self.assertEqual(tp0.last_seam_readmit["deferral_cleared"], 1)
        cleared = [ln for ln in caught.output if "#1068 PREFETCH DEFER CLEARED AT CUTOVER" in ln]
        self.assertEqual(len(cleared), 2, caught.output)
        self.assertIn("n=1", cleared[0])
        self.assertIn("rids=x", cleared[0])
        # the re-issue under the refused deferral leaves no mark either side
        self.assertEqual(x._969c_verdict, "declined:rate_limited")
        self.assertEqual(x_tp0._969c_verdict, "declined:attempted_but_unregistered")
        # next TP pass: the retry visits NOBODY on either rank -- the marked
        # set is empty, so no rank walks into the #580 vote alone
        tp1.prefetch_calls.clear()
        tp0.prefetch_calls.clear()
        self.assertEqual(tp1._retry_deferred_prefetches(), 0)
        self.assertEqual(tp0._retry_deferred_prefetches(), 0)
        self.assertEqual(tp1.prefetch_calls, [])
        self.assertEqual(tp0.prefetch_calls, [])
        self.assertFalse(tp1._prefetch_deferral_enabled())

    def test_the_readmit_clears_a_landed_hold_too(self):
        # `_prefetch_landed_hold_once` guards one more pass for a landing
        # whose operation the cutover also killed: cleared with the rest.
        pp = _Intake(lambda r: "issued", symmetric=False)
        r = _Req("h", seq=1)
        pp._add_request_to_queue(r)
        r._prefetch_landed_hold_once = True
        tp = _Intake(lambda r: "issued", symmetric=True, tp_size=3)
        tp.waiting_queue = [r]
        tp.readmit_seam_residents([], requeue_waiting=True)
        _assert_unmarked(self, r)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_cleared_cutover", 0), 1)
        self.assertFalse(tp._admission_held_for_deferred_prefetch(r))

    def test_a_mark_that_meets_a_refused_deferral_at_retry_is_dropped_by_name_never_retried(self):
        # Defense in depth behind the clearer: whatever planted a mark on a
        # rank whose deferral is refused, the retry must not run it.
        s = _Intake(lambda r: "declined:rate_limited", symmetric=True, tp_size=3)
        r = _Req("x", seq=1)
        s.waiting_queue = [r]
        _plant_mark(r)
        with self.assertLogs(LOG, level="WARNING") as caught:
            n = s._retry_deferred_prefetches()
        self.assertEqual(n, 0)
        self.assertEqual(s.prefetch_calls, [], "never rank-locally into prefetch_from_storage")
        _assert_unmarked(self, r)
        self.assertEqual(r.prefetch_defer_reason, "dropped:symmetric_vote")
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_dropped", 0), 1)
        lines = [ln for ln in caught.output if "#1068 PREFETCH DEFER DROPPED" in ln]
        self.assertEqual(len(lines), 1, caught.output)
        self.assertIn("rid=x", lines[0])
        self.assertIn("reason=symmetric_vote", lines[0])
        self.assertIn("site=retry", lines[0])
        self.assertIn("after_passes=", lines[0])
        self.assertFalse(s._admission_held_for_deferred_prefetch(r), "a dropped mark holds nothing")

    def test_the_refusal_is_read_live_at_every_retry(self):
        # A mark set while the deferral was enabled, then the tree turns
        # symmetric (the phase changed under it): the very next retry drops
        # the mark instead of retrying.
        s = _Intake(lambda r: "declined:rate_limited", symmetric=False)
        r = _Req("x", seq=1)
        s._add_request_to_queue(r)
        self.assertEqual(r.prefetch_deferred, "rate_limited")
        s.prefetch_calls.clear()
        s.tree_cache._symmetric = True
        self.assertEqual(s._retry_deferred_prefetches(), 0)
        self.assertEqual(s.prefetch_calls, [])
        self.assertIsNone(r.prefetch_deferred)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("defer_dropped", 0), 1)


class TestFollowersCarryNoMark(_Clean):
    """Review finding 2: the hold is PP0-authoritative, not rank-asymmetric.

    The verdict site is PP0 (#968). A follower refuses the deferral BY NAME
    (reason=pp_follower): it never sets a mark at intake, never retries, and
    drops a planted mark. Its membership comes from PP0's decision through
    the #631 row (`_pp_admission_incoming_effective`): PP0 skips the held rid
    before `add_one_req`, so `build_pp_admission_decision` (built from
    `can_run_list`) never names it, and the follower skips it as
    `pp_not_named`. There is no rank-local mark on a follower to converge.
    """

    def test_a_follower_never_marks_at_intake_and_never_retries(self):
        f = _Intake(lambda r: "declined:rate_limited", pp_size=3, pp_rank=1)
        r = _Req("w", seq=1)
        with self.assertLogs(LOG, level="INFO") as caught:
            f._add_request_to_queue(r)
        self.assertIsNone(getattr(r, "prefetch_deferred", None))
        self.assertEqual(r._969c_verdict, "declined:rate_limited")
        self.assertEqual(PREFETCH_GATE_COUNTS.get("deferred", 0), 0)
        self.assertEqual([ln for ln in caught.output if "PREFETCH DEFERRED" in ln], [])
        self.assertTrue(
            any("#1068 PREFETCH DEFERRAL REFUSED reason=pp_follower" in ln for ln in caught.output),
            caught.output,
        )
        self.assertEqual(f._prefetch_deferral_refusal_reason(), "pp_follower")
        # a planted mark on a follower is dropped by name, never retried
        _plant_mark(r)
        f.prefetch_calls.clear()
        with self.assertLogs(LOG, level="WARNING") as caught2:
            self.assertEqual(f._retry_deferred_prefetches(), 0)
        self.assertEqual(f.prefetch_calls, [])
        _assert_unmarked(self, r)
        self.assertTrue(
            any("DEFER DROPPED" in ln and "reason=pp_follower" in ln for ln in caught2.output),
            caught2.output,
        )
        self.assertFalse(f._admission_held_for_deferred_prefetch(r))

    def test_pp0_is_the_verdict_site(self):
        pp0 = _Intake(lambda r: "declined:rate_limited", pp_size=3, pp_rank=0)
        r = _Req("z", seq=1)
        pp0._add_request_to_queue(r)
        self.assertEqual(r.prefetch_deferred, "rate_limited")
        self.assertTrue(pp0._admission_held_for_deferred_prefetch(r))
        self.assertIsNone(pp0._prefetch_deferral_refusal_reason())
        single = _Intake(lambda r: "declined:rate_limited", pp_size=1, pp_rank=0)
        self.assertIsNone(single._prefetch_deferral_refusal_reason())

    def test_the_follower_join_that_carries_pp0s_hold_is_alive_in_the_tree(self):
        # (i) the row-authority predicate defaults ON for every multi-stage
        # PP form (pp_admission_congruence.pp_row_authority_enabled)
        from sglang.srt.managers.pp_admission_congruence import (
            pp_row_authority_enabled,
        )

        env = {k: v for k, v in os.environ.items() if k != "SGLANG_PP_ROW_AUTHORITY"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(
                pp_row_authority_enabled(types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3)))
            )
        # (ii) the downstream PP body WRITES the memo from the row: the
        # reconciled decision when a frame is pending, {} when provably none
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        body = inspect.getsource(SchedulerPPMixin._event_loop_pp_body)
        self.assertIn("self._pp_admission_incoming_effective = effective", body)
        self.assertIn("self._pp_admission_incoming_effective = {}", body)
        # (iii) the admission loop skips a rid PP0 did not name, under that
        # memo, and PP0's decision is built from can_run_list AFTER the loop
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        skip_at = src.find('_note_skip("pp_not_named"')
        self.assertGreater(skip_at, -1, "the follower must skip a rid PP0 did not name")
        guard_at = src.rfind("_pp_admission_incoming_effective is not None", 0, skip_at)
        self.assertGreater(guard_at, -1, "the skip hangs off the row memo")
        hold_at = src.find("_admission_held_for_deferred_prefetch(")
        build_at = src.find("build_pp_admission_decision(")
        self.assertGreater(build_at, hold_at, "PP0 decides AFTER the hold skipped the rid")
        # (iv) no stale claim that the memo is 'permanently None' may stand
        # beside the live writer: the two halves were reconciled in the
        # slice-3 fix (the claim dated from ff62244431, the writer from the
        # #631 row-authority commits 287d5d3946/e1da0a4d98 the same day)
        self.assertNotIn("permanently None", src)


class TestTheBoundIsPricedFromTheTree(_Clean):
    def test_a_tree_without_the_priced_timeout_refuses_deferral_by_name(self):
        # #606 form removed: the bound is read from the tree, never
        # defaulted. A tree that does not carry the #968/#1065 terms cannot
        # price the DEFER EXPIRED exit, so the deferral is refused by name
        # and rate_limited stays the pre-#1068 decline there.
        s = _Intake(lambda r: "declined:rate_limited", priced=False)
        r = _Req("u", seq=1)
        with self.assertLogs(LOG, level="WARNING") as caught:
            s._add_request_to_queue(r)
        self.assertIsNone(getattr(r, "prefetch_deferred", None))
        self.assertEqual(s._prefetch_deferral_refusal_reason(), "unpriced_timeout")
        self.assertTrue(
            any("#1068 PREFETCH DEFERRAL REFUSED reason=unpriced_timeout" in ln for ln in caught.output),
            caught.output,
        )
        self.assertEqual(PREFETCH_GATE_COUNTS.get("deferred", 0), 0)

    def test_the_bound_reads_the_tree_terms_without_defaults(self):
        s = _Intake(lambda r: "declined:rate_limited")
        s.tree_cache.prefetch_timeout_base = 1.0
        s.tree_cache.prefetch_timeout_per_page = 0.25
        self.assertAlmostEqual(s._deferred_prefetch_bound_s(SPAN), 1.0 + SPAN * 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
