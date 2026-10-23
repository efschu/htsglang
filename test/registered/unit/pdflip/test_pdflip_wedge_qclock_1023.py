# SPDX-License-Identifier: Apache-2.0
"""27b-qclock-1023: ADMISSION-WEDGE false alarm "1 queued, 0 running, NO first token for 398.0s" (boot
dkr27bint4arbar1int4ar10231723, 23.10. 17:35:55Z, form pp=2, no FLLIPER_PDFLIP_DUAL_LAYOUT). Root: the queue clock
(FLLIPER_ADMISSION_WEDGE_QUEUE_CLOCK, default on) was ANDed with FLLIPER_PDFLIP_DUAL_LAYOUT=1, so on this form the
verdict aged the request from last_first_token_progress_time (seeded at scheduler init) across ~6 min of empty idle.
Fix: clock not bound to the dual layout; first-token age counts only from the arrival of the oldest waiting request.

DANGER DIRECTIONS: a real stand (queued, 0 running, 25 s after the arrival) must still alarm; the arrival after a long
idle must not; switch 0 = the pre-fix verdict.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.schedule_batch import Req  # noqa: E402
from flliper.srt.managers.scheduler_components import invariant_checker as IC  # noqa: E402
from flliper.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

QC = "FLLIPER_ADMISSION_WEDGE_QUEUE_CLOCK"
DUAL = "FLLIPER_PDFLIP_DUAL_LAYOUT"


def _prod_env():
    """The 23.10. boot's form: queue clock at its default, NO dual layout."""
    e = {k: v for k, v in os.environ.items() if k not in (QC, DUAL)}
    return mock.patch.dict(os.environ, e, clear=True)


def _req(recv):
    return types.SimpleNamespace(time_stats=types.SimpleNamespace(scheduler_recv_time=recv))


class _Sched:
    def __init__(self, last_token, queue=()):
        self.is_initializing = False
        self.waiting_queue = list(queue)
        self.running_batch = types.SimpleNamespace(reqs=[])
        self.last_first_token_progress_time = last_token
        self.last_prefill_progress_time = None
        self.forward_ct = 0


class QueueClockProductionForm1023(CustomTestCase):
    def test_idle_400s_then_arrival_5s_stand_no_alarm_real_stand_25s_alarms(self):
        with _prod_env():
            s = _Sched(last_token=0.0)                       # stamp seeded at init
            for now in (60.0, 200.0, 399.0):                 # ~400 s empty idle: polls see an empty queue
                alarm, _ = IC.check_admission_wedge_once(s, now=now)
                self.assertFalse(alarm)
            s.waiting_queue = [_req(400.0)]                  # the request arrives
            alarm, detail = IC.check_admission_wedge_once(s, now=405.0)
            self.assertFalse(alarm, detail)                  # 5 s old: the boot alarmed here with "398.0s"
            self.assertNotIn("398", detail)
            alarm, detail = IC.check_admission_wedge_once(s, now=412.0)
            self.assertFalse(alarm, detail)
            alarm, detail = IC.check_admission_wedge_once(s, now=426.0)
            self.assertTrue(alarm, detail)                   # 26 s after the arrival, nothing served: a real stand
            self.assertIn("ADMISSION-WEDGE", detail)

    def test_poll_missed_the_empty_gap_arrival_stamp_bounds_the_age(self):
        """The queue emptied and refilled between two polls: the poll-seen stamp is still old; the new request's own
        arrival time must start the clock (empty -> non-empty transition)."""
        with _prod_env():
            s = _Sched(last_token=0.0, queue=[_req(10.0)])
            IC.check_admission_wedge_once(s, now=11.0)       # stamp = 11.0, queue never seen empty afterwards
            s.waiting_queue = [_req(500.0)]                  # old one served in between, a new one arrived at 500
            alarm, detail = IC.check_admission_wedge_once(s, now=505.0)
            self.assertFalse(alarm, detail)
            alarm, detail = IC.check_admission_wedge_once(s, now=526.0)
            self.assertTrue(alarm, detail)

    def test_oldest_of_several_waiting_requests_counts(self):
        with _prod_env():
            s = _Sched(last_token=0.0, queue=[_req(300.0), _req(390.0)])
            IC.check_admission_wedge_once(s, now=300.0)
            alarm, detail = IC.check_admission_wedge_once(s, now=325.0)
            self.assertTrue(alarm, detail)

    def test_entries_without_a_stamp_fall_back_to_the_poll_stamp(self):
        with _prod_env():
            s = _Sched(last_token=0.0, queue=[object()])
            alarm, detail = IC.check_admission_wedge_once(s, now=400.0)
            self.assertFalse(alarm, detail)
            alarm, detail = IC.check_admission_wedge_once(s, now=426.0)
            self.assertTrue(alarm, detail)

    def test_switch_zero_is_the_pre_fix_verdict(self):
        e = {k: v for k, v in os.environ.items() if k != DUAL}
        e[QC] = "0"
        with mock.patch.dict(os.environ, e, clear=True):
            s = _Sched(last_token=0.0, queue=[_req(400.0)])
            alarm, detail = IC.check_admission_wedge_once(s, now=405.0)
            self.assertTrue(alarm, detail)
            self.assertIn("405.0s", detail)


def _real_req(rid, time_stats=None):
    sp = SamplingParams(max_new_tokens=8, temperature=0)
    sp.normalize(None)
    return Req(rid=rid, origin_input_text="", origin_input_ids=[1, 2, 3], sampling_params=sp, vocab_size=32000,
               time_stats=time_stats)


class RebuiltReqKeepsItsArrival1023(CustomTestCase):
    """Review risk: Req.__init__ overwrote the carried scheduler_recv_time, so a wedge whose request is rebuilt each
    cycle (resubmit / rebootstrap, scheduler.py:8117/11117) restarted its age every cycle and never alarmed."""

    def test_fresh_req_arrival_is_now(self):
        r = _real_req("a")
        self.assertGreater(r.wedge_arrival_time, 0.0)
        self.assertEqual(r.wedge_arrival_time, r.time_stats.scheduler_recv_time)

    def test_rebuilt_req_keeps_the_original_arrival_but_metrics_stamp_is_still_reset(self):
        first = _real_req("a")
        first.time_stats.scheduler_recv_time = first.time_stats.first_scheduler_recv_time = 100.0  # arrived long ago
        again = _real_req("a", time_stats=first.time_stats)  # rebuilt from the carried time_stats
        self.assertEqual(again.wedge_arrival_time, 100.0)
        self.assertGreater(again.time_stats.scheduler_recv_time, 100.0)   # the metrics stamp keeps its old behaviour
        third = _real_req("a", time_stats=again.time_stats)
        self.assertEqual(third.wedge_arrival_time, 100.0)    # and again after a second rebuild

    def test_a_rebuild_loop_still_alarms(self):
        with _prod_env():
            q = _real_req("w")
            q.time_stats.first_scheduler_recv_time = q.wedge_arrival_time = 1000.0
            s = _Sched(last_token=0.0, queue=[q])
            IC.check_admission_wedge_once(s, now=1000.0)
            for now in (1005.0, 1010.0, 1015.0, 1019.0):     # the Req is rebuilt every cycle from the carried stats
                q = _real_req("w", time_stats=q.time_stats)
                s.waiting_queue = [q]
                alarm, detail = IC.check_admission_wedge_once(s, now=now)
                self.assertFalse(alarm, detail)              # < 20 s since the ORIGINAL arrival
            q = _real_req("w", time_stats=q.time_stats)
            s.waiting_queue = [q]
            alarm, detail = IC.check_admission_wedge_once(s, now=1026.0)
            self.assertTrue(alarm, detail)                   # 26 s: the loop no longer hides the stand

    def test_queue_beyond_the_scan_cap_drops_the_arrival_clamp(self):
        with _prod_env():
            s = _Sched(last_token=0.0, queue=[_req(5000.0)] * (IC._WEDGE_ARRIVAL_SCAN_CAP + 1))
            self.assertIsNone(IC._oldest_waiting_arrival(s))
            IC.check_admission_wedge_once(s, now=5001.0)
            alarm, detail = IC.check_admission_wedge_once(s, now=5030.0)
            self.assertTrue(alarm, detail)                   # the poll stamp alone still ages a real stand
