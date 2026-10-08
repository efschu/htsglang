# SPDX-License-Identifier: Apache-2.0
"""deskq 1988 (b) / report 1981: ADMISSION-WEDGE false alarm. Behind FLLIPER_ADMISSION_WEDGE_QUEUE_CLOCK (default off,
dual layout only) the age is min(D's last-first-token age, age of the non-empty queue) and a new alarm window clears
the recovery channel's stale last_outcome. A real stand (queued, 0 running, nothing for long) still alarms.

DANGER DIRECTIONS: switch off or flip form -> pre-fix verdict, nothing written onto the scheduler; armed -> the
arrival after a long decode does not alarm, the real stand does (FlipUnchanged1988 + the armed tests).
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import invariant_checker as IC  # noqa: E402
from flliper.srt.managers.wedge_recovery import RECOVERY_CHANNEL_ATTR, get_recovery_channel  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

QC = "FLLIPER_ADMISSION_WEDGE_QUEUE_CLOCK"
DUAL = "FLLIPER_PDFLIP_DUAL_LAYOUT"


class _Sched:
    """Only what check_admission_wedge_once / the recovery driver read."""

    def __init__(self, last_token, queued=0, running=0):
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = types.SimpleNamespace(reqs=[object()] * running)
        self.last_first_token_progress_time = last_token
        self.last_prefill_progress_time = None
        self.forward_ct = 0


def _env(on: bool, dual: bool = True):
    e = {QC: "1" if on else "0", DUAL: "1" if dual else "0"}
    return mock.patch.dict(os.environ, e)


class WedgeClockOff1981(CustomTestCase):
    def test_default_off_old_verdict_and_nothing_stamped(self):
        with _env(False):
            s = _Sched(last_token=0.0, queued=1)
            alarm, detail = IC.check_admission_wedge_once(s, now=100.0)
            self.assertTrue(alarm, detail)                       # the f11 false alarm shape, unchanged when off
            self.assertFalse(hasattr(s, "_wedge_queue_since"))
            self.assertNotIn("QUEUE-CLOCK", detail)

    def test_last_outcome_kept_when_off(self):
        with _env(False):
            s = _Sched(last_token=0.0, queued=1)
            ch = get_recovery_channel(s)
            ch.last_outcome = object()
            IC.check_admission_wedge_once(s, now=100.0)
            self.assertIsNotNone(ch.last_outcome)


class WedgeClockArmed1981(CustomTestCase):
    def test_arrival_after_long_decode_does_not_alarm_but_a_real_stand_does(self):
        with _env(True):
            s = _Sched(last_token=0.0, queued=0)                  # D decoded long, token clock is 100 s old
            alarm, _ = IC.check_admission_wedge_once(s, now=100.0)
            self.assertFalse(alarm)                               # empty queue: never an alarm
            s.waiting_queue = [object()]                          # the request arrives
            alarm, detail = IC.check_admission_wedge_once(s, now=105.0)
            self.assertFalse(alarm, detail)                       # f11: this poll used to alarm at once
            self.assertEqual(s._wedge_queue_since, 105.0)
            alarm, detail = IC.check_admission_wedge_once(s, now=115.0)
            self.assertFalse(alarm, detail)                       # queue age 10 s < 20 s
            alarm, detail = IC.check_admission_wedge_once(s, now=126.0)
            self.assertTrue(alarm, detail)                        # queued 21 s, 0 running, no token: REAL stand
            self.assertIn("ADMISSION-WEDGE", detail)
            self.assertIn("QUEUE-CLOCK", detail)

    def test_queue_emptying_drops_the_stamp(self):
        with _env(True):
            s = _Sched(last_token=0.0, queued=1)
            IC.check_admission_wedge_once(s, now=100.0)
            s.waiting_queue = []
            IC.check_admission_wedge_once(s, now=110.0)
            self.assertIsNone(s._wedge_queue_since)
            s.waiting_queue = [object()]
            alarm, _ = IC.check_admission_wedge_once(s, now=130.0)
            self.assertFalse(alarm, "a new arrival starts a new queue clock")

    def test_a_recent_token_still_wins(self):
        with _env(True):
            s = _Sched(last_token=95.0, queued=1)
            IC.check_admission_wedge_once(s, now=100.0)
            alarm, _ = IC.check_admission_wedge_once(s, now=120.0)   # token age 25 s, queue age 20 s
            self.assertTrue(alarm)                                  # min(25, 20) = 20 >= 20: still a stand

    def test_running_request_is_never_a_wedge(self):
        with _env(True):
            s = _Sched(last_token=0.0, queued=1, running=1)
            IC.check_admission_wedge_once(s, now=100.0)
            alarm, _ = IC.check_admission_wedge_once(s, now=300.0)
            self.assertFalse(alarm)

    def test_new_alarm_window_clears_a_stale_last_outcome(self):
        with _env(True):
            s = _Sched(last_token=0.0, queued=1)
            ch = get_recovery_channel(s)
            ch.last_outcome = types.SimpleNamespace(state="not_applicable")
            IC.check_admission_wedge_once(s, now=100.0)           # stamp
            alarm, detail = IC.check_admission_wedge_once(s, now=125.0)   # window starts
            self.assertTrue(alarm, detail)
            self.assertIsNone(ch.last_outcome, "the stale outcome of the previous window is gone")
            # an outcome recorded INSIDE the window survives the window's next polls
            ch.last_outcome = types.SimpleNamespace(state="actuated")
            IC.check_admission_wedge_once(s, now=135.0)
            self.assertIsNotNone(ch.last_outcome)

    def test_recovery_driver_uses_the_queue_age(self):
        # driver threshold 5 s: with the token age 100 s it would post at once; the queue is 1 s old -> no post
        with _env(True), mock.patch.dict(os.environ, {"FLLIPER_ADMISSION_WEDGE_RECOVERY_SECONDS": "5"}):
            s = _Sched(last_token=0.0, queued=1)
            s._wedge_queue_since = IC.time.perf_counter() - 1.0
            rec = IC.AdmissionWedgeRecovery(s, clock=lambda: 1000.0)
            rec.step(True)
            self.assertIsNone(getattr(s, RECOVERY_CHANNEL_ATTR, None), "no request posted on a 1 s old queue")
        with _env(False), mock.patch.dict(os.environ, {"FLLIPER_ADMISSION_WEDGE_RECOVERY_SECONDS": "5"}):
            s = _Sched(last_token=0.0, queued=1)
            s._wedge_queue_since = IC.time.perf_counter() - 1.0
            rec = IC.AdmissionWedgeRecovery(s, clock=lambda: 1000.0)
            rec.step(True)
            self.assertEqual(get_recovery_channel(s).requested_seq, 1, "off: the pre-fix driver posts")


class FlipUnchanged1988(CustomTestCase):
    def test_switch_on_without_the_dual_layout_is_the_old_verdict(self):
        with _env(True, dual=False):
            s = _Sched(last_token=0.0, queued=1)
            ch = get_recovery_channel(s)
            ch.last_outcome = object()
            alarm, detail = IC.check_admission_wedge_once(s, now=100.0)
            self.assertTrue(alarm, detail)
            self.assertFalse(hasattr(s, "_wedge_queue_since"))
            self.assertIsNotNone(ch.last_outcome)
            self.assertNotIn("QUEUE-CLOCK", detail)

    def test_env_unset_is_off(self):
        e = {k: v for k, v in os.environ.items() if k not in (QC, DUAL)}
        with mock.patch.dict(os.environ, e, clear=True):
            self.assertFalse(IC._wedge_queue_clock_armed())
