"""deskq 1507: ADMISSION-WEDGE false alarm after idle (NF boot 1005_185049).

Both verdicts aged a request from the LAST PROGRESS (classic: first-token clock; PREFILL-LIVELOCK: decode
clock), so a request arriving after a long idle met an old clock and the first 10 s poll that saw it alarmed.
Behind SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK (default off) the age of both verdicts and of the recovery driver is
counted from max(last progress, the poll that saw the queue/running set go from empty to non-empty).

DANGER DIRECTIONS: env off -> the pre-fix numbers, nothing written onto the scheduler; env on -> the arrival after
idle does not alarm before 20 s of its own waiting, and a stand that begins while the box is busy alarms on the
same poll as with the env off.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import time
import types
import unittest

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import invariant_checker as IC


class _Base(unittest.TestCase):
    def tearDown(self):  # Env.override() does not restore on an exception inside the block
        envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.clear()
        envs.SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS.clear()


def _sched(first_token, queued=0, running=0, decode=None, prefill=None):
    return types.SimpleNamespace(
        is_initializing=False,
        waiting_queue=[object()] * queued,
        running_batch=types.SimpleNamespace(reqs=[object()] * running),
        last_first_token_progress_time=first_token,
        last_prefill_progress_time=prefill,
        last_decode_progress_time=decode,
        forward_ct=0,
        _wedge_class_sample=None,
        weg2_dormant=False,
    )


def _set(s, queued=None, running=None):
    if queued is not None:
        s.waiting_queue = [object()] * queued
    if running is not None:
        s.running_batch = types.SimpleNamespace(reqs=[object()] * running)


class Alarm2ClassicAfterIdle(_Base):
    """HEALTH_CHECK after 90 s of idle: queue was empty, first-token clock 90 s old."""

    def _run(self, on):
        s = _sched(first_token=0.0)
        out = []
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on):
            out.append(IC.check_admission_wedge_once(s, now=90.0)[0])  # idle poll
            _set(s, queued=1)  # the request arrives
            for t in (95.0, 105.0, 114.0, 116.0, 126.0):  # polls every 10 s, +2 s jitter
                out.append(IC.check_admission_wedge_once(s, now=t)[0])
        return out

    def test_on_no_verdict_in_first_20s_then_yes(self):
        # first poll that sees the queue: t=95 (age 0); 105 (10 s); 114 (19 s); 116 (21 s) -> alarm
        self.assertEqual(self._run(True), [False, False, False, False, True, True])

    def test_off_is_the_old_false_alarm(self):
        self.assertEqual(self._run(False), [False, True, True, True, True, True])


class Alarm1PrefillLivelockAfterPause(_Base):
    """Burst 1 after a pause: 2 queued + 1 running, decode clock 100 s old (last decode round of burst 0)."""

    def _scenario(self, on):
        s = _sched(first_token=0.0, decode=0.0)
        out = []
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on):
            IC.check_admission_wedge_once(s, now=100.0)  # idle poll
            _set(s, queued=2, running=1)
            for t in (105.0, 115.0, 124.0, 126.0):
                out.append("PREFILL-LIVELOCK" in IC.check_admission_wedge_once(s, now=t)[1])
        return out

    def test_on_livelock_not_before_20s_of_own_waiting(self):
        self.assertEqual(self._scenario(True), [False, False, False, True])

    def test_off_is_the_old_false_alarm(self):
        self.assertEqual(self._scenario(False), [True, True, True, True])

    def test_a_group_that_never_decoded_is_still_not_judged(self):
        s = _sched(first_token=0.0, decode=None)
        _set(s, queued=2, running=1)
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True):
            for t in (100.0, 200.0, 400.0):
                self.assertNotIn("PREFILL-LIVELOCK", IC.check_admission_wedge_once(s, now=t)[1])


class RealWedgeNotLater(_Base):
    """A stand that begins while the box is busy is detected on the same poll as with the env off."""

    def _classic_polls(self, on):
        # busy from t=0, last first token at t=5, then nothing
        s = _sched(first_token=5.0, queued=1, decode=5.0)
        out = []
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on):
            for t in (0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0):
                out.append(IC.check_admission_wedge_once(s, now=t)[0])
        return out

    def test_classic_wedge_same_first_alarm_poll(self):
        # no first token since t=5 -> age>=20 at t=25 in both modes
        self.assertEqual(self._classic_polls(True), self._classic_polls(False))
        self.assertEqual(self._classic_polls(True), [False, False, False, False, True, True, True])

    def test_livelock_begun_while_busy_same_first_alarm_poll(self):
        def run(on):
            s = _sched(first_token=0.0, queued=2, running=3, decode=5.0)
            with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on):
                return [
                    "PREFILL-LIVELOCK" in IC.check_admission_wedge_once(s, now=t)[1]
                    for t in (0.0, 10.0, 24.0, 26.0, 40.0)
                ]

        self.assertEqual(run(True), run(False))
        self.assertEqual(run(True), [False, False, False, True, True])

    def test_arrival_with_nothing_served_still_alarms_after_20s(self):
        s = _sched(first_token=0.0)
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True):
            IC.check_admission_wedge_once(s, now=100.0)
            _set(s, queued=1)
            self.assertFalse(IC.check_admission_wedge_once(s, now=110.0)[0])
            alarm, detail = IC.check_admission_wedge_once(s, now=131.0)
        self.assertTrue(alarm, detail)
        self.assertIn("QUEUE-CLOCK", detail)

    def test_running_request_is_never_the_classic_wedge(self):
        s = _sched(first_token=0.0, queued=1, running=1)
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True):
            IC.check_admission_wedge_once(s, now=100.0)
            self.assertFalse(IC.check_admission_wedge_once(s, now=300.0)[0])


class QueueEmptiesAgain(_Base):
    def test_idle_poll_drops_the_floor_and_a_new_arrival_gets_a_new_one(self):
        s = _sched(first_token=0.0, queued=1)
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True):
            IC.check_admission_wedge_once(s, now=100.0)
            self.assertEqual(s._wedge_busy_since, 100.0)
            _set(s, queued=0)
            IC.check_admission_wedge_once(s, now=110.0)
            self.assertIsNone(s._wedge_busy_since)
            _set(s, queued=1)
            self.assertFalse(IC.check_admission_wedge_once(s, now=140.0)[0], "a new arrival, a new clock")
            self.assertEqual(s._wedge_busy_since, 140.0)


class EnvOffUnchanged(_Base):
    def test_default_is_off(self):
        self.assertFalse(envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.get())

    def test_off_verdict_and_detail_are_the_old_ones_and_nothing_is_stamped(self):
        s = _sched(first_token=0.0, queued=1)
        alarm, detail = IC.check_admission_wedge_once(s, now=100.0)
        self.assertTrue(alarm, detail)
        self.assertNotIn("QUEUE-CLOCK", detail)
        self.assertFalse(hasattr(s, "_wedge_busy_since"))
        self.assertIn("queue age 100.0s since last first-token progress", detail)

    def test_off_livelock_detail_unchanged_and_nothing_stamped(self):
        s = _sched(first_token=99.0, queued=2, running=3, decode=0.0)
        alarm, detail = IC.check_admission_wedge_once(s, now=100.0)
        self.assertFalse(alarm)
        self.assertIn("PREFILL-LIVELOCK", detail)
        self.assertIn("NO decode round for 100.0s", detail)
        self.assertFalse(hasattr(s, "_wedge_busy_since"))


class RecoveryDriverUsesTheSameAge(_Base):
    def _post(self, on):
        from sglang.srt.managers.wedge_recovery import RECOVERY_CHANNEL_ATTR

        s = _sched(first_token=time.perf_counter() - 100.0, queued=1)
        s._wedge_busy_since = time.perf_counter() - 1.0
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on), envs.SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS.override(5.0):
            IC.AdmissionWedgeRecovery(s, clock=lambda: 1000.0).step(True)
        return getattr(s, RECOVERY_CHANNEL_ATTR, None)

    def test_on_a_1s_old_queue_posts_nothing(self):
        self.assertIsNone(self._post(True))

    def test_off_the_old_driver_posts(self):
        self.assertIsNotNone(self._post(False))


class StuckStateStamp1515(_Base):
    """deskq 1515 (gap 3b of 1512): the busy-set floor never saw the set empty, but the state the classic verdict
    needs (queued>0, running==0) is new.

    Request A decodes for minutes (running=1, first-token clock old, busy floor = start of A). A ends, request B
    arrives inside the same 10 s poll interval and waits 2-3 s in the queue; the poll that lands there sees
    queued=1, running=0 with the busy floor and the first-token clock both ~120 s old -> false alarm."""

    BUSY_POLLS = (0.0, 10.0, 20.0, 60.0, 100.0, 110.0)

    def _flow(self, on):
        s = _sched(first_token=0.0, running=1)
        out = []
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on):
            for t in self.BUSY_POLLS:  # A decodes, the set is never seen empty
                IC.check_admission_wedge_once(s, now=t)
            _set(s, queued=1, running=0)  # A ended at 115, B queued at 117
            for t in (120.0, 125.0, 139.0, 141.0, 150.0):
                out.append(IC.check_admission_wedge_once(s, now=t)[0])
        return out

    def test_a_ends_b_queues_within_one_poll_no_false_alarm(self):
        # state first seen at 120 -> 20 s of its own waiting -> alarm only from 141 on
        self.assertEqual(self._flow(True), [False, False, False, True, True])

    def test_off_is_the_old_false_alarm(self):
        self.assertEqual(self._flow(False), [True, True, True, True, True])

    def test_stuck_stamp_is_dropped_when_the_state_ends(self):
        s = _sched(first_token=0.0, running=1)
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True):
            IC.check_admission_wedge_once(s, now=0.0)
            self.assertIsNone(getattr(s, "_wedge_stuck_since", None))
            _set(s, queued=1, running=0)
            IC.check_admission_wedge_once(s, now=100.0)
            self.assertEqual(s._wedge_stuck_since, 100.0)
            _set(s, running=1)  # B admitted
            IC.check_admission_wedge_once(s, now=110.0)
            self.assertIsNone(s._wedge_stuck_since)
            _set(s, queued=1, running=0)  # B done, C queued: a new entry, a new clock
            self.assertFalse(IC.check_admission_wedge_once(s, now=150.0)[0])
            self.assertEqual(s._wedge_stuck_since, 150.0)

    def test_real_wedge_after_a_ends_is_still_found_20s_after_its_first_poll(self):
        s = _sched(first_token=0.0, running=1)
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True):
            IC.check_admission_wedge_once(s, now=0.0)
            _set(s, queued=1, running=0)
            out = [IC.check_admission_wedge_once(s, now=t)[0] for t in (110.0, 120.0, 129.0, 131.0, 200.0, 400.0)]
        self.assertEqual(out, [False, False, False, True, True, True])

    def test_livelock_is_judged_by_the_busy_floor_not_the_stuck_stamp(self):
        def run(on):
            s = _sched(first_token=0.0, queued=2, running=3, decode=5.0)
            with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(on):
                got = [
                    "PREFILL-LIVELOCK" in IC.check_admission_wedge_once(s, now=t)[1]
                    for t in (0.0, 10.0, 24.0, 26.0, 40.0)
                ]
                return got, getattr(s, "_wedge_stuck_since", None)

        (on_polls, on_stuck), (off_polls, _) = run(True), run(False)
        self.assertEqual(on_polls, off_polls)
        self.assertEqual(on_polls, [False, False, False, True, True])
        self.assertIsNone(on_stuck)

    def test_off_nothing_is_stamped(self):
        s = _sched(first_token=0.0, queued=1)
        IC.check_admission_wedge_once(s, now=100.0)
        self.assertFalse(hasattr(s, "_wedge_stuck_since"))

    def test_recovery_driver_counts_from_the_stuck_stamp_too(self):
        from sglang.srt.managers.wedge_recovery import RECOVERY_CHANNEL_ATTR

        s = _sched(first_token=time.perf_counter() - 100.0, queued=1)
        s._wedge_busy_since = time.perf_counter() - 100.0  # A's start: old
        s._wedge_stuck_since = time.perf_counter() - 1.0  # the state is 1 s old
        with envs.SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK.override(True), envs.SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS.override(5.0):
            IC.AdmissionWedgeRecovery(s, clock=lambda: 1000.0).step(True)
        self.assertIsNone(getattr(s, RECOVERY_CHANNEL_ATTR, None))


if __name__ == "__main__":
    unittest.main()
