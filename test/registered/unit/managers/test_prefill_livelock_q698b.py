"""Q-698b: the admission-wedge watchdog names a PREFILL LIVELOCK.

NF y9n abl 76163d3aef, boot ...10032328, D 23:33:24-23:44:15Z: 3 requests running, 2 queued,
every pass a 53/61-token re-extend of a resumed SEAT-AGE victim (782 DISPLACE passes on TP0),
0 decode rounds for 11 min -- and NO ADMISSION-WEDGE line: the classic verdict calls any running
request "serving", and every re-extend commits an output token, which stamps the first-token
clock. The decode round is the honest clock: running > 0, queued > 0, no decode round and no
middle prefill chunk for >= 20 s is a livelock. Report only (the corridor-relief recovery stays
on the classic alarm); P (never decodes) and a dormant D are never judged.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import time
import types
import unittest

from flliper.srt.managers.scheduler_components import invariant_checker as IC
from flliper.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)


class Verdict(unittest.TestCase):
    def test_the_10032328_specimen_alarms(self):
        alarm, detail = IC.prefill_livelock_verdict(2, 3, 651.0, 651.0)
        self.assertTrue(alarm, detail)
        self.assertIn("ADMISSION-WEDGE PREFILL-LIVELOCK", detail)
        self.assertIn("2 queued, 3 running", detail)

    def test_classic_verdict_is_blind_to_it(self):
        # first-token clock freshly stamped by the victim's re-extend, 3 running
        alarm, _ = IC.admission_wedge_verdict(2, 3, 0.8, seconds_since_prefill_progress=651.0)
        self.assertFalse(alarm)

    def test_decoding_box_is_silent(self):
        self.assertFalse(IC.prefill_livelock_verdict(2, 3, 0.2, 651.0)[0])

    def test_long_chunked_prefill_is_work(self):
        self.assertFalse(IC.prefill_livelock_verdict(2, 3, 90.0, 1.5)[0])

    def test_never_decoded_group_is_not_judged(self):
        alarm, detail = IC.prefill_livelock_verdict(5, 3, None, None)
        self.assertFalse(alarm)
        self.assertIn("no decode round", detail)

    def test_nothing_queued_or_nothing_running_is_not_this_class(self):
        self.assertFalse(IC.prefill_livelock_verdict(0, 3, 600.0, None)[0])
        self.assertFalse(IC.prefill_livelock_verdict(3, 0, 600.0, None)[0])


def _stub(queued, running, decode_age, dormant=False):
    now = time.perf_counter()
    return types.SimpleNamespace(
        is_initializing=False,
        waiting_queue=[object()] * queued,
        running_batch=types.SimpleNamespace(reqs=[object()] * running),
        last_first_token_progress_time=now - 0.8,
        last_prefill_progress_time=None,
        last_decode_progress_time=None if decode_age is None else now - decode_age,
        forward_ct=0,
        _wedge_class_sample=None,
        pdflip_dormant=dormant,
    )


class CallEdge(unittest.TestCase):
    def test_check_logs_the_livelock_but_does_not_arm_recovery(self):
        with self.assertLogs(IC.logger, level="ERROR") as cap:
            alarm, detail = IC.check_admission_wedge_once(_stub(2, 3, 651.0), log_on_alarm=True)
        self.assertFalse(alarm, "the recovery driver stays on the classic alarm")
        self.assertIn("PREFILL-LIVELOCK", detail)
        self.assertTrue(any("ADMISSION-WEDGE PREFILL-LIVELOCK" in m for m in cap.output))

    def test_dormant_d_is_silent(self):
        _, detail = IC.check_admission_wedge_once(_stub(2, 3, 651.0, dormant=True), log_on_alarm=True)
        self.assertNotIn("PREFILL-LIVELOCK", detail)

    def test_p_group_without_a_decode_clock_is_silent(self):
        _, detail = IC.check_admission_wedge_once(_stub(2, 3, None), log_on_alarm=True)
        self.assertNotIn("PREFILL-LIVELOCK", detail)


class Wiring(unittest.TestCase):
    def test_decode_result_stamps_the_clock_first(self):
        src = inspect.getsource(SchedulerBatchResultProcessor.process_batch_result_decode)
        body = src.split('):', 1)[1]
        self.assertTrue(body.strip().startswith("self.record_decode_progress()"), body[:200])

    def test_scheduler_wires_the_clock(self):
        from flliper.srt.managers import scheduler as S

        self.assertIn("record_decode_progress=self.note_decode_progress", inspect.getsource(S))
        s = types.SimpleNamespace()
        S.Scheduler.note_decode_progress(s, ts=12.5)
        self.assertEqual(s.last_decode_progress_time, 12.5)


if __name__ == "__main__":
    unittest.main()
