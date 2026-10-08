"""deskq 52 (user decision 06.10.): the 20 s of the ADMISSION-WEDGE alarm (#699; it feeds the wedge status file,
the recovery and the intake-stall handover) and of PREFILL-LIVELOCK (Q-698b) become settable:

  FLLIPER_ADMISSION_WEDGE_SECONDS   (default 20.0)   seconds without a first token before the alarm
  FLLIPER_PREFILL_LIVELOCK_SECONDS  (default 20.0)   seconds without a decode round before the livelock line

Unset or non-positive = 20.0, i.e. the pre-change numbers byte for byte. The poll cadence (10 s) is not touched.

Red on the base (d10a4c3d60): the env is ignored there, so the moved-threshold cases alarm at 20 s anyway. The env is
set via ``mock.patch.dict(os.environ)`` so the file stays runnable against the base.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import os
import types
import unittest
from unittest import mock

from flliper.srt.managers.scheduler_components import invariant_checker as IC

NAMES = ("FLLIPER_ADMISSION_WEDGE_SECONDS", "FLLIPER_PREFILL_LIVELOCK_SECONDS")


def _env(**kv):
    clean = {k: v for k, v in os.environ.items() if k not in NAMES}
    clean.update(kv)
    return mock.patch.dict(os.environ, clean, clear=True)


def _sched(first_token=0.0, queued=0, running=0, decode=None, prefill=None):
    return types.SimpleNamespace(
        is_initializing=False,
        waiting_queue=[object()] * queued,
        running_batch=types.SimpleNamespace(reqs=[object()] * running),
        last_first_token_progress_time=first_token,
        last_prefill_progress_time=prefill,
        last_decode_progress_time=decode,
        forward_ct=0,
        _wedge_class_sample=None,
        pdflip_dormant=False,
    )


class AdmissionWedgeSeconds(unittest.TestCase):
    def _alarm(self, now):
        return IC.check_admission_wedge_once(_sched(queued=1), now=now)[0]

    def test_unset_is_the_old_20_seconds(self):
        with _env():
            self.assertFalse(self._alarm(19.0))
            self.assertTrue(self._alarm(21.0))

    def test_env_moves_the_alarm_and_the_verdict_text_follows(self):
        with _env(FLLIPER_ADMISSION_WEDGE_SECONDS="45"):
            self.assertFalse(self._alarm(30.0), "20 s must no longer alarm")
            alarm, detail = IC.check_admission_wedge_once(_sched(queued=1), now=50.0)
        self.assertTrue(alarm)
        self.assertIn(">= 45.0s", detail)

    def test_a_shorter_threshold_alarms_earlier(self):
        with _env(FLLIPER_ADMISSION_WEDGE_SECONDS="5"):
            self.assertTrue(self._alarm(6.0))

    def test_non_positive_reads_as_the_default(self):
        for raw in ("0", "-1"):
            with _env(FLLIPER_ADMISSION_WEDGE_SECONDS=raw):
                self.assertFalse(self._alarm(19.0), raw)
                self.assertTrue(self._alarm(21.0), raw)

    def test_the_livelock_env_does_not_move_the_classic_alarm(self):
        with _env(FLLIPER_PREFILL_LIVELOCK_SECONDS="300"):
            self.assertTrue(self._alarm(21.0))


class PrefillLivelockSeconds(unittest.TestCase):
    def _line(self, now):
        # 1 queued + 1 running: the classic verdict is silent (the box is "serving"); decode clock stands at 0.
        alarm, detail = IC.check_admission_wedge_once(_sched(queued=1, running=1, decode=0.0), now=now)
        self.assertFalse(alarm)  # the classic alarm and its recovery driver never follow this line
        return "PREFILL-LIVELOCK" in detail

    def test_unset_is_the_old_20_seconds(self):
        with _env():
            self.assertFalse(self._line(19.0))
            self.assertTrue(self._line(21.0))

    def test_env_moves_the_livelock_line(self):
        with _env(FLLIPER_PREFILL_LIVELOCK_SECONDS="60"):
            self.assertFalse(self._line(30.0))
            self.assertTrue(self._line(61.0))

    def test_non_positive_reads_as_the_default(self):
        with _env(FLLIPER_PREFILL_LIVELOCK_SECONDS="0"):
            self.assertFalse(self._line(19.0))
            self.assertTrue(self._line(21.0))

    def test_the_classic_env_does_not_move_the_livelock_line(self):
        with _env(FLLIPER_ADMISSION_WEDGE_SECONDS="300"):
            self.assertTrue(self._line(21.0))


if __name__ == "__main__":
    unittest.main()
