"""23.09. (fnFL2x41): D's first decode round died on its host rank with the
ranks' collective sequences apart right after the extend -- the host's next
eager launch was a 48-byte broadcast, the workers had finished a 72-byte one
the host never issued -- and no instrument named who asked for an EAGER
collective. ``barlink_capture_census.eager_note`` logs them, one line each.

What must hold for that to be usable on a boot and harmless on every other:
off by default, a hard budget, and a callsite that names the caller's frames,
not the transport's own plumbing.
"""
import logging
import unittest

from sglang.srt.distributed.device_communicators import barlink_capture_census as bcc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")


def _seam_that_asked(op, nbytes):
    """Stands in for eagle_worker_v2's draft-token broadcast site."""
    bcc.eager_note(op, nbytes, None, "tp:0")


class EagerTrace(unittest.TestCase):
    def setUp(self):
        self._env = bcc.os.environ.pop(bcc.ENV_EAGER_TRACE, None)
        bcc._EAGER_TRACE.update(left=None, seq=0)

    def tearDown(self):
        bcc.os.environ.pop(bcc.ENV_EAGER_TRACE, None)
        if self._env is not None:
            bcc.os.environ[bcc.ENV_EAGER_TRACE] = self._env
        bcc._EAGER_TRACE.update(left=None, seq=0)

    def test_off_by_default_logs_nothing(self):
        with self.assertNoLogs(bcc.logger, level=logging.INFO):
            for _ in range(5):
                _seam_that_asked("broadcast", 48)

    def test_the_budget_caps_the_lines_and_the_sequence_counts_them(self):
        bcc.os.environ[bcc.ENV_EAGER_TRACE] = "2"
        with self.assertLogs(bcc.logger, level=logging.INFO) as cm:
            for _ in range(4):
                _seam_that_asked("broadcast", 48)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("seq=0 group=tp:0 op=broadcast nbytes=48", cm.output[0])
        self.assertIn("seq=1 ", cm.output[1])

    def test_the_site_names_the_caller_not_the_instrument(self):
        bcc.os.environ[bcc.ENV_EAGER_TRACE] = "1"
        with self.assertLogs(bcc.logger, level=logging.INFO) as cm:
            _seam_that_asked("broadcast", 72)
        site = cm.output[0].split("site=", 1)[1]
        self.assertTrue(site.startswith("test_barlink_eager_trace_x41.py:"), site)
        self.assertNotIn("barlink_capture_census.py", site)


if __name__ == "__main__":
    unittest.main()
