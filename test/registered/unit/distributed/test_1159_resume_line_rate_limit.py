"""#1159 -- the #1073 RESUMING line must not be 41 percent of a boot log.

THE MEASUREMENT, boot weg1b3
(/spinning/evidence-665-f1/boot_855_weg1b3_6980c75eac_0902_234752.log,
200,766 lines): 82,350 of those lines are

    #1073 abort-word poll RESUMING under a still-open cold-build window: ...

41.0 % of the log, emitted at ~97 lines/s/rank from 23:59:54 onward, because
``abort_poll_suspended`` is called on the watchdog's own tick and the cap-
exceeded branch logged on EVERY tick once the 23:59:54 cutover-warmup window
stopped closing. The line is correct and the condition is real; what is wrong
is that a persistent condition was reported at tick rate, which buries the
event that opened the window.

THE PROPERTY PINNED HERE:
  * the FIRST line is emitted verbatim, unchanged from before this fix;
  * inside the window no second line is emitted;
  * the next line past the window CARRIES THE COUNT of what it suppressed, so
    the rate limit can never be mistaken for the condition having ended.

RED-FIRST: before the fix the emitter logged on every call, so
``test_it_is_silent_inside_the_window`` saw 3 lines instead of 1.
"""

import unittest

from sglang.srt.distributed.device_communicators import barlink_liveness as bl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, fmt, *args, **kw):
        self.warnings.append(fmt % args if args else fmt)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _Window:
    """Stand-in for ``barlink_build_window``: a fixed cap, a named env key."""

    ENV_CAP_S = "SGLANG_BARLINK_BUILD_WINDOW_CAP_S"

    @staticmethod
    def build_cap_s():
        return 60.0


class _JitColdBuild:
    @staticmethod
    def in_cold_build_window():
        return True


class _Watchdog:
    """The two attributes the emitter reads, and nothing else."""

    _abort_poll_suspended_since = None

    abort_poll_suspended = bl.PeerWatchdog.abort_poll_suspended


def _drive(monkeypatch, clock, log, wd, at):
    clock.t = at
    return wd.abort_poll_suspended()


class TheResumeLineIsRateLimited(unittest.TestCase):
    def setUp(self):
        self.log = _Log()
        self.clock = _Clock()
        self.wd = _Watchdog()

    def _patch(self, monkeypatch=None):
        import sys
        import types

        mod = types.ModuleType("sglang.srt.utils.jit_cold_build")
        mod.in_cold_build_window = _JitColdBuild.in_cold_build_window
        pkg = sys.modules["sglang.srt.utils"]
        self._saved_jit = getattr(pkg, "jit_cold_build", None)
        pkg.jit_cold_build = mod
        self._saved_win = bl_window_get()
        bl_window_set(_Window)
        self._saved_time = bl.time.monotonic
        bl.time.monotonic = self.clock
        self._saved_logger = bl.logger
        bl.logger = self.log

    def tearDown(self):
        import sys

        sys.modules["sglang.srt.utils"].jit_cold_build = self._saved_jit
        bl_window_set(self._saved_win)
        bl.time.monotonic = self._saved_time
        bl.logger = self._saved_logger

    def test_the_first_line_is_verbatim(self):
        self._patch()
        self.clock.t = 1000.0
        self.wd.abort_poll_suspended()  # opens the window
        self.clock.t = 1000.0 + 61.0  # past the 60 s cap
        self.assertFalse(self.wd.abort_poll_suspended())
        self.assertEqual(len(self.log.warnings), 1)
        line = self.log.warnings[0]
        self.assertIn("#1073 abort-word poll RESUMING under a still-open", line)
        self.assertIn("past the 60.0s cap the peers honour", line)
        self.assertNotIn("suppressed", line)

    def test_it_is_silent_inside_the_window(self):
        self._patch()
        self.clock.t = 1000.0
        self.wd.abort_poll_suspended()
        for t in (1061.0, 1063.0, 1069.9):
            self.clock.t = t
            self.wd.abort_poll_suspended()
        self.assertEqual(
            len(self.log.warnings),
            1,
            f"the persistent condition was reported at tick rate: "
            f"{len(self.log.warnings)} lines",
        )

    def test_the_next_line_carries_the_count(self):
        self._patch()
        self.clock.t = 1000.0
        self.wd.abort_poll_suspended()
        for t in (1061.0, 1063.0, 1069.9):
            self.clock.t = t
            self.wd.abort_poll_suspended()
        self.clock.t = 1061.0 + bl.RESUME_LINE_MIN_INTERVAL_S + 0.1
        self.wd.abort_poll_suspended()
        self.assertEqual(len(self.log.warnings), 2)
        self.assertIn("suppressed", self.log.warnings[1])
        # 2 ticks were swallowed between the two emitted lines.
        self.assertIn("2 further", self.log.warnings[1])

    def test_a_closed_window_forgets_the_rate_limit(self):
        """A new episode starts with its own verbatim first line."""
        self._patch()
        self.clock.t = 1000.0
        self.wd.abort_poll_suspended()
        self.clock.t = 1061.0
        self.wd.abort_poll_suspended()
        import sys

        sys.modules["sglang.srt.utils"].jit_cold_build.in_cold_build_window = lambda: (
            False
        )
        self.clock.t = 1062.0
        self.wd.abort_poll_suspended()  # window closed -> state reset
        sys.modules["sglang.srt.utils"].jit_cold_build.in_cold_build_window = lambda: (
            True
        )
        self.clock.t = 1063.0
        self.wd.abort_poll_suspended()  # re-opens
        self.clock.t = 1063.0 + 61.0
        self.wd.abort_poll_suspended()
        self.assertEqual(len(self.log.warnings), 2)
        self.assertNotIn("suppressed", self.log.warnings[1])


_WIN_HOLDER = {}


def bl_window_get():
    import sglang.srt.distributed.device_communicators as pkg

    return getattr(pkg, "barlink_build_window", None)


def bl_window_set(mod):
    import sglang.srt.distributed.device_communicators as pkg

    if mod is None:
        return
    pkg.barlink_build_window = mod
    import sys

    sys.modules["sglang.srt.distributed.device_communicators.barlink_build_window"] = (
        mod
    )


if __name__ == "__main__":
    unittest.main()
