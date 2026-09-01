# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1073: the abort-word poll stands aside inside a cold-build window.

RED-FIRST against 141705c4de, where `abort_poll_suspended` does not exist and
`_run` polls the device unconditionally.

THE THREE CAN-FAILS, AND THEY ARE NOT EQUALLY IMPORTANT. Showing that the poll
goes quiet inside the window is cheap and would be satisfied by a permanent
disarm, which is why it is the least valuable of the three. The one that
matters is (c): a poll that does not COME BACK is silent, and silence is
indistinguishable from health -- the exact class this whole strand keeps
paying for. So (c) drives the window to end ABNORMALLY (by exception) rather
than on the happy path.

  (a) polls OUTSIDE a window          -- the disarm is conditional, not total
  (b) silent INSIDE a window          -- it actually stands aside
  (c) polls again after an ABNORMAL   -- the dangerous direction
      window end, and after the cap

MEASURED BASIS (boot_855_1072cut, 2026-09-01 14:00Z, py-spy native, all three
ranks in the identical frame): MainThread in `cuModuleLoadData` under
`_init_handles_in_window`, this thread in `cuStreamSynchronize` under
`poll_status_word (barlink_bar1.py:4948)`. 0.0% CPU on every rank, GPU 100%,
PCIe idle, log frozen -- a cold build is CPU-bound, so 0% CPU proves it was
not compiling but blocked in the driver.
"""

from __future__ import annotations

import unittest


class _Recorder:
    """A stand-in watchdog carrying only what the suspension logic reads."""

    def __init__(self):
        self.polls = 0

    def poll_abort_words(self) -> int:
        self.polls += 1
        return 0


def _watchdog():
    """A real PeerWatchdog with its device duty replaced by a counter.

    The suspension decision under test is the REAL method; only the device
    call it guards is stubbed, so a change to that decision fails this test.
    """
    from sglang.srt.distributed.device_communicators import barlink_liveness

    wd = barlink_liveness.PeerWatchdog.__new__(barlink_liveness.PeerWatchdog)
    rec = _Recorder()
    wd.poll_abort_words = rec.poll_abort_words
    return wd, rec


class AbortPollStandsAsideForColdBuilds1073(unittest.TestCase):
    def setUp(self):
        from sglang.srt.utils import jit_cold_build

        # A leaked depth from another test would make (a) pass for the wrong
        # reason, so the precondition is asserted rather than assumed.
        self.assertFalse(
            jit_cold_build.in_cold_build_window(),
            "a cold-build window was already open before this test",
        )

    # -- (a) ---------------------------------------------------------------
    def test_the_poll_runs_outside_a_cold_build_window(self):
        """CAN-FAIL FOR A TOTAL DISARM. A fix that simply stopped polling
        would satisfy (b) and break the watchdog; this is what refuses it."""
        wd, _ = _watchdog()
        self.assertFalse(
            wd.abort_poll_suspended(),
            "the abort poll suspended itself with no cold-build window open -- "
            "that is a permanent blind spot, not the #1073 fix",
        )

    # -- (b) ---------------------------------------------------------------
    def test_the_poll_stands_aside_inside_a_cold_build_window(self):
        from sglang.srt.utils.jit_cold_build import cold_build_window

        wd, _ = _watchdog()
        with cold_build_window("test: pretend nvcc"):
            self.assertTrue(
                wd.abort_poll_suspended(),
                "the device poll kept running inside a cold-build window -- "
                "this is the cuStreamSynchronize that blocked cuModuleLoadData "
                "on all three ranks of boot_855_1072cut",
            )

    # -- (c) THE ONE THAT MATTERS -----------------------------------------
    def test_the_poll_resumes_after_the_window_ends_by_exception(self):
        """THE DANGEROUS DIRECTION. A poll that stays suspended is silent, and
        silence reads as health. The window closes in a `finally`, and the
        suspension is re-decided every round rather than latched, so an
        abnormal end must resume it with no cleanup path involved."""
        from sglang.srt.utils.jit_cold_build import cold_build_window

        wd, _ = _watchdog()

        class _Boom(RuntimeError):
            pass

        with self.assertRaises(_Boom):
            with cold_build_window("test: a load that raises"):
                self.assertTrue(wd.abort_poll_suspended())
                raise _Boom("the module load failed")

        self.assertFalse(
            wd.abort_poll_suspended(),
            "the abort poll stayed suspended after the cold-build window ended "
            "by EXCEPTION -- the watchdog would never look at the device "
            "again, and nothing would say so",
        )

    def test_the_poll_resumes_when_the_window_outlives_the_peer_cap(self):
        """The second bound, and it is the frist the PEERS already honour
        (`SGLANG_BARLINK_BUILD_WINDOW_CAP_S`), not a second timer for the same
        deadline. `in_cold_build_window()` is a bare depth counter with no
        clock, so a leaked depth would otherwise blind this thread for the life
        of the process."""
        import os

        from sglang.srt.distributed.device_communicators import barlink_build_window
        from sglang.srt.utils.jit_cold_build import cold_build_window

        wd, _ = _watchdog()
        prev = os.environ.get(barlink_build_window.ENV_CAP_S)
        os.environ[barlink_build_window.ENV_CAP_S] = "0.001"
        try:
            with cold_build_window("test: a window that overstays"):
                self.assertTrue(wd.abort_poll_suspended(), "first round suspends")
                import time as _t

                _t.sleep(0.05)
                self.assertFalse(
                    wd.abort_poll_suspended(),
                    "the poll stayed blind past the cap the peers honour -- "
                    "past it the window is void for everyone, so blindness "
                    "buys nothing",
                )
        finally:
            if prev is None:
                os.environ.pop(barlink_build_window.ENV_CAP_S, None)
            else:
                os.environ[barlink_build_window.ENV_CAP_S] = prev

    # -- the host half must NOT be disarmed --------------------------------
    def test_the_host_side_peer_probe_is_untouched(self):
        """The blindness is bounded partly because only the DEVICE half stands
        aside. If a future edit routed `probe_once` through the same guard, the
        watchdog would go fully blind inside every window and this says so."""
        import inspect

        from sglang.srt.distributed.device_communicators import barlink_liveness

        run_src = inspect.getsource(barlink_liveness.PeerWatchdog._run)
        probe_line = next(
            l for l in run_src.splitlines() if "self.probe_once()" in l
        )
        self.assertNotIn(
            "abort_poll_suspended",
            probe_line,
            "the host-side peer probe was put behind the cold-build "
            "suspension -- peer death is a /proc fact and must keep being read "
            "while the device is left to the loader",
        )

    # -- the span must be reported ----------------------------------------
    def test_the_suspended_span_is_counted_and_logged(self):
        """A watchdog that goes quiet without saying for how long is the next
        benign zero -- this strand has produced four of those today."""
        import inspect

        from sglang.srt.distributed.device_communicators import barlink_liveness

        src = inspect.getsource(barlink_liveness.PeerWatchdog._run)
        self.assertIn("_abort_poll_skipped", src)
        self.assertIn("RESUMED after", src)


if __name__ == "__main__":
    unittest.main()
