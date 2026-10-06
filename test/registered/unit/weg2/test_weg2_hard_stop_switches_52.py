"""deskq 52 (user decision 06.10.): the front's group-stopping watchdogs get a switch and, where there was none, a
threshold. Defaults are today's behaviour; an unset env changes nothing.

  W17 Weg2GroupDead      SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP (1)  SGLANG_WEG2_GROUP_DEAD_STREAK (2, minimum 1)
  W2  Weg2DrainStuck     SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP (1) SGLANG_WEG2_DRAIN_STUCK_REFUSALS (3, minimum 1)
  CONTROLLER-DEAD        SGLANG_WEG2_ENABLE_CONTROLLER_DEAD_STOP (1)
  FLIP STALL (feeds the deadman)   SGLANG_WEG2_FLIP_STALL_SLACK (4.0; non-positive = 4.0)

Red on the base (d10a4c3d60): the helpers/envs do not exist there, so every "switched/moved" case fails and the
default cases for the new helpers fail too (AttributeError); W17 and the flip stall bound fail on behaviour. The env is
set via ``mock.patch.dict(os.environ)`` (not ``envs.X.override``) so the file stays runnable against the base.
"""

import inspect
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front import FLIP_STALL_SLACK, Front  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

NAMES = (
    "SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP",
    "SGLANG_WEG2_GROUP_DEAD_STREAK",
    "SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP",
    "SGLANG_WEG2_DRAIN_STUCK_REFUSALS",
    "SGLANG_WEG2_ENABLE_CONTROLLER_DEAD_STOP",
    "SGLANG_WEG2_FLIP_STALL_SLACK",
)


def _env(**kv):
    """All six knobs cleared, then ``kv`` applied (an unset env = the default)."""
    clean = {k: v for k, v in os.environ.items() if k not in NAMES}
    clean.update(kv)
    return mock.patch.dict(os.environ, clean, clear=True)


def _bare():
    return object.__new__(Front)


def _w17(f, streak, **kw):
    kw.setdefault("state", "serving")
    kw.setdefault("ok", False)
    kw.setdefault("alive", True)
    return f.group_dead_should_stop(streak=streak, **kw)


class DefaultsPinned(unittest.TestCase):
    """Bounding-default pins (docs/dev/CONVENTION_bounding_defaults.md): the shipped defaults are today's values."""

    def test_unset_env_reads_todays_values(self):
        from sglang.srt.environ import envs

        with _env():
            self.assertIs(envs.SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP.get(), True)
            self.assertEqual(envs.SGLANG_WEG2_GROUP_DEAD_STREAK.get(), 2)
            self.assertIs(envs.SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP.get(), True)
            self.assertEqual(envs.SGLANG_WEG2_DRAIN_STUCK_REFUSALS.get(), 3)
            self.assertIs(envs.SGLANG_WEG2_ENABLE_CONTROLLER_DEAD_STOP.get(), True)
            self.assertEqual(envs.SGLANG_WEG2_FLIP_STALL_SLACK.get(), 4.0)
            self.assertEqual(FLIP_STALL_SLACK, 4.0)


class W17GroupDead(unittest.TestCase):
    def test_default_is_today_streak_2_stops(self):
        f = _bare()
        with _env():
            self.assertFalse(_w17(f, 1))
            self.assertTrue(_w17(f, 2))
            self.assertTrue(_w17(f, 5, alive=False, state="flipping"))  # #1411: a dead session stops in a flip
            self.assertTrue(_w17(f, 1, hold=True))  # a held rank stops at the first sighting
            self.assertFalse(_w17(f, 9, state="flipping"))  # #1378: silent /health behind a live leg

    def test_switch_off_removes_every_stop_path_of_w17(self):
        f = _bare()
        with _env(SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP="0"), mock.patch.object(front_mod, "logger") as log:
            self.assertFalse(_w17(f, 2))
            self.assertFalse(_w17(f, 5, alive=False, state="flipping"))
            self.assertFalse(_w17(f, 1, hold=True))
        self.assertTrue(
            any("SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP=0" in str(c.args[0]) for c in log.error.call_args_list)
        )

    def test_switch_off_does_not_log_when_nothing_would_have_stopped(self):
        f = _bare()
        with _env(SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP="0"), mock.patch.object(front_mod, "logger") as log:
            self.assertFalse(_w17(f, 1))
            self.assertFalse(_w17(f, 3, ok=True))
        log.error.assert_not_called()

    def test_streak_threshold_moves_the_stop(self):
        f = _bare()
        with _env(SGLANG_WEG2_GROUP_DEAD_STREAK="4"):
            self.assertFalse(_w17(f, 2))
            self.assertFalse(_w17(f, 3))
            self.assertTrue(_w17(f, 4))
            self.assertTrue(_w17(f, 1, hold=True), "a held rank is not a streak question")

    def test_streak_minimum_is_one_not_zero_or_negative(self):
        f = _bare()
        for raw in ("0", "-3"):
            with _env(SGLANG_WEG2_GROUP_DEAD_STREAK=raw):
                self.assertFalse(_w17(f, 0), raw)  # the poller never calls with 0, but 0 must not stop
                self.assertTrue(_w17(f, 1), raw)


class W2DrainStuck(unittest.TestCase):
    def test_default_three_in_a_row_stop(self):
        f = _bare()
        with _env():
            f.drain_refusals_in_a_row = 2
            self.assertFalse(f.drain_stuck_should_stop())
            f.drain_refusals_in_a_row = 3
            self.assertTrue(f.drain_stuck_should_stop())

    def test_switch_off_never_stops_but_says_so(self):
        f = _bare()
        f.drain_refusals_in_a_row = 7
        with _env(SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP="0"), mock.patch.object(front_mod, "logger") as log:
            self.assertFalse(f.drain_stuck_should_stop())
        self.assertTrue(
            any("SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP=0" in str(c.args[0]) for c in log.error.call_args_list)
        )

    def test_refusal_threshold_moves_the_stop(self):
        f = _bare()
        with _env(SGLANG_WEG2_DRAIN_STUCK_REFUSALS="5"):
            f.drain_refusals_in_a_row = 4
            self.assertFalse(f.drain_stuck_should_stop())
            f.drain_refusals_in_a_row = 5
            self.assertTrue(f.drain_stuck_should_stop())

    def test_the_flip_asks_the_helper_and_not_a_literal_3(self):
        """Wiring, structural (driving a real flip needs a loop, a session and two groups): the W2 stop in the
        flip path must go through the helper, and the literal ``>= 3`` must be gone."""
        src = inspect.getsource(Front)
        self.assertIn("if self.drain_stuck_should_stop():", src)
        self.assertNotIn("self.drain_refusals_in_a_row >= 3", src)


class ControllerDead(unittest.TestCase):
    def test_default_stops(self):
        with _env():
            self.assertTrue(_bare().controller_dead_should_stop())

    def test_switch_off_does_not_stop_but_says_so(self):
        with _env(SGLANG_WEG2_ENABLE_CONTROLLER_DEAD_STOP="0"), mock.patch.object(front_mod, "logger") as log:
            self.assertFalse(_bare().controller_dead_should_stop())
        self.assertTrue(
            any("SGLANG_WEG2_ENABLE_CONTROLLER_DEAD_STOP=0" in str(c.args[0]) for c in log.error.call_args_list)
        )

    def test_the_controller_gates_its_stop_on_the_helper(self):
        """Wiring, structural: the CONTROLLER-DEAD line and counter come first, the stop only behind the helper."""
        src = inspect.getsource(Front.controller)
        i_line, i_gate, i_stop = (
            src.index("controller_dead_line("),
            src.index("if self.controller_dead_should_stop():"),
            src.index("self.do_stop(*verdict)"),
        )
        self.assertLess(i_line, i_gate)
        self.assertLess(i_gate, i_stop)


class FlipStallSlack(unittest.TestCase):
    def _front_with_flip(self, ms):
        f = _bare()
        f.flip_log = [{"flip_ms": ms, "sleep": "P", "wake": "D"}]
        f.drain_deadline_s = 120.0
        return f

    def test_default_is_four_times_the_last_flip(self):
        with _env():
            bound, prov = self._front_with_flip(3200.0)._flip_stall_bound_s()
        self.assertAlmostEqual(bound, FLIP_STALL_SLACK * 3.2)
        self.assertEqual(bound, 4.0 * 3.2)
        self.assertTrue(prov.startswith("4x this boot's last measured flip"), prov)

    def test_env_moves_the_factor_and_the_provenance_names_it(self):
        with _env(SGLANG_WEG2_FLIP_STALL_SLACK="10"):
            bound, prov = self._front_with_flip(3200.0)._flip_stall_bound_s()
        self.assertAlmostEqual(bound, 32.0)
        self.assertTrue(prov.startswith("10x this boot's last measured flip"), prov)

    def test_non_positive_reads_as_the_default(self):
        for raw in ("0", "-2"):
            with _env(SGLANG_WEG2_FLIP_STALL_SLACK=raw):
                bound, _ = self._front_with_flip(3200.0)._flip_stall_bound_s()
            self.assertAlmostEqual(bound, 4.0 * 3.2, msg=raw)

    def test_before_the_first_flip_the_drain_deadline_stands_in_whatever_the_factor(self):
        f = _bare()
        f.flip_log = []
        f.drain_deadline_s = 120.0
        with _env(SGLANG_WEG2_FLIP_STALL_SLACK="10"):
            self.assertEqual(f._flip_stall_bound_s()[0], 120.0)


if __name__ == "__main__":
    unittest.main()
