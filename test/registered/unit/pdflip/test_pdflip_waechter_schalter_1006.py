# SPDX-License-Identifier: Apache-2.0
"""WAECHTER-SCHALTER 06.10. (27B-Sitz): switches and thresholds for the guards of the server code that stop the group or
the process hard (plus the ADMISSION-WEDGE / PREFILL-LIVELOCK alarms the user named) -- and unset, NOTHING changes.

User decisions pinned here (06.10. ~12:30Z): (a) switches ONLY for group-killing guards, WAIT_CAP_S is NOT adjustable;
(b) every default = today's behaviour; (c) no 'stop' mode for alarms; (d) the deadman is unchanged.

Per guard: (1) default = the old behaviour, (2) the threshold takes effect, (3) the switch takes effect. The MUTANT
classes are pinned as tests that FAIL if the code is mutated that way: a default that changes behaviour (every
``test_default_*``), a switch that does nothing (every ``test_*_off_*``), WAIT_CAP_S made adjustable
(``TestOutOfScope``), a 'stop' mode on an alarm (``TestOutOfScope``), a typo that silently switches a guard off
(``test_M_*``).

Hermetic: no GPU, no sockets, no processes; the Front is built directly, the scheduler is a namespace.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import signal
import threading
import time
import types
import unittest
from contextlib import contextmanager
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from flliper.srt import guard_switches as GS  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.scheduler_components import invariant_checker as IC  # noqa: E402
from flliper.srt.utils import watchdog as WD  # noqa: E402
from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import front_health as FH  # noqa: E402

_ALL = (
    "FLLIPER_ADMISSION_WEDGE_SECONDS", "FLLIPER_ADMISSION_WEDGE_MODE", "FLLIPER_PREFILL_LIVELOCK_SECONDS",
    "FLLIPER_PREFILL_LIVELOCK_MODE", "FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL", "FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL",
    "FLLIPER_PDFLIP_ENABLE_GROUP_DEAD_STOP", "FLLIPER_PDFLIP_GROUP_DEAD_STREAK", "FLLIPER_PDFLIP_ENABLE_DRAIN_STUCK_STOP",
    "FLLIPER_PDFLIP_DRAIN_STUCK_REFUSALS", "FLLIPER_PDFLIP_ENABLE_CONTROLLER_DEAD_STOP", "FLLIPER_PDFLIP_FLIP_STALL_SLACK",
    "FLLIPER_WEDGE_STATUS_DISABLE", "FLLIPER_PDFLIP_FRONT_HEALTH_FACTS",
    "FLLIPER_PDFLIP_HOST_GUARD_W22", "FLLIPER_PDFLIP_HOST_GUARD_W98", "FLLIPER_ADMISSION_WEDGE_RECOVERY_SECONDS",
)


@contextmanager
def env(**kw):
    """Set env vars for the block (None = unset), restore afterwards."""
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Clean(unittest.TestCase):
    """Every test starts with ALL new switches unset -- the default path."""

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in _ALL}
        os.environ["FLLIPER_WEDGE_STATUS_DISABLE"] = "1"

    def tearDown(self):
        for k in _ALL:
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v


# --------------------------------------------------------------------------------------------------------------
# the readers
# --------------------------------------------------------------------------------------------------------------
class TestReaders(_Clean):
    def test_default_every_threshold_reads_the_old_constant(self):
        self.assertEqual(IC._admission_wedge_threshold(), IC.ADMISSION_WEDGE_SECONDS)
        self.assertEqual(IC._prefill_livelock_threshold(), IC.ADMISSION_WEDGE_SECONDS)
        self.assertEqual(FH.w17_streak(), 2)
        self.assertEqual(IC._admission_wedge_mode(), "act")
        self.assertEqual(IC._prefill_livelock_mode(), "log")

    def test_the_declared_env_defaults_are_the_old_constants(self):
        self.assertEqual(envs.FLLIPER_ADMISSION_WEDGE_SECONDS.default, 20.0)
        self.assertEqual(envs.FLLIPER_PREFILL_LIVELOCK_SECONDS.default, 20.0)
        self.assertEqual(envs.FLLIPER_PDFLIP_FLIP_STALL_SLACK.default, front_mod.FLIP_STALL_SLACK)
        self.assertEqual(envs.FLLIPER_PDFLIP_GROUP_DEAD_STREAK.default, 2)
        self.assertEqual(envs.FLLIPER_PDFLIP_DRAIN_STUCK_REFUSALS.default, 3)
        for e in (
            envs.FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL, envs.FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL,
            envs.FLLIPER_PDFLIP_ENABLE_GROUP_DEAD_STOP, envs.FLLIPER_PDFLIP_ENABLE_DRAIN_STUCK_STOP,
            envs.FLLIPER_PDFLIP_ENABLE_CONTROLLER_DEAD_STOP,
        ):
            self.assertIs(e.default, True, e.name)
        self.assertEqual(
            (envs.FLLIPER_PDFLIP_HOST_GUARD_W22.default, envs.FLLIPER_PDFLIP_HOST_GUARD_W98.default), ("on", "on"))

    def test_nonpositive_and_garbage_thresholds_are_the_default_never_zero(self):
        for bad in ("0", "-3", "nan", "inf", "abc", ""):
            with self.subTest(bad=bad), env(FLLIPER_ADMISSION_WEDGE_SECONDS=bad):
                self.assertEqual(IC._admission_wedge_threshold(), 20.0)
        for bad in ("0", "-1", "x", "1.5"):
            with self.subTest(bad=bad), env(FLLIPER_PDFLIP_GROUP_DEAD_STREAK=bad):
                self.assertEqual(FH.w17_streak(), 2)

    def test_unknown_mode_word_is_the_default_and_warns_once(self):
        with env(FLLIPER_ADMISSION_WEDGE_MODE="banana"):
            GS._warned.clear()
            with self.assertLogs(GS.logger, level="WARNING") as cap:
                self.assertEqual(IC._admission_wedge_mode(), "act")
                self.assertEqual(IC._admission_wedge_mode(), "act")
            self.assertEqual(len(cap.output), 1)

    def test_mode_words_are_case_and_space_insensitive(self):
        with env(FLLIPER_ADMISSION_WEDGE_MODE="  OFF "):
            self.assertEqual(IC._admission_wedge_mode(), "off")


# --------------------------------------------------------------------------------------------------------------
# ADMISSION-WEDGE alarm + PREFILL-LIVELOCK (scheduler thread side)
# --------------------------------------------------------------------------------------------------------------
def _stub(queued, running, first_token_age, decode_age=None, prefill_age=None):
    now = time.perf_counter()
    return types.SimpleNamespace(
        is_initializing=False,
        waiting_queue=[object()] * queued,
        running_batch=types.SimpleNamespace(reqs=[object()] * running),
        last_first_token_progress_time=now - first_token_age,
        last_prefill_progress_time=None if prefill_age is None else now - prefill_age,
        last_decode_progress_time=None if decode_age is None else now - decode_age,
        forward_ct=0,
        _wedge_class_sample=None,
        pdflip_dormant=False,
    )


class TestAdmissionWedgeAlarm(_Clean):
    def test_default_threshold_is_20s(self):
        self.assertFalse(IC.check_admission_wedge_once(_stub(1, 0, 15.0))[0])
        self.assertTrue(IC.check_admission_wedge_once(_stub(1, 0, 25.0))[0])

    def test_threshold_takes_effect_both_ways(self):
        with env(FLLIPER_ADMISSION_WEDGE_SECONDS=5):
            self.assertTrue(IC.check_admission_wedge_once(_stub(1, 0, 8.0))[0])
            self.assertFalse(IC.check_admission_wedge_once(_stub(1, 0, 3.0))[0])
        with env(FLLIPER_ADMISSION_WEDGE_SECONDS=40):
            self.assertFalse(IC.check_admission_wedge_once(_stub(1, 0, 25.0))[0])
            self.assertTrue(IC.check_admission_wedge_once(_stub(1, 0, 45.0))[0])

    def test_the_detail_names_the_configured_threshold(self):
        with env(FLLIPER_ADMISSION_WEDGE_SECONDS=40):
            _, detail = IC.check_admission_wedge_once(_stub(1, 0, 45.0))
        self.assertIn(">= 40.0s", detail)


class _FakeDriver:
    instances = []

    def __init__(self, scheduler):
        self.steps = []
        _FakeDriver.instances.append(self)

    def step(self, alarm):
        self.steps.append(alarm)

    def recovery_status(self):
        return None


class TestAdmissionWedgeMode(_Clean):
    def _poll(self, **envkw):
        _FakeDriver.instances.clear()
        with env(**envkw), mock.patch.object(IC, "AdmissionWedgeRecovery", _FakeDriver):
            poll = IC.make_admission_wedge_poller(_stub(1, 0, 25.0))
            self.assertTrue(poll())
        return _FakeDriver.instances[-1]

    def test_default_act_drives_the_recovery(self):
        self.assertEqual(self._poll().steps, [True])

    def test_mode_act_explicit_is_the_default(self):
        self.assertEqual(self._poll(FLLIPER_ADMISSION_WEDGE_MODE="act").steps, [True])

    def test_mode_log_reports_but_never_attempts_recovery(self):
        d = self._poll(FLLIPER_ADMISSION_WEDGE_MODE="log")
        self.assertEqual(d.steps, [])

    def test_mode_off_starts_no_thread(self):
        with env(FLLIPER_ADMISSION_WEDGE_MODE="off"), mock.patch.object(IC.threading, "Thread") as T:
            with self.assertLogs(IC.logger, level="WARNING") as cap:
                self.assertIsNone(IC.create_admission_wedge_watchdog(object()))
        T.assert_not_called()
        self.assertTrue(any("NOT started" in m for m in cap.output))

    def _interval_of(self, **envkw):
        seen = []
        stop = threading.Event()
        stop.set()
        with env(**envkw), mock.patch.object(IC, "make_admission_wedge_poller", lambda s: (lambda: None)), \
                mock.patch.object(IC.time, "sleep", lambda s: seen.append(s)):
            t = IC.create_admission_wedge_watchdog(object(), stop=stop)
            t.join(2.0)
        return seen

    def test_default_poll_is_10s(self):
        self.assertEqual(self._interval_of(), [10.0])

    def test_configured_threshold_moves_the_poll_to_half(self):
        self.assertEqual(self._interval_of(FLLIPER_ADMISSION_WEDGE_SECONDS=8), [4.0])

    def test_default_logs_no_config_line(self):
        stop = threading.Event()
        stop.set()
        with mock.patch.object(IC, "make_admission_wedge_poller", lambda s: (lambda: None)), \
                mock.patch.object(IC.time, "sleep", lambda s: None), \
                self.assertLogs(IC.logger, level="INFO") as cap:
            IC.create_admission_wedge_watchdog(object(), stop=stop).join(2.0)
            IC.logger.info("sentinel")
        self.assertFalse(any("watchdog configured" in m for m in cap.output))


class TestPrefillLivelock(_Clean):
    def test_default_threshold_is_20s(self):
        self.assertIn("PREFILL-LIVELOCK", IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=25.0))[1])
        self.assertNotIn("PREFILL-LIVELOCK", IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=15.0))[1])

    def test_threshold_takes_effect_both_ways(self):
        with env(FLLIPER_PREFILL_LIVELOCK_SECONDS=5):
            self.assertIn("PREFILL-LIVELOCK", IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=8.0))[1])
        with env(FLLIPER_PREFILL_LIVELOCK_SECONDS=40):
            self.assertNotIn("PREFILL-LIVELOCK", IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=25.0))[1])

    def test_the_wedge_threshold_does_not_move_the_livelock_one(self):
        with env(FLLIPER_ADMISSION_WEDGE_SECONDS=100):
            self.assertIn("PREFILL-LIVELOCK", IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=25.0))[1])

    def test_mode_off_gives_no_livelock_verdict_and_no_error_line(self):
        with env(FLLIPER_PREFILL_LIVELOCK_MODE="off"):
            _, detail = IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=651.0), log_on_alarm=True)
        self.assertNotIn("PREFILL-LIVELOCK", detail)

    def test_mode_log_is_the_default_and_still_never_arms_recovery(self):
        alarm, detail = IC.check_admission_wedge_once(_stub(2, 3, 0.5, decode_age=651.0))
        self.assertFalse(alarm)
        self.assertIn("PREFILL-LIVELOCK", detail)


# --------------------------------------------------------------------------------------------------------------
# scheduler hard watchdog
# --------------------------------------------------------------------------------------------------------------
_REAL_SLEEP = time.sleep


def _hard_watchdog(timeout=0.2, soft=False):
    wd = WD.WatchdogRaw.__new__(WD.WatchdogRaw)  # no thread: the check is driven by hand
    wd.debug_name, wd.watchdog_timeout, wd.soft = "Scheduler", timeout, soft
    wd.get_counter, wd.is_active = (lambda: 7), (lambda: True)  # active, counter frozen
    wd.dump_info = wd.describe_arm = None
    wd.stall_age = None
    wd.parent_process = mock.Mock()
    return wd


def _trip(wd):
    """Run the check to its verdict; the 5 s pre-SIGQUIT sleep is shortened."""
    with mock.patch.object(WD, "pyspy_dump_schedulers", lambda *a, **k: None), mock.patch.object(
        WD.time, "sleep", side_effect=lambda s: _REAL_SLEEP(min(s, 0.02))
    ), mock.patch.object(WD, "logger") as log:
        wd._watchdog_once()
    return log


class TestSchedulerWatchdogKill(_Clean):
    def test_default_still_sends_sigquit_to_the_parent(self):
        wd = _hard_watchdog()
        _trip(wd)
        wd.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_switch_on_explicitly_is_the_default(self):
        wd = _hard_watchdog()
        with env(FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL="1"):
            _trip(wd)
        wd.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_switch_off_keeps_the_dump_and_the_line_but_sends_nothing(self):
        wd = _hard_watchdog()
        with env(FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL="0"):
            log = _trip(wd)
        wd.parent_process.send_signal.assert_not_called()
        lines = [str(c.args[0]) for c in log.error.call_args_list]
        self.assertTrue(any("watchdog timeout" in m for m in lines), lines)
        self.assertTrue(any("FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL=0" in m for m in lines), lines)

    def test_the_soft_watchdog_never_signals_either_way(self):
        for v in ("0", "1"):
            wd = _hard_watchdog(soft=True)
            with env(FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL=v):
                _trip(wd)
            wd.parent_process.send_signal.assert_not_called()

    def test_M_a_typo_never_switches_the_hard_watchdog_off(self):
        for word in ("flase", "of", "none", "disable"):
            with self.subTest(word=word), env(FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL=word):
                wd = _hard_watchdog()
                with self.assertWarns(UserWarning):
                    _trip(wd)
                wd.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_the_creator_is_untouched_it_always_returns_a_watchdog(self):
        with mock.patch.object(IC, "WatchdogRaw") as W, env(FLLIPER_ENABLE_SCHEDULER_WATCHDOG_KILL="0"):
            r = IC.create_scheduler_watchdog(types.SimpleNamespace(), 300.0)
        W.assert_called_once()
        self.assertIs(W.call_args.kwargs["soft"], False)
        self.assertIsNotNone(r)


# --------------------------------------------------------------------------------------------------------------
# SubprocessWatchdog
# --------------------------------------------------------------------------------------------------------------
class _Dead:
    def __init__(self, code=1, pid=4242):
        self.exitcode = code
        self.pid = pid

    def is_alive(self):
        return False


class TestSubprocessWatchdogKill(_Clean):
    def _check(self, **envkw):
        with env(**envkw), mock.patch.object(WD.os, "kill") as k:
            w = WD.SubprocessWatchdog([_Dead()], ["scheduler_0"])
            res = w._check_processes()
        return res, k

    def test_default_sends_sigquit_to_itself(self):
        res, k = self._check()
        self.assertTrue(res)
        k.assert_called_once_with(os.getpid(), signal.SIGQUIT)

    def test_switch_off_reports_but_does_not_signal(self):
        with self.assertLogs(WD.logger, level="ERROR") as cap:
            res, k = self._check(FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL="0")
        self.assertFalse(res, "the loop goes on watching")
        k.assert_not_called()
        self.assertTrue(any("is gone" in m for m in cap.output))
        self.assertTrue(any("FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL=0" in m for m in cap.output))

    def test_the_suppression_is_named_once_not_every_poll(self):
        with env(FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL="0"), mock.patch.object(WD.os, "kill"), \
                mock.patch.object(WD, "logger") as log:
            w = WD.SubprocessWatchdog([_Dead()], ["s"])
            w._check_processes()
            w._check_processes()
        n = sum("suppressed" in str(c.args[0]) for c in log.error.call_args_list)
        self.assertEqual(n, 1)

    def test_M_a_typo_stays_on(self):
        for word in ("flase", "off ", "disable"):
            with self.subTest(word=word):
                res, k = self._check(FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL=word)
                self.assertTrue(res)
                k.assert_called_once()

    def test_clean_exit_never_signals_in_any_setting(self):
        for v in ("1", "0"):
            with env(FLLIPER_ENABLE_SUBPROCESS_WATCHDOG_KILL=v), mock.patch.object(WD.os, "kill") as k:
                w = WD.SubprocessWatchdog([_Dead(code=0)], ["s"])
                self.assertFalse(w._check_processes())
                k.assert_not_called()

    def test_the_poll_interval_stays_a_constant(self):
        self.assertEqual(WD.SubprocessWatchdog([], [])._interval, 1.0)


# --------------------------------------------------------------------------------------------------------------
# front: W17, health poller, W2, flip stall, beacon
# --------------------------------------------------------------------------------------------------------------
def _bare_front():
    f = object.__new__(front_mod.Front)
    f.counters = __import__("collections").Counter()
    return f


def _front():
    return front_mod.Front("http://127.0.0.1:31000", "http://127.0.0.1:31001", "D", "t1006", "", 0, 0, {}, 45.0)


class TestW17(_Clean):
    def test_default_streak_is_2(self):
        f = _bare_front()
        self.assertFalse(f.group_dead_should_stop(state="serving", ok=False, alive=True, streak=1))
        self.assertTrue(f.group_dead_should_stop(state="serving", ok=False, alive=True, streak=2))

    def test_streak_takes_effect_both_ways(self):
        f = _bare_front()
        with env(FLLIPER_PDFLIP_GROUP_DEAD_STREAK=4):
            self.assertFalse(f.group_dead_should_stop(state="serving", ok=False, alive=True, streak=3))
            self.assertTrue(f.group_dead_should_stop(state="serving", ok=False, alive=True, streak=4))
            self.assertTrue(f.group_dead_should_stop(state="flipping", ok=False, alive=False, streak=4))
            self.assertFalse(f.group_dead_should_stop(state="flipping", ok=False, alive=False, streak=3))
        with env(FLLIPER_PDFLIP_GROUP_DEAD_STREAK=1):
            self.assertTrue(f.group_dead_should_stop(state="serving", ok=False, alive=True, streak=1))

    def test_a_held_rank_stops_at_once_whatever_the_streak(self):
        f = _bare_front()
        with env(FLLIPER_PDFLIP_GROUP_DEAD_STREAK=9):
            self.assertTrue(f.group_dead_should_stop(state="serving", ok=True, alive=True, streak=0, hold=True))

    def test_health_503_follows_the_same_streak(self):
        facts = lambda n: FH.GroupFacts(False, True, n, None, 0.0)
        self.assertIsNotNone(FH.unhealthy_reason(facts(2), "serving"))
        with env(FLLIPER_PDFLIP_GROUP_DEAD_STREAK=5):
            self.assertIsNone(FH.unhealthy_reason(facts(4), "serving"))
            self.assertIsNotNone(FH.unhealthy_reason(facts(5), "serving"))

    def test_switch_off_names_the_verdict_and_never_says_stop(self):
        f = _bare_front()
        with env(FLLIPER_PDFLIP_ENABLE_GROUP_DEAD_STOP="0"), self.assertLogs(front_mod.logger, level="ERROR") as cap:
            self.assertFalse(f.group_dead_should_stop(state="serving", ok=False, alive=False, streak=2))
            self.assertFalse(f.group_dead_should_stop(state="serving", ok=True, alive=True, streak=0, hold=True))
        self.assertTrue(any("suppressed" in m and "no STOP" in m for m in cap.output))

    def test_switch_off_does_not_invent_a_stop_where_none_was_due(self):
        f = _bare_front()
        with env(FLLIPER_PDFLIP_ENABLE_GROUP_DEAD_STOP="0"):
            self.assertFalse(f.group_dead_should_stop(state="serving", ok=False, alive=True, streak=1))
            self.assertFalse(f.group_dead_should_stop(state="serving", ok=True, alive=True, streak=5))

    def test_M_a_typo_stays_on(self):
        f = _bare_front()
        for word in ("flase", "off", "disable"):
            with self.subTest(word=word), env(FLLIPER_PDFLIP_ENABLE_GROUP_DEAD_STOP=word):
                with self.assertWarns(UserWarning):
                    self.assertTrue(f.group_dead_should_stop(state="serving", ok=False, alive=False, streak=2))

    def test_the_stop_sites_ask_the_gate_not_the_literal(self):
        src = inspect.getsource(front_mod.Front)
        self.assertEqual(len(re.findall(r'self\.do_stop\(\s*"W17 PdFlipGroupDead"', src)), 3)
        self.assertIn("streak < _front_health_mod().w17_streak()", src)
        self.assertNotIn("if streak < 2:", src)


class TestW2(_Clean):
    def _run(self, n_in_row, **envkw):
        f = _bare_front()
        f.drain_refusals_in_a_row = n_in_row
        with env(**envkw):
            return f.drain_stuck_should_stop()

    def test_default_three_in_a_row_stops(self):
        self.assertFalse(self._run(2))
        self.assertTrue(self._run(3))

    def test_count_takes_effect_both_ways(self):
        self.assertFalse(self._run(3, FLLIPER_PDFLIP_DRAIN_STUCK_REFUSALS=5))
        self.assertTrue(self._run(5, FLLIPER_PDFLIP_DRAIN_STUCK_REFUSALS=5))
        self.assertTrue(self._run(1, FLLIPER_PDFLIP_DRAIN_STUCK_REFUSALS=1))
        self.assertFalse(self._run(2, FLLIPER_PDFLIP_DRAIN_STUCK_REFUSALS=0), "0 reads as the default 3, never 'always'")

    def test_switch_off_names_it_and_does_not_stop(self):
        f = _bare_front()
        f.drain_refusals_in_a_row = 3
        with env(FLLIPER_PDFLIP_ENABLE_DRAIN_STUCK_STOP="0"), self.assertLogs(front_mod.logger, level="ERROR") as cap:
            self.assertFalse(f.drain_stuck_should_stop())
        self.assertTrue(any("W2 PdFlipDrainStuck" in m and "no STOP" in m for m in cap.output))

    def test_M_a_typo_stays_on(self):
        for word in ("flase", "off", "disable"):
            with self.subTest(word=word), self.assertWarns(UserWarning):
                self.assertTrue(self._run(3, FLLIPER_PDFLIP_ENABLE_DRAIN_STUCK_STOP=word))

    def test_the_flip_asks_the_gate_after_counting_w1_and_keeps_the_old_wording(self):
        src = inspect.getsource(front_mod.Front.flip)
        self.assertLess(src.index('self.counters["W1_PdFlipDrainRefused"] += 1'), src.index("self.drain_stuck_should_stop()"))
        self.assertNotIn(">= 3", src[src.index("self.drain_stuck_should_stop()") - 200:src.index("self.drain_stuck_should_stop()")])
        self.assertIn("'three' if self.drain_refusals_in_a_row == 3", src)


def _open_flip(front, t0=1000.0):
    front.state = "flipping"
    front._flip_t0 = t0
    front._flip_stage = "drain"
    front.epoch = 0
    return front


class TestFlipStall(_Clean):
    def test_default_bound_is_4x_the_last_flip(self):
        f = _front()
        f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 3200})
        bound, why = f._flip_stall_bound_s()
        self.assertAlmostEqual(bound, 12.8)
        self.assertIn("4x this boot's last measured flip", why)

    def test_slack_takes_effect(self):
        f = _front()
        f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 3200})
        with env(FLLIPER_PDFLIP_FLIP_STALL_SLACK=10):
            bound, why = f._flip_stall_bound_s()
        self.assertAlmostEqual(bound, 32.0)
        self.assertIn("10x this boot's last", why)

    def test_slack_does_not_touch_the_pre_first_flip_bound(self):
        f = _front()
        with env(FLLIPER_PDFLIP_FLIP_STALL_SLACK=10):
            self.assertEqual(f._flip_stall_bound_s()[0], front_mod.DRAIN_DEADLINE_DEFAULT_S)

    def test_default_fires_once(self):
        f = _open_flip(_front())
        line = f.flip_stall_check(now=1000.0 + front_mod.DRAIN_DEADLINE_DEFAULT_S)
        self.assertIsNotNone(line)
        self.assertEqual(f.counters["flip_stall"], 1)

    def test_nonpositive_slack_is_the_default(self):
        f = _front()
        f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 3200})
        for bad in ("0", "-2"):
            with env(FLLIPER_PDFLIP_FLIP_STALL_SLACK=bad):
                self.assertAlmostEqual(f._flip_stall_bound_s()[0], 12.8)

    def test_the_stall_check_never_stops_the_front(self):
        """Report-only detector: no do_stop in its source."""
        self.assertNotIn("do_stop", inspect.getsource(front_mod.Front.flip_stall_check))


class TestControllerDead(_Clean):
    def test_default_stops(self):
        self.assertTrue(_bare_front().controller_dead_should_stop())

    def test_switch_off_keeps_the_line_and_does_not_stop(self):
        with env(FLLIPER_PDFLIP_ENABLE_CONTROLLER_DEAD_STOP="0"), self.assertLogs(front_mod.logger, level="ERROR") as cap:
            self.assertFalse(_bare_front().controller_dead_should_stop())
        self.assertTrue(any("CONTROLLER-DEAD" in m and "suppressed" in m for m in cap.output))

    def test_M_a_typo_stays_on(self):
        for word in ("flase", "off"):
            with self.subTest(word=word), env(FLLIPER_PDFLIP_ENABLE_CONTROLLER_DEAD_STOP=word), self.assertWarns(UserWarning):
                self.assertTrue(_bare_front().controller_dead_should_stop())

    def test_the_escape_path_goes_through_the_gate(self):
        src = inspect.getsource(front_mod.Front.controller)
        i = src.index("flip_escape_verdict(")
        self.assertLess(src.index("self.controller_dead_should_stop()", i), src.index("self.do_stop(*verdict)", i))
        self.assertIn('self.counters["controller_dead"] += 1', src, "the counter and the traceback are not gated")


# --------------------------------------------------------------------------------------------------------------
# what must NOT be switchable / must stay as it was (user decisions 06.10. ~12:30Z)
# --------------------------------------------------------------------------------------------------------------
class TestOutOfScope(_Clean):
    def _all_env_names(self):
        return [n for n in dir(envs) if n.startswith("FLLIPER_")]

    def test_M_wait_cap_s_is_not_adjustable(self):
        """(a) WAIT_CAP_S (store read > 60 s) stays a constant: no env field, no reader, no change to the module."""
        from flliper.srt.managers import pdflip_store_told as ST

        self.assertEqual(ST.WAIT_CAP_S, 60.0)
        self.assertFalse([n for n in self._all_env_names() if "WAIT_CAP" in n or "STORE_TOLD_WAIT" in n])
        src = inspect.getsource(ST)
        self.assertNotIn("guard_switches", src)
        self.assertIsNotNone(re.search(r"^WAIT_CAP_S = 60\.0$", src, re.M), "the cap is a literal constant")
        self.assertEqual(len(re.findall(r"^\s*WAIT_CAP_S\s*=", src, re.M)), 1, "assigned once, nowhere else")
        for line in (l for l in src.splitlines() if "WAIT_CAP_S" in l):
            for word in ("environ", "getenv", "envs.", "guard_switches"):
                self.assertNotIn(word, line, "no reader on the cap: " + line)

    def test_M_no_stop_mode_for_alarms(self):
        """(c) the alarm modes know act/log/off (wedge) and log/off (livelock) -- 'stop' reads as the default."""
        for word in ("stop", "kill", "STOP"):
            with self.subTest(word=word):
                with env(FLLIPER_ADMISSION_WEDGE_MODE=word, FLLIPER_PREFILL_LIVELOCK_MODE=word):
                    self.assertEqual(IC._admission_wedge_mode(), "act")
                    self.assertEqual(IC._prefill_livelock_mode(), "log")
        self.assertFalse(hasattr(GS, "MODE_STOP"))
        self.assertFalse([n for n in self._all_env_names() if "FLIP_STALL_MODE" in n or "INTAKE_STALL" in n])

    def test_M_the_deadman_is_unchanged(self):
        """(d) the launcher keeps its fixed GRACE_S=600 and has no deadman switch; arm_deadman has no new parameter."""
        from flliper.srt.pdflip import launcher

        self.assertNotIn("deadman-grace", inspect.getsource(launcher.build_parser))
        self.assertNotIn('"--deadman"', inspect.getsource(launcher.build_parser))
        self.assertEqual(
            list(inspect.signature(launcher.arm_deadman).parameters),
            ["log", "boot_log", "port", "pattern", "probe_s", "tag", "name", "dry"],
        )
        self.assertIn("GRACE_S=600", inspect.getsource(launcher.arm_deadman))

    def test_the_request_abort_guards_and_poll_constants_have_no_new_env(self):
        names = self._all_env_names()
        for frag in ("HEALTH_POLL", "HEALTH_PROBE", "BEACON_BUSY", "SUBPROCESS_WATCHDOG_INTERVAL", "FLLIPER_WATCHDOG_ACTION"):
            self.assertFalse([n for n in names if frag in n], frag)
        self.assertEqual((FH.POLL_S, FH.PROBE_TIMEOUT_S), (5.0, 8.0))
        self.assertEqual(FH.poll_interval_s(), 5.0)
        self.assertEqual(FH.probe_timeout_s(), 8.0)


# --------------------------------------------------------------------------------------------------------------
# host guards W22 / W98 (user decision 06.10. ~11:00Z): AN/AUS each, separately
# --------------------------------------------------------------------------------------------------------------
class _StopSampling(BaseException):
    pass


class TestHostGuards(_Clean):
    def _run(self, **envkw):
        """Drive ``host_watermark_sampler`` for 3 ticks with both guards TRIPPED; returns (do_stop calls, log)."""
        f = _bare_front()
        f.host_watermark_period_s = 1.0
        f._observed_anon_drift_mib_per_min = 0.0
        f._anon_preboot_bytes = 0
        f.tag = "t1006"
        f.state = "serving"
        stops = []
        ticks = {"n": 0}

        class Latch:
            latched = True

            def __init__(self, **kw):
                pass

            def observe(self, *a, **k):
                return "W98 RATE-LATCH tripped (test)"

        async def fake_sleep(_s):
            ticks["n"] += 1
            if ticks["n"] >= 3:
                raise _StopSampling

        hl = front_mod.host_ledger
        patches = [
            mock.patch.object(hl, "resolve_margin", lambda **k: 1),
            mock.patch.object(hl, "watermark_provenance", lambda m: "prov"),
            mock.patch.object(hl, "RateLatch", Latch),
            mock.patch.object(hl, "read_cgroup_pressure",
                              lambda: {"nonreclaim_gib": 50.0, "file_gib": 1.0, "shmem_gib": 0.5}),
            mock.patch.object(hl, "latch_free_pool", lambda pr: (1.0, "t", 2.0)),
            mock.patch.object(hl, "arena_fill_gib", lambda tag: 0.0),
            mock.patch.object(hl, "read_cgroup", lambda: {"current": 100}),
            mock.patch.object(hl, "reap_mark_gib", lambda x: 90.0),
            mock.patch.object(hl, "read_cgroup_anon_bytes", lambda: 0),
            mock.patch.object(hl, "watermark_breach_verdict", lambda *a, **k: "W22 LEVEL breach (test)"),
            mock.patch.object(front_mod.Front, "_flip_cushion_note", lambda self, pr: None, create=True),
            mock.patch.object(front_mod.Front, "do_stop", lambda self, name, detail: stops.append(name)),
            mock.patch.object(front_mod.asyncio, "sleep", fake_sleep),
        ]
        with env(**envkw):
            for p in patches:
                p.start()
            try:
                with self.assertLogs(front_mod.logger, level="INFO") as cap:
                    try:
                        asyncio.new_event_loop().run_until_complete(f.host_watermark_sampler())
                    except _StopSampling:
                        pass
            finally:
                for p in reversed(patches):
                    p.stop()
        return stops, cap.output

    def test_default_both_on_stops_on_the_rate_latch_first_exactly_as_before(self):
        stops, log = self._run()
        self.assertEqual(stops, ["W98 PdFlipHostRateLatched"])
        self.assertFalse(any("HOST GUARD" in m for m in log), "default boot log carries no new line")

    def test_w98_off_leaves_w22_armed(self):
        stops, log = self._run(FLLIPER_PDFLIP_HOST_GUARD_W98="off")
        self.assertEqual(stops, ["W22 PdFlipHostWatermarkBreached"])
        self.assertTrue(any("HOST GUARD W98 AUS: Host-RAM ist nicht mehr geschuetzt" in m for m in log))
        self.assertFalse(any("HOST GUARD W22 AUS" in m for m in log))

    def test_w22_off_leaves_w98_armed(self):
        stops, log = self._run(FLLIPER_PDFLIP_HOST_GUARD_W22="0")
        self.assertEqual(stops, ["W98 PdFlipHostRateLatched"])
        self.assertTrue(any("HOST GUARD W22 AUS: Host-RAM ist nicht mehr geschuetzt" in m for m in log))

    def test_both_off_never_stops_and_warns_for_both(self):
        stops, log = self._run(FLLIPER_PDFLIP_HOST_GUARD_W22="off", FLLIPER_PDFLIP_HOST_GUARD_W98="false")
        self.assertEqual(stops, [])
        self.assertTrue(any("HOST GUARD W22 AUS" in m and "nicht mehr geschuetzt" in m for m in log))
        self.assertTrue(any("HOST GUARD W98 AUS" in m and "nicht mehr geschuetzt" in m for m in log))
        self.assertTrue(any("NOT stopping" in m for m in log))

    def test_M_a_typo_never_switches_a_host_guard_off(self):
        for word in ("of", "aus", "disable", "2"):
            with self.subTest(word=word):
                stops, log = self._run(FLLIPER_PDFLIP_HOST_GUARD_W22=word, FLLIPER_PDFLIP_HOST_GUARD_W98=word)
                self.assertEqual(stops, ["W98 PdFlipHostRateLatched"])
                self.assertFalse(any("HOST GUARD W22 AUS" in m for m in log))

    def test_flag_on_words(self):
        for w, want in (("on", True), ("1", True), ("TRUE", True), ("off", False), ("0", False), ("No", False), ("", True)):
            with self.subTest(w=w), env(FLLIPER_PDFLIP_HOST_GUARD_W22=w):
                self.assertIs(GS.flag_on(envs.FLLIPER_PDFLIP_HOST_GUARD_W22), want)


class TestWedgeRecoveryOrderWarning(_Clean):
    def test_all_defaults_no_warning(self):
        self.assertIsNone(IC._wedge_threshold_order_warning())

    def test_a_form_style_recovery_below_the_alarm_no_warning(self):
        with env(FLLIPER_ADMISSION_WEDGE_RECOVERY_SECONDS=2.0):
            self.assertIsNone(IC._wedge_threshold_order_warning())

    def test_recovery_larger_than_alarm_warns(self):
        with env(FLLIPER_ADMISSION_WEDGE_SECONDS=5):
            w = IC._wedge_threshold_order_warning()  # default recovery 60 > alarm 5
        self.assertIn("LARGER", w)
        self.assertIn("60.0s", w)
        self.assertIn("5.0s", w)
        with env(FLLIPER_ADMISSION_WEDGE_RECOVERY_SECONDS=90):
            self.assertIn("90.0s", IC._wedge_threshold_order_warning())

    def test_the_recovery_stays_its_own_value(self):
        with env(FLLIPER_ADMISSION_WEDGE_SECONDS=5):
            self.assertEqual(IC._admission_wedge_recovery_threshold(), IC.ADMISSION_WEDGE_RECOVERY_SECONDS)

    def test_the_warning_is_written_at_watchdog_start_and_not_by_default(self):
        stop = threading.Event()
        stop.set()

        def start(**envkw):
            with env(**envkw), mock.patch.object(IC, "make_admission_wedge_poller", lambda s: (lambda: None)), \
                    mock.patch.object(IC.time, "sleep", lambda s: None), \
                    self.assertLogs(IC.logger, level="INFO") as cap:
                IC.create_admission_wedge_watchdog(object(), stop=stop).join(2.0)
                IC.logger.info("sentinel")
            return cap.output

        self.assertFalse(any("LARGER" in m for m in start()))
        self.assertTrue(any("LARGER" in m for m in start(FLLIPER_ADMISSION_WEDGE_SECONDS=5)))


if __name__ == "__main__":
    unittest.main()
