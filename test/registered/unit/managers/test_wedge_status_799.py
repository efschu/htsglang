# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#799: the admission-wedge verdict leaves the scheduler process.

Hermetic: no GPU, no server, no thread except the one test that is ABOUT the
thread. Every path writes into a temporary directory.

WHAT THIS PINS. The #699/#739 detector was correct and unheard: it computed a
true verdict 146 times during boot 0822_0829 and handed it to ``logger.error``,
which has no consumer that can act. The export is the edge that makes the
verdict a fact a supervisor can read.

THE FALSIFIER is ``test_the_production_poller_publishes``. It drives
``make_admission_wedge_poller`` -- the callable the live watchdog thread runs
-- and not ``publish_verdict`` directly. A test that calls the publisher
proves the publisher works and says nothing about whether the detector ever
calls it, which is the exact shape of the defect being fixed. Delete the
``_publish`` call from the poller and this test goes red while every existing
#699 test stays green.

THE SECOND FALSIFIER is ``test_a_stale_verdict_is_not_a_verdict``. A file left
behind by a dead publisher must read as "no measurement", never as "fine": a
supervisor that believes an old healthy verdict is blinder than one with no
signal at all, because it does not know it is blind.
"""

import json
import os
import tempfile
import time
import unittest

from sglang.srt.managers import wedge_status as WS
from sglang.srt.managers.scheduler_components.invariant_checker import (
    make_admission_wedge_poller,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

#: A module-wide sink, and the env var is pointed at it for the WHOLE module
#: rather than set and unset per test. Reason, measured rather than
#: hypothetical: this module starts the real detector thread, and a thread
#: that fails to stop (which is exactly what the M12 mutant makes it do) keeps
#: publishing for the rest of the process. If the env override has been popped
#: by then, ``status_dir`` falls back to the REAL ``/run`` default and the
#: suite writes a stale verdict into the production path -- observed twice
#: during this ticket. Never restoring to "unset" removes the fallback.
_SINK = tempfile.TemporaryDirectory()


def setUpModule():
    os.environ[WS.ENV_DIR] = _SINK.name


def tearDownModule():
    os.environ[WS.ENV_DIR] = _SINK.name
    _SINK.cleanup()


class _use_dir:
    """Point the export at ``d``, then back to the module sink -- never to
    'unset', which is what re-arms the production fallback."""

    def __init__(self, d):
        self.d = d

    def __enter__(self):
        os.environ[WS.ENV_DIR] = self.d
        return self.d

    def __exit__(self, *exc):
        os.environ[WS.ENV_DIR] = _SINK.name
        return False


class _FakeRunningBatch:
    def __init__(self, reqs):
        self.reqs = reqs


class _FakeScheduler:
    """Only the attributes the wedge poller reads. Never a real boot."""

    def __init__(self, queued=0, running=0, age=0.0, pp=0, tp=1):
        now = time.perf_counter()
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = _FakeRunningBatch([object()] * running)
        self.last_first_token_progress_time = now - age
        self.last_prefill_progress_time = now - age
        self.ps = type("PS", (), {"pp_rank": pp, "tp_rank": tp})()


class TestPublishAndRead(CustomTestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            WS.publish_verdict("pp0-tp0", True, "wedged hard", directory=d)
            sig = WS.read_wedge_signal(d)
            self.assertIs(sig.verdict, True)
            self.assertIn("pp0-tp0", sig.detail)
            self.assertEqual(sig.ranks_seen, 1)

    def test_a_clean_verdict_reads_as_false_not_none(self):
        """False and None are different answers and must stay different."""
        with tempfile.TemporaryDirectory() as d:
            WS.publish_verdict("pp0-tp0", False, "all good", directory=d)
            self.assertIs(WS.read_wedge_signal(d).verdict, False)

    def test_no_files_is_no_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            sig = WS.read_wedge_signal(d)
            self.assertIsNone(sig.verdict)

    def test_a_missing_directory_is_no_measurement_not_a_crash(self):
        sig = WS.read_wedge_signal("/nonexistent/wedge/dir/799")
        self.assertIsNone(sig.verdict)

    def test_a_stale_verdict_is_not_a_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            WS.publish_verdict("pp0-tp0", False, "was fine long ago",
                               directory=d,
                               now=lambda: time.time() - 10_000.0)
            sig = WS.read_wedge_signal(d)
            self.assertIsNone(sig.verdict)
            self.assertTrue(sig.stale)

    def test_a_stale_WEDGE_also_stops_counting(self):
        """Staleness cuts both ways, deliberately. A dead publisher's last
        word must not restart a lane forever."""
        with tempfile.TemporaryDirectory() as d:
            WS.publish_verdict("pp0-tp0", True, "wedged, then died",
                               directory=d,
                               now=lambda: time.time() - 10_000.0)
            self.assertIsNone(WS.read_wedge_signal(d).verdict)

    def test_one_wedged_rank_wedges_the_lane(self):
        """OR, not majority: under pure TP the ranks are one collective, and
        a single rank that cannot admit work stalls the group."""
        with tempfile.TemporaryDirectory() as d:
            WS.publish_verdict("pp0-tp0", False, "fine", directory=d)
            WS.publish_verdict("pp1-tp0", False, "fine", directory=d)
            WS.publish_verdict("pp2-tp0", True, "WEDGE", directory=d)
            sig = WS.read_wedge_signal(d)
            self.assertIs(sig.verdict, True)
            self.assertEqual(sig.ranks_seen, 3)
            self.assertIn("pp2-tp0", sig.detail)

    def test_ranks_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                WS.publish_verdict(f"pp{i}-tp0", False, "fine", directory=d)
            names = [n for n in os.listdir(d) if n.endswith(".json")]
            self.assertEqual(len(names), 3, names)

    def test_a_corrupt_file_does_not_poison_the_read(self):
        with tempfile.TemporaryDirectory() as d:
            WS.publish_verdict("pp0-tp0", True, "WEDGE", directory=d)
            with open(os.path.join(d, "wedge.broken.json"), "w") as fh:
                fh.write("{not json")
            self.assertIs(WS.read_wedge_signal(d).verdict, True)

    def test_publish_never_raises_on_an_unwritable_target(self):
        """Telemetry must not be able to kill the scheduler it observes."""
        with tempfile.TemporaryDirectory() as d:
            blocked = os.path.join(d, "file-not-a-dir")
            open(blocked, "w").close()
            self.assertIsNone(
                WS.publish_verdict("pp0-tp0", True, "x", directory=blocked))

    def test_the_payload_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as d:
            path = WS.publish_verdict("pp0-tp0", True, "WEDGE", directory=d)
            with open(path) as fh:
                rec = json.load(fh)
            self.assertIs(rec["wedged"], True)
            self.assertEqual(rec["rank"], "pp0-tp0")
            self.assertIsInstance(rec["wall"], float)

    def test_the_kill_switch_disables_the_export(self):
        self.assertIsNone(WS.status_dir({WS.ENV_DISABLE: "1"}))
        self.assertEqual(WS.status_dir({}), WS.DEFAULT_STATUS_DIR)
        self.assertEqual(WS.status_dir({WS.ENV_DIR: "/tmp/x"}), "/tmp/x")


class TestTheDetectorPublishes(CustomTestCase):
    def test_the_production_poller_publishes_a_wedge(self):
        """THE falsifier. Drives the callable the live thread runs."""
        with tempfile.TemporaryDirectory() as d:
            with _use_dir(d):
                sched = _FakeScheduler(queued=9, running=0, age=64.7)
                poll = make_admission_wedge_poller(sched)
                self.assertIs(poll(), True)
                sig = WS.read_wedge_signal(d)
        self.assertIs(sig.verdict, True)
        self.assertIn("ADMISSION-WEDGE", sig.detail)

    def test_the_production_poller_also_publishes_the_healthy_verdict(self):
        """A file that appears only when things break cannot be told apart
        from a file that never appears; the healthy verdict is what gives the
        reader's staleness check something to measure against."""
        with tempfile.TemporaryDirectory() as d:
            with _use_dir(d):
                poll = make_admission_wedge_poller(_FakeScheduler(queued=0))
                self.assertIs(poll(), False)
                sig = WS.read_wedge_signal(d)
        self.assertIs(sig.verdict, False)

    def test_the_inherited_536_blind_spot_is_real_and_pinned(self):
        """Named honestly rather than papered over.

        ``admission_wedge_verdict`` returns "not wedged" whenever
        ``running > 0``, so a request starved behind a co-tenant that IS
        running produces no alarm -- and therefore no published wedge and no
        watchdog action. Transporting a verdict does not widen it. This test
        exists so the gap is a documented property with a test naming it,
        not a surprise found during the next outage.
        """
        with tempfile.TemporaryDirectory() as d:
            with _use_dir(d):
                sched = _FakeScheduler(queued=40, running=1, age=600.0)
                poll = make_admission_wedge_poller(sched)
                self.assertIs(poll(), False)
                sig = WS.read_wedge_signal(d)
        self.assertIs(sig.verdict, False, "#536 class is still invisible")

    def test_a_broken_check_publishes_nothing_rather_than_a_false_clean(self):
        class _Exploding:
            is_initializing = False

            def __getattr__(self, name):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as d:
            with _use_dir(d):
                poll = make_admission_wedge_poller(_Exploding())
                self.assertIsNone(poll())
                sig = WS.read_wedge_signal(d)
        self.assertIsNone(sig.verdict)

    def test_rank_labels_come_from_the_parallel_state(self):
        with tempfile.TemporaryDirectory() as d:
            with _use_dir(d):
                poll = make_admission_wedge_poller(
                    _FakeScheduler(queued=1, pp=2, tp=1))
                poll()
                names = os.listdir(d)
        self.assertIn("wedge.pp2-tp1.json", names)


class TestTheThreadRunsThePoll(CustomTestCase):
    def test_the_watchdog_thread_actually_calls_the_poller(self):
        """The last edge: factory -> thread -> poll -> file.

        Bounded by a deadline, and the failure condition is explicit: if no
        file appears within the budget the loop is not running the poll.
        """
        import threading

        from sglang.srt.managers.scheduler_components import invariant_checker

        stop = threading.Event()
        with tempfile.TemporaryDirectory() as d:
            with _use_dir(d):
                try:
                    t = invariant_checker.create_admission_wedge_watchdog(
                        _FakeScheduler(queued=9, running=0, age=64.7),
                        poll_interval=0.05, stop=stop)
                    deadline = time.time() + 10.0
                    verdict = None
                    while time.time() < deadline:
                        verdict = WS.read_wedge_signal(d).verdict
                        if verdict is not None:
                            break
                        time.sleep(0.05)
                finally:
                    # Stop and JOIN before leaving the guarded directory. A
                    # thread that refuses to stop must still never reach the
                    # production default -- ``_use_dir`` restores to the module
                    # sink, not to "unset", so even this failure mode writes
                    # into a temp path.
                    stop.set()
                    t.join(timeout=5.0)
        self.assertIs(verdict, True,
                      "the watchdog thread published nothing within 10s")
        self.assertFalse(t.is_alive(), "the watchdog thread would not stop")


if __name__ == "__main__":
    unittest.main()
