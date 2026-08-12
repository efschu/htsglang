"""#604: /health fast path must check scheduler subprocess liveness.

Before this fix, GET /health with
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION disabled returned 200 unconditionally
without checking whether the scheduler/detokenizer subprocesses were alive.
This meant a server with dead schedulers could still appear healthy to external
monitors.

The fix adds a subprocess liveness check in the fast path that returns 503
(naming the dead component) when any tracked subprocess is not alive.
"""

from __future__ import annotations

import json
import logging
import types
import unittest
from typing import List

from sglang.srt.entrypoints.http_server import (
    _GlobalState,
    _health_fast_path,
    _make_health_error_json,
    set_global_state,
)
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Duck-types multiprocessing.Process with controllable liveness."""

    def __init__(self, pid: int, alive: bool = True, exitcode=None):
        self.pid = pid
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


class _FakeWatchdog(SubprocessWatchdog):
    """The REAL SubprocessWatchdog with injectable fake processes, so these
    tests exercise the actual ``processes_with_names`` accessor the /health
    fast path consumes."""

    def __init__(self, processes: List[_FakeProcess], names: List[str]):
        super().__init__(processes=processes, process_names=names)


class _FakeTokenizerManager:
    """Minimal tokenizer_manager with an optional watchdog."""

    def __init__(self, watchdog=None):
        self._subprocess_watchdog = watchdog


class _FakeServerStatus:
    Starting = "Starting"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_global_state(watchdog=None):
    """Build a minimal _GlobalState for a test."""
    return set_global_state(
        _GlobalState(
            tokenizer_manager=_FakeTokenizerManager(watchdog),
            template_manager=types.SimpleNamespace(),
            scheduler_info={},
        )
    )


# ---------------------------------------------------------------------------
# Tests: _health_fast_path
# ---------------------------------------------------------------------------


class TestHealthFastPathAllAlive(unittest.TestCase):
    """All subprocesses alive -> 200."""

    def setUp(self):
        procs = [
            _FakeProcess(pid=1001, alive=True),
            _FakeProcess(pid=1002, alive=True),
        ]
        names = ["scheduler_0", "detokenizer_0"]
        _make_global_state(watchdog=_FakeWatchdog(processes=procs, names=names))

    def test_all_alive_returns_200(self):
        """When every subprocess is alive, /health fast path returns 200."""
        resp = _health_fast_path()
        self.assertEqual(resp.status_code, 200)

    def test_no_body_on_200(self):
        """A 200 response must not carry an error body."""
        resp = _health_fast_path()
        self.assertEqual(resp.body, b"")


class TestHealthFastPathOneDead(unittest.TestCase):
    """One dead subprocess -> 503 with component name in body."""

    def setUp(self):
        procs = [
            _FakeProcess(pid=1001, alive=True),
            _FakeProcess(pid=1002, alive=False, exitcode=139),  # segfault
        ]
        names = ["scheduler_0", "detokenizer_0"]
        _make_global_state(watchdog=_FakeWatchdog(processes=procs, names=names))

    def test_dead_subprocess_returns_503(self):
        """Can-fail proof: if someone restores the unconditional 200 this test
        goes red because status_code will no longer be 503."""
        resp = _health_fast_path()
        self.assertEqual(resp.status_code, 503)

    def test_response_names_the_component(self):
        """The 503 body must name the dead component."""
        resp = _health_fast_path()
        body = json.loads(resp.body)
        self.assertEqual(body["component"], "detokenizer_0")
        self.assertEqual(body["pid"], 1002)
        self.assertEqual(body["exit_code"], 139)

    def test_error_key_present(self):
        """The body must contain an 'error' key for machine parsing."""
        resp = _health_fast_path()
        body = json.loads(resp.body)
        self.assertIn("error", body)
        self.assertEqual(body["error"], "subprocess not alive")

    def test_first_dead_reported(self):
        """When multiple processes are dead, the first one is reported."""
        procs = [
            _FakeProcess(pid=2001, alive=False, exitcode=1),
            _FakeProcess(pid=2002, alive=False, exitcode=2),
        ]
        names = ["sched_A", "sched_B"]
        _make_global_state(watchdog=_FakeWatchdog(processes=procs, names=names))
        resp = _health_fast_path()
        body = json.loads(resp.body)
        self.assertEqual(body["component"], "sched_A")


class TestHealthFastPathNoWatchdog(unittest.TestCase):
    """No watchdog attached -> 200 (startup or engine-only mode)."""

    def setUp(self):
        _make_global_state(watchdog=None)

    def test_no_watchdog_returns_200(self):
        """When there is no watchdog (e.g., during early startup), the fast
        path still returns 200 rather than crashing."""
        resp = _health_fast_path()
        self.assertEqual(resp.status_code, 200)


class TestHealthFastPathEmptyWatchdog(unittest.TestCase):
    """Watchdog with zero processes -> 200."""

    def setUp(self):
        _make_global_state(watchdog=_FakeWatchdog(processes=[], names=[]))

    def test_empty_watchdog_returns_200(self):
        """An empty process list is trivially healthy."""
        resp = _health_fast_path()
        self.assertEqual(resp.status_code, 200)


class TestHealthFastPathZombieReaped(unittest.TestCase):
    """Process that is not alive but had no prior exitcode (zombie before
    reap) -> 503. is_alive() reaps the zombie, setting exitcode."""

    def setUp(self):
        # Simulate: is_alive() returns False and exitcode gets set after reap
        proc = _FakeProcess(pid=3001, alive=False, exitcode=11)  # SIGSEGV
        _make_global_state(watchdog=_FakeWatchdog(processes=[proc], names=["scheduler_0"]))

    def test_zombie_subprocess_returns_503(self):
        """After reap, the exitcode is set; the check still returns 503."""
        resp = _health_fast_path()
        self.assertEqual(resp.status_code, 503)
        body = json.loads(resp.body)
        self.assertEqual(body["pid"], 3001)
        self.assertEqual(body["exit_code"], 11)


class TestHealthFastPathExitCodeZero(unittest.TestCase):
    """Process that exited with code 0 (unexpected clean exit) -> 503.

    For a long-running scheduler, even a clean exit means it has died."""

    def setUp(self):
        proc = _FakeProcess(pid=4001, alive=False, exitcode=0)
        _make_global_state(watchdog=_FakeWatchdog(processes=[proc], names=["scheduler_0"]))

    def test_clean_exit_still_503(self):
        """A scheduler that exited cleanly is still dead."""
        resp = _health_fast_path()
        self.assertEqual(resp.status_code, 503)


# ---------------------------------------------------------------------------
# Tests: _make_health_error_json
# ---------------------------------------------------------------------------


class TestMakeHealthErrorJson(unittest.TestCase):
    """Unit tests for the JSON helper."""

    def test_basic_structure(self):
        body = _make_health_error_json("scheduler_0", 12345, 139)
        obj = json.loads(body)
        self.assertEqual(obj["component"], "scheduler_0")
        self.assertEqual(obj["pid"], 12345)
        self.assertEqual(obj["exit_code"], 139)
        self.assertIn("error", obj)

    def test_none_exit_code(self):
        body = _make_health_error_json("detokenizer_1", 99, None)
        obj = json.loads(body)
        self.assertIsNone(obj["exit_code"])


# ---------------------------------------------------------------------------
# Logging: verify error log is emitted
# ---------------------------------------------------------------------------


class _ErrorCapture(logging.Handler):
    """Collect error-level log records."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class TestHealthFastPathLogging(unittest.TestCase):
    """Verify the error log line fires for a dead subprocess."""

    def setUp(self):
        self.handler = _ErrorCapture()
        self.handler.setLevel(logging.ERROR)
        logger = logging.getLogger("sglang.srt.entrypoints.http_server")
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)

        procs = [_FakeProcess(pid=5001, alive=False, exitcode=139)]
        names = ["scheduler_0"]
        _make_global_state(watchdog=_FakeWatchdog(processes=procs, names=names))

    def tearDown(self):
        logger = logging.getLogger("sglang.srt.entrypoints.http_server")
        logger.removeHandler(self.handler)

    def test_error_log_fires(self):
        _health_fast_path()
        self.assertEqual(len(self.handler.records), 1)
        msg = self.handler.records[0].getMessage()
        self.assertIn("scheduler_0", msg)
        self.assertIn("5001", msg)

    def test_log_names_exit_code(self):
        _health_fast_path()
        msg = self.handler.records[0].getMessage()
        self.assertIn("139", msg)


# ---------------------------------------------------------------------------
# #485/C40: the liveness check must not depend on which health MODE is on
# ---------------------------------------------------------------------------


class TestADeadRankIsUnhealthyOnTheDefaultPath(unittest.TestCase):
    """#604's liveness check was real but only reachable with
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0, which is NOT the default.

    On a default boot /health goes to the generation path instead, and that
    path returns 200 as soon as ``last_receive_tstamp`` moves -- which any
    still-draining scheduler output does. So an instance could print healthy
    while a rank was gone, which is the #622 wedge signature. Liveness is a
    precondition of health in every mode, so it is checked before the mode is
    consulted at all.
    """

    def setUp(self):
        procs = [
            _FakeProcess(pid=7001, alive=False, exitcode=-9),
            _FakeProcess(pid=7002, alive=True),
        ]
        _make_global_state(
            watchdog=_FakeWatchdog(
                processes=procs, names=["scheduler_0", "detokenizer_0"]
            )
        )
        self.tm = _global_state_tokenizer_manager()
        self.tm.gracefully_exit = False
        self.tm.server_status = "Running"

    def _call(self, path: str):
        import asyncio

        from sglang.srt.entrypoints.http_server import health_generate

        request = types.SimpleNamespace(url=types.SimpleNamespace(path=path))
        return asyncio.run(health_generate(request))

    def test_health_is_503_when_a_rank_is_gone(self):
        # The fake tokenizer manager has no generate machinery at all, so if
        # the handler reaches the generation path this raises instead of
        # returning -- which is the second half of the assertion.
        resp = self._call("/health")
        self.assertEqual(resp.status_code, 503)

    def test_health_generate_is_503_when_a_rank_is_gone(self):
        resp = self._call("/health_generate")
        self.assertEqual(resp.status_code, 503)

    def test_the_body_still_names_the_dead_component(self):
        resp = self._call("/health")
        body = json.loads(resp.body)
        self.assertEqual(body["component"], "scheduler_0")
        self.assertEqual(body["pid"], 7001)
        self.assertEqual(body["exit_code"], -9)


def _global_state_tokenizer_manager():
    from sglang.srt.entrypoints.http_server import _global_state

    return _global_state.tokenizer_manager


if __name__ == "__main__":
    unittest.main()
