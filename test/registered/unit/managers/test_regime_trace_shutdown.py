"""#363 -- the verdict trace's summary line lands on every way a server stops.

The gates window found the gap: nothing called ``close_trace()`` at shutdown,
so a run that ended cleanly produced a trace the reader refuses -- "zero
desyncs" and "zero desyncs SO FAR" are different claims, and the refusal is
only useful if a clean run always produces the line.

DRIVEN, NOT SUPPLIED. The rule the same window taught: a test that hands the
code the value under test cannot falsify the code that computes it. So these
do not call ``close_trace()``; they stop a real child process the three ways a
scheduler process actually stops, and read the file afterwards.

  sigterm            a real SIGTERM to a real process. Neither ``finally`` nor
                     ``atexit`` runs on Python's default disposition, so only
                     the installed handler can produce the line -- and the
                     child sleeps after signalling itself, so a handler that
                     swallows the signal instead of chaining shows up as a
                     timeout rather than as a pass.
  exit               normal interpreter exit -> the atexit registration.
  finally            the shape of ``run_scheduler_process``: an exception,
                     then a ``finally`` calling the real helper.
  keyboardinterrupt  the same, for the BaseException that ``except Exception``
                     does not catch (this is what Ctrl-C and the window's own
                     SIGINT shutdown actually deliver).

The scheduler's own ``finally`` block is additionally checked structurally --
running it needs a GPU -- and that is stated rather than implied.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHILD = os.path.join(_HERE, "_regime_shutdown_child.py")
_PY_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "python"))


def _run_child(how, path, timeout=60):
    env = dict(os.environ)
    env["REGIME_PY"] = _PY_ROOT
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, _CHILD, path, how],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


class TestSummaryLandsOnEveryStopPath(CustomTestCase):
    def _drive(self, how):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            proc = _run_child(how, path)
            self.assertTrue(os.path.exists(path), proc.stderr[-500:])
            rows = _lines(path)
            return rows, proc

    def test_sigterm_writes_the_summary(self):
        """The path nothing else covers: no unwinding, no atexit."""
        rows, _proc = self._drive("sigterm")
        self.assertEqual(rows[-1]["kind"], "summary", rows[-1])
        self.assertIn("desyncs", rows[-1])

    def test_sigterm_still_kills_the_process(self):
        """The hook may change what the server WRITES on the way out, never
        how it dies. A handler that returned instead of chaining would let the
        child reach its sleep and time out here."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            proc = _run_child("sigterm", path, timeout=30)
        self.assertEqual(proc.returncode, -15, f"rc={proc.returncode}")

    def test_normal_exit_writes_the_summary(self):
        rows, _proc = self._drive("exit")
        self.assertEqual(rows[-1]["kind"], "summary")

    def test_the_finally_shape_writes_the_summary(self):
        """``run_scheduler_process``'s exception path, through the real
        helper the scheduler's finally calls."""
        rows, proc = self._drive("finally")
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
        self.assertEqual(rows[-1]["kind"], "summary")

    def test_keyboardinterrupt_writes_the_summary(self):
        """`except Exception` does not catch it; `finally` does. This is what
        Ctrl-C -- and the gates window's own SIGINT shutdown -- delivers."""
        rows, _proc = self._drive("keyboardinterrupt")
        self.assertEqual(rows[-1]["kind"], "summary")

    def test_the_verdicts_survive_alongside_the_summary(self):
        """A summary that arrived by truncating the run would pass every
        check above."""
        rows, _proc = self._drive("sigterm")
        verdicts = [r for r in rows if r["kind"] == "verdict"]
        self.assertEqual(len(verdicts), 3)
        self.assertEqual(rows[-1]["verdicts"], 3)


class TestHelperContract(CustomTestCase):
    def test_it_never_raises_on_a_scheduler_without_an_observer(self):
        from sglang.srt.managers.regime_runtime import close_regime_trace

        class _Bare:
            pass

        close_regime_trace(_Bare())  # must not raise

    def test_it_never_raises_when_the_observer_throws(self):
        """A teardown helper that can throw turns a clean shutdown into a
        confusing one, and an already-failing one into a lost traceback."""
        from sglang.srt.managers.regime_runtime import close_regime_trace

        class _Boom:
            def close_trace(self):
                raise RuntimeError("disk full")

        class _Sched:
            regime_observer = _Boom()

        close_regime_trace(_Sched())

    def test_no_trace_path_installs_no_signal_handler(self):
        """Off by default must stay free: an observer with no trace has
        nothing to flush and must not touch the process's signal disposition."""
        import signal

        from sglang.srt.managers.regime_runtime import (
            MODE_OBSERVE,
            RegimeObserver,
            install_trace_shutdown_hook,
        )

        before = signal.getsignal(signal.SIGTERM)
        obs = RegimeObserver(consensus_interval=2, tp_size=1, mode=MODE_OBSERVE)
        install_trace_shutdown_hook(obs)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)


class TestSchedulerWiring(CustomTestCase):
    """Structural: running the real block needs a GPU, so the check is that
    the call is IN the finally, on the scheduler's own source."""

    @staticmethod
    def _src():
        import pathlib

        import sglang.srt.managers.scheduler as mod

        return pathlib.Path(mod.__file__).read_text()

    def test_the_finally_block_closes_the_trace(self):
        src = self._src()
        fin = src.rindex("    finally:")
        block = src[fin : fin + 1400]
        self.assertIn("close_regime_trace(scheduler)", block)

    def test_it_is_closed_before_the_other_teardown(self):
        """Ordering matters only one way: the trace must not be lost if a
        later teardown step raises."""
        src = self._src()
        fin = src.rindex("    finally:")
        block = src[fin : fin + 1400]
        self.assertLess(
            block.index("close_regime_trace(scheduler)"),
            block.index("_shutdown_fpm()"),
        )

    def test_the_builder_installs_the_hook(self):
        import ast
        import pathlib

        import sglang.srt.managers.regime_runtime as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_regime_observer"
        )
        called = {
            c.func.id
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        self.assertIn("install_trace_shutdown_hook", called)


if __name__ == "__main__":
    unittest.main()
