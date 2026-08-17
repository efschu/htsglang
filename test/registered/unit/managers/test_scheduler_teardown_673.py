"""#673: the collectives must be destroyed before the interpreter tears down.

THE BUG THESE TESTS EXIST FOR. "terminate called without an active exception"
fires after a CLEAN drain -- 0 remaining requests, graceful exit logged, then a
C++ abort with no Python frame. The cause is not the work; it is the exit.
``ProcessGroupNCCL`` runs a watchdog and a ``HeartbeatMonitor`` as C++ threads
that are joined by the process group's DESTRUCTOR. If the group is never
destroyed, those ``std::thread`` objects are still joinable when the process
tears down -- and destroying a joinable ``std::thread`` calls
``std::terminate``, whose message is exactly that one, with no active exception
because there never was one.

The omission is provable by inspection and torch reports it in every boot log:
``destroy_distributed_environment`` / ``cleanup_dist_env_and_memory`` are
defined in ``distributed/parallel_state.py`` and, across all of
``python/sglang/srt``, called by nobody.

WHAT IS PINNED HERE, and why each one can fail:

* the teardown RUNS on the graceful path when armed -- the fix itself;
* it does NOT run on the exception path, ever. Destroying a group
  synchronises, and on the exception path the GPU may already be wedged; a
  teardown that hangs is worse than the abort it prevents, because the abort at
  least ends the process. (Same guard the neighbouring
  ``release_host_resources`` already carries.)
* it is OFF by default, so the shipped path is byte-identical while #722 owns
  barlink's abort window -- ``GroupCoordinator.destroy`` closes barlink before
  destroying the groups;
* it NEVER raises and is IDEMPOTENT: it runs inside a ``finally`` during
  shutdown, where an exception would replace a clean exit with a traceback, or
  mask the failure that caused the exit.

Hermetic: ``parallel_state`` is replaced by a recorder, so no collectives, no
GPU and no process groups are involved. What is under test is the ORDER and the
CONDITIONS of the calls, which is precisely what the specimens indict.
"""

import sys
import types
import unittest

from sglang.srt.managers import scheduler_teardown
from sglang.test.test_utils import CustomTestCase


class _Args:
    def __init__(self, armed):
        self.scheduler_distributed_teardown = armed


class _Scheduler:
    def __init__(self, armed=True):
        self.server_args = _Args(armed)
        self.gracefully_exit = True


class _Recorder(types.ModuleType):
    """Stands in for parallel_state, recording the destroy sequence."""

    def __init__(self, *, fail=None):
        super().__init__("sglang.srt.distributed.parallel_state")
        self.calls = []
        self._fail = fail or set()

    def destroy_model_parallel(self):
        self.calls.append("destroy_model_parallel")
        if "model" in self._fail:
            raise RuntimeError("model parallel already gone")

    def destroy_distributed_environment(self):
        self.calls.append("destroy_distributed_environment")
        if "env" in self._fail:
            raise RuntimeError("world already gone")


class TestSchedulerTeardown(CustomTestCase):
    def _install(self, recorder):
        """Swap the recorder in where the code LOOKS THE NAME UP.

        ``from sglang.srt.distributed import parallel_state`` resolves the
        package ATTRIBUTE, not ``sys.modules``, so patching sys.modules alone
        leaves the real module in place -- and the real destroy then runs
        inside the test process, quietly succeeding because nothing is
        initialised. Both are patched, so the test cannot pass by accident.
        """
        import sglang.srt.distributed as package

        name = "sglang.srt.distributed.parallel_state"
        real_module = sys.modules.get(name)
        real_attr = getattr(package, "parallel_state", None)
        sys.modules[name] = recorder
        package.parallel_state = recorder

        def _restore():
            if real_module is not None:
                sys.modules[name] = real_module
            else:
                sys.modules.pop(name, None)
            if real_attr is not None:
                package.parallel_state = real_attr

        self.addCleanup(_restore)
        return recorder

    # -- the fix -------------------------------------------------------------

    def test_the_graceful_path_destroys_the_groups(self):
        rec = self._install(_Recorder())
        done = scheduler_teardown.release_distributed(_Scheduler(), graceful=True)
        self.assertEqual(
            rec.calls, ["destroy_model_parallel", "destroy_distributed_environment"]
        )
        self.assertIn("model_parallel", done)

    def test_model_parallel_is_destroyed_before_the_world(self):
        """Order is not cosmetic: the model-parallel groups are built on top of
        the world, so tearing the world down first would destroy groups out
        from under their owner."""
        rec = self._install(_Recorder())
        scheduler_teardown.release_distributed(_Scheduler(), graceful=True)
        self.assertLess(
            rec.calls.index("destroy_model_parallel"),
            rec.calls.index("destroy_distributed_environment"),
        )

    # -- the guards ----------------------------------------------------------

    def test_the_exception_path_never_destroys(self):
        """On the exception path the GPU may be wedged and destroy()
        synchronises; a teardown that hangs is worse than the abort."""
        rec = self._install(_Recorder())
        self.assertIsNone(
            scheduler_teardown.release_distributed(_Scheduler(), graceful=False)
        )
        self.assertEqual(rec.calls, [])

    def test_off_by_default(self):
        """#722 owns barlink's abort window, and GroupCoordinator.destroy
        closes barlink. Shipped default must change nothing."""
        rec = self._install(_Recorder())
        self.assertIsNone(
            scheduler_teardown.release_distributed(
                _Scheduler(armed=False), graceful=True
            )
        )
        self.assertEqual(rec.calls, [])

    def test_a_scheduler_without_server_args_is_a_no_op(self):
        rec = self._install(_Recorder())
        bare = types.SimpleNamespace()
        self.assertIsNone(scheduler_teardown.release_distributed(bare, graceful=True))
        self.assertEqual(rec.calls, [])

    # -- shutdown-safety -----------------------------------------------------

    def test_a_failing_destroy_never_raises(self):
        """It runs in a finally during shutdown: raising here would replace a
        clean exit with a traceback, or mask the failure that caused the exit."""
        rec = self._install(_Recorder(fail={"model", "env"}))
        result = scheduler_teardown.release_distributed(_Scheduler(), graceful=True)
        self.assertIsNone(result)
        # Both were still ATTEMPTED: one failing does not skip the other.
        self.assertEqual(
            rec.calls, ["destroy_model_parallel", "destroy_distributed_environment"]
        )

    def test_a_partial_failure_still_reports_what_succeeded(self):
        rec = self._install(_Recorder(fail={"model"}))
        result = scheduler_teardown.release_distributed(_Scheduler(), graceful=True)
        self.assertEqual(result, "distributed_environment")

    def test_calling_twice_is_safe(self):
        rec = self._install(_Recorder())
        scheduler = _Scheduler()
        scheduler_teardown.release_distributed(scheduler, graceful=True)
        scheduler_teardown.release_distributed(scheduler, graceful=True)
        self.assertEqual(rec.calls.count("destroy_model_parallel"), 2)

    def test_the_gate_reads_the_server_args_flag(self):
        self.assertFalse(scheduler_teardown.distributed_teardown_enabled(_Args(False)))
        self.assertTrue(scheduler_teardown.distributed_teardown_enabled(_Args(True)))
        self.assertFalse(scheduler_teardown.distributed_teardown_enabled(None))


class TestTheOmissionIsReal(CustomTestCase):
    """The premise, pinned so it cannot rot: nothing else tears the groups down.

    If a future change adds a call to the distributed cleanup elsewhere in the
    serving path, this test tells whoever reads it that #673's premise moved --
    which is exactly when the flag's default should be revisited.
    """

    def test_cleanup_dist_env_has_no_caller_in_srt(self):
        import pathlib
        import re

        root = pathlib.Path(scheduler_teardown.__file__).resolve().parents[1]
        pattern = re.compile(r"cleanup_dist_env_and_memory\s*\(")
        callers = []
        for path in root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                snippet = text.splitlines()[line - 1].strip()
                if snippet.startswith("def "):
                    continue
                callers.append(f"{path.name}:{line}")
        self.assertEqual(
            callers,
            [],
            f"cleanup_dist_env_and_memory now has caller(s) {callers}; #673's "
            "premise (the groups are never destroyed on the serving path) has "
            "changed and the teardown flag's default should be revisited.",
        )


if __name__ == "__main__":
    unittest.main()
