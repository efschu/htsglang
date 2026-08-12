# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#656: SIGTERM during the launch phase must not orphan the rank processes.

VAL-R4, restoring serving: ``SIGTERM`` to the launcher **orphaned the three
rank processes**, which kept ~55 GB of VRAM across all three cards while the
parent was already gone. The replacement boot would have OOM'd.

The mechanism, at file:line as it stood:

* ``engine.py`` installed a launch-phase handler for **SIGQUIT only**. SIGTERM
  kept Python's default disposition -- immediate termination, no ``finally``,
  no ``atexit`` -- for the whole of ``_launch_subprocesses`` -> weight load ->
  warmup, i.e. the entire multi-minute boot window.
* The ranks are ``mp.Process`` children (non-daemonic) in the parent's own
  process group, so nothing else reaped them either.
* ``launch_server.py``'s ``finally: kill_process_tree(...)`` does not run on a
  default-disposition SIGTERM, and neither does ``Engine.shutdown``'s
  ``atexit``.
* The running-phase SIGTERM handler in ``tokenizer_manager`` is installed by
  ``auto_create_handle_loop()``, which runs on the FIRST REQUEST -- so it
  covers a serving instance and specifically not a booting one.

Under systemd the gap is masked: ``htsglang-serving@.service`` sets
``KillMode=control-group``, so the cgroup kill reaches the ranks. The shipped
shell path (``route_a_631_prod_boot.sh``, ``setsid``) has no such cover, which
is the path VAL-R4 was on.

These tests are hermetic: they assert the DISPOSITION is installed and that it
reaps the tree, without sending a real signal to the test runner.
"""

import signal
import threading
import unittest
from unittest import mock

from sglang.srt.entrypoints import engine as E
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Args:
    """The two attributes the handler installer reads."""

    def __init__(self, custom_sigquit_handler=None):
        self.custom_sigquit_handler = custom_sigquit_handler


class _SignalTableGuard:
    """Capture what gets installed without disturbing the test process."""

    def __enter__(self):
        self.installed = {}
        self._patch = mock.patch.object(
            signal, "signal",
            side_effect=lambda sig, h: self.installed.__setitem__(sig, h))
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class TestLaunchPhaseSigtermIsHandled(CustomTestCase):
    def test_sigterm_gets_a_launch_phase_handler(self):
        """THE fix test. Red before the fix: only SIGQUIT was installed."""
        with _SignalTableGuard() as g:
            E._install_launch_phase_signal_handlers(_Args())
        self.assertIn(signal.SIGTERM, g.installed,
                      "SIGTERM kept its default disposition during the launch "
                      "phase -- this is the orphaned-ranks bug")

    def test_sigquit_is_still_installed(self):
        """The existing child-failure path must be untouched."""
        with _SignalTableGuard() as g:
            E._install_launch_phase_signal_handlers(_Args())
        self.assertIn(signal.SIGQUIT, g.installed)

    def test_the_sigterm_handler_reaps_the_process_tree(self):
        """A handler that only logs would leave the ranks holding VRAM. It
        must call the tree reaper, which is what walks the mp.Process
        children the launcher spawned."""
        with _SignalTableGuard() as g:
            E._install_launch_phase_signal_handlers(_Args())
        handler = g.installed[signal.SIGTERM]
        with mock.patch.object(E, "kill_process_tree") as reaper, \
                mock.patch.object(E.sys, "exit") as _exit:
            handler(signal.SIGTERM, None)
        self.assertTrue(reaper.called,
                        "the SIGTERM handler did not reap the process tree")

    def test_the_sigterm_handler_exits_nonzero_rather_than_returning(self):
        """Returning from a signal handler resumes the interrupted boot. A
        launcher that keeps loading weights after being asked to stop is the
        same orphan bug wearing a different hat."""
        with _SignalTableGuard() as g:
            E._install_launch_phase_signal_handlers(_Args())
        handler = g.installed[signal.SIGTERM]
        with mock.patch.object(E, "kill_process_tree"), \
                mock.patch.object(E.sys, "exit") as _exit:
            handler(signal.SIGTERM, None)
        self.assertTrue(_exit.called, "handler returned into the boot")

    def test_a_custom_sigquit_handler_does_not_disable_the_sigterm_one(self):
        """``--custom-sigquit-handler`` is about crash dumps on child
        failure. It must not silently take the orphan guard with it."""
        with _SignalTableGuard() as g:
            E._install_launch_phase_signal_handlers(_Args(lambda *a: None))
        self.assertIn(signal.SIGTERM, g.installed)
        self.assertIn(signal.SIGQUIT, g.installed)

    def test_nothing_is_installed_off_the_main_thread(self):
        """Signal handlers can only be registered from the main thread;
        attempting it raises. The installer must decline, not throw, because
        it runs inside Engine() which embedders construct off-thread."""
        out = {}

        def run():
            with _SignalTableGuard() as g:
                E._install_launch_phase_signal_handlers(_Args())
                out["installed"] = dict(g.installed)

        t = threading.Thread(target=run)
        t.start()
        t.join()
        self.assertEqual(out["installed"], {})


if __name__ == "__main__":
    unittest.main()
