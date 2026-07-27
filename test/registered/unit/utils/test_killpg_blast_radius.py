"""Blast-radius guard for the opt-in scheduler killpg (#259, second finding).

A neighbouring server instance -- its own process tree, its own launch -- died
without a traceback while an OOM'd instance was being torn down.
``kill_process_tree`` cannot do that: it walks psutil children and never leaves
the tree. ``os.killpg`` can: a process group is inherited, so two servers
started from the same shell share one pgid, and SIGKILL to that group is
uncatchable and therefore silent on the victim.
"""

import unittest

import psutil

from sglang.srt.utils.common import process_group_is_confined_to_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid


def _fake_world(pid_to_pgid):
    """Build (process_iter, getpgid) stand-ins for a synthetic process table."""

    def process_iter(_attrs=None):
        return [_FakeProc(pid) for pid in pid_to_pgid]

    def getpgid(pid):
        try:
            return pid_to_pgid[pid]
        except KeyError:
            raise ProcessLookupError(pid)

    return process_iter, getpgid


class TestKillpgBlastRadius(unittest.TestCase):
    def setUp(self):
        self.me = psutil.Process()
        # This test process plus its children are the "owned" tree; the
        # synthetic foreign/own pids below are chosen not to collide with it.
        self.owned_pids = {self.me.pid} | {
            c.pid for c in self.me.children(recursive=True)
        }

    def _pid_not_owned(self, start=10_000_000):
        pid = start
        while pid in self.owned_pids:
            pid += 1
        return pid

    def test_group_shared_with_a_foreign_process_is_not_confined(self):
        foreign = self._pid_not_owned()
        process_iter, getpgid = _fake_world({self.me.pid: 4242, foreign: 4242})
        self.assertFalse(
            process_group_is_confined_to_tree(
                self.me.pid, 4242, _process_iter=process_iter, _getpgid=getpgid
            )
        )

    def test_group_holding_only_our_tree_is_confined(self):
        other_group = self._pid_not_owned()
        process_iter, getpgid = _fake_world(
            {self.me.pid: 4242, other_group: 9999}  # different pgid -> irrelevant
        )
        self.assertTrue(
            process_group_is_confined_to_tree(
                self.me.pid, 4242, _process_iter=process_iter, _getpgid=getpgid
            )
        )

    def test_vanished_process_does_not_break_the_scan(self):
        gone = self._pid_not_owned()
        process_iter, _ = _fake_world({self.me.pid: 4242, gone: 4242})

        def getpgid(pid):
            if pid == gone:
                raise ProcessLookupError(pid)
            return 4242

        self.assertTrue(
            process_group_is_confined_to_tree(
                self.me.pid, 4242, _process_iter=process_iter, _getpgid=getpgid
            )
        )

    def test_enumeration_failure_fails_closed(self):
        def boom(_attrs=None):
            raise RuntimeError("no /proc")

        self.assertFalse(
            process_group_is_confined_to_tree(
                self.me.pid, 4242, _process_iter=boom, _getpgid=lambda pid: 4242
            )
        )

    def test_dead_root_fails_closed(self):
        dead = self._pid_not_owned(start=20_000_000)
        process_iter, getpgid = _fake_world({dead: 4242})
        self.assertFalse(
            process_group_is_confined_to_tree(
                dead, 4242, _process_iter=process_iter, _getpgid=getpgid
            )
        )


if __name__ == "__main__":
    unittest.main()
