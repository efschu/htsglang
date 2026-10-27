# SPDX-License-Identifier: Apache-2.0
"""MPS teardown of ``--dual-mps on`` (boot dkr27bnvfp4dualbar1drei10271237, 27.10.).

THE DEFECT. ``start_dual_mps`` starts a private ``nvidia-cuda-mps-control -d`` (pipe dir
``/tmp/pdflip-dual-mps-<tag>/pipe``). The daemon and its ``nvidia-cuda-mps-server`` are no group's
child, so ``teardown()``'s TERM of the group pids never reached them: the refusal teardown of that boot
printed only ``TERM [424, 440]`` / ``KILL leftovers [440]`` and ``cards_empty=False``; a server that
outlives the groups keeps a context on every card.

Pinned (hermetic, no GPU, no MPS binary):
* ``stop_dual_mps``: ``quit`` (with ``get_server_list`` first) goes to the control program under THIS
  boot's pipe dir; a server that survives the quit is TERMed, then KILLed; the pipe/log dir is removed;
  the line starts ``PDFLIP-TEARDOWN MPS daemon stopped``; a survivor is named ``NOT stopped``;
* nothing armed and nothing there -> no line (a boot without --dual-mps prints nothing); armed but the
  dir gone -> ``absent``; a root outside ``/tmp/pdflip-dual-mps-`` is never removed;
* every exit path: teardown() calls it AFTER the group KILL and BEFORE the cards are read again; the
  cli() early exits (refusal / crash before any group spawned) call it; main() records the root in
  the state json at once.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import signal
import stat
import types

import pytest

from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DUAL_MPS_ROOT_PREFIX", str(tmp_path) + "/pdflip-dual-mps-")
    r = L.dual_mps_root("tagX")
    os.makedirs(r + "/pipe")
    os.makedirs(r + "/log")
    open(r + "/pipe/control", "w").close()
    return r


class _World:
    """A fake process table + clock: ``sleep`` advances the clock; ``quit`` ends the daemon only."""

    def __init__(self, daemon=11, server=22, quit_ends_server=False):
        self.alive_set = {daemon, server}
        self.daemon, self.server = daemon, server
        self.quit_ends_server = quit_ends_server
        self.t = 0.0
        self.runs, self.signals = [], []

    def scan(self, pipe):
        return {"daemon": [p for p in (self.daemon,) if p in self.alive_set],
                "server": [p for p in (self.server,) if p in self.alive_set]}

    def run(self, argv, **kw):
        self.runs.append((argv, kw))
        self.alive_set.discard(self.daemon)
        if self.quit_ends_server:
            self.alive_set.discard(self.server)
        return types.SimpleNamespace(returncode=0, stdout=f"{self.server}\n")

    def alive(self, pid):
        return pid in self.alive_set

    def kill(self, pid, sig):
        self.signals.append((pid, sig))
        if sig == signal.SIGKILL or self.term_works:
            self.alive_set.discard(pid)

    term_works = True

    def sleep(self, s):
        self.t += s

    def clock(self):
        return self.t

    def stop(self, root, **kw):
        return L.stop_dual_mps("tagX", root, armed=True, run=self.run, scan=self.scan, alive=self.alive,
                               kill=self.kill, sleep=self.sleep, clock=self.clock, **kw)


def test_quit_goes_to_this_boots_pipe_and_a_surviving_server_is_termed(root):
    w = _World()
    line = w.stop(root)
    argv, kw = w.runs[0]
    assert argv == ["nvidia-cuda-mps-control"]
    assert kw["input"] == "get_server_list\nquit\n"
    assert kw["env"]["CUDA_MPS_PIPE_DIRECTORY"] == root + "/pipe"
    assert kw["env"]["CUDA_MPS_LOG_DIRECTORY"] == root + "/log"
    assert w.signals == [(22, signal.SIGTERM)]
    assert line.startswith("PDFLIP-TEARDOWN MPS daemon stopped "), line
    assert "daemon_pids=[11]" in line and "server_pids=[22]" in line and "term=[22]" in line, line
    assert "dir_removed=True" in line and not os.path.exists(root)


def test_a_server_that_ignores_term_is_killed_and_a_survivor_is_named(root):
    w = _World()
    w.term_works = False
    line = w.stop(root)
    assert w.signals == [(22, signal.SIGTERM), (22, signal.SIGKILL)]
    assert line.startswith("PDFLIP-TEARDOWN MPS daemon stopped ") and "kill=[22]" in line, line

    root2 = L.dual_mps_root("tagY")
    os.makedirs(root2 + "/pipe")
    w2 = _World()
    w2.kill = lambda pid, sig: w2.signals.append((pid, sig))  # nothing dies
    line2 = L.stop_dual_mps("tagY", root2, armed=True, run=w2.run, scan=w2.scan, alive=w2.alive, kill=w2.kill,
                            sleep=w2.sleep, clock=w2.clock)
    assert line2.startswith("PDFLIP-TEARDOWN MPS daemon NOT stopped"), line2
    assert "alive=[22]" in line2


def test_a_clean_quit_needs_no_signal(root):
    w = _World(quit_ends_server=True)
    line = w.stop(root)
    assert w.signals == [] and "term=[]" in line and line.startswith("PDFLIP-TEARDOWN MPS daemon stopped ")


def test_no_daemon_and_no_control_pipe_never_starts_the_control_program(root):
    os.unlink(root + "/pipe/control")
    w = _World()
    w.alive_set.clear()
    line = w.stop(root)
    assert w.runs == [], "nvidia-cuda-mps-control without -d would START a daemon in the foreground"
    assert "quit_rc=not-run" in line and not os.path.exists(root)


def test_nothing_armed_nothing_there_prints_nothing_armed_but_gone_says_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DUAL_MPS_ROOT_PREFIX", str(tmp_path) + "/pdflip-dual-mps-")
    assert L.stop_dual_mps("none") == ""
    assert L.stop_dual_mps("") == ""
    assert L.stop_dual_mps("none", armed=True).startswith("PDFLIP-TEARDOWN MPS daemon absent")


def test_a_root_outside_the_prefix_is_never_touched(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    assert L.stop_dual_mps("x", str(victim), armed=True) == ""
    assert victim.exists()


def test_a_missing_control_binary_is_named_not_raised(root, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-1027")
    line = L.stop_dual_mps("tagX", root, armed=True, scan=lambda p: {"daemon": [], "server": []})
    assert "quit_rc=no-control-binary" in line and not os.path.exists(root)


def test_the_real_control_program_call_end_to_end(root, tmp_path, monkeypatch):
    """A fake nvidia-cuda-mps-control on PATH: the real subprocess carries the input and this boot's env."""
    rec = tmp_path / "rec.txt"
    ctl = tmp_path / "bin" / "nvidia-cuda-mps-control"
    ctl.parent.mkdir()
    ctl.write_text('#!/bin/bash\necho "pipe=$CUDA_MPS_PIPE_DIRECTORY" > "$FAKE_MPS_REC"\ncat >> "$FAKE_MPS_REC"\n')
    ctl.chmod(ctl.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(ctl.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_MPS_REC", str(rec))
    line = L.stop_dual_mps("tagX", root, armed=True, scan=lambda p: {"daemon": [], "server": []})
    assert rec.read_text() == f"pipe={root}/pipe\nget_server_list\nquit\n"
    assert "quit_rc=0" in line and line.startswith("PDFLIP-TEARDOWN MPS daemon stopped ")


def test_a_zombie_does_not_count_as_alive(tmp_path):
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    (proc / "123" / "stat").write_text("123 (nvidia-cuda-mps) Z 1 1 1 0 -1 0")
    (proc / "124").mkdir()
    (proc / "124" / "stat").write_text("124 (nvidia-cuda-mps) S 1 1 1 0 -1 0")
    me = os.getpid()
    (proc / str(me)).mkdir()
    (proc / str(me) / "stat").write_text(f"{me} (a b) c) Z 1")  # a ')' inside comm: split on the LAST one
    assert L._alive_not_zombie(me, proc_root=str(proc)) is False
    (proc / str(me) / "stat").write_text(f"{me} (a b) S 1")
    assert L._alive_not_zombie(me, proc_root=str(proc)) is True


def test_the_process_scan_matches_argv0_and_this_boots_pipe(tmp_path):
    proc = tmp_path / "proc"
    pipe = "/tmp/pdflip-dual-mps-A/pipe"

    def mk(pid, argv, env):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "environ").write_bytes(b"\0".join(e.encode() for e in env) + b"\0")

    mk(1, ["/usr/bin/nvidia-cuda-mps-control", "-d"], [f"CUDA_MPS_PIPE_DIRECTORY={pipe}", "A=b"])
    mk(2, ["nvidia-cuda-mps-server"], [f"CUDA_MPS_PIPE_DIRECTORY={pipe}"])
    mk(3, ["nvidia-cuda-mps-control", "-d"], ["CUDA_MPS_PIPE_DIRECTORY=/tmp/pdflip-dual-mps-OTHER/pipe"])
    mk(4, ["python3", "x"], [f"CUDA_MPS_PIPE_DIRECTORY={pipe}"])
    assert L._mps_pids_on_pipe(pipe, proc_root=str(proc)) == {"daemon": [1], "server": [2]}


# -- every exit path -------------------------------------------------------------------------------


def test_teardown_stops_the_daemon_after_the_group_kill_and_before_the_cards_are_read_again():
    src = inspect.getsource(L.teardown)
    kill = src.index('print(f"KILL leftovers {left}")')
    mps = src.index("stop_dual_mps(")
    store = src.index("store_dir = st.get")
    assert kill < mps < store
    assert 'st.get("dual_mps_root"' in src


def test_main_records_the_mps_root_in_the_state_at_once():
    src = inspect.getsource(L.main)
    i = src.index("ns._dual_mps_env = start_dual_mps(ns, log, dry)")
    tail = src[i:i + 500]
    assert "state.dual_mps_root = dual_mps_root(ns.tag)" in tail and "_write_state(state)" in tail
    assert "dual_mps_root" in {f.name for f in dataclasses.fields(L.BootState)}


def _state(root, pids):
    st = L.BootState(tag="tagX", tip="t", tree="/x", stamp="s")
    st.pids = pids
    st.dual_mps_root = root
    return st


def _cli_with(monkeypatch, exc, pids, root, capsys):
    calls = []
    monkeypatch.setattr(L, "stop_dual_mps", lambda tag, r="", **kw: calls.append((tag, r, kw)) or "MPSLINE")
    st = _state(root, pids)

    def fake_main(argv=None):
        L._ACTIVE_BOOT_STATE = st
        raise exc

    monkeypatch.setattr(L, "main", fake_main)
    monkeypatch.setattr(L, "boot_state_write", lambda *a, **k: None)
    monkeypatch.setattr(L, "teardown", lambda *a, **k: calls.append("teardown"))
    monkeypatch.setattr(L, "_write_state", lambda s: None)
    monkeypatch.setattr(L, "_cards_empty", lambda c: True)
    monkeypatch.setattr(L, "drop_admin_key_file", lambda: "gone")
    monkeypatch.setattr(L, "_ACTIVE_BOOT_STATE", None)
    return calls


def test_a_refusal_before_any_group_spawned_stops_the_daemon(monkeypatch, capsys):
    calls = _cli_with(monkeypatch, L.PdFlipLaunchRefused("W1 pre-spawn"), {"P": 0, "D": 0}, "/tmp/pdflip-dual-mps-tagX",
                      capsys)
    assert L.cli(["--fake"]) == 2
    assert calls == [("tagX", "/tmp/pdflip-dual-mps-tagX", {"armed": True})]
    assert "MPSLINE" in capsys.readouterr().out


def test_a_crash_before_any_group_spawned_stops_the_daemon_and_reraises(monkeypatch, capsys):
    monkeypatch.setattr(L.state_file_mod, "make_cause", lambda *a, **k: {})
    calls = _cli_with(monkeypatch, ValueError("boom"), {"P": 0}, "/tmp/pdflip-dual-mps-tagX", capsys)
    with pytest.raises(ValueError):
        L.cli(["--fake"])
    assert calls == [("tagX", "/tmp/pdflip-dual-mps-tagX", {"armed": True})]


def test_a_crash_with_a_running_group_leaves_the_daemon_to_teardown(monkeypatch, capsys):
    monkeypatch.setattr(L.state_file_mod, "make_cause", lambda *a, **k: {})
    calls = _cli_with(monkeypatch, ValueError("boom"), {"P": 4242}, "/tmp/pdflip-dual-mps-tagX", capsys)
    with pytest.raises(ValueError):
        L.cli(["--fake"])
    assert calls == [], "stopping MPS under live clients is undefined; teardown() ends clients then daemon"


def test_a_refusal_after_a_spawn_goes_through_teardown_only(monkeypatch, capsys):
    calls = _cli_with(monkeypatch, L.PdFlipLaunchRefused("W45 post-spawn"), {"P": 4242}, "/tmp/pdflip-dual-mps-tagX",
                      capsys)
    assert L.cli(["--fake"]) == 2
    assert calls == ["teardown"]
