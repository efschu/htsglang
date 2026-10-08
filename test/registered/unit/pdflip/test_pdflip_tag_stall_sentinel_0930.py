"""TAG-STALL-SENTINEL + P GC warning (30.09., NF y3z ep52).

Metal: PP0 stood still for 5.4 s process-wide -- every thread at once -- at
the first tag of P's sleep. Nothing named the frame: a Python sampler needs
the GIL the stall may hold, and the rank-side GC warning was off on P
(``--gc-warning-threshold-secs`` 0.0, FLLIPER_PDFLIP_GC_WARN_SECS absent).

* Per sleep tag, faulthandler's C watchdog dumps every thread's stack into
  ``pdflip_tagstall_<group>_r<rank>_<tag>_*.txt`` once the tag outlives
  FLLIPER_PDFLIP_TAG_STALL_SENTINEL_S (default 1.5); the tag's end reads the
  file size and names a fired dump, an empty one is removed.
* The launcher arms the scheduler GC warning for group P at 0.5 s by default.
RED on 1eac8b3461 (no sentinel, no P default), GREEN with the fix.
"""

from __future__ import annotations

import inspect
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _sentinel():
    from flliper.srt.pdflip import tag_stall_sentinel

    return tag_stall_sentinel


def _one_tag(tmp_path, seconds, tag="weights_0"):
    ts = _sentinel()
    armed = ts.arm(tag, rank=0, group="P", directory=str(tmp_path))
    time.sleep(seconds)
    return ts.disarm(armed)


def test_a_tag_that_stalls_2s_leaves_every_threads_stack_under_its_name(tmp_path):
    fired = _one_tag(tmp_path, 2.0)
    assert fired is not None and os.path.basename(fired).startswith("pdflip_tagstall_P_r0_weights_0_")
    text = open(fired).read()
    assert "tag=weights_0" in text.splitlines()[0]
    assert "Thread 0x" in text and "under the GIL" in text  # the GIL sampler's stack dump
    assert "test_pdflip_tag_stall_sentinel_0930.py" in text  # the frame the stall sat in
    # the sleep loop arms and disarms it around every tag
    from flliper.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as M,
    )

    src = inspect.getsource(M.release_memory_occupation)
    assert "_tag_stall.arm(tag" in src and "_tag_stall.disarm(_stall)" in src


def test_a_short_tag_leaves_no_file(tmp_path):
    assert _one_tag(tmp_path, 0.1) is None
    assert os.listdir(tmp_path) == []


def test_the_switch_off_arms_nothing(tmp_path):
    with envs.FLLIPER_PDFLIP_TAG_STALL_SENTINEL_S.override(0.0):
        assert _sentinel().arm("weights_0", rank=0, group="P", directory=str(tmp_path)) is None
        assert _one_tag(tmp_path, 2.0) is None
    assert os.listdir(tmp_path) == []


def test_the_gc_warning_is_armed_for_group_p_by_default(monkeypatch):
    from flliper.srt.pdflip import gc_instrument as gci
    from flliper.srt.pdflip import launcher as L

    ns = types.SimpleNamespace(env_p="")
    assert L.apply_p_gc_warn_default(ns) is not None
    env_p = L.parse_group_env(ns.env_p)
    assert env_p["FLLIPER_PDFLIP_GC_WARN_SECS"] == "0.5"
    # the P rank arms the warning from that env (no CLI flag)
    armed = []
    monkeypatch.setattr("flliper.srt.utils.common.configure_gc_warning", armed.append)
    out = gci.arm_after_boot(types.SimpleNamespace(gc_warning_threshold_secs=0.0), 0, env=env_p)
    assert out["warn"] == pytest.approx(0.5) and armed == [0.5]
    # --env-p names it: the stated value wins (here: off)
    ns2 = types.SimpleNamespace(env_p="FLLIPER_PDFLIP_GC_WARN_SECS=0")
    assert L.apply_p_gc_warn_default(ns2) is None
    assert gci.arm_after_boot(types.SimpleNamespace(gc_warning_threshold_secs=0.0), 0,
                              env=L.parse_group_env(ns2.env_p))["warn"] is None


def test_no_faulthandler_walk_without_the_gil():
    """27B z30y8: faulthandler's later-dump walked a changing frame without the
    GIL into a PP1 SIGSEGV. The sentinel arms the shared GIL sampler instead."""
    src = inspect.getsource(_sentinel())
    assert "dump_traceback_later(" not in src.replace("``faulthandler.dump_traceback_later``", "")
    assert "stall_sampler.arm(fh" in src and "stall_sampler.disarm(armed.sampler)" in src


def test_a_gil_held_stall_is_named_by_late_ms(tmp_path, caplog):
    """A stall holding the GIL delays the dump until release; late_ms says so."""
    import logging
    import threading

    ts = _sentinel()
    with caplog.at_level(logging.WARNING):
        armed = ts.arm("weights_1", rank=0, group="P", directory=str(tmp_path), timeout=0.2)
        t_end = time.perf_counter() + 1.0
        x = 0
        while time.perf_counter() < t_end:  # pure-Python spin: the sampler still gets the GIL
            x += 1
        fired = ts.disarm(armed)
    assert fired is not None and threading.active_count() >= 1
    line = next(m for m in caplog.messages if "TAG-STALL-SENTINEL fired" in m)
    assert "late_ms=" in line


# ------------------------------------------------------------------ y6b
# Review cda5a88fee: F1 a stall that held the GIL until the tag ended left no
# evidence (the sampler woke to `stop`, the file was removed, nothing logged);
# F2 late_ms came from the file's mtime = the LAST write (a repeat dump read as
# ~8000 ms "late"), on the wall clock; F4 a GIL never released left a
# header-only file nobody named; F7 a failed arm leaked the fd and the file.


def test_y6b_a_stall_the_sampler_never_saw_is_named_not_removed(tmp_path, caplog):
    import logging

    ts = _sentinel()
    with caplog.at_level(logging.WARNING):
        armed = ts.arm("weights_2", rank=0, group="P", directory=str(tmp_path), timeout=0.2)
        # the sampler woke only to `stop` -- what a GIL held in C until the tag's
        # end leaves it (it gets the GIL after disarm set the event)
        armed.sampler.stop.set()
        armed.sampler.thread.join(1.0)
        time.sleep(0.4)
        fired = ts.disarm(armed)
    assert fired is not None and os.path.exists(fired), "kept, not removed"
    line = next(m for m in caplog.messages if "TAG-STALL-SENTINEL fired" in m)
    assert "dump=missed" in line and "late_ms=" in line
    assert "No dump" in open(fired).read()


def test_y6b_late_ms_is_the_first_dump_not_the_last_write(tmp_path):
    from flliper.srt.pdflip import stall_sampler

    fh = open(tmp_path / "s.txt", "w")
    fh.write("hdr\n")
    s = stall_sampler.arm(fh, 0.2, repeat_s=0.3)
    t0 = time.time()
    time.sleep(0.8)                                 # a pure-Python tag: both dumps fire on time
    stall_sampler.disarm(s)
    fh.close()
    assert s.dumps == 2
    assert s.first_late_ms is not None and s.first_late_ms < 150, s.first_late_ms
    mtime_late_ms = (os.stat(tmp_path / "s.txt").st_mtime - (t0 + 0.2)) * 1000
    assert mtime_late_ms > 250, "the old reading: the repeat dump's mtime"
    assert not stall_sampler.missed(s)


def test_y6b_a_header_only_file_is_named_unreleased(tmp_path):
    ts = _sentinel()
    now = time.time()
    old = tmp_path / "pdflip_tagstall_P_r0_weights_0_1.txt"
    old.write_text("%s group=P rank=0 tag=weights_0 armed_unix=%.3f timeout_s=1.500 pid=4242\n"
                   % (ts.MARKER, now - 60))
    fired = tmp_path / "pdflip_tagstall_P_r0_weights_1_2.txt"
    fired.write_text("%s group=P rank=0 tag=weights_1 armed_unix=%.3f timeout_s=1.500 pid=4242\n"
                     "Timeout (1.5 s): ...\n" % (ts.MARKER, now - 60))
    fresh = tmp_path / "pdflip_tagstall_P_r0_weights_2_3.txt"
    fresh.write_text("%s group=P rank=0 tag=weights_2 armed_unix=%.3f timeout_s=1.500 pid=4242\n"
                     % (ts.MARKER, now))
    rows = ts.unreleased(str(tmp_path), now_unix=now)
    assert [r[0] for r in rows] == [str(old)], "header only AND past its timeout"
    assert rows[0][3] == 4242


def test_y6b_a_failed_arm_leaves_no_file(tmp_path, monkeypatch):
    from flliper.srt.pdflip import stall_sampler

    def boom(*a, **k):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(stall_sampler, "arm", boom)
    assert _sentinel().arm("weights_0", rank=0, group="P", directory=str(tmp_path), timeout=0.2) is None
    assert os.listdir(tmp_path) == []
