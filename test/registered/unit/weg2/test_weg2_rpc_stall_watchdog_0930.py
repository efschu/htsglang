"""RPC-STALL-WATCHDOG (30.09., hauenh P->D epoch 6: D TP0 silent for 6 s inside the wake RPC).

Pinned:
  * an artificial hang over the DEFAULT 3 s writes every thread's stack (the hanging frame named) into
    its own per-rank file, and the handler's result passes through;
  * a normal (short) leg leaves no file behind;
  * a raising handler disarms too (no timer left armed, no stray file);
  * 0 = off: no file, no timer;
  * both RPC handlers of the weight updater (release = sleep, resume = wake) are wrapped, for P and D alike.
"""

import glob
import os
import time
from unittest import mock

import pytest

from sglang.srt.weg2 import rpc_stall_watchdog as W


class _Leg:
    def __init__(self, hang_s):
        self.hang_s = hang_s

    def _weg2_rank(self):
        return 0

    def _weg2_group_name(self):
        return "D"

    @W.watched("resume")
    def resume(self, x):
        _sleep_in_a_named_frame(self.hang_s)
        return x + 1

    @W.watched("release")
    def release(self):
        raise RuntimeError("leg failed")


def _sleep_in_a_named_frame(s):
    time.sleep(s)


def _files(d):
    return sorted(glob.glob(os.path.join(d, "weg2_rpcstall_*.txt")))


def test_default_timeout_is_3_s():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_WEG2_RPC_STALL_WATCHDOG_S", None)
        assert W.timeout_s() == 3.0


def test_a_hang_over_3_s_writes_the_stacks(tmp_path):
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path)}):
        os.environ.pop("SGLANG_WEG2_RPC_STALL_WATCHDOG_S", None)          # the default 3 s
        assert _Leg(3.4).resume(1) == 2
    files = _files(tmp_path)
    assert len(files) == 1 and "_D_r0_resume_" in os.path.basename(files[0])
    text = open(files[0]).read()
    assert text.startswith("RPC-STALL-WATCHDOG group=D rank=0 kind=resume")
    assert "_sleep_in_a_named_frame" in text and "Thread" in text     # the frame it sat in


def test_a_normal_leg_leaves_no_file(tmp_path):
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path)}):
        os.environ.pop("SGLANG_WEG2_RPC_STALL_WATCHDOG_S", None)
        assert _Leg(0.05).resume(1) == 2
    assert _files(tmp_path) == []


def test_a_raising_leg_disarms(tmp_path):
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path),
                                      "SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0.3"}):
        with pytest.raises(RuntimeError):
            _Leg(0).release()
        time.sleep(0.5)                                                    # a timer left armed would fire
    assert _files(tmp_path) == []


def test_off_writes_nothing(tmp_path):
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path),
                                      "SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0"}):
        assert W.arm("resume", rank=0, group="D") is None
        assert _Leg(0.05).resume(1) == 2
    assert _files(tmp_path) == []


def test_both_weight_updater_rpcs_are_watched():
    from sglang.srt.managers.scheduler_components import weight_updater as WU

    cls = WU.SchedulerWeightUpdaterManager
    for name in ("release_memory_occupation", "resume_memory_occupation"):
        ours = W.watched("x")(lambda self: None).__code__
        fn, seen = getattr(cls, name), []
        while fn is not None:                      # other wrappers may sit around ours
            seen.append(fn.__code__)
            fn = getattr(fn, "__wrapped__", None)
        assert ours in seen, name


def test_z30y8_no_faulthandler_dump_the_python_sampler_writes_under_the_gil(tmp_path):
    """z30y8 (18:41:55, P PP1 SIGSEGV): faulthandler.dump_traceback_later walked the main thread's
    frames without the GIL while it was in logging.emit -- the dump stopped mid-stack and the rank died.
    The watchdog must not call it any more; the Python sampler still names the hanging frame."""
    import faulthandler

    with mock.patch.object(faulthandler, "dump_traceback_later",
                           side_effect=AssertionError("dump_traceback_later called")) as dtl, \
            mock.patch.dict(os.environ, {"SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path),
                                         "SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0.3"}):
        assert _Leg(0.6).resume(1) == 2
    assert dtl.call_count == 0
    text = open(_files(tmp_path)[0]).read()
    assert "Python sampler, under the GIL" in text and "_sleep_in_a_named_frame" in text
    assert "rpc-stall-sampler" not in text           # the sampler leaves itself out


def test_a_second_dump_follows_while_the_rpc_still_runs(tmp_path):
    with mock.patch.object(W, "REPEAT_S", 0.3), \
            mock.patch.dict(os.environ, {"SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path),
                                         "SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0.2"}):
        assert _Leg(0.9).resume(1) == 2
    text = open(_files(tmp_path)[0]).read()
    assert text.count("_sleep_in_a_named_frame") >= 2 and "Second dump" in text


def test_the_shared_stall_sampler_interface(tmp_path):
    """weg2/stall_sampler.py is the shared piece (NF's TAG-STALL-SENTINEL can switch to it):
    arm(fh, timeout_s, repeat_s) / disarm(s, join_s)."""
    from sglang.srt.weg2 import stall_sampler as S

    p = tmp_path / "s.txt"
    with open(p, "w") as fh:
        s = S.arm(fh, 0.2, repeat_s=0)
        _sleep_in_a_named_frame(0.5)
        S.disarm(s, 1.0)
        assert s.dumps == 1 and not s.thread.is_alive()
    assert "_sleep_in_a_named_frame" in p.read_text()
    with open(tmp_path / "q.txt", "w") as fh:
        s = S.arm(fh, 5.0)
        S.disarm(s, 1.0)                                 # stopped before the timeout: nothing written
        assert s.dumps == 0 and not s.thread.is_alive()
