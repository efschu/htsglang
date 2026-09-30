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

from flliper.srt.pdflip import rpc_stall_watchdog as W


class _Leg:
    def __init__(self, hang_s):
        self.hang_s = hang_s

    def _pdflip_rank(self):
        return 0

    def _pdflip_group_name(self):
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
    return sorted(glob.glob(os.path.join(d, "pdflip_rpcstall_*.txt")))


def test_default_timeout_is_3_s():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FLLIPER_PDFLIP_RPC_STALL_WATCHDOG_S", None)
        assert W.timeout_s() == 3.0


def test_a_hang_over_3_s_writes_the_stacks(tmp_path):
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_EVIDENCE_DIR": str(tmp_path)}):
        os.environ.pop("FLLIPER_PDFLIP_RPC_STALL_WATCHDOG_S", None)          # the default 3 s
        assert _Leg(3.4).resume(1) == 2
    files = _files(tmp_path)
    assert len(files) == 1 and "_D_r0_resume_" in os.path.basename(files[0])
    text = open(files[0]).read()
    assert text.startswith("RPC-STALL-WATCHDOG group=D rank=0 kind=resume")
    assert "_sleep_in_a_named_frame" in text and "Thread" in text     # the frame it sat in


def test_a_normal_leg_leaves_no_file(tmp_path):
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_EVIDENCE_DIR": str(tmp_path)}):
        os.environ.pop("FLLIPER_PDFLIP_RPC_STALL_WATCHDOG_S", None)
        assert _Leg(0.05).resume(1) == 2
    assert _files(tmp_path) == []


def test_a_raising_leg_disarms(tmp_path):
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_EVIDENCE_DIR": str(tmp_path),
                                      "FLLIPER_PDFLIP_RPC_STALL_WATCHDOG_S": "0.3"}):
        with pytest.raises(RuntimeError):
            _Leg(0).release()
        time.sleep(0.5)                                                    # a timer left armed would fire
    assert _files(tmp_path) == []


def test_off_writes_nothing(tmp_path):
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_EVIDENCE_DIR": str(tmp_path),
                                      "FLLIPER_PDFLIP_RPC_STALL_WATCHDOG_S": "0"}):
        assert W.arm("resume", rank=0, group="D") is None
        assert _Leg(0.05).resume(1) == 2
    assert _files(tmp_path) == []


def test_both_weight_updater_rpcs_are_watched():
    from flliper.srt.managers.scheduler_components import weight_updater as WU

    cls = WU.SchedulerWeightUpdaterManager
    for name in ("release_memory_occupation", "resume_memory_occupation"):
        ours = W.watched("x")(lambda self: None).__code__
        fn, seen = getattr(cls, name), []
        while fn is not None:                      # other wrappers may sit around ours
            seen.append(fn.__code__)
            fn = getattr(fn, "__wrapped__", None)
        assert ours in seen, name
