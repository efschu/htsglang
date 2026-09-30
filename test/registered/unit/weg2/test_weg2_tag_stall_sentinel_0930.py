"""TAG-STALL-SENTINEL + P GC warning (30.09., NF y3z ep52).

Metal: PP0 stood still for 5.4 s process-wide -- every thread at once -- at
the first tag of P's sleep. Nothing named the frame: a Python sampler needs
the GIL the stall may hold, and the rank-side GC warning was off on P
(``--gc-warning-threshold-secs`` 0.0, SGLANG_WEG2_GC_WARN_SECS absent).

* Per sleep tag, faulthandler's C watchdog dumps every thread's stack into
  ``weg2_tagstall_<group>_r<rank>_<tag>_*.txt`` once the tag outlives
  SGLANG_WEG2_TAG_STALL_SENTINEL_S (default 1.5); the tag's end reads the
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

from sglang.srt.environ import envs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _sentinel():
    from sglang.srt.weg2 import tag_stall_sentinel

    return tag_stall_sentinel


def _one_tag(tmp_path, seconds, tag="weights_0"):
    ts = _sentinel()
    armed = ts.arm(tag, rank=0, group="P", directory=str(tmp_path))
    time.sleep(seconds)
    return ts.disarm(armed)


def test_a_tag_that_stalls_2s_leaves_every_threads_stack_under_its_name(tmp_path):
    fired = _one_tag(tmp_path, 2.0)
    assert fired is not None and os.path.basename(fired).startswith("weg2_tagstall_P_r0_weights_0_")
    text = open(fired).read()
    assert "tag=weights_0" in text.splitlines()[0]
    assert "Thread 0x" in text or "Current thread" in text  # faulthandler's stack dump
    assert "test_weg2_tag_stall_sentinel_0930.py" in text  # the frame the stall sat in
    # the sleep loop arms and disarms it around every tag
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as M,
    )

    src = inspect.getsource(M.release_memory_occupation)
    assert "_tag_stall.arm(tag" in src and "_tag_stall.disarm(_stall)" in src


def test_a_short_tag_leaves_no_file(tmp_path):
    assert _one_tag(tmp_path, 0.1) is None
    assert os.listdir(tmp_path) == []


def test_the_switch_off_arms_nothing(tmp_path):
    with envs.SGLANG_WEG2_TAG_STALL_SENTINEL_S.override(0.0):
        assert _sentinel().arm("weights_0", rank=0, group="P", directory=str(tmp_path)) is None
        assert _one_tag(tmp_path, 2.0) is None
    assert os.listdir(tmp_path) == []


def test_the_gc_warning_is_armed_for_group_p_by_default(monkeypatch):
    from sglang.srt.weg2 import gc_instrument as gci
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(env_p="")
    assert L.apply_p_gc_warn_default(ns) is not None
    env_p = L.parse_group_env(ns.env_p)
    assert env_p["SGLANG_WEG2_GC_WARN_SECS"] == "0.5"
    # the P rank arms the warning from that env (no CLI flag)
    armed = []
    monkeypatch.setattr("sglang.srt.utils.common.configure_gc_warning", armed.append)
    out = gci.arm_after_boot(types.SimpleNamespace(gc_warning_threshold_secs=0.0), 0, env=env_p)
    assert out["warn"] == pytest.approx(0.5) and armed == [0.5]
    # --env-p names it: the stated value wins (here: off)
    ns2 = types.SimpleNamespace(env_p="SGLANG_WEG2_GC_WARN_SECS=0")
    assert L.apply_p_gc_warn_default(ns2) is None
    assert gci.arm_after_boot(types.SimpleNamespace(gc_warning_threshold_secs=0.0), 0,
                              env=L.parse_group_env(ns2.env_p))["warn"] is None
