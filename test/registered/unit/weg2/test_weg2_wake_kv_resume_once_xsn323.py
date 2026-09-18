"""xsn317/318/321: the early kv-only wake resumed kv_cache, the weights call's
late site resumed it AGAIN -> torch_memory_saver exit(1) ("Cannot resume
allocation that is not paused"), the P rank died without a traceback. The
RESUME half runs once per epoch; the CLEAR half follows at the late site.
Plus xsn321's secondary: the poll fault handler raised NameError (`t`)."""
from __future__ import annotations

import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

WU = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python", "sglang", "srt",
                  "managers", "scheduler_components", "weight_updater.py")


def _src():
    with open(os.path.normpath(WU)) as fh:
        return fh.read()


def test_resume_half_records_its_epoch_and_runs_once():
    s = _src()
    assert "_weg2_kv_resumed_epoch: object = None" in s
    body = s[s.index("def _weg2_kv_resume_part():"):s.index("def _weg2_kv_clear_part():")]
    guard = body.index("self._weg2_kv_resumed_epoch == _kv_epoch")
    resume = body.index("self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)")
    assert guard < resume, "the once-per-epoch guard sits BEFORE the tms resume"
    assert "return" in body[guard:resume]
    assert "self._weg2_kv_resumed_epoch = _kv_epoch" in body[resume:]


def test_late_site_takes_the_clear_half_after_an_early_resume():
    s = _src()
    i = s.index("# the late site: legs first")
    late = s[i:s.index("report: Dict[str, Any] = {}", i)]
    assert re.search(r"if _weg2_kv_resumed_early or \(_kv_epoch is not None and "
                     r"self\._weg2_kv_resumed_epoch == _kv_epoch\):\s*\n\s*_weg2_kv_clear_part\(\)", late)
    assert "self._weg2_kv_epoch_done == _kv_epoch" not in late.split("_weg2_kv_clear_part()")[0]


def test_field_is_declared_on_the_slots_dataclass():
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as M,
    )
    m = M.__new__(M)
    m._weg2_kv_resumed_epoch = 1.5
    assert m._weg2_kv_resumed_epoch == 1.5


def test_poll_fault_disarms_the_transport_and_survives(monkeypatch):
    from sglang.srt.distributed.device_communicators import barlink_abort_gate as g

    monkeypatch.setenv("SGLANG_WEG2_POLL_PYSPY", "0")
    g.reset_for_test()
    calls = []

    class T:
        _ctl_dev = None
        _abort_poll_dst = None

        def poll_status_word(self):
            raise RuntimeError("unknown parameter type")

        def _abort_poll_disarm(self, why):
            calls.append(why)

    t = T()
    g.register(t)
    try:
        monkeypatch.setattr(g, "abort_check_enabled", lambda: True)
        monkeypatch.setattr(g, "polling_paused", lambda: False)
        assert g.poll_status_words() == 0
    finally:
        g.unregister(t)
    assert len(calls) == 1 and "raised" in calls[0]
