# SPDX-License-Identifier: Apache-2.0
"""L15-REFILL-DMA + L15-FIX-CAP0-CHECK.

* the cap-0 refill's page-load mode comes from SGLANG_WEG2_L15_REFILL_MODE
  ("dma": one H2D per consecutive-slot run straight out of the registered
  arena, no CPU gather) and reaches the arena loader;
* a cap-0 rank whose refill landed (gen-checked) votes clean without
  re-reading 64 rows from L2 (that compared L2 with itself)."""

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import l15_refill

WU = wu.SchedulerWeightUpdaterManager


class _Host:
    def __init__(self):
        self.modes = []

    def _load_pages_all_layers(self, dev, slots, didx, lanes=None, mode=None):
        self.modes.append(mode)


def test_mode_flag_reaches_the_arena_loader():
    assert l15_refill.refill_mode({}) is None
    assert l15_refill.refill_mode({"SGLANG_WEG2_L15_REFILL_MODE": "DMA"}) == "dma"
    assert l15_refill.refill_mode({"SGLANG_WEG2_L15_REFILL_MODE": "bogus"}) is None
    h = _Host()
    plan = [(("a",), 5, 100, 1), (("a",), 6, 101, 1)]
    assert l15_refill.refill(plan, h, object(), 1, mode="dma") == 2
    assert l15_refill.refill(plan, h, object(), 1) == 2
    assert h.modes == ["dma", None]


def test_cap0_rank_after_a_landed_refill_skips_the_sample(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L15_CAP0_SAMPLE", raising=False)
    fs = SimpleNamespace(_l15_wake_manifest=object(), _l15_refill_done=True,
                         _weg2_rank=lambda: 0, scheduler=None)
    assert WU._l15_wake_sample_check(fs) == (0, 0, 0)
    assert fs._l15_refill_done is False          # one-shot


def test_without_the_refill_mark_the_sample_check_still_runs(monkeypatch):
    called = []
    fs = SimpleNamespace(_l15_wake_manifest=SimpleNamespace(spans=()),
                         _l15_refill_done=False, _weg2_rank=lambda: 1,
                         scheduler=SimpleNamespace(tp_size=3, server_args=None,
                                                   tp_worker=None, tree_cache=None))
    monkeypatch.setattr(wu.logger, "info", lambda *a, **k: called.append(a[0] if a else ""))
    out = WU._l15_wake_sample_check(fs)
    # no pools on this fake -> the real check path runs and folds into a vote
    assert out != (0, 0, 0) or any("L15-CHECK" in str(c) for c in called)
    assert not any("sample skipped" in str(c) for c in called)


def test_the_escape_hatch_keeps_the_old_sample(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L15_CAP0_SAMPLE", "1")
    called = []
    monkeypatch.setattr(wu.logger, "info", lambda *a, **k: called.append(a[0] if a else ""))
    fs = SimpleNamespace(_l15_wake_manifest=SimpleNamespace(spans=()),
                         _l15_refill_done=True, _weg2_rank=lambda: 0,
                         scheduler=SimpleNamespace(tp_size=3, server_args=None,
                                                   tp_worker=None, tree_cache=None))
    WU._l15_wake_sample_check(fs)
    assert not any("sample skipped" in str(c) for c in called)
