"""xsn342: the retain sweep is bounded -- wall budget and a full mamba arena end it."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import retain_publish as rp  # noqa: E402


def test_budget_env_default_and_off():
    assert rp.budget_s({}) == 0.4
    assert rp.budget_s({"SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS": "50"}) == 0.05
    assert rp.budget_s({"SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS": "0"}) == 0.0
    assert rp.budget_s({"SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS": "x"}) == 0.4


def test_sweep_clock_expires_only_with_a_budget():
    t = [0.0]
    c = rp.SweepClock(0.4, now=lambda: t[0])
    assert not c.expired()
    t[0] = 0.39
    assert not c.expired()
    t[0] = 0.4
    assert c.expired() and c.stop_reason == "budget"
    assert c.elapsed_ms() == 400.0
    c0 = rp.SweepClock(0.0, now=lambda: t[0])
    t[0] = 1e9
    assert not c0.expired() and c0.stop_reason is None


def test_mamba_full_stops_the_sweep_other_refusals_do_not():
    assert rp.mamba_full_stops_sweep("mamba_claim")
    assert not rp.mamba_full_stops_sweep("arena_claim")
    assert not rp.mamba_full_stops_sweep(None)
