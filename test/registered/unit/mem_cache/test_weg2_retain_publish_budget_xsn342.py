"""xsn342: the retain sweep is bounded -- wall budget and a full mamba arena end it."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import retain_publish as rp  # noqa: E402


def test_budget_env_default_and_off():
    assert rp.budget_s({}) == 0.8
    assert rp.budget_s({"SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS": "50"}) == 0.05
    assert rp.budget_s({"SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS": "0"}) == 0.0
    assert rp.budget_s({"SGLANG_WEG2_PUBLISH_AT_RETAIN_BUDGET_MS": "x"}) == 0.8


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


def test_dormant_standstill_is_a_wait_not_a_dead_read():
    assert rp.dormant_standstill_holds(True)
    assert not rp.dormant_standstill_holds(False)
    assert not rp.dormant_standstill_holds(None)


def test_scheduler_holds_the_w88_bound_while_dormant_and_sweep_takes_first():
    import inspect
    from sglang.srt.managers.scheduler import Scheduler
    src = inspect.getsource(Scheduler._weg2_note_prefetch_progress)
    i = src.index("dormant_standstill_holds")
    assert i < src.index("_weg2_prefetch_stall_passes()"), "the dormant wait must sit before the bounds"
    assert 'return "stalled"' in src[i:i + 900]
    from sglang.srt.mem_cache import unified_radix_cache as u
    s2 = inspect.getsource(u.UnifiedRadixCache.publish_unbacked_sweep)
    assert "queue = list(first or []) + [self.root_node]" in s2
