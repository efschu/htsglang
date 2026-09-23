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
    assert "queue = list(first or []) + ([] if chain_only else [self.root_node])" in s2


def test_chunk_publish_gate_and_budget():
    assert rp.publish_at_chunk_on({"SGLANG_WEG2_GROUP": "P"})
    assert not rp.publish_at_chunk_on({"SGLANG_WEG2_GROUP": "D"})
    assert not rp.publish_at_chunk_on({"SGLANG_WEG2_GROUP": "P", rp.CHUNK_ENV: "0"})
    assert rp.chunk_budget_s({}) == 0.15
    assert rp.chunk_budget_s({rp.CHUNK_BUDGET_ENV: "0"}) == 0.0
    assert rp.chunk_budget_s({rp.CHUNK_BUDGET_ENV: "x"}) == 0.15


def test_chunk_site_publishes_after_the_cleanup():
    import inspect
    from sglang.srt.mem_cache import unified_radix_cache as u
    src = inspect.getsource(u.UnifiedRadixCache.cache_unfinished_req)
    i = src.index("cleanup_after_caching_req")
    assert "self._weg2_publish_at_chunk(req, radix_key)" in src[i:]
    s2 = inspect.getsource(u.UnifiedRadixCache._weg2_publish_at_chunk)
    assert "chunk_budget_s()" in s2 and "first=first" in s2


def test_d_accepts_leg2_also_during_the_p_to_d_flip():
    assert rp.d_accepts_leg2("D", "serving", False)
    assert not rp.d_accepts_leg2("D", "flipping", True)   # D being put to sleep
    assert rp.d_accepts_leg2("P", "serving", True)
    assert rp.d_accepts_leg2("P", "flipping", True)
    assert not rp.d_accepts_leg2("P", "flipping", False)
    assert not rp.d_accepts_leg2("P", "STOP", True)


def test_chunk_sweep_is_chain_only():
    import inspect
    from sglang.srt.mem_cache import unified_radix_cache as u
    s = inspect.getsource(u.UnifiedRadixCache._weg2_publish_at_chunk)
    assert "chain_only=True" in s
    s2 = inspect.getsource(u.UnifiedRadixCache.publish_unbacked_sweep)
    assert "([] if chain_only else [self.root_node])" in s2


def test_chunk_chain_is_a_parent_walk_from_the_last_node():
    import inspect
    from sglang.srt.mem_cache import unified_radix_cache as u
    s = inspect.getsource(u.UnifiedRadixCache._weg2_publish_at_chunk)
    assert '_weg2_chain_from(getattr(req, "last_node", None))' in s

    class N:
        def __init__(self, parent):
            self.parent = parent
    root = N(None)
    a = N(root); b = N(a); c = N(b)
    fake = type("F", (), {"root_node": root})()
    assert u.UnifiedRadixCache._weg2_chain_from(fake, c) == [a, b, c]
    assert u.UnifiedRadixCache._weg2_chain_from(fake, root) == []
    assert u.UnifiedRadixCache._weg2_chain_from(fake, None) == []
