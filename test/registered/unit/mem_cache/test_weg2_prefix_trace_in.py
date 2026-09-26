"""Prefix trace (IN 26.09.): SGLANG_WEG2_PREFIX_TRACE gives every prefix miss a
token receipt -- and changes nothing while it is off.

Covers the switch module (weg2/prefix_trace.py), the REAL #1420 walk in
``UnifiedRadixCache._match_prefix_helper`` on CPU tensors (same fixture as
test_producer_phase_census_wiring_1061.py), the #1400/#1416 rid width and
sampling in weg2_store_told, the #1469 cap bypass, the #1427 ARENA-DROP keys
and the two once-per-process armed lines (MZ, #49).

Hermetic: CPU only, no boot, no GPU.
"""

import logging
from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.weg2 import prefix_trace as pt


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_PREFIX_TRACE", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PREFIX_TRACE_MIN_TOKENS", raising=False)
    pt._reset_for_tests()
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    UnifiedRadixCache._1420_n = 0  # the legacy cap counts per process
    yield
    pt._reset_for_tests()


def _arm(monkeypatch, min_tokens=None):
    monkeypatch.setenv("SGLANG_WEG2_PREFIX_TRACE", "1")
    if min_tokens is not None:
        monkeypatch.setenv("SGLANG_WEG2_PREFIX_TRACE_MIN_TOKENS", str(min_tokens))
    pt._reset_for_tests()


def _msgs(caplog, marker):
    return [r.getMessage() for r in caplog.records if marker in r.getMessage()]


# -- the switch -------------------------------------------------------------


def test_off_by_default_keeps_the_legacy_sampling_and_rid_width(caplog):
    with caplog.at_level(logging.INFO):
        assert pt.on() is False
    assert pt.min_tokens() == 1024
    assert pt.sampled(8, 8, 256) and not pt.sampled(9, 8, 256) and pt.sampled(512, 8, 256)
    assert pt.rid_text("weg2-14-20") == "weg2-14-"
    assert pt.walk_due("weg2-1-1", 0, 10**6) is False
    assert pt.once("x", 1) is False
    armed = _msgs(caplog, "#PT PREFIX-TRACE")
    assert len(armed) == 1 and "armed=0" in armed[0], armed


def test_armed_lifts_sampling_full_rid_and_announces_once(monkeypatch, caplog):
    _arm(monkeypatch)
    with caplog.at_level(logging.INFO):
        assert pt.on() and pt.on()
        assert pt.sampled(9, 8, 256)
        assert pt.rid_text("weg2-14-20") == "weg2-14-20"
    armed = _msgs(caplog, "#PT PREFIX-TRACE")
    assert len(armed) == 1 and "armed=1 min_tokens=1024" in armed[0], armed


def test_walk_due_one_line_per_rid_and_depth_above_the_minimum(monkeypatch):
    _arm(monkeypatch)
    assert pt.walk_due("weg2-1-1", 100, 1023) is False  # rest below the minimum
    assert pt.walk_due("weg2-1-1", 100, 1024) is True
    assert pt.walk_due("weg2-1-1", 100, 5000) is False  # same (rid, depth)
    assert pt.walk_due("weg2-1-1", 4096, 5000) is True  # a new stop depth
    assert pt.walk_due(None, 0, 5000) is False
    assert pt.once("1442", "r", 1, 2) is True and pt.once("1442", "r", 1, 2) is False


def test_dedup_tables_are_bounded(monkeypatch):
    _arm(monkeypatch, min_tokens=0)
    for i in range(pt.WALK_SEEN_MAX + 50):
        assert pt.walk_due(f"r{i}", 0, 1)
    assert len(pt._walk_seen) <= pt.WALK_SEEN_MAX


# -- the REAL #1420 walk ------------------------------------------------------


def _cache(**over):
    from dataclasses import replace

    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    return build_fixture(replace(CacheConfig(), **over))


def _insert(cache, allocator, tokens):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    value = allocator.alloc(len(tokens))
    assert value is not None
    cache.insert(InsertParams(key=RadixKey(tokens), value=value))


def _match(cache, tokens, rid=None):
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    req = SimpleNamespace(rid=rid, session=None) if rid is not None else None
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens), req=req))


STORED = array("q", range(1, 101))  # 100 tokens
# diverges at position 40 (split of the stored node), 100 tokens long
NEW = array("q", list(range(1, 41)) + [9000 + i for i in range(60)])


def test_traced_split_stop_names_rid_true_rest_and_tokens(monkeypatch, caplog):
    _arm(monkeypatch, min_tokens=32)
    cache, allocator, _ = _cache()
    _insert(cache, allocator, STORED)
    with caplog.at_level(logging.INFO):
        _match(cache, NEW, rid="weg2-7-3")
        _match(cache, NEW, rid="weg2-7-3")  # re-matched in a later pass: deduped
    lines = _msgs(caplog, "#1420 WALK-STOP")
    assert len(lines) == 1, lines
    line = lines[0]
    assert "rid=weg2-7-3 trace=1" in line, line
    # the TRUE rest after the split: 40 + 60 == len(NEW)
    assert " depth=40 remaining=60 " in line, line
    assert "want=9000 " in line and "have=['41']" in line, line
    assert "want_run=[9000, 9001, 9002" in line, line
    assert "have_run=[[41, 42, 43" in line, line
    assert "stop_node=" in line and "stop_hash=-" in line, line


def test_traced_root_stop_is_logged_rest_below_minimum_is_not(monkeypatch, caplog):
    _arm(monkeypatch, min_tokens=32)
    cache, allocator, _ = _cache()
    _insert(cache, allocator, STORED)
    cold = array("q", [7000 + i for i in range(50)])
    short_rest = array("q", list(range(1, 81)) + [8000 + i for i in range(10)])
    with caplog.at_level(logging.INFO):
        _match(cache, cold, rid="weg2-9-1")
        _match(cache, short_rest, rid="weg2-9-2")  # rest 10 < 32
    lines = _msgs(caplog, "#1420 WALK-STOP")
    assert len(lines) == 1, lines
    assert "depth=0 remaining=50" in lines[0] and "rid=weg2-9-1" in lines[0], lines


def test_off_walk_is_the_legacy_line_without_rid(caplog):
    cache, allocator, _ = _cache()
    _insert(cache, allocator, STORED)
    with caplog.at_level(logging.INFO):
        _match(cache, NEW, rid="weg2-7-3")
        _match(cache, array("q", [7000 + i for i in range(50)]), rid="weg2-9-1")
    lines = _msgs(caplog, "#1420 WALK-STOP")
    # the split stop only (a root stop is never logged untraced)
    assert len(lines) == 1, lines
    assert "rid=" not in lines[0] and "trace=" not in lines[0], lines
    assert " depth=40 remaining=60 want=9000 " in lines[0], lines


def test_bigram_walk_rest_adds_up_to_pt_minus_one(monkeypatch, caplog):
    """The classifier's own-walk rule: depth + remaining == pt - 1 (bigram)."""
    _arm(monkeypatch, min_tokens=32)
    cache, allocator, _ = _cache(is_eagle=True)
    _insert(cache, allocator, STORED)
    with caplog.at_level(logging.INFO):
        _match(cache, NEW, rid="weg2-7-4")
    lines = _msgs(caplog, "#1420 WALK-STOP")
    assert len(lines) == 1, lines
    import re

    m = re.search(r"depth=(\d+) remaining=(\d+) want=(\([^)]*\))", lines[0])
    assert m, lines[0]
    assert int(m.group(1)) + int(m.group(2)) == len(NEW) - 1, lines[0]
    assert m.group(3) == "(40, 9000)", lines[0]


# -- #1400 / #1416 rid width and sampling -------------------------------------


def test_store_told_rid_and_sampling_follow_the_switch(monkeypatch):
    from sglang.srt.managers import weg2_store_told as st

    req = SimpleNamespace(rid="weg2-14-20")
    assert st.rid8(req) == "weg2-14-" and st._rt("weg2-14-20") == "weg2-14-"
    assert st._log_due(8) and not st._log_due(9)
    _arm(monkeypatch)
    assert st.rid8(req) == "weg2-14-20" and st._rt("weg2-14-20") == "weg2-14-20"
    assert st._log_due(9) and st._log_due(10**6)


def test_anchor_clamp_line_carries_the_full_rid(monkeypatch, caplog):
    from sglang.srt.managers import weg2_store_told as st

    _arm(monkeypatch)
    cc = SimpleNamespace(get_hash_str=lambda *a, **k: "h", page_size=1)
    sched = SimpleNamespace(cache_controller=cc, tree_cache=None)
    req = SimpleNamespace(rid="weg2-14-20", origin_input_ids=list(range(100)))
    monkeypatch.setattr(st, "_tree_key_probe_armed", lambda: False)
    monkeypatch.setattr(st, "_anchored_pages_full_span", lambda *a, **k: 0)
    with caplog.at_level(logging.INFO):
        assert st._anchor_clamp(sched, req, 48) == 0
    lines = _msgs(caplog, "#1416 STORE-TOLD ANCHOR-CLAMP")
    assert lines and "rid=weg2-14-20 completed=48 anchored=0" in lines[0], lines


# -- #1469 EVICT / #1427 ARENA-DROP -------------------------------------------


def test_1469_uncapped_note_bypasses_and_does_not_spend_the_cap(monkeypatch, caplog):
    from sglang.srt.mem_cache.unified_cache_components import mamba_component as mc

    monkeypatch.setattr(mc, "_1469_N", mc._1469_CAP)
    with caplog.at_level(logging.INFO):
        mc._1469_note("RETAIN", rid="x")  # capped
        mc._1469_note("EVICT", _uncapped=True, node=5, parent=3, trace=1)
    lines = _msgs(caplog, "#1469 ")
    assert lines == ["#1469 EVICT node=5 parent=3 trace=1"], lines
    assert mc._1469_N == mc._1469_CAP + 1


def test_evict_site_names_parent_and_hashes_when_traced():
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._evict_component_and_detach_lru)
    i = src.index("_prefix_trace.on()")
    for field in ("_uncapped=True", "parent=", "hash_first=", "hash_last=", "trace=1"):
        assert field in src[i:], field


class _Arena:
    slot_bytes = 32768

    def __init__(self):
        self.freed = None

    def evict_candidates(self, need):
        return [(11, 0xABC, 0xDEF, 32768), (12, (1 << 64) - 2, 0x1, 32768)][:need]

    def free_slots(self, slots):
        self.freed = list(slots)


def test_arena_drop_names_dropped_keys_and_claim_when_traced(monkeypatch, caplog):
    from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

    _arm(monkeypatch)
    arena = _Arena()
    with caplog.at_level(logging.INFO):
        assert ArenaMHAHostPool._evict_for_claim(None, arena, 2, claim_stem="f9ec16cd_k") == 2
    lines = _msgs(caplog, "#1427 ARENA-DROP n=")
    assert len(lines) == 1, lines
    assert "trace=1 claim=f9ec16cd_k dropped=0000000000000abc,fffffffffffffffe" in lines[0], lines
    assert arena.freed == [11, 12]


def test_arena_drop_untraced_line_unchanged(caplog):
    from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

    monkey_n = getattr(ArenaMHAHostPool, "_1427_drop_n", 0)
    ArenaMHAHostPool._1427_drop_n = 0
    try:
        with caplog.at_level(logging.INFO):
            ArenaMHAHostPool._evict_for_claim(None, _Arena(), 1)
    finally:
        ArenaMHAHostPool._1427_drop_n = monkey_n
    lines = _msgs(caplog, "#1427 ARENA-DROP n=")
    assert len(lines) == 1 and "dropped=" not in lines[0] and "trace=" not in lines[0], lines


# -- the armed lines (MZ, #49) ------------------------------------------------


def test_mz_armed_line_once_per_process(monkeypatch, caplog):
    from sglang.srt.entrypoints.anthropic import serving as s

    monkeypatch.setattr(s, "_MZ_ANNOUNCED", False)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    with caplog.at_level(logging.INFO):
        s._announce_inline_system_in_place(True, True)
        s._announce_inline_system_in_place(True, True)
    lines = _msgs(caplog, "#MZ INLINE-SYSTEM-IN-PLACE")
    assert len(lines) == 1, lines
    assert "armed=1 merge=1 group=P pid=" in lines[0], lines
    assert "SGLANG_ANTHROPIC_INLINE_SYSTEM_IN_PLACE" in lines[0]


def test_front_49_armed_line_names_value_and_role(monkeypatch):
    from sglang.srt.weg2 import front

    monkeypatch.setattr(front, "front_span_inflight", lambda: True)
    line = front.front_span_inflight_line()
    assert line.startswith("WEG2-FRONT #49 SPAN-INFLIGHT armed=1 agent_span="), line
    assert " role=front pid=" in line, line
    assert "SGLANG_WEG2_FRONT_SPAN_INFLIGHT" in line
