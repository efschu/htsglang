# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-L (02.10.2026): WEG2-SLEEP-SUB -- the segments of a sleeping group's
flush and kv release, one line per sleep and rank.

27B N3u 1002_072908 against NF dauer10020814: P's ``#1476 DISPATCH
FlushCacheReqInput`` p50 164 ms on PP0 (NF < 20 ms), PP1 ``CHAIN-RECV blocked
152 ms`` before its own flush, PP1 ``WEG2-SLEEP-CHUNK paused in`` p50 154 ms
(NF 20 ms). Those lines carry totals only; the segment that costs the time is
the open question. Instrument only -- the flush and the release behave as
before. Hermetic, CPU. Red on d1e5da09dc (no flush_sub_timing module, no marks).
"""

from __future__ import annotations

import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

# RELEASE-INTEG 1002: the tree_reset / req_pool_clear marks sat inside the l15-lead's
# L15-PLAIN-RESET timing block (reset + pool clears); his block wins, so those two marks are
# dropped and alloc_clear covers the reset and both clears (the instrument tolerates gaps)
FLUSH_ORDER = ["sweep", "verdict", "store_join", "l15", "anchors", "l15_tail",
               "alloc_clear", "zero_kv", "grammar_metrics", "draft",
               "empty_cache", "scrub"]


@pytest.fixture
def fst(monkeypatch):
    from sglang.srt.weg2 import flush_sub_timing as m

    m._reset_for_tests()
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv("SGLANG_WEG2_L15", raising=False)
    yield m
    m._reset_for_tests()


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def step(self, ms):
        self.t += ms / 1000.0


def test_clock_segments_are_the_gaps_between_marks(fst):
    c = _Clock()
    fc = fst.FlushClock("rpc", now=c)
    c.step(5)
    fc.mark("verdict")
    c.step(150)
    fc.mark("empty_cache")
    assert [(n, round(ms)) for n, ms in fc.segments()] == [("verdict", 5), ("empty_cache", 150)]
    assert round(fc.total_ms()) == 155
    fc.path = "passed"
    assert fc.text() == "passed total=155 verdict=5 empty_cache=150"


def test_one_line_per_sleep_carries_rpc_release_kv_and_the_chain_wait(fst):
    c = _Clock()
    fst.note_chain_blocked(152.0, now=c)
    c.step(1)
    rpc = fst.FlushClock("rpc", now=c)
    c.step(160)
    rpc.mark("tree_reset")
    fst.end_flush(rpc, "passed")
    rel = fst.FlushClock("release", now=c)
    c.step(20)
    rel.mark("empty_cache")
    fst.end_flush(rel, "passed")
    ref = fst.FlushClock("rpc", now=c)
    c.step(45)
    ref.mark("verdict")
    fst.end_flush(ref, "refused")
    kv = fst.FlushClock("kv", now=c)
    c.step(130)
    kv.mark("kv_pause")
    kv.path = "kv"
    line = fst.sleep_line("P", 1, 0, kv)
    assert line.startswith("WEG2-SLEEP-SUB group=P pp=1 tp=0 ")
    assert "rpc_flush[passed total=160 tree_reset=160]" in line
    assert "release_flush[passed total=20 empty_cache=20]" in line
    assert "kv_release[kv total=130 kv_pause=130]" in line
    assert "chain_blocked_ms=152" in line
    assert "refused_polls=1 refused_ms=45" in line
    # the state is per sleep: the next line starts empty
    nxt = fst.sleep_line("P", 1, 0, None)
    assert "rpc_flush[none]" in nxt and "release_flush[none]" in nxt and "chain_blocked_ms=none" in nxt


def test_a_stale_chain_wait_is_not_charged_to_the_flush(fst):
    c = _Clock()
    fst.note_chain_blocked(300.0, now=c)
    c.step((fst.CHAIN_FRESH_S + 1) * 1000)
    rpc = fst.FlushClock("rpc", now=c)
    rpc.mark("verdict")
    fst.end_flush(rpc, "passed")
    assert "chain_blocked_ms=none" in fst.sleep_line("P", 1, 0, None)


def test_cheap_refused_polls_are_not_kept(fst):
    c = _Clock()
    ref = fst.FlushClock("rpc", now=c)
    c.step(3)
    ref.mark("verdict")
    fst.end_flush(ref, "refused")
    assert "refused_polls=0" in fst.sleep_line("P", 0, 0, None)


# ---- through Scheduler.flush_cache --------------------------------------------------

class _Batch:
    reqs = []

    def is_empty(self):
        return True


def _sched(group_idle=True):
    from sglang.srt.managers.scheduler import Scheduler

    calls = []
    rec = lambda name: (lambda *a, **k: calls.append(name))  # noqa: E731
    s = types.SimpleNamespace(
        enable_hierarchical_cache=False, running_batch=_Batch(), waiting_queue=[], chunked_req=None,
        anchor_tails=None, weg2_dormant=False, ps=types.SimpleNamespace(pp_size=3, pp_rank=1),
        tree_cache=types.SimpleNamespace(reset=rec("tree_reset")),
        req_to_token_pool=types.SimpleNamespace(clear=rec("req_pool_clear")),
        token_to_kv_pool_allocator=types.SimpleNamespace(clear=rec("alloc_clear")),
        grammar_manager=types.SimpleNamespace(clear=rec("grammar")),
        metrics_reporter=types.SimpleNamespace(reset_metrics=rec("metrics"), is_stats_logging_rank=True),
        draft_worker=None, _flush_zero_kv_buffers=rec("zero"))
    s.group_idle_verdict = lambda tp_group_verdict=False: (group_idle, "stub verdict")
    s.idle_blockers = lambda exempt_prefetch=(): [] if group_idle else ["waiting_queue"]
    for name in ("_weg2_join_store_writes_before_reset", "_weg2_note_lost_anchors", "_flush_zero_kv_wanted"):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s, calls


def test_flush_cache_marks_every_segment_in_order_and_behaves_as_before(fst, monkeypatch):
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.managers.scheduler import Scheduler

    emptied = []
    monkeypatch.setattr(sched_mod.current_platform, "empty_cache", lambda: emptied.append(1))
    s, calls = _sched()
    assert Scheduler.flush_cache(s) is True
    assert calls == ["tree_reset", "req_pool_clear", "alloc_clear", "grammar", "metrics"]
    assert emptied == [1]
    rpc = fst._STATE["rpc"]
    assert rpc.site == "rpc" and rpc.path == "passed"
    assert [n for n, _ in rpc.segments()] == FLUSH_ORDER
    assert Scheduler.flush_cache(s, zero_kv=False) is True
    rel = fst._STATE["release"]
    assert rel.site == "release" and [n for n, _ in rel.segments()] == FLUSH_ORDER


def test_flush_cache_refused_records_no_flush(fst, monkeypatch):
    from sglang.srt.managers.scheduler import Scheduler

    s, calls = _sched(group_idle=False)
    assert Scheduler.flush_cache(s) is False
    assert calls == [] and fst._STATE["rpc"] is None


def test_the_release_and_the_chain_receiver_feed_the_line():
    from sglang.srt.managers import pp_chain_receiver
    from sglang.srt.managers.scheduler_components import weight_updater

    src = inspect.getsource(weight_updater)
    for needle in ('_kvsub.mark("flush")', '_kvsub.mark("kv_pause")', '_kvsub.mark("sync")',
                   "_weg2_flush_sub.emit_sleep_line("):
        assert needle in src, needle
    assert "note_chain_blocked(_dt)" in inspect.getsource(pp_chain_receiver)


def test_emit_never_raises(fst, caplog):
    with caplog.at_level(logging.INFO):
        fst.emit_sleep_line("P", 0, 0, None)
    assert "WEG2-SLEEP-SUB group=P pp=0 tp=0" in caplog.text


# ---- the 27B-specific posten: the hand-off census inside every tree reset ----------

def _arena(tmp_path, stems, sb=64, extra=8):
    import shutil

    if shutil.which("gcc") is None:
        pytest.skip("needs gcc (arena.c)")
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    a = ShmArena(str(tmp_path / "arena.bin"), sb, len(stems) + extra)
    res = a.claim_slots(stems, [sb] * len(stems))
    a.complete_slots([s for s, _, _ in res], [g for _, _, g in res], [(0, sb)] * len(stems))
    return a


def test_red_census_encodes_each_kept_record_once_and_counts_the_same(tmp_path, monkeypatch):
    """N3u: P kept 90691 token keys at EVERY reset (page_size=1), D up to 416824;
    hp.census re-encoded every stem and built a tuple list each time (CPU:
    125-134 ms for 90691; NF's 3231 keys: 2 ms) -- inside the flush. RED on
    d1e5da09dc: find_slots per census, encode_stems never used."""
    from sglang.srt.mem_cache.storage.file import hicache_arena as ha
    from sglang.srt.weg2 import handoff_pending as hp

    stems = [f"{i:040x}.sfx" for i in range(300)]
    a = _arena(tmp_path, stems[:250])          # 250 complete, 50 absent
    recs = {(hp.PENDING, "r0"): (1, (None, None, stems[:100], stems[:100])),
            (hp.PENDING, "r1"): (1, (None, None, stems[100:300], stems[100:300])),
            (hp.PARK, "r2"): (1, (None, None, stems[:10], stems[:10]))}
    pool = types.SimpleNamespace(arena=a)
    keep = types.SimpleNamespace(rids=["r0", "r1", "r2"], roles=[hp.ROLE_HANDOFF, hp.ROLE_HANDOFF, hp.ROLE_PARK])
    pool.__dict__["_weg2_hp_keep"] = (None, keep)
    pool.__dict__["_weg2_hp_rid_keys"] = recs
    encodes, finds = [], []
    real_enc = ha.ShmArena.encode_stems
    monkeypatch.setattr(ha.ShmArena, "encode_stems", staticmethod(lambda st: encodes.append(len(st)) or real_enc(st)))
    monkeypatch.setattr(ha.ShmArena, "find_slots", lambda self, st: finds.append(len(st)) or [])
    first = hp.census(pool)
    second = hp.census(pool)
    assert first == second
    assert "handoff_kept=250/300 handoff_rids=2 park_kept=10/10 park_rids=1" in first
    assert sorted(encodes) == [10, 100, 200], "one encode per record, not per census"
    assert finds == [], "no per-stem tuple list on the reset path"
    # a replaced record (renewed mark) is encoded again; the old one leaves the cache
    recs[(hp.PENDING, "r0")] = (2, (None, None, stems[:50], stems[:50]))
    assert "handoff_kept=200/250" in hp.census(pool)
    assert sorted(encodes) == [10, 50, 100, 200]
    assert len(pool.__dict__["_weg2_hp_census_enc"]) == 3


def test_census_falls_back_on_an_arena_without_the_encoded_lookup(monkeypatch):
    from sglang.srt.weg2 import handoff_pending as hp

    class _Old:
        def find_slots(self, stems):
            return [(i, 2 if i % 2 == 0 else 0) for i in range(len(stems))]

    pool = types.SimpleNamespace(arena=_Old())
    keep = types.SimpleNamespace(rids=["r0"], roles=[hp.ROLE_HANDOFF])
    pool.__dict__["_weg2_hp_keep"] = (None, keep)
    pool.__dict__["_weg2_hp_rid_keys"] = {(hp.PENDING, "r0"): (1, (None, None, ["a", "b", "c"], None))}
    assert "handoff_kept=2/3" in hp.census(pool)
