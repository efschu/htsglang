"""MEMRANK-1015: the per-process WEG2-MEM-RANK line.

Why: NF host MemAvailable 1.6-2.3 GiB, anon +4 GiB in the first 2.5 h of boot
...042125 and no per-process series to say which process grew (deskq/done/
nf-flipstall-1015.md, proposal V2). These tests pin the instrument's contract:
the line appears with its fixed prefix and plausible fields, it sees growth,
it is local (no collective), it never raises, it is default ON, and the four
process kinds call it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
import types
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import mem_rank as mr  # noqa: E402

SRT = os.path.join(os.path.dirname(mr.__file__), "..")


def _kv(line: str) -> dict:
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


@pytest.fixture(autouse=True)
def _clean():
    mr._reset_for_tests()
    yield
    mr._reset_for_tests()


def test_line_has_fixed_prefix_and_plausible_fields():
    p = mr.MemRankProbe("PP0", group="P", rank=0)
    line = p.line()
    assert line.startswith("WEG2-MEM-RANK ")
    kv = _kv(line)
    assert kv["role"] == "PP0" and kv["group"] == "P" and kv["rank"] == "0"
    assert int(kv["pid"]) == os.getpid()
    rss, anon = float(kv["rss_mib"]), float(kv["anon_mib"])
    assert rss > 0 and 0 < anon < rss  # anon is RssAnon, not VmRSS (file-backed pages sit in rss only)
    assert float(kv["shmem_mib"]) >= 0 and float(kv["file_mib"]) >= 0
    assert float(kv["lck_mib"]) >= 0 and float(kv["swap_mib"]) >= 0
    assert float(kv["rss_peak_mib"]) >= rss - 1.0
    assert int(kv["threads"]) >= 1
    assert int(kv["pyblocks"]) > 1000
    assert int(kv["majflt"]) >= 0
    assert float(kv["cost_ms"]) >= 0
    # glibc: in-use bytes > 0 (mallinfo2 exists on this box; -1 only where it does not)
    assert float(kv["malloc_use_mib"]) > 0 or kv["malloc_use_mib"] == "-1"
    # first line: no delta yet -> nan, not a fake zero
    assert kv["d_anon_mib_min"] == "+nan"


def test_second_line_sees_anon_growth_and_blocks():
    p = mr.MemRankProbe("TP1", group="D", rank=1)
    k1 = _kv(p.line())
    time.sleep(0.05)
    ballast = [bytearray(1 << 20) for _ in range(96)]  # 96 MiB, touched by the zero-fill
    for b in ballast:
        b[::4096] = b"\x01" * len(b[::4096])
    k2 = _kv(p.line())
    assert float(k2["anon_mib"]) - float(k1["anon_mib"]) > 60
    assert float(k2["d_anon_mib_min"]) > 0
    assert int(k2["pyblocks"]) >= int(k1["pyblocks"])
    del ballast


def test_gauges_show_growth_of_a_registered_table(monkeypatch):
    fake = types.ModuleType("sglang.srt.weg2.tail_adopt")
    fake._JOBS = {}
    fake._AGREED = {}
    monkeypatch.setitem(sys.modules, "sglang.srt.weg2.tail_adopt", fake)
    p = mr.MemRankProbe("PP1", group="P", rank=1)
    assert _kv(p.line())["n_tail_jobs"] == "0"
    for i in range(7):
        fake._JOBS["rid%d" % i] = i
    kv = _kv(p.line())
    assert kv["n_tail_jobs"] == "7"
    assert kv["n_tail_agreed"] == "0"
    # a module this process never imported is ABSENT, not -1
    assert "n_tailho_writers" not in kv or sys.modules.get("sglang.srt.weg2.tail_handoff") is not None


def test_sumlen_and_ints_modes(monkeypatch):
    arena = types.ModuleType("sglang.srt.mem_cache.pool_host.arena_pool")
    arena._LOST_SETTLED = [4, 9, 2]
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.pool_host.arena_pool", arena)
    ledg = types.ModuleType("sglang.srt.mem_cache.storage.file.hicache_arena")
    ledg._EVICT_GENS = {"a": {1: 1, 2: 2}, "b": {3: 3}}
    ledg._LEDGERS = {}
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.storage.file.hicache_arena", ledg)
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    assert kv["n_a2_settle"] == "4/9/2"  # calls / re-pointed / ORPHANED
    assert kv["n_arena_evict_gens"] == "3"


def test_refledger_gauges_count_slots_bytes_and_held(monkeypatch):
    torch = pytest.importorskip("torch")
    ledg = types.ModuleType("sglang.srt.mem_cache.storage.file.hicache_arena")
    import threading

    class _L:
        def __init__(self, n):
            self.held = torch.zeros(n, dtype=torch.int32)
            self.gen = torch.full((n,), -1, dtype=torch.int64)
            self.lock = threading.Lock()

    led = _L(1000)
    led.held[3] = 2
    led.held[9] = 5
    ledg._LEDGERS = {("p", 1000): led}
    ledg._EVICT_GENS = {}
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.storage.file.hicache_arena", ledg)
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    assert kv["n_reflg_slots"] == "1000"
    assert kv["n_reflg_bytes"] == str(1000 * 4 + 1000 * 8)
    assert kv["n_reflg_held"] == "7"
    # a busy ledger never blocks the instrument: -1, and the line stands
    led.lock.acquire()
    try:
        kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    finally:
        led.lock.release()
    assert kv["n_reflg_held"] == "-1" and kv["n_reflg_slots"] == "1000"


def test_scheduler_object_gauges_and_top_containers():
    lru = types.SimpleNamespace(cache={i: i for i in range(5)})
    lru2 = types.SimpleNamespace(cache={i: i for i in range(3)})
    tree = types.SimpleNamespace(
        lru_lists={1: lru, 2: lru2}, host_lru_lists={1: types.SimpleNamespace(cache={0: 0})},
        _weg2_anchor_only_ids={1, 2}, ongoing_write_through={}, ongoing_load_back={},
        ongoing_backup={}, _retired_prefetch_attempts={"a": 1},
    )
    sched = types.SimpleNamespace(
        tree_cache=tree, waiting_queue=[1, 2, 3], running_batch=types.SimpleNamespace(reqs=[1]),
        big_table={i: i for i in range(500)}, small=[1], dq=deque(range(40)),
    )
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0, kind="scheduler", owner=sched).line())
    assert kv["n_tree_dev"] == "8" and kv["n_tree_host"] == "1"
    assert kv["n_waiting"] == "3" and kv["n_running"] == "1"
    assert kv["n_anchor_only"] == "2" and kv["n_retired_prefetch"] == "1"
    assert kv["top"].startswith("big_table:500,dq:40")
    assert "small" not in kv["top"]


def test_missing_attribute_is_absent_not_minus_one_and_a_failing_gauge_does_not_end_the_line():
    sched = types.SimpleNamespace(waiting_queue=[1], running_batch=types.SimpleNamespace(reqs=[]))
    p = mr.MemRankProbe("TP0", group="D", rank=0, kind="scheduler", owner=sched,
                        extra=[("boom", lambda: 1 // 0), ("fine", lambda: 5)])
    kv = _kv(p.line())
    assert "n_tree_dev" not in kv          # no tree on this object -> absent
    assert kv["n_boom"] == "-1" and kv["n_fine"] == "5"
    assert kv["n_waiting"] == "1"


def test_readers_never_raise_on_unreadable_proc(tmp_path):
    assert mr.read_status(str(tmp_path / "nope")) == {}
    assert mr.read_majflt(str(tmp_path / "nope")) == -1
    f = tmp_path / "stat"
    # comm holds a space and a ')': fields after it are state ppid pgrp session tty tpgid flags minflt cminflt majflt
    f.write_bytes(b"12 (we ird) name) S 1 2 3 4 5 6 70 80 99 11 12")
    assert mr.read_majflt(str(f)) == 99
    # the real file: field 12 of /proc/self/stat
    assert mr.read_majflt() >= 0


def test_probe_makes_no_collective(monkeypatch):
    """The line is local: a torch.distributed call anywhere in it is a failure."""
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    def _boom(*a, **k):
        raise AssertionError("WEG2-MEM-RANK must not enter a collective")

    for name in ("all_reduce", "all_gather", "all_gather_object", "broadcast", "barrier", "reduce",
                 "gather", "scatter", "send", "recv", "all_to_all", "broadcast_object_list"):
        if hasattr(dist, name):
            monkeypatch.setattr(dist, name, _boom)
    p = mr.MemRankProbe("PP2", group="P", rank=2)
    assert p.line().startswith(mr.MARKER)
    # and the module does not even import the collective API
    src = open(mr.__file__).read()
    assert "torch.distributed" not in src.replace("``torch.distributed``", "")
    assert "import zmq" not in src and "requests." not in src


def test_thread_emits_periodically_daemon_idempotent(caplog):
    sink = []
    p1 = mr.start("front", group="F", rank=0, log=sink.append, period=0.05)
    assert p1 is not None
    assert mr.start("front", group="F", rank=0, log=sink.append, period=0.05) is p1  # idempotent
    t0 = time.time()
    while len(sink) < 3 and time.time() - t0 < 5:
        time.sleep(0.02)
    assert len(sink) >= 3
    assert all(s.startswith("WEG2-MEM-RANK role=front ") for s in sink)
    import threading

    th = [t for t in threading.enumerate() if t.name == "weg2-mem-rank"]
    assert th and th[0].daemon
    mr._reset_for_tests()
    t0 = time.time()
    while any(t.is_alive() for t in th) and time.time() - t0 < 5:
        time.sleep(0.02)
    assert not any(t.is_alive() for t in th)


def test_thread_survives_a_failing_sink():
    n = {"c": 0}

    def sink(_):
        n["c"] += 1
        raise RuntimeError("log handler broke")

    mr.start("detokenizer", log=sink, period=0.03)
    t0 = time.time()
    while n["c"] < 3 and time.time() - t0 < 5:
        time.sleep(0.02)
    assert n["c"] >= 3  # the thread kept going after the sink raised


def test_default_on_and_switch_off(monkeypatch):
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_MEM_RANK.get() is True
    assert float(envs.SGLANG_WEG2_MEM_RANK_S.get()) == 60.0
    assert mr.enabled() is True
    monkeypatch.setenv("SGLANG_WEG2_MEM_RANK", "0")
    assert mr.enabled() is False
    assert mr.start("front", log=lambda s: None, period=0.05) is None
    assert mr.current() is None


def test_all_four_process_kinds_start_the_probe():
    """Source anchors: scheduler (every rank), front, detokenizer, tokenizer manager."""
    def _src(rel):
        return open(os.path.join(SRT, rel)).read()

    sched = _src("managers/scheduler.py")
    assert re.search(r'mem_rank\.start\(\s*\("PP" if _pp else "TP"\)', sched)
    assert 'kind="scheduler"' in sched
    assert 'mem_rank.start("front"' in _src("weg2/front.py")
    assert 'mem_rank.start("detokenizer"' in _src("managers/detokenizer_manager.py")
    assert 'mem_rank.start("tokenizer"' in _src("managers/tokenizer_manager.py")


def test_pinned_posts_sum_only_when_the_budget_module_is_loaded(monkeypatch):
    mod = types.ModuleType("sglang.srt.mem_cache.pinned_host_budget")
    mod._registered = {"a": types.SimpleNamespace(nbytes=3 << 20), "b": types.SimpleNamespace(nbytes=5 << 20)}
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.pinned_host_budget", mod)
    assert mr.read_pinned_posts() == 8 << 20
    kv = _kv(mr.MemRankProbe("PP0", group="P", rank=0).line())
    assert kv["pin_posts_mib"] == "8.0"
    monkeypatch.delitem(sys.modules, "sglang.srt.mem_cache.pinned_host_budget")
    assert mr.read_pinned_posts() is None


def test_malloc_split_tells_use_from_free():
    got = mr.read_malloc()
    if got is None:
        pytest.skip("no mallinfo2 on this libc")
    use, free = got
    assert use > 0 and free >= 0
    big = [bytearray(1 << 16) for _ in range(2000)]  # ~125 MiB of small chunks
    use2, _ = mr.read_malloc()
    assert use2 > use + 50 * (1 << 20)
    del big
