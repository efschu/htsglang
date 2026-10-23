"""STOREREAD-1023: the L3 store read of the 27B duo line at disk speed, and what a short read keeps.

Boot 15 (``boot_weg2_dkr27bnvfp4duorebarbar1duo10231145_77bdedf7b4_1023_114625``), P group:
* ``#1433 L3->L2 fill: 1024 of 1024 pages ... ms=4536 pages_per_s=226`` (PP0 11:51:17-37Z), ``PDFLIP-READ-STAGES
  ... l3fill_pages_per_s`` 293-760 on PP0 after the background evictor started (11:50:51Z, 408438 ``L3-EVICT-FILE``
  lines in 22 min), 1480-3626 before. The disk gives 40-48k pages/s cold to the same C reader (storeread probe on
  a copy of the store, 23.10.); with ONE busy Python thread in the process the server's path drops to 381-1188
  pages/s: one Python ``os.stat`` per page (``_existing_path``) and a return into Python every 16 pages per pool
  thread each wait for the GIL. Fixed: paths from the batched C stat, the batch read by C threads in one call.
* ``PF TOLD-FALLBACK rid=pdflip-0-3 told=53248 -> 0 reason=frist after=0.13s acks={}`` (11:52:30Z): PP0's own read
  took 92.29 s, the ``intake + TOTAL_S`` clamp put the Frist AT the publication, the follower had no time to ack and
  P recomputed 64500 tokens. Fixed: the own read is a lifesign like an armed room wait.
* ``#1157 PREFETCH REAPED req=pdflip-1-5 ... completed=0 elapsed=5.04s budget=5.00s`` with ``queue_parts pick=3785``:
  the budget ran out in the io queue. Fixed: the budget runs from the read's pick.
AutoRound boot (``..._1023_140032``, P 14:08:05-15Z) pdflip-11-20: ``claimed=0`` and still ``REAPED hit_pages=0
completed=0 elapsed=9.86s``. Fixed: a probe that answered 0 is a complete read.
RED on 9259a6f470 for every test below that names a switch (switch off = the old path, asserted alongside).
"""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flliper.srt.mem_cache import hicache_storage as hs  # noqa: E402

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

PAGE = 4096


def _pio():
    from flliper.srt.mem_cache.storage.file.pageio import load

    pio = load()
    assert pio is not None
    return pio


def _files(tmp_path, n, size=PAGE, missing=(), short=()):
    root = tmp_path / "pages"
    root.mkdir()
    paths = []
    for i in range(n):
        p = root / ("p%05d.bin" % i)
        if i not in missing:
            p.write_bytes(bytes([i % 251]) * (size // 2 if i in short else size))
        paths.append(str(p))
    return paths


# --- (1) pageio.read_pages_mt == read_pages, statuses and bytes -------------------------------------------------

@needs_gcc
def test_the_c_thread_read_equals_the_serial_read(tmp_path):
    pio = _pio()
    assert pio.has_read_mt
    paths = _files(tmp_path, 700, missing={3, 400}, short={9})
    ext = [((0, PAGE),)] * 700
    b1 = np.zeros((700, PAGE), dtype=np.uint8)
    b2 = np.zeros((700, PAGE), dtype=np.uint8)
    rc1 = pio.read_pages(paths, [PAGE] * 700, ext, [b1[i].ctypes.data for i in range(700)], True)
    seen = []
    rc2 = pio.read_pages_mt(paths, [PAGE] * 700, ext, [b2[i].ctypes.data for i in range(700)], True, 16,
                            progress=lambda: seen.append(1), poll_s=0.001)
    assert rc1 == rc2
    assert rc2[3] == 1 and rc2[400] == 1 and rc2[9] == 2 and rc2.count(0) == 697
    assert np.array_equal(b1, b2)
    assert seen, "the witness reports a read whose page counter moved"


@needs_gcc
def test_a_large_page_is_read_in_pieces_by_the_c_threads(tmp_path, monkeypatch):
    big = hs.L3_SPLIT_MIN_BYTES
    paths = _files(tmp_path, 3, size=big)
    bufs = [np.zeros(big, dtype=np.uint8) for _ in range(3)]
    rc, k = hs.l3_read_pages_parallel(_pio(), paths, big, [b.ctypes.data for b in bufs], threads=8)
    assert rc == [0, 0, 0] and k == 6      # 3 pages x 2 pieces of 4 MiB: the threads are capped at the pieces
    for i, b in enumerate(bufs):
        assert int(b[0]) == i % 251 and int(b[-1]) == i % 251


@needs_gcc
def test_the_parallel_read_is_one_c_call_by_default(tmp_path, monkeypatch):
    pio = _pio()
    paths = _files(tmp_path, 300)
    calls = {"mt": 0, "pages": 0}
    real_mt, real_pages = pio.read_pages_mt, pio.read_pages
    monkeypatch.setattr(pio, "read_pages_mt", lambda *a, **k: (calls.__setitem__("mt", calls["mt"] + 1),
                                                                real_mt(*a, **k))[1])
    monkeypatch.setattr(pio, "read_pages", lambda *a, **k: (calls.__setitem__("pages", calls["pages"] + 1),
                                                             real_pages(*a, **k))[1])
    buf = np.zeros((300, PAGE), dtype=np.uint8)
    rc, k = hs.l3_read_pages_parallel(pio, paths, PAGE, [buf[i].ctypes.data for i in range(300)], threads=16)
    assert rc.count(0) == 300 and k == 16 and calls == {"mt": 1, "pages": 0}
    monkeypatch.setenv(hs.L3_READ_MT_ENV, "0")      # the pool path: 16 threads x sub-calls of 16 pages
    calls.update(mt=0, pages=0)
    rc, k = hs.l3_read_pages_parallel(pio, paths, PAGE, [buf[i].ctypes.data for i in range(300)], threads=16)
    assert rc.count(0) == 300 and calls["mt"] == 0 and calls["pages"] == 16


# --- (2) the fill reads with the paths of its batched C stat, no Python stat per page ---------------------------

class _Arena:
    def __init__(self, n_slots=4096):
        self.buf = np.zeros((n_slots, PAGE), dtype=np.uint8)
        self.state, self.claimed, self.free = {}, {}, list(range(n_slots))

    def claim_slots(self, stems, totals):
        out = []
        for st in stems:
            if st in self.state:
                out.append((self.state[st], 2, 0))
            else:
                s = self.free.pop(0)
                self.claimed[s] = st
                out.append((s, 0, 1))
        return out

    def slot_ptr(self, slot):
        return self.buf[slot].ctypes.data

    def complete_slots(self, slots, gens, extents):
        for s in slots:
            self.state[self.claimed.pop(s)] = s
        return [1] * len(slots)

    def free_slots(self, slots):
        for s in slots:
            self.claimed.pop(s, None)
            self.free.append(s)


def _real_store(tmp_path, n, missing=()):
    """A HiCacheFile with its REAL stat/path code over a sharded directory (no arena dir, no L3 index)."""
    be = object.__new__(hs.HiCacheFile)
    be.file_path = str(tmp_path / "l3")
    os.makedirs(be.file_path)
    be._legacy_flat = False
    be._l3idx_tried = True
    be._l3idx = None
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    stems = ["%064x_m" % (i * 7919 + 1) for i in range(n)]
    for i, st in enumerate(stems):
        if i in missing:
            continue
        p = be._sharded_path(st)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(bytes([i % 251]) * PAGE)
    stat_calls = [0]
    real = hs.HiCacheFile._existing_path

    def counting(stem, _be=be):
        stat_calls[0] += 1
        return real(_be, stem)

    be._existing_path = counting
    return be, stems, stat_calls


@needs_gcc
def test_the_fill_takes_its_paths_from_the_batched_stat(tmp_path, monkeypatch):
    be, stems, stat_calls = _real_store(tmp_path, 600, missing={5})
    arena = _Arena()
    out = hs.HiCacheFile.arena_fill_from_disk(be, arena, stems, PAGE)
    assert out[5] is None and sum(o is not None for o in out) == 599
    for i, s in enumerate(out):
        if s is not None:
            assert int(arena.buf[s][0]) == i % 251
    assert stat_calls[0] == 0, "no Python os.stat per page on the read (the GIL hand-off per page)"
    # switch off: the per-page resolver as before -- the same answer
    monkeypatch.setenv(hs.L3_FILL_NOSTAT_ENV, "0")
    be2, stems2, stat_calls2 = _real_store(tmp_path / "b", 600, missing={5})
    out2 = hs.HiCacheFile.arena_fill_from_disk(be2, _Arena(), stems2, PAGE)
    assert [o is None for o in out2] == [o is None for o in out]
    assert stat_calls2[0] == 599


def _hog(stop):
    x = 0
    while not stop.is_set():
        for i in range(2000):
            x += i


@needs_gcc
def test_beside_a_busy_python_thread_the_fill_keeps_its_pace(tmp_path, monkeypatch):
    """The boot's shape, hermetic: one CPU-bound Python thread in the process. The old path (per-page stat + pool
    sub-calls) pays a GIL wait per page; the new one does not. Measured here >= 10x; asserted 3x (shared CI)."""
    n = 3000

    def run(tag):
        be, stems, _ = _real_store(tmp_path / tag, n)
        stop = threading.Event()
        t = threading.Thread(target=_hog, args=(stop,), daemon=True)
        t.start()
        try:
            t0 = time.perf_counter()
            out = hs.HiCacheFile.arena_fill_from_disk(be, _Arena(n + 8), stems, PAGE)
            return time.perf_counter() - t0, sum(o is not None for o in out)
        finally:
            stop.set()
            t.join()

    new_s, new_ok = run("new")
    monkeypatch.setenv(hs.L3_FILL_NOSTAT_ENV, "0")
    monkeypatch.setenv(hs.L3_READ_MT_ENV, "0")
    old_s, old_ok = run("old")
    assert new_ok == old_ok == n
    assert old_s > 3 * new_s, (old_s, new_s)


# --- (3) the evictor's per-file lines are capped per window, not only per run -----------------------------------

def test_one_file_runs_cannot_print_a_line_per_file(monkeypatch, caplog):
    from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    ev = object.__new__(LRUFileEvictor)
    monkeypatch.setenv("FLLIPER_HICACHE_L3_EVICT_FILE_LOG", "64")
    with caplog.at_level(logging.INFO):
        for k in range(2000):           # the bg evictor's 1-file runs (a waiter cut every batch)
            ev._evict_why, ev._evict_run_id, ev._evict_file_run_n = "cap_bg", k, 0
            ev._note_evicted_file("stem%d" % k, 32768, time.time())
    lines = [m for m in caplog.messages if m.startswith("L3-EVICT-FILE stem=")]
    assert 0 < len(lines) <= 64
    # the next window names what was suppressed
    ev._evict_file_win_t0 -= 11.0
    with caplog.at_level(logging.INFO):
        ev._evict_why, ev._evict_run_id, ev._evict_file_run_n = "cap_bg", 99999, 0
        ev._note_evicted_file("stemX", 32768, time.time())
    assert any(m.startswith("L3-EVICT-FILE WINDOW suppressed=%d" % (2000 - 64)) for m in caplog.messages)
    # window 0 = per run only (before)
    caplog.clear()
    monkeypatch.setenv("FLLIPER_HICACHE_L3_EVICT_FILE_LOG_WINDOW_S", "0")
    ev2 = object.__new__(LRUFileEvictor)
    with caplog.at_level(logging.INFO):
        for k in range(200):
            ev2._evict_why, ev2._evict_run_id, ev2._evict_file_run_n = "cap_bg", k, 0
            ev2._note_evicted_file("s%d" % k, 32768, time.time())
    assert len([m for m in caplog.messages if m.startswith("L3-EVICT-FILE stem=")]) == 200


# --- (4) the told's Frist after a long own read ------------------------------------------------------------------

def test_a_long_own_read_leaves_the_followers_their_frist(monkeypatch):
    from flliper.srt.managers import pdflip_told_fallback as fb

    pub, own, win = 1000.0, 92.29, 10.0           # pdflip-0-3, P 11:52:30Z
    assert fb.deadline(pub, own, win) == pub + fb.frist_s(win)
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=2))
    fb.pp0_open(sched, "pdflip-0-3", 53248, pub, own, win)
    assert fb.pp0_decide(sched, "pdflip-0-3", pub + 0.13) is None, "the follower's ack is still on its way"
    fb._pp0_open_map(sched)["pdflip-0-3"].acks[1] = 53248
    assert fb.pp0_decide(sched, "pdflip-0-3", pub + 3.0) == (53248, fb.REASON_ACKS)
    monkeypatch.setenv(fb.ENV_OWN_READ_LIFESIGN, "0")
    assert fb.deadline(pub, own, win) == pub      # before: the Frist at the publication


# --- (5) the prefetch budget: from the pick, and a probe that answered 0 is done --------------------------------

def _op(start, read_start=None, hit=None, hv=()):
    op = types.SimpleNamespace(start_time=start, hash_value=list(hv), probed_hit_tokens=hit)
    if read_start is not None:
        op.read_start_time = read_start
    return op


def test_the_budget_runs_from_the_pick_not_from_the_queue(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    now = time.monotonic()
    stub = types.SimpleNamespace(_prefetch_timeout_budget_s=lambda op: 5.0)
    check = urc.UnifiedRadixCache._prefetch_timeout_check_linear_func.__get__(stub)
    # pdflip-1-5: issued 5.04 s ago, picked 3.785 s after the issue
    assert not check(_op(now - 5.04, read_start=now - 5.04 + 3.785))
    assert check(_op(now - 9.0, read_start=now - 5.2))           # its own budget spent: ends
    assert check(_op(now - 5.04))                                 # never picked: the issue clock still ends it
    assert check(_op(now - 10.5, read_start=now - 0.1))           # picked late: the clock starts at issue + budget
    monkeypatch.setenv("FLLIPER_HICACHE_PREFETCH_BUDGET_FROM_READ", "0")
    assert check(_op(now - 5.04, read_start=now - 5.04 + 3.785))  # before


def test_a_probe_that_answered_zero_is_a_complete_read(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    assert urc._prefetch_nothing_to_read(_op(0.0, hit=0))
    assert not urc._prefetch_nothing_to_read(_op(0.0, hit=None))          # not probed yet: wait
    assert not urc._prefetch_nothing_to_read(_op(0.0, hit=8, hv=["a"]))
    monkeypatch.setenv("FLLIPER_HICACHE_PREFETCH_HIT0_DONE", "0")
    assert not urc._prefetch_nothing_to_read(_op(0.0, hit=0))


# --- (6) END-ANCHOR on D: a logprob request is priced with D's prefix cap ---------------------------------------

def test_a_logprob_request_is_priced_at_what_d_computes(monkeypatch):
    from flliper.srt.pdflip import front as F

    body = {"input_ids": list(range(3089)), "return_logprob": True, "logprob_start_len": 0}
    # pdflip-1-5: store 3088, D-PROBE uncached_d=1 -> priced kurz; D computed all 3089
    assert F.apply_logprob_prefix_cap(body, 3089, 1, 3088) == (3089, 0, 0)
    assert F.apply_logprob_prefix_cap(dict(body, logprob_start_len=3000), 3089, 1, 3088) == (89, 3000, 3000)
    assert F.apply_logprob_prefix_cap(dict(body, logprob_start_len=-1), 3089, 1, 3088) == (1, 3088, None)
    assert F.apply_logprob_prefix_cap({"input_ids": [1] * 3089}, 3089, 1, 3088) == (1, 3088, None)
    assert F.apply_logprob_prefix_cap({"prompt": "x", "echo": True, "logprobs": 1}, 3089, 1, 3088) == (3089, 0, 0)
    assert F.apply_logprob_prefix_cap(dict(body, logprob_start_len=3088), 3089, 5, 3084) == (5, 3084, None)
    monkeypatch.setenv("FLLIPER_PDFLIP_FRONT_LOGPROB_CAP", "0")
    assert F.apply_logprob_prefix_cap(body, 3089, 1, 3088) == (1, 3088, None)    # before


# --- (7) the C-thread read's witness: reported while pages move, silent while they do not ------------------------

def _fake_pageio(step_s, pages_moving):
    from flliper.srt.mem_cache.storage.file.pageio import PageIO

    class _Lib:
        def hicache_read_pages_mt(self, n, c_paths, c_total, n_ext, c_off, c_len, c_out, touch, status, k, done):
            for i in range(n):
                time.sleep(step_s)
                if pages_moving:
                    done._obj.value += 1
                status[i] = 0
            return n

    pio = object.__new__(PageIO)
    pio._lib = _Lib()
    pio._has_read_mt = True
    return pio


def test_the_c_thread_read_reports_only_moving_progress():
    seen = []
    st = _fake_pageio(0.03, True).read_pages_mt(["p"] * 20, [PAGE] * 20, [((0, PAGE),)] * 20, [0] * 20, True, 4,
                                               progress=lambda: seen.append(1), poll_s=0.05)
    assert st == [0] * 20 and len(seen) >= 4, len(seen)          # a reader that moves keeps its claims alive
    seen.clear()
    _fake_pageio(0.03, False).read_pages_mt(["p"] * 20, [PAGE] * 20, [((0, PAGE),)] * 20, [0] * 20, True, 4,
                                           progress=lambda: seen.append(1), poll_s=0.05)
    assert len(seen) <= 1, len(seen)                              # one at the start, then silent: reaped as before
