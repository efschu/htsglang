"""S23 review (18.10.) probes P1-P3, taken into the branch: the live claim of a sibling io thread of THIS process.

P1: the sibling READS its batch for longer than the JOIN-IDLE bound (1 s; metal: 754 pages in 1472 ms). The reader reports
    progress in the shared arena after every chunk (arena.touch_progress), the joined wait follows it, nothing reaps the
    live claim, the sibling's completion lands (status 1, not 6) and the joiner gets the page. (Reviewer's P1 slept in
    place of a read; the chosen root is reader progress, so the sibling here is a real ``arena_fill_from_disk``.)
P2: a stale registration (an older generation on the slot) does not shield a dead call's leftover (kills mutant M3).
P3: claim and registration are one step: a joiner inside the (artificially widened) window never sees an unregistered
    sibling claim (SELF-JOIN would take it).
P4: the register is keyed by arena: the same slot number in another arena does not shield a leftover.
"""
import logging
import os
import shutil
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.environ import envs
from flliper.srt.mem_cache import hicache_storage as hs
from flliper.srt.mem_cache.hicache_storage import HiCacheFile
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096


@pytest.fixture(autouse=True)
def _clean():
    hs._FILL_LIVE.clear()
    yield
    hs._FILL_LIVE.clear()


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _store(tmp_path, stems=("p1",)):
    root = tmp_path / "store"
    root.mkdir()
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x41" * TOTAL)
    return str(root)


def _on_io_thread(fn, name="hicache-prefetch-io-1"):
    res = {}

    def run():
        res["out"] = fn()

    t = threading.Thread(target=run, name=name)
    t.start()
    t.join(30)
    return res["out"]


def _slow_pageio(monkeypatch, per_page_s):
    """pageio whose read_pages is slow PER PAGE on the sibling's thread (io-0): the batch outlasts the idle bound however
    it is split; only the reader's progress witness (a touch after every sub-call) keeps the live claim."""
    from flliper.srt.mem_cache.storage.file import pageio as _pio_mod

    real_load = _pio_mod.load
    started = threading.Event()

    class _Slow:
        def __init__(self, inner):
            self._inner = inner

        def read_pages(self, paths, *a, **k):
            if threading.current_thread().name == "hicache-prefetch-io-0":
                started.set()
                time.sleep(per_page_s * len(paths))
            return self._inner.read_pages(paths, *a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(_pio_mod, "load", lambda *a, **k: _Slow(real_load(*a, **k)))
    return started


def test_p1_a_sibling_reading_slowly_per_page_longer_than_join_idle_is_not_reaped(tmp_path, monkeypatch, caplog):
    """Reviewer's P1s: slow PER PAGE (0.7 s x 3 pages = 2.1 s > the 1 s idle bound). Kills the 'no touch' mutant."""
    monkeypatch.delenv("FLLIPER_PDFLIP_L3FILL_JOIN_IDLE_MS", raising=False)   # default 1000 ms
    monkeypatch.setenv(hs.L3_READ_MT_ENV, "0")   # STOREREAD-1023: the pool path's per-sub-call witness (the C-thread
    # read's counter witness: test_storeread_1023::test_the_c_thread_read_reports_only_moving_progress)
    monkeypatch.setattr(hs, "L3_TOUCH_PAGES", 1)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path, ("p1", "p2", "p3")))
    started = _slow_pageio(monkeypatch, 0.7)
    sib = {}
    t = threading.Thread(
        target=lambda: sib.update(out=HiCacheFile.arena_fill_from_disk(be, arena, ["p1", "p2", "p3"], TOTAL, prefix=True)),
        name="hicache-prefetch-io-0")
    t.start()
    assert started.wait(10)
    time.sleep(0.1)
    with envs.FLLIPER_PDFLIP_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.INFO, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    t.join(20)
    msgs = [r.getMessage()[:200] for r in caplog.records]
    assert all(x is not None for x in sib["out"]), (sib, msgs)       # red = the sibling's completion refused (6)
    assert out[0] is not None and out[0] == sib["out"][0], (out, sib, msgs)
    assert not any("STALE-CLAIM-REAP" in m or "SELF-JOIN" in m for m in msgs), msgs
    assert not any("LATE-COMPLETE" in m for m in msgs), msgs


class _A:
    def __init__(self):
        self.touches = []

    def touch_progress(self, slots, gens):
        self.touches.append(threading.current_thread().name)
        return len(slots)


class _Pio:
    def __init__(self):
        self.calls = []

    def read_pages(self, paths, sizes, exts, ptrs, *a):
        self.calls.append((threading.current_thread().name, len(paths)))
        return [0] * len(paths)


def test_p5_a_batch_keeps_its_full_parallelism_and_has_no_serial_tail(monkeypatch):
    """The reviewer's P5 in the shape of the chosen design: ONE parallel read of the whole batch (no chunk barriers, no
    serial remainder chunk below L3_PARALLEL_MIN_PAGES), progress reported by the READING WORKER threads."""
    sizes = []
    real = hs.l3_read_pages_parallel

    def rec(pio, paths, total_bytes, ptrs, threads=None):
        sizes.append(len(paths))
        return real(pio, paths, total_bytes, ptrs, threads)

    monkeypatch.setattr(hs, "l3_read_pages_parallel", rec)
    monkeypatch.setattr(hs, "l3_read_threads", lambda: 4)
    be = object.__new__(HiCacheFile)
    n = 754                                           # D TP1 01:36:24: 754 pages in one batch
    todo = [(i, i, 1, "s%d" % i) for i in range(n)]
    arena, pio = _A(), _Pio()
    rc, k = be._l3_read_with_progress(arena, pio, todo, ["p"] * n, 4096, [0] * n)
    assert sizes == [n] and all(r == 0 for r in rc) and k == 4, (sizes, k)
    assert max(sz for _t, sz in pio.calls) <= hs.L3_TOUCH_PAGES          # sub-calls of at most L3_TOUCH_PAGES pages
    assert sum(sz for _t, sz in pio.calls) == n
    # L3TOUCH-1021: the clock moves at most every L3_TOUCH_MIN_S (a fast read: the first and the last touch only); with
    # the bound at 0 every sub-call touches, by the workers, as before
    assert 2 <= len(arena.touches) <= 2 + 4 * 8, len(arena.touches)
    monkeypatch.setattr(hs, "L3_TOUCH_MIN_S", 0.0)
    arena, pio = _A(), _Pio()
    be._l3_read_with_progress(arena, pio, todo, ["p"] * n, 4096, [0] * n)
    worker_touches = [t for t in arena.touches if t.startswith("l3-read")]
    assert len(worker_touches) >= n // hs.L3_TOUCH_PAGES - 4, (len(worker_touches), len(arena.touches))   # by the workers


def test_the_serial_small_batch_reports_progress_per_sub_call_too(monkeypatch):
    monkeypatch.setattr(hs, "l3_read_threads", lambda: 1)
    be = object.__new__(HiCacheFile)
    n = 50
    todo = [(i, i, 1, "s%d" % i) for i in range(n)]
    arena, pio = _A(), _Pio()
    monkeypatch.setattr(hs, "L3_TOUCH_MIN_S", 0.0)   # L3TOUCH-1021: unthrottled = one touch per sub-call as before
    be._l3_read_with_progress(arena, pio, todo, ["p"] * n, 4096, [0] * n)
    assert [sz for _t, sz in pio.calls] == [16, 16, 16, 2]
    assert len(arena.touches) >= 4 + 2          # before, one per sub-call, after


def test_p1b_a_sibling_that_stops_reading_is_reaped_after_the_idle_bound(tmp_path, monkeypatch, caplog):
    """The witness is the reader's chunk, not the registration: a registered sibling that makes no progress is reaped."""
    monkeypatch.setenv("FLLIPER_PDFLIP_L3FILL_JOIN_IDLE_MS", "200")
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path))
    hs._FILL_TL.active = True
    try:
        [(slot, status, gen)] = hs._fill_claim(arena, ["p1"], TOTAL)       # claimed, registered, never reads
    finally:
        hs._FILL_TL.active = False
    assert status == 0
    time.sleep(0.3)
    with envs.FLLIPER_PDFLIP_L3FILL_STALE_CLAIM_S.override(0.25), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != slot
    hs._fill_live_drop_thread()


def test_p2_a_stale_generation_registration_does_not_shield_a_dead_call(tmp_path, caplog):
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path))
    [(old, status, gen)] = arena.claim_slots(["p1"], [TOTAL])     # leftover of a dead call of this pid
    assert status == 0
    hs._FILL_LIVE[(hs._arena_key(arena), int(old))] = (int(gen) - 1, -12345)   # an older generation, another thread
    with envs.FLLIPER_PDFLIP_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != old
    assert any("SELF-JOIN" in r.getMessage() for r in caplog.records), [r.getMessage()[:120] for r in caplog.records]


def test_p3_claim_and_registration_are_one_step(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PDFLIP_L3FILL_JOIN_IDLE_MS", "3000")
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path))
    orig = arena.claim_slots
    in_window = threading.Event()

    def slow_claim(*a, **k):
        r = orig(*a, **k)
        if threading.current_thread().name == "hicache-prefetch-io-0":
            in_window.set()
            time.sleep(0.3)                     # widened: the C claim is done, the registration follows
        return r

    arena.claim_slots = slow_claim
    done = {}

    def sibling():
        hs._FILL_TL.active = True
        try:
            [(slot, status, gen)] = hs._fill_claim(arena, ["p1"], TOTAL)
            time.sleep(0.5)
            arena.slot_view(slot, TOTAL)[:] = b"\x41" * TOTAL
            done["status"] = arena.complete_slots([slot], [gen], [(0, TOTAL)])
        finally:
            hs._FILL_TL.active = False
            hs._fill_live_drop_thread()

    t = threading.Thread(target=sibling, name="hicache-prefetch-io-0")
    t.start()
    in_window.wait(5)
    with envs.FLLIPER_PDFLIP_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    t.join(10)
    msgs = [r.getMessage()[:160] for r in caplog.records]
    assert done.get("status") == [1], (done, out, msgs)      # red = SELF-JOIN inside the window took the live claim


def test_p4_the_register_is_keyed_by_arena_not_by_slot_alone(tmp_path, caplog):
    a1 = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    a2 = ShmArena(str(tmp_path / "mamba.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path))
    [(old, status, gen)] = a1.claim_slots(["p1"], [TOTAL])         # a dead call's leftover in arena 1
    assert status == 0
    hs._FILL_LIVE[(hs._arena_key(a2), int(old))] = (int(gen), -777)   # the SAME slot number and generation, ARENA 2
    with envs.FLLIPER_PDFLIP_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, a1, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != old
    assert any("SELF-JOIN" in r.getMessage() for r in caplog.records)
