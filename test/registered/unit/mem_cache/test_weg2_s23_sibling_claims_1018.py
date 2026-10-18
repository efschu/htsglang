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

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

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


def test_p1_a_sibling_reading_longer_than_join_idle_is_not_reaped(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", raising=False)   # default 1000 ms
    monkeypatch.setattr(hs, "L3_READ_CHUNK_PAGES", 1)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path, ("p1", "p2", "p3")))
    real = hs.l3_read_pages_parallel
    started = threading.Event()

    def slow_read(pio, paths, total_bytes, ptrs, threads=None):
        if threading.current_thread().name == "hicache-prefetch-io-0":
            started.set()
            time.sleep(0.7)                      # one chunk of a cold-disk batch: 3 chunks = 2.1 s > the 1 s idle bound
        return real(pio, paths, total_bytes, ptrs, threads)

    monkeypatch.setattr(hs, "l3_read_pages_parallel", slow_read)
    sib = {}
    t = threading.Thread(
        target=lambda: sib.update(out=HiCacheFile.arena_fill_from_disk(be, arena, ["p1", "p2", "p3"], TOTAL, prefix=True)),
        name="hicache-prefetch-io-0")
    t.start()
    assert started.wait(10)
    time.sleep(0.1)
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.INFO, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    t.join(20)
    msgs = [r.getMessage()[:200] for r in caplog.records]
    assert all(x is not None for x in sib["out"]), (sib, msgs)       # red = the sibling's completion refused (6)
    assert out[0] is not None and out[0] == sib["out"][0], (out, sib, msgs)
    assert not any("STALE-CLAIM-REAP" in m or "SELF-JOIN" in m for m in msgs), msgs
    assert not any("LATE-COMPLETE" in m for m in msgs), msgs


def test_p1b_a_sibling_that_stops_reading_is_reaped_after_the_idle_bound(tmp_path, monkeypatch, caplog):
    """The witness is the reader's chunk, not the registration: a registered sibling that makes no progress is reaped."""
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "200")
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path))
    hs._FILL_TL.active = True
    try:
        [(slot, status, gen)] = hs._fill_claim(arena, ["p1"], TOTAL)       # claimed, registered, never reads
    finally:
        hs._FILL_TL.active = False
    assert status == 0
    time.sleep(0.3)
    with envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.25), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != slot
    hs._fill_live_drop_thread()


def test_p2_a_stale_generation_registration_does_not_shield_a_dead_call(tmp_path, caplog):
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(_store(tmp_path))
    [(old, status, gen)] = arena.claim_slots(["p1"], [TOTAL])     # leftover of a dead call of this pid
    assert status == 0
    hs._FILL_LIVE[(hs._arena_key(arena), int(old))] = (int(gen) - 1, -12345)   # an older generation, another thread
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != old
    assert any("SELF-JOIN" in r.getMessage() for r in caplog.records), [r.getMessage()[:120] for r in caplog.records]


def test_p3_claim_and_registration_are_one_step(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "3000")
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
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
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
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, a1, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != old
    assert any("SELF-JOIN" in r.getMessage() for r in caplog.records)
