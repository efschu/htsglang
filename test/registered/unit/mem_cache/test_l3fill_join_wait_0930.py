"""L3FILL-JOINED (2) (30.09., NF y4a ep36): a prefix read ended at its first
JOINED stem -- another writer's open claim was a miss at once, and one such
page cut a 1070-page prefix at 146. Now the fill waits (bounded by
SGLANG_WEG2_L3FILL_JOIN_WAIT_MS, default 300) for the claim to complete, but
ONLY on a prefetch io thread (hicache-prefetch-io-<k>); the scheduler thread
never waits."""

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


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _setup(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    for s in ("p0", "p1", "p2"):
        (root / (s + ".bin")).write_bytes(b"\x11" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    [(slot, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # another writer, mid-fill
    assert status == 0
    return _backend(str(root)), arena, slot, gen


def _run_on(thread_name, fn):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn()), name=thread_name)
    th.start()
    th.join(10)
    return box["r"]


def test_the_prefetch_io_thread_waits_for_a_claim_that_completes(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)

    def finish_later():
        time.sleep(0.05)
        assert arena.complete_slots([slot], [gen], [(0, TOTAL)]) == [1]

    threading.Thread(target=finish_later).start()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert out[1] == slot                 # base: None -- the prefix ended at p1
    assert out[0] is not None and out[2] is not None   # base: p2 never read


def test_the_scheduler_thread_never_waits(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        assert hs.fill_join_wait_ms() == 0            # MainThread: no wait
        t0 = time.perf_counter()
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
        took = time.perf_counter() - t0
    assert out[1] is None and out[2] is None and took < 0.25
    assert _run_on("weg2-flip-lane", hs.fill_join_wait_ms) == 0
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        assert _run_on("hicache-prefetch-io-3", hs.fill_join_wait_ms) == 300


def test_a_claim_that_never_completes_is_a_miss_after_the_bound(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(80):
        t0 = time.perf_counter()
        out = _run_on("hicache-prefetch-io-1", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
        took = time.perf_counter() - t0
    assert out[0] is not None and out[1] is None and out[2] is None
    assert 0.07 <= took < 2.0
    assert arena.claim_info([slot])[0][5] == 1       # the other writer's claim untouched
