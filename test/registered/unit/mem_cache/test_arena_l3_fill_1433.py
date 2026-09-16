"""#1433: the L3 -> L2 return path. A page evicted to the disk store is read
straight into a fresh arena slot and completed; the arena is no longer a
write-only sink towards the disk."""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

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
    be._arena_evict_to_disk = lambda arena, want: 0
    return be


def test_a_page_on_disk_is_read_into_a_fresh_slot_and_completed(tmp_path):
    root = tmp_path / "store"; root.mkdir()
    (root / "onDisk.bin").write_bytes(b"\x5a" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 8)
    be = _backend(str(root))
    out = HiCacheFile.arena_fill_from_disk(be, arena, ["onDisk", "absent"], TOTAL)
    assert out[1] is None, "not on disk = None"
    slot = out[0]
    assert slot is not None and arena.find_slots(["onDisk"]) == [(slot, 2)]
    assert bytes(arena.slot_view(slot, TOTAL)[:8]) == b"\x5a" * 8
    # a second ask finds it COMPLETE without touching the disk
    assert HiCacheFile.arena_fill_from_disk(be, arena, ["onDisk"], TOTAL) == [slot]


def test_a_short_file_is_refused_and_the_slot_freed(tmp_path):
    root = tmp_path / "store"; root.mkdir()
    (root / "short.bin").write_bytes(b"\x01" * (TOTAL // 2))
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 4)
    be = _backend(str(root))
    assert HiCacheFile.arena_fill_from_disk(be, arena, ["short"], TOTAL) == [None]
    assert arena.find_slots(["short"]) == [(-1, 0)]
    assert arena.stats()["claimed"] == 0
