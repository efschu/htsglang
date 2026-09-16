"""#1410 (boot xsn159): a blob written by several ranks has ONE home. Once an
extent of it went to the disk tier, every later extent goes to disk too --
otherwise the arena slot stays CLAIMED and the disk partial never completes.
"""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import canonical_page_store as cps
from sglang.srt.mem_cache.hicache_storage import HiCacheFile


def _store(tmp_path):
    st = object.__new__(HiCacheFile)
    st._sharded_path = lambda stem: str(tmp_path / "ab" / f"{stem}.bin")
    (tmp_path / "ab").mkdir()
    return st


def test_cold_arena_never_stats_and_routes_to_arena(tmp_path):
    st = _store(tmp_path)
    arena = types.SimpleNamespace()
    # a marker on disk, but the arena has never refused: no stat, arena home
    open(cps.marker_path(st._sharded_path("s1")), "wb").write(b"x")
    assert st._arena_home_is_disk(arena, "s1") is False


def test_after_a_refusal_the_disk_marker_decides(tmp_path):
    st = _store(tmp_path)
    arena = types.SimpleNamespace()
    st._arena_note_disk_home(arena, "refused")
    assert st._arena_home_is_disk(arena, "refused") is True
    open(cps.marker_path(st._sharded_path("s2")), "wb").write(b"x")
    assert st._arena_home_is_disk(arena, "s2") is True, "a partial on disk"
    open(st._sharded_path("s3"), "wb").write(b"x")
    assert st._arena_home_is_disk(arena, "s3") is True, "a complete blob on disk"
    assert st._arena_home_is_disk(arena, "s4") is False, "nothing on disk: arena"
    assert "s2" in arena._disk_home and "s3" in arena._disk_home
