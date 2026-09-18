"""Posten 2 (18.09.): ARENA-GET find/ref on numpy arrays, against a real
tiny arena in tmpfs (the C helper needs no GPU)."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402


@pytest.fixture
def arena():
    path = os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(),
                        f"weg2-test-arena-{os.getpid()}.bin")
    a = ShmArena(path, 4096, 1024)
    try:
        yield a
    finally:
        a.close()
        os.unlink(path)


def test_numpy_find_and_ref_agree_with_the_list_forms(arena):
    stems = [f"{i:040x}_kv" for i in range(300)]
    arena.claim_slots(stems[:200], [1024] * 200)          # claimed, not complete
    fs, st = arena.find_slots_np(stems)
    assert fs.dtype == np.int64 and st.dtype == np.int8 and fs.shape == (300,)
    ref = arena.find_slots(stems)
    assert fs.tolist() == [s for s, _ in ref] and st.tolist() == [t for _, t in ref]
    assert (fs[:200] >= 0).all() and (fs[200:] < 0).all()
    # refs over the numpy view equal the list form (CLAIMED slots take refs, #1427)
    got = arena.ref_slots_np(fs[:200], +1)
    assert got == arena.ref_slots(fs[:200].tolist(), -1) == 200
    e = arena.find_slots_np([])
    assert e[0].shape == (0,) and arena.ref_slots_np(e[0], +1) == 0


def test_arena_get_uses_the_numpy_path():
    from sglang.srt.managers import cache_controller as cc
    src = open(cc.__file__).read()
    i = src.index("_fs, _st = pool.arena.find_slots_np(stems)")
    blk = src[i:i + 1200]
    assert "ref_slots_np(_fs[:lead], +1)" in blk and "flatnonzero((_fs < 0) | (_st != 2))" in blk
    assert "for i in range(lead, int(_fs.shape[0]))" in blk
