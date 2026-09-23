"""#1424 Stufe 3: a reader addresses a COMPLETE page IN PLACE -- slot lookup by
key, a reader reference that pins the slot against eviction, and the data
offset that makes `base + data_off + slot * slot_bytes` the page's address."""

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
SLOT = 256


def _arena(tmp_path, slots=8):
    return ShmArena(str(tmp_path / "a.bin"), SLOT, slots)


def _write(a, stem, fill, extents=((0, SLOT),)):
    pay = torch.full((sum(l for _, l in extents),), fill, dtype=torch.uint8)
    st = a.write([stem], [SLOT], [tuple(extents)], [pay.data_ptr()])
    return st[0], pay


def test_find_ref_and_in_place_read(tmp_path):
    a = _arena(tmp_path)
    st, _ = _write(a, "p1", 7)
    assert st == 1, "completed"
    st2, _ = _write(a, "p2", 9, extents=((0, SLOT // 2),))
    assert st2 == 0, "partial"
    found = a.find_slots(["p1", "p2", "absent"])
    assert found[0][1] == 2 and found[1][1] == 1 and found[2] == (-1, 0)
    slot = found[0][0]
    # the page is readable in place through the data offset
    off = a.data_offset() + slot * SLOT
    view = memoryview(a._mm)[off:off + SLOT]
    assert bytes(view[:4]) == b"\x07\x07\x07\x07"
    # a reader reference pins it: the clock cannot evict a referenced slot
    assert a.ref_slots([slot], +1) == 1
    assert a.ref_slots([found[1][0]], +1) == 1, "#1427: a CLAIMED slot takes a reader too (the writer's node holds one from the claim on)"
    assert a.ref_slots([found[1][0]], -1) == 1
    cands = a.evict_candidates(8)
    assert slot not in [c[0] for c in cands]
    assert a.ref_slots([slot], -1) == 1
    cands = a.evict_candidates(8)
    assert slot in [c[0] for c in cands] or a.find_slots(["p1"])[0][1] != 2


def test_release_never_goes_below_zero(tmp_path):
    a = _arena(tmp_path)
    _write(a, "p1", 1)
    slot = a.find_slots(["p1"])[0][0]
    assert a.ref_slots([slot], -1) == 0
    assert a.ref_slots([slot, -1], +1) == 1
    assert a.ref_slots([slot], -1) == 1
