"""#1404 BAUPLAN_SHM_ARENA_0916: the shared-memory page arena, protocol level.

Two writers complete one canonical page extent by extent, a reader cuts its
own extents, a second PROCESS on the same file sees the page, refused shapes
are refused, a full arena evicts complete unreferenced pages by clock, and
a crashed writer's half page stays invisible.
"""

import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.storage.file import hicache_arena as ha

pytestmark = pytest.mark.skipif(
    ha._load_lib() is None, reason="needs gcc"
)

PAGE = 32768


@pytest.fixture
def arena(tmp_path):
    path = str(tmp_path / "arena-test.bin")  # any mmap-able file; /dev/shm only matters for RAM residency in serving
    a = ha.ShmArena(path, PAGE, 64)
    yield a
    a.close()
    os.remove(path)


def _page(fill, n=PAGE):
    return torch.full((n,), fill, dtype=torch.uint8)


def test_two_stages_complete_one_page_and_a_reader_cuts_its_extents(arena):
    assert arena.fresh
    a = _page(7, 16384)
    b = _page(9, 16384)
    assert arena.write(["k1"], [PAGE], [[(0, 16384)]], [a.data_ptr()]) == [0]
    assert arena.lookup(["k1"]) == [False], "a partial page is invisible"
    assert arena.write(["k1"], [PAGE], [[(16384, 16384)]], [b.data_ptr()]) == [1]
    assert arena.lookup(["k1"]) == [True]
    out = torch.zeros(1024, dtype=torch.uint8)
    assert arena.read(["k1"], [PAGE], [[(16000, 1024)]], [out.data_ptr()]) == [0]
    assert out[:384].tolist() == [7] * 384 and out[384:].tolist() == [9] * 640
    assert arena.write(["k1"], [PAGE], [[(0, 16384)]], [a.data_ptr()]) == [2]
    assert arena.read(["k1"], [PAGE - 256], [[(0, 256)]], [out.data_ptr()]) == [2], "width"
    assert arena.read(["nope"], [PAGE], [[(0, 256)]], [out.data_ptr()]) == [1]


def test_unaligned_extent_is_refused_not_stored(arena):
    a = _page(1, 512)
    assert arena.write(["k2"], [PAGE], [[(100, 512)]], [a.data_ptr()]) == [3]
    assert arena.lookup(["k2"]) == [False]


def test_full_arena_reports_full_and_clock_evicts_complete_pages(arena):
    full = _page(3)
    keys = [f"f{i}" for i in range(70)]
    sts = arena.write(keys, [PAGE] * 70, [[(0, PAGE)]] * 70, [full.data_ptr()] * 70)
    assert sts.count(1) == 64 and sts.count(4) == 6
    st = arena.stats()
    assert st["complete"] == 64
    ev = arena.evict_candidates(8, keep_stems=["f0"])
    assert len(ev) == 8 and all(t == PAGE for _, _, _, t in ev)
    assert ha.key128("f0")[0] not in [lo for _, lo, _, _ in ev], "pinned stays"
    arena.free_slots([s for s, _, _, _ in ev])
    assert arena.stats()["complete"] == 56
    # the freed keys are gone, the rest still readable
    gone = [k for k in keys[:64] if not arena.lookup([k])[0]]
    assert len(gone) == 8 and "f0" not in gone
    again = arena.write(keys[64:], [PAGE] * 6, [[(0, PAGE)]] * 6, [full.data_ptr()] * 6)
    assert again == [1] * 6


def test_a_second_process_reads_what_the_first_wrote(arena):
    a = _page(5)
    assert arena.write(["shared"], [PAGE], [[(0, PAGE)]], [a.data_ptr()]) == [1]
    code = (
        "import os,torch,sys\n"
        "from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena\n"
        f"b=ShmArena({arena.path!r},{PAGE},64)\n"
        "o=torch.zeros(4096,dtype=torch.uint8)\n"
        f"st=b.read(['shared'],[{PAGE}],[[(1024,4096)]],[o.data_ptr()])\n"
        "print(st[0], int(o.min()), int(o.max()), b.fresh)\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "0 5 5 False"


def test_a_crashed_writer_leaves_an_invisible_half_page(arena):
    a = _page(7, 16384)
    assert arena.write(["half"], [PAGE], [[(0, 16384)]], [a.data_ptr()]) == [0]
    # the writer "dies" here; nobody completes the page
    assert arena.lookup(["half"]) == [False]
    out = torch.zeros(256, dtype=torch.uint8)
    assert arena.read(["half"], [PAGE], [[(0, 256)]], [out.data_ptr()]) == [1]
    # a later writer of the same key continues the same slot
    b = _page(9, 16384)
    assert arena.write(["half"], [PAGE], [[(16384, 16384)]], [b.data_ptr()]) == [1]
    assert arena.lookup(["half"]) == [True]
