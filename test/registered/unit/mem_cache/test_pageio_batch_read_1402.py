"""#1402: the batched page I/O helper (pageio.c via ctypes) and the file
backend's use of it -- one GIL release per batch instead of one per syscall.
"""

import os
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.storage.file import pageio

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


@pytest.fixture(scope="module")
def pio():
    p = pageio.load()
    assert p is not None, "the helper must build on a box with gcc"
    return p


def test_stat_sizes_reports_size_or_minus_one(tmp_path, pio):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 100)
    sizes = pio.stat_sizes([str(a), str(tmp_path / "missing.bin")])
    assert sizes == [100, -1]
    assert pio.stat_sizes([]) == []


def test_read_pages_cuts_extents_back_to_back_and_touches(tmp_path, pio):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(bytes(range(256)) * 4)  # 1024 bytes
    os.utime(blob, (1_000_000, 1_000_000))
    out = torch.zeros(300, dtype=torch.uint8)
    status = pio.read_pages(
        [str(blob)], [1024], [((0, 100), (512, 200))], [out.data_ptr()], touch=True
    )
    assert status == [0]
    assert out[:100].tolist() == list(range(100))
    assert out[100:300].tolist() == (list(range(256)) * 4)[512:712]
    assert os.stat(blob).st_mtime > 1_000_000 + 1, "touch bumped the mtime"


def test_read_pages_statuses(tmp_path, pio):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"y" * 64)
    os.utime(blob, (1_000_000, 1_000_000))
    out = [torch.zeros(64, dtype=torch.uint8) for _ in range(3)]
    status = pio.read_pages(
        [str(blob), str(tmp_path / "gone.bin"), str(blob)],
        [64, 64, 128],  # third: wrong width
        [((0, 64),), ((0, 64),), ((0, 64),)],
        [t.data_ptr() for t in out],
        touch=False,
    )
    assert status == [0, 1, 2]
    assert out[0].tolist() == [ord("y")] * 64
    assert int(out[2].sum()) == 0, "a refused page leaves the target alone"
    assert os.stat(blob).st_mtime == 1_000_000, "touch=False leaves the mtime"


def test_env_switch_disables_the_helper(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_PAGEIO", "0")
    monkeypatch.setattr(pageio, "_loaded", None)
    monkeypatch.setattr(pageio, "_failed", False)
    assert pageio.load() is None
    monkeypatch.setattr(pageio, "_loaded", None)
    monkeypatch.setattr(pageio, "_failed", False)
    monkeypatch.delenv("SGLANG_HICACHE_PAGEIO")
    assert pageio.load() is not None


def test_batch_beside_a_busy_thread_is_not_a_per_syscall_handoff(tmp_path, pio):
    """The reason this helper exists, on the desk: 64 pages read in one call
    beside a CPU-bound Python thread must cost far less than 64 Python-level
    reads there (measured 13 us vs 20.6 ms a page at the 5 ms default)."""
    import sys
    import threading

    paths = []
    for i in range(64):
        p = tmp_path / f"{i}.bin"
        p.write_bytes(bytes([i]) * 4096)
        paths.append(str(p))
    outs = [torch.zeros(4096, dtype=torch.uint8) for _ in paths]
    ptrs = [o.data_ptr() for o in outs]
    stop = False

    def spin():
        x = 0
        while not stop:
            for j in range(20000):
                x += j

    old = sys.getswitchinterval()
    sys.setswitchinterval(0.005)
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    time.sleep(0.02)
    try:
        t = time.perf_counter()
        status = pio.read_pages(paths, [4096] * 64, [((0, 4096),)] * 64, ptrs, False)
        batched = time.perf_counter() - t
        t = time.perf_counter()
        buf = bytearray(4096)
        for p in paths[:8]:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fstat(fd)
                os.preadv(fd, [memoryview(buf)], 0)
            finally:
                os.close(fd)
        per_syscall = (time.perf_counter() - t) / 8 * 64
    finally:
        stop = True
        th.join()
        sys.setswitchinterval(old)
    assert status == [0] * 64
    assert all(int(o[0]) == i for i, o in enumerate(outs))
    # one hand-off for the batch vs four per page: at least 5x, typically 100x
    assert batched * 5 < per_syscall, (batched, per_syscall)
