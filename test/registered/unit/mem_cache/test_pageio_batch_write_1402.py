"""#1402: the batched canonical extent WRITE (pageio.c hicache_write_pages)
speaks exactly the protocol of canonical_page_store.write_extents -- partial
blobs stay invisible, markers are byte-identical in both directions, and a
blob started by one implementation is completed by the other.
"""

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache import canonical_page_store as cps
from sglang.srt.mem_cache.storage.file import pageio

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 256


@pytest.fixture(scope="module")
def pio():
    p = pageio.load()
    assert p is not None
    return p


def _win(extents):
    return cps.CanonicalExtentWindow(label="t", total_bytes=TOTAL, extents=tuple(extents))


def _payload(extents, fill):
    n = sum(l for _, l in extents)
    return torch.full((n,), fill, dtype=torch.uint8)


def _final(tmp_path, name="p.bin"):
    d = tmp_path / "ab"
    d.mkdir(exist_ok=True)
    return str(d / name)


def test_partial_then_complete_in_c_matches_python_marker_and_bytes(tmp_path, pio):
    final = _final(tmp_path)
    a = [(0, 64), (192, 64)]
    pa = _payload(a, 7)
    st = pio.write_pages([final], [TOTAL], [a], [pa.data_ptr()], False)
    assert st == [1]
    assert not os.path.exists(final), "a partial blob is invisible"
    # the marker is what Python would have written
    marker = open(cps.marker_path(final), "rb").read()
    assert marker == cps._encode_marker(TOTAL, [(0, 64), (192, 256)])
    assert cps._decode_marker(marker, TOTAL) == ((0, 64), (192, 256))
    b = [(64, 128)]
    pb = _payload(b, 9)
    st = pio.write_pages([final], [TOTAL], [b], [pb.data_ptr()], False)
    assert st == [0]
    assert os.path.exists(final) and not os.path.exists(cps.marker_path(final))
    assert not os.path.exists(cps.part_path(final))
    data = open(final, "rb").read()
    assert data == bytes([7] * 64 + [9] * 128 + [7] * 64)
    # a third write of a complete blob is "already complete"
    assert pio.write_pages([final], [TOTAL], [b], [pb.data_ptr()], False) == [2]


def test_python_starts_and_c_completes_and_vice_versa(tmp_path, pio):
    a = [(0, 128)]
    b = [(128, 128)]
    # Python partial -> C completes
    f1 = _final(tmp_path, "one.bin")
    r = cps.write_extents(f1, _win(a), _payload(a, 1), fsync=False)
    assert not r.completed and not os.path.exists(f1)
    p2 = _payload(b, 2)  # held: the pointer must outlive the call
    assert pio.write_pages([f1], [TOTAL], [b], [p2.data_ptr()], False) == [0]
    assert open(f1, "rb").read() == bytes([1] * 128 + [2] * 128)
    # C partial -> Python completes
    f2 = _final(tmp_path, "two.bin")
    p3 = _payload(a, 3)
    assert pio.write_pages([f2], [TOTAL], [a], [p3.data_ptr()], False) == [1]
    r = cps.write_extents(f2, _win(b), _payload(b, 4), fsync=False)
    assert r.completed and os.path.exists(f2)
    assert open(f2, "rb").read() == bytes([3] * 128 + [4] * 128)
    out = torch.zeros(TOTAL, dtype=torch.uint8)
    assert cps.read_extents(f2, _win([(0, TOTAL)]), out)
    assert out.tolist() == [3] * 128 + [4] * 128


def test_a_foreign_marker_resets_the_partial(tmp_path, pio):
    final = _final(tmp_path, "three.bin")
    part = cps.part_path(final)
    with open(part, "wb") as f:
        f.write(b"\xff" * TOTAL)  # leftover of another geometry, no marker
    a = [(0, 128)]
    p5 = _payload(a, 5)
    assert pio.write_pages([final], [TOTAL], [a], [p5.data_ptr()], False) == [1]
    assert cps._decode_marker(open(cps.marker_path(final), "rb").read(), TOTAL) == ((0, 128),)
    assert open(part, "rb").read() == bytes([5] * 128 + [0] * 128), "reset, not reused"


def test_batch_mixes_statuses(tmp_path, pio):
    f_done = _final(tmp_path, "done.bin")
    with open(f_done, "wb") as f:
        f.write(b"z" * TOTAL)
    f_new = _final(tmp_path, "new.bin")
    full = [(0, TOTAL)]
    p1, p2 = _payload(full, 1), _payload(full, 2)
    st = pio.write_pages(
        [f_done, f_new], [TOTAL, TOTAL], [full, full],
        [p1.data_ptr(), p2.data_ptr()], False,
    )
    assert st == [2, 0]
    assert open(f_done, "rb").read() == b"z" * TOTAL, "a complete blob is never rewritten"
    assert open(f_new, "rb").read() == bytes([2] * TOTAL)
