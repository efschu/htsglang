# SPDX-License-Identifier: Apache-2.0
"""#456 -- the sparse hibernate write, and the properties that make it safe.

The claim being pinned is narrow and total: **a sparse image and a dense image
of the same payload are the same bytes.** Everything else about the feature
follows from that -- no reader changes, no manifest field, no version gate --
so the suite spends most of its weight there and on the detector that decides
which pages may be skipped.

Falsifiers (each fails without the mechanism it guards):

1. ``test_detector_exact_map`` -- an image with a KNOWN zero-page map: exactly
   those pages become holes and no others. The can-fail arm flips one byte
   inside a "zero" page; that page must be written dense.
2. ``test_sha256_dense_vs_sparse`` -- read-back sha256 equality on a real
   ``torch.save`` payload. Fails if a hole is ever punched over a live byte or
   a trailing hole is dropped.
3. ``test_st_blocks_shrinks`` -- allocation actually falls, guarded by an FS
   capability probe (ZFS folds zeros itself and would make the assertion
   meaningless rather than false).
4. ``test_resume_identical_state`` -- the resume path: state restored from the
   sparse image equals state restored from the dense image, tensor by tensor,
   byte for byte. Hermetic, CPU only.
5. ``test_trailing_hole_preserved`` -- an image ENDING in zeros keeps its
   length. ``lseek`` past EOF is not a write; without the ``ftruncate`` in
   ``close()`` the image would be short by exactly its final zero run.
6. ``test_unaligned_writes_are_byte_exact`` -- writes that straddle page
   boundaries, which is every ``torch.save`` record (they are 64-byte aligned,
   not 4096-byte aligned).
7. ``test_dense_escape_is_bit_identical`` -- ``SGLANG_HIBERNATE_DENSE_WRITE=1``
   produces the same bytes, not merely a loadable file.
8. ``test_park_uses_the_sparse_writer`` -- ``hibernate.park_weights_to_disk``
   reaches the shipped writer, so reverting the wiring cannot leave this file
   green.

Run:  CUDA_VISIBLE_DEVICES=99 python -m pytest tests/hibernate/test_sparse_write_456.py -q
"""

from __future__ import annotations

import ast
import hashlib
import os
import pathlib
import shutil

import numpy as np
import pytest
import torch

from sglang.srt.model_loader.sparse_write import (
    DENSE_WRITE_ENV,
    PAGE_SIZE,
    SparseFileWriter,
    allocated_bytes,
    data_extents,
    dense_write_forced,
    filesystem_supports_holes,
    hole_precise,
    torch_save_sparse,
    zero_page_mask,
)

# ---------------------------------------------------------------------------
# Filesystem capability. The byte-identity properties hold everywhere; the
# ALLOCATION properties can only be asserted where the filesystem reports them
# at page granularity. Probe rather than assume -- /tmp is ZFS on this box.
# ---------------------------------------------------------------------------

_CANDIDATE_DIRS = ("/dev/shm", "/tmp")


def _precise_dir():
    for d in _CANDIDATE_DIRS:
        if os.path.isdir(d) and os.access(d, os.W_OK) and hole_precise(d):
            return d
    return None


PRECISE_DIR = _precise_dir()
precise_only = pytest.mark.skipif(
    PRECISE_DIR is None,
    reason=(
        "no filesystem here reports 4 KiB holes exactly (checked "
        f"{', '.join(_CANDIDATE_DIRS)}); the allocation assertions would be "
        "testing the filesystem, not the writer"
    ),
)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sparse(path: str, blob: bytes, chunk: int = 1 << 20) -> SparseFileWriter:
    w = SparseFileWriter(path)
    mv = memoryview(blob)
    off = 0
    while off < len(blob):
        w.write(mv[off : off + chunk])
        off += chunk
    w.close()
    return w


def _payload():
    """A synthetic 'rank image': live tensors plus the pre-allocated zero
    buffers that are where the real image's 12.64 % of zero pages come from."""
    g = torch.Generator().manual_seed(456)
    return {
        "version": 2,
        "params": {
            "layer.0.qweight": torch.randint(
                0, 255, (512, 1024), dtype=torch.uint8, generator=g
            ),
            "layer.0.scratch": torch.zeros(1024, 1024, dtype=torch.uint8),
            "layer.1.qweight": torch.randint(
                0, 255, (512, 1024), dtype=torch.uint8, generator=g
            ),
            "layer.1.scratch": torch.zeros(512, 1024, dtype=torch.float16),
        },
        "static_state": {"buffers": [("rope.cos", torch.zeros(2048, 64))]},
        "byte_hash": "deadbeef",
    }


# ---------------------------------------------------------------------------
# 1. Detector
# ---------------------------------------------------------------------------


def test_detector_matches_the_306_predicate():
    """``zero_page_mask`` is the probe's ``(min == max) & (min == 0)``."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, 256, size=64 * PAGE_SIZE, dtype=np.uint8)
    zeros = [3, 4, 5, 17, 40, 63]
    for p in zeros:
        a[p * PAGE_SIZE : (p + 1) * PAGE_SIZE] = 0
    pages = a.reshape(-1, PAGE_SIZE)
    reference = (pages.min(axis=1) == pages.max(axis=1)) & (pages.min(axis=1) == 0)
    assert np.array_equal(zero_page_mask(a), reference)
    assert sorted(np.flatnonzero(zero_page_mask(a)).tolist()) == zeros


def test_detector_rejects_a_page_with_one_nonzero_byte():
    """CAN-FAIL ARM. A single non-zero byte anywhere disqualifies the page --
    at the start, in the middle, at the very last offset."""
    for pos in (0, 1, PAGE_SIZE // 2, PAGE_SIZE - 1):
        a = np.zeros(4 * PAGE_SIZE, dtype=np.uint8)
        a[2 * PAGE_SIZE + pos] = 1
        mask = zero_page_mask(a)
        assert mask.tolist() == [True, True, False, True], f"byte at {pos}"


def _holed_pages(path: str, n_pages: int):
    live = set()
    for start, end in data_extents(path):
        live.update(range(start // PAGE_SIZE, -(-end // PAGE_SIZE)))
    return sorted(set(range(n_pages)) - live)


@precise_only
@pytest.mark.parametrize("chunk_pages,chunk_extra", [(7, 0), (7, 1000), (1, 3)])
def test_detector_exact_map(chunk_pages, chunk_extra):
    """Exactly the known-zero pages become holes; no live page ever does.

    Parametrised over chunk sizes that are NOT page multiples, because holes
    are aligned in FILE-offset space, not in write-call space. A writer that
    paged relative to each ``write()`` instead of to the file offset still
    produces byte-correct output (a misaligned hole reads back as zeros too),
    so only the extent map catches it -- which is why this assertion is on
    extents and not on the bytes.
    """
    d = (
        pathlib.Path(PRECISE_DIR)
        / f".456_exact_{os.getpid()}_{chunk_pages}_{chunk_extra}"
    )
    d.mkdir(parents=True, exist_ok=True)
    chunk = chunk_pages * PAGE_SIZE + chunk_extra
    try:
        rng = np.random.default_rng(1)
        n_pages = 48
        a = rng.integers(1, 256, size=n_pages * PAGE_SIZE, dtype=np.uint8)
        zero_pages = [0, 1, 2, 10, 11, 30, 46, 47]
        for p in zero_pages:
            a[p * PAGE_SIZE : (p + 1) * PAGE_SIZE] = 0
        blob = a.tobytes()

        p = str(d / "exact.bin")
        w = _write_sparse(p, blob, chunk=chunk)
        assert os.path.getsize(p) == len(blob)
        assert open(p, "rb").read() == blob
        assert _holed_pages(p, n_pages) == zero_pages
        assert w.stats.hole_pages == len(zero_pages)

        # CAN-FAIL: corrupt one byte inside a "zero" page -> that page is
        # written dense and stops being a hole; the others are unaffected.
        a[11 * PAGE_SIZE + 4000] = 0xFF
        p2 = str(d / "corrupt.bin")
        _write_sparse(p2, a.tobytes(), chunk=chunk)
        assert _holed_pages(p2, n_pages) == [z for z in zero_pages if z != 11]
        assert open(p2, "rb").read() == a.tobytes()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@precise_only
def test_hole_map_is_independent_of_chunking():
    """The same image chunked seven different ways yields the SAME hole map.

    Holes live in file-offset space; ``write()`` call boundaries do not. The
    writer carries the trailing partial page into the next call so the two
    never get confused -- drop the carry and 4099-byte chunks punch one page
    in the whole image instead of eight.
    """
    d = pathlib.Path(PRECISE_DIR) / f".456_chunkindep_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        rng = np.random.default_rng(3)
        n_pages = 40
        a = rng.integers(1, 256, size=n_pages * PAGE_SIZE, dtype=np.uint8)
        zero_pages = [4, 5, 6, 7, 20, 39]
        for p in zero_pages:
            a[p * PAGE_SIZE : (p + 1) * PAGE_SIZE] = 0
        blob = a.tobytes()
        for chunk in (1, 64, 1000, 4095, 4096, 4099, 12345, len(blob)):
            p = str(d / f"c{chunk}.bin")
            w = _write_sparse(p, blob, chunk=chunk)
            assert open(p, "rb").read() == blob, chunk
            assert _holed_pages(p, n_pages) == zero_pages, chunk
            assert w.stats.hole_pages == len(zero_pages), chunk
            os.unlink(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Byte identity
# ---------------------------------------------------------------------------


def test_sha256_dense_vs_sparse(tmp_path):
    dense = str(tmp_path / "dense.pt")
    sparse = str(tmp_path / "sparse.pt")
    obj = _payload()
    assert torch_save_sparse(obj, dense, force_dense=True) is None
    stats = torch_save_sparse(obj, sparse, force_dense=False)
    assert stats is not None and stats.hole_pages > 0
    assert os.path.getsize(dense) == os.path.getsize(sparse)
    assert _sha(dense) == _sha(sparse)
    assert stats.written_bytes < stats.logical_bytes


def test_trailing_hole_preserved(tmp_path):
    """An image whose last bytes are zeros keeps its full length."""
    blob = b"\xaa" * PAGE_SIZE + b"\x00" * (5 * PAGE_SIZE)
    p = str(tmp_path / "tail.bin")
    w = _write_sparse(p, blob)
    assert w.stats.hole_pages == 5
    assert os.path.getsize(p) == len(blob)
    assert open(p, "rb").read() == blob


def test_unaligned_writes_are_byte_exact(tmp_path):
    """Records that straddle page boundaries -- the normal case for a
    ``torch.save`` container, whose records are 64-byte aligned."""
    rng = np.random.default_rng(2)
    blob = bytearray(rng.integers(1, 256, size=9 * PAGE_SIZE, dtype=np.uint8).tobytes())
    blob[3 * PAGE_SIZE : 6 * PAGE_SIZE] = b"\x00" * (3 * PAGE_SIZE)
    blob = bytes(blob)
    for chunk in (1, 63, 64, 4095, 4097, 5000, 3 * PAGE_SIZE + 17):
        p = str(tmp_path / f"u{chunk}.bin")
        _write_sparse(p, blob, chunk=chunk)
        assert open(p, "rb").read() == blob, f"chunk={chunk}"
        assert os.path.getsize(p) == len(blob), f"chunk={chunk}"


def test_empty_and_tiny_images(tmp_path):
    for blob in (b"", b"\x00", b"\x00" * (PAGE_SIZE - 1), b"\x00" * PAGE_SIZE):
        p = str(tmp_path / f"t{len(blob)}.bin")
        _write_sparse(p, blob)
        assert open(p, "rb").read() == blob
        assert os.path.getsize(p) == len(blob)


# ---------------------------------------------------------------------------
# 3. Allocation
# ---------------------------------------------------------------------------


@precise_only
def test_st_blocks_shrinks():
    d = pathlib.Path(PRECISE_DIR) / f".456_blocks_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    dense = str(d / "dense.pt")
    sparse = str(d / "sparse.pt")
    try:
        obj = _payload()
        torch_save_sparse(obj, dense, force_dense=True)
        torch_save_sparse(obj, sparse, force_dense=False)
        assert allocated_bytes(sparse) < allocated_bytes(dense)
        assert _sha(dense) == _sha(sparse)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_capability_probes_agree_with_reality(tmp_path):
    """The probes that guard the assertions above are themselves checked: a
    filesystem reported hole-precise must really return exact extents."""
    if hole_precise(str(tmp_path)):
        assert filesystem_supports_holes(str(tmp_path))
        blob = b"\xaa" * PAGE_SIZE + b"\x00" * (2 * PAGE_SIZE) + b"\xbb" * PAGE_SIZE
        p = str(tmp_path / "probe.bin")
        _write_sparse(p, blob)
        assert data_extents(p) == [(0, PAGE_SIZE), (3 * PAGE_SIZE, 4 * PAGE_SIZE)]


# ---------------------------------------------------------------------------
# 4. Resume path
# ---------------------------------------------------------------------------


def test_resume_identical_state(tmp_path):
    """The state a restore reads back is identical from either image.

    This is the property the hibernate restore actually depends on -- it does
    not compare files, it compares the tensors it loaded against the parked
    ``byte_hash``. Hermetic: synthetic payload, CPU only, no GPU.
    """
    dense = str(tmp_path / "dense.pt")
    sparse = str(tmp_path / "sparse.pt")
    obj = _payload()
    torch_save_sparse(obj, dense, force_dense=True)
    torch_save_sparse(obj, sparse, force_dense=False)

    a = torch.load(dense, map_location="cpu", weights_only=False)
    b = torch.load(sparse, map_location="cpu", weights_only=False)
    assert a.keys() == b.keys()
    assert a["params"].keys() == b["params"].keys()

    def digest(state):
        h = hashlib.sha256()
        for name in sorted(state["params"]):
            t = state["params"][name].contiguous()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            h.update(t.view(torch.uint8).reshape(-1).numpy().tobytes())
        for name, t in state["static_state"]["buffers"]:
            h.update(name.encode())
            h.update(t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes())
        return h.hexdigest()

    assert digest(a) == digest(b)
    for name in a["params"]:
        assert torch.equal(a["params"][name], b["params"][name]), name
    # And the zero buffers really are zero after the round trip -- a hole that
    # read back as anything else would be caught here, not just by the hash.
    assert not b["params"]["layer.0.scratch"].any()
    assert not b["params"]["layer.1.scratch"].any()


# ---------------------------------------------------------------------------
# 5. Default / escape hatch
# ---------------------------------------------------------------------------


def test_sparse_is_the_default(monkeypatch):
    monkeypatch.delenv(DENSE_WRITE_ENV, raising=False)
    assert dense_write_forced() is False


@pytest.mark.parametrize(
    "value,forced",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("", False),
    ],
)
def test_dense_escape_parses(value, forced, monkeypatch):
    monkeypatch.setenv(DENSE_WRITE_ENV, value)
    assert dense_write_forced() is forced


def test_dense_escape_is_bit_identical(tmp_path, monkeypatch):
    obj = _payload()
    explicit = str(tmp_path / "explicit.pt")
    torch_save_sparse(obj, explicit, force_dense=True)

    monkeypatch.setenv(DENSE_WRITE_ENV, "1")
    via_env = str(tmp_path / "env.pt")
    assert torch_save_sparse(obj, via_env) is None

    monkeypatch.delenv(DENSE_WRITE_ENV, raising=False)
    default = str(tmp_path / "default.pt")
    assert torch_save_sparse(obj, default) is not None

    assert _sha(explicit) == _sha(via_env) == _sha(default)


# ---------------------------------------------------------------------------
# 6. Wiring
# ---------------------------------------------------------------------------


def test_park_uses_the_sparse_writer():
    """``park_weights_to_disk`` calls the shipped writer, not ``torch.save``.

    Source-level so it needs no GPU and no distributed group: reverting the
    single line in ``hibernate.py`` must turn this file red.
    """
    import sglang.srt.model_loader.hibernate as hib

    src = pathlib.Path(hib.__file__).read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "park_weights_to_disk"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "torch_save_sparse" in calls
    assert "torch.save" not in calls
    assert hib.torch_save_sparse is torch_save_sparse
    assert hib.DENSE_WRITE_ENV == DENSE_WRITE_ENV


def test_restore_path_is_untouched():
    """No version gate, no manifest field, no reader change: the image format
    did not move, so ``HIBERNATE_VERSION`` must not have been bumped for this
    and the restore must still be a plain ``torch.load``."""
    import sglang.srt.model_loader.hibernate as hib

    assert hib.HIBERNATE_VERSION == 2
    src = pathlib.Path(hib.__file__).read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "restore_model_from_disk"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "torch.load" in calls
    assert not any("sparse" in c for c in calls)
