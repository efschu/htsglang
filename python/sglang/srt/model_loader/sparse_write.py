# SPDX-License-Identifier: Apache-2.0
"""Sparse image writes (#456) -- skip the all-zero pages of a disk image.

#306 measured a real #89 hibernate image (Qwen3.5-9B Q4_K_M, rank 0, 6.68 GiB)
and decomposed what a "compress the hibernate image" feature would actually
buy (``docs/dev/ANALYSE_306_lossless_ratio.md`` §J):

* **12.64 %** of the image's 4 KiB pages are entirely zero -- 221 542 of
  1 752 229. Removing those is a **1.1447x** ratio at zero CPU and with no
  decompress stage on the reload path.
* the residual codec gain over the non-hole part is only **1.0442x**, and the
  ZFS dataset the image lives on already returns 1.1735x by itself. The codec
  was refuted for this tier; the holes are the surviving win.

The holes are not incidental: a ``torch.save`` container of final GGUF weights
carries pre-allocated buffers written out in full. They are zeros in the image
because they were zeros in VRAM.

This module is the mechanism, kept separate from :mod:`hibernate` so a second
image writer can adopt it without importing the model loader.

**The format is not touched.** :class:`SparseFileWriter` is a byte sink that
sits UNDER whatever container is being written -- it detects all-zero pages in
FILE-OFFSET space and ``lseek()``s over them instead of calling ``write()``.
POSIX guarantees that a hole reads back as zeros, which is exactly what those
pages contained, so every reader -- ``torch.load``, ``mmap``, ``read()``,
``sha256sum`` -- sees a byte-identical file. There is no header, no framing and
no version gate, because there is nothing for a reader to know.

Page definition matches the probe's (``scripts/dev/306_ratio_probe/
image_whole.py``): 4096 bytes, all-zero, aligned to the file offset.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Hole granularity. 4 KiB is the page size the #306 probe measured with and
#: the smallest unit any of the candidate filesystems (ext4, xfs, tmpfs) will
#: deallocate. ZFS punches at its recordsize, so on ZFS this is a lower bound
#: on what actually becomes a hole -- see :func:`filesystem_supports_holes`.
PAGE_SIZE = 4096

#: Escape hatch. Sparse writing is safe by construction (readers see identical
#: bytes), so it is the default; this forces the old dense write for a
#: filesystem or a tool that mishandles holes.
DENSE_WRITE_ENV = "SGLANG_HIBERNATE_DENSE_WRITE"


def zero_page_mask(arr: np.ndarray, page: int = PAGE_SIZE) -> np.ndarray:
    """Boolean mask over ``arr.size // page`` pages: True where all-zero.

    Same predicate as the #306 probe's ``(min == max) & (min == 0)``, computed
    as a 64-bit OR-reduction because the detector is a second full pass over
    the image and its cost is what decides whether the skipped write is a win.

    Measured on this box, 1 GiB, best of 4 (``scripts/dev/456_sparse_write``):
    uint8 ``any`` 63.9 ms, uint64 ``bitwise_or.reduce`` **56.5 ms**, and three
    sampled pre-filters (first word per page, first+middle, contiguous copy of
    the first column) at 62.1 / 63.2 / 57.0 ms. The sampling idea -- a non-zero
    probe proves a page non-zero without reading the rest -- does NOT pay here:
    the hardware prefetcher streams the intervening cache lines regardless, so
    the strided read costs the same as the sequential one while adding a second
    gather stage. The simple reduction wins and is kept.
    """
    n_pages = arr.size // page
    if n_pages == 0:
        return np.zeros(0, dtype=bool)
    flat = arr[: n_pages * page]
    try:
        wide = flat.view(np.uint64).reshape(n_pages, page // 8)
    except ValueError:  # unviewable buffer -> the byte reduction is correct
        return np.asarray(~flat.reshape(n_pages, page).any(axis=1))
    return np.bitwise_or.reduce(wide, axis=1) == 0


class SparseWriteStats:
    """What one image write skipped. Cheap enough to always collect."""

    __slots__ = ("logical_bytes", "written_bytes", "hole_pages", "total_pages")

    def __init__(self) -> None:
        self.logical_bytes = 0
        self.written_bytes = 0
        self.hole_pages = 0
        self.total_pages = 0

    @property
    def hole_bytes(self) -> int:
        return self.hole_pages * PAGE_SIZE

    @property
    def hole_fraction(self) -> float:
        return self.hole_pages / self.total_pages if self.total_pages else 0.0

    @property
    def sparse_ratio(self) -> float:
        """Apparent size / bytes actually handed to ``write()``."""
        return self.logical_bytes / self.written_bytes if self.written_bytes else 1.0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SparseWriteStats(logical={self.logical_bytes}, "
            f"written={self.written_bytes}, holes={self.hole_pages}/"
            f"{self.total_pages}, ratio={self.sparse_ratio:.4f})"
        )


class SparseFileWriter:
    """Write-only file object that turns all-zero pages into holes.

    Implements the minimum ``torch.save`` asks of a buffer (``write`` +
    ``flush``); ``tell`` is provided because it is free and callers expect it.

    A page can only become a hole if it is ALIGNED in file-offset space, which
    is not the same as being aligned inside a ``write()`` call: a ``torch.save``
    record boundary lands on 64 bytes, not on 4096. So the writer holds back
    the trailing partial page of each call (< 4096 bytes) and completes it from
    the next one, which makes the set of detected holes independent of how the
    caller happens to chunk its writes. Without that carry, a caller writing in
    4099-byte pieces would punch one page in the whole image; with it, the
    result is identical to a single write of the concatenation.

    Data is not visible in the file until :meth:`close`, which also
    ``ftruncate``s to the logical size -- ``lseek`` past EOF is not a write, so
    a trailing zero run would otherwise silently shorten the image.
    """

    def __init__(self, path: str, *, page_size: int = PAGE_SIZE) -> None:
        if page_size <= 0 or page_size % 512:
            raise ValueError(
                f"page_size must be a positive multiple of 512, got {page_size}"
            )
        self._page = page_size
        self._path = path
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        # Invariant: ``_off`` (bytes already emitted, written or punched) is
        # always a multiple of ``_page``, and ``_pending`` holds the < page
        # bytes that logically follow it.
        self._off = 0
        self._pending = bytearray()
        self._closed = False
        self.stats = SparseWriteStats()

    # -- file-object surface -------------------------------------------------

    def write(self, data: Any) -> int:
        if self._closed:
            raise ValueError("write on a closed SparseFileWriter")
        mv = memoryview(data)
        if mv.format != "B" or mv.ndim != 1:
            mv = mv.cast("B")
        n = mv.nbytes
        if n == 0:
            return 0
        page = self._page

        taken = 0
        if self._pending:
            need = page - len(self._pending)
            taken = min(need, n)
            self._pending += mv[:taken]
            if len(self._pending) < page:
                self.stats.logical_bytes = self.tell()
                return n
            done = bytes(self._pending)
            self._pending.clear()
            self._emit(memoryview(done), np.frombuffer(done, dtype=np.uint8))

        rest = n - taken
        full = rest - (rest % page)
        if full:
            self._emit(
                mv[taken : taken + full],
                np.frombuffer(mv, dtype=np.uint8, count=full, offset=taken),
            )
        if full != rest:
            self._pending += mv[taken + full :]

        self.stats.logical_bytes = self.tell()
        return n

    def flush(self) -> None:
        # Nothing is buffered that can be emitted yet: a partial page must stay
        # pending until either it is completed or close() decides its fate.
        pass

    def tell(self) -> int:
        return self._off + len(self._pending)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            end = self.tell()
            if self._pending:
                # A trailing partial page: punch it if it is all zeros (the
                # ftruncate below materializes it as a hole), write it if not.
                if any(self._pending):
                    self._raw_write(memoryview(bytes(self._pending)))
                else:
                    self._punch(len(self._pending))
                self._pending.clear()
            os.ftruncate(self._fd, end)
            self.stats.logical_bytes = end
        finally:
            os.close(self._fd)

    # -- internals -----------------------------------------------------------

    def _emit(self, mv: memoryview, arr: np.ndarray) -> None:
        """Emit a whole number of page-aligned pages: write the live runs,
        lseek over the zero runs."""
        page = self._page
        nonzero = ~zero_page_mask(arr, page)
        self.stats.total_pages += int(nonzero.size)
        if not nonzero.any():
            self._punch(nonzero.size * page)
            self.stats.hole_pages += int(nonzero.size)
            return
        if nonzero.all():
            self._raw_write(mv)
            return
        # Run-length encode the mask: a long zero region becomes one lseek and
        # a long live region one write, so the syscall count tracks the number
        # of regions, not the number of pages.
        bounds = np.flatnonzero(nonzero[1:] != nonzero[:-1]) + 1
        starts = np.concatenate(([0], bounds)).tolist()
        ends = np.concatenate((bounds, [nonzero.size])).tolist()
        for s, e in zip(starts, ends):
            if nonzero[s]:
                self._raw_write(mv[s * page : e * page])
            else:
                self._punch((e - s) * page)
                self.stats.hole_pages += e - s

    def __enter__(self) -> "SparseFileWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _raw_write(self, mv: memoryview) -> None:
        off = 0
        total = mv.nbytes
        while off < total:
            off += os.write(self._fd, mv[off:])
        self._off += total
        self.stats.written_bytes += total

    def _punch(self, nbytes: int) -> None:
        os.lseek(self._fd, nbytes, os.SEEK_CUR)
        self._off += nbytes


def dense_write_forced(env: Optional[dict] = None) -> bool:
    """Is the ``SGLANG_HIBERNATE_DENSE_WRITE`` escape hatch engaged?"""
    raw = (env if env is not None else os.environ).get(DENSE_WRITE_ENV, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def torch_save_sparse(obj: Any, path: str, *, force_dense: Optional[bool] = None):
    """``torch.save(obj, path)`` with all-zero pages left as holes.

    Returns the :class:`SparseWriteStats` for the write, or ``None`` when the
    dense path was taken. The resulting file is byte-identical to the dense
    one on read-back either way.
    """
    import torch

    if force_dense is None:
        force_dense = dense_write_forced()
    if force_dense:
        # Through a file OBJECT, not the path: torch derives the ZIP archive
        # name from the path it is given, so ``torch.save(obj, path)`` and
        # ``torch.save(obj, sink)`` produce containers that differ in their
        # internal entry names (and therefore in length). Both branches take
        # the buffer path so the escape hatch is bit-identical to the default,
        # not merely equivalent -- which is what makes the dense-vs-sparse
        # sha256 gate in the tests a real gate.
        with open(path, "wb") as f:
            torch.save(obj, f)
        return None
    with SparseFileWriter(path) as sink:
        torch.save(obj, sink)
        stats = sink.stats
    # torch.save may end with a small dense record (the ZIP end-of-central-
    # directory), so the final offset is authoritative for the apparent size.
    stats.logical_bytes = max(stats.logical_bytes, os.path.getsize(path))
    return stats


def filesystem_supports_holes(directory: str, *, probe_bytes: int = 8 << 20) -> bool:
    """Does ``st_blocks`` on this filesystem actually shrink for a hole?

    Guards the ``st_blocks`` assertions in tests. ext4/xfs/tmpfs say yes; a
    filesystem that folds zeros by other means (ZFS with compression) may
    report the same allocation for a dense write and a sparse one, which makes
    the assertion meaningless there rather than false.
    """
    import tempfile

    fd, path = tempfile.mkstemp(dir=directory)
    try:
        os.ftruncate(fd, probe_bytes)
        os.fsync(fd)
        st = os.fstat(fd)
        return st.st_blocks * 512 < st.st_size
    except OSError:
        return False
    finally:
        os.close(fd)
        os.unlink(path)


def allocated_bytes(path: str) -> int:
    """Bytes the filesystem has allocated for ``path`` (``st_blocks`` x 512)."""
    return os.stat(path).st_blocks * 512


def data_extents(path: str):
    """``[(start, end), ...]`` of the file's non-hole regions, via SEEK_DATA.

    The direct read of what became a hole -- ``st_blocks`` only gives a total.
    Granularity is the filesystem's, not this module's: ext4/tmpfs answer at
    block granularity, ZFS answers at recordsize and will report a file with
    4 KiB holes as one solid extent. Diagnostics and tests only.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        out = []
        off = 0
        while off < size:
            try:
                start = os.lseek(fd, off, os.SEEK_DATA)
            except OSError:  # ENXIO: only holes remain
                break
            if start >= size:
                break
            try:
                end = min(os.lseek(fd, start, os.SEEK_HOLE), size)
            except OSError:
                end = size
            out.append((start, end))
            off = end
        return out
    finally:
        os.close(fd)


def hole_precise(directory: str) -> bool:
    """Does this filesystem report 4 KiB holes exactly (SEEK_DATA + st_blocks)?

    Guards the exact-extent falsifier. tmpfs and ext4 answer yes; ZFS answers
    no -- it folds and re-reports at recordsize, so a 4 KiB assertion there
    would be testing the filesystem, not this module.
    """
    import tempfile

    fd, path = tempfile.mkstemp(dir=directory)
    try:
        os.write(fd, b"\xaa" * PAGE_SIZE)
        os.lseek(fd, PAGE_SIZE * 3, os.SEEK_CUR)
        os.write(fd, b"\xbb" * PAGE_SIZE)
        os.ftruncate(fd, PAGE_SIZE * 5)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        want = [(0, PAGE_SIZE), (PAGE_SIZE * 4, PAGE_SIZE * 5)]
        return data_extents(path) == want
    except OSError:
        return False
    finally:
        if fd >= 0:
            os.close(fd)
        os.unlink(path)
