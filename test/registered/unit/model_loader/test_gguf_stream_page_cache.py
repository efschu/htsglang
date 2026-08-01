# SPDX-License-Identifier: Apache-2.0
"""#391 boot-9 blocker: the GGUF page cache must not compete with the host pool.

``gguf.GGUFReader`` maps each part with ``np.memmap``, so streaming a checkpoint
faults the whole file into the page cache and nothing gives it back. On boot
attempt 8 (2026-08-01) that left ~48+ GiB of clean file pages fighting the
loader's own anonymous memory on a 98.5 GiB swapless box: ``memory.current``
sat at 98.3 GiB for seven minutes while the kernel traded file for anon one for
one (file 81.6 -> 63.8 GiB, anon 15.1 -> 32.9 GiB) until a 2.4 GiB anon burst
inside one 15 s window outran reclaim and the OOM killer fired.

Four proofs, all hermetic and all able to fail:

a) **bounded vs accumulating** -- a reader loop over a temp file with the advice
   ON leaves near-zero resident pages, with it OFF the whole file stays resident
   (the boot-8 shape). Residency is read with ``mincore(2)``, which reports page
   cache residency for the mapped range, not RSS.
a2) **why not just fadvise** -- ``posix_fadvise(DONTNEED)`` alone, over a range
   that is still mapped, frees close to nothing. That is the naive fix, and it
   is measured failing here so nobody re-proposes it. (On the ZFS pool the
   checkpoints live on it does not free them after the unmap either; see the
   ladder comment in ``gguf_shards.py``.)
b) **byte identity** -- the bytes the loop reads are the same with the advice on
   and off. A read-only shared mapping of an unmodified file re-faults the same
   bytes, so this must hold exactly.
c) **opt-out** -- ``SGLANG_GGUF_STREAM_DROP_CACHE=0`` restores accumulation.
d) **never advise ahead** -- a contract test on the bookkeeping: no advised byte
   range may overlap an unconsumed tensor's extent, including under out-of-order
   consumption and pages shared between neighbouring tensors.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import logging
import mmap
import os
import tempfile
import unittest
from typing import List, Optional, Tuple

import numpy as np

from sglang.srt.model_loader.gguf_shards import (
    ConsumedPageDropper,
    make_stream_page_dropper,
    stream_drop_cache_enabled,
)

PAGE = os.sysconf("SC_PAGE_SIZE")
#: Big enough that the residency signal is unambiguous, small enough to write
#: and drop in well under a second even on a busy box.
FILE_BYTES = 192 << 20
#: One "tensor" per chunk of the synthetic file. 6 MiB keeps the extent count
#: realistic (32 extents) without making the loop slow.
CHUNK_BYTES = 6 << 20


def _load_mincore():
    """``mincore(void*, size_t, unsigned char*)`` via libc, or None."""
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    libc = ctypes.CDLL(name, use_errno=True)
    fn = getattr(libc, "mincore", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    fn.restype = ctypes.c_int
    return fn


_MINCORE = _load_mincore()


def resident_pages(array: np.memmap) -> int:
    """How many pages of ``array``'s mapping are present in the page cache."""
    length = array.nbytes
    npages = (length + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(npages)
    rc = _MINCORE(ctypes.c_void_p(array.ctypes.data), ctypes.c_size_t(length), vec)
    if rc != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(1 for byte in vec.raw[:npages] if byte & 1)


def write_temp_file(directory: str) -> str:
    path = os.path.join(directory, "shard-00001-of-00001.bin")
    block = np.arange(1 << 20, dtype=np.uint8)
    with open(path, "wb") as handle:
        written = 0
        while written < FILE_BYTES:
            chunk = block[: min(len(block), FILE_BYTES - written)]
            handle.write(chunk.tobytes())
            written += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def drop_whole_file(path: str) -> None:
    """Start every measurement from a cold page cache for this file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def extents(total: int, chunk: int) -> List[Tuple[int, int]]:
    return [(offset, min(chunk, total - offset)) for offset in range(0, total, chunk)]


class _FakeReaderTensor:
    def __init__(self, data_offset: int, n_bytes: int) -> None:
        self.data_offset = data_offset
        self.n_bytes = n_bytes


class _FakeReader:
    """Enough of ``gguf.GGUFReader`` for ``ConsumedPageDropper.for_readers``."""

    def __init__(self, array: np.memmap, layout: List[Tuple[int, int]]) -> None:
        self.data = array
        self.tensors = [_FakeReaderTensor(off, n) for off, n in layout]


class _RecordingDropper(ConsumedPageDropper):
    """Records the ranges it would advise instead of issuing the syscalls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.advised: List[Tuple[int, int, int]] = []

    def _advise(self, shard_index: int, start: int, end: int) -> None:
        self.calls += 1
        self.bytes_advised += end - start
        self.advised.append((shard_index, start, end))


@unittest.skipIf(_MINCORE is None, "mincore(2) is not reachable through libc")
class TestGgufStreamPageCache(unittest.TestCase):
    """Proofs (a), (a2), (b), (c) -- residency and byte identity on a real file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.mkdtemp(prefix="gguf-fadvise-")
        cls.path = write_temp_file(cls._dir)
        cls.layout = extents(FILE_BYTES, CHUNK_BYTES)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._dir, ignore_errors=True)

    def _stream(
        self,
        *,
        drop: bool,
        fadvise_only: bool = False,
        batch_bytes: int = 8 << 20,
    ) -> Tuple[str, int, int]:
        """Read the whole file through a memmap the way the loader does.

        Returns ``(sha256 of the stream, resident pages at the end, page count)``.
        """
        drop_whole_file(self.path)
        array = np.memmap(self.path, mode="r")
        digest = hashlib.sha256()
        try:
            dropper: Optional[ConsumedPageDropper] = None
            fd: Optional[int] = None
            if drop:
                dropper = ConsumedPageDropper.for_readers(
                    [self.path],
                    [_FakeReader(array, self.layout)],
                    batch_bytes=batch_bytes,
                )
            if fadvise_only:
                fd = os.open(self.path, os.O_RDONLY)
            try:
                previous: Optional[int] = None
                for index, (offset, nbytes) in enumerate(self.layout):
                    # The copy out of the mapping -- exactly what
                    # torch.tensor(repacked_gguf_bytes(...)) does per tensor.
                    digest.update(np.array(array[offset : offset + nbytes]).tobytes())
                    if dropper is not None:
                        if previous is not None:
                            dropper.consume(0, previous)
                        previous = index
                    if fd is not None:
                        os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
                if dropper is not None and previous is not None:
                    dropper.consume(0, previous)
                    dropper.flush()
                resident = resident_pages(array)
            finally:
                if dropper is not None:
                    dropper.close()
                if fd is not None:
                    os.close(fd)
        finally:
            total_pages = (array.nbytes + PAGE - 1) // PAGE
            del array
        return digest.hexdigest(), resident, total_pages

    def test_a_advice_on_bounds_the_page_cache(self) -> None:
        """(a) ON: near zero resident. OFF: the whole file -- the boot-8 shape."""
        _, resident_on, total = self._stream(drop=True)
        _, resident_off, _ = self._stream(drop=False)

        # The tail the batch has not flushed yet plus the kernel's readahead
        # window. Generous, and still two orders of magnitude below "all of it".
        bound = (32 << 20) // PAGE
        self.assertLess(
            resident_on,
            bound,
            f"advice ON left {resident_on} of {total} pages resident "
            f"({resident_on * PAGE / (1 << 20):.1f} MiB); expected under "
            f"{bound * PAGE / (1 << 20):.1f} MiB",
        )
        self.assertGreater(
            resident_off,
            int(total * 0.8),
            f"advice OFF left only {resident_off} of {total} pages resident -- "
            "the accumulation this feature removes was not reproduced, so the "
            "ON measurement proves nothing",
        )

    def test_a2_fadvise_alone_does_not_free_a_mapped_range(self) -> None:
        """(a2) The naive fix, measured failing.

        ``invalidate_mapping_pages()`` skips pages that are still mapped into a
        page table, so fadvise-only over a live mmap frees close to nothing.
        This is why the dropper unmaps with madvise FIRST.
        """
        _, resident_fadvise_only, total = self._stream(drop=False, fadvise_only=True)
        self.assertGreater(
            resident_fadvise_only,
            total // 2,
            "posix_fadvise alone freed a mapped range on this kernel; the "
            "two-syscall rationale in gguf_shards.py needs re-checking",
        )

    def test_b_advice_does_not_change_the_streamed_bytes(self) -> None:
        """(b) Re-faulting an advised range returns the same bytes."""
        digest_on, _, _ = self._stream(drop=True)
        digest_off, _, _ = self._stream(drop=False)
        self.assertEqual(digest_on, digest_off)

        with open(self.path, "rb") as handle:
            reference = hashlib.sha256()
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                reference.update(block)
        self.assertEqual(digest_on, reference.hexdigest())

    def test_c_opt_out_restores_accumulation(self) -> None:
        """(c) SGLANG_GGUF_STREAM_DROP_CACHE=0 hands back a no-op dropper."""
        previous = os.environ.get("SGLANG_GGUF_STREAM_DROP_CACHE")
        try:
            os.environ["SGLANG_GGUF_STREAM_DROP_CACHE"] = "0"
            self.assertFalse(stream_drop_cache_enabled())
            array = np.memmap(self.path, mode="r")
            try:
                dropper = make_stream_page_dropper(
                    [self.path], [_FakeReader(array, self.layout)]
                )
                self.assertNotIsInstance(dropper, ConsumedPageDropper)
                drop_whole_file(self.path)
                for index, (offset, nbytes) in enumerate(self.layout):
                    _ = np.array(array[offset : offset + nbytes])
                    dropper.consume(0, index)
                dropper.close()
                resident = resident_pages(array)
                total = (array.nbytes + PAGE - 1) // PAGE
            finally:
                del array
            self.assertGreater(
                resident,
                int(total * 0.8),
                "the opt-out did not restore accumulation",
            )

            os.environ["SGLANG_GGUF_STREAM_DROP_CACHE"] = "1"
            self.assertTrue(stream_drop_cache_enabled())
        finally:
            if previous is None:
                os.environ.pop("SGLANG_GGUF_STREAM_DROP_CACHE", None)
            else:
                os.environ["SGLANG_GGUF_STREAM_DROP_CACHE"] = previous


class TestNeverAdviseAhead(unittest.TestCase):
    """(d) The bookkeeping contract, checked without touching a real file."""

    #: Deliberately not page-aligned and deliberately adjacent, so consecutive
    #: extents share pages -- the case a naive page-rounding gets wrong.
    LAYOUT = [
        (0, 3000),
        (3008, PAGE * 3 + 17),
        (3008 + PAGE * 3 + 32, PAGE * 7 + 5),
        (3008 + PAGE * 10 + 64, PAGE * 2),
        (3008 + PAGE * 12 + 96, PAGE * 5 + 1),
    ]
    SIZE = LAYOUT[-1][0] + LAYOUT[-1][1]

    def _dropper(self) -> _RecordingDropper:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"\0" * self.SIZE)
            path = handle.name
        self.addCleanup(os.unlink, path)
        dropper = _RecordingDropper([path], [None], batch_bytes=1)
        dropper.register(0, self.LAYOUT)
        return dropper

    def _assert_no_overlap_with_unconsumed(
        self, dropper: _RecordingDropper, consumed: List[int]
    ) -> None:
        outstanding = [
            self.LAYOUT[i] for i in range(len(self.LAYOUT)) if i not in consumed
        ]
        for _shard, start, end in dropper.advised:
            self.assertLess(start, end, "an empty range was advised")
            self.assertEqual(start % PAGE, 0, "advised start is not page aligned")
            self.assertEqual(end % PAGE, 0, "advised end is not page aligned")
            for offset, nbytes in outstanding:
                self.assertFalse(
                    start < offset + nbytes and offset < end,
                    f"advised [{start},{end}) overlaps unconsumed tensor "
                    f"[{offset},{offset + nbytes})",
                )

    def test_in_order_consumption_never_advises_ahead(self) -> None:
        dropper = self._dropper()
        consumed: List[int] = []
        for index in range(len(self.LAYOUT)):
            dropper.consume(0, index)
            consumed.append(index)
            self._assert_no_overlap_with_unconsumed(dropper, consumed)
        dropper.flush()
        self._assert_no_overlap_with_unconsumed(dropper, consumed)
        self.assertGreater(len(dropper.advised), 0, "nothing was advised at all")

    def test_out_of_order_consumption_stalls_instead_of_running_ahead(self) -> None:
        """Consuming tensor 4 before 0 must not release tensor 0's pages."""
        dropper = self._dropper()
        for index in (4, 3, 2):
            dropper.consume(0, index)
        dropper.flush()
        self.assertEqual(
            dropper.advised,
            [],
            "the watermark ran past an unconsumed tensor",
        )
        dropper.consume(0, 0)
        dropper.consume(0, 1)
        dropper.flush()
        self._assert_no_overlap_with_unconsumed(dropper, list(range(len(self.LAYOUT))))
        self.assertGreater(len(dropper.advised), 0)

    def test_shared_page_is_held_until_both_neighbours_are_consumed(self) -> None:
        """Tensors 0 and 1 share page 0; it may only go once both are done."""
        dropper = self._dropper()
        dropper.consume(0, 0)
        dropper.flush()
        for _shard, start, end in dropper.advised:
            self.assertGreaterEqual(
                start,
                PAGE,
                "page 0 was advised while tensor 1 still holds bytes in it",
            )
        self._assert_no_overlap_with_unconsumed(dropper, [0])

    def test_a_never_consumed_tensor_caps_the_watermark(self) -> None:
        dropper = self._dropper()
        for index in (0, 1, 3, 4):
            dropper.consume(0, index)
        dropper.flush()
        ceiling = (self.LAYOUT[2][0] // PAGE) * PAGE
        for _shard, _start, end in dropper.advised:
            self.assertLessEqual(end, ceiling)
        self._assert_no_overlap_with_unconsumed(dropper, [0, 1, 3, 4])


class TestStreamProgressLine(unittest.TestCase):
    """(e) The dropper's evidence must land IN the stream, not only at close().

    ``close()`` speaks for a stream that reached its end. Boot 9 (#391) died
    at layer 12 of 43, so its log carried no dropper line at all and said
    nothing about whether the feature had run -- ``mincore(2)`` had to answer
    that from outside the process. A periodic line puts the same evidence
    ahead of the step that fails, which is the only place it is worth
    anything on a boot that does not finish.
    """

    EXTENT = 256 << 10
    COUNT = 32
    STEP = 1 << 20

    def _dropper(self, progress_bytes: int) -> _RecordingDropper:
        size = self.EXTENT * self.COUNT
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.truncate(size)
            path = handle.name
        self.addCleanup(os.unlink, path)
        dropper = _RecordingDropper(
            [path], [None], batch_bytes=1, progress_bytes=progress_bytes
        )
        dropper.register(0, extents(size, self.EXTENT))
        return dropper

    def _consume_all(self, dropper: _RecordingDropper) -> None:
        for index in range(self.COUNT):
            dropper.consume(0, index)
        dropper.flush()

    def _progress_records(self, records) -> List[str]:
        return [r.getMessage() for r in records if "so far" in r.getMessage()]

    def test_progress_lines_land_before_close(self) -> None:
        dropper = self._dropper(self.STEP)
        with self.assertLogs(
            "sglang.srt.model_loader.gguf_shards", level="INFO"
        ) as captured:
            self._consume_all(dropper)
        lines = self._progress_records(captured.records)
        # 8 MiB released in 1 MiB steps -> one line per step, all of them
        # emitted while the stream was still running.
        self.assertEqual(
            len(lines),
            (self.EXTENT * self.COUNT) // self.STEP,
            captured.output,
        )
        self.assertIn(f"consumer at tensor {self.COUNT}/{self.COUNT}", lines[-1])
        self.assertIn("consumer at tensor 4/32", lines[0])
        dropper.close()

    def test_no_progress_line_before_the_first_step_is_reached(self) -> None:
        """Falsifier for "it just logs on every flush": a step larger than the
        whole stream must produce no in-stream line at all."""
        dropper = self._dropper(1 << 30)
        with self.assertLogs(
            "sglang.srt.model_loader.gguf_shards", level="INFO"
        ) as captured:
            self._consume_all(dropper)
            logging.getLogger("sglang.srt.model_loader.gguf_shards").info("sentinel")
        self.assertEqual(self._progress_records(captured.records), [])

    def test_the_advised_ranges_are_identical_with_and_without_the_line(
        self,
    ) -> None:
        """The line is a log and nothing else: same calls, same ranges, same
        order, same totals."""
        loud = self._dropper(self.STEP)
        quiet = self._dropper(0)
        with self.assertLogs("sglang.srt.model_loader.gguf_shards", level="INFO"):
            self._consume_all(loud)
        self._consume_all(quiet)
        strip = lambda d: [(s, a, b) for s, a, b in d.advised]  # noqa: E731
        self.assertEqual(strip(loud), strip(quiet))
        self.assertEqual(loud.bytes_advised, quiet.bytes_advised)
        self.assertEqual(loud.calls, quiet.calls)
        self.assertGreater(loud.bytes_advised, 0)


@unittest.skipIf(_MINCORE is None, "mincore(2) is not reachable through libc")
class TestRealIteratorReleasesCache(unittest.TestCase):
    """(a)+(b) again, but through the REAL loader entry point.

    Everything above exercises the dropper directly. This one drives
    ``weight_utils.gguf_quant_weights_iterator`` over a genuine (tiny) split
    GGUF, so the wiring -- the extent registration, the one-iteration-behind
    consume, the finally-close -- is executed rather than reasoned about.
    """

    ARCH = "deepseek4"
    TENSOR_MIB = 6
    TENSORS_PER_PART = 2
    PARTS = 4

    def setUp(self) -> None:
        import gguf

        from sglang.srt.model_loader import gguf_shards

        self.tmp = tempfile.mkdtemp(prefix="gguf-stream-e2e-")
        self.addCleanup(self._cleanup)
        self.names = [
            f"blk.{i}.attn_norm.weight"
            for i in range(self.PARTS * self.TENSORS_PER_PART)
        ]
        rows = (self.TENSOR_MIB << 20) // (4 * 64)
        writer = gguf.GGUFWriter(
            os.path.join(self.tmp, "toy.gguf"),
            self.ARCH,
            split_max_tensors=self.TENSORS_PER_PART,
            small_first_shard=True,
        )
        writer.add_block_count(len(self.names))
        rng = np.random.default_rng(1234)
        for name in self.names:
            writer.add_tensor(
                name, rng.random((rows, 64), dtype=np.float32), raw_dtype=None
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.parts = sorted(
            os.path.join(self.tmp, f)
            for f in os.listdir(self.tmp)
            if f.endswith(".gguf")
        )
        self.name_map = {name: f"model.{name}" for name in self.names}
        self._saved_batch = gguf_shards._DEFAULT_DROP_BATCH_BYTES
        gguf_shards._DEFAULT_DROP_BATCH_BYTES = 1 << 20

    def _cleanup(self) -> None:
        import shutil

        from sglang.srt.model_loader import gguf_shards

        gguf_shards._DEFAULT_DROP_BATCH_BYTES = self._saved_batch
        gguf_shards._RESOLVED_CACHE.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _residency(self, path: str) -> Tuple[int, int]:
        array = np.memmap(path, mode="r")
        try:
            total = (array.nbytes + PAGE - 1) // PAGE
            return resident_pages(array), total
        finally:
            del array

    def _run(self, enabled: bool) -> Tuple[List[Tuple[str, bytes]], int, int]:
        from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

        previous = os.environ.get("SGLANG_GGUF_STREAM_DROP_CACHE")
        os.environ["SGLANG_GGUF_STREAM_DROP_CACHE"] = "1" if enabled else "0"
        for part in self.parts:
            drop_whole_file(part)
        try:
            items: List[Tuple[str, bytes]] = []
            mid_resident = mid_total = 0
            # The second part is the first one that carries tensors; by the time
            # the stream reaches the last part it is long behind the consumer.
            watched = self.parts[1]
            stream = gguf_quant_weights_iterator(self.parts[0], self.name_map)
            for name, tensor in stream:
                items.append((name, tensor.numpy().tobytes()))
                if len(items) == len(self.names) - 1:
                    mid_resident, mid_total = self._residency(watched)
            del stream
            return items, mid_resident, mid_total
        finally:
            if previous is None:
                os.environ.pop("SGLANG_GGUF_STREAM_DROP_CACHE", None)
            else:
                os.environ["SGLANG_GGUF_STREAM_DROP_CACHE"] = previous

    def test_stream_is_byte_identical_and_the_cache_is_released(self) -> None:
        items_on, resident_on, total = self._run(enabled=True)
        items_off, resident_off, _ = self._run(enabled=False)

        self.assertEqual(len(items_on), len(self.names))
        self.assertEqual(items_on, items_off)

        self.assertLess(
            resident_on,
            max(4, total // 4),
            f"the real iterator left {resident_on} of {total} pages of an "
            "already-passed part resident",
        )
        self.assertGreater(
            resident_off,
            (total * 3) // 4,
            f"with the opt-out set only {resident_off} of {total} pages were "
            "resident -- the accumulation the ON case is compared against was "
            "not reproduced",
        )


class TestDropperWiring(unittest.TestCase):
    """The mapping the dropper picks up from a reader is the real one."""

    def test_for_readers_takes_the_mapping_base_address(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"\0" * (PAGE * 4))
            path = handle.name
        self.addCleanup(os.unlink, path)
        array = np.memmap(path, mode="r")
        try:
            self.assertIsInstance(array._mmap, mmap.mmap)
            dropper = ConsumedPageDropper.for_readers(
                [path], [_FakeReader(array, [(0, PAGE * 4)])]
            )
            # madvise takes an address, and only the numpy view knows it.
            self.assertEqual(dropper._addrs[0], array.ctypes.data)
            self.assertEqual(dropper._map_len[0], PAGE * 4)
            dropper.close()
        finally:
            del array

    def test_reader_without_a_mapping_degrades_to_fadvise(self) -> None:
        """A read()-based reader must not crash the dropper."""
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"\0" * (PAGE * 4))
            path = handle.name
        self.addCleanup(os.unlink, path)
        dropper = ConsumedPageDropper([path], [None], batch_bytes=1)
        dropper.register(0, [(0, PAGE * 2), (PAGE * 2, PAGE * 2)])
        dropper.consume(0, 0)
        dropper.consume(0, 1)
        dropper.close()
        self.assertGreater(dropper.bytes_advised, 0)


if __name__ == "__main__":
    unittest.main()
