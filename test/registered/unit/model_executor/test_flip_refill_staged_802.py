"""#802: the file-backed flip refill READS the image; it does not fault it.

THE DEFECT. ``arena_refill`` moved the file-backed host image into the arena
with one ``dst.copy_`` straight off the mapping. A file-backed image is a
disk-backed ``mmap``, so that copy takes ONE SYNCHRONOUS MAJOR FAULT PER
4 KiB PAGE. Measured on this rig 2026-08-22 against the production-sized PP0
image (16 699 408 904 B) on the same ZFS pool serving uses::

    mmap_copy       14 377 ms   4 077 015 major faults   1 108 MiB/s
    pread_staged_8   6 138 ms          20 major faults   2 595 MiB/s

The pool writes at ~3 500 MiB/s, so the mapping path was leaving most of the
device on the floor. It was also link-independent -- all three ranks
converged on the same rate despite PCIe links differing by 1.80x -- which is
what says no DMA took part on either side.

WHY NOT AN madvise HINT. Because on this filesystem it does nothing, and
that is measured rather than assumed. ``MADV_WILLNEED`` over the cold
mapping returned 0 and populated NOTHING: 12 564 ms against a 12 572 ms
baseline, 4 077 052 faults against 4 077 045, and ``mincore`` residency 0.0
after the call. Per-chunk ``MADV_SEQUENTIAL`` was a 15.6x REGRESSION
(196 200 ms). A mechanism whose actuator the filesystem ignores is the #738
defect class, so this fix does not rely on one.

WHAT IS PINNED HERE, both directions each:

* the DEFAULT pinned-image path never reaches the staged code -- an image
  that was not registered routes to the original ``copy_``;
* a file-backed image DOES get a read fd, kept across the unlink;
* the registry refuses an address whose size does not match, rather than
  reading some other file into the arena;
* the staged path moves BYTE-IDENTICAL content (and the existing checksum
  would catch it if it did not);
* the env switch actually switches, so the A/B runs on one binary;
* on metal, the staged path's major-fault count COLLAPSES -- the guard that
  would have caught an inert fix.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _majflt() -> int:
    with open("/proc/self/stat") as fh:
        return int(fh.read().split()[11])


#: ``asm-generic/mman-common.h``. CPython only exposes the MADV_* constants its
#: build host had headers for, and MADV_PAGEOUT is missing from common builds,
#: so the number is spelled out -- the same reason #408 spells it out.
_MADV_PAGEOUT = 21
_PAGE = 4096


def _libc():
    import ctypes

    lib = ctypes.CDLL("libc.so.6", use_errno=True)
    lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    return lib


def _resident(image: torch.Tensor) -> float:
    """Fraction of the mapping still in the page cache, via mincore."""
    import ctypes

    lib = _libc()
    n = int(image.numel())
    npages = (n + _PAGE - 1) // _PAGE
    vec = ctypes.create_string_buffer(npages)
    if lib.mincore(ctypes.c_void_p(image.data_ptr()), n, vec) != 0:
        return 1.0
    return sum(1 for x in vec.raw[:npages] if x & 1) / npages


def _make_layout(nbytes: int):
    """A one-slot layout of ``nbytes``, which is all the refill needs."""
    from sglang.srt.model_executor.weights_arena import ArenaLayout, ArenaSlot

    slot = ArenaSlot(
        name="w",
        offset=0,
        nbytes=nbytes,
        dtype=torch.uint8,
        shape=(nbytes,),
        stride=(1,),
    )
    return ArenaLayout(slots=[slot], total_bytes=nbytes, aliases=[])


class TestStagedRefillRouting(CustomTestCase):
    """Routing, on CPU: who gets the staged path and who must not."""

    def tearDown(self):
        weights_arena._FILE_BACKED_IMAGES.clear()

    def test_unregistered_image_is_not_file_backed(self):
        """The DEFAULT pinned path must never reach the staged code.

        Can-fail: a ``_file_backed_meta`` that answered for any tensor would
        route the default images through a read path they have no fd for.
        """
        image = torch.zeros(4096, dtype=torch.uint8)
        self.assertIsNone(weights_arena._file_backed_meta(image))

    def test_registered_image_is_found(self):
        """The other direction: a registered image IS found."""
        image = torch.zeros(4096, dtype=torch.uint8)
        weights_arena._register_file_backed_image(image, fd=-1, total=4096, path="/x")
        meta = weights_arena._file_backed_meta(image)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.nbytes, 4096)

    def test_size_mismatch_is_refused(self):
        """An address whose recorded size disagrees is NOT used.

        Reading the wrong file would be caught by the arena checksum, but only
        after the arena had already been overwritten -- too late to be a
        diagnosis. Can-fail: drop the nbytes check and this returns the meta.
        """
        image = torch.zeros(4096, dtype=torch.uint8)
        weights_arena._register_file_backed_image(image, fd=-1, total=8192, path="/x")
        self.assertIsNone(weights_arena._file_backed_meta(image))

    def test_env_switch_moves_both_ways(self):
        """The A/B has to be runnable on ONE binary, so the switch must switch."""
        with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(True):
            self.assertTrue(weights_arena._staged_refill_enabled())
        with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(False):
            self.assertFalse(weights_arena._staged_refill_enabled())

    def test_file_backed_image_keeps_a_read_fd(self):
        """The fd must survive the unlink -- that is the whole enabler.

        Can-fail: without the ``os.open`` before ``os.unlink`` the registry
        stays empty and the refill has nothing to read from.
        """
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
                with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(d):
                    total = 1 << 20
                    image = weights_arena._file_backed_image(total)
            meta = weights_arena._file_backed_meta(image)
            self.assertIsNotNone(meta, "no read fd was kept for the image")
            self.assertEqual(meta.nbytes, total)
            # The fd reads the SAME bytes the mapping shows.
            image[:8] = torch.arange(8, dtype=torch.uint8)
            got = os.pread(meta.fd, 8, 0)
            self.assertEqual(bytes(got), bytes(range(8)))


@unittest.skipUnless(torch.cuda.is_available(), "staged refill stages through CUDA")
class TestStagedRefillOnDevice(CustomTestCase):
    """Content and fault behaviour, on a real device."""

    def tearDown(self):
        weights_arena._FILE_BACKED_IMAGES.clear()
        if weights_arena._staged_pool is not None:
            weights_arena._staged_pool.close()
            weights_arena._staged_pool = None

    def _image(self, d, payload: bytes):
        total = len(payload) + weights_arena._CHECKSUM_BYTES
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(d):
                image = weights_arena._file_backed_image(total)
        image[: len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        csum = weights_arena.uint8_checksum(image[: len(payload)])
        image[len(payload) :] = torch.tensor([csum], dtype=torch.int64).view(
            torch.uint8
        )
        return image, _make_layout(len(payload))

    def _cold_image(self, d, payload: bytes):
        """The same image, but with a mapping NOTHING has touched yet.

        Writing through the mapping (``_image`` above) leaves every page
        resident, and no advisory call can get them back out on this
        filesystem -- MADV_PAGEOUT returns 0 and evicts 0.4% even after an
        msync, the same inertness MADV_WILLNEED and fadvise show here.

        So the file is written with ordinary ``write`` instead, which on this
        pool lands in the ARC and does NOT populate the page cache, and only
        then mapped. That is also the honest shape of production: the image is
        written once at boot and read back at a flip much later.
        """
        total = len(payload) + weights_arena._CHECKSUM_BYTES
        body = bytearray(payload)
        tmp = torch.frombuffer(body, dtype=torch.uint8)
        csum = weights_arena.uint8_checksum(tmp)
        blob = bytes(body) + int(csum).to_bytes(8, "little", signed=True)
        path = os.path.join(d, "cold-image.img")
        with open(path, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        image = torch.from_file(path, shared=True, size=total, dtype=torch.uint8)
        fd = os.open(path, os.O_RDONLY)
        try:
            fd_direct = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError:
            fd_direct = None
        weights_arena._register_file_backed_image(image, fd, total, path, fd_direct)
        return image, _make_layout(len(payload))

    def test_staged_refill_is_byte_identical(self):
        """Same bytes as the mapping path, or the checksum raises."""
        payload = os.urandom(8 << 20)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, payload)
            arena = torch.zeros(len(payload), dtype=torch.uint8, device="cuda")
            with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(True):
                weights_arena.arena_refill(arena, layout, image)
            self.assertEqual(bytes(arena.cpu().numpy().tobytes()), payload)

    def test_disabled_env_still_refills_correctly(self):
        """The comparand arm must still be CORRECT, not merely slow."""
        payload = os.urandom(4 << 20)
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            image, layout = self._image(d, payload)
            arena = torch.zeros(len(payload), dtype=torch.uint8, device="cuda")
            with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(False):
                weights_arena.arena_refill(arena, layout, image)
            self.assertEqual(bytes(arena.cpu().numpy().tobytes()), payload)

    def test_staged_refill_does_not_fault_the_mapping(self):
        """THE GUARD THAT CATCHES AN INERT FIX.

        The point of #802 is not that the clock got smaller -- it is that the
        faults became reads. A fix that quietly fell back to the mapping would
        still pass every content assertion above and would be caught only
        here. Both directions are asserted: the disabled arm MUST fault, the
        enabled arm must not.
        """
        payload = os.urandom(64 << 20)
        pages = len(payload) // 4096
        with tempfile.TemporaryDirectory(dir="/spinning/flip_images_755") as d:
            image, layout = self._cold_image(d, payload)
            arena = torch.zeros(len(payload), dtype=torch.uint8, device="cuda")

            # THE TEST IS ONLY A GUARD IF THE MAPPING IS COLD -- a fallback to
            # a WARM mapping shows ~0 faults too, so the assertion would pass
            # vacuously and could never catch an inert fix.
            if _resident(image) > 0.1:
                self.skipTest(
                    "the mapping is already resident, so a warm fallback would "
                    "pass this test without doing anything -- refusing to "
                    "report a guard that cannot fail"
                )

            # Direction 1: the comparand MUST fault. If it does not, the image
            # was not actually cold and nothing below means anything.
            comparand = torch.zeros(len(payload), dtype=torch.uint8, device="cuda")
            before = _majflt()
            with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(False):
                weights_arena.arena_refill(comparand, layout, image)
            mapped_faults = _majflt() - before
            self.assertGreater(
                mapped_faults,
                pages // 2,
                f"the mapping path took only {mapped_faults} major faults "
                f"over {pages} pages -- the image was not actually cold, so "
                f"this test proves nothing",
            )

            # Direction 2: the staged path must NOT fault. It gets its OWN
            # cold image, because direction 1 just faulted this one in and no
            # advisory call on this filesystem can undo that.
            os.makedirs(os.path.join(d, "second"), exist_ok=True)
            image, layout = self._cold_image(os.path.join(d, "second"), payload)
            if _resident(image) > 0.1:
                self.skipTest("second image did not come up cold")
            before = _majflt()
            with envs.SGLANG_PHASE_FLIP_REFILL_STAGED.override(True):
                weights_arena.arena_refill(arena, layout, image)
            staged_faults = _majflt() - before

            self.assertLess(
                staged_faults,
                pages // 10,
                f"staged refill took {staged_faults} major faults over "
                f"{pages} pages ({mapped_faults} on the mapping path) -- it "
                f"is still walking the mapping",
            )
            self.assertEqual(bytes(arena.cpu().numpy().tobytes()), payload)


if __name__ == "__main__":
    unittest.main()
