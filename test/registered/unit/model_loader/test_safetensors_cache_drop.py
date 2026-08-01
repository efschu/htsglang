# SPDX-License-Identifier: Apache-2.0
"""#408: the safetensors page-cache drop must actually work on ZFS.

``weight_utils._drop_file_cache_after_load`` (opt-in via
``--weight-loader-drop-cache-after-load``, used to avoid CPU OOM in RL) used
to be ``posix_fadvise(POSIX_FADV_DONTNEED)`` alone. Measured on this rig's
ZFS pool (what ``/spinning`` is): a no-op. ``invalidate_mapping_pages()``
skips a page that is still mapped, and here it does not free it even after
the mapping goes away either -- the same ZFS finding the GGUF stream fix
(#391, ``gguf_shards.py``) already made and built a working ladder for:
``madvise(MADV_PAGEOUT)`` first, ``MADV_DONTNEED`` + ``posix_fadvise`` as a
fallback. This reuses that ladder (``gguf_shards.drop_page_cache_range``).

The safetensors path differs from the GGUF stream in one respect that
simplifies it: by the time the caller is ready to drop a shard's cache, the
whole file has already been read and copied out (there is no consumer still
reading behind it), so there is no watermark to keep -- one whole-file drop
after the fact is correct. It differs in a second respect that does NOT
simplify it: ``safetensors.safe_open``'s internal mmap is not exposed as an
object the way ``gguf.GGUFReader.data`` is, and a separate, freshly-opened
mapping of the same file created *after* the fact was tried and measured to
NOT work here -- with the original mapping still referenced elsewhere (which
nothing guarantees against), ``MADV_PAGEOUT`` through an unrelated second
mapping left the page cache untouched. What does work, measured: capturing
each tensor's own ``data_ptr()`` (its own live, already-faulted mapping
address) while the ``with safe_open(...)`` block is still open, and running
the ladder on those addresses before the block exits.

Four proofs, mirroring ``test_gguf_stream_page_cache.py``'s pattern, all
hermetic and able to fail:

a) **bounded vs accumulating** -- after a real safetensors write + read
   (mmap'd, byte-for-byte touched, the way ``safe_open`` + ``get_tensor``
   read weights), the fixed drop leaves near-zero resident pages; the old
   fadvise-only mechanism reproduces the ZFS no-op (page count == the whole
   file).
b) **byte identity** -- the tensor read back is identical whether or not the
   drop ran.
c) **opt-out** -- ``drop_cache_after_load=False`` through the real
   ``safetensors_weights_iterator`` leaves the accumulation from (a) intact;
   nothing about the flag's own gating changed.
d) **wiring** -- the real ``safetensors_weights_iterator`` entry point (not
   just the private helper) frees the cache and yields the same tensors as
   with the drop off.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import tempfile
import unittest
from typing import Tuple

import numpy as np
import safetensors.torch
import torch

from sglang.srt.model_loader.gguf_shards import _PAGE_SIZE as PAGE
from sglang.srt.model_loader.weight_utils import (
    _drop_file_cache_after_load,
    safetensors_weights_iterator,
)

#: Big enough that the residency signal is unambiguous, small enough to
#: write, read and drop in well under a second even on a busy box.
TENSOR_ELEMENTS = 32 * 1024 * 1024  # 128 MiB of float32


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


def resident_pages(path: str) -> Tuple[int, int]:
    """How many pages of ``path`` are resident in the page cache right now.

    Uses a fresh, independent mapping (not whatever mapping the caller used
    to read the file) so this is a page-cache-level measurement, not an
    artifact of one particular mapping's page table.
    """
    array = np.memmap(path, mode="r")
    try:
        length = array.nbytes
        npages = (length + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(npages)
        rc = _MINCORE(ctypes.c_void_p(array.ctypes.data), ctypes.c_size_t(length), vec)
        if rc != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        return sum(1 for byte in vec.raw[:npages] if byte & 1), npages
    finally:
        del array


def drop_whole_file(path: str) -> None:
    """Start every measurement from a cold page cache for this file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def write_safetensors_file(
    directory: str, seed: int = 1234
) -> Tuple[str, torch.Tensor]:
    path = os.path.join(directory, "shard-00001-of-00001.safetensors")
    rng = np.random.default_rng(seed)
    tensor = torch.from_numpy(
        rng.random((TENSOR_ELEMENTS,), dtype=np.float32).astype(np.float32)
    ).clone()
    safetensors.torch.save_file({"weight": tensor}, path)
    return path, tensor


@unittest.skipIf(_MINCORE is None, "mincore(2) is not reachable through libc")
class TestSafetensorsCacheDrop(unittest.TestCase):
    """Proofs (a) and (b) -- residency and byte identity, on the private helper."""

    @classmethod
    def setUpClass(cls) -> None:
        # A real ZFS-backed directory, same pool the GGUF fix measured
        # against, rather than relying on whatever /tmp happens to be.
        base = "/spinning" if os.path.isdir("/spinning") else tempfile.gettempdir()
        cls._dir = tempfile.mkdtemp(prefix="st-cache-drop-", dir=base)
        cls.path, cls.reference_tensor = write_safetensors_file(cls._dir)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._dir, ignore_errors=True)

    def _load_touch_and_drop(self, *, fixed: bool) -> torch.Tensor:
        """Read the file through safetensors exactly like the real loader
        does (``safe_open`` + ``get_tensor``, bytes actually copied out via
        ``.numpy().tobytes()`` -- what ``param.data.copy_()`` does downstream
        and what actually faults the pages in), then drop its cache.

        ``fixed=True`` mirrors the real call site in
        ``weight_utils.safetensors_weights_iterator``: extents are captured
        per tensor and the drop runs *inside* the ``with`` block, while the
        mapping they point into is still guaranteed alive.
        ``fixed=False`` reproduces the pre-#408 mechanism (plain fadvise,
        i.e. calling the same helper with no extents at all).
        """
        drop_whole_file(self.path)
        with safetensors.safe_open(self.path, framework="pt", device="cpu") as f:
            extents = []
            result = None
            for name in f.keys():
                t = f.get_tensor(name)
                _ = t.numpy().tobytes()
                if fixed:
                    extents.append((t.data_ptr(), t.numel() * t.element_size()))
                result = t
            if fixed:
                _drop_file_cache_after_load(self.path, extents)
        if not fixed:
            _drop_file_cache_after_load(self.path)
        return result

    def test_a_fixed_bounds_the_page_cache_fadvise_only_does_not(self) -> None:
        """(a) Fixed ladder: near zero resident. Old fadvise-only: the whole
        file stays resident -- the measured ZFS no-op this replaces."""
        self._load_touch_and_drop(fixed=False)
        resident_fadvise_only, total = resident_pages(self.path)

        self._load_touch_and_drop(fixed=True)
        resident_fixed, total_fixed = resident_pages(self.path)

        self.assertEqual(total, total_fixed)
        self.assertGreater(
            resident_fadvise_only,
            int(total * 0.9),
            f"fadvise-only left only {resident_fadvise_only} of {total} pages "
            "resident -- the ZFS no-op this test exists to reproduce did not "
            "reproduce, so the fixed-ladder comparison below proves nothing",
        )
        bound = (16 << 20) // PAGE
        self.assertLess(
            resident_fixed,
            bound,
            f"the fixed ladder left {resident_fixed} of {total_fixed} pages "
            f"resident ({resident_fixed * PAGE / (1 << 20):.1f} MiB); expected "
            f"under {bound * PAGE / (1 << 20):.1f} MiB",
        )

    def test_b_drop_does_not_change_the_loaded_bytes(self) -> None:
        """(b) The tensor read back is identical whether or not the drop ran,
        and a later re-fault of an advised range still returns the same
        bytes -- the mapping is read-only ``MAP_SHARED`` over an unmodified
        file, so a later access just re-reads it from disk."""
        tensor_fadvise_only = self._load_touch_and_drop(fixed=False)
        self.assertTrue(torch.equal(tensor_fadvise_only, self.reference_tensor))

        tensor_after_fixed_drop = self._load_touch_and_drop(fixed=True)
        self.assertTrue(torch.equal(tensor_after_fixed_drop, self.reference_tensor))

        tensor_refault = self._load_touch_and_drop(fixed=True)
        self.assertTrue(torch.equal(tensor_refault, self.reference_tensor))


@unittest.skipIf(_MINCORE is None, "mincore(2) is not reachable through libc")
class TestRealIteratorDropsCache(unittest.TestCase):
    """(c)+(d), through the REAL loader entry point.

    ``TestSafetensorsCacheDrop`` exercises the private helper directly; this
    drives ``weight_utils.safetensors_weights_iterator`` end to end, so the
    wiring -- the opt-in flag, the per-file call site -- is executed rather
    than reasoned about.
    """

    @classmethod
    def setUpClass(cls) -> None:
        base = "/spinning" if os.path.isdir("/spinning") else tempfile.gettempdir()
        cls._dir = tempfile.mkdtemp(prefix="st-cache-drop-iter-", dir=base)
        cls.path, cls.reference_tensor = write_safetensors_file(cls._dir, seed=99)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._dir, ignore_errors=True)

    def _run(self, *, drop_cache_after_load: bool):
        drop_whole_file(self.path)
        items = []
        for name, tensor in safetensors_weights_iterator(
            [self.path], drop_cache_after_load=drop_cache_after_load
        ):
            # Actually consume the bytes, the way ``default_weight_loader``'s
            # ``param.data.copy_(loaded_weight)`` does downstream -- an
            # untouched, zero-copy mmap view never faults a page in, so
            # skipping this would make the residency signal below meaningless
            # (nothing would ever have been resident in the first place).
            items.append((name, tensor.numpy().tobytes()))
        resident, total = resident_pages(self.path)
        return items, resident, total

    def test_d_iterator_is_byte_identical_and_frees_the_cache(self) -> None:
        items_on, resident_on, total = self._run(drop_cache_after_load=True)
        items_off, resident_off, _ = self._run(drop_cache_after_load=False)

        self.assertEqual(len(items_on), 1)
        self.assertEqual(items_on, items_off)
        name_on, bytes_on = items_on[0]
        self.assertEqual(name_on, "weight")
        self.assertEqual(bytes_on, self.reference_tensor.numpy().tobytes())

        self.assertLess(
            resident_on,
            max(4, total // 4),
            f"drop_cache_after_load=True left {resident_on} of {total} pages "
            "resident through the real iterator",
        )
        self.assertGreater(
            resident_off,
            (total * 3) // 4,
            f"(c) opt-out: drop_cache_after_load=False left only {resident_off} "
            f"of {total} pages resident -- the accumulation the ON case is "
            "compared against was not reproduced",
        )

    def test_c_opt_out_leaves_accumulation_unchanged(self) -> None:
        """(c) ``drop_cache_after_load=False`` behaves exactly as before #408:
        the flag's own gating is untouched by the mechanism change."""
        _, resident_off, total = self._run(drop_cache_after_load=False)
        self.assertGreater(
            resident_off,
            int(total * 0.9),
            "the opt-out no longer leaves the checkpoint's pages accumulated",
        )


if __name__ == "__main__":
    unittest.main()
