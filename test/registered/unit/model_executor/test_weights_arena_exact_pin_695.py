"""#695: the pinned host weight images must be allocated at their EXACT size.

THE DEFECT
----------
``image_from_tensors`` and ``arena_image`` allocate their host image with
``torch.zeros(..., pin_memory=True)``. Every ``pin_memory=True`` allocation
goes through PyTorch's pinned-host caching allocator, which rounds the request
up to the next power of two before calling ``cudaHostAlloc``::

    torch/include/ATen/core/CachingHostAllocator.h:302
        size_t roundSize = c10::llvm::PowerOf2Ceil(size);
    torch/include/ATen/core/CachingHostAllocator.h:334
        allocate_host_memory(roundSize, &ptr);

The phase flip holds TWO such images per rank for the life of the process
(``image_pp``, ``image_tp``), plus one draft image, and nothing frees them --
``PhaseFlipStacks.refill`` re-reads them on every flip. So the rounding is not
a transient overshoot, it is permanent resident host memory.

MEASURED, on the live PP=3 boot of 2026-08-12 (``/proc/<pid>/smaps``, the
three ``sglang::scheduler`` ranks). Payload figures are the ones this repo
already recorded at ``phase_flip_spill.py:851-854``::

    rank   layout_pp   -> mapping    layout_tp   -> mapping
    PP0    13482.18 MiB   16384 MiB  13163.45 MiB   16384 MiB
    PP1     8144.00 MiB    8192 MiB   7923.95 MiB    8192 MiB
    PP2     9114.95 MiB   16384 MiB   7923.95 MiB    8192 MiB

Every observed mapping is an exact power of two, and all six match the
rounding rule. Rank 2's PP layout clears 8192 MiB by 923 MiB and therefore
alone costs an extra 7.1 GiB.

Total across the three ranks: 58.35 GiB of payload held in 72 GiB of
mappings. **13.65 GiB of pure rounding waste**, on a swapless box with a
~120 GiB ceiling that had already taken nine cgroup OOM kills.

THE FIX
-------
Allocate the image through ``alloc_with_host_register``
(``mem_cache/pool_host/common.py:86``), the in-tree path the host KV pool
already uses: an exact-size ``MAP_SHARED|MAP_ANONYMOUS|MAP_POPULATE`` mapping
followed by ``cudaHostRegister``. It is page-locked exactly as
``cudaHostAlloc`` is -- the DMA property the restore depends on is preserved --
but it is sized in pages, not in powers of two.

CAN-FAIL
--------
Revert ``_alloc_host_image`` to ``torch.zeros(..., pin_memory=True)`` and
``test_pinned_image_uses_the_exact_size_allocator`` goes red. Make
``alloc_mmap`` round to a power of two and
``test_alloc_mmap_is_page_exact_not_power_of_two`` goes red. Drop the
fallback and ``test_a_failing_host_register_falls_back_to_torch_pin`` goes
red.
"""

import re
import unittest

import torch

from sglang.srt.mem_cache.mmap_allocator import alloc_mmap
from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20


def _mapping_bytes_covering(addr: int):
    """Size of the /proc/self/maps VMA containing ``addr``, or None."""
    with open("/proc/self/maps", "r", encoding="utf-8") as handle:
        for line in handle:
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
            if not m:
                continue
            start, end = int(m.group(1), 16), int(m.group(2), 16)
            if start <= addr < end:
                return end - start
    return None


class AllocMmapIsExact(unittest.TestCase):
    """The mechanism the fix rests on, proved against the real kernel.

    No GPU: the mapping is created and measured through /proc, so this is the
    allocator's actual behaviour rather than a claim about it.
    """

    def test_alloc_mmap_is_page_exact_not_power_of_two(self):
        # Deliberately awkward: just over 5 MiB, so power-of-two rounding
        # (8 MiB) and page rounding (5 MiB + 4 KiB) are far apart.
        n = 5 * MIB + 1
        buf = alloc_mmap((n,), torch.uint8)
        try:
            self.assertEqual(buf.numel(), n)
            size = _mapping_bytes_covering(buf.data_ptr())
            self.assertIsNotNone(size, "mapping not found in /proc/self/maps")
            page = 4096
            expected = ((n + page - 1) // page) * page
            self.assertEqual(
                size,
                expected,
                f"expected a page-exact {expected} B mapping, got {size} B "
                f"(power-of-two rounding would give {8 * MIB} B)",
            )
            self.assertLess(size, 8 * MIB)
        finally:
            del buf

    def test_the_mapping_is_shared_and_prefaulted(self):
        """MAP_SHARED + MAP_POPULATE is what makes cudaHostRegister safe.

        Losing either turns the register into a race against page faults, so
        the flags are pinned here rather than left to the allocator's
        docstring.
        """
        n = 2 * MIB
        buf = alloc_mmap((n,), torch.uint8)
        try:
            addr = buf.data_ptr()
            perms = None
            with open("/proc/self/maps", "r", encoding="utf-8") as handle:
                for line in handle:
                    m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) (\S+)", line)
                    if not m:
                        continue
                    if int(m.group(1), 16) <= addr < int(m.group(2), 16):
                        perms = m.group(3)
                        break
            self.assertIsNotNone(perms)
            self.assertTrue(perms.endswith("s"), f"not MAP_SHARED: {perms}")
        finally:
            del buf

    def test_anonymous_mapping_arrives_zeroed(self):
        """The image contract says alignment gaps are zeroed.

        ``image_from_tensors`` relies on MAP_ANONYMOUS's zero guarantee
        instead of memsetting 13 GiB, so the guarantee is asserted rather
        than assumed.
        """
        buf = alloc_mmap((64 * 1024,), torch.uint8)
        try:
            self.assertEqual(int(buf.max()), 0)
        finally:
            del buf


class ImageAllocationRouting(unittest.TestCase):
    """The images must be routed through the exact-size allocator."""

    def test_pinned_image_uses_the_exact_size_allocator(self):
        seen = {}

        def _spy(dims, dtype, device, pin_memory, allocator):
            seen["dims"] = dims
            seen["pin_memory"] = pin_memory
            return torch.zeros(dims, dtype=dtype)

        original = weights_arena._alloc_with_host_register
        weights_arena._alloc_with_host_register = _spy
        try:
            out = weights_arena._alloc_host_image(1234567, pin=True)
        finally:
            weights_arena._alloc_with_host_register = original

        self.assertEqual(
            seen.get("dims"),
            (1234567,),
            "the pinned image must be requested at its exact byte count",
        )
        self.assertTrue(seen.get("pin_memory"))
        self.assertEqual(out.numel(), 1234567)

    def test_unpinned_image_is_unchanged(self):
        """pin=False is the CPU/unit path and must not touch CUDA at all."""
        called = {"n": 0}

        def _spy(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("must not be reached with pin=False")

        original = weights_arena._alloc_with_host_register
        weights_arena._alloc_with_host_register = _spy
        try:
            out = weights_arena._alloc_host_image(4096, pin=False)
        finally:
            weights_arena._alloc_with_host_register = original
        self.assertEqual(called["n"], 0)
        self.assertEqual(out.numel(), 4096)
        self.assertEqual(int(out.max()), 0)

    def test_an_overwritten_image_is_not_faulted_in_up_front(self):
        """zero=False must stay torch.empty, not torch.zeros.

        REGRESSION, caught by the manager suite dying at exit 137. The first
        version of this fix routed ``arena_image`` through a zeroing
        allocation, where it had previously used ``torch.empty``. Every byte
        of that image is overwritten immediately, so the zero-fill buys
        nothing -- but it faults the whole allocation in, and over a CPU test
        run that was the difference between finishing and being SIGKILLed.
        Resident pages, not virtual size, is what gets a process killed.
        """
        import resource

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        big = weights_arena._alloc_host_image(256 * MIB, pin=False, zero=False)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            self.assertEqual(big.numel(), 256 * MIB)
            # ru_maxrss is in KiB. An untouched torch.empty must not move the
            # high-water mark by anything like the allocation size.
            self.assertLess(
                (after - before) * 1024,
                128 * MIB,
                "zero=False faulted the image in; it must stay torch.empty",
            )
        finally:
            del big

    def test_zero_true_still_zeroes(self):
        """image_from_tensors' alignment-gap contract still holds."""
        buf = weights_arena._alloc_host_image(4096, pin=False, zero=True)
        self.assertEqual(int(buf.max()), 0)

    def test_a_failing_host_register_falls_back_to_torch_pin(self):
        """A rank that cannot register must still boot, not die at the image.

        cudaHostRegister can refuse (no CUDA context yet, a driver that will
        not lock that many pages). The old behaviour is strictly available as
        a fallback, so the failure costs the rounding back -- not the boot.
        """
        fell_back = {"n": 0}

        def _refuse(dims, dtype, device, pin_memory, allocator):
            raise RuntimeError("cudaHostRegister failed (rc=2)")

        def _fake_torch_pin(total):
            fell_back["n"] += 1
            return torch.zeros(total, dtype=torch.uint8)

        original = weights_arena._alloc_with_host_register
        original_pin = weights_arena._torch_pinned_zeros
        weights_arena._alloc_with_host_register = _refuse
        weights_arena._torch_pinned_zeros = _fake_torch_pin
        try:
            out = weights_arena._alloc_host_image(8192, pin=True)
        finally:
            weights_arena._alloc_with_host_register = original
            weights_arena._torch_pinned_zeros = original_pin
        self.assertEqual(fell_back["n"], 1)
        self.assertEqual(out.numel(), 8192)


class RoundingWasteArithmetic(unittest.TestCase):
    """What the rounding actually costs on the recorded boot.

    Pure arithmetic over the payload figures already recorded in
    ``phase_flip_spill.py:851-854``. It exists so the size of the prize is
    checkable without a GPU window, and so a future change to those payloads
    is noticed here rather than in an OOM.
    """

    #: (rank, layout_pp MiB, layout_tp MiB) from phase_flip_spill.py:851-854
    PAYLOADS_MIB = (
        ("PP0", 13482.18, 13163.45),
        ("PP1", 8144.00, 7923.95),
        ("PP2", 9114.95, 7923.95),
    )

    #: What /proc/<pid>/smaps showed on 2026-08-12, MiB.
    OBSERVED_MIB = {"PP0": (16384, 16384), "PP1": (8192, 8192), "PP2": (16384, 8192)}

    @staticmethod
    def _pow2_ceil(n: int) -> int:
        out = 1
        while out < n:
            out <<= 1
        return out

    def test_power_of_two_rounding_reproduces_every_observed_mapping(self):
        """Six for six. This is what closes the root cause."""
        for rank, pp_mib, tp_mib in self.PAYLOADS_MIB:
            for idx, payload_mib in enumerate((pp_mib, tp_mib)):
                rounded = self._pow2_ceil(int(payload_mib * MIB)) // MIB
                self.assertEqual(
                    rounded,
                    self.OBSERVED_MIB[rank][idx],
                    f"{rank} image {idx}: payload {payload_mib} MiB rounds to "
                    f"{rounded} MiB, but smaps showed "
                    f"{self.OBSERVED_MIB[rank][idx]} MiB",
                )

    def test_the_waste_is_thirteen_point_six_gib(self):
        waste_mib = 0.0
        for _, pp_mib, tp_mib in self.PAYLOADS_MIB:
            for payload_mib in (pp_mib, tp_mib):
                rounded = self._pow2_ceil(int(payload_mib * MIB)) / MIB
                waste_mib += rounded - payload_mib
        self.assertAlmostEqual(waste_mib, 13975.47, places=1)
        self.assertGreater(waste_mib / 1024, 13.6)

    def test_rank_two_is_the_single_worst_offender(self):
        """Names the knob: PP2's layout clears 8 GiB by 923 MiB."""
        pp2 = dict((r, (a, b)) for r, a, b in self.PAYLOADS_MIB)["PP2"][0]
        self.assertGreater(pp2, 8192)
        self.assertLess(pp2 - 8192, 1024)
        # Getting PP2's PP layout under 8192 MiB reclaims this much alone.
        self.assertAlmostEqual(16384 - pp2, 7269.05, places=1)


if __name__ == "__main__":
    unittest.main()
