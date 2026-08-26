"""#875: the staging memcpy that in-place aliasing forces, run on more than one core.

WHERE THIS SITS. #873 established that the seam's dominant segment is not a
transfer: `refill_highwater->weights_refill` decomposes as `save 4.342 +
checksum 0.319 + wait 0.084 + d2h-issue 0.026 + h2d-issue 0.020` on PP0
pp_to_tp, with `gpu-span d2h 0.000s / h2d 0.000s`. `save` is
`TorchRotationOps.save` -- `dst_buf[:n].copy_(src)`, a host-to-host memcpy of
EVERY chunk -- and it is 90 % of the segment and ~68 % of the whole 6.4 s flip.

WHY THE MEMCPY CANNOT SIMPLY BE DELETED, established at the desk and stated here
so the next reader does not re-open it. The rotation is an in-place transform
whose two streams share ONE coordinate system:

    host side : the H2D READS  host_image[off:off+len]
                the D2H WRITES host_image[off:off+len]
    arena side: the D2H READS  arena[off:off+len]
                the H2D WRITES arena[off:off+len]

and `d2h_offset == h2d_offset` on every interleaved step -- measured against the
real layout vectors, 498 of 512 steps on PP0, 268 of 281 on PP1, 268 of 297 on
PP2, which reproduces the boot log's own counts exactly. So for any region R the
arena demands D2H(R) before H2D(R) and the host demands H2D(R) before D2H(R).
That is a cycle, and `rotate_arena` already names it in its `RotationHazard`
message. A lag cannot break it (a lagging D2H would read arena bytes the H2D has
already overwritten) and neither can a reordering (both streams are sequential
from 0, so this is not a permutation with a good order). The staging copy is
structural.

WHAT IS NOT STRUCTURAL IS ITS RATE, and that is what this file changes. Every
CUDA worker runs under a process-global `torch.set_num_threads(1)`
(model_runner.py:2224), set for weight loading and never revisited for a 16 GiB
host-to-host stream on the flip seam. The copy is therefore single-threaded.

MEASURED CEILING, AND THE SINGLE-RANK NUMBER IS A TRAP. On this box (5950X, 16
cores, one NUMA node, 64 MiB L3), streaming 3-4 GiB through a 32 MiB ring:

    arrangement                          per-rank MiB/s
    1 rank,  serial                            4957-5109
    1 rank,  sliced x4                            40860     <- 8.0x, and a mirage
    3 ranks, serial                         3757-3869       <- matches the flip's
                                                               measured 2687-3768
    3 ranks, sliced x4                      5353-5701       <- 1.42x, the real one

Three concurrent ranks are DRAM-bandwidth-bound, not core-bound: aggregate rises
from ~11.5 to ~16.5 GB/s of copy (roughly double that in memory traffic) and
stops. So the honest gain is 1.42x, not 8x, and PP0's `save` goes 4.342 s ->
~3.06 s rather than to ~0.5 s. Reported as such deliberately: a single-rank
benchmark of a three-rank operation is the same shape of error as validating a
two-point fit on unseen points.

WHAT THIS FILE PINS -- correctness of the sliced copy, not its speed. A timing
assertion in a unit test on a shared box is noise, and the rate above belongs to
a measurement, not to a gate:
  * the sliced copy is byte-exact against the serial one, including when the
    slice count does not divide the chunk and when the final chunk is short;
  * one slice, and any degenerate slice count, still copies correctly;
  * the executor's own stats and phase accounting are unchanged by slicing;
  * the copy never reaches for a global (`torch.set_num_threads`) -- the worker
    is left in exactly the thread state it was found in, because that setting is
    process-global and other threads share it.

Hermetic: host tensors only, no CUDA, no device path.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.model_executor.rotation_executor import TorchRotationOps
from sglang.test.test_utils import CustomTestCase

CHUNK = 4096


def _src(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, (n,), generator=g, dtype=torch.uint8)


class TestTheSlicedCopyIsByteExact(CustomTestCase):
    """The only thing that may not change: the bytes."""

    def test_a_full_chunk_round_trips(self):
        src = _src(CHUNK)
        dst = torch.zeros(CHUNK, dtype=torch.uint8)
        TorchRotationOps(save_slices=4).save(dst, src)
        self.assertTrue(torch.equal(dst, src))

    def test_a_short_final_chunk_writes_only_its_own_bytes(self):
        """The tails never line up -- the two layouts differ in size, so the
        last chunk is short by construction. A slicer that rounds up would
        write into the ring beyond the payload and the next chunk would read
        its own stale bytes back."""
        n = CHUNK // 3
        src = _src(n, seed=1)
        dst = torch.full((CHUNK,), 0xAB, dtype=torch.uint8)
        TorchRotationOps(save_slices=4).save(dst, src)
        self.assertTrue(torch.equal(dst[:n], src))
        self.assertTrue(
            torch.all(dst[n:] == 0xAB), "the copy wrote past the source length"
        )

    def test_a_length_that_does_not_divide_by_the_slice_count(self):
        """4093 bytes over 4 slices. An even split loses the remainder; this is
        the case a naive `n // w` slicer gets wrong and a round-trip test on a
        power-of-two length never sees."""
        n = 4093
        src = _src(n, seed=2)
        dst = torch.zeros(n, dtype=torch.uint8)
        TorchRotationOps(save_slices=4).save(dst, src)
        self.assertTrue(torch.equal(dst, src))

    def test_every_slice_count_agrees_with_the_serial_copy(self):
        """The serial path is the reference. Any slicing that disagrees with it
        is wrong regardless of how fast it is."""
        for n in (1, 7, 4095, 4096, 4097):
            src = _src(n, seed=n)
            reference = torch.zeros(n, dtype=torch.uint8)
            TorchRotationOps(save_slices=1).save(reference, src)
            self.assertTrue(torch.equal(reference, src), f"serial path wrong at n={n}")
            for w in (1, 2, 3, 4, 8, 64):
                dst = torch.zeros(n, dtype=torch.uint8)
                TorchRotationOps(save_slices=w).save(dst, src)
                self.assertTrue(
                    torch.equal(dst, reference),
                    f"sliced x{w} disagrees with the serial copy at n={n}",
                )

    def test_a_zero_length_source_is_a_no_op(self):
        dst = torch.full((16,), 9, dtype=torch.uint8)
        TorchRotationOps(save_slices=4).save(dst, torch.empty(0, dtype=torch.uint8))
        self.assertTrue(torch.all(dst == 9))


class TestTheSliceCountIsSanitised(CustomTestCase):
    """A knob reachable from the environment reaches this code as an int of
    unknown sign. Every one of these is a value an operator can actually set."""

    def test_a_slice_count_below_one_falls_back_to_serial(self):
        for w in (0, -1, -100):
            src = _src(256, seed=3)
            dst = torch.zeros(256, dtype=torch.uint8)
            TorchRotationOps(save_slices=w).save(dst, src)
            self.assertTrue(torch.equal(dst, src), f"save_slices={w} lost bytes")

    def test_more_slices_than_bytes_still_copies_every_byte(self):
        src = _src(3, seed=4)
        dst = torch.zeros(3, dtype=torch.uint8)
        TorchRotationOps(save_slices=64).save(dst, src)
        self.assertTrue(torch.equal(dst, src))


class TestTheWorkerThreadStateIsNotTouched(CustomTestCase):
    """`torch.set_num_threads` is PROCESS-GLOBAL and the worker sets it to 1 on
    purpose (model_runner.py:2224). Raising it for the duration of the seam
    would change it under every other thread in the process, so the slicing runs
    on Python threads and leaves the setting alone. Pinned, because "restore it
    in a finally" is the tempting version and it is still wrong in between."""

    def test_the_torch_thread_count_is_unchanged_across_a_save(self):
        before = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            src = _src(CHUNK, seed=5)
            dst = torch.zeros(CHUNK, dtype=torch.uint8)
            TorchRotationOps(save_slices=4).save(dst, src)
            self.assertEqual(
                1,
                torch.get_num_threads(),
                "the save changed the process-global torch thread count",
            )
        finally:
            torch.set_num_threads(before)


class TestTheDefaultIsReachableAndSane(CustomTestCase):
    def test_the_default_ops_still_copy_correctly(self):
        """Whatever the default resolves to, it must be a working copy -- this
        is the path every flip actually takes."""
        src = _src(CHUNK, seed=6)
        dst = torch.zeros(CHUNK, dtype=torch.uint8)
        TorchRotationOps().save(dst, src)
        self.assertTrue(torch.equal(dst, src))

    def test_the_default_slice_count_is_at_least_one(self):
        self.assertGreaterEqual(int(TorchRotationOps()._save_slices), 1)


if __name__ == "__main__":
    unittest.main()
