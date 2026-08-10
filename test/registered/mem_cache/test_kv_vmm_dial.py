# SPDX-License-Identifier: Apache-2.0
"""#330 VMM dial primitives: chunked commit + tail decommit on a real device.

Requires CUDA (driver VMM API); everything above these primitives is covered
by the hermetic CPU tests in test/registered/scheduler/test_vram_dial.py.

Gates:
* chunked commit_range creates whole-chunk extents; decommit_range releases
  the tail back to the DRIVER (device free memory rises by the released
  amount, within allocator noise) and never touches bytes below the keep
  point;
* KvVmmBufferOwner.shrink + finalize round-trips: content below the shrink
  point survives, re-grown rows read as zeros (the flush-identity rule);
* addresses are stable across shrink/grow (tensor data_ptr unchanged).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu")

MIB = 1024 * 1024


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (driver VMM API)")
class TestKvVmmDial(CustomTestCase):
    def test_arena_chunked_commit_and_tail_decommit(self):
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena

        arena = KvVmmArena(
            torch.cuda.current_device(),
            reserve_bytes=512 * MIB,
            commit_chunk_bytes=16 * MIB,
        )
        try:
            free_before, _ = torch.cuda.mem_get_info()
            arena.commit_range(0, 256 * MIB)
            self.assertEqual(arena.backed_bytes, 256 * MIB)
            self.assertEqual(arena.committed_bytes(0), 256 * MIB)
            free_committed, _ = torch.cuda.mem_get_info()
            self.assertLess(free_committed, free_before)

            released = arena.decommit_range(0, 64 * MIB)
            self.assertEqual(released, 192 * MIB)
            self.assertEqual(arena.committed_bytes(0), 64 * MIB)
            self.assertEqual(arena.backed_bytes, 64 * MIB)
            free_after, _ = torch.cuda.mem_get_info()
            # Driver-level release: free memory really comes back (allow a
            # few MiB of unrelated allocator noise).
            self.assertGreater(free_after, free_committed + 180 * MIB)

            # Mid-chunk keep point rounds UP: keeping 65 MiB keeps 5 chunks.
            arena.commit_range(0, 256 * MIB)
            released = arena.decommit_range(0, 65 * MIB)
            self.assertEqual(arena.committed_bytes(0), 80 * MIB)
            # Re-commit continues from the watermark.
            arena.commit_range(0, 128 * MIB)
            self.assertEqual(arena.committed_bytes(0), 128 * MIB)
        finally:
            arena.close()

    def test_a_buffer_subset_can_be_released_and_restored_alone(self):
        """#631 waved seam: per-buffer backing, not just per-owner.

        The phase flip hands the source layout's pages back and re-commits
        the destination's ONE LAYER WAVE at a time, so only one wave's
        payload is ever staged. That needs the arena to move a SUBSET of
        its buffers and leave the rest mapped and readable -- a whole-owner
        shrink would unmap layers the next wave still has to read.
        """
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
        from sglang.srt.mem_cache.memory_pool import KvBufferDesc

        rows, row_elems = 8192, 4096  # 64 MiB per buffer
        descs = [
            KvBufferDesc(
                f"b{i}", (rows, row_elems), row_bytes=row_elems * 2, tokens_per_row=1
            )
            for i in range(4)
        ]
        owner = KvVmmBufferOwner(
            device=f"cuda:{torch.cuda.current_device()}",
            device_id=torch.cuda.current_device(),
            store_dtype=torch.float16,
            page_size=1,
            reserved_num_tokens=rows - 1,
            buffer_descs=descs,
            commit_chunk_bytes=4 * MIB,
        )
        try:
            owner.finalize(4000)
            for i, t in enumerate(owner.tensors):
                t[:2000].fill_(float(i + 1))
            backed_full = owner.backed_bytes
            free_before, _ = torch.cuda.mem_get_info()

            # Release buffers 0 and 2 only.
            released = owner.shrink(1, buffer_indices=[0, 2])
            self.assertGreater(released, 0)
            self.assertLess(owner.backed_bytes, backed_full)
            free_after, _ = torch.cuda.mem_get_info()
            self.assertGreater(free_after, free_before)

            # THE POINT: the untouched buffers are still mapped, still hold
            # their bytes, and are still writable. If a subset release took
            # the whole owner down, this is where it would fault.
            for i in (1, 3):
                t = owner.tensors[i]
                self.assertTrue(bool((t[:2000] == float(i + 1)).all()))
                t[2000:2100].fill_(9.0)
                self.assertTrue(bool((t[2000:2100] == 9.0).all()))

            # Restoring the same subset returns the span, addresses intact.
            ptrs = [t.data_ptr() for t in owner.tensors]
            owner.finalize(4000, buffer_indices=[0, 2])
            self.assertEqual(ptrs, [t.data_ptr() for t in owner.tensors])
            self.assertEqual(owner.backed_bytes, backed_full)
            for i in (0, 2):
                owner.tensors[i][:100].fill_(5.0)
                self.assertTrue(bool((owner.tensors[i][:100] == 5.0).all()))
            # And the survivors were never disturbed by the round trip.
            for i in (1, 3):
                self.assertTrue(bool((owner.tensors[i][:2000] == float(i + 1)).all()))
        finally:
            owner.close()

    def test_an_out_of_range_buffer_index_is_refused_before_any_move(self):
        """A bad subset must raise BEFORE it unmaps anything.

        The seam calls this from inside the flip's no-return region, where
        a half-applied subset would leave one layout's layers backed and
        the other's not.
        """
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
        from sglang.srt.mem_cache.memory_pool import KvBufferDesc

        rows, row_elems = 4096, 1024
        descs = [
            KvBufferDesc(
                f"b{i}", (rows, row_elems), row_bytes=row_elems * 2, tokens_per_row=1
            )
            for i in range(2)
        ]
        owner = KvVmmBufferOwner(
            device=f"cuda:{torch.cuda.current_device()}",
            device_id=torch.cuda.current_device(),
            store_dtype=torch.float16,
            page_size=1,
            reserved_num_tokens=rows - 1,
            buffer_descs=descs,
            commit_chunk_bytes=4 * MIB,
        )
        try:
            owner.finalize(2000)
            backed = owner.backed_bytes
            with self.assertRaises(ValueError):
                owner.shrink(1, buffer_indices=[0, 7])
            self.assertEqual(owner.backed_bytes, backed)
        finally:
            owner.close()

    def test_owner_shrink_grow_roundtrip_stable_addresses(self):
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
        from sglang.srt.mem_cache.memory_pool import KvBufferDesc

        rows, row_elems = 8192, 4096  # 8k rows x 8 KiB = 64 MiB per buffer
        descs = [
            KvBufferDesc(
                f"b{i}", (rows, row_elems), row_bytes=row_elems * 2, tokens_per_row=1
            )
            for i in range(2)
        ]
        owner = KvVmmBufferOwner(
            device=f"cuda:{torch.cuda.current_device()}",
            device_id=torch.cuda.current_device(),
            store_dtype=torch.float16,
            page_size=1,
            reserved_num_tokens=rows - 1,
            buffer_descs=descs,
            commit_chunk_bytes=4 * MIB,
        )
        try:
            owner.finalize(4000)
            ptrs = [t.data_ptr() for t in owner.tensors]
            for t in owner.tensors:
                t[:2000].fill_(3.0)
            backed_full = owner.backed_bytes

            released = owner.shrink(2000)
            self.assertGreater(released, 0)
            self.assertLess(owner.backed_bytes, backed_full)
            self.assertEqual(ptrs, [t.data_ptr() for t in owner.tensors])
            # Content below the shrink point survives.
            for t in owner.tensors:
                self.assertTrue(bool((t[:2000] == 3.0).all()))

            owner.finalize(4000)
            self.assertEqual(ptrs, [t.data_ptr() for t in owner.tensors])
            for t in owner.tensors:
                self.assertTrue(bool((t[:2000] == 3.0).all()))
                # Re-grown rows are writable and hold what is written. (The
                # driver does NOT guarantee zeroed pages on same-process
                # re-commit -- which is exactly why the POOL layer zeroes
                # newly grown rows in runtime_set_backing_tokens.)
                t[3500:4000].fill_(7.0)
                self.assertTrue(bool((t[3500:4000] == 7.0).all()))
        finally:
            owner.close()


if __name__ == "__main__":
    unittest.main()
