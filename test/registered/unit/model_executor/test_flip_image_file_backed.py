"""Flip host images: an opt-in FILE-BACKED arm, so the post is reclaimable.

THE POST THIS EXISTS FOR. The phase flip holds two exact-size page-locked
host images per rank (``image_pp``, ``image_tp``) plus a draft image, for the
life of the process -- 68.7 GiB on the review composition
(SPECIMEN-2026-08-18T0516Z-*), NON-reclaimable, on a swapless 118-GiB
container whose review boot also carries hicache and mamba host posts. The
pin buys one thing: ``arena_refill``'s H2D copy at flip time is a DMA
(#690 measured the refill at 9,614.9 MiB per rank).

THE ARM. ``SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED=1`` (with
``SGLANG_PHASE_FLIP_IMAGE_DIR`` pointing at persistent storage) allocates the
image as a file-backed shared mapping instead: the pages are page cache --
written back and RECLAIMED under memory pressure, refaulted from disk at the
next flip. The flip then pays a pageable H2D copy (no DMA pin) plus, when the
box was actually under pressure, a disk refault; the #89 hibernate restore
(8-14 s for a full weight set, same pool) is the same-medium cold-read
anchor. That price is why the arm is OPT-IN and the default byte-identical
pinned path.

THE RULES PINNED HERE, both directions each:

* default unchanged -- envs unset route exactly as before (#695 exact-size
  register path, host post registered);
* the opt-in never lies: enabled without a directory REFUSES (a flag that
  silently falls back to pinned would claim reclaimability the ledger then
  plans on -- the #742 defect class); a volatile (tmpfs) directory REFUSES,
  because RAM-backed files are exactly the post this arm exists to remove;
* the file is unlinked after mapping (no stale multi-GiB files after a
  crash), zero-filled by the filesystem (the checksum contract needs the
  alignment gaps zeroed), and NOT registered as a pinned host post -- the
  registry sums non-reclaimable bytes and these are reclaimable;
* the whole image round-trips: snapshot -> file-backed image -> arena_refill
  restores byte-identical content with a passing checksum.
"""

import os
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Routing:
    """The #695 spy, extended with the file arm: records routing without
    allocating pinned memory (CPU box, no CUDA context)."""

    def __init__(self):
        self.exact = []
        self.torch_zeros = []
        self.posts = []

    def install(self, case):
        def _exact(dims, dtype, device, pin_memory, allocator):
            self.exact.append(int(dims[0]))
            return torch.zeros(dims, dtype=dtype)

        def _zeros(total):
            self.torch_zeros.append(int(total))
            return torch.zeros(total, dtype=torch.uint8)

        def _post(nbytes):
            self.posts.append(int(nbytes))

        for name, fn in (
            ("_alloc_with_host_register", _exact),
            ("_torch_pinned_zeros", _zeros),
            ("_torch_pinned_empty", _zeros),
            ("_register_image_post", _post),
        ):
            original = getattr(weights_arena, name)
            setattr(weights_arena, name, fn)
            case.addCleanup(setattr, weights_arena, name, original)
        return self


class TestDefaultStaysPinned(CustomTestCase):
    """Byte-identical default: with the new envs unset, nothing changes."""

    def setUp(self):
        self.routing = _Routing().install(self)

    def test_default_routes_to_the_exact_pin_path_and_registers(self):
        envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.clear()
        envs.SGLANG_PHASE_FLIP_EXACT_PIN.clear()
        out = weights_arena._alloc_host_image(123456, pin=True)
        self.assertEqual(self.routing.exact, [123456])
        self.assertEqual(self.routing.posts, [123456])
        self.assertEqual(out.numel(), 123456)

    def test_mode_string_reports_pinned(self):
        envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.clear()
        self.assertEqual(weights_arena.host_image_mode(), "pinned")


class TestTheFileBackedArm(CustomTestCase):
    def setUp(self):
        self.routing = _Routing().install(self)
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="flip-img-test-")
        self.addCleanup(self._cleanup_dir)

    def _cleanup_dir(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_opt_in_allocates_a_reclaimable_file_mapping(self):
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(self.dir):
                out = weights_arena._alloc_host_image(65536, pin=True)
        self.assertEqual(out.numel(), 65536)
        # None of the pinned machinery ran.
        self.assertEqual(self.routing.exact, [])
        self.assertEqual(self.routing.torch_zeros, [])

    def test_the_file_is_unlinked_after_mapping(self):
        """No stale multi-GiB files after a crash: the mapping holds the
        inode, the namespace does not."""
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(self.dir):
                out = weights_arena._alloc_host_image(4096, pin=True)
        self.assertEqual(os.listdir(self.dir), [])
        del out

    def test_the_mapping_arrives_zeroed(self):
        """The checksum contract: alignment-gap bytes must be zero, and a
        fresh file's pages read back zero by the filesystem's own rule."""
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(self.dir):
                out = weights_arena._alloc_host_image(8192, pin=True)
        self.assertEqual(int(out.sum().item()), 0)

    def test_no_pinned_host_post_is_registered(self):
        """The registry sums NON-reclaimable bytes; charging a reclaimable
        image would re-create the 122.7G-on-118G ledger verdict this arm
        exists to dissolve."""
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(self.dir):
                weights_arena._alloc_host_image(4096, pin=True)
        self.assertEqual(self.routing.posts, [])

    def test_unpinned_path_is_untouched_by_the_flag(self):
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(self.dir):
                out = weights_arena._alloc_host_image(2048, pin=False)
        self.assertEqual(os.listdir(self.dir), [])
        self.assertEqual(out.numel(), 2048)

    def test_mode_string_reports_file_backed(self):
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            self.assertEqual(weights_arena.host_image_mode(), "file-backed reclaimable")

    def test_the_env_is_read_per_call(self):
        envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.clear()
        weights_arena._alloc_host_image(1024, pin=True)
        self.assertEqual(self.routing.exact, [1024])
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(self.dir):
                weights_arena._alloc_host_image(2048, pin=True)
        self.assertEqual(self.routing.exact, [1024])


class TestTheOptInNeverLies(CustomTestCase):
    """Refusals, not fallbacks: an enabled arm that quietly pins anyway would
    claim reclaimability the ledger then plans on (the #742 class)."""

    def setUp(self):
        self.routing = _Routing().install(self)

    def test_enabled_without_a_directory_refuses(self):
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(""):
                with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                    weights_arena._alloc_host_image(4096, pin=True)
        self.assertIn("SGLANG_PHASE_FLIP_IMAGE_DIR", str(ctx.exception))

    def test_a_tmpfs_directory_refuses_with_the_fs_named(self):
        """RAM-backed files are exactly the non-reclaimable post this arm
        exists to remove; /dev/shm must be refused, by name."""
        if not os.path.isdir("/dev/shm"):
            self.skipTest("no /dev/shm on this box")
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override("/dev/shm"):
                with self.assertRaises(weights_arena.WeightsArenaError) as ctx:
                    weights_arena._alloc_host_image(4096, pin=True)
        self.assertIn("tmpfs", str(ctx.exception))

    def test_a_missing_directory_refuses(self):
        with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
            with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override("/nonexistent-dir-746"):
                with self.assertRaises(weights_arena.WeightsArenaError):
                    weights_arena._alloc_host_image(4096, pin=True)


class TestTheImageRoundTrips(CustomTestCase):
    """End to end on CPU: snapshot -> file-backed image -> arena_refill
    restores byte-identical content with a passing checksum. The flip's
    correctness machinery (checksum after copy) is arm-independent."""

    def test_snapshot_refill_roundtrip(self):
        import tempfile

        from sglang.srt.model_executor.weights_arena import (
            arena_refill,
            image_from_tensors,
            plan_arena_layout,
        )

        named = {
            "a.weight": torch.arange(300, dtype=torch.float32).reshape(30, 10),
            "b.weight": torch.full((17,), 3.25, dtype=torch.float32),
        }
        layout = plan_arena_layout(named)
        with tempfile.TemporaryDirectory(prefix="flip-img-rt-") as d:
            with envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.override(True):
                with envs.SGLANG_PHASE_FLIP_IMAGE_DIR.override(d):
                    image = image_from_tensors(named, layout, pin=True)
            arena = torch.zeros(layout.total_bytes, dtype=torch.uint8)
            arena_refill(arena, layout, image)
        for slot in layout.slots:
            seg = arena[slot.offset : slot.offset + slot.nbytes]
            view = torch.as_strided(seg.view(slot.dtype), slot.shape, slot.stride)
            self.assertTrue(torch.equal(view, named[slot.name]))


if __name__ == "__main__":
    unittest.main()
