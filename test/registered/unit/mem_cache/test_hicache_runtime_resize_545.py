"""
Unit tests for runtime resize of the HiCache disk tier (#545).

Covers the three layers that can be exercised without a GPU or a server:

* ``LRUFileEvictor.set_limits`` -- the state machine: grow, shrink-with-
  eviction, turning eviction on for an evictor that booted unbounded, the
  in-flight-write carve-out, and MLA non-owner ranks.
* ``HiCacheFile.resize`` / ``.capacity_stats`` -- the backend delegation, and
  that a backend without capacity accounting degrades cleanly.
* ``Scheduler.resize_hicache_storage_wrapped`` request validation -- bad
  sizes, hierarchical cache off, tree cache that cannot resize.

Both cache classes that can own a storage backend are checked for the method,
because a hybrid-SSM (GDN) model gets ``UnifiedRadixCache`` rather than
``HiRadixCache``.

These are pure CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest \
      test/registered/unit/mem_cache/test_hicache_runtime_resize_545.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorage,
    HiCacheStorageConfig,
)
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor
from sglang.test.test_utils import CustomTestCase

KIB = 1024


def _t(n_bytes: int, fill: int = 0) -> torch.Tensor:
    return torch.full((n_bytes,), fill, dtype=torch.uint8)


def _make_config(*, tp_rank=0, is_mla=False, extra_config=None):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
        extra_config=extra_config,
    )


class _EvictorHarness(CustomTestCase):
    """Builds evictors over a scratch directory holding real files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hicache-resize-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _evictor(self, *, max_size=None, min_free=None, tp_rank=0, is_mla=False):
        extra = {}
        if max_size is not None:
            extra["max_size"] = max_size
        if min_free is not None:
            extra["min_free_space"] = min_free
        return LRUFileEvictor(
            self.tmp,
            "_suffix",
            tp_rank=tp_rank,
            is_mla_model=is_mla,
            extra_config=extra,
        )

    def _write(self, evictor, key, n_bytes):
        """Admit and materialize one file, mirroring HiCacheFile.set()."""
        stem = f"{key}_suffix"
        if not evictor.reserve(stem, n_bytes, key=key):
            return False
        with open(os.path.join(self.tmp, f"{stem}.bin"), "wb") as f:
            f.write(b"\0" * n_bytes)
        evictor.commit(stem)
        return True

    def _files_on_disk(self):
        return sorted(f for f in os.listdir(self.tmp) if f.endswith(".bin"))


class TestSetLimitsStateMachine(_EvictorHarness):
    def test_grow_evicts_nothing(self):
        ev = self._evictor(max_size=10 * KIB)
        for i in range(5):
            self.assertTrue(self._write(ev, f"k{i}", KIB))
        before = ev.stats()["used_bytes"]

        out = ev.set_limits(max_size_bytes=100 * KIB)

        self.assertEqual(out["max_size_bytes"], 100 * KIB)
        self.assertEqual(out["freed_bytes"], 0)
        self.assertEqual(out["used_bytes"], before)
        self.assertEqual(len(self._files_on_disk()), 5)

    def test_shrink_evicts_until_under_the_new_cap(self):
        ev = self._evictor(max_size=100 * KIB)
        for i in range(10):
            self.assertTrue(self._write(ev, f"k{i}", KIB))
        self.assertEqual(ev.stats()["used_bytes"], 10 * KIB)

        out = ev.set_limits(max_size_bytes=4 * KIB)

        self.assertLessEqual(out["used_bytes"], 4 * KIB)
        self.assertGreater(out["freed_bytes"], 0)
        # Files are really gone from disk, not just from the index.
        self.assertEqual(len(self._files_on_disk()), out["num_entries"])

    def test_shrink_evicts_the_least_recently_used_first(self):
        ev = self._evictor(max_size=100 * KIB)
        for i in range(4):
            self.assertTrue(self._write(ev, f"k{i}", KIB))
        # Touch k0 so it is no longer the oldest.
        ev.touch("k0_suffix", os.path.join(self.tmp, "k0_suffix.bin"))

        ev.set_limits(max_size_bytes=2 * KIB)

        survivors = self._files_on_disk()
        self.assertIn("k0_suffix.bin", survivors)
        self.assertNotIn("k1_suffix.bin", survivors)

    def test_enabling_eviction_at_runtime_adopts_existing_files(self):
        """An evictor that booted unbounded keeps no index; resize must scan."""
        ev = self._evictor()  # no cap -> inert
        self.assertFalse(ev.configured)
        for i in range(8):
            self.assertTrue(self._write(ev, f"k{i}", KIB))
        # Inert evictor tracked nothing.
        self.assertEqual(ev.stats()["used_bytes"], 0)
        self.assertEqual(ev.stats()["num_entries"], 0)

        out = ev.set_limits(max_size_bytes=3 * KIB)

        self.assertTrue(out["configured"])
        self.assertTrue(out["enabled"])
        # It found the 8 KiB already on disk and evicted down to the cap.
        self.assertGreater(out["freed_bytes"], 0)
        self.assertLessEqual(out["used_bytes"], 3 * KIB)
        self.assertEqual(len(self._files_on_disk()), out["num_entries"])

    def test_in_flight_write_is_not_evicted_by_a_shrink(self):
        ev = self._evictor(max_size=100 * KIB)
        for i in range(4):
            self.assertTrue(self._write(ev, f"k{i}", KIB))
        # Reserve without committing: a backup thread mid-write.
        self.assertTrue(ev.reserve("pending_suffix", KIB, key="pending"))

        ev.set_limits(max_size_bytes=KIB)

        # The uncommitted reservation survived; committed neighbours went.
        self.assertIn("pending_suffix", ev._lru)

    def test_lifting_the_cap_disables_eviction(self):
        ev = self._evictor(max_size=10 * KIB)
        self.assertTrue(ev.configured)
        out = ev.set_limits(max_size_bytes=0, min_free_bytes=0)
        self.assertFalse(out["configured"])
        self.assertFalse(out["enabled"])

    def test_none_leaves_a_limit_unchanged(self):
        ev = self._evictor(max_size=10 * KIB, min_free=5 * KIB)
        out = ev.set_limits(max_size_bytes=20 * KIB)
        self.assertEqual(out["max_size_bytes"], 20 * KIB)
        self.assertEqual(out["min_free_bytes"], 5 * KIB)

    def test_non_owner_mla_rank_records_limits_but_stays_inert(self):
        ev = self._evictor(max_size=10 * KIB, tp_rank=1, is_mla=True)
        self.assertFalse(ev.is_storage_owner)
        out = ev.set_limits(max_size_bytes=20 * KIB)
        self.assertEqual(out["max_size_bytes"], 20 * KIB)
        self.assertTrue(out["configured"])
        self.assertFalse(out["enabled"])

    def test_stats_shape_is_stable(self):
        ev = self._evictor(max_size=10 * KIB)
        for key in (
            "configured",
            "enabled",
            "is_storage_owner",
            "max_size_bytes",
            "min_free_bytes",
            "eviction_ratio",
            "used_bytes",
            "num_entries",
        ):
            self.assertIn(key, ev.stats())


class TestBackendDelegation(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hicache-resize-be-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _backend(self, **extra):
        return HiCacheFile(_make_config(extra_config=extra or None), self.tmp)

    def test_file_backend_reports_and_resizes(self):
        b = self._backend(max_size=1024 * 1024)
        b.set("k0", _t(4096))

        stats = b.capacity_stats()
        self.assertEqual(stats["file_path"], self.tmp)
        self.assertEqual(stats["max_size_bytes"], 1024 * 1024)

        out = b.resize(max_size_bytes=2 * 1024 * 1024)
        self.assertEqual(out["max_size_bytes"], 2 * 1024 * 1024)
        self.assertEqual(out["file_path"], self.tmp)

    def test_base_class_degrades_cleanly(self):
        """A backend with no capacity accounting must say so, not crash."""
        self.assertIsNone(HiCacheStorage.capacity_stats(object()))
        self.assertIsNone(
            HiCacheStorage.resize(object(), max_size_bytes=1, min_free_bytes=1)
        )


class TestSchedulerRequestValidation(CustomTestCase):
    """Drives Scheduler.resize_hicache_storage_wrapped as an unbound method."""

    def _call(self, sched, **kwargs):
        from sglang.srt.managers.io_struct import ResizeHiCacheStorageReqInput
        from sglang.srt.managers.scheduler import Scheduler

        return Scheduler.resize_hicache_storage_wrapped(
            sched, ResizeHiCacheStorageReqInput(**kwargs)
        )

    def _sched(self, *, hierarchical=True, tree_cache=None):
        class _Sched:
            pass

        s = _Sched()
        s.enable_hierarchical_cache = hierarchical
        s.tree_cache = tree_cache if tree_cache is not None else _OkTree()
        return s

    def test_rejects_when_hierarchical_cache_is_off(self):
        out = self._call(self._sched(hierarchical=False), max_size_gb=10)
        self.assertFalse(out.success)
        self.assertIn("Hierarchical cache is not enabled", out.message)

    def test_rejects_empty_request(self):
        out = self._call(self._sched())
        self.assertFalse(out.success)
        self.assertIn("Nothing to resize", out.message)

    def test_rejects_zero_and_negative_max_size(self):
        for bad in (0, -1):
            out = self._call(self._sched(), max_size_gb=bad)
            self.assertFalse(out.success)
            self.assertIn("must be > 0", out.message)

    def test_rejects_negative_min_free(self):
        out = self._call(self._sched(), min_free_gb=-1)
        self.assertFalse(out.success)
        self.assertIn("must be >= 0", out.message)

    def test_rejects_tree_cache_without_resize_support(self):
        class _NoResize:
            pass

        out = self._call(self._sched(tree_cache=_NoResize()), max_size_gb=10)
        self.assertFalse(out.success)
        self.assertIn("does not support resize", out.message)

    def test_happy_path_converts_gib_and_returns_stats(self):
        tree = _OkTree()
        out = self._call(self._sched(tree_cache=tree), max_size_gb=2, min_free_gb=1)
        self.assertTrue(out.success)
        self.assertEqual(tree.seen, (2 * 1024**3, 1024**3))
        self.assertEqual(out.stats, {"used_bytes": 0})

    def test_backend_failure_is_reported_not_raised(self):
        out = self._call(self._sched(tree_cache=_FailTree()), max_size_gb=2)
        self.assertFalse(out.success)
        self.assertIn("boom", out.message)


class _OkTree:
    def __init__(self):
        self.seen = None

    def resize_storage_backend(self, max_size_bytes=None, min_free_bytes=None):
        self.seen = (max_size_bytes, min_free_bytes)
        return True, "ok", {"used_bytes": 0}


class _FailTree:
    def resize_storage_backend(self, max_size_bytes=None, min_free_bytes=None):
        raise RuntimeError("boom")


class TestBothCacheClassesExposeResize(CustomTestCase):
    """A hybrid-SSM (GDN) model gets UnifiedRadixCache, not HiRadixCache."""

    def test_hiradix_cache_has_resize(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        self.assertTrue(hasattr(HiRadixCache, "resize_storage_backend"))
        self.assertTrue(hasattr(HiRadixCache, "storage_capacity_stats"))

    def test_unified_radix_cache_has_resize(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.assertTrue(hasattr(UnifiedRadixCache, "resize_storage_backend"))
        self.assertTrue(hasattr(UnifiedRadixCache, "storage_capacity_stats"))

    def test_unified_radix_cache_resize_guards_on_disabled_storage(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        class _C:
            enable_storage = False
            cache_controller = None

        ok, msg, stats = UnifiedRadixCache.resize_storage_backend(
            _C(), max_size_bytes=1
        )
        self.assertFalse(ok)
        self.assertIsNone(stats)
        self.assertIn("not enabled", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
