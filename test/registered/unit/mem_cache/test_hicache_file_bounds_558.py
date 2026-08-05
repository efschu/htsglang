"""
Unit tests for the HiCacheFile disk-bound fixes of task #558.

Background (live incident 2026-08-05): a serving run configured with
``{"max_size": "100Gi", "min_free_space": "20Gi"}`` still put ~294 GiB and
11.7 million flat ``.bin`` files on one filesystem and drove it to 100%.
Three defects made that possible, and each gets a falsifier here:

1. The cap was enforced *per rank*. Every TP rank builds its own evictor over
   the same directory and seeds it from a scan filtered to that rank's own
   key suffix, so a TP=3 run silently spends 3x the configured cap
   (``TestSharedDirectoryBudget``).
2. Accounting used *apparent* file size while the filesystem charges
   *allocated* blocks. The 512-byte ``.draft`` pages of the incident cost
   8704 bytes each on disk -- a 17x undercount (``TestAllocatedAccounting``).
3. ``min_free_space`` was only ever consulted inside a write. Nothing looked
   at free space when the backend was idle, and running out of room produced
   a per-page warning flood instead of one loud error plus a latched write
   stop (``TestFreeSpaceWatchdog``).

``TestSharding`` covers the flat-directory fix: new writes go into 2-hex
prefix shards, pre-existing flat files stay readable and evictable.

These are pure CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_bounds_558.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import logging
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

_HEX = "0123456789abcdef"


def _key(i: int) -> str:
    """A realistic 64-hex page key whose first two chars vary with ``i``."""
    return f"{i:02x}" + f"{i:062x}"


def _t(n_bytes: int, fill: int = 7) -> torch.Tensor:
    return torch.full((n_bytes,), fill, dtype=torch.uint8)


def _make_backend(
    file_path: str,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    max_size=None,
    min_free=None,
    eviction_ratio=None,
    extra=None,
) -> HiCacheFile:
    extra_config = {
        "max_size": max_size,
        "min_free_space": min_free,
        "eviction_ratio": eviction_ratio,
    }
    extra_config.update(extra or {})
    cfg = HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
        extra_config=extra_config,
    )
    return HiCacheFile(cfg, file_path=file_path)


def _disk_usage(root: str):
    """(apparent bytes, allocated bytes, file count) below ``root``."""
    apparent = allocated = count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            try:
                st = os.stat(os.path.join(dirpath, fn))
            except OSError:
                continue
            apparent += st.st_size
            allocated += st.st_blocks * 512
            count += 1
    return apparent, allocated, count


class _TmpDirCase(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hicache558_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestSharedDirectoryBudget(_TmpDirCase):
    """max_size must bound the DIRECTORY, not each rank separately."""

    def test_cap_is_shared_across_tp_ranks(self):
        d = os.path.join(self.tmp, "shared")
        os.makedirs(d, exist_ok=True)
        page = 256 * 1024
        cap = 3 * 1024 * 1024  # 3 MiB for the whole directory

        backends = [
            _make_backend(d, tp_rank=r, tp_size=3, max_size=str(cap)) for r in range(3)
        ]
        # Every rank tries to write 2 MiB on its own: 6 MiB requested, 3 MiB allowed.
        for r, backend in enumerate(backends):
            for i in range(8):
                backend.set(_key(r * 16 + i), _t(page))

        apparent, _allocated, _count = _disk_usage(d)
        self.assertLessEqual(
            apparent,
            cap,
            f"directory holds {apparent} B, above the configured {cap} B cap "
            f"-- the cap is being applied per rank instead of per directory",
        )

    def test_per_rank_scope_is_opt_in(self):
        """The old (multiplying) semantics stay reachable, explicitly."""
        d = os.path.join(self.tmp, "perrank")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(
            d,
            tp_rank=0,
            tp_size=4,
            max_size="4Mi",
            extra={"max_size_scope": "per_rank"},
        )
        self.assertEqual(b._evictor.max_size_bytes, 4 * 1024 * 1024)

    def test_shared_scope_divides_by_writer_count(self):
        d = os.path.join(self.tmp, "sharedscope")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, tp_rank=0, tp_size=4, max_size="4Mi")
        self.assertEqual(b._evictor.max_size_bytes, 1024 * 1024)


class TestAllocatedAccounting(_TmpDirCase):
    """Accounting must follow allocated blocks, not apparent size."""

    def test_allocated_size_uses_blocks_not_length(self):
        """The exact numbers of the incident: 512 B of payload, 8704 B of disk.

        Filesystems charge allocated blocks. Accounting a page at its apparent
        length undercounted the incident's 5.8 million ``.draft`` pages 17-fold,
        so the byte cap bounded a quantity the disk does not care about.
        """
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        draft_page = SimpleNamespace(st_size=512, st_blocks=17)
        self.assertEqual(LRUFileEvictor._allocated_size(draft_page), 8704)

        # Delayed allocation (ZFS reports one block for a file that has not
        # reached a transaction group yet): never account below the payload.
        fresh_write = SimpleNamespace(st_size=64 * 1024, st_blocks=1)
        self.assertEqual(LRUFileEvictor._allocated_size(fresh_write), 64 * 1024)

        # Filesystems without st_blocks fall back to the apparent length.
        no_blocks = SimpleNamespace(st_size=4096)
        self.assertEqual(LRUFileEvictor._allocated_size(no_blocks), 4096)

    def test_used_bytes_covers_allocated_blocks(self):
        d = os.path.join(self.tmp, "alloc")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi")
        for i in range(32):
            self.assertTrue(b.set(_key(i), _t(512)))

        apparent, allocated, count = _disk_usage(d)
        self.assertEqual(count, 32)
        used = b._evictor.stats()["used_bytes"]
        self.assertGreaterEqual(
            used,
            allocated,
            f"evictor accounts {used} B but the files occupy {allocated} B "
            f"({apparent} B apparent) -- the cap undercounts real disk usage",
        )

    def test_scan_adopts_allocated_size(self):
        d = os.path.join(self.tmp, "allocscan")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi")
        for i in range(16):
            b.set(_key(i), _t(512))
        del b

        b2 = _make_backend(d, max_size="8Mi")
        _apparent, allocated, _count = _disk_usage(d)
        self.assertGreaterEqual(b2._evictor.stats()["used_bytes"], allocated)


class TestFreeSpaceWatchdog(_TmpDirCase):
    """Free space must be policed outside the write path, loudly."""

    @staticmethod
    def _statvfs_with_free(free_bytes: int):
        real = os.statvfs

        def fake(path):
            st = real(path)
            fields = list(st)
            # (bsize, frsize, blocks, bfree, bavail, ...)
            frsize = fields[1] or 4096
            fields[4] = free_bytes // frsize
            fields[3] = fields[4]
            return os.statvfs_result(fields)

        return fake

    def test_watchdog_latches_write_stop_and_logs_error(self):
        d = os.path.join(self.tmp, "watchdog")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi", min_free="1Gi")
        self.assertTrue(b.set(_key(1), _t(4096)))

        logger = logging.getLogger("sglang.srt.mem_cache.storage.file.lru_file_evictor")
        with mock.patch("os.statvfs", self._statvfs_with_free(64 * 1024 * 1024)):
            with self.assertLogs(logger, level="ERROR") as captured:
                b.check_disk_space(force=True)
            self.assertTrue(
                any("min_free" in line for line in captured.output),
                f"watchdog did not name the breached limit: {captured.output}",
            )
            self.assertTrue(b._evictor.write_stopped)
            # A stopped backend must miss, not write.
            _before = _disk_usage(d)[2]
            self.assertFalse(b.set(_key(2), _t(4096)))
            self.assertEqual(_disk_usage(d)[2], _before)

        # Space came back: the latch releases and writes resume.
        b.check_disk_space(force=True)
        self.assertFalse(b._evictor.write_stopped)
        self.assertTrue(b.set(_key(3), _t(4096)))

    def test_watchdog_holds_the_latch_inside_the_hysteresis_band(self):
        """Barely back above min_free is not a recovery: no write/stop flapping."""
        d = os.path.join(self.tmp, "watchdog_band")
        os.makedirs(d, exist_ok=True)
        min_free = 1024 * 1024 * 1024
        b = _make_backend(d, max_size="8Mi", min_free=str(min_free))
        with mock.patch("os.statvfs", self._statvfs_with_free(min_free // 64)):
            b.check_disk_space(force=True)
        self.assertTrue(b._evictor.write_stopped)

        # 1% above the watermark: still stopped (release needs 5% margin).
        with mock.patch("os.statvfs", self._statvfs_with_free(int(min_free * 1.01))):
            self.assertFalse(b.check_disk_space(force=True))
            self.assertTrue(b._evictor.write_stopped)

        # 10% above: released.
        with mock.patch("os.statvfs", self._statvfs_with_free(int(min_free * 1.10))):
            self.assertTrue(b.check_disk_space(force=True))
            self.assertFalse(b._evictor.write_stopped)

    def test_watchdog_is_rate_limited(self):
        d = os.path.join(self.tmp, "watchdog_rate")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi", min_free="1Gi", extra={})
        calls = []
        real = os.statvfs

        def counting(path):
            calls.append(path)
            return real(path)

        with mock.patch("os.statvfs", counting):
            for _ in range(50):
                b.check_disk_space()
        self.assertLessEqual(
            len(calls), 2, f"watchdog probed statvfs {len(calls)} times in a burst"
        )

    def test_watchdog_is_reentrant_safe_under_concurrent_writers(self):
        """The latch is read and written under a non-reentrant lock: no deadlock.

        reserve() consults the watchdog before taking the evictor lock, and the
        watchdog takes that same lock itself. Drive both from several threads at
        once; a nesting mistake would hang here rather than fail a comparison.
        """
        import threading

        d = os.path.join(self.tmp, "watchdog_threads")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi", min_free="1Mi")
        errors = []

        def worker(n):
            try:
                for i in range(25):
                    b.check_disk_space(force=(i % 5 == 0))
                    b.set(_key(n * 32 + i), _t(1024))
            except Exception as e:  # pragma: no cover - only on a real failure
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertFalse([t for t in threads if t.is_alive()], "watchdog deadlocked")
        self.assertEqual(errors, [])
        # Accounting survived the concurrency: no negative or lost bytes.
        stats = b._evictor.stats()
        self.assertGreaterEqual(stats["used_bytes"], 0)
        self.assertLessEqual(stats["used_bytes"], stats["max_size_bytes"])

    def test_watchdog_inert_without_min_free(self):
        d = os.path.join(self.tmp, "watchdog_off")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi")
        with mock.patch("os.statvfs", self._statvfs_with_free(0)):
            b.check_disk_space(force=True)
        self.assertFalse(b._evictor.write_stopped)


class TestSharding(_TmpDirCase):
    """New writes land in 2-hex shards; pre-existing flat files still work."""

    def test_new_writes_are_sharded(self):
        d = os.path.join(self.tmp, "shard")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d)
        k = _key(0xAB)
        self.assertTrue(b.set(k, _t(1024)))

        stem = b._get_suffixed_key(k)
        sharded = os.path.join(d, k[:2], f"{stem}.bin")
        flat = os.path.join(d, f"{stem}.bin")
        self.assertTrue(os.path.exists(sharded), f"expected sharded file at {sharded}")
        self.assertFalse(os.path.exists(flat), "new write must not be flat")

    def test_shard_fanout_is_bounded(self):
        d = os.path.join(self.tmp, "fanout")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d)
        for i in range(64):
            b.set(_key(i), _t(512))
        shards = [
            name for name in os.listdir(d) if os.path.isdir(os.path.join(d, name))
        ]
        self.assertEqual(len(shards), 64)
        for name in shards:
            self.assertEqual(len(name), 2)
            self.assertTrue(all(c in _HEX for c in name), name)
            self.assertEqual(len(os.listdir(os.path.join(d, name))), 1)

    def test_legacy_flat_file_is_read_through(self):
        d = os.path.join(self.tmp, "legacy")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d)
        k = _key(0x5C)
        stem = b._get_suffixed_key(k)
        payload = _t(2048, fill=3)
        with open(os.path.join(d, f"{stem}.bin"), "wb") as f:
            f.write(payload.numpy().tobytes())

        self.assertTrue(b.exists(k))
        out = b.get(k, torch.zeros(2048, dtype=torch.uint8))
        self.assertIsNotNone(out)
        self.assertTrue(torch.equal(out, payload))

    def test_legacy_flat_file_is_evictable(self):
        d = os.path.join(self.tmp, "legacy_evict")
        os.makedirs(d, exist_ok=True)
        stem_probe = _make_backend(d)
        old_key = _key(0x11)
        old_stem = stem_probe._get_suffixed_key(old_key)
        with open(os.path.join(d, f"{old_stem}.bin"), "wb") as f:
            f.write(b"\0" * (900 * 1024))
        del stem_probe

        b = _make_backend(d, max_size="1Mi", eviction_ratio="0.5")
        self.assertGreater(b._evictor.stats()["used_bytes"], 0)
        # Writing a fresh page must evict the legacy flat file.
        self.assertTrue(b.set(_key(0x22), _t(256 * 1024)))
        self.assertFalse(
            os.path.exists(os.path.join(d, f"{old_stem}.bin")),
            "legacy flat victim was not unlinked",
        )

    def test_startup_scan_walks_shards_and_flat(self):
        d = os.path.join(self.tmp, "scan")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d, max_size="8Mi")
        for i in range(8):
            b.set(_key(i), _t(4096))
        flat_stem = b._get_suffixed_key(_key(0xF0))
        with open(os.path.join(d, f"{flat_stem}.bin"), "wb") as f:
            f.write(b"\0" * 4096)
        del b

        b2 = _make_backend(d, max_size="8Mi")
        self.assertEqual(b2._evictor.stats()["num_entries"], 9)

    def test_clear_removes_sharded_and_flat(self):
        d = os.path.join(self.tmp, "clear")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d)
        for i in range(4):
            b.set(_key(i), _t(1024))
        flat_stem = b._get_suffixed_key(_key(0xE0))
        with open(os.path.join(d, f"{flat_stem}.bin"), "wb") as f:
            f.write(b"\0" * 1024)

        self.assertTrue(b.clear())
        self.assertEqual(_disk_usage(d)[2], 0)

    def test_roundtrip_after_restart(self):
        d = os.path.join(self.tmp, "roundtrip")
        os.makedirs(d, exist_ok=True)
        b = _make_backend(d)
        payload = _t(4096, fill=9)
        k = _key(0x7D)
        self.assertTrue(b.set(k, payload))
        del b

        b2 = _make_backend(d)
        self.assertTrue(b2.exists(k))
        out = b2.get(k, torch.zeros(4096, dtype=torch.uint8))
        self.assertIsNotNone(out)
        self.assertTrue(torch.equal(out, payload))


if __name__ == "__main__":
    unittest.main()
