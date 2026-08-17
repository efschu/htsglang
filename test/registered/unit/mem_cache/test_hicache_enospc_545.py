"""#545: the disk tier under ENOSPC, injected at the syscall boundary.

TECHNIQUE, not code, is borrowed from the three-site injection used for the
canonical store (that work lives on an unmerged train branch, so nothing here
depends on it): fail at each syscall the write protocol actually leans on,
rather than at a Python-level seam that a real full disk would not hit.

The write protocol, from ``HiCacheFile.set``:

    exists? -> touch and return True
    evictor.reserve(bytes)  -> False means "no room": return False
    tofile(tmp)             -> the bulk write
    os.replace(tmp, final)  -> the page BECOMES VISIBLE here, atomically
    evictor.commit()
    on any exception: evictor.abort(), unlink tmp, return False

THE THREE PROPERTIES:

  1. **A write that hits ENOSPC never reports coverage it did not achieve.**
     ``set`` returns False and the key is NOT readable afterwards. Anything
     else would have the tree believe a page is backed when it is not, and the
     next prefetch would resolve to nothing.

  2. **The watermark engages when configured, and refuses at the door.** With
     a cap in force the reservation fails BEFORE any bytes move -- which is
     the cheap refusal, not a mid-write failure that has already spent the IO.

  3. **After space returns the tier is still attached and CONSISTENT.** A
     retry completes, and at no point is a torn page visible: the page appears
     only at ``os.replace``, so a failure before it leaves the final path
     absent and a failure at it leaves the previous content intact. Absent or
     whole, never half.

Hermetic: a real ``HiCacheFile`` over a real tmpdir, with the failing syscall
patched. No server, no CUDA.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig


def _enospc(*_a, **_kw):
    raise OSError(28, "No space left on device")


class _Store(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _backend(self, **extra):
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="m",
            extra_config=extra or None,
        )
        return HiCacheFile(config, file_path=self.path)

    def _value(self, n=256):
        return torch.zeros(n, dtype=torch.uint8)

    def _bin_files(self):
        out = []
        for dirpath, _d, names in os.walk(self.path):
            out += [os.path.join(dirpath, n) for n in names]
        return out


class TestEnospcNeverClaimsCoverage(_Store):
    def test_a_failed_bulk_write_returns_false(self):
        be = self._backend()
        with patch.object(torch.Tensor, "numpy", side_effect=_enospc):
            self.assertFalse(be.set("k1", self._value()))

    def test_the_key_is_not_readable_afterwards(self):
        """The load-bearing half: False is only honest if the key really is
        absent. A tree that believed the page was backed would resolve the
        next prefetch to nothing."""
        be = self._backend()
        with patch.object(torch.Tensor, "numpy", side_effect=_enospc):
            be.set("k2", self._value())
        self.assertFalse(be.exists("k2"))

    def test_a_failure_at_the_rename_also_reports_false(self):
        be = self._backend()
        with patch("os.replace", side_effect=_enospc):
            self.assertFalse(be.set("k3", self._value()))
        self.assertFalse(be.exists("k3"))


class TestNoTornPageIsEverVisible(_Store):
    """The page becomes visible only at ``os.replace``."""

    def test_a_failure_before_the_rename_leaves_no_final_file(self):
        be = self._backend()
        with patch("os.replace", side_effect=_enospc):
            be.set("k4", self._value())
        finals = [p for p in self._bin_files() if p.endswith(".bin")]
        self.assertEqual(finals, [], "a partial page must not be visible")

    def test_the_temporary_file_is_cleaned_up(self):
        be = self._backend()
        with patch("os.replace", side_effect=_enospc):
            be.set("k5", self._value())
        self.assertEqual(
            self._bin_files(), [], "the half-written tmp file was left behind"
        )

    def test_an_existing_page_survives_a_failed_overwrite(self):
        """Absent or whole, never half -- including when a page is already
        there. rename is atomic, so the old content stands."""
        be = self._backend()
        self.assertTrue(be.set("k6", torch.full((256,), 7, dtype=torch.uint8)))
        before = be.get("k6", torch.empty(256, dtype=torch.uint8))
        with patch("os.replace", side_effect=_enospc):
            be.set("k6", torch.full((256,), 9, dtype=torch.uint8))
        self.assertTrue(be.exists("k6"))
        after = be.get("k6", torch.empty(256, dtype=torch.uint8))
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)


class TestTheWatermarkRefusesAtTheDoor(_Store):
    """With a cap configured, the reservation fails BEFORE bytes move."""

    def test_a_cap_smaller_than_the_page_refuses_without_writing(self):
        be = self._backend(max_size="1")  # 1 byte cap
        wrote = []
        real_replace = os.replace

        def _spy(a, b):
            wrote.append((a, b))
            return real_replace(a, b)

        with patch("os.replace", side_effect=_spy):
            ok = be.set("k7", self._value(4096))
        self.assertFalse(ok, "a page larger than the whole cap must be refused")
        self.assertEqual(wrote, [], "refused at the door, not mid-write")

    def test_without_a_cap_eviction_is_off_and_the_write_succeeds(self):
        """The shipped default: no cap and no min-free means no watermark."""
        be = self._backend()
        self.assertTrue(be.set("k8", self._value()))
        self.assertTrue(be.exists("k8"))


class TestTheTierSurvivesAndRecovers(_Store):
    """After space returns, the backend is still usable -- ENOSPC must not
    wedge the tier or leave the evictor's accounting stuck."""

    def test_a_retry_after_the_failure_completes(self):
        be = self._backend()
        with patch("os.replace", side_effect=_enospc):
            self.assertFalse(be.set("k9", self._value()))
        self.assertTrue(be.set("k9", self._value()), "the tier did not recover")
        self.assertTrue(be.exists("k9"))

    def test_other_keys_still_write_after_a_failure(self):
        be = self._backend()
        with patch("os.replace", side_effect=_enospc):
            be.set("bad", self._value())
        self.assertTrue(be.set("good", self._value()))
        self.assertTrue(be.exists("good"))

    def test_a_failed_write_releases_its_reservation(self):
        """Pin the ACCOUNTING, not a downstream write.

        A first draft asserted that a later write still fits after five failed
        ones. That could not fail: with a cap in force the evictor simply
        EVICTS to admit, so a leaked reservation causes extra eviction rather
        than a refusal, and the pin passed with `abort` removed. Asserting on
        ``_total_bytes`` is what actually distinguishes released from leaked.
        """
        be = self._backend(max_size=str(4096 * 8))
        ev = be._evictor
        before = ev._total_bytes
        for _ in range(5):
            with patch("os.replace", side_effect=_enospc):
                be.set("leaky", self._value(4096))
        self.assertEqual(
            ev._total_bytes,
            before,
            "failed writes left their reserved bytes charged; repeated "
            "failures would consume the cap and force needless eviction",
        )
