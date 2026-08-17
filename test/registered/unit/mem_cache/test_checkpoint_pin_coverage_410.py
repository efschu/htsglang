# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#410 slice 2 remainder: a pin that was DROPPED must not look like success.

WHAT SLICE 2 ALREADY DOES, and this suite does not re-test: it pins a
checkpoint's references before writing the manifest, the evictor skips pinned
stems, and ``create_checkpoint`` refuses a manifest whose references the store
does not hold (``ManifestIncomplete``).

THE HOLE THAT REMAINS. Those are two different authorities, checked one after
the other:

* ``verify_against_store`` asks ``HiCacheFile.exists``, which answers from the
  METADATA CACHE when it has an entry (``hicache_storage.py:1161``).
* ``pin_checkpoint`` asks the FILESYSTEM, via ``_existing_path`` + ``os.stat``
  inside ``stems_with_sizes``, which documents that "missing files are dropped
  rather than pinned at size 0".

Dropping is right for the BUDGET -- charging for protection nobody gets would
be the #715 error. What is wrong is that the drop is SILENT: nothing compares
what was requested against what was pinned. So a reference that disappeared
between the two checks, or whose metadata-cache entry outlived the file,
produces a checkpoint that reports success while protecting less than it
claims. The failure then surfaces at BRANCH -- "a reference was evicted,
refusing to branch" -- which is precisely the outcome slice 2 exists to
eliminate, arriving later and from further away.

This is the same class as the two failures slice 2's own message names: #698
was a trigger that never fired, slice 2's ledger was a protection never taken.
This is a protection PARTIALLY taken and reported as whole.

The fix does not try to make a deleted page survive -- nothing can. It moves
the refusal to CREATE, where the caller can still do something about it, and
names the references it could not pin.
"""

from __future__ import annotations

import os
import tempfile

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.session_checkpoint import (
    PinCoverageIncomplete,
    create_checkpoint,
)
from sglang.srt.mem_cache.session_manifest import build_manifest
from sglang.test.test_utils import CustomTestCase

IDENTITY = "sha256:410coverage"
PAGE = torch.arange(64, dtype=torch.uint8)


def _backend(root, *, extra=None):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen3.6-27B",
            model_identity_hash=IDENTITY,
            extra_config=extra,
        ),
        file_path=root,
    )


def _manifest(session_id, keys):
    return build_manifest(
        model_identity=IDENTITY,
        session_id=session_id,
        page_hashes=list(keys),
        requested_token_count=len(keys),
        gdn_blob_key=None,
        checkpoint_interval=None,
        is_hybrid_model=False,
    )


class TestPinCoverage(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _store_with_metadata_cache(self):
        # The metadata cache is what lets `exists` and the filesystem disagree.
        return _backend(self.root, extra={"enable_metadata_cache": "true"})

    def _delete_underlying_file(self, store, key):
        """Remove a page's file WITHOUT telling the metadata cache.

        This is not a contrived state: the evictor invalidates the cache when
        IT removes a file, but an out-of-band removal (an orphan reaper, an
        operator, a cleanup job, or a cache entry that outlives its file) does
        not. The point is that `exists` then answers from the cache.
        """
        stem = store._get_suffixed_key(key)
        path = store._existing_path(stem)
        os.remove(path)
        return path

    def test_the_setup_really_does_make_exists_lie(self):
        """Guard the guard: if `exists` told the truth the case is vacuous."""
        store = self._store_with_metadata_cache()
        store.set("aa_gone", PAGE)
        self.assertTrue(store.exists("aa_gone"))
        self._delete_underlying_file(store, "aa_gone")
        # The file is gone; `exists` still says yes, from the cache.
        self.assertTrue(
            store.exists("aa_gone"),
            "metadata cache did not retain the entry -- this suite's premise "
            "does not hold on this build and the tests below prove nothing",
        )

    def test_a_dropped_pin_is_refused_at_create_not_discovered_at_branch(self):
        store = self._store_with_metadata_cache()
        for key in ("aa_one", "aa_two", "aa_three"):
            store.set(key, PAGE)
        self._delete_underlying_file(store, "aa_two")

        manifest = _manifest("s1", ["aa_one", "aa_two", "aa_three"])
        with self.assertRaises(PinCoverageIncomplete) as caught:
            create_checkpoint(store, "cp1", manifest)
        message = str(caught.exception)
        self.assertIn("aa_two", message, "the refusal must name what it could not pin")

    def test_the_refused_create_leaves_no_manifest_and_no_pins(self):
        store = self._store_with_metadata_cache()
        for key in ("aa_one", "aa_two"):
            store.set(key, PAGE)
        self._delete_underlying_file(store, "aa_two")

        with self.assertRaises(PinCoverageIncomplete):
            create_checkpoint(store, "cp2", _manifest("s2", ["aa_one", "aa_two"]))

        # A half-created checkpoint is worse than none: it would hold pins for
        # a manifest that does not exist and cannot be branched from.
        self.assertEqual(store.pin_stats()["pinned_bytes"], 0)
        self.assertNotIn("cp2", list_ids(store))

    def test_a_whole_checkpoint_still_creates_and_pins(self):
        """The default path must be untouched by the new check."""
        store = self._store_with_metadata_cache()
        for key in ("aa_one", "aa_two"):
            store.set(key, PAGE)
        record = create_checkpoint(store, "cp_ok", _manifest("s3", ["aa_one", "aa_two"]))
        self.assertTrue(record.pinned)
        self.assertGreater(store.pin_stats()["pinned_bytes"], 0)

    def test_an_unpinned_checkpoint_skips_the_coverage_check(self):
        """pin=False is first-class; it promises nothing, so it checks nothing."""
        store = self._store_with_metadata_cache()
        for key in ("aa_one", "aa_two"):
            store.set(key, PAGE)
        self._delete_underlying_file(store, "aa_two")
        record = create_checkpoint(
            store, "cp_nopin", _manifest("s4", ["aa_one", "aa_two"]), pin=False
        )
        self.assertFalse(record.pinned)
        self.assertEqual(store.pin_stats()["pinned_bytes"], 0)

    def test_the_record_reports_full_coverage_when_whole(self):
        store = self._store_with_metadata_cache()
        for key in ("aa_one", "aa_two"):
            store.set(key, PAGE)
        record = create_checkpoint(store, "cp_cov", _manifest("s5", ["aa_one", "aa_two"]))
        self.assertEqual(record.references_pinned, 2)
        self.assertEqual(record.references_requested, 2)


def list_ids(store):
    from sglang.srt.mem_cache.session_checkpoint import list_checkpoints

    return list_checkpoints(store)


if __name__ == "__main__":
    import unittest

    unittest.main()
