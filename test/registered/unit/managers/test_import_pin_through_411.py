# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#411: an imported session's pages must be PINNED before it is seeded.

WHAT CUT 3 LEFT OPEN, in its own words. `NOTE_411_portable_sessions.md` §C5
recorded that the export refusal's advice -- "re-pin the checkpoint (#410 slice
2)" -- named a capability the branch lacked: `mem_cache/pin_ledger.py` did not
exist on that base, so the advice was honest but unactionable. It also filed the
Slice-2 dependency as the one "that decides whether the refusal is rare or
routine".

That dependency is now satisfied on this branch: the pin ledger is ported and
`take_file_tier_pins` exists. So import can finally do what export already
advised, and this suite is that closure.

WHY IMPORT NEEDS ITS OWN PIN, and why verification is not enough.
`import_bundle_and_seed` already materialises the payloads and then runs
`verify_restore` against the store. Verification proves the pages are there NOW.
It does not stop the file evictor reclaiming them a moment later, and an
imported bundle is exactly the case where that matters: the pages are freshly
written, so they are the youngest entries in an LRU -- but they are also
unreferenced by any running request until the seeded session starts using them.
A store under pressure can evict a just-imported session before it is ever read.

ORDERING, and it is the same rule as the checkpoint path. The pin is taken
AFTER verification (there is nothing to pin until the pages are in the store)
and BEFORE seeding. A session that cannot be protected must not be created:
the alternative is a live session whose prefix is already reclaimable, which
fails later, further from the cause, and to a user rather than to an operator.

A target with no file tier is not a failure -- #407 may hold this session
elsewhere -- and is reported unprotected rather than refused, exactly as the
checkpoint path does.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from sglang.srt.managers.session_checkpoint import (
    PinCoverageIncomplete,
    pin_imported_pages,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

IDENTITY = "sha256:411importpin"
PAGE = torch.arange(512, dtype=torch.uint8)


def _store(root):
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
            extra_config={"max_size": str(1 << 30)},
        ),
        file_path=root,
    )


def _manifest(kv_keys, mamba_key=None):
    """The shape `import_bundle` returns: a #261 snapshot under a checkpoint
    envelope. Only the fields this function reads are populated."""
    return {
        "kv_keys": list(kv_keys),
        "mamba_key": mamba_key,
        "checkpoint": {"session_id": "sess-1", "checkpoint_id": "cp-imported"},
    }


class TestImportedPagesArePinned(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_a_whole_import_pins_every_page(self):
        store = _store(self.root)
        keys = [f"aa_i{i}" for i in range(4)]
        for k in keys:
            store.set(k, PAGE)
        pins = pin_imported_pages(store, "cp-imported", _manifest(keys))
        self.assertEqual(pins.requested, 4)
        self.assertEqual(pins.pinned, 4)
        self.assertTrue(pins.protected)
        self.assertGreater(store.pin_stats()["pinned_bytes"], 0)

    def test_the_gdn_blob_is_pinned_too(self):
        """#212: a KV-only prefix is worth zero usable tokens on a hybrid
        model, so the mamba blob is part of what must survive, not an extra."""
        store = _store(self.root)
        keys = ["aa_g0", "aa_g1"]
        for k in keys:
            store.set(k, PAGE)
        store.set("aa_mamba", PAGE)
        pins = pin_imported_pages(
            store, "cp-hybrid", _manifest(keys, mamba_key="aa_mamba")
        )
        self.assertEqual(pins.requested, 3)
        self.assertEqual(pins.pinned, 3)

    def test_a_page_that_did_not_land_refuses_by_name(self):
        store = _store(self.root)
        keys = ["aa_x0", "aa_x1", "aa_x2"]
        for k in keys:
            store.set(k, PAGE)
        os.remove(store._pin_path(store._get_suffixed_key("aa_x1")))
        with self.assertRaises(PinCoverageIncomplete) as caught:
            pin_imported_pages(store, "cp-partial", _manifest(keys))
        self.assertIn("aa_x1", str(caught.exception))

    def test_the_refusal_leaves_no_pins_behind(self):
        store = _store(self.root)
        keys = ["aa_y0", "aa_y1"]
        for k in keys:
            store.set(k, PAGE)
        os.remove(store._pin_path(store._get_suffixed_key("aa_y1")))
        with self.assertRaises(PinCoverageIncomplete):
            pin_imported_pages(store, "cp-rollback", _manifest(keys))
        self.assertEqual(store.pin_stats()["pinned_bytes"], 0)


class TestNoFileTierIsReportedNotRefused(CustomTestCase):
    def test_a_target_without_a_file_tier(self):
        pins = pin_imported_pages(None, "cp-none", _manifest(["a", "b"]))
        self.assertFalse(pins.protected)
        self.assertEqual(pins.pinned, 0)
        self.assertIn("no file tier", pins.reason)


class TestTheImportPathTakesThePin(CustomTestCase):
    """The pin must be wired, and wired in the right ORDER."""

    def _src(self):
        import inspect

        from sglang.srt.managers.session_checkpoint import SessionCheckpointRuntime

        return inspect.getsource(SessionCheckpointRuntime.import_bundle_and_seed)

    def test_the_import_path_pins(self):
        self.assertIn("pin_imported_pages", self._src())

    def test_the_pin_precedes_the_seed(self):
        # Match the CALL, not the name: the docstring mentions branch_from
        # while describing the order, and an index() on the bare name finds
        # the prose rather than the statement. A first version of this test
        # did exactly that and failed against correct code.
        src = self._src()
        self.assertLess(
            src.index("pin_imported_pages("),
            src.index("controller.branch_from("),
            "a session that cannot be protected must not be created first",
        )

    def test_the_pin_follows_verification(self):
        src = self._src()
        self.assertLess(
            src.index("verify_restore"),
            src.index("pin_imported_pages"),
            "there is nothing to pin until the pages are in the store",
        )


if __name__ == "__main__":
    unittest.main()
