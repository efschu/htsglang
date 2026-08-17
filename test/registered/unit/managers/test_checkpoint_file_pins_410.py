# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A checkpoint must take FILE-TIER pins, and say so when it cannot.

WHAT THIS LINEAGE ALREADY HAD, and why it was not enough.
``SessionCheckpointRuntime._pin`` does ``tree.inc_lock_ref(node)``: it locks the
RADIX chain, in memory, for the life of the process. That is real protection and
it is the right protection for a running server -- but it is the wrong tier for
a checkpoint, whose whole promise is that a conversation can be branched later.

**An A checkpoint survives radix eviction and does not survive file-tier
eviction.** The pages it references live in the HiCache file store as ordinary
LRU entries; nothing stops the evictor reclaiming them, and nothing survives a
restart. The pin ledger ported in this branch is the tier that was missing, and
until the runtime actually calls it the protection is a component nobody
invokes -- the exact shape #410 slice 2 was written about, arriving one layer up.

WHAT "COVERAGE" MEANS HERE. Pinning is only meaningful for references the store
actually holds. ``stems_with_sizes`` drops a stem whose file is absent, which is
right for the budget and silent to the caller, so a checkpoint could pin four of
its six pages and report success. The two missing pages would then be evicted
and the failure would surface at the BRANCH -- later, and further from the
cause. So the question this asks at checkpoint time is exactly: **are this
checkpoint's pages on the file tier at all?**

NOT EVERY CHECKPOINT LIVES ON THE FILE TIER. #407 may place one on vram or host
(``TierKind.DEVICE`` / ``HOST``), where there are no file stems to pin. That is
not a failure and must not raise: it is a checkpoint with no file-tier
protection, and the honest thing is to record that rather than to invent a pin
or to refuse a placement the tier policy chose.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from sglang.srt.managers.session_checkpoint import (
    PinCoverageIncomplete,
    take_file_tier_pins,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

IDENTITY = "sha256:410filepins"
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


class TestFileTierPins(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_a_whole_checkpoint_pins_every_reference(self):
        store = _store(self.root)
        keys = [f"aa_k{i}" for i in range(4)]
        for k in keys:
            store.set(k, PAGE)
        pins = take_file_tier_pins(store, "cp1", keys)
        self.assertEqual(pins.requested, 4)
        self.assertEqual(pins.pinned, 4)
        self.assertEqual(pins.unpinned, ())
        self.assertGreater(store.pin_stats()["pinned_bytes"], 0)

    def test_a_missing_reference_refuses_by_name(self):
        store = _store(self.root)
        keys = [f"aa_m{i}" for i in range(3)]
        for k in keys:
            store.set(k, PAGE)
        # One page gone before the checkpoint is taken -- the evictor, an
        # operator, or simply a page that never reached the file tier.
        stem = store._get_suffixed_key("aa_m1")
        os.remove(store._pin_path(stem))

        with self.assertRaises(PinCoverageIncomplete) as caught:
            take_file_tier_pins(store, "cp2", keys)
        self.assertIn("aa_m1", str(caught.exception))

    def test_the_refused_checkpoint_leaves_no_pins_behind(self):
        store = _store(self.root)
        keys = ["aa_r0", "aa_r1"]
        for k in keys:
            store.set(k, PAGE)
        os.remove(store._pin_path(store._get_suffixed_key("aa_r1")))
        with self.assertRaises(PinCoverageIncomplete):
            take_file_tier_pins(store, "cp3", keys)
        # A half-pinned checkpoint holds bytes for a promise it cannot keep.
        self.assertEqual(store.pin_stats()["pinned_bytes"], 0)


class TestNoFileTierIsNotAFailure(CustomTestCase):
    """#407 may place a checkpoint on vram or host. That is not an error."""

    def test_a_store_of_none_reports_no_protection(self):
        pins = take_file_tier_pins(None, "cp4", ["a", "b"])
        self.assertEqual(pins.requested, 2)
        self.assertEqual(pins.pinned, 0)
        self.assertFalse(pins.protected)
        self.assertIn("no file tier", pins.reason)

    def test_a_backend_without_pinning_reports_no_protection(self):
        class _Old:  # a backend from before the pin ledger existed
            pass

        pins = take_file_tier_pins(_Old(), "cp5", ["a"])
        self.assertFalse(pins.protected)
        self.assertIn("does not support", pins.reason)

    def test_an_empty_reference_set_is_protected_vacuously(self):
        pins = take_file_tier_pins(None, "cp6", [])
        self.assertEqual(pins.requested, 0)
        self.assertTrue(pins.protected)


class TestTheRuntimeCallsIt(CustomTestCase):
    """A pin function nobody calls is the defect this whole slice is about."""

    def test_checkpoint_takes_file_tier_pins(self):
        import inspect

        from sglang.srt.managers.session_checkpoint import SessionCheckpointRuntime

        src = inspect.getsource(SessionCheckpointRuntime._checkpoint)
        self.assertIn("take_file_tier_pins", src)

    def test_the_runtime_can_reach_a_store(self):
        import inspect

        from sglang.srt.managers.session_checkpoint import SessionCheckpointRuntime

        src = inspect.getsource(SessionCheckpointRuntime._file_tier_store)
        self.assertIn("storage_backend", src)


if __name__ == "__main__":
    unittest.main()
