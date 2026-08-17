# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The file evictor must charge ALLOCATED bytes, not apparent ones.

WHY THIS IS A CORRECTNESS SLICE AND NOT A TIDY-UP. A filesystem charges the
blocks it allocated, not the length the file reports. On ZFS -- this rig's
filesystem, and the one behind the original incident -- a 64-byte page occupies
**512** allocated bytes, an 8x divergence. The incident that produced
``_allocated_size`` measured 512-byte pages occupying 8704 bytes each, a 17x
undercount across 5.8 million files: a store that believes it is using a
seventeenth of its real disk does not evict when it must, and fills the volume.

Larger files on ZFS report ``st_blocks == 1`` because of delayed allocation --
the write has not reached a transaction group yet -- which is exactly why
``_allocated_size`` is ``max(st_blocks * 512, st_size)`` and not ``st_blocks *
512``. Both halves of that max are load-bearing and both are exercised below.

THE SECOND REASON, which is what made this urgent. #410's pin ledger charges
ALLOCATED bytes. While this evictor charged apparent ones, ``reclaimable =
used_bytes - pinned_bytes`` subtracted two different units and could go
negative -- the store reported more bytes pinned than used. That is the #715
shape: not a wrong number, but two right numbers that cannot be combined. The
ledger's unit is the corroborated-correct one, so the evictor moves to it.

THE TEST THAT MATTERS is the first one: it does not assert an accounting field,
it asserts that an EVICTION DECISION CHANGES. A test that only checked
``used_bytes`` would pass against an evictor that computed the right number and
still evicted on the wrong one.
"""

from __future__ import annotations

import os
import tempfile

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor
from sglang.test.test_utils import CustomTestCase

IDENTITY = "sha256:410alloc"

#: Small enough that ZFS charges a whole 512-byte block for it. The divergence
#: this suite is about only exists for files below the block size, which is
#: precisely the regime the incident lived in.
TINY = torch.arange(64, dtype=torch.uint8)


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


class TestTheDivergenceIsRealHere(CustomTestCase):
    """Guard the guard: on a filesystem with no divergence this suite is vacuous."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_a_tiny_file_costs_more_than_it_reports(self):
        path = os.path.join(self.root, "probe.bin")
        with open(path, "wb") as f:
            f.write(b"x" * 64)
        st = os.stat(path)
        allocated = LRUFileEvictor._allocated_size(st)
        self.assertGreater(
            allocated,
            st.st_size,
            "this filesystem does not over-allocate small files, so the "
            "eviction-decision test below proves nothing here",
        )

    def test_the_max_protects_delayed_allocation(self):
        # A file whose blocks have not been assigned yet must never be charged
        # BELOW its payload -- st_blocks alone would report almost nothing.
        class _Delayed:
            st_blocks = 1
            st_size = 65536

        self.assertEqual(LRUFileEvictor._allocated_size(_Delayed()), 65536)

    def test_blocks_win_when_they_exceed_the_payload(self):
        class _Padded:
            st_blocks = 17
            st_size = 512

        self.assertEqual(LRUFileEvictor._allocated_size(_Padded()), 17 * 512)


class TestEvictionDecisionFollowsAllocatedBytes(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _tiny_allocated_cost(self) -> int:
        probe = os.path.join(self.root, "cost_probe.bin")
        with open(probe, "wb") as f:
            f.write(b"x" * int(TINY.numel()))
        cost = LRUFileEvictor._allocated_size(os.stat(probe))
        os.remove(probe)
        return cost

    def test_a_store_that_is_full_by_allocated_bytes_evicts(self):
        """THE POINT. Under apparent accounting this store believes it is
        nearly empty and evicts nothing; under allocated accounting it is over
        its cap and must evict. The decision itself flips."""
        cost = self._tiny_allocated_cost()
        room_for = 3
        store = _backend(
            self.root,
            extra={"max_size": str(room_for * cost), "eviction_ratio": 1.0},
        )
        for i in range(6):
            store.set(f"aa_page{i:02d}", TINY)

        remaining = [
            fn
            for fn in os.listdir(self.root)
            if fn.endswith(".bin") and fn.startswith("aa_page")
        ]
        self.assertLessEqual(
            len(remaining),
            room_for,
            f"wrote 6 tiny pages into a store sized for {room_for} ALLOCATED "
            f"pages ({cost} B each) and {len(remaining)} survived -- the store "
            "is accounting apparent bytes and therefore never noticed it was "
            "full",
        )

    def test_used_bytes_are_charged_in_allocated_units(self):
        cost = self._tiny_allocated_cost()
        store = _backend(self.root, extra={"max_size": str(100 * cost)})
        store.set("aa_only", TINY)
        used = store.capacity_stats()["used_bytes"]
        self.assertGreaterEqual(
            used,
            cost,
            "one tiny page must be charged its allocated cost, not its length",
        )


class TestPinAccountingIsNowCoherent(CustomTestCase):
    """The blocker this slice exists to clear."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_pinned_bytes_never_exceed_used_bytes(self):
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        for i in range(4):
            store.set(f"aa_p{i}", TINY)
        store.pin_checkpoint("ckpt", [f"aa_p{i}" for i in range(4)])
        stats = store.capacity_stats()
        self.assertGreater(stats["pinned_bytes"], 0)
        self.assertEqual(
            stats["accounting_overshoot_bytes"],
            0,
            "the ledger and the evictor are still charging different units",
        )
        self.assertLessEqual(stats["pinned_bytes"], stats["used_bytes"])

    def test_reclaimable_is_used_minus_pinned(self):
        store = _backend(self.root, extra={"max_size": str(1 << 30)})
        for i in range(4):
            store.set(f"aa_q{i}", TINY)
        store.pin_checkpoint("ckpt", ["aa_q0", "aa_q1"])
        stats = store.capacity_stats()
        self.assertEqual(
            stats["reclaimable_bytes"],
            stats["used_bytes"] - stats["pinned_bytes"],
            "the subtraction is only meaningful once both sides share a unit",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
