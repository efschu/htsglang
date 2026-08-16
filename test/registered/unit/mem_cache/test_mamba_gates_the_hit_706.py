"""#706 slice 2, the gating question: is a KV-only prefix hit worth anything on
a GDN-hybrid checkpoint?

It is not, and this file pins that rather than asserting it. Two independent
seams refuse a prefix whose mamba/GDN state is absent:

* STORAGE. ``HiCacheStorage.batch_exists_v2``'s own contract says "the final
  ``final_pages`` is the minimum across all pools, so a missing auxiliary page
  shrinks the usable prefix" (``hicache_storage.py``), and the file backend
  implements it as ``final_pages = min(final_pages, boundary)`` over every
  ``PoolTransfer``. The mamba pool is registered with
  ``PoolHitPolicy.TRAILING_PAGES`` (``hi_mamba_radix_cache.py``, both
  ``mamba_archive_transfers`` and ``mamba_prefetch_alloc``), so the blob for the
  tail page must be present or the boundary is 0 -- and 0 truncates the WHOLE
  KV prefix, however many KV pages were found. That is what the tests below
  exercise, against the real backend.

* DEVICE. ``MambaRadixCache._match_prefix_helper`` advances ``best_value_len``
  and ``best_last_node`` only at nodes where ``node.mamba_value is not None``
  (``mamba_radix_cache.py``). A matched chain whose nodes carry no mamba state
  therefore reports a usable prefix of zero, whatever the KV rows say. Not
  exercised here: constructing that cache needs device pools, and this suite is
  hermetic (CUDA_VISIBLE_DEVICES=""). It is cited, not assumed -- and it agrees
  with the storage seam, which is the one the cross-phase prefetch consults.

CONSEQUENCE, which is the point of the file: making only the KV page
geometry-neutral cannot produce a cross-flip hit on this model. 48 of the 64
layers are GDN. Slice 1 was worth landing on its own -- it is the page format,
and the mamba blob rides the same protocol -- but a phase-local GDN blob is a
finish line nobody crosses.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_CUT = [(0, 28), (28, 48), (48, 64)]
IDENTITY = "0123456789abcdef"
KEYS = ["cafe01", "cafe02", "cafe03"]


def _stage_window(lo, hi):
    return window_for_layers(
        SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _config(pp_rank, window):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=pp_rank,
        pp_size=3,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="Qwen3.6-27B",
        model_identity_hash=IDENTITY,
        canonical_kv_page=window,
    )


def _payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


class TestMambaGatesTheHit(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.stages = [
            HiCacheFile(_config(r, _stage_window(*PP_CUT[r])), file_path=self.root)
            for r in range(3)
        ]
        # A complete canonical KV page for every key: all three stages write.
        for key in KEYS:
            for stage in self.stages:
                stage.set(key, _payload(stage.canonical_kv_page))

    def _mamba_transfer(self):
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=[KEYS[-1]],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def test_kv_pages_are_all_present(self):
        """The premise: this is not a KV miss dressed up as a mamba miss."""
        for key in KEYS:
            self.assertTrue(self.stages[0].exists(key))

    def test_kv_only_prefix_is_truncated_to_nothing(self):
        result = self.stages[0].batch_exists_v2(KEYS, self._mamba_transfer())
        self.assertEqual(result.extra_pool_hit_pages.get(PoolName.KV), len(KEYS))
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertNotIn(PoolName.MAMBA, result.extra_pool_hit_pages)

    def test_the_mamba_blob_is_what_unlocks_it(self):
        """Same store, same KV pages, one blob added: the prefix appears. So the
        whole cross-flip hit rides on the GDN blob being findable, which is why
        slice 2 makes it phase-uniform too."""
        blob = torch.arange(48, dtype=torch.uint8)
        self.assertTrue(self.stages[0].set(f"{KEYS[-1]}.mamba", blob))
        result = self.stages[0].batch_exists_v2(KEYS, self._mamba_transfer())
        self.assertEqual(result.kv_hit_pages, len(KEYS))
        self.assertEqual(result.extra_pool_hit_pages.get(PoolName.MAMBA), len(KEYS))

    def test_a_phase_local_blob_is_invisible_to_the_other_phase(self):
        """The defect slice 2 removes, stated as a test. The blob written by PP
        stage 0 carries that stage's geometry in its key, so a backend on any
        other geometry does not find it -- and by the test above, not finding it
        costs the ENTIRE KV prefix, not just the GDN part."""
        blob = torch.arange(48, dtype=torch.uint8)
        self.stages[0].set(f"{KEYS[-1]}.mamba", blob)
        self.assertTrue(self.stages[0].exists(f"{KEYS[-1]}.mamba"))
        # A different PP stage, i.e. a different pp_rank in the key suffix.
        self.assertFalse(self.stages[1].exists(f"{KEYS[-1]}.mamba"))
        self.assertEqual(
            self.stages[1].batch_exists_v2(KEYS, self._mamba_transfer()).kv_hit_pages,
            0,
        )
        # ... while the KV pages themselves are shared across all of them.
        self.assertTrue(self.stages[1].exists(KEYS[-1]))

    def test_one_blob_per_stage_on_disk_today(self):
        """Three stages, three files, none of which the TP phase can name."""
        for stage in self.stages:
            stage.set(f"{KEYS[-1]}.mamba", torch.arange(48, dtype=torch.uint8))
        blobs = [
            name
            for _d, _s, files in os.walk(self.root)
            for name in files
            if ".mamba" in name
        ]
        self.assertEqual(len(blobs), 3)


if __name__ == "__main__":
    unittest.main()
