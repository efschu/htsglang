# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T8-T11): the draft KEY.

Under the canonical draft window the draft key loses the same geometry terms
the KV page loses (``_{tp_rank}_{tp_size}`` and ``_{pp_size}_{pp_rank}``): a
PP stage of group P and a TP rank of group D name the SAME file for the same
page hash and drafter identity. Without the window the key is byte-identical
to today's (backward compatibility). The presence probe answers the draft
prefix as an ALL_PAGES pool that never caps the KV claim.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import build_draft_window, window_for_layers
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolTransfer,
)
from sglang.test.test_utils import CustomTestCase

IDENTITY = "0123456789abcdef"
DRAFTER = "a30db4b7c362c786"
ATTN_LAYER_IDS = list(range(3, 64, 4))
KV_SPEC = CanonicalPageSpec(len(ATTN_LAYER_IDS), 64)
HEAD_DIM = 256
TOTAL_HEADS = 4
SHARES = ((0, 2), (2, 1), (3, 1))


class FakeDraftHostPool:
    def __init__(self, head_num, head_dim=HEAD_DIM, layer_num=1):
        self.head_num, self.head_dim, self.layer_num = head_num, head_dim, layer_num
        self.dtype = torch.uint8

    def get_size_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * 2


def _draft_window(tp_size, tp_rank):
    if tp_size == 1:
        return build_draft_window(FakeDraftHostPool(4), TOTAL_HEADS, 0, 4, tp_size=1, tp_rank=0)
    off, n = SHARES[tp_rank]
    return build_draft_window(FakeDraftHostPool(n), TOTAL_HEADS, off, n, tp_size=3, tp_rank=tp_rank)


def _config(*, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, kv=True, draft=None):
    kv_window = window_for_layers(KV_SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS) if kv else None
    return HiCacheStorageConfig(
        tp_rank=tp_rank, tp_size=tp_size, pp_rank=pp_rank, pp_size=pp_size,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
        enable_storage_metrics=False, is_page_first_layout=True,
        model_name="Qwen3.8-27B", model_identity_hash=IDENTITY,
        canonical_kv_page=kv_window, canonical_draft_page=draft,
    )


class TestDraftKey(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _backend(self, **kw):
        return HiCacheFile(_config(**kw), file_path=self.root)

    def test_t8_is_draft_key_recognises_identity_suffixed_keys(self):
        be = self._backend()
        self.assertTrue(be._is_draft_key(f"abc.draft-{DRAFTER}"))
        self.assertTrue(be._is_draft_key("abc.draft"))
        self.assertFalse(be._is_draft_key("abc"))
        self.assertFalse(be._is_draft_key("abc.mamba"))

    def test_t9_key_without_window_is_todays_key_with_window_geometry_free(self):
        legacy = self._backend(tp_rank=2, tp_size=3, pp_rank=1, pp_size=3)
        self.assertEqual(
            legacy._get_suffixed_key(f"cafe.draft-{DRAFTER}"),
            f"cafe.draft-{DRAFTER}_Qwen3.8-27B_{IDENTITY}_2_3_3_1",
        )
        p_stage = self._backend(tp_rank=0, tp_size=1, pp_rank=2, pp_size=3, draft=_draft_window(1, 0))
        d_keys = {
            self._backend(tp_rank=r, tp_size=3, draft=_draft_window(3, r))._get_suffixed_key(
                f"cafe.draft-{DRAFTER}"
            )
            for r in range(3)
        }
        self.assertEqual(len(d_keys), 1)
        key = d_keys.pop()
        self.assertEqual(p_stage._get_suffixed_key(f"cafe.draft-{DRAFTER}"), key)
        self.assertEqual(key, f"cafe.draft-{DRAFTER}_Qwen3.8-27B_{IDENTITY}")
        for bad in ("_0_1_3_2", "_0_3", "_1_3", "_2_3", "_3_2"):
            self.assertNotIn(bad, key)
        # the KV key of the same backends is unchanged by the draft window
        self.assertEqual(p_stage._get_suffixed_key("cafe"), f"cafe_Qwen3.8-27B_{IDENTITY}")

    def test_t10_window_dispatch_and_scan_suffixes(self):
        with_window = self._backend(tp_rank=1, tp_size=3, draft=_draft_window(3, 1))
        self.assertIs(with_window._canonical_window(f"cafe.draft-{DRAFTER}"), with_window.canonical_draft_page)
        self.assertIsNot(with_window._canonical_window(f"cafe.draft-{DRAFTER}"), with_window._canonical_kv_extents)
        self.assertIn(with_window.draft_config_suffix, with_window._group_scan_suffixes())
        without = self._backend(tp_rank=1, tp_size=3)
        self.assertIsNone(without._canonical_window(f"cafe.draft-{DRAFTER}"))
        self.assertIsNotNone(without._canonical_window("cafe"))

    def _fill(self, be, n_kv, draft_present):
        keys = [f"{i:02x}{'0' * 62}" for i in range(n_kv)]
        kv_page = torch.full((KV_SPEC.page_bytes,), 3, dtype=torch.uint8)
        draft_page = torch.full((2048,), 5, dtype=torch.uint8)
        for i, k in enumerate(keys):
            self.assertTrue(be.set(k, kv_page))
            if i in draft_present:
                self.assertTrue(be.set(f"{k}.draft-{DRAFTER}", draft_page))
        return keys

    def test_t11_presence_probe_answers_the_draft_prefix_without_capping_kv(self):
        be = self._backend(draft=_draft_window(1, 0))
        keys = self._fill(be, 10, set(range(8)))
        xfer = PoolTransfer(name=f"draft-{DRAFTER}", keys=[], hit_policy=PoolHitPolicy.ALL_PAGES, caps_claim=False)
        res = be.batch_exists_v2(keys, [xfer])
        self.assertEqual(res.kv_hit_pages, 10)
        self.assertEqual(res.extra_pool_hit_pages[f"draft-{DRAFTER}"], 8)
        hole = self._backend(draft=_draft_window(1, 0))
        # fresh directory for the hole shape
        hole.file_path = os.path.join(self.root, "hole")
        os.makedirs(hole.file_path, exist_ok=True)
        keys = self._fill(hole, 10, {0, 1, 2, 4, 5, 6, 7, 8, 9})
        res = hole.batch_exists_v2(keys, [xfer])
        self.assertEqual(res.kv_hit_pages, 10)
        self.assertEqual(res.extra_pool_hit_pages[f"draft-{DRAFTER}"], 3)
        # a capping transfer of the same pool still caps (upstream semantics kept)
        capping = PoolTransfer(name=f"draft-{DRAFTER}", keys=[], hit_policy=PoolHitPolicy.ALL_PAGES)
        self.assertEqual(hole.batch_exists_v2(keys, [capping]).kv_hit_pages, 3)


if __name__ == "__main__":
    unittest.main()
