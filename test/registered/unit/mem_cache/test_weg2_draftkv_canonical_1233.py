# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec WEG2_DRAFTKV_SPEC_0907 T1-T7).

The canonical DRAFT page is the whole 2048 B token page of the single NEXTN
draft layer (K half, V half; every kv head), written whole by group P (TP=1)
and cut on read by head extents on group D (TP=3, head shares 2/1/1). The
store's completeness marker (``read_extents`` all-or-nothing) IS the validity
bit -- no per-page flag, no zeros written by the store.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageError, CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    build_draft_window,
    read_extents,
    window_for_layers,
    write_extents,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

HEAD_DIM = 256
TOTAL_HEADS = 4
PAGE_BYTES = 2 * TOTAL_HEADS * HEAD_DIM  # fp8: 1 B per element -> 2048


class FakeDraftHostPool:
    """The three attributes ``build_draft_window`` reads plus the byte count
    the host pool itself reports (``pool_host/mha.py get_size_per_token``)."""

    def __init__(self, head_num, head_dim=HEAD_DIM, layer_num=1):
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.dtype = torch.uint8

    def get_size_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2


SHARES = ((0, 2), (2, 1), (3, 1))  # local_head_window(4, 3, r): offsets 0/2/3


def _rank_window(rank):
    off, n = SHARES[rank]
    return build_draft_window(
        FakeDraftHostPool(n), TOTAL_HEADS, off, n, tp_size=3, tp_rank=rank
    )


def _whole_page(tag=1):
    # K region: head h carries byte value tag+h; V region: tag+h+16.
    buf = bytearray()
    for shift in (0, 16):
        for h in range(TOTAL_HEADS):
            buf += bytes([(tag + h + shift) % 256]) * HEAD_DIM
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _slice(page, window):
    out = bytearray()
    for off, n in window.extents:
        out += bytes(page[off : off + n].tolist())
    return bytes(out)


class TestDraftPageSpec(CustomTestCase):
    def test_t1_spec_of_the_draft_page(self):
        spec = CanonicalPageSpec(1, PAGE_BYTES)
        self.assertEqual(spec.page_bytes, 2048)
        self.assertEqual(spec.half_cell_bytes, 1024)
        self.assertEqual(spec.slot_spans(0), ((0, 1024), (1024, 2048)))

    def test_t2_windows_partition_the_page_exactly_once(self):
        wins = [_rank_window(r) for r in range(3)]
        self.assertEqual([w.payload_bytes for w in wins], [1024, 512, 512])
        covered = []
        for w in wins:
            self.assertEqual(w.total_bytes, PAGE_BYTES)
            covered.extend((off, off + n) for off, n in w.extents)
        covered.sort()
        self.assertEqual(covered[0][0], 0)
        self.assertEqual(covered[-1][1], PAGE_BYTES)
        for (a_lo, a_hi), (b_lo, _b_hi) in zip(covered, covered[1:]):
            self.assertEqual(a_hi, b_lo, "gap or overlap between head extents")
        whole = build_draft_window(
            FakeDraftHostPool(TOTAL_HEADS), TOTAL_HEADS, 0, TOTAL_HEADS, tp_size=1, tp_rank=0
        )
        self.assertTrue(whole.is_whole)
        self.assertEqual(whole.extents, ((0, PAGE_BYTES),))

    def test_t3_refusals_payload_and_head_window(self):
        with self.assertRaises(CanonicalPageError):
            # pool says 1024 B/token but the window would cut 512
            build_draft_window(FakeDraftHostPool(2), TOTAL_HEADS, 2, 1, tp_size=3, tp_rank=1)
        with self.assertRaisesRegex(CanonicalPageError, "head window mismatch"):
            # offset 1 for rank 1 disagrees with local_head_window (2, 3)
            build_draft_window(FakeDraftHostPool(1), TOTAL_HEADS, 1, 1, tp_size=3, tp_rank=1)


class TestDraftPageStore(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "cafe.draft-abc.bin")

    def test_t4_whole_write_then_head_cut_reads(self):
        page = _whole_page()
        whole = build_draft_window(FakeDraftHostPool(4), TOTAL_HEADS, 0, 4, tp_size=1, tp_rank=0)
        res = write_extents(self.path, whole, page)
        self.assertTrue(res.completed)
        for r in range(3):
            w = _rank_window(r)
            out = torch.zeros(w.payload_bytes, dtype=torch.uint8)
            self.assertTrue(read_extents(self.path, w, out))
            self.assertEqual(bytes(out.tolist()), _slice(page, w))

    def test_t5_partial_page_is_not_served_and_untouched(self):
        page = _whole_page()
        w1 = _rank_window(1)
        write_extents(self.path, w1, torch.frombuffer(_slice(page, w1), dtype=torch.uint8).clone())
        self.assertFalse(os.path.exists(self.path))
        for r in range(3):
            w = _rank_window(r)
            out = torch.full((w.payload_bytes,), 0xEE, dtype=torch.uint8)
            self.assertFalse(read_extents(self.path, w, out))
            self.assertTrue(bool((out == 0xEE).all()), "the store must not write zeros")

    def test_t6_three_d_windows_in_any_order_publish_the_page(self):
        page = _whole_page(tag=40)
        for r in (2, 0, 1):
            w = _rank_window(r)
            write_extents(self.path, w, torch.frombuffer(_slice(page, w), dtype=torch.uint8).clone())
        self.assertTrue(os.path.exists(self.path))
        whole = build_draft_window(FakeDraftHostPool(4), TOTAL_HEADS, 0, 4, tp_size=1, tp_rank=0)
        out = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
        self.assertTrue(read_extents(self.path, whole, out))
        self.assertEqual(bytes(out.tolist()), bytes(page.tolist()))


ATTN_LAYER_IDS = list(range(3, 64, 4))
KV_SPEC = CanonicalPageSpec(len(ATTN_LAYER_IDS), 64)


def _config(draft=None):
    kv = window_for_layers(KV_SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
    return HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=True,
        model_name="Qwen3.8-27B", model_identity_hash="0123456789abcdef",
        canonical_kv_page=kv, canonical_draft_page=draft,
    ), kv


class TestInstallThirdSlot(CustomTestCase):
    def test_t7_install_refuses_on_off_and_width_change(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        whole = build_draft_window(FakeDraftHostPool(4), TOTAL_HEADS, 0, 4, tp_size=1, tp_rank=0)
        cfg, kv = _config(draft=whole)
        be = HiCacheFile(cfg, file_path=tmp.name)
        with self.assertRaises(CanonicalPageError):
            be.install_canonical_windows(kv, None, draft_page=None)  # switching off
        other = build_draft_window(
            FakeDraftHostPool(8), 8, 0, 8, tp_size=1, tp_rank=0
        )
        self.assertNotEqual(other.total_bytes, whole.total_bytes)
        with self.assertRaises(CanonicalPageError):
            be.install_canonical_windows(kv, None, draft_page=other)
        be.install_canonical_windows(kv, None, draft_page=_rank_window(1))  # same width
        self.assertEqual(be.canonical_draft_page.payload_bytes, 512)
        cfg2, kv2 = _config(draft=None)
        be2 = HiCacheFile(cfg2, file_path=tmp.name)
        with self.assertRaises(CanonicalPageError):
            be2.install_canonical_windows(kv2, None, draft_page=whole)  # switching on


if __name__ == "__main__":
    unittest.main()
