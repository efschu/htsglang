"""#239 S4b (F14), part 2: the paged owner form through the real file backend
and the controller's attach.

Form A x the token cut on NF D: page 64, S = 64, the attention host (TP0)
usually owns share 0, the two expert workers own the token rows. Before this
part the owner mode refused ``page_size != 1`` at attach (every rank of such a
group died there), a worker rode the null tier, and the store backup masked
WHOLE pages by owner -- a page of 64 tokens has two owners, so the mask was
wrong on both of them.

RED on abdd7ec919: ``HiCacheStorageConfig`` has no ``canonical_kv_owner_rows``,
the backend has no owner-row window or abstention, the controller has no
``page_owner_mask_ctx`` / ``_canonical_kv_owner_rows``, attach refuses page 64.
"""

from __future__ import annotations

import tempfile
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalAbstainWindow,
    CanonicalPageWindow,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

PAGE = 64
LAYERS = 4
ROW = 8  # bytes of one token row of one layer in one K/V half
SPEC = CanonicalPageSpec(num_attn_layers=LAYERS, kv_bytes_per_token_per_attn_layer=2 * PAGE * ROW)
WHOLE = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=LAYERS)
IDENTITY = "feedfacecafebeef"
HOST, W1, W2 = (PAGE, 64, 0, 0), (PAGE, 64, 0, 46), (PAGE, 64, 46, 64)


def _config(owner_rows, tp_rank):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=3,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="Qwen3.8-Flash-Next",
        model_identity_hash=IDENTITY,
        dcp_owner_mode=True,
        canonical_kv_page=WHOLE,
        canonical_kv_owner_rows=owner_rows,
    )


def _truth(kv, layer, tok):
    return (kv * 101 + layer * 17 + tok) % 251


def _page(fill):
    t = torch.empty((2, LAYERS, PAGE, ROW), dtype=torch.uint8)
    for kv in range(2):
        for layer in range(LAYERS):
            for tok in range(PAGE):
                t[kv, layer, tok, :] = fill(kv, layer, tok)
    return t.flatten()


class TestOwnerRowBackend(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.host = HiCacheFile(_config(HOST, 0), file_path=self.root)
        self.w1 = HiCacheFile(_config(W1, 1), file_path=self.root)
        self.w2 = HiCacheFile(_config(W2, 2), file_path=self.root)

    def test_windows_per_rank(self):
        self.assertIsInstance(self.host._canonical_kv_extents, CanonicalAbstainWindow)
        self.assertTrue(self.w1._canonical_kv_extents.identity)
        self.assertEqual(self.w1._canonical_kv_extents.payload_bytes, 2 * LAYERS * 46 * ROW)
        self.assertEqual(self.w2._canonical_kv_extents.payload_bytes, 2 * LAYERS * 18 * ROW)
        # one key on every rank: content only
        self.assertEqual(
            {b._get_suffixed_key("abc") for b in (self.host, self.w1, self.w2)}.__len__(), 1
        )

    def test_the_page_completes_with_both_owners_and_the_host_writes_nothing(self):
        truth = _page(_truth)
        r1 = _page(lambda kv, l, t: _truth(kv, l, t) if t < 46 else 255)
        r2 = _page(lambda kv, l, t: _truth(kv, l, t) if t >= 46 else 254)
        self.assertTrue(self.host.set("p0", torch.zeros(0, dtype=torch.uint8)))
        self.assertFalse(self.host.exists("p0"))
        self.assertTrue(self.w1.set("p0", r1))
        self.assertFalse(self.w1.exists("p0"))  # half a page is invisible
        self.assertTrue(self.w2.set("p0", r2))
        for b in (self.host, self.w1, self.w2):
            self.assertTrue(b.exists("p0"))
        # each owner reads back ITS rows into a full-width host page
        out = torch.full_like(truth, 9)
        self.assertIs(self.w2.get("p0", out), out)
        got, ref = out.view(2, LAYERS, PAGE, ROW), truth.view(2, LAYERS, PAGE, ROW)
        self.assertTrue(torch.equal(got[:, :, 46:], ref[:, :, 46:]))
        self.assertTrue(bool((got[:, :, :46] == 9).all()))
        # the host's read is served without touching its buffer
        sentinel = torch.full((3,), 5, dtype=torch.uint8)
        self.assertIs(self.host.get("p0", sentinel), sentinel)
        self.assertTrue(bool((sentinel == 5).all()))

    def test_the_batched_paths_take_the_identity_and_abstain_routes(self):
        r1 = _page(lambda kv, l, t: _truth(kv, l, t) if t < 46 else 255)
        r2 = _page(lambda kv, l, t: _truth(kv, l, t) if t >= 46 else 254)
        self.assertTrue(self.w1.batch_set(["q0", "q1"], [r1, r1]))
        self.assertTrue(self.host.batch_set(["q0", "q1"], [torch.zeros(0, dtype=torch.uint8)] * 2))
        self.assertTrue(self.w2.batch_set(["q0", "q1"], [r2, r2]))
        outs = [torch.zeros_like(r1), torch.zeros_like(r1)]
        res = self.w1.batch_get(["q0", "q1"], outs)
        self.assertTrue(all(r is not None for r in res))
        ref = _page(_truth).view(2, LAYERS, PAGE, ROW)
        for o in outs:
            self.assertTrue(torch.equal(o.view(2, LAYERS, PAGE, ROW)[:, :, :46], ref[:, :, :46]))
        host_out = [torch.zeros(1, dtype=torch.uint8)] * 2
        self.assertTrue(all(r is not None for r in self.host.batch_get(["q0", "q1"], host_out)))

    def test_without_owner_rows_the_window_is_the_old_one(self):
        b = HiCacheFile(_config(None, 0), file_path=self.root)
        self.assertEqual(b._canonical_kv_extents, WHOLE.as_extents())


class TestControllerOwnerRows(unittest.TestCase):
    def _ctl(self, ctx, page_size=PAGE, owner_rows=None):
        return types.SimpleNamespace(
            _dcp_owner_ctx=lambda: ctx,
            page_size=page_size,
            storage_config=types.SimpleNamespace(canonical_kv_owner_rows=owner_rows),
        )

    def test_owner_rows_only_for_the_paged_owner_form(self):
        from sglang.srt.managers.cache_controller import canonical_kv_owner_rows_for as f

        self.assertEqual(f((64, 46, 64), PAGE, WHOLE), (64, 64, 46, 64))
        self.assertEqual(f((64, 0, 0), PAGE, WHOLE), (64, 64, 0, 0))
        self.assertIsNone(f((64, 0, 46), 1, WHOLE))
        self.assertIsNone(f(None, PAGE, WHOLE))
        self.assertIsNone(f((64, 0, 46), PAGE, None))

    def test_no_page_mask_under_the_paged_owner_form(self):
        from sglang.srt.managers.cache_controller import HiCacheController as H

        self.assertEqual(H.page_owner_mask_ctx(self._ctl((3, 1, 2))), (3, 1, 2))
        self.assertIsNone(H.page_owner_mask_ctx(self._ctl((64, 0, 46), owner_rows=W1)))
        self.assertIsNone(H.page_owner_mask_ctx(self._ctl(None)))


class _StopAfterChecks(Exception):
    pass


class TestAttachLiftsPage1OnlyForOwnerRows(unittest.TestCase):
    def _attach(self, owner_rows, *, worker=False, holds_kv=False):
        from sglang.srt.managers import cache_controller as cc
        from sglang.srt import rank_role

        ctl = object.__new__(cc.HiCacheController)
        ctl.enable_storage = False
        ctl.page_size = PAGE
        ctl.mem_pool_host = object()
        ctl._stop_storage_threads = lambda: None
        ctl._destroy_prefetch_sync_groups = lambda: None
        ctl._generate_storage_config = lambda *a, **k: types.SimpleNamespace(
            host_role="retention",
            dcp_owner_mode=True,
            canonical_kv_page=WHOLE,
            canonical_kv_owner_rows=owner_rows,
            is_mla_model=False,
            tp_rank=1,
        )
        made = []

        def _create(*a, **k):
            made.append("factory")
            raise _StopAfterChecks()

        null = mock.MagicMock(side_effect=_StopAfterChecks)
        with mock.patch(
            "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend", _create
        ), mock.patch.object(rank_role, "this_rank_is_form_a_worker", lambda: worker), mock.patch.object(
            rank_role, "form_a_worker_holds_kv", lambda: holds_kv
        ), mock.patch(
            "sglang.srt.mem_cache.hicache_storage.FormAWorkerNullStorage", null
        ):
            with self.assertRaises(_StopAfterChecks):
                ctl.attach_storage_backend("file")
        return made, null

    def test_page64_owner_mode_passes_the_checks_with_owner_rows(self):
        made, _ = self._attach(W1)
        self.assertEqual(made, ["factory"])

    def test_page64_owner_mode_without_owner_rows_is_still_refused(self):
        from sglang.srt.managers import cache_controller as cc

        ctl = object.__new__(cc.HiCacheController)
        ctl.enable_storage = False
        ctl.page_size = PAGE
        ctl._stop_storage_threads = lambda: None
        ctl._generate_storage_config = lambda *a, **k: types.SimpleNamespace(
            host_role="retention", dcp_owner_mode=True, canonical_kv_page=WHOLE,
            canonical_kv_owner_rows=None, is_mla_model=False, tp_rank=0,
        )
        with self.assertRaisesRegex(NotImplementedError, "requires page_size == 1"):
            ctl.attach_storage_backend("file")

    def test_a_kv_holding_worker_takes_the_real_backend(self):
        made, null = self._attach(W2, worker=True, holds_kv=True)
        self.assertEqual(made, ["factory"])
        null.assert_not_called()

    def test_a_byteless_worker_keeps_the_null_tier(self):
        made, null = self._attach(W2, worker=True, holds_kv=False)
        self.assertEqual(made, [])
        null.assert_called_once()


if __name__ == "__main__":
    unittest.main()
