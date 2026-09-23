"""23.09. (fnFL2x54, Task #106): the compressed QSA index keys travel with the
KV page as a sidecar. What must hold on the desk: the host mirror is one byte
row per KV page and layer (page_size // ratio slots), partial pages are
refused, the canonical window of a PP stage is ONE extent at its layers'
ordinal offset, a non-contiguous or foreign layer set is refused, and the
storage backend routes a '{hash}.qsa_indexer' key to that window with the
KV key's suffix rule.
"""
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.canonical_page_store import CanonicalPageError
from sglang.srt.mem_cache.qsa_pool_host import (
    QSAPagedHostPool,
    build_qsa_index_window,
    qsa_index_bytes_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")

RATIO, PAGE, HEADS, DIM = 4, 64, 1, 128


def _pool(layer_ids, slots=256):
    """A device-side stand-in: compressed buffers [slot, head, dim] bf16 per
    full-attention layer of THIS rank, and the global->local layer map."""
    return SimpleNamespace(
        qsa_compress_ratio=RATIO,
        qsa_compressed_k_buffer_pool=[
            torch.zeros(slots, HEADS, DIM, dtype=torch.bfloat16) for _ in layer_ids
        ],
        full_attention_layer_id_mapping={g: i for i, g in enumerate(layer_ids)},
    )


class QsaIndexSidecar(unittest.TestCase):
    def test_bytes_per_token_and_page_block(self):
        pool = _pool([3, 7, 11])
        # 1 head x 128 x 2 B = 256 B per group of 4 tokens = 64 B/token/layer
        self.assertEqual(qsa_index_bytes_per_token([pool], PAGE), 3 * 64)
        host = QSAPagedHostPool([pool], num_host_tokens=4 * PAGE, page_size=PAGE, layout="layer_first",
                                pin_memory=False)
        self.assertEqual(host.item_bytes, PAGE // RATIO * 256)  # 4096 B per layer per page
        self.assertEqual(host.layer_num, 3)
        self.assertEqual(host.get_size_per_token(), 3 * 64)

    def test_partial_pages_are_refused(self):
        host = QSAPagedHostPool([_pool([3])], num_host_tokens=4 * PAGE, page_size=PAGE,
                                layout="layer_first", pin_memory=False)
        with self.assertRaises(ValueError):
            host._has_transfer_indices(torch.arange(PAGE + 3), torch.arange(PAGE + 3))
        self.assertTrue(host._has_transfer_indices(torch.arange(2 * PAGE), torch.arange(2 * PAGE)))

    def test_pp_stage_window_is_one_extent_at_its_ordinal(self):
        # the model's 12 full layers 3,7,...,47; PP1 holds 31,35,39 = ordinals 7..9
        ids = list(range(3, 48, 4))
        pool = _pool([31, 35, 39])
        host = QSAPagedHostPool([pool], num_host_tokens=2 * PAGE, page_size=PAGE,
                                layout="layer_first", pin_memory=False)
        w = build_qsa_index_window(ids, pool, host)
        self.assertEqual(w.total_bytes, 12 * 4096)
        self.assertEqual(tuple(w.extents), ((7 * 4096, 3 * 4096),))

    def test_foreign_or_gapped_layers_are_refused(self):
        ids = list(range(3, 48, 4))
        host = QSAPagedHostPool([_pool([3, 11])], num_host_tokens=PAGE, page_size=PAGE,
                                layout="layer_first", pin_memory=False)
        with self.assertRaises(CanonicalPageError):
            build_qsa_index_window(ids, _pool([3, 11]), host)  # gap at 7
        with self.assertRaises(CanonicalPageError):
            build_qsa_index_window(ids, _pool([4, 8]), host)  # not full-attention ids

    def test_storage_routes_the_qsa_key_to_its_window(self):
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile, PoolName

        # the key rules live on the file backend; no __init__ (no store dir)
        backend = HiCacheFile.__new__(HiCacheFile)
        window = SimpleNamespace(total_bytes=12 * 4096, extents=((0, 4096),))
        backend.canonical_qsa_page = window
        backend.canonical_draft_page = None
        backend.canonical_mamba_blob = None
        backend._canonical_kv_extents = None
        self.assertTrue(backend._is_qsa_key("abc." + PoolName.QSA_INDEXER))
        self.assertFalse(backend._is_qsa_key("abc." + PoolName.MAMBA))
        self.assertIs(backend._canonical_window("abc.qsa_indexer"), window)
        backend.canonical_qsa_page = None
        self.assertFalse(backend._is_qsa_key("abc.qsa_indexer"))
        self.assertIsNone(backend._canonical_window("abc.qsa_indexer"))

    def test_draft_install_without_qsa_page_keeps_the_window(self):
        # fnFL2x55: every P stage died at the draft window install, which
        # passes only its own slot -- qsa_page=None must mean "unchanged".
        from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageError
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        backend = HiCacheFile.__new__(HiCacheFile)
        kv = SimpleNamespace(spec=SimpleNamespace(page_bytes=786432),
                             as_extents=lambda: "kv-extents")
        qsa = SimpleNamespace(total_bytes=12 * 4096, extents=((0, 4096),))
        draft = SimpleNamespace(total_bytes=2048, extents=((0, 2048),))
        backend.canonical_kv_page = kv
        backend._canonical_kv_extents = "kv-extents"
        backend.canonical_mamba_blob = None
        backend.canonical_draft_page = None
        backend.canonical_qsa_page = qsa
        backend._rederive_suffixes = lambda: None
        try:
            backend.install_canonical_windows(kv, None, draft_page=draft)
        except AttributeError:
            pass  # the suffix re-derivation needs a real backend; the slot check ran
        self.assertIs(backend.canonical_qsa_page, qsa)
        self.assertIs(backend.canonical_draft_page, draft)
        other = SimpleNamespace(total_bytes=13 * 4096, extents=((0, 4096),))
        with self.assertRaises(CanonicalPageError):
            backend.install_canonical_windows(kv, None, draft_page=draft, qsa_page=other)


if __name__ == "__main__":
    unittest.main()
