"""23.09. (fnFL2x56, Task #106): the compressed QSA index page must survive
the carrier BYTE-IDENTICALLY, through the real chain -- three PP stage host
pools (7/3/2 of the 12 full-attention layers) write their canonical extents
with ``batch_set_v2``, the decode group's pool reads the whole page with
``batch_get_v2`` -- and the pool's own device<->host transport (``direct``
backend) must place every layer's block on the row of ITS KV page.

x56 on the metal: every attach line was right (windows, transfer domain,
arena pages with all three extents), the needle at 50 % still came back as
'4711' instead of '4711-QX'. This test pins the desk-checkable half of that
chain to the byte, so a wrong offset, a permuted layer or a row/slot mix-up
is named here and not by a needle.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.qsa_pool_host import QSAPagedHostPool, build_qsa_index_window
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-weg2-unit")

# the deployment's shape: 12 full-attention layers, PP cut 7/3/2, page 64,
# compress ratio 4, one index head of 128 bf16 -> 256 B per group, 4096 B per
# layer block, 49152 B per page.
ATTN_IDS = list(range(3, 48, 4))
STAGE_IDS = [ATTN_IDS[:7], ATTN_IDS[7:10], ATTN_IDS[10:]]
PAGE, RATIO, HEADS, DIM = 64, 4, 1, 128
GROUPS = PAGE // RATIO
SLOT_BYTES = HEADS * DIM * 2
BLOCK = GROUPS * SLOT_BYTES
N_PAGES = 3
CAP = (N_PAGES + 1) * GROUPS  # compressed slots (state_size = slots + page)
KEYS = ["cafe01", "cafe02", "cafe03"]
KV_SPEC = CanonicalPageSpec(num_attn_layers=len(ATTN_IDS), kv_bytes_per_token_per_attn_layer=64)
IDENTITY = "0123456789abcdef"


def _pattern_bytes(global_layer: int) -> torch.Tensor:
    """The compressed buffer of ONE global layer as bytes: byte(c, b) = f(layer, slot, byte)."""
    c = torch.arange(CAP).view(-1, 1)
    b = torch.arange(SLOT_BYTES).view(1, -1)
    return ((global_layer * 37 + c * 5 + b) % 251).to(torch.uint8).reshape(-1)


def _device_pool(layer_ids, device="cpu"):
    buffers = [
        _pattern_bytes(g).to(device).view(torch.bfloat16).view(CAP, HEADS, DIM)
        for g in layer_ids
    ]
    return SimpleNamespace(
        qsa_compress_ratio=RATIO,
        qsa_compressed_k_buffer_pool=buffers,
        full_attention_layer_id_mapping={g: i for i, g in enumerate(layer_ids)},
    )


def _host_pool(dev, device="cpu"):
    return QSAPagedHostPool([dev], num_host_tokens=(N_PAGES + 1) * PAGE, page_size=PAGE,
                            layout="layer_first", pin_memory=False)


def _config(kv_window, qsa_window):
    return HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=False,
        model_name="Qwen-qsa-roundtrip", model_identity_hash=IDENTITY,
        canonical_kv_page=kv_window, canonical_qsa_page=qsa_window,
    )


def _backend(root, layer_ids, dev, host):
    kv_w = window_for_layers(KV_SPEC, ATTN_IDS, layer_ids)
    qsa_w = build_qsa_index_window(ATTN_IDS, dev, host)
    be = HiCacheFile(_config(kv_w, qsa_w), file_path=root)
    be.file_path = root
    be.register_mem_host_pool_v2(host, PoolName.QSA_INDEXER)
    return be


def _expected_block(global_layer: int, page: int) -> torch.Tensor:
    return _pattern_bytes(global_layer).view(-1, BLOCK)[page]


def _mirror_device_to_host(host):
    """The D2H copy on the desk (the CUDA kernel is exercised below): host row
    p of layer l is device row p of layer l -- what ``backup_from_device_all_layer``
    does for page-aligned token indices."""
    for l, buf in enumerate(host.device_buffers):
        for p in range(N_PAGES):
            host.kv_buffer[l][p].copy_(buf[p])


class QsaIndexSidecarBytes(unittest.TestCase):
    def test_three_pp_stages_write_one_decode_pool_reads_byte_identically(self):
        with tempfile.TemporaryDirectory() as root:
            tokens = torch.arange(N_PAGES * PAGE)
            for ids in STAGE_IDS:
                dev = _device_pool(ids)
                host = _host_pool(dev)
                be = _backend(root, ids, dev, host)
                _mirror_device_to_host(host)
                res = be.batch_set_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=list(KEYS),
                                                    host_indices=tokens)])
                self.assertEqual(res[PoolName.QSA_INDEXER], [True] * N_PAGES, f"stage {ids}")

            dev_d = _device_pool(ATTN_IDS)
            host_d = _host_pool(dev_d)
            host_d.kv_buffer = [b.zero_() for b in host_d.kv_buffer]
            be_d = _backend(root, ATTN_IDS, dev_d, host_d)
            self.assertEqual(tuple(be_d.canonical_qsa_page.extents), ((0, len(ATTN_IDS) * BLOCK),))
            res = be_d.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=list(KEYS),
                                                  host_indices=tokens)])
            self.assertEqual(res[PoolName.QSA_INDEXER], [True] * N_PAGES)
            for l, g in enumerate(ATTN_IDS):
                for p in range(N_PAGES):
                    got = host_d.kv_buffer[l][p]
                    want = _expected_block(g, p)
                    self.assertTrue(torch.equal(got, want),
                                    f"layer {g} (ordinal {l}) page {p}: first bytes {got[:8].tolist()} vs {want[:8].tolist()}")

    def test_extents_of_the_stages_tile_the_page(self):
        offs = []
        for ids in STAGE_IDS:
            dev = _device_pool(ids)
            w = build_qsa_index_window(ATTN_IDS, dev, _host_pool(dev))
            offs.append(tuple(w.extents)[0])
        self.assertEqual(offs, [(0, 7 * BLOCK), (7 * BLOCK, 3 * BLOCK), (10 * BLOCK, 2 * BLOCK)])

    @unittest.skipUnless(torch.cuda.is_available(), "the transport kernels are CUDA-only")
    def test_pool_transport_direct_round_trip_on_cuda(self):
        """D2H then H2D with the pool's own ``direct`` transport: the bytes of
        every layer land on the row of their KV page, and back on the device
        rows they came from -- including rows loaded out of order."""
        from sglang.srt.mem_cache.qsa_pool_host import QSAPagedHostPool as _P

        dev = _device_pool(ATTN_IDS, device="cuda")
        host = _P([dev], num_host_tokens=(N_PAGES + 1) * PAGE, page_size=PAGE,
                  layout="layer_first", pin_memory=True)
        src_pages = [2, 0, 1]
        dev_idx = torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in src_pages]).cuda()
        host_idx = torch.arange(N_PAGES * PAGE)
        host.backup_from_device_all_layer(dev, host_idx, dev_idx, "direct")
        torch.cuda.synchronize()
        for l, g in enumerate(ATTN_IDS):
            for i, p in enumerate(src_pages):
                self.assertTrue(torch.equal(host.kv_buffer[l][i].cpu(), _expected_block(g, p)),
                                f"D2H layer {g} host row {i} <- device page {p}")
        for buf in dev.qsa_compressed_k_buffer_pool:
            buf.zero_()
        dst_pages = [1, 2, 0]
        dev_idx2 = torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in dst_pages]).cuda()
        for l in range(len(ATTN_IDS)):
            host.load_to_device_per_layer(dev, host_idx, dev_idx2, l, "direct")
        torch.cuda.synchronize()
        for l, g in enumerate(ATTN_IDS):
            rows = host.device_buffers[l]
            for i, p in enumerate(dst_pages):
                self.assertTrue(torch.equal(rows[p].cpu(), _expected_block(g, src_pages[i])),
                                f"H2D layer {g} device page {p} <- host row {i}")


if __name__ == "__main__":
    unittest.main()
