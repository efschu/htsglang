"""#1416g: the extra pools' pages of one prefetch are read in ONE batch.

NF z30e (ca2a9706ec, boot ...stvsyncbar1dauer09282117), PP0 PDFLIP-READ-STAGES:
``kv_ms`` 1-3 (the arena addresses the KV pages in place), ``extra_ms``
35-695, linear in pages -- pdflip-40-71 1275 pages 301 ms, pdflip-16-41 256 pages
55 ms (~0.22 ms/page). The QSA index sidecar went through ``_read_page`` one
page at a time. Pinned here, on the real HiCacheFile and QSAPagedHostPool of
the x56 byte test: the read goes through one ``batch_get`` (one arena call
per width, one ``pageio.read_pages`` for the rest), the bytes are those of
the per-page path, a miss stays a miss.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_qsa_index_sidecar_bytes_x56 as x56  # noqa: E402

from flliper.srt.mem_cache.hicache_storage import HiCacheFile, PoolName, PoolTransfer  # noqa: E402
from flliper.srt.mem_cache.qsa_pool_host import QSAPagedHostPool  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="stage-a-pdflip-unit")

TOKENS = torch.arange(x56.N_PAGES * x56.PAGE)


def _write_stages(root):
    for ids in x56.STAGE_IDS:
        dev = x56._device_pool(ids)
        host = x56._host_pool(dev)
        be = x56._backend(root, ids, dev, host)
        x56._mirror_device_to_host(host)
        res = be.batch_set_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=list(x56.KEYS),
                                            host_indices=TOKENS)])
        assert res[PoolName.QSA_INDEXER] == [True] * x56.N_PAGES


def _decode_backend(root, layout="layer_first"):
    dev = x56._device_pool(x56.ATTN_IDS)
    host = QSAPagedHostPool([dev], num_host_tokens=(x56.N_PAGES + 1) * x56.PAGE,
                            page_size=x56.PAGE, layout=layout, pin_memory=False)
    if layout == "layer_first":
        for b in host.kv_buffer:
            b.zero_()
    else:
        host.kv_buffer.zero_()
    return x56._backend(root, x56.ATTN_IDS, dev, host), host


def _layer_row(host, layer, row):
    if host.layout == "layer_first":
        return host.kv_buffer[layer][row]
    return host.kv_buffer[row].reshape(host.layer_num, host.item_bytes)[layer]


class ExtraPoolBatchRead(unittest.TestCase):
    def _assert_bytes(self, host, pages=range(x56.N_PAGES)):
        for l, g in enumerate(x56.ATTN_IDS):
            for p in pages:
                got = _layer_row(host, l, p)
                self.assertTrue(torch.equal(got, x56._expected_block(g, p)), f"layer {g} page {p}")

    def test_the_read_is_one_batch_not_a_get_per_page(self):
        with tempfile.TemporaryDirectory() as root:
            _write_stages(root)
            be, host = _decode_backend(root)
            real_get, real_batch = HiCacheFile.get, HiCacheFile.batch_get
            with mock.patch.object(HiCacheFile, "get", autospec=True, side_effect=real_get) as get, \
                    mock.patch.object(HiCacheFile, "batch_get", autospec=True, side_effect=real_batch) as bg:
                res = be.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=list(x56.KEYS),
                                                    host_indices=TOKENS)])
            self.assertEqual(res[PoolName.QSA_INDEXER], [True] * x56.N_PAGES)
            self.assertEqual(bg.call_count, 1)
            self.assertEqual(get.call_count, 0, "no per-page get (base: one per page)")
            self._assert_bytes(host)

    def test_page_first_layouts_read_the_same_bytes(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as root:
                _write_stages(root)
                be, host = _decode_backend(root, layout)
                res = be.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=list(x56.KEYS),
                                                    host_indices=TOKENS)])
                self.assertEqual(res[PoolName.QSA_INDEXER], [True] * x56.N_PAGES)
                self._assert_bytes(host)

    def test_a_missing_page_stays_a_miss_and_its_row_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            _write_stages(root)
            be, host = _decode_backend(root)
            keys = [x56.KEYS[0], "feedbeef", x56.KEYS[2]]
            res = be.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=keys,
                                                host_indices=TOKENS)])
            self.assertEqual(res[PoolName.QSA_INDEXER], [True, False, True])
            self._assert_bytes(host, pages=(0, 2))
            for l in range(len(x56.ATTN_IDS)):
                self.assertEqual(int(_layer_row(host, l, 1).abs().sum()), 0)

    def test_arena_pages_are_read_in_one_arena_call(self):
        from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as adir, \
                mock.patch.dict(os.environ, {"FLLIPER_HICACHE_ARENA_DIR": adir,
                                             "FLLIPER_HICACHE_ARENA_GIB": "0.0625"}):
            _write_stages(root)
            be, host = _decode_backend(root)
            real_read = ShmArena.read
            with mock.patch.object(ShmArena, "read", autospec=True, side_effect=real_read) as rd:
                res = be.batch_get_v2([PoolTransfer(name=PoolName.QSA_INDEXER, keys=list(x56.KEYS),
                                                    host_indices=TOKENS)])
            self.assertEqual(res[PoolName.QSA_INDEXER], [True] * x56.N_PAGES)
            self.assertEqual(rd.call_count, 1, [len(c.args[1]) for c in rd.call_args_list])
            self.assertEqual(len(rd.call_args_list[0].args[1]), x56.N_PAGES)
            self._assert_bytes(host)


class BatchSetterMatchesPerPage(unittest.TestCase):
    def test_vectorised_setter_equals_the_per_page_setter(self):
        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                dev = x56._device_pool(x56.ATTN_IDS)
                mk = lambda: QSAPagedHostPool([dev], num_host_tokens=(x56.N_PAGES + 1) * x56.PAGE,
                                              page_size=x56.PAGE, layout=layout, pin_memory=False)
                a, b = mk(), mk()
                numel = int(a.get_dummy_flat_data_page().numel())
                pages = torch.randint(0, 255, (3, numel), dtype=torch.uint8).view(a.dtype)
                idx = [2 * x56.PAGE, 0, 3 * x56.PAGE]
                for i, p in zip(idx, pages):
                    a.set_from_flat_data_page(i, p)
                b.set_from_flat_data_pages(idx, pages)
                for i in idx:
                    self.assertTrue(torch.equal(a.get_data_page(i), b.get_data_page(i)))


if __name__ == "__main__":
    unittest.main()
