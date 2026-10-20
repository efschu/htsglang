"""NF-PLE-SMALL (20.10.): an extend PLE gather below the pread range is served
from a host slot through the staged kernel, never through cold HMM faults.

Metal (rc28 D ``boot_..._297198745d_1020_174205.D.log``, Form A: TP0 is the only
PLE rank): every extend whose gather had < 2048 rows (T < 128) paid 0.11-1.42 s
outside the expert span -- T=95 (pdflip-2-50's 95-token tail after a 2812-token
chunk): gpu-ms 1875.7, em_ms 443.2, TP0 compute 1712.0, TP1/TP2 waiting
1472.0/1515.9 ms in tp.all_reduce -- while every gather of >= 2048 rows (pread)
paid 25-145 ms. The small gather went to the HMM kernel, which faults the 95.4
GiB checkpoint mmap on ZFS one cold page at a time.

Desk (CPU, real pread worker processes, real files; the staged kernel replaced
by a CPU emulation of its documented row rule): the small tail is served from
the prefetch the previous chunk queued (any size now), a fresh small extend is
predicted from its own published tokens, bytes equal the serial gather, a wrong
prediction costs reads and never bytes, the device ids are never read back, a
decode gather (no new extend hint) and a graph capture keep the old path.
"""

import ctypes
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging
import re
import tempfile
import types
import unittest
from array import array
from collections import namedtuple
from unittest import mock

import torch

from flliper.srt.models import qwen4_exp_ple_prefetch as pf
from flliper.srt.models import qwen4_exp_ple_table as pt
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

Range = namedtuple("Range", "start end")
ROWS_PER_SHARD = 64
SHARDS = 4
DIM = 160
RB = DIM * 2
HEADER = 100
MIN_ROWS = 16
LOGGER = "flliper.srt.models.qwen4_exp_ple_prefetch"


def _table(tmpdir):
    files, shard_files, shard_offsets = [], [], []
    for f in range(2):
        path = os.path.join(tmpdir, f"ple-{f}.safetensors")
        with open(path, "wb") as fh:
            fh.write(os.urandom(HEADER + 2 * ROWS_PER_SHARD * RB))
        files.append(path)
        for s in range(2):
            shard_files.append(path)
            shard_offsets.append(HEADER + s * ROWS_PER_SHARD * RB)
    return pt.CheckpointMappedPleTable(
        [0] * SHARDS, ROWS_PER_SHARD, ROWS_PER_SHARD * SHARDS, torch.bfloat16, DIM,
        keepalive=[], files=files, shard_files=shard_files, shard_offsets=shard_offsets,
    )


def _identity_hasher(tokens, lead):
    return tokens[lead:].clone()


def _serial(table, ids, vocab=(0, None)):
    base = pt.PleCheckpointPreadGather(table, min_rows=1, workers=4)
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
    base.gather_into(ids, out, vocab_start=vocab[0], vocab_end=vocab[1])
    base.close()
    return out


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


class _NoReadback(torch.Tensor):
    """The device's lookup ids: reading them back on the host is the sync the
    small path must not take."""

    @classmethod
    def __torch_function__(cls, func, types_, args=(), kwargs=None):
        if func in (torch.Tensor.cpu, torch.Tensor.tolist, torch.Tensor.numpy, torch.Tensor.item):
            raise AssertionError(f"device ids read back on the host ({func.__name__})")
        return super().__torch_function__(func, types_, args, kwargs or {})


def _plain(t):
    return t.as_subclass(torch.Tensor)


class _StagedEmu:
    """CPU emulation of ``_gather_ple_embedding_staged_kernel``: row r from the
    stage when ``stage_ids[r] == ids[r]`` (and in range), else from the table."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, flat_ids, out, addrs, counters):
        ids = _plain(flat_ids).clone()
        n = int(ids.numel())
        ids_addr, rows_addr = (int(x) for x in _plain(addrs))
        stage_ids = torch.frombuffer((ctypes.c_int64 * n).from_address(ids_addr), dtype=torch.int64).clone()
        stage_rows = torch.frombuffer(
            (ctypes.c_uint8 * (n * RB)).from_address(rows_addr), dtype=torch.uint8
        ).clone().view(torch.bfloat16).view(n, DIM)
        from_table = _serial(self.table, ids)
        hit = stage_ids == ids
        res = torch.where(hit.unsqueeze(1), stage_rows, from_table)
        out.view(n, DIM).copy_(res)
        counters[0] += n
        counters[1] += int(hit.sum())
        self.calls.append((n, int(hit.sum())))


class _Base(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.table = _table(self._tmp.name)
        pf._HINT = None
        # CPU desk: the small path's device predicates are the CUDA ones there
        self._p = [mock.patch.object(pf, "_small_device", lambda t: True),
                   mock.patch.object(pf, "_small_needs_pin", lambda t: False),
                   mock.patch.object(pf, "_capturing", lambda t: False)]
        for p in self._p:
            p.start()
        self.g = pf.PlePrefetchGather(
            pt.PleCheckpointPreadGather(self.table, min_rows=MIN_ROWS, workers=4), self.table,
            _identity_hasher, procs=2, threads=4,
        )
        self.emu = _StagedEmu(self.table)
        g = torch.Generator().manual_seed(5)
        self.ids = torch.randint(0, ROWS_PER_SHARD * SHARDS, (100,), generator=g)
        self.req = types.SimpleNamespace(extend_range=None, full_untruncated_fill_ids=array("q", self.ids.tolist()))

    def tearDown(self):
        self.g.close()
        for p in self._p:
            p.stop()
        pf._HINT = None
        self._tmp.cleanup()

    def publish(self, start, end, chunk):
        self.req.extend_range = Range(start, end)
        pf.publish_ple_next_chunk([self.req], chunk)

    def gather(self, ids):
        """What Qwen4ExpPinnedHostEmbedding.gather does for one gather."""
        out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
        dev = ids.as_subclass(_NoReadback)
        if self.g.wants(ids):
            self.g.gather_into(ids, out)
            return out, "pread"
        if self.g.wants_small(dev) and self.g.gather_small(dev, out, launch=self.emu):
            return out, "small"
        return out, "old"


class TestSmallExtendGather(_Base):
    def test_small_tail_is_served_from_the_previous_chunks_prefetch(self):
        # chunk 0: 90 rows (>= MIN_ROWS, pread), chunk 1: the 10-row tail (< MIN_ROWS)
        with self.assertLogs(LOGGER, level=logging.INFO) as cm:
            self.publish(0, 90, 90)
            out0, how0 = self.gather(self.ids[0:90])
            self.publish(90, 100, 90)
            out1, how1 = self.gather(self.ids[90:100])
        self.assertEqual((how0, how1), ("pread", "small"))
        text = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("next=queued:10", text)  # the tail is queued although below min_rows
        m = re.search(r"PLE-PREFETCH-SMALL rows=(\d+) staged=(\d+) src=(\w+) ready=(\w+) served=(\w+)", text)
        self.assertIsNotNone(m)
        self.assertEqual((m[1], m[2], m[3], m[5]), ("10", "10", "next", "yes"))
        self.assertTrue(_same(out1, _serial(self.table, self.ids[90:100])))
        self.assertEqual(self.emu.calls, [(10, 10)])  # every row from the stage

    def test_fresh_small_extend_is_predicted_from_its_own_tokens(self):
        self.publish(0, 12, 4096)
        with self.assertLogs(LOGGER, level=logging.INFO) as cm:
            out, how = self.gather(self.ids[0:12])
        self.assertEqual(how, "small")
        self.assertRegex("\n".join(r.getMessage() for r in cm.records), r"src=cur .*served=yes")
        self.assertTrue(_same(out, _serial(self.table, self.ids[0:12])))
        self.assertEqual(self.emu.calls, [(12, 12)])

    def test_a_wrong_prediction_costs_reads_never_bytes(self):
        self.publish(0, 12, 4096)
        actual = self.ids[0:12].clone()
        actual[3] = (actual[3] + 1) % (ROWS_PER_SHARD * SHARDS)
        actual[7] = (actual[7] + 5) % (ROWS_PER_SHARD * SHARDS)
        out, how = self.gather(actual)
        self.assertEqual(how, "small")
        self.assertTrue(_same(out, _serial(self.table, actual)))
        self.assertEqual(self.emu.calls, [(12, 10)])

    def test_prediction_shorter_than_the_gather_reads_the_rest_from_the_table(self):
        # a mixed batch: the extend's rows first, then rows nobody predicted
        self.publish(0, 8, 4096)
        actual = torch.cat([self.ids[0:8], self.ids[50:54]])
        # a stale stage from an earlier use of the slot must not serve rows 8..11
        self.g._slots[0].stage_ids_for(12)[:12].copy_(actual)
        out, how = self.gather(actual)
        self.assertEqual(how, "small")
        self.assertTrue(_same(out, _serial(self.table, actual)))
        self.assertEqual(self.emu.calls, [(12, 8)])

    def test_device_ids_are_never_read_back(self):
        # _NoReadback raises on .cpu()/.tolist()/.item(); the served path completes
        self.publish(0, 12, 4096)
        out, how = self.gather(self.ids[0:12])
        self.assertEqual(how, "small")

    def test_decode_gather_without_a_new_extend_hint_keeps_the_old_path(self):
        self.publish(0, 12, 4096)
        self.gather(self.ids[0:12])  # the extend consumes its hint
        out, how = self.gather(self.ids[20:24])  # a decode gather: no new hint published
        self.assertEqual(how, "old")
        self.assertEqual(len(self.emu.calls), 1)

    def test_graph_capture_keeps_the_old_path(self):
        self.publish(0, 12, 4096)
        with mock.patch.object(pf, "_capturing", lambda t: True):
            self.assertFalse(self.g.wants_small(self.ids[0:12]))

    def test_pread_range_is_unchanged(self):
        self.publish(0, 90, 4096)
        self.assertFalse(self.g.wants_small(self.ids[0:MIN_ROWS]))
        self.assertTrue(self.g.wants(self.ids[0:MIN_ROWS]))

    def test_nothing_predicted_falls_back_and_still_queues_the_next_chunk(self):
        # a chunk too long for the published current tokens and no prefetch:
        # not served, the caller gathers as before; the next chunk is queued
        with mock.patch.object(pf, "PLE_SMALL_CUR_TOKENS", 4):
            self.publish(0, 10, 10)
            with self.assertLogs(LOGGER, level=logging.INFO) as cm:
                out, how = self.gather(self.ids[0:10])
        self.assertEqual(how, "old")
        self.assertRegex("\n".join(r.getMessage() for r in cm.records), r"served=no .*next=queued:10")
        self.assertEqual(self.emu.calls, [])

    def test_unpinned_slot_on_a_device_keeps_the_old_path(self):
        with mock.patch.object(pf, "_small_needs_pin", lambda t: True):
            self.publish(0, 12, 4096)
            out, how = self.gather(self.ids[0:12])
        self.assertEqual(how, "old")


class TestSlotIsMappedForTheDevice(_Base):
    def test_slot_registration_maps_it(self):
        reg = mock.MagicMock(return_value=0)
        rt = types.SimpleNamespace(cudaHostRegister=reg, cudaHostUnregister=mock.MagicMock(return_value=0))
        with mock.patch.object(torch.cuda, "is_available", lambda: True), \
                mock.patch.object(torch.cuda, "cudart", lambda: rt):
            self.g._ensure_rows(0, 10)
        self.assertEqual(reg.call_args[0][2], pf._CUDA_HOST_REGISTER_PORTABLE_MAPPED)
        self.assertEqual(pf._CUDA_HOST_REGISTER_PORTABLE_MAPPED, 3)
        self.g._slots[0].pinned = False  # nothing was really registered


class TestPublishCurrentChunk(_Base):
    def test_small_chunk_carries_its_tokens_with_lead(self):
        self.publish(40, 52, 4096)
        (toks, lead), = pf._HINT.cur_parts
        self.assertEqual(lead, pf.PLE_PREFETCH_LEAD_TOKENS)
        self.assertTrue(torch.equal(toks, self.ids[40 - lead : 52]))

    def test_first_chunk_has_no_lead(self):
        self.publish(0, 12, 4096)
        (toks, lead), = pf._HINT.cur_parts
        self.assertEqual(lead, 0)
        self.assertTrue(torch.equal(toks, self.ids[0:12]))

    def test_large_chunk_carries_none(self):
        with mock.patch.object(pf, "PLE_SMALL_CUR_TOKENS", 10):
            self.publish(0, 90, 90)
        self.assertIsNone(pf._HINT.cur_parts)

    def test_mixed_batch_is_laid_out_in_request_order(self):
        # an extend request that completes in this chunk, then two running
        # decode requests with one row each (mix_with_running's layout)
        dec1 = types.SimpleNamespace(extend_range=Range(59, 60), full_untruncated_fill_ids=array("q", self.ids[:60].tolist()))
        dec2 = types.SimpleNamespace(extend_range=Range(79, 80), full_untruncated_fill_ids=array("q", self.ids[:80].tolist()))
        ext = types.SimpleNamespace(extend_range=Range(0, 12), full_untruncated_fill_ids=array("q", self.ids[:12].tolist()))
        pf.publish_ple_next_chunk([ext, dec1, dec2], 4096)
        parts = pf._HINT.cur_parts
        self.assertEqual([lead for _t, lead in parts], [0, 8, 8])
        self.assertTrue(torch.equal(parts[1][0], self.ids[51:60]))
        self.assertTrue(torch.equal(parts[2][0], self.ids[71:80]))

    def test_a_request_without_a_range_publishes_no_layout(self):
        odd = types.SimpleNamespace(extend_range=None, full_untruncated_fill_ids=array("q", [1, 2]))
        self.req.extend_range = Range(0, 12)
        pf.publish_ple_next_chunk([self.req, odd], 4096)
        self.assertIsNone(pf._HINT.cur_parts)


class TestMixedBatchServed(_Base):
    def test_fresh_extend_plus_decode_rows_all_from_the_slot(self):
        dec = types.SimpleNamespace(extend_range=Range(79, 80), full_untruncated_fill_ids=array("q", self.ids[:80].tolist()))
        ext = types.SimpleNamespace(extend_range=Range(0, 6), full_untruncated_fill_ids=array("q", self.ids[:6].tolist()))
        pf.publish_ple_next_chunk([ext, dec], 4096)
        actual = torch.cat([self.ids[0:6], self.ids[79:80]])  # identity hasher: one row per token
        out, how = self.gather(actual)
        self.assertEqual(how, "small")
        self.assertTrue(_same(out, _serial(self.table, actual)))
        self.assertEqual(self.emu.calls, [(7, 7)])


class TestAdmissionSubclassPassesTheSmallPath(CustomTestCase):
    def test_admit_gather_serves_small(self):
        from flliper.srt.models import qwen4_exp_ple_admit as ad

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(pf, "_small_device", lambda t: True), \
                mock.patch.object(pf, "_small_needs_pin", lambda t: False), \
                mock.patch.object(pf, "_capturing", lambda t: False):
            table = _table(tmp)
            pf._HINT = None
            g = ad.PleAdmitPrefetchGather(
                pt.PleCheckpointPreadGather(table, min_rows=MIN_ROWS, workers=4), table,
                _identity_hasher, procs=2, threads=4,
            )
            try:
                ids = torch.arange(0, 12)
                req = types.SimpleNamespace(extend_range=Range(0, 12), full_untruncated_fill_ids=array("q", ids.tolist()))
                pf.publish_ple_next_chunk([req], 4096)
                out = torch.empty((12, DIM), dtype=torch.bfloat16)
                emu = _StagedEmu(table)
                self.assertTrue(g.wants_small(ids))
                self.assertTrue(g.gather_small(ids.as_subclass(_NoReadback), out, launch=emu))
                self.assertTrue(_same(out, _serial(table, ids)))
            finally:
                g.close()
                pf._HINT = None


class TestEmbeddingDispatch(CustomTestCase):
    """Qwen4ExpPinnedHostEmbedding.gather: the small path sits between the
    pread range and the decode stage / HMM kernel."""

    def _self(self, *, small_served):
        from flliper.srt.models import qwen4_exp as q

        pread = types.SimpleNamespace(
            wants=lambda ids: False,
            wants_small=lambda ids: True,
            gather_small=mock.MagicMock(return_value=small_served),
        )
        stager = types.SimpleNamespace(launch=mock.MagicMock(return_value=True))
        me = types.SimpleNamespace(
            embedding_dim=DIM, _ckpt_backend=True, _ckpt_table=object(), _ckpt_pread=pread,
            _ckpt_prefetcher=None, _decode_stager=stager,
            shard_indices=types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=256),
            _launch_staged_small=lambda *a, **k: None, _block_d=256,
        )
        return q, me, pread, stager

    def test_served_small_returns_before_the_stage_and_kernel(self):
        q, me, pread, stager = self._self(small_served=True)
        out = torch.empty((3, 4, DIM), dtype=torch.bfloat16)
        got = q.Qwen4ExpPinnedHostEmbedding.gather(me, torch.zeros((3, 4), dtype=torch.long), out=out)
        self.assertIs(got, out)
        pread.gather_small.assert_called_once()
        stager.launch.assert_not_called()

    def test_unserved_small_falls_through_to_the_stage(self):
        q, me, pread, stager = self._self(small_served=False)
        out = torch.empty((3, 4, DIM), dtype=torch.bfloat16)
        q.Qwen4ExpPinnedHostEmbedding.gather(me, torch.zeros((3, 4), dtype=torch.long), out=out)
        pread.gather_small.assert_called_once()
        stager.launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
