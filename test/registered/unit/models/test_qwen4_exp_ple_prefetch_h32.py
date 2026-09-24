"""fnFL2 H32 (SGLANG_QWEN4_PLE_PREFETCH): the PLE pread gather one chunk ahead.

The pread gather of a prefill chunk runs in worker processes; the scheduler
publishes the chunked request's next chunk before each extend forward, and the
gather starts reading that chunk's rows (hashed on the host with the layer's own
n-gram hash) into the other slot of a two-slot ring right after serving its own
chunk. Desk only (CPU, real worker processes, real files): bytes equal to the
serial gather, the prefetch starts only after the current chunk is served, a
join waits only for the rest, a wrong prediction costs reads and never bytes,
the switch off keeps the serial gather object, the ring stays two deep, a lost
worker falls back to the serial gather.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import functools
import logging
import re
import tempfile
import time
import types
import unittest
from array import array
from collections import namedtuple
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.models import qwen4_exp_ple_prefetch as pf
from sglang.srt.models import qwen4_exp_ple_table as pt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

Range = namedtuple("Range", "start end")
ROWS_PER_SHARD = 64
SHARDS = 4
DIM = 160
RB = DIM * 2
HEADER = 100
LOGGER = "sglang.srt.models.qwen4_exp_ple_prefetch"


def _table(tmpdir):
    """Two files laid out like checkpoint shards (a header, then 2 shards each)."""
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
    """One row per token, row id = token (the model's hasher is tested apart)."""
    return tokens[lead:].clone()


def _serial(table, ids, vocab=(0, None)):
    base = pt.PleCheckpointPreadGather(table, min_rows=16, workers=4)
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
    base.gather_into(ids, out, vocab_start=vocab[0], vocab_end=vocab[1])
    base.close()
    return out


def _same(a, b):
    """Bitwise (random bytes as bf16 carry NaNs, which torch.equal calls unequal)."""
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _parse(records):
    out = []
    for r in records:
        m = re.search(r"PLE-PREFETCH chunk=(\d+) rows=(\d+) ready=(\w+) wait_ms=([\d.]+) "
                      r"gather_ms=([\d.]+) hit_rows=(\d+) read_rows=(\d+)", r.getMessage())
        if m:
            out.append(dict(chunk=int(m[1]), rows=int(m[2]), ready=m[3], wait_ms=float(m[4]),
                            gather_ms=float(m[5]), hit=int(m[6]), read=int(m[7])))
    return out


class _Run:
    """A chunked request driven like the scheduler + model do: publish, then gather."""

    def __init__(self, tc, table, ids, chunk, *, delay_s=0.0, procs=2):
        self.tc, self.table, self.chunk = tc, table, chunk
        self.req = types.SimpleNamespace(extend_range=None, full_untruncated_fill_ids=array("q", ids.tolist()))
        self.ids = ids
        self.g = pf.PlePrefetchGather(
            pt.PleCheckpointPreadGather(table, min_rows=16, workers=4), table, _identity_hasher,
            procs=procs, threads=4, delay_s=delay_s,
        )

    def step(self, c, actual=None):
        n = len(self.req.full_untruncated_fill_ids)
        start, end = c * self.chunk, min((c + 1) * self.chunk, n)
        self.req.extend_range = Range(start, end)
        pf.publish_ple_next_chunk([self.req], self.chunk)
        ids = self.ids[start:end] if actual is None else actual
        out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
        self.g.gather_into(ids, out)
        return ids, out

    def close(self):
        self.g.close()


class TestHostHashMirror(CustomTestCase):
    def test_host_hash_is_the_models_hash(self):
        from sglang.srt.models import qwen4_exp as q

        g = torch.Generator().manual_seed(7)
        eos = 248044
        sizes = torch.tensor([20000003 + 2 * i for i in range(16)], dtype=torch.long)
        offs = torch.cat([sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]])
        mult = torch.randint(1 << 40, 1 << 62, (3,), generator=g) | 1
        emb = types.SimpleNamespace(
            enable_ple_fusion=False, ngram_size=3, heads_per_ngram=8, layer_multipliers=mult,
            ngram_heads_vocab_sizes=sizes, ngram_heads_offsets=offs, eos_token_id=eos,
        )
        emb._shift_right_ignore_eos = functools.partial(q.Qwen4ExpNGramEmbedding._shift_right_ignore_eos, emb)
        toks = torch.randint(0, 248320, (4099,), generator=g)
        toks[torch.randint(0, 4099, (40,), generator=g)] = eos  # EOS inside windows
        pool = types.SimpleNamespace(ple_window_cache=None)
        with mock.patch.object(q, "get_req_to_token_pool", lambda: pool):
            # the model's windows: [history (last 2 tokens before the chunk) | chunk].unfold
            hist, chunk = toks[1000:1002], toks[1002:4099]
            ctx = torch.cat([hist, chunk]).unsqueeze(0).unfold(1, 3, 1)[0]
            want = q.Qwen4ExpNGramEmbedding._hash_contexts(emb, ctx).reshape(-1)
        p = pf.PleHashParams.of(emb)
        got = pf.ple_ngram_lookup_ids(pf.ple_chunk_windows(toks[994:4099], 8, 3, eos), p).reshape(-1)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(got.numel(), (4099 - 1002) * 16)
        hasher = pf.ple_next_chunk_hasher(emb)
        self.assertTrue(torch.equal(hasher(toks[1000:4099], 2), want))
        # request start: EOS stands in for the history
        w0 = pf.ple_chunk_windows(toks[:5], 0, 3, eos)
        self.assertEqual(w0[0].tolist(), [eos, eos, int(toks[0])])


class TestPublish(CustomTestCase):
    def setUp(self):
        pf.register_ple_prefetch_consumer()

    def tearDown(self):
        pf.unregister_ple_prefetch_consumer()

    def test_no_consumer_publishes_nothing(self):
        pf.unregister_ple_prefetch_consumer()
        try:
            req = types.SimpleNamespace(extend_range=Range(0, 4), full_untruncated_fill_ids=array("q", range(10)))
            self.assertIsNone(pf.publish_ple_next_chunk([req], 4))
        finally:
            pf.register_ple_prefetch_consumer()

    def test_next_chunk_of_the_chunked_request_with_lead(self):
        done = types.SimpleNamespace(extend_range=Range(0, 5), full_untruncated_fill_ids=array("q", range(5)))
        chunked = types.SimpleNamespace(extend_range=Range(16, 32), full_untruncated_fill_ids=array("q", range(100, 150)))
        h = pf.publish_ple_next_chunk([chunked, done], 16)
        self.assertEqual((h.cur_start, h.chunk_size, h.next_start, h.lead), (16, 16, 32, 8))
        self.assertEqual(h.next_tokens.tolist(), list(range(124, 148)))
        self.assertIs(pf.take_ple_next_chunk(-1), h)
        self.assertIsNone(pf.take_ple_next_chunk(h.gen))
        # the last chunk: nothing to prefetch, the current start still known
        chunked.extend_range = Range(48, 50)
        h = pf.publish_ple_next_chunk([chunked], 16)
        self.assertEqual((h.cur_start, h.next_start, h.next_tokens), (48, -1, None))
        # the chunk size falls back to the current chunk's length
        chunked.extend_range = Range(1, 5)
        h = pf.publish_ple_next_chunk([chunked], None)
        self.assertEqual((h.lead, h.next_tokens.tolist()), (5, list(range(100, 109))))


class TestPrefetchGather(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = _table(self.tmp.name)
        g = torch.Generator().manual_seed(3)
        # 5 chunks of 200 rows; ids past total_rows (256) are zero rows
        self.ids = torch.randint(0, 300, (1000,), generator=g)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bytes_identical_to_serial_and_later_chunks_hit(self):
        run = _Run(self, self.table, self.ids, 200)
        try:
            with self.assertLogs(LOGGER, logging.INFO) as cm:
                for c in range(5):
                    ids, out = run.step(c)
                    self.assertTrue(_same(out, _serial(self.table, ids)), f"chunk {c}")
            lines = _parse(cm.records)
            self.assertEqual([l["chunk"] for l in lines], [0, 1, 2, 3, 4])
            self.assertEqual((lines[0]["ready"], lines[0]["hit"], lines[0]["read"]), ("none", 0, 200))
            for l in lines[1:]:
                self.assertEqual((l["hit"], l["read"]), (200, 0))
                self.assertIn(l["ready"], ("yes", "no"))
            self.assertIsNone(run.g._pending)  # the last chunk queued nothing
        finally:
            run.close()

    def test_prefetch_starts_after_the_chunk_and_join_waits_only_the_rest(self):
        run = _Run(self, self.table, self.ids, 200, delay_s=0.4)
        try:
            with self.assertLogs(LOGGER, logging.INFO) as cm:
                self.assertIsNone(run.g._pending)
                t = time.monotonic()
                run.step(0)
                first = time.monotonic() - t
                # chunk 0 paid one gather, not two: the prefetch runs behind it
                self.assertGreaterEqual(first, 0.4)
                self.assertLess(first, 0.75)
                pend = run.g._pending
                self.assertIsNotNone(pend)
                self.assertTrue(torch.equal(pend.ids, self.ids[200:400]))
                self.assertFalse(run.g._workers.ready(pend.seq))  # still reading
                time.sleep(0.15)  # a short forward: the join waits the rest
                run.step(1)
                time.sleep(0.6)  # a long forward: the join waits nothing
                run.step(2)
            lines = _parse(cm.records)
            self.assertEqual(lines[1]["ready"], "no")
            self.assertGreater(lines[1]["wait_ms"], 100.0)
            self.assertLess(lines[1]["wait_ms"], 350.0)
            self.assertEqual(lines[2]["ready"], "yes")
            self.assertLess(lines[2]["wait_ms"], 60.0)
            self.assertGreaterEqual(lines[2]["gather_ms"], 400.0)
        finally:
            run.close()

    def test_a_wrong_prediction_costs_reads_not_bytes(self):
        run = _Run(self, self.table, self.ids, 200)
        try:
            with self.assertLogs(LOGGER, logging.INFO) as cm:
                run.step(0)
                actual = self.ids[200:400].clone()
                actual[::7] = (actual[::7] + 1) % 300  # 29 rows differ from the prediction
                _, out = run.step(1, actual=actual)
                self.assertTrue(_same(out, _serial(self.table, actual)))
                # longer than predicted: the tail is read on the spot
                longer = self.ids[400:650]
                _, out = run.step(2, actual=longer)
                self.assertTrue(_same(out, _serial(self.table, longer)))
            lines = _parse(cm.records)
            self.assertEqual((lines[1]["hit"], lines[1]["read"]), (200 - 29, 29))
            self.assertEqual((lines[2]["hit"], lines[2]["read"]), (200, 50))
        finally:
            run.close()

    def test_vocab_shard_zero_rows_match_serial(self):
        run = _Run(self, self.table, self.ids, 200)
        try:
            ids = self.ids[:200]
            out = torch.empty((200, DIM), dtype=torch.bfloat16)
            run.g.gather_into(ids, out, vocab_start=64, vocab_end=192)
            self.assertTrue(_same(out, _serial(self.table, ids, (64, 192))))
            self.assertTrue(bool((out[(ids < 64) | (ids >= 192)] == 0).all()))
        finally:
            run.close()

    def test_ring_stays_two_deep(self):
        run = _Run(self, self.table, torch.randint(0, 256, (3000,)), 200)
        try:
            served = []
            for c in range(15):
                pend = run.g._pending
                run.step(c)
                if pend is not None:
                    served.append(pend.slot)
                    if run.g._pending is not None:
                        self.assertNotEqual(run.g._pending.slot, pend.slot)
            self.assertEqual(len(run.g._slots), 2)
            self.assertEqual(len(run.g._workers.slot_fds), 2)
            self.assertEqual(set(served), {0, 1})
            cap = max(1 << 15, 200) * RB
            self.assertEqual(run.g._workers.slot_bytes, [cap, cap])
            for fd in run.g._workers.slot_fds:
                self.assertEqual(os.fstat(fd).st_size, cap)
        finally:
            run.close()

    def test_switch_off_keeps_the_serial_gather(self):
        base = pt.PleCheckpointPreadGather(self.table, min_rows=16, workers=2)
        try:
            with envs.SGLANG_QWEN4_PLE_PREFETCH.override(False):
                self.assertIs(pf.make_ple_prefetch_gather(base, self.table, _identity_hasher), base)
            self.assertIsNone(pf.make_ple_prefetch_gather(None, self.table, _identity_hasher))
            self.assertIs(pf.make_ple_prefetch_gather(base, self.table, None), base)
            with envs.SGLANG_QWEN4_PLE_PREFETCH.override(True), \
                    envs.SGLANG_QWEN4_PLE_PREFETCH_PROCS.override(3):
                g = pf.make_ple_prefetch_gather(base, self.table, _identity_hasher)
                self.assertIsInstance(g, pf.PlePrefetchGather)
                self.assertEqual(g._n_procs, 3)
                g._disable()
        finally:
            base.close()

    def test_a_lost_worker_falls_back_to_the_serial_gather(self):
        run = _Run(self, self.table, self.ids, 200)
        try:
            run.step(0)
            for pid in run.g._workers.pids():
                os.kill(pid, 9)
            time.sleep(0.2)
            with self.assertLogs(LOGGER, logging.ERROR):
                ids, out = run.step(1)
            self.assertTrue(_same(out, _serial(self.table, ids)))
            self.assertTrue(run.g._disabled)
            ids, out = run.step(2)
            self.assertTrue(_same(out, _serial(self.table, ids)))
        finally:
            run.close()


if __name__ == "__main__":
    unittest.main()
