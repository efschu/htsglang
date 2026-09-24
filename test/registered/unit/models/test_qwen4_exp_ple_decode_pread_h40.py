"""fnFL2 H40 (SGLANG_QWEN4_PLE_DECODE_PREAD): the verify round's PLE rows,
read by pread worker processes before the replay, served from a host stage.

Desk only (CPU, real worker processes, real files; the kernel runs in the
Triton interpreter on host memory -- its table and stage pointers are plain
host addresses there, exactly as on the rig under HMM). Cases:

* the host hash of the verify windows is the model's ``_hash_contexts``
  (EOS anywhere, history included) -- and H32's torch mirror;
* the staged kernel's output is bit-equal to the plain HMM kernel whatever
  the stage holds: hits, stale pairs, -1, out-of-range rows; it really TAKES
  staged rows (a lying stage shows through) and counts them;
* a round: the rows are in the stage when ``stage`` returns (before the
  verify would launch), bytes equal to the serial pread gather, the kernel
  then serves every row from the stage;
* a late worker (fake slow worker, budget below it): its rows keep id -1
  (the kernel reads them through HMM, bytes unchanged), the next round
  drains it first;
* a shorter round clears the previous tail; rows outside the rank's
  vocabulary are never staged;
* a lost worker: the stage goes off (every id -1), no exception reaches the
  round;
* ``begin``/``finish`` hand ``[history | verify row]`` per request to the
  stage, and do nothing without a stage, for an idle batch or a foreign
  shape; the switch off builds no stage (the plain kernel stays);
* the proof line ``PLE-DECODE-PREAD`` every LOG_EVERY rounds;
* the wiring: verify() begins before and finishes right before the target
  forward; the embedding's gather tries the stage before the plain kernel.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import functools
import logging
import pathlib
import re
import signal
import tempfile
import time
import types
import unittest
from unittest import mock

import torch
from triton.runtime.interpreter import InterpretedFunction

from sglang.srt.environ import envs
from sglang.srt.models import qwen4_exp_ple_decode_pread as dp
from sglang.srt.models import qwen4_exp_ple_prefetch as pf
from sglang.srt.models import qwen4_exp_ple_table as pt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

ROWS_PER_SHARD = 64
SHARDS = 4
TOTAL = ROWS_PER_SHARD * SHARDS
DIM = 160
RB = DIM * 2
HEADER = 100
EOS = 7
LOGGER = "sglang.srt.models.qwen4_exp_ple_decode_pread"
SRC = pathlib.Path(dp.__file__).resolve().parents[1]

STAGED = InterpretedFunction(dp._gather_ple_embedding_staged_kernel.fn)


def _plain_kernel():
    from sglang.srt.models import qwen4_exp as q

    return InterpretedFunction(q._gather_ple_embedding_from_shards_kernel.fn)


class _Files:
    """Two files laid out like checkpoint shards (a header, then 2 shards
    each), plus host copies of their bytes that stand in for the mmaps."""

    def __init__(self, tmpdir):
        files, shard_files, shard_offsets, bases, self.keep = [], [], [], [], []
        for f in range(2):
            path = os.path.join(tmpdir, f"ple-{f}.safetensors")
            data = bytearray(os.urandom(HEADER + 2 * ROWS_PER_SHARD * RB))
            with open(path, "wb") as fh:
                fh.write(data)
            buf = torch.frombuffer(data, dtype=torch.uint8)
            self.keep.append((data, buf))
            files.append(path)
            for s in range(2):
                off = HEADER + s * ROWS_PER_SHARD * RB
                shard_files.append(path)
                shard_offsets.append(off)
                bases.append(buf.data_ptr() + off)
        self.table = pt.CheckpointMappedPleTable(
            bases, ROWS_PER_SHARD, TOTAL, torch.bfloat16, DIM, keepalive=[],
            files=files, shard_files=shard_files, shard_offsets=shard_offsets,
        )

    def row(self, i):
        s, r = divmod(int(i), ROWS_PER_SHARD)
        data = self.keep[s // 2][1]
        off = HEADER + (s % 2) * ROWS_PER_SHARD * RB + r * RB
        return data[off : off + RB].view(torch.bfloat16)


def _same(a, b):
    """Bitwise (random bytes as bf16 carry NaNs)."""
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _emb(seed=7, eos=EOS, sizes=None):
    """A Qwen4ExpNGramEmbedding stand-in: ngram 3, 8 heads per n-gram."""
    from sglang.srt.models import qwen4_exp as q

    g = torch.Generator().manual_seed(seed)
    if sizes is None:
        sizes = torch.tensor([20000003 + 2 * i for i in range(16)], dtype=torch.long)
    offs = torch.cat([sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]])
    mult = torch.randint(1 << 40, 1 << 62, (3,), generator=g) | 1
    emb = types.SimpleNamespace(
        enable_ple_fusion=False, ngram_size=3, heads_per_ngram=8, layer_multipliers=mult,
        ngram_heads_vocab_sizes=sizes, ngram_heads_offsets=offs, eos_token_id=eos,
    )
    emb._shift_right_ignore_eos = functools.partial(q.Qwen4ExpNGramEmbedding._shift_right_ignore_eos, emb)
    return emb


def _small_emb():
    """Head sizes small enough that every id is a row of the 256-row table."""
    return _emb(sizes=torch.tensor([13, 11, 17, 19, 7, 5, 3, 23, 13, 11, 17, 19, 7, 5, 3, 23], dtype=torch.long))


def _model_ids(emb, ctx):
    """The model's ids for [bs, hist + w]: cat(history, row).unfold -> _hash_contexts."""
    from sglang.srt.models import qwen4_exp as q

    pool = types.SimpleNamespace(ple_window_cache=None)
    windows = ctx.unfold(1, 3, 1).reshape(-1, 3)
    with mock.patch.object(q, "get_req_to_token_pool", lambda: pool):
        return q.Qwen4ExpNGramEmbedding._hash_contexts(emb, windows).reshape(-1)


def _run_staged(table, ids, stage_ids, stage_rows, vocab=(0, TOTAL)):
    out = torch.full((ids.numel(), DIM), 3.0, dtype=torch.bfloat16)
    addrs = torch.tensor([stage_ids.data_ptr(), stage_rows.data_ptr()], dtype=torch.int64)
    ctr = torch.zeros(2, dtype=torch.int32)
    STAGED[(ids.numel(),)](
        torch.tensor(table.bases, dtype=torch.int64), table.shard_rows, ids, addrs, ctr, out,
        embedding_dim=DIM, tp_vocab_start=vocab[0], tp_vocab_end=vocab[1], is_fp8=False, BLOCK_D=256,
    )
    return out, ctr.tolist()


def _run_plain(table, ids, vocab=(0, TOTAL)):
    out = torch.full((ids.numel(), DIM), 3.0, dtype=torch.bfloat16)
    _plain_kernel()[(ids.numel(),)](
        torch.tensor(table.bases, dtype=torch.int64), table.shard_rows, ids, out,
        embedding_dim=DIM, tp_vocab_start=vocab[0], tp_vocab_end=vocab[1], is_fp8=False, BLOCK_D=256,
    )
    return out


def _stager(table, emb, **kw):
    kw.setdefault("procs", 2)
    kw.setdefault("threads", 2)
    kw.setdefault("budget_s", 5.0)
    kw.setdefault("log_every", 1000)
    kw.setdefault("vocab_start", 0)
    kw.setdefault("vocab_end", TOTAL)
    return dp.PleDecodeStager(table, functools.partial(pf.PleHashParams.of, emb), device=torch.device("cpu"), **kw)


def _stage(st, ctx):
    t = time.monotonic()
    st.stage(ctx.tolist(), sync_s=0.0, t_ready=t)


def _ctx(bs, w, seed, hi=248320):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (bs, 2 + w), generator=g)


class TestHostHash(CustomTestCase):
    def test_verify_windows_hash_like_the_model(self):
        emb = _emb()
        p = dp.PleHashParamsPy.of(pf.PleHashParams.of(emb))
        g = torch.Generator().manual_seed(11)
        for rnd in range(40):
            bs, w = 1 + rnd % 4, 1 + rnd % 5
            ctx = torch.randint(0, 248320, (bs, 2 + w), generator=g)
            # EOS anywhere: in the history, on the bonus, inside the drafts
            k = int(torch.randint(0, ctx.numel() + 1, (1,), generator=g))
            ctx.view(-1)[torch.randperm(ctx.numel(), generator=g)[:k]] = EOS
            want = _model_ids(emb, ctx)
            got = dp.ple_verify_row_ids(ctx.tolist(), p)
            self.assertEqual(got, want.tolist(), f"round {rnd}: {ctx.tolist()}")
            h32 = pf.ple_ngram_lookup_ids(ctx.unfold(1, 3, 1).reshape(-1, 3), pf.PleHashParams.of(emb)).reshape(-1)
            self.assertEqual(got, h32.tolist())
        self.assertEqual(len(dp.ple_verify_row_ids([[1, 2, 3, 4, 5, 6]], p)), 4 * 16)

    def test_eos_cases_by_hand(self):
        emb = _emb()
        p = dp.PleHashParamsPy.of(pf.PleHashParams.of(emb))
        for ctx in ([[EOS, EOS, 5, 6, 9, 10]], [[3, EOS, 5, 6, 9, 10]], [[3, 4, EOS, 6, EOS, 10]],
                    [[3, 4, 5, EOS, 9, 10]], [[EOS, 4, 5, 6, 9, EOS]]):
            self.assertEqual(dp.ple_verify_row_ids(ctx, p), _model_ids(emb, torch.tensor(ctx)).tolist(), ctx)


class TestStagedKernel(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = _Files(self.tmp.name)
        g = torch.Generator().manual_seed(5)
        self.ids = torch.randint(0, TOTAL + 40, (64,), generator=g)  # some out of range
        self.rows = torch.empty(64, DIM, dtype=torch.bfloat16)
        self.sid = torch.full((64,), -1, dtype=torch.int64)

    def tearDown(self):
        self.tmp.cleanup()

    def _garbage(self):
        self.rows.view(torch.int16).random_(-30000, 30000)

    def test_bytes_equal_the_plain_kernel_whatever_the_stage_holds(self):
        want = _run_plain(self.f.table, self.ids)
        self._garbage()
        # 1) empty stage
        out, ctr = _run_staged(self.f.table, self.ids, self.sid, self.rows)
        self.assertTrue(_same(out, want))
        in_range = int((self.ids < TOTAL).sum())
        self.assertEqual(ctr, [in_range, 0])
        # 2) half the rows staged under their id, the rest stale pairs
        #    (another id WITH that id's bytes) or -1 over garbage
        stale = 0
        for j in range(64):
            i = int(self.ids[j])
            if i >= TOTAL:
                self.sid[j] = i  # an out-of-range id is never taken from the stage
            elif j % 3 == 0:
                self.sid[j] = i
                self.rows[j] = self.f.row(i)
            elif j % 3 == 1:
                other = (i + 1) % TOTAL
                self.sid[j] = other
                self.rows[j] = self.f.row(other)
                stale += 1
        out, ctr = _run_staged(self.f.table, self.ids, self.sid, self.rows)
        self.assertTrue(_same(out, want))
        hits = sum(1 for j in range(64) if j % 3 == 0 and int(self.ids[j]) < TOTAL)
        self.assertEqual(ctr, [in_range, hits])
        self.assertTrue(stale > 0 and hits > 0)
        self.assertTrue(torch.all(out[self.ids >= TOTAL].float() == 0))

    def test_staged_rows_are_really_taken(self):
        # a stage that lies (right id, wrong bytes) shows through -- the
        # kernel does read the stage on a hit and only there
        lie = [j for j in range(64) if int(self.ids[j]) < TOTAL][:5]
        self._garbage()
        for j in lie:
            self.sid[j] = self.ids[j]
        out, ctr = _run_staged(self.f.table, self.ids, self.sid, self.rows)
        want = _run_plain(self.f.table, self.ids)
        for j in range(64):
            if j in lie:
                self.assertTrue(_same(out[j], self.rows[j]))
            else:
                self.assertTrue(_same(out[j], want[j]))
        self.assertEqual(ctr[1], len(lie))

    def test_vocab_bounds_like_the_plain_kernel(self):
        vocab = (64, 192)
        for j in range(64):
            self.sid[j] = self.ids[j]
            if int(self.ids[j]) < TOTAL:
                self.rows[j] = self.f.row(int(self.ids[j]))
        out, ctr = _run_staged(self.f.table, self.ids, self.sid, self.rows, vocab)
        self.assertTrue(_same(out, _run_plain(self.f.table, self.ids, vocab)))
        inside = int(((self.ids >= 64) & (self.ids < 192)).sum())
        self.assertEqual(ctr, [inside, inside])


class TestStagerRound(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = _Files(self.tmp.name)
        self.emb = _small_emb()
        self.p = dp.PleHashParamsPy.of(pf.PleHashParams.of(self.emb))
        self.stagers = []

    def tearDown(self):
        for st in self.stagers:
            st.close()
        self.tmp.cleanup()

    def _mk(self, **kw):
        st = _stager(self.f.table, self.emb, **kw)
        self.stagers.append(st)
        return st

    def _serve(self, st, ids):
        rows = st.stage_rows[: ids.numel()]
        return _run_staged(self.f.table, ids, st.stage_ids[: ids.numel()], rows)

    def test_rows_ready_before_the_verify_and_bitwise(self):
        st = self._mk(delay_s=0.05)
        ctx = _ctx(1, 4, seed=1, hi=100)
        t0 = time.monotonic()
        _stage(st, ctx)
        took = time.monotonic() - t0
        ids = torch.tensor(dp.ple_verify_row_ids(ctx.tolist(), self.p))
        self.assertEqual(ids.numel(), 64)
        self.assertTrue(torch.all(ids < TOTAL))
        # the round waited for the (slow) workers: every row is in when stage returns
        self.assertGreaterEqual(took, 0.05)
        self.assertEqual(st.stage_ids[:64].tolist(), ids.tolist())
        for j in range(64):
            self.assertTrue(_same(st.stage_rows[j], self.f.row(int(ids[j]))))
        out, ctr = self._serve(st, ids)
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(ctr, [64, 64])
        self.assertEqual((st.stats["rows"], st.stats["hit"], st.stats["late"]), (64, 64, 0))

    def test_late_worker_falls_back_and_is_drained_next_round(self):
        st = self._mk(delay_s=0.3, budget_s=0.02)
        ctx = _ctx(1, 4, seed=2, hi=100)
        t0 = time.monotonic()
        _stage(st, ctx)
        self.assertLess(time.monotonic() - t0, 0.25)  # the budget bounds the wait
        ids = torch.tensor(dp.ple_verify_row_ids(ctx.tolist(), self.p))
        self.assertEqual(st.stage_ids[:64].tolist(), [-1] * 64)  # no id without its bytes
        self.assertEqual((st.stats["hit"], st.stats["late"]), (0, 64))
        out, ctr = self._serve(st, ids)
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))  # HMM path, same bytes
        self.assertEqual(ctr, [64, 0])
        # next round (budget now enough): the late gather is joined first
        st._budget_s = 5.0
        ctx2 = _ctx(1, 4, seed=3, hi=100)
        t0 = time.monotonic()
        _stage(st, ctx2)
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)
        ids2 = torch.tensor(dp.ple_verify_row_ids(ctx2.tolist(), self.p))
        self.assertEqual(st.stage_ids[:64].tolist(), ids2.tolist())
        out, ctr = self._serve(st, ids2)
        self.assertTrue(_same(out, _run_plain(self.f.table, ids2)))
        self.assertEqual(ctr, [64, 64])

    def test_shorter_round_clears_the_tail_and_vocab_is_respected(self):
        st = self._mk(vocab_start=0, vocab_end=100)
        ctx = _ctx(3, 4, seed=4, hi=100)
        _stage(st, ctx)
        ids = dp.ple_verify_row_ids(ctx.tolist(), self.p)
        self.assertEqual(len(ids), 192)
        want = [i if i < 100 else -1 for i in ids]
        self.assertEqual(st.stage_ids[:192].tolist(), want)
        self.assertIn(-1, want)
        _stage(st, ctx[:1])
        ids1 = dp.ple_verify_row_ids(ctx[:1].tolist(), self.p)
        self.assertEqual(st.stage_ids[:64].tolist(), [i if i < 100 else -1 for i in ids1])
        self.assertEqual(st.stage_ids[64:256].tolist(), [-1] * 192)

    def _partial_round(self, st, seed):
        """A round with worker 1 stopped: worker 0's rows staged, worker 1's late."""
        pids = st._workers.pids() if st._workers is not None else None
        if pids is None:
            _stage(st, _ctx(1, 4, seed=99, hi=100))  # starts the workers
            pids = st._workers.pids()
        os.kill(pids[1], signal.SIGSTOP)
        st._budget_s = 0.15
        ctx = _ctx(1, 4, seed=seed, hi=100)
        _stage(st, ctx)
        return pids, ctx

    def test_partially_late_round_stages_only_answered_rows(self):
        st = self._mk()
        pids, ctx = self._partial_round(st, seed=21)
        try:
            ids = torch.tensor(dp.ple_verify_row_ids(ctx.tolist(), self.p))
            staged = st.stage_ids[:64]
            answered = st._workers.parts[0]
            late = st._workers.parts[1]
            self.assertEqual(answered.numel() + late.numel(), 64)
            self.assertEqual(staged[answered].tolist(), ids[answered].tolist())
            self.assertEqual(staged[late].tolist(), [-1] * late.numel())
            for j in answered.tolist():
                self.assertTrue(_same(st.stage_rows[j], self.f.row(int(ids[j]))))
            out, ctr = self._serve(st, ids)
            self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
            self.assertEqual(ctr, [64, answered.numel()])
            self.assertEqual(st.stats["late"], late.numel())
        finally:
            os.kill(pids[1], signal.SIGCONT)
        st._budget_s = 5.0
        ctx2 = _ctx(1, 4, seed=22, hi=100)
        _stage(st, ctx2)  # drains the late worker first
        ids2 = torch.tensor(dp.ple_verify_row_ids(ctx2.tolist(), self.p))
        self.assertEqual(st.stage_ids[:64].tolist(), ids2.tolist())

    def test_lost_worker_switches_the_stage_off(self):
        st = self._mk()
        pids, _ = self._partial_round(st, seed=5)
        self.assertTrue(bool((st.stage_ids[:64] >= 0).any()))  # worker 0's rows
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
        with self.assertLogs(LOGGER, logging.ERROR) as cm:
            _stage(st, _ctx(1, 4, seed=6, hi=100))  # the drain of worker 1 fails
        self.assertTrue(any("PLE-DECODE-PREAD disabled" in r for r in cm.output))
        self.assertFalse(st.active)
        self.assertEqual(st.stage_ids.tolist(), [-1] * st.capacity)
        _stage(st, _ctx(1, 4, seed=7, hi=100))  # no-op, no exception

    def test_retired_stage_keeps_its_memory_and_serves_nothing(self):
        st = self._mk()
        _stage(st, _ctx(1, 4, seed=8, hi=100))
        addr = st.stage_ids.data_ptr()
        with self.assertLogs(LOGGER, logging.ERROR):
            st.retire()
        self.assertFalse(st.active)
        self.assertIsNone(st._workers)
        self.assertEqual(st.stage_ids.data_ptr(), addr)  # a captured graph may still read it
        self.assertEqual(st.stage_ids.tolist(), [-1] * st.capacity)

    def test_proof_line_every_log_every_rounds(self):
        st = self._mk(log_every=2)
        with self.assertLogs(LOGGER, logging.INFO) as cm:
            for s in range(4):
                ctx = _ctx(1, 4, seed=10 + s, hi=100)
                ids = torch.tensor(dp.ple_verify_row_ids(ctx.tolist(), self.p))
                st.snapshot_counters()  # what begin() does, behind the draft
                _stage(st, ctx)
                self._serve_into(st, ids)
        lines = [r for r in cm.output if "PLE-DECODE-PREAD rounds=" in r]
        self.assertEqual(len(lines), 2)
        m = re.search(r"rounds=(\d+) rows=(\d+) hit=(\d+) late=(\d+) kernel_rows=(\d+) kernel_hit=(\d+) "
                      r"wait_ms=([\d.]+) wait_max_ms=([\d.]+) read_ms=([\d.]+) sync_ms=([\d.]+) procs=(\d+)", lines[1])
        self.assertIsNotNone(m, lines[1])
        self.assertEqual([int(m[i]) for i in (1, 2, 3, 4)], [2, 128, 128, 0])
        # the counters seen at a round's begin cover the verifies before it
        self.assertEqual((int(m[5]), int(m[6])), (128, 128))
        self.assertGreater(float(m[7]), 0.0)
        self.assertTrue(any("PLE-DECODE-PREAD on:" in r for r in cm.output))

    def _serve_into(self, st, ids):
        """The verify's gather through the stager's own counters."""
        addrs, ctr = st._device_state(torch.device("cpu"), capturing=False)
        out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
        STAGED[(ids.numel(),)](
            torch.tensor(self.f.table.bases, dtype=torch.int64), ROWS_PER_SHARD, ids, addrs, ctr, out,
            embedding_dim=DIM, tp_vocab_start=0, tp_vocab_end=TOTAL, is_fp8=False, BLOCK_D=256,
        )
        return out


class _FakeStager:
    active = True

    def __init__(self):
        self.rows = None
        self.snaps = 0

    def snapshot_counters(self):
        self.snaps += 1

    def stage(self, rows, *, sync_s, t_ready):
        self.rows = rows


class _Emb(torch.nn.Module):
    def __init__(self, stager):
        super().__init__()
        self._decode_stager = stager


class TestBeginFinish(CustomTestCase):
    def setUp(self):
        dp._MODEL_EMBEDDINGS.clear()
        self.fake = _FakeStager()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2), _Emb(self.fake))
        hist = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42]])
        self.pool = types.SimpleNamespace(
            get_mamba_indices=lambda req: req + 1,
            get_ngram_context=lambda idx: hist[idx],
        )
        self.batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_idle=lambda: False),
            req_pool_indices=torch.tensor([2, 0]),
        )
        self.vi = types.SimpleNamespace(draft_token=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]), draft_token_num=4)

    def tearDown(self):
        dp._MODEL_EMBEDDINGS.clear()

    def test_history_and_verify_row_per_request(self):
        with mock.patch.object(dp, "_LIVE_STAGERS", 1):
            stage = dp.begin_ple_verify_stage(self.model, self.pool, self.batch, self.vi)
            self.assertIsNotNone(stage)
            self.assertEqual(self.fake.snaps, 1)
            self.assertIsNone(self.fake.rows)  # begin never waits nor stages
            dp.finish_ple_verify_stage(stage)
        # req 2 -> mamba 3 -> [41, 42]; req 0 -> mamba 1 -> [21, 22]
        self.assertEqual(self.fake.rows, [[41, 42, 1, 2, 3, 4], [21, 22, 5, 6, 7, 8]])

    def test_nothing_to_stage(self):
        dp.finish_ple_verify_stage(None)
        with mock.patch.object(dp, "_LIVE_STAGERS", 0):
            self.assertIsNone(dp.begin_ple_verify_stage(self.model, self.pool, self.batch, self.vi))
        with mock.patch.object(dp, "_LIVE_STAGERS", 1):
            idle = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_idle=lambda: True),
                                         req_pool_indices=self.batch.req_pool_indices)
            self.assertIsNone(dp.begin_ple_verify_stage(self.model, self.pool, idle, self.vi))
            odd = types.SimpleNamespace(draft_token=torch.arange(7), draft_token_num=4)
            self.assertIsNone(dp.begin_ple_verify_stage(self.model, self.pool, self.batch, odd))
            self.assertIsNone(dp.begin_ple_verify_stage(self.model, object(), self.batch, self.vi))
            self.assertIsNone(dp.begin_ple_verify_stage(torch.nn.Linear(2, 2), self.pool, self.batch, self.vi))
            self.fake.active = False
            self.assertIsNone(dp.begin_ple_verify_stage(self.model, self.pool, self.batch, self.vi))
        self.assertIsNone(self.fake.rows)


class TestSwitch(CustomTestCase):
    def test_off_builds_no_stage_on_builds_one(self):
        with tempfile.TemporaryDirectory() as d:
            f = _Files(d)
            fn = functools.partial(pf.PleHashParams.of, _small_emb())
            with envs.SGLANG_QWEN4_PLE_DECODE_PREAD.override(False):
                self.assertIsNone(dp.make_ple_decode_stager(f.table, fn, vocab_start=0, vocab_end=TOTAL))
            with envs.SGLANG_QWEN4_PLE_DECODE_PREAD.override(True):
                self.assertIsNone(dp.make_ple_decode_stager(f.table, None, vocab_start=0, vocab_end=TOTAL))
                st = dp.make_ple_decode_stager(f.table, fn, vocab_start=0, vocab_end=TOTAL, device=torch.device("cpu"))
                try:
                    self.assertIsInstance(st, dp.PleDecodeStager)
                    self.assertEqual(st.stage_ids.tolist(), [-1] * st.capacity)
                    self.assertIsNone(st._workers)  # no process before a capture or a round
                    # device state only from a non-capturing launch (the graph's
                    # warm-up), never made inside a capture
                    self.assertIsNone(st.counters)
                    st.snapshot_counters()  # nothing to copy yet, no error
                    self.assertIsNone(st._device_state(torch.device("cpu"), capturing=True))
                    made = st._device_state(torch.device("cpu"), capturing=False)
                    self.assertIs(st._device_state(torch.device("cpu"), capturing=True), made)
                    self.assertEqual(made[0].tolist(), [st.stage_ids.data_ptr(), st.stage_rows.data_ptr()])
                finally:
                    st.close()
        self.assertTrue(envs.SGLANG_QWEN4_PLE_DECODE_PREAD.get())  # default on

    def test_oversized_gather_keeps_the_plain_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            f = _Files(d)
            st = _stager(f.table, _small_emb(), capacity=16)
            try:
                out = torch.empty(17, DIM, dtype=torch.bfloat16)
                self.assertFalse(st.launch(torch.arange(17), out, vocab_start=0, vocab_end=TOTAL, block_d=256))
            finally:
                st.close()


class TestWiring(CustomTestCase):
    def test_verify_begins_then_finishes_before_the_target_forward(self):
        src = (SRC / "speculative" / "eagle_worker_v2.py").read_text()
        body = src[src.index("    def verify(self, batch: ScheduleBatch):"):]
        body = body[: body.index("\n    def ", 10)]
        b = body.index("ple_stage = begin_ple_verify_stage(")
        p = body.index("eagle_prepare_for_verify(")
        f = body.index("finish_ple_verify_stage(ple_stage)")
        fwd = body.index("self.target_worker.forward_batch_generation(")
        self.assertLess(b, p)
        self.assertLess(p, f)
        self.assertLess(f, fwd)
        self.assertNotIn("forward_batch_generation(", body[f:fwd].replace("with _module_sync_window(", ""))

    def _gather(self, stager):
        """``Qwen4ExpPinnedHostEmbedding.gather`` on a checkpoint table, the
        plain kernel replaced by a recorder."""
        from sglang.srt.models import qwen4_exp as q

        calls = []

        class _Plain:
            def __getitem__(self, grid):
                return lambda *a, **k: calls.append(("plain", grid))

        emb = types.SimpleNamespace(
            embedding_dim=DIM, _ckpt_backend=True, _ckpt_pread=None, _ckpt_prefetcher=None,
            _ckpt_table=types.SimpleNamespace(bases_on=lambda d: None, shard_rows=64, dtype=torch.bfloat16),
            _decode_stager=stager, _block_d=256,
            shard_indices=types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=TOTAL),
        )
        ids = torch.arange(8).view(2, 4)
        out = torch.empty(2, 4, DIM, dtype=torch.bfloat16)
        with mock.patch.object(q, "_gather_ple_embedding_from_shards_kernel", _Plain()):
            got = q.Qwen4ExpPinnedHostEmbedding.gather(emb, ids, out)
        self.assertIs(got, out)
        return calls

    def test_gather_takes_the_stage_else_the_plain_kernel(self):
        seen = []

        class _St:
            def __init__(self, serve):
                self.serve = serve

            def launch(self, flat_ids, output, **kw):
                seen.append((flat_ids.tolist(), kw))
                return self.serve

        self.assertEqual(self._gather(_St(True)), [])
        self.assertEqual(seen[0][0], list(range(8)))
        self.assertEqual(seen[0][1], dict(vocab_start=0, vocab_end=TOTAL, block_d=256))
        self.assertEqual(self._gather(_St(False)), [("plain", (8,))])  # not served: plain kernel
        self.assertEqual(self._gather(None), [("plain", (8,))])  # switch off: the old path

    def test_layer_and_table_wiring(self):
        src = (SRC / "models" / "qwen4_exp.py").read_text()
        self.assertIn("self._decode_stager = make_ple_decode_stager(", src)
        self.assertIn("decode_stage_params = (", src)


if __name__ == "__main__":
    unittest.main()
