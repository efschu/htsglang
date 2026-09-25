"""fnFL2 H73 (SGLANG_WEG2_PLE_STAGE_AUTONOMOUS, SGLANG_WEG2_PLE_STAGE_BONUS_EARLY):
the pread WORKERS stage the verify round on their own. The device posts the
round's windows into a mailbox of the stage (then the round number into a flag
word); every worker polls the flag, hashes, reads the rows of its own stage
positions and publishes its own ``done`` word; the gate waits for all of them.
With BONUS_EARLY the next round's bonus rows are posted after the accept and
kept by the next verify round.

Desk only (CPU, real pread worker processes, real files; the kernels run in
the Triton interpreter on host memory, the table and stage pointers being
plain host addresses there, exactly as under HMM on the rig). Cases:

* the worker's stdlib hash is the model's ``_hash_contexts`` (and H40's);
* the all-words gate: not armed -> no poll; every word published -> go;
  one worker short -> bounded, a timeout; the last word written by another
  thread while it spins -> go; lanes past ``n_done`` never count;
* the layout: autonomous => gated, one done word per worker behind the ids,
  two mailboxes and a stats block; the H40/H69 layouts are unchanged;
* a round end to end: post -> the workers publish -> ids and bytes in the
  stage; the device FIRST (gate spinning), the post afterwards -> every row
  of the gather comes from the stage, bit-equal to the plain kernel;
* the bonus part: rows pre-staged after the accept are kept by the verify
  round (read 48, kept 16 at bs 1); another bonus is read again; bytes are
  always the file's;
* the worker's own logic in-process (``AutoStage`` on a plain buffer):
  position ownership ``r % procs``, a flag that never comes (missed, no
  done), an overtaken round, a stale announcement, a read error (id stays
  -1, done still published), the tail of a shorter round cleared, the bonus
  positions of a wider batch;
* the worker protocol: a bad share is refused (EINVAL), a remap drops it;
* round level: begin posts + arms (no hook, finish is a no-op, nothing
  published by the host), disarm; ``post_ple_bonus_stage``; the proof line;
* the switches (default off) reach the stager; the verify wiring posts the
  bonus after the mamba commit and the bonus fill.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import logging
import pathlib
import re
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch
from triton.runtime.interpreter import InterpretedFunction

from sglang.srt.environ import envs
from sglang.srt.models import qwen4_exp_ple_decode_pread as dp
from sglang.srt.models import qwen4_exp_ple_pread_worker as pw
from sglang.srt.models import qwen4_exp_ple_prefetch as pf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# the H40 file's fixtures: checkpoint-like files, the hash stand-ins, kernels
from test_qwen4_exp_ple_decode_pread_h40 import (  # noqa: E402
    DIM,
    EOS,
    RB,
    STAGED,
    TOTAL,
    _ctx,
    _emb,
    _Emb,
    _Files,
    _model_ids,
    _run_plain,
    _same,
    _small_emb,
    _stager,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LOGGER = "sglang.srt.models.qwen4_exp_ple_decode_pread"
SRC = pathlib.Path(dp.__file__).resolve().parents[1]
GATED = InterpretedFunction(dp._gather_ple_embedding_gated_kernel.fn)
PROCS = 2


def _gate_all():
    """The interpreted all-words gate (resolved lazily: absent before H73)."""
    return InterpretedFunction(dp._ple_stage_gate_all_kernel.fn)


def _wait(pred, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.002)
    return bool(pred())


@contextlib.contextmanager
def _interpreted_kernels():
    with mock.patch.object(dp, "_ple_stage_gate_all_kernel", _gate_all()), \
            mock.patch.object(dp, "_gather_ple_embedding_gated_kernel", GATED), \
            mock.patch.object(dp, "_gather_ple_embedding_staged_kernel", STAGED):
        yield


class TestWorkerHash(CustomTestCase):
    def test_stdlib_hash_is_the_model_hash(self):
        emb = _emb()
        p = dp.PleHashParamsPy.of(pf.PleHashParams.of(emb))
        g = torch.Generator().manual_seed(73)
        for rnd in range(40):
            bs, w = 1 + rnd % 4, 1 + rnd % 5
            ctx = torch.randint(0, 248320, (bs, 2 + w), generator=g)
            k = int(torch.randint(0, ctx.numel() + 1, (1,), generator=g))
            ctx.view(-1)[torch.randperm(ctx.numel(), generator=g)[:k]] = EOS
            got = pw.row_ids(
                ctx.tolist(), list(p.multipliers), list(p.head_vocab_sizes),
                list(p.head_offsets), p.heads_per_ngram, p.ngram_size, p.eos_token_id,
            )
            self.assertEqual(got, _model_ids(emb, ctx).tolist(), f"round {rnd}")
            self.assertEqual(got, dp.ple_verify_row_ids(ctx.tolist(), p))


def _gate_state(done, expect=-1):
    return (
        torch.tensor([expect], dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([done.data_ptr()], dtype=torch.int64),
    )


def _run_gate_all(gs, n_done, spins):
    expect, go, ctr, addr = gs
    _gate_all()[(1,)](expect, addr, go, ctr, N_DONE=n_done, MAX_SPINS=spins)
    return int(go[0]), ctr.tolist()


class TestGateAllKernel(CustomTestCase):
    def setUp(self):
        # three workers' words + the word behind them (not a worker's)
        self.done = torch.zeros(4, dtype=torch.int64)

    def test_not_armed_never_polls(self):
        self.assertEqual(_run_gate_all(_gate_state(self.done), 3, spins=1 << 30), (0, [0, 0]))

    def test_every_word_published_passes(self):
        self.done[:3] = torch.tensor([9, 7, 8])
        for expect in (1, 7):
            self.assertEqual(_run_gate_all(_gate_state(self.done, expect), 3, spins=4), (1, [1, 0]))

    def test_one_worker_short_times_out(self):
        for short in range(3):  # whichever worker it is
            self.done[:3] = 7
            self.done[short] = 6
            t0 = time.monotonic()
            self.assertEqual(_run_gate_all(_gate_state(self.done, 7), 3, spins=40), (0, [0, 1]))
            self.assertLess(time.monotonic() - t0, 30.0)

    def test_only_its_own_words_count(self):
        self.done[:3] = torch.tensor([7, 7, 7])
        self.done[3] = 0  # the word behind the three is not a worker's
        self.assertEqual(_run_gate_all(_gate_state(self.done, 7), 3, spins=4), (1, [1, 0]))
        self.assertEqual(_run_gate_all(_gate_state(self.done, 7), 1, spins=4), (1, [1, 0]))
        self.done[1] = 3
        self.assertEqual(_run_gate_all(_gate_state(self.done, 7), 1, spins=4), (1, [1, 0]))
        self.assertEqual(_run_gate_all(_gate_state(self.done, 7), 2, spins=4), (0, [0, 1]))

    def test_the_last_worker_releases_a_spinning_gate(self):
        self.done[:3] = torch.tensor([5, 5, 4])
        gs = _gate_state(self.done, 5)
        res = {}
        t = threading.Thread(target=lambda: res.update(r=_run_gate_all(gs, 3, spins=1 << 30)))
        t.start()
        time.sleep(0.05)
        self.assertTrue(t.is_alive())
        self.done[2] = 5  # the slowest worker's store
        t.join(timeout=60)
        self.assertFalse(t.is_alive())
        self.assertEqual(res["r"], (1, [1, 0]))


class _AutoCase(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = _Files(self.tmp.name)
        self.emb = _small_emb()
        self.p = dp.PleHashParamsPy.of(pf.PleHashParams.of(self.emb))
        self.stagers = []
        dp._MODEL_EMBEDDINGS.clear()

    def tearDown(self):
        for st in self.stagers:
            st.close()
        dp._MODEL_EMBEDDINGS.clear()
        self.tmp.cleanup()

    def _auto(self, **kw):
        kw.setdefault("procs", PROCS)
        kw.setdefault("threads", 2)
        kw.setdefault("gate_spins", 1 << 30)
        st = _stager(self.f.table, self.emb, autonomous=True, **kw)
        self.stagers.append(st)
        return st

    def _ids(self, ctx):
        return dp.ple_verify_row_ids(ctx.tolist(), self.p)

    def _stats(self, st, field):
        return int(st._wstats[:, field].sum())

    def _published(self, st, seq):
        return _wait(lambda: min(st.done_word.tolist()) >= seq)

    def _check_rows(self, st, ids, n=None):
        n = len(ids) if n is None else n
        self.assertEqual(st.stage_ids[:n].tolist(), list(ids[:n]))
        for j in range(n):
            self.assertTrue(_same(st.stage_rows[j], self.f.row(int(ids[j]))), f"row {j}")


class TestLayout(_AutoCase):
    def test_autonomous_layout(self):
        st = self._auto()
        cap, rb = st.capacity, self.f.table.row_bytes
        self.assertTrue(st.gated and st.autonomous)
        self.assertFalse(st.bonus_early)
        ids_end = cap * (rb + 8)
        self.assertEqual(
            st._nbytes,
            ids_end + dp._lines(8 * PROCS) + 2 * (dp._LINE + 8 * cap) + 8 * pw.STATS_WORDS * PROCS,
        )
        # one done word per worker, right behind the ids
        self.assertEqual(st.done_word.numel(), PROCS)
        self.assertEqual(st.done_word.data_ptr(), st.stage_ids.data_ptr() + cap * 8)
        self.assertEqual(st.done_word.tolist(), [0] * PROCS)
        lay = st._auto_layout()
        base = st._region.data_ptr()
        for part, (flag, ctx) in enumerate(st._mbox):
            key = "" if part == pw.PART_VERIFY else "_bonus"
            self.assertEqual(flag.data_ptr(), base + lay[f"flag{key}_off"])
            self.assertEqual(ctx.data_ptr(), base + lay[f"ctx{key}_off"])
            self.assertEqual(ctx.numel(), cap)
            self.assertEqual(int(flag[0]), 0)
            self.assertEqual(lay[f"flag{key}_off"] % dp._LINE, 0)
        self.assertEqual(tuple(st._wstats.shape), (PROCS, pw.STATS_WORDS))
        self.assertEqual(st._wstats.data_ptr(), base + lay["stats_off"])
        self.assertEqual(lay["stats_off"] + 8 * pw.STATS_WORDS * PROCS, st._nbytes)

    def test_h40_h69_layouts_unchanged_and_bonus_needs_autonomous(self):
        cap, rb = 4096, self.f.table.row_bytes
        off = _stager(self.f.table, self.emb)
        gated = _stager(self.f.table, self.emb, gated=True)
        lone_bonus = _stager(self.f.table, self.emb, bonus_early=True)
        self.stagers += [off, gated, lone_bonus]
        self.assertEqual(off._nbytes, cap * (rb + 8))
        self.assertEqual(gated._nbytes, cap * (rb + 8) + dp._GATE_BYTES)
        self.assertEqual(gated.done_word.numel(), 1)
        self.assertFalse(off.autonomous or gated.autonomous)
        self.assertFalse(lone_bonus.bonus_early or lone_bonus.autonomous)
        self.assertEqual(lone_bonus._nbytes, cap * (rb + 8))
        self.assertIsNone(gated._mbox)


class TestAutoRound(_AutoCase):
    def test_post_and_the_workers_stage_the_round(self):
        st = self._auto()
        ctx = _ctx(1, 4, seed=1, hi=100)
        ids = self._ids(ctx)
        self.assertEqual(len(ids), 64)
        self.assertTrue(st.post(ctx, 3))
        # the host never waited: the workers publish on their own
        self.assertTrue(self._published(st, 3))
        self.assertEqual(st.done_word.tolist(), [3] * PROCS)
        self._check_rows(st, ids)
        self.assertEqual(self._stats(st, pw.ST_ROUNDS), PROCS)
        self.assertEqual(self._stats(st, pw.ST_READ) + self._stats(st, pw.ST_REUSED), 64)
        self.assertEqual(self._stats(st, pw.ST_MISSED), 0)
        self.assertEqual(st._wstats[:, pw.ST_LAST].tolist(), [3] * PROCS)
        self.assertEqual(st.stats["rounds"], 1)
        self.assertEqual(st.stats["rows"], 64)
        # the mailbox holds the windows and the round
        self.assertEqual(st._mbox[0][1][:6].tolist(), ctx.reshape(-1).tolist())
        self.assertEqual(int(st._mbox[0][0][0]), 3)

    def test_device_first_then_post_serves_every_row_from_the_stage(self):
        st = self._auto()
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        self.assertTrue(st._ensure_auto())
        ctx = _ctx(1, 4, seed=2, hi=100)
        ids = torch.tensor(self._ids(ctx))
        st.arm_gate(5)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
        errors = []

        def device():
            try:
                with _interpreted_kernels():
                    ok = st.launch(ids, out, vocab_start=0, vocab_end=TOTAL, block_d=256)
                if not ok:
                    raise AssertionError("the autonomous stage did not serve the gather")
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)

        dev = threading.Thread(target=device)
        dev.start()
        time.sleep(0.05)
        self.assertTrue(dev.is_alive())  # the gate spins: no worker published 5
        self.assertTrue(st.post(ctx, 5))
        dev.join(timeout=120)
        self.assertFalse(dev.is_alive())
        if errors:
            raise errors[0]
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(st.counters.tolist(), [64, 64])  # every row from the stage
        self.assertEqual(st._gate_state(cpu, capturing=False)[2].tolist(), [1, 0])

    def test_a_round_no_worker_publishes_reads_hmm(self):
        st = self._auto(gate_spins=64)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        ctx = _ctx(1, 4, seed=3, hi=100)
        ids = torch.tensor(self._ids(ctx))
        st.arm_gate(dp._GATE_SEQ + 1000)  # a round nobody posted
        st.stage_ids[:64] = ids  # a lying stage under the right ids
        st.stage_rows[:64].view(torch.int16).random_(-30000, 30000)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16)
        with _interpreted_kernels():
            self.assertTrue(st.launch(ids, out, vocab_start=0, vocab_end=TOTAL, block_d=256))
        self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
        self.assertEqual(st.counters.tolist(), [64, 0])
        self.assertEqual(st._gate_state(cpu, capturing=False)[2].tolist(), [0, 1])

    def test_consecutive_rounds_and_a_shorter_one(self):
        st = self._auto()
        seq = 10
        for seed in (4, 5):
            seq += 1
            ctx = _ctx(1, 4, seed=seed, hi=100)
            self.assertTrue(st.post(ctx, seq))
            self.assertTrue(self._published(st, seq))
            self._check_rows(st, self._ids(ctx))
        # bs 1, one token: 16 rows; the old tail goes (hygiene, not correctness)
        short = _ctx(1, 1, seed=6, hi=100)
        seq += 1
        self.assertTrue(st.post(short, seq))
        self.assertTrue(self._published(st, seq))
        self._check_rows(st, self._ids(short))
        self.assertEqual(st.stage_ids[16:64].tolist(), [-1] * 48)

    def test_shapes_the_stage_does_not_take(self):
        st = self._auto()
        self.assertFalse(st.post(torch.zeros(6, dtype=torch.int64), 1))  # not 2-D
        self.assertFalse(st.post(torch.zeros((0, 6), dtype=torch.int64), 1))
        # 300 tokens x 16 rows > 4096: the gather keeps the plain kernel
        self.assertFalse(st.post(torch.zeros((60, 7), dtype=torch.int64), 1))
        self.assertEqual(st.stats["rounds"], 0)
        off = _stager(self.f.table, self.emb)
        self.stagers.append(off)
        self.assertFalse(off.post(_ctx(1, 4, seed=1), 1))  # not autonomous
        self.assertFalse(st.post(_ctx(1, 1, seed=1)[:, :3], 1, part=pw.PART_BONUS, width=4))


class TestBonusEarly(_AutoCase):
    def test_prestaged_bonus_rows_are_kept(self):
        st = self._auto(bonus_early=True)
        self.assertTrue(st.bonus_early)
        ctx = _ctx(1, 4, seed=7, hi=100)  # [h1 h2 | b d1 d2 d3]
        ids = self._ids(ctx)
        self.assertTrue(st.post(ctx[:, :3].contiguous(), 1, part=pw.PART_BONUS, width=4))
        self.assertTrue(_wait(lambda: self._stats(st, pw.ST_BONUS_ROUNDS) == PROCS))
        self.assertEqual(st.stage_ids[:16].tolist(), ids[:16])  # the bonus position
        self.assertEqual(st.stage_ids[16:64].tolist(), [-1] * 48)
        self.assertEqual(st.done_word.tolist(), [0] * PROCS)  # the bonus publishes nothing
        read_a = self._stats(st, pw.ST_BONUS_READ)
        self.assertTrue(st.post(ctx, 1))
        self.assertTrue(self._published(st, 1))
        self._check_rows(st, ids)
        self.assertEqual(self._stats(st, pw.ST_REUSED), 16)
        self.assertEqual(self._stats(st, pw.ST_READ), 48)
        self.assertEqual(read_a + self._stats(st, pw.ST_READ), 64)

    def test_another_bonus_is_read_again_and_a_wider_batch(self):
        st = self._auto(bonus_early=True)
        ctx = _ctx(2, 4, seed=8, hi=100)
        ids = self._ids(ctx)
        other = ctx[:, :3].clone()
        other[0, 2] += 1  # request 0's bonus was not the one the verify row starts with
        ids_other = dp.ple_verify_row_ids(other.tolist(), self.p)  # 2 x 16
        self.assertTrue(st.post(other, 4, part=pw.PART_BONUS, width=4))
        self.assertTrue(_wait(lambda: self._stats(st, pw.ST_BONUS_ROUNDS) == PROCS))
        # request i's bonus rows at positions (i * 4) * 16 + h
        self.assertEqual(st.stage_ids[0:16].tolist(), ids_other[0:16])
        self.assertEqual(st.stage_ids[64:80].tolist(), ids_other[16:32])
        self.assertTrue(st.post(ctx, 4))
        self.assertTrue(self._published(st, 4))
        self._check_rows(st, ids)  # bytes are the file's either way
        same = sum(ids[h] == ids_other[h] for h in range(16)) + 16  # request 1 kept whole
        self.assertEqual(self._stats(st, pw.ST_REUSED), same)
        self.assertEqual(self._stats(st, pw.ST_READ), 128 - same)


class _Buf:
    """A stage-sized plain buffer with torch views, for AutoStage in-process."""

    def __init__(self, st):
        self.raw = bytearray(st._nbytes)
        self.mv = memoryview(self.raw)
        self.q = torch.frombuffer(self.raw, dtype=torch.int64)
        cap = st.capacity
        ids0 = cap * st._rb // 8
        self.q[ids0 : ids0 + cap] = -1
        self.st = st
        self.lay = st._auto_layout()

    def ids(self, a, b):
        ids0 = self.st.capacity * self.st._rb // 8
        return self.q[ids0 + a : ids0 + b].tolist()

    def done(self):
        d0 = self.st._ids_end // 8
        return self.q[d0 : d0 + self.st._n_done].tolist()

    def post(self, ctx, seq, part=0):  # 0 = pw.PART_VERIFY
        key = "" if part == pw.PART_VERIFY else "_bonus"
        c0 = self.lay[f"ctx{key}_off"] // 8
        flat = ctx.reshape(-1)
        self.q[c0 : c0 + flat.numel()] = flat
        self.q[self.lay[f"flag{key}_off"] // 8] = seq

    def stat(self, index, field):
        return int(self.q[self.lay["stats_off"] // 8 + pw.STATS_WORDS * index + field])


class TestAutoStageInProcess(CustomTestCase):
    """The worker's own logic, no process: ownership, missed, overtaken, stale
    announcements, read errors, tails, bonus positions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = _Files(self.tmp.name)
        self.emb = _small_emb()
        self.p = dp.PleHashParamsPy.of(pf.PleHashParams.of(self.emb))
        # a stager only for its numbers (layout, share configs); never started
        self.st = _stager(self.f.table, self.emb, autonomous=True, procs=PROCS)
        self.buf = _Buf(self.st)
        self.fds = [os.open(path, os.O_RDONLY) for path in self.st._files]
        self.pool = ThreadPoolExecutor(2)
        self.autos = []

    def tearDown(self):
        for a in self.autos:
            a.release()
        self.pool.shutdown()
        for fd in self.fds:
            os.close(fd)
        self.st.close()
        self.tmp.cleanup()

    def _share(self, index, **over):
        cfg = self.st._auto_cfg(index, PROCS)
        cfg.update(over)
        a = pw.AutoStage(cfg, self.buf.mv, RB, 2)
        self.autos.append(a)
        return a

    def _round(self, a, seq, ctx, part=0, width=0):  # 0 = pw.PART_VERIFY
        a.round(part, seq, int(ctx.shape[0]), int(ctx.shape[1]), width, self.pool, self.fds)

    def test_each_worker_writes_only_its_positions(self):
        ctx = _ctx(1, 4, seed=11, hi=100)
        ids = dp.ple_verify_row_ids(ctx.tolist(), self.p)
        w0 = self._share(0)
        self.buf.post(ctx, 1)
        self._round(w0, 1, ctx)
        got = self.buf.ids(0, 64)
        self.assertEqual(got[0::2], ids[0::2])
        self.assertEqual(got[1::2], [-1] * 32)  # worker 1's positions untouched
        self.assertEqual(self.buf.done(), [1, 0])
        w1 = self._share(1)
        self._round(w1, 1, ctx)
        self.assertEqual(self.buf.ids(0, 64), ids)
        self.assertEqual(self.buf.done(), [1, 1])
        rows = self.buf.q.view(torch.bfloat16)  # the stage rows, by element
        for j in range(64):
            row = rows[j * DIM : (j + 1) * DIM]
            self.assertTrue(_same(row, self.f.row(ids[j])), f"row {j}")
        self.assertEqual(self.buf.stat(0, pw.ST_ROUNDS), 1)
        self.assertEqual(self.buf.stat(0, pw.ST_READ) + self.buf.stat(0, pw.ST_REUSED), 32)

    def test_a_flag_that_never_comes_is_missed_and_not_published(self):
        w0 = self._share(0, deadline_ms=20.0)
        ctx = _ctx(1, 4, seed=12, hi=100)
        t0 = time.monotonic()
        self._round(w0, 2, ctx)  # nothing posted
        self.assertGreaterEqual(time.monotonic() - t0, 0.015)
        self.assertEqual(self.buf.stat(0, pw.ST_MISSED), 1)
        self.assertEqual(self.buf.done(), [0, 0])
        self.assertEqual(self.buf.ids(0, 64), [-1] * 64)

    def test_overtaken_and_stale_announcements(self):
        w0 = self._share(0)
        a = _ctx(1, 4, seed=13, hi=100)
        b = _ctx(1, 4, seed=14, hi=100)
        self.buf.post(b, 5)  # round 5 already in the mailbox
        self._round(w0, 4, a)  # the announcement of 4 comes late: overtaken
        self.assertEqual(self.buf.stat(0, pw.ST_MISSED), 1)
        self.assertEqual(self.buf.done(), [0, 0])
        self._round(w0, 5, b)
        ids_b = dp.ple_verify_row_ids(b.tolist(), self.p)
        self.assertEqual(self.buf.ids(0, 64)[0::2], ids_b[0::2])
        self.assertEqual(self.buf.done(), [5, 0])
        rounds = self.buf.stat(0, pw.ST_ROUNDS)
        self._round(w0, 5, b)  # the same round again
        self._round(w0, 3, a)  # an older one
        self.assertEqual(self.buf.stat(0, pw.ST_ROUNDS), rounds)
        self.assertEqual(self.buf.done(), [5, 0])

    def test_a_read_error_leaves_the_id_off_and_still_publishes(self):
        w0 = self._share(0)
        ctx = _ctx(1, 4, seed=15, hi=100)
        self.buf.post(ctx, 1)
        with mock.patch.object(pw.os, "preadv", side_effect=OSError(5, "EIO")):
            self._round(w0, 1, ctx)
        self.assertEqual(self.buf.ids(0, 64)[0::2], [-1] * 32)  # HMM for these rows
        self.assertEqual(self.buf.done(), [1, 0])  # no gate waits out its bound
        self.assertEqual(self.buf.stat(0, pw.ST_ERRORS), 1)

    def test_bonus_positions_follow_the_verify_width(self):
        w0, w1 = self._share(0), self._share(1)
        ctx_a = _ctx(3, 1, seed=16, hi=100)  # [h1 h2 b] x 3 requests
        ids = dp.ple_verify_row_ids(ctx_a.tolist(), self.p)
        self.buf.post(ctx_a, 1, part=pw.PART_BONUS)
        for w in (w0, w1):
            self._round(w, 1, ctx_a, part=pw.PART_BONUS, width=4)
        for i in range(3):
            self.assertEqual(self.buf.ids(i * 64, i * 64 + 16), ids[i * 16 : (i + 1) * 16])
            self.assertEqual(self.buf.ids(i * 64 + 16, i * 64 + 64), [-1] * 48)
        self.assertEqual(self.buf.done(), [0, 0])
        self.assertEqual(self.buf.stat(0, pw.ST_BONUS_ROUNDS), 1)
        self.assertEqual(self.buf.stat(0, pw.ST_BONUS_READ) + self.buf.stat(1, pw.ST_BONUS_READ), 48)

    def test_bad_shares_are_refused(self):
        for over in ({"index": PROCS}, {"ids_off": 8}, {"stats_off": 1 << 40}):
            cfg = self.st._auto_cfg(0, PROCS)
            cfg.update(over)
            with self.assertRaises(ValueError):
                pw.AutoStage(cfg, self.buf.mv, RB, 2)


class TestWorkerProtocol(_AutoCase):
    def test_a_bad_share_is_refused_and_a_remap_drops_the_share(self):
        st = self._auto()
        w = st._ensure_workers()
        self.assertIsNotNone(w)
        bad = [st._auto_cfg(i, PROCS) for i in range(PROCS)]
        bad[1]["index"] = PROCS
        with self.assertRaises(pf.PleWorkerLost):
            w.configure_auto(0, bad)
        st2 = self._auto()
        self.assertTrue(st2._ensure_auto())
        w2 = st2._workers
        w2.map_slot(0, st2._nbytes)  # the workers release their views first
        self.assertTrue(all(p.poll() is None for p in w2._procs))
        # the share is gone: an announced round is ignored, nothing published
        ctx = _ctx(1, 4, seed=17, hi=100)
        self.assertTrue(st2.post(ctx, 1))
        time.sleep(0.3)
        self.assertEqual(st2.done_word.tolist(), [0] * PROCS)
        with self.assertRaises(ValueError):
            w2.configure_auto(0, [{}])  # one config per worker


class _Batch(types.SimpleNamespace):
    pass


class TestRoundLevel(_AutoCase):
    def _model(self, st):
        hist = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42]])
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), _Emb(st))
        pool = types.SimpleNamespace(
            get_mamba_indices=lambda req: req + 1,
            get_ngram_context=lambda idx: hist[idx],
        )
        batch = _Batch(
            forward_mode=types.SimpleNamespace(is_idle=lambda: False),
            req_pool_indices=torch.tensor([0]),
        )
        return model, pool, batch

    def test_begin_posts_and_arms_finish_does_nothing(self):
        st = self._auto()
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        gate = st._gate_state(cpu, capturing=False)
        model, pool, batch = self._model(st)
        vi = types.SimpleNamespace(draft_token=torch.tensor([5, 6, 7, 8]), draft_token_num=4)
        with mock.patch.object(st, "publish") as publish:
            stage = dp.begin_ple_verify_stage(model, pool, batch, vi)
            self.assertIsNotNone(stage)
            self.assertTrue(stage.autonomous)
            self.assertEqual(stage.seq, dp._GATE_SEQ)
            self.assertEqual(int(gate[0][0]), stage.seq)  # armed at begin
            self.assertFalse(dp.ple_stage_is_gated(stage))  # no hook, no second arm
            self.assertIs(dp.arm_ple_verify_gate(stage), stage)
            self.assertEqual(dp.finish_ple_verify_stage(stage), 0.0)
            publish.assert_not_called()
        # req 0 -> mamba 1 -> [21, 22] + the verify row
        ctx = torch.tensor([[21, 22, 5, 6, 7, 8]])
        self.assertEqual(st._mbox[0][1][:6].tolist(), ctx.reshape(-1).tolist())
        self.assertTrue(self._published(st, stage.seq))
        self._check_rows(st, self._ids(ctx))
        dp.disarm_ple_verify_gate(stage)
        self.assertEqual(int(gate[0][0]), -1)
        self.assertEqual(st.stats["rounds"], 1)

    def test_post_bonus_stage(self):
        st = self._auto(bonus_early=True)
        model, pool, batch = self._model(st)
        before = dp._BONUS_SEQ
        bonus = torch.tensor([9], dtype=torch.int32)
        self.assertTrue(dp.post_ple_bonus_stage(model, pool, batch, bonus, 4))
        self.assertEqual(dp._BONUS_SEQ, before + 1)
        self.assertEqual(st._mbox[1][1][:3].tolist(), [21, 22, 9])
        self.assertEqual(int(st._mbox[1][0][0]), dp._BONUS_SEQ)
        self.assertTrue(_wait(lambda: self._stats(st, pw.ST_BONUS_ROUNDS) == PROCS))
        want = self._ids(torch.tensor([[21, 22, 9]]))
        self.assertEqual(st.stage_ids[:16].tolist(), want)
        idle = _Batch(forward_mode=types.SimpleNamespace(is_idle=lambda: True),
                      req_pool_indices=batch.req_pool_indices)
        self.assertFalse(dp.post_ple_bonus_stage(model, pool, idle, bonus, 4))
        self.assertFalse(dp.post_ple_bonus_stage(model, pool, batch, torch.tensor([1, 2]), 4))
        self.assertFalse(dp.post_ple_bonus_stage(model, pool, batch, bonus, 0))
        self.assertFalse(dp.post_ple_bonus_stage(model, object(), batch, bonus, 4))
        self.assertFalse(dp.post_ple_bonus_stage(None, pool, batch, bonus, 4))
        self.assertEqual(dp._BONUS_SEQ, before + 1)

    def test_bonus_post_is_a_no_op_without_the_switch(self):
        st = self._auto()  # autonomous, no bonus
        model, pool, batch = self._model(st)
        self.assertFalse(dp.post_ple_bonus_stage(model, pool, batch, torch.tensor([9]), 4))
        self.assertEqual(int(st._mbox[1][0][0]), 0)

    def test_the_verify_paths_one_read_test_counts_live_bonus_stages(self):
        before = dp._LIVE_BONUS
        plain = self._auto()
        self.assertEqual(dp._LIVE_BONUS, before)
        bonus = self._auto(bonus_early=True)
        self.assertEqual(dp._LIVE_BONUS, before + 1)
        bonus.close()
        bonus.close()  # twice: counted once
        self.assertEqual(dp._LIVE_BONUS, before)
        model, pool, batch = self._model(plain)
        with mock.patch.object(dp, "_stagers_of", side_effect=AssertionError("scanned")):
            # no live bonus stage: not even the module scan
            self.assertFalse(dp.post_ple_bonus_stage(model, pool, batch, torch.tensor([9]), 4))

    def test_proof_line(self):
        st = self._auto(log_every=1)
        cpu = torch.device("cpu")
        st._device_state(cpu, capturing=False)
        st._gate_state(cpu, capturing=False)
        with self.assertLogs(LOGGER, logging.INFO) as cm:
            for seq, seed in ((1, 21), (2, 22)):
                ctx = _ctx(1, 4, seed=seed, hi=100)
                ids = torch.tensor(self._ids(ctx))
                st.snapshot_counters()  # what the post round does first
                self.assertTrue(st.post(ctx, seq))
                st.arm_gate(seq)
                self.assertTrue(self._published(st, seq))
                out = torch.empty(64, DIM, dtype=torch.bfloat16)
                with _interpreted_kernels():
                    st.launch(ids, out, vocab_start=0, vocab_end=TOTAL, block_d=256)
                self.assertTrue(_same(out, _run_plain(self.f.table, ids)))
                st.disarm_gate()
        lines = [r for r in cm.output if "PLE-DECODE-PREAD rounds=" in r]
        self.assertEqual(len(lines), 2)
        m = re.search(
            r"rounds=1 rows=64 hit=(\d+) late=(\d+) kernel_rows=(\d+) kernel_hit=(\d+) "
            r"wait_ms=[\d.]+ wait_max_ms=[\d.]+ read_ms=[\d.]+ sync_ms=0.00 procs=2 "
            r"gate_pass=(\d+) gate_timeout=(\d+) auto=1 stage_max_ms=[\d.]+ poll_ms=[\d.]+ "
            r"read_rows=(\d+) kept=(\d+) missed=(\d+) errors=(\d+) bonus_rows=0$",
            lines[1],
        )
        self.assertIsNotNone(m, lines[1])
        hit, late, k_rows, k_hit, g_pass, g_to, read, kept, missed, errors = map(int, m.groups())
        # the counters seen at the second post cover the first round's gather
        self.assertEqual((hit, late, k_rows, k_hit, g_pass, g_to), (64, 0, 64, 64, 1, 0))
        self.assertEqual((read + kept, missed, errors), (64, 0, 0))


class TestSwitches(_AutoCase):
    def test_default_off_and_reach_the_stager(self):
        self.assertFalse(envs.SGLANG_WEG2_PLE_STAGE_AUTONOMOUS.get())
        self.assertFalse(envs.SGLANG_WEG2_PLE_STAGE_BONUS_EARLY.get())
        fn = lambda: pf.PleHashParams.of(self.emb)  # noqa: E731
        for auto, bonus in ((False, False), (True, False), (True, True), (False, True)):
            with envs.SGLANG_WEG2_PLE_STAGE_AUTONOMOUS.override(auto), \
                    envs.SGLANG_WEG2_PLE_STAGE_BONUS_EARLY.override(bonus):
                st = dp.make_ple_decode_stager(
                    self.f.table, fn, vocab_start=0, vocab_end=TOTAL, device=torch.device("cpu")
                )
            self.stagers.append(st)
            self.assertEqual(st.autonomous, auto)
            self.assertEqual(st.bonus_early, auto and bonus)
            self.assertEqual(st.gated, auto)  # BEHIND_REPLAY stays off here
            self.assertEqual(st.done_word is not None, auto)
        self.assertEqual(st._n_done, 1)


def _between(text, start, end):
    i = text.index(start)
    return text[i : text.index(end, i + len(start))]


class TestVerifyWiring(CustomTestCase):
    def test_bonus_post_after_the_commit_and_the_bonus_fill(self):
        text = (SRC / "speculative" / "eagle_worker_v2.py").read_text()
        v = _between(text, "    def verify(self, batch: ScheduleBatch):",
                     "    def _finalize_accept_tree_path(")
        order = [
            "ple_stage = begin_ple_verify_stage(",
            "self.target_worker.forward_batch_generation(",
            "commit_mamba_states_after_verify(",
            "fill_bonus_tokens_func(",
            "post_ple_bonus_stage(",
            '_h58.note_span("accept_ms", _h58_t)',
        ]
        pos = [v.index(o) for o in order]
        self.assertEqual(pos, sorted(pos), list(zip(order, pos)))
        call = _between(v, "post_ple_bonus_stage(", "        else:")
        for arg in ("target_runner.model", "target_runner.req_to_token_pool", "batch",
                    "bonus_tokens", "int(verify_input.draft_token_num)"):
            self.assertIn(arg, call)
        # inside the non-idle branch (bonus_tokens exist there)
        branch = _between(v, "        if not batch.forward_mode.is_idle():\n            # #616",
                          "            bonus_tokens = torch.empty((0,)")
        self.assertIn("post_ple_bonus_stage(", branch)


if __name__ == "__main__":
    unittest.main()
