# SPDX-License-Identifier: Apache-2.0
"""Group D host gap, 26.09. (HG): three levers and one instrument, all default off.

* SGLANG_WEG2_D_EARLY_DRAFT -- the draft replay before the host wait when the
  exact compact draft lengths are known without round N's lengths (every
  committed length >= the draft window, page size 1);
* SGLANG_DFLASH_ACCEPT_SYNC_FUSED -- the five rank-0 broadcasts of the Triton
  accept outputs as ONE broadcast of a flat buffer the outputs are views of;
* SGLANG_BARLINK_BAR1_CANON_ORDER -- the oneshot all_reduce sums in rank order
  0..R-1 on every rank (bitwise equal results across ranks);
* SGLANG_WEG2_D_HOSTGAP_SPLIT -- the ``#DGAP-SPLIT`` line (post-wake marks).

CPU only. What is pinned: the decision is exact (never engages when the window
value could differ from the waited path), the values the draft plans with are
byte-identical to the waited path, the switches are off by default with the
old line / old buffers / old kernel order, the fused broadcast moves exactly
the bytes the five did, and the canonical order is rank-independent where the
old one is not (bf16 emulation of the kernel's per-add rounding).
"""

import ast
import inspect
import os
import re
import textwrap
import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from sglang.srt.managers import weg2_d_hostgap as dg  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class _Ev:
    def synchronize(self):
        pass


def _pending(lower_bound):
    p = dg.PendingSeqLensCpu(torch.zeros(8, dtype=torch.int64), _Ev(),
                             torch.tensor([0]))
    p.lower_bound = lower_bound
    return p


def _ready(lower_bound):
    return partial(_pending(lower_bound).complete, SimpleNamespace())


# ----------------------------------------------------------------------------
# early draft: the decision
# ----------------------------------------------------------------------------
class TestEarlyDraftDecision(CustomTestCase):
    W = 2048

    def test_switch_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dg.D_EARLY_DRAFT_ENV, None)
            self.assertFalse(dg.early_draft_on())
            b = SimpleNamespace(seq_lens_cpu=torch.tensor([5000]))
            self.assertIsNone(dg.snapshot_lower_bound(b))
            os.environ[dg.D_EARLY_DRAFT_ENV] = "1"
            self.assertTrue(dg.early_draft_on())
            os.environ.pop(dg.D_EARLY_DRAFT_ENV, None)

    def test_snapshot_is_an_int64_copy(self):
        with mock.patch.dict(os.environ, {dg.D_EARLY_DRAFT_ENV: "1"}):
            seed = torch.tensor([4000, 5000], dtype=torch.int32)
            b = SimpleNamespace(seq_lens_cpu=seed)
            lo = dg.snapshot_lower_bound(b)
            self.assertEqual(lo.dtype, torch.int64)
            b.seq_lens_cpu = None  # what the deferred resolve does next
            seed.fill_(0)          # the prepare buffer is reused next round
            self.assertEqual(lo.tolist(), [4000, 5000])
            self.assertIsNone(dg.snapshot_lower_bound(SimpleNamespace(seq_lens_cpu=None)))

    def test_lower_bound_rides_on_the_scheduler_partial(self):
        lo = torch.tensor([3000])
        self.assertIs(dg.pending_lower_bound(_ready(lo)), lo)
        self.assertIsNone(dg.pending_lower_bound(None))
        self.assertIsNone(dg.pending_lower_bound(lambda: None))
        self.assertIsNone(dg.pending_lower_bound(partial(print, 1)))

    def test_engages_only_when_every_row_is_past_the_window(self):
        W = self.W
        ok = lambda lo, **kw: dg.early_draft_window_exact(
            _ready(torch.tensor(lo, dtype=torch.int64)), len(lo), W,
            kw.get("page", 1), kw.get("upper"))
        self.assertTrue(ok([W]))
        self.assertTrue(ok([W, 58000, W + 1]))
        self.assertFalse(ok([W - 1]))
        self.assertFalse(ok([58000, W - 1]))
        self.assertFalse(ok([58000], page=16))  # page-aligned envelope, not exact
        # bs mismatch (the batch changed after the snapshot) -> keep the wait
        self.assertFalse(dg.early_draft_window_exact(
            _ready(torch.tensor([58000])), 2, W, 1))
        # no lower bound / no ready / no window
        self.assertFalse(dg.early_draft_window_exact(_ready(None), 1, W, 1))
        self.assertFalse(dg.early_draft_window_exact(None, 1, W, 1))
        self.assertFalse(dg.early_draft_window_exact(
            _ready(torch.tensor([58000])), 1, None, 1))
        # belt: a "lower bound" above the reservation bound is refused
        self.assertFalse(ok([60000], upper=torch.tensor([59000], dtype=torch.int32)))
        self.assertTrue(ok([58000], upper=torch.tensor([58016], dtype=torch.int32)))
        self.assertFalse(ok([58000], upper=torch.tensor([58016, 7])))

    def test_window_values_equal_the_waited_path(self):
        """The draft plans with seq_lens_cpu and its sum. For every exact
        length >= lower bound >= W, the waited path's host value
        (_compute_compact_draft_seq_lens_host on the published mirror) equals
        the early fill -- byte for byte, int32 like the buffer."""
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        W = self.W
        fake = SimpleNamespace(draft_window_size=W, page_size=1)
        g = torch.Generator().manual_seed(26)
        for _ in range(200):
            bs = int(torch.randint(1, 7, (1,), generator=g))
            lo = torch.randint(W, 200_000, (bs,), generator=g)
            exact = lo + torch.randint(0, 64, (bs,), generator=g)
            self.assertTrue(dg.early_draft_window_exact(_ready(lo), bs, W, 1, exact + 16))
            waited = torch.empty(bs, dtype=torch.int32)
            DFlashWorkerV2._compute_compact_draft_seq_lens_host(fake, exact, waited)
            early = torch.empty(bs, dtype=torch.int32)
            early.fill_(W)
            self.assertTrue(torch.equal(waited, early))
            self.assertEqual(int(waited.sum().item()), int(early.sum().item()))
        # and where the decision refuses, the two can differ (the guard matters)
        waited = torch.empty(1, dtype=torch.int32)
        DFlashWorkerV2._compute_compact_draft_seq_lens_host(
            fake, torch.tensor([W - 3]), waited)
        self.assertNotEqual(int(waited[0]), W)


# ----------------------------------------------------------------------------
# early draft: the worker and the scheduler, on the source
# ----------------------------------------------------------------------------
def _src(fn):
    return textwrap.dedent(inspect.getsource(fn))


class TestEarlyDraftWiring(CustomTestCase):
    def test_worker_reads_the_switches_once(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        init_src = inspect.getsource(DFlashWorkerV2.__init__)
        self.assertIn("self._early_draft = early_draft_on()", init_src)
        self.assertIn('"SGLANG_DFLASH_ACCEPT_SYNC_FUSED", "") == "1"', init_src)

    def test_order_in_the_draft_prep(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        src = _src(DFlashWorkerV2.forward_batch_generation)
        keys = [
            "noise_embedding = embed_module(block_ids)",
            "_defer_rebuild = (",
            "_early_draft = bool(",
            "if seq_lens_cpu_ready is not None and not _defer_rebuild:",
            "if not _early_draft:\n",
            "elif _early_draft:\n",
            "rebuild_window_rows_sync_free(",
            "if _early_draft:\n",
            "draft_seq_lens_sum = int(seq_lens_cpu.sum().item())",
            "draft_out = self.draft_model_runner.forward(forward_batch)",
            "_dgap.note_early_draft()",
            # the verify's belt: the wait an early round still owes
            "# --- 2) Target verify.",
            "seq_lens_cpu_ready()",
        ]
        pos, start = [], 0
        for k in keys:
            p = src.find(k, start)
            pos.append(p)
            if p >= 0:
                start = p
        self.assertTrue(all(p >= 0 for p in pos), list(zip(keys, pos)))
        self.assertEqual(pos, sorted(pos), list(zip(keys, pos)))
        guard = src[src.find("_early_draft = bool("):
                    src.find("if seq_lens_cpu_ready is not None and not _defer_rebuild:")]
        for need in ("self._early_draft", "seq_lens_cpu_ready is not None",
                     "self.use_compact_draft_cache", "self.draft_window_size",
                     "self.page_size", "draft_input.nxt_kv_lens_cpu"):
            self.assertIn(need, guard)

    def test_early_rounds_fill_the_window_and_skip_only_the_draft_waits(self):
        """Inside the draft prep every wait is gated by `not _early_draft`
        and every early branch fills the window and keeps the exact flag."""
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        fn = ast.parse(_src(DFlashWorkerV2.forward_batch_generation)).body[0]
        draft_call = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                          and ast.unparse(n.func) == "self.draft_model_runner.forward")
        early_if = [n for n in ast.walk(fn) if isinstance(n, (ast.If,))
                    and n.lineno < draft_call
                    and ast.unparse(n.test) in ("_early_draft", "not _early_draft")]
        self.assertGreaterEqual(len(early_if), 3)
        for n in early_if:
            body = "\n".join(ast.unparse(b) for b in n.body)
            if ast.unparse(n.test) == "_early_draft":
                self.assertIn("seq_lens_cpu.fill_(int(self.draft_window_size))", body)
            else:
                self.assertIn("seq_lens_cpu_ready()", body)
        # every draft-prep wait (before the draft launch, after the embedding)
        # sits under a `not _early_draft` / else-of-`_early_draft` branch
        embed = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
                     and ast.unparse(n).startswith("noise_embedding = embed_module"))
        guarded = set()
        for n in early_if:
            branch = n.body if ast.unparse(n.test) == "not _early_draft" else n.orelse
            for b in branch:
                for c in ast.walk(b):
                    if isinstance(c, ast.Call) and ast.unparse(c.func) == "seq_lens_cpu_ready":
                        guarded.add(c.lineno)
        waits = [c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and ast.unparse(c.func) == "seq_lens_cpu_ready"
                 and embed < c.lineno < draft_call]
        self.assertTrue(waits)
        self.assertEqual(set(waits), guarded, (waits, guarded))

    def test_scheduler_snapshots_before_the_resolve(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._run_batch_forward)
        keys = [
            "_d_hostgap.defer_eligible(self, batch)",
            "_lens_lo = _d_hostgap.snapshot_lower_bound(batch)",
            "resolve_seq_lens_cpu(\n                        batch, defer=True",
            "_pending_lens.lower_bound = _lens_lo",
            'fwd_kwargs["seq_lens_cpu_ready"] = partial(',
        ]
        pos = [src.find(k) for k in keys]
        self.assertTrue(all(p >= 0 for p in pos), list(zip(keys, pos)))
        self.assertEqual(pos, sorted(pos), list(zip(keys, pos)))


# ----------------------------------------------------------------------------
# the meter: early counter + split marks, default line unchanged
# ----------------------------------------------------------------------------
_OLD_LINE = re.compile(
    r"^#DGAP rounds=\d+ period_ms=[\d.]+ result_wait_ms=[\d.]+ \(publish [\d.]+, "
    r"copy_done [\d.]+\) hicache_ms=[\d.]+ host_other_ms=-?[\d.]+ crit_ms=(nan|[\d.]+) "
    r"\(max [\d.]+, n=\d+; publish wake -> draft replay launched\) deferred=\d+/\d+ "
    r"\(host clocks only; allreduce is device time inside the replays\)$"
)


def _run_rounds(m, clk, n, *, early=False, marks=()):
    m.end_round(False)
    line = None
    for _ in range(n):
        clk.t += 0.030
        if early:
            m.mark("draft")          # before the wake: not counted
            m.note_early_draft()
            m.mark_draft_launched()  # no wake yet: no crit sample
        m.mark_wake()
        for lab, dt in marks:
            clk.t += dt
            m.mark(lab)
        if not early:
            m.mark_draft_launched()
        line = m.end_round(True)
    return line


class TestMeterLevers(CustomTestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in (dg.D_EARLY_DRAFT_ENV, dg.D_HOSTGAP_SPLIT_ENV):
            os.environ.pop(k, None)

    def tearDown(self):
        self._env.stop()
        dg.reset_for_tests()

    def test_default_line_is_the_old_line(self):
        clk = _Clock()
        m = dg.DHostGap(4, clock=clk)
        with self.assertLogs(dg.logger, level="INFO") as cm:
            line = _run_rounds(m, clk, 4, marks=(("draft", 0.002),))
        self.assertRegex(line, _OLD_LINE)
        self.assertFalse(any("#DGAP-SPLIT" in r for r in cm.output))
        self.assertIsNone(m._split)

    def test_early_rounds_are_counted_and_have_no_crit(self):
        os.environ[dg.D_EARLY_DRAFT_ENV] = "1"
        clk = _Clock()
        m = dg.DHostGap(4, clock=clk)
        line = _run_rounds(m, clk, 4, early=True)
        self.assertIn("crit_ms=nan", line)
        self.assertTrue(line.endswith(" early_draft=4/4"), line)
        # the counter restarts with the window
        clk.t += 5.0
        line = _run_rounds(m, clk, 4)
        self.assertTrue(line.endswith(" early_draft=0/4"), line)

    def test_split_marks_after_the_wake(self):
        clk = _Clock()
        m = dg.DHostGap(2, clock=clk, split=True)
        with self.assertLogs(dg.logger, level="INFO") as cm:
            _run_rounds(m, clk, 2, early=True,
                        marks=(("vprep", 0.0005), ("vload", 0.0010), ("verify", 0.0002)))
        split = [r for r in cm.output if "#DGAP-SPLIT" in r]
        self.assertEqual(len(split), 1, cm.output)
        s = split[0]
        self.assertIn("draft=-", s)          # reached before the wake only
        self.assertIn("vprep=0.500", s)
        self.assertIn("vload=1.500", s)
        self.assertIn("verify=1.700", s)
        self.assertIn("n=0,0,0,0,2,2,2", s)

    def test_split_mark_is_inert_when_off(self):
        dg.reset_for_tests()
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: ""}):
            dg.split_mark("load")  # no meter: nothing happens
            self.assertIsNone(dg.meter())
        dg.reset_for_tests()
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: "8"}):
            m = dg.meter()
            self.assertIsNone(m._split)
            m.mark_wake()
            dg.split_mark("load")
            self.assertEqual(m._split_n["load"], 0)
        dg.reset_for_tests()
        with mock.patch.dict(os.environ, {dg.D_HOSTGAP_ENV: "8",
                                          dg.D_HOSTGAP_SPLIT_ENV: "1"}):
            m = dg.meter()
            m.mark_wake()
            dg.split_mark("load")
            self.assertEqual(m._split_n["load"], 1)

    def test_graph_runner_marks_load(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        src = _src(DecodeCudaGraphRunner.execute)
        a = src.find("self.load_batch(forward_batch, pp_proxy_tensors)")
        b = src.find("_d_hostgap_split_mark(")
        self.assertIn('"load" if self.model_runner.is_draft_model_runner else "vload"',
                      src[b: b + 120])
        c = src.find("output = self.backend.replay(")
        self.assertTrue(0 <= a < b < c, (a, b, c))


# ----------------------------------------------------------------------------
# fused accept broadcast
# ----------------------------------------------------------------------------
def _worker(fused, block=8):
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    w = DFlashWorkerV2.__new__(DFlashWorkerV2)
    w.device = torch.device("cpu")
    w.block_size = block
    w._accept_bonus_buffer_cap = 0
    w._accept_bonus_buffer_slot = 0
    w._accept_len_buf = None
    w._commit_lens_bufs = []
    w._bonus_id_bufs = []
    w._out_tokens_bufs = []
    w._new_seq_lens_bufs = []
    w._accept_sync_fused = fused
    w._accept_bonus_flats = []
    w._accept_bonus_last_flat = None
    return w


class TestFusedAcceptSync(CustomTestCase):
    def test_layout_alignment(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        for cap in range(1, 17):
            for block in (1, 4, 8, 16):
                (a, c, b, n, o), total = DFlashWorkerV2._fused_accept_layout(cap, block)
                self.assertEqual((a, c, b), (0, cap, 2 * cap))
                self.assertGreaterEqual(n, 3 * cap)
                self.assertEqual(n % 2, 0)
                self.assertEqual(o, n + 2 * cap)
                self.assertEqual(total, o + 2 * cap * block)

    def test_same_shapes_dtypes_as_the_old_buffers(self):
        for bs in (1, 2, 3, 5, 6):
            old, new = _worker(False), _worker(True)
            ro = old._next_accept_bonus_buffers(bs)
            rn = new._next_accept_bonus_buffers(bs)
            for x, y in zip(ro, rn):
                self.assertEqual((x.shape, x.dtype), (y.shape, y.dtype))
                self.assertTrue(y.is_contiguous())
            self.assertIsNone(old._accept_bonus_last_flat)
            flat = new._accept_bonus_last_flat
            self.assertIsNotNone(flat)
            base = flat.untyped_storage().data_ptr()
            for y in rn:  # every output is a view of this round's flat
                self.assertEqual(y.untyped_storage().data_ptr(), base)

    def test_slots_alternate_and_do_not_alias(self):
        w = _worker(True)
        r0 = w._next_accept_bonus_buffers(2)
        f0 = w._accept_bonus_last_flat
        r1 = w._next_accept_bonus_buffers(2)
        f1 = w._accept_bonus_last_flat
        self.assertNotEqual(f0.data_ptr(), f1.data_ptr())
        for t in r0:
            t.fill_(7)
        for t in r1:
            self.assertEqual(int(t.abs().sum()), 0)
        w._next_accept_bonus_buffers(2)
        self.assertIs(w._accept_bonus_last_flat, f0)

    def test_growth_keeps_the_fused_layout(self):
        w = _worker(True)
        w._next_accept_bonus_buffers(1)
        w._next_accept_bonus_buffers(5)
        self.assertGreaterEqual(w._accept_bonus_buffer_cap, 5)
        rn = w._next_accept_bonus_buffers(5)
        base = w._accept_bonus_last_flat.untyped_storage().data_ptr()
        self.assertTrue(all(t.untyped_storage().data_ptr() == base for t in rn))

    def test_one_broadcast_moves_the_bytes_of_five(self):
        """Rank 0 and rank 1 fill their outputs differently; the five
        per-tensor broadcasts (old) and the one flat broadcast (new) leave
        rank 1 with byte-identical values -- rank 0's."""
        g = torch.Generator().manual_seed(3)

        def fill(outs):
            acc, com, bon, out, nsl = outs
            acc.copy_(torch.randint(0, 8, acc.shape, generator=g, dtype=torch.int32))
            com.copy_(acc + 1)
            bon.copy_(torch.randint(0, 150000, bon.shape, generator=g, dtype=torch.int32))
            out.copy_(torch.randint(0, 150000, out.shape, generator=g))
            nsl.copy_(torch.randint(2048, 1 << 40, nsl.shape, generator=g))

        for bs in (1, 3, 6):
            src_old, dst_old = _worker(False), _worker(False)
            src_new, dst_new = _worker(True), _worker(True)
            so, do = src_old._next_accept_bonus_buffers(bs), dst_old._next_accept_bonus_buffers(bs)
            sn, dn = src_new._next_accept_bonus_buffers(bs), dst_new._next_accept_bonus_buffers(bs)
            fill(so)
            for a, b in zip(sn, so):
                a.copy_(b)
            fill(do)
            fill(dn)
            for s, d in zip(so, do):          # old: five broadcasts
                d.copy_(s)
            dst_new._accept_bonus_last_flat.copy_(src_new._accept_bonus_last_flat)  # new: one
            for x, y in zip(do, dn):
                self.assertTrue(torch.equal(x, y))

    def test_call_site_broadcasts_the_flat_once(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        src = _src(DFlashWorkerV2.forward_batch_generation)
        blk = src[src.find("if self._accept_sync_fused:"):]
        blk = blk[: blk.find("except Exception as e:")]
        self.assertIn("SpecTpSyncSite.DFLASH_ACCEPT_GREEDY,\n", blk)
        self.assertIn("self._accept_bonus_last_flat,", blk)
        self.assertIn("for _t in (accept_len, commit_lens, bonus, out_tokens, new_seq_lens):", blk)
        self.assertEqual(blk.count("self._tp_sync.sync("), 2)  # fused / old loop


# ----------------------------------------------------------------------------
# barlink oneshot: canonical rank order
# ----------------------------------------------------------------------------
def _ext_src():
    from sglang.srt.distributed.device_communicators import barlink_bar1_ext as ext

    return ext._CUDA_SRC


def _oneshot_old(xs, rank):
    s = xs[rank].clone()
    for q in range(len(xs)):
        if q == rank:
            continue
        s = s + xs[q]  # bf16 + bf16 rounds once per add, like __hadd2
    return s


def _oneshot_canon(xs, rank):
    s = xs[0].clone()
    for q in range(1, len(xs)):
        s = s + xs[q]
    return s


class TestBar1CanonOrder(CustomTestCase):
    def test_algo_codes(self):
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            bar1_algo_code,
        )

        self.assertEqual([bar1_algo_code(a) for a in ("mesh", "ring", "oneshot")], [0, 1, 2])
        self.assertEqual([bar1_algo_code(a, True) for a in ("mesh", "ring", "oneshot")], [0, 1, 3])

    def test_switch_default_off_and_read_at_init(self):
        from sglang.srt.distributed.device_communicators import barlink_bar1 as b1

        src = inspect.getsource(b1)
        self.assertIn('"SGLANG_BARLINK_BAR1_CANON_ORDER", ""\n        ) not in ("", "0", "no", "off", "false")', src)
        self.assertIn('bar1_algo_code(algo, getattr(self, "canon_order", False))', src)

    def test_kernel_source(self):
        s = _ext_src()
        # the oneshot picks the canonical reduction only under A.canon
        self.assertIn(
            "if (A.canon) reduceNPhaseCanon<T>(A.in, A.out, sRecvRS, R, r, n4, tid, nth);\n"
            "    else         reduceNPhase<T>(A.in, A.out, sRecvRS, R, r, n4, tid, nth);", s)
        # mesh still uses the old reduction (its chunks are reduced by one rank)
        self.assertEqual(s.count("reduceNPhaseCanon<T>("), 1)
        # 3 is folded into 2 BEFORE anything reads algo, and sets A.canon
        body = s[s.find("void bar1_all_reduce("):]
        i_map = body.find("if (canonOrder) algo = 2;")
        i_first_use = body.find("if (algo == 2) {")
        self.assertTrue(0 <= i_map < i_first_use, (i_map, i_first_use))
        self.assertIn("A.canon     = canonOrder ? 1 : 0;", body)
        # memset(0) of the args struct keeps canon = 0 for every other call
        self.assertIn("std::memset(&A, 0, sizeof(A));", body)
        # canonical form: own contribution from `in`, peers from recv, rank 0 first
        canon = s[s.find("void reduceNPhaseCanon("):]
        canon = canon[: canon.find("\n}\n")]
        self.assertIn("uint4 s = (rank == 0) ? in[j] : readV4(recv[0] + j);", canon)
        self.assertIn("for (int q = 1; q < R; ++q)", canon)
        self.assertIn("(q == rank) ? in[j] : readV4(recv[q] + j)", canon)
        self.assertNotIn("barrier", canon)

    def test_bf16_emulation_old_is_rank_dependent_canon_is_not(self):
        g = torch.Generator().manual_seed(12)
        R, n = 3, 20 * 1024 // 2  # a 20 KiB bf16 draft all_reduce
        xs = [torch.randn(n, generator=g).to(torch.bfloat16) * (10 ** (i - 1))
              for i in range(R)]
        old = [_oneshot_old(xs, r) for r in range(R)]
        can = [_oneshot_canon(xs, r) for r in range(R)]
        # old: ranks 0 and 1 agree (addition commutes), rank 2 does not
        self.assertTrue(torch.equal(old[0].view(torch.int16), old[1].view(torch.int16)))
        self.assertFalse(torch.equal(old[0].view(torch.int16), old[2].view(torch.int16)))
        # canonical: all ranks bitwise equal, and equal to rank 0's old result
        for r in range(R):
            self.assertTrue(torch.equal(can[r].view(torch.int16), can[0].view(torch.int16)))
        self.assertTrue(torch.equal(can[0].view(torch.int16), old[0].view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
