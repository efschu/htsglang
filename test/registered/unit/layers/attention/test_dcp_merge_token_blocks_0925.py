# SPDX-License-Identifier: Apache-2.0
"""27B RC7 (25.09.): the uneven-DCP LSE merge in TOKEN BLOCKS.

Boot weg2rc7_186c022f80 D TP0 (5090) died at 09:22:19 in
``cp_lse_ag_out_a2a_mha_uneven``: ``Tried to allocate 144.00 MiB`` = the a2a
``recv`` (W*H_local, T, D) fp32 = (3*12, 4096, 256) * 4 B, T = 4096 = the
chunk (``#969 EXTENT ... (4096, 8192, 4096, 4096)``: prefix 4096, extend
4096). A prefix-bearing forward at full chunk width (chunk 2+ of a D-direct
prefill under the X ceiling, or a multi-turn follow-up with <= X uncached
tokens over a D-radix prefix) runs the one-shot merge at ~483 MiB working set
on TP0; X <= 4096 boots never went past 231 prefix-bearing rows.

Pinned here, on CPU, no GPU:

* REAL gloo collectives on 3 and 2 ranks: blockwise == one-shot for the a2a and
  the all_reduce merge, uneven and even head splits, with and without
  ``return_lse``, fp32 and bf16 wire; the edge cases T = 0 (one block), 1,
  the block width, width + 1, and a wide T;
* the SAME collective sequence on every rank, 2 per block (LSE all-gather +
  head exchange), and the weightless head/worker lockstep (guard step per
  block, identical on head and workers);
* one block is today's call: the very result object, the same arguments;
* the derived width: 404 rows for the 27B TP0 a2a/fp32 geometry, rank-uniform,
  and the byte model it rests on bounds the real allocations of both bodies
  (counted with a TorchDispatchMode, so a later edit to the merge that
  allocates more fails here instead of on metal);
* the local final merge in the same blocks is byte-identical;
* every flashinfer merge site (3 head, 2 weightless worker) carries the width.
"""

import inspect
import os
import socket
import unittest
from unittest import mock

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

from sglang.srt.layers.dcp import comm  # noqa: E402


def _reset_merge_cache():
    comm._LSE_MERGE["mode"] = None
    comm._LSE_MERGE["dtype"] = None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------
# a real gloo group with the GroupCoordinator surface the merges use
# ---------------------------------------------------------------------------
class _GlooGroup:
    def __init__(self, rank, world):
        import torch.distributed as dist

        self._dist = dist
        self.rank_in_group = rank
        self.world_size = world
        self.log = []

    def all_gather(self, t, dim=0):
        assert dim == 0
        t = t.contiguous()
        self.log.append(("all_gather", tuple(t.shape)))
        out = t.new_empty((self.world_size * t.shape[0],) + tuple(t.shape[1:]))
        self._dist.all_gather_into_tensor(out, t)
        return out

    def all_reduce(self, t):
        # barlink's all_reduce is out-of-place; mirror that contract
        self.log.append(("all_reduce", tuple(t.shape)))
        out = t.clone()
        self._dist.all_reduce(out)
        return out

    def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
        self.log.append(("all_to_all", tuple(input.shape)))
        self._dist.all_to_all_single(
            output, input.contiguous(), output_split_sizes, input_split_sizes
        )
        return output


def _inputs(rank, T, H, D, dtype):
    g = torch.Generator().manual_seed(1000 + rank)
    o = torch.randn(T, H, D, generator=g).to(dtype)
    lse = torch.randn(T, H, generator=g) * 3
    if T >= 3:
        # an empty partial (lse -inf, o 0) and a poisoned one, as the kernels emit
        lse[1, 0] = float("-inf")
        o[1, 0] = 0
        o[2, 1, 0] = float("nan")
        lse[2, 1] = float("-inf")
    return o, lse


_CASES = [
    # (mode, wire, counts, T, block)
    ("a2a", "fp32", [4, 2, 2], 13, 5),
    ("a2a", "fp32", [4, 2, 2], 5, 5),
    ("a2a", "fp32", [4, 2, 2], 6, 5),
    ("a2a", "fp32", [4, 2, 2], 1, 5),
    ("a2a", "bf16", [4, 2, 2], 11, 4),
    ("a2a", "fp32", [3, 3, 3], 10, 3),
    ("ar", "fp32", [4, 2, 2], 13, 5),
    ("ar", "fp32", [4, 2, 2], 6, 5),
    ("ar", "bf16", [4, 2, 2], 11, 4),
    ("ar", "fp32", [3, 3, 3], 10, 3),
]


def _worker(rank, world, port, cases, q):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from sglang.srt.layers.dcp import comm as c

        res = []
        for idx, (mode, wire, counts, T, blk) in enumerate(cases):
            counts = counts[:world] if len(counts) >= world else counts
            H, D = sum(counts), 8
            c._LSE_MERGE["mode"] = mode
            c._LSE_MERGE["dtype"] = wire
            fn = (c.cp_lse_ag_out_a2a_mha_uneven if mode == "a2a"
                  else c.cp_lse_ag_out_ar_mha_uneven)
            for return_lse in (False, True):
                for odt in (torch.float32, torch.bfloat16):
                    o, lse = _inputs(rank, T, H, D, odt)
                    grp = _GlooGroup(rank, world)
                    one = fn(o, lse, grp, counts, return_lse=return_lse)
                    one_log = list(grp.log)
                    grp.log.clear()
                    blk_res = c.cp_lse_merge_token_blocks(
                        fn, o, lse, grp, counts, return_lse=return_lse, block_tokens=blk
                    )
                    one_o, one_l = one if return_lse else (one, None)
                    b_o, b_l = blk_res if return_lse else (blk_res, None)
                    n_blocks = len(c.lse_merge_token_spans(T, blk))
                    exact = torch.equal(one_o, b_o) and (
                        not return_lse or torch.equal(one_l, b_l))
                    tol = 1e-6 if wire == "fp32" else 1.6e-2  # bf16 wire: bf16 rounding
                    close = torch.allclose(one_o, b_o, rtol=tol, atol=tol, equal_nan=True)
                    maxdiff = float((one_o.double() - b_o.double()).abs().max()) if T else 0.0
                    res.append({
                        "case": idx, "return_lse": return_lse, "odt": str(odt),
                        "mode": mode, "n_blocks": n_blocks,
                        "exact": bool(exact), "close": bool(close), "maxdiff": maxdiff,
                        "wire": wire,
                        "shape": tuple(b_o.shape), "dtype": str(b_o.dtype),
                        "one_shape": tuple(one_o.shape),
                        "one_log": one_log, "blk_log": list(grp.log),
                        "lse_exact": (not return_lse) or torch.equal(one_l, b_l),
                    })
        # weightless lockstep: [H, 0, 0] on the all_reduce, guard per block
        wl = []
        counts = [6] + [0] * (world - 1)
        steps = []
        with mock.patch.object(c, "weightless_kv_active", return_value=True), \
             mock.patch.object(c, "guard_dcp_step",
                               side_effect=lambda name, g: steps.append(("guard", name))):
            c._LSE_MERGE["mode"] = "ar"
            c._LSE_MERGE["dtype"] = "fp32"
            o, lse = _inputs(rank, 11, 6, 8, torch.float32)
            grp = _GlooGroup(rank, world)
            is_head = rank == 0
            one = c.cp_lse_ag_out_ar_mha_uneven(o, lse, grp, counts, return_lse=is_head)
            steps.clear()
            grp.log.clear()
            got = c.cp_lse_merge_token_blocks(
                c.cp_lse_ag_out_ar_mha_uneven, o, lse, grp, counts,
                return_lse=is_head, block_tokens=4,
            )
            seq = [s for s in steps] + [("coll",) + e for e in grp.log]
            one_o = one[0] if is_head else one
            got_o = got[0] if is_head else got
            wl.append({"steps": steps, "coll": list(grp.log), "seq_len": len(seq),
                       "close": bool(torch.allclose(one_o, got_o, rtol=1e-6, atol=1e-6)),
                       "shape": tuple(got_o.shape)})
        q.put((rank, res, wl, None))
    except Exception as e:  # noqa: BLE001
        import traceback

        q.put((rank, None, None, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def _run_world(world, cases):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, cases, q)) for r in range(world)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(world):
        rank, res, wl, err = q.get(timeout=240)
        out[rank] = (res, wl, err)
    for p in procs:
        p.join(timeout=60)
    return out


class TestGlooBlockedMerge(CustomTestCase):
    """Real collectives, 3 and 2 ranks."""

    @classmethod
    def setUpClass(cls):
        cls.w3 = _run_world(3, _CASES)
        cls.w2 = _run_world(2, [(m, w, [4, 2], T, b) for (m, w, _c, T, b) in _CASES if _c == [4, 2, 2]])

    def _check(self, out, world):
        for r, (res, wl, err) in out.items():
            self.assertIsNone(err, f"rank {r}:\n{err}")
        ranks = sorted(out)
        n = len(out[0][0])
        for i in range(n):
            rows = [out[r][0][i] for r in ranks]
            for row in rows:
                ctx = (world, row["case"], row["mode"], row["return_lse"], row["odt"])
                # blockwise == one-shot, same shape/dtype, lse bit-exact
                self.assertEqual(row["shape"], row["one_shape"], ctx)
                self.assertEqual(row["dtype"], "torch.float32", ctx)
                self.assertTrue(row["lse_exact"], ctx)
                if row["mode"] == "a2a":
                    # the local sum over ranks runs in rank order whatever
                    # the width: bit-exact
                    self.assertTrue(row["exact"], ctx)
                else:
                    # the all_reduce's per-element order is the transport's
                    # (gloo's ring places an element by its offset in the
                    # message): fp reassociation of a W-term sum at most --
                    # fp32 rounding on an fp32 wire, bf16 rounding on a bf16 one
                    self.assertTrue(row["close"], ctx + (row["maxdiff"],))
                # two collectives per block, LSE all-gather then head exchange
                kinds = [k for k, _ in row["blk_log"]]
                self.assertEqual(len(kinds), 2 * row["n_blocks"], ctx)
                second = "all_to_all" if row["mode"] == "a2a" else "all_reduce"
                self.assertEqual(kinds, ["all_gather", second] * row["n_blocks"], ctx)
                if row["n_blocks"] == 1:
                    # one block IS today's call: identical collective record
                    self.assertEqual(row["blk_log"], row["one_log"], ctx)
            # the SAME sequence (families and shapes) on every rank
            seqs = [[k for k, _ in row["blk_log"]] for row in rows]
            self.assertTrue(all(s == seqs[0] for s in seqs), (world, i, seqs))
            nb = {row["n_blocks"] for row in rows}
            self.assertEqual(len(nb), 1, (world, i, nb))

    def test_three_ranks(self):
        self._check(self.w3, 3)
        rows = [r for r in self.w3[0][0]]
        census = {}
        for r in rows:
            k = (r["mode"], r["wire"])
            e, n, md = census.get(k, (0, 0, 0.0))
            census[k] = (e + int(r["exact"]), n + 1, max(md, r["maxdiff"]))
        print("\n[census rank0: (mode, wire) -> exact/total, max |diff|]", census)

    def test_two_ranks(self):
        self._check(self.w2, 2)

    def test_weightless_head_and_workers_in_lockstep(self):
        for out, world in ((self.w3, 3), (self.w2, 2)):
            wl = {r: out[r][1][0] for r in out}
            steps = [wl[r]["steps"] for r in sorted(wl)]
            colls = [[k for k, _ in wl[r]["coll"]] for r in sorted(wl)]
            # 11 rows at width 4 -> 3 blocks: one guard step + two collectives each
            self.assertEqual(steps[0], [("guard", "lse_merge")] * 3)
            self.assertTrue(all(s == steps[0] for s in steps), steps)
            self.assertEqual(colls[0], ["all_gather", "all_reduce"] * 3)
            self.assertTrue(all(c == colls[0] for c in colls), colls)
            self.assertTrue(all(wl[r]["close"] for r in wl), wl)
            self.assertEqual(wl[0]["shape"], (11, 6, 8))
            for r in range(1, world):
                self.assertEqual(wl[r]["shape"], (11, 0, 8))


class _FakeGroup:
    """Single-process stand-in: shape-correct collectives, no peers."""

    def __init__(self, world, rank=0):
        self.world_size = world
        self.rank_in_group = rank

    def all_gather(self, t, dim=0):
        return torch.cat([t] * self.world_size, dim=0)

    def all_reduce(self, t):
        return t.clone()  # out-of-place, like barlink

    def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
        output.zero_()
        return output


class TestOneBlockIsTodaysCall(CustomTestCase):
    def test_result_object_and_arguments_unchanged(self):
        sentinel = object()
        fn = mock.Mock(return_value=sentinel)
        grp = _FakeGroup(3)
        for T, b in ((0, 5), (1, 5), (5, 5), (7, 0), (9, 9)):
            o = torch.zeros(T, 4, 2)
            lse = torch.zeros(T, 4)
            fn.reset_mock()
            got = comm.cp_lse_merge_token_blocks(fn, o, lse, grp, [2, 1, 1],
                                                 return_lse=True, block_tokens=b)
            self.assertIs(got, sentinel)
            fn.assert_called_once()
            args, kw = fn.call_args
            self.assertIs(args[0], o)
            self.assertIs(args[1], lse)
            self.assertEqual(kw, {"return_lse": True})

    def test_single_rank_never_blocks(self):
        fn = mock.Mock(return_value="X")
        got = comm.cp_lse_merge_token_blocks(
            fn, torch.zeros(20, 2, 2), torch.zeros(20, 2), _FakeGroup(1), [2],
            block_tokens=3)
        self.assertEqual(got, "X")
        fn.assert_called_once()

    def test_spans(self):
        S = comm.lse_merge_token_spans
        self.assertEqual(S(0, 5), [(0, 0)])
        self.assertEqual(S(1, 5), [(0, 1)])
        self.assertEqual(S(5, 5), [(0, 5)])
        self.assertEqual(S(6, 5), [(0, 3), (3, 6)])
        self.assertEqual(S(4096, 0), [(0, 4096)])
        for T, b in ((6, 5), (4096, 404), (4096, 667), (8370, 404), (13, 5), (12288, 404)):
            sp = S(T, b)
            self.assertEqual(sp[0][0], 0)
            self.assertEqual(sp[-1][1], T)
            self.assertTrue(all(a[1] == c[0] for a, c in zip(sp, sp[1:])))
            self.assertTrue(all(0 < e - s <= b for s, e in sp), (T, b, sp))
            self.assertEqual(len(sp), -(-T // b))
            sizes = [e - s for s, e in sp]
            self.assertLessEqual(max(sizes) - min(sizes), 1)


class TestDerivedWidth(CustomTestCase):
    def setUp(self):
        _reset_merge_cache()

    def tearDown(self):
        _reset_merge_cache()

    def test_27b_tp0_geometry(self):
        with mock.patch.dict(os.environ, {"SGLANG_DCP_LSE_MERGE_DTYPE": "fp32"}):
            _reset_merge_cache()
            a2a = comm.lse_merge_block_tokens(4096, [12, 6, 6], 256, 2, "a2a")
            ar = comm.lse_merge_block_tokens(4096, [12, 6, 6], 256, 2, "ar")
            wl = comm.lse_merge_block_tokens(4096, [24, 0, 0], 256, 2, "ar")
        self.assertEqual(a2a, 404)
        self.assertEqual(ar, 667)
        self.assertEqual(wl, 573)
        # the crash forward: 4096 rows -> 11 blocks of <= 373 rows
        self.assertEqual(len(comm.lse_merge_token_spans(4096, a2a)), 11)
        # per block <= one [4096, 24, 256] bf16 partial; one-shot ~10x that
        per_tok = comm.lse_merge_bytes_per_token("a2a", 4, 24, 12, 3, 256, 2)
        ref = 4096 * 24 * 256 * 2
        self.assertLessEqual(per_tok * a2a, ref)
        self.assertGreater(per_tok * (a2a + 1), ref)
        # the one-shot at the crash width: ~487 MiB against the 48 MiB bound
        self.assertAlmostEqual(per_tok * 4096 / 2**20, 486.56, delta=0.01)

    def test_rank_uniform(self):
        # no rank input at all; a permutation of the shard vector changes nothing
        v = {comm.lse_merge_block_tokens(4096, c, 256, 2, "a2a")
             for c in ([12, 6, 6], [6, 12, 6], [6, 6, 12])}
        self.assertEqual(len(v), 1)

    def test_off_cases(self):
        self.assertEqual(comm.lse_merge_block_tokens(4096, [24], 256, 2, "a2a"), 0)
        self.assertEqual(comm.lse_merge_block_tokens(0, [12, 6, 6], 256, 2, "a2a"), 0)

    def test_bytes_model_bounds_the_real_allocations(self):
        """Sum every new storage the ONE-SHOT body allocates (TorchDispatchMode)
        and demand it stays under the model's per-token figure x T."""
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Count(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.bytes = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                ins = set()

                def _collect(x):
                    if isinstance(x, torch.Tensor):
                        ins.add(x.untyped_storage().data_ptr())
                    elif isinstance(x, (list, tuple)):
                        for y in x:
                            _collect(y)

                _collect(args)
                _collect(list(kwargs.values()))
                out = func(*args, **kwargs)
                outs = out if isinstance(out, (list, tuple)) else [out]
                for t in outs:
                    if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in ins:
                        self.bytes += t.untyped_storage().nbytes()
                return out

        T, D = 64, 32
        for counts in ([12, 6, 6], [8, 8, 8], [24, 0, 0]):
            H, W, m = sum(counts), len(counts), max(counts)
            for mode in ("a2a", "ar"):
                for wire in ("fp32", "bf16"):
                    for odt in (torch.bfloat16, torch.float32):
                        comm._LSE_MERGE["dtype"] = wire
                        fn = (comm.cp_lse_ag_out_a2a_mha_uneven if mode == "a2a"
                              else comm.cp_lse_ag_out_ar_mha_uneven)
                        o = torch.randn(T, H, D).to(odt)
                        lse = torch.randn(T, H)
                        rank = counts.index(m)
                        with _Count() as cnt:
                            fn(o, lse, _FakeGroup(W, rank), counts, return_lse=True)
                        model = comm.lse_merge_bytes_per_token(
                            mode, 2 if wire == "bf16" else 4, H, m, W, D, o.element_size())
                        self.assertLessEqual(
                            cnt.bytes, model * T,
                            (counts, mode, wire, odt, cnt.bytes, model * T))
                        # and it is not a loose bound: within 10 % of the real sum
                        # (unless this rank owns every head -- the weightless
                        # head -- where the head slice is the whole tensor and
                        # .contiguous() copies nothing)
                        if m < H:
                            self.assertGreaterEqual(
                                cnt.bytes, int(0.9 * model * T),
                                (counts, mode, wire, odt, cnt.bytes, model * T))


class TestFinalMergeBlocks(CustomTestCase):
    def test_blocked_final_merge_is_byte_identical(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        class _L:
            tp_q_head_num, head_dim = 3, 8

        g = torch.Generator().manual_seed(7)
        for T in (0, 1, 5, 6, 23):
            o_cur = torch.randn(T, 3, 8, generator=g).to(torch.bfloat16)
            lse_cur = torch.randn(T, 3, generator=g)
            o_pre = torch.randn(T, 3, 8, generator=g)
            lse_pre = torch.randn(T, 3, generator=g)
            if T >= 2:
                lse_pre[1] = float("-inf")
                o_pre[1] = 0
            q = torch.zeros(T, 24, dtype=torch.bfloat16)
            one = mock.Mock(_dcp_merge_block_tokens=0)
            blk = mock.Mock(_dcp_merge_block_tokens=5)
            a = fb.FlashInferAttnBackend._dcp_extend_final_merge(
                one, q, _L, o_cur, lse_cur, o_pre, lse_pre)
            b = fb.FlashInferAttnBackend._dcp_extend_final_merge(
                blk, q, _L, o_cur, lse_cur, o_pre, lse_pre)
            self.assertEqual(a.shape, (T, 24))
            self.assertEqual(a.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(a, b), T)

    def test_one_block_is_the_old_expression(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        g = torch.Generator().manual_seed(3)
        o_cur = torch.randn(9, 3, 8, generator=g).to(torch.bfloat16)
        lse_cur = torch.randn(9, 3, generator=g)
        o_pre = torch.randn(9, 3, 8, generator=g)
        lse_pre = torch.randn(9, 3, generator=g)
        final_lse = torch.logaddexp(lse_cur.float(), lse_pre.float())
        sc_cur = torch.nan_to_num(torch.exp(lse_cur - final_lse), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
        sc_pre = torch.nan_to_num(torch.exp(lse_pre - final_lse), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
        old = (o_cur.float() * sc_cur + o_pre.float() * sc_pre).to(torch.bfloat16).contiguous().view(-1, 24)

        class _L:
            tp_q_head_num, head_dim = 3, 8

        got = fb.FlashInferAttnBackend._dcp_extend_final_merge(
            mock.Mock(_dcp_merge_block_tokens=0), torch.zeros(9, 24, dtype=torch.bfloat16),
            _L, o_cur, lse_cur, o_pre, lse_pre)
        self.assertTrue(torch.equal(got, old))


class TestDispatcherAndSites(CustomTestCase):
    def setUp(self):
        _reset_merge_cache()

    def tearDown(self):
        _reset_merge_cache()

    def test_dispatcher_blocks_above_the_width_only(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        grp = _FakeGroup(3)
        for mode, T, want_blocks in (("a2a", 12, 3), ("a2a", 5, 1), ("ar", 12, 3), ("ar", 4, 1)):
            calls = []

            def _fake(o, lse, g, counts, return_lse=False, _c=calls):
                _c.append(o.shape[0])
                return (torch.zeros(o.shape[0], 2, 4), torch.zeros(o.shape[0], 2)) if return_lse \
                    else torch.zeros(o.shape[0], 2, 4)

            with mock.patch.object(comm, "lse_merge_mode", return_value=mode), \
                 mock.patch.object(comm, "weightless_kv_active", return_value=False), \
                 mock.patch.object(comm, "cp_lse_ag_out_a2a_mha_uneven", side_effect=_fake), \
                 mock.patch.object(fb, "cp_lse_ag_out_ar_mha_uneven", side_effect=_fake):
                o = torch.zeros(T, 4, 4)
                out, l = fb._dcp_uneven_merge(o, torch.zeros(T, 4), grp, [2, 1, 1],
                                              return_lse=True, block_tokens=5)
            self.assertEqual(len(calls), want_blocks, (mode, T, calls))
            self.assertEqual(sum(calls), T)
            self.assertEqual(tuple(out.shape), (T, 2, 4))
            self.assertEqual(tuple(l.shape), (T, 2))

    def test_weightless_derivation_uses_the_all_reduce(self):
        with mock.patch.object(comm, "lse_merge_mode", return_value="a2a"), \
             mock.patch.object(comm, "weightless_kv_active", return_value=True):
            self.assertEqual(comm.lse_merge_effective_mode(), "ar")
        with mock.patch.object(comm, "lse_merge_mode", return_value="a2a"), \
             mock.patch.object(comm, "weightless_kv_active", return_value=False):
            self.assertEqual(comm.lse_merge_effective_mode(), "a2a")

    def test_every_site_carries_the_width(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        cls = fb.FlashInferAttnBackend
        for name in ("_forward_decode_dcp", "_forward_extend_dcp"):
            body = inspect.getsource(getattr(cls, name))
            n_sites = body.count("_dcp_uneven_merge(")
            self.assertGreaterEqual(n_sites, 1, name)
            self.assertEqual(
                body.count("block_tokens=self._dcp_merge_block_tokens"), n_sites, name)
        for name in ("forward_decode_weightless_worker", "forward_extend_weightless_worker"):
            body = inspect.getsource(getattr(cls, name))
            self.assertIn("lse_merge_is_blocked(", body)
            self.assertIn("cp_lse_merge_token_blocks(", body)
            self.assertIn("block_tokens=self._dcp_merge_block_tokens", body)
        init = inspect.getsource(cls.__init__)
        self.assertIn("self._dcp_merge_block_tokens = _resolve_dcp_merge_block_tokens(", init)
        self.assertIn("self._dcp_merge_block_tokens = 0", init)

    def _runner(self, asked, dtype=torch.bfloat16, chunk=4096):
        sa = mock.Mock(chunked_prefill_size=chunk, max_prefill_tokens=16384,
                       dcp_lse_merge_block_tokens=asked)
        return mock.Mock(server_args=sa, model_config=mock.Mock(head_dim=256), dtype=dtype)

    def test_resolver(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        with mock.patch.object(comm, "lse_merge_mode", return_value="a2a"), \
             mock.patch.object(comm, "weightless_kv_active", return_value=False), \
             mock.patch.object(comm, "lse_merge_reduce_dtype", return_value="fp32"):
            with self.assertLogs(fb.logger, level="INFO") as cm:
                self.assertEqual(fb._resolve_dcp_merge_block_tokens(self._runner(None), [12, 6, 6]), 404)
            self.assertTrue(any("DCP-MERGE-BLOCK block_tokens=404 source=derived mode=a2a" in m
                                for m in cm.output), cm.output)
            self.assertEqual(fb._resolve_dcp_merge_block_tokens(self._runner(0), [12, 6, 6]), 0)
            self.assertEqual(fb._resolve_dcp_merge_block_tokens(self._runner(700), [12, 6, 6]), 700)
            with self.assertRaises(ValueError):
                fb._resolve_dcp_merge_block_tokens(self._runner(-1), [12, 6, 6])
            # no chunking configured: the budget falls back to max_prefill_tokens
            self.assertEqual(
                fb._resolve_dcp_merge_block_tokens(self._runner(None, chunk=-1), [12, 6, 6]),
                comm.lse_merge_block_tokens(16384, [12, 6, 6], 256, 2, "a2a"))

    def test_server_arg_parses(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        a = p.parse_args(["--model-path", "m"])
        self.assertIsNone(a.dcp_lse_merge_block_tokens)
        a = p.parse_args(["--model-path", "m", "--dcp-lse-merge-block-tokens", "0"])
        self.assertEqual(a.dcp_lse_merge_block_tokens, 0)
        a = p.parse_args(["--model-path", "m", "--dcp-lse-merge-block-tokens", "512"])
        self.assertEqual(a.dcp_lse_merge_block_tokens, 512)


if __name__ == "__main__":
    unittest.main()
