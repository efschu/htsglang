# SPDX-License-Identifier: Apache-2.0
"""Uneven-DCP collective fusion for the 27B D verify round (26.09.).

Two switches, both default off (code path unchanged):

* ``SGLANG_DCP_FUSE_KVQ_GATHER=1`` -- the KV write gather (A) and the q-head
  gather (B) of one full-attention layer become ONE all-gather
  (``comm.cp_all_gather_kvq_heads_uneven``). Pure data movement, so the
  gathered k/v/q must be BIT-identical to the two gathers.
* ``SGLANG_DCP_LSE_MERGE_FUSED=1`` (with ``SGLANG_DCP_LSE_MERGE=a2a``) -- the
  LSE all-gather (C) rides inside the head all_to_all (D); the receiver does
  the logsumexp/scale/sum. Same arithmetic on the same values, so on one
  device class it must be bit-identical to the two-collective a2a body.

Pinned on CPU with a threaded 3-rank group whose collectives really exchange
the ranks' inputs (a rendezvous per call), so a rank that issued a different
collective sequence would deadlock the test (bounded by a barrier timeout)
instead of passing.
"""

import inspect
import os
import threading
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.layers.dcp import comm  # noqa: E402


class _Rendezvous:
    """Shared state of one simulated world: every collective deposits the
    caller's input, waits for all ranks, then reads the peers' inputs."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=20)
        self.slots = [None] * world
        self.log = [[] for _ in range(world)]

    def exchange(self, rank, op, payload):
        self.log[rank].append(op)
        self.barrier.wait()
        self.slots[rank] = (op, payload)
        self.barrier.wait()
        got = list(self.slots)
        self.barrier.wait()
        ops = {o for o, _ in got}
        assert len(ops) == 1, f"ranks issued different collectives: {got!r:.200}"
        return [p for _, p in got]


class _Group:
    def __init__(self, rv: _Rendezvous, rank: int):
        self.rv = rv
        self.rank_in_group = rank
        self.world_size = rv.world

    def all_gather(self, t, dim=0):
        parts = self.rv.exchange(self.rank_in_group, ("all_gather", dim, tuple(t.shape)), t.clone())
        return torch.cat(parts, dim=dim)

    def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
        r = self.rank_in_group
        blocks = list(torch.split(input, list(input_split_sizes), dim=0))
        peers = self.rv.exchange(r, ("a2a",), [b.clone() for b in blocks])
        output.copy_(torch.cat([peers[s][r] for s in range(self.world_size)], dim=0))
        return output

    def all_reduce(self, t):
        parts = self.rv.exchange(self.rank_in_group, ("all_reduce", tuple(t.shape)), t.clone())
        acc = parts[0].clone()
        for p in parts[1:]:
            acc = acc + p
        return acc


def _run_world(world, fn):
    """Run fn(rank, group) on `world` threads; return the per-rank results
    and the per-rank collective logs."""
    rv = _Rendezvous(world)
    out = [None] * world
    err = []

    def body(r):
        try:
            out[r] = fn(r, _Group(rv, r))
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            rv.barrier.abort()

    ts = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        raise err[0]
    return out, rv.log


def _reset():
    comm._LSE_MERGE["mode"] = None
    comm._LSE_MERGE["dtype"] = None
    comm._LSE_MERGE["fused"] = None
    comm._KVQ_FUSE["on"] = None
    comm._KVQ_FUSE["max_rows"] = None


def _bitwise(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    it = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(it), b.contiguous().view(it))


class TestFusedKvqGather(CustomTestCase):
    def _case(self, kv_counts, q_counts, T=8, D=16, dtype=torch.bfloat16, seed=0):
        g = torch.Generator().manual_seed(seed)
        W = len(kv_counts)
        k = [torch.randn(T, kv_counts[r], D, generator=g).to(dtype) for r in range(W)]
        v = [torch.randn(T, kv_counts[r], D, generator=g).to(dtype) for r in range(W)]
        q = [torch.randn(T, q_counts[r], D, generator=g).to(dtype) for r in range(W)]

        def separate(r, grp):
            kv_full = comm.cp_all_gather_heads_uneven(torch.cat((k[r], v[r]), dim=0), grp, kv_counts)
            q_full = comm.cp_all_gather_heads_uneven(q[r], grp, q_counts)
            return kv_full[:T], kv_full[T:], q_full

        def fused(r, grp):
            return comm.cp_all_gather_kvq_heads_uneven(k[r], v[r], q[r], grp, kv_counts, q_counts)

        want, log_sep = _run_world(W, separate)
        got, log_fused = _run_world(W, fused)
        for r in range(W):
            for a, b in zip(got[r], want[r]):
                self.assertTrue(_bitwise(a, b), f"rank {r} differs")
                self.assertTrue(a.is_contiguous())
            # and against the obvious reference
            self.assertTrue(_bitwise(got[r][0], torch.cat(k, dim=1)))
            self.assertTrue(_bitwise(got[r][1], torch.cat(v, dim=1)))
            self.assertTrue(_bitwise(got[r][2], torch.cat(q, dim=1)))
            self.assertEqual(len(log_sep[r]), 2)
            self.assertEqual(len(log_fused[r]), 1, "fusion must issue ONE collective")

    def test_27b_geometry_bf16(self):
        # 27B full attention: kv 4 heads -> [2,1,1], q 24 heads -> [12,6,6]
        self._case([2, 1, 1], [12, 6, 6])

    def test_single_row_decode(self):
        self._case([2, 1, 1], [12, 6, 6], T=1)

    def test_odd_rows_and_fp16(self):
        self._case([2, 1, 1], [12, 6, 6], T=5, dtype=torch.float16, seed=3)

    def test_even_split(self):
        self._case([1, 1, 1], [8, 8, 8], T=4, seed=7)

    def test_rejects_mismatched_dtype(self):
        rv = _Rendezvous(1)
        with self.assertRaises(AssertionError):
            comm.cp_all_gather_kvq_heads_uneven(
                torch.zeros(2, 1, 4, dtype=torch.bfloat16),
                torch.zeros(2, 1, 4, dtype=torch.bfloat16),
                torch.zeros(2, 3, 4, dtype=torch.float32),
                _Group(rv, 0), [1], [3],
            )


class _MergeWorld:
    def __init__(self, counts, T=8, D=16, seed=0, neg_inf_rows=True, dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        self.counts = list(counts)
        self.W = len(counts)
        H = sum(counts)
        self.o = [torch.randn(T, H, D, generator=g).to(dtype) for _ in range(self.W)]
        self.lse = [torch.randn(T, H, generator=g) * 3 for _ in range(self.W)]
        if neg_inf_rows and T > 1:
            # a query with no rows on one rank (legal -inf), and one with no
            # rows anywhere (all -inf -> merged 0, lse -inf)
            self.lse[1][0, :] = float("-inf")
            for r in range(self.W):
                self.lse[r][T - 1, 0] = float("-inf")


class TestFusedLseMerge(CustomTestCase):
    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def _both(self, w, return_lse, env_extra=None):
        env = {"SGLANG_DCP_LSE_MERGE": "a2a"}
        env.update(env_extra or {})

        def call(r, grp):
            return comm.cp_lse_ag_out_a2a_mha_uneven(
                w.o[r], w.lse[r], grp, w.counts, return_lse=return_lse)

        with mock.patch.dict(os.environ, env):
            os.environ.pop("SGLANG_DCP_LSE_MERGE_FUSED", None)
            _reset()
            base, log_base = _run_world(w.W, call)
            os.environ["SGLANG_DCP_LSE_MERGE_FUSED"] = "1"
            _reset()
            fused, log_fused = _run_world(w.W, call)
        return base, fused, log_base, log_fused

    def _check(self, counts, T, return_lse, dtype=torch.float32, seed=0):
        w = _MergeWorld(counts, T=T, seed=seed, dtype=dtype)
        base, fused, log_base, log_fused = self._both(w, return_lse)
        for r in range(w.W):
            b = base[r] if return_lse else (base[r],)
            f = fused[r] if return_lse else (fused[r],)
            for x, y in zip(f, b):
                self.assertEqual(x.dtype, y.dtype)
                self.assertEqual(tuple(x.shape), tuple(y.shape))
                # bitwise, including -inf LSE positions
                self.assertTrue(_bitwise(x, y), f"rank {r}: fused merge not bit-identical")
            self.assertEqual([o[0] for o in log_base[r]], ["all_gather", "a2a"])
            self.assertEqual([o[0] for o in log_fused[r]], ["a2a"])

    def test_27b_heads_verify_rows(self):
        self._check([12, 6, 6], T=8, return_lse=True)
        self._check([12, 6, 6], T=8, return_lse=False)

    def test_single_row_and_odd_rows(self):
        self._check([12, 6, 6], T=1, return_lse=True)
        self._check([12, 6, 6], T=5, return_lse=True, seed=4)

    def test_bf16_partials(self):
        # the attention partials arrive in the query dtype; the wire is fp32
        self._check([12, 6, 6], T=8, return_lse=True, dtype=torch.bfloat16, seed=2)

    def test_even_heads(self):
        self._check([8, 8, 8], T=8, return_lse=True, seed=9)

    def test_rows_stay_16_byte_aligned(self):
        self.assertEqual((16 * 4 + comm.LSE_FUSED_TAIL * 4) % 16, 0)
        self.assertEqual((256 * 4 + comm.LSE_FUSED_TAIL * 4) % 16, 0)

    def test_bf16_wire_keeps_two_collectives(self):
        w = _MergeWorld([12, 6, 6], T=8, seed=1)
        _b, _f, _lb, log_fused = self._both(
            w, True, env_extra={"SGLANG_DCP_LSE_MERGE_DTYPE": "bf16"})
        for r in range(w.W):
            self.assertEqual([o[0] for o in log_fused[r]], ["all_gather", "a2a"])

    def test_wide_forward_keeps_two_collectives(self):
        w = _MergeWorld([12, 6, 6], T=8, seed=8)
        base, fused, _lb, log_fused = self._both(
            w, True, env_extra={"SGLANG_DCP_FUSE_MAX_ROWS": "4"})
        for r in range(w.W):
            self.assertEqual([o[0] for o in log_fused[r]], ["all_gather", "a2a"])
            for x, y in zip(fused[r], base[r]):
                self.assertTrue(_bitwise(x, y))

    def test_token_blocks_compose(self):
        w = _MergeWorld([12, 6, 6], T=9, seed=6)

        def call(r, grp):
            return comm.cp_lse_merge_token_blocks(
                comm.cp_lse_ag_out_a2a_mha_uneven, w.o[r], w.lse[r], grp, w.counts,
                return_lse=True, block_tokens=4)

        with mock.patch.dict(os.environ, {"SGLANG_DCP_LSE_MERGE": "a2a"}):
            os.environ.pop("SGLANG_DCP_LSE_MERGE_FUSED", None)
            _reset()
            base, _ = _run_world(3, call)
            os.environ["SGLANG_DCP_LSE_MERGE_FUSED"] = "1"
            _reset()
            fused, logs = _run_world(3, call)
        for r in range(3):
            for x, y in zip(fused[r], base[r]):
                self.assertTrue(_bitwise(x, y))
            self.assertEqual([o[0] for o in logs[r]], ["a2a"] * 3)


class TestDefaultsOff(CustomTestCase):
    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def test_env_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("SGLANG_DCP_FUSE_KVQ_GATHER", "SGLANG_DCP_LSE_MERGE_FUSED"):
                os.environ.pop(k, None)
            _reset()
            self.assertFalse(comm.dcp_fuse_kvq_gather())
            self.assertFalse(comm.lse_merge_fused())

    def test_clock_a2a_default_off(self):
        from sglang.srt.distributed import parallel_state as ps

        if "SGLANG_COLLECTIVE_CLOCK_A2A" not in os.environ:
            self.assertFalse(ps._CLOCK_A2A)
        src = inspect.getsource(ps.GroupCoordinator.all_to_all_single_v)
        self.assertIn("_CLOCK_A2A and _COLLECTIVE_CLOCK.armed", src)


class _Layer:
    tp_k_head_num = 1
    tp_v_head_num = 1
    tp_q_head_num = 6
    head_dim = 16


class TestBackendGate(CustomTestCase):
    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def _gate(self, env="1", **attrs):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        stub = type("S", (), {})()
        stub.uneven_dcp = attrs.get("uneven_dcp", True)
        stub.dcp_kv_replicated_heads = attrs.get("replicated", False)
        stub.weightless_kv = attrs.get("weightless", False)
        k = torch.zeros(8, 1, 16, dtype=attrs.get("kdtype", torch.bfloat16))
        q = torch.zeros(8, 6, 16, dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"SGLANG_DCP_FUSE_KVQ_GATHER": env}):
            _reset()
            return fb.FlashInferAttnBackend._dcp_kvq_fusable(stub, _Layer(), k, k, q)

    def test_gate(self):
        self.assertTrue(self._gate())
        self.assertFalse(self._gate(env="0"))
        self.assertFalse(self._gate(uneven_dcp=False))
        self.assertFalse(self._gate(replicated=True))
        self.assertFalse(self._gate(weightless=True))
        self.assertFalse(self._gate(kdtype=torch.float16))
        with mock.patch.dict(os.environ, {"SGLANG_DCP_FUSE_MAX_ROWS": "4"}):
            comm._KVQ_FUSE["max_rows"] = None
            self.assertFalse(self._gate())  # 8 rows > 4
        comm._KVQ_FUSE["max_rows"] = None

    def test_wiring(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        dec = inspect.getsource(fb.FlashInferAttnBackend._forward_decode_dcp)
        self.assertIn("self._dcp_kvq_fusable(layer, k, v, q_local)", dec)
        self.assertIn("self._dcp_write_gather_with_q(", dec)
        ext = inspect.getsource(fb.FlashInferAttnBackend._forward_extend_dcp)
        seq, overlapped = ext.split("OVERLAPPED SCHEDULING", 1)
        # fused only in the sequential (mode 0) schedule, and only with a prefix
        self.assertIn("elif has_prefix and self._dcp_kvq_fusable(layer, k, v, q_local):", seq)
        self.assertNotIn("_dcp_write_gather_with_q", overlapped)
        # the weightless workers keep the unfused A,B sequence
        for name in ("forward_decode_weightless_worker", "forward_extend_weightless_worker"):
            body = inspect.getsource(getattr(fb.FlashInferAttnBackend, name))
            self.assertNotIn("_dcp_write_gather_with_q", body)


if __name__ == "__main__":
    unittest.main()
