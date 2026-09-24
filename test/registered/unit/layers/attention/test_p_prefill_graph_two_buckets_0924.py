"""P prefill graph, TWO buckets in sequence (27B line, xsn434, --p-prefill-graph-
tiny 16): the GDN kernels of each captured bucket run on that bucket's OWN
static metadata, and every replay is bit-identical to the eager computation.

The capture order is the runner's: largest first (512), then the tiny bucket
(16). The static ``query_start_loc`` of each bucket is taken from the REAL
metadata builder (``MambaAttnBackendBase._extend_graph_metadata`` via
``init_forward_metadata_out_graph``, the bucket named as the runner names it),
and FLA's chunk tables come from the REAL pin (fla/index.py), so this proves
the fixed wiring end to end on the production kernels under the Triton
interpreter:

* each bucket's pinned chunk table is its own bucket's (512 -> 8 chunks,
  16 -> 1 chunk) -- red with the per-slot key, where the 16-token bucket read
  the 512 bucket's table;
* tiny-bucket replays (16-row tensors, garbage past the live length) and
  big-bucket replays AFTER the tiny capture are bit-identical to eager, state
  included, and no other state slot moves;
* the control the old design is judged by: the 16-row tensors run on the 512
  table (the per-slot sharing) are ALSO bit-identical -- the FLA kernels bound
  themselves by the live length, so the shared table was not an out-of-bounds
  access by itself.

Subprocess + shims: see test_p_prefill_graph_gdn_baked_grid_0924.py.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

BIG, TINY = 512, 16

_WORKER = textwrap.dedent(
    """
    import contextlib, json, os, sys, types
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    import triton
    import triton.language as tl
    import triton.runtime.autotuner as _at

    from sglang.srt.layers.attention.fla import index as fidx
    from sglang.srt.layers.attention.fla.chunk import ChunkGatedDeltaRuleFunction
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64 as KERNEL,
    )
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    import sglang.srt.layers.attention.fla.utils as fu

    _at.Autotuner._bench = lambda self, *a, config=None, **m: [0.0, 0.0, 0.0]
    fu.custom_device_ctx = lambda index: contextlib.nullcontext()

    @triton.jit
    def _exp(x):
        return tl.exp(x)

    @triton.jit
    def _exp2(x):
        return tl.math.exp2(x)

    @triton.jit
    def _log(x):
        return tl.log(x)

    @triton.jit
    def _log2(x):
        return tl.log2(x)

    _repl = {"exp": _exp, "exp2": _exp2, "log": _log, "log2": _log2}
    _orig = {tl.exp, tl.math.exp2, tl.log, tl.log2}
    for _name, _mod in list(sys.modules.items()):
        if _mod is None or not _name.startswith("sglang.srt.layers.attention.fla"):
            continue
        for _alias, _fn in _repl.items():
            if getattr(_mod, _alias, None) in _orig:
                setattr(_mod, _alias, _fn)

    H, K, V, SLOTS = 2, 64, 64, 3
    BIG, TINY = __BIG__, __TINY__

    class _Pool:
        def get_mamba_indices(self, rpi):
            return torch.full_like(rpi, 1, dtype=torch.int32)

        def translate_mamba_indices(self, x):
            return x

    be = object.__new__(MambaAttnBackendBase)
    be.device = torch.device("cpu")
    be.pad_slot_id = -1
    be.replayssm_write_pos_list = None
    be._extend_graph_static = {}
    be.req_to_token_pool = _Pool()

    def meta(n, bucket, capture):
        view = types.SimpleNamespace(
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            extend_start_loc=torch.tensor([0], dtype=torch.int64),
            extend_seq_lens=torch.tensor([n], dtype=torch.int64),
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([n], dtype=torch.int64),
            mamba_track_mask=None,
            spec_info=None,
            prefill_graph_bucket=bucket,
        )
        be.init_forward_metadata_out_graph(view, in_capture=capture)
        return be.forward_metadata.query_start_loc, be.forward_metadata.mamba_cache_indices

    def inputs(rows, n, seed, garbage_seed):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(1, rows, H, K, generator=g)
        k = torch.randn(1, rows, H, K, generator=g)
        v = torch.randn(1, rows, H, V, generator=g)
        gg = torch.nn.functional.logsigmoid(torch.randn(1, rows, H, generator=g))
        beta = torch.rand(1, rows, H, generator=g).sigmoid()
        gb = torch.Generator().manual_seed(garbage_seed)
        for t in (q, k, v):
            t[:, n:] = 50.0 * torch.randn(t[:, n:].shape, generator=gb)
        gg[:, n:] = -30.0 * torch.rand(gg[:, n:].shape, generator=gb)
        beta[:, n:] = torch.rand(beta[:, n:].shape, generator=gb)
        return q, k, v, gg, beta

    def s0():
        g = torch.Generator().manual_seed(7)
        return torch.randn(SLOTS, H, V, K, generator=g)

    def run(q, k, v, gg, beta, state, cu, idx):
        o, _ = ChunkGatedDeltaRuleFunction.apply(
            q, k, v, gg, beta, K ** -0.5, state, idx, cu, True,
        )
        return o

    def eager(q, k, v, gg, beta, n):
        st = s0()
        o = run(q[:, :n].contiguous(), k[:, :n].contiguous(), v[:, :n].contiguous(),
                gg[:, :n].contiguous(), beta[:, :n].contiguous(), st,
                torch.tensor([0, n], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32))
        return o, st

    def replay_case(rows, n, seed, garbage, bucket):
        q, k, v, gg, beta = inputs(rows, n, seed, garbage)
        o_ref, st_ref = eager(q, k, v, gg, beta, n)
        cu, idx = meta(n, bucket, capture=False)
        st = s0()
        o = run(q, k, v, gg, beta, st, cu, idx)
        return {
            "bucket": bucket, "n": n,
            "o_equal": bool(torch.equal(o[:, :n], o_ref)),
            "state_equal": bool(torch.equal(st[1], st_ref[1])),
            "others_untouched": bool(torch.equal(st[0], s0()[0]) and torch.equal(st[2], s0()[2])),
        }

    res = {"interpreted": type(getattr(KERNEL, "fn", KERNEL)).__name__}

    # CAPTURE, largest first (the runner's order): each capture is followed by
    # its warmup forward, the first use that computes the pinned tables.
    cu_big, idx_big = meta(BIG, BIG, capture=True)
    run(*inputs(BIG, BIG, 1, 3), s0(), cu_big, idx_big)
    cu_tiny, idx_tiny = meta(TINY, TINY, capture=True)
    run(*inputs(TINY, TINY, 2, 4), s0(), cu_tiny, idx_tiny)
    res["same_object"] = cu_big is cu_tiny
    res["big_chunks"] = int(fidx.graph_static_tables(cu_big)[("chunk_indices", 64)].shape[0])
    res["tiny_chunks"] = int(fidx.graph_static_tables(cu_tiny)[("chunk_indices", 64)].shape[0])
    res["tiny_offsets"] = fidx.graph_static_tables(cu_tiny)[("chunk_offsets", 64)].tolist()

    # REPLAYS: the tiny bucket, then the big one AFTER the tiny capture, then
    # the tiny one again.
    cases = []
    for n, garbage in ((1, 11), (7, 12), (TINY, 13)):
        cases.append(replay_case(TINY, n, 20 + n, garbage, TINY))
    for n, garbage in ((300, 14), (BIG, 15), (64, 16)):
        cases.append(replay_case(BIG, n, 40 + n, garbage, BIG))
    cases.append(replay_case(TINY, 5, 99, 17, TINY))
    res["cases"] = cases
    res["tables_after"] = [
        int(fidx.prepare_chunk_indices(cu_big, 64).shape[0]),
        int(fidx.prepare_chunk_indices(cu_tiny, 64).shape[0]),
    ]

    # CONTROL (the old per-slot sharing): 16-row tensors on the 512 table.
    shared = []
    for n, garbage in ((1, 31), (TINY, 32)):
        q, k, v, gg, beta = inputs(TINY, n, 60 + n, garbage)
        o_ref, st_ref = eager(q, k, v, gg, beta, n)
        cu_big[1] = n
        st = s0()
        o = run(q, k, v, gg, beta, st, cu_big, torch.tensor([1], dtype=torch.int32))
        shared.append({
            "n": n,
            "table_chunks": int(fidx.prepare_chunk_indices(cu_big, 64).shape[0]),
            "o_equal": bool(torch.equal(o[:, :n], o_ref)),
            "state_equal": bool(torch.equal(st[1], st_ref[1])),
        })
    res["shared_control"] = shared
    print("__RESULT__" + json.dumps(res))
    """
).replace("__BIG__", str(BIG)).replace("__TINY__", str(TINY))


def _probe():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "99"
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True,
        text=True,
        timeout=1500,
        env=env,
    )
    marker = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    if not marker:
        raise AssertionError(
            "interpreter probe produced no result\n"
            f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-3000:]}"
        )
    return json.loads(marker[-1][len("__RESULT__"):])


class TestTwoBucketsInSequence(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_the_kernels_really_run_under_the_interpreter(self):
        self.assertEqual(self.res["interpreted"], "InterpretedFunction")

    def test_each_bucket_owns_its_metadata_and_tables(self):
        self.assertFalse(self.res["same_object"])
        self.assertEqual(self.res["big_chunks"], BIG // 64)
        self.assertEqual(self.res["tiny_chunks"], 1)
        self.assertEqual(self.res["tiny_offsets"], [0, 1])
        self.assertEqual(self.res["tables_after"], [BIG // 64, 1])

    def test_every_replay_is_bit_identical_to_eager(self):
        self.assertEqual(len(self.res["cases"]), 7)
        buckets = {c["bucket"] for c in self.res["cases"]}
        self.assertEqual(buckets, {BIG, TINY})
        for case in self.res["cases"]:
            self.assertTrue(case["o_equal"], case)
            self.assertTrue(case["state_equal"], case)
            self.assertTrue(case["others_untouched"], case)

    def test_control_the_shared_table_alone_is_in_bounds(self):
        """What the old per-slot sharing did to the 16-row bucket: the 512
        table (8 chunk programs) on 16-row tensors -- still bit-identical, so
        the shared FLA table by itself was no out-of-bounds access."""
        for case in self.res["shared_control"]:
            self.assertEqual(case["table_chunks"], BIG // 64)
            self.assertTrue(case["o_equal"], case)
            self.assertTrue(case["state_equal"], case)


if __name__ == "__main__":
    unittest.main()
