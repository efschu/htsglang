"""P prefill graph (27B line, --p-prefill-graph): the GDN extend kernels are
correct with their GRID baked at the captured bucket and the sequence bound
read LIVE on device.

What the full prefill CUDA graph does to the linear-attention layers
--------------------------------------------------------------------
The graph is captured once at the bucket (512 tokens). A rest chunk of n < 512
tokens replays the SAME kernels with the SAME grids; only the static
``query_start_loc`` buffer (hybrid_linear_attn_backend._extend_graph_buffers)
is refreshed to ``[0, n]`` before the replay, and the token rows past n hold
whatever the padded rows computed upstream (zeros in, anything out).

* FLA's chunk tables (``prepare_chunk_indices`` / ``prepare_chunk_offsets``)
  are host-derived (``.tolist()``, a pageable H2D copy) and therefore PINNED to
  that static tensor at capture (fla/index.pin_graph_static_cu_seqlens): they
  keep the BUCKET's chunk count for the process lifetime.
* the triton ``causal_conv1d_fn`` takes its grid from ``seq_lens_cpu``, which
  the capture batch carries as ``[bucket]``.

The claim this file proves on the REAL production kernels (Triton interpreter,
CPU tensors, the 611 test's subprocess pattern): output rows ``[:n]`` and the
in-place state are BIT-IDENTICAL to the eager n-token computation, for n from
1 to the full bucket, with adversarial garbage in the padded rows, and no
state slot but the request's own is written. Plus the control that the baked
table really is the bucket-sized, pinned one (otherwise the equality would be
the eager path compared with itself).

Why a subprocess: see test_gdn_chunk_h_pad_sentinel_611.py (TRITON_INTERPRET
is read at decoration time). The interpreter shims (no CPU device context, no
autotune benchmarking, jit wrappers for the ``exp``/``log`` aliases FLA binds at
import) change nothing the kernels compute.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

BUCKET = 192  # three 64-token FLA chunks: small enough for the interpreter

_WORKER = textwrap.dedent(
    """
    import contextlib, json, os, sys
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
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn,
    )
    import sglang.srt.layers.attention.fla.utils as fu

    # --- interpreter shims (compute-neutral) ---------------------------
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

    H, K, V, SLOTS, BUCKET = 2, 64, 64, 3, __BUCKET__

    def inputs(n, garbage_seed):
        g = torch.Generator().manual_seed(1)
        q = torch.randn(1, BUCKET, H, K, generator=g)
        k = torch.randn(1, BUCKET, H, K, generator=g)
        v = torch.randn(1, BUCKET, H, V, generator=g)
        gg = torch.nn.functional.logsigmoid(torch.randn(1, BUCKET, H, generator=g))
        beta = torch.rand(1, BUCKET, H, generator=g).sigmoid()
        gb = torch.Generator().manual_seed(garbage_seed)
        for t in (q, k, v):
            t[:, n:] = 50.0 * torch.randn(t[:, n:].shape, generator=gb)
        gg[:, n:] = -30.0 * torch.rand(gg[:, n:].shape, generator=gb)
        beta[:, n:] = torch.rand(beta[:, n:].shape, generator=gb)
        return q, k, v, gg, beta

    def s0():
        g = torch.Generator().manual_seed(7)
        return torch.randn(SLOTS, H, V, K, generator=g)

    def run(q, k, v, gg, beta, state, cu):
        o, _ = ChunkGatedDeltaRuleFunction.apply(
            q, k, v, gg, beta, K ** -0.5, state,
            torch.tensor([1], dtype=torch.int32), cu, True,
        )
        return o

    res = {"interpreted": type(getattr(KERNEL, "fn", KERNEL)).__name__}

    # CAPTURE: the static tensor holds the bucket at its first use.
    static_cu = torch.zeros(2, dtype=torch.int32)
    static_cu[1] = BUCKET
    fidx.pin_graph_static_cu_seqlens(static_cu)
    run(*inputs(BUCKET, 3), s0(), static_cu)
    tables = fidx.graph_static_tables(static_cu)
    res["pinned_keys"] = sorted("%s:%d" % kk for kk in tables)
    baked = tables[("chunk_indices", 64)]
    res["baked_chunks"] = int(baked.shape[0])

    gdn = []
    for n, garbage in ((100, 11), (100, 12), (64, 13), (1, 14), (BUCKET, 15), (130, 16)):
        q, k, v, gg, beta = inputs(n, garbage)
        st_ref = s0()
        cu_ref = torch.tensor([0, n], dtype=torch.int32)
        o_ref = run(
            q[:, :n].contiguous(), k[:, :n].contiguous(), v[:, :n].contiguous(),
            gg[:, :n].contiguous(), beta[:, :n].contiguous(), st_ref, cu_ref,
        )
        # REPLAY: bucket-sized tensors, garbage past n, the static tensor.
        static_cu[1] = n
        st_g = s0()
        o_g = run(q, k, v, gg, beta, st_g, static_cu)
        gdn.append({
            "n": n,
            "o_equal": bool(torch.equal(o_g[:, :n], o_ref)),
            "state_equal": bool(torch.equal(st_g[1], st_ref[1])),
            "others_untouched": bool(
                torch.equal(st_g[0], s0()[0]) and torch.equal(st_g[2], s0()[2])
            ),
            "table_is_baked": bool(fidx.prepare_chunk_indices(static_cu, 64) is baked),
            "eager_chunks": int(fidx.prepare_chunk_indices(cu_ref, 64).shape[0]),
        })
    res["gdn"] = gdn

    DIM, W = 96, 4
    gw = torch.Generator().manual_seed(5)
    wts = torch.randn(DIM, W, generator=gw)
    bias = torch.randn(DIM, generator=gw)
    conv = []
    for n, init in ((100, True), (1, False), (BUCKET, True), (3, True)):
        g = torch.Generator().manual_seed(21 + n)
        x_full = torch.randn(BUCKET, DIM, generator=g)
        x_full[n:] = 1e4
        cs0 = torch.randn(SLOTS, DIM, W - 1, generator=g)
        cs_g, cs_r = cs0.clone(), cs0.clone()
        idx = torch.tensor([2], dtype=torch.int32)
        hin = torch.tensor([init])
        qsl = torch.tensor([0, n], dtype=torch.int32)
        out_g = causal_conv1d_fn(
            x_full.transpose(0, 1), wts, bias, cs_g, qsl, [BUCKET],
            cache_indices=idx, has_initial_state=hin, activation="silu",
        )
        out_r = causal_conv1d_fn(
            x_full[:n].transpose(0, 1), wts, bias, cs_r, qsl.clone(), [n],
            cache_indices=idx, has_initial_state=hin, activation="silu",
        )
        conv.append({
            "n": n,
            "out_equal": bool(torch.equal(out_g[:, :n], out_r)),
            "state_equal": bool(torch.equal(cs_g, cs_r)),
        })
    res["conv"] = conv

    # CONSECUTIVE CHUNKS OF ONE REQUEST (prefix > 0): the state written by
    # chunk k is the initial state of chunk k+1. Eager: every chunk exactly its
    # own tokens, fresh cu_seqlens. Graph: bucket tensors, garbage past n, the
    # ONE pinned static cu_seqlens refreshed per chunk. Same chunk boundaries on
    # both sides, so the comparison is bit-exact.
    multi = []
    for plan in ((100, 130, 64), (BUCKET, BUCKET, 40), (1, BUCKET, 7)):
        total = sum(plan)
        g = torch.Generator().manual_seed(31 + total)
        q = torch.randn(1, total, H, K, generator=g)
        k = torch.randn(1, total, H, K, generator=g)
        v = torch.randn(1, total, H, V, generator=g)
        gg = torch.nn.functional.logsigmoid(torch.randn(1, total, H, generator=g))
        beta = torch.rand(1, total, H, generator=g).sigmoid()
        st_e, st_g = s0(), s0()
        outs_e, outs_g = [], []
        pos = 0
        for i, n in enumerate(plan):
            sl = slice(pos, pos + n)
            outs_e.append(run(
                q[:, sl].contiguous(), k[:, sl].contiguous(), v[:, sl].contiguous(),
                gg[:, sl].contiguous(), beta[:, sl].contiguous(), st_e,
                torch.tensor([0, n], dtype=torch.int32),
            ))
            gb = torch.Generator().manual_seed(97 + i)
            def padded(x, fill):
                y = fill((1, BUCKET) + tuple(x.shape[2:]))
                y[:, :n] = x[:, sl]
                return y
            big = lambda s: 50.0 * torch.randn(s, generator=gb)
            static_cu[1] = n
            o = run(
                padded(q, big), padded(k, big), padded(v, big),
                padded(gg, lambda s: -30.0 * torch.rand(s, generator=gb)),
                padded(beta, lambda s: torch.rand(s, generator=gb)),
                st_g, static_cu,
            )
            outs_g.append(o[:, :n])
            pos += n
        multi.append({
            "plan": list(plan),
            "o_equal": bool(torch.equal(torch.cat(outs_g, 1), torch.cat(outs_e, 1))),
            "state_equal": bool(torch.equal(st_g[1], st_e[1])),
            "state_moved": not bool(torch.equal(st_e[1], s0()[1])),
        })
    res["gdn_multi"] = multi

    conv_multi = []
    for plan in ((100, 130, 3), (BUCKET, 1, BUCKET)):
        total = sum(plan)
        g = torch.Generator().manual_seed(61 + total)
        x = torch.randn(total, DIM, generator=g)
        cs0 = torch.randn(SLOTS, DIM, W - 1, generator=g)
        cs_e, cs_g = cs0.clone(), cs0.clone()
        idx = torch.tensor([2], dtype=torch.int32)
        outs_e, outs_g = [], []
        pos = 0
        for i, n in enumerate(plan):
            hin = torch.tensor([i > 0])
            xe = x[pos:pos + n]
            outs_e.append(causal_conv1d_fn(
                xe.transpose(0, 1), wts, bias, cs_e, torch.tensor([0, n], dtype=torch.int32), [n],
                cache_indices=idx, has_initial_state=hin, activation="silu",
            ))
            xg = torch.full((BUCKET, DIM), 1e4)
            xg[:n] = xe
            og = causal_conv1d_fn(
                xg.transpose(0, 1), wts, bias, cs_g, torch.tensor([0, n], dtype=torch.int32), [BUCKET],
                cache_indices=idx, has_initial_state=hin, activation="silu",
            )
            outs_g.append(og[:, :n])
            pos += n
        conv_multi.append({
            "plan": list(plan),
            "out_equal": bool(torch.equal(torch.cat(outs_g, 1), torch.cat(outs_e, 1))),
            "state_equal": bool(torch.equal(cs_g, cs_e)),
        })
    res["conv_multi"] = conv_multi
    print("__RESULT__" + json.dumps(res))
    """
).replace("__BUCKET__", str(BUCKET))


def _probe():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "99"
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("__RESULT__")
    ]
    if not marker:
        raise AssertionError(
            "interpreter probe produced no result\n"
            f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-3000:]}"
        )
    return json.loads(marker[-1][len("__RESULT__") :])


class TestGdnBakedGridLiveBound(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_the_kernels_really_run_under_the_interpreter(self):
        self.assertEqual(self.res["interpreted"], "InterpretedFunction")

    def test_the_table_is_the_buckets_and_it_is_pinned(self):
        """Control: without this every equality below could be the eager
        path compared with itself."""
        self.assertEqual(self.res["baked_chunks"], BUCKET // 64)
        self.assertIn("chunk_indices:64", self.res["pinned_keys"])
        for case in self.res["gdn"]:
            self.assertTrue(case["table_is_baked"], case)
        # at least one case really runs a SHORTER sequence on the baked grid
        self.assertTrue(
            any(c["eager_chunks"] < BUCKET // 64 for c in self.res["gdn"])
        )

    def test_gdn_output_rows_are_bit_identical_to_eager(self):
        for case in self.res["gdn"]:
            self.assertTrue(case["o_equal"], case)

    def test_gdn_final_state_is_bit_identical_and_no_other_slot_moves(self):
        for case in self.res["gdn"]:
            self.assertTrue(case["state_equal"], case)
            self.assertTrue(case["others_untouched"], case)

    def test_conv_rows_and_state_are_bit_identical_to_eager(self):
        for case in self.res["conv"]:
            self.assertTrue(case["out_equal"], case)
            self.assertTrue(case["state_equal"], case)

    def test_consecutive_chunks_carry_the_state_like_eager(self):
        """Prefix > 0, several chunks of one request (the xsn428 question,
        candidate 2): the graph form's chunk k+1 starts from the state its
        chunk k wrote, bit-identical to eager chunk by chunk."""
        self.assertEqual(len(self.res["gdn_multi"]), 3)
        for case in self.res["gdn_multi"]:
            self.assertTrue(case["state_moved"], case)  # control: real carry
            self.assertTrue(case["o_equal"], case)
            self.assertTrue(case["state_equal"], case)

    def test_consecutive_conv_chunks_carry_the_window_like_eager(self):
        self.assertEqual(len(self.res["conv_multi"]), 2)
        for case in self.res["conv_multi"]:
            self.assertTrue(case["out_equal"], case)
            self.assertTrue(case["state_equal"], case)


if __name__ == "__main__":
    unittest.main()
