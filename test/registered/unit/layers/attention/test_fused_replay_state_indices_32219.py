"""fused_replay_state_indices is bit-identical to the unfused replay prep (#32219).

Ported from upstream sglang #32219 (``test/registered/kernels/ops/mamba/
test_fused_replay_state_indices.py``). Upstream runs it on a GPU; here the SAME
production kernel runs under Triton's interpreter on CPU tensors (subprocess,
see test_gdn_chunk_h_pad_sentinel_611.py for why).

The unfused reference is the op sequence ``_replay_metadata`` launched for the
static hybrid pool before every decode / DFLASH target-verify graph replay:

    req_pool_indices[valid_bs:total_bs] = 0        # zero padded rows (side effect)
    mamba_indices = mapping[req_pool_indices]      # get_mamba_indices gather
    # identity v2p translate (static pool)
    mamba_indices[valid_bs:] = -1                  # padding sentinel
    state_indices[:total_bs].copy_(mamba_indices)

Pinned: identical state indices over [0, total_bs) (sentinels included), the
identical req_pool_indices side effect (padded rows zeroed -- a non-zeroed row
is a delayed illegal gather in a captured kernel), no write past total_bs, over
a bs x num_padding matrix with non-power-of-two sizes (BS_UPPER masking).

Fork addition: the fast path's eligibility is asked per replay and also
requires ``get_mamba_indices`` to be the base flat gather (the kernel reads the
mapping table directly); the unified pool (non-identity translate), a pool
overriding get_mamba_indices, ReplaySSM and a non-CUDA device keep the chain.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    from sglang.srt.layers.attention.mamba.mamba_state_indices_triton import (
        fused_replay_state_indices,
    )

    GUARD, GUARD_SENTINEL, OUT_POISON = 8, -7777, -12345
    REQ_POOL, MAMBA_POOL = 160, 4096

    def reference(req_pool_indices, mapping, out_buf, valid_bs, total_bs):
        req_pool_indices[valid_bs:total_bs] = 0
        mamba_indices = mapping[req_pool_indices[:total_bs]]
        mamba_indices[valid_bs:] = -1
        out_buf[: len(mamba_indices)].copy_(mamba_indices)

    failures, cases = [], 0
    for total_bs in (1, 2, 3, 5, 8, 13, 16, 33):
        for num_padding in sorted({0, 1, total_bs // 2, total_bs - 1, total_bs}):
            for seed in (0, 1):
                cases += 1
                g = torch.Generator().manual_seed(seed * 1000 + total_bs)
                valid_bs = total_bs - num_padding
                req_pool = torch.randint(0, REQ_POOL, (total_bs + GUARD,),
                                         generator=g, dtype=torch.int64)
                req_pool[total_bs:] = GUARD_SENTINEL
                mapping = torch.randint(0, MAMBA_POOL, (REQ_POOL,), generator=g,
                                        dtype=torch.int32)
                out = torch.full((total_bs + GUARD,), OUT_POISON, dtype=torch.int32)
                rp_ref, rp_f = req_pool.clone(), req_pool.clone()
                out_ref, out_f = out.clone(), out.clone()
                reference(rp_ref, mapping, out_ref, valid_bs, total_bs)
                ret = fused_replay_state_indices(
                    req_pool_indices=rp_f, mamba_index_mapping=mapping,
                    out_state_indices=out_f, valid_bs=valid_bs, total_bs=total_bs)
                ok = (
                    torch.equal(out_ref[:total_bs], out_f[:total_bs])
                    and torch.equal(ret, out_f[:total_bs])
                    and torch.equal(rp_ref[:total_bs], rp_f[:total_bs])
                    and bool((rp_f[valid_bs:total_bs] == 0).all())
                    and bool((out_f[valid_bs:total_bs] == -1).all())
                    and not bool((out_f[:total_bs] == OUT_POISON).any())
                    and bool((rp_f[total_bs:] == GUARD_SENTINEL).all())
                    and bool((out_f[total_bs:] == OUT_POISON).all())
                )
                if not ok:
                    failures.append([total_bs, num_padding, seed])
    print("__RESULT__" + json.dumps({"cases": cases, "failures": failures}))
    """)


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


class TestFusedReplayStateIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_bit_identical_to_reference_chain(self):
        self.assertGreater(self.res["cases"], 50)
        self.assertEqual(self.res["failures"], [])


class _Unified(HybridReqToTokenPool):
    def translate_mamba_indices(self, mamba_indices):  # non-identity v2p
        return mamba_indices + 1


class _Rerouted(HybridReqToTokenPool):
    def get_mamba_indices(self, req_indices):  # not the flat table gather
        return super().get_mamba_indices(req_indices)


class TestFastPathEligibility(CustomTestCase):
    def backend(self, pool, device="cuda:0", replayssm=None):
        be = object.__new__(MambaAttnBackendBase)
        be.req_to_token_pool = pool
        be.device = torch.device(device)
        be.replayssm_write_pos_list = replayssm
        return be

    def test_static_hybrid_pool_on_cuda_takes_the_fast_path(self):
        pool = object.__new__(HybridReqToTokenPool)
        self.assertTrue(self.backend(pool)._fused_state_indices_ok())

    def test_everything_else_keeps_the_reference_chain(self):
        cases = {
            "unified_translate": self.backend(object.__new__(_Unified)),
            "rerouted_gather": self.backend(object.__new__(_Rerouted)),
            "replayssm": self.backend(
                object.__new__(HybridReqToTokenPool), replayssm=[torch.zeros(1)]
            ),
            "cpu_device": self.backend(
                object.__new__(HybridReqToTokenPool), device="cpu"
            ),
            "not_a_hybrid_pool": self.backend(object()),
        }
        for name, be in cases.items():
            with self.subTest(case=name):
                self.assertFalse(be._fused_state_indices_ok())


if __name__ == "__main__":
    unittest.main()
