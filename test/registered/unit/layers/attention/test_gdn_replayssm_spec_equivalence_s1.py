"""ReplaySSM GDN spec-verify kernels match this line's recurrent DFLASH verify (S1).

27B ReplaySSM package, slice S1 (plan: REPLAYSSM_PLAN.md). The kernel file
``srt/layers/attention/fla/gdn_replayssm_spec_decode.py`` is upstream main's
(chain #28695 -> #32692 -> #36970 -> #35544), unwired here. Upstream only ever
ran the GDN route behind the EAGLE/MTP commit and refuses DFLASH for non-KDA
models, so before any wiring this pins the kernels against the kernel the 27B's
D group runs today: ``fused_sigmoid_gating_delta_rule_update`` with per-step
intermediate states (the "speculative intermediate state" the ring replaces),
followed by the scatter of the last accepted step.

Pinned under TRITON_INTERPRET on the production kernels (27B GDN head ratio 3,
dims shrunk; two requests on non-adjacent mamba slots and request rows, accept
3 and 1 incl. the bonus token, a track row at step 1):

* fp32 checkpoint (circular, deferred materialization): verify output, the
  track snapshot, and a SECOND verify that must reconstruct the accepted
  states from checkpoint + ring history, all equal to the recurrent path;
  cursors advance by the accepted counts;
* 16-bit checkpoint (fold every commit + hi/lo low parts; fp16 stands in for
  bf16, which the interpreter cannot run): verify output and the materialized
  checkpoint stay within the recurrent 16-bit path's own rounding of the fp32
  truth, the track snapshot too; cursors return to 0 after the fold; slots
  outside the batch are bit-untouched.
* ``replayssm_ring_bytes_per_req`` prices one request row as the allocation
  will (27B rank geometry, bf16 -> d/k plus their low parts, fp32 g).
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_WORKER = textwrap.dedent(
    """
    import json
    import os

    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

    import torch

    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as recurrent,
    )
    from sglang.srt.layers.attention.fla.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_circular,
        commit_gdn_replayssm_spec,
        gdn_replayssm_spec_decode,
    )

    torch.manual_seed(0)
    H, HV, K, V = 2, 6, 32, 32  # 27B head ratio 3 (16/48), dims shrunk for the interpreter
    STEPS, L = 4, 16  # DFLASH-like window, ring length
    NSLOT, NROW = 5, 3  # mamba slots (+pad), request rows (spec_state_size + 1)
    REQ_SLOT = [1, 3]  # physical mamba slot per request
    REQ_ROW = [0, 2]  # request row (replay index) per request
    B = len(REQ_SLOT)
    T = B * STEPS


    def inputs(dt):
        g = torch.Generator().manual_seed(1)
        q = torch.randn(T, H, K, generator=g).to(dt)
        k = torch.randn(T, H, K, generator=g).to(dt)
        v = (0.5 * torch.randn(T, HV, V, generator=g)).to(dt)
        a = torch.randn(T, HV, generator=g).to(dt)
        b = torch.randn(T, HV, generator=g).to(dt)
        A_log = (0.1 * torch.randn(HV, generator=g)).float()
        dt_bias = (0.1 * torch.randn(HV, generator=g)).float()
        state = (0.2 * torch.randn(NSLOT, HV, V, K, generator=g)).float()
        return q, k, v, a, b, A_log, dt_bias, state


    def run_recurrent(q, k, v, a, b, A_log, dt_bias, state, state_dtype):
        # The fork's DFLASH verify kernel: per-step intermediate states.
        st = state.to(state_dtype).clone()
        inter = torch.zeros(NSLOT, STEPS, HV, V, K, dtype=state_dtype)
        cu = torch.arange(0, T + 1, STEPS, dtype=torch.int32)
        idx = torch.tensor(REQ_SLOT, dtype=torch.int32)
        o = recurrent(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0,
            softplus_threshold=20.0, q=q.view(1, T, H, K), k=k.view(1, T, H, K),
            v=v.view(1, T, HV, V), b=b, initial_state_source=st,
            initial_state_indices=idx, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
            is_kda=False, disable_state_update=True, intermediate_states_buffer=inter,
            intermediate_state_indices=idx, cache_steps=STEPS,
            retrieve_parent_token=None,
        )
        return o.view(T, HV, V).float(), inter.float()


    def new_ring(dt, residual):
        d = torch.zeros(1, NROW, HV, L, V, dtype=dt)
        kk = torch.zeros(1, NROW, H, L, K, dtype=dt)
        g = torch.zeros(1, NROW, HV, L, dtype=torch.float32)
        rv = torch.zeros(1, NROW, HV, L, V, dtype=dt) if residual else None
        rk = torch.zeros(1, NROW, H, L, K, dtype=dt) if residual else None
        cur = (
            torch.zeros(NROW, dtype=torch.int32),
            torch.zeros(NROW, dtype=torch.int32),
            torch.zeros(NROW, dtype=torch.int8),
        )
        return d, kk, g, rv, rk, cur


    def rs_verify(q, k, v, a, b, A_log, dt_bias, ckpt, ring):
        d, kk, g, rv, rk, (wp, cb, fl) = ring
        out = torch.empty(T, HV, V, dtype=q.dtype)
        gdn_replayssm_spec_decode(
            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            checkpoint_state=ckpt[0], d_cache=d[0], k_cache=kk[0], g_cache=g[0],
            rawv_cache=None if rv is None else rv[0],
            rawk_cache=None if rk is None else rk[0], beta_cache=None, out=out,
            query_start_loc=torch.arange(0, T + 1, STEPS, dtype=torch.int32),
            ssm_state_indices=torch.tensor(REQ_SLOT, dtype=torch.int32),
            replay_indices=torch.tensor(REQ_ROW, dtype=torch.int32),
            write_pos=wp, cache_base=cb, is_flush=fl, max_cache_len=L,
            max_spec_len=STEPS, scale=K**-0.5, use_qk_l2norm_in_kernel=True,
            null_block_id=-1, dot_precision="ieee", launch_mode="verify",
        )
        return out.float()


    def rs_commit(ckpt, ring, accept, track_idx=None, track_step=None, fold=False):
        d, kk, g, rv, rk, (wp, cb, fl) = ring
        acc = torch.tensor(accept, dtype=torch.int32)
        rows = torch.tensor(REQ_ROW, dtype=torch.int32)
        commit_gdn_replayssm_spec(
            write_pos=wp, cache_base=cb, is_flush=fl, num_accepted=acc,
            replay_indices=rows, max_cache_len=L, max_spec_len=STEPS,
            fold_every_commit=fold, null_block_id=-1,
        )
        commit_gdn_replayssm_circular(
            checkpoint_state=ckpt, d_cache=d, k_cache=kk, g_cache=g,
            d_residual_cache=rv, k_residual_cache=rk,
            state_batch_indices=torch.tensor(REQ_SLOT, dtype=torch.int32),
            replay_indices=rows, write_pos=wp, cache_base=cb, is_flush=fl,
            accept_lens=acc, mamba_track_indices=track_idx,
            mamba_steps_to_track=track_step, null_block_id=-1,
        )


    def rel(a, b):
        return float((a - b).abs().max() / (b.abs().max() + 1e-12))


    res = {}
    accept = [3, 1]  # includes the bonus token
    TRACK_SLOT = 0

    # ---- fp32 checkpoint (circular, deferred materialization) -------------------
    q, k, v, a, b, A_log, dt_bias, state = inputs(torch.float32)
    o_ref, inter_ref = run_recurrent(q, k, v, a, b, A_log, dt_bias, state, torch.float32)
    ckpt = state.clone()[None]
    ring = new_ring(torch.float32, residual=False)
    o_rs = rs_verify(q, k, v, a, b, A_log, dt_bias, ckpt, ring)
    res["fp32_verify_rel"] = rel(o_rs, o_ref)
    rs_commit(
        ckpt, ring, accept,
        track_idx=torch.tensor([TRACK_SLOT, -1], dtype=torch.int32),
        track_step=torch.tensor([1, -1], dtype=torch.int64),
    )
    res["fp32_track_rel"] = rel(ckpt[0, TRACK_SLOT], inter_ref[REQ_SLOT[0], 1])
    # the effective state after the commit is checkpoint + ring history: a second
    # verify must reproduce the recurrent step-2 output from the accepted states
    q2, k2, v2, a2, b2, *_ = inputs(torch.float32)
    q2, k2, v2 = q2.flip(0), k2.flip(0), v2.flip(0)
    committed = state.clone()
    for r, slot in enumerate(REQ_SLOT):
        committed[slot] = inter_ref[slot, accept[r] - 1]
    o2_ref, inter2_ref = run_recurrent(q2, k2, v2, a2, b2, A_log, dt_bias, committed, torch.float32)
    o2_rs = rs_verify(q2, k2, v2, a2, b2, A_log, dt_bias, ckpt, ring)
    res["fp32_step2_verify_rel"] = rel(o2_rs, o2_ref)
    res["fp32_cursors_after_step1"] = [int(x) for x in ring[5][0]]

    # ---- 16-bit checkpoint (fold every commit + hi/lo low parts) -----------------
    dt16 = torch.float16  # stand-in for bf16: the interpreter cannot run bf16
    q, k, v, a, b, A_log, dt_bias, state = inputs(dt16)
    state16 = state.to(dt16).float()  # the checkpoint as stored
    o_truth, inter_truth = run_recurrent(
        q.float(), k.float(), v.float(), a.float(), b.float(), A_log, dt_bias,
        state16, torch.float32,
    )
    o_rec16, inter_rec16 = run_recurrent(q, k, v, a, b, A_log, dt_bias, state16, dt16)
    ckpt16 = state16.to(dt16).clone()[None]
    ring16 = new_ring(dt16, residual=True)
    o_rs16 = rs_verify(q, k, v, a, b, A_log, dt_bias, ckpt16, ring16)
    res["b16_verify_rel_rs"] = rel(o_rs16, o_truth)
    res["b16_verify_rel_recurrent"] = rel(o_rec16, o_truth)
    rs_commit(
        ckpt16, ring16, accept,
        track_idx=torch.tensor([TRACK_SLOT, -1], dtype=torch.int32),
        track_step=torch.tensor([1, -1], dtype=torch.int64), fold=True,
    )
    errs_rs, errs_rec = [], []
    for r, slot in enumerate(REQ_SLOT):
        truth = inter_truth[slot, accept[r] - 1]
        errs_rs.append(rel(ckpt16[0, slot].float(), truth))
        errs_rec.append(rel(inter_rec16[slot, accept[r] - 1], truth))
    res["b16_commit_rel_rs"] = max(errs_rs)
    res["b16_commit_rel_recurrent"] = max(errs_rec)
    res["b16_track_rel_rs"] = rel(ckpt16[0, TRACK_SLOT].float(), inter_truth[REQ_SLOT[0], 1])
    res["b16_cursors_after_commit"] = [int(x) for x in ring16[5][0]]
    untouched = [s for s in range(NSLOT) if s not in REQ_SLOT and s != TRACK_SLOT]
    res["b16_untouched_slots_exact"] = bool(
        torch.equal(ckpt16[0, untouched], state16.to(dt16)[untouched])
    )
    print("__RESULT__" + json.dumps(res))
    """
)


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


class TestReplaySSMMatchesRecurrentVerify(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_fp32_is_the_recurrent_verify(self):
        r = self.res
        self.assertLess(r["fp32_verify_rel"], 1e-5)
        self.assertLess(r["fp32_track_rel"], 1e-5)
        self.assertLess(r["fp32_step2_verify_rel"], 1e-5)
        # write_pos advanced by the accepted counts on rows 0 and 2 only
        self.assertEqual(r["fp32_cursors_after_step1"], [3, 0, 1])

    def test_16bit_stays_at_the_recurrent_rounding(self):
        r = self.res
        self.assertLess(r["b16_verify_rel_rs"], 4 * r["b16_verify_rel_recurrent"] + 1e-4)
        self.assertLessEqual(
            r["b16_commit_rel_rs"], 1.5 * r["b16_commit_rel_recurrent"] + 1e-4
        )
        self.assertLess(r["b16_track_rel_rs"], 2e-3)
        # fold every commit: the ring is empty again after each commit
        self.assertEqual(r["b16_cursors_after_commit"], [0, 0, 0])
        self.assertTrue(r["b16_untouched_slots_exact"])


class TestRingBytesPerReq(CustomTestCase):
    def test_27b_rank_row_price(self):
        from sglang.srt.configs.mamba_utils import BaseLinearStateParams

        # 27B D rank with HV_local 18 / H_local 6, head dims 128, 48 GDN layers
        fake = SimpleNamespace(
            shape=SimpleNamespace(temporal=(18, 128, 128), num_k_heads_per_tp=6),
            dtype=SimpleNamespace(conv=torch.bfloat16),
            layers=list(range(48)),
            is_kda=False,
        )
        got = BaseLinearStateParams.replayssm_ring_bytes_per_req(fake, record_len=16)
        per_layer = (
            18 * 16 * 128 * 2  # d
            + 6 * 16 * 128 * 2  # k
            + 18 * 16 * 4  # g (fp32)
            + 18 * 16 * 128 * 2  # d low part
            + 6 * 16 * 128 * 2  # k low part
        )
        self.assertEqual(got, per_layer * 48)
        # vs the intermediate state it replaces: 8 draft steps of the full state
        intermediate = 8 * 18 * 128 * 128 * 2 * 48
        self.assertLess(got * 20, intermediate)


if __name__ == "__main__":
    unittest.main()
