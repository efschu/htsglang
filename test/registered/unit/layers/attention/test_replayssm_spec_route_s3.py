"""ReplaySSM spec ring wired into the GDN verify and the DFLASH commit (S3).

27B ReplaySSM package, slice S3 (REPLAYSSM_PLAN.md). With the spec ring
allocated (S2):

* the verify metadata plans the request rows (``replayssm_spec_rows`` =
  req_pool_indices; eager, graph capture and graph replay) and re-states their
  cursors (write_pos = is_flush = 0) outside a capture;
* ``GDNAttnBackend.forward_extend`` routes the target verify to the ring
  (``_replayssm_target_verify``, upstream's kernel call) -- a draft tree or a
  pool with neither the intermediate state nor planned rows is refused;
* ``update_mamba_state_after_mtp_verify`` -- the hook DFLASH, the EAGLE/MTP
  commit and the lane all call -- folds the accepted window from the ring
  (fold every commit, any SSM dtype), writes the track snapshot, and rolls the
  conv state back from the conv verify windows;
* a draft-KV-only producer (no verify workspace) treats the flag as a no-op.

The interpreter half runs the production route functions on real CPU pools
(ring vs. intermediate), two layers, 27B head ratio 3 with shrunk dims, two
requests on non-adjacent rows/slots plus a padded row, a track crossing, then
a second step after the ring pool's cursors were overwritten with garbage (a
TMS restore) and re-stated by the metadata heal. The CUDA-only fused scatters
are replaced by a CPU reference for BOTH routes.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LAYERS = [0, 1]
H, HV, K, V = 2, 6, 16, 16


def _params():
    conv_dim = (2 * H + HV) * K
    shape = Mamba2StateShape(
        conv=[(conv_dim, 3)],
        temporal=(HV, V, K),
        intermediate_size=HV * V,
        conv_dim=conv_dim,
        ssm_state_size=K,
        num_heads=HV,
        head_dim=V,
        state_size=K,
        conv_kernel=4,
        num_k_heads_per_tp=H,
    )
    return Mamba2CacheParams(
        shape=shape,
        layers=LAYERS,
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
    )


def _rtp(spec):
    return HybridReqToTokenPool(
        size=4,
        mamba_size=6,
        mamba_spec_state_size=4,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
        cache_params=_params(),
        mamba_layer_ids=LAYERS,
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        linear_replayssm_cache_len=16,
        enable_linear_replayssm_spec=spec,
    )


def _backend(rtp):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )

    runner = SimpleNamespace(
        device="cpu",
        server_args=SimpleNamespace(
            speculative_eagle_topk=None, enable_unified_memory=False
        ),
        is_draft_worker=False,
        req_to_token_pool=rtp,
        token_to_kv_pool=None,
    )
    return MambaAttnBackendBase(runner)


def _verify_batch(rows):
    return SimpleNamespace(
        batch_size=len(rows),
        req_pool_indices=torch.tensor(rows, dtype=torch.int64),
        mamba_track_indices=None,
        _original_batch_size=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=torch.zeros(4 * len(rows), dtype=torch.int64),
        spec_info=SimpleNamespace(draft_token_num=4),
        mamba_track_mask=None,
    )


def _poison(pool):
    pool.replayssm_spec_write_pos.fill_(7)
    pool.replayssm_is_flush.fill_(1)
    pool.replayssm_cache_base.fill_(5)


class _Base(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestVerifyMetadataPlansRows(_Base):
    def test_eager_verify_plans_and_heals_the_batch_rows(self):
        rtp = _rtp(True)
        pool = rtp.mamba_pool
        _poison(pool)
        fb = _verify_batch([2, 4])
        md = _backend(rtp)._forward_metadata(fb)
        self.assertIs(md.replayssm_spec_rows, fb.req_pool_indices)
        self.assertEqual(pool.replayssm_spec_write_pos.tolist(), [7, 7, 0, 7, 0])
        self.assertEqual(pool.replayssm_is_flush.tolist(), [1, 1, 0, 1, 0])
        # the base is circular and harmless at write_pos 0 -- left alone
        self.assertEqual(pool.replayssm_cache_base.tolist(), [5] * 5)

    def test_flag_off_and_decode_plan_nothing(self):
        md = _backend(_rtp(False))._forward_metadata(_verify_batch([2, 4]))
        self.assertIsNone(md.replayssm_spec_rows)
        rtp = _rtp(True)
        fb = _verify_batch([2, 4])
        fb.forward_mode = ForwardMode.DECODE
        md = _backend(rtp)._forward_metadata(fb)
        self.assertIsNone(md.replayssm_spec_rows)
        self.assertEqual(rtp.mamba_pool.replayssm_spec_write_pos.tolist(), [0] * 5)

    def test_graph_replay_plans_the_static_rows_and_heals_outside_capture(self):
        rtp = _rtp(True)
        pool = rtp.mamba_pool
        be = _backend(rtp)
        be.init_cuda_graph_state(max_bs=4, max_num_tokens=16)
        spec = SimpleNamespace(draft_token_num=4)
        static = torch.tensor([2, 4, 3], dtype=torch.int64)  # row 3: stale pad
        _poison(pool)
        md = be._replay_metadata(
            3, static, ForwardMode.TARGET_VERIFY, spec, None, num_padding=1
        )
        self.assertIs(md.replayssm_spec_rows, static)
        self.assertEqual(static.tolist(), [2, 4, 0])  # padding -> request row 0
        self.assertEqual(pool.replayssm_spec_write_pos.tolist(), [0, 7, 0, 7, 0])
        self.assertEqual(pool.replayssm_is_flush.tolist(), [0, 1, 0, 1, 0])
        _poison(pool)
        md = be._replay_metadata(
            3,
            torch.tensor([2, 4, 1], dtype=torch.int64),
            ForwardMode.TARGET_VERIFY,
            spec,
            None,
            num_padding=0,
            in_capture=True,
        )
        self.assertIsNotNone(md.replayssm_spec_rows)
        self.assertEqual(pool.replayssm_spec_write_pos.tolist(), [7] * 5)


class TestRoutesRefuse(_Base):
    def _extend(self, rtp, rows, parent):
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
        from sglang.srt.layers.attention.mamba.mamba2_metadata import (
            ForwardMetadata,
        )

        stub = SimpleNamespace(
            forward_metadata=ForwardMetadata(
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                mamba_cache_indices=torch.tensor([1], dtype=torch.int32),
                retrieve_parent_token=parent,
                replayssm_spec_rows=rows,
            ),
            req_to_token_pool=rtp,
        )
        fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        layer = SimpleNamespace(layer_id=0)
        GDNAttnBackend.forward_extend(
            stub, layer, fb, torch.zeros(4, 8), torch.zeros(4, HV), torch.zeros(4, HV)
        )

    def test_ring_pool_without_planned_rows(self):
        with self.assertRaisesRegex(RuntimeError, "neither the per-draft"):
            self._extend(_rtp(True), None, None)

    def test_draft_tree_on_the_ring(self):
        with self.assertRaisesRegex(RuntimeError, "linear draft chain only"):
            self._extend(
                _rtp(True),
                torch.tensor([2], dtype=torch.int64),
                torch.zeros(1, 4, dtype=torch.int32),
            )

    def test_commit_without_planned_rows(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.mamba.mamba2_metadata import (
            ForwardMetadata,
        )

        hyb = object.__new__(HybridLinearAttnBackend)
        hyb.linear_attn_backend = SimpleNamespace(
            forward_metadata=ForwardMetadata(
                query_start_loc=None,
                mamba_cache_indices=torch.tensor([1], dtype=torch.int32),
            ),
            req_to_token_pool=_rtp(True),
        )
        with self.assertRaisesRegex(RuntimeError, "planned no ring rows"):
            hyb.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=torch.tensor([1]),
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=None,
            )


class TestProducerIsANoOp(_Base):
    def test_spec_for(self):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            _replayssm_spec_for,
        )

        on = SimpleNamespace(enable_linear_replayssm_spec=True)
        producer = SimpleNamespace(
            server_args=on, is_draft_kv_only_producer=True, hybrid_gdn_config=None
        )
        self.assertFalse(_replayssm_spec_for(producer))
        target = SimpleNamespace(
            server_args=on, is_draft_kv_only_producer=False, hybrid_gdn_config=object()
        )
        self.assertTrue(_replayssm_spec_for(target))
        target.hybrid_gdn_config = None
        with self.assertRaisesRegex(ValueError, "only the GDN verify route"):
            _replayssm_spec_for(target)


_WORKER = textwrap.dedent(
    """
    import json
    import os

    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

    from types import SimpleNamespace

    import torch

    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateDType,
        Mamba2StateShape,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as recurrent,
    )
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
        MambaAttnBackendBase,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    import sglang.srt.layers.attention.hybrid_linear_attn_backend as hlab


    def _ref_scatter(dst, src, dst_idx, steps):
        # CPU stand-in for the CUDA-only fused scatters (both routes use it):
        # dst[:, dst_idx[i]] = src[:, i, steps[i]] for steps[i] >= 0
        for i in range(steps.shape[0]):
            s, d = int(steps[i]), int(dst_idx[i])
            if s >= 0 and d >= 0:
                dst[:, d] = src[:, i, s]


    hlab.fused_conv_window_scatter_with_mask = _ref_scatter
    hlab.fused_mamba_state_scatter_with_mask = _ref_scatter

    LAYERS = [0, 1]
    H, HV, K, V = 2, 6, 32, 32  # 27B head ratio 3, dims shrunk for the interpreter
    STEPS, L = 4, 16
    ROWS = [2, 4]  # request rows (req_pool_indices) of the two live requests
    SLOTS = [1, 3]  # their mamba slots
    TRACK = 5  # extra_buffer track slot of request 0
    CONV_DIM = (2 * H + HV) * K


    def params(dt, temporal_dt):
        shape = Mamba2StateShape(
            conv=[(CONV_DIM, 3)],
            temporal=(HV, V, K),
            intermediate_size=HV * V,
            conv_dim=CONV_DIM,
            ssm_state_size=K,
            num_heads=HV,
            head_dim=V,
            state_size=K,
            conv_kernel=4,
            num_k_heads_per_tp=H,
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=LAYERS,
            dtype=Mamba2StateDType(conv=dt, temporal=temporal_dt),
        )


    def pool(spec, dt, temporal_dt):
        return HybridReqToTokenPool(
            size=4,
            mamba_size=6,
            mamba_spec_state_size=4,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
            cache_params=params(dt, temporal_dt),
            mamba_layer_ids=LAYERS,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=STEPS,
            speculative_eagle_topk=1,
            linear_replayssm_cache_len=L,
            enable_linear_replayssm_spec=spec,
        )


    def seed_state(rtp, gen):
        c = rtp.mamba_pool.mamba_cache
        c.temporal.copy_((0.2 * torch.randn(c.temporal.shape, generator=gen)).to(c.temporal.dtype))
        c.conv[0].copy_(torch.randn(c.conv[0].shape, generator=gen).to(c.conv[0].dtype))


    def copy_state(dst, src):
        d, s = dst.mamba_pool.mamba_cache, src.mamba_pool.mamba_cache
        d.temporal.copy_(s.temporal)
        d.conv[0].copy_(s.conv[0])


    def layers_(gen):
        return [
            SimpleNamespace(
                num_k_heads=H,
                head_k_dim=K,
                num_v_heads=HV,
                head_v_dim=V,
                A_log=(0.1 * torch.randn(HV, generator=gen)).float(),
                dt_bias=(0.1 * torch.randn(HV, generator=gen)).float(),
            )
            for _ in LAYERS
        ]


    def window(gen, dt, bs):
        T = bs * STEPS
        per_layer = []
        for _ in LAYERS:
            per_layer.append(
                dict(
                    q=torch.randn(1, T, H, K, generator=gen).to(dt),
                    k=torch.randn(1, T, H, K, generator=gen).to(dt),
                    v=(0.5 * torch.randn(1, T, HV, V, generator=gen)).to(dt),
                    a=torch.randn(T, HV, generator=gen).to(dt),
                    b=torch.randn(T, HV, generator=gen).to(dt),
                )
            )
        return per_layer


    def conv_windows(rtp, gen):
        w = rtp.mamba_pool.mamba_cache.intermediate_conv_window[0]
        w.copy_(torch.randn(w.shape, generator=gen).to(w.dtype))


    def verify(rtp, lay, win, slots, rows, bs, ring):
        # One target verify over every GDN layer, the way forward_extend routes.
        qsl = torch.arange(0, bs * STEPS + 1, STEPS, dtype=torch.int32)
        outs = []
        for li, (layer, x) in enumerate(zip(lay, win)):
            cache = rtp.mamba2_layer_cache(LAYERS[li])
            if ring:
                o = GDNAttnBackend._replayssm_target_verify(
                    SimpleNamespace(req_to_token_pool=rtp),
                    layer=layer,
                    query=x["q"],
                    key=x["k"],
                    value=x["v"],
                    a=x["a"],
                    b=x["b"],
                    layer_cache=cache,
                    cache_indices=slots,
                    replay_indices=rows,
                    query_start_loc=qsl,
                    draft_token_num=STEPS,
                )
            else:
                o = TritonGDNKernel().target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=x["q"],
                    k=x["k"],
                    v=x["v"],
                    a=x["a"],
                    b=x["b"],
                    ssm_states=cache.temporal,
                    cache_indices=slots,
                    query_start_loc=qsl,
                    intermediate_states_buffer=cache.intermediate_ssm,
                    intermediate_state_indices=torch.arange(bs, dtype=torch.int32),
                    cache_steps=STEPS,
                    retrieve_parent_token=None,
                )
            outs.append(o.reshape(bs * STEPS, HV, V).float())
        return outs


    def commit(rtp, slots, rows, last_steps, track_idx, track_steps, ring):
        hyb = object.__new__(HybridLinearAttnBackend)
        hyb.linear_attn_backend = SimpleNamespace(
            forward_metadata=ForwardMetadata(
                query_start_loc=None,
                mamba_cache_indices=slots,
                replayssm_spec_rows=rows if ring else None,
            ),
            req_to_token_pool=rtp,
        )
        hyb.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_steps,
            mamba_track_indices=track_idx,
            mamba_steps_to_track=track_steps,
            model=None,
        )


    def truth(state0, lay, win, bs):
        # fp32 recurrent run on the (rounded) inputs: per-step states per layer.
        inters, outs = [], []
        qsl = torch.arange(0, bs * STEPS + 1, STEPS, dtype=torch.int32)
        slots = torch.tensor(SLOTS + [-1] * (bs - 2), dtype=torch.int32)
        for li, (layer, x) in enumerate(zip(lay, win)):
            st = state0[li].float().clone()
            inter = torch.zeros(bs, STEPS, HV, V, K)
            o = recurrent(
                A_log=layer.A_log, a=x["a"].float(), dt_bias=layer.dt_bias,
                softplus_beta=1.0, softplus_threshold=20.0, q=x["q"].float(),
                k=x["k"].float(), v=x["v"].float(), b=x["b"].float(),
                initial_state_source=st, initial_state_indices=slots,
                use_qk_l2norm_in_kernel=True, cu_seqlens=qsl, is_kda=False,
                disable_state_update=True, intermediate_states_buffer=inter,
                intermediate_state_indices=torch.arange(bs, dtype=torch.int32),
                cache_steps=STEPS, retrieve_parent_token=None,
            )
            inters.append(inter)
            outs.append(o.reshape(bs * STEPS, HV, V).float())
        return outs, inters


    def rel(a, b):
        return float((a - b).abs().max() / (b.abs().max() + 1e-12))


    def run(dt, temporal_dt, tag, res):
        gen = torch.Generator().manual_seed(7)
        ring_p = pool(True, dt, temporal_dt)
        rec_p = pool(False, dt, temporal_dt)
        seed_state(rec_p, gen)
        copy_state(ring_p, rec_p)
        lay = layers_(gen)
        untouched = [s for s in range(7) if s not in SLOTS and s != TRACK]
        before_untouched = ring_p.mamba_pool.mamba_cache.temporal[:, untouched].clone()

        # ---- step 1: two live requests + one padded row (slot -1, request row 0),
        # accept 3 and 1 (bonus included), request 0 crosses a track point at step 1
        bs = 3
        slots = torch.tensor(SLOTS + [-1], dtype=torch.int32)
        rows = torch.tensor(ROWS + [0], dtype=torch.int64)
        win = window(gen, dt, bs)
        conv_windows(rec_p, gen)
        ring_p.mamba_pool.mamba_cache.intermediate_conv_window[0].copy_(
            rec_p.mamba_pool.mamba_cache.intermediate_conv_window[0]
        )
        state0 = ring_p.mamba_pool.mamba_cache.temporal.clone()
        t_out, t_inter = truth(state0, lay, win, bs)
        o_ring = verify(ring_p, lay, win, slots, rows, bs, ring=True)
        o_rec = verify(rec_p, lay, win, slots, rows, bs, ring=False)
        real = slice(0, 2 * STEPS)
        res[f"{tag}_s1_verify_rel_ring"] = max(rel(a[real], t[real]) for a, t in zip(o_ring, t_out))
        res[f"{tag}_s1_verify_rel_rec"] = max(rel(a[real], t[real]) for a, t in zip(o_rec, t_out))
        res[f"{tag}_s1_pad_out_zero"] = bool(all(torch.all(a[2 * STEPS:] == 0) for a in o_ring))
        last = torch.tensor([2, 0], dtype=torch.int64)
        track_idx = torch.tensor([TRACK, -1], dtype=torch.int64)
        track_steps = torch.tensor([1, -1], dtype=torch.int64)
        commit(ring_p, slots, rows, last, track_idx, track_steps, ring=True)
        commit(rec_p, slots, rows, last, track_idx, track_steps, ring=False)
        tr_ring = ring_p.mamba_pool.mamba_cache.temporal.float()
        tr_rec = rec_p.mamba_pool.mamba_cache.temporal.float()
        e_ring, e_rec = [], []
        for li in range(len(LAYERS)):
            for r, slot in enumerate(SLOTS):
                want = t_inter[li][r, int(last[r])]
                e_ring.append(rel(tr_ring[li, slot], want))
                e_rec.append(rel(tr_rec[li, slot], want))
            want_t = t_inter[li][0, 1]
            e_ring.append(rel(tr_ring[li, TRACK], want_t))
            e_rec.append(rel(tr_rec[li, TRACK], want_t))
        res[f"{tag}_s1_commit_rel_ring"] = max(e_ring)
        res[f"{tag}_s1_commit_rel_rec"] = max(e_rec)
        res[f"{tag}_s1_conv_equal"] = bool(
            torch.equal(ring_p.mamba_pool.mamba_cache.conv[0], rec_p.mamba_pool.mamba_cache.conv[0])
        )
        mp = ring_p.mamba_pool
        res[f"{tag}_s1_cursors"] = [
            mp.replayssm_spec_write_pos.tolist(),
            mp.replayssm_is_flush.tolist(),
        ]
        res[f"{tag}_s1_untouched_exact"] = bool(
            torch.equal(mp.mamba_cache.temporal[:, untouched], before_untouched)
        )

        # ---- step 2: the ring pool's cursors of the batch rows are garbage (as
        # after a TMS restore); the verify metadata's heal re-states them. Accept
        # 4 and 2, no track row this step.
        mp.replayssm_spec_write_pos[ROWS] = 7
        mp.replayssm_is_flush[ROWS] = 1
        mp.replayssm_cache_base[ROWS] = 5
        rows2 = torch.tensor(ROWS, dtype=torch.int64)
        slots2 = torch.tensor(SLOTS, dtype=torch.int32)
        got_rows = MambaAttnBackendBase._replayssm_spec_rows(
            SimpleNamespace(req_to_token_pool=ring_p), rows2, heal=True
        )
        res[f"{tag}_heal_returns_rows"] = bool(got_rows is rows2)
        res[f"{tag}_heal_cursors"] = [
            mp.replayssm_spec_write_pos.tolist(),
            mp.replayssm_is_flush.tolist(),
            mp.replayssm_cache_base.tolist(),
        ]
        bs = 2
        win2 = window(gen, dt, bs)
        conv_windows(rec_p, gen)
        ring_p.mamba_pool.mamba_cache.intermediate_conv_window[0].copy_(
            rec_p.mamba_pool.mamba_cache.intermediate_conv_window[0]
        )
        # truth continues from the RECURRENT pool's committed state (the reference
        # line), so the ring's step-1 rounding does not leak into step 2's metric
        state1 = rec_p.mamba_pool.mamba_cache.temporal.clone()
        ring_p.mamba_pool.mamba_cache.temporal.copy_(state1)
        t_out2, t_inter2 = truth(state1, lay, win2, bs)
        o_ring2 = verify(ring_p, lay, win2, slots2, rows2, bs, ring=True)
        o_rec2 = verify(rec_p, lay, win2, slots2, rows2, bs, ring=False)
        res[f"{tag}_s2_verify_rel_ring"] = max(rel(a, t) for a, t in zip(o_ring2, t_out2))
        res[f"{tag}_s2_verify_rel_rec"] = max(rel(a, t) for a, t in zip(o_rec2, t_out2))
        last2 = torch.tensor([3, 1], dtype=torch.int64)
        commit(ring_p, slots2, rows2, last2, None, None, ring=True)
        commit(rec_p, slots2, rows2, last2, None, None, ring=False)
        tr_ring = ring_p.mamba_pool.mamba_cache.temporal.float()
        tr_rec = rec_p.mamba_pool.mamba_cache.temporal.float()
        e_ring, e_rec = [], []
        for li in range(len(LAYERS)):
            for r, slot in enumerate(SLOTS):
                want = t_inter2[li][r, int(last2[r])]
                e_ring.append(rel(tr_ring[li, slot], want))
                e_rec.append(rel(tr_rec[li, slot], want))
        res[f"{tag}_s2_commit_rel_ring"] = max(e_ring)
        res[f"{tag}_s2_commit_rel_rec"] = max(e_rec)
        res[f"{tag}_s2_cursors_wp"] = mp.replayssm_spec_write_pos.tolist()


    res = {}
    run(torch.float32, torch.float32, "fp32", res)
    # fp16 stands in for bf16, which the interpreter cannot run
    run(torch.float16, torch.float16, "b16", res)
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
        timeout=1200,
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


class TestRingRouteMatchesTheRecurrentRoute(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_fp32(self):
        r = self.res
        for step in ("s1", "s2"):
            self.assertLess(r[f"fp32_{step}_verify_rel_ring"], 1e-5)
            self.assertLess(r[f"fp32_{step}_commit_rel_ring"], 1e-5)
        self.assertTrue(r["fp32_s1_pad_out_zero"])
        self.assertTrue(r["fp32_s1_conv_equal"])
        self.assertTrue(r["fp32_s1_untouched_exact"])

    def test_16bit(self):
        r = self.res
        for step in ("s1", "s2"):
            self.assertLess(
                r[f"b16_{step}_verify_rel_ring"],
                4 * r[f"b16_{step}_verify_rel_rec"] + 1e-4,
            )
            self.assertLessEqual(
                r[f"b16_{step}_commit_rel_ring"],
                1.5 * r[f"b16_{step}_commit_rel_rec"] + 1e-4,
            )
        self.assertTrue(r["b16_s1_pad_out_zero"])
        self.assertTrue(r["b16_s1_conv_equal"])
        self.assertTrue(r["b16_s1_untouched_exact"])

    def test_fold_every_commit_and_heal(self):
        r = self.res
        for tag in ("fp32", "b16"):
            # every commit folds, whatever the SSM dtype: cursors back to 0
            self.assertEqual(r[f"{tag}_s1_cursors"], [[0] * 5, [0] * 5])
            self.assertEqual(r[f"{tag}_s2_cursors_wp"], [0] * 5)
            # the heal re-states the batch rows' write_pos / is_flush only
            self.assertTrue(r[f"{tag}_heal_returns_rows"])
            self.assertEqual(
                r[f"{tag}_heal_cursors"], [[0] * 5, [0] * 5, [0, 0, 5, 0, 5]]
            )


if __name__ == "__main__":
    unittest.main()
