"""The spec ring survives garbage memory and stays out of the state set (S4).

27B ReplaySSM package, slice S4 (REPLAYSSM_PLAN.md: flip / TMS). The ring
lives in the mamba pool's KV_CACHE TMS region; after a pause/resume that
memory holds any bit pattern. Two things make that harmless:

* the verify metadata re-states write_pos / is_flush of the step's rows (S3
  heal), so the verify reads no history, and the compact commit now ZEROES
  the lanes past the accepted history after loading them (the one fork change
  inside the ported kernel) -- a NaN-filled ring with garbage cursors gives
  bit-identical outputs and checkpoints to a zero-initialised one, over three
  steps with and without a track row (red without the kernel change: NaN
  reaches the checkpoint);
* the ring is request-row verify scratch, not session state: the GDN state
  set (offload register bytes, export/import blob) excludes it under the spec
  flag, exactly like intermediate_ssm, while the SAME field names stay in the
  set under the slot-keyed decode ring.
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.offload_gdn_states import (
    mamba_state_set_nbytes,
    transient_state_fields,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

LAYERS = [0, 1, 2]
HV, HK, K, V = 6, 2, 16, 16
RING = ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk")


def _params():
    conv_dim = (2 * HK + HV) * K
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
        num_k_heads_per_tp=HK,
    )
    return Mamba2CacheParams(
        shape=shape,
        layers=LAYERS,
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
    )


def _spec_pool():
    # 6 state slots, only 2 request rows (+1): a slot index past the ring rows
    return MambaPool(
        size=6,
        spec_state_size=2,
        cache_params=_params(),
        mamba_layer_ids=LAYERS,
        device="cpu",
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        linear_replayssm_cache_len=16,
        enable_linear_replayssm_spec=True,
    )


class _Base(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestSpecRingIsNotSessionState(_Base):
    def test_blob_fields(self):
        names = _spec_pool().state_blob_fields()
        for n in RING:
            self.assertNotIn(n, names)
        self.assertIn("temporal", names)
        self.assertIn("conv", names)

    def test_blob_round_trip_on_a_slot_past_the_ring_rows(self):
        pool = _spec_pool()
        c = pool.mamba_cache
        g = torch.Generator().manual_seed(3)
        c.temporal.copy_(torch.randn(c.temporal.shape, generator=g).to(c.temporal.dtype))
        c.conv[0].copy_(torch.randn(c.conv[0].shape, generator=g).to(c.conv[0].dtype))
        blob = pool.export_state_blob(5)
        self.assertEqual(set(blob), set(pool.state_blob_fields()))
        pool.import_state_blob(4, blob)
        self.assertTrue(torch.equal(c.temporal[:, 4], c.temporal[:, 5]))
        self.assertTrue(torch.equal(c.conv[0][:, 4], c.conv[0][:, 5]))

    def test_set_bytes(self):
        pool = _spec_pool()
        c = pool.mamba_cache
        slots = pool.size + 1
        want = (
            c.temporal.numel() * c.temporal.element_size()
            + c.conv[0].numel() * c.conv[0].element_size()
        ) // slots
        self.assertEqual(mamba_state_set_nbytes(c, slots, spec_ring=True), want)
        # without the flag the names count as the decode ring's slot state
        self.assertGreater(mamba_state_set_nbytes(c, slots), want)

    def test_pd_transfer_dims_mirror_the_buffer_list(self):
        # get_state_dim_per_tensor must line up element-wise with the RDMA
        # buffer list of get_contiguous_buf_infos, which never ships the ring
        # (27B line: the low parts rawv/rawk were missing from the mirror).
        pool = _spec_pool()
        ptrs = pool.get_contiguous_buf_infos()[0]
        dims = pool.get_state_dim_per_tensor()
        self.assertEqual(len(dims), len(ptrs))
        c = pool.mamba_cache
        want = [c.conv[0].shape[2]] * len(LAYERS) + [c.temporal.shape[2]] * len(LAYERS)
        self.assertEqual(sorted(dims), sorted(want))

    def test_decode_ring_names_stay_state(self):
        self.assertFalse(set(RING) & set(transient_state_fields(False)))
        self.assertTrue(set(RING) <= set(transient_state_fields(True)))
        dec = MambaPool(
            size=6,
            spec_state_size=2,
            cache_params=_params(),
            mamba_layer_ids=LAYERS,
            device="cpu",
            speculative_num_draft_tokens=None,
            speculative_eagle_topk=None,
            enable_linear_replayssm=True,
            linear_replayssm_cache_len=16,
        )
        names = dec.state_blob_fields()
        for n in ("replayssm_d", "replayssm_k", "replayssm_g"):
            self.assertIn(n, names)


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
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
        MambaAttnBackendBase,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    import sglang.srt.layers.attention.hybrid_linear_attn_backend as hlab


    def _ref_scatter(dst, src, dst_idx, steps):
        # CPU stand-in for the CUDA-only fused conv-window scatter
        for i in range(steps.shape[0]):
            s, d = int(steps[i]), int(dst_idx[i])
            if s >= 0 and d >= 0:
                dst[:, d] = src[:, i, s]


    hlab.fused_conv_window_scatter_with_mask = _ref_scatter

    LAYERS = [0, 1]
    H, HV, K, V = 2, 6, 32, 32
    STEPS, L = 4, 16
    ROWS = [2, 4]
    SLOTS = [1, 3]
    TRACK = 5
    CONV_DIM = (2 * H + HV) * K
    RING = ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk")


    def pool(dt):
        shape = Mamba2StateShape(
            conv=[(CONV_DIM, 3)], temporal=(HV, V, K), intermediate_size=HV * V,
            conv_dim=CONV_DIM, ssm_state_size=K, num_heads=HV, head_dim=V,
            state_size=K, conv_kernel=4, num_k_heads_per_tp=H,
        )
        params = Mamba2CacheParams(
            shape=shape, layers=LAYERS, dtype=Mamba2StateDType(conv=dt, temporal=dt)
        )
        return HybridReqToTokenPool(
            size=4, mamba_size=6, mamba_spec_state_size=4, max_context_len=64,
            device="cpu", enable_memory_saver=False, cache_params=params,
            mamba_layer_ids=LAYERS, enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=STEPS, speculative_eagle_topk=1,
            linear_replayssm_cache_len=L, enable_linear_replayssm_spec=True,
        )


    def step(rtp, lay, win, slots, rows, last, track_idx, track_steps):
        bs = slots.shape[0]
        qsl = torch.arange(0, bs * STEPS + 1, STEPS, dtype=torch.int32)
        outs = []
        for li, (layer, x) in enumerate(zip(lay, win)):
            outs.append(
                GDNAttnBackend._replayssm_target_verify(
                    SimpleNamespace(req_to_token_pool=rtp), layer=layer,
                    query=x["q"], key=x["k"], value=x["v"], a=x["a"], b=x["b"],
                    layer_cache=rtp.mamba2_layer_cache(LAYERS[li]),
                    cache_indices=slots, replay_indices=rows,
                    query_start_loc=qsl, draft_token_num=STEPS,
                )
            )
        hyb = object.__new__(HybridLinearAttnBackend)
        hyb.linear_attn_backend = SimpleNamespace(
            forward_metadata=ForwardMetadata(
                query_start_loc=qsl, mamba_cache_indices=slots,
                replayssm_spec_rows=rows,
            ),
            req_to_token_pool=rtp,
        )
        hyb.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last, mamba_track_indices=track_idx,
            mamba_steps_to_track=track_steps, model=None,
        )
        return outs


    def run(dt, tag, res):
        gen = torch.Generator().manual_seed(11)
        clean, dirty = pool(dt), pool(dt)
        c = clean.mamba_pool.mamba_cache
        c.temporal.copy_((0.2 * torch.randn(c.temporal.shape, generator=gen)).to(dt))
        c.conv[0].copy_(torch.randn(c.conv[0].shape, generator=gen).to(dt))
        w = c.intermediate_conv_window[0]
        w.copy_(torch.randn(w.shape, generator=gen).to(dt))
        d = dirty.mamba_pool.mamba_cache
        d.temporal.copy_(c.temporal)
        d.conv[0].copy_(c.conv[0])
        d.intermediate_conv_window[0].copy_(w)
        # the dirty pool's ring and cursors hold garbage, as after a TMS restore
        for name in RING:
            t = getattr(d, name)
            if t is not None:
                t.fill_(float("nan"))
        mp = dirty.mamba_pool
        mp.replayssm_spec_write_pos.fill_(9)
        mp.replayssm_is_flush.fill_(1)
        mp.replayssm_cache_base.fill_(12345)
        lay = [
            SimpleNamespace(
                num_k_heads=H, head_k_dim=K, num_v_heads=HV, head_v_dim=V,
                A_log=(0.1 * torch.randn(HV, generator=gen)).float(),
                dt_bias=(0.1 * torch.randn(HV, generator=gen)).float(),
            )
            for _ in LAYERS
        ]
        rows = torch.tensor(ROWS, dtype=torch.int64)
        slots = torch.tensor(SLOTS, dtype=torch.int32)
        T = 2 * STEPS
        all_equal, all_finite = True, True
        for n, (last, track) in enumerate(
            (([2, 0], [1, -1]), ([3, 1], None), ([0, 3], [-1, 2]))
        ):
            win = [
                dict(
                    q=torch.randn(1, T, H, K, generator=gen).to(dt),
                    k=torch.randn(1, T, H, K, generator=gen).to(dt),
                    v=(0.5 * torch.randn(1, T, HV, V, generator=gen)).to(dt),
                    a=torch.randn(T, HV, generator=gen).to(dt),
                    b=torch.randn(T, HV, generator=gen).to(dt),
                )
                for _ in LAYERS
            ]
            # the verify metadata's heal, for both pools (a no-op on the clean one)
            for rtp in (clean, dirty):
                MambaAttnBackendBase._replayssm_spec_rows(
                    SimpleNamespace(req_to_token_pool=rtp), rows, heal=True
                )
            last_t = torch.tensor(last, dtype=torch.int64)
            ti = None if track is None else torch.tensor([TRACK, TRACK], dtype=torch.int64)
            ts = None if track is None else torch.tensor(track, dtype=torch.int64)
            o_c = step(clean, lay, win, slots, rows, last_t, ti, ts)
            o_d = step(dirty, lay, win, slots, rows, last_t, ti, ts)
            all_equal &= all(torch.equal(a, b) for a, b in zip(o_c, o_d))
            all_equal &= torch.equal(c.temporal, d.temporal)
            all_equal &= torch.equal(c.conv[0], d.conv[0])
            all_finite &= bool(torch.isfinite(d.temporal.float()).all())
            all_finite &= all(bool(torch.isfinite(o.float()).all()) for o in o_d)
            res[f"{tag}_step{n}_equal"] = bool(all_equal)
        res[f"{tag}_equal"] = bool(all_equal)
        res[f"{tag}_finite"] = bool(all_finite)
        # lanes past the windows still hold the garbage (nothing cleaned them)
        res[f"{tag}_garbage_left"] = bool(torch.isnan(d.replayssm_d.float()).any())


    res = {}
    run(torch.float32, "fp32", res)
    run(torch.float16, "b16", res)
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


class TestGarbageRingIsHarmless(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = _probe()

    def test_bit_identical_to_a_clean_ring(self):
        for tag in ("fp32", "b16"):
            for n in range(3):
                self.assertTrue(self.res[f"{tag}_step{n}_equal"], (tag, n))
            self.assertTrue(self.res[f"{tag}_finite"], tag)
            # the proof is not vacuous: NaN still sits in lanes never rewritten
            self.assertTrue(self.res[f"{tag}_garbage_left"], tag)


if __name__ == "__main__":
    unittest.main()
