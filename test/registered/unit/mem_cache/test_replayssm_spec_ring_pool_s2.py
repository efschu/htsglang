"""The ReplaySSM spec ring's pool half: flag, allocation, cursors, resets (S2).

27B ReplaySSM package, slice S2 (REPLAYSSM_PLAN.md). Under
``--enable-linear-replayssm-spec`` the GDN target verify's per-draft
intermediate state (``SpeculativeState.intermediate_ssm``) is not allocated;
a compact ring keyed by REQUEST row replaces it:

    d, rawv  [layers, rows, HV, L, V]   activation dtype (rawv: 16-bit only)
    k, rawk  [layers, rows, H,  L, K]   activation dtype (rawk: 16-bit only)
    g        [layers, rows, HV, L]      fp32
    cursors  write_pos / cache_base / is_flush  [rows]

with rows = spec_state_size + 1 (like the intermediate state) and HV/H the
rank's OWN heads (uneven GDN TP). Pinned here (CPU, real MambaPool /
HybridReqToTokenPool):

* flag off: byte-identical pool (intermediate present, no ring, no cursors);
* flag on: intermediate absent, ring shapes/dtypes for two uneven ranks,
  low parts only for 16-bit activations, the conv verify windows stay;
* runtime refusals: ring shorter than the widest verify window, KDA, decode
  ring together with the spec ring;
* fresh request rows start with an empty ring (HybridReqToTokenPool.alloc),
  reset_state() zeroes rings and cursors with no intermediate state present, and
  the RDMA/transfer buffer list never carries ring tensors;
* the not-yet-wired intermediate readers refuse loudly instead of scattering
  a None (S3 replaces the GDN ones with the ring route).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYERS = [0, 1, 2]
V = K = 16  # head dims (the ring is dim-agnostic; small keeps the test cheap)


def _params(hv, hk, conv_dtype=torch.bfloat16, kda=False):
    shape = Mamba2StateShape(
        conv=[((2 * hk + hv) * K, 3)],
        temporal=(hv, V, K),
        intermediate_size=hv * V,
        conv_dim=(2 * hk + hv) * K,
        ssm_state_size=K,
        num_heads=hv,
        head_dim=V,
        state_size=K,
        conv_kernel=4,
        num_k_heads_per_tp=hk,
    )
    cls = Mamba2CacheParams
    if kda:
        cls = type("KdaParams", (Mamba2CacheParams,), {"is_kda": property(lambda s: True)})
    return cls(
        shape=shape,
        layers=LAYERS,
        dtype=Mamba2StateDType(conv=conv_dtype, temporal=torch.bfloat16),
    )


def _pool(spec, hv=6, hk=2, conv_dtype=torch.bfloat16, draft=4, ring=16, **kw):
    return MambaPool(
        size=4,
        spec_state_size=2,
        cache_params=kw.pop("params", None) or _params(hv, hk, conv_dtype),
        mamba_layer_ids=LAYERS,
        device="cpu",
        speculative_num_draft_tokens=draft,
        speculative_eagle_topk=1,
        linear_replayssm_cache_len=ring,
        enable_linear_replayssm_spec=spec,
        **kw,
    )


class _Base(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestAllocation(_Base):
    def test_flag_off_is_the_old_pool(self):
        pool = _pool(False)
        c = pool.mamba_cache
        self.assertEqual(tuple(c.intermediate_ssm.shape), (3, 3, 4, 6, V, K))
        for name in ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk"):
            self.assertIsNone(getattr(c, name))
        self.assertIsNone(pool.replayssm_spec_write_pos)
        self.assertIsNone(pool.replayssm_cache_base)
        self.assertIsNone(pool.replayssm_is_flush)

    def test_flag_on_uneven_ranks(self):
        for hv, hk in ((18, 6), (15, 5)):  # the 27B D ranks' GDN heads
            with self.subTest(hv=hv, hk=hk):
                pool = _pool(True, hv=hv, hk=hk)
                c = pool.mamba_cache
                self.assertIsNone(c.intermediate_ssm)
                rows = 3  # spec_state_size + 1
                self.assertEqual(tuple(c.replayssm_d.shape), (3, rows, hv, 16, V))
                self.assertEqual(tuple(c.replayssm_k.shape), (3, rows, hk, 16, K))
                self.assertEqual(tuple(c.replayssm_g.shape), (3, rows, hv, 16))
                self.assertEqual(c.replayssm_d.dtype, torch.bfloat16)
                self.assertEqual(c.replayssm_g.dtype, torch.float32)
                self.assertEqual(c.replayssm_rawv.shape, c.replayssm_d.shape)
                self.assertEqual(c.replayssm_rawk.shape, c.replayssm_k.shape)
                self.assertEqual(tuple(pool.replayssm_spec_write_pos.shape), (rows,))
                self.assertEqual(pool.replayssm_is_flush.dtype, torch.int8)
                # the conv verify windows (conv rollback) stay
                self.assertEqual(len(c.intermediate_conv_window), 1)
                # the priced ring equals the allocated ring
                ring_bytes = sum(
                    t.numel() * t.element_size()
                    for t in (c.replayssm_d, c.replayssm_k, c.replayssm_g, c.replayssm_rawv, c.replayssm_rawk)
                )
                per_req = _params(hv, hk).replayssm_ring_bytes_per_req(record_len=16)
                self.assertEqual(ring_bytes, per_req * rows)

    def test_fp32_activations_have_no_low_parts(self):
        c = _pool(True, conv_dtype=torch.float32).mamba_cache
        self.assertEqual(c.replayssm_d.dtype, torch.float32)
        self.assertIsNone(c.replayssm_rawv)
        self.assertIsNone(c.replayssm_rawk)

    def test_runtime_refusals(self):
        with self.assertRaisesRegex(ValueError, "shorter than the widest verify window"):
            _pool(True, draft=32, ring=16)
        with self.assertRaisesRegex(ValueError, "must be >= 16"):
            _pool(True, draft=4, ring=8)
        with self.assertRaisesRegex(ValueError, "KDA"):
            _pool(True, params=_params(6, 2, kda=True))
        with self.assertRaisesRegex(ValueError, "exclusive"):
            _pool(True, enable_linear_replayssm=True)


class TestLifecycle(_Base):
    def test_reset_zeroes_rings_and_cursors(self):
        pool = _pool(True)
        c = pool.mamba_cache
        for name in ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk"):
            getattr(c, name).fill_(3)
        pool.replayssm_spec_write_pos.fill_(5)
        pool.replayssm_cache_base.fill_(7)
        pool.replayssm_is_flush.fill_(1)
        pool.reset_state()
        for name in ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk"):
            self.assertTrue(torch.all(getattr(c, name) == 0))
        for t in (pool.replayssm_spec_write_pos, pool.replayssm_cache_base, pool.replayssm_is_flush):
            self.assertTrue(torch.all(t == 0))

    def test_transfer_buffers_exclude_the_ring(self):
        pool = _pool(True)
        ptrs = pool.get_contiguous_buf_infos()[0]
        ring = {
            getattr(pool.mamba_cache, n).data_ptr()
            for n in ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk")
        }
        self.assertFalse(ring & set(ptrs))

    def test_fresh_request_rows_start_with_an_empty_ring(self):
        rtp = HybridReqToTokenPool(
            size=2,
            mamba_size=4,
            mamba_spec_state_size=2,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
            cache_params=_params(6, 2),
            mamba_layer_ids=LAYERS,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            linear_replayssm_cache_len=16,
            enable_linear_replayssm_spec=True,
        )
        pool = rtp.mamba_pool
        pool.replayssm_spec_write_pos.fill_(6)
        pool.replayssm_cache_base.fill_(6)
        pool.replayssm_is_flush.fill_(1)
        req = SimpleNamespace(
            rid="r0",
            req_pool_idx=None,
            mamba_pool_idx=torch.tensor(1),
            mamba_cow_src_index=torch.tensor(1),
            mamba_slot_acquired_this_admission=False,
            mamba_ping_pong_track_buffer=None,
        )
        rows = rtp.alloc([req])
        row = rows[0]
        self.assertEqual(int(pool.replayssm_spec_write_pos[row]), 0)
        self.assertEqual(int(pool.replayssm_cache_base[row]), 0)
        self.assertEqual(int(pool.replayssm_is_flush[row]), 0)
        others = [i for i in range(pool.replayssm_spec_write_pos.numel()) if i != row]
        self.assertTrue(torch.all(pool.replayssm_spec_write_pos[others] == 6))


class TestServerArgsStaticChecks(_Base):
    def args(self, **kw):
        a = ServerArgs(model_path="dummy")
        base = dict(
            enable_linear_replayssm_spec=True,
            enable_linear_replayssm=False,
            speculative_algorithm="DFLASH",
            speculative_eagle_topk=None,
            linear_attn_decode_backend="triton",
            disaggregation_mode="null",
            linear_replayssm_cache_len=16,
            mamba_ssm_dtype="bfloat16",
        )
        base.update(kw)
        for k, v in base.items():
            object.__setattr__(a, k, v)
        return a

    def test_the_27b_d_form_passes(self):
        self.args()._handle_linear_attn_backend()

    def test_refusals(self):
        cases = {
            "exclusive": dict(enable_linear_replayssm=True),
            "needs a speculative algorithm": dict(speculative_algorithm=None),
            "linear draft chain": dict(speculative_eagle_topk=2),
            "power of": dict(linear_replayssm_cache_len=12),
            ">= 16": dict(linear_replayssm_cache_len=8),
            "unified-memory": dict(enable_unified_memory=True),
            "PD disaggregation": dict(disaggregation_mode="decode"),
        }
        for msg, kw in cases.items():
            with self.subTest(case=msg):
                with self.assertRaisesRegex(ValueError, msg):
                    self.args(**kw)._handle_linear_attn_backend()


class TestUnwiredReadersRefuse(_Base):
    def test_intermediate_scatter_refuses_a_missing_intermediate(self):
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            scatter_mamba_states_after_mtp_verify,
        )

        with self.assertRaisesRegex(RuntimeError, "ReplaySSM spec ring"):
            scatter_mamba_states_after_mtp_verify(
                SimpleNamespace(temporal=torch.zeros(1), intermediate_ssm=None),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int64),
                None,
                None,
            )


if __name__ == "__main__":
    unittest.main()
