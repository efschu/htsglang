"""H68b METAL test (runs only with a CUDA device; skipped on the desk):
the NVFP4 fused-MoE Marlin kernel, fed by ``prepare_moe_nvfp4_layer_for_marlin_inplace``,
against a torch reference that dequantizes the CHECKPOINT-format tensors.

This is the first run of ``fused_marlin_moe`` with ``float4_e2m1f`` on this rig
(the dense Marlin-FP4 GEMM ran 31.07. in the phi0 microbench,
INTEGRATION_R3_VALIDATION.md:15960-15981; the MoE kernel never did). Real
Qwen3.8-Flash-Next expert geometry (hidden 2560, moe intermediate 640, group
16, gated SiLU), 8 experts, top-2 routing, 32 tokens.

Reference: E2M1 code table x E4M3 block scale x F32 global scale -> fp32
weights; y = sum_k w_k * W2_e (silu(W1_e x) * W3_e x) in fp32 from the same
bf16 input. Pass: cosine >= 0.999 and relative L2 error <= 2e-2 (bf16 math,
fp32 accumulation). One greppable line per run:
``NVFP4-MARLIN-MOE-METAL device=<name> cc=<x.y> cos=<> rel=<> PASS|FAIL``.

Run in a GPU window only (CUDA_VISIBLE_DEVICES=<card>, never empty):
    pytest -q test/registered/unit/weg2/test_weg2_nvfp4_marlin_moe_metal_h68b.py
"""

import types
import unittest

import pytest

try:
    import torch
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

# desk gate: the file imports and skips; the metal class runs in a GPU window
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, H, I, TOPK, M, G = 8, 2560, 640, 2, 32, 16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _dequant(packed_u8, block_scale_fp8, global_scale):
    """[N, K/2] uint8 (low nibble first), [N, K/16] e4m3, scalar -> [N, K] fp32."""
    lo = (packed_u8 & 0x0F).long()
    hi = (packed_u8 >> 4).long()
    codes = torch.stack((lo, hi), dim=-1).reshape(packed_u8.shape[0], -1)
    vals = E2M1.to(packed_u8.device)[codes]
    scale = block_scale_fp8.float().repeat_interleave(G, dim=1)
    return vals * scale * float(global_scale)


@unittest.skipUnless(torch.cuda.is_available(), "metal test: needs a CUDA device")
class MetalTest(unittest.TestCase):
    def test_marlin_nvfp4_moe_matches_the_dequant_reference(self):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            nvfp4_marlin_global_scale_1d,
            prepare_moe_nvfp4_layer_for_marlin_inplace,
        )

        dev = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device="cpu").manual_seed(68)
        w13 = torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, dtype=torch.uint8)
        w2 = torch.randint(0, 256, (E, H, I // 2), generator=g, dtype=torch.uint8)
        # E4M3 block scales in a checkpoint-like range, global scales ~1e-4
        s13 = (torch.rand(E, 2 * I, H // G, generator=g) * 3 + 0.5).to(torch.float8_e4m3fn)
        s2 = (torch.rand(E, H, I // G, generator=g) * 3 + 0.5).to(torch.float8_e4m3fn)
        gs13 = torch.rand(E, generator=g) * 1e-4 + 5e-5
        gs2 = torch.rand(E, generator=g) * 1e-4 + 5e-5
        x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16)
        logits = torch.randn(M, E, generator=g)
        topk_w, topk_ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)

        # --- reference (fp32) from the CHECKPOINT-format tensors
        xf = x.float()
        ref = torch.zeros(M, H)
        for e in range(E):
            w1 = _dequant(w13[e, :I], s13[e, :I], gs13[e])
            w3 = _dequant(w13[e, I:], s13[e, I:], gs13[e])
            w2e = _dequant(w2[e], s2[e], gs2[e])
            hdn = torch.nn.functional.silu(xf @ w1.T) * (xf @ w3.T)
            out_e = hdn @ w2e.T
            for k in range(TOPK):
                sel = topk_ids[:, k] == e
                ref[sel] += topk_w[sel, k : k + 1] * out_e[sel]

        # --- the layer as the ModelOpt method leaves it before the repack
        layer = torch.nn.Module()
        layer.quant_config = types.SimpleNamespace(group_size=G)
        layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
        layer.intermediate_size_per_partition = I
        layer.params_dtype = torch.bfloat16

        def P(t):
            return torch.nn.Parameter(t.to(dev), requires_grad=False)

        layer.w13_weight, layer.w2_weight = P(w13), P(w2)
        layer.w13_weight_scale, layer.w2_weight_scale = P(s13), P(s2)
        layer.w13_weight_scale_2, layer.w2_weight_scale_2 = P(gs13), P(gs2)
        prepare_moe_nvfp4_layer_for_marlin_inplace(layer)

        out = fused_marlin_moe(
            x.to(dev), layer.w13_weight, layer.w2_weight,
            layer.w13_weight_scale, layer.w2_weight_scale,
            logits.to(dev), topk_w.to(dev), topk_ids.to(dev).to(torch.int32),
            w1_global_scale=nvfp4_marlin_global_scale_1d(layer.w13_weight_scale_2),
            w2_global_scale=nvfp4_marlin_global_scale_1d(layer.w2_weight_scale_2),
            workspace=layer.workspace, num_bits=4, is_k_full=True,
        ).float().cpu()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        rel = ((out - ref).norm() / ref.norm()).item()
        ok = cos >= 0.999 and rel <= 2e-2 and bool(torch.isfinite(out).all())
        cc = ".".join(map(str, torch.cuda.get_device_capability(dev)))
        print(f"NVFP4-MARLIN-MOE-METAL device={torch.cuda.get_device_name(dev)!r} cc={cc} "
              f"cos={cos:.6f} rel={rel:.4e} {'PASS' if ok else 'FAIL'}")
        assert ok, (cos, rel)
