"""Microbench on ONE card (3080 = sm_86, 5090 = sm_120): Qwen3.8-27B 8-bit linears (mlp gate_up / down), FP8 against INT8.

Written 27.10. after the 3080s' BF16 rate doubled (gpu-unlock: BF16/FP16 x2.02, INT8 flat): which 8-bit form is faster
on the card now, and what does it cost in error? All four arms see the same [N, K] and the same bf16 activations:

(c) INT8 W8A8 -- today's INT8 standard model (per_token_quant_int8 + sgl_kernel.int8_scaled_mm)
(e) FP8  W8A16 -- weight-only FP8 through Marlin (prepare_fp8_layer_for_marlin + apply_fp8_marlin_linear), the form a
    card without FP8 tensor cores runs; its math is on the BF16 tensor cores
(d) cuBLAS bf16 x bf16 (torch.mm), the dense reference
(f) FP8  W8A8 native (torch._scaled_mm), only where the card has FP8 tensor cores (sm_89+: the 5090)

Times: CUDA events around back-to-back calls, median of 5 repetitions (same timer as bench_nvfp4_w4a8_sm86.py).
``--quality``: relative Frobenius error of (c), (e), (f) against the fp32 product with the UNquantised weights, on
Gaussian weights and on activations with a 1 % x20 outlier channel set. Every figure is a measurement on the card it ran on.
Run inside the image (sgl_kernel, the Marlin ops): ``python3 -m flliper.jit_kernel.benchmark.bench_w8_fp8_vs_int8 --gpu N``.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # NVML order == CUDA order on this rig

import torch  # noqa: E402

from flliper.jit_kernel.benchmark.bench_nvfp4_w4a8_sm86 import time_fn  # noqa: E402

SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408)}  # (N, K) of the 27B MLP (hidden 5120, intermediate 17408)


def quantise_int8_rows(w):
    s = w.abs().amax(dim=1, keepdim=True).float() / 127.0
    return torch.clamp((w.float() / s).round(), -127, 127).to(torch.int8), s


def quantise_fp8_rows(w):
    s = w.abs().amax(dim=1, keepdim=True).float() / 448.0
    return (w.float() / s).to(torch.float8_e4m3fn), s


def make_marlin_fp8(w8, s, n, k):
    from flliper.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w8.clone(), requires_grad=False)  # [N, K], size_k_first=False
    layer.weight_scale = torch.nn.Parameter(s.reshape(-1).clone(), requires_grad=False)
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = torch.bfloat16
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


def arm_fns(x, w, n, k, native_fp8):
    """name -> callable (activation fixed), plus the quantised operands kept alive by the closures."""
    from sgl_kernel import int8_scaled_mm

    from flliper.srt.layers.quantization.int8_kernel import per_token_quant_int8
    from flliper.srt.layers.quantization.marlin_utils_fp8 import apply_fp8_marlin_linear

    arms = {}
    w8i, si = quantise_int8_rows(w)
    w8t, wsi = w8i.t(), si.reshape(-1, 1).contiguous()
    xq, xs = per_token_quant_int8(x)
    arms["c_int8_w8a8_quant"] = lambda: per_token_quant_int8(x)
    arms["c_int8_w8a8_gemm"] = lambda: int8_scaled_mm(xq, w8t, xs, wsi, out_dtype=torch.bfloat16, bias=None)

    def _int8_total():
        q, sc = per_token_quant_int8(x)
        return int8_scaled_mm(q, w8t, sc, wsi, out_dtype=torch.bfloat16, bias=None)

    arms["c_int8_w8a8_total"] = _int8_total
    w8f, sf = quantise_fp8_rows(w)
    ml = make_marlin_fp8(w8f, sf, n, k)
    arms["e_fp8_w8a16_marlin"] = lambda: apply_fp8_marlin_linear(
        input=x, weight=ml.weight, weight_scale=ml.weight_scale, workspace=ml.workspace, size_n=n, size_k=k, bias=None)
    wt = w.t().contiguous()
    arms["d_bf16_cublas"] = lambda: torch.mm(x, wt)
    if native_fp8:
        xf = x.to(torch.float8_e4m3fn)
        one = torch.ones((), device=x.device, dtype=torch.float32)
        wf_t = w8f.t()
        arms["f_fp8_w8a8_native"] = lambda: torch._scaled_mm(xf, wf_t, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    return arms, (w8i, si, w8f, sf)


def quality(w, n, k):
    """Relative Frobenius error of the 8-bit forms against x_bf16 @ w^T in fp32, for two activation models."""
    from sgl_kernel import int8_scaled_mm

    from flliper.srt.layers.quantization.int8_kernel import per_token_quant_int8
    from flliper.srt.layers.quantization.marlin_utils_fp8 import apply_fp8_marlin_linear

    g = torch.Generator(device="cuda").manual_seed(11)
    w8i, si = quantise_int8_rows(w)
    w8f, sf = quantise_fp8_rows(w)
    ml = make_marlin_fp8(w8f, sf, n, k)
    out = []
    for model in ("gauss", "outlier1pct_x20"):
        x = torch.randn((64, k), generator=g, device="cuda")
        if model == "outlier1pct_x20":
            idx = torch.randperm(k, generator=g, device="cuda")[: k // 100]
            x[:, idx] *= 20
        x = x.to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        xq, xs = per_token_quant_int8(x)
        yc = int8_scaled_mm(xq, w8i.t(), xs, si.reshape(-1, 1).contiguous(), out_dtype=torch.bfloat16, bias=None).float()
        ye = apply_fp8_marlin_linear(input=x, weight=ml.weight, weight_scale=ml.weight_scale, workspace=ml.workspace,
                                     size_n=n, size_k=k, bias=None).float()
        out.append({"act": model, "int8_w8a8_rel_fro": ((yc - ref).norm() / ref.norm()).item(),
                    "fp8_w8a16_rel_fro": ((ye - ref).norm() / ref.norm()).item()})
        print(json.dumps({k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in out[-1].items()}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ms", default="1,8,16,48,512,4096")
    ap.add_argument("--layers", default="gate_up,down")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    props = torch.cuda.get_device_properties(a.gpu)
    native_fp8 = (props.major, props.minor) >= (8, 9)
    print(f"# card {a.gpu} {props.name} sm_{props.major}{props.minor} native_fp8={native_fp8}", flush=True)
    rows = []
    for lname in a.layers.split(","):
        n, k = SHAPES[lname]
        torch.manual_seed(5)
        w = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
        if a.quality:
            rows.append({"layer": lname, "quality": quality(w, n, k)})
            continue
        for m in (int(v) for v in a.ms.split(",")):
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            arms, _keep = arm_fns(x, w, n, k, native_fp8)
            r = {"layer": lname, "M": m, "N": n, "K": k}
            flops = 2.0 * m * n * k
            for name, fn in arms.items():
                us = time_fn(fn)
                r[name + "_us"] = round(us, 2)
                if not name.endswith(("_quant",)):
                    r[name + "_tops"] = round(flops / (us * 1e-6) / 1e12, 2)
            rows.append(r)
            print(json.dumps(r), flush=True)
            del arms, _keep
            torch.cuda.empty_cache()
        del w
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"card": a.gpu, "name": props.name, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
