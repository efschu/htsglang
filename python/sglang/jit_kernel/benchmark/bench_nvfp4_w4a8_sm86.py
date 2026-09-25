"""Microbench on ONE RTX 3080 (sm_86): Qwen3.8-27B NVFP4 linears (mlp gate_up / down, lm_head).

(a) W4A8-g16 on the INT8 tensor cores, native NVFP4 layout (this tree: sglang.jit_kernel.nvfp4_w4a8)
(b) today's Marlin FP4 W4A16 (--fp4-gemm-backend marlin: prepare_nvfp4_layer_for_marlin + apply_fp4_marlin_linear)
(c) today's INT8 W8A8 of the INT8 standard model (CompressedTensorsW8A8Int8: per_token_quant_int8 + int8_scaled_mm)
(d) cuBLAS bf16 x bf16 -> bf16 with FP32 accumulation (torch.mm), the dense reference

(a)/(b) run on the REAL layer-0 / lm_head NVFP4 tensors of the RadixArk checkpoint; (c) on random int8 weights of
the same [N, K] (its speed does not depend on the values). Times: CUDA events around back-to-back calls, median
of 5 repetitions. Every figure is a measurement on this card; nothing is extrapolated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

# NVML order == CUDA order on this rig (CUDA's default is fastest-first: index 0 would be the 5090).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch  # noqa: E402

CKPT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk"
PEAK_INT8_TOPS = 238.0
PEAK_BW_GBS = 760.0


def load_nvfp4(names):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))[
        "weight_map"
    ]
    ws, scs, ws2 = [], [], []
    for base in names:
        t = {}
        for s in ("weight", "weight_scale", "weight_scale_2"):
            with safe_open(os.path.join(CKPT, idx[base + s]), "pt") as f:
                t[s] = f.get_tensor(base + s)
        ws.append(t["weight"])
        scs.append(t["weight_scale"])
        ws2.append(t["weight_scale_2"].float().reshape(1))
    return (
        torch.cat(ws).cuda(),
        torch.cat(scs).cuda(),
        torch.cat(ws2).max().reshape(1).cuda(),
    )


def time_fn(fn, reps=5, min_ms=150.0):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    one = max(s.elapsed_time(e), 1e-3)
    iters = max(3, int(min_ms / one))
    res = []
    for _ in range(reps):
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        res.append(s.elapsed_time(e) * 1000.0 / iters)
    return statistics.median(res)


E2M1X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def dequant_fp32(w, sc, ws2):
    lut = torch.tensor(E2M1X2, dtype=torch.float32, device=w.device) * 0.5
    codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(w.shape[0], -1).long()
    vals = lut[codes].view(w.shape[0], -1, 16) * sc.float().unsqueeze(-1)
    return vals.view(w.shape[0], -1) * ws2.float()


def quality(args):
    """W4A8 (this kernel) and W4A16 (Marlin) against the fp32 W4A16 reference x_bf16 @ dequant(W)^T, on the real
    layer-0 tensors, for three activation models. Relative Frobenius error and max abs error / ref rms.
    """
    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_linear
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_nvfp4_layer_for_marlin,
    )
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    torch.backends.cuda.matmul.allow_tf32 = False
    L = "model.language_model.layers.0.mlp."
    out = []
    for lname, names in (
        ("gate_up", [L + "gate_proj.", L + "up_proj."]),
        ("down", [L + "down_proj."]),
    ):
        w, sc, ws2 = load_nvfp4(names)
        n, k = w.shape[0], w.shape[1] * 2
        wd = dequant_fp32(w, sc, ws2)
        wsz = swizzle_blockscale(sc)
        ml = torch.nn.Module()
        ml.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        ml.weight_scale = torch.nn.Parameter(sc.clone(), requires_grad=False)
        ml.weight_global_scale = torch.nn.Parameter(
            ws2.reshape(()).clone(), requires_grad=False
        )
        ml.output_size_per_partition, ml.input_size_per_partition, ml.params_dtype = (
            n,
            k,
            torch.bfloat16,
        )
        prepare_nvfp4_layer_for_marlin(ml)
        g = torch.Generator(device="cuda").manual_seed(11)
        for model in ("gauss", "outlier1pct_x20", "massive4ch_x100"):
            x = torch.randn((64, k), generator=g, device="cuda")
            if model == "outlier1pct_x20":
                idx = torch.randperm(k, generator=g, device="cuda")[: k // 100]
                x[:, idx] *= 20
            elif model == "massive4ch_x100":
                idx = torch.randperm(k, generator=g, device="cuda")[:4]
                x[:, idx] *= 100
            x = x.to(torch.bfloat16)
            ref = x.float() @ wd.t()
            ya = nvfp4_w4a8_linear(x, w, wsz, ws2, n).float()
            yb = apply_fp4_marlin_linear(
                input=x,
                weight=ml.weight,
                weight_scale=ml.weight_scale,
                weight_global_scale=ml.weight_global_scale,
                workspace=ml.workspace,
                size_n=n,
                size_k=k,
            ).float()
            rms = ref.pow(2).mean().sqrt()
            row = {
                "layer": lname,
                "act": model,
                "w4a8_rel_fro": ((ya - ref).norm() / ref.norm()).item(),
                "w4a16_marlin_rel_fro": ((yb - ref).norm() / ref.norm()).item(),
                "w4a8_maxabs_over_rms": ((ya - ref).abs().max() / rms).item(),
                "w4a16_marlin_maxabs_over_rms": ((yb - ref).abs().max() / rms).item(),
            }
            out.append(row)
            print(
                json.dumps(
                    {
                        kk: (round(v, 6) if isinstance(v, float) else v)
                        for kk, v in row.items()
                    }
                ),
                flush=True,
            )
        w = sc = wd = wsz = ml = None  # free before the next layer
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,8,16,48,512,4096")
    ap.add_argument("--layers", default="gate_up,down,lm_head")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip", default="", help="comma list of a,b,c,d to skip")
    ap.add_argument(
        "--configs",
        default="auto,exact",
        help="(a) tile variants: 'auto' (FP32 epilogue, by M), 'exact' (INT64 epilogue, by M), or config ids",
    )
    ap.add_argument("--quality", action="store_true")
    ap.add_argument(
        "--splits",
        default="auto",
        help="'auto' or a comma list of split_k values to sweep for (a)",
    )
    args = ap.parse_args()
    if args.quality:
        return quality(args)
    Ms = [int(v) for v in args.ms.split(",")]
    skip = set(args.skip.split(",")) if args.skip else set()

    from sgl_kernel import int8_scaled_mm

    from sglang.jit_kernel.nvfp4_w4a8 import (
        choose_split_k,
        config_for_m,
        exact_config_for_m,
        nvfp4_w4a8_gemm,
        nvfp4_w4a8_quantize_activation,
    )
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_nvfp4_layer_for_marlin,
    )
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    L = "model.language_model.layers.0.mlp."
    layers = {
        "gate_up": [L + "gate_proj.", L + "up_proj."],
        "down": [L + "down_proj."],
        "lm_head": ["lm_head."],
        # D-phase shards of a 3080 rank (layout contract §2 table, 32 units of 128): row / K slices of the real tensors
        "gate_up_d32": [L + "gate_proj.", L + "up_proj."],
        "down_d32": [L + "down_proj."],
        "lm_head_d": ["lm_head."],
    }
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    assert (props.major, props.minor) == (8, 6), props.name
    sms = props.multi_processor_count
    rows = []
    for lname in args.layers.split(","):
        w, sc, ws2 = load_nvfp4(layers[lname])
        if lname == "gate_up_d32":  # [gate 0:4096 | up 0:4096]
            half = w.shape[0] // 2
            w = torch.cat([w[:4096], w[half : half + 4096]]).contiguous()
            sc = torch.cat([sc[:4096], sc[half : half + 4096]]).contiguous()
        elif lname == "down_d32":  # K columns 0:4096
            w, sc = w[:, :2048].contiguous(), sc[:, :256].contiguous()
        elif lname == "lm_head_d":  # vocab rows 0:82816
            w, sc = w[:82816].contiguous(), sc[:82816].contiguous()
        n, k = w.shape[0], w.shape[1] * 2
        wsz = swizzle_blockscale(
            sc
        )  # native layout (sm_120 branch); no padding for any 27B shape
        mlayer = None
        if "b" not in skip:
            mlayer = torch.nn.Module()
            mlayer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
            mlayer.weight_scale = torch.nn.Parameter(sc.clone(), requires_grad=False)
            mlayer.weight_global_scale = torch.nn.Parameter(
                ws2.reshape(()).clone(), requires_grad=False
            )
            mlayer.output_size_per_partition = n
            mlayer.input_size_per_partition = k
            mlayer.params_dtype = torch.bfloat16
            prepare_nvfp4_layer_for_marlin(mlayer)
        w8 = w8s = None
        if "c" not in skip:
            w8 = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
            w8t = w8.t()  # CompressedTensorsW8A8Int8 keeps weight.t() of [N, K]
            w8s = torch.rand((n, 1), dtype=torch.float32, device="cuda") * 1e-3
        for m in Ms:
            if lname.startswith("lm_head") and m > 48:
                continue  # logits only for the sampled rows
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            flops = 2.0 * m * n * k
            r = {"layer": lname, "M": m, "N": n, "K": k}
            if "a" not in skip:
                xq, xs = nvfp4_w4a8_quantize_activation(x)
                out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                t_q = time_fn(lambda: nvfp4_w4a8_quantize_activation(x))
                r.update(a_quant_us=t_q)
                variants = {}
                for c in args.configs.split(","):
                    cfg = (
                        config_for_m(m)
                        if c == "auto"
                        else exact_config_for_m(m) if c == "exact" else int(c)
                    )
                    splits = (
                        [choose_split_k(m, n, k, sms, cfg)]
                        if args.splits == "auto"
                        else [int(v) for v in args.splits.split(",")]
                    )
                    for split in splits:
                        ws_buf = torch.empty(
                            (max(1, 2 * split * m * n),),
                            dtype=torch.float32,
                            device="cuda",
                        )
                        t = time_fn(
                            lambda: nvfp4_w4a8_gemm(
                                xq,
                                xs,
                                w,
                                wsz,
                                ws2,
                                n,
                                split_k=split,
                                config=cfg,
                                workspace=ws_buf,
                                out=out,
                            )
                        )
                        variants[f"{c}:{cfg}/s{split}"] = t
                        ws_buf = None
                best = min(variants, key=variants.get)
                r.update(a_variants_us=variants, a_best=best, a_gemm_us=variants[best])
                r.update(a_total_us=t_q + r["a_gemm_us"])
            if mlayer is not None:
                t_b = time_fn(
                    lambda: apply_fp4_marlin_linear(
                        input=x,
                        weight=mlayer.weight,
                        weight_scale=mlayer.weight_scale,
                        weight_global_scale=mlayer.weight_global_scale,
                        workspace=mlayer.workspace,
                        size_n=n,
                        size_k=k,
                    )
                )
                r.update(b_us=t_b)
            if w8 is not None:
                xq8, xs8 = per_token_quant_int8(x)
                t_cq = time_fn(lambda: per_token_quant_int8(x))
                t_cg = time_fn(
                    lambda: int8_scaled_mm(
                        xq8, w8t, xs8, w8s, out_dtype=torch.bfloat16, bias=None
                    )
                )
                r.update(c_quant_us=t_cq, c_gemm_us=t_cg, c_total_us=t_cq + t_cg)
            for key in ("a_gemm_us", "a_total_us", "b_us", "c_gemm_us", "c_total_us"):
                if key in r:
                    r[key.replace("_us", "_tops")] = flops / (r[key] * 1e-6) / 1e12
            # bytes that must move at least once (weights + scales + activations in + bf16 out)
            if "a_gemm_us" in r:
                by = n * k // 2 + n * k // 16 + m * k + m * n * 2 + 4 * m
                r["a_gemm_gbs"] = by / (r["a_gemm_us"] * 1e-6) / 1e9
            if "b_us" in r:
                by = n * k // 2 + n * k // 16 + m * k * 2 + m * n * 2
                r["b_gbs"] = by / (r["b_us"] * 1e-6) / 1e9
            if "c_gemm_us" in r:
                by = n * k + n * 4 + m * k + m * n * 2 + 4 * m
                r["c_gemm_gbs"] = by / (r["c_gemm_us"] * 1e-6) / 1e9
            rows.append(r)
            print(
                json.dumps(
                    {
                        kk: (round(v, 2) if isinstance(v, float) else v)
                        for kk, v in r.items()
                    }
                ),
                flush=True,
            )
        w = sc = wsz = mlayer = w8 = w8s = None  # free before the next layer / (d)
        torch.cuda.empty_cache()
        if "d" not in skip:
            # (d) cuBLAS bf16 x bf16 with FP32 accumulation (reduced-precision split-K reduction OFF)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            wb = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            for r in [r for r in rows if r["layer"] == lname]:
                m = r["M"]
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                t_d = time_fn(lambda: torch.mm(x, wb.t()))
                r.update(
                    d_us=t_d,
                    d_tops=2.0 * m * n * k / (t_d * 1e-6) / 1e12,
                    d_gbs=(n * k * 2 + m * k * 2 + m * n * 2) / (t_d * 1e-6) / 1e9,
                )
                print(
                    json.dumps({"layer": lname, "M": m, "d_us": round(t_d, 2)}),
                    flush=True,
                )
            wb = None
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "device": props.name,
                    "sms": sms,
                    "peak_int8_tops": PEAK_INT8_TOPS,
                    "rows": rows,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
