#!/usr/bin/env python3
"""N4C GPU micro test (#38 L8 + F's sm_12x hook), one card per process.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvml> python n4c_micro.py <out.json> [--sm12x]

On the card: for every 27B NVFP4 shape
  * repack round trip: in-place native->Marlin (gptq_marlin_repack) -> native, byte-equal;
    the GPU repack byte-equal to the pure-torch reference, band by band;
  * to_marlin / to_native ms (warm, second pass) and the peak working set
    (max_memory_allocated - allocated before), bound 32 MiB;
  * banded apply vs the unbanded Marlin tensor (the loader's own expressions):
    bit-equality and CUDA-graph time at M = 1, 16, 512.
--sm12x (5090 only): F's maybe_apply_sm12x_w4a16 (cute-dsl-native W4A16) vs the
native W4A4 path at M = 1, 8 on the D shards of the 5090, error vs an fp32
dequant reference and CUDA-graph time.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

dev = torch.device("cuda:0")
MIB = 1 << 20
OUT = sys.argv[1]
DO_SM12X = "--sm12x" in sys.argv
res = {"device": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability(0)),
       "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shapes": [], "sm12x": []}


def dump():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack  # noqa: E402
from sglang.srt.layers.quantization import nvfp4_marlin_inplace as mi  # noqa: E402
from sglang.srt.layers.quantization import nvfp4_native_mixed as nm  # noqa: E402
from sglang.srt.layers.quantization.marlin_utils import (  # noqa: E402
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (  # noqa: E402
    apply_fp4_marlin_linear,
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)

SHAPES = [
    ("D.gate_up.3080", 8192, 5120, True),
    ("D.down.3080", 5120, 4096, True),
    ("D.gate_up.5090", 18688, 5120, True),
    ("D.down.5090", 5120, 9344, True),
    ("D.lm_head", 82816, 5120, True),
    ("P.gate_up", 34816, 5120, True),
    ("P.down", 5120, 17408, True),
    ("P.lm_head", 248320, 5120, False),  # unbanded copy skipped (memory share)
]
GS = 0.0025


def sync():
    torch.cuda.synchronize()


def graph_us(fn, iters=50):
    for _ in range(3):
        fn()
    sync()
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            fn()
        run, mode = g.replay, "graph"
    except Exception as e:  # noqa: BLE001
        run, mode = fn, f"eager ({type(e).__name__})"
    for _ in range(3):
        run()
    sync()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        run()
    b.record()
    sync()
    return a.elapsed_time(b) * 1000 / iters, mode


class L(torch.nn.Module):
    pass


def make_layer(n, k, g):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    raw = torch.randint(0x28, 0x48, (n, k // 16), dtype=torch.uint8, device=dev, generator=g)
    sw = nm.swizzle_128x4(raw).contiguous()
    layer = L()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sw.view(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_interleaved = layer.weight_scale
    layer.params_dtype = torch.bfloat16
    layer.output_size_per_partition = n
    layer.weight_global_scale = torch.nn.Parameter(torch.tensor(GS, device=dev), requires_grad=False)
    mi.bind_global_scale(layer, torch.bfloat16)
    return layer, raw


def main():
    g = torch.Generator(device=dev).manual_seed(38)
    # full-range scale round trip on the GPU (the CPU test covers it too)
    raw = torch.arange(0, 256 * 64, device=dev).remainder(0x7F).to(torch.uint8).reshape(256, 64)
    sw = nm.swizzle_128x4(raw).view(torch.float8_e4m3fn).contiguous()
    b = sw.clone()
    mi.scale_band_to_marlin_(b, 1024, torch.bfloat16)
    mi.scale_band_to_native_(b, 1024)
    res["scale_fullrange_roundtrip_equal"] = bool(torch.equal(b.view(torch.uint8), sw.view(torch.uint8)))
    # warm the JIT (repack + Marlin GEMM) outside the timings
    lw, _ = make_layer(256, 512, g)
    mi.prepare_layer(lw)
    mi.apply(lw, torch.randn(4, 512, dtype=torch.bfloat16, device=dev))
    sync()
    del lw
    for name, n, k, unbanded in SHAPES:
        row = {"shape": name, "N": n, "K": k}
        try:
            layer, raw = make_layer(n, k, g)
            w0 = layer.weight.detach().clone()
            s0 = layer.weight_scale.detach().view(torch.uint8).clone()
            bands = mi.band_table(n, k)
            row["bands"] = len(bands)
            setattr(layer, mi.BANDS_ATTR, bands)
            setattr(layer, mi.LAYER_FLAG, True)
            layer.workspace = marlin_make_workspace(dev)
            # GPU repack vs pure-torch reference, band by band (on a copy)
            ref_eq = True
            for r0, r1 in bands:
                a = w0[r0:r1].clone()
                bb = w0[r0:r1].clone()
                mi.weight_band_to_marlin_(a, k, mi._gpu_repack)
                mi.weight_band_to_marlin_(bb, k, mi._ref_repack)
                ref_eq &= bool(torch.equal(a, bb))
                del a, bb
            row["gpu_repack_equals_reference"] = ref_eq
            times = []
            for rep in range(2):
                sync()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                mi.layer_to_marlin(layer, repack=mi._gpu_repack)
                sync()
                t1 = time.perf_counter()
                pk_m = torch.cuda.max_memory_allocated() - base
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t2 = time.perf_counter()
                mi.layer_to_native(layer)
                sync()
                t3 = time.perf_counter()
                pk_n = torch.cuda.max_memory_allocated() - base
                times.append(((t1 - t0) * 1e3, (t3 - t2) * 1e3, pk_m / MIB, pk_n / MIB))
            row["to_marlin_ms"], row["to_native_ms"], row["peak_to_marlin_mib"], row["peak_to_native_mib"] = [
                round(v, 2) for v in times[1]]
            row["first_pass_ms"] = [round(times[0][0], 2), round(times[0][1], 2)]
            row["roundtrip_equal"] = bool(torch.equal(layer.weight, w0)) and bool(
                torch.equal(layer.weight_scale.view(torch.uint8), s0))
            row["bytes_mib"] = round((n * k // 2 + n * k // 16) / MIB, 1)
            mi.layer_to_marlin(layer, repack=mi._gpu_repack)
            if unbanded:
                qw = gptq_marlin_repack(w0.view(torch.int32).T.contiguous(), torch.empty(0, dtype=torch.int, device=dev),
                                        k, n, 4)
                sc = raw.view(torch.float8_e4m3fn).T.contiguous().to(torch.bfloat16)
                sc = nvfp4_marlin_process_scales(marlin_permute_scales(s=sc, size_k=k, size_n=n, group_size=16))
                gsc = nvfp4_marlin_process_global_scale(torch.tensor(GS, device=dev).to(torch.bfloat16)).reshape(1)
                ws = marlin_make_workspace(dev)
                del w0, s0
                applies = []
                for M in (1, 16, 512):
                    x = torch.randn(M, k, dtype=torch.bfloat16, device=dev, generator=g)
                    full = lambda x=x: apply_fp4_marlin_linear(input=x, weight=qw, weight_scale=sc,  # noqa: E731
                                                               weight_global_scale=gsc, workspace=ws,
                                                               size_n=n, size_k=k, bias=None)
                    band = lambda x=x: mi.apply(layer, x)  # noqa: E731
                    o_f, o_b = full(), band()
                    sync()
                    d = (o_f.float() - o_b.float()).abs()
                    us_f, mode_f = graph_us(full)
                    us_b, mode_b = graph_us(band)
                    applies.append({"M": M, "bitequal": bool(torch.equal(o_f, o_b)),
                                    "max_abs_diff": float(d.max()), "ref_absmax": float(o_f.float().abs().max()),
                                    "us_unbanded": round(us_f, 1), "us_banded": round(us_b, 1),
                                    "mode": f"{mode_f}/{mode_b}"})
                row["apply"] = applies
                del qw, sc
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        res["shapes"].append(row)
        print(json.dumps(row), flush=True)
        dump()
        torch.cuda.empty_cache()
    if DO_SM12X:
        sm12x()
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()


def sm12x():
    from sglang.srt.layers.quantization import fp4_utils
    from sglang.srt.layers.quantization import nvfp4_sm12x_w4a16 as f
    from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp4LinearMethod

    fp4_utils.FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.CUTLASS
    fp4_utils.FP4_NATIVE_MIXED = True
    fp4_utils.FP4_NATIVE_MIXED_SHARED_LAYOUT = True
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    g = torch.Generator(device=dev).manual_seed(5)
    for name, n, k in (("D.gate_up.5090", 18688, 5120), ("D.down.5090", 5120, 9344)):
        row = {"shape": name}
        try:
            method = ModelOptFp4LinearMethod(ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))
            layer = torch.nn.Module()
            method.create_weights(layer, k, [n], k, n, torch.bfloat16, weight_loader=None)
            layer.to(dev)
            layer.weight.data.copy_(torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, device=dev, generator=g))
            layer.weight_scale.data.copy_(torch.randint(0x28, 0x48, layer.weight_scale.shape, dtype=torch.uint8,
                                                        device=dev, generator=g).view(torch.float8_e4m3fn))
            layer.input_scale.data.fill_(4.0 / (448 * 6))
            layer.weight_scale_2.data.fill_(GS)
            raw = layer.weight_scale.detach().view(torch.uint8).clone()
            wcodes = layer.weight.detach().clone()
            method.process_weights_after_loading(layer)
            codes = torch.stack((wcodes & 0xF, wcodes >> 4), -1).reshape(n, k).long()
            W = lut[codes] * raw.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * GS
            for M in (1, 8):
                x = torch.randn(M, k, dtype=torch.bfloat16, device=dev, generator=g)
                ref = x.float() @ W.t()
                os.environ[f.MAX_M_ENV] = "0"
                f._reset_for_tests()
                a4 = lambda x=x: method.apply(layer, x)  # noqa: E731
                o4 = a4()
                us4, m4 = graph_us(a4)
                os.environ[f.MAX_M_ENV] = "16"
                f._reset_for_tests()
                a16 = lambda x=x: method.apply(layer, x)  # noqa: E731
                o16 = a16()
                took = f.maybe_apply_sm12x_w4a16(layer, x, None, "cutlass") is not None
                us16, m16 = graph_us(a16)
                rel = lambda o: float((o.float() - ref).norm() / ref.norm())  # noqa: E731
                row[f"M{M}"] = {"w4a4_us": round(us4, 1), "w4a16_us": round(us16, 1), "w4a16_taken": took,
                                "rel_err_w4a4": round(rel(o4), 5), "rel_err_w4a16": round(rel(o16), 5),
                                "mode": f"{m4}/{m16}"}
            os.environ[f.MAX_M_ENV] = "0"
            f._reset_for_tests()
            del W
        except Exception as e:  # noqa: BLE001
            import traceback

            row["error"] = f"{type(e).__name__}: {e}"
            row["tb"] = traceback.format_exc()[-1500:]
        res["sm12x"].append(row)
        print(json.dumps(row), flush=True)
        dump()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
