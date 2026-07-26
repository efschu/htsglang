"""#190 stage 4: hammer the FP8 MARLIN linear on one sm8x gpu.

On sm80-sm88 sglang's Fp8LinearMethod sets use_marlin (can_auto_enable_marlin_fp8:
80 <= sm < 89), so an RTX 3080 runs every fp8 linear through
torch.ops.sglang.apply_fp8_marlin_linear -> gptq_marlin_gemm, NOT through the
triton block-fp8 matmul (triton has no fp8e4nv on Ampere at all).

The layer bisect put the first run-to-run divergence exactly on such a linear
(MLP gate_up_proj, bit-identical input hash), so this repeats that call with
identical inputs and counts mismatches.
"""

import argparse
import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)


class FakeLayer(torch.nn.Module):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=689)
    p.add_argument("--ms", type=str, default="")
    p.add_argument("--n", type=int, default=8704)   # gate_up per rank at TP=2
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--fp32-reduce", type=int, default=-1)
    p.add_argument("--zero-ws", type=int, default=0)
    a = p.parse_args()

    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(11)

    layer = FakeLayer()
    layer.output_size_per_partition = a.n
    layer.input_size_per_partition = a.k
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [a.block, a.block]
    # checkpoint layout for block fp8 is [n, k] (size_k_first=False)
    w = (torch.randn(a.n, a.k, generator=g, dtype=torch.float32) * 0.2).to(dev)
    layer.weight = torch.nn.Parameter(w.to(torch.float8_e4m3fn), requires_grad=False)
    ws = (
        torch.rand(
            (a.n + a.block - 1) // a.block, a.k // a.block, generator=g,
            dtype=torch.float32,
        )
        * 0.05
        + 0.01
    ).to(dev)
    layer.weight_scale_inv = torch.nn.Parameter(ws, requires_grad=False)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)

    print(
        f"# gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)} "
        f"N={a.n} K={a.k} iters={a.iters} fp32_reduce={a.fp32_reduce}"
    )
    print(f"{'M':>6} {'bad/iters':>12} {'rate':>9} {'firstbad':>9} {'worst':>11}")

    kw = {} if a.fp32_reduce < 0 else {"use_fp32_reduce": bool(a.fp32_reduce)}
    ms = [int(v) for v in a.ms.split(",")] if a.ms else [a.m]
    for M in ms:
        x = (
            (torch.randn(M, a.k, generator=g, dtype=torch.float32) * 0.5)
            .to(dev)
            .to(torch.bfloat16)
        )

        def step():
            if a.zero_ws:
                layer.workspace.zero_()
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=a.n,
                size_k=a.k,
                bias=None,
                **kw,
            )

        ref = step().clone()
        torch.cuda.synchronize()
        bad, firstbad, worst = 0, None, 0.0
        for i in range(a.iters):
            o = step()
            if not torch.equal(o, ref):
                bad += 1
                worst = max(worst, (o.float() - ref.float()).abs().max().item())
                if firstbad is None:
                    firstbad = i
        torch.cuda.synchronize()
        print(
            f"{M:>6} {f'{bad}/{a.iters}':>12} {bad / a.iters:>9.4f} "
            f"{str(firstbad):>9} {worst:>11.3e}"
        )


if __name__ == "__main__":
    main()
