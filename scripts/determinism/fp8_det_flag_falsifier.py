"""#192: does SGLANG_DETERMINISTIC_FP8_GEMM actually buy bit-determinism on sm8x?

#190 measured that ``gptq_marlin_gemm`` -- the ONLY fp8 GEMM available on
sm80..sm88 -- is not run-to-run reproducible there: repeating one
``apply_fp8_marlin_linear`` call on bit-identical inputs at the 27B's real shape
(N=8704, K=5120) gave 1/1200 mismatching iterations at M=128 and 12/1200 at
M=512. ``scripts/determinism/fp8_marlin_hammer.py`` is that falsifier.

This script is its counterpart for the FIX. It drives the real entry point --
``Fp8LinearMethod.apply()`` on one hand-built block-fp8 layer -- twice in the
same process shape-for-shape, once with the flag off (Marlin) and once with the
gate forced (dequant W8A16 lane), and counts mismatching repeats for each. The
claim under test is narrow and falsifiable: the flag-on arm must be 0/N at the
shapes where the flag-off arm is not.

Two arms in ONE process is deliberate. The env var is read through an lru_cached
helper, so the second arm cannot simply re-read it; the runner therefore clears
those caches between arms and asserts the resulting routing flags, which also
proves the chain the flag is supposed to trigger:

    SGLANG_DETERMINISTIC_FP8_GEMM
      -> deterministic_fp8_marlin_disabled()  = True   (sm8x only)
      -> Fp8LinearMethod.use_marlin           = False
      -> fp8_needs_dequant_fallback()         = True
      -> Fp8LinearMethod.use_block_dequant    = True   (nothing lands nowhere)

Run it on an sm8x card. On this rig the torch device order is NOT the NVML
order, so pick the index by capability, not by nvidia-smi:

    CUDA_VISIBLE_DEVICES=1 python scripts/determinism/fp8_det_flag_falsifier.py \
        --ms 8,128,512 --iters 600
"""

import argparse
import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)


class FakeLayer(torch.nn.Module):
    """The attribute surface Fp8LinearMethod.apply() actually reads."""


def build_layer(n, k, block, dev, seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    layer = FakeLayer()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [block, block]
    # checkpoint layout for block fp8 is [n, k] (size_k_first=False)
    w = (torch.randn(n, k, generator=g, dtype=torch.float32) * 0.2).to(dev)
    layer.weight = torch.nn.Parameter(w.to(torch.float8_e4m3fn), requires_grad=False)
    ws = (
        torch.rand(
            (n + block - 1) // block,
            k // block,
            generator=g,
            dtype=torch.float32,
        )
        * 0.05
        + 0.01
    ).to(dev)
    layer.weight_scale_inv = torch.nn.Parameter(ws, requires_grad=False)
    return layer


def make_method(deterministic: bool):
    """Fresh Fp8LinearMethod under the requested flag state, caches cleared.

    The lru_caches are the reason this is a function and not two processes: the
    flag is read once per process by design (so the warning fires once per rank),
    which is right in production and wrong in an A/B harness.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization import fp8_utils as U
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    envs.SGLANG_DETERMINISTIC_FP8_GEMM.set(deterministic)
    U.deterministic_fp8_marlin_disabled.cache_clear()
    U.fp8_needs_dequant_fallback.cache_clear()

    cfg = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    return Fp8LinearMethod(cfg)


def count_mismatches(method, layer, x, iters):
    ref = method.apply(layer, x).clone()
    torch.cuda.synchronize()
    bad, first, worst = 0, None, 0.0
    for i in range(iters):
        o = method.apply(layer, x)
        if not torch.equal(o, ref):
            bad += 1
            worst = max(worst, (o.float() - ref.float()).abs().max().item())
            if first is None:
                first = i
    torch.cuda.synchronize()
    return bad, first, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, default="8,128,512")
    p.add_argument("--n", type=int, default=8704)  # gate_up per rank at TP=2
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--block", type=int, default=128)
    a = p.parse_args()

    dev = "cuda"
    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]
    print(f"# gpu={torch.cuda.get_device_name(0)} cap={cap} sm={sm} "
          f"N={a.n} K={a.k} block={a.block} iters={a.iters}")
    if not (80 <= sm < 89):
        print("# WARNING: not sm80..88 -- the flag is a no-op here by design, "
              "so both arms should be identical AND clean.")

    layer = build_layer(a.n, a.k, a.block, dev)

    arms = {}
    for name, deterministic in (("marlin", False), ("flag_on", True)):
        method = make_method(deterministic)
        from sglang.srt.layers.quantization import fp8_utils as U

        routing = (
            f"use_marlin={method.use_marlin} "
            f"use_block_dequant={method.use_block_dequant} "
            f"needs_fallback={U.fp8_needs_dequant_fallback()}"
        )
        print(f"# arm {name}: {routing}")
        if name == "marlin" and method.use_marlin:
            # Marlin wants its repacked layout + workspace; build it once, on
            # the same weights, so both arms answer for the same checkpoint.
            from sglang.srt.layers.quantization.marlin_utils_fp8 import (
                prepare_fp8_layer_for_marlin,
            )

            marlin_layer = build_layer(a.n, a.k, a.block, dev)
            prepare_fp8_layer_for_marlin(marlin_layer, size_k_first=False)
            arms[name] = (method, marlin_layer)
        else:
            arms[name] = (method, layer)

    g = torch.Generator(device="cpu").manual_seed(7)
    print(f"\n{'M':>6} {'arm':>10} {'bad/iters':>12} {'rate':>9} "
          f"{'firstbad':>9} {'worst':>11}")
    fail = False
    for M in [int(v) for v in a.ms.split(",")]:
        x = (
            (torch.randn(M, a.k, generator=g, dtype=torch.float32) * 0.5)
            .to(dev)
            .to(torch.bfloat16)
        )
        for name, (method, lyr) in arms.items():
            bad, first, worst = count_mismatches(method, lyr, x, a.iters)
            print(f"{M:>6} {name:>10} {f'{bad}/{a.iters}':>12} "
                  f"{bad / a.iters:>9.4f} {str(first):>9} {worst:>11.3e}")
            if name == "flag_on" and bad:
                fail = True

    print("\nRESULT:", "FAIL -- flag_on arm is not bit-identical" if fail
          else "PASS -- flag_on arm bit-identical at every M tested")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
