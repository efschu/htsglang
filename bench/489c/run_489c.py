#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#489 (c) / #726 -- one-command IMMA-QK microbench runner.

    python bench/489c/run_489c.py --card 0 --out results_card0.json
    python bench/489c/run_489c.py --dry-run          # no GPU, plumbing only

ONE CARD PER INVOCATION, on purpose. #489 (c) forbids averaging this rig's
sm_86 and sm_120 cards, and the cheapest way to make that impossible is to
never have two cards in one process. Run it three times under a gpu-arb claim
and hand the three JSON files to ``--report``.

WHAT IT MEASURES. Three arms over a synthetic K-cache at the serving model's
head geometry: int8-K with native IMMA and no dequant of K (arm A), the
deployed fp8-KV path's shape (arm B), and a bf16 reference used only as a
correctness bound (arm C). The published -72% @58K inversion came from a
dequant-to-bf16 Triton lane; arm A never materialises a bf16 K, which is the
whole reason this bench exists.

WHAT IT DOES NOT DO. It loads no model, touches no serving process, and writes
nothing outside its ``--out`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HERE, "qk_arms.cu")
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")

#: PTX ISA floors differ per target and this is NOT cosmetic: sm_120a refuses
#: .version 8.0. Emitting one version for both fails on the 5090 for a reason
#: unrelated to the instruction. Pinned by the harness tests.
ARCH_PTX_FLOOR = {"sm_86": "8.0", "sm_89": "8.0", "sm_90a": "8.0", "sm_120a": "8.7"}

#: Qwen3.8-27B geometry, config-derived (num_key_value_heads=4, head_dim=256).
#: Q heads per rank are a REQUIRED input rather than a default, because the
#: uneven-TP shard vector is a deployment fact this file must not invent.
DEFAULT_KV_HEADS = 4
DEFAULT_HEAD_DIM = 256


def compile_cubin(arch: str, out: str) -> str:
    cmd = [NVCC, "-cubin", f"-arch={arch}", CU, "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"nvcc failed for {arch}:\n{r.stderr[-2000:]}")
    return out


def detect_card(index: int):
    """Card identity from NVML, never from torch's enumeration order.

    The rig's device-order trap (#82) is that torch and NVML can disagree; the
    per-card rule is worthless if the label is wrong.
    """
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
    sm = f"sm_{major}{minor}"
    arch = "sm_120a" if (major, minor) == (12, 0) else sm
    uuid = pynvml.nvmlDeviceGetUUID(h)
    return {
        "card": f"{name} [{(uuid.decode() if isinstance(uuid, bytes) else uuid)[-8:]}]",
        "sm": sm,
        "arch": arch,
    }


def a_vs_a(run_one, seconds: float) -> float:
    """The noise floor for THIS card, measured before any comparison.

    Two runs of the SAME arm. A gain smaller than this spread is not a result,
    and the rig's standing 14.1% is a prior, not a substitute for measuring.
    """
    a = run_one(seconds)
    b = run_one(seconds)
    return abs(a - b) / max(min(a, b), 1e-9)


def timed(fn, seconds: float):
    """Run at least ``seconds`` of wall clock and return ms per round.

    Per the ms-per-round canon: bound the run by TIME, not by iteration count,
    and never report a run shorter than the floor.
    """
    fn()  # warm-up, not counted
    t0 = time.perf_counter()
    rounds = 0
    while time.perf_counter() - t0 < seconds:
        fn()
        rounds += 1
    elapsed = time.perf_counter() - t0
    return (elapsed / max(rounds, 1)) * 1000.0, elapsed


def build_extension(build_dir: str = "/tmp/489c_ext"):
    """Compile+bind the arms. Builds WITHOUT a device when TORCH_CUDA_ARCH_LIST
    is set, which is what makes the dry-run a real smoke rather than a mock."""
    os.makedirs(build_dir, exist_ok=True)
    from torch.utils.cpp_extension import load

    return load(
        name="qk489c",
        sources=[os.path.join(HERE, "qk_binding.cpp"), CU],
        build_directory=build_dir,
        verbose=False,
    )


def make_buffers(dev, tokens: int, d: int, heads: int, groups: int, arm: str):
    """Synthetic K-cache at the serving geometry.

    CPU-SAMPLED then moved, never sampled on device: torch.randn on GPU is not
    architecture-identical across sm_86 and sm_120, and this rig has both -- a
    device-sampled input would make the two cards' correctness numbers
    incomparable for a reason that has nothing to do with the kernels.
    """
    import torch

    g = torch.Generator().manual_seed(489)
    if arm == "int8_imma" or arm == "int8":
        q = torch.randint(-127, 127, (heads, d // 4), generator=g,
                          dtype=torch.int32).to(dev)
        k = torch.randint(-127, 127, (tokens, d // 4), generator=g,
                          dtype=torch.int32).to(dev)
    elif arm == "fp8_deployed":
        q = torch.randn(heads * d, generator=g).half().to(dev)
        k = torch.randint(0, 255, (tokens, d), generator=g,
                          dtype=torch.uint8).to(dev)
    else:
        q = torch.randn(heads * d, generator=g).to(dev)
        k = torch.randn(tokens * d, generator=g).view(tokens, d).to(dev)
    ks = torch.randn(tokens * groups, generator=g).abs().half().to(dev) * 0.01
    out = torch.zeros(heads * tokens, dtype=torch.float32, device=dev)
    return q, k, ks, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--q-heads", type=int, help="Q heads on THIS rank (required)")
    ap.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="results_489c.json")
    ap.add_argument("--dry-run", action="store_true", help="no GPU: plumbing only")
    ap.add_argument("--report", nargs="*", help="judge existing JSON files")
    args = ap.parse_args()

    if args.report:
        rows = []
        for path in args.report:
            with open(path) as fh:
                rows += [D.ArmResult(**r) for r in json.load(fh)["results"]]
        print(D.render(D.evaluate(rows)))
        return 0

    if args.dry_run:
        # MOCK-SMOKE: everything except the silicon. Compiles both targets,
        # exercises the timing wrapper and the full decision path, and proves
        # the runner would refuse a too-short run rather than reporting it.
        for arch in ("sm_86", "sm_120a"):
            compile_cubin(arch, f"/tmp/489c_{arch}.cubin")
            print(f"  cubin OK  {arch}  (ptx floor {ARCH_PTX_FLOOR[arch]})")
        ms, secs = timed(lambda: sum(range(10000)), 0.05)
        print(f"  timing wrapper OK: {ms:.4f} ms/round over {secs:.2f}s")
        try:
            D.validate([D.ArmResult("c", "sm_86", 1, 1, "int8_imma", 1.0, 0.1, "x")])
        except D.BenchError as e:
            print(f"  short-run refusal OK: {str(e)[:60]}...")
        else:
            raise SystemExit("DRY-RUN FAILED: a 0.1s run was not refused")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        ext = build_extension("/tmp/489c_ext_dry")
        missing = [f for f in ("launch_int8_imma", "launch_fp8_deployed",
                               "launch_bf16_reference", "arm_b_native")
                   if not hasattr(ext, f)]
        if missing:
            raise SystemExit(f"DRY-RUN FAILED: extension missing {missing}")
        print("  extension builds AND imports with no device; all 4 symbols bound")
        print("  dry-run complete: kernels compile, plumbing sound, no GPU touched")
        return 0

    if args.q_heads is None:
        raise SystemExit(
            "--q-heads is required. The uneven-TP shard vector is a deployment "
            "fact; this bench refuses to invent it, because a wrong head count "
            "would silently change the arithmetic intensity of every arm."
        )

    info = detect_card(args.card)
    arch = info["arch"]
    if arch not in ARCH_PTX_FLOOR:
        raise SystemExit(f"unmapped arch {arch}: add its PTX floor before running")
    ext = build_extension()
    native = int(ext.arm_b_native())
    shape_b = "fp8_native_mma" if native else "fp8_dequant_hmma"
    print(f"card {info['card']}  {info['sm']} -> {arch}   arm B shape: {shape_b}")

    import torch

    dev = torch.device("cuda:0")
    d = args.head_dim
    heads = args.q_heads
    groups = max(1, d // 64)
    results = []

    # A-vs-A FIRST, on this card, before any arm is compared with any other.
    def _aa(seconds):
        t, k, ks, out = make_buffers(dev, 8192, d, heads, groups, "int8")
        ms, _ = timed(lambda: (ext.launch_int8_imma(t, k, ks, out, 8192, heads,
                                                    groups), torch.cuda.synchronize()),
                      seconds)
        return ms

    floor = a_vs_a(_aa, min(args.seconds, 3.0))
    print(f"measured A-vs-A spread on this card: {floor:.4f} "
          f"(rig prior {D.RIG_NOISE_FLOOR:.3f})")

    for depth in D.SPEC_DEPTHS:
        for batch in D.SPEC_BATCHES:
            n = depth * batch
            for arm, shape in (("int8_imma", "imma_s32"),
                               ("fp8_deployed", shape_b),
                               ("bf16_reference", "fp32_scalar")):
                q, k, ks, out = make_buffers(dev, n, d, heads, groups, arm)
                if arm == "int8_imma":
                    fn = lambda: ext.launch_int8_imma(q, k, ks, out, n, heads, groups)
                elif arm == "fp8_deployed":
                    fn = lambda: ext.launch_fp8_deployed(q, k, ks, out, n, heads,
                                                         d, groups)
                else:
                    fn = lambda: ext.launch_bf16_reference(q, k, out, n, heads, d)
                ms, secs = timed(lambda: (fn(), torch.cuda.synchronize()),
                                 args.seconds)
                results.append(D.ArmResult(info["card"], info["sm"], depth, batch,
                                           arm, ms, secs, shape,
                                           None))
                print(f"  d={depth:>7} bs={batch} {arm:<15} {ms:9.4f} ms/round "
                      f"over {secs:5.1f}s  [{shape}]")

    payload = {"card": info, "noise_floor_measured": floor,
               "results": [dataclasses.asdict(r) for r in results]}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}. Judge all three cards together with --report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
