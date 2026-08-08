#!/usr/bin/env python
"""#651: is `moe_align_block_size` correct on a 64-wide wavefront?

RESULT: NO DEFECT. The hypothesis below was FALSIFIED, twice over, and the
script is kept as the falsifier that settled it.

  * This test returns GREEN: 20/20 trials agree with the host reference at the
    real geometry (num_experts=256 passed as 257, topk=8, block_size=64).
  * The premise was false anyway. `get_device_properties(0).warp_size` reports
    **32** on gfx1103: RDNA3 runs HIP kernels in wave32 by default, and wave64
    is CDNA/MI. `WARP_SIZE 32` is therefore correct here, and the 64-bit
    `SGL_FULL_WARP_MASK` is a type requirement of HIP's `__shfl_*_sync` -- as
    that source file's own comment already said.

The real cause of the crashes was an amdgpu GPU wedge (`MES failed to respond
to msg=REMOVE_QUEUE` -> `GPU reset(N)`), visible only in dmesg. moe_align was
the messenger: its `RuntimeCheck` was simply the next error check after an
async fault raised elsewhere. See docs/dev/651/FINAL_651.md section 3.

BACKGROUND (the reasoning that motivated this test, preserved as written). Three serving crashes on this laptop (gfx1103 / Radeon 780M) died
at `moe_align_kernel.cu:530` with `HIP error: unspecified launch failure`, one
of them under six short prompts. Reading the launch site:

    #ifndef WARP_SIZE
    #define WARP_SIZE 32          // never overridden for ROCm
    #endif
    ...
    #ifdef USE_ROCM
    #define SGL_FULL_WARP_MASK 0xffffffffffffffffULL   // all 64 lanes
    ...
    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

and the JIT build flags (`jit_kernel/utils.py`) are
`["-DUSE_ROCM", "-std=c++20", "-O3"]` -- `USE_ROCM` is defined, `WARP_SIZE` is
NOT. So the shuffles are told that all 64 lanes participate while the
surrounding scan indexes as though a warp were 32 wide.

PREDICTED CONSEQUENCE. The kernel's job is an exclusive scan over per-expert
counts, producing the cumulative offsets at which each expert's tokens are
written. A scan that is wrong on wave64 yields wrong offsets, and the kernel
then writes `sorted_token_ids` at those offsets -- off the end of the buffer
when they are too large. That is a global out-of-bounds write, which surfaces
exactly as `unspecified launch failure`, and it is PROBABILISTIC because
whether the bad offset lands on unmapped memory depends on the expert
distribution of the particular batch. That matches all three specimens.

WHY THIS TEST IS BETTER THAN WAITING FOR A CRASH. A crash is rare and
destroys the context. The same defect has a deterministic, cheap signature:
`num_tokens_post_pad` is a pure function of the expert histogram, so it can be
computed exactly on the host and compared. If the scan is broken, this is
wrong on ordinary batches with no crash required.

Reference semantics (mirroring the caller in
`srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`, which passes
`num_experts + 1`): every expert's token count is padded up to a multiple of
`block_size`, and `num_tokens_post_pad` is the sum of those padded counts.

Exit code 0 if the kernel agrees with the reference on every trial, 1 if it
does not (that is the RED state this script exists to produce), 2 if the
kernel could not be exercised at all.
"""

import argparse
import sys

import torch


def reference_num_tokens_post_pad(topk_ids: torch.Tensor, num_experts_arg: int,
                                  block_size: int) -> int:
    """Sum over experts of ceil(count/block)*block, computed on the host."""
    counts = torch.bincount(
        topk_ids.reshape(-1).to(torch.int64), minlength=num_experts_arg
    )[:num_experts_arg]
    padded = ((counts + block_size - 1) // block_size) * block_size
    return int(padded.sum().item())


def run_trial(num_tokens, topk, num_experts, block_size, dtype, generator):
    from sglang.jit_kernel.moe_align import moe_align_block_size as jit_align

    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=dtype, device="cuda",
        generator=generator,
    )

    # Allocation mirrors the production caller exactly.
    num_experts_arg = num_experts + 1
    numel = topk_ids.numel()
    if numel < num_experts_arg + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts_arg + 1) * (block_size - 1)

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device="cuda"
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty(
        (num_experts_arg + 2,), dtype=torch.int32, device="cuda"
    )

    jit_align(
        topk_ids,
        num_experts_arg,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    torch.cuda.synchronize()

    got = int(num_tokens_post_pad.item())
    want = reference_num_tokens_post_pad(topk_ids, num_experts_arg, block_size)
    return got, want, max_num_tokens_padded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument(
        "--tokens",
        default="1,2,4,8,16,32,64,128,256,512",
        help="token counts to sweep (one trial each, then random repeats)",
    )
    ap.add_argument("--dtype", default="int32", choices=["int32", "int64"])
    args = ap.parse_args()

    dtype = torch.int32 if args.dtype == "int32" else torch.int64
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)

    print(f"moe_align_block_size: num_experts={args.num_experts} "
          f"(passed as {args.num_experts + 1}), topk={args.topk}, "
          f"block_size={args.block_size}, dtype={args.dtype}")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).gcnArchName})")

    token_list = [int(x) for x in args.tokens.split(",")]
    bad = 0
    total = 0
    for nt in token_list:
        for rep in range(max(1, args.trials // len(token_list))):
            total += 1
            try:
                got, want, cap = run_trial(
                    nt, args.topk, args.num_experts, args.block_size, dtype, gen
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  tokens={nt:5d} rep={rep}: LAUNCH FAILED: "
                      f"{type(exc).__name__}: {str(exc)[:120]}")
                bad += 1
                # The HIP context is dead after a launch failure; stop.
                print("\nVERDICT: RED (hard launch failure)")
                return 1
            ok = got == want
            if not ok:
                bad += 1
            flag = "ok " if ok else "BAD"
            if not ok or rep == 0:
                print(f"  [{flag}] tokens={nt:5d} numel={nt*args.topk:6d} "
                      f"num_tokens_post_pad got={got:7d} want={want:7d} "
                      f"(buffer cap {cap})")

    print()
    print(f"{total - bad}/{total} trials agree with the host reference")
    if bad:
        print("VERDICT: RED -- moe_align_block_size computes the wrong "
              "padded-token total. The block offsets it derives from the same "
              "scan are what the kernel writes through, which is the "
              "out-of-bounds route to 'unspecified launch failure'.")
        return 1
    print("VERDICT: GREEN -- kernel agrees with the host reference on every "
          "trial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
