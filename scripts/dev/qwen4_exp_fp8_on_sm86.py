#!/usr/bin/env python3
"""Can sm86 handle the fp8 PLE table, and where exactly does it refuse?

Register #1036, and it exists because my own wording implied a hardware limit that
I never established. The operator's objection is the right one: an RTX 3080 has fp16
tensor cores, so an fp8 TABLE is no obstacle to computing in fp16 -- you widen the
bytes and multiply in fp16. Nothing about this model asks a card to do fp8 ARITHMETIC.

Three separate questions, answered separately, all by COMPILING and COMPARING rather
than timing -- so the answers survive a loaded machine:

  A. Does the GPU widen fp8 -> bf16 at all on this card? (torch, device-side)
  B. Does the SHIPPED Triton kernel compile there? (it loads via the fp8e4nv type)
  C. Does a Triton kernel that loads the SAME BYTES as uint8 and widens them
     through a 256-entry lookup table compile and give identical values?

If C succeeds where B fails, the limit is TRITON'S TYPE SUPPORT in one kernel of
mine, NOT the silicon -- and the host gather is then an implementation detail rather
than a hardware necessity. That distinction decides whether the zero-copy path is
recoverable on two thirds of this rig.

Tiny: a 4096-row table, KB of traffic. No checkpoint. Far under the 512 MB line.
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "python")

ROWS, DIM = 4096, 160


@triton.jit
def _widen_via_lut(w_ptr, lut_ptr, ids_ptr, out_ptr, dim, BLOCK_D: tl.constexpr):
    """Load the fp8 bytes as UINT8 and widen through a 256-entry table.

    The shipped kernel asks Triton for a pointer of type fp8e4nv, which the Ampere
    backend does not offer (it has fp8e4b15 and fp8e5). But the bytes are just bytes:
    read them as uint8 and the widening becomes a gather from a 256-entry LUT, which
    is EXACT by construction -- every one of the 256 encodings, including subnormals
    and NaN, is whatever torch itself says it is. No arithmetic reconstruction of the
    exponent, so no chance of getting subnormals subtly wrong.
    """
    row = tl.load(ids_ptr + tl.program_id(0))
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    byte = tl.load(w_ptr + row * dim + offs, mask=mask, other=0).to(tl.int32)
    val = tl.load(lut_ptr + byte, mask=mask, other=0.0)
    tl.store(out_ptr + tl.program_id(0) * dim + offs, val, mask=mask)


def main() -> int:
    if not torch.cuda.is_available():
        print("needs a visible CUDA device")
        return 2
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name(0)}  sm{cap[0]}{cap[1]}")
    print(f"        fp16 tensor cores: yes (sm70+). fp8 tensor cores: "
          f"{'yes' if cap >= (8, 9) else 'NO -- and irrelevant here, see below'}")

    torch.manual_seed(0)
    ref32 = (torch.randn(ROWS, DIM) * 0.05)
    w_cpu = ref32.to(torch.float8_e4m3fn)
    ids = torch.arange(0, ROWS, 7, device="cuda", dtype=torch.long)
    truth = w_cpu.index_select(0, ids.cpu()).to(torch.float32).cuda()

    ok = True

    # ---- A. does the device widen fp8 -> bf16 at all?
    try:
        w_dev = w_cpu.cuda()
        got = w_dev.index_select(0, ids).to(torch.bfloat16).to(torch.float32)
        same = torch.equal(got, truth)
        print(f"\n  {'OK ' if same else 'BAD'} A. device-side fp8 -> bf16 (torch): "
              f"exact={same}")
        print("        so the CARD converts fp8 fine; this is the path the host")
        print("        gather uses for the widening after the row copy.")
        ok &= same
    except Exception as exc:
        print(f"\n  BAD A. device-side fp8 -> bf16 FAILED: {str(exc)[:60]}")
        ok = False

    # ---- B. the shipped kernel, which loads through the fp8e4nv type
    from sglang.srt.models.qwen4_exp import (
        _gather_ple_embedding_from_pinned_kernel,
    )

    w_pin = w_cpu.clone().pin_memory()
    out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
    try:
        _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            w_pin.data_ptr(), ids, out, DIM, 0, ROWS,
            is_fp8=True, BLOCK_D=triton.next_power_of_2(DIM),
        )
        torch.cuda.synchronize()
        same = torch.equal(out.to(torch.float32), truth)
        print(f"\n  OK  B. shipped zero-copy kernel COMPILES here, exact={same}")
        b_works = True
        ok &= same
    except Exception as exc:
        msg = str(exc).strip().splitlines()[-1][:66]
        print(f"\n  --  B. shipped zero-copy kernel REFUSED at COMPILE time:")
        print(f"        {msg}")
        print("        Note WHAT is unsupported: the fp8e4nv *pointer type* in a")
        print("        Triton load. Not a multiply, not a tensor-core op.")
        b_works = False

    # ---- C. same bytes, loaded as uint8, widened through a LUT
    lut = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .cuda()
    )
    lut = torch.nan_to_num(lut, nan=0.0)  # the one NaN encoding; unused by this table
    try:
        w_u8 = w_cpu.view(torch.uint8).cuda()
        out_c = torch.empty(ids.numel(), DIM, dtype=torch.float32, device="cuda")
        _widen_via_lut[(ids.numel(),)](
            w_u8, lut, ids, out_c, DIM, BLOCK_D=triton.next_power_of_2(DIM)
        )
        torch.cuda.synchronize()
        same = torch.equal(out_c, truth)
        print(f"\n  {'OK ' if same else 'BAD'} C. uint8 + 256-entry LUT in Triton: "
              f"COMPILES, exact={same}")
        ok &= same
        if same and not b_works:
            print("\n  ==> THE LIMIT IS TRITON'S TYPE SUPPORT IN MY KERNEL, NOT THE CARD.")
            print("      Same bytes, same arch, bit-exact values, one Triton kernel.")
            print("      So the zero-copy path IS recoverable on sm86; the host gather")
            print("      is an implementation detail today, not a hardware necessity.")
            print("      UNBUILT: this probe reads a DEVICE table, while the real")
            print("      kernel reads PINNED HOST memory over PCIe. That part is")
            print("      unchanged by the dtype and already works for bf16, but this")
            print("      probe does not prove it, and no cost is claimed either way --")
            print("      a stopwatch on this shared box would be worthless.")
    except Exception as exc:
        msg = str(exc).strip().splitlines()[-1][:66]
        print(f"\n  BAD C. uint8 + LUT ALSO refused: {msg}")
        print("        then the arch really cannot read these bytes in Triton")
        ok = False

    print(f"\n{'ALL CHECKS PASS' if ok else 'SOMETHING FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
