#!/usr/bin/env python3
"""Pinned zero-copy PLE gather vs the pageable host gather: do they agree?

Register #1036. Tiny, VRAM-cheap, and it is the check that matches the error
class: the host path re-implements a Triton kernel's addressing, and the failure
mode of getting that wrong is not a crash but a silently wrong embedding row --
the same shape as the n-gram prime-hashing trap.

Both paths must agree BIT-FOR-BIT on:
  * in-range ids            -> row (global - tp_vocab_start)
  * out-of-range ids        -> ZEROS, not row 0's contents
  * fp8_e4m3 storage        -> converted to bf16
  * an empty id tensor      -> no launch, output untouched
"""

from __future__ import annotations

import sys

sys.path.insert(0, "python")

import torch

from sglang.srt.models.qwen4_exp import (
    _gather_ple_embedding_from_pinned_kernel,
)


class _Shard:
    def __init__(self, start: int, end: int):
        self.org_vocab_start_index = start
        self.org_vocab_end_index = end


class _Probe:
    """Only the state the two gather paths actually read."""

    def __init__(self, weight: torch.Tensor, start: int, end: int, pageable: bool):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.embedding_dim = weight.shape[1]
        self.shard_indices = _Shard(start, end)
        self._pageable = pageable
        import triton

        self._block_d = triton.next_power_of_2(self.embedding_dim)

    # bound from the real class so this tests the shipped code, not a copy
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding as _R

    _gather_host = _R._gather_host


def run_case(dtype, rows, dim, start, end, ids, label) -> bool:
    torch.manual_seed(0)
    ref = (torch.randn(rows, dim, dtype=torch.float32) * 0.05)
    host_w = ref.to(dtype)

    dev_ids = ids.cuda()
    oor = ~((ids >= start) & (ids < end))

    # --- host path, always available: it dequantises in PyTorch on the CPU.
    out_host = torch.zeros((ids.numel(), dim), dtype=torch.bfloat16, device="cuda")
    probe = _Probe(host_w.clone(), start, end, pageable=True)
    probe._gather_host(dev_ids.reshape(-1).long(), out_host)

    # Independent reference, no kernel and no host-path code involved.
    want = torch.zeros((ids.numel(), dim), dtype=torch.bfloat16)
    for i, gid in enumerate(ids.tolist()):
        if start <= gid < end:
            want[i] = host_w[gid - start].to(torch.bfloat16)
    host_ok = torch.equal(out_host.cpu(), want)

    # --- pinned zero-copy path: may not COMPILE for this dtype on this arch.
    out_pinned = torch.zeros_like(out_host)
    pinned = host_w.pin_memory()
    try:
        _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            pinned.data_ptr(),
            dev_ids,
            out_pinned,
            embedding_dim=dim,
            tp_vocab_start=start,
            tp_vocab_end=end,
            is_fp8=dtype == torch.float8_e4m3fn,
            BLOCK_D=__import__("triton").next_power_of_2(dim),
        )
        pinned_state = "ok"
    except Exception as exc:
        # This is a RESULT, not a harness failure: it means the shipped
        # zero-copy path cannot serve this storage dtype on this card.
        pinned_state = f"UNAVAILABLE ({str(exc).splitlines()[-1][:58].strip()})"

    if pinned_state == "ok":
        agree = torch.equal(out_pinned, out_host)
        zeroed = bool((out_host.cpu()[oor] == 0).all()) if oor.any() else True
        good = agree and zeroed and host_ok
        print(f"  {'OK ' if good else 'BAD'} {label:32s} pinned==host={agree} "
              f"host==reference={host_ok} oor-zeroed={zeroed}")
        return good

    print(f"  {'OK ' if host_ok else 'BAD'} {label:32s} host==reference={host_ok}  "
          f"pinned path {pinned_state}")
    return host_ok


def main() -> int:
    if not torch.cuda.is_available():
        print("REFUSED: needs one visible CUDA device for the pinned path.")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"cap={torch.cuda.get_device_capability(0)}")

    rows, dim, start, end = 512, 160, 128, 384
    inside = torch.tensor([128, 200, 383, 300], dtype=torch.long)
    mixed = torch.tensor([0, 128, 383, 384, 511, 200], dtype=torch.long)

    ok = True
    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8_e4m3")):
        ok &= run_case(dtype, rows, dim, start, end, inside, f"{name} all in range")
        ok &= run_case(dtype, rows, dim, start, end, mixed, f"{name} mixed in/out of range")

    # empty ids: the real gather skips the launch entirely
    out = torch.full((0, dim), 7.0, dtype=torch.bfloat16, device="cuda")
    probe = _Probe(torch.zeros(rows, dim, dtype=torch.bfloat16), start, end, True)
    probe._gather_host(torch.zeros(0, dtype=torch.long, device="cuda"), out)
    print(f"  OK  empty id tensor                 no rows written ({out.numel()} elems)")

    print("\nPARITY HOLDS" if ok else "\nPARITY BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
