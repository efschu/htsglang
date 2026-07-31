#!/usr/bin/env python3
"""Isolate the DEVICE-GUARD effect from the ARCHITECTURE limit (#340, r10).

`c_micro.py` (arm C) launched the block-FP8 Triton matmul on a foreign card and
found two different failures stacked on top of each other:

  A  no guard    -> ValueError: Pointer argument (at 0) cannot be accessed
  B  with guard  -> CompilationError: type fp8e4nv not supported in this
                    architecture

B proves the guard fixed what it was meant to fix -- the pointer error is gone
-- but it cannot show a SUCCESS, because the foreign card is an sm86 RTX 3080
and triton has no fp8e4nv there. The two facts are entangled in one probe.

This probe removes the entanglement by using a kernel that compiles on BOTH
sm86 and sm120: a plain bf16 tiled matmul. Then "guard missing" is the only
variable left, and case B/C can reach a numerically checked SUCCESS.

  A  foreign tensors, current device = host, no guard -> expect FAILURE
  B  identical, wrapped in torch.cuda.device(foreign) -> expect SUCCESS
  C  the real LaneColumnParallelShell over two parts on two cards -> SUCCESS

A's pointer error is raised by triton's launch stub before any CUDA API call,
so it does not poison the context the way an illegal access does; all three
cases therefore share one process. Pass a case letter to run just one.
"""

from __future__ import annotations

import json
import sys
import traceback

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/spinning/wt-340/python")

M, K, N_PART = 64, 256, 128


@triton.jit
def _bf16_matmul_kernel(
    a_ptr,  # [M, K] row-major
    b_ptr,  # [N, K] row-major
    c_ptr,  # [M, N] row-major
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * K + kk[None, :],
            mask=(offs_m[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :] * K + kk[:, None],
            mask=(offs_n[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_bf16_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x @ w.T`` through Triton. Arch-neutral: bf16 in, fp32 accumulate."""
    assert x.is_contiguous() and w.is_contiguous()
    m, k = x.shape
    n = w.shape[0]
    c = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    _bf16_matmul_kernel[grid](x, w, c, m, n, k, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c


class _TritonQuantMethod:
    """Minimal quant_method whose apply() launches the arch-neutral kernel."""

    def apply(self, layer, x, bias=None):
        out = triton_bf16_linear(x, layer.weight)
        return out if bias is None else out + bias


class _Part(torch.nn.Module):
    """The smallest object LaneColumnParallelShell accepts as a lane part."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.output_partition_sizes = [weight.shape[0]]
        self.gather_output = False
        self.skip_bias_add = False
        self.bias = None
        self.quant_method = _TritonQuantMethod()


def _sample(rows: int):
    # Sampled on the CPU and moved: on-GPU randn is not architecture identical
    # across sm86/sm120, and both cards see these tensors.
    scale = K**-0.5
    x = torch.randn(M, K, dtype=torch.float32) * scale
    w = torch.randn(rows, K, dtype=torch.float32) * scale
    return x, w


def _row(case: str, guard: bool, fn):
    row = {
        "case": case,
        "guard": guard,
        "result": "SUCCESS",
        "exc_type": None,
        "exc": None,
        "max_abs_delta": None,
    }
    try:
        row["max_abs_delta"] = fn()
    except BaseException as exc:  # noqa: BLE001 - the failure IS the datum
        row.update(
            result="FAILURE",
            exc_type=type(exc).__name__,
            exc=str(exc).splitlines()[0][:300],
        )
        traceback.print_exc(file=sys.stderr)
    print(f"  {case:26s} {row['result']}  {row['exc'] or ''}", flush=True)
    return row


def _foreign_launch(host: int, foreign: int, guard: bool) -> float:
    x, w = _sample(N_PART)
    xd = x.to(f"cuda:{foreign}").to(torch.bfloat16)
    wd = w.to(f"cuda:{foreign}").to(torch.bfloat16)
    torch.cuda.set_device(host)
    if guard:
        with torch.cuda.device(foreign):
            out = triton_bf16_linear(xd, wd)
    else:
        out = triton_bf16_linear(xd, wd)
    torch.cuda.synchronize(foreign)
    ref = x @ w.t()
    return float((out.float().cpu() - ref).abs().max())


def _lane_shell(host: int, foreign: int) -> float:
    from sglang.srt.model_executor.dual_group_lane import LaneColumnParallelShell

    x, w = _sample(2 * N_PART)
    parts = [
        _Part(w[:N_PART].to(f"cuda:{host}").to(torch.bfloat16).contiguous()),
        _Part(w[N_PART:].to(f"cuda:{foreign}").to(torch.bfloat16).contiguous()),
    ]
    shell = LaneColumnParallelShell(parts)
    torch.cuda.set_device(host)
    out, _ = shell(x.to(f"cuda:{host}").to(torch.bfloat16))
    torch.cuda.synchronize(host)
    torch.cuda.synchronize(foreign)
    return float((out.float().cpu() - x @ w.t()).abs().max())


def main() -> int:
    report = {"visible": [], "host_device": None, "foreign_device": None, "cases": []}
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    report["visible"] = [torch.cuda.get_device_name(i) for i in range(count)]
    if count < 2:
        report["error"] = f"need at least two visible cards, saw {count}"
        print(json.dumps(report))
        print(report["error"], file=sys.stderr)
        return 2

    host = max(
        range(count), key=lambda i: torch.cuda.get_device_properties(i).total_memory
    )
    foreign = next(i for i in range(count) if i != host)
    report["host_device"] = host
    report["foreign_device"] = foreign

    wanted = sys.argv[1].upper() if len(sys.argv) > 1 else "ABC"
    if "A" in wanted:
        report["cases"].append(
            _row(
                "A: foreign, no guard",
                False,
                lambda: _foreign_launch(host, foreign, False),
            )
        )
    if "B" in wanted:
        report["cases"].append(
            _row(
                "B: foreign, device guard",
                True,
                lambda: _foreign_launch(host, foreign, True),
            )
        )
    if "C" in wanted:
        report["cases"].append(
            _row("C: lane shell, two cards", True, lambda: _lane_shell(host, foreign))
        )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
