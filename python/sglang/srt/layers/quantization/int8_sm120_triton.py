"""INT8 W8A8 small-M GEMM for sm_120 (RTX 5090): Triton, exact int32 split-K.

Why (Agent KD, /spinning/gpu-arb/docs/INT8_DOWNPROJ_SM120.md): on sm_120,
``sgl_kernel.int8_scaled_mm`` runs the sm89 table (CUTLASS 2.x). For M <= 16
that is a 16x64x128 / 16x128x128 tile with split-K 1, so down_proj
(N = 5120) launches 80 CTAs on 170 SMs, and gate_up (N = 32u) falls off a
wave edge between u = 680 and 684 (171 tiles -> a second wave).

This kernel picks the N tile and the split-K per (N, K, M) from a table
MEASURED on the 5090 (int8_mm_sweep 20260926T171511Z, see SM120_TABLE). The
split-K partial sums are int32 and are summed in int32, so the accumulator
is exactly the one CUTLASS forms; the epilogue is CUTLASS's
EpilogueVisitorPerRowPerCol order ``float(acc) * (scale_col * scale_row)``
rounded RTNE to bf16. The sweep measured the output bit-identical to sgl in
every cell and every config (bitwise_eq_vs_sgl = 1.0 on 1024 columns).

Scope: sm_120 only (exactly (12, 0) -- the only device measured), M <= 16,
bias-free, bf16 out, (N, K) in the table. Anything else returns None and the
caller runs sgl. Switch: SGLANG_INT8_SM120_TRITON (default off).

CUDA graphs: no host sync, no autotune; the grid and all constexprs come from
host ints (tensor shapes + table). Triton compiles on the first call of each
(config, specialization), which is the eager warmup that precedes every
capture (full_cuda_graph_backend.capture_one -> run_capture_warmups), and
lands in TRITON_CACHE_DIR (container: /root/.triton, bind-mounted from the
acceptance dir by host_acceptance.sh).

Instrument: one log line "INT8-SM120-TRITON armed" per process, one line per
distinct (N, K, M-bucket) decision, and host-call counters (``counters()``).
The counters count host calls (eager + capture), not graph replays.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages); BLOCK_M is always 16.
Cfg = Tuple[int, int, int, int, int]

BLOCK_M = 16
MAX_M = 16
M_BUCKETS = (8, 16)
REDUCE_BLOCK = 1024
SM120 = (12, 0)

TABLE_SOURCE = (
    "/spinning/evidence-665-f1/int8_mm_sweep_20260926T171511Z/sweep.json "
    "(RTX 5090 NVML 1, exclusive window qw7rd4, sgl_kernel wheel 67f03cfa, "
    "triton 3.6.0, torch 2.11.0+cu130)"
)

# (N, K) -> {M bucket: cfg}. Derived by ``derive_table`` from TABLE_SOURCE:
# per (op, u, M) the Triton lane with the lowest median among the lanes whose
# whole replay range (5 CUDA-graph replays) lies below sgl's whole replay
# range and whose output is bit-identical to sgl; no such lane -> no entry
# (sgl). gate_up: N = 32u, K = 5120; down: N = 5120, K = 16u. Comment per
# row: op u | M: sgl us -> Triton us (median per call).
SM120_TABLE: Dict[Tuple[int, int], Dict[int, Cfg]] = {
    # gate_up 584 | 8: 61.37 -> 59.78 | 16: 60.96 -> 60.06
    (18688, 5120): {8: (64, 256, 1, 4, 3), 16: (32, 256, 1, 4, 3)},
    # gate_up 628 | 8: 65.67 -> 64.97 | 16: 65.86 -> 64.70
    (20096, 5120): {8: (32, 256, 1, 4, 3), 16: (64, 256, 1, 4, 3)},
    # gate_up 644 | 8: 67.30 -> 66.10 | 16: 68.32 -> 66.38
    (20608, 5120): {8: (64, 256, 1, 4, 3), 16: (64, 256, 1, 4, 3)},
    # gate_up 652 | 8: 68.55 -> 67.14 | 16: sgl
    (20864, 5120): {8: (64, 256, 1, 4, 3)},
    # gate_up 676 | 8: 70.29 -> 69.24 | 16: 70.81 -> 69.42
    (21632, 5120): {8: (32, 256, 1, 4, 3), 16: (64, 256, 1, 4, 3)},
    # gate_up 680 | 8: 70.76 -> 69.54 | 16: 71.34 -> 69.67
    (21760, 5120): {8: (64, 256, 1, 4, 3), 16: (128, 256, 1, 4, 3)},
    # gate_up 684 | 8: 77.84 -> 70.47 | 16: 77.83 -> 71.05
    (21888, 5120): {8: (64, 128, 1, 4, 4), 16: (64, 128, 1, 4, 4)},
    # gate_up 704 | 8: 78.97 -> 73.18 | 16: 79.16 -> 73.25
    (22528, 5120): {8: (32, 128, 1, 4, 4), 16: (64, 128, 1, 4, 4)},
    # gate_up 707 | 8: 78.48 -> 72.99 | 16: 78.75 -> 73.45
    (22624, 5120): {8: (32, 128, 1, 4, 4), 16: (32, 128, 1, 4, 4)},
    # down 584 | 8: sgl | 16: 30.82 -> 30.66
    (5120, 9344): {16: (32, 256, 1, 4, 3)},
    # down 628 | 8: 34.37 -> 33.79 | 16: 34.36 -> 33.83
    (5120, 10048): {8: (128, 256, 4, 4, 3), 16: (128, 256, 4, 4, 3)},
    # down 644 | 8: 36.08 -> 34.65 | 16: 35.66 -> 34.81
    (5120, 10304): {8: (64, 256, 2, 4, 3), 16: (64, 256, 2, 4, 3)},
    # down 652 | 8: 35.59 -> 34.73 | 16: 35.28 -> 34.68
    (5120, 10432): {8: (64, 256, 4, 4, 3), 16: (32, 256, 4, 4, 3)},
    # down 676 | 8: 36.59 -> 36.22 | 16: 36.89 -> 36.39
    (5120, 10816): {8: (32, 128, 4, 4, 4), 16: (128, 256, 4, 4, 3)},
    # down 680 | 8: sgl | 16: sgl  (no entry)
    # down 684 | 8: sgl | 16: 36.79 -> 36.04
    (5120, 10944): {16: (32, 256, 1, 4, 3)},
    # down 704 | 8: 39.10 -> 37.22 | 16: 39.22 -> 37.19
    (5120, 11264): {8: (32, 256, 2, 4, 3), 16: (64, 256, 4, 4, 3)},
    # down 707 | 8: 40.99 -> 38.00 | 16: 41.05 -> 38.35
    (5120, 11312): {8: (32, 256, 4, 4, 3), 16: (64, 256, 4, 4, 3)},
}


def cfg_name(cfg: Cfg) -> str:
    bn, bk, s, w, st = cfg
    return f"tri_n{bn}_k{bk}_s{s}_w{w}_st{st}"


def _parse_cfg_name(name: str) -> Cfg:
    # "tri_n64_k256_s4_w4_st3" -> (64, 256, 4, 4, 3)
    parts = name.split("_")
    if len(parts) != 6 or parts[0] != "tri":
        raise ValueError(f"not a Triton lane name: {name!r}")
    return (
        int(parts[1][1:]),
        int(parts[2][1:]),
        int(parts[3][1:]),
        int(parts[4][1:]),
        int(parts[5][2:]),
    )


def derive_table(sweep: dict) -> Dict[Tuple[int, int], Dict[int, Cfg]]:
    """The SM120_TABLE rule applied to a parsed int8_mm_sweep sweep.json."""
    table: Dict[Tuple[int, int], Dict[int, Cfg]] = {}
    for cell in sweep["cells"]:
        n, k = int(cell["N"]), int(cell["K"])
        for m_str, mv in cell["m"].items():
            lanes = mv.get("lanes", {})
            sgl = lanes.get("sgl")
            if not sgl or "us_all" not in sgl:
                continue
            sgl_lo = min(sgl["us_all"])
            best = None
            for name, lane in lanes.items():
                if not name.startswith("tri_") or "us_all" not in lane:
                    continue
                if lane.get("bitwise_eq_vs_sgl") != 1.0:
                    continue
                if max(lane["us_all"]) >= sgl_lo:
                    continue
                if best is None or lane["us"] < best[1]:
                    best = (name, lane["us"])
            if best is not None:
                table.setdefault((n, k), {})[int(m_str)] = _parse_cfg_name(best[0])
    return table


def m_bucket(m: int) -> Optional[int]:
    """1..8 -> 8, 9..16 -> 16 (measured only at 8 and 16; BLOCK_M is 16 for
    both, so the grid is the same inside a bucket), else None."""
    if m < 1 or m > MAX_M:
        return None
    return 8 if m <= 8 else 16


def lookup(n: int, k: int, m: int, table: Optional[Dict] = None) -> Optional[Cfg]:
    b = m_bucket(m)
    if b is None:
        return None
    row = (SM120_TABLE if table is None else table).get((n, k))
    if row is None:
        return None
    return row.get(b)


def k_per_split(k: int, block_k: int, split_k: int) -> int:
    """K elements per split, a multiple of BLOCK_K (same as the bench)."""
    k_iters = -(-k // block_k)
    return -(-k_iters // split_k) * block_k


# ---------------------------------------------------------------------------
# Triton kernels (built lazily: TRITON_INTERPRET is read at decoration time,
# so the interpreter test sets it before the first call).
_KERNELS: dict = {}
_KERNEL_LOCK = threading.Lock()


def _kernels() -> dict:
    if _KERNELS:
        return _KERNELS
    with _KERNEL_LOCK:
        if _KERNELS:
            return _KERNELS
        import triton
        import triton.language as tl

        @triton.jit
        def _i8mm_sm120_kernel(
            a_ptr,
            b_ptr,
            sa_ptr,
            sb_ptr,
            c_ptr,
            p_ptr,
            M,
            N,
            K,
            stride_am,
            stride_bn,
            stride_cm,
            K_PER_SPLIT,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            SPLIT_K: tl.constexpr,
        ):
            # A: [M, K] int8 row-major (stride_ak == 1). B: [K, N] int8
            # column-major, i.e. layer.weight after process_weights_after_loading
            # (element (k, n) at b_ptr + n * stride_bn + k). C: [M, N] bf16.
            pid_n = tl.program_id(0)
            pid_k = tl.program_id(1)
            pid_m = tl.program_id(2)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            mask_m = offs_m < M
            mask_n = offs_n < N
            k_lo = pid_k * K_PER_SPLIT
            k_hi = tl.minimum(k_lo + K_PER_SPLIT, K)
            a_ptrs = (
                a_ptr
                + offs_m[:, None].to(tl.int64) * stride_am
                + (k_lo + offs_k)[None, :]
            )
            b_ptrs = (
                b_ptr
                + offs_n[None, :].to(tl.int64) * stride_bn
                + (k_lo + offs_k)[:, None]
            )
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kk in range(k_lo, k_hi, BLOCK_K):
                mask_k = (kk + offs_k) < k_hi
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K
            mask_c = mask_m[:, None] & mask_n[None, :]
            if SPLIT_K == 1:
                sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
                sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
                # CUTLASS EpilogueVisitorPerRowPerCol: accum * (scale_col * scale_row)
                c = acc.to(tl.float32) * (sb[None, :] * sa[:, None])
                c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
                tl.store(
                    c_ptrs,
                    c.to(c_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
                    mask=mask_c,
                )
            else:
                p_ptrs = p_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
                tl.store(p_ptrs, acc, mask=mask_c)

        @triton.jit
        def _i8mm_sm120_reduce(
            p_ptr,
            sa_ptr,
            sb_ptr,
            c_ptr,
            M,
            N,
            stride_cm,
            SPLIT_K: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            # Sum of the int32 partials in int32 (exact), then the same epilogue.
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < M * N
            acc = tl.zeros((BLOCK,), dtype=tl.int32)
            for s in tl.static_range(SPLIT_K):
                acc += tl.load(p_ptr + s * M * N + offs, mask=mask, other=0)
            row = offs // N
            col = offs % N
            sa = tl.load(sa_ptr + row, mask=mask, other=0.0)
            sb = tl.load(sb_ptr + col, mask=mask, other=0.0)
            c = acc.to(tl.float32) * (sb * sa)
            tl.store(
                c_ptr + row * stride_cm + col,
                c.to(c_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
                mask=mask,
            )

        _KERNELS.update(
            triton=triton, kernel=_i8mm_sm120_kernel, reduce=_i8mm_sm120_reduce
        )
    return _KERNELS


def launch(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    cfg: Cfg,
) -> torch.Tensor:
    """out[M, N] = (x_q @ weight) * x_scale[:, None] * weight_scale[None, :].

    No host sync; the int32 split-K workspace is a plain torch.empty (inside
    a capture it comes from the graph pool). Scales are read flat."""
    t = _kernels()
    triton = t["triton"]
    bn, bk, s, nw, ns = cfg
    m, k = x_q.shape
    n = weight.shape[1]
    kps = k_per_split(k, bk, s)
    grid = (triton.cdiv(n, bn), s, triton.cdiv(m, BLOCK_M))
    part = (
        torch.empty((s, m, n), dtype=torch.int32, device=x_q.device) if s > 1 else out
    )
    t["kernel"][grid](
        x_q,
        weight,
        x_scale,
        weight_scale,
        out,
        part,
        m,
        n,
        k,
        x_q.stride(0),
        weight.stride(1),
        out.stride(0),
        kps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT_K=s,
        num_warps=nw,
        num_stages=ns,
    )
    if s > 1:
        t["reduce"][(triton.cdiv(m * n, REDUCE_BLOCK),)](
            part,
            x_scale,
            weight_scale,
            out,
            m,
            n,
            out.stride(0),
            SPLIT_K=s,
            BLOCK=REDUCE_BLOCK,
            num_warps=4,
        )
    return out


# ---------------------------------------------------------------------------
# Dispatch
_CAPS: Dict[int, Tuple[int, int]] = {}
_COUNTERS: Dict[str, int] = {"triton": 0, "sgl_fallback": 0, "inactive": 0}
_SEEN: set = set()
_STATE = {"armed_logged": False, "inactive_logged": False}


def _on_cuda(t: torch.Tensor) -> bool:
    return t.is_cuda


def _device_capability(device: torch.device) -> Tuple[int, int]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cap = _CAPS.get(idx)
    if cap is None:
        cap = tuple(torch.cuda.get_device_capability(idx))
        _CAPS[idx] = cap
    return cap


def counters() -> Dict[str, int]:
    """Host-call counts since process start (eager and capture calls; a graph
    replay makes no host call and is not counted)."""
    return dict(_COUNTERS)


def _reset_for_tests() -> None:
    _CAPS.clear()
    _SEEN.clear()
    for key in _COUNTERS:
        _COUNTERS[key] = 0
    _STATE["armed_logged"] = False
    _STATE["inactive_logged"] = False


def _layout_ok(x_q, weight, x_scale, weight_scale) -> bool:
    m, k = x_q.shape
    return (
        x_q.dtype == torch.int8
        and weight.dtype == torch.int8
        and weight.dim() == 2
        and weight.shape[0] == k
        and x_q.stride(1) == 1
        and weight.stride(0) == 1
        and x_scale.dtype == torch.float32
        and weight_scale.dtype == torch.float32
        and x_scale.numel() == m
        and weight_scale.numel() == weight.shape[1]
        and x_scale.is_contiguous()
        and weight_scale.is_contiguous()
    )


def maybe_int8_scaled_mm(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """The Triton result, or None when this call stays on sgl int8_scaled_mm.

    Called only with the switch on. Same argument meaning as
    sgl_kernel.int8_scaled_mm(x_q, weight, x_scale, weight_scale, out_dtype,
    bias)."""
    if not _on_cuda(x_q):
        return None
    cap = _device_capability(x_q.device)
    if cap != SM120:
        _COUNTERS["inactive"] += 1
        if not _STATE["inactive_logged"]:
            _STATE["inactive_logged"] = True
            logger.info(
                "INT8-SM120-TRITON inactive on this rank: device %s is sm_%d%d "
                "(switch on; the dispatch needs sm_120) -> sgl int8_scaled_mm",
                x_q.device,
                cap[0],
                cap[1],
            )
        return None
    if not _STATE["armed_logged"]:
        _STATE["armed_logged"] = True
        logger.info(
            "INT8-SM120-TRITON armed: device %s sm_120, %d (N,K) shapes in the "
            "table, M<=%d, bias-free bf16 only; source %s",
            x_q.device,
            len(SM120_TABLE),
            MAX_M,
            TABLE_SOURCE,
        )
    if bias is not None or out_dtype != torch.bfloat16 or x_q.dim() != 2:
        _COUNTERS["sgl_fallback"] += 1
        return None
    m, k = x_q.shape
    n = weight.shape[-1]
    cfg = lookup(n, k, m)
    if cfg is not None and not _layout_ok(x_q, weight, x_scale, weight_scale):
        cfg = None
    key = (n, k, m_bucket(m) or m)
    if key not in _SEEN:
        _SEEN.add(key)
        if cfg is not None:
            logger.info(
                "INT8-SM120-TRITON use N=%d K=%d M<=%d -> %s",
                n,
                k,
                key[2],
                cfg_name(cfg),
            )
        elif m_bucket(m) is not None:
            logger.info(
                "INT8-SM120-TRITON sgl N=%d K=%d M<=%d (no table entry or layout)",
                n,
                k,
                key[2],
            )
    if cfg is None:
        _COUNTERS["sgl_fallback"] += 1
        return None
    _COUNTERS["triton"] += 1
    out = torch.empty((m, n), dtype=out_dtype, device=x_q.device)
    return launch(x_q, weight, x_scale, weight_scale, out, cfg)


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Print the SM120_TABLE rule applied to an int8_mm_sweep sweep.json"
    )
    ap.add_argument("sweep_json")
    args = ap.parse_args(argv)
    with open(args.sweep_json) as f:
        table = derive_table(json.load(f))
    for (n, k), row in sorted(table.items(), key=lambda kv: (kv[0][1] == 5120, kv[0])):
        print(
            f"    ({n}, {k}): {{{', '.join(f'{b}: {row[b]}' for b in sorted(row))}}},"
        )
    print(f"# {len(table)} shapes; identical to SM120_TABLE: {table == SM120_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
