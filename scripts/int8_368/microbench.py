#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #368 -- per-GEMM microbenchmark of the INT8 W8A8 decode path.

WHAT THIS ANSWERS
=================
The designated standard model (Qwen3.6-27B-INT8-W8A8) loses 4-10 % decode
throughput against the FP8 checkpoint at small batch (#354). Roughly a third
of that is explained by the -6.5 % accept length. The open question is the
rest: is the INT8 linear path itself slow in the decode regime?

#370 established that there is nothing to tune by config generation --
``CompressedTensorsW8A8Int8.apply_weights`` dispatches to the CUTLASS
``sgl_kernel.int8_scaled_mm`` and never consults the Triton config directory,
so there is no #255 analog. What is left is to price the two kernels that DO
run, separately, at the shapes and batch sizes the serving path uses:

    python/sglang/srt/layers/quantization/compressed_tensors/schemes/
        compressed_tensors_w8a8_int8.py:213   x_q, x_scale = per_token_quant_int8(x)
        compressed_tensors_w8a8_int8.py:215   int8_scaled_mm(x_q, layer.weight,
                                                x_scale, layer.weight_scale,
                                                out_dtype=x.dtype, bias=bias)

Two kernel launches per linear layer, no fusion, no epilogue reuse. The
activation quant is a Triton row kernel (``int8_kernel.py:79``,
``_per_token_quant_int8``, one program per token row, ``num_stages=1``); at
M=1 it is a one-row launch whose cost is essentially launch overhead. The
GEMM is CUTLASS. Which of the two dominates at M<=8 decides the remedy:

    quant dominant           -> fusion candidate (quant into the previous
                                layer's epilogue, or a fused quant+GEMM)
    gemm slow at M<=8        -> kernel/dispatch candidate (a GEMV-shaped
                                path, or a dispatch threshold)
    neither above the floor  -> the deficit is accept-length physics; close
                                #368 with the evidence

The runsheet (RUNSHEET.md next to this file) spells the decision tree out
with the arbitration protocol and the card budget.

WHAT IS MEASURED
================
Per (shape, M), eight lanes, each timed SEPARATELY so the split is visible
rather than inferred:

    bf16_linear         F.linear(x, w_bf16)              -- the unquantized floor
    int8_quant          per_token_quant_int8(x)          -- activation quant only
    int8_gemm           int8_scaled_mm(pre-quantized)    -- GEMM only
    int8_fused          quant + GEMM                     -- the serving path verbatim
    fp8_quant           sglang_per_token_quant_fp8(x)    -- FP8 act quant only
    fp8_gemm            fp8_scaled_mm(pre-quantized)     -- FP8 GEMM only
    fp8_ct_fused        apply_fp8_linear, compressed-tensors branch
    fp8_block_fused     w8a8_block_fp8_linear, the branch Fp8LinearMethod takes
                        for the deployed (block-quantized) Qwen3.6-27B-FP8

``int8_fused`` is not assumed to equal ``int8_quant + int8_gemm``: the
difference between the two is the launch-gap and allocator cost that only a
fused measurement shows, and that difference is exactly what a fusion would
recover. It is reported.

Two FP8 references, because they are genuinely different kernels. The
compressed-tensors FP8 scheme (per-token activation scale, per-channel weight
scale, CUTLASS scaled-mm) is the structural twin of the INT8 scheme -- same
shape of work, so it isolates "is INT8 arithmetic slower than FP8 arithmetic
here". The deployed Qwen3.6-27B-FP8 checkpoint is something else: it carries
``weight_block_size [128, 128]``, so ``Fp8LinearMethod`` takes its
``block_quant`` branch (fp8.py:1132) into ``w8a8_block_fp8_linear``, with a
group-128 activation quant and a block-scaled GEMM. That is the lane #354
measured end to end. Neither FP8 lane re-measures #354; they are the per-GEMM
context for it.

SHAPES
======
Derived from the checkpoint's own config.json and the shard plan, not
hardcoded. Under uneven TP the per-rank N differs per rank, so the shape set
is a function of (tp_size, --rank-tp-ratio, --rank-mlp-ratio, rank). The
partition arithmetic is the largest-remainder split the serving path uses;
when sglang is importable the derivation is cross-checked against
``sglang.srt.distributed.utils.partition_units`` and disagreement is fatal.

Only layers that are actually INT8 are included. The checkpoint's ignore
list keeps ``in_proj_b`` / ``in_proj_a`` (the GDN ba projection), ``lm_head``
and everything matching ``re:.*mtp.*`` in bf16 -- so the whole MTP draft
model is unquantized and contributes no INT8 GEMM at all.

TIMING
======
CUDA events around a burst of N iterations, burst size auto-calibrated to a
target wall time so short and long shapes get comparable statistical weight.
Lanes are INTERLEAVED: one burst per lane per round, same rotation every
round, so clock drift and thermal drift hit every lane equally instead of
accumulating in whichever lane ran last.

Every lane is instantiated TWICE, on independent tensors (``lane`` and
``lane#A2``), and both copies sit in the same rotation. The A-vs-A spread
between the two copies IS the noise floor at that operating point and is
reported next to every comparison. Distributions are reported as
median / p5 / p95 over rounds, never as means -- a mean over a burst
distribution with a launch-stall tail is not a number anyone can act on.

No percent-threshold stop rule is built in. The script reports the numbers;
the runsheet reports gain AND effort as a pair.

DRY RUN
=======
``--dry-run`` executes every code path on CPU with pure-torch stand-ins for
the two CUDA kernels, so the harness is proven to run end to end before a
card window is claimed. The full, real shape table is still derived and
emitted; only the tensors that are actually multiplied are capped
(``--max-dim``) so a CPU can carry them. Dry-run output is stamped
``"dry_run": true`` and every timing carries ``"stub": true`` -- it is path
coverage, never a measurement.

USAGE
=====
    # desk, no card:
    python3 scripts/int8_368/microbench.py --dry-run

    # card, inside a claimed arbitration window:
    CUDA_VISIBLE_DEVICES=<idx> python3 scripts/int8_368/microbench.py \
        --out /tmp/int8_368.<card>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import torch

# --------------------------------------------------------------------------
# Shard arithmetic
# --------------------------------------------------------------------------


def partition_units_local(units: int, weights: Sequence[int]) -> list:
    """Largest-remainder split of `units` over ranks proportional to
    `weights`, every rank >= 1 unit, ties toward the lower rank index.

    A deliberate second implementation of
    ``sglang.srt.distributed.utils._partition_units_raw`` so this script
    derives shapes without importing the serving stack (a desk machine
    without sgl_kernel can still print the table). When sglang IS
    importable, ``cross_check_partition`` compares the two and refuses to
    continue on disagreement -- a shape table that silently drifts from the
    serving split would be worse than no table.
    """
    n = len(weights)
    if units < n:
        raise ValueError(
            f"Cannot give each of {n} ranks at least one of {units} units."
        )
    total_w = sum(weights)
    quotas = [units * w / total_w for w in weights]
    sizes = [max(int(q), 1) for q in quotas]
    remaining = units - sum(sizes)
    if remaining < 0:
        for _ in range(-remaining):
            i = max(range(n), key=lambda r: (sizes[r], -r))
            sizes[i] -= 1
        remaining = 0
    order = sorted(
        range(n), key=lambda r: (quotas[r] - int(quotas[r]), -r), reverse=True
    )
    for k in range(remaining):
        order_i = order[k % n]
        sizes[order_i] += 1
    assert sum(sizes) == units and all(s >= 1 for s in sizes)
    return sizes


def cross_check_partition(cases: Sequence[tuple]) -> str:
    """Compare the local split against sglang's for every (units, weights)
    the shape derivation used. Returns a provenance string for the JSON."""
    try:
        from sglang.srt.distributed.utils import _partition_units_raw  # noqa: PLC0415
    except Exception as ex:  # sglang not importable on a bare desk machine
        return f"local-only ({type(ex).__name__}: {ex})"
    for units, weights in cases:
        mine = partition_units_local(units, weights)
        theirs = list(_partition_units_raw(units, list(weights)))
        if mine != theirs:
            raise SystemExit(
                f"Shard arithmetic disagrees with the serving stack for "
                f"units={units} weights={list(weights)}: local {mine} vs "
                f"sglang {theirs}. Refusing to emit a shape table that does "
                f"not match what the ranks would actually build."
            )
    return "cross-checked against sglang.srt.distributed.utils._partition_units_raw"


@dataclass
class Shape:
    """One linear layer's per-rank GEMM, in serving orientation.

    ``n`` is the output width (weight rows before the transpose), ``k`` the
    reduction dim. The INT8 scheme stores ``layer.weight`` transposed, so the
    kernel sees ``(k, n)`` column-major -- built that way below.
    """

    name: str
    n: int
    k: int
    module: str
    layers: int  # how many layers of the model carry this GEMM
    note: str = ""
    plans: list = field(default_factory=list)  # which shard plans produce it
    exec_n: Optional[int] = None  # capped size actually multiplied (dry run)
    exec_k: Optional[int] = None

    @property
    def run_n(self) -> int:
        return self.exec_n if self.exec_n is not None else self.n

    @property
    def run_k(self) -> int:
        return self.exec_k if self.exec_k is not None else self.k

    @property
    def capped(self) -> bool:
        return self.exec_n is not None or self.exec_k is not None


#: Two shard-plan vectors are deployed, and the final INT8-vs-FP8 comparison
#: uses BOTH: prefill runs on the phase-optimal vector (#354's --rank-mlp-ratio,
#: the 16,1,1 class, which loads the 5090 with almost the whole MLP), decode
#: runs on the auto vector. A shape table for one vector only would price half
#: the deployment.
DEFAULT_PLANS = "auto=30,17,17;prefopt=16,1,1"

#: The three shapes #255 queued for the sm120 FP8 Triton tuner
#: (docs/rig-runbook.md, "the three shapes #255 left queued for sm120").
#: Carried so the INT8 numbers land on the operating points the FP8 tuner was
#: measured at. They correspond to a rank with 12 q heads / 2 kv heads (qkv
#: 7168x5120, o_proj 5120x3072) and a GDN rank with 21 value heads (out_proj
#: 5120x2688) -- the last one belongs to an older auto vector that neither
#: plan above reproduces.
#:
#: The MLP pair below is the one #370's idle-tuner queue targets. It is the
#: prefopt vector seen through the FP8 checkpoint's UNIT FAMILY, which is not
#: the INT8 one:
#:
#:   Qwen3.6-27B-FP8 carries weight_block_size [128,128], so
#:   _quant_block_aligned_units coarsens the dense MLP to 17408/128 = 136
#:   units. partition_units(136, [16,1,1]) = [121, 8, 7] -> rank 0 holds
#:   121*128 = 15488, gate_up = 30976. Those are the numbers in the FP8 boot
#:   logs and in IDLE_TUNER_QUEUE_370.md.
#:
#:   Qwen3.6-27B-INT8-W8A8 is channel/token quantized with no weight block, so
#:   its family is 17408/gcd(17408,16) = 1088 units of 16.
#:   partition_units(1088, [16,1,1]) = [967, 61, 60] -> rank 0 holds 15472,
#:   gate_up = 30944.
#:
#: Same vector, different unit family, 16 elements apart. Both are real
#: deployed shapes; the derived prefopt rows below are the INT8 ones, and
#: these two reference rows are the FP8 ones, kept so the two checkpoints can
#: be read against each other at the shapes each actually runs.
REFERENCE_SHAPES = [
    ("ref255_qkv", 7168, 5120, "#255 FP8 tuner queue (older auto vector)"),
    ("ref255_gdn_out", 5120, 2688, "#255 FP8 tuner queue (older auto vector)"),
    ("ref255_o_proj", 5120, 3072, "#255 FP8 tuner queue (older auto vector)"),
    (
        "ref370_fp8block_mlp_gate_up",
        30976,
        5120,
        "#370 idle-tuner queue: prefopt 16,1,1 x FP8 128-block unit family",
    ),
    (
        "ref370_fp8block_mlp_down",
        5120,
        15488,
        "#370 idle-tuner queue: prefopt 16,1,1 x FP8 128-block unit family",
    ),
]


#: #855 shape set, taken from ANALYSE_854 rather than re-derived: the worked
#: 12:10:10 example (ANALYSE_854 4.1, docs/dev/ANALYSE_854_w8a16_vs_w8a8.md:
#: 372-386) and the un-sharded projections its 9 step 0 names. The W8A16
#: lane coarsens shards to 128 (not the INT8 lane's 16), which is why every
#: K below is 128-aligned -- the two schemes do not share a shard table, so
#: deriving these from the INT8 unit family would have priced the wrong
#: shapes.
SHAPES_855 = [
    # ratio 12:10:10, intermediate 17408 = 136 units of 128 -> 51/43/42
    ("m855_gate_up_r0", 13056, 5120, "12:10:10 rank0 gate_up (2 x 6528)"),
    ("m855_down_r0", 5120, 6528, "12:10:10 rank0 down_proj (K-shard)"),
    ("m855_down_r1", 5120, 5504, "12:10:10 rank1 down_proj (K-shard)"),
    # o_proj / GDN out_proj, K=6144 = 48 units -> 18/15/15
    ("m855_out_proj_r0", 5120, 2304, "12:10:10 rank0 o_proj / GDN out_proj"),
    ("m855_out_proj_r1", 5120, 1920, "12:10:10 rank1 o_proj / GDN out_proj"),
    # un-sharded (TP=1) projections listed in ANALYSE_854 9 step 0
    ("m855_full_gate_up_out", 17408, 5120, "ANALYSE_854 9.0 un-sharded"),
    ("m855_full_down", 5120, 17408, "ANALYSE_854 9.0 un-sharded"),
    ("m855_full_qkv", 12288, 5120, "ANALYSE_854 9.0 un-sharded"),
    ("m855_full_gdn_out", 5120, 6144, "ANALYSE_854 9.0 un-sharded"),
    ("m855_full_10240", 10240, 5120, "ANALYSE_854 9.0 un-sharded"),
]

#: The minimum decision set: one large sharded GEMM, one mid, one small.
#: Deliberately not a ladder -- the verdict question (does Marlin lose enough
#: at the deployed operating points to veto W8A16) is coarse, and three
#: shapes that separate cleanly from the A-vs-A floor answer it.
SHAPES_855_MIN = ["m855_gate_up_r0", "m855_down_r0", "m855_out_proj_r0"]

SHAPE_PRESETS = {
    "none": [],
    "855": [s[0] for s in SHAPES_855],
    "855min": SHAPES_855_MIN,
}


def derive_shapes(
    cfg: dict,
    tp_size: int,
    ratio: Sequence[int],
    mlp_ratio: Sequence[int],
    rank: int,
) -> tuple[list, list, dict]:
    """Per-rank INT8 GEMM shapes for a Qwen3.5/3.6 dense text model.

    Mirrors, module by module:
      * ``Qwen3_5Attention``  (models/qwen3_5.py:928 qkv_proj, :947 o_proj)
      * ``Qwen3_5GatedDeltaNet`` (:252 in_proj_qkvz, :347 out_proj)
      * ``Qwen2MoeMLP`` (models/qwen2_moe.py:226 gate_up_proj, :~250 down_proj)

    Returns (shapes, partition_cases, facts).
    """
    t = cfg.get("text_config", cfg)
    hidden = int(t["hidden_size"])
    n_layers = int(t["num_hidden_layers"])
    inter = int(t["intermediate_size"])
    n_heads = int(t["num_attention_heads"])
    n_kv = int(t["num_key_value_heads"])
    head_dim = int(t.get("head_dim") or hidden // n_heads)
    gate = 2 if t.get("attn_output_gate", False) else 1

    k_heads = int(t["linear_num_key_heads"])
    v_heads = int(t["linear_num_value_heads"])
    head_k = int(t["linear_key_head_dim"])
    head_v = int(t["linear_value_head_dim"])

    layer_types = t.get("layer_types") or []
    if layer_types:
        n_full = sum(1 for x in layer_types if x == "full_attention")
        n_linear = sum(1 for x in layer_types if x == "linear_attention")
    else:
        interval = int(t.get("full_attention_interval", 1))
        n_full = n_layers // interval
        n_linear = n_layers - n_full

    uniform = len(set(ratio)) == 1
    cases: list = []

    def split(units: int, weights: Sequence[int]) -> int:
        if tp_size == 1:
            return units
        cases.append((units, tuple(weights)))
        return partition_units_local(units, weights)[rank]

    if tp_size > 1 and not uniform:
        # kv >= tp: kv heads are the indivisible unit for the q dimension too
        # (attn_q_partition_units), and groups is None (no REPLICATED-KV).
        if n_kv < tp_size:
            raise SystemExit(
                f"num_key_value_heads ({n_kv}) < tp_size ({tp_size}): that is "
                "the REPLICATED-KV geometry, whose q split carries a "
                "kv-boundary alignment constraint this script does not "
                "reimplement. Derive the shapes from a live rank instead."
            )
        loc_kv = split(n_kv, ratio)
        loc_q = split(n_kv, ratio) * (n_heads // n_kv)
        loc_kh = split(k_heads, ratio)
        loc_vh = split(k_heads, ratio) * (v_heads // k_heads)
        mlp_units = inter // math.gcd(inter, 16)
        loc_inter = partition_units_local(mlp_units, mlp_ratio)[rank] * (
            inter // mlp_units
        )
        if tp_size != len(mlp_ratio):
            raise SystemExit("--rank-mlp-ratio length must equal --tp-size")
        cases.append((mlp_units, tuple(mlp_ratio)))
    else:
        for total, name in (
            (n_heads, "q"),
            (n_kv, "kv"),
            (k_heads, "gdn-k"),
            (inter, "mlp"),
        ):
            if total % tp_size:
                raise SystemExit(
                    f"Even TP={tp_size} does not divide the {name} dimension "
                    f"({total}). Pass a --ratio for the uneven plan."
                )
        loc_kv = n_kv // tp_size
        loc_q = n_heads // tp_size
        loc_kh = k_heads // tp_size
        loc_vh = v_heads // tp_size
        loc_inter = inter // tp_size

    shapes = [
        Shape(
            "attn_qkv",
            gate * loc_q * head_dim + 2 * loc_kv * head_dim,
            hidden,
            "Qwen3_5Attention.qkv_proj",
            n_full,
            f"{loc_q} q heads x{gate} (output gate) + 2x{loc_kv} kv heads, head_dim {head_dim}",
        ),
        Shape(
            "attn_o",
            hidden,
            loc_q * head_dim,
            "Qwen3_5Attention.o_proj",
            n_full,
            "row-parallel, input = local q width",
        ),
        Shape(
            "gdn_in_qkvz",
            2 * loc_kh * head_k + 2 * loc_vh * head_v,
            hidden,
            "Qwen3_5GatedDeltaNet.in_proj_qkvz",
            n_linear,
            f"[q,k,z,v] merged: 2x{loc_kh}x{head_k} + 2x{loc_vh}x{head_v}",
        ),
        Shape(
            "gdn_out",
            hidden,
            loc_vh * head_v,
            "Qwen3_5GatedDeltaNet.out_proj",
            n_linear,
            "row-parallel, input = local value width",
        ),
        Shape(
            "mlp_gate_up",
            2 * loc_inter,
            hidden,
            "Qwen2MoeMLP.gate_up_proj",
            n_layers,
            f"gate+up merged, local intermediate {loc_inter}",
        ),
        Shape(
            "mlp_down",
            hidden,
            loc_inter,
            "Qwen2MoeMLP.down_proj",
            n_layers,
            "row-parallel",
        ),
    ]

    facts = {
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "intermediate_size": inter,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "attn_output_gate": bool(t.get("attn_output_gate", False)),
        "linear_num_key_heads": k_heads,
        "linear_num_value_heads": v_heads,
        "full_attention_layers": n_full,
        "linear_attention_layers": n_linear,
        "local_q_heads": loc_q,
        "local_kv_heads": loc_kv,
        "local_gdn_k_heads": loc_kh,
        "local_gdn_v_heads": loc_vh,
        "local_intermediate": loc_inter,
        "not_quantized_by_ignore_list": [
            "linear_attn.in_proj_b / in_proj_a (the merged in_proj_ba stays bf16)",
            "lm_head",
            "re:.*mtp.* -- the whole MTP draft model runs unquantized",
        ],
    }
    return shapes, cases, facts


# --------------------------------------------------------------------------
# Kernels: real ones on a card, pure-torch stand-ins for the dry run
# --------------------------------------------------------------------------


@dataclass
class Kernels:
    per_token_quant_int8: Callable
    int8_scaled_mm: Callable
    per_token_quant_fp8: Optional[Callable]
    fp8_scaled_mm: Optional[Callable]
    apply_fp8_linear: Optional[Callable]
    block_fp8_linear: Optional[Callable]
    fp8_dtype: torch.dtype
    stub: bool
    missing: list = field(default_factory=list)
    # --- #855 Marlin wNa16 arm (weight-only int8, group 128) ---------------
    # Resolved through the SERVING path's own helpers, not a private copy:
    # CompressedTensorsWNA16.apply_weights calls apply_gptq_marlin_linear
    # (schemes/compressed_tensors_wNa16.py:327), and the repack that produces
    # the kernel's weight layout is the same gptq_marlin_repack the scheme
    # runs in process_weights_after_loading (wNa16.py:250).
    apply_gptq_marlin_linear: Optional[Callable] = None
    gptq_marlin_repack: Optional[Callable] = None
    marlin_make_workspace: Optional[Callable] = None
    marlin_permute_scales: Optional[Callable] = None
    marlin_make_empty_g_idx: Optional[Callable] = None
    marlin_wtype: object = None


def _stub_per_token_quant_int8(x: torch.Tensor):
    x32 = x.to(torch.float32)
    absmax = x32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127.0
    return torch.round(x32 * (127.0 / absmax)).to(torch.int8), scale


def _stub_int8_scaled_mm(a, b, sa, sb, out_dtype, bias=None):
    acc = a.to(torch.float32) @ b.to(torch.float32)
    acc = acc * sa.view(-1, 1).to(torch.float32) * sb.view(1, -1).to(torch.float32)
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    return acc.to(out_dtype)


def _stub_per_token_quant_fp8(x: torch.Tensor, fp8_dtype: torch.dtype):
    x32 = x.to(torch.float32)
    finfo_max = 448.0 if fp8_dtype == torch.float8_e4m3fn else 57344.0
    absmax = x32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = absmax / finfo_max
    return (x32 / scale).clamp(-finfo_max, finfo_max).to(fp8_dtype), scale


def _stub_fp8_scaled_mm(a, b, sa, sb, out_dtype, bias=None):
    return _stub_int8_scaled_mm(a, b, sa, sb, out_dtype, bias)


def _stub_block_fp8_linear(
    input, weight, block_size, weight_scale, input_scale=None, bias=None
):
    """Stand-in for the block-wise fp8 linear. Weight is (n, k) here -- the
    block path does NOT pre-transpose, unlike the per-channel one."""
    acc = input.to(torch.float32) @ weight.to(torch.float32).t()
    acc = acc * float(weight_scale.mean())
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    return acc.to(input.dtype)


#: Block-FP8 backends selectable with --fp8-block-backend. "auto" is what a
#: bare process resolves through dispatch_w8a8_block_fp8_linear, i.e. what a
#: server WITHOUT an explicit --fp8-gemm-backend would pick for this card.
#: On sm120 that is flashinfer_gemm_..._with_fallback, whose CUTLASS kernel
#: prints "Arch conditional MMA instruction used without targeting
#: appropriate compute capability" on every launch (measured 2026-08-01,
#: millions of lines per minute) -- so the lane is pinned explicitly for a
#: measurement and the auto verdict is recorded as a note, not benched.
BLOCK_FP8_BACKENDS = {
    "triton": "triton_w8a8_block_fp8_linear",
    "cutlass": "cutlass_w8a8_block_fp8_linear_with_fallback",
    "deepgemm": "deepgemm_w8a8_block_fp8_linear_with_fallback",
    "flashinfer": "flashinfer_gemm_w8a8_block_fp8_linear_with_fallback",
}


def load_kernels(dry_run: bool, block_backend: str = "auto") -> Kernels:
    fp8_dtype = torch.float8_e4m3fn
    if dry_run:
        try:
            torch.zeros(2, dtype=torch.bfloat16).to(fp8_dtype)
        except Exception:
            # A CPU build without float8 conversion: the stub lane still has
            # to execute, so it runs on bf16 storage. Stamped in the JSON.
            fp8_dtype = torch.bfloat16
        return Kernels(
            per_token_quant_int8=_stub_per_token_quant_int8,
            int8_scaled_mm=_stub_int8_scaled_mm,
            per_token_quant_fp8=lambda x: _stub_per_token_quant_fp8(x, fp8_dtype),
            fp8_scaled_mm=_stub_fp8_scaled_mm,
            apply_fp8_linear=None,  # replaced by a stub closure in build_lanes
            block_fp8_linear=_stub_block_fp8_linear,
            fp8_dtype=fp8_dtype,
            stub=True,
        )

    missing = []
    from sglang.srt.layers.quantization.int8_kernel import (  # noqa: PLC0415
        per_token_quant_int8,
    )
    from sgl_kernel import int8_scaled_mm  # noqa: PLC0415

    try:
        from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: PLC0415
            sglang_per_token_quant_fp8,
        )
    except Exception as ex:
        sglang_per_token_quant_fp8 = None
        missing.append(f"sglang_per_token_quant_fp8: {type(ex).__name__}: {ex}")
    try:
        from sgl_kernel import fp8_scaled_mm  # noqa: PLC0415
    except Exception as ex:
        fp8_scaled_mm = None
        missing.append(f"fp8_scaled_mm: {type(ex).__name__}: {ex}")
    try:
        from sglang.srt.layers.quantization.fp8_utils import (  # noqa: PLC0415
            apply_fp8_linear,
        )
    except Exception as ex:
        apply_fp8_linear = None
        missing.append(f"apply_fp8_linear: {type(ex).__name__}: {ex}")
    try:
        # Same dispatcher Fp8LinearMethod uses for a block-quantized
        # checkpoint (fp8.py:1132 `if self.block_quant:` ->
        # self.w8a8_block_fp8_linear). Resolved for the current device, so it
        # must be called after the device is selected.
        from sglang.srt.layers.quantization import fp8_utils  # noqa: PLC0415

        if block_backend == "auto":
            block_fp8_linear = fp8_utils.dispatch_w8a8_block_fp8_linear()
        else:
            block_fp8_linear = getattr(fp8_utils, BLOCK_FP8_BACKENDS[block_backend])
    except Exception as ex:
        block_fp8_linear = None
        missing.append(f"dispatch_w8a8_block_fp8_linear: {type(ex).__name__}: {ex}")

    # #855: the Marlin wNa16 arm. In-tree JIT (sglang.jit_kernel.gptq_marlin),
    # NOT an sgl_kernel symbol -- so unlike the INT8 arm it cannot be lost to
    # the #384 wheel swap. Missing here means the JIT could not build.
    marlin: dict = {}
    try:
        from sglang.jit_kernel.gptq_marlin_repack import (  # noqa: PLC0415
            gptq_marlin_repack,
        )
        from sglang.srt.layers.quantization.marlin_utils import (  # noqa: PLC0415
            apply_gptq_marlin_linear,
            marlin_make_empty_g_idx,
            marlin_make_workspace,
            marlin_permute_scales,
        )
        from sglang.srt.layers.quantization.utils import (  # noqa: PLC0415
            get_scalar_types,
        )

        _, _scalar_types = get_scalar_types()
        marlin = dict(
            apply_gptq_marlin_linear=apply_gptq_marlin_linear,
            gptq_marlin_repack=gptq_marlin_repack,
            marlin_make_workspace=marlin_make_workspace,
            marlin_permute_scales=marlin_permute_scales,
            marlin_make_empty_g_idx=marlin_make_empty_g_idx,
            # WNA16_SUPPORTED_TYPES_MAP[8] (wNa16.py:56): 8-bit symmetric,
            # the type a W8A16 compressed-tensors checkpoint resolves to.
            marlin_wtype=_scalar_types.uint8b128,
        )
    except Exception as ex:
        missing.append(f"marlin_wna16: {type(ex).__name__}: {ex}")

    return Kernels(
        per_token_quant_int8=per_token_quant_int8,
        int8_scaled_mm=int8_scaled_mm,
        per_token_quant_fp8=sglang_per_token_quant_fp8,
        fp8_scaled_mm=fp8_scaled_mm,
        apply_fp8_linear=apply_fp8_linear,
        block_fp8_linear=block_fp8_linear,
        fp8_dtype=fp8_dtype,
        stub=False,
        missing=missing,
        **marlin,
    )


# --------------------------------------------------------------------------
# Lane construction
# --------------------------------------------------------------------------

ALL_LANES = [
    "bf16_linear",
    "int8_quant",
    "int8_gemm",
    "int8_fused",
    "fp8_quant",
    "fp8_gemm",
    "fp8_ct_fused",
    "fp8_block_fused",
]
#: #855. Opt-in only (`--lanes ...,+marlin_wna16`), so every #368 default
#: invocation keeps its exact lane set and remains comparable to the recorded
#: #368 results.
OPTIONAL_LANES: list = ["marlin_wna16"]

#: Qwen3.6-27B-FP8's ``weight_block_size``. Also the block the #255/#370
#: idle-tuner queue tunes for (``--block-n 128 --block-k 128``).
FP8_BLOCK = (128, 128)

#: #855: the W8A16 candidates surveyed in ANALYSE_854 3 are group-128
#: pack-quantized, and 128 is also what the wNa16 lane coarsens uneven-TP
#: shards to (ANALYSE_854 4.1).
MARLIN_GROUP = 128


@dataclass
class Weights:
    """One independent weight set for a shape. Two of these give the A-vs-A
    pair; they are built separately so the pair also carries the
    allocator/address component of the noise, not only the clock's. Built
    ONCE per shape and reused across every M -- rebuilding a 16320x5120
    weight per M would put more wall time into torch.randn than into the
    kernels being measured."""

    w_bf16: torch.Tensor
    w_i8_t: torch.Tensor
    ws_i8: torch.Tensor
    w_f8_t: Optional[torch.Tensor]
    ws_f8_chan: Optional[torch.Tensor]
    w_f8_blk: Optional[torch.Tensor]  # (n, k), NOT transposed
    ws_f8_blk: Optional[torch.Tensor]  # (ceil(n/128), ceil(k/128)) float32
    # --- #855 Marlin wNa16 (weight-only int8, group 128) ------------------
    w_marlin: Optional[torch.Tensor] = None  # gptq_marlin_repack output
    ws_marlin: Optional[torch.Tensor] = None  # marlin_permute_scales output
    marlin_workspace: Optional[torch.Tensor] = None
    marlin_empty: Optional[torch.Tensor] = None  # g_idx / sort_indices / zp
    marlin_note: str = ""


@dataclass
class Operand:
    """A weight set plus the activations for one M."""

    w: Weights
    x: torch.Tensor
    xq_i8: torch.Tensor
    xs_i8: torch.Tensor
    xq_f8: Optional[torch.Tensor]
    xs_f8: Optional[torch.Tensor]


def build_weights(shape: Shape, dev: torch.device, kn: Kernels, seed: int) -> Weights:
    n, k = shape.run_n, shape.run_k
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Sampled on CPU and moved: on-GPU randn is not architecture-identical
    # across sm86/sm120, and the two cards must see the same input bytes.
    w_bf16 = (
        torch.randn(n, k, generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    )
    # INT8 weight in serving layout: the scheme stores weight.t(), so the
    # kernel's mat_b is (k, n) with stride(0) == 1.
    w_i8 = torch.randint(-127, 127, (n, k), generator=g, dtype=torch.int8).to(dev)
    ws_i8 = torch.rand(n, 1, generator=g, dtype=torch.float32).to(dev).add_(0.01)

    w_f8_t = ws_f8_chan = w_f8_blk = ws_f8_blk = None
    if kn.per_token_quant_fp8 is not None:
        w_f8_t = w_bf16.to(kn.fp8_dtype).t()
        ws_f8_chan = (
            torch.rand(n, 1, generator=g, dtype=torch.float32).to(dev).add_(0.01)
        )
    if kn.block_fp8_linear is not None:
        # Block-quantized layout: weight stays (n, k) and the scale is one
        # value per 128x128 tile (Qwen3.6-27B-FP8's weight_block_size).
        w_f8_blk = w_bf16.to(kn.fp8_dtype)
        bn, bk = FP8_BLOCK
        ws_f8_blk = (
            torch.rand(
                (n + bn - 1) // bn,
                (k + bk - 1) // bk,
                generator=g,
                dtype=torch.float32,
            )
            .to(dev)
            .add_(0.01)
        )
    w_marlin = ws_marlin = marlin_ws = marlin_empty = None
    marlin_note = ""
    if kn.apply_gptq_marlin_linear is not None:
        # Marlin's own shape rules, checked here rather than crashed in CUDA:
        # GPTQ_MARLIN_MIN_THREAD_K = 128 / MIN_THREAD_N = 64 (marlin_utils.py:
        # 57, verify_marlin_supports_shape at :191-208), and a group of 128
        # must divide K. Under uneven TP the W8A16 lane coarsens every shard
        # to 128 for exactly this reason (linear.py:361-362, ANALYSE_854 4.1),
        # so a real deployed shard always passes; a shard that does not is a
        # finding, not a harness limit.
        if k % MARLIN_GROUP or n % 64:
            marlin_note = (
                f"skipped: K={k} must be %{MARLIN_GROUP} and N={n} must be %64"
            )
        else:
            pack_factor = 32 // 8
            # GPTQ-serialized layout the scheme hands to gptq_marlin_repack:
            # (K // pack_factor, N) int32, K-packed. Random bits are a valid
            # weight -- Marlin's cost does not depend on the values.
            w_gptq = torch.randint(
                -(2**31),
                2**31 - 1,
                (k // pack_factor, n),
                generator=g,
                dtype=torch.int32,
            ).to(dev)
            marlin_empty = kn.marlin_make_empty_g_idx(dev)
            w_marlin = kn.gptq_marlin_repack(
                w_gptq.contiguous(),
                perm=marlin_empty,
                size_k=k,
                size_n=n,
                num_bits=8,
            )
            s_raw = (
                torch.rand(k // MARLIN_GROUP, n, generator=g, dtype=torch.float32)
                .mul_(0.01)
                .add_(0.001)
                .to(dev)
                .to(torch.bfloat16)
            )
            ws_marlin = kn.marlin_permute_scales(
                s_raw.contiguous(), size_k=k, size_n=n, group_size=MARLIN_GROUP
            )
            marlin_ws = kn.marlin_make_workspace(dev)
            del w_gptq

    return Weights(
        w_bf16,
        w_i8.t(),
        ws_i8,
        w_f8_t,
        ws_f8_chan,
        w_f8_blk,
        ws_f8_blk,
        w_marlin=w_marlin,
        ws_marlin=ws_marlin,
        marlin_workspace=marlin_ws,
        marlin_empty=marlin_empty,
        marlin_note=marlin_note,
    )


def build_operand(
    w: Weights, shape: Shape, m: int, dev: torch.device, kn: Kernels, seed: int
) -> Operand:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (
        torch.randn(m, shape.run_k, generator=g, dtype=torch.float32)
        .to(dev)
        .to(torch.bfloat16)
    )
    xq_i8, xs_i8 = kn.per_token_quant_int8(x)
    xq_f8 = xs_f8 = None
    if kn.per_token_quant_fp8 is not None:
        xq_f8, xs_f8 = kn.per_token_quant_fp8(x)
    return Operand(w, x, xq_i8, xs_i8, xq_f8, xs_f8)


def build_lanes(op: Operand, kn: Kernels, want: Sequence[str]) -> dict:
    """name -> zero-arg callable. Only lanes whose kernels exist are built."""
    out: dict = {}
    dt = op.x.dtype

    if "bf16_linear" in want:
        out["bf16_linear"] = lambda: torch.nn.functional.linear(op.x, op.w.w_bf16)
    if "int8_quant" in want:
        out["int8_quant"] = lambda: kn.per_token_quant_int8(op.x)
    if "int8_gemm" in want:
        out["int8_gemm"] = lambda: kn.int8_scaled_mm(
            op.xq_i8, op.w.w_i8_t, op.xs_i8, op.w.ws_i8, out_dtype=dt, bias=None
        )

    if "int8_fused" in want:

        def _int8_fused():
            # Verbatim CompressedTensorsW8A8Int8.apply_weights body
            # (compressed_tensors_w8a8_int8.py:213-217); bias is None because
            # every quantized linear in this checkpoint is bias-free.
            x_q, x_scale = kn.per_token_quant_int8(op.x)
            return kn.int8_scaled_mm(
                x_q, op.w.w_i8_t, x_scale, op.w.ws_i8, out_dtype=dt, bias=None
            )

        out["int8_fused"] = _int8_fused

    if (
        "marlin_wna16" in want
        and kn.apply_gptq_marlin_linear is not None
        and op.w.w_marlin is not None
    ):
        # Verbatim CompressedTensorsWNA16.apply_weights body
        # (schemes/compressed_tensors_wNa16.py:311-341): no activation quant
        # exists on this path at all -- that is the whole structural claim of
        # W8A16, and it is why this lane is compared against int8_FUSED (quant
        # + GEMM), which is the complete serving op on the W8A8 side.
        # is_k_full=True: marlin_is_k_full(has_g_idx=False, ...) returns True
        # for an act-order-free checkpoint regardless of row-parallelism
        # (marlin_utils.py, marlin_is_k_full).
        out["marlin_wna16"] = lambda: kn.apply_gptq_marlin_linear(
            input=op.x,
            weight=op.w.w_marlin,
            weight_scale=op.w.ws_marlin,
            weight_zp=op.w.marlin_empty,
            g_idx=op.w.marlin_empty,
            g_idx_sort_indices=op.w.marlin_empty,
            workspace=op.w.marlin_workspace,
            wtype=kn.marlin_wtype,
            output_size_per_partition=op.w.w_bf16.shape[0],
            input_size_per_partition=op.w.w_bf16.shape[1],
            is_k_full=True,
            bias=None,
        )

    if kn.per_token_quant_fp8 is not None:
        if "fp8_quant" in want:
            out["fp8_quant"] = lambda: kn.per_token_quant_fp8(op.x)
        if "fp8_gemm" in want and kn.fp8_scaled_mm is not None:
            out["fp8_gemm"] = lambda: kn.fp8_scaled_mm(
                op.xq_f8,
                op.w.w_f8_t,
                op.xs_f8,
                op.w.ws_f8_chan,
                out_dtype=dt,
                bias=None,
            )
        if "fp8_ct_fused" in want:
            if kn.apply_fp8_linear is not None:
                out["fp8_ct_fused"] = lambda: kn.apply_fp8_linear(
                    input=op.x,
                    weight=op.w.w_f8_t,
                    weight_scale=op.w.ws_f8_chan,
                    input_scale=None,
                    bias=None,
                    use_per_token_if_dynamic=True,
                    compressed_tensor_quant=True,
                )
            elif kn.fp8_scaled_mm is not None:

                def _fp8_stub_fused():
                    q, s = kn.per_token_quant_fp8(op.x)
                    return kn.fp8_scaled_mm(
                        q, op.w.w_f8_t, s, op.w.ws_f8_chan, out_dtype=dt, bias=None
                    )

                out["fp8_ct_fused"] = _fp8_stub_fused
    # The block-fp8 path quantizes activations in groups of block_k and
    # asserts K % block_k == 0 (fp8_kernel.py:649). It therefore CANNOT run a
    # shard whose K is not 128-aligned -- e.g. the auto vector's mlp_down
    # K=8160 under the INT8 16-element unit family. That is not a harness
    # limitation but the reason the FP8 checkpoint coarsens the same
    # dimension to 128-element units (8192) in the first place. Skipped, not
    # crashed, and its absence at those shapes is itself the finding.
    block_ok = op.x.shape[-1] % FP8_BLOCK[1] == 0
    if "fp8_block_fused" in want and kn.block_fp8_linear is not None and block_ok:
        # The lane the deployed Qwen3.6-27B-FP8 checkpoint actually runs.
        # Its quantization_config carries weight_block_size [128, 128], so
        # Fp8LinearMethod takes the `if self.block_quant:` branch
        # (fp8.py:1132) into w8a8_block_fp8_linear and NEVER reaches
        # apply_fp8_linear. Different activation quant
        # (per_token_group_quant_fp8 at group 128, not per-token) and a
        # different GEMM -- so the per-channel lanes above are the structural
        # twin of INT8, and this one is what #354 measured end to end.
        out["fp8_block_fused"] = lambda: kn.block_fp8_linear(
            input=op.x,
            weight=op.w.w_f8_blk,
            block_size=list(FP8_BLOCK),
            weight_scale=op.w.ws_f8_blk,
            input_scale=None,
            bias=None,
        )
    return out


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def time_burst(fn: Callable, iters: int, cuda: bool) -> float:
    """Mean ms per iteration over one uninterrupted burst.

    A burst is the atom; the DISTRIBUTION is taken over bursts (rounds), not
    over single iterations -- single-iteration CUDA-event pairs measure the
    event overhead as much as the kernel at these sizes.
    """
    if cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def calibrate(fn: Callable, target_ms: float, cuda: bool, max_iters: int) -> int:
    for _ in range(3):
        fn()
    if cuda:
        torch.cuda.synchronize()
    probe = max(time_burst(fn, 3, cuda), 1e-4)
    return max(1, min(max_iters, int(target_ms / probe)))


def capture_graph(fn: Callable, iters: int):
    """A CUDA graph containing `iters` back-to-back calls of `fn`.

    THE point of the whole graph mode: the eager measurement prices a launch
    per kernel, and decode in this stack does not pay that -- it replays a
    captured graph. Timing one replay of `iters` bodies and dividing gives
    the per-op cost with the launch amortized to a graph-node dispatch,
    which is the number a fusion decision must be made against. The eager
    number alone would credit a fusion with removing a cost the graph has
    already removed.

    `iters` bodies rather than one: a single-body graph would measure the
    replay call's own overhead as much as the work.

    Warmup happens on a side stream, which the capture API requires -- the
    first calls also JIT the Triton quant kernel and pick the CUTLASS tile,
    and neither may happen inside the capture.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    captured = []
    with torch.cuda.graph(graph):
        for _ in range(iters):
            captured.append(fn())
    torch.cuda.synchronize()

    # #591 falsifier ("the bench could not tell an empty CUDA graph from a
    # fast one"): a capture that recorded no work replays in microseconds and
    # reports as the fastest lane in the table. Prove the replay WRITES:
    # zero the captured output, replay, require it to come back non-zero.
    # Inputs are random bf16, so an all-zero result is not a legal outcome.
    probe = _first_tensor(captured[-1]) if captured else None
    if probe is None or probe.numel() == 0:
        raise RuntimeError("graph capture produced no inspectable output tensor")
    probe.zero_()
    graph.replay()
    torch.cuda.synchronize()
    if not bool(torch.any(probe != 0).item()):
        raise RuntimeError(
            "captured graph replayed without writing its output -- empty or "
            "no-op capture, refusing to time it (#591)"
        )
    del captured
    return graph


def _first_tensor(obj):
    """First tensor in a lane's return value (lanes return a tensor or a
    (quantized, scale) tuple)."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def summarize(samples: Sequence[float]) -> dict:
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": statistics.median(s),
        "p5_ms": s[max(0, int(0.05 * (len(s) - 1)))],
        "p95_ms": s[min(len(s) - 1, int(math.ceil(0.95 * (len(s) - 1))))],
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def run_point(
    shape: Shape,
    m: int,
    w_a: Weights,
    w_b: Weights,
    dev: torch.device,
    kn: Kernels,
    want: Sequence[str],
    rounds: int,
    target_ms: float,
    max_iters: int,
    cuda: bool,
    seed: int,
    graph_iters: int = 0,
) -> dict:
    op_a = build_operand(w_a, shape, m, dev, kn, seed)
    op_b = build_operand(w_b, shape, m, dev, kn, seed + 977)
    lanes_a = build_lanes(op_a, kn, want)
    lanes_b = build_lanes(op_b, kn, want)

    # Probe every lane ONCE before it enters the rotation. A lane can be
    # constructible and still not runnable on this card -- measured on sm86,
    # where the triton block-fp8 matmul rejects fp8e4nv ("not supported in
    # this architecture", Ampere has only fp8e4b15/fp8e5) and took the whole
    # battery down with it. An unrunnable lane is a RESULT about the card,
    # so it is recorded by name and reason and the rest of the run proceeds.
    rotation = []
    lane_failures: dict = {}
    for name in want:
        if name not in lanes_a:
            continue
        try:
            lanes_a[name]()
            lanes_b[name]()
            if cuda:
                torch.cuda.synchronize()
        except Exception as ex:
            lane_failures[name] = f"{type(ex).__name__}: {str(ex)[:220]}"
            continue
        rotation.append((name, lanes_a[name]))
        rotation.append((name + "#A2", lanes_b[name]))
    if not rotation:
        return {
            "skipped": "no lane could run at this point",
            "lane_failures": lane_failures,
        }

    # Graph variants join the SAME rotation, so eager and replay are
    # interleaved against one clock and one thermal state -- the comparison
    # between them is the deliverable, and measuring them in separate passes
    # would put the drift straight into it.
    graphs: dict = {}
    graph_notes: dict = {}
    if graph_iters and cuda:
        for name, fn in list(rotation):
            try:
                g = capture_graph(fn, graph_iters)
            except Exception as ex:
                graph_notes[name] = f"{type(ex).__name__}: {str(ex)[:160]}"
                continue
            graphs[name] = g
            rotation.append((name + "@graph", g.replay))

    iters = {name: calibrate(fn, target_ms, cuda, max_iters) for name, fn in rotation}
    samples: dict = {name: [] for name, _ in rotation}
    for _ in range(rounds):
        for name, fn in rotation:
            samples[name].append(time_burst(fn, iters[name], cuda))

    lanes: dict = {}
    for name, _ in rotation:
        scale = graph_iters if name.endswith("@graph") else 1
        # One replay executes graph_iters bodies; report per-OP so graph and
        # eager rows are the same unit.
        lanes[name] = summarize([s / scale for s in samples[name]])
        lanes[name]["iters_per_burst"] = iters[name]
        if scale != 1:
            lanes[name]["ops_per_replay"] = graph_iters
        if kn.stub:
            lanes[name]["stub"] = True

    noise: dict = {}
    for name in want:
        if name in lanes and name + "#A2" in lanes:
            a = lanes[name]["median_ms"]
            b = lanes[name + "#A2"]["median_ms"]
            base = min(a, b)
            noise[name] = {
                "a_vs_a_abs_ms": abs(a - b),
                "a_vs_a_rel": (abs(a - b) / base) if base else None,
            }

    derived = {}
    if "int8_quant" in lanes and "int8_gemm" in lanes and "int8_fused" in lanes:
        parts = lanes["int8_quant"]["median_ms"] + lanes["int8_gemm"]["median_ms"]
        fused = lanes["int8_fused"]["median_ms"]
        derived["int8_quant_share_of_fused"] = (
            lanes["int8_quant"]["median_ms"] / fused if fused else None
        )
        derived["int8_gemm_share_of_fused"] = (
            lanes["int8_gemm"]["median_ms"] / fused if fused else None
        )
        # Positive = the fused path costs more than its two parts measured
        # in isolation, i.e. launch-gap/allocator cost a fusion could recover.
        derived["int8_fused_minus_parts_ms"] = fused - parts
    for fp8_lane, key in (
        ("fp8_ct_fused", "int8_over_fp8_ct"),
        ("fp8_block_fused", "int8_over_fp8_block"),
    ):
        if "int8_fused" in lanes and fp8_lane in lanes:
            f8 = lanes[fp8_lane]["median_ms"]
            derived[key] = lanes["int8_fused"]["median_ms"] / f8 if f8 else None
    if "int8_fused" in lanes and "bf16_linear" in lanes:
        bf = lanes["bf16_linear"]["median_ms"]
        derived["int8_over_bf16"] = (
            lanes["int8_fused"]["median_ms"] / bf if bf else None
        )

    # The fusion decision input. Everything a fusion could remove that graph
    # replay has ALREADY removed must not be credited to the fusion.
    for name in want:
        g = name + "@graph"
        if name in lanes and g in lanes:
            eager, rep = lanes[name]["median_ms"], lanes[g]["median_ms"]
            derived[f"{name}_graph_over_eager"] = rep / eager if eager else None
            derived[f"{name}_launch_removed_by_graph_ms"] = eager - rep
    if "int8_quant@graph" in lanes and "int8_fused@graph" in lanes:
        fg = lanes["int8_fused@graph"]["median_ms"]
        derived["int8_quant_share_of_fused_graph"] = (
            lanes["int8_quant@graph"]["median_ms"] / fg if fg else None
        )
        if "int8_gemm@graph" in lanes:
            derived["int8_gemm_share_of_fused_graph"] = (
                lanes["int8_gemm@graph"]["median_ms"] / fg if fg else None
            )
            derived["int8_fused_minus_parts_graph_ms"] = fg - (
                lanes["int8_quant@graph"]["median_ms"]
                + lanes["int8_gemm@graph"]["median_ms"]
            )
    if "fp8_block_fused@graph" in lanes and "int8_fused@graph" in lanes:
        f8 = lanes["fp8_block_fused@graph"]["median_ms"]
        derived["int8_over_fp8_block_graph"] = (
            lanes["int8_fused@graph"]["median_ms"] / f8 if f8 else None
        )

    # #855 verdict inputs. The comparison is marlin_wna16 (a COMPLETE W8A16
    # op: no activation quant exists on that path) against int8_fused (the
    # complete W8A8 op: per-token quant + GEMM). Comparing it against
    # int8_gemm alone would hand W8A16 a cost the deployed lane really pays.
    for suffix in ("", "@graph"):
        mk = "marlin_wna16" + suffix
        if mk not in lanes:
            continue
        mv = lanes[mk]["median_ms"]
        for other in ("int8_fused", "bf16_linear", "int8_gemm"):
            ok = other + suffix
            if ok in lanes and lanes[ok]["median_ms"]:
                derived[f"marlin_over_{other}{suffix}"] = mv / lanes[ok]["median_ms"]

    del op_a, op_b, lanes_a, lanes_b, graphs
    out = {"lanes": lanes, "noise_floor": noise, "derived": derived}
    if graph_notes:
        out["graph_capture_failures"] = graph_notes
    if lane_failures:
        out["lane_failures"] = lane_failures
    return out


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def environment(dev: torch.device, cuda: bool, kn: Kernels) -> dict:
    env = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "kernels_stubbed": kn.stub,
        "kernels_missing": kn.missing,
        "fp8_dtype": str(kn.fp8_dtype),
    }
    if cuda:
        props = torch.cuda.get_device_properties(dev)
        env["device_name"] = props.name
        env["capability"] = f"sm{props.major}{props.minor}"
        env["total_memory_mib"] = props.total_memory // (1024 * 1024)
        env["torch_cuda"] = torch.version.cuda
        try:
            env["driver"] = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                .stdout.strip()
                .splitlines()[0]
            )
        except Exception as ex:
            env["driver"] = f"unavailable: {type(ex).__name__}"
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.strip()
    except Exception:
        env["git_commit"] = ""
    try:
        import sgl_kernel  # noqa: PLC0415

        env["sgl_kernel"] = getattr(sgl_kernel, "__version__", "unknown")
    except Exception as ex:
        env["sgl_kernel"] = f"unavailable: {type(ex).__name__}"
    return env


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DEFAULT_CONFIG = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8/config.json"
)


def int_list(text: str) -> list:
    return [int(p) for p in text.replace(" ", "").split(",") if p]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="checkpoint config.json")
    ap.add_argument("--tp-size", type=int, default=3)
    ap.add_argument(
        "--plans",
        default=DEFAULT_PLANS,
        help="semicolon-separated label=vector shard plans to derive shapes "
        f"for (default {DEFAULT_PLANS!r}: the auto vector decode runs on and "
        "the phase-optimal vector prefill runs on)",
    )
    ap.add_argument(
        "--ratio",
        default="",
        help="single --rank-tp-ratio vector; collapses --plans to one plan",
    )
    ap.add_argument(
        "--mlp-ratio", default="", help="--rank-mlp-ratio vector (with --ratio)"
    )
    ap.add_argument(
        "--rank", type=int, default=0, help="which rank's shard shapes to bench"
    )
    ap.add_argument("--m", default="1,2,4,8,16,2048", help="token counts")
    ap.add_argument(
        "--lanes",
        default="",
        help="comma list; prefix a name with + to add an optional lane",
    )
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--target-ms", type=float, default=20.0, help="wall time per burst")
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument(
        "--max-dim",
        type=int,
        default=0,
        help="cap N and K of the tensors actually multiplied (0 = no cap)",
    )
    ap.add_argument(
        "--no-reference-shapes",
        action="store_true",
        help="omit the #255 tuner shapes and the #370 FP8-block MLP pair",
    )
    ap.add_argument(
        "--fp8-block-backend",
        default="auto",
        choices=["auto"] + sorted(BLOCK_FP8_BACKENDS),
        help="which block-fp8 backend the fp8_block_fused lane uses; 'auto' "
        "is dispatch_w8a8_block_fp8_linear's own answer for this card",
    )
    ap.add_argument(
        "--graph-iters",
        type=int,
        default=0,
        help="capture a CUDA graph of this many bodies per lane and measure "
        "replay alongside eager in the same rotation (0 = eager only). This "
        "is the mode the fusion decision needs: graph replay already removes "
        "the per-launch cost that eager timing attributes to the quant.",
    )
    ap.add_argument("--seed", type=int, default=368)
    ap.add_argument("--out", default="", help="JSON output path")
    ap.add_argument(
        "--dry-run", action="store_true", help="CPU stubs, path coverage only"
    )
    ap.add_argument(
        "--shapes-only", action="store_true", help="print the shape table and exit"
    )
    ap.add_argument(
        "--shape-preset",
        choices=sorted(SHAPE_PRESETS),
        default="none",
        help="#855: add the ANALYSE_854 shape set (855) or its minimum "
        "decision subset (855min) to the table",
    )
    ap.add_argument(
        "--drop-derived-shapes",
        action="store_true",
        help="#855: measure ONLY the --shape-preset shapes. The derived "
        "table is the INT8 unit family's shard plan; the W8A16 lane coarsens "
        "to 128 instead, so mixing the two would price shapes neither "
        "checkpoint runs.",
    )
    ap.add_argument(
        "--check-imports",
        action="store_true",
        help="resolve the real kernels and exit; needs no card, so it is the "
        "pre-flight to run BEFORE claiming an arbitration window",
    )
    args = ap.parse_args(argv)

    if args.check_imports:
        kn = load_kernels(dry_run=False, block_backend=args.fp8_block_backend)
        for label, obj in (
            ("per_token_quant_int8", kn.per_token_quant_int8),
            ("int8_scaled_mm", kn.int8_scaled_mm),
            ("sglang_per_token_quant_fp8", kn.per_token_quant_fp8),
            ("fp8_scaled_mm", kn.fp8_scaled_mm),
            ("apply_fp8_linear", kn.apply_fp8_linear),
            ("w8a8_block_fp8_linear (dispatched)", kn.block_fp8_linear),
            ("apply_gptq_marlin_linear (#855 wNa16)", kn.apply_gptq_marlin_linear),
            ("gptq_marlin_repack (#855 wNa16)", kn.gptq_marlin_repack),
        ):
            print(f"{'OK  ' if obj is not None else 'MISS'} {label}")
        for line in kn.missing:
            print(f"     {line}")
        print(f"cuda_available={torch.cuda.is_available()}")
        return 0 if not kn.missing else 1

    if not 0 <= args.rank < max(1, args.tp_size):
        ap.error("--rank out of range")

    # A plan is label=vector. --ratio (with optional --mlp-ratio) collapses
    # the sweep to one plan, for a one-off question about a specific vector.
    if args.ratio:
        base = int_list(args.ratio)
        plans = [("custom", base, int_list(args.mlp_ratio) if args.mlp_ratio else base)]
    else:
        plans = []
        for chunk in args.plans.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            label, _, vec = chunk.partition("=")
            if not vec:
                ap.error(f"--plans entry {chunk!r} is not label=vector")
            v = int_list(vec)
            plans.append((label, v, v))
    for label, v, _ in plans:
        if args.tp_size > 1 and len(v) != args.tp_size:
            ap.error(
                f"plan {label!r} has {len(v)} entries but --tp-size is {args.tp_size}"
            )

    shapes: list = []
    by_shape: dict = {}
    cases: list = []
    facts_by_plan: dict = {}
    plan_layers: dict = {}
    if args.drop_derived_shapes:
        # #855 runs a literal shape set from ANALYSE_854 and never derives a
        # shard plan, so it must not require a checkpoint on disk. The INT8
        # checkpoint this default points at is not even the same model any
        # more (the standard model moved 3.6 -> 3.8).
        plans = [(p[0], p[1], p[2]) for p in plans][:1]
        facts_by_plan[plans[0][0]] = {
            "note": "not derived: --drop-derived-shapes",
            "local_q_heads": "-",
            "local_kv_heads": "-",
            "local_gdn_k_heads": "-",
            "local_intermediate": "-",
        }
        cfg = {}
        plan_layers[plans[0][0]] = 0
    else:
        with open(args.config) as fh:
            cfg = json.load(fh)
    for label, base, mlp in plans if not args.drop_derived_shapes else []:
        derived_shapes, plan_cases, facts = derive_shapes(
            cfg, args.tp_size, base, mlp, args.rank
        )
        cases.extend(plan_cases)
        facts_by_plan[label] = facts
        # Counted before dedupe: two modules of one plan can share a GEMM
        # (attn_o and gdn_out are both 5120x3072 under the auto vector), and
        # the per-token launch count must still see both.
        plan_layers[label] = sum(s.layers for s in derived_shapes)
        for s in derived_shapes:
            key = (s.n, s.k)
            if key in by_shape:
                # Plans (or modules) that land on the same GEMM are measured
                # once and attributed to all -- the kernel cannot tell them
                # apart.
                by_shape[key].plans.append(f"{label}/{s.name} x{s.layers}")
                continue
            s.plans = [f"{label}/{s.name} x{s.layers}"]
            s.name = f"{label}_{s.name}"
            by_shape[key] = s
            shapes.append(s)
    facts = facts_by_plan[plans[0][0]]
    provenance = cross_check_partition(cases)

    if args.drop_derived_shapes:
        if args.shape_preset == "none":
            ap.error("--drop-derived-shapes needs a --shape-preset to measure")
        shapes = []
        by_shape = {}

    if args.shape_preset != "none":
        selected = set(SHAPE_PRESETS[args.shape_preset])
        for name, n, k, why in SHAPES_855:
            if name not in selected:
                continue
            if (n, k) in by_shape:
                by_shape[(n, k)].plans.append(name)
                continue
            s = Shape(name, n, k, why, 0, "#855 ANALYSE_854 shape", [name])
            by_shape[(n, k)] = s
            shapes.append(s)

    if not args.no_reference_shapes and not args.drop_derived_shapes:
        for name, n, k, why in REFERENCE_SHAPES:
            if (n, k) in by_shape:
                by_shape[(n, k)].plans.append(name)
                continue
            s = Shape(name, n, k, why, 0, "reference operating point", [name])
            by_shape[(n, k)] = s
            shapes.append(s)

    # Dry run defaults: real shape table, small tensors, few rounds.
    if args.dry_run:
        if args.max_dim == 0:
            args.max_dim = 256
        if args.rounds == ap.get_default("rounds"):
            args.rounds = 3
        args.target_ms = min(args.target_ms, 1.0)
        args.max_iters = min(args.max_iters, 20)

    if args.max_dim:
        cap = args.max_dim
        for s in shapes:
            # Keep the kernel's own divisibility rules (K % 16, N % 8) intact
            # so the capped run still exercises a legal shape.
            s.exec_n = min(s.n, cap - cap % 8)
            s.exec_k = min(s.k, cap - cap % 16)

    m_list = int_list(args.m)
    if args.dry_run and args.m == ap.get_default("m"):
        m_list = [1, 2, 8]

    want = list(ALL_LANES)
    if args.lanes:
        explicit = [p for p in args.lanes.replace(" ", "").split(",") if p]
        added = [p[1:] for p in explicit if p.startswith("+")]
        named = [p for p in explicit if not p.startswith("+")]
        if named:
            want = named
        want = want + [a for a in added if a not in want]
    unknown = [w for w in want if w not in ALL_LANES + OPTIONAL_LANES]
    if unknown:
        ap.error(f"unknown lane(s): {unknown}; known: {ALL_LANES + OPTIONAL_LANES}")

    # ---- shape table -----------------------------------------------------
    print("=" * 78)
    print(
        f"Task #368 INT8 decode microbench -- shapes for rank {args.rank} of TP={args.tp_size}"
    )
    print(f"config      {args.config}")
    for label, base, mlp in plans:
        f = facts_by_plan[label]
        print(
            f"plan        {label:<10} tp {base}  mlp {mlp}  -> q {f['local_q_heads']} "
            f"kv {f['local_kv_heads']} gdn-k {f['local_gdn_k_heads']} "
            f"intermediate {f['local_intermediate']}"
        )
    print(f"partition   {provenance}")
    print("=" * 78)
    print(f"{'shape':<30}{'N(out)':>9}{'K(in)':>9}{'layers':>8}  module")
    illegal = []
    for s in shapes:
        # int8_scaled_mm's own N % 8 / K % 16 rules
        # (w8a8_int8.verify_int8_scaled_mm_supports_shape). A shape that
        # violates them would abort inside CUTLASS, so name it here.
        if s.n % 8 or s.k % 16:
            illegal.append(f"{s.name} N={s.n} K={s.k}")
        print(f"{s.name:<30}{s.n:>9}{s.k:>9}{s.layers:>8}  {s.module}")
        if len(s.plans) > 1:
            print(f"{'':<30}also: {', '.join(s.plans[1:])}")
        if s.note:
            print(f"{'':<30}{s.note}")
        if s.capped:
            print(f"{'':<30}[dry run] tensors multiplied at N={s.run_n} K={s.run_k}")
    print("-" * 78)
    for label, n_gemms in plan_layers.items():
        print(
            f"plan {label}: {n_gemms} INT8 linear layers per decoded token on "
            f"this rank ({2 * n_gemms} kernel launches -- quant + GEMM each)"
        )
    for line in facts.get("not_quantized_by_ignore_list", []):
        print(f"  not INT8: {line}")
    if illegal:
        print(
            "  !! int8_scaled_mm requires N % 8 == 0 and K % 16 == 0; "
            f"these shards do not satisfy it: {illegal}"
        )
    print("=" * 78)
    if args.shapes_only:
        return 0

    kn = load_kernels(args.dry_run, args.fp8_block_backend)
    cuda = (not args.dry_run) and torch.cuda.is_available()
    if not args.dry_run and not cuda:
        print(
            "No CUDA device visible. Use --dry-run for the desk path.", file=sys.stderr
        )
        return 2
    dev = torch.device("cuda") if cuda else torch.device("cpu")

    env = environment(dev, cuda, kn)
    print(f"device      {env.get('device_name', 'cpu')} {env.get('capability', '')}")
    print(f"lanes       {want}")
    print(f"M           {m_list}")
    print(f"rounds      {args.rounds}   target {args.target_ms} ms/burst")
    if args.graph_iters:
        print(f"graph       CUDA-graph replay, {args.graph_iters} bodies per capture")
    if kn.stub:
        print("!! DRY RUN: pure-torch stand-ins, NOT a measurement !!")
    print("=" * 78)

    results = []
    t_start = time.time()
    for s in shapes:
        # Weights are M-independent and expensive to sample; build the
        # A-vs-A pair once per shape and hold it across the whole M sweep.
        w_a = build_weights(s, dev, kn, args.seed)
        w_b = build_weights(s, dev, kn, args.seed + 977)
        for m in m_list:
            t0 = time.time()
            point = run_point(
                s,
                m,
                w_a,
                w_b,
                dev,
                kn,
                want,
                args.rounds,
                args.target_ms,
                args.max_iters,
                cuda,
                args.seed + m,
                args.graph_iters,
            )
            row = {"shape": asdict(s), "m": m, "wall_s": round(time.time() - t0, 3)}
            row.update(point)
            results.append(row)
            if "lanes" in point:
                med = {
                    k: round(v["median_ms"], 5)
                    for k, v in point["lanes"].items()
                    if "#A2" not in k
                }
                wall = row["wall_s"]
                print(f"{s.name:<30} M={m:<6} {wall:>6.2f}s  {med}")
            else:
                print(f"{s.name:<30} M={m:<6} {point}")
        del w_a, w_b
        if cuda:
            torch.cuda.empty_cache()
    total_s = round(time.time() - t_start, 2)
    print("=" * 78)
    print(f"total measurement wall time: {total_s} s")

    payload = {
        "task": "368",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": bool(args.dry_run),
        "environment": env,
        "model_config": args.config,
        "model_facts": facts_by_plan,
        "plan": {
            "tp_size": args.tp_size,
            "rank": args.rank,
            "plans": [
                {"label": label, "ratio": base, "mlp_ratio": mlp}
                for label, base, mlp in plans
            ],
            "partition_provenance": provenance,
            "int8_layers_per_token_per_plan": plan_layers,
        },
        "settings": {
            "m": m_list,
            "lanes": want,
            "rounds": args.rounds,
            "target_ms": args.target_ms,
            "max_iters": args.max_iters,
            "max_dim": args.max_dim,
            "fp8_block_backend": args.fp8_block_backend,
            "fp8_block_backend_resolved": getattr(
                kn.block_fp8_linear, "__name__", str(kn.block_fp8_linear)
            ),
            "seed": args.seed,
            "graph_iters": args.graph_iters,
        },
        "shapes": [asdict(s) for s in shapes],
        "results": results,
        "total_wall_s": total_s,
    }
    out = args.out
    if not out:
        slug = env.get("device_name", "cpu").replace(" ", "_")
        suffix = "dryrun" if args.dry_run else time.strftime("%Y%m%d-%H%M%S")
        out = f"/tmp/int8_368_microbench.{slug}.{suffix}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
