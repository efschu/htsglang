#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #337 -- target registry for the TensorRT-RTX vs. own-kernel microbench.

WHAT A "TARGET" IS
==================
A target is one *part* of the deployed model, expressed as an ordered CHAIN of
stages. Two stage kinds exist and the distinction is the whole design:

    quant   -- ``per_token_quant_int8``, our Triton kernel. It runs in EVERY
               arm, identically, and it is NEVER inside a TensorRT engine.
    engine  -- a subgraph that TensorRT-RTX compiles into an engine, and for
               which a bit-comparable torch implementation exists.

WHY THE QUANT STAGE CAN NOT BE INSIDE THE ENGINE
------------------------------------------------
TensorRT's ``IQuantizeLayer`` documents, verbatim in the installed library
(``tensorrt_rtx 1.6.1.120``, ``trt.IQuantizeLayer.__doc__``):

    "The subgraph which terminates with the scale tensor must be a build-time
     constant."

Our deployed activation scale is per TOKEN and computed from the activation at
runtime (``compressed_tensors_w8a8_int8.py``: ``x_q, x_scale =
per_token_quant_int8(x)``). It is by construction not a build-time constant, so
TensorRT cannot express it. ``IDynamicQuantizeLayer`` does do runtime scales but
its own docstring restricts it to "kFP4 (NVFP4 quantization) or kFP8 (MXFP8)"
output with block sizes 16 or 32 -- neither INT8 nor per-token.

That is a hard capability fact, not a modelling choice, and it fixes the engine
boundary: every INT8 GEMM's activation quant happens outside the engine, in our
own kernel, in every arm. The consequence is that the arms carry EQUAL TOTAL
WORK -- which is the only way the comparison means anything. What TensorRT still
gets to fuse is everything else: the dequantize pair, the GEMM, the dual-scale
epilogue, the bias, the activation function, the elementwise chain.

HOW THE DEPLOYED ARITHMETIC IS REPRODUCED EXACTLY
=================================================
``int8_scaled_mm(x_q, W_q, x_scale, w_scale, out_dtype, bias)`` computes

    out[m,n] = (sum_k x_q[m,k] * W_q[n,k]) * x_scale[m] * w_scale[n] + bias[n]

In the TensorRT network that is:

    DQ(x_q, scale=1.0)              -- per-tensor, constant, folds away
    DQ(W_q, scale=w_scale, axis=0)  -- per-output-channel, a build-time
                                       constant because weights are constant.
                                       This is the deployed weight quantization
                                       EXACTLY: the checkpoint stores
                                       ``*.weight_scale`` as [N,1] bf16.
    MatMul(., ., TRANSPOSE)
    Mul(., x_scale)                 -- the per-token scale enters as an ORDINARY
                                       runtime elementwise tensor, not as a Q/DQ
                                       scale, which is what sidesteps the
                                       build-time-constant restriction.
    (+ bias)  ->  cast to bf16

Same arithmetic, same operand precision, no fp16 downgrade anywhere (see
``build_engines.py`` for why a downgrade is structurally impossible in this
TensorRT version). Results are not expected to be bit-identical -- the epilogue
scaling happens in a different order and possibly a different intermediate
precision than CUTLASS uses -- so the harness runs a tolerance gate and reports
``max_abs_diff`` rather than claiming byte identity.

SHAPES
======
Derived from the deployed checkpoint's own ``config.json`` and the deployed
shard plan, never hardcoded. The shard arithmetic is taken from the #368
harness (``scripts/int8_368/microbench.py``), which cross-checks itself against
``sglang.srt.distributed.utils._partition_units_raw`` and refuses to emit a
table that disagrees with what the ranks would actually build.

Decode runs on the auto vector ``[30,17,17]`` over TP=3. Rank 0 is the 5090
(sm120), ranks 1 and 2 are the 3080s (sm86). Both rank geometries are targets,
because a TensorRT verdict that only holds for the fat rank is not a verdict
about the deployment.

WEIGHTS
=======
Real values from the deployed checkpoint
(``Qwen3.6-27B-INT8-W8A8/model.safetensors``), sliced to the per-rank shard
EXTENT. What is reproduced: the INT8 weight value distribution and the real
per-output-channel ``weight_scale`` magnitudes -- which is what makes the
tolerance gate meaningful. What is NOT reproduced: the exact head-index
permutation inside a shard. That permutation changes neither GEMM cost nor the
comparison (both arms consume the identical tensor), so it is not worth the
loader complexity; it is recorded here so nobody later mistakes the slice for a
faithful rank reconstruction.

``--random-weights`` falls back to shape-faithful randoms. Per the rig's
cross-arch rule those are sampled on the CPU and moved, never generated on the
device, so two cards see identical bytes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MB368_PATH = os.path.join(REPO_ROOT, "scripts", "int8_368", "microbench.py")

DEFAULT_CONFIG = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8/config.json"
)
#: The deployed decode shard vector (auto-performance), TP=3.
DEFAULT_RATIO = (30, 17, 17)

#: rank -> the architecture that rank runs on, on this rig.
RANK_ARCH = {0: "sm120", 1: "sm86", 2: "sm86"}


# --------------------------------------------------------------------------
# Shard geometry, borrowed from the #368 harness so the two can not drift
# --------------------------------------------------------------------------


def load_mb368():
    """Import the #368 microbench module by path.

    Not a package import: the file lives under ``scripts/`` and is not on any
    import path. Registered in ``sys.modules`` before execution because its
    dataclasses resolve their own module during class creation.
    """
    if not os.path.exists(MB368_PATH):
        raise SystemExit(
            f"#368 harness not found at {MB368_PATH}. The shard arithmetic and "
            f"its cross-check against the serving stack live there; this script "
            f"deliberately does not carry a third copy of it."
        )
    spec = importlib.util.spec_from_file_location("mb368", MB368_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mb368"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ShardGeometry:
    """Per-rank widths of every INT8 linear in one decode layer."""

    rank: int
    tp_size: int
    ratio: tuple
    arch: str
    hidden: int
    local_q_heads: int
    local_kv_heads: int
    head_dim: int
    local_gdn_k_heads: int
    local_gdn_v_heads: int
    gdn_head_dim: int
    local_intermediate: int
    attn_output_gate: bool
    partition_provenance: str
    facts: dict = field(default_factory=dict)

    # -- attention (full-attention layers) --
    @property
    def qkv_n(self) -> int:
        q = self.local_q_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        kv = 2 * self.local_kv_heads * self.head_dim
        return q + kv

    @property
    def attn_o_k(self) -> int:
        return self.local_q_heads * self.head_dim

    # -- gated delta net (linear-attention layers) --
    @property
    def gdn_in_qkvz_n(self) -> int:
        qk = 2 * self.local_gdn_k_heads * self.gdn_head_dim
        zv = 2 * self.local_gdn_v_heads * self.gdn_head_dim
        return qk + zv

    @property
    def gdn_out_k(self) -> int:
        return self.local_gdn_v_heads * self.gdn_head_dim

    # -- mlp --
    @property
    def gate_up_n(self) -> int:
        return 2 * self.local_intermediate


def derive_geometry(
    config_path: str,
    tp_size: int,
    ratio: Sequence[int],
    rank: int,
) -> ShardGeometry:
    mb = load_mb368()
    with open(config_path) as fh:
        cfg = json.load(fh)
    _shapes, cases, facts = mb.derive_shapes(cfg, tp_size, list(ratio), list(ratio), rank)
    provenance = mb.cross_check_partition(cases)
    text = cfg.get("text_config", cfg)
    return ShardGeometry(
        rank=rank,
        tp_size=tp_size,
        ratio=tuple(ratio),
        arch=RANK_ARCH.get(rank, "unknown"),
        hidden=facts["hidden_size"],
        local_q_heads=facts["local_q_heads"],
        local_kv_heads=facts["local_kv_heads"],
        head_dim=facts["head_dim"],
        local_gdn_k_heads=facts["local_gdn_k_heads"],
        local_gdn_v_heads=facts["local_gdn_v_heads"],
        gdn_head_dim=int(
            text.get("linear_key_head_dim", text.get("linear_value_head_dim", 128))
        ),
        local_intermediate=facts["local_intermediate"],
        attn_output_gate=bool(facts.get("attn_output_gate", True)),
        partition_provenance=provenance,
        facts=facts,
    )


# --------------------------------------------------------------------------
# Stage / target model
# --------------------------------------------------------------------------


@dataclass
class GemmSpec:
    """One INT8 linear in serving orientation.

    ``n`` is the output width, ``k`` the reduction dim. Weight is stored [N,K]
    INT8 with a per-output-channel scale of length N, which is exactly the
    checkpoint's layout.
    """

    name: str
    n: int
    k: int
    ckpt: Optional[tuple] = None  # (tensor-key template, row-slice, col-slice)
    epilogue: str = "none"  # none | silu_mul
    layers_per_token: int = 0
    module: str = ""
    note: str = ""


@dataclass
class Stage:
    """One step of a target's chain.

    kind:
      ``quant``   our ``per_token_quant_int8``; runs in every arm, never inside
                  an engine (TensorRT cannot express it -- see the module
                  docstring).
      ``engine``  a TensorRT-compiled subgraph with a torch twin.
      ``bridge``  a width adapter standing in for a stage that is deliberately
                  NOT measured. Used once, for the attention core: qkv emits
                  ``local_q_heads*head_dim*2 + 2*local_kv_heads*head_dim`` and
                  o_proj consumes ``local_q_heads*head_dim``, and the paged-KV
                  attention kernel in between has no stock-TensorRT expression.
                  The bridge narrows to the consumed width and is executed
                  IDENTICALLY in both arms, so it cancels out of every ratio.
    """

    kind: str  # "quant" | "engine" | "bridge"
    name: str
    gemm: Optional[GemmSpec] = None
    #: for engine stages: how the output feeds the next stage
    #: for bridge stages: the width handed to the next stage
    out_width: int = 0


@dataclass
class Target:
    name: str
    arch: str
    rank: int
    stages: list
    description: str
    layers_per_token: int
    optional: bool = False

    @property
    def engine_stages(self) -> list:
        return [s for s in self.stages if s.kind == "engine"]


def _ckpt(kind: str) -> tuple:
    """Checkpoint key templates for the deployed INT8 checkpoint.

    Layer 0 is a linear-attention (GDN) layer, layer 3 the first
    full-attention layer -- verified against the safetensors header of
    ``Qwen3.6-27B-INT8-W8A8``.
    """
    L = "model.language_model.layers"
    return {
        "gate": (f"{L}.3.mlp.gate_proj", "rows"),
        "up": (f"{L}.3.mlp.up_proj", "rows"),
        "down": (f"{L}.3.mlp.down_proj", "cols"),
        "q": (f"{L}.3.self_attn.q_proj", "rows"),
        "k": (f"{L}.3.self_attn.k_proj", "rows"),
        "v": (f"{L}.3.self_attn.v_proj", "rows"),
        "o": (f"{L}.3.self_attn.o_proj", "cols"),
        "gdn_qkv": (f"{L}.0.linear_attn.in_proj_qkv", "rows"),
        "gdn_z": (f"{L}.0.linear_attn.in_proj_z", "rows"),
        "gdn_out": (f"{L}.0.linear_attn.out_proj", "cols"),
    }[kind]


def build_targets(geo: ShardGeometry, include_optional: Sequence[str] = ()) -> list:
    """The target matrix: five engines by default, one optional sixth.

    Selection rationale is in TARGET_SELECTION.md next to this file. In short:
    the three standalone GEMMs are the shapes that dominate the decode step by
    layer count (64/64/16 of 64 layers), the two chains are where a fusion can
    actually pay, and the GDN conv chain is held back because #325 has not
    fixed its fusion boundary yet.
    """
    h = geo.hidden
    gate_up = GemmSpec(
        name="mlp_gate_up",
        n=geo.gate_up_n,
        k=h,
        ckpt=("gate_up", None, None),
        layers_per_token=64,
        module="Qwen2MoeMLP.gate_up_proj",
        note=f"gate+up merged, local intermediate {geo.local_intermediate}",
    )
    down = GemmSpec(
        name="mlp_down",
        n=h,
        k=geo.local_intermediate,
        ckpt=("down", None, None),
        layers_per_token=64,
        module="Qwen2MoeMLP.down_proj",
        note="row-parallel",
    )
    qkv = GemmSpec(
        name="attn_qkv",
        n=geo.qkv_n,
        k=h,
        ckpt=("qkv", None, None),
        layers_per_token=16,
        module="Qwen3_5Attention.qkv_proj",
        note=(
            f"{geo.local_q_heads} q heads"
            f"{' x2 (output gate)' if geo.attn_output_gate else ''}"
            f" + 2x{geo.local_kv_heads} kv, head_dim {geo.head_dim}"
        ),
    )
    attn_o = GemmSpec(
        name="attn_o",
        n=h,
        k=geo.attn_o_k,
        ckpt=("o", None, None),
        layers_per_token=16,
        module="Qwen3_5Attention.o_proj",
        note="row-parallel, input = local q width",
    )
    gdn_in = GemmSpec(
        name="gdn_in_qkvz",
        n=geo.gdn_in_qkvz_n,
        k=h,
        ckpt=("gdn_qkvz", None, None),
        layers_per_token=48,
        module="Qwen3_5GatedDeltaNet.in_proj_qkvz",
        note=(
            f"[q,k,z,v] merged: 2x{geo.local_gdn_k_heads}x{geo.gdn_head_dim}"
            f" + 2x{geo.local_gdn_v_heads}x{geo.gdn_head_dim}"
        ),
    )

    def T(name, stages, desc, lpt, optional=False):
        return Target(
            name=name,
            arch=geo.arch,
            rank=geo.rank,
            stages=stages,
            description=desc,
            layers_per_token=lpt,
            optional=optional,
        )

    targets = [
        T(
            "gemm_mlp_gate_up",
            [Stage("quant", "quant_in"), Stage("engine", "gate_up", gate_up, gate_up.n)],
            "The single most-executed INT8 GEMM in the model: every one of the "
            "64 layers runs it once per token.",
            64,
        ),
        T(
            "gemm_mlp_down",
            [Stage("quant", "quant_in"), Stage("engine", "down", down, down.n)],
            "The row-parallel partner of gate_up, also 64x per token, and the "
            "one whose K is the shard-dependent dimension.",
            64,
        ),
        T(
            "gemm_attn_qkv",
            [Stage("quant", "quant_in"), Stage("engine", "qkv", qkv, qkv.n)],
            "The widest attention projection; 16 full-attention layers. Carries "
            "the output-gate doubling, so it is the shape a TensorRT tactic is "
            "least likely to have been tuned for.",
            16,
        ),
        T(
            "chain_mlp_block",
            [
                Stage("quant", "quant_in"),
                Stage("engine", "gate_up_silu", gate_up_with_silu(gate_up), geo.local_intermediate),
                Stage("quant", "quant_mid"),
                Stage("engine", "down", down, down.n),
            ],
            "The whole MLP block as it runs in decode: gate_up, SiLU-gate "
            "multiply, down. This is where a fusion can actually pay, because "
            "the SiLU-multiply and both dual-scale epilogues are TensorRT's to "
            "fuse while the two activation quants stay ours in every arm.",
            64,
        ),
        T(
            "chain_decode_layer",
            [
                Stage("quant", "quant_attn_in"),
                Stage("engine", "qkv", qkv, qkv.n),
                Stage("bridge", "attention_core_excluded", None, attn_o.k),
                Stage("quant", "quant_attn_out"),
                Stage("engine", "attn_o", attn_o, attn_o.n),
                Stage("quant", "quant_mlp_in"),
                Stage("engine", "gate_up_silu", gate_up_with_silu(gate_up), geo.local_intermediate),
                Stage("quant", "quant_mlp_mid"),
                Stage("engine", "down", down, down.n),
            ],
            "One full-attention decode layer's INT8 LINEAR CHAIN: qkv, o_proj, "
            "gate_up with its SiLU-gate multiply, down -- four GEMMs with four "
            "activation quants between them, which is what one decode layer of "
            "this rank actually issues. Two things are deliberately NOT in it, "
            "on BOTH arms: the attention core (paged-KV flashinfer, no stock "
            "TensorRT expression -- a placeholder would price a fiction) and "
            "the RMSNorms and residual adds (they sit upstream of an activation "
            "quant that has to stay outside the engine anyway, so putting them "
            "in one arm only would be the asymmetry this design exists to "
            "avoid). What is measured is exactly the part a per-part engine "
            "could replace today, and nothing it could not.",
            16,
        ),
    ]

    if "gdn_conv" in include_optional:
        targets.append(
            T(
                "chain_gdn_conv",
                [
                    Stage("quant", "quant_in"),
                    Stage("engine", "gdn_in", gdn_in, gdn_in.n),
                ],
                "The #325 fusion candidate's input projection. Held optional: "
                "#325 has not fixed the conv+gating fusion boundary, and a "
                "microbench of a boundary that may move is a number with a "
                "short shelf life.",
                48,
                optional=True,
            )
        )
    return targets


def gate_up_with_silu(base: GemmSpec) -> GemmSpec:
    return GemmSpec(
        name=base.name + "_silu",
        n=base.n,
        k=base.k,
        ckpt=base.ckpt,
        epilogue="silu_mul",
        layers_per_token=base.layers_per_token,
        module=base.module + " + silu_and_mul",
        note=base.note + "; output halves to the local intermediate",
    )


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def _cpu_randn(shape, seed, dtype):
    """CPU-sampled randoms.

    The rig's cross-arch rule: ``torch.randn`` on device is not architecture
    identical, so anything two cards must agree on is sampled on the CPU and
    moved.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)


@dataclass
class GemmWeights:
    q: "object"  # int8 [N,K] cpu tensor
    scale: "object"  # float32 [N] cpu tensor
    source: str


def load_gemm_weights(
    spec: GemmSpec,
    geo: ShardGeometry,
    model_dir: str,
    random_weights: bool,
    seed: int,
) -> GemmWeights:
    """Real shard-extent weights from the checkpoint, or shape-faithful randoms.

    Column-parallel weights (``rows``) are sliced on the output dim, row-parallel
    weights (``cols``) on the reduction dim; that is the same axis the serving
    loader shards on.
    """
    import torch

    if random_weights:
        q = torch.randint(
            -127, 128, (spec.n, spec.k), generator=torch.Generator().manual_seed(seed),
            dtype=torch.int8,
        )
        scale = (_cpu_randn((spec.n,), seed + 1, torch.float32).abs() * 1e-3 + 1e-4)
        return GemmWeights(q, scale, "shape-faithful random, CPU-sampled")

    from safetensors import safe_open

    path = os.path.join(model_dir, "model.safetensors")
    parts = []
    scales = []
    keys = {
        "mlp_gate_up": ["gate", "up"],
        "mlp_down": ["down"],
        "attn_qkv": ["q", "k", "v"],
        "attn_o": ["o"],
        "gdn_in_qkvz": ["gdn_qkv", "gdn_z"],
        "gdn_out": ["gdn_out"],
    }[spec.name.replace("_silu", "")]

    with safe_open(path, framework="pt", device="cpu") as fh:
        for key in keys:
            base, axis = _ckpt(key)
            w = fh.get_slice(base + ".weight")
            s = fh.get_slice(base + ".weight_scale")
            full = w.get_shape()
            if axis == "rows":
                take = _rows_for(spec, geo, key, full[0])
                parts.append(w[0:take, :])
                scales.append(s[0:take, :].reshape(-1))
            else:
                take = _cols_for(spec, geo, key, full[1])
                parts.append(w[:, 0:take])
                scales.append(s[:].reshape(-1))

    q = torch.cat(parts, dim=0 if _ckpt(keys[0])[1] == "rows" else 1)
    scale = (
        torch.cat(scales, dim=0)
        if _ckpt(keys[0])[1] == "rows"
        else scales[0]
    ).float()
    if q.shape != (spec.n, spec.k):
        raise SystemExit(
            f"{spec.name}: assembled weight {tuple(q.shape)} does not match the "
            f"derived shard shape {(spec.n, spec.k)}. The shard arithmetic and "
            f"the checkpoint disagree; refusing to bench a shape that is not "
            f"the deployed one."
        )
    return GemmWeights(q.contiguous(), scale.contiguous(), "checkpoint shard extent")


def _rows_for(spec: GemmSpec, geo: ShardGeometry, key: str, full: int) -> int:
    gate = 2 if geo.attn_output_gate else 1
    return {
        "gate": geo.local_intermediate,
        "up": geo.local_intermediate,
        "q": geo.local_q_heads * geo.head_dim * gate,
        "k": geo.local_kv_heads * geo.head_dim,
        "v": geo.local_kv_heads * geo.head_dim,
        "gdn_qkv": (2 * geo.local_gdn_k_heads + geo.local_gdn_v_heads)
        * geo.gdn_head_dim,
        "gdn_z": geo.local_gdn_v_heads * geo.gdn_head_dim,
    }[key]


def _cols_for(spec: GemmSpec, geo: ShardGeometry, key: str, full: int) -> int:
    return {
        "down": geo.local_intermediate,
        "o": geo.attn_o_k,
        "gdn_out": geo.gdn_out_k,
    }[key]


def shape_table(targets: Sequence[Target]) -> list:
    rows = []
    for t in targets:
        for st in t.engine_stages:
            g = st.gemm
            rows.append(
                {
                    "target": t.name,
                    "arch": t.arch,
                    "rank": t.rank,
                    "stage": st.name,
                    "n": g.n,
                    "k": g.k,
                    "epilogue": g.epilogue,
                    "layers_per_token": g.layers_per_token,
                    "module": g.module,
                    "note": g.note,
                }
            )
    return rows


if __name__ == "__main__":  # shape table, desk, no card, no engine
    import argparse

    ap = argparse.ArgumentParser(description="Print the #337 target shape table.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tp-size", type=int, default=3)
    ap.add_argument("--ratio", default="30,17,17")
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--optional", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ratio = [int(x) for x in a.ratio.split(",")]
    opt = [x for x in a.optional.split(",") if x]
    out = []
    for rank in [int(r) for r in a.ranks.split(",")]:
        geo = derive_geometry(a.config, a.tp_size, ratio, rank)
        ts = build_targets(geo, opt)
        out.append(
            {
                "rank": rank,
                "arch": geo.arch,
                "partition_provenance": geo.partition_provenance,
                "geometry": {
                    "hidden": geo.hidden,
                    "local_q_heads": geo.local_q_heads,
                    "local_kv_heads": geo.local_kv_heads,
                    "local_gdn_k_heads": geo.local_gdn_k_heads,
                    "local_gdn_v_heads": geo.local_gdn_v_heads,
                    "local_intermediate": geo.local_intermediate,
                },
                "targets": [
                    {"name": t.name, "layers_per_token": t.layers_per_token,
                     "optional": t.optional,
                     "stages": [s.kind + ":" + s.name for s in t.stages]}
                    for t in ts
                ],
                "shapes": shape_table(ts),
            }
        )
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for blk in out:
            print(f"=== rank {blk['rank']} ({blk['arch']}) ratio {ratio} ===")
            print(f"    {blk['partition_provenance']}")
            print(f"    {blk['geometry']}")
            print(f"    {'stage':22s} {'N':>7s} {'K':>7s} {'epi':9s} {'x/token':>8s}")
            seen = set()
            for r in blk["shapes"]:
                sig = (r["stage"], r["n"], r["k"], r["epilogue"])
                if sig in seen:
                    continue
                seen.add(sig)
                print(
                    f"    {r['stage']:22s} {r['n']:7d} {r['k']:7d} "
                    f"{r['epilogue']:9s} {r['layers_per_token']:8d}"
                )
            print()
