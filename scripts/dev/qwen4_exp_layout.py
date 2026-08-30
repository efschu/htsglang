#!/usr/bin/env python3
"""Solve the TP3 VRAM layout for qwen4_exp and emit it as provenance lines.

Register #1036. DESK ONLY: no GPU, no CUDA context, no NVML, no torch. It reads
`config.json` and a table of MEASURED byte classes, and it SOLVES. It never
hand-picks a split -- hand-pinning is a named defect class on this fork
(`server_args.py` refuses a solve-vs-pin pair rather than resolving it), so
this script emits the solved vector plus the constraint that bound it.

WHY THIS EXISTS. The boot-time solver (`--rank-tp-ratio auto`) and the VRAM
ledger (`--enable-vram-ledger`) both answer this question AFTER a card is
loaded. For a checkpoint whose routed-expert stack (65.0 GiB) and n-gram
embedding table (95.4 GiB) each exceed the rig's whole VRAM (71.84 GiB), "we
will find out at load time" is not a plan. This is the pre-boot oracle: it says
whether a KV target is reachable, and when it is not, WHICH post it traded
against.

OUTPUT FORMAT is not invented. Four emitters already in the tree are matched
line for line, because a new provenance format is a second bookkeeping layer:

  * `uneven_perf.py:7817-7830` -- "CHOSEN <x> vector: ... (materialized ...;
    predicted ... >= floor ...; predicted per-rank ...; predicted ... vs the
    reference split)" plus the companion "floor check: ... -- OK".
  * `pool_configurator.py:552-556` -- "KV pool sizing: available_bytes=%d
    (%.3f GiB), cell_size=%d, page_size=%d -> max_total_num_tokens=%d".
  * `model_runner_kv_cache_mixin.py:5196-5207` -- "KV token sizing: rank %d
    local capacity %d tokens, min-reduced across ranks to %d (%s; %d stranded
    on this rank). Global addressable KV = %d x dcp_size(%d)."; and `:6231-6235`
    -- "Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR=%s to raise
    max_total_num_tokens from %d to ~%d (per-rank profiled capacity %s; active
    vector %s leaves ranks idle)."
  * `mem_ledger/terms.py:407-425` -- the per-card itemization, one line per post
    with its mark and derivation, then the residual row.

WHAT IS GENUINELY UNDOCUMENTED is named as a constant with the alternative
spelled out next to it, never left silent. Search this file for ASSUMPTION.
Three earlier assumptions were RETIRED once upstream's own implementation
became readable from the local object store (`git show
upstream/qwen4-main-squashed:<path>`, ref 99c9362e6685db579c469f6e0e566b08827b3477):
the PLE row geometry, the PLE side-state shapes, and the speculation bounds are
now quoted, not guessed.

Exit status: 0 when the emitted layout satisfies every invariant, 1 when an
invariant breaks or no layout exists at all (an infeasible layout must never be
printed as an answer), 2 on a usage/input error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from typing import List, Optional, Sequence, Tuple

_MIB = 1024.0 * 1024.0
_GIB = 1024.0 * 1024.0 * 1024.0

# ---------------------------------------------------------------------------
# MEASURED INPUTS. Every number here was measured off the real checkpoint by
# the #1036 census (CHECKPOINT_FACTS.md) or off this rig by an instrument that
# is named. Nothing in this block is a guess, and nothing below this block is a
# literal.
# ---------------------------------------------------------------------------

#: Weight byte classes, GiB, from CHECKPOINT_FACTS.md (safetensors index scan).
#: `DENSE_CORE_GIB` is the WHOLE dense side INCLUDING the two optional posts;
#: they are subtracted when the corresponding --no-* flag is given, so the three
#: numbers cannot drift apart.
DENSE_CORE_GIB = 14.846
MTP_GIB = 4.856
VISION_GIB = 0.836
ROUTED_EXPERTS_GIB = 65.040

#: This rig, in TP RANK order -- NOT NVML order. RigLinkCensus measured the
#: ordering trap: the x4-linked 3080 is NVML 0, and the standing CVD string puts
#: the 5090 first, so NVML0 -> cuda:1 -> TP RANK 1. A vector written in NVML
#: order hands the crippled link's share to the 5090.
#: H2D GB/s: the fork's own `rigmon/card_probe` cache (timed 64 MiB pinned
#: transfer per card, driver 595.58.03).
#: Achieved GB/s: what the expert lane actually reaches in a boot
#: (MoEPrefetchScout, off the #439 expert_stats artifacts) -- 13-30 % of link.
DEFAULT_CARDS: Tuple[Tuple[str, int, float, float], ...] = (
    ("RTX 5090 (cuda:0, Gen4 x8)", 32607, 14.42, 3.04),
    ("RTX 3080 (cuda:1, Gen4 x4)", 20480, 6.47, 1.92),
    ("RTX 3080 (cuda:2, Gen4 x8)", 20480, 13.33, 1.74),
)

#: The corridor law, quoted from code rather than from an acceptance doc:
#: `managers/corridor_guard.py` CORRIDOR_LAW_MIB = 1024, CORRIDOR_BAND_FRACTION
#: = 0.20, so the graded band is 819-1229 MiB free per card. The 400 MiB rule
#: that older acceptance prose uses predates this and is NOT the law.
CORRIDOR_LAW_MIB = 1024
CORRIDOR_BAND_FRACTION = 0.20

#: Per-card runtime overhead that is NOT weights, KV, experts or recurrent
#: state: activation peak + graph capture + prefill scratch + hardware residual.
#: The rig's own measured anchor, from `docs/rig-runbook.md:728-734`: 2700 MiB on
#: a 3080 is deliberate, and 2200 "boots, survives the short warmup, reports
#: fired up -- and OOMs in the GDN prefill scratch on the first real prefill".
#: The ledger itemizes these as five separate terms; this script charges the
#: measured aggregate, because an aggregate that WAS measured beats five models
#: that were not.
RUNTIME_OVERHEAD_MIB = 2700

#: Expert-residency hit rate. Measured on boot #439 (`expert_stats`, reported by
#: MoEPrefetchScout): a 45.6 %-RESIDENT set achieved 81.7 % hit under
#: `policy=equal`, against ~98.6 % for an oracle-LFU set of the SAME size. Both
#: points are at the same residency, which is what makes them a comparison of
#: POLICY rather than of size.
MEASURED_RESIDENT_FRACTION = 0.456
MEASURED_MISS_EQUAL = 1.0 - 0.817
MEASURED_MISS_HEAT = 1.0 - 0.986

#: MoEPrefetchScout's published anchor, used ONLY as a cross-check of the model
#: below (printed with its ratio, never silently reconciled): at 34 % cold the
#: decode ceiling is 13.5 tok/s at the achieved H2D and 49.6 tok/s at the link
#: ceiling.
ANCHOR_COLD_FRACTION = 0.34
ANCHOR_CEILING_ACHIEVED_TOK_S = 13.5
ANCHOR_CEILING_LINK_TOK_S = 49.6

#: The Marlin grouped-MoE tile rule: N % 128 == 0 and K % 64 == 0. Read in
#: sgl-kernel source by W4A4Scout.BlackwellFp4Moe and re-verified at the literal
#: SHA by PlannerCensus (marlin.cuh min_thread_n/k = 64, max_thread_n = 256;
#: moe_wna16_marlin.cuh candidate tiles {128,128,256} / {64,128,128} /
#: {64,256,256}, so thread_n is ONLY EVER 128 or 256). For a MoE expert that is
#: w13 (fused gate_up, N = 2 * I_r, K = hidden) and w2 (N = hidden, K = I_r),
#: which reduces to I_r % 64 == 0 on every rank.
#:
#: THE FORK ALREADY GUARANTEES THIS AND THE SCRIPT DOES NOT RE-IMPOSE IT. The
#: expert intermediate is partitioned in INDIVISIBLE UNITS by
#: `fused_moe_triton.layer.moe_uneven_tp_units` +
#: `distributed.utils.partition_units`, and for a compressed-tensors config the
#: unit is lcm(group_size, GPTQ_MARLIN_MIN_THREAD_K) -- 128 on this checkpoint,
#: EXECUTED not read (see fork_moe_units). Every emittable shard is therefore a
#: multiple of 128, so w13 N is a multiple of 256 and w2 K of 128: both rules
#: hold with margin, and upstream #37089's I_r = 160 is arithmetically
#: unemittable through --rank-tp-ratio. The tile check below is kept as an
#: INVARIANT, not a filter, because it is the only thing that would catch a lane
#: handing a width down outside that machinery (units=None), which is the one
#: live hazard on this axis.
MARLIN_N_MULTIPLE = 128
MARLIN_K_MULTIPLE = 64

#: Fallback unit width, used ONLY when the fork is not importable. Labelled as
#: such in the output, never silently substituted.
FALLBACK_MOE_UNIT_WIDTH = 128

#: ASSUMPTION 1 -- the miss model between the two measured points. A power law
#: through ONE measured point per policy: miss = (1 - resident) ** k, with k
#: solved from that point. It is the weakest defensible shape that reproduces
#: the measurement and is monotone. THE ALTERNATIVE, selectable with
#: --miss-model linear, is miss = 1 - resident, i.e. routing is uniform over
#: experts; boot #439 REFUTES that (it would predict 54.4 % miss where 18.3 %
#: was measured), so linear is offered as a pessimistic bound, not as a rival.
MISS_MODELS = ("power", "linear")

#: ASSUMPTION 2 -- the indexer cache holds one index-key row per token per
#: full-attention layer, UNCOMPRESSED. `indexer_compress_ratio` is 4 in the
#: config, and if it applies to the cache the post is 4x smaller. Defaulting to
#: 1 is the conservative direction for a VRAM ledger; --indexer-compress-ratio 4
#: prices the other reading. The fp8 form carries a per-128-element scale, as
#: `mem_cache/memory_pool.py` get_index_k_with_scale_buffer does.
INDEXER_SCALE_BYTES_PER_GROUP = 4
INDEXER_SCALE_GROUP = 128

#: ASSUMPTION 3 -- every one of the 48 layers carries the 512-expert MoE FFN.
#: The config gives one `num_experts` and no per-layer MoE mask, and the measured
#: routed total (65.040 GiB) divides by 512 x 48 into the measured
#: per-expert-per-layer figure (2.71 MiB), which corroborates it. THE ALTERNATIVE
#: would be a dense-FFN prefix; there is no such field in the config and no such
#: tensor class in the checkpoint.

#: ASSUMPTION 4 -- WHERE AN EXPERT LIVES. Two placements are possible and they
#: price the cold set differently, so the choice is a flag, not a default buried
#: in arithmetic.
#:   tp_slice (DEFAULT): plain MoE tensor parallelism. Every rank holds ALL
#:     `num_experts` experts, each sliced to width I_r, and every rank
#:     participates in every routed activation. This is what the fork's runtime
#:     does at world size 3, because expert parallelism at ep_size=3 is asserted
#:     impossible (`fused_moe_triton/layer.py:413` at the pin, per PlannerCensus).
#:   ep_shard: expert parallelism. Each rank owns a DISJOINT subset of experts at
#:     full width, and the cold subset is split by measured H2D exactly as
#:     `expert_offload.resolve_host_shard_ratio` does. Kept because the fork's
#:     capacity solver partitions the MLP family in whole-expert units
#:     (`uneven_perf.py:4117-4121`, mlp_units = num_experts), and because a world
#:     size that permits EP would use it.
#: The invariant below is written so it holds in BOTH placements: the
#: width-weighted hot + cold mass must equal num_experts x n_layers exactly.
EXPERT_PLACEMENTS = ("tp_slice", "ep_shard")

#: Upstream's own PLE geometry, read out of the local object store and
#: independently confirmed against the safetensors headers by MoEPrefetchScout
#: (the formula reproduces 95.3675 GiB against 95.368 GiB measured). Quoted, not
#: modelled:
#:   n_grams    = (ngram_size - 1) * heads_per_ngram          -> 16
#:   row width  = ple_embed_dim // n_grams                    -> 160
#:   table rows = sum(ngram_vocab_size_base + 2i + 1, i < n_grams)
#: and `Qwen4ExpPinnedHostEmbedding` extends `VocabParallelEmbedding` with
#: pin_memory=True (models/qwen4_exp.py:750,803), so the table is ROW-sharded
#: across TP in PINNED HOST memory and each rank gathers only its own rows.
PLE_STAGING_DOUBLE_BUFFER = 2

#: Host-RAM regimes, GiB. BOTH are reported; --host-regime picks which one BINDS
#: the solve. "serving" is what is free while the standing boot serves;
#: "dedicated" is this box with serving stopped, which is what a #1036 window
#: actually gets -- hence the default. SwapTotal is 0 on this rig, so a miss is
#: an OOM kill, not a slowdown.
HOST_RAM_SERVING_GIB = 36.0
HOST_RAM_DEDICATED_GIB = 110.0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Card:
    """One physical card, in TP RANK order."""

    name: str
    total_mib: int
    h2d_gbs: float
    h2d_achieved_gbs: float


@dataclasses.dataclass(frozen=True)
class Geometry:
    """Everything the solve needs, read from config.json. No literals."""

    n_layers: int
    n_full_attention: int
    n_linear_attention: int
    hidden: int
    head_dim: int
    kv_heads: int
    num_experts: int
    experts_per_tok: int
    moe_intermediate: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_compress_ratio: int
    hc_count: int
    hc_lowrank: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ple_embed_dim: int
    ple_conv_kernel_size: int
    n_ple_layers: int
    #: The checkpoint's own ``quantization_config``, carried verbatim so the
    #: fork's ``moe_uneven_tp_units`` can be CALLED on the real thing.
    quant_config: Optional[dict] = None

    @classmethod
    def from_config(cls, path: str) -> "Geometry":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        text = cfg.get("text_config") or cfg
        layer_types = list(text["layer_types"])
        return cls(
            n_layers=int(text["num_hidden_layers"]),
            n_full_attention=sum(1 for t in layer_types if t == "full_attention"),
            n_linear_attention=sum(1 for t in layer_types if t == "linear_attention"),
            hidden=int(text["hidden_size"]),
            head_dim=int(text["head_dim"]),
            kv_heads=int(text["num_key_value_heads"]),
            num_experts=int(text["num_experts"]),
            experts_per_tok=int(text["num_experts_per_tok"]),
            moe_intermediate=int(text["moe_intermediate_size"]),
            indexer_kv_heads=int(text["indexer_kv_heads"]),
            indexer_head_dim=int(text["indexer_head_dim"]),
            indexer_compress_ratio=int(text["indexer_compress_ratio"]),
            hc_count=int(text["hc_count"]),
            hc_lowrank=int(text["hc_lowrank"]),
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
            ngram_size=int(text["ngram_size"]),
            heads_per_ngram=int(text["heads_per_ngram"]),
            ngram_vocab_size_base=int(text["ngram_vocab_size_base"]),
            ple_embed_dim=int(text["ple_embed_dim"]),
            ple_conv_kernel_size=int(text["ple_conv_kernel_size"]),
            n_ple_layers=len(list(text.get("ple_layer_ids") or ())),
            quant_config=cfg.get("quantization_config"),
        )

    @property
    def expert_slots(self) -> int:
        """Total routed expert-layer slots in the model: E x MoE layers."""
        return self.num_experts * self.n_layers

    # -- attention --------------------------------------------------------

    def kv_bytes_per_token(self, dtype_bytes: float) -> float:
        """K and V for every full-attention layer, one token.

        12 layers x 2 KV heads x 256 head_dim x 2 (K+V) x dtype bytes.
        """
        return self.n_full_attention * self.kv_heads * self.head_dim * 2 * dtype_bytes

    def indexer_bytes_per_token(self, dtype_bytes: float, compress: int) -> float:
        """The QSA indexer's own key cache, one token. ASSUMPTION 2."""
        per_row = self.indexer_kv_heads * self.indexer_head_dim * dtype_bytes
        if dtype_bytes < 2:  # fp8 carries a per-group scale
            groups = max(
                1,
                (self.indexer_kv_heads * self.indexer_head_dim) // INDEXER_SCALE_GROUP,
            )
            per_row += groups * INDEXER_SCALE_BYTES_PER_GROUP
        return self.n_full_attention * per_row / max(compress, 1)

    # -- recurrent state --------------------------------------------------

    def gdn_state_bytes_per_request(self, ssm_bytes: float) -> Tuple[float, float]:
        """(recurrent SSM state, conv state) bytes for ONE request, all layers.

        Same shape the fork's own sizer uses (`uneven_perf._mamba_pool_bytes`):
        heads_per_unit = v_heads // k_heads, state = heads_per_unit x v_dim x
        k_dim x dtype, summed over the GDN units and the GDN layers.
        """
        heads_per_unit = max(
            self.linear_num_value_heads // max(self.linear_num_key_heads, 1), 1
        )
        per_unit_layer = (
            heads_per_unit
            * self.linear_value_head_dim
            * self.linear_key_head_dim
            * ssm_bytes
        )
        ssm = per_unit_layer * self.linear_num_key_heads * self.n_linear_attention
        conv_dim = (
            self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
        )
        conv = (
            self.linear_conv_kernel_dim * conv_dim * ssm_bytes * self.n_linear_attention
        )
        return ssm, conv

    def ple_side_state_bytes_per_request(
        self, conv_bytes: float, draft_tokens: int
    ) -> float:
        """PLE per-request GPU side states, quoted from upstream.

        `configs/qwen4_exp.py:99-111` (ref 99c9362e66):
          short_conv_state_shape = (hidden_size * hc_count,
                                    (ple_conv_kernel_size - 1) * ngram_size)
          ngram_context_len      = ngram_size - 1
        and `mem_cache/ple_state_pool.py` allocates ShortConvPool as
        (layers, size+1) + state_shape and NGramPool as (size+1, context_len) in
        int64, plus an `intermediate_*` copy per draft token when speculative
        decoding is on. BOTH pools allocate inside GPU_MEMORY_TYPE_KV_CACHE, so
        they come out of KV capacity rather than sitting beside it.
        """
        if not self.n_ple_layers:
            return 0.0
        channels = self.hidden * self.hc_count
        state_len = max(self.ple_conv_kernel_size - 1, 0) * self.ngram_size
        conv = self.n_ple_layers * channels * state_len * conv_bytes
        ngram = max(self.ngram_size - 1, 0) * 8  # int64 context row
        return conv + ngram + conv * max(draft_tokens, 0)

    # -- weights that do not TP-shard -------------------------------------

    def hyper_connection_bytes(self, dtype_bytes: float) -> float:
        """The hyper-connection mixers, all layers. NOT TP-sharded.

        Measured shapes: input_mix_weight_down (hc_lowrank, hidden x hc_count),
        input_mix_weight_up (hidden x hc_count, hc_lowrank), block_inject_weight
        (hc_count, hidden x hc_count), hc_norm (hidden x hc_count,). The wide
        axis is hidden x hc_count = 10240 on this checkpoint.
        """
        wide = self.hidden * self.hc_count
        params = (
            self.hc_lowrank * wide  # down
            + wide * self.hc_lowrank  # up
            + self.hc_count * wide  # block inject
            + wide  # norm
        )
        return params * dtype_bytes * self.n_layers

    # -- PLE --------------------------------------------------------------

    @property
    def ple_n_grams(self) -> int:
        return max((self.ngram_size - 1) * self.heads_per_ngram, 1)

    @property
    def ple_row_width(self) -> int:
        return self.ple_embed_dim // self.ple_n_grams

    @property
    def ple_table_rows(self) -> int:
        """sum(ngram_vocab_size_base + 2i + 1 for i in range(n_grams))."""
        return sum(
            self.ngram_vocab_size_base + 2 * i + 1 for i in range(self.ple_n_grams)
        )

    def ple_bytes(self, bits: float, group: int, scale_bytes: int) -> float:
        """PLE table bytes at a given weight width, from the row geometry."""
        elements = float(self.ple_table_rows) * self.ple_row_width
        body = elements * bits / 8.0
        scales = (elements / max(group, 1)) * scale_bytes if bits < 16 else 0.0
        return body + scales

    def ple_gather_bytes_per_token(self, dtype_bytes: float) -> float:
        """One token's PLE rows: n_grams rows of row_width, read every step."""
        return self.ple_n_grams * self.ple_row_width * dtype_bytes


# ---------------------------------------------------------------------------
# The mechanism this script MUST agree with: the offload buffer's slot count.
# Quoted from `layers/moe/expert_offload.py` (resident_slot_count :589-592,
# scratch_slot_count :595-612, ExpertResidencyPlanner.buffer_size :496-498)
# rather than re-derived, because a second copy of a formula disagrees silently
# and only on the configuration nobody tested.
# ---------------------------------------------------------------------------

SCRATCH_SLOT_FLOOR = 8
SCRATCH_SLOT_DIVISOR = 4


def scratch_slot_count(resident_count: int, override: Optional[int] = None) -> int:
    """C = max(8, resident // 4), or the SGLANG_MOE_SCRATCH_SLOTS override.

    expert_offload.py:595-612. The override is the CAP, so a pinned value is a
    real layout lever and the script reports what the default costs.
    """
    if override is not None:
        return max(1, int(override))
    return max(SCRATCH_SLOT_FLOOR, resident_count // SCRATCH_SLOT_DIVISOR)


def offload_floor_slots(override: Optional[int] = None) -> int:
    """Smallest buffer an offload lane can have: 1 resident + its scratch.

    `resident_slot_count` floors at 1 and `scratch_slot_count` floors at 8, so no
    rank can run the lane below this. It is a HARD constraint and the solver
    reports it by name when it binds.
    """
    return 1 + scratch_slot_count(1, override)


def _ensure_fork_importable() -> None:
    """Put this checkout's ``python/`` on sys.path, once.

    The script lives at ``scripts/dev/`` inside the tree it reports on, so the
    fork it must agree with is two directories up. Resolving it from ``__file__``
    rather than from the caller's PYTHONPATH is what stops the script from
    silently falling back to its own UNVERIFIED arithmetic when it is run from
    the repo root -- which is how it is run.
    """
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkg = os.path.join(root, "python")
    if os.path.isdir(os.path.join(pkg, "sglang")) and pkg not in sys.path:
        sys.path.insert(0, pkg)


def fork_moe_units(intermediate: int, quant_config: Optional[dict]) -> Tuple[int, str]:
    """Units the FORK will actually partition the expert intermediate into.

    CALLS the fork's own `moe_uneven_tp_units` against a real
    `CompressedTensorsConfig` built from the checkpoint's own
    `quantization_config`, rather than re-deriving which branch fires. That
    matters: eleven agent-hours went into READING that branch table and six
    independent readings got it wrong, because `weight_block_size` is a
    @property and a grep for `self.weight_block_size =` is structurally blind to
    it. Executing the function is the only sound instrument, and a second copy of
    the branch logic here would be exactly the duplication this script exists to
    avoid.
    """
    _ensure_fork_importable()
    if quant_config:
        try:
            from sglang.srt.layers.moe.fused_moe_triton.layer import (
                moe_uneven_tp_units,
            )
            from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
                CompressedTensorsConfig,
            )

            qc = CompressedTensorsConfig.from_config(quant_config)
            units = int(moe_uneven_tp_units(intermediate, qc))
            if units > 0 and intermediate % units == 0:
                return units, (
                    "EXECUTED fused_moe_triton.layer.moe_uneven_tp_units() "
                    "against a real CompressedTensorsConfig built from this "
                    "checkpoint's quantization_config"
                )
        except Exception as e:  # pragma: no cover - import/config shape
            note = f"fork call failed ({type(e).__name__}: {e})"
        else:
            note = "fork call returned an unusable unit count"
    else:
        note = "no quantization_config in the checkpoint config"
    width = FALLBACK_MOE_UNIT_WIDTH
    while width > 1 and intermediate % width:
        width //= 2
    return intermediate // max(width, 1), (
        f"FALLBACK, UNVERIFIED: {note}; assuming a {width}-element unit"
    )


def fork_partition_units(units: int, weights: Sequence[int]) -> Tuple[List[int], str]:
    """The unit vector the FORK will actually serve for these ratio weights.

    CALLS `distributed.utils.partition_units`. This is the read-back that stops
    the provenance line from recording a split that never happened: the
    partition is largest-remainder rounding with every rank >= 1 unit and ties
    broken toward the lower rank index, and at a 5-unit grain it distorts the
    requested ratio badly and silently -- a requested 50/25/25 is served as
    60/20/20. Nothing in the runtime warns about that.
    """
    try:
        _ensure_fork_importable()
        from sglang.srt.distributed.utils import partition_units

        served = [int(x) for x in partition_units(units, list(weights), None)]
        if sum(served) == units and all(s >= 1 for s in served):
            return served, "EXECUTED distributed.utils.partition_units()"
    except Exception:  # pragma: no cover - import shape
        pass
    return _largest_remainder(units, weights), (
        "FALLBACK, UNVERIFIED: largest-remainder, >= 1 unit per rank, ties to "
        "the lower rank index (the documented contract of partition_units)"
    )


def _largest_remainder(units: int, weights: Sequence[int]) -> List[int]:
    n = len(weights)
    units = max(units, n)
    total = float(sum(weights)) or 1.0
    exact = [units * w / total for w in weights]
    base = [max(int(x), 1) for x in exact]
    while sum(base) > units:
        i = int(max(range(n), key=lambda j: (base[j] - exact[j], -j)))
        if base[i] <= 1:
            break
        base[i] -= 1
    while sum(base) < units:
        i = int(max(range(n), key=lambda j: (exact[j] - base[j], -j)))
        base[i] += 1
    return base


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RankPosts:
    """Every byte this rank spends, one field per post. MiB."""

    rank: int
    card: Card
    slot_mib: float = 0.0  # one expert, all layers, at THIS rank's slice width
    intermediate: int = 0
    weights_dense_mib: float = 0.0
    weights_hot_expert_mib: float = 0.0
    kv_mib: float = 0.0
    indexer_mib: float = 0.0
    gdn_state_mib: float = 0.0
    ple_side_state_mib: float = 0.0
    scratch_mib: float = 0.0
    ple_staging_mib: float = 0.0
    ple_resident_mib: float = 0.0
    overhead_mib: float = 0.0
    reserve_mib: float = 0.0
    tokens: int = 0
    hot_per_layer: int = 0
    cold_per_layer: int = 0
    scratch_slots: int = 0

    @property
    def local_per_layer(self) -> int:
        return self.hot_per_layer + self.cold_per_layer

    def posts(self) -> List[Tuple[str, float, str, str]]:
        """(name, mib, mark, derivation) in the ledger's row order."""
        return [
            (
                "model weights (dense shard)",
                self.weights_dense_mib,
                "modeled",
                "TP shard of the dense core + the replicated hyper-connection "
                "mixers (not TP-shardable)",
            ),
            (
                "routed experts (resident)",
                self.weights_hot_expert_mib,
                "modeled",
                f"{self.hot_per_layer} of {self.local_per_layer} local experts "
                f"per layer on GPU, at slice width {self.intermediate}",
            ),
            (
                "MoE expert-offload spill scratch",
                self.scratch_mib,
                "modeled",
                f"{self.scratch_slots} scratch slots/layer of the "
                "fixed-resident buffer [capped by SGLANG_MOE_SCRATCH_SLOTS]",
            ),
            (
                "KV cache",
                self.kv_mib,
                "modeled",
                f"{self.tokens} tokens x full-attention K+V",
            ),
            (
                "QSA indexer cache",
                self.indexer_mib,
                "modeled",
                f"{self.tokens} tokens x index-key rows",
            ),
            (
                "GDN recurrent state",
                self.gdn_state_mib,
                "modeled",
                "SSM state + conv state, this rank's GDN-unit share",
            ),
            (
                "PLE side state",
                self.ple_side_state_mib,
                "modeled",
                "ShortConvPool + NGramPool per request, inside the KV region "
                "(configs/qwen4_exp.py:99-111 @99c9362e66)",
            ),
            (
                "PLE table (resident)",
                self.ple_resident_mib,
                "modeled",
                "n-gram embedding held on GPU instead of pinned host",
            ),
            (
                "PLE gather staging",
                self.ple_staging_mib,
                "modeled",
                "double-buffered row gather [capped by rows/token x batch]",
            ),
            (
                "runtime overhead (aggregate)",
                self.overhead_mib,
                "measured",
                "activation + graph capture + prefill scratch + hardware "
                "residual, rig-runbook.md:728-734",
            ),
        ]

    @property
    def demand_mib(self) -> float:
        return sum(p[1] for p in self.posts())

    @property
    def committed_mib(self) -> float:
        return self.demand_mib + self.reserve_mib

    @property
    def corridor_mib(self) -> float:
        """Free VRAM left on the card once every post is spent."""
        return self.card.total_mib - self.demand_mib


@dataclasses.dataclass
class Layout:
    tokens_global: int
    ranks: List[RankPosts]
    host_need_gib: float
    cold_host_bytes: float
    feasible: bool
    binding: str
    fully_resident: bool
    hot_mass: int  # width-weighted, see ASSUMPTION 4
    cold_mass: int

    @property
    def cold_fraction(self) -> float:
        total = self.hot_mass + self.cold_mass
        return self.cold_mass / total if total else 0.0


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


class Solver:
    def __init__(self, args, geom: Geometry) -> None:
        self.a = args
        self.g = geom
        self.cards: List[Card] = list(args.cards)
        self.n = len(self.cards)

        dense_gib = DENSE_CORE_GIB
        if not args.mtp:
            dense_gib -= MTP_GIB
        if not args.vision:
            dense_gib -= VISION_GIB
        self.dense_total_bytes = dense_gib * _GIB
        self.dense_replicated_bytes = min(
            geom.hyper_connection_bytes(2.0), self.dense_total_bytes
        )
        self.dense_sharded_bytes = self.dense_total_bytes - self.dense_replicated_bytes

        self.routed_total_bytes = ROUTED_EXPERTS_GIB * _GIB
        #: one expert, ALL layers, at FULL intermediate width
        self.full_slot_mib = (
            self.routed_total_bytes / geom.num_experts / _MIB
        )

        self.kv_dtype_bytes = 1.0 if args.kv_dtype == "fp8" else 2.0
        idx_bytes = (
            self.kv_dtype_bytes
            if args.indexer_dtype == "kv"
            else (1.0 if args.indexer_dtype == "fp8" else 2.0)
        )
        self.kv_bpt = geom.kv_bytes_per_token(self.kv_dtype_bytes)
        self.idx_bpt = geom.indexer_bytes_per_token(
            idx_bytes, args.indexer_compress_ratio
        )
        self.cell_bytes = self.kv_bpt + self.idx_bpt

        bs = max(args.batch_size, 1)
        ssm_bytes = 2.0 if args.ssm_dtype == "bf16" else 4.0
        ssm, conv = geom.gdn_state_bytes_per_request(ssm_bytes)
        self.gdn_state_bytes = (ssm + conv) * bs
        self.ple_side_state_bytes = (
            geom.ple_side_state_bytes_per_request(ssm_bytes, args.draft_tokens) * bs
        )

        self.ple_bits = {"bf16": 16.0, "int8": 8.0, "int4": 4.0}[args.ple_dtype]
        self.ple_bytes = geom.ple_bytes(
            self.ple_bits, args.ple_quant_group, args.ple_scale_bytes
        )
        self.ple_gather_bpt = geom.ple_gather_bytes_per_token(2.0)
        self.ple_staging_bytes = self.ple_gather_bpt * bs * PLE_STAGING_DOUBLE_BUFFER

        # Dense weight ratio: the gcd-reduced budgets, which is exactly what
        # `--rank-tp-ratio auto` does (`server_args._resolve_auto_rank_tp_ratio`:
        # "the weights are the gcd-reduced budgets").
        self.budgets = [
            max(c.total_mib - args.reserve_mib - RUNTIME_OVERHEAD_MIB, 0)
            for c in self.cards
        ]
        gcd = 0
        for b in self.budgets:
            gcd = math.gcd(gcd, int(b))
        self.weight_ratio = [int(b) // max(gcd, 1) for b in self.budgets]
        rsum = sum(self.weight_ratio) or 1
        self.weight_share = [r / rsum for r in self.weight_ratio]

        # MoE intermediate: split in INDIVISIBLE units, so every rank's I_r
        # clears the Marlin tile rule. The GRAIN is not assumed: it is obtained
        # by executing the fork's own function, and the SERVED vector is read
        # back from the fork's own partitioner, so the provenance line records
        # the split that will actually happen rather than the one requested.
        self.moe_units_total, self.moe_units_source = fork_moe_units(
            geom.moe_intermediate,
            geom.quant_config if args.derive_moe_units else None,
        )
        if args.moe_units:
            self.moe_units_total = int(args.moe_units)
            self.moe_units_source = f"OVERRIDDEN by --moe-units {args.moe_units}"
        self.moe_unit_width = geom.moe_intermediate // max(self.moe_units_total, 1)
        #: What a plain capacity-driven `--rank-tp-ratio auto` would serve. It is
        #: the REFERENCE, not the answer: capacity says nothing about which card
        #: sits on the crippled x4 link, and the expert slice width decides how
        #: many cold-expert bytes cross that link per token.
        self.capacity_unit_vector, self.moe_partition_source = fork_partition_units(
            self.moe_units_total, self.weight_ratio
        )
        self.requested_share = list(self.weight_share)
        self._apply_moe_vector(self.capacity_unit_vector)

        # Cold-expert split under ep_shard: the fork already resolves this BY
        # CARD UUID from the measured pinned H2D per rank
        # (`expert_offload.resolve_host_shard_ratio`, precedence 2). A faster
        # link carries more of the cold set, because it can refill it. Setting
        # the env by hand is what re-introduces the NVML/rank ordering trap.
        link_sum = sum(c.h2d_gbs for c in self.cards) or 1.0
        self.cold_share = [c.h2d_gbs / link_sum for c in self.cards]

        self.host_ram_gib = (
            args.host_ram_serving_gib
            if args.host_regime == "serving"
            else args.host_ram_dedicated_gib
        )

    def marlin_verdict(self) -> Tuple[bool, List[str]]:
        """Per-rank Marlin tile check on the emitted intermediate vector."""
        notes: List[str] = []
        ok = True
        for r, i_r in enumerate(self.intermediate):
            n_w13 = 2 * i_r
            good = (
                n_w13 % MARLIN_N_MULTIPLE == 0
                and self.g.hidden % MARLIN_K_MULTIPLE == 0
                and self.g.hidden % MARLIN_N_MULTIPLE == 0
                and i_r % MARLIN_K_MULTIPLE == 0
            )
            ok = ok and good
            notes.append(
                f"rank{r} I={i_r}: w13 N={n_w13} K={self.g.hidden}, w2 "
                f"N={self.g.hidden} K={i_r} -- {'OK' if good else 'NO VALID TILE'}"
            )
        return ok, notes

    # -- one candidate ----------------------------------------------------

    def evaluate(self, tokens_global: int) -> Layout:
        a, g = self.a, self.g
        ranks: List[RankPosts] = []
        for r, card in enumerate(self.cards):
            p = RankPosts(rank=r, card=card)
            p.intermediate = self.intermediate[r]
            p.slot_mib = self.full_slot_mib * (
                self.slice_frac[r] if a.expert_placement == "tp_slice" else 1.0
            )
            p.reserve_mib = float(a.reserve_mib)
            p.overhead_mib = float(RUNTIME_OVERHEAD_MIB)
            p.weights_dense_mib = (
                self.dense_sharded_bytes * self.weight_share[r]
                + self.dense_replicated_bytes
            ) / _MIB
            p.gdn_state_mib = (self.gdn_state_bytes * self.weight_share[r]) / _MIB
            p.ple_side_state_mib = self.ple_side_state_bytes / _MIB
            p.ple_staging_mib = self.ple_staging_bytes / _MIB
            p.ple_resident_mib = (
                (self.ple_bytes * self.weight_share[r]) / _MIB
                if a.ple_residency == "gpu"
                else 0.0
            )
            ranks.append(p)

        # Fixed posts are settled; the two ELASTIC posts are KV tokens and the
        # resident-expert band, and they compete for exactly the same bytes.
        # Reserve the offload lane's hard floor BEFORE handing anything to KV,
        # so a "reachable" KV target can never be one that leaves a rank unable
        # to run the lane at all.
        floor = offload_floor_slots(a.moe_scratch_slots)
        kv_room = [
            max(p.card.total_mib - p.committed_mib - floor * p.slot_mib, 0.0)
            for p in ranks
        ]

        # Token vector: proportional to the bytes each rank has left for KV.
        # Under uneven DCP a rank holds ALL KV heads for ITS token slice
        # (kv_heads=2 < tp=3, so heads are REPLICATED and TOKENS are split), so
        # per-token bytes are identical on every rank and proportional-to-room is
        # the exact water-filling solution.
        room_sum = sum(kv_room)
        if room_sum > 0 and tokens_global > 0:
            assigned = 0
            for r, p in enumerate(ranks):
                p.tokens = int(tokens_global * kv_room[r] / room_sum)
                assigned += p.tokens
            if assigned < tokens_global:
                ranks[
                    int(max(range(self.n), key=lambda i: kv_room[i]))
                ].tokens += tokens_global - assigned
        for p in ranks:
            p.kv_mib = p.tokens * self.kv_bpt / _MIB
            p.indexer_mib = p.tokens * self.idx_bpt / _MIB

        # Resident-expert band. `hot` always comes from VRAM; where `cold` comes
        # from depends on the placement (ASSUMPTION 4).
        if a.expert_placement == "tp_slice":
            e_local = [g.num_experts] * self.n
        else:
            e_local = None  # solved jointly with cold, below

        for _ in range(4):
            for r, p in enumerate(ranks):
                cap = (
                    e_local[r]
                    if e_local is not None
                    else max(p.local_per_layer, g.num_experts // self.n)
                )
                room = p.card.total_mib - p.committed_mib + (
                    p.weights_hot_expert_mib + p.scratch_mib
                )
                cand = min(
                    max(int(room / p.slot_mib) if p.slot_mib > 0 else 0, 0), cap
                )
                while cand > 0:
                    c = scratch_slot_count(cand, a.moe_scratch_slots)
                    slots = min(cand + c, cap) if cand < cap else cand
                    if slots * p.slot_mib <= room:
                        break
                    cand -= 1
                p.hot_per_layer = cand
                if e_local is not None:
                    p.cold_per_layer = max(cap - cand, 0)
                p.scratch_slots = (
                    min(scratch_slot_count(cand, a.moe_scratch_slots), cap - cand)
                    if (cand > 0 and p.cold_per_layer > 0)
                    else 0
                )
                p.weights_hot_expert_mib = cand * p.slot_mib
                p.scratch_mib = p.scratch_slots * p.slot_mib

            if e_local is not None:
                break
            # ep_shard: what residency could not fund is cold, split by link.
            total_hot = sum(p.hot_per_layer for p in ranks)
            if total_hot >= g.num_experts:
                scaled = _largest_remainder(
                    g.num_experts, [max(p.hot_per_layer, 1) for p in ranks]
                )
                for r, p in enumerate(ranks):
                    p.hot_per_layer = scaled[r]
                    p.cold_per_layer = 0
                    p.scratch_slots = 0
                    p.weights_hot_expert_mib = scaled[r] * p.slot_mib
                    p.scratch_mib = 0.0
                break
            cold_total = g.num_experts - total_hot
            cold = [int(cold_total * s) for s in self.cold_share]
            drift = cold_total - sum(cold)
            cold[int(max(range(self.n), key=lambda i: self.cold_share[i]))] += drift
            changed = any(p.cold_per_layer != cold[r] for r, p in enumerate(ranks))
            for r, p in enumerate(ranks):
                p.cold_per_layer = cold[r]
            if not changed:
                break

        fully_resident = all(p.cold_per_layer == 0 for p in ranks)

        # Width-weighted expert mass, so the invariant reads the same in both
        # placements (ASSUMPTION 4). Integer arithmetic: no rounding slack.
        width = [
            p.intermediate if a.expert_placement == "tp_slice" else g.moe_intermediate
            for p in ranks
        ]
        hot_mass = sum(
            p.hot_per_layer * width[r] * g.n_layers for r, p in enumerate(ranks)
        )
        cold_mass = sum(
            p.cold_per_layer * width[r] * g.n_layers for r, p in enumerate(ranks)
        )

        cold_host_bytes = sum(
            p.cold_per_layer * p.slot_mib * _MIB for p in ranks
        )
        ple_host = self.ple_bytes if a.ple_residency == "host" else 0.0
        host_need_gib = (cold_host_bytes + ple_host) / _GIB

        binding = ""
        # The tile rule enters the SOLVER, not a post-hoc check. Every reviewer
        # of this axis converged on that: a search that optimises VRAM and
        # scores throughput will happily SELECT a geometry the kernel cannot
        # tile, and the failure is a hard "Invalid thread config" abort rather
        # than a slow layout. At the fork's served grain no such vector is
        # reachable, so this constraint is normally inert -- it is here for the
        # case where it is not (an overridden grain, or a lane handing a width
        # down outside the units machinery).
        tiles_ok, _ = self.marlin_verdict()
        if not tiles_ok:
            return Layout(
                tokens_global=tokens_global,
                ranks=ranks,
                host_need_gib=0.0,
                cold_host_bytes=0.0,
                feasible=False,
                binding=(
                    f"MoE intermediate vector {self.intermediate} has no valid "
                    f"Marlin tile (needs N % {MARLIN_N_MULTIPLE} == 0 and K % "
                    f"{MARLIN_K_MULTIPLE} == 0); this is upstream #37089's abort"
                ),
                fully_resident=fully_resident,
                hot_mass=hot_mass,
                cold_mass=cold_mass,
            )

        feasible = True
        for r, p in enumerate(ranks):
            if p.corridor_mib < p.reserve_mib - 1e-9:
                feasible = False
                binding = (
                    f"card {r} ({p.card.name}) is short "
                    f"{p.reserve_mib - p.corridor_mib:.0f} MiB against the "
                    f"{p.reserve_mib:.0f} MiB reserve"
                )
                break
            if not fully_resident and p.hot_per_layer < 1:
                feasible = False
                binding = (
                    f"card {r} ({p.card.name}) cannot fund the offload lane's "
                    f"hard floor of {floor} slots/layer (1 resident + its "
                    "scratch, expert_offload.py:589-612)"
                )
                break
        if feasible and host_need_gib > self.host_ram_gib + 1e-9:
            feasible = False
            binding = (
                f"HOST RAM in the '{a.host_regime}' regime: need "
                f"{host_need_gib:.3f} GiB (cold experts "
                f"{cold_host_bytes / _GIB:.3f} + pinned PLE "
                f"{ple_host / _GIB:.3f}) against {self.host_ram_gib:.3f} GiB "
                "available, and SwapTotal is 0"
            )
        if feasible and a.max_cold_fraction is not None:
            cf = cold_mass / max(hot_mass + cold_mass, 1)
            if cf > a.max_cold_fraction + 1e-9:
                feasible = False
                binding = (
                    f"cold fraction {cf:.3%} exceeds the --max-cold-fraction "
                    f"objective {a.max_cold_fraction:.3%}"
                )

        return Layout(
            tokens_global=tokens_global,
            ranks=ranks,
            host_need_gib=host_need_gib,
            cold_host_bytes=cold_host_bytes,
            feasible=feasible,
            binding=binding,
            fully_resident=fully_resident,
            hot_mass=hot_mass,
            cold_mass=cold_mass,
        )

    # -- the search -------------------------------------------------------

    def _apply_moe_vector(self, units_vec: Sequence[int]) -> None:
        """Install one candidate expert-intermediate unit vector.

        `served_share` vs `requested_share` is the read-back that keeps the
        provenance line honest: the units layer rounds and never warns, and at
        five units over three ranks a requested 50/25/25 is served as 60/20/20.
        """
        self.moe_unit_vector = list(units_vec)
        self.intermediate = [u * self.moe_unit_width for u in self.moe_unit_vector]
        itotal = sum(self.intermediate) or 1
        self.slice_frac = [i / itotal for i in self.intermediate]
        self.served_share = list(self.slice_frac)
        self.ratio_distortion = max(
            abs(x - y) for x, y in zip(self.requested_share, self.served_share)
        )

    def moe_candidates(self) -> List[List[int]]:
        """Every ORDERED unit vector the fork can serve, >= 1 unit per rank.

        Ordered, not "up to permutation", because the three links are not equal
        (14.42 / 6.47 / 13.33 GB/s in RANK order, and rank 1 is the x4 card).
        The same multiset permuted differently changes the cold-expert bytes that
        must cross the slowest link, so rank ORDER is a decision variable and not
        a presentation detail -- measured by MoEPrefetchScout at up to 2x on the
        transfer bill for one shape.
        """
        n, total = self.n, self.moe_units_total

        def comps(k: int, left: int) -> List[List[int]]:
            if k == 1:
                return [[left]] if left >= 1 else []
            out: List[List[int]] = []
            for first in range(1, left - k + 2):
                for rest in comps(k - 1, left - first):
                    out.append([first] + rest)
            return out

        return comps(n, total) or [[max(total // n, 1)] * n]

    def round_trip(self, units_vec: Sequence[int]) -> Tuple[List[int], bool]:
        """The ratio that produces `units_vec`, verified through the fork.

        A unit vector is its own ratio, but that is a claim about
        `partition_units`, so it is CHECKED rather than asserted -- the whole
        point of this section is that requested and served vectors differ.
        """
        served, _ = fork_partition_units(self.moe_units_total, list(units_vec))
        return served, list(served) == list(units_vec)

    def _solve_kv(self) -> Tuple[Layout, Layout, str]:
        """Largest reachable KV target for the CURRENTLY INSTALLED MoE vector."""
        target = int(self.a.kv_target_tokens)
        reference = self.evaluate(0)
        at_target = self.evaluate(target)
        if at_target.feasible:
            return at_target, reference, "target reachable"
        if not reference.feasible:
            return reference, reference, "NO LAYOUT EXISTS, even at zero KV tokens"
        lo, hi, best = 0, target, reference
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = self.evaluate(mid)
            if cand.feasible:
                best, lo = cand, mid + 1
            else:
                hi = mid - 1
        best.binding = best.binding or at_target.binding
        return best, reference, "target NOT reachable"

    def solve(self) -> Tuple[Layout, Layout, str]:
        """(chosen, reference, note).

        TWO nested solves, in objective order:
          OUTER -- every ORDERED expert-intermediate unit vector the fork can
            serve. Scored on (KV tokens reached, then decode ceiling at the
            ACHIEVED H2D). The outer loop exists because capacity and throughput
            disagree: the widest expert slice wants the biggest card, the
            SMALLEST slice wants the x4-linked card, and on this rig those are
            different ranks. A capacity-only split therefore loads the crippled
            link with cold-expert traffic and binds the whole rig there.
          INNER -- the largest reachable KV target for that vector.

        The reference is the same solve at ZERO KV tokens -- the layout that
        spends everything on residency. It is what the chosen layout's decode
        delta is measured AGAINST, the same way `uneven_perf` compares a chosen
        vector to the VRAM-auto split. A number is not a result without a floor.

        Scoring on ACHIEVED rather than LINK bandwidth is a deliberate, stated
        choice: the two orderings disagree (MoEPrefetchScout measured the
        link-proportional vector winning at the ceiling and losing at today's
        rates, because the measured per-rank H2D is NOT proportional to link
        width -- rank 2 is the slowest achieved despite an x8 link). The
        link-ceiling winner is reported alongside, so the choice is visible and
        re-measurable rather than baked in. --score-at link flips it.
        """
        self.candidates: List[dict] = []
        best_key = None
        best: Optional[Tuple[Layout, Layout, str]] = None
        best_vec: List[int] = list(self.moe_unit_vector)
        for vec in self.moe_candidates():
            self._apply_moe_vector(vec)
            chosen, reference, note = self._solve_kv()
            ach = decode_ceiling(
                chosen,
                self,
                link=False,
                policy=self.a.residency_policy,
                model=self.a.miss_model,
            )
            lnk = decode_ceiling(
                chosen,
                self,
                link=True,
                policy=self.a.residency_policy,
                model=self.a.miss_model,
            )
            self.candidates.append(
                {
                    "units": list(vec),
                    "intermediate": list(self.intermediate),
                    "feasible": chosen.feasible,
                    "tokens": chosen.tokens_global,
                    "cold_fraction": chosen.cold_fraction,
                    "tok_s_achieved": ach.tok_s,
                    "tok_s_link": lnk.tok_s,
                    "bound_rank_achieved": ach.bound_rank,
                    "distortion": self.ratio_distortion,
                    "binding": chosen.binding,
                }
            )
            score = ach.tok_s if self.a.score_at == "achieved" else lnk.tok_s
            key = (bool(chosen.feasible), chosen.tokens_global, score)
            if best_key is None or key > best_key:
                best_key, best, best_vec = key, (chosen, reference, note), list(vec)
        self._apply_moe_vector(best_vec)
        assert best is not None
        return best


# ---------------------------------------------------------------------------
# Throughput objective -- cold fraction is solved against, not left over
# ---------------------------------------------------------------------------


def miss_fraction(resident_fraction: float, policy: str, model: str) -> float:
    resident = min(max(resident_fraction, 0.0), 1.0)
    cold = 1.0 - resident
    if model == "linear" or cold <= 0.0:
        return cold
    anchor = MEASURED_MISS_HEAT if policy == "heat" else MEASURED_MISS_EQUAL
    k = math.log(anchor) / math.log(1.0 - MEASURED_RESIDENT_FRACTION)
    return min(max(cold**k, 0.0), 1.0)


@dataclasses.dataclass
class Ceiling:
    per_rank_tok_s: List[float]
    bound_rank: int
    tok_s: float
    expert_bytes_per_token: List[float]
    ple_bytes_per_token: List[float]


def decode_ceiling(
    layout: Layout, solver: Solver, *, link: bool, policy: str, model: str
) -> Ceiling:
    """tok/s ceiling imposed by the PCIe lane, per rank.

    Both host-resident classes share the lane: the cold routed experts (the
    dominant term) and the pinned PLE rows (n_grams rows/token, row-sharded, so
    each rank pulls its own share). Compute is NOT modelled -- decode here is
    memory/link bound by one to two orders of magnitude, and a compute term
    would only hide the lane.

    Under tp_slice every rank participates in EVERY routed activation (it holds
    a width slice of every expert), so activations do not divide by rank; under
    ep_shard a rank only sees the activations that route to the experts it owns.
    """
    g = solver.g
    acts_per_token = g.experts_per_tok * g.n_layers
    per_rank: List[float] = []
    expert_bytes: List[float] = []
    ple_bytes: List[float] = []
    for p in layout.ranks:
        local = p.local_per_layer
        resident = p.hot_per_layer / local if local else 1.0
        miss = miss_fraction(resident, policy, model)
        if solver.a.expert_placement == "tp_slice":
            acts_r = acts_per_token
        else:
            acts_r = acts_per_token * (local / g.num_experts if g.num_experts else 0.0)
        eb = acts_r * miss * (p.slot_mib * _MIB / g.n_layers)
        pb = (
            solver.ple_gather_bpt / max(len(layout.ranks), 1)
            if solver.a.ple_residency == "host"
            else 0.0
        )
        expert_bytes.append(eb)
        ple_bytes.append(pb)
        total = eb + pb
        gbs = p.card.h2d_gbs if link else p.card.h2d_achieved_gbs
        per_rank.append((gbs * 1e9 / total) if total > 0 else float("inf"))
    bound = int(min(range(len(per_rank)), key=lambda i: per_rank[i]))
    return Ceiling(per_rank, bound, per_rank[bound], expert_bytes, ple_bytes)


def anchor_cross_check(solver: Solver) -> List[str]:
    """Reproduce MoEPrefetchScout's published anchor with THIS model and print
    the ratio. Never reconciled silently: the published pair is checked for
    internal consistency too, and if it is not internally consistent the line
    says so instead of picking the convenient half."""
    g = solver.g
    acts = g.experts_per_tok * g.n_layers
    resident = 1.0 - ANCHOR_COLD_FRACTION
    out: List[str] = []
    for model in MISS_MODELS:
        vals = {}
        for label, link in (("link", True), ("achieved", False)):
            rows = []
            for r, c in enumerate(solver.cards):
                miss = miss_fraction(resident, "equal", model)
                slice_mib = solver.full_slot_mib * (
                    solver.slice_frac[r]
                    if solver.a.expert_placement == "tp_slice"
                    else 1.0 / solver.n
                )
                b = acts * miss * (slice_mib * _MIB / g.n_layers)
                gbs = c.h2d_gbs if link else c.h2d_achieved_gbs
                rows.append(gbs * 1e9 / b if b > 0 else float("inf"))
            vals[label] = min(rows)
        out.append(
            f"  miss-model {model:<6}: link {vals['link']:.1f} tok/s vs published "
            f"{ANCHOR_CEILING_LINK_TOK_S} "
            f"(ratio {vals['link'] / ANCHOR_CEILING_LINK_TOK_S:.2f}x); achieved "
            f"{vals['achieved']:.1f} tok/s vs published "
            f"{ANCHOR_CEILING_ACHIEVED_TOK_S} "
            f"(ratio {vals['achieved'] / ANCHOR_CEILING_ACHIEVED_TOK_S:.2f}x)"
        )
    link_sum = sum(c.h2d_gbs for c in solver.cards)
    ach_sum = sum(c.h2d_achieved_gbs for c in solver.cards)
    out.append(
        "  INTERNAL CHECK of the published pair: its own link/achieved ratio is "
        f"{ANCHOR_CEILING_LINK_TOK_S / ANCHOR_CEILING_ACHIEVED_TOK_S:.2f}x, while "
        f"this rig's link/achieved bandwidth ratio is {link_sum / ach_sum:.2f}x. "
        "The two published numbers therefore do NOT come from one byte count, so "
        "neither is used as a calibration target -- they are an "
        "order-of-magnitude sanity band only."
    )
    return out


# ---------------------------------------------------------------------------
# Emission -- the tree's formats, matched line for line
# ---------------------------------------------------------------------------


def emit_ledger(p: RankPosts) -> List[str]:
    """`mem_ledger/terms.py:407-425` render(), for a card that is not loaded
    yet. Same column layout, same residual row, same verdict vocabulary."""
    rows: List[Tuple[str, float, str, str]] = [
        ("user reserve", p.reserve_mib, "operator", "external headroom")
    ]
    rows.extend(p.posts())
    name_w = max([len(r[0]) for r in rows] + [24])
    mark_w = max([len(r[2]) for r in rows] + [10])
    corridor = p.corridor_mib
    verdict = (
        f"FITS, free corridor {corridor:.0f} MiB"
        if corridor >= p.reserve_mib
        else f"OVERCOMMITTED by {p.reserve_mib - corridor:.0f} MiB"
    )
    lines = [
        f"VRAM ledger for {p.card.name} (ranks: {p.rank}): "
        f"{p.card.total_mib} MiB total -- {verdict}"
    ]
    for name, mib, mark, why in rows:
        lines.append(f"  {name:<{name_w}}  {mib:>7.0f} MiB  {mark:<{mark_w}}  {why}")
    lines.append(f"  {'-' * name_w}  {'-' * 7}      {'-' * mark_w}")
    lines.append(f"  {'user reserve + demand':<{name_w}}  {p.committed_mib:>7.0f} MiB")
    lines.append(
        f"  {'free corridor (residual)':<{name_w}}  {corridor:>7.0f} MiB  "
        f"{'residual':<{mark_w}}  graded against the "
        f"{int(CORRIDOR_LAW_MIB * (1 - CORRIDOR_BAND_FRACTION))}-"
        f"{int(CORRIDOR_LAW_MIB * (1 + CORRIDOR_BAND_FRACTION))} MiB band "
        f"(corridor_guard.CORRIDOR_LAW_MIB={CORRIDOR_LAW_MIB}, band fraction "
        f"{CORRIDOR_BAND_FRACTION})"
    )
    return lines


def report(solver: Solver, chosen: Layout, reference: Layout, note: str) -> int:
    a, g = solver.a, solver.g
    out: List[str] = []
    w = out.append

    w(
        "=== qwen4_exp TP3 LAYOUT SOLVE (#1036, DESK/PREDICTED -- SOLVED, "
        "NOT INSTALLED) ==="
    )
    w(
        f"config    : {a.config}  ({g.n_layers} layers = {g.n_linear_attention} "
        f"linear_attention + {g.n_full_attention} full_attention; "
        f"{g.num_experts} routed experts/layer, top-{g.experts_per_tok}, "
        f"intermediate {g.moe_intermediate})"
    )
    w(
        "cards     : "
        + ", ".join(
            f"rank{r} {c.name} {c.total_mib} MiB (H2D link {c.h2d_gbs} / "
            f"achieved {c.h2d_achieved_gbs} GB/s)"
            for r, c in enumerate(solver.cards)
        )
    )
    w(
        f"placement : {a.expert_placement} "
        + (
            "(every rank holds all experts at slice width I_r; ep_size=3 is "
            "asserted impossible at fused_moe_triton/layer.py:413)"
            if a.expert_placement == "tp_slice"
            else "(disjoint expert subsets at full width; cold split by "
            "measured H2D)"
        )
    )
    w(
        f"byte class: dense {solver.dense_total_bytes / _GIB:.3f} GiB "
        f"(mtp={'in' if a.mtp else 'out'}, vision={'in' if a.vision else 'out'}; "
        f"of which {solver.dense_replicated_bytes / _GIB:.3f} GiB replicated "
        f"hyper-connection mixers) | routed experts "
        f"{solver.routed_total_bytes / _GIB:.3f} GiB "
        f"({solver.full_slot_mib:.1f} MiB per expert over {g.n_layers} layers at "
        "full width)"
    )
    w(
        f"ple geom  : {g.ple_table_rows} rows x {g.ple_row_width} dim "
        f"({g.ple_n_grams} n-gram lookups/token = "
        f"{solver.ple_gather_bpt / 1024:.1f} KiB/token at bf16); table "
        f"{solver.ple_bytes / _GIB:.3f} GiB at {a.ple_dtype} "
        f"(residency={a.ple_residency})"
    )
    w(
        f"kv geom   : cell_size={int(solver.cell_bytes)} B/token = "
        f"{int(solver.kv_bpt)} B KV ({g.n_full_attention} full-attn layers x "
        f"{g.kv_heads} KV heads x {g.head_dim} head_dim x 2 for K+V x "
        f"{solver.kv_dtype_bytes:g} B = {solver.kv_bpt / 1024:.0f} KiB) + "
        f"{int(solver.idx_bpt)} B indexer (compress ratio "
        f"{a.indexer_compress_ratio})"
    )
    w(
        f"state     : GDN {solver.gdn_state_bytes / _MIB:.1f} MiB + PLE side "
        f"{solver.ple_side_state_bytes / _MIB:.1f} MiB at batch {a.batch_size} "
        f"({g.n_linear_attention} GDN layers, {g.linear_num_key_heads} units, "
        f"ssm dtype {a.ssm_dtype}, {a.draft_tokens} draft tokens <= QSA cap "
        f"{g.indexer_compress_ratio})"
    )
    w("")

    # -- the emitted vectors, in the tree's provenance-line format ---------
    tok_vec = [p.tokens for p in chosen.ranks]
    ceil_link = decode_ceiling(
        chosen, solver, link=True, policy=a.residency_policy, model=a.miss_model
    )
    ceil_ach = decode_ceiling(
        chosen, solver, link=False, policy=a.residency_policy, model=a.miss_model
    )
    ref_ach = decode_ceiling(
        reference, solver, link=False, policy=a.residency_policy, model=a.miss_model
    )
    delta = (ceil_ach.tok_s / ref_ach.tok_s - 1.0) * 100.0 if ref_ach.tok_s else 0.0
    marlin_ok, marlin_notes = solver.marlin_verdict()

    w(
        f"CHOSEN TP weight vector: {','.join(map(str, solver.weight_ratio))} "
        f"(materialized dense MiB "
        f"{[int(p.weights_dense_mib) for p in chosen.ranks]}; gcd-reduced from "
        f"budgets {solver.budgets} MiB = total - reserve {a.reserve_mib} - "
        f"overhead {RUNTIME_OVERHEAD_MIB}, the same rule as --rank-tp-ratio auto)"
    )
    w(
        f"CHOSEN MoE intermediate vector: "
        f"{','.join(map(str, solver.moe_unit_vector))} (materialized I_r "
        f"{solver.intermediate} from {solver.moe_units_total} indivisible "
        f"{solver.moe_unit_width}-wide units; Marlin tile floor N % "
        f"{MARLIN_N_MULTIPLE} == 0 and K % {MARLIN_K_MULTIPLE} == 0 -- "
        f"{'OK' if marlin_ok else 'REFUSED'}; requested share "
        f"{[round(x, 3) for x in solver.requested_share]} vs SERVED share "
        f"{[round(x, 3) for x in solver.served_share]}, worst-case distortion "
        f"{solver.ratio_distortion:.1%} -- the units layer rounds and never "
        "warns, so this is the split that happens)"
    )
    w(f"  units grain: {solver.moe_units_source}")
    w(f"  units split: {solver.moe_partition_source}")
    rt_served, rt_ok = solver.round_trip(solver.moe_unit_vector)
    w(
        f"  paste-ready : --rank-moe-ratio "
        f"{','.join(map(str, solver.moe_unit_vector))}  (round-trip through the "
        f"fork's partition_units -> {rt_served} -- "
        f"{'VERIFIED' if rt_ok else 'DOES NOT REPRODUCE, do not use'})"
    )
    w(
        f"  vs capacity : --rank-tp-ratio auto would serve MoE units "
        f"{solver.capacity_unit_vector} (I_r "
        f"{[u * solver.moe_unit_width for u in solver.capacity_unit_vector]}), "
        "which is the VRAM-proportional answer and ignores which card sits on "
        "the x4 link"
    )
    for line in marlin_notes:
        w(f"  tile check: {line}")
    w(
        f"CHOSEN EXPERT RESIDENCY vector: "
        f"{','.join(str(p.hot_per_layer) for p in chosen.ranks)} (materialized "
        f"slots {[p.hot_per_layer * g.n_layers for p in chosen.ranks]} hot / "
        f"{[p.cold_per_layer * g.n_layers for p in chosen.ranks]} cold; "
        f"predicted cold fraction ~{chosen.cold_fraction:.3%} <= ceiling "
        + (
            "none"
            if a.max_cold_fraction is None
            else format(a.max_cold_fraction, ".3%")
        )
        + f"; predicted per-rank local experts/layer "
        f"{[p.local_per_layer for p in chosen.ranks]}; predicted decode step "
        f"{delta:+.1f}% vs the all-residency reference split)"
    )
    w(
        f"CHOSEN MoE SCRATCH vector: "
        f"{','.join(str(p.scratch_slots) for p in chosen.ranks)} (materialized "
        f"MiB {[int(p.scratch_mib) for p in chosen.ranks]}; derived by "
        f"scratch_slot_count = max({SCRATCH_SLOT_FLOOR}, R // "
        f"{SCRATCH_SLOT_DIVISOR})"
        + (
            f", OVERRIDDEN to {a.moe_scratch_slots} by --moe-scratch-slots"
            if a.moe_scratch_slots is not None
            else ""
        )
        + "; capped by SGLANG_MOE_SCRATCH_SLOTS, which is the only lever on this "
        "post)"
    )
    ok = chosen.feasible and chosen.tokens_global >= a.kv_target_tokens
    w(
        f"CHOSEN KV token vector: {','.join(map(str, tok_vec))} (materialized "
        f"global addressable KV {chosen.tokens_global} tokens; predicted ctx "
        f"~{chosen.tokens_global} {'>=' if ok else '<'} floor "
        f"{a.kv_target_tokens}; predicted per-rank capacity {tok_vec}; predicted "
        f"cell_size {int(solver.cell_bytes)} B)"
    )
    w(
        f"floor check: predicted ctx of chosen vector {chosen.tokens_global} "
        f"{'>=' if ok else '<'} {a.kv_target_tokens} (100% of the acceptance "
        f"target) -- {'OK' if ok else 'REFUSED'}"
    )
    if not ok:
        w(
            f"floor check: largest reachable KV target is {chosen.tokens_global} "
            f"tokens. AT THE TARGET the layout was refused because "
            f"{chosen.binding or 'the reserve could not be funded'}. The post it "
            f"TRADED AGAINST is the resident-expert band: cold fraction moves "
            f"from {reference.cold_fraction:.3%} at zero KV to "
            f"{chosen.cold_fraction:.3%} at the reachable target."
        )
    w("")

    # -- the KV sizing emitters -------------------------------------------
    for r, p in enumerate(chosen.ranks):
        pool_bytes = (p.kv_mib + p.indexer_mib) * _MIB
        w(
            f"[TP{r}] KV pool sizing: available_bytes={int(pool_bytes)} "
            f"({pool_bytes / _GIB:.3f} GiB), cell_size={int(solver.cell_bytes)}, "
            f"page_size={a.page_size} -> max_total_num_tokens="
            f"{int(pool_bytes / solver.cell_bytes) if solver.cell_bytes else 0}"
        )
    caps = [p.tokens for p in chosen.ranks]
    agreed = min(caps) if caps else 0
    for r, p in enumerate(chosen.ranks):
        w(
            f"[TP{r}] KV token sizing: rank {r} local capacity {p.tokens} tokens, "
            f"min-reduced across ranks to {agreed} "
            f"({'THIS RANK BINDS' if p.tokens == agreed else 'another rank binds'}"
            f"; {p.tokens - agreed} stranded on this rank). Global addressable "
            f"KV = {agreed} x dcp_size({a.dcp_size})."
        )
    even = agreed * a.dcp_size
    if chosen.tokens_global > even:
        gcd = 0
        for t in caps:
            gcd = math.gcd(gcd, int(t))
        vec = [t // max(gcd, 1) for t in caps]
        w(
            f"Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR="
            f"{','.join(map(str, vec))} to raise max_total_num_tokens from {even} "
            f"to ~{chosen.tokens_global} (per-rank profiled capacity {caps}; "
            f"active vector {[1] * len(caps)} leaves ranks idle)."
        )
    w("")

    for p in chosen.ranks:
        out.extend(emit_ledger(p))
        w("")

    # -- the throughput objective ----------------------------------------
    w("=== DECODE CEILING (cold fraction as an OBJECTIVE, not a residual) ===")
    w(
        f"residency policy {a.residency_policy} / miss model {a.miss_model}: "
        f"per-rank host bytes per decode token -- cold experts "
        f"{[round(b / _MIB, 2) for b in ceil_ach.expert_bytes_per_token]} MiB + "
        f"pinned PLE rows "
        f"{[round(b / 1024, 2) for b in ceil_ach.ple_bytes_per_token]} KiB"
    )
    w(
        "ceiling at LINK H2D     : "
        + ", ".join(f"rank{r} {v:.1f}" for r, v in enumerate(ceil_link.per_rank_tok_s))
        + f" tok/s -> rig {ceil_link.tok_s:.1f} tok/s (rank "
        f"{ceil_link.bound_rank} binds)"
    )
    w(
        "ceiling at ACHIEVED H2D : "
        + ", ".join(f"rank{r} {v:.1f}" for r, v in enumerate(ceil_ach.per_rank_tok_s))
        + f" tok/s -> rig {ceil_ach.tok_s:.1f} tok/s (rank "
        f"{ceil_ach.bound_rank} binds)"
    )
    resident_now = 1.0 - chosen.cold_fraction
    w(
        f"policy lever at the SAME resident set size ({resident_now:.1%} "
        f"resident): miss "
        f"{miss_fraction(resident_now, 'equal', a.miss_model):.3%} under "
        f"policy=equal vs {miss_fraction(resident_now, 'heat', a.miss_model):.3%} "
        "under heat-ranked residency (boot #439 measured 81.7% -> ~98.6% hit at "
        "45.6% residency, i.e. the lever is WHICH experts are resident, not "
        "prefetch)"
    )
    cands = getattr(solver, "candidates", [])
    if cands:
        w("")
        w(
            "candidate expert-slice ORDERINGS (rank order matters: rank 1 is the "
            f"x4 card; scored at --score-at {a.score_at}):"
        )
        w(
            f"  {'units':<10} {'I_r':<20} {'KV tok':>9} {'cold%':>7} "
            f"{'tok/s ach':>10} {'tok/s link':>11}  note"
        )
        best_link = max(cands, key=lambda c: (c["feasible"], c["tok_s_link"]))
        best_ach = max(cands, key=lambda c: (c["feasible"], c["tok_s_achieved"]))
        for c in cands:
            tags = []
            if c["units"] == list(solver.moe_unit_vector):
                tags.append("CHOSEN")
            if c is best_ach:
                tags.append("best@achieved")
            if c is best_link:
                tags.append("best@link")
            if c["units"] == list(solver.capacity_unit_vector):
                tags.append("VRAM-auto")
            if not c["feasible"]:
                tags.append(f"INFEASIBLE: {c['binding'][:60]}")
            w(
                f"  {','.join(map(str, c['units'])):<10} "
                f"{str(c['intermediate']):<20} {c['tokens']:>9} "
                f"{c['cold_fraction'] * 100:>6.1f}% {c['tok_s_achieved']:>10.1f} "
                f"{c['tok_s_link']:>11.1f}  {' | '.join(tags)}"
            )
        if best_ach["units"] != best_link["units"]:
            w(
                "  CONDITIONAL: the two H2D regimes pick DIFFERENT orderings "
                f"({','.join(map(str, best_ach['units']))} at today's achieved "
                f"rates vs {','.join(map(str, best_link['units']))} at the link "
                "ceiling), because the measured per-rank rates are not "
                "proportional to link width. Re-measure after any H2D overlap "
                "change instead of pinning this vector."
            )
    w(f"anchor cross-check at {ANCHOR_COLD_FRACTION:.0%} cold:")
    out.extend(anchor_cross_check(solver))
    w("")

    # -- host RAM, both regimes ------------------------------------------
    ple_host = solver.ple_bytes if a.ple_residency == "host" else 0.0
    need = chosen.host_need_gib
    w("=== HOST RAM (both regimes reported; --host-regime picks the binding one) ===")
    for label, avail_gib in (
        ("serving", a.host_ram_serving_gib),
        ("dedicated", a.host_ram_dedicated_gib),
    ):
        slack = avail_gib - need
        mark = "FITS" if slack >= 0 else f"SHORT by {-slack:.3f} GiB"
        binds = " <- BINDING" if label == a.host_regime else ""
        w(
            f"  {label:<10} available {avail_gib:.3f} GiB, need {need:.3f} GiB "
            f"(cold experts {chosen.cold_host_bytes / _GIB:.3f} + pinned PLE "
            f"{ple_host / _GIB:.3f}) -- {mark}{binds}"
        )
    w(
        "  note: SwapTotal is 0 on this rig, so a host-RAM miss is an OOM kill, "
        "not a slowdown. The PLE table is pinned host memory by construction "
        "(Qwen4ExpPinnedHostEmbedding extends VocabParallelEmbedding, "
        "pin_memory=True), so it is not pageable either."
    )
    w("")

    # -- notes -----------------------------------------------------------
    w("=== NOTES ===")
    w(
        f"  The MoE intermediate is served in {solver.moe_unit_width}-wide units "
        f"({solver.moe_units_total} of them), so every I_r is a multiple of "
        f"{solver.moe_unit_width} and both Marlin rules (N % "
        f"{MARLIN_N_MULTIPLE}, K % {MARLIN_K_MULTIPLE}) hold BY CONSTRUCTION -- "
        "upstream #37089's I_r = 160 is arithmetically unemittable through "
        f"--rank-tp-ratio. At TP1 this checkpoint is clean anyway (w13 "
        f"N={2 * g.moe_intermediate}, w2 K={g.moe_intermediate}). The tile check "
        "is ALSO a feasibility constraint inside the solve, so a tile-invalid "
        "ordering can never be selected even at an overridden grain; the "
        "invariant below is the belt-and-braces restatement."
    )
    w(
        "  The expert-offload spill scratch is now a ledger term "
        "(mem_ledger/engine.py TERM_MOE_OFFLOAD_SCRATCH). Before that it was "
        "accounted only by MEASUREMENT (KV sizing reads live free VRAM after the "
        "offload releases, model_runner_kv_cache_mixin.py:350-353), which is "
        "anonymous, unplannable, and lost entirely under "
        "SGLANG_MOE_OFFLOAD_KV_REGAIN=0 (environ.py:1507-1508)."
    )
    w(
        f"  The scratch band is NOT small at this expert count: the derived C = "
        f"max({SCRATCH_SLOT_FLOOR}, R // {SCRATCH_SLOT_DIVISOR}) costs "
        f"{[int(p.scratch_mib) for p in chosen.ranks]} MiB per card here. It is "
        "the one post with a direct override, so it is the first lever to reach "
        "for -- and lowering it trades against wave-splitting in "
        "breakable_offload, not against correctness."
    )

    # -- INVARIANTS. The script must exit nonzero rather than print an
    #    infeasible layout as an answer.
    failures: List[str] = []
    if not chosen.feasible:
        failures.append(f"NO FEASIBLE LAYOUT: {chosen.binding}")
    for r, p in enumerate(chosen.ranks):
        budget = p.card.total_mib - p.reserve_mib
        if p.demand_mib > budget + 1e-6:
            failures.append(
                f"INVARIANT 1 BROKEN on card {r} ({p.card.name}): posts "
                f"{p.demand_mib:.1f} MiB > capacity - reserve {budget:.1f} MiB"
            )
    expected_mass = g.num_experts * g.moe_intermediate * g.n_layers
    got_mass = chosen.hot_mass + chosen.cold_mass
    if got_mass != expected_mass:
        failures.append(
            f"INVARIANT 2 BROKEN: width-weighted hot {chosen.hot_mass} + cold "
            f"{chosen.cold_mass} = {got_mass}, expected {g.num_experts} x "
            f"{g.moe_intermediate} x {g.n_layers} = {expected_mass}"
        )
    # The same identity in RAW COUNTS, which is how the acceptance criterion is
    # worded ("hot + cold == 512 x 48"). Under tp_slice every rank holds all
    # num_experts experts at a slice width, so the per-rank count is the one
    # that must be exact; under ep_shard the ranks partition the set and the
    # SUM is. Both are checked, so neither placement can hide a lost expert.
    per_layer = [p.hot_per_layer + p.cold_per_layer for p in chosen.ranks]
    if a.expert_placement == "tp_slice":
        bad = [r for r, v in enumerate(per_layer) if v != g.num_experts]
        if bad:
            failures.append(
                f"INVARIANT 2 BROKEN (raw counts): rank(s) {bad} do not hold all "
                f"{g.num_experts} experts per layer -- got {per_layer}; "
                f"x {g.n_layers} layers should be {g.expert_slots} slots on "
                "every rank"
            )
    elif sum(per_layer) != g.num_experts:
        failures.append(
            f"INVARIANT 2 BROKEN (raw counts): the ranks partition "
            f"{sum(per_layer)} experts per layer, expected {g.num_experts} "
            f"({g.expert_slots} slots over {g.n_layers} layers)"
        )
    if not marlin_ok:
        failures.append(
            "INVARIANT 3 BROKEN: the emitted MoE intermediate vector has no "
            "valid Marlin tile on at least one rank (see the tile check above)"
        )
    w("")
    w("=== INVARIANTS ===")
    if failures:
        for f in failures:
            w(f"  {f}")
        w("  VERDICT: REFUSED -- this layout is not an answer.")
    else:
        hot_slots = sum(p.hot_per_layer * g.n_layers for p in chosen.ranks)
        cold_slots = sum(p.cold_per_layer * g.n_layers for p in chosen.ranks)
        w(
            "  INVARIANT 1 OK: every card's posts <= capacity - reserve (margins "
            f"MiB {[int(p.card.total_mib - p.reserve_mib - p.demand_mib) for p in chosen.ranks]})"
        )
        w(
            f"  INVARIANT 2 OK: width-weighted hot {chosen.hot_mass} + cold "
            f"{chosen.cold_mass} = {got_mass} == {g.num_experts} x "
            f"{g.moe_intermediate} x {g.n_layers}; raw slot counts hot "
            f"{hot_slots} + cold {cold_slots} = {hot_slots + cold_slots} over "
            f"{len(chosen.ranks)} rank(s) at slice widths "
            f"{[p.intermediate for p in chosen.ranks]}"
        )
        w(
            f"  INVARIANT 3 OK: every rank's I_r clears N % {MARLIN_N_MULTIPLE} "
            f"== 0 and K % {MARLIN_K_MULTIPLE} == 0 (enforced as a feasibility "
            "constraint in the solve, so a tile-invalid candidate is REFUSED "
            "before it can be scored -- see the candidate table)"
        )
        w(f"  VERDICT: {'ACCEPTED' if ok else 'ACCEPTED BELOW TARGET'} ({note})")

    print("\n".join(out))
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _card(spec: str) -> Card:
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "--card wants NAME:TOTAL_MIB[:H2D_GBS[:H2D_ACHIEVED_GBS]]"
        )
    return Card(
        name=parts[0],
        total_mib=int(parts[1]),
        h2d_gbs=float(parts[2]) if len(parts) > 2 else 1.0,
        h2d_achieved_gbs=float(parts[3]) if len(parts) > 3 else 1.0,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="qwen4_exp_layout.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config",
        default="/spinning/qwen38-flash-next/ckpt/config.json",
        help="checkpoint config.json (the geometry is READ, never assumed)",
    )
    ap.add_argument(
        "--card",
        action="append",
        dest="cards",
        type=_card,
        metavar="NAME:MIB[:H2D[:ACHIEVED]]",
        help="one per TP RANK, in RANK order (the x4 card is NVML0 = cuda:1 = "
        "rank 1). Repeatable. Default: this rig.",
    )
    ap.add_argument("--kv-target-tokens", type=int, default=262144)
    ap.add_argument("--kv-dtype", choices=("fp8", "bf16"), default="fp8")
    ap.add_argument(
        "--indexer-dtype",
        choices=("kv", "fp8", "bf16"),
        default="kv",
        help="'kv' follows --kv-dtype",
    )
    ap.add_argument("--indexer-compress-ratio", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--draft-tokens",
        type=int,
        default=0,
        help="speculative_num_draft_tokens. QSA raises unless this is <= the "
        "checkpoint's indexer_compress_ratio "
        "(qwen_sparse_attn_backend.py:266 @99c9362e66), so the script refuses "
        "rather than pricing an impossible layout.",
    )
    ap.add_argument("--page-size", type=int, default=1)
    ap.add_argument("--dcp-size", type=int, default=1)
    ap.add_argument("--ssm-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--reserve-mib", type=int, default=CORRIDOR_LAW_MIB)
    ap.add_argument(
        "--expert-placement", choices=EXPERT_PLACEMENTS, default="tp_slice"
    )
    ap.add_argument(
        "--moe-units",
        type=int,
        default=None,
        help="override the expert-intermediate unit COUNT. Default: whatever "
        "the fork's own moe_uneven_tp_units() returns for this checkpoint.",
    )
    ap.add_argument(
        "--no-derive-moe-units",
        dest="derive_moe_units",
        action="store_false",
        default=True,
        help="do not import the fork to derive the unit grain (uses the "
        "labelled UNVERIFIED fallback instead).",
    )
    ap.add_argument(
        "--moe-scratch-slots",
        type=int,
        default=None,
        help="SGLANG_MOE_SCRATCH_SLOTS equivalent. Default: the derived "
        "max(8, R // 4).",
    )
    ap.add_argument("--ple-dtype", choices=("bf16", "int8", "int4"), default="int4")
    ap.add_argument("--ple-residency", choices=("host", "gpu"), default="host")
    ap.add_argument("--ple-quant-group", type=int, default=32)
    ap.add_argument("--ple-scale-bytes", type=int, default=2)
    ap.add_argument(
        "--miss-model",
        choices=MISS_MODELS,
        default="power",
        help="power: anchored on boot #439. linear: uniform routing (REFUTED by "
        "#439, kept as a pessimistic bound).",
    )
    ap.add_argument(
        "--residency-policy",
        choices=("equal", "heat"),
        default="heat",
        help="heat = heat-ranked residency (the measured 98.6%% hit); equal = "
        "policy=equal (81.7%%).",
    )
    ap.add_argument("--max-cold-fraction", type=float, default=None)
    ap.add_argument(
        "--score-at",
        choices=("achieved", "link"),
        default="achieved",
        help="which H2D figure the OUTER solve scores expert-slice orderings "
        "on. 'achieved' is today's measured 3.04/1.92/1.74 GB/s; 'link' is the "
        "14.42/6.47/13.33 ceiling. The two disagree about the best ordering, so "
        "the choice is explicit and both are always reported.",
    )
    ap.add_argument("--mtp", action="store_true", default=True)
    ap.add_argument("--no-mtp", dest="mtp", action="store_false")
    ap.add_argument("--vision", action="store_true", default=True)
    ap.add_argument("--no-vision", dest="vision", action="store_false")
    ap.add_argument("--host-ram-serving-gib", type=float, default=HOST_RAM_SERVING_GIB)
    ap.add_argument(
        "--host-ram-dedicated-gib", type=float, default=HOST_RAM_DEDICATED_GIB
    )
    ap.add_argument(
        "--host-regime",
        choices=("serving", "dedicated"),
        default="dedicated",
        help="which host-RAM figure BINDS the solve. Default 'dedicated': a "
        "#1036 window takes the box, so the standing boot's ~36 GiB is not the "
        "constraint. Both are always reported.",
    )
    a = ap.parse_args(argv)

    if not a.cards:
        a.cards = [Card(*c) for c in DEFAULT_CARDS]

    try:
        geom = Geometry.from_config(a.config)
    except (OSError, KeyError, ValueError) as e:
        print(f"cannot read geometry from {a.config}: {e}", file=sys.stderr)
        return 2

    # QSA feasibility bounds, quoted rather than modelled: the backend RAISES
    # instead of degrading, so a layout that violates them is not a slower
    # layout, it is a boot that dies.
    if a.draft_tokens > geom.indexer_compress_ratio:
        print(
            f"--draft-tokens {a.draft_tokens} exceeds the checkpoint's "
            f"indexer_compress_ratio {geom.indexer_compress_ratio}; "
            "qwen_sparse_attn_backend.py:266 raises NotImplementedError there "
            "(the pending index-key ring keys state by position % ratio, so a "
            "wider verify window collides inside one forward)",
            file=sys.stderr,
        )
        return 2
    if a.moe_units and geom.moe_intermediate % a.moe_units:
        print(
            f"--moe-units {a.moe_units} does not divide the intermediate "
            f"{geom.moe_intermediate}",
            file=sys.stderr,
        )
        return 2

    solver = Solver(a, geom)
    chosen, reference, note = solver.solve()
    return report(solver, chosen, reference, note)


if __name__ == "__main__":
    sys.exit(main())
