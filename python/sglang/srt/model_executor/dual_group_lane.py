# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi-group runtime (#121/#274) slice B: the in-process PD lane, built.

Slice A proved the algebra (``distributed/dual_group.py``); this module makes
one lane REAL: a full-width (weight-TP=1) module tree whose parallel linears
are SHELLS over N sharded part trees living in the same process.  For the
first lane (lane 0, the PD lane on the shared card) the parts are

* the resident serving-group rank's modules (SHARED bytes -- the same tensor
  objects, verified by ``data_ptr`` identity, never copied), and
* one freshly loaded COMPLEMENT tree per complement segment of the plan
  (bytes that exist nowhere else in this process -- they are what the other
  cards hold).

The lane group's collectives are local tensor ops (``local_column_gather``,
``local_row_reduce``); the lane has NO communicator, so nothing here may ever
call into a process group.  Every function in this file is rank-local by
contract (the rank-local-before-collective rule); a collective on this path
is a bug, not a tuning problem.

Multi-group form (#274 addendum 8): nothing in here is pair-hardcoded.  A
plan may have any number of segments; shells take a SEQUENCE of parts; the
scheduler holds a LIST of lanes.  Slice B instantiates exactly one lane.

Known two-segment assumptions are named in the slice report, not hidden.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from sglang.srt.distributed.dual_group import (
    NestedGroupPlan,
    check_nesting,
    derive_nested_plan,
    format_vram_posts,
    local_column_gather,
    local_row_reduce,
    local_row_split,
    transformer_nesting_probes,
)
from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    scoped_tp_partition_ratios,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

__all__ = [
    "LaneColumnParallelShell",
    "LaneRowParallelShell",
    "LaneVocabEmbeddingShell",
    "LaneLmHeadShell",
    "assemble_lane_shells",
    "build_lane_model",
    "derive_lane_plan",
    "lane_geometry_override",
    "verify_shared_bytes",
]


# ---------------------------------------------------------------------------
# Geometry scopes
# ---------------------------------------------------------------------------


def lane_geometry_override(fast_size: int, fast_rank: int):
    """The ParallelContext override for building/loading one lane part.

    ``fast_rank`` is a rank OF THE LANE GROUP (size ``fast_size``); the full
    hull view is ``(1, 0)``.  Mirrors the weightless/draft-solo build override
    (model_runner.py): tp + moe_tp + attn_tp move together, and the linears
    cache these values at construction, so the override holds through every
    later forward.  The dcp fields are forced OFF: the lane's KV pool is its
    own, un-token-sharded pool; a live ``dcp_enabled`` read during a lane
    forward must never see the serving group's DCP geometry.
    """
    return get_parallel().override(
        tp_size=fast_size,
        tp_rank=fast_rank,
        moe_tp_size=fast_size,
        moe_tp_rank=fast_rank,
        attn_tp_size=fast_size,
        attn_tp_rank=fast_rank,
        dcp_enabled=False,
        dcp_size=1,
        dcp_rank=0,
        attn_dcp_size=1,
        attn_dcp_rank=0,
    )


def installed_family_ratios(tp_size: int) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """The per-family vectors currently installed for a group of ``tp_size``.

    The base vector governs attention/GDN; ``mlp``/``moe``/``vocab`` may carry
    their own vectors (--rank-mlp-ratio etc.).  Only vectors of matching
    length are returned -- the same ``len(ratios) == tp_size`` discriminator
    the runtime itself applies.
    """
    fams = []
    for name in ("mlp", "moe", "vocab"):
        vec = get_tp_partition_ratios(name)
        base = get_tp_partition_ratios(None)
        if vec is not None and vec != base and len(vec) == tp_size:
            fams.append((name, tuple(int(w) for w in vec)))
    return tuple(fams)


def derive_lane_plan(server_args, model_config) -> NestedGroupPlan:
    """Derive + verify the lane plan from the booted serving-group config.

    Boot-time gate: raises with the full nesting report when the installed
    ratio pair does not nest for THIS model's unit counts (the 65-of-497
    class; e.g. plain ``[6,1,1]`` does NOT nest for a 4-kv-head model -- the
    min-1 bump splits the 4 kv groups ``[2,1,1]`` while the lane wants
    ``[3,1]``.  A base of ``[2,1,1]`` with mlp/vocab family vectors carrying
    the capacity spread does nest, exactly).
    """
    base = get_tp_partition_ratios(None)
    if not base or len(base) != server_args.tp_size:
        raise ValueError(
            "--dual-group-lane needs an explicit uneven-TP plan "
            "(--rank-tp-ratio with one integer weight per rank); the even "
            "default has no shared-rank nesting to exploit."
        )
    plan = derive_nested_plan(base)
    plan = NestedGroupPlan(
        big_ratio=plan.big_ratio,
        segments=plan.segments,
        family_ratios=installed_family_ratios(server_args.tp_size),
    )

    cfg = model_config.hf_text_config
    intermediate = getattr(cfg, "intermediate_size", None)
    num_experts = getattr(cfg, "num_experts", None)
    linear_attn_units = getattr(cfg, "gdn_tp_units", None) or getattr(
        cfg, "linear_num_key_heads", None
    )
    vocab_units = None
    padding = 64  # DEFAULT_VOCAB_PADDING_SIZE
    vocab_size = getattr(model_config, "vocab_size", None)
    if vocab_size:
        vocab_units = (vocab_size + padding - 1) // padding
    probes = transformer_nesting_probes(
        plan,
        num_attention_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        intermediate_size=intermediate,
        num_experts=num_experts,
        linear_attn_units=linear_attn_units,
        vocab_units=vocab_units,
    )
    check_nesting(plan, probes)
    logger.info("dual-group lane plan verified: %s", plan.describe())
    return plan


# ---------------------------------------------------------------------------
# Shells: full-width forwards over N sharded parts, collectives replaced by
# local ops.  Parts are stored as a plain tuple so they are NOT registered as
# submodules (their parameters belong to their own trees; the shared parts are
# the serving group's resident modules and must not be double-walked).
# ---------------------------------------------------------------------------


class LaneColumnParallelShell(nn.Module):
    """Full-width column-parallel forward: per-sub-output concat of N parts.

    Covers ColumnParallelLinear, MergedColumnParallelLinear and
    QKVParallelLinear alike -- ``output_partition_sizes`` of each part lists
    that part's sub-output widths in packing order, which is exactly what
    ``local_column_gather`` needs to rebuild the canonical
    ``[q_all, k_all, v_all]`` (or ``[gate_all, up_all]``) layout.
    """

    def __init__(self, parts: Sequence[nn.Module]):
        super().__init__()
        if not parts:
            raise ValueError("LaneColumnParallelShell: no parts.")
        self._lane_parts = tuple(parts)
        self._lane_sub_sizes = tuple(
            tuple(int(s) for s in p.output_partition_sizes) for p in parts
        )
        n_sub = len(self._lane_sub_sizes[0])
        for f, sizes in enumerate(self._lane_sub_sizes):
            if len(sizes) != n_sub:
                raise ValueError(
                    f"LaneColumnParallelShell: part 0 packs {n_sub} sub-outputs "
                    f"but part {f} packs {len(sizes)} -- these are not shards "
                    "of the same layer."
                )
        for p in parts:
            if getattr(p, "gather_output", False):
                raise ValueError(
                    "LaneColumnParallelShell over a gather_output=True linear: "
                    "the shell IS the gather; a part that gathers would call a "
                    "real collective inside the lane."
                )
            if p.skip_bias_add and p.bias is not None:
                raise NotImplementedError(
                    "LaneColumnParallelShell: skip_bias_add with a bias is not "
                    "composed yet (no vehicle model needs it)."
                )

    def forward(self, input_):
        parts = self._lane_parts
        outs = [
            p.quant_method.apply(
                p, input_, None if (p.bias is None or p.skip_bias_add) else p.bias
            )
            for p in parts
        ]
        return local_column_gather(outs, self._lane_sub_sizes), None


class LaneRowParallelShell(nn.Module):
    """Full-width row-parallel forward: split at the unit boundary, apply each
    part's shard, add (== a real N-rank all-reduce, executed locally).

    The bias convention mirrors RowParallelLinear (bias applied once, by rank
    0): the shell adds part 0's bias exactly once after the reduce.
    """

    def __init__(self, parts: Sequence[nn.Module]):
        super().__init__()
        if not parts:
            raise ValueError("LaneRowParallelShell: no parts.")
        self._lane_parts = tuple(parts)
        self._lane_in_sizes = tuple(int(p.input_size_per_partition) for p in parts)
        p0 = parts[0]
        self.skip_bias_add = p0.skip_bias_add

    def forward(self, input_, skip_all_reduce: bool = False, forward_batch=None):
        # skip_all_reduce is accepted for signature parity with
        # RowParallelLinear; the lane's "all-reduce" is the local sum below and
        # is never optional (without it the output is a partial product).
        parts = self._lane_parts
        pieces = local_row_split(input_, self._lane_in_sizes)
        outs = [
            p.quant_method.apply(p, piece, None)
            for p, piece in zip(parts, pieces)
        ]
        out = local_row_reduce(outs)
        bias = parts[0].bias
        if bias is not None and not self.skip_bias_add:
            out = out + bias
            bias = None
        return out, (bias if self.skip_bias_add else None)


class LaneVocabEmbeddingShell(nn.Module):
    """Full-vocab embedding over N vocab-sharded parts: per-part masked lookup
    (each part's quant_method on its own resident/quantized bytes), then a
    local sum -- the all-reduce of VocabParallelEmbedding.forward, executed
    locally.  Mirrors that forward minus the collective and minus the
    symmetric-memory context (lane buffers are never symmetric)."""

    def __init__(self, parts: Sequence[nn.Module]):
        super().__init__()
        if not parts:
            raise ValueError("LaneVocabEmbeddingShell: no parts.")
        self._lane_parts = tuple(parts)

    def forward(self, input_):
        from sglang.srt.layers.vocab_parallel_embedding import (
            get_masked_input_and_mask,
        )

        out = None
        for p in self._lane_parts:
            idx = p.shard_indices
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                idx.org_vocab_start_index,
                idx.org_vocab_end_index,
                idx.num_org_vocab_padding,
                idx.added_vocab_start_index,
                idx.added_vocab_end_index,
            )
            part_out = p.quant_method.embedding(p, masked_input.long())
            part_out.masked_fill_(input_mask.unsqueeze(-1), 0)
            out = part_out if out is None else out + part_out
        return out


class _LaneLmHeadQuantMethod:
    """Quant-method adapter so the logits processor's existing dispatch
    (``should_apply_lm_head_quant_method``) computes lane logits through the
    parts' own (possibly quantized-resident) heads and concatenates the vocab
    shards in rank order -- the local replacement of the uneven-vocab logits
    gather."""

    def __init__(self, shell: "LaneLmHeadShell"):
        self._shell = shell

    def apply(self, layer, x, bias=None):
        parts = self._shell._lane_parts
        outs = [p.quant_method.apply(p, x, bias) for p in parts]
        return torch.cat(outs, dim=-1)


class LaneLmHeadShell(nn.Module):
    """Stands in for ParallelLMHead in the hull: exposes a ``quant_method``
    whose ``apply`` produces FULL-vocab logits from the N parts.  The
    ``qweight`` attribute aliases part 0's packed bytes only so the GGUF
    branch of ``should_apply_lm_head_quant_method`` recognizes the head; the
    compute never reads it directly."""

    def __init__(self, parts: Sequence[nn.Module]):
        super().__init__()
        if not parts:
            raise ValueError("LaneLmHeadShell: no parts.")
        self._lane_parts = tuple(parts)
        self.quant_method = _LaneLmHeadQuantMethod(self)
        self.bias = None
        p0 = parts[0]
        if hasattr(p0, "qweight"):
            # Plain attribute alias (not a registered parameter): dispatch key
            # for the GGUF quantized-resident branch.
            self.__dict__["qweight"] = p0.qweight

    def forward(self, input_):
        raise RuntimeError(
            "LaneLmHeadShell.forward is not used; logits go through "
            "quant_method.apply (same contract as ParallelLMHead)."
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

#: Column-parallel modules that are NOT plain GEMMs and must not be shelled --
#: their weights are consumed by other ops (GDN conv weights feed
#: RadixLinearAttention directly).  They are composed by value instead.
_COMPOSED_LINEAR_SUFFIXES = ("conv1d",)


def _module_dict(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def assemble_lane_shells(
    hull: nn.Module, part_models: Sequence[nn.Module]
) -> Dict[str, int]:
    """Replace every parallel linear / vocab layer of ``hull`` with a shell
    over the same-named modules of ``part_models`` (lane-rank order).

    Returns a counter dict for the boot log.  Raises when a parallel module of
    the hull has no counterpart in a part tree -- a silent skip here would be
    a silently wrong forward.
    """
    from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    part_dicts = [_module_dict(m) for m in part_models]
    hull_modules = list(hull.named_modules())
    counts = {"column": 0, "row": 0, "embedding": 0, "lm_head": 0, "composed": 0}
    composed_prefixes: List[str] = []

    def parts_for(name: str) -> List[nn.Module]:
        parts = []
        for f, d in enumerate(part_dicts):
            if name not in d:
                raise ValueError(
                    f"dual-group lane: hull module {name!r} has no counterpart "
                    f"in lane part {f} -- the part trees do not match the hull "
                    "architecture."
                )
            parts.append(d[name])
        return parts

    def set_by_name(name: str, new: nn.Module) -> None:
        parent_name, _, attr = name.rpartition(".")
        parent = hull.get_submodule(parent_name) if parent_name else hull
        setattr(parent, attr, new)

    for name, module in hull_modules:
        if not name:
            continue
        if isinstance(module, ParallelLMHead):
            set_by_name(name, LaneLmHeadShell(parts_for(name)))
            counts["lm_head"] += 1
        elif isinstance(module, VocabParallelEmbedding):
            set_by_name(name, LaneVocabEmbeddingShell(parts_for(name)))
            counts["embedding"] += 1
        elif isinstance(module, RowParallelLinear):
            set_by_name(name, LaneRowParallelShell(parts_for(name)))
            counts["row"] += 1
        elif isinstance(module, ColumnParallelLinear):
            if name.endswith(_COMPOSED_LINEAR_SUFFIXES):
                parent_name = name.rsplit(".", 1)[0]
                _compose_column_weights_inplace(
                    module,
                    parts_for(name),
                    _gdn_conv_sub_sizes(parts_for(parent_name)),
                )
                counts["composed"] += 1
                composed_prefixes.append(name + ".")
            else:
                set_by_name(name, LaneColumnParallelShell(parts_for(name)))
                counts["column"] += 1
    hull._lane_composed_prefixes = tuple(composed_prefixes)
    return counts


def _gdn_conv_sub_sizes(gdn_parts: Sequence[nn.Module]) -> List[List[int]]:
    """Per-part conv-channel group widths of a GDN mixer: [q, k, v] groups,
    each sized by that part's LOCAL head counts.  The conv1d linear itself
    reports one flat partition; the 3-group layout is a property of the
    mamba_v2 packed weight, so it is derived from the owning mixer."""
    sizes = []
    for p in gdn_parts:
        if not hasattr(p, "local_num_k_heads"):
            raise ValueError(
                f"composed conv linear inside {type(p).__name__} without GDN "
                "head attributes -- unknown packed-conv layout, refusing to "
                "guess."
            )
        k = int(p.local_num_k_heads) * int(p.head_k_dim)
        v = int(p.local_num_v_heads) * int(p.head_v_dim)
        sizes.append([k, k, v])
    return sizes


def _compose_column_weights_inplace(
    hull_linear: nn.Module,
    parts: Sequence[nn.Module],
    per_part_sub_sizes: Sequence[Sequence[int]],
) -> None:
    """Fill a non-GEMM column-parallel weight (GDN conv1d) by value.

    The full weight is the per-sub-group concatenation of the parts' shards
    along dim 0 (channels), same regrouping rule as ``local_column_gather``.
    IN-PLACE ``copy_`` on purpose: RadixLinearAttention captured a VIEW of
    this storage at construction; reassigning ``.data`` would strand it.
    """
    n_sub = len(per_part_sub_sizes[0])
    pieces = []
    for s in range(n_sub):
        for f, p in enumerate(parts):
            sizes = per_part_sub_sizes[f]
            if p.weight.data.shape[0] != sum(sizes):
                raise ValueError(
                    f"composed column linear: part {f} weight dim0 "
                    f"{p.weight.data.shape[0]} != declared groups {sizes}."
                )
            off = sum(sizes[:s])
            pieces.append(p.weight.data[off : off + sizes[s]])
    full = torch.cat(pieces, dim=0)
    if full.shape != hull_linear.weight.shape:
        raise ValueError(
            f"composed weight shape {tuple(full.shape)} != hull "
            f"{tuple(hull_linear.weight.shape)} for a composed column linear."
        )
    hull_linear.weight.data.copy_(full)
    if getattr(hull_linear, "bias", None) is not None:
        raise NotImplementedError("composed column linear with bias")


def _finalize_hull_params(
    hull: nn.Module,
    shared_model: nn.Module,
    part_models: Sequence[nn.Module],
) -> Dict[str, int]:
    """Give every remaining REAL hull parameter its value.

    * replicated params (same shape as the shared tree's): alias the shared
      tree's storage (``.data`` reassignment -- zero new bytes, and the
      data_ptr gate can verify the aliasing),
    * per-head GDN vectors (``dt_bias``, ``A_log``): composed by value via
      in-place copy (RadixLinearAttention captured these Parameter objects at
      construction),
    * anything else: a hard error naming the parameter -- an unknown sharded
      parameter must fail loudly, not stay at init garbage.
    """
    shared_params = dict(shared_model.named_parameters())
    part_params = [dict(m.named_parameters()) for m in part_models]
    counts = {"aliased": 0, "composed_vec": 0}
    composed_prefixes = getattr(hull, "_lane_composed_prefixes", ())
    for name, param in hull.named_parameters():
        if any(name.startswith(pfx) for pfx in composed_prefixes):
            continue  # filled by _compose_column_weights_inplace
        if param.device.type == "meta":
            raise ValueError(
                f"dual-group lane: hull parameter {name!r} is still on meta "
                "after shell assembly -- the hull must be built on the target "
                "device (lazy quantized params carry no storage anyway)."
            )
        sp = shared_params.get(name)
        if sp is not None and sp.shape == param.shape:
            param.data = sp.data
            counts["aliased"] += 1
            continue
        base = name.rsplit(".", 1)[-1]
        if base in ("dt_bias", "A_log"):
            pieces = [pp[name].data for pp in part_params]
            full = torch.cat(pieces, dim=0)
            if full.shape != param.shape:
                raise ValueError(
                    f"dual-group lane: composed {name} shape "
                    f"{tuple(full.shape)} != hull {tuple(param.shape)}."
                )
            param.data.copy_(full)
            counts["composed_vec"] += 1
            continue
        raise ValueError(
            f"dual-group lane: hull parameter {name!r} (shape "
            f"{tuple(param.shape)}) is neither replicated (shared-tree shape "
            f"{None if sp is None else tuple(sp.shape)}) nor a known composed "
            "vector. Refusing to run with an unfilled parameter."
        )
    return counts


def verify_shared_bytes(
    hull: nn.Module, shared_model: nn.Module, shared_fast_rank: int
) -> int:
    """The corrected byte gate: data_ptr IDENTITY of every shared shard.

    Bitwise lane==serving-group output equality is structurally impossible
    (the column split changes GEMM blocking); what IS exact and binary is
    that the lane computes with the serving rank's very tensor objects.
    Checks (a) every shell's shared part parameter against the resident
    model's parameter of the same name, (b) every aliased replicated hull
    parameter.  Returns the number of verified pointers; raises naming the
    first divergence (congruent-lane convention).
    """
    shared = {n: p.data_ptr() for n, p in shared_model.named_parameters()}
    checked = 0
    # (a) shell parts of the shared lane rank.
    for name, module in hull.named_modules():
        parts = getattr(module, "_lane_parts", None)
        if parts is None:
            continue
        part = parts[shared_fast_rank]
        for pn, pp in part.named_parameters():
            full_name = f"{name}.{pn}" if name else pn
            recorded = shared.get(full_name)
            if recorded is None:
                raise AssertionError(
                    f"dual-group lane: shared shell part parameter "
                    f"{full_name!r} has no counterpart in the resident model."
                )
            if pp.data_ptr() != recorded:
                raise AssertionError(
                    "dual-group lane: shared-byte gate FAILED -- parameter "
                    f"{full_name!r} has storage 0x{pp.data_ptr():x} in the "
                    f"lane but 0x{recorded:x} in the resident serving rank. "
                    "The lane must compute with the resident shard's very "
                    "bytes; a second copy voids the VRAM plan."
                )
            checked += 1
    # (b) aliased replicated hull params.
    for name, param in hull.named_parameters():
        recorded = shared.get(name)
        if recorded is not None and param.data_ptr() == recorded:
            checked += 1
    if checked == 0:
        raise AssertionError(
            "dual-group lane: shared-byte gate checked zero parameters -- "
            "the gate itself is broken, do not trust this boot."
        )
    return checked


# ---------------------------------------------------------------------------
# Building the lane model inside the lane ModelRunner
# ---------------------------------------------------------------------------


def _load_complement_model(
    lane_runner, plan: NestedGroupPlan, fast_rank: int
) -> nn.Module:
    """Load one complement part: the lane group's rank ``fast_rank`` shard.

    Runs the STOCK loader under (a) the lane's own partition vectors --
    without the scope the resident group's vector does not apply to a group
    of another size and the loader silently falls back to the even split --
    and (b) the ParallelContext override for (fast_size, fast_rank).  The v2
    parameter loaders read tp geometry from exactly these two sources.
    """
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.model_loader import get_model_loader

    fams = {name: list(vec) for name, vec in plan.fast_family_ratios}
    t0 = time.perf_counter()
    with scoped_tp_partition_ratios(
        list(plan.fast_ratio), fams or None
    ), lane_geometry_override(plan.fast_size, fast_rank):
        loader = get_model_loader(
            load_config=lane_runner.load_config,
            model_config=lane_runner.model_config,
        )
        model = loader.load_model(
            model_config=lane_runner.model_config,
            device_config=DeviceConfig(lane_runner.device, lane_runner.gpu_id),
        )
    logger.info(
        "dual-group lane: complement rank %d (of ratio %s) loaded in %.1f s",
        fast_rank,
        list(plan.fast_ratio),
        time.perf_counter() - t0,
    )
    return model.eval()


def _build_hull(lane_runner) -> nn.Module:
    """Construct the full-width hull tree on the TARGET device.

    Deliberately NOT on meta: quantized big weights are lazily-initialized
    parameters (no storage) either way, while the small REAL tensors the
    build creates (GDN conv weights, dt_bias/A_log, norms) are exactly the
    storages that RadixLinearAttention captures views of at construction --
    building them real lets the assembly fill them IN-PLACE instead of
    re-plumbing captured references.  Measured cost: a few MiB.
    """
    from sglang.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
        set_default_torch_dtype,
    )

    quant_config = _get_quantization_config(
        lane_runner.model_config, lane_runner.load_config
    )
    with lane_geometry_override(1, 0):
        with set_default_torch_dtype(lane_runner.model_config.dtype):
            with torch.device(lane_runner.device):
                hull = _initialize_model(
                    lane_runner.model_config,
                    lane_runner.load_config,
                    quant_config,
                )
    return hull.eval()


def build_lane_model(lane_runner) -> nn.Module:
    """The lane ModelRunner's load_model body: complement parts + hull +
    shells + the shared-byte gate.  Rank-local by contract."""
    host_runner = lane_runner.dual_group_host_runner
    plan: NestedGroupPlan = lane_runner.dual_group_plan
    if host_runner is None or plan is None:
        raise ValueError(
            "dual-group lane runner needs dual_group_host_runner and "
            "dual_group_plan."
        )

    free_before = torch.cuda.mem_get_info(lane_runner.gpu_id)[0]

    # Part models in lane-rank order: shared segments reuse the resident
    # model; complement segments load their shard.
    part_models: List[nn.Module] = []
    for f, seg in enumerate(plan.segments):
        if len(seg) == 1:
            if plan.shared_big_rank(f) != host_runner.tp_rank:
                raise ValueError(
                    f"dual-group lane: shared segment {f} covers serving rank "
                    f"{plan.shared_big_rank(f)} but this process is serving "
                    f"rank {host_runner.tp_rank} -- the lane must be built in "
                    "the shared rank's process."
                )
            part_models.append(host_runner.model)
        else:
            part_models.append(_load_complement_model(lane_runner, plan, f))

    free_after_complement = torch.cuda.mem_get_info(lane_runner.gpu_id)[0]
    hull = _build_hull(lane_runner)
    counts = assemble_lane_shells(hull, part_models)
    fill = _finalize_hull_params(hull, host_runner.model, part_models)

    shared_ranks = plan.shared_fast_ranks
    checked = 0
    for f in shared_ranks:
        checked += verify_shared_bytes(hull, part_models[f], f)
    logger.info(
        "dual-group lane model assembled: shells column=%d row=%d embed=%d "
        "lm_head=%d composed=%d; params aliased=%d composed_vec=%d; "
        "shared-byte gate PASSED (%d data_ptr identities).",
        counts["column"],
        counts["row"],
        counts["embedding"],
        counts["lm_head"],
        counts["composed"],
        fill["aliased"],
        fill["composed_vec"],
        checked,
    )

    free_after = torch.cuda.mem_get_info(lane_runner.gpu_id)[0]
    added_mib = max(0, (free_before - free_after)) >> 20
    complement_mib = max(0, (free_before - free_after_complement)) >> 20
    lane_runner.dual_group_lane_weight_added_mib = int(added_mib)
    # The §5 posts block, with MEASURED numbers (the shared post is exact by
    # the gate above; the nested/duplicated posts are what mem_get_info saw).
    from sglang.srt.distributed.dual_group import DUPLICATED, NESTED, SHARED, VramPost

    posts = (
        VramPost(
            name=f"shared serving-rank shard (segments {list(plan.shared_fast_ranks)})",
            status=SHARED,
            mib=0,
            why=f"data_ptr-verified, {checked} identities",
        ),
        VramPost(
            name="lane complement shard(s)",
            status=NESTED,
            mib=int(complement_mib),
            why="bytes the other cards hold; this card now holds the full "
            "weights exactly once",
        ),
        VramPost(
            name="hull tree residue (composed conv/state vectors, buffers)",
            status=DUPLICATED,
            mib=int(added_mib - complement_mib),
            why="small real tensors of the full-width hull; big weights are "
            "lazy/shelled",
        ),
    )
    logger.info("%s", format_vram_posts(posts, f"cuda:{lane_runner.gpu_id}"))
    # Keep references so the parts stay alive (shells hold them too, via
    # plain tuples that named_parameters() does not walk).
    lane_runner.dual_group_part_models = part_models
    return hull


# ---------------------------------------------------------------------------
# Lane runner bring-up + the serial tick driver (S1 pattern)
# ---------------------------------------------------------------------------


def resolve_speed_dial(server_args) -> Tuple[int, int]:
    """Resolve --dual-group-lane-speed-dial onto the lane's two capacity
    posts and return them (pure: the serving config is not touched, the
    caller writes them onto the lane's own args view).

    "Speed through sacrifice" is a first-class regulator, not an emergent
    property of tuning two unrelated numbers (DESIGN_201, resource principle
    3): one dial, both posts, logged. It only ever reduces -- the configured
    values are the capacity end of the scale, so an unset dial is exactly the
    old behavior.

    Geometric interpolation, because both posts are capacity-like: halving
    the sessions and halving the pool are the same KIND of step wherever you
    are on the scale, so equal dial increments should buy equal factors.
    """
    dial = getattr(server_args, "dual_group_lane_speed_dial", None)
    budget = int(server_args.dual_group_lane_budget_mib or 0)
    requests = int(server_args.dual_group_lane_max_requests)
    if dial is None:
        return budget, requests
    if not 0.0 <= float(dial) <= 1.0:
        raise ValueError(
            "--dual-group-lane-speed-dial must be in [0.0, 1.0], got "
            f"{dial!r}."
        )
    dial = float(dial)
    # Minimum end of the scale: one session, one eighth of the pool.
    new_budget = max(1, int(round(budget * (0.125**dial))))
    new_requests = max(1, int(round(requests * ((1.0 / max(requests, 1)) ** dial))))
    if (new_budget, new_requests) != (budget, requests):
        logger.info(
            "dual-group lane speed dial %.2f: budget %d -> %d MiB, "
            "max_requests %d -> %d (capacity given up for speed).",
            dial,
            budget,
            new_budget,
            requests,
            new_requests,
        )
    return new_budget, new_requests


def _lane_server_args_view(server_args):
    """A shallow, lane-scoped view of the server args.

    The lane runner reuses the whole ModelRunner machinery, which reads its
    knobs from ``self.server_args``; the lane's knobs differ (no speculation,
    no DCP, its own concurrency and mamba-slot count, no uneven-TP plan --
    the lane's geometry comes from the scoped partition vectors instead).
    A shallow copy keeps every unrelated field identical; mutating shared
    sub-objects through the view is forbidden (the graph-plan harmonizer,
    the one mutator of ``cuda_graph_config``, is bypassed by the rank-local
    graph flag).
    """
    import copy

    view = copy.copy(server_args)
    lane_budget_mib, lane_requests = resolve_speed_dial(server_args)
    view.dual_group_lane_budget_mib = lane_budget_mib
    view.dual_group_lane_max_requests = lane_requests
    view.speculative_algorithm = None
    view.speculative_draft_model_path = None
    if hasattr(view, "speculative_cross_algorithm"):
        view.speculative_cross_algorithm = False
    view.dcp_size = 1
    view.max_running_requests = lane_requests
    # The lane has no radix cache (its batches use a no-op tree-cache stub),
    # so the mamba slot ratio is 1 slot per request (with the radix cache on,
    # _calculate_mamba_ratio charges ~5 slots/request for the extra-buffer
    # strategy and a 2-slot lane pool admits zero requests -- measured boot
    # failure). One spare slot for the ping-pong margin.
    view.disable_radix_cache = True
    view.max_mamba_cache_size = lane_requests + 1
    view.rank_tp_ratio = None
    view.rank_mlp_ratio = None
    view.rank_moe_ratio = None
    view.rank_vocab_ratio = None
    view.rank_gpu_memory_mib = None
    view.disaggregation_topology = None
    # Own graph-plan object: the lane's capture setup must never write into
    # the serving group's (shared, already-captured) plan.
    view.cuda_graph_config = copy.deepcopy(server_args.cuda_graph_config)
    # Thin the lane's prefill tier ladder: the full-width lane pays a larger
    # per-tier capture footprint than a serving shard, and the lane runs
    # whole chunks, not fine-grained batch mixes. Tiers stay within 2x of
    # each other, so tier padding costs at most 2x on the smallest prompts.
    prefill_cfg = view.cuda_graph_config.prefill
    keep = [t for t in prefill_cfg.bs if t in (16, 32, 64, 128, 256, 512, 1024, 1536, 2048)]
    if keep:
        prefill_cfg.bs = keep
        prefill_cfg.max_bs = max(keep)
    if server_args.dual_group_lane_eager:
        # Eager bring-up: the runner machinery still builds its EagerRunner
        # (forward dispatch needs it); only the captures are skipped. The
        # disable flags were already RESOLVED into the phase config at
        # __post_init__, so the phases themselves are set to DISABLED here.
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

        view.disable_cuda_graph = True
        view.disable_prefill_cuda_graph = True
        view.disable_decode_cuda_graph = True
        for phase in Phase.ALL:
            getattr(view.cuda_graph_config, phase).backend = Backend.DISABLED
    return view


class DualGroupLane:
    """One lane of the multi-group runtime: its plan, its runner, and a
    serial job driver (a lane TICK runs instead of a serving-group iteration,
    rank-locally, with no collectives -- the S1 execution model of
    DESIGN_121 §6).

    Job = greedy generation: one whole-prompt prefill (the protected PD
    quantity, timed), then one decode step per tick.  Timings are wall-clock
    around synchronized forwards -- the lane is the only user of its stream
    during a tick (serial contract).
    """

    def __init__(self, lane_id: int, plan: NestedGroupPlan, runner, concurrent: bool = False):
        self.lane_id = lane_id
        self.plan = plan
        self.runner = runner
        self.jobs: List[Dict[str, Any]] = []
        self.active: Optional[Dict[str, Any]] = None
        self.results: List[Dict[str, Any]] = []
        self._runtime_scope = None
        # -- concurrency (slice C) ---------------------------------------
        self.concurrent = bool(concurrent)
        self.stream = None
        self._thread = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._submitted = threading.Event()
        self._stop = threading.Event()
        self._idle_since: Optional[float] = time.monotonic()
        self._last_wall_ms: Optional[float] = None
        self._busy = False
        self._admission_waits: List[float] = []
        self.lending = None  # set by the scheduler when stage-2 lending is on
        self.results_total = 0
        self.prefill_tokens_total = 0
        self.decode_steps_total = 0

    # -- job interface (rank-local; called from the scheduler loop) -------

    def enqueue(self, job: Dict[str, Any]) -> None:
        input_ids = job.get("input_ids")
        if not input_ids:
            raise ValueError("lane job needs non-empty input_ids")
        with self._lock:
            self.jobs.append(
                {
                    "input_ids": [int(t) for t in input_ids],
                    "max_new_tokens": int(job.get("max_new_tokens", 32)),
                    "output_ids": [],
                    "prefill_ms": None,
                    "decode_ms": [],
                }
            )
        # PD priority (addendum 5): work has arrived for the protected class.
        # Reclaim anything lent to the scavenger BEFORE waking the worker --
        # the reclaim latency is the guarantee quality of that priority and
        # is measured, not assumed.
        if self.lending is not None:
            self.lending.on_lane_work_arrived()
        self._idle_since = None
        self._wake.set()

    @property
    def has_work(self) -> bool:
        with self._lock:
            return self.active is not None or bool(self.jobs) or self._busy

    def stats(self) -> Dict[str, Any]:
        stats = {
            "lane_id": self.lane_id,
            "plan": self.plan.describe(),
            "queued": len(self.jobs),
            "active": self.active is not None,
            # The result LIST is capped (it carries per-job token ids); the
            # counters are monotone, so a measurement window is counted by
            # differencing them and merely SAMPLED by the list.
            "results_total": self.results_total,
            "prefill_tokens_total": self.prefill_tokens_total,
            "decode_steps_total": self.decode_steps_total,
            "results": self.results[-8:],
            "weight_added_mib": getattr(
                self.runner, "dual_group_lane_weight_added_mib", None
            ),
            "max_total_num_tokens": getattr(
                self.runner, "max_total_num_tokens", None
            ),
            "concurrent": self.concurrent,
        }
        if self._admission_waits:
            waits = self._admission_waits[-64:]
            stats["admission_wait_ms"] = {
                "n": len(waits),
                "mean": round(sum(waits) / len(waits), 4),
                "max": round(max(waits), 4),
            }
        if self.lending is not None:
            stats["lending"] = self.lending.stats()
        return stats

    # -- concurrent driver (slice C) -------------------------------------

    def start_worker(self) -> None:
        """Start the lane's own thread and its high-priority CUDA stream.

        The thread is where de-globalization pays off: it enters the lane
        scope ONCE and stays in it, so every ``get_server_args()`` /
        geometry / per-lane-resource read on this thread resolves to the
        lane for the thread's whole life, while the scheduler thread keeps
        reading the serving group's values. Nothing is swapped, so the two
        may run at the same time.

        The stream carries the PD priority in the only place the hardware
        honours it: a high-priority stream's blocks are scheduled ahead of a
        normal-priority stream's AS BLOCKS RETIRE -- preemption at the
        natural grain, never mid-kernel (addendum 5).
        """
        if self._thread is not None:
            return
        import os

        low, high = torch.cuda.Stream.priority_range()
        # Escape hatch, not a tuning knob: NCCL's collective kernels
        # spin-wait, so if the protected lane ever starved the serving
        # group's freshly launched all-reduce the symptom would be a
        # rank-0-late group stall rather than a slowdown. Setting this to 0
        # makes both classes equal-priority and isolates that question.
        high = int(os.environ.get("SGLANG_DUAL_GROUP_LANE_STREAM_PRIORITY", high))
        self.stream = torch.cuda.Stream(
            device=self.runner.gpu_id, priority=high
        )
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"dual-group-lane-{self.lane_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "dual-group lane %d worker started: own thread + CUDA stream "
            "(priority %d of range [%d, %d]); the serving group keeps the "
            "default stream.",
            self.lane_id,
            high,
            low,
            high,
        )

    def stop_worker(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=10.0)
        self._thread = None

    def _worker_loop(self) -> None:
        with self._lane_runtime_scope():
            torch.cuda.set_device(self.runner.gpu_id)
            while not self._stop.is_set():
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                if self._stop.is_set():
                    break
                while not self._stop.is_set():
                    with self._lock:
                        if self.active is None and not self.jobs:
                            self._busy = False
                            if self._idle_since is None:
                                self._idle_since = time.monotonic()
                            break
                        self._busy = True
                    try:
                        with torch.cuda.stream(self.stream):
                            self._step_locked_scope()
                    except Exception:
                        logger.exception(
                            "dual-group lane %d step failed; dropping the "
                            "active job.",
                            self.lane_id,
                        )
                        with self._lock:
                            self.active = None

    def _step_locked_scope(self) -> None:
        """One lane step on the worker thread. Already inside the lane scope
        (entered once for the thread) -- do NOT re-enter it per step."""
        with self._lock:
            if self.active is None:
                self.active = self.jobs.pop(0)
            job = self.active
        self._submitted.clear()
        with torch.no_grad():
            if job["prefill_ms"] is None:
                self._prefill(job)
            else:
                self._decode_step(job)
        if (
            len(job["output_ids"]) >= job["max_new_tokens"]
            or (job["output_ids"] and job["output_ids"][-1] < 0)
        ):
            self._finish(job)

    def note_admission(self, waited_ms: float) -> None:
        """Record how long the scheduler yielded to let the lane submit
        first at a grain boundary (the two-class scheduler's cost to the
        scavenger, measured rather than assumed)."""
        self._admission_waits.append(waited_ms)
        if len(self._admission_waits) > 512:
            del self._admission_waits[:256]

    @property
    def idle_seconds(self) -> float:
        since = self._idle_since
        return 0.0 if since is None else (time.monotonic() - since)

    # -- the tick ---------------------------------------------------------

    def tick(self) -> bool:
        """Run ONE lane step (a whole-prompt prefill or one decode step).
        Returns True when it did work.  Rank-local; never touches a
        communicator; runs under the lane geometry override so any residual
        live geometry read sees the lane, not the serving group."""
        with self._lock:
            if self.active is None:
                if not self.jobs:
                    return False
                self.active = self.jobs.pop(0)
            job = self.active
        with self._lane_runtime_scope(), torch.no_grad():
            if job["prefill_ms"] is None:
                self._prefill(job)
            else:
                self._decode_step(job)
        if (
            len(job["output_ids"]) >= job["max_new_tokens"]
            or (job["output_ids"] and job["output_ids"][-1] < 0)
        ):
            self._finish(job)
        return True

    @contextlib.contextmanager
    def _lane_runtime_scope(self):
        """The lane's execution scope: its identity, its config and its
        geometry, all CONTEXT-LOCAL (#274 slice C).

        The batch/forward machinery reads ``get_server_args()`` at runtime --
        e.g. ``prepare_for_extend`` gates the mamba-radix track machinery on
        it, which the lane (no radix cache, its own pool without track
        buffers) must not enter.  Slice B published the lane's args into the
        PROCESS slot for the duration of a tick and restored them after; that
        swap is exactly what forbade concurrency, because a serving forward
        running at the same time would read the lane's config.

        ``lane_scope`` installs an OVERLAY in a context variable instead. In
        the serial mode this is observably the same thing (same thread, same
        nesting); in the concurrent mode the lane worker thread carries its
        own overlay and the scheduler thread keeps reading the serving
        config. The lane id in the same scope keys the per-lane process
        resources (graph memory pool, GGUF dequant workspace) that would
        otherwise alias between lanes.
        """
        from sglang.srt.runtime_context import lane_scope

        with lane_scope(self.scope_lane_id, self.runner.server_args):
            with lane_geometry_override(1, 0):
                yield

    @property
    def scope_lane_id(self) -> Optional[int]:
        """The identity this lane presents to the per-lane RESOURCE keys
        (graph memory pool, GGUF dequant workspace).

        In SERIAL mode it is ``None``: the lane and the serving group never
        run at the same time, so sharing those resources is sound, and
        keeping them shared is what makes the serial mode byte-for-byte the
        slice-B mode (same capture pool, same workspace, same VRAM posts) --
        the hard gate slice C is not allowed to move.

        In CONCURRENT mode it is the lane id, and the extra capture pool plus
        the extra dequant workspace are the named VRAM price of concurrency.
        """
        return self.lane_id if self.concurrent else None

    def _make_batch(self, job):
        from array import array

        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runner = self.runner
        sampling_params = SamplingParams(
            temperature=0, max_new_tokens=job["max_new_tokens"]
        )
        sampling_params.normalize(None)
        req = Req(
            rid=f"dual-group-lane-{self.lane_id}",
            origin_input_text="",
            origin_input_ids=array("q", job["input_ids"]),
            sampling_params=sampling_params,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))

        tree_cache = _LaneTreeCacheStub(
            page_size=runner.server_args.page_size,
            device=runner.device,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        return batch

    def _timed_forward(self, batch):
        """One lane forward, timed on the LANE's stream only.

        Slice B synchronized the whole DEVICE around the forward, which is
        correct for a serial tick and fatal for a concurrent one: it would
        wait for the serving group's kernels too, so every lane timing would
        include the serving group's work and the lane would serialize behind
        it. The concurrent path times with CUDA events on the lane stream and
        synchronizes that stream alone. Wall-clock is still reported for the
        serial path, where the two agree.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        runner = self.runner
        if self.stream is None:
            torch.cuda.synchronize(runner.gpu_id)
            t0 = time.perf_counter()
            forward_batch = ForwardBatch.init_new(batch, runner)
            logits_output = runner.forward(forward_batch).logits_output
            next_token_ids = runner.sample(logits_output, forward_batch)
            torch.cuda.synchronize(runner.gpu_id)
            return next_token_ids, (time.perf_counter() - t0) * 1000.0

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start_ev.record(self.stream)
        forward_batch = ForwardBatch.init_new(batch, runner)
        logits_output = runner.forward(forward_batch).logits_output
        next_token_ids = runner.sample(logits_output, forward_batch)
        end_ev.record(self.stream)
        self._submitted.set()
        self.stream.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self._last_wall_ms = wall_ms
        # Device time on the lane's stream: what the lane's own kernels took,
        # including the SM share it lost to the serving group but excluding
        # any time spent waiting for the GIL between launches.
        return next_token_ids, start_ev.elapsed_time(end_ev)

    def _prefill(self, job):
        batch = self._make_batch(job)
        batch.prepare_for_extend()
        if (
            batch.input_ids is None
            and getattr(batch, "prefill_input_ids_cpu", None) is not None
        ):
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        next_token_ids, ms = self._timed_forward(batch)
        job["prefill_ms"] = ms
        # WALL minus DEVICE is the discriminator for where a degradation
        # comes from (#274 slice C): device time is what the lane's own
        # kernels took on the lane stream, wall time additionally contains
        # everything the lane waited for before its kernels could run
        # (interpreter/GIL, launch queue, admission). A degradation that
        # shows up in DEVICE time is SM competition; one that shows up only
        # in the wall/device GAP is submission granularity.
        job["prefill_wall_ms"] = self._last_wall_ms
        job["output_ids"].append(int(next_token_ids[0].item()))
        job["_batch"] = batch
        job["_next"] = next_token_ids
        # Captured NOW for the cleanup: the batch's req list is not stable
        # across the decode preparations.
        job["_req"] = batch.reqs[0]
        job["_req_pool_idx"] = int(batch.reqs[0].req_pool_idx)

    def _decode_step(self, job):
        batch = job["_batch"]
        batch.input_ids = job["_next"].to(torch.int64)
        batch.prepare_for_decode()
        next_token_ids, ms = self._timed_forward(batch)
        job["decode_ms"].append(ms)
        job["output_ids"].append(int(next_token_ids[0].item()))
        job["_next"] = next_token_ids

    def _finish(self, job):
        batch = job.pop("_batch", None)
        job.pop("_next", None)
        if batch is not None:
            # Release the lane pool slots (no radix cache: free directly).
            # Allocated tokens = prompt + one per decode step; the LAST
            # sampled token was never written into the KV pool.
            n_alloc = len(job["input_ids"]) + max(0, len(job["output_ids"]) - 1)
            idx = job.get("_req_pool_idx")
            req = job.pop("_req", None)
            if idx is not None and req is not None:
                kv_indices = batch.req_to_token_pool.req_to_token[idx, :n_alloc]
                batch.token_to_kv_pool_allocator.free(kv_indices)
                pool = batch.req_to_token_pool
                # Hybrid pool: the GDN state slot is freed separately and
                # takes the Req (it reads mamba_pool_idx / req_pool_idx).
                if (
                    hasattr(pool, "free_mamba_cache")
                    and getattr(req, "mamba_pool_idx", None) is not None
                ):
                    pool.free_mamba_cache(req)
                pool.free(req)
        decode_ms = job["decode_ms"]
        result = {
            "input_len": len(job["input_ids"]),
            "output_ids": job["output_ids"],
            "prefill_ms": round(job["prefill_ms"], 2),
            "decode_ms_mean": (
                round(sum(decode_ms) / len(decode_ms), 3) if decode_ms else None
            ),
            "decode_steps": len(decode_ms),
        }
        if job.get("prefill_wall_ms") is not None:
            result["prefill_wall_ms"] = round(job["prefill_wall_ms"], 2)
            result["prefill_wait_ms"] = round(
                job["prefill_wall_ms"] - job["prefill_ms"], 2
            )
        with self._lock:
            self.results.append(result)
            self.results_total += 1
            self.prefill_tokens_total += result["input_len"]
            self.decode_steps_total += result["decode_steps"]
            self.active = None
            if not self.jobs:
                self._busy = False
                self._idle_since = time.monotonic()
        logger.info(
            "dual-group lane %d job done: prefill %d tokens in %.1f ms "
            "(%.1f ms/1k), %d decode steps, mean %.2f ms/step.",
            self.lane_id,
            result["input_len"],
            result["prefill_ms"],
            result["prefill_ms"] * 1000.0 / max(1, result["input_len"]),
            result["decode_steps"],
            result["decode_ms_mean"] or 0.0,
        )


class LaneLending:
    """Elastic lane occupancy, stage 2: VRAM return with a reclaim guarantee
    (#274 / DESIGN_201 addendum 4+5).

    Stage 1 (compute share) is the scheduler's business and is free: the lane
    simply stops submitting. Stage 2 is the one with a price, and the price
    is what this class measures.

    THE CONTRACT.  The lane's pool budget is RESERVED: it is never lent
    without a reclaim guarantee. What may be lent is the part of it the lane
    is demonstrably not using, and the borrower may only put EVACUABLE
    content there -- discardable scratch, or spillable session KV. Permanent
    posts are refused. Reclaim is therefore always possible without waiting
    for the borrower's work to finish; it costs a free (and, for spillable
    content, an evacuation), and that cost is the guarantee quality of the PD
    priority.

    THE HYSTERESIS.  Lending is only worth it if the borrower holds the bytes
    long enough to earn back the reclaim. The lane must be idle for
    ``threshold_s`` before anything is lent (#156 pattern: the threshold
    keeps the occupancy from flapping when the lane is merely between jobs),
    and once lent, a minimum hold prevents an immediate reversal.

    WHAT IS BUILT HERE.  The mechanism, its instrumentation and the
    discardable-scratch borrower (the content class that needs no evacuation
    path). The spillable-session-KV borrower -- the serving group placing
    cold sessions in the lane's idle region via the #236/#242 machinery,
    which turns the lent bytes into serving CAPACITY rather than serving
    scratch -- reuses this same lend/reclaim interface and is the named
    follow-on.
    """

    def __init__(
        self,
        lane,
        lend_mib: int,
        threshold_s: float,
        min_hold_s: float = 1.0,
    ):
        self.lane = lane
        self.lend_mib = int(lend_mib)
        self.threshold_s = float(threshold_s)
        self.min_hold_s = float(min_hold_s)
        self._borrowed = None
        self._lent_at: Optional[float] = None
        self.lend_events = 0
        self.reclaim_events = 0
        self.refused_min_hold = 0
        self.lend_ms: List[float] = []
        self.reclaim_ms: List[float] = []

    @property
    def is_lent(self) -> bool:
        return self._borrowed is not None

    def maybe_lend(self) -> bool:
        """Called by the serving group at a grain boundary. Lends only when
        the lane has been idle past the amortization threshold."""
        if self._borrowed is not None or self.lend_mib <= 0:
            return False
        if self.lane.has_work or self.lane.idle_seconds < self.threshold_s:
            return False
        t0 = time.perf_counter()
        try:
            self._borrowed = torch.empty(
                self.lend_mib * (1 << 20),
                dtype=torch.uint8,
                device=f"cuda:{self.lane.runner.gpu_id}",
            )
        except torch.cuda.OutOfMemoryError:
            # The lane's idle bytes are not necessarily free bytes: the
            # caching allocator may have handed them out already. Refusing is
            # correct -- the reservation is the lane's, not the borrower's.
            self.lend_mib = 0
            logger.warning(
                "dual-group lane %d: stage-2 lend refused (out of memory); "
                "lending disabled for this boot.",
                self.lane.lane_id,
            )
            return False
        self._lent_at = time.monotonic()
        self.lend_events += 1
        self.lend_ms.append((time.perf_counter() - t0) * 1000.0)
        return True

    def on_lane_work_arrived(self) -> None:
        """PD work arrived: give the bytes back NOW. The elapsed time is the
        reclaim latency -- the number the priority promise is worth."""
        if self._borrowed is None:
            return
        if (
            self._lent_at is not None
            and (time.monotonic() - self._lent_at) < self.min_hold_s
        ):
            # Hysteresis, one-sided: a reclaim is never REFUSED (that would
            # break the guarantee), it is only counted, so a flapping
            # workload is visible in the stats instead of silently paying.
            self.refused_min_hold += 1
        t0 = time.perf_counter()
        self._borrowed = None
        # Discardable scratch: the free IS the evacuation. The bytes return
        # to the one caching allocator both lanes draw from (slice A's
        # single-address-space property), so nothing is copied.
        torch.cuda.empty_cache()
        self._lent_at = None
        self.reclaim_events += 1
        self.reclaim_ms.append((time.perf_counter() - t0) * 1000.0)

    def stats(self) -> Dict[str, Any]:
        def _summary(xs):
            if not xs:
                return None
            tail = xs[-64:]
            return {
                "n": len(tail),
                "mean_ms": round(sum(tail) / len(tail), 3),
                "max_ms": round(max(tail), 3),
            }

        return {
            "lend_mib": self.lend_mib,
            "threshold_s": self.threshold_s,
            "is_lent": self.is_lent,
            "lend_events": self.lend_events,
            "reclaim_events": self.reclaim_events,
            "refused_min_hold": self.refused_min_hold,
            "lend": _summary(self.lend_ms),
            "reclaim": _summary(self.reclaim_ms),
        }


class _LaneTreeCacheStub:
    """Minimal tree-cache stand-in for lane batches (no prefix caching).
    Mirrors benchmark/one_batch.py's TreeCacheNamespace."""

    def __init__(self, page_size, device, token_to_kv_pool_allocator):
        self.page_size = page_size
        self.device = device
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, params) -> None:
        return None


def build_dual_group_lanes(scheduler) -> List[DualGroupLane]:
    """Build the configured lanes for this scheduler process (slice B: one).

    Called AFTER the serving group finished its own init (model, pools,
    graphs) so the lane's rank-local bring-up cannot interleave with any
    group-collective phase.  Only the shared rank builds a lane; every other
    rank returns an empty list.
    """

    server_args = scheduler.server_args
    if not getattr(server_args, "dual_group_lane", False):
        return []

    host_runner = scheduler.tp_worker.model_runner
    plan_shared_big_rank = 0  # slice B: the lane shares serving rank 0
    if host_runner.tp_rank != plan_shared_big_rank:
        return []

    if server_args.pp_size > 1 or server_args.enable_dp_attention:
        raise ValueError(
            "--dual-group-lane supports single-node tensor parallelism only "
            "(no PP, no DP attention)."
        )
    if not server_args.dual_group_lane_budget_mib:
        raise ValueError(
            "--dual-group-lane requires --dual-group-lane-budget-mib (the "
            "lane's rank-local pool budget in MiB)."
        )

    plan = derive_lane_plan(server_args, host_runner.model_config)

    lane_args = _lane_server_args_view(server_args)
    lane_id = 0
    concurrent = bool(getattr(server_args, "dual_group_lane_concurrent", False))
    logger.info(
        "dual-group lane %d bring-up (rank-local): %s", lane_id, plan.describe()
    )
    # The whole bring-up runs under the lane scope: helpers like
    # check_cuda_graph_backend read the ACTIVE args, not runner.server_args,
    # and must see the lane's graph plan (measured: the lane otherwise
    # captures/skips by the serving group's plan). Slice B swapped the
    # process slot here; slice C installs the context overlay instead, so
    # the bring-up is safe even though the other ranks are already
    # event-loop-ready.
    #
    # The scope carries the same lane IDENTITY the lane will forward under,
    # so the graph capture lands in the pool the replay will use --
    # concurrent lanes capture into their own pool, serial lanes keep
    # sharing the serving group's exactly as in slice B.
    from sglang.srt.runtime_context import lane_scope

    scope_lane_id = lane_id if concurrent else None
    with lane_scope(scope_lane_id, lane_args):
        return _build_lane_under_scope(
            scheduler, server_args, lane_args, plan, lane_id, host_runner, concurrent
        )


def _build_lane_under_scope(
    scheduler, server_args, lane_args, plan, lane_id, host_runner, concurrent=False
) -> List[DualGroupLane]:
    from sglang.srt.model_executor.model_runner import ModelRunner

    with lane_geometry_override(1, 0):
        runner = ModelRunner(
            model_config=host_runner.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=host_runner.gpu_id,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=host_runner.dist_port,
            server_args=lane_args,
            is_draft_worker=True,
            is_dual_group_lane=True,
            dual_group_host_runner=host_runner,
            dual_group_plan=plan,
            dual_group_lane_id=lane_id,
        )
        # Rank-local graphs: drops the plan-harmonization all_gather_object,
        # the capture-group barrier and the per-warmup backend barriers --
        # the same flag the draft-solo capture uses (#194 hang family).
        runner.spec_solo_rank_local_graphs = True
        runner.alloc_memory_pool()
        runner.init_attention_backends()
        if server_args.dual_group_lane_eager:
            logger.warning(
                "dual-group lane %d runs EAGER (--dual-group-lane-eager); "
                "eager is a bring-up state, not an end state.",
                lane_id,
            )
        # Always: builds the eager phase runner (forward dispatch requires
        # it); captures are skipped when the lane view disables them.
        runner.init_cuda_graphs()

    lane = DualGroupLane(
        lane_id=lane_id, plan=plan, runner=runner, concurrent=concurrent
    )
    lend_mib = int(getattr(server_args, "dual_group_lane_lend_mib", 0) or 0)
    if lend_mib > 0:
        lane.lending = LaneLending(
            lane,
            lend_mib=lend_mib,
            threshold_s=float(
                getattr(server_args, "dual_group_lane_lend_threshold_s", 5.0)
            ),
        )
    if concurrent:
        lane.start_worker()
    logger.info(
        "dual-group lane %d ready: max_total_num_tokens=%d, "
        "max_running_requests=%d, graphs=%s, mode=%s, lend=%d MiB.",
        lane_id,
        runner.max_total_num_tokens,
        runner.max_running_requests,
        "eager" if server_args.dual_group_lane_eager else "captured",
        "concurrent" if concurrent else "serial",
        lend_mib,
    )
    return [lane]
