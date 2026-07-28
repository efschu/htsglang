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
import os
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
        outs = [p.quant_method.apply(p, piece, None) for p, piece in zip(parts, pieces)]
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

    @property
    def embedding_dim(self):
        """Hidden width of the table. The parts are VOCAB-sharded, so every
        part carries the full width -- this is not a per-part quantity that
        has to be summed. It is what tells a language embedding apart from a
        companion tower's when a target holds both (contract 5)."""
        return getattr(self._lane_parts[0], "embedding_dim", None)

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
        # Same discriminator as the embedding shell: the INPUT width of the
        # head, which is the model's hidden size (the parts shard the vocab,
        # i.e. the output).
        self.__dict__["embedding_dim"] = getattr(p0, "embedding_dim", None)
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
        # Alias FIRST, then judge. A meta parameter that gets the shared
        # tree's storage is filled, and the NEXTN head's hull is built on
        # meta on purpose (it has no GDN conv views to preserve, and its
        # vocab tables would otherwise cost 2.37 GiB that are replaced by
        # the lane target's shells moments later).
        sp = shared_params.get(name)
        if sp is not None and sp.shape == param.shape:
            if param.device.type == "meta":
                # A meta placeholder cannot take .data from a real tensor
                # (set_data rejects the type change). Replace the Parameter
                # OBJECT on its parent instead -- which shares the very same
                # object, so the data_ptr gate still sees identity and the
                # byte count is still zero.
                parent_name, _, attr = name.rpartition(".")
                parent = hull.get_submodule(parent_name) if parent_name else hull
                setattr(parent, attr, sp)
            else:
                param.data = sp.data
            counts["aliased"] += 1
            continue
        if param.device.type == "meta":
            raise ValueError(
                f"dual-group lane: hull parameter {name!r} is still on meta "
                "and has no counterpart in the shared tree -- it would run "
                "with no storage at all."
            )
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
    with (
        scoped_tp_partition_ratios(list(plan.fast_ratio), fams or None),
        lane_geometry_override(plan.fast_size, fast_rank),
    ):
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


def _build_hull(lane_runner, device=None) -> nn.Module:
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
            with torch.device(device or lane_runner.device):
                hull = _initialize_model(
                    lane_runner.model_config,
                    lane_runner.load_config,
                    quant_config,
                )
    return hull.eval()


def build_lane_draft_model(lane_runner) -> nn.Module:
    """The NEXTN head, assembled for the lane (#274 slice C).

    The head is not a separate model: it is one extra decoder layer plus a
    concat projection, sharded across the serving group in EXACTLY the
    geometry of the main model, and it SHARES embed_tokens / lm_head with the
    target. So it needs no new mechanism -- the complement-load + hull +
    shell assembly applies unchanged, with the plan's unit counts already
    proven to nest (same families, same counts).

    Two things are specific to the head:

    * the vocab. The draft's embed/lm_head are the TARGET's modules, and the
      lane target already owns full-vocabulary shells over the packed
      per-rank shards. The draft hull is pointed at those, so the lane holds
      one set of vocab tables, not two.
    * WHY THIS PATH AND NOT THE EXISTING ONE. ``--speculative-draft-placement
      solo`` builds the same SHAPE (an unsharded, collective-free draft), but
      assembles its full tables through ``_solo_init_lm_head``, which is a
      GROUP COLLECTIVE -- forbidden on the lane's rank-local bring-up path,
      and refused outright for GGUF packed vocab. The lane's shells do the
      same job rank-locally, which is why the head can exist here at all.
    """
    hull = build_lane_model(lane_runner, kind="draft")
    lane_target = getattr(lane_runner, "dual_group_lane_target_model", None)
    if lane_target is not None and hasattr(hull, "set_embed_and_head_modules"):
        hidden_size = int(lane_runner.model_config.hidden_size)
        embed, head = _find_lane_vocab_shells(lane_target, hidden_size)
        if embed is None or head is None:
            raise ValueError(
                "dual-group lane head: could not locate the lane target's "
                "embedding/lm_head shells "
                f"(embed={embed is not None}, head={head is not None}). "
                "The head must share them -- a second set of full-vocabulary "
                "tables is 2.37 GiB, and a head whose embedding is missing "
                "fails at its first forward instead of at bring-up."
            )
        hull.set_embed_and_head_modules(embed, head)
        logger.info(
            "dual-group lane draft: embed/lm_head pointed at the lane "
            "target's full-vocabulary shells at width %d (one set of "
            "tables, not two).",
            hidden_size,
        )
    return hull


def _find_lane_vocab_shells(lane_target: nn.Module, hidden_size: int):
    """Locate the lane target's vocabulary shells by TYPE **and WIDTH**.

    The attribute path differs per model family (``model.embed_tokens`` on a
    causal-LM wrapper, ``model.model.embed_tokens`` behind a conditional-
    generation wrapper), and guessing wrong is silent: ``set_embed_and_head_
    modules`` skips a ``None`` argument, so the head keeps its own -- or, if
    it had none, dies at its first forward with a NoneType call. So the search
    is by the shell types the lane itself installed, which is family-neutral.

    CONTRACT 5, and the reason type alone is NOT enough: type is not unique.
    A multimodal target carries more than one vocab-parallel table (this
    vehicle logs ``embed=2``), and ``modules()`` returns them in registration
    order, so "the first one of the right type" is a coin flip between the
    LANGUAGE embedding and a companion tower's. Picking wrong is silent at
    bring-up and surfaces one forward later as a shape error from inside a
    fused kernel -- measured here as a cutlass signature dump out of
    ``pre_fc_norm_embedding``, width 1152 (the companion tower) against the
    head's 5120.

    Width is the discriminator that actually means something: the head
    concatenates its embedding output with hidden states and projects with
    ``fc``, so the ONLY embedding it can use is the one whose ``embedding_dim``
    is the model's hidden size. Selecting on that is family-neutral for the
    same reason type was, and it is self-verifying: a target with no matching
    shell, or with two, is a configuration this code must not guess about.
    """
    embeds = [
        m for m in lane_target.modules() if isinstance(m, LaneVocabEmbeddingShell)
    ]
    heads = [m for m in lane_target.modules() if isinstance(m, LaneLmHeadShell)]

    def _pick(candidates, what):
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        matching = [
            c for c in candidates if getattr(c, "embedding_dim", None) == hidden_size
        ]
        if len(matching) == 1:
            return matching[0]
        widths = [getattr(c, "embedding_dim", None) for c in candidates]
        raise ValueError(
            f"dual-group lane head: {len(candidates)} {what} shells in the "
            f"lane target with widths {widths}, and "
            f"{len(matching)} of them match the model's hidden size "
            f"{hidden_size}. The head's embedding output is concatenated with "
            "its hidden states before `fc`, so exactly one width can be "
            "correct. Refusing to guess -- guessing here is silent at "
            "bring-up and fails one forward later inside a fused kernel."
        )

    return _pick(embeds, "embedding"), _pick(heads, "lm_head")


def build_lane_model(lane_runner, kind: str = "target") -> nn.Module:
    """The lane ModelRunner's load_model body: complement parts + hull +
    shells + the shared-byte gate.  Rank-local by contract."""
    host_runner = lane_runner.dual_group_host_runner
    plan: NestedGroupPlan = lane_runner.dual_group_plan
    if host_runner is None or plan is None:
        raise ValueError(
            "dual-group lane runner needs dual_group_host_runner and dual_group_plan."
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
    # The NEXTN head's hull goes on META: unlike the target it has no GDN
    # mixer whose conv views are captured at construction (its single layer
    # is full attention), and its own vocab tables are 2.37 GiB that the
    # lane target's shells replace immediately. Everything the head's hull
    # still needs real is replicated and is aliased onto the shared tree's
    # storage by _finalize_hull_params.
    hull = _build_hull(lane_runner, device="meta" if kind == "draft" else None)
    counts = assemble_lane_shells(hull, part_models)
    fill = _finalize_hull_params(hull, host_runner.model, part_models)

    shared_ranks = plan.shared_fast_ranks
    checked = 0
    for f in shared_ranks:
        checked += verify_shared_bytes(hull, part_models[f], f)
    if checked == 0:
        # The gate is the whole point of the shared segment: if a segment is
        # declared shared, its bytes MUST be the resident ones. Zero
        # identities means the assembly silently produced a private copy.
        raise ValueError(
            f"dual-group lane ({kind}): shared-byte gate found 0 data_ptr "
            f"identities across shared segments {list(shared_ranks)} -- the "
            "lane would compute on copies, not on the serving rank's bytes."
        )
    logger.info(
        "dual-group lane %s model assembled: shells column=%d row=%d embed=%d "
        "lm_head=%d composed=%d; params aliased=%d composed_vec=%d; "
        "shared-byte gate PASSED (%d data_ptr identities).",
        kind,
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
    logger.info("%s", format_vram_posts(posts, f"cuda:{lane_runner.gpu_id} ({kind})"))
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
            f"--dual-group-lane-speed-dial must be in [0.0, 1.0], got {dial!r}."
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


def split_lane_budget(server_args, target_model_config, draft_model_config):
    """Split the operator's ONE lane budget between the lane's target and its
    NEXTN head (#274 slice C).

    Resource principle 2: no VRAM is duplicated that does not have to be.
    Both lane runners read ``--dual-group-lane-budget-mib``, so without a
    split the head would silently claim a SECOND full budget -- the lane's
    capacity post would double behind the operator's back. It is one budget
    and it is divided, not two budgets that happen to share a name.

    The head is one decoder layer against the target's many, so its KV need
    is that ratio of the target's. A floor keeps the pool above the
    page/slot minimum on models with very many layers.
    """
    budget = int(server_args.dual_group_lane_budget_mib or 0)
    target_layers = max(1, int(getattr(target_model_config, "num_hidden_layers", 1)))
    draft_layers = max(1, int(getattr(draft_model_config, "num_hidden_layers", 1)))
    draft_mib = max(64, -(-budget * draft_layers // target_layers))
    draft_mib = min(draft_mib, max(64, budget // 4))
    return budget - draft_mib, draft_mib


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
    keep = [
        t for t in prefill_cfg.bs if t in (16, 32, 64, 128, 256, 512, 1024, 1536, 2048)
    ]
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

    def __init__(
        self,
        lane_id: int,
        plan: NestedGroupPlan,
        runner,
        concurrent: bool = False,
        draft_runner=None,
        spec_steps: int = 3,
    ):
        self.lane_id = lane_id
        self.plan = plan
        self.runner = runner
        # Speculation on the lane (#274 slice C): the NEXTN head runner, or
        # None. Its presence is what turns a lane step into a verify round.
        self.draft_runner = draft_runner
        self.spec_steps = int(spec_steps)
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
                    # Per-job overrides, both absent by default so the lane
                    # behaves exactly as the server flags say. They exist so
                    # the coherence gate can record its reference side and
                    # its speculative side from ONE boot -- same weights,
                    # same pools, same captures. Comparing two boots would
                    # put boot-to-boot variance inside the gate.
                    "spec": job.get("spec"),
                    "verify": job.get("verify"),
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
            "max_total_num_tokens": getattr(self.runner, "max_total_num_tokens", None),
            "concurrent": self.concurrent,
            "spec": (
                None
                if self.draft_runner is None
                else {"algorithm": "nextn-chain", "steps": self.spec_steps}
            ),
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
        self.stream = torch.cuda.Stream(device=self.runner.gpu_id, priority=high)
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
                            "dual-group lane %d step failed; dropping the active job.",
                            self.lane_id,
                        )
                        self.drop_active()

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
            elif self._job_spec_on(job):
                self._spec_step(job)
            else:
                self._decode_step(job)
        if len(job["output_ids"]) >= job["max_new_tokens"] or (
            job["output_ids"] and job["output_ids"][-1] < 0
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
            elif self._job_spec_on(job):
                # Same three-way dispatch as the concurrent worker's
                # ``_step_locked_scope``. Round 1 wired the speculative round
                # into the worker path only, so a SERIAL lane assembled the
                # NEXTN head and then never asked it for a proposal -- the
                # chain was unreachable in the very mode that is the default.
                self._spec_step(job)
            else:
                self._decode_step(job)
        if len(job["output_ids"]) >= job["max_new_tokens"] or (
            job["output_ids"] and job["output_ids"][-1] < 0
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

    # -- speculation on the lane (chain, topk 1) --------------------------

    @property
    def spec_active(self) -> bool:
        return self.draft_runner is not None

    def _job_spec_on(self, job) -> bool:
        """Speculation for THIS job: the server flag, unless the job says no.

        ``job["spec"] is None`` -- the normal case -- keeps the flag's answer,
        so nothing about the default path depends on this.
        """
        if not self.spec_active:
            return False
        return job.get("spec") is not False

    def _draft_forward(self, batch_d, hidden_states):
        """One NEXTN head forward on the lane's draft runner.

        Rank-local like everything else on this path: the head's group
        collectives are the same local ops the target's shells use, and its
        hidden input is handed in rather than gathered.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        spec = EagleDraftInput(
            hidden_states=hidden_states,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        batch_d.spec_info = spec
        batch_d.capture_hidden_mode = CaptureHiddenMode.LAST
        fb = ForwardBatch.init_new(batch_d, self.draft_runner)
        fb.spec_info = spec
        fb.capture_hidden_mode = CaptureHiddenMode.LAST
        out = self.draft_runner.forward(fb).logits_output
        return out

    def _propose(self, job):
        """K chain proposals from the head, greedy, topk 1.

        Deliberately a CHAIN and not a tree: topk > 1 under this rig's
        conditions is a measured loss, and a tree would additionally need the
        verify mask machinery the lane does not have.
        """
        proposals = []
        hidden = job["_hidden"]
        token = job["_next"]
        batch_d = job["_batch_d"]
        for _ in range(self.spec_steps):
            batch_d.input_ids = token.to(torch.int64)
            batch_d.prepare_for_decode()
            out = self._draft_forward(batch_d, hidden)
            token = out.next_token_logits.argmax(dim=-1)
            hidden = out.hidden_states
            proposals.append(int(token[0].item()))
            job["_kv_len_draft"] = int(job.get("_kv_len_draft") or 0) + 1
        return proposals

    def _make_batch(self, job, runner=None):
        from array import array

        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runner = runner or self.runner
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

    def _timed_forward_raw(self, batch, capture_mode=None):
        """A lane forward that returns the LOGITS OUTPUT instead of sampled
        ids -- the verify path needs the per-candidate argmax and the hidden
        states, not one sampled token."""
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        runner = self.runner
        if capture_mode is not None:
            batch.capture_hidden_mode = capture_mode
        t0 = time.perf_counter()
        if self.stream is None:
            torch.cuda.synchronize(runner.gpu_id)
        fb = ForwardBatch.init_new(batch, runner)
        if capture_mode is not None:
            fb.capture_hidden_mode = capture_mode
        self._last_fb = fb
        out = runner.forward(fb).logits_output
        if self.stream is None:
            torch.cuda.synchronize(runner.gpu_id)
        else:
            self._submitted.set()
            self.stream.synchronize()
        return out, (time.perf_counter() - t0) * 1000.0

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

    # -- one-shot row/position diagnosis (env-gated, off by default) -------

    def _dbg_on(self) -> bool:
        return bool(os.environ.get("SGLANG_LANE_SPEC_DEBUG"))

    def _dbg(self, tag: str, payload: dict) -> None:
        import json

        logger.info("[lane-spec-dbg] %s %s", tag, json.dumps(payload, default=str))

    def _dbg_prefill_rows(self, out, input_ids):
        """Does row i of ``hidden_states`` predict token i+1?

        The prompt extend is the uncontested case: prefix empty, one request,
        every row a real position. Teacher forcing on prose must hit far above
        chance, and the LAST row must reproduce ``next_token_logits`` exactly
        -- the processor derives that row from the very same tensor. Both
        checks together separate "wrong tensor / wrong lm_head path" from
        "right tensor, wrong row".
        """
        try:
            hs = out.hidden_states
            # Only the tail: full-vocabulary logits for every prompt row are a
            # multi-GiB allocation on this vocabulary and OOM the lane budget.
            tail = min(9, hs.shape[0])
            preds = self._candidate_logits(hs[-tail:]).argmax(dim=-1)
            tgt = torch.tensor(list(input_ids[-tail + 1 :]), device=preds.device)
            n = min(len(tgt), preds.shape[0] - 1)
            match = int((preds[:n] == tgt[:n]).sum().item())
            self._dbg(
                "prefill",
                {
                    "hidden_rows": int(hs.shape[0]),
                    "n_input_ids": len(input_ids),
                    "teacher_forcing_match_tail": f"{match}/{n}",
                    "row_last_argmax": int(preds[-1].item()),
                    "next_token_logits_argmax": int(
                        out.next_token_logits.argmax(dim=-1)[0].item()
                    ),
                    "next_token_logits_shape": list(out.next_token_logits.shape),
                    "preds_tail": [int(x) for x in preds[-5:].tolist()],
                    "input_tail": list(input_ids[-5:]),
                },
            )
        except Exception as exc:  # diagnosis must never take the lane down
            self._dbg("prefill_failed", {"error": repr(exc)})

    def _dbg_verify_rows(self, job, cand, proposals, out, preds, n_cached):
        try:
            fb = getattr(self, "_last_fb", None)

            def _l(t, k=8):
                if t is None:
                    return None
                if isinstance(t, torch.Tensor):
                    return [int(x) for x in t.flatten()[:k].tolist()]
                return list(t)[:k]

            req = job["_req"]
            self._dbg(
                "verify",
                {
                    "round": self._dbg_round,
                    "n_cached": n_cached,
                    "cand": cand,
                    "proposals": proposals,
                    "hidden_rows": int(out.hidden_states.shape[0]),
                    "preds": [int(x) for x in preds.tolist()],
                    "next_token_logits_argmax": int(
                        out.next_token_logits.argmax(dim=-1)[0].item()
                    ),
                    "batch_input_ids": _l(job["_batch"].input_ids),
                    "extend_start_loc": _l(getattr(fb, "extend_start_loc", None)),
                    "extend_prefix_lens": _l(getattr(fb, "extend_prefix_lens", None)),
                    "extend_seq_lens": _l(getattr(fb, "extend_seq_lens", None)),
                    "seq_lens": _l(getattr(fb, "seq_lens", None)),
                    "positions": _l(getattr(fb, "positions", None)),
                    "out_cache_loc": _l(getattr(fb, "out_cache_loc", None)),
                    "req_kv_committed_len": getattr(req, "kv_committed_len", None),
                    "req_already_computed": getattr(req, "already_computed", None),
                    "req_mamba_pool_idx": str(getattr(req, "mamba_pool_idx", None)),
                    "req_mamba_needs_clear": getattr(req, "mamba_needs_clear", None),
                    "prefix_len": int(len(req.prefix_indices)),
                },
            )
        except Exception as exc:
            self._dbg("verify_failed", {"error": repr(exc)})

    def _dbg_single_token_probe(self, job, n_cached, clear_state=False):
        """A verify extend of LENGTH ONE at the same position.

        This is exactly what a plain decode step would compute, expressed in
        the same extend machinery the verify uses. If its prediction matches
        the no-spec reference, extend-with-prefix carries the cached context
        correctly and the fault is specific to MULTI-token verify rows; if it
        does not, the fault is in the extend-with-prefix path itself (prefix
        KV or recurrent state), independent of speculation.

        Diagnosis only: it advances the target's recurrent state by one token,
        which is not rewindable, so the round that follows it is polluted by
        construction. Runs once, under the debug env var.
        """
        from array import array

        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        batch = job["_batch"]
        req = job["_req"]
        idx = job["_req_pool_idx"]
        one = [int(job["_next"][0].item())]
        req.full_untruncated_fill_ids = array(
            "q", list(req.origin_input_ids) + job["output_ids"][:-1] + one
        )
        req.prefix_indices = batch.req_to_token_pool.req_to_token[idx, :n_cached]
        req.set_extend_range(n_cached, n_cached + 1)
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        if clear_state:
            # Forces the deferred zeroing of this request's recurrent slot in
            # prepare_for_extend -- the same mechanism a freshly allocated
            # slot uses.
            req.mamba_needs_clear = True
        batch.prepare_for_extend()
        if (
            batch.input_ids is None
            and getattr(batch, "prefill_input_ids_cpu", None) is not None
        ):
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        out, _ = self._timed_forward_raw(batch, CaptureHiddenMode.FULL)
        p = self._candidate_logits(out.hidden_states).argmax(dim=-1)
        self._dbg(
            "probe_len1",
            {
                "n_cached": n_cached,
                "token_in": one,
                "hidden_rows": int(out.hidden_states.shape[0]),
                "pred": int(p[0].item()),
                "next_token_logits_argmax": int(
                    out.next_token_logits.argmax(dim=-1)[0].item()
                ),
                "mamba_cleared": bool(clear_state),
            },
        )
        self._dbg_rollback_one(job, n_cached)

    def _dbg_rollback_one(self, job, n_cached):
        """Give one written token slot back and restore the length counters."""
        batch = job["_batch"]
        req = job["_req"]
        idx = job["_req_pool_idx"]
        surplus = batch.req_to_token_pool.req_to_token[idx, n_cached : n_cached + 1]
        batch.token_to_kv_pool_allocator.free(surplus)
        req.kv_committed_len = n_cached
        req.kv_allocated_len = n_cached
        req.already_computed = n_cached

    def _dbg_decode_probe(self, job, n_cached):
        """One DECODE step from the same state the verify starts from.

        The no-spec lane decodes, the verify extends. If these two disagree on
        the very same state and the very same input token, the disagreement is
        the bug -- and it is not a speculation bug at all, it is the
        continuation path.
        """
        batch = job["_batch"]
        saved = batch.input_ids
        batch.input_ids = job["_next"].to(torch.int64)
        batch.prepare_for_decode()
        out, _ = self._timed_forward_raw(batch)
        self._dbg(
            "probe_decode",
            {
                "n_cached": n_cached,
                "token_in": [int(job["_next"][0].item())],
                "next_token_logits_argmax": int(
                    out.next_token_logits.argmax(dim=-1)[0].item()
                ),
            },
        )
        batch.input_ids = saved
        self._dbg_rollback_one(job, n_cached)

    def _dbg_state_probes(self, job, n_cached):
        """The three numbers that separate the candidates, in one round.

        1. DECODE from the clean post-prefill state -- the path the no-spec
           lane uses, i.e. the known-good continuation.
        2. Continued EXTEND with the recurrent state deliberately ZEROED.
        3. Continued EXTEND with whatever state is there.

        If 2 and 3 agree, the continued extend never read the stored recurrent
        state, and the reference in step 1 is not needed to say so.

        Every probe advances state that cannot be rewound, so the round that
        follows them is meaningless by construction. Debug only.
        """
        self._dbg_decode_probe(job, n_cached)
        self._dbg_single_token_probe(job, n_cached, clear_state=True)
        self._dbg_single_token_probe(job, n_cached, clear_state=False)

    def _candidate_logits(self, hidden_states):
        """Full-vocabulary logits for EVERY row of ``hidden_states``.

        The lane's own reduction of ``LogitsProcessor._get_logits``: the lane
        is tp_size=1 and its ``LaneLmHeadShell`` already yields full-vocab
        logits by concatenating the per-rank shards locally, so the DP gather
        and the TP all-gather that method performs are both no-ops here --
        and must stay no-ops, because a collective on this path would break
        the lane's rank-local contract. The scale/softcap tail is kept
        because dropping it would silently change the argmax on any model
        that sets them.
        """
        model = self.runner.model
        processor = getattr(model, "logits_processor", None)
        lm_head = getattr(model, "lm_head", None)
        if processor is None or lm_head is None:
            raise ValueError(
                "dual-group lane verify: the lane target exposes no "
                f"logits_processor/lm_head (processor={processor is not None}, "
                f"lm_head={lm_head is not None}); per-candidate logits cannot "
                "be computed and greedy verify would be unsound."
            )
        if getattr(processor, "do_tensor_parallel_all_gather", False):
            raise ValueError(
                "dual-group lane verify: the lane target's logits processor "
                "wants a tensor-parallel all-gather. The lane is rank-local "
                "by contract and its lm_head shell already produces full "
                "vocabulary logits -- this combination is not supported."
            )
        logits = processor._compute_lm_head(hidden_states, lm_head)
        if getattr(processor, "logit_scale", None) is not None:
            logits = logits * processor.logit_scale
        softcap = getattr(processor, "final_logit_softcapping", None)
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    def _verify_by_decode(self, job, proposals, n_cached):
        """Verify by consuming the candidates one DECODE at a time.

        THE DEFAULT, and the only strategy that computes the right tokens on
        this target (see ``_verify`` for why the batched one does not). Same
        accept rule, same emitted tokens; only the forward mode differs.

        It buys NO latency: the number of forwards equals the number of
        emitted tokens, which is exactly what plain decoding costs. So this
        is a CORRECTNESS bridge, not the finished feature -- speculation on
        the lane cannot pay until the verify is a single forward again, and
        the way to get there is ``ForwardMode.TARGET_VERIFY`` rather than a
        hand-rolled extend.

        Two properties come for free and are worth naming: nothing beyond the
        accepted prefix is ever written, so there is no KV surplus to roll
        back, and the target's recurrent state advances by exactly the
        emitted tokens instead of by the whole rejected candidate block --
        which is precisely the defect the batched path has.
        """
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        batch = job["_batch"]
        cand = [int(job["_next"][0].item())] + list(proposals)
        total_ms = 0.0
        n_accept = 0
        preds: List[int] = []
        hidden = None
        for i, tok in enumerate(cand):
            batch.input_ids = torch.tensor(
                [tok], dtype=torch.int64, device=batch.device
            )
            batch.prepare_for_decode()
            out, ms = self._timed_forward_raw(batch, CaptureHiddenMode.LAST)
            total_ms += ms
            preds.append(int(out.next_token_logits.argmax(dim=-1)[0].item()))
            hidden = out.hidden_states[-1:]
            if i >= len(proposals) or preds[i] != proposals[i]:
                break
            n_accept += 1

        emitted = list(proposals[:n_accept]) + [preds[n_accept]]
        job["_kv_len"] = n_cached + n_accept + 1
        job["_hidden"] = hidden
        job["_next"] = torch.tensor(
            [preds[n_accept]], dtype=torch.int64, device=batch.device
        )
        return emitted, n_accept, total_ms

    def _verify_mode(self, job) -> str:
        """Which verify strategy this round uses.

        ``seqdecode`` unless something explicitly asks for ``extend``. The
        default is the slow-but-correct one on purpose -- see ``_verify``.
        """
        return (
            job.get("verify")
            or os.environ.get("SGLANG_LANE_SPEC_VERIFY")
            or "seqdecode"
        )

    def _verify(self, job, proposals):
        """One verify forward of the LANE TARGET over [last, *proposals].

        Greedy accept: the target's output at candidate i predicts what
        should follow it, so proposal i+1 is accepted exactly when it equals
        that argmax. The first mismatch stops the chain and the target's own
        argmax there becomes the next token -- so a round always yields at
        least one token, which is what makes speculation free of correctness
        risk under greedy sampling.

        NOT THE DEFAULT, and the reason is a property of this target rather
        than of the arithmetic above. Verifying K+1 candidates in ONE
        continued extend is sound for a pure-attention model, where rolling
        back a rejected candidate means freeing its KV slot -- KV is
        addressed per position. A hybrid GDN target also carries a RECURRENT
        state (conv window + SSM), and one extend advances that state over
        every candidate, accepted or not. There is no slot to free: the state
        is a single running value per request. So from the first rejection
        on, the KV says "n accepted tokens" and the recurrent state says "all
        K+1", and the two disagree for the rest of the request. Measured: the
        chain tracks its own no-spec continuation for three rounds and then
        walks away from it (first divergence at index 4, gate below the
        established noise floor).

        The codebase already owns the missing piece -- ``MambaPool.
        SpeculativeState`` keeps ``intermediate_ssm`` and
        ``intermediate_conv_window`` per draft-token step so the state can be
        restored to the accepted prefix -- but only ``ForwardMode.
        TARGET_VERIFY`` reaches it (see ``GDNAttnBackend``'s
        ``if is_target_verify:`` arm). A hand-rolled EXTEND bypasses it by
        construction. Making the lane build a real verify input is the work
        that turns speculation here from correct-but-slow into useful.
        """
        from array import array

        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        batch = job["_batch"]
        req = job["_req"]
        idx = job["_req_pool_idx"]
        n_cached = job["_kv_len"]
        cand = [int(job["_next"][0].item())] + list(proposals)

        self._dbg_round = getattr(self, "_dbg_round", 0) + 1
        if self._dbg_on() and self._dbg_round == 1:
            self._dbg_state_probes(job, n_cached)

        if self._verify_mode(job) == "seqdecode":
            return self._verify_by_decode(job, proposals, n_cached)

        req.full_untruncated_fill_ids = array(
            "q", list(req.origin_input_ids) + job["output_ids"][:-1] + cand
        )
        req.prefix_indices = batch.req_to_token_pool.req_to_token[idx, :n_cached]
        req.set_extend_range(n_cached, n_cached + len(cand))
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.prepare_for_extend()
        if (
            batch.input_ids is None
            and getattr(batch, "prefill_input_ids_cpu", None) is not None
        ):
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        out, ms = self._timed_forward_raw(batch, CaptureHiddenMode.FULL)
        # CONTRACT 8: NOT out.next_token_logits. That field is
        # [#SEQUENCE, vocab] -- the logits processor selects one row per
        # request (the last extend position) via `sample_indices`. Verify
        # needs one row per CANDIDATE. Reading it positionally made
        # `preds[0]` the continuation of the LAST proposal instead of the
        # prediction after `cand[0]`, so the comparison against
        # `proposals[0]` essentially never held: measured accept length
        # exactly 1.000 over all 63 rounds (n_accept always 0, so preds[1:]
        # was never even indexed and the shape mismatch stayed silent), and
        # the emitted token continued a rejected 3-token suffix -- which is
        # why the lane produced a repetition loop rather than the reference
        # continuation.
        #
        # CaptureHiddenMode.FULL already returns the post-norm hidden states
        # for EVERY token in the extend, which is the same tensor the
        # processor derives its own logits from, so the per-candidate logits
        # are one lm_head application away.
        preds = self._candidate_logits(out.hidden_states).argmax(dim=-1)

        if self._dbg_on() and self._dbg_round <= 2:
            self._dbg_verify_rows(job, cand, proposals, out, preds, n_cached)

        n_accept = 0
        for i, prop in enumerate(proposals):
            if int(preds[i].item()) != prop:
                break
            n_accept += 1
        emitted = list(proposals[:n_accept]) + [int(preds[n_accept].item())]

        # KV bookkeeping: the candidates were all written; keep only the
        # accepted prefix plus the token the target itself produced.
        kept = n_accept + 1
        if kept < len(cand):
            surplus = batch.req_to_token_pool.req_to_token[
                idx, n_cached + kept : n_cached + len(cand)
            ]
            batch.token_to_kv_pool_allocator.free(surplus)
        job["_kv_len"] = n_cached + kept
        job["_hidden"] = out.hidden_states[n_accept : n_accept + 1]
        job["_next"] = preds[n_accept : n_accept + 1]
        return emitted, n_accept, ms

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
        if self._job_spec_on(job):
            # Speculation needs the target's hidden states: the head's input
            # is the target's hidden at the position it continues from.
            from sglang.srt.model_executor.forward_batch_info import (
                CaptureHiddenMode,
            )

            out, ms = self._timed_forward_raw(batch, CaptureHiddenMode.FULL)
            next_token_ids = out.next_token_logits.argmax(dim=-1)
            if self._dbg_on():
                self._dbg_prefill_rows(out, job["input_ids"])
            job["_hidden"] = out.hidden_states[-1:]
            job["_kv_len"] = len(job["input_ids"])
            # Prime the head's own KV over the same prompt, using the
            # target's hidden states -- an MTP head attends over its own
            # cache, so it cannot start proposing from an empty one.
            batch_d = self._make_batch(job, runner=self.draft_runner)
            batch_d.prepare_for_extend()
            if (
                batch_d.input_ids is None
                and getattr(batch_d, "prefill_input_ids_cpu", None) is not None
            ):
                batch_d.input_ids = batch_d.prefill_input_ids_cpu.to(
                    batch_d.device, non_blocking=True
                )
                batch_d.prefill_input_ids_cpu = None
            # The head's input is SHIFTED one position left against the
            # target's: an MTP head at row i consumes the target's hidden
            # state of position i together with the token at position i+1,
            # and predicts position i+2. The same construction as
            # ``EAGLEWorker._draft_extend_for_prefill``. Feeding the prompt
            # unshifted primes every row of the head's KV against the wrong
            # hidden state -- it costs no correctness under greedy verify
            # (a bad proposal is simply rejected) but it depresses the accept
            # length, which is the only thing speculation buys.
            batch_d.input_ids = torch.cat(
                (
                    batch_d.input_ids[1:],
                    next_token_ids.to(batch_d.input_ids.dtype).reshape(1),
                )
            )
            self._draft_forward(batch_d, out.hidden_states)
            job["_batch_d"] = batch_d
            # The head's OWN allocated token count. It is not the target's:
            # the head keeps every proposal it ever made, including rejected
            # ones (only the target's KV is rolled back in _verify), so the
            # two counters diverge as soon as one proposal is rejected.
            job["_kv_len_draft"] = len(job["input_ids"])
        else:
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

    def _spec_step(self, job):
        """One speculative round on the lane: K proposals, one verify.

        Reported per ROUND, not per token: ms/round and accept length are the
        two numbers that decide whether speculation paid, and a per-token
        average hides both.
        """
        proposals = self._propose(job)
        emitted, n_accept, ms = self._verify(job, proposals)
        job["decode_ms"].append(ms)
        job["output_ids"].extend(emitted)
        job.setdefault("_accept", []).append(n_accept + 1)

    def _decode_step(self, job):
        batch = job["_batch"]
        batch.input_ids = job["_next"].to(torch.int64)
        batch.prepare_for_decode()
        next_token_ids, ms = self._timed_forward(batch)
        job["decode_ms"].append(ms)
        job["output_ids"].append(int(next_token_ids[0].item()))
        job["_next"] = next_token_ids

    def drop_active(self) -> None:
        """Abandon the active job AND give its pool slots back.

        A tick that raises leaves the job's slots taken in both the target's
        and the head's request tables. With ``max_running_requests`` 1 the
        head has exactly one slot, so a single dropped job otherwise makes
        every LATER job fail in ``alloc_req_slots`` -- one real error turning
        into an unbounded run of misleading ones.
        """
        with self._lock:
            job = self.active
            self.active = None
            if not self.jobs:
                self._busy = False
                self._idle_since = time.monotonic()
        if job is None:
            return
        self._release_draft_batch(job)
        job.pop("_batch_d", None)
        batch = job.pop("_batch", None)
        req = job.pop("_req", None)
        idx = job.get("_req_pool_idx")
        if batch is None or req is None or idx is None:
            return
        try:
            pool = batch.req_to_token_pool
            n_alloc = int(job.get("_kv_len") or 0)
            if n_alloc > 0:
                batch.token_to_kv_pool_allocator.free(pool.req_to_token[idx, :n_alloc])
            if (
                hasattr(pool, "free_mamba_cache")
                and getattr(req, "mamba_pool_idx", None) is not None
            ):
                pool.free_mamba_cache(req)
            pool.free(req)
        except Exception:
            logger.exception(
                "dual-group lane %d: releasing the dropped job failed.",
                self.lane_id,
            )

    def _release_draft_batch(self, job) -> None:
        """Give the head's request slot and KV back.

        Separate from ``_finish`` because it must also run when a job is
        DROPPED: a lane tick that raises still leaves the head's slot taken,
        and with one slot the lane is then permanently unable to start
        another job -- one failure turns into every later job failing.
        """
        batch_d = job.get("_batch_d")
        if batch_d is None:
            return
        try:
            reqs = getattr(batch_d, "reqs", None) or []
            for req in reqs:
                idx = getattr(req, "req_pool_idx", None)
                if idx is None:
                    continue
                pool = batch_d.req_to_token_pool
                n_alloc = int(job.get("_kv_len_draft") or 0)
                if n_alloc > 0:
                    batch_d.token_to_kv_pool_allocator.free(
                        pool.req_to_token[idx, :n_alloc]
                    )
                if (
                    hasattr(pool, "free_mamba_cache")
                    and getattr(req, "mamba_pool_idx", None) is not None
                ):
                    pool.free_mamba_cache(req)
                pool.free(req)
        except Exception:
            # Best-effort: a failure to release must not mask the error that
            # caused the drop, and must not itself abort the lane.
            logger.exception(
                "dual-group lane %d: releasing the head's batch failed.",
                self.lane_id,
            )

    def _finish(self, job):
        batch = job.pop("_batch", None)
        job.pop("_next", None)
        # The HEAD's batch holds a slot in the head's OWN request table
        # (contract 3), and that table has exactly `max_running_requests`
        # entries -- one, here. Releasing only the target's batch leaks it,
        # and the leak is invisible on the job that caused it: the NEXT job
        # dies in `alloc_req_slots` with `available_size()=0`, which reads
        # like a sizing problem and is a lifetime problem. Measured exactly
        # that way -- the first job failed for its own reason, and every
        # later job then failed for this one.
        self._release_draft_batch(job)
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
        job.pop("_batch_d", None)
        decode_ms = job["decode_ms"]
        accepts = job.get("_accept") or []
        result = {
            "input_len": len(job["input_ids"]),
            "spec_mode": self._job_spec_on(job),
            "verify_mode": self._verify_mode(job) if accepts else None,
            "spec_rounds": len(accepts) or None,
            "accept_len_mean": (
                round(sum(accepts) / len(accepts), 3) if accepts else None
            ),
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
    # Speculation on the lane: ONE budget, split -- decided BEFORE the target
    # runner sizes its pools, because afterwards it is too late.
    lane_draft_mib = 0
    if getattr(server_args, "dual_group_lane_spec", False):
        _hdr = _serving_draft_runner(scheduler)
        if _hdr is None:
            raise ValueError(
                "--dual-group-lane-spec: the serving group has no draft "
                "runner to nest into."
            )
        target_mib, lane_draft_mib = split_lane_budget(
            server_args, host_runner.model_config, _hdr.model_config
        )
        lane_args.dual_group_lane_budget_mib = target_mib
        logger.info(
            "dual-group lane %d budget split (one budget, not two): target "
            "%d MiB + NEXTN head %d MiB = %d MiB.",
            lane_id,
            target_mib,
            lane_draft_mib,
            server_args.dual_group_lane_budget_mib,
        )
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
            scheduler,
            server_args,
            lane_args,
            plan,
            lane_id,
            host_runner,
            concurrent,
            lane_draft_mib,
        )


def _disable_graph_phases(args) -> None:
    """Turn every cuda-graph phase off on a lane args view (in place)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    args.disable_cuda_graph = True
    args.disable_prefill_cuda_graph = True
    args.disable_decode_cuda_graph = True
    for phase in Phase.ALL:
        getattr(args.cuda_graph_config, phase).backend = Backend.DISABLED


def _serving_draft_runner(scheduler):
    """The serving group's draft runner, whatever the spec worker calls it.

    EAGLEWorkerV2 wraps an inner EagleDraftWorker and does NOT re-export
    ``draft_runner`` at the top level (the similarly-shaped property nearby
    answers a different question -- which runner ran the last
    shared-buffer-reading phase). Probing by name rather than assuming keeps
    this working across the spec-worker variants and fails with a sentence
    instead of an AttributeError.
    """
    worker = getattr(scheduler, "draft_worker", None)
    if worker is None:
        return None
    for path in (
        ("_draft_worker", "draft_runner"),
        ("draft_runner",),
        ("model_runner",),
    ):
        obj = worker
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "model_config"):
            return obj
    return None


def _build_lane_draft_runner(
    scheduler,
    server_args,
    lane_args,
    plan,
    lane_id,
    lane_target_runner,
    draft_budget_mib,
):
    """Bring up the lane's NEXTN head as a second lane runner (#274).

    Mirrors the serving group's arrangement -- a draft runner alongside the
    target runner in one process -- with the lane's two invariants kept:
    rank-local (no collective anywhere on this path) and shared-byte (the
    head's rank-0 shard is the resident one, verified by data_ptr).

    Its host is the serving group's DRAFT runner, not the target runner: the
    shards the lane's head nests into are the head's shards.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    host_draft_runner = _serving_draft_runner(scheduler)
    if host_draft_runner is None:
        raise ValueError(
            "--dual-group-lane-spec: the serving group has no draft runner "
            "to nest into. The lane's NEXTN head is assembled from the "
            "serving head's shards; without them there is nothing to share."
        )

    logger.info(
        "dual-group lane %d: assembling the NEXTN head (chain, topk 1) from "
        "the serving head's shards.",
        lane_id,
    )
    import copy

    draft_args = copy.copy(lane_args)
    draft_args.dual_group_lane_budget_mib = int(draft_budget_mib)
    # Own graph-plan object: disabling the head's phases must not touch the
    # lane target's (already-captured) plan.
    draft_args.cuda_graph_config = copy.deepcopy(lane_args.cuda_graph_config)
    with lane_geometry_override(1, 0):
        draft_runner = ModelRunner(
            model_config=host_draft_runner.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=lane_target_runner.gpu_id,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=host_draft_runner.dist_port,
            server_args=draft_args,
            is_draft_worker=True,
            is_dual_group_lane=True,
            is_dual_group_lane_draft=True,
            dual_group_host_runner=host_draft_runner,
            dual_group_plan=plan,
            dual_group_lane_id=lane_id,
            dual_group_lane_target_model=lane_target_runner.model,
            # OWN request table, deliberately. Sharing the target's looked
            # like the obvious saving and is wrong: the head runs its own
            # ScheduleBatch, which allocates its own slot, so at
            # max_running_requests 1 the two batches fight over the single
            # slot and the head's prefill dies with
            # "alloc_req_slots runs out of memory, available_size()=0"
            # (measured). The table is mrr x context_len x 8 B -- ~128 KiB
            # here, which is not a post worth that failure mode.
        )
        draft_runner.spec_solo_rank_local_graphs = True
    logger.info(
        "dual-group lane %d NEXTN head assembled (pools follow the target's).",
        lane_id,
    )
    return draft_runner


def _finish_lane_draft_runner(draft_runner, lane_id, target_runner):
    """Pools, backends and graphs of the lane's NEXTN head.

    Separate from the assembly because the assembly has to happen before the
    lane target's pools exist (transient vocab tables) while these have to
    happen after (they must see what the target actually took). The head's KV
    is one layer against the target's many and comes out of the SAME lane
    budget -- the lane's capacity post does not grow behind the operator's
    back.
    """
    # Resource principle 2 again: the head's per-token KV is 1/16 of the
    # target's here, so its budget slice buys 16x the slots the target has --
    # slots it can never use, because it follows the target's sequences.
    # Cap it at the target's token count and the surplus is simply not
    # allocated (measured: ~325 MiB on this vehicle).
    draft_runner.dual_group_lane_token_cap = int(
        getattr(target_runner, "max_total_num_tokens", 0) or 0
    )
    # The head's graph plan is DISABLED on every phase, and that has to be
    # decided BEFORE the scope below publishes this args view.
    _disable_graph_phases(draft_runner.server_args)

    # CONTRACT 4, SOLVED: the head's bring-up runs under the HEAD's args view,
    # not the lane target's.
    #
    # The enclosing bring-up scope publishes ``lane_args`` -- the lane
    # TARGET's config, which has graphs ENABLED (capturing the lane's prefill
    # is the point of the lane). The head is a second runner under that scope
    # with its OWN deep-copied graph plan, and the two disagree about who is
    # authoritative: ``check_cuda_graph_backend`` asks the ACTIVE args (the
    # target's -> enabled, no early return) while
    # ``GraphSharedOutput.create_for_model_runner`` asks
    # ``model_runner.server_args`` (the head's -> disabled -> None). So
    # init_decode_cuda_graph walked past its own DISABLED guard and built a
    # DecodeCudaGraphRunner, which dereferenced the None shared output ->
    # 'NoneType' has no get_logits_buffer. Same split would have hit prefill:
    # init_prefill_cuda_graph deliberately does NOT take the draft-worker skip
    # for a dual-group lane runner, so the head would have captured prefill
    # graphs against the target's plan too.
    #
    # Re-entering the Slice-C overlay with the head's args makes the two agree
    # -- both phases hit their DISABLED early return, nothing is captured, and
    # no shared-output buffer is dereferenced. This is the same mechanism, and
    # the same reasoning, as the comment on the enclosing scope: helpers read
    # the ACTIVE args, so the active args must be the ones belonging to the
    # runner being brought up. The whole body is inside it (not just the graph
    # call) so pool sizing and backend init see the head's plan as well --
    # an attention backend that allocates cuda-graph state for a plan the head
    # never captures is wasted lane budget.
    #
    # The three contracts solved before this one, each found by a boot, so the
    # next iteration does not rediscover them:
    #   1. the head needs its own KV sizing (its config reports the target's
    #      64 layers; the configurator lands on cell size 0),
    #   2. it needs init_cuda_graphs for the EAGER runner alone, otherwise
    #      forward dispatch has no eager_runner,
    #   3. it must NOT share the target's req_to_token_pool -- its own
    #      ScheduleBatch allocates its own slot and at mrr 1 the two batches
    #      deadlock on the single slot.
    from sglang.srt.runtime_context import current_lane_id, lane_scope

    # Same lane IDENTITY as the enclosing scope (read, not re-derived: it
    # decides which graph pool / per-lane buffer the head lands in, and it
    # must not drift from the scope the lane will forward under).
    with lane_scope(current_lane_id(), draft_runner.server_args):
        _finish_lane_draft_runner_scoped(draft_runner, lane_id)

    logger.info(
        "dual-group lane %d NEXTN head ready: max_total_num_tokens=%d.",
        lane_id,
        draft_runner.max_total_num_tokens,
    )


def _finish_lane_draft_runner_scoped(draft_runner, lane_id):
    """Pools, backends and graphs of the head, under the head's args view."""
    with lane_geometry_override(1, 0):
        draft_runner.alloc_memory_pool()
        draft_runner.init_attention_backends()
        # init_cuda_graphs() is called even though the head runs eager: it
        # also builds the EAGER phase runner, without which forward dispatch
        # has no `eager_runner` at all (measured: AttributeError on the
        # head's first forward). Every graph phase is DISABLED in the active
        # view, so this builds the dispatcher and captures nothing.
        draft_runner.init_cuda_graphs()
        # NAMED GAP, not an oversight: the head runs EAGER.
        # Its decode graph would be captured by the plain DecodeCudaGraphRunner,
        # which builds a dummy batch with spec_info=None -- and an MTP forward
        # reads forward_batch.spec_info.hidden_states, so the capture dies on
        # a NoneType. The serving group avoids this by capturing drafts
        # through the EAGLE-specific draft graph runner, which knows to build
        # an EagleDraftInput. Routing the lane's head there is the follow-on;
        # until then the head is one layer of eager against a graph-captured
        # target, which is where the time actually is.
        logger.warning(
            "dual-group lane %d NEXTN head runs EAGER: the generic decode "
            "capture builds spec_info=None and an MTP forward dereferences "
            "it. Needs the EAGLE draft graph runner -- named gap, not an "
            "end state.",
            lane_id,
        )


def _build_lane_under_scope(
    scheduler,
    server_args,
    lane_args,
    plan,
    lane_id,
    host_runner,
    concurrent=False,
    lane_draft_mib=0,
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

        # ORDER MATTERS, and it cost a boot to learn: the NEXTN head's MODEL
        # is assembled BEFORE either runner allocates its pools.
        #
        # The head's complement load constructs the draft's own
        # ParallelLMHead before anything can hand it the target's -- a
        # transient of 1.19 GiB at this vocabulary, freed moments later by
        # set_embed_and_head_modules. Allocating the lane target's pools
        # first leaves ~600 MiB and that transient OOMs. Assembled here it
        # has the whole pre-pool headroom, and the freed bytes are back
        # before either pool is sized -- the same reasoning the serving
        # group's early embed/head share is built on.
        draft_runner = None
        if getattr(server_args, "dual_group_lane_spec", False):
            draft_runner = _build_lane_draft_runner(
                scheduler,
                server_args,
                lane_args,
                plan,
                lane_id,
                runner,
                lane_draft_mib,
            )

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

        if draft_runner is not None:
            _finish_lane_draft_runner(draft_runner, lane_id, runner)

    lane = DualGroupLane(
        lane_id=lane_id,
        plan=plan,
        runner=runner,
        concurrent=concurrent,
        draft_runner=draft_runner,
        spec_steps=int(getattr(server_args, "dual_group_lane_spec_steps", 3)),
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
