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
import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from sglang.srt.distributed.dual_group import (
    NestedGroupPlan,
    check_nesting,
    derive_nested_plan,
    format_vram_posts,
    lane_part_device_indices,
    local_column_gather,
    local_row_reduce,
    local_row_split,
    transformer_nesting_probes,
)
from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    scoped_tp_partition_ratios,
)
from sglang.srt.managers.admission_limiter import (
    AdmissionLimiter,
    admission_limiter_scope,
)
from sglang.srt.model_executor.lane_device_clock import LaneDeviceClock
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# #673: how long teardown waits for a lane worker before detaching it. Bounded
# and short on purpose -- the worker can be inside a kernel launch or a stream
# sync where it never observes the stop event, and hanging shutdown is worse
# than the abort it prevents. Replaces an unmeasured 10 s, and matches the
# kvso-dest-io sibling so the family has one deadline class.
DGL_WORKER_JOIN_TIMEOUT_S: float = 2.0
DGL_LOG_PREFIX = "[#673 dual-group-lane]"

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

    ``moe_ep`` moves with them and is forced to the SINGLE-GROUP view.
    ``FusedMoE.__init__`` reads ``get_parallel().moe_ep_size / moe_ep_rank``
    directly, and the lane's expert algebra is the moe-TP one (each part is a
    partial sum of the same full-width output).  Leaving the ambient EP
    geometry in place would make a freshly loaded lane part claim the SAME
    local expert range as the resident one, and would let the part build an
    EP dispatcher bound to the production all-to-all group.  Expert
    parallelism in the serving group is refused outright at build time
    (``_assert_lane_moe_is_pure_tp``); this override keeps the construction
    honest for every other case.
    """
    return get_parallel().override(
        tp_size=fast_size,
        tp_rank=fast_rank,
        moe_tp_size=fast_size,
        moe_tp_rank=fast_rank,
        moe_ep_size=1,
        moe_ep_rank=0,
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
    # The quantization block size comes from the CHECKPOINT, not from a
    # guess: it is what decides in how coarse a unit the layers will really
    # partition the MLP/MoE dimensions (#274 families slice B).
    quant_cfg = getattr(model_config.hf_config, "quantization_config", None) or {}
    if not isinstance(quant_cfg, dict):
        quant_cfg = getattr(quant_cfg, "__dict__", {}) or {}
    probes = transformer_nesting_probes(
        plan,
        num_attention_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        intermediate_size=intermediate,
        num_experts=num_experts,
        linear_attn_units=linear_attn_units,
        vocab_units=vocab_units,
        moe_intermediate_size=getattr(cfg, "moe_intermediate_size", None),
        weight_block_size=quant_cfg.get("weight_block_size"),
        quant_method=getattr(model_config, "quantization", None),
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


def _part_device(part: nn.Module) -> Optional[torch.device]:
    """The device a lane part's weights live on (None: no tensors at all)."""
    for t in part.parameters():
        return t.device
    for t in part.buffers():
        return t.device
    return None


def _part_devices(parts: Sequence[nn.Module]) -> Tuple[Optional[torch.device], ...]:
    return tuple(_part_device(p) for p in parts)


def _spans_cards(parts: Sequence[nn.Module]) -> bool:
    """Whether these parts sit on more than one device (#274 families 2, arm C).

    A lane that does not fit on one card places its materialized parts on
    other cards. The shells stay the same substitution -- gather is still a
    ``cat``, reduce is still an ``add`` -- but the ACTIVATION now travels to
    the part and the result travels back. That is the whole cost of the
    two-card lane, and it is the small object: one ``[tokens, hidden]``
    tensor per shell instead of a weight shard.
    """
    seen = {d for d in _part_devices(parts) if d is not None}
    return len(seen) > 1


def _on(x: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
    """Move an activation to a part's device; a no-op on the one-card lane."""
    if device is None or x.device == device:
        return x
    return x.to(device, non_blocking=True)


#: Stateless and reentrant, so one shared instance serves every no-op guard
#: below instead of allocating one per shell per forward.
_NO_DEVICE_SWITCH = contextlib.nullcontext()


def _active_device(device: Optional[torch.device], home: Optional[torch.device] = None):
    """Make a part's card the CURRENT cuda device for that part's compute.

    The exact companion of ``_on``: wherever the activation has to TRAVEL, the
    context has to travel with it. Moving the tensor alone is not the same
    thing as switching the context the kernel is launched into, and the two are
    only interchangeable for backends that derive their own guard from the
    tensors they are handed. cuBLAS does -- which is why the dense bf16
    two-card lane survived without this. Triton does not: it validates every
    pointer argument against the ACTIVE context and rejects a correctly-placed
    activation on a card that is not the current one:

        ValueError: Pointer argument (at 0) cannot be accessed from Triton

    Every FP8 and INT4 quant method in this tree launches Triton, so a lane
    whose parts span cards has to switch the context per part, not just the
    tensors.

    ``home`` is the card the shell was called on. Passing it keeps the ONE-card
    lane -- the common case, and the one that runs under CUDA graphs -- free of
    any ``cudaSetDevice`` at all: same condition as ``_on``'s early return, so
    a part that needs no hop needs no guard either.
    """
    if device is None or device == home or device.type != "cuda":
        return _NO_DEVICE_SWITCH
    return torch.cuda.device(device)


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
        self._lane_part_devices = _part_devices(parts)

    def forward(self, input_):
        parts = self._lane_parts
        devices = self._lane_part_devices
        home = input_.device
        outs = []
        for p, dev in zip(parts, devices):
            with _active_device(dev, home):
                out = p.quant_method.apply(
                    p,
                    _on(input_, dev),
                    None if (p.bias is None or p.skip_bias_add) else p.bias,
                )
            outs.append(_on(out, home))
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
        self._lane_part_devices = _part_devices(parts)
        p0 = parts[0]
        self.skip_bias_add = p0.skip_bias_add

    def forward(self, input_, skip_all_reduce: bool = False, forward_batch=None):
        # skip_all_reduce is accepted for signature parity with
        # RowParallelLinear; the lane's "all-reduce" is the local sum below and
        # is never optional (without it the output is a partial product).
        parts = self._lane_parts
        home = input_.device
        pieces = local_row_split(input_, self._lane_in_sizes)
        outs = []
        for p, piece, dev in zip(parts, pieces, self._lane_part_devices):
            with _active_device(dev, home):
                part_out = p.quant_method.apply(p, _on(piece, dev), None)
            outs.append(_on(part_out, home))
        out = local_row_reduce(outs)
        bias = parts[0].bias
        if bias is not None and not self.skip_bias_add:
            out = out + _on(bias, out.device)
            bias = None
        return out, (bias if self.skip_bias_add else None)


class LaneFusedMoEShell(nn.Module):
    """Full-width MoE forward over N sharded expert parts (#274 slice, arm C).

    The fifth shell class, and the only one that is not a linear layer. The
    expert tensors (``w13_weight``/``w2_weight`` plus their scales, packed
    weights and expert maps) never pass through the
    ``output_partition_sizes`` / ``input_size_per_partition`` interface the
    other shells stand on, which is why the taxonomy could not carry them.

    What it CAN stand on is the shape of the collective. A ``FusedMoE`` rank
    always produces a PARTIAL SUM of the same full-width output, in both shard
    modes:

    * expert-dim sharding -- each rank owns a disjoint expert range and
      contributes zero for foreign experts (they route to the zero-padding
      expert),
    * intermediate-dim sharding -- each rank computes a slice of every
      expert's inner GEMM.

    One all-reduce combines either of them, so the local substitution is the
    same one ``LaneRowParallelShell`` performs: call each part's
    ``forward_local`` (which is ``forward_impl`` with the group all-reduce
    split off) and add. Exact, not an approximation -- the addition is the
    one the collective would have performed.

    The routing decision is NOT re-derived here. ``topk_output`` is computed
    once by the hull's replicated router over the global expert numbering and
    handed to every part unchanged; each part translates it to its own local
    numbering exactly as it does when it runs in the serving group.
    """

    def __init__(self, parts: Sequence[nn.Module]):
        super().__init__()
        if not parts:
            raise ValueError("LaneFusedMoEShell: no parts.")
        missing = [p for p in parts if not hasattr(p, "forward_local")]
        if missing:
            raise ValueError(
                "LaneFusedMoEShell: part module "
                f"{type(missing[0]).__name__} has no forward_local(); the "
                "lane cannot split its group all-reduce off and would sum "
                "already-reduced outputs."
            )
        if _spans_cards(parts):
            raise ValueError(
                "LaneFusedMoEShell: this lane places its parts on different "
                "cards, and the expert path is not carried across cards. The "
                "linear shells move one activation per shell; a FusedMoE part "
                "consumes the ROUTING as well (topk ids/weights, and on some "
                "paths a dispatcher), so a cross-card expert shell is its own "
                "build and is refused rather than half-done."
            )
        self._lane_parts = tuple(parts)
        p0 = parts[0]
        # Read-only geometry the surrounding block asks the experts about.
        for attr in ("num_experts", "top_k", "hidden_size", "layer_id"):
            if hasattr(p0, attr):
                setattr(self, attr, getattr(p0, attr))

    def forward(self, hidden_states, topk_output, **kwargs):
        out = None
        for p in self._lane_parts:
            piece = p.forward_local(hidden_states, topk_output)
            out = piece if out is None else out + piece
        return out


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
        self._lane_part_devices = _part_devices(parts)

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
        home = input_.device
        for p, dev in zip(self._lane_parts, self._lane_part_devices):
            idx = p.shard_indices
            with _active_device(dev, home):
                masked_input, input_mask = get_masked_input_and_mask(
                    _on(input_, dev),
                    idx.org_vocab_start_index,
                    idx.org_vocab_end_index,
                    idx.num_org_vocab_padding,
                    idx.added_vocab_start_index,
                    idx.added_vocab_end_index,
                )
                part_out = p.quant_method.embedding(p, masked_input.long())
                part_out.masked_fill_(input_mask.unsqueeze(-1), 0)
            part_out = _on(part_out, home)
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
        shell = self._shell
        home = x.device
        outs = []
        for p, dev in zip(shell._lane_parts, shell._lane_part_devices):
            with _active_device(dev, home):
                part_out = p.quant_method.apply(p, _on(x, dev), bias)
            outs.append(_on(part_out, home))
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
        self._lane_part_devices = _part_devices(parts)
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
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    part_dicts = [_module_dict(m) for m in part_models]
    hull_modules = list(hull.named_modules())
    counts = {
        "column": 0,
        "row": 0,
        "embedding": 0,
        "lm_head": 0,
        "composed": 0,
        "moe": 0,
    }
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
        if isinstance(module, FusedMoE):
            # Before the linear branches on purpose: an EP subclass of
            # FusedMoE is still a FusedMoE, and its expert tensors must never
            # fall through to a linear shell.
            set_by_name(name, LaneFusedMoEShell(parts_for(name)))
            counts["moe"] += 1
        elif isinstance(module, ParallelLMHead):
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


def _compose_home_device(
    target: torch.Tensor, parts: Sequence[nn.Module]
) -> Optional[torch.device]:
    """Where a composed-by-value tensor is assembled.

    The hull's own device when it has one; a meta hull has none, and then the
    first part's device is the answer -- lane rank order puts the host's own
    resident shard first, so that IS the lane's card.
    """
    if target is not None and target.device.type != "meta":
        return target.device
    for p in parts:
        dev = _part_device(p)
        if dev is not None and dev.type != "meta":
            return dev
    return None


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
    # The composed tensors are built from EVERY part, and a card-spanning lane
    # has parts on more than one device. The pieces are gathered on the HULL's
    # card -- the hull is what they are written into, and it is the lane's own
    # card by construction. These are conv kernels and per-head vectors, so
    # the gather is kilobytes and happens once at bring-up.
    home = _compose_home_device(hull_linear.weight, parts)
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
            pieces.append(_on(p.weight.data[off : off + sizes[s]], home))
    full = torch.cat(pieces, dim=0)
    if full.shape != hull_linear.weight.shape:
        raise ValueError(
            f"composed weight shape {tuple(full.shape)} != hull "
            f"{tuple(hull_linear.weight.shape)} for a composed column linear."
        )
    if hull_linear.weight.device.type == "meta":
        # Meta hull (every family whose full-width weights are not lazily
        # allocated): this ONE tensor has to become real, because it is
        # composed by value rather than shelled. It is a conv kernel, i.e.
        # kilobytes -- which is the whole point of not making the rest real.
        hull_linear.weight = nn.Parameter(
            torch.empty_like(full, device=full.device), requires_grad=False
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
        base = name.rsplit(".", 1)[-1]
        if param.device.type == "meta" and base not in ("dt_bias", "A_log"):
            raise ValueError(
                f"dual-group lane: hull parameter {name!r} is still on meta "
                "and has no counterpart in the shared tree -- it would run "
                "with no storage at all."
            )
        if base in ("dt_bias", "A_log"):
            home = _compose_home_device(param, part_models)
            pieces = [_on(pp[name].data, home) for pp in part_params]
            full = torch.cat(pieces, dim=0)
            if param.device.type == "meta":
                # Same reason as the composed conv weight: a per-head GDN
                # vector is composed BY VALUE, so it needs storage even on a
                # meta hull. Per head, not per weight matrix -- kilobytes.
                parent_name, _, attr = name.rpartition(".")
                parent = hull.get_submodule(parent_name) if parent_name else hull
                param = nn.Parameter(torch.empty_like(full), requires_grad=False)
                setattr(parent, attr, param)
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


def _load_lane_part(
    lane_runner, plan: NestedGroupPlan, fast_rank: int, gpu_id: Optional[int] = None
) -> nn.Module:
    """Load one MATERIALIZED part: the lane group's rank ``fast_rank`` shard.

    Runs the STOCK loader under (a) the lane's own partition vectors --
    without the scope the resident group's vector does not apply to a group
    of another size and the loader silently falls back to the even split --
    and (b) the ParallelContext override for (fast_size, fast_rank).  The v2
    parameter loaders read tp geometry from exactly these two sources.

    ``gpu_id`` is the IN-PROCESS cuda index the part is loaded onto; it
    differs from the lane's own card only for a lane that spans two cards
    (arm C).  The device is set for the whole load, not just handed to the
    ``DeviceConfig``: the loaders allocate through ``torch.empty`` and
    ``torch.cuda.current_device()`` in several places, and one of them
    landing on the wrong card is a silent half-placed part.
    """
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.model_loader import get_model_loader

    if gpu_id is None:
        gpu_id = lane_runner.gpu_id
    fams = {name: list(vec) for name, vec in plan.fast_family_ratios}
    t0 = time.perf_counter()
    with (
        (
            torch.cuda.device(gpu_id)
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        ),
        scoped_tp_partition_ratios(list(plan.fast_ratio), fams or None),
        lane_geometry_override(plan.fast_size, fast_rank),
    ):
        loader = get_model_loader(
            load_config=lane_runner.load_config,
            model_config=lane_runner.model_config,
        )
        model = loader.load_model(
            model_config=lane_runner.model_config,
            device_config=DeviceConfig(lane_runner.device, gpu_id),
        )
    logger.info(
        "dual-group lane: part rank %d (of ratio %s) loaded on cuda:%d in %.1f s",
        fast_rank,
        list(plan.fast_ratio),
        gpu_id,
        time.perf_counter() - t0,
    )
    return model.eval()


def hull_needs_real_storage(model_config) -> bool:
    """Whether the hull must be built with real storage rather than on meta.

    The real build exists for ONE reason: ``RadixLinearAttention`` captures
    views of its conv weight / ``dt_bias`` / ``A_log`` Parameter objects at
    construction time, so those have to be real for
    ``_compose_column_weights_inplace`` to fill them in place instead of
    re-plumbing captured references.  Only linear-attention (GDN/mamba)
    families have such tensors.

    For every other family the choice is not cosmetic, it decides whether the
    lane can exist at all: the full-width hull is allocated ON TOP of the
    serving shard and the lane parts, only to be dropped again by
    ``assemble_lane_shells`` moments later.  On this rig that transient peak
    is the difference between a lane that boots and a CUDA OOM in the hull's
    first attention layer (#274 families slice A).

    Whether the peak exists at all depends on the QUANTIZATION, not on the
    family.  GGUF registers ``GGUFUninitializedParameter``, so a real hull
    costs kilobytes -- which is why the GGUF-GDN vehicle never noticed the
    real build.  Every other path (unquantized ``unquant.py``, and fp8's
    ``create_weights``) allocates the full-width weight with ``torch.empty``.
    The FP8-GDN vehicle sits in the intersection the slice-A predicate got
    wrong: linear attention AND eagerly allocated weights, i.e. a real hull
    of the entire model (28.75 GiB at 27B) that no card here can absorb
    (#274 families slice 2).  Such families build on meta as well; the two
    composed-by-value tensor classes (the conv kernel and the per-head GDN
    vectors) are materialized individually where they are filled, which is
    kilobytes rather than the whole model.
    """
    cfg = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    if cfg is None:
        return True
    linear_attention = bool(
        getattr(cfg, "gdn_tp_units", None)
        or getattr(cfg, "linear_num_key_heads", None)
        or getattr(cfg, "linear_attn_config", None)
        or getattr(cfg, "mamba_d_ssm", None)
    )
    hull_weights_are_lazy = getattr(model_config, "quantization", None) == "gguf"
    return linear_attention and hull_weights_are_lazy


def _fill_hull_buffers(hull: nn.Module, shared_model: nn.Module) -> int:
    """Give every meta BUFFER of a meta-built hull the shared tree's storage.

    ``_finalize_hull_params`` walks parameters only.  Buffers (rotary caches,
    norm scales registered as buffers, ...) are replicated by construction --
    none of them carries a TP shard -- so the shared tree's counterpart is
    the right one by name.  Anything left on meta is a hard error: a meta
    buffer survives assembly silently and fails one forward later inside a
    kernel.
    """
    shared_buffers = dict(shared_model.named_buffers())
    filled = 0
    for name, buf in list(hull.named_buffers()):
        if buf is None or buf.device.type != "meta":
            continue
        sb = shared_buffers.get(name)
        if sb is None or sb.shape != buf.shape:
            raise ValueError(
                f"dual-group lane: hull buffer {name!r} (shape "
                f"{tuple(buf.shape)}) is still on meta and has no replicated "
                f"counterpart in the shared tree (found "
                f"{None if sb is None else tuple(sb.shape)}). It would be "
                "read with no storage at all."
            )
        parent_name, _, attr = name.rpartition(".")
        parent = hull.get_submodule(parent_name) if parent_name else hull
        setattr(parent, attr, sb)
        filled += 1
    return filled


def _refresh_captured_linear_attention_tensors(hull: nn.Module) -> int:
    """Re-point the linear-attention layer's CAPTURED tensor references.

    ``RadixLinearAttention.__init__`` stores plain references to its mixer's
    ``conv1d.weight`` view, ``A_log`` and ``dt_bias``. On a meta hull those
    references are meta, and the two composed-by-value tensors are given real
    storage by REPLACING the Parameter object on the owning mixer -- which
    leaves the captured reference behind, pointing at nothing. The kernel then
    fails with "All inputs must be on the same device", which is the good
    outcome; the bad one would be a silent read.

    This is why the slice-A predicate built such families with a real hull at
    all. Refreshing the three references afterwards is the cheaper half of
    that trade: it costs one walk, and it lets the hull stay on meta for a
    family whose real hull is the whole model (#274 families slice 2).

    Returns the number of references refreshed. Nothing to refresh is a valid
    answer -- a dense family has no such layer.
    """
    refreshed = 0
    for _, mixer in hull.named_modules():
        attn = getattr(mixer, "attn", None)
        if attn is None or not hasattr(attn, "conv_weights"):
            continue
        conv1d = getattr(mixer, "conv1d", None)
        if conv1d is not None and getattr(conv1d, "weight", None) is not None:
            w = conv1d.weight
            attn.conv_weights = w.view(w.size(0), w.size(2))
            attn.bias = conv1d.bias
            refreshed += 1
        for name in ("A_log", "dt_bias"):
            owned = getattr(mixer, name, None)
            if owned is not None:
                setattr(attn, name, owned)
                refreshed += 1
    return refreshed


def _build_hull(lane_runner, device=None) -> nn.Module:
    """Construct the full-width hull tree, on meta unless the family forbids it.

    See ``hull_needs_real_storage``: linear-attention families need real
    storage for the captured conv views, everything else is built on meta so
    that an unquantized full-width hull never allocates a second copy of the
    model on the lane card.
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


def resolve_lane_part_gpu_ids(server_args, plan, host_gpu_id: int) -> Tuple[int, ...]:
    """IN-PROCESS cuda index for every lane part, in lane-rank order.

    Unset ``--dual-group-lane-part-gpu-id`` keeps every part on the host
    rank's card -- the one-card lane of slices A-C, byte-for-byte the same
    build.  When it IS set it names PHYSICAL GPU ids (the same space as
    ``--rank-gpu-id``), which this process can only address because the
    parent listed them in ``CUDA_VISIBLE_DEVICES`` before the process
    started; the translation to in-process indices uses that list.
    """
    spec = getattr(server_args, "dual_group_lane_part_gpu_id", None)
    if not spec:
        return tuple(host_gpu_id for _ in range(plan.fast_size))
    if len(spec) != plan.fast_size:
        raise ValueError(
            f"--dual-group-lane-part-gpu-id has {len(spec)} entries but the "
            f"lane group has {plan.fast_size} ranks ({plan.describe()}). One "
            "physical GPU id per LANE rank, not per serving rank."
        )
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = (
        [int(x) for x in visible_env.split(",") if x != ""] if visible_env else None
    )
    host_phys = visible[0] if visible else host_gpu_id
    indices = lane_part_device_indices(host_phys, spec, visible)
    host_fast = plan.host_fast_rank(0)
    if indices[host_fast] != host_gpu_id:
        raise ValueError(
            f"--dual-group-lane-part-gpu-id puts lane rank {host_fast} on "
            f"physical GPU {spec[host_fast]}, but that rank IS the host "
            f"rank's resident shard and cannot move off its card "
            f"(physical {host_phys}). Name the host card there."
        )
    return tuple(int(i) for i in indices)


def _lane_spans_cards(server_args) -> bool:
    """Whether the configured lane places a part on a foreign card."""
    spec = getattr(server_args, "dual_group_lane_part_gpu_id", None)
    return bool(spec) and len(set(spec)) > 1


def _assert_lane_moe_is_pure_tp(model: nn.Module) -> None:
    """Refuse a lane over an EXPERT-PARALLEL MoE serving group.

    The lane's expert algebra (``LaneFusedMoEShell``) is the moe-TP one: N
    parts, each a partial sum of the same full-width output, combined by an
    addition that stands in for one all-reduce. Expert parallelism is a
    different decomposition -- it routes tokens between ranks with a live
    all-to-all dispatcher -- and a lane part built under it would (a) claim
    the same local expert range as the resident part rather than a
    complementary one, and (b) carry a dispatcher bound to the production
    communicator into a lane forward, which is exactly the wire operation
    this runtime is built not to have. Refused at build time, named.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    for name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        ep_size = int(getattr(module, "moe_ep_size", 1) or 1)
        if ep_size > 1:
            raise ValueError(
                f"dual-group lane: the serving group runs expert parallelism "
                f"(moe_ep_size={ep_size} on {name!r}). The lane's expert "
                "shell substitutes ONE all-reduce with an addition, which is "
                "the moe-TP decomposition; EP routes tokens between ranks and "
                "has no local substitute. Run the serving group with "
                "moe-TP-only expert sharding, or run without --dual-group-lane."
            )
        dispatcher = getattr(module, "dispatcher", None)
        if dispatcher is not None and type(dispatcher).__name__ not in (
            "StandardDispatcher",
        ):
            raise ValueError(
                f"dual-group lane: expert module {name!r} dispatches through "
                f"{type(dispatcher).__name__}, which is a cross-rank all-to-all "
                "path. The lane has no communicator; only the standard "
                "(pure-TP) dispatcher can run inside it."
            )
        if getattr(module, "_moe_offload_enabled", False):
            raise ValueError(
                f"dual-group lane: expert module {name!r} has expert offload "
                "enabled (#77, SGLANG_MOE_RESIDENT_EXPERT_FRACTION). Every "
                "materialized lane part would install its OWN pinned host "
                "cache beside the serving group's -- host memory nobody "
                "budgeted. The combination is refused until the lane parts "
                "register with the resident offloader instead."
            )


def build_lane_model(lane_runner, kind: str = "target") -> nn.Module:
    """The lane ModelRunner's load_model body: lane parts + hull + shells +
    the shared-byte gate.  Rank-local by contract."""
    host_runner = lane_runner.dual_group_host_runner
    plan: NestedGroupPlan = lane_runner.dual_group_plan
    if host_runner is None or plan is None:
        raise ValueError(
            "dual-group lane runner needs dual_group_host_runner and dual_group_plan."
        )

    _assert_lane_moe_is_pure_tp(host_runner.model)

    host_fast_rank = plan.host_fast_rank(host_runner.tp_rank)
    part_gpu_ids = resolve_lane_part_gpu_ids(
        lane_runner.server_args, plan, lane_runner.gpu_id
    )
    cards = sorted(set(part_gpu_ids))
    free_before = {c: torch.cuda.mem_get_info(c)[0] for c in cards}

    # Part models in lane-rank order: the host's own segment reuses the
    # resident model (shared bytes); every other segment is materialized on
    # its assigned card -- a complement (several BIG ranks) or, at BIG
    # tp_size 2, another rank's byte-identical singleton that this process
    # simply cannot reach.
    part_models: List[nn.Module] = []
    for f in range(plan.fast_size):
        if f == host_fast_rank:
            part_models.append(host_runner.model)
        else:
            part_models.append(
                _load_lane_part(lane_runner, plan, f, gpu_id=part_gpu_ids[f])
            )

    free_after_complement = {c: torch.cuda.mem_get_info(c)[0] for c in cards}
    # The NEXTN head's hull goes on META: unlike the target it has no GDN
    # mixer whose conv views are captured at construction (its single layer
    # is full attention), and its own vocab tables are 2.37 GiB that the
    # lane target's shells replace immediately. Everything the head's hull
    # still needs real is replicated and is aliased onto the shared tree's
    # storage by _finalize_hull_params.
    on_meta = kind == "draft" or not hull_needs_real_storage(lane_runner.model_config)
    hull = _build_hull(lane_runner, device="meta" if on_meta else None)
    counts = assemble_lane_shells(hull, part_models)
    fill = _finalize_hull_params(hull, host_runner.model, part_models)
    fill["buffers"] = _fill_hull_buffers(hull, host_runner.model) if on_meta else 0
    # After the composed tensors got their storage, and before any forward.
    fill["captured"] = (
        _refresh_captured_linear_attention_tensors(hull) if on_meta else 0
    )

    checked = verify_shared_bytes(hull, part_models[host_fast_rank], host_fast_rank)
    if checked == 0:
        # The gate is the whole point of the host segment: its bytes MUST be
        # the resident ones. Zero identities means the assembly silently
        # produced a private copy.
        raise ValueError(
            f"dual-group lane ({kind}): shared-byte gate found 0 data_ptr "
            f"identities for host lane rank {host_fast_rank} -- the lane "
            "would compute on copies, not on the serving rank's bytes."
        )
    logger.info(
        "dual-group lane %s model assembled (hull on %s, parts on cuda:%s): "
        "shells column=%d "
        "row=%d embed=%d lm_head=%d moe=%d composed=%d; params aliased=%d "
        "composed_vec=%d buffers=%d captured=%d; shared-byte gate PASSED (%d "
        "data_ptr "
        "identities).",
        kind,
        "meta" if on_meta else "device",
        ",".join(str(g) for g in part_gpu_ids),
        counts["column"],
        counts["row"],
        counts["embedding"],
        counts["lm_head"],
        counts["moe"],
        counts["composed"],
        fill["aliased"],
        fill["composed_vec"],
        fill["buffers"],
        fill["captured"],
        checked,
    )

    free_after = {c: torch.cuda.mem_get_info(c)[0] for c in cards}
    home = lane_runner.gpu_id
    added_mib = max(0, free_before[home] - free_after[home]) >> 20
    part_mib = max(0, free_before[home] - free_after_complement[home]) >> 20
    lane_runner.dual_group_lane_weight_added_mib = int(added_mib)
    # The §5 posts block, with MEASURED numbers (the shared post is exact by
    # the gate above; the nested/duplicated posts are what mem_get_info saw).
    from sglang.srt.distributed.dual_group import DUPLICATED, NESTED, SHARED, VramPost

    posts = [
        VramPost(
            name=f"shared serving-rank shard (lane rank {host_fast_rank})",
            status=SHARED,
            mib=0,
            why=f"data_ptr-verified, {checked} identities",
        ),
        VramPost(
            name="lane part shard(s) on this card",
            status=NESTED,
            mib=int(part_mib),
            why="bytes the other cards hold; this card now holds its share of "
            "the full weights exactly once",
        ),
        VramPost(
            name="hull tree residue (composed conv/state vectors, buffers)",
            status=DUPLICATED,
            mib=int(added_mib - part_mib),
            why="small real tensors of the full-width hull; big weights are "
            "lazy/shelled",
        ),
    ]
    for c in cards:
        if c == home:
            continue
        # A foreign card carries the part in FULL: the resident shard over
        # there belongs to another process and cannot be aliased from here.
        # That is the price of the two-card lane, and it is stated as such.
        posts.append(
            VramPost(
                name=f"lane part shard on foreign card cuda:{c}",
                status=DUPLICATED,
                mib=int(max(0, free_before[c] - free_after[c]) >> 20),
                why="another process owns the resident copy of these bytes; "
                "the lane cannot alias across process boundaries",
            )
        )
    logger.info("%s", format_vram_posts(posts, f"cuda:{home} ({kind})"))
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


def kv_bearing_layer_count(model_config, is_head: bool = False) -> int:
    """How many layers of ``model_config`` actually hold a KV cache.

    This is NOT ``num_hidden_layers``, and the difference is the whole point:

    * On a hybrid model most layers are linear/GDN and hold recurrent state,
      not KV. Only ``full_attention_layer_ids`` pays per token. On the Qwen3.5
      family that is every fourth layer -- 16 of 64.
    * A NEXTN head's config is built from the TARGET's checkpoint, so it
      reports the target's layer count AND the target's full-attention layer
      ids even though the head is ONE decoder layer. The head therefore has
      to be read off ``num_nextn_predict_layers``, which the draft-config
      derivation sets explicitly (``ModelConfig._maybe_patch_draft_arch``),
      and never off either of the two layer lists it inherited.

    Both branches are family-neutral -- a model without a hybrid split falls
    through to its plain layer count -- and both are config-only, so this can
    run before any pool, model or CUDA context exists.
    """
    if is_head:
        nextn = getattr(model_config, "num_nextn_predict_layers", None)
        if nextn is None:
            hf = getattr(model_config, "hf_config", None)
            nextn = getattr(hf, "num_nextn_predict_layers", None)
        if isinstance(nextn, int) and nextn > 0:
            return nextn
        # No NEXTN declaration: fall through to the generic count rather than
        # guessing 1. A head whose config says nothing is not known to be one
        # layer deep, and over-reserving is the recoverable direction.

    hf = getattr(model_config, "hf_config", None)
    text = hf.get_text_config() if hasattr(hf, "get_text_config") else hf
    for holder in (text, model_config):
        ids = getattr(holder, "full_attention_layer_ids", None)
        if ids:
            return len(ids)
    return max(1, int(getattr(model_config, "num_hidden_layers", 1) or 1))


@dataclass(frozen=True)
class LaneBudgetSplit:
    """How the operator's ONE lane budget is divided, itemized (#313 idiom).

    Carried as an object rather than a bare pair so the boot log can say WHY
    each post has the size it has -- the same reason
    :class:`LadderReserveDemand` carries its posts.
    """

    budget_mib: int
    target_mib: int
    draft_mib: int
    target_kv_layers: int
    draft_kv_layers: int

    def ledger(self) -> str:
        return (
            f"lane budget {self.budget_mib} MiB = target {self.target_mib} MiB "
            f"({self.target_kv_layers} KV-bearing layer(s)) + NEXTN head "
            f"{self.draft_mib} MiB ({self.draft_kv_layers}); the head's share "
            f"is {self.draft_kv_layers}/"
            f"{self.draft_kv_layers + self.target_kv_layers} because both "
            "pools hold the SAME token count at the same page size and their "
            "per-token cells differ only in that layer count"
        )


def split_lane_budget(server_args, target_model_config, draft_model_config):
    """Split the operator's ONE lane budget between the lane's target and its
    NEXTN head (#274 slice C), from the RANK-LOCAL per-token KV cell.

    Resource principle 2: no VRAM is duplicated that does not have to be.
    Both lane runners read ``--dual-group-lane-budget-mib``, so without a
    split the head would silently claim a SECOND full budget -- the lane's
    capacity post would double behind the operator's back. It is one budget
    and it is divided, not two budgets that happen to share a name.

    THE SPLIT RULE, and why it is this one. The head follows the target's
    sequences token for token, so the only correct size for its pool is the
    target's TOKEN COUNT -- not a fraction of the target's bytes. Both pools
    are built from the same ``kv_heads * (head_dim + v_head_dim) * elem_size``
    cell and differ only in how many layers pay it, so "same tokens" is
    exactly the layer-count ratio::

        draft_mib / budget = L_head / (L_head + L_target)

    and the dtype, the head dimensions and the page size all cancel. On the
    Qwen3.5 vehicle that is 1/(1+16) -- 94 MiB of a 1600 MiB budget.

    WHAT THIS REPLACES, named because it was measured and not guessed: the
    previous rule took the ratio of ``num_hidden_layers``, which on a real
    NEXTN draft config is the TARGET's own layer count (the draft config is
    derived from the target's checkpoint). The ratio therefore came out at 1.0
    on every real boot and the result was decided by the clamp behind it --
    a flat quarter of the budget, 400 MiB, of which the head's pool then used
    ~75 and the token cap threw the remaining ~325 away: not allocated, and
    not given back to the lane target either. The unit test that covered the
    rule fed a draft config reporting ONE layer, which no real draft config
    does, so the arithmetic was never exercised on its production input.

    Config-only and rank-local: no profiling, no collective, and callable
    before either runner has a pool -- which it must be, because the lane
    target sizes its pool from what is left here.
    """
    budget = int(server_args.dual_group_lane_budget_mib or 0)
    target_layers = kv_bearing_layer_count(target_model_config)
    draft_layers = kv_bearing_layer_count(draft_model_config, is_head=True)
    # Rounded UP: the head running one page short of the target is a hard
    # failure mode (it cannot follow the sequence), one page of slack is not.
    draft_mib = -(-budget * draft_layers // (draft_layers + target_layers))
    # The head must stay allocatable, and it must never crowd out the target
    # it exists to serve. Both bounds only ever bind on a degenerate ratio.
    draft_mib = max(1, min(draft_mib, max(1, budget // 2)))
    split = LaneBudgetSplit(
        budget_mib=budget,
        target_mib=budget - draft_mib,
        draft_mib=draft_mib,
        target_kv_layers=target_layers,
        draft_kv_layers=draft_layers,
    )
    logger.info("dual-group %s", split.ledger())
    return split.target_mib, split.draft_mib


def lane_chain_verify_mask(n_cached: int, draft_token_num: int, device=None):
    """The FULL_MASK tree mask of a topk-1 CHAIN, for ONE request.

    Layout contract, taken from ``build_tree_kernel_efficient``: a flat bool
    vector of ``seq_lens_sum * D + D * D * bs`` entries, row-major over the D
    draft tokens, each row covering ``seq_len_i + D`` key positions. Row i
    sees the whole committed prefix plus candidates 0..i -- a chain has no
    branching, so the draft-to-draft block is plain lower-triangular. Written
    out here rather than obtained from the tree kernel because that kernel
    needs an EAGLE draft's ``parent_list``/``top_scores_index``, which the
    lane's hand-rolled chain never builds.
    """
    prefix = torch.ones((draft_token_num, n_cached), dtype=torch.bool, device=device)
    block = torch.tril(
        torch.ones((draft_token_num, draft_token_num), dtype=torch.bool, device=device)
    )
    return torch.cat((prefix, block), dim=1).flatten()


def build_lane_chain_verify_input(candidates, n_cached: int, device=None):
    """An ``EagleVerifyInput`` describing the lane's chain as a verify TREE.

    The lane proposes a chain (topk 1) and verifies it greedily itself, so
    only the fields the TARGET forward reads have to be real: the candidate
    ids, their absolute positions, the attention mask, and ``draft_token_num``
    (which the GDN backend uses as the per-request stride of the verify and
    as the step width of the intermediate state caches). The ``retrieve_*``
    fields describe the chain topology for completeness; the tree-verify
    sampling kernels that consume them are not on this path (the lane's
    accept rule is its own), and the GDN backend reads them only for
    ``topk > 1``.
    """
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    cand = list(candidates)
    d = len(cand)
    idx = torch.arange(d, dtype=torch.long, device=device)
    next_token = torch.where(idx < d - 1, idx + 1, torch.full_like(idx, -1))
    return EagleVerifyInput(
        draft_token=torch.tensor(cand, dtype=torch.int64, device=device),
        custom_mask=lane_chain_verify_mask(n_cached, d, device=device),
        positions=torch.arange(n_cached, n_cached + d, dtype=torch.long, device=device),
        retrieve_index=idx.view(1, d),
        retrieve_next_token=next_token.view(1, d),
        retrieve_next_sibling=torch.full((1, d), -1, dtype=torch.long, device=device),
        retrieve_cum_len=None,
        spec_steps=d - 1,
        topk=1,
        draft_token_num=d,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=n_cached,
        seq_lens_cpu=torch.tensor([n_cached], dtype=torch.int64),
    )


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


def plan_prefill_chunks(prefix_len: int, total_len: int, chunk: int):
    """The chunk spans of a chunked lane prefill: ``[(start, end), ...]``
    tiling ``[prefix_len, total_len)`` exactly once, in order, each span at
    most ``chunk`` tokens (§13.10 point 1).

    Pure and total: every valid input yields a plan whose spans are
    contiguous and whose last end is ``total_len``; invalid inputs raise
    instead of degrading to an empty plan (an empty plan would skip the
    prefill silently and surface as a KV-less decode much later).
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if not 0 <= prefix_len < total_len:
        raise ValueError(
            f"prefix_len {prefix_len} outside [0, {total_len}) -- nothing to " "prefill"
        )
    spans = []
    start = prefix_len
    while start < total_len:
        end = min(start + chunk, total_len)
        spans.append((start, end))
        start = end
    return spans


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
        spec_rungs=None,
        spec_adaptive: bool = False,
        spec_hysteresis: int = 4,
    ):
        self.lane_id = lane_id
        self.plan = plan
        self.runner = runner
        # #287: the lane's OWN concurrency knob, built lazily in
        # ``admission_limiter``. Per lane on purpose -- a serving-group
        # throttle must not reach into the lane and vice versa.
        self._admission_limiter = None
        # Speculation on the lane (#274 slice C): the NEXTN head runner, or
        # None. Its presence is what turns a lane step into a verify round.
        self.draft_runner = draft_runner
        self.spec_steps = int(spec_steps)
        # #274 round 7a: the chain-length LADDER and the policy that walks it.
        # ``spec_rungs`` None means the pre-ladder behaviour -- one rung, the
        # configured spec_steps -- and the policy then only ever answers with
        # that one value, so the round shape is unchanged.
        from sglang.srt.model_executor.lane_spec_policy import LaneSpecPolicy

        self.spec_rungs: Tuple[int, ...] = (
            tuple(spec_rungs) if spec_rungs else (int(spec_steps),)
        )
        self.spec_policy = LaneSpecPolicy(
            self.spec_rungs,
            adaptive=bool(spec_adaptive),
            hysteresis=int(spec_hysteresis),
            default_rung=(
                int(spec_steps) if int(spec_steps) in self.spec_rungs else None
            ),
        )
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
        # Off-ladder chunk sizes already warned about (once per size per
        # lane; the warning names a capture-economics fact, not an error).
        self._chunk_ladder_warned: set = set()
        #: Top-2 logit margin of the last committed plain-decode token, or
        #: None. Only written under SGLANG_LANE_MARGIN_PROBE (see
        #: _margin_probe_on); the commit sites read it right after the
        #: forward that produced it.
        self._last_margin: Optional[float] = None
        self._busy = False
        self._admission_waits: List[float] = []
        self.lending = None  # set by the scheduler when stage-2 lending is on
        # Pairing objective of the two-class scheduler (#274 slice D): set by
        # the scheduler when --dual-group-lane-pairing is on. None keeps the
        # job pick byte-identical to the FIFO pop(0) below.
        self.pairing_policy = None
        self.results_total = 0
        self.prefill_tokens_total = 0
        self.decode_steps_total = 0
        # Fine-grained monotone work counters for the online card-equivalent
        # estimator (#274 slice D, S1).  The three counters above are bumped
        # at JOB FINISH, which is the wrong grain for a 1-second window: a
        # 32-token decode job takes over a second, so a window sees either 0
        # or a whole job.  These are bumped the moment the work COMPLETES,
        # and they are kept separate per ARM because a prefill-shaped and a
        # decode-shaped step have different solo floors and may not be
        # averaged into one rate.
        self.work_total = {"prefill_tokens": 0, "decode_tokens": 0}
        # Monotone TIME counters next to the monotone WORK counters (#284).
        # work_total answers "how much did the lane get done"; this answers
        # "how much of the card did it have while doing it", and the r8 caveat
        # is that the first question alone cannot tell a lane that lost the
        # card from a lane whose kernels ran slower on it.
        self.device_clock = LaneDeviceClock(
            lambda: torch.cuda.Event(enable_timing=True), None
        )
        self._register_offload_items()

    def _rung_size_source(self, rung: int):
        """Live size source of one capture rung: the #93 tagged-pool state
        record (``footprint_bytes``: measured paused bytes when available,
        else noted-tensor bytes). Empty (0) until the rung is captured;
        ``maybe_refresh_item_sizes()`` at worker start -- and any later
        post-capture refresh -- turns it into the real figure, so on GPU the
        true number appears without further change."""

        def source() -> object:
            try:
                from sglang.srt.speculative.adaptive_graph_memory import (
                    AdaptiveGraphMemoryManager,
                    get_active_manager,
                )

                mgr = get_active_manager()
                if mgr is None:
                    return 0
                tag = AdaptiveGraphMemoryManager.tag_for_steps(rung)
                rec = getattr(mgr, "_states", {}).get(tag)
                return rec if rec is not None else 0
            except Exception:
                return 0

        return source

    def _cold_lane_size_source(self):
        """Live size source of the whole cold lane: the sum of everything
        this lane alone owns that is ALREADY booked in the register (rungs,
        drafter head, lane workspaces, lane input-buffer pools). An honest
        lower bound -- draft-KV remains are not itemized yet (they start
        fresh at the turn, Erg. 6)."""

        def source() -> int:
            from sglang.srt.model_executor.offload_register import (
                get_global_register,
            )

            reg = get_global_register()
            if reg is None:
                return 0
            own_prefixes = (
                f"lane{self.lane_id}/graph_rung/",
                f"lane{self.lane_id}/drafter_head",
                f"lane_workspace/{self.scope_lane_id}/",
                f"input_buffer/{self.scope_lane_id}/",
            )
            total = 0
            for klass in ("graph_rungs", "drafter_heads", "lane_workspaces"):
                for item in reg.items_of_class(klass):
                    if item.item_id.startswith(own_prefixes):
                        total += item.size_bytes
            return total

        return source

    def _register_offload_items(self) -> None:
        """#286 offload register adapter: book this lane's capture-ladder
        rungs (Erg. 4), its drafter head (Erg. 6) and the whole cold lane
        (class d) as register items -- registration, hot-criterion wiring,
        LIVE size sources and movement-payload binding; the moving itself is
        the backend's job. Gated behind SGLANG_OFFLOAD_REGISTER (default off
        => no-op; the default path is byte-unchanged)."""
        from sglang.srt.model_executor.offload_movement import (
            SuspendPayload,
            TagPayload,
        )
        from sglang.srt.model_executor.offload_register import (
            maybe_bind_movement_payload,
            maybe_register_item,
            offload_register_enabled,
        )

        if not offload_register_enabled():
            return
        source_device = int(getattr(self.runner, "gpu_id", 0) or 0)
        for rung in self.spec_rungs:
            if rung == 0:
                # K=0 is the plain no-spec decode entry; it costs no extra
                # graph pool, so there is nothing to park.
                continue
            item_id = f"lane{self.lane_id}/graph_rung/k{rung}"
            maybe_register_item(
                item_id,
                "graph_rungs",
                0,
                # The ACTIVE rung is hot and must stay resident; captured
                # graphs must survive a park without re-capture, hence the
                # VA-stability requirement (#93 VMM remap in the GPU phase).
                hot=lambda k=rung: self.spec_policy.current == k,
                va_stable_required=True,
                time_constant_tier="turn",
                size_source=self._rung_size_source(rung),
            )
            try:
                from sglang.srt.speculative.adaptive_graph_memory import (
                    AdaptiveGraphMemoryManager,
                )

                # Movement route of a rung: its #93 tagged capture pool.
                # GPU-phase wiring note: pausing this tag must be arbitrated
                # WITH AdaptiveGraphMemoryManager.ensure_active (not behind
                # its back) -- that arbitration is a named open GPU item.
                maybe_bind_movement_payload(
                    item_id,
                    TagPayload(AdaptiveGraphMemoryManager.tag_for_steps(rung)),
                    source_device,
                )
            except Exception:
                pass  # no tag vocabulary available (non-spec build)
        if self.draft_runner is not None:
            maybe_register_item(
                f"lane{self.lane_id}/drafter_head",
                "drafter_heads",
                0,
                # The head is hot while the lane is mid-tick; between jobs it
                # is the Erg.-6 park candidate. The phase mask records that a
                # round only needs it during the draft step (Erg. 7c stage-2
                # interface; the turn tier governs actual parking for now).
                hot=lambda: self._busy,
                va_stable_required=True,
                phase_mask=("draft",),
                time_constant_tier="turn",
                # Live size: the draft runner's model parameters/buffers --
                # on GPU this resolves to the real head footprint at the
                # first refresh, without further change.
                size_source=lambda: self.draft_runner,
            )
            # No payload bind yet: the head is va_stable_required, so its
            # route is a #93 TAGGED allocation of the head weights at load
            # time (VMM remap, Erg. 6). Allocating the head into a tagged
            # region is an open GPU-phase item; binding a TensorPayload here
            # would be refused by the backend (and rightly so).
        # Class d: the WHOLE cold lane (everything this lane owns alone --
        # graphs, workspaces, draft-KV remains; the byte-shared weights
        # belong to the group and are never part of the item). Parked via the
        # #89 suspend path.
        cold_id = f"lane{self.lane_id}/cold_lane"
        maybe_register_item(
            cold_id,
            "cold_lane",
            0,
            hot=lambda: self._busy or bool(self.jobs) or self.active is not None,
            va_stable_required=True,
            time_constant_tier="turn",
            size_source=self._cold_lane_size_source(),
        )
        # #89 suspend route. The tag below is the named ATTACH POINT for
        # lane-scoped memory-saver tags; today's tags are process-wide
        # (weights/kv/graphs), so creating this lane-scoped tag in the saver
        # is a named open GPU-phase item (audit table).
        maybe_bind_movement_payload(
            cold_id,
            SuspendPayload(tags=(f"lane{self.lane_id}/own",)),
            source_device,
        )

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
                    # Diagnostic cap on the accept length (see
                    # _verify_by_target_verify). Per JOB rather than per
                    # process for the same reason as the two above: the capped
                    # and the uncapped side have to come from ONE boot, or the
                    # comparison carries boot-to-boot variance instead of the
                    # one difference it is meant to isolate.
                    "tv_max_accept": job.get("tv_max_accept"),
                    # Round 6, same reasoning again: False makes this job's
                    # verify skip its captured graph and run eager, so the
                    # replay-vs-eager byte gate is one boot rather than two.
                    "verify_graph": job.get("verify_graph"),
                    # Round 7a, three more of the same kind. ``spec_steps``
                    # PINS a rung for this job (that is how each rung of the
                    # ladder gets its own measured row from one boot);
                    # ``adaptive`` overrides the server flag either way, so
                    # the adaptive arm and the fixed arms it is judged against
                    # come from ONE boot; ``head_graph`` False makes this job's
                    # head forwards eager, which is the head's replay-vs-eager
                    # byte gate.
                    "spec_steps": job.get("spec_steps"),
                    "adaptive": job.get("adaptive"),
                    "head_graph": job.get("head_graph"),
                    # One extra lm_head per round over all K+1 rows, so it is
                    # per job: the round that answers the 13-of-96 argmax
                    # question must not distort the same boot's timing table.
                    "argmax_check": job.get("argmax_check"),
                    # Round 7b posten 0, and the same reasoning a fifth time:
                    # False keeps the head's runaway sequence length -- the
                    # defect as it stood -- so the fix and the state it fixes
                    # are measured against each other in ONE boot rather than
                    # across two, on a quantity (accept length) that is
                    # content-driven and therefore worst of all to compare
                    # across boots.
                    "draft_rollback": job.get("draft_rollback"),
                    # Round 7c posten 2, and the same reasoning a sixth time:
                    # False leaves the head's OWN hidden in the KV of the
                    # accepted positions -- the state before this round's fix --
                    # so "re-seeded against the target" and "not re-seeded" are
                    # two arms of ONE boot on the same token ids.
                    "draft_reseed": job.get("draft_reseed"),
                    # §13.10, and the same reasoning once more: a per-job
                    # chunk size (0 forces the single forward under a set
                    # server flag) lets the chunked and the unchunked prefill
                    # run as two arms of ONE boot -- the coherence gate the
                    # chunking posten owes is a same-boot byte comparison,
                    # not an argument.
                    "prefill_chunk": job.get("prefill_chunk"),
                    # #404: a label the caller puts on the job so the pool
                    # checksum records say WHICH arm produced them. The lane
                    # never reads it; it only travels, which is the point -- a
                    # jsonl whose lines cannot be attributed to an arm has to
                    # be joined by arrival order, and arrival order is exactly
                    # what a concurrent lane does not guarantee.
                    "probe_tag": job.get("probe_tag"),
                }
            )
        # PD priority (addendum 5): work has arrived for the protected class.
        # Reclaim anything lent to the scavenger BEFORE waking the worker --
        # the reclaim latency is the guarantee quality of that priority and
        # is measured, not assumed.
        if self.lending is not None:
            self.lending.on_lane_work_arrived()
        self._idle_since = None
        # The busy interval opens when work ARRIVES, not when the worker picks
        # it up: the time between the two is time the lane held work and did
        # not run, which is exactly the quantity the duty cycle is meant to
        # expose (a feeder gap would otherwise hide inside it).
        self.device_clock.mark_busy()
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
            # Per-arm work completed, bumped at completion rather than at job
            # finish -- the numerator of share_lane (see lane_share.py).
            "work_total": dict(self.work_total),
            # The TIME counters that turn share_lane from a number into a
            # diagnosis (#284): device ms on the lane's own stream and wall ms
            # spent holding work, both monotone, both differenced per window.
            "device_clock": self.device_clock.snapshot().to_json(),
            "results": self.results[-8:],
            "weight_added_mib": getattr(
                self.runner, "dual_group_lane_weight_added_mib", None
            ),
            "max_total_num_tokens": getattr(self.runner, "max_total_num_tokens", None),
            "concurrent": self.concurrent,
            "spec": (
                None
                if self.draft_runner is None
                else {
                    "algorithm": "nextn-chain",
                    "steps": self.spec_steps,
                    # Round 7a: the ladder and the policy walking it, so an
                    # external instrument can read the rung the lane is on
                    # without differencing per-job results.
                    "rungs": list(self.spec_rungs),
                    "policy": self.spec_policy.stats(),
                }
            ),
        }
        if self.pairing_policy is not None:
            stats["pairing"] = self.pairing_policy.snapshot()
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

        from sglang.srt.model_executor.offload_register import (
            maybe_refresh_item_sizes,
        )

        # #286: by worker start the lane's allocations (rung pools, drafter
        # head) exist, so the live size sources now resolve to real bytes
        # (no-op when SGLANG_OFFLOAD_REGISTER is off).
        maybe_refresh_item_sizes()

        low, high = torch.cuda.Stream.priority_range()
        # Escape hatch, not a tuning knob: NCCL's collective kernels
        # spin-wait, so if the protected lane ever starved the serving
        # group's freshly launched all-reduce the symptom would be a
        # rank-0-late group stall rather than a slowdown. Setting this to 0
        # makes both classes equal-priority and isolates that question.
        high = int(os.environ.get("SGLANG_DUAL_GROUP_LANE_STREAM_PRIORITY", high))
        self.stream = torch.cuda.Stream(device=self.runner.gpu_id, priority=high)
        # The clock records on the lane's stream from here on. Before this
        # point (and on the serial path) it records on the current stream,
        # which is the lane's for the duration of a serial tick -- the same
        # measurement, taken where the lane is the only writer.
        self.device_clock.bind_stream(self.stream)
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

    def stop_worker(self, timeout_s: float = DGL_WORKER_JOIN_TIMEOUT_S) -> str:
        """Stop this lane's worker thread. Returns what happened (#673).

        THE HANDLE IS ONLY CLEARED ON A REAL JOIN. The previous version set
        ``self._thread = None`` UNCONDITIONALLY after a timed-out join, with no
        ``is_alive()`` check and no log. A thread still running -- with its own
        CUDA stream, launching kernels -- was detached silently AND the object
        forgot it existed, so a second call returned success for a thread it
        had abandoned. That is the #673 abort shape with the evidence deleted.

        THE DEADLINE IS BOUNDED, and short. The old 10 s was never measured
        against anything; a teardown budget is seconds, not tens of seconds,
        and this worker can be parked in a kernel launch or a stream sync where
        it will not observe the stop event at all. Waiting longer would trade
        an abort for a hang, which is worse: the abort at least ends the
        process.

        Never raises, idempotent.
        """
        thread = getattr(self, "_thread", None)
        if thread is None or not thread.is_alive():
            self._thread = None
            return "already stopped"
        self._stop.set()
        self._wake.set()
        thread.join(timeout=float(timeout_s))
        if thread.is_alive():
            logger.warning(
                "%s %s did not stop within %.2fs and is being DETACHED "
                "deliberately. It owns a CUDA stream and launches kernels, so "
                "it is most likely inside one; waiting further would hang "
                "shutdown instead of ending it. The handle is KEPT so the leak "
                "stays visible -- a later stop will report 'detached' again "
                "rather than claiming success.",
                DGL_LOG_PREFIX,
                thread.name,
                float(timeout_s),
            )
            return "detached"
        self._thread = None
        return "joined"

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
                            idle = True
                        else:
                            self._busy = True
                            idle = False
                    if idle:
                        self.device_clock.mark_idle()
                        break
                    try:
                        with torch.cuda.stream(self.stream):
                            self._step_locked_scope()
                    except Exception as exc:
                        # #867: THE WORST EXPRESSION OF THE CLASS. Dropping the
                        # job and continuing is right for a job-local failure
                        # and catastrophic for a poisoned CUDA context: this is
                        # a daemon thread, so it would go straight back round
                        # and launch the NEXT job's kernels into a dead
                        # context, job after job, burying the origin under one
                        # identical traceback per job.
                        from sglang.srt.distributed.device_communicators import (
                            barlink_abort_gate,
                        )

                        if barlink_abort_gate.is_poison_error(exc):
                            source = f"dual-group lane {self.lane_id} step"
                            if barlink_abort_gate.record_poison(source, exc):
                                logger.error(
                                    "#867 dual-group lane %d standing down: the "
                                    "step hit an UNSURVIVABLE CUDA fault (%s). "
                                    "The context is unusable, so this thread "
                                    "stops rather than launching the next job "
                                    "into it. Treat THIS as the origin, not the "
                                    "traceback that lands next.",
                                    self.lane_id,
                                    exc,
                                )
                            self.drop_active()
                            return
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
                # Pairing objective (#274 slice D): a work-conserving reorder
                # of THIS queue is the policy's whole actuation surface. With
                # the policy off or absent this is the FIFO pop(0) it always
                # was, byte for byte. The pick runs once per JOB (not per
                # forward), so it adds nothing to the Python between two lane
                # forwards -- the submission-gap half of #284's loss.
                idx = 0
                if self.pairing_policy is not None and len(self.jobs) > 1:
                    idx = self.pairing_policy.pick(self.jobs)
                self.active = self.jobs.pop(idx)
            job = self.active
        self._submitted.clear()
        with torch.no_grad():
            if job["prefill_ms"] is None:
                self._prefill(job)
            elif self._job_spec_on(job):
                self._spec_round(job)
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
                    self.device_clock.mark_idle()
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
                self._spec_round(job)
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
            with admission_limiter_scope(self.admission_limiter):
                with lane_geometry_override(1, 0):
                    yield

    @property
    def admission_limiter(self) -> AdmissionLimiter:
        """The lane's own admission limit (#287).

        The lane is dimensioned by --dual-group-lane-max-requests, so that
        value is both its ceiling and its start; the auto controller stays
        off because a PD prefill lane's concurrency is a placement decision,
        not a pressure response. It exists as a separate instance so that a
        read inside the lane's scope resolves to the LANE's number and never
        to the serving group's -- the same de-globalization the config
        overlay does one line above.
        """
        if self._admission_limiter is None:
            self._admission_limiter = AdmissionLimiter(
                max(1, int(self.runner.server_args.dual_group_lane_max_requests)),
                lane_id=self.lane_id,
            )
        return self._admission_limiter

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
        # Round 7a: did this forward replay the head's captured graph? Asked
        # of the runner that decides it, not inferred from a timing -- a head
        # that silently fell back to eager is the one way the round-7a table
        # can look like a regression when nothing regressed (the same argument
        # as ``verify_graph_rounds`` in round 6).
        graph_runner = getattr(self.draft_runner, "decode_cuda_graph_runner", None)
        self._head_graph_last = bool(
            graph_runner is not None and graph_runner.lane_draft_can_replay(fb)
        )
        # The one forward on the lane's paths with no event pair of its own,
        # and the one whose cost the round's ms/token has to carry: a K-chain
        # runs K of these per verify. The span defers its read (#284), so
        # adding it costs two event records and no synchronize.
        with self.device_clock.span():
            out = self.draft_runner.forward(fb).logits_output
        return out

    # -- proposal perturbation (test-only, env-gated, off by default) -------
    #
    # #404 round 2. Everything else this window built OBSERVES the rollback;
    # this one drives it. The remaining hypothesis after residue volume,
    # captured static buffers and the ``_kv_len`` read side were eliminated is
    # that the leak needs corrupted proposal CONTENT -- so the falsifier has to
    # produce some, deterministically, at a chosen round, and then ask whether
    # a single committed token moved.
    #
    # The property under test is an invariance, and it is the one the greedy
    # accept rule promises: a REJECTED proposal is not a token, it is a guess
    # that cost a KV slot and a recurrent step, and the target's own prediction
    # decides what gets committed. Corrupting proposal ``i`` of round ``r``
    # therefore has exactly one legal effect -- the chain stops at ``i``
    # instead of later -- and exactly one illegal one: a committed position
    # that differs from the unperturbed run. The second is a leak, and it is
    # localizable by the probe above, because both runs record the same
    # surfaces at the same committed lengths.
    #
    # ``SGLANG_LANE_PROPOSAL_PERTURB="round:index:delta"``, all three integers:
    # at the round-th speculative round of a job (0-based, job-local), add
    # ``delta`` to proposal ``index``. Unset, nothing in this file behaves
    # differently -- there is no default, no "sometimes", and no way to reach
    # the hook from a serving path that did not ask for it by name. A malformed
    # value raises rather than degrading to a no-op: a falsifier that silently
    # did not fire would be reported as evidence of an invariance it never
    # tested.
    @staticmethod
    def _proposal_perturbation() -> Optional[Tuple[int, int, int]]:
        raw = os.environ.get("SGLANG_LANE_PROPOSAL_PERTURB")
        if not raw:
            return None
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                "SGLANG_LANE_PROPOSAL_PERTURB must be 'round:index:delta' "
                f"(three integers), got {raw!r}"
            )
        try:
            round_index, token_index, delta = (int(p) for p in parts)
        except ValueError:
            raise ValueError(
                "SGLANG_LANE_PROPOSAL_PERTURB must be 'round:index:delta' "
                f"(three integers), got {raw!r}"
            ) from None
        if round_index < 0 or token_index < 0:
            raise ValueError(
                "SGLANG_LANE_PROPOSAL_PERTURB: round and index are 0-based and "
                f"non-negative, got {raw!r}"
            )
        return round_index, token_index, delta

    def _perturb_proposals(self, job, proposals: List[int]) -> List[int]:
        """Corrupt one proposal token, if this job's round is the chosen one.

        The round index is JOB-LOCAL and counted by the hook itself, so the
        falsifier lands on the same round whatever else the lane is serving and
        whatever else the round happens to record. A process-wide counter would
        make it depend on the traffic around it; borrowing one of the round's
        own lists would make it depend on which caller drove the round.
        """
        spec = self._proposal_perturbation()
        if spec is None:
            return proposals
        round_index, token_index, delta = spec
        seen = int(job.get("_perturb_rounds") or 0)
        job["_perturb_rounds"] = seen + 1
        if seen != round_index or token_index >= len(proposals):
            return proposals
        if not getattr(self, "_perturb_announced", False):
            self._perturb_announced = True
            logger.warning(
                "dual-group lane %d: SGLANG_LANE_PROPOSAL_PERTURB is set. This "
                "is a FALSIFIER hook -- it corrupts speculative proposals on "
                "purpose. It must not be set on a serving boot.",
                self.lane_id,
            )
        before = int(proposals[token_index])
        after = before + int(delta)
        proposals = list(proposals)
        proposals[token_index] = after
        job.setdefault("_perturbed", []).append(
            {"round": round_index, "index": token_index, "from": before, "to": after}
        )
        return proposals

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
        steps = int(job.get("_rung") or self.spec_steps)
        # Round 7b posten 0, the direct falsifier: the head's own sequence
        # length against the target's. They must be EQUAL at the top of every
        # round -- the head's next forward writes the token at target position
        # ``_kv_len`` -- and any non-zero difference means the head is being
        # asked about a position the sequence is not at.
        draft_len = int(batch_d.seq_lens[0].item())
        job["_round_start"] = draft_len
        job.setdefault("_draft_lag", []).append(
            draft_len - int(job.get("_kv_len") or 0)
        )
        # Cleared per round: the full-accept catch-up in ``_rollback_draft``
        # may only ever use THIS round's verify output.
        job["_verify_hidden"] = None
        job["_verify_last_token"] = None
        job["_verify_rows"] = None
        job["_verify_tokens"] = None
        with self._head_graph_scope(job):
            for _ in range(steps):
                batch_d.input_ids = token.to(torch.int64)
                batch_d.prepare_for_decode()
                out = self._draft_forward(batch_d, hidden)
                job["_head_graph"] = job.get("_head_graph", 0) + int(
                    getattr(self, "_head_graph_last", False)
                )
                job["_head_forwards"] = job.get("_head_forwards", 0) + 1
                token = out.next_token_logits.argmax(dim=-1)
                hidden = out.hidden_states
                proposals.append(int(token[0].item()))
                job["_kv_len_draft"] = int(job.get("_kv_len_draft") or 0) + 1
        # LAST, so the head's own KV and hidden chain are exactly what they
        # would have been: the falsifier corrupts what the verify is asked
        # about, not what the head believes it proposed.
        return self._perturb_proposals(job, proposals)

    @contextlib.contextmanager
    def _head_graph_scope(self, job):
        """Let a job run its head forwards EAGER (``head_graph: false``).

        The head's replay-vs-eager byte gate, per job for the same reason the
        verify's is (round 6): both arms have to come from ONE boot, or the
        gate carries boot-to-boot variance instead of the single difference it
        exists to isolate. Suppressing the capture flag is the whole mechanism
        -- ``can_run_graph`` reads it and the forward falls back to the eager
        runner -- and it is restored unconditionally.
        """
        runner = getattr(self.draft_runner, "decode_cuda_graph_runner", None)
        if job is None or job.get("head_graph") is not False or runner is None:
            yield
            return
        saved = runner._lane_draft_captured
        runner._lane_draft_captured = False
        try:
            yield
        finally:
            runner._lane_draft_captured = saved

    def _make_batch(self, job, runner=None, req=None):
        from array import array

        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runner = runner or self.runner
        if req is None:
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
        # else: a CONTINUING chunked-prefill request (#274 §13.10). The req
        # keeps its req_pool_idx and mamba slot -- ReqToTokenPool.alloc
        # reuses the slot of a request with committed KV, which is what
        # carries the GDN/mamba recurrent state across chunk forwards -- and
        # the caller has already advanced prefix_indices and extend_range.

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

    # -- top-2 margin probe (env-gated, off by default) --------------------
    #
    # #284 -> r12: whether a speculative flip is inherent batch-shape numerics
    # or a defect is decided by the top-2 LOGIT MARGIN at the flipped
    # position. The r12 control answered that for stock NEXTN over
    # /generate's logprob channel, but the lane commits its tokens itself and
    # reports only ``output_ids``, so the same question could not be asked of
    # the lane. This records the margin at every position the lane COMMITS.
    #
    # Off by default and gated on an env var for the reason the row oracle is:
    # it costs a topk over the vocabulary and a device read per committed
    # token, which is real money on a decode path whose whole point is round
    # time. Nothing reads ``_margins`` unless the flag is set.
    def _margin_probe_on(self) -> bool:
        return bool(os.environ.get("SGLANG_LANE_MARGIN_PROBE"))

    @staticmethod
    def _top2_margin(next_token_logits) -> float:
        """logit(top1) - logit(top2) for row 0, as a float.

        Logits, not logprobs: softmax is shift-invariant, so the gap between
        the two best entries is the same quantity either way, and this side
        already holds the logits.
        """
        top2 = torch.topk(next_token_logits[0], 2)
        return float(top2.values[0].item() - top2.values[1].item())

    def _record_margin(self, job, value) -> None:
        if value is not None:
            job.setdefault("_margins", []).append(round(float(value), 6))

    def _record_verify_margins(self, job, logits, n_emitted: int) -> None:
        """One margin per EMITTED token of a batched verify round.

        Verify row ``i`` is by construction the forward that decided
        ``emitted[i]``: rows ``0 .. n_accept`` map one-to-one onto the emitted
        block and the rows past the first rejection are dropped. That is the
        same rule ``_verify_by_decode`` applies to its per-candidate forwards,
        and it has to hold here too, because the result row advertises
        ``margins`` as aligned with ``output_ids``.

        Without this the alignment breaks silently and in the direction that
        matters. ``target_verify`` is the DEFAULT verify mode, so a
        speculative job used to record exactly one margin -- the prefill
        token's -- against 64 committed ids, and ``r12/verdict.py`` reads
        ``margins[first_divergent_index]`` positionally. A short list makes
        the lane's own perturbation band unmeasurable and hands the world-A /
        world-B decision back to the pre-registered ``NEAR_TIE_ABS``
        constant, which is precisely what #284 recorded must not decide it.

        Costs a topk per emitted token and is therefore gated, like every
        other call site, on ``SGLANG_LANE_MARGIN_PROBE``.
        """
        if not self._margin_probe_on() or logits is None:
            return
        rows = int(getattr(logits, "shape", (0,))[0])
        for i in range(min(int(n_emitted), rows)):
            self._record_margin(job, self._top2_margin(logits[i : i + 1]))

    # -- per-round pool checksum probe (env-gated, off by default) ---------
    #
    # #404. The bracket window falsified the VOLUME story for the pool-axis
    # rollback -- 18 arms, 738 rejected candidate rows, a flat dose response --
    # and left the instrument where it started: "the tokens diverged somewhere
    # downstream" is a detection, not a localisation. This turns it into one.
    # After every COMMITTED round the lane hashes the three surfaces a rejected
    # candidate can leave residue in, and a no-spec job standing at the SAME
    # committed position must produce the same hashes. A mismatch names the
    # surface and the round instead of the token index of a downstream flip.
    #
    # The surfaces, and why each boundary is where it is:
    #
    #   ``map``   ``req_to_token[idx, :committed_len]`` -- the slot mapping the
    #             round committed. The FREED TAIL is excluded deliberately: the
    #             verify writes ``d`` candidate pointers into the row and frees
    #             only the rejected candidates' SLOTS, leaving their pointers
    #             behind (`_verify_by_target_verify`). Those stale pointers
    #             differ between a speculative and a non-speculative job by
    #             construction, so hashing them would make every arm red and
    #             carry no information at all. ``SGLANG_LANE_POOL_CHECKSUM_TAIL``
    #             points the probe at the tail instead, which is the probe's own
    #             can-fail arm: an instrument that cannot be pointed at the
    #             wrong place and shown to miss has not been calibrated.
    #   ``kv``    the KV pool rows those committed slots point to -- key and
    #             value, every full-attention layer. This is the surface the
    #             rollback is accused of leaking into.
    #   ``conv`` / ``ssm``   the request's PERSISTENT recurrent state, hashed
    #             apart so a GDN-carrier and a KV-carrier separate on sight.
    #             ``MambaPool.SpeculativeState``'s per-draft-step
    #             ``intermediate_*`` scratch is NOT hashed: it is the state
    #             axis' equivalent of the freed tail, holding the last
    #             candidate's state whatever was accepted, and
    #             ``update_mamba_state_after_mtp_verify`` is what moves the
    #             accepted step out of it into the two buffers above.
    #
    # TWO READINGS come out of one record set, and the cheaper one needs no
    # reference job at all:
    #
    #   APPEND-ONLY (within a job). Every record also carries the digest of the
    #   prefix that was already committed BEFORE the round -- ``map_stable`` and
    #   ``kv_stable``. Those must equal the PREVIOUS record's ``map`` / ``kv``.
    #   A rejected candidate leaking into a row the lane had already committed
    #   breaks that equality at the round it happened, in one job, with no
    #   second run to align against. It is free: a second pass over the host
    #   copies the full-prefix digest already paid for, not a second D2H.
    #
    #   CROSS-JOB (spec against no-spec). ``kv`` / ``conv`` / ``ssm`` join on
    #   ``committed_len``, which both paths maintain identically. ``map`` does
    #   NOT join across jobs and is not meant to: it hashes physical slot ids,
    #   which the allocator hands out differently to two jobs even when both are
    #   correct.
    #
    #   Everything the cross-job side reads is addressed LOGICALLY, and since
    #   round 2 that is a structural property rather than an assurance. ``kv``
    #   is the digest OF THE PER-POSITION DIGEST LIST, in logical order:
    #   position p is hashed from the pool rows ``req_to_token[idx, p]`` points
    #   at, so two jobs whose allocators numbered the same content differently
    #   produce the same ``kv``, and a permutation of the same rows is
    #   distinguishable from a change of content (the multiset of per-position
    #   digests survives the first and not the second). ``conv`` / ``ssm`` hash
    #   the STATE CONTENT at the request's mamba slot, never the slot id.
    #
    #   The byte digests are the wrong instrument for the cross-job reading on a
    #   stack whose forwards are not bitwise reproducible run to run, and this
    #   window measured that rather than assuming it: in the 2026-08-02 STEPS=3
    #   window, two no-spec REFERENCE jobs on the same prompt disagreed on
    #   100 % of committed positions at round 0 -- the prefill, before any
    #   speculation exists -- with disjoint per-position digest sets, while
    #   their emitted tokens were identical. A hash has no tolerance, so a
    #   last-mantissa-bit difference reads exactly like a leaked row.
    #   ``kv_num`` / ``conv_num`` / ``ssm_num`` are the reading that survives
    #   that: per position (and per state surface) a float32 SUM and ABSMAX of
    #   the same bytes, so the reader can compare against a noise floor it
    #   MEASURED from two reference draws instead of against zero. The hashes
    #   stay -- they are exact within a job, which is where the append-only
    #   reading lives -- and the numeric fields are what joins across jobs.
    #
    # COST, stated rather than implied. One D2H of the committed prefix per
    # round, per full-attention layer: at 24 layers x 8 kv heads x 128 dims of
    # fp8 that is ~48 KiB per committed token, so a 700-token position costs
    # ~33 MiB of device-to-host traffic in that one round and the traffic grows
    # linearly with the position. It is off by default and it must never be on
    # while a timing table is being produced. What it does NOT do is enter
    # ``round_ms``: every call site sits after the round's wall clock has been
    # taken, so the per-round timings a checksum run reports stay comparable to
    # a clean run's even though the job's total duration does not.
    #
    # ORDERING. The hash reads device memory the verify forward wrote, so on
    # the captured path it must land after the REPLAY, never inside a capture.
    # Both are enforced rather than argued: an event recorded on the lane's own
    # stream is synchronized before the first read, and a stream that is
    # capturing is refused outright. Capture happens once inside
    # ``init_cuda_graphs`` with no job in flight, so the second guard is a
    # belt-and-braces assertion of a structural property.
    def _pool_checksum_on(self) -> bool:
        return bool(os.environ.get("SGLANG_LANE_POOL_CHECKSUM"))

    @staticmethod
    def _pool_checksum_tail() -> int:
        """Width of the FREED TAIL the can-fail arm hashes instead (0 = off)."""
        try:
            return int(os.environ.get("SGLANG_LANE_POOL_CHECKSUM_TAIL") or 0)
        except ValueError:
            return 0

    @staticmethod
    def _pool_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().contiguous().cpu()

    @staticmethod
    def _pool_feed(h, t: torch.Tensor) -> None:
        """Feed a CPU tensor to ``h`` as its RAW bytes, shape and dtype first.

        Bytes, not a float reduction: the question is whether two runs wrote
        the same values, and a sum over a differently-ordered set of the same
        magnitudes can collide. Shape and dtype go in as well so a surface that
        changed size cannot hash equal to one that did not.
        """
        h.update(f"|{tuple(t.shape)}|{t.dtype}|".encode())
        flat = t.reshape(-1)
        try:
            raw = flat.view(torch.uint8)
        except RuntimeError:
            # A dtype torch declines to reinterpret (rare, and never on the
            # kv/state dtypes this lane uses). Widening is injective on
            # everything but NaN payloads, which no pool row carries.
            raw = flat.to(torch.float32).view(torch.uint8)
        h.update(raw.numpy().tobytes())

    @classmethod
    def _pool_digest(cls, *tensors) -> str:
        h = hashlib.blake2b(digest_size=16)
        for t in tensors:
            if t is None:
                h.update(b"|none|")
            else:
                cls._pool_feed(h, cls._pool_cpu(t))
        return h.hexdigest()

    def _pool_checksum_kv_pool(self):
        """The lane target's KV cache, via the allocator that owns it."""
        alloc = getattr(self.runner, "token_to_kv_pool_allocator", None)
        getter = getattr(alloc, "get_kvcache", None)
        return getter() if getter is not None else None

    @staticmethod
    def _pool_checksum_kv_layers(kv_pool) -> List[int]:
        """The layer ids ``get_key_buffer`` accepts on this pool.

        A hybrid GDN pool holds KV for the FULL-ATTENTION layers only and
        refuses any other id by name; a dense pool numbers its layers from
        ``start_layer``. Asking the pool rather than the model is what keeps
        this correct for both without a family switch.
        """
        mapping = getattr(kv_pool, "full_attention_layer_id_mapping", None)
        if mapping:
            return sorted(int(k) for k in mapping)
        start = int(getattr(kv_pool, "start_layer", 0) or 0)
        num = int(getattr(kv_pool, "layer_num", 0) or 0)
        return list(range(start, start + num))

    @staticmethod
    def _pool_position_stats(rows) -> List[List[float]]:
        """Per LOGICAL position, ``[sum, absmax]`` in float32 over all layers.

        The cross-job join, and the reason it exists is measured rather than
        supposed: a digest answers "the same bytes?", and two correct jobs on
        this stack do not produce the same bytes (see the block comment). Sum
        and absmax are the cheapest pair that separates the two failure modes a
        reader has to tell apart -- a last-bit difference moves the sum by
        ~1e-3 of itself and the absmax by nothing, a leaked row moves both by
        O(1) -- and they are computed from the SAME host copies the digests
        were, so the cost is arithmetic and not a second D2H.
        """
        if not rows:
            return []
        positions = int(rows[0][0].shape[0])
        total = torch.zeros(positions, dtype=torch.float32)
        peak = torch.zeros(positions, dtype=torch.float32)
        for k, v in rows:
            for t in (k, v):
                flat = t.reshape(positions, -1).to(torch.float32)
                total += flat.sum(dim=1)
                peak = torch.maximum(peak, flat.abs().amax(dim=1))
        return [
            [round(float(s), 6), round(float(m), 6)]
            for s, m in zip(total.tolist(), peak.tolist())
        ]

    def _pool_checksum_kv(self, kv_pool, index, stable: int, per_pos: bool):
        """Digests of the KV rows ``index`` points at, in LOGICAL order.

        ``index`` is ``req_to_token[idx, lo:hi]``, so entry ``p`` is the slot
        holding logical position ``lo + p`` and every digest below is addressed
        by position, never by slot id. The aggregate is the digest OF THE
        PER-POSITION DIGESTS rather than of the concatenated rows: same
        sensitivity, and it makes "the aggregate cannot depend on which slots
        the allocator drew" a property of the construction instead of a
        sentence in a comment.

        Returns ``(digest, stable_digest, per_position, per_position_stats)``.
        ``stable_digest`` covers only the first ``stable`` positions -- the
        ones that were ALREADY committed before this round -- and it costs
        nothing extra: it is a second pass over the same host copies, not a
        second D2H. That second digest is what makes the probe self-sufficient.
        Comparing it against the PREVIOUS round's full digest asks the
        append-only question directly -- did anything this round wrote or freed
        change a row the lane had already committed? -- and answers it without
        a reference job, at the round it happened.
        """
        layers = self._pool_checksum_kv_layers(kv_pool)
        rows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_id in layers:
            rows.append(
                (
                    self._pool_cpu(kv_pool.get_key_buffer(layer_id)[index]),
                    self._pool_cpu(kv_pool.get_value_buffer(layer_id)[index]),
                )
            )
        if not rows:
            return None, None, None, None
        per: List[str] = []
        for p in range(int(index.shape[0])):
            hp = hashlib.blake2b(digest_size=8)
            for k, v in rows:
                self._pool_feed(hp, k[p])
                self._pool_feed(hp, v[p])
            per.append(hp.hexdigest())
        h = hashlib.blake2b(digest_size=16)
        h.update("|".join(per).encode())
        hs = hashlib.blake2b(digest_size=16)
        hs.update("|".join(per[:stable]).encode())
        digest = h.hexdigest()
        stable_digest = hs.hexdigest() if stable > 0 else None
        stats = self._pool_position_stats(rows)
        return digest, stable_digest, (per if per_pos else None), stats

    @staticmethod
    def _pool_state_stats(*tensors) -> List[float]:
        """``[sum, absmax]`` of a state surface, in float32.

        The state's cross-job join, for the same reason the KV positions have
        one. A slot id never enters it: the caller has already selected the
        request's slot, so what is reduced here is content.
        """
        total = 0.0
        peak = 0.0
        for t in tensors:
            flat = t.detach().reshape(-1).to(torch.float32)
            total += float(flat.sum().item())
            peak = max(peak, float(flat.abs().max().item()) if flat.numel() else 0.0)
        return [round(total, 6), round(peak, 6)]

    def _pool_checksum_state(self, job):
        """The request's persistent state, by CONTENT and not by slot.

        Returns ``(conv digest, ssm digest, conv stats, ssm stats)``.
        ``cache.conv`` is a per-layer list of ``[num_layers, num_slots, ...]``
        tensors and ``cache.temporal`` is one of the same shape, so ``[:, slot]``
        selects this request's state across every layer -- the state itself,
        which two jobs standing at the same committed position must agree on
        however differently the pool numbered their slots.
        """
        req = job.get("_req")
        pool = getattr(self.runner, "req_to_token_pool", None)
        mamba_pool = getattr(pool, "mamba_pool", None)
        cache = getattr(mamba_pool, "mamba_cache", None)
        slot = getattr(req, "mamba_pool_idx", None)
        if cache is None or slot is None:
            # A family without recurrence has no such surface, and saying so
            # with None is not the same as saying it hashed empty.
            return None, None, None, None
        slot = int(slot)
        conv_rows = [c[:, slot] for c in cache.conv]
        ssm_rows = [cache.temporal[:, slot]]
        conv = self._pool_digest(*conv_rows)
        ssm = self._pool_digest(*ssm_rows)
        return (
            conv,
            ssm,
            self._pool_state_stats(*conv_rows),
            self._pool_state_stats(*ssm_rows),
        )

    @staticmethod
    def _pool_committed_len(job) -> int:
        """How many positions this job has COMMITTED, on either path.

        ``_kv_len`` is the speculative path's own counter; a non-speculative
        job never sets it, and for that side the count is the prompt plus one
        per decode step (the last sampled token was never written to the pool
        -- the same arithmetic ``_finish`` frees by). The two agree at every
        position on a speculative job, which is what makes the field usable as
        the JOIN KEY between a spec and a no-spec run.
        """
        n = job.get("_kv_len")
        if n is not None:
            return int(n)
        return len(job["input_ids"]) + max(0, len(job["output_ids"]) - 1)

    def _pool_checksum_path(self) -> Optional[str]:
        """Where this lane/rank writes its jsonl, or None.

        The env value is a PREFIX, not a filename: under TP every rank runs
        this code with the same environment, and one shared file would
        interleave lines from processes that are not even at the same round.
        """
        prefix = os.environ.get("SGLANG_LANE_POOL_CHECKSUM_PATH")
        if not prefix:
            return None
        rank = int(getattr(self.runner, "tp_rank", 0) or 0)
        return f"{prefix}.lane{self.lane_id}.rank{rank}.jsonl"

    def _record_pool_checksum(self, job, *, path: str, n_accept=None) -> None:
        """One record per committed round. Never raises into the lane."""
        if not self._pool_checksum_on():
            return
        capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        try:
            if capturing is not None and capturing():
                return
        except Exception:
            pass
        t0 = time.perf_counter()
        try:
            if self.stream is not None:
                # AFTER the replay: an event on the lane's own stream, waited
                # on before the first device read. The forward helpers already
                # synchronize this stream, so in practice this returns at once
                # -- it is here so the ordering is a property of the probe and
                # not of what happens to call it.
                event = torch.cuda.Event()
                event.record(self.stream)
                event.synchronize()
            n_committed = self._pool_committed_len(job)
            batch = job.get("_batch")
            idx = job.get("_req_pool_idx")
            if batch is None or idx is None or n_committed <= 0:
                return
            row = batch.req_to_token_pool.req_to_token[int(idx)]
            tail = self._pool_checksum_tail()
            if tail > 0:
                lo = n_committed
                hi = min(int(row.shape[0]), n_committed + tail)
            else:
                lo, hi = 0, n_committed
            slots = row[lo:hi]
            # The prefix this round INHERITED. Under the freed-tail can-fail
            # arm there is no such prefix, so the append-only digests are
            # withheld rather than computed over a region they do not describe.
            prev_len = int(job.get("_pool_prev_len") or 0)
            stable = 0 if tail > 0 else max(0, min(prev_len, hi - lo))
            per_pos = bool(os.environ.get("SGLANG_LANE_POOL_CHECKSUM_PER_POS"))
            kv_pool = self._pool_checksum_kv_pool()
            kv_digest = kv_stable = kv_positions = kv_stats = None
            if kv_pool is not None and hi > lo:
                kv_digest, kv_stable, kv_positions, kv_stats = self._pool_checksum_kv(
                    kv_pool, slots.to(torch.int64), stable, per_pos
                )
            conv, ssm, conv_stats, ssm_stats = self._pool_checksum_state(job)
            records = job.setdefault("_pool_checksums", [])
            rec: Dict[str, Any] = {
                "tag": job.get("probe_tag"),
                "lane": self.lane_id,
                "rank": int(getattr(self.runner, "tp_rank", 0) or 0),
                "round": len(records),
                "path": path,
                "n_accept": n_accept,
                "rung": int(job.get("_rung") or 0) if path == "spec" else 0,
                "committed_len": n_committed,
                "prev_committed_len": prev_len,
                "kv_len": job.get("_kv_len"),
                "emitted": len(job.get("output_ids") or []),
                "region": "freed_tail" if tail > 0 else "committed_prefix",
                "map": self._pool_digest(slots),
                "map_stable": (
                    self._pool_digest(slots[:stable]) if stable > 0 else None
                ),
                "kv": kv_digest,
                "kv_stable": kv_stable,
                "conv": conv,
                "ssm": ssm,
                # The cross-job fields. Always present when their surface is,
                # because the reading that joins two jobs is the numeric one
                # and a diagnostic that needs a second env var to be usable is
                # a diagnostic that will be run without it.
                "kv_num": kv_stats,
                "conv_num": conv_stats,
                "ssm_num": ssm_stats,
            }
            if kv_positions is not None:
                rec["kv_pos"] = kv_positions
            rec["probe_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            job["_pool_prev_len"] = n_committed
            records.append(rec)
            out_path = self._pool_checksum_path()
            if out_path:
                with open(out_path, "a") as handle:
                    handle.write(json.dumps(rec) + "\n")
        except Exception:
            # A diagnostic that can kill the job it is diagnosing is worse than
            # no diagnostic; the failure is loud in the log and the round goes
            # on. Once, so a per-round fault cannot flood the boot.
            if not getattr(self, "_pool_checksum_failed", False):
                self._pool_checksum_failed = True
                logger.exception(
                    "dual-group lane %d: the pool checksum probe failed; it is "
                    "disabled for the rest of this process.",
                    self.lane_id,
                )
            os.environ.pop("SGLANG_LANE_POOL_CHECKSUM", None)

    def _timed_forward_raw(self, batch, capture_mode=None):
        """A lane forward that returns the LOGITS OUTPUT instead of sampled
        ids -- the verify path needs the per-candidate argmax and the hidden
        states, not one sampled token."""
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        runner = self.runner
        if capture_mode is not None:
            batch.capture_hidden_mode = capture_mode
        t0 = time.perf_counter()
        # This path RETURNS wall time -- every caller uses it as the round's
        # own timing and the recorded tables are wall -- so the device clock
        # gets its own event pair rather than the returned number. Missing
        # this cost the r9 boot the occupancy of every SPECULATIVE arm: the
        # verify runs here, the plain decode runs in _timed_forward, and only
        # the second one was feeding the clock, so a speculative window
        # reported the head's forwards as if they were the whole lane.
        start_ev = end_ev = None
        if self.stream is None:
            torch.cuda.synchronize(runner.gpu_id)
        else:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record(self.stream)
        fb = ForwardBatch.init_new(batch, runner)
        if capture_mode is not None:
            fb.capture_hidden_mode = capture_mode
        self._last_fb = fb
        out = runner.forward(fb).logits_output
        if self.stream is None:
            torch.cuda.synchronize(runner.gpu_id)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            self.device_clock.add_device_ms(wall_ms)
            return out, wall_ms
        end_ev.record(self.stream)
        self._submitted.set()
        self.stream.synchronize()
        self.device_clock.add_device_ms(start_ev.elapsed_time(end_ev))
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
            self._last_margin = (
                self._top2_margin(logits_output.next_token_logits)
                if self._margin_probe_on()
                else None
            )
            torch.cuda.synchronize(runner.gpu_id)
            ms = (time.perf_counter() - t0) * 1000.0
            self.device_clock.add_device_ms(ms)
            return next_token_ids, ms

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start_ev.record(self.stream)
        forward_batch = ForwardBatch.init_new(batch, runner)
        logits_output = runner.forward(forward_batch).logits_output
        next_token_ids = runner.sample(logits_output, forward_batch)
        self._last_margin = (
            self._top2_margin(logits_output.next_token_logits)
            if self._margin_probe_on()
            else None
        )
        end_ev.record(self.stream)
        self._submitted.set()
        self.stream.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        self._last_wall_ms = wall_ms
        # Device time on the lane's stream: what the lane's own kernels took,
        # including the SM share it lost to the serving group but excluding
        # any time spent waiting for the GIL between launches.
        device_ms = start_ev.elapsed_time(end_ev)
        self.device_clock.add_device_ms(device_ms)
        return next_token_ids, device_ms

    # -- one-shot row/position diagnosis (env-gated, off by default) -------

    def _dbg_on(self) -> bool:
        return bool(os.environ.get("SGLANG_LANE_SPEC_DEBUG"))

    def _dbg(self, tag: str, payload: dict) -> None:
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

    # -- the row oracle: WHICH sublayer breaks rows >= 1 (round 5) ---------

    def _lane_decoder_layers(self):
        """The target's decoder layers, in layer_id order.

        Named-module walk rather than an attribute path: the lane's target is
        an assembled hull whose container attribute differs per model family,
        and a wrong path here would silently yield an empty hook set (a
        diagnostic that always says "no divergence" is worse than none).
        """
        import re

        found = {}
        for name, mod in self.runner.model.named_modules():
            m = re.fullmatch(r"(?:.*\.)?layers\.(\d+)", name)
            if m:
                found[int(m.group(1))] = mod
        if not found:
            raise ValueError(
                "dual-group lane row oracle: no `layers.<i>` modules under the "
                "lane target; the oracle cannot attribute a divergence to a "
                "layer."
            )
        return [found[i] for i in sorted(found)]

    def _snapshot_recurrent(self):
        """Clone the lane's persistent conv + SSM state (all layers, all slots).

        The verify forward is state-PURE up to the commit, so a snapshot taken
        here and restored afterwards makes the two oracle arms start from
        literally the same recurrent state -- which is the whole point: any
        row difference is then the forward's, not the state's.
        """
        cache = self.runner.req_to_token_pool.mamba_pool.mamba_cache
        return (
            [c.clone() for c in cache.conv],
            cache.temporal.clone(),
        )

    def _restore_recurrent(self, snap):
        cache = self.runner.req_to_token_pool.mamba_pool.mamba_cache
        conv, temporal = snap
        for dst, src in zip(cache.conv, conv):
            dst.copy_(src)
        cache.temporal.copy_(temporal)

    def _dbg_row_oracle(self, job, proposals, n_cached):
        """Per-LAYER comparison of the TARGET_VERIFY rows against a decode oracle.

        Round 4 proved WHERE the fault is not (verify input, candidate KV
        slots, state commit -- all green with the accept cap at 0) and left one
        question: inside one verify forward, is the chain across draft steps
        broken by the FULL-ATTENTION mask or by the GDN recurrence? Both would
        look the same from outside: row 0 right, rows >= 1 progressively less
        input-dependent.

        This separates them in ONE round, on the rig, without a second boot.
        The verify forward is state-pure until the commit, so:

        1. snapshot conv/SSM, run the TARGET_VERIFY forward with a hook on
           every decoder layer, restore the snapshot and free the KV slots;
        2. run the same candidates as D plain DECODEs -- the known-good
           continuation, one row per step -- with the same hooks;
        3. compare layer by layer, row i against step i.

        Row 0 is byte-green under the cap, so ITS per-layer difference is the
        numeric floor of the comparison (two different kernels, same math).
        The first layer where row 1 leaves that floor is the culprit, and its
        membership in ``attn_backend.full_attn_layers`` is the answer:
        full-attn layer -> the mask; linear layer -> the GDN chain.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        batch = job["_batch"]
        req = job["_req"]
        idx = job["_req_pool_idx"]
        device = batch.device
        cand = [int(job["_next"][0].item())] + list(proposals)
        d = len(cand)

        layers = self._lane_decoder_layers()
        full_attn = set(getattr(self.runner.attn_backend, "full_attn_layers", []) or [])
        rec: List[List[torch.Tensor]] = [[] for _ in layers]
        rec_in: List[List[torch.Tensor]] = [[] for _ in layers]

        def _first_2d(args):
            for t in args:
                if (
                    isinstance(t, torch.Tensor)
                    and t.dim() == 2
                    and t.is_floating_point()
                ):
                    return t
            return None

        def _mk_hook(i):
            def _hook(_mod, inp, kwargs, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor):
                    rec[i].append(h.detach().float().cpu())
                # The decoder layer takes its hidden states positionally on
                # some model families and by keyword on others; scan both, or
                # the layer-input column silently reads None.
                hin = _first_2d(list(inp) + list(kwargs.values()))
                if hin is not None:
                    rec_in[i].append(hin.detach().float().cpu())

            return _hook

        handles = [
            mod.register_forward_hook(_mk_hook(i), with_kwargs=True)
            for i, mod in enumerate(layers)
        ]

        # Sub-module resolution inside the FIRST divergent layer: "layer 0 is
        # wrong" names a layer, not a kernel. Hooking every named child of
        # layer 0 turns the verdict into a module name in execution order.
        sub_names = [n for n, _ in layers[0].named_modules() if n]
        sub_rec: Dict[str, List[torch.Tensor]] = {n: [] for n in sub_names}

        def _mk_sub_hook(name):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) and out else out
                if isinstance(t, torch.Tensor):
                    sub_rec[name].append(t.detach().float().cpu())

            return _hook

        handles += [
            mod.register_forward_hook(_mk_sub_hook(n))
            for n, mod in layers[0].named_modules()
            if n
        ]

        # Inside the first GDN layer: the conv chain and the recurrent scan are
        # two separate kernels, and "layer 0 is wrong" does not say which. Spy
        # on both, test-side (module attribute + bound method), reverted below
        # -- no probe is left in the hot path.
        import sglang.srt.layers.attention.linear.gdn_backend as _gdnb

        conv_rec: List[torch.Tensor] = []
        attn_rec: List[torch.Tensor] = []
        _orig_conv = _gdnb.causal_conv1d_update
        _lin = getattr(self.runner.attn_backend, "linear_attn_backend", None)
        _orig_fx = getattr(_lin, "forward_extend", None)
        _orig_fd = getattr(_lin, "forward_decode", None)

        def _conv_spy(*a, **kw):
            out = _orig_conv(*a, **kw)
            conv_rec.append(
                {
                    "x": a[0].detach().float().cpu(),
                    "out": out.detach().float().cpu(),
                }
            )
            return out

        def _mk_attn_spy(fn):
            def _spy(*a, **kw):
                out = fn(*a, **kw)
                if isinstance(out, torch.Tensor):
                    attn_rec.append(
                        {
                            "a": kw["a"].detach().float().cpu(),
                            "b": kw["b"].detach().float().cpu(),
                            "out": out.detach().float().cpu(),
                        }
                    )
                return out

            return _spy

        _gdnb.causal_conv1d_update = _conv_spy
        if _orig_fx is not None:
            _lin.forward_extend = _mk_attn_spy(_orig_fx)
        if _orig_fd is not None:
            _lin.forward_decode = _mk_attn_spy(_orig_fd)
        # Module hooks do not fire inside a replayed cuda graph, and the lane's
        # DECODE is captured -- so the decode arm would record nothing and the
        # oracle would report "no divergence" for the wrong reason. Force both
        # arms eager for the duration; lane-local, restored below.
        saved_graph_runner = self.runner.decode_cuda_graph_runner
        self.runner.decode_cuda_graph_runner = None

        # Everything the two arms touch, saved before either runs.
        snap_state = self._snapshot_recurrent()
        saved = {
            "seq_lens": batch.seq_lens.clone(),
            "seq_lens_cpu": batch.seq_lens_cpu.clone(),
            "orig_seq_lens": batch.orig_seq_lens.clone(),
            "seq_lens_sum": batch.seq_lens_sum,
            "input_ids": batch.input_ids,
            "decode_batch_idx": req.decode_batch_idx,
            "kv_committed_len": req.kv_committed_len,
            "kv_allocated_len": req.kv_allocated_len,
            "already_computed": getattr(req, "already_computed", None),
            "row": batch.req_to_token_pool.req_to_token[
                idx, n_cached : n_cached + d
            ].clone(),
        }

        def _restore():
            batch.seq_lens.copy_(saved["seq_lens"])
            batch.seq_lens_cpu.copy_(saved["seq_lens_cpu"])
            batch.orig_seq_lens.copy_(saved["orig_seq_lens"])
            batch.seq_lens_sum = saved["seq_lens_sum"]
            batch.input_ids = saved["input_ids"]
            req.decode_batch_idx = saved["decode_batch_idx"]
            req.kv_committed_len = saved["kv_committed_len"]
            req.kv_allocated_len = saved["kv_allocated_len"]
            if saved["already_computed"] is not None:
                req.already_computed = saved["already_computed"]
            batch.req_to_token_pool.req_to_token[idx, n_cached : n_cached + d] = saved[
                "row"
            ]
            self._restore_recurrent(snap_state)

        def _take_per_forward():
            """Drain the kernel spies and clear them.

            The spies fire once per GDN LAYER per forward, so a flat list is
            layer-major within a forward. Draining after each forward is what
            makes ``[forward][layer]`` addressable -- reading the flat list as
            if it were step-major compares layer 1 against step 1, which is a
            different tensor entirely.
            """
            c, a_ = list(conv_rec), list(attn_rec)
            conv_rec.clear()
            attn_rec.clear()
            return c, a_

        def _take_rows():
            r_out = [list(x) for x in rec]
            r_in = [list(x) for x in rec_in]
            for x in rec:
                x.clear()
            for x in rec_in:
                x.clear()
            return r_out, r_in

        def _take_sub():
            s = {n: list(v) for n, v in sub_rec.items()}
            for v in sub_rec.values():
                v.clear()
            return s

        def _run_tv():
            out_cache_loc = batch.token_to_kv_pool_allocator.alloc(d)
            batch.req_to_token_pool.req_to_token[idx, n_cached : n_cached + d] = (
                out_cache_loc.to(torch.int32)
            )
            verify_input = build_lane_chain_verify_input(cand, n_cached, device=device)
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.spec_info = verify_input
            batch.input_ids = verify_input.draft_token
            batch.out_cache_loc = out_cache_loc
            batch.seq_lens_sum = n_cached
            try:
                out_tv, _ = self._timed_forward_raw(batch, CaptureHiddenMode.FULL)
                preds = [int(x) for x in self._verify_predictions(out_tv, d).tolist()]
            finally:
                batch.spec_info = None
                batch.forward_mode = ForwardMode.DECODE
            batch.token_to_kv_pool_allocator.free(out_cache_loc)
            return preds

        try:
            # -- arm A: the single TARGET_VERIFY forward ----------------------
            tv_preds = _run_tv()
            tv_rows, tv_in = _take_rows()
            tv_conv, tv_attn = _take_per_forward()
            tv_sub = _take_sub()
            _restore()

            # -- arm A2: the SAME forward again from the same restored state.
            # A verify that does not even reproduce itself is reading memory
            # nobody wrote -- a different bug from a wrong chain, and worth one
            # forward to exclude.
            tv2_preds = _run_tv()
            tv2_rows, _ = _take_rows()
            _take_per_forward()
            _take_sub()
            _restore()

            # -- arm B: the same candidates as D plain DECODEs ----------------
            sd_preds = []
            sd_conv, sd_attn, sd_sub = [], [], []
            for tok in cand:
                batch.input_ids = torch.tensor([tok], dtype=torch.int64, device=device)
                batch.prepare_for_decode()
                out_sd, _ = self._timed_forward_raw(batch, CaptureHiddenMode.LAST)
                sd_preds.append(int(out_sd.next_token_logits.argmax(dim=-1)[0].item()))
                c, a_ = _take_per_forward()
                sd_conv.append(c)
                sd_attn.append(a_)
                sd_sub.append(_take_sub())
            sd_rows, sd_in = _take_rows()
            surplus = batch.req_to_token_pool.req_to_token[idx, n_cached : n_cached + d]
            batch.token_to_kv_pool_allocator.free(surplus)
            _restore()
        finally:
            for h in handles:
                h.remove()
            self.runner.decode_cuda_graph_runner = saved_graph_runner
            _gdnb.causal_conv1d_update = _orig_conv
            if _orig_fx is not None:
                _lin.forward_extend = _orig_fx
            if _orig_fd is not None:
                _lin.forward_decode = _orig_fd

        dump_path = os.environ.get("SGLANG_LANE_SPEC_ORACLE_DUMP")
        if dump_path and not os.path.exists(dump_path):
            # Raw tensors to disk BEFORE any analysis: a bug in the comparison
            # below then costs a re-analysis, not a re-boot. Layers are capped
            # so the dump stays small.
            torch.save(
                {
                    "cand": cand,
                    "d": d,
                    "n_cached": n_cached,
                    "tv_preds": tv_preds,
                    "sd_preds": sd_preds,
                    "tv2_preds": tv2_preds,
                    "full_attn_layers": sorted(full_attn),
                    "tv_conv": tv_conv[:2],
                    "sd_conv": [f[:2] for f in sd_conv],
                    "tv_attn": tv_attn[:2],
                    "sd_attn": [f[:2] for f in sd_attn],
                    "tv_rows": [r[0] if r else None for r in tv_rows[:8]],
                    "tv2_rows": [r[0] if r else None for r in tv2_rows[:8]],
                    "sd_rows": [r[:d] for r in sd_rows[:8]],
                    "tv_in": [r[0] if r else None for r in tv_in[:8]],
                    "sd_in": [r[:d] for r in sd_in[:8]],
                    "tv_sub": tv_sub,
                    "sd_sub": sd_sub,
                },
                dump_path,
            )

        # -- the comparison ---------------------------------------------------
        def _rows(rel_a, rel_b):
            """Relative max-diff of row i in a against row i in b, per row."""
            per_row = []
            for i in range(d):
                x, y = rel_a(i), rel_b(i)
                if x is None or y is None or x.numel() != y.numel():
                    per_row.append(None)
                    continue
                denom = float(y.abs().max().item()) or 1.0
                per_row.append(round(float((x - y).abs().max().item()) / denom, 6))
            return per_row

        report = []
        for li in range(len(layers)):
            if not tv_rows[li] or len(sd_rows[li]) < d:
                continue
            tv_h = tv_rows[li][0]
            if tv_h.shape[0] != d:
                continue
            entry = {
                "layer": li,
                "kind": "full" if li in full_attn else "linear",
                "rel_maxdiff_per_row": _rows(
                    lambda i: tv_h[i].reshape(-1),
                    lambda i: sd_rows[li][i].reshape(-1),
                ),
            }
            if tv_in[li] and len(sd_in[li]) >= d and tv_in[li][0].shape[0] == d:
                tv_hi = tv_in[li][0]
                entry["in_rel_maxdiff_per_row"] = _rows(
                    lambda i: tv_hi[i].reshape(-1),
                    lambda i: sd_in[li][i].reshape(-1),
                )
            report.append(entry)

        # First GDN layer, split into its two kernels. TV runs ONE call whose
        # tensors carry D rows; the decode arm runs D calls of one row each.
        # The TV row split is by equal share of the flattened tensor rather
        # than by a named axis: the conv kernel's layout is [1, dim, D]
        # (token-minor) while the scan's is [1, D, HV, V] (token-major), and
        # only the conv case needs the transposed read.
        def _chunk(t, i):
            flat = t.reshape(-1)
            w = flat.numel() // d
            return flat[i * w : (i + 1) * w]

        def _conv_row(t, i):
            return t[0, :, i].reshape(-1) if t.dim() == 3 else _chunk(t, i)

        inside = {}
        if tv_conv and len(sd_conv) >= d and all(f for f in sd_conv[:d]):
            c = tv_conv[0]
            for key, field in (("conv_in", "x"), ("conv_out", "out")):
                inside[f"{key}_rel_maxdiff_per_row"] = _rows(
                    lambda i, f=field: _conv_row(c[f], i),
                    lambda i, f=field: sd_conv[i][0][f].reshape(-1),
                )
            inside["conv_shapes"] = [
                list(c["out"].shape),
                list(sd_conv[0][0]["out"].shape),
            ]
        if tv_attn and len(sd_attn) >= d and all(f for f in sd_attn[:d]):
            t = tv_attn[0]
            for field in ("a", "b", "out"):
                inside[f"ssm_{field}_rel_maxdiff_per_row"] = _rows(
                    lambda i, f=field: _chunk(t[f], i),
                    lambda i, f=field: sd_attn[i][0][f].reshape(-1),
                )
            inside["ssm_shapes"] = [
                list(t["out"].shape),
                list(sd_attn[0][0]["out"].shape),
            ]

        # Layer 0's children, in execution order: the first one whose row 1
        # leaves row 0's floor is the module that breaks the chain.
        submods = []
        for name in sub_names:
            tv_t = tv_sub.get(name) or []
            if not tv_t or any(not sd_sub[i].get(name) for i in range(d)):
                continue
            t = tv_t[0]
            per_row = _rows(
                lambda i, t=t: _chunk(t, i),
                lambda i, n=name: sd_sub[i][n][0].reshape(-1),
            )
            submods.append(
                {
                    "module": name,
                    "shapes": [list(t.shape), list(sd_sub[0][name][0].shape)],
                    "rel_maxdiff_per_row": per_row,
                }
            )

        # Self-repeat: TV against TV from the same restored state.
        selfrep = []
        if tv2_rows and tv_rows and tv_rows[0] and tv2_rows[0] is not None:
            t1, t2 = tv_rows[0][0], tv2_rows[0]
            selfrep = _rows(lambda i: t1[i].reshape(-1), lambda i: t2[i].reshape(-1))

        # The verdict: the first layer whose row 1 leaves row 0's floor by more
        # than 10x (and by more than 1e-3 absolute-relative, so a floor of
        # exactly 0 cannot manufacture a hit).
        culprit = None
        for e in report:
            r = e["rel_maxdiff_per_row"]
            if len(r) < 2 or r[0] is None or r[1] is None:
                continue
            if r[1] > max(10.0 * r[0], 1e-3):
                culprit = e
                break
        self._dbg(
            "row_oracle",
            {
                "n_cached": n_cached,
                "cand": cand,
                "proposals": list(proposals),
                "tv_preds": tv_preds,
                "tv2_preds": tv2_preds,
                "sd_preds": sd_preds,
                "tv_vs_tv2_layer0_per_row": selfrep,
                "n_layers": len(layers),
                "full_attn_layers": sorted(full_attn),
                "first_divergent_layer": culprit,
                "inside_first_gdn_layer": inside,
                "layer0_submodules": submods,
                "per_layer": report,
            },
        )

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

        THE DEFAULT. Since round 5 it is no longer the ONLY strategy that
        computes the right tokens -- ``target_verify`` does too -- but it is
        still the one with the most gate behind it (see ``_verify_mode``).
        Same accept rule, same emitted tokens; only the forward mode differs.

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
        row_margins: List[float] = []
        hidden = None
        for i, tok in enumerate(cand):
            batch.input_ids = torch.tensor(
                [tok], dtype=torch.int64, device=batch.device
            )
            batch.prepare_for_decode()
            out, ms = self._timed_forward_raw(batch, CaptureHiddenMode.LAST)
            total_ms += ms
            preds.append(int(out.next_token_logits.argmax(dim=-1)[0].item()))
            if self._margin_probe_on():
                row_margins.append(self._top2_margin(out.next_token_logits))
            hidden = out.hidden_states[-1:]
            if i >= len(proposals) or preds[i] != proposals[i]:
                break
            n_accept += 1

        emitted = list(proposals[:n_accept]) + [preds[n_accept]]
        # One margin per EMITTED token: verify row i is the forward that
        # decided emitted[i], so the two lists line up by construction and
        # the tail of row_margins (rows past the first rejection) is dropped.
        for m in row_margins[: len(emitted)]:
            self._record_margin(job, m)
        job["_kv_len"] = n_cached + n_accept + 1
        # Cloned for the reason spelled out in ``_verify_by_target_verify``:
        # this row outlives the forward that produced it by a whole round.
        job["_hidden"] = None if hidden is None else hidden.clone()
        job["_next"] = torch.tensor(
            [preds[n_accept]], dtype=torch.int64, device=batch.device
        )
        return emitted, n_accept, total_ms

    def _verify_state_buffers(self, draft_token_num: int):
        """The lane pool's per-step verify state caches, or a loud refusal.

        ``TARGET_VERIFY`` on a GDN hybrid is only correct because the backend
        parks one conv window + one SSM state PER DRAFT STEP and the accepted
        prefix is committed back afterwards. Those buffers exist only when the
        pool was built with ``speculative_num_draft_tokens``, and their step
        axis is exactly that wide. Checking both here turns two silent
        failures -- an ``AttributeError`` deep inside the kernel, and an
        out-of-bounds step index that would read another request's state --
        into one sentence at the call site.

        A model family WITHOUT recurrent state (dense Llama-class, MoE without
        a linear-attention branch) has nothing to roll back: rejecting a
        candidate only frees its KV slots, which the caller does anyway. Such
        a pool carries no ``mamba_pool`` at all, and that case returns None
        rather than refusing -- the refusal above is about a hybrid pool built
        without the speculative axis, not about the absence of recurrence.
        """
        from sglang.srt.mem_cache.memory_pool import MambaPool

        pool = self.runner.req_to_token_pool
        mamba_pool = getattr(pool, "mamba_pool", None)
        if mamba_pool is None:
            return None
        cache = getattr(mamba_pool, "mamba_cache", None)
        if not isinstance(cache, MambaPool.SpeculativeState):
            raise ValueError(
                "dual-group lane verify: the lane's mamba pool carries no "
                "MambaPool.SpeculativeState, so a TARGET_VERIFY forward would "
                "advance the recurrent state over rejected candidates with "
                "nothing to restore from. The pool is sized from "
                "server_args.max_speculative_num_draft_tokens; run with "
                f"SGLANG_LANE_SPEC_VERIFY=seqdecode instead (got {type(cache)})."
            )
        width = int(cache.intermediate_ssm.shape[2])
        if width < draft_token_num:
            raise ValueError(
                "dual-group lane verify: the lane pool's intermediate verify "
                f"state is {width} draft steps wide, but the lane proposes a "
                f"chain of {draft_token_num} candidates. Lower "
                "--dual-group-lane-spec-steps or raise the serving group's "
                "--speculative-num-draft-tokens (the pool is sized from it)."
            )
        return cache

    def _verify_predictions(self, out, draft_token_num: int):
        """One argmax per CANDIDATE, without paying for the lm_head twice.

        Under ``TARGET_VERIFY`` the logits processor selects no row at all
        (``sample_indices is None``), so ``next_token_logits`` is already
        ``[#CANDIDATE, vocab]`` -- the same tensor ``_candidate_logits`` would
        rebuild from the captured hidden states, one lm_head application later.
        The row count is CHECKED rather than assumed: contract 8 was exactly a
        silent row-selection mismatch (``[#SEQUENCE, vocab]`` read
        positionally), and it cost two rounds because nothing complained. If
        the selection rule ever changes, this falls back to the explicit
        reduction instead of indexing the wrong rows.

        Still open, and honestly still open: over 96 instrumented rounds the
        two sources disagreed on 13 argmaxes (7 on row 1, 3 on row 0, 3 on
        rows 2-3). Every one of those rounds also carried round 4's broken
        rows >= 1, so the disagreement may be nothing but two lm_head
        applications tying on a degenerate distribution. Round 5 fixed the rows
        but did NOT re-measure this: the instrument is the per-round debug
        trace, which adds an lm_head over every row and would have distorted
        the timing table that round 5 also had to produce. One debug job
        settles it. What round 5 does establish is that on the rows it did
        compare, the two sources agree -- the oracle's ``tv_preds`` come from
        ``next_token_logits`` and matched the decode oracle on all K+1 rows.
        """
        logits = getattr(out, "next_token_logits", None)
        if logits is not None and logits.shape[0] == draft_token_num:
            return logits.argmax(dim=-1)
        return self._candidate_logits(out.hidden_states).argmax(dim=-1)

    @contextlib.contextmanager
    def _verify_graph_scope(self, draft_token_num: int, job=None):
        """Run the verify forward under the lane's VERIFY graph entry.

        Yields True when the forward can replay a graph and False when it
        stays eager, so the round can report which of the two it paid for
        rather than leaving that to be inferred from a timing.

        The scope is the ONLY thing that puts the decode graph runner into
        verify shaping, and it is entered per forward rather than held open:
        the lane's plain decode step runs between two verify rounds on the same
        runner and must find it exactly as it was.
        """
        runner = getattr(self.runner, "decode_cuda_graph_runner", None)
        if (
            # Per-JOB falsifier, not a server flag: the replay-vs-eager byte
            # gate has to run both arms against the same weights, pools and
            # captures. Comparing two boots would put boot-to-boot variance
            # inside a gate whose whole purpose is to catch graph-replay
            # corruption (a known GGUF family, #52/#53).
            (job is not None and job.get("verify_graph") is False)
            or runner is None
            # Round 7a: the ladder makes this a membership test rather than an
            # equality one -- and it is tested against what was CAPTURED, not
            # against what was configured, so a rung that was thinned away or
            # whose capture raised falls back to eager instead of replaying a
            # neighbour's graph.
            or draft_token_num not in getattr(runner, "_lane_verify_captured", ())
        ):
            yield False
            return
        with runner.lane_verify_shape(draft_token_num):
            yield True

    def _verify_by_target_verify(self, job, proposals, n_cached):
        """Verify all K+1 candidates in ONE ``ForwardMode.TARGET_VERIFY``.

        COHERENT since round 5, and still not the default -- see the status at
        the bottom of this docstring for why those are two separate questions.

        This is the forward the batched EXTEND of round 1 was trying to be,
        minus the defect that made that one wrong: TARGET_VERIFY is the only
        mode that reaches ``GDNAttnBackend``'s ``if is_target_verify:`` arm,
        which parks the conv window and the SSM state of EVERY draft step in
        ``MambaPool.SpeculativeState`` instead of letting one continued extend
        drag the single running state over rejected candidates. After the
        accept rule has picked a prefix,
        ``update_mamba_state_after_mtp_verify`` scatters the state of the last
        ACCEPTED step back into the persistent caches -- so the recurrent
        state and the KV agree on the same prefix, which is exactly what the
        extend path could not do.

        Committing is not optional: the verify forward leaves the persistent
        conv/ssm caches holding the LAST candidate's state regardless of what
        was accepted. Skipping the commit would reproduce the round-1 defect
        with more machinery.

        ROUND-5 STATUS. Round 4 left this path byte-exact with the accept
        length capped at 0 and red without the cap: row 0 of ``preds`` tracked
        the no-spec continuation, rows >= 1 did not. The cause was NOT in this
        file and not in the verify input -- it was ``local_row_split`` handing
        the row-parallel shells a STRIDED view of the full-width activation,
        which GGUF's mat-VEC kernel (selected for <= 8 rows, i.e. exactly the
        K+1 rows of a verify) reads as if it were contiguous. Row 0 landed on
        the right bytes; every row after it did not. Localised by running this
        forward and a decode oracle from the same restored recurrent state and
        comparing layer by layer, then module by module: ``linear_attn`` exact
        on all rows, ``mlp.down_proj`` off by 100 % on rows 1-3.

        With the slices contiguous, uncapped ``target_verify`` is byte-green
        against the lane's own no-spec trajectory on all three gate prompts
        (floor measured first in the same boot), reproducible run to run, and
        its accept length is >= the bridge's on every prompt.

        ROUND-6 STATUS: this forward has its own graph capture now (a second
        decode-runner entry, ``lanetv``), and that is what finally separates
        the two modes. One TARGET_VERIFY forward went 68.4 -> 27.2 ms
        (2.5x) and a whole round 77.0 -> 35.9 ms, against 75-96 ms for the
        ``seqdecode`` bridge, which cannot be captured at all -- it is
        accept-many separate decode forwards, so its cost RISES with the accept
        length while this one is flat. On the mode question the answer is now
        unambiguous, byte-gated, and reproducible run to run.

        On the question of whether the lane should speculate at all it is
        still no, and the number says so plainly: 35.9 ms per round against
        16.1 ms per captured decode step is break-even at accept **2.22**
        (was 4.78), while the measured accept band tops out at 1.76 on the
        friendliest prompt. The remaining post is the head: K = 3 eager draft
        forwards are 8.6 ms of the 35.9. Capturing those (the head runs eager
        by a NAMED GAP -- see _finish_lane_draft_runner_scoped) would take the
        round to roughly 30 ms and break-even to ~1.9, which is STILL above the
        measured band. So the head capture is worth doing and is not by itself
        the thing that would make lane speculation pay here.

        The lever that DOES reach is the chain LENGTH, because the captured
        verify is ~12.5 ms fixed plus ~3.7 ms per candidate row (16.1 / 21.5 /
        27.2 ms at 1 / 2 / 4 rows): one extra row costs 0.23 of a decode step,
        so a shorter chain buys break-even directly. At K = 1 the round is
        24.8 ms and break-even 1.53, and on predictable content that is
        14.9 ms/token against 16.2 no-spec -- the first winning measurement
        this path has produced. It sits ON the threshold rather than above it
        (the same arm's floor run drew accept 1.43 and lost), so K = 1 is a
        knob for known workloads, not a new default.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        batch = job["_batch"]
        req = job["_req"]
        idx = job["_req_pool_idx"]
        device = batch.device
        cand = [int(job["_next"][0].item())] + list(proposals)
        d = len(cand)
        self._verify_state_buffers(d)

        # The candidates need KV slots BEFORE the forward: the attention plan
        # reads req_to_token[idx, n_cached : n_cached + d] to build its kv
        # indices (EagleVerifyInput.generate_attn_arg_prefill extends the
        # paged length by draft_token_num).
        page_size = int(self.runner.server_args.page_size or 1)
        if page_size != 1:
            raise ValueError(
                "dual-group lane verify: the flat candidate allocation below "
                f"assumes page_size 1, got {page_size}. A paged pool needs "
                "alloc_paged_token_slots_decode with the request's last slot, "
                "which this path does not build."
            )
        out_cache_loc = batch.token_to_kv_pool_allocator.alloc(d)
        if out_cache_loc is None:
            raise RuntimeError(
                f"dual-group lane verify: the lane pool cannot serve {d} "
                "candidate slots for one verify round."
            )
        batch.req_to_token_pool.req_to_token[idx, n_cached : n_cached + d] = (
            out_cache_loc.to(torch.int32)
        )

        verify_input = build_lane_chain_verify_input(cand, n_cached, device=device)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.input_ids = verify_input.draft_token
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = n_cached
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        try:
            with self._verify_graph_scope(d, job) as captured:
                out, ms = self._timed_forward_raw(batch, CaptureHiddenMode.FULL)
            job["_verify_graph"] = job.get("_verify_graph", 0) + int(captured)
            preds = self._verify_predictions(out, d)

            # The 13-of-96 argmax question from round 4, as a per-JOB switch
            # rather than the process-wide debug env: the check costs one extra
            # lm_head over all K+1 rows per round, which would otherwise land
            # in the same boot's timing table. Off by default; one job carries
            # it, every other job in the boot stays comparable.
            if job.get("argmax_check"):
                alt = self._candidate_logits(out.hidden_states).argmax(dim=-1)
                rows = job.setdefault("_argmax_rows", [0] * d)
                for i in range(d):
                    if int(alt[i].item()) != int(preds[i].item()):
                        rows[i] += 1
                job["_argmax_rounds"] = job.get("_argmax_rounds", 0) + 1

            n_accept = 0
            for i, prop in enumerate(proposals):
                if int(preds[i].item()) != prop:
                    break
                n_accept += 1
            # Falsifier knob (diagnostic, unset by default): capping the
            # accept length isolates the verify ROW that is under suspicion.
            # With the cap at 0 only row 0 is ever emitted and only step 0 is
            # ever committed -- so a run that is coherent under the cap and
            # incoherent without it puts the fault in rows/steps >= 1, and one
            # that is incoherent under the cap puts it in row 0 or the commit.
            cap = job.get("tv_max_accept")
            if cap is None:
                cap = os.environ.get("SGLANG_LANE_SPEC_TV_MAX_ACCEPT")
            if cap is not None:
                n_accept = min(n_accept, int(cap))
            # After the cap, so a falsifier arm records exactly the block it
            # emitted rather than the block it could have emitted.
            if self._margin_probe_on():
                vlogits = getattr(out, "next_token_logits", None)
                if vlogits is None or int(vlogits.shape[0]) != d:
                    # Same fallback ``_verify_predictions`` takes, so the
                    # margins are read off the very rows the argmaxes were.
                    vlogits = self._candidate_logits(out.hidden_states)
                self._record_verify_margins(job, vlogits, n_accept + 1)
            if self._dbg_on():
                self._dbg(
                    "tv_round",
                    {
                        "n_cached": n_cached,
                        "cand": cand,
                        "proposals": list(proposals),
                        "preds": [int(x) for x in preds.tolist()],
                        "n_accept": n_accept,
                        "hidden_rows": int(out.hidden_states.shape[0]),
                        "logits_rows": int(
                            getattr(out, "next_token_logits", preds).shape[0]
                        ),
                        "preds_from_hidden": [
                            int(x)
                            for x in self._candidate_logits(out.hidden_states)
                            .argmax(dim=-1)
                            .tolist()
                        ],
                    },
                )

            # Commit the recurrent state of the LAST ACCEPTED step. topk == 1
            # makes the tree a chain, so the accepted node's tree step is just
            # accept_len - 1 == n_accept. mamba_track_* stay None: the lane has
            # no radix cache, hence no prefix-cache tracking points.
            #
            # Families without recurrence never enter here: their backend has
            # no such method, and there is no state that the rejected steps
            # could have advanced (see _verify_state_buffers). The guard is on
            # the backend, not on a config flag, because the backend is what
            # owns the buffers.
            commit_state = getattr(
                self.runner.attn_backend, "update_mamba_state_after_mtp_verify", None
            )
            if commit_state is not None:
                commit_state(
                    last_correct_step_indices=torch.tensor(
                        [n_accept], dtype=torch.int64, device=device
                    ),
                    mamba_track_indices=None,
                    mamba_steps_to_track=None,
                    model=self.runner.model,
                )
        finally:
            batch.spec_info = None
            batch.forward_mode = ForwardMode.DECODE

        emitted = list(proposals[:n_accept]) + [int(preds[n_accept].item())]
        kept = n_accept + 1
        if kept < d:
            batch.token_to_kv_pool_allocator.free(out_cache_loc[kept:])

        # Bookkeeping that prepare_for_decode would otherwise do: the verify
        # consumed `kept` positions, and the next round starts from there.
        new_len = n_cached + kept
        batch.seq_lens.fill_(new_len)
        batch.seq_lens_cpu.fill_(new_len)
        batch.orig_seq_lens.fill_(new_len)
        batch.seq_lens_sum = None
        req.decode_batch_idx += kept
        req.kv_committed_len = new_len
        req.kv_allocated_len = new_len

        job["_kv_len"] = new_len
        # CLONED, for the same reason the row block below is, and the two are
        # deliberately no longer asymmetric. ``out.hidden_states`` is the
        # verify forward's output; under a captured replay it is a SLICE OF A
        # RETAINED STATIC BUFFER (the graph runner returns
        # ``output.hidden_states[: raw_num_token]`` of the tensor it kept from
        # the capture), and holding a reference to a slice of that buffer does
        # not stop the next replay of the same shape from writing through it.
        # This row is read a whole round later, by ``_propose``, with the
        # rollback's head forwards in between.
        #
        # What this clone does NOT do is change the committed tokens, and that
        # is worth stating where the temptation to assume otherwise is: the
        # only consumer of ``_hidden`` is ``_draft_forward``, so a value that
        # arrived corrupted here can move the PROPOSALS and nothing else. The
        # target's verify forward reads ``input_ids`` and the pool, never a
        # hidden state, and every committed token is ``preds[i]`` for
        # ``i <= n_accept``, whose conditioning tokens are by construction the
        # committed prefix. The one route from a proposal to a committed token
        # runs through what the REJECTED candidates leave behind in the pool
        # (their KV slots, the per-step recurrent intermediates, the attention
        # workspace) -- which is a property of the rollback, not of this row.
        job["_hidden"] = out.hidden_states[n_accept : n_accept + 1].clone()
        job["_next"] = preds[n_accept : n_accept + 1]
        # The whole candidate row block, for ``_rollback_draft``: the accepted
        # positions have to be re-run against the TARGET's hidden states, and
        # the full-accept catch-up needs the last candidate on top of that.
        # Only this path stashes them -- the seqdecode bridge produces no such
        # row block, and its rounds keep the old behaviour.
        #
        # CLONED, not viewed. ``out.hidden_states`` is the verify forward's
        # output; the re-seed below issues further forwards before it has read
        # every row, and a view that a later forward may write through is the
        # shared-buffer defect this branch has already paid for three times
        # (round D2's share_input_buffer pool, round D3's flashinfer
        # workspace). 4 rows of bf16 hidden is ~40 KiB per round.
        #
        # ``_verify_last_token`` and ``_verify_hidden`` are views into these
        # two clones and therefore carry the guarantee with them.
        if d >= 2:
            job["_verify_rows"] = out.hidden_states[: d - 1].clone()
            job["_verify_tokens"] = verify_input.draft_token[:d].clone()
            job["_verify_last_token"] = job["_verify_tokens"][d - 1 : d]
            job["_verify_hidden"] = job["_verify_rows"][d - 2 : d - 1]
        return emitted, n_accept, ms

    def _verify_mode(self, job) -> str:
        """Which verify strategy this round uses.

        ``seqdecode`` -- the correctness bridge of round 3 -- unless something
        explicitly asks for another one.

        Round 6 is where that stops being the right default, and the change is
        deliberately NOT made here: promoting a default is a merge decision,
        and this branch's job was to produce the evidence for it. The evidence
        is that captured ``target_verify`` now dominates the bridge on every
        measured axis -- 35.9 vs 75-96 ms per round, byte-identical to its own
        eager arm at the vehicle's reproducibility floor, byte-identical to the
        no-spec trajectory over the 12-token gate on all three prompts, accept
        length never below the bridge's -- and the bridge cannot be captured at
        all, because it IS accept-many separate forwards.
        ``target_verify`` is the default since R7b's merge (promotion evidence:
        R6 byte gates over three boots, R7a/R7b gates green, captured it
        dominates the bridge on every axis); ``seqdecode`` stays reachable as
        the fallback flag it has been.

        ``extend`` remains the measurably wrong batched path of round 1, kept
        reachable as a live falsifier. All three stay one explicit word away.
        """
        return (
            job.get("verify")
            or os.environ.get("SGLANG_LANE_SPEC_VERIFY")
            or "target_verify"
        )

    def _verify(self, job, proposals):
        """One verify forward of the LANE TARGET over [last, *proposals].

        Greedy accept: the target's output at candidate i predicts what
        should follow it, so proposal i+1 is accepted exactly when it equals
        that argmax. The first mismatch stops the chain and the target's own
        argmax there becomes the next token -- so a round always yields at
        least one token, which is what makes speculation free of correctness
        risk under greedy sampling.

        THE DISPATCHER for the three verify strategies, and itself the body of
        the WRONG one (``extend``), kept reachable so the falsifier keeps
        running. The reason it is wrong is a property of this target rather
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

        The two strategies that DO compute the right tokens:
        ``target_verify`` (the default, ``_verify_by_target_verify``) runs the
        same single forward in ``ForwardMode.TARGET_VERIFY``, which is the
        only mode that reaches the per-step state caches of ``MambaPool.
        SpeculativeState`` and can therefore restore the recurrent state to
        the accepted prefix; ``seqdecode`` (``_verify_by_decode``) sidesteps
        the problem by consuming the candidates one DECODE at a time, which is
        correct at the cost of one forward per emitted token.
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
        # Row oracle (round 5): its own env gate, because it is expensive
        # (one extra verify forward + D decodes) and, unlike the probes above,
        # it restores every piece of state it touches -- so the round that
        # follows it is NOT polluted and the run stays comparable. A fault in
        # the diagnosis never takes the lane down: it is reported and the round
        # proceeds (state is restored inside, before the analysis).
        oracle_rounds = os.environ.get("SGLANG_LANE_SPEC_ROW_ORACLE")
        if oracle_rounds and self._dbg_round <= int(oracle_rounds):
            try:
                self._dbg_row_oracle(job, proposals, n_cached)
            except Exception as exc:
                self._dbg("row_oracle_failed", {"error": repr(exc)})

        mode = self._verify_mode(job)
        if mode == "target_verify":
            return self._verify_by_target_verify(job, proposals, n_cached)
        if mode == "seqdecode":
            return self._verify_by_decode(job, proposals, n_cached)
        if mode != "extend":
            raise ValueError(
                f"dual-group lane verify: unknown strategy {mode!r} "
                "(target_verify | seqdecode | extend)."
            )

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
        # Cloned for the reason spelled out in ``_verify_by_target_verify``.
        job["_hidden"] = out.hidden_states[n_accept : n_accept + 1].clone()
        job["_next"] = preds[n_accept : n_accept + 1]
        return emitted, n_accept, ms

    def _prefill_chunk_size(self, job) -> int:
        """Chunk size (tokens) for THIS job's prefill; 0 means the single
        whole-prompt forward of slices A-D.

        The job's ``prefill_chunk`` wins over the lane flag
        (``--dual-group-lane-prefill-chunk``), including an explicit job 0
        that switches chunking OFF under a set flag. The DEFAULT is 0 on
        both: nothing about the existing path depends on this method.
        """
        explicit = job.get("prefill_chunk")
        if explicit is None:
            explicit = getattr(
                self.runner.server_args, "dual_group_lane_prefill_chunk", None
            )
        chunk = int(explicit or 0)
        return chunk if chunk > 0 else 0

    def _warn_off_ladder_chunk(self, chunk: int) -> None:
        """§13.10 point 2: the chunk size is not a free parameter.

        The lane's prefill forwards run against its captured prefill tier
        ladder; a chunk on a rung replays that rung's graph exactly, a chunk
        between rungs pads up to the next tier, and a chunk above the top
        tier falls back to eager. None of these are errors -- but the second
        and third quietly change what a "chunk" costs, so choosing an
        off-ladder size is worth one loud line, once per size.
        """
        if chunk in self._chunk_ladder_warned:
            return
        cfg = getattr(self.runner.server_args, "cuda_graph_config", None)
        tiers = getattr(getattr(cfg, "prefill", None), "bs", None)
        if not tiers:
            return
        self._chunk_ladder_warned.add(chunk)
        if chunk not in tiers:
            logger.warning(
                "dual-group lane %d: prefill chunk %d is not on the lane's "
                "prefill tier ladder %s -- chunks pad up to the next tier "
                "or run eager above the top one, so the measured ms/chunk "
                "will not be the rung's",
                self.lane_id,
                chunk,
                sorted(tiers),
            )

    def _prefill_chunked(self, job, chunk: int):
        """Chunked lane prefill (#274 D2-Posten 4, DESIGN_121 §13.10).

        The single whole-prompt forward becomes a loop of extend forwards
        over the SAME request: each chunk writes its KV (and, under spec,
        primes the NEXTN head over the same span), only the last chunk emits
        a token. ``work_total["prefill_tokens"]`` advances at the CHUNK
        boundary -- the finer grain is the declared yield of this posten
        (the pairing decider's saturation signal ages per forward).

        Reached only when a chunk size was configured by name; the default
        path is the untouched single-forward branch of ``_prefill``.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
        )

        input_ids = job["input_ids"]
        n = len(input_ids)
        spans = plan_prefill_chunks(0, n, chunk)
        # Tiling guard, before any forward: a plan that does not tile [0, n)
        # exactly would write some KV position twice or never, and nothing
        # downstream of here raises for that.
        pos = 0
        for start, end in spans:
            if start != pos or end <= start:
                raise RuntimeError(
                    f"lane prefill chunk plan does not tile the prompt: span "
                    f"({start}, {end}) at position {pos} of {n}"
                )
            pos = end
        if pos != n:
            raise RuntimeError(f"lane prefill chunk plan stops at {pos} of {n} tokens")

        spec_on = self._job_spec_on(job)
        # Loop-carried; the tiling guard above proves the loop runs at least
        # once, so every one of these is set by the time the tail reads it.
        req: Any = None
        req_d: Any = None
        batch: Any = None
        batch_d: Any = None
        prefix: Any = None
        prefix_d: Any = None
        out: Any = None
        next_token_ids: Any = None
        chunk_ms: List[float] = []
        for start, end in spans:
            last = end == n
            # -- target chunk: prefix must hold exactly [0, start) ---------
            if req is None:
                batch = self._make_batch(job)
            else:
                req.prefix_indices = prefix
                batch = self._make_batch(job, req=req)
            req = batch.reqs[0]
            req.set_extend_range(start, end)
            batch.prepare_for_extend()
            self._resolve_extend_input_ids(batch)
            if spec_on:
                out, ms = self._timed_forward_raw(batch, CaptureHiddenMode.FULL)
                if last:
                    next_token_ids = out.next_token_logits.argmax(dim=-1)
                if self._dbg_on():
                    self._dbg_prefill_rows(out, list(input_ids[start:end]))
            elif last:
                next_token_ids, ms = self._timed_forward(batch)
            else:
                # Middle chunk: KV writes are the product; the last-row
                # logits the runner computes anyway are discarded.
                _, ms = self._timed_forward_raw(batch)
            chunk_ms.append(ms)
            prefix = (
                batch.out_cache_loc
                if prefix is None
                else torch.cat((prefix, batch.out_cache_loc))
            )
            # -- head chunk (spec): same span, input shifted one left ------
            if spec_on:
                if req_d is None:
                    batch_d = self._make_batch(job, runner=self.draft_runner)
                else:
                    req_d.prefix_indices = prefix_d
                    batch_d = self._make_batch(job, runner=self.draft_runner, req=req_d)
                req_d = batch_d.reqs[0]
                req_d.set_extend_range(start, end)
                batch_d.prepare_for_extend()
                self._resolve_extend_input_ids(batch_d)
                ids = batch_d.input_ids
                # §13.10 point 3, the part with the real risk: the head's
                # rows for [start, end) consume tokens [start+1, end+1).
                # For a MIDDLE chunk the token at ``end`` is the PROMPT's
                # own -- the target's argmax there is a prediction the
                # prompt overrules, and priming with it would depress the
                # accept length for the same reason the unshifted feed in
                # the single-forward path would. Only the FINAL chunk has
                # no prompt token at ``end`` and takes the target's.
                if last:
                    tail = next_token_ids.to(ids.dtype).reshape(1)
                else:
                    tail = torch.tensor(
                        [int(input_ids[end])], dtype=ids.dtype, device=ids.device
                    )
                batch_d.input_ids = torch.cat((ids[1:], tail))
                self._draft_forward(batch_d, out.hidden_states)
                prefix_d = (
                    batch_d.out_cache_loc
                    if prefix_d is None
                    else torch.cat((prefix_d, batch_d.out_cache_loc))
                )
            # The counter's chunk-boundary advance: after this line the
            # tokens of THIS chunk are work done, visible to any reader
            # between two chunk forwards.
            self.work_total["prefill_tokens"] += end - start

        if spec_on:
            # Cloned for the reason spelled out in ``_verify_by_target_verify``
            # (same as the single-forward path).
            job["_hidden"] = out.hidden_states[-1:].clone()
            job["_kv_len"] = n
            job["_batch_d"] = batch_d
            job["_kv_len_draft"] = n
        # Sum of the chunk forwards' wall times -- the same quantity the
        # single-forward path records, at the same instrument (the head
        # primes are outside it there too). The per-chunk list is the
        # posten's own product: ms/chunk against chunk size is measurement
        # duty 4 of §13.10.
        job["prefill_ms"] = sum(chunk_ms)
        job["prefill_wall_ms"] = self._last_wall_ms
        job["prefill_chunk_ms"] = chunk_ms
        job["output_ids"].append(int(next_token_ids[0].item()))
        self._record_margin(job, self._last_margin)
        job["_batch"] = batch
        job["_next"] = next_token_ids
        job["_req"] = batch.reqs[0]
        job["_req_pool_idx"] = int(batch.reqs[0].req_pool_idx)
        # #404 anchor record, same position as the single-forward path.
        self._record_pool_checksum(job, path="prefill")

    @staticmethod
    def _resolve_extend_input_ids(batch) -> None:
        """H2D shim for deferred pinned input ids -- the chunked path's copy
        of the inline block in the single-forward branch."""
        if (
            batch.input_ids is None
            and getattr(batch, "prefill_input_ids_cpu", None) is not None
        ):
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None

    def _prefill(self, job):
        chunk = self._prefill_chunk_size(job)
        if chunk:
            self._warn_off_ladder_chunk(chunk)
            return self._prefill_chunked(job, chunk)
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
            # Cloned for the reason spelled out in
            # ``_verify_by_target_verify``, and here the further forward the
            # comment there talks about is two statements below: the head's
            # own prefill runs before anything reads this row.
            job["_hidden"] = out.hidden_states[-1:].clone()
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
        self.work_total["prefill_tokens"] += len(job["input_ids"])
        job["output_ids"].append(int(next_token_ids[0].item()))
        self._record_margin(job, self._last_margin)
        job["_batch"] = batch
        job["_next"] = next_token_ids
        # Captured NOW for the cleanup: the batch's req list is not stable
        # across the decode preparations.
        job["_req"] = batch.reqs[0]
        job["_req_pool_idx"] = int(batch.reqs[0].req_pool_idx)
        # #404 anchor record. Two jobs are only comparable from a position they
        # both reached identically; if the PROMPT already hashes differently
        # every downstream record is noise, and without this record that would
        # be read as a round-1 divergence.
        self._record_pool_checksum(job, path="prefill")

    def _choose_rung(self, job) -> int:
        """The chain length for the NEXT round of this job.

        The whole of the ladder's runtime side: everything below it is already
        captured, so this returns a number and nothing re-records. Per-job
        overrides are applied to the policy object rather than around it,
        because the policy is what R7b extends and a second decision site
        outside it would be a second thing to keep in step.
        """
        ctx: Dict[str, Any] = {}
        adaptive = job.get("adaptive")
        if adaptive is not None:
            self.spec_policy.adaptive = (
                bool(adaptive) and len(self.spec_policy.rungs) > 1
            )
        pin = job.get("spec_steps")
        if isinstance(pin, (list, tuple)):
            # #404: a per-round SCHEDULE, cycled. A scalar pin is constant for
            # the whole job, so a pinned job can never be MIXED-RUNG -- and the
            # mixed shape is the only one that reaches the READ side of the
            # ``_kv_len`` advance (a verify round taking ``n_cached`` right
            # after a K = 0 round, ``727bff334a``). Reaching it through the
            # adaptive policy instead would make the measurement depend on what
            # the policy happened to decide; ``[0, 1]`` reaches it on every
            # second round, in a fixed order, on any boot.
            if pin:
                index = len(job.get("_rungs") or [])
                ctx["rung"] = int(pin[index % len(pin)])
        elif pin is not None:
            ctx["rung"] = int(pin)
        return int(self.spec_policy.choose(ctx))

    def _spec_round(self, job):
        """One round of the lane's speculative decode, at the policy's rung.

        K = 0 is a rung like any other and is the lane's PLAIN decode step --
        the entry that has been byte-green since round 5 and that the ladder is
        explicitly not allowed to disturb. Routing K = 0 here rather than
        forcing a one-proposal chain is what makes "speculation off" reachable
        from inside a speculative job, which is what an adaptive policy needs
        in order to be able to lose gracefully.
        """
        rung = self._choose_rung(job)
        job.setdefault("_rungs", []).append(rung)
        if rung <= 0:
            self._decode_step(job)
            # The plain decode step is this boot's measurement of the K = 0
            # rung's cost -- the break-even denominator, measured rather than
            # taken from a table. The falsifier arms do not change this step
            # (it has no verify and no head forward), so it is always evidence.
            self.spec_policy.observe(0, job["decode_ms"][-1], 1)
            return
        job["_rung"] = rung
        self._spec_step(job)

    def _spec_step(self, job):
        """One speculative round on the lane: K proposals, one verify.

        Reported per ROUND, not per token: ms/round and accept length are the
        two numbers that decide whether speculation paid, and a per-token
        average hides both.
        """
        # ROUND wall time, not verify time. Rounds 1-3 reported the verify
        # alone, which silently left the head's K draft forwards -- the other
        # half of what speculation costs -- out of every ms/round and
        # ms/token figure on this path. The verify time is kept alongside so
        # the two structural posts (head, verify) stay separable.
        t0 = time.perf_counter()
        proposals = self._propose(job)
        emitted, n_accept, ms = self._verify(job, proposals)
        self._rollback_draft(job, n_accept)
        round_ms = (time.perf_counter() - t0) * 1000.0
        job["decode_ms"].append(round_ms)
        job.setdefault("_verify_ms", []).append(ms)
        job.setdefault("_propose_ms", []).append(round_ms - ms)
        job["output_ids"].extend(emitted)
        job.setdefault("_accept", []).append(n_accept + 1)
        # #404: the committed surfaces of THIS round, after the rollback has
        # given the rejected candidates' slots back and after ``round_ms`` was
        # taken -- so the probe's D2H stays out of every reported timing.
        self._record_pool_checksum(job, path="spec", n_accept=n_accept)
        # RAW per-position counters, next to the policy's EMA'd ones. The
        # policy EMAs because it decides from the number and old content must
        # stop voting; a falsifier wants the whole job weighted equally, and
        # the reference side (accept_position_probe) counts raw too.
        rung = int(job.get("_rung") or self.spec_steps)
        reached = job.setdefault("_pos_reached", {})
        hits = job.setdefault("_pos_hits", {})
        for j in range(rung):
            if j > n_accept:
                break
            reached[j] = reached.get(j, 0) + 1
            if j < n_accept:
                hits[j] = hits.get(j, 0) + 1
        # Feed the policy AFTER the round, with what the round actually cost
        # and produced. Both halves of the break-even come from here.
        #
        # A FALSIFIER round is not evidence, and leaving that out cost a
        # measurement: the byte gates run the same rungs with the graphs
        # switched off, an eager verify costs 68 ms against the captured
        # 21 ms, and those rounds landed in the per-rung cost EMA. The policy
        # then believed K=1 cost 77 ms per round and pinned itself to K=0 for
        # a reason that has nothing to do with the content (measured, round 7a
        # boot 6: round_ms {0: 16.1, 1: 77.1, 2: 75.2, 3: 52.7} against
        # measured graph costs of 24 / 28 / 34). The diagnostic arms are
        # deliberately not the operating point, so they must not price it.
        if job.get("verify_graph") is False or job.get("head_graph") is False:
            job["_policy_skipped"] = job.get("_policy_skipped", 0) + 1
        else:
            self.spec_policy.observe(
                int(job.get("_rung") or self.spec_steps), round_ms, n_accept + 1
            )
        # A speculative round emits accept+1 tokens, so the ARM is the same
        # decode arm -- what changes is how much work one round completes,
        # which is exactly what the rate is meant to capture.
        self.work_total["decode_tokens"] += len(emitted)

    def _rollback_draft(self, job, n_accept: int) -> None:
        """Put the head's own sequence back where the target's is.

        Round 7b posten 0. ``_propose`` advances the head by K positions per
        round; the verify commits ``n_accept + 1``. Nothing put the difference
        back, so from the second round on the head answered about a position
        the sequence was not at, over a KV cache holding the rejected
        proposals -- measured on this vehicle as a lag of up to 299 positions
        by round 164, growing every round. It had been written down as a
        bookkeeping note ("the head keeps every proposal it ever made") rather
        than recognised as a defect, which is why four rounds of measurements
        carried it.

        The rejected proposals occupy head KV slots ``kept .. K-1`` of this
        round; they are given back here rather than at job end, which is also
        what stops the head's pool filling K times faster than the target's.

        FULL ACCEPT is the one case the truncation cannot express on its own:
        the round then commits ``K + 1`` tokens while the head only ever ran K
        forwards, so the head is short exactly the bonus token's position. One
        extra head forward fills it -- input token ``cand[K]`` against the
        TARGET's hidden state of row ``K-1``, the same pairing every other step
        uses. It costs a head forward only on rounds where every proposal was
        accepted, i.e. exactly the rounds that paid for themselves.
        """
        batch_d = job.get("_batch_d")
        if batch_d is None or job.get("draft_rollback") is False:
            return
        start = int(job.get("_round_start") or 0)
        steps = int(job.get("_rung") or self.spec_steps)
        kept = int(n_accept) + 1
        idx = int(batch_d.reqs[0].req_pool_idx)
        pool = batch_d.req_to_token_pool

        rows = job.get("_verify_rows")
        tokens = job.get("_verify_tokens")
        reseed = (
            job.get("draft_reseed") is not False
            and kept >= 2
            and rows is not None
            and tokens is not None
            and rows.shape[0] >= kept - 1
            and tokens.shape[0] >= kept
        )
        if reseed:
            # Round 7c posten 2: re-run the ACCEPTED positions against the
            # target's hidden states instead of keeping the head's own.
            #
            # Positions ``start+1 .. start+kept-1`` were written by _propose
            # with the head's hidden as input, because that is all the head had
            # while it was speculating. The verify has since produced the
            # target's hidden for exactly those rows, and the serving group
            # uses it: ``EagleWorkerV2._draft_extend_for_decode`` re-runs the
            # head over the accepted block with ``logits_output.hidden_states``
            # of the target. Leaving the head's own there was the last
            # structural difference between the two chains.
            #
            # Sequential decode forwards rather than one batched extend: at
            # K = 3 this is 1-3 forwards, each replayable on the head graph
            # captured in round 7a (2.58 ms), while an extend over 1-3 rows
            # would have to run eager. The serving group pays one batched
            # extend because it carries a whole batch; the lane carries one
            # request.
            self._truncate_draft(job, batch_d, pool, idx, start + 1, start + steps)
            with self._head_graph_scope(job):
                for j in range(1, kept):
                    batch_d.input_ids = tokens[j : j + 1].to(torch.int64)
                    batch_d.prepare_for_decode()
                    self._draft_forward(batch_d, rows[j - 1 : j])
                    job["_head_forwards"] = job.get("_head_forwards", 0) + 1
                    job["_kv_len_draft"] = int(job.get("_kv_len_draft") or 0) + 1
                    job["_reseed_forwards"] = job.get("_reseed_forwards", 0) + 1
            return

        if kept > steps:
            # Full accept: run the missing position rather than leave a hole.
            hidden = job.get("_verify_hidden")
            token = job.get("_verify_last_token")
            if hidden is not None and token is not None:
                batch_d.input_ids = token.to(torch.int64)
                batch_d.prepare_for_decode()
                self._draft_forward(batch_d, hidden)
                job["_head_forwards"] = job.get("_head_forwards", 0) + 1
                job["_kv_len_draft"] = int(job.get("_kv_len_draft") or 0) + 1
            return

        surplus = steps - kept
        if surplus <= 0:
            return
        self._truncate_draft(job, batch_d, pool, idx, start + kept, start + steps)

    def _truncate_draft(self, job, batch_d, pool, idx, keep_len: int, end: int) -> None:
        """Cut the head's sequence back to ``keep_len`` and give the slots back.

        Both callers need the same five pieces of bookkeeping that
        ``prepare_for_decode`` would otherwise own (allocator, seq lens on both
        devices, the request's decode index and its two length fields); doing
        it in one place is what keeps the re-seed path and the plain rollback
        from drifting apart.
        """
        if end <= keep_len:
            return
        surplus = end - keep_len
        batch_d.token_to_kv_pool_allocator.free(pool.req_to_token[idx, keep_len:end])
        batch_d.seq_lens.fill_(keep_len)
        if getattr(batch_d, "seq_lens_cpu", None) is not None:
            batch_d.seq_lens_cpu.fill_(keep_len)
        if getattr(batch_d, "orig_seq_lens", None) is not None:
            batch_d.orig_seq_lens.fill_(keep_len)
        batch_d.seq_lens_sum = None
        req_d = batch_d.reqs[0]
        req_d.decode_batch_idx = max(0, int(req_d.decode_batch_idx) - surplus)
        req_d.kv_committed_len = keep_len
        req_d.kv_allocated_len = keep_len
        job["_kv_len_draft"] = int(job.get("_kv_len_draft") or 0) - surplus

    def _decode_step(self, job):
        batch = job["_batch"]
        batch.input_ids = job["_next"].to(torch.int64)
        batch.prepare_for_decode()
        next_token_ids, ms = self._timed_forward(batch)
        job["decode_ms"].append(ms)
        job["output_ids"].append(int(next_token_ids[0].item()))
        self._record_margin(job, self._last_margin)
        job["_next"] = next_token_ids
        # ``prepare_for_decode`` advanced the batch by one position, so
        # ``_kv_len`` has to follow it. It is not decoration: the K = 0 rung
        # of the ladder routes a SPECULATIVE job through this step, and
        # ``_verify`` takes ``n_cached = job["_kv_len"]`` as the length of the
        # committed prefix. Left behind, the next verify round would write its
        # candidate slot pointers over positions the plain step had already
        # committed, place the chain at absolute positions one short, and mask
        # a prefix that is one token too small -- a corruption of committed
        # content, not a lost proposal. The rung ladder is the only caller
        # that can reach it (a non-speculative job never reads the field), so
        # the pinned single-rung recipe does not exercise it.
        if job.get("_kv_len") is not None:
            job["_kv_len"] = int(job["_kv_len"]) + 1
        # #404: the same record the speculative round emits. This is the path a
        # NO-SPEC job travels end to end, so it is where the comparison target
        # comes from -- and it is also the K = 0 rung of a speculative job, so
        # a mixed-rung job's records interleave the two on one committed axis.
        self._record_pool_checksum(job, path="decode")
        self.work_total["decode_tokens"] += 1

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
            # Present only under SGLANG_LANE_MARGIN_PROBE=1. One entry per
            # committed token, aligned with output_ids, so r12's verdict.py
            # can read a lane flip against the lane's OWN perturbation band
            # instead of against stock's by analogy.
            **({"margins": job["_margins"]} if job.get("_margins") else {}),
            "prefill_ms": round(job["prefill_ms"], 2),
            "decode_ms_mean": (
                round(sum(decode_ms) / len(decode_ms), 3) if decode_ms else None
            ),
            "decode_steps": len(decode_ms),
        }
        # The two structural posts of a speculative round, reported apart:
        # a mean alone cannot say whether the head or the verify is what
        # costs, and both are eager for different reasons.
        for key, label in (
            ("_verify_ms", "verify_ms_mean"),
            ("_propose_ms", "propose_ms_mean"),
        ):
            xs = job.get(key) or []
            if xs:
                result[label] = round(sum(xs) / len(xs), 3)
        # How many of the rounds replayed the lane's captured verify graph.
        # Reported rather than assumed: a verify that silently fell back to
        # eager is the one way a timing table can look like a regression when
        # nothing regressed.
        if accepts and job.get("verify") in (None, "target_verify"):
            result["verify_graph_rounds"] = job.get("_verify_graph", 0)
        # Round 7a: the head's counterpart, plus which rungs the job actually
        # ran. The rung histogram is what makes an adaptive job's number
        # readable at all -- an adaptive ms/token is a mixture, and reporting
        # the mixture weights is the difference between a measurement and an
        # anecdote.
        if job.get("_head_forwards"):
            result["head_forwards"] = job["_head_forwards"]
            result["head_graph_forwards"] = job.get("_head_graph", 0)
            # Round 7c posten 2: how many of those forwards were the re-seed,
            # i.e. what the alignment with the serving group's chain COSTS.
            # It is zero on rounds that accepted nothing, so the price scales
            # with the accept length -- reporting it next to the round time is
            # what keeps that visible instead of buried in a mean.
            result["reseed_forwards"] = job.get("_reseed_forwards", 0)
        rungs = job.get("_rungs") or []
        if rungs:
            hist: Dict[int, int] = {}
            for k in rungs:
                hist[k] = hist.get(k, 0) + 1
            result["rungs"] = dict(sorted(hist.items()))
            result["rung_mean"] = round(sum(rungs) / len(rungs), 3)
            result["policy"] = self.spec_policy.stats()
        # Round 7b posten 0: the per-position curve in raw counts, and the
        # head's lag against the target. Both are what decides between "the
        # head is weak" and "the lane's chain feeds it the wrong position".
        reached = job.get("_pos_reached") or {}
        if reached:
            hits = job.get("_pos_hits") or {}
            result["accept_positions"] = {
                "reached": dict(sorted(reached.items())),
                "hits": dict(sorted(hits.items())),
                "rate": [
                    round(hits.get(j, 0) / reached[j], 5)
                    for j in range(max(reached) + 1)
                    if reached.get(j)
                ],
            }
        lags = job.get("_draft_lag") or []
        if lags:
            result["draft_lag"] = {
                "first": lags[0],
                "last": lags[-1],
                "max_abs": max(abs(x) for x in lags),
                "nonzero_rounds": sum(1 for x in lags if x != 0),
                "rounds": len(lags),
            }
        # #404: present only under SGLANG_LANE_POOL_CHECKSUM=1. Carried on the
        # result row as well as written to the jsonl so a harness that already
        # polls /get_server_info needs no file access on the server's host --
        # the two are the same records, not two instruments.
        checksums = job.get("_pool_checksums")
        if checksums:
            result["pool_checksums"] = checksums
            result["pool_checksum_rounds"] = len(checksums)
        if job.get("_argmax_rounds"):
            result["argmax_check"] = {
                "rounds": job["_argmax_rounds"],
                "rows_disagreeing": job.get("_argmax_rows"),
            }
        if job.get("prefill_wall_ms") is not None:
            result["prefill_wall_ms"] = round(job["prefill_wall_ms"], 2)
            result["prefill_wait_ms"] = round(
                job["prefill_wall_ms"] - job["prefill_ms"], 2
            )
        # §13.10: present only when the prefill ran chunked. ms/chunk against
        # chunk size is measurement duty 4 of the chunking posten, and the
        # harness reads it off the result row like everything else.
        if job.get("prefill_chunk_ms") is not None:
            result["prefill_chunks"] = len(job["prefill_chunk_ms"])
            result["prefill_chunk_ms"] = [round(x, 3) for x in job["prefill_chunk_ms"]]
        with self._lock:
            self.results.append(result)
            self.results_total += 1
            self.prefill_tokens_total += result["input_len"]
            self.decode_steps_total += result["decode_steps"]
            self.active = None
            if not self.jobs:
                self._busy = False
                self._idle_since = time.monotonic()
        self._zero_lane_workspaces()
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

    def _zero_lane_workspaces(self) -> None:
        """Restore the #50 flashinfer boot contract for THIS lane's
        workspaces, at this lane's job boundary.

        The serving group does this at request finish (Scheduler, guarded by
        SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST). Before #274 slice D3
        that call zeroed every registered workspace in the process, so a
        concurrent lane's scratch was wiped mid-forward by a foreign thread;
        the registry is keyed by lane now, which fixes the corruption and, on
        its own, would leave the lane with no zeroing at all. A lane job is
        the lane's request, so it is the right boundary: the counterpart runs
        here, on the lane's own thread, inside the lane scope, and reaches
        exactly this lane's bucket.

        A SERIAL lane keeps ``scope_lane_id is None`` and shares the serving
        group's workspaces deliberately; zeroing them at a lane job boundary
        would change the serving group's behaviour in a mode that is meant to
        be byte-for-byte slice B, so it is skipped there.
        """
        from sglang.srt.environ import envs

        if self.scope_lane_id is None:
            return
        if not envs.SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST.get():
            return
        fi_mod = sys.modules.get("sglang.srt.layers.attention.flashinfer_backend")
        if fi_mod is None:
            return
        try:
            with torch.cuda.stream(self.stream):
                fi_mod.zero_flashinfer_workspaces()
        except Exception:
            logger.exception(
                "dual-group lane %d: zeroing its flashinfer workspaces failed.",
                self.lane_id,
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

    # A lane whose parts sit on two cards runs EAGER, and this has to be
    # decided BEFORE the args view is taken: the view RESOLVES the disable
    # flags into the phase config, so a later flip reaches nothing and the
    # lane's own prefill graph still captures (measured -- the capture died
    # in cudaErrorStreamCaptureIsolation on the shells' first cross-card hop).
    # A CUDA graph is recorded on one stream of one device; a device-to-device
    # copy whose peer work is not on the capture stream is not a graphable
    # operation. Named in the log, because eager is a state the reader has to
    # know about.
    if _lane_spans_cards(server_args) and not server_args.dual_group_lane_eager:
        # Through override(), not a bare assignment: this runs inside
        # Scheduler.__init__ on the live, already-resolved server_args, which
        # is exactly the post-resolution mutation the config contract routes
        # through one audited entry point. Same resulting value, plus
        # provenance, and it no longer needs the strict guard to look away.
        server_args.override("dual_group_lane.spans_cards", dual_group_lane_eager=True)
        logger.warning(
            "dual-group lane spans cards (--dual-group-lane-part-gpu-id %s): "
            "forcing EAGER. The shells' cross-card activation hops are not "
            "capturable in one device's graph.",
            list(server_args.dual_group_lane_part_gpu_id or ()),
        )

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


def stop_dual_group_lanes(scheduler) -> list:
    """Stop every lane this module built for ``scheduler`` (#673).

    THE COUNTERPART THAT WAS MISSING. ``build_dual_group_lanes`` had no
    opposite number, so ``DualGroupLane.stop_worker`` sat with ZERO callers and
    every lane worker -- each owning a live ``torch.cuda.Stream`` -- was leaked
    at exit. The component keeps the knowledge of how to stop what it built;
    the caller only decides whether now is the time.

    Never raises and never stops early: one lane refusing to stop must not
    strand the lanes behind it in the list, because those are the ones still
    holding streams.
    """
    lanes = getattr(scheduler, "dual_group_lanes", None) or []
    outcomes = []
    for lane in lanes:
        try:
            outcomes.append(lane.stop_worker())
        except Exception as e:  # noqa: BLE001 - teardown must not raise
            logger.warning(
                "%s stopping lane %s failed: %s",
                DGL_LOG_PREFIX,
                getattr(lane, "lane_id", "?"),
                e,
            )
    return outcomes


def resolve_lane_spec_rungs(server_args) -> Tuple[int, ...]:
    """The lane's chain-length ladder as a sorted tuple of K values (#274 R7a).

    Unset ``--dual-group-lane-spec-rungs`` resolves to the single configured
    ``--dual-group-lane-spec-steps``, which is the pre-ladder shape down to the
    VRAM: one verify graph, one operating point. The ladder is opt-in because
    each additional rung is another graph pool on the lane's card, and that is
    a post the operator has to be able to decline.
    """
    from sglang.srt.model_executor.lane_spec_policy import parse_lane_spec_rungs

    rungs = parse_lane_spec_rungs(
        getattr(server_args, "dual_group_lane_spec_rungs", None)
    )
    if rungs:
        return rungs
    return (int(getattr(server_args, "dual_group_lane_spec_steps", 3)),)


def _enable_decode_graph_phase(args) -> None:
    """Re-enable ONLY the decode cuda-graph phase on a lane args view.

    The inverse of the ``_disable_graph_phases`` below, applied to the lane's
    NEXTN head after that call, and deliberately not a parameter of it: the
    head's PREFILL graph must stay off (its extend shapes follow the target's
    prompt and are not a fixed ladder), and a single function that could do
    either would make the difference easy to lose.

    The three legacy booleans are set as well as the phase backend, because
    ``check_cuda_graph_backend`` reads the phase config while other call sites
    still read ``disable_cuda_graph`` -- contract 4 in
    ``_finish_lane_draft_runner`` is exactly what happens when the two
    disagree.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    args.disable_cuda_graph = False
    args.disable_decode_cuda_graph = False
    args.disable_prefill_cuda_graph = True
    getattr(args.cuda_graph_config, Phase.DECODE).backend = Backend.FULL


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

    # #274 round 7a: and then the DECODE phase is given back, which is the
    # whole of the head-capture opening. Round 6 left the head eager for a
    # named reason -- the generic decode capture builds ``spec_info=None`` and
    # an MTP forward dereferences it -- so the capture had to be disabled
    # wholesale. The runner-side fix (``_lane_draft_spec_info``) removes that
    # reason, and this is where the phase is re-enabled for it. PREFILL stays
    # off: the head's extend shapes follow the target's prompt, they are not a
    # fixed ladder, and the head's prefill is one forward per JOB against K per
    # round -- the time is not there.
    head_graph = bool(
        getattr(draft_runner.server_args, "dual_group_lane_spec_head_graph", True)
    ) and not bool(getattr(draft_runner.server_args, "dual_group_lane_eager", False))
    if head_graph:
        _enable_decode_graph_phase(draft_runner.server_args)
        # The RUNNER-side flag is set inside the scoped bring-up below, not
        # here, and that is not a style choice: ``alloc_memory_pool`` re-inits
        # the block of graph-runner fields this one lives in (the same block
        # that holds ``dual_group_lane_verify_tokens``), and it runs inside
        # that call. Setting it here is silently undone -- measured, boot 1 of
        # round 7a, which reproduced round 6's named gap verbatim
        # (``'NoneType' object has no attribute 'hidden_states'``) because the
        # capture then saw the field as False. The lane TARGET already obeys
        # this ordering; the head now does too.
    else:
        logger.warning(
            "dual-group lane %d NEXTN head runs EAGER "
            "(--no-dual-group-lane-spec-head-graph or --dual-group-lane-eager).",
            lane_id,
        )

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
    # -- and round 7a is where that agreement starts EARNING something rather
    # than merely preventing a crash: with the head's decode phase enabled
    # above, both readers now see ENABLED, so the shared-output buffer is
    # created and the head's one decode graph is captured; prefill still sees
    # DISABLED on both sides and is still skipped. This is the same mechanism,
    # and the same reasoning, as the comment on the enclosing scope: helpers read
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
        _finish_lane_draft_runner_scoped(draft_runner, lane_id, head_graph=head_graph)

    logger.info(
        "dual-group lane %d NEXTN head ready: max_total_num_tokens=%d.",
        lane_id,
        draft_runner.max_total_num_tokens,
    )


def _finish_lane_draft_runner_scoped(draft_runner, lane_id, head_graph: bool = False):
    """Pools, backends and graphs of the head, under the head's args view."""
    with lane_geometry_override(1, 0):
        draft_runner.alloc_memory_pool()
        draft_runner.init_attention_backends()
        # #274 round 7a: BETWEEN the pool and the capture, exactly like the
        # lane target's ``dual_group_lane_verify_tokens``. ``alloc_memory_pool``
        # re-inits the graph-runner field block, so anything written before it
        # is gone by the time the capture reads it.
        if head_graph:
            draft_runner.dual_group_lane_draft_capture = True
        # init_cuda_graphs() is called even though the head runs eager: it
        # also builds the EAGER phase runner, without which forward dispatch
        # has no `eager_runner` at all (measured: AttributeError on the
        # head's first forward). Every graph phase is DISABLED in the active
        # view, so this builds the dispatcher and captures nothing.
        # THE NAMED GAP OF ROUND 6 IS CLOSED HERE (#274 round 7a). It was:
        # the head's decode graph would be captured by the plain
        # DecodeCudaGraphRunner, which builds a dummy batch with
        # spec_info=None -- and an MTP forward reads
        # forward_batch.spec_info.hidden_states, so the capture died on a
        # NoneType. The fix is NOT to route the head through the EAGLE draft
        # graph runner (that runner captures the whole topk/tree/sampling draft
        # loop, which the lane's greedy topk-1 chain does not have and would
        # have to be undone again); it is one targeted branch in
        # ``get_spec_info`` that builds a real EagleDraftInput over a static
        # hidden-states buffer, plus this decode phase being enabled above.
        #
        # init_cuda_graphs() is called even when the head runs eager: it also
        # builds the EAGER phase runner, without which forward dispatch has no
        # `eager_runner` at all (measured: AttributeError on the head's first
        # forward). With the head graph off, every phase is DISABLED in the
        # active view and this builds the dispatcher and captures nothing.
        draft_runner.init_cuda_graphs()


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
        # #274 round 6: ask the lane target's decode graph runner for a chain-
        # VERIFY entry beside its plain decode ones. Set here, between
        # construction and init_cuda_graphs(), for the same reason
        # spec_solo_rank_local_graphs is: the capture reads it, and the capture
        # happens inside init_cuda_graphs. Eager lanes and a disabled graph plan
        # capture nothing at all, so the field would only be misleading there.
        # Round 7a: a LADDER of such entries, one per configured rung. K = 0 is
        # deliberately absent from it -- that rung IS the plain decode entry
        # this runner already captured, so it costs no graph and must not be
        # re-recorded as a one-row verify.
        lane_rungs = resolve_lane_spec_rungs(server_args)
        if (
            getattr(server_args, "dual_group_lane_spec", False)
            and getattr(server_args, "dual_group_lane_spec_graph", True)
            and not server_args.dual_group_lane_eager
        ):
            verify_rungs = tuple(k + 1 for k in lane_rungs if k >= 1)
            if verify_rungs:
                runner.dual_group_lane_verify_tokens = verify_rungs
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
        spec_rungs=lane_rungs,
        spec_adaptive=bool(
            getattr(server_args, "dual_group_lane_spec_adaptive", False)
        ),
        spec_hysteresis=int(
            getattr(server_args, "dual_group_lane_spec_adaptive_hysteresis", 4)
        ),
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
