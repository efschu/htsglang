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
"""Dual-group runtime (#121): two overlapping TP groups over ONE weight set.

The setting is one process that carries a rank of the large group (BIG, the
serving federation) and, on the SAME card and out of the SAME weight bytes, a
second self-sufficient group (FAST, the PD lane) that can prefill on its own.

The property that makes this pay is NESTING. Take BIG = TP=3 with
``--rank-tp-ratio 6,1,1`` over 8 units and FAST = a 2-rank group whose first
rank is BIG rank 0 and whose second rank covers BIG ranks 1 and 2, i.e.
FAST ratio ``6,2``. Then

* FAST rank 0's shard IS BIG rank 0's shard -- same unit range, therefore the
  same tensor objects, therefore zero additional bytes;
* FAST rank 1's shard is exactly what the other cards hold, so the shared card
  ends up holding 6/8 + 2/8 = the FULL weights, ONCE.

Both FAST ranks live in one process, so the group's collectives are not wire
operations at all: an all-gather is a ``cat`` and an all-reduce is an ``add``
over two locally resident shards (:func:`local_column_gather`,
:func:`local_row_reduce`). The FAST group therefore has no communicator, which
is why it needs neither the NCCL >= 2.30 multi-rank-per-GPU threshold nor an
MPS daemon -- the two conditions that gate the two-process co-location path.

NESTING IS NOT A CONSTRUCTION GUARANTEE, IT IS A CHECKED PROPERTY.
``partition_units`` is largest-remainder with minimum-one bumping, and the
kv-group alignment can move boundaries again. BIG rank 0 and FAST rank 0 share
the same real quota, hence the same floor -- but the remainder pass can hand a
bump to one and not the other, and then the two "shared" shards are different
unit ranges and the bytes are NOT shareable. So the property is verified per
(unit count, kv groups, family) probe, and a violation is a boot-time
rejection naming the family, the unit count, both partitions and the segment
that broke (:func:`check_nesting`).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.distributed.utils import partition_units

__all__ = [
    "LaneCardDemand",
    "NestedGroupPlan",
    "NestingProbe",
    "VramPost",
    "check_nesting",
    "derive_nested_plan",
    "format_vram_posts",
    "lane_card_demands",
    "lane_part_device_indices",
    "lane_visible_physical_gpus",
    "lane_vram_posts",
    "local_column_gather",
    "local_row_reduce",
    "local_row_split",
    "nesting_failures",
    "transformer_nesting_probes",
]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NestedGroupPlan:
    """A FAST group expressed as a segmentation of the BIG group's ranks.

    ``segments[f]`` is the tuple of BIG ranks that FAST rank ``f`` covers.
    Segments are contiguous, ordered, non-empty and partition ``range(len(
    big_ratio))`` -- that is what makes the prefix sums line up, and only
    matching prefix sums make the shared shard the same BYTES rather than
    merely the same number of bytes.

    A segment of length 1 is a SHARED rank: its shard is byte-identical to
    that BIG rank's resident shard. A longer segment is a COMPLEMENT rank: its
    shard has to be materialized on the lane's card, because it lives on other
    cards in the BIG group.
    """

    big_ratio: Tuple[int, ...]
    segments: Tuple[Tuple[int, ...], ...]
    #: Per-family BIG vectors (``mlp``, ``moe``, ``vocab``) as (name, vector)
    #: pairs -- the same shape ``set_tp_partition_ratios`` takes, kept as a
    #: tuple so the plan stays hashable.
    family_ratios: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if len(self.big_ratio) < 2:
            raise ValueError(
                "A dual-group plan needs a BIG group of at least 2 ranks, got "
                f"--rank-tp-ratio {list(self.big_ratio)}."
            )
        if any(not isinstance(w, int) or w <= 0 for w in self.big_ratio):
            raise ValueError(
                f"BIG ratio entries must be positive integers, got "
                f"{list(self.big_ratio)}."
            )
        if not self.segments:
            raise ValueError("A dual-group plan needs at least one FAST rank.")
        flat: List[int] = []
        for seg in self.segments:
            if not seg:
                raise ValueError("Empty segment in a dual-group plan.")
            if tuple(seg) != tuple(range(seg[0], seg[0] + len(seg))):
                raise ValueError(
                    f"Segment {list(seg)} is not a contiguous run of BIG "
                    "ranks. A FAST rank must cover consecutive BIG ranks, "
                    "otherwise its shard is not one unit range and the "
                    "nesting cannot make the bytes shareable."
                )
            flat.extend(seg)
        if flat != list(range(len(self.big_ratio))):
            raise ValueError(
                f"Segments {[list(s) for s in self.segments]} must partition "
                f"the BIG ranks 0..{len(self.big_ratio) - 1} in order, got "
                f"{flat}."
            )
        for name, vec in self.family_ratios:
            if len(vec) != len(self.big_ratio):
                raise ValueError(
                    f"Family vector {name!r} has {len(vec)} entries but the "
                    f"BIG ratio has {len(self.big_ratio)} ({list(vec)} vs "
                    f"{list(self.big_ratio)})."
                )

    # -- derived ---------------------------------------------------------

    def big_ratio_for(self, family: Optional[str] = None) -> Tuple[int, ...]:
        """The BIG vector a layer of this family shards by (falls back to the
        base vector, exactly like ``get_tp_partition_ratios``)."""
        if family is not None:
            for name, vec in self.family_ratios:
                if name == family:
                    return tuple(vec)
        return self.big_ratio

    def fast_ratio_for(self, family: Optional[str] = None) -> Tuple[int, ...]:
        """The FAST vector for a family: segment sums of the BIG vector."""
        big = self.big_ratio_for(family)
        return tuple(sum(big[r] for r in seg) for seg in self.segments)

    @property
    def fast_ratio(self) -> Tuple[int, ...]:
        """The FAST group's --rank-tp-ratio equivalent."""
        return self.fast_ratio_for(None)

    @property
    def fast_family_ratios(self) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """The family vectors to install for the FAST group."""
        return tuple(
            (name, self.fast_ratio_for(name)) for name, _ in self.family_ratios
        )

    @property
    def fast_size(self) -> int:
        return len(self.segments)

    @property
    def shared_fast_ranks(self) -> Tuple[int, ...]:
        return tuple(f for f, seg in enumerate(self.segments) if len(seg) == 1)

    @property
    def complement_fast_ranks(self) -> Tuple[int, ...]:
        return tuple(f for f, seg in enumerate(self.segments) if len(seg) > 1)

    def shared_big_rank(self, fast_rank: int) -> Optional[int]:
        """The BIG rank whose resident shard FAST rank ``fast_rank`` reuses,
        or None when that FAST rank is a complement."""
        seg = self.segments[fast_rank]
        return seg[0] if len(seg) == 1 else None

    # -- host view (#274 families slice 2) --------------------------------
    #
    # A segment of length 1 is byte-shareable IN PRINCIPLE, but only the
    # process that owns that BIG rank can share the bytes: a resident shard
    # of another rank lives in another process and is not addressable here.
    # The two helpers below split the segments the way the BUILD needs them,
    # from the point of view of the process that hosts the lane:
    #
    #   * exactly one fast rank is ALIASED   -- its segment is {host},
    #   * every other fast rank is MATERIALIZED -- the lane loads that shard
    #     itself, whether it is a complement (segment longer than 1) or
    #     another rank's byte-identical singleton (BIG tp_size == 2).
    #
    # For the slice-B shape (host = BIG rank 0, segments ({0}, {1..n-1}))
    # this reproduces the old shared/complement split exactly.

    def host_fast_rank(self, host_big_rank: int) -> int:
        """The fast rank whose shard is the host process's resident shard."""
        for f, seg in enumerate(self.segments):
            if seg == (host_big_rank,):
                return f
        raise ValueError(
            f"dual-group plan {self.describe()} has no singleton segment for "
            f"serving rank {host_big_rank}: the lane must be built in a "
            "process whose resident shard is one whole lane shard, otherwise "
            "there is nothing to share and the lane is just a second model."
        )

    def materialized_fast_ranks(self, host_big_rank: int) -> Tuple[int, ...]:
        """The fast ranks the lane has to load itself, in lane-rank order."""
        host = self.host_fast_rank(host_big_rank)
        return tuple(f for f in range(self.fast_size) if f != host)

    def describe(self) -> str:
        parts = []
        for f, seg in enumerate(self.segments):
            if len(seg) == 1:
                parts.append(f"v{f}=BIG rank {seg[0]} (shared bytes)")
            else:
                ranks = ",".join(str(r) for r in seg)
                parts.append(f"v{f}=BIG ranks {ranks} (complement copy)")
        return (
            f"BIG ratio {list(self.big_ratio)} -> FAST ratio "
            f"{list(self.fast_ratio)}: " + "; ".join(parts)
        )


def derive_nested_plan(
    big_ratio: Sequence[int], shared_big_rank: int = 0
) -> NestedGroupPlan:
    """The slice-A shape: a 2-rank FAST group on ONE card.

    FAST rank 0 reuses ``shared_big_rank``'s shard, FAST rank 1 covers every
    other BIG rank. The remaining ranks must be contiguous, which for a
    2-segment split means ``shared_big_rank`` is the first or the last rank.
    """
    big = tuple(int(w) for w in big_ratio)
    n = len(big)
    if not 0 <= shared_big_rank < n:
        raise ValueError(
            f"shared_big_rank {shared_big_rank} is outside the BIG group (0..{n - 1})."
        )
    others = [r for r in range(n) if r != shared_big_rank]
    if others != list(range(others[0], others[0] + len(others))):
        raise ValueError(
            f"Sharing BIG rank {shared_big_rank} of a {n}-rank group leaves "
            f"the non-shared ranks {others}, which are not contiguous. A "
            "two-rank FAST group can only share the FIRST or the LAST BIG "
            "rank; pick that rank, or give the segmentation explicitly."
        )
    segments: Tuple[Tuple[int, ...], ...]
    if shared_big_rank == 0:
        segments = ((shared_big_rank,), tuple(others))
    else:
        segments = (tuple(others), (shared_big_rank,))
    return NestedGroupPlan(big_ratio=big, segments=segments)


# ---------------------------------------------------------------------------
# Nesting verification
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NestingProbe:
    """One sharded dimension the nesting has to survive.

    ``units`` is the indivisible unit count of that dimension (attention
    heads, kv heads, the 16-element MLP units, experts, vocab padding units),
    ``groups`` the kv-head-group alignment constraint of the Q dimension
    (task #116) or None everywhere else.
    """

    what: str
    units: int
    groups: Optional[int] = None
    family: Optional[str] = None
    #: The FAST group's unit count / group constraint when they differ from
    #: the BIG group's. They CAN differ: the REPLICATED-KV geometry engages on
    #: ``kv_heads < tp_size``, so a model with 2 kv heads is replicated-kv at
    #: TP=3 and normal at TP=2 -- the two groups then do not even split the
    #: same dimension the same way, and nesting is impossible rather than
    #: merely violated. None means "same as the BIG value".
    fast_units: Optional[int] = None
    fast_groups: Optional[int] = None
    fast_groups_set: bool = False

    def big_geometry(self) -> Tuple[int, Optional[int]]:
        return self.units, self.groups

    def fast_geometry(self) -> Tuple[int, Optional[int]]:
        units = self.units if self.fast_units is None else self.fast_units
        groups = self.fast_groups if self.fast_groups_set else self.groups
        return units, groups


def nesting_failures(
    plan: NestedGroupPlan, probes: Sequence[NestingProbe]
) -> List[str]:
    """Every probe on which the FAST split does NOT nest in the BIG split.

    Empty list == the shared segments occupy identical unit ranges in both
    groups, so the shared shards are the same bytes.
    """
    out: List[str] = []
    for probe in probes:
        fam = f" family {probe.family!r}" if probe.family else ""
        big_ratio = plan.big_ratio_for(probe.family)
        fast_ratio = plan.fast_ratio_for(probe.family)
        big_units, big_groups = probe.big_geometry()
        fast_units, fast_groups = probe.fast_geometry()
        if (big_units, big_groups) != (fast_units, fast_groups):
            out.append(
                f"{probe.what}{fam}: the two groups split this dimension with "
                f"DIFFERENT geometry -- BIG uses {big_units} units "
                f"(kv-group alignment {big_groups}), FAST uses {fast_units} "
                f"units (alignment {fast_groups}). Nesting is not violated "
                "here, it is undefined: no shard of one geometry is a shard "
                "of the other. This is the REPLICATED-KV threshold "
                "(kv heads < ranks) engaging for one group and not the other"
            )
            continue
        try:
            big_sizes = partition_units(big_units, big_ratio, big_groups)
            fast_sizes = partition_units(fast_units, fast_ratio, fast_groups)
        except ValueError as exc:
            out.append(
                f"{probe.what}{fam}: {probe.units} units cannot be split at "
                f"all -- {exc}"
            )
            continue
        for f, seg in enumerate(plan.segments):
            want = sum(big_sizes[r] for r in seg)
            got = fast_sizes[f]
            if want == got:
                continue
            kind = "shared" if len(seg) == 1 else "complement"
            out.append(
                f"{probe.what}{fam}: FAST rank {f} ({kind}, covering BIG "
                f"ranks {list(seg)}) gets {got} of {big_units} units under "
                f"ratio {list(fast_ratio)}, but those BIG ranks hold {want} "
                f"under ratio {list(big_ratio)} "
                f"(BIG split {big_sizes}, FAST split {fast_sizes}). The "
                "shards are different unit ranges, so their bytes cannot be "
                "shared."
            )
    return out


def transformer_nesting_probes(
    plan: NestedGroupPlan,
    *,
    num_attention_heads: int,
    num_kv_heads: int,
    intermediate_size: Optional[int] = None,
    num_experts: Optional[int] = None,
    linear_attn_units: Optional[int] = None,
    vocab_units: Optional[int] = None,
    moe_intermediate_size: Optional[int] = None,
    weight_block_size: Optional[Sequence[int]] = None,
    quant_method: Optional[str] = None,
) -> List[NestingProbe]:
    """The probes a dense/GDN/MoE transformer actually needs.

    The attention geometry is asked of the SAME helpers the runtime uses
    (`attn_q_partition_units`, `attn_q_partition_groups`), each under its own
    group's installed vector, so this cannot drift from what the layers do.
    That is also why the geometry is asked twice: those helpers switch on
    ``kv_heads < tp_size``, and the two groups have different sizes.

    The same discipline applies to BLOCK-QUANTIZED weights (#274 families
    slice B).  An FP8 checkpoint with ``weight_block_size`` cannot be split
    inside a quantization block, so the layers re-express the element-granular
    MLP/MoE families in block units (``_quant_block_aligned_units``).  A
    nesting verdict computed on the RAW unit count is then a verdict about a
    partition the layers never perform: for intermediate 17408 with block 128
    the raw count is 1088 and the real one 136, and the two disagree in both
    directions over a swept ratio grid -- including the dangerous one, where
    the raw count says "nested" and the block count says "not".  So the block
    size is asked of the checkpoint and applied here, exactly as at
    construction.  The scale grids (``ceil(out/block_n)`` etc.) need no probe
    of their own: they carry the SAME unit count as the weight by
    construction, which is why one aligned probe covers both.
    """
    import math

    from sglang.srt.distributed.utils import (
        ACTIVATION_VEC_ELEMS,
        attn_q_partition_groups,
        attn_q_partition_units,
        block_aligned_units,
        scoped_tp_partition_ratios,
    )

    block_n = block_k = None
    if weight_block_size and len(weight_block_size) == 2:
        block_n, block_k = int(weight_block_size[0]), int(weight_block_size[1])
        if quant_method == "gguf":
            # GGUF quantizes along the INPUT dim only; an output-dim split
            # never cuts a block, and coarsening on its nominal [256,256]
            # would wrongly merge fine-grained head units (see
            # _quant_block_aligned_units).
            block_n = None

    def _aligned(total, units, what, family):
        probes = []
        for block, axis in ((block_n, "output"), (block_k, "input")):
            u = block_aligned_units(total, units, block)
            if not any(p.units == u and p.family == family for p in probes):
                label = what if u == units else f"{what} ({axis} quant blocks)"
                probes.append(NestingProbe(what=label, units=u, family=family))
        return probes

    big_ratio = plan.big_ratio
    fast_ratio = plan.fast_ratio
    with scoped_tp_partition_ratios(big_ratio):
        big_q_units = attn_q_partition_units(
            num_attention_heads, num_kv_heads, len(big_ratio)
        )
        big_q_groups = attn_q_partition_groups(num_kv_heads, len(big_ratio))
    with scoped_tp_partition_ratios(fast_ratio):
        fast_q_units = attn_q_partition_units(
            num_attention_heads, num_kv_heads, len(fast_ratio)
        )
        fast_q_groups = attn_q_partition_groups(num_kv_heads, len(fast_ratio))

    probes = [
        NestingProbe(
            what="attention q heads (qkv q block, o_proj input)",
            units=big_q_units,
            groups=big_q_groups,
            fast_units=fast_q_units,
            fast_groups=fast_q_groups,
            fast_groups_set=True,
        )
    ]
    if intermediate_size:
        probes.extend(
            _aligned(
                intermediate_size,
                intermediate_size // math.gcd(intermediate_size, ACTIVATION_VEC_ELEMS),
                "MLP intermediate",
                "mlp",
            )
        )
    if num_experts:
        probes.append(NestingProbe(what="MoE experts", units=num_experts, family="moe"))
    if moe_intermediate_size:
        # The MoE family shards the per-expert INTERMEDIATE dimension whenever
        # the expert dimension itself is not the shard axis (everything except
        # the GGUF expert-shard path). Its unit count is not num_experts, so it
        # needs its own probe.
        probes.extend(
            _aligned(
                moe_intermediate_size,
                moe_intermediate_size
                // math.gcd(moe_intermediate_size, ACTIVATION_VEC_ELEMS),
                "MoE per-expert intermediate",
                "moe",
            )
        )
    if linear_attn_units:
        probes.append(
            NestingProbe(
                what="linear-attention (GDN) k-head units",
                units=linear_attn_units,
            )
        )
    if vocab_units:
        probes.append(
            NestingProbe(what="vocabulary", units=vocab_units, family="vocab")
        )
    return probes


def check_nesting(plan: NestedGroupPlan, probes: Sequence[NestingProbe]) -> None:
    """Raise unless the FAST split nests in the BIG split on every probe."""
    failures = nesting_failures(plan, probes)
    if not failures:
        return
    raise ValueError(
        "Dual-group plan is not nested -- the PD lane cannot share the BIG "
        "group's weight bytes with this ratio pair.\n  "
        + plan.describe()
        + "\n  "
        + "\n  ".join(failures)
        + "\n  Fix: choose a --rank-tp-ratio whose shared rank keeps the same "
        "unit range in both groups (a ratio that divides every listed unit "
        "count exactly is always nested), or place the lane on a different "
        "rank."
    )


# ---------------------------------------------------------------------------
# The FAST group's collectives, done locally
# ---------------------------------------------------------------------------


def local_column_gather(parts, per_part_sub_sizes):
    """All-gather of a column-parallel output, as a local ``cat``.

    ``parts[f]`` is FAST rank ``f``'s output slice; ``per_part_sub_sizes[f]``
    lists the widths of that rank's SUB-OUTPUTS in the order they are packed
    into it (``[gate, up]`` for a merged column linear, ``[q, k, v]`` for a
    QKV linear, a single entry for a plain one).

    The concatenation is per SUB-OUTPUT, not per rank. Concatenating whole
    rank slices would produce ``[q0,k0,v0,q1,k1,v1]``; the canonical full
    layout every consumer expects is ``[q_all, k_all, v_all]``. This is pure
    data movement, so it is bit-identical to a real all-gather.
    """
    import torch

    if len(parts) != len(per_part_sub_sizes):
        raise ValueError(
            f"local_column_gather: {len(parts)} parts but "
            f"{len(per_part_sub_sizes)} sub-size lists."
        )
    n_sub = len(per_part_sub_sizes[0])
    for f, sizes in enumerate(per_part_sub_sizes):
        if len(sizes) != n_sub:
            raise ValueError(
                "local_column_gather: every rank must declare the same number "
                f"of sub-outputs; rank 0 has {n_sub}, rank {f} has "
                f"{len(sizes)}."
            )
        if parts[f].shape[-1] != sum(sizes):
            raise ValueError(
                f"local_column_gather: rank {f}'s part has last dimension "
                f"{parts[f].shape[-1]} but its sub-sizes {list(sizes)} sum to "
                f"{sum(sizes)}."
            )
    pieces = []
    for s in range(n_sub):
        for f, sizes in enumerate(per_part_sub_sizes):
            off = sum(sizes[:s])
            pieces.append(parts[f][..., off : off + sizes[s]])
    return torch.cat(pieces, dim=-1)


def local_row_split(x, per_part_sizes):
    """Split a full-width row-parallel input into the FAST ranks' slices.

    Each slice is handed out CONTIGUOUS. On a real rank the row-parallel input
    is that rank's own activation tensor -- densely packed, row stride equal to
    its shard width. The shell's slice of a full-width tensor is the only place
    in the system where a kernel sees that input strided (row stride = FULL
    width), and not every kernel reads strides: GGUF's ``fused_mul_mat_gguf``
    picks the mat-VEC kernel for ``x.shape[0] <= 8`` -- exactly the K+1 rows of
    a lane verify forward -- and that kernel quantizes the activation assuming
    a contiguous row stride. With a view it therefore reads row i at the full
    width's offset: row 0 lands on the right bytes and every row after it does
    not. That is #274's rows>=1 defect, measured to the module
    (``mlp.down_proj``, rows 1-3 wrong while ``linear_attn`` and
    ``mlp.gate_up_proj`` are exact).

    A one-row input (plain decode) and a single-part split are contiguous
    already, so this copies only in the case that was broken -- the paths that
    were byte-green stay bit-for-bit unchanged.
    """
    total = sum(per_part_sizes)
    if x.shape[-1] != total:
        raise ValueError(
            f"local_row_split: input has last dimension {x.shape[-1]} but the "
            f"per-rank input sizes {list(per_part_sizes)} sum to {total}."
        )
    out = []
    off = 0
    for size in per_part_sizes:
        piece = x[..., off : off + size]
        out.append(piece if piece.is_contiguous() else piece.contiguous())
        off += size
    return out


def local_row_reduce(parts):
    """All-reduce of a row-parallel output, as a local sum.

    Bit-identical to a real 2-rank all-reduce (both are one addition), but
    deliberately NOT identical to a monolithic GEMM over the full k axis, nor
    to an N-rank reduction for N != len(parts): float addition is not
    associative and the accumulation order differs. Whoever compares the lane
    against the federation has to state that tolerance up front.
    """
    if not parts:
        raise ValueError("local_row_reduce: no parts.")
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


# ---------------------------------------------------------------------------
# VRAM posting of the shared card
# ---------------------------------------------------------------------------

SHARED = "shared"
NESTED = "nested"
DUPLICATED = "duplicated"
#: A post the operator already sized with ``--rank-gpu-memory-mib``. Not a
#: sharing status like the three above -- it marks the items that were on the
#: card BEFORE the lane, so a card ledger can add the lane's own posts to them
#: instead of pretending the card starts empty (#400).
BUDGETED = "budgeted"


@dataclasses.dataclass(frozen=True)
class VramPost:
    """One item of the shared card's memory, and why it costs what it costs.

    ``status`` is one of ``shared`` (the same bytes serve both groups),
    ``nested`` (bytes the lane adds, but which exist exactly once on this
    card and nowhere else in this process) or ``duplicated`` (a real second
    copy, deliberately accepted).
    """

    name: str
    status: str
    mib: int
    why: str


def lane_vram_posts(
    plan: NestedGroupPlan,
    *,
    shared_fast_rank: int,
    total_weight_mib: int,
    weight_units: int,
    lane_kv_mib: int,
    lane_state_mib: int,
    lane_scratch_mib: int,
) -> Tuple[VramPost, ...]:
    """The shared card's memory items with their sharing status.

    ``total_weight_mib`` is the whole model's weight footprint and
    ``weight_units`` the unit count the weight dimensions are split into, so
    the shard sizes follow the same partition the runtime uses.
    """
    big_sizes = partition_units(weight_units, plan.big_ratio)
    big_rank = plan.shared_big_rank(shared_fast_rank)
    if big_rank is None:
        raise ValueError(
            f"FAST rank {shared_fast_rank} covers BIG ranks "
            f"{list(plan.segments[shared_fast_rank])} and is a complement, not "
            "the shared rank."
        )
    per_unit = total_weight_mib / weight_units
    resident = int(round(big_sizes[big_rank] * per_unit))
    complement_units = weight_units - big_sizes[big_rank]
    complement = int(round(complement_units * per_unit))
    return (
        VramPost(
            name=f"BIG rank {big_rank} weight shard "
            f"({big_sizes[big_rank]}/{weight_units} units)",
            status=SHARED,
            mib=resident,
            why="the lane computes with these very tensor objects; verified "
            "by data_ptr identity, not assumed",
        ),
        VramPost(
            name=f"lane complement shard ({complement_units}/{weight_units} units)",
            status=NESTED,
            mib=complement,
            why="these bytes exist nowhere else in this process -- they are "
            "what the other cards hold. Adding them makes this card hold the "
            "full weights exactly ONCE (not twice); without them the lane "
            "cannot prefill on its own",
        ),
        VramPost(
            name="lane shell model tree",
            status=SHARED,
            mib=0,
            why="built on the meta device, zero weight bytes "
            "(precedent: _build_weightless_worker_meta_model)",
        ),
        VramPost(
            name="lane KV pool",
            status=DUPLICATED,
            mib=lane_kv_mib,
            why="the lane needs request-disjoint slots; deliberately small, "
            "every MiB here is missing from the BIG group's pool",
        ),
        VramPost(
            name="lane linear-attention state pool",
            status=DUPLICATED,
            mib=lane_state_mib,
            why="a second state pool for the lane's sessions; budget only, "
            "not a correctness question",
        ),
        VramPost(
            name="lane activations / scratch",
            status=DUPLICATED,
            mib=lane_scratch_mib,
            why="per forward, never shared. Under serial lane ticks this peak "
            "does not overlap the BIG group's peak",
        ),
        VramPost(
            name="CUDA context",
            status=SHARED,
            mib=0,
            why="same process -- this is the item a second co-located PROCESS "
            "would additionally pay",
        ),
    )


def format_vram_posts(posts: Sequence[VramPost], card: str) -> str:
    """The boot-log block: every item, its sharing status and the sum."""
    width = max(len(p.name) for p in posts)
    lines = [f"dual-group lane on {card}: memory items"]
    for p in posts:
        lines.append(f"  {p.name:<{width}}  {p.mib:>7} MiB  {p.status:<10} {p.why}")
    added = sum(p.mib for p in posts if p.status != SHARED)
    lines.append(
        f"  {'-> added by the lane':<{width}}  {added:>7} MiB  "
        "(shared items cost nothing)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation-time card ledger (#400)
# ---------------------------------------------------------------------------
#
# ``lane_vram_posts`` / ``format_vram_posts`` above are the RUNTIME posting:
# they run inside build_lane_model with mem_get_info deltas, i.e. after the
# bytes are already on the card. That is a report, not a guard, and #349 arm L
# is what a report costs -- the lane's complement shard and its pool were
# never weighed against the rank's --rank-gpu-memory-mib budget, the boot was
# accepted, and the process died at 31.14 GiB in use on a 31.34 GiB card while
# loading the complement.
#
# Both of the lane's big items are decided by configuration alone:
#
#   * the pool is --dual-group-lane-budget-mib, mandatory with the lane;
#   * the complement shard is a fixed unit fraction of the model weights that
#     the nesting plan states exactly (BIG 2,1,1 -> FAST 2,2 makes the host
#     card materialize 2 of 4 units on top of the 2 it already holds).
#
# So they can be charged BEFORE any card is touched. The ledger below is
# deliberately a FLOOR: it prices the items whose size follows from the config
# and names the ones it does not price, so a refusal is always true (the floor
# alone already exceeds the card) and never invented.


@dataclasses.dataclass(frozen=True)
class LaneCardDemand:
    """Everything a configured dual-group lane will place on ONE card.

    ``posts`` carries the serving ranks' own budgets (``BUDGETED``) plus the
    lane's items, so ``charged_mib`` is the whole card's committed floor and
    can be compared against ``total_mib`` directly.

    ``unpriced`` names the items that exist on this card but whose size is not
    knowable from configuration (hull residue, lane activations, the lane's
    graph pool). ``unbounded`` names the items that SHOULD be priced here and
    could not be -- an empty tuple is the normal case, a non-empty one means
    the ledger cannot answer and the caller must refuse rather than guess.
    """

    gpu_id: int
    total_mib: int
    posts: Tuple[VramPost, ...]
    unpriced: Tuple[str, ...] = ()
    unbounded: Tuple[str, ...] = ()

    @property
    def charged_mib(self) -> int:
        return sum(p.mib for p in self.posts)

    @property
    def lane_added_mib(self) -> int:
        return sum(p.mib for p in self.posts if p.status not in (SHARED, BUDGETED))

    @property
    def fits(self) -> bool:
        return not self.unbounded and self.charged_mib <= self.total_mib

    def ledger(self, card: Optional[str] = None) -> str:
        """The itemization a refusal (or the boot log) prints."""
        name = card if card is not None else f"GPU {self.gpu_id}"
        width = max([len(p.name) for p in self.posts] + [20])
        lines = [
            f"dual-group lane budget on {name}: "
            f"{self.charged_mib} MiB charged of {self.total_mib} MiB total"
        ]
        for p in self.posts:
            lines.append(f"  {p.name:<{width}}  {p.mib:>7} MiB  {p.status:<10} {p.why}")
        for item in self.unbounded:
            lines.append(f"  {item:<{width}}  {'?':>7} MiB  cannot bound")
        if self.unpriced:
            lines.append(
                "  not priced here (the sum above is a FLOOR, not the full "
                "footprint): " + ", ".join(self.unpriced)
            )
        return "\n".join(lines)


#: Items that live on the lane's card but whose size does not follow from the
#: configuration. Named rather than estimated: a number nobody measured is the
#: defect ``pinned_reserve_shortfall_note`` exists to expose, and the guard's
#: job is to refuse what provably cannot fit, not to predict the peak.
LANE_UNPRICED_ITEMS: Tuple[str, ...] = (
    "hull tree residue (composed conv/state vectors, buffers)",
    "lane activations / scratch",
    "lane CUDA graph capture pool",
)


def lane_card_demands(
    plan: NestedGroupPlan,
    *,
    rank_gpu_id: Sequence[int],
    rank_budget_mib: Sequence[int],
    card_total_mib: Mapping[int, int],
    lane_pool_mib: int,
    total_weight_mib: Optional[int],
    lane_part_gpu_id: Optional[Sequence[int]] = None,
    host_big_rank: int = 0,
) -> Tuple[LaneCardDemand, ...]:
    """Per-card committed floor for a configured lane, in ``rank_gpu_id`` order.

    ``total_weight_mib`` is the whole model's resident weight footprint; None
    means it could not be derived, which yields an ``unbounded`` entry on
    every card that has to materialize a lane part rather than a silent zero.

    Pure: no NVML, no torch, no checkpoint. Everything comes from the plan, the
    placement vectors and the two MiB numbers the operator wrote down.
    """
    host_gpu = int(rank_gpu_id[host_big_rank])
    host_fast = plan.host_fast_rank(host_big_rank)
    if lane_part_gpu_id:
        part_gpu = [int(g) for g in lane_part_gpu_id]
    else:
        part_gpu = [host_gpu] * plan.fast_size

    weight_units = sum(plan.big_ratio)
    fast_ratio = plan.fast_ratio
    per_unit = (
        None if total_weight_mib is None else float(total_weight_mib) / weight_units
    )

    posts: Dict[int, List[VramPost]] = {}
    unbounded: Dict[int, List[str]] = {}

    def add(gpu: int, post: VramPost) -> None:
        posts.setdefault(int(gpu), []).append(post)

    # What the operator already sized, per card.
    for rank, gpu in enumerate(rank_gpu_id):
        add(
            gpu,
            VramPost(
                name=f"serving rank {rank} budget (--rank-gpu-memory-mib)",
                status=BUDGETED,
                mib=int(rank_budget_mib[rank]),
                why="the rank's ENTIRE budget: weights, KV, runtime state. "
                "The lane's items come on top of it, not out of it",
            ),
        )

    # The lane's materialized parts. The host's own segment is shared bytes
    # and costs nothing; every other lane rank is a real load.
    for f in range(plan.fast_size):
        if f == host_fast:
            add(
                host_gpu,
                VramPost(
                    name=f"lane rank {f} shard (host's resident segment)",
                    status=SHARED,
                    mib=0,
                    why="the same tensor objects the serving rank already "
                    "holds; verified by data_ptr identity at bring-up",
                ),
            )
            continue
        gpu = part_gpu[f]
        units = fast_ratio[f]
        flag = (
            "--dual-group-lane-part-gpu-id" if lane_part_gpu_id else "--dual-group-lane"
        )
        label = (
            f"lane rank {f} complement shard ({units}/{weight_units} units, " f"{flag})"
        )
        if per_unit is None:
            unbounded.setdefault(int(gpu), []).append(label)
            continue
        add(
            gpu,
            VramPost(
                name=label,
                status=NESTED if gpu == host_gpu else DUPLICATED,
                mib=int(round(units * per_unit)),
                why=(
                    "bytes the other cards hold; this card must materialize "
                    "them so the lane can run on its own"
                    if gpu == host_gpu
                    else "a full second copy: the resident shard on that card "
                    "belongs to another process and cannot be aliased"
                ),
            ),
        )

    # The lane's own pool, always on the host card.
    add(
        host_gpu,
        VramPost(
            name="lane pool (--dual-group-lane-budget-mib)",
            status=DUPLICATED,
            mib=int(lane_pool_mib),
            why="KV + linear-attention state + workspace of the lane runner; "
            "the ENTIRE pool item, no ceiling applied on top",
        ),
    )

    out = []
    for gpu in sorted(set(int(g) for g in rank_gpu_id) | set(posts) | set(unbounded)):
        out.append(
            LaneCardDemand(
                gpu_id=gpu,
                total_mib=int(card_total_mib[gpu]),
                posts=tuple(posts.get(gpu, ())),
                unpriced=LANE_UNPRICED_ITEMS if gpu == host_gpu else (),
                unbounded=tuple(unbounded.get(gpu, ())),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Lane part placement across cards (#274 families slice 2, arm C)
# ---------------------------------------------------------------------------
#
# By default every lane part lives on the host rank's card: the lane is the
# slice-A "second group on ONE card". A lane whose weights do not fit on one
# card can place its MATERIALIZED parts on other cards instead. That does not
# need a communicator -- the lane's collectives are already local tensor ops
# (section 4) -- it turns them into cross-device copies of the ACTIVATION,
# which is the smaller object by orders of magnitude.
#
# The one thing it does need is visibility: with --rank-gpu-id each scheduler
# process sees exactly one card (CUDA_VISIBLE_DEVICES isolation), so the host
# process has to be given the foreign cards too. Both sides -- the parent that
# composes CUDA_VISIBLE_DEVICES and the lane that has to name a device inside
# the process -- derive their answer from the two functions below, so the
# order cannot drift between them.


def lane_visible_physical_gpus(
    host_physical_gpu: int, part_physical_gpus: Sequence[int]
) -> List[int]:
    """The physical GPU ids the lane HOST process must see, in order.

    The host card is always first, so ``cuda:0`` keeps meaning "this rank's
    card" inside the process exactly as it does without a lane.
    """
    order = [int(host_physical_gpu)]
    for gid in part_physical_gpus:
        gid = int(gid)
        if gid not in order:
            order.append(gid)
    return order


def lane_part_device_indices(
    host_physical_gpu: int,
    part_physical_gpus: Sequence[int],
    visible_physical_gpus: Optional[Sequence[int]] = None,
) -> List[int]:
    """Per-fast-rank IN-PROCESS device index for the given physical placement.

    ``visible_physical_gpus`` is what CUDA_VISIBLE_DEVICES says (None: every
    card is visible under its own physical index). A physical id that the
    process cannot see is a hard error naming the flag -- the alternative is
    an out-of-range cuda index deep inside the loader.
    """
    if visible_physical_gpus is None:
        return [int(g) for g in part_physical_gpus]
    visible = [int(g) for g in visible_physical_gpus]
    out = []
    for f, gid in enumerate(part_physical_gpus):
        gid = int(gid)
        if gid not in visible:
            raise ValueError(
                f"--dual-group-lane-part-gpu-id places lane rank {f} on "
                f"physical GPU {gid}, but the lane host process only sees "
                f"{visible} (CUDA_VISIBLE_DEVICES). The host card is "
                f"{host_physical_gpu}; a foreign card has to be listed in the "
                "flag BEFORE the process starts, it cannot be attached later."
            )
        out.append(visible.index(gid))
    return out
