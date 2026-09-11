# SPDX-License-Identifier: Apache-2.0
"""Weg-2 #1273 -- the weight-byte EXCHANGE.

Spec of record: ``/spinning/gpu-arb/weg2/WEG2_REUSE_SPEC_0908.md``.

Under ``--weg2-weight-source exchange`` the weights region is opened with
``enable_cpu_backup=False`` (``model_executor/model_runner.py:2419-2425``),
which makes ``pause`` a pure unmap and ``resume`` a pure remap
(``core.cpp:229-231``, ``:362-364``).  No host granule is acquired on the
weights path, so the host ring is not created at all; between the
destination's remap and its first use the exchange injects the bytes
VRAM->VRAM.

THIS FILE IS BUILT BY TWO SLICES, kept in separate labelled sections so the
two branches meet without arguing about line order:

* **SECTION S1 -- THE PLAN** (spec 6/S1, below): which bytes of which rank's
  storage become which bytes of which other rank's storage.  Pure arithmetic
  over replicated geometry, which is what lets every rank derive the same plan
  independently and then HANDSHAKE it (Gate 0, spec 3.3) rather than trust it.
* **SECTION S2 -- COVERAGE ARMING (W84) AND THE DRAFT TAG** (spec 6/S2, 4.1,
  below).  S2 consumes exactly one thing from S1 and nothing else:
  ``{tag: {parameter name: planned bytes}}`` (see :data:`PlanBytes`), where
  the bytes are the sum of that parameter's ``XchgDesc.nbytes``.

  TODO(S6): that adapter is not wired here.  ``register_plan_provider`` has no
  registrant yet, so under ``exchange`` every rank votes ``ok=False`` with
  ``NO_PLAN_REASON`` -- deliberate fail-closed, not a working arm.  S6 registers
  a provider that sums ``XchgDesc.nbytes`` per ``param_name`` over
  ``XchgPlan.raw_descs``.

Everything here is hermetic: no CUDA, no torch_memory_saver, no checkpoint.
It moves nothing and opens nothing.  The shared region, Gate 0 and the wave
gate are S3; the transport is S4.

THE THREE READINGS THAT ARE WRONG AND LOOK RIGHT (S1)
-----------------------------------------------------
1. **Checkpoint sub-block offsets for a fused parameter.** SECTION 1af's
   IDENTICAL-SUBBLOCK probe is CHECKPOINT-space; it proves no repack occurs and
   nothing about the device. On the device ``in_proj_qkvz`` is ONE buffer of
   four rank-local sub-blocks whose offsets are the prefix sum of THIS RANK'S
   sizes (``layers/linear.py:1088-1101``), not of the full ones. Reading the
   checkpoint's offsets as device offsets writes ``k`` over ``q`` with no error
   at all. The two coincide only for block 0; what coincides on rank 0 is the
   GLOBAL start, which is why a single-rank check does not see the defect.
   ``device_block_offsets`` is that law; ``Block.global_start`` and
   ``Block.dev_row`` are the two coordinate systems held deliberately apart.
2. **Logical shape instead of storage.** ``.t()`` in the int8 quant path is a
   view: the logical shape is ``[K, N_local]`` over storage ``[N_local, K]``.
   ``StorageGeom.of`` therefore derives everything from ``stride()`` and
   ``element_size()`` and never reads ``shape`` to produce an offset.
3. **Element counts where bytes are meant.** Every pitch, run and offset a
   descriptor carries is in BYTES. The quantized classes are int8, whose
   itemsize is 1, so a missing ``element_size()`` multiply is invisible on them
   and wrong by 2x on the bf16 embedding.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace as _dc_replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.distributed.utils import partition_sizes
from sglang.srt.managers.weg2_memory_saver import (
    Weg2XchgCoverageRefused,
    is_weights_family_tag,
    layer_id_from_module_name,
    weight_chunk_tag,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ATTRIBUTE",
    "BUFFER",
    "Block",
    "COALESCE_FLOOR_BYTES",
    "COLS",
    "COVERAGE_REFUSAL_MARKER",
    "COVER_LINE_PREFIX",
    "CoverageVote",
    "FLAT",
    "GroupLayout",
    "LiveTensor",
    "MIB",
    "NO_PLAN_REASON",
    "PARAMETER",
    "ParamGeom",
    "PlanBytes",
    "REPLICATED",
    "RESIDENT_LINE_PREFIX",
    "ROWS",
    "RUNNER_SHAPE_REFUSAL_MARKER",
    "RunnerShape",
    "SHAPE_DRAFT",
    "SHAPE_DUAL_GROUP_LANE",
    "SHAPE_PHASE_FLIP_TP_STACK",
    "SHAPE_PRIMARY",
    "SHARD_FAMILIES",
    "STRIDED2D",
    "ShortParameter",
    "StorageGeom",
    "TagCoverage",
    "VOCAB_FAMILY",
    "WEIGHT_SOURCE_ENV",
    "WEIGHT_SOURCE_EXCHANGE",
    "WEIGHT_SOURCE_RING",
    "Weg2XchgPlanDisagree",
    "Weg2XchgRunnerShapeUnknown",
    "Weg2XchgSourceMissing",
    "XchgDesc",
    "XchgPlan",
    "ZEROFILL",
    "arm_coverage",
    "arm_coverage_at_load",
    "boot_vote",
    "build_coverage",
    "build_plan",
    "classify_runner",
    "coalesce",
    "coverage_refusal_message",
    "derive_waves",
    "device_block_offsets",
    "emit_plan_line",
    "exchange_armed",
    "piece_histogram",
    "plan_id",
    "plan_provider",
    "refuse_if_not_ok",
    "register_plan_provider",
    "resident_line",
    "roll_forward_refusal_message",
    "roll_forward_weights_tag",
    "runner_shape_refusal_message",
    "shard_offsets",
    "tag_of_parameter_name",
    "unmergeable_below_floor",
    "walk_live_tensors",
    "weight_source",
    "weight_source_for_test",
    "weights_region_tag_for",
]

# ===========================================================================
# SECTION S1 -- THE PLAN (spec 6/S1).  Hermetic, no CUDA, no allocation.
# ===========================================================================

#: Copy primitive per descriptor (spec §2.2).
FLAT = "FLAT"
STRIDED2D = "STRIDED2D"
ZEROFILL = "ZEROFILL"

#: Which STORAGE axis a parameter is sharded along.  ``ROWS`` is every
#: column-parallel class (contiguous row ranges -> ``cudaMemcpyAsync``);
#: ``COLS`` is every row-parallel class (``cudaMemcpy2DAsync``); ``REPLICATED``
#: is the norms and the replicated weight scales.
ROWS = 0
COLS = 1
REPLICATED = -1

#: E4 bounds the granularity: pieces >= 2 MiB issued async cost <= 1 %, while
#: 256 KiB async costs 4.6-6.4 %.  Only a per-copy SYNC is expensive (2.04x),
#: which is why this is a coalescing floor and not a refusal.
COALESCE_FLOOR_BYTES = 2 * 1024 * 1024

#: The one shard family whose vector does NOT fall back to the base plan.
#: ``distributed/utils.py:1585 tp_vocab_ratios``: "deliberately does NOT fall
#: back to the base --rank-tp-ratio vector ... the vocab dimension of
#: VocabParallelEmbedding / ParallelLMHead keeps the classic EVEN split under a
#: plain uneven-TP plan".  Planning the vocabulary with the base vector puts the
#: largest single parameter of the model at wrong offsets on both sides
#: IDENTICALLY, so no tiling check and no handshake can see it (spec §8 R5).
VOCAB_FAMILY = "vocab"

#: The named families the fork installs beside the base vector
#: (``layers/linear.py:605,899,2064 tp_family``;
#: ``layers/moe/fused_moe_triton/layer.py:433 moe_tp_family``).
SHARD_FAMILIES = (VOCAB_FAMILY, "mlp", "moe")


#: W68 -- THE REGION'S CLASS, IMPORTED, because for a while there were TWO.
#:
#: This module defined its own ``Weg2XchgPlanDisagree(RuntimeError)`` while
#: ``weight_exchange_region.py:260`` defined another, both documented W68 and
#: both named the same -- so the W-code census, which keys on (code, NAME),
#: could not see the duplication, and neither could review.  What it cost is
#: concrete: the TRANSPORT raises the region's, so an
#: ``except weight_exchange.Weg2XchgPlanDisagree`` around a staging call caught
#: NOTHING and the refusal escaped under a different identity than the one the
#: handler named.  Two classes for one W-code is a second identity for one
#: fact, which is the same defect family as a second ledger.
#:
#: The REGION's is the survivor rather than this one because it is the one the
#: transport already raises, so nothing that currently works changes identity.
#: Imported at module level: the region's only sglang import is lazy inside
#: ``fence_budget_s()``, so there is no cycle.
#:
#: Raised here when a destination tensor is not tiled exactly by its sources
#: (two sources for one byte, or a piece that runs past the destination's own
#: storage), when a live tensor's layout cannot be expressed as
#: (rows, cols, pitch), and by the plan provider when the derivation refuses.
from sglang.srt.weg2.weight_exchange_region import (  # noqa: E402
    Weg2XchgPlanDisagree,
)


class Weg2XchgSourceMissing(RuntimeError):
    """W74 Weg2XchgSourceMissing -- a destination byte range has no VRAM source
    and no ZEROFILL descriptor.

    The concrete shape this guards is the draft/MTP class leaking back into the
    weights family: D's base tag carries the whole NEXTN draft runner, which is
    never paused and never exchanged (spec §4.1).

    TODO(S6): the RPC preamble raises this class again, before the first
    ``resume``, so the refusal lands where ``front.py:2658-2663``'s exit shape
    applies and the SOURCE is provably untouched.
    """


# ---------------------------------------------------------------------------
# Storage-space geometry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageGeom:
    """What a live tensor looks like in STORAGE, in elements plus an itemsize.

    Derived from ``stride()`` and ``element_size()`` only.  ``shape`` is read
    solely to pair with the strides -- never to produce an offset -- because a
    ``.t()`` view reports ``[K, N_local]`` over storage ``[N_local, K]`` and a
    plan built from it is transposed-wrong for every column-parallel class
    (spec §2.4 rule 2).
    """

    rows: int
    cols: int
    pitch: int  # elements between the starts of two storage rows; >= cols
    itemsize: int

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols * self.itemsize

    @property
    def contiguous(self) -> bool:
        return self.pitch == self.cols

    @classmethod
    def of(cls, tensor) -> StorageGeom:
        shape = tuple(int(s) for s in tensor.shape)
        stride = tuple(int(s) for s in tensor.stride())
        itemsize = int(tensor.element_size())
        if len(shape) == 0:
            return cls(rows=1, cols=1, pitch=1, itemsize=itemsize)
        if len(shape) == 1:
            if stride[0] != 1:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: 1-D tensor with stride {stride} "
                    f"is not a contiguous run; a descriptor cannot name it."
                )
            return cls(rows=1, cols=shape[0], pitch=shape[0], itemsize=itemsize)
        if len(shape) > 2:
            # Only a genuinely contiguous block can be flattened to rows; a
            # guessed pitch is exactly the silent defect this module removes.
            flat = stride[-1] == 1 and all(
                stride[i] == stride[i + 1] * shape[i + 1] for i in range(len(shape) - 1)
            )
            if not flat:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {len(shape)}-D tensor shape "
                    f"{shape} stride {stride} is not a contiguous block, so it "
                    f"has no storage rows a copy primitive can name."
                )
            rows = 1
            for s in shape[:-1]:
                rows *= s
            return cls(rows=rows, cols=shape[-1], pitch=shape[-1], itemsize=itemsize)
        if stride[1] == 1:
            rows, cols, pitch = shape[0], shape[1], stride[0]
        elif stride[0] == 1:
            # A `.t()` view.  The STORAGE is the untransposed matrix; reading
            # `shape` here is exactly the defect this class exists to remove.
            rows, cols, pitch = shape[1], shape[0], stride[1]
        else:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: tensor shape {shape} stride {stride} "
                f"has no unit-stride axis, so it has no storage rows a copy "
                f"primitive can name."
            )
        if pitch < cols:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: tensor shape {shape} stride {stride} "
                f"yields pitch {pitch} < row width {cols}; the rows overlap, "
                f"which no memcpy2d can express."
            )
        return cls(rows=rows, cols=cols, pitch=pitch, itemsize=itemsize)


# ---------------------------------------------------------------------------
# Shard geometry.  Replicated by construction, which is what lets both sides
# of a pair derive it independently and then agree on it.
# ---------------------------------------------------------------------------


def shard_offsets(
    total: int,
    ratios: Optional[Sequence[int]],
    tp_size: int,
    units: Optional[int] = None,
    groups: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """``[(start, size)]`` per rank of a sharded dimension of ``total``.

    This is ``distributed/utils.py:1808 tp_loaded_shard_start``'s law computed
    for every rank at once: with no ratio vector it is the classic even split
    ``rank * shard_size``; with one it is the prefix sum over
    ``partition_sizes``.  A plain prefix sum is what makes every shard a
    contiguous unit range of the full dimension, and that is the property the
    tiling in ``build_plan`` rests on.

    A vector of the WRONG LENGTH is refused, never downgraded.  The fork reads
    a length mismatch as "this plan does not apply to a group of this size"
    (``:1539 tp_partition_sizes``), but that is an ACTIVATION law and it belongs
    where the group's size is known -- ``GroupLayout.ratios_for`` -- not in the
    arithmetic, where a silently even split is the #1275 class of defect:
    uneven distribution disabled by an accident of length, on both sides
    equally, with nothing downstream able to see it.
    """
    if tp_size <= 0:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: tp_size {tp_size} is not a group size."
        )
    if ratios is not None and len(ratios) and len(ratios) != tp_size:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: ratio vector {list(ratios)} has "
            f"{len(ratios)} entries for a group of {tp_size} ranks. Selecting "
            f"a vector for a group is GroupLayout.ratios_for's job; one that "
            f"reaches the shard arithmetic must already apply to this group."
        )
    if not ratios:
        if total % tp_size != 0:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: dimension {total} is not divisible "
                f"by tp_size {tp_size} and no ratio vector applies."
            )
        sizes = [total // tp_size] * tp_size
    else:
        sizes = [int(s) for s in partition_sizes(total, list(ratios), units, groups)]
    out, acc = [], 0
    for size in sizes:
        out.append((acc, int(size)))
        acc += int(size)
    if acc != total:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: per-rank sizes {sizes} sum to {acc}, "
            f"not to the dimension {total}."
        )
    return out


@dataclass(frozen=True)
class Block:
    """One packed output of a fused parameter, on one rank.

    ``global_start`` is the offset in the FULL (unsharded) dimension -- the
    coordinate both sides of a pair share.  ``dev_row`` is the offset in THIS
    rank's storage.  They are different numbers on every rank but rank 0, and
    confusing them is defect #1 in this module's docstring.
    """

    block: int
    global_start: int
    dev_row: int
    size: int

    @property
    def global_end(self) -> int:
        return self.global_start + self.size


def device_block_offsets(
    output_sizes: Sequence[int],
    ratios: Optional[Sequence[int]],
    tp_size: int,
    units: Optional[int] = None,
    groups: Optional[int] = None,
) -> List[List[Block]]:
    """Per rank, the device row offset of every packed output.

    THE DEVICE LAW, ``layers/linear.py:1088-1101``::

        shard_offset = sum(tp_partition_size(sz, ...) for sz in output_sizes[:b])

    -- the prefix sum of THIS RANK'S per-output sizes.  The checkpoint law is
    the prefix sum of the FULL sizes, and it is what SECTION 1af's probe
    reports.  For ``in_proj_qkvz`` with ``output_sizes=[k, k, v, v]``
    (``models/qwen3_5.py:543``) the device rows are ``0 / g_k / 2*g_k /
    2*g_k + g_v`` with the rank's OWN ``g_k`` and ``g_v``.
    """
    per_block = [
        shard_offsets(int(sz), ratios, tp_size, units, groups) for sz in output_sizes
    ]
    full_prefix, acc = [], 0
    for sz in output_sizes:
        full_prefix.append(acc)
        acc += int(sz)
    out: List[List[Block]] = []
    for rank in range(tp_size):
        blocks, dev_row = [], 0
        for b, ranges in enumerate(per_block):
            start, size = ranges[rank]
            blocks.append(
                Block(
                    block=b,
                    global_start=full_prefix[b] + start,
                    dev_row=dev_row,
                    size=size,
                )
            )
            dev_row += size
        out.append(blocks)
    return out


# ---------------------------------------------------------------------------
# The inputs: one side's layout, one parameter's geometry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupLayout:
    """One of the two process groups, as the plan needs to see it.

    ``cards[n]`` is the NVML index rank ``n`` runs on.  Both groups receive the
    same ``CUDA_VISIBLE_DEVICES`` uuid string, so rank ``n`` of either group
    runs on ``cards[n]`` and a pair with ``src_rank == dst_rank`` is on-card by
    construction (spec §1.3).

    ``tp_size == 1`` with more than one rank is the PP form (group P on boot
    weg2sb4: ``--tp-size 1 --pp-size 3``): every rank holds WHOLE tensors, and
    which rank holds a given one is ``ParamGeom.stage``.  ``tp_size ==
    len(cards)`` is the TP form (group D).  Anything between is out of the V1
    scope this spec states as a hard limit and is refused rather than guessed.

    ``base`` is this group's first GLOBAL rank number, so the 6x6 byte matrix
    is indexed the same way in both directions.  The two groups' ranges must be
    DISJOINT: with both at the default 0 the matrix folds to 3x3, P rank n's
    sends land in D rank n's cell, and the symmetry check of spec §3.3 then
    passes on a folded matrix.

    ``ratios`` is the base shard vector; ``family_ratios`` the named ones the
    fork installs beside it (``distributed/utils.py:116``).  Which of the two a
    parameter gets is ``ratios_for``, and it is not one law but two.
    """

    name: str
    cards: Tuple[int, ...]
    tp_size: int
    ratios: Optional[Sequence[int]] = None
    base: int = 0
    family_ratios: Optional[Dict[str, Sequence[int]]] = None

    @property
    def n_ranks(self) -> int:
        return len(self.cards)

    def card_of(self, rank: int) -> int:
        return int(self.cards[rank])

    def ratios_for(self, family: Optional[str]) -> Optional[Sequence[int]]:
        """The vector that shards a parameter of ``family`` in THIS group.

        Two laws, and reading the second as the first is silent wrongness on
        the largest single parameter of the model:

        * a NAMED family falls back to the base vector when it has none of its
          own -- ``distributed/utils.py:1539 tp_partition_sizes`` ->
          ``:166 get_tp_partition_ratios``;
        * the VOCABULARY does not -- ``:1585 tp_vocab_ratios``, and a UNIFORM
          vocab vector IS the even split and reports as inactive, which keeps
          the classic path byte-identical.  ``VocabParallelEmbedding`` /
          ``ParallelLMHead`` (``layers/vocab_parallel_embedding.py:299-304``)
          therefore keep the even split under a plain uneven-TP plan while the
          MLP of the same model follows the ratio: ONE plan, TWO laws, which is
          why the vector cannot live on the group alone.
        """
        fams = self.family_ratios or {}
        if family == VOCAB_FAMILY:
            vec = fams.get(VOCAB_FAMILY)
            if not vec or len(vec) != self.tp_size or len(set(vec)) == 1:
                return None
            return list(vec)
        if family is not None:
            vec = fams.get(family)
            if vec:
                return list(vec)
        return self.ratios

    def validate(self) -> None:
        """Refuse a layout whose shard arithmetic could only be guessed at."""
        if self.n_ranks <= 0:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: group {self.name!r} has no cards."
            )
        if self.tp_size not in (1, self.n_ranks):
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: group {self.name!r} has tp_size "
                f"{self.tp_size} over {self.n_ranks} ranks. The V1 scope is "
                f"pure TP (tp_size == ranks) or the PP form (tp_size == 1)."
            )
        if int(self.base) < 0:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: group {self.name!r} base "
                f"{self.base} is not a global rank number."
            )
        vectors = [("base", self.ratios)] + sorted((self.family_ratios or {}).items())
        for label, vec in vectors:
            if vec and len(vec) != self.tp_size:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: group {self.name!r} carries a "
                    f"{label} shard vector {list(vec)} of {len(vec)} entries "
                    f"for {self.tp_size} ranks. The fork treats a length "
                    f"mismatch as 'does not apply' and falls back to the even "
                    f"split (distributed/utils.py:1539); here it means two "
                    f"groups' plans were mixed, and it is refused (#1275)."
                )

    @classmethod
    def from_installed(
        cls,
        name: str,
        cards: Sequence[int],
        tp_size: int,
        base: int = 0,
    ) -> GroupLayout:
        """This process's INSTALLED shard plan as a layout.

        The producer of ``family_ratios``: S2/S3 read the plan here instead of
        re-deriving the two activation laws at their own call sites.
        """
        from sglang.srt.distributed.utils import (
            get_tp_partition_ratios,
            tp_vocab_ratios,
        )

        fams: Dict[str, Sequence[int]] = {}
        for fam in SHARD_FAMILIES:
            vec = (
                tp_vocab_ratios(int(tp_size))
                if fam == VOCAB_FAMILY
                else get_tp_partition_ratios(fam)
            )
            if vec and len(vec) == int(tp_size):
                fams[fam] = list(vec)
        base_vec = get_tp_partition_ratios(None)
        if base_vec and len(base_vec) != int(tp_size):
            # The fork's own activation law: a vector of another length does
            # not apply to this group (``distributed/utils.py:1539``).
            base_vec = None
        layout = cls(
            name=name,
            cards=tuple(int(c) for c in cards),
            tp_size=int(tp_size),
            ratios=list(base_vec) if base_vec else None,
            base=int(base),
            family_ratios=fams or None,
        )
        layout.validate()
        return layout


@dataclass(frozen=True)
class ParamGeom:
    """One logical parameter, in storage space, as both sides agree on it.

    ``rows_full``/``cols_full`` are the UNSHARDED storage extents; for the
    vocabulary ``rows_full`` is the PADDED extent, because that is what the
    sharded side partitions.  ``blocks`` names the packed outputs of a fused
    column-parallel parameter (``()`` = one block).

    ``pad_units`` is the number of trailing units of the sharded axis that are
    DECLARED pad: D pads the vocabulary to ``64 * tp`` while P at tp=1 pads to
    64 and 248320 is already a multiple of 64, so 128 rows exist on no card and
    in no checkpoint and are ZEROFILL by design (spec §2.2).  ``src_extent``
    additionally clips how much of the rest actually has a source; anything
    between it and the pad is a GAP and is refused by name (W74), never zeroed.

    ``family`` selects the shard vector (``GroupLayout.ratios_for``): the
    vocabulary keeps the even split under an uneven base plan while the MLP of
    the same model follows the ratio, so the vector cannot be a property of the
    group alone.  ``None`` is the base vector, which is what every class whose
    layer passes no ``tp_family`` gets.

    ``stage`` is the rank of a ``tp_size == 1`` group that holds this parameter
    (``None`` = every rank of such a group holds it, which is the vision tower
    in D-wake, spec §1.4).  ``dst_widths``/``dst_extents`` exist so a test can
    seed a shard vector the geometry could never produce.
    """

    name: str
    tag: str
    shard_axis: int
    rows_full: int
    cols_full: int
    itemsize: int
    blocks: Tuple[int, ...] = ()
    units: Optional[int] = None
    groups: Optional[int] = None
    family: Optional[str] = None
    pad_units: int = 0
    src_extent: Optional[int] = None
    stage: Optional[int] = None
    dst_widths: Optional[Sequence[int]] = None
    dst_extents: Optional[Sequence[int]] = None

    def replace(self, **kw) -> ParamGeom:
        return _dc_replace(self, **kw)

    def validate(self) -> None:
        """Refuse a geometry that would mis-plan in SILENCE.

        Each of these has a concrete wrong plan behind it, not a type error:
        ``pad_units`` past the axis makes ``content_units`` negative, the copy
        loop never runs and the WHOLE parameter becomes ZEROFILL -- the
        destination then serves zeroed weights with no refusal anywhere;
        blocks that do not sum to the axis put the block coordinates and the
        pad in different spaces; a column shard with packed outputs would be
        routed through ``device_block_offsets``, whose ``dev_row`` is a ROW
        prefix sum.
        """
        if self.rows_full <= 0 or self.cols_full <= 0 or self.itemsize <= 0:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {self.name}: extents "
                f"({self.rows_full}, {self.cols_full}) itemsize "
                f"{self.itemsize} do not describe a tensor."
            )
        if self.shard_axis not in (ROWS, COLS, REPLICATED):
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {self.name}: shard axis "
                f"{self.shard_axis} is neither ROWS, COLS nor REPLICATED."
            )
        if self.blocks:
            if self.shard_axis != ROWS:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {self.name}: packed outputs "
                    f"{list(self.blocks)} on a non-ROWS shard axis. Spec §2.2 "
                    f"has no such class, and device_block_offsets' dev_row is "
                    f"a ROW prefix sum -- it would name the wrong coordinate "
                    f"space rather than fail."
                )
            if any(int(b) <= 0 for b in self.blocks):
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {self.name}: packed output "
                    f"sizes {list(self.blocks)} are not all positive."
                )
            if sum(int(b) for b in self.blocks) != self.shard_total:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {self.name}: packed outputs "
                    f"{list(self.blocks)} sum to "
                    f"{sum(int(b) for b in self.blocks)}, not to the sharded "
                    f"axis' {self.shard_total}."
                )
        if not 0 <= int(self.pad_units) <= self.shard_total:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {self.name}: pad_units "
                f"{self.pad_units} against an axis of {self.shard_total}. A "
                f"pad past the axis turns the whole parameter into ZEROFILL."
            )
        if self.src_extent is not None and not (
            0 <= int(self.src_extent) <= self.content_units
        ):
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {self.name}: src_extent "
                f"{self.src_extent} against {self.content_units} non-pad "
                f"units."
            )
        if (
            self.dst_widths is not None
            and sum(int(w) for w in self.dst_widths) != self.shard_total
        ):
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {self.name}: seeded widths "
                f"{list(self.dst_widths)} sum to "
                f"{sum(int(w) for w in self.dst_widths)}, not to the sharded "
                f"axis' {self.shard_total}."
            )

    @classmethod
    def of(
        cls,
        tensor,
        *,
        name: str,
        tag: str,
        shard_axis: int,
        shard_total: int,
        shard_dim: Optional[int] = None,
        **kw,
    ) -> ParamGeom:
        """The geometry of a LIVE parameter -- the seam S2's walk hands over.

        Everything but the sharded axis' FULL extent (which one rank cannot
        see) is read from ``stride()``/``element_size()`` through
        ``StorageGeom``, never from ``shape``: a ``.t()`` view reports
        ``[K, N_local]`` over storage ``[N_local, K]``, so a geometry taken
        from the logical shape is transposed-wrong for every column-parallel
        class (spec §2.4 rule 2).  ``shard_total`` is in the units the storage
        axis counts in -- storage ROWS for a ROWS shard, storage COLUMNS for a
        COLS shard.
        """
        ndim = int(tensor.dim()) if hasattr(tensor, "dim") else len(tensor.shape)
        if ndim > 2 and (shard_dim is None or int(shard_dim) not in (0, ndim - 1)):
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {name} is {ndim}-D and its shard "
                f"axis is not one a (rows, cols, pitch) triple can name. A "
                f"contiguous block flattens to storage rows only for a LEADING "
                f"axis shard (conv1d's [C, 1, K]: pass shard_dim=0) or a "
                f"trailing one; an expert-major MoE weight [E, N_local, K] "
                f"sharded on N is E strided bands. TODO(S2): that class needs "
                f"its own descriptor kind before it can be exchanged."
            )
        live = StorageGeom.of(tensor)
        if shard_axis == COLS:
            rows_full, cols_full = live.rows, int(shard_total)
        elif shard_axis == ROWS:
            rows_full, cols_full = int(shard_total), live.cols
        else:
            rows_full, cols_full = live.rows, live.cols
        geom = cls(
            name=name,
            tag=tag,
            shard_axis=shard_axis,
            rows_full=rows_full,
            cols_full=cols_full,
            itemsize=live.itemsize,
            **kw,
        )
        geom.validate()
        return geom

    @property
    def shard_total(self) -> int:
        """The sharded axis' full extent, padding included."""
        return self.cols_full if self.shard_axis == COLS else self.rows_full

    @property
    def content_units(self) -> int:
        """The extent that is not declared pad."""
        return self.shard_total - int(self.pad_units)

    @property
    def source_units(self) -> int:
        """How much of the axis actually has a source."""
        return self.content_units if self.src_extent is None else int(self.src_extent)


# ---------------------------------------------------------------------------
# The descriptor.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XchgDesc:
    """Spec §2.1.  Every offset, pitch and run is in BYTES.

    ``param_name`` is the identity -- never an allocation index, because the
    C++ side iterates an ``unordered_map`` (``core.cpp:206-222``) whose order
    differs between the two processes.

    ``src_ptr``/``dst_ptr`` are ``data_ptr()`` values, boot-stable because
    ``cuMemAddressReserve`` appears only in ``malloc`` and pause/resume re-map
    at the same address (spec §1.1).  They are ``None`` on the side a rank does
    not own, and the plan id never includes them, so it can be compared across
    processes.
    """

    tag: str
    src_rank: int  # -1 = no source (ZEROFILL)
    dst_rank: int
    param_name: str
    kind: str
    nbytes: int
    rows: int
    run_bytes: int
    spitch: int
    dpitch: int
    src_off: int = 0
    dst_off: int = 0
    pieces: int = 1
    src_ptr: Optional[int] = None
    dst_ptr: Optional[int] = None

    def replace(self, **kw) -> XchgDesc:
        return _dc_replace(self, **kw)

    @property
    def on_card(self) -> bool:
        return self.src_rank == self.dst_rank

    def key(self) -> tuple:
        """The pointer-free identity that goes into the plan id."""
        return (
            self.tag,
            self.param_name,
            self.src_rank,
            self.dst_rank,
            self.kind,
            self.nbytes,
            self.rows,
            self.run_bytes,
            self.spitch,
            self.dpitch,
            self.src_off,
            self.dst_off,
            self.pieces,
        )


# ---------------------------------------------------------------------------
# Waves.
# ---------------------------------------------------------------------------


def derive_waves(
    family_tags: Sequence[str],
    tag_cards: Dict[str, Tuple[int, ...]],
    cards: Sequence[int],
) -> List[List[str]]:
    """Partition the weights family into waves that each free bytes on EVERY
    card (spec §1.2).

    Each chunk tag is a LAYER BAND, so under a PP cut it names the one or two
    stages its range overlaps (``weg2_memory_saver.py:1824 chunk_tag_cards``);
    the base tag spans everything and closes the last wave.  A TP group has no
    layer split and returns an EMPTY map, which is read as "uniform", never as
    "no bytes".

    ON THE WORD "FEWEST".  The spec says "the fewest waves such that every wave
    frees bytes on every card".  Read literally that is ONE wave -- the whole
    family at once trivially frees on every card -- which is today's shape and
    the arm §5.5 prices WORST.  The derivation that reproduces the spec's own
    table (and refuses nine) is the MAXIMUM wave count under the every-card
    constraint, bounded above by the card the fewest tags touch.  More waves
    lower the residency peak, so maximising is what the schedule is for.
    """
    tags = list(family_tags)
    if not tags:
        return []
    base, chunk_tags = tags[-1], tags[:-1]
    wanted = {int(c) for c in cards}

    if not tag_cards:
        # A TP group has no layer split and returns an EMPTY map, and its own
        # docstring says the caller reads that as UNIFORM
        # (``weg2_memory_saver.py:1824``).  Read per tag instead, every tag
        # independently covers every card, the loop below admits exactly one
        # tag per wave, and the schedule becomes 8 chunk waves + the base --
        # the nine-wave arm §1.2 refuses on MEASURED transport grounds
        # (+25.8 %/+36.8 % P->D, +29.8 %/+30.0 % D->P).  Uniform means one.
        return [list(chunk_tags) + [base]]

    def cards_of(tag: str) -> set:
        return {int(c) for c in tag_cards.get(tag, tuple(wanted))}

    waves: List[List[str]] = []
    remaining = list(chunk_tags)
    while remaining:
        covered: set = set()
        wave: List[str] = []
        rest: List[str] = []
        for tag in remaining:
            if cards_of(tag) - covered:
                wave.append(tag)
                covered |= cards_of(tag)
            else:
                rest.append(tag)
        if covered != wanted:
            # This wave cannot free on every card on its own.  Only the LAST
            # wave may look like that, because the base tag spans every card
            # and closes it -- so everything left joins here rather than
            # shipping a wave that starves a card.
            wave = wave + rest
            rest = []
        waves.append(wave)
        remaining = rest
    if not waves:
        waves = [[]]
    waves[-1].append(base)
    return waves


# ---------------------------------------------------------------------------
# Pieces: histogram, id, coalescing.
# ---------------------------------------------------------------------------

_HIST_ORDER = ("<4K", "<64K", "<256K", "<2M", "<8M", "<32M", ">=32M")


def _histogram_bucket(nbytes: int) -> str:
    for limit, label in (
        (4 << 10, "<4K"),
        (64 << 10, "<64K"),
        (256 << 10, "<256K"),
        (2 << 20, "<2M"),
        (8 << 20, "<8M"),
        (32 << 20, "<32M"),
    ):
        if nbytes < limit:
            return label
    return ">=32M"


def piece_histogram(descs: Sequence[XchgDesc]) -> Dict[str, int]:
    """Piece-size histogram, printed on ``WEG2-XCHG-PLAN``.

    R3 (piece granularity) is UNMEASURED -- E4 bounds only >= 2 MiB async
    (<= 1 %) and 256 KiB async (4.6-6.4 %) -- so the distribution is published
    rather than asserted, and the denominator is every descriptor in the plan.
    """
    out = {k: 0 for k in _HIST_ORDER}
    for d in descs:
        out[_histogram_bucket(d.nbytes)] += 1
    return {k: v for k, v in out.items() if v}


def plan_id(descs: Sequence[XchgDesc], waves: Sequence[Sequence[str]] = ()) -> str:
    """A 12-hex digest over the descriptor GEOMETRY and the wave partition,
    pointers excluded.

    Two processes hold the same tensors at different addresses, so an id that
    moved with the address could never be compared -- and comparing it is the
    whole job (W68: "the plan hash != the front's").

    TAKE THE **RAW** LIST.  ``XchgDesc.key()`` excludes the pointers, but
    ``coalesce`` CONSUMES them: which pieces merge is a function of the pointer
    table, so a digest over the merged list is address-dependent after all.  A
    rank is producer or consumer, never both (spec §1.1), and the front holds
    no table at all -- three parties, three different merges, one comparison
    that could never succeed.

    The WAVE PARTITION is folded in because two partitions can leave the
    descriptor order untouched, and a schedule disagreement is one of the three
    things W68 names (spec §3.3).
    """
    h = hashlib.blake2b(digest_size=6)
    for wave in waves:
        h.update(repr(tuple(wave)).encode("utf-8"))
        h.update(b"|")
    for d in descs:
        h.update(repr(d.key()).encode("utf-8"))
    return h.hexdigest()


def _side_addr(d: XchgDesc, side: str, delta: int) -> tuple:
    """A comparable address on one side of a descriptor, ``delta`` bytes in.

    With a pointer it is the address; without one, only offsets WITHIN the same
    parameter can be compared, so the name is part of the key.  The two spaces
    never compare equal, which is exactly ``_adjacent``'s rule that a mixed
    pair (one side pointered, one not) is not adjacent.
    """
    ptr, off = (d.src_ptr, d.src_off) if side == "src" else (d.dst_ptr, d.dst_off)
    if ptr is not None:
        return ("p", ptr + off + delta)
    return ("n", d.param_name, off + delta)


def _span_key(d: XchgDesc, delta: int) -> tuple:
    """The pair identity plus both sides' addresses -- what ``_adjacent``
    compares, as one hashable key."""
    return (
        d.tag,
        d.src_rank,
        d.dst_rank,
        _side_addr(d, "src", delta),
        _side_addr(d, "dst", delta),
    )


def _adjacent(a: XchgDesc, b: XchgDesc, side: str) -> bool:
    """True when ``b`` starts exactly where ``a`` ends on ``side``.

    With real ``data_ptr()`` values this sees adjacency ACROSS parameters, which
    is where most of the coalescing win is (the TMS arena hands out sequential
    addresses).  Without them it can only compare within one parameter, and it
    says so by refusing rather than by assuming.
    """
    if side == "src":
        pa, pb, oa, ob = a.src_ptr, b.src_ptr, a.src_off, b.src_off
    else:
        pa, pb, oa, ob = a.dst_ptr, b.dst_ptr, a.dst_off, b.dst_off
    if pa is not None and pb is not None:
        return pa + oa + a.nbytes == pb + ob
    if pa is None and pb is None and a.param_name == b.param_name:
        return oa + a.nbytes == ob
    return False


def coalesce(
    descs: Sequence[XchgDesc], floor: int = COALESCE_FLOOR_BYTES
) -> List[XchgDesc]:
    """Merge FLAT pieces that are adjacent in BOTH address spaces.

    21 of the 24 classes are contiguous row ranges within one tag, so this is
    nearly total; the three 2-D classes emit one ``Memcpy2D`` each and are never
    merged, because their pitches are their shape.

    THE MERGE CONDITION IS ``either side is under the floor``, not "the run is
    under the floor".  Stopping at the floor leaves a TAIL: 12 adjacent 256 KiB
    pieces would close a 2 MiB run and then strand a 1 MiB remainder, which is
    exactly the sub-floor piece the coalescer exists to remove.  Absorbing a
    small neighbour into a large run costs one issue and removes one.
    """
    out: List[XchgDesc] = []
    for d in descs:
        if d.kind != FLAT or not out:
            out.append(d)
            continue
        prev = out[-1]
        if (
            prev.kind == FLAT
            and prev.tag == d.tag
            and prev.src_rank == d.src_rank
            and prev.dst_rank == d.dst_rank
            and (prev.nbytes < floor or d.nbytes < floor)
            and _adjacent(prev, d, "src")
            and _adjacent(prev, d, "dst")
        ):
            out[-1] = prev.replace(
                nbytes=prev.nbytes + d.nbytes,
                run_bytes=prev.run_bytes + d.run_bytes,
                pieces=prev.pieces + d.pieces,
            )
        else:
            out.append(d)
    return out


def unmergeable_below_floor(
    descs: Sequence[XchgDesc], floor: int = COALESCE_FLOOR_BYTES
) -> List[XchgDesc]:
    """The pieces that stayed under the floor with no neighbour left to merge.

    A LITERAL "no piece below the floor survives" is unreachable and would be a
    test that can only pass by lying: ``A_log`` and ``dt_bias`` are 84 bytes on
    this model and sit in their own allocations.  What IS checkable is that no
    small piece kept a mergeable neighbour, which is what this reports and what
    the histogram publishes.
    """
    starts: Dict[tuple, XchgDesc] = {}
    ends: Dict[tuple, XchgDesc] = {}
    for d in descs:
        if d.kind != FLAT:
            continue
        starts.setdefault(_span_key(d, 0), d)
        ends.setdefault(_span_key(d, d.nbytes), d)
    left = []
    for d in descs:
        if d.kind != FLAT or d.nbytes >= floor:
            continue
        after = starts.get(_span_key(d, d.nbytes))
        before = ends.get(_span_key(d, 0))
        if (after is not None and after is not d) or (
            before is not None and before is not d
        ):
            continue
        left.append(d)
    return left


# ---------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XchgPlan:
    """One direction's plan.

    ``byte_matrix`` is the 6x6 GLOBAL-rank matrix Gate 0 compares cell by cell
    (spec §3.3); ``tag_bytes`` is the per-tag total it compares against
    ``tms_tag_bytes`` (spec §1.3 step 9) -- both are produced here so S3 does
    not re-derive them from the descriptor list a second time.
    ``skipped_tags`` names every parameter the plan deliberately did not carry,
    counted: the draft/MTP family is never exchanged (spec §4.1) and a silent
    drop is indistinguishable from a typo.
    """

    descs: Tuple[XchgDesc, ...]
    raw_descs: Tuple[XchgDesc, ...]
    waves: Tuple[Tuple[str, ...], ...]
    byte_matrix: Tuple[Tuple[int, ...], ...]
    plan_id: str
    src_group: str
    dst_group: str
    tag_bytes: Tuple[Tuple[str, int], ...] = ()
    skipped_tags: Tuple[Tuple[str, int], ...] = ()

    @property
    def oncard_bytes(self) -> int:
        return sum(d.nbytes for d in self.descs if d.kind != ZEROFILL and d.on_card)

    @property
    def cross_bytes(self) -> int:
        return sum(d.nbytes for d in self.descs if d.kind != ZEROFILL and not d.on_card)

    def oncard_bytes_by_rank(self) -> Dict[int, int]:
        """The diagonal, PER CARD -- which is the denominator ``hop_ms`` has.

        ``oncard_bytes`` is the sum over all three cards and is what spec
        section 6/S6's ``oncard_gib=10.28`` names; the on-card LANE, however,
        runs once per card between two co-located processes, so a hop budget
        priced against the sum is priced against a lane that does not exist.
        S5-pre's own extrapolation ("10.28 GiB = 329 batches") took the sum for
        one lane; keeping both numbers, each with its denominator, is what
        stops that reading from being repeated.
        """
        out: Dict[int, int] = {}
        for d in self.descs:
            if d.kind != ZEROFILL and d.on_card:
                out[int(d.src_rank)] = out.get(int(d.src_rank), 0) + int(d.nbytes)
        return out

    @property
    def zerofill_bytes(self) -> int:
        return sum(d.nbytes for d in self.descs if d.kind == ZEROFILL)

    @property
    def min_piece_bytes(self) -> int:
        moved = [d.nbytes for d in self.descs if d.kind != ZEROFILL]
        return min(moved) if moved else 0

    def log_line(self, direction: Optional[str] = None) -> str:
        """The acceptance line of spec §6/S1.  Every figure is computed from
        THIS plan; none is quoted from the spec's projection.

        ``unmergeable`` is beside ``min_piece_mib`` because the spec's literal
        "no piece below 2 MiB survives coalescing" is unreachable (``A_log`` is
        84 bytes and has no neighbour) and this slice enforces a different
        invariant -- "no small piece kept a mergeable neighbour".  A published
        substitute is checkable; an argued one is not, and ``min_piece_mib``
        alone cannot tell the two apart.
        """
        if direction is None:
            direction = f"{self.src_group}2{self.dst_group}"
        hist = "/".join(f"{k}:{v}" for k, v in piece_histogram(self.descs).items())
        gib = float(1 << 30)
        # S5: the diagonal's batch count is known HERE, at plan time, so the
        # slot size is a decision this plan makes and the priced hop is a
        # number the boot can hold against the measured ``hop_ms=`` on the
        # WEG2-XCHG-ONCARD line.  Priced for the BUSIEST card, named, because
        # the lane is per card (see oncard_bytes_by_rank).
        from sglang.srt.weg2 import weight_exchange_transport as _tp

        by_rank = self.oncard_bytes_by_rank()
        hot_rank, hot_bytes = (
            max(by_rank.items(), key=lambda kv: kv[1]) if by_rank else (-1, 0)
        )
        oncard = _tp.plan_oncard_slot_bytes(hot_bytes)
        return (
            "WEG2-XCHG-PLAN "
            f"dir={direction} waves={len(self.waves)} descs={len(self.raw_descs)} "
            f"coalesced={len(self.descs)} "
            f"min_piece_mib={self.min_piece_bytes / (1 << 20):.6f} "
            f"unmergeable={len(unmergeable_below_floor(self.descs))} "
            f"bytes_gib={(self.oncard_bytes + self.cross_bytes) / gib:.2f} "
            f"oncard_gib={self.oncard_bytes / gib:.2f} "
            f"cross_gib={self.cross_bytes / gib:.2f} "
            f"zerofill_mib={self.zerofill_bytes / (1 << 20):.2f} "
            f"hist={hist} "
            f"oncard_priced_rank={hot_rank} "
            f"oncard_priced_gib={hot_bytes / gib:.2f} "
            f"{oncard.tokens()} "
            f"plan_id={self.plan_id}"
        )


def emit_plan_line(plan: XchgPlan, direction: Optional[str] = None, logger=None) -> str:
    """Log the acceptance line and return it.

    One emitter, so S2 and S6 do not each grow their own format string and drift
    apart the way the front's docstring drifted from its own constant.
    """
    line = plan.log_line(direction)
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)
    logger.info("%s", line)
    return line


def _blocks_of(geom: ParamGeom, layout: GroupLayout, is_dst: bool) -> List[List[Block]]:
    """Per rank of ``layout``, the blocks that rank holds.  An empty list means
    the rank holds nothing of this parameter."""
    if layout.tp_size == 1 and layout.n_ranks >= 1:
        # PP form (or a single-rank group): whole tensors, one holder each.
        if geom.stage is not None and not 0 <= int(geom.stage) < layout.n_ranks:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {geom.name} names stage "
                f"{geom.stage} in group {layout.name!r}, which has "
                f"{layout.n_ranks} ranks. No rank would match, every block "
                f"list would come back empty and the parameter would vanish "
                f"from the plan with no refusal at all -- fail-open in the "
                f"direction where this group is the DESTINATION (spec §1.4)."
            )
        holders = (
            list(range(layout.n_ranks)) if geom.stage is None else [int(geom.stage)]
        )
        sizes = list(geom.blocks) if geom.blocks else [geom.content_units]
        full, acc = [], 0
        for b, sz in enumerate(sizes):
            full.append(Block(block=b, global_start=acc, dev_row=acc, size=int(sz)))
            acc += int(sz)
        return [list(full) if r in holders else [] for r in range(layout.n_ranks)]
    if layout.tp_size != layout.n_ranks:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: group {layout.name!r} has tp_size "
            f"{layout.tp_size} over {layout.n_ranks} ranks. The V1 scope is "
            f"pure TP (tp_size == ranks) or the PP form (tp_size == 1); a mixed "
            f"form is refused rather than guessed."
        )
    if geom.shard_axis == REPLICATED:
        total = geom.content_units
        return [
            [Block(block=0, global_start=0, dev_row=0, size=total)]
            for _ in range(layout.n_ranks)
        ]
    if is_dst and geom.dst_widths is not None:
        if len(geom.dst_widths) != layout.n_ranks:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {geom.name}: {len(geom.dst_widths)} "
                f"seeded widths for {layout.n_ranks} ranks."
            )
        out, acc = [], 0
        for w in geom.dst_widths:
            out.append([Block(block=0, global_start=acc, dev_row=0, size=int(w))])
            acc += int(w)
        return out
    # THE FAMILY AXIS.  The real shard boundary is a function of (total,
    # tp_size, units, FAMILY, groups) -- ``distributed/utils.py:1539`` -- and
    # the vocabulary's vector never falls back to the base one (``:1585``).
    # One vector per group would plan ``embed_tokens``/``lm_head`` at ratio
    # offsets under the standing uneven-TP form while the hardware keeps the
    # even split, on both sides identically.
    ratios = layout.ratios_for(geom.family)
    if geom.blocks:
        return device_block_offsets(
            geom.blocks, ratios, layout.tp_size, geom.units, geom.groups
        )
    ranges = shard_offsets(
        geom.shard_total, ratios, layout.tp_size, geom.units, geom.groups
    )
    return [
        [Block(block=0, global_start=start, dev_row=0, size=size)]
        for start, size in ranges
    ]


def _pick_source(
    candidates: Sequence[Tuple[int, Block]], d_rank: int
) -> Tuple[int, Block]:
    """Prefer the CO-LOCATED source: rank ``n`` of either group runs on
    ``cards[n]`` (checked once per plan by ``_check_cards``), so a same-index
    pair crosses no link at all (spec §2.2 -- the replicated classes are
    on-card only).  Otherwise the lowest rank, so the choice is deterministic
    on every rank that derives it."""
    for s_rank, block in candidates:
        if s_rank == d_rank:
            return s_rank, block
    return min(candidates, key=lambda c: c[0])


def _live_storage(
    geom_of, layout: GroupLayout, rank: int, geom: ParamGeom, blocks, strided: bool
) -> Optional[StorageGeom]:
    """This rank's tensor as the HARDWARE has it, or ``None`` when the caller
    has no table (the hermetic path, and every rank that holds nothing).

    ``StorageGeom`` is the module's answer to spec §2.4 rule 2, and a rule that
    only the tests apply is not a rule: with no production caller a pitch taken
    from ``shape`` instead of ``stride()`` leaves every fixture green, because
    every fixture is contiguous.  Production passes ``param.data_ptr()``'s own
    tensor here (S3's pointer table walks the same parameters).
    """
    if geom_of is None or not blocks:
        return None
    live = geom_of(layout.name, rank, geom.name)
    if live is None:
        return None
    if not isinstance(live, StorageGeom):
        live = StorageGeom.of(live)
    if live.itemsize != int(geom.itemsize):
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: {geom.name} on {layout.name!r} rank "
            f"{rank}: the plan says itemsize {geom.itemsize}, the live tensor "
            f"says {live.itemsize}. Weight scales are FP32 on device and BF16 "
            f"in the checkpoint (spec §2.4 rule 3), so this is the exact shape "
            f"in which every offset is wrong by a constant factor."
        )
    fixed_live = live.rows if strided else live.cols
    fixed_plan = int(geom.rows_full) if strided else int(geom.cols_full)
    if fixed_live != fixed_plan:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: {geom.name} on {layout.name!r} rank "
            f"{rank}: the UNSHARDED axis is {fixed_plan} in the plan and "
            f"{fixed_live} in the live tensor's storage."
        )
    return live


def _shape_piece(
    strided: bool,
    contiguous: bool,
    span: int,
    cols: int,
    itemsize: int,
    rows: int,
    s_dev: int,
    d_dev: int,
    s_pitch: int,
    d_pitch: int,
) -> Dict[str, int]:
    """One piece's BYTE geometry, in the three shapes a copy can take.

    A row-sharded class is FLAT only while BOTH sides' rows are packed at the
    row width.  A destination whose rows sit in a wider arena has a pitch, and
    then every row offset is a multiple of the PITCH -- reading it as the row
    width lands each row short by the padding, silently.
    """
    if strided:
        return dict(
            kind=STRIDED2D,
            nbytes=rows * span * itemsize,
            rows=rows,
            run_bytes=span * itemsize,
            spitch=s_pitch * itemsize,
            dpitch=d_pitch * itemsize,
            src_off=s_dev * itemsize,
            dst_off=d_dev * itemsize,
        )
    if contiguous:
        nbytes = span * cols * itemsize
        return dict(
            kind=FLAT,
            nbytes=nbytes,
            rows=1,
            run_bytes=nbytes,
            spitch=0,
            dpitch=0,
            src_off=s_dev * cols * itemsize,
            dst_off=d_dev * cols * itemsize,
        )
    return dict(
        kind=STRIDED2D,
        nbytes=span * cols * itemsize,
        rows=span,
        run_bytes=cols * itemsize,
        spitch=s_pitch * itemsize,
        dpitch=d_pitch * itemsize,
        src_off=s_dev * s_pitch * itemsize,
        dst_off=d_dev * d_pitch * itemsize,
    )


def _emit(
    geom: ParamGeom,
    src: GroupLayout,
    dst: GroupLayout,
    ptr_of: Optional[Callable[[str, int, str], Optional[int]]],
    geom_of: Optional[Callable[[str, int, str], object]] = None,
) -> List[XchgDesc]:
    """Every descriptor of one parameter: intersect the two sides' ranges in
    GLOBAL coordinates, then translate each overlap into both sides' own device
    coordinates."""
    src_blocks = _blocks_of(geom, src, is_dst=False)
    dst_blocks = _blocks_of(geom, dst, is_dst=True)
    source_units = geom.source_units
    content_units = geom.content_units
    itemsize = int(geom.itemsize)
    strided = geom.shard_axis == COLS
    rows = int(geom.rows_full) if strided else 1
    cols = int(geom.cols_full)

    def ptr(group: str, rank: int) -> Optional[int]:
        return None if ptr_of is None else ptr_of(group, rank, geom.name)

    src_live = [
        _live_storage(geom_of, src, r, geom, blocks, strided)
        for r, blocks in enumerate(src_blocks)
    ]
    dst_live = [
        _live_storage(geom_of, dst, r, geom, blocks, strided)
        for r, blocks in enumerate(dst_blocks)
    ]
    src_extent_of, src_pitch_of = [], []
    for r, blocks in enumerate(src_blocks):
        derived = sum(b.size for b in blocks)
        live = src_live[r]
        if live is not None:
            actual = live.cols if strided else live.rows
            if actual != derived:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {geom.name} src_rank={r}: the "
                    f"plan reads {derived} units from a source whose own "
                    f"storage holds {actual}. The shard vector and the "
                    f"hardware disagree."
                )
        src_extent_of.append(derived)
        src_pitch_of.append(
            live.pitch if live is not None else (derived if strided else cols)
        )

    out: List[XchgDesc] = []
    for d_rank, d_blocks in enumerate(dst_blocks):
        if not d_blocks:
            continue
        live = dst_live[d_rank]
        if geom.dst_extents is not None:
            d_extent = int(geom.dst_extents[d_rank])
        elif live is not None:
            # THE DESTINATION'S EXTENT COMES FROM THE HARDWARE.  Derived from
            # the same block list the descriptors come from, "the plan writes
            # past the destination's own storage" can only fire through a seed
            # -- i.e. never in production, which is the one place it matters.
            d_extent = live.cols if strided else live.rows
        else:
            d_extent = sum(b.size for b in d_blocks)
        d_pitch = live.pitch if live is not None else (d_extent if strided else cols)
        covered: List[Tuple[int, int]] = []
        for db in d_blocks:
            cursor = db.global_start
            stop_at = min(db.global_end, source_units)
            while cursor < stop_at:
                cands = [
                    (s_rank, sb)
                    for s_rank, blocks in enumerate(src_blocks)
                    for sb in blocks
                    if sb.block == db.block
                    and sb.global_start <= cursor < sb.global_end
                ]
                if not cands:
                    # A hole in the middle: advance to the next source start so
                    # the tiling check names the gap instead of looping.
                    nxt = [
                        sb.global_start
                        for blocks in src_blocks
                        for sb in blocks
                        if sb.block == db.block and sb.global_start > cursor
                    ]
                    cursor = min(nxt) if nxt else stop_at
                    continue
                s_rank, sb = _pick_source(cands, d_rank)
                end = min(stop_at, sb.global_end)
                span = end - cursor
                s_dev = sb.dev_row + (cursor - sb.global_start)
                d_dev = db.dev_row + (cursor - db.global_start)
                s_pitch = src_pitch_of[s_rank]
                out.append(
                    XchgDesc(
                        tag=geom.tag,
                        src_rank=s_rank,
                        dst_rank=d_rank,
                        param_name=geom.name,
                        src_ptr=ptr(src.name, s_rank),
                        dst_ptr=ptr(dst.name, d_rank),
                        **_shape_piece(
                            strided,
                            s_pitch == cols and d_pitch == cols,
                            span,
                            cols,
                            itemsize,
                            rows,
                            s_dev,
                            d_dev,
                            s_pitch,
                            d_pitch,
                        ),
                    )
                )
                covered.append((d_dev, d_dev + span))
                cursor = end
            # DECLARED pad: units past the content extent exist on no card and
            # in no checkpoint, so they are zeroed rather than fetched.
            pad_lo = max(db.global_start, content_units)
            if pad_lo < db.global_end:
                d_dev = db.dev_row + (pad_lo - db.global_start)
                span = db.global_end - pad_lo
                shape = _shape_piece(
                    strided,
                    d_pitch == cols,
                    span,
                    cols,
                    itemsize,
                    rows,
                    0,
                    d_dev,
                    d_pitch,
                    d_pitch,
                )
                shape.update(kind=ZEROFILL, spitch=0, src_off=0)
                out.append(
                    XchgDesc(
                        tag=geom.tag,
                        src_rank=-1,
                        dst_rank=d_rank,
                        param_name=geom.name,
                        dst_ptr=ptr(dst.name, d_rank),
                        **shape,
                    )
                )
                covered.append((d_dev, d_dev + span))
        _check_tiles(geom, d_rank, d_extent, covered)
    return out


def _check_tiles(
    geom: ParamGeom, d_rank: int, extent: int, covered: Sequence[Tuple[int, int]]
) -> None:
    """Every destination byte has exactly one source: no gap, no overlap, and
    nothing past the destination's own storage.  Units are units of the sharded
    axis, which is what both refusal messages name."""
    cursor = 0
    for lo, hi in sorted(covered):
        if lo > cursor:
            raise Weg2XchgSourceMissing(
                f"W74 Weg2XchgSourceMissing: {geom.name} dst_rank={d_rank}: "
                f"units [{cursor}, {lo}) of this rank's {extent} have no VRAM "
                f"source and no ZEROFILL descriptor."
            )
        if lo < cursor:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {geom.name} dst_rank={d_rank}: unit "
                f"range [{lo}, {hi}) overlaps a range already claimed up to "
                f"{cursor}; two sources for one destination byte."
            )
        cursor = hi
    if cursor > extent:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: {geom.name} dst_rank={d_rank}: the plan "
            f"writes {cursor} units into a destination whose own storage holds "
            f"{extent}. The shard vector and the hardware disagree."
        )
    if cursor < extent:
        raise Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing: {geom.name} dst_rank={d_rank}: units "
            f"[{cursor}, {extent}) have no VRAM source and no ZEROFILL "
            f"descriptor."
        )


def _check_cards(src: GroupLayout, dst: GroupLayout) -> None:
    """``on_card`` is ``src_rank == dst_rank``, which means ONE CARD only while
    both groups run rank ``n`` on ``cards[n]``.

    Both groups receive the same ``CUDA_VISIBLE_DEVICES`` uuid string (spec
    §1.3), so they do -- but that is an assumption until the two vectors are
    compared, and the consequence of it being wrong is not a wrong number: S4
    routes an ``on_card`` pair down the ``cudaIpc`` lane with no PCIe key, for
    a transfer that actually crosses a link.  Compared here, it is a theorem.
    """
    if src.n_ranks != dst.n_ranks or any(
        src.card_of(r) != dst.card_of(r) for r in range(src.n_ranks)
    ):
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: group {src.name!r} runs on cards "
            f"{list(src.cards)} and group {dst.name!r} on {list(dst.cards)}. "
            f"on_card is rank equality, which is a statement about ONE card "
            f"only while the two vectors agree rank by rank."
        )


def _check_bases(src: GroupLayout, dst: GroupLayout) -> None:
    """The two groups' GLOBAL rank ranges must be disjoint.

    With both at the default ``base=0`` the 6x6 matrix is 3x3: P rank ``n``'s
    sends and D rank ``n``'s land in the same cell, the symmetry check of spec
    §3.3 passes on a FOLDED matrix, and Gate 0's #802 discipline is defeated by
    a dataclass default.
    """
    lo_a, hi_a = int(src.base), int(src.base) + src.n_ranks
    lo_b, hi_b = int(dst.base), int(dst.base) + dst.n_ranks
    if lo_a < hi_b and lo_b < hi_a:
        raise Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: group {src.name!r} occupies global "
            f"ranks [{lo_a}, {hi_a}) and {dst.name!r} [{lo_b}, {hi_b}). The "
            f"base numbers overlap, so the 6x6 byte matrix folds and its "
            f"symmetry check passes on cells that hold two groups' bytes."
        )


def build_plan(
    inventory: Sequence[ParamGeom],
    src: GroupLayout,
    dst: GroupLayout,
    waves: Sequence[Sequence[str]],
    ptr_of: Optional[Callable[[str, int, str], Optional[int]]] = None,
    geom_of: Optional[Callable[[str, int, str], object]] = None,
    floor: int = COALESCE_FLOOR_BYTES,
    skip_tags: Sequence[str] = (),
) -> XchgPlan:
    """The plan for one direction.

    The inventory is walked in SORTED NAME ORDER, never in the order it arrives:
    the C++ side iterates an ``unordered_map`` (``core.cpp:206-222``,
    ``:316-333``) whose order differs between the two processes, and a plan that
    depended on it could not be compared across ranks at all.  Descriptors are
    then ordered by (wave, dst_rank, name, dst offset, src_rank), which is the
    order the transport issues them in and the order coalescing needs.

    ``geom_of(group, rank, name)`` returns the live tensor (or its
    ``StorageGeom``) so every pitch and every destination extent comes from the
    HARDWARE; without it the plan is derived arithmetic only, which is the
    hermetic path the tests use.

    NOTHING HERE MAY PASS BY BEING EMPTY.  An empty descriptor list has a fixed
    ``plan_id`` identical on all six ranks and an all-zero byte matrix, so
    every Gate 0 comparison succeeds and ``resume`` then leaves the destination
    with mapped-but-unfilled ACTIVE pages: the boot serves undefined weights,
    with six ranks in agreement.  Every tag in ``waves`` must produce at least
    one descriptor, every parameter must produce at least one, and a tag that
    is carried by no wave must be declared in ``skip_tags`` (the draft/MTP
    family, spec §4.1) rather than dropped.
    """
    src.validate()
    dst.validate()
    _check_cards(src, dst)
    _check_bases(src, dst)

    wave_of: Dict[str, int] = {}
    for w, wave in enumerate(waves):
        for tag in wave:
            if tag in wave_of:
                raise Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: tag {tag!r} appears in wave "
                    f"{wave_of[tag]} and wave {w}. The wave list must be a "
                    f"PERMUTATION of the weights family; last-write-wins would "
                    f"accept a schedule the front's own guard "
                    f"(front.py:2656-2663) refuses."
                )
            wave_of[tag] = w

    skip = {str(t) for t in skip_tags}
    emitted: Dict[str, int] = {tag: 0 for tag in wave_of}
    skipped: Dict[str, int] = {}
    raw: List[XchgDesc] = []
    for geom in sorted(inventory, key=lambda g: g.name):
        geom.validate()
        if geom.tag not in wave_of:
            if geom.tag not in skip:
                raise Weg2XchgSourceMissing(
                    f"W74 Weg2XchgSourceMissing: {geom.name} carries tag "
                    f"{geom.tag!r}, which no wave carries and which was not "
                    f"declared skippable. Dropping it would remove it from the "
                    f"descriptors AND from the byte matrix, so Gate 0's "
                    f"per-tag comparison could not see it either."
                )
            skipped[geom.tag] = skipped.get(geom.tag, 0) + 1
            continue
        descs = _emit(geom, src, dst, ptr_of, geom_of)
        if not descs:
            raise Weg2XchgSourceMissing(
                f"W74 Weg2XchgSourceMissing: {geom.name} (tag {geom.tag!r}) "
                f"produced no descriptor: no rank of group {dst.name!r} holds "
                f"any of it. A parameter that is in the inventory and in a "
                f"wave has to move."
            )
        emitted[geom.tag] += len(descs)
        raw.extend(descs)

    barren = sorted(tag for tag, n in emitted.items() if n == 0)
    if barren:
        raise Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing: wave tags {barren} produced no "
            f"descriptor at all. A wave that moves nothing is an agreement "
            f"between six ranks that no byte has to arrive."
        )
    if not raw:
        raise Weg2XchgSourceMissing(
            "W74 Weg2XchgSourceMissing: the plan is empty. Its plan_id is a "
            "fixed digest and its byte matrix is all zeros, so every Gate 0 "
            "comparison would succeed while the destination keeps whatever its "
            "remapped pages held."
        )

    raw.sort(
        key=lambda d: (
            wave_of[d.tag],
            d.dst_rank,
            d.param_name,
            d.dst_off,
            d.src_rank,
        )
    )
    merged = coalesce(raw, floor)

    n = max(src.base + src.n_ranks, dst.base + dst.n_ranks)
    matrix = [[0] * n for _ in range(n)]
    tag_bytes: Dict[str, int] = {}
    for d in merged:
        if d.kind == ZEROFILL:
            continue
        matrix[src.base + d.src_rank][dst.base + d.dst_rank] += d.nbytes
        tag_bytes[d.tag] = tag_bytes.get(d.tag, 0) + d.nbytes
    return XchgPlan(
        descs=tuple(merged),
        raw_descs=tuple(raw),
        waves=tuple(tuple(w) for w in waves),
        byte_matrix=tuple(tuple(row) for row in matrix),
        # The RAW list, not the merged one: which pieces merge depends on the
        # pointer table, and the two processes hold different tables while the
        # front holds none.
        plan_id=plan_id(raw, waves),
        src_group=src.name,
        dst_group=dst.name,
        tag_bytes=tuple(sorted(tag_bytes.items())),
        skipped_tags=tuple(sorted(skipped.items())),
    )


# ===========================================================================
# SECTION S2 -- COVERAGE ARMING (W84) AND THE DRAFT TAG (spec 6/S2, 4.1).
# ===========================================================================

MIB = 1024 * 1024


# ===========================================================================
# THE MODE.  Boot-scoped and region-scoped (spec section 1.1): one boolean,
# decided once, at model_runner.py's region().
# ===========================================================================

WEIGHT_SOURCE_RING = "ring"
WEIGHT_SOURCE_EXCHANGE = "exchange"
#: #1273 S5.  RING-AUTHORITATIVE: the ring refills every weight byte exactly as
#: it does today and the exchange runs beside it into buffers nothing reads.
#: ``exchange_armed()`` is FALSE here, deliberately -- that predicate gates the
#: region's ``enable_cpu_backup`` and the pause/resume shape, and the shadow
#: changes neither.
WEIGHT_SOURCE_SHADOW = "shadow"

#: MINIMAL INTERFACE, TODO(S6, spec section 6/S6): the user-facing flag is
#: ``--weg2-weight-source {ring|exchange}`` on ``server_args``, and the
#: launcher publishes it to BOTH groups' worker processes.  Until S6 lands that
#: flag, the mode is read from this env var -- the same channel the launcher
#: already uses for the chunk geometry (``SGLANG_WEG2_WEIGHT_CHUNK_LAYERS``,
#: weg2_memory_saver.py), so no new transport is invented here.  S6 replaces
#: the body of :func:`weight_source` with the server-arg read and adds the flag
#: to ``p_form_key``'s blacklist BY NAME (spec risk R8 / the #1275 admin-key
#: trap): a new flag not named there gives every boot its own form and W48 then
#: discards the source boot's weight statement every boot.
WEIGHT_SOURCE_ENV = "SGLANG_WEG2_WEIGHT_SOURCE"


def weight_source() -> str:
    """``ring`` (today, byte for byte), ``exchange`` or ``shadow``.

    Anything unrecognised is ``ring``: this reads an ENVIRONMENT variable, and
    the one direction a typo may not take is "arm something".
    """
    value = (os.environ.get(WEIGHT_SOURCE_ENV, "") or "").strip().lower()
    if value in (WEIGHT_SOURCE_EXCHANGE, WEIGHT_SOURCE_SHADOW):
        return value
    return WEIGHT_SOURCE_RING


def exchange_armed() -> bool:
    """Does the EXCHANGE own the weight bytes?  False under ``shadow``.

    The one predicate that decides ``enable_cpu_backup`` and therefore whether
    ``pause`` is a pure unmap.  Under ``shadow`` the answer is no -- the ring
    is authoritative -- and reading it as yes there would delete the very
    ground truth the shadow compares against.
    """
    return weight_source() == WEIGHT_SOURCE_EXCHANGE


def shadow_armed() -> bool:
    """Does the shadow run beside the ring?  Never implies :func:`exchange_armed`."""
    return weight_source() == WEIGHT_SOURCE_SHADOW


@contextmanager
def weight_source_for_test(source: str) -> Iterator[str]:
    """Set the mode for the duration of a block.  Tests only."""
    previous = os.environ.get(WEIGHT_SOURCE_ENV)
    os.environ[WEIGHT_SOURCE_ENV] = source
    try:
        yield source
    finally:
        if previous is None:
            os.environ.pop(WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[WEIGHT_SOURCE_ENV] = previous


# ===========================================================================
# S2 -- THE DRAFT TAG (spec section 4.1)
# ===========================================================================

RUNNER_SHAPE_REFUSAL_MARKER = "W76 Weg2XchgRunnerShapeUnknown"


class Weg2XchgRunnerShapeUnknown(RuntimeError):
    """W76 -- a secondary ModelRunner this predicate cannot classify.

    ``is_draft_worker`` is a CONSTRUCTION gate with several producers and only
    one of them holds draft weights (model_runner.py:514-521).  The tag switch
    of spec section 4.1 must therefore classify the runner, not read the gate;
    a shape it cannot classify is refused by name rather than guessed, because
    both guesses are wrong in a way that shows up only at the next flip:
    guessing DRAFT takes a full target model out of the weights family (never
    paused, never exchanged, permanently resident, no refusal names it), and
    guessing PRIMARY puts unsourced bytes into it (W74).

    W76 is free per the spec section 7 census (highest assigned W67, this
    branch) and per ``test_weg2_wcode_uniqueness_1263``.

    RENUMBERED BY THE serve-next5 REPLAY (2026-09-09): this branch enumerated
    its codes against base ``3ea18deb95`` (census maximum W50); the serve line
    had since claimed W51-W66, so all 16 moved to W67-W82 -- the first free
    numbers above the MERGED maximum -- with the exception NAMES unchanged.
    The argument below is the ORIGINAL enumeration and is kept as the reason
    the number was enumerated, not as the derivation of today's number.
    """


#: The four shapes a ModelRunner can have, as far as WHICH WEIGHTS IT HOLDS is
#: concerned.  This is the question ``is_draft_worker`` is repeatedly mistaken
#: for; naming the shapes is what makes the mistake un-writable.
SHAPE_PRIMARY = "primary"
SHAPE_DRAFT = "draft"
SHAPE_PHASE_FLIP_TP_STACK = "phase_flip_tp_stack"
SHAPE_DUAL_GROUP_LANE = "dual_group_lane"


@dataclass(frozen=True)
class RunnerShape:
    """The ModelRunner construction flags that decide which weights it holds.

    Read off the runner rather than passed as one boolean, because the boolean
    was the defect: ``is_draft_worker and not is_phase_flip_tp_stack`` subtracts
    ONE of the wrong producers and leaves the #274 dual-group lane -- built with
    ``is_draft_worker=True, is_phase_flip_tp_stack=False``
    (model_runner.py:486-489) and holding the assembled FULL-WIDTH TARGET hull
    (``build_lane_model``, :2496-2513) -- classified as a drafter.
    """

    is_draft_worker: bool
    is_phase_flip_tp_stack: bool = False
    is_dual_group_lane: bool = False
    speculative_configured: bool = False

    @classmethod
    def of(cls, runner: Any) -> RunnerShape:
        server_args = getattr(runner, "server_args", None)
        return cls(
            is_draft_worker=bool(getattr(runner, "is_draft_worker", False)),
            is_phase_flip_tp_stack=bool(
                getattr(runner, "is_phase_flip_tp_stack", False)
            ),
            is_dual_group_lane=bool(getattr(runner, "is_dual_group_lane", False)),
            speculative_configured=bool(
                getattr(server_args, "speculative_algorithm", None)
            ),
        )

    def as_fields(self) -> str:
        return (
            f"is_draft_worker={self.is_draft_worker} "
            f"is_phase_flip_tp_stack={self.is_phase_flip_tp_stack} "
            f"is_dual_group_lane={self.is_dual_group_lane} "
            f"speculative_configured={self.speculative_configured}"
        )


def classify_runner(shape: RunnerShape) -> Optional[str]:
    """Which weights this runner holds, or ``None`` when that is not derivable.

    The producers of ``is_draft_worker``, each subtracted BY NAME:

    * ``is_phase_flip_tp_stack`` -- #631's secondary TP decode stack, the full
      TARGET model rebuilt under the flip group's geometry;
    * ``is_dual_group_lane`` -- #274's lane, the assembled full-width TARGET
      hull (and, with ``is_dual_group_lane_draft``, its NEXTN head, whose vocab
      shells ALIAS the lane target's -- one allocation may not straddle two
      tags, so the whole lane stays in the family);
    * a speculative draft worker -- the only producer that holds draft weights,
      and it exists only when this boot carries a speculative config.

    ``None`` is a fourth producer this function has never heard of: it sets the
    construction gate, sets none of the known exclusions, and the boot carries
    no speculative config, so there is no shape to attribute it to.
    """
    if not shape.is_draft_worker:
        return SHAPE_PRIMARY
    if shape.is_phase_flip_tp_stack:
        return SHAPE_PHASE_FLIP_TP_STACK
    if shape.is_dual_group_lane:
        return SHAPE_DUAL_GROUP_LANE
    if shape.speculative_configured:
        return SHAPE_DRAFT
    return None


def runner_shape_refusal_message(shape: RunnerShape) -> str:
    return (
        f"{RUNNER_SHAPE_REFUSAL_MARKER}: this ModelRunner is constructed with "
        f"is_draft_worker=True but is neither the #631 phase-flip TP stack, nor "
        f"the #274 dual-group lane, nor a speculative draft worker (this boot "
        f"carries no speculative config): {shape.as_fields()}. Under "
        f"--weg2-weight-source exchange the weights region tag decides whether "
        f"these bytes are exchanged at all, and neither answer is safe for a "
        f"shape nobody classified -- the draft tag would take a full target "
        f"model out of the weights family, the base tag would put bytes with no "
        f"VRAM source into it (W74). Classify the new producer in "
        f"classify_runner() or run this boot with --weg2-weight-source ring."
    )


def weights_region_tag_for(shape: RunnerShape) -> str:
    """The tag ``model_runner``'s weights region is opened with.

    Group D's base tag carries the ENTIRE NEXTN draft runner -- measured
    1382/1311/1311 MiB on boot weg2sb4 (``Load weight end.
    type=Qwen3_5ForCausalLMMTP ... mem usage=1.35/1.28/1.28 GB``), of which the
    checkpoint's ``mtp.*`` term is only 0.396 GiB; the rest is embed/head
    shards the draft runner re-materialises.  Group P carries no
    ``--speculative-*`` in this form, so NONE of those bytes has a VRAM source
    on the other side.  Under ``exchange`` they get their own tag, which
    ``is_weights_family_tag`` does not match, so they are never in a leg, never
    in a census and never in a wave, and every REMAINING destination byte has a
    source -- which is what lets the ring go to zero rather than to MTP-only.

    Cost: 1.35/1.28/1.28 GiB permanently resident per card (spec section 5).

    TWO DEVIATIONS FROM THE SPEC'S PSEUDOCODE, both deliberate:

    1. **Gated on the mode.**  Spec 4.1 writes the tag switch unconditionally;
       spec 1.1 requires ``ring`` to stay "today, byte for byte".  An
       unconditional switch changes the ring arm too -- the drafter would stop
       being paused, moving ~1.3 GiB/card from the host ring into permanent
       residency on a path nobody asked to change.  So the switch is armed by
       :func:`exchange_armed`.
    2. **A classified shape, not ``is_draft_worker``.**  The spec names the
       construction gate; the tree names that gate a trap in its own words
       (model_runner.py:514-521), and subtracting only the phase-flip stack from
       it -- the first form of this function -- reproduced the trap one producer
       later for the #274 lane.  See :func:`classify_runner`.
    """
    kind = classify_runner(shape)
    if kind is None:
        if exchange_armed():
            raise Weg2XchgRunnerShapeUnknown(runner_shape_refusal_message(shape))
        # Under ``ring`` nothing changes for anybody, including a shape this
        # predicate cannot name: today's path, byte for byte.
        return GPU_MEMORY_TYPE_WEIGHTS
    if kind == SHAPE_DRAFT and exchange_armed():
        return GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    return GPU_MEMORY_TYPE_WEIGHTS


RESIDENT_LINE_PREFIX = "WEG2-XCHG-RESIDENT"


def resident_line(*, tag: str, mib: float, rank: int, mode: str) -> str:
    """The acceptance line for the tag this runner's weights actually live under.

    An OBSERVATION, not a restatement of a constant (refuter F1): the first form
    of this line hardcoded ``tag=weights_draft`` and then computed
    ``in_family`` from that literal, so ``in_family=no`` was a property of a
    string this module owns -- it could never print ``yes``, and it printed the
    same under ``ring``, where the drafter IS in the family.  The tag is now
    the one the region was opened with, ``mib`` is the saver's own per-tag
    census for THAT tag, and the mode is on the line, so the line distinguishes
    an armed exchange from an unarmed one.

    Per rank, because a rank knows its own card and no other: the spec's
    ``mib=1382/1311/1311`` is the operator's three-card read of three of these
    lines, not a line any single rank can emit truthfully.  Field order puts
    ``tag= mib= in_family=`` adjacent, which is the adjacency the spec greps.
    """
    return (
        f"{RESIDENT_LINE_PREFIX} tag={tag} mib={float(mib):.1f} "
        f"in_family={'yes' if is_weights_family_tag(tag) else 'no'} "
        f"rank={int(rank)} mode={mode}"
    )


def roll_forward_weights_tag(*, has_draft_shard: bool) -> Optional[str]:
    """The tag W73's roll-forward reload may open its ONE region with.

    ``None`` means REFUSE.  ``_weg2_wake_reload_weights``
    (weight_updater.py:531-706) refills BOTH shards of a process through ONE
    ``update_weights_from_disk`` inside ONE region -- the code says so at
    :576-597 and refuses any arrangement where the two shards disagree.  Under
    ``exchange`` the two shards want two different region tags (the drafter's
    is out of the family by construction), and one region carries one tag, so
    there is no tag that is right for both: the base tag re-tags the drafter's
    repacked parameters back into the weights family and the next flip then has
    a destination range with no VRAM source (W74), while the draft tag does the
    same thing to the main shard in the other direction.

    Refusing is the only answer this slice can give.  Making the roll-forward
    work under ``exchange`` means one region per runner, which is a change to
    the reload's shape and belongs to S6 with W73 (spec section 3.6).
    """
    if not exchange_armed():
        return GPU_MEMORY_TYPE_WEIGHTS
    if has_draft_shard:
        return None
    return GPU_MEMORY_TYPE_WEIGHTS


def roll_forward_refusal_message() -> str:
    return (
        "--weg2-weight-source exchange gives the NEXTN/MTP draft shard its own "
        f"region tag ({GPU_MEMORY_TYPE_WEIGHTS_DRAFT}, spec section 4.1), and "
        "the backup-OFF wake refills BOTH shards of this process through ONE "
        "update_weights_from_disk inside ONE region "
        "(weight_updater.py:576-597). One region carries one tag, so this "
        "roll-forward would re-tag the drafter's repacked parameters into the "
        "weights family and the next flip would have a destination range with "
        "no VRAM source (W74). Refusing rather than re-tagging. OWNER: S6 -- "
        "W73's roll-forward needs one region per runner (spec section 3.6)."
    )


# ===========================================================================
# S2 -- COVERAGE ARMING (W84, spec section 2.4 rule 4 and section 6/S2)
# ===========================================================================

COVER_LINE_PREFIX = "WEG2-XCHG-COVER"
COVERAGE_REFUSAL_MARKER = "W84 Weg2XchgCoverageRefused"

#: MINIMAL INTERFACE, TODO(S1, branch weg2/xchg-s1-0908): the plan half of this
#: module must expose its parameter population in exactly this shape --
#: ``{tag: {parameter name: PLANNED BYTES}}``, one entry per exchanged tag,
#: over ``named_parameters()`` names, where the bytes are the sum of that
#: parameter's ``XchgDesc.nbytes`` (the descriptors the exchange will actually
#: move), NOT the live tensor's storage size.
#:
#: NAMES ALONE ARE NOT ENOUGH, and that is the whole point (refuter F3): with a
#: name set, a parameter counts as covered the moment its NAME appears, and the
#: bytes charged are the live tensor's -- so a plan whose ``in_proj_qkvz``
#: carries three device sub-blocks instead of four (spec section 2.2 / risk R5,
#: the case that writes ``k`` over ``q`` on ranks 1 and 2) passes with
#: ``uncovered=0`` and a plausible slack.  The arming check for SILENT
#: wrongness must be able to see a partially covered parameter, so the plan
#: hands over its byte claim and this check compares it against the storage.
PlanBytes = Mapping[str, Mapping[str, int]]

#: What ``tms_tag_bytes`` answers for one tag, in bytes; 0 means THE SAVER
#: COULD NOT ANSWER and is printed as such, never read as "this tag is empty"
#: (weight_updater.py:355-376).
TagBytesFn = Callable[[str], int]

PARAMETER = "parameter"
BUFFER = "buffer"
ATTRIBUTE = "attribute"


@dataclass(frozen=True)
class LiveTensor:
    """One live tensor found under an exchanged tag, and how it was found."""

    name: str
    module_path: str
    kind: str  # PARAMETER | BUFFER | ATTRIBUTE
    tag: str
    nbytes: int
    storage_key: Tuple[Any, ...]
    dtype: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class ShortParameter:
    """A parameter the plan carries, but not all of."""

    name: str
    tag: str
    planned_bytes: int
    live_bytes: int


@dataclass(frozen=True)
class TagCoverage:
    """The arithmetic for one exchanged tag, on one rank."""

    rank: int
    tag: str
    mode: str
    planned_bytes: int
    buffers_bytes: int
    tms_bytes: int
    uncovered: Tuple[LiveTensor, ...]
    short: Tuple[ShortParameter, ...]
    missing: Tuple[str, ...]
    n_parameters: int
    n_buffers: int
    n_attributes: int

    @property
    def slack_bytes(self) -> int:
        """``tms_tag_bytes`` minus what this walk accounts for.

        PRINTED, NEVER COMPARED FOR EQUALITY (spec section 6/S2).  Allocator
        overhang is real and measured at +0.08 to +0.58 GiB per rank, so an
        equality assert would refuse every boot.  It can also be NEGATIVE --
        that is the saver answering 0 because the running hook has no
        ``tms_tag_bytes`` symbol, an ABSENCE the reader must see rather than a
        deficit the code invents a rule about.
        """
        return int(self.tms_bytes) - int(self.planned_bytes) - int(self.buffers_bytes)

    @property
    def ok(self) -> bool:
        return not (self.uncovered or self.short or self.missing)

    def cover_line(self) -> str:
        """The acceptance line, spec field order verbatim.

        Fields after ``uncovered=`` are APPENDED, never interleaved: the
        greppable prefix stays exactly what the spec names, and the counts that
        follow are the DENOMINATOR of ``uncovered`` -- a population number
        without its population is the trap this campaign keeps paying for.
        ``planned_mib`` is the PLAN's byte claim, so a plan that covers a
        parameter only partly shows up here as slack rather than vanishing.
        """
        return (
            f"{COVER_LINE_PREFIX} rank={int(self.rank)} tag={self.tag} "
            f"planned_mib={self.planned_bytes / MIB:.1f} "
            f"buffers_mib={self.buffers_bytes / MIB:.1f} "
            f"tms_mib={self.tms_bytes / MIB:.1f} "
            f"slack_mib={self.slack_bytes / MIB:.1f} "
            f"uncovered={len(self.uncovered)} "
            f"short={len(self.short)} missing={len(self.missing)} "
            f"params={self.n_parameters} buffers={self.n_buffers} "
            f"attrs={self.n_attributes} "
            f"tms_answered={'yes' if self.tms_bytes else 'no'} "
            f"mode={self.mode}"
        )


def tag_of_parameter_name(
    name: str, *, region_tag: str = GPU_MEMORY_TYPE_WEIGHTS
) -> str:
    """The torch_memory_saver tag a tensor of this NAME was allocated under.

    THE REGION WINS OVER THE NAME, and it has to (review F1 / refuter F4).
    Inside the BASE weights region, name-derived and allocation-derived are the
    same thing: every layer is constructed inside ``weight_chunk_scope(idx)``
    (utils/common.py:2017) and every post-load repack inside
    ``weight_chunk_scope(layer_id_from_module_name(name))``
    (model_loader/loader.py:941), both of which take the chunk from the layer
    id, and that identity is the only reason a per-tag census can be compared
    against a per-name plan.  Inside ANY OTHER weights region the identity
    breaks in exactly one direction: ``weight_chunk_scope`` is a no-op there
    (weg2_memory_saver.py, #1273 S2), so the drafter's one-layer
    ``Qwen3_5ForCausalLM`` block (qwen3_5_mtp.py:277-283) is ALLOCATED under
    ``weights_draft`` while its name still says ``layers.0.``.  A region-blind
    reading returns ``weights_0`` for those bytes and would then census a
    family tag that does not exist in that process.

    The clamp for a layer beyond the last chunk is ``weight_chunk_tag``'s, not
    a second copy of it.
    """
    if region_tag != GPU_MEMORY_TYPE_WEIGHTS:
        return region_tag
    layer_id = layer_id_from_module_name(name)
    if layer_id is None:
        return GPU_MEMORY_TYPE_WEIGHTS
    return weight_chunk_tag(layer_id) or GPU_MEMORY_TYPE_WEIGHTS


def _storage_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identity of the ALLOCATION behind a tensor, so a view is not a second
    tensor.  ``self.lm_head = self.model.embed_tokens`` (qwen3_5_mtp.py:288) is
    the shape in the tree; charging its bytes twice would inflate the census
    comparison and, worse, a stray view of covered bytes would be reported as
    an uncovered page that does not exist."""
    try:
        storage = tensor.untyped_storage()
        nbytes = int(storage.nbytes())
        if nbytes > 0:
            return ("storage", int(storage.data_ptr()), nbytes)
    except Exception:  # noqa: BLE001 -- meta/fake tensors have no storage
        pass
    return ("object", id(tensor))


def _nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return int(tensor.numel()) * int(tensor.element_size())


def _join(module_path: str, attr: str) -> str:
    return f"{module_path}.{attr}" if module_path else attr


def walk_live_tensors(
    model: torch.nn.Module, *, region_tag: str = GPU_MEMORY_TYPE_WEIGHTS
) -> List[LiveTensor]:
    """Every live tensor of the model, with the tag it lives under.

    Three populations, and the third is the reason this function exists:

    * ``named_parameters()`` -- what the plan transports;
    * ``named_buffers()`` -- what ``_export_static_state`` carries across the
      flip (weight_updater.py:1291 -> :1755); the largest is the rope
      ``cos_sin_cache`` (rotary_embedding/base.py:173, measured +300/+181/+210
      MiB on D's ``weights_0``);
    * every module's ``__dict__`` -- a plain ``self.x = torch.zeros(...)``
      is neither a Parameter nor a buffer, appears in NO standard iterator, and
      is exactly the byte the exchange would leave undefined.

    BOUND OF THE CHECK, stated rather than assumed: a tensor reachable only
    through a container attribute (a list, a dict, a dataclass field on a
    module) is not found by this walk.  It refuses what it can see and names
    its population on the log line; it does not claim to be exhaustive over
    every reachable object graph.
    """
    out: List[LiveTensor] = []

    def _add(name: str, tensor: torch.Tensor, kind: str, module_path: str) -> None:
        out.append(
            LiveTensor(
                name=name,
                module_path=module_path,
                kind=kind,
                tag=tag_of_parameter_name(name, region_tag=region_tag),
                nbytes=_nbytes(tensor),
                storage_key=_storage_key(tensor),
                dtype=str(tensor.dtype),
                shape=tuple(tensor.shape),
            )
        )

    for name, param in model.named_parameters():
        _add(name, param, PARAMETER, name.rsplit(".", 1)[0] if "." in name else "")
    for name, buf in model.named_buffers():
        if buf is None:
            continue
        _add(name, buf, BUFFER, name.rsplit(".", 1)[0] if "." in name else "")
    for module_path, module in model.named_modules():
        for attr, value in list(vars(module).items()):
            if not isinstance(value, torch.Tensor):
                continue
            _add(_join(module_path, attr), value, ATTRIBUTE, module_path)
    return out


def build_coverage(
    model: torch.nn.Module,
    *,
    rank: int,
    planned_bytes_by_tag: PlanBytes,
    tag_bytes: TagBytesFn,
    region_tag: str = GPU_MEMORY_TYPE_WEIGHTS,
    mode: Optional[str] = None,
) -> Dict[str, TagCoverage]:
    """The per-tag arithmetic, over the exchanged tags only.

    An exchanged tag is a WEIGHTS-FAMILY tag: the base tag plus
    ``weights_<integer>``.  ``weights_draft`` is deliberately not one, so the
    drafter's bytes are never charged to a family tag's slack and never
    counted as an uncovered page -- and under a draft region there is no family
    tag at all, so this returns no rows rather than censusing tags that do not
    exist in that process.

    Three refusal populations, not one:

    * ``uncovered`` -- a live tensor under an exchanged tag that is neither a
      plan parameter nor a registered buffer: a page the destination never
      receives;
    * ``short`` -- a plan parameter whose byte claim does not equal the live
      storage: the partially-tiled parameter, which a name-only check cannot
      see (risk R5);
    * ``missing`` -- a plan parameter with no live tensor on this rank: a plan
      built against a different shard geometry.
    """
    mode = mode if mode is not None else weight_source()
    live = walk_live_tensors(model, region_tag=region_tag)

    tags = {t.tag for t in live if is_weights_family_tag(t.tag)}
    tags |= {t for t in planned_bytes_by_tag if is_weights_family_tag(t)}

    rows: Dict[str, TagCoverage] = {}
    for tag in sorted(tags):
        planned_of_tag = dict(planned_bytes_by_tag.get(tag, {}))
        of_tag = [t for t in live if t.tag == tag]

        covered_storage = set()
        seen_names = set()
        planned_bytes = 0
        buffers_bytes = 0
        uncovered: List[LiveTensor] = []
        short: List[ShortParameter] = []
        n_par = n_buf = n_attr = 0

        # Parameters and buffers first: they define WHICH allocations are
        # accounted for, and only then can an attribute be judged as an alias
        # of one of them rather than as a page of its own.
        for t in of_tag:
            if t.kind == PARAMETER:
                n_par += 1
                if t.name in planned_of_tag:
                    seen_names.add(t.name)
                    claim = int(planned_of_tag[t.name])
                    if claim != t.nbytes:
                        short.append(
                            ShortParameter(
                                name=t.name,
                                tag=tag,
                                planned_bytes=claim,
                                live_bytes=t.nbytes,
                            )
                        )
                    if t.storage_key not in covered_storage:
                        covered_storage.add(t.storage_key)
                        # The PLAN's bytes, never the model's: a plan that
                        # tiles only three of four device sub-blocks must move
                        # this number, not the storage size it failed to cover.
                        planned_bytes += claim
                else:
                    uncovered.append(t)
            elif t.kind == BUFFER:
                n_buf += 1
                if t.storage_key not in covered_storage:
                    covered_storage.add(t.storage_key)
                    buffers_bytes += t.nbytes
        for t in of_tag:
            if t.kind != ATTRIBUTE:
                continue
            n_attr += 1
            if t.storage_key not in covered_storage:
                uncovered.append(t)

        rows[tag] = TagCoverage(
            rank=int(rank),
            tag=tag,
            mode=mode,
            planned_bytes=planned_bytes,
            buffers_bytes=buffers_bytes,
            tms_bytes=int(tag_bytes(tag) or 0),
            uncovered=tuple(uncovered),
            short=tuple(short),
            missing=tuple(sorted(set(planned_of_tag) - seen_names)),
            n_parameters=n_par,
            n_buffers=n_buf,
            n_attributes=n_attr,
        )
    return rows


def coverage_refusal_message(rows: Mapping[str, TagCoverage]) -> str:
    """The W84 text: every refused tensor named, with its module path."""
    parts: List[str] = []
    for tag in sorted(rows):
        row = rows[tag]
        for t in row.uncovered:
            parts.append(
                f"UNCOVERED tag={tag} kind={t.kind} name={t.name} "
                f"module={t.module_path or '<root>'} "
                f"dtype={t.dtype} shape={list(t.shape)} mib={t.nbytes / MIB:.3f}"
            )
        for s in row.short:
            parts.append(
                f"SHORT tag={tag} name={s.name} "
                f"planned_mib={s.planned_bytes / MIB:.3f} "
                f"live_mib={s.live_bytes / MIB:.3f} "
                f"delta_mib={(s.live_bytes - s.planned_bytes) / MIB:.3f}"
            )
        for name in row.missing:
            parts.append(f"MISSING tag={tag} name={name}")
    return (
        f"{COVERAGE_REFUSAL_MARKER}: {len(parts)} finding(s). A live tensor "
        "under an exchanged tag that is neither a plan Parameter nor a "
        "registered buffer (UNCOVERED), a plan Parameter whose byte claim does "
        "not equal its live storage (SHORT), or a plan Parameter with no live "
        "tensor on this rank (MISSING) all mean the same thing: the exchange "
        "has no source for some destination page and the destination would "
        "serve whatever its arena held. " + " | ".join(parts) + " -- either the "
        "plan is extended to cover them, or that tag runs "
        "--weg2-weight-source ring (spec section 6/S2 stop-loss). The exchange "
        "does not ship a tag it cannot fully account for."
    )


@dataclass(frozen=True)
class CoverageVote:
    """This rank's arming verdict.  A VOTE, not a decision.

    #802 / "Raenge nie uneins" (refuter F5): the first form of this check
    raised ``Weg2XchgCoverageRefused`` straight out of the arming call.  At the
    call site spec section 6/S2 names -- the end of weight loading -- there is
    no group fence in scope, so one rank whose plan is short would die at boot
    while the other five continued into a collective that no longer has six
    members.  The derivation stays rank-local; the DECISION belongs to whoever
    holds a fence, so this type crosses that boundary and
    :func:`refuse_if_not_ok` is called only where a fence exists (the wake
    RPC's preamble, spec section 3.6, S6).
    """

    rank: int
    mode: str
    region_tag: str
    rows: Dict[str, TagCoverage]
    ok: bool
    reason: str


def refuse_if_not_ok(vote: CoverageVote) -> CoverageVote:
    """Raise W84 for a failing vote.  ONLY where a group fence covers it."""
    if not vote.ok:
        raise Weg2XchgCoverageRefused(vote.reason)
    return vote


def arm_coverage(
    model: torch.nn.Module,
    *,
    rank: int,
    planned_bytes_by_tag: PlanBytes,
    tag_bytes: TagBytesFn,
    log: Optional[Callable[[str], None]] = None,
    region_tag: str = GPU_MEMORY_TYPE_WEIGHTS,
) -> CoverageVote:
    """Compute this rank's arming vote and emit the acceptance lines.

    NEVER RAISES for a coverage finding -- see :class:`CoverageVote`.  Callers
    that hold a fence pass the result to :func:`refuse_if_not_ok`.
    """
    emit = log if log is not None else logger.info
    mode = weight_source()
    rows = build_coverage(
        model,
        rank=rank,
        planned_bytes_by_tag=planned_bytes_by_tag,
        tag_bytes=tag_bytes,
        region_tag=region_tag,
        mode=mode,
    )
    for tag in sorted(rows):
        emit(rows[tag].cover_line())
    ok = all(rows[tag].ok for tag in rows)
    return CoverageVote(
        rank=int(rank),
        mode=mode,
        region_tag=region_tag,
        rows=rows,
        ok=ok,
        reason="" if ok else coverage_refusal_message(rows),
    )


# ---------------------------------------------------------------------------
# THE WIRING.  An instrument nobody calls measures nothing.
# ---------------------------------------------------------------------------

#: The plan half (S1) registers its byte population here and S6's flag arms the
#: mode; this seam is what lets the check be WIRED before either exists.  A
#: SEAM, not a second bookkeeping: nothing is stored here but the callable that
#: answers "what does the plan carry for this model", and the answer is
#: recomputed at every call.
_PLAN_PROVIDER: Optional[Callable[[Any], PlanBytes]] = None

#: This rank's standing boot vote, for the fenced re-check to fold into the
#: group's decision (spec section 3.6: the RPC preamble re-checks coverage
#: BEFORE the first ``resume``, where an abandon still costs nothing).
_BOOT_VOTE: Optional[CoverageVote] = None


def plan_bytes_from_descs(descs: Any) -> PlanBytes:
    """``{tag: {param_name: planned bytes}}`` -- S1's own adapter, at last.

    S1 wrote the contract and left the wire: *"S6 registers a provider that
    sums ``XchgDesc.nbytes`` per ``param_name`` over ``XchgPlan.raw_descs``"*.
    This is that sum.

    PER PARAMETER, NOT PER DESCRIPTOR, and the difference is the whole reason
    the adapter exists: one parameter is many descriptors -- one per
    destination rank, and fewer-but-wider ones after ``coalesce`` merges
    adjacent spans -- while the coverage arm (W84) asks how many bytes of THIS
    PARAMETER the plan accounts for.

    ZEROFILL IS NOT PLANNED BYTES.  A zerofill piece has no source and moves
    nothing over the link; the destination memsets it locally
    (``weight_exchange_transport.apply_zerofill``).  Counting it would tell the
    coverage arm that a byte range is accounted for by a transfer that never
    happens, which is the direction that serves undefined weights.
    """
    out: Dict[str, Dict[str, int]] = {}
    for d in descs or ():
        if getattr(d, "kind", None) == ZEROFILL:
            continue
        tag = str(getattr(d, "tag", ""))
        name = str(getattr(d, "param_name", ""))
        bucket = out.setdefault(tag, {})
        bucket[name] = bucket.get(name, 0) + int(getattr(d, "nbytes", 0))
    return out


#: The env var the launcher publishes this rank's Weg-2 group on
#: (``launcher.build_env``, ``SGLANG_WEG2_GROUP``).  Read through
#: ``weg2_memory_saver.weg2_group_name`` so there is one reader.
GROUP_ENV = "SGLANG_WEG2_GROUP"


def default_plan_provider(*, rank: int, region_tag: str = "") -> Callable[[Any], PlanBytes]:
    """The registrant S1's TODO asked for, deriving through the SHADOW.

    IT DOES NOT FORK THE DERIVATION, and that is the one design decision in
    here.  ``weight_exchange_shadow.derive_leg_plan`` is the producer BOTH
    ENDS of the shadow already use -- the plan whose ``plan_digest`` Gate 0
    agrees across all six ranks, narrowed by the mine/theirs manifest census --
    so the bytes this provider reports are the bytes that plan will move.  A
    second derivation here would be a second plan, and a plan two ranks derive
    differently is precisely what Gate 0 exists to catch; keeping two
    producers of it in one tree is how they begin to differ.

    THE IMPORT IS LAZY because ``weight_exchange_shadow`` imports THIS module;
    a module-level import would be a cycle.

    IT RAISES W68 RATHER THAN RETURNING ``{}``.  ``derive_leg_plan`` answers
    ``(None, reason)`` instead of raising, and that reason is the only thing
    that says WHY -- an empty mapping would reach the coverage arm as "this
    plan accounts for nothing" and be reported as an UNDER-COVERAGE of the
    weights, which is a different and much more confusing finding than "there
    is no plan".

    ``hook`` is the SOURCE side deliberately: the coverage question is
    rank-local ("is every weight byte of this runner accounted for"), it is
    asked at the END OF WEIGHT LOADING when no peer manifest exists yet, and
    the source hook is the side whose ``src_ptr`` this rank owns.  No
    ``agreed`` set is passed for the same reason, and ``require_agreement``
    stays False: at load time there is no co-located peer to agree with, and
    demanding one would refuse every boot before its first flip.
    """

    def provider(model: Any) -> PlanBytes:
        from sglang.srt.managers.weg2_memory_saver import weg2_group_name
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        group = (weg2_group_name() or "").strip()
        if group not in ("P", "D"):
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree plan-provider: this rank has no "
                f"Weg-2 group identity ({GROUP_ENV}={group!r}), so the cut the "
                f"plan describes is unknown. Guessing a group would plan the "
                f"PP cut on a TP rank and grade the weights against the wrong "
                f"shard map -- a silent wrong answer. Refusing instead; the "
                f"launcher publishes {GROUP_ENV} in build_env."
            )
        peer = "D" if group == "P" else "P"
        plan, reason = sh.derive_leg_plan(
            hook=sh.HOOK_SOURCE, group=group, peer=peer, rank=int(rank),
            model=model, region_tag=str(region_tag),
        )
        if plan is None:
            raise Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree plan-provider: the exchange is "
                f"armed on this rank (group={group} rank={rank}) but the plan "
                f"derivation refused: {reason}. Reported as a refusal and not "
                f"as an empty plan, because an empty plan reaches the coverage "
                f"arm as 'nothing is accounted for' and is graded as an "
                f"under-coverage of the weights rather than as an absent plan."
            )
        return plan_bytes_from_descs(getattr(plan, "descs", ()))

    return provider


def install_default_plan_provider(*, rank: int, region_tag: str = "") -> bool:
    """Install :func:`default_plan_provider` under ``exchange``, if none is set.

    ``True`` when this call installed one.  THE ARM SELF-ARMS, and it does so
    here rather than at the call site because the call site
    (``model_executor/model_runner.py``, the end of weight loading) is not a
    file this slice owns -- and because one place should answer "does this rank
    have a plan provider".

    AN ALREADY-REGISTERED PROVIDER WINS.  A test's or a later slice's provider
    must not be replaced by the default, or the default becomes the only
    reachable producer and every caller that installed its own is silently
    disarmed.
    """
    if not exchange_armed():
        return False
    if plan_provider() is not None:
        return False
    register_plan_provider(default_plan_provider(rank=rank,
                                                 region_tag=region_tag))
    return True


def register_plan_provider(fn: Optional[Callable[[Any], PlanBytes]]) -> None:
    """Register (or, with ``None``, clear) the plan's byte population source."""
    global _PLAN_PROVIDER
    _PLAN_PROVIDER = fn


def plan_provider() -> Optional[Callable[[Any], PlanBytes]]:
    return _PLAN_PROVIDER


def boot_vote() -> Optional[CoverageVote]:
    """This rank's arming vote from the end of weight loading, or ``None``."""
    return _BOOT_VOTE


def _record_boot_vote(vote: Optional[CoverageVote]) -> None:
    global _BOOT_VOTE
    _BOOT_VOTE = vote


NO_PLAN_REASON = (
    f"{COVERAGE_REFUSAL_MARKER}: --weg2-weight-source exchange is armed but no "
    "plan provider is registered on this rank, so NOTHING is accounted for: "
    "every weight page of every exchanged tag would be filled from a plan that "
    "does not exist. S1 registers the plan (register_plan_provider) and S6 "
    "arms the mode; a boot that has the mode without the plan is refused here "
    "rather than at the first flip."
)


def arm_coverage_at_load(
    model: Optional[torch.nn.Module],
    *,
    rank: int,
    tag_bytes: TagBytesFn,
    region_tag: str,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[CoverageVote]:
    """THE CALL SITE, spec section 6/S2: the end of weight loading.

    A no-op under ``ring`` -- no lines, no walk, no cost on the default path.
    Under ``exchange`` it emits this runner's ``WEG2-XCHG-RESIDENT`` line, and
    for a runner whose weights are IN the exchanged family it also builds the
    coverage rows, emits one ``WEG2-XCHG-COVER`` line per tag, and records the
    vote.

    It does NOT raise: there is no group fence at the end of weight loading
    (refuter F5).  The refusal text is logged with its marker so an operator
    sees it at boot, and the vote is recorded for the fenced re-check in the
    wake RPC's preamble, which is where §3.6 says an abandon still costs
    nothing.
    """
    emit = log if log is not None else logger.info
    if not exchange_armed():
        _record_boot_vote(None)
        return None
    mode = weight_source()

    def _bytes(tag: str) -> int:
        try:
            return int(tag_bytes(tag) or 0)
        except Exception:  # noqa: BLE001 -- an absent instrument is an absence
            return 0

    emit(
        resident_line(
            tag=region_tag, mib=_bytes(region_tag) / MIB, rank=rank, mode=mode
        )
    )
    if region_tag != GPU_MEMORY_TYPE_WEIGHTS:
        # This runner's weights are out of the exchanged family by construction
        # (spec section 4.1).  There is nothing for the exchange to cover here,
        # and nothing for it to get wrong; the RESIDENT line above is the whole
        # statement.
        vote = CoverageVote(
            rank=int(rank),
            mode=mode,
            region_tag=region_tag,
            rows={},
            ok=True,
            reason="",
        )
        _record_boot_vote(vote)
        return vote

    # THE TODO's OTHER HALF (S6 step 6b): register the provider.  Before
    # this, `exchange` was fail-closed BY OMISSION -- every rank voted
    # ok=False with NO_PLAN_REASON because nothing had ever been registered,
    # which is a deliberate refusal but not a working arm.
    install_default_plan_provider(rank=int(rank), region_tag=str(region_tag))
    provider = plan_provider()
    if provider is None or model is None:
        vote = CoverageVote(
            rank=int(rank),
            mode=mode,
            region_tag=region_tag,
            rows={},
            ok=False,
            reason=NO_PLAN_REASON,
        )
        _record_boot_vote(vote)
        logger.error("%s", vote.reason)
        return vote

    vote = arm_coverage(
        model,
        rank=rank,
        planned_bytes_by_tag=provider(model),
        tag_bytes=_bytes,
        log=emit,
        region_tag=region_tag,
    )
    _record_boot_vote(vote)
    if not vote.ok:
        logger.error("%s", vote.reason)
    return vote
