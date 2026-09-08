# SPDX-License-Identifier: Apache-2.0
"""Weg-2 weight-byte exchange -- the PLAN half (#1273, spec §6/S1).

Under ``--weg2-weight-source exchange`` the weights region is opened with
``enable_cpu_backup=False`` (``model_executor/model_runner.py:2419-2425``), which
makes ``pause`` a pure unmap and ``resume`` a pure remap: no host granule is
acquired anywhere on the weights path, so the host ring is not created at all.
Between the destination's remap and its first use the exchange injects the bytes
directly, card to card. This module answers the only question that has to be
settled before a byte moves: **which bytes of which rank's storage become which
bytes of which other rank's storage.**

Nothing here touches CUDA or allocates. It is pure arithmetic over replicated
geometry, which is what lets every rank derive the same plan independently and
then HANDSHAKE it (Gate 0, spec §3.3) rather than trust it.

THE THREE READINGS THAT ARE WRONG AND LOOK RIGHT
------------------------------------------------
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

WHAT THIS MODULE DOES NOT DO
----------------------------
It moves nothing and opens nothing. The shared region, Gate 0 and the wave gate
are S3; the transport is S4; the coverage arming (W51) is S2. The two refusals
defined here are raised again by those slices on the same objects -- see the
TODO markers on each class.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace as _dc_replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.distributed.utils import partition_sizes

__all__ = [
    "COALESCE_FLOOR_BYTES",
    "COLS",
    "FLAT",
    "REPLICATED",
    "ROWS",
    "STRIDED2D",
    "ZEROFILL",
    "Block",
    "GroupLayout",
    "ParamGeom",
    "StorageGeom",
    "Weg2XchgPlanDisagree",
    "Weg2XchgSourceMissing",
    "XchgDesc",
    "XchgPlan",
    "build_plan",
    "coalesce",
    "derive_waves",
    "device_block_offsets",
    "piece_histogram",
    "plan_id",
    "shard_offsets",
    "unmergeable_below_floor",
]

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


class Weg2XchgPlanDisagree(RuntimeError):
    """W52 Weg2XchgPlanDisagree -- the plan does not describe the hardware.

    Raised here when a destination tensor is not tiled exactly by its sources
    (two sources for one byte, or a piece that runs past the destination's own
    storage), and when a live tensor's layout cannot be expressed as
    (rows, cols, pitch).

    TODO(S3): Gate 0 raises this same class for an asymmetric 6x6 byte matrix,
    a per-tag total that disagrees with ``tms_tag_bytes``
    (``entrypoint.cpp:131``), and a plan id that disagrees with the front's.
    """


class Weg2XchgSourceMissing(RuntimeError):
    """W58 Weg2XchgSourceMissing -- a destination byte range has no VRAM source
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
    def of(cls, tensor) -> "StorageGeom":
        shape = tuple(int(s) for s in tensor.shape)
        stride = tuple(int(s) for s in tensor.stride())
        itemsize = int(tensor.element_size())
        if len(shape) == 0:
            return cls(rows=1, cols=1, pitch=1, itemsize=itemsize)
        if len(shape) == 1:
            if stride[0] != 1:
                raise Weg2XchgPlanDisagree(
                    f"W52 Weg2XchgPlanDisagree: 1-D tensor with stride {stride} "
                    f"is not a contiguous run; a descriptor cannot name it."
                )
            return cls(rows=1, cols=shape[0], pitch=shape[0], itemsize=itemsize)
        if len(shape) > 2:
            # Only a genuinely contiguous block can be flattened to rows; a
            # guessed pitch is exactly the silent defect this module removes.
            flat = stride[-1] == 1 and all(
                stride[i] == stride[i + 1] * shape[i + 1]
                for i in range(len(shape) - 1)
            )
            if not flat:
                raise Weg2XchgPlanDisagree(
                    f"W52 Weg2XchgPlanDisagree: {len(shape)}-D tensor shape "
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
                f"W52 Weg2XchgPlanDisagree: tensor shape {shape} stride {stride} "
                f"has no unit-stride axis, so it has no storage rows a copy "
                f"primitive can name."
            )
        if pitch < cols:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree: tensor shape {shape} stride {stride} "
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
    for every rank at once: with no ratio vector installed (or one that does not
    match ``tp_size``) it is the classic even split ``rank * shard_size``; with
    one it is the prefix sum over ``partition_sizes``.  A plain prefix sum is
    what makes every shard a contiguous unit range of the full dimension, and
    that is the property the tiling in ``build_plan`` rests on.
    """
    if tp_size <= 0:
        raise Weg2XchgPlanDisagree(
            f"W52 Weg2XchgPlanDisagree: tp_size {tp_size} is not a group size."
        )
    if not ratios or len(ratios) != tp_size:
        if total % tp_size != 0:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree: dimension {total} is not divisible "
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
            f"W52 Weg2XchgPlanDisagree: per-rank sizes {sizes} sum to {acc}, "
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
    is indexed the same way in both directions.
    """

    name: str
    cards: Tuple[int, ...]
    tp_size: int
    ratios: Optional[Sequence[int]] = None
    base: int = 0

    @property
    def n_ranks(self) -> int:
        return len(self.cards)

    def card_of(self, rank: int) -> int:
        return int(self.cards[rank])


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
    between it and the pad is a GAP and is refused by name (W58), never zeroed.

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
    pad_units: int = 0
    src_extent: Optional[int] = None
    stage: Optional[int] = None
    dst_widths: Optional[Sequence[int]] = None
    dst_extents: Optional[Sequence[int]] = None

    def replace(self, **kw) -> "ParamGeom":
        return _dc_replace(self, **kw)

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

    def replace(self, **kw) -> "XchgDesc":
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


def plan_id(descs: Sequence[XchgDesc]) -> str:
    """A 12-hex digest over the descriptor GEOMETRY, pointers excluded.

    Two processes hold the same tensors at different addresses, so an id that
    moved with the address could never be compared -- and comparing it is the
    whole job (W52: "the plan hash != the front's").
    """
    h = hashlib.blake2b(digest_size=6)
    for d in descs:
        h.update(repr(d.key()).encode("utf-8"))
    return h.hexdigest()


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
    left = []
    for d in descs:
        if d.kind != FLAT or d.nbytes >= floor:
            continue
        if any(
            o is not d
            and o.kind == FLAT
            and o.tag == d.tag
            and o.src_rank == d.src_rank
            and o.dst_rank == d.dst_rank
            and (
                (_adjacent(d, o, "src") and _adjacent(d, o, "dst"))
                or (_adjacent(o, d, "src") and _adjacent(o, d, "dst"))
            )
            for o in descs
        ):
            continue
        left.append(d)
    return left


# ---------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XchgPlan:
    descs: Tuple[XchgDesc, ...]
    raw_descs: Tuple[XchgDesc, ...]
    waves: Tuple[Tuple[str, ...], ...]
    byte_matrix: Tuple[Tuple[int, ...], ...]
    plan_id: str
    src_group: str
    dst_group: str

    @property
    def oncard_bytes(self) -> int:
        return sum(d.nbytes for d in self.descs if d.kind != ZEROFILL and d.on_card)

    @property
    def cross_bytes(self) -> int:
        return sum(
            d.nbytes for d in self.descs if d.kind != ZEROFILL and not d.on_card
        )

    @property
    def zerofill_bytes(self) -> int:
        return sum(d.nbytes for d in self.descs if d.kind == ZEROFILL)

    @property
    def min_piece_bytes(self) -> int:
        moved = [d.nbytes for d in self.descs if d.kind != ZEROFILL]
        return min(moved) if moved else 0

    def log_line(self, direction: Optional[str] = None) -> str:
        """The acceptance line of spec §6/S1.  Every figure is computed from
        THIS plan; none is quoted from the spec's projection."""
        if direction is None:
            direction = f"{self.src_group}2{self.dst_group}"
        hist = "/".join(f"{k}:{v}" for k, v in piece_histogram(self.descs).items())
        gib = float(1 << 30)
        return (
            "WEG2-XCHG-PLAN "
            f"dir={direction} waves={len(self.waves)} descs={len(self.raw_descs)} "
            f"coalesced={len(self.descs)} "
            f"min_piece_mib={self.min_piece_bytes / (1 << 20):.6f} "
            f"bytes_gib={(self.oncard_bytes + self.cross_bytes) / gib:.2f} "
            f"oncard_gib={self.oncard_bytes / gib:.2f} "
            f"cross_gib={self.cross_bytes / gib:.2f} "
            f"zerofill_mib={self.zerofill_bytes / (1 << 20):.2f} "
            f"hist={hist} plan_id={self.plan_id}"
        )


def _blocks_of(geom: ParamGeom, layout: GroupLayout, is_dst: bool) -> List[List[Block]]:
    """Per rank of ``layout``, the blocks that rank holds.  An empty list means
    the rank holds nothing of this parameter."""
    if layout.tp_size == 1 and layout.n_ranks >= 1:
        # PP form (or a single-rank group): whole tensors, one holder each.
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
            f"W52 Weg2XchgPlanDisagree: group {layout.name!r} has tp_size "
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
                f"W52 Weg2XchgPlanDisagree: {geom.name}: {len(geom.dst_widths)} "
                f"seeded widths for {layout.n_ranks} ranks."
            )
        out, acc = [], 0
        for w in geom.dst_widths:
            out.append([Block(block=0, global_start=acc, dev_row=0, size=int(w))])
            acc += int(w)
        return out
    if geom.blocks:
        return device_block_offsets(
            geom.blocks, layout.ratios, layout.tp_size, geom.units, geom.groups
        )
    ranges = shard_offsets(
        geom.shard_total, layout.ratios, layout.tp_size, geom.units, geom.groups
    )
    return [
        [Block(block=0, global_start=start, dev_row=0, size=size)]
        for start, size in ranges
    ]


def _pick_source(
    candidates: Sequence[Tuple[int, Block]], d_rank: int
) -> Tuple[int, Block]:
    """Prefer the CO-LOCATED source: rank ``n`` of either group runs on
    ``cards[n]``, so a same-index pair crosses no link at all (spec §2.2 --
    the replicated classes are on-card only).  Otherwise the lowest rank, so
    the choice is deterministic on every rank that derives it."""
    for s_rank, block in candidates:
        if s_rank == d_rank:
            return s_rank, block
    return min(candidates, key=lambda c: c[0])


def _emit(
    geom: ParamGeom,
    src: GroupLayout,
    dst: GroupLayout,
    ptr_of: Optional[Callable[[str, int, str], Optional[int]]],
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

    src_extent_of = [sum(b.size for b in blocks) for blocks in src_blocks]
    out: List[XchgDesc] = []
    for d_rank, d_blocks in enumerate(dst_blocks):
        if not d_blocks:
            continue
        if geom.dst_extents is not None:
            d_extent = int(geom.dst_extents[d_rank])
        else:
            d_extent = sum(b.size for b in d_blocks)
        d_pitch = d_extent if strided else cols
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
                s_pitch = src_extent_of[s_rank] if strided else cols
                if strided:
                    out.append(
                        XchgDesc(
                            tag=geom.tag, src_rank=s_rank, dst_rank=d_rank,
                            param_name=geom.name, kind=STRIDED2D,
                            nbytes=rows * span * itemsize, rows=rows,
                            run_bytes=span * itemsize,
                            spitch=s_pitch * itemsize, dpitch=d_pitch * itemsize,
                            src_off=s_dev * itemsize, dst_off=d_dev * itemsize,
                            src_ptr=ptr(src.name, s_rank),
                            dst_ptr=ptr(dst.name, d_rank),
                        )
                    )
                else:
                    nbytes = span * cols * itemsize
                    out.append(
                        XchgDesc(
                            tag=geom.tag, src_rank=s_rank, dst_rank=d_rank,
                            param_name=geom.name, kind=FLAT, nbytes=nbytes,
                            rows=1, run_bytes=nbytes, spitch=0, dpitch=0,
                            src_off=s_dev * cols * itemsize,
                            dst_off=d_dev * cols * itemsize,
                            src_ptr=ptr(src.name, s_rank),
                            dst_ptr=ptr(dst.name, d_rank),
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
                if strided:
                    out.append(
                        XchgDesc(
                            tag=geom.tag, src_rank=-1, dst_rank=d_rank,
                            param_name=geom.name, kind=ZEROFILL,
                            nbytes=rows * span * itemsize, rows=rows,
                            run_bytes=span * itemsize, spitch=0,
                            dpitch=d_pitch * itemsize,
                            dst_off=d_dev * itemsize,
                            dst_ptr=ptr(dst.name, d_rank),
                        )
                    )
                else:
                    nbytes = span * cols * itemsize
                    out.append(
                        XchgDesc(
                            tag=geom.tag, src_rank=-1, dst_rank=d_rank,
                            param_name=geom.name, kind=ZEROFILL, nbytes=nbytes,
                            rows=1, run_bytes=nbytes, spitch=0, dpitch=0,
                            dst_off=d_dev * cols * itemsize,
                            dst_ptr=ptr(dst.name, d_rank),
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
                f"W58 Weg2XchgSourceMissing: {geom.name} dst_rank={d_rank}: "
                f"units [{cursor}, {lo}) of this rank's {extent} have no VRAM "
                f"source and no ZEROFILL descriptor."
            )
        if lo < cursor:
            raise Weg2XchgPlanDisagree(
                f"W52 Weg2XchgPlanDisagree: {geom.name} dst_rank={d_rank}: unit "
                f"range [{lo}, {hi}) overlaps a range already claimed up to "
                f"{cursor}; two sources for one destination byte."
            )
        cursor = hi
    if cursor > extent:
        raise Weg2XchgPlanDisagree(
            f"W52 Weg2XchgPlanDisagree: {geom.name} dst_rank={d_rank}: the plan "
            f"writes {cursor} units into a destination whose own storage holds "
            f"{extent}. The shard vector and the hardware disagree."
        )
    if cursor < extent:
        raise Weg2XchgSourceMissing(
            f"W58 Weg2XchgSourceMissing: {geom.name} dst_rank={d_rank}: units "
            f"[{cursor}, {extent}) have no VRAM source and no ZEROFILL "
            f"descriptor."
        )


def build_plan(
    inventory: Sequence[ParamGeom],
    src: GroupLayout,
    dst: GroupLayout,
    waves: Sequence[Sequence[str]],
    ptr_of: Optional[Callable[[str, int, str], Optional[int]]] = None,
    floor: int = COALESCE_FLOOR_BYTES,
) -> XchgPlan:
    """The plan for one direction.

    The inventory is walked in SORTED NAME ORDER, never in the order it arrives:
    the C++ side iterates an ``unordered_map`` (``core.cpp:206-222``,
    ``:316-333``) whose order differs between the two processes, and a plan that
    depended on it could not be compared across ranks at all.  Descriptors are
    then ordered by (wave, dst_rank, name, dst offset, src_rank), which is the
    order the transport issues them in and the order coalescing needs.
    """
    wave_of = {}
    for w, wave in enumerate(waves):
        for tag in wave:
            wave_of[tag] = w
    raw: List[XchgDesc] = []
    for geom in sorted(inventory, key=lambda g: g.name):
        if geom.tag not in wave_of:
            continue
        raw.extend(_emit(geom, src, dst, ptr_of))
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
    for d in merged:
        if d.kind == ZEROFILL:
            continue
        matrix[src.base + d.src_rank][dst.base + d.dst_rank] += d.nbytes
    return XchgPlan(
        descs=tuple(merged),
        raw_descs=tuple(raw),
        waves=tuple(tuple(w) for w in waves),
        byte_matrix=tuple(tuple(row) for row in matrix),
        plan_id=plan_id(merged),
        src_group=src.name,
        dst_group=dst.name,
    )
