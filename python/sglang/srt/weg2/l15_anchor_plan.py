# SPDX-License-Identifier: Apache-2.0
"""L15-07: the GDN END-anchor cut plan, both directions (pure).

An END anchor is one GDN recurrent state (temporal + conv) per linear layer.
The same bytes are cut two ways at once:

* a P (PP) stage owns WHOLE linear layers -- all heads, all conv channels;
* a D (TP) rank owns a HEAD SHARE of every layer, and its conv sub-blocks
  [query | key | value] are sharded independently.

This module turns those two cuts into byte pieces -- contiguous ranges of
the canonical blob moved between a source's compact buffer and a
destination's compact buffer -- for the two L1.5 data paths:

* ``d_to_p``: the hot D->P handover (plan 2.0) -- D rank r's head share of
  stage s's layers becomes stage s's full-layer state;
* ``p_to_d``: the deposit (plan 2.5) -- stage s's full layers become each D
  rank's head share; TP0 is routed ``skip`` (it fills its rows from L2 at
  the wake, it is never a deposit destination).

Pure: no CUDA, no I/O, no allocator, no model knowledge. The head cut is
NOT re-derived here -- the byte layout comes from
``mem_cache.hicache_migrate`` (``MambaBlobSpec`` and the extent functions
the arena and the store read/write path use), and the geometry arrives as
input (stage layer ranges, the per-rank ratio vector), so the same code
serves 27B and NF.

Canonical blob (``MambaBlobSpec``): [temporal of all layers][conv of all
layers], both layer-major; inside a conv layer the channels run
q | k | v, each sub-block sharded independently. An owner's COMPACT buffer
is the concatenation of its ranges in range order, so a piece's
``src_off`` / ``dst_off`` is the position of ``canon_off`` inside that
concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from sglang.srt.mem_cache.hicache_migrate import (
    MambaBlobSpec,
    conv_extents,
    layer_extents,
    temporal_extents,
)

__all__ = ["AnchorPiece", "plan_anchor", "rank_ranges", "stage_ranges"]


@dataclass(frozen=True)
class AnchorPiece:
    """One contiguous byte range of the canonical anchor blob.

    ``canon_off`` / ``length`` locate the piece in the canonical blob;
    ``src_off`` / ``dst_off`` are the byte offsets inside the compact
    buffers of ``src`` (``"tp<r>"`` or ``"pp<s>"``) and ``dst``, i.e. the
    position of ``canon_off`` within the concatenation of that owner's
    ranges. ``route`` is ``"local"`` when source and destination sit on the
    same physical card (D2D copy), ``"lane"`` when it rides the barlink
    lanes, ``"skip"`` when the piece is named but not moved (TP0 in the
    deposit -- its rows come from L2 at the wake).
    """

    src: str
    dst: str
    canon_off: int
    length: int
    src_off: int
    dst_off: int
    route: str


def stage_ranges(
    spec: MambaBlobSpec, layer_lo: int, layer_hi: int
) -> List[Tuple[int, int]]:
    """Canonical byte ranges of a P stage owning linear layers [lo, hi).

    Reuses ``layer_extents``: TWO ranges -- one temporal, one conv -- whose
    concatenation is exactly ``spec.for_layers(lo, hi)``'s blob (the two
    layer-major regions of the blob sit far apart; a flat slice of
    ``total_bytes`` would take the wrong layers, see ``layer_extents``).
    """
    if not 0 <= layer_lo <= layer_hi <= spec.num_layers:
        raise ValueError(
            f"stage layer range [{layer_lo}, {layer_hi}) is not within "
            f"[0, {spec.num_layers})"
        )
    return [(int(off), int(ln)) for off, ln in layer_extents(spec, layer_lo, layer_hi)]


def rank_ranges(
    spec: MambaBlobSpec, ratios: Sequence[int], rank: int
) -> List[Tuple[int, int]]:
    """Canonical byte ranges of D rank ``r``'s head share, zero-length
    ranges dropped -- a zero-share rank (NF form) owns no bytes at all.

    Reuses ``temporal_extents`` + ``conv_extents``: per layer the temporal
    head slice, then the three conv sub-block slices (q | k | v, sharded
    independently -- a flat conv slice is the documented wrong-channels
    bug class). The concatenation is the rank's own compact blob.
    """
    rs = list(temporal_extents(spec, list(ratios), rank)) + list(
        conv_extents(spec, list(ratios), rank)
    )
    return [(int(off), int(ln)) for off, ln in rs if ln > 0]


def _concat_offsets(ranges: Sequence[Tuple[int, int]]) -> List[int]:
    """Position of each range inside its owner's compact buffer."""
    out, pos = [], 0
    for _, ln in ranges:
        out.append(pos)
        pos += ln
    return out


def _dst_key(dst: str) -> Tuple[str, int]:
    """Split ``tp<r>`` / ``pp<s>`` into ``(letters, rank)`` for NUMERIC order.

    Plain string order misranks two-digit ids (``tp10`` < ``tp2`` lexicographically);
    ranking by ``(letters, int)`` puts ``tp2`` before ``tp10`` while still keeping
    all ``tp*`` together and all ``pp*`` together.
    """
    i = len(dst)
    while i > 0 and dst[i - 1].isdigit():
        i -= 1
    return (dst[:i], int(dst[i:]) if i < len(dst) else 0)


def _intersect(
    a: Sequence[Tuple[int, int]],
    apos: Sequence[int],
    b: Sequence[Tuple[int, int]],
    bpos: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    """Overlaps of two ascending, disjoint canonical range lists.

    Returns ``(canon_off, length, pos_in_a_buffer, pos_in_b_buffer)`` per
    overlap. Both inputs must be ascending by canonical offset and
    disjoint -- the property ``rank_ranges`` / ``stage_ranges`` provide.
    """
    out: List[Tuple[int, int, int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        a_off, a_ln = a[i]
        b_off, b_ln = b[j]
        lo = a_off if a_off > b_off else b_off
        hi = min(a_off + a_ln, b_off + b_ln)
        if lo < hi:
            out.append((lo, hi - lo, apos[i] + lo - a_off, bpos[j] + lo - b_off))
        if a_off + a_ln < b_off + b_ln:
            i += 1
        else:
            j += 1
    return out


def plan_anchor(
    spec: MambaBlobSpec,
    ratios: Sequence[int],
    stage_layers: Sequence[Tuple[int, int]],
    stage_card: Sequence[int],
    rank_card: Sequence[int],
    direction: str,
    tp0_skip: bool = True,
    skip_ranks=None,
) -> List[AnchorPiece]:
    """Byte pieces of ONE END anchor between the D ranks and the P stages.

    ``direction`` is ``"d_to_p"`` (hot handover: src rank -> dst stage) or
    ``"p_to_d"`` (deposit: src stage -> dst rank; with ``tp0_skip`` every
    piece bound for ``"tp0"`` carries route ``"skip"``). A piece is one
    non-empty intersection of a rank's ranges with a stage's ranges.
    Deterministic order: by (dst, dst_off).
    """
    if direction not in ("d_to_p", "p_to_d"):
        raise ValueError(f"unknown direction {direction!r}")
    if len(stage_card) != len(stage_layers):
        raise ValueError(
            f"{len(stage_layers)} stages but {len(stage_card)} stage cards"
        )
    if len(rank_card) != len(ratios):
        raise ValueError(
            f"{len(ratios)} ranks but {len(rank_card)} rank cards"
        )

    stages = [stage_ranges(spec, lo, hi) for lo, hi in stage_layers]
    ranks = [rank_ranges(spec, ratios, r) for r in range(len(ratios))]
    stage_pos = [_concat_offsets(rs) for rs in stages]
    rank_pos = [_concat_offsets(rs) for rs in ranks]

    pieces: List[AnchorPiece] = []
    for s, srs in enumerate(stages):
        if not srs:
            continue
        for r, rrs in enumerate(ranks):
            if not rrs:
                continue
            same_card = stage_card[s] == rank_card[r]
            # L15 hardware-generic: ``skip_ranks`` (the cap-0 ranks, any
            # card) replaces the positional tp0_skip when given
            _skip = (r in set(skip_ranks)) if skip_ranks is not None else (
                tp0_skip and r == 0)
            if direction == "p_to_d" and _skip:
                route = "skip"
            else:
                route = "local" if same_card else "lane"
            for canon_off, ln, rpos, spos in _intersect(
                rrs, rank_pos[r], srs, stage_pos[s]
            ):
                if direction == "d_to_p":
                    src, dst, src_off, dst_off = (
                        f"tp{r}",
                        f"pp{s}",
                        rpos,
                        spos,
                    )
                else:
                    src, dst, src_off, dst_off = (
                        f"pp{s}",
                        f"tp{r}",
                        spos,
                        rpos,
                    )
                pieces.append(
                    AnchorPiece(
                        src=src,
                        dst=dst,
                        canon_off=canon_off,
                        length=ln,
                        src_off=src_off,
                        dst_off=dst_off,
                        route=route,
                    )
                )
    pieces.sort(key=lambda p: (_dst_key(p.dst), p.dst_off))
    return pieces
