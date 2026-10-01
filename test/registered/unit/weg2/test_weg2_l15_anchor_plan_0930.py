# SPDX-License-Identifier: Apache-2.0
"""L15-07: the GDN END-anchor cut plan, both directions (pure, CPU only).

The canonical blob (hicache_migrate.MambaBlobSpec) is cut two ways at once:
a P stage owns whole linear layers (all heads), a D rank owns a head share
of every layer. The plan pieces must cover every canonical byte exactly once
per direction, per owner the lengths must add up to that owner's compact
buffer, and a byte-level roundtrip on CPU tensors must be identical.
"""

import pytest
import torch

from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec
from sglang.srt.weg2.l15_anchor_plan import (
    AnchorPiece,
    _dst_key,
    plan_anchor,
    rank_ranges,
    stage_ranges,
)

SPEC = MambaBlobSpec(
    num_layers=6,
    num_heads=16,
    head_dim=4,
    state_size=4,
    conv_dim=48,
    conv_width=3,
    key_dim=16,
    value_dim=16,
    units=16,
    temporal_itemsize=2,
    conv_itemsize=2,
)
STAGES = [(0, 3), (3, 5), (5, 6)]
STAGE_CARD = [1, 0, 2]
RANK_CARD = [1, 0, 2]
UNEVEN = [3725, 2264, 2259]
ZERO_FIRST = [0, 9, 7]


def _idx(name: str) -> int:
    return int(name[2:])


def _compact(buf: torch.Tensor, ranges) -> torch.Tensor:
    if not ranges:
        return torch.zeros(0, dtype=torch.uint8)
    return torch.cat([buf[off : off + ln] for off, ln in ranges])


def _plan(direction, ratios, tp0_skip=True):
    return plan_anchor(
        SPEC, ratios, STAGES, STAGE_CARD, RANK_CARD, direction, tp0_skip=tp0_skip
    )


def test_stage_ranges_concatenate_to_stage_blob():
    for lo, hi in STAGES:
        rs = stage_ranges(SPEC, lo, hi)
        assert len(rs) == 2  # one temporal range, one conv range
        assert sum(ln for _, ln in rs) == SPEC.for_layers(lo, hi).total_bytes


def test_rank_ranges_cover_the_rank_shard():
    for ratios in (UNEVEN, ZERO_FIRST):
        for r in range(len(ratios)):
            rs = rank_ranges(SPEC, ratios, r)
            assert all(ln > 0 for _, ln in rs)
            assert sum(ln for _, ln in rs) == SPEC.shard_for_rank(ratios, r).total_bytes


@pytest.mark.parametrize("ratios", [UNEVEN, ZERO_FIRST], ids=["uneven", "zero_first"])
@pytest.mark.parametrize("direction", ["d_to_p", "p_to_d"])
def test_pieces_cover_canonical_exactly_once(direction, ratios):
    pieces = _plan(direction, ratios)
    intervals = sorted((p.canon_off, p.canon_off + p.length) for p in pieces)
    pos = 0
    for lo, hi in intervals:
        assert lo == pos, f"gap or overlap at byte {pos}: piece starts {lo}"
        pos = hi
    assert pos == SPEC.total_bytes


@pytest.mark.parametrize("ratios", [UNEVEN, ZERO_FIRST], ids=["uneven", "zero_first"])
def test_per_owner_lengths_equal_compact_buffer(ratios):
    d2p = _plan("d_to_p", ratios)
    p2d = _plan("p_to_d", ratios, tp0_skip=False)
    for s, (lo, hi) in enumerate(STAGES):
        stage_bytes = SPEC.for_layers(lo, hi).total_bytes
        assert sum(p.length for p in d2p if p.dst == f"pp{s}") == stage_bytes
        assert sum(p.length for p in p2d if p.src == f"pp{s}") == stage_bytes
    for r in range(len(ratios)):
        rank_bytes = SPEC.shard_for_rank(ratios, r).total_bytes
        assert sum(p.length for p in d2p if p.src == f"tp{r}") == rank_bytes
        assert sum(p.length for p in p2d if p.dst == f"tp{r}") == rank_bytes


@pytest.mark.parametrize("ratios", [UNEVEN, ZERO_FIRST], ids=["uneven", "zero_first"])
def test_roundtrip_on_cpu_tensors(ratios):
    gen = torch.Generator().manual_seed(20260930)
    canonical = torch.randint(
        0, 256, (SPEC.total_bytes,), dtype=torch.uint8, generator=gen
    )
    stage_bufs = [
        _compact(canonical, stage_ranges(SPEC, lo, hi)) for lo, hi in STAGES
    ]
    rank_bufs = [
        _compact(canonical, rank_ranges(SPEC, ratios, r)) for r in range(len(ratios))
    ]

    # d_to_p: rank compact buffers -> stage buffers
    dst = [torch.zeros_like(b) for b in stage_bufs]
    for p in _plan("d_to_p", ratios):
        dst[_idx(p.dst)][p.dst_off : p.dst_off + p.length] = rank_bufs[_idx(p.src)][
            p.src_off : p.src_off + p.length
        ]
    for got, want in zip(dst, stage_bufs):
        assert torch.equal(got, want)

    # p_to_d: stage buffers -> rank compact buffers (no skip, full coverage)
    dst = [torch.zeros_like(b) for b in rank_bufs]
    for p in _plan("p_to_d", ratios, tp0_skip=False):
        dst[_idx(p.dst)][p.dst_off : p.dst_off + p.length] = stage_bufs[_idx(p.src)][
            p.src_off : p.src_off + p.length
        ]
    for got, want in zip(dst, rank_bufs):
        assert torch.equal(got, want)


def test_tp0_skip_route():
    pieces = _plan("p_to_d", UNEVEN)
    tp0 = [p for p in pieces if p.dst == "tp0"]
    assert tp0, "tp0 owns heads under the uneven ratios, so it must appear"
    assert all(p.route == "skip" for p in tp0)
    # a zero-share rank produces no pieces at all, in either field
    pieces = _plan("p_to_d", ZERO_FIRST)
    assert not [p for p in pieces if "tp0" in (p.src, p.dst)]
    pieces = _plan("d_to_p", ZERO_FIRST)
    assert not [p for p in pieces if "tp0" in (p.src, p.dst)]


@pytest.mark.parametrize("ratios", [UNEVEN, ZERO_FIRST], ids=["uneven", "zero_first"])
@pytest.mark.parametrize("direction", ["d_to_p", "p_to_d"])
def test_route_is_local_exactly_when_the_cards_match(direction, ratios):
    tp0_skip = direction == "d_to_p"  # p_to_d must not skip to see every route
    for p in _plan(direction, ratios, tp0_skip=tp0_skip):
        if direction == "d_to_p":
            s, r = _idx(p.dst), _idx(p.src)
        else:
            s, r = _idx(p.src), _idx(p.dst)
        want = "local" if STAGE_CARD[s] == RANK_CARD[r] else "lane"
        assert p.route == want, f"{p.src} -> {p.dst}: route {p.route} != {want}"


def test_pieces_are_deterministically_ordered():
    pieces = _plan("d_to_p", UNEVEN)
    key = [(p.dst, p.dst_off) for p in pieces]
    assert key == sorted(key)
    assert all(isinstance(p, AnchorPiece) for p in pieces)


def test_dst_key_orders_ranks_numerically():
    # Two-digit ids must rank by VALUE (tp2 before tp10), and tp* after pp*.
    assert _dst_key("tp2") == ("tp", 2)
    assert _dst_key("pp1") == ("pp", 1)
    assert sorted(["tp10", "tp2", "pp1"], key=_dst_key) == ["pp1", "tp2", "tp10"]
