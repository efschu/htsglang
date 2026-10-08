"""L15-08 (30.09.2026): the L1.5 KV row plan, both directions, pure.

Pins ``flliper.srt.pdflip.l15_row_plan`` (the hot-handover ``d_to_p`` and the
phase-2 deposit ``p_to_d``) against the acceptance criteria of
L15-PLAN-0930 section L15-08:

* every (layer, token) cell is carried EXACTLY ONCE per direction;
* the owner of every carried slot matches the weighted owner rule of
  ``layers/dcp/owner.py`` (checked against ``dcp_weighted_write_slots``
  itself, not against a paraphrase);
* no piece exceeds the lane ``slot_bytes``;
* TP0 (rank 0, the 5090) is never a deposit destination in ``p_to_d`` --
  it fills its rows from L2 at the wake (plan 2.1 "TP0 fill");
* a zero-share rank (the NF form, vector [0,9,7] over S=16) plans cleanly
  and produces no traffic attributed to it.

Geometry is the plan's input, not a constant of this test's making: the
27B form is 16 attention layers, P attention cut 10/3/3, D token vector
[7,4,5] over S=16; the NF form keeps the layer cut and swaps the token
vector to [0,9,7]. Hermetic: torch CPU only, no CUDA, no process group.
"""

import random

import pytest
import torch

from flliper.srt.layers.dcp.owner import dcp_weighted_write_slots
from flliper.srt.pdflip.l15_row_plan import (
    RowPiece,
    _dst_key,
    owner_prefix,
    owned_rows,
    plan_d_to_p,
    plan_p_to_d,
)

# 27B geometry (L15-PLAN-0930 L15-08): 16 attention layers, P cut 10/3/3.
STAGE_LAYERS = [list(range(10)), [10, 11, 12], [13, 14, 15]]
STAGE_CARD = [1, 0, 2]
RANK_CARD = [1, 0, 2]
VEC_27B = [7, 4, 5]
VEC_NF = [0, 9, 7]
ROW_BYTES = 1024
BIG_SLOT = 8 * 1024 * 1024
TINY_SLOT = 4096


def _owner_of(global_slot: int, prefix) -> int:
    """The owner rule restated independently for the cross-check: the rank r
    with prefix[r] <= (L % S) < prefix[r+1]; a zero-width range never owns."""
    off = global_slot % prefix[-1]
    owner = -1
    for r in range(len(prefix) - 1):
        if prefix[r] <= off < prefix[r + 1]:
            owner = r
    return owner


def test_owner_prefix():
    assert owner_prefix(VEC_27B) == [0, 7, 11, 16]
    assert owner_prefix(VEC_NF) == [0, 0, 9, 16]


def _d_slots(n=200):
    rng = random.Random(7)
    return rng.sample(range(4096), n)


@pytest.mark.parametrize("slot_bytes", [BIG_SLOT, TINY_SLOT])
def test_d_to_p_covers_every_cell_exactly_once(slot_bytes):
    d_slots = _d_slots()
    p_row0 = 512
    pieces = plan_d_to_p(
        d_slots, owner_prefix(VEC_27B), STAGE_LAYERS, STAGE_CARD, RANK_CARD,
        p_row0, ROW_BYTES, slot_bytes,
    )
    seen = {}
    for p in pieces:
        for layer in p.layers:
            for tok, dst_row in zip(p.tokens, p.dst_rows):
                assert dst_row == p_row0 + tok
                key = (layer, tok)
                assert key not in seen, f"cell {key} carried twice"
                seen[key] = p
    expected = {(layer, tok) for layer in range(16) for tok in range(len(d_slots))}
    assert set(seen) == expected


@pytest.mark.parametrize("slot_bytes", [BIG_SLOT, TINY_SLOT])
def test_p_to_d_covers_every_cell_exactly_once_per_stage(slot_bytes):
    prefix = owner_prefix(VEC_27B)
    a, b, e0 = 37, 300, 1000
    for stage, layers in enumerate(STAGE_LAYERS):
        pieces = plan_p_to_d(
            a, b, e0, prefix, stage, STAGE_LAYERS, STAGE_CARD, RANK_CARD,
            ROW_BYTES, slot_bytes, tp0_skip=False,
        )
        seen = {}
        for p in pieces:
            # P-side rows are the chunk token index (identity convention).
            assert p.src_rows == p.tokens
            for layer in p.layers:
                for tok, dst_row in zip(p.tokens, p.dst_rows):
                    key = (layer, tok)
                    assert key not in seen, f"cell {key} carried twice"
                    seen[key] = p
        expected = {(layer, tok) for layer in layers for tok in range(a, b)}
        assert set(seen) == expected


def test_compact_rows_match_owner_py():
    """The plan's compact rows and owners are ``dcp_weighted_write_slots``'s,
    not a re-derivation that could drift from it."""
    prefix = owner_prefix(VEC_27B)
    d_slots = _d_slots()
    for rank in range(len(prefix) - 1):
        idx, rows = owned_rows(d_slots, prefix, rank)
        loc, mask = dcp_weighted_write_slots(
            torch.tensor(d_slots, dtype=torch.int64),
            prefix[-1], prefix[rank], prefix[rank + 1],
            prefix[rank + 1] - prefix[rank],
        )
        ref_idx = [int(i) for i in mask.nonzero(as_tuple=True)[0]]
        ref_rows = [int(rows_v) for rows_v in loc[mask]]
        assert idx == ref_idx
        assert rows == ref_rows
        # ... and the independent owner rule agrees about WHICH tokens.
        assert [_owner_of(d_slots[i], prefix) for i in idx] == [rank] * len(idx)


def test_d_to_p_owner_and_src_rows_match_rule():
    prefix = owner_prefix(VEC_27B)
    d_slots = _d_slots()
    pieces = plan_d_to_p(
        d_slots, prefix, STAGE_LAYERS, STAGE_CARD, RANK_CARD, 0, ROW_BYTES, BIG_SLOT,
    )
    for p in pieces:
        rank = int(p.src[2:])
        for tok, src_row in zip(p.tokens, p.src_rows):
            slot = d_slots[tok]
            assert _owner_of(slot, prefix) == rank
            ratio = prefix[rank + 1] - prefix[rank]
            assert src_row == (slot // prefix[-1]) * ratio + (slot % prefix[-1] - prefix[rank])


@pytest.mark.parametrize("slot_bytes", [BIG_SLOT, TINY_SLOT])
def test_piece_sizes_within_slot_bytes(slot_bytes):
    prefix = owner_prefix(VEC_27B)
    d_slots = _d_slots()
    pieces = plan_d_to_p(
        d_slots, prefix, STAGE_LAYERS, STAGE_CARD, RANK_CARD, 0, ROW_BYTES, slot_bytes,
    )
    pieces += plan_p_to_d(
        37, 300, 1000, prefix, 1, STAGE_LAYERS, STAGE_CARD, RANK_CARD,
        ROW_BYTES, slot_bytes, tp0_skip=False,
    )
    assert pieces
    for p in pieces:
        assert p.nbytes == len(p.tokens) * len(p.layers) * ROW_BYTES
        assert p.nbytes <= slot_bytes


def test_tp0_never_a_deposit_destination():
    prefix = owner_prefix(VEC_27B)
    pieces = plan_p_to_d(
        37, 300, 1000, prefix, 1, STAGE_LAYERS, STAGE_CARD, RANK_CARD,
        ROW_BYTES, BIG_SLOT, tp0_skip=True,
    )
    skipped_tokens = set()
    for p in pieces:
        if p.dst == "tp0":
            assert p.route == "skip"
            skipped_tokens.update(p.tokens)
    # The skipped set is exactly the tokens owned by rank 0 under the rule.
    expected = {t for t in range(37, 300) if _owner_of(1000 + t, prefix) == 0}
    assert skipped_tokens == expected
    for p in pieces:
        if p.route != "skip":
            assert int(p.dst[2:]) != 0


def test_nf_zero_share_rank_produces_no_traffic():
    """NF form [0,9,7] over S=16: rank 0 owns no slot, so no piece may name
    tp0 on either side, in either direction."""
    prefix = owner_prefix(VEC_NF)
    d_slots = _d_slots()
    p_pieces = plan_p_to_d(
        37, 300, 1000, prefix, 1, STAGE_LAYERS, STAGE_CARD, RANK_CARD,
        ROW_BYTES, BIG_SLOT, tp0_skip=True,
    )
    assert p_pieces
    assert all(p.dst != "tp0" for p in p_pieces)
    d_pieces = plan_d_to_p(
        d_slots, prefix, STAGE_LAYERS, STAGE_CARD, RANK_CARD, 0, ROW_BYTES, BIG_SLOT,
    )
    assert all(p.src != "tp0" for p in d_pieces)


def test_route_local_exactly_when_cards_match():
    prefix = owner_prefix(VEC_27B)
    d_slots = _d_slots()
    pieces = plan_d_to_p(
        d_slots, prefix, STAGE_LAYERS, STAGE_CARD, RANK_CARD, 0, ROW_BYTES, BIG_SLOT,
    )
    for p in pieces:
        r, s = int(p.src[2:]), int(p.dst[2:])
        assert p.route == ("local" if RANK_CARD[r] == STAGE_CARD[s] else "lane")
    for stage in range(len(STAGE_LAYERS)):
        pieces = plan_p_to_d(
            37, 300, 1000, prefix, stage, STAGE_LAYERS, STAGE_CARD, RANK_CARD,
            ROW_BYTES, BIG_SLOT, tp0_skip=False,
        )
        for p in pieces:
            r = int(p.dst[2:])
            assert p.route == ("local" if RANK_CARD[r] == STAGE_CARD[stage] else "lane")


def test_deterministic_order_by_dst_then_first_dst_row():
    prefix = owner_prefix(VEC_27B)
    d_slots = _d_slots()
    run1 = plan_d_to_p(
        d_slots, prefix, STAGE_LAYERS, STAGE_CARD, RANK_CARD, 0, ROW_BYTES, TINY_SLOT,
    )
    run2 = plan_d_to_p(
        d_slots, prefix, STAGE_LAYERS, STAGE_CARD, RANK_CARD, 0, ROW_BYTES, TINY_SLOT,
    )
    assert run1 == run2
    keys = [(p.dst, p.dst_rows[0]) for p in run1]
    assert keys == sorted(keys)


def test_dst_key_orders_ranks_numerically():
    # Two-digit ids must rank by VALUE (tp2 before tp10), and tp* after pp*.
    assert _dst_key("tp2") == ("tp", 2)
    assert _dst_key("pp1") == ("pp", 1)
    assert sorted(["tp10", "tp2", "pp1"], key=_dst_key) == ["pp1", "tp2", "tp10"]


def test_rowpiece_is_frozen_dataclass_of_tuples():
    d_slots = _d_slots()
    pieces = plan_d_to_p(
        d_slots, owner_prefix(VEC_27B), STAGE_LAYERS, STAGE_CARD, RANK_CARD,
        0, ROW_BYTES, BIG_SLOT,
    )
    assert pieces
    p = pieces[0]
    assert isinstance(p, RowPiece)
    for field in ("src", "dst", "layers", "tokens", "src_rows", "dst_rows", "route"):
        value = getattr(p, field)
        assert isinstance(value, (str, tuple)), field
    with pytest.raises(Exception):
        p.nbytes = 0  # frozen
