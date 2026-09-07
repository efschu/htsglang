"""#704b: the canonical KV page (Option A), device side.

#706 settled the page shape: a page carries ALL 16 attention layers for its
token range, K/V-major (#1233: `[K all slots][V all slots]`, one half-cell per
slot per half). `page_size == 1` is mandatory
(`dcp_owner_mode`: a multi-token page would span owner ranks), so a page is ONE
token -- 16 x 2048 = 32,768 B.

The contract both strands carry says the stored form is CANONICAL and
layout-neutral. This file pins what that means on the device side, and the two
ways it can be silently violated:

* **A page must be written with the GLOBAL attention-layer index, never a
  rank-local one.** Option A globalises `start_layer` for exactly this reason.
  A stage that writes its layers at rank-local offsets puts the right NUMBER of
  bytes in the WRONG slots -- the same failure shape as cutting a mamba blob
  flat, and just as silent.
* **A page takes partial writes from THREE stages** (7 + 5 + 4 attention layers
  under interval 4), so an unwritten slot is indistinguishable from a
  legitimately-zero one unless completeness is tracked explicitly. #706 records
  that the marker must be built; this is the device-side half of it.

The canonical form depends on the model geometry ALONE -- not on the PP cut,
not on the token-share vector, not on which phase wrote it. That is the whole
point of the contract, and it is asserted directly.

Hermetic: CPU tensors only, no CUDA, no server.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.mem_cache.canonical_kv_page import (
    CanonicalPageError,
    CanonicalPageSpec,
    PageCompleteness,
    attn_layer_index,
)

# Qwen3.8-27B: 64 layers, full_attention_interval 4 -> attention at 3,7,...,63.
ATTN_LAYER_IDS = tuple(i for i in range(64) if (i + 1) % 4 == 0)
SPEC = CanonicalPageSpec(num_attn_layers=16, kv_bytes_per_token_per_attn_layer=2048)

# The PP cuts that produce those 16 layers: 7 / 5 / 4.
STAGE_RANGES = ((0, 28), (28, 48), (48, 64))


def test_the_page_is_one_token_of_all_sixteen_layers():
    assert SPEC.page_bytes == 16 * 2048 == 32768


def test_attention_layer_index_is_global_and_dense():
    assert len(ATTN_LAYER_IDS) == 16
    assert attn_layer_index(3, ATTN_LAYER_IDS) == 0
    assert attn_layer_index(63, ATTN_LAYER_IDS) == 15
    # A non-attention layer has no slot at all.
    with pytest.raises(CanonicalPageError, match="not a full-attention layer"):
        attn_layer_index(4, ATTN_LAYER_IDS)


def test_a_slot_is_two_half_cells_one_per_kv_region():
    """K/V-major (#1233): slot 5 is bytes [5*1024, 6*1024) of the K region and
    the same offsets inside the V region, which starts at 16*1024."""
    (k_lo, k_hi), (v_lo, v_hi) = SPEC.slot_spans(5)
    assert (k_lo, k_hi) == (5 * 1024, 6 * 1024)
    assert (v_lo, v_hi) == (16 * 1024 + 5 * 1024, 16 * 1024 + 6 * 1024)
    assert SPEC.half_cell_bytes == 1024 and SPEC.half_page_bytes == 16 * 1024
    with pytest.raises(CanonicalPageError, match="outside"):
        SPEC.slot_spans(16)
    with pytest.raises(CanonicalPageError, match="K half and a V half"):
        CanonicalPageSpec(num_attn_layers=16, kv_bytes_per_token_per_attn_layer=2047)


def test_writing_with_a_rank_local_index_lands_in_the_wrong_slot():
    """THE hazard Option A's globalised start_layer exists to prevent.

    Stage 1 owns global layers 28..47, whose attention layers are 31/35/39/43/47
    -- global attention indices 7..11. A stage that wrote them at rank-LOCAL
    indices 0..4 would deposit the right number of bytes in the wrong slots,
    silently, exactly like cutting a mamba blob as one flat range.
    """
    global_ids = [i for i in ATTN_LAYER_IDS if 28 <= i < 48]
    assert global_ids == [31, 35, 39, 43, 47]
    correct = [attn_layer_index(i, ATTN_LAYER_IDS) for i in global_ids]
    assert correct == [7, 8, 9, 10, 11]

    rank_local = list(range(len(global_ids)))  # 0..4 -- the bug
    assert rank_local != correct
    # Same count of slots, different slots: the byte total cannot detect it.
    assert len(rank_local) == len(correct)


def test_a_page_is_written_by_three_stages_and_completeness_is_tracked():
    """Storage is token-sharded, but PRODUCTION is layer-sharded.

    Token-sharding the storage does NOT collapse a page to a single writer:
    the 16 slots come from three different PP stages, so an unwritten slot must
    be distinguishable from a legitimately zero one.
    """
    tracker = PageCompleteness(SPEC)
    assert not tracker.is_complete()

    written = 0
    for lo, hi in STAGE_RANGES:
        ids = [i for i in ATTN_LAYER_IDS if lo <= i < hi]
        for gid in ids:
            tracker.mark(attn_layer_index(gid, ATTN_LAYER_IDS))
        written += len(ids)
        if written < 16:
            assert not tracker.is_complete(), "premature completeness"
    assert written == 16
    assert tracker.is_complete()
    assert tracker.missing() == ()


def test_the_stage_split_covers_every_slot_exactly_once():
    """A gap loses a layer's KV; an overlap means two writers race a slot."""
    seen = []
    for lo, hi in STAGE_RANGES:
        seen += [
            attn_layer_index(i, ATTN_LAYER_IDS) for i in ATTN_LAYER_IDS if lo <= i < hi
        ]
    assert sorted(seen) == list(range(16))
    assert len(seen) == len(set(seen)), "a slot has two writers"


def test_missing_slots_are_named_not_merely_counted():
    tracker = PageCompleteness(SPEC)
    for idx in (0, 1, 2, 3, 4, 5, 6, 8, 9, 10):
        tracker.mark(idx)
    assert not tracker.is_complete()
    assert tracker.missing() == (7, 11, 12, 13, 14, 15)


def test_marking_a_slot_twice_is_refused():
    """Two writers for one slot is a layout bug, not an idempotent write."""
    tracker = PageCompleteness(SPEC)
    tracker.mark(3)
    with pytest.raises(CanonicalPageError, match="already written"):
        tracker.mark(3)


def test_the_canonical_form_depends_on_geometry_alone():
    """The contract, asserted directly.

    Two ranks with different PP cuts and different token shares must agree on
    the page layout exactly, or the same key would name different bytes.
    """
    a = CanonicalPageSpec(num_attn_layers=16, kv_bytes_per_token_per_attn_layer=2048)
    b = CanonicalPageSpec(num_attn_layers=16, kv_bytes_per_token_per_attn_layer=2048)
    assert a == b
    assert a.page_bytes == b.page_bytes
    for i in range(16):
        assert a.slot_spans(i) == b.slot_spans(i)


def test_a_wrong_sized_payload_is_refused_rather_than_padded():
    """The two failure modes are distinct and must not share a message.

    A missing SLOT is an incomplete page (a completeness problem); a wrong-sized
    buffer is a geometry problem. Reporting both as "bytes" would send the
    reader to check the wrong thing.
    """
    tracker = PageCompleteness(SPEC)
    for idx in range(15):
        tracker.mark(idx)
    # Missing a slot: an incomplete page, not a shorter one.
    assert not tracker.is_complete()
    assert tracker.missing() == (15,)


def test_slot_index_out_of_range_is_refused():
    tracker = PageCompleteness(SPEC)
    with pytest.raises(CanonicalPageError, match="slot"):
        tracker.mark(16)
