"""#524: fuse the draft-pick sync into ONE host-path collective.

Background. #517 named five BAR1 broadcasts running on the HOST path per
NEXTN decode round (docs/dev/NOTE_517_bar1_guard_desk.md §1.2): three
verify-result syncs and two draft-pick syncs. The three verify-result syncs
were subsequently fused into one by ``pack_accept_payload`` (#616c), so the
current tree issues THREE, not five: one fused verify sync plus the two
draft-pick broadcasts that ``_broadcast_draft_picks`` still emits one per
tensor through ``capture_safe_tp_broadcast``'s loop.

This module pins the remaining fusion: the draft-pick tuple becomes ONE
buffer. Unlike the accept payload, the draft picks have MIXED dtypes
(``topk_index`` is integer, ``topk_p`` is float32 with the SAME shape), so
the fusion is byte-level rather than element-level.

Two properties matter and are tested separately:

* ROUNDTRIP: a mixed-dtype pack/unpack is value-exact, including the float
  bit patterns -- a byte pack must not round-trip through a wider dtype.
* IN PLACE: callers hold other references to these tensors (the returned
  ``EagleDraftInput`` aliases them), so the unpack must write through to the
  SAME objects rather than rebind, exactly as ``unpack_accept_payload`` does.

Hermetic: CPU tensors only, no CUDA, no distributed backend.
"""

import pytest
import torch

from sglang.srt.speculative.spec_utils import (
    draft_pick_payload_bytes,
    pack_draft_picks,
    unpack_draft_picks,
)

BS = 3
TOPK = 4


def _picks():
    """A representative decode-round pick pair: same shape, different dtype."""
    topk_index = torch.arange(BS * TOPK, dtype=torch.int64).reshape(BS, TOPK)
    topk_p = torch.linspace(0.01, 0.99, BS * TOPK, dtype=torch.float32).reshape(
        BS, TOPK
    )
    return topk_index, topk_p


def test_roundtrip_is_value_exact_across_mixed_dtypes():
    topk_index, topk_p = _picks()
    want_index = topk_index.clone()
    want_p = topk_p.clone()

    packed = pack_draft_picks((topk_index, topk_p, None))

    # Simulate the receiving rank: clobber the destinations, then unpack.
    topk_index.zero_()
    topk_p.zero_()
    unpack_draft_picks(packed, (topk_index, topk_p, None))

    assert torch.equal(topk_index, want_index)
    # Bit-exact, not allclose: a byte pack that detoured through a wider
    # dtype would still pass allclose on these values.
    assert torch.equal(topk_p, want_p)


def test_unpack_writes_through_to_the_caller_s_objects():
    topk_index, topk_p = _picks()
    want_p = topk_p.clone()
    alias = topk_p  # what EagleDraftInput holds

    packed = pack_draft_picks((topk_index, topk_p, None))
    topk_p.zero_()
    unpack_draft_picks(packed, (topk_index, topk_p, None))

    assert alias is topk_p
    assert torch.equal(alias, want_p)


def test_none_entries_are_skipped_and_do_not_consume_payload():
    topk_index, topk_p = _picks()
    with_none = pack_draft_picks((topk_index, topk_p, None))
    without = pack_draft_picks((topk_index, topk_p))
    assert with_none.numel() == without.numel()


def test_rejection_sampling_shape_carries_the_third_tensor():
    topk_index, topk_p = _picks()
    draft_probs = torch.rand(BS, 17, dtype=torch.float32)
    packed = pack_draft_picks((topk_index, topk_p, draft_probs))
    assert packed.numel() == draft_pick_payload_bytes((topk_index, topk_p, draft_probs))
    want = draft_probs.clone()
    draft_probs.zero_()
    unpack_draft_picks(packed, (topk_index, topk_p, draft_probs))
    assert torch.equal(draft_probs, want)


def test_fused_length_matches_no_individual_tensor():
    """The #616c property, carried over: a residual desync must be loud.

    A fused payload whose byte length equalled one of its members would let a
    pairing shift substitute that member with nothing for NCCL to reject.
    """
    topk_index, topk_p = _picks()
    tensors = (topk_index, topk_p, None)
    fused = draft_pick_payload_bytes(tensors)
    for t in tensors:
        if t is not None:
            assert fused != t.numel() * t.element_size()


def test_wrong_length_payload_is_refused_by_name():
    topk_index, topk_p = _picks()
    packed = pack_draft_picks((topk_index, topk_p, None))
    truncated = packed[:-1].clone()
    with pytest.raises(ValueError, match="draft-pick payload length mismatch"):
        unpack_draft_picks(truncated, (topk_index, topk_p, None))


def test_non_contiguous_destination_is_refused_rather_than_silently_dropped():
    """reshape(-1) on a non-contiguous tensor returns a COPY, so an in-place
    write through it would update nothing. Refuse loudly instead."""
    topk_index, topk_p = _picks()
    skewed = topk_p.t()  # non-contiguous view
    with pytest.raises(ValueError, match="contiguous"):
        pack_draft_picks((topk_index, skewed, None))
