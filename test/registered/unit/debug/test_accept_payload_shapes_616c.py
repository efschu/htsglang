"""Round-trip, ordering, and error tests for accept-broadcast pack/unpack.

Hermetic CPU-only unit tests for
:func:`sglang.srt.speculative.eagle_utils.pack_accept_payload`, and
:func:`sglang.srt.speculative.eagle_utils.unpack_accept_payload`.

No CUDA, no imports beyond torch and the three functions.
"""

import pytest
import torch

from sglang.srt.speculative.eagle_utils import (
    pack_accept_payload,
    unpack_accept_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensors(bs: int, width: int, device=torch.device("cpu")):
    """Create the three tensors with known content, all int32 on CPU."""
    predict = torch.arange(bs * width, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, width), -1, dtype=torch.int32, device=device)
    # Fill with distinct per-slot values so any mix-up is detectable.
    for i in range(bs):
        accept_index[i] = torch.arange(width, dtype=torch.int32, device=device) - 100
    num_correct_drafts = torch.arange(bs, dtype=torch.int32, device=device) + 1
    return predict, accept_index, num_correct_drafts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAcceptPayloadRoundTrip:
    """pack -> unpack preserves every tensor byte-for-byte."""

    def test_smallest_case_bs1_w1(self):
        """bs=1, width=1 -- the absolute minimal configuration."""
        predict, accept_index, num_correct_drafts = _make_tensors(1, 1)
        expect_predict = predict.clone()
        expect_accept_index = accept_index.clone()
        expect_num_correct = num_correct_drafts.clone()

        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
        unpack_accept_payload(packed, predict, accept_index, num_correct_drafts)

        torch.testing.assert_close(predict, expect_predict)
        torch.testing.assert_close(accept_index, expect_accept_index)
        torch.testing.assert_close(num_correct_drafts, expect_num_correct)

    def test_bs8_w6_round_trip(self):
        """bs=8, width=6 -- realistic batch."""
        predict, accept_index, num_correct_drafts = _make_tensors(8, 6)
        expect_predict = predict.clone()
        expect_accept_index = accept_index.clone()
        expect_num_correct = num_correct_drafts.clone()

        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
        unpack_accept_payload(packed, predict, accept_index, num_correct_drafts)

        torch.testing.assert_close(predict, expect_predict)
        torch.testing.assert_close(accept_index, expect_accept_index)
        torch.testing.assert_close(num_correct_drafts, expect_num_correct)

    def test_packed_numel_is_sum_of_parts(self):
        """packed.numel() == predict.numel() + accept_index.numel() + num_correct_drafts.numel()."""
        bs, width = 8, 6
        predict, accept_index, num_correct_drafts = _make_tensors(bs, width)
        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

        assert packed.numel() == (
            predict.numel() + accept_index.numel() + num_correct_drafts.numel()
        )


class TestUnpackInPlace:
    """unpack_accept_payload mutates the original tensor objects."""

    def test_same_object_id_before_and_after(self):
        """id(tensor) is unchanged -- no re-binding occurs."""
        predict, accept_index, num_correct_drafts = _make_tensors(4, 3)
        ids_before = (id(predict), id(accept_index), id(num_correct_drafts))

        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
        unpack_accept_payload(packed, predict, accept_index, num_correct_drafts)

        ids_after = (id(predict), id(accept_index), id(num_correct_drafts))
        assert ids_before == ids_after, "unpack_accept_payload must write in place"

    def test_shapes_unchanged_after_unpack(self):
        """Tensor shapes are preserved through the round-trip."""
        predict, accept_index, num_correct_drafts = _make_tensors(5, 4)
        shapes_before = (predict.shape, accept_index.shape, num_correct_drafts.shape)

        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
        unpack_accept_payload(packed, predict, accept_index, num_correct_drafts)

        shapes_after = (predict.shape, accept_index.shape, num_correct_drafts.shape)
        assert shapes_before == shapes_after


class TestPackingOrder:
    """Verify the wire layout: predict | accept_index | num_correct_drafts."""

    def test_predict_is_first_segment(self):
        """packed[:predict.numel()] == predict.flatten()."""
        bs, width = 3, 5
        predict, accept_index, num_correct_drafts = _make_tensors(bs, width)
        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

        torch.testing.assert_close(packed[: predict.numel()], predict.reshape(-1))

    def test_accept_index_is_middle_segment(self):
        """packed[predict_len : predict_len + accept_index.numel()] == accept_index.flatten()."""
        bs, width = 3, 5
        predict, accept_index, num_correct_drafts = _make_tensors(bs, width)
        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

        n_p = predict.numel()
        n_a = accept_index.numel()
        torch.testing.assert_close(packed[n_p : n_p + n_a], accept_index.reshape(-1))

    def test_num_correct_drafts_is_last_segment(self):
        """packed[predict_len + accept_index.numel() :] == num_correct_drafts.flatten()."""
        bs, width = 3, 5
        predict, accept_index, num_correct_drafts = _make_tensors(bs, width)
        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

        n_p = predict.numel()
        n_a = accept_index.numel()
        torch.testing.assert_close(packed[n_p + n_a :], num_correct_drafts.reshape(-1))


class TestNegativeValues:
    """accept_index legitimately holds -1 padding; values must survive."""

    def test_negative_values_survive_round_trip(self):
        """-1 (and other negatives) pass through pack/unpack unchanged."""
        bs, width = 2, 4
        predict = torch.full((bs * width,), 7, dtype=torch.int32)
        accept_index = torch.full((bs, width), -1, dtype=torch.int32)
        num_correct_drafts = torch.full((bs,), 0, dtype=torch.int32)

        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
        unpack_accept_payload(packed, predict, accept_index, num_correct_drafts)

        assert (accept_index == -1).all(), (
            "Negative padding values (-1) must survive the round-trip"
        )
        assert (predict == 7).all()
        assert (num_correct_drafts == 0).all()


class TestPackedDtype:
    """The packed buffer must be torch.int32."""

    def test_packed_dtype_is_int32(self):
        predict, accept_index, num_correct_drafts = _make_tensors(2, 3)
        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
        assert packed.dtype == torch.int32, (
            f"packed buffer dtype must be int32, got {packed.dtype}"
        )


class TestWrongLengthError:
    """unpack_accept_payload raises on a truncated packed buffer."""

    def test_truncated_buffer_raises_valueerror(self):
        """Truncating the packed buffer by one element must raise ValueError."""
        bs, width = 4, 3
        predict, accept_index, num_correct_drafts = _make_tensors(bs, width)
        packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

        truncated = packed[:-1]  # one element short
        with pytest.raises(ValueError, match="length mismatch"):
            unpack_accept_payload(truncated, predict, accept_index, num_correct_drafts)
