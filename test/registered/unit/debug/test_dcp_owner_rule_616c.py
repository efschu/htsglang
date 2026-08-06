# Copyright 2023-2026 SGLang Team
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

"""Pure-CPU unit tests for the weighted DCP owner rule (#616c).

Exercises dcp_weighted_read_slots and dcp_weighted_owned_lengths with
explicit hand-computed expected values -- no CUDA, no collectives.
"""

import pytest
import torch

from sglang.srt.layers.dcp.owner import (
    dcp_weighted_owned_lengths,
    dcp_weighted_read_slots,
)


class TestOwnershipPredicate:
    """Test 1: ownership mask for cp_S=4, cp_lo=1, cp_hi=3 on locs 0..15."""

    def test_owner_mask_explicit(self):
        # loc % 4 in [1, 3) means remainder is 1 or 2.
        # loc:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        # mod4: 0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3
        # own?  F  T  T  F  F  T  T  F  F  T  T  F  F  T  T  F
        full_kv = torch.arange(16, dtype=torch.int32)
        _, owned = dcp_weighted_read_slots(
            full_kv, cp_S=4, cp_lo=1, cp_hi=3, cp_ratio=2
        )

        expected = torch.tensor(
            [
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        )
        assert owned.tolist() == expected.tolist(), (
            f"ownership mask mismatch: got {owned.tolist()}"
        )


class TestCompactMapping:
    """Test 2: compact slot values for the owned locs from Test 1."""

    def test_compact_values_explicit(self):
        # Owned locs (from test 1): 1,2,5,6,9,10,13,14
        # Formula: (L // 4) * 2 + (L % 4 - 1)
        # L=1:  (0)*2 + (1-1) = 0
        # L=2:  (0)*2 + (2-1) = 1
        # L=5:  (1)*2 + (1-1) = 2
        # L=6:  (1)*2 + (2-1) = 3
        # L=9:  (2)*2 + (1-1) = 4
        # L=10: (2)*2 + (2-1) = 5
        # L=13: (3)*2 + (1-1) = 6
        # L=14: (3)*2 + (2-1) = 7
        full_kv = torch.arange(16, dtype=torch.int32)
        compact, owned = dcp_weighted_read_slots(
            full_kv, cp_S=4, cp_lo=1, cp_hi=3, cp_ratio=2
        )
        owned_compact = compact[owned]

        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
        assert owned_compact.tolist() == expected.tolist(), (
            f"compact mapping mismatch: got {owned_compact.tolist()}"
        )


class TestDisjointCoverage:
    """Test 3: three ranks with cp_S=3 and windows (0,1),(1,2),(2,3) cover
    every slot in 0..20 exactly once."""

    @pytest.fixture(scope="class")
    def rank_masks(self):
        full_kv = torch.arange(21, dtype=torch.int32)
        # Rank 0: owns remainder 0
        _, mask0 = dcp_weighted_read_slots(
            full_kv, cp_S=3, cp_lo=0, cp_hi=1, cp_ratio=1
        )
        # Rank 1: owns remainder 1
        _, mask1 = dcp_weighted_read_slots(
            full_kv, cp_S=3, cp_lo=1, cp_hi=2, cp_ratio=1
        )
        # Rank 2: owns remainder 2
        _, mask2 = dcp_weighted_read_slots(
            full_kv, cp_S=3, cp_lo=2, cp_hi=3, cp_ratio=1
        )
        return mask0, mask1, mask2

    def test_sum_is_all_ones(self, rank_masks):
        mask0, mask1, mask2 = rank_masks
        total = mask0.to(torch.int64) + mask1.to(torch.int64) + mask2.to(torch.int64)
        expected = torch.ones(21, dtype=torch.int64)
        assert total.tolist() == expected.tolist(), (
            "combined masks do not cover every slot exactly once"
        )

    def test_masks_are_disjoint(self, rank_masks):
        mask0, mask1, mask2 = rank_masks
        assert not (mask0 & mask1).any(), "rank0 and rank1 overlap"
        assert not (mask0 & mask2).any(), "rank0 and rank2 overlap"
        assert not (mask1 & mask2).any(), "rank1 and rank2 overlap"


class TestOwnedLengths:
    """Test 4: dcp_weighted_owned_lengths does a correct segmented sum."""

    def test_two_request_counts(self):
        # Two requests with lengths [5, 7], total 12 slots.
        # Request 0 owns slots at indices 0-4, request 1 at indices 5-11.
        # Hand-written ownership:
        #   Index:  0  1  2  3  4  |  5  6  7  8  9 10 11
        #   Loc:    0  1  2  3  4  |  5  6  7  8  9 10 11
        #   mod4:   0  1  2  3  0  |  1  2  3  0  1  2  3
        #   own(1-3): F T T F F  |  T  T F  F  T  T  F
        #   Req0 owned:  2  (indices 1,2)
        #   Req1 owned:  4  (indices 5,6,9,10)
        owned = torch.tensor(
            [
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        )
        lens64 = torch.tensor([5, 7], dtype=torch.int64)
        result = dcp_weighted_owned_lengths(owned, lens64)

        expected = torch.tensor([2, 4], dtype=torch.int64)
        assert result.tolist() == expected.tolist(), (
            f"owned lengths mismatch: got {result.tolist()}"
        )

    def test_counts_sum_to_total_owned(self):
        """The sum of per-request counts must equal total owned slots."""
        owned = torch.tensor(
            [
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
            ],
            dtype=torch.bool,
        )
        lens64 = torch.tensor([5, 7], dtype=torch.int64)
        result = dcp_weighted_owned_lengths(owned, lens64)
        assert result.sum().item() == int(owned.sum().item()), (
            "segmented sum does not equal total owned count"
        )


class TestEmptyInput:
    """Test 5: zero-length inputs produce empty outputs without errors."""

    def test_empty_read_slots(self):
        full_kv = torch.empty(0, dtype=torch.int32)
        compact, owned = dcp_weighted_read_slots(
            full_kv, cp_S=4, cp_lo=1, cp_hi=3, cp_ratio=2
        )
        assert compact.numel() == 0, "compact should be empty"
        assert owned.numel() == 0, "owned should be empty"

    def test_empty_owned_lengths(self):
        owned = torch.empty(0, dtype=torch.bool)
        lens64 = torch.tensor([5, 7], dtype=torch.int64)
        result = dcp_weighted_owned_lengths(owned, lens64)
        assert result.tolist() == [0, 0], (
            "empty owned mask should yield zero counts per request"
        )
