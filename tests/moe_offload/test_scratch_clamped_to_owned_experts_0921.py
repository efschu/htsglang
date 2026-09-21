"""fnFL2v64 (21.09.): C must be clamped to the rows the rank OWNS.

The staging plan builds the buffer as ``min(R + C, E)`` -- a rank cannot hold
more rows than it owns -- while the pool-mode capture asserts
``buffer_size == R + C``.  With E=61, R=2 and the default C=60 the two
disagree and the decode graph capture refuses:

    Capture cuda graph failed: pool mode requires buffer_size == R+C (61 != 2+60)

The clamp belongs in ``scratch_slot_count``, the single source both sides
read, so the identity holds by construction instead of by coincidence.
"""

import pytest

from sglang.srt.layers.moe.expert_offload import (
    plan_load_time_staging,
    scratch_slot_count,
)


def test_without_the_expert_count_nothing_changes(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_SCRATCH_SLOTS", raising=False)
    assert scratch_slot_count(40) == 10  # max(8, 40 // 4)
    assert scratch_slot_count(4) == 8  # the floor of 8


def test_it_clamps_to_the_rows_that_exist(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_SCRATCH_SLOTS", raising=False)
    # the fnFL2v64 rank: 61 owned, 2 resident -> at most 59 scratch
    assert scratch_slot_count(2, 61) == 8  # default is already below the room
    assert scratch_slot_count(200, 300) == 50  # want 50, room 100 -> 50
    assert scratch_slot_count(250, 300) == 50  # want 62, room 50 -> clamped
    assert scratch_slot_count(59, 61) == 2  # room 2 is the minimum that passes


def test_the_env_override_is_clamped_too(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_SCRATCH_SLOTS", "60")
    assert scratch_slot_count(2) == 60  # unclamped: the v64 value
    assert scratch_slot_count(2, 61) == 59  # clamped to the room


def test_a_rank_without_room_is_refused_not_clamped(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_SCRATCH_SLOTS", raising=False)
    with pytest.raises(ValueError, match="scratch row"):
        scratch_slot_count(60, 61)  # room 1: the staging width C-1 would be 0
    with pytest.raises(ValueError, match="scratch row"):
        scratch_slot_count(61, 61)  # room 0


@pytest.mark.parametrize(
    "E,fraction", [(61, 0.0188), (61, 0.5), (128, 0.25), (64, 0.9)]
)
def test_the_plan_satisfies_the_pool_identity(E, fraction, monkeypatch):
    """buffer_slots == R + C -- the assert the capture makes, on real shapes."""
    monkeypatch.delenv("SGLANG_MOE_SCRATCH_SLOTS", raising=False)
    plan = plan_load_time_staging(E, fraction)
    assert plan is not None, (E, fraction)
    R = plan.resident_count
    C = scratch_slot_count(R, E)
    assert plan.buffer_slots == R + C, (E, fraction, R, C, plan.buffer_slots)
    assert plan.buffer_slots <= E
    assert C >= 2  # the staging width is C-1 and must stay >= 1
