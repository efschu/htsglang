# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MoE expert-offload planner + wave logic (Variant-C
B2b fixed-resident + scratch model).

No CUDA required. Exercises:
  * ``ExpertResidencyPlanner`` -- fixed resident set [0,R) at slot==id, spill
    experts (>=R) fetched into scratch slots [R, R+C) in sorted order,
    history-independent (deterministic);
  * ``plan_token_waves`` -- partition tokens into waves whose union of unique
    SPILL experts is <= scratch (resident experts impose no wave budget).

Run:  python -m pytest tests/moe_offload/test_planner.py -q
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "python"),
)

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertResidencyPlanner,
    plan_token_waves,
    resident_slot_count,
    scratch_slot_count,
)


def _all_rows(waves):
    return [t for w in waves for t in w]


# --------------------------------------------------------------------------- #
# plan_token_waves (spill-based)
# --------------------------------------------------------------------------- #
def test_single_wave_when_all_resident():
    # R=4: experts {0,1,2,3} are all resident -> no spill -> one wave regardless.
    experts = [[0, 1], [1, 2], [2, 3], [0, 3]]
    waves = plan_token_waves(experts, resident_count=4, scratch=2)
    assert len(waves) == 1
    assert waves[0] == [0, 1, 2, 3]


def test_resident_experts_impose_no_wave_budget():
    # All experts < R: any number of them coexist in one wave (scratch unused).
    experts = [[e % 8] for e in range(50)]  # all in [0,8) resident
    waves = plan_token_waves(experts, resident_count=8, scratch=2)
    assert len(waves) == 1
    assert _all_rows(waves) == list(range(50))


def test_spill_experts_split_by_scratch():
    # R=4, scratch=2. Tokens each route to one distinct SPILL expert (>=4);
    # unique spill per wave must be <= 2.
    experts = [[4], [5], [6], [7], [8]]  # 5 distinct spill experts
    waves = plan_token_waves(experts, resident_count=4, scratch=2)
    assert len(waves) == 3  # ceil(5/2)
    for w in waves:
        spill = {e for t in w for e in experts[t] if e >= 4}
        assert len(spill) <= 2
    assert _all_rows(waves) == list(range(5))


def test_mixed_resident_and_spill():
    # Resident experts free; only spill counts. top-2 = one resident + one spill.
    experts = [[0, 10], [1, 11], [2, 12]]  # spill {10,11,12}
    waves = plan_token_waves(experts, resident_count=4, scratch=2)
    # spill unique across all = 3 > 2 -> at least 2 waves.
    assert len(waves) >= 2
    for w in waves:
        spill = {e for t in w for e in experts[t] if e >= 4}
        assert len(spill) <= 2


def test_padding_minus_one_ignored():
    experts = [[0, -1], [10, -1], [-1, -1]]
    waves = plan_token_waves(experts, resident_count=4, scratch=2)
    # spill {10} <= 2 -> single wave; -1 padding never counted.
    assert len(waves) == 1
    assert waves[0] == [0, 1, 2]


def test_single_token_exceeds_scratch_raises():
    # A token routing to more unique SPILL experts than scratch is unservable.
    experts = [[10, 11, 12]]  # 3 spill, scratch=2
    with pytest.raises(ValueError):
        plan_token_waves(experts, resident_count=4, scratch=2)


# --------------------------------------------------------------------------- #
# ExpertResidencyPlanner (fixed-resident + scratch)
# --------------------------------------------------------------------------- #
def test_resident_experts_map_to_id_no_fetch():
    p = ExpertResidencyPlanner(num_local_experts=256, resident_count=64, scratch=16)
    slot_of, fetch = p.resolve([0, 1, 63])  # all resident
    assert fetch == []
    assert slot_of == {0: 0, 1: 1, 63: 63}
    assert p.stats.fetches == 0


def test_spill_experts_fetched_into_sorted_scratch():
    p = ExpertResidencyPlanner(num_local_experts=256, resident_count=64, scratch=16)
    # resident 0,5 -> slots 0,5; spill 200,70,150 (sorted 70,150,200) -> scratch.
    slot_of, fetch = p.resolve([200, 5, 70, 150, 0])
    assert slot_of[0] == 0 and slot_of[5] == 5  # resident at id
    assert slot_of[70] == 64 and slot_of[150] == 65 and slot_of[200] == 66
    assert fetch == [(70, 64), (150, 65), (200, 66)]  # sorted spill -> scratch


def test_resolve_is_deterministic_history_independent():
    p = ExpertResidencyPlanner(num_local_experts=256, resident_count=64, scratch=16)
    a_slot, a_fetch = p.resolve([200, 70, 5])
    p.resolve([99, 100, 101])  # intervening call
    b_slot, b_fetch = p.resolve([200, 70, 5])
    assert a_slot == b_slot and a_fetch == b_fetch  # pure function of needed set


def test_resolve_rejects_spill_overflow():
    p = ExpertResidencyPlanner(num_local_experts=256, resident_count=4, scratch=2)
    with pytest.raises(RuntimeError):
        p.resolve([10, 11, 12])  # 3 spill > scratch 2


def test_buffer_size():
    p = ExpertResidencyPlanner(num_local_experts=256, resident_count=64, scratch=16)
    assert p.buffer_size == 80
    # capped at E
    q = ExpertResidencyPlanner(num_local_experts=70, resident_count=64, scratch=16)
    assert q.buffer_size == 70


def test_wave_by_wave_serves_overflow_batch():
    """Partition an overflow batch into waves and resolve each; every needed
    expert lands in a valid in-buffer slot, resident at id and spill in scratch,
    and no wave exceeds the scratch budget."""
    E, top_k, T = 256, 8, 128
    R, C = 64, 16
    rng = __import__("random").Random(7)
    experts = [rng.sample(range(E), top_k) for _ in range(T)]
    p = ExpertResidencyPlanner(num_local_experts=E, resident_count=R, scratch=C)
    waves = plan_token_waves(experts, resident_count=R, scratch=C)
    for w in waves:
        needed = sorted({e for t in w for e in experts[t]})
        slot_of, fetch = p.resolve(needed)
        assert set(slot_of.keys()) == set(needed)
        slots = list(slot_of.values())
        assert len(set(slots)) == len(slots)  # distinct slots
        assert all(0 <= s < R + C for s in slots)
        # resident experts map to their id; spill land in scratch.
        for e, s in slot_of.items():
            if e < R:
                assert s == e
            else:
                assert R <= s < R + C
        assert len(fetch) <= C


def test_resident_slot_count():
    assert resident_slot_count(256, 1.0) == 256
    assert resident_slot_count(256, 0.25) == 64
    assert resident_slot_count(256, 0.0) == 1  # clamped to >= 1


def test_scratch_slot_count_default_and_override():
    assert scratch_slot_count(64) == 16  # max(8, 64//4)
    assert scratch_slot_count(16) == 8  # floor of 8
    os.environ["SGLANG_MOE_SCRATCH_SLOTS"] = "24"
    try:
        assert scratch_slot_count(64) == 24
    finally:
        del os.environ["SGLANG_MOE_SCRATCH_SLOTS"]


def test_fully_resident_passthrough():
    p = ExpertResidencyPlanner(num_local_experts=8, resident_count=8, scratch=4)
    assert p.fully_resident
    slot_of, fetch = p.resolve([0, 3, 7])
    assert fetch == []
    assert slot_of == {0: 0, 3: 3, 7: 7}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
