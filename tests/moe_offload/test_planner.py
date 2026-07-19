# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MoE expert-offload TIERED planner + wave logic.

No CUDA required. Exercises:
  * ``plan_token_waves`` -- token-partition wave splitting that counts only
    SPILL experts against the cache size C (resident experts are free);
  * ``ExpertResidencyPlanner`` -- resident-direct vs spill-cache routing with
    DETERMINISTIC, history-independent slot assignment (the reproducibility
    fix: identical inputs -> identical slot layout -> identical output, run to
    run and request to request).

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
    default_spill_slots,
    plan_token_waves,
    resident_slot_count,
)


# --------------------------------------------------------------------------- #
# plan_token_waves  (n_resident == 0: every expert counts, pin-everything case)
# --------------------------------------------------------------------------- #
def _all_rows(waves):
    return [t for w in waves for t in w]


def test_single_wave_when_union_fits():
    experts = [[0, 1], [1, 2], [2, 3], [0, 3]]
    waves = plan_token_waves(experts, n_slots=4)
    assert len(waves) == 1
    assert waves[0] == [0, 1, 2, 3]


def test_partition_is_complete_and_ordered():
    experts = [[e, e + 1] for e in range(0, 40, 2)]
    waves = plan_token_waves(experts, n_slots=6)
    rows = _all_rows(waves)
    assert rows == list(range(len(experts)))
    assert sorted(rows) == rows


def test_each_wave_union_within_n_slots():
    rng = __import__("random").Random(1234)
    E, top_k, T, n_slots = 256, 8, 300, 64
    experts = [rng.sample(range(E), top_k) for _ in range(T)]
    waves = plan_token_waves(experts, n_slots=n_slots)
    assert len(waves) > 1
    for w in waves:
        union = set()
        for t in w:
            union |= set(experts[t])
        assert len(union) <= n_slots
    assert _all_rows(waves) == list(range(T))


def test_overflow_needs_more_experts_than_n_slots_does_not_crash():
    n_slots, K = 16, 10
    experts = [[e] for e in range(n_slots + K)]
    waves = plan_token_waves(experts, n_slots=n_slots)
    assert len(waves) >= 2
    assert _all_rows(waves) == list(range(n_slots + K))


def test_padding_minus_one_ignored():
    experts = [[0, -1], [1, -1], [-1, -1]]
    waves = plan_token_waves(experts, n_slots=2)
    assert len(waves) == 1
    assert waves[0] == [0, 1, 2]


def test_single_token_exceeds_n_slots_raises():
    experts = [[0, 1, 2, 3]]
    with pytest.raises(ValueError):
        plan_token_waves(experts, n_slots=3)


# --------------------------------------------------------------------------- #
# plan_token_waves  (TIERED: resident experts do not consume spill slots)
# --------------------------------------------------------------------------- #
def test_resident_experts_never_count_toward_waves():
    experts = [[i % 100] for i in range(500)]
    waves = plan_token_waves(experts, n_slots=4, n_resident=100)
    assert len(waves) == 1
    assert _all_rows(waves) == list(range(500))


def test_only_spill_experts_drive_wave_split():
    experts = [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]]
    waves = plan_token_waves(experts, n_slots=2, n_resident=10)
    assert len(waves) >= 3
    for w in waves:
        spill_union = {e for t in w for e in experts[t] if e >= 10}
        assert len(spill_union) <= 2
    assert _all_rows(waves) == list(range(5))


def test_single_token_spill_over_cache_raises():
    experts = [[0, 10, 11, 12]]  # 3 spill experts, C=2
    with pytest.raises(ValueError):
        plan_token_waves(experts, n_slots=2, n_resident=10)


# --------------------------------------------------------------------------- #
# ExpertResidencyPlanner -- deterministic slot assignment
# --------------------------------------------------------------------------- #
def test_warm_start_cache():
    # n_resident=0 (pin-everything case): slots [0,C) warm-started with [0,C).
    p = ExpertResidencyPlanner(num_local_experts=256, n_resident=0, n_spill_slots=64)
    for i in range(64):
        assert p._slot_contents[i] == i


def test_tiered_layout_and_buffer():
    p = ExpertResidencyPlanner(num_local_experts=256, n_resident=192, n_spill_slots=8)
    assert p.buffer_slots == 200
    assert not p.fully_resident
    assert p._slot_contents == {192 + i: 192 + i for i in range(8)}


def test_resident_direct_no_fetch():
    p = ExpertResidencyPlanner(num_local_experts=256, n_resident=192, n_spill_slots=8)
    slot_of, fetch = p.resolve([0, 5, 191])  # all resident
    assert fetch == []
    assert slot_of == {0: 0, 5: 5, 191: 191}
    assert p.stats.resident_hits == 3
    assert p.stats.fetches == 0


def test_spill_packs_to_sorted_low_slots():
    # C=8 spill slots at [192,200). Needed spill experts pack, sorted, to
    # slots 192,193,... regardless of the experts' absolute ids.
    p = ExpertResidencyPlanner(num_local_experts=256, n_resident=192, n_spill_slots=8)
    slot_of, fetch = p.resolve([250, 200, 199])  # spill, unsorted input
    # sorted spill = [199, 200, 250] -> slots 192, 193, 194
    assert slot_of == {199: 192, 200: 193, 250: 194}
    # each target slot warm-held a different expert -> all three fetch.
    assert sorted(fetch) == [(199, 192), (200, 193), (250, 194)]


def test_warm_spill_hit_avoids_fetch():
    # Requesting exactly the warm-start experts in order -> already in target
    # slots -> no fetch.
    p = ExpertResidencyPlanner(num_local_experts=256, n_resident=192, n_spill_slots=8)
    slot_of, fetch = p.resolve([192, 193, 194])
    assert slot_of == {192: 192, 193: 193, 194: 194}
    assert fetch == []
    assert p.stats.hits == 3


def test_resolve_is_history_independent():
    # THE reproducibility property: resolve() output depends only on the current
    # needed set, never on prior calls. Two planners driven by DIFFERENT
    # histories give IDENTICAL slot_of for the same final wave.
    p1 = ExpertResidencyPlanner(num_local_experts=256, n_resident=128, n_spill_slots=8)
    p2 = ExpertResidencyPlanner(num_local_experts=256, n_resident=128, n_spill_slots=8)
    p1.resolve([130, 131, 132])
    p2.resolve([200, 201, 202, 203, 204])
    p2.resolve([150])
    wave = [10, 250, 129, 199]
    s1, _ = p1.resolve(wave)
    s2, _ = p2.resolve(wave)
    assert s1 == s2, "slot assignment must be history-independent (reproducible)"
    # deterministic sorted layout (R=128): resident 10 direct; spill sorted
    # [129, 199, 250] -> slots 128, 129, 130.
    assert s1 == {10: 10, 129: 128, 199: 129, 250: 130}


def test_full_spill_wave_uses_all_slots():
    R, C, N = 4, 4, 256
    p = ExpertResidencyPlanner(num_local_experts=N, n_resident=R, n_spill_slots=C)
    needed = [200, 201, 202, 203]  # 4 spill == C
    slot_of, fetch = p.resolve(needed)
    assert set(slot_of.keys()) == set(needed)
    assert sorted(slot_of.values()) == [4, 5, 6, 7]  # all spill slots, distinct


def test_resolve_rejects_overflow_wave():
    p = ExpertResidencyPlanner(num_local_experts=256, n_resident=0, n_spill_slots=4)
    with pytest.raises(RuntimeError):
        p.resolve([10, 11, 12, 13, 14])


def test_tiered_wave_by_wave_serves_overflow_with_spill_gt_C():
    """End-to-end tiered planner over an overflow batch (spill experts > C):
    every needed expert resolves to a valid, distinct-per-wave slot; resident
    experts served directly; layout is a pure function of each wave."""
    E, top_k, T = 256, 8, 128
    R, C = 192, 8
    rng = __import__("random").Random(7)
    experts = [rng.sample(range(E), top_k) for _ in range(T)]
    p = ExpertResidencyPlanner(num_local_experts=E, n_resident=R, n_spill_slots=C)
    waves = plan_token_waves(experts, n_slots=C, n_resident=R)
    assert len(waves) > 1
    for w in waves:
        needed = sorted({e for t in w for e in experts[t]})
        slot_of, _ = p.resolve(needed)
        assert set(slot_of.keys()) == set(needed)
        for e, s in slot_of.items():
            if e < R:
                assert s == e            # resident direct
            else:
                assert R <= s < R + C    # spill cache slot
        spill_slots = [s for e, s in slot_of.items() if e >= R]
        assert len(set(spill_slots)) == len(spill_slots)  # distinct within wave


def test_self_reproducible_over_a_forward_sequence():
    # Replaying the SAME sequence of waves on a fresh planner (as a second
    # identical request would) yields the identical slot layout at every step.
    R, C, N = 128, 8, 256
    rng = __import__("random").Random(11)
    seq = [sorted(rng.sample(range(N), 6)) for _ in range(40)]
    a = ExpertResidencyPlanner(num_local_experts=N, n_resident=R, n_spill_slots=C)
    b = ExpertResidencyPlanner(num_local_experts=N, n_resident=R, n_spill_slots=C)
    for wave in seq:
        sa, _ = a.resolve(wave)
        sb, _ = b.resolve(wave)
        assert sa == sb


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_resident_slot_count():
    assert resident_slot_count(256, 1.0) == 256
    assert resident_slot_count(256, 0.75) == 192
    assert resident_slot_count(256, 0.5) == 128
    assert resident_slot_count(256, 0.25) == 64
    assert resident_slot_count(256, 0.0) == 1
    assert resident_slot_count(256, 0.001) == 1


def test_default_spill_slots():
    assert default_spill_slots(8) == 8
    assert default_spill_slots(16) == 16
    assert default_spill_slots(2) == 8       # const floor
    assert default_spill_slots(None) == 8
    assert default_spill_slots(0) == 8


def test_spill_cache_clamped_to_spill_set():
    p = ExpertResidencyPlanner(num_local_experts=10, n_resident=6, n_spill_slots=100)
    assert p.n_spill_slots == 4           # clamped to N-R
    assert p.buffer_slots == 10
    slot_of, fetch = p.resolve([6, 7, 8, 9])
    assert fetch == []                    # all 4 spill experts warm-started in place
    assert set(slot_of.keys()) == {6, 7, 8, 9}


def test_fully_resident_passthrough():
    p = ExpertResidencyPlanner(num_local_experts=8, n_resident=8, n_spill_slots=0)
    assert p.fully_resident
    slot_of, fetch = p.resolve([0, 3, 7])
    assert fetch == []
    assert slot_of == {0: 0, 3: 3, 7: 7}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
