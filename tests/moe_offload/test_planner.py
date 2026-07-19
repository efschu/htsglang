# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MoE expert-offload planner + wave logic.

No CUDA required: exercises ``ExpertResidencyPlanner`` (LRU residency) and
``plan_token_waves`` (token-partition wave splitting that makes >n_slots
forwards serviceable without crashing, the fix for the prefill KeyError).

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
)


# --------------------------------------------------------------------------- #
# plan_token_waves
# --------------------------------------------------------------------------- #
def _all_rows(waves):
    return [t for w in waves for t in w]


def test_single_wave_when_union_fits():
    # 4 tokens, top-2, union = {0,1,2,3} == n_slots -> exactly one wave.
    experts = [[0, 1], [1, 2], [2, 3], [0, 3]]
    waves = plan_token_waves(experts, n_slots=4)
    assert len(waves) == 1
    assert waves[0] == [0, 1, 2, 3]


def test_partition_is_complete_and_ordered():
    experts = [[e, e + 1] for e in range(0, 40, 2)]  # 20 tokens, disjoint-ish
    waves = plan_token_waves(experts, n_slots=6)
    rows = _all_rows(waves)
    # every token appears exactly once, in original order.
    assert rows == list(range(len(experts)))
    assert sorted(rows) == rows


def test_each_wave_union_within_n_slots():
    rng = __import__("random").Random(1234)
    E, top_k, T, n_slots = 256, 8, 300, 64
    experts = [rng.sample(range(E), top_k) for _ in range(T)]
    waves = plan_token_waves(experts, n_slots=n_slots)
    assert len(waves) > 1  # overflow really happened (this is the prefill case)
    for w in waves:
        union = set()
        for t in w:
            union |= set(experts[t])
        assert len(union) <= n_slots, f"wave union {len(union)} > {n_slots}"
    assert _all_rows(waves) == list(range(T))


def test_overflow_needs_more_experts_than_n_slots_does_not_crash():
    # n_slots + K unique experts across the batch: must wave-split, not raise.
    n_slots, K = 16, 10
    experts = [[e] for e in range(n_slots + K)]  # each token 1 distinct expert
    waves = plan_token_waves(experts, n_slots=n_slots)
    assert len(waves) >= 2
    assert _all_rows(waves) == list(range(n_slots + K))


def test_padding_minus_one_ignored():
    experts = [[0, -1], [1, -1], [-1, -1]]
    waves = plan_token_waves(experts, n_slots=2)
    # unique real experts {0,1} <= 2 -> single wave; -1 padding never counted.
    assert len(waves) == 1
    assert waves[0] == [0, 1, 2]


def test_single_token_exceeds_n_slots_raises():
    # A token routing to more unique experts than n_slots is unservable.
    experts = [[0, 1, 2, 3]]
    with pytest.raises(ValueError):
        plan_token_waves(experts, n_slots=3)


# --------------------------------------------------------------------------- #
# ExpertResidencyPlanner
# --------------------------------------------------------------------------- #
def test_warm_start_residency():
    p = ExpertResidencyPlanner(num_local_experts=256, n_slots=64)
    # experts [0,64) pre-resident at slot == expert id.
    for e in range(64):
        assert p.slot_of_expert[e] == e


def test_resolve_hit_no_fetch():
    # LRU path (deterministic=False): experts already resident are not re-fetched.
    p = ExpertResidencyPlanner(num_local_experts=256, n_slots=64, deterministic=False)
    slot_of, fetch = p.resolve([0, 1, 63])
    assert fetch == []
    assert slot_of == {0: 0, 1: 1, 63: 63}
    assert p.stats.fetches == 0


def test_resolve_deterministic_sorted_slots():
    # Deterministic path (default): needed experts -> slots [0,k) in sorted
    # order, history-independent, all (re)fetched. Two calls with the same
    # needed set give the identical layout regardless of intervening calls.
    p = ExpertResidencyPlanner(num_local_experts=256, n_slots=64)
    slot_a, fetch_a = p.resolve([63, 1, 0])
    assert slot_a == {0: 0, 1: 1, 63: 2}
    assert fetch_a == [(0, 0), (1, 1), (63, 2)]
    p.resolve([200, 5, 9])  # intervening call changes nothing for the next
    slot_b, fetch_b = p.resolve([63, 1, 0])
    assert slot_b == slot_a and fetch_b == fetch_a  # byte-identical layout


def test_resolve_miss_fetches_and_evicts_lru():
    p = ExpertResidencyPlanner(
        num_local_experts=256, n_slots=4, deterministic=False
    )
    # resident {0,1,2,3}. Touch 0 so it's MRU, then miss 100 -> evict LRU (1).
    p.resolve([0])
    slot_of, fetch = p.resolve([100])
    assert len(fetch) == 1
    (exp, slot) = fetch[0]
    assert exp == 100
    assert 100 in slot_of
    # expert 1 (least recently used, and not 0) should have been evicted.
    assert 1 not in p.slot_of_expert
    assert 0 in p.slot_of_expert


def test_resolve_never_evicts_needed_this_wave():
    # Fill n_slots with a full wave of misses: every resident expert is needed,
    # none may be evicted mid-wave; distinct slots for all.
    p = ExpertResidencyPlanner(num_local_experts=256, n_slots=8)
    needed = list(range(200, 208))  # 8 misses == n_slots
    slot_of, fetch = p.resolve(needed)
    assert set(slot_of.keys()) == set(needed)
    assert len(set(slot_of.values())) == 8  # all distinct slots
    assert len(fetch) == 8


def test_resolve_rejects_overflow_wave():
    # resolve must be given <= n_slots unique experts; more is a caller bug.
    p = ExpertResidencyPlanner(num_local_experts=256, n_slots=4)
    with pytest.raises(RuntimeError):
        p.resolve([10, 11, 12, 13, 14])


def test_wave_by_wave_serves_full_overflow_batch():
    """End-to-end planner: partition an overflow batch into waves and resolve
    each; every needed expert lands in a valid, distinct-per-wave slot and the
    LRU never drops an expert needed within its own wave."""
    E, top_k, T, n_slots = 256, 8, 128, 32
    rng = __import__("random").Random(7)
    experts = [rng.sample(range(E), top_k) for _ in range(T)]
    p = ExpertResidencyPlanner(num_local_experts=E, n_slots=n_slots)
    waves = plan_token_waves(experts, n_slots=n_slots)
    assert len(waves) > 1
    for w in waves:
        needed = sorted({e for t in w for e in experts[t]})
        slot_of, _ = p.resolve(needed)
        # every needed expert resolved to a real slot in-range and distinct.
        assert set(slot_of.keys()) == set(needed)
        slots = list(slot_of.values())
        assert len(set(slots)) == len(slots)
        assert all(0 <= s < n_slots for s in slots)


def test_resident_slot_count():
    assert resident_slot_count(256, 1.0) == 256
    assert resident_slot_count(256, 0.25) == 64
    assert resident_slot_count(256, 0.0) == 1  # clamped to >= 1
    assert resident_slot_count(256, 0.001) == 1


def test_fully_resident_passthrough():
    p = ExpertResidencyPlanner(num_local_experts=8, n_slots=8)
    assert p.fully_resident
    slot_of, fetch = p.resolve([0, 3, 7])
    assert fetch == []
    assert slot_of == {0: 0, 3: 3, 7: 7}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
