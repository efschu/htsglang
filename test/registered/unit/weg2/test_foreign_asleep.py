"""Dual-model: the one VRAM number that crosses between the two stacks.

Design addendum 8.4: each stack's budget already charges its OWN sleeping
group (dormant_other). In the dual form every card additionally carries the
OTHER model's two groups, both asleep. Planners stay separate per model
(user 24.09.); only ``foreign_asleep`` per card -- keyed by NVML UUID, the one
identity both stacks share -- is read from the other stack's vram_plan.json.
"""
import pytest

from sglang.srt.weg2.foreign_asleep import (
    ForeignAsleepRefused,
    foreign_asleep_mib,
)

U5090, UA, UB = "GPU-5090", "GPU-3080a", "GPU-3080b"


def plan(p=None, d=None, schema="weg2.vram_plan/1"):
    return {"schema": schema, "boot_id": "nf-x",
            "cards": [{"uuid": U5090}, {"uuid": UA}, {"uuid": UB}],
            "asleep": {"P": p if p is not None else {U5090: 1252, UA: 652, UB: 636},
                       "D": d if d is not None else {U5090: 1938, UA: 716, UB: 718}}}


def test_sum_of_both_foreign_groups_per_card():
    got = foreign_asleep_mib(plan(), own_uuids=[U5090, UA, UB])
    assert got.mib == {U5090: 3190, UA: 1368, UB: 1354}
    assert got.provenance.startswith("RECORD")


def test_served_growth_is_added_per_group():
    got = foreign_asleep_mib(plan(), own_uuids=[U5090, UA, UB],
                             growth_mib={"P": {U5090: 360, UA: 338, UB: 738},
                                         "D": {U5090: 386, UA: 32, UB: 34}})
    assert got.mib == {U5090: 3936, UA: 1738, UB: 2126}


def test_deep_sleep_floor_replaces_the_residue_when_measured():
    got = foreign_asleep_mib(plan(), own_uuids=[U5090, UA, UB],
                             deep_floor_mib={U5090: 600, UA: 300, UB: 300})
    assert got.mib == {U5090: 1200, UA: 600, UB: 600}
    assert "deep" in got.provenance


def test_a_card_the_other_plan_does_not_know_is_refused():
    with pytest.raises(ForeignAsleepRefused, match="GPU-new"):
        foreign_asleep_mib(plan(), own_uuids=[U5090, UA, "GPU-new"])


def test_a_group_with_a_missing_card_is_refused_not_zeroed():
    with pytest.raises(ForeignAsleepRefused, match="D"):
        foreign_asleep_mib(plan(d={U5090: 1938, UA: 716}), own_uuids=[U5090, UA, UB])


def test_foreign_schema_is_refused():
    with pytest.raises(ForeignAsleepRefused, match="schema"):
        foreign_asleep_mib(plan(schema="weg2.vram_plan/0"), own_uuids=[U5090])


def test_no_foreign_plan_means_no_term():
    got = foreign_asleep_mib(None, own_uuids=[U5090, UA, UB])
    assert got.mib == {U5090: 0, UA: 0, UB: 0} and got.provenance == "NONE single-model"
