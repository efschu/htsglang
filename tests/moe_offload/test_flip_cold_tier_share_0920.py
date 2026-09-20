# SPDX-License-Identifier: Apache-2.0
"""Slice 3: switching the shared cold tier on for a TWO-GROUP flip.

Hermetic: no /dev/shm writes except into pytest's tmp_path, no torch, no
device. The one non-pure test is the two-opener check on a real tmpfs-shaped
file, which uses tmp_path and never touches /dev/shm.
"""
from __future__ import annotations

import os

import pytest

from sglang.srt.flip_cold_tier_share import (
    COLD_TIER_INSTANCE_ENV,
    COLD_TIER_SWITCH_ENV,
    ColdTierHolder,
    ShardMap,
    Weg2FlipColdTierSplit,
    assert_one_instance,
    build_cold_tier_group_env,
    check_shard_alignment,
    may_unlink,
)


# --------------------------------------------------------------------------
# Prior art: the names this module mirrors must not drift
# --------------------------------------------------------------------------
def test_the_instance_env_name_matches_cold_tier_fetch():
    """This module keeps the name as a literal so it stays desk-pure. That
    only works while the literal is right."""
    from sglang.srt.layers.moe import cold_tier_fetch

    assert cold_tier_fetch.COLD_TIER_INSTANCE_ENV == COLD_TIER_INSTANCE_ENV


def test_the_switch_name_matches_the_environ_inventory():
    from sglang.srt.environ import envs

    assert hasattr(envs, COLD_TIER_SWITCH_ENV)


def test_the_shared_pinned_path_is_mmap_plus_register_not_pin_memory():
    """Risk R4: ``page-locked exactly`` holds only for the
    mmap+cudaHostRegister path. If this ever becomes pin_memory, every host
    number in the design is too small (Memory PINNED-EXAKT-DREI-FALLEN)."""
    import inspect

    from sglang.srt.layers.moe import shared_pinned

    src = inspect.getsource(shared_pinned.shared_pinned_empty)
    assert "cudaHostRegister" in src
    assert "pin_memory" not in src


# --------------------------------------------------------------------------
# Gap 1: one instance id in BOTH groups
# --------------------------------------------------------------------------
def test_both_groups_get_the_same_instance():
    frags = build_cold_tier_group_env("deadbeefcafe0001")
    assert set(frags) == {"P", "D"}
    for g in ("P", "D"):
        assert frags[g][COLD_TIER_SWITCH_ENV] == "1"
        assert frags[g][COLD_TIER_INSTANCE_ENV] == "deadbeefcafe0001"
    assert assert_one_instance(frags) == "deadbeefcafe0001"


def test_an_empty_instance_is_refused_because_each_group_would_mint_its_own():
    """publish_cold_tier_instance() mints uuid4 per SERVER PROCESS
    (cold_tier_fetch.py:149-165, called engine.py:683). Two groups = two ids
    = two pools, while both report shared=true."""
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        build_cold_tier_group_env("")
    msg = str(exc.value)
    assert "W117 Weg2FlipColdTierSplit" in msg
    assert "engine.py:683" in msg
    assert "uuid4" in msg


def test_two_different_instances_are_refused_with_the_saving_named():
    envs = {
        "P": {COLD_TIER_SWITCH_ENV: "1", COLD_TIER_INSTANCE_ENV: "aaaa"},
        "D": {COLD_TIER_SWITCH_ENV: "1", COLD_TIER_INSTANCE_ENV: "bbbb"},
    }
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        assert_one_instance(envs)
    msg = str(exc.value)
    assert "two pools, not one" in msg
    assert "29.77" in msg


def test_the_switch_on_in_one_group_only_is_refused():
    envs = {
        "P": {COLD_TIER_SWITCH_ENV: "1", COLD_TIER_INSTANCE_ENV: "aaaa"},
        "D": {},
    }
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        assert_one_instance(envs)
    assert "half-shared" in str(exc.value).lower()


def test_the_switch_on_with_no_instance_is_refused():
    envs = {"P": {COLD_TIER_SWITCH_ENV: "1"}, "D": {COLD_TIER_SWITCH_ENV: "1"}}
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        assert_one_instance(envs)
    assert "mint its own id" in str(exc.value)


def test_deliberately_private_pools_are_stated_not_absent():
    """A flip that runs private pools on purpose must say so, so the slice-2
    ledger prints shared=false on all six ranks."""
    frags = build_cold_tier_group_env("", enabled=False)
    assert frags["P"][COLD_TIER_SWITCH_ENV] == "0"
    assert COLD_TIER_INSTANCE_ENV not in frags["D"]
    assert assert_one_instance(frags) == ""


def test_the_launcher_may_mint_for_more_than_two_groups():
    frags = build_cold_tier_group_env("f00d", groups=("P", "D", "X"))
    assert assert_one_instance(frags) == "f00d"


# --------------------------------------------------------------------------
# Gap 2: the two layouts cut the expert set differently
# --------------------------------------------------------------------------
def _by_layer(rank, layers, experts=4):
    return ShardMap.of("PP3", rank, [(l, e) for l in layers for e in range(experts)])


def _by_expert(rank, experts, layers=6):
    return ShardMap.of("FormA", rank, [(l, e) for l in range(layers) for e in experts])


def test_identical_shards_align():
    left = {0: ShardMap.of("PP3", 0, [(0, 1), (0, 2)])}
    right = {0: ShardMap.of("FormA", 0, [(0, 2), (0, 1)])}
    check_shard_alignment(left, right)  # no raise


def test_the_real_cut_is_orthogonal_and_is_refused():
    """PP3 cuts by LAYER, Form A by EXPERT INDEX. Rank 0 of the two does NOT
    hold the same rows, so max-per-slot is not the arithmetic -- the union
    is, and nobody measured it."""
    left = {
        0: _by_layer(0, [0, 1, 2]),
        1: _by_layer(1, [3, 4]),
        2: _by_layer(2, [5]),
    }
    right = {
        0: _by_expert(0, [0]),
        1: _by_expert(1, [1, 2]),
        2: _by_expert(2, [3]),
    }
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        check_shard_alignment(left, right)
    msg = str(exc.value)
    assert "DIFFERENT cold rows" in msg
    assert "40.97" in msg
    assert "70.74" in msg
    assert "LOWER bound" in msg
    # the refusal must itemise the slot, not just say "no"
    assert "rank 0" in msg and "union" in msg


def test_a_rank_slot_with_one_reader_is_a_sum_term_not_a_max_term():
    left = {0: _by_layer(0, [0]), 1: _by_layer(1, [1])}
    right = {0: _by_layer(0, [0])}
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        check_shard_alignment(left, right)
    assert "SUM term" in str(exc.value)


def test_empty_shards_are_refused():
    with pytest.raises(Weg2FlipColdTierSplit):
        check_shard_alignment({}, {})


# --------------------------------------------------------------------------
# Teardown by holder, never by pattern
# --------------------------------------------------------------------------
def test_the_holder_may_unlink_its_own_epoch():
    h = ColdTierHolder("epoch1", 4242)
    assert may_unlink(h, caller_pid=4242, caller_instance="epoch1") is True


def test_a_foreign_epoch_is_refused():
    h = ColdTierHolder("epoch1", 4242)
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        may_unlink(h, caller_pid=4242, caller_instance="epoch2")
    assert "by ITS holder or by nobody" in str(exc.value)


def test_a_peer_may_not_tear_down_for_the_holder():
    h = ColdTierHolder("epoch1", 4242)
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        may_unlink(h, caller_pid=9999, caller_instance="epoch1")
    assert "not the holder" in str(exc.value)


def test_a_live_peer_blocks_the_unlink():
    h = ColdTierHolder("epoch1", 4242)
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        may_unlink(h, 4242, "epoch1", live_peer_pids=[4242, 5555])
    assert "5555" in str(exc.value)


def test_an_unlink_without_an_epoch_is_a_pattern_sweep():
    """Memory SHM-RESIDUE-NUR-PER-HALTER, stated in the refusal."""
    with pytest.raises(Weg2FlipColdTierSplit) as exc:
        may_unlink(ColdTierHolder("", 1), 1, "")
    assert "SHM-RESIDUE-NUR-PER-HALTER" in str(exc.value)


# --------------------------------------------------------------------------
# The mechanics the sharing rests on: a second opener attaches, does not size
# --------------------------------------------------------------------------
def test_a_second_opener_attaches_to_the_same_bytes_and_does_not_resize(tmp_path):
    """``open_shared_file`` is the primitive both groups go through. The
    second opener must see the first opener's size and bytes."""
    from sglang.srt.layers.moe.shared_pinned import open_shared_file

    path = str(tmp_path / "store" / "seg.bin")
    fd1, created1 = open_shared_file(path, 4096)
    assert created1 is True
    os.pwrite(fd1, b"\x5a" * 16, 0)
    os.close(fd1)

    fd2, created2 = open_shared_file(path, 4096)
    assert created2 is False
    assert os.fstat(fd2).st_size == 4096
    assert os.pread(fd2, 16, 0) == b"\x5a" * 16
    os.close(fd2)


def test_a_size_disagreement_between_layouts_is_refused_loudly(tmp_path):
    """Two layouts disagreeing on a store's shape must never silently alias."""
    from sglang.srt.layers.moe.shared_pinned import open_shared_file

    path = str(tmp_path / "seg.bin")
    fd, _ = open_shared_file(path, 4096)
    os.close(fd)
    with pytest.raises(ValueError) as exc:
        open_shared_file(path, 8192)
    assert "4096" in str(exc.value) and "8192" in str(exc.value)
