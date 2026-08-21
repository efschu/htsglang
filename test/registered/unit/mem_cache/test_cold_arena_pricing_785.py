# SPDX-License-Identifier: Apache-2.0
"""#785: charging the DERIVED arena tail on a cold record, and only there.

The cold seam record carries ``arena_fixed_bytes = 0`` -- not because the tail
is zero but because the only way to learn it used to be a previous boot. On
this rig rank 2's tail is 2215 MiB, so a cold boot armed at 1523 MiB instead of
3226, sized against the missing 1703 MiB and died when the NEXTN draft weights
landed. The two-boot protocol was the workaround.

These tests pin the three properties that make charging it safe: it applies
ONLY to a cold record, it never overwrites a measurement, and a derived zero is
reported as a derived zero rather than silently looking like a missing one.
"""

import dataclasses

from sglang.srt.managers import phase_flip_seam_reserve as seam
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)

MIB = 1048576

#: boot_735_default791b / 735-tail785, rank 2: layout_pp and layout_tp.
RANK2_PP = int(10789.22 * MIB)
RANK2_TP = int(8573.78 * MIB)
RANK2_TAIL_MIB = 2215


class _Runner:
    """The two attributes the method actually reads."""

    def __init__(self, derivation, rank=2):
        self._arena_tail_derivation = derivation
        self._rank = rank

    def _seam_world_rank(self):
        return self._rank

    price = ModelRunnerKVCacheMixin._maybe_price_cold_arena_tail


def _cold():
    return seam.SeamReserve(provenance=seam.PROVENANCE_COLD)


def test_a_cold_record_is_charged_the_derived_tail():
    """THE FIX. 0 becomes 2215 MiB, from this boot's own layouts."""
    priced = _Runner((RANK2_PP, RANK2_TP)).price(_cold())
    assert round(priced.arena_fixed_bytes / MIB) == RANK2_TAIL_MIB


def test_the_charge_reaches_the_arming_floor_and_not_only_the_solve():
    """Why the floor moves: with no per-leg measurement the draw IS the tail.

    This is the term that separates an arming floor of 1523 from one of 3226,
    which is the difference between a boot that flips and one that OOMs when
    the draft weights land.
    """
    cold = _cold()
    assert cold.arming_draw_bytes() == 0
    priced = _Runner((RANK2_PP, RANK2_TP)).price(cold)
    assert round(priced.arming_draw_bytes() / MIB) == RANK2_TAIL_MIB


def test_a_measurement_is_never_overwritten_by_a_derivation():
    """A record is a measurement of THIS rig and outranks a model of it."""
    measured = dataclasses.replace(
        _cold(), provenance="measured", arena_fixed_bytes=999 * MIB
    )
    priced = _Runner((RANK2_PP, RANK2_TP)).price(measured)
    assert priced is measured


def test_a_cold_record_that_already_carries_a_tail_is_left_alone():
    """Adds a number where there was none; never replaces one."""
    partial = dataclasses.replace(_cold(), arena_fixed_bytes=7 * MIB)
    assert _Runner((RANK2_PP, RANK2_TP)).price(partial) is partial


def test_no_derivation_means_no_charge_rather_than_a_guess():
    """The pre-#785 path, unchanged, when the probe declined and said why."""
    assert _Runner(None).price(_cold()).arena_fixed_bytes == 0


def test_a_rank_whose_pp_layout_is_smaller_is_charged_nothing():
    """CAN-FAIL GUARD for the clamp.

    Rank 1 on this rig has layout_pp 8008.96 < layout_tp 8573.78. Charging it
    a negative tail would ADD budget it does not have; charging it a positive
    one would hold memory the arena never reaches.
    """
    rank1 = _Runner((int(8008.96 * MIB), int(8573.78 * MIB)), rank=1)
    assert rank1.price(_cold()).arena_fixed_bytes == 0


def test_the_charge_is_the_subtraction_and_not_the_pp_layout():
    """The failure that would look plausible: charging layout_pp itself.

    That is 10789 MiB instead of 2215 -- a rank that would size its pool to
    almost nothing and still look like it had been priced correctly.
    """
    priced = _Runner((RANK2_PP, RANK2_TP)).price(_cold())
    assert priced.arena_fixed_bytes < RANK2_PP // 4
