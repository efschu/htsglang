# SPDX-License-Identifier: Apache-2.0
"""#782: two boots of the same command line must report the same capacity.

Identical argv on an identical commit (87c050ef83, cut (31,16,17) / attn
(7,4,5)) produced three different pool sizes on the binding rank:

    bootA   13:38  seam record COLD          -> 430000  (pin; profiled 658150)
    bootA2  13:45  record written 13:42:40   -> 146390
    standS  13:50  record written 13:49:28   -> 161378

The state carried between them is
``~/.cache/sglang/kv_budget-<digest>-seam-rank<N>.json``. ``max_total_tokens``
is not part of the digest, so every one of those boots read and overwrote the
same three files.

Two defects, and this module pins the second one:

  1. the COLD/warm cliff -- a first boot prices the arena tail at 0, so rank 2's
     arming floor is 1523 MiB instead of 3226 MiB and its pool is 4.4x too big;
  2. WARM/WARM DRIFT -- ``floor_allowed_tokens`` solves the id space from the
     free column the previous boot measured, making capacity a first-order
     recurrence in itself:

         T(n+1) = T(n) + (free_measured(n) - floor) / cell

     It converges, but onto an asymptote rather than a value: the last observed
     step was a 6 MiB residual on a 4808 MiB baseline, i.e. +614 tokens, and the
     one before it +14988. A sequence that converges still never repeats, so no
     two boots agree.

The fix is a directional noise latch: a GROWTH smaller than
``POOL_LATCH_TOLERANCE_BYTES`` is not taken, which turns the asymptotic fixed
point into an exact one. A SHRINK is always taken however small, because a
lower re-solve means the rank can no longer hold its arming floor at the
current size -- refusing a small growth cannot breach a floor, refusing a small
shrink can.

Numbers below are the measured rig values, not constructed ones.
"""

import pytest

from sglang.srt.managers import phase_flip_seam_reserve as sr

MIB = 1 << 20

# Binding rank (PP2, an RTX 3080) at cut (31,16,17) / attn (7,4,5).
CELL = 10240  # bytes per token, from "KV pool sizing: ... cell_size=10240"
FLOOR = 3226 * MIB  # "ARMING FLOOR 3226 MiB", = corridor law + arena tail + margin

# What each boot READ, taken from the record the previous boot wrote.
BOOTA2_READ_ID = 430000  # written by the cold bootA
BOOTA2_READ_FREE = 456 * MIB  # "456 MiB free at an id space of 430000"
BOOTA2_SOLVED = 146390

STANDS_READ_ID = 146390
STANDS_READ_FREE = 3372 * MIB  # "3372 MiB free at an id space of 146390"
STANDS_SOLVED = 161378

# What standS itself wrote at 13:54:45, i.e. what the NEXT boot reads.
NEXT_READ_ID = 161378
NEXT_READ_FREE = 3232 * MIB


def _reserve(id_space: int, free_at_measure: int) -> sr.SeamReserve:
    """A warm record with only the fields the floor solve reads."""
    return sr.SeamReserve(
        fixed_bytes=139 * MIB,
        arena_fixed_bytes=2215 * MIB,
        per_row_bytes=2079.4,
        id_space=id_space,
        free_at_measure_bytes=free_at_measure,
        provenance=sr.PROVENANCE_STORED,
    )


def _solve(id_space: int, free_at_measure: int) -> int:
    return sr.floor_allowed_tokens(CELL, _reserve(id_space, free_at_measure), FLOOR)


# ---------------------------------------------------------------------------
# The defect, reproduced from the rig's own numbers.
# ---------------------------------------------------------------------------


def test_recurrence_reproduces_the_observed_boot_sequence():
    """The solve is the thing that produced 146390 and then 161378.

    Establishes that this module is pinning the real mechanism and not a model
    of it: both transitions fall out of the recorded free column alone.
    """
    assert _solve(BOOTA2_READ_ID, BOOTA2_READ_FREE) == pytest.approx(
        BOOTA2_SOLVED, abs=64
    )
    assert _solve(STANDS_READ_ID, STANDS_READ_FREE) == pytest.approx(
        STANDS_SOLVED, abs=64
    )


def test_without_a_latch_the_sequence_never_repeats_a_value():
    """CAN-FAIL PROOF for the latch: the raw recurrence keeps moving.

    Computed the way the pre-#782 code did -- no latch -- so a regression that
    removes the latch does not quietly make this file pass anyway.
    """
    raw = STANDS_READ_ID + (STANDS_READ_FREE - FLOOR) // CELL
    assert raw != STANDS_READ_ID, "the 146390 -> 161378 step must be real"

    raw_next = NEXT_READ_ID + (NEXT_READ_FREE - FLOOR) // CELL
    assert raw_next != NEXT_READ_ID, (
        "the residual is small but nonzero, which is exactly why capacity "
        "differed between boots"
    )
    assert 0 < (raw_next - NEXT_READ_ID) * CELL <= sr.POOL_LATCH_TOLERANCE_BYTES


# ---------------------------------------------------------------------------
# The fix.
# ---------------------------------------------------------------------------


def test_sub_tolerance_growth_does_not_move_the_id_space():
    """The 6 MiB residual standS left behind must not buy 614 tokens."""
    assert _solve(NEXT_READ_ID, NEXT_READ_FREE) == NEXT_READ_ID


def test_two_consecutive_boots_agree_exactly():
    """THE ACCEPTANCE PROPERTY: same record shape in, same capacity out.

    Boot N latches, writes a record at the same id space, and boot N+1 reads it
    and latches again. Equality is exact, not approximate -- the whole point is
    a capacity that can be reported twice.
    """
    first = _solve(NEXT_READ_ID, NEXT_READ_FREE)
    # The next boot runs at `first` and re-measures a free column within the
    # noise band of the one before it.
    second = _solve(first, NEXT_READ_FREE + 3 * MIB)
    third = _solve(second, NEXT_READ_FREE - 2 * MIB)
    assert first == second == third == NEXT_READ_ID


def test_a_real_correction_is_still_taken():
    """The latch is a noise floor, not a freeze.

    The 146390 -> 161378 step is 146 MiB and must survive; a latch wide enough
    to swallow it would have pinned the pool at a value measured under the
    previous boot's geometry.
    """
    assert (STANDS_SOLVED - STANDS_READ_ID) * CELL > sr.POOL_LATCH_TOLERANCE_BYTES
    assert _solve(STANDS_READ_ID, STANDS_READ_FREE) == pytest.approx(
        STANDS_SOLVED, abs=64
    )


def test_a_shrink_is_never_latched_however_small():
    """DIRECTIONAL SAFETY. A lower re-solve is a floor violation, not noise.

    One token below the current size still means the rank cannot hold its
    arming floor there, so it is taken even though the same magnitude of growth
    would have been refused.
    """
    shrink_free = FLOOR - 1 * MIB
    solved = _solve(NEXT_READ_ID, shrink_free)
    assert solved < NEXT_READ_ID
    assert (NEXT_READ_ID - solved) * CELL <= sr.POOL_LATCH_TOLERANCE_BYTES, (
        "this is deliberately a SUB-tolerance move, so the test fails if the "
        "latch is ever made symmetric"
    )


def test_cold_record_is_unchanged():
    """A first boot has no free column and must solve nothing.

    ``None`` and ``0`` stay distinct: the caller falls back to the subtrahend,
    which is the pre-#678 arithmetic exactly.
    """
    cold = sr.SeamReserve(provenance=sr.PROVENANCE_COLD)
    assert sr.floor_allowed_tokens(CELL, cold, FLOOR) is None
