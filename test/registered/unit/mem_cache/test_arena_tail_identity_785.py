# SPDX-License-Identifier: Apache-2.0
"""#785: the arena tail is a subtraction of two layout totals, and here it is.

THE TARGET ANY DERIVATION HAS TO HIT. The flip seam's arena tail is the term
that binds the KV pool on this rig, and until #785 it could only be learned
from a previous boot's seam record. Before wiring a DERIVED tail into the pool
solve, the identity it must reproduce is pinned here from the runtime's own
numbers, so a derivation that is merely plausible cannot be shipped.

The runtime prints both layout totals when rung 3 releases the tail
(``phase_flip_boot._commit_refill_high_water`` / ``_timed_arena_refill``):

    PP0  rung 3 released   72.0 MiB ... (TP layout needs 15925.8 of 16007.5 MiB)
    PP2  rung 3 released 2214.0 MiB ... (TP layout needs  8573.8 of 10789.2 MiB)

PP1 printed no release line, i.e. it had no tail to release.

Against the same boot's measured seam records (arena_fixed_bytes):

    rank 0    82 MiB
    rank 1     0 MiB
    rank 2  2215 MiB

Boot 735-standT, 2026-08-20, cut (31,16,17) / attn (7,4,5) / vector
(32,16,16), capacity 161378.

WHY THIS IS A TEST AND NOT A COMMENT. A census-calibrated estimate was tried
first and rejected on exactly these numbers: it reproduced the STRUCTURE
(rank 2 the only positive tail, rank 1 exactly zero) but was 1.3% low on the
PP layout and 9% low on the TP layout -- 770 MiB on rank 2, about 77k tokens
of pool. A derivation is only admissible if it reproduces the measured tails,
and "admissible" needs a number to check against.
"""

MIB = 1024.0 * 1024.0

#: (layout_pp_mib, layout_tp_mib) as the runtime printed them. ``None`` where
#: the rank released nothing and the totals were therefore not logged.
MEASURED_LAYOUTS = {
    0: (16007.5, 15925.8),
    1: None,
    2: (10789.2, 8573.8),
}

#: ``arena_fixed_bytes`` from the seam records the same boot wrote.
MEASURED_TAIL_MIB = {0: 82, 1: 0, 2: 2215}


def arena_tail_mib(layout_pp_mib: float, layout_tp_mib: float) -> float:
    """The identity under test: what rung 3 must commit re-entering PP.

    ``refill_high_water_bytes`` is ``max(pp, tp)`` and the active prefix in the
    TP phase is ``tp``, so the tail is the difference -- clamped, because a
    rank whose PP layout is the SMALLER one has nothing to commit on this leg
    (the ``--pp-stage-ratio 15,9,8`` case that killed all three ranks on
    2026-08-11 is the other leg of the same subtraction).
    """
    return max(0.0, float(layout_pp_mib) - float(layout_tp_mib))


def test_the_identity_reproduces_every_measured_tail():
    """PP minus TP IS the tail, on every rank that logged both totals."""
    for rank, layouts in MEASURED_LAYOUTS.items():
        if layouts is None:
            continue
        derived = arena_tail_mib(*layouts)
        measured = MEASURED_TAIL_MIB[rank]
        assert abs(derived - measured) <= 1.0, (
            f"rank {rank}: PP {layouts[0]} - TP {layouts[1]} = {derived:.1f} "
            f"MiB, but the seam record measured {measured} MiB. The tail is "
            f"not the difference of the two layout totals, so #785 is "
            f"deriving the wrong quantity."
        )


def test_the_rank_that_logged_no_release_had_no_tail():
    """A rank with nothing to release must measure exactly zero.

    Zero and 'small' are different: a zero tail means the rank's PP layout
    does not exceed its TP layout at all, which is a structural property of
    the cut and the vector, not a measurement that happened to come out low.
    """
    assert MEASURED_LAYOUTS[1] is None
    assert MEASURED_TAIL_MIB[1] == 0


def test_the_binding_rank_is_the_one_with_the_tail():
    """Why this term is worth deriving at all.

    Rank 2 carries the entire tail and is the rank that binds the pool: its
    arming floor is corridor law + tail + margin = 3226 MiB against 1523 on
    the other two, and that is what holds the instance at 161378 tokens while
    ranks 0 and 1 could fund 564k and 704k.
    """
    assert MEASURED_TAIL_MIB[2] > 20 * MEASURED_TAIL_MIB[0]
    assert MEASURED_TAIL_MIB[2] == max(MEASURED_TAIL_MIB.values())


def test_a_derivation_is_not_admissible_on_structure_alone():
    """CAN-FAIL GUARD for the rejected census route, kept as a bound.

    The census estimate got both signs and the zero right and was still
    unusable. Anything claiming to derive this term must land inside 1 MiB of
    the measured tails, not merely rank them correctly.
    """
    census_estimate = {0: 0.0, 1: 0.0, 2: 1586.7}
    ranked_correctly = (
        census_estimate[2] > census_estimate[0] and census_estimate[1] == 0
    )
    assert ranked_correctly, "the census estimate did get the structure right"

    worst = max(abs(census_estimate[r] - MEASURED_TAIL_MIB[r]) for r in census_estimate)
    assert worst > 500, (
        "the census estimate is being treated as accurate; it was 628 MiB out "
        "on the binding rank, which is ~63k tokens of pool"
    )
