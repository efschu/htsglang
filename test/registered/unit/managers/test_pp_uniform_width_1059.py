"""#1059 ARM A': the four danger directions of this seam, as executable assertions.

RED-FIRST FIXTURE IS BOOT 27 VERBATIM
(`boot_855_1057acc_0840f82601_0831_154751.log`, 16:01:42Z):

    [PP0] #969N ADMIT slot=0 fwd_ct=77 bs=2 extend=4096 rids=[0584c835,0cf766fa]
    [PP1] #969N ADMIT slot=0 fwd_ct=77 bs=2 extend=2596 rids=[0584c835,0cf766fa]

Same slot, same fwd_ct, same rids, different width -- because each rank seeded
its prefix from its OWN post-cutover HiCache hit (PP1 host_hit=15932/13376,
PP0 prefix 4094). Exactly one divergent (slot, fwd_ct) pair existed in the whole
boot, and it killed the group via #631.

THE FOUR DANGER DIRECTIONS OF THIS SEAM, each with a named prior death:

  1. REFUSAL WITHOUT A WAY ONWARD -- boot 15 (#1048, 1448 refusals on one rid
     until the ring wedged) and #995 v1 (175 refusals, dead window).
  2. APPLY SKIPPED -- the told value arrives and nothing consumes it; this is
     the present state of the tree (`_pp_admission_incoming_effective` has 4
     writers, all None/{}), i.e. boot 27 itself.
  3. PER-RANK RE-DERIVATION of the uniform decision -- the decision taken more
     than once is not a decision; it is the 26th divergent input.
  4. SHORTFALL CHANGES THE GROUP-VISIBLE GEOMETRY -- the substitution's own
     residual risk: recomputing missing prefix bytes is EXECUTION and must not
     move the batch shape.
"""

import pytest

from sglang.srt.managers.pp_uniform_width import uniform_pass_geometry

# Boot 27, verbatim.
TOLD_PREFIX = 4094
TOLD_EXTEND = 4096
PP1_LOCAL_PREFIX = 13376  # PP1's own host hit -- MORE cache than PP0
PP2_LOCAL_PREFIX = 0  # a rank whose tier holds nothing for this rid


def test_boot27_divergence_cannot_recur_across_ranks():
    """DIRECTION 3: one decision, obeyed identically, whatever the local tier.

    The boot-27 fixture: three ranks with three different local prefixes must
    present ONE geometry. A mutant that re-derives per rank fails here.
    """
    geoms = [
        uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, local)
        for local in (TOLD_PREFIX, PP1_LOCAL_PREFIX, PP2_LOCAL_PREFIX)
    ]
    widths = {(g.prefix, g.extend) for g in geoms}
    assert widths == {(4094, 4096)}, (
        f"ranks presented {len(widths)} different geometries {widths}; "
        "boot 27 died of exactly this (4096 vs 2596)"
    )


def test_the_geometry_is_a_function_of_told_alone():
    """DIRECTION 4: the local tier may reach `shortfall` and NOTHING else.

    Sweeps the local prefix across its whole meaningful range and asserts the
    group-visible numbers never move. This is the assertion that kills a mutant
    which derives the geometry from the local tier.
    """
    for local in range(0, 20000, 613):
        g = uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, local)
        assert g.prefix == TOLD_PREFIX, f"local={local} moved the prefix"
        assert g.extend == TOLD_EXTEND, f"local={local} moved the extend"


def test_the_shortfall_rank_still_runs_the_told_shape_while_recomputing():
    """DIRECTION 4, the boot-27 case named in the build order.

    Told prefix 4094 while this rank's tier holds 2596: it recomputes the gap
    (EXECUTION, rank-local) and presents the told shape (DECISION, uniform).
    """
    g = uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, 2596)
    assert (g.prefix, g.extend) == (TOLD_PREFIX, TOLD_EXTEND)
    assert g.shortfall == 4094 - 2596, "the recompute cost must be counted"
    assert g.adopted is True


def test_surplus_cache_is_execution_too_and_never_widens_the_batch():
    """A rank with MORE cache than told does not get a different batch either."""
    g = uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, PP1_LOCAL_PREFIX)
    assert (g.prefix, g.extend) == (TOLD_PREFIX, TOLD_EXTEND)
    assert g.shortfall == 0, "surplus is not a shortfall"


@pytest.mark.parametrize("local", [0, 1, 2596, 4094, 13376, 999999])
def test_there_is_no_refusal_path_for_any_input(local):
    """DIRECTION 1: the boot-15 shape must be unrepresentable here.

    Every input maps to a runnable geometry. No exception, no None geometry, no
    sentinel a caller could mistake for 'refuse and wait'. If a future edit adds
    a refusal branch, it has to break one of these.
    """
    g = uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, local)
    assert g is not None
    assert isinstance(g.prefix, int) and g.prefix >= 0
    assert g.extend == TOLD_EXTEND
    assert g.shortfall >= 0


def test_absent_told_is_bit_for_bit_the_old_behaviour():
    """DIRECTION 2's honest half: no fact means NO adoption, never a substitute.

    An older sender, a stand-in, or a pass PP0 did not name must behave exactly
    as before -- otherwise this change is not safe to land dark.
    """
    g = uniform_pass_geometry(None, None, 7777)
    assert g.prefix == 7777, "absent told must leave the local prefix alone"
    assert g.adopted is False
    assert g.shortfall == 0


def test_apply_skipped_is_detectable_by_the_adopted_flag():
    """DIRECTION 2: a told value that arrives and is ignored must be visible.

    `adopted` is the receipt. A mutant that drops the told value on the floor
    while a fact was present flips this to False and fails.
    """
    assert uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, 0).adopted is True
    assert uniform_pass_geometry(None, None, 0).adopted is False
