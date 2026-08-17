# SPDX-License-Identifier: Apache-2.0
"""#723: the frontier must be Pareto-complete in POOL, not only in speed.

The defect: `solve_prefill_frontier` enumerated, for each lead depth, ONLY the
tail split that minimises pipelined time. At n0=30 that is [30,16,18], so
[30,18,16] was never generated -- yet [30,16,18] holds 343,951 tokens while
[30,18,16] holds 436,275, the incumbent's own pool. A cut that STRICTLY
DOMINATES the incumbent (equal pool, more speed) was invisible to the solver
that exists to find it, and reached the #702 decision table only because it was
priced by hand.

These tests use the same calibrated fixture as test_prefill_frontier_702.py.
"""

import pytest

from test_prefill_frontier_702 import (  # noqa: E402  (sibling fixture module)
    CELL,
    INCUMBENT,
    MS_PER_LAYER,
    OBSERVED_POOL,
    _attn_for,
    _avail_for,
    _solve,
)


def _pool_of(counts):
    attn = _attn_for(counts)
    avail = _avail_for(counts, attn)
    return min(b / (a * CELL) for b, a in zip(avail, attn))


def _ms_of(counts):
    return max(MS_PER_LAYER[i] * counts[i] for i in range(3))


def _by_counts(f):
    return {p.counts: p for p in f.points}


# --- the red-first case -----------------------------------------------------


def test_THE_POOL_DOMINANT_CUT_IS_GENERATED():
    """RED-FIRST. [30,18,16] must appear on the generated candidate list.

    It is not a marginal addition: it holds the incumbent's exact pool
    (436,275, because it leaves the binder PP2 untouched) and is 1.11x faster.
    A solver that cannot produce it cannot present the user a complete
    decision set.
    """
    f = _solve()
    assert (30, 18, 16) in _by_counts(f), (
        "[30,18,16] is absent: the enumeration is still taking only the "
        "speed-argmin tail per lead depth"
    )


def test_it_is_pool_dominant_over_its_own_depths_speed_optimum():
    """Why it must be generated: at the SAME lead depth it trades 6% of speed
    for 27% more pool, and the old enumeration silently made that choice."""
    f = _solve()
    pts = _by_counts(f)
    assert (30, 16, 18) in pts, "the speed optimum at n0=30 must still be there"
    roomy, fast = pts[(30, 18, 16)], pts[(30, 16, 18)]
    assert roomy.coupled_pool_tokens > fast.coupled_pool_tokens
    assert roomy.compute_speedup < fast.compute_speedup
    assert roomy.coupled_pool_tokens == pytest.approx(436_275, rel=1e-3)
    assert roomy.coupled_pool_tokens / fast.coupled_pool_tokens > 1.25


def test_it_strictly_dominates_the_incumbent():
    """The property that makes its absence a defect rather than a gap."""
    f = _solve()
    pts = _by_counts(f)
    inc, roomy = pts[INCUMBENT], pts[(30, 18, 16)]
    assert roomy.coupled_pool_tokens >= inc.coupled_pool_tokens - 1.0
    assert roomy.compute_speedup > inc.compute_speedup
    assert roomy.compute_speedup == pytest.approx(1.1111, abs=1e-3)


# --- what must NOT move -----------------------------------------------------

# The speed-optimal tail at each lead depth, as generated BEFORE #723, with its
# compute speedup. These are the rows the #702 block was written from; if any
# of them moves, the fix changed an answer instead of adding one.
_PRE_723_SPEED_OPTIMA = {
    (28, 20, 16): 1.0000,
    (28, 17, 19): 1.1199,
    (29, 17, 18): 1.1765,
    (30, 16, 18): 1.1821,
    (31, 16, 17): 1.2500,
    (32, 15, 17): 1.2517,
    (33, 15, 16): 1.3299,
    (34, 15, 15): 1.3333,
    (35, 14, 15): 1.4186,
    (36, 14, 14): 1.4286,
    (37, 13, 14): 1.5199,
    (38, 13, 13): 1.5385,
    (39, 12, 13): 1.6368,
    (40, 12, 12): 1.6667,
    (41, 11, 12): 1.7732,
    (42, 11, 11): 1.8182,
    (43, 10, 11): 1.9344,
    (44, 10, 10): 2.0000,
    (45, 9, 10): 1.9578,
    (46, 7, 11): 1.9152,
    (47, 6, 11): 1.8745,
    (48, 5, 11): 1.8354,
    (49, 4, 11): 1.7980,
    (50, 2, 12): 1.7620,
    (51, 1, 12): 1.7274,
}


def test_every_pre_723_speed_optimum_SURVIVES_UNCHANGED():
    """The fix must ADD rows, never move one."""
    f = _solve()
    pts = _by_counts(f)
    for counts, speedup in _PRE_723_SPEED_OPTIMA.items():
        assert counts in pts, f"{counts} disappeared from the frontier"
        assert pts[counts].compute_speedup == pytest.approx(speedup, abs=5e-4)


def test_the_headline_picks_are_untouched():
    """[42,11,11] 1.660x without the lever, [44,10,10] 2.000x with it."""
    f = _solve()
    assert f.best_without_pipelining().counts == (42, 11, 11)
    assert f.best_without_pipelining().net_no_pipelining == pytest.approx(
        1.660, abs=0.03
    )
    assert f.best_with_pipelining().counts == (44, 10, 10)
    assert f.best_with_pipelining().net_pipelined == pytest.approx(2.000, abs=0.02)


def test_the_incumbent_is_still_first_and_the_deepest_is_still_last():
    f = _solve()
    assert f.points[0].counts == INCUMBENT
    assert f.points[-1].counts[0] == max(p.counts[0] for p in f.points)


# --- the Pareto property itself ---------------------------------------------


def test_no_generated_cut_is_dominated_on_BOTH_axes_at_its_depth():
    """The definition of the set being kept. A cut that is both slower AND
    smaller than a sibling at the same lead depth is not a trade, it is a
    worse cut, and generating it would pad the user's decision set."""
    f = _solve()
    by_depth: dict[int, list] = {}
    for p in f.points:
        by_depth.setdefault(p.counts[0], []).append(p)
    for depth, group in by_depth.items():
        for a in group:
            for b in group:
                if a.counts == b.counts:
                    continue
                dominated = (
                    b.compute_speedup >= a.compute_speedup
                    and b.coupled_pool_tokens >= a.coupled_pool_tokens
                    and (
                        b.compute_speedup > a.compute_speedup
                        or b.coupled_pool_tokens > a.coupled_pool_tokens
                    )
                )
                assert not dominated, (
                    f"at depth {depth}, {a.counts} is dominated by {b.counts} "
                    "on both axes and should not have been generated"
                )


def test_needs_pipelining_is_a_property_of_the_DEPTH_not_the_tail():
    """Siblings at one lead depth share the flag.

    The flag says 'this DEPTH cannot be cashed in until the overhead is
    hidden'. Computing it per tail would let a roomier, slower sibling be
    flagged as needing a lever it does not need -- it is not reaching for
    depth, it is trading speed for pool.
    """
    f = _solve()
    by_depth: dict[int, set] = {}
    for p in f.points:
        by_depth.setdefault(p.counts[0], set()).add(p.needs_pipelining)
    for depth, flags in by_depth.items():
        assert len(flags) == 1, f"depth {depth} has inconsistent lever flags"


def test_the_seam_refusal_still_bounds_the_depth():
    """Adding tails must not smuggle past the provider's refusal."""
    f = _solve()
    assert max(p.counts[0] for p in f.points) < 62


# --- what the fix actually surfaced -----------------------------------------


def test_the_fix_surfaces_a_STRICTLY_BETTER_candidate_than_30_18_16():
    """[31,17,16] dominates the cut this ticket was opened for.

    Same pool as the incumbent (it leaves the binder PP2 alone), 1.1765x, and
    -- unlike [30,18,16]'s 1.11x -- its gain clears the rig's 14.1% A-vs-A
    noise floor, so a boot could actually confirm it. The old enumeration
    emitted neither.
    """
    f = _solve(noise_floor=0.141)
    pts = _by_counts(f)
    assert (31, 17, 16) in pts, "the better candidate is still not generated"
    best, prev = pts[(31, 17, 16)], pts[(30, 18, 16)]
    assert best.coupled_pool_tokens == pytest.approx(prev.coupled_pool_tokens, rel=1e-6)
    assert best.compute_speedup > prev.compute_speedup
    assert best.below_noise_floor is False
    assert prev.below_noise_floor is True


def test_POOL_NEUTRALITY_ALONE_DOES_NOT_MEAN_SAFE():
    """The distinction the model cannot see, and a booted counter-example.

    My first cut of this test asserted that every pool-neutral cut keeps the
    incumbent's (7,5,4) attention split. It does not: [32,16,16] is predicted
    pool-neutral with (8,4,4), and [32,16,16] is precisely the layout that was
    BOOTED and FAILED its pool gate (STAGE 1 VERDICT, 2026-08-16).

    The reason is outside this solver. `available_bytes_for_cut` shifts a
    bracket captured at the incumbent by weight and GDN terms only
    (`seam_holdback.py:145-147`), so a change in the ARMING FLOOR is invisible
    to it -- and [32,16,16] measured rank0's floor rise 1728 -> 2255 MiB.
    Applying that measured delta turns its predicted +0.0% into -4.7%.

    So the split, not the predicted pool, is what carries the risk. This test
    pins both groups exist rather than the tidier claim that was false.
    """
    f = _solve()
    inc_pool = _by_counts(f)[INCUMBENT].coupled_pool_tokens
    neutral = [
        p
        for p in f.points
        if abs(p.coupled_pool_tokens - inc_pool) < 1.0 and p.counts != INCUMBENT
    ]
    assert neutral, "expected a pool-neutral family"
    splits = {p.attn_counts for p in neutral}
    assert (7, 5, 4) in splits, "the floor-unexposed family must exist"
    assert (8, 4, 4) in splits, (
        "the floor-EXPOSED family must also be present: if it disappears, the "
        "booted [32,16,16] counter-example no longer has a row and the table "
        "would imply pool-neutral means safe"
    )
    assert (32, 16, 16) in {p.counts for p in neutral}


def test_cuts_that_BEAT_the_incumbent_pool_now_exist():
    """The old frontier's roomiest row WAS the incumbent. It no longer is."""
    f = _solve()
    inc_pool = _by_counts(f)[INCUMBENT].coupled_pool_tokens
    better = [p for p in f.points if p.coupled_pool_tokens > inc_pool + 1.0]
    assert better, "no cut improves on the incumbent pool"
    for p in better:
        assert p.compute_speedup >= 1.0 - 1e-9
    assert max(p.coupled_pool_tokens for p in better) / inc_pool > 1.10
