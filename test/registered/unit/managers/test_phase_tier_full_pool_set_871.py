# SPDX-License-Identifier: Apache-2.0
"""#871: build the phase host tier with the FULL pool set, and prove the guard
stops firing because its precondition is MET -- not because it was removed.

THE DEFECT. `build_phase_flip_host_pools` built the 'tp' staging pin with ONE
entry, `PoolName.KV`. On a hybrid model the live tier carries KV *and* MAMBA, so
`check_pool_coverage` computed `missing = {MAMBA}` and refused the rebind --
correctly, on every cutover. Measured on the W40 #857 acceptance boot: 60
refusals, 0 arms, `#cached-token: 0` on all 243 prefill batches. A refused
rebind leaves the #718 device tier DISARMED, so `load()` returns None and every
read-through misses; each retracted prefix is then recomputed in full.

THE PRECONDITION WAS ALREADY WRITTEN DOWN, in the guard's own docstring
(`hicache_phase_binding.check_pool_coverage`):

    "A phase host tier has to be built with the FULL POOL SET before this
     rebind can arm; until then the #718 disarm is the correct state and a
     read-through miss is the correct cost."

So this is #718/#847's unmet precondition, not a new finding.

WHAT THIS FILE PINS, AND WHY BOTH DIRECTIONS ARE REQUIRED. The difference
between a FIX and a DISARM is not visible from a passing rebind alone -- a
deleted guard also produces a passing rebind. So the acceptance is a PAIR:

    full pool set   -> the guard does NOT fire   (the precondition is met)
    narrowed tier   -> the guard DOES fire       (the guard is still there)

Delete `check_pool_coverage` and the second test goes red. Make it fire
unconditionally and the first goes red. Neither test alone can tell those apart,
which is exactly why the operator asked to see both.

Hermetic: pure set-comparison and pure-function tests. No device pool, no CUDA,
no scheduler, no network. The pool ALLOCATION itself is metal-only and is
deliberately NOT claimed here -- see the module note at the bottom.
"""

import pytest

from sglang.srt.managers.phase_flip_runtime import (
    FENCE_BLIND_STREAK,
    advance_fence_blind_streak,
)
from sglang.srt.mem_cache.hicache_phase_binding import (
    PhasePools,
    RebindRefused,
    check_pool_coverage,
)
from sglang.srt.mem_cache.hicache_storage import PoolName


class _Tier:
    """A host tier stand-in. `check_pool_coverage` reads exactly `entry_map`."""

    def __init__(self, *names):
        self.entry_map = {n: object() for n in names}


class _Reader:
    def __init__(self, tier):
        self.mem_pool_host = tier


def _pools(*names):
    return PhasePools(
        phase="tp", device_pool=object(), host_pool=_Tier(*names), allocator=object()
    )


# ------------------------------------------- THE PAIR THAT SEPARATES FIX/DISARM


def test_a_full_pool_set_lets_the_rebind_arm():
    """#871's acceptance: kv+mamba incoming against a kv+mamba reader is clean.

    This is what the fix buys. Before it, the incoming tier was KV-only and this
    same comparison produced `missing={MAMBA}` on every cutover.
    """
    readers = {"cache_controller": _Reader(_Tier(PoolName.KV, PoolName.MAMBA))}
    check_pool_coverage(readers, _pools(PoolName.KV, PoolName.MAMBA))


def test_a_narrowed_tier_is_still_refused():
    """THE GUARD IS NOT DISARMED. This is the exact pre-#871 shape.

    If someone 'fixes' #871 by weakening `check_pool_coverage`, this goes red.
    """
    readers = {"cache_controller": _Reader(_Tier(PoolName.KV, PoolName.MAMBA))}
    with pytest.raises(RebindRefused) as e:
        check_pool_coverage(readers, _pools(PoolName.KV))
    msg = str(e.value)
    assert "mamba" in msg.lower(), msg
    assert "lose its host backing" in msg, msg


def test_the_guard_does_not_fire_on_a_wider_incoming_tier():
    """Only pools the reader ALREADY names are required -- the docstring's rule.

    Pins that the fix did not turn the comparison into an equality, which would
    refuse a legitimately richer tier.
    """
    readers = {"cache_controller": _Reader(_Tier(PoolName.KV))}
    check_pool_coverage(readers, _pools(PoolName.KV, PoolName.MAMBA))


def test_a_reader_naming_no_tier_is_not_turned_into_a_coverage_claim():
    """The scheduler names the allocator only; it must not manufacture a claim."""
    readers = {"scheduler": _Reader(None), "tree_cache": _Reader(_Tier(PoolName.KV))}
    check_pool_coverage(readers, _pools(PoolName.KV))


@pytest.mark.parametrize(
    "incoming,bound,refuses",
    [
        ((PoolName.KV,), (PoolName.KV,), False),
        ((PoolName.KV,), (PoolName.KV, PoolName.MAMBA), True),  # the #871 defect
        ((PoolName.KV, PoolName.MAMBA), (PoolName.KV, PoolName.MAMBA), False),
        ((PoolName.KV, PoolName.MAMBA), (PoolName.KV,), False),
        ((PoolName.MAMBA,), (PoolName.KV, PoolName.MAMBA), True),
    ],
)
def test_the_coverage_truth_table(incoming, bound, refuses):
    """The whole table, so neither direction can be broken without a red row."""
    readers = {"cache_controller": _Reader(_Tier(*bound))}
    if refuses:
        with pytest.raises(RebindRefused):
            check_pool_coverage(readers, _pools(*incoming))
    else:
        check_pool_coverage(readers, _pools(*incoming))


# ------------------------------------------------- THE CHECK (scope item 3)


def test_the_streak_advances_only_on_a_work_retracting_blind_fence():
    assert advance_fence_blind_streak(0, released=True, persisted_nothing=True) == 1
    assert advance_fence_blind_streak(3, released=True, persisted_nothing=True) == 4


def test_an_idle_cutover_never_accumulates_a_streak():
    """THE CRYING-WOLF GUARD, mirroring #719's busy gate.

    A fence over an empty tree is CORRECT to persist nothing. If this counted,
    an idle instance would alarm and the alarm would be ignored -- which is how
    the condition stayed invisible in the first place.
    """
    assert advance_fence_blind_streak(3, released=False, persisted_nothing=True) == 0


def test_a_fence_that_persisted_something_resets_the_streak():
    """Reports a SUSTAINED condition, never a historical one."""
    assert advance_fence_blind_streak(9, released=True, persisted_nothing=False) == 0


def test_the_threshold_is_above_an_ordinary_quiet_stretch():
    """#719 settled that two consecutive quiet cutovers are ordinary."""
    assert FENCE_BLIND_STREAK > 2


def test_the_w37g_specimen_reaches_the_alarm():
    """12 flips, zero completions: every cutover retracts and persists nothing."""
    streak = 0
    for _ in range(12):
        streak = advance_fence_blind_streak(
            streak, released=True, persisted_nothing=True
        )
    assert streak >= FENCE_BLIND_STREAK


def test_an_alternating_instance_never_alarms():
    """One empty fence between good ones is legitimate and must stay silent."""
    streak = 0
    for i in range(20):
        streak = advance_fence_blind_streak(
            streak, released=True, persisted_nothing=(i % 2 == 0)
        )
        assert streak < FENCE_BLIND_STREAK


# NOT CLAIMED BY THIS FILE, stated rather than left to be assumed:
# whether the kv+mamba pin can actually be ALLOCATED on this box. That needs a
# real device pool and a real host budget, so it is decidable only on metal and
# belongs to a boot window, not to this suite. What is pinned here is the
# DECISION -- that a full pool set arms and a narrowed one does not -- and the
# streak logic. The allocation, its HOST-LEDGER post, and whether
# `#cached-token` becomes non-zero are the window's questions.
