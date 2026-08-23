# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#834 B step 1: ``recover_kv_backing`` splits into a LOCAL and a COLLECTIVE half.

WHY THE SPLIT IS THE FIRST THING BUILT. #830 F2 measured this function at
99-101% of the cutover term across 63 rank-flips in two independent boots -- the
cutover went 50.2 ms with HiCache off to 3448.9 / 3773.0 ms with it on, and the
gap tracking it brackets only this call. It is a KV-pool grow running inside the
no-return window with the requests parked and the ring mid-rebuild.

The design note at the call site (phase_flip_runtime.py, at the tp_to_pp
``recover_kv_backing``) says what to do about it, in order, and step 1 is:

    "Split recover_kv_backing into grow (rank-local) and level (collective) as
    separate callables. The seam then runs grow-then-level as today, with no
    behaviour change, and the split is provable by test before anything moves."

THIS FILE IS THAT PROOF, and it is deliberately about the split itself rather
than about the deferral built on top of it. If the composition of the two halves
is not the shipped function, everything downstream is built on sand.

THE TWO HALVES ARE NOT INTERCHANGEABLE and the file's red arms are about why:

  * the LOCAL half (``grow_kv_backing_local``) reaches
    ``runtime_set_backing_rows`` -> cuMemCreate/cuMemMap. It touches no
    collective -- ``kv_backing_relief.py`` has none anywhere -- so its timing
    is free. It also RAISES this rank's exposure to its own new backing, via
    ``clamp_exposure_to_backing`` at the end of ``recover()``, which is correct
    only when a levelling follows immediately.
  * the COLLECTIVE half (``level_kv_backing_to_group``) is one MIN reduction
    over three integers plus, at most, one non-allocating cap engage. It is
    cheap, and it is the only thing standing between a corridor-bounded grow
    and a group with two different id spaces -- measured on this rig at
    210944 / 124928 / 131072 backed rows. An id one rank exposes and a peer
    cannot map aborts ALL THREE inside ``store_kvcache``'s bounds assert.

The harness is #792's fleet, reused rather than rebuilt: it is the one stub in
the tree whose fake channel already carries ``recover_kv_backing``'s own
three-field payload.
"""

import unittest

from sglang.srt.managers import phase_flip_spill as pfs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_residency_cap_flip_levelling_792 import (  # noqa: E402 (sibling harness)
    POOL,
    _group_channel,
    _rank,
    _scheduler,
)

register_cpu_ci(est_time=5)

#: A LIVE SET SMALL ENOUGH THAT THE LEVELLING ACTUALLY APPLIES, and this is not
#: a convenience -- it is the difference between two different states of the
#: same code. #792's own fleet is deliberately the DECLINE case: its group
#: minimum (1024) sits below the group's live floor (3417), so
#: ``level_kv_backing_to_group`` correctly refuses to level and returns None.
#: That fixture proves the decline and cannot prove the levelling. A split that
#: is only ever exercised on the declining path is a split nobody has watched
#: do its job, so this file carries both fleets and says which is which.
LIVE_TOP_SMALL = 256
GROUP_MIN_LEVELLED = 2048


def _levelling_rank(backed):
    """A rank whose group CAN be levelled: live set well under the group min."""
    return _rank(backed=backed, live_top=LIVE_TOP_SMALL, capped_at=None)


def _levelling_fleet():
    """Uneven backing, a live floor far below the poorest rank's backing."""
    return [
        _levelling_rank(POOL),
        _levelling_rank(GROUP_MIN_LEVELLED),
        _levelling_rank(POOL),
    ]


def _declining_fleet():
    """#792's shape: the group minimum is BELOW the group's live floor, so no
    honest level exists and the levelling declines."""
    from test_residency_cap_flip_levelling_792 import _fleet

    return _fleet()


def _grower(backed):
    """A rank that will actually GROW when ``recover()`` is called.

    ``_rank`` leaves ``_rows_at_boot`` None, which makes ``recover()`` return 0
    at its first line -- that is instr12's state and exactly right for #792's
    subject. A deferred grow is the opposite case: the rank REMEMBERS a boot
    reservation it has not got back yet, which is the state whose cost #830 F2
    measured at 99% of the cutover.
    """
    relief = _levelling_rank(backed)
    relief._rows_at_boot = POOL
    return relief


def _exposures(fleet):
    return [r.exposed_rows() for r in fleet]


def _backings(fleet):
    return [r.backed_rows() for r in fleet]


class TestTheSplitIsTheShippedFunction(CustomTestCase):
    """Step 1's own acceptance: composition == the function it replaced."""

    def test_green_grow_then_level_reproduces_recover_kv_backing(self):
        """A/B on the SAME fleet shape, run twice: once through the shipped
        entry point, once through the two halves called in order. Both the
        returned grow and the resulting exposure must agree, or the split is
        not a split but a rewrite."""
        via_whole = _levelling_fleet()
        reduce_whole = _group_channel(via_whole)
        whole_returns = [
            pfs.recover_kv_backing(_scheduler(r), reduce_fn=reduce_whole)
            for r in via_whole
        ]

        via_halves = _levelling_fleet()
        reduce_halves = _group_channel(via_halves)
        halves_returns = []
        for r in via_halves:
            sched = _scheduler(r)
            halves_returns.append(pfs.grow_kv_backing_local(sched))
            pfs.level_kv_backing_to_group(sched, reduce_halves)

        self.assertEqual(whole_returns, halves_returns)
        self.assertEqual(_exposures(via_whole), _exposures(via_halves))
        self.assertEqual(_backings(via_whole), _backings(via_halves))

    def test_green_the_levelling_reports_the_level_it_agreed(self):
        """The return value is NEW and load-bearing: a deferred grow has to
        know which level it must clamp back to, and before the split there was
        nothing that could tell it."""
        fleet = _levelling_fleet()
        reduce_fn = _group_channel(fleet)
        levels = [
            pfs.level_kv_backing_to_group(_scheduler(r), reduce_fn) for r in fleet
        ]
        self.assertEqual([GROUP_MIN_LEVELLED] * len(fleet), levels)

    def test_green_an_already_even_group_still_reports_its_level(self):
        """THE COMMONEST CASE, and the one a naive implementation gets wrong.

        An even group needs no cap applied to be level -- so an implementation
        that reports the level only from the branch that MOVES rows returns
        None here. A deferred grow reading that None concludes there is no
        agreed level and must refuse to expose anything, permanently, on a
        perfectly healthy fleet."""
        fleet = [_levelling_rank(POOL) for _ in range(3)]
        reduce_fn = _group_channel(fleet)
        levels = [
            pfs.level_kv_backing_to_group(_scheduler(r), reduce_fn) for r in fleet
        ]
        self.assertEqual([POOL] * 3, levels)
        for level in levels:
            self.assertIsNotNone(
                level,
                "an even group agreed just as much as an uneven one; it simply "
                "needed no cap to get there",
            )

    def test_can_fail_the_local_half_reaches_no_collective(self):
        """RED ARM for the claim the whole deferral rests on. If the local half
        ever touches the channel, deferring it to a rank-local cadence becomes
        the 2026-08-08 boots 9/10 PP wedge -- a blocking reduction entered at a
        local cadence pairing with a peer blocked in a pipeline recv."""
        fleet = _levelling_fleet()
        calls = []
        reduce_fn = _group_channel(fleet)

        def watched(vals):
            calls.append(list(vals))
            return reduce_fn(vals)

        for r in fleet:
            pfs.grow_kv_backing_local(_scheduler(r))
        self.assertEqual(
            [],
            calls,
            "grow_kv_backing_local reached a collective; the deferral's whole "
            "safety argument is that it cannot",
        )

        # AND THE LEVELLING HALF MUST REACH IT. Without this second half the
        # arm above passes trivially on a split that put the collective
        # nowhere at all -- which would be a far worse bug than the one it
        # watches for, and would look identical from here.
        #
        # A RECORDER, NOT A TRIPWIRE. ``level_kv_backing_to_group`` catches
        # every exception by design ("the ranks may now expose different id
        # spaces ... a lost flip, never a half-flipped group"), so a channel
        # that raises proves nothing: the raise is swallowed and the test
        # cannot tell that from a channel never called. Counting calls can.
        pfs.level_kv_backing_to_group(_scheduler(fleet[0]), watched)
        self.assertEqual(
            1, len(calls), "the levelling half must reach the channel exactly once"
        )
        self.assertEqual(
            3,
            len(calls[0]),
            "the payload is [backed, -backed, -floor]; a shorter one is the "
            "#792 truncation that leaves the floor unknown",
        )

    def test_can_fail_the_levelling_is_not_a_no_op(self):
        """RED ARM. A levelling that returns a number without capping anything
        is a logging change wearing the name of the guard that stops a
        three-rank abort."""
        fleet = _levelling_fleet()
        before = _exposures(fleet)
        reduce_fn = _group_channel(fleet)
        for r in fleet:
            pfs.level_kv_backing_to_group(_scheduler(r), reduce_fn)
        after = _exposures(fleet)
        self.assertNotEqual(
            before,
            after,
            "the uneven fleet must come out capped; if exposure is unchanged "
            "the levelling did not level",
        )
        self.assertEqual(
            [GROUP_MIN_LEVELLED] * len(fleet),
            after,
            "every rank must expose the group's poorest backing, and no more",
        )

    def test_can_fail_a_declined_levelling_reports_NO_level(self):
        """RED ARM, and the one that guards the deferral's worst input.

        #792's fleet is the state where no honest level exists: the group's
        poorest rank backs fewer rows than the group's LIVE SET occupies, so
        capping to the minimum would confiscate ids the radix tree is holding
        (measured: cap levelled to 40960 against a highest live row of 136720,
        63641 ids confiscated, the next prefill out-of-memory against 67674
        evictable tokens). The levelling correctly DECLINES.

        A decline is not a level, and reporting it as one is worse here than it
        was before the split: a deferred grow reading a level believes it has
        group permission to expose up to it. So the decline must come back as
        None, and the caller's own arm (hazard 1, in the runtime file) refuses
        rather than exposes."""
        fleet = _declining_fleet()
        reduce_fn = _group_channel(fleet)
        levels = [
            pfs.level_kv_backing_to_group(_scheduler(r), reduce_fn) for r in fleet
        ]
        self.assertEqual(
            [None] * len(fleet),
            levels,
            "a declined levelling must report NO level; handing the group "
            "minimum back as an agreed one authorises exactly the exposure "
            "the decline exists to prevent",
        )

    def test_can_fail_no_channel_never_levels_and_never_pretends_to(self):
        """A single-rank shape is level with itself. Reporting a level it never
        agreed would hand a deferred grow a ceiling nobody voted for."""
        relief = _levelling_rank(POOL)
        self.assertIsNone(pfs.level_kv_backing_to_group(_scheduler(relief), None))


class TestExposureClampIsTheDeferralsInvariant(CustomTestCase):
    """#834 B step 2, at the level the design note says to write it first:

        "Defer ONLY the grow, and keep the pool's exposure clamped to the
        pre-grow group level until the levelling runs, so no rank can expose an
        id a peer has not backed. That is the invariant to write the red arm
        against FIRST."
    """

    def test_green_a_grown_rank_can_be_held_at_the_agreed_level(self):
        """The primitive itself: back rows, expose none of them."""
        fleet = _levelling_fleet()
        reduce_fn = _group_channel(fleet)
        levels = [
            pfs.level_kv_backing_to_group(_scheduler(r), reduce_fn) for r in fleet
        ]
        agreed = levels[0]

        # Now the deferred grow lands on one rank, rank-locally.
        grower = fleet[1]
        sched = _scheduler(grower)
        pfs.grow_kv_backing_local(sched)
        pfs.clamp_kv_exposure_to_level(sched, agreed)

        self.assertLessEqual(
            grower.exposed_rows(),
            agreed,
            "a rank that grew after the seam levelled must not expose the new "
            "rows; a peer has not backed them",
        )

    def test_can_fail_the_grow_alone_raises_exposure_above_the_agreed_level(self):
        """RED ARM, AND THE HAZARD ITSELF.

        This is not a hypothetical: ``recover()`` ends in
        ``clamp_exposure_to_backing``, which lifts the allocator to this rank's
        OWN new backing. That is correct when the levelling follows in the next
        statement, and it is the three-rank abort when it does not. The arm
        pins that the danger is REAL -- an unclamped deferred grow does exceed
        the agreed level -- because an invariant test that would pass even
        without the clamp proves nothing about the clamp."""
        fleet = _levelling_fleet()
        reduce_fn = _group_channel(fleet)
        agreed = pfs.level_kv_backing_to_group(_scheduler(fleet[1]), reduce_fn)
        for r in (fleet[0], fleet[2]):
            pfs.level_kv_backing_to_group(_scheduler(r), reduce_fn)

        grower = fleet[1]
        grower._rows_at_boot = POOL
        sched = _scheduler(grower)
        pfs.grow_kv_backing_local(sched)
        self.assertGreater(
            grower.exposed_rows(),
            agreed,
            "if an unclamped grow does NOT raise exposure past the agreed "
            "level on this fixture, the clamp above is being proved against a "
            "fixture that cannot fail and the invariant is untested",
        )
        # And the clamp is what puts it back.
        pfs.clamp_kv_exposure_to_level(sched, agreed)
        self.assertLessEqual(grower.exposed_rows(), agreed)

    def test_can_fail_an_unknown_level_never_becomes_a_cap(self):
        """#721's rule, applied to a clamp instead of a refusal: a number we do
        not have must not turn into an action. A 0 or negative 'level' is the
        #792 decline arriving here, and capping to it would confiscate the
        whole id space."""
        relief = _levelling_rank(POOL)
        sched = _scheduler(relief)
        before = relief.exposed_rows()
        self.assertEqual(0, pfs.clamp_kv_exposure_to_level(sched, 0))
        self.assertEqual(0, pfs.clamp_kv_exposure_to_level(sched, -1))
        self.assertEqual(before, relief.exposed_rows())


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
