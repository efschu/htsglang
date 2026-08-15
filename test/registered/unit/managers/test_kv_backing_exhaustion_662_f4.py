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
"""#662-F4: the rung disqualified itself for the life of the process.

THE MEASUREMENT, from metal on 2026-08-15. A tp_to_pp gate, all three ranks:

    KV-BACKING proposal on device 0: rows current=407051 floor=1157
      (max_live=644 + admission reserve 512, slack=405894)
      | need = floor 1536 + delta 256 + want 1522 = 3314 MiB
    KV-BACKING shrink to 222081 rows reported 0 MiB but the driver's free
      column did not move, so this pool cannot pay
    ...
    KV-BACKING proposal on device 0: rows current=222081 floor=51713
      (max_live=51200 + admission reserve 512, slack=170368)
      | this rank's arena is EXHAUSTED (a previous shrink returned no driver
        bytes), so it stops asking; the deficit is not computed

170 368 rows of slack, refused. The tp_to_pp seam then abandoned nine times
for want of ~500 MiB and the instance never reached the prefill layout again.

TWO DEFECTS, AND THE FIRST CAUSES THE SECOND.

1. THE SLACK WAS FICTION. ``_current_rows`` read ``full_pool_backed_rows``,
   whose name promises a measurement and which returns ``full_kv_pool.size``,
   a CONFIGURED row count. The #330 dial writes ``size`` on every step, so the
   two agreed for as long as the dial was the only writer. The phase flip is
   not the dial: ``release_backing`` / ``restore_backing`` unmap and remap the
   pages and state in their own comment that "SIZING IS NOT TOUCHED". So
   during the TP phase the PP layout's pool holds no committed extents at all
   while ``size`` still reports its pre-flip count. The shrink above could not
   have returned a byte -- there was nothing mapped to release.

2. THE ZERO WAS THEN READ AS PERMANENT. ``_exhausted`` was a bool set once and
   never reconsidered, so one shrink of an emptied layout switched the only
   rung that can pay real bytes off for the rest of the process -- on BOTH
   legs, including the leg where the same pool is fully backed and can pay.

Exhaustion is evidence about ONE backing level. These tests pin that reading:
it survives while the backing stands still, and expires the moment it moves.

Hermetic: tensor-backed fakes, no CUDA, no scheduler.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as kbr

MIB = 1024 * 1024

#: One row costs a MiB here, so rows and MiB read interchangeably.
BYTES_PER_ROW = MIB

#: The configured row count. ``size`` reports this forever; the flip does not
#: touch it.
CONFIGURED_ROWS = 4_000

#: Rows a request in flight holds.
LIVE_ROWS = 40

RESERVE = 512


class _FakeAllocator:
    def __init__(self, size: int):
        self.size = size
        self.page_size = 1
        self._free_listeners = []
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)

    def register_free_listener(self, on_free, on_clear=None):
        self._free_listeners.append((on_free, on_clear))

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)

    def free(self, idx):
        self.free_pages = torch.cat((self.free_pages, idx))
        for on_free, _ in self._free_listeners:
            on_free(idx)


class _Desc:
    def __init__(self, row_bytes: int):
        self.row_bytes = row_bytes
        self.tokens_per_row = 1


class _Spec:
    def __init__(self, row_bytes: int):
        self.desc = _Desc(row_bytes)


class _FakeOwner:
    """One buffer carrying the whole per-row cost, which is all the geometry
    reader needs: it sums ``row_bytes // tokens_per_row`` across specs."""

    def __init__(self, bytes_per_row: int):
        self._specs = [_Spec(bytes_per_row)]


class _FlipPool:
    """A VA-reserved pool with the phase flip's two independent facts.

    ``size`` is the CONFIGURED row count. ``backed_bytes`` is what the arena
    actually has mapped. The flip moves the second and never the first, which
    is the whole subject of this file, so the fake keeps them apart the way
    the real pool does.

    ``runtime_set_backing_rows`` releases from the PHYSICAL level, so asking an
    emptied pool to shrink returns zero -- not because the arena refused, but
    because there was nothing there.
    """

    def __init__(self, rows: int, card=None):
        self.size = rows
        self.page_size = 1
        self._bytes_per_row = BYTES_PER_ROW
        self._card = card
        self.supports_backing_spans = True
        self.calls = []
        self.backed_bytes = rows * BYTES_PER_ROW
        # The arena descriptors ``row_geometry`` prices against. A rebind
        # re-derives geometry from the pool it moves to, so the fake has to
        # carry the same shape the real pool family does.
        self._post_capture_owner = _FakeOwner(BYTES_PER_ROW)

    # -- what the phase flip does to this pool, and only this ------------
    def flip_release(self) -> None:
        """The seam hands this layout's pages back; ``size`` is untouched."""
        self.backed_bytes = 0

    def flip_restore(self) -> None:
        """The seam remaps this layout at its unchanged addresses."""
        self.backed_bytes = self.size * BYTES_PER_ROW

    # -- the #330 dial ---------------------------------------------------
    def runtime_set_backing_rows(self, rows: int) -> int:
        # Converges in BOTH directions, like the real dial: a grow commits,
        # a shrink releases, and the release is measured from the PHYSICAL
        # level -- so asking an emptied pool to shrink returns zero.
        rows = int(rows)
        self.calls.append(rows)
        want = rows * BYTES_PER_ROW
        released = max(0, self.backed_bytes - want)
        self.backed_bytes = want
        self.size = rows
        if self._card is not None:
            self._card.free += released
        return released


class _Card:
    def __init__(self, free_mib: int):
        self.free = free_mib * MIB

    def probe(self) -> int:
        return self.free


def _rig(card_free_mib: int = 1100):
    card = _Card(card_free_mib)
    pool = _FlipPool(CONFIGURED_ROWS, card=card)
    alloc = _FakeAllocator(CONFIGURED_ROWS)
    relief = kbr.KvBackingRelief(
        pool,
        alloc,
        live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
        bytes_per_row=BYTES_PER_ROW,
        probe=card.probe,
        admission_reserve_rows=RESERVE,
    )
    return relief, pool, card, alloc


class BackedRowsAreMeasuredNotConfiguredTest(unittest.TestCase):
    """Defect 1: the rung believed a pool the flip had emptied was full."""

    def test_a_resident_layout_reads_its_backing(self):
        relief, pool, _card, _alloc = _rig()
        self.assertEqual(relief.backed_rows(), CONFIGURED_ROWS)

    def test_an_emptied_layout_reports_no_backing_not_its_configured_size(self):
        # THE HEADLINE. During the TP phase the PP pool is unmapped, and
        # ``size`` keeps reporting the pre-flip count. Before the fix this
        # returned CONFIGURED_ROWS and every number downstream was fiction.
        relief, pool, _card, _alloc = _rig()
        pool.flip_release()
        self.assertEqual(pool.size, CONFIGURED_ROWS, "the flip must not size")
        self.assertEqual(relief.backed_rows(), 0)

    def test_a_pool_without_the_reading_keeps_the_previous_behaviour(self):
        # A pool that never flips does not expose ``backed_bytes``. It must
        # fall back exactly as before, or this fix would be a regression for
        # every non-flip boot.
        class _Plain:
            size = 1234
            page_size = 1
            supports_backing_spans = True

            def runtime_set_backing_rows(self, rows):
                return 0

        relief = kbr.KvBackingRelief(
            _Plain(),
            _FakeAllocator(1234),
            live_slots_fn=lambda: torch.tensor([1], dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
        )
        self.assertEqual(relief.backed_rows(), 1234)


class AnEmptiedLayoutIsNotAFundingOpportunityTest(unittest.TestCase):
    """The proposal must not invent slack out of a configured row count."""

    def test_the_rung_abstains_instead_of_proposing_a_phantom_shrink(self):
        relief, pool, _card, _alloc = _rig()
        pool.flip_release()
        proposal = relief.propose(
            want_bytes=1522 * MIB, floor_bytes=1536 * MIB, delta_bytes=256 * MIB
        )
        self.assertEqual(proposal, kbr.ABSTAIN)
        # An abstention cancels the group's shrink, which is correct here:
        # nobody has backing to give on this leg.
        self.assertIsNone(kbr.collective_kv_target([proposal[0], 0, 0, 0]))

    def test_no_shrink_is_attempted_against_an_emptied_layout(self):
        relief, pool, _card, _alloc = _rig()
        pool.flip_release()
        relief.propose(
            want_bytes=1522 * MIB, floor_bytes=1536 * MIB, delta_bytes=256 * MIB
        )
        self.assertEqual(pool.calls, [], "there is nothing mapped to release")

    def test_an_emptied_layout_does_not_latch_the_rung_off(self):
        # THE CHAIN, END TO END. The rung is consulted while the layout is
        # emptied (the tp_to_pp gate), and must still be able to pay on the
        # next leg, when the same pool is resident again. Before the fix the
        # first consultation shrank a pool that could not pay, latched
        # ``_exhausted``, and the rung never funded anything again.
        relief, pool, card, _alloc = _rig()
        pool.flip_release()
        relief.propose(
            want_bytes=1522 * MIB, floor_bytes=1536 * MIB, delta_bytes=256 * MIB
        )
        pool.flip_restore()
        self.assertEqual(
            relief.free_up_to(600 * MIB),
            600 * MIB,
            "a restored layout must be fundable again",
        )


class ExhaustionIsEvidenceAboutOneLevelTest(unittest.TestCase):
    """Defect 2, and the inversion of the process-lifetime pin.

    ``test_a_pool_that_paid_nothing_is_not_asked_twice`` in
    ``test_kv_backing_relief_631`` still holds and still must: a rung that
    paid nothing is not asked again AT THE SAME LEVEL. What it never meant --
    and what cost the prefill layout -- is that it may never be asked again at
    all.
    """

    def _futile(self):
        """A rig whose shrink returns nothing while the card never moves."""
        card = _Card(1100)
        pool = _FlipPool(CONFIGURED_ROWS, card=None)  # card deliberately fixed
        alloc = _FakeAllocator(CONFIGURED_ROWS)
        relief = kbr.KvBackingRelief(
            pool,
            alloc,
            live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
        )
        return relief, pool, card

    def test_the_same_level_is_not_asked_twice(self):
        relief, pool, _card = self._futile()
        relief.free_up_to(600 * MIB)
        relief.free_up_to(600 * MIB)
        self.assertEqual(len(pool.calls), 1)

    def test_a_moved_backing_re_arms_the_rung(self):
        # The inverted pin. The arena that could not pay at one level is a
        # different proposition at another, and one no-op is not a verdict on
        # the process. Here the phase flip restores the layout underneath the
        # rung -- exactly the metal sequence -- and the rung must ask again.
        relief, pool, _card = self._futile()
        relief.free_up_to(600 * MIB)
        self.assertEqual(len(pool.calls), 1)
        # The metal sequence: this layout goes inactive for a phase and comes
        # back. Its pages were released to the driver and remapped from new
        # handles, so the earlier no-op says nothing about the arena now. The
        # rung IS consulted while the layout is down -- ``propose`` is on the
        # gate's unconditional path, which is the property the collective
        # depends on -- and that consultation is where it sees the zero.
        pool.flip_release()
        relief.propose(
            want_bytes=512 * MIB, floor_bytes=1536 * MIB, delta_bytes=256 * MIB
        )
        pool.flip_restore()
        relief.free_up_to(600 * MIB)
        self.assertEqual(len(pool.calls), 2, "a moved backing is new evidence")

    def test_the_trace_stops_blaming_the_arena_once_the_backing_moved(self):
        relief, pool, _card = self._futile()
        relief.free_up_to(600 * MIB)
        self.assertTrue(relief._exhausted)
        pool.flip_release()
        self.assertFalse(relief._exhausted)

    def test_recovery_still_clears_exhaustion_outright(self):
        # Unchanged behaviour, pinned so the level-keying cannot quietly
        # replace it: a full recovery clears the marker whatever the level.
        relief, pool, card = self._futile()
        relief.free_up_to(600 * MIB)
        self.assertTrue(relief._exhausted)
        card.free = 40_000 * MIB
        relief.recover()
        self.assertFalse(relief._exhausted)


if __name__ == "__main__":
    unittest.main()


class TheFunderFollowsTheResidentLayoutTest(unittest.TestCase):
    """#662-F4, second half: fund the seam from the pool that HAS pages.

    The scheduler's KV pool is the PP layout's. On the pp_to_tp leg that is the
    SOURCE -- backed, with slack above the live set, able to pay. On the
    tp_to_pp leg the same pool is the DESTINATION, and the seam emptied it a
    phase ago. A rung captured on it can only ever fund one of the two legs,
    which is why the leg into the prefill layout still had no funder after the
    exclusion in ``collective_kv_backing_relief`` was lifted.

    The money on that leg is the TP layout's pool: it is the source, it is
    fully backed, and the rows above its high-water mark hold nothing.
    """

    def _two_layouts(self):
        card = _Card(1100)
        pp = _FlipPool(CONFIGURED_ROWS, card=card)
        tp = _FlipPool(CONFIGURED_ROWS, card=card)
        alloc = _FakeAllocator(CONFIGURED_ROWS)
        phase = {"active": "pp"}

        def pool_fn():
            return tp if phase["active"] == "tp" else pp

        relief = kbr.KvBackingRelief(
            pp,
            alloc,
            live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
            pool_fn=pool_fn,
        )
        return relief, pp, tp, card, phase

    def test_in_the_pp_phase_it_funds_from_the_scheduler_pool(self):
        relief, pp, tp, _card, _phase = self._two_layouts()
        self.assertEqual(relief.free_up_to(600 * MIB), 600 * MIB)
        self.assertEqual(len(pp.calls), 1)
        self.assertEqual(tp.calls, [], "the inactive layout is not touched")

    def test_in_the_tp_phase_it_funds_from_the_resident_layout(self):
        # THE HEADLINE. The flip has cut over, so the PP pool is empty and the
        # TP pool holds the pages. Before this, the rung proposed against the
        # empty one and released nothing.
        relief, pp, tp, _card, phase = self._two_layouts()
        phase["active"] = "tp"
        pp.flip_release()
        self.assertEqual(relief.free_up_to(600 * MIB), 600 * MIB)
        self.assertEqual(tp.calls, [3400], "the resident layout pays")
        self.assertEqual(pp.calls, [], "the emptied layout is not asked")

    def test_the_bound_pool_is_never_asked_for_pages_it_does_not_have(self):
        relief, pp, tp, _card, phase = self._two_layouts()
        phase["active"] = "tp"
        pp.flip_release()
        relief.propose(
            want_bytes=1522 * MIB, floor_bytes=1536 * MIB, delta_bytes=256 * MIB
        )
        self.assertEqual(pp.calls, [])

    def test_exhaustion_does_not_follow_the_rung_across_layouts(self):
        # A marker is a fact about ONE arena. Carrying it across a rebind would
        # let a futile shrink of one layout silence the other.
        relief, pp, tp, card, phase = self._two_layouts()
        pp._card = None  # the release returns nothing the driver can see
        self.assertEqual(relief.free_up_to(600 * MIB), 0, "a futile shrink")
        self.assertTrue(relief._exhausted, "the PP arena is marked")
        # Asked again in the same phase, the marked arena stays silent.
        relief.free_up_to(600 * MIB)
        self.assertEqual(len(pp.calls), 1)
        # The other layout is a different arena and owes nothing to that mark.
        phase["active"] = "tp"
        self.assertEqual(relief.free_up_to(600 * MIB), 600 * MIB)
        self.assertEqual(tp.calls, [3400])

    def test_the_id_space_does_not_follow_the_rung_across_layouts(self):
        # exposed_rows feeds the collective cap agreement. Letting it track
        # whichever layout is resident would make the group's agreed id space
        # depend on each rank's phase -- the capacity desync of HANDOFF_675 1a.
        relief, pp, tp, _card, phase = self._two_layouts()
        tp.size = CONFIGURED_ROWS // 2
        before = relief.exposed_rows()
        phase["active"] = "tp"
        relief.backed_rows()  # force a rebind
        self.assertEqual(relief.exposed_rows(), before)

    def test_an_unresolvable_pool_keeps_the_previous_behaviour(self):
        card = _Card(1100)
        pp = _FlipPool(CONFIGURED_ROWS, card=card)

        def boom():
            raise RuntimeError("no stacks installed")

        relief = kbr.KvBackingRelief(
            pp,
            _FakeAllocator(CONFIGURED_ROWS),
            live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
            pool_fn=boom,
        )
        self.assertEqual(relief.free_up_to(600 * MIB), 600 * MIB)
        self.assertEqual(len(pp.calls), 1)


class ARefusalMustSayWhatTheRungDecidedTest(unittest.TestCase):
    """A silent decline is indistinguishable from a rung that never ran.

    The proposal trace is edge-triggered on the deficit's sign, which keeps a
    steady state quiet -- correct -- but a REFUSAL is not an edge, so at the
    one moment a reader needs the terms there are none. Measured the hard way
    on 2026-08-15: the seam was refused by 59 MiB, this rung had emitted
    nothing for five minutes, and that was read as "the rung was never
    consulted", sending the diagnosis after a missing call that did not exist.
    It had been consulted at every gate and had declined quietly.

    "Declined", "abstained" and "never reached" have three different fixes, so
    the refusal has to distinguish them.
    """

    def _relief(self):
        card = _Card(4000)
        pool = _FlipPool(CONFIGURED_ROWS, card=card)
        return kbr.KvBackingRelief(
            pool,
            _FakeAllocator(CONFIGURED_ROWS),
            live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
        )

    def test_a_rung_that_never_proposed_says_exactly_that(self):
        summary = self._relief().last_proposal_summary()
        self.assertIn("NO proposal", summary)
        self.assertIn("not reached", summary)

    def test_a_quiet_decline_is_still_reportable(self):
        relief = self._relief()
        # Plenty free, so the deficit is negative and the trace stays silent.
        relief.propose(
            want_bytes=10 * MIB,
            floor_bytes=10 * MIB,
            delta_bytes=0,
            cheap_relief_bytes=0,
        )
        summary = relief.last_proposal_summary()
        self.assertIn("KV rung:", summary)
        self.assertIn("no change", summary)
        self.assertIn("slack=", summary), "the number the refusal argument turns on"

    def test_the_terms_survive_even_when_the_trace_did_not_log(self):
        relief = self._relief()
        relief.propose(want_bytes=10 * MIB, floor_bytes=10 * MIB, delta_bytes=0)
        first = relief.last_proposal_summary()
        # A second identical round is not an edge, so nothing logs -- and the
        # summary must still be available.
        relief.propose(want_bytes=10 * MIB, floor_bytes=10 * MIB, delta_bytes=0)
        self.assertEqual(relief.last_proposal_summary(), first)

    def test_an_emptied_layout_reports_its_abstention_not_a_decline(self):
        relief = self._relief()
        relief._pool.flip_release()
        relief.propose(want_bytes=10 * MIB, floor_bytes=10 * MIB, delta_bytes=0)
        # An abstain never reaches the trace, so the summary correctly reports
        # that no proposal was made rather than inventing one.
        self.assertIn("NO proposal", relief.last_proposal_summary())


class ExhaustionMustNotBeSelfLockingTest(unittest.TestCase):
    """The deadlock, pinned. Keying exhaustion to the LEVEL alone cannot work.

    A shrink that releases nothing leaves the physical level exactly where it
    was. So a marker keyed only to the level marks the level the rung is stuck
    at, and the only thing that could move it is a successful shrink -- which
    the marker now prevents. Measured on this rig 2026-08-15: a shrink to
    94955 rows returned no driver bytes at 12:16:00, and 47 seconds later all
    three ranks were still declining with 72981 rows of slack in front of
    them, at an unchanged level, while 77k tokens sat pending.

    The target is what makes the evidence falsifiable: "a shrink to X returned
    nothing" says nothing about a shrink deeper than X, because release is
    extent-granular and a deeper ask clears extents a shallower one cannot.
    """

    def _rig(self):
        card = _Card(1100)
        pool = _FlipPool(CONFIGURED_ROWS, card=None)  # shrinks return nothing
        alloc = _FakeAllocator(CONFIGURED_ROWS)
        relief = kbr.KvBackingRelief(
            pool,
            alloc,
            live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
        )
        return relief, pool, card

    def test_the_same_ask_is_still_declined(self):
        relief, pool, _card = self._rig()
        relief.free_up_to(600 * MIB)
        self.assertTrue(relief._declines_target(pool.size))

    def test_a_DEEPER_ask_is_a_different_question(self):
        """The escape from the deadlock, and the whole point of the target."""
        relief, pool, _card = self._rig()
        relief.free_up_to(600 * MIB)
        failed = relief._exhausted_target_rows
        self.assertIsNotNone(failed)
        deeper = max(0, failed - max(1, relief._min_release_rows()))
        self.assertFalse(
            relief._declines_target(deeper),
            "a target below the failed one by a release granularity must re-arm",
        )

    def test_a_shallower_ask_is_the_same_question(self):
        relief, pool, _card = self._rig()
        relief.free_up_to(600 * MIB)
        failed = relief._exhausted_target_rows
        self.assertTrue(relief._declines_target(failed + 1000))

    def test_the_level_alone_can_never_unstick_it(self):
        """Why the level test is not sufficient on its own.

        The metal case is an arena with nothing to release: the shrink clears
        no extent, so the PHYSICAL level does not move either. A marker keyed
        to that level then marks the level the rung is stuck at, for ever.
        Modelled faithfully here -- the ordinary `_FlipPool` models the
        retain-handles variant instead, where the backing moves and only the
        driver's column does not."""

        class _NothingToRelease(_FlipPool):
            def runtime_set_backing_rows(self, rows):
                self.calls.append(int(rows))
                self.size = int(rows)  # the dial's assertion...
                return 0  # ...but the arena cleared no extent

        card = _Card(1100)
        pool = _NothingToRelease(CONFIGURED_ROWS, card=None)
        relief = kbr.KvBackingRelief(
            pool,
            _FakeAllocator(CONFIGURED_ROWS),
            live_slots_fn=lambda: torch.arange(1, LIVE_ROWS + 1, dtype=torch.int64),
            bytes_per_row=BYTES_PER_ROW,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
        )
        before = relief.backed_rows()
        relief.free_up_to(600 * MIB)
        self.assertEqual(
            relief.backed_rows(), before, "a zero-byte shrink moves no pages"
        )
        self.assertTrue(relief._exhausted, "so the level-keyed marker still holds")
        # ...and only the target escape can answer a deeper ask.
        failed = relief._exhausted_target_rows
        deeper = max(0, failed - max(1, relief._min_release_rows()))
        self.assertFalse(relief._declines_target(deeper))
