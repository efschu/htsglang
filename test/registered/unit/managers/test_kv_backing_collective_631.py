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
"""#656 item 12: A REFUSAL MAY BE DECIDED LOCALLY. A CAPACITY MAY NOT.

This is the law that took the KV rung off by default, and it was learned on
metal (HANDOFF_675 §1a). The device half worked -- 1840 MiB reclaimed from a
card sitting at 4 MiB free -- and then the group wedged: ``/health`` reported
"couldn't get a response from detokenizer" while every rank was alive and
logging normally. The ranks had capped to 449039 / 451037 / 175225 / 145734
rows.

Nothing about those bytes was wrong. What was wrong is that each rank sized
its own shrink from its own free memory and its own live set, and the cap
changes ``available_size()``, which feeds ADMISSION. A PP group whose ranks
disagree about how much work the group may take desyncs.

So the shrink target is now decided ONCE, by an element-wise MIN reduction
over a proposal every rank makes, at a point every rank executes
unconditionally. These tests pin the three properties that make that safe:

1. **Uniformity** -- every rank ends at the same absolute row target, or none
   of them shrinks. There is no third outcome.
2. **Safety** -- the agreed target is never below ANY rank's highest live row.
   The most-pressed rank sets the ambition; the most-loaded rank sets the
   limit; the limit wins.
3. **Veto by construction** -- a rank that cannot participate (no relief
   object, an unreadable live set) makes the group decline, rather than being
   skipped while its peers cap.

Hermetic: no CUDA, no distributed. The reduction is a plain element-wise MIN
over the proposals of a simulated fleet, which is exactly the contract
``default_collective_min`` implements.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as kbr

MIB = 1024 * 1024


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


class _FakePool:
    def __init__(self, rows: int, bytes_per_row: int = 4096, *, card=None):
        self.size = rows
        self.page_size = 1
        self._bytes_per_row = bytes_per_row
        self._card = card
        self.supports_backing_spans = True
        self.backing_commit_chunk_bytes = 0
        self.calls = []

    def runtime_set_backing_rows(self, rows: int) -> int:
        rows = int(rows)
        self.calls.append(rows)
        delta = self.size - rows
        self.size = rows
        released = max(0, delta) * self._bytes_per_row
        if self._card is not None:
            self._card.free += released
        return released


class _Card:
    def __init__(self, free_mib):
        self.free = free_mib * MIB

    def probe(self):
        return self.free


def _rank(rows=500000, live=(), free_mib=8000, bytes_per_row=4096, buffers=0):
    """One rank's worth of the rung: pool, allocator, card, relief."""
    card = _Card(free_mib)
    pool = _FakePool(rows, bytes_per_row, card=card)
    alloc = _FakeAllocator(rows)
    relief = kbr.KvBackingRelief(
        pool,
        alloc,
        live_slots_fn=lambda: torch.tensor(list(live), dtype=torch.int64),
        bytes_per_row=bytes_per_row,
        probe=card.probe,
        buffers=buffers,
    )
    return relief, pool, alloc, card


def _reduce(proposals):
    """Element-wise MIN, i.e. the contract of ``default_collective_min``."""
    return [min(vals) for vals in zip(*proposals)]


def _agree(reliefs, *, want_mib=500, floor_mib=1024, delta_mib=256):
    """Run one full group decision and return (target, per-rank proposals)."""
    proposals = [
        r.propose(
            want_bytes=want_mib * MIB,
            floor_bytes=floor_mib * MIB,
            delta_bytes=delta_mib * MIB,
        )
        for r in reliefs
    ]
    return kbr.collective_kv_target(_reduce(proposals)), proposals


class TheTargetIsOneNumberForTheWholeGroupTest(unittest.TestCase):
    """Property 1: uniformity. The metal failure was four different numbers."""

    def test_the_pressed_rank_sets_the_target_and_every_rank_takes_it(self):
        # Rank 0 is short of memory; its peers are comfortable. Under the old
        # rank-local rung only rank 0 capped, and the group desynced.
        pressed, _p0, _a0, _c0 = _rank(free_mib=1100, live=(1000,))
        easy_a, _p1, _a1, _c1 = _rank(free_mib=9000, live=(1000,))
        easy_b, _p2, _a2, _c2 = _rank(free_mib=9000, live=(1000,))

        target, proposals = _agree([pressed, easy_a, easy_b])

        self.assertIsNotNone(target, "the pressed rank should have driven a shrink")
        # The unpressed peers proposed no shrink at all...
        self.assertEqual(
            proposals[1][0],
            kbr._SHRINK_SCALE,  # #796: "no change" is now the NEUTRAL element of the
            # MIN (a proportion of 1.0), not this rank's own row count. Encoding
            # it as `current` is what let the smallest pool in an uneven fleet
            # silently set the group's ambition.
        )
        self.assertEqual(proposals[2][0], kbr._SHRINK_SCALE)
        # ...and still land on the pressed rank's number, which is the point.
        self.assertLess(target, 500000)
        # #796: the group's decision is a PROPORTION and ``target`` is that
        # proportion converted against this pool's 500000 rows, so the two are
        # compared in one unit. The law is unchanged -- the pressed rank's
        # number is the group's -- only its currency is.
        self.assertEqual(target, kbr._rows_for_ppm(proposals[0][0], 500000))

    def test_applying_the_agreed_target_leaves_every_rank_at_the_same_rows(self):
        reliefs = [
            _rank(free_mib=1100, live=(1000,))[0],
            _rank(free_mib=9000, live=(1000,))[0],
            _rank(free_mib=9000, live=(1000,))[0],
        ]
        target, _ = _agree(reliefs)
        for r in reliefs:
            r.apply_target(target)
        sizes = {int(r._pool.size) for r in reliefs}
        self.assertEqual(sizes, {target}, "ranks must not disagree about capacity")
        caps = {r._cap.cap for r in reliefs}
        self.assertEqual(caps, {target}, "the admission cap must be identical too")

    def test_no_pressure_anywhere_means_nobody_shrinks(self):
        reliefs = [_rank(free_mib=9000, live=(1000,))[0] for _ in range(3)]
        target, _ = _agree(reliefs)
        self.assertIsNone(target)


class TheLimitWinsOverTheAmbitionTest(unittest.TestCase):
    """Property 2: safety. A shared target must clear EVERY rank's live set."""

    def test_the_target_never_drops_below_a_peers_highest_live_row(self):
        # Rank 0 is desperate and would like to cap very low. Rank 2 has a live
        # row at 460000. Unmapping under it is cudaErrorIllegalAddress, which
        # kills every rank rather than raising.
        pressed, _, _, _ = _rank(free_mib=1030, live=(1000,))
        loaded, _, _, _ = _rank(free_mib=9000, live=(460000,))
        idle, _, _, _ = _rank(free_mib=9000, live=(1000,))

        target, proposals = _agree([pressed, loaded, idle], want_mib=4000)

        self.assertIsNotNone(target)
        self.assertLess(proposals[0][0], 460000, "the ambition should be lower")
        self.assertGreater(target, 460000, "but the limit must win")

    def test_a_loaded_peer_can_cancel_the_shrink_entirely(self):
        pressed, _, _, _ = _rank(free_mib=1030, live=(1000,))
        full, _, _, _ = _rank(free_mib=9000, live=(499999,))
        target, _ = _agree([pressed, full], want_mib=4000)
        self.assertIsNone(target, "there is no row range free on every rank")


class ARankThatCannotJoinVetoesTest(unittest.TestCase):
    """Property 3: the failure mode must be 'nobody caps', never 'some cap'."""

    def test_a_rank_with_no_relief_object_stops_the_group(self):
        pressed, _, _, _ = _rank(free_mib=1100, live=(1000,))
        proposals = [
            pressed.propose(
                want_bytes=500 * MIB, floor_bytes=1024 * MIB, delta_bytes=256 * MIB
            ),
            kbr.ABSTAIN,
        ]
        self.assertIsNone(kbr.collective_kv_target(_reduce(proposals)))

    def test_an_unreadable_live_set_abstains_rather_than_guessing(self):
        card = _Card(1100)
        pool = _FakePool(500000, card=card)

        def _boom():
            raise RuntimeError("live-set probe is down")

        relief = kbr.KvBackingRelief(
            pool,
            _FakeAllocator(500000),
            live_slots_fn=_boom,
            bytes_per_row=4096,
            probe=card.probe,
        )
        easy, _, _, _ = _rank(free_mib=9000, live=(1000,))
        target, _ = _agree([relief, easy])
        self.assertIsNone(target)
        self.assertEqual(pool.calls, [], "no rank may shrink on an abstention")

    def test_abstain_is_neutral_when_read_as_a_proposal(self):
        # The abstention has to survive the MIN reduction as an abstention: if
        # it reduced to something the group could act on, a rank that cannot
        # cap would be carried along by its peers.
        self.assertIsNone(kbr.collective_kv_target(list(kbr.ABSTAIN)))


class AFutileRankDoesNotStopAWorkingOneTest(unittest.TestCase):
    """Exhaustion is about one arena, and it must not be contagious.

    A rank whose arena cannot release partially (§1d: the release granularity
    is one commit chunk in EVERY buffer) reports zero bytes. That is evidence
    about ITS arena, so it stops ASKING -- but it still honours the group's
    target, because leaving it uncapped is exactly the admission disagreement
    this whole mechanism exists to prevent.
    """

    def test_an_exhausted_rank_asks_for_nothing_but_still_obeys(self):
        futile, futile_pool, _, _ = _rank(free_mib=1100, live=(1000,))
        futile._exhausted = True
        pressed, _, _, _ = _rank(free_mib=1100, live=(1000,))

        target, proposals = _agree([futile, pressed])

        self.assertEqual(
            proposals[0][0],
            kbr._SHRINK_SCALE,
            "an exhausted rank stops asking (#796: as the neutral proportion)",
        )
        self.assertIsNotNone(target)
        futile.apply_target(target)
        self.assertEqual(int(futile_pool.size), target)

    def test_a_group_where_everyone_is_exhausted_stops_by_itself(self):
        reliefs = []
        for _ in range(3):
            r, _, _, _ = _rank(free_mib=1100, live=(1000,))
            r._exhausted = True
            reliefs.append(r)
        target, _ = _agree(reliefs)
        self.assertIsNone(target, "no veto flag needed: the ambition is gone")


class TheProposalAccountsForCheaperReliefFirstTest(unittest.TestCase):
    """Tier order survives the move out of the guard's ladder.

    The rung now runs BEFORE the corridor guard rather than inside its ladder,
    so it has to respect the tier law itself: torch's allocator cache is free
    money and must be counted as available before KV capacity is spent. The
    estimate is the hoard size, which OVERSTATES what ``empty_cache`` returns
    -- deliberately, because overstating cheap relief understates the KV ask,
    and under-shrinking is recoverable (the guard refuses and the flip is
    retried) while over-shrinking costs admission capacity for nothing.
    """

    def test_cached_bytes_reduce_the_ask(self):
        greedy, _, _, _ = _rank(free_mib=1100, live=(1000,))
        modest, _, _, _ = _rank(free_mib=1100, live=(1000,))
        bare = greedy.propose(
            want_bytes=500 * MIB, floor_bytes=1024 * MIB, delta_bytes=256 * MIB
        )
        with_cache = modest.propose(
            want_bytes=500 * MIB,
            floor_bytes=1024 * MIB,
            delta_bytes=256 * MIB,
            cheap_relief_bytes=400 * MIB,
        )
        self.assertGreater(
            with_cache[0], bare[0], "cheap relief must shrink the KV ask"
        )

    def test_cheap_relief_that_covers_the_deficit_cancels_the_ask(self):
        r, _, _, _ = _rank(free_mib=1100, live=(1000,))
        proposal = r.propose(
            want_bytes=500 * MIB,
            floor_bytes=1024 * MIB,
            delta_bytes=256 * MIB,
            cheap_relief_bytes=4000 * MIB,
        )
        self.assertEqual(proposal[0], kbr._SHRINK_SCALE)


class RecoveryIsStillTheThingThatMakesThisNotASmallerPoolTest(unittest.TestCase):
    def test_every_rank_returns_to_its_boot_rows(self):
        reliefs = [_rank(free_mib=1100, live=(1000,))[0] for _ in range(3)]
        target, _ = _agree(reliefs)
        for r in reliefs:
            r.apply_target(target)
        for r in reliefs:
            r.recover()
        for r in reliefs:
            self.assertEqual(int(r._pool.size), 500000)
            self.assertFalse(r._cap.engaged)


class _LazyPool(_FakePool):
    """A pool whose ``size`` is a RESERVATION and whose backing is separate.

    This is the real shape and the one the first metal boot fell over. On the
    hybrid VMM pool ``size`` keeps "stock rows semantics" and never moves --
    ``initial_backing_rows`` says so in as many words -- while the physically
    committed span lives in ``full_pool_backed_rows``. Reading ``size`` as the
    current backing therefore over-reads it by everything already released,
    and the next "shrink" asks for MORE rows than are committed, which is a
    GROW: ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` on a card the rung
    had just been asked to relieve.
    """

    def __init__(self, rows: int, bytes_per_row: int = 4096, *, card=None):
        super().__init__(rows, bytes_per_row, card=card)
        self._backed = rows
        self.grew_to = []

    @property
    def full_pool_backed_rows(self) -> int:
        return self._backed

    def runtime_set_backing_rows(self, rows: int) -> int:
        """Converges the backing in BOTH directions, like the real one.

        That symmetry is the trap: an argument above the committed span is not
        a smaller shrink, it is a commit, and it takes memory off the card. The
        fake therefore charges the card for a grow and raises the driver's own
        error when the card cannot pay -- so a caller that mistakes the
        reservation for the backing fails here the way it failed on metal.
        """
        rows = int(rows)
        self.calls.append(rows)
        if rows > self._backed:
            self.grew_to.append(rows)
            need = (rows - self._backed) * self._bytes_per_row
            if self._card is not None:
                if need > self._card.free:
                    raise RuntimeError("cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY")
                self._card.free -= need
            self._backed = rows
            return 0
        released = (self._backed - rows) * self._bytes_per_row
        self._backed = rows
        if self._card is not None:
            self._card.free += released
        return released


class TheCurrentSpanIsWhatIsBackedNotWhatIsReservedTest(unittest.TestCase):
    """Metal, 2026-08-11: the second shrink of a boot tried to GROW."""

    def _lazy_rank(self, free_mib=1100, live=(80,)):
        card = _Card(free_mib)
        pool = _LazyPool(500000, card=card)
        relief = kbr.KvBackingRelief(
            pool,
            _FakeAllocator(500000),
            live_slots_fn=lambda: torch.tensor(list(live), dtype=torch.int64),
            bytes_per_row=4096,
            probe=card.probe,
        )
        return relief, pool, card

    def test_a_second_shrink_never_asks_above_the_committed_span(self):
        relief, pool, card = self._lazy_rank()
        first, _ = _agree([relief], want_mib=500)
        relief.apply_target(first)
        self.assertLess(pool.full_pool_backed_rows, 500000)
        # The card is tighter again; the group agrees a second target. It must
        # be read against what is BACKED now, not against the reservation.
        card.free = 1100 * MIB
        second, proposals = _agree([relief], want_mib=500)
        self.assertEqual(
            proposals[0][2],
            pool.full_pool_backed_rows,
            "the proposal's 'current' must be the committed span",
        )
        if second is not None:
            relief.apply_target(second)
        self.assertEqual(pool.grew_to, [], "a shrink must never commit pages")

    def test_a_target_above_the_committed_span_is_a_no_op_not_a_grow(self):
        relief, pool, _card = self._lazy_rank()
        relief.apply_target(300000)
        self.assertEqual(pool.full_pool_backed_rows, 300000)
        # A peer's ambition, agreed while this rank was already lower.
        self.assertEqual(relief.apply_target(400000), 0)
        self.assertEqual(pool.grew_to, [])


class RecoveryMayNotBreachTheCorridorItWasRelievingTest(unittest.TestCase):
    """Metal, 2026-08-11: recovery re-committed the pool and left 6 MiB free.

    ``recover`` grew straight back to the boot reservation with no reference
    to the corridor law, on a card whose relief had been the reason for the
    shrink. Measured: ``free 6 -> 292 MiB`` on rank 1, and a ``cuMemCreate``
    OOM on the way. Undo is allocation, and here it ran at an idle boundary
    that was not as idle as the design assumed.

    A partial recovery is a CAPACITY LOSS and a full one that breaches is a
    FAULT, so the rung takes the loss: it recovers as far as the law permits,
    keeps the cap at that level, and remembers the boot rows for later.
    """

    def _rank_at(self, free_mib, backed):
        card = _Card(free_mib)
        pool = _LazyPool(500000, card=card)
        pool._backed = backed
        pool.size = 500000
        relief = kbr.KvBackingRelief(
            pool,
            _FakeAllocator(500000),
            live_slots_fn=lambda: torch.tensor([80], dtype=torch.int64),
            bytes_per_row=MIB,  # 1 MiB per row keeps the arithmetic readable
            probe=card.probe,
        )
        relief._rows_at_boot = 500000
        relief._cap.engage(backed)
        return relief, pool, card

    def test_recovery_stops_at_the_corridor_law(self):
        # 3000 MiB free, law floor 1024 -> at most 1976 rows may be committed.
        relief, pool, _card = self._rank_at(3000, 400000)
        relief.recover()
        self.assertLessEqual(pool.full_pool_backed_rows, 400000 + 1976)
        self.assertGreater(pool.full_pool_backed_rows, 400000)
        self.assertNotIn(500000, pool.grew_to, "the full grow must never be attempted")

    def test_a_partial_recovery_keeps_the_cap_at_the_level_it_reached(self):
        relief, pool, _card = self._rank_at(3000, 400000)
        relief.recover()
        self.assertTrue(relief._cap.engaged, "uncapped rows above the backing")
        self.assertEqual(relief._cap.cap, pool.full_pool_backed_rows)

    def test_a_partial_recovery_remembers_the_boot_rows_for_later(self):
        relief, pool, card = self._rank_at(3000, 400000)
        relief.recover()
        self.assertIsNotNone(relief._rows_at_boot)
        card.free = 200_000 * MIB  # the pressure is gone
        relief.recover()
        self.assertEqual(pool.full_pool_backed_rows, 500000)
        self.assertFalse(relief._cap.engaged)
        self.assertIsNone(relief._rows_at_boot)

    def test_a_card_with_no_headroom_recovers_nothing_and_stays_capped(self):
        relief, pool, _card = self._rank_at(900, 400000)
        self.assertEqual(relief.recover(), 0)
        self.assertEqual(pool.full_pool_backed_rows, 400000)
        self.assertTrue(relief._cap.engaged)


class ExhaustionIsOnlyEvidenceWhenTheAskCouldHavePaidTest(unittest.TestCase):
    """A group target shallower than this rank's granularity proves nothing.

    Release is one commit chunk in EVERY buffer, and the three PP stages here
    hold 28 / 20 / 16 of them, so the ranks do not share a granularity. Once
    the target became collective a rank could be handed a shrink smaller than
    its own smallest possible release. It returns zero — correctly — and the
    old rule read that as "this arena cannot pay" and stopped it asking for
    the rest of the phase. That is a voice with real bytes silenced by a
    number it did not choose, and nothing in the logs would look wrong.
    """

    def _coarse_rank(self):
        """Granularity 1000 rows: one 4 MiB chunk in each of 1000 buffers."""
        card = _Card(1100)
        pool = _FakePool(500000, 4096, card=card)
        pool.backing_commit_chunk_bytes = 4096
        relief = kbr.KvBackingRelief(
            pool,
            _FakeAllocator(500000),
            live_slots_fn=lambda: torch.tensor([80], dtype=torch.int64),
            bytes_per_row=4096,
            probe=card.probe,
            buffers=1000,
        )
        return relief, pool, card

    def test_a_sub_granularity_target_does_not_exhaust_the_rank(self):
        relief, pool, card = self._coarse_rank()
        self.assertEqual(relief._min_release_rows(), 1000)
        # A shallower ask than one release quantum, and the card does not move.
        pool._card = None  # the fake stops crediting the card: zero measured
        relief.apply_target(500000 - 10)
        self.assertFalse(
            relief._exhausted,
            "a target smaller than this rank's quantum is not evidence",
        )

    def test_a_full_size_ask_that_pays_nothing_still_exhausts(self):
        relief, pool, card = self._coarse_rank()
        pool._card = None
        relief.apply_target(500000 - 5000)
        self.assertTrue(
            relief._exhausted,
            "an ask well above the quantum that returns nothing IS evidence",
        )


class _Sched:
    pass


class _Guard:
    floor_bytes = 1024 * MIB
    delta_bytes = 256 * MIB
    device_index = 0


class TheRungShrinksOnOneLegOnlyTest(unittest.TestCase):
    """pp->tp gives up backing that is about to hold nothing. tp->pp does not.

    The scheduler's KV pool is the PP LAYOUT's pool. Capping it as the
    instance leaves PP costs nothing real for the whole TP phase, and pp->tp
    is also the leg whose refusal is fatal under strict purity. Capping it as
    the instance RE-ENTERS PP is churn that ``recover_kv_backing`` undoes
    within the same flip -- and the undo is a ``cuMemCreate``, the one
    operation this chain has already paid 2.5 GiB to learn not to do near a
    tight card.

    The abstention must still ENTER the reduction, which is what the call
    count asserts.
    """

    def _sched_with_relief(self):
        from sglang.srt.managers import phase_flip_spill as pfs

        relief, pool, _alloc, _card = _rank(free_mib=1100, live=(1000,))
        sched = _Sched()
        setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, relief)
        return sched, pool

    def _run(self, direction):
        from sglang.srt.managers import phase_flip_spill as pfs

        sched, pool = self._sched_with_relief()
        calls = []

        def reduce(vals, **kw):
            calls.append(list(vals))
            return list(vals)

        freed = pfs.collective_kv_backing_relief(
            sched,
            reduce,
            want_bytes=500 * MIB,
            guard=_Guard(),
            direction=direction,
        )
        return freed, pool, calls

    def test_pp_to_tp_shrinks(self):
        freed, pool, calls = self._run("pp_to_tp")
        self.assertGreater(freed, 0)
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(len(calls), 1)

    def test_tp_to_pp_is_funded_too_662(self):
        """#662: the leg into the PREFILL layout must have a funder.

        This pin was inverted on 2026-08-15 and the reason is a measured
        production failure, not a preference. With the rung excluded here,
        every tp_to_pp arm on the max-KV vector abandoned with

            seam entry margin short: want 2194 MiB, free 2892 -> 3098 MiB,
            reclaimed 206 MiB from [allocator-cache]

        -- 206 MiB of torch cache against a 2194 MiB ask, because
        allocator-cache was the only tier left that pays. Eight abandons
        install the "seam unfundable" guard and the instance NEVER ENTERS
        THE PREFILL LAYOUT AGAIN, so long prompts are prefilled at TP speed
        forever. That is the reported defect.

        The old exclusion was right that the shrink is undone by the
        post-cutover recovery, and wrong that this makes it pointless: the
        undo is the price of the flip happening at all.
        """
        freed, pool, calls = self._run("tp_to_pp")
        self.assertGreater(freed, 0, "the tp_to_pp leg must be fundable")
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(len(calls), 1)

    def test_tp_to_pp_abstains_when_funding_is_disabled(self):
        """The shipped one-leg behaviour, kept as a VALUE of the same term.

        SGLANG_SEAM_FUND_TP_TO_PP=0 restores it exactly, which is what makes
        the metal comparison a one-variable experiment.
        """
        import os

        from sglang.srt.managers import phase_flip_spill as pfs

        prev = os.environ.get(pfs.ENV_FUND_TP_TO_PP)
        os.environ[pfs.ENV_FUND_TP_TO_PP] = "0"
        try:
            freed, pool, calls = self._run("tp_to_pp")
        finally:
            if prev is None:
                os.environ.pop(pfs.ENV_FUND_TP_TO_PP, None)
            else:
                os.environ[pfs.ENV_FUND_TP_TO_PP] = prev
        self.assertEqual(freed, 0)
        self.assertEqual(pool.calls, [], "the pool that is about to go live")
        self.assertEqual(len(calls), 1, "abstaining is not the same as walking away")


if __name__ == "__main__":
    unittest.main(verbosity=2)
