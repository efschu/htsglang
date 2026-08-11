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
        self.assertEqual(proposals[1][0], 500000)
        self.assertEqual(proposals[2][0], 500000)
        # ...and still land on the pressed rank's number, which is the point.
        self.assertLess(target, 500000)
        self.assertEqual(target, proposals[0][0])

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

        self.assertEqual(proposals[0][0], 500000, "an exhausted rank stops asking")
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
        self.assertEqual(proposal[0], 500000)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
