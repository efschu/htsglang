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
"""#656 C22 root cause: THE SHRINK IS COLLECTIVE, THE RECOVERY WAS NOT.

Measured on metal, boot ``boot_m1`` 2026-08-13, and this file exists because
the number is exact rather than suggestive.

The flip's pre-move frame ballot abandoned four consecutive ``pp_to_tp``
rounds because rank 1 framed a different payload from its peers. The pool
census at the same instant::

    PP0  size=578390 free=309574 cached=268816 unaccounted=0
    PP2  size=578390 free=309574 cached=268816 unaccounted=0
    PP1  size=578390 free=269170 cached=268816 unaccounted=40404

``309574 - 269170 = 40404``, exactly, with every rank agreeing on the
resident request and on the tree. Rank 1's allocator was holding 40404 rows
that neither the tree nor any request accounted for, so its live-slot
enumeration -- and therefore the length of the payload it framed -- differed
from its peers'.

**The 40404 rows have a name and it was in the log.** Two lines in the whole
33-minute boot, both on PP1, and both at the exact instant a divergence
opened::

    14:04:23 PP1  KV-BACKING recovered to 544077 of 578390 rows (corridor-bounded)
    14:42:25 PP1  KV-BACKING recovered to 537986 of 578390 rows (corridor-bounded)

``578390 - 544077 = 34313``; ``578390 - 537986 = 40404``. Those are the two
divergence deltas the census reported, to the row.

So the mechanism is not a leak, an eviction race, or a spill sentinel. The
KV backing rung shrinks the pool to fund the seam and re-grows it on the way
back, and:

* the SHRINK target is decided ONCE for the group, by a MIN reduction over
  every rank's proposal (``collective_kv_target``, #656 item 12) -- which is
  why every ``KV-BACKING released`` line in the boot shows an identical row
  count on all three ranks;
* the RECOVERY was decided by each rank ALONE, bounded by its own distance
  from the corridor law (``KvBackingRelief.recover``). The rank nearest the
  law -- rank 1, the binding card on this rig -- recovers less than its peers
  and stays capped where they do not.

That is precisely the failure the collective shrink was introduced to end,
reached from the other direction: *a refusal may be decided locally, a
CAPACITY may not*. The bound itself is right and stays (recovery is an
allocation and it may not breach the corridor it was relieving). What was
missing is that the group must then live at the level the poorest rank can
afford, because under pure PP every rank holds the same token rows and the
group can never use more of them than its minimum anyway.

Hermetic: no CUDA, no distributed. The reduction is a plain element-wise MIN
over a simulated fleet, which is the contract ``default_collective_min``
implements.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as kbr

MIB = 1024 * 1024

#: The metal boot's numbers, used verbatim so a reader can join this file to
#: the log line it was written from.
BOOT_ROWS = 578390
PP1_RECOVERED = 537986
DIVERGENCE = BOOT_ROWS - PP1_RECOVERED  # 40404


class _Card:
    def __init__(self, free_mib):
        self.free = int(free_mib) * MIB

    def probe(self):
        return self.free


class _Alloc:
    def __init__(self, size: int):
        self.size = size
        self.page_size = 1
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self._listeners = []

    def register_free_listener(self, on_free, on_clear=None):
        self._listeners.append((on_free, on_clear))

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)


class _Pool:
    """Reservation and backing are separate, as on the real VMM pool."""

    def __init__(self, rows: int, bytes_per_row: int, card: _Card):
        self.size = rows
        self.page_size = 1
        self._bytes_per_row = bytes_per_row
        self._card = card
        self._backed = rows
        self.supports_backing_spans = True
        self.backing_commit_chunk_bytes = 0
        self.calls = []

    @property
    def full_pool_backed_rows(self) -> int:
        return self._backed

    def runtime_set_backing_rows(self, rows: int) -> int:
        rows = int(rows)
        self.calls.append(rows)
        if rows > self._backed:
            need = (rows - self._backed) * self._bytes_per_row
            if need > self._card.free:
                raise RuntimeError("cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY")
            self._card.free -= need
            self._backed = rows
            return 0
        released = (self._backed - rows) * self._bytes_per_row
        self._backed = rows
        self._card.free += released
        return released


#: 1 MiB per row keeps every number in this file readable as MiB.
ROW_BYTES = MIB


def _rank(*, free_mib, backed=BOOT_ROWS, live=(80,), rows=BOOT_ROWS):
    card = _Card(free_mib)
    pool = _Pool(rows, ROW_BYTES, card)
    pool._backed = backed
    relief = kbr.KvBackingRelief(
        pool,
        _Alloc(rows),
        live_slots_fn=lambda: torch.tensor(list(live), dtype=torch.int64),
        bytes_per_row=ROW_BYTES,
        probe=card.probe,
        law_floor_bytes=1024 * MIB,
        admission_reserve_rows=512,
    )
    return relief, pool, card


def _reduce(proposals):
    """Element-wise MIN -- the contract of ``default_collective_min``."""
    return [min(vals) for vals in zip(*proposals)]


def _agree(reliefs):
    """One group cap decision: propose, reduce, decide."""
    return kbr.collective_cap_target(_reduce([r.cap_proposal() for r in reliefs]))


def _reconcile(reliefs):
    target = _agree(reliefs)
    if target is not None:
        for r in reliefs:
            r.reconcile_to(target)
    return target


class TheMetalDivergenceIsReproducedAndClosedTest(unittest.TestCase):
    """The 40404 rows, built from the log's own numbers and then agreed away."""

    def _fleet(self):
        """Three ranks after a shrink, on the way back through recovery.

        Rank 1 is the binding card: 1024 MiB law and free memory that leaves
        it able to re-commit only as far as 537986 rows, which is what the
        metal line says it reached. Its peers can afford the whole way back.
        """
        shrunk = 510710  # the boot's own pre-cutover backing (67680 withheld)
        fleet = []
        # 1 MiB per row, so "free MiB above the law" IS "rows it may commit".
        # The peers can afford the whole 67680 rows back; rank 1 can afford
        # exactly the 27276 that take it to the level the metal line reports.
        for free_mib in (
            1024 + 90_000,
            1024 + (PP1_RECOVERED - shrunk),
            1024 + 90_000,
        ):
            relief, pool, card = _rank(free_mib=free_mib, backed=shrunk)
            relief._rows_at_boot = BOOT_ROWS
            relief._cap.engage(shrunk)
            fleet.append((relief, pool, card))
        return fleet

    def test_unilateral_recovery_reproduces_the_40404_row_divergence(self):
        """The defect, stated as a test: this is what the metal boot did."""
        fleet = self._fleet()
        for relief, _pool, _card in fleet:
            relief.recover()
        exposed = [r.exposed_rows() for r, _, _ in fleet]
        self.assertEqual(exposed[0], BOOT_ROWS)
        self.assertEqual(exposed[2], BOOT_ROWS)
        self.assertEqual(exposed[1], PP1_RECOVERED)
        self.assertEqual(
            exposed[0] - exposed[1],
            DIVERGENCE,
            "the reproduction must land on the metal number exactly",
        )

    def test_the_group_ends_at_one_row_level_after_reconciliation(self):
        fleet = self._fleet()
        for relief, _pool, _card in fleet:
            relief.recover()
        target = _reconcile([r for r, _, _ in fleet])
        self.assertEqual(target, PP1_RECOVERED)
        self.assertEqual(
            {r.exposed_rows() for r, _, _ in fleet},
            {PP1_RECOVERED},
            "every rank must expose the same absolute row count, or the "
            "flip's live-slot enumeration cannot agree",
        )

    def test_no_rank_exposes_a_row_it_has_not_backed(self):
        """The invariant the cap exists for, checked on every rank."""
        fleet = self._fleet()
        for relief, _pool, _card in fleet:
            relief.recover()
        _reconcile([r for r, _, _ in fleet])
        for i, (relief, pool, _card) in enumerate(fleet):
            self.assertLessEqual(
                relief.exposed_rows(),
                pool.full_pool_backed_rows,
                f"rank {i} would hand out an unmapped row",
            )

    def test_the_healthy_ranks_keep_their_pages_and_only_withhold_ids(self):
        """Levelling down is an ID decision, never a cuMemUnmap.

        The bytes stay committed: releasing them here would be a second,
        unpriced shrink at a point the corridor gate has already reasoned
        about, and re-committing them is the one operation this chain has
        paid to learn not to do near a tight card.
        """
        fleet = self._fleet()
        for relief, _pool, _card in fleet:
            relief.recover()
        backed_before = [p.full_pool_backed_rows for _r, p, _c in fleet]
        _reconcile([r for r, _, _ in fleet])
        self.assertEqual(
            [p.full_pool_backed_rows for _r, p, _c in fleet], backed_before
        )

    def test_when_the_pressure_lifts_the_group_recovers_together(self):
        """Agreement must not be a ratchet: it goes up as well as down."""
        fleet = self._fleet()
        for relief, _pool, _card in fleet:
            relief.recover()
        _reconcile([r for r, _, _ in fleet])
        # rank 1's card frees up; its own recovery now reaches the boot rows
        fleet[1][2].free = 200_000 * MIB
        for relief, _pool, _card in fleet:
            relief.recover()
        target = _reconcile([r for r, _, _ in fleet])
        self.assertEqual(target, BOOT_ROWS)
        self.assertEqual({r.exposed_rows() for r, _, _ in fleet}, {BOOT_ROWS})
        for relief, _pool, _card in fleet:
            self.assertFalse(
                relief._cap.engaged, "a group back at boot rows holds no cap"
            )


class TheAgreementNeverCommitsAPageTest(unittest.TestCase):
    """Metal, 2026-08-13 15:40:23Z, on the first boot of this mechanism.

    The proposal was ``backed + (free - law) / bytes_per_row`` -- what
    ``recover`` would be ALLOWED to commit -- so on the pp->tp leg the
    levelling tried to grow the pool straight back to the boot rows, handing
    back the very rows the collective shrink had just taken to fund the seam::

        KV-BACKING cap agreement could not commit to the agreed level 579130
          (reached 558837): cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY   x3
        [#631 seam-census] CORRIDOR LAW BROKEN during tp_to_pp rank 0 at
          stage 'entry': free 3 MiB is below the 1024 MiB floor
        CORRIDOR-GUARD REFUSED ... free 4 -> 4 MiB, reclaimed 0 MiB
        PHASE-FLIP FLIP ABANDONED (pool too small for the live set)

    Three minutes into the boot the instance was parked in TP with a 9-token
    prefill it could not run. Growing has exactly ONE owner and this is not
    it.
    """

    def test_reconcile_never_touches_the_backing(self):
        relief, pool, _card = _rank(free_mib=1024 + 90_000, backed=400_000)
        relief._rows_at_boot = BOOT_ROWS
        relief._cap.engage(400_000)
        calls = len(pool.calls)
        relief.reconcile_to(BOOT_ROWS)  # a target above what is backed
        self.assertEqual(len(pool.calls), calls, "the agreement committed pages")
        self.assertEqual(pool.full_pool_backed_rows, 400_000)
        self.assertLessEqual(relief.exposed_rows(), 400_000)

    def test_a_rank_proposes_what_it_has_backed_not_what_it_could_commit(self):
        """The card has room for the whole pool; the proposal must ignore it."""
        relief, _pool, _card = _rank(free_mib=1024 + 500_000, backed=400_000)
        relief._rows_at_boot = BOOT_ROWS
        relief._cap.engage(400_000)
        self.assertEqual(relief.cap_proposal()[0], 400_000)

    def test_levelling_up_is_a_cap_release_over_pages_already_held(self):
        """The level rises without an allocation, which is why it is safe.

        A rank levelled DOWN kept its pages, so when the poorest rank recovers
        and the group target rises, its peers follow by releasing a cap -- no
        commit anywhere on the path.
        """
        relief, pool, _card = _rank(free_mib=1024 + 90_000, backed=BOOT_ROWS)
        relief.reconcile_to(500_000)
        self.assertEqual(relief.exposed_rows(), 500_000)
        calls = len(pool.calls)
        relief.reconcile_to(BOOT_ROWS)
        self.assertEqual(relief.exposed_rows(), BOOT_ROWS)
        self.assertEqual(len(pool.calls), calls, "levelling up committed pages")


class TheLimitStillWinsOverTheAmbitionTest(unittest.TestCase):
    """A levelled-down rank may never be capped below its own live set."""

    def test_the_target_clears_every_ranks_live_high_water(self):
        a, _p, _c = _rank(free_mib=1024 + 90_000, backed=BOOT_ROWS, live=(80,))
        # A rank whose resident set reaches far up the pool, and a poorer rank
        # that sets the level -- but may not set it below that live row.
        b, _p, _c = _rank(free_mib=1024 + 90_000, backed=BOOT_ROWS, live=(400_000,))
        poor, _p, _c = _rank(free_mib=1024, backed=450_000, live=(80,))
        poor._rows_at_boot = BOOT_ROWS
        poor._cap.engage(450_000)
        target = _agree([a, b, poor])
        self.assertIsNotNone(target)
        self.assertEqual(target, 450_000, "the poorest rank sets the level")
        self.assertGreater(
            target,
            400_000,
            "a target below a peer's live row is cudaErrorIllegalAddress, "
            "which kills every rank rather than raising",
        )

    def test_a_target_no_rank_can_back_is_declined_rather_than_forced(self):
        """When the floor is above what the poorest rank can commit.

        There is no honest level here: the poor rank cannot expose the rows
        its peers must keep. The group declines and leaves the divergence
        for the ballot to catch, which is a refused flip -- never a rank
        handing out memory it has not mapped.
        """
        loaded, _p, _c = _rank(free_mib=8000, backed=BOOT_ROWS, live=(400_000,))
        poor, _p, _c = _rank(free_mib=1024, backed=300_000, live=(80,))
        poor._rows_at_boot = BOOT_ROWS
        poor._cap.engage(300_000)
        target = _agree([loaded, poor])
        self.assertIsNone(target)


class ARankThatCannotJoinVetoesTest(unittest.TestCase):
    """One abstention cancels the decision, exactly as for the shrink."""

    def test_an_abstaining_rank_makes_the_group_decline(self):
        good, _p, _c = _rank(free_mib=8000, backed=BOOT_ROWS)
        self.assertIsNone(
            kbr.collective_cap_target(
                _reduce([good.cap_proposal(), kbr.CAP_ABSTAIN])
            )
        )

    def test_an_unreadable_live_set_abstains_rather_than_guessing(self):
        def boom():
            raise RuntimeError("live set unavailable")

        relief, _pool, _card = _rank(free_mib=8000)
        relief._live_slots_fn = boom
        self.assertEqual(tuple(relief.cap_proposal()), tuple(kbr.CAP_ABSTAIN))


class AnAgreedGroupIsLeftAloneTest(unittest.TestCase):
    """The common case must cost nothing: no cap churn on a healthy group."""

    def test_a_group_already_at_one_level_does_not_touch_its_allocators(self):
        fleet = [_rank(free_mib=8000, backed=BOOT_ROWS) for _ in range(3)]
        target = _agree([r for r, _, _ in fleet])
        self.assertEqual(target, BOOT_ROWS)
        for relief, pool, _card in fleet:
            calls = len(pool.calls)
            self.assertEqual(relief.reconcile_to(target), 0)
            self.assertEqual(len(pool.calls), calls, "no backing call")
            self.assertFalse(relief._cap.engaged)


class ItRidesTheExistingReductionAndAddsNoneTest(unittest.TestCase):
    """One gate call, one collective -- the count is itself an invariant.

    ``on_round``'s census diffs the collective COUNT across ranks to catch
    desyncs, and the gate's own tests pin one reduction per gate call. So the
    levelling widens the KV rung's existing payload from 4 fields to 8 rather
    than adding a reduction of its own.
    """

    def _fleet_and_channel(self):
        shrunk = 510710
        fleet = []
        for free_mib in (1024 + 90_000, 1024 + (PP1_RECOVERED - shrunk), 1024 + 90_000):
            relief, pool, card = _rank(free_mib=free_mib, backed=shrunk)
            relief._rows_at_boot = BOOT_ROWS
            relief._cap.engage(shrunk)
            relief.recover()
            fleet.append(relief)
        return fleet

    def test_one_reduction_carries_both_decisions_and_levels_the_group(self):
        from sglang.srt.managers import phase_flip_spill as spill

        fleet = self._fleet_and_channel()
        sent = []

        payloads = [None]

        def reduce_fn(vals, **_kw):
            sent.append(list(vals))
            return list(vals)

        # Each rank proposes into the same reduction; the MIN is taken over
        # all three payloads, exactly as ``default_collective_min`` does.
        proposals = []
        for relief in fleet:
            sched = type("S", (), {})()
            setattr(sched, spill.KV_BACKING_RELIEF_ATTR, relief)
            spill.collective_kv_backing_relief(
                sched,
                reduce_fn,
                want_bytes=0,
                guard=None,
                direction="tp_to_pp",
            )
            proposals.append(sent[-1])
        del payloads
        for p in proposals:
            self.assertEqual(len(p), 8, "4 shrink fields + 4 cap fields")
        self.assertEqual(len(sent), 3, "exactly one reduction per rank")

        # Now the real MIN across the group, applied on every rank.
        reduced = _reduce(proposals)
        for relief in fleet:
            sched = type("S", (), {})()
            setattr(sched, spill.KV_BACKING_RELIEF_ATTR, relief)
            spill.apply_cap_agreement(sched, reduced[4:8])
        self.assertEqual(
            {r.exposed_rows() for r in fleet},
            {PP1_RECOVERED},
            "the group must end level at the poorest rank's affordable level",
        )


if __name__ == "__main__":
    unittest.main()
