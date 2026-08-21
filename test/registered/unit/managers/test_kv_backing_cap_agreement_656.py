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


class TheFreeListORDERMustBeLevelledTooTest(unittest.TestCase):
    """Metal, boot_v2 2026-08-13 16:00:30-16:04Z. The second divergence.

    The levelling worked -- three ranks at 579870 / 579722 / 578606 exposed
    rows converged to 578606 in one round, and every POOL CENSUS field was
    identical from the next round on. And the FRAME diverged anyway, six
    rounds running: PP1 framed 250257408 against 1658515222.

    The pool was level; the ORDER was not. ``KvRowCap.release`` restores the
    withheld ids with ``torch.sort``, deliberately -- "the allocator takes
    from the FRONT" -- while ``_apply`` is a mask filter that PRESERVES the
    eviction order. So a rank that had to change its cap ended with a SORTED
    free list and a rank that did not kept its eviction order. Same
    membership, different order, therefore different row ids handed to the
    next request, therefore different live slot sets, therefore different
    frames.

    A group decision has to be applied group-symmetrically or it is not a
    group decision. When the group is not level, EVERY rank runs the same
    normalisation -- including the ranks already at the target, whose only
    job is to end in the same order as the ones that moved.
    """

    def _fleet(self, levels):
        fleet = []
        for lvl in levels:
            relief, pool, card = _rank(free_mib=1024 + 90_000, backed=BOOT_ROWS)
            if lvl < BOOT_ROWS:
                relief._rows_at_boot = BOOT_ROWS
                relief._cap.engage(lvl)
            # Scramble the free list the way eviction does, so "sorted" is a
            # property this test can actually observe rather than assume.
            alloc = relief._alloc
            alloc.free_pages = torch.flip(alloc.free_pages, dims=(0,))
            fleet.append((relief, pool, card))
        return fleet

    def test_every_rank_ends_with_the_same_free_list_order(self):
        fleet = self._fleet([BOOT_ROWS, 577_722, 576_606])
        _reconcile([r for r, _, _ in fleet])
        firsts = [tuple(r._alloc.free_pages[:64].tolist()) for r, _, _ in fleet]
        self.assertEqual(
            len(set(firsts)),
            1,
            "the ranks hand out different row ids next, so their live slot "
            "sets and therefore their wire frames must diverge",
        )

    def test_a_rank_already_at_the_target_still_normalises(self):
        """The rank that does not move is the one that breaks the tie.

        It is also the one an early return is most tempting for, which is
        exactly how this defect got in.
        """
        fleet = self._fleet([BOOT_ROWS, 576_606, 576_606])
        _reconcile([r for r, _, _ in fleet])
        for relief, _pool, _card in fleet:
            pages = relief._alloc.free_pages.tolist()
            self.assertEqual(pages, sorted(pages))

    def test_a_group_already_level_is_not_churned(self):
        """Normalisation is for the unlevel case only -- it is not free."""
        fleet = self._fleet([BOOT_ROWS, BOOT_ROWS, BOOT_ROWS])
        before = [tuple(r._alloc.free_pages[:8].tolist()) for r, _, _ in fleet]
        target = _agree([r for r, _, _ in fleet])
        self.assertIsNone(target, "a level group has nothing to agree")
        after = [tuple(r._alloc.free_pages[:8].tolist()) for r, _, _ in fleet]
        self.assertEqual(before, after)


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
            kbr.collective_cap_target(_reduce([good.cap_proposal(), kbr.CAP_ABSTAIN]))
        )

    def test_an_unreadable_live_set_abstains_rather_than_guessing(self):
        def boom():
            raise RuntimeError("live set unavailable")

        relief, _pool, _card = _rank(free_mib=8000)
        relief._live_slots_fn = boom
        self.assertEqual(tuple(relief.cap_proposal()), tuple(kbr.CAP_ABSTAIN))


class AnAgreedGroupIsLeftAloneTest(unittest.TestCase):
    """The common case must cost nothing: no cap churn on a healthy group."""

    def test_a_group_already_at_one_level_has_nothing_to_agree(self):
        """The no-op is decided from the REDUCED view, never per rank.

        It used to be decided per rank, by an early return in
        ``reconcile_to``, and that is precisely what let one rank normalise
        its free list while another did not -- see
        TheFreeListORDERMustBeLevelledTooTest.
        """
        fleet = [_rank(free_mib=8000, backed=BOOT_ROWS) for _ in range(3)]
        self.assertIsNone(_agree([r for r, _, _ in fleet]))
        for relief, pool, _card in fleet:
            calls = len(pool.calls)
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
            # 4 shrink + 4 cap + 4 live-slot (#656 C22-d). The width is the
            # load-bearing claim, not the number: every decision this rung
            # makes rides ONE reduction, and each round that widens it is a
            # decision that did NOT get a collective of its own. R2 moved it
            # 4 -> 8 for the cap agreement; C22-d moves it 8 -> 12 for the
            # live slot set.
            self.assertEqual(
                len(p), 12, "4 shrink fields + 4 cap fields + 4 live-slot fields"
            )
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


#: The 2026-08-14 boot, verbatim, so this file joins to its own log lines.
BOOT_ROWS_E = 585390
PP1_RECOVERED_E = 546236


class TheRecoveryMustLevelToTheGroupTest(unittest.TestCase):
    """#656 C22-e: the grow is rank-local, the ID SPACE it produces may not be.

    The measurement, 2026-08-14 on the rig, with the C22-d live-slot
    agreement already in the tree::

        05:40 PP2  proposal: current=585390 floor=585903 (max_live=585390)
        05:40 PP1  proposal: current=546236 floor=546749 (max_live=546236)
        05:44 all  FLIP ABANDONED ... the group's union reaches row 585390 and
                   the poorest rank has only 546236 rows BACKED

    ``recover`` is bounded by each rank's own distance from the corridor law,
    so the corridor-bounded rank came back at 546236 while its peers went to
    585390. From that instant the peers could hand out ids the poorest rank
    cannot map -- and BOTH downstream repairs then correctly refuse: the cap
    agreement because ``max_floor > capable``, the live-slot union because
    framing a row above a rank's backing is a read of unmapped memory. Two
    correct refusals and a flip that never happens again. The divergence has
    to be prevented where it is created.
    """

    def _fleet(self):
        """PP1 corridor-bound, peers rich -- the metal's shape."""
        poor, _p, _c = _rank(
            free_mib=1024 + (PP1_RECOVERED_E - 500_000),
            backed=500_000,
            rows=BOOT_ROWS_E,
        )
        rich = [
            _rank(free_mib=1024 + 200_000, backed=500_000, rows=BOOT_ROWS_E)[0]
            for _ in range(2)
        ]
        for r in [poor] + rich:
            r._rows_at_boot = BOOT_ROWS_E
        return [rich[0], poor, rich[1]]

    def _recover_all(self, fleet):
        from sglang.srt.managers import phase_flip_spill as pfs

        for relief in fleet:
            sched = type("S", (), {})()
            setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, relief)
            pfs.recover_kv_backing(sched)
        return [r.backed_rows() for r in fleet]

    def test_without_levelling_the_id_spaces_diverge_and_nothing_can_repair(self):
        """The dead end, reproduced: both downstream repairs decline."""
        fleet = self._fleet()
        backed = self._recover_all(fleet)
        self.assertLess(
            backed[1],
            backed[0],
            "fixture: the corridor-bounded rank must come back lower",
        )
        # The peers' allocators now span ids the poorest rank cannot map.
        self.assertGreater(max(r.exposed_rows() for r in fleet), backed[1])
        # And once a peer's live set reaches up there, the cap agreement has
        # no honest level left: its floor is above the poorest rank's ceiling.
        fleet[0]._live_slots_fn = lambda: torch.tensor([backed[0]], dtype=torch.int64)
        self.assertIsNone(
            _agree(fleet),
            "collective_cap_target must DECLINE here -- if it returned a "
            "level it would be withholding ids a peer's request is using, "
            "which is the one thing it may never do",
        )

    def test_recover_kv_backing_reduces_the_backing_and_levels(self):
        """The payload is ``[backed, -backed, -floor]`` and the group MIN decides.

        #792 widened it by the third field: the level this decides is applied
        to the ID SPACE, so the group's live set has to be able to veto it. See
        ``test_residency_cap_flip_levelling_792.py`` for what the two-field
        payload agreed to on metal.
        """
        from sglang.srt.managers import phase_flip_spill as pfs

        fleet = self._fleet()
        backed = self._recover_all(fleet)
        group_min = min(backed)
        floors = [r.live_floor_rows() for r in fleet]
        self.assertLess(max(floors), group_min, "fixture: an idle live set cannot veto")
        seen = []

        levelled = []
        for relief, own in zip(fleet, backed):
            sched = type("S", (), {})()
            setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, relief)

            def reduce_fn(vals, _own=own):
                seen.append(list(vals))
                return [group_min, -max(backed), -max(floors)]

            levelled.append(pfs.recover_kv_backing(sched, reduce_fn=reduce_fn))

        self.assertEqual(
            [v[0] for v in seen],
            list(backed),
            "each rank must propose the rows it has BACKED, not its cap and "
            "not its reservation -- the ceiling that matters is what is "
            "physically mapped",
        )
        self.assertEqual([v[1] for v in seen], [-b for b in backed])
        self.assertEqual(
            [v[2] for v in seen],
            [-f for f in floors],
            "each rank must also propose the level its own live set forbids "
            "going below, or the group can agree to confiscate the ids the "
            "radix tree is holding (#792)",
        )
        self.assertEqual(
            {r.exposed_rows() for r in fleet},
            {group_min},
            "every rank must expose the SAME id space after the levelling; "
            "one rank above the group is the whole defect",
        )

    def test_a_levelled_peer_still_remembers_it_owes_itself_a_recovery(self):
        """NOT a ratchet. ``reconcile_to`` clears ``_rows_at_boot`` when the
        level reaches the ceiling it knows about, and a rank that recovered
        fully has already cleared it -- so levelling such a rank down with
        ``reconcile_to`` alone would cap it AND destroy the record that it is
        owed a recovery, which ``recover`` needs (it returns 0 immediately on
        a None). The level has to be able to rise again as soon as the
        poorest rank can fund it."""
        from sglang.srt.managers import phase_flip_spill as pfs

        rich, pool, card = _rank(
            free_mib=1024 + 200_000, backed=500_000, rows=BOOT_ROWS_E
        )
        rich._rows_at_boot = BOOT_ROWS_E
        sched = type("S", (), {})()
        setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, rich)
        pfs.recover_kv_backing(sched)
        self.assertEqual(rich.backed_rows(), BOOT_ROWS_E, "fixture: full recovery")
        self.assertIsNone(rich._rows_at_boot, "fixture: and the memory cleared")

        rich.level_recovery_to(PP1_RECOVERED_E)
        self.assertEqual(rich.exposed_rows(), PP1_RECOVERED_E)
        self.assertIsNotNone(
            rich._rows_at_boot,
            "the levelled rank forgot its reservation, so recover() will "
            "return 0 for the rest of the boot and the group level becomes a "
            "ratchet -- the one outcome the standing rule forbids",
        )
        # And it climbs back when asked.
        pfs.recover_kv_backing(sched)
        self.assertEqual(rich.exposed_rows(), BOOT_ROWS_E)

    def test_levelling_releases_no_pages_it_is_an_id_decision(self):
        fleet = self._fleet()
        backed = self._recover_all(fleet)
        before = [r.backed_rows() for r in fleet]
        for r in fleet:
            r.level_recovery_to(min(backed))
        self.assertEqual(
            [r.backed_rows() for r in fleet],
            before,
            "levelling committed or released BACKING; it may only move ids, "
            "or a seam that was affordable a moment ago is not any more",
        )


class TheNormalisationMayNotCostDeviceMemoryTest(unittest.TestCase):
    """#656 C22-e: the sort runs every seam now, and the seam is where the
    corridor is tightest.

    ``torch.sort`` on a device tensor allocates a values AND an indices
    tensor, so replacing the free list with ``torch.sort(pages).values`` costs
    ~3x the list in transient DEVICE memory -- 14 MiB at this rig's 586642
    rows. That was invisible while the sort ran only on the rare rounds the
    cap agreement moved a rank. C22-d made it run every round on every rank,
    and gpu0's continuous minimum went from 1028/1084 MiB with 0 samples below
    the 1024 law (159212 samples, two soak boots) to 978/990 MiB with 2 and 4
    samples BELOW it. The law is a hard user limit.

    The invariant that keeps it free: the sort writes back through ``copy_``
    into the SAME storage, so the device allocates nothing. Pinned by tensor
    IDENTITY, which is the observable proxy for "no new allocation" on a CPU
    desk.
    """

    def test_sorting_reuses_the_same_storage(self):
        relief, _pool, _card = _rank(free_mib=1024 + 10, rows=64)
        alloc = relief._cap._alloc
        alloc.free_pages = torch.tensor([9, 3, 7, 1], dtype=torch.int64)
        before = alloc.free_pages
        ptr = before.data_ptr()
        relief.normalize_free_lists()
        self.assertIs(
            alloc.free_pages,
            before,
            "the free list object was REPLACED -- on a device tensor that is "
            "a fresh allocation at the seam, which is where the corridor law "
            "has the least room",
        )
        self.assertEqual(alloc.free_pages.data_ptr(), ptr, "storage moved")
        self.assertEqual(alloc.free_pages.tolist(), [1, 3, 7, 9])

    def test_an_already_sorted_list_is_not_written_at_all(self):
        relief, _pool, _card = _rank(free_mib=1024 + 10, rows=64)
        alloc = relief._cap._alloc
        alloc.free_pages = torch.tensor([1, 3, 7, 9], dtype=torch.int64)
        before = alloc.free_pages.clone()
        relief.normalize_free_lists()
        self.assertTrue(torch.equal(alloc.free_pages, before))

    def test_it_is_idempotent_which_is_what_makes_it_safe_every_round(self):
        relief, _pool, _card = _rank(free_mib=1024 + 10, rows=64)
        alloc = relief._cap._alloc
        alloc.free_pages = torch.tensor([5, 2, 8], dtype=torch.int64)
        relief.normalize_free_lists()
        first = alloc.free_pages.clone()
        for _ in range(3):
            relief.normalize_free_lists()
        self.assertTrue(
            torch.equal(alloc.free_pages, first),
            "running the normalisation on every seam round must converge, or "
            "the ranks chase each other's order forever",
        )

    def test_release_still_restores_every_withheld_id_in_order(self):
        """The merge changes the tensor SIZE, so one device allocation is
        unavoidable there -- but the ids and their order must be exactly what
        the device path produced, or the group's handout order changes with
        the plumbing."""
        relief, _pool, _card = _rank(free_mib=1024 + 10, rows=64)
        alloc = relief._cap._alloc
        alloc.free_pages = torch.arange(1, 65, dtype=torch.int64)
        relief._cap.engage(40)
        self.assertEqual(relief._cap.withheld, 24)
        relief._cap.release()
        self.assertEqual(
            alloc.free_pages.tolist(),
            list(range(1, 65)),
            "release must hand back every withheld id, ascending",
        )


if __name__ == "__main__":
    unittest.main()
