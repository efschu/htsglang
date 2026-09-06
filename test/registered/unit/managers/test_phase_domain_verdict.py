"""#1068 weg1 S0: THE VERDICT BUS -- one carrier for every rank-divergent fact.

WHAT THIS PINS. `managers/phase_domain_verdict.py` is transport only: it
declares TWO buses, packs them, and reads them back. It decides nothing of its
own, and every behaviour it carries belongs to a later slice. What can still go
wrong is the LAYOUT, and a layout defect on a MIN reduce is silent in exactly
the direction that matters -- a slot read at the wrong index, a wrong-polarity
neutral, or a width the two halves disagree about all read as "the group
agrees".

THE FOUR SHAPES THESE TESTS EXIST FOR, each a one-character edit away:

* the `(x, -x)` pair's negation dropped, so `group_min != group_max` can never
  be true and the STOP silently never fires;
* an AND-slot whose producer lands in a LATER batch packed with the falsy
  default `0` instead of the MIN-neutral `1`, so every rank raises from the
  first scheduler iteration and no boot between B1 and its producer's batch
  ever reaches a probe;
* a hard-coded census offset or a literal width, which reads a count pair as a
  per-rank flag the next time a term is added;
* a term DECLARED with its neutral value and never actually READ, which is
  indistinguishable from a real read for as long as the value stays neutral and
  deletes a group STOP on the batch that first moves it.

The three-rank fixtures reduce element-wise BY HAND (`min` over the three
payloads). No `torch.distributed`, no CUDA, no scheduler: the property under
test is arithmetic over a list of ints, and the reduce that carries it is
already proven elsewhere.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.managers import prefetch_ballot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8)


def _reduce_min(payloads):
    """The element-wise MIN the `all_reduce` at scheduler.py:7182-7183 does."""
    widths = {len(p) for p in payloads}
    assert len(widths) == 1, f"ranks packed different widths: {widths}"
    return [min(col) for col in zip(*payloads)]


class _StandInGroup:
    """A stand-in for the bound `HostPoolGroup`.

    Carries only what the payload builder reads: S1-C18's two named class
    attributes (S7 fills them at B6) and the accessor slot 15's right-hand term
    is read through.
    """

    def __init__(self, *, discard_ok=1, backup_width=0, domain=4, entry_map=None):
        self.host_ring_discard_ok = discard_ok
        self.d_backup_width = backup_width
        self._domain = domain
        self.entry_map = {} if entry_map is None else entry_map

    def expected_transfer_layer_domain(self, bound_phase):
        return self._domain

    @property
    def transfer_layer_domain(self):
        return self._domain


class _RacingGroup:
    """A bound group whose backup-width counter is written by ANOTHER THREAD.

    Every READ of `d_backup_width` lets the storage backup thread land its next
    queued increment immediately afterwards -- the exact window a read-then-zero
    implementation would zero away. A WRITE to the attribute is the reset a
    read-and-zero builder performs, and it destroys whatever arrived in that
    window.
    """

    def __init__(self, *, arrivals=(), domain=4):
        self.host_ring_discard_ok = 1
        self.entry_map = {}
        self.total = 0
        self._arrivals = list(arrivals)
        self._domain = domain

    @property
    def d_backup_width(self):
        seen = self.total
        if self._arrivals:
            self.total += self._arrivals.pop(0)
        return seen

    @d_backup_width.setter
    def d_backup_width(self, value):
        self.total = int(value)

    def expected_transfer_layer_domain(self, bound_phase):
        return self._domain

    @property
    def transfer_layer_domain(self):
        return self._domain


class _StandInCounter:
    def __init__(self, num_layers=4):
        self.num_layers = num_layers


class _StandInController:
    def __init__(self, group=None, *, has_draft=False, draft_pool=None, layers=4):
        self.mem_pool_host = group
        self.has_draft = has_draft
        self.mem_pool_host_draft = draft_pool
        self.layer_done_counter = _StandInCounter(layers)


class _StandInTreeCache:
    def __init__(self, controller=None, components=None):
        self.cache_controller = controller
        self.components = {} if components is None else components


class _StandInPS:
    def __init__(self, tp_rank=0):
        self.tp_rank = tp_rank


class _StandInScheduler:
    """The smallest object `build_phase_domain_payload` can read.

    Everything absent is an ABSENT PRODUCER and must read its neutral: that is
    the B1 state for nineteen of the twenty-one scalar terms.
    """

    def __init__(self, *, tree_cache=None, tp_rank=0, req_to_token_pool=None):
        self.tree_cache = tree_cache
        self.ps = _StandInPS(tp_rank)
        self.req_to_token_pool = req_to_token_pool


def _bare(tp_rank=0):
    return _StandInScheduler(tp_rank=tp_rank)


def _with_group(group, tp_rank=0, **controller_kwargs):
    return _StandInScheduler(
        tree_cache=_StandInTreeCache(_StandInController(group, **controller_kwargs)),
        tp_rank=tp_rank,
    )


class TheDigestPairIsTheUniformityCheck(unittest.TestCase):
    def test_digest_pair_detects_disagreement(self):
        """Two ranks agree on the transfer-domain digest, one does not.

        Every rank of the reduce holds the same min and max, so every rank
        raises on the SAME pass -- the property `prefetch_ballot.py:139-143`
        documents and the one that makes this a group STOP rather than a
        rank-local one.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        agree = pdv.phase_domain_digest(["kv", "mamba"])
        odd = pdv.phase_domain_digest(["kv"])
        self.assertNotEqual(agree, odd)

        locals_ = [
            {"d_domain": agree},
            {"d_domain": agree},
            {"d_domain": odd},
        ]
        payloads = [pdv.pack_phase_domain_payload(t) for t in locals_]
        reduced = _reduce_min(payloads)

        for rank, local in enumerate(locals_):
            with self.subTest(rank=rank):
                with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                    pdv.unpack_phase_domain(
                        reduced, rank=rank, local=local, world_size=3
                    )
                msg = str(caught.exception)
                self.assertIn("#1206 PHASE DOMAIN DIVERGENCE STOP", msg)
                self.assertIn("#1206 TRANSFER DOMAIN DIVERGENT", msg)
                self.assertIn("rank=%d" % rank, msg)
                self.assertIn("local=%d" % local["d_domain"], msg)
                self.assertIn("group_min=%d" % min(agree, odd), msg)
                self.assertIn("group_max=%d" % max(agree, odd), msg)

    def test_and_slot_zero_refuses_on_every_rank(self):
        """One rank votes 0 on the loader-coverage AND-slot; all three refuse."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        locals_ = [
            {"loader_covers_own_layers": 1},
            {"loader_covers_own_layers": 0},
            {"loader_covers_own_layers": 1},
        ]
        reduced = _reduce_min([pdv.pack_phase_domain_payload(t) for t in locals_])

        for rank, local in enumerate(locals_):
            with self.subTest(rank=rank):
                with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                    pdv.unpack_phase_domain(
                        reduced, rank=rank, local=local, world_size=3
                    )
                self.assertIn("#1206 LOADER COVERAGE REFUSED", str(caught.exception))


class AnAndSlotStopCarriesTheRanksOwnNumbers(unittest.TestCase):
    """THE STOP LINE MUST NOT LIE ABOUT THE VOTE IT DIED ON.

    Measured, boot_855_weg1b12s1f3_6d78227979_0906_134543.log, slot 15, the
    three ranks deduplicated:

        STOP rank=0 ... local=0 group_min=0 group_max=0 per_rank=[1,1,1,...]
        STOP rank=1 ... local=0 group_min=0 group_max=0 per_rank=[1,1,1,...]
        STOP rank=2 ... local=1 group_min=0 group_max=0 per_rank=[1,1,1,...]

    Two numbers on that line are not the group's. `group_max` was a COPY of
    `group_min` -- a 1-wide AND slot on a MIN reduce carries no maximum, so the
    line read as a unanimous refusal when a genuine MAX over {0,0,1} is 1 and
    the refusal was 2-of-3. And `per_rank` was the LOADBACK census, another
    term's per-rank row, rendered under the failing term's name where it
    contradicted the two `local=0` votes beside it. Instrument-Text-luegt
    class A, on the one line a reader has after a group STOP.
    """

    def _rebind_votes(self, votes):
        """One local-terms mapping per rank: this rank's slot-15 vote, and the
        same vote written into ITS OWN position of slot 15's census."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        mappings = []
        for rank, vote in enumerate(votes):
            census = [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS
            if rank < pdv.PHASE_DOMAIN_CENSUS_SLOTS:
                census[rank] = vote
            mappings.append(
                {
                    "rebind_domain_within_driven": vote,
                    "census_rebind": census,
                }
            )
        return mappings

    def _fields(self, exc):
        """The RENDERED half of the STOP line, cut off before the law sentence.

        MEASURED, this round: an assertion for `group_max=1` over the WHOLE
        message passed under a mutant that restored `group_max=group_min`,
        because the law sentence beside it spelled the same token in prose.
        Every assertion below reads this side of the `--` only.
        """
        head = str(exc).split(" -- ")[0]
        self.assertNotIn("RAENGE", head)
        return head

    def test_a_two_of_three_refusal_prints_a_group_max_of_one(self):
        """The MAX is packed as its own slot, so it is a real MAX over the
        ranks and not the minimum wearing a second name."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        locals_ = self._rebind_votes([0, 0, 1])
        reduced = _reduce_min([pdv.pack_phase_domain_payload(t) for t in locals_])

        for rank, local in enumerate(locals_):
            with self.subTest(rank=rank):
                with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                    pdv.unpack_phase_domain(
                        reduced, rank=rank, local=local, world_size=3
                    )
                msg = self._fields(caught.exception)
                self.assertIn("#1206 REBIND DOMAIN OUTSIDE DRIVEN", msg)
                self.assertIn("group_min=0 group_max=1 phase=", msg)

    def test_a_unanimous_refusal_still_prints_a_group_max_of_zero(self):
        """MUST-NOT-FIRE PARTNER. A max that is always 1 would be as useless
        as a max that is always the min: the number has to MOVE with the
        votes, so the unanimous case is driven too."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        locals_ = self._rebind_votes([0, 0, 0])
        reduced = _reduce_min([pdv.pack_phase_domain_payload(t) for t in locals_])

        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
            pdv.unpack_phase_domain(reduced, rank=0, local=locals_[0], world_size=3)
        msg = self._fields(caught.exception)
        self.assertIn("group_min=0 group_max=0 phase=", msg)

    def test_the_stop_prints_the_failing_terms_own_per_rank_row(self):
        """`per_rank` is slot 15's census, so it agrees with the `local=`
        values the three ranks print -- [0,0,1], not the loadback row."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        locals_ = self._rebind_votes([0, 0, 1])
        reduced = _reduce_min([pdv.pack_phase_domain_payload(t) for t in locals_])

        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
            pdv.unpack_phase_domain(reduced, rank=2, local=locals_[2], world_size=3)
        msg = self._fields(caught.exception)
        # Positions at or above the world size are `?`, never the healthy 1 a
        # real rank also writes.
        self.assertIn("per_rank=[0,0,1,?,?,?,?,?]", msg)
        self.assertIn("census_width=%d" % pdv.PHASE_DOMAIN_CENSUS_SLOTS, msg)
        self.assertIn("local=1", msg)

    def test_a_term_with_no_census_does_not_borrow_another_terms_row(self):
        """The loader-coverage slot has no per-rank census on this bus. Its
        STOP line says so instead of printing the loadback census under its
        own name, which is the shape the metal line had."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        locals_ = [
            {"loader_covers_own_layers": 1},
            {"loader_covers_own_layers": 0},
            {"loader_covers_own_layers": 1},
        ]
        reduced = _reduce_min([pdv.pack_phase_domain_payload(t) for t in locals_])

        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
            pdv.unpack_phase_domain(reduced, rank=0, local=locals_[0], world_size=3)
        msg = self._fields(caught.exception)
        self.assertIn("#1206 LOADER COVERAGE REFUSED", msg)
        self.assertIn("per_rank=[none]", msg)
        self.assertIn("census_width=0", msg)
        self.assertNotIn("per_rank=[1,1,1", msg)

    def test_the_loadback_term_still_prints_its_own_census(self):
        """MUST-NOT-FIRE PARTNER for the row above: the term the census
        BELONGS to keeps printing it, so the fix removes a borrowed row rather
        than the mechanism."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        locals_ = []
        for rank, vote in enumerate([1, 1, 0]):
            census = [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS
            census[rank] = vote
            locals_.append(
                {"loadback_coverage_complete": vote, "census_loadback": census}
            )
        reduced = _reduce_min([pdv.pack_phase_domain_payload(t) for t in locals_])

        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
            pdv.unpack_phase_domain(reduced, rank=0, local=locals_[0], world_size=3)
        msg = self._fields(caught.exception)
        self.assertIn("#1206 LOADBACK COVERAGE INCOMPLETE", msg)
        self.assertIn("per_rank=[1,1,0,?,?,?,?,?]", msg)
        self.assertIn("group_min=0 group_max=1 phase=", msg)

    def test_the_builder_fills_slot_fifteens_census_at_this_ranks_position(self):
        """THE PRODUCER, not only the rendering: a census nobody fills renders
        the neutral row on every rank and is indistinguishable from a healthy
        group -- the absence that made the metal line unreadable."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        # domain 4 against a counter of 9: this rank's slot-15 vote is 0.
        refusing = pdv.build_phase_domain_payload(
            _with_group(_StandInGroup(domain=4), tp_rank=2, layers=9)
        )
        at = pdv.index_of("census_rebind")
        row = refusing[at : at + pdv.PHASE_DOMAIN_CENSUS_SLOTS]
        self.assertEqual(row, [1, 1, 0, 1, 1, 1, 1, 1])
        self.assertEqual(pdv.slot_of(refusing, "rebind_domain_within_driven"), 0)

        healthy = pdv.build_phase_domain_payload(
            _with_group(_StandInGroup(domain=4), tp_rank=2, layers=4)
        )
        self.assertEqual(
            healthy[at : at + pdv.PHASE_DOMAIN_CENSUS_SLOTS],
            [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS,
        )
        self.assertEqual(pdv.slot_of(healthy, "rebind_domain_within_driven"), 1)

    def test_the_builder_fills_the_loadback_census_from_the_loadback_vote(self):
        """THE OTHER census's producer, on the same helper. A census filled
        with a CONSTANT instead of its term's vote is indistinguishable from a
        healthy group for as long as the term never refuses, which on this bus
        is the whole of B1 -- the same never-actually-read shape T-46 arm 4
        exists for."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        scheduler = _with_group(_StandInGroup(domain=4), tp_rank=1, layers=4)
        setattr(
            scheduler.tree_cache.cache_controller, pdv.LOADBACK_INCOMPLETE_ATTR, True
        )
        payload = pdv.build_phase_domain_payload(scheduler)

        at = pdv.index_of("census_loadback")
        self.assertEqual(
            payload[at : at + pdv.PHASE_DOMAIN_CENSUS_SLOTS],
            [1, 0, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(pdv.slot_of(payload, "loadback_coverage_complete"), 0)
        # Slot 15 is healthy in the same payload, so the two censuses cannot be
        # one row read twice.
        self.assertEqual(
            payload[
                pdv.index_of("census_rebind") : pdv.index_of("census_rebind")
                + pdv.PHASE_DOMAIN_CENSUS_SLOTS
            ],
            [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS,
        )


class ThePayloadRidesTheSeamWithoutMovingTheBallot(unittest.TestCase):
    """GREEN REGRESSION LOCK, not a red-first row.

    `scheduler.py:7143-7149` states the discipline this pins: everything before
    the #1203 seam is indexed from the HEAD and the ballot is indexed from the
    TAIL, so an insertion AT the seam leaves both readings intact. Written over
    an arbitrary insertion width on purpose -- the property is about WHERE the
    block goes, not how wide it is, and a test that needed the new module could
    not have been green before it existed.
    """

    def test_payload_insertion_leaves_the_ballot_slice_intact(self):
        head = [11, -11, 12, -12, 13]
        ballot = prefetch_ballot.build_prefetch_ballot_payload(["a", "b"], {})
        self.assertEqual(len(ballot), prefetch_ballot.PREFETCH_BALLOT_SLOTS + 2)

        for width in (0, 1, 21, 29, 40):
            with self.subTest(insertion_width=width):
                inserted = list(range(100, 100 + width))
                seam_at = len(head)
                vals = head + inserted + ballot
                tail_at = len(vals) - (prefetch_ballot.PREFETCH_BALLOT_SLOTS + 2)
                # The TAIL reading still lands on the first ballot element.
                self.assertEqual(vals[tail_at], ballot[0])
                self.assertEqual(vals[tail_at:], ballot)
                # The HEAD-captured index recovers exactly the inserted block.
                self.assertEqual(vals[seam_at : seam_at + width], inserted)
                # And the index a builder would capture AFTER the ballot append
                # -- mutant 4 -- does not: it runs off the end of the payload.
                # Skipped at width 0, where an empty block is trivially equal
                # to an empty slice and the assertion would say nothing.
                if width:
                    after_at = len(vals)
                    self.assertNotEqual(vals[after_at : after_at + width], inserted)


class TheWidthIsDerivedAndCheckedBothWays(unittest.TestCase):
    """T-45 -- the WIDTH invariant."""

    def test_payload_width_matches_the_derived_constant(self):
        from sglang.srt.managers import phase_domain_verdict as pdv

        # DERIVED, never typed: re-derive it here off the declared layout, so
        # this assertion follows a term added to the table instead of pinning
        # a literal that would have to be edited beside it (mutant 7).
        derived = sum(term.width for term in pdv.PHASE_DOMAIN_LAYOUT)
        self.assertEqual(pdv.PHASE_DOMAIN_SLOTS, derived)
        # 13 scalar terms, EVERY ONE of them two slots wide -- the five AND
        # terms carry `(v, -v)` like the pairs since round 4, so the STOP line
        # has a real maximum -- plus TWO census blocks, one per censused term.
        self.assertEqual(
            pdv.PHASE_DOMAIN_SLOTS, 26 + 2 * pdv.PHASE_DOMAIN_CENSUS_SLOTS
        )
        self.assertEqual(
            [t.name for t in pdv.PHASE_DOMAIN_LAYOUT if t.kind == pdv.CENSUS_BLOCK],
            ["census_loadback", "census_rebind"],
        )

        every_term_present = {
            "loader_covers_own_layers": 1,
            "d_domain": 7,
            "d_host_prov": 1,
            "d_ownership": 2,
            "d_state_src": 3,
            "loadback_coverage_complete": 1,
            "draft_tier_domain_matches": 1,
            "d_host_unresolvable": 4,
            "d_slot_ownership": 5,
            "rebind_domain_within_driven": 1,
            "d_geom": 6,
            "host_ring_discarded": 1,
            "d_backup_width": 8,
            "census_loadback": [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS,
            "census_rebind": [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS,
        }
        self.assertEqual(
            len(pdv.pack_phase_domain_payload(every_term_present)),
            pdv.PHASE_DOMAIN_SLOTS,
        )
        self.assertEqual(
            len(pdv.build_phase_domain_payload(_bare())), pdv.PHASE_DOMAIN_SLOTS
        )

        # A slice of the wrong width returns None and raises NOTHING -- the
        # discipline prefetch_ballot.py:145-147 states, with the `+ 2` NOT
        # copied: PHASE_DOMAIN_SLOTS is the TOTAL width here.
        good = pdv.pack_phase_domain_payload(every_term_present)
        for wrong in (good[:-1], good + [0]):
            with self.subTest(width=len(wrong)):
                self.assertIsNone(
                    pdv.unpack_phase_domain(wrong, rank=0, local={}, world_size=3)
                )


class APayloadWithNoProducersIsSilent(unittest.TestCase):
    """T-46 -- the NEUTRAL-VALUE invariant. THE BOOT-KILLER PIN.

    Eleven of the thirteen scalar terms have no producer until B2..B6. A
    builder who packs an unproduced AND-slot with the falsy `0` makes every
    rank raise from the first scheduler iteration after B1.
    """

    def test_a_payload_with_no_producers_raises_nothing(self):
        from sglang.srt.managers import phase_domain_verdict as pdv

        # Arms 1-3: three identical stand-in ranks, every term absent.
        payloads = [pdv.build_phase_domain_payload(_bare(r)) for r in range(3)]
        reduced = _reduce_min(payloads)

        for rank in range(3):
            with self.subTest(rank=rank):
                verdict = pdv.unpack_phase_domain(
                    reduced, rank=rank, local={}, world_size=3
                )
                self.assertIsNotNone(verdict)
                for name in (
                    "loader_covers_own_layers",
                    "loadback_coverage_complete",
                    "draft_tier_domain_matches",
                    "rebind_domain_within_driven",
                    "host_ring_discarded",
                ):
                    self.assertEqual(verdict.and_slots[name], 1, name)
                self.assertEqual(len(verdict.and_slots), 5)
                for name in (
                    "d_domain",
                    "d_host_prov",
                    "d_ownership",
                    "d_state_src",
                    "d_host_unresolvable",
                    "d_slot_ownership",
                    "d_geom",
                    "d_backup_width",
                ):
                    self.assertEqual(verdict.pairs[name], (0, 0), name)
                self.assertEqual(len(verdict.pairs), 8)

    def test_arm_four_the_two_s7_terms_are_read_and_not_typed(self):
        """T-46 arm 4 -- the arm that tells a READ from a hard-coded neutral.

        Without it a builder who types the neutrals in satisfies every other
        assertion for the whole of B1..B5, because a hard-coded `1` and a real
        read of `1` are indistinguishable while the value never moves.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        group = _StandInGroup(discard_ok=0, backup_width=3)
        scheduler = _with_group(group)
        payload = pdv.build_phase_domain_payload(scheduler)

        self.assertEqual(pdv.slot_of(payload, "host_ring_discarded"), 0)
        self.assertEqual(pdv.pair_of(payload, "d_backup_width"), (3, -3))
        # The flag is read AND restored to 1 in the same statement; the counter
        # is MONOTONIC and is never written by the payload builder.
        self.assertEqual(group.host_ring_discard_ok, 1)
        self.assertEqual(group.d_backup_width, 3)
        # The builder's own bookmark advanced instead.
        self.assertEqual(scheduler._d_backup_width_last_seen, 3)

        # A freshly built group reads the INITIAL value 0 (initial value, not
        # reset target), and a counter that has moved reads its delta ONCE.
        fresh = _StandInGroup()
        fresh_sched = _with_group(fresh)
        self.assertEqual(
            pdv.pair_of(pdv.build_phase_domain_payload(fresh_sched), "d_backup_width"),
            (0, 0),
        )
        fresh.d_backup_width = 3
        self.assertEqual(
            pdv.pair_of(pdv.build_phase_domain_payload(fresh_sched), "d_backup_width"),
            (3, -3),
        )
        self.assertEqual(
            pdv.pair_of(pdv.build_phase_domain_payload(fresh_sched), "d_backup_width"),
            (0, 0),
        )

    def test_arm_five_an_increment_between_two_builds_is_not_lost(self):
        """T-46 arm 5 -- the INTERLEAVED increment, driven rather than argued.

        S7's writer is the storage backup THREAD and the reader is the
        scheduler loop, so a read-then-zero is a cross-thread
        read/modify/write: an increment that lands between the read and the
        zero is discarded silently, and the discarded one is by construction
        the one that happened while the scheduler was busy -- i.e. under load.

        `_RacingGroup` lands exactly that increment in exactly that window. A
        test that merely increments BETWEEN two builds does not tell the two
        implementations apart at all -- measured: it passes under read-and-zero
        -- which is why the race is modelled instead of described.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        group = _RacingGroup(arrivals=[2])
        scheduler = _with_group(group)
        first = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.pair_of(first, "d_backup_width"), (0, 0))
        second = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.pair_of(second, "d_backup_width"), (2, -2))
        # And the counter itself is MONOTONIC: nothing in the payload builder
        # ever writes it, so the total the writer thread reached survives.
        self.assertEqual(group.total, 2)


class TheBootReduceLayoutIsDeclaredOnceAndDerived(unittest.TestCase):
    """T-53 -- the boot bus's T-45 and T-46 in one."""

    def test_the_boot_reduce_layout_is_thirteen_wide_and_neutral_when_unfilled(self):
        from sglang.srt.managers import phase_domain_verdict as pdv

        derived = sum(row.width for row in pdv.BOOT_REDUCE_LAYOUT)
        self.assertEqual(pdv.PHASE_BOOT_REDUCE_SLOTS, derived)
        self.assertEqual(derived, 13)
        self.assertEqual(
            len(pdv.build_boot_reduce_payload()), pdv.PHASE_BOOT_REDUCE_SLOTS
        )

        and_rows = [
            "owned_equals_driven",
            "layer_mapping_non_empty",
            "counter_index_space_known",
            "draft_tier_domain_matches_at_boot",
            "write_through_ring_ok",
            "one_wave_floor_ok",
            "anchor_floor_ok",
            "ring_arity_ok",
            "host_pool_build_ok",
        ]
        pair_rows = ["d_ring_format", "d_entry_map"]
        # 9 AND-slots + 2x2 pair scalars = 13, the derived width.
        self.assertEqual(len(and_rows) + 2 * len(pair_rows), derived)

        reduced = _reduce_min([pdv.build_boot_reduce_payload() for _ in range(3)])
        for rank in range(3):
            with self.subTest(rank=rank):
                verdict = pdv.unpack_boot_reduce(reduced, rank=rank, local={})
                self.assertIsNotNone(verdict)
                for name in and_rows:
                    self.assertEqual(verdict.and_slots[name], 1, name)
                for name in pair_rows:
                    self.assertEqual(verdict.pairs[name], (0, 0), name)

        good = pdv.build_boot_reduce_payload()
        for wrong in (good[:-1], good + [0]):
            with self.subTest(width=len(wrong)):
                self.assertIsNone(pdv.unpack_boot_reduce(wrong, rank=0, local={}))

    def test_a_rank_that_could_not_build_its_host_pools_stops_the_group(self):
        """T-53 arm 4 -- the FILLED path of row 12.

        D-85: an int64 MIN carries no string, so every rank prints ITS OWN
        line. The rank that recorded the failure prints its own message; a
        healthy rank says so and points at the peer's log.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        failed = {
            "host_pool_build_ok": 0,
            "host_pool_build_msg": "#1206 TRANSFER DOMAIN SHORTFALL owned=24 driven=32",
        }
        healthy = {"host_pool_build_ok": 1}
        locals_ = [healthy, failed, healthy]
        reduced = _reduce_min([pdv.build_boot_reduce_payload(t) for t in locals_])
        self.assertEqual(reduced[pdv.boot_index_of("host_pool_build_ok")], 0)

        for rank, local in enumerate(locals_):
            with self.subTest(rank=rank):
                with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                    pdv.unpack_boot_reduce(reduced, rank=rank, local=local)
                msg = str(caught.exception)
                self.assertIn("host_pool_build_ok", msg)
                if local is failed:
                    self.assertIn(failed["host_pool_build_msg"], msg)
                else:
                    self.assertIn(
                        "#1068 host_pool_build_ok: row 12 healthy on this rank; "
                        "a peer recorded the failure, see its log",
                        msg,
                    )


class _ReduceCarryingSchedulerStandIn:
    """Enough of a Scheduler for `_update_uniform_pool_budget` to run.

    The read set is the twenty-two `self.<name>` references between
    `scheduler.py:6817` and `:7261`, enumerated rather than guessed.
    """

    def __init__(self):
        self.kv_session_offload = None
        self.tp_cpu_group = object()
        self.token_to_kv_pool_allocator = mock.Mock(available_size=lambda: 4096)
        self.server_args = mock.Mock(dcp_size=1)
        self.tree_cache = None
        self.waiting_queue = []
        self.ps = _StandInPS(0)
        self._pass_prefetch_verdicts = None
        self._uniform_min_avail = None
        self._uniform_budget_deficit = None
        self._uniform_corridor_width = None
        self._uniform_head_inputs = None
        self._uniform_prefetch_ballot = None

    def _local_host_avail(self):
        return 1

    def _local_mamba_avail(self):
        return 1

    def _local_corridor_width_ceiling(self):
        return 1

    def _local_head_prefix_matches(self):
        return [], {}

    def _local_admit_limit(self):
        return 1

    def _local_seam_premise_vote(self):
        return 1

    def _drain_prefetch_progress(self):
        return {}

    def _publish_uniform_evict_floor(self, *a, **k):
        return None

    def _publish_uniform_host_floor(self, *a, **k):
        return None

    def _publish_uniform_mamba_floor(self, *a, **k):
        return None


class ThePayloadIsBuiltWithTheReduceAndNotWithThePass(unittest.TestCase):
    """T-47 -- the arithmetic the reset rule depends on.

    The packed reduce does NOT run in the PP phase: `_update_uniform_pool_budget`
    returns at `scheduler.py:6978` under the world-size guard at `:6916-6917`,
    and `--enable-phase-flip` rebuilds the TP group to size 1 for the whole PP
    phase. So a counter cleared once per SCHEDULER PASS is cleared on passes
    where nothing read it, and every PP-phase detection is erased before the
    next TP reduce can vote it -- a group STOP deleted by a reset.

    This row does NOT establish the PP half of D-33 and is not offered as
    doing so; that half is refuted by measurement on Boot 11.
    """

    def test_the_payload_is_built_once_per_reduce_and_not_once_per_pass(self):
        from sglang.srt.managers import phase_domain_verdict as pdv
        from sglang.srt.managers import scheduler as scheduler_mod

        reduces = []
        builds = []
        real_build = pdv.build_phase_domain_payload

        def _record_build(scheduler):
            builds.append(1)
            return real_build(scheduler)

        world = {"size": 3}
        standin = _ReduceCarryingSchedulerStandIn()

        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=lambda *a, **k: reduces.append(1)
        ), mock.patch.object(
            torch.distributed, "get_world_size", side_effect=lambda *a, **k: world["size"]
        ), mock.patch.object(
            scheduler_mod.uniform_floor_scope, "report_scope", lambda *a, **k: None
        ), mock.patch.object(
            pdv,
            "build_phase_domain_payload",
            _record_build,
        ):
            for _ in range(2):
                scheduler_mod.Scheduler._update_uniform_pool_budget(standin)
            self.assertEqual(len(reduces), 2)
            self.assertEqual(len(builds), 2)

            world["size"] = 1
            for _ in range(2):
                scheduler_mod.Scheduler._update_uniform_pool_budget(standin)
            self.assertEqual(len(reduces), 2, "the PP phase takes no reduce")
            self.assertEqual(
                len(builds), 2, "the payload must be built with the reduce, not the pass"
            )


if __name__ == "__main__":
    unittest.main()
