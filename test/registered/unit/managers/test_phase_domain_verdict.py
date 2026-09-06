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

# THE REAL KEY TYPES, not stand-ins for them. Two of this module's routes cross
# a package boundary to reach the dict key they index with, and an import that
# resolves to nothing is invisible to a test that keys its fixture with a
# stand-in: the fixture and the module would then agree on a symbol neither of
# them got from the tree. Driving the real `ComponentType.MAMBA` and
# `PoolName.MAMBA` is what makes "the route resolves" an assertion instead of
# an assumption.
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache_components import ComponentType
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


class _PartialGroup:
    """Group-shaped, declaring exactly the attributes it is NOT told to withhold.

    Group-shaped means it carries the `entry_map` `HostPoolGroup.__init__` sets
    at `memory_pool_host.py:1902` -- the discriminator the route uses, because
    `cache_controller.mem_pool_host` is NOT always a group (the non-hybrid
    controller assigns a plain host pool at `cache_controller.py:630`).

    So this object IS the declared holder, and a withheld attribute is a ROUTE
    DEFECT rather than an absent producer: S1-C18 declares all of them on
    `HostPoolGroup` at B1.
    """

    def __init__(self, *, withhold=(), domain=4):
        self.entry_map = {}
        self._domain = domain
        if "host_ring_discard_ok" not in withhold:
            self.host_ring_discard_ok = 1
        if "d_backup_width" not in withhold:
            self.d_backup_width = 0
        if "expected_transfer_layer_domain" not in withhold:
            self.expected_transfer_layer_domain = lambda bound_phase: self._domain
        if "transfer_layer_domain" not in withhold:
            self.transfer_layer_domain = self._domain


class _PlainHostPool:
    """What `cache_controller.mem_pool_host` holds on a NON-hybrid boot.

    `cache_controller.py:630` `        self.mem_pool_host = mem_pool_host` binds
    whatever the controller was built with, and on that path it is a host pool,
    not a `HostPoolGroup`. It carries no `entry_map`, declares none of S1-C18's
    attributes, and is not a rank that voted badly -- it is a configuration
    with no host-pool group, so every term behind that hop stays NEUTRAL.
    """


class _StandInDraftPool:
    def __init__(self, layer_num=4):
        self.layer_num = layer_num


class _StandInComponent:
    """A built tree component, carrying the two per-pass counters the route
    reads off it: S4-C5's host-provenance refusals (MAMBA only) and S3-C9's
    unresolvable-pool count (declared on three component classes and SUMMED)."""

    def __init__(self, *, host_prov=0, unresolvable=0):
        self._host_prov_refusals = host_prov
        self._host_unresolvable_count = unresolvable


class _StandInHostPool:
    """The MAMBA entry's `host_pool` -- S4-C6's geometry-mismatch counter."""

    def __init__(self, *, geom=0):
        self._geom_mismatch_count = geom


class _StandInEntry:
    def __init__(self, host_pool):
        self.host_pool = host_pool


class _StandInAllocator:
    def __init__(self, *, slot_ownership=0):
        self._slot_ownership_refusals = slot_ownership


class _StandInReqToTokenPool:
    def __init__(self, *, ownership=0, state_src=0, allocator=None):
        self._ownership_split_count = ownership
        self._state_src_contradictions = state_src
        self.mamba_allocator = allocator


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
        self.assertEqual(
            pdv.PHASE_DOMAIN_SLOTS, 21 + pdv.PHASE_DOMAIN_CENSUS_SLOTS
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
            "census": [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS,
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

    Nineteen of the twenty-one scalar terms have no producer until B2..B6. A
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

    def test_arm_six_every_per_pass_count_is_read_and_then_cleared(self):
        """T-46 arm 6 -- arm 4's discipline applied to the OTHER SIX terms.

        Arm 4 tells a READ from a hard-coded neutral for slots 18 and 19-20 by
        driving their home to a NON-neutral value. The six per-pass COUNT terms
        had no such arm, so two one-line defects were invisible: a builder who
        never reads the counter at all, and one who reads it and never clears
        it. Both leave every other assertion in this file satisfied, because
        every other fixture leaves all six counters at zero.

        Two of the six cross a package boundary to reach their home
        (`ComponentType.MAMBA` for slots 3-4, `PoolName.MAMBA` for slots
        16-17), so this arm is also what makes a route import that resolves to
        nothing fail: a swallowed ImportError answers the same neutral a
        healthy pass does.

        Six DISTINCT values, so a term wired to the wrong slot cannot pass by
        reading its neighbour's count.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        host_pool = _StandInHostPool(geom=6)
        group = _StandInGroup(entry_map={PoolName.MAMBA: _StandInEntry(host_pool)})
        mamba = _StandInComponent(host_prov=2, unresolvable=1)
        swa = _StandInComponent(unresolvable=3)
        allocator = _StandInAllocator(slot_ownership=5)
        pool = _StandInReqToTokenPool(ownership=3, state_src=7, allocator=allocator)
        scheduler = _StandInScheduler(
            tree_cache=_StandInTreeCache(
                _StandInController(group),
                components={ComponentType.MAMBA: mamba, ComponentType.SWA: swa},
            ),
            req_to_token_pool=pool,
        )

        expected = {
            "d_host_prov": 2,
            # SUMMED over the built components -- 1 on MAMBA plus 3 on SWA. A
            # single-component read would leave two of the three a detection
            # with no vote, and would read 1 here.
            "d_host_unresolvable": 4,
            "d_ownership": 3,
            "d_state_src": 7,
            "d_slot_ownership": 5,
            "d_geom": 6,
        }
        first = pdv.build_phase_domain_payload(scheduler)
        for name, value in expected.items():
            with self.subTest(term=name):
                self.assertEqual(pdv.pair_of(first, name), (value, -value))

        # READ AND CLEARED at this site, not once per scheduler pass: the
        # packed reduce does not run in the PP phase at all, so a per-pass
        # reset erases every PP-phase detection before the next TP reduce can
        # vote it.
        self.assertEqual(mamba._host_prov_refusals, 0)
        self.assertEqual(mamba._host_unresolvable_count, 0)
        self.assertEqual(swa._host_unresolvable_count, 0)
        self.assertEqual(pool._ownership_split_count, 0)
        self.assertEqual(pool._state_src_contradictions, 0)
        self.assertEqual(allocator._slot_ownership_refusals, 0)
        self.assertEqual(host_pool._geom_mismatch_count, 0)

        second = pdv.build_phase_domain_payload(scheduler)
        for name in expected:
            with self.subTest(term=name, build="second"):
                self.assertEqual(pdv.pair_of(second, name), (0, 0))

        # A count that has moved and been read is not a divergence on its own:
        # three ranks reporting the SAME count agree, and the MAX-consumed
        # pairs are the two exceptions that stop the group anyway.
        agreeing = _reduce_min([list(first) for _ in range(3)])
        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
            pdv.unpack_phase_domain(agreeing, rank=0, local={}, world_size=3)
        self.assertIn("d_geom", str(caught.exception))

    def test_arm_seven_a_missing_declaration_is_a_route_stop(self):
        """T-46 arm 7 -- the absence the neutral rule does NOT cover.

        The neutral-value rule is scoped to a term whose PRODUCER has not
        landed. It says nothing about a holder that IS the declared one and
        does not carry the declaration: answering the MIN-neutral there turns a
        missing S1-C18 declaration into a silently healthy vote on the very
        terms that exist to STOP the group -- the getattr-default-on-a-ledger-
        path shape the route table rejects by name.

        Both directions are driven, because the refusal is only correct if the
        NON-holder stays neutral: a boot with no host-pool group must not gain
        a STOP it never had.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        withheld = (
            ("host_ring_discard_ok", {}),
            ("d_backup_width", {}),
            ("expected_transfer_layer_domain", {}),
            (
                "transfer_layer_domain",
                {"has_draft": True, "draft_pool": _StandInDraftPool(4)},
            ),
        )
        for name, controller_kwargs in withheld:
            with self.subTest(missing=name):
                scheduler = _with_group(
                    _PartialGroup(withhold=(name,)), **controller_kwargs
                )
                with self.assertRaises(RuntimeError) as caught:
                    pdv.build_phase_domain_payload(scheduler)
                msg = str(caught.exception)
                self.assertIn("#1068 PHASE-DOMAIN ROUTE STOP", msg)
                self.assertIn("missing=%s" % name, msg)
                self.assertIn("memory_pool_host.py:1897", msg)
                self.assertIn("_PartialGroup", msg)

        # The complete declaration set raises nothing and votes healthy.
        whole = _with_group(
            _PartialGroup(), has_draft=True, draft_pool=_StandInDraftPool(4)
        )
        payload = pdv.build_phase_domain_payload(whole)
        self.assertEqual(pdv.slot_of(payload, "host_ring_discarded"), 1)
        self.assertEqual(pdv.pair_of(payload, "d_backup_width"), (0, 0))
        self.assertEqual(pdv.slot_of(payload, "rebind_domain_within_driven"), 1)
        self.assertEqual(pdv.slot_of(payload, "draft_tier_domain_matches"), 1)

        # AND THE OTHER DIRECTION: a bound object that is NOT the declared
        # holder is a configuration with no host-pool group, not a rank that
        # voted badly. It raises nothing and every term behind the hop reads
        # its neutral.
        plain = _with_group(
            _PlainHostPool(), has_draft=True, draft_pool=_StandInDraftPool(4)
        )
        neutral = pdv.build_phase_domain_payload(plain)
        self.assertEqual(pdv.slot_of(neutral, "host_ring_discarded"), 1)
        self.assertEqual(pdv.pair_of(neutral, "d_backup_width"), (0, 0))
        self.assertEqual(pdv.slot_of(neutral, "rebind_domain_within_driven"), 1)
        self.assertEqual(pdv.slot_of(neutral, "draft_tier_domain_matches"), 1)
        self.assertIsNotNone(
            pdv.unpack_phase_domain(
                _reduce_min([neutral] * 3), rank=0, local={}, world_size=3
            )
        )


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

    def test_arm_five_the_boot_digest_pairs_carry_a_verdict(self):
        """T-53 arm 5 -- the boot bus's copy of the packed bus's shape #1.

        The packed bus pins its digest pair from both sides. The boot bus's two
        pairs were exercised at their neutral `(0, 0)` only, and at zero a
        dropped negation and a live one read alike -- so the one-character edit
        that makes `group_min != group_max` unreachable, and the predicate
        being disabled outright, were both invisible on this bus.

        Both directions, because either alone is satisfied by a defect:
        AGREEMENT on a NON-ZERO digest must raise nothing (a dropped negation
        makes it raise, since `min != -max` for any nonzero digest), and a
        DISAGREEMENT must raise on every rank (a disabled predicate makes it
        silent).
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        agree = pdv.phase_domain_digest(["kv", "mamba"])
        odd = pdv.phase_domain_digest(["kv"])
        self.assertNotEqual(agree, odd)
        self.assertGreater(min(agree, odd), 0)

        for row in ("d_ring_format", "d_entry_map"):
            # AGREEMENT on a nonzero digest: no STOP.
            uniform = [{row: agree} for _ in range(3)]
            reduced = _reduce_min([pdv.build_boot_reduce_payload(t) for t in uniform])
            for rank, local in enumerate(uniform):
                with self.subTest(row=row, rank=rank, arm="agree"):
                    verdict = pdv.unpack_boot_reduce(reduced, rank=rank, local=local)
                    self.assertIsNotNone(verdict)
                    self.assertEqual(verdict.pairs[row], (agree, agree))

            # DISAGREEMENT: every rank of the reduce holds the same min and
            # max, so every rank raises on the SAME pass.
            locals_ = [{row: agree}, {row: agree}, {row: odd}]
            reduced = _reduce_min([pdv.build_boot_reduce_payload(t) for t in locals_])
            for rank, local in enumerate(locals_):
                with self.subTest(row=row, rank=rank, arm="diverge"):
                    with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                        pdv.unpack_boot_reduce(reduced, rank=rank, local=local)
                    msg = str(caught.exception)
                    self.assertIn("row=%s" % row, msg)
                    self.assertIn("local=%d" % local[row], msg)
                    self.assertIn("group_min=%d" % min(agree, odd), msg)
                    self.assertIn("group_max=%d" % max(agree, odd), msg)

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
        ), mock.patch(
            # PINNED, NOT INHERITED. `_update_uniform_pool_budget` re-imports
            # this symbol on every call, and a sibling suite in the same
            # process leaves it permanently replaced: measured, running
            # `test_collective_family_siblings_610.py` first turns
            # `sglang.srt.distributed.utils.uneven_dcp_active` from the
            # function into a `lambda *a: True` that outlives the test (its
            # `mock.patch` runs inside `run_ranks`' THREADS, and overlapping
            # patch/restore pairs on one global restore a mock rather than the
            # original). This pin's subject is the BUILD CADENCE -- how often
            # the payload is built relative to the reduce -- and the
            # uneven-DCP admission arm is another family's branch. Reading it
            # off a neighbour's leaked global makes this row measure the
            # neighbour: it goes RED with an AttributeError on the stand-in's
            # `tree_cache`, and S0 mutants 4 and 6 lose their named killer at
            # exactly the moment the desk gate runs the suites together.
            "sglang.srt.distributed.utils.uneven_dcp_active",
            lambda *a, **k: False,
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
