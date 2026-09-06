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

import re
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
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
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


class _RealEntryHostPool:
    """The five attributes the real ``HostPoolGroup.__init__`` reads off an
    entry's host pool (``memory_pool_host.py:1898-1915``). Enough to build the
    tree's own group object, which is the point: the group under test is the
    class the route names, not a stand-in that agrees with the reader.
    """

    layout = "layer_first"
    page_size = 1
    device = "cpu"
    size = 8
    can_use_write_back_jit = False


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

    def test_arm_two_the_census_offset_is_the_expression_both_documents_pin(self):
        """T-45 arm 2 -- S0 mutant 7, the DERIVED-CONSTANT half.

        The census offset has moved four times (15 -> 16 -> 18 -> 21) and a
        hard-coded offset reads a count pair as a per-rank flag without saying
        anything. The offset is pinned here as the EXPRESSION
        `PHASE_DOMAIN_SLOTS - PHASE_DOMAIN_CENSUS_SLOTS`, which is the same
        expression the S7 spec's T-S7-8 demands at B6, so the two documents
        pin ONE offset rather than two literals that can disagree.

        The width beside it is the per-KIND arithmetic off the declared table,
        never the total typed out: the enumerated kinds are the assertion and
        the total is derived from them, so a row added to the table moves both
        halves together.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        self.assertEqual(
            pdv.index_of("census"),
            pdv.PHASE_DOMAIN_SLOTS - pdv.PHASE_DOMAIN_CENSUS_SLOTS,
        )
        kinds = [term.kind for term in pdv.PHASE_DOMAIN_LAYOUT]
        self.assertEqual(kinds.count(pdv.AND_SLOT), 5)
        self.assertEqual(kinds.count(pdv.DIVERGENCE_PAIR), 6)
        self.assertEqual(kinds.count(pdv.MAX_PAIR), 2)
        self.assertEqual(kinds.count(pdv.CENSUS_BLOCK), 1)
        self.assertEqual(len(kinds), 14)
        self.assertEqual(
            pdv.PHASE_DOMAIN_SLOTS,
            kinds.count(pdv.AND_SLOT)
            + 2 * (kinds.count(pdv.DIVERGENCE_PAIR) + kinds.count(pdv.MAX_PAIR))
            + pdv.PHASE_DOMAIN_CENSUS_SLOTS,
        )

    def test_arm_three_neither_width_nor_offset_is_typed_as_a_literal(self):
        """T-45 arm 3 -- S0 mutant 7 read STRUCTURALLY, because the VALUE of a
        literal that is right today is indistinguishable from the derivation.

        `PHASE_DOMAIN_SLOTS = 29` and `at = 21` in `_render_census` both pass
        every value assertion in this file at this tip, and both go silently
        wrong on the next row added to the table -- which has happened four
        times. A value assertion therefore cannot reach this mutant while the
        table stands still, so this arm asserts the SHAPE the spec mandates:
        each width is a sum over its OWN declared layout, and `_render_census`
        reaches its head index through `index_of`, never through a number.
        """
        import ast

        from sglang.srt.managers import phase_domain_verdict as pdv

        with open(pdv.__file__) as handle:
            module = ast.parse(handle.read())

        assigned = {}
        for node in module.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value

        for constant, layout in (
            ("PHASE_DOMAIN_SLOTS", "PHASE_DOMAIN_LAYOUT"),
            ("PHASE_BOOT_REDUCE_SLOTS", "BOOT_REDUCE_LAYOUT"),
        ):
            with self.subTest(constant=constant):
                value = assigned.get(constant)
                self.assertIsNotNone(
                    value, "%s is not assigned at module level" % constant
                )
                self.assertNotIsInstance(
                    value,
                    ast.Constant,
                    "%s is typed as a literal instead of derived from %s"
                    % (constant, layout),
                )
                names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
                self.assertIn(
                    layout,
                    names,
                    "%s does not derive from %s" % (constant, layout),
                )

        render = [
            n
            for n in ast.walk(module)
            if isinstance(n, ast.FunctionDef) and n.name == "_render_census"
        ]
        self.assertEqual(len(render), 1)
        heads = [
            n.value
            for n in ast.walk(render[0])
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "at"
        ]
        self.assertEqual(len(heads), 1, "_render_census has no single head index")
        self.assertIsInstance(
            heads[0],
            ast.Call,
            "the census head index is typed in rather than derived",
        )
        self.assertIsInstance(heads[0].func, ast.Name)
        self.assertEqual(heads[0].func.id, "index_of")


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

    def test_arm_seven_the_two_absence_classes_are_not_one(self):
        """T-46 arm 7 -- TWO absences, and the layout answers them differently.

        (a) A term whose PRODUCER has not landed reads its MIN-neutral. That is
        the rule S0-C1's neutral-value list states for the whole table
        ("An AND-slot with no producer is packed `1`") and it is what T-46
        pins for slots 10 and 15: their numbered changes are S1-C12 and S1-C16
        and the payload builder answers the neutral when the route hands it
        nothing.

        (b) The two D-68 attributes are the ONE exception the spec writes out,
        and it names the shape it rejects rather than a slot number:
        "`getattr(group, \"host_ring_discard_ok\", 1)` IS REJECTED BY NAME ...
        on a bus term it would turn a missing declaration into a silently
        healthy vote". So those two are read with NO default -- the bare
        attribute reads S0-C2 itself writes ("slot 18 <- `group.host_ring_
        discard_ok`", "`delta = group.d_backup_width - last_seen`") -- and a
        missing declaration dies naming the holder and the attribute.

        THE DEATH IS AN `AttributeError` FROM THE READ, NOT A REFUSAL THIS
        MODULE INVENTS. D-20 admits no new rank-local raise and no numbered
        change owns one here, so the no-default read carries the group-uniform
        death without adding a `raise` statement of its own -- which is what
        the next arm asserts over the module's own source.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        # (b) THE TWO D-68 ATTRIBUTES -- no default, so a group-shaped holder
        # that does not declare them dies naming both facts.
        for name in ("host_ring_discard_ok", "d_backup_width"):
            with self.subTest(no_default=name):
                scheduler = _with_group(_PartialGroup(withhold=(name,)))
                with self.assertRaises(AttributeError) as caught:
                    pdv.build_phase_domain_payload(scheduler)
                msg = str(caught.exception)
                self.assertIn(name, msg)
                self.assertIn("_PartialGroup", msg)

        # (a) THE TWO ROUTE READS THE NEUTRAL RULE DOES COVER. Withholding
        # them is the B1 state of a slot whose producer has not landed, and
        # the payload answers the MIN-neutral 1 rather than stopping a boot.
        neutral_arms = (
            ("expected_transfer_layer_domain", {}, "rebind_domain_within_driven"),
            (
                "transfer_layer_domain",
                {"has_draft": True, "draft_pool": _StandInDraftPool(4)},
                "draft_tier_domain_matches",
            ),
        )
        for name, controller_kwargs, term in neutral_arms:
            with self.subTest(neutral=name):
                scheduler = _with_group(
                    _PartialGroup(withhold=(name,)), **controller_kwargs
                )
                payload = pdv.build_phase_domain_payload(scheduler)
                self.assertEqual(pdv.slot_of(payload, term), 1)
                self.assertIsNotNone(
                    pdv.unpack_phase_domain(
                        _reduce_min([payload] * 3), rank=0, local={}, world_size=3
                    )
                )

        # (c) THE TREE'S OWN `HostPoolGroup`, not a stand-in. It carries the
        # `entry_map` the route discriminates on and NONE of S1's four
        # attributes, which is exactly what a boot sees if S1-C3/S1-C16/S1-C18
        # do not land. Slots 10 and 15 must not stop that boot; the two
        # no-default reads are the only terms that may.
        real = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=_RealEntryHostPool(),
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                )
            ]
        )
        for attribute in (
            "transfer_layer_domain",
            "expected_transfer_layer_domain",
            "host_ring_discard_ok",
            "d_backup_width",
        ):
            self.assertFalse(hasattr(real, attribute), attribute)
        with self.assertRaises(AttributeError) as caught:
            pdv.build_phase_domain_payload(
                _with_group(real, has_draft=True, draft_pool=_StandInDraftPool(4))
            )
        self.assertIn("host_ring_discard_ok", str(caught.exception))

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

    def test_arm_eight_a_class_declared_counter_is_cleared_on_the_class(self):
        """T-46 arm 8 -- the clear lands where the PRODUCER wrote it.

        The spec declares two of the six per-pass counters as CLASS attributes
        in a class body -- S5-C6's is "a CLASS ATTRIBUTE
        `_state_src_contradictions = 0` in the body of class
        `HybridReqToTokenPool` (`memory_pool.py:1835-1837`)" -- and a producer
        that increments the CLASS is the shape that declaration invites.

        THE HAZARD: a clear written to the INSTANCE creates a shadowing
        instance attribute. The class value goes on climbing, the instance
        reads 0 for the life of that object, and the term votes 0 for ever
        while the detection keeps firing -- a group STOP deleted by a reset,
        with no failing assertion anywhere else in this file, because every
        other fixture declares its counters on the instance.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        class _ClassCounterPool:
            """S5-C6's declaration form: the counter lives in the class body."""

            _state_src_contradictions = 0

        pool = _ClassCounterPool()
        scheduler = _StandInScheduler(req_to_token_pool=pool)

        type(pool)._state_src_contradictions += 3
        first = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.pair_of(first, "d_state_src"), (3, -3))
        self.assertEqual(type(pool)._state_src_contradictions, 0)
        # THE KILL, stated directly: nothing was written to the instance, so
        # nothing shadows the class the producer keeps writing to.
        self.assertNotIn("_state_src_contradictions", vars(pool))

        # The producer increments the CLASS again. A shadowed instance value
        # would hide it for ever; the class read sees it on the next build.
        type(pool)._state_src_contradictions += 4
        second = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.pair_of(second, "d_state_src"), (4, -4))

        third = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.pair_of(third, "d_state_src"), (0, 0))

    def test_arm_nine_the_module_names_no_refusal_of_its_own(self):
        """The one refusal this slice owns is `PhaseDomainDivergence`.

        D-20 admits a new rank-local raise only for a boot-time invariant
        before the first collective, and this module's reads run per pass at
        `scheduler.py:7171`. `PhaseDomainDivergence` is not that shape -- it is
        raised off the REDUCED payload, so every rank holds the same min and
        max and raises on the same pass -- and it is the single row of S0's
        Refusals table. A second named marker minted here would be a refusal
        with no numbered change, no graded acceptance row and no count on
        Boot 12.
        """
        import inspect

        from sglang.srt.managers import phase_domain_verdict as pdv

        source = inspect.getsource(pdv)
        self.assertNotIn("PHASE-DOMAIN ROUTE STOP", source)
        markers = sorted(
            {m.strip() for m in re.findall(r"#1(?:068|206|924)[A-Z -]*", source)}
        )
        self.assertEqual(
            [m for m in markers if "STOP" in m],
            ["#1206 PHASE DOMAIN DIVERGENCE STOP"],
            "the only named STOP this module may mint is its own divergence: "
            "%s" % (markers,),
        )


class ThePerRankCensusIsWhatTheStopPrints(unittest.TestCase):
    """T-46 arms TEN, ELEVEN and TWELVE -- D-15's per-rank census, end to end.

    The census block had no assertion of any kind: its width was fed in as an
    INPUT by T-45's every-term-present fixture and read back by T-46's
    neutral-value arms, and nothing anywhere read `verdict.per_rank`, the
    `per_rank=[...]` field of a STOP line, or which slot a rank writes. Four
    one-line edits therefore passed the whole suite -- the `?` derived from
    the VALUE 1 instead of from the world size, every rank writing `census[0]`,
    the MIN-neutral sentinel packed `0`, and the head index typed in.

    All three rendering cases live in one fixture, because they cannot be
    separated: under MIN a slot reduces to 1 both when rank r voted GOOD and
    when no rank r exists, so only the world size the caller supplies tells
    them apart, and an arm that pins one case without the others is satisfied
    by the rule that contradicts it.
    """

    def _rank(self, tp_rank, *, incomplete=False):
        from sglang.srt.managers import phase_domain_verdict as pdv

        controller = _StandInController(None)
        if incomplete:
            setattr(controller, pdv.LOADBACK_INCOMPLETE_ATTR, True)
        return _StandInScheduler(
            tree_cache=_StandInTreeCache(controller), tp_rank=tp_rank
        )

    @staticmethod
    def _per_rank_field(message):
        found = re.search(r"per_rank=\[([^\]]*)\]", message)
        assert found is not None, "the STOP line carries no per_rank field: %s" % (
            message,
        )
        return found.group(1)

    def test_arm_ten_the_census_carries_each_ranks_own_flag_to_the_stop_line(self):
        """EVERY rank refuses in turn, and the group STOP names which one.

        `census[r]` reduces to exactly rank r's own flag because every OTHER
        rank wrote the MIN-neutral 1 there. That is what makes `0` unambiguous
        and what a shared slot, a `0` sentinel or a value-derived `?` each
        destroy in its own direction.

        THE REFUSING RANK IS A LOOP VARIABLE, NOT THE LITERAL 1. With rank 1 as
        the only refuser the census write's own rank guard
        (`phase_domain_verdict.py:574`) survives an off-by-one -- `0 <= rank`
        narrowed to `0 < rank` still writes slot 1, so rank 0's refusal turns
        into a healthy `1` and the STOP names NO refusing rank while still
        stopping the group. Rank 0 is the boundary the guard has, so it is the
        one the arm has to drive.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        for refuser in range(3):
            payloads = [
                pdv.build_phase_domain_payload(
                    self._rank(r, incomplete=(r == refuser))
                )
                for r in range(3)
            ]
            reduced = _reduce_min(payloads)
            expected = ",".join(
                ["0" if r == refuser else "1" for r in range(3)] + ["?"] * 5
            )

            for rank in range(3):
                with self.subTest(refuser=refuser, rank=rank):
                    with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                        pdv.unpack_phase_domain(
                            reduced, rank=rank, local={}, world_size=3
                        )
                    message = str(caught.exception)
                    self.assertIn("term=loadback_coverage_complete", message)
                    # Three rules in one string: the refuser's own 0, the other
                    # two ranks' own 1, and `?` for the five slots no rank owns
                    # -- decided by the world size, never by the value 1 a
                    # healthy rank also writes.
                    self.assertEqual(self._per_rank_field(message), expected)
                    self.assertIn(
                        "census_width=%d" % pdv.PHASE_DOMAIN_CENSUS_SLOTS, message
                    )

        # The healthy group renders 1 where a rank voted well, `?` where no
        # rank owns the slot -- so a `?` derived from the value would blank the
        # whole list and a `0` sentinel would report three refusals that never
        # happened.
        healthy = _reduce_min(
            [pdv.build_phase_domain_payload(self._rank(r)) for r in range(3)]
        )
        verdict = pdv.unpack_phase_domain(healthy, rank=0, local={}, world_size=3)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.per_rank, "1,1,1,?,?,?,?,?")
        self.assertEqual(verdict.census, [1, 1, 1, None, None, None, None, None])
        self.assertEqual(verdict.census_width, pdv.PHASE_DOMAIN_CENSUS_SLOTS)

    def test_arm_eleven_a_typed_in_offset_reads_a_count_pair_as_a_flag(self):
        """S0 mutant 7's hazard in the message it corrupts, not in a constant.

        The three literals the mutant names are all offsets the census HAD, so
        the wrong slice starts inside the pairs that precede it: at 18 the
        first three census positions render `host_ring_discarded` and both
        halves of `d_backup_width`. This arm drives the two apart by giving
        the count pair a value no census slot can hold, so the corrupted
        rendering is legible in the STOP line itself.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        payload = pdv.pack_phase_domain_payload(
            {"census": [1, 0, 1, 1, 1, 1, 1, 1], "d_backup_width": 5}
        )
        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
            pdv.unpack_phase_domain(payload, rank=0, local={}, world_size=3)
        field = self._per_rank_field(str(caught.exception))
        self.assertEqual(field, "1,0,1,?,?,?,?,?")
        self.assertNotIn("5", field)

    def test_arm_twelve_the_loadback_flag_is_read_and_cleared_not_typed(self):
        """Slot 9 is the SEVENTH route read and the only one nothing pinned.

        Arm 4 pins slots 18 and 19-20 and arm 6 pins the six per-pass counts;
        slot 9 fell between them, so replacing its read with the MIN-neutral
        left the suite green -- a term declared and never read, which is
        indistinguishable from a real read until the value first moves and
        which deletes `#1206 LOADBACK COVERAGE INCOMPLETE` and every census
        value with it.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        controller = _StandInController(None)
        setattr(controller, pdv.LOADBACK_INCOMPLETE_ATTR, True)
        scheduler = _StandInScheduler(
            tree_cache=_StandInTreeCache(controller), tp_rank=1
        )

        payload = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.slot_of(payload, "loadback_coverage_complete"), 0)
        self.assertEqual(
            pdv.local_terms_from_payload(payload)["census"],
            [1, 0, 1, 1, 1, 1, 1, 1],
        )
        # Read AND cleared at the payload-build site, in the same block, so the
        # next reduce reads the next pass's detection and not this one again.
        self.assertIs(getattr(controller, pdv.LOADBACK_INCOMPLETE_ATTR), False)

        again = pdv.build_phase_domain_payload(scheduler)
        self.assertEqual(pdv.slot_of(again, "loadback_coverage_complete"), 1)
        self.assertEqual(
            pdv.local_terms_from_payload(again)["census"],
            [1] * pdv.PHASE_DOMAIN_CENSUS_SLOTS,
        )


#: S7's OWN group-STOP text for slot 18, typed out here as the independent
#: witness. `WEG1_S7_HOST_RING_SPEC_0905.md`'s group-STOP line is the source and
#: the build spec's S0-C3 says the consumer written at B1 "carries as a
#: constant" that text -- so a test that imported the module's own constant
#: would assert the module agrees with itself and would pass against any
#: sentence a B1 builder invented. TWO rendered terms, `rank` and `group_min`:
#: the phase and the per-ring occupancy have NO producer on this bus at B1 and
#: are S7's own observation line at the clear.
S7_SLOT_18_STOP = (
    "#1206 HOST RING NOT DISCARDED: rank %(rank)d observed a non-empty host "
    "ring at a cutover (group_min=%(group_min)d); the rings are the "
    "phase-neutral carrier only because they are EMPTY at every cutover and "
    "refilled from L3. The phase and the per-ring occupancy are on that rank's "
    "own '#1206 HOST RING NOT DISCARDED (observation)' line at the clear."
)


class TheTwoS7ConsumersAreWrittenAtB1(unittest.TestCase):
    """S0-C3's second half -- the consumers for slot 18 and pair 19-20.

    Both terms are DECLARED at B1 and both producers land at B6, and S7 "edits
    nothing in scheduler.py 7171/7199 or phase_domain_verdict.py". So a defect
    in either consumer cannot be repaired in the slice that first drives it:
    the only lawful place is here, four to six batches before anything can
    observe it. That is why these arms exist at B1 rather than at B6, and it is
    the same schedule argument S0-C1 makes for declaring the slots themselves.

    Neither consumer can FIRE on a healthy B1 payload -- with the neutrals the
    reduce yields `group_min == 1` and `group_max == 0` -- which is exactly why
    T-46 cannot reach them and why they need driven arms of their own.
    """

    def test_the_slot_eighteen_stop_is_s7s_own_constant(self):
        """One rank votes 0, two vote 1: every rank STOPs with S7's text.

        TWO defects live here and neither is visible anywhere else in this
        file. (a) The AND-slot loop can be made to SKIP this term -- the
        payload, the reduce and the verdict object all stay correct and the
        group STOP is simply gone. (b) The message can be an invented sentence
        rendering terms the bus does not carry, which is the refusal-string
        form of the Instrument-Text-luegt hazard and which S7's own record
        struck twice before it reached this shape.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        payloads = [
            pdv.pack_phase_domain_payload({"host_ring_discarded": 0} if r == 1 else {})
            for r in range(3)
        ]
        reduced = _reduce_min(payloads)
        self.assertEqual(pdv.slot_of(reduced, "host_ring_discarded"), 0)

        for rank in range(3):
            with self.subTest(rank=rank):
                with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                    pdv.unpack_phase_domain(
                        reduced,
                        rank=rank,
                        local=pdv.local_terms_from_payload(payloads[rank]),
                        world_size=3,
                        phase="tp",
                    )
                message = str(caught.exception)
                self.assertEqual(
                    message,
                    S7_SLOT_18_STOP % {"rank": rank, "group_min": 0},
                )
                # The three absences T-S7-8 arm 1 asserts at B6, plus the two
                # this consumer's own B1 draft rendered: a term whose value
                # comes from somewhere other than this bus has no place on a
                # group STOP, and `local=`/`term=` are the two the generic
                # renderer would have added.
                for absent in (
                    "per_rank=",
                    "kv=",
                    "gdn=",
                    "phase=",
                    "local=",
                    "term=",
                    "census_width=",
                ):
                    self.assertNotIn(absent, message, absent)

    def test_a_max_pair_stops_on_one_ranks_count_and_on_a_uniform_one(self):
        """The MAX pairs' predicate is `group_max > 0`, in BOTH directions.

        Slots 16-17 and 19-20 are packed exactly like the five divergence
        pairs and consumed differently, so the predicate is one character away
        from each of the two silent forms, and each form is silent on a
        DIFFERENT case:

        * `group_min > 0` is silent on the realistic case -- ONE rank counting
          and the others at zero -- and fires only when every rank counted;
        * the `min != max` consumer the other five pairs use is silent on a
          count that is nonzero and UNIFORM, which for a wrong-answer
          condition must stop the group too.

        Both cases are driven for both pairs, because a pair is wired
        term-by-term and a consumer copied onto the second one is not caught by
        an arm that drives the first.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        for name in ("d_geom", "d_backup_width"):
            for label, counts in (
                ("one rank counted", (6, 0, 0)),
                ("every rank counted", (6, 6, 6)),
            ):
                with self.subTest(term=name, case=label):
                    payloads = [
                        pdv.pack_phase_domain_payload({name: c} if c else {})
                        for c in counts
                    ]
                    reduced = _reduce_min(payloads)
                    for rank in range(3):
                        with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                            pdv.unpack_phase_domain(
                                reduced, rank=rank, local={}, world_size=3
                            )
                        message = str(caught.exception)
                        self.assertIn(f"term={name}", message)
                        self.assertIn(f"group_min={min(counts)}", message)
                        self.assertIn("group_max=6", message)

    def test_local_terms_read_the_positive_half_of_every_pair(self):
        """`local=` on a STOP is THIS rank's contribution, not its negation.

        The pair is packed `(x, -x)`, so a reader off by one slot answers `-x`
        and every count STOP at the live seam then prints a negative number for
        a rank that counted. Nothing else in this file reaches the pair branch
        of `local_terms_from_payload`: the two arms that assert `local=` in a
        message build that dict by hand, and the two calls at the seam read
        only the census.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        # Six DISTINCT values, so a reader that lands on a neighbouring slot
        # cannot pass by answering the neighbour's count.
        counts = {
            "d_host_prov": 2,
            "d_ownership": 3,
            "d_state_src": 7,
            "d_host_unresolvable": 4,
            "d_slot_ownership": 5,
            "d_geom": 6,
        }
        payload = pdv.pack_phase_domain_payload(counts)
        local = pdv.local_terms_from_payload(payload)
        for name, value in counts.items():
            with self.subTest(term=name):
                self.assertEqual(local[name], value)


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

    The read set is the twenty-three DISTINCT `self.<name>` references between
    `scheduler.py:6817` and `:7273`, enumerated rather than guessed. Twenty-two
    are declared here; the twenty-third, `_phase_domain_layout_announced`, is
    the emitter's own bookmark, created by the method on its first pass and
    deliberately absent so the first call takes the announcing branch. Two
    further names are read through `getattr` and so are not `self.<name>`
    references at all -- `tp_cpu_group` (`scheduler.py:6916`) and
    `phase_flip_active_stack` (`:7204`) -- and both are declared below.

    `tp_rank` and `phase` ARE ARGUMENTS, not constants: they are the two seam
    inputs `unpack_phase_domain` is handed (`scheduler.py:7262`, `:7265`), and
    a fixture that can only be built at rank 0 with no phase satisfies a seam
    that types both of them in.
    """

    def __init__(self, *, tp_rank=0, phase=None):
        self.kv_session_offload = None
        self.tp_cpu_group = object()
        self.token_to_kv_pool_allocator = mock.Mock(available_size=lambda: 4096)
        self.server_args = mock.Mock(dcp_size=1)
        self.tree_cache = None
        self.waiting_queue = []
        self.phase_flip_active_stack = phase
        self.ps = _StandInPS(tp_rank)
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
        unpacks = []
        real_build = pdv.build_phase_domain_payload
        real_unpack = pdv.unpack_phase_domain

        def _record_build(scheduler):
            builds.append(1)
            return real_build(scheduler)

        def _record_unpack(*args, **kwargs):
            unpacks.append(1)
            return real_unpack(*args, **kwargs)

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
        ), mock.patch.object(
            pdv,
            "unpack_phase_domain",
            _record_unpack,
        ):
            for _ in range(2):
                scheduler_mod.Scheduler._update_uniform_pool_budget(standin)
            self.assertEqual(len(reduces), 2)
            self.assertEqual(len(builds), 2)
            # THE CONSUMER IS COUNTED BESIDE THE BUILDER. A payload that is
            # built and reduced but never unpacked is a vote with no reader --
            # the same deletion as a detection with no vote, arrived at from
            # the other end, and it passes every assertion this file makes
            # about the module itself.
            self.assertEqual(
                len(unpacks), 2, "the reduced slice must be read back at the seam"
            )

            world["size"] = 1
            for _ in range(2):
                scheduler_mod.Scheduler._update_uniform_pool_budget(standin)
            self.assertEqual(len(reduces), 2, "the PP phase takes no reduce")
            self.assertEqual(
                len(builds), 2, "the payload must be built with the reduce, not the pass"
            )
            self.assertEqual(len(unpacks), 2, "the PP phase reads nothing back")

    def test_arm_two_a_refused_term_stops_the_group_at_the_live_seam(self):
        """The STOP is at the SEAM, not only inside the module.

        Deleting the unpack call and its layout STOP from `scheduler.py`
        leaves the module, its layout and every arm above untouched and
        silently removes EVERY group STOP this bus exists for. That is
        Section 5.6 rule 3's "a deleted STOP wearing a log line" read from the
        other end: a vote whose consumer does not exist. This arm drives one
        refusing term through the real `_update_uniform_pool_budget` and
        asserts the raise leaves the method.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv
        from sglang.srt.managers import scheduler as scheduler_mod

        refused = pdv.pack_phase_domain_payload({"loadback_coverage_complete": 0})
        # RANK 2, NOT RANK 0, AND A NAMED PHASE. At rank 0 with no phase every
        # assertion below is satisfied by a typed-in literal, which is what let
        # `rank=0` and `local=None` be hard-coded into the seam and survive the
        # whole suite.
        standin = _ReduceCarryingSchedulerStandIn(tp_rank=2, phase="tp_decode")

        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=lambda *a, **k: None
        ), mock.patch.object(
            torch.distributed, "get_world_size", side_effect=lambda *a, **k: 3
        ), mock.patch.object(
            scheduler_mod.uniform_floor_scope, "report_scope", lambda *a, **k: None
        ), mock.patch(
            "sglang.srt.distributed.utils.uneven_dcp_active",
            lambda *a, **k: False,
        ), mock.patch.object(
            pdv, "build_phase_domain_payload", lambda scheduler: list(refused)
        ):
            with self.assertRaises(pdv.PhaseDomainDivergence) as caught:
                scheduler_mod.Scheduler._update_uniform_pool_budget(standin)
        message = str(caught.exception)
        self.assertIn("term=loadback_coverage_complete", message)
        self.assertIn("#1206 LOADBACK COVERAGE INCOMPLETE", message)
        # ALL FOUR ARGUMENTS THE SEAM SUPPLIES, one assertion each, and each
        # one driven to a value no literal at the seam can produce. Every arm
        # above this class calls `unpack_phase_domain` DIRECTLY with a
        # hand-built argument list, so this is the only row that reads what
        # `scheduler.py:7262-7265` actually passes.
        #
        # `rank` (`:7262`) is the term slot 18's S7 constant interpolates
        # (`phase_domain_verdict.py:140`), so a hard-coded rank makes every
        # rank of the group print the same rank number on the one STOP this
        # bus exists for -- invisible until S7 lands its producer at B6.
        self.assertIn("rank=2", message)
        # `local` (`:7263`) is THIS rank's own contribution, decoded back out
        # of the slice it packed. Dropped, the STOP prints `local=None` and the
        # reader cannot tell a rank that refused from one that was outvoted.
        self.assertIn("local=0", message)
        # `phase` (`:7265`) says which half of the cutover the group stopped
        # in. Dropped, it reads `phase=None`, which is also what a boot with no
        # phase flip prints -- the two absence classes collapsed into one.
        self.assertIn("phase=tp_decode", message)
        # `world_size` (`:7264`) is what the census's `?` is decided by.
        # Dropped, every rank at or above the width renders `1` and the STOP
        # names five phantom healthy ranks instead of saying it does not know:
        # the module's own docstring calls that deriving `?` from the value 1,
        # which a healthy rank also writes.
        self.assertIn("census_width=8", message)
        self.assertIn("per_rank=[1,1,1,?,?,?,?,?]", message)

    def test_arm_three_a_wrong_width_slice_stops_at_the_seam(self):
        """The layout STOP's one can-fail arm.

        `unpack_phase_domain` returns None ONLY for a slice of the wrong width,
        and at the live seam the slice and the width it is measured against
        both derive from `PHASE_DOMAIN_SLOTS` -- so no payload can reach this
        branch, and deleting it whole left the suite green. The hazard it
        guards is a FUTURE seam that slices by a typed-in head or width, which
        is exactly the shape `#791b PREFETCH-BALLOT LAYOUT STOP`
        (`scheduler.py:7292-7297`, pre-existing) guards twenty-five lines
        below -- `:7267` here against `:7292` there.

        The neutering is the affordance the seam's own comment names
        (`scheduler.py:7197-7198`: the module is resolved at call time "so a
        test can still neuter exactly one of its functions"), and the arm pins
        all three denominators, because a STOP that names none is a STOP whose
        reader cannot tell WHICH width was wrong.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv
        from sglang.srt.managers import scheduler as scheduler_mod

        standin = _ReduceCarryingSchedulerStandIn(tp_rank=1, phase="tp_decode")

        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=lambda *a, **k: None
        ), mock.patch.object(
            torch.distributed, "get_world_size", side_effect=lambda *a, **k: 3
        ), mock.patch.object(
            scheduler_mod.uniform_floor_scope, "report_scope", lambda *a, **k: None
        ), mock.patch(
            "sglang.srt.distributed.utils.uneven_dcp_active",
            lambda *a, **k: False,
        ), mock.patch.object(
            pdv, "unpack_phase_domain", lambda *a, **k: None
        ):
            with self.assertRaises(RuntimeError) as caught:
                scheduler_mod.Scheduler._update_uniform_pool_budget(standin)
        message = str(caught.exception)
        self.assertIn("#1068 PHASE-DOMAIN LAYOUT STOP", message)
        self.assertNotIn("#791b", message)

        head = re.search(r"head=(\d+)", message)
        expected = re.search(r"expected=(\d+)", message)
        available = re.search(r"available=(\d+)", message)
        for name, found in (
            ("head", head),
            ("expected", expected),
            ("available", available),
        ):
            self.assertIsNotNone(found, "the STOP names no %s: %s" % (name, message))
        self.assertEqual(int(expected.group(1)), pdv.PHASE_DOMAIN_SLOTS)
        # AND THE MEASUREMENT THAT SAYS THIS BRANCH CANNOT FIRE ON A REAL
        # PAYLOAD, asserted rather than argued: at the live seam the slice is
        # always at least as wide as the constant it is checked against, so the
        # only way here is the neutering above.
        self.assertGreaterEqual(int(available.group(1)), pdv.PHASE_DOMAIN_SLOTS)
        self.assertGreater(int(head.group(1)), 0)


if __name__ == "__main__":
    unittest.main()
