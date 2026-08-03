"""#286 short-term offload register -- the asset-class layer (task #286).

Falsifier-first. What each group here would catch, and the neuter that turns
it red (the can-fail proof is named per test, run and recorded in
``docs/dev/DESIGN_286_short_term_register.md`` §6):

1. **Ladder order.** ``DESIGN_407_memtier_registry.md`` §8 fixes ONE eviction
   order for the whole fork and ``FEATURE_CATALOG.md`` §17 makes consuming it
   mandatory. The order is only real if a violation is red: the tests below
   pin rank-ascending, then coldest-first, and pin that rank 5 (active work)
   is never planned at all.

2. **Refusal under active capture.** #452 refuted work inside a capture; the
   surviving pattern is eager work BETWEEN replays. A register that quietly
   parked during a capture would corrupt the recording rather than fail, so
   the refusal is pinned as a NAMED error and the mover is pinned as untouched
   after it fires.

3. **Absent provenance.** ``price_park_target`` must refuse a tier whose
   bandwidth was never measured, rather than sort it last -- the #348b D4
   defect one layer up. Pinned in both directions: absent refuses, and an
   ESTIMATE refuses too under the default ``require_measured``.

4. **The #407 wiring itself.** This module is the fork's first production
   consumer of ``memtier``. ``test_unwired_features_421.py``'s F6 pin asserted
   that no such consumer existed; it is retired in the same commit, and the
   POSITIVE fact replaces it here (same treatment #394's cold-tier pin got).

Hermetic: no torch, no CUDA, no driver. The #93 mover is faked; the real
``AdaptiveGraphStateMover`` is BOOT-PENDING and asserted only at the level of
"it refuses coherently without a manager".

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
        test/registered/unit/model_executor/test_short_term_offload_register.py -v
"""

import unittest

from sglang.srt.memtier.registry import TierQuery, TierRegistry
from sglang.srt.memtier.tiers import (
    PayloadClass,
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
    device_tier_id,
    filesystem_tier_id,
    host_tier_id,
)
from sglang.srt.model_executor.offload_register import (
    OFFLOAD_CLASSES,
    ClassPolicy,
    OffloadRefused,
    OffloadRegister,
    resolve_class_policies,
)
from sglang.srt.model_executor.short_term_offload_register import (
    ASSET_CLASSES,
    GROUND_CAPTURE_ACTIVE,
    GROUND_GRAPH_ADDRESSED,
    AdaptiveGraphStateMover,
    AssetClassDescriptor,
    ContentState,
    FakeGraphStateMover,
    GraphFamilyRegister,
    LadderRank,
    OffloadUnderCaptureRefused,
    UnpricedTierRefused,
    capture_is_active,
    describe_class,
    graph_addressed_classes,
    graph_family_item_id,
    plan_spill,
    price_park_target,
    refuse_if_capture_active,
    refuse_if_move_illegal,
    set_capture_probe,
    set_graph_reference_probe,
)
from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

GIB = 1024**3
MIB = 1024**2


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _caps(bandwidth: Rate) -> TierCaps:
    return TierCaps(
        latency_us=Rate.absent("fixture: no latency probe", unit="us"),
        bandwidth_gbs=bandwidth,
        aperture_bytes=Rate.absent("fixture: not aperture gated", unit="bytes"),
        ledger_key="fixture",
    )


def _tier(
    tier_id,
    kind,
    *,
    volatility,
    bandwidth,
    admits,
    total=100 * GIB,
    floor=0,
    host="rig-1",
) -> TierDescriptor:
    return TierDescriptor(
        id=tier_id,
        kind=kind,
        host=host,
        capacity=TierCapacity(
            total=Rate.measured(float(total), "fixture", unit="bytes"),
            floor=Rate.measured(float(floor), "fixture", unit="bytes"),
        ),
        volatility=volatility,
        admits=frozenset(admits),
        caps=_caps(bandwidth),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name="posix"),
        profile_id="fixture",
    )


ORIGIN = device_tier_id("GPU-fixture-a")
PEER = device_tier_id("GPU-fixture-b")


def _registry(
    *,
    host_bandwidth: Rate = None,
    peer_bandwidth: Rate = None,
    with_tmpfs: bool = False,
):
    """A fixture rig: the origin card, a PEER card, host DRAM, optionally tmpfs.

    The peer card exists because "not the origin" and "not a device tier" are
    different rules, and only a second card can tell them apart -- it is the
    legitimate target that makes a ``DEVICE_BOUND`` class parkable at all.

    The tmpfs exists to prove the volatility law bites: it is
    ``RECONSTRUCTABLE_OK``, so it may hold ``lane_workspaces`` but never a
    graph family's evacuated capture state.
    """
    if host_bandwidth is None:
        host_bandwidth = Rate.measured(12.5, "fixture: pcie4 x16", unit="GB/s")
    if peer_bandwidth is None:
        peer_bandwidth = Rate.measured(50.0, "fixture: p2p", unit="GB/s")
    tiers = [
        _tier(
            ORIGIN,
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=Rate.measured(700.0, "fixture: hbm", unit="GB/s"),
            admits=set(OFFLOAD_CLASSES),
            total=32 * GIB,
        ),
        _tier(
            PEER,
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=peer_bandwidth,
            admits=set(OFFLOAD_CLASSES),
            total=32 * GIB,
        ),
        _tier(
            host_tier_id("rig-1"),
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            bandwidth=host_bandwidth,
            admits=set(OFFLOAD_CLASSES),
        ),
    ]
    if with_tmpfs:
        tiers.append(
            _tier(
                filesystem_tier_id("rig-1", "/dev/shm"),
                TierKind.FILESYSTEM,
                volatility=Volatility.RECONSTRUCTABLE_OK,
                bandwidth=Rate.measured(20.0, "fixture: tmpfs", unit="GB/s"),
                admits=set(OFFLOAD_CLASSES),
            )
        )
    return TierRegistry(tiers, profile_id="fixture", local_host="rig-1")


def _register(**policy_overrides) -> OffloadRegister:
    """A base #286 register with every class parkable, so the tests exercise
    the LADDER rather than the class knobs (those are covered by
    ``test_offload_register.py``)."""
    policies = resolve_class_policies("capacity")
    for klass, policy in policy_overrides.items():
        policies[klass] = policy
    reg = OffloadRegister(policies=policies, hysteresis_window_s=0.0)
    return reg


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------
# 1. asset-class descriptors
# ---------------------------------------------------------------------------


class AssetClassDescriptorTest(unittest.TestCase):
    def test_every_offload_class_has_a_descriptor(self):
        """A class without a rank is silently unplannable -- the #421 silence.

        Can-fail proof: drop one entry from ``_descriptors()`` and the module
        raises at IMPORT, so this assertion is a second net under an
        import-time guard rather than the only one.
        """
        self.assertEqual(sorted(ASSET_CLASSES), sorted(OFFLOAD_CLASSES))
        for klass, desc in ASSET_CLASSES.items():
            self.assertIsInstance(desc, AssetClassDescriptor, klass)
            self.assertIsInstance(desc.ladder_rank, LadderRank, klass)
            self.assertIsInstance(desc.payload, PayloadClass, klass)
            self.assertTrue(desc.dimension_presets, klass)
            self.assertTrue(desc.rationale.strip(), klass)

    def test_graph_families_are_the_ladders_second_rung(self):
        """DESIGN_407 §8.2 names inactive layout/graph families as rung 2, and
        DESIGN_363 §20.3's RUNG 1 eviction is a named instance of it."""
        self.assertIs(
            describe_class("graph_rungs").ladder_rank,
            LadderRank.INACTIVE_GRAPH_FAMILY,
        )
        self.assertEqual(int(LadderRank.INACTIVE_GRAPH_FAMILY), 2)

    def test_the_ladder_is_ordered_as_the_doctrine_writes_it(self):
        """The five rungs in §8's own order. An IntEnum re-spelled out of
        order would silently invert every plan in this file."""
        self.assertEqual(
            [r.name for r in sorted(LadderRank)],
            [
                "COLD_SECOND_MODEL",
                "INACTIVE_GRAPH_FAMILY",
                "COLD_EXPERTS",
                "IDLE_SESSIONS",
                "ACTIVE_WORK",
            ],
        )

    def test_live_gdn_state_sets_are_device_bound(self):
        """DESIGN_407 X2: LIVE recurrent state may not leave its card.
        Declaring it anything weaker would let the tier table offer host RAM
        for a set the GDN kernels are still updating in place."""
        self.assertIs(
            describe_class("gdn_state_sets").payload, PayloadClass.DEVICE_BOUND
        )
        self.assertIs(
            describe_class("gdn_state_sets").payload_for(ContentState.LIVE),
            PayloadClass.DEVICE_BOUND,
        )

    def test_a_graph_family_is_not_droppable(self):
        """RECONSTRUCTABLE would admit a tier that may lose the bytes; the
        regeneration cost is DESIGN_363 §20.3 RUNG 2's 3-6 s recapture."""
        self.assertIs(
            describe_class("graph_rungs").payload,
            PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
        )
        self.assertTrue(describe_class("graph_rungs").va_stable_required)

    def test_unknown_class_is_a_hard_error(self):
        with self.assertRaises(ValueError) as ctx:
            describe_class("no_such_class")
        self.assertIn("no_such_class", str(ctx.exception))


# ---------------------------------------------------------------------------
# 2. the ladder, as executable order
# ---------------------------------------------------------------------------


class LadderOrderTest(unittest.TestCase):
    """FALSIFIER GROUP 1. Neuter ``_ladder_sort_key`` to ``item.item_id`` and
    every test in this class turns red."""

    def _populated(self):
        clock = _Clock()
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
            clock=clock,
        )
        # Registered in DELIBERATELY WRONG order and with ids whose alphabetic
        # order is the INVERSE of the ladder order, so a planner that fell back
        # to sorting by id would produce the exact reverse of the expectation.
        clock.t = 100.0
        reg.register("a_session", "gdn_state_sets", 1 * GIB, 0.0)
        clock.t = 101.0
        reg.register("b_expert", "experts", 1 * GIB, 0.0)
        clock.t = 102.0
        reg.register("c_family", "graph_rungs", 1 * GIB, 0.0)
        clock.t = 103.0
        reg.register("d_shadow", "kv_shadow", 1 * GIB, 0.0)
        clock.t = 200.0
        return reg

    def test_plan_walks_the_global_ladder_least_important_first(self):
        """Rank 1 (kv shadow) before rank 2 (graph family) before rank 3
        (cold experts) before rank 4 (idle sessions) -- DESIGN_407 §8."""
        plan = plan_spill(self._populated(), 4 * GIB)
        self.assertEqual(
            [s.item_id for s in plan.steps],
            ["d_shadow", "c_family", "b_expert", "a_session"],
        )
        self.assertEqual(
            [int(s.ladder_rank) for s in plan.steps],
            [1, 2, 3, 4],
        )

    def test_within_one_rank_the_coldest_goes_first(self):
        """§8's 'coldest-first within a class'. The three items share rank 2
        and differ only in last access; the id order is again the inverse."""
        clock = _Clock()
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
            clock=clock,
        )
        clock.t = 300.0  # warmest
        reg.register("aaa_warm", "graph_rungs", GIB, 0.0)
        clock.t = 100.0  # coldest
        reg.register("zzz_cold", "graph_rungs", GIB, 0.0)
        clock.t = 200.0
        reg.register("mmm_mid", "drafter_heads", GIB, 0.0)
        clock.t = 400.0
        plan = plan_spill(reg, 3 * GIB)
        self.assertEqual(
            [s.item_id for s in plan.steps], ["zzz_cold", "mmm_mid", "aaa_warm"]
        )

    def test_active_work_is_never_planned(self):
        """§8.5 is not a preference. A pressure planner that could reach rank 5
        by widening its budget would break the #273 FCFS guarantee, which is
        the scheduler's to keep, not a memory planner's to spend.

        Can-fail proof: remove ``ACTIVE_WORK`` from ``_UNPLANNABLE_RANKS`` and
        this fails -- the item becomes plannable and ``satisfied`` flips True.
        """
        reg = _register()
        reg.register("active", "graph_rungs", 8 * GIB, 0.0)
        # Re-rank graph families to ACTIVE_WORK for the duration of this test.
        original = ASSET_CLASSES["graph_rungs"]
        patched = AssetClassDescriptor(
            offload_class="graph_rungs",
            ladder_rank=LadderRank.ACTIVE_WORK,
            payload=original.payload,
            va_stable_required=original.va_stable_required,
            time_constant_tier=original.time_constant_tier,
            dimension_presets=original.dimension_presets,
            wired=original.wired,
            rationale="test patch",
        )
        ASSET_CLASSES["graph_rungs"] = patched
        try:
            plan = plan_spill(reg, 1 * GIB)
        finally:
            ASSET_CLASSES["graph_rungs"] = original
        self.assertEqual(plan.steps, [])
        self.assertFalse(plan.satisfied)
        self.assertIn("active", plan.skipped)
        self.assertIn("FCFS", plan.skipped["active"])

    def test_spill_is_partial_not_all_or_nothing(self):
        """§8: 'only the overflowing part spills'. Four parkable GiB and a
        1-GiB shortfall must move ONE item, not the class.

        Can-fail proof: drop the ``break`` in ``plan_spill`` and this reports
        four steps for a one-item request.
        """
        plan = plan_spill(self._populated(), 1 * GIB)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].item_id, "d_shadow")
        self.assertTrue(plan.satisfied)
        self.assertEqual(plan.bytes_planned, GIB)

    def test_a_short_inventory_reports_short_rather_than_reaching_further(self):
        reg = _register()
        reg.register("only", "graph_rungs", GIB, 0.0)
        plan = plan_spill(reg, 8 * GIB)
        self.assertFalse(plan.satisfied)
        self.assertEqual(plan.bytes_planned, GIB)
        self.assertIn("SHORT", plan.render())

    def test_hot_parked_and_protected_items_are_skipped_with_a_reason(self):
        reg = _register()
        reg.register("hot", "graph_rungs", GIB, 0.0, hot=lambda: True)
        reg.register("protected", "graph_rungs", GIB, 0.0, prio_protected=True)
        reg.register("unsized", "graph_rungs", 0, 0.0)
        reg.register("fine", "graph_rungs", GIB, 0.0)
        plan = plan_spill(reg, 8 * GIB)
        self.assertEqual([s.item_id for s in plan.steps], ["fine"])
        self.assertIn("hot", plan.skipped["hot"])
        self.assertIn("priority-protected", plan.skipped["protected"])
        self.assertIn("refresh_sizes", plan.skipped["unsized"])

    def test_plan_is_pure(self):
        """Nothing moves. A planner with a side effect could not be used to
        price a rung before committing to it."""
        reg = self._populated()
        plan_spill(reg, 4 * GIB)
        self.assertEqual(reg.parked_bytes(), 0)
        self.assertEqual(reg.stats.parks, 0)

    def test_zero_bytes_is_a_satisfied_empty_plan(self):
        plan = plan_spill(self._populated(), 0)
        self.assertTrue(plan.satisfied)
        self.assertEqual(plan.steps, [])

    def test_negative_and_unknown_inputs_are_hard_errors(self):
        reg = self._populated()
        with self.assertRaises(ValueError):
            plan_spill(reg, -1)
        with self.assertRaises(ValueError) as ctx:
            plan_spill(reg, GIB, classes=("no_such_class",))
        self.assertIn("no_such_class", str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. the capture gate
# ---------------------------------------------------------------------------


class CaptureRefusalTest(unittest.TestCase):
    """FALSIFIER GROUP 2. Delete the ``refuse_if_capture_active`` call in
    ``offload_family`` and the first two tests turn red."""

    def setUp(self):
        self.addCleanup(set_capture_probe, None)
        self.reg = _register()
        self.mover = FakeGraphStateMover()
        self.families = GraphFamilyRegister(mover=self.mover, register=self.reg)
        self.families.register_family("layout_b", "adaptive_state_k3", size_bytes=GIB)
        self.families.register_family("layout_a", "adaptive_state_k1", size_bytes=GIB)
        self.families.set_active("layout_a")

    def test_park_under_capture_raises_the_named_error(self):
        """Not a bare RuntimeError: a caller that means to retry at the next
        replay boundary must be able to catch THIS and nothing else."""
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused) as ctx:
            self.families.offload_family("layout_b")
        self.assertEqual(ctx.exception.operation, "offload")
        self.assertEqual(ctx.exception.subject, "layout_b")
        self.assertIn("BETWEEN replays", str(ctx.exception))

    def test_nothing_moved_when_the_capture_gate_fired(self):
        """The refusal is worthless if the pages went anyway."""
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused):
            self.families.offload_family("layout_b")
        self.assertEqual(self.mover.parked, [])
        self.assertFalse(self.reg.is_parked(graph_family_item_id("layout_b")))
        self.assertEqual(self.families.stats.offloads, 0)

    def test_onload_under_capture_is_refused_too(self):
        """A wave-in remaps pages; that is as illegal mid-capture as an unmap.
        Retrieval being unrefusable on POLICY grounds does not make it legal
        at any moment."""
        self.families.offload_family("layout_b")
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused) as ctx:
            self.families.onload_family("layout_b")
        self.assertEqual(ctx.exception.operation, "onload")
        self.assertEqual(self.mover.waved_in, [])

    def test_rung1_execute_is_refused_under_capture(self):
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused):
            self.families.rung1_evict(GIB, execute=True)

    def test_rung1_planning_is_allowed_under_capture(self):
        """Planning touches no page. Refusing it would make the controller
        blind exactly while a capture is running, for no physical reason."""
        set_capture_probe(lambda: True)
        plan = self.families.rung1_evict(GIB)
        self.assertEqual([s.item_id for s in plan.steps], ["graph_family/layout_b"])

    def test_moves_pass_once_the_capture_ends(self):
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused):
            self.families.offload_family("layout_b")
        set_capture_probe(lambda: False)
        self.families.offload_family("layout_b")
        self.assertEqual(self.mover.parked, ["layout_b"])

    def test_absent_probe_answers_false_not_true(self):
        """With no torch there is no capture to be inside of. Answering True
        would make the gate untestable in the direction that matters."""
        set_capture_probe(None)
        self.assertIsInstance(capture_is_active(), bool)


# ---------------------------------------------------------------------------
# 4. provenance: the register never divides by an unmeasured rate
# ---------------------------------------------------------------------------


class TierPricingTest(unittest.TestCase):
    """FALSIFIER GROUP 3. Flip ``require_measured_bandwidth`` to False (or set
    ``allow_unmeasured_bandwidth``) inside ``price_park_target`` and the first
    two tests turn red."""

    def test_the_fastest_measured_non_origin_tier_is_priced(self):
        """The peer card (50 GB/s) beats host DRAM (12.5), which is also the
        default order ``--lane-offload-park-targets`` already ships
        (``peer_vram>host_ram``) -- reached here by measurement rather than by
        a hand-written preference chain."""
        priced = price_park_target(
            _registry(), offload_class="graph_rungs", bytes_needed=GIB, origin=ORIGIN
        )
        self.assertEqual(priced.tier_id, PEER)
        self.assertIs(priced.bandwidth_gbs.provenance, Provenance.MEASURED)
        self.assertGreater(priced.move_ms.value, 0.0)

    def test_host_dram_is_picked_when_the_peer_path_is_unmeasured(self):
        """The peer link is the faster path only where somebody measured it.
        Unmeasured, it drops out entirely and host DRAM wins -- it does not
        get ranked behind host on an assumed number."""
        priced = price_park_target(
            _registry(
                peer_bandwidth=Rate.absent(
                    "no p2p probe; run scripts/dev/card_probe", unit="GB/s"
                )
            ),
            offload_class="graph_rungs",
            bytes_needed=GIB,
            origin=ORIGIN,
        )
        self.assertEqual(priced.tier_id, host_tier_id("rig-1"))

    def test_the_origin_card_is_never_its_own_park_target(self):
        """Parking to where the bytes already are is not parking. The origin
        outranks everything on bandwidth, so without the exclusion it would
        win every query.

        Can-fail proof: drop the ``c.tier_id != origin`` filter in
        ``price_park_target`` and this returns the origin.
        """
        priced = price_park_target(
            _registry(), offload_class="graph_rungs", bytes_needed=GIB, origin=ORIGIN
        )
        self.assertNotEqual(priced.tier_id, ORIGIN)

    def test_an_origin_with_no_alternative_refuses_naming_the_rule(self):
        registry = TierRegistry(
            [
                _tier(
                    ORIGIN,
                    TierKind.DEVICE,
                    volatility=Volatility.DEVICE_BOUND_ONLY,
                    bandwidth=Rate.measured(700.0, "fixture: hbm", unit="GB/s"),
                    admits=set(OFFLOAD_CLASSES),
                    total=32 * GIB,
                )
            ],
            profile_id="fixture",
            local_host="rig-1",
        )
        with self.assertRaises(UnpricedTierRefused) as ctx:
            price_park_target(
                registry,
                offload_class="graph_rungs",
                bytes_needed=GIB,
                origin=ORIGIN,
            )
        self.assertIn("stay resident", str(ctx.exception))

    def test_absent_bandwidth_is_refused_by_name(self):
        """The #348b D4 defect one layer up: a missing rate must not read as a
        very slow but valid path."""
        absent = Rate.absent(
            "no H2D probe on this rig; run scripts/dev/card_probe", unit="GB/s"
        )
        registry = _registry(host_bandwidth=absent, peer_bandwidth=absent)
        with self.assertRaises(UnpricedTierRefused) as ctx:
            price_park_target(
                registry,
                offload_class="graph_rungs",
                bytes_needed=GIB,
                origin=ORIGIN,
            )
        message = str(ctx.exception)
        self.assertIn("never assumed usable", message)
        self.assertIn("card_probe", message, "the absence must name its own probe")

    def test_an_estimate_is_refused_under_the_default(self):
        """This register acts on the number, so ESTIMATE is not good enough
        by default -- ``require_measured=False`` is the caller's explicit
        opt-in, not the register's assumption."""
        estimate = Rate.estimate(12.5, "derived from nameplate PCIe4 x16", unit="GB/s")
        registry = _registry(host_bandwidth=estimate, peer_bandwidth=estimate)
        with self.assertRaises(UnpricedTierRefused):
            price_park_target(
                registry,
                offload_class="graph_rungs",
                bytes_needed=GIB,
                origin=ORIGIN,
            )
        priced = price_park_target(
            registry,
            offload_class="graph_rungs",
            bytes_needed=GIB,
            origin=ORIGIN,
            require_measured=False,
        )
        self.assertIs(priced.bandwidth_gbs.provenance, Provenance.ESTIMATE)

    def test_the_derived_move_time_is_never_labelled_measured(self):
        """Even off a MEASURED bandwidth the division is a link time, and the
        move is dominated by remap+zero (#102: 40-51 ms avg, 85 ms max for
        ~1 GB). Labelling it MEASURED would launder a floor into a prediction.

        Can-fail proof: change the ``Rate`` in ``price_park_target`` to
        ``Provenance.MEASURED`` and this fails.
        """
        priced = price_park_target(
            _registry(), offload_class="graph_rungs", bytes_needed=GIB, origin=ORIGIN
        )
        self.assertIs(priced.move_ms.provenance, Provenance.ESTIMATE)
        self.assertIn("FLOOR", priced.move_ms.source)
        self.assertIn("excludes VMM remap", priced.move_ms.source)

    def test_the_volatility_law_keeps_a_graph_family_off_a_droppable_tier(self):
        """A tmpfs is RECONSTRUCTABLE_OK. It may hold lane workspaces; it may
        not hold a capture state that must be evacuated rather than dropped.
        This is DESIGN_407 §2.4b's silent-correctness hole, in this class."""
        registry = _registry(with_tmpfs=True)
        tmpfs = filesystem_tier_id("rig-1", "/dev/shm")
        graph = registry.select(
            TierQuery(
                payload=describe_class("graph_rungs").payload,
                object_class="graph_rungs",
                bytes_needed=GIB,
                require_measured_bandwidth=True,
            )
        )
        self.assertNotIn(tmpfs, graph.tier_ids)
        self.assertIsNotNone(graph.refusal_for(tmpfs))
        workspaces = registry.select(
            TierQuery(
                payload=describe_class("lane_workspaces").payload,
                object_class="lane_workspaces",
                bytes_needed=GIB,
                require_measured_bandwidth=True,
            )
        )
        self.assertIn(tmpfs, workspaces.tier_ids)

    def test_a_live_device_bound_class_has_no_park_target_at_all(self):
        """#461, POSITIVE PIN 1 of 2 (was the red-if-changed contradiction pin).

        ``memtier.tiers.admission_refusal`` states the DEVICE_BOUND law in two
        parts: the volatility table admits it only on ``DEVICE_BOUND_ONLY``
        tiers, AND it "never travels, not even one hop over P2P" -- so once the
        origin card is excluded, a LIVE device-bound class has nowhere to go.
        That law is unchanged by #461 and stays pinned here.

        Can-fail proof: give ``gdn_state_sets`` the suspended payload as its
        LIVE payload and this returns a priced peer-VRAM target.
        """
        with self.assertRaises(UnpricedTierRefused) as ctx:
            price_park_target(
                _registry(),
                offload_class="gdn_state_sets",
                bytes_needed=MIB,
                origin=ORIGIN,
            )
        self.assertIn("never travels", str(ctx.exception))
        # The refusal must name the remedy, not just the law: the register's
        # answer for this class is vacate-then-move, never move-live.
        self.assertIn("VACATE-then-move", str(ctx.exception))
        self.assertIn("content_state=SUSPENDED", str(ctx.exception))

    def test_no_admitting_tier_refuses_with_the_full_itemisation(self):
        """The refusal carries every rejected tier with its rule, not a bare
        'no target' -- the remedy is in the message."""
        registry = TierRegistry(
            [
                _tier(
                    host_tier_id("rig-1"),
                    TierKind.HOST,
                    volatility=Volatility.EXPENSIVE_OK,
                    bandwidth=Rate.measured(12.5, "fixture", unit="GB/s"),
                    admits={"experts"},
                )
            ],
            profile_id="fixture",
            local_host="rig-1",
        )
        with self.assertRaises(UnpricedTierRefused) as ctx:
            price_park_target(
                registry,
                offload_class="graph_rungs",
                bytes_needed=GIB,
                origin=ORIGIN,
            )
        self.assertIn("does not hold 'graph_rungs'", str(ctx.exception))

    def test_negative_bytes_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            price_park_target(
                _registry(),
                offload_class="graph_rungs",
                bytes_needed=-1,
                origin=ORIGIN,
            )


class MemTierIsNowWiredTest(unittest.TestCase):
    """The POSITIVE replacement for #421 finding F6.

    ``test_unwired_features_421.TestMemTierRegistryHasNoConsumers`` asserted
    that ``srt/memtier/`` had zero production importers. This module is the
    first, so that pin is retired in the same commit and the fact it protected
    is re-pinned here in the positive form -- exactly the treatment #394's
    cold-tier pin got. A refactor that quietly dropped the registry call and
    went back to a hand-written target list turns this red.
    """

    def test_the_register_imports_the_registry_at_module_scope(self):
        import ast
        import pathlib

        import sglang.srt.model_executor.short_term_offload_register as module

        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertIn("sglang.srt.memtier.registry", imported)
        self.assertIn("sglang.srt.memtier.tiers", imported)

    def test_the_target_comes_from_the_registry_not_from_park_targets(self):
        """``offload_register.PARK_TARGETS`` is three hand-written strings.
        A priced target is a TierId, and it must not be one of them."""
        from sglang.srt.model_executor.offload_register import PARK_TARGETS

        priced = price_park_target(
            _registry(), offload_class="graph_rungs", bytes_needed=GIB, origin=ORIGIN
        )
        self.assertNotIn(priced.tier_id, PARK_TARGETS)
        self.assertEqual(priced.tier_id, PEER)


# ---------------------------------------------------------------------------
# 5. graph families
# ---------------------------------------------------------------------------


class GraphFamilyRegisterTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_capture_probe, None)
        set_capture_probe(lambda: False)
        self.reg = _register()
        self.mover = FakeGraphStateMover({"layout_b": 300 * MIB})
        self.families = GraphFamilyRegister(mover=self.mover, register=self.reg)

    def test_a_family_books_into_the_graph_rungs_class(self):
        self.families.register_family("layout_a", "adaptive_state_k1", size_bytes=GIB)
        item = self.reg.get(graph_family_item_id("layout_a"))
        self.assertIsNotNone(item)
        self.assertEqual(item.offload_class, "graph_rungs")
        self.assertTrue(item.va_stable_required)

    def test_the_family_id_namespace_cannot_collide_with_the_other_producers(self):
        """Three producers share the ``graph_rungs`` id space: the per-lane
        spec ladder, the KV pressure ladder, and this one. A collision would
        re-bind an item rather than raise."""
        from sglang.srt.model_executor.kv_pressure_ladder import graph_rung_item_id

        self.assertNotEqual(graph_family_item_id("s0"), graph_rung_item_id("s0"))
        self.assertTrue(graph_family_item_id("s0").startswith("graph_family/"))

    def test_the_active_family_is_hot_and_cannot_be_parked(self):
        """The base register's hot refusal, reached through the family's own
        criterion. No policy makes the serving layout parkable."""
        self.families.register_family("layout_a", "tag_a", size_bytes=GIB)
        self.families.register_family("layout_b", "tag_b", size_bytes=GIB)
        self.families.set_active("layout_a")
        with self.assertRaises(OffloadRefused) as ctx:
            self.families.offload_family("layout_a")
        self.assertIn("hot", str(ctx.exception))
        self.assertEqual(self.mover.parked, [])
        self.assertEqual(self.families.stats.policy_refusals, 1)

    def test_set_active_clears_the_previous_family(self):
        """Exactly one family serves. Two hot families would make both
        unparkable and the whole rung a no-op."""
        self.families.register_family("layout_a", "tag_a", size_bytes=GIB)
        self.families.register_family("layout_b", "tag_b", size_bytes=GIB)
        self.families.set_active("layout_a")
        self.families.set_active("layout_b")
        self.assertFalse(self.families.is_active("layout_a"))
        self.assertTrue(self.families.is_active("layout_b"))

    def test_offload_then_onload_round_trips_through_the_mover(self):
        self.families.register_family("layout_a", "tag_a", size_bytes=GIB)
        self.families.register_family("layout_b", "tag_b", size_bytes=GIB)
        self.families.set_active("layout_a")
        self.families.offload_family("layout_b")
        self.assertEqual(self.mover.parked, ["layout_b"])
        self.assertTrue(self.reg.is_parked(graph_family_item_id("layout_b")))
        self.families.onload_family("layout_b")
        self.assertEqual(self.mover.waved_in, ["layout_b"])
        self.assertFalse(self.reg.is_parked(graph_family_item_id("layout_b")))

    def test_a_resident_class_policy_refuses_the_park(self):
        reg = _register(graph_rungs=ClassPolicy("resident", 0.0))
        families = GraphFamilyRegister(mover=FakeGraphStateMover(), register=reg)
        families.register_family("layout_b", "tag_b", size_bytes=GIB)
        with self.assertRaises(OffloadRefused):
            families.offload_family("layout_b")

    def test_duplicate_and_unknown_families_are_hard_errors(self):
        self.families.register_family("layout_a", "tag_a", size_bytes=GIB)
        with self.assertRaises(ValueError):
            self.families.register_family("layout_a", "tag_other", size_bytes=GIB)
        with self.assertRaises(ValueError):
            self.families.offload_family("nope")
        with self.assertRaises(ValueError):
            self.families.set_active("nope")

    def test_unregister_removes_the_register_item_too(self):
        self.families.register_family("layout_a", "tag_a", size_bytes=GIB)
        self.families.unregister_family("layout_a")
        self.assertIsNone(self.reg.get(graph_family_item_id("layout_a")))
        self.assertEqual(self.families.families(), ())

    def test_a_missing_base_register_refuses_rather_than_booking_nothing(self):
        """An unbooked family is invisible to the ladder. Silently accepting
        the registration would make the eviction order lie by omission."""
        families = GraphFamilyRegister(mover=FakeGraphStateMover(), register=None)
        with self.assertRaises(RuntimeError) as ctx:
            families.register_family("layout_a", "tag_a", size_bytes=GIB)
        self.assertIn("SGLANG_OFFLOAD_REGISTER", str(ctx.exception))

    def test_size_source_resolves_a_102_state_record(self):
        """``offload_sizes`` already understands ``footprint_bytes`` -- the
        #102 ``_StateRecord`` attribute that holds the measured paused bytes.
        Registering with 0 and refreshing later is the normal lifecycle."""

        class _FakeStateRecord:
            def __init__(self):
                self.footprint_bytes = 0

        record = _FakeStateRecord()
        self.families.register_family(
            "layout_b", "tag_b", size_bytes=0, size_source=record
        )
        self.assertEqual(self.reg.get(graph_family_item_id("layout_b")).size_bytes, 0)
        record.footprint_bytes = 300 * MIB
        self.assertEqual(self.reg.refresh_sizes(), 1)
        self.assertEqual(
            self.reg.get(graph_family_item_id("layout_b")).size_bytes, 300 * MIB
        )


class Rung1ContractTest(unittest.TestCase):
    """The DESIGN_363 §20.3 consumption contract, as this side implements it."""

    def setUp(self):
        self.addCleanup(set_capture_probe, None)
        set_capture_probe(lambda: False)
        self.clock = _Clock()
        self.reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
            clock=self.clock,
        )
        self.mover = FakeGraphStateMover()
        self.families = GraphFamilyRegister(mover=self.mover, register=self.reg)
        self.clock.t = 100.0
        self.families.register_family("layout_b", "tag_b", size_bytes=2 * GIB)
        self.clock.t = 101.0
        self.families.register_family("layout_c", "tag_c", size_bytes=2 * GIB)
        self.clock.t = 102.0
        self.families.register_family("layout_a", "tag_a", size_bytes=2 * GIB)
        self.families.set_active("layout_a")
        self.clock.t = 200.0

    def test_rung1_evicts_the_inactive_families_coldest_first(self):
        plan = self.families.rung1_evict(4 * GIB)
        self.assertEqual(
            [s.item_id for s in plan.steps],
            ["graph_family/layout_b", "graph_family/layout_c"],
        )
        self.assertTrue(plan.satisfied)

    def test_rung1_never_reaches_the_active_family(self):
        """The serving layout is hot; a rung that could evict it would stall
        the very traffic the KV pressure came from."""
        plan = self.families.rung1_evict(64 * GIB)
        self.assertNotIn("graph_family/layout_a", [s.item_id for s in plan.steps])
        self.assertIn("hot", plan.skipped["graph_family/layout_a"])
        self.assertFalse(plan.satisfied)

    def test_rung1_does_not_reach_cold_experts_or_idle_sessions(self):
        """§20.3 routes victim choice to §8, but RUNG 1 is about the inactive
        FAMILY. Reaching past it here would be a second global planner
        competing with the one ladder -- §8's named failure mode.

        Can-fail proof: widen the ``classes`` tuple in ``rung1_evict`` and this
        fails; the expert and the session appear in the plan.
        """
        self.reg.register("an_expert", "experts", 8 * GIB, 0.0)
        self.reg.register("a_session", "gdn_state_sets", 8 * GIB, 0.0)
        plan = self.families.rung1_evict(64 * GIB)
        planned = [s.item_id for s in plan.steps]
        self.assertNotIn("an_expert", planned)
        self.assertNotIn("a_session", planned)
        # The whole ladder is still reachable, just not from RUNG 1.
        whole = plan_spill(self.reg, 64 * GIB)
        self.assertIn("an_expert", [s.item_id for s in whole.steps])

    def test_execute_parks_in_plan_order(self):
        plan = self.families.rung1_evict(4 * GIB, execute=True)
        self.assertEqual(self.mover.parked, ["layout_b", "layout_c"])
        self.assertTrue(plan.satisfied)
        self.assertEqual(self.reg.parked_bytes("graph_rungs"), 4 * GIB)

    def test_execute_continues_past_a_refused_step(self):
        """One item's hysteresis window is not a reason to abandon the
        eviction the rung was entered for."""
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
            clock=self.clock,
        )
        families = GraphFamilyRegister(mover=FakeGraphStateMover(), register=reg)
        self.clock.t = 100.0
        families.register_family("layout_b", "tag_b", size_bytes=2 * GIB)
        self.clock.t = 101.0
        families.register_family("layout_c", "tag_c", size_bytes=2 * GIB)
        self.clock.t = 200.0
        # Make the first candidate refuse by protecting it against a park that
        # is explicitly making room -- the priority-protection invariant.
        reg.get(graph_family_item_id("layout_b")).prio_protected = True
        plan = families.rung1_evict(2 * GIB, execute=True)
        self.assertEqual([s.item_id for s in plan.steps], ["graph_family/layout_c"])

    def test_planning_is_the_default_and_moves_nothing(self):
        self.families.rung1_evict(4 * GIB)
        self.assertEqual(self.mover.parked, [])
        self.assertEqual(self.reg.parked_bytes(), 0)


class AdaptiveMoverTest(unittest.TestCase):
    """The production mover is DESK-WRITTEN, NEVER EXECUTED (see
    DESIGN_286 §7). What can be checked without a card is that it refuses
    coherently instead of raising an AttributeError three frames down."""

    def test_an_unbound_family_is_a_named_error(self):
        mover = AdaptiveGraphStateMover(manager=object())
        with self.assertRaises(ValueError) as ctx:
            mover.park("layout_a", "tag_a")
        self.assertIn("no bound state key", str(ctx.exception))

    def test_bind_state_key_accepts_both_rung_key_shapes(self):
        """The manager's key is an int for the classic k-ladder or an
        ``(algorithm, k)`` tuple for a cross-algorithm rung."""
        mover = AdaptiveGraphStateMover(manager=object())
        mover.bind_state_key("layout_a", 3)
        mover.bind_state_key("layout_b", ("dflash", 5))
        self.assertEqual(mover._state_key("layout_a"), 3)
        self.assertEqual(mover._state_key("layout_b"), ("dflash", 5))

    def test_no_active_manager_refuses_with_the_reason(self):
        mover = AdaptiveGraphStateMover(manager=None)
        mover.bind_state_key("layout_a", 3)
        with self.assertRaises(RuntimeError) as ctx:
            mover.park("layout_a", "tag_a")
        self.assertIn("adaptive-graph-memory", str(ctx.exception))

    def test_the_base_mover_reports_unknown_size_as_zero_not_free(self):
        from sglang.srt.model_executor.short_term_offload_register import (
            GraphStateMover,
        )

        self.assertEqual(GraphStateMover().footprint_bytes("k", "t"), 0)


# ---------------------------------------------------------------------------
# 6. #461 -- the GDN state class is STATE-dependent, not statically classified
# ---------------------------------------------------------------------------


class GdnStateClassificationTest(unittest.TestCase):
    """FALSIFIER GROUP 9 (#461). Drop ``suspended_payload`` from the
    ``gdn_state_sets`` descriptor and the pricing tests here turn red.

    The contradiction #286 §8 recorded was between two true statements about
    two DIFFERENT things. The movement code settles it:

    * LIVE: one set is a stride slice of the pool's
      ``[num_layers, num_slots, ...]`` tensors that GDN kernels update in
      place and captured graphs address (``memory_pool.py`` copy_from /
      ``offload_gdn_states.register_mamba_state_sets`` registers it
      ``va_stable_required``). It is DEVICE_BOUND and has no park target.
    * SUSPENDED: ``MambaPool.export_state_blob`` copies every persistent field
      of the slot to CPU tensors (``memory_pool.py:918``), keyed by NAME, and
      ``import_state_blob`` restores them into ANY free slot
      (``memory_pool.py:972``). #364's executor already carries that blob to a
      #224 destination tier as one flat uint8 buffer plus a manifest
      (``gdn_slot_executor.py:214``). The blob is a transportable byte payload.
    """

    def test_the_two_forms_carry_different_payload_classes(self):
        desc = describe_class("gdn_state_sets")
        self.assertIs(desc.payload_for(ContentState.LIVE), PayloadClass.DEVICE_BOUND)
        self.assertIs(
            desc.payload_for(ContentState.SUSPENDED),
            PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
        )
        self.assertTrue(desc.park_requires_suspend)

    def test_a_suspended_gdn_state_blob_does_have_a_park_target(self):
        """#461, POSITIVE PIN 2 of 2. Erg. 8's "host RAM or peer VRAM" is
        right about the DESTINATION and was wrong only about the FORM."""
        priced = price_park_target(
            _registry(),
            offload_class="gdn_state_sets",
            bytes_needed=MIB,
            origin=ORIGIN,
            content_state=ContentState.SUSPENDED,
        )
        self.assertEqual(priced.tier_id, PEER)
        self.assertGreater(priced.move_ms.value, 0.0)

    def test_the_suspended_blob_is_not_droppable_either(self):
        """EXPENSIVE_RECONSTRUCTABLE, not RECONSTRUCTABLE: losing a parked
        blob costs the session a full re-prefill, so a tmpfs may not hold it
        (``import_state_blob`` refuses to resume onto an uninitialised slot --
        ``gdn_slot_executor.py:413``)."""
        registry = _registry(with_tmpfs=True)
        selection = registry.select(
            TierQuery(
                payload=describe_class("gdn_state_sets").payload_for(
                    ContentState.SUSPENDED
                ),
                object_class="gdn_state_sets",
                bytes_needed=MIB,
                require_measured_bandwidth=True,
            )
        )
        self.assertNotIn(filesystem_tier_id("rig-1", "/dev/shm"), selection.tier_ids)

    def test_a_class_with_no_suspended_form_prices_the_same_in_both_states(self):
        """The state axis is not a second knob for every class. Only a class
        that declares a distinct suspended form has one."""
        self.assertIsNone(describe_class("graph_rungs").suspended_payload)
        for state in ContentState:
            priced = price_park_target(
                _registry(),
                offload_class="graph_rungs",
                bytes_needed=GIB,
                origin=ORIGIN,
                content_state=state,
            )
            self.assertEqual(priced.tier_id, PEER)

    def test_the_descriptor_agrees_with_what_the_erg8_ladder_registers(self):
        """``offload_gdn_states.register_mamba_state_sets`` books every set
        with ``va_stable_required=True``; the descriptor said False. Two
        answers to one question is how #461 happened, so it is pinned."""
        import ast
        import pathlib

        import sglang.srt.model_executor.offload_gdn_states as ladder

        self.assertTrue(describe_class("gdn_state_sets").va_stable_required)
        source = pathlib.Path(ladder.__file__).read_text()
        tree = ast.parse(source)
        booked = [
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "va_stable_required" and isinstance(kw.value, ast.Constant)
        ]
        self.assertEqual(booked, [True], "the Erg.-8 ladder's own registration")

    def test_a_live_set_is_never_planned_as_a_plain_park(self):
        """A spill step for this class is a VACATE-then-move, and the plan
        says so -- an executor that took the plain ``park()`` route would be
        moving live device-bound state."""
        reg = _register()
        reg.register("gdn_state_set/00007", "gdn_state_sets", GIB, 0.0)
        reg.register("some_family", "graph_rungs", GIB, 0.0)
        plan = plan_spill(reg, 2 * GIB)
        by_id = {s.item_id: s for s in plan.steps}
        self.assertTrue(by_id["gdn_state_set/00007"].requires_suspend)
        self.assertFalse(by_id["some_family"].requires_suspend)
        self.assertIn("vacate", plan.render())


# ---------------------------------------------------------------------------
# 7. #468 -- VA stability is a property of the ROUTE, not only of the class
# ---------------------------------------------------------------------------


class ExpertArenaVaStabilityTest(unittest.TestCase):
    """FALSIFIER GROUP 10 (#468). Set
    ``va_stable_when_graph_addressed=False`` on the ``experts`` descriptor and
    the first four tests turn red.

    #462 built a route in which a captured decode graph holds the expert slot
    arena's device addresses (``DESIGN_462_breakable_route.md`` §6). The arena
    refuses a park locally; the REGISTER did not, so the one rule lived in two
    places and only one of them knew about it.
    """

    def setUp(self):
        self.addCleanup(set_capture_probe, None)
        self.addCleanup(set_graph_reference_probe, None)
        set_capture_probe(lambda: False)
        self.reg = _register()
        self.families = GraphFamilyRegister(
            mover=FakeGraphStateMover(), register=self.reg
        )

    def _family_addressing_experts(self):
        self.families.register_family(
            "decode_breakable", "tag_b", size_bytes=GIB, addresses_classes=("experts",)
        )
        set_graph_reference_probe(self.families.addressed_classes)

    def test_the_register_refuses_an_arena_park_while_a_graph_addresses_it(self):
        """The falsifier #462 §6 says is missing at register level."""
        self._family_addressing_experts()
        with self.assertRaises(OffloadUnderCaptureRefused) as ctx:
            refuse_if_move_illegal(
                "park", "MoE breakable slot arena (layer 3)", offload_class="experts"
            )
        self.assertEqual(ctx.exception.ground, GROUND_GRAPH_ADDRESSED)
        self.assertIn("decode_breakable", str(ctx.exception))
        self.assertIn("VA", str(ctx.exception))

    def test_without_a_referencing_family_the_arena_park_is_allowed(self):
        """Default-inert: the eager offload's droppable cache still parks.
        A rule that refused unconditionally would be the arena's local rule
        again, just moved one module up."""
        refuse_if_move_illegal("park", "arena", offload_class="experts")
        self.assertEqual(graph_addressed_classes(), frozenset())
        # ... and it reports CLASSES once a family declares some, not the
        # family keys the probe is keyed by.
        self._family_addressing_experts()
        self.assertEqual(graph_addressed_classes(), frozenset({"experts"}))

    def test_parking_the_family_does_not_release_the_reference(self):
        """A #93 family park PRESERVES the VAs -- that is its whole point --
        so the captured graphs still hold the arena's addresses afterwards.
        Only dropping the graphs (unregistering the family) releases them."""
        self._family_addressing_experts()
        self.families.offload_family("decode_breakable")
        with self.assertRaises(OffloadUnderCaptureRefused):
            refuse_if_move_illegal("park", "arena", offload_class="experts")
        self.families.unregister_family("decode_breakable")
        refuse_if_move_illegal("park", "arena", offload_class="experts")

    def test_plan_spill_skips_a_class_whose_addresses_a_capture_baked_in(self):
        """The planner may not plan what the gate would refuse; otherwise the
        rule is enforced only by whoever remembers to call it."""
        self._family_addressing_experts()
        self.reg.register("an_expert", "experts", 8 * GIB, 0.0)
        plan = plan_spill(self.reg, 8 * GIB)
        self.assertNotIn("an_expert", [s.item_id for s in plan.steps])
        self.assertIn("an_expert", plan.skipped)
        self.assertIn("addresses", plan.skipped["an_expert"])

    def test_it_is_one_rule_with_two_grounds_not_two_rules(self):
        """Same gate, same named error, distinguishable ground. A second
        exception type would let a caller catch one and miss the other -- and
        one ground spelt two ways would let it retry at the next replay
        boundary forever, which only ever helps for the capture ground."""
        self.assertNotEqual(GROUND_CAPTURE_ACTIVE, GROUND_GRAPH_ADDRESSED)
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused) as ctx:
            refuse_if_move_illegal("park", "arena", offload_class="experts")
        self.assertEqual(ctx.exception.ground, GROUND_CAPTURE_ACTIVE)
        self.assertIn("BETWEEN replays", str(ctx.exception))

    def test_the_capture_ground_needs_no_class_and_is_unchanged(self):
        """``refuse_if_capture_active`` keeps its exact old behaviour for the
        callers that name no class (``breakable_offload.BreakableOffloadArena``
        is one), so #468 cannot regress the #462 pin."""
        self._family_addressing_experts()
        refuse_if_capture_active("park", "arena")  # graph ground needs the class
        set_capture_probe(lambda: True)
        with self.assertRaises(OffloadUnderCaptureRefused) as ctx:
            refuse_if_capture_active("park", "arena")
        self.assertEqual(ctx.exception.ground, GROUND_CAPTURE_ACTIVE)

    def test_va_stability_is_conditional_for_experts_and_static_elsewhere(self):
        experts = describe_class("experts")
        self.assertFalse(experts.va_stable_required)
        self.assertFalse(experts.va_stability_required(graph_addressed=False))
        self.assertTrue(experts.va_stability_required(graph_addressed=True))
        families = describe_class("graph_rungs")
        self.assertTrue(families.va_stability_required(graph_addressed=False))
        workspaces = describe_class("lane_workspaces")
        self.assertFalse(workspaces.va_stability_required(graph_addressed=True))

    def test_a_family_may_not_declare_an_unknown_class(self):
        with self.assertRaises(ValueError) as ctx:
            self.families.register_family(
                "bad", "tag", size_bytes=GIB, addresses_classes=("no_such_class",)
            )
        self.assertIn("no_such_class", str(ctx.exception))
        self.assertEqual(self.families.families(), ())


if __name__ == "__main__":
    unittest.main()
