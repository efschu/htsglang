"""The engine registry: derived M, eviction policy, informative rejection.

Hermetic. Every card is a mock with a declared total, and every engine is a
fake adapter that records the states it was asked for and allocates nothing.
That is deliberate: the policy in ``arbiter.py`` is the part that decides
whether a real boot is even attempted, so it has to be testable without one.
The adapters that *do* boot are exercised in the card window, not here.

    python -m pytest test/registered/registry/test_registry.py -v
"""

import tempfile
import unittest
from pathlib import Path

from sglang.srt.registry.adapter import (
    AdapterError,
    EstimateError,
    register_adapter,
)
from sglang.srt.registry.arbiter import (
    DEFAULT_PROMOTION_COST_MS,
    EngineRegistry,
    PromotionRejected,
    RegistrationRejected,
    UnknownEngineError,
)
from sglang.srt.registry.ledger import MIB, ReservationStore, TenantState
from sglang.srt.registry.spec import (
    EngineClass,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
    SpecError,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

GIB = 1024 * MIB
CARD_3080_A = "GPU-3080aaaa-0000-0000-0000-00000000000a"
CARD_3080_B = "GPU-3080bbbb-0000-0000-0000-00000000000b"
CARD_5090 = "GPU-5090cccc-0000-0000-0000-00000000000c"

#: The reference rig, by NVML totals rather than by name: two 20 GiB cards and
#: one 32 GiB card. Unlike cards are the whole point of deriving M from bytes.
RIG = {CARD_3080_A: 20 * GIB, CARD_3080_B: 20 * GIB, CARD_5090: 32 * GIB}


class FakeAdapter:
    """Costs what its spec declares; records residency; never touches a device."""

    klass = 1

    def __init__(self, spec, context):
        self.spec = spec
        self.context = context
        self._state = ResidencyState.COLD
        self._cards = ()
        self.history = []
        self.fail_promote = bool(spec.launch.get("fail_promote", False))
        if spec.launch.get("uncostable"):
            raise EstimateError(f"engine {spec.engine_id!r} declares no budget")

    def estimate(self, spec, cards):
        per_card = int(spec.launch["mib_per_card"]) * MIB
        return ResourceProfile(
            posts={c: {"declared": per_card} for c in cards},
            peak_bytes={c: per_card for c in cards},
            steady_bytes={c: per_card // 2 for c in cards},
        )

    def bind(self, cards):
        self._cards = tuple(cards)

    def state(self):
        return self._state

    def pids(self):
        return (4242,) if self._state != ResidencyState.COLD else ()

    def promote(self, target):
        if self.fail_promote:
            raise AdapterError("boom")
        self.history.append(("promote", target))
        self._state = target

    def demote(self, target):
        self.history.append(("demote", target))
        self._state = target

    def measured(self):
        if self._state == ResidencyState.COLD:
            return {}
        return {
            c: int(self.spec.launch["mib_per_card"]) * MIB // 2 for c in self._cards
        }

    def health(self):
        from sglang.srt.registry.adapter import Health

        return Health(ok=True, detail="fake")


register_adapter("fake", FakeAdapter)


def spec(
    engine_id, mib_per_card, cards, *, klass=1, pinned=False, priority=0, **launch
):
    return EngineSpec(
        engine_id=engine_id,
        klass=EngineClass(klass),
        adapter="fake",
        placement=tuple(cards),
        pinned=pinned,
        priority=priority,
        launch={"mib_per_card": mib_per_card, **launch},
    )


class RegistryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.now = 1_000_000.0
        self.store = ReservationStore(
            self.root,
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
        )
        self.registry = EngineRegistry(
            store=self.store,
            card_totals=RIG,
            clock=lambda: self.now,
        )
        self.addCleanup(self.registry.shutdown)

    def advance(self, seconds):
        self.now += seconds


class SpecValidationTest(RegistryTestCase):
    def test_engine_id_must_be_a_safe_name(self):
        for bad in ("", "../escape", "a/b", "x" * 65):
            with self.assertRaises(SpecError, msg=bad):
                EngineSpec(engine_id=bad, klass=1, adapter="fake")

    def test_auto_placement_cannot_be_mixed_with_explicit_cards(self):
        with self.assertRaises(SpecError):
            EngineSpec(
                engine_id="e",
                klass=1,
                adapter="fake",
                placement=("auto", CARD_5090),
            )

    def test_duplicate_card_in_placement_is_refused(self):
        with self.assertRaises(SpecError):
            EngineSpec(
                engine_id="e",
                klass=1,
                adapter="fake",
                placement=(CARD_5090, CARD_5090),
            )

    def test_unknown_field_is_an_error_not_a_silent_drop(self):
        with self.assertRaises(SpecError) as ctx:
            EngineSpec.from_json(
                {"engine_id": "e", "klass": 1, "adapter": "fake", "pined": True}
            )
        self.assertIn("pined", str(ctx.exception))

    def test_spec_round_trips_through_json(self):
        original = spec("e", 4096, [CARD_5090], priority=3)
        self.assertEqual(EngineSpec.from_json(original.to_json()), original)


class RegistrationTest(RegistryTestCase):
    def test_registration_does_not_boot(self):
        self.registry.register(spec("a", 8192, [CARD_5090]))
        instance = self.registry.instance("a")
        self.assertEqual(instance.state, ResidencyState.COLD)
        self.assertEqual(self.registry.adapter("a").history, [])
        # And nothing is reserved: a registered engine holds no bytes.
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 0)

    def test_a_spec_that_cannot_fit_an_empty_rig_is_refused_immediately(self):
        with self.assertRaises(RegistrationRejected) as ctx:
            self.registry.register(spec("too-big", 40 * 1024, [CARD_5090]))
        message = str(ctx.exception)
        self.assertIn("too-big", message)
        self.assertIn(CARD_5090, message)
        self.assertIn("NVML total 32768 MiB", message)

    def test_a_spec_that_merely_does_not_fit_right_now_is_still_registrable(self):
        self.registry.register(spec("incumbent", 30 * 1024, [CARD_5090]))
        self.registry.ensure_state("incumbent", ResidencyState.HOT)
        result = self.registry.register(spec("later", 20 * 1024, [CARD_5090]))
        self.assertFalse(result.fits)
        self.assertIn("does not fit now", result.reason)

    def test_placement_on_an_unknown_card_names_the_known_ones(self):
        with self.assertRaises(RegistrationRejected) as ctx:
            self.registry.register(spec("a", 1024, ["GPU-not-a-card"]))
        self.assertIn("GPU-not-a-card", str(ctx.exception))
        self.assertIn(CARD_5090, str(ctx.exception))

    def test_uncostable_spec_fails_at_registration(self):
        with self.assertRaises(EstimateError):
            self.registry.register(
                EngineSpec(
                    engine_id="u",
                    klass=1,
                    adapter="fake",
                    placement=(CARD_5090,),
                    launch={"uncostable": True},
                )
            )

    def test_duplicate_registration_needs_replace(self):
        self.registry.register(spec("a", 1024, [CARD_5090]))
        with self.assertRaises(RegistrationRejected):
            self.registry.register(spec("a", 2048, [CARD_5090]))
        self.registry.register(spec("a", 2048, [CARD_5090]), replace=True)
        self.assertEqual(
            self.registry.instance("a").profile.total_peak_bytes, 2048 * MIB
        )

    def test_auto_placement_picks_the_card_with_the_most_room(self):
        self.registry.register(spec("auto", 16 * 1024, ["auto"]))
        self.assertEqual(self.registry.instance("auto").cards, (CARD_5090,))

    def test_auto_placement_moves_on_when_the_biggest_card_is_taken(self):
        self.registry.register(spec("big", 30 * 1024, [CARD_5090]))
        self.registry.ensure_state("big", ResidencyState.HOT)
        self.registry.register(spec("auto", 16 * 1024, ["auto"]))
        self.assertIn(
            self.registry.instance("auto").cards[0], (CARD_3080_A, CARD_3080_B)
        )


class LifecycleTest(RegistryTestCase):
    def test_add_promote_demote_remove(self):
        self.registry.register(spec("a", 8192, [CARD_5090]))
        self.registry.ensure_state("a", ResidencyState.HOT)
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 8192 * MIB)
        self.assertEqual(self.store.read(CARD_5090).tenant("a").state, TenantState.HOT)

        self.registry.ensure_state("a", ResidencyState.WARM_GPU)
        self.assertEqual(
            self.store.read(CARD_5090).tenant("a").state, TenantState.WARM_GPU
        )
        # WARM_GPU still holds device bytes, so it still counts.
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 8192 * MIB)

        self.registry.ensure_state("a", ResidencyState.COLD)
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 0)
        self.assertIsNone(self.store.read(CARD_5090).tenant("a"))

        self.registry.deregister("a")
        with self.assertRaises(UnknownEngineError):
            self.registry.instance("a")

    def test_deregister_releases_a_hot_engine(self):
        self.registry.register(spec("a", 8192, [CARD_5090]))
        self.registry.ensure_state("a", ResidencyState.HOT)
        self.registry.deregister("a")
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 0)

    def test_measured_bytes_reach_the_ledger_and_produce_waste(self):
        self.registry.register(spec("a", 8192, [CARD_5090]))
        self.registry.ensure_state("a", ResidencyState.HOT)
        self.registry.refresh_measured()
        card = next(c for c in self.registry.cards() if c.card_uuid == CARD_5090)
        self.assertEqual(card.reserved_bytes, 8192 * MIB)
        self.assertEqual(card.measured_bytes, 4096 * MIB)
        self.assertEqual(card.waste_bytes, 4096 * MIB)

    def test_a_failed_promotion_gives_the_bytes_straight_back(self):
        self.registry.register(spec("bad", 8192, [CARD_5090], fail_promote=True))
        with self.assertRaises(AdapterError):
            self.registry.ensure_state("bad", ResidencyState.HOT)
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 0)
        self.assertEqual(self.registry.instance("bad").state, ResidencyState.COLD)

    def test_promotion_cost_is_measured_not_assumed(self):
        self.registry.register(spec("a", 1024, [CARD_5090]))
        self.assertIsNone(self.registry.instance("a").promotion_cost_ms)
        self.registry.ensure_state("a", ResidencyState.HOT)
        self.assertIsNotNone(self.registry.instance("a").promotion_cost_ms)

    def test_multi_card_engine_holds_one_slot_per_card(self):
        self.registry.register(spec("tp2", 15 * 1024, [CARD_3080_A, CARD_3080_B]))
        self.registry.ensure_state("tp2", ResidencyState.HOT)
        slots = {s.card_uuid: s for s in self.registry.slots()}
        self.assertEqual(set(slots), {CARD_3080_A, CARD_3080_B})
        self.assertEqual(slots[CARD_3080_A].reserved_bytes, 15 * GIB)


class DerivedHotCapacityTest(RegistryTestCase):
    """§7.2: M comes out of the ledger, it is not configured."""

    def test_m_is_derived_from_bytes_on_unlike_cards(self):
        # Three 16 GiB engines. The 32 GiB card takes two; a 20 GiB card takes
        # one. A count-based cap could not express that.
        self.registry.register(spec("big-1", 16 * 1024, [CARD_5090]))
        self.registry.register(spec("big-2", 16 * 1024, [CARD_5090]))
        self.registry.register(spec("big-3", 16 * 1024, [CARD_5090]))
        capacity = self.registry.hot_capacity()
        self.assertEqual(capacity.count, 1)
        self.assertEqual(capacity.engine_ids, ("big-1",))
        excluded = dict(capacity.excluded)
        self.assertIn("big-2", excluded)
        self.assertIn("corridor", excluded["big-2"])

    def test_the_same_engines_on_a_bigger_card_raise_m(self):
        self.registry.register(spec("a", 15 * 1024, [CARD_5090]))
        self.registry.register(spec("b", 15 * 1024, [CARD_5090]))
        self.assertEqual(self.registry.hot_capacity().count, 2)

    def test_m_counts_per_card_not_per_rig(self):
        self.registry.register(spec("a", 19 * 1024, [CARD_3080_A]))
        self.registry.register(spec("b", 19 * 1024, [CARD_3080_B]))
        self.registry.register(spec("c", 31 * 1024, [CARD_5090]))
        self.assertEqual(self.registry.hot_capacity().count, 3)

    def test_max_hot_caps_after_the_derivation_never_instead_of_it(self):
        self.registry.register(spec("a", 1024, [CARD_5090]))
        self.registry.register(spec("b", 1024, [CARD_5090]))
        self.registry.register(spec("c", 1024, [CARD_5090]))
        self.assertEqual(self.registry.hot_capacity().count, 3)
        self.registry.max_hot = 2
        capped = self.registry.hot_capacity()
        self.assertEqual(capped.count, 2)
        self.assertTrue(capped.capped_by_max_hot)
        self.assertIn("registry-max-hot", dict(capped.excluded)["c"])

    def test_pinned_and_default_hot_engines_are_derived_first(self):
        self.registry.register(spec("plain", 16 * 1024, [CARD_5090], priority=9))
        self.registry.register(spec("pinned", 16 * 1024, [CARD_5090], pinned=True))
        self.assertEqual(self.registry.hot_capacity().engine_ids, ("pinned",))

    def test_capacity_is_a_question_about_an_empty_rig(self):
        """Whoever is hot right now must not change what *fits*."""
        self.registry.register(spec("a", 16 * 1024, [CARD_5090]))
        self.registry.register(spec("b", 15 * 1024, [CARD_5090]))
        before = self.registry.hot_capacity()
        self.registry.ensure_state("a", ResidencyState.HOT)
        self.assertEqual(self.registry.hot_capacity().count, before.count)


class DefaultHotTest(RegistryTestCase):
    """§7.3: the resting set, validated at registration, not at idle."""

    def test_an_unsatisfiable_default_set_is_refused_when_it_is_declared(self):
        self.registry.register(spec("a", 20 * 1024, [CARD_5090]))
        self.registry.register(spec("b", 20 * 1024, [CARD_5090]))
        with self.assertRaises(RegistrationRejected) as ctx:
            self.registry.set_default_hot(["a", "b"])
        self.assertIn("could never return to it", str(ctx.exception))

    def test_default_hot_names_must_exist(self):
        with self.assertRaises(RegistrationRejected):
            self.registry.set_default_hot(["ghost"])

    def test_a_satisfiable_default_set_is_accepted_and_reachable(self):
        self.registry.register(spec("a", 8 * 1024, [CARD_5090]))
        self.registry.register(spec("b", 8 * 1024, [CARD_5090]))
        self.registry.set_default_hot(["a", "b"])
        changed = self.registry.return_to_idle(force=True)
        self.assertEqual(sorted(changed), ["a", "b"])
        self.assertEqual(self.registry.instance("a").state, ResidencyState.HOT)

    def test_returning_to_idle_demotes_everything_outside_the_set(self):
        self.registry.register(spec("keep", 8 * 1024, [CARD_5090]))
        self.registry.register(spec("drop", 8 * 1024, [CARD_5090]))
        self.registry.set_default_hot(["keep"])
        self.registry.ensure_state("drop", ResidencyState.HOT)
        self.registry.return_to_idle(force=True)
        self.assertEqual(self.registry.instance("drop").state, ResidencyState.COLD)
        self.assertEqual(self.registry.instance("keep").state, ResidencyState.HOT)

    def test_idle_waits_for_the_idle_window(self):
        self.registry.register(spec("a", 1024, [CARD_5090]))
        self.registry.ensure_state("a", ResidencyState.HOT)
        self.assertEqual(self.registry.return_to_idle(), [])
        self.advance(self.registry.idle_after_s + 1)
        self.assertEqual(self.registry.return_to_idle(), ["a"])

    def test_a_deregistered_engine_leaves_the_default_set(self):
        self.registry.register(spec("a", 1024, [CARD_5090]))
        self.registry.set_default_hot(["a"])
        self.registry.deregister("a")
        self.assertEqual(self.registry.default_hot, ())


class AdmissionAndEvictionTest(RegistryTestCase):
    """§7.5, including the part that matters most: rejection is informative."""

    def setUp(self):
        super().setUp()
        self.registry.register(spec("incumbent", 20 * 1024, [CARD_5090], priority=1))
        self.registry.register(spec("newcomer", 20 * 1024, [CARD_5090], priority=5))
        self.registry.ensure_state("incumbent", ResidencyState.HOT)

    def test_promotion_evicts_the_lowest_priority_candidate(self):
        self.registry.ensure_state("newcomer", ResidencyState.HOT)
        self.assertEqual(self.registry.instance("incumbent").state, ResidencyState.COLD)
        self.assertEqual(self.registry.instance("newcomer").state, ResidencyState.HOT)

    def test_a_pinned_engine_is_never_evicted_automatically(self):
        object.__setattr__(self.registry.instance("incumbent").spec, "pinned", True)
        with self.assertRaises(PromotionRejected) as ctx:
            self.registry.ensure_state("newcomer", ResidencyState.HOT)
        self.assertIn("pinned", str(ctx.exception))
        self.assertEqual(self.registry.instance("incumbent").state, ResidencyState.HOT)

    def test_rejection_carries_the_projected_wait_and_the_eviction(self):
        with self.assertRaises(PromotionRejected) as ctx:
            self.registry.ensure_state(
                "newcomer", ResidencyState.HOT, max_promotion_wait_ms=1.0
            )
        rejection = ctx.exception
        self.assertEqual(rejection.would_evict, ("incumbent",))
        self.assertEqual(
            rejection.projected_wait_ms,
            DEFAULT_PROMOTION_COST_MS[1] + 15_000.0,
        )
        self.assertTrue(rejection.cost_is_estimated)
        payload = rejection.to_json()
        self.assertEqual(payload["would_evict"], ["incumbent"])
        self.assertEqual(payload["shortfalls"][0]["card_uuid"], CARD_5090)
        self.assertGreater(payload["shortfalls"][0]["shortfall_bytes"], 0)
        # And it is a real sentence, not a code.
        self.assertIn("exceeds the caller's budget", str(rejection))

    def test_a_measured_cost_replaces_the_estimate_in_the_rejection(self):
        self.registry.instance("newcomer").observe_promotion(1234.0)
        self.registry.instance("incumbent").observe_demotion(56.0)
        with self.assertRaises(PromotionRejected) as ctx:
            self.registry.ensure_state(
                "newcomer", ResidencyState.HOT, max_promotion_wait_ms=1.0
            )
        self.assertFalse(ctx.exception.cost_is_estimated)
        self.assertEqual(ctx.exception.projected_wait_ms, 1290.0)

    def test_eviction_prefers_a_candidate_outside_the_default_set(self):
        # 3 + 6 held on a 20 GiB card; the newcomer wants 14. Evicting the
        # casual tenant alone is enough (3 + 14 + 0.4 <= 20), so the resting
        # engine must survive even though it is the older one.
        self.registry.register(spec("resting", 3 * 1024, [CARD_3080_A]))
        self.registry.register(spec("casual", 6 * 1024, [CARD_3080_A], priority=-5))
        self.registry.register(spec("wants-in", 14 * 1024, [CARD_3080_A]))
        self.registry.set_default_hot(["resting"])
        self.registry.ensure_state("resting", ResidencyState.HOT)
        self.registry.ensure_state("casual", ResidencyState.HOT)
        self.registry.ensure_state("wants-in", ResidencyState.HOT)
        self.assertEqual(self.registry.instance("casual").state, ResidencyState.COLD)
        self.assertEqual(self.registry.instance("resting").state, ResidencyState.HOT)

    def test_least_recently_used_breaks_a_priority_tie(self):
        self.registry.register(spec("old", 9 * 1024, [CARD_3080_B]))
        self.registry.register(spec("new", 9 * 1024, [CARD_3080_B]))
        self.registry.register(spec("wants-in", 15 * 1024, [CARD_3080_B]))
        self.registry.ensure_state("old", ResidencyState.HOT)
        self.advance(60.0)
        self.registry.ensure_state("new", ResidencyState.HOT)
        self.registry.ensure_state("wants-in", ResidencyState.HOT)
        self.assertEqual(self.registry.instance("old").state, ResidencyState.COLD)

    def test_allow_eviction_false_refuses_without_touching_anybody(self):
        with self.assertRaises(PromotionRejected):
            self.registry.ensure_state(
                "newcomer", ResidencyState.HOT, allow_eviction=False
            )
        self.assertEqual(self.registry.instance("incumbent").state, ResidencyState.HOT)

    def test_thrash_between_two_engines_is_named(self):
        for _ in range(3):
            self.registry.ensure_state("newcomer", ResidencyState.HOT)
            self.registry.ensure_state("incumbent", ResidencyState.HOT)
        self.assertTrue(self.registry.thrash_events)
        event = self.registry.thrash_events[0]
        self.assertEqual(sorted(event["engines"]), ["incumbent", "newcomer"])


class PlanTest(RegistryTestCase):
    """§7.4's dry run: an answer in milliseconds, without a GPU window."""

    def test_plan_for_a_spec_that_is_not_registered_registers_nothing(self):
        result = self.registry.plan(spec("hypothetical", 8 * 1024, [CARD_5090]))
        self.assertTrue(result.fits)
        self.assertEqual(self.registry.engines(), ())
        self.assertEqual(self.store.read(CARD_5090).reserved_bytes, 0)

    def test_plan_names_the_eviction_it_would_perform(self):
        self.registry.register(spec("incumbent", 20 * 1024, [CARD_5090]))
        self.registry.ensure_state("incumbent", ResidencyState.HOT)
        result = self.registry.plan(spec("newcomer", 20 * 1024, [CARD_5090]))
        self.assertTrue(result.fits)
        self.assertEqual(result.would_evict, ("incumbent",))
        self.assertGreater(result.projected_wait_ms, 0)
        self.assertIn("fits after eviction", result.reason)

    def test_plan_says_so_when_nothing_can_be_evicted(self):
        self.registry.register(spec("pinned", 20 * 1024, [CARD_5090], pinned=True))
        self.registry.ensure_state("pinned", ResidencyState.HOT)
        result = self.registry.plan(spec("newcomer", 20 * 1024, [CARD_5090]))
        self.assertFalse(result.fits)
        self.assertEqual(result.would_evict, ())
        self.assertIn("does not fit even after", result.reason)
        self.assertIn(CARD_5090, result.to_json()["shortfall_detail"])

    def test_plan_reports_the_slack_the_reservation_carries(self):
        """#287/#330: reservation is the peak; the gap to steady is a posten."""
        self.registry.register(spec("a", 8 * 1024, [CARD_5090]))
        profile = self.registry.instance("a").profile
        self.assertEqual(profile.slack_bytes()[CARD_5090], 4 * GIB)


class CorridorTest(RegistryTestCase):
    """#330: the 400 MiB corridor belongs to the card, not to a tenant."""

    def test_no_tenant_may_eat_the_corridor(self):
        # 32368 + 300 = 32668 MiB would fit a 32768 MiB card if the corridor
        # were negotiable. It is not: it belongs to the card.
        self.registry.register(spec("a", 32 * 1024 - 400, [CARD_5090]))
        self.registry.register(spec("b", 300, [CARD_5090]))
        self.registry.ensure_state("a", ResidencyState.HOT)
        with self.assertRaises(PromotionRejected) as ctx:
            self.registry.ensure_state("b", ResidencyState.HOT, allow_eviction=False)
        self.assertIn("corridor 400 MiB", str(ctx.exception))

    def test_corridor_state_is_checked_against_the_driver_when_readable(self):
        free = {CARD_5090: 100 * MIB, CARD_3080_A: 8 * GIB, CARD_3080_B: 8 * GIB}
        registry = EngineRegistry(
            store=self.store,
            card_totals=RIG,
            clock=lambda: self.now,
            free_bytes_resolver=free.get,
        )
        views = {c.card_uuid: c for c in registry.cards()}
        self.assertFalse(views[CARD_5090].corridor_ok)
        self.assertTrue(views[CARD_3080_A].corridor_ok)

    def test_capture_lock_is_per_card_and_exclusive(self):
        with self.registry.capture_lock(CARD_5090, purpose="graph capture"):
            # A second holder on the same card must not get in; a different
            # card must.
            from sglang.srt.registry.ledger import CardBusyError

            with self.assertRaises(CardBusyError):
                with self.registry.capture_lock(CARD_5090, timeout=0):
                    pass
            with self.registry.capture_lock(CARD_3080_A, timeout=0):
                pass


class CoTenancyWithM2Test(RegistryTestCase):
    """The Class-3 video tenant of M2 and a registry engine, one ledger.

    M2 wrote its reservation through ``sglang.srt.video_enhance.reservation``
    while the registry was not yet built. That name now re-exports the
    registry's ledger, so the two see each other. If they ever stopped doing
    so, each would believe it had the whole card.
    """

    def test_the_m2_import_path_is_the_same_store(self):
        from sglang.srt.video_enhance import reservation as m2

        self.assertIs(m2.ReservationStore, ReservationStore)

        tenant = m2.ReservationStore(
            self.root,
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
        )
        tenant.acquire(
            card_uuid=CARD_5090,
            tenant_id="video-enhance",
            klass=3,
            reserved_bytes=24 * GIB,
        )
        self.registry.register(spec("llm", 12 * 1024, [CARD_5090]))
        with self.assertRaises(PromotionRejected) as ctx:
            self.registry.ensure_state("llm", ResidencyState.HOT)
        self.assertIn("video-enhance", str(ctx.exception))

    def test_the_registry_sees_the_m2_tenant_in_its_card_view(self):
        from sglang.srt.video_enhance import reservation as m2

        tenant = m2.ReservationStore(
            self.root,
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
        )
        tenant.acquire(
            card_uuid=CARD_5090,
            tenant_id="video-enhance",
            klass=3,
            reserved_bytes=8 * GIB,
        )
        view = {c.card_uuid: c for c in self.registry.cards()}[CARD_5090]
        self.assertIn("video-enhance", view.tenants)
        self.assertEqual(view.reserved_bytes, 8 * GIB)
        self.assertEqual(view.available_bytes, 32 * GIB - 8 * GIB - 400 * MIB)

    def test_an_orphaned_m2_tenant_stops_blocking_the_registry(self):
        store = ReservationStore(
            self.root,
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
            pid_alive=lambda pid: False,
        )
        store.acquire(
            card_uuid=CARD_5090,
            tenant_id="crashed-video",
            klass=3,
            reserved_bytes=30 * GIB,
            lease_seconds=60.0,
            pid=999_999,
        )
        registry = EngineRegistry(store=store, card_totals=RIG, clock=lambda: self.now)
        self.addCleanup(registry.shutdown)
        registry.register(spec("llm", 20 * 1024, [CARD_5090]))
        with self.assertRaises(PromotionRejected):
            registry.ensure_state("llm", ResidencyState.HOT, allow_eviction=False)
        self.advance(61.0)
        # The lease lapsed and the pid is gone; acquire reaps it in the same
        # critical section, so the next admission simply succeeds.
        registry.ensure_state("llm", ResidencyState.HOT)
        self.assertEqual(registry.instance("llm").state, ResidencyState.HOT)


if __name__ == "__main__":
    unittest.main()
