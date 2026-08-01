"""Tier identity, the volatility law, and what the registry refuses (#407).

What is pinned here, in the order it matters:

1. **Identity is never positional.** ``vram:0`` is refused at the door. That
   is DESIGN_407's contradiction C2: on this rig the 5090 is CUDA ordinal 0
   and NVML index 1, so an index that crosses a process boundary names a
   different card. A future diff that relaxed the grammar to "whatever the
   caller passed" turns this red.

2. **Volatility is a refusal, not a ranking.** A hibernate image
   (``persistence_required``) offered a tmpfs tier is the silent-correctness
   hole DESIGN_407 §2.4b names, and the falsifier below is exactly it.

3. **An absent number is not a bad number.** A tier with no measured
   bandwidth is refused against a bandwidth floor rather than sorted last --
   the #348b D4 defect, one layer down: a missing rate used to read as an
   extremely slow but valid card.

Hermetic: pure records over inlined fixtures. No driver, no mount, no wire --
which is also why this uses ``unittest.TestCase`` rather than
``CustomTestCase``, exactly as ``test_cost_model.py`` and
``test_ledger_multicard.py`` do: there is no resource to leak.

    python -m pytest test/registered/unit/memtier/test_tier_records.py -v
"""

import unittest

from sglang.srt.memtier.registry import (
    RefusalRule,
    TierQuery,
    TierRegistry,
    UnknownTier,
)
from sglang.srt.memtier.tiers import (
    ADMITTED_PAYLOADS,
    HEALTH_VERDICTS,
    PayloadClass,
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierIdError,
    TierKind,
    TierTransport,
    Volatility,
    admission_refusal,
    blob_tier_id,
    device_tier_id,
    filesystem_tier_id,
    host_tier_id,
    parse_tier_id,
)
from sglang.srt.planner import rig_coupling
from sglang.srt.planner.cost_model import Rate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024**3


def _caps(bandwidth: Rate, *, ledger_key: str = "bucket") -> TierCaps:
    return TierCaps(
        latency_us=Rate.absent("no latency probe in this fixture", unit="us"),
        bandwidth_gbs=bandwidth,
        aperture_bytes=Rate.absent("not aperture gated", unit="bytes"),
        ledger_key=ledger_key,
    )


def _tier(
    tier_id,
    kind,
    *,
    volatility,
    bandwidth,
    host="rig-1",
    total=100 * GIB,
    floor=0,
    admits=frozenset({"experts"}),
    health=None,
    properties=None,
    transport_name="posix",
) -> TierDescriptor:
    return TierDescriptor(
        id=tier_id,
        kind=kind,
        host=host,
        capacity=TierCapacity(
            total=(
                Rate.absent("fixture: unknown total", unit="bytes")
                if total is None
                else Rate.measured(float(total), "fixture", unit="bytes")
            ),
            floor=(
                Rate.absent("fixture: floor not measured before boot", unit="bytes")
                if floor is None
                else Rate.measured(float(floor), "fixture", unit="bytes")
            ),
        ),
        volatility=volatility,
        admits=admits,
        caps=_caps(bandwidth),
        health=health or TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name=transport_name),
        properties=properties or {},
        profile_id="fixture",
    )


class TierIdGrammarTest(unittest.TestCase):
    def test_four_forms_round_trip(self):
        """The grammar is the ledger's key space; a silent widening here would
        make two spellings of one tier two tiers."""
        self.assertEqual(
            parse_tier_id(device_tier_id("GPU-abc-1")).kind, TierKind.DEVICE
        )
        self.assertEqual(parse_tier_id(host_tier_id("rig-2")).host, "rig-2")
        parsed = parse_tier_id(filesystem_tier_id("rig-1", "/spinning"))
        self.assertEqual((parsed.host, parsed.mount), ("rig-1", "/spinning"))
        parsed = parse_tier_id(blob_tier_id("mooncake", "cluster-a"))
        self.assertEqual((parsed.backend, parsed.scope), ("mooncake", "cluster-a"))

    def test_a_device_index_is_refused_by_name(self):
        """C2, the silent card swap: an ordinal must never become a tier id.

        Can-fail proof: a grammar that accepted any string accepts ``vram:0``,
        and the message stops naming the ordinal-vs-NVML divergence.
        """
        with self.assertRaises(TierIdError) as ctx:
            parse_tier_id("vram:0")
        self.assertIn("device INDEX", str(ctx.exception))
        self.assertIn("UUID", str(ctx.exception))

    def test_unknown_prefix_and_relative_mount_are_refused(self):
        for bad in ("nvme:/spinning", "fs:rig-1:spinning", "blob:mooncake", "host:"):
            with self.assertRaises(TierIdError, msg=bad):
                parse_tier_id(bad)

    def test_declared_but_unenumerated_card_parses_and_is_marked_unbound(self):
        """NORDSTERN: rig-2's cards are declared so the gap is a dashboard row.

        The ``is_bound`` flag is what keeps them from being usable anyway.
        """
        parsed = parse_tier_id("vram:unenumerated@rig-2")
        self.assertEqual(parsed.host, "rig-2")
        self.assertFalse(parsed.is_bound)
        self.assertTrue(parse_tier_id("vram:GPU-abc-1").is_bound)


class TierRecordTest(unittest.TestCase):
    def test_kind_must_agree_with_the_id(self):
        """A record whose id says ``fs:`` and whose kind says HOST would route
        a filesystem query onto host RAM."""
        with self.assertRaises(ValueError):
            _tier(
                "fs:rig-1:/spinning",
                TierKind.HOST,
                volatility=Volatility.PERSISTENT,
                bandwidth=Rate.measured(1.8, "fixture", unit="GB/s"),
            )

    def test_admits_is_checked_against_the_one_offload_vocabulary(self):
        """#407 subsumes vocabularies rather than adding a fifth one; a typo'd
        class name must not create a silent private one."""
        with self.assertRaises(ValueError) as ctx:
            _tier(
                "host:rig-1",
                TierKind.HOST,
                volatility=Volatility.EXPENSIVE_OK,
                bandwidth=Rate.measured(38.0, "fixture", unit="GB/s"),
                admits=frozenset({"expert"}),
            )
        self.assertIn("OFFLOAD_CLASSES", str(ctx.exception))

    def test_a_degraded_verdict_must_carry_a_reason(self):
        """An unexplained ``block`` is indistinguishable from a bug, and the
        row it renders is the only place a user learns the tier is gone."""
        with self.assertRaises(ValueError):
            TierHealth(reachable=False, verdict="block")
        with self.assertRaises(ValueError):
            TierHealth(reachable=True, verdict="degraded", reason="x")

    def test_health_and_provenance_vocabularies_match_the_dashboard(self):
        """External-source literals: the strings are ``rig_coupling``'s
        contract, and a tier row shares a table with a coupling row."""
        self.assertEqual(
            HEALTH_VERDICTS,
            (rig_coupling.OK, rig_coupling.WARN, rig_coupling.BLOCK),
        )

    def test_headroom_is_absent_when_the_floor_is(self):
        """#400: the caller refuses rather than guesses. A headroom nobody
        could compute is not a large headroom, and before a boot the #330
        pinned floor genuinely does not exist."""
        tier = _tier(
            "vram:GPU-abc-1",
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=Rate.measured(1558.0, "fixture", unit="GB/s"),
            floor=None,
        )
        headroom = tier.capacity.headroom()
        self.assertTrue(headroom.is_absent)
        self.assertIn("floor", headroom.source)

    def test_headroom_subtracts_reserved_and_corridor(self):
        capacity = TierCapacity(
            total=Rate.measured(100.0, "fixture", unit="bytes"),
            floor=Rate.measured(10.0, "fixture", unit="bytes"),
            reserved=20,
            corridor=5,
        )
        self.assertEqual(capacity.headroom().require("h"), 65.0)

    def test_role_is_relative_to_the_reader_and_the_origin(self):
        """ "Peer" is a relation, not a property: one card's VRAM is local to
        the rank that owns it and peer to the rank beside it."""
        card = _tier(
            "vram:GPU-abc-1",
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=Rate.measured(1558.0, "fixture", unit="GB/s"),
            transport_name="barlink-bar1",
        )
        self.assertEqual(card.role(origin="vram:GPU-abc-1"), "vram-local")
        self.assertEqual(card.role(origin="vram:GPU-other-2"), "vram-peer-bar1")
        self.assertEqual(card.role(local_host="rig-2"), "vram-remote")
        nvme = _tier(
            "fs:rig-1:/spinning",
            TierKind.FILESYSTEM,
            volatility=Volatility.PERSISTENT,
            bandwidth=Rate.measured(1.8, "fixture", unit="GB/s"),
            properties={"medium": "nvme"},
        )
        self.assertEqual(nvme.role(local_host="rig-1"), "nvme-local")
        self.assertEqual(nvme.role(local_host="rig-2"), "nvme-remote")


class VolatilityLawTest(unittest.TestCase):
    def test_every_tier_volatility_has_an_admission_row(self):
        """Completeness bookkeeping: a fifth volatility added without a row
        would raise ``KeyError`` deep inside a spill decision."""
        self.assertEqual(set(ADMITTED_PAYLOADS), set(Volatility))

    def test_persistence_required_is_refused_a_non_persistent_tier(self):
        """THE falsifier. A hibernate image on a tmpfs is a hibernate that does
        not survive a reboot, and nothing in the tree notices today.

        Can-fail proof: an implementation that ranked tiers by cost instead of
        refusing by class admits the tmpfs tier here, because it is the
        fastest thing in the fixture.
        """
        tmpfs = _tier(
            "fs:rig-1:/dev/shm",
            TierKind.FILESYSTEM,
            volatility=Volatility.EXPENSIVE_OK,
            bandwidth=Rate.measured(40.0, "fixture", unit="GB/s"),
        )
        nvme = _tier(
            "fs:rig-1:/spinning",
            TierKind.FILESYSTEM,
            volatility=Volatility.PERSISTENT,
            bandwidth=Rate.measured(1.8, "fixture", unit="GB/s"),
        )
        refusal = admission_refusal(tmpfs, PayloadClass.PERSISTENCE_REQUIRED)
        self.assertIsNotNone(refusal)
        self.assertIn("persistence_required", refusal)
        self.assertIsNone(admission_refusal(nvme, PayloadClass.PERSISTENCE_REQUIRED))

    def test_device_bound_content_refuses_every_tier_but_its_own_card(self):
        """X2: recurrent state that made a lossy or reordered round trip is a
        correctness failure. Both cards are ``DEVICE_BOUND_ONLY``, so only the
        origin relation can tell them apart."""
        own = _tier(
            "vram:GPU-abc-1",
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=Rate.measured(1558.0, "fixture", unit="GB/s"),
        )
        peer = _tier(
            "vram:GPU-def-2",
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=Rate.measured(723.0, "fixture", unit="GB/s"),
        )
        self.assertIsNone(
            admission_refusal(own, PayloadClass.DEVICE_BOUND, origin=own.id)
        )
        refusal = admission_refusal(peer, PayloadClass.DEVICE_BOUND, origin=own.id)
        self.assertIn("never travels", refusal or "")

    def test_an_unenumerated_card_admits_nothing(self):
        """Declared-but-unbound is a dashboard row, not a spill target."""
        remote = _tier(
            "vram:unenumerated@rig-2",
            TierKind.DEVICE,
            volatility=Volatility.DEVICE_BOUND_ONLY,
            bandwidth=Rate.measured(0.83, "fixture", unit="GB/s"),
            host="rig-2",
            admits=frozenset(),
        )
        refusal = admission_refusal(remote, PayloadClass.RECONSTRUCTABLE)
        self.assertIn("never been enumerated", refusal or "")


class RegistryEnumerationTest(unittest.TestCase):
    def setUp(self):
        self.blocked = _tier(
            "fs:rig-2:/",
            TierKind.FILESYSTEM,
            volatility=Volatility.PERSISTENT,
            bandwidth=Rate.absent("never probed", unit="GB/s"),
            host="rig-2",
            health=TierHealth(
                reachable=False, verdict="block", reason="never probed, in any unit"
            ),
        )
        self.nvme = _tier(
            "fs:rig-1:/spinning",
            TierKind.FILESYSTEM,
            volatility=Volatility.PERSISTENT,
            bandwidth=Rate.measured(1.8, "fixture: cold read", unit="GB/s"),
            properties={"medium": "nvme", "flock": "yes"},
        )
        self.host = _tier(
            "host:rig-1",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            bandwidth=Rate.estimate(38.0, "fixture: DDR4-3200 assumed", unit="GB/s"),
        )
        self.registry = TierRegistry(
            [self.host, self.nvme, self.blocked],
            profile_id="fixture",
            local_host="rig-1",
        )

    def test_blocked_tiers_are_enumerated_not_omitted(self):
        """Omission is how a spill target silently becomes a different spill
        target. The registry's own rule, and the one a "tidy up the listing"
        diff breaks first."""
        self.assertIn("fs:rig-2:/", [t.id for t in self.registry.tiers()])
        self.assertIn("fs:rig-2:/", self.registry.ids())
        self.assertEqual(len(self.registry.tiers()), len(self.registry.ids()))

    def test_unknown_tier_names_what_is_declared(self):
        with self.assertRaises(UnknownTier) as ctx:
            self.registry.get("fs:rig-1:/nope")
        self.assertIn("fs:rig-1:/spinning", str(ctx.exception))

    def test_duplicate_ids_are_refused_at_construction(self):
        with self.assertRaises(ValueError):
            TierRegistry(
                [self.nvme, self.nvme], profile_id="fixture", local_host="rig-1"
            )

    def test_a_blocked_tier_is_refused_with_its_own_reason(self):
        selection = self.registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE)
        )
        refusal = selection.refusal_for("fs:rig-2:/")
        self.assertEqual(refusal.rule, RefusalRule.HEALTH)
        self.assertIn("never probed", refusal.reason)

    def test_absent_bandwidth_is_refused_against_a_floor_not_ranked_last(self):
        """#286, lifted: an unmeasured path is NEVER assumed usable, however
        tempting. Can-fail proof: an implementation that read an absent rate
        as 0.0 would also refuse -- so the assertion is on the RULE, which
        distinguishes "too slow" from "nobody knows"."""
        selection = self.registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE, min_bandwidth_gbs=0.1)
        )
        refusal = selection.refusal_for("fs:rig-2:/")
        self.assertEqual(refusal.rule, RefusalRule.HEALTH)  # blocked first
        registry = TierRegistry(
            [
                _tier(
                    "fs:rig-1:/dev/shm",
                    TierKind.FILESYSTEM,
                    volatility=Volatility.EXPENSIVE_OK,
                    bandwidth=Rate.absent("no tmpfs probe", unit="GB/s"),
                )
            ],
            profile_id="fixture",
            local_host="rig-1",
        )
        refusal = registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE, min_bandwidth_gbs=0.1)
        ).refusal_for("fs:rig-1:/dev/shm")
        self.assertEqual(refusal.rule, RefusalRule.BANDWIDTH_ABSENT)
        self.assertIn("never assumed usable", refusal.reason)

    def test_unmeasured_bandwidth_is_admitted_only_when_asked_for(self):
        registry = TierRegistry(
            [
                _tier(
                    "fs:rig-1:/dev/shm",
                    TierKind.FILESYSTEM,
                    volatility=Volatility.EXPENSIVE_OK,
                    bandwidth=Rate.absent("no tmpfs probe", unit="GB/s"),
                )
            ],
            profile_id="fixture",
            local_host="rig-1",
        )
        query = TierQuery(payload=PayloadClass.RECONSTRUCTABLE)
        self.assertEqual(registry.select(query).tier_ids, ())
        permissive = TierQuery(
            payload=PayloadClass.RECONSTRUCTABLE, allow_unmeasured_bandwidth=True
        )
        self.assertEqual(registry.select(permissive).tier_ids, ("fs:rig-1:/dev/shm",))

    def test_a_measurement_outranks_a_faster_estimate(self):
        """The ordering key's whole point: the registry never prefers a guess.

        The fixture is deliberately adversarial -- the estimate is 21x faster
        than the measurement -- so a key that sorted on bandwidth alone puts
        the estimate first and this turns red.
        """
        selection = self.registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE)
        )
        self.assertEqual(selection.tier_ids, ("fs:rig-1:/spinning", "host:rig-1"))
        self.assertEqual(selection.candidates[0].order_key[0], 0)
        self.assertEqual(selection.candidates[1].order_key[0], 1)
        self.assertTrue(any("estimate" in n for n in selection.candidates[1].notes))

    def test_require_measured_refuses_the_estimate(self):
        selection = self.registry.select(
            TierQuery(
                payload=PayloadClass.RECONSTRUCTABLE, require_measured_bandwidth=True
            )
        )
        self.assertEqual(selection.tier_ids, ("fs:rig-1:/spinning",))
        self.assertEqual(
            selection.refusal_for("host:rig-1").rule,
            RefusalRule.BANDWIDTH_UNMEASURED,
        )

    def test_capacity_refusal_itemises_the_arithmetic(self):
        selection = self.registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE, bytes_needed=1000 * GIB)
        )
        refusal = selection.refusal_for("fs:rig-1:/spinning")
        self.assertEqual(refusal.rule, RefusalRule.CAPACITY)
        self.assertIn("corridor", refusal.reason)

    def test_property_filter_is_a_capability_check_not_a_name_check(self):
        """#89 needs flock, not "a filesystem". A tier that grew a name the
        consumer did not recognise must still be selectable on capability."""
        selection = self.registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE, require={"flock": "yes"})
        )
        self.assertEqual(selection.tier_ids, ("fs:rig-1:/spinning",))
        self.assertEqual(selection.refusal_for("host:rig-1").rule, RefusalRule.PROPERTY)

    def test_object_class_filter_refuses_by_name(self):
        registry = TierRegistry(
            [
                _tier(
                    "host:rig-1",
                    TierKind.HOST,
                    volatility=Volatility.EXPENSIVE_OK,
                    bandwidth=Rate.measured(38.0, "fixture", unit="GB/s"),
                    admits=frozenset({"experts"}),
                )
            ],
            profile_id="fixture",
            local_host="rig-1",
        )
        selection = registry.select(
            TierQuery(
                payload=PayloadClass.RECONSTRUCTABLE, object_class="gdn_state_sets"
            )
        )
        self.assertEqual(selection.tier_ids, ())
        self.assertEqual(
            selection.refusal_for("host:rig-1").rule, RefusalRule.OBJECT_CLASS
        )

    def test_every_tier_appears_exactly_once_as_candidate_or_refusal(self):
        """A tier that fell out of both lists would be invisible -- neither
        offered nor explained."""
        selection = self.registry.select(
            TierQuery(payload=PayloadClass.RECONSTRUCTABLE)
        )
        seen = set(selection.tier_ids) | {r.tier_id for r in selection.refusals}
        self.assertEqual(seen, set(self.registry.ids()))

    def test_gate_rows_cover_every_tier_and_keep_the_dashboard_vocabulary(self):
        rows = self.registry.gate_rows()
        self.assertEqual(len(rows), len(self.registry.ids()))
        self.assertEqual({r.verdict for r in rows}, {"ok", "block"})
        by_key = {r.key: r for r in rows}
        remote = by_key["memtier:fs:rig-2:/"]
        self.assertEqual(remote.provenance, "absent")
        self.assertTrue(remote.remedy)
        self.assertEqual(by_key["memtier:host:rig-1"].provenance, "estimate")


if __name__ == "__main__":
    unittest.main()
