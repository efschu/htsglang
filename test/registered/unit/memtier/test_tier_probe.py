"""The probe interface: declared, refusing, and unable to launder a guess (#407).

Cut 1 ships no card time. What it does ship is the rule that decides how a
number ever becomes MEASURED, and the four ways to break it are the four
can-fail proofs below:

* a non-``ok`` outcome that carries a value anyway;
* an ``ok`` outcome carrying an ESTIMATE, i.e. a formula wearing a probe's
  label;
* an ``ok`` outcome with no rate at all;
* a second measurement quietly overwriting a first one, which loses the
  provenance of whichever run was real.

Each raises rather than logging, because the failing path must not be the
quiet one. The design's motivation is concrete: three numbers a reader would
assume were measured are not (the peer-VRAM microsecond class is an
assumption, NVMe latency does not exist in any unit, host DRAM bandwidth comes
off an assumed DDR4-3200 peak), and a registry that filled them in would be
worse than one that has them, because a wrong number outranks a missing one in
every comparison it appears in.

Hermetic: records and stub probes only.

    python -m pytest test/registered/unit/memtier/test_tier_probe.py -v
"""

import unittest

from sglang.srt.memtier.probe import (
    PROBES,
    ProbeOutcome,
    ProbeTarget,
    ProvenanceUpgradeRefused,
    UnimplementedProbe,
    apply_outcome,
    missing_measurements,
    probe_by_id,
    probes_for,
    require_measured,
    run_probe,
)
from sglang.srt.memtier.profile import bundled_profile
from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
)
from sglang.srt.planner.cost_model import Rate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _nvme_tier(latency=None):
    return TierDescriptor(
        id="fs:rig-1:/spinning",
        kind=TierKind.FILESYSTEM,
        host="rig-1",
        capacity=TierCapacity(
            total=Rate.measured(1.0, "fixture", unit="bytes"),
            floor=Rate.measured(0.0, "fixture", unit="bytes"),
        ),
        volatility=Volatility.PERSISTENT,
        admits=frozenset({"experts"}),
        caps=TierCaps(
            latency_us=latency
            or Rate.absent("no fio, no iodepth, nothing in the tree", unit="us"),
            bandwidth_gbs=Rate.measured(1.8, "fixture: cold read", unit="GB/s"),
            aperture_bytes=Rate.absent("not aperture gated", unit="bytes"),
            ledger_key="nvme_spinning",
        ),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name="posix"),
        properties={"medium": "nvme"},
        profile_id="fixture",
    )


def _measured_latency(value=95.0):
    return Rate.measured(value, "fio, iodepth 1, cold", unit="us")


class ProbeCatalogueTest(unittest.TestCase):
    def test_probe_ids_are_unique_and_resolvable(self):
        """Bookkeeping: the catalogue is keyed by id in ``apply_outcome``, so
        a duplicate would silently retarget one probe's results."""
        ids = [p.id for p in PROBES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(probe_by_id("M5").target, ProbeTarget.LATENCY)

    def test_a_probe_applies_only_where_its_property_gate_says(self):
        """M5 is an NVMe latency probe; offering it for the tmpfs tier would
        put an NVMe number on a RAM-backed filesystem."""
        nvme = _nvme_tier()
        tmpfs = nvme.evolve(id="fs:rig-1:/dev/shm", properties={"medium": "tmpfs"})
        self.assertIn("M5", [p.id for p in probes_for(nvme)])
        self.assertNotIn("M5", [p.id for p in probes_for(tmpfs)])

    def test_every_declared_probe_names_a_harness_and_what_it_unblocks(self):
        """The catalogue's only job in cut 1 is to turn an absence into a work
        item; an entry without a harness is a blank cell with extra steps."""
        for spec in PROBES:
            self.assertTrue(spec.harness.strip(), msg=spec.id)
            self.assertTrue(spec.unblocks.strip(), msg=spec.id)

    def test_the_bar1_ladder_records_the_harness_it_cannot_use(self):
        """DESIGN_407 §4 M1's structural caveat: barlink implements collectives
        only and has no send/recv, so the ladder cannot come from the harness
        everybody assumed. Losing that sentence costs somebody an afternoon."""
        self.assertIn("no send/recv", probe_by_id("M1").harness)


class StubProbeTest(unittest.TestCase):
    def test_the_default_runner_returns_absent_and_names_the_harness(self):
        tier = _nvme_tier()
        outcome = run_probe(probe_by_id("M5"), tier)
        self.assertEqual(outcome.status, "absent")
        self.assertIn("fio", outcome.reason)
        self.assertIsNone(outcome.rate)

    def test_an_inapplicable_probe_says_so_rather_than_running(self):
        host = _nvme_tier().evolve(
            id="host:rig-1", kind=TierKind.HOST, properties={"medium": "dram"}
        )
        outcome = run_probe(probe_by_id("M5"), host)
        self.assertEqual(outcome.status, "absent")
        self.assertIn("filesystem", outcome.reason)

    def test_a_stub_outcome_leaves_the_record_untouched(self):
        tier = _nvme_tier()
        outcome = UnimplementedProbe(probe_by_id("M5")).measure(tier)
        self.assertIs(apply_outcome(tier, outcome), tier)

    def test_a_non_ok_outcome_must_say_why(self):
        with self.assertRaises(ValueError):
            ProbeOutcome(probe_id="M5", tier_id="fs:rig-1:/spinning", status="error")
        with self.assertRaises(ValueError):
            ProbeOutcome(
                probe_id="M5", tier_id="fs:rig-1:/spinning", status="nope", reason="x"
            )


class ProvenanceLawTest(unittest.TestCase):
    def setUp(self):
        self.tier = _nvme_tier()

    def test_a_measurement_from_an_ok_probe_lands_with_its_own_source(self):
        outcome = ProbeOutcome(
            probe_id="M5",
            tier_id=self.tier.id,
            status="ok",
            rate=_measured_latency(),
        )
        updated = apply_outcome(self.tier, outcome)
        self.assertEqual(updated.caps.latency_us.value, 95.0)
        self.assertIn("fio", updated.caps.latency_us.source)
        # nothing else moved
        self.assertEqual(
            updated.caps.bandwidth_gbs.value, self.tier.caps.bandwidth_gbs.value
        )

    def test_a_failed_probe_may_not_deliver_a_number(self):
        """CAN-FAIL 1: the shape of a laundered guess -- a probe that errored
        but "has a reasonable value anyway"."""
        outcome = ProbeOutcome(
            probe_id="M5",
            tier_id=self.tier.id,
            status="error",
            reason="fio not installed",
            rate=_measured_latency(),
        )
        with self.assertRaises(ProvenanceUpgradeRefused) as ctx:
            apply_outcome(self.tier, outcome)
        self.assertIn("does not get to deliver a number", str(ctx.exception))

    def test_an_estimate_cannot_enter_through_a_probe(self):
        """CAN-FAIL 2: a formula wearing a probe's label. The host tier's
        38 GB/s is exactly this shape and belongs in the profile as an
        ESTIMATE, not in a probe result as a measurement."""
        outcome = ProbeOutcome(
            probe_id="M5",
            tier_id=self.tier.id,
            status="ok",
            rate=Rate.estimate(100.0, "derived from the bandwidth", unit="us"),
        )
        with self.assertRaises(ProvenanceUpgradeRefused) as ctx:
            apply_outcome(self.tier, outcome)
        self.assertIn("belongs in the profile", str(ctx.exception))

    def test_an_ok_outcome_with_no_rate_is_refused(self):
        """CAN-FAIL 3: a success that quietly writes nothing reads downstream
        as "the probe ran and the number is still absent", which is a lie
        about the probe."""
        outcome = ProbeOutcome(probe_id="M5", tier_id=self.tier.id, status="ok")
        with self.assertRaises(ProvenanceUpgradeRefused):
            apply_outcome(self.tier, outcome)

    def test_an_existing_measurement_is_not_silently_overwritten(self):
        """CAN-FAIL 4: two runs, two sources, one field. Overwriting loses
        which run the number came from -- and a re-measurement must be a
        deliberate act, not the default."""
        measured = apply_outcome(
            self.tier,
            ProbeOutcome(
                probe_id="M5",
                tier_id=self.tier.id,
                status="ok",
                rate=_measured_latency(95.0),
            ),
        )
        with self.assertRaises(ProvenanceUpgradeRefused) as ctx:
            apply_outcome(
                measured,
                ProbeOutcome(
                    probe_id="M5",
                    tier_id=measured.id,
                    status="ok",
                    rate=Rate.measured(120.0, "fio, iodepth 4, warm", unit="us"),
                ),
            )
        self.assertIn("already MEASURED", str(ctx.exception))
        self.assertEqual(measured.caps.latency_us.value, 95.0)

    def test_an_outcome_for_another_tier_is_refused(self):
        outcome = ProbeOutcome(
            probe_id="M5",
            tier_id="fs:rig-1:/dev/shm",
            status="ok",
            rate=_measured_latency(),
        )
        with self.assertRaises(ProvenanceUpgradeRefused):
            apply_outcome(self.tier, outcome)


class RefusalIfUnmeasuredTest(unittest.TestCase):
    def test_require_measured_names_the_probe_that_would_produce_it(self):
        """#286's rule as a call a consumer cannot forget: an unmeasured path
        is never assumed usable, and the refusal is actionable."""
        with self.assertRaises(ProvenanceUpgradeRefused) as ctx:
            require_measured(_nvme_tier(), ProbeTarget.LATENCY)
        self.assertIn("M5", str(ctx.exception))
        self.assertIn("absent", str(ctx.exception))

    def test_require_measured_returns_the_value_once_it_exists(self):
        tier = _nvme_tier(latency=_measured_latency(95.0))
        self.assertEqual(require_measured(tier, ProbeTarget.LATENCY), 95.0)

    def test_missing_measurements_is_a_work_list_that_shrinks(self):
        """The dashboard query. It must shrink by exactly the pair that was
        just filled -- a stale list is how a measured number stays invisible."""
        tier = _nvme_tier()
        gaps = missing_measurements([tier])
        self.assertIn((tier.id, probe_by_id("M5")), gaps)
        filled = apply_outcome(
            tier,
            ProbeOutcome(
                probe_id="M5", tier_id=tier.id, status="ok", rate=_measured_latency()
            ),
        )
        self.assertNotIn((filled.id, probe_by_id("M5")), missing_measurements([filled]))
        self.assertEqual(len(missing_measurements([filled])), len(gaps) - 1)

    def test_the_shipped_profile_has_the_absences_the_design_recorded(self):
        """Ties the catalogue to the data: the NVMe latency gap (M5) and the
        card aperture gap (M2) are the two DESIGN_407 §4 names, and they show
        up as work items rather than as blank cells."""
        profile = bundled_profile()
        gaps = missing_measurements(profile.tiers)
        pairs = {(tier_id, spec.id) for tier_id, spec in gaps}
        self.assertIn(("fs:rig-1:/spinning", "M5"), pairs)
        self.assertIn(("fs:rig-1:/dev/shm", "M6"), pairs)


if __name__ == "__main__":
    unittest.main()
