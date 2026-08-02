"""Reading the artifacts this tree already writes (#407 slice 1).

#407 adds no probe. These tests pin that the three EXISTING artifact formats
are read as they are written, and that nothing about them can slip past the
provenance law on the way in.

The formats, and where the shape comes from:

* ``card_probe`` -- ``rigmon/card_probe.py``'s ``CardProbeProfile.to_json``:
  ``{"version": 1, "cards": [...], "pairs": [...]}``, UUID-keyed.
* ``rig artifact`` -- ``planner/rig_artifact.py``'s ``Measurement.to_json``
  rows under ``measurements``, with the ``ok | warn | error | absent`` status
  vocabulary ``comm_suite`` shares.
* ``capability_matrix`` -- ``scripts/p2p_readiness``'s envelope, BDF-keyed
  ``directed_pairs`` plus a ``devices`` table that carries both names. Read the
  same way ``barlink_path_rates.load_p2p_capability_matrix`` reads it, down to
  the legacy ``pairs`` key and the refusal to consume nominal BAR1 fields.

Four laws, each with a falsifier below:

1. an adapter never writes a tier field -- it emits outcomes, and
   ``apply_outcome`` is the only writer;
2. a row the producer did not mark usable yields no value;
3. a number is never re-labelled: an ``h2d`` figure does not become a host
   tier's DRAM bandwidth by being nearby;
4. a row that cannot be assigned to exactly one tier is skipped, not guessed.

    python -m pytest test/registered/unit/memtier/test_tier_adapters.py -v
"""

import unittest

from sglang.srt.memtier.adapters import (
    ARTIFACT_ROUTES,
    apply_outcomes,
    from_capability_matrix,
    from_card_probe,
    from_rig_artifact,
)
from sglang.srt.memtier.bootstrap import bootstrap_tiers
from sglang.srt.memtier.probe import ProbeTarget, probe_by_id
from sglang.srt.memtier.profile import CardFact, FilesystemFact, LocalFacts
from sglang.srt.planner.cost_model import Provenance
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024**3
UUID_A = "GPU-11111111-2222-3333-4444-555555555555"
UUID_B = "GPU-66666666-7777-8888-9999-aaaaaaaaaaaa"
BDF_A = "0000:07:00.0"
BDF_B = "0000:0b:00.0"

FACTS = LocalFacts(
    cards=(
        CardFact(
            uuid=UUID_A, model="Synthetic Card A", total_bytes=24 * GIB, bdf=BDF_A
        ),
        CardFact(
            uuid=UUID_B, model="Synthetic Card B", total_bytes=48 * GIB, bdf=BDF_B
        ),
    ),
    host_total_bytes=128 * GIB,
    host_available_bytes=100 * GIB,
    filesystems=(
        FilesystemFact(
            mount="/fast", total_bytes=4 * 10**12, available_bytes=3 * 10**12
        ),
    ),
)


def tiers(fs_type="ext4"):
    return bootstrap_tiers(FACTS, host="unit", fs_types={"/fast": fs_type})


def device_tier(records, uuid):
    return next(t for t in records if t.parsed.card_key == uuid)


def host_tier(records):
    return next(t for t in records if t.id.startswith("host:"))


CARD_PROBE = {
    "version": 1,
    "uuids": [UUID_A, UUID_B],
    "cards": [
        {
            "uuid": UUID_A,
            "name": "Synthetic Card A",
            "membw_read_gbs": 700.0,
            "membw_gemv_gbs": 690.0,
            "h2d_gbs": 12.5,
            "d2h_gbs": 12.1,
        },
        {
            "uuid": UUID_B,
            "name": "Synthetic Card B",
            "membw_read_gbs": 1500.0,
            "h2d_gbs": 6.4,
        },
    ],
    "pairs": [
        {
            "src_uuid": UUID_A,
            "dst_uuid": UUID_B,
            "bandwidth_gbs": 4.5,
            "latency_us": 9.0,
        },
    ],
}


class TestCardProbeAdapter(unittest.TestCase):
    def test_membw_read_becomes_device_bandwidth(self):
        records = tiers()
        report = from_card_probe(CARD_PROBE, records)
        self.assertEqual(report.errors, ())
        self.assertEqual(len(report.outcomes), 2)
        updated, refusals = apply_outcomes(records, report.outcomes)
        self.assertEqual(refusals, ())
        rate = device_tier(updated, UUID_A).caps.bandwidth_gbs
        self.assertIs(rate.provenance, Provenance.MEASURED)
        self.assertEqual(rate.value, 700.0)
        self.assertIn("card_probe membw read arm", rate.source)

    def test_h2d_is_not_relabelled_as_host_dram_bandwidth(self):
        """Law 3. The PCIe edge does not become the host memory's number."""
        records = tiers()
        updated, _ = apply_outcomes(
            records, from_card_probe(CARD_PROBE, records).outcomes
        )
        self.assertTrue(host_tier(updated).caps.bandwidth_gbs.is_absent)

    def test_a_card_the_registry_does_not_declare_is_unrouted(self):
        report = from_card_probe(
            {"version": 1, "cards": [{"uuid": "GPU-nope", "membw_read_gbs": 1.0}]},
            tiers(),
        )
        self.assertEqual(report.outcomes, ())
        self.assertEqual(len(report.unrouted), 1)
        self.assertIn("not a tier in this registry", report.unrouted[0][1])

    def test_a_row_without_membw_is_skipped_not_zeroed(self):
        report = from_card_probe(
            {"version": 1, "cards": [{"uuid": UUID_A, "h2d_gbs": 12.5}]}, tiers()
        )
        self.assertEqual(report.outcomes, ())
        self.assertEqual(len(report.skipped), 1)

    def test_a_foreign_artifact_version_is_refused_wholesale(self):
        report = from_card_probe(dict(CARD_PROBE, version=99), tiers())
        self.assertEqual(report.outcomes, ())
        self.assertEqual(len(report.errors), 1)
        self.assertIn("version 99", report.errors[0])

    def test_the_pairs_list_is_left_to_the_cost_model(self):
        """An edge fact stays an edge fact; no tier cap comes off `pairs`."""
        records = tiers()
        report = from_card_probe(CARD_PROBE, records)
        self.assertTrue(all(o.probe_id == "M9" for o in report.outcomes))
        self.assertTrue(
            all(
                probe_by_id(o.probe_id).target is ProbeTarget.BANDWIDTH
                for o in report.outcomes
            )
        )


def measurement(row_id, value, **extra):
    row = {
        "id": row_id,
        "label": row_id,
        "source": "comm_suite",
        "unit": "GB/s",
        "value": value,
        "status": "ok",
        "n": 5,
        "spread_pct": 1.2,
        "taken_at": "2026-08-02",
        "context": {},
        "note": "",
    }
    row.update(extra)
    return row


class TestRigArtifactAdapter(unittest.TestCase):
    def test_a_routed_row_fills_the_declared_field(self):
        records = tiers()
        document = {
            "schema": "htsglang-rig-artifact/v1",
            "measurements": [measurement("comm/memtier_host_dram/read", 51.2)],
        }
        report = from_rig_artifact(document, records)
        self.assertEqual(len(report.outcomes), 1)
        updated, refusals = apply_outcomes(records, report.outcomes)
        self.assertEqual(refusals, ())
        rate = host_tier(updated).caps.bandwidth_gbs
        self.assertIs(rate.provenance, Provenance.MEASURED)
        self.assertEqual(rate.value, 51.2)
        self.assertIn("n=5", rate.source)

    def test_an_unclaimed_id_is_reported_rather_than_dropped(self):
        report = from_rig_artifact(
            {"measurements": [measurement("comm/noise_floor/floor", 1.0)]}, tiers()
        )
        self.assertEqual(report.outcomes, ())
        self.assertEqual(len(report.unrouted), 1)
        self.assertIn("no ARTIFACT_ROUTES entry", report.unrouted[0][1])

    def test_a_row_the_producer_marked_absent_yields_nothing(self):
        """Law 2, and the four statuses stay four."""
        for status in ("error", "absent"):
            with self.subTest(status=status):
                report = from_rig_artifact(
                    {
                        "measurements": [
                            measurement(
                                "comm/memtier_host_dram/read",
                                51.2,
                                status=status,
                                note="the arm did not run",
                            )
                        ]
                    },
                    tiers(),
                )
                self.assertEqual(report.outcomes, ())
                self.assertIn(status, report.skipped[0][1])

    def test_a_warn_row_is_kept_and_carries_its_reservation(self):
        report = from_rig_artifact(
            {
                "measurements": [
                    measurement(
                        "comm/memtier_host_dram/read",
                        51.2,
                        status="warn",
                        note="one rank only",
                    )
                ]
            },
            tiers(),
        )
        self.assertEqual(len(report.outcomes), 1)
        self.assertIn("WARN: one rank only", report.outcomes[0].rate.source)

    def test_a_unit_mismatch_is_skipped_and_never_converted(self):
        report = from_rig_artifact(
            {
                "measurements": [
                    measurement("comm/memtier_host_dram/read", 51200.0, unit="MB/s")
                ]
            },
            tiers(),
        )
        self.assertEqual(report.outcomes, ())
        self.assertIn("not converted on a guess", report.skipped[0][1])

    def test_an_ambiguous_route_skips_rather_than_picks(self):
        """Law 4: two candidate device tiers, no context, no assignment."""
        report = from_rig_artifact(
            {
                "measurements": [
                    measurement(
                        "comm/memtier_bar1_ladder/point_latency",
                        3.0,
                        unit="us (median)",
                    )
                ]
            },
            tiers(),
        )
        self.assertEqual(report.outcomes, ())
        self.assertIn("not assigned by guessing", report.skipped[0][1])

    def test_an_explicit_tier_context_disambiguates(self):
        records = tiers()
        report = from_rig_artifact(
            {
                "measurements": [
                    measurement(
                        "comm/memtier_bar1_ladder/point_latency",
                        3.0,
                        unit="us (median)",
                        context={"tier_id": device_tier(records, UUID_A).id},
                    )
                ]
            },
            records,
        )
        self.assertEqual(len(report.outcomes), 1)
        updated, _ = apply_outcomes(records, report.outcomes)
        self.assertEqual(device_tier(updated, UUID_A).caps.latency_us.value, 3.0)
        self.assertTrue(device_tier(updated, UUID_B).caps.latency_us.is_absent)

    def test_the_nvme_route_only_fires_on_an_nvme_tier(self):
        row = measurement("comm/memtier_nvme/read_latency", 90.0, unit="us (median)")
        on_ext4 = from_rig_artifact({"measurements": [row]}, tiers("ext4"))
        self.assertEqual(on_ext4.outcomes, ())
        self.assertIn("no tier of kind", on_ext4.skipped[0][1])

    def test_every_declared_route_names_a_declared_probe(self):
        for route in ARTIFACT_ROUTES:
            with self.subTest(route=route.id):
                self.assertIs(probe_by_id(route.probe_id).target, route.target)


CAPABILITY_MATRIX = {
    "schema_version": 3,
    "kind": "capability_matrix",
    "host": "unit",
    "devices": [
        {"pci_bus_id": BDF_A, "uuid": UUID_A, "name": "Synthetic Card A"},
        {"pci_bus_id": BDF_B, "uuid": UUID_B, "name": "Synthetic Card B"},
    ],
    "directed_pairs": [
        {
            "src_pci": BDF_A,
            "dst_pci": BDF_B,
            "effective_max_single_copy_bytes": 96 * 1024**2,
            "dst_bar1_nominal_bytes": 256 * 1024**2,
        },
        {
            "src_pci": BDF_B,
            "dst_pci": BDF_A,
            "effective_max_single_copy_bytes": 32 * 1024**2,
            "dst_bar1_nominal_bytes": 256 * 1024**2,
        },
    ],
}


class TestCapabilityMatrixAdapter(unittest.TestCase):
    def test_aperture_is_the_narrowest_inbound_window(self):
        records = tiers()
        matrix = dict(CAPABILITY_MATRIX)
        matrix["directed_pairs"] = list(CAPABILITY_MATRIX["directed_pairs"]) + [
            {
                "src_pci": "0000:0c:00.0",
                "dst_pci": BDF_B,
                "effective_max_single_copy_bytes": 16 * 1024**2,
            }
        ]
        matrix["devices"] = list(CAPABILITY_MATRIX["devices"])
        report = from_capability_matrix(matrix, records)
        updated, refusals = apply_outcomes(records, report.outcomes)
        self.assertEqual(refusals, ())
        self.assertEqual(
            device_tier(updated, UUID_B).caps.aperture_bytes.value, 16 * 1024**2
        )
        self.assertIn(
            "narrowest inbound", device_tier(updated, UUID_B).caps.aperture_bytes.source
        )

    def test_the_nominal_window_is_never_read(self):
        records = tiers()
        report = from_capability_matrix(CAPABILITY_MATRIX, records)
        updated, _ = apply_outcomes(records, report.outcomes)
        for uuid in (UUID_A, UUID_B):
            self.assertNotEqual(
                device_tier(updated, uuid).caps.aperture_bytes.value, 256 * 1024**2
            )

    def test_a_row_with_no_effective_aperture_is_skipped(self):
        matrix = dict(
            CAPABILITY_MATRIX,
            directed_pairs=[
                {"src_pci": BDF_A, "dst_pci": BDF_B, "dst_bar1_nominal_bytes": 1}
            ],
        )
        report = from_capability_matrix(matrix, tiers())
        self.assertEqual(report.outcomes, ())
        self.assertIn("nominal", report.skipped[0][1])

    def test_bdf_domains_are_normalised(self):
        matrix = dict(
            CAPABILITY_MATRIX,
            directed_pairs=[
                {
                    "src_pci": "07:00.0",
                    "dst_pci": "0B:00.0",
                    "effective_max_single_copy_bytes": 4096,
                }
            ],
        )
        report = from_capability_matrix(matrix, tiers())
        self.assertEqual(len(report.outcomes), 1)

    def test_the_legacy_row_key_is_still_read(self):
        matrix = {k: v for k, v in CAPABILITY_MATRIX.items() if k != "directed_pairs"}
        matrix["pairs"] = CAPABILITY_MATRIX["directed_pairs"]
        self.assertEqual(len(from_capability_matrix(matrix, tiers()).outcomes), 2)

    def test_a_devices_table_without_uuids_refuses_the_whole_artifact(self):
        matrix = dict(
            CAPABILITY_MATRIX,
            devices=[{"pci_bus_id": BDF_A}, {"pci_bus_id": BDF_B}],
        )
        report = from_capability_matrix(matrix, tiers())
        self.assertEqual(report.outcomes, ())
        self.assertIn("NVML order is not stable", report.errors[0])

    def test_a_wrong_envelope_kind_is_refused(self):
        report = from_capability_matrix(
            dict(CAPABILITY_MATRIX, kind="d2d_bench"), tiers()
        )
        self.assertEqual(report.outcomes, ())
        self.assertIn("expected 'capability_matrix'", report.errors[0])

    def test_no_rows_at_all_is_an_error_not_an_empty_result(self):
        matrix = {k: v for k, v in CAPABILITY_MATRIX.items() if k != "directed_pairs"}
        report = from_capability_matrix(matrix, tiers())
        self.assertIn("indistinguishable", report.errors[0])


class TestProvenanceLawOnIngest(unittest.TestCase):
    def test_a_second_conflicting_measurement_is_refused_and_returned(self):
        """Law 1: the writer refuses, and apply_outcomes surfaces the refusal."""
        records = tiers()
        first, _ = apply_outcomes(
            records, from_card_probe(CARD_PROBE, records).outcomes
        )
        second = dict(CARD_PROBE)
        second["cards"] = [
            dict(CARD_PROBE["cards"][0], membw_read_gbs=42.0, name="relabelled")
        ]
        updated, refusals = apply_outcomes(
            first, from_card_probe(second, first).outcomes
        )
        self.assertEqual(len(refusals), 1)
        self.assertIn("already MEASURED", refusals[0])
        # the earlier measurement survived
        self.assertEqual(device_tier(updated, UUID_A).caps.bandwidth_gbs.value, 700.0)

    def test_an_identical_re_measurement_is_idempotent(self):
        records = tiers()
        once, _ = apply_outcomes(records, from_card_probe(CARD_PROBE, records).outcomes)
        twice, refusals = apply_outcomes(
            once, from_card_probe(CARD_PROBE, once).outcomes
        )
        self.assertEqual(refusals, ())
        self.assertEqual(device_tier(twice, UUID_A).caps.bandwidth_gbs.value, 700.0)

    def test_an_outcome_for_an_unknown_tier_is_reported(self):
        records = tiers()
        outcomes = from_card_probe(CARD_PROBE, records).outcomes
        _, refusals = apply_outcomes(records[:1], outcomes)
        self.assertTrue(any("has no such tier" in r for r in refusals))


if __name__ == "__main__":
    unittest.main()
