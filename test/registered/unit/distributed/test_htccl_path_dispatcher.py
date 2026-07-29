"""#279 HTCCL path dispatcher skeleton: registry, latched dispatch, loaders.

What is protected, in order of importance:

1. PLACEHOLDER NEUTRALITY. Without measured rate tables the dispatcher --
   enabled or not, empty or placeholder-filled -- must produce EXACTLY the
   selection today's #240 class wiring produces. This is the contract that
   makes the skeleton shippable before the NCCL/system-RAM reference
   measurement exists.
2. GRAPH SAFETY. Path choices are latched and flip only at round/capture
   boundaries; a boundary call during an active capture is refused.
3. PRIORITY PROTECTION. A protected class is never displaced to a slower
   path to make room for another class.
4. The three source parsers (p2p_readiness JSON, #278 GDR TSV, the
   nccl_reference schema) consume only effective/measured values and
   survive malformed input as per-row errors, never aborts.

CPU only: no torch.cuda, no real transports, no files outside tempdirs.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.htccl import HTCCLCommunicator
from sglang.srt.distributed.device_communicators.htccl_path_dispatcher import (
    HINT_GLOO,
    HINT_TRANSPORT,
    PROVENANCE_MEASURED,
    STATUS_QUO,
    DispatchRequest,
    PathDispatcher,
    PathProfile,
    RatePoint,
    bus_saturation_sensor,
    maybe_build_dispatcher,
    refine_transport_choice,
)
from sglang.srt.distributed.device_communicators.htccl_path_rates import (
    GDR_TSV_COLUMNS,
    NCCL_REFERENCE_ROW_FIELDS,
    LoadResult,
    apply_apertures,
    load_gdr_matrix_tsv,
    load_nccl_reference,
    load_p2p_capability_matrix,
    load_p2p_d2d_bench,
    load_rate_tables,
    new_nccl_reference_envelope,
    placeholder_profile,
)
from sglang.srt.model_executor.offload_bus_budget import BusBudgetArbiter
from sglang.srt.model_executor.offload_register import OffloadRegister
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

KIB = 1024
MIB = 1024 * 1024


def measured(name, base_ms=0.1, per_byte_ms=0.0, **kw):
    return PathProfile(
        name=name,
        provenance=PROVENANCE_MEASURED,
        base_ms=base_ms,
        per_byte_ms=per_byte_ms,
        **kw,
    )


def two_measured_paths(dispatcher, fast_kw=None, slow_kw=None):
    """fast (0.1 ms) and slow (1.0 ms) paths on class 'collective'."""
    dispatcher.register_path(measured("fast", 0.1, **(fast_kw or {})))
    dispatcher.register_path(measured("slow", 1.0, **(slow_kw or {})))


class TestPathProfile(CustomTestCase):
    def test_affine_fit_two_points_exact(self):
        p = PathProfile("p", points=[RatePoint(0, 2.0), RatePoint(1000, 12.0)])
        p.fit()
        self.assertAlmostEqual(p.base_ms, 2.0)
        self.assertAlmostEqual(p.per_byte_ms, 0.01)
        self.assertAlmostEqual(p.ms(500), 7.0)

    def test_affine_fit_least_squares_monotone(self):
        pts = [RatePoint(s, 1.0 + s * 0.001) for s in (64, 256, 1024, 4096)]
        p = PathProfile("p", points=pts)
        p.fit()
        self.assertGreaterEqual(p.base_ms, 0.0)
        self.assertLess(p.ms(64), p.ms(1 * MIB))

    def test_single_point_fit(self):
        p = PathProfile("p", points=[RatePoint(4096, 3.5)])
        p.fit()
        self.assertAlmostEqual(p.ms(10), 3.5)

    def test_register_rejects_unknown_provenance(self):
        d = PathDispatcher()
        with self.assertRaises(ValueError):
            d.register_path(PathProfile("p", provenance="guessed"))


class TestPlaceholderNeutrality(CustomTestCase):
    """Hard rule 1 -- the most important tests in this file."""

    def test_empty_registry_is_status_quo(self):
        d = PathDispatcher()
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertTrue(dec.status_quo)
        self.assertEqual(dec.path, STATUS_QUO)

    def test_any_placeholder_candidate_forces_status_quo(self):
        d = PathDispatcher()
        d.register_path(measured("fast", 0.1))
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.htccl_path_dispatcher",
            level="WARNING",
        ):
            d.register_path(placeholder_profile("unmeasured"))
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertTrue(dec.status_quo)
        self.assertIn("placeholder", dec.reason)

    def test_all_measured_enables_choice(self):
        d = PathDispatcher()
        two_measured_paths(d)
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertFalse(dec.status_quo)
        self.assertEqual(dec.path, "fast")

    def test_sensor_and_latency_never_consulted_under_placeholder(self):
        d = PathDispatcher()
        d.register_path(placeholder_profile("p", "test"))

        def boom(*a):
            raise AssertionError(
                "hook consulted although placeholder rates are present"
            )

        d.set_saturation_sensor(boom)
        d.set_offload_latency_term(boom)
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertTrue(dec.status_quo)

    def test_htccl_select_identical_with_and_without_dispatcher(self):
        """The acceptance test of the briefing: with the flag on but no
        measured data, _select answers EXACTLY as without the dispatcher,
        across transports, ops and sizes."""

        class FakeDevice:
            def handles(self, op, nbytes):
                return op in ("all_reduce", "all_gather", "reduce_scatter")

        class FakeShm:
            def handles(self, op, nbytes):
                return op == "all_reduce" and nbytes <= 64 * MIB

        placeholder_only = PathDispatcher()
        placeholder_only.register_path(placeholder_profile("p", "test"))
        for dispatcher in (None, PathDispatcher(), placeholder_only):
            for transport in (None, FakeDevice(), FakeShm()):
                for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast"):
                    for nbytes in (0, 4 * KIB, 64 * MIB, 65 * MIB):
                        comm = HTCCLCommunicator.__new__(HTCCLCommunicator)
                        comm.transport = transport
                        comm._path_dispatcher = None
                        baseline = HTCCLCommunicator._select(comm, op, nbytes)
                        comm2 = HTCCLCommunicator.__new__(HTCCLCommunicator)
                        comm2.transport = transport
                        comm2._path_dispatcher = dispatcher
                        self.assertIs(
                            HTCCLCommunicator._select(comm2, op, nbytes),
                            baseline,
                            f"divergence: dispatcher={dispatcher}, "
                            f"transport={transport}, op={op}, nbytes={nbytes}",
                        )
                if dispatcher is not None:
                    dispatcher.round_boundary()

    def test_flag_gates_construction(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HTCCL_PATH_DISPATCHER", None)
            self.assertIsNone(maybe_build_dispatcher())
        with mock.patch.dict(
            os.environ, {"SGLANG_HTCCL_PATH_DISPATCHER": "1"}
        ):
            d = maybe_build_dispatcher()
            self.assertIsInstance(d, PathDispatcher)
            self.assertEqual(d.paths(), {})


class TestDispatchAndOverflow(CustomTestCase):
    def test_best_path_by_cost_is_size_dependent(self):
        d = PathDispatcher()
        # low base + steep slope vs high base + flat slope: small messages
        # go one way, bulk the other -- the size-aware split.
        d.register_path(measured("latency_opt", 0.01, per_byte_ms=1e-6))
        d.register_path(measured("bandwidth_opt", 0.3, per_byte_ms=1e-8))
        small = d.decide(DispatchRequest("collective", 8 * KIB))
        bulk = d.decide(DispatchRequest("collective", 8 * MIB))
        self.assertEqual(small.path, "latency_opt")
        self.assertEqual(bulk.path, "bandwidth_opt")

    def test_saturation_overflow_to_next_best(self):
        d = PathDispatcher()
        two_measured_paths(d)
        d.set_saturation_sensor(lambda name: 1.0 if name == "fast" else 0.0)
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertEqual(dec.path, "slow")
        self.assertTrue(dec.overflowed)

    def test_all_saturated_stays_on_best(self):
        d = PathDispatcher()
        two_measured_paths(d)
        d.set_saturation_sensor(lambda name: 1.0)
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertEqual(dec.path, "fast")
        self.assertFalse(dec.overflowed)

    def test_aperture_excludes_oversized_transfers(self):
        d = PathDispatcher()
        two_measured_paths(d, fast_kw={"aperture_bytes": 256 * MIB})
        below = d.decide(DispatchRequest("collective", 128 * MIB))
        above = d.decide(DispatchRequest("collective", 512 * MIB))
        self.assertEqual(below.path, "fast")
        self.assertEqual(above.path, "slow")

    def test_all_over_aperture_is_status_quo(self):
        d = PathDispatcher()
        d.register_path(measured("windowed", aperture_bytes=1 * MIB))
        dec = d.decide(DispatchRequest("collective", 2 * MIB))
        self.assertTrue(dec.status_quo)
        self.assertIn("aperture", dec.reason)

    def test_unknown_class_is_status_quo(self):
        d = PathDispatcher()
        two_measured_paths(d)
        dec = d.decide(DispatchRequest("bulk", 4 * KIB))
        self.assertTrue(dec.status_quo)


class TestPriorityProtection(CustomTestCase):
    """Hard rule 3."""

    def test_protected_is_never_overflowed(self):
        d = PathDispatcher()
        two_measured_paths(d)
        d.set_saturation_sensor(lambda name: 1.0 if name == "fast" else 0.0)
        protected = d.decide(
            DispatchRequest("collective", 4 * KIB, priority=0, protected=True)
        )
        normal = d.decide(DispatchRequest("collective", 4 * KIB, priority=1))
        self.assertEqual(protected.path, "fast")
        self.assertFalse(protected.overflowed)
        self.assertEqual(normal.path, "slow")

    def test_protected_keeps_best_path_across_boundaries_under_pressure(self):
        d = PathDispatcher()
        two_measured_paths(d)
        saturated = {"value": 0.0}
        d.set_saturation_sensor(
            lambda name: saturated["value"] if name == "fast" else 0.0
        )
        req = DispatchRequest("collective", 4 * KIB, priority=0, protected=True)
        self.assertEqual(d.decide(req).path, "fast")
        saturated["value"] = 1.0  # other traffic saturates the best path
        d.round_boundary()
        self.assertEqual(d.decide(req).path, "fast")


class TestBoundariesAndCapture(CustomTestCase):
    """Hard rule 2."""

    def test_decision_latched_until_round_boundary(self):
        d = PathDispatcher()
        two_measured_paths(d)
        req = DispatchRequest("collective", 4 * KIB)
        self.assertEqual(d.decide(req).path, "fast")
        d.set_saturation_sensor(lambda name: 1.0 if name == "fast" else 0.0)
        self.assertEqual(d.decide(req).path, "fast", "flip inside a round")
        d.round_boundary()
        self.assertEqual(d.decide(req).path, "slow")

    def test_round_boundary_refused_during_capture(self):
        d = PathDispatcher()
        two_measured_paths(d)
        req = DispatchRequest("collective", 4 * KIB)
        self.assertEqual(d.decide(req).path, "fast")
        d.begin_capture()
        d.set_saturation_sensor(lambda name: 1.0 if name == "fast" else 0.0)
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.htccl_path_dispatcher",
            level="WARNING",
        ):
            d.round_boundary()
        self.assertEqual(d.decide(req).path, "fast", "flip inside a capture")

    def test_new_key_during_capture_is_status_quo_and_stable(self):
        d = PathDispatcher()
        two_measured_paths(d)
        d.begin_capture()
        req = DispatchRequest("collective", 4 * KIB)
        first = d.decide(req)
        self.assertTrue(first.status_quo)
        self.assertIs(d.decide(req), first, "replay must see the identical decision")

    def test_end_capture_is_a_boundary(self):
        d = PathDispatcher()
        two_measured_paths(d)
        d.begin_capture()
        req = DispatchRequest("collective", 4 * KIB)
        self.assertTrue(d.decide(req).status_quo)
        d.end_capture()
        self.assertEqual(d.decide(req).path, "fast")

    def test_lanes_latch_independently(self):
        d = PathDispatcher()
        two_measured_paths(d)
        a = d.decide(DispatchRequest("collective", 4 * KIB, lane="lane_a"))
        d.set_saturation_sensor(lambda name: 1.0 if name == "fast" else 0.0)
        b = d.decide(DispatchRequest("collective", 4 * KIB, lane="lane_b"))
        self.assertEqual(a.path, "fast")
        self.assertEqual(b.path, "slow")
        # lane_a's latch is untouched by lane_b's fresh computation.
        self.assertEqual(
            d.decide(DispatchRequest("collective", 4 * KIB, lane="lane_a")).path,
            "fast",
        )


class TestLatencyTermAndBus(CustomTestCase):
    """Hard rule 4 + the named #286 bus interface."""

    def test_parked_latency_term_changes_choice(self):
        d = PathDispatcher()
        d.register_path(measured("via_parked", 0.1, offload_class="workspace"))
        d.register_path(measured("plain", 0.3))
        self.assertEqual(
            d.decide(DispatchRequest("collective", 4 * KIB)).path, "via_parked"
        )
        d.round_boundary()
        d.set_offload_latency_term(
            lambda klass: 5.0 if klass == "workspace" else 0.0
        )
        self.assertEqual(
            d.decide(DispatchRequest("collective", 4 * KIB)).path, "plain"
        )

    def test_offload_register_latency_term_signature_attaches(self):
        reg = OffloadRegister()
        d = PathDispatcher()
        d.register_path(measured("p", offload_class="workspace"))
        d.set_offload_latency_term(reg.latency_term_ms)
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertEqual(dec.path, "p")  # nothing parked -> 0.0 ms term

    def test_bus_saturation_sensor_reads_pending_demand(self):
        arb = BusBudgetArbiter(total_rate_bytes_per_s=1000.0)
        arb.register_consumer("expert_streaming", weight=1.0, priority=0)
        arb.register_consumer("stage2_phase", weight=1.0, priority=1)
        sensor = bus_saturation_sensor(
            arb, {"pcie_path": "stage2_phase", "other": "expert_streaming"}
        )
        self.assertEqual(sensor("pcie_path"), 0.0)
        # Exhaust stage2_phase far past its share so the next request denies.
        arb.request("stage2_phase", 10_000_000)
        arb.request("expert_streaming", 10_000_000)
        denied = arb.request("stage2_phase", 10_000_000)
        self.assertFalse(denied.granted)
        self.assertEqual(sensor("pcie_path"), 1.0)
        self.assertEqual(sensor("unmapped_path"), 0.0)

    def test_bus_sensor_drives_overflow(self):
        arb = BusBudgetArbiter(total_rate_bytes_per_s=1000.0)
        arb.register_consumer("stage2_phase", weight=1.0, priority=1)
        arb.request("stage2_phase", 10_000_000)
        arb.request("stage2_phase", 10_000_000)
        d = PathDispatcher()
        two_measured_paths(d)
        d.set_saturation_sensor(
            bus_saturation_sensor(arb, {"fast": "stage2_phase"})
        )
        dec = d.decide(DispatchRequest("collective", 4 * KIB))
        self.assertEqual(dec.path, "slow")
        self.assertTrue(dec.overflowed)


class TestRefineTransportChoice(CustomTestCase):
    """The thin htccl hook, including the measured-decision actuation."""

    def _measured_dispatcher(self, hint):
        d = PathDispatcher()
        d.register_path(measured("winner", 0.1, transport_hint=hint))
        d.register_path(measured("loser", 1.0))
        return d

    def test_none_dispatcher_is_identity(self):
        sentinel = object()
        self.assertIs(refine_transport_choice(None, "all_reduce", 4, sentinel), sentinel)

    def test_gloo_hint_routes_to_inline_plane(self):
        d = self._measured_dispatcher(HINT_GLOO)
        self.assertIsNone(refine_transport_choice(d, "all_reduce", 4 * KIB, object()))

    def test_transport_hint_keeps_transport(self):
        d = self._measured_dispatcher(HINT_TRANSPORT)
        sentinel = object()
        self.assertIs(
            refine_transport_choice(d, "all_reduce", 4 * KIB, sentinel), sentinel
        )

    def test_unactionable_hint_falls_back_to_status_quo(self):
        d = self._measured_dispatcher(None)
        sentinel = object()
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.htccl_path_dispatcher",
            level="WARNING",
        ):
            self.assertIs(
                refine_transport_choice(d, "all_reduce", 4 * KIB, sentinel),
                sentinel,
            )


class TestP2PReadinessParsers(CustomTestCase):
    """Source 1: only EFFECTIVE values are consumed."""

    def _capability_payload(self):
        return {
            "schema_version": 3,
            "kind": "capability_matrix",
            "pairs": [
                {
                    "src_pci": "01:00.0",
                    "dst_pci": "05:00.0",
                    "dst_bar1_nominal_bytes": 256 * MIB,
                    "effective_max_single_copy_bytes": 224 * MIB,
                    "effective_max_region_chunked_bytes": 240 * MIB,
                },
                {  # nominal-only: measured aperture absent -> NOT consumed
                    "src_pci": "05:00.0",
                    "dst_pci": "01:00.0",
                    "dst_bar1_nominal_bytes": 32 * 1024 * MIB,
                    "effective_max_single_copy_bytes": None,
                },
                {"dst_pci": "0b:00.0"},  # malformed
            ],
        }

    def test_capability_matrix_effective_only(self):
        res = load_p2p_capability_matrix(self._capability_payload())
        self.assertEqual(res.apertures, {("01:00.0", "05:00.0"): 224 * MIB})
        self.assertEqual(len(res.skipped), 1)
        self.assertIn("nominal-only", res.skipped[0])
        self.assertEqual(len(res.errors), 1)

    def test_capability_matrix_wrong_kind(self):
        res = load_p2p_capability_matrix({"kind": "d2d_bench", "schema_version": 3})
        self.assertFalse(res.apertures)
        self.assertTrue(res.errors)

    def test_d2d_bench_profiles_and_error_points(self):
        payload = {
            "schema_version": 3,
            "kind": "d2d_bench",
            "pairs": [
                {
                    "src_pci": "01:00.0",
                    "dst_pci": "05:00.0",
                    "mode": "direct",
                    "points": [
                        {"size_bytes": 64 * KIB, "median_s": 1e-5, "p95_s": 2e-5,
                         "gib_per_s": 6.1},
                        {"size_bytes": 1 * MIB, "median_s": 1e-4, "p95_s": 2e-4,
                         "gib_per_s": 9.8},
                        {"size_bytes": 512 * MIB, "error": "mapping failure"},
                    ],
                },
                {
                    "src_pci": "01:00.0",
                    "dst_pci": "05:00.0",
                    "mode": "staged",
                    "points": [
                        {"size_bytes": 64 * KIB, "median_s": 3e-5, "p95_s": 4e-5,
                         "gib_per_s": 2.0},
                    ],
                },
            ],
        }
        res = load_p2p_d2d_bench(payload)
        names = sorted(p.name for p in res.profiles)
        self.assertEqual(
            names,
            ["d2d_direct:01:00.0->05:00.0", "host_staged:01:00.0->05:00.0"],
        )
        direct = next(p for p in res.profiles if p.name.startswith("d2d_direct"))
        self.assertEqual(direct.provenance, PROVENANCE_MEASURED)
        self.assertEqual(len(direct.points), 2)
        self.assertAlmostEqual(
            direct.capacity_bytes_per_s, 9.8 * 1024**3, delta=1e6
        )
        self.assertEqual(len(res.skipped), 1)  # the aperture-failure point

    def test_apply_apertures_direct_only(self):
        direct = measured("d2d_direct:01:00.0->05:00.0")
        staged = measured("host_staged:01:00.0->05:00.0")
        gdr = measured("gdr_direct@d4:01:00.0->05:00.0")
        nccl = measured("nccl:all_reduce:01:00.0->05:00.0")
        apply_apertures(
            [direct, staged, gdr, nccl],
            {("01:00.0", "05:00.0"): 224 * MIB},
        )
        self.assertEqual(direct.aperture_bytes, 224 * MIB)
        self.assertEqual(gdr.aperture_bytes, 224 * MIB)
        self.assertIsNone(staged.aperture_bytes)
        self.assertIsNone(nccl.aperture_bytes)


class TestGdrMatrixParser(CustomTestCase):
    """Source 2: the #278 crossrig-ladder TSV wire rows."""

    ROWS = [
        "# pair\tdirection\tmodus\tro\tdepth\tsize_bytes\titers\tp10_us\tmedian_us\tp90_us\tMB_per_s",
        "i_05:00.0_to_0a:00.0\tintra\tgdr\toff\t1\t20480\t234705\t9.428\t9.528\t9.688\t1074.7",
        "i_05:00.0_to_0a:00.0\tintra\tgdr\toff\t4\t20480\t100000\t9.428\t38.0\t40.0\t1074.7",
        "i_05:00.0_to_0a:00.0\tintra\tstage\toff\t1\t20480\t159185\t14.036\t14.272\t15.019\t717.5",
        "i_05:00.0_to_0a:00.0\tintra\tgdr\ton\t1\t20480\t1000\t9.0\t9.1\t9.2\t1100.0",
    ]

    def test_parse_real_format_rows(self):
        res = load_gdr_matrix_tsv(self.ROWS)
        self.assertFalse(res.errors)
        names = sorted(p.name for p in res.profiles)
        self.assertEqual(
            names,
            [
                "gdr_direct+ro@d1:05:00.0->0a:00.0",
                "gdr_direct@d1:05:00.0->0a:00.0",
                "gdr_direct@d4:05:00.0->0a:00.0",
                "nic_staged@d1:05:00.0->0a:00.0",
            ],
        )
        d1 = next(p for p in res.profiles if p.name == "gdr_direct@d1:05:00.0->0a:00.0")
        d4 = next(p for p in res.profiles if p.name == "gdr_direct@d4:05:00.0->0a:00.0")
        self.assertAlmostEqual(d1.points[0].ms, 9.528e-3)
        # one round carries depth messages: per-message time is median/depth.
        self.assertAlmostEqual(d4.points[0].ms, 38.0e-3 / 4)
        self.assertEqual(d1.provenance, PROVENANCE_MEASURED)
        self.assertAlmostEqual(d1.capacity_bytes_per_s, 1074.7e6)

    def test_malformed_rows_are_errors_not_aborts(self):
        rows = self.ROWS + [
            "too\tfew\tcolumns",
            "i_05:00.0_to_0a:00.0\tintra\twarp\toff\t1\t20480\t1\t1\t1\t1\t1",
            "i_05:00.0_to_0a:00.0\tintra\tgdr\toff\tX\t20480\t1\t1\t1\t1\t1",
            "neutral_pre\tb2a\tstage\toff\t1\t20480\t2000\t1\t1\t1\t-",
        ]
        res = load_gdr_matrix_tsv(rows)
        self.assertEqual(len(res.errors), 4)
        self.assertEqual(len(res.profiles), 4)  # good rows still loaded

    def test_column_layout_matches_artifact_header(self):
        header = self.ROWS[0].lstrip("# ").split("\t")
        self.assertEqual(tuple(header), GDR_TSV_COLUMNS)


class TestNcclReferenceSchema(CustomTestCase):
    """Source 3: format defined NOW so the pending measurement can write
    directly loadable JSON."""

    def _row(self, **kw):
        row = {
            "op": "all_reduce",
            "transport": "SHM",
            "world": 2,
            "src_pci": "01:00.0",
            "dst_pci": "05:00.0",
            "size_bytes": 20480,
            "iters": 5000,
            "p50_us": 46.6,
            "p99_us": 120.0,
            "load": "idle",
        }
        row.update(kw)
        return row

    def test_envelope_round_trips_through_loader(self):
        payload = new_nccl_reference_envelope()
        payload["rows"] = [
            self._row(),
            self._row(size_bytes=1048576, p50_us=326.1, p99_us=700.0),
            self._row(load="nic_1mib_stream", p99_us=380.0),
        ]
        res = load_nccl_reference(payload)
        self.assertFalse(res.errors)
        names = sorted(p.name for p in res.profiles)
        self.assertEqual(
            names,
            [
                "nccl:all_reduce:01:00.0->05:00.0",
                "nccl:all_reduce:01:00.0->05:00.0@load=nic_1mib_stream",
            ],
        )
        idle = next(p for p in res.profiles if "@load" not in p.name)
        loaded = next(p for p in res.profiles if "@load" in p.name)
        self.assertEqual(len(idle.points), 2)  # idle rows -> p50 cost model
        self.assertAlmostEqual(idle.points[0].ms, 46.6e-3)
        self.assertAlmostEqual(loaded.points[0].ms, 380.0e-3)  # load -> p99

    def test_missing_field_is_row_error(self):
        payload = new_nccl_reference_envelope()
        bad = self._row()
        del bad["p99_us"]  # the symmetric-load lesson: p99 is mandatory
        payload["rows"] = [bad, self._row()]
        res = load_nccl_reference(payload)
        self.assertEqual(len(res.errors), 1)
        self.assertIn("p99_us", res.errors[0])
        self.assertEqual(len(res.profiles), 1)

    def test_wrong_schema_version_rejected(self):
        payload = new_nccl_reference_envelope()
        payload["schema_version"] = 99
        res = load_nccl_reference(payload)
        self.assertTrue(res.errors)
        self.assertFalse(res.profiles)

    def test_row_fields_frozen(self):
        self.assertEqual(
            NCCL_REFERENCE_ROW_FIELDS,
            {
                "op", "transport", "world", "src_pci", "dst_pci",
                "size_bytes", "iters", "p50_us", "p99_us", "load",
            },
        )


class TestLoadRateTables(CustomTestCase):
    def test_missing_sources_are_loud_and_yield_nothing_measured(self):
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.htccl_path_rates",
            level="WARNING",
        ) as logs:
            res = load_rate_tables()
        self.assertEqual(res.profiles, [])
        # one loud line per missing source
        self.assertEqual(
            len([r for r in logs.records if "not available" in r.getMessage()]), 4
        )

    def test_available_sources_merge_and_apertures_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = os.path.join(tmp, "capability_matrix.json")
            d2d = os.path.join(tmp, "d2d_bench.json")
            with open(cap, "w") as f:
                json.dump(
                    {
                        "schema_version": 3,
                        "kind": "capability_matrix",
                        "pairs": [
                            {
                                "src_pci": "01:00.0",
                                "dst_pci": "05:00.0",
                                "effective_max_single_copy_bytes": 224 * MIB,
                            }
                        ],
                    },
                    f,
                )
            with open(d2d, "w") as f:
                json.dump(
                    {
                        "schema_version": 3,
                        "kind": "d2d_bench",
                        "pairs": [
                            {
                                "src_pci": "01:00.0",
                                "dst_pci": "05:00.0",
                                "mode": "direct",
                                "points": [
                                    {"size_bytes": 64 * KIB, "median_s": 1e-5}
                                ],
                            }
                        ],
                    },
                    f,
                )
            with self.assertLogs(
                "sglang.srt.distributed.device_communicators.htccl_path_rates",
                level="WARNING",
            ):  # the two still-missing sources stay loud
                res = load_rate_tables(p2p_capability_json=cap, p2p_d2d_json=d2d)
        self.assertEqual(len(res.profiles), 1)
        prof = res.profiles[0]
        self.assertEqual(prof.name, "d2d_direct:01:00.0->05:00.0")
        self.assertEqual(prof.aperture_bytes, 224 * MIB)

    def test_unreadable_json_is_error_not_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "broken.json")
            with open(bad, "w") as f:
                f.write("{not json")
            with self.assertLogs(
                "sglang.srt.distributed.device_communicators.htccl_path_rates",
                level="WARNING",
            ):
                res = load_rate_tables(p2p_capability_json=bad)
        self.assertTrue(res.errors)
        self.assertEqual(res.profiles, [])

    def test_load_result_default_shape(self):
        res = LoadResult()
        self.assertEqual(
            (res.profiles, res.apertures, res.errors, res.skipped), ([], {}, [], [])
        )


if __name__ == "__main__":
    unittest.main()
