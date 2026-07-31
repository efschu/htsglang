"""CPU unit tests for the short probe (#213) and the transport choice (#214)."""

import socket
import threading
import time
import unittest

from sglang.srt.rigmon.probe import (
    BUDGET_SECONDS,
    MEASURED,
    CardState,
    ProbeResult,
    echo_server,
    endpoint_id,
    estimate_budget,
    from_hardware_profile,
    measure_host_link,
)
from sglang.srt.rigmon.transport import (
    VERDICT_RECOMMENDED,
    VERDICT_UNAVAILABLE,
    VERDICT_UNKNOWN,
    choose_all_pairs,
    choose_transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


#: Shaped exactly like a real ~/.cache/sglang/hw_profile-*.json on this rig.
HW_PROFILE = {
    "version": 1,
    "driver": "580.95.05",
    "created": "2026-07-21 07:06:43",
    "probe_seconds": 11.4,
    "gpus": {
        "GPU-5090": {
            "name": "NVIDIA GeForce RTX 5090",
            "cuda_index": 0,
            "total_mib": 32607,
            "gemm_tflops": 233.91,
            "membw_gbs": 1664.1,
        },
        "GPU-3080a": {
            "name": "NVIDIA GeForce RTX 3080",
            "cuda_index": 1,
            "total_mib": 20480,
            "gemm_tflops": 63.17,
            "membw_gbs": 718.2,
        },
    },
    "links": {
        "GPU-3080a|GPU-5090": {"p2p_gbs": 2.41},
        "__group__": {"ar_10kb_us": 41.2, "ar_1mb_us": 812.7},
    },
}


class TestBindingTheExistingProbe(CustomTestCase):
    def test_cards_come_across_with_their_measured_ceilings(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        self.assertEqual(len(p.cards), 2)
        c = p.card(endpoint_id("rig1", "GPU-5090"))
        self.assertAlmostEqual(c.gemm_tflops, 233.91)
        self.assertAlmostEqual(c.membw_gbs, 1664.1)
        self.assertEqual(c.total_mib, 32607)

    def test_the_gemm_figure_names_its_dtype(self):
        """A compute ceiling without its dtype is not a ceiling."""
        p = from_hardware_profile(HW_PROFILE, "rig1")
        self.assertTrue(all(c.gemm_dtype for c in p.cards))

    def test_pairs_become_ordered_and_the_reverse_says_it_is_mirrored(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        a, b = endpoint_id("rig1", "GPU-3080a"), endpoint_id("rig1", "GPU-5090")
        fwd, rev = p.link(a, b), p.link(b, a)
        self.assertEqual(fwd.direction, MEASURED)
        self.assertTrue(fwd.measured)
        self.assertFalse(rev.measured)
        self.assertIn("asymmetric", rev.direction)
        self.assertAlmostEqual(fwd.bandwidth_gbs, rev.bandwidth_gbs)

    def test_the_matrix_is_complete_and_the_gap_list_is_empty(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        self.assertEqual(p.missing_pairs(), [])

    def test_group_latency_stays_a_group_figure(self):
        """One all-reduce over every rank is not a per-pair latency; spreading
        it across the pairs would invent numbers."""
        p = from_hardware_profile(HW_PROFILE, "rig1")
        self.assertEqual(len(p.group_latencies), 1)
        self.assertAlmostEqual(p.group_latencies[0].allreduce_10kb_us, 41.2)
        self.assertTrue(all(l.latency_us is None for l in p.links))

    def test_a_bandwidth_carries_the_message_size_it_came_from(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        self.assertEqual(p.links[0].bandwidth_bytes, 1024 * 1024)

    def test_no_profile_is_not_an_empty_probe_but_a_stated_absence(self):
        p = from_hardware_profile(None, "rig1")
        self.assertEqual(p.cards, [])
        self.assertTrue(p.notes)
        self.assertIn("Fit questions", p.notes[0])


class TestStateTravelsWithThePoint(CustomTestCase):
    def _throttled(self):
        return from_hardware_profile(
            HW_PROFILE,
            "rig1",
            card_states={
                "GPU-3080a": CardState(
                    sm_clock_mhz=1695,
                    sm_clock_max_mhz=1905,
                    temp_c=88.0,
                    throttle_reasons=("sw_thermal_slowdown",),
                )
            },
        )

    def test_a_throttled_point_is_kept_and_marked_not_dropped(self):
        p = self._throttled()
        self.assertEqual(len(p.cards), 2)
        self.assertEqual(len(p.throttled_cards), 1)
        card = p.throttled_cards[0]
        self.assertAlmostEqual(card.state.clock_ratio, 1695 / 1905)

    def test_the_warning_names_the_card_the_reason_and_the_consequence(self):
        w = " ".join(self._throttled().state_warnings())
        self.assertIn("sw_thermal_slowdown", w)
        self.assertIn("understates", w)

    def test_a_state_read_later_is_not_claimed_as_the_measurement_state(self):
        """A cached probe read today carries today's clocks. That is a
        different statement from 'this is how the card was when measured'."""
        p = from_hardware_profile(
            HW_PROFILE,
            "rig1",
            card_states={
                "GPU-3080a": CardState(
                    sm_clock_mhz=1695,
                    sm_clock_max_mhz=1905,
                    throttle_reasons=("sw_thermal_slowdown",),
                    sampled_at=time.time(),
                )
            },
        )
        w = " ".join(p.state_warnings())
        self.assertIn("RIGHT NOW", w)
        self.assertIn("different moment", w)
        self.assertNotIn("was throttled while measured", w)

    def test_a_contemporary_state_does_describe_the_measurement(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        at = p.created + 5
        p2 = from_hardware_profile(
            HW_PROFILE,
            "rig1",
            card_states={
                "GPU-3080a": CardState(
                    throttle_reasons=("sw_power_cap",), sampled_at=at
                )
            },
        )
        self.assertIn("while measured", " ".join(p2.state_warnings()))

    def test_state_without_a_timestamp_makes_no_claim_either_way(self):
        w = " ".join(self._throttled().state_warnings())
        self.assertNotIn("accompany these numbers", w)

    def test_mirrored_pairs_are_declared_in_the_warnings(self):
        w = " ".join(from_hardware_profile(HW_PROFILE, "rig1").state_warnings())
        self.assertIn("mirrored", w)

    def test_an_old_probe_reports_its_age(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        future = p.created + 30 * 24 * 3600
        self.assertTrue(p.is_stale(now=future))
        self.assertIn("re-probe", " ".join(p.state_warnings(now=future)).lower())

    def test_a_fresh_probe_says_nothing_about_age(self):
        p = from_hardware_profile(HW_PROFILE, "rig1")
        w = " ".join(p.state_warnings(now=p.created + 60))
        self.assertNotIn("re-probe", w.lower())


class TestBudget(CustomTestCase):
    def test_three_cards_fit_the_thirty_second_target(self):
        b = estimate_budget(3)
        self.assertTrue(b.fits)
        self.assertLessEqual(b.estimate_s, BUDGET_SECONDS)

    def test_measuring_both_directions_is_a_decision_not_a_default(self):
        """On four endpoints it is the difference between fitting and not."""
        self.assertTrue(estimate_budget(4).fits)
        self.assertFalse(estimate_budget(4, measure_both_directions=True).fits)

    def test_the_ordered_pair_count_is_reported_whole(self):
        self.assertEqual(estimate_budget(3).ordered_pairs, 6)


class TestMergeAcrossTheRigBoundary(CustomTestCase):
    def test_two_nodes_merge_without_identity_collisions(self):
        a = from_hardware_profile(HW_PROFILE, "rig1")
        b = from_hardware_profile(HW_PROFILE, "rig2")
        m = a.merge(b)
        self.assertEqual(len(m.cards), 4)
        self.assertEqual(len({c.id for c in m.cards}), 4)
        self.assertTrue(m.crosses_rig_boundary)

    def test_the_cross_node_pairs_are_reported_missing_not_invented(self):
        m = from_hardware_profile(HW_PROFILE, "rig1").merge(
            from_hardware_profile(HW_PROFILE, "rig2")
        )
        missing = m.missing_pairs()
        self.assertTrue(missing)
        for src, dst in missing:
            self.assertNotEqual(src.split("/")[0], dst.split("/")[0])


class TestHostLinkOverLoopback(CustomTestCase):
    def test_it_measures_latency_and_bandwidth_against_a_real_socket(self):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        host, port = srv.getsockname()

        def serve():
            conn, _ = srv.accept()
            echo_server(conn)
            conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            link = measure_host_link(
                host, port, large_bytes=256 * 1024, rounds=5, timeout_s=5.0
            )
        finally:
            srv.close()
            t.join(timeout=5)

        self.assertIsNotNone(link.latency_us)
        self.assertIsNotNone(link.bandwidth_gbs)
        self.assertGreater(link.bandwidth_gbs, 0)
        self.assertFalse(link.same_node)

    def test_it_does_not_claim_to_be_a_device_to_device_figure(self):
        link = measure_host_link("127.0.0.1", 1, timeout_s=0.2)
        self.assertIn("host-to-host", link.transport)

    def test_an_unreachable_peer_is_a_stated_absence_not_a_zero(self):
        link = measure_host_link("127.0.0.1", 1, timeout_s=0.2)
        self.assertIsNone(link.bandwidth_gbs)
        self.assertIn("unreachable", link.note)


class TestTransportChoice(CustomTestCase):
    def _probe(self):
        return from_hardware_profile(HW_PROFILE, "rig1")

    def test_a_staged_pair_is_recognised_from_its_measured_bandwidth(self):
        p = self._probe()
        a, b = endpoint_id("rig1", "GPU-3080a"), endpoint_id("rig1", "GPU-5090")
        c = choose_transport(p.link(a, b))
        self.assertEqual(c.chosen.key, "nccl-pcie")
        self.assertIn("staging regime", c.chosen.reason)

    def test_a_starved_link_says_placement_beats_tuning(self):
        p = self._probe()
        a, b = endpoint_id("rig1", "GPU-3080a"), endpoint_id("rig1", "GPU-5090")
        c = choose_transport(p.link(a, b))
        self.assertIn("placement matters more", c.chosen.reason)

    def test_a_real_p2p_pair_is_recognised_too(self):
        p = self._probe()
        a, b = endpoint_id("rig1", "GPU-3080a"), endpoint_id("rig1", "GPU-5090")
        link = p.link(a, b)
        link.bandwidth_gbs = 240.0
        c = choose_transport(link)
        self.assertEqual(c.chosen.key, "nccl-p2p")

    def test_co_located_ranks_need_no_device_hop(self):
        p = self._probe()
        a, b = endpoint_id("rig1", "GPU-3080a"), endpoint_id("rig1", "GPU-5090")
        c = choose_transport(p.link(a, b), colocated=True)
        self.assertEqual(c.chosen.key, "shm")

    def test_an_unmeasured_pair_yields_unknown_never_a_default(self):
        c = choose_transport(None)
        self.assertIsNone(c.chosen)
        self.assertTrue(all(o.verdict == VERDICT_UNKNOWN for o in c.options))
        self.assertIn("Run the short probe", c.options[0].reason)

    def test_the_choice_carries_the_measurement_it_rests_on(self):
        p = self._probe()
        a, b = endpoint_id("rig1", "GPU-3080a"), endpoint_id("rig1", "GPU-5090")
        c = choose_transport(p.link(a, b))
        self.assertAlmostEqual(c.measurement["bandwidth_gbs"], 2.41)
        self.assertAlmostEqual(c.chosen.evidence["bandwidth_gbs"], 2.41)

    def test_an_unavailable_transport_is_shown_with_what_is_missing(self):
        from sglang.srt.rigmon.probe import LinkRate

        link = LinkRate(src="rig1/a", dst="rig2/b", bandwidth_gbs=1.1,
                        same_node=False, transport="tcp (host-to-host)")
        c = choose_transport(link, available_facilities=[])
        rdma = [o for o in c.options if o.key == "rdma"][0]
        self.assertEqual(rdma.verdict, VERDICT_UNAVAILABLE)
        self.assertEqual(rdma.missing_facilities, ["rdma"])
        self.assertIn("installed rather than", rdma.reason)

    def test_rdma_wins_across_the_boundary_once_it_is_available(self):
        from sglang.srt.rigmon.probe import LinkRate

        link = LinkRate(src="rig1/a", dst="rig2/b", bandwidth_gbs=1.1,
                        same_node=False, transport="tcp (host-to-host)")
        self.assertEqual(
            choose_transport(link, available_facilities=["rdma"]).chosen.key, "rdma"
        )
        self.assertEqual(choose_transport(link).chosen.key, "barlink-ucx")

    def test_intra_and_inter_candidates_do_not_mix(self):
        from sglang.srt.rigmon.probe import LinkRate

        inter = choose_transport(
            LinkRate(src="a", dst="b", bandwidth_gbs=1.0, same_node=False)
        )
        self.assertNotIn("nccl-p2p", {o.key for o in inter.options})
        intra = choose_transport(
            LinkRate(src="a", dst="b", bandwidth_gbs=1.0, same_node=True)
        )
        self.assertNotIn("gloo-tcp", {o.key for o in intra.options})

    def test_every_pair_including_the_gaps_gets_an_entry(self):
        merged = from_hardware_profile(HW_PROFILE, "rig1").merge(
            from_hardware_profile(HW_PROFILE, "rig2")
        )
        choices = choose_all_pairs(merged)
        keys = {(c.src, c.dst) for c in choices}
        for src, dst in merged.missing_pairs():
            self.assertIn((src, dst), keys)
        unknown = [c for c in choices if c.chosen is None]
        self.assertEqual(len(unknown), len(merged.missing_pairs()))


if __name__ == "__main__":
    unittest.main()
