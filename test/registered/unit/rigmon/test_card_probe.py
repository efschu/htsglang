"""CPU unit tests for the short card probe (#213): rates, ordered pair matrix,
cache, and the projection onto the probe data model.

Every GPU-touching step is stubbed. What is under test is the part that has to
be right whether or not a card is present: the honesty rules (a missing fp8
rate stays missing, a host-staged pair says so, a throttled point is kept and
marked), the cache key, and the orchestration order.
"""

import dataclasses
import json
import os
import tempfile
import time
import unittest

from sglang.srt.rigmon import card_probe as cp
from sglang.srt.rigmon.probe import MEASURED
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _card(uuid, name, **kw):
    kw.setdefault("cuda_index", 0)
    kw.setdefault("membw_read_gbs", 1000.0)
    kw.setdefault("membw_copy_gbs", 900.0)
    kw.setdefault("membw_gemv_gbs", 800.0)
    kw.setdefault("gemm_bf16_tflops", 100.0)
    return cp.CardProbeMeasurement(uuid=uuid, name=name, **kw)


class TestCardMeasurement(CustomTestCase):
    def test_membw_score_is_the_streaming_peak_not_the_gemv_rate(self):
        c = _card(
            "u",
            "card",
            membw_read_gbs=1660.0,
            membw_copy_gbs=1601.0,
            membw_gemv_gbs=1400.0,
        )
        self.assertEqual(c.membw_gbs, 1660.0)
        # The decode divisor stays its own number.
        self.assertEqual(c.membw_gemv_gbs, 1400.0)

    def test_membw_score_absent_when_nothing_was_measured(self):
        c = cp.CardProbeMeasurement(uuid="u", name="card", cuda_index=0)
        self.assertIsNone(c.membw_gbs)

    def test_throttle_state_and_clock_ratio(self):
        c = _card(
            "u",
            "card",
            sm_clock_mhz=1695,
            sm_clock_max_mhz=1905,
            throttle_reasons=["sw_thermal_slowdown"],
        )
        self.assertTrue(c.throttled)
        self.assertAlmostEqual(c.clock_ratio, 1695 / 1905)

    def test_round_trip_json(self):
        c = _card(
            "u",
            "card",
            gemm_fp8_tflops=None,
            fp8_note="no fp8 path",
            h2d_gbs=11.5,
            d2h_gbs=12.5,
            temp_c=88.0,
        )
        back = cp.CardProbeMeasurement.from_json(json.loads(json.dumps(c.to_json())))
        self.assertEqual(back.uuid, "u")
        self.assertIsNone(back.gemm_fp8_tflops)
        self.assertEqual(back.fp8_note, "no fp8 path")
        self.assertEqual(back.h2d_gbs, 11.5)
        self.assertEqual(back.d2h_gbs, 12.5)
        # Derived fields are emitted for readers but must not become state:
        # the three membw rates stay apart in storage.
        self.assertIn("membw_gbs", c.to_json())
        fields = {f.name for f in dataclasses.fields(cp.CardProbeMeasurement)}
        self.assertNotIn("membw_gbs", fields)


class TestPairMatrix(CustomTestCase):
    def test_ordered_lookup_distinguishes_the_two_directions(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[_card("a", "A"), _card("b", "B")],
            pairs=[
                cp.PairMeasurement("a", "b", bandwidth_gbs=5.1),
                cp.PairMeasurement("b", "a", bandwidth_gbs=3.2),
            ],
        )
        self.assertEqual(p.pair("a", "b").bandwidth_gbs, 5.1)
        self.assertEqual(p.pair("b", "a").bandwidth_gbs, 3.2)
        self.assertIsNone(p.pair("a", "a"))

    def test_transport_label_travels_with_the_number(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[_card("a", "A"), _card("b", "B")],
            pairs=[cp.PairMeasurement("a", "b", bandwidth_gbs=5.1)],
        )
        self.assertEqual(p.transports, [cp.HOST_STAGING])
        self.assertTrue(
            any("HOST-STAGING" in c for c in p.caveats()),
            p.caveats(),
        )

    def test_p2p_matrix_does_not_carry_the_host_staging_caveat(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[_card("a", "A"), _card("b", "B")],
            pairs=[
                cp.PairMeasurement(
                    "a",
                    "b",
                    bandwidth_gbs=200.0,
                    transport=cp.P2P_DIRECT,
                    peer_access=True,
                )
            ],
        )
        self.assertFalse(any("HOST-STAGING" in c for c in p.caveats()))


class TestCaveats(CustomTestCase):
    def test_throttled_card_is_kept_and_marked(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[
                _card(
                    "a",
                    "A",
                    sm_clock_mhz=1695,
                    sm_clock_max_mhz=1905,
                    throttle_reasons=["sw_thermal_slowdown"],
                )
            ],
        )
        # Kept.
        self.assertEqual(len(p.cards), 1)
        # And marked.
        self.assertTrue(any("throttled while measured" in c for c in p.caveats()))

    def test_partial_fp8_coverage_is_named(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[
                _card("a", "RTX 5090", gemm_fp8_tflops=400.0),
                _card(
                    "b",
                    "RTX 3080",
                    gemm_fp8_tflops=None,
                    fp8_note="compute capability 8.6 has no fp8 tensor path",
                ),
            ],
        )
        cav = " ".join(p.caveats())
        self.assertIn("RTX 3080", cav)
        self.assertIn("no fp8 tensor path", cav)

    def test_uniform_absence_of_fp8_is_not_a_per_card_caveat(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[_card("a", "RTX 3080"), _card("b", "RTX 3080")],
        )
        self.assertFalse(any("fp8" in c for c in p.caveats()))

    def test_stale_probe_says_so(self):
        old = time.time() - 30 * 24 * 3600
        p = cp.CardProbeProfile(created=old, cards=[_card("a", "A")])
        self.assertTrue(p.is_stale())
        self.assertTrue(any("re-probe" in c for c in p.caveats()))

    def test_fresh_probe_has_no_age_caveat(self):
        p = cp.CardProbeProfile(created=time.time(), cards=[_card("a", "A")])
        self.assertFalse(p.is_stale())
        self.assertFalse(any("re-probe" in c for c in p.caveats()))


class TestCache(CustomTestCase):
    def test_key_covers_cards_driver_and_version(self):
        a = cp.card_probe_cache_path(["u1", "u2"], "580.95.05")
        # Order of the uuid list must not matter.
        self.assertEqual(a, cp.card_probe_cache_path(["u2", "u1"], "580.95.05"))
        # A driver update is a different key: it moves clocks and the p2p
        # verdict, so the old rates must not be reused silently.
        self.assertNotEqual(a, cp.card_probe_cache_path(["u1", "u2"], "581.0"))
        # A different rig is a different key.
        self.assertNotEqual(a, cp.card_probe_cache_path(["u1", "u3"], "580.95.05"))

    def test_save_load_round_trip(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            created_str="2026-07-28 01:00:00",
            driver="580.95.05",
            cards=[_card("a", "A", h2d_gbs=11.0), _card("b", "B")],
            pairs=[cp.PairMeasurement("a", "b", bandwidth_gbs=5.1, latency_us=290.0)],
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "card_probe-test.json")
            cp.save_card_probe(p, path)
            back = cp.load_card_probe(path)
        self.assertIsNotNone(back)
        self.assertEqual(back.driver, "580.95.05")
        self.assertEqual(len(back.cards), 2)
        self.assertEqual(back.cards[0].h2d_gbs, 11.0)
        self.assertEqual(back.pair("a", "b").latency_us, 290.0)

    def test_version_mismatch_is_ignored_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "card_probe-old.json")
            with open(path, "w") as f:
                json.dump({"version": cp.CARD_PROBE_VERSION + 1, "cards": []}, f)
            self.assertIsNone(cp.load_card_probe(path))

    def test_missing_and_corrupt_files_are_absent_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(cp.load_card_probe(os.path.join(d, "nope.json")))
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as f:
                f.write("{not json")
            self.assertIsNone(cp.load_card_probe(bad))

    def test_measured_rates_are_empty_without_a_probe(self):
        self.assertEqual(cp.measured_card_rates(cp.CardProbeProfile()), {})

    def test_measured_rates_expose_every_rate_a_ranking_needs(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[_card("a", "A", gemm_fp8_tflops=400.0, h2d_gbs=11.0, d2h_gbs=12.0)],
        )
        r = cp.measured_card_rates(p)["a"]
        for k in (
            "membw_gbs",
            "membw_gemv_gbs",
            "gemm_bf16_tflops",
            "gemm_fp8_tflops",
            "h2d_gbs",
            "d2h_gbs",
            "throttled",
        ):
            self.assertIn(k, r)


class TestProjectionOntoProbeResult(CustomTestCase):
    def test_both_directions_are_measured_never_mirrored(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            node_id="rig1",
            cards=[_card("a", "A"), _card("b", "B")],
            pairs=[
                cp.PairMeasurement("a", "b", bandwidth_gbs=5.1),
                cp.PairMeasurement("b", "a", bandwidth_gbs=3.2),
            ],
        )
        r = cp.to_probe_result(p)
        self.assertEqual([x.direction for x in r.links], [MEASURED, MEASURED])
        self.assertEqual(r.missing_pairs(), [])
        self.assertEqual(r.link("rig1/a", "rig1/b").bandwidth_gbs, 5.1)
        self.assertEqual(r.link("rig1/b", "rig1/a").bandwidth_gbs, 3.2)

    def test_new_rates_survive_the_projection(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[
                _card(
                    "a",
                    "A",
                    gemm_fp8_tflops=400.0,
                    h2d_gbs=11.0,
                    d2h_gbs=12.0,
                    membw_gemv_gbs=1400.0,
                )
            ],
        )
        card = cp.to_probe_result(p).cards[0]
        self.assertEqual(card.gemm_fp8_tflops, 400.0)
        self.assertEqual(card.membw_gemv_gbs, 1400.0)
        self.assertEqual(card.h2d_gbs, 11.0)
        self.assertEqual(card.d2h_gbs, 12.0)

    def test_state_travels_into_the_projection(self):
        p = cp.CardProbeProfile(
            created=1000.0,
            cards=[
                _card(
                    "a",
                    "A",
                    temp_c=88.0,
                    sm_clock_mhz=1695,
                    sm_clock_max_mhz=1905,
                    throttle_reasons=["sw_thermal_slowdown"],
                )
            ],
        )
        card = cp.to_probe_result(p).cards[0]
        self.assertTrue(card.throttled)
        self.assertEqual(card.state.temp_c, 88.0)
        # The state describes the moment of the measurement, so it carries the
        # probe's own timestamp rather than "now".
        self.assertEqual(card.state.sampled_at, 1000.0)


class TestRunOrchestration(CustomTestCase):
    """``run_card_probe`` without a GPU: the sequencing and the bookkeeping."""

    def _stub(self, n_cards=3, driver="580.95.05"):
        gpus = [
            {"cuda_index": i, "uuid": f"u{i}", "name": f"card{i}", "total_mib": 20480}
            for i in range(n_cards)
        ]
        calls = {"cards": [], "pairs": 0}

        def fake_inventory():
            return gpus, driver

        def fake_states():
            return {
                f"u{i}": {
                    "sm_clock_mhz": 1900,
                    "sm_clock_max_mhz": 1905,
                    "temp_c": 50.0,
                    "throttle_reasons": [],
                }
                for i in range(n_cards)
            }

        def fake_measure_card(cuda_index, uuid, name, total_mib=None, state_fn=None):
            calls["cards"].append(uuid)
            st = (state_fn or (lambda: {}))() or {}
            return _card(
                uuid,
                name,
                cuda_index=cuda_index,
                total_mib=total_mib,
                sm_clock_mhz=st.get("sm_clock_mhz"),
            )

        def fake_matrix(g):
            calls["pairs"] += 1
            return [
                cp.PairMeasurement(a["uuid"], b["uuid"], bandwidth_gbs=5.0)
                for a in g
                for b in g
                if a["uuid"] != b["uuid"]
            ]

        return gpus, calls, fake_inventory, fake_states, fake_measure_card, fake_matrix

    def test_every_card_and_every_ordered_pair(self):
        _, calls, inv, states, mcard, matrix = self._stub()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            orig = (
                cp._inventory,
                cp._card_states,
                cp.measure_card,
                cp.measure_pair_matrix,
            )
            cp._inventory, cp._card_states = inv, states
            cp.measure_card, cp.measure_pair_matrix = mcard, matrix
            try:
                prof = cp.run_card_probe(node_id="rig1", path=path)
            finally:
                (
                    cp._inventory,
                    cp._card_states,
                    cp.measure_card,
                    cp.measure_pair_matrix,
                ) = orig
            self.assertTrue(os.path.exists(path))
        self.assertEqual(calls["cards"], ["u0", "u1", "u2"])
        self.assertEqual(calls["pairs"], 1)
        # n*(n-1) ordered pairs, not n*(n-1)/2.
        self.assertEqual(len(prof.pairs), 6)
        self.assertEqual(prof.driver, "580.95.05")
        self.assertEqual(prof.node_id, "rig1")
        self.assertGreater(prof.created, 0)
        self.assertIsNotNone(prof.duration_s)
        self.assertTrue(any("cached to" in n for n in prof.notes))

    def test_single_card_has_no_pair_matrix_and_says_so(self):
        _, calls, inv, states, mcard, matrix = self._stub(n_cards=1)
        orig = (cp._inventory, cp._card_states, cp.measure_card, cp.measure_pair_matrix)
        cp._inventory, cp._card_states = inv, states
        cp.measure_card, cp.measure_pair_matrix = mcard, matrix
        try:
            prof = cp.run_card_probe(save=False)
        finally:
            (
                cp._inventory,
                cp._card_states,
                cp.measure_card,
                cp.measure_pair_matrix,
            ) = orig
        self.assertEqual(calls["pairs"], 0)
        self.assertEqual(prof.pairs, [])
        self.assertTrue(any("no pair matrix" in n for n in prof.notes))

    def test_progress_reports_every_step(self):
        _, _, inv, states, mcard, matrix = self._stub()
        seen = []
        orig = (cp._inventory, cp._card_states, cp.measure_card, cp.measure_pair_matrix)
        cp._inventory, cp._card_states = inv, states
        cp.measure_card, cp.measure_pair_matrix = mcard, matrix
        try:
            cp.run_card_probe(
                save=False, progress=lambda d, t, label: seen.append((d, t, label))
            )
        finally:
            (
                cp._inventory,
                cp._card_states,
                cp.measure_card,
                cp.measure_pair_matrix,
            ) = orig
        # 3 cards + the pair matrix + the terminal call.
        self.assertEqual(len(seen), 5)
        self.assertEqual(seen[0][0], 0)
        self.assertEqual(seen[-1], (4, 4, "done"))

    def test_budget_estimate_covers_this_rig(self):
        from sglang.srt.rigmon.probe import estimate_budget

        b = estimate_budget(3)
        self.assertLessEqual(b.estimate_s, cp.DEFAULT_MAX_AGE_S)
        self.assertTrue(b.fits, b.to_json())


class TestJobStore(CustomTestCase):
    """Start + poll, with the measurement itself replaced."""

    def _store(self, runner):
        store = cp.ProbeJobStore()
        store.synchronous = True
        store.runner = runner
        return store

    def test_a_finished_job_carries_the_profile_and_its_path(self):
        prof = cp.CardProbeProfile(created=time.time(), cards=[_card("a", "A")])
        store = self._store(lambda node_id: (prof, "/tmp/x.json"))
        job = store.start()
        self.assertEqual(job.state, cp.OK)
        self.assertEqual(job.path, "/tmp/x.json")
        self.assertEqual(store.get(job.job_id).profile, prof)
        self.assertIsNotNone(job.to_json()["elapsed_s"])
        # Nothing is running any more, so a poller can stop.
        self.assertIsNone(store.active())

    def test_a_failure_carries_a_reason_and_a_remedy(self):
        def boom(node_id):
            raise RuntimeError("no cards visible")

        store = self._store(boom)
        job = store.start()
        self.assertEqual(job.state, cp.ERROR)
        self.assertIn("no cards visible", job.error)
        self.assertTrue(job.remedy)
        self.assertIsNone(job.profile)

    def test_a_second_start_joins_the_running_one(self):
        import threading

        gate = threading.Event()
        prof = cp.CardProbeProfile(created=time.time(), cards=[_card("a", "A")])
        calls = []

        def slow(node_id):
            calls.append(node_id)
            gate.wait(5)
            return prof, "/tmp/x.json"

        store = cp.ProbeJobStore()
        store.runner = slow
        first = store.start()
        second = store.start()
        # Two probes on the same cards would measure each other.
        self.assertEqual(first.job_id, second.job_id)
        gate.set()
        for _ in range(200):
            if store.get(first.job_id).state != cp.RUNNING:
                break
            time.sleep(0.02)
        self.assertEqual(store.get(first.job_id).state, cp.OK)
        self.assertEqual(len(calls), 1)

    def test_unknown_job_id_is_absent(self):
        self.assertIsNone(cp.ProbeJobStore().get("nope"))


class TestTextRendering(CustomTestCase):
    def test_absent_rates_render_as_absent_not_as_zero(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            created_str="2026-07-28 01:00:00",
            cards=[_card("a", "RTX 3080", gemm_fp8_tflops=None, h2d_gbs=None)],
        )
        txt = cp.format_text(p)
        self.assertIn("RTX 3080", txt)
        # A dash, never a 0.0 that would read as a measurement.
        self.assertIn("-", txt)
        self.assertNotIn("0.0 GB/s", txt)

    def test_every_pair_row_names_its_path(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[_card("a", "A"), _card("b", "B")],
            pairs=[cp.PairMeasurement("a", "b", bandwidth_gbs=4.4, latency_us=22.0)],
        )
        txt = cp.format_text(p)
        self.assertIn("via " + cp.HOST_STAGING, txt)


class TestNoDuplicateWarnings(CustomTestCase):
    def test_the_projection_does_not_repeat_what_probe_result_derives(self):
        p = cp.CardProbeProfile(
            created=time.time(),
            cards=[
                _card("a", "RTX 5090", gemm_fp8_tflops=400.0),
                _card(
                    "b",
                    "RTX 3080",
                    sm_clock_mhz=1695,
                    sm_clock_max_mhz=1905,
                    throttle_reasons=["sw_power_cap"],
                ),
            ],
            pairs=[cp.PairMeasurement("a", "b", bandwidth_gbs=4.4)],
        )
        r = cp.to_probe_result(p)
        lines = list(r.notes) + list(r.state_warnings())
        throttle = [x for x in lines if "throttled" in x.lower()]
        self.assertEqual(len(throttle), 1, lines)
        # The two the ProbeResult cannot derive are carried, exactly once each.
        self.assertEqual(len([x for x in lines if "fp8" in x]), 1, lines)
        self.assertEqual(len([x for x in lines if "HOST-STAGING" in x]), 1, lines)

    def test_rate_caveats_exclude_state_and_age(self):
        p = cp.CardProbeProfile(
            created=time.time() - 30 * 24 * 3600,
            cards=[
                _card(
                    "a",
                    "RTX 3080",
                    throttle_reasons=["sw_power_cap"],
                )
            ],
        )
        rc = p.rate_caveats()
        self.assertFalse(any("throttled" in x for x in rc), rc)
        self.assertFalse(any("re-probe" in x for x in rc), rc)
        # But the full list still has both.
        full = p.caveats()
        self.assertTrue(any("throttled" in x for x in full))
        self.assertTrue(any("re-probe" in x for x in full))


if __name__ == "__main__":
    unittest.main()
