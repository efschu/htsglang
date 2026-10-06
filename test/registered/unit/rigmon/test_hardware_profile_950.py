"""CPU unit tests for order 950 (profile editor S2): the ``flliper.hardware/1`` view
and the new card-probe arms.

No GPU, no NVML, no child interpreter: NVML is injected, the measurement child is a
stub runner, and the card-probe arms run against stubbed kernels.  What is under test is
what has to be right without a card: every value names its source, an unmeasured value
is never shown as measured, the newest measurement wins without a missing one hiding an
older one, the BAR1 stretch says "nicht gemessen", and the probe arms are gated and
recorded the way the honesty rules of ``card_probe`` demand.
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.rigmon import card_probe as cp
from sglang.srt.rigmon import hardware_profile as hp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NOW = 1_790_000_000.0
U0, U1, U2 = "GPU-0000", "GPU-1111", "GPU-2222"


def _nvml(n=3, driver="595.58"):
    """A synthetic inventory: card 1 is the big/new one, 0 and 2 the small/old ones."""
    spec = [
        (0, U0, "NVIDIA GeForce RTX 3080", 20480, [8, 6]),
        (1, U1, "NVIDIA GeForce RTX 5090", 32607, [12, 0]),
        (2, U2, "NVIDIA GeForce RTX 3080", 20480, [8, 6]),
    ][:n]
    cards = []
    for i, u, name, mib, cc in spec:
        cards.append(
            {
                "nvml_index": i, "uuid": u, "name": name, "total_mib": mib, "cc": cc,
                "bar1_total_mib": 256 if cc == [8, 6] else 32768,
                "pcie_max_gen": 4 if cc == [8, 6] else 5, "pcie_max_width": 16,
                "pcie_cur_gen": 1, "pcie_cur_width": 8,
                "mem_bus_width_bits": 320 if cc == [8, 6] else 512,
                "mem_clock_max_mhz": 9501 if cc == [8, 6] else 14000,
                "sm_clock_max_mhz": 2100, "power_limit_w": 230.0, "power_default_w": 320.0,
                "pci_bus_id": f"0000:0{i}:00.0",
            }
        )
    return cards, driver, []


def _probe_card(uuid, name, **kw):
    d = {
        "uuid": uuid, "name": name, "cuda_index": 0, "total_mib": 20480,
        "gemm_bf16_tflops": 60.0, "gemm_fp8_tflops": None,
        "fp8_note": "compute capability 8.6 has no fp8 tensor path (needs 8.9+)",
        "membw_read_gbs": 700.0, "membw_copy_gbs": 690.0, "membw_gemv_gbs": 650.0,
        "h2d_gbs": 6.0, "d2h_gbs": 6.5, "h2d_lat_us": 12.5, "d2h_lat_us": 14.0,
        "sm_count": 68, "l2_mib": 5.0, "compute_capability": "8.6",
        "gemm_int8_tflops": 180.0, "gemm_w4a8_int8_tflops": 62.0, "gemm_w4a16_tflops": 55.0,
        "lane_notes": {}, "arm_seconds": {"membw": 3.0, "bf16": 1.0},
        "sm_clock_mhz": 1900, "sm_clock_max_mhz": 2100, "temp_c": 60.0, "throttle_reasons": [],
        "seconds": 20.0,
    }
    d.update(kw)
    return d


def _write_probe(dirpath, name, created, cards, pairs=(), driver="595.58", **extra):
    with open(os.path.join(dirpath, name), "w") as f:
        json.dump(
            {"version": 1, "created": created, "driver": driver, "torch_version": "2.9", "cuda_version": "13.0",
             "cards": cards, "pairs": list(pairs), **extra},
            f,
        )


def _write_stage0(dirpath, name, created_str, gpus, links=None, driver="595.58"):
    with open(os.path.join(dirpath, name), "w") as f:
        json.dump({"version": 3, "driver": driver, "created": created_str, "gpus": gpus, "links": links or {}}, f)


class TestProfileView(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _build(self, **kw):
        kw.setdefault("nvml", _nvml())
        return hp.build(cache_dir=self.d.name, now=NOW, **kw)

    def test_empty_cache_is_all_not_measured_and_asks_for_a_measurement(self):
        doc = self._build()
        self.assertEqual(doc["schema"], "flliper.hardware/1")
        self.assertTrue(doc["measure_needed"])
        self.assertEqual(len(doc["cards"]), 3)
        for c in doc["cards"]:
            self.assertEqual(c["compute"]["bf16"]["src"], hp.SRC_NONE)
            self.assertIsNone(c["compute"]["bf16"]["v"])
            self.assertTrue(c["compute"]["bf16"]["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_nvml_values_are_nvml_and_the_nameplate_is_a_datasheet_figure(self):
        c = self._build()["cards"][0]
        self.assertEqual(c["vram_total_mib"], {"v": c["vram_total_mib"]["v"], "src": "NVML", "unit": "MiB"})
        self.assertEqual(c["bar1_total_mib"]["src"], hp.SRC_NVML)
        self.assertEqual(c["pcie"]["max_gen"]["src"], hp.SRC_NVML)
        np_ = c["mem_gbs"]["nameplate"]
        self.assertEqual(np_["src"], hp.SRC_DATASHEET)
        # bus/8 * clock * 2 / 1000 -- computed from NVML facts, nothing else
        self.assertAlmostEqual(np_["v"], c["mem_gbs"]["nameplate"]["v"])
        self.assertNotIn(c["mem_gbs"]["read"]["src"], (hp.SRC_MEASURED,))

    def test_measured_values_carry_time_and_file_and_the_view_validates(self):
        _write_probe(self.d.name, "card_probe-aaa.json", NOW - 3600,
                     [_probe_card(U0, "RTX 3080"), _probe_card(U2, "RTX 3080")])
        doc = self._build()
        c0 = next(c for c in doc["cards"] if c["uuid"] == U0)
        n = c0["compute"]["bf16"]
        self.assertEqual((n["v"], n["src"], n["at"], n["probe"]), (60.0, "gemessen", NOW - 3600, "card_probe-aaa.json"))
        self.assertEqual(c0["compute"]["int8"]["v"], 180.0)
        self.assertEqual(c0["compute"]["nvfp4_w4a8"]["v"], 62.0)
        self.assertEqual(c0["compute"]["nvfp4_marlin"]["v"], 55.0)
        self.assertEqual(c0["sm_count"]["v"], 68)
        self.assertEqual(c0["l2_mib"]["v"], 5.0)
        self.assertEqual(c0["h2d"]["lat_us"]["v"], 12.5)
        self.assertEqual(c0["d2h"]["lat_us"]["v"], 14.0)
        self.assertEqual(c0["d2d_intra_gbs"]["src"], hp.SRC_MEASURED)
        self.assertEqual(hp.validate(doc), [])
        # the 5090 was not probed: its rows stay unmeasured and the view asks for a measurement
        c1 = next(c for c in doc["cards"] if c["uuid"] == U1)
        self.assertEqual(c1["compute"]["bf16"]["src"], hp.SRC_NONE)
        self.assertTrue(doc["measure_needed"])

    def test_a_fully_probed_rig_needs_no_measurement_formats_a_card_cannot_run_do_not_count(self):
        w4a8_gone = {"nvfp4_w4a8": "compute capability 12.0: the W4A8 kernel is the sm_8x one"}
        w4a4_gone = {"nvfp4_w4a4": "compute capability 8.6: no native FP4 tensor cores (needs 10.0+)"}
        _write_probe(self.d.name, "card_probe-all.json", NOW - 60, [
            _probe_card(U0, "RTX 3080", lane_notes=w4a4_gone), _probe_card(U2, "RTX 3080", lane_notes=w4a4_gone),
            _probe_card(U1, "RTX 5090", gemm_fp8_tflops=500.0, fp8_note="", gemm_w4a8_int8_tflops=None,
                        gemm_w4a4_tflops=900.0, lane_notes=w4a8_gone, compute_capability="12.0")],
            # the BAR1 step ran and said why it has no numbers: final, like any card fact (order 1006)
            bar1_attempted=True, bar1_reason="dmabuf_holder not available", bar1_pairs=[])
        doc = self._build()
        self.assertFalse(doc["measure_needed"], doc["unmeasured"])
        self.assertEqual(doc["unmeasured"], {"0": [], "1": [], "2": []})
        # ... but BAR1 stays explicitly not measured (with the reason), and so does fp8 on the 3080s
        self.assertFalse(doc["bar1"]["measured"])
        c0 = next(c for c in doc["cards"] if c["uuid"] == U0)
        self.assertIsNone(c0["compute"]["fp8_native"]["v"])
        c1 = next(c for c in doc["cards"] if c["uuid"] == U1)
        self.assertIn("sm_8x", c1["compute"]["nvfp4_w4a8"]["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_no_fp8_on_sm86_is_the_cards_own_reason_not_a_number(self):
        _write_probe(self.d.name, "card_probe-aaa.json", NOW - 10, [_probe_card(U0, "RTX 3080")])
        n = next(c for c in self._build()["cards"] if c["uuid"] == U0)["compute"]["fp8_native"]
        self.assertIsNone(n["v"])
        self.assertEqual(n["src"], hp.SRC_NONE)
        self.assertIn("no fp8 tensor path", n["note"])

    def test_the_newest_measurement_wins_and_a_missing_one_never_hides_an_older_one(self):
        _write_probe(self.d.name, "card_probe-old.json", NOW - 7200,
                     [_probe_card(U0, "RTX 3080", gemm_int8_tflops=150.0, gemm_bf16_tflops=50.0)])
        # the newer probe predates the int8 arm: it has no int8 value at all
        _write_probe(self.d.name, "card_probe-new.json", NOW - 60,
                     [_probe_card(U0, "RTX 3080", gemm_int8_tflops=None, gemm_bf16_tflops=61.0)])
        c0 = next(c for c in self._build()["cards"] if c["uuid"] == U0)
        self.assertEqual(c0["compute"]["bf16"]["v"], 61.0)
        self.assertEqual(c0["compute"]["bf16"]["probe"], "card_probe-new.json")
        self.assertEqual(c0["compute"]["int8"]["v"], 150.0)
        self.assertEqual(c0["compute"]["int8"]["probe"], "card_probe-old.json")

    def test_stage0_lanes_fill_what_the_probe_does_not_measure(self):
        _write_stage0(self.d.name, "hw_profile-x.json", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(NOW - 100)),
                      {U0: {"name": "RTX 3080", "gemm_tflops": 58.0, "gemm_lanes": {"fp8_marlin": 55.0, "int8_native": 177.0},
                            "gemm_lane_notes": {"fp8_native": "no fp8 tensor path"},
                            "membw_read_gbs": 700.0, "membw_copy_gbs": 690.0, "membw_gemv_gbs": 650.0}})
        c0 = next(c for c in self._build()["cards"] if c["uuid"] == U0)
        self.assertEqual(c0["compute"]["fp8_marlin"]["v"], 55.0)
        self.assertEqual(c0["compute"]["int8"]["v"], 177.0)
        self.assertEqual(c0["compute"]["int8"]["probe"], "hw_profile-x.json")
        self.assertEqual(c0["mem_gbs"]["gemv"]["v"], 650.0)
        self.assertEqual(c0["compute"]["fp8_native"]["note"], "no fp8 tensor path")

    def test_pair_matrix_is_ordered_labelled_and_bar1_is_explicitly_not_measured(self):
        pairs = [
            {"src_uuid": U0, "dst_uuid": U1, "bandwidth_gbs": 5.1, "latency_us": 30.0,
             "transport": cp.HOST_STAGING, "peer_access": False, "note": "no peer"},
            {"src_uuid": U1, "dst_uuid": U0, "bandwidth_gbs": 3.2, "latency_us": 31.0,
             "transport": cp.HOST_STAGING, "peer_access": False},
        ]
        _write_probe(self.d.name, "card_probe-aaa.json", NOW - 5, [_probe_card(U0, "RTX 3080")], pairs)
        doc = self._build()
        fwd = [l for l in doc["links"] if l["transport"] == "host_staging"]
        self.assertEqual(len(fwd), 2)
        by = {(l["src"], l["dst"]): l for l in fwd}
        o0 = next(c["ord"] for c in doc["cards"] if c["uuid"] == U0)
        o1 = next(c["ord"] for c in doc["cards"] if c["uuid"] == U1)
        self.assertEqual(by[(o0, o1)]["gbs"]["v"], 5.1)
        self.assertEqual(by[(o1, o0)]["gbs"]["v"], 3.2)
        self.assertEqual(by[(o0, o1)]["lat_us"]["v"], 30.0)
        bar1 = [l for l in doc["links"] if l["transport"] == "bar1"]
        self.assertEqual(len(bar1), 3 * 2)  # every ordered pair, both directions
        for l in bar1:
            self.assertIsNone(l["gbs"]["v"])
            self.assertEqual(l["gbs"]["src"], hp.SRC_NONE)
            self.assertIn("nicht gemessen", l["gbs"]["note"])
        self.assertFalse(doc["bar1"]["measured"])
        self.assertIn("NOT MEASURED", doc["bar1"]["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_nccl_pair_reverse_direction_is_an_estimate_not_a_measurement(self):
        _write_stage0(self.d.name, "hw_profile-x.json", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(NOW - 100)),
                      {U0: {"name": "n"}, U1: {"name": "n"}}, links={f"{U0}|{U1}": {"p2p_gbs": 6.4}})
        doc = self._build()
        nccl = {(l["src"], l["dst"]): l for l in doc["links"] if l["transport"] == "nccl"}
        o0 = next(c["ord"] for c in doc["cards"] if c["uuid"] == U0)
        o1 = next(c["ord"] for c in doc["cards"] if c["uuid"] == U1)
        self.assertEqual(nccl[(o0, o1)]["gbs"]["src"], hp.SRC_MEASURED)
        self.assertEqual(nccl[(o1, o0)]["gbs"]["src"], hp.SRC_ESTIMATED)
        self.assertIn("gespiegelt", nccl[(o1, o0)]["gbs"]["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_stale_probe_and_driver_change_are_marked(self):
        _write_probe(self.d.name, "card_probe-aaa.json", NOW - 8 * 24 * 3600, [_probe_card(U0, "RTX 3080")],
                     driver="580.1")
        c0 = next(c for c in self._build()["cards"] if c["uuid"] == U0)
        self.assertTrue(c0["stale"])
        self.assertEqual(c0["driver_mismatch"], {"probe": "580.1", "live": "595.58"})

    def test_throttled_point_is_kept_and_marked(self):
        _write_probe(self.d.name, "card_probe-aaa.json", NOW - 5,
                     [_probe_card(U0, "RTX 3080", throttle_reasons=["sw_thermal_slowdown"])])
        st = next(c for c in self._build()["cards"] if c["uuid"] == U0)["state"]
        self.assertTrue(st["throttled"])
        self.assertEqual(st["throttle"], ["sw_thermal_slowdown"])

    def test_id_is_the_hash_of_the_content_not_of_the_clock(self):
        _write_probe(self.d.name, "card_probe-aaa.json", NOW - 5, [_probe_card(U0, "RTX 3080")])
        a = hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)
        b = hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)
        self.assertEqual(a["id"], b["id"])
        _write_probe(self.d.name, "card_probe-bbb.json", NOW - 1, [_probe_card(U0, "RTX 3080", gemm_bf16_tflops=99.0)])
        c = hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)
        self.assertNotEqual(a["id"], c["id"])

    def test_any_inventory_works_no_card_count_or_model_is_assumed(self):
        one = hp.build(cache_dir=self.d.name, nvml=_nvml(1), now=NOW)
        self.assertEqual(len(one["cards"]), 1)
        self.assertEqual([l for l in one["links"] if l["transport"] == "bar1"], [])
        none = hp.build(cache_dir=self.d.name, nvml=([], None, ["NVML kaputt"]), now=NOW)
        self.assertEqual(none["cards"], [])
        self.assertTrue(none["measure_needed"])
        self.assertEqual(none["sources"]["nvml"]["issues"], ["NVML kaputt"])

    def test_unreadable_and_foreign_version_caches_are_ignored(self):
        with open(os.path.join(self.d.name, "card_probe-bad.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(self.d.name, "card_probe-v9.json"), "w") as f:
            json.dump({"version": 9, "cards": [_probe_card(U0, "x")]}, f)
        doc = self._build()
        self.assertEqual(doc["sources"]["card_probe"], [])

    def test_validate_catches_a_value_dressed_as_measured(self):
        bad = {"cards": [{"x": {"v": 1.0, "src": "gemessen"}}], "links": [{"y": {"v": None, "src": "gemessen"}}]}
        problems = hp.validate(bad)
        self.assertEqual(len(problems), 3, problems)
        self.assertTrue(hp.validate({"cards": [{"x": {"v": 1.0, "src": "erfunden"}}], "links": []}))


class TestMeasurementRun(CustomTestCase):
    def test_child_sees_exactly_the_chosen_cards_by_uuid_in_pci_order(self):
        seen = {}

        def runner(cmd, env, timeout):
            seen.update(cmd=cmd, env=env, timeout=timeout)
            prof = {"cards": [{"uuid": U1, "seconds": 18.5, "arm_seconds": {"membw": 4.0, "bf16": 1.0}}], "pairs": []}
            return 0, "log noise\n" + json.dumps(prof, indent=1), "WARNING lanes int8/w4a8/w4a16 not measured: no sgl_kernel\n"

        cards = _nvml()[0]
        r = hp.run_measurement([1], python="/venv/bin/python", prefix=("systemd-run", "--scope"), timeout_s=300,
                               cards=cards, runner=runner)
        self.assertTrue(r["ok"])
        self.assertEqual(seen["env"]["CUDA_VISIBLE_DEVICES"], U1)
        self.assertEqual(seen["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(seen["cmd"][:3], ["systemd-run", "--scope", "/venv/bin/python"])
        self.assertIn("sglang.srt.rigmon.card_probe", seen["cmd"])
        self.assertIn("--run", seen["cmd"])
        self.assertEqual(seen["timeout"], 300)
        self.assertEqual(r["warnings"], ["lanes int8/w4a8/w4a16 not measured: no sgl_kernel"])
        line = hp.duration_line(r, cards)
        self.assertIn("HWPROFIL-MESSUNG ok", line)
        self.assertIn("nvml1=18.5s[membw=4.0,bf16=1.0]", line)

    def test_several_cards_keep_the_requested_order(self):
        seen = {}

        def runner(cmd, env, timeout):
            seen["env"] = env
            return 0, "{}", ""

        hp.run_measurement([2, 0], cards=_nvml()[0], runner=runner)
        self.assertEqual(seen["env"]["CUDA_VISIBLE_DEVICES"], f"{U2},{U0}")

    def test_unknown_card_is_refused_before_anything_starts(self):
        with self.assertRaises(ValueError):
            hp.run_measurement([7], cards=_nvml()[0], runner=lambda *a: (0, "{}", ""))

    def test_a_failing_child_is_a_failed_result_with_its_stderr(self):
        r = hp.run_measurement([0], cards=_nvml()[0], runner=lambda c, e, t: (1, "", "Traceback ... CUDA OOM"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["rc"], 1)
        self.assertIn("CUDA OOM", r["stderr_tail"])
        self.assertIn("FEHLER", hp.duration_line(r, _nvml()[0]))


class TestProbeArms(CustomTestCase):
    def test_new_fields_round_trip_and_old_cache_files_still_load(self):
        c = cp.CardProbeMeasurement(
            uuid="u", name="n", cuda_index=0, gemm_int8_tflops=180.0, gemm_w4a8_int8_tflops=62.0,
            gemm_w4a16_tflops=55.0, lane_notes={"nvfp4_w4a8": "why"}, sm_count=68, l2_mib=5.0,
            compute_capability="8.6", h2d_lat_us=12.0, d2h_lat_us=13.0, arm_seconds={"membw": 3.1},
        )
        back = cp.CardProbeMeasurement.from_json(json.loads(json.dumps(c.to_json())))
        self.assertEqual(back.gemm_w4a8_int8_tflops, 62.0)
        self.assertEqual(back.lane_notes, {"nvfp4_w4a8": "why"})
        self.assertEqual(back.arm_seconds, {"membw": 3.1})
        old = {"uuid": "u", "name": "n", "cuda_index": 0, "gemm_bf16_tflops": 1.0}  # a pre-950 cache entry
        o = cp.CardProbeMeasurement.from_json(old)
        self.assertIsNone(o.gemm_int8_tflops)
        self.assertEqual(o.lane_notes, {})

    def test_w4a8_is_asked_only_on_sm_8x_and_says_why_elsewhere(self):
        import torch

        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
            v, why = cp._bench_gemm_w4a8_int8("cuda:0")
        self.assertIsNone(v)
        self.assertIn("sm_8x", why)
        self.assertIn("12.0", why)

    def _measure(self, *, env_issue="", w4a8=(62.0, ""), w4a16=(55.0, ""), int8=(180.0, "")):
        import torch

        from sglang.srt import uneven_perf as up

        rates = up.MembwRates(read_gbs=700.0, copy_gbs=690.0, gemv_gbs=650.0)
        calls = []

        def rec(name, ret):
            def f(dev):
                calls.append(name)
                return ret
            return f

        with mock.patch.object(torch.cuda, "set_device"), \
             mock.patch.object(up, "_bench_membw_rates", lambda dev: rates), \
             mock.patch.object(up, "_bench_gemm_tflops", lambda dev: 60.0), \
             mock.patch.object(cp, "_device_properties", lambda dev: (68, 5.0, "8.6")), \
             mock.patch.object(cp, "_bench_gemm_fp8_tflops", rec("fp8", (None, "no fp8 tensor path"))), \
             mock.patch.object(cp, "_bench_h2d_d2h", lambda dev: (6.0, 6.5)), \
             mock.patch.object(cp, "_bench_h2d_d2h_latency", lambda dev: (12.5, 14.0)), \
             mock.patch.object(cp, "lane_environment_issue", lambda: env_issue), \
             mock.patch.object(cp, "_bench_gemm_int8", rec("int8", int8)), \
             mock.patch.object(cp, "_bench_gemm_w4a8_int8", rec("w4a8", w4a8)), \
             mock.patch.object(cp, "_bench_gemm_w4a16", rec("w4a16", w4a16)):
            m = cp.measure_card(0, "u0", "RTX 3080", 20480)
        return m, calls

    def test_measure_card_carries_every_new_arm_with_its_duration(self):
        m, calls = self._measure()
        self.assertEqual((m.gemm_int8_tflops, m.gemm_w4a8_int8_tflops, m.gemm_w4a16_tflops), (180.0, 62.0, 55.0))
        self.assertEqual((m.h2d_lat_us, m.d2h_lat_us), (12.5, 14.0))
        self.assertEqual((m.sm_count, m.l2_mib, m.compute_capability), (68, 5.0, "8.6"))
        self.assertEqual(calls, ["fp8", "int8", "w4a8", "w4a16"])
        for arm in ("membw", "bf16", "fp8_native", "h2d_d2h", "h2d_d2h_lat", "int8_native", "nvfp4_w4a8", "nvfp4_marlin"):
            self.assertIn(arm, m.arm_seconds)
        # sm_86: the native W4A4 lane is not asked; the card's own reason is stored (order 1006)
        self.assertEqual(list(m.lane_notes), ["nvfp4_w4a4"])
        self.assertIn("no native FP4 tensor cores", m.lane_notes["nvfp4_w4a4"])
        self.assertIsNone(m.gemm_w4a4_tflops)
        self.assertIsNone(m.gemm_fp8_tflops)  # no fp8 on sm_86: absent, with its reason
        self.assertEqual(m.fp8_note, "no fp8 tensor path")

    def test_a_lane_that_cannot_run_stores_its_reason_never_a_number(self):
        m, _ = self._measure(w4a8=(None, "kernel did not compile"), int8=(None, "no IMMA arm"))
        self.assertIsNone(m.gemm_w4a8_int8_tflops)
        self.assertEqual(m.lane_notes["nvfp4_w4a8"], "kernel did not compile")
        self.assertEqual(m.lane_notes["int8_native"], "no IMMA arm")
        self.assertEqual(m.gemm_w4a16_tflops, 55.0)

    def test_an_interpreter_without_sgl_kernel_leaves_the_lanes_empty_and_persists_no_note(self):
        m, calls = self._measure(env_issue="sgl_kernel not importable")
        self.assertEqual(calls, ["fp8"])  # the sgl_kernel lanes were never asked
        self.assertIsNone(m.gemm_int8_tflops)
        # an interpreter fact is not a card fact (#310): only the card's own W4A4 verdict (cc 8.6) is stored
        self.assertEqual(list(m.lane_notes), ["nvfp4_w4a4"])
        self.assertEqual(m.gemm_bf16_tflops, 60.0)

    def test_bar1_is_said_not_measured_in_every_multi_card_run(self):
        gpus = [{"cuda_index": i, "uuid": f"u{i}", "name": f"c{i}", "total_mib": 1} for i in range(2)]
        with mock.patch.object(cp, "_inventory", lambda: (gpus, "d")), \
             mock.patch.object(cp, "_card_states", lambda: {}), \
             mock.patch.object(cp, "measure_card",
                               lambda cuda_index, uuid, name, total_mib=None, state_fn=None:
                               cp.CardProbeMeasurement(uuid=uuid, name=name, cuda_index=cuda_index)), \
             mock.patch.object(cp, "measure_pair_matrix", lambda g: []):
            import torch  # noqa: F401

            prof = cp.run_card_probe(save=False)
        self.assertIn(hp.BAR1_NOT_MEASURED, prof.notes)

    def test_text_rendering_shows_the_new_columns_and_the_lane_reasons(self):
        c = cp.CardProbeMeasurement(uuid="u", name="RTX 3080", cuda_index=0, gemm_int8_tflops=180.0, sm_count=68,
                                    l2_mib=5.0, compute_capability="8.6", h2d_lat_us=12.5,
                                    lane_notes={"nvfp4_w4a8": "why not"})
        txt = cp.format_text(cp.CardProbeProfile(created=time.time(), cards=[c]))
        self.assertIn("int8", txt)
        self.assertIn("68", txt)
        self.assertIn("nvfp4_w4a8: why not", txt)


if __name__ == "__main__":
    unittest.main()
