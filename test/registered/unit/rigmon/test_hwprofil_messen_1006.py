"""CPU unit tests for order 1006: "Hardwareprofil messen" must deliver exactly the values the profile shows.

The arms for SM count / L2 / int8 / NVFP4 W4A8 / NVFP4 W4A16 / host latency were built by order 950 (and are tested in
``test_hardware_profile_950``); what 1006 adds, and what is pinned here:

* ``bar1_probe`` -- the BAR1 stretch per ORDERED pair: command and environment of the children (cards by UUID, rank order),
  the merge of the children's reports (a rate belongs to the card it names, a pair without a number carries its reason,
  nothing is invented), the real parent/child process plumbing with tiny stand-in children, the timeout that kills only
  its own children;
* ``card_probe`` -- the step is wired into ``--run`` (and only there), stored in ``bar1_pairs``, never mixed into the
  host-staging ``pairs``, a failed step keeps the card rates and records its reason; the host latency headline is the
  MEDIAN with the minimum beside it;
* ``hardware_profile`` -- the BAR1 column of the view: measured per direction with source/time/file, a missing pair is
  "nicht gemessen" with its own reason, a newer failure never hides an older measurement, a step that never ran is an
  open gap while one that ran and failed is final.

No GPU, no NVML, no torch kernels: the children are ``python -c`` stand-ins or injected runners.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_hardware_profile_950 import (  # noqa: E402
    NOW, U0, U1, U2, _nvml, _probe_card, _write_probe,
)

from sglang.srt.rigmon import bar1_probe as bp  # noqa: E402
from sglang.srt.rigmon import card_probe as cp  # noqa: E402
from sglang.srt.rigmon import hardware_profile as hp  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

UUIDS = [U0, U1, U2]


def _row(s, d, bw=5.5, lat=9.0, **kw):
    r = {"src": s, "dst": d, "nbytes": 16 << 20, "repeats": 3, "window_mib": 40.0, "bandwidth_gbs": bw,
         "latency_us": lat, "lat_min_us": lat - 1, "lat_n": 200}
    r.update(kw)
    return r


def _report(rank, uuid=None, rows=None, failed=None, window=40.0):
    rep = {"rank": rank, "uuid": uuid or UUIDS[rank]}
    if failed:
        rep["failed"] = failed
    else:
        rep["pairs"] = rows if rows is not None else [_row(rank, d) for d in range(3) if d != rank]
        rep["window_mib"] = window
    return rep


def _out(rep):
    return (0, "noise line\n" + bp.MARKER + json.dumps(rep) + "\n", "")


class TestBar1Commands(CustomTestCase):
    def test_rank_command_carries_the_uuids_in_rank_order_and_the_rank(self):
        c = bp.rank_command("/venv/py", 1, UUIDS, 300.0)
        self.assertEqual(c[:3], ["/venv/py", "-m", "sglang.srt.rigmon.bar1_probe"])
        self.assertEqual(c[c.index("--rank") + 1], "1")
        self.assertEqual(c[c.index("--uuids") + 1], ",".join(UUIDS))

    def test_env_shows_every_card_by_uuid_in_pci_order_and_a_local_rendezvous(self):
        e = bp.rank_env(UUIDS, 4711, base={"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}, extcache="/cache")
        self.assertEqual(e["CUDA_VISIBLE_DEVICES"], ",".join(UUIDS))   # never torch indices
        self.assertEqual(e["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual((e["MASTER_ADDR"], e["MASTER_PORT"]), ("127.0.0.1", "4711"))
        self.assertEqual(e["TORCH_EXTENSIONS_DIR"], "/cache")
        self.assertEqual(e["PATH"], "/bin")

    def test_an_extension_dir_the_caller_already_named_is_not_overridden(self):
        e = bp.rank_env(UUIDS, 1, base={"TORCH_EXTENSIONS_DIR": "/mine"}, extcache="/cache")
        self.assertEqual(e["TORCH_EXTENSIONS_DIR"], "/mine")

    def test_parse_takes_the_last_marker_line_and_ignores_noise(self):
        out = "x\n" + bp.MARKER + '{"rank": 0, "uuid": "a"}\nWARN\n' + bp.MARKER + '{"rank": 0, "uuid": "b"}\n'
        self.assertEqual(bp.parse_rank_output(out)["uuid"], "b")
        self.assertIsNone(bp.parse_rank_output("no marker"))
        self.assertIsNone(bp.parse_rank_output(bp.MARKER + "{broken"))


class TestBar1Merge(CustomTestCase):
    def test_all_six_directions_measured_each_with_its_own_numbers_and_label(self):
        # direction matters: the rate of s -> d is the SENDER's number; make every direction different
        results = [_out(_report(r, rows=[_row(r, d, bw=1.0 + 10 * r + d, lat=5.0 + r + d)
                                         for d in range(3) if d != r])) for r in range(3)]
        res = bp.merge_reports(UUIDS, results)
        self.assertEqual(res.reason, "")
        self.assertEqual(len(res.pairs), 6)
        by = {(p["src_uuid"], p["dst_uuid"]): p for p in res.pairs}
        for s in range(3):
            for d in range(3):
                if s != d:
                    p = by[(UUIDS[s], UUIDS[d])]
                    self.assertEqual(p["bandwidth_gbs"], 1.0 + 10 * s + d)
                    self.assertEqual(p["latency_us"], 5.0 + s + d)
                    self.assertEqual(p["transport"], bp.BAR1_DIRECT)
        self.assertEqual(res.window_mib, 40.0)
        self.assertIn("one-sided posted writes", res.pairs[0]["note"])

    def test_a_pair_whose_byte_proof_failed_is_a_pair_without_number_with_its_reason_and_the_rest_stays(self):
        rows1 = [_row(1, 0, bw=None, latency_us=None, reason="no byte-level proof for this direction"), _row(1, 2)]
        results = [_out(_report(0)), _out(_report(1, rows=rows1)), _out(_report(2))]
        res = bp.merge_reports(UUIDS, results)
        by = {(p["src_uuid"], p["dst_uuid"]): p for p in res.pairs}
        bad = by[(U1, U0)]
        self.assertIsNone(bad["bandwidth_gbs"])
        self.assertIsNone(bad["latency_us"])
        self.assertIn("byte-level proof", bad["note"])
        self.assertEqual(sum(1 for p in res.pairs if p["bandwidth_gbs"] is not None), 5)
        self.assertIn("byte-level proof", res.reason)

    def test_a_transport_that_did_not_come_up_leaves_every_pair_of_that_rank_without_number(self):
        failed = {"stage": "setup", "reason": "dmabuf_holder not available"}
        results = [_out(_report(0)), _out(_report(1, failed=failed)), _out(_report(2))]
        res = bp.merge_reports(UUIDS, results)
        from1 = [p for p in res.pairs if p["src_uuid"] == U1]
        self.assertEqual(len(from1), 2)
        for p in from1:
            self.assertIsNone(p["bandwidth_gbs"])
            self.assertIn("dmabuf_holder", p["note"])
        self.assertTrue(all(p["bandwidth_gbs"] is not None for p in res.pairs if p["src_uuid"] != U1))

    def test_numbers_of_a_child_that_sat_on_another_card_are_discarded(self):
        # MUTANT guard: rank 1 reports card U2: its rates must not be filed under U1
        results = [_out(_report(0)), _out(_report(1, uuid=U2)), _out(_report(2))]
        res = bp.merge_reports(UUIDS, results)
        from1 = [p for p in res.pairs if p["src_uuid"] == U1]
        for p in from1:
            self.assertIsNone(p["bandwidth_gbs"])
            self.assertIn("reported card", p["note"])

    def test_a_dead_or_silent_child_is_a_reason_with_its_exit_code(self):
        results = [_out(_report(0)), (-9, "", "Killed"), _out(_report(2))]
        res = bp.merge_reports(UUIDS, results)
        p = next(p for p in res.pairs if p["src_uuid"] == U1)
        self.assertIsNone(p["bandwidth_gbs"])
        self.assertIn("rc=-9", p["note"])

    def test_a_non_positive_rate_is_not_a_measurement(self):
        rows = [_row(0, 1, bw=0.0), _row(0, 2, bw=-1.0)]
        res = bp.merge_reports(UUIDS, [_out(_report(0, rows=rows)), _out(_report(1)), _out(_report(2))])
        self.assertTrue(all(p["bandwidth_gbs"] is None for p in res.pairs if p["src_uuid"] == U0))

    def test_no_results_at_all_is_every_pair_without_number(self):
        res = bp.merge_reports(UUIDS, [])
        self.assertEqual(len(res.pairs), 6)
        self.assertTrue(all(p["bandwidth_gbs"] is None and p["note"] for p in res.pairs))
        self.assertTrue(res.reason)


class TestBar1Run(CustomTestCase):
    def test_runner_gets_one_command_per_card_and_the_merged_result_comes_back(self):
        seen = {}

        def runner(cmds, env, timeout):
            seen.update(cmds=cmds, env=env, timeout=timeout)
            return [_out(_report(r)) for r in range(3)]

        res = bp.run_bar1_probe([{"uuid": u} for u in UUIDS], python="/venv/py", timeout_s=123.0, runner=runner,
                                port=5555, env={"PATH": "/bin"}, extcache="/c")
        self.assertEqual(len(seen["cmds"]), 3)
        self.assertEqual([c[c.index("--rank") + 1] for c in seen["cmds"]], ["0", "1", "2"])
        self.assertEqual(seen["env"]["MASTER_PORT"], "5555")
        self.assertEqual(seen["timeout"], 123.0)
        self.assertEqual(res.reason, "")
        self.assertIsNotNone(res.seconds)

    def test_a_runner_that_raises_is_a_reason_not_an_exception(self):
        def runner(*a):
            raise OSError("no fork")

        res = bp.run_bar1_probe([{"uuid": u} for u in UUIDS], runner=runner)
        self.assertEqual(len(res.pairs), 6)
        self.assertIn("could not be started", res.reason)
        self.assertTrue(all(p["bandwidth_gbs"] is None for p in res.pairs))

    def test_fewer_than_two_cards_has_no_pairs(self):
        res = bp.run_bar1_probe([{"uuid": U0}], runner=lambda *a: [])
        self.assertEqual(res.pairs, [])
        self.assertIn("fewer than two", res.reason)

    def test_real_processes_report_through_stdout_and_the_parent_merges(self):
        # stand-in children: a python -c that prints exactly the report of its rank (no torch, no GPU)
        def cmd(rank):
            rep = _report(rank)
            code = "import sys;print(%r + %r)" % (bp.MARKER, json.dumps(rep))
            return [sys.executable, "-c", code]

        out = bp._spawn_all([cmd(r) for r in range(3)], dict(os.environ), 60.0)
        self.assertEqual([o[0] for o in out], [0, 0, 0])
        res = bp.merge_reports(UUIDS, out)
        self.assertEqual(res.reason, "")
        self.assertEqual(len(res.pairs), 6)

    def test_the_timeout_kills_only_our_children_and_reports_rc_124(self):
        t0 = time.time()
        out = bp._spawn_all([[sys.executable, "-c", "import time;time.sleep(60)"]], dict(os.environ), 2.0)
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(out[0][0], 124)
        self.assertIn("Zeitüberschreitung", out[0][2])


class TestCardProbeBar1Wiring(CustomTestCase):
    def _run(self, **kw):
        gpus = [{"cuda_index": i, "uuid": u, "name": f"c{i}", "total_mib": 1} for i, u in enumerate(UUIDS)]
        with mock.patch.object(cp, "_inventory", lambda: (gpus, "d")), \
             mock.patch.object(cp, "_card_states", lambda: {}), \
             mock.patch.object(cp, "measure_card",
                               lambda cuda_index, uuid, name, total_mib=None, state_fn=None:
                               cp.CardProbeMeasurement(uuid=uuid, name=name, cuda_index=cuda_index, gemm_bf16_tflops=60.0)), \
             mock.patch.object(cp, "measure_pair_matrix", lambda g: []):
            return cp.run_card_probe(save=False, **kw)

    def test_the_step_is_off_for_library_callers_and_then_says_so(self):
        with mock.patch.object(bp, "run_bar1_probe", side_effect=AssertionError("must not run")):
            prof = self._run()
        self.assertFalse(prof.bar1_attempted)
        self.assertIn(hp.BAR1_NOT_MEASURED, prof.notes)

    def test_the_step_result_is_stored_in_its_own_list_never_in_the_host_staging_pairs(self):
        fake = bp.Bar1Result(pairs=[
            {"src_uuid": U0, "dst_uuid": U1, "bandwidth_gbs": 5.5, "latency_us": 9.0, "transport": bp.BAR1_DIRECT,
             "peer_access": True, "bytes_moved": 1 << 24, "note": "n"},
            {"src_uuid": U1, "dst_uuid": U0, "bandwidth_gbs": None, "latency_us": None, "transport": bp.BAR1_DIRECT,
             "peer_access": False, "bytes_moved": 0, "note": "no proof"}], reason="no proof", seconds=41.0, window_mib=40.0)
        with mock.patch.object(bp, "run_bar1_probe", return_value=fake) as m:
            prof = self._run(bar1=True)
        self.assertEqual([[g["uuid"] for g in c.args[0]] for c in m.call_args_list], [UUIDS])  # by UUID, probe order
        self.assertTrue(prof.bar1_attempted)
        self.assertEqual(len(prof.bar1_pairs), 2)
        self.assertEqual(prof.pairs, [])
        self.assertEqual(prof.bar1_reason, "no proof")
        self.assertEqual((prof.bar1_seconds, prof.bar1_window_mib), (41.0, 40.0))
        self.assertNotIn(hp.BAR1_NOT_MEASURED, prof.notes)
        self.assertTrue(any("BAR1 stretch: no proof" in n for n in prof.notes))
        back = cp.CardProbeProfile.from_json(json.loads(json.dumps(prof.to_json())))
        self.assertEqual(back.bar1_pairs[0].bandwidth_gbs, 5.5)
        self.assertTrue(back.bar1_attempted)
        self.assertEqual(back.bar1_reason, "no proof")
        self.assertIn("BAR1 stretch per ordered pair", cp.format_text(prof))

    def test_a_failing_step_keeps_the_card_rates_and_records_why(self):
        with mock.patch.object(bp, "run_bar1_probe", side_effect=RuntimeError("holder missing")):
            prof = self._run(bar1=True)
        self.assertTrue(prof.bar1_attempted)
        self.assertEqual([c.gemm_bf16_tflops for c in prof.cards], [60.0] * 3)
        self.assertIn("holder missing", prof.bar1_reason)
        self.assertTrue(all(p.bandwidth_gbs is None for p in prof.bar1_pairs))

    def test_one_card_has_no_bar1_step(self):
        gpus = [{"cuda_index": 0, "uuid": U0, "name": "c0", "total_mib": 1}]
        with mock.patch.object(cp, "_inventory", lambda: (gpus, "d")), mock.patch.object(cp, "_card_states", lambda: {}), \
             mock.patch.object(cp, "measure_card", lambda cuda_index, uuid, name, total_mib=None, state_fn=None:
                               cp.CardProbeMeasurement(uuid=uuid, name=name, cuda_index=cuda_index)), \
             mock.patch.object(bp, "run_bar1_probe", side_effect=AssertionError("must not run")):
            prof = cp.run_card_probe(save=False, bar1=True)
        self.assertFalse(prof.bar1_attempted)

    def test_the_command_line_run_turns_the_step_on_and_no_bar1_turns_it_off(self):
        seen = []

        def fake_run(**kw):
            seen.append(kw)
            return cp.CardProbeProfile(created=time.time())

        with mock.patch.object(cp, "run_card_probe", side_effect=fake_run), \
             mock.patch.object(cp, "lane_environment_issue", lambda: ""):
            cp._main(["--run", "--json"])
            cp._main(["--run", "--json", "--no-bar1"])
            cp._main(["--run", "--json", "--bar1-timeout-s", "77"])
        self.assertEqual([k["bar1"] for k in seen], [True, False, True])
        self.assertEqual(seen[2]["bar1_timeout_s"], 77.0)

    def test_old_probe_files_without_the_bar1_fields_still_load(self):
        old = {"version": 1, "created": 1.0, "cards": [{"uuid": "u", "name": "n", "cuda_index": 0}], "pairs": []}
        p = cp.CardProbeProfile.from_json(old)
        self.assertEqual((p.bar1_pairs, p.bar1_attempted, p.bar1_reason), ([], False, ""))


class TestHostLatencyIsTheMedian(CustomTestCase):
    def test_median_is_the_typical_value_not_the_minimum(self):
        self.assertEqual(cp._median([5.0, 100.0, 7.0]), 7.0)
        self.assertEqual(cp._median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_the_latency_function_returns_median_first_and_minimum_beside_it_with_scripted_clock(self):
        import torch

        class Buf:
            def copy_(self, other, non_blocking=False):
                return self

        n = cp._LAT_ITERS
        # per direction: n samples of 10 us, except one 1 us (the minimum) and a few 500 us outliers
        durs = [10e-6] * n
        durs[3], durs[7], durs[9], durs[11] = 1e-6, 500e-6, 500e-6, 500e-6
        clock = []
        t = 0.0
        for _ in range(2):          # H2D then D2H, same script
            for d in durs:
                clock += [t, t + d]
                t += 1.0
        it = iter(clock)
        with mock.patch.object(torch, "empty", lambda *a, **k: Buf()), \
             mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
             mock.patch.object(torch.cuda, "empty_cache", lambda: None), \
             mock.patch.object(cp.time, "perf_counter", lambda: next(it)):
            h2d_p50, d2h_p50, h2d_min, d2h_min = cp._bench_h2d_d2h_latency("cuda:0")
        self.assertEqual((h2d_p50, d2h_p50), (10.0, 10.0))
        self.assertEqual((h2d_min, d2h_min), (1.0, 1.0))

    def test_measure_card_stores_median_headline_and_minimum_beside_it(self):
        import torch

        from sglang.srt import uneven_perf as up

        rates = up.MembwRates(read_gbs=1.0, copy_gbs=1.0, gemv_gbs=1.0)
        with mock.patch.object(torch.cuda, "set_device"), \
             mock.patch.object(up, "_bench_membw_rates", lambda dev: rates), \
             mock.patch.object(up, "_bench_gemm_tflops", lambda dev: 60.0), \
             mock.patch.object(cp, "_device_properties", lambda dev: (68, 5.0, "8.6")), \
             mock.patch.object(cp, "_bench_gemm_fp8_tflops", lambda dev: (None, "no fp8")), \
             mock.patch.object(cp, "_bench_h2d_d2h", lambda dev: (6.0, 6.5)), \
             mock.patch.object(cp, "_bench_h2d_d2h_latency", lambda dev: (12.5, 14.0, 9.5, 10.0)), \
             mock.patch.object(cp, "lane_environment_issue", lambda: "x"):
            m = cp.measure_card(0, "u0", "RTX", 1)
        self.assertEqual((m.h2d_lat_us, m.d2h_lat_us, m.h2d_lat_min_us, m.d2h_lat_min_us), (12.5, 14.0, 9.5, 10.0))

    def test_the_view_shows_the_median_and_names_the_minimum_and_the_method(self):
        with tempfile.TemporaryDirectory() as d:
            _write_probe(d, "card_probe-a.json", NOW - 5,
                         [_probe_card(U0, "RTX 3080", h2d_lat_us=12.5, d2h_lat_us=14.0,
                                      h2d_lat_min_us=9.5, d2h_lat_min_us=10.0)])
            doc = hp.build(cache_dir=d, nvml=_nvml(), now=NOW)
        c0 = next(c for c in doc["cards"] if c["uuid"] == U0)
        n = c0["h2d"]["lat_us"]
        self.assertEqual((n["v"], n["src"]), (12.5, hp.SRC_MEASURED))
        self.assertIn("Median", n["note"])
        self.assertIn("9.5", n["note"])
        self.assertEqual(hp.validate(doc), [])


def _bar1_pair(s, d, bw, lat=9.0, note="n", transport=bp.BAR1_DIRECT):
    return {"src_uuid": s, "dst_uuid": d, "bandwidth_gbs": bw, "latency_us": lat if bw is not None else None,
            "transport": transport, "peer_access": bw is not None, "note": note}


ALL6 = [(a, b) for a in UUIDS for b in UUIDS if a != b]


def _write_bar1_probe(d, name, created, pairs, attempted=True, reason="", cards=None):
    no_fp4 = {"nvfp4_w4a4": "compute capability 8.6: no native FP4 tensor cores (needs 10.0+)"}
    cards = cards or [_probe_card(U0, "RTX 3080", lane_notes=no_fp4), _probe_card(U2, "RTX 3080", lane_notes=no_fp4),
                      _probe_card(U1, "RTX 5090", gemm_fp8_tflops=500.0, fp8_note="", gemm_w4a8_int8_tflops=None,
                                  gemm_w4a4_tflops=900.0,
                                  lane_notes={"nvfp4_w4a8": "compute capability 12.0: the W4A8 kernel is the sm_8x one"},
                                  compute_capability="12.0")]
    with open(os.path.join(d, name), "w") as f:
        json.dump({"version": 1, "created": created, "driver": "595.58", "torch_version": "2.9", "cuda_version": "13.0",
                   "cards": cards, "pairs": [], "bar1_pairs": pairs, "bar1_attempted": attempted, "bar1_reason": reason}, f)


class TestProfileBar1Column(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _build(self):
        return hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)

    def _bar1(self, doc):
        return {(l["src"], l["dst"]): l for l in doc["links"] if l["transport"] == "bar1"}

    def test_a_fully_measured_matrix_is_measured_per_direction_with_source_time_and_file(self):
        pairs = [_bar1_pair(a, b, 1.0 + i) for i, (a, b) in enumerate(ALL6)]
        _write_bar1_probe(self.d.name, "card_probe-b.json", NOW - 30, pairs)
        doc = self._build()
        ord_of = {c["uuid"]: c["ord"] for c in doc["cards"]}
        bar1 = self._bar1(doc)
        self.assertEqual(len(bar1), 6)
        for i, (a, b) in enumerate(ALL6):
            l = bar1[(ord_of[a], ord_of[b])]       # by UUID-derived ordinal, not by file/NVML position
            self.assertEqual((l["gbs"]["v"], l["gbs"]["src"], l["gbs"]["at"], l["gbs"]["probe"]),
                             (1.0 + i, "gemessen", NOW - 30, "card_probe-b.json"))
            self.assertEqual(l["lat_us"]["v"], 9.0)
        self.assertTrue(doc["bar1"]["measured"] and doc["bar1"]["complete"])
        self.assertEqual((doc["bar1"]["pairs_measured"], doc["bar1"]["pairs_total"]), (6, 6))
        self.assertNotIn("bar1", doc["unmeasured"])
        self.assertFalse(doc["measure_needed"], doc["unmeasured"])
        self.assertEqual(hp.validate(doc), [])

    def test_the_host_staging_matrix_and_the_bar1_matrix_do_not_overwrite_each_other(self):
        _write_bar1_probe(self.d.name, "card_probe-b.json", NOW - 30, [_bar1_pair(a, b, 4.0) for a, b in ALL6])
        # same ordered pairs also present as host staging in the same file
        path = os.path.join(self.d.name, "card_probe-b.json")
        d = json.load(open(path))
        d["pairs"] = [{"src_uuid": a, "dst_uuid": b, "bandwidth_gbs": 2.0, "latency_us": 30.0,
                       "transport": cp.HOST_STAGING, "peer_access": False} for a, b in ALL6]
        json.dump(d, open(path, "w"))
        doc = self._build()
        self.assertEqual(len([l for l in doc["links"] if l["transport"] == "host_staging"]), 6)
        self.assertEqual({l["gbs"]["v"] for l in doc["links"] if l["transport"] == "bar1"}, {4.0})
        self.assertEqual({l["gbs"]["v"] for l in doc["links"] if l["transport"] == "host_staging"}, {2.0})

    def test_a_pair_without_number_is_nicht_gemessen_with_its_own_reason_the_rest_stays_measured(self):
        pairs = [_bar1_pair(a, b, 3.0) for a, b in ALL6]
        pairs[2] = _bar1_pair(*ALL6[2], None, note="BAR1 1->0 not measured: no byte-level proof for this direction")
        _write_bar1_probe(self.d.name, "card_probe-b.json", NOW - 30, pairs, reason="no byte-level proof")
        doc = self._build()
        ord_of = {c["uuid"]: c["ord"] for c in doc["cards"]}
        bad = self._bar1(doc)[(ord_of[ALL6[2][0]], ord_of[ALL6[2][1]])]
        self.assertIsNone(bad["gbs"]["v"])
        self.assertEqual(bad["gbs"]["src"], hp.SRC_NONE)
        self.assertIn("no byte-level proof", bad["gbs"]["note"])
        self.assertEqual(doc["bar1"]["pairs_measured"], 5)
        self.assertFalse(doc["bar1"]["complete"])
        self.assertTrue(doc["bar1"]["measured"])
        self.assertIn("5 von 6", doc["bar1"]["note"])
        self.assertEqual(hp.validate(doc), [])
        # the step RAN and said why: final, "Hardwareprofil messen" does not stay lit for it
        self.assertNotIn("bar1", doc["unmeasured"])

    def test_a_step_that_ran_and_failed_everywhere_is_final_with_the_reason_on_every_pair(self):
        pairs = [_bar1_pair(a, b, None, note="BAR1 not measured: dmabuf_holder not available") for a, b in ALL6]
        _write_bar1_probe(self.d.name, "card_probe-b.json", NOW - 30, pairs, reason="dmabuf_holder not available")
        doc = self._build()
        for l in self._bar1(doc).values():
            self.assertIsNone(l["gbs"]["v"])
            self.assertIn("dmabuf_holder", l["gbs"]["note"])
        self.assertFalse(doc["bar1"]["measured"])
        self.assertIn("dmabuf_holder", doc["bar1"]["note"])
        self.assertNotIn("bar1", doc["unmeasured"])
        self.assertFalse(doc["measure_needed"])

    def test_a_probe_without_the_step_leaves_bar1_an_open_gap_and_the_button_lit(self):
        _write_bar1_probe(self.d.name, "card_probe-b.json", NOW - 30, [], attempted=False)
        doc = self._build()
        self.assertEqual(len(doc["unmeasured"]["bar1"]), 6)
        self.assertTrue(doc["measure_needed"])
        for l in self._bar1(doc).values():
            self.assertIsNone(l["gbs"]["v"])
            self.assertIn("nicht gemessen", l["gbs"]["note"])
        self.assertIn("NOT MEASURED", doc["bar1"]["note"])
        self.assertFalse(doc["bar1"]["measured"])

    def test_a_newer_failed_attempt_never_hides_an_older_measurement(self):
        _write_bar1_probe(self.d.name, "card_probe-old.json", NOW - 7200, [_bar1_pair(a, b, 6.0) for a, b in ALL6])
        _write_bar1_probe(self.d.name, "card_probe-new.json", NOW - 60,
                          [_bar1_pair(a, b, None, note="BAR1 not measured: holder busy") for a, b in ALL6],
                          reason="holder busy")
        doc = self._build()
        for l in self._bar1(doc).values():
            self.assertEqual((l["gbs"]["v"], l["gbs"]["probe"]), (6.0, "card_probe-old.json"))

    def test_the_newest_measurement_wins_per_ordered_pair(self):
        _write_bar1_probe(self.d.name, "card_probe-old.json", NOW - 7200, [_bar1_pair(a, b, 6.0) for a, b in ALL6])
        _write_bar1_probe(self.d.name, "card_probe-new.json", NOW - 60, [_bar1_pair(a, b, 7.5) for a, b in ALL6])
        doc = self._build()
        self.assertEqual({l["gbs"]["v"] for l in self._bar1(doc).values()}, {7.5})

    def test_a_pair_of_a_card_that_is_not_in_the_inventory_is_ignored(self):
        _write_bar1_probe(self.d.name, "card_probe-b.json", NOW - 30,
                          [_bar1_pair(a, b, 3.0) for a, b in ALL6] + [_bar1_pair(U0, "GPU-9999", 99.0)])
        doc = self._build()
        self.assertEqual(len(self._bar1(doc)), 6)
        self.assertNotIn(99.0, [l["gbs"]["v"] for l in doc["links"]])

    def test_the_duration_line_names_the_bar1_step(self):
        res = {"ok": True, "rc": 0, "seconds": 200.0,
               "profile": {"cards": [], "pairs": [], "bar1_attempted": True, "bar1_seconds": 48.2,
                           "bar1_pairs": [{"bandwidth_gbs": 5.0}] * 5 + [{"bandwidth_gbs": None}]}}
        self.assertIn("bar1=48.2s[5/6]", hp.duration_line(res, _nvml()[0]))


class TestNativeW4A4(CustomTestCase):
    """Nachtrag 06.10. ~11:50Z: native NVFP4 W4A4 only where the card has FP4 tensor cores (sm_12x); everywhere else
    "nicht gemessen" with the card's reason, never a number."""

    WHY86 = "compute capability 8.6: no native FP4 tensor cores (needs 10.0+)"

    def _measure(self, cc, *, env_issue="", w4a4=(900.0, "")):
        import torch

        from sglang.srt import uneven_perf as up

        rates = up.MembwRates(read_gbs=1.0, copy_gbs=1.0, gemv_gbs=1.0)
        calls = []

        def w4a4_fn(dev):
            calls.append("w4a4")
            return w4a4

        with mock.patch.object(torch.cuda, "set_device"), \
             mock.patch.object(up, "_bench_membw_rates", lambda dev: rates), \
             mock.patch.object(up, "_bench_gemm_tflops", lambda dev: 60.0), \
             mock.patch.object(cp, "_device_properties", lambda dev: (68, 5.0, cc)), \
             mock.patch.object(cp, "_bench_gemm_fp8_tflops", lambda dev: (None, "x")), \
             mock.patch.object(cp, "_bench_h2d_d2h", lambda dev: (6.0, 6.5)), \
             mock.patch.object(cp, "_bench_h2d_d2h_latency", lambda dev: (12.5, 14.0, 9.0, 10.0)), \
             mock.patch.object(cp, "lane_environment_issue", lambda: env_issue), \
             mock.patch.object(cp, "_bench_gemm_int8", lambda dev: (180.0, "")), \
             mock.patch.object(cp, "_bench_gemm_w4a8_int8", lambda dev: (62.0, "")), \
             mock.patch.object(cp, "_bench_gemm_w4a16", lambda dev: (55.0, "")), \
             mock.patch.object(cp, "_bench_gemm_w4a4_native", w4a4_fn):
            m = cp.measure_card(0, "u0", "card", 1)
        return m, calls

    def test_the_gate_names_the_card_fact(self):
        self.assertEqual(cp._w4a4_unsupported_reason("12.0"), "")
        self.assertEqual(cp._w4a4_unsupported_reason("12.1"), "")
        self.assertIn("no native FP4 tensor cores", cp._w4a4_unsupported_reason("8.6"))
        self.assertIn("no native FP4 tensor cores", cp._w4a4_unsupported_reason("8.9"))
        self.assertIn("not asked", cp._w4a4_unsupported_reason("10.0"))   # datacenter Blackwell: another kernel
        self.assertIn("unknown", cp._w4a4_unsupported_reason(None))

    def test_sm120_is_measured_through_the_native_path_and_timed_as_an_arm(self):
        m, calls = self._measure("12.0")
        self.assertEqual(calls, ["w4a4"])
        self.assertEqual(m.gemm_w4a4_tflops, 900.0)
        self.assertNotIn("nvfp4_w4a4", m.lane_notes)
        self.assertIn("nvfp4_w4a4", m.arm_seconds)

    def test_sm86_never_asks_the_native_kernel_and_stores_the_reason_MUTANT_marked_measured(self):
        # MUTANT guard: a 3080 must not get a W4A4 number even though the stub would happily return one
        m, calls = self._measure("8.6")
        self.assertEqual(calls, [])
        self.assertIsNone(m.gemm_w4a4_tflops)
        self.assertIn("no native FP4 tensor cores", m.lane_notes["nvfp4_w4a4"])
        self.assertNotIn("nvfp4_w4a4", m.arm_seconds)

    def test_the_card_reason_is_stored_even_when_the_interpreter_cannot_run_the_lanes(self):
        m, calls = self._measure("8.6", env_issue="sgl_kernel not importable")
        self.assertEqual(calls, [])
        self.assertIn("nvfp4_w4a4", m.lane_notes)       # a card fact, not an interpreter fact
        m12, calls12 = self._measure("12.0", env_issue="sgl_kernel not importable")
        self.assertEqual(calls12, [])                     # an interpreter fact: not asked, no note persisted
        self.assertNotIn("nvfp4_w4a4", m12.lane_notes)
        self.assertIsNone(m12.gemm_w4a4_tflops)

    def test_a_native_lane_that_fails_on_sm120_stores_its_reason_never_a_number(self):
        m, _ = self._measure("12.0", w4a4=(None, "NVFP4 native GEMM did not run: RuntimeError: x"))
        self.assertIsNone(m.gemm_w4a4_tflops)
        self.assertIn("did not run", m.lane_notes["nvfp4_w4a4"])

    def test_the_bench_says_why_when_the_fork_kernel_is_not_there_and_leaves_the_backend_global_alone(self):
        from sglang.srt.layers.quantization import fp4_utils

        before = fp4_utils.FP4_GEMM_RUNNER_BACKEND
        with mock.patch.object(fp4_utils, "has_fork_nvfp4_cutlass_kernel", lambda: False):
            v, why = cp._bench_gemm_w4a4_native("cuda:0")
        self.assertIsNone(v)
        self.assertIn("not available", why)
        self.assertIs(fp4_utils.FP4_GEMM_RUNNER_BACKEND, before)

    def test_the_row_is_in_the_view_and_the_formats_list(self):
        self.assertIn(("nvfp4_w4a4", "TFLOPS", "NVFP4 W4A4 (nativ)"), hp.COMPUTE_FORMATS)
        self.assertIn("nvfp4_w4a4", hp.PROBE_FORMATS)

    def test_only_the_sm120_card_gets_the_row_filled_the_3080s_keep_their_reason(self):
        with tempfile.TemporaryDirectory() as d:
            _write_probe(d, "card_probe-a.json", NOW - 5, [
                _probe_card(U0, "RTX 3080", lane_notes={"nvfp4_w4a4": self.WHY86}),
                _probe_card(U2, "RTX 3080", lane_notes={"nvfp4_w4a4": self.WHY86}),
                _probe_card(U1, "RTX 5090", gemm_w4a4_tflops=910.5, compute_capability="12.0",
                            lane_notes={"nvfp4_w4a8": "compute capability 12.0: the W4A8 kernel is the sm_8x one"})])
            doc = hp.build(cache_dir=d, nvml=_nvml(), now=NOW)
        by = {c["uuid"]: c for c in doc["cards"]}
        n5090 = by[U1]["compute"]["nvfp4_w4a4"]
        self.assertEqual((n5090["v"], n5090["src"], n5090["unit"], n5090["probe"]),
                         (910.5, "gemessen", "TFLOPS", "card_probe-a.json"))
        for u in (U0, U2):
            n = by[u]["compute"]["nvfp4_w4a4"]
            self.assertIsNone(n["v"])
            self.assertEqual(n["src"], hp.SRC_NONE)
            self.assertIn("no native FP4 tensor cores", n["note"])
        self.assertEqual(hp.validate(doc), [])
        self.assertIn("nvfp4_w4a4", [f["key"] for f in doc["formats"]])

    def test_a_corrupt_probe_that_puts_a_w4a4_number_on_a_3080_is_refused_by_the_view(self):
        # MUTANT guard on the data side: "W4A4 wird auf der 3080 als gemessen markiert"
        with tempfile.TemporaryDirectory() as d:
            _write_probe(d, "card_probe-bad.json", NOW - 5, [_probe_card(U0, "RTX 3080", gemm_w4a4_tflops=555.0)])
            doc = hp.build(cache_dir=d, nvml=_nvml(), now=NOW)
        n = next(c for c in doc["cards"] if c["uuid"] == U0)["compute"]["nvfp4_w4a4"]
        self.assertIsNone(n["v"])
        self.assertEqual(n["src"], hp.SRC_NONE)
        self.assertIn("verworfen", n["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_a_probe_from_before_the_row_leaves_it_an_open_gap(self):
        with tempfile.TemporaryDirectory() as d:
            _write_probe(d, "card_probe-old.json", NOW - 5, [_probe_card(U0, "RTX 3080")])
            doc = hp.build(cache_dir=d, nvml=_nvml(1), now=NOW)
        self.assertIn("nvfp4_w4a4", doc["unmeasured"]["0"])
        self.assertTrue(doc["measure_needed"])

    def test_text_table_and_round_trip_carry_the_new_field(self):
        c = cp.CardProbeMeasurement(uuid="u", name="RTX 5090", cuda_index=0, gemm_w4a4_tflops=910.5, sm_count=170,
                                    compute_capability="12.0")
        self.assertEqual(cp.CardProbeMeasurement.from_json(json.loads(json.dumps(c.to_json()))).gemm_w4a4_tflops, 910.5)
        txt = cp.format_text(cp.CardProbeProfile(created=time.time(), cards=[c]))
        self.assertIn("w4a4", txt)
        self.assertIn("910.5", txt)


if __name__ == "__main__":
    unittest.main()
