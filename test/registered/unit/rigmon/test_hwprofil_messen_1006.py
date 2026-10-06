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
from sglang.srt.rigmon import nccl_probe as npb  # noqa: E402
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

    def test_the_second_latency_travels_with_its_kind_and_is_absent_where_the_pair_has_no_number(self):
        rows0 = [_row(0, 1, latency_device_us=1.3), _row(0, 2, bw=None, latency_us=None, reason="no proof")]
        res = bp.merge_reports(UUIDS, [_out(_report(0, rows=rows0)), _out(_report(1)), _out(_report(2))])
        by = {(p["src_uuid"], p["dst_uuid"]): p for p in res.pairs}
        self.assertEqual(by[(U0, U1)]["latency_device_us"], 1.3)
        self.assertIn("KEIN Rundlauf", by[(U0, U1)]["latency_device_kind"])
        self.assertIsNone(by[(U0, U2)]["latency_device_us"])
        self.assertEqual(by[(U0, U2)]["latency_device_kind"], "")
        self.assertIsNone(by[(U1, U0)]["latency_device_us"])           # that sender did not report one: absent, not zero

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
                   "cards": cards, "pairs": [], "bar1_pairs": pairs, "bar1_attempted": attempted, "bar1_reason": reason,
                   # the NCCL step ran and said why it has no numbers: final, so these tests keep their meaning
                   "nccl_attempted": True, "nccl_reason": "NCCL did not come up", "nccl_pairs": []}, f)


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
        # these staging pairs are in the pre-1006 format (no serial field): their number is the SERIAL one, the pipelined rate is open
        self.assertEqual({l["gbs_serial"]["v"] for l in doc["links"] if l["transport"] == "host_staging"}, {2.0})
        self.assertEqual({l["gbs"]["v"] for l in doc["links"] if l["transport"] == "host_staging"}, {None})

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


# ---------------------------------------------------------------------------------------------------------------------
# Auftrag 1006, Nachtrag 14:30-14:40Z: pipelined host staging, PCIe link, NCCL way, D2D table with three ways
# ---------------------------------------------------------------------------------------------------------------------


class _FT:
    """A tensor stand-in that only records WHAT is copied and in which order (no torch kernels, no GPU)."""

    def __init__(self, name, log, off=0, n=0):
        self.name, self.log, self.off, self.n = name, log, off, n

    def __getitem__(self, sl):
        return _FT(self.name, self.log, sl.start, sl.stop - sl.start)

    def copy_(self, src, non_blocking=False):
        ch = cp._PIPE_CHUNK
        if self.name.startswith("host"):                      # device -> host buffer: a D2H of chunk off/ch
            self.log.append(("D2H", src.off // ch, self.name))
        else:                                                 # host buffer -> destination slice: an H2D of chunk off/ch
            self.log.append(("H2D", self.off // ch, src.name))
        return self


class TestPipelinedStaging(CustomTestCase):
    def _run(self, per_run_s=0.01, nbytes=cp._XFER_BYTES):
        import itertools

        import torch

        log = []
        hosts = itertools.count()
        ev_n = itertools.count()
        n = nbytes // cp._PIPE_CHUNK

        class Ev:
            def __init__(self):
                i = next(ev_n) % (2 * n)
                self.kind, self.k = ("d", i) if i < n else ("h", i - n)

            def record(self, stream):
                log.append(("rec", self.kind, self.k))

            def synchronize(self):
                log.append(("sync", self.kind, self.k))

        calls = itertools.count()

        def clock():
            c = next(calls)
            return (c // 2) * 1.0 + (per_run_s if c % 2 else 0.0)      # every once(): t0, t0 + per_run_s

        a, b = _FT("a", log), _FT("b", log)
        with mock.patch.object(torch, "empty", lambda *x, **k: _FT("host%d" % next(hosts), log)), \
             mock.patch.object(torch.cuda, "Stream", lambda device=None: object()), \
             mock.patch.object(torch.cuda, "Event", Ev), \
             mock.patch.object(torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext()), \
             mock.patch.object(torch.cuda, "synchronize", lambda *x, **k: None), \
             mock.patch.object(cp.time, "perf_counter", clock):
            gbs = cp._staged_pipelined_gbs("cuda:0", "cuda:1", a, b, nbytes)
        return gbs, log

    def test_rate_is_bytes_over_the_median_of_whole_copies(self):
        gbs, _ = self._run(per_run_s=0.01)
        self.assertAlmostEqual(gbs, cp._XFER_BYTES / 1e9 / 0.01, places=6)

    def test_every_chunk_goes_down_and_up_exactly_once_in_order(self):
        _, log = self._run()
        n = cp._XFER_BYTES // cp._PIPE_CHUNK
        one_run = [x for x in log if x[0] in ("D2H", "H2D")]
        per = 2 * n
        first = one_run[:per]
        self.assertEqual([x[1] for x in first if x[0] == "D2H"], list(range(n)))
        self.assertEqual([x[1] for x in first if x[0] == "H2D"], list(range(n)))

    def test_d2h_of_the_next_chunk_is_issued_before_the_h2d_of_the_previous_one_that_is_the_overlap(self):
        # MUTANT guard "serial instead of pipeline": a serial staging waits for H2D k-1 before it issues D2H k
        _, log = self._run()
        n = cp._XFER_BYTES // cp._PIPE_CHUNK
        ops = [x for x in log if x[0] in ("D2H", "H2D")][: 2 * n]
        idx = {(op, k): i for i, (op, k, _b) in enumerate(ops)}
        for k in range(1, n):
            self.assertLess(idx[("D2H", k)], idx[("H2D", k - 1)], f"chunk {k}")

    def test_the_h2d_of_a_chunk_is_not_waited_for_before_the_next_d2h_is_issued(self):
        # MUTANT guard "serial instead of pipeline" (a wait on H2D k-1 right after issuing it): D2H k must come before that wait
        _, log = self._run()
        n = cp._XFER_BYTES // cp._PIPE_CHUNK
        run = log[: next(i for i, x in enumerate(log) if x == ("sync", "h", n - 1)) + 1]
        for k in range(1, n - 1):                  # H2D n-2 is never waited for on its own: stream order covers it
            d2h_k = next(i for i, x in enumerate(run) if x[:2] == ("D2H", k))
            wait_prev = next(i for i, x in enumerate(run) if x == ("sync", "h", k - 1))
            self.assertLess(d2h_k, wait_prev, f"chunk {k}")

    def test_the_pair_headline_is_the_pipelined_rate_and_the_serial_one_sits_beside_it(self):
        # MUTANT guard "headline swapped": bandwidth_gbs must be the pipelined figure, bandwidth_serial_gbs the serial one
        import torch

        with mock.patch.object(cp, "_peer_ok", lambda a, b: False), \
             mock.patch.object(cp, "_time_copy_gbs", lambda dev, fn, nbytes=cp._XFER_BYTES: 3.0), \
             mock.patch.object(cp, "_staged_pipelined_gbs", lambda *a, **k: 9.0), \
             mock.patch.object(torch, "empty", lambda *a, **k: _FT("x", [])), \
             mock.patch.object(torch.cuda, "set_device", lambda *a: None), \
             mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
             mock.patch.object(torch.cuda, "empty_cache", lambda: None):
            gbs, lat, transport, peer, serial = cp._measure_one_pair(0, 1, staging=_FT("host", []))
        self.assertEqual((gbs, serial), (9.0, 3.0))
        self.assertEqual((transport, peer), (cp.HOST_STAGING, False))

    def test_over_peer_access_the_rate_is_direct_and_there_is_no_serial_figure(self):
        import torch

        with mock.patch.object(cp, "_peer_ok", lambda a, b: True), \
             mock.patch.object(cp, "_time_copy_gbs", lambda dev, fn, nbytes=cp._XFER_BYTES: 3.0), \
             mock.patch.object(torch, "empty", lambda *a, **k: _FT("x", [])), \
             mock.patch.object(torch.cuda, "set_device", lambda *a: None), \
             mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
             mock.patch.object(torch.cuda, "empty_cache", lambda: None):
            gbs, _lat, transport, _peer, serial = cp._measure_one_pair(0, 1)
        self.assertEqual((gbs, serial, transport), (3.0, None, cp.P2P_DIRECT))

    def test_a_buffer_is_reused_only_after_the_h2d_that_read_it_was_waited_for(self):
        _, log = self._run()
        n = cp._XFER_BYTES // cp._PIPE_CHUNK
        run = log[: next(i for i, x in enumerate(log) if x == ("sync", "h", n - 1)) + 1]
        for k in range(2, n):
            d2h_k = next(i for i, x in enumerate(run) if x[:2] == ("D2H", k))
            sync_prev = next(i for i, x in enumerate(run) if x == ("sync", "h", k - 2))
            self.assertLess(sync_prev, d2h_k, f"chunk {k}")
        bufs = {(x[1], x[2]) for x in run if x[0] == "D2H"}
        self.assertEqual({b for _k, b in bufs}, {"host0", "host1"})        # exactly two pinned buffers, alternating

    def test_too_small_a_copy_is_not_pipelined_and_gives_no_number(self):
        gbs, _ = self._run(nbytes=cp._PIPE_CHUNK)
        self.assertIsNone(gbs)


class TestPcieLink(CustomTestCase):
    def test_theory_rates_per_generation_and_width(self):
        self.assertEqual(hp.pcie_theory_gbs(4, 4), 7.88)
        self.assertEqual(hp.pcie_theory_gbs(4, 8), 15.75)
        self.assertEqual(hp.pcie_theory_gbs(4, 16), 31.51)
        self.assertIsNone(hp.pcie_theory_gbs(6, 16))      # not tabulated: no number, not a guess
        self.assertIsNone(hp.pcie_theory_gbs(None, 8))
        self.assertIsNone(hp.pcie_theory_gbs(4, 0))

    def test_link_of_a_card_is_read_by_uuid(self):
        cards = [{"uuid": U0, "pcie_cur_gen": 4, "pcie_cur_width": 4, "pcie_max_gen": 4, "pcie_max_width": 16},
                 {"uuid": U1, "pcie_cur_gen": 5, "pcie_cur_width": 8, "pcie_max_gen": 5, "pcie_max_width": 16}]
        with mock.patch.object(hp, "read_nvml", lambda: (cards, "d", [])):
            self.assertEqual(cp._pcie_link_of(U1), {"gen_cur": 5, "width_cur": 8, "gen_max": 5, "width_max": 16})
            self.assertEqual(cp._pcie_link_of("GPU-unknown"), {})

    def test_nvml_trouble_is_an_empty_link_never_an_exception(self):
        with mock.patch.object(hp, "read_nvml", side_effect=RuntimeError("nvml gone")):
            self.assertEqual(cp._pcie_link_of(U0), {})

    def test_the_view_shows_link_and_utilisation_against_the_CURRENT_width_not_the_maximum(self):
        with tempfile.TemporaryDirectory() as d:
            _write_probe(d, "card_probe-a.json", NOW - 5, [
                _probe_card(U0, "RTX 3080", h2d_gbs=6.5, d2h_gbs=6.6, pcie_gen_cur=4, pcie_width_cur=4,
                            pcie_gen_max=4, pcie_width_max=16)])
            doc = hp.build(cache_dir=d, nvml=_nvml(1), now=NOW)
        k = doc["cards"][0]["link"]
        self.assertEqual((k["gen_cur"]["v"], k["width_cur"]["v"], k["width_max"]["v"]), (4, 4, 16))
        self.assertEqual(k["gen_cur"]["src"], hp.SRC_NVML)
        self.assertEqual(k["theory_gbs"]["v"], 7.88)
        self.assertEqual(k["theory_gbs"]["src"], hp.SRC_DATASHEET)
        self.assertAlmostEqual(k["h2d_pct"]["v"], 82.5, places=1)       # 6.5 / 7.88, NOT 6.5 / 31.51
        self.assertAlmostEqual(k["d2h_pct"]["v"], 83.8, places=1)
        self.assertEqual(k["h2d_pct"]["src"], hp.SRC_ESTIMATED)         # a calculation, labelled as one
        self.assertIn("Rechnung", k["h2d_pct"]["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_a_probe_without_link_fields_shows_the_link_as_nicht_gemessen_with_reason_and_keeps_the_button_lit(self):
        card = _probe_card(U0, "RTX 3080")
        for key in ("pcie_gen_cur", "pcie_width_cur", "pcie_gen_max", "pcie_width_max"):
            card.pop(key)
        with tempfile.TemporaryDirectory() as d:
            _write_probe(d, "card_probe-old.json", NOW - 5, [card])
            doc = hp.build(cache_dir=d, nvml=_nvml(1), now=NOW)
        k = doc["cards"][0]["link"]
        for key in ("gen_cur", "width_cur", "theory_gbs", "h2d_pct", "d2h_pct"):
            self.assertIsNone(k[key]["v"], key)
            self.assertTrue(k[key]["note"], key)
        self.assertIn("link", doc["unmeasured"]["0"])
        self.assertEqual(hp.validate(doc), [])

    def test_measure_card_records_the_link_it_read_after_the_transfer_arm(self):
        import torch

        from sglang.srt import uneven_perf as up

        rates = up.MembwRates(read_gbs=1.0, copy_gbs=1.0, gemv_gbs=1.0)
        with mock.patch.object(torch.cuda, "set_device"), \
             mock.patch.object(up, "_bench_membw_rates", lambda dev: rates), \
             mock.patch.object(up, "_bench_gemm_tflops", lambda dev: 60.0), \
             mock.patch.object(cp, "_device_properties", lambda dev: (68, 5.0, "8.6")), \
             mock.patch.object(cp, "_bench_gemm_fp8_tflops", lambda dev: (None, "x")), \
             mock.patch.object(cp, "_bench_h2d_d2h", lambda dev: (6.0, 6.5)), \
             mock.patch.object(cp, "_pcie_link_of", lambda uuid: {"gen_cur": 4, "width_cur": 4, "gen_max": 4, "width_max": 16}
                               if uuid == "u0" else {}), \
             mock.patch.object(cp, "_bench_h2d_d2h_latency", lambda dev: (12.5, 14.0, 9.0, 10.0)), \
             mock.patch.object(cp, "lane_environment_issue", lambda: "x"):
            m = cp.measure_card(0, "u0", "card", 1)
            m2 = cp.measure_card(0, "u9", "card", 1)
        self.assertEqual((m.pcie_gen_cur, m.pcie_width_cur, m.pcie_gen_max, m.pcie_width_max), (4, 4, 4, 16))
        self.assertEqual(m2.pcie_width_cur, None)
        self.assertIn("pcie_link", m.arm_seconds)


class TestNcclWay(CustomTestCase):
    VIA_P2P = "nccl INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read"
    VIA_SHM = "nccl INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct\nnccl INFO Channel 01/0 : 1[1] -> 0[0] via SHM/direct/direct"

    def _rep(self, rank, uuid, rate=None, lat=None, failed=None):
        r = {"rank": rank, "uuid": uuid}
        if failed:
            r["failed"] = failed
        else:
            r["rate"] = rate or []
            r["lat"] = lat or []
        return r

    def _results(self, via=VIA_SHM, a_rates=(None, 9.0), b_rates=(8.0, None)):
        # rank 0 = U0, rank 1 = U1.  The RECEIVER owns the rate of a direction; the SENDER owns its latency.
        r0 = self._rep(0, U0, rate=[{"src": 1, "dst": 0, "gbs": a_rates[1]}] if a_rates[1] else [],
                       lat=[{"src": 0, "dst": 1, "latency_us": 31.0, "n": 200}])
        r1 = self._rep(1, U1, rate=[{"src": 0, "dst": 1, "gbs": b_rates[0]}] if b_rates[0] else [],
                       lat=[{"src": 1, "dst": 0, "latency_us": 33.0, "n": 200}])
        return [(0, "x\n" + npb.MARKER + json.dumps(r0) + "\n", via), (0, npb.MARKER + json.dumps(r1) + "\n", via)]

    def test_command_and_env_nccl_debug_only_in_the_child_and_nothing_else_forced(self):
        base = {"PATH": "/bin", "NCCL_P2P_DISABLE": "1", "CUDA_VISIBLE_DEVICES": "0"}
        e = npb.rank_env([U0, U1], 4711, base=base)
        self.assertEqual(e["NCCL_DEBUG"], "INFO")
        self.assertEqual(e["NCCL_P2P_DISABLE"], "1")                       # the operating environment is kept
        self.assertEqual(e["CUDA_VISIBLE_DEVICES"], f"{U0},{U1}")
        self.assertEqual(e["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("NCCL_DEBUG", base)                               # the caller's dict is untouched
        c = npb.rank_command("/py", 1, [U0, U1], 100.0)
        self.assertEqual(c[c.index("--rank") + 1], "1")
        self.assertEqual(c[c.index("--uuids") + 1], f"{U0},{U1}")

    def test_the_chosen_transport_is_read_from_the_nccl_log(self):
        self.assertEqual(npb.parse_transports(self.VIA_P2P), ["P2P/CUMEM/read"])
        self.assertEqual(npb.parse_transports(self.VIA_SHM), ["SHM/direct/direct"])
        self.assertEqual(npb.parse_transports("nothing useful"), [])

    def test_merge_gives_each_direction_the_numbers_of_its_owner_and_names_the_transport(self):
        pairs, tr = npb.merge_pair([U0, U1], self._results(a_rates=(None, 9.0), b_rates=(8.0, None)))
        by = {(p["src_uuid"], p["dst_uuid"]): p for p in pairs}
        self.assertEqual(by[(U0, U1)]["bandwidth_gbs"], 8.0)      # measured by the RECEIVER U1
        self.assertEqual(by[(U1, U0)]["bandwidth_gbs"], 9.0)      # measured by the RECEIVER U0
        self.assertEqual(by[(U0, U1)]["latency_us"], 31.0)        # started by the SENDER U0
        self.assertEqual(by[(U1, U0)]["latency_us"], 33.0)
        self.assertIn("SHM/direct/direct", by[(U0, U1)]["transport"])
        self.assertIn("SHM/direct/direct", by[(U0, U1)]["note"])
        self.assertEqual(tr, ["SHM/direct/direct"])

    def test_the_device_side_latency_comes_from_the_sender_of_the_direction_and_is_labelled(self):
        res = self._results()
        r0 = json.loads(res[0][1].split(npb.MARKER)[1])
        r1 = json.loads(res[1][1][len(npb.MARKER):])
        r0["lat"][0]["latency_device_us"] = 11.5
        r1["lat"][0]["latency_device_us"] = 12.5
        res[0] = (0, npb.MARKER + json.dumps(r0) + "\n", res[0][2])
        res[1] = (0, npb.MARKER + json.dumps(r1) + "\n", res[1][2])
        pairs, _ = npb.merge_pair([U0, U1], res)
        by = {(p["src_uuid"], p["dst_uuid"]): p for p in pairs}
        self.assertEqual(by[(U0, U1)]["latency_device_us"], 11.5)
        self.assertEqual(by[(U1, U0)]["latency_device_us"], 12.5)
        self.assertIn("ein Synchronize am Ende", by[(U0, U1)]["latency_device_kind"])
        self.assertIn("Rundlauf / 2", by[(U0, U1)]["latency_device_kind"])

    def test_a_pair_run_that_died_is_two_pairs_without_number_with_the_reason(self):
        pairs, _ = npb.merge_pair([U0, U1], [(1, "", "Traceback: NCCL error"), (-9, "", "")])
        self.assertTrue(all(p["bandwidth_gbs"] is None and p["latency_us"] is None for p in pairs))
        self.assertTrue(all("not measured" in p["note"] for p in pairs))

    def test_numbers_of_a_child_on_the_wrong_card_are_discarded(self):
        res = self._results()
        wrong = json.loads(res[1][1][len(npb.MARKER):])
        wrong["uuid"] = "GPU-9999"
        res[1] = (0, npb.MARKER + json.dumps(wrong) + "\n", "")
        pairs, _ = npb.merge_pair([U0, U1], res)
        by = {(p["src_uuid"], p["dst_uuid"]): p for p in pairs}
        self.assertIsNone(by[(U0, U1)]["bandwidth_gbs"])
        self.assertIn("reported card", by[(U0, U1)]["note"])

    def test_nccl_that_did_not_come_up_is_the_reason(self):
        f = {"stage": "worker", "reason": "RuntimeError: NCCL init timeout"}
        res = [(0, npb.MARKER + json.dumps(self._rep(0, U0, failed=f)) + "\n", ""),
               (0, npb.MARKER + json.dumps(self._rep(1, U1, failed=f)) + "\n", "")]
        pairs, _ = npb.merge_pair([U0, U1], res)
        self.assertTrue(all(p["bandwidth_gbs"] is None and "NCCL init timeout" in p["note"] for p in pairs))

    def test_run_one_pair_run_per_unordered_pair_each_with_its_own_cap_and_result_sorted_by_card_order(self):
        seen = []

        def runner(cmds, env, timeout):
            seen.append((cmds, env, timeout))
            u = cmds[0][cmds[0].index("--uuids") + 1].split(",")
            r0 = self._rep(0, u[0], rate=[{"src": 1, "dst": 0, "gbs": 5.0}], lat=[{"src": 0, "dst": 1, "latency_us": 20.0, "n": 200}])
            r1 = self._rep(1, u[1], rate=[{"src": 0, "dst": 1, "gbs": 6.0}], lat=[{"src": 1, "dst": 0, "latency_us": 21.0, "n": 200}])
            return [(0, npb.MARKER + json.dumps(r0) + "\n", self.VIA_P2P), (0, npb.MARKER + json.dumps(r1) + "\n", self.VIA_P2P)]

        res = npb.run_nccl_probe([{"uuid": u} for u in UUIDS], timeout_s=240.0, runner=runner, env={"PATH": "/bin"}, port=1)
        self.assertEqual(len(seen), 3)                                       # (0,1) (0,2) (1,2)
        self.assertTrue(all(t == 80.0 for _c, _e, t in seen))                # own cap per pair-run
        self.assertEqual(len(res.pairs), 6)
        self.assertEqual([(p["src_uuid"], p["dst_uuid"]) for p in res.pairs],
                         [(a, b) for a in UUIDS for b in UUIDS if a != b])
        self.assertEqual(res.reason, "")
        self.assertEqual(res.transports, ["P2P/CUMEM/read"])

    def test_a_runner_that_raises_or_a_used_up_budget_is_a_reason_not_an_exception(self):
        def boom(*a):
            raise OSError("no fork")

        res = npb.run_nccl_probe([{"uuid": u} for u in UUIDS], runner=boom)
        self.assertEqual(len(res.pairs), 6)
        self.assertTrue(all(p["bandwidth_gbs"] is None for p in res.pairs))
        self.assertIn("no fork", res.reason)

    def test_fewer_than_two_cards_has_no_nccl_step(self):
        self.assertIn("fewer than two", npb.run_nccl_probe([{"uuid": U0}], runner=lambda *a: []).reason)


class TestCardProbeNcclWiring(CustomTestCase):
    def _run(self, **kw):
        gpus = [{"cuda_index": i, "uuid": u, "name": f"c{i}", "total_mib": 1} for i, u in enumerate(UUIDS)]
        with mock.patch.object(cp, "_inventory", lambda: (gpus, "d")), \
             mock.patch.object(cp, "_card_states", lambda: {}), \
             mock.patch.object(cp, "measure_card",
                               lambda cuda_index, uuid, name, total_mib=None, state_fn=None:
                               cp.CardProbeMeasurement(uuid=uuid, name=name, cuda_index=cuda_index, gemm_bf16_tflops=60.0)), \
             mock.patch.object(cp, "measure_pair_matrix", lambda g: []):
            return cp.run_card_probe(**kw)

    def test_off_for_library_callers(self):
        with mock.patch.object(npb, "run_nccl_probe", side_effect=AssertionError("must not run")):
            prof = self._run(save=False)
        self.assertFalse(prof.nccl_attempted)

    def test_result_is_stored_in_its_own_list_and_round_trips(self):
        fake = npb.NcclResult(pairs=[
            {"src_uuid": U0, "dst_uuid": U1, "bandwidth_gbs": 5.0, "latency_us": 30.0, "transport": "nccl send/recv (SHM/direct/direct)",
             "peer_access": False, "bytes_moved": 1, "note": "n"}], reason="", seconds=12.0)
        with mock.patch.object(npb, "run_nccl_probe", return_value=fake) as m:
            prof = self._run(save=False, nccl=True)
        self.assertEqual([[g["uuid"] for g in c.args[0]] for c in m.call_args_list], [UUIDS])
        self.assertTrue(prof.nccl_attempted)
        self.assertEqual(prof.pairs, [])
        self.assertEqual(prof.bar1_pairs, [])
        back = cp.CardProbeProfile.from_json(json.loads(json.dumps(prof.to_json())))
        self.assertEqual((back.nccl_pairs[0].bandwidth_gbs, back.nccl_attempted, back.nccl_seconds), (5.0, True, 12.0))
        self.assertIn("NCCL send/recv per ordered pair", cp.format_text(prof))

    def test_a_failing_step_keeps_the_card_rates_and_records_why(self):
        with mock.patch.object(npb, "run_nccl_probe", side_effect=RuntimeError("nccl missing")):
            prof = self._run(save=False, nccl=True)
        self.assertTrue(prof.nccl_attempted)
        self.assertEqual([c.gemm_bf16_tflops for c in prof.cards], [60.0] * 3)
        self.assertIn("nccl missing", prof.nccl_reason)
        self.assertEqual(len(prof.nccl_pairs), 6)

    def test_the_cache_file_exists_after_the_card_stage_before_any_optional_way_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "card_probe-x.json")
            seen = {}

            def bar1(gpus, **kw):
                seen["file_when_bar1_starts"] = os.path.exists(path)
                return bp.Bar1Result(reason="x")

            with mock.patch.object(bp, "run_bar1_probe", bar1), \
                 mock.patch.object(npb, "run_nccl_probe", return_value=npb.NcclResult(reason="y")):
                self._run(path=path, bar1=True, nccl=True)
            self.assertTrue(seen["file_when_bar1_starts"])
            with open(path) as fh:
                final = json.load(fh)
            self.assertTrue(final["bar1_attempted"])                # re-saved after the way

    def test_cli_nccl_flags(self):
        seen = []

        def fake_run(**kw):
            seen.append(kw)
            return cp.CardProbeProfile(created=time.time())

        with mock.patch.object(cp, "run_card_probe", side_effect=fake_run), mock.patch.object(cp, "lane_environment_issue", lambda: ""):
            cp._main(["--run", "--json"])
            cp._main(["--run", "--json", "--no-nccl"])
            cp._main(["--run", "--json", "--nccl-timeout-s", "99"])
        self.assertEqual([k["nccl"] for k in seen], [True, False, True])
        self.assertEqual(seen[2]["nccl_timeout_s"], 99.0)

    def test_the_two_ways_together_stay_inside_eight_minutes(self):
        self.assertLessEqual(bp.DEFAULT_TIMEOUT_S + npb.DEFAULT_TIMEOUT_S, 8 * 60)


def _nccl_pair(s, d, bw, lat=30.0, via="SHM/direct/direct"):
    return {"src_uuid": s, "dst_uuid": d, "bandwidth_gbs": bw, "latency_us": lat if bw is not None else None,
            "transport": f"nccl send/recv ({via})", "peer_access": False, "note": f"NCCL chose: {via}" if bw is not None else "NCCL not measured: x"}


def _stage_pair(s, d, pipe, serial, lat=21.0):
    return {"src_uuid": s, "dst_uuid": d, "bandwidth_gbs": pipe, "bandwidth_serial_gbs": serial, "latency_us": lat,
            "transport": cp.HOST_STAGING, "peer_access": False, "note": "staged"}


class TestD2DTable(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _doc(self, bar1=None, nccl=None, stage=None, **extra):
        cards = [_probe_card(U0, "RTX 3080", lane_notes={"nvfp4_w4a4": "no native FP4 tensor cores"}),
                 _probe_card(U2, "RTX 3080", lane_notes={"nvfp4_w4a4": "no native FP4 tensor cores"}),
                 _probe_card(U1, "RTX 5090", gemm_w4a4_tflops=900.0, compute_capability="12.0",
                             lane_notes={"nvfp4_w4a8": "sm_8x"})]
        kw = dict(pairs=stage if stage is not None else [])
        if bar1 is not None:
            kw.update(bar1_attempted=True, bar1_pairs=bar1, bar1_reason="" if bar1 else "dmabuf_holder not available")
        if nccl is not None:
            kw.update(nccl_attempted=True, nccl_pairs=nccl, nccl_reason="" if nccl else "NCCL did not come up")
        kw.update(extra)
        _write_probe(self.d.name, "card_probe-a.json", NOW - 30, cards, **kw)
        return hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)

    def _row(self, doc, a, b):
        ord_of = {c["uuid"]: c["ord"] for c in doc["cards"]}
        return next(r for r in doc["d2d"]["pairs"] if (r["src"], r["dst"]) == (ord_of[a], ord_of[b]))

    def test_the_headline_is_barlink_bar1_and_the_columns_come_in_that_order(self):
        doc = self._doc()
        self.assertEqual(doc["d2d"]["headline"], "barlink_bar1")
        self.assertEqual([c["key"] for c in doc["d2d"]["columns"]], ["barlink_bar1", "nccl", "host_staging"])
        self.assertIn("Betriebsweg", doc["d2d"]["columns"][0]["label"])
        self.assertIn("Fallback, nicht der Betriebsweg", doc["d2d"]["columns"][2]["label"])
        self.assertEqual(len(doc["d2d"]["pairs"]), 6)

    def test_without_a_bar1_measurement_the_headline_is_nicht_gemessen_even_if_staging_and_nccl_are_measured(self):
        # MUTANT guard: "host staging (or NCCL) as the headline number"
        ST = [_stage_pair(a, b, 12.0, 6.0) for a in UUIDS for b in UUIDS if a != b]
        NC = [_nccl_pair(a, b, 9.0) for a in UUIDS for b in UUIDS if a != b]
        doc = self._doc(bar1=None, nccl=NC, stage=ST)
        for r in doc["d2d"]["pairs"]:
            for k in ("gbs", "lat_us"):
                n = r["barlink_bar1"][k]
                self.assertIsNone(n["v"])
                self.assertEqual(n["src"], hp.SRC_NONE)
                self.assertTrue(n["note"])
            self.assertEqual(r["host_staging"]["gbs"]["v"], 12.0)
            self.assertEqual(r["nccl"]["gbs"]["v"], 9.0)
        self.assertEqual(hp.validate(doc), [])

    def test_all_three_ways_side_by_side_each_with_rate_and_latency_and_never_swapped(self):
        # every way gets different numbers, so a swapped column cannot pass
        BA = [_bar1_pair(a, b, 4.0 + i, lat=7.0 + i) for i, (a, b) in enumerate((a, b) for a in UUIDS for b in UUIDS if a != b)]
        NC = [_nccl_pair(a, b, 20.0 + i, lat=40.0 + i) for i, (a, b) in enumerate((a, b) for a in UUIDS for b in UUIDS if a != b)]
        ST = [_stage_pair(a, b, 60.0 + i, 30.0 + i, lat=80.0 + i) for i, (a, b) in enumerate((a, b) for a in UUIDS for b in UUIDS if a != b)]
        doc = self._doc(bar1=BA, nccl=NC, stage=ST)
        for i, (a, b) in enumerate((a, b) for a in UUIDS for b in UUIDS if a != b):
            r = self._row(doc, a, b)
            self.assertEqual((r["barlink_bar1"]["gbs"]["v"], r["barlink_bar1"]["lat_us"]["v"]), (4.0 + i, 7.0 + i))
            self.assertEqual((r["nccl"]["gbs"]["v"], r["nccl"]["lat_us"]["v"]), (20.0 + i, 40.0 + i))
            self.assertEqual((r["host_staging"]["gbs"]["v"], r["host_staging"]["gbs_serial"]["v"], r["host_staging"]["lat_us"]["v"]),
                             (60.0 + i, 30.0 + i, 80.0 + i))
            for way in (r["barlink_bar1"], r["nccl"], r["host_staging"]):
                self.assertEqual(way["gbs"]["src"], hp.SRC_MEASURED)
            self.assertIn("SHM/direct/direct", r["nccl"]["transport"])
        self.assertTrue(doc["bar1"]["complete"] and doc["nccl"]["complete"])
        self.assertFalse(doc["measure_needed"], doc["unmeasured"])
        self.assertEqual(hp.validate(doc), [])

    def test_a_failed_nccl_way_is_nicht_gemessen_with_its_reason_in_its_own_column_only(self):
        BA = [_bar1_pair(a, b, 4.0) for a in UUIDS for b in UUIDS if a != b]
        ST = [_stage_pair(a, b, 12.0, 6.0) for a in UUIDS for b in UUIDS if a != b]
        doc = self._doc(bar1=BA, nccl=[], stage=ST)
        for r in doc["d2d"]["pairs"]:
            self.assertIsNone(r["nccl"]["gbs"]["v"])
            self.assertIn("NCCL did not come up", r["nccl"]["gbs"]["note"])
            self.assertEqual(r["barlink_bar1"]["gbs"]["v"], 4.0)
        self.assertFalse(doc["nccl"]["measured"])
        self.assertNotIn("nccl", doc["unmeasured"])                  # ran and said why: final

    def test_an_nccl_step_that_never_ran_is_an_open_gap(self):
        doc = self._doc(bar1=[])
        self.assertEqual(len(doc["unmeasured"]["nccl"]), 6)
        self.assertTrue(doc["measure_needed"])
        self.assertIn("nicht gemessen", doc["d2d"]["pairs"][0]["nccl"]["gbs"]["note"])

    def test_a_pre_pipelined_probe_shows_its_number_as_the_serial_one_and_the_pipelined_rate_as_not_measured(self):
        old = [{"src_uuid": a, "dst_uuid": b, "bandwidth_gbs": 6.9, "latency_us": 21.0, "transport": cp.HOST_STAGING,
                "peer_access": False} for a in UUIDS for b in UUIDS if a != b]            # no bandwidth_serial_gbs: the old format
        doc = self._doc(stage=old)
        r = doc["d2d"]["pairs"][0]["host_staging"]
        self.assertIsNone(r["gbs"]["v"])
        self.assertIn("pipelined nicht gemessen", r["gbs"]["note"])
        self.assertEqual(r["gbs_serial"]["v"], 6.9)                                    # never relabelled as pipelined

    def test_no_bar1_number_ever_comes_from_the_nccl_or_the_staging_pairs(self):
        # MUTANT guard "NCCL number in the BAR1 column": only NCCL measured -> the BAR1 headline stays empty
        NC = [_nccl_pair(a, b, 99.0) for a in UUIDS for b in UUIDS if a != b]
        doc = self._doc(nccl=NC)
        self.assertEqual({r["barlink_bar1"]["gbs"]["v"] for r in doc["d2d"]["pairs"]}, {None})
        self.assertEqual({l["gbs"]["v"] for l in doc["links"] if l["transport"] == "bar1"}, {None})
        self.assertEqual({l["gbs"]["v"] for l in doc["links"] if l["transport"] == "nccl_pair"}, {99.0})

    def test_the_view_makes_no_capability_or_faster_slower_claim(self):
        doc = self._doc()
        text = json.dumps([doc["d2d"], doc["bar1"], doc["nccl"]], ensure_ascii=False).lower()
        for bad in ("schneller", "langsamer", "fähigkeit", "kann der link", "faster", "slower", "capability of the link"):
            self.assertNotIn(bad, text)
        for k, v in hp.D2D_DEFINITIONS.items():
            self.assertTrue(v, k)
        # the definitions say what each latency IS, in the plain words the reader needs
        self.assertIn("SENDERseite", hp.D2D_DEFINITIONS["barlink_bar1"])
        self.assertIn("keine Zustellzeit", hp.D2D_DEFINITIONS["barlink_bar1"])
        self.assertIn("Rückweg / 2", hp.D2D_DEFINITIONS["nccl"])
        self.assertIn("seriell", hp.D2D_DEFINITIONS["host_staging"])


class TestSecondLatencyAndReferences(CustomTestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _doc(self, **kw):
        cards = [_probe_card(U0, "RTX 3080"), _probe_card(U2, "RTX 3080"), _probe_card(U1, "RTX 5090")]
        _write_probe(self.d.name, "card_probe-a.json", NOW - 30, cards, **kw)
        return hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)

    def test_both_latencies_sit_in_their_own_nodes_per_way_with_the_kind_in_the_note(self):
        ordered = [(a, b) for a in UUIDS for b in UUIDS if a != b]
        BA = [dict(_bar1_pair(a, b, 4.0, lat=10.5), latency_device_us=1.25, latency_device_kind="4-kB-Schreibzugriffe hintereinander im Stream (KEIN Rundlauf)")
              for a, b in ordered]
        NC = [dict(_nccl_pair(a, b, 9.0, lat=31.0), latency_device_us=14.0, latency_device_kind="Ping-Pong 200 Runden, Rundlauf / 2") for a, b in ordered]
        doc = self._doc(bar1_attempted=True, bar1_pairs=BA, bar1_reason="", nccl_attempted=True, nccl_pairs=NC, nccl_reason="")
        r = doc["d2d"]["pairs"][0]
        self.assertEqual((r["barlink_bar1"]["lat_us"]["v"], r["barlink_bar1"]["lat_dev_us"]["v"]), (10.5, 1.25))
        self.assertEqual((r["nccl"]["lat_us"]["v"], r["nccl"]["lat_dev_us"]["v"]), (31.0, 14.0))
        self.assertIn("KEIN Rundlauf", r["barlink_bar1"]["lat_dev_us"]["note"])
        self.assertIn("Start", r["barlink_bar1"]["lat_us"]["note"])
        self.assertIn("keine Wire-Latenz", r["barlink_bar1"]["lat_us"]["note"])
        self.assertEqual(hp.validate(doc), [])

    def test_a_probe_without_the_second_latency_says_so_and_shows_no_number(self):
        ordered = [(a, b) for a in UUIDS for b in UUIDS if a != b]
        doc = self._doc(bar1_attempted=True, bar1_pairs=[_bar1_pair(a, b, 4.0) for a, b in ordered], bar1_reason="",
                        nccl_attempted=True, nccl_pairs=[], nccl_reason="NCCL did not come up")
        r = doc["d2d"]["pairs"][0]
        self.assertIsNone(r["barlink_bar1"]["lat_dev_us"]["v"])
        self.assertTrue(r["barlink_bar1"]["lat_dev_us"]["note"])
        self.assertIsNone(r["nccl"]["lat_dev_us"]["v"])

    def test_the_references_carry_the_measured_values_with_file_and_line_and_no_comparison(self):
        refs = hp.D2D_REFERENCES
        blob = " ".join(r["what"] + " " + r["source"] for r in refs)
        for needle in ("28,22", "323,2", "6,02", "7,30", "37,41", "45,59", "0,08 ms", "0,53-0,80", "SHM/direct/direct", "32,4", "361,3"):
            self.assertIn(needle, blob, needle)
        for src in ("barlink_bar1.py:75-83", "roundbench_fixed_0907.out:19", "bench_host_transport.py:10-12", "ANALYSE_732_bar1_repricing.md:58-64",
                    "27b-nvfp4-dual.env:122", "nccl_transport.json", "hw_profile-9a5e9b49b7dc.json"):
            self.assertIn(src, blob, src)
        self.assertTrue(all(r["what"] and r["source"] for r in refs))
        low = blob.lower()
        for bad in ("schneller", "langsamer", "fähigkeit", "faster", "slower"):
            self.assertNotIn(bad, low)
        self.assertEqual(list(self._doc()["d2d"]["references"]), list(refs))

    def test_the_cited_numbers_are_really_in_the_cited_places(self):
        # the references are only worth anything if they match their sources: check the ones that live in this tree
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")

        def lines(rel):
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                return fh.read().splitlines()

        bb = lines("python/sglang/srt/distributed/device_communicators/barlink_bar1.py")
        self.assertIn("28.22", " ".join(bb[74:83]))
        self.assertIn("4077.43", " ".join(bb[74:83]))
        self.assertTrue(bb[1527].startswith("DEFAULT_ROUND_US = 323.2"))
        self.assertTrue(bb[1536].startswith("DEFAULT_WIRE_GBPS = 6.02"))
        host = " ".join(lines("benchmark/bench_host_transport.py")[9:12])
        self.assertIn("7.30 us", host)
        self.assertIn("37.41 us", host)
        an = lines("docs/dev/ANALYSE_732_bar1_repricing.md")
        self.assertIn("45.59", an[57])

    def test_the_stage0_nccl_row_is_not_called_p2p_it_goes_over_the_host_and_carries_its_date(self):
        # MUTANT guard: the old label claimed peer-to-peer on a rig that has none
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(NOW - 100))
        with open(os.path.join(self.d.name, "hw_profile-x.json"), "w") as f:
            json.dump({"version": 3, "driver": "595.58", "created": created, "gpus": {U0: {"name": "n"}, U1: {"name": "n"}},
                       "links": {f"{U0}|{U1}": {"p2p_gbs": 5.1}}}, f)
        doc = hp.build(cache_dir=self.d.name, nvml=_nvml(), now=NOW)
        nccl = [l for l in doc["links"] if l["transport"] == "nccl"]
        self.assertEqual(len(nccl), 2)                                   # measured direction + the mirrored one
        want = time.strftime("%d.%m.%Y", time.localtime(NOW - 100))
        for l in nccl:
            self.assertEqual(l["transport_label"], f"NCCL über Host (Stufe-0-Probe, {want})")
            self.assertNotIn("p2p", l["transport_label"].lower())
        self.assertEqual({l["gbs"]["src"] for l in nccl}, {hp.SRC_MEASURED, hp.SRC_ESTIMATED})


if __name__ == "__main__":
    unittest.main()
