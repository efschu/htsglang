# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for comm_suite.py + rig_profile_source.py -- the #271 sources.

Hermetic: every arm is replaced by a stub, so nothing here measures, boots,
takes a card or opens a socket. What is tested is the SUITE's contract, not
the physics it wraps:

  * a failing arm is recorded as a finding and the run continues (no
    survivorship sample), a cancelled arm is absent rather than failed;
  * arms run CPU-first, so a rig whose cards belong to somebody else still
    produces an artifact;
  * a GPU arm that cannot get a card window is absent WITH the holder named
    -- and the quiet flag is honored;
  * the run turns into rows of the shared schema, aggregates only, with the
    context a comparison needs;
  * the profile source reads caches without measuring and re-keys GPU UUIDs
    and model paths to things that may be shared.
"""

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.planner import comm_suite, rig_artifact, rig_profile_source
from sglang.srt.planner.comm_suite import (
    ARMS,
    ArmResult,
    CommSuiteJobStore,
    arm_specs_json,
    to_sections,
)

CELLS = {"all_reduce/20KiB": {"median_us": 37.2, "p5_us": 30.0,
                              "p95_us": 44.0, "n": 60, "gbit_s": 4.4,
                              "spread_pct": 37.6}}


def _store(runners=None):
    store = CommSuiteJobStore()
    store.synchronous = True
    base = {a.id: (lambda ctx, _a=a: ArmResult(_a.id, "absent",
                                               absent_reason="stub"))
            for a in ARMS}
    base["rig_profile"] = lambda ctx: ArmResult(
        "rig_profile", "ok",
        facts={"cards": [{"index": 0, "name": "NVIDIA GeForce RTX 5090",
                          "vram_mib": 32607}],
               "card_count": 1, "card_summary": "1x RTX 5090",
               "driver": "595.58.03", "cuda": "13.0"})
    base.update(runners or {})
    store.runners = base
    return store


class CatalogueTest(unittest.TestCase):
    def test_every_arm_declares_kind_question_and_budget(self):
        for spec in arm_specs_json():
            self.assertIn(spec["kind"],
                          ("inventory", "cpu", "gpu", "network"), spec["id"])
            self.assertTrue(spec["question"], spec["id"])
            self.assertGreater(spec["budget_s"], 0, spec["id"])

    def test_the_whole_suite_fits_the_short_run_promise(self):
        # The budgets are ceilings, not expected times; the sum still has to
        # be a number a user would sit through.
        total = sum(a.budget_s for a in ARMS)
        self.assertLessEqual(total, 400.0)

    def test_cpu_arms_exist_so_a_busy_rig_still_yields_something(self):
        kinds = {a.kind for a in ARMS}
        self.assertIn("cpu", kinds)
        self.assertIn("gpu", kinds)


class RunTest(unittest.TestCase):
    def test_a_failing_arm_is_data_and_the_run_continues(self):
        def boom(ctx):
            raise RuntimeError("ncclInternalError: unhandled system error")

        store = _store({
            "collective_gloo": boom,
            "collective_htccl_ucx": lambda c: ArmResult(
                "collective_htccl_ucx", "ok", cells=dict(CELLS)),
        })
        job = store.start()
        self.assertEqual(job.state, "ok")
        failed = job.results["collective_gloo"]
        self.assertEqual(failed.status, "error")
        self.assertIn("ncclInternalError", failed.error)
        self.assertEqual(job.results["collective_htccl_ucx"].status, "ok")

    def test_arms_run_cpu_before_gpu(self):
        order = []

        def rec(arm_id):
            def f(ctx):
                order.append(arm_id)
                return ArmResult(arm_id, "ok")
            return f

        store = _store({a.id: rec(a.id) for a in ARMS})
        store.start()
        kinds = [next(a.kind for a in ARMS if a.id == i) for i in order]
        last_cpu = max(i for i, k in enumerate(kinds)
                       if k in ("inventory", "cpu"))
        first_gpu = min((i for i, k in enumerate(kinds) if k == "gpu"),
                        default=len(kinds))
        self.assertLess(last_cpu, first_gpu)

    def test_a_cancelled_arm_is_absent_not_failed(self):
        store = _store()
        job = comm_suite.CommSuiteJob(job_id="t", state="running",
                                      started_at=time.time(),
                                      selected=["collective_gloo"])
        job.cancel("collective_gloo")
        store._one(job, comm_suite._RunCtx(job=job), "collective_gloo")
        self.assertEqual(job.results["collective_gloo"].status, "absent")

    def test_single_flight(self):
        store = CommSuiteJobStore()
        store.synchronous = False
        started = {}

        def slow(ctx):
            time.sleep(0.4)
            return ArmResult("rig_profile", "ok", facts={"card_count": 0})

        store.runners = {a.id: (lambda c, _i=a.id: ArmResult(_i, "ok"))
                         for a in ARMS}
        store.runners["rig_profile"] = slow
        a = store.start()
        b = store.start()
        started["same"] = a.job_id == b.job_id
        self.assertTrue(started["same"])


class CardWindowTest(unittest.TestCase):
    def test_a_held_card_makes_gpu_arms_absent_with_the_holder_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, "gpu-card-{}.lock")
            os.mkdir(lock.format(0))
            with open(os.path.join(lock.format(0), "info"), "w") as f:
                f.write("owner=someone-elses-job\npurpose=x\n")
            with mock.patch.object(comm_suite, "LOCK_DIR_FMT", lock), \
                 mock.patch.object(comm_suite, "LEGACY_LOCK_DIR",
                                   os.path.join(tmp, "none")), \
                 mock.patch.object(comm_suite, "QUIET_LOCK_DIR",
                                   os.path.join(tmp, "quiet")):
                win = comm_suite._CardWindow([0])
                self.assertFalse(win.acquire())
                self.assertIn("someone-elses-job", win.reason)

    def test_the_quiet_flag_stops_a_new_gpu_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            quiet = os.path.join(tmp, "gpu-quiet.lock")
            os.mkdir(quiet)
            with open(os.path.join(quiet, "info"), "w") as f:
                f.write("owner=latency-window\n")
            with mock.patch.object(comm_suite, "QUIET_LOCK_DIR", quiet), \
                 mock.patch.object(comm_suite, "LOCK_DIR_FMT",
                                   os.path.join(tmp, "c{}.lock")), \
                 mock.patch.object(comm_suite, "LEGACY_LOCK_DIR",
                                   os.path.join(tmp, "none")):
                win = comm_suite._CardWindow([0])
                self.assertFalse(win.acquire())
                self.assertIn("quiet window", win.reason)

    def test_a_busy_card_is_not_free_even_when_the_lock_is(self):
        # A host-side process is invisible to the container's compute-apps
        # query, so memory.used is the check that counts (runbook §7.1).
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(comm_suite, "LOCK_DIR_FMT",
                                   os.path.join(tmp, "c{}.lock")), \
                 mock.patch.object(comm_suite, "LEGACY_LOCK_DIR",
                                   os.path.join(tmp, "none")), \
                 mock.patch.object(comm_suite, "QUIET_LOCK_DIR",
                                   os.path.join(tmp, "quiet")), \
                 mock.patch.object(comm_suite, "_nvidia_smi_cards",
                                   return_value=[{"index": 0,
                                                  "used_mib": 17000}]):
                win = comm_suite._CardWindow([0])
                self.assertFalse(win.acquire())
                self.assertIn("17000 MiB", win.reason)
                self.assertEqual(win.held, [],
                                 "a failed acquire must release everything")

    def test_gpu_arms_are_absent_with_the_reason_when_no_window(self):
        store = _store()
        with mock.patch.object(comm_suite._CardWindow, "acquire",
                               return_value=False), \
             mock.patch.object(comm_suite._CardWindow, "reason",
                               "card 1 is held by other-job", create=True):
            job = store.start()
        gpu_ids = [a.id for a in ARMS if a.kind == "gpu"]
        for arm_id in gpu_ids:
            self.assertEqual(job.results[arm_id].status, "absent", arm_id)
            self.assertTrue(job.results[arm_id].absent_reason, arm_id)


class GpuArmShapeTest(unittest.TestCase):
    """The GPU arms' RESULT SHAPING, with the measurement mocked out.

    The cards on this rig belong to another job most of the time, so these
    arms are usually absent. That is exactly why the code between the worker
    and the artifact must be tested here: a field name that is only wrong
    once a card frees up is a bug that surfaces at the worst moment.
    """

    def test_nccl_arm_maps_cells_and_the_exactness_verdict(self):
        payload = {"world": 3, "exact_mismatches": 0, "cells": dict(CELLS)}
        with mock.patch.object(comm_suite, "_worker_arm",
                               return_value=(payload, "")):
            ctx = comm_suite._RunCtx(job=mock.MagicMock(), card_count=3)
            res = comm_suite._arm_collective_nccl(ctx)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.facts["world"], 3)
        self.assertIn("all_reduce/20KiB", res.cells)

    def test_nccl_arm_reports_an_inexact_result_as_an_error(self):
        payload = {"world": 3, "exact_mismatches": 2, "cells": {}}
        with mock.patch.object(comm_suite, "_worker_arm",
                               return_value=(payload, "")):
            res = comm_suite._arm_collective_nccl(
                comm_suite._RunCtx(job=mock.MagicMock(), card_count=3))
        self.assertEqual(res.status, "error")
        self.assertIn("inexact", res.error)

    def test_nccl_arm_is_absent_on_a_single_card_rig(self):
        res = comm_suite._arm_collective_nccl(
            comm_suite._RunCtx(job=mock.MagicMock(), card_count=1))
        self.assertEqual(res.status, "absent")
        self.assertIn("two cards", res.absent_reason)

    def test_shm_arm_says_it_has_no_all_gather(self):
        payload = {"world": 2, "exact_mismatches": 0, "slot_bytes": 266240,
                   "cells": dict(CELLS)}
        with mock.patch.object(comm_suite, "_worker_arm",
                               return_value=(payload, "")):
            res = comm_suite._arm_collective_htccl_shm(
                comm_suite._RunCtx(job=mock.MagicMock(), card_count=3))
        self.assertEqual(res.status, "ok")
        self.assertTrue(any("all_gather" in n for n in res.notes))

    def test_gdr_crossover_is_absent_when_the_binary_is_not_built(self):
        # On this box (and any box that never built the handover binary out
        # of tree) the arm must be absent, never an error, and must name
        # both the missing binary and the env var that could point at it.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(comm_suite.GDR_CROSSOVER_BIN_ENV, None)
            res = comm_suite._arm_gdr_crossover(
                comm_suite._RunCtx(job=mock.MagicMock()))
        self.assertEqual(res.status, "absent")
        self.assertIn(comm_suite.GDR_CROSSOVER_BIN_ENV, res.absent_reason)
        self.assertIn("BUILD.md", res.absent_reason)
        self.assertIn("card window", res.absent_reason)

    def test_gdr_crossover_is_absent_pointing_at_a_configured_but_missing_path(self):
        with mock.patch.dict(os.environ,
                             {comm_suite.GDR_CROSSOVER_BIN_ENV: "/no/such/bin"}):
            res = comm_suite._arm_gdr_crossover(
                comm_suite._RunCtx(job=mock.MagicMock()))
        self.assertEqual(res.status, "absent")
        self.assertIn("/no/such/bin", res.absent_reason)

    def test_gdr_crossover_maps_the_ladder_and_finds_the_crossover_size(self):
        payload = {
            "pair": "5090<->3080",
            "sizes": {
                "8B": {"direct_us": 4.99, "staged_us": 6.6, "n": 2000},
                "4KiB": {"direct_us": 6.08, "staged_us": 5.31, "n": 2000},
                "64KiB": {"direct_us": 44.84, "staged_us": 15.50, "n": 2000},
                "1MiB": {"direct_us": 634.13, "staged_us": 185.57, "n": 2000},
            },
        }
        with mock.patch.object(comm_suite, "_gdr_crossover_bin",
                               return_value="/fake/gpurdma_04_bench"), \
             mock.patch.object(comm_suite, "_gdr_bench_run",
                               return_value=payload):
            res = comm_suite._arm_gdr_crossover(
                comm_suite._RunCtx(job=mock.MagicMock()))
        self.assertEqual(res.status, "ok")
        self.assertIn("direct/8B", res.cells)
        self.assertIn("staged/8B", res.cells)
        self.assertEqual(res.cells["direct/8B"]["median_us"], 4.99)
        # direct wins at 8B (4.99<6.6), loses starting at 4KiB (6.08>5.31)
        self.assertEqual(res.facts["crossover_at"], "4KiB")
        self.assertTrue(any("property of THIS rig" in n for n in res.notes))

    def test_gdr_crossover_is_an_error_not_an_absence_when_the_binary_fails(self):
        with mock.patch.object(comm_suite, "_gdr_crossover_bin",
                               return_value="/fake/gpurdma_04_bench"), \
             mock.patch.object(comm_suite, "_gdr_bench_run",
                               side_effect=RuntimeError("segfault")):
            res = comm_suite._arm_gdr_crossover(
                comm_suite._RunCtx(job=mock.MagicMock()))
        self.assertEqual(res.status, "error")
        self.assertIn("segfault", res.error)

    def test_gdr_crossover_is_registered_as_a_gpu_arm(self):
        spec = next(a for a in ARMS if a.id == "gdr_crossover")
        self.assertEqual(spec.kind, "gpu")
        self.assertIn("gdr_crossover", comm_suite.ARM_RUNNERS)
        self.assertIs(comm_suite.ARM_RUNNERS["gdr_crossover"],
                      comm_suite._arm_gdr_crossover)

    def test_card_probe_arm_reads_the_probe_json_keys_that_exist(self):
        profile = mock.MagicMock()
        profile.to_json.return_value = {
            "cards": [{"uuid": "GPU-a", "name": "NVIDIA GeForce RTX 5090"}],
            "pairs": [{"src_uuid": "GPU-a", "dst_uuid": "GPU-b",
                       "bandwidth_gbs": 4.44,
                       "transport": "host staging (pinned)"}],
        }
        with mock.patch("sglang.srt.rigmon.card_probe._run_probe_subprocess",
                        return_value=(profile, "/tmp/x.json")):
            res = comm_suite._arm_card_probe(
                comm_suite._RunCtx(job=mock.MagicMock()))
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.facts["pairs"]), 1)
        self.assertTrue(any("host staging" in n for n in res.notes))
        self.assertTrue(any("P2P" in n for n in res.notes))

    def test_card_probe_pairs_reach_the_digest_without_uuids(self):
        store = _store({"card_probe": lambda c: ArmResult(
            "card_probe", "ok", facts={
                "cards": [
                    {"uuid": "GPU-aaaa", "name": "NVIDIA GeForce RTX 5090"},
                    {"uuid": "GPU-bbbb", "name": "NVIDIA GeForce RTX 3080"}],
                "pairs": [{"src_uuid": "GPU-aaaa", "dst_uuid": "GPU-bbbb",
                           "bandwidth_gbs": 4.44,
                           "transport": "host staging (pinned)"}]})})
        # The GPU phase would otherwise refuse: on this rig the real card
        # locks are usually held. The window is what THAT test covers; here
        # the question is what the pair rows look like once it opens.
        with mock.patch.object(comm_suite._CardWindow, "acquire",
                               return_value=True):
            job = store.start()
        digest = rig_artifact.build_digest([to_sections(job)])
        ids = [r["id"] for r in digest["measurements"]]
        self.assertIn("pair/RTX 5090->RTX 3080/bandwidth", ids)
        self.assertFalse(any("GPU-" in i for i in ids))


class SectionsTest(unittest.TestCase):
    def _job(self):
        store = _store({
            "noise_floor": lambda c: ArmResult(
                "noise_floor", "ok",
                facts={"cell": "all_reduce/20KiB", "floor_pct": 3.8}),
            "collective_htccl_ucx": lambda c: ArmResult(
                "collective_htccl_ucx", "ok", cells=dict(CELLS),
                facts={"world": 2}),
            "byte_gate": lambda c: ArmResult(
                "byte_gate", "error",
                error="4 collectives did NOT match the reference at atol 0"),
        })
        return store.start()

    def test_cells_become_rows_with_unit_spread_and_context(self):
        sections = to_sections(self._job())
        rows = {m.id: m for m in sections.measurements}
        key = "comm/collective_htccl_ucx/all_reduce/20KiB"
        self.assertIn(key, rows)
        row = rows[key]
        self.assertEqual(row.value, 37.2)
        self.assertEqual(row.spread_pct, 37.6)
        self.assertEqual(row.n, 60)
        self.assertEqual(row.context["op"], "all_reduce")
        self.assertEqual(row.context["size_kib"], 20)
        self.assertEqual(row.context["world"], 2)

    def test_a_failed_arm_becomes_an_error_signature_and_an_absent_capability(self):
        sections = to_sections(self._job())
        self.assertTrue(any("byte_gate" in e.where for e in sections.errors))
        caps = {c.name: c for c in sections.capabilities}
        self.assertEqual(caps["comm/byte_gate"].provenance, "absent")

    def test_the_noise_floor_is_carried_into_the_shared_notes(self):
        sections = to_sections(self._job())
        self.assertTrue(any("3.8" in n for n in sections.notes))

    def test_a_full_run_produces_an_anonymous_digest(self):
        digest = rig_artifact.build_digest([to_sections(self._job())])
        rig_artifact.assert_anonymized(digest)
        self.assertTrue(digest["fingerprint"]["id"].startswith("rig-"))
        self.assertIn("comm_suite", digest["sources"])


class ProfileSourceTest(unittest.TestCase):
    PROBE = {
        "created": 1785000000.0,
        "driver": "595.58.03", "torch_version": "2.11.0", "cuda_version": "13.0",
        "cards": [
            {"uuid": "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
             "name": "NVIDIA GeForce RTX 5090", "cuda_index": 0,
             "total_mib": 32607, "gemm_bf16_tflops": 232.0,
             "gemm_fp8_tflops": 566.9, "membw_read_gbs": 1660.4,
             "h2d_gbs": 14.4, "d2h_gbs": 14.3},
            {"uuid": "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
             "name": "NVIDIA GeForce RTX 3080", "cuda_index": 1,
             "total_mib": 20470, "gemm_bf16_tflops": 59.0,
             "gemm_fp8_tflops": None,
             "fp8_note": "compute capability 8.6 has no fp8 tensor path",
             "membw_read_gbs": 700.0},
            {"uuid": "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4",
             "name": "NVIDIA GeForce RTX 3080", "cuda_index": 2,
             "total_mib": 20470, "membw_read_gbs": 690.0},
        ],
        "pairs": [{"src_uuid": "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
                   "dst_uuid": "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
                   "bandwidth_gbs": 4.44, "latency_us": 22.4,
                   "transport": "host staging (pinned)", "peer_access": False}],
        "caveats": [],
    }
    ROW = {
        "candidate": "auto", "model_path": "/spinning/models/Qwen3.6-27B-FP8",
        "tp_size": 3, "timestamp": 1785202635.0, "chosen_vector": "2,1,1",
        "decode_tok_s": 92.55, "prefill_tok_s": 1151.1, "ms_per_verify": 33.0,
        "max_total_num_tokens": 502528, "unbootable": "",
        "output_verdict": "coherent prose",
        "boot_log": "SECRET LOG /spinning/private/thing 10.10.10.2",
        "rank_compute_wait": [{"rank": 0, "compute_ms": 175.5,
                               "wait_ms": 1552.2}],
    }

    def _sections(self):
        with mock.patch.object(rig_profile_source, "_latest_card_probe",
                               return_value=self.PROBE), \
             mock.patch.object(rig_profile_source, "_split_probe_rows",
                               return_value=[self.ROW]):
            return rig_profile_source.to_sections()

    def test_uuids_are_re_keyed_to_model_and_ordinal(self):
        ids = {m.id for m in self._sections().measurements}
        self.assertIn("card/RTX 5090/membw_read_gbs", ids)
        self.assertIn("card/RTX 3080#0/membw_read_gbs", ids)
        self.assertIn("card/RTX 3080#1/membw_read_gbs", ids)
        self.assertFalse(any("GPU-" in i for i in ids))

    def test_model_paths_become_families(self):
        self.assertEqual(
            rig_profile_source.model_family("/spinning/x/Qwen3.6-27B-FP8"),
            "Qwen3.6 27B FP8")
        ids = {m.id for m in self._sections().measurements}
        self.assertIn("serving/Qwen3.6 27B FP8/auto/decode_tok_s", ids)
        self.assertFalse(any("/spinning" in i for i in ids))

    def test_the_boot_log_never_reaches_the_digest(self):
        digest = rig_artifact.build_digest([self._sections()])
        blob = json.dumps(digest)
        self.assertNotIn("SECRET LOG", blob)
        self.assertNotIn("10.10.10.2", blob)
        rig_artifact.assert_anonymized(digest)

    def test_missing_fp8_is_an_absent_capability_with_the_reason(self):
        caps = {c.name: c for c in self._sections().capabilities}
        self.assertEqual(caps["card/RTX 3080#0/fp8_gemm"].provenance, "absent")
        self.assertIn("8.6", caps["card/RTX 3080#0/fp8_gemm"].note)
        self.assertEqual(caps["card/RTX 5090/fp8_gemm"].provenance, "measured")

    def test_an_unbootable_candidate_is_a_finding_not_a_hole(self):
        row = dict(self.ROW, unbootable="OOM at 2.37 GiB before a single "
                                        "KV token", decode_tok_s=None)
        with mock.patch.object(rig_profile_source, "_latest_card_probe",
                               return_value=self.PROBE), \
             mock.patch.object(rig_profile_source, "_split_probe_rows",
                               return_value=[row]):
            s = rig_profile_source.to_sections()
        self.assertTrue(s.errors)
        caps = {c.name: c for c in s.capabilities}
        self.assertFalse(caps["topology/Qwen3.6 27B FP8/auto"].value)

    def test_available_reads_nothing_expensive_and_measures_nothing(self):
        with mock.patch.object(rig_profile_source, "_latest_card_probe",
                               return_value=None), \
             mock.patch.object(rig_profile_source, "_split_probe_rows",
                               return_value=[]):
            avail = rig_profile_source.available()
        self.assertFalse(avail["any"])
        self.assertFalse(avail["card_probe"]["present"])

    def test_both_sources_land_in_one_digest_with_one_fingerprint(self):
        store = _store()
        job = store.start()
        digest = rig_artifact.build_digest(
            [self._sections(), to_sections(job)])
        self.assertEqual(sorted(digest["sources"]),
                         ["comm_suite", "hardware_profile"])
        self.assertTrue(digest["fingerprint"]["id"].startswith("rig-"))
        rig_artifact.assert_anonymized(digest)


class WebuiEndpointTest(unittest.TestCase):
    """The share ENDPOINTS must not be a way around the preview rule."""

    def test_submit_refuses_without_a_previewed_report(self):
        from sglang.srt.planner import webui

        out = webui.share_rig_submit_payload({"confirmed": True,
                                              "token": "ghp_x" * 6})
        self.assertFalse(out["ok"])
        self.assertIn("preview first", out["error"])

    def test_submit_refuses_without_confirmation_and_posts_nothing(self):
        from sglang.srt.planner import webui

        calls = []

        def api(method, url, token, body=None, timeout=30.0):
            calls.append(method)
            return 200, {}

        with mock.patch.object(rig_artifact, "submit",
                               side_effect=rig_artifact.submit) as spy:
            out = webui.share_rig_submit_payload({
                "report": "text", "digest": {"fingerprint": {"id": "rig-a"}},
                "token": "ghp_" + "x" * 20, "confirmed": False})
        self.assertFalse(out["ok"])
        self.assertIn("did not confirm", out["error"])
        self.assertEqual(calls, [])
        self.assertTrue(spy.called)

    def test_preview_says_so_when_there_is_nothing_to_share(self):
        from sglang.srt.planner import webui

        with mock.patch.object(rig_profile_source, "to_sections",
                               side_effect=AssertionError("must not be called")):
            out = webui.share_rig_preview_payload({"sources": []})
        self.assertFalse(out["ok"])
        self.assertIn("nothing to share", out["error"])

    def test_token_endpoint_never_returns_the_token(self):
        from sglang.srt.planner import webui

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp},
                                 clear=False):
                secret = "ghp_" + "z" * 24
                out = webui.share_rig_token_payload({"action": "save",
                                                     "token": secret})
                self.assertTrue(out["token_stored"])
                self.assertNotIn(secret, json.dumps(out))
                out = webui.share_rig_token_payload({"action": "forget"})
                self.assertFalse(out["token_stored"])

    def test_the_arm_catalogue_endpoint_measures_nothing(self):
        from sglang.srt.planner import webui

        with mock.patch.object(comm_suite.JOBS, "start",
                               side_effect=AssertionError("must not start")):
            out = webui.commsuite_arms_payload()
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["arms"]), len(ARMS))


if __name__ == "__main__":
    unittest.main()
