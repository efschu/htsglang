"""CPU tests for the #232 split probe: contract, persistence, honesty lines.

No GPU and no boot. The measurement itself is stubbed at three seams the
module exposes for exactly this -- ``server_factory``, ``measure`` and the job
store's ``runner`` -- so what is tested here is the machinery around the
measurement: the lock, the reserve derivation, the store's guard, the table's
deltas and its refusal to invent them, and the job's single-flight.
"""

import json
import os
import tempfile
import threading
import time

from sglang.srt.planner import split_probe as sp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _row(candidate, prefill, decode, ms_verify, kv, ts=1000.0, **kw):
    return sp.SplitProbeResult(
        candidate=candidate,
        chosen_vector=kw.pop("chosen_vector", candidate),
        prefill_tok_s=prefill,
        decode_tok_s=decode,
        ms_per_verify=ms_verify,
        max_total_num_tokens=kv,
        timestamp=ts,
        **kw,
    )


class TestGpuLock(CustomTestCase):
    def test_a_second_acquire_is_refused_while_the_first_holds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lock")
            first = sp.gpu_lock(path, label="first")
            first.acquire()
            try:
                with self.assertRaises(sp.GpuLockBusy) as cm:
                    sp.gpu_lock(path, label="second").acquire()
                self.assertIn("split probe", str(cm.exception))
            finally:
                first.release()

    def test_release_lets_the_next_one_in(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lock")
            with sp.gpu_lock(path):
                pass
            self.assertFalse(os.path.exists(path))
            with sp.gpu_lock(path):
                self.assertTrue(os.path.isdir(path))

    def test_a_lock_whose_owner_is_gone_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lock")
            os.mkdir(path)
            # A pid that cannot exist: 0 is never a user process, and the owner
            # file is the only thing the reclaim path has to go on.
            with open(os.path.join(path, "owner.json"), "w") as f:
                json.dump({"pid": 2**22 - 1, "label": "dead", "at": 0}, f)
            lock = sp.gpu_lock(path)
            try:
                lock.acquire()
            except sp.GpuLockBusy:  # pragma: no cover - only if that pid lives
                self.skipTest("the sentinel pid happens to exist here")
            try:
                self.assertIsNotNone(lock.reclaimed_from)
            finally:
                lock.release()

    def test_the_lock_is_released_even_when_the_run_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lock")

            def boom(*a, **kw):
                raise RuntimeError("the boot died")

            with self.assertRaises(RuntimeError):
                sp.run_split_probe(
                    "/no/such/model",
                    candidate="auto",
                    lock_path=path,
                    store_path=os.path.join(d, "store.jsonl"),
                    server_factory=boom,
                    busy_fn=lambda: {},
                )
            self.assertFalse(os.path.exists(path), "a probe that died kept the cards")


class TestReserve(CustomTestCase):
    def test_the_optimizer_s_own_choice_is_not_bumped(self):
        reserve, note = sp.reserve_for_candidate(
            "auto", 3, (3000, 2700, 2700), model_path="/m"
        )
        self.assertEqual(reserve, [3000, 2700, 2700])
        self.assertIn("unbumped", note)

    def test_a_concentrated_candidate_bumps_only_the_rank_that_gained_mass(self):
        # The scratch grows with the rank's share; the stub is the real
        # function's shape (monotone in share) without a checkpoint.
        def scratch(model_path, tp, share):
            return 300.0 * share

        reserve, note = sp.reserve_for_candidate(
            "6,1,1", 3, (3000, 2700, 2700), model_path="/m", scratch_fn=scratch
        )
        # share 0.75 vs base 1/3 -> +300*(0.75-0.333) = +125 MiB on rank 0.
        self.assertEqual(reserve, [3125, 2700, 2700])
        self.assertIn("rank 0 +125 MiB", note)
        self.assertIn("#265", note)

    def test_a_checkpoint_without_gdn_layers_says_so_instead_of_guessing(self):
        reserve, note = sp.reserve_for_candidate(
            "6,1,1",
            3,
            (3000, 2700, 2700),
            model_path="/m",
            scratch_fn=lambda *a: None,
        )
        self.assertEqual(reserve, [3000, 2700, 2700])
        self.assertIn("no GDN layers", note)

    def test_a_malformed_candidate_is_rejected_by_name(self):
        with self.assertRaises(sp.SplitProbeRejected):
            sp.reserve_for_candidate("6,1", 3, (3000, 2700, 2700), model_path="/m")
        with self.assertRaises(sp.SplitProbeRejected):
            sp.reserve_for_candidate("six", 3, (3000, 2700, 2700), model_path="/m")


class TestLaunchCommand(CustomTestCase):
    def test_metrics_and_the_device_timer_are_never_optional(self):
        cmd = sp.launch_command("/m", "auto", (3000, 2700, 2700), 8899)
        self.assertIn("--enable-metrics", cmd)
        self.assertIn("--enable-metrics-for-all-schedulers", cmd)
        env = sp.launch_env(base={})
        self.assertEqual(env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"], "1")

    def test_auto_pins_no_vector_and_a_candidate_pins_exactly_its_own(self):
        self.assertNotIn("--rank-mlp-ratio", sp.launch_command("/m", "auto", [1], 1))
        cmd = sp.launch_command("/m", "6,1,1", (3000, 2700, 2700), 8899)
        self.assertEqual(cmd[cmd.index("--rank-mlp-ratio") + 1], "6,1,1")

    def test_an_inherited_vector_override_cannot_win_over_the_pinned_one(self):
        env = sp.launch_env(base={"SGLANG_UNEVEN_MLP_VECTOR": "9,1,1"})
        self.assertNotIn("SGLANG_UNEVEN_MLP_VECTOR", env)


class TestPrefillLineParsing(CustomTestCase):
    LOG = """
[2026-07-28 00:33:36 TP0] Prefill rank batch, #new-token: 2048, #cached-token: 0, #chunks: 1, gpu-ms: 1730.0 (compute 175.0, wait 1555.0)
[2026-07-28 00:33:36 TP1] Prefill rank batch, #new-token: 2048, #cached-token: 0, #chunks: 1, gpu-ms: 1729.0 (compute 541.0, wait 1188.0)
[2026-07-28 00:33:37 TP0] Prefill rank batch, #new-token: 2048, #cached-token: 0, #chunks: 1, gpu-ms: 1732.0 (compute 177.0, wait 1555.0)
[2026-07-28 00:33:38 TP0] Prefill rank batch, #new-token: 320, #cached-token: 0, #chunks: 1, gpu-ms: 200.0 (compute 20.0, wait 180.0)
[2026-07-28 00:33:39 TP0] Prefill rank batch, #new-token: 2048, #cached-token: 0, #chunks: 3, gpu-ms: 5000.0 (compute 500.0, wait 4500.0)
[2026-07-28 00:33:40 TP0] Prefill batch, #new-token: 2048, #cached-token: 0
"""

    def test_only_steady_single_chunk_lines_count(self):
        rows = sp.parse_rank_compute_wait(self.LOG)
        by_rank = {r["rank"]: r for r in rows}
        self.assertEqual(sorted(by_rank), [0, 1])
        # The 320-token tail and the 3-chunk fold are both excluded.
        self.assertEqual(by_rank[0]["chunks"], 2)
        self.assertEqual(by_rank[0]["new_token"], 2048)
        self.assertAlmostEqual(by_rank[0]["compute_ms"], 176.0, places=1)
        self.assertAlmostEqual(by_rank[1]["wait_ms"], 1188.0, places=1)

    def test_a_log_without_the_split_yields_nothing_rather_than_zeros(self):
        self.assertEqual(sp.parse_rank_compute_wait("no such line here"), [])


class TestOutputJudgement(CustomTestCase):
    def test_a_repetition_loop_is_named_as_one(self):
        v = sp.judge_output("the cat " * 60)
        self.assertIn("degenerate", v)

    def test_prose_is_accepted(self):
        words = " ".join(f"word{i}" for i in range(80))
        self.assertIn("coherent", sp.judge_output(words))

    def test_no_output_is_not_silently_fine(self):
        self.assertIn("no output", sp.judge_output(""))


class TestStore(CustomTestCase):
    def test_a_row_that_claims_measurement_must_carry_both_figures(self):
        store = sp.SplitProbeStore()
        half = sp.SplitProbeResult(candidate="6,1,1", prefill_tok_s=1200.0)
        with self.assertRaises(sp.SplitProbeRejected) as cm:
            store.ingest(half)
        self.assertIn("half-measured", str(cm.exception))

    def test_an_unbootable_row_is_a_result_and_is_kept(self):
        store = sp.SplitProbeStore()
        store.ingest(
            sp.SplitProbeResult(candidate="16,1,2", unbootable="OOM at 4500 MiB")
        )
        self.assertEqual(len(store), 1)

    def test_a_round_trip_through_the_file_keeps_every_field(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            entry = _row("6,1,1", 1244.1, 80.9, 37.307, 261632)
            entry.rank_compute_wait = [
                {
                    "rank": 0,
                    "chunks": 9,
                    "gpu_ms": 1.0,
                    "compute_ms": 2.0,
                    "wait_ms": 3.0,
                }
            ]
            sp.SplitProbeStore().append_to_file(path, entry)
            back = sp.SplitProbeStore.load(path).entries()[0]
            self.assertEqual(back.max_total_num_tokens, 261632)
            self.assertEqual(back.rank_compute_wait[0]["wait_ms"], 3.0)

    def test_a_hand_edited_unmeasured_row_does_not_survive_the_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            with open(path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "candidate": "8,1,1",
                            "provenance": "measured",
                            "version": sp.SPLIT_PROBE_VERSION,
                            "decode_tok_s": 999.0,
                        }
                    )
                    + "\n"
                )
            self.assertEqual(len(sp.SplitProbeStore.load(path)), 0)

    def test_a_row_of_another_version_is_ignored_rather_than_reinterpreted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            row = _row("6,1,1", 1.0, 2.0, 3.0, 4).to_json()
            row["version"] = sp.SPLIT_PROBE_VERSION + 1
            with open(path, "w") as f:
                f.write(json.dumps(row) + "\n")
            self.assertEqual(len(sp.SplitProbeStore.load(path)), 0)

    def test_a_re_measurement_supersedes_the_older_row(self):
        store = sp.SplitProbeStore(
            [
                _row("auto", 1.0, 1.0, 1.0, 1, ts=100.0),
                _row("auto", 2.0, 2.0, 2.0, 2, ts=200.0),
            ]
        )
        self.assertEqual(store.latest()["auto"].prefill_tok_s, 2.0)


class TestImport264(CustomTestCase):
    def test_the_two_hand_measured_rows_import_once(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            self.assertEqual(len(sp.import_264_rows(path)), 2)
            self.assertEqual(len(sp.import_264_rows(path)), 0)
            store = sp.SplitProbeStore.load(path)
            self.assertEqual(len(store), 2)
            for e in store.entries():
                self.assertEqual(e.provenance, sp.IMPORTED)
                self.assertIn("#264", e.source)

    def test_the_imported_rows_reproduce_the_deltas_264_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            sp.import_264_rows(path)
            table = sp.tipping_point_table(path)
            row = [r for r in table["rows"] if r["candidate"] == "6,1,1"][0]
            self.assertAlmostEqual(row["delta"]["prefill_pct"], 8.2, places=1)
            self.assertAlmostEqual(row["delta"]["decode_pct"], -13.7, places=1)
            self.assertAlmostEqual(row["delta"]["max_kv_pct"], -47.9, places=1)


class TestTable(CustomTestCase):
    def test_every_ladder_candidate_is_a_row_measured_or_not(self):
        table = sp.tipping_point_table(store=sp.SplitProbeStore())
        self.assertEqual(len(table["rows"]), len(sp.LADDER))
        for r in table["rows"]:
            self.assertFalse(r["measured"])
            self.assertIn("not measured", r["missing_reason"])
        self.assertEqual(table["measured_count"], 0)

    def test_no_baseline_means_no_deltas_rather_than_deltas_against_nothing(self):
        store = sp.SplitProbeStore([_row("6,1,1", 1244.1, 80.9, 37.3, 261632)])
        table = sp.tipping_point_table(store=store)
        row = [r for r in table["rows"] if r["candidate"] == "6,1,1"][0]
        self.assertNotIn("delta", row)
        self.assertIn("no baseline yet", table["summary"])

    def test_an_unbootable_candidate_carries_its_reason_and_no_deltas(self):
        store = sp.SplitProbeStore(
            [
                _row("auto", 1149.6, 93.71, 32.6, 502528),
                sp.SplitProbeResult(
                    candidate="16,1,2", unbootable="OOM after a raise", timestamp=1.0
                ),
            ]
        )
        row = [
            r
            for r in sp.tipping_point_table(store=store)["rows"]
            if r["candidate"] == "16,1,2"
        ][0]
        self.assertTrue(row["measured"])
        self.assertIn("OOM", row["unbootable"])
        self.assertNotIn("delta", row)

    def test_the_cost_of_measuring_is_stated_in_the_reader_s_terms(self):
        self.assertIn(
            "6-8 minutes",
            sp.tipping_point_table(store=sp.SplitProbeStore())["cost_note"],
        )


class _FakeServer:
    """A server that boots instantly, logs what was asked for, and stops."""

    log = ""
    oom_until_reserve = 0
    starts: list = []

    def __init__(self, cmd, env, log_path, port):
        self.cmd, self.log_path, self.port = cmd, log_path, port
        self.stopped = False
        type(self).starts.append(list(cmd))

    def log_text(self):
        return type(self).log

    def start(self):
        pass

    def wait_ready(self, timeout=None):
        idx = self.cmd.index("--rank-auto-reserve-mib") + 1
        rank0 = int(self.cmd[idx].split(",")[0])
        if rank0 < type(self).oom_until_reserve:
            raise sp._BootOOM("CUDA out of memory")

    def stop(self):
        self.stopped = True


def _fake_measure(port, prefill_tokens=0, decode_seconds=0):
    return {
        "max_total_num_tokens": 502528,
        "prefill_tokens": prefill_tokens,
        "prefill_cached_tokens": 0,
        "prefill_wall_s": 17.4,
        "prefill_tok_s": 1149.6,
        "decode_tokens": 1500,
        "decode_wall_s": 16.0,
        "decode_tok_s": 93.71,
        "accept_length": 3.05,
        "verify_ct": 491,
        "ms_per_verify": 32.6,
        "output_head": "Tensor parallelism splits ...",
        "output_verdict": "coherent prose",
    }


class TestRunOrchestration(CustomTestCase):
    def setUp(self):
        _FakeServer.log = TestPrefillLineParsing.LOG + "\nCHOSEN MLP vector: 2,1,1\n"
        _FakeServer.oom_until_reserve = 0
        _FakeServer.starts = []

    def _run(self, tmp, candidate="auto", **kw):
        return sp.run_split_probe(
            "/m",
            candidate=candidate,
            lock_path=os.path.join(tmp, "lock"),
            store_path=os.path.join(tmp, "s.jsonl"),
            log_dir=tmp,
            server_factory=_FakeServer,
            measure=_fake_measure,
            busy_fn=lambda: {},
            **kw,
        )

    def test_a_card_someone_else_is_using_stops_the_probe_before_it_boots(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(sp.GpuLockBusy) as cm:
                sp.run_split_probe(
                    "/m",
                    lock_path=os.path.join(d, "lock"),
                    store_path=os.path.join(d, "s.jsonl"),
                    log_dir=d,
                    server_factory=_FakeServer,
                    measure=_fake_measure,
                    busy_fn=lambda: {"GPU-abc": "4711"},
                )
            self.assertIn("Nothing was started", str(cm.exception))
            self.assertEqual(_FakeServer.starts, [])

    def test_a_run_records_the_chosen_vector_and_the_per_rank_split(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []
            r = self._run(d, progress=lambda *a: seen.append(a))
            self.assertEqual(r.chosen_vector, "2,1,1")
            self.assertEqual(len(r.rank_compute_wait), 2)
            self.assertEqual(r.provenance, sp.MEASURED)
            self.assertTrue(seen, "the run reported no progress at all")
            self.assertEqual(
                len(sp.SplitProbeStore.load(os.path.join(d, "s.jsonl"))), 1
            )

    def test_an_oom_is_retried_once_at_a_raised_reserve_and_both_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            _FakeServer.oom_until_reserve = 4500
            r = self._run(d, candidate="6,1,1", reserve_mib=[3000, 2700, 2700])
            self.assertEqual(r.unbootable, "")
            self.assertEqual(r.reserve_retried_from[0], 3000)
            self.assertEqual(r.reserve_mib[0], 4500)
            self.assertIn("first boot OOMed", r.reserve_note)
            self.assertEqual(len(_FakeServer.starts), 2)

    def test_a_candidate_that_ooms_after_the_raise_is_recorded_unbootable(self):
        with tempfile.TemporaryDirectory() as d:
            _FakeServer.oom_until_reserve = 99999
            r = self._run(d, candidate="16,1,2", reserve_mib=[3000, 2700, 2700])
            self.assertIn("does not fit", r.unbootable)
            self.assertEqual(len(_FakeServer.starts), 2)
            # A finding, not a hole: it survives the store's guard.
            self.assertEqual(
                len(sp.SplitProbeStore.load(os.path.join(d, "s.jsonl"))), 1
            )


class TestJobStore(CustomTestCase):
    def _store(self, runner):
        store = sp.SplitProbeJobStore()
        store.synchronous = True
        store.runner = runner
        return store

    def test_a_finished_job_carries_its_row(self):
        row = _row("auto", 1149.6, 93.71, 32.6, 502528)
        store = self._store(lambda req: row)
        job = store.start({"model_path": "/m", "candidate": "auto"})
        self.assertEqual(job.state, sp.OK)
        self.assertEqual(job.result.max_total_num_tokens, 502528)
        self.assertIsNone(store.active())

    def test_a_failure_carries_an_error_and_a_remedy(self):
        def boom(req):
            raise RuntimeError("the boot died")

        job = self._store(boom).start({"model_path": "/m"})
        self.assertEqual(job.state, sp.ERROR)
        self.assertIn("the boot died", job.error)
        self.assertTrue(job.remedy)

    def test_a_busy_lock_is_reported_without_advising_a_kill(self):
        def busy(req):
            raise sp.GpuLockBusy("the cards are held")

        job = self._store(busy).start({"model_path": "/m"})
        self.assertEqual(job.state, sp.ERROR)
        self.assertIn("Never kill a process you did not start", job.remedy)

    def test_a_second_start_joins_the_running_one(self):
        gate = threading.Event()
        calls = []

        def slow(req):
            calls.append(req)
            gate.wait(5)
            return _row("auto", 1.0, 1.0, 1.0, 1)

        store = sp.SplitProbeJobStore()
        store.runner = slow
        first = store.start({"model_path": "/m", "candidate": "auto"})
        for _ in range(200):
            if calls:
                break
            time.sleep(0.01)
        second = store.start({"model_path": "/m", "candidate": "6,1,1"})
        self.assertEqual(first.job_id, second.job_id)
        gate.set()
        for _ in range(500):
            if store.active() is None:
                break
            time.sleep(0.01)
        self.assertEqual(len(calls), 1)

    def test_an_unknown_job_id_is_not_invented(self):
        self.assertIsNone(sp.SplitProbeJobStore().get("nope"))
