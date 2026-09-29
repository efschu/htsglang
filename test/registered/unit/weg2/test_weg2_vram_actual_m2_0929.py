"""VRAM-Vertrag M2 (29.09.): the rank writes its VRAM actual per PID and
category into its RankState (block ``vram``), not into a log line.

User 28.09. ~20:45Z: "über logfiles? ... das muss man professionell ordentlich
machen"; 29.09.: "Aktuell ist die VRAM-Belegung Frickelarbeit ... Dabei haben
wir für alles Sonden und Messwerte". Pinned here
(/spinning/gpu-arb/docs/VRAM-VERTRAG-0929.md §3.4, §3.7, §4 M2):

  (a) attribution on fake NVML/torch/tags: ``sum(actual) == nvml_pid_mib``
      (within the ±64 MiB reader tolerance, exact by construction), and a
      remainder beyond 256 MiB is a finding that NAMES its parts;
  (b) the schema-2 reader refuses a foreign record version by name; a
      foreign ``vram`` block costs only the block (dropped by name as
      VRAM-ACTUAL-BLOCK-REFUSED), never the record or the W7/W10 gate;
  (c) every mark reaches the block through its real call edge
      (flight_recorder.mark, the WEG2-VRAM-PEAK window end, the flip leg,
      the attach, the scheduler bind) -- a mutant without the edge dies;
  (d) no write per decode round (count test);
  (e) IPC Phase-2 rule: nothing here reads log text as a control value.

No GPU, no NVML, no CUDA.
"""

import ast
import inspect
import json
import os
import re
import tempfile
import unittest

import pytest

try:
    from sglang.srt.environ import envs
    from sglang.srt.mem_ledger import flight_recorder as fr
    from sglang.srt.model_executor import vram_peak_window as vpw
    from sglang.srt.weg2 import rank_state as rs
    from sglang.srt.weg2 import vram_actual as va
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

M = 1 << 20
ME = 4242
PARTNER = 4343
FOREIGN = 9999


def _reading(**kw):
    base = dict(nvml_pid=0, reserved=0, allocated=0)
    base.update(kw)
    return va.Reading(**base)


def _d_rank_state(tp=0):
    return rs.build_rank_state(
        group="D", tp_rank=tp, tp_size=3, pp_rank=0, pp_size=1, form_a_worker=tp > 0,
        canonical_on=True, canonical_kv_built=True, canonical_blob_built=tp == 0,
        has_mamba_pool=True, page_size=64, owner_ctx=None, seq=1,
    )


class _Fakes:
    """A rank's live readings, scripted: torch (reserved, allocated), NVML
    per PID, saver tags (+ paused), structural providers."""

    def __init__(self):
        self.reserved = 20000 * M
        self.allocated = 19000 * M
        self.procs = {ME: 21000 * M, PARTNER: 1200 * M, FOREIGN: 300 * M}
        self.tags = {"weights": 1000 * M, "weights_0": 9000 * M, "kv_cache": 6000 * M,
                     "cuda_graph": 500 * M, "weights_draft": 0}
        self.paused = frozenset()
        self.prov = {"experts_resident": 3000 * M, "experts_lru": 1000 * M, "state_pools": 800 * M,
                     "kv_pool": 0, "kv_arena_backed": 0, "target_params": 0, "draft_params": 0}
        self.nvml_calls = 0

    def install(self):
        va._REC.fake_torch = lambda: (self.reserved, self.allocated)

        def procs():
            self.nvml_calls += 1
            return dict(self.procs)

        va._REC.fake_nvml = procs
        va._REC.fake_tags = lambda: (dict(self.tags), self.paused)
        va._REC.fake_providers = lambda: dict(self.prov)
        va._REC.pid = ME
        va._REC.pid_class = {PARTNER: "partner", FOREIGN: "foreign"}


class _Base(unittest.TestCase):
    def setUp(self):
        self._ov = envs.SGLANG_WEG2_VRAM_ACTUAL.override(True)
        self._ov.__enter__()
        va.reset()
        self.f = _Fakes()
        self.f.install()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self._group = os.environ.get("SGLANG_WEG2_GROUP")
        os.environ["SGLANG_WEG2_GROUP"] = "D"

    def tearDown(self):
        va.reset()
        self._ov.__exit__(None, None, None)
        self.tmp.cleanup()
        if self._group is None:
            os.environ.pop("SGLANG_WEG2_GROUP", None)
        else:
            os.environ["SGLANG_WEG2_GROUP"] = self._group

    def _record(self):
        states, bad = rs.read_group_states(self.dir)
        self.assertEqual(bad, [])
        self.assertEqual(len(states), 1)
        return states[0]


# --------------------------------------------------------------------------
# (a) attribution
# --------------------------------------------------------------------------


class TestAttribution(unittest.TestCase):
    def _check_invariant(self, actual, nvml_mib):
        self.assertLessEqual(abs(sum(actual.values()) - nvml_mib), va.INVARIANT_TOL_MIB)
        self.assertEqual(sum(actual.values()), nvml_mib)  # exact by construction
        self.assertEqual(set(actual), set(va.CATEGORIES) | {va.UNATTRIBUTED})

    def test_every_source_lands_in_its_category(self):
        # D TP0 awake: weights tags (holding the expert rows), kv tag holding
        # the mamba pool, graph tag, untagged workspaces, CUDA context
        r = _reading(
            nvml_pid=21000 * M, reserved=20000 * M, allocated=19000 * M,
            tags={"weights": 1000 * M, "weights_0": 9000 * M, "kv_cache": 6000 * M,
                  "cuda_graph": 500 * M},
            experts_resident=3000 * M, experts_lru=1000 * M, state_pools=800 * M,
            ctx_baseline=900 * M,
        )
        actual, parts, src, findings = va.attribute(r)
        self._check_invariant(actual, 21000)
        self.assertEqual(actual["weights"], 6000)            # 10000 tagged - 4000 expert rows
        self.assertEqual(actual["experts_resident"], 3000)
        self.assertEqual(actual["experts_lru"], 1000)
        self.assertEqual(actual["kv"], 5200)                 # 6000 tag - 800 state pools
        self.assertEqual(actual["state_pools"], 800)
        self.assertEqual(actual["graphs"], 500)
        self.assertEqual(actual["allocator_slack"], 1000)
        self.assertEqual(actual["cuda_ctx"], 900)
        # untagged allocated 19000-16500 = 2500, non-torch 21000-20000 = 1000 (ctx 900 + 100)
        self.assertEqual(parts, {"torch_untagged_allocated": 2500, "non_torch_growth": 100})
        self.assertEqual(actual[va.UNATTRIBUTED], 2600)
        (finding,) = findings
        self.assertIn("unattributed 2600 MiB beyond 256", finding)
        self.assertIn("torch_untagged_allocated=2500", finding)
        self.assertIn("non_torch_growth=100", finding)
        self.assertIn("carved out of weights", src["experts_resident"])

    def test_remainder_under_256_is_quiet_above_is_named(self):
        quiet = _reading(nvml_pid=10800 * M, reserved=10200 * M, allocated=10100 * M,
                         tags={"weights": 10000 * M}, ctx_baseline=600 * M)
        actual, parts, _src, findings = va.attribute(quiet)
        self.assertEqual(actual[va.UNATTRIBUTED], 100)
        self.assertEqual(findings, [])
        loud = _reading(nvml_pid=11100 * M, reserved=10200 * M, allocated=10100 * M,
                        tags={"weights": 10000 * M}, ctx_baseline=600 * M)
        actual, parts, _src, findings = va.attribute(loud)
        self.assertEqual(actual[va.UNATTRIBUTED], 400)
        self.assertEqual(parts["non_torch_growth"], 300)
        self.assertEqual(len(findings), 1)
        self.assertIn("non_torch_growth=300", findings[0])

    def test_z30_d_tp0_sleep_negative_remainder_is_named_unbacked(self):
        """Measured WEG2-DC-BREAKDOWN (boot ...09290654, D TP0 release):
        nvml_proc 23548 = tms_resident 21552 + torch_untagged 2400 + other -404;
        kv_cache 2630 + cuda_graph 498 paused; reserved 27080 allocated 26387."""
        tags = {"weights": 1258 * M, "weights_0": 20294 * M, "kv_cache": 2630 * M, "cuda_graph": 498 * M}
        r = _reading(nvml_pid=23548 * M, reserved=27080 * M, allocated=26387 * M, tags=tags,
                     paused=frozenset({"kv_cache", "cuda_graph"}), ctx_baseline=500 * M)
        actual, parts, _src, findings = va.attribute(r)
        self._check_invariant(actual, 23548)
        self.assertEqual(actual["kv"], 0)
        self.assertEqual(actual["graphs"], 0)
        self.assertEqual(actual["weights"], 21552)
        self.assertEqual(actual["cuda_ctx"], 0)          # non-torch is negative
        self.assertEqual(parts["unbacked_reservation"], -404)
        self.assertEqual(parts["torch_untagged_allocated"], 26387 - 24680)
        self.assertTrue(findings and "unbacked_reservation=-404" in findings[0])

    def test_band_tags_carry_the_experts(self):
        r = _reading(nvml_pid=5000 * M, reserved=4600 * M, allocated=4600 * M,
                     tags={"weights_3": 1000 * M, "weights_3_e0": 3000 * M, "weights_3_e1": 600 * M},
                     experts_resident=3000 * M, experts_lru=600 * M, ctx_baseline=400 * M)
        actual, _parts, src, _f = va.attribute(r)
        self._check_invariant(actual, 5000)
        self.assertEqual((actual["weights"], actual["experts_resident"], actual["experts_lru"]), (1000, 3000, 600))
        self.assertIn("band tags", src["experts_resident"])

    def test_untagged_drafter_and_unknown_tag_are_named(self):
        r = _reading(nvml_pid=4000 * M, reserved=3500 * M, allocated=3400 * M,
                     tags={"weights": 1000 * M, "surprise_tag": 200 * M},
                     draft_params=1300 * M, ctx_baseline=500 * M)
        actual, parts, src, _f = va.attribute(r)
        self._check_invariant(actual, 4000)
        self.assertEqual(actual["draft"], 1300)
        self.assertIn("untagged", src["draft"])
        self.assertEqual(parts["tag:surprise_tag"], 200)

    def test_without_nvml_there_is_no_remainder_and_it_says_so(self):
        actual, _parts, _src, findings = va.attribute(_reading(nvml_pid=None, reserved=M, allocated=M))
        self.assertEqual(actual[va.UNATTRIBUTED], 0)
        self.assertIn("nvml_pid unreadable", findings[0])


# --------------------------------------------------------------------------
# (b) schema
# --------------------------------------------------------------------------


class TestSchema(_Base):
    def _block(self):
        return va.on_mark("weights_loaded")

    def test_rank_state_2_roundtrips_with_vram(self):
        block = self._block()
        s = va.attach(_d_rank_state(), self.dir)
        self.assertEqual(s.schema, 2)
        back = rs.RankState.from_json(s.to_json())
        self.assertEqual(back, s)
        vb = va.VramBlock.from_dict(back.vram)
        self.assertEqual(vb.nvml_pid_mib, 21000)
        self.assertEqual(vb.partner_mib, 1200)
        self.assertEqual(vb.foreign_mib, 300)
        self.assertNotIn("foreign", vb.actual)
        self.assertIsNone(vb.plan_id)
        self.assertEqual(vb.contract, {})
        self.assertIsNotNone(block)

    def test_schema_1_reads_without_vram_and_refuses_it_with(self):
        d = json.loads(_d_rank_state().to_json())
        d["schema"] = 1
        d.pop("vram")
        self.assertEqual(rs.RankState.from_json(json.dumps(d)).schema, 1)
        d["vram"] = None
        with self.assertRaisesRegex(rs.RankStateSchemaError, r"schema 1 has no field\(s\) \['vram'\]"):
            rs.RankState.from_json(json.dumps(d))

    def test_foreign_version_is_refused_by_name(self):
        d = json.loads(_d_rank_state().to_json())
        d["schema"] = 3
        with self.assertRaisesRegex(rs.RankStateSchemaError, r"RankState schema 3, this reader knows \[1, 2\]"):
            rs.RankState.from_json(json.dumps(d))

    def test_a_foreign_block_is_dropped_by_name_the_record_stands(self):
        good = va.attach(_d_rank_state(), self.dir)
        cases = [
            (lambda v: v.update(surprise=1), r"has no field\(s\) \['surprise'\]"),
            (lambda v: v.update(schema="weg2.rank_vram/2"), r"schema 'weg2.rank_vram/2'"),
            (lambda v: v["actual"].update(foreign=5), r"unknown categor\(y/ies\) \['foreign'\]"),
            (lambda v: v["actual"].pop("kv"), r"actual lacks \['kv'\]"),
            (lambda v: v["marks"].update(x={"cost_ms": 1}), r"marks\['x'\] has no field\(s\) \['cost_ms'\]"),
        ]
        for mutate, pattern in cases:
            d = json.loads(good.to_json())
            mutate(d["vram"])
            back = rs.RankState.from_json(json.dumps(d))
            self.assertIsNone(back.vram)
            self.assertRegex(back.vram_refused, pattern)
            # the W7/W10 facts are untouched
            self.assertEqual((back.role, back.kv_page_active, back.gdn_blob_active),
                             (good.role, good.kv_page_active, good.gdn_blob_active))

    def test_broken_block_keeps_the_gate_green_and_is_a_named_finding(self):
        """Coordinator 29.09.: the block is display and record, not a control
        value -- a broken one may neither make the record unreadable nor stop
        W7/W10. The finding lands in the launcher log and in events.jsonl."""
        from sglang.srt.weg2 import launcher
        from sglang.srt.weg2 import state_file as sf

        root = os.path.join(self.dir, "state")
        os.makedirs(root)
        d = sf.init(root, "t1-boot-20260929T000000Z-0001", "boot", {})
        sf.transition(d, "launching")
        spec = launcher.GroupSpec("D", 0, ["true"], os.path.join(self.dir, "boot.D.log"), {})
        with envs.WEG2_STATE_DIR.override(d):
            sd = launcher.rank_state_dir_for(spec)
            for tp, worker, rows in ((0, False, (64, 0, 40)), (1, True, (64, 40, 52)), (2, True, (64, 52, 64))):
                st = rs.build_rank_state(
                    group="D", tp_rank=tp, tp_size=3, pp_rank=0, pp_size=1, form_a_worker=worker,
                    canonical_on=True, canonical_kv_built=True, canonical_blob_built=not worker,
                    has_mamba_pool=True, page_size=64, owner_ctx=rows, seq=1)
                rs.write_rank_state(va.attach(st, sd), sd)
            path = os.path.join(sd, "D.tp1pp0.json")
            with open(path) as f:
                rec = json.load(f)
            rec["vram"]["surprise"] = 1
            with open(path, "w") as f:
                f.write(json.dumps(rec))
            lines = []
            launcher.canonical_state_gate(spec, 3, lines.append)  # no Weg2LaunchRefused
            st = sf.read(d)
        self.assertEqual(st["invariants"]["W7_W10_CanonicalWindow"]["D"]["verdict"], "pass")
        (finding,) = [l for l in lines if l.startswith("VRAM-ACTUAL-BLOCK-REFUSED group D tp1pp0: ")]
        self.assertIn("has no field(s) ['surprise']", finding)
        (ev,) = [e for e in sf.events(d) if e["type"] == "vram_actual_block_refused"]
        self.assertEqual((ev["data"]["rank"], ev["data"]["code"]), ("tp1pp0", "VRAM_ACTUAL_BLOCK_REFUSED"))
        shown = [l for l in lines if l.startswith("VRAM-ACTUAL group D ")]
        self.assertEqual(len(shown), 2)  # tp0 and tp2 still displayed

    def test_switch_off_leaves_the_record_as_before(self):
        va.reset()
        with envs.SGLANG_WEG2_VRAM_ACTUAL.override(False):
            s = va.attach(_d_rank_state(), self.dir)
            self.assertIsNone(s.vram)
            self.assertIsNone(va.on_mark("weights_loaded"))
            self.assertFalse(va.on_window("round", peak_alloc=M, peak_reserved=M, start_alloc=0))


# --------------------------------------------------------------------------
# (c) the call edges, each at the edge itself
# --------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self):
        self.a = self.r = self.pa = self.pr = 19000 * M

    def alloc_free(self, n):
        self.pa = max(self.pa, self.a + n)
        self.pr = max(self.pr, self.r + n)
        self.r = max(self.r, self.a + n)

    def memory_stats(self):
        return {"allocated_bytes.all.peak": self.pa, "reserved_bytes.all.peak": self.pr,
                "allocated_bytes.all.current": self.a, "reserved_bytes.all.current": self.r}

    def reset_peak_memory_stats(self):
        self.pa, self.pr = self.a, self.r

    def mem_get_info(self):
        return 4000 * M, 32607 * M

    def is_current_stream_capturing(self):
        return False


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_extend(self):
        return self.kind == "extend"

    def is_decode(self):
        return self.kind == "decode"

    def is_target_verify(self):
        return False


class _FB:
    def __init__(self, kind, bs=1, rows=1):
        self.forward_mode = _Mode(kind)
        self.batch_size = bs
        self.input_ids = type("I", (), {"shape": (rows,)})()


class _Runner:
    tp_rank = 0
    is_draft_model_runner = False


class TestCallEdges(_Base):
    def setUp(self):
        super().setUp()
        vpw.reset_since_pools()
        self.cuda = _FakeCuda()
        rs.write_rank_state(va.attach(_d_rank_state(), self.dir), self.dir)

    def test_flight_recorder_mark_forwards_every_boot_post(self):
        self.assertIsNone(os.environ.get(fr.DIR_ENV))  # the flight recorder itself is unarmed
        for phase in sorted(fr.VRAM_ACTUAL_PHASES):
            before = self._record().vram["seq"]
            fr.mark(phase, rank=0)
            rec = self._record()
            self.assertEqual(rec.vram["mark"], phase)
            self.assertEqual(rec.vram["seq"], before + 1)
            self.assertIn(phase, rec.vram["marks"])
        fr.mark("nccl_init_begin", rank=0)  # not a VRAM post: no write
        self.assertNotIn("nccl_init_begin", self._record().vram["marks"])

    def test_pre_weight_load_freezes_the_context_baseline(self):
        self.f.reserved, self.f.procs[ME] = 10 * M, 610 * M
        fr.mark("pre_weight_load", rank=0)
        self.f.reserved, self.f.allocated, self.f.procs[ME] = 20000 * M, 19000 * M, 21000 * M
        fr.mark("weights_loaded", rank=0)
        self.assertEqual(self._record().vram["actual"]["cuda_ctx"], 600)

    def test_the_serving_tree_calls_every_vram_post(self):
        """The producer side of the edge: the boot posts are LITERAL calls of
        flight_recorder.mark in the runner and the scheduler process."""
        from sglang.srt.managers import scheduler
        from sglang.srt.model_executor import model_runner

        called = set()
        for mod in (model_runner, scheduler):
            for node in ast.walk(ast.parse(inspect.getsource(mod))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "mark" and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    called.add(node.args[0].value)
        self.assertEqual(fr.VRAM_ACTUAL_PHASES - called, set())

    def test_scheduler_binds_before_boot_complete(self):
        from sglang.srt.managers import scheduler

        src = inspect.getsource(scheduler.run_scheduler_process)
        bind = src.find("bind_scheduler(scheduler)")
        done = src.find('flight_recorder.mark("boot_complete"')
        self.assertGreater(bind, 0)
        self.assertGreater(done, bind)

    def test_window_end_reaches_the_block_with_the_d_state_key(self):
        for bs in (1, 3, 2):
            vpw.on_forward_end(_Runner(), _FB("decode", bs=bs), self.cuda)
        self.cuda.alloc_free(700 * M)
        vpw._close(self.cuda, 0, "round")
        peaks = self._record().vram["peak_by_state"]
        self.assertIn("n=3,S-,extend=0", peaks)
        self.assertEqual(peaks["n=3,S-,extend=0"]["allocated_peak"], 19700)
        self.assertEqual(peaks["n=3,S-,extend=0"]["nvml_pid"], 21000)

    def test_p_stau_counts_kv_holders_not_the_waiting_queue(self):
        """Planner seat 29.09.: stau = requests holding KV on P in this phase
        (being prefilled + prefilled for the flip); the waiting queue holds no
        page. Old rule len(waiting)+len(running) put 2 KV holders on stau=4."""
        os.environ["SGLANG_WEG2_GROUP"] = "P"

        class _Req:
            def __init__(self, rid):
                self.rid = rid

        class _Sched:
            waiting_queue = [_Req("w1"), _Req("w2"), _Req("w3")]
            running_batch = type("B", (), {"reqs": []})()
            last_batch = None
            chunked_req = None

        sched = _Sched()
        va.bind_scheduler(sched)
        va._REC.fake_providers = lambda: dict(self.f.prov)
        for rid, transient in (("a", 3000), ("b", 3500)):  # a finishes, then b is prefilled
            sched.last_batch = type("B", (), {"reqs": [_Req(rid)]})()
            self.cuda.alloc_free(transient * M)
            vpw.on_forward_end(_Runner(), _FB("extend", bs=1, rows=16384), self.cuda)
        peaks = self._record().vram["peak_by_state"]
        self.assertEqual(sorted(k for k in peaks if k.startswith("chunk=")),
                         ["chunk=16384,stau=1", "chunk=16384,stau=2"])
        # P's sleep leg ends the phase: the next chunk counts from 1 again
        vpw._STATE["cuda_override"] = self.cuda
        try:
            class Mgr:
                scheduler = None

                @vpw.flip_leg("release")
                def release(self):
                    return 1

            Mgr().release()
        finally:
            vpw._STATE.pop("cuda_override", None)
        self.assertIn("flip=P->D,leg=sleep", self._record().vram["peak_by_state"])
        sched.last_batch = type("B", (), {"reqs": [_Req("c")]})()
        self.assertEqual(va._p_stau(sched), 1)

    def test_flip_leg_reaches_the_block_with_direction(self):
        vpw._STATE["cuda_override"] = self.cuda
        try:
            class Mgr:
                scheduler = None

                @vpw.flip_leg("release")
                def release(self):
                    return 1

                @vpw.flip_leg("resume")
                def resume(self):
                    return 2

            Mgr().release()
            Mgr().resume()
        finally:
            vpw._STATE.pop("cuda_override", None)
        peaks = self._record().vram["peak_by_state"]
        self.assertIn("flip=D->P,leg=sleep", peaks)
        self.assertIn("flip=P->D,leg=wake", peaks)
        self.assertEqual(va.flip_state_key("P", "release"), "flip=P->D,leg=sleep")
        self.assertEqual(va.flip_state_key("P", "resume"), "flip=D->P,leg=wake")

    def test_attach_edge_in_the_cache_controller(self):
        from sglang.srt.managers.cache_controller import weg2_publish_rank_state

        class _Ctl:
            tp_rank, tp_size, pp_rank, pp_size, page_size = 0, 3, 0, 1, 64

        d2 = os.path.join(self.dir, "second")
        with envs.SGLANG_WEG2_RANK_STATE_DIR.override(d2):
            weg2_publish_rank_state(
                _Ctl(), form_a_worker=False, canonical_on=True, canonical_kv_built=True,
                canonical_blob_built=True, has_mamba_pool=True, owner_ctx=None,
            )
        states, bad = rs.read_group_states(d2)
        self.assertEqual(bad, [])
        (s,) = states
        self.assertEqual(s.vram["mark"], "attach")
        self.assertEqual(s.vram["nvml_pid_mib"], 21000)

    def test_launcher_shows_the_blocks_and_decides_nothing(self):
        from sglang.srt.weg2 import launcher

        class _Spec:
            name = "D"
            log = os.path.join(self.dir, "x.D.log")

        lines = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(launcher, "rank_state_dir_for", lambda spec: self.dir)
            n = launcher.rank_vram_display(_Spec(), lines.append)
        self.assertEqual(n, 1)
        (line,) = [l for l in lines if l.startswith("VRAM-ACTUAL group D tp0pp0: ")]
        self.assertIn("nvml_pid=21000", line)
        self.assertIn("partner=1200 foreign=300", line)
        gate = inspect.getsource(launcher.canonical_state_gate)
        self.assertIn("rank_vram_display(spec, log, states=states)", gate)


# --------------------------------------------------------------------------
# (d) never a write per decode round
# --------------------------------------------------------------------------


class TestNoWritePerRound(_Base):
    def test_thousand_rounds_write_only_on_a_new_max(self):
        cuda = _FakeCuda()
        vpw.reset_since_pools()
        rs.write_rank_state(va.attach(_d_rank_state(), self.dir), self.dir)
        with envs.SGLANG_WEG2_VRAM_PEAK_ROUNDS.override(4):
            w0, n0 = va.write_count(), self.f.nvml_calls
            for i in range(1000):
                if i == 500:
                    cuda.alloc_free(900 * M)  # one new high in the steady state
                vpw.on_forward_end(_Runner(), _FB("decode", bs=1 + (i // 100) % 2), cuda)
            writes, nvml = va.write_count() - w0, self.f.nvml_calls - n0
        # 250 windows closed; 2 keys (n=1, n=2) + the one new high on each
        self.assertLessEqual(writes, 4)
        self.assertEqual(nvml, writes)  # the NVML read only where a write follows
        peaks = self._record().vram["peak_by_state"]
        self.assertEqual(sum(v["n"] for v in peaks.values()) >= 1, True)
        self.assertEqual(set(peaks), {"n=1,S-,extend=0", "n=2,S-,extend=0"})

    def test_decode_forward_calls_nothing_but_the_window_end(self):
        """The decode branch of on_forward_end: a counter and an int compare;
        vram_actual is reached only from _close."""
        src = inspect.getsource(vpw.on_forward_end)
        self.assertNotIn("vram_actual", src)
        self.assertIn("vram_actual.on_window", inspect.getsource(vpw._close))


# --------------------------------------------------------------------------
# (e) IPC Phase-2 rule: no log text as a control value
# --------------------------------------------------------------------------

#: the 27B Phase-2 rule's log-reading gate helpers (test_weg2_ipc_phase2_0929)
_LOG_READERS = re.compile(r"\b(count_marker|canonical_marker_counts|gate_w10|gate_w11|_cc\.census|"
                          r"_cc\.form_a_worker_ranks)\(")


class TestNoLogTextAsControl(unittest.TestCase):
    def test_the_writer_reads_no_log(self):
        src = inspect.getsource(va)
        code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
        self.assertIsNone(_LOG_READERS.search(code))
        self.assertIsNone(re.search(r"^\s*(import re\b|from re import)", code, re.M))
        opens = [l.strip() for l in code.splitlines() if re.search(r"\bopen\(", l)]
        self.assertEqual(opens, ['with open(f"/proc/{int(pid)}/environ", "rb") as f:'])  # pid class only

    def test_the_launcher_display_reads_records_only(self):
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.rank_vram_display)
        self.assertIsNone(_LOG_READERS.search(src))
        self.assertNotIn("open(", src)
        self.assertIn("read_group_states", src)


if __name__ == "__main__":
    unittest.main()
