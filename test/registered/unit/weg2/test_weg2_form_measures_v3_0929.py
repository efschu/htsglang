"""Form-Messmatrix v3 (27B-FORM-MESSMATRIX, 29.09.): schema, identity
separation, v2 migration, no interpolation, structural "unmoeglich" cells,
the --calib selection and the import of a run_matrix.sh directory."""

import json
import os
import tempfile
import unittest

from sglang.srt.weg2 import form_measures as fm

U5090 = "GPU-5090"
U3080 = "GPU-3080x8"
ID27 = fm.Identity(model="qwen27b", precision=("nvfp4",), cards=(U5090,), image_rev="2c5fd4a2b7")
ID27_TP2 = fm.Identity(model="qwen27b", precision=("nvfp4", "w4a8"), cards=(U5090, U3080), image_rev="2c5fd4a2b7")
IDNF = fm.Identity(model="nextflash", precision=("int4",), cards=(U5090,), image_rev="2c5fd4a2b7")

SPEC = fm.matrix_spec_from_value({
    "forms": {
        "A": {"arm": "a1", "axes": {"weights": "1", "transport": "none"}},
        "B77": {"arm": "a2_7723", "axes": {"weights": "77,23", "transport": "nccl"}},
        "A_KV": {"arm": None, "startable": False, "why_not": "F15 wired=False",
                 "axes": {"weights": "1,0,0", "transport": "bar1"}},
    },
    "bs": [1, 2, 3], "depths": ["2k", "97k"], "texts": ["code", "prose"], "temps": ["warm"],
    "capacity": {"A": {"97k": 2}},
}, provenance="test")


def _measured(ax, ms, at="2026-09-29T07:00:00Z", rev="2c5fd4a2b7"):
    return fm.Cell(axes=ax, state=fm.STATE_MEASURED, round_ms_median=ms, n=10, at=at, image_rev=rev)


class TestCellsAndIdentity(unittest.TestCase):
    def test_identity_separates_models(self):
        s = fm.FormMeasuresV3()
        s.upsert(ID27, _measured({"form": "A", "bs": 1, "depth": "2k", "text": "code", "temp": "warm"}, 23.8))
        self.assertEqual(s.round_ms(ID27, form="A", bs=1, depth="2k", text="code", temp="warm"), 23.8)
        self.assertIsNone(s.round_ms(IDNF, form="A", bs=1, depth="2k", text="code", temp="warm"))

    def test_no_interpolation_and_depth_bucketed_up(self):
        s = fm.FormMeasuresV3()
        s.upsert(ID27, _measured({"form": "A", "bs": 1, "depth": "2k", "text": "code", "temp": "warm"}, 23.8))
        s.upsert(ID27, _measured({"form": "A", "bs": 3, "depth": "2k", "text": "code", "temp": "warm"}, 22.5))
        # bs2 is between two measured cells: still missing
        self.assertIsNone(s.round_ms(ID27, form="A", bs=2, depth="2k", text="code", temp="warm"))
        # 1500 tokens bucket UP to 2k; 3000 tokens bucket to 10k, which has no cell
        self.assertEqual(s.round_ms(ID27, form="A", bs=1, depth=1500, text="code", temp="warm"), 23.8)
        self.assertIsNone(s.round_ms(ID27, form="A", bs=1, depth=3000, text="code", temp="warm"))
        self.assertEqual(fm.depth_label(245760), "240k")
        with self.assertRaises(fm.FormMeasuresError):
            fm.depth_label(10 ** 7)

    def test_impossible_needs_reason_measured_needs_value(self):
        with self.assertRaises(fm.FormMeasuresError):
            fm.Cell(axes={"form": "A"}, state=fm.STATE_IMPOSSIBLE_START)
        with self.assertRaises(fm.FormMeasuresError):
            fm.Cell(axes={"form": "A"}, state=fm.STATE_MEASURED)
        with self.assertRaises(fm.FormMeasuresError):
            fm.Cell(axes={"nope": 1})

    def test_newer_measurement_wins_older_does_not(self):
        s = fm.FormMeasuresV3()
        ax = {"form": "A", "bs": 1, "depth": "2k", "text": "code", "temp": "warm"}
        s.upsert(ID27, _measured(ax, 23.8, at="2026-09-29T07:00:00Z"))
        s.upsert(ID27, _measured(ax, 30.0, at="2026-09-28T07:00:00Z"))
        self.assertEqual(s.round_ms(ID27, **ax), 23.8)
        s.upsert(ID27, _measured(ax, 21.0, at="2026-09-30T07:00:00Z"))
        self.assertEqual(s.round_ms(ID27, **ax), 21.0)
        s.upsert(ID27, fm.Cell(axes=ax, state=fm.STATE_UNMEASURED))
        self.assertEqual(s.round_ms(ID27, **ax), 21.0)

    def test_save_load_roundtrip_atomic(self):
        s = fm.FormMeasuresV3(allreduce={"bar1": {"x": {"1": {"graph_median_us": 23}}}})
        s.upsert(ID27, _measured({"form": "A", "bs": 1, "depth": "2k", "text": "code", "temp": "warm"}, 23.8))
        s.upsert(ID27, fm.Cell(axes={"form": "A_KV", "bs": 1}, state=fm.STATE_IMPOSSIBLE_START, reason="F15"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "fm.json")
            s.save(p)
            self.assertEqual(os.listdir(d), ["fm.json"])
            t = fm.load(p)
        self.assertEqual(t.round_ms(ID27, form="A", bs=1, depth="2k", text="code", temp="warm"), 23.8)
        self.assertEqual(t.lookup(ID27, form="A_KV", bs=1).state, fm.STATE_IMPOSSIBLE_START)
        self.assertEqual(t.allreduce, s.allreduce)
        self.assertIsNone(fm.load("/nonexistent/fm.json").lookup(ID27, form="A"))


class TestV2Migration(unittest.TestCase):
    V2 = {"schema": fm.SCHEMA_V2,
          "allreduce": {"bar1": {}},
          "rounds": {"a1": {U5090: {"1": {"median_ms": 23.8, "transport": "none", "n": 63}}},
                     "a2_7723": {f"{U5090}+{U3080}": {"6": {"median_ms": 42.8, "transport": "nccl", "n": 56},
                                                      "5": {"median_ms": None, "transport": "nccl"}}}}}

    def test_migration_without_identity_matches_no_boot(self):
        s = fm.from_doc(self.V2)
        self.assertTrue(s.migrated_from_v2)
        self.assertIsNone(s.round_ms(ID27, form="a1", bs=1, transport="none"))
        anon = fm.Identity(model=None, cards=(U5090,))
        self.assertEqual(s.round_ms(anon, form="a1", bs=1, transport="none"), 23.8)

    def test_migration_with_identity_and_v2_view(self):
        s = fm.from_doc(self.V2, v2_identity=ID27_TP2)
        view = s.rounds_ms_v2_view(ID27_TP2, transport="nccl")
        self.assertEqual(view, {"a2_7723": {f"{U5090}+{U3080}": {6: 42.8}}})
        # a depth-asking view never returns the depth-less migrated cells
        self.assertEqual(s.rounds_ms_v2_view(ID27_TP2, transport="nccl", depth="2k"), {})
        self.assertEqual(s.lookup(ID27_TP2, form="a2_7723", bs=5, transport="nccl").state, fm.STATE_UNMEASURED)

    def test_unknown_schema_refused(self):
        with self.assertRaises(fm.FormMeasuresError):
            fm.from_doc({"schema": "weg2.form_measures/1"})


class TestSelection(unittest.TestCase):
    def test_structural_cells_never_selected_and_marked(self):
        s = fm.FormMeasuresV3()
        allc = fm.select("all", SPEC, s, ID27)
        forms = {ax["form"] for _, ax in allc}
        self.assertNotIn("A_KV", forms)
        self.assertFalse(any(ax["form"] == "A" and ax["depth"] == "97k" and ax["bs"] == 3 for _, ax in allc))
        n = fm.mark_structural(SPEC, s, ID27)
        self.assertEqual(n, 12 + 2)  # A_KV 3 bs x 2 depths x 2 texts; A bs3@97k x 2 texts
        c = s.lookup(ID27, form="A", bs=3, depth="97k", text="code", temp="warm", weights="1", transport="none")
        self.assertEqual(c.state, fm.STATE_IMPOSSIBLE_CAPACITY)
        # an empty run of that cell (global bands) must not erase the structure
        s.upsert(ID27, fm.Cell(axes=c.axes, state=fm.STATE_UNMEASURED, reason="0 rounds"))
        self.assertEqual(s.lookup(ID27, **c.axes).state, fm.STATE_IMPOSSIBLE_CAPACITY)

    def test_missing_stale_explicit(self):
        s = fm.FormMeasuresV3()
        ax = {"form": "A", "bs": 1, "depth": "2k", "text": "code", "temp": "warm", "weights": "1",
              "transport": "none"}
        s.upsert(ID27, _measured(ax, 23.8, at="2026-09-20T00:00:00Z", rev="old"))
        n_all = len(fm.select("all", SPEC, s, ID27))
        self.assertEqual(len(fm.select("missing", SPEC, s, ID27)), n_all - 1)
        now = fm._parse_at("2026-09-29T00:00:00Z")
        self.assertEqual(len(fm.select("stale:7d", SPEC, s, ID27, now=now)), n_all)
        self.assertEqual(len(fm.select("stale:30d", SPEC, s, ID27, now=now)), n_all - 1)
        self.assertEqual(len(fm.select("stale:rev", SPEC, s, ID27, image_rev="new")), n_all)
        sel = fm.select("form=B77;bs=1..2;depth=97k;text=prose", SPEC, s, ID27)
        self.assertEqual(sorted(ax["bs"] for _, ax in sel), [1, 2])
        with self.assertRaises(fm.FormMeasuresError):
            fm.select("weights=1", SPEC, s, ID27)

    def test_plan_env_bands(self):
        sel = fm.select("form=A;depth=2k,97k;text=code", SPEC, fm.FormMeasuresV3(), ID27)
        env = fm.plan_env(sel)
        self.assertEqual(env["ARMS"], "a1")
        self.assertEqual(env["LADDER_BANDS"], "code@97k:1,2;code@2k:1,2,3")

    def test_profile_record_27b(self):
        spec = fm.matrix_spec("qwen27b", "nvfp4")
        # int8 has its record since 29.09. (Resplit je Flip, test_weg2_d_resplit_measured_0929);
        # a format without one is still refused, never a default matrix
        self.assertEqual(fm.matrix_spec("qwen27b", "int8").forms[0].name, "C")
        with self.assertRaises(fm.FormMeasuresError):
            fm.matrix_spec("qwen27b", "fp8")
        names = {f.name for f in spec.forms}
        self.assertTrue({"A", "B77", "B88", "A_KV", "C"} <= names)
        self.assertFalse(next(f for f in spec.forms if f.name == "A_KV").startable)
        self.assertEqual(spec.capacity["A"]["240k"], 1)


def _write_run(d):
    t0 = 1790664000.0
    lad = [
        # bs1 code@2k warm: one group, 2 streams... (bs1 = 1 stream)
        {"role": "stream", "point": "code@2k", "kind": "code", "target_tokens": 2048, "bs": 1, "temp": "warm",
         "rep": 0, "t_send": t0, "t_end": t0 + 2.0, "completion_tokens": 512},
        {"role": "group", "point": "code@2k", "bs": 1, "temp": "warm", "rep": 0, "streams_short": 0},
        # bs2 prose@2k warm: a short stream -> group skipped
        {"role": "stream", "point": "prose@2k", "kind": "prose", "target_tokens": 2048, "bs": 2, "temp": "warm",
         "rep": 0, "t_send": t0 + 10, "t_end": t0 + 12, "completion_tokens": 28, "short": True},
        {"role": "stream", "point": "prose@2k", "kind": "prose", "target_tokens": 2048, "bs": 2, "temp": "warm",
         "rep": 0, "t_send": t0 + 10, "t_end": t0 + 12, "completion_tokens": 512},
        {"role": "group", "point": "prose@2k", "bs": 2, "temp": "warm", "rep": 0, "streams_short": 1},
    ]
    with open(os.path.join(d, "a2_7723_ladder.jsonl"), "w") as f:
        for r in lad:
            f.write(json.dumps(r) + "\n")
    lines = []
    rnd = 0
    for i in range(8):
        t = t0 + 0.1 + i * 0.2
        rnd += 1
        lines.append(f"[2026-09-29 06:44:06 TP0] Decode rank batch, rank: 0, #round: {rnd}, t: {t:.3f}, bs: 1, "
                     f"#rows: 8, #fwd: 2, gpu-ms: 24.3 (compute 18.9, wait 5.4) (wait by family: x)")
        lines.append(f"[2026-09-29 06:44:06 TP1] Decode rank batch, rank: 1, #round: {rnd}, t: {t:.3f}, bs: 1, "
                     f"#rows: 8, #fwd: 2, gpu-ms: 24.{i % 3} (compute 15.9, wait 8.4) (wait by family: x)")
    for i in range(8):
        t = t0 + 10.1 + i * 0.2
        rnd += 1
        lines.append(f"[2026-09-29 06:44:16 TP0] Decode rank batch, rank: 0, #round: {rnd}, t: {t:.3f}, bs: 2, "
                     f"#rows: 16, #fwd: 2, gpu-ms: 27.8 (compute 19.5, wait 8.2)")
    lines.append("[2026-09-29 06:44:06] Decode batch, #running-req: 1, accept len: 5.5, gen throughput")
    with open(os.path.join(d, "a2_7723_server.log"), "w") as f:
        f.write("\n".join(lines) + "\n")


class TestImport(unittest.TestCase):
    def test_import_synthetic_run(self):
        fs = next(f for f in SPEC.forms if f.name == "B77")
        s = fm.FormMeasuresV3()
        with tempfile.TemporaryDirectory() as d:
            _write_run(d)
            st = fm.import_calib_run(d, {"a2_7723": (ID27_TP2, fs)}, s, boot="b1", image_rev="r",
                                     at="2026-09-29T06:45:00Z")
        self.assertEqual(st, {"measured": 1, "unmeasured": 0, "skipped_short": 1})
        c = s.lookup(ID27_TP2, form="B77", bs=1, depth="2k", text="code", temp="warm", weights="77,23",
                     transport="nccl")
        self.assertEqual(c.state, fm.STATE_MEASURED)
        self.assertEqual(c.n, 8)
        self.assertEqual(c.compute_ms, [18.9, 15.9])
        self.assertEqual(c.wait_ms, [5.4, 8.4])
        self.assertGreaterEqual(c.round_ms_median, 24.2)  # slowest rank per round
        self.assertIsNone(s.lookup(ID27_TP2, form="B77", bs=2, depth="2k", text="prose", temp="warm",
                                   weights="77,23", transport="nccl"))

    def test_pre_fix_ladder_short_streams_detected_without_flag(self):
        fs = next(f for f in SPEC.forms if f.name == "B77")
        s = fm.FormMeasuresV3()
        with tempfile.TemporaryDirectory() as d:
            _write_run(d)
            p = os.path.join(d, "a2_7723_ladder.jsonl")
            rows = [json.loads(l) for l in open(p)]
            for r in rows:  # an old ladder: no short / streams_short fields
                r.pop("short", None)
                r.pop("streams_short", None)
            with open(p, "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            st = fm.import_calib_run(d, {"a2_7723": (ID27_TP2, fs)}, s)
        self.assertEqual(st["skipped_short"], 1)
        self.assertIsNone(s.lookup(ID27_TP2, form="B77", bs=2, depth="2k", text="prose", temp="warm",
                                   weights="77,23", transport="nccl"))

    def test_min_rounds_leaves_unmeasured(self):
        fs = next(f for f in SPEC.forms if f.name == "B77")
        s = fm.FormMeasuresV3()
        with tempfile.TemporaryDirectory() as d:
            _write_run(d)
            st = fm.import_calib_run(d, {"a2_7723": (ID27_TP2, fs)}, s, min_rounds=50)
        self.assertEqual(st["unmeasured"], 1)

    def test_cli_plan(self):
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "fm.json")
            out = os.path.join(d, "out.txt")
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = fm.main(["plan", "--store", store, "--profile", "qwen27b", "--fmt", "nvfp4",
                              "--identity", json.dumps(ID27.to_json()), "--calib", "form=A;depth=240k",
                              "--mark-structural"])
            self.assertEqual(rc, 0)
            self.assertIn('ARMS="a1"', buf.getvalue())
            self.assertIn("LADDER_BANDS=", buf.getvalue())
            s = fm.load(store)
            self.assertEqual(s.lookup(ID27, form="A_KV", bs=1, depth="2k", text="code", temp="warm",
                                      roles="host,kv_only,kv_only", weights="1,0,0",
                                      transport="bar1").state, fm.STATE_IMPOSSIBLE_START)


class TestHeatMisses(unittest.TestCase):
    def _rec(self, d, rank, not_local, steps, kind="moe_heat"):
        p = os.path.join(d, f"moe_heat_D_tp{rank}_{len(os.listdir(d))}.json")
        with open(p, "w") as f:
            json.dump({"kind": kind, "version": 1, "group": "D", "rank": rank,
                       "layers": [{"counts": [5, 3], "not_local": not_local, "steps": steps}]}, f)
        return p

    def test_sum_per_rank_and_refuse_foreign(self):
        with tempfile.TemporaryDirectory() as d:
            ps = [self._rec(d, 0, 4, 10), self._rec(d, 0, 6, 10), self._rec(d, 1, 0, 0)]
            m = fm.heat_misses(ps)
            self.assertEqual(m[0]["misses_per_step"], 0.5)
            self.assertEqual(m[1]["steps"], 0)
            with self.assertRaises(fm.FormMeasuresError):
                fm.heat_misses([self._rec(d, 2, 1, 1, kind="other")])


if __name__ == "__main__":
    unittest.main()
