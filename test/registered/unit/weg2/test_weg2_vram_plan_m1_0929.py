# SPDX-License-Identifier: Apache-2.0
"""VRAM-VERTRAG M1 (29.09.): vram_plan.json is a VIEW on today's budget passes.

The launcher writes one ``weg2.vram_plan/1`` per budget pass (P budgets, map,
dry, early D, D-only, real D) with the solvers' posts, the other group's sleep
residue, a closure per card x phase x state and the profile's hand pins as
OVERRIDE. Nothing about the boot changes; the plan makes the idle VRAM visible
before the boot (rc12z30r3: 540-960 MiB per card fallow in D, VRAM-VERTRAG-0929
§4 M1).
"""
import ast
import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import vram_plan as vp
from sglang.srt.weg2 import vram_plan_view as vpv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BIG, S0, S2 = "GPU-31d7ef41", "GPU-5c648f96", "GPU-62dbbae1"


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=S0, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid=BIG, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid=S2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
    ])


def _fit(rank, *, budget, kv_rest, resident=29, local=201, kv_tokens=262144):
    return SimpleNamespace(rank=rank, n_layers=48, slot_mib=2.417, buffer_rows=resident + 48,
                           resident_rows=resident, local_experts=local, fixed_mib=1000.0,
                           draft_vocab_delta_mib=0.0, mamba_mib=0.0, spec_mib=0.0,
                           kv_tokens=kv_tokens, kv_mib=1728.0, fraction=0.45,
                           activation_mib=1024.0, budget_mib=float(budget), kv_rest_mib=float(kv_rest))


def _z30r3_view(objective="maxkv", **fit_kw):
    """rc12z30r3's D(Karte, Erwartung): booked P residue 2054/1188/1586 (the
    legacy dc_expect_d + windows), measured 1326/718/980 (launcher docstring of
    d_expect_dormant_other). The solver closed each rank to its KV rest."""
    cards = _cards()
    v = vpv.PlanView()
    v.reset(overrides=[{"group": "D", "key": "--max-total-tokens", "value": "262144",
                        "source": "--extra-d"}], identity={"profile": "nextflash"})
    booked = {BIG: 2054, S0: 1188, S2: 1586}
    measured = {BIG: 1326, S0: 718, S2: 980}
    terms, prelim = [], []
    for c in cards:
        terms.append({"carve": 0, "floor": 700, "dormant": booked[c.uuid], "growth": 0, "awake": 404})
        prelim.append(c.total_mib - 700 - booked[c.uuid] - 404)
    v.note_budget("D(Karte, Erwartung)", cards, prelim, prelim, terms)
    v.note_asleep("P", cards, measured, "RECORD(p records max, 8 boots)")
    fits = [_fit(i, budget=prelim[i], kv_rest=20, **fit_kw) for i in range(3)]
    v.note_d_solve("D(Karte, Erwartung)", cards, fits, objective=objective, kv_tokens_max=262144,
                   seats=6, stage_rows={})
    return cards, v


class Schema(CustomTestCase):
    def _plan(self):
        cards, v = _z30r3_view()
        return v.build("map", cards, boot_id="t", d_label="D(Karte, Erwartung)")

    def test_foreign_schema_is_refused_by_name(self):
        p = dict(self._plan(), schema="weg2.vram_plan/2")
        with self.assertRaises(vp.VramPlanRefused) as cm:
            vp.read_plan(p)
        self.assertEqual(cm.exception.code, "VRAM_PLAN_SCHEMA")

    def test_unknown_field_is_refused_by_name(self):
        p = dict(self._plan(), extra=1)
        with self.assertRaises(vp.VramPlanRefused) as cm:
            vp.read_plan(p)
        self.assertEqual(cm.exception.code, "VRAM_PLAN_FIELD")
        self.assertIn("extra", str(cm.exception))

    def test_unknown_rank_field_and_bad_provenance_are_refused(self):
        p = self._plan()
        p["groups"]["D"]["ranks"]["tp0"]["guess"] = 1
        with self.assertRaises(vp.VramPlanRefused):
            vp.read_plan(p)
        p = self._plan()
        p["groups"]["D"]["ranks"]["tp0"]["provenance"]["kv"] = "hand"
        with self.assertRaises(vp.VramPlanRefused):
            vp.read_plan(p)

    def test_tampered_plan_is_refused(self):
        p = self._plan()
        p["closure"][0]["rest_mib"] += 1
        with self.assertRaises(vp.VramPlanRefused) as cm:
            vp.read_plan(p)
        self.assertEqual(cm.exception.code, "VRAM_PLAN_ID")


class PlanId(CustomTestCase):
    def test_same_inputs_same_id(self):
        cards, v1 = _z30r3_view()
        _c, v2 = _z30r3_view()
        a = v1.build("map", cards, boot_id="t", d_label="D(Karte, Erwartung)")
        b = v2.build("map", cards, boot_id="t", d_label="D(Karte, Erwartung)")
        self.assertEqual(a["plan_id"], b["plan_id"])
        self.assertTrue(a["plan_id"].startswith("sha256:"))
        _c, v3 = _z30r3_view(objective="maxperf")
        c = v3.build("map", cards, boot_id="t", d_label="D(Karte, Erwartung)")
        self.assertNotEqual(a["plan_id"], c["plan_id"])


class Idle(CustomTestCase):
    def test_z30r3_d_rest_is_idle_without_a_bound(self):
        cards, v = _z30r3_view()
        plan = v.build("map", cards, d_label="D(Karte, Erwartung)")
        d = {c["card"]: c for c in plan["closure"] if c["phase"] == "D awake"}
        # booked - measured + the solver's KV rest: 728+20 / 470+20 / 606+20
        self.assertEqual({u: d[u]["rest_mib"] for u in d}, {BIG: 748, S0: 490, S2: 626})
        self.assertTrue(all(d[u]["idle"] and not d[u]["bound_by"] for u in d))
        self.assertEqual(len([x for x in vp.idle_items(plan) if x.startswith("D awake")]), 3)

    def test_objective_maxperf_binds_the_rest(self):
        cards, v = _z30r3_view(objective="maxperf")
        plan = v.build("map", cards, d_label="D(Karte, Erwartung)")
        d = [c for c in plan["closure"] if c["phase"] == "D awake"]
        self.assertTrue(all(c["bound_by"] == "objective" and not c["idle"] for c in d))

    def test_ctx_max_with_every_expert_resident_binds_the_rest(self):
        cards, v = _z30r3_view(resident=201)
        plan = v.build("map", cards, d_label="D(Karte, Erwartung)")
        d = [c for c in plan["closure"] if c["phase"] == "D awake"]
        self.assertTrue(all(c["bound_by"] == "ctx_max" for c in d))
        # experts full but KV below the context: nothing binds, the rest is idle
        cards, v = _z30r3_view(resident=201, kv_tokens=131072)
        plan = v.build("map", cards, d_label="D(Karte, Erwartung)")
        self.assertTrue(all(c["idle"] for c in plan["closure"] if c["phase"] == "D awake"))


def _pfit(stage, *, rows, fraction, headroom):
    return SimpleNamespace(stage=stage, fraction=fraction, buffer_rows=rows, layer_row_mib=19.3,
                           row_card_mib=19.3, expert_mib=8000.0, kv_mib=544.0, transient_mib=3267.0,
                           transient_source="RECORD(P_ACTIVATION_MIB)", draft_mib=0.0,
                           prompt_tokens=262144, growth_mib=329.0, headroom_mib=float(headroom),
                           near_oom_mib=400.0, ceiling_fraction=0.996)


class ExpertsFull(CustomTestCase):
    """-e2cut-z30x-frp (29.09.): PP2 holds all 512 expert rows (f 0.9956 after the
    H25 draft post) and keeps 1572 MiB headroom -- not waste, no expert can take
    it and P's KV is sized by its chunk admission. The plan names it
    ``experts_full`` instead of IDLE; a stage below full stays IDLE."""

    def _plan(self, rows, fraction):
        cards = _cards()
        v = vpv.PlanView()
        v.reset(overrides=[], identity={"profile": "nextflash"})
        v.note_p_card(cards, [_pfit(2, rows=rows, fraction=fraction, headroom=1572)],
                      [0.33, 0.701, fraction], [32, 32, 32], "fnFL2x163", num_experts=512)
        plan = v.build("p_budget", cards)
        return [c for c in plan["closure"] if c["phase"] == "P awake"][0], plan

    def test_full_stage_rest_is_bound_not_idle(self):
        c, plan = self._plan(512, 0.995605)
        self.assertEqual((c["bound_by"], c["idle"], c["rest_to"]), ("experts_full", False, "none"))
        # the log line still names the rest, with its reason instead of "-"
        self.assertEqual([x for x in vp.idle_items(plan) if x.startswith("P awake")],
                         ["P awake/nvml2:1172(experts_full)"])
        vp.read_plan(plan)  # the reader knows the reason

    def test_stage_below_full_stays_idle(self):
        c, _ = self._plan(408, 0.7339)
        self.assertEqual((c["bound_by"], c["idle"], c["rest_to"]), ("", True, "experts_resident"))

    def test_bound_by_of(self):
        self.assertEqual(vp.bound_by_of(experts_full=True, rest_to="none"), "experts_full")
        self.assertEqual(vp.bound_by_of(experts_full=True, ctx_at_max=True), "ctx_max")
        self.assertEqual(vp.bound_by_of(experts_full=True, rest_to="kv"), "")

    def test_launcher_passes_the_expert_count(self):
        src = inspect.getsource(L.p_card_verdict)
        self.assertEqual(src.count("num_experts=int(num_experts))"), 2)


class ArgvBudgets(CustomTestCase):
    """(a) budget_mib per rank == --rank-gpu-memory-mib of the argv, card
    order. Numbers of the 29.09. dry runs on cb98c3d94a + M0 (m1a2 -st-cut z30v,
    m1y -yarn2). 27B (x27r on z30x 188b82ec2e, profile 27b.env): the host
    ledger (W87, host RAM -- not a VRAM post) refuses before any argv, so that
    run used a fixture that replaces only the host verdict by a stub arm; its
    budgets and argv are the launcher's own."""

    RUNS = {"st-cut": {"P": [28192, 17976, 17704], "D": [26312, 17640, 17640]},
            "yarn2": {"P": [28176, 18048, 17768], "D": [26344, 17696, 17888]},
            "27b": {"P": [25280, 14928, 14648], "D": [25920, 15272, 15272]}}

    def test_plan_budgets_are_the_argv_budgets(self):
        cards = _cards()
        for run, b in self.RUNS.items():
            v = vpv.PlanView()
            zeros = [{"carve": 0, "floor": 0, "dormant": 0, "awake": 0}] * 3
            v.note_budget("P", cards, b["P"], b["P"], zeros)
            v.note_budget("D(dry, expectation)", cards, b["D"], b["D"], zeros)
            plan = v.build("dry", cards, d_label="D(dry, expectation)")
            for g in ("P", "D"):
                argv = ["--x", "1", "--rank-gpu-memory-mib", ",".join(str(x) for x in b[g])]
                self.assertEqual(vp.argv_budget_mismatches(plan, g, argv), [], (run, g))
            wrong = ["--rank-gpu-memory-mib", "1,2,3"]
            self.assertTrue(vp.argv_budget_mismatches(plan, "D", wrong))


class Wiring(CustomTestCase):
    """(c) the view lives on its call edges: without them it stays empty."""

    def test_budgets_from_dc_notes_every_pass_on_every_return(self):
        src = inspect.getsource(L.budgets_from_dc)
        tree = ast.parse(src.lstrip() if src.startswith(" ") else src)
        fn = tree.body[0]
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        notes = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "note_budget"]
        self.assertEqual(len(returns), len(notes))
        L.vram_view().reset()
        L.budgets_from_dc(_cards(), {BIG: 1320, S0: 712, S2: 696}, [].append, "D")
        self.assertIn("D", L.vram_view().budgets)

    def test_each_pass_emits(self):
        self.assertIn("vram_plan_emit(", inspect.getsource(L.log_d_rank_vram_solve))
        self.assertIn("note_d_solve(", inspect.getsource(L.log_d_rank_vram_solve))
        main = inspect.getsource(L.main)
        self.assertIn('vram_plan_emit(ns, cards, "p_budget", log, group="P")', main)
        self.assertIn("vram_view().reset(overrides=_vpv.profile_pins(ns)", main)
        self.assertIn("note_asleep(", inspect.getsource(L.d_expect_dormant_other))
        self.assertIn("note_p_card(", inspect.getsource(L.p_card_verdict))
        self.assertEqual(set(L._VRAM_PASS_BY_LABEL.values()) | {"p_budget", "d_early"},
                         set(vp.PASSES))

    def test_a_pass_without_solve_still_writes_its_plan(self):
        # 27B (dense) leaves the D solve at the form gate: the pass must still
        # write its plan, the D closure named open -- every early return of
        # the solve's own body goes through _plan_without_solve().
        src = inspect.getsource(L.log_d_rank_vram_solve)
        fn = ast.parse(src).body[0]

        def own(node):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                yield ch
                yield from own(ch)

        # the LRU-floor re-solve returns the recursive call, which emits itself
        bare = [n for n in own(fn) if isinstance(n, ast.Return) and n.value is None]
        calls = [n for n in own(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_plan_without_solve"]
        self.assertTrue(bare)
        self.assertEqual(len(bare), len(calls))
        cards = _cards()
        v = vpv.PlanView()
        zeros = [{"carve": 0, "floor": 0, "dormant": 0, "awake": 0}] * 3
        v.note_budget("D(dry, expectation)", cards, [25920, 15272, 15272],
                      [25920, 15272, 15272], zeros)
        plan = v.build("dry", cards, d_label="D(dry, expectation)")
        self.assertTrue(any(o.startswith("D.closure UNMEASURED") for o in plan["open"]))

    def test_emit_writes_the_state_dir_and_one_line(self):
        import json
        import tempfile

        cards, v = _z30r3_view()
        L._VRAM_VIEW = v
        lines = []
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("WEG2_STATE_DIR")
            os.environ["WEG2_STATE_DIR"] = d
            try:
                plan = L.vram_plan_emit(SimpleNamespace(tag="t"), cards, "map", lines.append,
                                        d_label="D(Karte, Erwartung)")
            finally:
                if old is None:
                    os.environ.pop("WEG2_STATE_DIR", None)
                else:
                    os.environ["WEG2_STATE_DIR"] = old
            got = json.load(open(os.path.join(d, "vram_plan-map.json")))
            self.assertEqual(got["plan_id"], plan["plan_id"])
            self.assertEqual(json.load(open(os.path.join(d, "vram_plan.json")))["pass"], "map")
        self.assertTrue(lines[0].startswith("VRAM-PLAN pass=map plan_id=sha256:"))
        self.assertIn("overrides=[D:--max-total-tokens=262144]", lines[0])


class Pins(CustomTestCase):
    def test_profile_pins_are_overrides(self):
        ns = SimpleNamespace(extra_p="--max-total-tokens 262144 --kv-cache-dtype fp8_e4m3",
                             extra_d="--rank-moe-ratio 183,137,168 --cuda-graph-bs-decode 1 2 3",
                             env_p="SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_X=1",
                             env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48",
                             pp_cut_expert_device_fraction="0.324,0.637,0.39")
        got = {(o["group"], o["key"]): o["value"] for o in vpv.profile_pins(ns)}
        self.assertEqual(got[("P", "--max-total-tokens")], "262144")
        self.assertEqual(got[("D", "--rank-moe-ratio")], "183,137,168")
        self.assertEqual(got[("D", "--cuda-graph-bs-decode")], "1 2 3")  # a list flag, whole
        self.assertEqual(got[("D", "SGLANG_MOE_SCRATCH_SLOTS")], "100,48,48")
        self.assertEqual(got[("launcher", "--pp-cut-expert-device-fraction")], "0.324,0.637,0.39")
        self.assertNotIn(("P", "SGLANG_X"), got)


if __name__ == "__main__":
    unittest.main()
