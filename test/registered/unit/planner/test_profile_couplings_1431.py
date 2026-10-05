"""PROFIL-EDITOR S4a (Auftrag 1431): die Kopplungs-Engine ``planner/profile_couplings.py`` (C1-C4 rechenbar, C5-C7 als Kante).

Gepinnt (GPU-frei; Modellprofile aus den Config-Fixtures von Auftrag 960, Hardwareprofile synthetisch):

* GOLDEN gegen die Aufzeichnung des Referenz-Boots NF INT4-abl (Kartenplaner-Record ``nf-int4-abl.json``, Boot dkrnfint4h6ablbar1dauer10030924):
  - der P-Schnitt ``--pp-stage-ratio 29,11,8`` ergibt aus den Layer-Familien dieselben Attention-Layer ``7,3,2`` wie ``--pp-attn-stage-ratio``;
  - KV-Pool bei 262144 Token (K+V, fp8): 1802 / 778 / 512 MiB (rank_posts.P.pp0..2.kv_pools) == Nutzlast der Engine innerhalb 1,5 %;
  - Mamba-/Zustandspool 1098 / 399 / 299 MiB (kv_posts: "mamba state pool + ... reserve", 32 Slots, bf16-SSM) == Engine innerhalb 2 %;
  - KV-Zelle 1088 B je Attention-Layer und Token und die Metall-Zellen 8704 / 4352 / 3264 (fnFL2w123, mit einem Draft-Layer je Stufe);
  - Experten-Pufferregel: 0,33 mit 32 Scratch-Zeilen = 201 von 512 (NF y8), 0,95 -> 487 Zeilen (fnFL2w73);
* Erhaltungssaetze: ein Layer-Verschieben aendert die Summe der Gewichte und des KV-Preises nicht, nur die Verteilung;
* ein Ueberlauf wird gezeigt, nie still beschnitten; Vektorlaengen, die nicht zur Kartenzahl passen, werden benannt verweigert;
* synthetische Inventare: 3 Karten (Referenz 5090 + 2x3080), 2 Karten, 1 Karte, sm86 / sm89 / sm120;
* C5-C7 sind benannte Kanten mit ``computed: False``.
"""

import copy
import json
import os
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import expert_residency as ER  # noqa: E402
from sglang.srt.planner import profile_couplings as PC  # noqa: E402
from sglang.srt.weg2 import model_profile as MP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "weg2", "fixtures", "profil_s3_1003")
MIB = float(1 << 20)

# Referenz-Rig (Karten in Planer-Reihenfolge: 5090 vor den 3080, wie order_cards)
RIG3 = [("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)]

_MODELS = {}


def model(name):
    if name not in _MODELS:
        _MODELS[name] = MP.estimate(os.path.join(FX, name))
    return _MODELS[name]


def hw3():
    return PC.synthetic_hardware(RIG3)


# Aufzeichnung des Referenz-Boots (Kartenplaner-Record nf-int4-abl.json, 03.10. 09:24Z)
REC_P = {
    "stage_layers": [29, 11, 8],
    "attn_ratio": [7, 3, 2],
    "budget_mib": [28440, 17568, 18200],
    "kv_pool_mib": [901 + 901, 389 + 389, 256 + 256],     # kv_pools k_mib + v_mib bei 262144 Token
    "mamba_reserve_mib": [1098, 399, 299],                # kv_posts.posts_mib["mamba state pool + ... reserve"]
    "slots": 32,
    "tokens": 262144,
}


def rec_settings(**kw):
    s = {"stage_layers": REC_P["stage_layers"], "budget_mib": REC_P["budget_mib"], "kv_dtype": "fp8_e4m3",
         "context_tokens": REC_P["tokens"], "chunk_tokens": 16384, "mamba_slots": REC_P["slots"]}
    s.update(kw)
    return s


class TestGoldenReferenceBoot(unittest.TestCase):
    def setUp(self):
        self.m = model("nextflash_int4mixed")
        self.out = PC.c1_layer_split(hw3(), self.m, rec_settings())

    def test_attention_layers_per_stage_equal_the_recorded_attn_stage_ratio(self):
        self.assertEqual([s["attn_layers"] for s in self.out["stages"]], REC_P["attn_ratio"])

    def test_kv_payload_matches_the_recorded_kv_pools(self):
        payload = float(self.m["kv"]["variants"]["fp8_e4m3"]["payload_bytes"]["v"])
        for i, s in enumerate(self.out["stages"]):
            mine = payload * s["attn_layers"] * REC_P["tokens"] / MIB
            rec = REC_P["kv_pool_mib"][i]
            self.assertAlmostEqual(mine / rec, 1.0, delta=0.015, msg="Stufe %d: Engine %.1f MiB, Aufzeichnung %d MiB" % (i, mine, rec))

    def test_state_pool_matches_the_recorded_mamba_reserve(self):
        # die Aufzeichnung rechnet den SSM-Zustand in bf16 (1,5586 MiB je Linear-Layer und Slot)
        bf16 = float(self.m["state"]["variants_mib"]["bfloat16"])
        for i, s in enumerate(self.out["stages"]):
            mine = s["linear_layers"] * bf16 * REC_P["slots"]
            self.assertAlmostEqual(mine / REC_P["mamba_reserve_mib"][i], 1.0, delta=0.02, msg="Stufe %d" % i)

    def test_cell_is_1088_and_metal_stage_cells_with_one_draft_layer(self):
        self.assertEqual(self.m["kv"]["variants"]["fp8_e4m3"]["payload_bytes"]["v"] + self.m["kv"]["variants"]["fp8_e4m3"]["scale_bytes"]["v"], 1088.0)
        out = PC.c1_layer_split(hw3(), self.m, rec_settings(draft=True))
        cells = [(s["attn_layers"] + 1) * 1088 for s in out["stages"]]
        self.assertEqual(cells, [8704, 4352, 3264])
        # und der KV-Preis der Engine ist genau Token x diese Zelle
        for s, cell in zip(out["stages"], cells):
            self.assertAlmostEqual(s["terms"]["kv"]["v"], REC_P["tokens"] * cell / MIB, places=2)

    def test_budget_is_taken_from_the_input_not_invented(self):
        self.assertEqual([s["budget_mib"]["v"] for s in self.out["stages"]], REC_P["budget_mib"])
        self.assertEqual({s["budget_mib"]["src"] for s in self.out["stages"]}, {PC.SRC_INPUT})

    def test_buffer_rule_goldens(self):
        self.assertEqual(ER.buffer_rows(local_experts=512, fraction=0.33, scratch_rows=32), 201)
        self.assertEqual(ER.resident_rows(512, 0.95), 487)
        self.assertAlmostEqual(MP.expert_buffer_fraction(512, 0.33, 32), 201 / 512.0)


class TestTermsAreAccountedFor(unittest.TestCase):
    def test_needs_is_the_sum_of_the_terms_and_free_is_budget_minus_needs(self):
        out = PC.c1_layer_split(hw3(), model("nextflash_int4mixed"), rec_settings(fixed_overhead_mib=[1200, 900, 900]))
        for s in out["stages"]:
            total = sum(t["v"] for t in s["terms"].values())
            self.assertAlmostEqual(s["needs_mib"], total, delta=0.01)
            self.assertAlmostEqual(s["free_mib"], s["budget_mib"]["v"] - s["needs_mib"], delta=0.01)
            self.assertEqual(s["overflow_mib"], round(max(0.0, -s["free_mib"]), 3))

    def test_every_term_names_its_source(self):
        out = PC.c1_layer_split(hw3(), model("nextflash_int4mixed"), rec_settings())
        for s in out["stages"]:
            for name, t in s["terms"].items():
                self.assertTrue(t["src"], name)
        self.assertEqual(out["stages"][0]["terms"]["fixed"]["src"], PC.SRC_DEFAULT)
        self.assertTrue(any("OBERGRENZE" in w for w in out["warnings"]), "ohne fixed_overhead_mib muss die Obergrenze benannt sein")

    def test_fixed_overhead_given_removes_the_upper_bound_warning(self):
        out = PC.c1_layer_split(hw3(), model("nextflash_int4mixed"), rec_settings(fixed_overhead_mib=1000))
        self.assertFalse(any("OBERGRENZE" in w for w in out["warnings"]))
        self.assertEqual(out["stages"][1]["terms"]["fixed"]["src"], PC.SRC_INPUT)


class TestC1MoveLayers(unittest.TestCase):
    def setUp(self):
        self.m = model("qwen27b_int8_vocabembed")
        self.s = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 100000, "chunk_tokens": 2048, "mamba_slots": 4}

    def test_moving_a_layer_conserves_total_weights_and_kv_price(self):
        r = PC.c1_move_layers(hw3(), self.m, self.s, src=1, dst=0, n=4)
        before = PC.c1_layer_split(hw3(), self.m, self.s)
        after = PC.c1_layer_split(hw3(), self.m, dict(self.s, stage_layers=r["after_layers"]))
        for key in ("weights", "kv", "state"):
            tb = sum(s["terms"][key]["v"] for s in before["stages"])
            ta = sum(s["terms"][key]["v"] for s in after["stages"])
            self.assertAlmostEqual(tb, ta, delta=0.05, msg=key)
        self.assertEqual(sum(r["after_layers"]), sum(r["before_layers"]))

    def test_source_loses_and_destination_gains_weight(self):
        r = PC.c1_move_layers(hw3(), self.m, self.s, src=1, dst=0, n=2)
        row_src, row_dst = r["rows"][1], r["rows"][0]
        self.assertLess(row_src["weights_mib"][1], row_src["weights_mib"][0])
        self.assertGreater(row_dst["weights_mib"][1], row_dst["weights_mib"][0])
        self.assertGreater(row_src["free_mib"][1], row_src["free_mib"][0])
        self.assertLess(row_dst["free_mib"][1], row_dst["free_mib"][0])
        self.assertTrue(any("Du verschiebst 2 Layer" in h for h in r["hints"]))

    def test_non_adjacent_stages_and_too_many_layers_are_refused(self):
        with self.assertRaises(PC.CouplingError):
            PC.c1_move_layers(hw3(), self.m, self.s, src=0, dst=2, n=1)
        with self.assertRaises(PC.CouplingError):
            PC.c1_move_layers(hw3(), self.m, self.s, src=1, dst=0, n=99)

    def test_overflow_is_shown_never_clipped(self):
        tight = dict(self.s, budget_mib=[30000, 3000, 3000])
        out = PC.compute(hw3(), self.m, tight)
        over = [s for s in out["c1"]["stages"] if s["overflow_mib"] > 0]
        self.assertTrue(over, "3000 MiB Budget tragen 16 Layer des 27B nicht")
        for s in over:
            self.assertLess(s["free_mib"], 0.0)
            self.assertAlmostEqual(s["overflow_mib"], -s["free_mib"], delta=0.01)
        self.assertTrue(any("ueber dem Budget" in h for h in out["hints"]))
        self.assertTrue(out["c1"]["overflow_cards"])


class TestC2ExpertResidency(unittest.TestCase):
    def test_largest_fraction_fits_and_a_larger_one_does_not(self):
        m = model("nextflash_int4mixed")
        r = PC.c2_expert_residency(hw3(), m, rec_settings(scratch_rows=32, moe_resident_fraction=[0.33, 0.701, 0.9956]))
        self.assertTrue(r["applicable"])
        for row in r["rows"]:
            self.assertGreaterEqual(row["fraction_max"], 0.0)
            self.assertLessEqual(row["fraction_max"], 1.0)
        # eine gesetzte Fraction oberhalb der Obergrenze wird als nicht passend gemeldet, mit Hinweis
        r2 = PC.c2_expert_residency(hw3(), m, rec_settings(scratch_rows=32, moe_resident_fraction=1.0, budget_mib=[10000, 6000, 6000]))
        self.assertTrue(any(not row["fits"] for row in r2["rows"]))
        self.assertTrue(any("zu gross" in h or "nicht einmal" in h for h in r2["hints"]))

    def test_more_kv_budget_pressure_means_a_smaller_fraction(self):
        m = model("nextflash_int4mixed")
        small = PC.c2_expert_residency(hw3(), m, rec_settings(scratch_rows=32, context_tokens=65536))
        big = PC.c2_expert_residency(hw3(), m, rec_settings(scratch_rows=32, context_tokens=524288))
        for a, b in zip(small["rows"], big["rows"]):
            self.assertGreaterEqual(a["fraction_max"], b["fraction_max"])

    def test_dense_model_has_no_expert_coupling(self):
        r = PC.c2_expert_residency(hw3(), model("qwen27b_int8_vocabembed"), {"stage_layers": [32, 16, 16]})
        self.assertFalse(r["applicable"])


class TestC3ChunkAndC4Context(unittest.TestCase):
    def test_a_bigger_chunk_costs_activation_and_kv_capacity(self):
        m = model("qwen27b_int8_vocabembed")
        s = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "chunk_tokens": 2048, "context_tokens": 100000}
        r = PC.c3_chunk(hw3(), m, s, new_chunk_tokens=8192)
        rate = float(m["activation"]["extend_rate_mib_per_row"]["v"])
        for row in r["rows"]:
            self.assertAlmostEqual(row["activation_mib"][1] - row["activation_mib"][0], (8192 - 2048) * rate, delta=0.05)
            self.assertLess(row["kv_capacity_tokens"][1], row["kv_capacity_tokens"][0])
        self.assertTrue(any("Chunk 2048 -> 8192" in h for h in r["hints"]))

    def test_unknown_extend_rate_is_named_not_priced(self):
        m = copy.deepcopy(model("qwen27b_int8_vocabembed"))
        m["activation"]["extend_rate_mib_per_row"] = {"v": None, "src": "nicht gemessen"}
        r = PC.c3_chunk(hw3(), m, {"stage_layers": [32, 16, 16]}, new_chunk_tokens=4096)
        self.assertTrue(any("unbekannt" in h for h in r["hints"]))

    def test_context_target_fits_or_names_what_is_missing(self):
        m = model("qwen27b_int8_vocabembed")
        ok = PC.c4_context_target(hw3(), m, {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 4096})
        self.assertTrue(ok["fits"])
        no = PC.c4_context_target(hw3(), m, {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 4000000,
                                              "budget_mib": [31000, 19000, 19000]})
        self.assertFalse(no["fits"])
        self.assertTrue(any("es fehlen" in h for h in no["hints"]))
        for row in no["rows"]:
            self.assertEqual(row["fits"], row["missing_mib"] <= 0.0)

    def test_context_floor_is_the_minimum_over_stages(self):
        out = PC.c1_layer_split(hw3(), model("qwen27b_int8_vocabembed"), {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3"})
        caps = [s["kv_capacity_tokens"] for s in out["stages"] if s["kv_capacity_tokens"] is not None]
        self.assertEqual(out["context_floor_tokens"], min(caps))


class TestInventories(unittest.TestCase):
    """Synthetische Inventare: 3, 2 und 1 Karte; sm86 / sm89 / sm120 (Name + Speichergroesse; keine Rig-Konstante im Modul)."""

    CASES = {
        "3 Karten Referenz": ([("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)], [32, 16, 16]),
        "2 Karten sm89": ([("NVIDIA GeForce RTX 4090", 24564, 900.0), ("NVIDIA GeForce RTX 4090", 24564, 900.0)], [32, 32]),
        "1 Karte sm120": ([("NVIDIA RTX PRO 6000 Blackwell", 97887, 1500.0)], [64]),
        "2 Karten sm86": ([("NVIDIA GeForce RTX 3090", 24576, 850.0), ("NVIDIA RTX A6000", 49140, 700.0)], [20, 44]),
    }

    def test_every_inventory_gets_a_plan_or_a_named_refusal(self):
        m = model("qwen27b_int8_vocabembed")
        for name, (cards, layers) in self.CASES.items():
            out = PC.compute(PC.synthetic_hardware(cards), m, {"stage_layers": layers, "kv_dtype": "fp8_e4m3", "context_tokens": 32768,
                                                              "chunk_tokens": 2048})
            self.assertEqual(len(out["c1"]["stages"]), len(cards), name)
            self.assertEqual(out["schema"], PC.SCHEMA)

    def test_wrong_vector_length_is_refused_by_name(self):
        m = model("qwen27b_int8_vocabembed")
        with self.assertRaises(PC.CouplingError) as cm:
            PC.compute(hw3(), m, {"stage_layers": [32, 32]})
        self.assertIn("vector_length", str(cm.exception))
        with self.assertRaises(PC.CouplingError):
            PC.compute(hw3(), m, {"stage_layers": [32, 16, 16], "budget_mib": [1, 2]})
        with self.assertRaises(PC.CouplingError):
            PC.compute(hw3(), m, {"stage_layers": [32, 16, 17]})      # summiert nicht auf n_layers

    def test_missing_gemv_rate_leaves_the_stage_time_unmeasured(self):
        m = model("qwen27b_int8_vocabembed")
        hw = PC.synthetic_hardware([("NVIDIA GeForce RTX 4090", 24564, None), ("NVIDIA GeForce RTX 4090", 24564, 900.0)])
        out = PC.c1_layer_split(hw, m, {"stage_layers": [32, 32], "kv_dtype": "fp8_e4m3"})
        self.assertIsNone(out["stages"][0]["decode_ms"])
        self.assertIn("nicht gemessen", out["stages"][0]["decode_src"])
        self.assertIsNone(out["makespan_ms"])

    def test_stage_time_follows_the_slower_card(self):
        m = model("qwen27b_int8_vocabembed")
        fast = PC.synthetic_hardware([("A", 40000, 1000.0), ("B", 40000, 1000.0)])
        slow = PC.synthetic_hardware([("A", 40000, 1000.0), ("B", 40000, 250.0)])
        st = {"stage_layers": [32, 32], "kv_dtype": "fp8_e4m3"}
        self.assertGreater(PC.c1_layer_split(slow, m, st)["makespan_ms"], PC.c1_layer_split(fast, m, st)["makespan_ms"])


class TestEdgesAndSettings(unittest.TestCase):
    def test_c5_to_c7_are_named_edges_without_a_calculation(self):
        self.assertEqual([e["id"] for e in PC.EDGES], ["C5", "C6", "C7"])
        for e in PC.EDGES:
            self.assertFalse(e["computed"])
            self.assertTrue(e["text"] and e["touches"] and e["now"])
        out = PC.compute(hw3(), model("qwen27b_int8_vocabembed"), {"stage_layers": [32, 16, 16]})
        self.assertEqual([e["id"] for e in out["edges"]], ["C5", "C6", "C7"])

    def test_settings_from_the_recorded_argv(self):
        args = {"--pp-stage-ratio": "29,11,8", "--pp-attn-stage-ratio": "7,3,2", "--rank-gpu-memory-mib": "28440,17568,18200",
                "--chunked-prefill-size": "16384", "--max-kv-per-request": "262144", "--kv-cache-dtype": "fp8_e4m3",
                "--rank-moe-resident-fraction": "0.33,0.701,0.995605"}
        s = PC.settings_from_server(args, model("nextflash_int4mixed"))
        self.assertEqual(s["stage_layers"], [29, 11, 8])
        self.assertEqual(s["attn_layers"], [7, 3, 2])
        self.assertEqual(s["budget_mib"], [28440.0, 17568.0, 18200.0])
        self.assertEqual(s["chunk_tokens"], 16384)
        self.assertEqual(s["context_tokens"], 262144)
        self.assertEqual(s["kv_dtype"], "fp8_e4m3")
        self.assertEqual(s["moe_resident_fraction"], [0.33, 0.701, 0.995605])
        # das Ergebnis rechnet durch, und die gepinnte Attention-Zaehlung gilt
        out = PC.compute(hw3(), model("nextflash_int4mixed"), s)
        self.assertEqual([x["attn_layers"] for x in out["c1"]["stages"]], [7, 3, 2])

    def test_pinned_attention_must_sum_to_the_models_attention_layers(self):
        with self.assertRaises(PC.CouplingError):
            PC.compute(hw3(), model("nextflash_int4mixed"), {"stage_layers": [29, 11, 8], "attn_layers": [7, 3, 3]})


class TestBridgeCall(unittest.TestCase):
    """``run`` / ``python -m`` : das Gegenstueck zum Dashboard-Kindprozess (kartenplan_build/runner.py, op ``couplings``)."""

    def test_run_answers_json_and_never_crashes_on_bad_input(self):
        req = {"what": "compute", "hardware": hw3(), "model": model("qwen27b_int8_vocabembed"), "settings": {"stage_layers": [32, 16, 16]}}
        ok = PC.run(req)
        self.assertTrue(ok["ok"])
        json.dumps(ok)       # serialisierbar
        bad = PC.run(dict(req, settings={"stage_layers": [32, 32]}))
        self.assertFalse(bad["ok"])
        self.assertIn("vector_length", bad["error"])
        self.assertFalse(PC.run(dict(req, what="nope"))["ok"])
        mv = PC.run(dict(req, what="move", src=1, dst=0, n=1))
        self.assertTrue(mv["ok"] and mv["result"]["move"] == {"src": 1, "dst": 0, "n": 1})

    def test_server_args_feed_the_settings(self):
        req = {"what": "context", "hardware": hw3(), "model": model("nextflash_int4mixed"),
               "server_args": {"--pp-stage-ratio": "29,11,8", "--max-kv-per-request": "262144", "--kv-cache-dtype": "fp8_e4m3"}}
        res = PC.run(req)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["result"]["context_tokens"], 262144)

    def test_module_main_roundtrip(self):
        req = {"what": "compute", "hardware": hw3(), "model": model("qwen27b_int8_vocabembed"), "settings": {"stage_layers": [32, 16, 16]}}
        repo = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
        env = dict(os.environ, PYTHONPATH=os.path.join(repo, "python"), CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, "-m", "sglang.srt.planner.profile_couplings"], input=json.dumps(req), capture_output=True, text=True,
                           env=env, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr[-400:])
        self.assertTrue(json.loads(p.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
