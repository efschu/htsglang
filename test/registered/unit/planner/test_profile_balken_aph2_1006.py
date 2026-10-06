"""AP-H2 (Auftrag 880, Plan 1006 §3 AP-H Teil 2): Balken je Karte und Phase im Datenvertrag ``flliper.balken/1`` (GPU-frei).

Gepinnt:

* Summenregel des Vertrags: Segmente zusammenhaengend in logischer Reihenfolge, Summe = max(Kartengroesse, Posten); Reserve und Frei folgen aus
  Budget und Posten; ein Ueberlauf ueber das Budget zehrt die Reserve auf, erst ein Posten ueber der KARTE laesst den Balken ueber die
  Kartengrenze wachsen (``beyond_card_mib``), nichts wird abgeschnitten;
* D-Phase (neu): Handrechnung gegen das Modellprofil (Anteile aus ``--rank-tp-ratio``, Besitz aus ``--rank-moe-ratio``, Pufferregel aus FR_D und
  Scratch), Festposten aus ``--d-foreign-context-mib`` + ``--d-nontorch-mib``;
* Draft-Term (neu): P traegt den MTP-Kopf nur ohne ``--draft-kv-on-p off``, D solo nur auf dem Host-Rang, Gewicht aus dem Draft-Profil
  (``model["draft"]["external"]``), DFlash2 = nicht gerechnet;
* nicht belegbare Terme sind ``mib: None`` ("nicht gerechnet") mit Grund, nie 0 und nie geraten; ``Frei`` nennt dann die Obergrenze;
* Betriebsformen: Einzelkarte (eine Phase), nur TP, Flip, Dual (beide Phasen, keine Summe); eine unrechenbare Phase laesst die andere stehen.
"""

import copy
import json
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import expert_residency as ER  # noqa: E402
from sglang.srt.planner import profile_couplings as PC  # noqa: E402
from sglang.srt.weg2 import model_profile as MP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "weg2", "fixtures", "profil_s3_1003")
MIB = 1024.0 * 1024.0
RIG3 = [("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)]
ORDER = ["weights", "experts", "draft", "kv", "state", "activation", "fixed", "reserve", "free"]
_M = {}


def model(name):
    if name not in _M:
        _M[name] = MP.estimate(os.path.join(FX, name))
    return _M[name]


def hw(cards=RIG3):
    return PC.synthetic_hardware(cards)


# Profilzeilen der NF-abl-Form (docker/profiles/nf-int4-h6-abl.env), auf die hier relevanten Flags gekuerzt
NF_ARGS = {"--pp-stage-ratio": "29,11,8", "--pp-attn-stage-ratio": "7,3,2", "--pp-cut-expert-device-fraction": "0.330,0.701,0.652",
           "--pp-cut-expert-lru-rows": "32,32,32", "--max-kv-per-request": "262144", "--d-foreign-context-mib": "1446,896,894",
           "--d-nontorch-mib": "1981,528,524", "--draft-kv-on-p": "off", "--d-kv-token-cut": "owned",
           "--profile": "nextflash", "--user-reserve-mib": "0,0,0"}
NF_PA = {"P": {"--kv-cache-dtype": "fp8_e4m3", "--max-mamba-cache-size": "32", "--mamba-ssm-dtype": "bfloat16", "--chunked-prefill-size": "16384"},
         "D": {"--rank-role": "host,worker,worker", "--rank-tp-ratio": "1,0,0", "--rank-moe-ratio": "183,137,168",
               "--rank-moe-resident-fraction": "0.06,0.51,0.48", "--speculative-draft-placement": "solo", "--speculative-algorithm": "NEXTN",
               "--kv-cache-dtype": "fp8_e4m3", "--max-mamba-cache-size": "32", "--mamba-ssm-dtype": "bfloat16"}}
NF_PE = {"P": {"SGLANG_MOE_SCRATCH_SLOTS": "32"}, "D": {"SGLANG_MOE_SCRATCH_SLOTS": "104,48,48", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION": "0.06,0.51,0.48"}}


def nf(**kw):
    a = dict(kw.pop("args", NF_ARGS))
    return PC.phase_bars(hw(), kw.pop("m", model("nextflash_int4mixed")), a, kw.pop("pa", NF_PA), kw.pop("pe", NF_PE), **kw)


def seg(bar, name):
    return next((s for s in bar["segments"] if s["name"] == name), None)


def stage(total, budget, **posts):
    """Eine Vertrags-Stufe von Hand: posts = {name: mib | None}."""
    terms = {k: {"v": v, "src": "Eingabe", "note": ""} for k, v in posts.items()}
    return {"ord": 0, "label": "Karte 0", "total_mib": float(total), "budget_mib": {"v": float(budget), "src": "Profilzeile", "note": "--rank-gpu-memory-mib"},
            "terms": terms}


class TestContractSumRule(unittest.TestCase):
    def test_fits_inside_the_budget_the_segments_fill_exactly_the_card(self):
        b = PC.contract_bar(stage(1000, 900, weights=300, kv=200), "P")
        self.assertEqual([s["name"] for s in b["segments"]], ["weights", "kv", "reserve", "free"])
        self.assertEqual((seg(b, "reserve")["mib"], seg(b, "free")["mib"]), (100.0, 400.0))
        self.assertAlmostEqual(sum(s["mib"] for s in b["segments"]), 1000.0, places=6)
        self.assertEqual((b["overflow_mib"], b["beyond_card_mib"], b["over_text"]), (0.0, 0.0, ""))

    def test_over_the_budget_eats_the_reserve_but_not_the_card(self):
        b = PC.contract_bar(stage(1000, 900, weights=600, kv=350), "P")          # 950 Posten: 50 ueber dem Budget, 50 von 100 Reserve
        self.assertEqual(b["overflow_mib"], 50.0)
        self.assertEqual(b["beyond_card_mib"], 0.0)
        self.assertEqual(seg(b, "reserve")["mib"], 50.0)
        self.assertIsNone(seg(b, "free"))
        self.assertAlmostEqual(sum(s["mib"] for s in b["segments"]), 1000.0, places=6)
        self.assertIn("ueber dem Budget", b["over_text"])
        self.assertIn("aufgezehrt", seg(b, "reserve")["detail"])

    def test_over_the_card_the_bar_grows_past_the_edge_and_nothing_is_clipped(self):
        b = PC.contract_bar(stage(1000, 900, weights=700, kv=500), "D")          # 1200 Posten auf einer 1000er Karte
        self.assertEqual(b["beyond_card_mib"], 200.0)
        self.assertEqual(b["overflow_mib"], 300.0)
        self.assertEqual([s["name"] for s in b["segments"]], ["weights", "kv"])
        self.assertEqual(sum(s["mib"] for s in b["segments"]), 1200.0)          # Summe = Posten > Kartengroesse
        self.assertIn("ueber der KARTE", b["over_text"])

    def test_not_computed_terms_are_null_not_zero_and_bound_the_free_figure(self):
        b = PC.contract_bar(stage(1000, 900, weights=300, kv=None, fixed=None), "P")
        self.assertEqual(b["not_computed"], ["KV", "Festposten"])
        kv = seg(b, "kv")
        self.assertIsNone(kv["mib"])
        self.assertEqual((kv["herkunft"], kv["gerechnet"]), ("nicht gerechnet", False))
        self.assertEqual(b["posts_mib"], 300.0)
        self.assertIn("OBERGRENZE", seg(b, "free")["detail"])
        self.assertIn("KV", seg(b, "free")["detail"])

    def test_zero_terms_are_left_out_and_the_order_is_the_contract_order(self):
        b = PC.contract_bar(stage(1000, 900, fixed=5, kv=7, weights=3, draft=0, state=9, experts=11, activation=13), "P")
        self.assertEqual([s["name"] for s in b["segments"]], ["weights", "experts", "kv", "state", "activation", "fixed", "reserve", "free"])
        self.assertEqual([k for k in ORDER if k in {s["name"] for s in b["segments"]}], [s["name"] for s in b["segments"]])


class TestEveryBarOfTheReferenceRig(unittest.TestCase):
    def setUp(self):
        self.res = nf()

    def test_schema_form_and_phases(self):
        self.assertEqual(self.res["schema"], "flliper.balken/1")
        self.assertEqual((self.res["form"], list(self.res["phases"])), ("flip", ["P", "D"]))
        self.assertTrue(all(ph["ok"] for ph in self.res["phases"].values()))
        json.dumps(self.res)

    def test_sum_rule_and_order_hold_for_every_card_and_phase(self):
        for name, ph in self.res["phases"].items():
            self.assertEqual(len(ph["bars"]), 3, name)
            for b in ph["bars"]:
                names = [s["name"] for s in b["segments"]]
                self.assertEqual([k for k in ORDER if k in names], names, (name, b["label"]))
                got = sum(s["mib"] for s in b["segments"] if s["mib"] is not None)
                self.assertAlmostEqual(got, max(b["total_mib"], b["posts_mib"]), delta=0.01, msg=(name, b["label"]))
                for s in b["segments"]:
                    self.assertTrue(s["herkunft"], (name, b["label"], s["name"]))
                    self.assertTrue(s["detail"], (name, b["label"], s["name"]))

    def test_unmeasured_fixed_posts_are_not_computed_in_p_not_zero(self):
        for b in self.res["phases"]["P"]["bars"]:
            self.assertIsNone(seg(b, "fixed")["mib"])
            self.assertIn("Festposten", b["not_computed"])

    def test_the_geometry_overshoot_on_a_3080_is_shown_with_the_reserve_eaten(self):
        # ohne gemessene Aktivierungsspitze rechnet die Engine die Geometrie (benannte Grenze, Test 1432): Karte 1 laeuft ueber das Budget
        b = self.res["phases"]["P"]["bars"][1]
        self.assertGreater(b["overflow_mib"], 0)
        self.assertEqual(b["beyond_card_mib"], 0.0)
        self.assertTrue(any("Budget" in h for h in self.res["hints"]))

    def test_p_inputs_name_the_profile_lines_they_came_from(self):
        seen = {x["was"]: x for x in self.res["phases"]["P"]["inputs"]}
        self.assertIn("--pp-stage-ratio", seen["stage_layers"]["herkunft"])
        self.assertIn("--pp-cut-expert-device-fraction", seen["moe_resident_fraction"]["herkunft"])
        self.assertIn("--pp-cut-expert-lru-rows", seen["scratch_rows"]["herkunft"])
        self.assertIn("Annahme", seen["budget_mib"]["herkunft"])


class TestDPhaseHandCalculation(unittest.TestCase):
    def test_dense_model_even_tp_splits_weights_state_and_kv_by_rank(self):
        m = copy.deepcopy(model("qwen27b_int8_vocabembed"))
        m["weights"]["mtp_bytes"] = {"v": 0, "src": "Index"}          # ohne Draft: der KV-Anteil ist rein der des Ziels
        w = m["weights"]
        dense = (sum(w["layer_bytes"]["v"]) + w["embed_bytes"]["v"] + w["lm_head_bytes"]["v"]) / MIB
        res = PC.phase_bars(hw([("A", 32607, 1000.0), ("A", 32607, 1000.0)]), m, {"--max-kv-per-request": "65536"},
                            {"P": {}, "D": {}}, {}, form="d_only")
        self.assertEqual(list(res["phases"]), ["D"])
        for b in res["phases"]["D"]["bars"]:
            self.assertAlmostEqual(seg(b, "weights")["mib"], dense / 2, delta=0.01)
            self.assertIsNone(seg(b, "experts"))                       # dichtes Modell
            self.assertIn("gleichmaessiger TP", seg(b, "weights")["detail"])
        attn = sum(1 for f in m["arch"]["layer_families"]["v"] if f == "attn")
        cell = m["kv"]["cell_bytes_per_attn_layer_token"]["v"]
        if m["arch"]["heads_kv"]["v"] % 2 == 0:
            self.assertAlmostEqual(seg(res["phases"]["D"]["bars"][0], "kv")["mib"], 65536 * attn * cell / MIB / 2, delta=0.01)
        else:
            self.assertIsNone(seg(res["phases"]["D"]["bars"][0], "kv")["mib"])

    def test_uneven_tp_and_moe_ratio_on_the_reference_form(self):
        m = model("nextflash_int4mixed")
        w = m["weights"]
        E = m["experts"]["n"]["v"]
        dense = (sum(w["layer_bytes"]["v"]) + w["embed_bytes"]["v"] + w["lm_head_bytes"]["v"]) / MIB
        d = nf()["phases"]["D"]["bars"]
        # --rank-tp-ratio 1,0,0: der Host traegt alle dichten Gewichte, die Worker keine
        self.assertAlmostEqual(seg(d[0], "weights")["mib"], dense, delta=0.01)
        self.assertIsNone(seg(d[1], "weights"))
        self.assertIsNone(seg(d[2], "weights"))
        # Experten: Besitz je Rang aus --rank-moe-ratio, Zeilen aus der Pufferregel (FR_D, Scratch), MiB je Zeile ueber alle Layer
        per_expert = sum(w["layer_expert_bytes"]["v"]) / E / MIB
        own = [183, 137, 168] if sum([183, 137, 168]) == E else [int(round(E * r / 488.0)) for r in (183, 137, 168)]
        for i, (fr, sc) in enumerate(zip((0.06, 0.51, 0.48), (104, 48, 48))):
            rows = ER.buffer_rows(local_experts=own[i], fraction=fr, scratch_rows=sc)
            self.assertAlmostEqual(seg(d[i], "experts")["mib"], rows * per_expert, delta=0.01, msg="Rang %d" % i)
            self.assertEqual(seg(d[i], "experts")["herkunft"], "Naeherung (nicht der Loeser)")

    def test_fixed_post_is_the_sum_of_the_two_profile_lines(self):
        d = nf()["phases"]["D"]["bars"]
        self.assertEqual([seg(b, "fixed")["mib"] for b in d], [1446 + 1981, 896 + 528, 894 + 524])
        self.assertEqual(seg(d[0], "fixed")["herkunft"], "Profilzeile")
        self.assertIn("--d-foreign-context-mib", seg(d[0], "fixed")["detail"])
        self.assertIn("--d-nontorch-mib", seg(d[0], "fixed")["detail"])

    def test_fixed_posts_sit_outside_the_budget_exactly_as_in_the_launcher(self):
        # Befund 1 (Fix-Runde 1): Launcher: verfuegbar = Karte - fremd - nichttorch - reserve (pp_cut.d_rank_available_mib), gefragt = Budget.
        # Budget = Karte - fremd - nichttorch -> die Festposten stehen EINMAL im Balken (nicht noch einmal in der Reserve), Frei = Budget - Posten.
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-gpu-memory-mib"] = "29180,19056,19062"
        d = nf(pa=pa)["phases"]["D"]["bars"]
        for b, fx, tot in zip(d, (3427, 1424, 1418), (32607, 20480, 20480)):
            self.assertTrue(seg(b, "fixed")["ausserhalb_budget"])
            self.assertEqual(b["outside_budget_mib"], fx)
            self.assertEqual(b["available_mib"], tot - fx)
            self.assertIsNone(seg(b, "reserve"))                                   # Budget == verfuegbar: keine Reserve, Festposten nicht doppelt
            inside = b["posts_mib"] - fx
            self.assertAlmostEqual(seg(b, "free")["mib"], (tot - fx) - inside, delta=0.01)       # nach Launcher-Semantik, nicht um fx zu klein
            self.assertAlmostEqual(sum(s["mib"] for s in b["segments"] if s["mib"] is not None), max(tot, b["posts_mib"]), delta=0.01)
            self.assertEqual(b["budget_over_available_mib"], 0)

    def test_budget_larger_than_the_available_is_named_not_hidden(self):
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-gpu-memory-mib"] = "31583,19456,19456"
        b = nf(pa=pa)["phases"]["D"]["bars"][0]
        self.assertAlmostEqual(b["budget_over_available_mib"], 31583 - (32607 - 3427), delta=0.01)
        self.assertIn("groesser als das Verfuegbare", b["over_text"])
        self.assertEqual(b["beyond_card_mib"], 0)

    def test_contract_bar_by_hand_outside_posts_shrink_available_not_free(self):
        st = stage(1000, 600, weights=300, kv=100)
        st["terms"]["fixed"] = {"v": 150.0, "src": "Profilzeile", "note": "", "outside_budget": True}
        b = PC.contract_bar(st, "D")
        self.assertEqual([(s["name"], s["mib"]) for s in b["segments"]], [("weights", 300), ("kv", 100), ("fixed", 150), ("reserve", 250), ("free", 200)])
        self.assertEqual((b["overflow_mib"], b["beyond_card_mib"], b["budget_over_available_mib"]), (0, 0, 0))
        st2 = stage(1000, 600, weights=500, kv=200)                                 # 700 im Budget > 600
        st2["terms"]["fixed"] = {"v": 150.0, "src": "Profilzeile", "note": "", "outside_budget": True}
        b2 = PC.contract_bar(st2, "D")
        self.assertEqual(b2["overflow_mib"], 100)
        self.assertEqual(seg(b2, "reserve")["mib"], 150)                              # 850 verfuegbar - 700 Posten
        self.assertEqual(sum(s["mib"] for s in b2["segments"]), 1000)

    def test_d_budget_defaults_to_card_minus_corridor_and_takes_the_group_flag(self):
        d = nf()["phases"]["D"]["bars"]
        # Festposten (fremd + nichttorch) liegen ausserhalb des Budgets (Launcher: verfuegbar = Karte - fremd - nichttorch - reserve)
        # Korridor = 1024 stated law + 404 eingebauter Wach-Ueberschuss (launcher.py:266-269), auf 8 MiB abgerundet (launcher.py:15394)
        # Fix-Runde 5: Launcher-Formel MIT den Records des Profils nextflash (--profile nextflash, --user-reserve-mib 0,0,0):
        #   Karte 0: 32607 - Boden 1024 (Wach-Rest-Record 2672 ersetzt die 404) - dc 3427 - Wachstum 482 - Wach-Rest 2672 = 25002 -> 25000
        #   Karte 1: 20480 - (1024 + 404) - dc 1424 - Wachstum 572 = 17056;  Karte 2: 20480 - 1428 - dc 1418 - Wachstum 384 = 17250 -> 17248
        self.assertEqual([b["budget_mib"] for b in d], [25000, 17056, 17248])
        self.assertEqual(d[0]["budget_herkunft"], "gerechnet")
        for b in d:
            self.assertEqual(b["budget_over_available_mib"], 0)
            self.assertAlmostEqual(seg(b, "reserve")["mib"], b["available_mib"] - b["budget_mib"], places=2)
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-gpu-memory-mib"] = "28000,17000,17000"
        d2 = nf(pa=pa)["phases"]["D"]["bars"]
        self.assertEqual([b["budget_mib"] for b in d2], [28000, 17000, 17000])
        self.assertEqual(d2[0]["budget_herkunft"], "Profilzeile")
        self.assertEqual(seg(d2[0], "reserve")["mib"], 32607 - 3427 - 28000)          # verfuegbar - Budget

    def test_unsolvable_assignments_are_not_computed_and_say_why(self):
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-tp-ratio"] = "auto"
        pa["D"]["--rank-moe-ratio"] = "link"
        d = nf(pa=pa)["phases"]["D"]["bars"]
        for b in d:
            self.assertIsNone(seg(b, "weights")["mib"])
            self.assertIn("loest der Launcher", seg(b, "weights")["detail"])
            self.assertIsNone(seg(b, "experts")["mib"])
            self.assertIn("link", seg(b, "experts")["detail"])

    def test_kv_is_only_computed_where_the_distribution_is_known(self):
        d = nf()["phases"]["D"]["bars"]                               # ungleicher TP + Token-Schnitt: nicht gerechnet
        for b in d:
            self.assertIsNone(seg(b, "kv")["mib"])
            self.assertIn("--d-kv-token-cut", seg(b, "kv")["detail"])
        a = dict(NF_ARGS)
        a.pop("--d-kv-token-cut")
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-kv-ratio"] = "2,1,1"                            # ausdruecklicher Token-Eigentumsvektor: Anteile belegt
        pa["D"].pop("--speculative-algorithm")
        m = copy.deepcopy(model("nextflash_int4mixed"))
        m["weights"]["mtp_bytes"] = {"v": 0, "src": "Index"}          # ohne Draft: keine Draft-KV-Zeile auf dem Host
        kvs = [seg(b, "kv")["mib"] for b in nf(m=m, args=a, pa=pa)["phases"]["D"]["bars"]]
        self.assertAlmostEqual(kvs[0] / kvs[1], 2.0, places=3)
        self.assertAlmostEqual(kvs[1], kvs[2], places=6)
        # der Gesamtpreis steht als Hinweis, auch wenn die Verteilung nicht gerechnet ist
        self.assertTrue(any("KV gesamt" in h for h in nf()["hints"]))


class TestDraftTerm(unittest.TestCase):
    def ext_model(self, p_mib=500.0, d_mib=300.0):
        m = copy.deepcopy(model("nextflash_int4mixed"))
        m["draft"]["external"] = {"bytes_without_lm_head": {"v": int(p_mib * MIB), "src": "Index"},
                                  "bytes_without_embed_lm_head": {"v": int(d_mib * MIB), "src": "Index"}, "kv": {"attn_layers": {"v": 1, "src": "Index"}}}
        return m

    def test_p_carries_the_head_only_without_draft_kv_on_p_off(self):
        m = self.ext_model()
        off = nf(m=m)["phases"]["P"]["bars"]
        self.assertTrue(all(seg(b, "draft") is None for b in off))
        a = dict(NF_ARGS)
        a["--draft-kv-on-p"] = "on"
        on = nf(m=m, args=a)["phases"]["P"]["bars"]
        self.assertIsNone(seg(on[0], "draft"))
        self.assertIsNone(seg(on[1], "draft"))
        d = seg(on[2], "draft")                                        # letzte Stufe, Gewicht ohne lm_head aus dem Draft-Profil
        self.assertAlmostEqual(d["mib"], 500.0, delta=0.01)
        self.assertEqual(d["herkunft"], "Modellprofil/Hardwareprofil (Index)")
        self.assertIn("lm_head", d["detail"])

    def test_p_default_is_on_when_the_flag_is_absent(self):
        a = dict(NF_ARGS)
        a.pop("--draft-kv-on-p")
        on = nf(m=self.ext_model(), args=a)["phases"]["P"]["bars"]
        self.assertAlmostEqual(seg(on[2], "draft")["mib"], 500.0, delta=0.01)

    def test_d_solo_lands_on_the_host_rank_only_with_the_shared_embedding_left_out(self):
        d = nf(m=self.ext_model())["phases"]["D"]["bars"]
        self.assertAlmostEqual(seg(d[0], "draft")["mib"], 300.0, delta=0.01)
        self.assertIsNone(seg(d[1], "draft"))
        self.assertIsNone(seg(d[2], "draft"))
        self.assertIn("solo auf Rang 0", seg(d[0], "draft")["detail"])
        self.assertIn("Einbettung und lm_head", seg(d[0], "draft")["detail"])

    def test_d_solo_draft_adds_one_kv_row_on_the_host_only(self):
        a = dict(NF_ARGS)
        a.pop("--d-kv-token-cut")
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-kv-ratio"] = "1,1,1"
        with_d = [seg(b, "kv")["mib"] for b in nf(m=self.ext_model(), args=a, pa=pa)["phases"]["D"]["bars"]]
        m0 = copy.deepcopy(model("nextflash_int4mixed"))
        m0["weights"]["mtp_bytes"] = {"v": 0, "src": "Index"}
        pa0 = copy.deepcopy(pa)
        pa0["D"].pop("--speculative-algorithm")
        without = [seg(b, "kv")["mib"] for b in nf(m=m0, args=a, pa=pa0)["phases"]["D"]["bars"]]
        cell = 1088.0 / MIB                                             # fp8_e4m3: 1088 B je Attention-Layer und Token (Metall fnFL2w123)
        self.assertAlmostEqual(with_d[0] - without[0], 262144 * 1 * cell, delta=0.01)       # eine Draft-Attention-Zeile (aus dem Draft-Profil)
        self.assertAlmostEqual(with_d[1], without[1], places=6)
        self.assertAlmostEqual(with_d[2], without[2], places=6)

    def test_d_split_follows_the_tp_share(self):
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--speculative-draft-placement"] = "split"
        pa["D"]["--rank-tp-ratio"] = "2,1,1"
        d = nf(m=self.ext_model(), pa=pa)["phases"]["D"]["bars"]
        self.assertEqual([round(seg(b, "draft")["mib"], 3) for b in d], [150.0, 75.0, 75.0])

    def test_dflash_and_unprofiled_drafts_are_not_computed(self):
        a = dict(NF_ARGS)
        a["--dflash-draft-path"] = "/m/dflash"
        a["--draft-kv-on-p"] = "on"
        r = nf(args=a)
        self.assertEqual(r["draft"]["kind"], "dflash")
        self.assertIsNone(seg(r["phases"]["P"]["bars"][2], "draft")["mib"])
        self.assertIn("DFlash2", seg(r["phases"]["P"]["bars"][2], "draft")["detail"])
        self.assertTrue(all(seg(b, "draft")["mib"] is None for b in r["phases"]["D"]["bars"]))
        m = copy.deepcopy(model("nextflash_int4mixed"))
        m["weights"]["mtp_bytes"] = {"v": 0, "src": "Index"}
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--speculative-draft-model-path"] = "/m/draft"
        r2 = nf(m=m, pa=pa)
        self.assertEqual(r2["draft"]["kind"], "nextn")
        self.assertIsNone(seg(r2["phases"]["D"]["bars"][0], "draft")["mib"])
        self.assertIn("nicht profiliert", seg(r2["phases"]["D"]["bars"][0], "draft")["detail"])

    def test_no_spec_flag_and_no_mtp_head_means_no_draft(self):
        m = copy.deepcopy(model("nextflash_int4mixed"))
        m["weights"]["mtp_bytes"] = {"v": 0, "src": "Index"}
        pa = copy.deepcopy(NF_PA)
        pa["D"].pop("--speculative-algorithm")
        r = nf(m=m, pa=pa)
        self.assertEqual(r["draft"]["kind"], "none")
        for ph in r["phases"].values():
            self.assertTrue(all(seg(b, "draft") is None for b in ph["bars"]))

    def test_draft_mib_override_replaces_the_target_models_mtp_bytes(self):
        s = {"stage_layers": [29, 11, 8], "budget_mib": [28440, 17568, 18200], "kv_dtype": "fp8_e4m3", "draft": True, "draft_mib": 123.0,
             "draft_src": "Index", "draft_note": "aus dem Draft-Profil"}
        st = PC._stage_terms(hw(), model("nextflash_int4mixed"), s)["stages"]
        self.assertEqual(st[2]["terms"]["draft"]["v"], 123.0)
        self.assertEqual(st[2]["terms"]["draft"]["src"], "Index")
        self.assertEqual(st[0]["terms"]["draft"]["v"], 0.0)


class TestForms(unittest.TestCase):
    def test_detect_form(self):
        self.assertEqual(PC.detect_form({}, [], 1), "single")
        self.assertEqual(PC.detect_form({}, ["--d-only"], 3), "d_only")
        self.assertEqual(PC.detect_form({"--dual-share": ""}, [], 3), "dual")
        self.assertEqual(PC.detect_form({}, ["--dual-layout"], 3), "dual")
        self.assertEqual(PC.detect_form({}, [], 3), "flip")
        self.assertEqual(PC.detect_form({}, [], 3, "d_only"), "d_only")
        self.assertEqual(PC.detect_form({}, [], 3, "unsinn"), "flip")

    def test_single_card_is_one_phase_with_all_layers_on_the_card(self):
        m = model("qwen27b_int8_vocabembed")
        r = PC.phase_bars(hw([("NVIDIA GeForce RTX 5090", 32607, 1400.0)]), m, {"--max-kv-per-request": "32768"}, {}, {})
        self.assertEqual((r["form"], list(r["phases"])), ("single", ["alle"]))
        (b,) = r["phases"]["alle"]["bars"]
        self.assertEqual(b["phase"], "alle")
        self.assertIn("stage_layers", [x["was"] for x in r["phases"]["alle"]["inputs"]])
        self.assertIsNotNone(seg(b, "weights"))

    def test_dual_shows_both_phases_and_forms_no_sum(self):
        a = dict(NF_ARGS)
        a["--dual-share"] = ""
        r = nf(args=a)
        self.assertEqual((r["form"], list(r["phases"])), ("dual", ["P", "D"]))
        self.assertTrue(any("Summe beider Balken ist nicht gerechnet" in h for h in r["hints"]))
        self.assertNotIn("Summe", r["phases"])

    def test_one_unsolvable_phase_leaves_the_other_standing(self):
        a = dict(NF_ARGS)
        a.pop("--pp-stage-ratio")
        r = nf(args=a)
        self.assertFalse(r["phases"]["P"]["ok"])
        self.assertIn("--pp-stage-ratio fehlt", r["phases"]["P"]["error"])
        self.assertTrue(r["phases"]["D"]["ok"])
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-tp-ratio"] = "1,0"                                # falsche Vektorlaenge
        r2 = nf(pa=pa)
        self.assertTrue(r2["phases"]["P"]["ok"])
        self.assertIn("vector_length", r2["phases"]["D"]["error"])

    def test_user_overrides_apply_to_the_phase_settings(self):
        r = nf(overrides={"activation_mib": [5984, 2952, 2944], "ssm_dtype": "bfloat16"})
        for b, tr in zip(r["phases"]["P"]["bars"], (5984, 2952, 2944)):
            self.assertAlmostEqual(seg(b, "activation")["mib"], tr, delta=0.01)
            self.assertEqual(seg(b, "activation")["herkunft"], "Eingabe (Nutzer/Profil)")


class TestBridge(unittest.TestCase):
    def test_run_phase_bars_answers_the_contract_and_the_approx_payload(self):
        res = PC.run({"what": "phase_bars", "hardware": hw(), "model": model("nextflash_int4mixed"), "server_args": NF_ARGS,
                      "phase_args": NF_PA, "phase_env": NF_PE})
        self.assertTrue(res["ok"], res)
        r = res["result"]
        self.assertEqual(r["schema"], "flliper.balken/1")
        self.assertEqual(r["approx"]["stage_layers"], [29, 11, 8])
        self.assertEqual(r["approx"]["slots"], 32)
        json.dumps(res)

    def test_legacy_bars_and_server_args_are_unchanged(self):
        res = PC.run({"what": "bars", "hardware": hw(), "model": model("nextflash_int4mixed"), "server_args": {"--pp-stage-ratio": "29,11,8"}})
        self.assertTrue(res["ok"], res)
        self.assertEqual(list(res["result"]["phases"]), ["alle"])

    def test_bad_input_is_a_named_error_not_a_crash(self):
        res = PC.run({"what": "phase_bars", "hardware": {"cards": []}, "model": model("nextflash_int4mixed"), "server_args": {}})
        self.assertFalse(res["ok"])


# Dual-Referenzprofil (docker/profiles_release/27b-nvfp4-dual.env, 262k-Variante): P-Budgets 6610,5050,5200 und Mamba 8 Slots stehen in --extra-p
# (= Gruppe P, NICHT im gemeinsamen Profil: D darf P's Budgets nicht erben), Schnitt 31,17,16, Overhead 2500
DUAL_ARGS = {"--pp-stage-ratio": "31,17,16", "--pp-attn-stage-ratio": "7,5,4", "--max-kv-per-request": "262144", "--draft-kv-on-p": "off",
             "--dual-share": "", "--dual-p-overhead-mib": "2500",
             "--dual-unified-kv": "on", "--dual-p-kv-max-tokens": "196608", "--p-chunk-max": "1024",
             "--user-reserve-mib": "1800,1400,1400"}                      # vererbt aus 27b-base.env:78 (Release-Dual); --profile fehlt: Standard qwen27b
DUAL_PA = {"P": {"--rank-gpu-memory-mib": "6610,5050,5200", "--max-mamba-cache-size": "8"}}


def dual(**kw):
    a = dict(kw.pop("args", DUAL_ARGS))
    return PC.phase_bars(hw(), kw.pop("m", model("qwen27b_int8_vocabembed")), a, kw.pop("pa", copy.deepcopy(DUAL_PA)), {}, **kw)


class TestDualSharePhase(unittest.TestCase):
    """Fix-Runde 3, Befund 1: unter --dual-share zaehlen P's Gewichte NICHT gegen das P-Budget (Belege im Kommentar von _dual_share_p_terms)."""

    def test_the_release_dual_form_shows_no_overflow_and_keeps_weights_as_a_reference(self):
        r = dual()
        self.assertEqual(r["form"], "dual")
        p = r["phases"]["P"]
        for b, bud in zip(p["bars"], (6610, 5050, 5200)):
            self.assertEqual((b["overflow_mib"], b["beyond_card_mib"], b["budget_over_available_mib"], b["over_text"]), (0.0, 0.0, 0.0, ""))
            self.assertEqual(b["budget_mib"], bud)
            names = [x["name"] for x in b["segments"]]
            self.assertNotIn("weights", names)
            self.assertNotIn("experts", names)
            self.assertNotIn("kv", names)                                              # --dual-unified-kv on: Karten-KV-Pool
            refs = {x["name"]: x for x in b["shared_with_d"]}
            self.assertGreater(refs["weights"]["mib"], 1000)                           # die vollen Gewichte als Referenz (vorher gegen das Budget gezaehlt)
            self.assertEqual(refs["weights"]["ref"], "shared")
            self.assertGreater(refs["kv"]["mib"], 0)
            self.assertEqual(refs["activation"]["ref"], "in_festposten")
            self.assertIsNone(refs["diff"]["mib"])                                      # Diff nicht gerechnet, nie geraten
            self.assertIn("Diff der P-Gewichte", b["not_computed"])
            self.assertIn("OBERGRENZE", next(x for x in b["segments"] if x["name"] == "free")["detail"])
            # Summenregel unveraendert: Segmente = Karte, Referenz zaehlt nicht mit
            self.assertAlmostEqual(sum(x["mib"] for x in b["segments"]), b["total_mib"], places=2)
        self.assertIsNone(p["context_floor_tokens"])                                     # "Kontext-Boden 0 Token" war eine falsche Zahl

    def test_only_p_own_posts_count_against_the_budget_and_the_overhead_sits_outside(self):
        b = dual()["phases"]["P"]["bars"][0]
        inside = {x["name"]: x["mib"] for x in b["segments"] if not x.get("ausserhalb_budget") and x["name"] not in ("reserve", "free")}
        self.assertEqual(list(inside), ["state"])                                        # Mamba/GDN-Zustand ist P-eigen
        fx = next(x for x in b["segments"] if x["name"] == "fixed")
        self.assertEqual((fx["mib"], fx.get("ausserhalb_budget")), (2500.0, True))
        self.assertEqual(fx["herkunft"], "Eingabe (Nutzer/Profil)")
        self.assertIn("launcher.py:22713", fx["detail"])
        self.assertEqual(b["outside_budget_mib"], 2500.0)
        self.assertEqual(b["available_mib"], b["total_mib"] - 2500.0)

    def test_without_the_overhead_flag_the_launcher_default_1500_is_named_as_an_assumption(self):
        a = dict(DUAL_ARGS)
        del a["--dual-p-overhead-mib"]
        fx = next(x for x in dual(args=a)["phases"]["P"]["bars"][0]["segments"] if x["name"] == "fixed")
        self.assertEqual(fx["mib"], 1500.0)
        self.assertEqual(fx["herkunft"], "Annahme dieser Rechnung")
        self.assertIn("Standard des Launchers", fx["detail"])

    def test_without_unified_kv_the_kv_still_counts_against_the_p_budget(self):
        a = dict(DUAL_ARGS)
        a["--dual-unified-kv"] = "off"
        b = dual(args=a)["phases"]["P"]["bars"][0]
        self.assertIn("kv", [x["name"] for x in b["segments"]])
        self.assertNotIn("kv", [x["name"] for x in b["shared_with_d"]])
        self.assertGreater(b["overflow_mib"], 0)                                          # 262144 Token KV gegen 6610 MiB: das ist dann echt ueber dem Budget

    def test_a_p_own_post_over_the_budget_is_still_an_overflow(self):
        pa = copy.deepcopy(DUAL_PA)
        pa["P"]["--rank-gpu-memory-mib"] = "100,100,100"
        for b in dual(pa=pa)["phases"]["P"]["bars"]:
            self.assertGreater(b["overflow_mib"], 0)
            self.assertIn("ueber dem Budget", b["over_text"])

    def test_flip_and_d_only_are_unchanged_by_the_dual_share_branch(self):
        flip = dual(args={k: v for k, v in DUAL_ARGS.items() if not k.startswith("--dual-")}, form="flip")
        for b in flip["phases"]["P"]["bars"]:
            self.assertIn("weights", [x["name"] for x in b["segments"]])                  # Flip: volle Gewichte gegen das Budget
            self.assertEqual(b["shared_with_d"], [])
        self.assertTrue(flip["phases"]["P"]["bars"][0]["overflow_mib"] > 0)
        dual_no_share = {k: v for k, v in DUAL_ARGS.items() if k != "--dual-share"}
        r = dual(args=dual_no_share, tokens=["--dual-layout"])                            # --dual-layout ohne --dual-share: P haelt eigene Gewichte
        self.assertIn("weights", [x["name"] for x in r["phases"]["P"]["bars"][0]["segments"]])
        self.assertEqual(r["phases"]["P"]["bars"][0]["shared_with_d"], [])
        d = dual(form="d_only")
        self.assertEqual(list(d["phases"]), ["D"])

    def test_the_hint_explains_the_reference_row(self):
        self.assertTrue(any("Union-Image von D" in h for h in dual()["hints"]))


class TestLauncherSemanticsOfTheTexts(unittest.TestCase):
    """Fix-Runde 3, Befund 2: der Launcher meldet DARUEBER und bootet weiter (launcher.py:19902-19958); --d-reserve-mib (launcher.py:19886) geht ins Verfuegbare."""

    def test_budget_over_available_names_the_launcher_not_a_refusal(self):
        b = PC.contract_bar(dict(stage(1000, 900, weights=300, fixed=150), terms={"weights": {"v": 300.0, "src": "Eingabe", "note": ""},
                                                                                     "fixed": {"v": 150.0, "src": "Eingabe", "note": "", "outside_budget": True}}), "D")
        self.assertEqual(b["budget_over_available_mib"], 50.0)
        self.assertIn("Launcher meldet DARUEBER und startet trotzdem", b["over_text"])
        self.assertNotIn("lehnt ab", b["over_text"])
        self.assertNotIn("Force", b["over_text"])

    def test_no_text_of_the_contract_claims_a_refusal(self):
        for b in (PC.contract_bar(stage(1000, 900, weights=700, kv=500), "D"), PC.contract_bar(stage(1000, 900, weights=600, kv=350), "P")):
            self.assertNotIn("lehnt ab", b["over_text"])

    def test_the_user_reserve_is_subtracted_from_the_available_for_the_verdict(self):
        st = stage(1000, 900, weights=300)
        st["user_reserve_mib"] = 200.0
        b = PC.contract_bar(st, "D")
        self.assertEqual(b["budget_over_available_mib"], 100.0)                           # 900 gefragt, 1000 - 200 = 800 verfuegbar
        self.assertEqual(b["user_reserve_mib"], 200.0)
        self.assertIn("Nutzerreserve 200 MiB (--d-reserve-mib)", b["over_text"])
        self.assertAlmostEqual(sum(s["mib"] for s in b["segments"]), 1000.0, places=6)    # nicht als eigenes Segment gezeichnet
        self.assertEqual(PC.contract_bar(stage(1000, 900, weights=300), "D")["budget_over_available_mib"], 0.0)

    def test_d_phase_reads_d_reserve_mib_from_the_group_line(self):
        m = model("nextflash_int4mixed")
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--d-reserve-mib"] = "1000,1000,1000"
        r = PC.phase_bars(hw(), m, dict(NF_ARGS), pa, NF_PE)
        d = r["phases"]["D"]
        self.assertEqual([b["user_reserve_mib"] for b in d["bars"]], [1000.0] * 3)
        self.assertIn("user_reserve_mib", [x["was"] for x in d["inputs"]])
        self.assertTrue(any("--d-reserve-mib" in h for h in d["hints"]))
        r0 = PC.phase_bars(hw(), m, dict(NF_ARGS), NF_PA, NF_PE)
        self.assertEqual([b["user_reserve_mib"] for b in r0["phases"]["D"]["bars"]], [0.0] * 3)


class TestDualShareDPhase(unittest.TestCase):
    """Fix-Runde 4, Befund 1: unter --dual-share bemisst der Launcher D aus P's PLAN (dc = P-Budget + --dual-p-overhead-mib je Karte, launcher.py:27289-27301)."""

    def test_the_release_dual_d_bar_holds_p_plan_outside_its_budget(self):
        d = dual()["phases"]["D"]["bars"]
        dc = (6610 + 2500, 5050 + 2500, 5200 + 2500)
        for b, tot, x, rb in zip(d, (32607, 20480, 20480), dc, (3191, 2079, 2067)):
            fx = seg(b, "fixed")
            self.assertEqual((fx["mib"], fx.get("ausserhalb_budget")), (x, True))
            self.assertIn("aus P-Plan", fx["detail"])
            self.assertIn("launcher.py:14976", fx["detail"])
            self.assertEqual(b["outside_budget_mib"], x)
            self.assertEqual(b["available_mib"], tot - x)
            # Launcher qwen27b (budget_rest_from_records): (Karte - dc - gebuchter Rest D_AWAKE_REST_BOOKED_MIB) // 8 * 8; --user-reserve-mib 1800,1400,1400
            # und die 404 werden daneben NICHT gebucht (launcher.py:15316-15322)
            self.assertEqual(b["budget_mib"], (tot - x - rb) // 8 * 8)
            self.assertEqual((b["overflow_mib"], b["beyond_card_mib"], b["budget_over_available_mib"]), (0.0, 0.0, 0.0))
            self.assertNotIn("weights", [r["name"] for r in b["shared_with_d"]])             # D ist der Owner: seine Gewichte zaehlen gegen SEIN Budget
            self.assertGreater(seg(b, "weights")["mib"], 1000)
        self.assertEqual([b["budget_mib"] for b in d], [20304, 10848, 10712])                 # Probe: launcher.budgets_from_dc mit allen Argumenten, dc 9110/7550/7700 (Klasse unten)

    def test_the_old_wrong_budget_31583_is_gone_and_d_does_not_inherit_p_budgets(self):
        d = dual()["phases"]["D"]
        self.assertNotIn(32607 - 1024, [b["budget_mib"] for b in d["bars"]])
        # ein --rank-gpu-memory-mib im gemeinsamen Profil gilt fuer P (P erbt es), D bemisst der Launcher trotzdem aus P's Plan
        a = dict(DUAL_ARGS)
        a["--rank-gpu-memory-mib"] = "6610,5050,5200"
        r = dual(args=a, pa={"P": {"--max-mamba-cache-size": "8"}})
        self.assertEqual([b["budget_mib"] for b in r["phases"]["D"]["bars"]], [20304, 10848, 10712])
        self.assertEqual([b["budget_mib"] for b in r["phases"]["P"]["bars"]], [6610, 5050, 5200])
        # eine eigene Zeile der Gruppe D wird als ignoriert benannt, nicht stillschweigend uebernommen
        pa = copy.deepcopy(DUAL_PA)
        pa["D"] = {"--rank-gpu-memory-mib": "30000,18000,18000"}
        r = dual(pa=pa)
        self.assertEqual([b["budget_mib"] for b in r["phases"]["D"]["bars"]], [20304, 10848, 10712])
        self.assertTrue(any(x["was"] == "budget_mib" and "ignoriert" in x["herkunft"] for x in r["phases"]["D"]["inputs"]))

    def test_without_the_p_group_budget_the_d_budget_is_an_upper_bound_not_a_number_pretending_to_be_the_plan(self):
        d = dual(pa={})["phases"]["D"]["bars"]
        for b, tot, rb in zip(d, (32607, 20480, 20480), (3191, 2079, 2067)):
            self.assertIsNone(seg(b, "fixed")["mib"])
            self.assertIn("nicht gerechnet", seg(b, "fixed")["detail"])
            self.assertNotIn("CUDA-Kontext der schlafenden Phase", seg(b, "fixed")["detail"])    # Befund 2: Text je Form, unter --dual-share ist es P's Plan
            self.assertEqual(b["budget_mib"], (tot - rb) // 8 * 8)
            self.assertIn("fixed", [x["name"] for x in b["segments"]])
            self.assertIn("Festposten", b["not_computed"])

    def test_the_overhead_default_is_named_when_the_flag_is_absent(self):
        a = dict(DUAL_ARGS)
        del a["--dual-p-overhead-mib"]
        b = dual(args=a)["phases"]["D"]["bars"][0]
        self.assertEqual(seg(b, "fixed")["mib"], 6610 + 1500)
        self.assertEqual(seg(b, "fixed")["herkunft"], "Naeherung (nicht der Loeser)")
        self.assertIn("Standard des Launchers", seg(b, "fixed")["detail"])

    def test_the_hint_names_the_d_side(self):
        self.assertTrue(any("D-Seite" in h and "P's Plan" in h for h in dual()["hints"]))
        self.assertFalse(any("D-Seite" in h for h in dual(args={k: v for k, v in DUAL_ARGS.items() if k != "--dual-share"}, tokens=["--dual-layout"])["hints"]))


FLIP27_ARGS = {"--pp-stage-ratio": "31,17,16", "--pp-attn-stage-ratio": "7,5,4", "--max-kv-per-request": "262144", "--draft-kv-on-p": "off",
               "--d-foreign-context-mib": "1446,896,894", "--d-nontorch-mib": "1981,528,524", "--user-reserve-mib": "1800,1400,1400"}   # Release-Flip 27b-base (qwen27b)


class TestLauncherFormulaForEveryPhaseAndForm(unittest.TestCase):
    """Fix-Runde 5 (Zusatz des Planer-Sitzes): fuer jede Phase (P, D) und jede Form (flip, dual, d_only, single) stimmen Budget, Festposten und
    Rest mit der Launcher-Formel ueberein.  Die D-Phase wird gegen ``launcher.budgets_from_dc`` gehalten, aufgerufen mit ALLEN Argumenten, die der
    Launcher in ``_d_spec_from`` uebergibt (launcher.py:27207-27214: overshoot, corridor_sample_path, corridor_constrain=True, user_reserve_by_card,
    dormant_growth, charge_driver_carve, driver_carve_min_total_mib, awake_rest, terms_out, booked_rest_kwargs) -- fuer Release-Flip (27b-base),
    Release-Dual, NF abl und d_only.  Der Korridor-Boden kommt aus SGLANG_CORRIDOR_LAW_FLOOR_MIB=1024 (host-unabhaengig)."""

    TOTALS = (32607, 20480, 20480)

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.weg2 import launcher as L
            from sglang.srt.planner import pp_cut as PP
        except Exception as exc:                                  # pragma: no cover - ohne Launcher-Import nicht pruefbar
            raise unittest.SkipTest("launcher import: %r" % (exc,))
        cls.L, cls.PP = L, PP

    def cards(self, reserved=(0, 0, 0)):
        names = ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080")     # kalibrierte Klassen: driver_carve_charged liest den Namen
        return [self.L.Card(nvml_index=i, uuid="U%d" % i, name=nm, total_mib=t, reserved_mib=r) for i, (nm, t, r) in enumerate(zip(names, self.TOTALS, reserved))]

    def launcher_budgets(self, dc, profile="qwen27b", reserve="0", env_d="", reserved=(0, 0, 0)):
        """``_d_spec_from`` (launcher.py:27199-27214) Zeile fuer Zeile mit dem echten Launcher-Code."""
        L, cards = self.L, self.cards(reserved)
        ns = types.SimpleNamespace(profile=profile, env_d=env_d)
        L.apply_profile_torch_cache_cap_default(ns)               # Registerzeile -> --env-d (launcher.py:14257)
        urc = L.parse_user_reserve(reserve, cards)
        _grow, _grow_prov = L.served_dormant_growth(cards, ns.profile)
        _rest, _rest_prov = L.d_awake_rest(cards, ns.profile)
        _over, _over_prov = L.d_overshoot_record(ns.profile)
        env = {k: v for k, v in os.environ.items() if k != "SGLANG_WEG2_BUDGET_REST_RECORD"}
        env["SGLANG_CORRIDOR_LAW_FLOOR_MIB"] = "1024"
        terms = []
        with mock.patch.dict(os.environ, env, clear=True):
            out = L.budgets_from_dc(
                cards, {c.uuid: int(dc[i]) for i, c in enumerate(cards)}, lambda *_: None, "D", overshoot_mib=_over,
                overshoot_provenance=_over_prov, corridor_sample_path=None, corridor_constrain=True, user_reserve_by_card=urc,
                dormant_growth_mib=_grow, dormant_growth_provenance=_grow_prov,
                charge_driver_carve=L.budget_charges_driver_carve(ns.profile), driver_carve_min_total_mib=L.driver_carve_min_total_mib(ns.profile),
                awake_rest_mib=_rest, awake_rest_provenance=_rest_prov, terms_out=terms,
                **L.booked_rest_kwargs(cards, ns.profile, "D", None, capped=L.torch_cache_cap_armed(ns)))
        return out

    def check_d_bar(self, bars, dc, available, **kw):
        """Budget = Launcher-Budget; Festposten = dc ausserhalb; Verfuegbar = d_rank_available_mib; Reserve = Verfuegbar - max(Budget, Posten im Budget)."""
        want = self.launcher_budgets(dc, **kw)
        self.assertEqual([b["budget_mib"] for b in bars], [float(x) for x in want])
        for b, x, av, w in zip(bars, dc, available, want):
            self.assertEqual(b["outside_budget_mib"], float(x))
            self.assertEqual(b["available_mib"], float(av))
            inside = b["posts_mib"] - b["outside_budget_mib"]
            self.assertAlmostEqual(seg(b, "reserve")["mib"], max(0.0, av - max(w, inside)), places=2)
            self.assertAlmostEqual(sum(x2["mib"] for x2 in b["segments"] if x2["mib"] is not None), max(b["total_mib"], b["posts_mib"] + 0.0), delta=0.01)

    def test_release_flip_27b_base_books_the_measured_rest_and_not_the_user_reserve(self):
        fo, nt = (1446, 896, 894), (1981, 528, 524)
        dc = [a + b for a, b in zip(fo, nt)]
        av = self.PP.d_rank_available_mib(card_total_mib=self.TOTALS, foreign_context_mib=fo, nontorch_mib=nt)
        r = PC.phase_bars(hw(), model("qwen27b_int8_vocabembed"), dict(FLIP27_ARGS), {}, {})
        self.assertEqual(r["form"], "flip")
        d = r["phases"]["D"]["bars"]
        self.check_d_bar(d, dc, av, profile="qwen27b", reserve="1800,1400,1400")
        # von Hand: (Karte - dc - gebuchter Rest 3191/2079/2067) // 8 * 8; die Nutzerreserve 1800/1400/1400 wird daneben nicht gebucht
        self.assertEqual([b["budget_mib"] for b in d], [(t - x - rb) // 8 * 8 for t, x, rb in zip(self.TOTALS, dc, (3191, 2079, 2067))])
        self.assertIn("gebuchter Rest", d[0]["budget_note"] if "budget_note" in d[0] else seg(d[0], "reserve")["detail"])
        self.assertIn("--user-reserve-mib (1800)", seg(d[0], "reserve")["detail"])

    def test_the_reserve_path_charges_user_reserve_in_the_floor_where_no_booked_rest_exists(self):
        """Befund 1 der Runde 5: --user-reserve-mib hebt den Korridor-Boden und senkt das D-Budget um genau diesen Betrag (Profil ohne gebuchten Rest)."""
        fo, nt = (1446, 896, 894), (1981, 528, 524)
        dc = [a + b for a, b in zip(fo, nt)]
        av = self.PP.d_rank_available_mib(card_total_mib=self.TOTALS, foreign_context_mib=fo, nontorch_mib=nt)
        a0, a1 = dict(NF_ARGS, **{"--user-reserve-mib": "0,0,0"}), dict(NF_ARGS, **{"--user-reserve-mib": "1800,1400,1400"})
        b0 = [b["budget_mib"] for b in nf(args=a0, form="d_only")["phases"]["D"]["bars"]]
        r1 = nf(args=a1, form="d_only")["phases"]["D"]["bars"]
        self.check_d_bar(r1, dc, av, profile="nextflash", reserve="1800,1400,1400")
        for x0, b1, res in zip(b0, r1, (1800, 1400, 1400)):
            self.assertAlmostEqual(x0 - b1["budget_mib"], res, delta=7.01)                    # um genau die Reserve (Rundung auf 8)
        self.assertIn("Nutzerreserve 1800", seg(r1[0], "reserve")["detail"])

    def test_d_phase_flip_and_d_only_foreign_plus_nontorch_is_dc(self):
        fo, nt = (1446, 896, 894), (1981, 528, 524)
        dc = [a + b for a, b in zip(fo, nt)]
        av = self.PP.d_rank_available_mib(card_total_mib=self.TOTALS, foreign_context_mib=fo, nontorch_mib=nt)
        flip = nf()
        self.check_d_bar(flip["phases"]["D"]["bars"], dc, av, profile="nextflash", reserve="0,0,0")
        d_only = nf(form="d_only")
        self.assertEqual(list(d_only["phases"]), ["D"])
        self.check_d_bar(d_only["phases"]["D"]["bars"], dc, av, profile="nextflash", reserve="0,0,0")
        self.assertEqual([b["budget_mib"] for b in d_only["phases"]["D"]["bars"]], [b["budget_mib"] for b in flip["phases"]["D"]["bars"]])

    def test_d_phase_flip_without_fixed_lines_has_dc_zero_and_the_fixed_post_is_not_computed(self):
        a = {k: v for k, v in NF_ARGS.items() if k not in ("--d-foreign-context-mib", "--d-nontorch-mib")}
        d = nf(args=a)["phases"]["D"]["bars"]
        self.assertEqual([b["budget_mib"] for b in d], [float(x) for x in self.launcher_budgets((0, 0, 0), profile="nextflash", reserve="0,0,0")])
        for b in d:
            self.assertIsNone(seg(b, "fixed")["mib"])
            self.assertEqual(b["outside_budget_mib"], 0.0)

    def test_d_phase_dual_share_dc_is_the_launcher_planned_dc(self):
        pb = [6610, 5050, 5200]
        dc_launcher = self.L.dual_share_planned_dc(self.cards(), pb, "", 2500)
        dc = [dc_launcher[c.uuid] for c in self.cards()]
        av = [t - x for t, x in zip(self.TOTALS, dc)]
        self.check_d_bar(dual()["phases"]["D"]["bars"], dc, av, profile="qwen27b", reserve="1800,1400,1400")
        # --extra-p mit kleinerem Wert senkt das Launcher-Budget (min), das Profil nennt nur den Wert der Gruppe P
        self.assertEqual(self.L.dual_share_planned_dc(self.cards(), [9000, 9000, 9000], "--rank-gpu-memory-mib 6610,5050,5200", 2500), dc_launcher)

    def test_every_booked_path_of_the_launcher_is_met_cap_switch_scalar_reserve_and_known_carve(self):
        fo, nt = (1446, 896, 894), (1981, 528, 524)
        dc = [a + b for a, b in zip(fo, nt)]
        av = self.PP.d_rank_available_mib(card_total_mib=self.TOTALS, foreign_context_mib=fo, nontorch_mib=nt)
        # Torch-Cache-Kappe: --env-d nennt den Schalter (0 = ungekappter Rest), sonst gilt die Registerzeile
        for env in ("SGLANG_WEG2_TORCH_CACHE_CAP=0", "SGLANG_WEG2_TORCH_CACHE_CAP=1"):
            pe = {"D": {"SGLANG_WEG2_TORCH_CACHE_CAP": env.split("=")[1]}}
            r = PC.phase_bars(hw(), model("qwen27b_int8_vocabembed"), dict(FLIP27_ARGS), {}, pe)
            self.check_d_bar(r["phases"]["D"]["bars"], dc, av, profile="qwen27b", reserve="1800,1400,1400", env_d=env)
        # skalare Reserve gilt fuer jede Karte (launcher.py:14152-14158)
        a = dict(NF_ARGS, **{"--user-reserve-mib": "1000"})
        self.check_d_bar(nf(args=a, form="d_only")["phases"]["D"]["bars"], dc, av, profile="nextflash", reserve="1000")
        # bekannter Treiber-Carve (Hardwareprofil driver_reserved_mib): nextflash bucht ihn auf jeder Karte, qwen27b im gebuchten-Rest-Pfad ebenfalls (launcher.py:15331)
        carve = (518, 425, 425)
        h = hw()
        for c, v in zip(h["cards"], carve):
            c["driver_reserved_mib"] = {"v": float(v), "src": "NVML"}
        for prof, args, m in (("qwen27b", FLIP27_ARGS, model("qwen27b_int8_vocabembed")), ("nextflash", NF_ARGS, model("nextflash_int4mixed"))):
            pa = NF_PA if prof == "nextflash" else {}
            r = PC.phase_bars(h, m, dict(args), pa, NF_PE if prof == "nextflash" else {})
            self.check_d_bar(r["phases"]["D"]["bars"], dc, av, profile=prof, reserve=args["--user-reserve-mib"], reserved=carve)

    def test_a_wrong_reserve_vector_is_refused_like_the_launcher(self):
        a = dict(NF_ARGS, **{"--user-reserve-mib": "1800,1400"})
        r = nf(args=a, form="d_only")["phases"]["D"]
        self.assertFalse(r["ok"])
        self.assertIn("--user-reserve-mib", r["error"])

    def test_the_unpriced_inputs_are_named_in_the_budget_tooltip(self):
        t = seg(nf()["phases"]["D"]["bars"][1], "reserve")["detail"]
        for word in ("NICHT GERECHNET", "Treiber-Carve", "Korridor-Boden", "Korridor-Pass"):
            self.assertIn(word, t)

    def test_d_phase_explicit_budget_is_taken_as_asked_and_the_rest_is_available_minus_budget(self):
        pa = copy.deepcopy(NF_PA)
        pa["D"]["--rank-gpu-memory-mib"] = "28000,17000,17000"
        for b, dcx in zip(nf(pa=pa)["phases"]["D"]["bars"], (3427, 1424, 1418)):
            self.assertEqual(b["outside_budget_mib"], float(dcx))
            self.assertAlmostEqual(seg(b, "reserve")["mib"], b["total_mib"] - dcx - b["budget_mib"], places=2)

    def test_p_phase_flip_budget_is_the_group_line_and_the_rest_is_card_minus_budget(self):
        pa = copy.deepcopy(NF_PA)
        pa["P"]["--rank-gpu-memory-mib"] = "30000,18000,18000"
        for b, bud in zip(nf(pa=pa)["phases"]["P"]["bars"], (30000, 18000, 18000)):
            self.assertEqual((b["budget_mib"], b["outside_budget_mib"], b["available_mib"]), (bud, 0.0, b["total_mib"]))
            self.assertIsNone(seg(b, "fixed")["mib"])                                      # kein Festposten gerechnet (nur am Metall zu messen)
            inside = b["posts_mib"]
            self.assertAlmostEqual(seg(b, "reserve")["mib"], max(0.0, b["total_mib"] - max(bud, inside)), places=2)

    def test_p_phase_dual_budget_is_the_group_line_and_the_fixed_post_is_the_overhead_outside(self):
        dc_launcher = self.L.dual_share_planned_dc(self.cards(), [6610, 5050, 5200], "", 2500)
        for b, bud, c in zip(dual()["phases"]["P"]["bars"], (6610, 5050, 5200), self.cards()):
            self.assertEqual(b["budget_mib"], bud)
            self.assertEqual(seg(b, "fixed")["mib"], float(dc_launcher[c.uuid] - bud))      # Launcher: dc = Budget + Overhead -> Overhead = dc - Budget
            self.assertEqual(b["available_mib"], b["total_mib"] - 2500)
            inside = b["posts_mib"] - b["outside_budget_mib"]
            self.assertAlmostEqual(seg(b, "reserve")["mib"], b["available_mib"] - max(bud, inside), places=2)

    def test_single_one_phase_budget_is_the_flag_and_nothing_sits_outside(self):
        r = PC.phase_bars(hw(RIG3[:1]), model("qwen27b_int8_vocabembed"), {"--rank-gpu-memory-mib": "30000", "--max-kv-per-request": "4096"}, {}, {})
        self.assertEqual((r["form"], list(r["phases"])), ("single", ["alle"]))
        b = r["phases"]["alle"]["bars"][0]
        self.assertEqual((b["budget_mib"], b["outside_budget_mib"], b["available_mib"]), (30000, 0.0, 32607))
        self.assertAlmostEqual(seg(b, "reserve")["mib"], max(0.0, 32607 - max(30000, b["posts_mib"])), places=2)

    def test_every_dual_share_rule_of_p_has_a_d_counterpart(self):
        r = dual()
        p, d = r["phases"]["P"]["bars"][0], r["phases"]["D"]["bars"][0]
        # P: Gewichte/Experten/KV als Referenz (shared) -> D (Owner): Gewichte und KV zaehlen gegen das eigene Budget
        self.assertEqual({x["name"] for x in p["shared_with_d"]} >= {"weights", "kv"}, True)
        self.assertEqual({"weights", "kv"} <= {x["name"] for x in d["segments"]}, True)
        self.assertEqual(d["shared_with_d"], [])
        # P: Festposten = Overhead ausserhalb -> D: Festposten = P's Plan ausserhalb; beide tragen die Launcher-Zeile als Beleg
        self.assertTrue(seg(p, "fixed")["ausserhalb_budget"] and seg(d, "fixed")["ausserhalb_budget"])
        self.assertIn("launcher.py:22713", seg(p, "fixed")["detail"])
        self.assertIn("launcher.py:14976", seg(d, "fixed")["detail"])
        # P: Kontext-Boden unter --dual-share nicht gerechnet -> D hat keinen Kontext-Boden-Wert, der falsch sein koennte
        self.assertIsNone(r["phases"]["P"]["context_floor_tokens"])
        self.assertNotIn("context_floor_tokens", r["phases"]["D"])
        # P: Hinweiszeile -> D: Hinweiszeile
        self.assertTrue(any("Union-Image von D" in h for h in r["hints"]) and any("D-Seite" in h for h in r["hints"]))


if __name__ == "__main__":
    unittest.main()
