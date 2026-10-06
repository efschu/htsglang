"""AP-F Einzelkarte (Planer-Workflow 06.10.2026, Plan §3 AP-F, R5b): ``weg2/propose_single.py``.

Abnahme (Plan):
* Laptop-Profil SYNTHETISCH (APU, 25600 MiB adressierbar, MoE-Hybrid 35B-A3B; alle Werte "unbelegt am Rig", Memory ``laptop-efeu-tp14``)
  -> der Vorschlag passt;
* Rig-Einzelkarte RTX 5090 (NVML 32607 MiB, ~/.claude/CLAUDE.md) + Qwen3.8-27B -> FIT-Verdikt mit Zahlen: NVFP4 passt, INT8 passt nicht.

Die Rechenregeln werden gegen eine UNABHAENGIGE Handrechnung aus den rohen Profilzahlen geprueft (nicht gegen die eigene Ausgabe), und die
ServerArgs-Parse-Pruefung faehrt im echten Kindprozess gegen ``ServerArgs.add_cli_args`` (kein GPU-Zugriff).
"""

import copy
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import model_profile as MP  # noqa: E402
from sglang.srt.weg2 import propose_single as PS  # noqa: E402
from test_planer_apb_modell_1006 import dflash2_dir  # noqa: E402

FX = os.path.join(HERE, "fixtures", "planer_apf_1006")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
MIB = float(1 << 20)

#: Rig-Einzelkarte: RTX 5090, NVML-Gesamtspeicher 32607 MiB (~/.claude/CLAUDE.md, Rig-Regeln).
CARD_5090 = {"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607, "total_src": "Kartenprofil (NVML 32607 MiB laut Rig-Regeln)"}
#: Laptop efeu-TP14 als Beispiel (APU, adressierbare Decke 25600 MiB laut Memory laptop-efeu-tp14) -- am Rig UNBELEGT.
CARD_LAPTOP = {"name": "AMD Radeon 780M (APU, Beispiel efeu-TP14)", "total_mib": 25600, "usable_mib": 25600, "unified": True, "platform": "rocm",
               "total_src": "unbelegt (Memory laptop-efeu-tp14: adressierbare Decke 25600 MiB; torch-Gesamtspeicher der APU nicht gemessen)"}


def n(v, src="Index", **kw):
    d = {"v": v, "src": src}
    d.update(kw)
    return d


def synthetic_laptop_profile():
    """flliper.model/1-Auszug fuer ein 35B-A3B-Hybridmodell (10 Attention + 30 GDN-Layer, 2 KV-Koepfe x 256).  ALLE Zahlen sind Annahmen aus der
    Memory-Datei laptop-efeu-tp14 (75 MB/Slot gemessen am Laptop; Gewichte ~17 GB 'Dienst') oder Geometrie-Beispiel -> Quelle ``unbelegt``."""
    U = "unbelegt"
    return {
        "schema": "flliper.model/1", "path": "/models/synthetic-35b-a3b", "id": "synthetic-laptop",
        "format": n("gguf-q3", U),
        "arch": {"family": n("moe", U), "hybrid": n(True, U), "n_layers": n(40, U), "layer_counts": n({"attn": 10, "gdn": 30, "mamba": 0}, U), "vision": n(False, U)},
        "weights": {"total_bytes": n(17000 * 1 << 20, U), "visual_bytes": n(0, U), "mtp_bytes": n(0, U)},
        "kv": {"attn_layers": n(10, U),
               "variants": {"auto": {"cell_bytes_per_attn_layer_token": n(2048.0, U)}, "fp8_e4m3": {"cell_bytes_per_attn_layer_token": n(1088.0, U)}}},
        "state": {"linear_layers": n(30, U), "per_request_bytes": n(75_000_000, "gemessen am Laptop (Memory), unbelegt am Rig")},
        "context": {"max_position_embeddings": n(262144, U)},
        "draft": {"mtp_layers": n(0, U)},
    }


def q27(name):
    """Qwen3.8-27B aus der echten config.json (Kopie als Fixture); Gewichte = Geometrieformel (weicht vom Index um < 0,03 % ab, geprueft unten)."""
    return MP.estimate(os.path.join(FX, name))


def hand_posten_nvfp4_fp8(mp, kv_tokens, slots, seats=1, draft=True):
    """UNABHAENGIGE Handrechnung (MiB) aus den rohen Profilzahlen: Gewichte, Draft, KV fp8, Mamba, Spekulation."""
    w = mp["weights"]
    main = (w["total_bytes"]["v"] - w["mtp_bytes"]["v"] - w["visual_bytes"]["v"]) / MIB
    drf = w["mtp_bytes"]["v"] / MIB if draft else 0.0
    cell = mp["kv"]["variants"]["fp8_e4m3"]["cell_bytes_per_attn_layer_token"]["v"]
    n_attn = mp["kv"]["attn_layers"]["v"]
    kv = kv_tokens * cell * n_attn / MIB
    kv_d = kv_tokens * cell * mp["draft"]["mtp_layers"]["v"] / MIB if draft else 0.0
    per_req = mp["state"]["per_request_bytes"]["v"]
    mamba = slots * per_req / MIB
    spec = (min(seats, slots // 3) * 4 * per_req / MIB) if draft else 0.0
    return main, drf, kv, kv_d, mamba, spec


class TestRegeln(unittest.TestCase):
    """Bausteine nach server_args.py (Datei:Zeile in den Docstrings)."""

    def test_capacity_tier_klassen(self):
        self.assertEqual(PS.capacity_tier(16 * 1024), (2048, 8))
        self.assertEqual(PS.capacity_tier(20480), (2048, 24))        # RTX 3080 20 GiB: 20*1024 nicht < 20*1024 -> zweite Klasse
        self.assertEqual(PS.capacity_tier(32607), (2048, 24))        # 5090
        self.assertEqual(PS.capacity_tier(32607, tp_size=4), (2048, 80))
        self.assertEqual(PS.capacity_tier(40 * 1024), (4096, 32))
        self.assertEqual(PS.capacity_tier(80 * 1024), (8192, 256))
        self.assertEqual(PS.capacity_tier(180 * 1024), (16384, 512))
        self.assertEqual(PS.capacity_tier(None), (4096, 160))

    def test_prefill_graph_sizes_und_reserve_handrechnung(self):
        sizes = PS.prefill_graph_sizes(2048)
        # 4..32 step 4 = 8, 48..256 step 16 = 14, 288..512 step 32 = 8, 576..1024 step 64 = 8, 1280..2048 step 256 = 4  -> 42
        self.assertEqual(len(sizes), 42)
        self.assertEqual(sizes[0], 4)
        self.assertEqual(sizes[-1], 2048)
        # 5090: 512 + 2048*1.5 + 1/8*1024 + 1*2 + 42*8 = 512 + 3072 + 128 + 2 + 336 = 4050
        self.assertEqual(PS.reserve_mib(32607, 2048, 1, 42), 4050.0)
        # ROCm (APU): kein Prefill-Graph-Posten
        self.assertEqual(PS.reserve_mib(25600, 2048, 2, 42, cuda=False), 512 + 3072 + 128 + 4)
        # ueber 60 GiB: mindestens 10 GiB
        self.assertEqual(PS.reserve_mib(80 * 1024, 2048, 1, 42), 10240.0)

    def test_reserve_und_tier_gegen_echte_serverargs(self):
        """Gegenprobe im Kindprozess: ``_generate_prefill_cuda_graph_batch_sizes`` und ``_apply_gpu_mem_capacity_defaults`` der echten Klasse."""
        import subprocess
        code = r'''
import json, sys, types
sys.path.insert(0, sys.argv[1])
from sglang.srt.server_args import ServerArgs
out = {"sizes": {str(m): ServerArgs._generate_prefill_cuda_graph_batch_sizes(None, m) for m in (512, 2048, 8192)}, "tier": {}}
for g in (16 * 1024, 20480, 32607, 40 * 1024, 80 * 1024, 180 * 1024):
    for tp in (1, 4):
        me = types.SimpleNamespace(chunked_prefill_size=None, tp_size=tp, cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(max_bs=None)))
        ServerArgs._apply_gpu_mem_capacity_defaults(me, g)
        out["tier"]["%d/%d" % (g, tp)] = [me.chunked_prefill_size, me.cuda_graph_config.decode.max_bs]
sys.stdout.write("\nJSON:" + json.dumps(out) + "\n")
'''
        cp = subprocess.run([sys.executable, "-c", code, PS._tree_python_dir()], capture_output=True, text=True, timeout=180, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        line = next((x for x in reversed(cp.stdout.splitlines()) if x.startswith("JSON:")), None)
        if line is None:
            self.skipTest("sglang.srt.server_args nicht importierbar: %s" % cp.stderr[-200:])
        real = json.loads(line[5:])
        for m, sizes in real["sizes"].items():
            self.assertEqual(PS.prefill_graph_sizes(int(m)), sizes, m)
        for key, (cps, bs) in real["tier"].items():
            g, tp = (int(x) for x in key.split("/"))
            self.assertEqual(PS.capacity_tier(g, tp), (cps, bs), key)


class TestRig5090Qwen27(unittest.TestCase):
    """Abnahme: Rig-Einzelkarte 5090 + 27B -> FIT-Verdikt mit Zahlen."""

    def test_config_fixture_stimmt_mit_dem_index_ueberein(self):
        mp = q27("q27_nvfp4")
        self.assertEqual(mp["weights"]["total_bytes"]["src"], "geschätzt")
        if os.path.isdir(MC + "Qwen3.8-27B-NVFP4"):
            real = MP.estimate(MC + "Qwen3.8-27B-NVFP4")
            self.assertEqual(real["weights"]["total_bytes"]["src"], "Index")
            rel = abs(real["weights"]["total_bytes"]["v"] - mp["weights"]["total_bytes"]["v"]) / real["weights"]["total_bytes"]["v"]
            self.assertLess(rel, 0.001)
        else:
            self.skipTest("Rig-Modellordner nicht gemountet")

    def test_nvfp4_passt_mit_zahlen(self):
        mp = q27("q27_nvfp4")
        p = PS.propose_single(mp, CARD_5090)
        self.assertEqual(p["schema"], PS.SCHEMA)
        self.assertEqual(p["form"], "einzelkarte")
        self.assertEqual(p["verdikt"]["state"], "passt", p["verdikt"]["text"])
        self.assertEqual(p["verdikt"]["art"], "Planer-Rechnung")
        flags = {e["flag"]: e for e in p["flags"]}
        # Leiter: Modell-Dtype (bf16) passt nicht, fp8 passt; Draft bleibt
        self.assertEqual([r["schritt"] for r in p["relaxations"]], ["KV-Dtype fp8_e4m3"])
        self.assertEqual(flags["--kv-cache-dtype"]["wert"], "fp8_e4m3")
        self.assertEqual(flags["--no-enable-multimodal"]["wert"], True)
        self.assertEqual(flags["--speculative-algorithm"]["wert"], "NEXTN")
        self.assertEqual(flags["--context-length"]["wert"], 131072)
        self.assertEqual(flags["--max-running-requests"]["wert"], 1)
        # Reserve (Handrechnung 4050) -> Bruchteil = floor3((32607 - 4050) / 32607)
        self.assertAlmostEqual(flags["--mem-fraction-static"]["wert"], math.floor((32607 - 4050) / 32607 * 1000) / 1000, places=6)
        self.assertEqual(flags["--mem-fraction-static"]["wert"], 0.875)
        budget = 0.875 * 32607
        # unabhaengige Handrechnung der Posten mit den ENDGUELTIGEN Pool-Groessen
        kv_tokens = flags["--max-total-tokens"]["wert"]
        slots = flags["--max-mamba-cache-size"]["wert"]
        main, drf, kv, kv_d, mamba, spec = hand_posten_nvfp4_fp8(mp, kv_tokens, slots)
        total = main + drf + kv + kv_d + mamba + spec
        self.assertAlmostEqual(p["fit"]["statisch_summe_mib"], total, delta=0.6)
        self.assertLessEqual(total, budget)
        self.assertAlmostEqual(p["fit"]["statisch_budget_mib"], budget, delta=0.1)
        self.assertAlmostEqual(p["fit"]["frei_mib"], budget - total, delta=0.7)
        self.assertGreaterEqual(kv_tokens, 131072)               # KV-Pflicht = Ziel-Kontext
        self.assertGreaterEqual(slots, 3)                         # Mamba-Untergrenze 3 Slots je Anfrage
        # derselbe Bruchteil, den die ServerArgs-Heuristik selbst waehlen wuerde (gleiche Formel)
        self.assertEqual(p["fit"]["reserve_bedarf_mib"], 4050.0)
        # Der Rest ist aufgebraucht: ein weiterer Mamba-Slot oder 64... Token passten nicht mehr
        self.assertLess(p["fit"]["frei_mib"], mp["state"]["per_request_bytes"]["v"] / MIB + 1.0)
        self.assertGreaterEqual(p["fit"]["max_context_tokens_fit"], 131072)
        self.assertTrue(all(c["ok"] for c in p["fit"]["checks"]))
        # die Posten tragen Herkunft und Formel
        for post in p["fit"]["posten"]:
            self.assertTrue(post["herkunft"] and post["formel"], post)

    def test_int8_passt_nicht_mit_zahlen(self):
        mp = q27("q27_int8")
        p = PS.propose_single(mp, CARD_5090)
        self.assertEqual(p["verdikt"]["state"], "passt nicht")
        self.assertIn("es fehlen", p["verdikt"]["text"])
        # Leiter ging bis zum Ende: fp8 und Draft aus
        self.assertEqual([r["schritt"] for r in p["relaxations"]], ["KV-Dtype fp8_e4m3", "Draft weggelassen"])
        w = mp["weights"]
        main = (w["total_bytes"]["v"] - w["mtp_bytes"]["v"] - w["visual_bytes"]["v"]) / MIB
        flags = {e["flag"]: e for e in p["flags"]}
        main2, drf, kv, kv_d, mamba, spec = hand_posten_nvfp4_fp8(mp, flags["--max-total-tokens"]["wert"], flags["--max-mamba-cache-size"]["wert"], draft=False)
        self.assertAlmostEqual(main2, main)
        need = main + kv + mamba
        budget = 0.875 * 32607
        self.assertGreater(need, budget)
        self.assertAlmostEqual(p["fit"]["fehlt_mib"], need - budget, delta=0.7)
        self.assertNotIn("frei_mib", p["fit"])
        self.assertEqual(p["fit"]["max_context_tokens_fit"], 0)   # nach den Gewichten bleibt nicht einmal die Mamba-Untergrenze + 1 Token
        for f in ("--max-total-tokens", "--mem-fraction-static", "--max-mamba-cache-size"):
            self.assertEqual(flags[f]["verdikt"]["state"], "verweigert", f)
            self.assertEqual(flags[f]["verdikt"]["code"], "FIT-STATIC", f)
            self.assertIn("fehlen", flags[f]["verdikt"]["grund"])
        # Flags ohne Passungsbezug bleiben "geht" (die Karte ist nur fuer die Pools zu klein)
        self.assertEqual(flags["--kv-cache-dtype"]["verdikt"]["state"], "geht")
        self.assertEqual(flags["--model-path"]["verdikt"]["state"], "geht")

    def test_int8_am_index_wie_in_der_config(self):
        if not os.path.isdir(MC + "Qwen3.8-27B-INT8-gdncov"):
            self.skipTest("Rig-Modellordner nicht gemountet")
        real = MP.estimate(MC + "Qwen3.8-27B-INT8-gdncov")
        p = PS.propose_single(real, CARD_5090)
        self.assertEqual(p["verdikt"]["state"], "passt nicht")
        self.assertIn("Index", p["fit"]["posten"][0]["herkunft"])
        self.assertAlmostEqual(p["fit"]["posten"][0]["mib"], (real["weights"]["total_bytes"]["v"] - real["weights"]["mtp_bytes"]["v"] - real["weights"]["visual_bytes"]["v"]) / MIB, delta=0.1)

    def test_explizites_kv_dtype_wird_nie_still_geaendert(self):
        mp = q27("q27_nvfp4")
        p = PS.propose_single(mp, CARD_5090, {"kv_dtype": "auto"})
        flags = {e["flag"]: e for e in p["flags"]}
        self.assertEqual(flags["--kv-cache-dtype"]["wert"], "auto")
        self.assertEqual(flags["--kv-cache-dtype"]["herkunft"], "Ziel")
        # bf16-KV mit Ziel-Kontext 131072: Gewichte 20644 + KV 8192 + Mamba >= 28531 -> passt nicht; die Leiter nimmt nur den Draft weg
        self.assertEqual([r["schritt"] for r in p["relaxations"]], ["Draft weggelassen"])
        self.assertEqual(p["verdikt"]["state"], "passt nicht")
        main, _, _, _, _, _ = hand_posten_nvfp4_fp8(mp, 0, 0, draft=False)
        kv_auto = 131072 * mp["kv"]["variants"]["auto"]["cell_bytes_per_attn_layer_token"]["v"] * 16 / MIB
        self.assertAlmostEqual(kv_auto, 8192.0)
        self.assertGreater(main + kv_auto + 3 * mp["state"]["per_request_bytes"]["v"] / MIB, 0.875 * 32607)

    def test_kleinerer_kontext_mit_bf16_kv_passt(self):
        mp = q27("q27_nvfp4")
        p = PS.propose_single(mp, CARD_5090, {"kv_dtype": "auto", "context_tokens": 65536})
        self.assertEqual(p["verdikt"]["state"], "passt", p["verdikt"]["text"])
        flags = {e["flag"]: e for e in p["flags"]}
        self.assertEqual(flags["--kv-cache-dtype"]["wert"], "auto")
        self.assertEqual(p["relaxations"], [])
        self.assertGreaterEqual(flags["--max-total-tokens"]["wert"], 65536)

    def test_draft_aus_und_sitze(self):
        mp = q27("q27_nvfp4")
        p = PS.propose_single(mp, CARD_5090, {"draft": "off", "seats": 2, "context_tokens": 65536})
        flags = {e["flag"]: e for e in p["flags"]}
        self.assertIsNone(flags["--speculative-algorithm"]["wert"])
        self.assertNotIn("--speculative-algorithm", p["argv"])
        self.assertNotIn("--speculative-num-steps", flags)
        self.assertEqual(flags["--max-running-requests"]["wert"], 2)
        self.assertEqual(flags["--cuda-graph-max-bs-decode"]["wert"], 2)
        self.assertGreaterEqual(flags["--max-mamba-cache-size"]["wert"], 6)    # 2 Anfragen x 3 Slots
        self.assertEqual(p["verdikt"]["state"], "passt")
        names = [x["name"] for x in p["fit"]["posten"]]
        self.assertFalse(any("Draft" in x or "Spekulation" in x for x in names))

    def test_hicache_nur_mit_host_budget(self):
        mp = q27("q27_nvfp4")
        none = PS.propose_single(mp, CARD_5090, {"draft": "off", "context_tokens": 65536})
        f0 = {e["flag"]: e for e in none["flags"]}
        self.assertIsNone(f0["--enable-hierarchical-cache"]["wert"])
        self.assertEqual(f0["--enable-hierarchical-cache"]["zustand"], "unbelegt")
        self.assertNotIn("--hicache-ratio", f0)
        self.assertNotIn("--page-size", f0)
        p = PS.propose_single(mp, CARD_5090, {"draft": "off", "context_tokens": 65536, "host_ram_mib": 32768})
        f = {e["flag"]: e for e in p["flags"]}
        self.assertTrue(f["--enable-hierarchical-cache"]["wert"])
        self.assertEqual(f["--page-size"]["wert"], 64)
        self.assertEqual(f["--max-total-tokens"]["wert"] % 64, 0)
        key = "fp8_e4m3" if f["--kv-cache-dtype"]["wert"].startswith("fp8") else "auto"      # ohne Draft und mit 65536 Kontext reicht evtl. der Modell-Dtype
        kv_main_mib = f["--max-total-tokens"]["wert"] * mp["kv"]["variants"][key]["cell_bytes_per_attn_layer_token"]["v"] * 16 / MIB
        ratio = f["--hicache-ratio"]["wert"]
        self.assertLessEqual(ratio * kv_main_mib, 32768 + 1e-6)            # nie mehr Host als das Budget
        self.assertGreater((ratio + 0.01) * kv_main_mib, 32768 - 1e-6)    # und auf 0,01 genau ausgeschoepft
        # diskrete Karte: das Hostpool zaehlt nicht gegen die Karte
        self.assertFalse(any("HiCache" in x["name"] for x in p["fit"]["posten"]))

    def test_uebersteuerung_rechnet_die_passung_neu(self):
        mp = q27("q27_nvfp4")
        base = PS.propose_single(mp, CARD_5090)
        over = PS.propose_single(mp, CARD_5090, overrides={"--max-total-tokens": 400000})
        fo = {e["flag"]: e for e in over["flags"]}
        self.assertEqual(fo["--max-total-tokens"]["wert"], 400000)
        self.assertEqual(fo["--max-total-tokens"]["zustand"], "übersteuert")
        self.assertEqual(fo["--max-total-tokens"]["herkunft"], "übersteuert")
        self.assertIn("vorgeschlagen war", fo["--max-total-tokens"]["begruendung"])
        self.assertEqual(over["verdikt"]["state"], "passt nicht")
        self.assertEqual(fo["--max-total-tokens"]["verdikt"]["state"], "verweigert")
        self.assertEqual(fo["--max-total-tokens"]["verdikt"]["code"], "FIT-STATIC")
        self.assertEqual(base["verdikt"]["state"], "passt")
        # KV-Posten = 400000 Token x 16 Layer x 2176 B (Handrechnung)
        kv = next(x for x in over["fit"]["posten"] if x["name"] == "KV-Pool (Hauptmodell)")
        cell = mp["kv"]["variants"]["fp8_e4m3"]["cell_bytes_per_attn_layer_token"]["v"]
        self.assertAlmostEqual(kv["mib"], 400000 * cell * 16 / MIB, delta=0.1)
        # Flags, die der Planer nicht kennt: Durchgriff, unbelegt
        x = PS.propose_single(mp, CARD_5090, overrides={"--attention-backend": "triton"})
        fx = {e["flag"]: e for e in x["flags"]}
        self.assertEqual(fx["--attention-backend"]["wert"], "triton")
        self.assertEqual(fx["--attention-backend"]["verdikt"]["state"], "unbelegt")
        self.assertEqual(x["argv"][-2:], ["--attention-backend", "triton"])
        # Draft per Uebersteuerung abwaehlen -> Spekulationsposten entfallen, Passung neu
        y = PS.propose_single(mp, CARD_5090, overrides={"--speculative-algorithm": None})
        self.assertFalse(any("Draft" in z["name"] or "Spekulation" in z["name"] for z in y["fit"]["posten"]))
        self.assertNotIn("--speculative-algorithm", y["argv"])

    def test_reserve_ziel_ersetzt_die_heuristik(self):
        mp = q27("q27_nvfp4")
        p = PS.propose_single(mp, CARD_5090, {"reserve_mib": 2000})
        self.assertEqual(p["fit"]["reserve_bedarf_mib"], 2000.0)
        self.assertEqual(p["fit"]["reserve_herkunft"], "Ziel")
        f = {e["flag"]: e for e in p["flags"]}
        self.assertEqual(f["--mem-fraction-static"]["wert"], math.floor((32607 - 2000) / 32607 * 1000) / 1000)

    def test_deterministisch_und_json(self):
        mp = q27("q27_nvfp4")
        a = PS.propose_single(mp, CARD_5090, {"seats": 2})
        b = PS.propose_single(copy.deepcopy(mp), dict(CARD_5090), {"seats": 2})
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        for e in a["flags"]:
            self.assertEqual(set(e), {"flag", "wert", "herkunft", "zustand", "begruendung", "verdikt"})
            self.assertIn(e["zustand"], ("vorgeschlagen", "übersteuert", "unbelegt"))
            self.assertEqual(set(e["verdikt"]), {"state", "code", "grund", "art", "parse"})
            self.assertEqual(e["verdikt"]["art"], "Planer-Rechnung")
            self.assertTrue(e["begruendung"])

    def test_karte_aus_hardwareprofil(self):
        hw = {"schema": "flliper.hardware/1", "cards": [{"ord": 0, "name": "NVIDIA GeForce RTX 5090", "vram_total_mib": {"v": 32607, "src": "NVML"},
                                                         "cc": [12, 0], "mem_gbs": {"nameplate": {"v": 1792.0, "src": "Datenblatt"}}}]}
        c = PS.card_from_hardware(hw, 0)
        self.assertEqual(c["total_mib"], 32607.0)
        self.assertEqual(c["total_src"], "Kartenprofil (NVML)")
        self.assertEqual(c["bandwidth_gbs"], 1792.0)
        p = PS.propose_single(q27("q27_nvfp4"), c)
        self.assertEqual(p["verdikt"]["state"], "passt")
        with self.assertRaises(PS.ProposeSingleError):
            PS.card_from_hardware(hw, 3)


class TestLaptopSynthetisch(unittest.TestCase):
    """Abnahme: Laptop-Profil synthetisch (APU 25600 MiB adressierbar, 35B-A3B-Hybrid) -> der Vorschlag passt; alles geborgte ist "unbelegt"."""

    def setUp(self):
        self.mp = synthetic_laptop_profile()
        self.goals = {"seats": 2, "context_tokens": 131072, "host_ram_mib": 2000}
        self.p = PS.propose_single(self.mp, CARD_LAPTOP, self.goals)
        self.f = {e["flag"]: e for e in self.p["flags"]}

    def test_passt_mit_zahlen(self):
        p = self.p
        self.assertEqual(p["verdikt"]["state"], "passt", p["verdikt"]["text"])
        # ROCm: Reserve ohne Prefill-Graph-Posten, 512 + 3072 + 128 + max_bs(2) x 2 = 3716; Hostpool 2000 zaehlt gegen die Decke (APU)
        self.assertEqual(p["fit"]["reserve_bedarf_mib"], 3716.0)
        fr = math.floor((25600 - 2000 - 3716) / 25600 * 1000) / 1000
        self.assertEqual(self.f["--mem-fraction-static"]["wert"], fr)
        budget = fr * 25600
        # gegebene Reserve = Karte - Budget - Hostpool, nie unter dem Bedarf und hoechstens 0,1 % der Karte darueber (Abrundung auf 3 Stellen)
        self.assertGreaterEqual(p["fit"]["reserve_gegeben_mib"], 3716.0)
        self.assertLess(p["fit"]["reserve_gegeben_mib"], 3716.0 + 25.6 + 0.1)
        self.assertAlmostEqual(p["fit"]["reserve_gegeben_mib"], 25600 - budget - 2000, delta=0.1)
        # Posten per Handrechnung
        kv_tokens = self.f["--max-total-tokens"]["wert"]
        slots = self.f["--max-mamba-cache-size"]["wert"]
        cell = {"auto": 2048.0, "fp8_e4m3": 1088.0}[self.f["--kv-cache-dtype"]["wert"]]
        total = 17000.0 + kv_tokens * cell * 10 / MIB + slots * 75_000_000 / MIB
        self.assertAlmostEqual(p["fit"]["statisch_summe_mib"], total, delta=0.6)
        self.assertLessEqual(total, budget)
        self.assertGreaterEqual(kv_tokens, 131072)
        self.assertGreaterEqual(slots, 2 * 3)
        self.assertEqual(self.f["--max-running-requests"]["wert"], 2)
        self.assertEqual(self.f["--cuda-graph-max-bs-decode"]["wert"], 2)
        self.assertNotIn("--speculative-algorithm", p["argv"])            # kein MTP-Kopf im Modell
        self.assertIn("MoE im einzelnen Server", " ".join(p["fit"]["hinweise"]))
        self.assertIn("APU: Geraet und Host teilen den Speicher", " ".join(p["fit"]["hinweise"]))
        host = next(x for x in p["fit"]["posten"] if x["name"].startswith("HiCache-Hostpool"))
        self.assertEqual(host["mib"], 2000.0)
        # das Hostpool liegt NICHT im statischen Posten, sondern verengt den Bruchteil
        self.assertAlmostEqual(p["fit"]["statisch_summe_mib"], sum(x["mib"] for x in p["fit"]["posten"] if not x["name"].startswith("HiCache")), delta=0.6)

    def test_hicache(self):
        self.assertTrue(self.f["--enable-hierarchical-cache"]["wert"])
        self.assertEqual(self.f["--page-size"]["wert"], 64)
        cell = {"auto": 2048.0, "fp8_e4m3": 1088.0}[self.f["--kv-cache-dtype"]["wert"]]
        kv_mib = self.f["--max-total-tokens"]["wert"] * cell * 10 / MIB
        self.assertLessEqual(self.f["--hicache-ratio"]["wert"] * kv_mib, 2000 + 1e-6)

    def test_geborgtes_ist_unbelegt(self):
        for fl in ("--max-total-tokens", "--mem-fraction-static", "--max-mamba-cache-size", "--kv-cache-dtype"):
            self.assertEqual(self.f[fl]["zustand"], "unbelegt", fl)
            self.assertIn("geborgt, unbelegt", self.f[fl]["begruendung"], fl)
        self.assertTrue(all("unbelegt" in x["herkunft"] for x in self.p["fit"]["posten"] if x["name"] != "HiCache-Hostpool (gleicher Speicher)"))
        self.assertTrue(any("APU-Speichermodell" in u for u in self.p["unbelegt"]))
        self.assertTrue(any("__post_init__" in u for u in self.p["unbelegt"]))

    def test_gleiche_form_auf_diskreter_karte_ohne_host_posten(self):
        card = dict(CARD_LAPTOP, unified=False, platform="cuda")
        p = PS.propose_single(self.mp, card, self.goals)
        self.assertFalse(any(x["name"].startswith("HiCache-Hostpool") for x in p["fit"]["posten"]))
        self.assertEqual(p["fit"]["reserve_bedarf_mib"], 3716.0 + len(PS.prefill_graph_sizes(2048)) * 8)

    def test_zu_kleine_karte_passt_nicht(self):
        card = dict(CARD_LAPTOP, total_mib=18000, usable_mib=18000)
        p = PS.propose_single(self.mp, card, self.goals)
        self.assertEqual(p["verdikt"]["state"], "passt nicht")
        self.assertGreater(p["fit"]["fehlt_mib"], 0)
        # Reserve und Hostpool allein lassen 18000 - 2000 - 3716 = 12284 MiB < Gewichte 17000
        self.assertIn("es fehlen", p["verdikt"]["text"])

    def test_reserve_groesser_als_karte(self):
        card = dict(CARD_LAPTOP, total_mib=3000, usable_mib=3000)
        p = PS.propose_single(self.mp, card, {"seats": 1})
        self.assertEqual(p["verdikt"]["state"], "passt nicht")
        f = {e["flag"]: e for e in p["flags"]}
        self.assertIsNone(f["--mem-fraction-static"]["wert"])
        self.assertEqual(f["--mem-fraction-static"]["verdikt"]["state"], "verweigert")
        self.assertEqual(f["--mem-fraction-static"]["verdikt"]["code"], "FIT-RESERVE")


class TestExternerDraft(unittest.TestCase):
    def test_dflash2_draft_mit_gleitfenster(self):
        mp = q27("q27_nvfp4")
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = dflash2_dir(tmp)
            dp = MP.estimate_draft(d)
            p = PS.propose_single(mp, CARD_5090, {"context_tokens": 65536}, draft_profile=dp)
        f = {e["flag"]: e for e in p["flags"]}
        self.assertEqual(f["--speculative-algorithm"]["wert"], "DFLASH")
        self.assertEqual(f["--speculative-draft-model-path"]["wert"], dp["path"])
        self.assertEqual(f["--speculative-dflash-block-size"]["wert"], 8)
        self.assertNotIn("--speculative-num-steps", f)
        post = {x["name"]: x for x in p["fit"]["posten"]}
        drf = next(v for k, v in post.items() if k.startswith("Gewichte (Draft"))
        self.assertAlmostEqual(drf["mib"], dp["total_bytes"]["v"] / MIB, delta=0.1)
        kvd = next(v for k, v in post.items() if k.startswith("KV-Pool (Draft"))
        dtype_key = "fp8_e4m3" if f["--kv-cache-dtype"]["wert"].startswith("fp8") else "auto"
        cell = dp["kv"]["cell_bytes_per_attn_layer_token"][dtype_key]["v"]
        # Gleitfenster 2048 x 5 Layer: NICHT der volle Kontext (Plan: AP-B L6)
        self.assertAlmostEqual(kvd["mib"], 2048 * 5 * cell / MIB, delta=0.1)
        self.assertIn("Gleitfenster 2048", kvd["formel"])
        # Spekulationszustand rechnet mit dem Verifikationsfenster 8
        sp = post["Mamba-Zwischenzustaende (Spekulation)"]
        self.assertAlmostEqual(sp["mib"], 1 * 8 * mp["state"]["per_request_bytes"]["v"] / MIB, delta=0.1)


class TestEingaben(unittest.TestCase):
    def test_fehler_benannt(self):
        mp = q27("q27_nvfp4")
        with self.assertRaises(PS.ProposeSingleError):
            PS.propose_single(mp, {"name": "x"})
        with self.assertRaises(PS.ProposeSingleError):
            PS.propose_single(mp, CARD_5090, {"unbekanntes_ziel": 1})
        with self.assertRaises(PS.ProposeSingleError):
            PS.propose_single(mp, CARD_5090, {"seats": 0})
        bad = copy.deepcopy(mp)
        del bad["weights"]["total_bytes"]
        with self.assertRaises(PS.ProposeSingleError):
            PS.propose_single(bad, CARD_5090)

    def test_fehlende_kv_zelle_ist_unbelegt_statt_geraten(self):
        mp = copy.deepcopy(q27("q27_nvfp4"))
        del mp["kv"]["variants"]
        p = PS.propose_single(mp, CARD_5090)
        self.assertEqual(p["verdikt"]["state"], "unbelegt", p["verdikt"]["text"])
        self.assertTrue(any("KV-Zelle" in u for u in p["unbelegt"]))
        self.assertFalse(p["fit"]["passt"])

    def test_kontext_ueber_modellmaximum_wird_vermerkt(self):
        p = PS.propose_single(q27("q27_nvfp4"), CARD_5090, {"context_tokens": 300000, "kv_dtype": "fp8_e4m3", "draft": "off"})
        self.assertTrue(any("max_position_embeddings" in h for h in p["fit"]["hinweise"]))


class TestServerArgsParse(unittest.TestCase):
    """Ausfuehrbarkeit: argparse von ``ServerArgs.add_cli_args`` im Kindprozess (kein GPU-Zugriff)."""

    @classmethod
    def setUpClass(cls):
        cls.rig = PS.propose_single(q27("q27_nvfp4"), CARD_5090, {"seats": 2, "host_ram_mib": 16384})
        cls.laptop = PS.propose_single(synthetic_laptop_profile(), CARD_LAPTOP, {"seats": 2, "host_ram_mib": 2000})
        cls.batch = PS.serverargs_parse_batch([
            cls.rig["argv"], cls.laptop["argv"],
            ["--model-path", "/x", "--no-such-flag", "1"],
            ["--model-path", "/x", "--kv-cache-dtype", "bogus"],
            ["--model-path", "/x", "--cuda-graph-max-bs", "8"],
        ])

    def setUp(self):
        if not self.batch.get("available"):
            self.skipTest("ServerArgs nicht importierbar: %s" % self.batch.get("error"))

    def test_vorschlaege_parsen(self):
        r = self.batch["results"]
        self.assertTrue(r[0]["ok"], r[0]["error"])
        self.assertTrue(r[1]["ok"], r[1]["error"])
        flags = r[0]["flags"]
        # Rundlauf: der geparste Wert ist der vorgeschlagene
        by = {e["flag"]: e["wert"] for e in self.rig["flags"] if e["wert"] is not None}
        self.assertEqual(flags["--max-total-tokens"]["value"], by["--max-total-tokens"])
        self.assertEqual(flags["--max-total-tokens"]["dest"], "max_total_tokens")
        self.assertEqual(flags["--kv-cache-dtype"]["value"], "fp8_e4m3")
        self.assertAlmostEqual(flags["--mem-fraction-static"]["value"], by["--mem-fraction-static"])
        self.assertEqual(flags["--max-mamba-cache-size"]["value"], by["--max-mamba-cache-size"])
        self.assertEqual(flags["--max-running-requests"]["value"], 2)
        self.assertEqual(flags["--hicache-ratio"]["value"], by["--hicache-ratio"])
        self.assertEqual(flags["--page-size"]["value"], 64)
        self.assertIs(flags["--enable-hierarchical-cache"]["value"], True)
        self.assertEqual(flags["--no-enable-multimodal"]["dest"], "enable_multimodal")
        self.assertIs(flags["--no-enable-multimodal"]["value"], False)     # --no-enable-multimodal setzt das Tri-State-Feld auf False (server_args.py:1061-1076)
        self.assertEqual(flags["--speculative-algorithm"]["value"], "NEXTN")
        self.assertFalse(any(info["deprecated"] for info in flags.values()), "kanonische Namen, kein veralteter Alias")

    def test_fehler_werden_benannt(self):
        r = self.batch["results"]
        self.assertFalse(r[2]["ok"])
        self.assertIn("--no-such-flag", r[2]["error"])
        self.assertFalse(r[3]["ok"])
        self.assertIn("--kv-cache-dtype", r[3]["error"])

    def test_veralteter_alias_wird_erkannt(self):
        row = self.batch["results"][4]
        self.assertTrue(row["ok"], row["error"])
        self.assertTrue(row["flags"]["--cuda-graph-max-bs"]["deprecated"])
        self.assertEqual(row["flags"]["--cuda-graph-max-bs"]["dest"], "cuda_graph_max_bs_decode")

    def test_apply_parse_traegt_verdikte_ein(self):
        p = copy.deepcopy(self.rig)
        ok = PS.apply_parse(p, {"available": True, "ok": True, "error": None, "flags": self.batch["results"][0]["flags"]})
        self.assertEqual(ok["verdikt"]["art"], "Planer-Rechnung + ServerArgs-Parse")
        self.assertEqual(ok["verdikt"]["state"], "passt")
        for e in ok["flags"]:
            if e["wert"] is not None:
                self.assertEqual(e["verdikt"]["parse"], "ok", e["flag"])
        bad = copy.deepcopy(self.rig)
        err = "argument --kv-cache-dtype: invalid choice: 'bogus'"
        PS.apply_parse(bad, {"available": True, "ok": False, "error": err, "flags": {}})
        fb = {e["flag"]: e for e in bad["flags"]}
        self.assertEqual(fb["--kv-cache-dtype"]["verdikt"]["parse"], "Fehler")
        self.assertEqual(fb["--kv-cache-dtype"]["verdikt"]["state"], "verweigert")
        self.assertEqual(fb["--kv-cache-dtype"]["verdikt"]["code"], "PARSE")
        self.assertEqual(fb["--max-total-tokens"]["verdikt"]["parse"], "nicht geprüft")
        self.assertEqual(bad["verdikt"]["state"], "passt nicht")

    def test_check_verbindet_beides(self):
        p = PS.check(PS.propose_single(synthetic_laptop_profile(), CARD_LAPTOP, {"seats": 2}))
        self.assertTrue(p["parse"]["available"])
        self.assertTrue(p["parse"]["ok"], p["parse"]["error"])
        self.assertEqual(p["verdikt"]["state"], "passt")


class TestParseNichtVerfuegbar(unittest.TestCase):
    def test_kein_interpreter_ist_nicht_geprueft_nie_ok(self):
        r = PS.serverargs_parse(["--model-path", "/x"], python="/nonexistent/python3")
        self.assertFalse(r["available"])
        self.assertIsNone(r["ok"])
        p = PS.propose_single(q27("q27_nvfp4"), CARD_5090)
        before = p["verdikt"]["art"]
        PS.apply_parse(p, r)
        self.assertEqual(p["verdikt"]["art"], before)                     # kein "+ ServerArgs-Parse", wenn nichts lief
        self.assertTrue(all(e["verdikt"]["parse"] == "nicht geprüft" for e in p["flags"] if e["wert"] is not None))


if __name__ == "__main__":
    unittest.main()
