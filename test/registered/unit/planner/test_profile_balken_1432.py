"""PROFIL-EDITOR S4b (Auftrag 1432): Balken je Karte und Phase, Ueberlauf als eigenes Segment, Browser-Naeherung (Python-Referenz).

Gepinnt (GPU-frei, Modellprofile aus den Config-Fixtures von Auftrag 960):

* GOLDEN gegen die Aufzeichnung des Referenz-Boots NF INT4-abl (``kartenplan_data/nf-int4-abl.json``, ``rank_posts.P``): mit den GEMESSENEN
  Eingaben des Boots (Aktivierungsspitze je Stufe 5984 / 2952 / 2944 MiB, SSM bfloat16, 32 Slots, Budgets 28440 / 17568 / 18200) verweigert die
  Engine das Profil NICHT -- der Boot ist gelaufen --, und der Rest im Budget liegt neben dem aufgezeichneten ``rest_mib`` (3883 / 2657 / 3506)
  innerhalb der benannten Toleranz;
* ohne die gemessenen Eingaben ist die Aktivierung Geometrie-Naeherung (eine Rate fuer alle Stufen) und die Engine zeigt auf den 3080 einen
  UEBERLAUF, den der Boot nicht hatte: das ist die benannte Grenze der Naeherung, kein Messwert (Test pinnt, dass der Ueberlauf gezeigt wird);
* Summenregel des Balkens: Summe der Segmente == Kartengroesse (ohne Ueberlauf) bzw. Kartengroesse + Ueberlauf (mit);
* Ueberlauf ist ein eigenes Segment ``overflow`` mit den betroffenen Posten, nie still beschnitten;
* Phasen P/D und Spitze; Browser-Naeherung (``approx_terms``) gegen die Server-Antwort innerhalb 0,5 MiB bei Verschiebung des Layer-Schnitts.
"""

import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.planner import profile_couplings as PC  # noqa: E402
from flliper.srt.pdflip import model_profile as MP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "pdflip", "fixtures", "profil_s3_1003")
RIG3 = [("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)]
_M = {}


def model(name):
    if name not in _M:
        _M[name] = MP.estimate(os.path.join(FX, name))
    return _M[name]


def hw3():
    return PC.synthetic_hardware(RIG3)


# Aufzeichnung des Referenz-Boots (rank_posts.P.ppN.kv_posts)
REC = {"rest_mib": [3883, 2657, 3506], "transient_mib": [5984, 2952, 2944], "budget_mib": [28440, 17568, 18200]}
REC_TOL_MIB = 1300.0     # Engine rechnet Gewichte/Experten aus der Config (-4 %) und die KV-Zelle mit Skalenpuffer (+5 %): benannte Toleranz


def ref_settings(**kw):
    s = {"stage_layers": [29, 11, 8], "budget_mib": REC["budget_mib"], "kv_dtype": "fp8_e4m3", "context_tokens": 262144, "chunk_tokens": 16384,
         "mamba_slots": 32, "scratch_rows": 32, "moe_resident_fraction": [0.33, 0.701, 0.995605]}
    s.update(kw)
    return s


def measured(**kw):
    return ref_settings(ssm_dtype="bfloat16", activation_mib=REC["transient_mib"], **kw)


class TestGoldenReferenceBoot(unittest.TestCase):
    def setUp(self):
        self.bars = PC.bars_for(hw3(), model("nextflash_int4mixed"), measured())["phases"]["alle"]["bars"]

    def test_with_the_boots_own_measurements_the_profile_fits_as_the_boot_did(self):
        for b in self.bars:
            self.assertEqual(b["overflow_mib"], 0.0, b["label"])
            self.assertGreater(b["free_mib"], 0.0, b["label"])

    def test_free_in_budget_is_near_the_recorded_rest(self):
        for b, rec in zip(self.bars, REC["rest_mib"]):
            self.assertLess(abs(b["free_mib"] - rec), REC_TOL_MIB, "%s: Engine %.0f MiB, Aufzeichnung %d MiB" % (b["label"], b["free_mib"], rec))
            self.assertLessEqual(b["free_mib"], rec + 1.0, "die Engine darf nicht grosszuegiger sein als der Boot (Obergrenze ohne Festposten ist sie ohnehin)")

    def test_activation_and_state_segments_carry_the_recorded_values(self):
        for b, tr in zip(self.bars, REC["transient_mib"]):
            seg = {s["key"]: s for s in b["segments"]}
            self.assertAlmostEqual(seg["activation"]["mib"], tr, delta=0.01)
            self.assertEqual(seg["activation"]["src"], PC.SRC_INPUT)
        st = [next(s for s in b["segments"] if s["key"] == "state")["mib"] for b in self.bars]
        for mine, rec in zip(st, (1098, 399, 299)):
            self.assertAlmostEqual(mine / rec, 1.0, delta=0.02)

    def test_geometry_approximation_without_measurements_shows_the_overflow_it_cannot_rule_out(self):
        bars = PC.bars_for(hw3(), model("nextflash_int4mixed"), ref_settings())["phases"]["alle"]["bars"]
        over = [b for b in bars if b["overflow_mib"] > 0]
        self.assertTrue(over, "eine Rate fuer alle Stufen ueberschaetzt die Aktivierung der 3080: der Ueberlauf muss SICHTBAR sein, nicht verschwiegen")
        for b in over:
            seg = {s["key"]: s for s in b["segments"]}
            self.assertIn("overflow", seg)
            self.assertAlmostEqual(seg["overflow"]["mib"], b["overflow_mib"], delta=0.01)
            self.assertTrue(seg["overflow"]["cut"], "der Ueberlauf nennt die betroffenen Posten")


class TestBarContract(unittest.TestCase):
    def test_segments_sum_to_the_card_plus_overflow(self):
        for tight in (None, [30000, 3000, 3000]):
            s = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 100000, "chunk_tokens": 2048, "mamba_slots": 4}
            if tight:
                s["budget_mib"] = tight
            for b in PC.bars_for(hw3(), model("qwen27b_int8_vocabembed"), s)["phases"]["alle"]["bars"]:
                total = sum(g["mib"] for g in b["segments"])
                self.assertAlmostEqual(total, b["total_mib"] + b["overflow_mib"], delta=0.02, msg=b["label"])

    def test_overflow_is_its_own_segment_and_named_in_the_hint(self):
        s = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 100000, "budget_mib": [30000, 3000, 3000]}
        r = PC.bars_for(hw3(), model("qwen27b_int8_vocabembed"), s)
        bars = r["phases"]["alle"]["bars"]
        self.assertEqual(bars[0]["overflow_mib"], 0.0)
        for b in bars[1:]:
            self.assertGreater(b["overflow_mib"], 0.0)
            ov = [g for g in b["segments"] if g["key"] == "overflow"]
            self.assertEqual(len(ov), 1)
            self.assertAlmostEqual(sum(c["mib"] for c in ov[0]["cut"]), ov[0]["mib"], delta=0.01)
            self.assertNotIn("free_in_budget", [g["key"] for g in b["segments"]])
        self.assertTrue(any("over the budget" in h for h in r["hints"]))
        self.assertTrue(any("force" in h for h in r["hints"]))

    def test_no_free_segment_when_overflowing_and_no_overflow_segment_when_fitting(self):
        s = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 4096}
        for b in PC.bars_for(hw3(), model("qwen27b_int8_vocabembed"), s)["phases"]["alle"]["bars"]:
            keys = [g["key"] for g in b["segments"]]
            self.assertNotIn("overflow", keys)
            self.assertIn("free_in_budget", keys)
            self.assertIn("corridor", keys)

    def test_every_segment_names_its_origin(self):
        s = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "budget_mib": [30000, 19000, 19000]}
        for b in PC.bars_for(hw3(), model("qwen27b_int8_vocabembed"), s)["phases"]["alle"]["bars"]:
            for g in b["segments"]:
                self.assertTrue(g["origin"] and g["what"], g["key"])


class TestPhases(unittest.TestCase):
    def test_peak_is_the_larger_phase_per_card(self):
        m = model("qwen27b_int8_vocabembed")
        base = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 50000}
        r = PC.bars_for(hw3(), m, base, phases={"P": {"chunk_tokens": 8192}, "D": {"chunk_tokens": 512}})
        self.assertEqual(set(r["phases"]), {"P", "D"})
        for i, pk in enumerate(r["Spitze"]["bars"]):
            self.assertEqual(pk["from_phase"], "P")        # grosser Chunk = hoehere Spitze
            self.assertEqual(pk["needs_mib"], max(r["phases"][p]["bars"][i]["needs_mib"] for p in ("P", "D")))
        self.assertLess(r["phases"]["D"]["bars"][0]["needs_mib"], r["phases"]["P"]["bars"][0]["needs_mib"])

    def test_single_phase_has_no_peak(self):
        r = PC.bars_for(hw3(), model("qwen27b_int8_vocabembed"), {"stage_layers": [32, 16, 16]})
        self.assertNotIn("Spitze", r)


class TestBrowserApproximation(unittest.TestCase):
    TOL_MIB = 0.5      # benannte Toleranz: Rundung der Nutzlast (4 Nachkommastellen je Layer) ueber 64 Layer

    def check(self, m, base, counts_list):
        pl = PC.approx_payload(hw3(), m, base)
        for counts in counts_list:
            server = PC.c1_layer_split(hw3(), m, dict(base, stage_layers=counts))["stages"]
            mine = PC.approx_terms(pl, counts)
            for sv, ap in zip(server, mine):
                self.assertAlmostEqual(ap["needs"], sv["needs_mib"], delta=self.TOL_MIB, msg=str(counts))
                self.assertAlmostEqual(ap["free"], sv["free_mib"], delta=self.TOL_MIB, msg=str(counts))
                for k in ("weights", "experts", "kv", "state", "activation"):
                    self.assertAlmostEqual(ap[k], sv["terms"][k]["v"], delta=self.TOL_MIB, msg="%s %s" % (counts, k))

    def test_moved_layers_dense_model(self):
        base = {"stage_layers": [32, 16, 16], "kv_dtype": "fp8_e4m3", "context_tokens": 100000, "mamba_slots": 4}
        self.check(model("qwen27b_int8_vocabembed"), base, [[32, 16, 16], [34, 15, 15], [28, 20, 16], [20, 22, 22], [40, 12, 12]])

    def test_moved_layers_moe_with_draft_and_measured_inputs(self):
        base = measured(draft=True)
        self.check(model("nextflash_int4mixed"), base, [[29, 11, 8], [31, 10, 7], [27, 12, 9], [24, 14, 10]])

    def test_approximation_is_not_the_servers_solver(self):
        # eine geaenderte Experten-Fraction ist NICHT Teil der Naeherung (sie bleibt die des Servers): ein neuer Server-Aufruf gibt andere Werte
        m = model("nextflash_int4mixed")
        pl = PC.approx_payload(hw3(), m, measured())
        s2 = measured(moe_resident_fraction=[0.5, 0.5, 0.5])
        server = PC.c1_layer_split(hw3(), m, s2)["stages"][0]["needs_mib"]
        self.assertGreater(abs(PC.approx_terms(pl, [29, 11, 8])[0]["needs"] - server), 100.0)


class TestRunBars(unittest.TestCase):
    def test_run_bars_is_json_and_carries_the_approx_payload(self):
        req = {"what": "bars", "hardware": hw3(), "model": model("nextflash_int4mixed"), "settings": measured(),
               "phases": {"P": {"chunk_tokens": 16384}, "D": {"chunk_tokens": 1024, "activation_mib": [900, 900, 900]}}}
        res = PC.run(req)
        self.assertTrue(res["ok"], res)
        json.dumps(res)
        r = res["result"]
        self.assertEqual(set(r["phases"]), {"P", "D"})
        self.assertIn("Spitze", r)
        self.assertEqual(r["approx"]["n_stages"], 3)

    def test_run_bars_bad_input_is_ok_false(self):
        bad = PC.run({"what": "bars", "hardware": hw3(), "model": model("nextflash_int4mixed"), "settings": {"stage_layers": [29, 11]}})
        self.assertFalse(bad["ok"])
        self.assertIn("vector_length", bad["error"])

    def test_unknown_ssm_dtype_is_refused_by_name(self):
        with self.assertRaises(PC.CouplingError):
            PC.c1_layer_split(hw3(), model("nextflash_int4mixed"), ref_settings(ssm_dtype="int3"))


if __name__ == "__main__":
    unittest.main()
