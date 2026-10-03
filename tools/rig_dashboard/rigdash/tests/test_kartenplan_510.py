"""Item 510 (Kartenplaner): Katalog, PCIe/Transportwahl, Gate, Phasenaufschlüsselung, Ablehnungsgründe und
FIXTURE-ABGLEICH gegen die Planer-Aufzeichnungen der letzten echten Boots.

Gepinnt:
  * Katalog: jede Speichervariante ein eigener Eintrag, Rig-Karten mit NVML-Record (3080 20 GB = 20480, 5090 = 32607),
    fremde Karten als Datenblatt gekennzeichnet; Turing nur als deaktivierter Eintrag.
  * Transport: Rig-Preset -> barlink BAR1; Chipsatz -> ungeprüft; ungepatchter Host oder BAR1 < 168 MiB -> NCCL; eine Karte -> keiner.
  * Gate: die ORIGINAL-Planer-Funktionen (card_identity/topology aus dem Fixture-Baum 044316dd1a) lassen das Rig durch und
    verweigern 4090 (HW-ARCH), 4 Karten (HW-COUNT/HW-TOPOLOGY), 3x3090 (HW-UNCALIBRATED).
  * Fixture (Rig + 27b-int8 / nf-int4-abl): Plan = Aufzeichnung des Boots; Budgets gleich vram_plan.json und boot-JSON;
    jede Phase schließt auf die Karte (Rest < 8 MiB je Rundung); der im Record gespeicherte Planer-Nachlauf
    (launcher.budgets_from_dc) ist für alle fünf Records gleich.
  * Kein Geheimnis in der Antwort.
Der Planer selbst (Kindprozess, torch) läuft hier NICHT; sein Ergebnis steht im Record (planer_nachrechnung) und wird mit
``python3 -m kartenplan_build.bridge`` erneuert.
"""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kartenplan as K  # noqa: E402
from rigdash import kartenplan_catalog as CAT  # noqa: E402
from rigdash import kartenplan_transport as TR  # noqa: E402
from rigdash import redact  # noqa: E402

TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
P4 = {"gen": 4, "lanes": 16, "rebar": True, "chipset": False}


def rig(profile):
    kp = K.Kartenplaner(tree=TREE)
    return kp, kp.plan({"profile": profile, "cards": kp.catalog()["rig_preset"]["cards"]})


class TestKatalog(unittest.TestCase):
    def test_alle_speichervarianten_sind_eigene_eintraege(self):
        ids = {c["id"] for c in CAT.CATALOG}
        for want in ("rtx5060ti-8", "rtx5060ti-16", "rtx4060ti-8", "rtx4060ti-16", "rtx3080-10", "rtx3080-12", "rtx3080-20",
                     "rtx3090ti-24", "rtx3090-24", "rtx3080ti-12", "rtx5090-32", "rtx5080-16", "rtx5070ti-16", "rtx5070-12",
                     "rtx4090-24", "rtx4080s-16", "rtx4080-16", "rtx4070tis-16", "rtx4070ti-12", "rtx4070s-12", "rtx4070-12"):
            self.assertIn(want, ids)

    def test_rig_karten_haben_nvml_record_fremde_datenblatt(self):
        self.assertEqual(CAT.card("rtx3080-20")["usable_mib"], 20480)
        self.assertEqual(CAT.card("rtx5090-32")["usable_mib"], 32607)
        self.assertTrue(CAT.card("rtx3080-20")["usable_src"].startswith("NVML-Record"))
        self.assertTrue(CAT.card("rtx4090-24")["usable_src"].startswith("Datenblatt"))
        self.assertEqual(CAT.card("rtx3080-10")["usable_mib"], 10 * 1024)

    def test_turing_nur_als_deaktivierter_eintrag(self):
        pub = {c["id"] for c in CAT.catalog_public()}
        self.assertNotIn("rtx2080ti-11", pub)
        self.assertNotIn("rtx2080ti-22", pub)
        self.assertFalse(CAT.card("rtx2080ti-22")["enabled"])
        self.assertEqual(CAT.card("rtx2080ti-22")["arch"], "sm75")

    def test_architekturen(self):
        self.assertEqual(CAT.card("rtx5070-12")["arch"], "sm120")
        self.assertEqual(CAT.card("rtx4070-12")["arch"], "sm89")
        self.assertEqual(CAT.card("rtx3090-24")["arch"], "sm86")

    def test_profile_sind_die_beschlossenen(self):
        self.assertEqual([p["id"] for p in CAT.PROFILES], ["27b-int8", "nf-int4-abl", "27b-nvfp4-dual", "27b-fp8", "27b-gguf-iq4xs"])
        gg = CAT.profile("27b-gguf-iq4xs")["variants"]
        self.assertEqual([v["id"] for v in gg], ["UD-IQ4_XS"])
        self.assertTrue(gg[0]["boots"])


class TestTransport(unittest.TestCase):
    def test_link_rechnung(self):
        self.assertAlmostEqual(TR.link_gbs(4, 16), 31.5, delta=0.1)
        self.assertAlmostEqual(TR.link_gbs(5, 8), 31.5, delta=0.1)
        self.assertAlmostEqual(TR.link_gbs(3, 4), 3.94, delta=0.01)

    def test_bar1(self):
        self.assertEqual(TR.bar1_mib(20480, False)["mib"], 256)
        self.assertEqual(TR.bar1_mib(20480, True)["mib"], 32768)
        self.assertEqual(TR.bar1_mib(32607, True)["mib"], 32768)

    def test_slot_begrenzt_die_karte(self):
        e = CAT.card("rtx5090-32")
        l = TR.per_card_link(e, {"gen": 4, "lanes": 8, "rebar": False, "chipset": False})
        self.assertEqual((l["effective"]["gen"], l["effective"]["lanes"]), (4, 8))
        self.assertEqual(l["limited_by"], "Slot")

    def _links(self, cards, **slot):
        s = dict({"gen": 4, "lanes": 16, "rebar": False, "chipset": False}, **slot)
        ls = [TR.per_card_link(CAT.card(c), s) for c in cards]
        return ls, [CAT.label(CAT.card(c)) for c in cards]

    def test_rig_barlink_belegt(self):
        ls, lb = self._links(["rtx5090-32", "rtx3080-20", "rtx3080-20"])
        t = TR.choose_transport(ls, lb)
        self.assertEqual((t["transport"], t["confidence"]), ("bar1", "belegt"))
        self.assertTrue(any("256 MiB" in r for r in t["reasons"]))

    def test_chipsatz_macht_barlink_ungeprueft(self):
        ls, lb = self._links(["rtx5090-32", "rtx3080-20", "rtx3080-20"])
        ls[1] = TR.per_card_link(CAT.card("rtx3080-20"), {"gen": 4, "lanes": 4, "rebar": False, "chipset": True})
        t = TR.choose_transport(ls, lb)
        self.assertEqual((t["transport"], t["confidence"]), ("bar1", "ungeprüft"))
        self.assertTrue(any("Chipsatz" in w for w in t["warnings"]))

    def test_ungepatchter_host_nccl(self):
        ls, lb = self._links(["rtx3090-24", "rtx3090-24"])
        t = TR.choose_transport(ls, lb, host_patched=False)
        self.assertEqual(t["transport"], "nccl")
        self.assertTrue(any("595.58.03" in r for r in t["reasons"]))

    def test_zu_kleines_bar1_nccl(self):
        ls, lb = self._links(["rtx3090-24", "rtx3090-24"])
        for l in ls:
            l["bar1_mib"] = 128
        t = TR.choose_transport(ls, lb)
        self.assertEqual(t["transport"], "nccl")
        self.assertTrue(any("168 MiB" in r for r in t["reasons"]))

    def test_eine_karte_kein_transport(self):
        ls, lb = self._links(["rtx5090-32"])
        self.assertEqual(TR.choose_transport(ls, lb)["transport"], "keiner")

    def test_ungueltige_pcie_angaben(self):
        with self.assertRaises(ValueError):
            TR.normalize_slot({"gen": 2, "lanes": 16})
        with self.assertRaises(ValueError):
            TR.normalize_slot({"gen": 4, "lanes": 2})


class TestGateUndAblehnung(unittest.TestCase):
    def setUp(self):
        self.kp = K.Kartenplaner(tree=TREE)

    def plan(self, profile, cards):
        return self.kp.plan({"profile": profile, "cards": [{"card": c, "pcie": P4} for c in cards]})

    def codes(self, r):
        return {x["code"] for x in r["verdict"]["reasons"]}

    def test_gate_laedt_original_planer_module(self):
        r = self.plan("27b-int8", ["rtx5090-32", "rtx3080-20", "rtx3080-20"])
        self.assertTrue(r["gate"]["available"])
        self.assertIn("card_identity.py", r["gate"]["tree_card_identity"])
        self.assertEqual([o["class"] for o in r["gate"]["order"]], ["RTX5090", "RTX3080", "RTX3080"])

    def test_4090_wird_vom_planer_verweigert(self):
        r = self.plan("27b-int8", ["rtx5090-32", "rtx3080-20", "rtx4090-24"])
        self.assertFalse(r["verdict"]["goes"])
        self.assertIn("HW-ARCH", self.codes(r))
        self.assertTrue(any("compute capability 8.9" in x["text"] for x in r["verdict"]["reasons"]))
        self.assertNotIn("plan", r)
        self.assertIn("naeherung", r)

    def test_vier_karten_count_und_topology(self):
        r = self.plan("27b-int8", ["rtx3090-24"] * 4)
        self.assertTrue({"HW-COUNT", "HW-TOPOLOGY", "HW-UNCALIBRATED"} <= self.codes(r))

    def test_sechs_karten(self):
        r = self.plan("nf-int4-abl", ["rtx5090-32"] * 6)
        self.assertFalse(r["verdict"]["goes"])
        self.assertIn("HW-TOPOLOGY", self.codes(r))

    def test_drei_3090_unkalibriert(self):
        r = self.plan("27b-int8", ["rtx3090-24"] * 3)
        self.assertEqual(self.codes(r), {"HW-UNCALIBRATED"})

    def test_3080_10gb_leiht_keine_20gb_zahlen(self):
        r = self.plan("27b-int8", ["rtx5090-32", "rtx3080-10", "rtx3080-10"])
        self.assertIn("HW-UNCALIBRATED", self.codes(r))

    def test_sm75_wird_abgelehnt_mit_grund(self):
        r = self.plan("27b-int8", ["rtx5090-32", "rtx3080-20", "rtx2080ti-22"])
        self.assertFalse(r["verdict"]["goes"])
        self.assertIn("ARCH-sm75", self.codes(r))
        txt = " ".join(x["text"] for x in r["verdict"]["reasons"])
        self.assertIn("kein bf16", txt)

    def test_sm89_status_ehrlich(self):
        st = CAT.arch_status([8, 9], "fp8")
        self.assertFalse(st["ok"])
        self.assertIn("FP8-SM89-FALLBACK", st["why"])
        self.assertIn("ungebaut", st["level"])

    def test_zu_viele_oder_keine_karten(self):
        with self.assertRaises(ValueError):
            self.plan("27b-int8", ["rtx3090-24"] * 7)
        with self.assertRaises(ValueError):
            self.plan("27b-int8", [])

    def test_gate_fehlt_wird_benannt(self):
        kp = K.Kartenplaner(tree="/nicht/da")
        kp.tree = None
        r = kp.plan({"profile": "27b-int8", "cards": [{"card": "rtx5090-32", "pcie": P4}]})
        self.assertIn("GATE-FEHLT", {x["code"] for x in r["verdict"]["reasons"]})

    def test_naeherung_ist_gekennzeichnet_und_ohne_plan(self):
        r = self.plan("27b-int8", ["rtx3090-24"] * 4)
        n = r["naeherung"]
        self.assertIn("KEIN Planer-Ergebnis", n["label"])
        self.assertNotIn("docker_run", json.dumps(n))
        self.assertTrue(any("geschätzt" in t for t in n["notes"]))

    def test_eine_5090_passt_rechnerisch_nicht(self):
        r = self.plan("27b-int8", ["rtx5090-32"])
        self.assertFalse(r["naeherung"]["fits"])


class TestFixtureAbgleich(unittest.TestCase):
    """Rig-Setup + Profil muss den Plan des letzten echten Boots reproduzieren."""

    CASES = {"27b-int8": ("D", [27792, 17384, 17168], "P", [26344, 8312, 13864]),
             "nf-int4-abl": ("D", [26824, 17672, 17472], "P", [28440, 17568, 18200])}

    def test_budgets_gleich_vram_plan_und_boot_log(self):
        for prof, (g1, b1, g2, b2) in self.CASES.items():
            kp, r = rig(prof)
            self.assertTrue(r["verdict"]["goes"], prof)
            bars = r["plan"]["einfach"]["bars"]
            self.assertEqual([b[g1]["budget"]["budget_mib"] for b in bars], b1, prof)
            self.assertEqual([b[g2]["budget"]["budget_mib"] for b in bars], b2, prof)
            rec = K.load_record(prof)
            self.assertEqual(rec["budgets_final"][g1] if g1 in rec["budgets_final"] else b1, b1)
            self.assertEqual(rec["budgets_final"][g2], b2)

    def test_phasen_schliessen_auf_die_karte(self):
        for prof in self.CASES:
            kp, r = rig(prof)
            for b in r["plan"]["einfach"]["bars"]:
                for g in ("P", "D"):
                    self.assertAlmostEqual(b[g]["rest_mib"], 0, delta=8, msg="%s %s Ordinal %s" % (prof, g, b["ordinal"]))
                    self.assertEqual(sum(s["mib"] for s in b[g]["segments"]) + b[g]["rest_mib"], b["total_mib"])

    def test_phasenaufschluesselung_hat_beide_layouts_und_spitze(self):
        kp, r = rig("27b-int8")
        x = r["plan"]["experte"]
        self.assertEqual(set(x["phases"]), {"P", "D"})
        self.assertEqual(len(x["peak"]), 3)
        keys_d = {s["key"] for s in x["phases"]["D"][0]["segments"]}
        keys_p = {s["key"] for s in x["phases"]["P"][1]["segments"]}
        self.assertTrue({"weights", "draft", "kv", "state", "graphs", "carve", "asleep"} <= keys_d)
        self.assertIn("l15", keys_p)     # der L1.5-Post der 3080 steht im P-Layout

    def test_nf_experten_fuellen_freien_vram(self):
        kp, r = rig("nf-int4-abl")
        d0 = {s["key"]: s["mib"] for s in r["plan"]["experte"]["phases"]["D"][0]["segments"]}
        self.assertGreater(d0["experts_lru"], 10000)

    def test_flags_mit_erklaerung_und_bound_by(self):
        kp, r = rig("27b-int8")
        f = r["plan"]["experte"]["flags"]
        d = {x["name"]: x for x in f["groups"]["D"]}
        self.assertEqual(d["--rank-gpu-memory-mib"]["value"], "27792,17384,17168")
        self.assertEqual(d["--rank-gpu-memory-mib"]["set_by"], "Planer")
        self.assertIn("budgets_from_dc", d["--rank-gpu-memory-mib"]["bound_by"])
        self.assertTrue(d["--tp-size"]["explained"])
        self.assertEqual(d["--tp-size"]["value"], "3")
        self.assertGreater(f["totals"]["explained"], 50)

    def test_context_und_sitze(self):
        kp, r = rig("27b-int8")
        c = r["plan"]["einfach"]["context"]
        self.assertEqual((c["context_tokens"], c["seats_d"], c["seats_p"]), (262144, 6, 2))

    def test_docker_run_traegt_profil_image_transport(self):
        kp, r = rig("nf-int4-abl")
        d = r["plan"]["einfach"]["docker_run"]
        self.assertIn("HTSGLANG_PROFILE=nf-int4-h6-abl", d["docker_run"])
        self.assertIn("HTSGLANG_TRANSPORT=bar1", d["docker_run"])
        self.assertIn("/mnt/nf-experts", d["docker_run"])
        self.assertIn("htsglang:cu130-weg2", d["docker_run"])

    def test_pcie_aenderung_aendert_die_budgets_nicht(self):
        kp = K.Kartenplaner(tree=TREE)
        cards = [{"card": c, "pcie": P4} for c in ("rtx5090-32", "rtx3080-20", "rtx3080-20")]
        a = kp.plan({"profile": "27b-int8", "cards": cards})
        b = kp.catalog()["rig_preset"]["cards"]
        c = kp.plan({"profile": "27b-int8", "cards": b})
        ba = [x["D"]["budget"]["budget_mib"] for x in a["plan"]["einfach"]["bars"]]
        bc = [x["D"]["budget"]["budget_mib"] for x in c["plan"]["einfach"]["bars"]]
        self.assertEqual(ba, bc)

    def test_karten_reihenfolge_egal(self):
        kp = K.Kartenplaner(tree=TREE)
        r = kp.plan({"profile": "27b-int8", "cards": [{"card": c, "pcie": P4} for c in ("rtx3080-20", "rtx3080-20", "rtx5090-32")]})
        self.assertTrue(r["verdict"]["goes"])
        self.assertEqual(r["plan"]["einfach"]["bars"][0]["card_label"], "RTX 5090 32 GB")


class TestRecords(unittest.TestCase):
    IDS = ("27b-int8", "nf-int4-abl", "27b-nvfp4-dual", "27b-fp8", "27b-gguf-iq4xs")

    def test_alle_records_da_planer_nachgerechnet_gleich(self):
        for rid in self.IDS:
            rec = K.load_record(rid)
            self.assertIsNotNone(rec, rid)
            n = rec["planer_nachrechnung"]
            self.assertTrue(n["all_match"], rid)
            self.assertTrue(n["rows"], rid)
            for row in n["rows"]:
                self.assertEqual(row["got_mib"], row["expected_mib"], "%s %s" % (rid, row["id"]))

    def test_plan_id_wird_nachgerechnet(self):
        for rid in ("27b-int8", "nf-int4-abl", "27b-nvfp4-dual"):
            rec = K.load_record(rid)
            self.assertTrue(K.plan_id_ok(rec["vram_plan"]), rid)
        rec = K.load_record("27b-int8")
        rec["vram_plan"]["cards"][0]["total_mib"] += 1
        self.assertFalse(K.plan_id_ok(rec["vram_plan"]))

    def test_kein_geheimnis_in_den_antworten(self):
        for prof in ("27b-int8", "nf-int4-abl"):
            kp, r = rig(prof)
            body = json.dumps(r)
            # redact.guard darf höchstens eine Kommentarstelle schwärzen (Profil-Kommentar nennt WEG2-GROUP-ENV), nie einen Wert
            self.assertEqual(len(redact.guard(body)) - len(body) <= 0, True)
            self.assertNotIn("adminkey", body.lower())
            self.assertNotIn("admin-api-key", body.lower())

    def test_service_paket_hat_keinen_log_parser(self):
        # der Dienst liest nur JSON-Records; die Log-Leser liegen im Schreibtisch-Werkzeug kartenplan_build/
        pkg = os.path.dirname(HERE)
        for fn in sorted(os.listdir(pkg)):
            if fn.startswith("kartenplan") and fn.endswith(".py"):
                self.assertNotIn("re.compile", open(os.path.join(pkg, fn)).read(), fn)


if __name__ == "__main__":
    unittest.main()
