"""AP-I (Profil-Planer 06.10.): Issue-Text "Laufbericht" des Profil-Editors.  Kein Rig, keine GPU, kein gpuq, kein Docker.

Geprüft wird: (1) der Text enthält ALLE Blöcke (Hardwareprofil Kurzform, Modellprofil, Betriebsform, Vorschlag + Übersteuerungen, Verdikte/Force,
Versionen, Platzhalter Messergebnis/Boot-Log-Auszug) und die Werte darin stammen aus Profil, Trockenlauf, Hardware- und Modellprofil;
(2) er ist redigiert: kein Geheimnis (nach Wert UND nach Name der Zeile), kein Hostpfad; (3) der Vorschlags-Block kommt aus dem Profil-JSON
(planner_value / profile_value / origin) und nimmt zusätzliche Felder (state, verdict) einer Zeile ohne Umbau auf; (4) die Forcebarkeit wird aus
dem Register neu gelesen, nicht aus der Browserantwort; (5) Route und Oberfläche (node mit DOM-Attrappe).
"""

import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import hwprofil, redact  # noqa: E402
from rigdash import kartenplan as K  # noqa: E402
from rigdash import profil as P  # noqa: E402
from rigdash import server as S  # noqa: E402

FIXTURE_TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
REPO_CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
STATIC = os.path.join(os.path.dirname(HERE), "static")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)

SECRET_VALUE = "hf_GEHEIM1234567890"
ENV = """\
# shellcheck shell=bash
PROFILE_NAME=demo
PROFILE_LINE=nf
PROFILE_FORMAT=int4-mixed
PROFILE_STATUS=experimentell
PROFILE_OWNER="the owner"
PROFILE_CARD_COUNT=3
PROFILE_INVENTORY=RTX5090,RTX3080,RTX3080
PROFILE_ARGS=(--model /spinning/llm_stuff/models-cache/Qwen3.8-27B --p-bs 2 --pp-stage-ratio 29,11,8 --p-hostgap
              "--extra-p=--rank-moe-ratio 183,137,168" --env-p "SGLANG_MOE_SCRATCH_SLOTS=74,48,48")
profile_form_env() {
  _form SGLANG_WEG2_OWNED_BASE stated
}
"""
RIG = [{"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 4}}, {"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8}},
       {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}]


def editor(tmp):
    rel = os.path.join(tmp, "rel")
    os.makedirs(rel)
    with open(os.path.join(rel, "demo.env"), "w") as fh:
        fh.write(ENV)
    return P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=FIXTURE_TREE), release_dir=rel, user_dir=os.path.join(tmp, "u"), tree=FIXTURE_TREE,
                          catalog_file=REPO_CATALOG)


def _n(v, src="NVML", unit=None):
    return {"v": v, "src": src, **({"unit": unit} if unit else {})}


HW = {"schema": "flliper.hardware/1", "id": "sha256:abcdef0123456789abcdef", "driver": "575.57.08", "cuda": "13.0", "torch": "2.9.1",
      "cards": [
          {"ord": 0, "nvml_index": 1, "name": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "sm_count": _n(170, "Datenblatt"), "vram_total_mib": _n(32607, unit="MiB"),
           "pcie": {"max_gen": _n(5), "max_width": _n(16)}, "mem_gbs": {"read": _n(1650, "gemessen", "GB/s"), "nominal": _n(1792, "Datenblatt", "GB/s")}},
          {"ord": 1, "nvml_index": 0, "name": "NVIDIA GeForce RTX 3080", "cc": [8, 6], "sm_count": _n(68, "Datenblatt"), "vram_total_mib": _n(20480, unit="MiB"),
           "pcie": {"max_gen": _n(4), "max_width": _n(16)}, "mem_gbs": {"nominal": _n(760, "Datenblatt", "GB/s")}},
      ]}
VERSIONS = {"tree_rev": "173161c595de23e0", "image": "ghcr.io/efschu/htsglang:0.1.0-cu130", "driver": "575.57.08", "cuda": "13.0", "torch": "2.9.1", "rigdash": "r1006"}
MODEL = {"schema": "flliper.model/1", "path": "/spinning/llm_stuff/models-cache/Qwen3.8-27B", "config_sha": "0123456789abcdef",
         "format": _n("int8-w8a8", "Index"),
         "arch": {"family": _n("dense", "config"), "hybrid": _n(True, "config"), "n_layers": _n(64, "config"), "layer_counts": _n({"attn": 16, "gdn": 48, "mamba": 0}, "config"),
                  "hidden": _n(5120, "config"), "heads_q": _n(24, "config"), "heads_kv": _n(4, "config"), "head_dim": _n(256, "config"), "attention": _n("full", "config")},
         "weights": {"total_bytes": _n(29 * (1 << 30), "Index")},
         "kv": {"cell_bytes_per_attn_layer_token": _n(1088, "geschätzt")},
         "state": {"per_linear_layer_per_slot_mib": _n(1.5, "geschätzt")},
         "draft": {"mtp_layers": _n(1, "config")}, "context": {"max_position_embeddings": _n(262144, "config")}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pfi1006_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ed = editor(self.tmp)
        self.doc = self.ed.load("release", "demo")["doc"]

    def edited(self, edits):
        return self.ed.edit(self.doc, edits)["doc"]

    def report(self, doc=None, dry="auto", **kw):
        doc = doc or self.doc
        if dry == "auto":
            dry = self.ed.dry_run(doc, RIG[:2])
        kw.setdefault("cards", RIG[:2])
        kw.setdefault("model", MODEL)
        kw.setdefault("hardware_md", hwprofil.issue_short(HW))
        kw.setdefault("versions", VERSIONS)
        kw.setdefault("now", 1791000000.0)
        return self.ed.issue_report(doc, dry=dry, **kw)


class AllBlocks(Base):
    def test_every_block_is_there_in_order(self):
        r = self.report()
        self.assertTrue(r["ok"])
        self.assertEqual(r["format"], "markdown")
        t = r["text"]
        pos = []
        for b in P.ISSUE_BLOCKS:
            self.assertIn("### " + b, t, b)
            pos.append(t.index("### " + b))
        self.assertEqual(pos, sorted(pos), "Blöcke in fester Reihenfolge")
        self.assertEqual(r["blocks"], list(P.ISSUE_BLOCKS))
        self.assertTrue(t.startswith("## Laufbericht (Profil-Editor): `demo`"))
        self.assertEqual(r["filename"], "laufbericht-demo.md")

    def test_hardware_short_form_is_the_ap_a_block_and_not_the_long_form(self):
        t = self.report()["text"]
        self.assertIn("RTX 5090", t)
        self.assertIn("170 (Datenbl.)", t)                         # SM-Zahl mit Herkunft: dieselben Zellenbausteine wie issue_text
        self.assertIn("1650 GB/s (gem.)", t)
        self.assertIn("32607 MiB (NVML)", t)
        self.assertIn("Gen5 x16", t)
        self.assertNotIn("### Karten (NVML-Identität)", t)          # die Langform bleibt im Hardware-Abschnitt
        self.assertNotIn("UUID", t)

    def test_hardware_missing_says_so(self):
        t = self.report(hardware_md="")["text"]
        self.assertIn("### Hardwareprofil (Kurzform)", t)
        self.assertIn("Hardwareprofil nicht verfügbar (unbelegt)", t)

    def test_model_block_carries_values_with_source_and_only_the_folder_name(self):
        t = self.report()["text"]
        self.assertIn("`Qwen3.8-27B`", t)
        self.assertIn("29.00 GiB (Index)", t)
        self.assertIn("dicht (config)", t)
        self.assertIn("attn 16, gdn 48 (config)", t)
        self.assertIn("24 / 4 / 256 (config)", t)
        self.assertIn("1088 B (geschätzt)", t)
        self.assertIn("262144 (config)", t)
        self.assertNotIn("models-cache", t)

    def test_no_model_profile_names_the_model_of_the_profile(self):
        t = self.report(model=None)["text"]
        self.assertIn("Kein Modellprofil geschätzt", t)
        self.assertIn("`Qwen3.8-27B`", t)                           # --model des Profils, nur der Ordnername
        t2 = self.report(model={"schema": "falsch"})["text"]
        self.assertIn("Kein Modellprofil geschätzt", t2)

    def test_betriebsform_block(self):
        t = self.report()["text"]
        self.assertIn("| Betriebsform | Flip PP/TP |", t)
        self.assertIn("| Abgeleitet aus | weder --d-only noch --dual-* im Profil, also die Standardform des Launchers (aus den Flags des Profils gelesen, keine Wahl des Planers) |", t)
        self.assertIn("| Linie | nf |", t)
        self.assertIn("| Profilstand | experimentell |", t)
        self.assertIn("| Karten (Trockenlauf) | 2: ", t)
        self.assertIn("| Kartenzahl laut Profil | 3 |", t)
        self.assertIn("| Inventar laut Profil | RTX5090,RTX3080,RTX3080 |", t)
        self.assertIn("Basis release `demo`", t)

    def test_versions_block(self):
        t = self.report()["text"]
        for want in ("| Baum (Revision) | 173161c595de23e0 |", "| Image | ghcr.io/efschu/htsglang:0.1.0-cu130 |", "| Treiber | 575.57.08 |",
                     "| CUDA / torch (Messprozess) | 13.0 / 2.9.1 |", "| Dashboard | r1006 |"):
            self.assertIn(want, t)
        self.assertRegex(t, r"\| Basisprofil \(sha256\) \| `[0-9a-f]{16}` \|")

    def test_versions_missing_are_unbelegt_never_guessed(self):
        os.environ.pop("SGLANG_IMAGE_TAG", None)
        t = self.report(versions={})["text"]
        self.assertIn("| Baum (Revision) | unbelegt |", t)
        self.assertIn("| Image | unbelegt (SGLANG_IMAGE_TAG nicht gesetzt) |", t)
        self.assertIn("| Treiber | unbelegt |", t)

    def test_placeholder_for_measurement_and_boot_log(self):
        t = self.report()["text"]
        tail = t.split("### Messergebnis / Boot-Log-Auszug")[1]
        self.assertIn("```text", tail)
        self.assertIn("(Boot-Log-Auszug hier einfügen)", tail)
        self.assertIn("Ergebnis:", tail)
        self.assertIn("<!--", tail)
        self.assertEqual(t.count("```"), 2)                         # der Block ist geschlossen: ein Issue-Rendern bleibt heil


class VerdictsAndForce(Base):
    def test_no_dry_run_is_said_not_hidden(self):
        t = self.report(dry=None)["text"]
        self.assertIn("Kein Trockenlauf gefahren", t)
        self.assertIn("Noch kein Trockenlauf für dieses Profil", t)       # force_hint "kein_trockenlauf"
        self.assertIn("| Karten (gewählt, noch kein Trockenlauf) | 2: ", t)

    def test_rejections_with_register_state(self):
        dry = self.ed.dry_run(self.doc, RIG[:2])
        codes = {q["code"] for q in dry["rejections"]}
        self.assertIn("HW-COUNT", codes)
        t = self.report(dry=dry)["text"]
        self.assertIn("Der Planer lehnt", t)
        self.assertIn("| `HW-COUNT` |", t)
        self.assertIn("FLLIPER_FORCE=1", t)                               # forcebar im Register: die Force-Zeile steht da
        row = next(x for x in t.split("\n") if x.startswith("| `HW-COUNT`"))
        self.assertIn("force:", row)

    def test_a_forged_force_claim_from_the_browser_is_not_believed(self):
        forged = {"ok": True, "verdict": "alles gut", "rejections": [{"code": "HW-ARCH", "text": "HW-ARCH: x", "forcebar": True, "force_state": "force"},
                                                                       {"code": "ERFUNDEN-1", "text": "?", "forcebar": True, "force_state": "force"}], "notes": []}
        t = self.report(dry=forged)["text"]
        row = next(x for x in t.split("\n") if x.startswith("| `HW-ARCH`"))
        self.assertIn("blockiert", row)
        self.assertNotIn("force:", row)
        row2 = next(x for x in t.split("\n") if x.startswith("| `ERFUNDEN-1`"))
        self.assertIn("unbekannter Code", row2)
        self.assertIn("bleiben auch mit Force bestehen", t.replace("Auch mit Force bestehen bleiben", "bleiben auch mit Force bestehen"))
        self.assertNotIn("Beim Serverstart `FLLIPER_FORCE=1` setzen", t)  # nichts forcebar

    def test_clean_dry_run(self):
        t = self.report(dry={"ok": True, "verdict": "Der Planer lehnt dieses Profil auf den gewählten Karten nicht ab.", "rejections": [], "notes": ["Hinweis eins"]})["text"]
        self.assertIn("nicht ab.", t)
        self.assertIn("Der Planer lehnt nichts ab; Force wird nicht gebraucht.", t)
        self.assertIn("- Hinweis: Hinweis eins", t)


class Proposal(Base):
    def test_diff_comes_from_the_profile_json(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "set", "value": "4"}])
        doc["meta"]["planner"]["flag:--p-bs"] = "6"
        doc["meta"]["planner"]["flag:--pp-stage-ratio"] = "29,11,8"           # Vorschlag = Wert: keine Abweichung
        doc["meta"]["planner"]["flag:--tp-extra"] = "7"                       # Vorschlag ohne Zeile
        t = self.report(doc=doc, dry=None)["text"]
        sec = t.split("### Vorschlag und Übersteuerungen")[1].split("### Verdikte und Force")[0]
        row = next(x for x in sec.split("\n") if x.startswith("| `flag:--p-bs`"))
        self.assertEqual([c.strip() for c in row.strip("|").split("|")], ["`flag:--p-bs`", "4", "2", "6", "Nutzer"])     # aktuell / Profil / Vorschlag / Herkunft
        self.assertNotIn("`flag:--pp-stage-ratio`", sec)                      # unverändert und gleich dem Vorschlag: nicht im Diff
        self.assertNotIn("`flag:--model`", sec)
        self.assertIn("1 gegenüber dem geladenen Profil geändert, 1 als Nutzer gesetzt, 2 mit Planer-Vorschlag, 1 weichen vom Vorschlag ab", sec)
        self.assertIn("Vorschlag ohne Zeile im Profil: `flag:--tp-extra` = 7", sec)

    def test_deleted_value_is_listed(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "delete"}])
        sec = self.report(doc=doc, dry=None)["text"].split("### Vorschlag und Übersteuerungen")[1]
        self.assertIn("Gegenüber dem geladenen Profil entfernt: `flag:--p-bs`", sec)

    def test_a_profile_without_proposal_says_so(self):
        sec = self.report(dry=None)["text"].split("### Vorschlag und Übersteuerungen")[1].split("### Verdikte und Force")[0]
        self.assertIn("kein Planer-Vorschlag vor", sec)
        self.assertIn("Keine Abweichung", sec)

    def test_vectors_render_and_a_deleted_bare_flag_with_a_proposal_is_listed_twice(self):
        doc = self.edited([{"key": "flag:--p-hostgap", "op": "delete"}, {"key": "extra:P:--rank-moe-ratio", "op": "set", "value": "1,2,3"}])
        doc["meta"]["planner"]["flag:--p-hostgap"] = ""
        sec = self.report(doc=doc, dry=None)["text"].split("### Vorschlag und Übersteuerungen")[1]
        row = next(x for x in sec.split("\n") if x.startswith("| `extra:P:--rank-moe-ratio`"))
        self.assertEqual([c.strip() for c in row.strip("|").split("|")], ["`extra:P:--rank-moe-ratio`", "1,2,3", "183,137,168", "–", "Nutzer"])
        self.assertIn("Gegenüber dem geladenen Profil entfernt: `flag:--p-hostgap`", sec)
        self.assertIn("Vorschlag ohne Zeile im Profil: `flag:--p-hostgap` = (leer, Schalter an)", sec)

    def test_a_bare_flag_shows_as_on(self):
        doc = self.doc
        doc["meta"]["planner"]["flag:--p-hostgap"] = "1"                        # Vorschlag nennt einen Wert, das Profil den Schalter
        sec = self.report(doc=doc, dry=None)["text"].split("### Vorschlag und Übersteuerungen")[1]
        row = next(x for x in sec.split("\n") if x.startswith("| `flag:--p-hostgap`"))
        self.assertEqual([c.strip() for c in row.strip("|").split("|")], ["`flag:--p-hostgap`", "an", "an", "1", "Profil"])

    def test_extra_fields_from_the_oracle_become_columns_without_a_rebuild(self):
        view = {"rows": [{"key": "flag:--p-bs", "name": "--p-bs", "value": "4", "profile_value": "2", "planner_value": "6", "origin": "nutzer", "origin_label": "Nutzer",
                          "changed": True, "state": "übersteuert", "verdict": {"code": "FIT", "text": "x"}},
                         {"key": "flag:--x", "name": "--x", "value": "1", "profile_value": "1", "planner_value": None, "origin": "profil", "changed": False}],
                "removed": [], "planner_only": []}
        sec = "\n".join(P._issue_proposal(view))
        self.assertIn("| Wert | Aktuell | Profil | Vorschlag (Planer) | Herkunft | Zustand | Verdikt |", sec)
        self.assertIn("| `flag:--p-bs` | 4 | 2 | 6 | Nutzer | übersteuert | FIT |", sec)

    def test_long_table_is_cut_with_a_note(self):
        rows = [{"key": "flag:--v%d" % i, "name": "--v%d" % i, "value": "1", "profile_value": "0", "planner_value": None, "origin": "nutzer", "changed": True} for i in range(P.ISSUE_MAX_ROWS + 7)]
        sec = "\n".join(P._issue_proposal({"rows": rows, "removed": [], "planner_only": []}))
        self.assertIn("… und 7 weitere abweichende Werte (gekürzt).", sec)
        self.assertEqual(sum(1 for x in sec.split("\n") if x.startswith("| `flag:--v")), P.ISSUE_MAX_ROWS)


class Betriebsform(unittest.TestCase):
    def test_forms(self):
        f = P.issue_betriebsform
        self.assertEqual(f(["--p-bs", "--dual-share"], 3)["form"], "Dual PP/TP")
        self.assertEqual(f(["--dual-layout"], 3)["form"], "Dual PP/TP")
        self.assertEqual(f(["--d-only", "--p-bs"], 3)["form"], "nur TP")
        self.assertEqual(f(["--p-bs"], 1)["form"], "Einzelkarte")
        self.assertEqual(f(["--p-bs"], 3)["form"], "Flip PP/TP")
        self.assertEqual(f(["--p-bs"], None)["form"], "Flip PP/TP")
        self.assertEqual(f(["--d-only", "--dual-share"], 2)["form"], "Dual PP/TP")     # Dual schlägt --d-only, wie der Launcher --dual-share impliziert

    def test_a_dual_profile_reports_dual(self):
        tmp = tempfile.mkdtemp(prefix="pfi1006b_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ed = editor(tmp)
        doc = ed.edit(ed.load("release", "demo")["doc"], [{"key": "flag:--dual-share", "op": "set", "value": ""}])["doc"]
        t = ed.issue_report(doc, dry=None, cards=RIG, versions=VERSIONS)["text"]
        self.assertIn("| Betriebsform | Dual PP/TP |", t)
        self.assertIn("| Abgeleitet aus | Flag --dual-share im Profil", t)


class Redaction(Base):
    def dirty(self):
        doc = self.edited([{"key": "env:P:HF_TOKEN", "op": "set", "value": SECRET_VALUE},
                           {"key": "flag:--admin-api-key", "op": "set", "value": "adminschluessel987654"},
                           {"key": "flag:--p-bs", "op": "set", "value": "4"},
                           {"key": "env:D:CACHE_DIR", "op": "set", "value": "/root/.cache/huggingface/hub"},
                           {"key": "var:PROFILE_OWNER", "op": "set", "value": "matthias token=abcdefgh12345678"}])
        doc["meta"]["planner"]["env:P:HF_TOKEN"] = "hf_VORSCHLAG1234567"
        return doc

    def test_no_secret_and_no_host_path_in_the_text(self):
        model = json.loads(json.dumps(MODEL))
        model["weights"]["note"] = "/spinning/x"
        dry = {"ok": True, "verdict": "x /spinning/gpu-arb/holder", "notes": ["Fehler in /root/.claude/jobs/a7d09d42/tmp: Authorization: Bearer abcdef0123456789", "ok"],
               "rejections": [{"code": "HW-COUNT", "text": "HW-COUNT: Modell unter /spinning/llm_stuff/models fehlt, api_key=abcdefgh12345678"}], "cards": [{"label": "RTX 3080 20 GB"}]}
        hw = dict(HW, cards=[dict(HW["cards"][0], name="NVIDIA GeForce RTX 5090 /var/lib/flliper/hardware.json")])
        r = self.report(doc=self.dirty(), dry=dry, model=model, hardware_md=hwprofil.issue_short(hw),
                        versions=dict(VERSIONS, image="reg/x:1 /opt/rigdash/current"))
        t = r["text"]
        for bad in (SECRET_VALUE, "hf_VORSCHLAG1234567", "adminschluessel987654", "abcdefgh12345678", "abcdef0123456789", "/spinning", "/root", "/var/lib", "/opt/", "/home"):
            self.assertNotIn(bad, t, bad)
        self.assertIn("`env:P:HF_TOKEN`", t)                               # die Zeile bleibt sichtbar, nur der Wert ist weg
        row = next(x for x in t.split("\n") if x.startswith("| `env:P:HF_TOKEN`"))
        self.assertEqual(row.count("<entfernt>"), 2)                       # aktuell und Vorschlag; das Profil trug die Zeile nicht (–)
        self.assertTrue(row.rstrip().endswith("Nutzer |"))
        self.assertIn("<Pfad entfernt>", t)
        self.assertIn("| `flag:--p-bs` | 4 |", t)                           # gewöhnliche Werte bleiben lesbar

    def test_tokens_in_a_flag_name_are_not_secrets(self):
        self.assertFalse(redact.secret_name("--max-total-tokens"))
        self.assertFalse(redact.secret_name("--tokenizer"))
        self.assertFalse(redact.secret_name("--tp-prefill-max-tokens"))
        for n in ("HF_TOKEN", "--hf-token", "GITHUB_PAT", "OPENAI_API_KEY", "--admin-api-key", "DB_PASSWORD", "SECRET_KEY", "--api-key"):
            self.assertTrue(redact.secret_name(n), n)
        self.assertEqual(redact.value_for_issue("--max-total-tokens", "4096"), "4096")
        self.assertEqual(redact.value_for_issue("HF_TOKEN", "x"), "<entfernt>")

    def test_token_as_a_catalog_word_is_not_a_secret_but_a_token_credential_is(self):
        # Befund 1 (Review): ``token`` mitten im Namen oder als Token-ID/Zaehler loescht sonst genau die Werte, die der Laufbericht zeigen soll
        for n, v in (("--d-token-placement", "bandwidth"), ("--d-kv-token-cut", "owned"), ("--turn-anchor-token", "248045"),
                     ("--uneven-token-vector", "1,2,3"), ("SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION", "1"),
                     ("SGLANG_UNEVEN_TOKEN_VECTOR", "4,5"), ("--fork-anchor-token", "7"), ("SGLANG_WEG2_LANE_COVERAGE_TOKEN", "x"),
                     ("--bucket-time-to-first-token", "0.1"), ("--kt-max-deferred-experts-per-token", "2")):
            self.assertFalse(redact.secret_name(n), n)
            self.assertEqual(redact.value_for_issue(n, v), v, n)
        for n in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "--hf-token", "--token", "MY_SERVICE_TOKEN", "HF_TOKEN_FILE", "SGLANG_WEG2_BOOT_TOKEN",
                  "--auth-token", "SLACK_BOT_TOKEN"):
            self.assertTrue(redact.secret_name(n), n)
            self.assertEqual(redact.value_for_issue(n, "abc"), "<entfernt>", n)

    def test_secret_names_over_the_whole_catalog_are_exactly_the_expected_list(self):
        with open(REPO_CATALOG, encoding="utf-8") as f:
            names = sorted(json.load(f)["entries"])
        self.assertGreater(len(names), 2000)
        hits = [n for n in names if redact.secret_name(n)]
        self.assertEqual(hits, ["--admin-api-key", "--api-key", "--ssl-keyfile-password", "SGLANG_REGISTRY_ADMIN_API_KEY", "SGLANG_REGISTRY_API_KEY",
                                "SGLANG_WEG2_BOOT_TOKEN"])
        for n in names:
            if "token" in n.lower() and n not in hits:           # jeder harmlose Token-Name behaelt seinen Wert
                self.assertEqual(redact.value_for_issue(n, "7"), "7", n)

    def test_overriding_a_token_flag_shows_its_value_in_the_proposal_block(self):
        doc = self.edited([{"key": "flag:--d-token-placement", "op": "set", "value": "roundrobin"},
                           {"key": "flag:--turn-anchor-token", "op": "set", "value": "248046"}])
        t = self.report(doc=doc, dry=None)["text"]
        row = next((x for x in t.split("\n") if x.startswith("| `flag:--d-token-placement`")), "")
        self.assertIn("roundrobin", row)
        self.assertNotIn("<entfernt>", row)
        self.assertIn("248046", next((x for x in t.split("\n") if x.startswith("| `flag:--turn-anchor-token`")), ""))

    def test_version_facts_read_the_release_image_sources(self):
        # Befund 2 (Review): im Release-Image heisst der Baum /opt/htsglang/src, die Revision steht in der Image-ENV bzw. im git-Baum
        sha = "173161c595de23e0aa11bb22cc33dd44ee55ff66"
        vf = hwprofil.version_facts({}, {"tree": "/opt/htsglang/src/python"}, environ={"HTSGLANG_REVISION": sha})
        self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), (sha, "Image-ENV HTSGLANG_REVISION"))
        vf = hwprofil.version_facts({}, {"tree": "/nonexistent/python"}, environ={"HTSGLANG_REVISION_NF": "abcdef1234567", "HTSGLANG_REVISION_27B": sha,
                                                                                  "HTSGLANG_REVISION": "0000000", "STAND": "nf"})
        self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), ("abcdef1234567", "Image-ENV HTSGLANG_REVISION_NF"))
        vf = hwprofil.version_facts({}, {}, environ={"SGLANG_BUILD_COMMIT": sha})
        self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), (sha, "Image-ENV SGLANG_BUILD_COMMIT"))
        # Dockerfile-Defaults sind kein Beleg
        vf = hwprofil.version_facts({}, {}, environ={"SGLANG_BUILD_COMMIT": "unknown", "SGLANG_IMAGE_TAG": "local/sglang:dev"})
        self.assertIsNone(vf["tree_rev"])
        self.assertTrue(vf["image_default"])
        self.assertEqual(hwprofil.version_tree_text(vf), "unbelegt")
        self.assertIn("unbelegt (Default local/sglang:dev", hwprofil.version_image_text(vf))
        # kein Env-Wert, der nach nichts aussieht
        self.assertIsNone(hwprofil.version_facts({}, {}, environ={"HTSGLANG_REVISION": "not-a-sha"})["tree_rev"])

    def test_version_facts_read_git_head_of_the_tree_and_ignore_a_foreign_repo(self):
        d = tempfile.mkdtemp()
        try:
            def git(*a):
                return subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t"] + list(a), capture_output=True, text=True, check=True).stdout.strip()
            git("init", "-q")
            os.makedirs(os.path.join(d, "python"))
            with open(os.path.join(d, "python", "f.py"), "w") as f:
                f.write("x=1\n")
            git("add", "-A")
            git("commit", "-q", "-m", "c")
            head = git("rev-parse", "HEAD")
            for tree in (d, os.path.join(d, "python")):
                vf = hwprofil.version_facts({}, {"tree": tree}, environ={"HTSGLANG_REVISION": "deadbeef0"})
                self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), (head, "git HEAD des Baums"))      # gemessen schlaegt die Soll-Revision
            self.assertIn("%s (git HEAD des Baums)" % head, hwprofil.version_tree_text(vf))
            os.makedirs(os.path.join(d, "a", "b"))
            vf = hwprofil.version_facts({}, {"tree": os.path.join(d, "a", "b")}, environ={})       # Ordner in einem FREMDEN Repository: kein Beleg
            self.assertIsNone(vf["tree_rev"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_laufbericht_names_the_revision_of_a_release_image(self):
        sha = "173161c595de23e0aa11bb22cc33dd44ee55ff66"
        vf = hwprofil.version_facts({"driver": "575.57.08"}, {"tree": "/opt/htsglang/src/python"},
                                    environ={"HTSGLANG_REVISION": sha, "SGLANG_IMAGE_TAG": "local/sglang:dev"})
        t = self.report(doc=self.edited([]), versions=vf)["text"]
        self.assertIn("| Baum (Revision) | %s (Image-ENV HTSGLANG_REVISION) |" % sha, t)
        self.assertIn("| Image | unbelegt (Default local/sglang:dev", t)

    def test_every_markdown_row_stays_one_line(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "set", "value": "a|b\nc"}])
        t = self.report(doc=doc, dry=None)["text"]
        row = next(x for x in t.split("\n") if x.startswith("| `flag:--p-bs`"))
        self.assertEqual(row.count("|"), 6)                                # 5 Spalten

    def test_rejects_a_foreign_document(self):
        with self.assertRaises(P.ProfilError):
            self.ed.issue_report({"schema": "x"})


class Existing(Base):
    def test_export_is_unchanged_by_the_report(self):
        before = self.ed.export_env(self.doc)
        self.report()
        self.assertEqual(self.ed.export_env(self.doc), before)
        self.assertTrue(before["verified"])


class Routes(unittest.TestCase):
    def serve(self, hw=None):
        tmp = tempfile.mkdtemp(prefix="pfi1006r_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ns = dict(edition="rig", profil=editor(tmp), version="t")
        if hw is not None:
            ns["hwprofil"] = hw
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(SimpleNamespace(**ns)))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return srv.server_address[1]

    def call(self, port, path, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        c.request("POST", path, body=json.dumps(body), headers=dict(headers or {}, **{"Content-Type": "application/json"}))
        r = c.getresponse()
        return r.status, r.read().decode()

    def test_route_builds_the_report_from_hardware_service_and_browser_state(self):
        class Hw:
            @staticmethod
            def issue_parts():
                return {"ok": True, "short": hwprofil.issue_short(HW), "versions": VERSIONS}

        port = self.serve(Hw())
        st, txt = self.call(port, "/api/profil/load", {"kind": "release", "name": "demo"})
        doc = json.loads(txt)["doc"]
        st, txt = self.call(port, "/api/profil/dry", {"doc": doc, "cards": RIG[:2]})
        dry = json.loads(txt)
        st, txt = self.call(port, "/api/profil/issue", {"doc": doc, "dry": dry, "cards": RIG[:2], "model": MODEL})
        self.assertEqual(st, 200, txt)
        j = json.loads(txt)
        self.assertEqual(j["blocks"], list(P.ISSUE_BLOCKS))
        self.assertIn("170 (Datenbl.)", j["text"])
        self.assertIn("| `HW-COUNT` |", j["text"])
        self.assertIn("173161c595de23e0", j["text"])

    def test_route_survives_a_dead_hardware_service(self):
        class Dead:
            @staticmethod
            def issue_parts():
                raise RuntimeError("nvml weg /spinning/x")

        port = self.serve(Dead())
        doc = json.loads(self.call(port, "/api/profil/load", {"kind": "release", "name": "demo"})[1])["doc"]
        st, txt = self.call(port, "/api/profil/issue", {"doc": doc})
        self.assertEqual(st, 200, txt)
        self.assertIn("Hardwareprofil nicht verfügbar (unbelegt)", json.loads(txt)["text"])
        port2 = self.serve()                                                # gar kein Hardware-Dienst im App-Objekt
        self.assertEqual(self.call(port2, "/api/profil/issue", {"doc": doc})[0], 200)

    def test_route_is_lan_only_and_rejects_garbage(self):
        port = self.serve()
        self.assertEqual(self.call(port, "/api/profil/issue", {"doc": {}}, headers={"X-Forwarded-For": "1.2.3.4"})[0], 403)
        st, txt = self.call(port, "/api/profil/issue", {"doc": {"schema": "nein"}})
        self.assertEqual(st, 400)
        self.assertFalse(json.loads(txt)["ok"])


class HwProfilBlocks(unittest.TestCase):
    def test_short_form_is_a_subset_of_the_long_form_cells(self):
        short, long_ = hwprofil.issue_short(HW), hwprofil.issue_text(HW)
        for cell in ("170 (Datenbl.)", "32607 MiB (NVML)", "1650 GB/s (gem.)", "Gen5 x16"):
            self.assertIn(cell, short)
            self.assertIn(cell, long_)
        self.assertLess(len(short), len(long_))

    def test_version_facts_and_long_text_unchanged(self):
        os.environ.pop("SGLANG_IMAGE_TAG", None)
        f = hwprofil.version_facts(HW, {"tree": "/opt/x/releases/173161c595de23e0/python", "rigdash": "r"})
        self.assertEqual((f["tree_rev"], f["driver"], f["image"], f["rigdash"]), ("173161c595de23e0", "575.57.08", None, "r"))
        t = hwprofil.issue_text(HW, versions={"tree": "/opt/x/releases/173161c595de23e0/python"})
        self.assertIn("| Baum | 173161c595de23e0 |", t)
        self.assertIn("| Image | unbelegt (SGLANG_IMAGE_TAG nicht gesetzt) |", t)


HARNESS = r"""
const STATIC = process.argv[2];
let unhandled = [];
process.on("unhandledRejection", (e) => unhandled.push(String(e && e.message || e)));
process.on("uncaughtException", (e) => unhandled.push(String(e && e.message || e)));
const ta = { selected: false, select() { this.selected = true; } };
const root = { innerHTML: "", _h: {}, addEventListener(t, f) { this._h[t] = f; }, querySelector() { return null; }, querySelectorAll() { return []; } };
global.window = global;
global.document = { getElementById(id) { return id === "pf-root" ? root : id === "pf-pick" ? { value: "release:p" } : id === "pf-name" ? { value: "x" } : id === "pf-issue-text" ? ta : id === "tab-profil" ? { hidden: true } : null; },
  activeElement: null, body: { appendChild() {} }, createElement() { return { style: {}, setAttribute() {}, getBoundingClientRect() { return { width: 0, height: 0 }; } }; } };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.CSS = { escape: (s) => s };
let copied = null;
Object.defineProperty(global, "navigator", { value: { clipboard: { writeText: async (t) => { copied = t; } } }, configurable: true });
const calls = [];
const VIEW = { rows: [], planner_only: [], removed: [], coverage: { rows: 0, erklaert: 0, kuratiert: 0, geerntet: 0, profil_kommentar: 0, unerklaert: 0, geaendert: 0 } };
const DOC = { name: "p", id: "sha256:one", line: "nf", args: [], meta: {}, vars: [] };
global.fetch = async (url, opt) => {
  const p = String(url).replace(/^api\/profil\//, "");
  const body = opt && opt.body ? JSON.parse(opt.body) : null;
  calls.push({ p, body });
  let out;
  if (p === "list") out = { ok: true, release: [{ name: "p" }], user: [], cards: [{ id: "a", label: "A", arch: "sm86" }], rig_preset: { cards: [{ card: "a", pcie: { gen: 4, lanes: 8 } }] }, register: [] };
  else if (p === "load") out = { ok: true, doc: DOC, view: VIEW, name: "p", line: "nf", groups: [] };
  else if (p === "dry") out = { ok: true, goes: true, verdict: "ok", rejections: [], notes: [], cards: [] };
  else if (p === "issue") out = { ok: true, format: "markdown", text: "## Laufbericht <b>x</b>\n### Modellprofil", blocks: ["Modellprofil"], filename: "laufbericht-p.md" };
  else out = { ok: false, error: "unerwartet " + p };
  return { ok: out.ok, status: out.ok ? 200 : 400, text: async () => JSON.stringify(out) };
};
require(STATIC + "/profil_balken.js");
require(STATIC + "/profil.js");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const click = (dataset) => root._h.click({ target: { closest: () => ({ dataset }) } });
(async () => {
  await window.RigProfil.show(); await sleep(30);
  const out = {};
  out.buttonBeforeLoad = root.innerHTML.indexOf('data-act="issue"') >= 0;
  click({ act: "load" }); await sleep(60);
  out.button = root.innerHTML.indexOf('data-act="issue"') >= 0;
  out.noTextYet = root.innerHTML.indexOf("pf-issue-text") < 0;
  click({ act: "issue" }); await sleep(60);
  const c = calls.filter((x) => x.p === "issue");
  out.issueCalls = c.length;
  out.body = c[0] && c[0].body;
  out.shown = root.innerHTML.indexOf("Laufbericht &lt;b&gt;x&lt;/b&gt;") >= 0;          // maskiert, nie als HTML
  out.rawHtml = root.innerHTML.indexOf("<b>x</b>") >= 0;
  out.contains = root.innerHTML.indexOf("Modellprofil") >= 0;
  out.fresh = root.innerHTML.indexOf("Veraltet") < 0;
  click({ act: "issue-copy" }); await sleep(30);
  out.copied = copied;
  click({ act: "dry" }); await sleep(60);                                                // der Trockenlauf ändert die Grundlage
  out.stale = root.innerHTML.indexOf("Veraltet") >= 0;
  click({ act: "issue" }); await sleep(60);
  out.freshAgain = root.innerHTML.indexOf("Veraltet") < 0;
  click({ act: "issue-close" }); await sleep(10);
  out.closed = root.innerHTML.indexOf("pf-issue-text") < 0;
  click({ act: "issue" }); await sleep(60);
  click({ act: "load" }); await sleep(60);                                               // anderes Profil laden: der Text gehört nicht mehr dazu
  out.droppedOnLoad = root.innerHTML.indexOf("pf-issue-text") < 0;
  out.unhandled = unhandled;
  console.log(JSON.stringify(out));
})();
"""


@unittest.skipUnless(NODE, "node fehlt")
class UiJs(unittest.TestCase):
    def test_button_text_copy_and_staleness(self):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(HARNESS)
        self.addCleanup(os.unlink, fh.name)
        r = subprocess.run([NODE, fh.name, STATIC], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        o = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(o["unhandled"], [])
        self.assertFalse(o["buttonBeforeLoad"], "ohne Profil kein Laufbericht")
        self.assertTrue(o["button"])
        self.assertTrue(o["noTextYet"])
        self.assertEqual(o["issueCalls"], 1)
        self.assertEqual(set(o["body"]), {"doc", "dry", "cards", "model"})
        self.assertEqual(o["body"]["doc"]["name"], "p")
        self.assertIsNone(o["body"]["dry"])                                  # noch kein Trockenlauf
        self.assertEqual(o["body"]["cards"], [{"card": "a", "pcie": {"gen": 4, "lanes": 8}}])
        self.assertIsNone(o["body"]["model"])
        self.assertTrue(o["shown"])
        self.assertFalse(o["rawHtml"])
        self.assertTrue(o["contains"])
        self.assertTrue(o["fresh"])
        self.assertIn("### Modellprofil", o["copied"])
        self.assertTrue(o["stale"])
        self.assertTrue(o["freshAgain"])
        self.assertTrue(o["closed"])
        self.assertTrue(o["droppedOnLoad"])


class Css(unittest.TestCase):
    def test_text_area_has_a_rule(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        self.assertIn(".pf-issue-text {", html)


if __name__ == "__main__":
    unittest.main()
