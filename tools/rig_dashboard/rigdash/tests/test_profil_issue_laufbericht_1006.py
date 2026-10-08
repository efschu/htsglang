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
with open(REPO_CATALOG, encoding="utf-8") as _f:
    KNOWN = frozenset(json.load(_f)["entries"])          # die Katalognamen: nur fuer sie zeigt der Laufbericht einen Wert
HIDDEN = redact.HIDDEN_UNKNOWN
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
              "--extra-p=--rank-moe-ratio 183,137,168" --env-p "FLLIPER_MOE_SCRATCH_SLOTS=74,48,48")
profile_form_env() {
  _form FLLIPER_PDFLIP_OWNED_BASE stated
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
          {"ord": 0, "nvml_index": 1, "name": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "sm_count": _n(170, "Datasheet"), "vram_total_mib": _n(32607, unit="MiB"),
           "pcie": {"max_gen": _n(5), "max_width": _n(16)}, "mem_gbs": {"read": _n(1650, "gemessen", "GB/s"), "nominal": _n(1792, "Datasheet", "GB/s")}},
          {"ord": 1, "nvml_index": 0, "name": "NVIDIA GeForce RTX 3080", "cc": [8, 6], "sm_count": _n(68, "Datasheet"), "vram_total_mib": _n(20480, unit="MiB"),
           "pcie": {"max_gen": _n(4), "max_width": _n(16)}, "mem_gbs": {"nominal": _n(760, "Datasheet", "GB/s")}},
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
        self.assertTrue(t.startswith("## Run report (profile editor): `demo`"))
        self.assertEqual(r["filename"], "laufbericht-demo.md")

    def test_hardware_short_form_is_the_ap_a_block_and_not_the_long_form(self):
        t = self.report()["text"]
        self.assertIn("RTX 5090", t)
        self.assertIn("170 (datasheet)", t)                         # SM-Zahl mit Herkunft: dieselben Zellenbausteine wie issue_text
        self.assertIn("1650 GB/s (meas.)", t)
        self.assertIn("32607 MiB (NVML)", t)
        self.assertIn("Gen5 x16", t)
        self.assertNotIn("### Cards (NVML identity)", t)          # die Langform bleibt im Hardware-Abschnitt
        self.assertNotIn("UUID", t)

    def test_hardware_missing_says_so(self):
        t = self.report(hardware_md="")["text"]
        self.assertIn("### Hardware profile (short form)", t)
        self.assertIn("Hardware profile not available (unverified)", t)

    def test_model_block_carries_values_with_source_and_only_the_folder_name(self):
        t = self.report()["text"]
        self.assertIn("`Qwen3.8-27B`", t)
        self.assertIn("29.00 GiB (Index)", t)
        self.assertIn("dense (config)", t)
        self.assertIn("attn 16, gdn 48 (config)", t)
        self.assertIn("24 / 4 / 256 (config)", t)
        self.assertIn("1088 B (estimated)", t)
        self.assertIn("262144 (config)", t)
        self.assertNotIn("models-cache", t)

    def test_no_model_profile_names_the_model_of_the_profile(self):
        t = self.report(model=None)["text"]
        self.assertIn("No model profile estimated", t)
        self.assertIn("`Qwen3.8-27B`", t)                           # --model des Profils, nur der Ordnername
        t2 = self.report(model={"schema": "falsch"})["text"]
        self.assertIn("No model profile estimated", t2)

    def test_betriebsform_block(self):
        t = self.report()["text"]
        self.assertIn("| Operating mode | Flip PP/TP |", t)
        self.assertIn("| Derived from | neither --d-only nor --dual-* in the profile, so the launcher's standard form (read from the flags of the profile, not a choice of the planner) |", t)
        self.assertIn("| Line | nf |", t)
        self.assertIn("| Profile status | experimentell |", t)
        self.assertIn("| Cards (Dry run) | 2: ", t)
        self.assertIn("| Card count per profile | 3 |", t)
        self.assertIn("| Inventory per profile | RTX5090,RTX3080,RTX3080 |", t)
        self.assertIn("Basis release `demo`", t)

    def test_versions_block(self):
        t = self.report()["text"]
        for want in ("| Tree (revision) | 173161c595de23e0 |", "| Image | ghcr.io/efschu/htsglang:0.1.0-cu130 |", "| Driver | 575.57.08 |",
                     "| CUDA / torch (measuring process) | 13.0 / 2.9.1 |", "| Dashboard | r1006 |"):
            self.assertIn(want, t)
        self.assertRegex(t, r"\| Base profile \(sha256\) \| `[0-9a-f]{16}` \|")

    def test_versions_missing_are_unbelegt_never_guessed(self):
        os.environ.pop("FLLIPER_IMAGE_TAG", None)
        t = self.report(versions={})["text"]
        self.assertIn("| Tree (revision) | unverified |", t)
        self.assertIn("| Image | unverified (FLLIPER_IMAGE_TAG not set) |", t)
        self.assertIn("| Driver | unverified |", t)

    def test_placeholder_for_measurement_and_boot_log(self):
        t = self.report()["text"]
        tail = t.split("### Measurement result / boot log excerpt")[1]
        self.assertIn("```text", tail)
        self.assertIn("(paste the boot log excerpt here)", tail)
        self.assertIn("Result:", tail)
        self.assertIn("<!--", tail)
        self.assertEqual(t.count("```"), 2)                         # der Block ist geschlossen: ein Issue-Rendern bleibt heil


class VerdictsAndForce(Base):
    def test_no_dry_run_is_said_not_hidden(self):
        t = self.report(dry=None)["text"]
        self.assertIn("No dry run was made", t)
        self.assertIn("No dry run for this profile yet", t)       # force_hint "kein_trockenlauf"
        self.assertIn("| Cards (selected, no dry run yet) | 2: ", t)

    def test_rejections_with_register_state(self):
        dry = self.ed.dry_run(self.doc, RIG[:2])
        codes = {q["code"] for q in dry["rejections"]}
        self.assertIn("HW-COUNT", codes)
        t = self.report(dry=dry)["text"]
        self.assertIn("The planner refuses", t)
        self.assertIn("| `HW-COUNT` |", t)
        self.assertIn("FLLIPER_FORCE=1", t)                               # forcebar im Register: die Force-Zeile steht da
        row = next(x for x in t.split("\n") if x.startswith("| `HW-COUNT`"))
        self.assertIn("force:", row)

    def test_a_forged_force_claim_from_the_browser_is_not_believed(self):
        forged = {"ok": True, "verdict": "all good", "rejections": [{"code": "HW-ARCH", "text": "HW-ARCH: x", "forcebar": True, "force_state": "force"},
                                                                       {"code": "ERFUNDEN-1", "text": "?", "forcebar": True, "force_state": "force"}], "notes": []}
        t = self.report(dry=forged)["text"]
        row = next(x for x in t.split("\n") if x.startswith("| `HW-ARCH`"))
        self.assertIn("blocked", row)
        self.assertNotIn("force:", row)
        row2 = next(x for x in t.split("\n") if x.startswith("| `ERFUNDEN-1`"))
        self.assertIn("unknown code", row2)
        self.assertIn("remain even with force", t.replace("Remaining even with force", "remain even with force"))
        self.assertNotIn("Set `FLLIPER_FORCE=1` at the server start", t)  # nichts forcebar

    def test_clean_dry_run(self):
        t = self.report(dry={"ok": True, "verdict": "The planner does not refuse this profile on the chosen cards.", "rejections": [], "notes": ["note one"]})["text"]
        self.assertIn("not refuse this profile", t)
        self.assertIn("The planner refuses nothing; force is not needed.", t)
        self.assertIn("- Note: note one", t)


class Proposal(Base):
    def test_diff_comes_from_the_profile_json(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "set", "value": "4"}])
        doc["meta"]["planner"]["flag:--p-bs"] = "6"
        doc["meta"]["planner"]["flag:--pp-stage-ratio"] = "29,11,8"           # Vorschlag = Wert: keine Abweichung
        doc["meta"]["planner"]["flag:--d-token-placement"] = "7"                       # Vorschlag ohne Zeile
        t = self.report(doc=doc, dry=None)["text"]
        sec = t.split("### Proposal and overrides")[1].split("### Verdicts and force")[0]
        row = next(x for x in sec.split("\n") if x.startswith("| `flag:--p-bs`"))
        self.assertEqual([c.strip() for c in row.strip("|").split("|")], ["`flag:--p-bs`", "4", "2", "6", "User"])     # aktuell / Profil / Vorschlag / Herkunft
        self.assertNotIn("`flag:--pp-stage-ratio`", sec)                      # unverändert und gleich dem Vorschlag: nicht im Diff
        self.assertNotIn("`flag:--model`", sec)
        self.assertIn("1 changed against the loaded profile, 1 set by the user, 2 with a planner proposal, 1 differ from the proposal", sec)
        self.assertIn("Proposal without a row in the profile: `flag:--d-token-placement` = 7", sec)

    def test_deleted_value_is_listed(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "delete"}])
        sec = self.report(doc=doc, dry=None)["text"].split("### Proposal and overrides")[1]
        self.assertIn("Removed against the loaded profile: `flag:--p-bs`", sec)

    def test_a_profile_without_proposal_says_so(self):
        sec = self.report(dry=None)["text"].split("### Proposal and overrides")[1].split("### Verdicts and force")[0]
        self.assertIn("no planner proposal for this profile", sec)
        self.assertIn("No deviation", sec)

    def test_vectors_render_and_a_deleted_bare_flag_with_a_proposal_is_listed_twice(self):
        doc = self.edited([{"key": "flag:--p-hostgap", "op": "delete"}, {"key": "extra:P:--rank-moe-ratio", "op": "set", "value": "1,2,3"}])
        doc["meta"]["planner"]["flag:--p-hostgap"] = ""
        sec = self.report(doc=doc, dry=None)["text"].split("### Proposal and overrides")[1]
        row = next(x for x in sec.split("\n") if x.startswith("| `extra:P:--rank-moe-ratio`"))
        self.assertEqual([c.strip() for c in row.strip("|").split("|")], ["`extra:P:--rank-moe-ratio`", "1,2,3", "183,137,168", "–", "User"])
        self.assertIn("Removed against the loaded profile: `flag:--p-hostgap`", sec)
        self.assertIn("Proposal without a row in the profile: `flag:--p-hostgap` = (empty, switch on)", sec)

    def test_a_bare_flag_shows_as_on(self):
        doc = self.doc
        doc["meta"]["planner"]["flag:--p-hostgap"] = "1"                        # Vorschlag nennt einen Wert, das Profil den Schalter
        sec = self.report(doc=doc, dry=None)["text"].split("### Proposal and overrides")[1]
        row = next(x for x in sec.split("\n") if x.startswith("| `flag:--p-hostgap`"))
        self.assertEqual([c.strip() for c in row.strip("|").split("|")], ["`flag:--p-hostgap`", "on", "on", "1", "Profile"])

    def test_extra_fields_from_the_oracle_become_columns_without_a_rebuild(self):
        view = {"rows": [{"key": "flag:--p-bs", "name": "--p-bs", "value": "4", "profile_value": "2", "planner_value": "6", "origin": "nutzer", "origin_label": "User",
                          "changed": True, "state": "übersteuert", "verdict": {"code": "FIT", "text": "x"}},
                         {"key": "flag:--x", "name": "--x", "value": "1", "profile_value": "1", "planner_value": None, "origin": "profil", "changed": False}],
                "removed": [], "planner_only": []}
        sec = "\n".join(P._issue_proposal(view, KNOWN))
        self.assertIn("| Value | Current | Profile | Proposal (planner) | Origin | State | Verdict |", sec)
        self.assertIn("| `flag:--p-bs` | 4 | 2 | 6 | User | übersteuert | FIT |", sec)

    def test_long_table_is_cut_with_a_note(self):
        rows = [{"key": "flag:--v%d" % i, "name": "--v%d" % i, "value": "1", "profile_value": "0", "planner_value": None, "origin": "nutzer", "changed": True} for i in range(P.ISSUE_MAX_ROWS + 7)]
        sec = "\n".join(P._issue_proposal({"rows": rows, "removed": [], "planner_only": []}, KNOWN))
        self.assertIn("… and 7 more deviating values (truncated).", sec)
        self.assertEqual(sum(1 for x in sec.split("\n") if x.startswith("| `flag:--v")), P.ISSUE_MAX_ROWS)


class Betriebsform(unittest.TestCase):
    def test_forms(self):
        f = P.issue_betriebsform
        self.assertEqual(f(["--p-bs", "--dual-share"], 3)["form"], "Dual PP/TP")
        self.assertEqual(f(["--dual-layout"], 3)["form"], "Dual PP/TP")
        self.assertEqual(f(["--d-only", "--p-bs"], 3)["form"], "TP only")
        self.assertEqual(f(["--p-bs"], 1)["form"], "Single card")
        self.assertEqual(f(["--p-bs"], 3)["form"], "Flip PP/TP")
        self.assertEqual(f(["--p-bs"], None)["form"], "Flip PP/TP")
        self.assertEqual(f(["--d-only", "--dual-share"], 2)["form"], "Dual PP/TP")     # Dual schlägt --d-only, wie der Launcher --dual-share impliziert

    def test_a_dual_profile_reports_dual(self):
        tmp = tempfile.mkdtemp(prefix="pfi1006b_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ed = editor(tmp)
        doc = ed.edit(ed.load("release", "demo")["doc"], [{"key": "flag:--dual-share", "op": "set", "value": ""}])["doc"]
        t = ed.issue_report(doc, dry=None, cards=RIG, versions=VERSIONS)["text"]
        self.assertIn("| Operating mode | Dual PP/TP |", t)
        self.assertIn("| Derived from | Flag --dual-share in the profile", t)


class Redaction(Base):
    def dirty(self):
        doc = self.edited([{"key": "env:P:HF_TOKEN", "op": "set", "value": SECRET_VALUE},
                           {"key": "flag:--admin-api-key", "op": "set", "value": "adminschluessel987654"},
                           {"key": "flag:--p-bs", "op": "set", "value": "4"},
                           {"key": "env:D:FLLIPER_CACHE_DIR", "op": "set", "value": "/root/.cache/huggingface/hub"},
                           {"key": "var:PROFILE_STATUS", "op": "set", "value": "matthias token=abcdefgh12345678"}])
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
        self.assertEqual(row.count("<redacted>"), 2)                       # aktuell und Vorschlag; das Profil trug die Zeile nicht (–)
        self.assertTrue(row.rstrip().endswith("User |"))
        self.assertIn("<path redacted>", t)
        self.assertIn("| `flag:--p-bs` | 4 |", t)                           # gewöhnliche Werte bleiben lesbar

    def test_tokens_in_a_flag_name_are_not_secrets(self):
        self.assertFalse(redact.secret_name("--max-total-tokens"))
        self.assertFalse(redact.secret_name("--tokenizer"))
        self.assertFalse(redact.secret_name("--tp-prefill-max-tokens"))
        for n in ("HF_TOKEN", "--hf-token", "GITHUB_PAT", "OPENAI_API_KEY", "--admin-api-key", "DB_PASSWORD", "SECRET_KEY", "--api-key",
                  "OPENROUTER_KEY", "OPENAI_KEY", "WANDB_KEY", "ANTHROPIC_KEY", "--key"):
            self.assertTrue(redact.secret_name(n), n)
        for n in ("FLLIPER_LOG_DECODE_GRAPH_KEY", "FLLIPER_PDFLIP_TOLD_PROBE_TREE_KEY", "--ssl-keyfile", "KEYBOARD", "MONKEY"):
            self.assertFalse(redact.secret_name(n), n)
        self.assertEqual(redact.value_for_issue("--max-total-tokens", "4096", KNOWN), "4096")
        self.assertEqual(redact.value_for_issue("HF_TOKEN", "x"), "<redacted>")

    def test_user_set_vendor_key_envs_never_reach_the_proposal_block(self):
        # Befund 1 (Fix-Runde 2): ein vom Nutzer gesetzter Env mit beliebigem Praefix + KEY darf nicht im Klartext stehen
        leaks = {"OPENROUTER_KEY": "sk-or-v1-LEAKME0123456789", "WANDB_KEY": "wandbLEAK0123456789", "OPENAI_KEY": "sk-openaiLEAK0123456789",
                 "ANTHROPIC_KEY": "sk-ant-LEAK0123456789"}
        doc = self.edited([{"key": "env:P:" + n, "op": "set", "value": v} for n, v in leaks.items()])
        t = self.report(doc=doc, dry=None)["text"]
        for n, v in leaks.items():
            self.assertNotIn(v, t, n)
            row = next(x for x in t.split("\n") if x.startswith("| `env:P:%s`" % n))
            self.assertIn("<redacted>", row)

    def test_reviewer_leaks_appear_as_entfernt_in_the_report(self):
        # Befund 1 (Fix-Runde 3): Pluralnamen, HF_AUTH und eine URL mit user:pass duerfen nicht im Klartext stehen
        leaks = {"OPENAI_API_KEYS": "sk-proj-LEAKA0123456789", "HF_AUTH": "hf_LEAKB0123456789abcdef", "MY_STUFF": "ghp_LEAKC0123456789abcdef0123",
                 "MODEL_MIRROR": "https://alice:hunter2pw@mirror.example.org/models"}
        doc = self.edited([{"key": "env:P:" + n, "op": "set", "value": v} for n, v in leaks.items()])
        dry = {"ok": False, "verdict": "x", "notes": ["pull von postgres://dbuser:s3cretpw@db.internal:5432/x"], "rejections": [], "cards": []}
        t = self.report(doc=doc, dry=dry)["text"]
        for n, v in leaks.items():
            for frag in (v, v.split(":")[-1] if "@" in v else v):
                self.assertNotIn(frag.split("@")[0] if "@" in frag else frag, t, n)
            row = next(x for x in t.split("\n") if x.startswith("| `env:P:%s`" % n))
            self.assertTrue("<redacted>" in row or HIDDEN in row, n)
        self.assertNotIn("hunter2pw", t)
        self.assertNotIn("s3cretpw", t)
        self.assertIn("dbuser:<redacted>@db.internal", t)                # Nutzer bleibt lesbar, das Passwort ist weg

    def test_value_shapes_are_cut_whatever_the_name(self):
        for v in ("sk-abcdefgh12345678", "sk-ant-api03-AbCdEf0123456789xyz", "sk-proj-AbCd0123456789EfGh", "hf_AbCdEfGhIjKlMnOp",
                  "ghp_AbCdEfGhIjKlMnOp0123456789", "gho_AbCdEfGhIjKlMnOp0123456789", "ghu_AbCdEfGhIjKlMnOp0123456789", "ghs_AbCdEfGhIjKlMnOp0123456789",
                  "github_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwx", "xoxb-1234567890-abcdefghij", "xoxp-1234567890-abcdefghij", "xoxa-1234567890-abcdef",
                  "xoxr-1234567890-abcdef", "AKIAIOSFODNN7EXAMPLE"):
            for form in (v, "key=" + v, "| `env:P:FOO_BAR` | %s | x |" % v, "export FOO=%s" % v):
                out = redact.text_for_issue(form)
                self.assertNotIn(v, out, form)
                if "github_pat" not in v:                                  # ``GITHUB_PAT`` laesst clean() die ganze Zeile fallen: dann ist sie weg
                    self.assertIn("<redacted>", out, form)
        self.assertEqual(redact.text_for_issue("Authorization-frei: Bearer abcdef0123456789xyz"), "Authorization-frei: Bearer <redacted>")
        self.assertEqual(redact.text_for_issue("curl https://alice:hunter2@host.example/x"), "curl https://alice:<redacted>@host.example/x")
        self.assertNotIn("hunter2", redact.text_for_issue("redis://:hunter2@cache:6379/0"))
        long_tok = "Zk3" + "aB9xQ" * 8                                       # 43 Zeichen, gemischt
        for form in ("FOO=" + long_tok, "foo: " + long_tok, "--bar=" + long_tok, "| FOO=" + long_tok + " |"):
            out = redact.text_for_issue(form)
            self.assertNotIn(long_tok, out, form)
            self.assertIn("<redacted>", out, form)

    # ---- Fix-Runde 4: jeder Ausgabepfad des Laufberichts mit derselben Sonde (Praefix-loses Token, JWT, Hostpfade) ----
    TOK = "Zq8vN3kLp0Wm7Rt2Yx5Bc9Df4Gh6Jk1Ls"                           # 33 Zeichen, gemischt, kein Anbieter-Praefix
    TOK_LOW = "a1b2c3d4e5f60718293a4b5c6d7e8f90"                       # 32 Zeichen klein, hex-artig (kein SHA: 32 != 40/64)
    JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    PATHS = ("/nvme/hf", "/workspace/models/Qwen", "~/cache/hf", "/scratch/run1", "/models/Qwen3-27B", "$HOME/x/y")
    # Teilstrings, die NIE im Text stehen duerfen (die Ordner davor); die letzte Pfadkomponente darf bleiben
    PATH_LEAKS = ("/nvme", "/workspace", "~/", "/scratch", "/models/", "$HOME", "cache/hf", "models/Qwen", "x/y")

    def assert_clean(self, t, where):
        for bad in (self.TOK, self.TOK_LOW, self.JWT, self.JWT.split(".")[1], self.JWT.split(".")[2]) + self.PATH_LEAKS:
            self.assertNotIn(bad, t, "%s: %s" % (where, bad))

    def test_probe_in_env_value_flag_value_and_proposal(self):
        # Befund 1/2/3: Wert OHNE ``=`` in einer Tabellenzelle, harmloser Name; Env-Wert, Flag-Wert, Vorschlag, Profilwert
        edits = [{"key": "env:P:MY_THING", "op": "set", "value": self.TOK}, {"key": "env:D:OTHER_THING", "op": "set", "value": self.TOK_LOW},
                 {"key": "env:P:JWT_THING", "op": "set", "value": self.JWT},
                 {"key": "env:P:FLLIPER_CACHE_DIR", "op": "set", "value": "/nvme/hf"}, {"key": "env:D:FLLIPER_DG_CACHE_DIR", "op": "set", "value": "~/cache/hf"},
                 {"key": "env:D:FLLIPER_DEBUG_HOLD_DIR", "op": "set", "value": "$HOME/x/y"},
                 {"key": "flag:--model", "op": "set", "value": "/workspace/models/Qwen"},
                 {"key": "flag:--download-dir", "op": "set", "value": "/scratch/run1"},
                 {"key": "flag:--extra-thing", "op": "set", "value": self.TOK}]
        doc = self.edited([e for e in edits if not e["key"].startswith("flag:--extra")])
        doc["meta"]["planner"]["env:P:MY_THING"] = self.TOK                # auch der Vorschlag-Wert
        t = self.report(doc=doc, dry=None)["text"]
        self.assert_clean(t, "Tabellen")
        row = next(x for x in t.split("\n") if x.startswith("| `env:P:MY_THING`"))
        self.assertEqual(row.count(HIDDEN), 2, row)                         # unbekannter Schluessel: Wert ausgeblendet (aktuell und Vorschlag)
        self.assertIn("<hostpfad>/hf", next(x for x in t.split("\n") if x.startswith("| `env:P:FLLIPER_CACHE_DIR`")))
        self.assertIn("<hostpfad>/Qwen", next(x for x in t.split("\n") if "--model" in x and x.startswith("| `flag:")))
        # die Zeile bleibt, nur das Geheimnis ist weg
        self.assertIn("`env:D:OTHER_THING`", t)
        # der Wert ohne Zeilenkontext (so reicht ihn _issue_cell weiter)
        for tok in (self.TOK, "`%s`" % self.TOK, self.TOK_LOW, self.JWT):
            self.assertEqual(redact.value_for_issue("MY_THING", tok, KNOWN), HIDDEN)                       # unbekannter Schluessel
            self.assertEqual(redact.value_for_issue("FLLIPER_CACHE_DIR", tok, KNOWN), "<redacted>")        # Katalog-Schluessel: die Wertform schneidet
            self.assertEqual(redact.value_for_issue("MY_THING", tok), HIDDEN)                              # ohne Katalog: geschlossen

    def test_probe_in_free_text_notes_verdict_rejection_hardware_model_and_var(self):
        dry = {"ok": False, "verdict": "Planer lehnt ab: %s und %s" % (self.TOK, self.PATHS[0]),
               "notes": ["Token %s im Log" % self.TOK_LOW, "JWT %s" % self.JWT, "Cache bei %s und %s" % (self.PATHS[2], self.PATHS[3]),
                         "Modell %s" % self.PATHS[1], "Home %s" % self.PATHS[5]],
               "rejections": [{"code": "HW-COUNT", "text": "HW-COUNT: Pfad %s, Token %s, Home %s" % (self.PATHS[4], self.TOK, self.PATHS[2])}],
               "cards": [{"label": "RTX 3080 20 GB"}]}
        hw = dict(HW, cards=[dict(HW["cards"][0], name="NVIDIA RTX 5090 %s %s %s %s" % (self.TOK, self.JWT, self.PATHS[0], self.PATHS[2]))])
        model = json.loads(json.dumps(MODEL))
        model["weights"]["note"] = "%s %s" % (self.TOK, self.PATHS[1])
        model["path"] = "/nvme/models/Qwen3.8-27B"
        doc = self.edited([{"key": "var:PROFILE_STATUS", "op": "set", "value": "x %s %s %s" % (self.TOK, self.JWT, self.PATHS[0])}])
        vers = dict(VERSIONS, image="reg/x:1 %s /nvme/img" % self.TOK, driver="575 ~/drv")
        t = self.report(doc=doc, dry=dry, hardware_md=hwprofil.issue_short(hw), model=model, versions=vers)["text"]
        self.assert_clean(t, "Freitext")
        self.assertIn("Qwen3.8-27B", t)                                   # nur der Ordnername des Modells, wie bisher

    def test_probe_in_the_force_and_docker_run_blocks(self):
        dry = {"ok": False, "verdict": "x", "notes": [], "cards": [],
               "rejections": [{"code": "PROFIL-STATUS", "text": "PROFIL-STATUS: %s %s" % (self.TOK, "/nvme/hf")}]}
        t = self.report(dry=dry)["text"]
        self.assert_clean(t, "Force")
        use = self.ed.use_hint("demo", dry=dry)
        self.assert_clean("\n".join(use["docker_run"]) + use["text"], "docker run")

    def test_paths_rules(self):
        for src, want in (("/nvme/hf", "<hostpfad>/hf"), ("x /workspace/models/Qwen y", "x <hostpfad>/Qwen y"), ("~/a/b", "<hostpfad>/b"),
                          ("$HOME/a", "<hostpfad>/a"), ("--model=/nvme/m", "--model=<hostpfad>/m"), ("/scratch", "<hostpfad>/scratch"),
                          ("/root/x", "<path redacted>"), ("/app/python", "/app/python"), ("POST /api/profil/issue", "POST /api/profil/issue")):
            self.assertEqual(redact.paths(src), want, src)
        for keep in ("https://host.example/a/b", "ja/nein", "GB/s", "RTX 3080 / 5090", "a / b", "1/2", "~", "25 %"):
            self.assertEqual(redact.paths(keep), keep, keep)
        once = redact.text_for_issue("p=/nvme/hf ~/x")
        self.assertEqual(redact.text_for_issue(once), once)               # zweimal gelaufen (Hardware-Block, dann Gesamttext) aendert nichts

    def test_jwt_and_bare_runs(self):
        for form in (self.JWT, "Bearer-frei " + self.JWT, "x=" + self.JWT, "| a | %s |" % self.JWT, "token " + self.TOK, "| a | %s |" % self.TOK_LOW):
            out = redact.text_for_issue(form)
            self.assertNotIn(self.JWT, out, form)
            self.assertNotIn(self.TOK, out, form)
            self.assertNotIn(self.TOK_LOW, out, form)
            self.assertIn("<redacted>", out, form)
        for keep in ("task-runner-big-name-0123456789-abcdef", "173161c595de23e0aa11bb22cc33dd44ee55ff66", "sha256:" + "ab12" * 16,
                     "FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN_X_Y_Z_0123456789", "eyJ", "eyJ.a.b", "Qwen3.6-27B-AWQ-BF16-INT4-some-very-long-variant-name-v2"):
            self.assertEqual(redact.text_for_issue(keep), keep, keep)

    def test_launcher_class_names_stay_readable_but_real_base64_does_not(self):
        """Nacharbeit 1006 Runde 6, Befund 2: ``PdFlipTpOperatingPointInfeasible`` (30 Zeichen, Gross/Klein/Ziffer) war als Base64 geschwaerzt."""
        for text in ("W64 PdFlipTpOperatingPointInfeasible: position 3 derives weights [20, 12, 8]",
                     "W71 PdFlipXchgResidencyUnarmable: the exchange's predicted VRAM residency does not fit",
                     "W64 PdFlipTpOperatingPointInfeasible: x | W71 PdFlipXchgResidencyUnarmable: y",
                     "PdFlipXchgSemaphoreNotRearmed PdFlipPeerLegAborted PdFlipDualCompactBreach"):
            self.assertEqual(redact.text_for_issue(text), text, text)
        # ein echtes Geheimnis neben dem Klassennamen wird weiter geschnitten
        out = redact.text_for_issue("W64 PdFlipTpOperatingPointInfeasible key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY and Ab3dE9xQ2mZpL0vK7sT4wY8nR1cF6hJ5gU")
        self.assertIn("PdFlipTpOperatingPointInfeasible", out)
        self.assertNotIn("wJalrXUtnFEMI", out)
        self.assertNotIn("Ab3dE9xQ2mZpL0vK7sT4wY8nR1cF6hJ5gU", out)

    def test_a_token_shaped_camelcase_run_is_cut_only_source_names_stay(self):
        """Nacharbeit 1006 Runde 7, Befund 3: die Ausnahme ist eine LISTE (Bezeichner aus dem Launcher-/pdflip-Quelltext), keine Form."""
        for probe in ("AbcdEfghIjklMnopQrstUvwxYz12Ab", "Abcd1Efgh2Ijkl3Mnop4Qrst5Uvwx6Yzab", "PdFlipAbcdEfghIjklMnopQrstUvwxYz12"):
            out = redact.text_for_issue("note " + probe)
            self.assertNotIn(probe, out, probe)
            self.assertIn("<redacted>", out, probe)
        self.assertEqual(redact.text_for_issue("W64 PdFlipTpOperatingPointInfeasible: x"), "W64 PdFlipTpOperatingPointInfeasible: x")
        # die Liste kommt aus dem Quelltext: ein Name, der dort als Klasse steht, bleibt lesbar (Beleg: class PdFlipDKvStageWavesRefused)
        self.assertIn("PdFlipDKvStageWavesRefused", redact.known_idents())
        self.assertIn("PdFlipTpOperatingPointInfeasible", redact.known_idents())
        self.assertNotIn("AbcdEfghIjklMnopQrstUvwxYz12Ab", redact.known_idents())

    def test_without_a_source_tree_only_the_builtin_names_stay(self):
        saved = (redact._known_cache, os.environ.get("HWPROFIL_TREE"), os.environ.get("KARTENPLAN_TREE"), redact._tree_candidates)
        try:
            redact._known_cache = None
            os.environ.pop("HWPROFIL_TREE", None)
            os.environ.pop("KARTENPLAN_TREE", None)
            redact._tree_candidates = lambda: []
            self.assertEqual(redact.known_idents(), redact._KNOWN_IDENT_BUILTIN)
            self.assertEqual(redact.text_for_issue("W64 PdFlipTpOperatingPointInfeasible: x"), "W64 PdFlipTpOperatingPointInfeasible: x")
            self.assertNotIn("AbcdEfghIjklMnopQrstUvwxYz12Ab", redact.text_for_issue("note AbcdEfghIjklMnopQrstUvwxYz12Ab"))
            self.assertIsNone(redact._known_cache)            # ohne Baum nichts gemerkt: der naechste Aufruf sucht neu
        finally:
            redact._known_cache, redact._tree_candidates = saved[0], saved[3]
            if saved[1] is not None:
                os.environ["HWPROFIL_TREE"] = saved[1]
            if saved[2] is not None:
                os.environ["KARTENPLAN_TREE"] = saved[2]

    def test_names_ending_in_a_credential_word_are_secrets_singular_and_plural(self):
        for n in ("OPENAI_API_KEYS", "MY_KEYS", "HF_AUTH", "--auth", "DB_PASS", "--pass", "MY_SECRETS", "DB_PASSWORDS", "SERVICE_CREDENTIALS", "GH_PAT",
                  "HF_TOKENS", "--auth-tokens", "--credential"):
            self.assertTrue(redact.secret_name(n), n)
            self.assertEqual(redact.value_for_issue(n, "x"), "<redacted>", n)
        for n in ("--max-total-tokens", "--auth-backend", "--pass-through", "FLLIPER_LOG_DECODE_GRAPH_KEY", "FLLIPER_X_TREE_KEYS",
                  "--bypass", "PATH", "KEYS_PER_SEC_X", "--tokens-per-second", "FLLIPER_HICACHE_BIGRAM_KEYS", "FLLIPER_PDFLIP_MAMBA_STATE_KEYS",
                  "FLLIPER_PDFLIP_D_TWIN_PASS", "--kv-session-offload-budget-session-tokens"):
            self.assertFalse(redact.secret_name(n), n)

    def test_values_that_are_no_secrets_survive_the_shape_layer(self):
        sha = "173161c595de23e0aa11bb22cc33dd44ee55ff66"
        for v in (sha, "sha256:" + "ab12" * 16, "tree_sha=" + sha, "image=flliper:0.1.0-cu130", "disk-cache-size=10", "task-runner-big-name-0123456789-abcdef",
                  "FLLIPER_LOG_DECODE_GRAPH_KEY=1", "model=Qwen3.6-27B-AWQ-BF16-INT4-some-very-long-variant-name-v2", "tag: FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN_X_Y_Z_0123456789",
                  "hf_hub_cache=1", "sk-learn is a library", "https://example.org/path:8080/x", "user@host", "ssh://git@host/repo.git"):
            self.assertEqual(redact.text_for_issue(v), v, v)

    def test_token_as_a_catalog_word_is_not_a_secret_but_a_token_credential_is(self):
        # Befund 1 (Review): ``token`` mitten im Namen oder als Token-ID/Zaehler loescht sonst genau die Werte, die der Laufbericht zeigen soll
        for n, v in (("--d-token-placement", "bandwidth"), ("--d-kv-token-cut", "owned"), ("--turn-anchor-token", "248045"),
                     ("--uneven-token-vector", "1,2,3"), ("FLLIPER_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION", "1"),
                     ("FLLIPER_UNEVEN_TOKEN_VECTOR", "4,5"), ("--fork-anchor-token", "7"), ("FLLIPER_PDFLIP_LANE_COVERAGE_TOKEN", "x"),
                     ("--bucket-time-to-first-token", "0.1"), ("--kt-max-deferred-experts-per-token", "2")):
            self.assertFalse(redact.secret_name(n), n)
            self.assertEqual(redact.value_for_issue(n, v, KNOWN), v, n)
        for n in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "--hf-token", "--token", "MY_SERVICE_TOKEN", "HF_TOKEN_FILE", "FLLIPER_PDFLIP_BOOT_TOKEN",
                  "--auth-token", "SLACK_BOT_TOKEN"):
            self.assertTrue(redact.secret_name(n), n)
            self.assertEqual(redact.value_for_issue(n, "abc"), "<redacted>", n)

    def test_secret_names_over_the_whole_catalog_are_exactly_the_expected_list(self):
        with open(REPO_CATALOG, encoding="utf-8") as f:
            names = sorted(json.load(f)["entries"])
        self.assertGreater(len(names), 2000)
        hits = [n for n in names if redact.secret_name(n)]
        self.assertEqual(hits, ["--admin-api-key", "--api-key", "--ssl-keyfile-password", "FLLIPER_REGISTRY_ADMIN_API_KEY", "FLLIPER_REGISTRY_API_KEY",
                                "FLLIPER_PDFLIP_BOOT_TOKEN"])
        for n in names:
            if "token" in n.lower() and n not in hits:           # jeder harmlose Token-Name behaelt seinen Wert
                self.assertEqual(redact.value_for_issue(n, "7", KNOWN), "7", n)

    def test_overriding_a_token_flag_shows_its_value_in_the_proposal_block(self):
        doc = self.edited([{"key": "flag:--d-token-placement", "op": "set", "value": "roundrobin"},
                           {"key": "flag:--turn-anchor-token", "op": "set", "value": "248046"}])
        t = self.report(doc=doc, dry=None)["text"]
        row = next((x for x in t.split("\n") if x.startswith("| `flag:--d-token-placement`")), "")
        self.assertIn("roundrobin", row)
        self.assertNotIn("<redacted>", row)
        self.assertIn("248046", next((x for x in t.split("\n") if x.startswith("| `flag:--turn-anchor-token`")), ""))

    def test_version_facts_read_the_release_image_sources(self):
        # Befund 2 (Review): im Release-Image heisst der Baum /opt/htsglang/src, die Revision steht in der Image-ENV bzw. im git-Baum
        sha = "173161c595de23e0aa11bb22cc33dd44ee55ff66"
        vf = hwprofil.version_facts({}, {"tree": "/opt/htsglang/src/python"}, environ={"HTSGLANG_REVISION": sha})
        self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), (sha, "Image-ENV HTSGLANG_REVISION"))
        vf = hwprofil.version_facts({}, {"tree": "/nonexistent/python"}, environ={"HTSGLANG_REVISION_NF": "abcdef1234567", "HTSGLANG_REVISION_27B": sha,
                                                                                  "HTSGLANG_REVISION": "0000000", "STAND": "nf"})
        self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), ("abcdef1234567", "Image-ENV HTSGLANG_REVISION_NF"))
        vf = hwprofil.version_facts({}, {}, environ={"FLLIPER_BUILD_COMMIT": sha})
        self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), (sha, "Image-ENV FLLIPER_BUILD_COMMIT"))
        # Dockerfile-Defaults sind kein Beleg
        vf = hwprofil.version_facts({}, {}, environ={"FLLIPER_BUILD_COMMIT": "unknown", "FLLIPER_IMAGE_TAG": "local/flliper:dev"})
        self.assertIsNone(vf["tree_rev"])
        self.assertTrue(vf["image_default"])
        self.assertEqual(hwprofil.version_tree_text(vf), "unverified")
        self.assertIn("unverified (default local/flliper:dev", hwprofil.version_image_text(vf))
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
                self.assertEqual((vf["tree_rev"], vf["tree_rev_src"]), (head, "git HEAD of the tree"))      # gemessen schlaegt die Soll-Revision
            self.assertIn("%s (git HEAD of the tree)" % head, hwprofil.version_tree_text(vf))
            os.makedirs(os.path.join(d, "a", "b"))
            vf = hwprofil.version_facts({}, {"tree": os.path.join(d, "a", "b")}, environ={})       # Ordner in einem FREMDEN Repository: kein Beleg
            self.assertIsNone(vf["tree_rev"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_laufbericht_names_the_revision_of_a_release_image(self):
        sha = "173161c595de23e0aa11bb22cc33dd44ee55ff66"
        vf = hwprofil.version_facts({"driver": "575.57.08"}, {"tree": "/opt/htsglang/src/python"},
                                    environ={"HTSGLANG_REVISION": sha, "FLLIPER_IMAGE_TAG": "local/flliper:dev"})
        t = self.report(doc=self.edited([]), versions=vf)["text"]
        self.assertIn("| Tree (revision) | %s (Image-ENV HTSGLANG_REVISION) |" % sha, t)
        self.assertIn("| Image | unverified (default local/flliper:dev", t)

    def test_every_markdown_row_stays_one_line(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "set", "value": "a|b\nc"}])
        t = self.report(doc=doc, dry=None)["text"]
        row = next(x for x in t.split("\n") if x.startswith("| `flag:--p-bs`"))
        self.assertEqual(row.count("|"), 6)                                # 5 Spalten

    def test_rejects_a_foreign_document(self):
        with self.assertRaises(P.ProfileError):
            self.ed.issue_report({"schema": "x"})



class StructuralAllowRule(Base):
    """Fix-Runde 5: Werte nur fuer Catalog-Schluessel (und nicht per Name Geheimnis); jeder andere Schluessel des Nutzers zeigt nur seinen Namen.
    Die Wertmuster bleiben die zweite Schicht (auch fuer Catalog-Schluessel); Pfade werden normalisiert."""
    AWS = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    AZURE = "PV2c7Y2ccEFykpwliZwJBl1tQ57j/XHICdb03E6gK109O3L0aonSzrcIKLpVrfCeYUAywgtSDET5eXC4+gYlIQ=="
    DISCORD = ".".join(("MTk4NjIyNDgz" + "NDcxOTI1MjQ4", "Cl2" + "FMQ", "ZnCjm1XVW7vRze" + "4b7Cq4se7kKWs"))      # zusammengesetzt: kein Wortlaut im Quelltext (Push-Schutz)
    MIXED24 = "Zq8vN3kLp0Wm7Rt2Yx5Bc9Df"                                  # 24 Zeichen, gemischt, kein Anbieter-Praefix
    PROBES = {"MY_AWS": AWS, "MY_AZURE": AZURE, "MY_DISCORD": DISCORD, "MY_THING": MIXED24}

    def row(self, t, key):
        return next(x for x in t.split("\n") if x.startswith("| `%s`" % key))

    def test_secrets_under_neutral_names_are_hidden_end_to_end(self):
        doc = self.edited([{"key": "env:P:" + n, "op": "set", "value": v} for n, v in self.PROBES.items()])
        for n, v in self.PROBES.items():
            doc["meta"]["planner"]["env:P:" + n] = v                       # auch der Vorschlag-Wert
        t = self.report(doc=doc, dry=None)["text"]
        for n, v in self.PROBES.items():
            for frag in (v, v[:20], v[-12:]):
                self.assertNotIn(frag, t, n)
            row = self.row(t, "env:P:" + n)
            self.assertEqual(row.count(HIDDEN), 2, row)                    # aktuell und Vorschlag; der Schluesselname bleibt sichtbar
        self.assertIn("`env:P:MY_THING`", t)

    def test_the_same_secrets_in_free_text_are_cut_by_the_value_shapes(self):
        dry = {"ok": False, "verdict": "Planer: %s" % self.AWS, "notes": ["Azure %s" % self.AZURE, "Discord: %s" % self.DISCORD, "x=%s" % self.AWS],
               "rejections": [{"code": "HW-COUNT", "text": "HW-COUNT: key %s und %s" % (self.AZURE, self.DISCORD)}], "cards": []}
        t = self.report(dry=dry)["text"]
        for v in (self.AWS, self.AZURE, self.DISCORD):
            for frag in (v, v[:20], v[-12:]):
                self.assertNotIn(frag, t, frag)
        self.assertIn("<redacted>", t)
        for form in (self.AWS, self.AZURE, "| a | %s |" % self.AZURE, "k=" + self.AWS, "\"%s\"" % self.AWS):
            out = redact.text_for_issue(form)
            self.assertNotIn(self.AWS, out, form)
            self.assertNotIn(self.AZURE, out, form)
            self.assertIn("<redacted>", out, form)

    def test_catalog_keys_with_harmless_values_stay_visible(self):
        doc = self.edited([{"key": "flag:--p-bs", "op": "set", "value": "4"},
                           {"key": "env:P:FLLIPER_CACHE_DIR", "op": "set", "value": "/models-cache/Qwen3.8-27B/cache"},
                           {"key": "flag:--d-token-placement", "op": "set", "value": "roundrobin"}])
        t = self.report(doc=doc, dry=None)["text"]
        self.assertIn("| `flag:--p-bs` | 4 |", t)
        self.assertIn("roundrobin", self.row(t, "flag:--d-token-placement"))
        row = self.row(t, "env:P:FLLIPER_CACHE_DIR")
        self.assertIn("/models-cache/Qwen3.8-27B/cache", row)               # ein Pfad unter dem Mount des Containers ist kein Hostpfad
        self.assertNotIn(HIDDEN, row)
        self.assertNotIn("<hostpfad>", row)

    def test_a_catalog_key_with_a_secret_value_is_cut_by_the_second_layer(self):
        doc = self.edited([{"key": "env:P:FLLIPER_CACHE_DIR", "op": "set", "value": self.AWS}, {"key": "env:D:FLLIPER_DG_CACHE_DIR", "op": "set", "value": self.DISCORD}])
        t = self.report(doc=doc, dry=None)["text"]
        for v in (self.AWS, self.DISCORD):
            self.assertNotIn(v[:20], t)
        self.assertIn("<redacted>", self.row(t, "env:P:FLLIPER_CACHE_DIR"))
        self.assertIn("<redacted>", self.row(t, "env:D:FLLIPER_DG_CACHE_DIR"))

    def test_a_secret_name_in_the_catalog_stays_entfernt_and_unknown_is_not_entfernt(self):
        doc = self.edited([{"key": "flag:--api-key", "op": "set", "value": "klartext"}, {"key": "env:P:SOME_FREE_NAME", "op": "set", "value": "1"}])
        t = self.report(doc=doc, dry=None)["text"]
        self.assertNotIn("klartext", t)
        self.assertIn("<redacted>", self.row(t, "flag:--api-key"))
        self.assertIn(HIDDEN, self.row(t, "env:P:SOME_FREE_NAME"))          # auch ein harmloser Wert: der Schluessel ist unbekannt

    def test_the_rule_in_redact_directly(self):
        self.assertEqual(redact.value_for_issue("--p-bs", "4", KNOWN), "4")
        self.assertEqual(redact.value_for_issue("flag:--p-bs", "4", KNOWN), "4")                   # Profilschluessel mit Praefix
        self.assertEqual(redact.value_for_issue("env:P:FLLIPER_CACHE_DIR", "/app/x", KNOWN), "/app/x")
        self.assertEqual(redact.value_for_issue("env:P:MY_THING", "4", KNOWN), HIDDEN)
        self.assertEqual(redact.value_for_issue("extra:P:--nicht-im-catalog", "4", KNOWN), HIDDEN)
        self.assertEqual(redact.value_for_issue("--p-bs", "4"), HIDDEN)                            # ohne Katalog nichts zeigen
        self.assertEqual(redact.value_for_issue("--p-bs", "4", frozenset()), HIDDEN)
        self.assertEqual(redact.value_for_issue("MY_THING", "", KNOWN), "")                        # leer ist kein Geheimnis
        self.assertEqual(redact.value_for_issue("HF_TOKEN", "x", KNOWN), "<redacted>")
        self.assertEqual(redact.bare_key("env:D:HF_HOME"), "HF_HOME")
        self.assertEqual(redact.bare_key("var:PROFILE_NAME"), "PROFILE_NAME")

    def test_new_shapes_spare_names_hashes_and_hosts(self):
        for keep in ("registry.example-company-internal.com", "Qwen3.6-27B-AWQ-BF16-INT4.gguf", "model-00001-of-00004.safetensors", "ghcr.io/efschu/htsglang:0.1.0-cu130",
                     "173161c595de23e0aa11bb22cc33dd44ee55ff66", "/models-cache/Qwen3Coder30BA3BInstructX1/abcDEF12345", "ja/nein", "GB/s", "RTX 3080 / 5090",
                     "python/flliper/srt/pdflip/profile_json.py", "https://host.example/a/b/c", "1.2.3.4"):
            self.assertEqual(redact.text_for_issue(keep), keep, keep)
        for secret in (self.DISCORD, "x." + self.MIXED24 + ".Cl2" + "FMQ" + "y" * 16, "Ab1+" * 10 + "==", self.AWS):
            self.assertEqual(redact.text_for_issue(secret), "<redacted>", secret)

    def test_paths_are_normalised_before_they_are_judged(self):
        for src, want in (("file:///nvme/private/model", "<hostpfad>/model"), ("--model=file:///nvme/private/model", "--model=<hostpfad>/model"),
                          ("file://host/data/x", "<path redacted>"), ("FILE:///scratch/a", "<hostpfad>/a"),
                          ("/app/../../root/.ssh/id_rsa", "<path redacted>"), ("/app/../nvme/x", "<hostpfad>/x"), ("/app/../app/x", "/app/x"),
                          ("/models-cache/../../etc/shadow", "<path redacted>"), ("/app/x/./y", "/app/x/y"), ("~/../../etc/x", "<hostpfad>/x"),
                          ("\"/nvme/my models/Qwen\"", "\"<hostpfad>/Qwen\""), ("'/home/al ice/My Documents/k'", "'<path redacted>'"),
                          ("`/app/with space/x`", "`/app/with space/x`"), ("/app/x and /nvme/y", "/app/x and <hostpfad>/y")):
            self.assertEqual(redact.paths(src), want, src)
            self.assertEqual(redact.paths(redact.paths(src)), want, "zweimal: " + src)

    def test_path_probes_end_to_end(self):
        edits = {"--model": "file:///nvme/private/model", "--download-dir": "/app/../../root/.ssh/id_rsa", "--served-model-name": "\"/nvme/my models/Qwen\""}
        doc = self.edited([{"key": "flag:" + k, "op": "set", "value": v} for k, v in edits.items()])
        t = self.report(doc=doc, dry=None)["text"]
        for bad in ("nvme", "private", "/root", ".ssh", "id_rsa", "my models", "file://"):
            self.assertNotIn(bad, t, bad)
        self.assertIn("<hostpfad>/model", self.row(t, "flag:--model"))
        self.assertIn("<path redacted>", self.row(t, "flag:--download-dir"))


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
        self.assertIn("170 (datasheet)", j["text"])
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
        self.assertIn("Hardware profile not available (unverified)", json.loads(txt)["text"])
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
        for cell in ("170 (datasheet)", "32607 MiB (NVML)", "1650 GB/s (meas.)", "Gen5 x16"):
            self.assertIn(cell, short)
            self.assertIn(cell, long_)
        self.assertLess(len(short), len(long_))

    def test_version_facts_and_long_text_unchanged(self):
        os.environ.pop("FLLIPER_IMAGE_TAG", None)
        f = hwprofil.version_facts(HW, {"tree": "/opt/x/releases/173161c595de23e0/python", "rigdash": "r"})
        self.assertEqual((f["tree_rev"], f["driver"], f["image"], f["rigdash"]), ("173161c595de23e0", "575.57.08", None, "r"))
        t = hwprofil.issue_text(HW, versions={"tree": "/opt/x/releases/173161c595de23e0/python"})
        self.assertIn("| Tree | 173161c595de23e0 |", t)
        self.assertIn("| Image | unverified (FLLIPER_IMAGE_TAG not set) |", t)


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
const VIEW = { rows: [], planner_only: [], removed: [], coverage: { rows: 0, explained: 0, curated: 0, harvested: 0, profil_kommentar: 0, unexplained: 0, changed: 0 } };
const DOC = { name: "p", id: "sha256:one", line: "nf", args: [], meta: {}, vars: [] };
global.fetch = async (url, opt) => {
  const p = String(url).replace(/^api\/profil\//, "");
  const body = opt && opt.body ? JSON.parse(opt.body) : null;
  calls.push({ p, body });
  let out;
  if (p === "list") out = { ok: true, release: [{ name: "p" }], user: [], cards: [{ id: "a", label: "A", arch: "sm86" }], rig_preset: { cards: [{ card: "a", pcie: { gen: 4, lanes: 8 } }] }, register: [] };
  else if (p === "load") out = { ok: true, doc: DOC, view: VIEW, name: "p", line: "nf", groups: [] };
  else if (p === "dry") out = { ok: true, goes: true, verdict: "ok", rejections: [], notes: [], cards: [] };
  else if (p === "issue") out = { ok: true, format: "markdown", text: "## Run report <b>x</b>\n### Model profile", blocks: ["Model profile"], filename: "run-report-p.md" };
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
  out.shown = root.innerHTML.indexOf("Run report &lt;b&gt;x&lt;/b&gt;") >= 0;          // maskiert, nie als HTML
  out.rawHtml = root.innerHTML.indexOf("<b>x</b>") >= 0;
  out.contains = root.innerHTML.indexOf("Model profile") >= 0;
  out.fresh = root.innerHTML.indexOf("Stale") < 0;
  click({ act: "issue-copy" }); await sleep(30);
  out.copied = copied;
  click({ act: "dry" }); await sleep(60);                                                // der Trockenlauf ändert die Grundlage
  out.stale = root.innerHTML.indexOf("Stale") >= 0;
  click({ act: "issue" }); await sleep(60);
  out.freshAgain = root.innerHTML.indexOf("Stale") < 0;
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
        self.assertEqual(set(o["body"]), {"doc", "dry", "cards", "model", "proposal"})
        self.assertEqual(o["body"]["doc"]["name"], "p")
        self.assertIsNone(o["body"]["dry"])                                  # noch kein Trockenlauf
        self.assertEqual(o["body"]["cards"], [{"card": "a", "pcie": {"gen": 4, "lanes": 8}}])
        self.assertIsNone(o["body"]["model"])
        self.assertTrue(o["shown"])
        self.assertFalse(o["rawHtml"])
        self.assertTrue(o["contains"])
        self.assertTrue(o["fresh"])
        self.assertIn("### Model profile", o["copied"])
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
