"""AP-H2 (Auftrag 880): Balken je Karte und Phase im Vertrag ``flliper.balken/1`` -- Dashboard-Seite.

Gepinnt:

* Gruppenzeilen des Profils (``--extra-p/-d``, ``--env-p/-d``) werden je Phase gelesen, in BEIDEN Darstellungen, die der Importer liefert (Wert in
  ``values`` oder -- ohne Specs importiert -- als eigener Eintrag danach, auch in der ``flag=wert``-Form); Draft-Pfad aus Koerper, Flag, Gruppenzeile,
  ``PROFILE_DRAFT``;
* Route ``what=phase_bars``: Draft-Verzeichnis wird mitprofiliert, ein nicht lesbares Draft-Verzeichnis ist ``draft_error`` und rechnet ohne es;
  ``what=bars`` bleibt unveraendert (kein Draft, keine Gruppenzeilen);
* Darstellung (``static/profil_balken.js``, Node): ein zusammenhaengender Balken je Karte in Vertragsreihenfolge, Chips "nicht gerechnet", rote
  Zone hinter der Kartenkante bei Ueberlauf, Tooltip mit MiB/Herkunft/Grund, HTML escaped, fehlgeschlagene Phase als Meldung;
* Naeherung im Browser: ``toContract``/``approxContractBars`` halten Summenregel und Reihenfolge ein; mit echtem Planer-Baum
  (``COUPLINGS_TREE``) stimmt das JS-``contractBar`` mit ``contract_bar`` (Python) ueberein und die Naeherung liegt innerhalb 0,5 MiB neben der
  P-Phase des Servers.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import profil_recompute as R  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
JS = os.path.join(STATIC, "profil_balken.js")
REAL_TREE = os.environ.get("COUPLINGS_TREE")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)
ABL_ENV = "/spinning/gpu-arb/docker/profiles/nf-int4-h6-abl.env"

#: Form des importierten NF-abl-Profils OHNE Specs: der Gruppentext steht als eigener Eintrag nach dem nackten --extra-p; D-Text in der eq-Form
ABL_DOC = {
    "vars": [{"name": "PROFILE_DRAFT", "value": "/m/draft-var"}],
    "args": [
        {"flag": "--pp-stage-ratio", "values": ["29,11,8"]},
        {"flag": "--env-p", "values": ["SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.330,0.701,0.652"]},
        {"flag": "--env-d", "values": ["SGLANG_MOE_SCRATCH_SLOTS=104,48,48;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.06,0.51,0.48"]},
        {"flag": "--extra-p", "values": []},
        {"flag": "--max-total-tokens 262144 --kv-cache-dtype fp8_e4m3 --max-mamba-cache-size 32 --hicache-size 2", "values": []},
        {"flag": "--extra-d", "values": []},
        {"flag": "--max-total-tokens 262144 --rank-role host,worker,worker --rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
                 "--speculative-draft-model-path /m/draft-extra --cuda-graph-bs-decode 1 2 3 4 5 6 --cuda-graph-backend-decode",
         "values": ["full --cuda-graph-backend-prefill=disabled --hicache-size 4"], "eq": True},
        {"token": "--d-only"},
    ],
}
#: Form MIT Specs: der Wert steht in values (--extra-p=...)
SPEC_DOC = {"args": [{"flag": "--extra-p", "values": ["--max-running-requests=1 --rank-gpu-memory-mib 6610,5050,5200 --max-mamba-cache-size=8"], "eq": True},
                     {"flag": "--extra-d", "values": ['--json-model-override-args "{\\"language_model_only\\":true}" --rank-tp-ratio 1,1,1']},
                     {"flag": "--dual-share"}]}


class TestGroupLines(unittest.TestCase):
    def test_both_representations_give_the_group_texts(self):
        gt = R.group_texts(ABL_DOC)
        self.assertEqual(set(gt), {"--extra-p", "--extra-d", "--env-p", "--env-d"})
        self.assertTrue(gt["--extra-p"].startswith("--max-total-tokens 262144 --kv-cache-dtype"))
        self.assertIn("--cuda-graph-backend-decode=full --cuda-graph-backend-prefill=disabled", gt["--extra-d"])      # eq-Form zusammengesetzt
        sp = R.group_texts(SPEC_DOC)
        self.assertEqual(sp["--extra-p"], "--max-running-requests=1 --rank-gpu-memory-mib 6610,5050,5200 --max-mamba-cache-size=8")

    def test_scoped_args_read_flag_equals_value_multi_values_and_quoted_json(self):
        d = R.scoped_args(R.group_texts(ABL_DOC)["--extra-d"])
        self.assertEqual(d["--rank-tp-ratio"], "1,0,0")
        self.assertEqual(d["--rank-moe-ratio"], "183,137,168")
        self.assertEqual(d["--cuda-graph-bs-decode"], "1 2 3 4 5 6")
        self.assertEqual((d["--cuda-graph-backend-decode"], d["--cuda-graph-backend-prefill"], d["--hicache-size"]), ("full", "disabled", "4"))
        p = R.scoped_args(R.group_texts(SPEC_DOC)["--extra-p"])
        self.assertEqual((p["--max-running-requests"], p["--rank-gpu-memory-mib"], p["--max-mamba-cache-size"]), ("1", "6610,5050,5200", "8"))
        j = R.scoped_args(R.group_texts(SPEC_DOC)["--extra-d"])
        self.assertEqual(j["--json-model-override-args"], '{"language_model_only":true}')
        self.assertEqual(R.scoped_args("not a flag"), {})
        self.assertEqual(R.scoped_args("--bare --x 1"), {"--bare": "", "--x": "1"})
        self.assertEqual(R.scoped_args('--broken "unclosed'), {"--broken": '"unclosed'})

    def test_env_map(self):
        e = R.env_map("A=1;B=0.1,0.2;;C=x=y;bare")
        self.assertEqual(e, {"A": "1", "B": "0.1,0.2", "C": "x=y"})

    def test_phase_inputs_split_the_groups_and_keep_the_bare_tokens(self):
        pi = R.phase_inputs(ABL_DOC)
        self.assertEqual(pi["phase_args"]["P"]["--max-mamba-cache-size"], "32")
        self.assertEqual(pi["phase_args"]["D"]["--rank-role"], "host,worker,worker")
        self.assertEqual(pi["phase_env"]["P"]["SGLANG_MOE_SCRATCH_SLOTS"], "32")
        self.assertEqual(pi["phase_env"]["D"]["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"], "0.06,0.51,0.48")
        self.assertEqual(pi["tokens"], ["--d-only"])
        empty = R.phase_inputs({"args": []})
        self.assertEqual((empty["phase_args"], empty["phase_env"], empty["tokens"]), ({"P": {}, "D": {}}, {"P": {}, "D": {}}, []))

    def test_a_bare_group_flag_does_not_swallow_the_next_plain_flag(self):
        doc = {"args": [{"flag": "--extra-p", "values": []}, {"flag": "--p-bs", "values": ["6"]}]}
        self.assertEqual(R.group_texts(doc), {})

    def test_draft_path_order(self):
        self.assertEqual(R.draft_path_of(ABL_DOC), "/m/draft-extra")                       # Gruppenzeile vor PROFILE_DRAFT
        self.assertEqual(R.draft_path_of({"args": [{"flag": "--dflash-draft-path", "values": ["/m/dflash"]}], "vars": []}), "/m/dflash")
        self.assertEqual(R.draft_path_of({"args": [], "vars": [{"name": "PROFILE_DRAFT", "value": "/m/v"}]}), "/m/v")
        self.assertEqual(R.draft_path_of(ABL_DOC, {"draft_path": "/m/body"}), "/m/body")
        self.assertIsNone(R.draft_path_of({"args": [], "vars": []}))

    @unittest.skipUnless(os.path.exists(ABL_ENV) and REAL_TREE, "echtes abl-Profil und COUPLINGS_TREE fehlen")
    def test_the_real_abl_profile_yields_the_reference_group_lines(self):
        out = subprocess.run([sys.executable, "-c", "import json,sys;sys.path.insert(0,sys.argv[1]);from sglang.srt.weg2 import profile_json as P;"
                              "print(json.dumps(P.import_env(sys.argv[2])))", REAL_TREE, ABL_ENV], capture_output=True, text=True, timeout=120,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore"))
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        pi = R.phase_inputs(json.loads(out.stdout))
        self.assertEqual(pi["phase_args"]["D"]["--rank-tp-ratio"], "1,0,0")
        self.assertEqual(pi["phase_args"]["D"]["--rank-moe-ratio"], "183,137,168")
        self.assertEqual(pi["phase_env"]["D"]["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"], "0.06,0.51,0.48")
        self.assertEqual(pi["phase_env"]["P"]["SGLANG_MOE_SCRATCH_SLOTS"], "32")
        self.assertEqual(pi["phase_args"]["P"]["--max-mamba-cache-size"], "32")


class TestRequest(unittest.TestCase):
    def test_phase_bars_request_carries_the_group_lines_and_the_form(self):
        req = R.build_request({"doc": ABL_DOC, "what": "phase_bars", "form": "flip"}, hardware={"cards": []}, model={"arch": {}})
        self.assertEqual(req["what"], "phase_bars")
        self.assertEqual(req["phase_args"]["D"]["--rank-tp-ratio"], "1,0,0")
        self.assertEqual(req["phase_env"]["P"]["SGLANG_MOE_SCRATCH_SLOTS"], "32")
        self.assertEqual((req["tokens"], req["form"]), (["--d-only"], "flip"))
        self.assertEqual(req["server_args"]["--pp-stage-ratio"], "29,11,8")

    def test_bars_request_stays_as_it_was(self):
        req = R.build_request({"doc": ABL_DOC, "what": "bars"}, hardware={}, model={})
        for k in ("phase_args", "phase_env", "tokens", "form"):
            self.assertNotIn(k, req)

    def test_phase_bars_is_an_allowed_operation(self):
        self.assertIn("phase_bars", R.WHAT)


class TestRouteDraft(unittest.TestCase):
    """Route ``what=phase_bars`` gegen einen Fake-Worker (Aufbau wie test_profil_balken_1432)."""

    def serve(self, est):
        import http.client  # noqa: F401
        import threading
        from http.server import ThreadingHTTPServer
        from types import SimpleNamespace

        from rigdash import server as S
        from rigdash.tests.test_profil_balken_1432 import Hw, fake_service

        tmp = tempfile.mkdtemp(prefix="pf_aph2_")
        self.addCleanup(shutil.rmtree, tmp, True)
        svc = fake_service(tmp)
        self.addCleanup(svc.close)
        app = SimpleNamespace(edition="rig", profil=None, hwprofil=Hw(), modellprofil=est, couplings=svc, version="t")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return srv.server_address[1]

    def call(self, port, body):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        c.request("POST", "/api/profil/recompute", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read().decode() or "{}")

    class Est:
        def __init__(self, bad_draft=False):
            self.calls, self.bad_draft = [], bad_draft

        def estimate(self, req):
            self.calls.append(dict(req))
            if req.get("draft_path") and self.bad_draft:
                raise ValueError("%s liegt nicht unter einer Modellwurzel" % req["draft_path"])
            return {"ok": True, "profile": {"schema": "flliper.model/1", "path": req["path"], "draft": bool(req.get("draft_path"))}}

    DOC = dict(ABL_DOC, vars=[{"name": "PROFILE_MODEL", "value": "/m/nf"}, {"name": "PROFILE_DRAFT", "value": "/m/draft-var"}])

    def test_the_draft_directory_is_profiled_along_and_the_request_carries_the_group_lines(self):
        est = self.Est()
        st, j = self.call(self.serve(est), {"doc": self.DOC, "what": "phase_bars"})
        self.assertEqual(st, 200)
        self.assertEqual(est.calls[0]["draft_path"], "/m/draft-extra")
        echo = j["result"]["echo"]
        self.assertEqual(echo["what"], "phase_bars")
        self.assertEqual(echo["phase_args"]["D"]["--rank-moe-ratio"], "183,137,168")
        self.assertTrue(echo["model"]["draft"])
        self.assertNotIn("draft_error", j)

    def test_an_unreadable_draft_directory_is_named_and_the_bars_run_without_it(self):
        est = self.Est(bad_draft=True)
        st, j = self.call(self.serve(est), {"doc": self.DOC, "what": "phase_bars"})
        self.assertEqual(st, 200)
        self.assertIn("nicht profiliert", j["draft_error"])
        self.assertIn("Modellwurzel", j["draft_error"])
        self.assertEqual(len(est.calls), 2)
        self.assertNotIn("draft_path", est.calls[1])
        self.assertFalse(j["result"]["echo"]["model"]["draft"])

    def test_the_legacy_bars_operation_does_not_touch_the_draft(self):
        est = self.Est()
        st, j = self.call(self.serve(est), {"doc": self.DOC, "what": "bars"})
        self.assertEqual(st, 200)
        self.assertEqual(est.calls, [{"path": "/m/nf", "kv_dtype": None}])
        self.assertNotIn("phase_args", j["result"]["echo"])


@unittest.skipUnless(NODE, "node fehlt")
class TestJsContract(unittest.TestCase):
    def js(self, script):
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def bar(self, **kw):
        b = {"card": 0, "label": "Karte 0 (RTX 5090)", "phase": "P", "total_mib": 1000, "budget_mib": 900, "budget_herkunft": "Profilzeile",
             "segments": [{"name": "weights", "label": "Gewichte", "mib": 300, "herkunft": "Modellprofil/Hardwareprofil (Index)", "detail": "dichte Gewichte"},
                          {"name": "kv", "label": "KV", "mib": 200, "herkunft": "Naeherung (nicht der Loeser)", "detail": "Kontext x Layer x Zelle"},
                          {"name": "fixed", "label": "Festposten", "mib": None, "herkunft": "nicht gerechnet", "detail": "nur am Metall zu messen"},
                          {"name": "reserve", "label": "Reserve", "mib": 100, "herkunft": "Profilzeile", "detail": "Kartengroesse - Budget"},
                          {"name": "free", "label": "Frei", "mib": 400, "herkunft": "gerechnet", "detail": "Budget - Posten"}],
             "posts_mib": 500, "free_mib": 400, "overflow_mib": 0, "beyond_card_mib": 0, "not_computed": ["Festposten"]}
        b.update(kw)
        return b

    def test_one_contiguous_bar_per_card_in_contract_order_with_chips_for_what_is_not_computed(self):
        o = self.js("const M=require(%r);const b=%s;console.log(JSON.stringify({html:M.render([b],{base:0}),m:M.model(b)}))" % (JS, json.dumps(self.bar())))
        html = o["html"]
        self.assertEqual(html.count('class="kp-bar"'), 1)
        order = [m["s"]["key"] for m in o["m"]["rows"]]
        self.assertEqual(order, ["weights", "kv", "reserve", "free"])                 # null-Segment wird nicht gezeichnet
        self.assertAlmostEqual(o["m"]["sum"], 1000)
        self.assertIn("Festposten</b>: nicht gerechnet", html)
        self.assertIn("nur am Metall zu messen", html)                                 # Grund als Tooltip des Chips
        self.assertIn("Rest 400 MiB (Obergrenze)", html)                              # nicht gerechnete Posten: "Rest" ist eine Obergrenze
        self.assertNotIn("kp-bz", html)
        self.assertIn('class="kp-ph-chip">P<', html)

    def test_the_bar_grows_past_the_card_edge_with_a_red_zone_and_a_note(self):
        b = self.bar(segments=[{"name": "weights", "label": "Gewichte", "mib": 700, "herkunft": "x", "detail": "d"},
                               {"name": "kv", "label": "KV", "mib": 500, "herkunft": "y", "detail": "d"}],
                     posts_mib=1200, free_mib=0, overflow_mib=300, beyond_card_mib=200, not_computed=[])
        o = self.js("const M=require(%r);const b=%s;console.log(JSON.stringify({html:M.render([b],{base:0}),tip:M.tip(b,1)}))" % (JS, json.dumps(b)))
        self.assertIn('class="kp-bz"', o["html"])
        self.assertIn("Kartenende", o["html"])
        self.assertIn("200 MiB über der Karte", o["html"])
        self.assertIn('role="alert"', o["html"])
        # Segment 2 (KV 700..1200) beginnt VOR der Kartenkante (1000): 200 MiB davon liegen dahinter
        self.assertIn("200 MiB dieses Postens liegen HINTER der Kartengrenze", o["tip"])
        self.assertIn("Herkunft: <b>y</b>", o["tip"])

    def test_over_budget_inside_the_card_is_a_note_without_a_red_zone(self):
        b = self.bar(overflow_mib=50, beyond_card_mib=0)
        html = self.js("const M=require(%r);console.log(JSON.stringify(M.render([%s],{base:0})))" % (JS, json.dumps(b)))
        self.assertIn("50 MiB über dem Budget", html)
        self.assertNotIn("kp-bz", html)

    def test_render_phases_lists_each_phase_the_inputs_and_a_failed_phase_as_a_message(self):
        res = {"phases": {"P": {"ok": True, "label": "P-Phase <x>", "bars": [self.bar(), self.bar(label="Karte 1")], "inputs": [{"was": "stage_layers", "wert": "29,11,8", "herkunft": "Profilzeile --pp-stage-ratio"}]},
                          "D": {"ok": False, "label": "D-Phase", "error": "vector_length: <b>zu kurz</b>", "bars": []}}}
        o = self.js("const M=require(%r);const r=M.renderPhases(%s,{base:0});console.log(JSON.stringify({html:r.html,n:r.bars.length}))" % (JS, json.dumps(res)))
        self.assertEqual(o["n"], 2)
        self.assertEqual(o["html"].count('class="pf-bar"'), 2)
        self.assertIn("P-Phase &lt;x&gt;", o["html"])
        self.assertIn("Profilzeile --pp-stage-ratio", o["html"])
        self.assertIn("vector_length: &lt;b&gt;zu kurz&lt;/b&gt;", o["html"])
        self.assertNotIn("<b>zu kurz</b>", o["html"])

    def test_a_phase_without_an_ok_field_counts_as_ok_and_nonmatching_bars_from_the_old_shape_still_draw(self):
        res = {"phases": {"alle": {"bars": [{"label": "K0", "total_mib": 100, "budget_mib": 90, "overflow_mib": 0, "free_mib": 10,
                                             "segments": [{"key": "weights", "label": "Gewichte", "mib": 80, "origin": "x"}]}]}}}
        o = self.js("const M=require(%r);const r=M.renderPhases(%s,{base:0});console.log(JSON.stringify({html:r.html,n:r.bars.length}))" % (JS, json.dumps(res)))
        self.assertEqual(o["n"], 1)
        self.assertIn("K0", o["html"])

    def test_contract_bar_golden_matches_the_sum_rule(self):
        o = self.js("const M=require(%r);console.log(JSON.stringify([M.contractBar('K','P',1000,900,'Profilzeile',[{name:'weights',label:'W',mib:600},{name:'kv',label:'KV',mib:350}]),"
                    "M.contractBar('K','D',1000,900,'x',[{name:'weights',label:'W',mib:700},{name:'kv',label:'KV',mib:500}]),"
                    "M.contractBar('K','D',1000,900,'x',[{name:'weights',label:'W',mib:300},{name:'kv',label:'KV',mib:null,detail:'warum'}])]))" % JS)
        eaten, beyond, nc = o
        self.assertEqual((eaten["overflow_mib"], eaten["beyond_card_mib"]), (50, 0))
        self.assertEqual([(s["name"], s["mib"]) for s in eaten["segments"]], [("weights", 600), ("kv", 350), ("reserve", 50)])
        self.assertEqual((beyond["overflow_mib"], beyond["beyond_card_mib"]), (300, 200))
        self.assertEqual(sum(s["mib"] for s in beyond["segments"]), 1200)
        self.assertEqual(nc["not_computed"], ["KV"])
        self.assertIn("OBERGRENZE", nc["segments"][-1]["detail"])

    PAYLOAD = {"n_stages": 2, "stage_layers": [2, 2], "layer_dense_mib": [100, 80, 80, 100], "layer_expert_mib": [400, 400, 400, 400],
               "layer_attn": [1, 0, 0, 1], "embed_mib": 50, "lm_head_mib": 70, "replicated_mib": 0, "draft_mib": 0, "draft_layers": 0,
               "buf_fracs": [0.5, 0.25], "cell_mib": 0.001, "context_tokens": 1000, "chunk_rows": 100, "extend_rate_mib": 0.5,
               "activation_mib": None, "state_per_layer_mib": 2, "slots": 3, "fixed_mib": [0, 0], "budget_mib": [1000, 400], "total_mib": [1200, 1200]}

    def test_the_browser_approximation_becomes_contract_bars_with_unmeasured_fixed_posts(self):
        o = self.js("const M=require(%r);console.log(JSON.stringify(M.approxContractBars(%s,[2,2],['K0','K1'],'P')))" % (JS, json.dumps(self.PAYLOAD)))
        b0, b1 = o
        self.assertEqual(b0["phase"], "P")
        names = [s["name"] for s in b0["segments"]]
        self.assertEqual(names, ["weights", "experts", "kv", "state", "activation", "fixed", "reserve", "free"])
        self.assertEqual(b0["not_computed"], ["Festposten"])
        # Karte 0: 230 + 400 + 1 + 6 + 50 = 687 Posten gegen 1000 Budget
        self.assertAlmostEqual(b0["posts_mib"], 687)
        self.assertAlmostEqual(sum(s["mib"] for s in b0["segments"] if s["mib"] is not None), 1200)
        # Karte 1: Budget 400 < 250 + 200 + 1 + 6 + 50 = 507 Posten -> 107 ueber dem Budget, Reserve (800) um 107 gekuerzt, Kartensumme bleibt 1200
        self.assertAlmostEqual(b1["overflow_mib"], 107)
        self.assertEqual(b1["beyond_card_mib"], 0)
        self.assertAlmostEqual(sum(s["mib"] for s in b1["segments"] if s["mib"] is not None), 1200)
        self.assertNotIn("free", [s["name"] for s in b1["segments"]])

    def test_the_page_loads_the_styles_and_the_tab_asks_for_phase_bars(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        for needle in (".kp-bz", ".ks-r", ".kp-ncs", ".kp-nc "):
            self.assertIn(needle, html)
        pj = open(os.path.join(STATIC, "profil.js"), encoding="utf-8").read()
        for needle in ('what: "phase_bars"', "renderPhases", "approxContractBars", "draft_error"):
            self.assertIn(needle, pj)
        head = open(JS, encoding="utf-8").read().split("(function (root)")[0]
        for needle in ("flliper.balken/1", '"herkunft"', '"mib":', "NICHT GERECHNET", "Summenregel"):
            self.assertIn(needle, head)


@unittest.skipUnless(REAL_TREE and os.path.isdir(REAL_TREE) and NODE, "COUPLINGS_TREE (<py-Baum>/python) oder node nicht gesetzt")
class TestRealTree(unittest.TestCase):
    SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from sglang.srt.planner import profile_couplings as PC
from sglang.srt.weg2 import model_profile as MP
fx = os.path.join(sys.argv[1], "..", "test", "registered", "unit", "weg2", "fixtures", "profil_s3_1003", "nextflash_int4mixed")
hw = PC.synthetic_hardware([("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)])
model = MP.estimate(os.path.abspath(fx))
st = {"ord": 0, "label": "Karte 0", "total_mib": 1000.0, "budget_mib": {"v": 900.0, "src": "Profilzeile", "note": ""},
      "terms": {k: {"v": v, "src": "Eingabe", "note": ""} for k, v in json.loads(sys.argv[2]).items()}}
print(json.dumps({"hw": hw, "model": model, "bar": PC.contract_bar(st, "P")}))
"""

    def setUp(self):
        from rigdash.tests.test_profil_balken_1432 import profiles_via_subprocess
        prof = profiles_via_subprocess(REAL_TREE, "nextflash_int4mixed")
        self.model, self.hw = prof["model"], prof["hw"]
        self.svc = R.CouplingsService(REAL_TREE, python=sys.executable)
        self.addCleanup(self.svc.close)

    def js(self, script):
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def ask(self, **kw):
        req = {"what": "phase_bars", "hardware": self.hw, "model": self.model,
               "server_args": {"--pp-stage-ratio": "29,11,8", "--pp-attn-stage-ratio": "7,3,2", "--pp-cut-expert-device-fraction": "0.330,0.701,0.652",
                               "--pp-cut-expert-lru-rows": "32,32,32", "--max-kv-per-request": "262144", "--draft-kv-on-p": "off"},
               "phase_args": {"P": {"--kv-cache-dtype": "fp8_e4m3", "--max-mamba-cache-size": "32", "--mamba-ssm-dtype": "bfloat16",
                                    "--chunked-prefill-size": "16384"},
                              "D": {"--rank-tp-ratio": "1,0,0", "--rank-moe-ratio": "183,137,168", "--rank-moe-resident-fraction": "0.06,0.51,0.48"}},
               "phase_env": {"P": {}, "D": {"SGLANG_MOE_SCRATCH_SLOTS": "104,48,48"}}}
        req.update(kw)
        return self.svc.request(req)

    def test_the_worker_answers_the_contract_for_both_phases(self):
        r = self.ask()
        self.assertTrue(r["ok"], r)
        res = r["result"]
        self.assertEqual((res["schema"], res["form"], list(res["phases"])), ("flliper.balken/1", "flip", ["P", "D"]))
        self.assertEqual(len(res["phases"]["P"]["bars"]), 3)
        self.assertEqual(len(res["phases"]["D"]["bars"]), 3)
        self.assertEqual(self.svc.starts, 1)

    def test_js_contract_bar_equals_python_contract_bar(self):
        posts = {"weights": 300.0, "experts": 120.0, "draft": 0.0, "kv": None, "state": 9.5, "activation": 80.0, "fixed": None}
        out = subprocess.run([sys.executable, "-c", self.SCRIPT, REAL_TREE, json.dumps(posts)], capture_output=True, text=True, timeout=120,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore"))
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        py = json.loads(out.stdout)["bar"]
        jsin = [{"name": k, "label": k, "mib": v} for k, v in posts.items()]
        jb = self.js("const M=require(%r);console.log(JSON.stringify(M.contractBar('Karte 0','P',1000,900,'x',%s)))" % (JS, json.dumps(jsin)))
        self.assertEqual([(s["name"], s["mib"]) for s in jb["segments"] if s["mib"] is not None],
                         [(s["name"], s["mib"]) for s in py["segments"] if s["mib"] is not None])
        for k in ("posts_mib", "free_mib", "overflow_mib", "beyond_card_mib"):
            self.assertAlmostEqual(jb[k], py[k], places=6, msg=k)

    def test_browser_approximation_stays_within_half_a_mib_of_the_p_phase(self):
        res = self.ask()["result"]
        pl = res["approx"]
        for counts in ([29, 11, 8], [31, 10, 7], [26, 13, 9]):
            args = {"--pp-stage-ratio": ",".join(map(str, counts)), "--pp-cut-expert-device-fraction": "0.330,0.701,0.652", "--pp-cut-expert-lru-rows": "32,32,32",
                    "--max-kv-per-request": "262144", "--draft-kv-on-p": "off"}
            srv = self.ask(server_args=args)["result"]["phases"]["P"]["bars"]
            js = self.js("const M=require(%r);console.log(JSON.stringify(M.approxContractBars(%s,%s,['a','b','c'],'P')))" % (JS, json.dumps(pl), json.dumps(counts)))
            for a, s in zip(js, srv):
                self.assertAlmostEqual(a["posts_mib"], s["posts_mib"], delta=0.5, msg=str(counts))

    def test_d_phase_has_no_weights_on_zero_weight_ranks_and_reserve_is_the_default_corridor(self):
        d = self.ask()["result"]["phases"]["D"]["bars"]
        self.assertFalse(any(s["name"] == "weights" for s in d[1]["segments"]))
        self.assertEqual(next(s for s in d[0]["segments"] if s["name"] == "reserve")["mib"], 1024.0)


if __name__ == "__main__":
    unittest.main()
