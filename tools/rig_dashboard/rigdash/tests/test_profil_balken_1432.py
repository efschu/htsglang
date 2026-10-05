"""PROFIL-EDITOR S4b (Auftrag 1432): ``/api/profil/recompute``, der langlebige Worker, die Balken und die Browser-Näherung.

Gepinnt:
* Route: nur Edition rig, nur LAN (Proxy 403, Release 404); der Körper wird zur Anfrage an die Kopplungs-Engine (Hardwareprofil, Modellprofil,
  Server-Argumente des Profils); jede Störung (kein Hardwareprofil, kein Modellpfad, Worker tot) kommt als ``ok: false`` mit Grund;
* Worker (``CouplingsService``): startet erst bei der ersten Anfrage, bleibt für mehrere Anfragen derselbe Prozess, startet nach Tod und nach
  Zeitüberschreitung neu, eine verspätete Antwort einer früheren Anfrage wird übersprungen (gegen einen Fake-Worker, deterministisch);
* Browser-Näherung (``static/profil_balken.js::approx``): Handrechnung mit festem Payload (Golden), Überlauf als eigenes Segment, HTML escaped;
* mit echtem Planer-Baum (``COUPLINGS_TREE`` = ``<py-Integrationsbaum>/python``): Kopplungs-Antwort mit Balken und ``approx``, und die JS-Näherung
  liegt für verschobene Layer innerhalb 0,5 MiB neben der Server-Antwort (läuft nur, wenn der Baum gesetzt ist).
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

from rigdash import profil_recompute as R  # noqa: E402
from rigdash import server as S  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
JS = os.path.join(STATIC, "profil_balken.js")
REAL_TREE = os.environ.get("COUPLINGS_TREE")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)

FAKE_WORKER = r'''
import json, sys, time
print(json.dumps({"id": 0, "ok": True, "ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    rid = req.pop("id")
    what = req.get("what")
    if what == "die":
        sys.exit(3)
    if what == "sleep":
        time.sleep(5)
    if what == "late":
        print(json.dumps({"id": rid - 1, "ok": True, "result": "veraltet"}), flush=True)
    print(json.dumps({"id": rid, "ok": True, "result": {"echo": req}}), flush=True)
'''


def fake_service(tmp, **kw):
    path = os.path.join(tmp, "fake_worker.py")
    with open(path, "w") as fh:
        fh.write(FAKE_WORKER)
    return R.CouplingsService(tmp, python=sys.executable, worker=path, **kw)

SETUP_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from sglang.srt.planner import profile_couplings as PC
from sglang.srt.weg2 import model_profile as MP
fx = os.path.join(sys.argv[1], "..", "test", "registered", "unit", "weg2", "fixtures", "profil_s3_1003", sys.argv[2])
hw = PC.synthetic_hardware([("NVIDIA GeForce RTX 5090", 32607, 1400.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0), ("NVIDIA GeForce RTX 3080", 20480, 700.0)])
print(json.dumps({"hw": hw, "model": MP.estimate(os.path.abspath(fx))}))
"""


def profiles_via_subprocess(tree, fixture):
    """Hardware- und Modellprofil im KINDPROZESS bauen: der Dashboard-Testprozess darf sglang nicht importieren
    (test_modellprofil_960::test_tree_is_found_by_file_path_not_by_import)."""
    p = subprocess.run([sys.executable, "-c", SETUP_SCRIPT, tree, fixture], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore"))
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-600:])
    return json.loads(p.stdout)


class TestRequestBuilding(unittest.TestCase):
    def test_args_of_takes_the_last_value_per_flag(self):
        doc = {"args": [{"flag": "--pp-stage-ratio", "values": ["29,11,8"]}, {"flag": "--p-bs", "values": ["2"]},
                        {"flag": "--p-bs", "values": ["3"]}, {"flag": "--p-hostgap", "values": []}, {"noflag": 1}]}
        self.assertEqual(R.args_of(doc), {"--pp-stage-ratio": "29,11,8", "--p-bs": "3", "--p-hostgap": ""})

    def test_build_request_carries_profiles_and_server_args(self):
        req = R.build_request({"doc": {"args": [{"flag": "--p-bs", "values": ["2"]}]}, "what": "move", "src": 1, "dst": 0, "n": 2,
                               "settings": {"context_tokens": 4096}, "phases": {"P": {"chunk_tokens": 8192}}},
                              hardware={"cards": []}, model={"arch": {}})
        self.assertEqual(req["what"], "move")
        self.assertEqual(req["server_args"], {"--p-bs": "2"})
        self.assertEqual((req["src"], req["dst"], req["n"]), (1, 0, 2))
        self.assertEqual(req["phases"], {"P": {"chunk_tokens": 8192}})
        self.assertEqual(req["settings"], {"context_tokens": 4096})

    def test_bad_requests_are_refused_before_the_worker(self):
        for body in ({"what": "start"}, {"doc": []}, {"doc": {}, "settings": []}, {"doc": {}, "phases": []}):
            with self.assertRaises(R.RecomputeError):
                R.build_request(body, hardware={}, model={})


class TestWorkerLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf1432_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_starts_on_first_request_and_stays_one_process(self):
        svc = fake_service(self.tmp)
        self.addCleanup(svc.close)
        self.assertEqual(svc.starts, 0)
        a = svc.request({"what": "bars", "x": 1})
        b = svc.request({"what": "bars", "x": 2})
        self.assertEqual((a["result"]["echo"]["x"], b["result"]["echo"]["x"]), (1, 2))
        self.assertEqual(svc.starts, 1)

    def test_a_dead_worker_is_named_and_restarted_by_the_next_request(self):
        svc = fake_service(self.tmp)
        self.addCleanup(svc.close)
        svc.request({"what": "bars"})
        dead = svc.request({"what": "die"})
        self.assertFalse(dead["ok"])
        self.assertIn("gestorben", dead["error"])
        again = svc.request({"what": "bars", "x": 3})
        self.assertTrue(again["ok"], again)
        self.assertEqual(svc.starts, 2)

    def test_a_timeout_is_named_and_the_worker_is_replaced(self):
        svc = fake_service(self.tmp, timeout_s=0.7)
        self.addCleanup(svc.close)
        slow = svc.request({"what": "sleep"})
        self.assertFalse(slow["ok"])
        self.assertIn("nicht fertig", slow["error"])
        self.assertTrue(svc.request({"what": "bars"})["ok"])
        self.assertEqual(svc.starts, 2)

    def test_a_late_answer_of_an_earlier_request_is_skipped(self):
        svc = fake_service(self.tmp)
        self.addCleanup(svc.close)
        svc.request({"what": "bars"})
        r = svc.request({"what": "late"})
        self.assertEqual(r["result"]["echo"]["what"], "late")      # nicht "veraltet"

    def test_missing_tree_or_python_is_named(self):
        r = R.CouplingsService(None).request({"what": "bars"})
        self.assertFalse(r["ok"])
        self.assertIn("kein Planer-Baum", r["error"])
        r = R.CouplingsService(self.tmp, python="/nicht/da/python").request({"what": "bars"})
        self.assertFalse(r["ok"])
        self.assertIn("Python der sglang-Umgebung fehlt", r["error"])

    def test_a_tree_without_the_module_is_named_by_the_real_worker(self):
        fixture = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
        svc = R.CouplingsService(fixture, python=sys.executable)
        self.addCleanup(svc.close)
        r = svc.request({"what": "bars"})
        self.assertFalse(r["ok"])
        self.assertIn("profile_couplings", r["error"])


class Hw:
    def __init__(self, ok=True):
        self.ok = ok

    def get(self):
        if not self.ok:
            return {"ok": False, "error": "nvml fehlt"}
        return {"ok": True, "profile": {"schema": "flliper.hardware/1", "cards": [{"ord": 0}]}}


class Est:
    def __init__(self):
        self.calls = []

    def estimate(self, req):
        self.calls.append(req)
        if req["path"].startswith("/nicht"):
            raise ValueError("%s liegt nicht unter einer Modellwurzel" % req["path"])
        return {"ok": True, "profile": {"schema": "flliper.model/1", "path": req["path"]}}


class TestRoute(unittest.TestCase):
    def serve(self, edition="rig", hw=None, tmp=None):
        self.est = Est()
        svc = fake_service(tmp)
        self.addCleanup(svc.close)
        app = SimpleNamespace(edition=edition, profil=None, hwprofil=hw or Hw(), modellprofil=self.est, couplings=svc, version="t")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return srv.server_address[1]

    def call(self, port, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        c.request("POST", "/api/profil/recompute", body=json.dumps(body), headers=h)
        r = c.getresponse()
        txt = r.read().decode()
        try:
            return r.status, json.loads(txt or "{}")
        except ValueError:
            return r.status, {"raw": txt}          # 404 "not found" ist Klartext

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf1432r_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    DOC = {"vars": [{"name": "PROFILE_MODEL", "value": "/m/nf"}], "args": [{"flag": "--pp-stage-ratio", "values": ["29,11,8"]},
                                                                               {"flag": "--kv-cache-dtype", "values": ["fp8_e4m3"]}]}

    def test_the_body_becomes_the_engine_request(self):
        port = self.serve(tmp=self.tmp)
        st, j = self.call(port, {"doc": self.DOC, "what": "bars"})
        self.assertEqual(st, 200)
        echo = j["result"]["echo"]
        self.assertEqual(echo["what"], "bars")
        self.assertEqual(echo["server_args"]["--pp-stage-ratio"], "29,11,8")
        self.assertEqual(echo["hardware"]["schema"], "flliper.hardware/1")
        self.assertEqual(echo["model"]["path"], "/m/nf")
        self.assertEqual(self.est.calls[0], {"path": "/m/nf", "kv_dtype": "fp8_e4m3"})
        self.assertEqual(j["model_path"], "/m/nf")

    def test_model_path_falls_back_to_the_bodys_and_the_flag(self):
        port = self.serve(tmp=self.tmp)
        self.call(port, {"doc": {"args": [{"flag": "--model-path", "values": ["/m/flag"]}]}, "what": "bars"})
        self.call(port, {"doc": self.DOC, "model_path": "/m/body", "what": "bars"})
        self.assertEqual([c["path"] for c in self.est.calls], ["/m/flag", "/m/body"])

    def test_failures_are_ok_false_with_a_reason(self):
        port = self.serve(hw=Hw(ok=False), tmp=self.tmp)
        st, j = self.call(port, {"doc": self.DOC})
        self.assertEqual(st, 200)
        self.assertFalse(j["ok"])
        self.assertIn("Hardwareprofil nicht verfügbar", j["error"])
        port = self.serve(tmp=self.tmp)
        self.assertIn("kein Modellpfad", self.call(port, {"doc": {"args": []}})[1]["error"])
        st, j = self.call(port, {"doc": {"vars": [{"name": "PROFILE_MODEL", "value": "/nicht/hier"}]}})
        self.assertEqual(st, 400)
        self.assertIn("Modellwurzel", j["error"])
        self.assertEqual(self.call(port, {"doc": self.DOC, "what": "start"})[0], 400)

    def test_lan_only_and_in_the_release_edition_too(self):
        # Auftrag 1984: die Balken sind Rechnung ohne Rig-Eingriff und kommen mit dem Reiter Profil ins Release (weiter nur LAN)
        port = self.serve(tmp=self.tmp)
        self.assertEqual(self.call(port, {"doc": self.DOC}, headers={"X-Forwarded-For": "1.2.3.4"})[0], 403)
        rel = self.serve("release", tmp=self.tmp)
        self.assertEqual(self.call(rel, {"doc": self.DOC})[0], 200)
        self.assertEqual(self.call(rel, {"doc": self.DOC}, headers={"X-Forwarded-For": "1.2.3.4"})[0], 403)
        c = http.client.HTTPConnection("127.0.0.1", rel, timeout=30)
        c.request("GET", "/profil_balken.js")
        self.assertEqual(c.getresponse().status, 200)


@unittest.skipUnless(NODE, "node fehlt")
class TestJsApprox(unittest.TestCase):
    #: 2 Stufen, 4 Layer (attn, gdn, gdn, attn), alles in MiB: Handrechnung
    PAYLOAD = {"n_stages": 2, "stage_layers": [2, 2], "layer_dense_mib": [100, 80, 80, 100], "layer_expert_mib": [400, 400, 400, 400],
               "layer_attn": [1, 0, 0, 1], "embed_mib": 50, "lm_head_mib": 70, "replicated_mib": 0, "draft_mib": 0, "draft_layers": 0,
               "buf_fracs": [0.5, 0.25], "cell_mib": 0.001, "context_tokens": 1000, "chunk_rows": 100, "extend_rate_mib": 0.5,
               "activation_mib": None, "state_per_layer_mib": 2, "slots": 3, "fixed_mib": [10, 20], "budget_mib": [1000, 1000],
               "total_mib": [1200, 1200]}

    def run_js(self, script):
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_hand_calculation_golden(self):
        o = self.run_js("const M=require(%r);console.log(JSON.stringify(M.approx(%s,[2,2])))" % (JS, json.dumps(self.PAYLOAD)))
        # Stufe 0: dichte Gewichte 100+80+Einbettung 50 = 230; Experten 0.5 x 800 = 400; KV 1000 x 1 Attn x 0.001 = 1; Zustand 1 Linear x 2 x 3 = 6;
        #          Aktivierung 100 x 0.5 = 50; fest 10 -> 697
        s0 = o[0]
        self.assertEqual((s0["weights"], s0["experts"], s0["kv"], s0["state"], s0["activation"], s0["fixed"]), (230, 400, 1, 6, 50, 10))
        self.assertEqual(s0["needs"], 697)
        self.assertEqual(s0["free"], 303)
        # Stufe 1: 80+100+lm_head 70 = 250; Experten 0.25 x 800 = 200; KV 1; Zustand 6; Aktivierung 50; fest 20 -> 527
        self.assertEqual((o[1]["weights"], o[1]["experts"], o[1]["needs"]), (250, 200, 527))

    def test_moving_a_layer_moves_the_posts(self):
        o = self.run_js("const M=require(%r);console.log(JSON.stringify(M.approx(%s,[3,1])))" % (JS, json.dumps(self.PAYLOAD)))
        self.assertEqual(o[0]["weights"], 310)                    # 100+80+80 + 50
        self.assertEqual(o[1]["weights"], 170)                    # 100 + 70
        self.assertEqual(o[0]["state"], 12)                       # 2 Linear-Layer
        self.assertEqual(o[1]["state"], 0)

    def test_overflow_is_its_own_segment_and_the_cut_posts_are_named(self):
        pl = dict(self.PAYLOAD, budget_mib=[400, 1000])
        o = self.run_js("const M=require(%r);console.log(JSON.stringify(M.approxBars(%s,[2,2],['K0','K1'])))" % (JS, json.dumps(pl)))
        b = o[0]
        keys = [s["key"] for s in b["segments"]]
        self.assertIn("overflow", keys)
        self.assertNotIn("free_in_budget", keys)
        self.assertAlmostEqual(b["overflow_mib"], 697 - 400, places=6)
        ov = next(s for s in b["segments"] if s["key"] == "overflow")
        self.assertTrue(ov["cut"])
        self.assertAlmostEqual(sum(s["mib"] for s in b["segments"]), b["total_mib"] + b["overflow_mib"], places=6)
        self.assertNotIn("overflow", [s["key"] for s in o[1]["segments"]])

    def test_wrong_cut_is_refused_and_html_is_escaped(self):
        o = self.run_js(
            "const M=require(%r);let err=null;try{M.approx(%s,[1,1])}catch(e){err=e.message}\n"
            "const bar={label:'<b>K</b>',total_mib:100,budget_mib:80,overflow_mib:0,free_mib:10,segments:[{key:'kv',label:'<i>KV</i>',mib:70,origin:'x'}]};\n"
            "console.log(JSON.stringify({err,html:M.render([bar],{base:0}),tip:M.tip(bar,0)}))" % (JS, json.dumps(self.PAYLOAD)))
        self.assertIn("passt nicht zum Modell", o["err"])
        self.assertNotIn("<b>K</b>", o["html"])
        self.assertIn("&lt;b&gt;K&lt;/b&gt;", o["html"])
        self.assertNotIn("<i>KV</i>", o["tip"])

    def test_the_page_loads_the_module_and_the_tab_draws_the_fold(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        self.assertIn('<script src="profil_balken.js"></script>', S.edition_page(html, "rig"))
        self.assertIn('<script src="profil_balken.js"></script>', S.edition_page(html, "release"))
        pj = open(os.path.join(STATIC, "profil.js"), encoding="utf-8").read()
        for needle in ('api("recompute"', "ProfilBalken.attach", 'data-fold="bars"', "setTimeout(recompute, 300)"):
            self.assertIn(needle, pj)


@unittest.skipUnless(REAL_TREE and os.path.isdir(REAL_TREE) and NODE, "COUPLINGS_TREE (<py-Baum>/python) oder node nicht gesetzt")
class TestRealTree(unittest.TestCase):
    def setUp(self):
        prof = profiles_via_subprocess(REAL_TREE, "nextflash_int4mixed")
        self.model, self.hw = prof["model"], prof["hw"]
        self.svc = R.CouplingsService(REAL_TREE, python=sys.executable)
        self.addCleanup(self.svc.close)
        self.settings = {"stage_layers": [29, 11, 8], "budget_mib": [28440, 17568, 18200], "kv_dtype": "fp8_e4m3", "context_tokens": 262144,
                         "chunk_tokens": 16384, "mamba_slots": 32, "scratch_rows": 32, "ssm_dtype": "bfloat16",
                         "activation_mib": [5984, 2952, 2944], "moe_resident_fraction": [0.33, 0.701, 0.995605]}

    def ask(self, **kw):
        req = {"what": "bars", "hardware": self.hw, "model": self.model, "settings": dict(self.settings)}
        req.update(kw)
        return self.svc.request(req)

    def test_bars_with_approx_payload_and_overflow_when_the_budget_is_cut(self):
        ok = self.ask()
        self.assertTrue(ok["ok"], ok)
        bars = ok["result"]["phases"]["alle"]["bars"]
        self.assertEqual(len(bars), 3)
        self.assertTrue(all(b["overflow_mib"] == 0 for b in bars))
        tight = self.ask(settings=dict(self.settings, budget_mib=[28440, 12000, 18200]))["result"]["phases"]["alle"]["bars"]
        self.assertGreater(tight[1]["overflow_mib"], 0)
        self.assertIn("overflow", [s["key"] for s in tight[1]["segments"]])
        self.assertEqual(self.svc.starts, 1)

    def test_browser_approximation_stays_within_half_a_mib_of_the_server(self):
        res = self.ask()["result"]
        pl = res["approx"]
        for counts in ([29, 11, 8], [31, 10, 7], [26, 13, 9], [24, 14, 10]):
            js = self.run_js("const M=require(%r);console.log(JSON.stringify(M.approx(%s,%s)))" % (JS, json.dumps(pl), json.dumps(counts)))
            server = self.ask(settings=dict(self.settings, stage_layers=counts))["result"]["phases"]["alle"]["bars"]
            for ap, sv in zip(js, server):
                self.assertAlmostEqual(ap["needs"], sv["needs_mib"], delta=0.5, msg=str(counts))
                self.assertAlmostEqual(ap["free"], sv["budget_mib"] - sv["needs_mib"], delta=0.5, msg=str(counts))

    def test_phases_and_peak(self):
        r = self.ask(phases={"P": {}, "D": {"activation_mib": [900, 900, 900]}})["result"]
        self.assertEqual(set(r["phases"]), {"P", "D"})
        self.assertEqual({b["from_phase"] for b in r["Spitze"]["bars"]}, {"P"})

    def test_bad_cut_is_a_named_error(self):
        r = self.ask(settings=dict(self.settings, stage_layers=[29, 11]))
        self.assertFalse(r["ok"])
        self.assertIn("vector_length", r["error"])

    def run_js(self, script):
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)


if __name__ == "__main__":
    unittest.main()
