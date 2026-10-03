"""PROFIL-EDITOR S3 (Auftrag 960): die Route ``POST /api/modellprofil/schaetzen`` und das Modul ``static/modellprofil.js``.

Gepinnt:
  * der Schätzer (``sglang/srt/weg2/model_profile.py``, reine Standardbibliothek) wird aus dem Fixture-Planer-Baum per Dateipfad
    geladen, nicht über ``import sglang``; die Antwort ist ``flliper.model/1`` mit Quelle an jedem Wert, ohne Geheimnis;
  * der Pfad muss unter einer Modellwurzel liegen (relativ, ``..``, Symlink hinaus, NUL, fehlend: ValueError / HTTP 400);
  * die Antwort wird je Pfad+Dateistand gemerkt, eine geänderte Datei rechnet neu;
  * Route nur im LAN (Proxy 403), nicht im Release (404); Körper <= 64 KiB; das Modul wird ausgeliefert;
  * ``modellprofil.js`` baut aus dem Profil Zeilen mit Quelle und escaped HTML (Node, wenn vorhanden).
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import modellprofil as M  # noqa: E402
from rigdash import redact  # noqa: E402
from rigdash import server as S  # noqa: E402

TREE = os.path.join(HERE, "fixtures", "modellprofil", "planner_tree", "python")
JS = os.path.join(os.path.dirname(HERE), "static", "modellprofil.js")
DT = {"BF16": 2, "U8": 1, "I8": 1, "F32": 4, "I32": 4}

CFG = {"architectures": ["TinyForCausalLM"], "model_type": "tiny",
       "text_config": {"model_type": "tiny_text", "num_hidden_layers": 4, "full_attention_interval": 2, "hidden_size": 16,
                       "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32, "vocab_size": 32,
                       "max_position_embeddings": 2048, "dtype": "bfloat16", "linear_num_key_heads": 2, "linear_key_head_dim": 4,
                       "linear_num_value_heads": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4}}


def write_model(root, name, shards=1):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(CFG, fh)
    tensors = {"model.language_model.embed_tokens.weight": ("BF16", (32, 16)), "lm_head.weight": ("BF16", (32, 16))}
    for i in range(4):
        p = "model.language_model.layers.%d." % i
        if i % 2 == 1:
            tensors[p + "self_attn.q_proj.weight"] = ("BF16", (16, 16))
        else:
            tensors[p + "linear_attn.in_proj_qkv.weight"] = ("BF16", (32, 16))
        for nm in ("gate_proj", "up_proj", "down_proj"):
            tensors[p + "mlp." + nm + ".weight"] = ("BF16", (32, 16))
    items = list(tensors.items())
    for s in range(shards):
        header, off = {}, 0
        for n, (dt, shape) in items[s::shards]:
            nb = DT[dt]
            for x in shape:
                nb *= x
            header[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
            off += nb
        blob = json.dumps(header).encode()
        with open(os.path.join(d, "model-%05d-of-%05d.safetensors" % (s + 1, shards)), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob)
    return d, sum(DT[dt] * (shape[0] * (shape[1] if len(shape) > 1 else 1)) for dt, shape in tensors.values())


def leaves(o, path=""):
    if isinstance(o, dict):
        if "v" in o:
            yield path, o
        else:
            for k, v in o.items():
                yield from leaves(v, path + "/" + k)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        self.model, self.total = write_model(self.root, "tiny")
        self.est = M.ModelEstimator(tree=TREE, roots=[self.root])

    def tearDown(self):
        self.tmp.cleanup()


class TestEstimator(Base):
    def test_tree_is_found_by_file_path_not_by_import(self):
        self.assertEqual(self.est.tree, TREE)
        self.assertEqual(M.find_tree(TREE), TREE)
        self.est.module()
        self.assertNotIn("sglang", sys.modules)                    # der Schätzer kommt per Dateipfad, ohne torch-Kette
        self.assertTrue(self.est._mod.__file__.endswith(M.MODULE_REL))

    def test_estimate_returns_flliper_model_1_with_sources(self):
        r = self.est.estimate({"path": self.model})
        self.assertTrue(r["ok"])
        p = r["profile"]
        self.assertEqual(p["schema"], "flliper.model/1")
        self.assertEqual(p["weights"]["total_bytes"]["v"], self.total)
        self.assertEqual(p["weights"]["total_bytes"]["src"], "Index")
        self.assertEqual(p["arch"]["layer_counts"]["v"], {"attn": 2, "gdn": 2, "mamba": 0})
        n = 0
        for path, leaf in leaves(p):
            n += 1
            self.assertIn(leaf["src"], ("config", "Index", "geschätzt", "stat"), path)
        self.assertGreater(n, 30)
        redact.guard(json.dumps(r, default=str))                 # nichts, was die Tür sperrt
        self.assertNotIn("registry_fields", r)

    def test_registry_fields_on_request(self):
        r = self.est.estimate({"path": self.model, "registry": "tiny-1"})
        f = r["registry_fields"]
        self.assertEqual((f["id"], f["arch"], f["attn"], f["d_layout"]), ("tiny-1", "dense", "full", "paged_dcp"))
        r2 = self.est.estimate({"path": self.model, "registry": True})
        self.assertEqual(r2["registry_fields"]["id"], "tiny")
        json.dumps(r2)                                            # Tupel -> Listen, alles serialisierbar

    def test_options_select_the_value(self):
        r = self.est.estimate({"path": self.model, "kv_dtype": "fp8_e4m3", "mamba_ssm_dtype": "bfloat16"})
        self.assertEqual(r["profile"]["kv"]["chosen"]["v"], "fp8_e4m3")
        self.assertEqual(r["profile"]["state"]["ssm_dtype"]["v"], "bfloat16")

    def test_path_checks(self):
        bad = [({"path": "tiny"}, "absolut"), ({"path": ""}, "fehlt"), ({}, "fehlt"), ({"path": None}, "fehlt"),
               ({"path": self.model + "\x00"}, "unzulässig"), ({"path": "/etc"}, "Modellwurzel"),
               ({"path": os.path.join(self.model, "..", "..", "etc")}, "Modellwurzel"),
               ({"path": os.path.join(self.root, "gibt-es-nicht")}, "existiert nicht"),
               ({"path": self.model, "draft_path": "/etc"}, "draft_path"),
               ({"path": self.model, "kv_dtype": "int3"}, "kv_dtype"),
               ({"path": self.model, "mamba_ssm_dtype": "f8"}, "mamba_ssm_dtype"),
               ({"path": self.model, "gguf_file": "../x.gguf"}, "gguf_file"),
               ({"path": self.model, "registry": "Groß Geschrieben"}, "Kennung")]
        for req, word in bad:
            with self.assertRaises(ValueError, msg=str(req)) as cm:
                self.est.estimate(req)
            self.assertIn(word, str(cm.exception), req)
        with self.assertRaises(ValueError):
            self.est.estimate("kein objekt")

    def test_symlink_out_of_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as outside:
            write_model(outside, "fremd")
            link = os.path.join(self.root, "link")
            os.symlink(os.path.join(outside, "fremd"), link)
            with self.assertRaises(ValueError) as cm:
                self.est.estimate({"path": link})
            self.assertIn("Modellwurzel", str(cm.exception))

    def test_cache_and_invalidation(self):
        a = self.est.estimate({"path": self.model})
        b = self.est.estimate({"path": self.model})
        self.assertFalse(a["cached"])
        self.assertTrue(b["cached"])
        shard = [f for f in os.listdir(self.model) if f.endswith(".safetensors")][0]
        os.utime(os.path.join(self.model, shard), (1, 1))
        self.assertFalse(self.est.estimate({"path": self.model})["cached"])

    def test_unreadable_model_is_a_400_text(self):
        os.makedirs(os.path.join(self.root, "leer"))
        with self.assertRaises(ValueError) as cm:
            self.est.estimate({"path": os.path.join(self.root, "leer")})
        self.assertIn("config.json", str(cm.exception))

    def test_missing_tree_is_named(self):
        e = M.ModelEstimator(tree="/gibt/es/nicht", roots=[self.root])
        e.tree = None
        with self.assertRaises(M.ModellprofilUnavailable) as cm:
            e.estimate({"path": self.model})
        self.assertIn("model_profile.py", str(cm.exception))

    def test_models_listing_is_stat_only(self):
        os.makedirs(os.path.join(self.root, "ohne-config"))
        m = self.est.models()
        self.assertEqual([x["name"] for x in m["models"]], ["tiny"])
        self.assertEqual(m["models"][0]["shards"], 1)
        self.assertEqual(m["roots"], [self.root])


class TestRoute(Base):
    def setUp(self):
        super().setUp()
        self.edition = "rig"
        app = SimpleNamespace(version="test", edition="rig", modellprofil=self.est)
        self.app = app
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def call(self, path, body=None, headers=None, method=None):
        data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        req = urllib.request.Request(self.base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                     headers=dict({"Content-Type": "application/json"} if data is not None else {}, **(headers or {})))
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_post_returns_the_profile(self):
        st, ct, body = self.call("/api/modellprofil/schaetzen", {"path": self.model, "registry": True})
        self.assertEqual(st, 200)
        self.assertEqual(ct, "application/json")
        j = json.loads(body)
        self.assertEqual(j["profile"]["schema"], "flliper.model/1")
        self.assertEqual(j["profile"]["weights"]["total_bytes"]["v"], self.total)
        self.assertEqual(j["registry_fields"]["arch"], "dense")

    def test_bad_requests_are_400_with_text(self):
        for body in ({"path": "/etc"}, {"path": "relativ"}, {}):
            st, _, raw = self.call("/api/modellprofil/schaetzen", body)
            self.assertEqual(st, 400, body)
            self.assertFalse(json.loads(raw)["ok"])
        st, _, raw = self.call("/api/modellprofil/schaetzen", b"{nicht json", method="POST")
        self.assertEqual(st, 400)
        self.assertIn("JSON", json.loads(raw)["error"])
        st, _, raw = self.call("/api/modellprofil/schaetzen", b"[1,2]", method="POST")
        self.assertEqual(st, 400)

    def test_body_limit(self):
        st, _, raw = self.call("/api/modellprofil/schaetzen", json.dumps({"path": "/" + "x" * 70000}).encode(), method="POST")
        self.assertEqual(st, 400)
        self.assertIn("zu groß", json.loads(raw)["error"])

    def test_lan_only_not_via_proxy(self):
        st, _, raw = self.call("/api/modellprofil/schaetzen", {"path": self.model}, headers={"X-Forwarded-For": "1.2.3.4"})
        self.assertEqual(st, 403)
        self.assertIn("LAN", json.loads(raw)["error"])
        st, _, _ = self.call("/api/modellprofil/modelle", headers={"X-Forwarded-Prefix": "/rigdash"})
        self.assertEqual(st, 403)

    def test_not_in_the_release_edition(self):
        self.app.edition = "release"
        self.assertEqual(self.call("/api/modellprofil/schaetzen", {"path": self.model})[0], 404)
        self.assertEqual(self.call("/api/modellprofil/modelle")[0], 404)
        self.assertEqual(self.call("/modellprofil.js")[0], 404)

    def test_unknown_post_path_is_404(self):
        self.assertEqual(self.call("/api/modellprofil/anderes", {"x": 1})[0], 404)
        self.assertEqual(self.call("/api/live", {"x": 1})[0], 404)

    def test_list_and_module_are_served(self):
        st, _, raw = self.call("/api/modellprofil/modelle")
        self.assertEqual(st, 200)
        self.assertEqual([m["name"] for m in json.loads(raw)["models"]], ["tiny"])
        st, ct, body = self.call("/modellprofil.js")
        self.assertEqual(st, 200)
        self.assertIn("javascript", ct)
        self.assertIn(b"ModellProfil", body)

    def test_no_tree_is_503(self):
        self.est.tree = None
        self.est._mod = None
        st, _, raw = self.call("/api/modellprofil/schaetzen", {"path": self.model})
        self.assertEqual(st, 503)
        self.assertIn("model_profile.py", json.loads(raw)["error"])


NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)


@unittest.skipUnless(NODE, "node fehlt")
class TestJsModule(Base):
    def run_js(self, script):
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_rows_carry_the_source_and_html_is_escaped(self):
        prof = self.est.estimate({"path": self.model})["profile"]
        prof["format"]["v"] = "<b>x</b>"
        js = ("const M = require(%r);\nconst p = %s;\nconst rows = M.zeilen(p);\n"
              "const out = {n: rows.length, groups: [...new Set(rows.map(r => r.gruppe))], allsrc: rows.every(r => ['config','Index','geschätzt','stat'].includes(r.src)),\n"
              "  total: rows.find(r => r.label === 'Gewichte gesamt'), html: M.tabelle(p), bytes: [M.bytes(512), M.bytes(1536), M.bytes(3*1048576), M.bytes(5*1073741824)]};\n"
              "console.log(JSON.stringify(out));\n" % (JS, json.dumps(prof)))
        o = json.loads(self.run_js(js))
        self.assertGreater(o["n"], 10)
        self.assertTrue(o["allsrc"])
        self.assertIn("Gewichte", o["groups"])
        self.assertEqual(o["total"]["src"], "Index")
        self.assertEqual(o["total"]["roh"], self.total)
        self.assertNotIn("<b>x</b>", o["html"])
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", o["html"])
        self.assertEqual(o["bytes"][0], "512 B")

    def test_schaetzen_posts_json_and_surfaces_errors(self):
        js = ("const M = require(%r);\nconst calls = [];\n"
              "globalThis.fetch = async (url, o) => { calls.push([url, o && o.method, o && o.body]); "
              "if (url.includes('fehler')) return {ok: false, status: 400, text: async () => JSON.stringify({ok: false, error: 'liegt nicht unter einer Modellwurzel'})}; "
              "return {ok: true, status: 200, text: async () => JSON.stringify({ok: true, profile: {schema: 'flliper.model/1'}})}; };\n"
              "(async () => { const r = await M.schaetzen('/m/x', {kv_dtype: 'fp8_e4m3', registry: true}); let err = null;\n"
              "  try { globalThis.fetch = async () => ({ok: false, status: 400, text: async () => JSON.stringify({ok: false, error: 'liegt nicht unter einer Modellwurzel'})}); await M.schaetzen('/etc'); } catch (e) { err = e.message; }\n"
              "  console.log(JSON.stringify({calls, schema: r.profile.schema, err})); })();\n" % JS)
        o = json.loads(self.run_js(js))
        self.assertEqual(o["calls"][0][0], "api/modellprofil/schaetzen")
        self.assertEqual(o["calls"][0][1], "POST")
        self.assertEqual(json.loads(o["calls"][0][2]), {"path": "/m/x", "kv_dtype": "fp8_e4m3", "registry": True})
        self.assertEqual(o["schema"], "flliper.model/1")
        self.assertIn("Modellwurzel", o["err"])


if __name__ == "__main__":
    unittest.main()
