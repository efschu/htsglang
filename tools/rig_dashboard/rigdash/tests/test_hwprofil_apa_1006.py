"""AP-A (Profil-Planer 06.10.): Hardwareprofil gespeichert, Katalog vorbelegt, Issue-Text.  Kein Rig, keine GPU, kein gpuq.

Geprüft wird: (1) der Katalog hat genau drei vorbelegte Karten (RTX 5090, RTX 3080 20 GB, RTX 3090), behält alle übrigen und
nennt die Herkunft je Eintrag und Feld; (2) das Profil wird beim ersten Aufruf gespeichert, danach nur verglichen, "Neu
erfassen" ersetzt es, in Rig UND Release, ohne gpuq; (3) der Issue-Text hat alle Blöcke und lässt weder Geheimnisse noch
Hostpfade durch.  Der echte Profilbaum (HWPROFIL_TREE) liefert das Modul; NVML wird eingespielt.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import hwprofil, redact, server  # noqa: E402
from rigdash import kartenplan_catalog as CAT  # noqa: E402

STATIC = server.STATIC
PRESETS = {"rtx5090-32", "rtx3080-20", "rtx3090-24"}


def _node():
    for c in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node", shutil.which("bun")):
        if c and os.path.exists(c):
            return c
    return None


def _fixture():
    """Die Fixture des Baum-Tests (``test/registered/unit/rigmon/_hwprofile_fixture_1006.py``) oder ``None``."""
    tree = os.environ.get("HWPROFIL_TREE")
    if not tree or not os.path.isfile(os.path.join(tree, hwprofil.MODULE_REL)):
        return None
    path = os.path.join(os.path.dirname(tree.rstrip("/")), "test", "registered", "unit", "rigmon", "_hwprofile_fixture_1006.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("hwprofile_fixture_1006", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCatalog(unittest.TestCase):
    def test_exactly_three_cards_are_preset(self):
        self.assertEqual({e["id"] for e in CAT.presets()}, PRESETS)
        self.assertEqual({e["id"] for e in CAT.catalog_public() if e["preset"]}, PRESETS)

    def test_the_other_cards_stay_in_the_catalog(self):
        ids = {e["id"] for e in CAT.CATALOG}
        self.assertEqual(len(CAT.CATALOG), 24)
        for keep in ("rtx5080-16", "rtx4090-24", "rtx3090ti-24", "rtx3080-10", "rtx3070-8", "rtx2080ti-22"):
            self.assertIn(keep, ids)
        for e in CAT.CATALOG:
            if e["id"] not in PRESETS:
                self.assertFalse(e["preset"], e["id"])

    def test_every_card_names_its_origin(self):
        for e in CAT.catalog_public(include_disabled=True):
            self.assertIn(e["origin"], CAT.ORIGIN_LABELS, e["id"])
            self.assertEqual(e["origin_label"], CAT.ORIGIN_LABELS[e["origin"]])
            self.assertEqual(set(e["origin_fields"]), {"vram", "mem_bw", "pcie"})
            if e["measured_on_rig"]:
                self.assertEqual(e["origin"], "measured_on_rig")
            else:
                self.assertEqual(e["origin"], "Datenblatt", e["id"])

    def test_rig_cards_are_measured_but_the_3080_20g_bandwidth_is_borrowed(self):
        by = {e["id"]: e for e in CAT.CATALOG}
        self.assertEqual(by["rtx5090-32"]["origin"], "measured_on_rig")
        self.assertEqual(by["rtx5090-32"]["origin_fields"]["mem_bw"], "Datenblatt")
        self.assertEqual(by["rtx3080-20"]["origin_fields"], {"vram": "measured_on_rig", "mem_bw": "borrowed-unbelegt", "pcie": "Datenblatt"})
        self.assertEqual(by["rtx3090-24"]["origin"], "Datenblatt")           # preset, aber am Rig nicht gemessen

    def test_match_nvml(self):
        self.assertEqual(CAT.match_nvml("NVIDIA GeForce RTX 5090", 32607, (12, 0))["id"], "rtx5090-32")
        self.assertEqual(CAT.match_nvml("NVIDIA GeForce RTX 3080", 20480, [8, 6])["id"], "rtx3080-20")
        self.assertEqual(CAT.match_nvml("NVIDIA GeForce RTX 3080", 10200, [8, 6])["id"], "rtx3080-10")   # fremde Karte: einige MiB unter Nennwert
        self.assertIsNone(CAT.match_nvml("NVIDIA GeForce RTX 3080", 16384, [8, 6]))                      # keine Variante dieser Größe: nie geraten
        self.assertIsNone(CAT.match_nvml("NVIDIA GeForce RTX 3080", 20480, [8, 9]))                      # andere cc
        self.assertIsNone(CAT.match_nvml("NVIDIA RTX A6000", 49140, [8, 6]))

    def test_datasheet_of(self):
        d = CAT.datasheet_of({"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480, "cc": [8, 6]})
        self.assertEqual(d["mem_bw_gbs"], 760)
        self.assertTrue(d["bw_note"].startswith("BORROWED, unverified"))
        self.assertTrue(d["catalog"]["preset"])
        d = CAT.datasheet_of({"name": "NVIDIA GeForce RTX 4090", "total_mib": 24564, "cc": [8, 9]})
        self.assertEqual((d["mem_bw_gbs"], d["catalog"]["preset"], d["catalog"]["origin"]), (1008, False, "Datenblatt"))
        self.assertNotIn("GEBORGT", d["bw_note"])
        self.assertEqual(CAT.datasheet_of({"name": "unbekannt", "total_mib": 1, "cc": [1, 1]}), {})


class TestRedact(unittest.TestCase):
    def test_paths_and_secrets(self):
        t = redact.text_for_issue("Image /spinning/gpu-arb/x.json und /var/lib/flliper/hardware.json\nADMIN-KEY abc\ntoken=abcdefgh12345678 ok /api/hwprofil 1/2")
        self.assertNotIn("/spinning", t)
        self.assertNotIn("/var/lib", t)
        self.assertNotIn("ADMIN-KEY", t)
        self.assertNotIn("abcdefgh12345678", t)
        self.assertIn("/api/hwprofil", t)         # eine URL-Route ist kein Hostpfad
        self.assertIn("1/2", t)


def _gq_down(method, url, body=None, headers=None, timeout=10.0):
    raise hwprofil.GpuqUnavailable("gpuq nicht erreichbar (Test)")


class _App:
    version = "test-1006"

    def __init__(self, hw, edition="rig"):
        self.hwprofil, self.edition = hw, edition


def _serve(app):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _req(srv, method, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (srv.server_address[1], path), data=data, method=method,
                               headers=dict({"content-type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class RealBase(unittest.TestCase):
    """Das echte ``hardware_profile.py`` des Baums, NVML eingespielt, Speicherort in einem Wegwerfordner."""

    def setUp(self):
        self.fx = _fixture()
        if self.fx is None:
            self.skipTest("HWPROFIL_TREE zeigt auf keinen Baum mit hardware_profile.py und test/.../_hwprofile_fixture_1006.py")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cache = os.path.join(self.tmp, "cache")
        os.makedirs(self.cache)
        self.fx.write_probe(self.cache)
        self.path = os.path.join(self.tmp, "state", "hardware.json")
        self.now = [self.fx.NOW]
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for k in (hwprofil.PERSIST_ENV, "SGLANG_IMAGE_TAG"):
            os.environ.pop(k, None)

    def make(self, nvml=None, persist=True, edition="rig", **kw):
        hw = hwprofil.HwProfil(http=_gq_down, tree=os.environ["HWPROFIL_TREE"], cache_dir=self.cache,
                               persist_path=self.path if persist else None, clock=lambda: self.now[0], synchronous=True,
                               edition=edition, versions=lambda: {"rigdash": "test-1006", "edition": edition}, **kw)
        self.set_nvml(hw, nvml or self.fx.nvml())
        return hw

    def set_nvml(self, hw, nvml):
        hw._module().read_nvml = lambda: nvml
        hw._view_cache = None


class TestPersistedProfile(RealBase):
    def test_first_get_writes_the_file_once_and_reports_it(self):
        hw = self.make()
        self.assertFalse(os.path.exists(self.path))
        out = hw.get()
        self.assertTrue(out["ok"])
        self.assertEqual(out["persist"]["state"], "erst_erfasst")
        self.assertTrue(out["persist"]["enabled"])
        self.assertTrue(os.path.isfile(self.path))
        self.assertEqual(json.load(open(self.path))["schema"], "flliper.hardware/1")
        self.assertNotIn(self.tmp, json.dumps(out))              # die Antwort nennt keinen Pfad des Rechners
        self.now[0] += 60
        hw._view_cache = None
        self.assertEqual(hw.get()["persist"]["state"], "vorhanden")

    def test_profile_is_merged_with_the_data_sheet(self):
        out = self.make().get()
        by = {c["uuid"]: c for c in out["profile"]["cards"]}
        c5090, c3080 = by[self.fx.U1], by[self.fx.U0]
        self.assertEqual(c5090["catalog"]["id"], "rtx5090-32")
        self.assertEqual(c5090["mem_gbs"]["nominal"], {"v": 1792, "src": "Datenblatt", "unit": "GB/s", "note": mock.ANY})
        self.assertIn("rtx5090-32", c5090["mem_gbs"]["nominal"]["note"])
        self.assertEqual(c3080["catalog"]["id"], "rtx3080-20")
        self.assertEqual(c3080["catalog"]["origin_fields"]["mem_bw"], "borrowed-unbelegt")
        self.assertTrue(c3080["mem_gbs"]["nominal"]["note"].startswith("BORROWED, unverified"))
        self.assertEqual(out["problems"], [])

    def test_without_a_storage_path_nothing_is_written(self):
        hw = self.make(persist=False)
        out = hw.get()
        self.assertEqual(out["persist"], {"enabled": False})
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(hw.recapture()["ok"])

    def test_the_environment_names_the_file(self):
        os.environ[hwprofil.PERSIST_ENV] = self.path
        hw = hwprofil.HwProfil(http=_gq_down, tree=os.environ["HWPROFIL_TREE"], cache_dir=self.cache, clock=lambda: self.now[0])
        self.assertEqual(hw.persist_path, self.path)
        self.assertEqual(hwprofil.default_persist_path({}), "/var/lib/flliper/hardware.json")
        self.assertEqual(hwprofil.default_persist_path({hwprofil.PERSIST_ENV: "/x/y.json"}), "/x/y.json")

    def test_a_changed_inventory_is_shown_and_the_file_is_kept(self):
        hw = self.make()
        hw.get()
        raw = open(self.path).read()
        self.set_nvml(hw, self.fx.nvml(n=2, driver="600.1"))
        out = hw.get()
        self.assertEqual(out["persist"]["state"], "abweichend")
        self.assertTrue(any("Driver" in c for c in out["persist"]["drift"]["changes"]))
        self.assertEqual(open(self.path).read(), raw)

    def test_recapture_replaces_the_file_without_gpuq(self):
        hw = self.make()
        hw.get()
        self.now[0] += 120
        self.set_nvml(hw, self.fx.nvml(n=2))
        res = hw.recapture()
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["persist"]["state"], "neu_erfasst")
        doc = json.load(open(self.path))
        self.assertEqual(len(doc["cards"]), 2)
        self.assertEqual(doc["capture"], {"at": self.fx.NOW + 120, "reason": "Neu erfassen"})
        self.assertEqual(hw.get()["persist"]["state"], "vorhanden")

    def test_recapture_with_silent_nvml_keeps_the_file(self):
        hw = self.make()
        hw.get()
        raw = open(self.path).read()
        self.set_nvml(hw, ([], None, ["pynvml nicht lesbar (Test)"]))
        res = hw.recapture()
        self.assertFalse(res["ok"])
        self.assertIn("nothing saved", res["error"])
        self.assertEqual(open(self.path).read(), raw)
        out = hw.get()                         # und die Seite zeigt das gespeicherte Profil
        self.assertEqual(out["persist"]["state"], "nur_gespeichert")
        self.assertTrue(out["persist"]["from_persisted"])
        self.assertEqual(len(out["profile"]["cards"]), 3)

    def test_a_successful_measurement_refreshes_the_stored_profile(self):
        hw = self.make()
        hw.get()
        mod = hw._module()
        self.now[0] += 500
        hw.runner = lambda cmd, env, timeout: (0, json.dumps({"cards": [], "pairs": []}), "")
        hw._run(mod, [0], 30.0)
        self.assertEqual(hw._job["state"], "ok")
        self.assertEqual(json.load(open(self.path))["capture"], {"at": self.fx.NOW + 500, "reason": "Neu erfassen"})

    def test_a_failed_write_is_shown_not_raised(self):
        blocker = os.path.join(self.tmp, "datei")
        open(blocker, "w").close()
        self.path = os.path.join(blocker, "hardware.json")
        out = self.make().get()
        self.assertTrue(out["ok"])
        self.assertEqual(out["persist"]["state"], "nicht_schreibbar")
        self.assertEqual(len(out["profile"]["cards"]), 3)


class TestIssueText(RealBase):
    def test_all_blocks_and_values(self):
        os.environ["SGLANG_IMAGE_TAG"] = "flliper:0.1.0-cu130"
        hw = self.make()
        txt = hw.issue()["text"]
        for needle in ("## Hardware profile (`flliper.hardware/1`)", "| Driver | 595.58 |", "| CUDA / torch (measuring process) | 13.0 / 2.9 |",
                       "| Image | flliper:0.1.0-cu130 |", "| Dashboard | test-1006 |", "| Cards | 3 |",
                       "### Cards (NVML identity)", "### Memory and compute", "### Catalog card and origin of the datasheet values",
                       "### Card to card (measured)", "### Origin of the values"):
            self.assertIn(needle, txt)
        for c in self.fx.nvml()[0]:                               # NVML-Identität: Index, Name, UUID, PCI-Bus
            self.assertNotIn(c["uuid"], txt)                      # Nacharbeit 06.10.: die UUID steht nie im Issue-Text (immer "<redacted>")
            self.assertIn(c["pci_bus_id"], txt)
        self.assertIn("| <redacted> |", txt)
        self.assertIn("RTX 5090", txt)
        self.assertIn("| 170 (meas.) |", txt)                      # SM-Zahl, gemessen
        self.assertIn("32607 MiB (NVML)", txt)                    # Größe
        self.assertIn("2100 MHz (NVML)", txt)                     # Takt
        self.assertIn("1792 GB/s (datasheet)", txt)                # Nennbandbreite aus dem Katalog
        self.assertIn("210.0 TFLOPS (meas.)", txt)                 # Messrate des 5090
        self.assertIn("RTX 3080 20 GB (modded)", txt)
        self.assertIn("borrowed-unbelegt", txt)
        self.assertIn("not measured", txt)                      # fp8 der 3080

    def test_unknown_versions_say_unbelegt_and_state_whether_stored(self):
        # ein Baum ohne git und ohne Revisions-ENV (der echte Baum der Testlaeufe ist ein git-Baum und nennt seine Revision, Review 1006 Befund 2)
        hw = self.make()
        with mock.patch.dict(os.environ), mock.patch.object(hwprofil, "_git_head", return_value=None):
            for k in ("HTSGLANG_REVISION", "HTSGLANG_REVISION_27B", "HTSGLANG_REVISION_NF", "SGLANG_BUILD_COMMIT", "SGLANG_IMAGE_TAG", "STAND"):
                os.environ.pop(k, None)
            txt = hw.issue()["text"]
        self.assertIn("| Image | unverified (SGLANG_IMAGE_TAG not set) |", txt)
        self.assertIn("| Tree | unverified |", txt)
        self.assertIn("(saved)", txt)
        txt2 = self.make(persist=False).issue()["text"]
        self.assertIn("(live view, not saved)", txt2)

    def test_the_tree_revision_comes_only_from_a_staged_release_path(self):
        d = hwprofil.issue_text({"cards": []}, versions={"tree": "/opt/rigdash/kartenplan/profil/releases/0123abcd4567/python"})
        self.assertIn("| Tree | 0123abcd4567 |", d)
        self.assertIn("| Tree | unverified |", hwprofil.issue_text({"cards": []}, versions={"tree": "/opt/rigdash/kartenplan/current/python"}))

    def test_secrets_and_host_paths_are_removed(self):
        evil = ["/spinning/geheim/pfad broke", "token=abcdefgh12345678 broke", "ADMIN-KEY minted -> /root/x.adminkey"]
        hw = self.make(nvml=(self.fx.nvml()[0], "595.58", evil))
        txt = hw.issue()["text"]
        for bad in ("/spinning", "geheim", "abcdefgh12345678", "ADMIN-KEY", ".adminkey", "/root", self.tmp, "hardware.json"):
            self.assertNotIn(bad, txt)
        self.assertIn("<path redacted>", txt)

    def test_a_card_name_cannot_break_the_table(self):
        cards, drv, iss = self.fx.nvml()
        cards[0] = dict(cards[0], name="NVIDIA GeForce RTX 3080 | evil\nline")
        txt = self.make(nvml=(cards, drv, iss)).issue()["text"]
        self.assertNotIn("evil\nline", txt)
        self.assertNotIn("3080 | evil", txt)

    def test_no_cards_is_an_honest_empty_profile(self):
        txt = hwprofil.issue_text({"schema": "flliper.hardware/1", "cards": [], "driver": None}, versions={})
        self.assertIn("| Cards | 0 |", txt)
        self.assertIn("| Driver | unverified |", txt)


class TestRoutes(RealBase):
    def test_issue_and_recapture_routes_in_rig_and_release(self):
        for edition in ("rig", "release"):
            hw = self.make(edition=edition)
            srv = _serve(_App(hw, edition))
            self.addCleanup(srv.shutdown)
            st, body = _req(srv, "GET", "/api/hwprofil/issue")
            self.assertEqual(st, 200, edition)
            j = json.loads(body)
            self.assertEqual((j["ok"], j["format"]), (True, "markdown"))
            self.assertIn("## Hardware profile", j["text"])
            st, body = _req(srv, "POST", "/api/hwprofil/recapture", {})
            self.assertEqual(st, 200, (edition, body))
            self.assertEqual(json.loads(body)["persist"]["state"], "neu_erfasst")
            self.assertIn("persist", json.loads(_req(srv, "GET", "/api/hwprofil")[1]))

    def test_lan_only(self):
        srv = _serve(_App(self.make()))
        self.addCleanup(srv.shutdown)
        for method, path in (("GET", "/api/hwprofil/issue"), ("POST", "/api/hwprofil/recapture")):
            self.assertEqual(_req(srv, method, path, {} if method == "POST" else None, {"X-Forwarded-For": "1.2.3.4"})[0], 403, path)

    def test_recapture_without_a_storage_path_is_a_conflict_with_the_reason(self):
        srv = _serve(_App(self.make(persist=False)))
        self.addCleanup(srv.shutdown)
        st, body = _req(srv, "POST", "/api/hwprofil/recapture", {})
        self.assertEqual(st, 409)
        self.assertIn(hwprofil.PERSIST_ENV, json.loads(body)["error"])

    def test_measure_is_still_blocked_in_release(self):
        srv = _serve(_App(self.make(edition="release"), "release"))
        self.addCleanup(srv.shutdown)
        self.assertEqual(_req(srv, "POST", "/api/hwprofil/measure", {"cards": [1]})[0], 403)


class TestOldModuleWithoutPersistence(unittest.TestCase):
    """Ein älterer Baum (Modul ohne ``capture``/``datasheet``) darf die Seite nicht brechen."""

    STUB = ('def build(cache_dir=None, **kw):\n return {"schema": "flliper.hardware/1", "id": "sha256:" + "0" * 64, "cards": [], "links": [], "measure_needed": True}\n'
            'def validate(doc):\n return []\n')

    def test_get_works_and_persist_is_off(self):
        with tempfile.TemporaryDirectory() as t:
            d = os.path.join(t, "python", "sglang", "srt", "rigmon")
            os.makedirs(d)
            open(os.path.join(d, "hardware_profile.py"), "w").write(self.STUB)
            hw = hwprofil.HwProfil(http=_gq_down, tree=os.path.join(t, "python"), persist_path=os.path.join(t, "hw.json"))
            out = hw.get()
            self.assertTrue(out["ok"])
            self.assertEqual(out["persist"], {"enabled": False})
            self.assertFalse(os.path.exists(os.path.join(t, "hw.json")))
            self.assertFalse(hw.recapture()["ok"])


class TestScripts(unittest.TestCase):
    DOC = {
        "ok": True,
        "profile": {
            "schema": "flliper.hardware/1", "id": "sha256:" + "ab" * 32, "driver": "595.58",
            "formats": [{"key": "bf16", "unit": "TFLOPS", "label": "bf16"}],
            "cards": [
                {"ord": 0, "nvml_index": 1, "name": "NVIDIA GeForce RTX 5090", "class_key": "RTX5090", "cc": [12, 0],
                 "sm_count": {"v": 170, "src": "Datenblatt", "note": "Datenblatt-Katalog weg2/hw_sim.py"},
                 "l2_mib": {"v": None, "src": "nicht gemessen", "note": "x"},
                 "vram_total_mib": {"v": 32607, "src": "NVML", "unit": "MiB"}, "bar1_total_mib": {"v": 32768, "src": "NVML"},
                 "mem_gbs": {"read": {"v": None, "src": "nicht gemessen", "note": "n"}, "copy": {"v": None, "src": "nicht gemessen", "note": "n"},
                             "gemv": {"v": None, "src": "nicht gemessen", "note": "n"},
                             "nameplate": {"v": 1792.0, "src": "Datenblatt", "unit": "GB/s"},
                             "nominal": {"v": 1792, "src": "Datenblatt", "unit": "GB/s", "note": "Katalog"}},
                 "compute": {"bf16": {"v": None, "src": "nicht gemessen", "note": "n"}},
                 "h2d": {"gbs": {"v": None, "src": "nicht gemessen", "note": "n"}, "lat_us": {"v": None, "src": "nicht gemessen", "note": "n"}},
                 "d2h": {"gbs": {"v": None, "src": "nicht gemessen", "note": "n"}, "lat_us": {"v": None, "src": "nicht gemessen", "note": "n"}},
                 "pcie": {k: {"v": 5, "src": "NVML"} for k in ("max_gen", "max_width", "cur_gen", "cur_width")},
                 "power": {"limit_w": {"v": 575, "src": "NVML", "unit": "W"}},
                 "clocks": {"sm_max_mhz": {"v": 3090, "src": "NVML", "unit": "MHz"}, "mem_max_mhz": {"v": 14001, "src": "NVML", "unit": "MHz"}},
                 "mem_bus_bits": {"v": 512, "src": "NVML", "unit": "bit"},
                 "catalog": {"id": "rtx5090-32", "label": "RTX 5090 32 GB", "preset": True, "origin": "measured_on_rig",
                             "origin_label": "am Rig gemessen", "origin_fields": {"vram": "measured_on_rig", "mem_bw": "borrowed-unbelegt", "pcie": "Datenblatt"}},
                 "state": None, "probed_at": None}],
            "links": [], "bar1": {"measured": True, "note": ""}, "sources": {"card_probe": [], "stage0": [], "nvml": {"issues": []}},
            "measure_needed": True},
        "window": None, "job": {"state": "idle"}, "problems": [],
        "persist": {"enabled": True, "state": "abweichend", "label": "saved, DEVIATES", "captured_at": 1000.0, "reason": "first start",
                    "id": "sha256:" + "cd" * 32, "drift": {"same": False, "changes": ["Driver was 1, now 2"]}, "error": None},
    }

    def _render(self, doc):
        node = _node()
        if not node:
            self.skipTest("weder node noch bun vorhanden")
        script = ("const H=require(process.argv[1]);const d=JSON.parse(process.argv[2]);process.stdout.write(H.render(d,{now:1100}));")
        out = subprocess.run([node, "-e", script, os.path.join(STATIC, "hwprofil.js"), json.dumps(doc)], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_render_shows_catalog_origin_nominal_bandwidth_clocks_and_the_stored_state(self):
        h = self._render(self.DOC)
        for needle in ("Catalog card", "RTX 5090 32 GB", "measured_on_rig", "bandwidth borrowed", "Nominal bandwidth (catalog)", "SM clock max.",
                       "Memory bus width", "Stored profile", "DEVIATES", "Driver was 1, now 2", "datash."):
            self.assertIn(needle, h)
        self.assertNotIn("/var/lib", h)

    def test_render_of_an_old_answer_has_no_new_rows_and_says_not_stored(self):
        old = json.loads(json.dumps(self.DOC))
        c = old["profile"]["cards"][0]
        for k in ("catalog", "clocks", "mem_bus_bits"):
            c.pop(k)
        c["mem_gbs"].pop("nominal")
        old.pop("persist")
        h = self._render(old)
        self.assertNotIn("Catalog card", h)
        self.assertNotIn("Nominal bandwidth (catalog)", h)
        self.assertIn("not stored", h)

    def test_scripts_wire_the_buttons_and_the_collapsed_catalog(self):
        js = open(os.path.join(STATIC, "hwprofil.js"), encoding="utf-8").read()
        for needle in ('data-act="recapture"', 'data-act="issue"', 'data-act="copy"', "/recapture", "/issue", "Recapture", "Issue text (hardware profile)"):
            self.assertIn(needle, js)
        self.assertNotIn("eval(", js)
        self.assertNotIn("setInterval", js)
        pj = open(os.path.join(STATIC, "profil.js"), encoding="utf-8").read()
        for needle in ("more cards (datasheet, without measured rates)", 'data-act="cadd-id"', "c.preset", 'data-fold="cmore"'):
            self.assertIn(needle, pj)


if __name__ == "__main__":
    unittest.main()
