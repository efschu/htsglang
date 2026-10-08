"""Auftrag 950 (Profil-Editor S2): die Routen /api/hwprofil.  Kein Rig, keine GPU, kein gpuq, kein Kindprozess.

Geprüft wird, was ohne Karte stimmen muss: dass die Messung NUR in einem gebuchten Fenster läuft (Eigentümer
profil-editor, nur die gewählten Karten, 15 min seit Auftrag 1006, kein not_before), dass ein wartendes Fenster nur den Status
zurückgibt und nichts misst, dass das Fenster nach der Messung SOFORT zurückgegeben wird (auch nach Fehler), dass
eine belegte Karte nicht gemessen wird, und dass das Token das Haus nie verlässt.  gpuq ist ein Stub (Funktion
``http``), das Profilmodul eine kleine Datei im Baum (die echte liegt auf der 27B-Linie und hat ihre eigenen Tests).
"""

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from http.server import ThreadingHTTPServer  # noqa: E402

from rigdash import hwprofil, server  # noqa: E402

STATIC = server.STATIC

STUB_MODULE = '''
CALLS = []
RESULT = {"ok": True, "rc": 0, "seconds": 61.5, "warnings": [], "profile": {"cards": []}, "stderr_tail": ""}
RAISE = None
NVML = [{"nvml_index": i, "uuid": "GPU-%d" % i, "name": "NVIDIA GeForce RTX %d" % (3080 + i)} for i in range(3)]


def build(cache_dir=None, **kw):
    return {"schema": "flliper.hardware/1", "id": "sha256:" + "0" * 64,
            "cards": [{"ord": i, "nvml_index": c["nvml_index"], "name": c["name"]} for i, c in enumerate(NVML)],
            "links": [], "measure_needed": True}


def validate(doc):
    return []


def read_nvml():
    return list(NVML), "595", []


def run_measurement(cards, python=None, prefix=(), timeout_s=540.0, pythonpath=None, runner=None, **kw):
    CALLS.append({"cards": list(cards), "python": python, "prefix": list(prefix), "timeout_s": timeout_s,
                  "pythonpath": pythonpath})
    if RAISE:
        raise RAISE
    return dict(RESULT)


def duration_line(res, cards):
    return "HWPROFIL-MESSUNG stub"
'''


class FakeGpuq:
    """Der Fensterplan als Funktion ``http(method, url, body, headers)``."""

    def __init__(self, state="running", seconds_left=600.0, used=None, down=False):
        self.state, self.seconds_left, self.down = state, seconds_left, down
        self.used = used or {}
        self.log = []
        self.bookings = {}
        self.n = 0

    def cards(self):
        return [{"index": i, "name": "NVIDIA GeForce RTX %d" % (3080 + i), "short_name": "c%d" % i,
                 "total_mib": 20480, "used_mib": self.used.get(i, 1), "free_mib": 20000, "busy": False} for i in range(3)]

    def __call__(self, method, url, body=None, headers=None, timeout=10.0):
        if self.down:
            raise hwprofil.GpuqUnavailable("gpuq nicht erreichbar (stub)")
        path = url.split("8770", 1)[1]
        self.log.append((method, path, body, dict(headers or {})))
        if method == "GET" and path == "/api/v1/cards":
            return 200, self.cards()
        if method == "POST" and path == "/api/v1/bookings":
            self.n += 1
            bid = "bk%d" % self.n
            b = {"id": bid, "token": "TOKEN-%d" % self.n, "state": self.state, "cards": body["cards"],
                 "start": {"local": "2026-10-03T21:00:00+02:00"}, "end": {"local": "2026-10-03T21:10:00+02:00"},
                 "seconds_left": self.seconds_left, "note": "karte 1: belegt" if self.state == "pending" else "",
                 "plan_revision": 7}
            self.bookings[bid] = b
            return 200, dict(b)
        if path.startswith("/api/v1/bookings/"):
            bid = path.split("/")[4].split("?")[0]
            b = self.bookings.get(bid)
            if method == "GET":
                return (200, {k: v for k, v in b.items() if k != "token"}) if b else (404, {"detail": "unbekannt"})
            if method == "DELETE":
                if b is None:
                    return 404, {"detail": "unbekannt"}
                b["state"] = "released"
                return 200, {"id": bid, "state": "released"}
        return 404, {"detail": "?"}

    def calls(self, method, prefix):
        return [c for c in self.log if c[0] == method and c[1].startswith(prefix)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = os.path.join(self.tmp, "python")
        d = os.path.join(self.tree, "flliper", "srt", "rigmon")
        os.makedirs(d)
        with open(os.path.join(d, "hardware_profile.py"), "w") as fh:
            fh.write(STUB_MODULE)
        self.gq = FakeGpuq()

    def make(self, gq=None, **kw):
        kw.setdefault("tree", self.tree)
        kw.setdefault("synchronous", True)
        return hwprofil.HwProfil(http=gq or self.gq, **kw)

    def mod(self, hw):
        m = hw._module()
        del m.CALLS[:]
        m.RAISE = None
        return m


class TestMeasureOnlyInABookedWindow(Base):
    def test_books_exactly_the_chosen_cards_for_fifteen_minutes_as_profil_editor(self):
        hw = self.make()
        m = self.mod(hw)
        out = hw.measure({"cards": [2, 1]})
        post = self.gq.calls("POST", "/api/v1/bookings")
        self.assertEqual(len(post), 1)
        body = post[0][2]
        self.assertEqual(body["owner"], "profil-editor")
        self.assertEqual(body["cards"], [1, 2])
        self.assertEqual(body["duration"], "15m")  # Auftrag 1006: BAR1-Schritt + kalte JIT-Übersetzung
        self.assertNotIn("not_before", body)
        self.assertNotIn("mib", body)  # exklusiv: eine Ratenmessung neben fremder Last misst die Last mit
        self.assertEqual(out["action"], "messung_gestartet")
        self.assertEqual(len(m.CALLS), 1)
        self.assertEqual(m.CALLS[0]["cards"], [1, 2])

    def test_the_window_goes_back_at_once_after_the_run_with_the_token(self):
        hw = self.make()
        self.mod(hw)
        hw.measure({"cards": [0]})
        dele = self.gq.calls("DELETE", "/api/v1/bookings/bk1")
        self.assertEqual(len(dele), 1)
        self.assertEqual(dele[0][3].get("X-Gpuq-Token"), "TOKEN-1")
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")
        self.assertEqual(hw._job["state"], "ok")
        self.assertIsNone(hw._booking)
        # erst gemessen, dann zurückgegeben (nicht umgekehrt)
        order = [c[0] for c in self.gq.log if c[1].startswith("/api/v1/bookings")]
        self.assertLess(order.index("POST"), order.index("DELETE"))

    def test_the_window_goes_back_after_a_failed_run_and_after_an_exception(self):
        hw = self.make()
        m = self.mod(hw)
        m.RESULT = dict(m.RESULT, ok=False, rc=1, stderr_tail="CUDA OOM")
        hw.measure({"cards": [0]})
        self.assertEqual(hw._job["state"], "error")
        self.assertIn("CUDA OOM", hw._job["error"])
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")
        m.RAISE = RuntimeError("Kindprozess weg")
        hw.measure({"cards": [0]})
        self.assertEqual(hw._job["state"], "error")
        self.assertIn("Kindprozess weg", hw._job["error"])
        self.assertEqual(self.gq.bookings["bk2"]["state"], "released")

    def test_a_pending_window_measures_nothing_and_stays_booked(self):
        self.gq.state = "pending"
        hw = self.make()
        m = self.mod(hw)
        out = hw.measure({"cards": [1]})
        self.assertEqual(out["action"], "wartet")
        self.assertEqual(out["window"]["state"], "pending")
        self.assertEqual(out["window"]["id"], "bk1")
        self.assertEqual(m.CALLS, [])
        self.assertEqual(self.gq.calls("DELETE", "/"), [])
        self.assertIsNotNone(hw._booking)

    def test_pressing_again_takes_up_the_waiting_window_instead_of_booking_a_second(self):
        self.gq.state = "pending"
        hw = self.make()
        m = self.mod(hw)
        hw.measure({"cards": [1]})
        self.gq.bookings["bk1"]["state"] = "running"
        out = hw.measure({"cards": [1]})
        self.assertEqual(out["action"], "messung_gestartet")
        self.assertEqual(len(self.gq.calls("POST", "/api/v1/bookings")), 1)
        self.assertEqual(len(m.CALLS), 1)
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")

    def test_other_cards_release_the_old_window_and_book_a_new_one(self):
        self.gq.state = "pending"
        hw = self.make()
        self.mod(hw)
        hw.measure({"cards": [1]})
        hw.measure({"cards": [0, 2]})
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")
        self.assertEqual(self.gq.bookings["bk2"]["cards"], [0, 2])

    def test_unplannable_is_refused_with_the_reason_and_nothing_stays_booked(self):
        self.gq.state = "unplannable"
        hw = self.make()
        m = self.mod(hw)
        out = hw.measure({"cards": [1]})
        self.assertEqual(out["action"], "abgelehnt")
        self.assertFalse(out["ok"])
        self.assertEqual(m.CALLS, [])
        self.assertIsNone(hw._booking)

    def test_a_card_that_is_busy_despite_the_window_is_not_measured(self):
        self.gq.used = {1: 9000}
        hw = self.make()
        m = self.mod(hw)
        out = hw.measure({"cards": [0, 1]})
        self.assertEqual(out["action"], "abgelehnt")
        self.assertIn("occupied", out["error"])
        self.assertEqual(m.CALLS, [])
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")

    def test_a_window_with_too_little_time_left_is_given_back(self):
        self.gq.seconds_left = 40.0
        hw = self.make()
        m = self.mod(hw)
        out = hw.measure({"cards": [0]})
        self.assertEqual(out["action"], "abgelehnt")
        self.assertEqual(m.CALLS, [])
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")

    def test_the_child_timeout_never_outlives_the_window(self):
        self.gq.seconds_left = 200.0
        hw = self.make()
        m = self.mod(hw)
        hw.measure({"cards": [0]})
        self.assertEqual(m.CALLS[0]["timeout_s"], 175.0)
        self.gq.seconds_left = 900.0
        hw.measure({"cards": [0]})
        self.assertEqual(m.CALLS[1]["timeout_s"], hwprofil.CHILD_CAP_S)

    def test_python_prefix_and_the_measure_tree_reach_the_child(self):
        hw = self.make(python="/venv/bin/python", prefix=["systemd-run", "--scope"], measure_tree="/full/python")
        m = self.mod(hw)
        hw.measure({"cards": [0]})
        c = m.CALLS[0]
        self.assertEqual((c["python"], c["prefix"], c["pythonpath"]), ("/venv/bin/python", ["systemd-run", "--scope"], "/full/python"))

    def test_a_second_press_during_a_run_does_not_start_a_second_run(self):
        hw = self.make()
        m = self.mod(hw)
        hw._job = {"state": "running", "cards": [0]}
        out = hw.measure({"cards": [0]})
        self.assertEqual(out["action"], "laeuft_bereits")
        self.assertEqual(m.CALLS, [])
        self.assertEqual(self.gq.calls("POST", "/api/v1/bookings"), [])


class TestRequestsAndErrors(Base):
    def test_bad_bodies_are_value_errors(self):
        hw = self.make()
        self.mod(hw)
        for bad in ({}, {"cards": []}, {"cards": "1"}, {"cards": [1, 1]}, {"cards": [True]}, {"cards": [0.5]}, {"cards": [7]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                hw.measure(bad)
        self.assertEqual(self.gq.calls("POST", "/api/v1/bookings"), [])

    def test_gpuq_down_is_a_refusal_not_a_crash(self):
        hw = self.make(gq=FakeGpuq(down=True))
        self.mod(hw)
        out = hw.measure({"cards": [0]})
        self.assertFalse(out["ok"])
        self.assertIn("gpuq", out["error"])

    def test_the_token_never_appears_in_any_answer(self):
        self.gq.state = "pending"
        hw = self.make()
        self.mod(hw)
        a = hw.measure({"cards": [0]})
        b = hw.get()
        self.assertNotIn("TOKEN", json.dumps(a) + json.dumps(b))

    def test_no_planner_tree_says_how_to_stage(self):
        hw = hwprofil.HwProfil(http=self.gq, tree=os.path.join(self.tmp, "nirgends"))
        out = hw.get()
        self.assertFalse(out["ok"])
        self.assertIn("hardware_profile.py", out["error"])
        with self.assertRaises(RuntimeError):
            hw.measure({"cards": [0]})


class TestReadAndHousekeeping(Base):
    def test_get_returns_the_profile_the_window_and_the_gpuq_cards(self):
        self.gq.state = "pending"
        hw = self.make()
        self.mod(hw)
        hw.measure({"cards": [1]})
        out = hw.get()
        self.assertTrue(out["ok"])
        self.assertEqual(out["profile"]["schema"], "flliper.hardware/1")
        self.assertEqual(out["window"]["state"], "pending")
        self.assertEqual([c["index"] for c in out["gpuq"]["cards"]], [0, 1, 2])
        self.assertEqual(out["owner"], "profil-editor")
        self.assertEqual(out["problems"], [])

    def test_a_finished_window_leaves_memory_on_the_next_read(self):
        self.gq.state = "pending"
        hw = self.make()
        self.mod(hw)
        hw.measure({"cards": [1]})
        self.gq.bookings["bk1"]["state"] = "over"
        hw.get()
        self.assertIsNone(hw._booking)

    def test_a_window_nobody_uses_is_given_back_on_the_next_read(self):
        t = [1000.0]
        self.gq.state = "pending"
        hw = self.make(clock=lambda: t[0])
        self.mod(hw)
        hw.measure({"cards": [1]})
        self.gq.bookings["bk1"]["state"] = "running"
        hw.get()
        self.assertIsNotNone(hw._booking)
        t[0] += hwprofil.IDLE_RELEASE_S + 1
        hw.get()
        self.assertIsNone(hw._booking)
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")

    def test_cancel_returns_a_waiting_window(self):
        self.gq.state = "pending"
        hw = self.make()
        self.mod(hw)
        hw.measure({"cards": [1]})
        self.assertEqual(hw.cancel(), {"ok": True, "released": True})
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")
        self.assertEqual(hw.cancel(), {"ok": True, "released": False})

    def test_a_restart_returns_the_window_the_old_process_held(self):
        self.gq.state = "pending"
        state = os.path.join(self.tmp, "state")
        hw = self.make(state_dir=state)
        self.mod(hw)
        hw.measure({"cards": [1]})
        self.assertTrue(os.path.exists(os.path.join(state, "hwprofil_window.json")))
        hw2 = self.make(state_dir=state)       # neuer Prozess
        self.assertEqual(self.gq.bookings["bk1"]["state"], "released")
        self.assertFalse(os.path.exists(os.path.join(state, "hwprofil_window.json")))
        self.assertIsNone(hw2._booking)


class _App:
    version = "test"

    def __init__(self, hw, edition="rig"):
        self.hwprofil, self.edition = hw, edition


def _serve(app):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _req(srv, method, path, body=None, headers=None):
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (srv.server_address[1], path), data=data, method=method,
                               headers=dict({"content-type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read().decode(), resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e.headers


class TestHttpRoutes(Base):
    def setUp(self):
        super().setUp()
        self.hw = self.make(synchronous=True)
        self.mod(self.hw)
        self.srv = _serve(_App(self.hw))
        self.addCleanup(self.srv.shutdown)

    def test_get_and_post_roundtrip(self):
        st, body, _ = _req(self.srv, "GET", "/api/hwprofil")
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(body)["profile"]["schema"], "flliper.hardware/1")
        st, body, _ = _req(self.srv, "POST", "/api/hwprofil/measure", {"cards": [1]})
        self.assertEqual(st, 200, body)
        self.assertEqual(json.loads(body)["action"], "messung_gestartet")
        self.assertNotIn("TOKEN", body)

    def test_pending_is_a_conflict_with_the_window_status(self):
        self.gq.state = "pending"
        st, body, _ = _req(self.srv, "POST", "/api/hwprofil/measure", {"cards": [1]})
        out = json.loads(body)
        self.assertEqual(st, 200)  # wartet ist kein Fehler: ok=True
        self.assertEqual(out["action"], "wartet")

    def test_refusal_is_409_bad_body_is_400(self):
        self.gq.state = "unplannable"
        st, _, _ = _req(self.srv, "POST", "/api/hwprofil/measure", {"cards": [1]})
        self.assertEqual(st, 409)
        st, body, _ = _req(self.srv, "POST", "/api/hwprofil/measure", {"cards": []})
        self.assertEqual(st, 400)
        self.assertFalse(json.loads(body)["ok"])
        st, _, _ = _req(self.srv, "POST", "/api/hwprofil/measure", b"{kaputt")
        self.assertEqual(st, 400)
        st, _, _ = _req(self.srv, "POST", "/api/hwprofil/measure", [1, 2])
        self.assertEqual(st, 400)

    def test_cancel_route(self):
        st, body, _ = _req(self.srv, "POST", "/api/hwprofil/cancel", {})
        self.assertEqual((st, json.loads(body)), (200, {"ok": True, "released": False}))

    def test_unknown_post_path_is_404(self):
        self.assertEqual(_req(self.srv, "POST", "/api/nirgends", {})[0], 404)

    def test_lan_only(self):
        for method, path, body in (("GET", "/api/hwprofil", None), ("POST", "/api/hwprofil/measure", {"cards": [1]})):
            st, _, _ = _req(self.srv, method, path, body, {"X-Forwarded-For": "1.2.3.4"})
            self.assertEqual(st, 403, path)
        self.assertEqual(self.gq.calls("POST", "/api/v1/bookings"), [])

    def test_release_edition_shows_but_never_measures(self):
        # Auftrag 1984: Anzeige (GET) und Skript auch im Release; Messen und Fenster zurueckgeben bucht gpuq: dort 403, keine Buchung
        srv = _serve(_App(self.hw, "release"))
        self.addCleanup(srv.shutdown)
        self.assertEqual(_req(srv, "GET", "/api/hwprofil")[0], 200)
        self.assertEqual(_req(srv, "GET", "/hwprofil.js")[0], 200)
        for path, body in (("/api/hwprofil/measure", {"cards": [1]}), ("/api/hwprofil/cancel", {})):
            st, txt, _ = _req(srv, "POST", path, body)
            self.assertEqual(st, 403, path)
            self.assertIn("gpuq", json.loads(txt)["error"])
        self.assertEqual(self.gq.calls("POST", "/api/v1/bookings"), [])

    def test_the_script_is_served_in_the_rig_edition(self):
        st, body, h = _req(self.srv, "GET", "/hwprofil.js")
        self.assertEqual(st, 200)
        self.assertIn("javascript", h["Content-Type"])
        self.assertIn("HwProfil", body)


def _node():
    for c in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node", shutil.which("bun")):
        if c and os.path.exists(c):
            return c
    return None


class TestDisplayModule(unittest.TestCase):
    """hwprofil.js: eigenes Modul für den Editor-Reiter (930).  Ausgeführt wird es nur, wenn node/bun da ist."""

    DOC = {
        "ok": True,
        "profile": {
            "schema": "flliper.hardware/1", "id": "sha256:" + "ab" * 32, "driver": "595.58",
            "formats": [{"key": "bf16", "unit": "TFLOPS", "label": "bf16"}, {"key": "int8", "unit": "TOPS", "label": "int8 W8A8"}],
            "cards": [
                {"ord": i, "nvml_index": i, "name": "NVIDIA GeForce RTX 3080", "class_key": "RTX3080", "cc": [8, 6],
                 "sm_count": {"v": 68, "src": "gemessen", "at": 1000.0, "probe": "card_probe-a.json"},
                 "l2_mib": {"v": None, "src": "nicht gemessen", "note": "Messarm noch nicht gelaufen"},
                 "vram_total_mib": {"v": 20480, "src": "NVML", "unit": "MiB"},
                 "bar1_total_mib": {"v": 256, "src": "NVML", "unit": "MiB"},
                 "mem_gbs": {k: {"v": 700.0, "src": "gemessen", "at": 1000.0, "probe": "p", "unit": "GB/s"} for k in ("read", "copy", "gemv")},
                 "compute": {"bf16": {"v": 60.0, "src": "gemessen", "at": 1000.0, "probe": "p", "unit": "TFLOPS"},
                             "int8": {"v": None, "src": "nicht gemessen", "note": "kein sgl_kernel", "unit": "TOPS"}},
                 "h2d": {"gbs": {"v": 6.0, "src": "gemessen", "at": 1000.0, "probe": "p"}, "lat_us": {"v": 12.5, "src": "gemessen", "at": 1000.0, "probe": "p"}},
                 "d2h": {"gbs": {"v": 6.0, "src": "gemessen", "at": 1000.0, "probe": "p"}, "lat_us": {"v": 12.5, "src": "gemessen", "at": 1000.0, "probe": "p"}},
                 "pcie": {k: {"v": 4, "src": "NVML"} for k in ("max_gen", "max_width", "cur_gen", "cur_width")},
                 "power": {"limit_w": {"v": 230, "src": "NVML", "unit": "W"}},
                 "state": {"sm_mhz": {"v": 1900, "src": "gemessen", "at": 1000.0, "probe": "p"}, "sm_max_mhz": {"v": 2100, "src": "gemessen", "at": 1000.0, "probe": "p"},
                           "throttled": True, "throttle": ["sw_thermal_slowdown"]},
                 "probed_at": 1000.0, "stale": False}
                for i in range(2)],
            "links": [
                {"src": 0, "dst": 1, "transport": "host_staging", "gbs": {"v": 5.1, "src": "gemessen", "at": 1000.0, "probe": "p", "unit": "GB/s"},
                 "lat_us": {"v": 30.0, "src": "gemessen", "at": 1000.0, "probe": "p", "unit": "µs"}},
                {"src": 0, "dst": 1, "transport": "bar1", "gbs": {"v": None, "src": "nicht gemessen", "note": "BAR1 nicht gemessen"},
                 "lat_us": {"v": None, "src": "nicht gemessen", "note": "x"}}],
            "bar1": {"measured": False, "note": "BAR1 stretch per pair: NOT MEASURED."},
            "sources": {"card_probe": [{"file": "card_probe-a.json", "created": 1000.0, "cards": 2}], "stage0": [], "nvml": {"issues": []}},
            "measure_needed": True},
        "window": {"id": "bk1", "state": "pending", "cards": [1], "seconds_left": None, "note": "karte 1: belegt"},
        "job": {"state": "ok", "line": "HWPROFIL-MESSUNG ok", "warnings": ["lanes nicht gemessen: kein sgl_kernel"]},
        "problems": [],
    }

    def test_structure(self):
        with open(os.path.join(STATIC, "hwprofil.js"), encoding="utf-8") as fh:
            js = fh.read()
        for needle in ("HwProfil", "render", "mount", "api/hwprofil", "/measure", "/cancel"):
            self.assertIn(needle, js)
        self.assertNotIn("eval(", js)
        self.assertNotIn("localStorage", js)       # kein Zustand im Browser: das Profil kommt vom Dienst
        self.assertNotIn("setInterval", js)        # kein Hintergrundabfrager (KEINE-CRONS)

    def test_renders_with_node(self):
        node = _node()
        if not node:
            self.skipTest("weder node noch bun vorhanden")
        script = ("const H=require(process.argv[1]);const d=JSON.parse(process.argv[2]);"
                  "process.stdout.write(H.render(d,{now:1100}));")
        out = subprocess.run([node, "-e", script, os.path.join(STATIC, "hwprofil.js"), json.dumps(self.DOC)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        h = out.stdout
        self.assertIn("nicht gemessen", h)
        self.assertIn("kein sgl_kernel", h)        # der Grund steht im Hover
        self.assertIn("meas.", h)                   # Quellenmarke
        self.assertIn("NVML", h)
        self.assertIn("host staging", h)
        self.assertIn("BAR1 stretch per pair: NOT MEASURED", h)
        self.assertIn("throttled: sw_thermal_slowdown", h)
        self.assertIn("Window bk1", h)
        self.assertIn("lanes nicht gemessen", h)
        self.assertNotIn("<script", h)
        # Escapes: ein Gerätename mit Markup wird nicht eingebettet
        evil = json.loads(json.dumps(self.DOC))
        evil["profile"]["cards"][0]["name"] = "<img src=x onerror=alert(1)>"
        out = subprocess.run([node, "-e", script, os.path.join(STATIC, "hwprofil.js"), json.dumps(evil)],
                             capture_output=True, text=True, timeout=30)
        self.assertNotIn("<img", out.stdout)


class TestRealModule(unittest.TestCase):
    """Der echte Baum der 27B-Linie (Zweig desk/profil-s2-hw-py), wenn er erreichbar ist (HWPROFIL_TREE)."""

    def test_real_profile_through_the_route_layer(self):
        tree = os.environ.get("HWPROFIL_TREE")
        if not tree or not os.path.isfile(os.path.join(tree, hwprofil.MODULE_REL)):
            self.skipTest("HWPROFIL_TREE zeigt auf keinen Baum mit hardware_profile.py")
        with tempfile.TemporaryDirectory() as d:
            hw = hwprofil.HwProfil(http=FakeGpuq(), tree=tree, cache_dir=d)
            mod = hw._module()
            self.assertEqual(mod.SCHEMA, "flliper.hardware/1")
            out = hw.get()
            self.assertTrue(out["ok"])
            self.assertEqual(out["problems"], [])


if __name__ == "__main__":
    unittest.main()
