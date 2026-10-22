"""SEATS 1022: tab "Seats & devices" -- the UI of rdx-seatmgr (rdma-display multiseat on the Proxmox host).

What this pins:
* /api/seats/* is passed to rdx-seatmgr (GET state, POST assign/forget/claim/claim/cancel) with the body and the status;
* it moves devices between the seat containers: 403 through the public proxy, 404 in the release edition, 502 with a
  reason when rdx-seatmgr does not answer, 404 for any other path;
* the tab and seats.js are in the rig edition only;
* seats.js renders one column per seat plus "Unassigned", blinks an active device, offers no choice for a device the
  host uses, shows the claim countdown -- and every text it shows has a German entry.
"""

import json
import os
import shutil
import subprocess
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rigdash import server
from rigdash.tests.test_i18n_0810 import SAME_IN_BOTH, Texts, german, memory

STATIC = os.path.join(os.path.dirname(os.path.abspath(server.__file__)), "static")
NODE = next((p for p in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node") if p and os.path.exists(p)), None)

STATE = {
    "blink_s": 2.0, "rules": [],
    "seats": [
        {"seat": 1, "ct": 3101, "name": "RDMA Display - 5090", "gpu": "0000:0d:00.0", "sound": "pci-0000:0d:00.1",
         "running": True, "monitor": True, "claim_s": None, "claimed": None},
        {"seat": 2, "ct": 3102, "name": "RDMA Display - 3080a", "gpu": "0000:0e:00.0", "sound": "pci-0000:0e:00.1",
         "running": False, "monitor": False, "claim_s": 11.2, "claimed": None},
    ],
    "devices": [
        {"id": "usb:1-3", "kind": "usb", "name": "PixArt USB Optical Mouse", "path": "pci-0000:0a:00.1-usb-0:3",
         "devid": "usb:093a:2510", "roles": ["mouse"], "keys": True, "assignable": True, "host_drivers": ["usbhid"],
         "nodes": [{"dev": "/dev/input/event2"}, {"dev": "/dev/hidraw1"}], "seat": 1, "how": "default (monitor)",
         "active": True, "last_input_s": 0.4, "rules": []},
        {"id": "sound:0000:0d:00.1", "kind": "sound", "name": "NVidia (0000:0d:00.1)", "path": "pci-0000:0d:00.1",
         "devid": "pci:10de:22e8", "roles": ["sound"], "keys": False, "assignable": True, "host_drivers": [],
         "nodes": [{"dev": "/dev/snd/controlC1"}], "seat": 1, "how": "GPU sound", "active": False, "last_input_s": None, "rules": []},
        {"id": "usb:2-4", "kind": "usb", "name": "Realtek USB LAN", "path": "pci-0000:0a:00.3-usb-0:4",
         "devid": "usb:0bda:8156", "roles": [], "keys": False, "assignable": False, "host_drivers": ["r8152"],
         "nodes": [{"dev": "/dev/bus/usb/002/003"}], "seat": None, "how": "unassigned", "active": False, "last_input_s": None, "rules": []},
        {"id": "usb:3-4", "kind": "usb", "name": "AURA LED Controller", "path": "pci-0000:0a:00.3-usb-0:4",
         "devid": "usb:0b05:18f3", "roles": [], "keys": False, "assignable": True, "host_drivers": ["usbhid"],
         "nodes": [{"dev": "/dev/hidraw0"}], "seat": None, "how": "port rule", "active": False, "last_input_s": None,
         "rules": [{"kind": "port", "match": "x", "seat": 2}]},
    ],
}


class _FakeSeatmgr(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _answer(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _FakeSeatmgr.seen.append(("GET", self.path, None))
        return self._answer(200, STATE) if self.path == "/api/state" else self._answer(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        _FakeSeatmgr.seen.append(("POST", self.path, body))
        if body.get("seat") == 9:
            return self._answer(400, {"error": "no seat 9"})
        return self._answer(200, {"ok": True, "echo": body, "path": self.path})


class _App:
    version = "test"

    def __init__(self, url, edition="rig"):
        self.seatmgr_url, self.edition = url, edition


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _req(srv, method, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (srv.server_address[1], path), data=data, method=method,
                               headers=dict({"content-type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


class TestSeatsRoutes(unittest.TestCase):
    def setUp(self):
        _FakeSeatmgr.seen = []
        self.fake = _serve(_FakeSeatmgr)
        self.addCleanup(self.fake.shutdown)
        self.url = "http://127.0.0.1:%d" % self.fake.server_address[1]

    def dash(self, url=None, edition="rig"):
        srv = _serve(server.make_handler(_App(self.url if url is None else url, edition)))
        self.addCleanup(srv.shutdown)
        return srv

    def test_state_and_actions_are_passed_on(self):
        d = self.dash()
        st, body = _req(d, "GET", "/api/seats/state")
        self.assertEqual((st, body["seats"][0]["ct"]), (200, 3101))
        for path, target, req in (("/api/seats/assign", "/api/assign", {"id": "usb:1-3", "seat": 2, "remember": "port"}),
                                  ("/api/seats/forget", "/api/forget", {"id": "usb:1-3"}),
                                  ("/api/seats/claim", "/api/claim", {"seat": 1, "seconds": 15}),
                                  ("/api/seats/claim/cancel", "/api/claim/cancel", {"seat": 1})):
            st, body = _req(d, "POST", path, req)
            self.assertEqual((st, body["path"], body["echo"]), (200, target, req))
        st, body = _req(d, "POST", "/api/seats/assign", {"id": "usb:1-3", "seat": 9})
        self.assertEqual((st, body["error"]), (400, "no seat 9"))        # seatmgr's refusal reaches the page

    def test_lan_only_rig_only_and_errors(self):
        st, body = _req(self.dash(), "GET", "/api/seats/state", headers={"X-Forwarded-For": "1.2.3.4"})
        self.assertEqual(st, 403)
        st, _ = _req(self.dash(), "POST", "/api/seats/claim", {"seat": 1}, headers={"X-Forwarded-Prefix": "/rigdash"})
        self.assertEqual(st, 403)
        self.assertEqual(_req(self.dash(edition="release"), "GET", "/api/seats/state")[0], 404)
        self.assertEqual(_req(self.dash(), "GET", "/api/seats/other")[0], 404)
        self.assertEqual(_req(self.dash(url=""), "GET", "/api/seats/state")[0], 404)       # --seatmgr '' = off
        st, body = _req(self.dash(url="http://127.0.0.1:9"), "GET", "/api/seats/state")
        self.assertEqual(st, 502)
        self.assertIn("not reachable", body["error"])
        self.assertFalse([s for s in _FakeSeatmgr.seen if s[0] == "POST"])                 # nothing went through

    def test_tab_and_module_are_rig_only(self):
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            page = fh.read()
        self.assertIn('data-tab="seats"', page)
        self.assertIn('<script src="seats.js"></script>', page)
        self.assertIn("/seats.js", server.DEV_STATIC_FILES)
        rel = server.edition_page(page, "release")
        self.assertNotIn('data-tab="seats"', rel)
        self.assertNotIn("seats.js", rel)
        self.assertNotIn("st-cols", rel)


@unittest.skipUnless(NODE, "no node")
class TestSeatsJs(unittest.TestCase):
    def render(self, state, now=1792664300):
        js = ("require(%s); process.stdout.write(globalThis.RigSeats.render(%s, {now: %d}));"
              % (json.dumps(os.path.join(STATIC, "seats.js")), json.dumps(state), now))
        return subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True, timeout=30).stdout

    def test_columns_blink_locked_claim(self):
        html = self.render(STATE)
        self.assertEqual(html.count('<section class="st-col'), 3)                        # two seats + Unassigned
        self.assertIn('st-dev st-blink" data-id="usb:1-3"', html)
        lan = html.split('data-id="usb:2-4"')[1].split('class="st-dev')[0]
        self.assertIn("Used by the host", lan)
        self.assertNotIn("<select", lan)
        self.assertIn("Press a key on the device for this seat", html)
        self.assertIn('<span class="num">12 s</span>', html)
        self.assertIn('data-act="claim" data-seat="1"', html)
        aura = html.split('data-id="usb:3-4"')[1]
        self.assertIn('data-act="forget"', aura)                                          # it has a rule
        self.assertIn('data-act="port" disabled', aura)                                   # unassigned: nothing to remember

    def test_every_text_has_german(self):
        mem = memory()
        p = Texts()
        st = json.loads(json.dumps(STATE))
        st["seats"][1]["claim_s"] = None
        st["seats"][0]["claimed"] = {"id": "usb:1-3", "name": "PixArt USB Optical Mouse", "t": 1792664295}
        p.feed(self.render(STATE) + self.render(st) + self.render({"seats": [], "devices": []}) + self.render(None))
        data = {d["name"] for d in STATE["devices"]} | {s["name"] for s in STATE["seats"]} \
            | {d["path"] for d in STATE["devices"]} | {d["devid"] for d in STATE["devices"]}
        missing = [t for t in p.found if t not in data and t not in SAME_IN_BOTH and not t.startswith(("/dev/", "CT "))
                   and any(c.isalpha() for c in t) and german(t, mem) == t and not t.endswith(" s")]
        self.assertFalse(missing, missing)


if __name__ == "__main__":
    unittest.main()
