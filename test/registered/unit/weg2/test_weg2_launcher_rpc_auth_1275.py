# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1275 fix 2: every launcher RPC carries the admin key.

THE DEFECT, and it is mine. #1275 minted a per-boot admin key, armed it on both
groups, and taught the FRONT to authenticate -- then boot weg2sb5 died 39 s
after P READY::

    launcher.py sleep_group() -> http("POST", .../release_memory_occupation)
    P: "POST /release_memory_occupation HTTP/1.1" 401 Unauthorized
    WEG2-LAUNCH REFUSED: sleep(P) failed: HTTP 401 '{"error":"Unauthorized"}'

THE CLASS: access control introduced at the SERVER and wired into ONE OF TWO
CLIENTS. #1275 enumerated the ROUTES exhaustively (parsed out of
http_server.py's decorators, so route drift could not hide) and then enumerated
the CALLERS by assumption -- "the front drives the flip" -- and stopped there.
The launcher sleeps group P during the START SEQUENCE, before the front process
exists. `Authorization` appeared exactly once in launcher.py, inside a curl
EXAMPLE string.

WHAT THIS SUITE PINS, so the next call site fails at the desk and not at boot:
the enumeration is done BY AST over the module rather than by a hand-kept list,
and the auth lives at the single door `http()` rather than at each call site --
which is the only version of this fix a future caller cannot silently miss.
"""

import ast
import inspect
import json
import os
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.srt.weg2 import admin_key as ak
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

LAUNCHER_SRC = inspect.getsource(L)


def _tree():
    return ast.parse(LAUNCHER_SRC)


def http_call_sites():
    """Every `http(...)` call in the launcher, as (line, enclosing function)."""
    tree = _tree()
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "http":
                out.append((n.lineno, fn.name))
    return sorted(set(out))


class TheCallerEnumeration(CustomTestCase):
    """RED-FIRST: the guard that would have caught sb5 at the desk."""

    def test_the_single_door_attaches_the_bearer(self):
        src = inspect.getsource(L.http)
        self.assertIn("Authorization", src)
        self.assertIn("Bearer", src)
        self.assertIn("_ADMIN_KEY", src)

    def test_every_rpc_site_goes_through_that_door(self):
        """No call site may build its own request.

        `urllib.request.Request` / `urlopen` must appear ONLY inside `http`. A
        second door is how the first one stops being sufficient.
        """
        tree = _tree()
        offenders = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name == "http":
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Attribute) and n.attr in ("urlopen", "Request"):
                    offenders.append((fn.name, n.lineno, n.attr))
        self.assertEqual(offenders, [], f"HTTP built outside http(): {offenders}")

    def test_the_site_count_is_derived_not_hand_kept(self):
        self.assertEqual(L.rpc_site_count(), len(http_call_sites()))
        self.assertGreaterEqual(L.rpc_site_count(), 2)

    def test_the_two_known_sites_are_the_sb5_ones(self):
        """Names the enumeration explicitly, so a NEW site shows up as a diff
        rather than passing silently under a >= check."""
        fns = sorted({fn for _ln, fn in http_call_sites()})
        self.assertIn("sleep_group", fns, "the site that killed sb5")
        self.assertIn("wait_ready", fns, "the /health probe")

    def test_no_bearer_before_the_key_is_armed(self):
        """An unkeyed boot must send nothing -- pre-#1275 behaviour."""
        L.set_admin_key(None)
        try:
            self.assertIsNone(L._ADMIN_KEY)
        finally:
            L.set_admin_key(None)


class _Handler(BaseHTTPRequestHandler):
    """A group that behaves like an ADMIN_OPTIONAL route with a key configured."""

    KEY = "the-boot-key"
    seen = []

    def do_POST(self):  # noqa: N802
        auth = self.headers.get("Authorization")
        _Handler.seen.append(auth)
        if auth != f"Bearer {_Handler.KEY}":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"Unauthorized"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"success":true}')

    def log_message(self, *a):  # noqa: A003
        pass


class TheSb5Shape(CustomTestCase):
    """sleep_group against a stub that REQUIRES the key: 401 before, 200 after."""

    def setUp(self):
        _Handler.seen = []
        self.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()
        self.addCleanup(self.srv.shutdown)
        self.addCleanup(lambda: L.set_admin_key(None))

    def test_red_first_unkeyed_sleep_is_the_sb5_401(self):
        L.set_admin_key(None)
        with self.assertRaises(Exception) as cm:
            L.sleep_group(self.port, lambda *_a, **_k: None, "P", ["weights_0"])
        self.assertIn("401", str(cm.exception))
        self.assertEqual(_Handler.seen, [None])

    def test_green_the_armed_launcher_sleeps_successfully(self):
        L.set_admin_key(_Handler.KEY)
        L.sleep_group(self.port, lambda *_a, **_k: None, "P", ["weights_0"])
        self.assertEqual(_Handler.seen, [f"Bearer {_Handler.KEY}"])

    def test_the_health_probe_carries_it_too(self):
        """One door means /health is authenticated as well -- harmless on a
        NORMAL route with no api_key, and no per-path allowlist to drift."""
        L.set_admin_key(_Handler.KEY)
        code, _body = L.http("POST", f"http://127.0.0.1:{self.port}/anything", {})
        self.assertEqual(code, 200)


class TheKeyDiesWithTheBoot(CustomTestCase):
    def setUp(self):
        self.addCleanup(lambda: L.set_admin_key(None))

    def test_drop_removes_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = ak.write(os.path.join(d, "k"), "abc")
            L.set_admin_key("abc", p)
            msg = L.drop_admin_key_file()
            self.assertFalse(os.path.exists(p))
            self.assertIn("removed", msg)

    def test_drop_is_idempotent_and_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gone")
            L.set_admin_key("abc", p)
            self.assertIn("already gone", L.drop_admin_key_file())

    def test_no_key_file_is_not_an_error(self):
        L.set_admin_key(None, "")
        self.assertIn("no key file", L.drop_admin_key_file())

    def test_the_teardown_path_drops_it(self):
        """A teardown runs in a FRESH process that minted nothing, so the path
        must come off the state json, not off the module global."""
        src = inspect.getsource(L.teardown)
        self.assertIn("drop_admin_key_file", src)
        self.assertIn('st.get("admin_key_file"', src)

    def test_the_killer_path_drops_it_too(self):
        """weg2sb5 refused and left its key file behind."""
        src = inspect.getsource(L.cli)
        self.assertIn("drop_admin_key_file", src)

    def test_the_state_carries_the_path_and_never_the_key(self):
        st = L.BootState(tag="t", tip="x", tree="/", stamp="s")
        st.admin_key_file = "/g/weg2/boot_t.adminkey"
        blob = json.dumps(st.__dict__, default=str)
        self.assertIn("admin_key_file", blob)
        self.assertNotIn("admin_api_key", blob)


class TheAuthLineIsPublished(CustomTestCase):
    def test_the_launcher_prints_the_site_count(self):
        src = inspect.getsource(L.main)
        self.assertIn("WEG2-LAUNCH RPC auth=bearer sites=", src)
        self.assertIn("rpc_site_count()", src)

    def test_arming_precedes_the_first_rpc(self):
        """`set_admin_key` must run before any group is slept.

        AST order inside `main`: the arming statement's line must precede every
        `http(`/`sleep_group(` call site in that function.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(L.main)))
        arm = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "set_admin_key"]
        uses = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ("sleep_group", "http")]
        self.assertTrue(arm, "main never arms the key")
        if uses:
            self.assertLess(min(arm), min(uses),
                            "an RPC is issued before the key is armed")


if __name__ == "__main__":
    unittest.main()
