"""#1275 fix 5: the admin key is absent from EVERY position of internal_states.

sb5e read the live 43-char key at `internal_states[0].admin_api_key`, verbatim,
WITHOUT a bearer, on all three ports. Fix 5 sent that producer through
`redacted_dict()` and gated both info routes ADMIN_OPTIONAL.

This test asserts the property the boot actually violated, and asserts it the
way the attacker reads it: WALK THE WHOLE JSON and fail if the key bytes appear
ANYWHERE -- not `body["admin_api_key"] == "<redacted>"` at one known key. The
first K-2 fix asserted the known position and the key turned up one level down
in a second producer; a positional assertion cannot catch a producer nobody
enumerated.
"""

import dataclasses
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from sglang.srt.server_args import REDACTED, ServerArgs
from sglang.test.test_utils import CustomTestCase

KEY = "sb5e-live-admin-key-0123456789abcdefghijk"   # 43 chars, the sb5e shape


def walk(node, hits, path="$"):
    """Every scalar in the tree, with its path -- so a hit NAMES where it sat."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and KEY in k:
                hits.append(f"{path} (as a KEY)")
            walk(v, hits, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            walk(v, hits, f"{path}[{i}]")
    elif isinstance(node, str) and KEY in node:
        hits.append(path)


def bare_server_args():
    a = ServerArgs.__new__(ServerArgs)
    for f in dataclasses.fields(ServerArgs):
        d = f.default if f.default is not dataclasses.MISSING else (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        object.__setattr__(a, f.name, d)
    a.model_path = "/model"
    a.admin_api_key = KEY
    a.api_key = None
    a.dtype = "bfloat16"
    a.model_config = SimpleNamespace(dtype=None)
    return a


class TheKeyIsNowhereInTheBody(CustomTestCase):
    def setUp(self):
        from sglang.srt.entrypoints import http_server

        self.sa = bare_server_args()
        # internal_states carries the SAME redacted dict the scheduler produces
        internal = [dict(self.sa.redacted_dict(), last_gen_throughput=0.0)]

        async def _internal():
            return internal

        self.prev = getattr(http_server, "_global_state", None)
        http_server._global_state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                server_args=self.sa, get_internal_state=_internal),
            scheduler_info={"status": "ready"},
        )
        self.addCleanup(setattr, http_server, "_global_state", self.prev)
        self.client = TestClient(http_server.app, raise_server_exceptions=False)

    def test_the_key_appears_nowhere_in_the_whole_json(self):
        for path in ("/get_server_info", "/server_info"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, f"{path}: {r.text[:200]}")
            hits = []
            walk(r.json(), hits)
            self.assertEqual(hits, [], f"{path}: key bytes at {hits}")

    def test_the_key_is_absent_from_internal_states_specifically(self):
        """The exact sb5e position."""
        r = self.client.get("/get_server_info")
        self.assertEqual(r.status_code, 200, r.text[:200])
        states = r.json().get("internal_states") or []
        self.assertTrue(states, "internal_states missing -- the walk proves nothing")
        self.assertEqual(states[0]["admin_api_key"], REDACTED)
        self.assertNotIn(KEY, r.text)

    def test_the_walk_can_actually_find_a_key(self):
        """Not vacuous: the walker finds the key at a nested position."""
        hits = []
        walk({"internal_states": [{"admin_api_key": KEY}]}, hits)
        self.assertEqual(hits, ["$.internal_states[0].admin_api_key"])


class TheRoutesRefuseWithoutTheBearer(CustomTestCase):
    """The auth half. The middleware is installed at LAUNCH, not at import, so
    it is installed here explicitly -- otherwise TestClient exercises an app
    with no auth at all and a 200 would prove nothing."""

    def test_admin_optional_denies_an_unkeyed_request_when_a_key_is_configured(self):
        from sglang.srt.utils.auth import AuthLevel, decide_request_auth

        # no Authorization header, admin key configured -> DENIED
        d = decide_request_auth(
            method="GET", path="/get_server_info",
            auth_level=AuthLevel.ADMIN_OPTIONAL, authorization_header=None,
            api_key=None, admin_api_key=KEY)
        self.assertFalse(d.allowed)
        # correct bearer -> allowed
        d2 = decide_request_auth(
            method="GET", path="/get_server_info",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
            authorization_header=f"Bearer {KEY}", api_key=None, admin_api_key=KEY)
        self.assertTrue(d2.allowed)

    def test_both_info_routes_carry_that_level(self):
        import inspect
        import re

        from sglang.srt.entrypoints import http_server

        src = inspect.getsource(http_server)
        for path in ("/get_server_info", "/server_info"):
            m = re.search(
                r'@app\.(?:api_route|get|post)\(\s*"%s"[^\n]*\n@auth_level\(AuthLevel\.ADMIN_OPTIONAL\)'
                % re.escape(path), src)
            self.assertIsNotNone(m, f"{path} is not ADMIN_OPTIONAL")


if __name__ == "__main__":
    unittest.main()
