# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1288: the front's OWN group reads must go through the bearer-carrying seam.

THE REGRESSION. #1275 fix 5 (`ea3419fd84`) moved `/server_info` and
`/get_server_info` to `ADMIN_OPTIONAL`, which on a keyed boot means "require
the admin key". `Front.rpc` had learned the bearer in the same ticket -- but
the front's own internal reads had not, because they were plain
`session.get(...)` calls written at their point of use.

MEASURED ON BOOT weg2sb5f (`4f762260ba`), from the group's own access log
`/spinning/evidence-665-f1/boot_weg2_weg2sb5f_4f762260ba_0909_050638.D.log`:

    GET /server_info      401 x 180     200 x 1
    GET /get_server_info  401 x 507     200 x 2

Consequences, neither fatal and both silent: the #915 D-pool admission gate
went OFF by name (`d_pool_read_failed=85`) and priced nothing for the rest of
the boot, and `_draft_terms` fell to `accept_src=none` with both draft
counters stuck at 0. `/get_server_info` was NOT named in the boot record --
it is the sibling this sweep found, and it failed 2.8x more often.

THE FIX IS THE RULE, not the three call sites: exactly TWO methods may open an
HTTP call the front itself asks for -- `_rpc_attempt` (POST) and `group_get`
(GET) -- and both take their headers from the one injector
`admin_key.auth_headers`. `SeamIsStructural` below enforces that with an AST
scan.

WHY AN AST SCAN AND NOT A GREP (the #1285 lesson, paid for twice in this
campaign). A grep-shaped guard goes green for reasons that have nothing to do
with the code: it matches its own explanatory comment, or it resolves a repo
root that does not exist and scans nothing. Both failure modes look exactly
like a pass. So this file proves its scanner can FAIL -- `test_the_scanner_
reports_a_planted_violation` runs the identical scanner over a synthetic
module with a bare `session.get` in a non-seam method and asserts it is
reported -- and asserts a non-zero call census on the real file, so a broken
parse cannot pass by vacancy.
"""

import ast
import asyncio
import inspect
import os
import textwrap
import unittest
from typing import List, Optional, Tuple

from sglang.srt.utils.auth import AuthLevel, decide_request_auth
from sglang.srt.weg2 import admin_key as ak
from sglang.srt.weg2.front import Front
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

FRONT_REL = "python/sglang/srt/weg2/front.py"


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    while here != "/" and not os.path.isdir(os.path.join(here, "python", "sglang")):
        here = os.path.dirname(here)
    return here


def scan_http_calls(src: str, seam: Tuple[str, ...], passthrough: Tuple[str, ...],
                    ) -> Tuple[List[Tuple[str, int, str]], int]:
    """Every ``<something>.get/.post(`` call, with the method that encloses it.

    Returns ``(violations, total_calls_seen)``. A violation is a call whose
    enclosing function is neither a seam nor a named passthrough. The total is
    returned so a caller can refuse a zero-census scan -- an empty result from
    a broken parse must never read as "clean".

    Deliberately syntactic on the ATTRIBUTE NAME rather than on the receiver:
    `self.session.get`, `session.get` and a future `sess.get` are all the shape
    that must live in the seam, and typing the receiver here would re-introduce
    exactly the gap this guard exists to close.
    """
    tree = ast.parse(src)
    owner: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner.append((node.lineno, node.end_lineno or node.lineno, node.name))

    def enclosing(lineno: int) -> str:
        best, best_span = "<module>", None
        for lo, hi, name in owner:
            if lo <= lineno <= hi and (best_span is None or (hi - lo) < best_span):
                best, best_span = name, hi - lo
        return best

    violations: List[Tuple[str, int, str]] = []
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in ("get", "post"):
            continue
        # A URL is the discriminator between `session.get(url)` and
        # `dict.get(key)`: the first argument must be a string or an f-string.
        # `ast.Str` is deliberately not referenced -- it is gone in 3.12 and a
        # NameError here would look like a clean scan.
        if not node.args or not isinstance(node.args[0], (ast.Constant,
                                                          ast.JoinedStr)):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant):
            if not isinstance(first.value, str):
                continue
            text = first.value
        else:
            text = ast.dump(first)
        if "url" not in text and "http" not in text:
            continue
        total += 1
        where = enclosing(node.lineno)
        if where not in seam and where not in passthrough:
            violations.append((where, node.lineno, fn.attr))
    return violations, total


class SeamIsStructural(CustomTestCase):
    """RED-FIRST at `4f762260ba`: five internal calls sit outside the seam."""

    def setUp(self):
        self.src = open(os.path.join(_repo_root(), FRONT_REL),
                        encoding="utf-8").read()

    def test_the_scanner_reports_a_planted_violation(self):
        """CAN-FAIL PROOF. Without this the guard is green-by-vacancy (#1285)."""
        planted = textwrap.dedent(
            '''
            class Front:
                async def group_get(self, g, path):
                    async with self.session.get(f"{g.url}{path}") as r:
                        return r.status

                async def _sneaky_new_reader(self, g):
                    async with self.session.get(f"{g.url}/server_info") as r:
                        return r.status
            '''
        )
        bad, total = scan_http_calls(planted, ("group_get",), ())
        self.assertEqual(total, 2, "the scanner did not see both calls")
        self.assertEqual([v[0] for v in bad], ["_sneaky_new_reader"])

    def test_a_planted_post_is_caught_too(self):
        planted = textwrap.dedent(
            '''
            class Front:
                async def handle_thing(self, g):
                    async with self.session.post(f"{g.url}/flush_cache") as r:
                        return r.status
            '''
        )
        bad, total = scan_http_calls(planted, ("_rpc_attempt",), ())
        self.assertEqual(total, 1)
        self.assertEqual(bad[0][0], "handle_thing")
        self.assertEqual(bad[0][2], "post")

    def test_the_scanner_ignores_dict_get(self):
        """A `.get("key")` is not an HTTP call; a guard that flags it would be
        turned off by the next reader."""
        planted = 'def f(d):\n    return d.get("epoch")\n'
        bad, total = scan_http_calls(planted, (), ())
        self.assertEqual((bad, total), ([], 0))

    def test_the_real_front_has_no_call_outside_the_seam(self):
        """THE GUARD. Red at 4f762260ba: _d_pool_reading, _draft_terms,
        handle_health, handle_health_generate, health_poller, handle_abort."""
        bad, total = scan_http_calls(self.src, Front.SEAM_METHODS,
                                     Front.PASSTHROUGH_METHODS)
        self.assertGreaterEqual(total, 4,
                           "the scan found almost no HTTP calls in front.py -- "
                           "the parse or the path is wrong, and an empty result "
                           "must never read as clean")
        self.assertEqual(
            bad, [],
            "these open an HTTP call outside the seam; if it is a client "
            "proxy add it to Front.PASSTHROUGH_METHODS with its reason, "
            f"otherwise route it through group_get/rpc: {bad}")

    def test_the_seam_and_passthrough_names_exist_on_the_class(self):
        """Both lists are load-bearing; a typo would silently widen the guard."""
        for name in Front.SEAM_METHODS + Front.PASSTHROUGH_METHODS:
            self.assertTrue(callable(getattr(Front, name, None)),
                            f"{name} is listed but is not a method of Front")

    def test_there_is_exactly_one_get_seam_and_one_post_seam(self):
        self.assertEqual(set(Front.SEAM_METHODS), {"group_get", "_rpc_attempt"})
        self.assertIn("session.get(", inspect.getsource(Front.group_get))
        self.assertIn("session.post(", inspect.getsource(Front._rpc_attempt))

    def test_both_seams_use_the_one_injector(self):
        """No SECOND header injector -- both read `admin_key.auth_headers`.

        The ONE textual claim worth making, and only because a second injector
        would be a second NAME. How the headers reach the call (`headers=` vs
        a `**kw` dict) is a style detail, and an earlier version of this test
        asserted the literal `headers=` and went red on `group_get`'s `**kw`
        while the code was correct -- the assert-on-a-literal failure this
        campaign keeps paying for. The BEHAVIOUR is proved below, on the wire.
        """
        for name in Front.SEAM_METHODS:
            src = inspect.getsource(getattr(Front, name))
            self.assertIn("auth_headers", src,
                          f"{name} builds headers without the one injector")


class _Recorded:
    """The one `async with session.get(...)` result, capturing its kwargs."""

    def __init__(self, sink, status=200, body=b"{}", ctype="application/json"):
        self.sink, self.status, self.body, self.ctype = sink, status, body, ctype

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def read(self):
        return self.body

    @property
    def content_type(self):
        return self.ctype


class _CapturingSession:
    def __init__(self, status=200, body=b"{}"):
        self.calls = []
        self._status, self._body = status, body

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return _Recorded(self.calls, self._status, self._body)


def _front_with_key(key: Optional[str], session) -> Front:
    f = Front.__new__(Front)
    f.admin_key = key
    f.session = session
    return f


class TheHeaderSurvivesTheRealAuthDecision(CustomTestCase):
    """Not a TestClient 200: the middleware is installed at LAUNCH, so a
    TestClient built here would answer 200 with no key configured and prove
    nothing. The honest desk equivalent is the PURE decision function the
    middleware itself calls -- `decide_request_auth` -- fed with the headers
    `group_get` actually put on the wire.
    """

    KEY = "boot-key-abc123"

    def _headers_group_get_sends(self, key: Optional[str]) -> dict:
        sess = _CapturingSession()
        f = _front_with_key(key, sess)
        g = type("G", (), {"url": "http://127.0.0.1:30032", "name": "D"})()
        asyncio.run(f.group_get(g, "/server_info", 1.0))
        self.assertEqual(len(sess.calls), 1)
        return sess.calls[0][2]["headers"]

    def _decide(self, headers: dict, path="/server_info", method="GET",
                level=AuthLevel.ADMIN_OPTIONAL):
        return decide_request_auth(
            method=method, path=path,
            authorization_header=headers.get("Authorization"),
            api_key=None, admin_api_key=self.KEY, auth_level=level,
        )

    def test_red_first_no_bearer_is_refused_by_the_real_decision(self):
        """The measured 401. A front that sends no header IS refused."""
        self.assertFalse(self._decide({}).allowed)
        self.assertEqual(self._decide({}).error_status_code, 401)

    def test_the_seam_header_is_accepted(self):
        headers = self._headers_group_get_sends(self.KEY)
        self.assertEqual(headers, {"Authorization": f"Bearer {self.KEY}"})
        self.assertTrue(self._decide(headers).allowed)

    def test_the_sibling_route_is_accepted_too(self):
        """/get_server_info -- 507 x 401 on sb5f, the unrecorded sibling."""
        headers = self._headers_group_get_sends(self.KEY)
        self.assertTrue(self._decide(headers, path="/get_server_info").allowed)

    def test_a_wrong_key_is_still_refused(self):
        """Proves the accept above is the KEY and not merely a header."""
        headers = ak.auth_headers("some-other-key")
        self.assertFalse(self._decide(headers).allowed)

    def test_an_unkeyed_boot_sends_nothing_and_is_allowed(self):
        """Backward compatibility: no key configured anywhere -> allowed."""
        headers = self._headers_group_get_sends(None)
        self.assertEqual(headers, {})
        d = decide_request_auth(
            method="GET", path="/server_info", authorization_header=None,
            api_key=None, admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL)
        self.assertTrue(d.allowed)

    def test_health_needs_no_bearer_which_is_why_it_may_share_the_seam(self):
        """`decide_request_auth` allows /health* by PREFIX before any key is
        consulted -- so routing the health probes through the seam changes
        nothing for them and keeps the rule one rule."""
        d = decide_request_auth(
            method="GET", path="/health_generate", authorization_header=None,
            api_key=None, admin_api_key=self.KEY,
            auth_level=AuthLevel.ADMIN_OPTIONAL)
        self.assertTrue(d.allowed)

    def test_the_abort_route_is_the_third_sibling(self):
        """/abort_request is ADMIN_OPTIONAL and in FRONT_ADMIN_ROUTES: front-
        driven, so it needs the bearer exactly as the flip legs do."""
        self.assertIn("/abort_request", ak.FRONT_ADMIN_ROUTES)
        self.assertFalse(self._decide({}, path="/abort_request",
                                      method="POST").allowed)


class ThePassthroughMustNotLaunder(CustomTestCase):
    """The exclusion is a SAFETY property, and its absence would be the worse
    bug: the front listens unauthenticated, so a proxy that attached the
    front's admin bearer would hand any anonymous caller D's ADMIN_OPTIONAL
    routes -- including the body that carries `admin_api_key` itself, which is
    the leak #1275 fix 3 closed."""

    def test_the_proxies_do_not_attach_the_front_key(self):
        for name in Front.PASSTHROUGH_METHODS:
            src = inspect.getsource(getattr(Front, name))
            self.assertNotIn("auth_headers", src,
                             f"{name} proxies a client request and must not "
                             f"carry the front's admin credential")

    def test_a_proxied_request_carries_no_front_credential(self):
        """THE SAFETY PROPERTY ON THE WIRE, not a word in a comment.

        An earlier version of this test grepped the source for "laundering"
        and went red on prose while the code was right -- the #995 trap in its
        source-inspection form, and the third time this campaign has paid for
        it. What matters is that a PROXIED request leaves without the front's
        Authorization header even when the front holds a key.
        """
        sess = _CapturingSession()
        f = _front_with_key("boot-key-abc123", sess)
        f.awake = "D"
        f.groups = {"D": type("G", (), {"url": "http://127.0.0.1:30032",
                                        "name": "D"})()}
        req = type("R", (), {"path_qs": "/get_server_info"})()
        asyncio.run(f.handle_passthrough_get(req))
        self.assertEqual(len(sess.calls), 1)
        self.assertNotIn("headers", sess.calls[0][2],
                         "a client proxy attached the front's admin bearer -- "
                         "any anonymous caller could then read D's "
                         "ADMIN_OPTIONAL routes, including the key itself")


class TheFailureNamesItsCause(CustomTestCase):
    """sb5f's front log carried ONE line about this, and it said group D
    "published no reading" while D was answering 401 to everything."""

    def test_the_unreadable_line_interpolates_a_cause(self):
        src = inspect.getsource(Front._d_pool_reading)
        self.assertIn("D-POOL UNREADABLE (cause=%s)", src)
        self.assertIn("HTTP {status}", src)

    def test_a_401_is_named_as_an_auth_refusal_not_as_silence(self):
        src = inspect.getsource(Front._d_pool_reading)
        self.assertIn("401", src)
        self.assertIn("admin-key-file", src)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
