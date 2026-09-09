# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1288: loopback peers reach the ADMIN routes without a bearer.

THE MEASUREMENT. #1275 fix 5 (`ea3419fd84`) moved `/server_info` and
`/get_server_info` to `ADMIN_OPTIONAL`, which on a keyed boot means "require
the admin key". `Front.rpc` learned the bearer in that ticket; the front's OWN
reads of those routes did not, because they were plain `session.get` calls
written at their point of use. On boot weg2sb5f (`4f762260ba`), from group D's
own access log:

    GET /server_info      401 x 180   200 x 1
    GET /get_server_info  401 x 507   200 x 2

`/get_server_info` was NOT named in the boot record -- it is the sibling this
sweep found, and it failed 2.8x more often. Consequences, both silent: the
#915 D-pool admission gate went OFF by name (`d_pool_read_failed` 85 -> 207
across the long-prompt arm) and priced nothing, and `_draft_terms` fell to
`accept_src=none` with both draft counters at 0 for the whole boot.

THE SHAPE OF THE FIX IS A USER DECISION (2026-09-09): trust LOOPBACK in the
auth decision itself rather than plumbing a second copy of the key into every
internal caller. Every port of this deployment binds 127.0.0.1
(`launcher.py:1527`), and the processes that drive the admin routes are on
this host by construction. The alternative -- one bearer per caller -- is
second bookkeeping beside the transport's own fact, and #1288 IS what that
bookkeeping costs when one caller is missed. sb5d's 503s (#1282, a dtype 500
on the same read) and sb5f's (401 on the same read) are the same failure shape
with two different producers: a read with no fallback whose consumer turns
"unknown" into a verdict.

THE ONE THING THAT MAKES THIS SAFE OR UNSAFE is where the peer comes from.
`scope["client"]` is the address the kernel accepted the connection from.
`X-Forwarded-For`, `X-Real-IP` and `Host` are strings the CALLER chooses, and
honouring them would let any remote request claim loopback and take the admin
routes. There is no reverse proxy in front of these ports, so there is also
nothing legitimate to learn from them. `ForgedHeadersDoNotGrantTrust` below
pins both directions of that.

THE COST, stated rather than waved past: this reopens the browser-CSRF vector
`http_server.cors_policy` describes (audit #506/#510, finding A2-F3) -- a page
open in a browser ON THIS HOST could POST to the admin routes with no key. The
assumption that makes it acceptable is that this LXC runs no browser. If that
ever changes, this trust must go with it.
"""

import inspect
import unittest

from sglang.srt.utils import auth as auth_mod
from sglang.srt.utils.auth import (
    AuthLevel,
    decide_request_auth,
    peer_is_loopback,
)
from sglang.srt.weg2 import admin_key as ak
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

KEY = "boot-key-abc123"


def _decide(peer, authz=None, path="/server_info", method="GET",
            level=AuthLevel.ADMIN_OPTIONAL, admin_api_key=KEY, api_key=None):
    return decide_request_auth(
        method=method, path=path, authorization_header=authz,
        api_key=api_key, admin_api_key=admin_api_key, auth_level=level,
        peer_host=peer,
    )


class TheMeasuredRefusal(CustomTestCase):
    """RED-FIRST at `4f762260ba`: `peer_host` does not exist, and a bearer-less
    loopback GET of `/server_info` is refused -- the measured 401."""

    def test_red_first_a_bearerless_loopback_read_is_now_allowed(self):
        """THE fix, as the exact request group D refused 180 times."""
        self.assertTrue(_decide("127.0.0.1").allowed)

    def test_the_sibling_route_too(self):
        """/get_server_info -- 507 x 401 on sb5f, unrecorded until this sweep."""
        self.assertTrue(_decide("127.0.0.1", path="/get_server_info").allowed)

    def test_the_flip_legs_too(self):
        """The four routes the front drives every flip with. If these ever
        401 the flip dies at the quiesce, which is what #1275 existed to
        prevent -- now prevented by the transport instead of by a header."""
        for path in ak.FRONT_ADMIN_ROUTES:
            self.assertTrue(_decide("127.0.0.1", path=path, method="POST").allowed,
                            f"{path} refused a loopback caller")

    def test_ipv6_loopback_and_the_mapped_form(self):
        for peer in ("::1", "::ffff:127.0.0.1"):
            self.assertTrue(_decide(peer).allowed, peer)

    def test_admin_force_is_also_reachable_from_loopback(self):
        self.assertTrue(_decide("127.0.0.1", level=AuthLevel.ADMIN_FORCE).allowed)

    def test_admin_force_without_a_key_is_still_reachable_from_loopback(self):
        """ADMIN_FORCE with no key configured is a 403 for everyone else; the
        local operator is not everyone else."""
        self.assertTrue(_decide("127.0.0.1", level=AuthLevel.ADMIN_FORCE,
                                admin_api_key=None).allowed)


class RemotePeersAreUnchanged(CustomTestCase):
    """Defence in depth if anyone ever passes `--host 0.0.0.0`."""

    def test_a_remote_peer_without_a_bearer_is_refused(self):
        d = _decide("10.0.0.5")
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 401)

    def test_a_remote_peer_with_the_bearer_is_allowed(self):
        self.assertTrue(
            _decide("10.0.0.5", authz=f"Bearer {KEY}").allowed)

    def test_a_remote_peer_with_the_wrong_bearer_is_refused(self):
        self.assertFalse(_decide("10.0.0.5", authz="Bearer nope").allowed)

    def test_an_unknown_peer_is_not_trusted(self):
        """MUTANT: `None` peer (no `client` in the scope, a non-network
        transport). Unknown must never read as trusted -- that is the
        naive-truthy shape this campaign keeps paying for."""
        self.assertFalse(_decide(None).allowed)
        self.assertFalse(_decide("").allowed)

    def test_mutant_a_prefix_match_would_trust_the_wrong_network(self):
        """MUTANT: had the check been `peer.startswith("127.")` or a substring
        test, these would pass. They must not."""
        for peer in ("127.0.0.1.evil.com", "10.0.0.1", "192.168.0.101",
                     "1127.0.0.1", "127.0.0.10"):
            self.assertFalse(peer_is_loopback(peer), peer)
        # ...while the real loopback forms still do
        for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(peer_is_loopback(peer), peer)


class ForgedHeadersDoNotGrantTrust(CustomTestCase):
    """THE LOAD-BEARING PROPERTY. The peer is a transport fact, never a header.

    Both directions are pinned, because getting either one wrong is a
    different bug: honouring the header grants admin to any remote caller,
    and letting a header REVOKE loopback would re-break the front.
    """

    def test_a_loopback_peer_claiming_a_remote_forwarded_for_still_passes(self):
        """The header is irrelevant, not merely outvoted."""
        self.assertTrue(
            _decide("127.0.0.1", authz=None).allowed,
            "a loopback peer must pass regardless of what headers say")

    def test_a_remote_peer_claiming_loopback_still_needs_the_bearer(self):
        """MUTANT: the whole attack. X-Forwarded-For: 127.0.0.1 from 10.0.0.5."""
        self.assertFalse(_decide("10.0.0.5").allowed)

    def test_the_decision_function_takes_no_header_but_authorization(self):
        """Structural: `decide_request_auth` cannot consult a forwarding
        header because it is never given one."""
        params = set(inspect.signature(decide_request_auth).parameters)
        self.assertIn("peer_host", params)
        for forbidden in ("headers", "x_forwarded_for", "x_real_ip", "host"):
            self.assertNotIn(forbidden, params)

    def test_the_middleware_reads_the_scope_not_the_headers(self):
        """The one place the peer is derived. `scope["client"]` is the address
        the kernel accepted; anything from `request.headers` is caller-chosen."""
        src = inspect.getsource(auth_mod.add_api_key_middleware)
        self.assertIn('scope.get("client")', src)
        self.assertIn("peer_host=peer_host", src)
        # NOT a prose scan: the comment beside this code NAMES the headers in
        # order to say they are not used, so grepping the window for their
        # spelling fails on the explanation while the code is right -- the
        # #995 trap in its source-inspection form, and the third time this
        # branch has paid for it. What must not exist is a READ of them, and
        # `test_no_forwarding_header_is_read_anywhere_in_the_module` asserts
        # exactly that, module-wide.

    def test_no_forwarding_header_is_read_anywhere_in_the_module(self):
        src = inspect.getsource(auth_mod)
        for forbidden in ("X-Forwarded-For", "X-Real-IP"):
            # The docstrings name them to say they are NOT used; a
            # `headers.get(...)` of them is what must not exist.
            self.assertNotIn(f'headers.get("{forbidden}")', src)
            self.assertNotIn(f"headers.get('{forbidden}')", src)


class NothingElseMoved(CustomTestCase):
    """Key generation, redaction and the K-2 guards stay exactly as they are."""

    def test_normal_routes_are_untouched_by_the_loopback_trust(self):
        """A NORMAL route with an api_key set still needs it, from anywhere:
        the trust is scoped to the ADMIN levels, not to the whole server."""
        d = decide_request_auth(
            method="POST", path="/generate", authorization_header=None,
            api_key="user-key", admin_api_key=KEY,
            auth_level=AuthLevel.NORMAL, peer_host="127.0.0.1")
        self.assertFalse(d.allowed,
                         "loopback must not open the user-facing api_key")

    def test_health_stays_open_to_everyone(self):
        self.assertTrue(_decide("10.0.0.5", path="/health_generate").allowed)

    def test_the_key_helpers_are_unchanged(self):
        self.assertEqual(ak.auth_headers(None), {})
        self.assertEqual(ak.auth_headers("K")["Authorization"], "Bearer K")
        self.assertNotIn("s3cr3t", ak.redact("s3cr3t"))

    def test_an_unkeyed_boot_still_allows_everyone(self):
        d = decide_request_auth(
            method="GET", path="/server_info", authorization_header=None,
            api_key=None, admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL, peer_host="10.0.0.5")
        self.assertTrue(d.allowed)

    def test_the_default_peer_is_absent_not_loopback(self):
        """Every existing caller that does not pass `peer_host` must keep the
        OLD behaviour -- so the default cannot be a trusted value."""
        d = decide_request_auth(
            method="GET", path="/server_info", authorization_header=None,
            api_key=None, admin_api_key=KEY,
            auth_level=AuthLevel.ADMIN_OPTIONAL)
        self.assertFalse(d.allowed)


class ThePolicyIsAnnounced(CustomTestCase):
    """A trust decision visible only in the source is one nobody audits."""

    def test_one_grepable_line_per_process(self):
        src = inspect.getsource(auth_mod.add_api_key_middleware)
        self.assertIn("WEG2-AUTH loopback=trusted bearer=required-for-remote",
                      src)

    def test_the_line_says_where_the_peer_comes_from(self):
        src = inspect.getsource(auth_mod.add_api_key_middleware)
        i = src.find("WEG2-AUTH")
        self.assertIn("ASGI scope", src[i:i + 400])

    def test_the_browser_csrf_cost_is_written_down(self):
        """Named in the code, not only in a commit body."""
        src = inspect.getsource(auth_mod)
        self.assertIn("CSRF", src)
        self.assertIn("#506", src)


class TheFrontSendsNoHeaderOnItsInternalReads(CustomTestCase):
    """ONE JOB, ONE MOVER: the transport carries the trust, so the callers
    carry nothing. A bearer added here would be the second bookkeeping this
    ticket exists to remove."""

    def test_the_pool_read_sends_no_auth_header(self):
        from sglang.srt.weg2.front import Front

        src = inspect.getsource(Front._d_pool_reading)
        self.assertNotIn("auth_headers", src)
        self.assertIn("/server_info", src)

    def test_the_draft_terms_read_sends_no_auth_header(self):
        from sglang.srt.weg2.front import Front

        self.assertNotIn("auth_headers",
                         inspect.getsource(Front._draft_terms))

    def test_the_flip_rpc_KEEPS_its_bearer(self):
        """NOT removed: `Front.rpc` is the one caller that must keep working
        if the bind ever changes, and #1275's guard still owns it."""
        from sglang.srt.weg2.front import Front

        self.assertIn("auth_headers", inspect.getsource(Front._rpc_attempt))

    def test_the_unreadable_refusal_names_its_cause(self):
        """sb5f's front log carried ONE line about this and it said group D
        "published no reading", while D was answering 401 to everything."""
        from sglang.srt.weg2.front import Front

        src = inspect.getsource(Front._d_pool_reading)
        self.assertIn("D-POOL UNREADABLE (cause=%s)", src)
        self.assertIn("HTTP {status}", src)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
