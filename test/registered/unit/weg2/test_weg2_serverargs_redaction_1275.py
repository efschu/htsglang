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
"""#1275 fix 3: the server never logs -- or serves -- its own admin key.

K-2, measured on boot weg2sb5b: each GROUP dumps its own ServerArgs at startup
through ``logger.info(f"{server_args=}")``, so ``admin_api_key='<key>'``
appeared once in P.log and once in D.log (front.log clean).

AND A WORSE ONE FOUND WHILE ENUMERATING THE CHANNELS RATHER THAN THE SITES:
``/get_server_info`` carries NO ``@auth_level``, so under an admin-key-only boot
``decide_request_auth``'s NORMAL branch returns ``allowed=True`` -- and the
handler returned ``**dataclasses.asdict(server_args)``, i.e. it handed
``admin_api_key`` to any unauthenticated caller who could reach the port. A log
file is durable but local; that endpoint is live and remote, and it gives away
the exact credential every ADMIN_OPTIONAL route is gated on.

TWO KINDS OF CHANNEL, TWO MECHANISMS, and the distinction is the whole point:
``__repr__`` covers every f-string/print/log by construction, but
``dataclasses.asdict`` and ``vars()`` DO NOT GO THROUGH IT. Those need
``redacted_dict()`` explicitly, at each serialising site.
"""

import dataclasses
import inspect
import re
import unittest

from sglang.srt.server_args import (
    REDACTED,
    SECRET_FIELD_NAMES,
    ServerArgs,
    is_secret_field,
)
from sglang.test.test_utils import CustomTestCase

SECRET = "s3cr3t-admin-key-do-not-log"


def bare_server_args(**over):
    """A ServerArgs without __post_init__ -- hermetic, no accelerator needed."""
    a = ServerArgs.__new__(ServerArgs)
    for f in dataclasses.fields(ServerArgs):
        if f.default is not dataclasses.MISSING:
            d = f.default
        elif f.default_factory is not dataclasses.MISSING:
            d = f.default_factory()
        else:
            d = None
        object.__setattr__(a, f.name, d)
    a.model_path = "/model"
    for k, v in over.items():
        setattr(a, k, v)
    return a


class TheStartupLineIsRedacted(CustomTestCase):
    """RED-FIRST: this is the exact string sb5b wrote into P.log and D.log."""

    def test_the_repr_the_startup_log_prints_has_no_key_bytes(self):
        a = bare_server_args(admin_api_key=SECRET, api_key="another-secret")
        line = f"{a=}"  # verbatim the engine.py / scheduler.py form
        self.assertNotIn(SECRET, line)
        self.assertNotIn("another-secret", line)
        self.assertIn(REDACTED, line)

    def test_the_named_helper_exists_and_backs_the_repr(self):
        a = bare_server_args(admin_api_key=SECRET)
        self.assertEqual(a.redacted_repr(), repr(a))
        self.assertNotIn(SECRET, a.redacted_repr())

    def test_the_shipped_values_are_untouched(self):
        """Redaction is on the rendering, never on the object."""
        a = bare_server_args(admin_api_key=SECRET, api_key="k2")
        repr(a)
        a.redacted_dict()
        self.assertEqual(a.admin_api_key, SECRET)
        self.assertEqual(a.api_key, "k2")

    def test_an_unset_key_is_not_rendered_as_redacted(self):
        """None must stay None, or a keyless boot reads as if it had one."""
        a = bare_server_args()
        self.assertIsNone(a.redacted_dict()["admin_api_key"])
        self.assertIn("admin_api_key=None", repr(a))


class TheSerialisingChannels(CustomTestCase):
    """asdict/vars do NOT inherit __repr__ -- each needs the explicit call."""

    def test_redacted_dict_hides_the_secrets(self):
        a = bare_server_args(admin_api_key=SECRET, api_key="k2")
        d = a.redacted_dict()
        self.assertEqual(d["admin_api_key"], REDACTED)
        self.assertEqual(d["api_key"], REDACTED)

    def test_get_server_info_uses_it(self):
        from sglang.srt.entrypoints import http_server

        src = inspect.getsource(http_server)
        i = src.index('"/get_server_info"')
        window = src[i:i + 2500]
        self.assertIn("redacted_dict()", window)
        self.assertNotIn("dataclasses.asdict(server_args)", window,
                         "/get_server_info still serialises the raw dataclass")

    def test_set_internal_state_uses_it(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.set_internal_state)
        self.assertIn("redacted_dict()", src)
        self.assertNotIn("dict(vars(get_server_args()))", src)

    def test_the_grpc_bridge_uses_it(self):
        from sglang.srt.entrypoints import grpc_bridge

        src = inspect.getsource(grpc_bridge)
        self.assertNotIn("dataclasses.asdict(self.server_args)", src)
        self.assertIn("self.server_args.redacted_dict()", src)


class TheClassifierIsPrecise(CustomTestCase):
    """The naive `*_token` rule would have wrecked every diagnostic."""

    def test_exactly_the_real_secrets_are_selected(self):
        names = [f.name for f in dataclasses.fields(ServerArgs)]
        selected = sorted(n for n in names if is_secret_field(n))
        self.assertEqual(
            selected, ["admin_api_key", "api_key", "ssl_keyfile_password"])

    def test_token_counts_are_not_redacted(self):
        """MEASURED: a literal `*_token` rule matches 16 token COUNTS on this
        ServerArgs. Redacting them hides no secret and destroys the diagnostic
        value of every boot log and every /get_server_info response."""
        for n in ("max_total_tokens", "max_prefill_tokens",
                  "num_reserved_decode_tokens", "bucket_time_to_first_token",
                  "kt_max_deferred_experts_per_token", "speculative_token_map",
                  "uneven_token_vector"):
            self.assertFalse(is_secret_field(n), f"{n} must stay visible")

    def test_a_path_beside_a_credential_stays_visible(self):
        self.assertFalse(is_secret_field("ssl_keyfile"))
        self.assertTrue(is_secret_field("ssl_keyfile_password"))

    def test_a_future_credential_is_covered_by_the_pattern(self):
        for n in ("upstream_api_key", "webhook_secret", "registry_password",
                  "vault_credential"):
            self.assertTrue(is_secret_field(n), n)

    def test_the_explicit_names_are_all_real_fields(self):
        names = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertTrue(SECRET_FIELD_NAMES <= names,
                        f"stale entries: {SECRET_FIELD_NAMES - names}")


class NoLogSiteRendersARawServerArgs(CustomTestCase):
    """The enumeration, machine-checked -- the fix-2 lesson applied here.

    Every site that renders a ServerArgs into text goes through the repr, which
    is redacted. This asserts none of them reaches around it with asdict/vars.
    """

    def test_the_known_dump_sites_use_the_repr(self):
        from sglang.srt.entrypoints import engine

        src = inspect.getsource(engine)
        self.assertGreaterEqual(src.count('f"{server_args=}"'), 2)

    def test_no_module_logs_an_asdict_of_server_args(self):
        import pathlib

        root = pathlib.Path(__file__)
        while root.name != "python" and root.parent != root:
            root = root.parent
        offenders = []
        for f in (root / "sglang" / "srt").rglob("*.py"):
            body = f.read_text(errors="replace")
            for m in re.finditer(
                r"(logger\.\w+|print)\([^)]*(asdict\(\s*(self\.)?server_args|"
                r"vars\(\s*(self\.)?server_args)", body):
                offenders.append(f"{f.name}:{body[:m.start()].count(chr(10))+1}")
        self.assertEqual(offenders, [], f"raw ServerArgs serialised into a log: {offenders}")


if __name__ == "__main__":
    unittest.main()
