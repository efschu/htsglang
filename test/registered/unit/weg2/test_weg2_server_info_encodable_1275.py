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
"""#1275 fix 4: /get_server_info must still ENCODE, not just redact.

THE REGRESSION WAS MINE. #1275 fix 3 replaced `**dataclasses.asdict(server_args)`
with `**server_args.redacted_dict()`, and `redacted_dict` built its dict with
`dict(vars(self))`. Those are NOT the same shape: `vars()` returns every
instance attribute (662 on this class), `asdict` returns only the DECLARED
FIELDS (661). The one extra is `model_config`, set in `__post_init__` -- and
the line directly above the one I edited said so:

    # server_args.model_config is not serializable but should be excluded by asdict.

So the endpoint began handing a live ModelConfig to the encoder:
    TypeError: 'torch.dtype' object is not iterable
    TypeError: vars() argument must have __dict__ attribute
sb4 0, sb5b 0, sb5c 468 (masked, 40 requests), sb5d 7196 -- and the front's
#915 D-pool read rides that endpoint, so d_pool_read_failed 1130 /
leg2_failures 1116 / 1142 requests NOT SERVED, with group D perfectly healthy.

WHY A UNIT TEST DID NOT CATCH IT, and what this file changes: fix 3's tests
asserted the redaction (`redacted_dict()["admin_api_key"] == REDACTED`) and the
call site (`"redacted_dict()" in source`). Both stayed green, because both
questions were about the SECRET. Nobody asked whether the result still
SERIALISED. This test drives the real FastAPI route through TestClient, so the
question it asks is the one the boot asks.
"""

import dataclasses
import unittest
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from sglang.srt.server_args import REDACTED, ServerArgs
from sglang.test.test_utils import CustomTestCase

SECRET = "s3cr3t-admin-key"


def bare_server_args():
    a = ServerArgs.__new__(ServerArgs)
    for f in dataclasses.fields(ServerArgs):
        d = f.default if f.default is not dataclasses.MISSING else (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        object.__setattr__(a, f.name, d)
    a.model_path = "/model"
    a.admin_api_key = SECRET
    a.api_key = None
    # the two shapes that blew up on sb5d
    a.dtype = torch.bfloat16                       # a DECLARED field holding a torch.dtype
    a.model_config = SimpleNamespace(              # a NON-field attribute vars() would leak
        dtype=torch.bfloat16, hf_config=object())
    return a


class TheEndpointStillEncodes(CustomTestCase):
    """RED-FIRST: on the fix-3 tree this returns 500, not 200."""

    def setUp(self):
        from sglang.srt.entrypoints import http_server

        self.sa = bare_server_args()
        self.prev = getattr(http_server, "_global_state", None)
        http_server._global_state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                server_args=self.sa,
                get_internal_state=self._internal,
            ),
            scheduler_info={"status": "ready"},
        )
        self.addCleanup(setattr, http_server, "_global_state", self.prev)
        self.client = TestClient(http_server.app, raise_server_exceptions=False)

    async def _internal(self):
        return [{"load": 0}]

    def _get(self):
        r = self.client.get("/get_server_info")
        return r

    def test_the_endpoint_returns_200_not_500(self):
        r = self._get()
        self.assertEqual(r.status_code, 200, r.text[:400])

    def test_the_admin_key_is_still_redacted_in_the_body(self):
        r = self._get()
        self.assertEqual(r.status_code, 200, r.text[:400])
        self.assertNotIn(SECRET, r.text)
        self.assertEqual(r.json()["admin_api_key"], REDACTED)

    def test_model_config_is_absent_exactly_as_asdict_had_it(self):
        """The non-field attribute that broke the encoder must not be in the
        body -- `asdict` never included it and the response never carried it."""
        r = self._get()
        self.assertEqual(r.status_code, 200, r.text[:400])
        self.assertNotIn("model_config", r.json())

    def test_the_dtype_field_renders_as_it_did_before(self):
        r = self._get()
        self.assertEqual(r.status_code, 200, r.text[:400])
        self.assertEqual(r.json()["dtype"], str(torch.bfloat16))

    def test_the_shape_is_the_asdict_shape(self):
        """Field-for-field parity with the pre-fix-3 serialisation."""
        body = self._get().json()
        for name in (f.name for f in dataclasses.fields(ServerArgs)):
            self.assertIn(name, body, f"{name} vanished from /get_server_info")


class TheCanFailGuard(CustomTestCase):
    """A type the encoder cannot render must make a TEST red, not the endpoint 500."""

    def test_redacted_dict_is_the_asdict_shape_and_nothing_else(self):
        sa = bare_server_args()
        self.assertEqual(
            set(sa.redacted_dict()), {f.name for f in dataclasses.fields(ServerArgs)},
            "redacted_dict drifted from the asdict field set -- the sb5d bug",
        )
        self.assertNotIn("model_config", sa.redacted_dict())

    def test_every_value_the_endpoint_would_return_is_encodable(self):
        """The guard proper: encode each field and NAME the one that fails,
        instead of discovering it as a 500 under load."""
        from fastapi.encoders import jsonable_encoder

        sa = bare_server_args()
        bad = []
        for k, v in sa.redacted_dict().items():
            try:
                jsonable_encoder(v)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{k}: {type(v).__name__} -> {type(e).__name__}")
        self.assertEqual(bad, [], f"unencodable fields would 500 the endpoint: {bad}")

    def test_the_guard_can_actually_fail(self):
        """Proof the check above is not vacuous."""
        from fastapi.encoders import jsonable_encoder

        with self.assertRaises(Exception):
            jsonable_encoder(SimpleNamespace(x=object()).__class__)  # a type, not an instance
            raise RuntimeError("unreachable")


class TheSiblings(CustomTestCase):
    def test_set_internal_state_pops_model_config_and_redacts(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.set_internal_state)
        self.assertIn("redacted_dict()", src)
        self.assertIn('server_args.pop("model_config", None)', src,
                      "the pop must stay -- it is free insurance now that "
                      "redacted_dict is asdict-based, and load-bearing if it ever is not")

    def test_the_grpc_path_uses_the_same_helper(self):
        import inspect

        from sglang.srt.entrypoints import grpc_bridge

        self.assertIn("self.server_args.redacted_dict()", inspect.getsource(grpc_bridge))


if __name__ == "__main__":
    unittest.main()
