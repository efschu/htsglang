"""Regression tests for HTTP route introspection around `/v1/loads`.

`/v1/loads` is the only endpoint wired up via `app.include_router()`
(`http_server.py` does `app.include_router(v1_loads_router)`); every other
endpoint is a direct `@app.get`/`@app.post` on the top-level `app`. Some
FastAPI versions represent an `include_router()`-registered route as a
single aggregate route object (a private `_IncludedRouter`) that matches
its whole sub-tree but does not itself expose a per-request `.path`
attribute. `_get_fastapi_request_path()` in `sglang.srt.utils.common`
(used by the request-tracking Prometheus middleware, which runs before
routing on every request) used to assume every top-level route exposes
`.path`, so it crashed with:

    AttributeError: '_IncludedRouter' object has no attribute 'path'

turning every request to `/v1/loads` into a 500, independent of any other
server flags.

This file covers two things:

* `TestGetFastapiRequestPathStub` — a fastapi-version-independent unit test
  that simulates the exact `_IncludedRouter` shape (aggregate route, no
  `.path`, but a public `effective_route_contexts()`) so the regression is
  caught even when the installed fastapi predates the private class that
  originally triggered it.
* `TestV1LoadsEndToEnd` — an end-to-end `TestClient` request against a real
  app assembled the same way as `http_server.py` (`include_router` +
  `_get_fastapi_request_path` invoked from a pre-routing middleware),
  proving `GET /v1/loads` no longer 500s and returns a plausible body.
"""

import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.routing import Match

from sglang.srt.entrypoints.v1_loads import router as v1_loads_router
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.utils.common import _get_fastapi_request_path
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _make_request(*, app_routes, path: str) -> Request:
    """Build a minimal `Request` bound to a fake app exposing `.routes`."""
    fake_app = SimpleNamespace(routes=app_routes)
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "server": ("testserver", 80),
        "headers": [],
        "app": fake_app,
    }
    return Request(scope)


class _StubLeafRoute:
    """Stand-in for a fully-resolved leaf route (e.g. `_EffectiveRouteContext`)."""

    def __init__(self, path: str, *, full_match: bool):
        self.path = path
        self._full_match = full_match

    def matches(self, scope):
        return (Match.FULL if self._full_match else Match.NONE), {}


class _StubIncludedRouter:
    """Stand-in for FastAPI's private `_IncludedRouter`.

    Matches its whole sub-tree as a single aggregate (like the real class)
    but deliberately has no `.path` attribute, and exposes the leaf routes
    only through `effective_route_contexts()`.
    """

    def __init__(self, leaves, *, full_match: bool = True):
        self._leaves = leaves
        self._full_match = full_match

    def matches(self, scope):
        return (Match.FULL if self._full_match else Match.NONE), {}

    def effective_route_contexts(self):
        return self._leaves


class TestGetFastapiRequestPathStub(CustomTestCase):
    """Fastapi-version-independent reproduction of the `_IncludedRouter` shape."""

    def test_included_router_leaf_path_resolves_without_crash(self):
        leaf = _StubLeafRoute("/v1/loads", full_match=True)
        aggregate = _StubIncludedRouter([leaf])
        request = _make_request(app_routes=[aggregate], path="/v1/loads")

        # This used to raise `AttributeError: '_IncludedRouter' object has
        # no attribute 'path'` when `aggregate.path` was accessed directly.
        path, is_handled = _get_fastapi_request_path(request)

        self.assertEqual(path, "/v1/loads")
        self.assertTrue(is_handled)

    def test_included_router_non_matching_leaf_falls_back_to_raw_path(self):
        leaf = _StubLeafRoute("/v1/loads", full_match=False)
        aggregate = _StubIncludedRouter([leaf], full_match=False)
        request = _make_request(app_routes=[aggregate], path="/unmapped")

        path, is_handled = _get_fastapi_request_path(request)

        self.assertEqual(path, "/unmapped")
        self.assertFalse(is_handled)

    def test_directly_registered_route_unaffected(self):
        # Plain routes without `effective_route_contexts()` (the pre-existing
        # shape for every non-`include_router` endpoint) must resolve exactly
        # as before.
        route = _StubLeafRoute("/health", full_match=True)
        request = _make_request(app_routes=[route], path="/health")

        path, is_handled = _get_fastapi_request_path(request)

        self.assertEqual(path, "/health")
        self.assertTrue(is_handled)


class _FakeTokenizerManager:
    metrics_collector = None

    def __init__(self, loads):
        self._loads = loads

    async def get_loads(self, include=None, dp_rank=None):
        results = self._loads
        if dp_rank is not None:
            results = [load for load in results if load.dp_rank == dp_rank]
        return results


def _build_app() -> FastAPI:
    """Assemble a minimal app mirroring `http_server.py`'s wiring:
    a directly-registered route plus `/v1/loads` via `include_router()`,
    with a pre-routing middleware calling the fixed
    `_get_fastapi_request_path()` on every request (the exact call site
    that used to crash).
    """
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(v1_loads_router)

    @app.middleware("http")
    async def resolve_path_before_routing(request, call_next):
        # Exercises the fixed function in the same position as the real
        # Prometheus request-tracking middleware: before the router has
        # matched the request.
        _get_fastapi_request_path(request)
        return await call_next(request)

    return app


class TestV1LoadsEndToEnd(CustomTestCase):
    def setUp(self):
        self.app = _build_app()

    def test_get_v1_loads_returns_200_with_plausible_body(self):
        from sglang.srt.entrypoints import v1_loads

        fake_manager = _FakeTokenizerManager(
            [
                LoadSnapshot(
                    dp_rank=0,
                    num_running_reqs=2,
                    num_waiting_reqs=1,
                    num_total_tokens=128,
                )
            ]
        )
        self.app.dependency_overrides[v1_loads._get_tokenizer_manager] = (
            lambda: fake_manager
        )

        with TestClient(self.app) as client:
            response = client.get("/v1/loads")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("timestamp", body)
        self.assertIn("version", body)
        self.assertEqual(len(body["loads"]), 1)
        self.assertEqual(body["loads"][0]["dp_rank"], 0)
        self.assertEqual(body["loads"][0]["num_running_reqs"], 2)

    def test_directly_registered_route_still_200(self):
        with TestClient(self.app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
