"""
Unit tests for HTTP server admin auth.

Usage:
    python3 -m pytest test/registered/unit/utils/test_http_server_auth.py -v
"""

import unittest

from sglang.srt.utils.auth import (
    AuthLevel,
    _get_auth_level_from_app_and_scope,
    app_has_admin_force_endpoints,
    auth_level,
    decide_request_auth,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")


def _minimal_scope(path: str) -> dict:
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
    }


class TestHttpServerAdminAuth(unittest.TestCase):
    def _decide(
        self,
        *,
        method: str,
        path: str,
        authorization_header: str | None,
        api_key: str | None,
        admin_api_key: str | None,
        auth_level: AuthLevel,
    ):
        return decide_request_auth(
            method=method,
            path=path,
            authorization_header=authorization_header,
            api_key=api_key,
            admin_api_key=admin_api_key,
            auth_level=auth_level,
        )

    def test_no_keys_configured(self):
        # No keys configured -> NORMAL + ADMIN_OPTIONAL are open (legacy),
        # but ADMIN_FORCE must be rejected (403) explicitly.
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 403)

    def test_api_key_only(self):
        # api_key configured -> NORMAL requires api_key (legacy).
        self.assertFalse(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )

        # ADMIN_OPTIONAL requires api_key when only api_key is configured.
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer wrong",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        # ADMIN_FORCE must be rejected even if api_key is configured (403).
        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header="Bearer user",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 403)

    def test_admin_api_key_only(self):
        # admin_api_key only:
        # - normal endpoints open
        # - optional/force endpoints require admin_api_key
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )

        # Optional endpoints require admin_api_key when admin_api_key is configured.
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer admin",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header="Bearer admin",
            api_key=None,
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(d.allowed)

    def test_with_both_api_keys(self):
        # both api_key and admin_api_key configured:
        # - normal endpoints require api_key
        # - optional endpoints require admin_api_key (api_key is NOT accepted)
        # - force endpoints require admin_api_key
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        # Optional endpoints must require admin_api_key when both keys are configured.
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_force_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_force_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )

    def test_options_is_always_allowed(self):
        # CORS preflight should never be blocked.
        self.assertTrue(
            self._decide(
                method="OPTIONS",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )

    def test_health_and_metrics_are_always_allowed(self):
        # Health/metrics endpoints are always public by design, regardless of auth level / keys.
        combos = [
            dict(api_key=None, admin_api_key=None),
            dict(api_key="user", admin_api_key=None),
            dict(api_key=None, admin_api_key="admin"),
            dict(api_key="user", admin_api_key="admin"),
        ]
        paths_allowed = [
            "/health",
            "/health_generate",
            "/metrics",
            "/metrics/",
            "/metrics/prometheus",
        ]
        for keys in combos:
            for path in paths_allowed:
                self.assertTrue(
                    self._decide(
                        method="GET",
                        path=path,
                        authorization_header=None,
                        api_key=keys["api_key"],
                        admin_api_key=keys["admin_api_key"],
                        auth_level=AuthLevel.ADMIN_FORCE,
                    ).allowed,
                    msg=f"expected allowed for {path=} with {keys=}",
                )


class TestAuthLevelRouteIntrospection(unittest.TestCase):
    """`_get_auth_level_from_app_and_scope` / `app_has_admin_force_endpoints`
    walk `app.routes` looking for the decorated endpoint. Endpoints
    registered via `app.include_router()` (like `/v1/loads`) can be
    represented, on some FastAPI versions, as a single aggregate route that
    matches its whole sub-tree but exposes neither `.path` nor `.endpoint`
    directly (see the sibling `AttributeError` fix for
    `sglang.srt.utils.common._get_fastapi_request_path`). Before the fix,
    that silently made `@auth_level(...)`-decorated endpoints behind
    `include_router()` fall back to `AuthLevel.NORMAL` -- a silent
    privilege *widening*, not a crash, which is why it's covered
    separately here.
    """

    def _build_app(self):
        from fastapi import APIRouter, FastAPI

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        sub_router = APIRouter()

        @sub_router.get("/v1/loads")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def get_loads():
            return {"loads": []}

        app.include_router(sub_router)
        return app

    def test_included_router_endpoint_auth_level_is_detected(self):
        app = self._build_app()
        scope = _minimal_scope("/v1/loads")

        level = _get_auth_level_from_app_and_scope(app, scope)

        self.assertEqual(level, AuthLevel.ADMIN_FORCE)

    def test_included_router_endpoint_counts_toward_admin_force_check(self):
        app = self._build_app()

        self.assertTrue(app_has_admin_force_endpoints(app))

    def test_directly_registered_endpoint_still_defaults_to_normal(self):
        app = self._build_app()
        scope = _minimal_scope("/health")

        level = _get_auth_level_from_app_and_scope(app, scope)

        self.assertEqual(level, AuthLevel.NORMAL)


if __name__ == "__main__":
    unittest.main()
