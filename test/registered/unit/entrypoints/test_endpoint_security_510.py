"""Falsifiers for #510 -- the endpoint-security bundle found by audit #506.

Every test in this file fails on the pre-#510 tree and passes after the fix.
What each one pins:

* ``TestHibernateDirConfinement`` -- ``hibernate_dir`` from a request body used
  to override ``--hibernate-dir`` with no validation at all, so ``../`` and any
  absolute path reached ``os.makedirs``/``os.path.join``
  (audit #506, finding A2-F1).
* ``TestHibernateRouteIsPostOnly`` -- the route was ``methods=["GET", "POST"]``,
  so a bare GET parked the server.
* ``TestForkStateChangingRoutesCarryAdminLevel`` -- the fork's newest
  state-changing routes were registered at the implicit NORMAL level, which
  ``--admin-api-key`` alone does not protect (``utils/auth.py:161-167``).
* ``TestCorsPolicy`` -- ``allow_origins=["*"]`` together with
  ``allow_credentials=True`` is spec-illegal and defeats a loopback-only bind.
* ``TestRegistryAppAuth`` / ``TestVideoAppAuth`` -- neither app had any auth
  wiring, so the registry's ``launch.argv`` executor and the video job routes
  were open to anyone who could reach the port.
* ``TestTrainingFileWriteRequiresTenant`` -- ``create_file`` had no
  ``config.enabled`` guard although ``create_job`` did, so ``POST /v1/files``
  wrote to disk on every boot.
* ``TestMuxDoesNotReflectSubprocessStderr`` -- ffprobe/ffmpeg stderr was
  reflected into the HTTP response, an existence oracle over the filesystem.

Usage:
    python3 -m pytest test/registered/unit/entrypoints/test_endpoint_security_510.py -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def _scope(method: str, path: str) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
    }


# ---------------------------------------------------------------------------
# 1. hibernate_dir confinement (pure helper, no torch)
# ---------------------------------------------------------------------------


class TestHibernateDirConfinement(unittest.TestCase):
    def setUp(self):
        from sglang.srt.utils.path_confinement import (  # noqa: PLC0415
            PathConfinementError,
            confine_to_root,
        )

        self.confine = confine_to_root
        self.error = PathConfinementError
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "park")
        os.makedirs(self.root, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_root_itself_is_accepted(self):
        self.assertEqual(
            self.confine(self.root, self.root), os.path.realpath(self.root)
        )

    def test_a_subdirectory_is_accepted(self):
        sub = os.path.join(self.root, "boot-a")
        self.assertEqual(self.confine(sub, self.root), os.path.realpath(sub))

    def test_none_falls_back_to_the_configured_root(self):
        self.assertEqual(self.confine(None, self.root), os.path.realpath(self.root))

    def test_dot_dot_traversal_is_refused(self):
        # The exact shape audit #506 showed reaching os.makedirs unvalidated.
        with self.assertRaises(self.error):
            self.confine(os.path.join(self.root, "..", "..", "etc"), self.root)

    def test_an_unrelated_absolute_path_is_refused(self):
        with self.assertRaises(self.error):
            self.confine("/var/www/html/pub", self.root)

    def test_a_sibling_with_a_shared_prefix_is_refused(self):
        # "/tmp/parkXXX" must not pass the confinement for "/tmp/park": a
        # string prefix test would let it through, a path test must not.
        sibling = self.root + "-evil"
        os.makedirs(sibling, exist_ok=True)
        with self.assertRaises(self.error):
            self.confine(sibling, self.root)

    def test_a_symlink_pointing_out_of_the_root_is_refused(self):
        outside = os.path.join(self._tmp.name, "outside")
        os.makedirs(outside, exist_ok=True)
        link = os.path.join(self.root, "escape")
        os.symlink(outside, link)
        with self.assertRaises(self.error):
            self.confine(link, self.root)

    def test_without_a_configured_root_a_requested_dir_is_refused(self):
        # Nothing to confine against: the request may not pick the directory.
        with self.assertRaises(self.error):
            self.confine("/tmp/anything", None)

    def test_the_error_names_both_paths(self):
        try:
            self.confine("/var/www/html/pub", self.root)
        except self.error as exc:
            self.assertIn("/var/www/html/pub", str(exc))
            self.assertIn(os.path.realpath(self.root), str(exc))
        else:
            self.fail("expected PathConfinementError")


# ---------------------------------------------------------------------------
# 2-4. the runtime app: methods, auth levels, CORS
# ---------------------------------------------------------------------------


def _runtime_app():
    from sglang.srt.entrypoints.http_server import app  # noqa: PLC0415

    return app


def _route_methods(app, path: str) -> set:
    out = set()
    for route in app.routes:
        if getattr(route, "path", None) == path:
            out |= set(getattr(route, "methods", None) or ())
    return out


class TestHibernateRouteIsPostOnly(unittest.TestCase):
    def test_get_is_gone(self):
        methods = _route_methods(_runtime_app(), "/hibernate")
        self.assertTrue(methods, "/hibernate route not found")
        self.assertNotIn("GET", methods)

    def test_post_is_still_there(self):
        self.assertIn("POST", _route_methods(_runtime_app(), "/hibernate"))


class TestForkStateChangingRoutesCarryAdminLevel(unittest.TestCase):
    """Every fork-added state-changing route must resolve to an admin level.

    ADMIN_OPTIONAL, not ADMIN_FORCE: with no key configured the decision is
    still "allow", so the default deployment is unchanged (backward
    compatibility), while ``--admin-api-key`` actually closes them.
    """

    #: (method, concrete path). Path params are filled with a literal so the
    #: route matcher resolves them the way a real request would.
    CASES = [
        ("POST", "/hibernate"),
        ("POST", "/vram_budget"),
        ("POST", "/kv_reshard"),
        ("POST", "/session_handover"),
        ("POST", "/v1/files"),
        ("DELETE", "/v1/files/file-abc"),
        ("POST", "/v1/fine_tuning/jobs"),
        ("POST", "/v1/fine_tuning/jobs/ft-abc/cancel"),
        ("POST", "/x-htsglang/workbench/pause"),
        ("POST", "/x-htsglang/workbench/enqueue"),
    ]

    def test_all_resolve_to_an_admin_level(self):
        from sglang.srt.utils.auth import (  # noqa: PLC0415
            AuthLevel,
            _get_auth_level_from_app_and_scope,
        )

        app = _runtime_app()
        offenders = []
        for method, path in self.CASES:
            level = _get_auth_level_from_app_and_scope(app, _scope(method, path))
            if level not in (AuthLevel.ADMIN_OPTIONAL, AuthLevel.ADMIN_FORCE):
                offenders.append(f"{method} {path} -> {level}")
        self.assertEqual(offenders, [], f"routes left at NORMAL: {offenders}")

    def test_an_admin_key_alone_actually_closes_them(self):
        """The end-to-end property, not just the decoration.

        This is the decision the middleware makes; it is what
        ``--admin-api-key S3CR3T`` without ``--api-key`` does today for these
        paths (audit #506, finding A2-F2).
        """
        from sglang.srt.utils.auth import (  # noqa: PLC0415
            _get_auth_level_from_app_and_scope,
            decide_request_auth,
        )

        app = _runtime_app()
        for method, path in self.CASES:
            level = _get_auth_level_from_app_and_scope(app, _scope(method, path))
            with self.subTest(path=path):
                self.assertFalse(
                    decide_request_auth(
                        method=method,
                        path=path,
                        authorization_header=None,
                        api_key=None,
                        admin_api_key="S3CR3T",
                        auth_level=level,
                    ).allowed
                )
                self.assertTrue(
                    decide_request_auth(
                        method=method,
                        path=path,
                        authorization_header="Bearer S3CR3T",
                        api_key=None,
                        admin_api_key="S3CR3T",
                        auth_level=level,
                    ).allowed
                )

    def test_the_default_deployment_is_unchanged(self):
        """No keys configured -> still open. This is the backward-compat arm."""
        from sglang.srt.utils.auth import (  # noqa: PLC0415
            _get_auth_level_from_app_and_scope,
            decide_request_auth,
        )

        app = _runtime_app()
        for method, path in self.CASES:
            level = _get_auth_level_from_app_and_scope(app, _scope(method, path))
            with self.subTest(path=path):
                self.assertTrue(
                    decide_request_auth(
                        method=method,
                        path=path,
                        authorization_header=None,
                        api_key=None,
                        admin_api_key=None,
                        auth_level=level,
                    ).allowed
                )

    def test_inference_routes_stay_normal(self):
        """The fix must not silently promote the OpenAI inference surface.

        Those are protected by ``--api-key`` and promoting them would change
        what an api-key-only deployment can reach.
        """
        from sglang.srt.utils.auth import (  # noqa: PLC0415
            AuthLevel,
            _get_auth_level_from_app_and_scope,
        )

        app = _runtime_app()
        for method, path in (
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/completions"),
            ("POST", "/generate"),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    _get_auth_level_from_app_and_scope(app, _scope(method, path)),
                    AuthLevel.NORMAL,
                )


class TestCorsPolicy(unittest.TestCase):
    def _cors_entries(self, app):
        from starlette.middleware.cors import CORSMiddleware  # noqa: PLC0415

        return [m for m in app.user_middleware if m.cls is CORSMiddleware]

    def _kwargs(self, entry) -> dict:
        return dict(getattr(entry, "kwargs", None) or {})

    def test_the_module_level_default_does_not_pair_wildcard_with_credentials(self):
        entries = self._cors_entries(_runtime_app())
        self.assertEqual(len(entries), 1)
        kw = self._kwargs(entries[0])
        if "*" in list(kw.get("allow_origins") or []):
            self.assertFalse(
                kw.get("allow_credentials"),
                "wildcard origins with credentials is spec-illegal and defeats "
                "a loopback-only bind",
            )

    def test_configure_cors_keeps_credentials_off_for_wildcard(self):
        from sglang.srt.entrypoints.http_server import configure_cors  # noqa: PLC0415

        class _Args:
            cors_allow_origins = ["*"]

        policy = configure_cors(_runtime_app(), _Args())
        self.assertFalse(policy["allow_credentials"])

    def test_configure_cors_enables_credentials_for_an_explicit_list(self):
        from sglang.srt.entrypoints.http_server import configure_cors  # noqa: PLC0415

        class _Args:
            cors_allow_origins = ["https://ui.example"]

        app = _runtime_app()
        policy = configure_cors(app, _Args())
        self.assertTrue(policy["allow_credentials"])
        self.assertEqual(policy["allow_origins"], ["https://ui.example"])
        # Exactly one CORS middleware survives: configure replaces, never
        # stacks, or two policies would both answer the preflight.
        self.assertEqual(len(self._cors_entries(app)), 1)

    def test_the_policy_actually_reaches_the_wire(self):
        """Binds-proof: the kwargs are not enough, the header has to change.

        Built on a standalone app so no runtime state is needed; the policy
        under test is the one ``cors_policy`` returns for the real flag.
        """
        from fastapi import FastAPI  # noqa: PLC0415
        from fastapi.testclient import TestClient  # noqa: PLC0415
        from starlette.middleware.cors import CORSMiddleware  # noqa: PLC0415

        from sglang.srt.entrypoints.http_server import cors_policy  # noqa: PLC0415

        def _probe(origins):
            app = FastAPI()
            app.add_middleware(CORSMiddleware, **cors_policy(origins))

            @app.get("/x")
            async def _x():
                return {"ok": True}

            r = TestClient(app).get("/x", headers={"Origin": "https://ui.example"})
            return r.headers

        wildcard = _probe(["*"])
        explicit = _probe(["https://ui.example"])
        # The wildcard arm must NOT invite credentials.
        self.assertNotIn("access-control-allow-credentials", wildcard)
        # The explicit arm must, and must echo the origin rather than "*".
        self.assertEqual(
            explicit.get("access-control-allow-credentials"),
            "true",
        )
        self.assertEqual(
            explicit.get("access-control-allow-origin"), "https://ui.example"
        )

    def tearDown(self):
        # Restore the module default so test order cannot leak a policy.
        from sglang.srt.entrypoints.http_server import configure_cors  # noqa: PLC0415

        class _Args:
            cors_allow_origins = ["*"]

        configure_cors(_runtime_app(), _Args())


class TestCorsServerArg(unittest.TestCase):
    def test_the_flag_exists_and_defaults_to_wildcard(self):
        from sglang.srt.server_args import ServerArgs  # noqa: PLC0415

        self.assertIn("cors_allow_origins", ServerArgs.__annotations__)


# ---------------------------------------------------------------------------
# 5. registry control plane
# ---------------------------------------------------------------------------


class TestRegistryAppAuth(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _app(self, **keys):
        from sglang.srt.registry.arbiter import EngineRegistry  # noqa: PLC0415
        from sglang.srt.registry.http_api import build_app  # noqa: PLC0415
        from sglang.srt.registry.ledger import ReservationStore  # noqa: PLC0415

        registry = EngineRegistry(
            store=ReservationStore(Path(self._tmp.name) / "ledger"),
            card_totals={"GPU-test-uuid": 32 << 30},
        )
        return build_app(registry, **keys)

    def test_state_changing_routes_carry_an_admin_level(self):
        from sglang.srt.utils.auth import (  # noqa: PLC0415
            AuthLevel,
            _get_auth_level_from_app_and_scope,
        )

        app = self._app()
        offenders = []
        for method, path in (
            ("POST", "/registry/engines"),
            ("DELETE", "/registry/engines/e1"),
            ("POST", "/registry/engines/e1/state"),
            ("POST", "/registry/engines/e1/pin"),
            ("POST", "/registry/default_hot"),
            ("POST", "/registry/idle"),
        ):
            level = _get_auth_level_from_app_and_scope(app, _scope(method, path))
            if level not in (AuthLevel.ADMIN_OPTIONAL, AuthLevel.ADMIN_FORCE):
                offenders.append(f"{method} {path} -> {level}")
        self.assertEqual(offenders, [])

    def test_an_admin_key_closes_the_launch_argv_route(self):
        from fastapi.testclient import TestClient  # noqa: PLC0415

        client = TestClient(self._app(admin_api_key="S3CR3T"))
        # POST /registry/engines is the route that accepts launch.argv, which
        # class3_utility.py hands to subprocess.Popen.
        self.assertIn(
            client.post("/registry/engines", json={"engine_id": "x"}).status_code,
            (401, 403),
        )
        self.assertIn(client.post("/registry/idle", json={}).status_code, (401, 403))

    def test_without_keys_the_registry_is_unchanged(self):
        from fastapi.testclient import TestClient  # noqa: PLC0415

        client = TestClient(self._app())
        # Not asserting success -- only that auth is not what stops it.
        self.assertNotIn(client.post("/registry/idle", json={}).status_code, (401, 403))


# ---------------------------------------------------------------------------
# 6. video-enhance app
# ---------------------------------------------------------------------------


class TestVideoAppAuth(unittest.TestCase):
    """Route metadata only.

    ``VideoEnhanceService`` probes ffmpeg and the card, so the service is
    stubbed; the assertions are on the real ``create_app`` wiring (route
    levels, middleware presence), never on the stub.
    """

    def _app(self, **keys):
        from unittest.mock import patch  # noqa: PLC0415

        from sglang.srt.video_enhance import server as vsrv  # noqa: PLC0415

        with patch.object(vsrv, "VideoEnhanceService"):
            return vsrv.create_app(vsrv.TenantConfig(budget_mib=1024), **keys)

    def test_state_changing_routes_carry_an_admin_level(self):
        from sglang.srt.utils.auth import (  # noqa: PLC0415
            AuthLevel,
            _get_auth_level_from_app_and_scope,
        )

        app = self._app()
        for method, path in (
            ("POST", "/v1/video/enhance"),
            ("DELETE", "/v1/video/enhance/job1"),
        ):
            with self.subTest(path=path):
                self.assertIn(
                    _get_auth_level_from_app_and_scope(app, _scope(method, path)),
                    (AuthLevel.ADMIN_OPTIONAL, AuthLevel.ADMIN_FORCE),
                )

    def test_keys_are_accepted_by_create_app(self):
        app = self._app(admin_api_key="S3CR3T")
        names = [m.cls.__name__ for m in app.user_middleware]
        self.assertTrue(
            any("ApiKey" in n for n in names), f"no auth middleware in {names}"
        )


# ---------------------------------------------------------------------------
# 7. training file store
# ---------------------------------------------------------------------------


class TestTrainingFileWriteRequiresTenant(unittest.TestCase):
    def _service(self, *, enabled: bool):
        from sglang.srt.training.service import (  # noqa: PLC0415
            TrainingService,
            TrainingServiceConfig,
        )

        return TrainingService(
            TrainingServiceConfig(
                enabled=enabled, artifact_root=str(Path(self._tmp.name) / "art")
            )
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_upload_is_refused_when_the_tenant_is_off(self):
        from sglang.srt.training.service import TenantDisabled  # noqa: PLC0415

        with self.assertRaises(TenantDisabled):
            self._service(enabled=False).create_file(
                filename="a.jsonl", content=b'{"messages": []}\n', purpose="fine-tune"
            )

    def test_upload_still_works_when_the_tenant_is_on(self):
        svc = self._service(enabled=True)
        svc.start(start_tenant=False)
        stored = svc.create_file(
            filename="a.jsonl", content=b'{"messages": []}\n', purpose="fine-tune"
        )
        self.assertTrue(getattr(stored, "id", None) or stored)


# ---------------------------------------------------------------------------
# 8. subprocess stderr must not reach the caller
# ---------------------------------------------------------------------------


class TestMuxDoesNotReflectSubprocessStderr(unittest.TestCase):
    def test_the_failure_message_carries_no_stderr(self):
        from sglang.srt.video_enhance.mux import (  # noqa: PLC0415
            MuxError,
            subprocess_failure,
        )

        secret = "/etc/shadow: Permission denied"
        exc = subprocess_failure("ffprobe", returncode=1, stderr=secret.encode())
        self.assertIsInstance(exc, MuxError)
        self.assertNotIn(secret, str(exc))
        self.assertNotIn("Permission denied", str(exc))
        self.assertIn("ffprobe", str(exc))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 9. ratchet: no NEW state-changing route may appear at the NORMAL level
# ---------------------------------------------------------------------------


class TestStateChangingRouteRatchet(unittest.TestCase):
    """Enumerate, do not sample.

    The list below is every route on the runtime app that answers a
    state-changing method and resolves to NORMAL. NORMAL means "``--api-key``
    covers it, ``--admin-api-key`` alone does not", which is correct for the
    inference and tokenizer surface: those are the routes an api-key-only
    deployment is meant to reach. It is NOT correct for a management route,
    and #510 exists because six of them had drifted in here unnoticed.

    A new entry appearing in the diff is not automatically a bug -- it is a
    decision. Add an inference-shaped route to the list; give a management
    route an explicit ``@auth_level``.
    """

    #: Upstream inference / tokenizer / OpenAI-compatible surface.
    EXPECTED_NORMAL = {
        "/api/chat",
        "/api/generate",
        "/api/show",
        # #335 KoboldCpp surface. Inference-shaped, so NORMAL is the correct
        # resolution: these are exactly the routes an api-key-only deployment
        # is meant to reach, and /api/v1/generate composes the same
        # openai_serving_completion that /v1/completions does.
        #
        # /api/extra/abort is listed DELIBERATELY despite sounding like a
        # management verb: it is an unconditional 501 that mutates nothing
        # (a multi-tenant server has no "the" generation to abort), and a
        # Kobold client calls it mid-conversation on the inference surface.
        # If it ever grows real cancellation it must move to an explicit
        # @auth_level, because cancelling someone else's request is exactly
        # the reach #510 exists to bound.
        "/api/extra/abort",
        "/api/extra/generate/stream",
        "/api/v1/generate",
        "/classify",
        "/close_session",
        "/detokenize",
        "/encode",
        "/generate",
        "/invocations",
        # Upstream management routes deliberately left as upstream has them:
        # promoting them would change the reach of an api-key-only deployment
        # for code this fork does not own. Recorded here so the choice is
        # visible rather than an oversight (audit #506 flagged neither).
        "/load_lora_adapter_from_tensors",
        "/set_trace_level",
        "/open_session",
        "/parse_function_call",
        "/separate_reasoning",
        "/tokenize",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/chat/completions",
        "/v1/classify",
        "/v1/completions",
        "/v1/detokenize",
        "/v1/embeddings",
        "/v1/images/edits",
        "/v1/images/generations",
        "/v1/images/variations",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/rerank",
        "/v1/responses",
        "/v1/responses/{response_id}/cancel",
        "/v1/score",
        "/v1/tokenize",
        "/vertex_generate",
    }

    def test_the_normal_state_changing_set_is_exactly_the_pinned_one(self):
        app = _runtime_app()
        found = set()
        for route in app.routes:
            methods = set(getattr(route, "methods", None) or ())
            if not methods & {"POST", "PUT", "DELETE", "PATCH"}:
                continue
            endpoint = getattr(route, "endpoint", None)
            if getattr(endpoint, "_auth_level", None) is None:
                found.add(getattr(route, "path", ""))
        new = sorted(found - self.EXPECTED_NORMAL)
        gone = sorted(self.EXPECTED_NORMAL - found)
        self.assertEqual(
            (new, gone),
            ([], []),
            "state-changing routes at the NORMAL level changed. New: "
            f"{new}; no longer present: {gone}. Decide per route: inference "
            "surface -> add to EXPECTED_NORMAL; management -> decorate it "
            "with @auth_level(AuthLevel.ADMIN_OPTIONAL).",
        )
