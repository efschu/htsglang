"""Hermetic tests for the model-aware Anthropic split proxy.

Two mock servers stand in for api.anthropic.com and the local htsglang front,
so the whole routing decision is exercised without a GPU, a network, or a
credential. Each mock records what it received; the assertions are about which
one got the request and what the body looked like when it arrived.

``RouterTestCase`` and ``ThinkingOnDefaultTestCase`` build their app with
``local_wait_s=0``, i.e. the hold buffer explicitly OFF: they are exercising
routing/shim/policy behaviour, not the buffer, and a backend-unreachable
scenario there must fail immediately, not hang for the 300s production
default. The buffer itself is ``LocalBackendBufferTestCase`` below, which
picks small wait/poll values so it stays fast.
"""

import asyncio
import contextlib
import json
import os
import tempfile
import time
import unittest

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer, unused_port

from sglang.srt.entrypoints.anthropic.router import (
    NOTHINK_ALIAS_SUFFIX,
    STATS_PATH,
    THINKING_ALIAS_SUFFIX,
    create_app,
)

LOCAL_MODEL = "Qwen3.6-27B"
THINKING_ALIAS = LOCAL_MODEL + THINKING_ALIAS_SUFFIX
NOTHINK_ALIAS = LOCAL_MODEL + NOTHINK_ALIAS_SUFFIX
REMOTE_MODEL = "claude-opus-4-6"


def _make_backend(name):
    """A mock endpoint that records requests and can stream or fail."""
    state = {"requests": [], "name": name, "fail": False}

    async def handler(request):
        raw = await request.read()
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        state["requests"].append(
            {
                "path": request.path,
                "query": request.query_string,
                "method": request.method,
                "headers": dict(request.headers),
                "body": body,
                "raw": raw,
            }
        )
        if state["fail"]:
            return web.json_response(
                {"type": "error", "error": {"type": "overloaded_error"}}, status=529
            )
        if body and body.get("stream"):
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            for event in ("message_start", "content_block_delta", "message_stop"):
                await resp.write(
                    f"event: {event}\ndata: "
                    f'{{"type":"{event}","backend":"{name}"}}\n\n'.encode()
                )
            await resp.write_eof()
            return resp
        return web.json_response({"backend": name, "type": "message"})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    return app, state


class RouterTestCase(AioHTTPTestCase):
    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        return create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            # Off: this suite is about routing/shim behaviour, not the hold
            # buffer (see LocalBackendBufferTestCase). With the buffer's
            # generous 300s production default, a closed local_server would
            # hang the unreachable-backend test instead of failing fast.
            local_wait_s=0,
        )

    async def tearDownAsync(self):
        await self.upstream_server.close()
        await self.local_server.close()
        await super().tearDownAsync()

    def _body(self, model, **overrides):
        body = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        body.update(overrides)
        return body

    # ---------- routing ----------

    async def test_local_model_goes_to_the_local_front(self):
        resp = await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "local")
        self.assertEqual(len(self.local["requests"]), 1)
        self.assertEqual(self.upstream["requests"], [])

    async def test_other_model_goes_upstream(self):
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "upstream")
        self.assertEqual(len(self.upstream["requests"]), 1)
        self.assertEqual(self.local["requests"], [])

    async def test_body_without_a_model_goes_upstream(self):
        """A non-Messages call (no model field) must not be captured locally."""
        resp = await self.client.get("/v1/models")
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.upstream["requests"]), 1)
        self.assertEqual(self.local["requests"], [])

    async def test_path_and_query_are_preserved(self):
        await self.client.post(
            "/v1/messages/count_tokens?beta=true", json=self._body(LOCAL_MODEL)
        )
        got = self.local["requests"][0]
        self.assertEqual(got["path"], "/v1/messages/count_tokens")
        self.assertEqual(got["query"], "beta=true")

    async def test_auth_headers_are_forwarded_untouched(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(REMOTE_MODEL),
            headers={
                "Authorization": "Bearer sk-test-token",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "tools-2024-04-04",
            },
        )
        headers = self.upstream["requests"][0]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-test-token")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["anthropic-beta"], "tools-2024-04-04")

    async def test_absent_accept_encoding_becomes_identity(self):
        """Never hand back a content coding the client did not advertise.

        The response body is forwarded without decompression, so aiohttp's
        default ``Accept-Encoding: gzip, deflate`` would have made the proxy
        return gzip to a client that never asked for it.
        """
        await self.client.post(
            "/v1/messages",
            json=self._body(REMOTE_MODEL),
            skip_auto_headers=["Accept-Encoding"],
        )
        headers = self.upstream["requests"][0]["headers"]
        self.assertEqual(headers["Accept-Encoding"], "identity")

    async def test_client_accept_encoding_is_forwarded_untouched(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(REMOTE_MODEL),
            headers={"Accept-Encoding": "gzip"},
        )
        headers = self.upstream["requests"][0]["headers"]
        self.assertEqual(headers["Accept-Encoding"], "gzip")

    async def test_credentials_are_never_logged(self):
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.router", level="DEBUG"
        ) as log:
            await self.client.post(
                "/v1/messages",
                json=self._body(REMOTE_MODEL),
                headers={"Authorization": "Bearer sk-secret-value"},
            )
        joined = "\n".join(log.output)
        self.assertNotIn("sk-secret-value", joined)
        self.assertNotIn("Authorization", joined)

    # ---------- the thinking shim ----------

    async def test_absent_thinking_is_filled_in_for_the_local_front(self):
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(
            self.local["requests"][0]["body"]["thinking"], {"type": "disabled"}
        )

    async def test_explicit_thinking_is_overridden_to_disabled(self):
        """The plain local id is ALWAYS the no-thinking arm.

        Claude Code attaches its own thinking config to subagent requests;
        honoring it made the local model think despite the no-thinking policy
        (live incident 2026-08-04: "Thought for 14s" + stray </think> in
        text). The -think alias is the only route to the thinking arm.
        """
        await self.client.post(
            "/v1/messages",
            json=self._body(
                LOCAL_MODEL, thinking={"type": "enabled", "budget_tokens": 1024}
            ),
        )
        self.assertEqual(
            self.local["requests"][0]["body"]["thinking"],
            {"type": "disabled"},
        )

    async def test_upstream_bodies_are_byte_identical(self):
        """The shim is local-only: Anthropic must see exactly what was sent."""
        body = self._body(REMOTE_MODEL)
        await self.client.post("/v1/messages", json=body)
        self.assertNotIn("thinking", self.upstream["requests"][0]["body"])

    async def test_shim_can_be_switched_off(self):
        app = create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            apply_shim=False,
        )
        server = TestServer(app)
        await server.start_server()
        try:
            from aiohttp.test_utils import TestClient

            async with TestClient(server) as client:
                await client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        finally:
            await server.close()
        self.assertNotIn("thinking", self.local["requests"][0]["body"])

    # ---------- the thinking alias ----------

    async def test_thinking_alias_reaches_the_local_front(self):
        resp = await self.client.post("/v1/messages", json=self._body(THINKING_ALIAS))
        self.assertEqual((await resp.json())["backend"], "local")
        self.assertEqual(self.upstream["requests"], [])

    async def test_thinking_alias_rewrites_the_model_to_the_real_id(self):
        """The local front has no ``-think`` checkpoint to serve."""
        await self.client.post("/v1/messages", json=self._body(THINKING_ALIAS))
        self.assertEqual(self.local["requests"][0]["body"]["model"], LOCAL_MODEL)

    async def test_thinking_alias_forces_adaptive_thinking(self):
        await self.client.post("/v1/messages", json=self._body(THINKING_ALIAS))
        self.assertEqual(
            self.local["requests"][0]["body"]["thinking"], {"type": "adaptive"}
        )

    async def test_thinking_alias_overrides_an_explicit_client_value(self):
        """Naming the alias IS the request for thinking.

        Claude Code sends ``disabled`` of its own accord on some paths; if that
        won here the thinking arm would silently be the default arm.
        """
        await self.client.post(
            "/v1/messages",
            json=self._body(THINKING_ALIAS, thinking={"type": "disabled"}),
        )
        self.assertEqual(
            self.local["requests"][0]["body"]["thinking"], {"type": "adaptive"}
        )

    async def test_thinking_alias_un_aliases_count_tokens_without_thinking(self):
        await self.client.post(
            "/v1/messages/count_tokens", json=self._body(THINKING_ALIAS)
        )
        got = self.local["requests"][0]["body"]
        self.assertEqual(got["model"], LOCAL_MODEL)
        self.assertNotIn("thinking", got)

    async def test_thinking_alias_of_an_unknown_model_goes_upstream(self):
        """The suffix is only meaningful on a configured local id."""
        await self.client.post(
            "/v1/messages", json=self._body(REMOTE_MODEL + THINKING_ALIAS_SUFFIX)
        )
        self.assertEqual(len(self.upstream["requests"]), 1)
        self.assertEqual(self.local["requests"], [])
        self.assertEqual(
            self.upstream["requests"][0]["body"]["model"],
            REMOTE_MODEL + THINKING_ALIAS_SUFFIX,
        )

    async def test_thinking_alias_ignores_the_shim_switch(self):
        """``--no-thinking-shim`` disables the default-arm fill-in only."""
        app = create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            apply_shim=False,
        )
        server = TestServer(app)
        await server.start_server()
        try:
            from aiohttp.test_utils import TestClient

            async with TestClient(server) as client:
                await client.post("/v1/messages", json=self._body(THINKING_ALIAS))
        finally:
            await server.close()
        self.assertEqual(
            self.local["requests"][0]["body"]["thinking"], {"type": "adaptive"}
        )

    async def test_shim_does_not_touch_count_tokens(self):
        await self.client.post(
            "/v1/messages/count_tokens", json=self._body(LOCAL_MODEL)
        )
        self.assertNotIn("thinking", self.local["requests"][0]["body"])

    # ---------- streaming ----------

    async def test_streaming_passes_through_from_the_local_front(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body(LOCAL_MODEL, stream=True)
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers["Content-Type"], "text/event-stream")
        chunks = [chunk.decode() async for chunk, _ in resp.content.iter_chunks()]
        text = "".join(chunks)
        self.assertIn("event: message_start", text)
        self.assertIn("event: message_stop", text)
        self.assertIn('"backend":"local"', text)

    async def test_streaming_passes_through_from_upstream(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body(REMOTE_MODEL, stream=True)
        )
        text = await resp.text()
        self.assertIn('"backend":"upstream"', text)
        self.assertIn("event: message_stop", text)

    # ---------- errors ----------

    async def test_backend_error_status_and_body_survive(self):
        self.local["fail"] = True
        resp = await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(resp.status, 529)
        self.assertEqual((await resp.json())["error"]["type"], "overloaded_error")

    async def test_unreachable_backend_yields_an_anthropic_error_envelope(self):
        await self.local_server.close()
        resp = await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(resp.status, 502)
        payload = await resp.json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "api_error")

    # ---------- counters ----------

    async def test_stats_separate_local_from_upstream(self):
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        stats = await (await self.client.get(STATS_PATH)).json()
        self.assertEqual(stats["local"], 1)
        self.assertEqual(stats["upstream"], 2)
        self.assertEqual(stats["local_models"], [LOCAL_MODEL])
        self.assertEqual(stats["thinking_aliases"], [THINKING_ALIAS])

    async def test_thinking_alias_counts_as_a_local_request(self):
        await self.client.post("/v1/messages", json=self._body(THINKING_ALIAS))
        stats = await (await self.client.get(STATS_PATH)).json()
        self.assertEqual(stats["local"], 1)
        self.assertEqual(stats["upstream"], 0)

    async def test_stats_path_is_not_proxied(self):
        await self.client.get(STATS_PATH)
        self.assertEqual(self.upstream["requests"], [])
        self.assertEqual(self.local["requests"], [])


class ThinkingOnDefaultTestCase(AioHTTPTestCase):
    """The Qwen3.8 arm: the PLAIN id defaults to thinking on, with effort.

    The 3.6 arm is covered by RouterTestCase, whose app takes the defaults.
    Everything here is about the flipped default and the effort encoding.
    """

    EFFORT = "medium"

    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        self.policy_path = os.path.join(
            tempfile.mkdtemp(prefix="router-policy-"), "policy.json"
        )
        return create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            local_wait_s=0,  # this suite is about policy, not the hold buffer
            thinking_enabled=True,
            effort=self.EFFORT,
            policy_file=self.policy_path,
        )

    async def tearDownAsync(self):
        await self.upstream_server.close()
        await self.local_server.close()
        await super().tearDownAsync()

    def _body(self, model, **overrides):
        body = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        body.update(overrides)
        return body

    def _sent(self):
        return self.local["requests"][-1]["body"]

    # ---------- the flipped default ----------

    async def test_plain_id_gets_adaptive_thinking(self):
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(self._sent()["thinking"], {"type": "adaptive"})

    async def test_plain_id_gets_the_configured_effort(self):
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(self._sent()["output_config"]["effort"], self.EFFORT)

    async def test_client_thinking_off_does_not_win_the_arm(self):
        # The arm is the deployment's decision; the client's lever is the id.
        await self.client.post(
            "/v1/messages",
            json=self._body(LOCAL_MODEL, thinking={"type": "disabled"}),
        )
        self.assertEqual(self._sent()["thinking"], {"type": "adaptive"})

    async def test_nothink_alias_reaches_the_cheap_arm(self):
        await self.client.post("/v1/messages", json=self._body(NOTHINK_ALIAS))
        sent = self._sent()
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertEqual(sent["model"], LOCAL_MODEL)

    async def test_nothink_alias_carries_no_effort(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(NOTHINK_ALIAS, output_config={"effort": "low"}),
        )
        self.assertNotIn("output_config", self._sent())

    # ---------- effort override and encoding ----------

    async def test_per_request_effort_overrides_the_default(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(LOCAL_MODEL, output_config={"effort": "low"}),
        )
        self.assertEqual(self._sent()["output_config"]["effort"], "low")

    async def test_client_high_effort_becomes_the_omitted_field(self):
        # "high"/"xhigh"/"max" would all reach the template as a value it
        # rejects. The strongest arm is the ABSENT field, so that is what a
        # client asking for maximum reasoning must be turned into.
        for asked in ("high", "xhigh", "max"):
            with self.subTest(asked=asked):
                await self.client.post(
                    "/v1/messages",
                    json=self._body(LOCAL_MODEL, output_config={"effort": asked}),
                )
                self.assertNotIn("output_config", self._sent())

    async def test_normalization_preserves_other_output_config_fields(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(
                LOCAL_MODEL,
                output_config={
                    "effort": "max",
                    "task_budget": {"type": "tokens", "total": 100},
                },
            ),
        )
        sent = self._sent()
        self.assertNotIn("effort", sent["output_config"])
        self.assertEqual(sent["output_config"]["task_budget"]["total"], 100)

    async def test_think_alias_forces_effort_over_a_client_value(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(THINKING_ALIAS, output_config={"effort": "low"}),
        )
        self.assertEqual(self._sent()["output_config"]["effort"], self.EFFORT)

    async def test_count_tokens_is_not_given_an_effort(self):
        await self.client.post(
            "/v1/messages/count_tokens", json=self._body(LOCAL_MODEL)
        )
        self.assertNotIn("output_config", self._sent())

    # ---------- live policy file ----------

    async def test_policy_file_flips_the_arm_without_a_restart(self):
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(self._sent()["thinking"], {"type": "adaptive"})

        with open(self.policy_path, "w") as fh:
            json.dump({"thinking": "off"}, fh)
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(self._sent()["thinking"], {"type": "disabled"})

    async def test_policy_file_changes_the_effort_without_a_restart(self):
        with open(self.policy_path, "w") as fh:
            json.dump({"effort": "xhigh"}, fh)
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        # xhigh is the omitted field, so the strongest arm carries no key.
        self.assertNotIn("output_config", self._sent())

        os.utime(self.policy_path, None)
        with open(self.policy_path, "w") as fh:
            json.dump({"effort": "low"}, fh)
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(self._sent()["output_config"]["effort"], "low")

    async def test_a_broken_policy_file_leaves_the_flags_standing(self):
        with open(self.policy_path, "w") as fh:
            fh.write("{not json")
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        sent = self._sent()
        self.assertEqual(sent["thinking"], {"type": "adaptive"})
        self.assertEqual(sent["output_config"]["effort"], self.EFFORT)

    async def test_a_bad_effort_value_in_the_policy_file_is_ignored(self):
        with open(self.policy_path, "w") as fh:
            json.dump({"effort": "turbo"}, fh)
        await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(self._sent()["output_config"]["effort"], self.EFFORT)

    async def test_stats_report_the_effective_policy(self):
        stats = await (await self.client.get(STATS_PATH)).json()
        self.assertEqual(stats["policy"]["thinking_enabled"], True)
        self.assertEqual(stats["policy"]["effort"], self.EFFORT)
        self.assertEqual(stats["nothink_aliases"], [NOTHINK_ALIAS])


class LocalBackendBufferTestCase(unittest.IsolatedAsyncioTestCase):
    """The local-backend hold buffer (task #675).

    Unlike the suites above, these tests bring the local backend up and down
    MID-TEST, so they manage TestServer/TestClient lifecycles by hand instead
    of AioHTTPTestCase's single fixed get_application(). The local backend's
    listen port is fixed via unused_port() up front so it can be "absent"
    (nothing listening -> connection refused, simulating a backend that is
    still booting) and then started later on that same port (simulating the
    boot completing), without the router's local_base ever changing.
    """

    async def asyncSetUp(self):
        upstream_app, self.upstream = _make_backend("upstream")
        self.upstream_server = TestServer(upstream_app)
        await self.upstream_server.start_server()
        self.local_port = unused_port()
        self.local_server = None
        self.local = None
        self._extra_clients = []
        self._extra_servers = []

    async def asyncTearDown(self):
        for client in self._extra_clients:
            await client.close()
        for server in self._extra_servers:
            await server.close()
        await self.upstream_server.close()
        if self.local_server is not None:
            await self.local_server.close()

    def _local_base(self):
        return f"http://127.0.0.1:{self.local_port}"

    async def _start_local_backend(self):
        """Bring the local backend "up" on the port the router already targets."""
        local_app, self.local = _make_backend("local")
        self.local_server = TestServer(local_app, port=self.local_port)
        await self.local_server.start_server()

    def _body(self, model=LOCAL_MODEL, **overrides):
        body = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        body.update(overrides)
        return body

    async def _make_router(self, **kwargs):
        app = create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=self._local_base(),
            **kwargs,
        )
        server = TestServer(app)
        await server.start_server()
        client = TestClient(server)
        await client.start_server()
        self._extra_servers.append(server)
        self._extra_clients.append(client)
        return app, server, client

    # ---------- held, then succeeds ----------

    async def test_request_is_held_and_then_succeeds_once_backend_comes_up(self):
        _, _, client = await self._make_router(
            local_wait_s=5.0, local_poll_interval_s=0.15
        )

        async def _send():
            return await client.post("/v1/messages", json=self._body())

        task = asyncio.create_task(_send())
        # Give the first connect attempt time to fail (backend not started
        # yet) and register itself in the held queue.
        await asyncio.sleep(0.4)
        stats = await (await client.get(STATS_PATH)).json()
        self.assertEqual(stats["buffer_queued"], 1)
        self.assertGreater(stats["buffer_current_wait_s"], 0)
        self.assertEqual(len(self.upstream["requests"]), 0)  # never fell to upstream

        await self._start_local_backend()
        resp = await asyncio.wait_for(task, timeout=5.0)
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "local")

        stats = await (await client.get(STATS_PATH)).json()
        self.assertEqual(stats["buffer_queued"], 0)
        self.assertEqual(stats["buffer_succeeded"], 1)
        self.assertEqual(stats["buffer_gave_up"], 0)

    async def test_streaming_request_is_held_and_completes_once_backend_is_up(self):
        """Requirement: a streaming request must not be half-started.

        No SSE bytes -- not even headers -- can have reached the client while
        the backend was down, so the response, once it does arrive, must be
        the complete, uncorrupted stream.
        """
        _, _, client = await self._make_router(
            local_wait_s=5.0, local_poll_interval_s=0.15
        )

        async def _send():
            return await client.post("/v1/messages", json=self._body(stream=True))

        task = asyncio.create_task(_send())
        await asyncio.sleep(0.4)
        await self._start_local_backend()
        resp = await asyncio.wait_for(task, timeout=5.0)
        self.assertEqual(resp.status, 200)
        text = await resp.text()
        self.assertIn("event: message_start", text)
        self.assertIn("event: message_stop", text)
        self.assertIn('"backend":"local"', text)

    # ---------- gives up honestly ----------

    async def test_timeout_returns_honest_502_with_waited_seconds(self):
        _, _, client = await self._make_router(
            local_wait_s=0.6, local_poll_interval_s=0.15
        )
        start = time.monotonic()
        resp = await client.post("/v1/messages", json=self._body())
        elapsed = time.monotonic() - start

        self.assertEqual(resp.status, 502)
        payload = await resp.json()
        message = payload["error"]["message"]
        self.assertRegex(message, r"held the request for \d+s")
        self.assertIn("local backend was down", message)
        # It actually waited roughly the configured cap, not an instant 502.
        self.assertGreaterEqual(elapsed, 0.5)
        self.assertLess(elapsed, 3.0)

        stats = await (await client.get(STATS_PATH)).json()
        self.assertEqual(stats["buffer_gave_up"], 1)
        self.assertEqual(stats["buffer_succeeded"], 0)
        self.assertEqual(stats["buffer_queued"], 0)

    async def test_queue_full_gives_up_immediately_without_waiting(self):
        _, _, client = await self._make_router(
            local_wait_s=5.0, local_poll_interval_s=0.15, max_buffered=1
        )

        async def _send():
            return await client.post("/v1/messages", json=self._body())

        first = asyncio.create_task(_send())
        await asyncio.sleep(0.3)
        stats = await (await client.get(STATS_PATH)).json()
        self.assertEqual(stats["buffer_queued"], 1)

        start = time.monotonic()
        second = await client.post("/v1/messages", json=self._body())
        elapsed = time.monotonic() - start
        self.assertEqual(second.status, 502)
        payload = await second.json()
        self.assertIn("queue is full", payload["error"]["message"])
        # Rejected on arrival, not held for anywhere near the 5s wait cap.
        self.assertLess(elapsed, 0.3)

        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first

        stats = await (await client.get(STATS_PATH)).json()
        self.assertEqual(stats["buffer_gave_up"], 1)

    async def test_disabled_buffer_fails_immediately_like_before(self):
        """``local_wait_s=0`` is the pre-buffer behaviour, byte for byte."""
        _, _, client = await self._make_router(local_wait_s=0)
        start = time.monotonic()
        resp = await client.post("/v1/messages", json=self._body())
        elapsed = time.monotonic() - start
        self.assertEqual(resp.status, 502)
        self.assertLess(elapsed, 0.3)
        stats = await (await client.get(STATS_PATH)).json()
        self.assertEqual(stats["buffer_queued"], 0)
        self.assertEqual(stats["buffer_gave_up"], 0)  # never entered the hold path

    # ---------- healthy path is untouched ----------

    async def test_healthy_path_has_no_added_latency(self):
        """Requirement: zero added latency when upstream/local is healthy.

        Trivial check, not a benchmark: with the backend up throughout, the
        buffer-enabled app (wait_s=300, the production default magnitude)
        must not be meaningfully slower than the buffer-disabled app, because
        a healthy first connect attempt takes the exact same code path
        (``_connect_once``) either way.
        """
        await self._start_local_backend()
        _, _, client_off = await self._make_router(local_wait_s=0)
        _, _, client_on = await self._make_router(
            local_wait_s=300.0, local_poll_interval_s=2.0
        )

        async def _avg_latency(client, n=20):
            # One warm-up call, excluded, so connection setup cost is equal.
            await client.post("/v1/messages", json=self._body())
            total = 0.0
            for _ in range(n):
                start = time.monotonic()
                resp = await client.post("/v1/messages", json=self._body())
                total += time.monotonic() - start
                self.assertEqual(resp.status, 200)
            return total / n

        off_avg = await _avg_latency(client_off)
        on_avg = await _avg_latency(client_on)
        # Generous bound: this asserts "no meaningful regression", not tight
        # equality -- CI noise on a shared box should never flake this.
        self.assertLess(on_avg, off_avg + 0.05)


if __name__ == "__main__":
    unittest.main()
