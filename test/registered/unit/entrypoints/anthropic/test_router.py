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


class AllowedModelsPolicyTestCase(AioHTTPTestCase):
    """``allowed_models`` in the policy file (#1319).

    Claude Code falls back to another model on its own when a request fails
    at the HTTP level; the router cannot stop the client from trying, but it
    can refuse to SERVE an unapproved model. These tests pin the refusal
    (403 before either backend), the pass-through of listed ids, the
    exemption of model-less bodies, the hot reload in both directions and the
    non-fatal handling of a malformed list.
    """

    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        self.policy_path = os.path.join(
            tempfile.mkdtemp(prefix="router-allowlist-"), "policy.json"
        )
        return create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            local_wait_s=0,  # about the allow-list, not the hold buffer
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

    def _write_policy(self, obj):
        # Bump mtime first so a rewrite inside the same clock tick still reloads.
        if os.path.exists(self.policy_path):
            os.utime(self.policy_path, None)
        with open(self.policy_path, "w") as fh:
            json.dump(obj, fh)

    async def _stats(self):
        resp = await self.client.get(STATS_PATH)
        return await resp.json()

    async def test_no_key_means_every_model_passes(self):
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertNotEqual(resp.status, 403)
        self.assertEqual(len(self.upstream["requests"]), 1)
        self.assertEqual((await self._stats())["refused_model"], 0)

    async def test_an_unlisted_model_is_refused_before_either_backend(self):
        self._write_policy({"allowed_models": [LOCAL_MODEL, THINKING_ALIAS]})
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertEqual(resp.status, 403)
        err = await resp.json()
        self.assertEqual(err["type"], "error")
        self.assertEqual(err["error"]["type"], "permission_error")
        self.assertIn(REMOTE_MODEL, err["error"]["message"])
        self.assertIn(LOCAL_MODEL, err["error"]["message"])
        self.assertEqual(self.upstream["requests"], [])
        self.assertEqual(self.local["requests"], [])
        stats = await self._stats()
        self.assertEqual(stats["refused_model"], 1)
        self.assertEqual(stats["upstream"], 0)
        self.assertEqual(stats["local"], 0)
        self.assertEqual(
            sorted(stats["policy"]["allowed_models"]),
            sorted([LOCAL_MODEL, THINKING_ALIAS]),
        )

    async def test_listed_ids_pass_and_the_alias_is_matched_as_sent(self):
        self._write_policy({"allowed_models": [LOCAL_MODEL, THINKING_ALIAS]})
        resp = await self.client.post("/v1/messages", json=self._body(THINKING_ALIAS))
        self.assertNotEqual(resp.status, 403)
        resp = await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertNotEqual(resp.status, 403)
        self.assertEqual(len(self.local["requests"]), 2)
        self.assertEqual(self.upstream["requests"], [])
        # The nothink alias is a different id: not listed, so refused.
        resp = await self.client.post("/v1/messages", json=self._body(NOTHINK_ALIAS))
        self.assertEqual(resp.status, 403)
        self.assertEqual(len(self.local["requests"]), 2)

    async def test_a_body_without_a_model_is_not_subject_to_the_list(self):
        self._write_policy({"allowed_models": [LOCAL_MODEL]})
        resp = await self.client.post(
            "/v1/messages",
            json={"max_tokens": 1, "messages": [{"role": "user", "content": "x"}]},
        )
        self.assertNotEqual(resp.status, 403)
        self.assertEqual(len(self.upstream["requests"]), 1)

    async def test_the_list_applies_to_every_proxied_path(self):
        self._write_policy({"allowed_models": [LOCAL_MODEL]})
        resp = await self.client.post(
            "/v1/messages/count_tokens", json={"model": REMOTE_MODEL, "messages": []}
        )
        self.assertEqual(resp.status, 403)
        self.assertEqual(self.upstream["requests"], [])

    async def test_the_list_is_hot_reloaded_in_both_directions(self):
        self._write_policy({"allowed_models": [LOCAL_MODEL]})
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertEqual(resp.status, 403)
        self._write_policy({})
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertNotEqual(resp.status, 403)
        self.assertEqual(len(self.upstream["requests"]), 1)
        self._write_policy({"allowed_models": [LOCAL_MODEL]})
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertEqual(resp.status, 403)
        self.assertEqual(len(self.upstream["requests"]), 1)

    async def test_a_malformed_list_is_ignored_and_nothing_is_refused(self):
        for bad in ("claude-x", [], [1, 2], {"a": 1}):
            self._write_policy({"allowed_models": bad})
            resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
            self.assertNotEqual(resp.status, 403, bad)
        self.assertEqual(len(self.upstream["requests"]), 4)
        self.assertEqual((await self._stats())["refused_model"], 0)


# ---------------------------------------------------------------------------
# OpenRouter third arm (router/openrouter-arm-0914)
# ---------------------------------------------------------------------------
#
# Two classes below. ``ProductionRoutingPinTestCase`` is the ACCEPTANCE GATE
# named in the task briefing: written BEFORE the third arm existed, wired
# with today's real production CLI shape (``--local-model Qwen3.8-27B
# --local-thinking on --local-effort xhigh``, and the real
# ``allowed_models`` list from /etc/htsglang/router-policy.json as it stood
# on 2026-09-14) and the real production model ids -- ``claude-opus-5``,
# ``claude-sonnet-5``, ``Qwen3.8-27B-think``. It must pass BOTH before and
# after the OpenRouter arm is added, byte-identically: this is the danger
# direction named in the briefing (an existing model silently starts
# routing somewhere else), and this class is a set of mutants on exactly
# that direction, not a design wishlist.
#
# ``OpenRouterArmTestCase`` is the new-feature test: it is expected to FAIL
# until the arm is implemented (there is no ``openrouter_models``/
# ``openrouter_base`` parameter on ``create_app`` yet), and is the red half
# of red-first for the new behaviour itself.

PROD_OPUS = "claude-opus-5"
PROD_SONNET = "claude-sonnet-5"
PROD_LOCAL_MODEL = "Qwen3.8-27B"
PROD_THINK_ALIAS = PROD_LOCAL_MODEL + THINKING_ALIAS_SUFFIX
PROD_NOTHINK_ALIAS = PROD_LOCAL_MODEL + NOTHINK_ALIAS_SUFFIX
# Real allowed_models as read from /etc/htsglang/router-policy.json,
# 2026-09-14, before this branch touches it.
PROD_ALLOWED_MODELS_0914 = [
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-sonnet-5",
    "Qwen3.8-27B-think",
    "Qwen3.8-27B",
]


class ProductionRoutingPinTestCase(AioHTTPTestCase):
    """Pin today's real routing decision for the three live production ids.

    This is the regression gate for "byte identity for all existing
    models" (task briefing, Phase C). It mirrors the actual deployed CLI
    line (10-qwen38.conf): local model Qwen3.8-27B, thinking forced ON with
    xhigh effort for the plain id, plus the real allowed_models list.
    """

    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        self.policy_path = os.path.join(
            tempfile.mkdtemp(prefix="router-prodpin-"), "policy.json"
        )
        with open(self.policy_path, "w") as fh:
            json.dump(
                {
                    "thinking": "on",
                    "effort": "xhigh",
                    "allowed_models": PROD_ALLOWED_MODELS_0914,
                },
                fh,
            )
        return create_app(
            local_models=[PROD_LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            local_wait_s=0,
            thinking_enabled=True,
            effort="xhigh",
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

    async def test_claude_opus_5_goes_upstream_untouched(self):
        body = self._body(PROD_OPUS)
        resp = await self.client.post("/v1/messages", json=body)
        self.assertNotEqual(resp.status, 403)
        self.assertEqual((await resp.json())["backend"], "upstream")
        self.assertEqual(len(self.upstream["requests"]), 1)
        self.assertEqual(self.local["requests"], [])
        self.assertNotIn("thinking", self.upstream["requests"][0]["body"])

    async def test_claude_sonnet_5_goes_upstream_untouched(self):
        body = self._body(PROD_SONNET)
        resp = await self.client.post("/v1/messages", json=body)
        self.assertNotEqual(resp.status, 403)
        self.assertEqual((await resp.json())["backend"], "upstream")
        self.assertEqual(len(self.upstream["requests"]), 1)
        self.assertEqual(self.local["requests"], [])
        self.assertNotIn("thinking", self.upstream["requests"][0]["body"])

    async def test_qwen38_think_alias_still_goes_local_with_adaptive_thinking(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body(PROD_THINK_ALIAS)
        )
        self.assertNotEqual(resp.status, 403)
        self.assertEqual((await resp.json())["backend"], "local")
        self.assertEqual(self.upstream["requests"], [])
        sent = self.local["requests"][0]["body"]
        self.assertEqual(sent["model"], PROD_LOCAL_MODEL)
        self.assertEqual(sent["thinking"], {"type": "adaptive"})

    async def test_qwen38_nothink_alias_still_reaches_the_cheap_arm(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body(PROD_NOTHINK_ALIAS)
        )
        self.assertEqual(resp.status, 403)  # not in the 0914 allowed_models
        self.assertEqual(self.local["requests"], [])
        self.assertEqual(self.upstream["requests"], [])

    async def test_unlisted_model_is_still_refused_before_either_backend(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body("some-other-model")
        )
        self.assertEqual(resp.status, 403)
        self.assertEqual(self.local["requests"], [])
        self.assertEqual(self.upstream["requests"], [])


class OpenRouterArmTestCase(AioHTTPTestCase):
    """The new third bucket: a listed model slug routes to a translator port.

    Everything not in ``local_models`` and not in ``openrouter_models``
    keeps going upstream exactly as before -- this class only exercises the
    NEW branch; ``ProductionRoutingPinTestCase`` above is what proves the
    old branches are untouched.

    A real (fake) key is configured in a temp file for most of this class,
    because the key-ABSENT behaviour (refuse by name) is its own,
    separately named class below (``OpenRouterMissingKeyTestCase``) --
    mixing "key present" and "key absent" fixtures in one class makes it too
    easy for a future edit to add a case under the wrong assumption.
    """

    OPENROUTER_MODEL = "qwen/qwen3.8-flash"
    FAKE_KEY = "sk-or-v1-test-fake-openrouter-key-0914"

    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        openrouter_app, self.openrouter = _make_backend("openrouter")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        self.openrouter_server = TestServer(openrouter_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        await self.openrouter_server.start_server()
        self.key_path = os.path.join(
            tempfile.mkdtemp(prefix="router-openrouter-key-"), "openrouter.key"
        )
        with open(self.key_path, "w") as fh:
            fh.write(self.FAKE_KEY + "\n")
        return create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            local_wait_s=0,
            openrouter_models=[self.OPENROUTER_MODEL],
            openrouter_base=str(self.openrouter_server.make_url("")).rstrip("/"),
            openrouter_key_file=self.key_path,
        )

    async def tearDownAsync(self):
        await self.upstream_server.close()
        await self.local_server.close()
        await self.openrouter_server.close()
        await super().tearDownAsync()

    def _body(self, model, **overrides):
        body = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        body.update(overrides)
        return body

    def _write_key(self, content):
        if os.path.exists(self.key_path):
            os.utime(self.key_path, None)
        with open(self.key_path, "w") as fh:
            fh.write(content)

    async def test_listed_openrouter_model_goes_to_the_translator_port(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "openrouter")
        self.assertEqual(len(self.openrouter["requests"]), 1)
        self.assertEqual(self.local["requests"], [])
        self.assertEqual(self.upstream["requests"], [])

    async def test_openrouter_gets_no_thinking_shim_and_keeps_its_model(self):
        """The thinking shim is local-only; the third arm gets no shim.

        This used to assert the openrouter body was byte-identical. That
        stopped being true when the cache-marker edit landed (see
        ``OpenRouterCacheMarkerTestCase``): the arm still gets no thinking
        shim and no model rewrite, but ``cache_control`` markers ARE added
        on ``/v1/messages``. Inverted rather than deleted, so the two
        properties that do still hold stay pinned.
        """
        body = self._body(self.OPENROUTER_MODEL)
        await self.client.post("/v1/messages", json=body)
        self.assertNotIn("thinking", self.openrouter["requests"][0]["body"])
        self.assertEqual(self.openrouter["requests"][0]["body"]["model"], self.OPENROUTER_MODEL)

    async def test_unlisted_model_still_goes_upstream_not_openrouter(self):
        resp = await self.client.post("/v1/messages", json=self._body(REMOTE_MODEL))
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "upstream")
        self.assertEqual(self.openrouter["requests"], [])

    async def test_local_model_still_goes_local_not_openrouter(self):
        resp = await self.client.post("/v1/messages", json=self._body(LOCAL_MODEL))
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "local")
        self.assertEqual(self.openrouter["requests"], [])

    async def test_stats_reports_openrouter_models_and_counter(self):
        await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        resp = await self.client.get(STATS_PATH)
        stats = await resp.json()
        self.assertIn(self.OPENROUTER_MODEL, stats["openrouter_models"])
        self.assertEqual(stats["openrouter"], 1)
        self.assertTrue(stats["openrouter_key_configured"])

    async def test_the_clients_own_credential_never_reaches_openrouter(self):
        """The client's Anthropic key must not leak to a third party.

        Forwarding it to OpenRouter would be a strictly worse mistake than
        a failed request: it hands our Anthropic secret to a different
        company. The translator gets OUR key from the key file instead.
        """
        await self.client.post(
            "/v1/messages",
            json=self._body(self.OPENROUTER_MODEL),
            headers={"x-api-key": "sk-ant-this-must-never-leave-the-building"},
        )
        headers = self.openrouter["requests"][0]["headers"]
        self.assertNotEqual(
            headers.get("x-api-key"), "sk-ant-this-must-never-leave-the-building"
        )

    async def test_the_deployments_own_key_reaches_the_translator(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(self.OPENROUTER_MODEL),
            headers={"x-api-key": "sk-ant-whatever-the-client-sent"},
        )
        headers = self.openrouter["requests"][0]["headers"]
        self.assertEqual(headers["x-api-key"], self.FAKE_KEY)

    async def test_authorization_header_is_also_scrubbed(self):
        await self.client.post(
            "/v1/messages",
            json=self._body(self.OPENROUTER_MODEL),
            headers={"Authorization": "Bearer sk-ant-also-must-not-leak"},
        )
        headers = self.openrouter["requests"][0]["headers"]
        self.assertNotIn("sk-ant-also-must-not-leak", str(headers))
        self.assertEqual(headers.get("x-api-key"), self.FAKE_KEY)

    async def test_credentials_never_logged_for_openrouter_arm(self):
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.router", level="DEBUG"
        ) as log:
            await self.client.post(
                "/v1/messages",
                json=self._body(self.OPENROUTER_MODEL),
                headers={"x-api-key": "sk-ant-super-secret"},
            )
        joined = "\n".join(log.output)
        self.assertNotIn("sk-ant-super-secret", joined)
        # The DEPLOYMENT's own key must not appear either -- this is the
        # more important half, and the one a naive `logger.info("headers=%s"
        # % headers)` after the credential swap would fail.
        self.assertNotIn(self.FAKE_KEY, joined)

    async def test_key_is_hot_reloaded_without_a_restart(self):
        NEW_KEY = "sk-or-v1-rotated-test-key"
        self._write_key(NEW_KEY + "\n")
        await self.client.post("/v1/messages", json=self._body(self.OPENROUTER_MODEL))
        headers = self.openrouter["requests"][-1]["headers"]
        self.assertEqual(headers["x-api-key"], NEW_KEY)

    async def test_without_openrouter_models_configured_nothing_changes(self):
        """No --openrouter-model given: identical to before this branch."""
        upstream_app, upstream = _make_backend("upstream")
        local_app, local = _make_backend("local")
        upstream_server = TestServer(upstream_app)
        local_server = TestServer(local_app)
        await upstream_server.start_server()
        await local_server.start_server()
        try:
            app = create_app(
                local_models=[LOCAL_MODEL],
                upstream_base=str(upstream_server.make_url("")).rstrip("/"),
                local_base=str(local_server.make_url("")).rstrip("/"),
                local_wait_s=0,
            )
            server = TestServer(app)
            await server.start_server()
            try:
                async with TestClient(server) as client:
                    resp = await client.post(
                        "/v1/messages",
                        json=self._body(self.OPENROUTER_MODEL),
                    )
                    self.assertEqual((await resp.json())["backend"], "upstream")
            finally:
                await server.close()
        finally:
            await upstream_server.close()
            await local_server.close()

    async def test_key_never_appears_in_any_response_or_log_across_every_route(self):
        """Repo guard: a NEW serialization site that forgets to redact the
        key must turn this test red.

        Rather than hand-pick one code path (the class above for #1275/
        #1282/#1283 was defeated exactly that way -- a fix for one sink left
        a second, unrelated one open), this sweeps every route the app
        currently registers and every log line produced along the way, at
        the most permissive log level. A future route that spreads app
        state (config dump, debug endpoint, an expanded /__router/stats)
        without excluding the key breaks this test the same day it is
        added, not months later.
        """
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.router", level="DEBUG"
        ) as log:
            resp1 = await self.client.post(
                "/v1/messages",
                json=self._body(self.OPENROUTER_MODEL),
                headers={"x-api-key": "sk-ant-sweep-probe"},
            )
            body1 = await resp1.read()
            resp2 = await self.client.get(STATS_PATH)
            body2 = await resp2.read()
            resp3 = await self.client.post(
                "/v1/messages/count_tokens",
                json=self._body(self.OPENROUTER_MODEL),
            )
            body3 = await resp3.read()
        joined_logs = "\n".join(log.output)
        for label, blob in (
            ("stream response", body1),
            ("stats response", body2),
            ("count_tokens response", body3),
            ("logs", joined_logs.encode()),
        ):
            self.assertNotIn(
                self.FAKE_KEY.encode(), blob, f"key leaked into {label}"
            )


class OpenRouterMissingKeyTestCase(AioHTTPTestCase):
    """No key configured: refuse by name, never fall back silently.

    This is the coordinator's explicit correction: an empty/missing/
    placeholder key file must behave like an unlisted model in the
    allowed_models check (#1319) -- a named 403, before either backend is
    touched -- and must NEVER be treated as "connection failed, try
    upstream" or any other silent substitution. #1319's own root cause was
    exactly this kind of silent fallback (a refused local model landing an
    Opus agent on Fable for 54 turns), so this arm is held to the same bar.
    """

    OPENROUTER_MODEL = "qwen/qwen3.8-flash"

    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        openrouter_app, self.openrouter = _make_backend("openrouter")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        self.openrouter_server = TestServer(openrouter_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        await self.openrouter_server.start_server()
        self.key_dir = tempfile.mkdtemp(prefix="router-openrouter-nokey-")
        self.key_path = os.path.join(self.key_dir, "openrouter.key")
        # No file written yet -- covers "missing" for the first sub-test;
        # other sub-tests write empty/placeholder content explicitly.
        return create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            local_wait_s=0,
            openrouter_models=[self.OPENROUTER_MODEL],
            openrouter_base=str(self.openrouter_server.make_url("")).rstrip("/"),
            openrouter_key_file=self.key_path,
        )

    async def tearDownAsync(self):
        await self.upstream_server.close()
        await self.local_server.close()
        await self.openrouter_server.close()
        await super().tearDownAsync()

    def _body(self, model, **overrides):
        body = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        body.update(overrides)
        return body

    async def _stats(self):
        resp = await self.client.get(STATS_PATH)
        return await resp.json()

    async def test_missing_key_file_refuses_by_name_before_either_backend(self):
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 403)
        err = await resp.json()
        self.assertEqual(err["type"], "error")
        self.assertEqual(err["error"]["type"], "permission_error")
        self.assertIn(self.OPENROUTER_MODEL, err["error"]["message"])
        self.assertEqual(self.openrouter["requests"], [])
        self.assertEqual(self.local["requests"], [])
        self.assertEqual(self.upstream["requests"], [])

    async def test_missing_key_never_falls_back_to_local_or_upstream(self):
        """The #1319 failure class, applied to this arm."""
        for _ in range(3):
            resp = await self.client.post(
                "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
            )
            self.assertEqual(resp.status, 403)
        stats = await self._stats()
        self.assertEqual(stats["local"], 0)
        self.assertEqual(stats["upstream"], 0)
        self.assertEqual(stats["openrouter"], 0)
        self.assertEqual(stats["openrouter_no_key"], 3)
        self.assertFalse(stats["openrouter_key_configured"])

    async def test_empty_key_file_refuses_the_same_way(self):
        with open(self.key_path, "w") as fh:
            fh.write("")
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 403)

    async def test_whitespace_only_key_file_refuses_the_same_way(self):
        with open(self.key_path, "w") as fh:
            fh.write("   \n\n")
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 403)

    async def test_old_format_placeholder_line_refuses_the_same_way(self):
        """The since-corrected 'OPENROUTER_API_KEY=<...>' draft format.

        A key file left over from that draft (or a user who copy-pasted the
        old instructions) must not be forwarded as a literal credential
        containing the string 'OPENROUTER_API_KEY='.
        """
        with open(self.key_path, "w") as fh:
            fh.write("OPENROUTER_API_KEY=PLACEHOLDER_USER_FILLS_THIS\n")
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 403)

    async def test_key_becomes_available_after_being_filled_in_no_restart(self):
        """The other half of hot-reload: 403 -> works, without a restart."""
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 403)
        with open(self.key_path, "w") as fh:
            fh.write("sk-or-v1-freshly-filled-in\n")
        resp = await self.client.post(
            "/v1/messages", json=self._body(self.OPENROUTER_MODEL)
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["backend"], "openrouter")
        headers = self.openrouter["requests"][0]["headers"]
        self.assertEqual(headers["x-api-key"], "sk-or-v1-freshly-filled-in")


class OpenRouterCacheMarkerTestCase(AioHTTPTestCase):
    """Prompt-cache breakpoints the router adds on the openrouter arm.

    WHY THE ROUTER AND NOT THE CLIENT. A ``cache_control`` marker names a
    prefix breakpoint, and openrouter.ai honours EVERY marker in a request,
    not only the first (measured 2026-09-14 against
    ``qwen/qwen3.8-flash``: system-only marker left a 7,716-token
    conversation billed in full every turn; system + conversation tail cut
    ``input_tokens`` to 6 and read 11,756 from cache). Claude Code marks its
    system block and leaves the conversation unmarked, so the growing
    history was paid for at full price on every turn. The router is the only
    party that knows the request is openrouter-bound, so the router adds
    what is missing.

    The property that makes this safe is that the edit is ADDITIVE: it never
    moves or removes a marker the client placed, and never exceeds the
    documented four-breakpoint ceiling.
    """

    OPENROUTER_MODEL = "qwen/qwen3.8-flash"
    FAKE_KEY = "sk-or-v1-test-fake-openrouter-key-cache"

    async def get_application(self):
        upstream_app, self.upstream = _make_backend("upstream")
        local_app, self.local = _make_backend("local")
        openrouter_app, self.openrouter = _make_backend("openrouter")
        self.upstream_server = TestServer(upstream_app)
        self.local_server = TestServer(local_app)
        self.openrouter_server = TestServer(openrouter_app)
        await self.upstream_server.start_server()
        await self.local_server.start_server()
        await self.openrouter_server.start_server()
        self.key_path = os.path.join(
            tempfile.mkdtemp(prefix="router-cache-key-"), "openrouter.key"
        )
        with open(self.key_path, "w") as fh:
            fh.write(self.FAKE_KEY + "\n")
        return create_app(
            local_models=[LOCAL_MODEL],
            upstream_base=str(self.upstream_server.make_url("")).rstrip("/"),
            local_base=str(self.local_server.make_url("")).rstrip("/"),
            local_wait_s=0,
            openrouter_models=[self.OPENROUTER_MODEL],
            openrouter_base=str(self.openrouter_server.make_url("")).rstrip("/"),
            openrouter_key_file=self.key_path,
        )

    async def tearDownAsync(self):
        await self.upstream_server.close()
        await self.local_server.close()
        await self.openrouter_server.close()
        await super().tearDownAsync()

    def _turn(self, text, role="user"):
        return {"role": role, "content": [{"type": "text", "text": text}]}

    def _conversation(self, turns=6, model=None, system=None):
        """A Claude-Code-shaped body: block-list turns, optional system."""
        messages = []
        for i in range(turns):
            messages.append(self._turn(f"user turn {i}"))
            messages.append(self._turn(f"assistant turn {i}", role="assistant"))
        body = {
            "model": model or self.OPENROUTER_MODEL,
            "max_tokens": 64,
            "messages": messages,
        }
        if system is not None:
            body["system"] = system
        return body

    def _markers(self, payload):
        """Every (carrier, index) position holding a cache_control marker."""
        found = []
        for key in ("system", "tools"):
            value = payload.get(key)
            if isinstance(value, list):
                for i, block in enumerate(value):
                    if isinstance(block, dict) and "cache_control" in block:
                        found.append((key, i))
        for mi, message in enumerate(payload.get("messages", [])):
            content = message.get("content")
            if isinstance(content, list):
                for bi, block in enumerate(content):
                    if isinstance(block, dict) and "cache_control" in block:
                        found.append((f"msg{mi}", bi))
        return found

    async def test_the_conversation_tail_gets_a_marker(self):
        """The defect: an unmarked conversation was re-billed every turn."""
        await self.client.post("/v1/messages", json=self._conversation())
        sent = self.openrouter["requests"][0]["body"]
        self.assertTrue(
            self._markers(sent),
            "no cache_control marker reached openrouter -- the conversation "
            "is billed at full price on every turn",
        )

    async def test_two_rolling_markers_so_one_writes_and_one_reads(self):
        """A single rolling marker would write a cache nobody reads.

        The newest marked turn writes the cache for the NEXT request; the
        one behind it is what this request reads back.
        """
        await self.client.post("/v1/messages", json=self._conversation())
        sent = self.openrouter["requests"][0]["body"]
        self.assertEqual(len(self._markers(sent)), 2)

    async def test_markers_land_on_user_turns_newest_first(self):
        body = self._conversation(turns=6)
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        marked = [name for name, _ in self._markers(sent)]
        # 12 messages, user turns at even indices; newest two are 8 and 10.
        self.assertEqual(sorted(marked), ["msg10", "msg8"])
        for name in marked:
            index = int(name[3:])
            self.assertEqual(sent["messages"][index]["role"], "user")

    async def test_a_client_marker_is_never_moved_or_removed(self):
        """Additive by construction: the client's own choice survives."""
        body = self._conversation(turns=3)
        body["system"] = [
            {"type": "text", "text": "a big system prompt",
             "cache_control": {"type": "ephemeral"}}
        ]
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertIn(("system", 0), self._markers(sent))
        self.assertEqual(
            sent["system"][0]["cache_control"], {"type": "ephemeral"}
        )

    async def test_a_clients_marker_VALUE_survives_unchanged(self):
        """The danger direction the identical-value tests cannot see.

        Every other test here puts ``{"type": "ephemeral"}`` on both sides, so
        a router that OVERWRITES the client's marker instead of counting it
        reads exactly like one that leaves it alone -- measured: mutating the
        ``if "cache_control" in block`` test to ``if False`` left all 91 tests
        green. A marker carrying a field we do not write makes the two
        distinguishable, and pins the contract that is actually promised:
        additive, never rewriting what the client chose.
        """
        body = self._conversation(turns=4)
        client_choice = {"type": "ephemeral", "ttl": "1h"}
        body["messages"][-2]["content"][-1]["cache_control"] = client_choice
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertEqual(sent["messages"][-2]["content"][-1]["cache_control"],
                         client_choice)

    async def test_the_four_breakpoint_ceiling_is_never_exceeded(self):
        """Anthropic's documented ceiling; over it the request is rejected."""
        body = self._conversation(turns=4)
        body["system"] = [
            {"type": "text", "text": f"system {i}",
             "cache_control": {"type": "ephemeral"}}
            for i in range(3)
        ]
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertLessEqual(len(self._markers(sent)), 4)

    async def test_a_client_that_already_spent_the_budget_is_left_alone(self):
        body = self._conversation(turns=4)
        body["system"] = [
            {"type": "text", "text": f"system {i}",
             "cache_control": {"type": "ephemeral"}}
            for i in range(4)
        ]
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertEqual(len(self._markers(sent)), 4)
        self.assertEqual([n for n, _ in self._markers(sent)], ["system"] * 4)

    async def test_a_turn_the_client_marked_itself_counts_as_one_of_the_pair(self):
        """No double-marking: the client's tail marker is one of the two."""
        body = self._conversation(turns=4)
        body["messages"][-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertEqual(len(self._markers(sent)), 2)

    async def test_a_string_content_turn_is_skipped_not_rewritten(self):
        """A bare string has no block to attach a marker to.

        The router does NOT restructure the client's message into a block
        list to create one -- it looks further back instead, and says so in
        the log. Restructuring would change what the backend renders on a
        path where we cannot verify the rendering.
        """
        body = self._conversation(turns=3)
        body["messages"][-2]["content"] = "a bare string turn"
        await self.client.post("/v1/messages", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertEqual(sent["messages"][-2]["content"], "a bare string turn")
        self.assertTrue(self._markers(sent))

    async def test_other_endpoints_are_not_touched(self):
        """count_tokens must stay byte-identical: it prices, not generates."""
        body = self._conversation()
        await self.client.post("/v1/messages/count_tokens", json=body)
        sent = self.openrouter["requests"][0]["body"]
        self.assertEqual(self._markers(sent), [])

    async def test_the_local_arm_is_untouched(self):
        """Local backend caching is not this router's business."""
        body = self._conversation(model=LOCAL_MODEL)
        await self.client.post("/v1/messages", json=body)
        sent = self.local["requests"][0]["body"]
        self.assertEqual(self._markers(sent), [])

    async def test_the_upstream_arm_stays_a_pure_byte_pipe(self):
        body = self._conversation(model=REMOTE_MODEL)
        await self.client.post("/v1/messages", json=body)
        sent = self.upstream["requests"][0]["body"]
        self.assertEqual(self._markers(sent), [])

    async def test_an_unreadable_body_is_forwarded_unchanged(self):
        """Never turn a body we cannot parse into a 500."""
        resp = await self.client.post(
            "/v1/messages",
            data=b"not json at all",
            headers={"content-type": "application/json"},
        )
        # No model means no openrouter routing; the point is that the
        # router did not raise on the way.
        self.assertIn(resp.status, (200, 403))

    async def test_a_body_with_no_messages_is_forwarded_unchanged(self):
        body = {"model": self.OPENROUTER_MODEL, "max_tokens": 8, "messages": []}
        resp = await self.client.post("/v1/messages", json=body)
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.openrouter["requests"][0]["body"]["messages"], [])
