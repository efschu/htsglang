"""Anthropic Messages front conformance, driven over the REAL HTTP boundary.

Every case here posts to the actual FastAPI ``app`` from
``sglang.srt.entrypoints.http_server`` through ``TestClient``, with only the
OpenAI serving handler and its tokenizer manager mocked. That is deliberate:
the sibling ``test_serving.py`` calls ``AnthropicServing`` methods directly
and therefore cannot see anything the route layer does — request validation,
the Anthropic error envelope, or SSE framing. These are the gaps the #540
audit found, and each one is proven by a test that fails on the pre-change
tree (see docs/dev/TICKET_540_anthropic_front_conformance.md for the
before/after matrix).

No GPU, no model load: run with CUDA_VISIBLE_DEVICES=99.
"""

import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi.testclient import TestClient  # noqa: E402

from sglang.srt.entrypoints.anthropic import serving as anthropic_serving  # noqa: E402
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.http_server import app  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionResponse,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


# ---------- backend doubles ----------


class _FakeChat:
    """Stands in for OpenAIServingChat at the exact surface the front uses."""

    def __init__(
        self,
        *,
        stream_lines=None,
        response=None,
        chat_template=None,
        reasoning_always_on=False,
        stream_delay=0.0,
        trace=None,
    ):
        self.stream_lines = stream_lines or []
        self.response = response
        self.stream_delay = stream_delay
        self.trace = trace
        self.reasoning_always_on = reasoning_always_on
        self.apply_reasoning_calls: list[bool] = []
        self.seen_chat_requests: list[object] = []
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template),
            create_abort_task=lambda adapted_request: None,
        )

    # -- request plumbing --

    def _validate_request(self, chat_request):
        self.seen_chat_requests.append(chat_request)
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return self.response

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        import asyncio

        delay = self.stream_delay
        trace = self.trace

        async def _gen():
            for index, line in enumerate(self.stream_lines):
                if delay:
                    await asyncio.sleep(delay)
                if trace is not None:
                    trace.append(("backend_yield", index))
                yield line

        return _gen()

    # -- reasoning toggle --

    def apply_reasoning_enabled(self, chat_request, enabled):
        if self.reasoning_always_on and not enabled:
            raise ValueError(
                "Reasoning parser 'always-on-model' is always-on and cannot "
                "be disabled via Anthropic thinking"
            )
        self.apply_reasoning_calls.append(enabled)

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"


def _chunk(choices=None, usage=None):
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices or [],
    }
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


def _choice(delta, finish_reason=None, matched_stop=None):
    out = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    if matched_stop is not None:
        out["matched_stop"] = matched_stop
    return out


def _completion(
    *, content="hello", tool_calls=None, finish_reason="stop", matched_stop=None
):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-test",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "matched_stop": matched_stop,
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
    )


def _parse_sse(raw: str):
    """Return [(event_name, parsed_data_or_None), ...] in wire order."""
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                body = line[len("data: ") :].strip()
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = body
        if name is not None:
            events.append((name, payload))
    return events


def _drive_asgi(asgi_app, body: dict, trace: list) -> None:
    """POST ``body`` straight into the ASGI app, recording every wire send.

    Bypasses TestClient on purpose: its transport buffers the whole
    streaming response, which hides WHEN each frame left the app. Appends
    ``("wire", <event name>)`` per emitted SSE frame, interleaved with
    whatever the backend double appends.
    """
    import asyncio

    raw = json.dumps(body).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
        ],
        "client": ("127.0.0.1", 51000),
        "server": ("testserver", 80),
    }

    async def _run():
        delivered = False
        never = asyncio.Event()

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": raw, "more_body": False}
            # Starlette races the response against a disconnect watcher that
            # loops on receive(). Returning anything here would spin it; block
            # instead — the task group cancels this once the body is done.
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] != "http.response.body":
                return
            text = message.get("body", b"").decode("utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("event: "):
                    trace.append(("wire", line[len("event: ") :].strip()))

        # Bounded so a future regression fails loudly instead of hanging CI.
        await asyncio.wait_for(asgi_app(scope, receive, send), timeout=30)

    asyncio.run(_run())


class _FrontTestCase(unittest.TestCase):
    """Binds a fake backend onto the real app for the duration of a test."""

    def _client(self, fake: _FakeChat) -> TestClient:
        previous = getattr(app.state, "anthropic_serving", None)
        app.state.anthropic_serving = AnthropicServing(fake)
        self.addCleanup(setattr, app.state, "anthropic_serving", previous)
        return TestClient(app)

    def _body(self, **overrides):
        body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        }
        body.update(overrides)
        return body


# ---------- G1: extended thinking defaults to OFF ----------


class TestThinkingDefaultsOff(_FrontTestCase):
    def test_absent_thinking_disables_reasoning(self):
        """No ``thinking`` field must disable reasoning for THIS request.

        Pre-change the front only touched the toggle when ``thinking`` was
        present, so a boot with ``--reasoning-parser qwen3`` answered every
        plain Claude Code request with a thinking block that consumed the
        whole max_tokens budget.
        """
        fake = _FakeChat(response=_completion())
        client = self._client(fake)

        resp = client.post("/v1/messages", json=self._body())

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(fake.apply_reasoning_calls, [False])

    def test_absent_thinking_matches_explicit_disabled(self):
        """Absent and ``{"type":"disabled"}`` produce the SAME toggle call."""
        absent = _FakeChat(response=_completion())
        self._client(absent).post("/v1/messages", json=self._body())

        explicit = _FakeChat(response=_completion())
        self._client(explicit).post(
            "/v1/messages", json=self._body(thinking={"type": "disabled"})
        )

        self.assertEqual(absent.apply_reasoning_calls, explicit.apply_reasoning_calls)
        self.assertEqual(absent.apply_reasoning_calls, [False])

    def test_explicit_enabled_still_turns_reasoning_on(self):
        fake = _FakeChat(response=_completion())
        resp = self._client(fake).post(
            "/v1/messages",
            json=self._body(thinking={"type": "enabled", "budget_tokens": 1024}),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(fake.apply_reasoning_calls, [True])

    def test_always_on_model_still_serves_plain_requests(self):
        """The default-off override must not 400 an always-on reasoning model.

        This is the one deliberate divergence from "absent == disabled":
        an explicit ``{"type":"disabled"}`` is a request the model cannot
        honour and still raises, but an ABSENT field is not a request.
        """
        fake = _FakeChat(response=_completion(), reasoning_always_on=True)
        client = self._client(fake)

        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level="WARNING"
        ) as log:
            resp = client.post("/v1/messages", json=self._body())

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(any("thinking-off" in line for line in log.output))


# ---------- G2: unknown content blocks degrade per block ----------


class TestUnknownContentBlocks(_FrontTestCase):
    def test_unknown_block_type_does_not_reject_the_conversation(self):
        """A ``document`` block (or any future tag) must not 400 the request."""
        fake = _FakeChat(response=_completion())
        client = self._client(fake)

        body = self._body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "text", "data": "spec.pdf"},
                        },
                        {"type": "text", "text": "summarise the attachment"},
                    ],
                }
            ]
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level="WARNING"
        ) as log:
            resp = client.post("/v1/messages", json=body)

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(any("document" in line for line in log.output))
        # The rest of the turn is converted normally.
        chat_request = fake.seen_chat_requests[-1]
        self.assertEqual(chat_request.messages[-1].content, "summarise the attachment")

    def test_several_unknown_tags_all_pass(self):
        fake = _FakeChat(response=_completion())
        client = self._client(fake)
        for tag in (
            "mcp_tool_use",
            "server_tool_use",
            "web_search_tool_result",
            "container_upload",
        ):
            with self.subTest(tag=tag):
                body = self._body(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": tag, "id": "x1", "extra": {"a": 1}},
                                {"type": "text", "text": "go"},
                            ],
                        }
                    ]
                )
                resp = client.post("/v1/messages", json=body)
                self.assertEqual(resp.status_code, 200)

    def test_malformed_known_block_still_reports_a_precise_error(self):
        """A ``text`` block without ``text`` must NOT silently become unknown.

        The permissive fallback is only for tags we do not model; known tags
        keep their exact Pydantic validation error.
        """
        fake = _FakeChat(response=_completion())
        client = self._client(fake)

        body = self._body(messages=[{"role": "user", "content": [{"type": "text"}]}])
        resp = client.post("/v1/messages", json=body)

        self.assertEqual(resp.status_code, 400)
        payload = resp.json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("text", payload["error"]["message"])


# ---------- G6: redacted_thinking in history ----------


class TestRedactedThinkingHistory(_FrontTestCase):
    def test_redacted_thinking_does_not_reject_the_conversation(self):
        fake = _FakeChat(response=_completion())
        client = self._client(fake)

        body = self._body(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "AAAA"},
                        {"type": "text", "text": "previous answer"},
                    ],
                },
                {"role": "user", "content": "and now?"},
            ]
        )
        resp = client.post("/v1/messages", json=body)
        self.assertEqual(resp.status_code, 200)


# ---------- G3: stop_sequence reason and value ----------


class TestStopSequence(_FrontTestCase):
    def test_non_streaming_reports_the_matched_stop_sequence(self):
        fake = _FakeChat(response=_completion(content="draft ", matched_stop="<<END>>"))
        client = self._client(fake)

        resp = client.post("/v1/messages", json=self._body(stop_sequences=["<<END>>"]))

        payload = resp.json()
        self.assertEqual(payload["stop_reason"], "stop_sequence")
        self.assertEqual(payload["stop_sequence"], "<<END>>")

    def test_template_stop_is_not_reported_as_a_stop_sequence(self):
        """A stop string the CALLER did not ask for stays ``end_turn``."""
        fake = _FakeChat(
            response=_completion(content="draft", matched_stop="<|im_end|>")
        )
        client = self._client(fake)

        resp = client.post("/v1/messages", json=self._body(stop_sequences=["<<END>>"]))

        payload = resp.json()
        self.assertEqual(payload["stop_reason"], "end_turn")
        self.assertIsNone(payload["stop_sequence"])

    def test_streaming_message_delta_reports_the_stop_sequence(self):
        fake = _FakeChat(
            stream_lines=[
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "draft "})]),
                _chunk([_choice({}, finish_reason="stop", matched_stop="<<END>>")]),
                "data: [DONE]\n\n",
            ]
        )
        client = self._client(fake)

        resp = client.post(
            "/v1/messages",
            json=self._body(stream=True, stop_sequences=["<<END>>"]),
        )
        events = dict(_parse_sse(resp.text))

        self.assertEqual(
            events["message_delta"]["delta"]["stop_reason"], "stop_sequence"
        )
        self.assertEqual(events["message_delta"]["delta"]["stop_sequence"], "<<END>>")


# ---------- G4 + G7: streaming liveness and explicit nulls ----------


class TestStreamingLiveness(_FrontTestCase):
    def test_message_start_precedes_the_first_content_event(self):
        fake = _FakeChat(
            stream_lines=[
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "hi"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        client = self._client(fake)

        resp = client.post("/v1/messages", json=self._body(stream=True))
        self.assertEqual(resp.status_code, 200)
        names = [name for name, _ in _parse_sse(resp.text)]

        self.assertEqual(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )

    def test_message_start_reaches_the_wire_before_the_backend_speaks(self):
        """The liveness claim itself, not just the resulting event order.

        Ordering alone cannot catch the deferral defect: the old code also
        put message_start first — it just withheld it until the backend had
        produced a chunk. The distinguishing observation is the INTERLEAVING
        of the two sides, so this drives the ASGI app directly and records
        both wire sends and backend yields into one ordered trace. (A
        ``TestClient.stream`` timing check cannot see this: the test
        transport buffers the whole response before ``iter_lines`` returns,
        which would make a latency assertion vacuous.)
        """
        trace: list[tuple[str, object]] = []
        fake = _FakeChat(
            stream_lines=[
                _chunk([_choice({"role": "assistant", "content": "hi"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ],
            trace=trace,
        )
        previous = getattr(app.state, "anthropic_serving", None)
        app.state.anthropic_serving = AnthropicServing(fake)
        self.addCleanup(setattr, app.state, "anthropic_serving", previous)

        _drive_asgi(app, self._body(stream=True), trace)

        wire_start = next(
            i
            for i, (kind, value) in enumerate(trace)
            if kind == "wire" and value == "message_start"
        )
        first_backend = next(
            i for i, (kind, _) in enumerate(trace) if kind == "backend_yield"
        )
        self.assertLess(
            wire_start,
            first_backend,
            "message_start must reach the wire before the backend yields its "
            f"first chunk; trace was {trace[:6]}",
        )

    def test_message_start_carries_explicit_stop_nulls(self):
        fake = _FakeChat(
            stream_lines=[
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "hi"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        resp = self._client(fake).post("/v1/messages", json=self._body(stream=True))
        message = dict(_parse_sse(resp.text))["message_start"]["message"]

        self.assertIn("stop_reason", message)
        self.assertIsNone(message["stop_reason"])
        self.assertIn("stop_sequence", message)
        self.assertIsNone(message["stop_sequence"])

    def test_non_streaming_response_carries_explicit_stop_sequence_null(self):
        fake = _FakeChat(response=_completion())
        resp = self._client(fake).post("/v1/messages", json=self._body())
        payload = resp.json()

        self.assertIn("stop_sequence", payload)
        self.assertIsNone(payload["stop_sequence"])

    def test_ping_frames_are_emitted_while_the_backend_is_silent(self):
        """A slow backend must not leave the wire silent.

        The cadence constant is shortened for the test; the generator reads
        it at runtime, so patching the module attribute is enough.
        """
        original = anthropic_serving.PING_INTERVAL_SECONDS
        anthropic_serving.PING_INTERVAL_SECONDS = 0.02
        self.addCleanup(setattr, anthropic_serving, "PING_INTERVAL_SECONDS", original)

        fake = _FakeChat(
            stream_lines=[
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "hi"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ],
            stream_delay=0.12,
        )
        resp = self._client(fake).post("/v1/messages", json=self._body(stream=True))
        names = [name for name, _ in _parse_sse(resp.text)]

        self.assertIn("ping", names)
        # message_start still comes first, and pings never displace content.
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertEqual(
            [n for n in names if n != "ping"],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )


# ---------- G5: toolu_ id normalisation and round trip ----------


class TestToolUseIds(_FrontTestCase):
    def test_non_streaming_tool_use_id_is_anthropic_shaped(self):
        fake = _FakeChat(
            response=_completion(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            )
        )
        resp = self._client(fake).post("/v1/messages", json=self._body())
        payload = resp.json()

        blocks = [b for b in payload["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(blocks), 1)
        self.assertTrue(
            blocks[0]["id"].startswith("toolu_"),
            f"expected an Anthropic tool id, got {blocks[0]['id']!r}",
        )

    def test_streaming_tool_use_id_is_anthropic_shaped(self):
        fake = _FakeChat(
            stream_lines=[
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": "{}",
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        resp = self._client(fake).post("/v1/messages", json=self._body(stream=True))
        starts = [
            payload
            for name, payload in _parse_sse(resp.text)
            if name == "content_block_start"
        ]

        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0]["content_block"]["id"].startswith("toolu_"))

    def test_emitted_tool_id_is_accepted_back_on_tool_result(self):
        """The id we emit must be the id we accept — the round trip is the
        whole point of normalising outgoing ids."""
        first = _FakeChat(
            response=_completion(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            )
        )
        payload = self._client(first).post("/v1/messages", json=self._body()).json()
        emitted_id = next(
            b["id"] for b in payload["content"] if b["type"] == "tool_use"
        )

        second = _FakeChat(response=_completion())
        client = self._client(second)
        resp = client.post(
            "/v1/messages",
            json=self._body(
                messages=[
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": emitted_id,
                                "name": "get_weather",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": emitted_id,
                                "content": "sunny",
                            }
                        ],
                    },
                ]
            ),
        )

        self.assertEqual(resp.status_code, 200)
        chat_request = second.seen_chat_requests[-1]
        tool_messages = [
            m for m in chat_request.messages if getattr(m, "role", None) == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, emitted_id)

    def test_arbitrary_inbound_tool_ids_are_never_rewritten(self):
        """Older clients and replayed transcripts carry foreign id shapes."""
        fake = _FakeChat(response=_completion())
        client = self._client(fake)
        resp = client.post(
            "/v1/messages",
            json=self._body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "legacy-id-42",
                                "content": "ok",
                            }
                        ],
                    }
                ]
            ),
        )

        self.assertEqual(resp.status_code, 200)
        chat_request = fake.seen_chat_requests[-1]
        tool_messages = [
            m for m in chat_request.messages if getattr(m, "role", None) == "tool"
        ]
        self.assertEqual(tool_messages[0].tool_call_id, "legacy-id-42")


if __name__ == "__main__":
    unittest.main()
