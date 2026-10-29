"""Agent-client robustness of the Anthropic front (1029), over the real HTTP boundary.

Same harness as ``test_conformance_http`` (real FastAPI ``app`` through
``TestClient``, only the OpenAI chat handler faked). Each case guards a
defect found by comparing against the real-world failures listed in the
MegaCapybara changelog; see the per-class docstring for the mechanism.

No GPU, no model: run with CUDA_VISIBLE_DEVICES=99.
"""

import json
import re
import unittest

from flliper.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi import HTTPException  # noqa: E402
from fastapi.responses import ORJSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from flliper.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from flliper.srt.entrypoints.http_server import app  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

from test_conformance_http import _completion, _FakeChat  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HALF = "\ud83d"  # lone high surrogate

# Claude Code's trigger for compacting the conversation (external-source
# literal, as the real Anthropic API words it).
_CLAUDE_CODE_TOO_LONG = re.compile(
    r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)"
)
_TOKENIZER_REFUSAL = (
    "The input (300000 tokens) is longer than the model's context length "
    "(262144 tokens)."
)
_SCHEDULER_REFUSAL = (
    "Input length (300000 tokens) exceeds the maximum allowed length "
    "(262144 tokens). Use a shorter input or enable --allow-auto-truncate."
)


class _Case(unittest.TestCase):
    def _client(self, fake: _FakeChat) -> TestClient:
        previous = getattr(app.state, "anthropic_serving", None)
        app.state.anthropic_serving = AnthropicServing(fake)
        self.addCleanup(setattr, app.state, "anthropic_serving", previous)
        return TestClient(app, raise_server_exceptions=False)

    def _body(self, **overrides):
        body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        }
        body.update(overrides)
        return body


class TestLoneSurrogates(_Case):
    """A broken emoji (JSON ``"\\ud83d"`` without its low half) answered 500.

    Response side: a tool call whose parsed arguments carried the lone
    surrogate made ``JSONResponse`` raise UnicodeEncodeError. Request side:
    the text reached the tokenizer untouched (TypeError there) and, living in
    the client's history, failed every later turn.
    """

    def test_tool_call_arguments_with_broken_emoji_answer_200(self):
        calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Bash",
                    "arguments": json.dumps({"command": "echo x" + _HALF}, ensure_ascii=False),
                },
            }
        ]
        fake = _FakeChat(response=_completion(tool_calls=calls, finish_reason="tool_calls"))
        response = self._client(fake).post("/v1/messages", json=self._body())
        self.assertEqual(response.status_code, 200, response.text)
        block = [b for b in response.json()["content"] if b["type"] == "tool_use"][0]
        self.assertEqual(block["input"], {"command": "echo x�"})

    def test_broken_emoji_in_the_prompt_is_replaced_before_the_backend(self):
        fake = _FakeChat(response=_completion())
        # json.dumps escapes the lone surrogate as "\ud83d", as a JS client does.
        raw = json.dumps(
            self._body(messages=[{"role": "user", "content": "hi " + _HALF}])
        )
        response = self._client(fake).post(
            "/v1/messages",
            content=raw,
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        seen = fake.seen_chat_requests[0].messages[0].content
        self.assertEqual(seen, "hi �")

    def test_valid_request_and_response_are_unchanged(self):
        calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "Bash",
                    "arguments": json.dumps({"command": "echo \U0001f600"}, ensure_ascii=False),
                },
            }
        ]
        fake = _FakeChat(response=_completion(tool_calls=calls, finish_reason="tool_calls"))
        response = self._client(fake).post(
            "/v1/messages",
            json=self._body(messages=[{"role": "user", "content": "hi \U0001f600"}]),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            fake.seen_chat_requests[0].messages[0].content, "hi \U0001f600"
        )
        block = [b for b in response.json()["content"] if b["type"] == "tool_use"][0]
        self.assertEqual(block["input"], {"command": "echo \U0001f600"})


class _RefusingChat(_FakeChat):
    """Backend whose generation refuses the request the way the real one does."""

    def __init__(self, *, message, kind):
        super().__init__(response=None)
        self.message = message
        self.kind = kind

    async def _handle_non_streaming_request(self, *args):
        if self.kind == "http":
            raise HTTPException(status_code=400, detail=self.message)
        return ORJSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": self.message,
                    "type": "BadRequestError",
                    "code": 400,
                }
            },
        )

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        message = self.message

        async def _gen():
            raise ValueError(message)
            yield ""  # pragma: no cover -- makes this an async generator

        return _gen()


class TestPromptTooLong(_Case):
    """An input that does not fit the context never made Claude Code compact.

    The server answered with its own wording ("The input (N tokens) is longer
    than the model's context length ..." with error.type "BadRequestError");
    Claude Code only compacts on the Anthropic API's ``prompt is too long: N
    tokens > M maximum`` and stayed stuck on the same error. Both refusals
    (tokenizer manager, scheduler) are now worded like the real API.
    """

    def _assert_too_long(self, message: str):
        match = _CLAUDE_CODE_TOO_LONG.search(message)
        self.assertIsNotNone(match, message)
        self.assertEqual((match.group(1), match.group(2)), ("300000", "262144"))

    def test_tokenizer_refusal_non_streaming(self):
        fake = _RefusingChat(message=_TOKENIZER_REFUSAL, kind="response")
        response = self._client(fake).post("/v1/messages", json=self._body())
        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self._assert_too_long(error["message"])

    def test_scheduler_refusal_non_streaming(self):
        fake = _RefusingChat(message=_SCHEDULER_REFUSAL, kind="http")
        response = self._client(fake).post("/v1/messages", json=self._body())
        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self._assert_too_long(error["message"])

    def test_refusal_before_the_first_streamed_chunk(self):
        fake = _RefusingChat(message=_TOKENIZER_REFUSAL, kind="stream")
        response = self._client(fake).post(
            "/v1/messages", json=self._body(stream=True)
        )
        data = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        errors = [d for d in data if d.get("type") == "error"]
        self.assertEqual(len(errors), 1, response.text)
        self.assertEqual(errors[0]["error"]["type"], "invalid_request_error")
        self._assert_too_long(errors[0]["error"]["message"])

    def test_other_bad_requests_keep_their_message(self):
        fake = _RefusingChat(message="top_k must be positive", kind="http")
        response = self._client(fake).post("/v1/messages", json=self._body())
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["message"], "top_k must be positive")


if __name__ == "__main__":
    unittest.main()
