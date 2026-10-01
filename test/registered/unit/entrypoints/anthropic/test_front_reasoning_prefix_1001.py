"""/v1/messages thinking history renders like /v1/chat/completions.

Real Qwen3.8-27B template and tokenizer: the tokens of round 1 (prompt plus
the generation, thinking included) must be a prefix of the round-2 prompt
after the Anthropic front converted the replayed history. Before the fix the
front spliced re-wrapped thinking into ``content``; the template then rendered
an empty think block followed by the wrapped text and the prefix broke at the
previous assistant turn, so every follow-up recomputed that whole turn.

The rendering goes through the real ``OpenAIServingChat._apply_jinja_template``
on a minimal stand-in object, so the chat side's handling of the assistant
``reasoning_content`` field is part of what is tested.
"""

import logging
import os
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: E402
    OpenAIServingChat,
)
from sglang.srt.parser.jinja_template_utils import (  # noqa: E402
    detect_jinja_template_content_format,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODEL_DIR = os.environ.get(
    "SGLANG_TEST_QWEN38_27B_DIR",
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8",
)

_TOOLS = [
    {
        "name": "ls",
        "description": "List files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]

REASONING = "The user wants the files listed. I should call ls on the cwd."
ANSWER = "Listing the directory."


class _RealTemplateChat:
    """Just enough of OpenAIServingChat to run its real Jinja path."""

    supports_native_reasoning_history = (
        OpenAIServingChat.supports_native_reasoning_history
    )
    wrap_reasoning_history = OpenAIServingChat.wrap_reasoning_history
    _apply_jinja_template = OpenAIServingChat._apply_jinja_template
    _encode_messages = OpenAIServingChat._encode_messages
    _handle_last_assistant_message = OpenAIServingChat._handle_last_assistant_message
    _append_assistant_prefix_to_prompt_ids = (
        OpenAIServingChat._append_assistant_prefix_to_prompt_ids
    )

    def __init__(self, tokenizer):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=tokenizer,
            server_args=SimpleNamespace(served_model_name="Qwen3.8-27B"),
        )
        self.template_manager = SimpleNamespace(
            jinja_template_content_format=detect_jinja_template_content_format(
                tokenizer.chat_template
            ),
            reasoning_config=None,
        )
        self.chat_encoding_spec = None
        self.default_chat_template_kwargs = {}
        self._tokenizer_auto_adds_specials = False
        self._reasoning_detector = SimpleNamespace(
            think_start_token="<think>",
            think_start_self_label="",
            think_end_token="</think>",
        )

    def apply_reasoning_enabled(self, request, enabled):
        request.chat_template_kwargs = dict(
            request.chat_template_kwargs or {}, enable_thinking=enabled
        )


@unittest.skipUnless(
    os.path.isfile(os.path.join(MODEL_DIR, "chat_template.jinja")),
    f"Qwen3.8-27B template not available at {MODEL_DIR}",
)
class TestFrontReasoningPrefix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        cls.chat = _RealTemplateChat(cls.tok)
        cls.serving = AnthropicServing(cls.chat)

    def _ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def _prompt_ids(self, messages, tools=True):
        body = {
            "model": "Qwen3.8-27B",
            "max_tokens": 1024,
            "system": "You are a terse coding agent.",
            "messages": messages,
            "thinking": {"type": "adaptive"},
        }
        if tools:
            body["tools"] = _TOOLS
        request = AnthropicMessagesRequest(**body)
        chat_request = self.serving._convert_to_chat_completion_request(request)
        tools_dump = (
            [t.model_dump() for t in chat_request.tools] if chat_request.tools else None
        )
        return self.chat._apply_jinja_template(
            chat_request, tools_dump, False
        ).prompt_ids

    def _assert_prefix(self, round1, round2):
        n = 0
        while n < min(len(round1), len(round2)) and round1[n] == round2[n]:
            n += 1
        self.assertEqual(
            n,
            len(round1),
            "round-1 tokens are not a prefix of the round-2 prompt: diverge at "
            f"{n}/{len(round1)}: round1={self.tok.decode(round1[n:n + 12])!r} "
            f"round2={self.tok.decode(round2[n:n + 12])!r}",
        )

    def test_template_is_detected_as_native(self):
        self.assertTrue(self.chat.supports_native_reasoning_history())

    def test_tool_loop_round2_extends_round1(self):
        """(i) tool loop: thinking + text + tool_use, then the tool_result."""
        user = {"role": "user", "content": "List the files."}
        p1 = self._prompt_ids([user])
        self.assertEqual(self.tok.decode(p1[-5:]), "<|im_start|>assistant\n<think>\n")
        generation = self._ids(
            REASONING
            + "\n</think>\n\n"
            + ANSWER
            + "\n\n<tool_call>\n<function=ls>\n<parameter=path>\n.\n</parameter>\n"
            "</function>\n</tool_call>"
            + "<|im_end|>"
        )
        p2 = self._prompt_ids(
            [
                user,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": REASONING},
                        {"type": "text", "text": ANSWER},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "ls",
                            "input": {"path": "."},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "a.py\nb.py",
                        }
                    ],
                },
            ]
        )
        self._assert_prefix(p1 + generation, p2)
        self.assertGreater(len(p2), len(p1) + len(generation))

    def test_text_turn_round2_extends_round1(self):
        """(i) plain turn: thinking + text, then a new user message."""
        user = {"role": "user", "content": "What is 2+2?"}
        p1 = self._prompt_ids([user], tools=False)
        generation = self._ids("2+2 is 4.\n</think>\n\n4<|im_end|>")
        p2 = self._prompt_ids(
            [
                user,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "2+2 is 4."},
                        {"type": "text", "text": "4"},
                    ],
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            tools=False,
        )
        self._assert_prefix(p1 + generation, p2)

    def _with_wrap_path(self, fn):
        """Run ``fn`` with the old wrap-into-content path forced."""
        self.chat.supports_native_reasoning_history = lambda: False
        try:
            return fn()
        finally:
            del self.chat.supports_native_reasoning_history

    def test_without_thinking_prompt_is_unchanged(self):
        """(ii) no thinking block: same prompt as the old path."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            {"role": "user", "content": "again"},
        ]
        self.assertEqual(
            self._prompt_ids(messages),
            self._with_wrap_path(lambda: self._prompt_ids(messages)),
        )

    def test_redacted_thinking_is_skipped(self):
        """(iii) redacted_thinking: skipped with a warning, never rendered."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "OPAQUEBYTES"},
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            ids = self._prompt_ids(messages)
        self.assertTrue(any("redacted_thinking" in line for line in log.output))
        self.assertNotIn("OPAQUEBYTES", self.tok.decode(ids))
        plain = [
            messages[0],
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            messages[2],
        ]
        self.assertEqual(ids, self._prompt_ids(plain))


if __name__ == "__main__":
    unittest.main()
