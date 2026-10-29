"""History tool calls with undecodable JSON arguments must not fail the turn (1029).

Mechanism: an agent client (OpenCode, Cline) replays the whole conversation on
every turn. One earlier assistant tool call whose ``function.arguments`` is
not decodable JSON -- an empty string for a call without parameters, a string
cut off by a broken stream, unescaped inner quotes -- used to raise
``ValueError`` in ``parse_tool_call_arguments`` and answer 400 for EVERY
later turn of that conversation. The call is now shown to the model with an
empty argument object (vLLM #48922 does the same); decodable arguments, and the
rejection of VALID JSON that is not an object, stay as they were.

No GPU, no model: run with CUDA_VISIBLE_DEVICES=99.
"""

from flliper.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from typing import Optional
from unittest.mock import Mock

from flliper.srt.entrypoints.openai.protocol import ChatCompletionRequest
from flliper.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _TokenizerManager:
    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
        )
        self.model_config.hf_config = Mock(architectures=["LlamaForCausalLM"])
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1


class _TemplateManager:
    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = "string"
        self.completion_template_name: Optional[str] = None
        self.reasoning_config = None
        self.force_reasoning = False


def _request(arguments: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="x",
        messages=[
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
    )


class TestHistoryToolCallArguments(unittest.TestCase):
    def setUp(self):
        self.tm = _TokenizerManager()
        self.chat = OpenAIServingChat(self.tm, _TemplateManager())

    def _rendered_arguments(self, arguments: str):
        self.tm.tokenizer.apply_chat_template.reset_mock()
        self.chat._process_messages(_request(arguments), is_multimodal=False)
        messages = self.tm.tokenizer.apply_chat_template.call_args.args[0]
        return messages[1]["tool_calls"][0]["function"]["arguments"]

    def test_undecodable_arguments_are_shown_without_arguments(self):
        for bad in ["", "   ", '{"command": "ls', '{"a": "x "y" z"}', "{'a': 1}"]:
            with self.subTest(arguments=bad):
                with self.assertLogs(
                    "flliper.srt.entrypoints.openai.serving_chat", level="WARNING"
                ):
                    self.assertEqual(self._rendered_arguments(bad), {})

    def test_decodable_arguments_are_unchanged(self):
        self.assertEqual(
            self._rendered_arguments('{"command": "ls", "run_in_background": true}'),
            {"command": "ls", "run_in_background": True},
        )
        self.assertEqual(self._rendered_arguments("{}"), {})

    def test_valid_json_that_is_not_an_object_is_still_rejected(self):
        # Pinned by test_serving_chat.test_jinja_rejects_non_object_tool_call_arguments:
        # only UNDECODABLE text is recovered; a decodable scalar/array is a
        # client error and keeps answering 400.
        for arguments in ['"ls"', '["ls"]', "42", "null"]:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                    self._rendered_arguments(arguments)


if __name__ == "__main__":
    unittest.main()
