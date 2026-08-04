"""
Unit tests for the server-wide chat-template kwargs default
(``--chat-template-default-kwargs``) and, specifically, what it buys us:
``preserve_thinking`` as a serving default.

Why this matters (#544): a multi-turn agent replays the whole conversation on
every turn. If the renderer drops prior-turn think blocks while the model
actually generated them, the turn-N prompt is no longer a prefix of
"turn-(N-1) prompt + what the model produced", so the radix/KV prefix cache
misses and the server re-prefills almost the entire context every turn. With
``preserve_thinking`` the think blocks stay in, and the prefix matches
byte-for-byte.

The prefix-identity test renders through the real Qwen3.6 chat template, so it
is skipped when that tokenizer is not on this box. Everything else is pure
CPU and needs neither a tokenizer nor a server.

Run with:
    python3 -m pytest \
      test/registered/unit/entrypoints/openai/test_chat_template_default_kwargs_544.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import json
import os
import unittest

from sglang.srt.entrypoints.openai.serving_chat import merge_chat_template_kwargs
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

TOKENIZER_PATH = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8"

# One simulated agent turn: the model emitted a think block and then an answer.
THINK_TEXT = "The user wants the capital. I know this one."
ANSWER_TEXT = "Paris."


def _generated_stream() -> str:
    """Exactly what the model emits for turn 1.

    The generation prompt already ends in ``<think>\n``, so the stream starts
    with the reasoning body -- not with a ``<think>`` tag.
    """
    return f"{THINK_TEXT}\n</think>\n\n{ANSWER_TEXT}"


class TestMergeChatTemplateKwargs(CustomTestCase):
    """Precedence rules for the three kwargs sources."""

    def test_defaults_applied_when_request_is_silent(self):
        merged = merge_chat_template_kwargs({"preserve_thinking": True}, None, None)
        self.assertEqual(merged, {"preserve_thinking": True})

    def test_request_overrides_default_key_by_key(self):
        merged = merge_chat_template_kwargs(
            {"preserve_thinking": True, "other": 1},
            None,
            {"preserve_thinking": False},
        )
        # The explicit client choice wins; unrelated defaults survive.
        self.assertEqual(merged, {"preserve_thinking": False, "other": 1})

    def test_reasoning_effort_is_layered_between_default_and_request(self):
        merged = merge_chat_template_kwargs({"reasoning_effort": "low"}, "high", None)
        self.assertEqual(merged["reasoning_effort"], "high")
        merged = merge_chat_template_kwargs({}, "high", {"reasoning_effort": "low"})
        self.assertEqual(merged["reasoning_effort"], "low")

    def test_no_defaults_is_todays_behaviour(self):
        # Byte-identical to the pre-#544 path: nothing injected.
        self.assertEqual(merge_chat_template_kwargs({}, None, None), {})
        self.assertEqual(
            merge_chat_template_kwargs({}, None, {"enable_thinking": True}),
            {"enable_thinking": True},
        )

    def test_defaults_are_not_mutated_by_a_request(self):
        defaults = {"preserve_thinking": True}
        merge_chat_template_kwargs(defaults, "high", {"preserve_thinking": False})
        self.assertEqual(defaults, {"preserve_thinking": True})


class TestServerArgValidation(CustomTestCase):
    """--chat-template-default-kwargs must fail fast on malformed input."""

    def _validate(self, raw):
        args = ServerArgs.__new__(ServerArgs)
        args.chat_template_default_kwargs = raw
        args._handle_chat_template_default_kwargs()

    def test_valid_json_object_accepted(self):
        self._validate('{"preserve_thinking": true}')

    def test_none_accepted(self):
        self._validate(None)

    def test_malformed_json_rejected(self):
        with self.assertRaises(ValueError) as cm:
            self._validate("{preserve_thinking: true}")
        self.assertIn("valid JSON", str(cm.exception))

    def test_non_object_json_rejected(self):
        for raw in ("[1, 2]", '"preserve_thinking"', "42"):
            with self.assertRaises(ValueError) as cm:
                self._validate(raw)
            self.assertIn("JSON object", str(cm.exception))


@unittest.skipUnless(
    os.path.isdir(TOKENIZER_PATH), f"tokenizer not present at {TOKENIZER_PATH}"
)
class TestPreserveThinkingPrefixIdentity(CustomTestCase):
    """The point of the flag: turn-2 prompt must extend the turn-1 stream.

    Simulates the two renders an agent triggers across one tool/answer
    roundtrip and checks whether

        render(turn 1) + "<think>...</think>answer" + turn-2 framing

    is a genuine prefix relationship. Without ``preserve_thinking`` the turn-2
    render strips the think block and the prefix breaks; with it, it holds.
    """

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, trust_remote_code=True
        )

    def _render(self, messages, **template_kwargs):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
            **template_kwargs,
        )

    @property
    def _turn1_messages(self):
        return [{"role": "user", "content": "What is the capital of France?"}]

    @property
    def _turn2_messages(self):
        return [
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": ANSWER_TEXT,
                "reasoning_content": THINK_TEXT,
            },
            {"role": "user", "content": "And of Spain?"},
        ]

    def test_template_understands_preserve_thinking(self):
        with_pt = self._render(self._turn2_messages, preserve_thinking=True)
        without_pt = self._render(self._turn2_messages)
        self.assertIn(THINK_TEXT, with_pt)
        self.assertNotIn(THINK_TEXT, without_pt)

    def test_prefix_identity_holds_with_preserve_thinking(self):
        turn1 = self._render(self._turn1_messages, preserve_thinking=True)
        turn2 = self._render(self._turn2_messages, preserve_thinking=True)
        # What the server actually streamed back for turn 1.
        generated = _generated_stream()
        self.assertTrue(
            turn2.startswith(turn1 + generated),
            "turn-2 prompt must extend 'turn-1 prompt + generated stream' "
            f"byte-for-byte.\nturn1+gen:\n{turn1 + generated!r}\n"
            f"turn2:\n{turn2[: len(turn1 + generated) + 80]!r}",
        )

    def test_prefix_identity_breaks_without_preserve_thinking(self):
        # The regression this flag exists to prevent. If this ever passes, the
        # template changed and the default may no longer be load-bearing.
        turn1 = self._render(self._turn1_messages)
        turn2 = self._render(self._turn2_messages)
        generated = _generated_stream()
        self.assertFalse(turn2.startswith(turn1 + generated))

    def test_token_level_common_prefix_is_longer_with_preserve_thinking(self):
        """Same claim in the unit that actually decides KV reuse: tokens."""

        def common_prefix_len(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n

        generated = _generated_stream()

        def measure(**kw):
            turn1 = self._render(self._turn1_messages, **kw)
            turn2 = self._render(self._turn2_messages, **kw)
            stream = self.tokenizer.encode(turn1 + generated)
            nxt = self.tokenizer.encode(turn2)
            return common_prefix_len(stream, nxt), len(stream)

        pt_hit, pt_total = measure(preserve_thinking=True)
        no_hit, _ = measure()

        self.assertGreater(pt_hit, no_hit)
        # With preservation the whole previous turn is reusable.
        self.assertEqual(pt_hit, pt_total)


class TestBootConfigIsParseable(CustomTestCase):
    """The exact JSON we intend to boot with must survive validation."""

    def test_intended_boot_value(self):
        raw = '{"preserve_thinking": true}'
        args = ServerArgs.__new__(ServerArgs)
        args.chat_template_default_kwargs = raw
        args._handle_chat_template_default_kwargs()
        self.assertEqual(json.loads(raw), {"preserve_thinking": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
