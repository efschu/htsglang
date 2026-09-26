"""Befund M (26.09.): inline ``role: "system"`` messages vs. the prefix cache.

Claude Code sends mid-conversation system messages (``api_system``: tool
additions/removals, output_config). For a chat template that only takes a
system turn at the head (Qwen3.8), the front hoists them into that head turn,
so the first turn that carries one re-renders everything behind the system
text. Measured on D: radix WALK-STOP and arena key chain both stop at the end
of the system turn (depth 3876 / 4655, stored ``(13, 248046)`` = ".<|im_end|>",
new ``(13, 198)`` = ".\\n" -- the "\\n".join of the hoist), census
``MambaComponent:absent``, W31, X-REQUEUE, one P epoch per such turn.

``SGLANG_ANTHROPIC_INLINE_SYSTEM_IN_PLACE=1`` renders non-leading inline system
messages in place as a user turn, so turn N stays a prefix of turn N+1.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

import jinja2  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# The head-only system shape of the Qwen3.8 template (chat_template.jinja):
# one system turn first, ``content|trim``, and a raise for any later one.
HEAD_ONLY_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content|trim + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{%- if not loop.first %}"
    "{{- raise_exception('System message must be at the beginning.') }}"
    "{%- endif %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content|trim + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
)
INLINE_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- message.role }}: {{ message.content }}\n"
    "{%- endfor %}"
)


class _FakeChat:
    def __init__(self, chat_template):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )

    def apply_reasoning_enabled(self, chat_request, enabled):
        pass

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"


def _raise(msg):
    raise ValueError(msg)


def _render(chat_request) -> str:
    env = jinja2.Environment()
    env.globals["raise_exception"] = _raise
    msgs = [
        {"role": m.role, "content": m.content if isinstance(m.content, str) else ""}
        for m in chat_request.messages
    ]
    return env.from_string(HEAD_ONLY_TEMPLATE).render(messages=msgs)


def _req(messages):
    return AnthropicMessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 8,
            "system": "You are an agent.",
            "messages": messages,
        }
    )


TURN_N = [
    {"role": "user", "content": "task"},
    {"role": "assistant", "content": "calling a tool"},
    {"role": "user", "content": "tool output 1"},
]
# Turn N+1: the client inserted an api_system message, then continued.
TURN_N1 = TURN_N + [
    {"role": "assistant", "content": "next step"},
    {"role": "system", "content": "Tool Foo is now available."},
    {"role": "user", "content": "tool output 2"},
]


class TestInlineSystemInPlace(unittest.TestCase):
    def _serving(self, template, on):
        with envs.SGLANG_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.override(on):
            return AnthropicServing(_FakeChat(template))

    def test_default_off_hoists_and_breaks_the_prefix(self):
        """Default unchanged: every inline system message goes to the head,
        which is exactly the prefix break of Befund M."""
        s = self._serving(HEAD_ONLY_TEMPLATE, False)
        self.assertTrue(s._merge_inline_system)
        self.assertFalse(s._inline_system_in_place)
        n = s._convert_to_chat_completion_request(_req(TURN_N))
        n1 = s._convert_to_chat_completion_request(_req(TURN_N1))
        self.assertEqual(
            n1.messages[0].content, "You are an agent.\nTool Foo is now available."
        )
        self.assertNotIn("system", [m.role for m in n1.messages[1:]])
        rn, rn1 = _render(n), _render(n1)
        self.assertFalse(rn1.startswith(rn))
        # The break sits at the end of the system text: "." then "\n", where
        # turn N had "." then "<|im_end|>" -- the measured WALK-STOP bigram.
        cut = len("<|im_start|>system\nYou are an agent.")
        self.assertEqual(rn[:cut], rn1[:cut])
        self.assertTrue(rn[cut:].startswith("<|im_end|>"))
        self.assertEqual(rn1[cut], "\n")

    def test_on_renders_in_place_and_keeps_the_prefix(self):
        s = self._serving(HEAD_ONLY_TEMPLATE, True)
        self.assertTrue(s._inline_system_in_place)
        n = s._convert_to_chat_completion_request(_req(TURN_N))
        n1 = s._convert_to_chat_completion_request(_req(TURN_N1))
        self.assertEqual(n1.messages[0].role, "system")
        self.assertEqual(n1.messages[0].content, "You are an agent.")
        roles = [m.role for m in n1.messages]
        self.assertEqual(
            roles, ["system", "user", "assistant", "user", "assistant", "user", "user"]
        )
        self.assertEqual(
            n1.messages[5].content,
            "<system-reminder>\nTool Foo is now available.\n</system-reminder>",
        )
        rn, rn1 = _render(n), _render(n1)
        self.assertTrue(rn1.startswith(rn))

    def test_on_leading_inline_system_still_hoisted(self):
        """A leading inline system run is as stable as the ``system`` field:
        it stays in the head turn, and a later one renders in place."""
        s = self._serving(HEAD_ONLY_TEMPLATE, True)
        msgs = [
            {"role": "system", "content": "Lead rule."},
            {"role": "user", "content": "go"},
            {"role": "system", "content": "Later rule."},
            {"role": "user", "content": "more"},
        ]
        r = s._convert_to_chat_completion_request(_req(msgs))
        self.assertEqual(r.messages[0].content, "You are an agent.\nLead rule.")
        self.assertEqual([m.role for m in r.messages], ["system", "user", "user", "user"])
        self.assertIn("Later rule.", r.messages[2].content)
        _render(r)  # the head-only template must accept the result

    def test_on_empty_inline_system_is_dropped(self):
        s = self._serving(HEAD_ONLY_TEMPLATE, True)
        msgs = [
            {"role": "user", "content": "go"},
            {"role": "system", "content": "   "},
            {"role": "user", "content": "more"},
        ]
        r = s._convert_to_chat_completion_request(_req(msgs))
        self.assertEqual([m.role for m in r.messages], ["system", "user", "user"])

    def test_on_inline_capable_template_is_untouched(self):
        """A template that takes system turns anywhere never merges, so the
        switch has nothing to do there."""
        s = self._serving(INLINE_TEMPLATE, True)
        self.assertFalse(s._merge_inline_system)
        r = s._convert_to_chat_completion_request(_req(TURN_N1))
        self.assertEqual(
            [m.role for m in r.messages],
            ["system", "user", "assistant", "user", "assistant", "system", "user"],
        )


if __name__ == "__main__":
    unittest.main()
