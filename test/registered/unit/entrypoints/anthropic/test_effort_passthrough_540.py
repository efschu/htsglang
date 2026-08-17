"""#540: the Anthropic effort knob must not rewrite a supported request.

THE DEFECT THIS CLOSES. ``reasoning_effort`` is injected into the chat-template
kwargs (``openai/serving_chat.py:175``), so whatever the Anthropic front puts
there reaches the template verbatim. Every Qwen3.8 checkpoint on this box ends
its effort handling with

    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ...') }}

so an unsupported value is not a soft downgrade -- it raises inside the render
and surfaces as HTTP 500. The front used to collapse ``xhigh`` onto ``max``,
which meant a client asking for the one top-tier value the model DOES accept
had its request rewritten into one the model rejects, while the same tier was
reachable by simply omitting the field (the template defaults to ``xhigh``).

The fix is passthrough, and the interesting property is not "xhigh equals
xhigh" -- it is that the value we emit is one the deployed template will
accept. ``test_the_emitted_value_is_accepted_by_the_real_template`` asserts
exactly that, against the supported set parsed out of the checkpoint's own
``chat_template.jinja``, so the test tracks the model rather than restating
this module.

Hermetic: reads a template FILE at most, never loads a model, never starts a
server, never touches a GPU.
"""

import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

#: A deployed checkpoint whose template constrains the effort vocabulary. Used
#: only if present; the assertion degrades to a skip elsewhere.
_TEMPLATE = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8/chat_template.jinja"
)


class _FakeChat:
    def __init__(self):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=None)
        )

    def apply_reasoning_enabled(self, chat_request, enabled):
        pass


def _template_effort_vocabulary(path):
    """The values the checkpoint's own template accepts, parsed from it."""
    with open(path) as handle:
        text = handle.read()
    match = re.search(r"reasoning_effort not in \(([^)]*)\)", text)
    if not match:
        return None
    return {v.strip().strip("'\"") for v in match.group(1).split(",") if v.strip()}


class TestEffortPassthrough(CustomTestCase):
    def _convert(self, **overrides):
        data = {
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
        data.update(overrides)
        request = AnthropicMessagesRequest.model_validate(data)
        return AnthropicServing(_FakeChat())._convert_to_chat_completion_request(
            request
        )

    def test_xhigh_is_passed_through_not_rewritten(self):
        """The fix. Previously this emitted 'max', which the deployed template
        rejects with raise_exception -> HTTP 500."""
        chat = self._convert(output_config={"effort": "xhigh"})
        self.assertEqual(chat.reasoning_effort, "xhigh")

    def test_the_emitted_value_is_accepted_by_the_real_template(self):
        """The deployed-family case, checked against the checkpoint itself.

        This is the assertion that would have caught the defect: it does not
        care what the mapping is, only that what we send is renderable.
        """
        if not os.path.exists(_TEMPLATE):
            self.skipTest(f"checkpoint template not present: {_TEMPLATE}")
        vocabulary = _template_effort_vocabulary(_TEMPLATE)
        self.assertIsNotNone(vocabulary, "template no longer constrains effort")

        emitted = self._convert(output_config={"effort": "xhigh"}).reasoning_effort
        self.assertIn(
            emitted,
            vocabulary,
            f"the front emits {emitted!r}, which this checkpoint's template "
            f"rejects (it accepts {sorted(vocabulary)}) -- that render raises "
            f"and the client sees HTTP 500",
        )

    def test_the_tiers_the_deployed_family_supports_pass_through(self):
        for effort in ("medium", "low"):
            with self.subTest(effort=effort):
                chat = self._convert(output_config={"effort": effort})
                self.assertEqual(chat.reasoning_effort, effort)

    def test_omitting_effort_still_sends_nothing(self):
        """Omission is the template's own default path and must stay untouched."""
        self.assertIsNone(getattr(self._convert(), "reasoning_effort", None))

    def test_high_is_still_the_clients_own_value(self):
        """Deliberately NOT remapped. 'high' is a legitimate OpenAI tier that
        this model family does not implement; silently promoting it to 'xhigh'
        would answer a different question than the client asked. The template's
        raise names the supported set, which is the honest outcome -- turning
        that into a 400 needs template introspection and is a separate cut."""
        chat = self._convert(output_config={"effort": "high"})
        self.assertEqual(chat.reasoning_effort, "high")

    def test_the_collapse_survives_as_an_opt_in(self):
        """A deployment whose template names its top tier 'max' can restore the
        old behaviour -- explicitly, and it is logged."""
        with patch.dict(os.environ, {"SGLANG_ANTHROPIC_XHIGH_EFFORT": "max"}):
            chat = self._convert(output_config={"effort": "xhigh"})
        self.assertEqual(chat.reasoning_effort, "max")

    def test_the_opt_in_collapse_is_logged_by_name(self):
        with patch.dict(os.environ, {"SGLANG_ANTHROPIC_XHIGH_EFFORT": "max"}):
            with self.assertLogs(
                "sglang.srt.entrypoints.anthropic.serving", level="INFO"
            ) as captured:
                self._convert(output_config={"effort": "xhigh"})
        joined = "\n".join(captured.output)
        self.assertIn("SGLANG_ANTHROPIC_XHIGH_EFFORT", joined)

    def test_an_empty_override_does_not_blank_the_effort(self):
        """An unset-but-present env var must not turn into an empty effort."""
        with patch.dict(os.environ, {"SGLANG_ANTHROPIC_XHIGH_EFFORT": ""}):
            chat = self._convert(output_config={"effort": "xhigh"})
        self.assertEqual(chat.reasoning_effort, "xhigh")


if __name__ == "__main__":
    unittest.main()
