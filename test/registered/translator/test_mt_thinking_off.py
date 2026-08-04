# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Every MT request says thinking is off, explicitly, on the wire.

TWO independent reasons, either sufficient on its own:

* **Correctness.** A reasoning model's chain of thought has no marker this
  stage could reliably strip, and whatever survives is READ ALOUD in the
  speaker's own cloned voice. The measured instance: Qwen3.6-27B-FP8 answered
  a German->Spanish request with "Here's a thinking process: 1. Analyze User
  Input..." and hit the token limit before producing any translation.
* **Latency.** The #541 A/B put thinking at 2.5x wall and 3.7x tokens for
  equal quality. Every call on this path has a person waiting mid-sentence.

WHY EXPLICIT AND NOT A TEMPLATE DEFAULT, which is what this file exists to
pin: the default belongs to whichever chat template the SERVED CHECKPOINT
ships, and this tenant does not own that. The checkpoint behind
``--mt-model default`` can be swapped at any restart (runbook §14) and the
translator would never know. A field that is always on the wire cannot be
flipped by a change to someone else's file.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \\
      python -m pytest test/registered/translator/test_mt_thinking_off.py -v
"""

import asyncio
import unittest

from sglang.srt.translator.mt import MtConfig, OpenAiMt


class RecordingTransport:
    """Captures every request body the client would send."""

    def __init__(self, content: str = "hola") -> None:
        self.bodies = []
        self.content = content

    def post(self, _path, json=None, **_kwargs):
        self.bodies.append(json)
        transport = self

        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": transport.content}}]
                }

        async def _post():
            return Response()

        return _post()

    def thinking_flags(self):
        return [
            (b.get("chat_template_kwargs") or {}).get("enable_thinking", "ABSENT")
            for b in self.bodies
        ]


def client(**overrides):
    mt = OpenAiMt(MtConfig(base_url="http://127.0.0.1:30030/v1", **overrides))
    transport = RecordingTransport()
    mt._http = lambda: transport  # noqa: SLF001 - the seam under test
    return mt, transport


class TestThinkingIsAlwaysExplicit(unittest.TestCase):
    def test_a_translation_carries_the_flag(self):
        mt, transport = client()
        asyncio.run(mt.translate("Guten Tag", "de", "es"))
        self.assertEqual(transport.thinking_flags(), [False])

    def test_the_ask_path_carries_it_too(self):
        """A second body builder that forgot it is how 'always' becomes
        'usually'. The name extractor calls the same reasoning model."""
        mt, transport = client()
        asyncio.run(mt.ask("You classify names.", "Is 'Ana' a name?"))
        self.assertEqual(transport.thinking_flags(), [False])

    def test_the_default_config_is_off(self):
        self.assertFalse(MtConfig(base_url="x").enable_thinking)

    def test_an_absent_key_is_not_good_enough(self):
        """The point of the whole file, stated as an assertion.

        ``"ABSENT"`` is what the previous implementation produced when the
        launcher was given --mt-thinking, and an absent key hands the decision
        to the served checkpoint's template.
        """
        mt, transport = client()
        asyncio.run(mt.translate("Guten Tag", "de", "es"))
        self.assertNotIn("ABSENT", transport.thinking_flags())
        self.assertIn("chat_template_kwargs", transport.bodies[0])

    def test_opting_in_sends_true_rather_than_omitting_the_key(self):
        mt, transport = client(enable_thinking=True)
        asyncio.run(mt.translate("Guten Tag", "de", "es"))
        self.assertEqual(transport.thinking_flags(), [True])

    def test_other_template_kwargs_survive_the_stamp(self):
        """Merged, not replaced: an operator's template kwargs and this flag
        must not knock each other out."""
        mt, transport = client(
            extra_body={"chat_template_kwargs": {"add_generation_prompt": True}}
        )
        asyncio.run(mt.translate("Guten Tag", "de", "es"))
        kwargs = transport.bodies[0]["chat_template_kwargs"]
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["enable_thinking"])

    def test_extra_body_cannot_silently_drop_the_flag(self):
        """extra_body is applied first and the flag last, on purpose: an
        operator passing --mt-extra-body must not be able to turn thinking
        back on by accident."""
        mt, transport = client(
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        asyncio.run(mt.translate("Guten Tag", "de", "es"))
        self.assertEqual(transport.thinking_flags(), [False])

    def test_every_request_of_a_multi_turn_session_carries_it(self):
        mt, transport = client()
        for text in ("Guten Tag", "Wie geht es dir", "Bis bald"):
            asyncio.run(mt.translate(text, "de", "es"))
        self.assertEqual(transport.thinking_flags(), [False, False, False])


class TestLauncherWiring(unittest.TestCase):
    def test_the_launcher_default_is_thinking_off(self):
        from sglang.srt.translator.launch import build_parser

        self.assertFalse(build_parser().parse_args([]).mt_thinking)
        self.assertTrue(build_parser().parse_args(["--mt-thinking"]).mt_thinking)


if __name__ == "__main__":
    unittest.main()
