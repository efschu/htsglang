# SPDX-License-Identifier: Apache-2.0
"""REUSE-TEXT-IDS-1007 (Hebel 2a): group D hands a text-only chat request's
rendered ids to the tokenizer manager instead of decoding them to text that
the tokenizer manager encodes again.

Backport of upstream ``_can_reuse_text_only_prompt_ids`` with its
``_prompt_text_round_trip_is_lossy`` probe, armed only by
``SGLANG_ENABLE_REUSE_TEXT_ONLY_PROMPT_IDS`` on group D, plus a decode probe:
the old path's ids are ``encode(decode(ids))``, so the reuse is the same
request only where that round trip is the identity. Desk measurement (NF
tokenizer, 142762 tokens): re-encode 242 ms per turn.

Pinned on the real NF tokenizer and chat template: the reused ids are the old
path's ids to the token; images, a continued final message and a waiting #1442
hand-off keep the text; off, or on any other group, nothing changes.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.entrypoints.openai import serving_chat as SC  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402

CKPT = os.environ.get("WEG2_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "tokenizer.json")) and \
    os.path.isfile(os.path.join(CKPT, "chat_template.jinja"))

MESSAGES = [
    {"role": "system", "content": "Du bist ein Agent ; antworte kurz ."},
    {"role": "user", "content": "Lies  die Datei:\n\n\tdef f(x) :\n\t\treturn x ** 2  # ok ?"},
    {"role": "assistant", "content": "Gelesen . Sie quadriert x , n'est-ce pas ?"},
    {"role": "user", "content": "Café 日本語 \U0001f642 ... und weiter !"},
]
IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=="


class _LossyDecode:
    """The NF tokenizer whose decode cleans spaces before punctuation, as
    ``clean_up_tokenization_spaces`` would: the round trip is not the identity."""

    def __init__(self, tok):
        self._tok = tok

    def decode(self, ids, **kw):
        return self._tok.decode(ids, **kw).replace(" ,", ",").replace(" .", ".")

    def __getattr__(self, name):
        return getattr(self._tok, name)


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class ReuseTextIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        cls.tok = get_tokenizer(CKPT, tokenizer_mode="auto", trust_remote_code=True)

    def _chat(self, *, on=True, group="D", tok=None):
        from sglang.srt.parser.template_manager import TemplateManager
        from sglang.srt.weg2.front_tokens import _hf_config_stub, _server_args_namespace

        ns = _server_args_namespace({"model_path": CKPT, "tokenizer_path": CKPT})
        mc = SimpleNamespace(hf_config=_hf_config_stub(CKPT), is_multimodal=True,
                             get_default_sampling_params=lambda: {})
        tm = SimpleNamespace(tokenizer=tok or self.tok, processor=None, server_args=ns,
                             model_config=mc, model_path=CKPT)
        templates = TemplateManager()
        templates.initialize_templates(tm, model_path=CKPT, chat_template=None,
                                       completion_template=None)
        with envs.SGLANG_ENABLE_REUSE_TEXT_ONLY_PROMPT_IDS.override(on), \
                mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": group}):
            return SC.OpenAIServingChat(tm, templates)

    def _convert(self, chat, messages=MESSAGES, **kw):
        req = ChatCompletionRequest(model="m", messages=messages, rid="weg2-4-11", **kw)
        return chat._convert_to_internal_request(req)[0]

    def test_text_turn_carries_exactly_the_ids_the_text_path_tokenizes(self):
        old = self._convert(self._chat(on=False))
        self.assertIsNone(old.input_ids)
        new = self._convert(self._chat())
        self.assertIsNone(new.text)
        self.assertEqual(new.input_ids, self.tok(old.text)["input_ids"])

    def test_off_or_another_group_keeps_the_text(self):
        for kw in ({"on": False}, {"group": "P"}, {"group": ""}):
            with self.subTest(**kw):
                chat = self._chat(**kw)
                self.assertFalse(chat._reuse_text_only_prompt_ids)
                got = self._convert(chat)
                self.assertIsNone(got.input_ids)
                self.assertIsInstance(got.text, str)

    def test_images_continuation_and_handoff_keep_the_text(self):
        chat = self._chat()
        self.assertTrue(chat._reuse_text_only_prompt_ids)
        with_image = MESSAGES[:-1] + [{"role": "user", "content": [
            {"type": "text", "text": "Was ist das ?"},
            {"type": "image_url", "image_url": {"url": IMAGE}}]}]
        cont = MESSAGES + [{"role": "assistant", "content": "Also"}]
        cases = {
            "image": dict(messages=with_image),
            "continue_final_message": dict(messages=cont, continue_final_message=True),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                got = self._convert(chat, **kw)
                self.assertIsNone(got.input_ids)
                self.assertIsInstance(got.text, str)
        with mock.patch.object(SC.weg2_handoff, "exists", return_value=True) as ex:
            got = self._convert(chat)
        ex.assert_called_once_with("weg2-4-11")
        self.assertIsNone(got.input_ids)

    def test_a_lossy_round_trip_is_not_armed(self):
        self.assertFalse(self._chat(tok=_LossyDecode(self.tok))._reuse_text_only_prompt_ids)


if __name__ == "__main__":
    unittest.main()
