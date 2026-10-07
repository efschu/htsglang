# SPDX-License-Identifier: Apache-2.0
"""FRONT-SEND-IDS-1007 (Hebel 2b): the D leg of a text-only chat turn carries
the front's X-EXACT ids, so group D neither renders nor encodes it.

The front counts every arrival with D's own serving code and tokenizer; D then
renders, encodes, decodes and its tokenizer manager encodes the decoded text
again (desk, NF tokenizer, 142762 tokens: render 23 + encode 223 + decode 37 ms
on top of the 242 ms Hebel 2a removes). The ids the front sends are D's ids
when D renders the same text AND ``encode(decode(ids)) == ids`` on every
``<|im_start|>`` segment -- the bit the segment encoder now keeps per new
segment. The first condition is the metal's to prove: FRONT-IDS / D-IDS shadow
lines (log only) before ``SGLANG_ENABLE_WEG2_FRONT_SEND_IDS`` is turned on.

Pinned here, on the real NF tokenizer and template: the front's ids ARE the
old path's D ids, D takes the posted ids unchanged; a lossy segment, media, a
continued final message, a ``reasoning_effort`` in the template kwargs and a
leg after P's leg 1 keep the text; switch off = the posted body is the
payload object itself.
"""
from __future__ import annotations

import collections
import inspect
import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import front_tokens as FT  # noqa: E402
from sglang.srt.weg2 import prompt_ids_digest as PD  # noqa: E402

CKPT = os.environ.get("WEG2_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "tokenizer.json")) and \
    os.path.isfile(os.path.join(CKPT, "chat_template.jinja"))

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Liest eine Datei .",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
PAYLOAD = {
    "model": "m", "max_tokens": 8, "tools": TOOLS,
    "messages": [
        {"role": "system", "content": "Du bist ein Agent ; antworte kurz ."},
        {"role": "user", "content": "Lies  die Datei:\n\n\tdef f(x) :\n\t\treturn x ** 2  # ok ?"},
        {"role": "assistant", "content": "Gelesen . Sie quadriert x , n'est-ce pas ?"},
        {"role": "user", "content": "Café 日本語 \U0001f642 ... und weiter !"},
    ],
}


class _LossyDecode:
    def __init__(self, tok):
        self._tok = tok

    def decode(self, ids, **kw):
        return self._tok.decode(ids, **kw).replace(" ,", ",")

    def __getattr__(self, name):
        return getattr(self._tok, name)


def _count(rt_ok=True, n=3):
    return FT.Count(n=n, ids=np.arange(n, dtype=np.int32), ms=0.0, reused=0, encoded=n,
                    round_trip_ok=rt_ok)


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class RealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        cls.tok = get_tokenizer(CKPT, tokenizer_mode="auto", trust_remote_code=True)
        cls.ft = FT.FrontTokens()
        with envs.SGLANG_ENABLE_WEG2_FRONT_SEND_IDS.override(True), \
                envs.SGLANG_WEG2_FRONT_TOKENIZER_PREWARM.override(False):
            cls.ft.load({"model_path": CKPT, "tokenizer_path": CKPT}, is_multimodal=True)
        assert cls.ft.state == "ready", cls.ft.why

    def _d_chat(self):
        """Group D's chat serving, Hebel 2a off: the old text path."""
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.parser.template_manager import TemplateManager

        ns = FT._server_args_namespace({"model_path": CKPT, "tokenizer_path": CKPT})
        mc = SimpleNamespace(hf_config=FT._hf_config_stub(CKPT), is_multimodal=True,
                             get_default_sampling_params=lambda: {})
        tm = SimpleNamespace(tokenizer=self.tok, processor=None, server_args=ns,
                             model_config=mc, model_path=CKPT)
        templates = TemplateManager()
        templates.initialize_templates(tm, model_path=CKPT, chat_template=None,
                                       completion_template=None)
        return OpenAIServingChat(tm, templates)

    def _d_ids(self, chat, payload):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        g = chat._convert_to_internal_request(ChatCompletionRequest(**payload))[0]
        return g, (g.input_ids if g.input_ids is not None else self.tok(g.text)["input_ids"])

    def test_front_ids_are_ds_old_path_ids_and_d_takes_them_as_sent(self):
        chat = self._d_chat()
        for turn in range(2):  # the second count is served from the segment cache
            c = self.ft.count("/v1/chat/completions", dict(PAYLOAD))
            self.assertIs(c.round_trip_ok, True, turn)
            self.assertTrue(FT.send_ids_eligible(path="/v1/chat/completions",
                                                 payload=PAYLOAD, count=c))
            _, old_ids = self._d_ids(chat, dict(PAYLOAD))
            self.assertEqual(c.ids.tolist(), old_ids)
        g, ids = self._d_ids(chat, dict(PAYLOAD, input_ids=c.ids.tolist()))
        self.assertIsNone(g.text)
        self.assertEqual(ids, old_ids)

    def test_a_lossy_segment_is_named_and_stays_named_while_cached(self):
        seg = FT.SegmentEncoder(_LossyDecode(self.tok), track_round_trip=True)
        clean = "<|im_start|>user\nkein Komma hier<|im_end|>\n"
        lossy = "<|im_start|>user\nmit , Komma<|im_end|>\n"
        seg.encode(clean)
        self.assertIs(seg.last_round_trip_ok, True)
        seg.encode(clean + lossy)
        self.assertIs(seg.last_round_trip_ok, False)
        seg.encode(clean + lossy)  # both segments from the cache now
        self.assertEqual(seg.last_encoded, 0)
        self.assertIs(seg.last_round_trip_ok, False)
        untracked = FT.SegmentEncoder(self.tok)
        untracked.encode(clean)
        self.assertIsNone(untracked.last_round_trip_ok)


class Eligible(unittest.TestCase):
    def test_only_a_sound_text_only_chat_render_is_sent(self):
        ok = dict(PAYLOAD)
        self.assertTrue(FT.send_ids_eligible(path="/v1/chat/completions", payload=ok, count=_count()))
        image = dict(PAYLOAD, messages=PAYLOAD["messages"] + [{"role": "user", "content": [
            {"type": "text", "text": "?"}, {"type": "image_url", "image_url": {"url": "data:x"}}]}])
        cases = {
            "anthropic path": ("/v1/messages", ok, _count()),
            "/generate": ("/generate", ok, _count()),
            "round trip unknown (switch off)": ("/v1/chat/completions", ok, _count(rt_ok=None)),
            "lossy segment": ("/v1/chat/completions", ok, _count(rt_ok=False)),
            "media": ("/v1/chat/completions", image, _count()),
            "continued final message": ("/v1/chat/completions",
                                        dict(ok, continue_final_message=True), _count()),
            "reasoning_effort in template kwargs": (
                "/v1/chat/completions",
                dict(ok, chat_template_kwargs={"reasoning_effort": "low"}), _count()),
            "ids already given": ("/v1/chat/completions", dict(ok, input_ids=[1]), _count()),
            "empty render": ("/v1/chat/completions", ok, _count(n=0)),
        }
        for name, (path, payload, count) in cases.items():
            with self.subTest(name):
                self.assertFalse(FT.send_ids_eligible(path=path, payload=payload, count=count))


def _front_stub():
    return SimpleNamespace(counters=collections.Counter())


class FrontBody(unittest.TestCase):
    def setUp(self):
        from sglang.srt.weg2.front import Front

        self.Front = Front

    def _note(self, stub, rid="weg2-2-7", on=True):
        with envs.SGLANG_ENABLE_WEG2_FRONT_SEND_IDS.override(on):
            self.Front._send_ids_note(stub, rid, "/v1/chat/completions", dict(PAYLOAD), _count())

    def test_switch_off_posts_the_payload_object_itself(self):
        stub = _front_stub()
        self._note(stub, on=False)
        payload = dict(PAYLOAD)
        self.assertIs(self.Front._send_ids_body(stub, "weg2-2-7", payload, None, False), payload)
        self.assertNotIn("_send_ids", stub.__dict__)

    def test_ids_go_once_and_only_on_a_leg_without_p(self):
        stub = _front_stub()
        self._note(stub)
        body = self.Front._send_ids_body(stub, "weg2-2-7", dict(PAYLOAD), None, False)
        self.assertEqual(body["input_ids"], [0, 1, 2])
        self.assertEqual(body["messages"], PAYLOAD["messages"])
        payload = dict(PAYLOAD)
        self.assertIs(self.Front._send_ids_body(stub, "weg2-2-7", payload, None, False), payload)
        after_p = SimpleNamespace(d_direct=False, skip_leg1=False)
        self._note(stub)
        self.assertIs(self.Front._send_ids_body(stub, "weg2-2-7", payload, after_p, False), payload)
        for pending, single in ((SimpleNamespace(d_direct=True, skip_leg1=False), False),
                                (SimpleNamespace(d_direct=False, skip_leg1=True), False),
                                (after_p, True)):
            self._note(stub)
            self.assertIn("input_ids", self.Front._send_ids_body(stub, "weg2-2-7", dict(PAYLOAD),
                                                                pending, single))

    def test_wiring(self):
        leg2 = inspect.getsource(self.Front.leg2)
        self.assertIn("_d_body = Front._send_ids_body(self, rid, payload, pending, single_prefill)", leg2)
        self.assertIn("json=with_cached_tier_ask(_d_body, request.path)", leg2)
        self.assertIn("Front._send_ids_note(self, rid, path, payload, c)",
                      inspect.getsource(self.Front._x_exact_price))


class Shadow(unittest.TestCase):
    def test_d_ids_line_only_on_group_d_with_the_switch(self):
        self.assertFalse(PD.d_digest_armed(group="D"))
        with envs.SGLANG_LOG_WEG2_PROMPT_IDS_DIGEST.override(True):
            self.assertTrue(PD.d_digest_armed(group="D"))
            self.assertFalse(PD.d_digest_armed(group="P"))
        with self.assertLogs(PD.logger, logging.INFO) as logs:
            PD.log_d_ids(rid="weg2-2-7", ids=[0, 1, 2], src="text")
        self.assertEqual(logs.records[0].getMessage(),
                         f"WEG2 D-IDS rid=weg2-2-7 n=3 sha={FT.ids_digest(np.arange(3))} src=text")

    def test_tokenizer_manager_logs_the_ids_it_hands_on(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        self.assertFalse(TokenizerManager.weg2_prompt_ids_digest)
        src = inspect.getsource(TokenizerManager._tokenize_one_request)
        self.assertIn("if self.weg2_prompt_ids_digest and mm_inputs is None:", src)
        self.assertLess(src.index("log_d_ids("), src.index("self._validate_one_request(obj, input_ids)"))


if __name__ == "__main__":
    unittest.main()
