# SPDX-License-Identifier: Apache-2.0
"""GROUP-SEG-1022 (#157): a group's tokenizer process encodes and decodes a
media prompt per ``<|im_start|>`` piece, each piece cached by its content.

Pinned on the real NF tokenizer and the fork's own serving code (serving_chat
built as a group's tokenizer manager holds it: the PLAIN tokenizer, no front
wrapper), on an Anthropic /v1/messages agent conversation with an image:

* EXACT: prompt_ids, the decoded prompt and the tokenizer manager's ids equal
  the whole-call results bit for bit, turn after turn.
* DANGER "falsche IDs bei Praefix-Abweichung": an earlier message changed
  (late in the message, the head unchanged) after the cache saw the old
  conversation -- the result is the whole call's, the piece is encoded anew.
* DANGER "P/D bekommen unterschiedliche IDs": two groups (two tokenizer
  objects, two caches) with different histories -- D warm over every turn,
  P cold -- produce the same ids for the same prompt, and they are the whole
  call's.
* COST: a follow-up turn encodes/decodes only its new pieces (deterministic
  counts), and the timings at 24k / 100k / 208k tokens (printed; the report's
  table) -- segmented clearly below the whole call from 100k on.
"""
from __future__ import annotations

import asyncio
import base64
import glob
import io
import os
import random
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from flliper.srt.pdflip import front_tokens as FT  # noqa: E402
from flliper.srt.pdflip import group_seg_tokens as G  # noqa: E402

CKPT = os.environ.get("PDFLIP_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "tokenizer.json")) and \
    os.path.isfile(os.path.join(CKPT, "chat_template.jinja"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = sorted(glob.glob(os.path.join(_HERE, "..", "..", "..", "..", "python", "flliper", "srt", "pdflip", "*.py")))
CORPUS = "".join(open(f, encoding="utf-8").read() for f in _SRC[:40]) or ("def f(x):\n    return x ** 2\n" * 4000)
PROSE = ("Ich habe die Datei gelesen -- der Pfad läuft über den Scheduler → dann antworte ich kurz. "
         "Startmenues, n'est-ce pas ? Café 日本語 \U0001f642 e\u0301 \t\r\n") * 8


def _png_b64() -> str:
    try:
        from PIL import Image

        img = Image.fromarray((np.random.RandomState(0).rand(64, 80, 3) * 255).astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 -- the template only writes the placeholder
        return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 64).decode()


IMG = _png_b64()


def _turns(n_turns: int, chunk_chars: int = 2000, edit=None):
    """An agent conversation: tool results (code), assistant prose, an image in turn 1."""
    msgs = []
    pos = 0
    for i in range(n_turns):
        chunk = CORPUS[pos:pos + chunk_chars]
        pos = (pos + chunk_chars) % max(1, len(CORPUS) - chunk_chars)
        text = "Tool result %d:\n%s" % (i, chunk)
        if edit is not None and edit[0] == i:
            text = text[:-10] + edit[1] + text[-10:]
        content = [{"type": "text", "text": text}]
        if i == 1:
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": IMG}},
                       {"type": "text", "text": "Screenshot dazu.\n" + text}]
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": PROSE + str(i)}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": "Weiter, Turn %d." % n_turns}]})
    return {"model": "m", "max_tokens": 64, "system": "Du bist ein Agent ; antworte kurz . " * 20,
            "messages": msgs}


def _group_stack(path):
    """serving_chat + anthropic as a GROUP's tokenizer manager holds them (the plain tokenizer)."""
    from flliper.srt.entrypoints.anthropic.serving import AnthropicServing
    from flliper.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from flliper.srt.parser.template_manager import TemplateManager
    from flliper.srt.utils.hf_transformers.tokenizer import get_tokenizer

    ns = FT._server_args_namespace({"model_path": path, "tokenizer_path": path})
    tok = get_tokenizer(path, tokenizer_mode=ns.tokenizer_mode, trust_remote_code=False)
    mc = SimpleNamespace(hf_config=FT._hf_config_stub(path), is_multimodal=True,
                         get_default_sampling_params=lambda: {})
    tm = SimpleNamespace(tokenizer=tok, processor=None, server_args=ns, model_config=mc, model_path=path)
    templates = TemplateManager()
    templates.initialize_templates(tm, model_path=path, chat_template=None)
    chat = OpenAIServingChat(tm, templates)
    return tok, chat, AnthropicServing(chat)


def _render(chat, anth, payload):
    from flliper.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

    req = anth._convert_to_chat_completion_request(AnthropicMessagesRequest(**payload))
    r = chat._process_messages(req, True)
    return list(r.prompt_ids), r.prompt, r.image_data


class _Plain:
    """The whole-call baseline: GROUP-SEG off for the duration (for_tokenizer -> None)."""

    def __enter__(self):
        self._orig = G.for_tokenizer
        G.for_tokenizer = lambda tok: None
        return self

    def __exit__(self, *a):
        G.for_tokenizer = self._orig


def _tm_tokenize(tok, text):
    """TokenizerManager._tokenize_texts on a stub manager (single string, no async batcher)."""
    from flliper.srt.managers.tokenizer_manager import TokenizerManager

    tm = TokenizerManager.__new__(TokenizerManager)
    tm.tokenizer = tok
    tm.async_dynamic_batch_tokenizer = None
    ids, _ = asyncio.run(TokenizerManager._tokenize_texts(tm, text))
    return list(ids)


class NotArmed(unittest.TestCase):
    """A tokenizer the load checks refuse: every call is the whole call."""

    class _Tok:
        all_special_tokens = []
        added_tokens_decoder = {}

        def encode(self, text, **kw):
            return [1] + [ord(c) for c in text]  # adds a special to every encode

        def decode(self, ids, **kw):
            return "".join(chr(i) for i in ids if i != 1)

        def __call__(self, texts, **kw):
            return {"input_ids": [self.encode(t) for t in texts]}

    def test_refused_tokenizer_runs_the_whole_call(self):
        tok = self._Tok()
        g = G.for_tokenizer(tok)
        self.assertIsNotNone(g)
        self.assertFalse(g.enc_ok)
        self.assertFalse(g.dec_ok)
        self.assertEqual(G.seg_encode(tok, "ab<|im_start|>c"), tok.encode("ab<|im_start|>c"))
        self.assertEqual(G.seg_decode(tok, [1, 97, 98]), "ab")
        self.assertEqual(G.seg_encode_call(tok, "xy"), [1, 120, 121])

    def test_front_wrapper_keeps_its_own_encode(self):
        calls = []

        class _Seg:
            def encode(self, text, **kw):
                calls.append(text)
                return [7]

        w = FT._TokWrapper(self._Tok(), _Seg())
        self.assertEqual(G.seg_encode(w, "abc"), [7])
        self.assertEqual(calls, ["abc"])


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class RealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, cls.chat, cls.anth = _group_stack(CKPT)

    def _g(self):
        return G.for_tokenizer(self.tok)

    def test_armed_on_the_nf_tokenizer(self):
        g = self._g()
        self.assertTrue(g.enc_ok, g.enc_why)
        self.assertTrue(g.dec_ok, g.dec_why)
        self.assertTrue(g.call_ok, g.call_why)

    def test_media_turns_equal_the_whole_call_and_reuse_pieces(self):
        g = self._g()
        prev_n = None
        for n in (2, 3, 4, 5):
            payload = _turns(n)
            with _Plain():
                ids0, text0, img0 = _render(self.chat, self.anth, payload)
                tm0 = _tm_tokenize(self.tok, text0)
            ids1, text1, img1 = _render(self.chat, self.anth, payload)
            enc_reused, dec_reused = g.seg.last_reused, g.last_dec_reused
            g.seg.last_reused = -1  # what the tokenizer manager's own encode reuses, below
            tm1 = _tm_tokenize(self.tok, text1)
            self.assertEqual(ids1, ids0)
            self.assertEqual(text1, text0)
            self.assertEqual(tm1, tm0)
            self.assertEqual(tm1, list(self.tok([text0])["input_ids"][0]))
            self.assertEqual(len(img1 or ()), 1)
            self.assertIn("<|image_pad|>", text1)
            if prev_n is not None:
                # the follow-up turn: every piece of the previous turn's prompt but its last user message
                # and the generation prompt (< 300 tokens) came from the caches, through serving_chat
                # (encode + decode) and the tokenizer manager (encode of the decoded text)
                self.assertGreaterEqual(enc_reused, prev_n - 300)
                self.assertGreaterEqual(dec_reused, prev_n - 300)
                self.assertGreaterEqual(g.seg.last_reused, prev_n - 300)
            prev_n = len(ids1)

    def test_prefix_deviation_is_encoded_anew(self):
        """DANGER falsche IDs bei Praefix-Abweichung: an earlier message changed late (its head unchanged)."""
        g = self._g()
        _render(self.chat, self.anth, _turns(5))  # the cache holds the old conversation
        for edit in ((0, " GEAENDERT "), (1, "\U0001f642"), (3, "<|im_end|>"), (2, "\t \n")):
            payload = _turns(5, edit=edit)
            with _Plain():
                ids0, text0, _ = _render(self.chat, self.anth, payload)
                tm0 = _tm_tokenize(self.tok, text0)
            ids1, text1, _ = _render(self.chat, self.anth, payload)
            self.assertGreater(g.seg.last_encoded, 0, "the changed piece is encoded, not taken from the cache")
            tm1 = _tm_tokenize(self.tok, text1)
            self.assertEqual(ids1, ids0, edit)
            self.assertEqual(text1, text0, edit)
            self.assertEqual(tm1, tm0, edit)

    def test_p_and_d_get_the_same_ids(self):
        """DANGER P/D unterschiedliche IDs: two groups, two caches, different histories."""
        tok_p, chat_p, anth_p = _group_stack(CKPT)
        self.assertIsNot(tok_p, self.tok)
        for n in (2, 3, 4, 5):  # D saw every turn ...
            _render(self.chat, self.anth, _turns(n, edit=(1, " d-turn %d " % n)))
        payload = _turns(6, edit=(1, " d-turn 5 "))
        ids_d, text_d, _ = _render(self.chat, self.anth, payload)
        tm_d = _tm_tokenize(self.tok, text_d)
        ids_p, text_p, _ = _render(chat_p, anth_p, payload)  # ... P only this one
        tm_p = _tm_tokenize(tok_p, text_p)
        with _Plain():
            ids0, text0, _ = _render(self.chat, self.anth, payload)
        self.assertEqual(ids_d, ids_p)
        self.assertEqual(ids_d, ids0)
        self.assertEqual(text_d, text_p)
        self.assertEqual(text_d, text0)
        self.assertEqual(tm_d, tm_p)
        self.assertEqual(tm_d, list(self.tok([text0])["input_ids"][0]))

    def test_decode_of_arbitrary_id_runs(self):
        """Cut pieces of id runs no encode produces (a multi-byte character split before the marker,
        marker runs, random vocabulary ids): the segmented decode equals the whole decode."""
        g = self._g()
        m = g.marker_id
        rnd = random.Random(7)
        pool = self.tok.encode("Café 日本語 \U0001f642\U0001f44d\U0001f3fd e\u0301 \t\r\n<|im_end|> n't .")
        for _ in range(400):
            seq = [rnd.randrange(0, 150000) if rnd.random() < 0.3 else rnd.choice(pool)
                   for _ in range(rnd.randint(1, 40))]
            for _ in range(rnd.randint(0, 4)):
                seq.insert(rnd.randint(0, len(seq)), m)
            self.assertEqual(g.decode(seq), self.tok.decode(seq), seq)
            self.assertEqual(g.decode(seq), self.tok.decode(seq), seq)  # the cached pieces too

    def test_runtime_24k_100k_208k(self):
        """Measured, before (whole call) and after (segmented, the previous turn seen): serving_chat's
        render+encode+decode plus the tokenizer manager's encode -- the D-MATCH-PROBE's text work."""
        rows = []
        n10 = len(_render(self.chat, self.anth, _turns(10))[0])
        per_turn = n10 - len(_render(self.chat, self.anth, _turns(9))[0])
        for target in (24000, 100000, 208000):
            n = max(3, int(round((target - n10) / per_turn)) + 10)
            prev, cur = _turns(n - 1), _turns(n)
            tok_g, chat_g, anth_g = _group_stack(CKPT)  # a fresh group: its own cache
            g = G.for_tokenizer(tok_g)
            _render(chat_g, anth_g, prev)  # the session's previous turn
            _tm_tokenize(tok_g, _render(chat_g, anth_g, prev)[1])
            best_plain = best_seg = 1e18
            for _ in range(3):
                with _Plain():
                    t0 = time.perf_counter()
                    ids0, text0, _ = _render(chat_g, anth_g, cur)
                    tm0 = _tm_tokenize(tok_g, text0)
                    best_plain = min(best_plain, time.perf_counter() - t0)
            # cold first: what the FIRST turn of a session costs with the caches (all pieces new)
            tok_c, chat_c, anth_c = _group_stack(CKPT)
            t0 = time.perf_counter()
            idsc, textc, _ = _render(chat_c, anth_c, cur)
            tmc = _tm_tokenize(tok_c, textc)
            cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            ids1, text1, _ = _render(chat_g, anth_g, cur)
            enc_new, dec_new = g.seg.last_encoded, g.last_dec_decoded
            tm1 = _tm_tokenize(tok_g, text1)
            best_seg = time.perf_counter() - t0
            self.assertEqual(ids1, ids0)
            self.assertEqual(tm1, tm0)
            self.assertEqual(idsc, ids0)
            self.assertEqual(tmc, tm0)
            new = len(ids1) - len(_render(chat_g, anth_g, prev)[0])
            # only the new pieces are encoded/decoded (the last assistant + user piece of the previous
            # turn's render differ by the generation prompt: bounded by two turns)
            self.assertLessEqual(enc_new, 2 * per_turn + new)
            self.assertLessEqual(dec_new, 2 * per_turn + new)
            if target >= 100000:
                self.assertLess(best_seg, 0.5 * best_plain)
            rows.append((len(ids1), best_plain * 1e3, cold * 1e3, best_seg * 1e3, enc_new, dec_new))
        for r in rows:
            print("GROUP-SEG-RUNTIME n=%d whole_ms=%.1f cold_seg_ms=%.1f warm_seg_ms=%.1f "
                  "encoded_now=%d decoded_now=%d" % r)


if __name__ == "__main__":
    unittest.main()
