# SPDX-License-Identifier: Apache-2.0
"""GROUP-SEG-1022 review (#157): the danger directions the author's tests left open.

* PROPERTY on the production tokenizer (PDFLIP_TEST_NF_CKPT): random texts built from the characters that
  break BPE / pre-tokenisation / NFC at a cut (whitespace runs, CRLF, combining marks, emoji, CJK, contractions,
  every special token incl. the marker inside user text) -- segmented encode == whole encode, the tokenizer
  manager's call == tok([t]), and the segmented decode of random id runs == whole decode, all on ONE warm
  instance (cache hits included). Before: only in /tmp/tok1022/prop.py, not in the repo.
* STRUCTURAL encode check: a tokenizer with an added token that overlaps the marker ("Q<|im") passed the
  sample load checks and returned other ids than the whole encode; it must not arm.
* DECODE KEY int64: id x + 2**32 hashed as int32 hit x's cached text where the whole decode raises.
* ENTRY BOUND: many short distinct pieces must not grow either cache past ``cap_entries``.
* FALLBACK: a failing segmented call returns the whole call's result.
"""
from __future__ import annotations

import os
import random
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import group_seg_tokens as G  # noqa: E402

CKPT = os.environ.get("PDFLIP_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "tokenizer.json"))

ALPHABET = list("abc xyz\t\n  .,;:!?'\"()[]{}<>|/\\-_=+*&^%$#@~`") + [
    "ä", "ß", "日本", "語", "\U0001f642", "\U0001f44d\U0001f3fd", "́", "é", "\r\n", "\r",
    "<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "</think>", "<think>",
    "\n\n\n", " n't", " 's", " .", " ,", "<|im", "_start|>", "<|", "|>", "   ", " ", "​", "﻿"]


def _load(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


@unittest.skipUnless(HAVE_CKPT, f"no tokenizer at {CKPT}")
class Review(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _load(CKPT)

    def test_property_encode_call_decode_on_one_warm_instance(self):
        g = G.GroupSegTokens(self.tok)
        self.assertTrue(g.enc_ok and g.dec_ok and g.call_ok, (g.enc_why, g.dec_why, g.call_why))
        m = g.marker_id
        rnd = random.Random(1022)
        texts = []
        for trial in range(2000):
            t = "".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(1, 60)))
            if trial % 3 == 0 and texts:  # a later turn: an earlier text plus new pieces (cache hits)
                t = rnd.choice(texts) + "<|im_start|>" + t
            texts.append(t)
            whole = list(self.tok.encode(t))
            self.assertEqual(g.encode(t), whole, repr(t))
            self.assertEqual(g.encode_call(t), list(self.tok([t])["input_ids"][0]), repr(t))
            seq = [rnd.randrange(0, 248000) if rnd.random() < 0.3 else rnd.choice(whole or [m])
                   for _ in range(rnd.randint(1, 30))]
            for _ in range(rnd.randint(0, 3)):
                seq.insert(rnd.randint(0, len(seq)), m)
            self.assertEqual(g.decode(seq), self.tok.decode(seq), seq)
            self.assertEqual(g.decode(whole), self.tok.decode(whole), repr(t))
        self.assertGreater(g.seg.last_reused + sum(1 for _ in g.seg.cache), 0)
        self.assertEqual(g.fallbacks, 0)

    def test_overlapping_added_token_does_not_arm(self):
        from tokenizers import AddedToken

        tok = _load(CKPT)
        tok.add_special_tokens({"additional_special_tokens": [AddedToken("Q<|im", special=True, normalized=False)]},
                               replace_extra_special_tokens=False)
        t = "hello Q<|im_start|>user\nhi<|im_end|>\n"
        g = G.GroupSegTokens(tok)
        self.assertFalse(g.enc_ok, g.enc_why)
        self.assertEqual(g.encode(t), list(tok.encode(t)))
        self.assertEqual(g.encode_call(t), list(tok([t])["input_ids"][0]))

    def test_decode_key_does_not_wrap_at_int32(self):
        g = G.GroupSegTokens(self.tok)
        m = g.marker_id
        a = [m, 100, 200]
        b = [m, 100, 200 + 2 ** 32]
        g.decode(a)  # cached
        with self.assertRaises(OverflowError):
            self.tok.decode(b)
        with self.assertRaises(OverflowError):
            g.decode(b)

    def test_entry_bound(self):
        g = G.GroupSegTokens(self.tok, cap_entries=100)
        t = "".join("<|im_start|>u%d\n" % i for i in range(1000))
        self.assertEqual(g.encode(t), list(self.tok.encode(t)))
        ids = list(self.tok.encode(t))
        self.assertEqual(g.decode(ids), self.tok.decode(ids))
        self.assertLessEqual(len(g.seg.cache), 100)
        self.assertLessEqual(len(g.dec_cache), 100)
        self.assertEqual(g.seg.cached_tokens, sum(len(v) for v in g.seg.cache.values()))
        self.assertEqual(g.dec_tokens, sum(n for _, n in g.dec_cache.values()))

    def test_failing_segmented_call_returns_the_whole_call(self):
        g = G.GroupSegTokens(self.tok)
        t = "<|im_start|>user\nHallo Welt<|im_end|>\n<|im_start|>assistant\n"
        ids = list(self.tok.encode(t))

        def boom(*a, **k):
            raise RuntimeError("Already borrowed")

        g.seg.encode = boom
        g._decode_pieces = boom
        self.assertEqual(g.encode(t), ids)
        self.assertEqual(g.encode_call(t), ids)
        self.assertEqual(g.decode(ids), self.tok.decode(ids))
        self.assertEqual(g.fallbacks, 3)


if __name__ == "__main__":
    unittest.main()
