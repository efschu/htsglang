# SPDX-License-Identifier: Apache-2.0
"""MM-SUM-1007: an agent that sends the SAME SEVERAL images every turn is priced
exactly from the second turn on -- an image already inside D's cached prefix
forces no flip.

NF boot dkrnfint4h6abl..int11 (07.10.), front log ..._1007_084418.front.log: one OMP
session, five screenshots in every request. 83 times ``W102 ... reason=vision_p_only``
-- a flip D->P (and back) for 61..1286 new tokens each turn -- and 85 times
``X-EXACT-FALLBACK reason=multimodal-unseen-image ... images=5 known=0``, never one
``X-EXACT-MM-LEARN``. ``mm_learn`` reads ONE image's token count from the prompt a
group realised and only when exactly one of the request's images is unknown: five
unseen images at once are one equation with five unknowns, so the front never learned
any of them and the cached-image exception (``mm_cached``: the image ends inside D's
cached prefix, no tower, no flip) never applied.

The fix keeps the HASH of the image set (the ordered tuple of ``mm_image_key``) and the
SUM of its token counts as P realised it. The sum is all the front needs: the prompt
length and the end of the LAST image do not depend on how the sum splits.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import (  # noqa: E402
    Count,
    TokenSpans,
    mm_expand,
    mm_expand_sum,
    mm_image_key,
    mm_learn,
    mm_sum_learn,
)

X = 4113
IMG = 248056
#: what P really expanded each of the five screenshots to (unknown to the front)
REAL_K = [1000, 1204, 868, 1000, 1000]
TOTAL = sum(REAL_K)
URLS = ["data:image/png;base64,%02d" % i for i in range(5)]
KEYS = [mm_image_key({"url": u, "detail": "auto", "max_dynamic_patch": None}) for u in URLS]


def _text(n, seed):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


def _compact(*segments):
    """text segments with one placeholder between every two -- the compact render"""
    parts = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(np.array([IMG], np.int32))
    return np.concatenate(parts)


# turn 1: system + five screenshots interleaved with text, then the question
SEGS = [_text(9000, 11), _text(8000, 12), _text(7000, 13), _text(6000, 14), _text(5000, 15),
        _text(30000, 16)]
COMPACT_1 = _compact(*SEGS)                       # 5 placeholders
REAL_1 = COMPACT_1.size - 5 + TOTAL               # the prompt P realises
# turn 2: the whole first prompt, then 700 new text tokens (the same five images)
COMPACT_2 = np.concatenate([COMPACT_1, _text(700, 17)])
REAL_2 = COMPACT_2.size - 5 + TOTAL


class _Tok:
    state = "ready"
    why = ""
    image_token_id = IMG

    def __init__(self):
        self.m = {}
        self.executor = None
        self.by_text = {}
        self.keys = {}

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        raise AssertionError("an image request is counted with count_mm")

    def count_mm(self, path, payload):
        t = payload["t"]
        ids = self.by_text[t]
        return (Count(n=int(ids.size), ids=ids, ms=1.0, reused=0, encoded=int(ids.size)),
                list(self.keys.get(t, KEYS)))


def _front(awake="D"):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = awake
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok()
    f.ftok.by_text = {"t1": COMPACT_1, "t2": COMPACT_2}
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = None
    f._store_probe_t = time.monotonic()
    return f


def _price(f, rid, t):
    est = int(f.ftok.by_text[t].size)
    return asyncio.run(f._x_exact_price(rid, "/v1/chat/completions", {"t": t}, t, est, est,
                                        multimodal=True))


def _serve_first_turn(f):
    """turn 1: five unseen images -> fallback, P serves the prompt and flushes"""
    assert _price(f, "pdflip-0-1", "t1") is None
    f._p_leg1_store_note("pdflip-0-1", "t1", REAL_1)
    assert f._p_flush_store_presence() == 1


# ---- the helpers -----------------------------------------------------------------------

def test_one_by_one_learning_cannot_split_five_unseen_images():
    """the premise of the bug: nothing of the five is learnable alone"""
    assert mm_learn(COMPACT_1.size, KEYS, {}, REAL_1) is None


def test_the_sum_is_the_prompt_minus_the_text_the_front_counted():
    assert mm_sum_learn(COMPACT_1.size, KEYS, REAL_1) == TOTAL
    assert mm_sum_learn(COMPACT_1.size, KEYS, 0) is None
    assert mm_sum_learn(COMPACT_1.size, KEYS, COMPACT_1.size - 1) is None, \
        "fewer image tokens than images is no count"
    assert mm_sum_learn(COMPACT_1.size, [], REAL_1) is None


def test_the_sum_expansion_gives_the_exact_length_and_the_exact_end_of_the_last_image():
    exact = mm_expand(COMPACT_1, IMG, KEYS, dict(zip(KEYS, REAL_K)))
    e = mm_expand_sum(COMPACT_1, IMG, KEYS, TOTAL)
    assert e.ids.size == exact.ids.size == REAL_1
    assert e.image_end == exact.image_end, "the end of the LAST image needs no split"
    assert e.first_image == exact.first_image == 9000
    assert np.array_equal(e.ids[e.image_end:], exact.ids[exact.image_end:]), \
        "every position after the last image is the real position"
    assert mm_expand_sum(COMPACT_1, IMG, KEYS, None) is None, "an unseen set has no expansion"
    assert mm_expand_sum(COMPACT_1, IMG, KEYS[:4], TOTAL) is None, "placeholders must match"
    assert mm_expand_sum(COMPACT_1, IMG, KEYS, 3) is None, "fewer tokens than images"


def test_the_same_set_expands_the_same_way_every_turn():
    """the equal split is deterministic: turn 2 matches turn 1 token for token"""
    e1 = mm_expand_sum(COMPACT_1, IMG, KEYS, TOTAL)
    e2 = mm_expand_sum(COMPACT_2, IMG, KEYS, TOTAL)
    assert np.array_equal(e2.ids[: e1.ids.size], e1.ids)


# ---- the front: the OMP session ------------------------------------------------------

def test_the_second_turn_with_the_same_five_images_is_priced_short_and_cached(caplog):
    """Bug regression: before the fix this returned None (chars/3 fallback, ``known=0``)
    and the vision rule flipped D->P for 700 new tokens."""
    caplog.set_level(logging.INFO)
    f = _front()
    _serve_first_turn(f)
    assert any(m.startswith("PDFLIP X-EXACT-MM-SUM rid=pdflip-0-1 images=5 total_image_tokens=%d" % TOTAL)
               for m in caplog.messages)
    xx = _price(f, "pdflip-2-3", "t2")
    assert xx is not None, "no chars/3 fallback for an image set the front has seen served"
    assert xx.n == REAL_2
    assert xx.mm and xx.mm_cached, "the images end inside what D already holds"
    assert xx.pending <= X, "SHORT: no flip pair for a few hundred new tokens"
    assert xx.credit >= xx.image_end
    assert f.counters["x_exact_mm_sum_priced"] == 1
    assert not any("X-EXACT-FALLBACK rid=pdflip-2-3" in m for m in caplog.messages)


def test_a_set_with_one_new_image_still_runs_the_tower_on_p(caplog):
    """a NEW screenshot is no hit of the old set: the tower must run for it"""
    caplog.set_level(logging.INFO)
    f = _front()
    _serve_first_turn(f)
    new_key = mm_image_key({"url": "data:image/png;base64,NEW", "detail": "auto",
                            "max_dynamic_patch": None})
    f.ftok.keys["t2"] = KEYS[:4] + [new_key]
    assert _price(f, "pdflip-2-4", "t2") is None
    assert any("X-EXACT-FALLBACK rid=pdflip-2-4 reason=multimodal-unseen-image" in m
               and "images=5 known=0" in m for m in caplog.messages)


def test_the_order_of_the_images_is_part_of_the_set():
    """the front's ids have one position per image run: a reordered set is another set"""
    f = _front()
    _serve_first_turn(f)
    f.ftok.keys["t2"] = list(reversed(KEYS))
    assert _price(f, "pdflip-2-5", "t2") is None


def test_an_unseen_set_keeps_the_fallback_and_says_why(caplog):
    caplog.set_level(logging.INFO)
    f = _front()
    assert _price(f, "pdflip-0-1", "t1") is None
    assert any("X-EXACT-FALLBACK rid=pdflip-0-1 reason=multimodal-unseen-image" in m
               and "images=5 known=0" in m for m in caplog.messages)
    assert f.counters["x_exact_mm_sum_priced"] == 0

