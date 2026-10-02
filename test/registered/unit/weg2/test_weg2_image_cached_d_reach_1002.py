# SPDX-License-Identifier: Apache-2.0
"""IMAGE-CACHED-1002: an image is "cached for D" only inside what D's OWN
admission covers -- never past the groups' KV page floor of the prompt.

y7n (NF, cad1bf38f9), front log ..._1002_133536.front.log:
  * weg2-0-3 13:38:46: a 128-token image request (image 64 tokens, image_end
    110), fresh -> W102 short -> long, P leg 1, D leg 2 (cached 124).
    STORE-PRESENCE p_flush ``weg2-0-3:128`` (13:39:00), D finish
    ``PRESENCE-DEPTH-CAP credit 128 -> 64`` (13:39:02).
  * weg2-2-6 13:39:05: the SAME request: ``X-EXACT-PRICE ... credit=64
    src=d_leg2_cached ... mm_cached=0/1 image_end=110`` -> W102 -> P, a flip
    pair.

N5j (27B, 582d0fca0d), the same probe: weg2-18-24 13:52:40 ``credit=127 ...
mm_cached=1/1`` -> ``image-cached`` -> SHORT, D served cached_tokens=127 in
0.6 s. The 27B's page is 1 (its handback anchors at N-1); NF's is 64.

What blocks NF is not D's Mamba anchor: D's admission matches at most n-1
tokens aligned DOWN to the page, i.e. page_floor(127) = 64 on NF -- D computes
[64, 128) itself whatever anchor it holds, and [64, 110) are image positions
(no tower on D: vision_d_guard W123). D-direct legs on NF read exactly that
floor (y7n weg2-2-4: prompt 2711, cached 2688 = page_floor(2710)).

The front's MM-XPRICE test ``credit >= image_end`` ignored that floor: any
credit up to n (P's flush anchor ``weg2-0-3:128`` is credited as 128 until
D's finish replaces it) declared the image cached, routed SHORT and released
a queued P-only request -- D then refuses the image positions (W123 / reroute
through P): a wasted admission on top of the P leg. These tests are red on
cad1bf38f9.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import Count, TokenSpans  # noqa: E402

X = 4560
IMG = 248056
IMG_TOKENS = 64
URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."


def _text(n, seed):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


# weg2-0-3's compact render: 46 text tokens, the placeholder, 18 tokens
# (65 compact, 128 expanded, image [46, 110) -- 'compact=65 ... realised=128')
SMALL = np.concatenate([_text(46, 1), np.array([IMG], np.int32), _text(18, 2)])
# the same image with 300 tokens of question after it (n = 410, image_end 110)
LONG = np.concatenate([_text(46, 1), np.array([IMG], np.int32), _text(300, 3)])


def _key():
    from sglang.srt.weg2.front_tokens import mm_image_key

    return mm_image_key({"url": URL, "detail": "auto", "max_dynamic_patch": None})


class _Tok:
    state = "ready"
    why = ""
    image_token_id = IMG

    def __init__(self, by_text):
        self.m = {}
        self.executor = None
        self.by_text = by_text

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        raise AssertionError("an image request is counted with count_mm")

    def count_mm(self, path, payload):
        ids = self.by_text[payload["t"]]
        return (Count(n=int(ids.size), ids=ids, ms=1.0, reused=0, encoded=int(ids.size)),
                [_key()])


def _front(page, awake="D"):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 0
    f.awake = awake
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok({"s": SMALL, "s-repeat": SMALL, "l": LONG, "l-repeat": LONG})
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = None
    f._store_probe_t = time.monotonic()
    f._store_probe_info = {"page_size": page}
    return f


def _price(f, rid, t):
    return asyncio.run(f._x_exact_price(rid, "/v1/chat/completions", {"t": t}, t, 23, 23,
                                        multimodal=True))


def _serve_on_p(f, rid, t, n):
    """weg2-0-3: fresh -> fallback; P serves it (n tokens) and its sleep flush
    publishes the END-ANCHOR page_floor(n) (STORE-PRESENCE src=p_flush)."""
    assert _price(f, rid, t) is None
    f._p_leg1_store_note(rid, t, n)
    assert f._p_flush_store_presence() == 1


# ---- the pure reach ---------------------------------------------------------------------

def test_d_reach_is_the_credit_under_the_admission_page_floor():
    from sglang.srt.weg2.front_tokens import mm_d_cached, mm_d_reach

    # NF y7n weg2-2-6: whatever the witness says, D computes [64, 128)
    assert mm_d_reach(64, 128, 64) == (64, 64)
    assert mm_d_reach(124, 128, 64) == (64, 64)
    assert mm_d_reach(128, 128, 64) == (64, 64)
    assert mm_d_cached([110], 128, 128, 64) == (0, 64, 64)
    # 27B N5j weg2-18-24: page 1, D's prefix 127 covers image_end 110
    assert mm_d_reach(127, 128, 1) == (127, 127)
    assert mm_d_cached([110], 127, 128, 1) == (1, 127, 127)
    # a long prompt: the floor does not bind, the credit does
    assert mm_d_cached([110], 384, 410, 64) == (1, 384, 384)
    assert mm_d_cached([110, 300], 256, 410, 64) == (1, 256, 384), "second image past the credit"


# ---- the arrival price / route ----------------------------------------------------------

def test_y7n_store_anchor_credit_never_declares_an_image_in_the_floor_page_cached(caplog):
    """The p_flush anchor credits weg2-0-3's ids with 128 (> image_end 110) before
    D's finish caps it -- an identical request arriving then was priced
    mm_cached and routed SHORT on cad1bf38f9."""
    caplog.set_level(logging.INFO)
    f = _front(page=64)
    _serve_on_p(f, "weg2-0-3", "s", 128)
    xx = _price(f, "weg2-2-6", "s-repeat")
    assert xx is not None and xx.n == 128 and xx.image_end == 110
    assert xx.credit == 128, "the store witness itself is unchanged"
    assert (xx.d_reach, xx.d_floor) == (64, 64)
    assert not xx.mm_cached, "D would compute image positions 64..109"
    assert any(m.startswith("WEG2 X-EXACT-PRICE rid=weg2-2-6 ")
               and m.endswith("d_reach=64 mm=1 mm_cached=0/1 image_end=110")
               for m in caplog.messages)


def test_27b_page_one_keeps_serving_the_repeat_from_d():
    f = _front(page=1)
    _serve_on_p(f, "weg2-16-23", "s", 128)
    xx = _price(f, "weg2-18-24", "s-repeat")
    assert xx.mm_cached and xx.d_reach == 127 and xx.d_floor == 127


def test_nf_long_prompt_with_the_image_below_the_floor_routes_as_text():
    """The image in a long prompt's history: D's floor (page_floor(409) = 384)
    is past image_end 110 -- cached, no flip, as on the 27B."""
    f = _front(page=64)
    _serve_on_p(f, "weg2-0-3", "l", 410)
    xx = _price(f, "weg2-2-6", "l-repeat")
    assert xx.credit == 384 and (xx.d_reach, xx.d_floor) == (384, 384)
    assert xx.mm_cached and xx.pending == 410 - 384


def test_the_uncached_verdict_names_why(caplog):
    caplog.set_level(logging.INFO)
    f = _front(page=64)
    _serve_on_p(f, "weg2-0-3", "s", 128)
    xx = _price(f, "weg2-2-6", "s-repeat")
    assert f._w102_image_uncached("weg2-2-6", xx) == "floor"
    assert any(m.startswith("W102 Weg2VisionStage rid=weg2-2-6 image-uncached why=floor "
                            "image_end=110 d_reach=64 (credit=128 src=") and "floor=64 of n=128 "
               "at page=64" in m for m in caplog.messages)
    f2 = _front(page=64)
    assert _price(f2, "weg2-0-1", "l") is None
    f2._mm_ktok()[_key()] = IMG_TOKENS
    xx2 = _price(f2, "weg2-0-2", "l-repeat")
    assert xx2.credit == 0 and f2._w102_image_uncached("weg2-0-2", xx2) == "credit"
    assert f2.counters["w102_image_uncached_credit"] == 1


def test_wiring_the_route_forces_p_exactly_when_not_cached():
    src = inspect.getsource(F)
    assert "self._w102_image_uncached(rid, _xx)" in src
    assert ("if _verdict == VERDICT_STAGE and not _mm_cached and route != \"none\" "
            "and route != \"long\":") in src
    assert "p_only=_verdict == VERDICT_STAGE and not _mm_cached" in src


# ---- the reprice at STORE-PRESENCE ------------------------------------------------------

def _queued(f, rid, text):
    p = F.Pending(rid, "/v1/chat/completions", {}, text, time.time(),
                  asyncio.new_event_loop().create_future(), est_prompt=23,
                  est_uncached=23, p_only=True)
    f.queue.append(p)
    return p


def test_reprice_keeps_a_queued_image_request_p_only_inside_the_floor_page(caplog):
    """cad1bf38f9 released it at the flush (credit 128 >= 110) -> a D admission
    that must compute the image."""
    caplog.set_level(logging.INFO)
    f = _front(page=64, awake="P")
    assert _price(f, "weg2-0-3", "s") is None
    assert _price(f, "weg2-2-6", "s-repeat") is None, "queued before the image was learned"
    p = _queued(f, "weg2-2-6", "s-repeat")
    f._p_leg1_store_note("weg2-0-3", "s", 128)
    f._p_flush_store_presence()
    assert p.mm_image_end == 110
    assert p.p_only, "D's admission floor (64) lies inside the image"
    assert not any("image-cached at reprice" in m for m in caplog.messages)


def test_reprice_still_releases_on_page_one():
    f = _front(page=1, awake="P")
    assert _price(f, "weg2-16-23", "s") is None
    assert _price(f, "weg2-18-24", "s-repeat") is None
    p = _queued(f, "weg2-18-24", "s-repeat")
    f._p_leg1_store_note("weg2-16-23", "s", 128)
    f._p_flush_store_presence()
    assert not p.p_only


def test_unknown_page_keeps_the_pre_1002_price():
    f = _front(page=1)
    del f._store_probe_info
    assert f._d_page_size() == 1


# ---- D's extend never touches an image inside its prefix ---------------------------------

def test_d_extend_never_embeds_an_image_inside_its_prefix():
    """The image-cached SHORT on D: D (no tower) extends [prefix, n) with the
    image wholly below prefix. The chunked-prefill embedding skips such an
    item -- the tower function is never called -- at exactly the boundary
    vision_d_guard admits (image_end <= covered; offsets are inclusive), and
    the guard refuses one page lower (NF floor 64 < image_end 110)."""
    import torch

    from sglang.srt.managers import mm_utils
    from sglang.srt.weg2 import vision_d_guard as G

    def _no_tower(items):
        raise AssertionError("D embedded an image position it holds in its prefix")

    item = type("Item", (), {"offsets": [(46, 109)]})()
    ids = torch.zeros(18, dtype=torch.int64)
    for prefix in (110, 384):
        emb, out = mm_utils._get_chunked_prefill_embedding(
            _no_tower, [item], [0, 1], [prefix], [18], [[(46, 109)]], ids)
        assert emb is None and out is ids
    req = type("Req", (), {"multimodal_inputs": type("MM", (), {"mm_items": [item]})()})()
    assert G.image_end(req) == 110
    assert G.verdict(req, covered=110) == G.ADMIT and G.verdict(req, covered=64) == G.REFUSE
