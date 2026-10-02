# SPDX-License-Identifier: Apache-2.0
"""MM-XPRICE (02.10.): an image request is priced exactly, like text, and an
image the shared store already holds forces neither a flip nor a tower.

y7h-noH4 (23c8fb584e), front log ..._1002_105106.front.log, P log beside it:
  * weg2-0-1 (10:54:21): the image needle, fresh -- P leg 1 prompt 73112,
    cached 3264 (the tower ran, correctly).
  * weg2-1-5 (10:54:57): the SAME request again. ``X-EXACT-FALLBACK
    reason=multimodal`` -> chars/3 76654, ``presence_src=none``, LONG,
    ``vision_flip_urgent=True`` -> flip epoch 3 D->P (3.7 s), P leg 1 prompt
    73112 cached 73088 = 24 new tokens, flip epoch 4 P->D (3.0 s). On P the
    transient tower still loaded and tore down (``W102 Weg2VisionStage run=2
    ... load 478, encode 20, teardown 413``) for an image inside the prefix.
  * weg2-4-18 (10:57:31): a follow-up turn whose image sat in the
    conversation: ``W102 ... route short -> long (P)`` with est_uncached 1658
    < X=4113, flip epoch 5, P computed 1478 new (cached 73088/74566).
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import time
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import Count, TokenSpans  # noqa: E402

X = 4113
IMG = 248056          # image_token_id of the NF checkpoint (config.json)
IMG_TOKENS = 1024     # the needle image as P expands it (any K works)
URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."


def _text(n, seed):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


# the compact render of the needle: 70000 text tokens, the image's single
# placeholder, then the question -- the expanded prompt is 73112 tokens
HEAD = _text(70000, 1)
TAIL = _text(73112 - 70000 - IMG_TOKENS, 2)
COMPACT_1 = np.concatenate([HEAD, np.array([IMG], np.int32), TAIL])
# the follow-up turn: the whole first prompt, then 1454 new tokens
COMPACT_2 = np.concatenate([COMPACT_1, _text(74566 - 73112, 3)])


def _key():
    from sglang.srt.weg2.front_tokens import mm_image_key

    return mm_image_key({"url": URL, "detail": "auto", "max_dynamic_patch": None})


class _Tok:
    state = "ready"
    why = ""
    image_token_id = IMG

    def __init__(self):
        self.m = {}
        self.executor = None
        self.by_text = {}

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
    f.ftok.by_text = {"t1": COMPACT_1, "t2": COMPACT_2, "t1-repeat": COMPACT_1}
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = None
    f._store_probe_t = time.monotonic()
    return f


def _price(f, rid, t, est=76654):
    return asyncio.run(f._x_exact_price(rid, "/v1/chat/completions", {"t": t}, t, est, est,
                                        multimodal=True))


def _serve_needle(f):
    """weg2-0-1: fresh image -> fallback; P serves it (73112) and sleeps (flush)."""
    assert _price(f, "weg2-0-1", "t1") is None
    f._p_leg1_store_note("weg2-0-1", "t1", 73112)  # learns the image's token count first
    assert f._p_flush_store_presence() == 1


# ---- the helpers -----------------------------------------------------------------------

def test_expansion_puts_every_later_token_at_its_real_position():
    from sglang.srt.weg2.front_tokens import mm_expand

    k = _key()
    e = mm_expand(COMPACT_1, IMG, [k], {k: IMG_TOKENS})
    assert e.ids.size == 73112
    assert (e.first_image, e.image_end, e.ends) == (70000, 70000 + IMG_TOKENS, [71024])
    assert np.array_equal(e.ids[71024:], TAIL) and np.array_equal(e.ids[:70000], HEAD)
    assert mm_expand(COMPACT_1, IMG, [k], {}) is None, "an unseen image has no expansion"
    assert mm_expand(COMPACT_1, IMG, [k, k], {k: 3}) is None, "placeholders must match the images"


def test_the_token_count_is_learned_from_the_prompt_p_served():
    from sglang.srt.weg2.front_tokens import mm_learn

    k = _key()
    assert mm_learn(COMPACT_1.size, [k], {}, 73112) == (k, IMG_TOKENS)
    assert mm_learn(COMPACT_1.size, [k], {k: IMG_TOKENS}, 73112) is None, "nothing unknown"
    assert mm_learn(COMPACT_1.size, [k, "other"], {}, 73112) is None, "two unknowns: undetermined"


# ---- the front: weg2-1-5 and weg2-4-18 ------------------------------------------------

def test_weg2_1_5_a_repeated_image_request_is_priced_short_and_cached(caplog):
    caplog.set_level(logging.INFO)
    f = _front()
    _serve_needle(f)
    xx = _price(f, "weg2-1-5", "t1")
    assert xx is not None, "no chars/3 fallback for an image the front has seen served"
    assert (xx.pending, xx.credit, xx.n) == (24, 73088, 73112)
    assert xx.mm and xx.mm_cached and xx.image_end == 71024
    assert xx.pending <= X, "SHORT: no flip pair for 24 tokens"
    assert any(m.startswith("WEG2 X-EXACT-PRICE rid=weg2-1-5 pending=24 tokens=73112 credit=73088")
               and m.endswith("mm=1 mm_cached=1/1 image_end=71024") for m in caplog.messages)


def test_weg2_4_18_a_follow_up_turn_with_the_image_in_its_history_is_short():
    f = _front()
    _serve_needle(f)
    xx = _price(f, "weg2-4-18", "t2", est=1658)
    assert (xx.pending, xx.credit, xx.n) == (74566 - 73088, 73088, 74566)
    assert xx.pending == 1478 <= X and xx.mm_cached


def test_an_unseen_image_keeps_the_fallback_and_says_why(caplog):
    caplog.set_level(logging.INFO)
    f = _front()
    assert _price(f, "weg2-0-1", "t1") is None
    assert any("X-EXACT-FALLBACK rid=weg2-0-1 reason=multimodal-unseen-image" in m
               and "images=1 known=0" in m for m in caplog.messages)


def test_a_queued_p_only_image_request_is_released_at_the_store_presence(caplog):
    """weg2-1-5's arrival order: priced (fallback, image not learned yet) and
    queued P-only before P's flush published weg2-0-1's anchor."""
    caplog.set_level(logging.INFO)
    f = _front(awake="P")
    assert _price(f, "weg2-0-1", "t1") is None
    assert _price(f, "weg2-1-5", "t1-repeat") is None, "queued before the image was learned"
    p = F.Pending("weg2-1-5", "/v1/chat/completions", {}, "t1-repeat", time.time(),
                  asyncio.new_event_loop().create_future(), est_prompt=76654,
                  est_uncached=76654, p_only=True)
    f.queue.append(p)
    f._p_leg1_store_note("weg2-0-1", "t1", 73112)
    f._p_flush_store_presence()
    assert not p.p_only, "its image is in the store: no flip of its own"
    assert p.est_uncached == 24 and p.mm_image_end == 71024
    assert any("W102 Weg2VisionStage rid=weg2-1-5 image-cached at reprice" in m
               for m in caplog.messages)


def test_both_forcing_sites_spare_a_cached_image():
    src = inspect.getsource(F)
    assert "_mm_cached = bool(_xx is not None and getattr(_xx, \"mm_cached\", False))" in src
    assert "p_only=_verdict == VERDICT_STAGE and not _mm_cached" in src
    # IMAGE-CACHED-1002: the forcing branch is no longer an elif of the
    # cached one (an uncached verdict names its why first)
    assert ("if _verdict == VERDICT_STAGE and not _mm_cached and route != \"none\" "
            "and route != \"long\":") in src


# ---- P: no tower for a cached image ---------------------------------------------------

def _req(rid, offsets, prefix=0, host=0):
    item = types.SimpleNamespace(offsets=offsets, precomputed_embeddings=None, feature=object())
    return types.SimpleNamespace(rid=rid, multimodal_inputs=types.SimpleNamespace(mm_items=[item]),
                                 prefix_indices=[0] * prefix, host_hit_length=host)


def _sched(told=None, armed=True):
    return types.SimpleNamespace(_weg2_store_told_armed=armed, _weg2_store_told=told or {})


def test_p_skips_the_tower_when_the_told_prefix_holds_the_image(caplog):
    from sglang.srt.weg2 import vision_rank_runner as V

    caplog.set_level(logging.INFO)
    r = _req("weg2-1-5", [(70000, 71023)])
    s = _sched({"weg2-1-5": 73088})
    stage, wait = V.skip_cached(s, [r])
    assert (stage, wait) == ([], []) and r._weg2_vision_skip
    assert V.unstaged_items(r) == [], "PP0's verdict goes out as staged: followers release it"
    assert any(m.startswith("W102 Weg2VisionStage SKIP rid=weg2-1-5 reason=image-cached "
                            "covered=73088 src=told image_end=71024") for m in caplog.messages)


def test_p_stages_a_fresh_image_and_waits_for_an_outstanding_told():
    from sglang.srt.weg2 import vision_rank_runner as V

    fresh = _req("weg2-0-1", [(70000, 71023)])
    assert V.skip_cached(_sched({"weg2-0-1": 3264}), [fresh]) == ([fresh], [])
    local = _req("weg2-9-9", [(70000, 71023)], prefix=72000)
    assert V.skip_cached(_sched({}), [local]) == ([], [local]), "told outstanding, own match covers"
    cold = _req("weg2-9-8", [(70000, 71023)], prefix=100)
    assert V.skip_cached(_sched({}), [cold]) == ([cold], []), "nothing to wait for"
    nostore = _req("weg2-9-7", [(70000, 71023)], prefix=72000)
    assert V.skip_cached(_sched(armed=False), [nostore]) == ([], []) and nostore._weg2_vision_skip


def test_the_admission_belt_withdraws_a_skip_the_prefix_no_longer_covers():
    from sglang.srt.weg2 import vision_rank_runner as V

    r = _req("weg2-1-5", [(70000, 71023)], prefix=64000)
    r._weg2_vision_skip = True
    assert V.skip_still_covered(None, r) is False and not r._weg2_vision_skip
    assert V.unstaged_items(r), "staged in the next pass"
    ok = _req("weg2-1-6", [(70000, 71023)], prefix=73088)
    ok._weg2_vision_skip = True
    assert V.skip_still_covered(None, ok) is True


def test_wiring_the_belt_sits_in_the_admission_loop():
    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S)
    assert "skip_still_covered(self, req)" in src
    assert '_note_skip("weg2_vision_skip_refuted", req.rid)' in src


# ---- 27B rule: only a FULLY cached image skips the vision path ------------------------

def test_a_partially_cached_image_keeps_the_w102_path():
    """The store holds the prefix only into the image (70464 < image_end 71024):
    the tower must run before the prefill -- priced, but not mm_cached."""
    from sglang.srt.weg2.front_tokens import mm_expand

    f = _front()
    assert _price(f, "weg2-0-1", "t1") is None
    k = _key()
    f._mm_ktok()[k] = IMG_TOKENS
    ids = mm_expand(COMPACT_1, IMG, [k], f._mm_ktok()).ids
    f.tspans.record_store_depth(ids, 70464)
    xx = _price(f, "weg2-1-5", "t1")
    assert xx is not None and xx.credit == 70464 and not xx.mm_cached


def test_p_stages_a_partially_cached_image():
    from sglang.srt.weg2 import vision_rank_runner as V

    r = _req("weg2-1-5", [(70000, 71023)], prefix=70464)
    assert V.skip_cached(_sched({"weg2-1-5": 70464}), [r]) == ([r], [])
    assert not getattr(r, "_weg2_vision_skip", False)


# ---- CONTEXT-GATE (y7i 11:32-11:33Z: 358446 tokens > 262144, a flip pair every 12 s) --

def _gate_front(ctx=262144):
    f = _front()
    f._store_probe_info = {"context_length": ctx}
    return f


def _body(resp):
    import json

    return resp.status, json.loads(resp.body)


def test_an_over_context_request_is_answered_400_before_any_route(caplog):
    caplog.set_level(logging.INFO)
    f = _gate_front()
    xx = types.SimpleNamespace(n=358446)
    status, body = _body(f._context_gate("weg2-14-50", "t", xx, 365462))
    assert status == 400 and body["error"]["code"] == 400
    assert body["error"]["message"] == ("The input (358446 tokens) is longer than the model's "
                                        "context length (262144 tokens).")
    assert any(m.startswith("WEG2 CONTEXT-GATE REFUSED rid=weg2-14-50 tokens=358446 ctx=262144 "
                            "by=exact") for m in caplog.messages)


def test_the_inline_data_uri_estimate_is_refused_and_named(caplog):
    """No exact count: the request text is mostly a base64 data URI (OpenWebUI task
    prompt with the chat history); chars/3 is then close to the real count."""
    caplog.set_level(logging.INFO)
    f = _gate_front()
    text = "summarise " + "data:image/png;base64," + "A" * 1_090_000
    status, _ = _body(f._context_gate("weg2-15-51", text, None, 365462))
    assert status == 400
    assert any(m.startswith("WEG2 INLINE-DATAURI rid=weg2-15-51 chars=1090022") for m in caplog.messages)


def test_an_image_requests_compact_count_is_a_lower_bound_for_the_gate():
    f = _gate_front()
    f._x_exact_lb = collections.OrderedDict({"weg2-17-52": 300000})
    status, _ = _body(f._context_gate("weg2-17-52", "t", None, 10))
    assert status == 400


def test_the_gate_passes_what_fits_and_a_plain_estimate_inside_the_margin():
    f = _gate_front()
    assert f._context_gate("r", "t", types.SimpleNamespace(n=262144), 300000) is None
    assert f._context_gate("r", "plain text", None, 365462) is None, "chars/3 may over-count 60 %"
    assert _gate_front(ctx=0)._context_gate("r", "t", types.SimpleNamespace(n=10 ** 6), 0) is None


def test_wiring_the_gate_runs_before_any_verdict_seat_or_flip():
    src = inspect.getsource(F)
    g = src.index("_ctx_refusal = self._context_gate(rid, text, _xx, est_prompt)")
    assert g < src.index("route = serviceable_route(remainder, carrier_est,")
    assert g < src.index("p = Pending(rid, request.path, payload, text, time.time(), fut,")
    assert src.index("_xx = await self._x_exact_price(rid, request.path, payload, text,") < g
