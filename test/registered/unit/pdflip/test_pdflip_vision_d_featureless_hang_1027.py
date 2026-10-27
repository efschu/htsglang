"""Vision on D (``FLLIPER_PDFLIP_VISION_D=1``): feature-less items must never hang the D stage loop (27.10.).

METAL (boot formb98v / 11:27 int8, pdflip-2-2): ``running=1 queue=1`` for 8 minutes, the same rid answered ">1700x in 10 s"
by the admission loop's ``pdflip_vision_stage_d`` skip. Mechanism, three links:

  1. the D probe handler cached the FEATURE-LESS copy of the items for D's leg (``d_cache_mm(..., towerless=True)``) -- right
     for a tower-less D, wrong for a D with its own stage (the stage encodes from the leg's pixels);
  2. the front priced the uncached image like text (``mm_cached = not image_in_tail`` is always True when D encodes), so
     the request went to D as "image-cached" instead of as a tower pass;
  3. the D verdict marked the request ``stage`` pass after pass; ``d_pass`` plans only items that carry a feature, so it
     planned nothing, nobody refused, nothing ended the wait.

The fix closes every link (pixels kept, uncached image priced as uncached, feature-less -> refusal through P, stage wait
bounded). Each case below is one link; the loop cases run the scheduler's own order (``d_pass`` first, then the verdict of
every waiting request) and fail on the old code by never leaving the queue.
"""

import json
import logging
import types

import numpy as np
import pytest
import torch

from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.pdflip import d_match_probe as dmp
from flliper.srt.pdflip import vision_d_guard as g
from flliper.srt.pdflip import vision_d_runner as vdr
from test_pdflip_vision_d_runner_1027 import _Comm, _Wire
from test_pdflip_vision_rows_tail_1021 import COVERED, HASH, IMAGE, N, _ans, _item, _sched, store  # noqa: F401

PASSES = 40          # the real loop runs ~170 passes/s: far more than a stage needs (it runs in the NEXT pass)


def _req(rid, spans, featured):
    """A request whose items are the given inclusive-offset spans; ``featured[i]``: item i still carries its pixels."""
    items = []
    for k, (span, feat) in enumerate(zip(spans, featured)):
        it = _item(span=span, h=HASH + k)
        if feat:
            it.feature = torch.zeros(1)
        items.append(it)
    return types.SimpleNamespace(
        rid=rid, full_untruncated_fill_ids=list(range(N)), output_ids=[],
        multimodal_inputs=types.SimpleNamespace(mm_items=items),
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)))


def _stage_features_only(scheduler, reqs):
    """The leader's stage: rows for every item that has pixels, nothing for one that has none."""
    for r in reqs:
        for it in r.multimodal_inputs.mm_items:
            if it.feature is not None:
                it.precomputed_embeddings = torch.zeros(4, 8, dtype=torch.bfloat16)
    return True, "", ""


def _group_d(queue):
    h = _sched()
    h._pdflip_vision_d_stage = True
    h.waiting_queue = queue
    h._pdflip_vision_d_leader = True
    h._pdflip_vision_refused = set()
    h._pdflip_vision_origin_aborts = []
    h._pdflip_vision_runs = 0
    return h


def _admission(queue, passes=PASSES):
    """The scheduler's order per pass -- D's vision pass, then the verdict of each waiting request. Returns
    {rid: (verdict, pass number)} for every request that left the queue."""
    h = _group_d(queue)
    left = {}
    for n in range(1, passes + 1):
        vdr.d_pass(h, comm=_Comm(_Wire(), True), stage=_stage_features_only)
        for r in list(queue):
            v = Scheduler._pdflip_vision_d_verdict(h, r, None)
            if v == "stage":
                continue
            left[r.rid] = (v, n)
            queue.remove(r)
        if not queue:
            break
    return left


def test_a_featureless_item_is_refused_through_p_in_one_pass_not_staged_for_ever(store):
    """THE HANG: the item arrived without its pixels (the probe copy), so no pass can encode it. It is refused in the pass
    it is first seen (W123 -> the front's X-REQUEUE through P), never marked ``stage`` again and again."""
    req = _req("ft", [IMAGE], [False])
    assert _admission([req]) == {"ft": (g.REFUSE, 1)}
    assert not getattr(req, "_pdflip_vision_d_stage_wanted", False)


def test_a_featureless_request_does_not_hold_up_a_normal_one_in_the_same_queue(store):
    """Mixed queue: the featured request is encoded by the stage (pass 1 marks, pass 2 encodes and admits), the
    feature-less one is refused in pass 1 -- neither waits for the other."""
    q = [_req("ft", [IMAGE], [False]), _req("ok", [IMAGE], [True])]
    assert _admission(q) == {"ft": (g.REFUSE, 1), "ok": (g.ADMIT, 2)}


def test_a_request_with_one_encodable_and_one_featureless_image_ends_in_a_refusal(store):
    """Mixed request: image A is encoded by the stage, image B has no pixels. After the pass the verdict still needs a tower
    for B, nothing can plan it -- that is the refusal through P, not a second ``stage`` for ever."""
    other = (IMAGE[1] + 10, IMAGE[1] + 100)
    q = [_req("mix", [IMAGE, other], [True, False])]
    assert _admission(q) == {"mix": (g.REFUSE, 2)}


def test_a_stage_that_never_runs_is_refused_by_name_after_the_defer_bound(store, caplog):
    """The wait itself is bounded (the defer bound does not cover a ``stage`` verdict): a marked request whose pass never
    ran is refused in the pass after DEFER_BOUND_PASSES, with the named line W125c."""
    h = _group_d([])
    req = _req("never", [IMAGE], [True])
    with caplog.at_level(logging.ERROR):
        seen = [Scheduler._pdflip_vision_d_verdict(h, req, None) for _ in range(g.DEFER_BOUND_PASSES + 2)]
    assert seen[:g.DEFER_BOUND_PASSES] == ["stage"] * g.DEFER_BOUND_PASSES
    assert seen[g.DEFER_BOUND_PASSES:] == [g.REFUSE, g.REFUSE]
    assert sum("W125c" in r.getMessage() and "never" in r.getMessage() for r in caplog.records) == 2


def test_the_refusal_of_a_featureless_item_is_the_w50_reroute_through_p(store):
    """W123/W50 stays intact: the verdict's refusal of an unstreamed feature-less request is answered with the front's
    X-REQUEUE text (W50 marker, extent, W123 named inside), through the real answer path."""
    from flliper.srt.pdflip import front as F

    req = _req("ft", [IMAGE], [False])
    h = _group_d([req])
    assert Scheduler._pdflip_vision_d_verdict(h, req, None) == g.REFUSE
    sent = []
    h.tree_cache = None
    h.enable_hicache_storage = h.enable_hierarchical_cache = False
    h.ipc_channels = types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
        send_output=lambda out, r: sent.append(out)))
    Scheduler._pdflip_answer_vision_d_refusals(h, [req], None)
    msg = sent[0].finished_reason["message"]
    assert F.x_refusal_marker_in(msg) and g.W_NOT_IN_PREFIX in msg
    assert F._d_refusal_extent(msg.encode()) == N - COVERED
    assert h.waiting_queue == []


def test_a_streamed_featureless_request_keeps_the_plain_w123(store):
    """A request that already sent a byte cannot go back through P (RESUME-VIA-P): feature-less or not, it is refused as
    before and answered with the named W123."""
    req = _req("ft", [IMAGE], [False])
    req.output_ids = [1, 2, 3]
    h = _group_d([req])
    assert Scheduler._pdflip_vision_d_verdict(h, req, None) == g.REFUSE
    assert not g.reroute_eligible(req)


# ------------------------------------------------------------------------------------------------ the front's pricing
def test_an_uncached_image_is_not_priced_like_text_when_d_encodes_it():
    """Link 2: with D's own stage ``image_in_tail`` is always False (the image is no P-only tail), but the image is still
    not cached -- it needs a tower pass. ``image_cached`` answers the vision terms, not ``not image_in_tail``."""
    new = dmp.exact_price(_ans(N, COVERED, [IMAGE]), COVERED, d_encodes=True)
    assert new.image_in_tail is False and dmp.image_cached(new) is False
    rows = dmp.exact_price(_ans(N, COVERED, [IMAGE], rows=[IMAGE]), COVERED, d_encodes=True)
    assert dmp.image_cached(rows) is True                       # P's rows exist: computed like text
    inside = dmp.exact_price(_ans(N, COVERED, [(100, 600)]), COVERED, d_encodes=True)
    assert dmp.image_cached(inside) is True                     # inside the covered prefix: no tower pass either


def test_the_front_reports_the_uncached_image_as_not_cached_with_vision_on_d():
    """Through the real ``Front._x_exact_price``: vision on D, D's probe names the image span and no rows -> the priced
    answer says ``mm_cached`` False (before: True, "image-cached ... no tower", and D got an image without rows); with
    published rows it is cached."""
    import asyncio

    from test_pdflip_front_x_exact import _FakeTokens, _front, _payload

    f = _front(True)
    f.vision_on_d = True
    f.ftok = _FakeTokens(127482)
    f.session = object()
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(N) + 10), "device_tokens": COVERED, "host_tokens": 0,
            "key_match_depth": COVERED, "mm_spans": [list(IMAGE)], "mm_rows": [],
            "tokenize_ms": 641.3, "handle_ms": 23.4, "total_ms": 734.4}

    async def post(gr, b, timeout):
        return 200, json.dumps(body)

    async def store_depth(rid, ids, timeout_s, extra_key=None):
        return COVERED, "l3_index"

    f._d_probe_post = post
    f._store_probe_depth = store_depth
    payload = _payload(100)
    payload["messages"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]})
    xx = asyncio.run(f._x_exact_price("pdflip-2-2", "/v1/messages", payload, "t", 179, 127482))
    assert xx.image_in_tail is False and xx.mm_cached is False
    body["mm_rows"] = [list(IMAGE)]
    xx = asyncio.run(f._x_exact_price("pdflip-2-2b", "/v1/messages", payload, "t2", 179, 127482))
    assert xx.mm_cached is True


# ---------------------------------------------------------------------------------------------- the pixels on D's legs
@pytest.mark.parametrize("env,vision_cfg,want", [
    ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION_D": "1"}, True, False),   # D encodes: the leg keeps its pixels
    ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION_D": "0"}, True, True),    # kill switch: tower-less D, no pixels
    ({"FLLIPER_PDFLIP_GROUP": "D"}, True, True),                                    # dual layout: the launcher publishes no flag
    ({"FLLIPER_PDFLIP_GROUP": "P", "FLLIPER_PDFLIP_VISION_D": "1"}, True, False),   # not group D: never a tower-less D
])
def test_the_d_probe_caches_pixels_when_d_has_its_own_stage(env, vision_cfg, want):
    """Link 1: ``http_server.pdflip_match_probe`` hands ``d_leg_without_pixels`` to ``d_cache_mm``; True strips the
    cached copy of D's leg."""
    cfg = types.SimpleNamespace(is_multimodal=True, hf_config=types.SimpleNamespace(
        language_model_only=True, vision_config=object() if vision_cfg else None))
    assert vdr.d_leg_without_pixels(cfg, env) is want


def test_d_cache_mm_keeps_the_originals_for_a_d_with_a_stage():
    """The consumer of that flag: ``False`` caches the ORIGINAL message (pixels) for the leg, ``True`` the feature-less copy."""
    from flliper.srt.pdflip import tok_ahead as tah

    item = types.SimpleNamespace(feature=torch.ones(2), set_pad_value=lambda: None)
    mm = types.SimpleNamespace(mm_items=[item])
    sched, cache = tah.d_cache_mm(mm, False)
    assert cache is mm and cache.mm_items[0].feature is not None and sched.mm_items[0].feature is None
    sched, cache = tah.d_cache_mm(mm, True)
    assert cache is sched and cache.mm_items[0].feature is None


def test_the_d_mm_item_cache_stays_off_on_a_d_with_its_own_stage(monkeypatch):
    """Link 1, second door: the D-MM-ITEM-CACHE also builds D legs from feature-less items; it must not arm where D
    encodes."""
    from flliper.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    from flliper.srt.pdflip import d_mm_item_cache as K

    sa = types.SimpleNamespace(language_only=False, skip_tokenizer_init=False, keep_mm_feature_on_device=False)
    cfg = types.SimpleNamespace(is_multimodal=True, hf_config=types.SimpleNamespace(
        language_model_only=True, vision_config=object()))
    proc = object.__new__(QwenVLImageProcessor)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_VISION_D", "1")
    assert "encodes images itself" in K.armed_reason(server_args=sa, model_config=cfg, mm_processor=proc)
    monkeypatch.setenv("FLLIPER_PDFLIP_VISION_D", "0")
    assert "encodes images itself" not in K.armed_reason(server_args=sa, model_config=cfg, mm_processor=proc)
