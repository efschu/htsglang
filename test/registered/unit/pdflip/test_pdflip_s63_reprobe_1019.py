"""S63 (NF 19.10.): exact numbers AT THE DECISION -- F1 re-probe before a set verdict that flips or assigns P, EARLY-FLIP
on the exact price, F2 one tokenization per group and request (TOK-AHEAD).

User 19.10.: "warum wird irgendwas geschätzt? alle zahlen liegen doch schwarz auf weiß vor zu jedem zeitpunkt?"
SOLL v2 R1/R3/R4/R7/R8 (the verdict after T_halb), R2 (a kurz request never waits), R9.

F1 (review S56 F1): a big member priced 9000 at its arrival waits T_halb (R4); meanwhile D's coverage grows (a writeback,
a park anchor) so that D holds all but 300 of it. Before S63 the verdict after T_halb flipped on the arrival's 9000 -- a
flip under X (R7). Now the member is probed again (D's tree read-only + the store) right before the verdict stands.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import types

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front, _pending  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402  (T_halb 4.5 s, F_halb 0.5: SOLL section 5)

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import tok_ahead as tah  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-pdflip-unit")

X = 6000
N_BIG = 9000


def _verdicts(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP SET-VERDICT" in r.getMessage()]


def _fx(running=("dec",)):
    f = _front(running=list(running), n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X
    f.t_awake = time.time() - 60.0
    f.session = object()
    return f


def _book(f, rid, n, covered, age_s, spans=()):
    ids = np.arange(n, dtype=np.int32) + 7
    ans = dmp.ProbeAnswer(ids=ids, device_tokens=covered, host_tokens=0, key_match_depth=covered,
                          mm_spans=tuple(spans), extra_key=None, extra_key_known=True)
    xp = dmp.exact_price(ans, 0)
    F.Front._reprobe_note(f, rid, dmp.ReprobeEntry(ids=ids, spans=tuple(spans), extra_key=None, extra_key_known=True,
                                                    path="/v1/messages", payload={"rid": rid}, ns_known=True, ns=None,
                                                    t=time.monotonic() - age_s, price=xp))
    return xp


class _FakeD:
    """D's probe endpoint: answers with the CURRENT coverage of each rid (``cover``), after ``delay`` s; ``hang`` never."""

    def __init__(self, cover, delay=0.02, hang=False):
        self.cover, self.delay, self.hang, self.asked = dict(cover), delay, hang, []

    def install(self, monkeypatch):
        fake = self

        def start(self_f, rid, path, payload, timeout_s=None):
            fake.asked.append((rid, path, dict(payload)))

            async def go():
                if fake.hang:
                    await asyncio.sleep(3600)
                await asyncio.sleep(fake.delay)
                ids = dmp.ids_from_b64(payload["input_ids_b64"]) if "input_ids_b64" in payload else None
                cov = int(fake.cover[rid])
                return 200, json.dumps({"success": True, "ids": dmp.ids_to_b64(ids), "device_tokens": cov,
                                        "host_tokens": 0, "key_match_depth": cov, "mm_spans": [],
                                        "extra_key": None})

            t = asyncio.ensure_future(go())
            t._dmp_req = (path, payload)
            return t

        async def store(self_f, rid, ids, timeout_s, extra_key=None):
            return 0, "none"

        monkeypatch.setattr(F.Front, "_d_probe_start", start)
        monkeypatch.setattr(F.Front, "_store_probe_depth", store)
        return self


async def _drive(f, until_s, step=0.05):
    out = []
    t_end = time.time() + until_s
    while time.time() < t_end:
        out.append(f._decode_collect(time.time()))
        if out[-1] in ("P", "D"):
            break
        await asyncio.sleep(step)
    return out


# ---------------------------------------------------------------------------------------------------------- F1

def test_f1_coverage_grows_between_arrival_and_t_half_no_flip(monkeypatch, caplog):
    """THE REGRESSION (R4 + R7): big at arrival (9000), D's coverage grows to 8700 before T_halb ends -> the re-probe
    finds u=300: no flip, the entry is a SHORT for D now."""
    _on(monkeypatch)
    f = _fx()
    p = _pending("big", N_BIG, time.time() - 4.6)   # T_halb over: R4 'big member alone' flips on the arrival number
    f.queue = [p]
    _book(f, "big", N_BIG, 0, age_s=4.6)
    d = _FakeD({"big": 8700}).install(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        out = asyncio.run(_drive(f, 1.0))
    assert "P" not in out, (out, _verdicts(caplog))
    assert f.counters["set_flip_p"] == 0 and f.counters["set_park_no_seat"] == 0
    assert not [m for m in _verdicts(caplog) if "verdict=FLIP_P" in m], _verdicts(caplog)
    assert p.est_uncached == 300 and p.d_eligible is True
    assert [a[0] for a in d.asked] == ["big"] and d.asked[0][1] == "/generate", d.asked
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PDFLIP SET-REPROBE rid=big" in m and "u 9000 -> 300" in m for m in msgs), msgs
    assert f._set_st()["go_p"] == set()


def test_f1_coverage_shrinks_the_flip_stands_on_the_new_number(monkeypatch, caplog):
    """Both directions: a member priced halb at arrival whose coverage D evicted is big NOW -> R4 flip on 9000."""
    _on(monkeypatch)
    f = _fx()
    f._set_st()["shorts"]["h"] = (time.time() - 4.6, 3200, X)
    f._set_st()["shorts"]["h2"] = (time.time() - 4.6, 3300, X)   # S=6500 >= X: R3 flips -- on stale numbers
    _book(f, "h", N_BIG, N_BIG - 3200, age_s=4.6)
    _book(f, "h2", 3300, 0, age_s=4.6)
    _FakeD({"h": 0, "h2": 0}).install(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        out = asyncio.run(_drive(f, 1.0))
    assert out[-1] == "P", out
    (line,) = [m for m in _verdicts(caplog) if "verdict=FLIP_P" in m or "verdict=PARK_NO_SEAT" in m]
    assert "S=12300" in line, line   # 9000 + 3300: D's numbers at the verdict
    assert f.counters["set_reprobe_begin"] == 1


def test_f1_burst_of_fresh_arrivals_flips_at_once_without_a_reprobe(monkeypatch, caplog):
    """Burst advantage (R3): two halb arrivals probed just now, S >= X -> the flip at the first look, no re-probe."""
    _on(monkeypatch)
    f = _fx()
    now = time.time()
    f._set_st()["shorts"]["a"] = (now, 3500, X)
    f._set_st()["shorts"]["b"] = (now, 3200, X)
    _book(f, "a", 3500, 0, age_s=0.05)
    _book(f, "b", 3200, 0, age_s=0.02)
    d = _FakeD({"a": 0, "b": 0}).install(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(time.time()) == "P"
    assert d.asked == [] and f.counters["set_reprobe_begin"] == 0


def test_f1_r2_a_kurz_arrival_never_waits_for_a_reprobe(monkeypatch, caplog):
    """R2: while the re-probe of a set is in flight the flip is not decided -- a kurz request takes its D seat now."""
    _on(monkeypatch)
    f = _fx()
    f.queue = [_pending("big", N_BIG, time.time() - 4.6)]
    _book(f, "big", N_BIG, 0, age_s=4.6)
    _FakeD({"big": 0}, delay=2.0).install(monkeypatch)

    async def run():
        assert f._decode_collect(time.time()) == "hold"
        assert F.Front._reprobe_pending(f)
        assert F.Front._set_flip_foreseeable(f, time.time()) is False
        t0 = time.time()
        seat = await f._acquire_short_seat("k", 300, uncached=300)
        return seat, time.time() - t0

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        seat, waited = asyncio.run(run())
    assert seat is not None and waited < 0.5, waited


def test_f1_a_d_that_never_answers_does_not_hang_the_set(monkeypatch, caplog):
    """Hänger guard: no answer -> bounded by the probe deadline, named SET-REPROBE-FAILED, the set decides."""
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "200")
    monkeypatch.setenv("FLLIPER_PDFLIP_FRONT_EXACT_TIMEOUT_MS", "100")
    f = _fx()
    f.queue = [_pending("big", N_BIG, time.time() - 4.6)]
    _book(f, "big", N_BIG, 0, age_s=4.6)
    _FakeD({"big": 0}, hang=True).install(monkeypatch)
    t0 = time.time()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        out = asyncio.run(asyncio.wait_for(_drive(f, 4.0), 5.0))
    assert out[-1] == "P" and time.time() - t0 < 3.0, (out, time.time() - t0)
    assert f.counters["set_reprobe_failed"] == 1
    assert any("PDFLIP SET-REPROBE-FAILED rid=big" in r.getMessage() for r in caplog.records)


def test_f1_the_queue_sum_verdict_is_gated_too(monkeypatch):
    _on(monkeypatch)
    f = _fx()
    f.queue = [_pending("q1", 3500, time.time() - 1.0), _pending("q2", 3500, time.time() - 1.0)]
    _book(f, "q1", 3500, 0, age_s=1.0)
    _book(f, "q2", 3500, 0, age_s=1.0)
    _FakeD({"q1": 3400, "q2": 3400}).install(monkeypatch)
    f._decode_collect = lambda now, live_q=None: None   # isolate the queue-sum site

    async def run():
        first = F.Front._set_flip_justified(f, time.time())
        await asyncio.sleep(0.2)
        return first, F.Front._set_flip_justified(f, time.time())

    first, second = asyncio.run(run())
    assert first is False and second is False      # 100 + 100 after the re-probe: no flip under X (R7)
    assert [q.est_uncached for q in f.queue] == [100, 100]


def test_f1_switch_default_on_and_off_is_the_old_path(monkeypatch):
    assert envs.FLLIPER_PDFLIP_SET_REPROBE.default is True
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_PDFLIP_SET_REPROBE", "0")
    f = _fx()
    f.queue = [_pending("big", N_BIG, time.time() - 4.6)]
    _book(f, "big", N_BIG, 0, age_s=4.6)
    d = _FakeD({"big": 8700}).install(monkeypatch)
    assert f._decode_collect(time.time()) == "P" and d.asked == []


def test_f1_reprobe_body_is_ds_own_key_and_the_identity_is_checked():
    ids = np.arange(50, dtype=np.int32)
    e = dmp.ReprobeEntry(ids=ids, spans=((3, 9),), extra_key="salt", extra_key_known=True, path="/v1/messages",
                         payload={"x": 1}, ns_known=True, ns="salt", t=0.0)
    path, body = dmp.reprobe_body(e, "r")
    assert path == "/generate" and body["extra_key"] == "salt"
    assert np.array_equal(dmp.ids_from_b64(body["input_ids_b64"]), ids)
    e.extra_key_known = False
    assert dmp.reprobe_body(e, "r") == ("/v1/messages", {"x": 1})
    ok = dmp.ProbeAnswer(ids=ids.copy(), device_tokens=5, host_tokens=0, key_match_depth=5, mm_spans=())
    bad = dmp.ProbeAnswer(ids=ids[:-1].copy(), device_tokens=5, host_tokens=0, key_match_depth=5, mm_spans=())
    assert dmp.reprobe_same_key(e, ok) and not dmp.reprobe_same_key(e, bad)
    assert dmp.with_spans(ok, e.spans).mm_spans == ((3, 9),)


def test_f1_an_image_covered_now_is_no_longer_p_only(monkeypatch):
    """R9 from D's number at the verdict: the image span (ends at 2000) is inside D's coverage now."""
    _on(monkeypatch)
    f = _fx()
    p = _pending("img", 7000, time.time() - 4.6)
    p.p_only = True
    f.queue = [p]
    _book(f, "img", 7000, 0, age_s=4.6, spans=((100, 2000),))
    _FakeD({"img": 6900}).install(monkeypatch)
    asyncio.run(_drive(f, 0.6))
    assert p.p_only is False and p.est_uncached == 100


# ---------------------------------------------------------------------------------------------------------- F2

class _Obj:
    def __init__(self, rid, text="t", image_data=None):
        self.rid, self.text, self.input_ids, self.image_data = rid, text, None, image_data
        self.video_data = self.audio_data = self.mm_hashes = self.modalities = self.input_embeds = None


def test_f2_digest_identity():
    a = _Obj("pdflip-1-1", "hello", "data:image/png;base64," + "A" * 5000)
    b = _Obj("pdflip-1-1", "hello", ["data:image/png;base64," + "A" * 5000])   # the tokenizer lists a single datum
    c = _Obj("pdflip-1-1", "hello", ["data:image/png;base64," + "B" * 5000])
    d = _Obj("pdflip-1-1", "hello!", ["data:image/png;base64," + "A" * 5000])
    assert tah.obj_digest(a) == tah.obj_digest(b)
    assert tah.obj_digest(a) != tah.obj_digest(c) and tah.obj_digest(a) != tah.obj_digest(d)


def test_f2_cache_take_once_mismatch_inflight_ttl():
    async def run():
        ta = tah.TokAhead(keep=2, ttl_s=60.0)
        ta.put("pdflip-1-1~probe", "dg", [1, 2, 3], "MM", None, 12.0, "d_probe")
        e, why = await ta.take("pdflip-1-1", "other")
        assert e is None and why == "digest"
        ta.put("pdflip-1-1", "dg", [1, 2, 3], "MM", None, 12.0, "d_probe")
        e, why = await ta.take("pdflip-1-1", "dg")
        assert e.input_ids == [1, 2, 3] and e.mm_inputs == "MM"
        assert (await ta.take("pdflip-1-1", "dg"))[0] is None   # one leg, one reuse
        fut = ta.begin("pdflip-2-2", "d2")

        async def later():
            await asyncio.sleep(0.05)
            ta.put("pdflip-2-2", "d2", [9], "MM2", None, 50.0, "p_ahead")

        asyncio.ensure_future(later())
        e2, _ = await ta.take("pdflip-2-2", "d2", wait_s=2.0)   # awaited, not tokenized twice
        assert e2.mm_inputs == "MM2" and fut.done()
        ta.begin("pdflip-3-3", "d3")
        asyncio.get_running_loop().call_later(0.05, ta.fail, "pdflip-3-3")
        e3, why3 = await ta.take("pdflip-3-3", "d3", wait_s=2.0)  # a failed ahead never strands the leg
        assert e3 is None and why3 == "none"
        short = tah.TokAhead(keep=2, ttl_s=0.0)
        short.put("pdflip-4-4", "d4", [1], None, None, 1.0, "x")
        await asyncio.sleep(0.01)
        assert (await short.take("pdflip-4-4", "d4"))[1] == "none"

    asyncio.run(run())


def test_f2_scheduler_copy_keeps_the_originals_whole():
    class Item:
        def __init__(self):
            self.feature, self.pad_value = "PIXELS", None

        def set_pad_value(self):
            self.pad_value = 4242

    mm = types.SimpleNamespace(mm_items=[Item(), Item()], other=1)
    c = tah.scheduler_copy_without_features(mm)
    assert [i.feature for i in mm.mm_items] == ["PIXELS", "PIXELS"]
    assert [i.feature for i in c.mm_items] == [None, None] and [i.pad_value for i in c.mm_items] == [4242, 4242]
    assert c.other == 1 and c is not mm


def _tm(monkeypatch):
    from flliper.srt.managers import tokenizer_manager as tmm

    tm = tmm.TokenizerManager.__new__(tmm.TokenizerManager)
    tm.pdflip_tok_ahead = tah.TokAhead()
    tm.calls = []
    tm._validate_mm_limits = lambda obj: tm.calls.append("limits")
    tm._validate_mm_lane_support = lambda obj: tm.calls.append("lane")
    tm._validate_one_request = lambda obj, ids: tm.calls.append(("len", len(ids)))
    tm._create_tokenized_object = lambda obj, text, ids, emb, mm, tt: ("TOKENIZED", obj.rid, list(ids), mm)
    return tmm, tm


def test_f2_the_leg_takes_the_probe_tokenization_by_digest(monkeypatch, caplog):
    tmm, tm = _tm(monkeypatch)
    probe = tmm.GenerateReqInput(rid="pdflip-5-5~probe", text="hi", image_data=["data:x" + "Q" * 400])
    tm.pdflip_tok_ahead.put(probe.rid, tah.obj_digest(probe), [5, 6, 7], "MMITEMS", None, 640.0, "d_probe")
    leg = tmm.GenerateReqInput(rid="pdflip-5-5", text="hi", image_data="data:x" + "Q" * 400,
                               sampling_params={"max_new_tokens": 99}, stream=True)
    with caplog.at_level(logging.INFO):
        out = asyncio.run(tm._pdflip_tok_ahead_take(leg))
    assert out == ("TOKENIZED", "pdflip-5-5", [5, 6, 7], "MMITEMS")
    assert "limits" in tm.calls and ("len", 3) in tm.calls
    assert any("PDFLIP TOK-AHEAD-REUSE rid=pdflip-5-5 src=d_probe n=3 images=0" in r.getMessage()
               for r in caplog.records)


def test_f2_a_different_request_is_named_and_tokenizes_itself(monkeypatch, caplog):
    tmm, tm = _tm(monkeypatch)
    probe = tmm.GenerateReqInput(rid="pdflip-6-6~probe", text="hi", image_data=["data:x" + "Q" * 400])
    tm.pdflip_tok_ahead.put(probe.rid, tah.obj_digest(probe), [5, 6, 7], "MM", None, 640.0, "d_probe")
    leg = tmm.GenerateReqInput(rid="pdflip-6-6", text="hi", image_data=["data:x" + "R" * 400])
    with caplog.at_level(logging.INFO):
        assert asyncio.run(tm._pdflip_tok_ahead_take(leg)) is None
    assert any("PDFLIP TOK-AHEAD-MISMATCH rid=pdflip-6-6 why=digest" in r.getMessage() for r in caplog.records)
    # the probe's own tokenization never takes anything
    assert asyncio.run(tm._pdflip_tok_ahead_take(probe)) is None


def test_f2_front_posts_ahead_to_p_only_for_a_media_prompt_that_can_reach_p(monkeypatch):
    f = _fx()
    f.groups["P"] = types.SimpleNamespace(name="P", url="http://p")
    sent = []

    async def post(self_f, g, rid, path, payload):
        sent.append(rid)

    monkeypatch.setattr(F.Front, "_p_tok_ahead_post", post)
    img = {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}]}]}
    txt = {"messages": [{"role": "user", "content": "hello"}]}
    xx = types.SimpleNamespace(unpriced=False, image_in_tail=False)

    async def run():
        r = [F.Front._p_tok_ahead_maybe(f, "a", "/v1/messages", img, xx, 3500),      # halb: can reach P
             F.Front._p_tok_ahead_maybe(f, "b", "/v1/messages", img, xx, 300),       # kurz: D serves it (R2)
             F.Front._p_tok_ahead_maybe(f, "c", "/v1/messages", txt, xx, 9000),      # text: nothing to process
             F.Front._p_tok_ahead_maybe(f, "d", "/v1/messages", img,
                                        types.SimpleNamespace(unpriced=False, image_in_tail=True), 300)]  # R9
        f.awake = "P"
        r.append(F.Front._p_tok_ahead_maybe(f, "e", "/v1/messages", img, xx, 9000))  # P awake: its leg comes now
        await asyncio.sleep(0.01)
        return r

    assert asyncio.run(run()) == [True, False, False, True, False]
    assert sorted(sent) == ["a", "d"]
    assert envs.FLLIPER_PDFLIP_TOK_AHEAD.default is True
