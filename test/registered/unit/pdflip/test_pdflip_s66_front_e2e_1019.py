"""RV66 (review S66, Front 2c) -- end to end through Front.handle_generate, no structural pins:

(a) an UNPRICED image request (D probe fails) is routed to D (leg2), not p_only, counter set;
(b) D's admission refuses it W123 (image in its uncached tail) -> the front's requeue marks it p_only (SOLL R9) and it
    waits for P (queue), never a second D leg;
(c) switch off -> the pre-S66 rule (route long, p_only by rule);
(d) a PRICED probe that found an image in the uncached tail stays p_only (R9 unchanged);
(e) an image-cached priced request stays on D (unchanged)."""
import asyncio
import json
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "150")
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "10")

from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from test_pdflip_front_x_exact import _FakeTokens, _front  # noqa: E402

N = 3000


def _img_payload():
    return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "a" * 200},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}


def _mk(monkeypatch, probe_body=None, store=(0, "none")):
    f = _front(True)
    f.vision = F.VISION_MODE_TRANSIENT
    f.ftok = _FakeTokens(N)
    f.session = object()
    f._vision_warm = lambda *a, **k: None
    monkeypatch.setattr(F.Front, "_p_tok_ahead_maybe", staticmethod(lambda *a, **k: None))
    f.legs = []

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.routed.append(("short", rid))
        f.legs.append((rid, pending))
        return F.web.json_response({})

    f.leg2 = leg2

    async def post(g, b, timeout):
        if probe_body is None:
            return 503, "KeyError: 'pdflip-0-1~probe'"
        return 200, json.dumps(probe_body)

    f._d_probe_post = post

    async def st(rid, ids, timeout_s, extra_key=None):
        return store

    f._store_probe_depth = st
    return f


def _drive(f, payload):
    async def go():
        t = asyncio.create_task(f.handle_generate(F.web.Request.__new__(F.web.Request)
                                                  if False else _R(payload)))
        await asyncio.sleep(0.6)
        if not t.done():
            t.cancel()
            try:
                await t
            except BaseException:  # noqa: BLE001
                pass
    asyncio.run(go())


class _R:
    def __init__(self, p):
        self._p, self.path = p, "/v1/messages"

    async def json(self):
        return self._p


def test_a_unpriced_image_request_goes_to_d_not_p_only(monkeypatch, caplog):
    f = _mk(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _img_payload())
    msgs = [r.getMessage() for r in caplog.records]
    assert any("D-PROBE-UNPRICED" in m for m in msgs), msgs[-20:]
    assert f.counters["w102_unpriced_d_admission"] == 1
    assert not any("is prefilled on P by rule" in m for m in msgs)
    assert f.routed and f.routed[0][0] == "short", (f.routed, dict(f.counters))
    pend = f.legs[0][1]
    assert pend is None or not getattr(pend, "p_only", False)


def test_b_d_refuses_w123_requeue_is_p_only_and_waits_for_p(monkeypatch):
    f = _mk(monkeypatch)
    _drive(f, _img_payload())
    rid, pend = f.legs[0]
    body = (b'{"error": "W50 PdFlipTpPrefillExceeded (vision reroute): group D cannot serve this request itself; '
            b'this request\'s extent after prefix matching is 2900. Re-routed ... '
            b'W123 PdFlipVisionImageNotInPrefixOnD"}')
    f.routed.clear()

    async def run():
        t = asyncio.create_task(f._requeue_after_x_refusal(SimpleNamespace(path="/v1/messages"), rid,
                                                           _img_payload(), "t", False, pend, None, body))
        for _ in range(200):
            if f.queue:
                break
            await asyncio.sleep(0.005)
        q = list(f.queue)
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
        return q

    q = asyncio.run(run())
    assert len(q) == 1 and q[0].p_only is True, q
    assert q[0].est_uncached >= 2900 or q[0].est_uncached == N, q[0].est_uncached
    assert f.counters["w123_requeue_p_only"] == 1
    assert not f.routed, "no second D leg"


def test_c_switch_off_restores_the_rule(monkeypatch, caplog):
    monkeypatch.setenv(F.VISION_UNPRICED_TO_D_ENV, "0")
    f = _mk(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _img_payload())
    msgs = [r.getMessage() for r in caplog.records]
    assert any("is prefilled on P by rule" in m for m in msgs)
    assert f.counters["w102_unpriced_d_admission"] == 0
    assert not f.routed


def test_d_priced_image_in_tail_stays_p_only(monkeypatch, caplog):
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(N) + 10), "device_tokens": 1024, "host_tokens": 0,
            "key_match_depth": 1024, "mm_spans": [[2000, 2200]], "tokenize_ms": 1.0, "handle_ms": 1.0,
            "total_ms": 2.0}
    f = _mk(monkeypatch, probe_body=body, store=(1024, "l3_index"))
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _img_payload())
    msgs = [r.getMessage() for r in caplog.records]
    assert f.counters["w102_unpriced_d_admission"] == 0
    assert any("p_only" in m and "uncached tail" in m for m in msgs) or any(
        "is prefilled on P by rule" in m for m in msgs), msgs[-15:]
    assert not f.routed, "an image in the uncached tail never goes to D"


def test_e_priced_image_cached_stays_on_d(monkeypatch, caplog):
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(N) + 10), "device_tokens": 2944, "host_tokens": 0,
            "key_match_depth": 2944, "mm_spans": [[100, 300]], "tokenize_ms": 1.0, "handle_ms": 1.0,
            "total_ms": 2.0}
    f = _mk(monkeypatch, probe_body=body, store=(2944, "l3_index"))
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        _drive(f, _img_payload())
    msgs = [r.getMessage() for r in caplog.records]
    assert any("image-cached" in m for m in msgs), msgs[-15:]
    assert f.routed and f.counters["w102_unpriced_d_admission"] == 0
