"""RANKSTATS-S3 "Vorschlag DASHBOARD-GRAFIKEN" (29.09., angenommen): the P->D hand-off
apart from the cache in ``state.json front.served_tokens``.

Feld 1: ``D`` mixes a D-direct leg (``cached`` = a real prefix hit) and leg 2 after
a P prefill of the SAME rid (``cached`` = what D takes over from P). The front
writes the second case once more under ``D_after_P`` -- a subset of ``D``, same
row shape, cumulative -- exactly when ``pending is not None and pending.leg1_ran``.

Feld 2: each row counts ``cached_tier {device, host, storage}`` from the answer's
CachedTokensDetails (meta_info / sglext / usage.prompt_tokens_details). An answer
without the detail adds nothing to the tiers; the row then carries no tier key,
so a 27B answer without details leaves ``served_tokens`` byte-identical.

Hermetic, CPU, loopback only: the REAL ``Front.leg2`` (streamed and non-streamed
branch behind a real aiohttp front app, as H85) against a fake D.
"""
from __future__ import annotations

import asyncio
import json
import time

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, Pending
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

TIERS = {"device": 600, "host": 150, "storage": 50, "storage_backend": "file"}


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _fake_d_app(plan: dict) -> web.Application:
    """``plan[rid]`` = (prompt, cached, details or None, holder): what D answers."""

    async def chat(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        rid = payload["rid"]
        prompt, cached, details, holder = plan[rid]
        ptd = {"cached_tokens": cached}
        if details is not None and holder == "usage":
            ptd["cached_tokens_details"] = details
        usage = {"prompt_tokens": prompt, "completion_tokens": 3, "prompt_tokens_details": ptd}
        extra = {"sglext": {"cached_tokens_details": details}} if details is not None and holder == "sglext" else {}
        if not payload.get("stream"):
            body = {"id": rid, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}], "usage": usage}
            body.update(extra)
            return web.json_response(body)
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        await resp.write(_sse({"id": rid, "choices": [{"index": 0, "delta": {"content": "ok"},
                                                       "finish_reason": None}]}))
        await resp.write(_sse({"id": rid, "choices": [{"index": 0, "delta": {},
                                                       "finish_reason": "stop"}]}))
        await resp.write(_sse(dict({"id": rid, "choices": [], "usage": usage}, **extra)))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def info(request: web.Request) -> web.Response:
        return web.json_response({"internal_states": [{}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/get_server_info", info)
    return app


def _pending(rid: str, leg1_ran: bool) -> Pending:
    p = Pending(rid=rid, path="/v1/chat/completions", payload={}, text="q",
                t_arrive=time.time(), fut=None)
    p.leg1_ran = leg1_ran
    return p


async def _legs(legs, plan: dict) -> Front:
    """``legs`` = [(rid, stream, pending-or-None)] through the REAL ``Front.leg2``."""
    pendings = {rid: pend for rid, _, pend in legs}

    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        rid = request.headers["x-rid"]
        payload["rid"] = rid
        return await front.leg2(request, rid, payload, "q", bool(payload.get("stream")), pendings[rid])

    d = TestServer(_fake_d_app(plan))
    await d.start_server()
    front = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=262144, tp_prefill_max_tokens=4096)
    front.session = aiohttp.ClientSession()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    try:
        async with aiohttp.ClientSession() as cs:
            for rid, stream, _ in legs:
                body = {"model": "m", "max_tokens": 8, "stream": stream,
                        "messages": [{"role": "user", "content": "q"}]}
                async with cs.post(f"http://{host}:{port}/v1/chat/completions", json=body,
                                   headers={"x-rid": rid}) as resp:
                    await resp.read()
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front


def _served(front: Front) -> dict:
    return front._ipc_front_fields()["served_tokens"]


# ------------------------------------------------------------------ Feld 1
def test_red_first_leg2_after_p_counts_under_d_after_p_both_branches():
    plan = {"weg2-0-1": (1000, 900, None, None), "weg2-0-2": (2000, 1800, None, None),
            "weg2-0-3": (50, 10, None, None), "weg2-0-4": (70, 0, None, None)}
    front = asyncio.run(_legs([
        ("weg2-0-1", True, _pending("weg2-0-1", True)),    # stream, after P
        ("weg2-0-2", False, _pending("weg2-0-2", True)),   # non-stream, after P
        ("weg2-0-3", False, _pending("weg2-0-3", False)),  # pending, P never ran
        ("weg2-0-4", True, None),                          # D-direct, no pending
    ], plan))
    st = _served(front)
    assert st["D"] == {"n": 4, "prompt": 3120, "cached": 2710, "completion": 12}
    assert st["D_after_P"] == {"n": 2, "prompt": 3000, "cached": 2700, "completion": 6}


def test_no_leg_after_p_writes_no_d_after_p_row_and_d_is_unchanged():
    """27B path byte-identical: no hand-off, no detail -> served_tokens = the old keys only."""
    plan = {"weg2-0-1": (100, 40, None, None)}
    front = asyncio.run(_legs([("weg2-0-1", False, None)], plan))
    assert _served(front) == {"D": {"n": 1, "prompt": 100, "cached": 40, "completion": 3}}


# ------------------------------------------------------------------ Feld 2
def test_red_first_cached_tier_from_usage_details_and_sglext_both_branches():
    plan = {"weg2-0-1": (1000, 800, TIERS, "usage"), "weg2-0-2": (1000, 800, TIERS, "sglext"),
            "weg2-0-3": (10, 0, None, None)}
    front = asyncio.run(_legs([
        ("weg2-0-1", True, _pending("weg2-0-1", True)),
        ("weg2-0-2", False, None),
        ("weg2-0-3", False, None),                         # no detail: tiers unchanged
    ], plan))
    st = _served(front)
    assert st["D"]["cached_tier"] == {"device": 1200, "host": 300, "storage": 100}
    assert st["D"]["cached"] == 1600
    assert st["D_after_P"]["cached_tier"] == {"device": 600, "host": 150, "storage": 50}


def test_cached_tier_of_holders_and_absence():
    f = front_mod.cached_tier_of
    assert f({"meta_info": {"cached_tokens_details": {"device": 5, "host": 2}}}) == \
        {"device": 5, "host": 2, "storage": 0}
    assert f({"usage": {"prompt_tokens_details": {"cached_tokens": 3}}}) is None
    assert f({"meta_info": {"cached_tokens": 3}}) is None
    assert f(None) is None
    tail = (_sse({"choices": [{"delta": {"content": "x"}}]})
            + _sse({"usage": {"prompt_tokens": 9}, "sglext": {"cached_tokens_details": {"host": 4}}})
            + b"data: [DONE]\n\n")
    assert front_mod.cached_tier_stream_tail(tail) == {"device": 0, "host": 4, "storage": 0}
    assert front_mod.cached_tier_stream_tail(_sse({"usage": {"prompt_tokens": 9}})) is None
