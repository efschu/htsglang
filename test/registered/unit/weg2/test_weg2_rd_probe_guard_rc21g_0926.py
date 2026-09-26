"""rc2.1g Pick bb086e1120 (27B Review V RC7b) auf der NF-Linie: ein Fehler in
der r_D-Probe macht aus einem BEDIENTEN leg 2 nie einen 503.

27B: ``_sample_r_d`` war nur am gestreamten Aufruf per try/except bewacht; auf
dem nicht gestreamten Zweig entkam eine Ausnahme, nachdem D 200 geantwortet
hatte. NF hat kein ``_sample_r_d`` (RC7-X/H85 nicht auf der Linie): H84s Probe
steht inline im nicht gestreamten Zweig von ``Front.leg2`` und war dort gar
nicht bewacht -- dieselbe Klasse. Jetzt gezaehlt (``r_d_probe_errors``) und
geloggt (``WEG2 X R_D-PROBE-ERROR``), nie weitergereicht.

Hermetisch, CPU, nur Loopback: die ECHTE ``Front.leg2`` gegen ein falsches D
(OpenAI-Draht, wie test_weg2_x_live_ceiling_h84._serve_once).
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front

PT, PFS = 4096, 2.0          # 4096 uncached in 2,0 s -> r_D 2048


async def _serve_once():
    async def chat(request: web.Request) -> web.Response:
        return web.json_response({
            "id": "x", "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
            "usage": {"prompt_tokens": PT, "completion_tokens": 8,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "sglext": {"weg2_prefill_s": PFS}})

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/get_server_info", info)
    d = TestServer(app)
    await d.start_server()
    f = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
              carrier_max_tokens=262144, tp_prefill_max_tokens=4096)
    f.session = aiohttp.ClientSession()
    try:
        req = SimpleNamespace(path="/v1/chat/completions")
        resp = await f.leg2(req, "weg2-0-1", {"messages": [{"role": "user", "content": "q"}]},
                            "q", False, None)
    finally:
        await f.session.close()
        await d.close()
    return f, resp


def _boom(*_a, **_k):
    raise RuntimeError("probe exploded")


def test_red_first_a_probe_error_leaves_the_served_leg_at_200(caplog):
    caplog.set_level(logging.WARNING, logger="weg2.front")
    with envs.SGLANG_WEG2_X_RD_MIN_UNCACHED.override(512), \
            mock.patch.object(front_mod, "r_d_probe", _boom):
        f, resp = asyncio.run(_serve_once())
    assert resp.status == 200
    assert f.counters["r_d_probe_errors"] == 1
    assert len(f._x_samples["r_d"]) == 0
    assert any("WEG2 X R_D-PROBE-ERROR rid=weg2-0-1 RuntimeError" in r.getMessage()
               for r in caplog.records)


def test_without_an_error_the_sample_is_taken_as_before():
    with envs.SGLANG_WEG2_X_RD_MIN_UNCACHED.override(512):
        f, resp = asyncio.run(_serve_once())
    assert resp.status == 200
    assert f.counters["r_d_probe_errors"] == 0
    assert list(f._x_samples["r_d"]) == [PT / PFS]
