"""W3 Weg2DrainWitnessUnreachable (29.09., NF z30w-park epoch 113).

The drain's progress witness (#1317c, ``GET {group}/get_server_info``) ran on
the shared session's 3600 s timeout only. On a dead P ring the drain hung inside
it and the boot ended in a blind DEADMAN_FLIP_STALL instead of a named stop.
Pinned here: a server-info endpoint that never answers ends the drain within the
read's own bound as a NAMED W3 carrying the group -- no retry, no W1 streak."""
from __future__ import annotations

import asyncio
import collections
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import ClientSession, web  # noqa: E402

from sglang.srt.weg2 import front as F  # noqa: E402


async def _hanging_group(answer: bool):
    """A group whose /get_server_info never answers (answer=False) or answers."""
    async def info(_request):
        if not answer:
            await asyncio.sleep(5.0)  # far past the patched bound; short so cleanup is quick
        return web.json_response({"internal_states": [{"weg2_decode_progress": {"tokens": 7}}]})

    app = web.Application()
    app.router.add_get("/get_server_info", info)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _front(session, stops):
    ns = types.SimpleNamespace(session=session, counters=collections.Counter(), drain_deadline_s=0.05,
                               _drain_progress=None, state="flipping")
    ns._weg2_decode_progress = lambda g: F.Front._weg2_decode_progress(ns, g)
    ns._stop_progress_unreachable = lambda e: F.Front._stop_progress_unreachable(ns, e)
    ns._flip_ledger = lambda g: g.outstanding

    def do_stop(name, detail):
        stops.append((name, detail))
        ns.state = "STOP"

    ns.do_stop = do_stop
    return ns


def test_hanging_server_info_is_a_named_w3_within_the_bound(monkeypatch):
    monkeypatch.setattr(F, "PROGRESS_READ_TIMEOUT_S", 0.3)

    async def run():
        runner, url = await _hanging_group(answer=False)
        stops = []
        try:
            async with ClientSession() as session:
                ns = _front(session, stops)
                g = types.SimpleNamespace(name="P", url=url, outstanding={"weg2-113-7": object()})
                t0 = time.monotonic()
                ok = await F.Front.drain(ns, g)
                return ok, time.monotonic() - t0, stops, ns
        finally:
            await runner.cleanup()

    ok, took, stops, ns = asyncio.run(run())
    assert ok is False and took < 0.3 + 1.0                 # the read's bound, not the 3600 s session
    assert len(stops) == 1                                  # one stop, no retry
    name, detail = stops[0]
    assert name == "W3 Weg2DrainWitnessUnreachable"
    assert "group P" in detail and "/get_server_info" in detail
    assert ns.counters["W3_progress_unreachable"] == 1


def test_answering_server_info_keeps_the_drain_witness(monkeypatch):
    """Negative branch: a group that answers is read as before (no stop)."""
    monkeypatch.setattr(F, "PROGRESS_READ_TIMEOUT_S", 0.3)

    async def run():
        runner, url = await _hanging_group(answer=True)
        stops = []
        try:
            async with ClientSession() as session:
                ns = _front(session, stops)
                g = types.SimpleNamespace(name="D", url=url, outstanding={})
                return await F.Front.drain(ns, g), stops
        finally:
            await runner.cleanup()

    ok, stops = asyncio.run(run())
    assert ok is True and stops == []


def test_the_bound_covers_one_measured_p_chunk():
    """The read is answered between two forward passes; the longest measured
    is a P 16k chunk on PP0 (~5.6 s, x135). A bound below it would stop a
    healthy P mid-chunk by name."""
    assert F.PROGRESS_READ_TIMEOUT_S >= 2 * 5.6
