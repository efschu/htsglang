"""PARK-HANDBACK (user 02.10.: while P is awake every prefill is P's; D after the
flip only decodes, E2 HANDBACK d_compute=0).

METAL y6x (boot_weg2_dkrnfint4bar1dauer10020710_fceeda8493_1002_071105.front.log):
07:19:39,738 'PDFLIP-ROUTE rid=pdflip-12-39 SHORT -> D ... remainder=954', 'D-ADMIT
... source=short'; 07:19:40,324 'PARK-RUNNING epoch=12 ... in_flight_held=
['pdflip-12-39']' -- D held it unstarted over the flip; 07:19:51 'LEG2-FIRST-CONTENT
rid=pdflip-12-39 epoch=14 via=d_direct': D prefilled 954 tokens after the P->D flip.

Fix: a D-prefill leg (no P leg 1) that D holds unstarted at a park and whose
stream is still in the lookahead is handed back -- its leg closes, D's hold is
aborted by name, P's batch takes it.

27B port (02.10.): every 27B park (wait bound, PARK-IMMEDIATE, PARK-SEAT-FREE,
the manual flip park) precedes a D->P flip, so the handback never fights a
D phase; a handed-back rid is not counted parked as well (no double effect).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import aiohttp  # noqa: E402
import pytest  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402

X = 4096
TEXT = "user:" + "u" * 3000


@pytest.fixture
def ros(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv("FLLIPER_PDFLIP_RESUME_VIA_P", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_RESUME_OPEN_STREAM", raising=False)
    return monkeypatch


def _fake_d(state):
    async def chat(request):
        state["posts"] += 1
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        state["held"].set()
        try:
            await state["release"].wait()   # D holds it: nothing streamed
        except asyncio.CancelledError:
            state["closed"] += 1
            raise
        return resp

    async def abort(request):
        state["aborts"].append((await request.json()).get("rid"))
        return web.json_response({"success": True})

    async def info(request):
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/abort_request", abort)
    app.router.add_get("/get_server_info", info)
    return app


async def _held_then_handed_back(park_rids):
    state = {"held": asyncio.Event(), "release": asyncio.Event(), "posts": 0, "closed": 0,
             "aborts": []}
    d = TestServer(_fake_d(state))
    await d.start_server()
    url = str(d.make_url("")).rstrip("/")
    front = F.Front("http://p", url, "D", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    front.groups["D"].url = url
    front.session = aiohttp.ClientSession()
    front.epoch = 12

    async def handler(request):
        payload = await request.json()
        return await front.leg2(request, "pdflip-12-39", payload, TEXT, True, None)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    try:
        async with aiohttp.ClientSession() as cs:
            post = asyncio.ensure_future(cs.post(f"http://{host}:{port}/v1/chat/completions",
                                                 json={"messages": [], "stream": True}))
            await asyncio.wait_for(state["held"].wait(), 10)
            await asyncio.sleep(0.2)                 # the lookahead runs
            handed = front._park_handback(park_rids)  # D's park answer
            for _ in range(60):                      # ROS polls every 0.5 s
                if front.queue:
                    break
                await asyncio.sleep(0.1)
            queued = list(front.queue)
            # end the test's legs: the handed-back one waits for P, the kept one for D
            for q in queued:
                if not q.fut.done():
                    q.fut.set_exception(RuntimeError("test end"))
            state["release"].set()
            await asyncio.wait_for(asyncio.gather(post, return_exceptions=True), 10)
    finally:
        state["release"].set()
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front, handed, queued, state


def test_pdflip_12_39_held_unstarted_at_the_park_goes_to_ps_batch(ros, caplog):
    caplog.set_level(logging.INFO)
    front, handed, queued, state = asyncio.run(_held_then_handed_back(["pdflip-12-39"]))
    assert handed == ["pdflip-12-39"]
    assert len(queued) == 1, "the request joined P's queue"
    p = queued[0]
    assert p.rid == "pdflip-12-39" and not p.d_direct and not p.skip_leg1 and not p.leg1_done
    assert state["aborts"] == ["pdflip-12-39"], "D's hold dropped by name (one path per rid)"
    assert state["posts"] == 1
    assert front.counters["park_handback"] == 1
    assert front.counters["W50_PdFlipTpPrefillExceeded"] == 0, "not an X refusal"
    assert "pdflip-12-39" not in front._x_requeues
    msgs = caplog.messages
    assert any("PDFLIP PARK-HANDBACK rid=pdflip-12-39 epoch=12" in m for m in msgs)
    assert any("PDFLIP P-BATCH-TAKES rid=pdflip-12-39" in m and "where=park_handback" in m for m in msgs)


def test_a_rid_the_park_does_not_name_stays():
    front, handed, queued, state = asyncio.run(_held_then_handed_back(["pdflip-99-1"]))
    assert handed == [] and queued == [] and state["aborts"] == []


def test_only_d_prefill_legs_in_the_lookahead_are_handed_back(caplog):
    caplog.set_level(logging.WARNING)
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.epoch = 3
    f.tp_prefill_max_tokens = X
    f._rvp_state = lambda: None
    f._rvp_requeue, f._front_price = {}, {"a": 200}
    f._leg2_lookahead = {"a", "handoff"}
    f._d_prefill_legs = {"a": 0.0, "committed": 0.0}   # 'handoff' is an after_p leg
    out = f._park_handback(["a", "handoff", "committed", "a"])
    assert out == ["a"]
    assert f._rvp_requeue["a"]["reason"] == F.PARK_HANDBACK_REASON
    assert "handoff" not in f._rvp_requeue and "committed" not in f._rvp_requeue
    assert f.counters["park_handback_committed"] == 1
    assert any("PDFLIP PARK-HANDBACK rid=committed kept=committed" in m for m in caplog.messages)


def test_the_park_answers_held_list_is_read():
    body = json.dumps({"success": True, "parked": ["x"], "held": ["pdflip-12-39", 7], "epoch": 12})
    assert F.park_held_rids(body) == ["pdflip-12-39"]
    assert F.park_held_rids("not json") == []
    assert F.park_held_rids(json.dumps({"parked": []})) == []


def test_the_park_hands_back_held_and_late_rids():
    import inspect

    src = inspect.getsource(F.Front)
    i = src.index("verdict, rids, why = phase_policy.park_verdict(code, text)")
    blk = src[i:i + 6000]
    assert ("_handed = set(Front._park_handback(\n"
            "            self, park_held_rids(text) + [r for r in self._flip_ledger(D) if r not in rids]))") in blk
    # 27B port: a handed-back rid is not also parked (no stale _d_parked hiding its NEW leg 2)
    assert "known = [r for r in known if r not in _handed]" in blk
    assert "late = [r for r in self._flip_ledger(D) if r not in rids and r not in _handed]" in blk
    assert "still = [r for r in self._flip_ledger(D) if r not in _handed]" in blk
    j = src.index("async def _requeue_park_handback(")
    assert '(getattr(self, "_d_parked", None) or {}).pop(rid, None)' in src[j:j + 3000]
