"""H85: the r_D probe's SECOND carrier -- D's prefill clock on /get_server_info
-- for the legs whose body cannot carry ``weg2_prefill_s``.

THE GAP (H84, 890b655dc4 / b89592806a). The front's live X samples r_D as
``uncached / weg2_prefill_s`` and takes the time from the leg-2 BODY only:
``meta_info`` on ``/generate``, ``sglext`` on a NON-streamed OpenAI answer.
A streamed leg 2 never reached the probe at all, and ``/v1/messages`` carries
neither field. The agent fleet streams and speaks Anthropic Messages, so under
agent load the probe had no sample and X stayed at its start X 4096.

THE CARRIER. Group D records each finished prefill (prefill_finished -
forward_entry) with a per-process ``seq`` and publishes the newest as
``internal_states[0].weg2_prefill_s``; the front reads it on the
``/get_server_info`` read it already makes after every leg 2 and ATTRIBUTES it:
newer than the mark the front held at the leg's start, and either the leg's
rid or the ONLY new prefill, and in both cases the leg's own uncached extent.
A value of another request never becomes a sample.

Hermetic, CPU, loopback only: the REAL ``Front.leg2`` (streamed branch behind a
real aiohttp front app, as H61) against a fake D whose ``/get_server_info``
publishes the block shape of ``prefill_clock.snapshot``; the REAL D-side
recorder; the D wiring by AST (the scheduler does not run on CPU).
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2.front import Front
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

START_X = 4096
BOOT = "d-boot-1"


class _FakeD:
    """D's prefill clock as ``prefill_clock.snapshot`` publishes it."""

    def __init__(self, publish: bool = True):
        self.publish = publish
        self.seq = 0
        self.recent: list = []

    def prefill(self, rid: str, prompt: int, cached: int, s: float, stamped: bool = True):
        self.seq += 1
        if stamped:
            self.recent.append({"seq": self.seq, "rid": rid, "s": s,
                                "prompt": prompt, "cached": cached})

    def block(self) -> dict:
        return {"boot": BOOT, "seq": self.seq, "recent": list(self.recent[-32:])}


def _sse(obj, event: str = "") -> bytes:
    head = f"event: {event}\n".encode() if event else b""
    return head + b"data: " + json.dumps(obj).encode() + b"\n\n"


def _fake_d_app(fake: _FakeD, plan: dict) -> web.Application:
    """``plan[front_rid]`` = (d_rid or None, prompt, cached, s, extra_prefills,
    body_extra): what D prefills when that leg arrives. ``d_rid`` None = D
    honours the front's rid; ``extra_prefills`` = other prefills D finishes in
    the same window (a /health_generate, traffic past the front)."""

    def _record(front_rid: str):
        d_rid, prompt, cached, s, extra, _ = plan[front_rid]
        for e in extra:
            fake.prefill(*e)
        if s is not None:
            fake.prefill(d_rid or front_rid, prompt, cached, s)
        else:
            fake.prefill(d_rid or front_rid, prompt, cached, 1.0, stamped=False)
        return prompt, cached

    async def chat(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        rid = payload["rid"]
        prompt, cached = _record(rid)
        usage = {"prompt_tokens": prompt, "completion_tokens": 3,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if not payload.get("stream"):
            body = {"id": rid, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}], "usage": usage}
            body.update(plan[rid][5])
            return web.json_response(body)
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        await resp.write(_sse({"id": rid, "choices": [{"index": 0, "delta": {"content": "ok"},
                                                       "finish_reason": None}]}))
        await resp.write(_sse({"id": rid, "choices": [{"index": 0, "delta": {},
                                                       "finish_reason": "stop"}]}))
        await resp.write(_sse({"id": rid, "choices": [], "usage": usage}))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def messages(request: web.Request) -> web.StreamResponse:
        # D's Anthropic adapter drops the undeclared `rid`: D prefills under
        # its OWN rid. message_start ships at once with input_tokens 0, the
        # closing message_delta carries input (= prompt - cached) and cache.
        payload = await request.json()
        prompt, cached = _record(payload["rid"])
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        await resp.write(_sse({"type": "message_start", "message": {
            "id": "msg_x", "type": "message", "role": "assistant", "content": [],
            "usage": {"input_tokens": 0, "output_tokens": 0}}}, "message_start"))
        await resp.write(_sse({"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}},
                              "content_block_start"))
        await resp.write(_sse({"type": "content_block_delta", "index": 0,
                               "delta": {"type": "text_delta", "text": "ok"}},
                              "content_block_delta"))
        await resp.write(_sse({"type": "content_block_stop", "index": 0}, "content_block_stop"))
        await resp.write(_sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                               "usage": {"input_tokens": prompt - cached, "output_tokens": 3,
                                         "cache_read_input_tokens": cached}}, "message_delta"))
        await resp.write(_sse({"type": "message_stop"}, "message_stop"))
        await resp.write_eof()
        return resp

    async def info(request: web.Request) -> web.Response:
        st = {"weg2_prefill_s": fake.block()} if fake.publish else {}
        return web.json_response({"internal_states": [st]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/messages", messages)
    app.router.add_get("/get_server_info", info)
    return app


def _front_app(front: Front) -> web.Application:
    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        rid = request.headers["x-rid"]
        payload["rid"] = rid  # what handle_generate does (#1442)
        return await front.leg2(request, rid, payload, "q", bool(payload.get("stream")), None)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_post("/v1/messages", handler)
    return app


async def _legs(legs, plan: dict, publish: bool = True) -> Front:
    """Serve ``legs`` = [(front_rid, path, stream)] one after another (each
    alone on D) through the REAL ``Front.leg2``."""
    fake = _FakeD(publish)
    d = TestServer(_fake_d_app(fake, plan))
    await d.start_server()
    front = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=262144, tp_prefill_max_tokens=START_X)
    front.session = aiohttp.ClientSession()
    runner = web.AppRunner(_front_app(front))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    try:
        async with aiohttp.ClientSession() as cs:
            for rid, path, stream in legs:
                body = {"model": "m", "max_tokens": 8, "stream": stream,
                        "messages": [{"role": "user", "content": "q"}]}
                async with cs.post(f"http://{host}:{port}{path}", json=body,
                                   headers={"x-rid": rid}) as resp:
                    await resp.read()
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front


def _rd_lines(caplog) -> list:
    return [r.getMessage() for r in caplog.records if "WEG2 X R_D " in r.getMessage()]


def _plan(**legs) -> dict:
    """rid -> (d_rid, prompt, cached, s, extra_prefills, body_extra)."""
    out = {}
    for rid, spec in legs.items():
        spec = tuple(spec)
        spec += ((), {})[len(spec) - 4:]  # defaults: extra_prefills=(), body_extra={}
        out[rid.replace("_", "-")] = spec
    return out


# --------------------------------------------------------- (a) the new carrier
def test_red_first_a_streamed_openai_leg_samples_ds_server_info_time(caplog):
    """The streamed leg carries no weg2_prefill_s in its body. D honours the
    front's rid; its record (4100 uncached in 4.0 s) is attributed by rid and
    is NEWER than the mark the first leg's read left. On b89592806a the
    streamed branch never reached the probe: no sample at all."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    plan = _plan(weg2_0_1=(None, 3000, 0, 2.5), weg2_0_2=(None, 4200, 100, 4.0))
    f = asyncio.run(_legs([("weg2-0-1", "/v1/chat/completions", True),
                           ("weg2-0-2", "/v1/chat/completions", True)], plan))
    assert list(f._x_samples["r_d"]) == [pytest.approx(4100 / 4.0)]
    lines = _rd_lines(caplog)
    assert any("rid=weg2-0-2" in m and "d_prefill_src=server_info" in m
               and "d_prefill_attr=rid" in m for m in lines), lines
    # The first leg of a boot has no mark to attribute against: no sample.
    assert any("rid=weg2-0-1" in m and "d_prefill_attr=no_mark" in m for m in lines), lines


def test_red_first_a_messages_leg_samples_the_only_new_prefill(caplog):
    """/v1/messages: D runs the request under its OWN rid, so no rid matches;
    the record is this leg's because it is the only prefill D finished since
    the mark and its extent (6000 - 1000) is the leg's own (input_tokens 5000
    on the Anthropic wire)."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    plan = _plan(weg2_0_1=("d-own-a", 2500, 0, 2.0), weg2_0_2=("d-own-b", 6000, 1000, 5.0))
    f = asyncio.run(_legs([("weg2-0-1", "/v1/messages", True),
                           ("weg2-0-2", "/v1/messages", True)], plan))
    assert list(f._x_samples["r_d"]) == [pytest.approx(5000 / 5.0)]
    assert any("rid=weg2-0-2" in m and "d_prefill_src=server_info" in m
               and "d_prefill_attr=sole_new" in m for m in _rd_lines(caplog))


def test_red_first_without_body_field_and_server_info_no_sample_by_name():
    """A D that publishes nothing: both streamed legs reach the probe and are
    refused by name -- never the wall (on b89592806a they never reached it)."""
    plan = _plan(weg2_0_1=(None, 3000, 0, 2.5), weg2_0_2=(None, 4200, 100, 4.0))
    f = asyncio.run(_legs([("weg2-0-1", "/v1/chat/completions", True),
                           ("weg2-0-2", "/v1/chat/completions", True)], plan, publish=False))
    assert len(f._x_samples["r_d"]) == 0
    assert f.counters["r_d_skipped_no_prefill_time"] == 2


# ----------------------------------- (b) a value of another request never counts
def test_a_second_new_prefill_makes_a_messages_leg_ambiguous(caplog):
    """Another prefill finished on D inside the leg (a /health_generate, 1
    token) AND a record with the leg's exact extent exists: with no rid to
    match, two new prefills are no attribution -- no sample."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    plan = _plan(weg2_0_1=("d-own-a", 2500, 0, 2.0),
                 weg2_0_2=("d-own-b", 6000, 1000, 5.0, (("HEALTH_CHECK_1", 1, 0, 0.05),)))
    f = asyncio.run(_legs([("weg2-0-1", "/v1/messages", True),
                           ("weg2-0-2", "/v1/messages", True)], plan))
    assert len(f._x_samples["r_d"]) == 0
    assert any("rid=weg2-0-2" in m and "ambiguous(new=2)" in m for m in _rd_lines(caplog))


def test_an_older_record_of_the_same_rid_is_no_sample():
    """A re-route re-sends the SAME rid. Its first prefill (2.5 s) is on D's
    ring; the second finished without both stamps. The rid still matches the
    OLD record -- it is not newer than the mark, so it is no sample."""
    plan = _plan(weg2_0_1=(None, 4200, 0, 2.5))
    f = asyncio.run(_legs([("weg2-0-1", "/v1/chat/completions", True)], plan))
    assert len(f._x_samples["r_d"]) == 0
    plan2 = _plan(weg2_0_1=(None, 4200, 0, None))
    fake_mark = f._d_prefill_mark
    assert fake_mark == (BOOT, 1)

    async def again() -> Front:
        fake = _FakeD()
        fake.prefill("weg2-0-1", 4200, 0, 2.5)          # the first attempt, seq 1
        d = TestServer(_fake_d_app(fake, plan2))
        await d.start_server()
        g = Front("http://p", str(d.make_url("")).rstrip("/"), "D", "t", "", 0, 0, {}, 45.0,
                  carrier_max_tokens=262144, tp_prefill_max_tokens=START_X)
        g._d_prefill_mark = fake_mark
        g.session = aiohttp.ClientSession()
        try:
            req = SimpleNamespace(path="/v1/chat/completions")
            await g.leg2(req, "weg2-0-1", {"rid": "weg2-0-1", "messages": []}, "q", False, None)
        finally:
            await g.session.close()
            await d.close()
        return g

    g = asyncio.run(again())
    assert len(g._x_samples["r_d"]) == 0
    assert g.counters["r_d_skipped_no_prefill_time"] == 1


def test_the_body_wins_over_server_info(caplog):
    """A non-streamed OpenAI leg carries sglext.weg2_prefill_s (H84): that is
    the sample, whatever D's ring says."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    plan = _plan(weg2_0_1=(None, 3000, 0, 2.5),
                 weg2_0_2=(None, 4100, 0, 9.0, (), {"sglext": {"weg2_prefill_s": 4.1}}))
    f = asyncio.run(_legs([("weg2-0-1", "/v1/chat/completions", True),
                           ("weg2-0-2", "/v1/chat/completions", False)], plan))
    assert list(f._x_samples["r_d"]) == [pytest.approx(1000.0)]
    assert any("rid=weg2-0-2" in m and "d_prefill_src=body" in m for m in _rd_lines(caplog))


@pytest.mark.parametrize("block,rid,uncached,mark,want", [
    # the leg's rid, newer than the mark, its extent -> its time
    ({"boot": "b", "seq": 3, "recent": [{"seq": 3, "rid": "r", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, ("b", 2), (2.0, "rid")),
    # no rid match, the ONLY new prefill, its extent -> its time
    ({"boot": "b", "seq": 3, "recent": [{"seq": 3, "rid": "d", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, ("b", 2), (2.0, "sole_new")),
    # no mark yet (first leg of a boot)
    ({"boot": "b", "seq": 3, "recent": [{"seq": 3, "rid": "r", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, None, (None, "no_mark")),
    # D restarted under the front: its seq restarted too
    ({"boot": "b2", "seq": 1, "recent": [{"seq": 1, "rid": "d", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, ("b", 7), (None, "d_restarted")),
    # the rid's record is not newer than the mark (re-route re-sent the rid)
    ({"boot": "b", "seq": 2, "recent": [{"seq": 2, "rid": "r", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, ("b", 2), (None, "absent")),
    # a new prefill was finished, but without a record -> not another's value
    ({"boot": "b", "seq": 3, "recent": [{"seq": 2, "rid": "d", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, ("b", 2), (None, "absent")),
    # two new prefills beyond the ring's reach count too (evicted records)
    ({"boot": "b", "seq": 40, "recent": [{"seq": 40, "rid": "d", "s": 2.0, "prompt": 5000, "cached": 1000}]},
     "r", 4000, ("b", 2), (None, "ambiguous(new=38)")),
    # the right rid, another extent
    ({"boot": "b", "seq": 3, "recent": [{"seq": 3, "rid": "r", "s": 2.0, "prompt": 5000, "cached": 0}]},
     "r", 4000, ("b", 2), (None, "extent_mismatch(d=5000,leg=4000)")),
])
def test_attribution_refuses_every_record_that_is_not_provably_the_legs(block, rid, uncached, mark, want):
    from sglang.srt.weg2 import prefill_clock

    assert prefill_clock.attribute(block, rid, uncached, mark) == want


# ------------------------------------------------------------- (c) D's side
def test_d_records_on_a_weg2_d_only_and_every_prefill_takes_a_seq():
    """Off the Weg-2 D group nothing moves; on it every finished prefill takes
    a seq -- an unstamped one too, so the reader sees it as a second prefill
    instead of mistaking the other one for this leg's."""
    from sglang.srt.weg2 import prefill_clock as pc

    pc._reset_for_tests()

    def req(rid, fe, pf, n=5000, cached=1000):
        return SimpleNamespace(rid=rid, origin_input_ids=[0] * n, cached_tokens=cached,
                               time_stats=SimpleNamespace(forward_entry_time=fe,
                                                          prefill_finished_time=pf))

    pc.note_prefill_finished(req("p", 10.0, 12.0), SimpleNamespace(tp_prefill_max_tokens=0))
    assert pc.snapshot()["seq"] == 0 and pc.snapshot()["recent"] == []
    d = SimpleNamespace(tp_prefill_max_tokens=START_X)
    pc.note_prefill_finished(req("a", 10.0, 12.5), d)
    pc.note_prefill_finished(req("b", 0.0, 13.0), d)          # forward entry never stamped
    snap = pc.snapshot()
    assert snap["seq"] == 2 and snap["boot"] == pc.BOOT
    assert snap["recent"] == [{"seq": 1, "rid": "a", "s": 2.5, "prompt": 5000, "cached": 1000}]
    for i in range(40):
        pc.note_prefill_finished(req(f"x{i}", 1.0, 2.0), d)
    assert len(pc.snapshot()["recent"]) == pc.RING_MAX
    pc._reset_for_tests()


def _src(mod: str) -> ast.Module:
    return ast.parse(Path(importlib.util.find_spec(mod).origin).read_text())


def _func(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    raise AssertionError(f"{cls}.{name} not found")


def test_the_prefill_result_path_records_and_server_info_publishes():
    """The wiring the scheduler cannot show on CPU: the recorder is called
    right after ``set_prefill_finished_time`` of the generation prefill path,
    with the scheduler's server_args; ``get_internal_state`` publishes the
    snapshot under the key the front reads, only when armed."""
    from sglang.srt.weg2 import prefill_clock as pc

    fn = _func(_src("sglang.srt.managers.scheduler_components.batch_result_processor"),
               "SchedulerBatchResultProcessor", "process_batch_result_prefill")
    found = False
    for node in ast.walk(fn):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for a, b in zip(body, body[1:]):
            if (isinstance(a, ast.Expr) and "set_prefill_finished_time" in ast.unparse(a)
                    and isinstance(b, ast.Expr)
                    and "note_prefill_finished(req, self.server_args)" in ast.unparse(b)):
                found = True
    assert found, "note_prefill_finished must follow set_prefill_finished_time"

    gis = ast.unparse(_func(_src("sglang.srt.managers.scheduler"), "Scheduler",
                            "get_internal_state"))
    assert "_weg2_prefill_clock.armed(self.server_args)" in gis
    assert ("ret[_weg2_prefill_clock.INTERNAL_STATE_KEY] = _weg2_prefill_clock.snapshot()"
            in gis)
    assert pc.INTERNAL_STATE_KEY == "weg2_prefill_s"
