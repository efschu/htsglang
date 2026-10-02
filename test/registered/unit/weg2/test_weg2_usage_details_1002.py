# SPDX-License-Identifier: Apache-2.0
"""USAGE-DETAILS (02.10., second order): ``usage.total_tokens_details``.

Every client answer's final usage (OpenAI body / the stream's ``choices: []``
usage chunk, Anthropic ``usage`` / ``message_delta``) carries one additive
object: flips, the decode's interruptions (``sleep`` / ``sleep_s`` /
``sleep_causes``), queue / TTFT / decode times, P and D computed tokens, the
route -- and the tier split / reasoning count where a rank reported them.

Pinned here:
* gap attribution (pure): a gap counts only above max(4 x median, 0.25 s) and
  only with a known cause (flip > park > prefill_d); a long gap without one is
  returned as unnamed and logged ``WEG2-USAGE-GAP-UNNAMED``;
* the zero object of a D-only request;
* both API formats, body and stream, through the front's real leg 1 / leg 2;
* a real stream gap under a flip / park / D prefill window, and an unnamed one.

Hermetic: fake P and D on loopback, no CUDA, no model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import usage_true as UT  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TEXT = "a flipped agent turn"
RID = "weg2-2-1"
PROMPT, D_CACHED, P_CACHED, OUT = 20000, 19990, 4000, 7
P_TIERS = {"device": 1000, "host": 2000, "storage": 1000}  # sums to P_CACHED
_ORIG_SR = web.StreamResponse


# ------------------------------------------------------------ pure: gaps --


def _clock(times):
    c = UT.TokenClock()
    for t in times:
        c.note(t)
    return c


def _steady(t0, n, dt=0.02):
    return [t0 + i * dt for i in range(n)]


def test_gap_under_a_flip_window_counts_as_flip():
    ts = _steady(100.0, 20) + _steady(102.0, 20)  # one 1.62 s gap at ~100.38
    n, s, causes, unnamed = UT.attribute_gaps(_clock(ts), [(100.5, 101.5)], [], [])
    assert (n, causes, unnamed) == (1, {"flip": 1, "prefill_d": 0, "park": 0}, [])
    assert abs(s - (102.0 - 100.38)) < 1e-6


def test_gap_under_a_park_window_counts_as_park():
    ts = _steady(100.0, 20) + _steady(101.0, 20)
    n, s, causes, unnamed = UT.attribute_gaps(_clock(ts), [], [(100.4, 100.9)], [])
    assert (n, causes["park"], unnamed) == (1, 1, [])


def test_gap_under_another_requests_d_prefill_counts_as_prefill_d():
    ts = _steady(100.0, 20) + _steady(100.9, 20)
    n, _s, causes, unnamed = UT.attribute_gaps(_clock(ts), [], [], [(100.4, 100.8)])
    assert (n, causes["prefill_d"], unnamed) == (1, 1, [])


def test_prefill_d_gap_counts_only_the_neighbour_prefill_overlap():
    # y8a weg2-20-79 (02.10. 18:35:50-18:36:40): D decoded 2741 tokens while the
    # stream stayed silent 49.8 s (tool call built); two neighbour D prefills
    # (weg2-20-80 2.37 s, weg2-20-81 4.09 s) fell into it. Booked was 49.8 s of
    # prefill_d sleep and decode_s 0.59 -- the sleep is the prefills' 6.46 s.
    ts = _steady(0.0, 20) + _steady(50.2, 20)
    adm = [(10.42, 12.79), (17.46, 21.55), (11.0, 12.0)]  # the last one nests
    n, s, causes, unnamed = UT.attribute_gaps(_clock(ts), [], [], adm)
    assert (n, causes, unnamed) == (1, {"flip": 0, "prefill_d": 1, "park": 0}, [])
    assert abs(s - (2.37 + 4.09)) < 1e-6


def test_prefill_d_overlap_is_clipped_to_the_gap():
    ts = _steady(100.0, 20) + _steady(101.0, 20)  # gap 100.38 -> 101.0
    n, s, causes, _u = UT.attribute_gaps(_clock(ts), [], [], [(99.0, 100.5)])
    assert (n, causes["prefill_d"]) == (1, 1)
    assert abs(s - (100.5 - 100.38)) < 1e-6


def test_flip_takes_precedence_over_park_and_prefill():
    ts = _steady(100.0, 20) + _steady(101.0, 20)
    _n, _s, causes, _u = UT.attribute_gaps(_clock(ts), [(100.5, 100.6)], [(100.4, 100.9)],
                                           [(100.4, 100.8)])
    assert causes == {"flip": 1, "prefill_d": 0, "park": 0}


def test_unnamed_gap_is_reported_not_counted():
    ts = _steady(100.0, 20) + _steady(100.9, 20)
    n, s, causes, unnamed = UT.attribute_gaps(_clock(ts), [(200.0, 201.0)], [], [])
    assert (n, s) == (0, 0.0) and sum(causes.values()) == 0
    assert unnamed == [520]


def test_gap_below_four_medians_never_counts():
    # median gap 0.2 s -> threshold 0.8 s; a 0.6 s gap under a flip is no sleep
    ts = [100.0 + 0.2 * i for i in range(20)]
    ts += [ts[-1] + 0.6 + 0.2 * i for i in range(20)]
    n, _s, _c, unnamed = UT.attribute_gaps(_clock(ts), [(0.0, 1e9)], [], [])
    assert n == 0 and unnamed == []


def test_body_windows_count_flips_and_parks_overlapping_the_leg():
    n, s, causes = UT.attribute_windows(10.0, 20.0, [(5.0, 12.0), (30.0, 31.0)], [(15.0, 16.0)])
    assert (n, causes) == (2, {"flip": 1, "prefill_d": 0, "park": 1})
    assert abs(s - 3.0) < 1e-9


# ---------------------------------------------------------------- wires --


def _oa_usage(prompt=PROMPT, cached=D_CACHED, out=OUT):
    return {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out,
            "prompt_tokens_details": {"cached_tokens": cached}}


def _oa_body(prompt=PROMPT, cached=D_CACHED, tiers=None) -> bytes:
    u = _oa_usage(prompt, cached)
    body = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}], "usage": u}
    if tiers is not None:
        body["sglext"] = {"cached_tokens_details": tiers}
    return json.dumps(body).encode()


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def _oa_stream(n=3):
    out = [_sse({"choices": [{"index": 0, "delta": {"content": f"t{i}"}, "finish_reason": None}]})
           for i in range(n)]
    out.append(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    out.append(_sse({"choices": [], "usage": _oa_usage()}))
    out.append(b"data: [DONE]\n\n")
    return out


def _anth_body() -> bytes:
    return json.dumps({"id": "msg", "type": "message", "role": "assistant", "model": "m",
                       "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                       "stop_sequence": None,
                       "usage": {"input_tokens": PROMPT - D_CACHED,
                                 "cache_read_input_tokens": D_CACHED, "output_tokens": OUT}}).encode()


class _FakeChat:
    def __init__(self, lines):
        self.lines = lines
        self.tokenizer_manager = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None))

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        async def _gen():
            for line in self.lines:
                yield line
        return _gen()

    def apply_reasoning_enabled(self, *a, **kw):
        return None


def _chunk(choices, usage):
    return "data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                  "model": "m", "choices": choices, "usage": usage}) + "\n\n"


def _anth_stream():
    usage = _oa_usage()
    lines = [_chunk([{"index": 0, "delta": {"role": "assistant", "content": f"t{i}"},
                      "finish_reason": None}], usage) for i in range(3)]
    lines.append(_chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}], usage))
    lines.append(_chunk([], usage))
    lines.append("data: [DONE]\n\n")
    serving = AnthropicServing(_FakeChat(lines))
    req = AnthropicMessagesRequest(model="m", max_tokens=64, stream=True,
                                   messages=[{"role": "user", "content": TEXT}])

    async def _collect():
        out = []
        async for sse in serving._generate_anthropic_stream(
            adapted_request=object(), processed_request=object(),
            anthropic_request=req, raw_request=object(),
        ):
            out.append(sse)
        return "".join(out).encode()

    loop = asyncio.new_event_loop()
    try:
        blob = loop.run_until_complete(_collect())
    finally:
        loop.close()
    return [part + b"\n\n" for part in blob.split(b"\n\n") if part.strip()]


# ---------------------------------------------------------------- fakes --


def _fake_group(path, answer, hooks=None):
    """``answer``: a body (bytes) or SSE chunks; ``hooks[i]`` = an async
    callable run after chunk ``i`` (a pause with its cause)."""

    async def handler(request):
        await request.read()
        if isinstance(answer, bytes):
            return web.Response(body=answer, content_type="application/json")
        resp = _ORIG_SR(status=200)
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        for i, chunk in enumerate(answer):
            await resp.write(chunk)
            if hooks:
                # one token per read, as D streams it (back-to-back writes
                # coalesce into one read, and one read is one gap sample)
                await asyncio.sleep(0.02)
                if i in hooks:
                    await hooks[i]()
        await resp.write_eof()
        return resp

    async def info(request):
        return web.json_response({})

    app = web.Application()
    app.router.add_post(path, handler)
    app.router.add_get("/get_server_info", info)
    return app


def _p_body(path, prompt, cached):
    if path == "/v1/messages":
        return json.dumps({"id": "m", "type": "message", "role": "assistant", "model": "m",
                           "content": [{"type": "text", "text": "x"}], "stop_reason": "max_tokens",
                           "usage": {"input_tokens": prompt - cached, "cache_read_input_tokens": cached,
                                     "output_tokens": 1},
                           "sglext": {"cached_tokens_details": P_TIERS}}).encode()
    return _oa_body(prompt, cached, tiers=P_TIERS)


async def _run(path, d_answer, stream, p_legs=((PROMPT, P_CACHED),), hooks=None, before_leg2=None,
               route=None):
    front = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                    carrier_max_tokens=27466, tp_prefill_max_tokens=4096)
    front.session = aiohttp.ClientSession()
    book = F.Front._req_book(front)
    book.arrive(RID, time.time(), front.epoch)
    if route is not None and hasattr(front, "_usage_route_note"):
        front._usage_route_note(RID, route, front.epoch)
    servers = []
    pending = None
    try:
        await asyncio.sleep(0.01)  # a measurable queue
        for prompt, cached in p_legs:
            p_srv = TestServer(_fake_group(path, _p_body(path, prompt, cached)))
            await p_srv.start_server()
            servers.append(p_srv)
            front.groups["P"].url = str(p_srv.make_url("")).rstrip("/")
            body = {"model": "m", "max_tokens": 64, "stream": stream,
                    "messages": [{"role": "user", "content": TEXT}]}
            pending = F.Pending(RID, path, body, TEXT, time.time(),
                                asyncio.get_event_loop().create_future(), est_prompt=PROMPT)
            await front.leg1(pending)
        if before_leg2 is not None:
            before_leg2(front)
        d_srv = TestServer(_fake_group(path, d_answer, hooks))
        await d_srv.start_server()
        servers.append(d_srv)
        front.groups["D"].url = str(d_srv.make_url("")).rstrip("/")

        async def handler(request):
            payload = await request.json()
            return await front.leg2(request, RID, payload, TEXT, stream, pending)

        app = web.Application()
        app.router.add_post(path, handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0][:2]
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as cs:
                body = {"model": "m", "max_tokens": 64, "stream": stream,
                        "messages": [{"role": "user", "content": TEXT}]}
                async with cs.post(f"http://{host}:{port}{path}", json=body) as resp:
                    got = await resp.read()
        finally:
            await runner.cleanup()
    finally:
        await front.session.close()
        for s in servers:
            await s.close()
    return front, got


def _objs(got):
    return [json.loads(line[5:]) for line in got.split(b"\n")
            if line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]")]


def _lines(caplog, mark):
    return [r.getMessage() for r in caplog.records if mark in r.getMessage()]


ZERO_CAUSES = {"flip": 0, "prefill_d": 0, "park": 0}


# ------------------------------------------------------------ D-only zero --


def test_d_only_request_gets_the_zero_object(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False, p_legs=(), route="short"))
    u = json.loads(got)["usage"]
    d = u["total_tokens_details"]
    assert d["flips"] == 0 and d["sleep"] == 0 and d["sleep_s"] == 0.0
    assert d["sleep_causes"] == ZERO_CAUSES
    assert d["p_computed"] == 0 and d["d_computed"] == PROMPT - D_CACHED
    assert d["route"] == "short" and d["route_epoch"] == 0
    # the standard keys are D's, untouched (no P leg: no cached fix)
    assert u["prompt_tokens_details"]["cached_tokens"] == D_CACHED
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (PROMPT, OUT, PROMPT + OUT)
    assert "ttft_s" not in d  # no token stream on a body: never guessed
    assert len(_lines(caplog, "WEG2-USAGE-DETAILS")) == 1


def test_d_only_stream_gets_the_zero_object():
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_stream(), True, p_legs=()))
    finals = [o for o in _objs(got) if o.get("usage")]
    assert len(finals) == 1
    d = finals[0]["usage"]["total_tokens_details"]
    assert (d["flips"], d["sleep"], d["sleep_causes"], d["p_computed"]) == (0, 0, ZERO_CAUSES, 0)
    assert "ttft_s" in d and "decode_s" in d
    # the content chunks pass byte for byte
    assert all(b"total_tokens_details" not in c for c in _oa_stream())
    assert got.startswith(b"".join(_oa_stream()[:4]))


# ------------------------------------------------- both formats, flipped --


def _flip_between(front):
    fp = F.Front._flip_phase(front)
    t = time.time()
    fp.vorlauf("P>D", "test", t)
    fp.layer("P>D", t, "test")
    fp.done(t + 0.001)


def test_openai_body_flipped_carries_the_details(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False, before_leg2=_flip_between))
    u = json.loads(got)["usage"]
    d = u["total_tokens_details"]
    assert d["flips"] == 1
    assert d["p_computed"] == PROMPT - P_CACHED and d["d_computed"] == PROMPT - D_CACHED
    assert d["route"] == "long"
    assert d["queue_s"] > 0
    # the tier split P reported for the prefill it computed (where its hits came from)
    ptd = u["prompt_tokens_details"]
    assert (ptd["cached_device"], ptd["cached_l2"], ptd["cached_l3"]) == (1000, 2000, 990)
    assert ptd["cached_tokens"] == 3990  # the USAGE-TRUE fix still holds
    m = _lines(caplog, "WEG2-USAGE-DETAILS")
    assert len(m) == 1 and "flips=1" in m[0] and "p_computed=16000" in m[0]


def test_openai_stream_flipped_final_usage_chunk_carries_the_details():
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_stream(), True, before_leg2=_flip_between))
    objs = _objs(got)
    assert [o for o in objs if o.get("usage") and o["choices"]][:1] == []
    final = [o for o in objs if o.get("usage")][-1]
    d = final["usage"]["total_tokens_details"]
    assert d["flips"] == 1 and d["p_computed"] == PROMPT - P_CACHED
    assert d["sleep"] == 0 and d["sleep_causes"] == ZERO_CAUSES
    assert d["ttft_s"] >= 0 and d["decode_s"] >= 0
    assert final["usage"]["prompt_tokens_details"]["cached_l3"] == 990


def test_anthropic_body_flipped_carries_the_details():
    _f, got = asyncio.run(_run("/v1/messages", _anth_body(), False, before_leg2=_flip_between))
    u = json.loads(got)["usage"]
    d = u["total_tokens_details"]
    assert d["flips"] == 1 and d["p_computed"] == PROMPT - P_CACHED
    # no standard object on this wire: the split rides in the detail object
    assert (d["cached_device"], d["cached_l2"], d["cached_l3"]) == (1000, 2000, 990)
    assert "prompt_tokens_details" not in u
    assert u["cache_read_input_tokens"] == 3990


def test_anthropic_stream_message_delta_carries_the_details():
    frames = _anth_stream()
    _f, got = asyncio.run(_run("/v1/messages", frames, True, before_leg2=_flip_between))
    objs = _objs(got)
    deltas = [o for o in objs if o.get("type") == "message_delta"]
    assert len(deltas) == 1
    d = deltas[0]["usage"]["total_tokens_details"]
    assert d["flips"] == 1 and d["p_computed"] == PROMPT - P_CACHED
    assert "total_tokens_details" not in json.dumps([o for o in objs if o.get("type") != "message_delta"])


# ------------------------------------------- a real gap and its cause ----


def _gap_run(cause):
    """A D stream that stops 0.6 s after its 6th token while ``cause`` holds."""
    chunks = _oa_stream(n=12)
    state = {}

    async def pause():
        front = state["front"]
        t = time.time()
        if cause == "flip":
            fp = F.Front._flip_phase(front)
            fp.layer("D>P", t, "test")
            await asyncio.sleep(0.6)
            fp.done(time.time())
        elif cause == "park":
            F.Front._req_book(front).park(RID, t, "wait_bound:test", front.epoch)
            await asyncio.sleep(0.6)
            F.Front._rb_resume(front, [RID], "flip_to_d")
        elif cause == "prefill_d":
            # another request's leg 2 admitted to D (its prefill window)
            opener = getattr(front, "_d_admit_open", None)  # absent on the parent tree
            e = opener("weg2-other", t) if opener is not None else [None, t, None]
            await asyncio.sleep(0.6)
            e[2] = time.time()
        else:
            await asyncio.sleep(0.6)

    def grab(front):
        state["front"] = front

    return asyncio.run(_run("/v1/chat/completions", chunks, True, p_legs=(),
                            hooks={5: pause}, before_leg2=grab))


def _final_details(got):
    return [o for o in _objs(got) if o.get("usage")][-1]["usage"]["total_tokens_details"]


def test_stream_gap_during_a_flip_is_one_flip_sleep():
    _f, got = _gap_run("flip")
    d = _final_details(got)
    assert d["sleep"] == 1 and d["sleep_causes"] == {"flip": 1, "prefill_d": 0, "park": 0}
    assert 0.5 < d["sleep_s"] < 2.0
    assert d["flips"] == 1


def test_stream_gap_during_a_park_is_one_park_sleep():
    _f, got = _gap_run("park")
    d = _final_details(got)
    assert d["sleep_causes"] == {"flip": 0, "prefill_d": 0, "park": 1} and d["flips"] == 0


def test_stream_gap_during_a_d_prefill_is_one_prefill_d_sleep():
    _f, got = _gap_run("prefill_d")
    d = _final_details(got)
    assert d["sleep_causes"] == {"flip": 0, "prefill_d": 1, "park": 0}


def test_stream_gap_without_a_cause_is_logged_unnamed(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    _f, got = _gap_run(None)
    d = _final_details(got)
    assert d["sleep"] == 0 and d["sleep_s"] == 0.0
    m = _lines(caplog, "WEG2-USAGE-GAP-UNNAMED")
    # the 0.6 s pause (a starved test process may add a shorter one)
    assert m and all(f"rid={RID} ms=" in x for x in m)
    assert max(int(x.rsplit("ms=", 1)[1]) for x in m) >= 550
    # decode_s excludes nothing here, and the gap is inside it
    assert d["decode_s"] >= 0.5


def test_decode_s_excludes_the_sleep():
    _f, got = _gap_run("flip")
    d = _final_details(got)
    assert d["decode_s"] < 0.5  # the 0.6 s flip gap is not decode time
    assert abs(d["decode_tps"] - OUT / d["decode_s"]) < 0.5  # completion / decode_s


# ------------------------------------------------------ reasoning count --


def test_reasoning_tokens_counted_by_the_front_tokenizer_when_d_did_not():
    body = json.loads(_oa_body())
    body["choices"][0]["message"]["reasoning_content"] = "a b c d e"
    raw = json.dumps(body).encode()

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def give_tok(front):
        front.ftok = SimpleNamespace(_tok=_Tok())

    _f, got = asyncio.run(_run("/v1/chat/completions", raw, False, p_legs=(), before_leg2=give_tok))
    assert json.loads(got)["usage"]["completion_tokens_details"] == {"reasoning_tokens": 5}
    # D's own count is never overwritten
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 3}
    _f, got = asyncio.run(_run("/v1/chat/completions", json.dumps(body).encode(), False, p_legs=(),
                               before_leg2=give_tok))
    assert json.loads(got)["usage"]["completion_tokens_details"] == {"reasoning_tokens": 3}


# ---------------------------------------------------- tier split (02.10.) --


def test_trim_tiers_removes_the_excess_deepest_first_never_negative():
    t = {"cached_device": 1000, "cached_l2": 2000, "cached_l3": 1000}
    assert UT.trim_tiers(t, 3990) == {"cached_device": 1000, "cached_l2": 2000, "cached_l3": 990}
    assert UT.trim_tiers(t, 1500) == {"cached_device": 1000, "cached_l2": 500, "cached_l3": 0}
    assert UT.trim_tiers(t, 200) == {"cached_device": 200, "cached_l2": 0, "cached_l3": 0}
    assert UT.trim_tiers(t, 0) == {"cached_device": 0, "cached_l2": 0, "cached_l3": 0}
    # a tier the rank did not report is not invented; a split below the target is never raised
    assert UT.trim_tiers({"cached_device": 50, "cached_l2": 70}, 100) == {"cached_device": 50, "cached_l2": 50}
    assert UT.trim_tiers({"cached_device": 50}, 100) == {"cached_device": 50}


def test_flipped_split_sums_to_the_corrected_cached_tokens():
    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_body(), False))
    ptd = json.loads(got)["usage"]["prompt_tokens_details"]
    assert ptd["cached_device"] + ptd["cached_l2"] + ptd["cached_l3"] == ptd["cached_tokens"] == 3990
    _f, got = asyncio.run(_run("/v1/messages", _anth_body(), False))
    u = json.loads(got)["usage"]
    d = u["total_tokens_details"]
    assert d["cached_device"] + d["cached_l2"] + d["cached_l3"] == u["cache_read_input_tokens"] == 3990


def test_d_only_follow_up_shows_only_ds_split_never_an_earlier_p_leg():
    """An earlier rid (an earlier turn of the same conversation) ran a P leg and
    sits in the ledger; this rid is served on D alone: its split is D's own,
    summing to D's cached count -- nothing from any P leg."""
    d_tiers = {"device": 19000, "host": 990, "storage": 0}

    def earlier_turn(front):
        front._p_leg_note("weg2-2-0", PROMPT, P_CACHED,
                          {"cached_device": 1000, "cached_l2": 2000, "cached_l3": 1000})

    _f, got = asyncio.run(_run("/v1/chat/completions", _oa_body(tiers=d_tiers), False, p_legs=(),
                               before_leg2=earlier_turn))
    u = json.loads(got)["usage"]
    ptd = u["prompt_tokens_details"]
    assert (ptd["cached_device"], ptd["cached_l2"], ptd["cached_l3"]) == (19000, 990, 0)
    assert ptd["cached_device"] + ptd["cached_l2"] + ptd["cached_l3"] == ptd["cached_tokens"] == D_CACHED
    assert u["total_tokens_details"]["p_computed"] == 0
