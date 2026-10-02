"""FLIP-LEGS 02.10.: P's PLE hint is tokenized off the HTTP event loop.

MEASURED (N5a b49f0282c2 1002_114540, D->P flip ep7): the front posted a
/weg2/ple_prefetch_hint for a 110k-token prompt to P 0.13 s before the flip
(`WEG2 PLE-HINT ... ms=1366`); P's HTTP server tokenized it ON its event loop,
and P's resume RPC for the flip reached PP0's scheduler 720 ms after the front
issued it (CTRL-RECV) -- every other D->P flip of N4p/N4q/N5a: 1-3 ms. The
depositors waited 715-873 ms on credit, legs 2036 ms against ~1400.

Pinned (red before): the builder runs in a worker thread while the loop keeps
ticking (a 0.6 s tokenization stand-in), returns the same hint, names
off_loop/ms; the switch restores the on-loop form; the endpoint uses it and
prints WEG2-PLE-HINT-BUILD.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ple_admit_hint as ph  # noqa: E402

DELAY = 0.6


def _slow_encode(text):
    time.sleep(DELAY)
    return [ord(c) for c in text]


async def _with_ticker(coro):
    ticks = []

    async def tick():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    t = asyncio.ensure_future(tick())
    await asyncio.sleep(0)
    try:
        out = await coro
    finally:
        ticks.append(time.monotonic())   # the gap up to the coroutine's end counts
        t.cancel()
    gap = max((b - a for a, b in zip(ticks, ticks[1:])), default=0.0)
    return out, gap


def test_switch_default_on():
    assert ph.hint_off_loop_on({})
    for off in ("0", "false", "no", "off"):
        assert not ph.hint_off_loop_on({ph.OFF_LOOP_ENV: off})


def test_the_loop_keeps_ticking_while_the_hint_is_tokenized(monkeypatch):
    monkeypatch.delenv(ph.OFF_LOOP_ENV, raising=False)
    threads = []

    def enc(text):
        threads.append(threading.get_ident())
        return _slow_encode(text)

    body = {"path": "/generate", "payload": {"rid": "weg2-6-9", "text": "abc"}}

    async def go():
        loop_tid = threading.get_ident()
        (hint, ms, off), gap = await _with_ticker(ph.build_ple_prefetch_hint_off_loop(
            body, serving_chat=None, serving_completion=None, encode=enc))
        return hint, ms, off, gap, loop_tid

    hint, ms, off, gap, loop_tid = asyncio.run(go())
    assert (hint.rid, hint.input_ids) == ("weg2-6-9", [97, 98, 99])
    assert off is True and ms >= DELAY * 1000 - 50
    assert threads and all(t != loop_tid for t in threads)
    assert gap < 0.2, f"the loop was held {gap:.3f} s"


def test_switch_off_tokenizes_on_the_loop(monkeypatch):
    monkeypatch.setenv(ph.OFF_LOOP_ENV, "0")
    body = {"path": "/generate", "payload": {"rid": "r", "text": "ab"}}

    async def go():
        return await _with_ticker(ph.build_ple_prefetch_hint_off_loop(
            body, serving_chat=None, serving_completion=None, encode=_slow_encode))

    (hint, ms, off), gap = asyncio.run(go())
    assert hint.input_ids == [97, 98] and off is False
    assert gap >= DELAY - 0.1


def test_the_coroutine_form_and_the_sync_form_agree():
    body = {"path": "/generate", "payload": {"rid": "g", "text": "ab"}}
    kw = dict(serving_chat=None, serving_completion=None, encode=lambda t: [1, 2])
    a = asyncio.run(ph.build_ple_prefetch_hint(body, **kw))
    b = ph.build_ple_prefetch_hint_sync(body, **kw)
    assert (a.rid, a.input_ids) == (b.rid, b.input_ids) == ("g", [1, 2])
    assert ph.build_ple_prefetch_hint_sync({"path": "/x", "payload": {"rid": "g"}}, **kw) is None


def test_the_endpoint_builds_off_the_loop_and_says_so():
    src = open(os.path.join(os.path.dirname(ph.__file__), "..", "entrypoints", "http_server.py")).read()
    i = src.index("async def weg2_ple_prefetch_hint")
    j = src.index("@app.api_route", i)
    body = src[i:j]
    assert "build_ple_prefetch_hint_off_loop(" in body
    assert "await build_ple_prefetch_hint(" not in body
    assert "WEG2-PLE-HINT-BUILD rid=%s tokens=%s ms=%.0f off_loop=%s" in body
