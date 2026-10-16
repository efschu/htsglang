"""DECODE-COLLECT NODECODE 1015, review condition A1 (done/nf-collect-nodecode-1015.md, review JA_MIT_AUFLAGE): the decode
witness of ``Front._dc_decoding`` is "no open X-SUM-PRICE row" (``DPrefillInflight.is_open``). leg 2 closed that row ONLY when
the lookahead HEAD already carried content (front.py ``_has_content`` -> ``self._d_pf_book().done(rid=rid)``); a stream the
front commits WITHOUT content -- on the Anthropic wire the lookahead gives up after ``message_start`` + 7 pings
(ANTHROPIC_LOOKAHEAD_MAX_CHUNKS = 8) while D still prefills or holds the rid parked -- got its content later in the body
(``_push``), which checked ``stream_has_content(chunk)`` but never called ``done()``. The only other close was the leg end.
At fc9444c027 such a rid counted as "no decode" (``carried``) for its WHOLE decode: a later arrival on that D was priced with
its full tokens in ``carried`` and, without a hold, sent to P at once (sticky for the phase = a D->P flip).

Measured (reviewer, 69 front logs 13.-15.10.): 337 of 18759 booked SHORT/d_direct legs (stream=1, status=200,
completion_tokens>0) never log LEG2-FIRST-CONTENT. The real case pinned here, boot
/spinning/docker-acceptance/nf/evidence/boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10130457_cfeb2896c1_1013_045811.front.log:

* :33865 06:30:44,203 X-EXACT-PRICE rid=weg2-198-622 pending=5832 tokens=47816; :33867 X-ROUTE est_uncached=5832 X=6404
* :33911 06:30:58,846 WEG2-ROUTE weg2-198-622 SHORT -> D remainder=5832; :33912 06:30:58,847 D-ADMIT seat=5/6 source=short
* :34208 06:33:44,141 PARK-HANDBACK rid=weg2-198-622 kept=committed (its stream was committed without content)
* :34709 06:34:43,172 USAGE-DETAILS ttft_s=227.905; :34714 06:34:43,481 WEG2-SERVED group=D leg=2 stream=1status=200
  completion_tokens=173 -- and NO LEG2-FIRST-CONTENT line for this rid anywhere in the log.

Fix: ``_push`` closes the row once, at the first chunk with content (``stream_has_content``), in the body as well.
Mutants (each must turn a test here red): "done only on the head" (the fc9444c027 code) and "done on every chunk"
(a ping in the body would end D's prefill -- the carried prefill would count as a decode and hold arrivals again).
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_x_sum_price_book_lifecycle_1010 import _drop, _FakeD  # noqa: E402

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

RID = "weg2-198-622"
X = 6404          # :33867 X-ROUTE ... X=6404
PENDING = 5832    # :33865 X-EXACT-PRICE ... pending=5832

START = b'event: message_start\ndata: {"type":"message_start"}\n\n'
PING = b'event: ping\ndata: {"type": "ping"}\n\n'
CONTENT = (b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
           b'"content_block":{"type":"text","text":""}}\n\n'
           b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
           b'"delta":{"type":"text_delta","text":"hi"}}\n\n')


class _LateContentBody:
    """D's Anthropic stream for a rid D still prefills / holds parked: ``message_start`` + 7 pings (the lookahead's 8
    chunks -- it commits the stream without content), one more ping read by the body loop, then -- once ``gate`` is set --
    the first content; after it D decodes (the next read never returns, the rid stays in ``outstanding``)."""

    def __init__(self, gate: asyncio.Event):
        self.gate = gate
        self.chunks = [START] + [PING] * 7 + [PING]
        self.reads = 0
        self.at_gate = asyncio.Event()
        self.decoding = asyncio.Event()

    async def readany(self):
        self.reads += 1
        if self.reads <= len(self.chunks):
            return self.chunks[self.reads - 1]
        if self.reads == len(self.chunks) + 1:
            self.at_gate.set()  # the 9th chunk (a ping) went through the body loop; D still prefills
            await self.gate.wait()
            return CONTENT
        self.decoding.set()  # past the first content: D decodes
        await asyncio.sleep(3600)
        return b""

    async def iter_any(self):
        yield await self.readany()


def _front():
    f = Front("http://p", "http://d", "D", "nodecode-a1", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    return f


def _req():
    from aiohttp.test_utils import make_mocked_request

    return make_mocked_request("POST", "/v1/messages")


def _payload():
    return {"stream": True, "model": "m", "max_tokens": 64000,
            "messages": [{"role": "user", "content": "x"}]}


def _run(check_at_gate, check_decoding):
    async def body():
        f = _front()
        gate = asyncio.Event()
        stream = _LateContentBody(gate)
        f.session = _FakeD(stream)
        f._d_pf_book().grant(rid=RID, tokens=PENDING, now=time.time())  # :33911 SHORT -> D remainder=5832
        task = asyncio.ensure_future(f.leg2(_req(), RID, _payload(), "x", True, None, seat=None))
        try:
            await asyncio.wait_for(stream.at_gate.wait(), 5.0)
            await asyncio.sleep(0.05)
            check_at_gate(f)
            gate.set()
            await asyncio.wait_for(stream.decoding.wait(), 5.0)
            await asyncio.sleep(0.05)
            check_decoding(f)
        finally:
            await _drop(task)

    asyncio.run(body())


def test_committed_without_content_the_row_stays_open_through_pings():
    """Gegenprobe (mutant "done on every chunk"): the lookahead committed the stream on message_start + 7 pings and the
    body read one more ping -- D has produced nothing yet, its prefill is NOT over: the row stays open, the rid is carried
    (5832 of the sum), not a decode."""
    def at_gate(f):
        assert RID in f.groups["D"].outstanding, "leg 2 holds the rid on D"
        assert f._d_pf_book().is_open(rid=RID), "a ping is not content: D still prefills, the row must stay open"
        assert f._d_carried_tokens() == PENDING, "the prefilling SHORT is carried with its priced 5832"
        assert not f._dc_decoding(), "a rid D still prefills is not a decode (NODECODE 1015)"

    _run(at_gate, lambda f: None)


def test_real_198_622_late_first_content_in_the_body_ends_the_row():
    """A1 repro (weg2-198-622, cfeb2896c1_1013_045811 :34208 kept=committed, :34714 completion_tokens=173, no
    LEG2-FIRST-CONTENT): the stream was committed without content; the content comes later through the body loop. From
    that chunk on D decodes this rid: the X-SUM-PRICE row is closed, carried drops to 0, and it counts as a decode. Red
    on fc9444c027 (done only on the head: the row stayed open for the whole decode, 5832 carried, no decode)."""
    def decoding(f):
        assert RID in f.groups["D"].outstanding, "D decodes it: leg 2 still holds the rid"
        assert not f._d_pf_book().is_open(rid=RID), (
            "first content in the body = D's prefill is over: the X-SUM-PRICE row must end here, not at the leg end")
        assert f._d_carried_tokens() == 0, "a decoding rid is not carried"
        assert f._dc_decoding(), "the rid decodes on D: DECODE-COLLECT may hold for it"

    _run(lambda f: None, decoding)


def test_late_first_content_is_logged_once(caplog):
    """The instrument for the live check: the late first content is named once (LEG2-FIRST-CONTENT-BODY), so the next NF
    boot can count these legs next to the 337/18759 of 13.-15.10."""
    import logging

    caplog.set_level(logging.INFO, logger="weg2.front")

    def decoding(f):
        lines = [r.getMessage() for r in caplog.records if "LEG2-FIRST-CONTENT-BODY" in r.getMessage()]
        assert len(lines) == 1 and f"rid={RID}" in lines[0], lines

    _run(lambda f: None, decoding)


def test_head_with_content_closes_once_and_the_body_does_not_reopen():
    """Gegenprobe: the ordinary leg (content in the lookahead head) still ends its row at the head; later chunks change
    nothing (no second close, no log line from the body)."""
    import logging

    class _HeadContent(_LateContentBody):
        def __init__(self, gate):
            super().__init__(gate)
            self.chunks = [START, CONTENT, PING]

    async def body():
        f = _front()
        gate = asyncio.Event()
        stream = _HeadContent(gate)
        f.session = _FakeD(stream)
        f._d_pf_book().grant(rid=RID, tokens=PENDING, now=time.time())
        task = asyncio.ensure_future(f.leg2(_req(), RID, _payload(), "x", True, None, seat=None))
        try:
            await asyncio.wait_for(stream.at_gate.wait(), 5.0)
            await asyncio.sleep(0.05)
            assert not f._d_pf_book().is_open(rid=RID)
            assert f._dc_decoding()
        finally:
            await _drop(task)

    logging.getLogger("weg2.front").setLevel(logging.INFO)
    asyncio.run(body())
