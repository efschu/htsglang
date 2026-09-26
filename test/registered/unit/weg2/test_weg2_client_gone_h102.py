"""H102 (#23p, 2026-09-26): a client that hangs up before its answer must
reach the work behind it.

MEASURED (dkrnfbar1agent0925 2021 Z. 6098, 2237 Z. 1164/2347): leg 1 runs in
the front's controller task, the handler only awaits the request's future,
and aiohttp 3.14 does not cancel a handler on a hang-up. P prefilled for a
dead client to the end; only leg 2 fell on ``ClientConnectionResetError``.

Now a per-request watcher sees the client's transport close and, per state:
queued -> dequeued; p-leg1 -> ``/abort_request`` on P (the intake-stall
path); handoff -> dropped before D; d -> unchanged (leg 2's write fails and
closes D's connection); parked (H91c) -> ``/abort_request`` on D. One line
``WEG2-CLIENT-GONE rid=... state=... action=...`` each.

Hermetic: fake P, fake D and the front are real in-process aiohttp servers on
loopback; the front runs under ``AppRunner`` as in production (handler
cancellation OFF -- ``TestServer`` turns it on and would hide the defect); the
client is a raw socket that really closes. The controller does not run: the
test moves the request between states the way the drain and the admitter do.
No CUDA, no model.
"""

import asyncio
import json
import logging

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as fr
from sglang.srt.weg2.front import Front

PROMPT = "a short agent turn"


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def gone(self):
        return [m for m in self.lines if m.startswith("WEG2-CLIENT-GONE")]


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _fake_group(seen: dict, name: str) -> web.Application:
    """A group server: its generate call runs until /abort_request names its
    rid (then answers like an aborted request), or ``seen[name+'_release']``
    is set. Streams when asked to."""
    aborted = {}

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        rid = body.get("rid")
        seen.setdefault(f"{name}_calls", []).append(rid)
        ev = aborted.setdefault(rid, asyncio.Event())
        if body.get("stream"):
            resp = web.StreamResponse(status=200)
            resp.content_type = "text/event-stream"
            await resp.prepare(request)
            try:
                await resp.write(_sse({"choices": [{"index": 0, "delta": {"content": "t0"},
                                                    "finish_reason": None}]}))
                for i in range(1, 400):
                    if ev.is_set():
                        await resp.write(_sse({"choices": [{"index": 0, "delta": {},
                                                            "finish_reason": "abort"}]}))
                        break
                    if not seen.get(f"{name}_park"):
                        await resp.write(_sse({"choices": [{"index": 0, "delta": {"content": f"t{i}"},
                                                            "finish_reason": None}]}))
                    await asyncio.sleep(0.1)
                await resp.write_eof()
                seen[f"{name}_completed"] = True
            except ConnectionResetError:
                seen[f"{name}_conn_closed"] = True
            except asyncio.CancelledError:
                # TestServer cancels the handler when the front closes the
                # connection -- what a group server's disconnect check does.
                seen[f"{name}_conn_closed"] = True
                raise
            return resp
        for _ in range(400):
            if ev.is_set():
                return web.json_response({"error": {"message": "Abort request"}}, status=400)
            await asyncio.sleep(0.05)
        seen[f"{name}_completed"] = True
        return web.json_response({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 1,
                                            "prompt_tokens_details": {"cached_tokens": 0}}})

    async def abort(request: web.Request) -> web.Response:
        body = await request.json()
        seen.setdefault(f"{name}_aborts", []).append(body.get("rid"))
        aborted.setdefault(body.get("rid"), asyncio.Event()).set()
        return web.json_response({})

    async def info(request: web.Request) -> web.Response:
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/abort_request", abort)
    app.router.add_get("/get_server_info", info)
    return app


async def _until(cond, timeout=5.0):
    t = 0.0
    while not cond():
        await asyncio.sleep(0.02)
        t += 0.02
        if t > timeout:
            raise AssertionError("condition not reached")


class _Rig:
    def __init__(self):
        self.seen: dict = {}
        self.lines = _Lines()

    async def __aenter__(self):
        self.p = TestServer(_fake_group(self.seen, "p"))
        self.d = TestServer(_fake_group(self.seen, "d"))
        await self.p.start_server()
        await self.d.start_server()
        purl = str(self.p.make_url("")).rstrip("/")
        durl = str(self.d.make_url("")).rstrip("/")
        self.front = Front(purl, durl, "P", "t", "", 0, 0, {}, 45.0,
                           carrier_max_tokens=27466, tp_prefill_max_tokens=4096)
        self.front.groups["P"].url = purl
        self.front.groups["D"].url = durl
        self.front.session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.host, self.port = self.runner.addresses[0][:2]
        fr.logger.addHandler(self.lines)
        return self

    async def _handler(self, request):
        try:
            return await self.front.handle_generate(request)
        finally:
            self.front.counters["test_handler_done"] += 1

    async def __aexit__(self, *exc):
        fr.logger.removeHandler(self.lines)
        await self.runner.cleanup()
        await self.front.session.close()
        await self.p.close()
        await self.d.close()

    async def open_client(self, stream: bool):
        """A raw client: sends one request and keeps the socket; close() is a
        real hang-up."""
        reader, writer = await asyncio.open_connection(self.host, self.port)
        body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                           "stream": stream, "max_tokens": 64}).encode()
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Type: application/json\r\nContent-Length: "
                     + str(len(body)).encode() + b"\r\n\r\n" + body)
        await writer.drain()
        return reader, writer

    def pending(self):
        return self.front.queue[0] if self.front.queue else None


async def _hang_up(writer):
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:  # noqa: BLE001
        pass


def _run(coro):
    with envs.SGLANG_WEG2_PLE_ADMIT_HINT.override(False):
        return asyncio.run(coro)


# ---------------------------------------------------------------- queued
async def _queued():
    async with _Rig() as rig:
        _r, w = await rig.open_client(stream=False)
        await _until(lambda: len(rig.front.queue) == 1)
        p = rig.pending()
        await _hang_up(w)
        await _until(lambda: rig.front.counters["test_handler_done"] == 1)
        await asyncio.sleep(0.3)
        return rig, p


def test_queued_hang_up_is_dequeued_and_named():
    rig, p = _run(_queued())
    assert len(rig.front.queue) == 0, "a dead client's request must leave the queue"
    assert p.client_gone
    gone = rig.lines.gone()
    assert len(gone) == 1, gone
    assert f"rid={p.rid} state=queued action=dequeued" in gone[0]
    assert rig.seen.get("p_calls") is None and rig.seen.get("p_aborts") is None


# ---------------------------------------------------------------- p-leg1
async def _p_leg1():
    async with _Rig() as rig:
        _r, w = await rig.open_client(stream=False)
        await _until(lambda: len(rig.front.queue) == 1)
        p = rig.front.queue.popleft()           # the drain takes it
        leg1 = asyncio.ensure_future(rig.front.leg1(p))
        await _until(lambda: p.rid in rig.front.groups["P"].outstanding
                     and rig.seen.get("p_calls"))
        await _hang_up(w)
        try:
            await asyncio.wait_for(leg1, 5.0)
        except Exception as e:  # noqa: BLE001 -- P answers the abort with an error
            rig.seen["leg1_error"] = repr(e)
        await _until(lambda: rig.front.counters["test_handler_done"] == 1)
        return rig, p


def test_p_leg1_hang_up_aborts_on_p_like_the_intake_stall():
    rig, p = _run(_p_leg1())
    assert rig.seen.get("p_aborts") == [p.rid], rig.seen
    assert rig.seen.get("p_completed") is not True, "P must not prefill to the end for a dead client"
    assert p.client_gone and p.fut.done()
    gone = rig.lines.gone()
    assert len(gone) == 1, gone
    assert f"rid={p.rid} state=p-leg1 action=abort-p status=200" in gone[0]


async def _p_slot_wait():
    async with _Rig() as rig:
        _r, w = await rig.open_client(stream=False)
        await _until(lambda: len(rig.front.queue) == 1)
        p = rig.front.queue.popleft()           # popped, waiting for a P slot
        await _hang_up(w)
        await _until(lambda: rig.front.counters["test_handler_done"] == 1)
        return rig, p


def test_p_slot_wait_hang_up_skips_leg1():
    rig, p = _run(_p_slot_wait())
    assert p.client_gone and p.fut.done()
    gone = rig.lines.gone()
    assert len(gone) == 1 and f"rid={p.rid} state=p-leg1 action=leg1-skipped" in gone[0], gone
    assert rig.seen.get("p_aborts") is None     # nothing on P to abort yet
    # the drain's `one` reads the flag before it POSTs leg 1
    src = open(fr.__file__).read()
    i = src.index("async def one(p: Pending) -> Pending:")
    blk = src[i:i + 700]
    assert blk.index("if p.client_gone:") < blk.index("await self.leg1(p)")
    j = src.index("def _on_leg1_done(p: Pending)")
    assert "if p.client_gone:" in src[j:j + 500]


# ---------------------------------------------------------------- handoff
async def _handoff():
    async with _Rig() as rig:
        _r, w = await rig.open_client(stream=False)
        await _until(lambda: len(rig.front.queue) == 1)
        p = rig.front.queue.popleft()
        p.leg1_done = True                       # leg 1 answered
        rig.front._ready_for_d.append(p)         # waiting for D (flip / seat)
        rig.front._sync_batch_gate()
        await _hang_up(w)
        await _until(lambda: rig.front.counters["test_handler_done"] == 1)
        return rig, p


def test_handoff_hang_up_is_never_handed_to_d():
    rig, p = _run(_handoff())
    assert not rig.front._ready_for_d
    assert rig.front._batch_gate.is_set()
    gone = rig.lines.gone()
    assert len(gone) == 1 and f"rid={p.rid} state=handoff action=dropped-before-d" in gone[0], gone
    assert rig.seen.get("d_calls") is None


# ---------------------------------------------------------------- d and parked
async def _on_d(park: bool):
    async with _Rig() as rig:
        r, w = await rig.open_client(stream=True)
        await _until(lambda: len(rig.front.queue) == 1)
        p = rig.front.queue.popleft()
        p.leg1_done = True
        rig.front.awake = "D"
        p.fut.set_result(True)                   # the admitter hands it to D
        await _until(lambda: p.rid in rig.front.groups["D"].outstanding
                     and rig.seen.get("d_calls"))
        await r.readuntil(b"t0")                 # the answer has started
        if park:
            rig.seen["d_park"] = True            # D stops streaming: parked
            await asyncio.sleep(0.15)
            rig.front._d_parked = {p.rid: 0.0}   # H91c's ledger of parked rids
        await _hang_up(w)
        await _until(lambda: rig.front.counters["test_handler_done"] == 1, timeout=8.0)
        # leg 2's own write error may end the handler before the watcher's
        # tick; the watcher still names the state once.
        await asyncio.sleep(0.5)
        return rig, p


def test_d_hang_up_keeps_todays_leg2_abort_and_is_named():
    rig, p = _run(_on_d(park=False))
    gone = rig.lines.gone()
    assert len(gone) == 1 and f"rid={p.rid} state=d action=none(leg2-path)" in gone[0], gone
    assert rig.seen.get("d_aborts") is None     # no new call on D: today's path
    assert rig.seen.get("d_conn_closed") is True
    assert rig.seen.get("d_completed") is not True


def test_parked_hang_up_releases_the_park_on_d():
    rig, p = _run(_on_d(park=True))
    assert rig.seen.get("d_aborts") == [p.rid], rig.seen
    assert p.rid not in rig.front._d_parked
    gone = rig.lines.gone()
    assert len(gone) == 1 and f"rid={p.rid} state=parked action=abort-d-park status=200" in gone[0], gone


# ---------------------------------------------------------------- unchanged paths
async def _read_to_end():
    async with _Rig() as rig:
        r, w = await rig.open_client(stream=False)
        await _until(lambda: len(rig.front.queue) == 1)
        p = rig.front.queue.popleft()
        p.leg1_done = True
        rig.front.awake = "D"
        rig.seen["d_calls"] = []
        p.fut.set_result(True)
        # the fake D answers its non-stream call only after ~20 s unless
        # aborted; shorten: abort it through the test so it answers now.
        await _until(lambda: rig.seen.get("d_calls"))
        async with aiohttp.ClientSession() as cs:
            await cs.post(f"{rig.front.groups['D'].url}/abort_request", json={"rid": p.rid})
        data = await r.read(65536)
        await asyncio.sleep(0.5)
        await _hang_up(w)
        await asyncio.sleep(0.5)
        return rig, p, data


def test_hang_up_after_the_answer_is_not_client_gone():
    rig, p, data = _run(_read_to_end())
    assert data.startswith(b"HTTP/1.1 ")
    assert rig.lines.gone() == []
    assert rig.front.counters["test_handler_done"] == 1


async def _switch_off():
    with envs.SGLANG_WEG2_ENABLE_CLIENT_GONE_ABORT.override(False):
        async with _Rig() as rig:
            _r, w = await rig.open_client(stream=False)
            await _until(lambda: len(rig.front.queue) == 1)
            await _hang_up(w)
            await asyncio.sleep(0.8)
            n = len(rig.front.queue)
            p = rig.pending()
            if p is not None and not p.fut.done():
                p.fut.set_exception(RuntimeError("test teardown"))
            await _until(lambda: rig.front.counters["test_handler_done"] == 1)
            return rig, n


def test_switch_off_is_the_pre_h102_behaviour():
    rig, n = _run(_switch_off())
    assert n == 1
    assert rig.lines.gone() == []


def test_switch_is_declared_default_on():
    assert envs.SGLANG_WEG2_ENABLE_CLIENT_GONE_ABORT.get() is True
