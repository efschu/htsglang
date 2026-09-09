"""#1285: the front's RPC connection instrument, and the retry that reads it.

WHY A REAL SOCKET AND NOT A MOCK.  The whole question of #1285 is what aiohttp
does with a POOLED KEEPALIVE CONNECTION whose peer has closed it, and at which
point in the request the failure surfaces.  A mocked session answers whatever
the mock was written to believe; only a real client against a real listening
socket can say whether `ServerDisconnectedError` arrives before the response
line (retryable -- the handler never ran) or during the body (NOT retryable --
the handler ran and its effect is applied).  The server here is a raw asyncio
one on purpose: `aiohttp.web` gives no control over *when* a connection is
dropped, and the drop timing IS the specimen.

Boot weg2sb5e: both gathered flip legs came back `ServerDisconnectedError`
after ~117 s with `RPC_TIMEOUT_S=900`, P logged no access line, and nothing
recorded which socket the request went out on.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import re

from aiohttp import ServerDisconnectedError

from sglang.srt.weg2.front import (
    Group,
    rpc_leg_name,
    rpc_pool_counts,
)


# ---------------------------------------------------------------- test server
class Recorder:
    """A raw HTTP/1.1 server with scripted, per-connection fault behaviour."""

    def __init__(self, script):
        #: one entry per ACCEPTED connection, consumed in order; the last entry
        #: repeats for any further connection.
        self.script = list(script)
        self.requests = []          # (conn_index, request line, body)
        self.conns = 0
        self._server = None
        self.port = 0

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _mode(self, idx):
        return self.script[idx] if idx < len(self.script) else self.script[-1]

    async def _handle(self, reader, writer):
        idx = self.conns
        self.conns += 1
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                m = re.search(rb"Content-Length: (\d+)", head, re.I)
                body = await reader.readexactly(int(m.group(1))) if m else b""
                line = head.split(b"\r\n")[0].decode()
                self.requests.append((idx, line, body.decode()))
                mode = self._mode(idx)
                if mode == "drop_before_response":
                    # The stale-keepalive shape: nothing at all comes back.
                    writer.close()
                    return
                if mode == "partial":
                    # A response LINE and a promised body that never arrives:
                    # the handler ran, so this must NOT be retried.
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\n\r\n")
                    writer.write(b"half")
                    await writer.drain()
                    writer.close()
                    return
                payload = b'{"ok":true}'
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s"
                             % (len(payload), payload))
                await writer.drain()
                if mode == "ok_then_close":
                    # Answer, then drop the connection the client just pooled.
                    writer.close()
                    return
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass


class FrontStub:
    """Only the two attributes `Front.rpc` / `Front.leg_rpc` actually touch."""

    admin_key = None

    def __init__(self, session):
        self.session = session

    # bind the real implementations under test
    from sglang.srt.weg2.front import Front as _F
    _rpc_attempt = _F._rpc_attempt
    rpc = _F.rpc
    leg_rpc = _F.leg_rpc
    del _F


def sync(fn):
    """Run one coroutine test on its own loop.

    NOT pytest-asyncio: the remote desk container does not carry it, and a test
    whose whole point is real sockets must not also depend on a plugin being
    installed wherever it runs.
    """

    @functools.wraps(fn)
    def wrapper(*a, **k):
        return asyncio.run(fn(*a, **k))

    return wrapper


async def _front(script):
    from aiohttp import ClientSession, ClientTimeout
    from sglang.srt.weg2.front import SportTCPConnector, make_rpc_trace_config

    srv = Recorder(script)
    await srv.start()
    session = ClientSession(timeout=ClientTimeout(total=30),
                            connector=SportTCPConnector(),
                            trace_configs=[make_rpc_trace_config()])
    return srv, session, FrontStub(session)


def _lines(caplog, marker):
    return [r.getMessage() for r in caplog.records if marker in r.getMessage()]


SLEEP = "/release_memory_occupation"
WAKE = "/resume_memory_occupation"


# ------------------------------------------------------------------- the leg
def test_leg_names_are_per_direction():
    assert rpc_leg_name(SLEEP) == "sleep"
    assert rpc_leg_name(WAKE) == "wake"
    assert rpc_leg_name("/flush_cache") == "other"


# --------------------------------------------------------- red-first: retry
@sync
async def test_a_peer_closed_pooled_connection_is_not_reused():
    """MEASURED, and it is evidence about #1285 rather than about this code.

    The leading hypothesis for weg2sb5e was a STALE POOLED KEEPALIVE
    connection: the peer closed it during one of the long idle gaps the five
    stalled drains left, and the next leg wrote into the corpse.  Under
    aiohttp 3.14.1 with the front's default connector that does NOT happen --
    the closed connection is evicted and the request goes out on a fresh one,
    succeeding.  So a bare stale keepalive does not by itself produce the
    weg2sb5e symptom, and the hypothesis stays UNPROVEN.
    """
    srv, session, front = await _front(["ok_then_close", "ok"])
    g = Group(name="P", url=srv.url)
    try:
        assert (await front.rpc(g, "/health", None, 30))[0] == 200
        await asyncio.sleep(0.05)                            # let the FIN land
        code, text = await front.rpc(g, SLEEP, {"tags": ["kv_cache"], "epoch": "e1"}, 30)
        assert code == 200, text
        assert {i for i, _, _ in srv.requests} == {0, 1}
    finally:
        await session.close()
        await srv.stop()


@sync
async def test_a_drop_before_the_response_is_retried_once_and_succeeds():
    """RED against the old code: `rpc` alone returns 0/ServerDisconnectedError.

    The specimen shape: the server accepts the request, never answers, and
    closes.  That is what the client saw on weg2sb5e -- and what P's missing
    access line is consistent with.
    """
    srv, session, front = await _front(["drop_before_response", "ok"])
    g = Group(name="P", url=srv.url)
    try:
        # old behaviour, still available and still unretried
        code, text = await front.rpc(g, SLEEP, {"tags": ["kv_cache"], "epoch": "e1"}, 30)
        assert code == 0 and "Disconnected" in text, text
    finally:
        await session.close()
        await srv.stop()

    # new behaviour: one retry on a FRESH connection
    srv2, session2, front2 = await _front(["drop_before_response", "ok"])
    g2 = Group(name="P", url=srv2.url)
    try:
        code, text = await front2.leg_rpc(
            g2, SLEEP, {"tags": ["kv_cache"], "epoch": "e1"}, 30)
        assert code == 200, text
        # MUTANT M3 (reuse the shared session instead of a fresh connection):
        # the retry's 200 would return its connection to the SHARED pool.  The
        # fresh force_close session leaves nothing behind.
        assert rpc_pool_counts(front2.session)[0] == 0
        # the retry really was a second connection, server-side
        assert {i for i, _, _ in srv2.requests} == {0, 1}
    finally:
        await session2.close()
        await srv2.stop()


@sync
async def test_a_leg_without_an_epoch_is_never_retried():
    """No epoch on the request -> no dedup key on the far side -> no retry."""
    srv, session, front = await _front(["drop_before_response"])
    g = Group(name="P", url=srv.url)
    try:
        code, _ = await front.leg_rpc(g, SLEEP, {"tags": ["kv_cache"]}, 30)
        assert code == 0
        assert len(srv.requests) == 1, srv.requests
    finally:
        await session.close()
        await srv.stop()


@sync
async def test_partial_response_is_never_retried():
    """The handler RAN.  Re-sending would apply the leg twice."""
    srv, session, front = await _front(["partial", "ok"])
    g = Group(name="D", url=srv.url)
    try:
        code, text = await front.leg_rpc(g, WAKE, {"tags": ["weights"], "epoch": "e2"}, 30)
        assert code == 0, text
        # MUTANT M1 (retry on a partial response): the server would see two.
        assert len(srv.requests) == 1, srv.requests
    finally:
        await session.close()
        await srv.stop()


@sync
async def test_retry_happens_exactly_once():
    """Every connection drops.  Exactly two attempts, never three."""
    srv, session, front = await _front(["drop_before_response"])
    g = Group(name="P", url=srv.url)
    try:
        code, text = await front.leg_rpc(g, WAKE, {"tags": ["weights"], "epoch": "e3"}, 30)
        assert code == 0 and "Disconnected" in text
        # MUTANT M2 (retry twice): three requests would reach the server.
        assert len(srv.requests) == 2, srv.requests
    finally:
        await session.close()
        await srv.stop()


@sync
async def test_non_leg_rpc_is_never_retried():
    srv, session, front = await _front(["drop_before_response"])
    g = Group(name="P", url=srv.url)
    try:
        code, _ = await front.rpc(g, "/flush_cache", None, 30)
        assert code == 0
        assert len(srv.requests) == 1
    finally:
        await session.close()
        await srv.stop()


# ------------------------------------------------------------- the instrument
@sync
async def test_issued_returned_lines_carry_every_named_field(caplog):
    srv, session, front = await _front(["ok"])
    g = Group(name="D", url=srv.url)
    try:
        with caplog.at_level(logging.INFO, logger="weg2.front"):
            code, _ = await front.rpc(g, WAKE, {"tags": ["weights"], "epoch": "e4"}, 30)
        assert code == 200
        issued = _lines(caplog, "WEG2-RPC ISSUED")
        returned = _lines(caplog, "WEG2-RPC RETURNED")
        assert len(issued) == 1 and len(returned) == 1
        for f in ("leg=wake", "group=D", f"path={WAKE}", "epoch=e4",
                  "conn=", "sport=", "pool_idle=", "pool_total="):
            assert f in issued[0], issued[0]
            assert f in returned[0], returned[0]
        assert "code=200" in returned[0] and "ms=" in returned[0]
        # conn/sport are RESOLVED on the returned line, never left pending
        assert "conn=new" in returned[0], returned[0]
        assert re.search(r"sport=\d+", returned[0]), returned[0]
    finally:
        await session.close()
        await srv.stop()


@sync
async def test_raised_and_retry_lines_carry_every_named_field(caplog):
    srv, session, front = await _front(["drop_before_response", "ok"])
    g = Group(name="P", url=srv.url)
    try:
        with caplog.at_level(logging.INFO, logger="weg2.front"):
            code, _ = await front.leg_rpc(g, SLEEP, {"tags": ["kv_cache"], "epoch": "e5"}, 30)
        assert code == 200
        raised = _lines(caplog, "WEG2-RPC RAISED")
        retry = _lines(caplog, "WEG2-RPC RETRY")
        assert len(raised) == 1 and len(retry) == 1
        for f in ("leg=sleep", "group=P", f"path={SLEEP}", "epoch=e5",
                  "conn=", "sport=n/a", "pool_idle=", "pool_total=",
                  "after_ms=", "retryable=True"):
            assert f in raised[0], raised[0]
        for f in ("leg=sleep", "group=P", "reason="):
            assert f in retry[0], retry[0]
    finally:
        await session.close()
        await srv.stop()


# ------------------------------------------- the far side: epoch-scoped dedup
#
# The retry above is only safe because the HANDLER makes it safe.  Measured on
# this tree, neither leg is idempotent: `resume_memory_occupation` opens with
# `offload_tags.remove(tag)` (KeyError on a repeat) and
# `release_memory_occupation` pauses unconditionally while deriving
# `sleep_begins` / `family_paused_before` from `len(self.offload_tags)`.
from dataclasses import dataclass  # noqa: E402

from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
    Weg2LegLedger,
)


@dataclass
class _Req:
    tags: list
    epoch: object = None


def _mgr():
    return SchedulerWeightUpdaterManager(
        tp_worker=None, draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


def test_no_epoch_means_no_dedup_key_stock_path_untouched():
    m = _mgr()
    assert m._weg2_leg_key("release", _Req(tags=["kv_cache"])) is None
    assert m._weg2_leg_replay("release", _Req(tags=["kv_cache"])) is None


def test_ledger_replays_only_the_same_op_epoch_and_tag_set():
    m = _mgr()
    req = _Req(tags=["weights", "kv_cache"], epoch="b7:f3")
    sentinel = object()
    assert m._weg2_leg_replay("release", req) is None
    m._weg2_leg_commit("release", req, sentinel)
    # same leg, tags in the other order -> the SAME leg
    assert m._weg2_leg_replay(
        "release", _Req(tags=["kv_cache", "weights"], epoch="b7:f3")) is sentinel
    # the other op, the other epoch, a different tag set -> all distinct legs
    assert m._weg2_leg_replay("resume", req) is None
    assert m._weg2_leg_replay(
        "release", _Req(tags=["weights", "kv_cache"], epoch="b7:f4")) is None
    assert m._weg2_leg_replay("release", _Req(tags=["weights"], epoch="b7:f3")) is None


def test_ledger_survives_a_slots_dataclass():
    """The trap this class has now hit three times: `slots=True` turns a
    lazily-assigned attribute into an AttributeError raised ONLY on the retry
    path -- after a failure, which is the worst place to learn it."""
    assert "weg2_leg_ledger" in SchedulerWeightUpdaterManager.__slots__
    m = _mgr()
    first = m._weg2_leg_ledger_obj()
    assert m._weg2_leg_ledger_obj() is first


def test_ledger_is_bounded():
    led = Weg2LegLedger(cap=2)
    for i in range(4):
        led.record(Weg2LegLedger.key("release", i, ["kv_cache"]), i)
    assert led.recorded(Weg2LegLedger.key("release", 0, ["kv_cache"])) is None
    assert led.recorded(Weg2LegLedger.key("release", 3, ["kv_cache"])) == 3


def test_both_handlers_guard_first_and_commit_every_return():
    """Structural, because the next `return` anyone adds must not skip the
    ledger: a leg that returns without recording is a leg the retry re-applies.
    """
    import ast
    import inspect

    src = inspect.getsource(SchedulerWeightUpdaterManager)
    tree = ast.parse("class C:\n" + "\n".join(
        "    " + ln for ln in src.splitlines()[1:]))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name, op in (("release_memory_occupation", "release"),
                     ("resume_memory_occupation", "resume")):
        fn = fns[name]
        body = [s for s in fn.body if not isinstance(s, ast.Expr)]
        first = body[0]
        assert isinstance(first, ast.Assign), (name, ast.dump(first)[:120])
        assert "replay" in ast.dump(first) and "_weg2_leg_replay" in ast.dump(first), name
        assert op in ast.dump(first), name
        rets = [n for n in ast.walk(fn)
                if isinstance(n, ast.Return) and n.value is not None]
        # every value-carrying return either replays or commits
        for r in rets:
            d = ast.dump(r)
            assert ("_weg2_leg_commit" in d) or ("replay" in d), (name, d[:200])


# ------------------------------------------------- the discriminator, direct
#
# WHY A FAKE HERE AND REAL SOCKETS EVERYWHERE ELSE.  Measured against aiohttp
# 3.14.1 on this tree: a peer that drops mid-body -- FIN or RST, both tried --
# raises `ClientPayloadError`, which is NOT a `ClientConnectionError`.  So on
# real sockets the exception CLASS alone already stops a retry, and the
# `not got_response` guard cannot be shown to carry any weight.  It carries
# weight against the shape aiohttp is not obliged to keep raising that way, and
# the only way to exercise that branch is to hand `_rpc_attempt` the shape
# directly.  A test that cannot fail on a broken guard is not a test.
class _RaisingResponse:
    status = 200
    _protocol = None

    async def read(self):
        raise ServerDisconnectedError("dropped while reading the body")


class _Ctx:
    async def __aenter__(self):
        return _RaisingResponse()

    async def __aexit__(self, *a):
        return False


class _SessionStub:
    connector = None

    def post(self, *a, **k):
        return _Ctx()


@sync
async def test_a_connection_error_after_the_response_line_is_not_retryable():
    front = FrontStub(_SessionStub())
    g = Group(name="P", url="http://127.0.0.1:1")
    code, text, retryable = await front._rpc_attempt(
        _SessionStub(), g, SLEEP, {"tags": ["kv_cache"], "epoch": "e9"}, 5)
    assert code == 0 and "Disconnected" in text
    # MUTANT M1 (drop the `not got_response` guard): this flips to True and the
    # front would re-apply a leg whose handler has already run.
    assert retryable is False
