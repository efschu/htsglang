"""Q-1368 (04.10.): a LEG1-EARLY whose 503 lands while its request is still
queued was never read.

NF nf9 (boot ..._0c6e856785_1004_105315, front log): LEG1-EARLY rid=pdflip-84-299
epoch=84 posted at the D->P begin (11:30:37); PP0 answered it PDFLIP-INTAKE-STALL
503 (11:31:29); the P drain ended on its cap (11:31:32 P-DRAIN epoch=85
prefilled=1 queue_at_exit=8) without reaching the rid -- no requeue, no
/abort_request, the client hung until the W3-STOP (11:31:44), and at 11:33:38
"Task exception was never retrieved ... leg1 on P returned 503". The only
reader was ``one(p)`` of the drain (front.py ``await _early``), and the
consumed future was never cleared: a later phase re-awaited the same failed
leg, requeued it again and never posted a fresh one (x_requeues unbounded).

Pinned (red before): the early leg's verdict is read while its request is
queued (stall -> requeue at the head + /abort_request on P; anything else ->
the client's future); the drain's one(p) consumes it exactly once; hang-up /
STOP cancel a leg still in flight (abort on P for the hang-up); the requeue
count is bounded. Hermetic, no CUDA.
"""
from __future__ import annotations

import asyncio
import collections
import gc
import inspect
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402

STALL = ("leg1 on P returned 503: b'{\"error\": \"PDFLIP-INTAKE-STALL rid=pdflip-84-299 "
         "need_tokens=25408 rem_total_tokens=15680 gate=adder_no_token\"}'")


class _Front:
    """The real Front methods on a small double (the controller does not run)."""

    _requeue_intake_stalled = F.Front._requeue_intake_stalled
    _client_state = F.Front._client_state
    _on_client_gone = F.Front._on_client_gone

    def __init__(self):
        self.queue = collections.deque()
        self._ready_for_d = collections.deque()
        self.counters = collections.Counter()
        self.groups = {"P": SimpleNamespace(name="P", url="http://p", outstanding={}),
                       "D": SimpleNamespace(name="D", url="http://d", outstanding={})}
        self._p_intake_stalled = False
        self.epoch, self.state, self.awake = 84, "flipping", "D"
        self.calls = []
        self.rpc_gate = None

    async def rpc(self, g, path, payload, timeout):
        self.calls.append((g.name, path, payload))
        if self.rpc_gate is not None:
            await self.rpc_gate.wait()
        return 200, b"{}"


def _pending(rid, loop):
    return F.Pending(rid=rid, path="/v1/chat/completions", payload={}, text="", t_arrive=time.time(),
                     fut=loop.create_future(), est_prompt=25408)


def _post_early(f, p, leg):
    """The flip's post site: the task, then (with the fix) its watch."""
    p._leg1_early = asyncio.ensure_future(leg)
    watch = getattr(F.Front, "_leg1_early_watch", None)
    if watch is not None:
        watch(f, p)
    return p._leg1_early


async def _fail(msg, delay=0.01):
    await asyncio.sleep(delay)
    raise RuntimeError(msg)


async def _settle(n=10):
    for _ in range(n):
        await asyncio.sleep(0.01)


def test_early_503_while_queued_is_requeued_and_aborted_without_the_drain():
    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        older, p = _pending("pdflip-79-295", loop), _pending("pdflip-84-299", loop)
        f.queue.extend([older, p])          # the drain ended on its cap: nobody dispatches p
        task = _post_early(f, p, _fail(STALL))
        await _settle()
        assert list(f.queue)[0] is p and list(f.queue)[1] is older, [q.rid for q in f.queue]
        assert p.intake_stalled and p.x_requeues == 1 and f._p_intake_stalled
        assert ("P", "/abort_request", {"rid": "pdflip-84-299"}) in f.calls, f.calls
        assert not p.fut.done()                         # the client waits for the next P phase
        assert p._leg1_early is None                    # consumed: the next phase posts a fresh leg 1
        assert [q.rid for q in F.leg1_early_candidates(f.queue, 1)] == ["pdflip-84-299"]
        assert f.counters["leg1_early_unread"] == 1
        assert task._log_traceback is False             # read: no "never retrieved"
    asyncio.run(case())


def test_a_drain_taking_p_waits_for_the_abort_of_its_early_leg():
    """The abort of the unread early leg is on the wire when the next drain
    takes p: its fresh leg 1 must not reach P before the abort does (the
    abort would kill it). one(p) waits on ``p._leg1_abort``."""
    src = inspect.getsource(F.Front)
    j = src.index("async def one(p: Pending) -> Pending:")
    body = src[j:j + 2500]
    k = body.index("if not await Front._leg1_abort_landed(self, p):")
    assert k < body.index("await self.leg1(p)", k)

    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        f.rpc_gate = asyncio.Event()
        p = _pending("pdflip-84-299", loop)
        f.queue.append(p)
        _post_early(f, p, _fail(STALL))
        await _settle()
        ab = getattr(p, "_leg1_abort", None)
        assert ab is not None and not ab.done() and f.calls[-1][1] == "/abort_request"
        f.queue.popleft()                               # the next drain takes p
        landed = asyncio.ensure_future(F.Front._leg1_abort_landed(f, p))
        await _settle(3)
        assert not landed.done(), "the fresh leg 1 went out before the abort landed"
        f.rpc_gate.set()
        assert await landed is True and ab.done() and p._leg1_abort is None
    asyncio.run(case())


def test_early_failure_while_queued_answers_the_client():
    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        p = _pending("pdflip-9-1", loop)
        f.queue.append(p)
        _post_early(f, p, _fail("leg1 on P returned 500: b'boom'"))
        await _settle()
        assert len(f.queue) == 0 and p.fut.done()
        assert "returned 500" in str(p.fut.exception())
        assert f.counters["leg1_failures"] == 1 and f.calls == []
    asyncio.run(case())


def test_the_drain_that_holds_p_stays_the_reader():
    """p popped by the drain (its one(p) awaits the leg): the callback does
    nothing, one(p)'s handler requeues it once."""
    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        p = _pending("pdflip-84-299", loop)
        task = _post_early(f, p, _fail(STALL))
        p._leg1_early = None                            # one(p) consumes it before awaiting
        try:
            await task
        except RuntimeError:
            pass
        await _settle()
        assert len(f.queue) == 0 and f.calls == [] and f.counters["leg1_early_unread"] == 0
    asyncio.run(case())


def test_the_drain_consumes_the_early_leg_once():
    src = inspect.getsource(F.Front)
    j = src.index("async def one(p: Pending) -> Pending:")
    body = src[j:j + 1500]
    k = body.index('_early = getattr(p, "_leg1_early", None)')
    assert body.index("p._leg1_early = None", k) < body.index("await _early", k)
    flip = inspect.getsource(F.Front.flip)
    assert "Front._leg1_early_watch(self, _ep)" in flip
    gate = inspect.getsource(F.Front._early_flip_gate)
    assert "Front._leg1_early_watch(self, _ep)" in gate


def test_hang_up_cancels_the_early_leg_and_aborts_it_on_p():
    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        p = _pending("pdflip-5-5", loop)
        f.queue.append(p)

        async def held():                               # dormant P holds the leg
            f.groups["P"].outstanding[p.rid] = time.time()
            try:
                await asyncio.sleep(30)
            finally:
                f.groups["P"].outstanding.pop(p.rid, None)
        task = _post_early(f, p, held())
        await _settle(2)
        await f._on_client_gone(p.rid, p)
        await _settle(2)
        assert task.cancelled() and p._leg1_early is None
        assert ("P", "/abort_request", {"rid": "pdflip-5-5"}) in f.calls, f.calls
        assert p.rid not in f.groups["P"].outstanding and len(f.queue) == 0
        assert isinstance(p.fut.exception(), F.PdFlipClientGone)
    asyncio.run(case())


def test_stop_cancels_an_early_leg_in_flight():
    src = inspect.getsource(F.Front.do_stop)
    i = src.index("p.fut.set_exception(self.stop)")
    assert 'Front._leg1_early_cancel(self, p, "stop")' in src[i:src.index("self.queue.clear()")]

    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        p = _pending("pdflip-6-6", loop)
        task = _post_early(f, p, asyncio.sleep(30))
        assert F.Front._leg1_early_cancel(f, p, "stop") is False   # not in P.outstanding
        await _settle(2)
        assert task.cancelled() and p._leg1_early is None
    asyncio.run(case())


def test_the_requeue_is_bounded():
    async def case():
        loop = asyncio.get_running_loop()
        f = _Front()
        p = _pending("pdflip-7-7", loop)
        p.x_requeues = F.X_REQUEUE_CAP
        await f._requeue_intake_stalled(p, RuntimeError(STALL))
        assert len(f.queue) == 0 and p.fut.done() and "P-INTAKE-STALL-CAP" in str(p.fut.exception())
        assert f.calls == [("P", "/abort_request", {"rid": "pdflip-7-7"})]   # still aborted on every rank
        q = _pending("pdflip-7-8", loop)
        q.x_requeues = F.X_REQUEUE_CAP - 1
        await f._requeue_intake_stalled(q, RuntimeError(STALL))
        assert list(f.queue) == [q] and not q.fut.done()
    asyncio.run(case())
    gc.collect()
