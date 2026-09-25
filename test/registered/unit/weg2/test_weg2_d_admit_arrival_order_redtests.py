"""Law 2 (user 2026-09-07): D admits the OLDEST requests first -- also when
leg 1 finishes out of order.

27B redtests (base e15e7602b7): since #1459c the P drain pool hands finished
legs to ``_on_leg1_done`` in COMPLETION order (and within one
``asyncio.wait`` batch in set order), and ``_ready_for_d.append`` turned that
into D's admission order. Under dormant admit (#1443) the admitter also took a
younger request whose leg 1 had ended while an OLDER one was still prefilling
on P, so at the flip the older request found D's seats taken. Slice A's T3 is
the end-to-end pin (red on the base: D admitted r03, r02, r04, r05, r00, r06);
these two are its deterministic halves, no HTTP.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, bodies run by ``asyncio.run``.
"""

import asyncio
import time

from sglang.srt.weg2.front import Front, Pending
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pending(rid: str, t_arrive: float) -> Pending:
    return Pending(rid, "/generate", {}, "x", t_arrive,
                   asyncio.get_event_loop().create_future())


async def _until(pred, timeout: float = 5.0, tick: float = 0.02) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(tick)
    return False


def test_ready_for_d_is_kept_in_arrival_order_not_completion_order():
    async def body():
        f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, d_bs=2)
        ps = [_pending(f"r{i}", 100.0 + i) for i in range(6)]
        for i in (3, 1, 4, 0, 5, 2):  # leg 1 finished in this order
            f._queue_ready_for_d(ps[i])
        assert [p.rid for p in f._ready_for_d] == [f"r{i}" for i in range(6)]
        # equal arrival times keep their completion order (stable)
        tie = _pending("tie", 102.0)
        f._queue_ready_for_d(tie)
        assert [p.rid for p in f._ready_for_d].index("tie") == 3

    asyncio.run(body())


def test_a_younger_ready_request_waits_for_an_older_leg_in_flight():
    async def body():
        f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, d_bs=2)
        f.dormant_admit = True  # #1443: D takes leg 2 behind an awake P
        assert f._d_accepts_leg2()
        old, young = _pending("old", 100.0), _pending("young", 200.0)
        f._leg1_inflight[old.rid] = old.t_arrive  # still prefilling on P
        f._queue_ready_for_d(young)
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        try:
            await asyncio.sleep(0.4)
            assert not young.fut.done(), (
                "a younger request took a D seat while an older one was still on P")
            assert list(f._ready_for_d) == [young], "held, not popped (do_stop reaches it)"
            # the older leg ends: it is queued AHEAD of the younger one and
            # admitted first
            f._leg1_inflight.pop(old.rid)
            f._queue_ready_for_d(old)
            f._sync_batch_gate()
            assert await _until(lambda: old.fut.done()), "the older request was never admitted"
            assert not young.fut.done(), "admitted in completion order, not arrival order"
            old.posted_evt.set()
            assert await _until(lambda: young.fut.done()), "the younger one must follow"
            young.posted_evt.set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


def test_a_request_waiting_in_the_queue_does_not_hold_d_back():
    """Only legs IN FLIGHT hold a younger request back: an older request that
    is back in the queue (intake-stall requeue) would otherwise block D's seats
    until the next P phase."""

    async def body():
        f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0, d_bs=2)
        f.dormant_admit = True
        requeued, young = _pending("requeued", 100.0), _pending("young", 200.0)
        f.queue.appendleft(requeued)
        f._queue_ready_for_d(young)
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        try:
            assert await _until(lambda: young.fut.done()), (
                "a queued (not in-flight) older request blocked D admission")
            young.posted_evt.set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())
