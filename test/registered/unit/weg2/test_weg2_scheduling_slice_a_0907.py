"""WEG2_SCHEDULING_SPEC_0907 slice A (laws 1, 2, 4-knob, 5) -- T1..T10.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU, no launcher side
effects.  The P and D groups are real aiohttp test servers so the front's
HTTP seams (leg 1, leg 2, the flip RPCs, the absence of a
``/get_server_info`` round trip) are exercised as they ship; the scheduler
half is driven through the two new pure methods with a stub ``self``.

There is no pytest-asyncio in this venv, so every async test body is run by
``asyncio.run`` from a plain test function.
"""

import asyncio
import inspect
import logging
import re
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from aiohttp import ClientSession, ClientTimeout, web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import launcher as launcher_mod
from sglang.srt.weg2.front import (Front, Pending, Seat, double_prefill_verdict,
                                   is_x_refusal)


# ------------------------------------------------------------------ fakes
class FakeGroup:
    """A stand-in for group P or group D.

    Records an ordered timeline of everything the front asks it to do, so a
    test can assert on ORDER (all 16 prefills before the first flip) and on
    CONCURRENCY (never more than 4 leg-1 POSTs in flight) rather than only
    on totals.
    """

    def __init__(self, name: str, delay: float = 0.02, info_status: int = 200):
        self.name = name
        self.delay = delay
        self.info_status = info_status
        self.timeline: List[str] = []
        self.gen_marks: List[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.info_hits = 0
        self.hold: Optional[Dict[str, asyncio.Event]] = None
        self.refuse_x_for: Dict[str, int] = {}
        #: FIX 2: the IN-BAND shape -- a 200 whose FIRST chunk carries the
        #: W31 abort, which is what tokenizer_manager.py:1518-1537 emits for
        #: a STREAMED request (the non-stream leg raises HTTPException(503)).
        self.refuse_x_inband_for: Dict[str, int] = {}
        self.x_refusals: List[str] = []
        self.server: Optional[TestServer] = None
        self.url = ""

    def _mark(self, payload: dict) -> str:
        text = front_mod.request_text(payload)
        m = re.search(r"MARK(\S+)", text)
        return m.group(1) if m else "?"

    async def _generate(self, request: web.Request) -> web.Response:
        payload = await request.json()
        mark = self._mark(payload)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self.timeline.append(f"gen:{mark}")
        self.gen_marks.append(mark)
        try:
            left = self.refuse_x_for.get(mark, 0)
            if left > 0:
                self.refuse_x_for[mark] = left - 1
                self.x_refusals.append(mark)
                return web.json_response(
                    {"error": f"W50 Weg2TpPrefillExceeded rid=? uncached=99999"}, status=503
                )
            if payload.get("stream"):
                inband = self.refuse_x_inband_for.get(mark, 0)
                resp = web.StreamResponse(status=200)
                resp.content_type = "text/event-stream"
                await resp.prepare(request)
                if inband > 0:
                    self.refuse_x_inband_for[mark] = inband - 1
                    self.x_refusals.append(mark)
                    await resp.write(
                        b'data: {"meta_info": {"finish_reason": {"type": "abort", '
                        b'"status_code": 503, "message": "W50 Weg2TpPrefillExceeded: '
                        b'this group may prefill at most 10 uncached tokens itself"}}}\n\n'
                    )
                else:
                    await resp.write(b'data: {"text": "x"}\n\n')
                    await resp.write(
                        b'data: {"meta_info": {"prompt_tokens": 100, '
                        b'"completion_tokens": 1, "cached_tokens": 100}}\n\n'
                    )
                await resp.write_eof()
                return resp
            if self.hold is not None:
                self.hold.setdefault(mark, asyncio.Event())
                await self.hold[mark].wait()
            else:
                await asyncio.sleep(self.delay)
            return web.json_response(
                {
                    "choices": [{"text": "x"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 100},
                    },
                }
            )
        finally:
            self.in_flight -= 1
            self.timeline.append(f"done:{mark}")

    async def _ok(self, request: web.Request) -> web.Response:
        self.timeline.append(f"rpc:{request.path.strip('/')}")
        return web.json_response({"ok": True})

    async def _info(self, request: web.Request) -> web.Response:
        self.info_hits += 1
        return web.json_response({"x": 1}, status=self.info_status)

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/generate", self._generate)
        app.router.add_post("/v1/completions", self._generate)
        app.router.add_post("/flush_cache", self._ok)
        app.router.add_post("/release_memory_occupation", self._ok)
        app.router.add_post("/resume_memory_occupation", self._ok)
        app.router.add_get("/health", self._ok)
        app.router.add_get("/get_server_info", self._info)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.close()

    def release_all(self) -> None:
        for ev in (self.hold or {}).values():
            ev.set()
        # and stop holding NEW arrivals -- a refilled seat must be able to
        # complete, or the test measures its own scaffolding.
        self.hold = None

    def release(self, mark: str) -> None:
        assert self.hold is not None
        self.hold.setdefault(mark, asyncio.Event()).set()


class Harness:
    """Front + both fake groups + a client, all on the running loop."""

    def __init__(self, **front_kwargs):
        self.front_kwargs = front_kwargs
        self.p = FakeGroup("P")
        self.d = FakeGroup("D")
        self.front: Optional[Front] = None
        self.tasks: List[asyncio.Task] = []
        self.server: Optional[TestServer] = None
        self.client: Optional[ClientSession] = None
        self.posts: List[asyncio.Task] = []

    async def __aenter__(self) -> "Harness":
        await self.p.start()
        await self.d.start()
        kw = dict(
            awake="D", tag="t", store_dir="", prefill_sid=0, decode_sid=0,
            dc_reserve={}, w_s=45.0,
        )
        kw.update(self.front_kwargs)
        self.front = Front(self.p.url, self.d.url, **kw)
        self.front.session = ClientSession(timeout=ClientTimeout(total=30))
        app = web.Application(client_max_size=1024 ** 3)
        app.router.add_post("/generate", self.front.handle_generate)
        app.router.add_post("/v1/completions", self.front.handle_generate)
        app.router.add_get("/weg2/state", self.front.handle_state)
        self.server = TestServer(app)
        await self.server.start_server()
        self.client = ClientSession(timeout=ClientTimeout(total=30))
        self.tasks = [
            asyncio.create_task(self.front.controller()),
            asyncio.create_task(self.front.d_admitter()),
        ]
        return self

    async def __aexit__(self, *exc) -> None:
        for t in self.posts:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self.posts, return_exceptions=True)
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.client is not None:
            await self.client.close()
        if self.front is not None and self.front.session is not None:
            await self.front.session.close()
        if self.server is not None:
            await self.server.close()
        await self.p.stop()
        await self.d.stop()

    def post(self, mark: str, chars: int = 3000, stream: bool = False) -> asyncio.Task:
        body = {"prompt": f"MARK{mark} " + f"{mark}" * chars}
        if stream:
            body["stream"] = True
        t = asyncio.create_task(self._post(body))
        self.posts.append(t)
        return t

    async def _post(self, body: dict):
        async with self.client.post(str(self.server.make_url("/generate")), json=body) as r:
            return r.status, await r.text()


async def _until(pred, timeout: float = 10.0, tick: float = 0.02) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(tick)
    return False


# ------------------------------------------------------------- T1, law 1
def test_t1_p_drains_the_whole_backlog_before_the_flip_at_p_concurrency():
    """T1: 16 queued at --p-concurrency 4 -> all 16 prefilled BEFORE the
    first flip(P,D), at most 4 leg-1 POSTs in flight, mid-drain arrivals
    included."""

    async def body():
        async with Harness(awake="P", p_concurrency=4, d_bs=8,
                           tp_prefill_max_tokens=10, idle_layout="D") as h:
            tasks = [h.post(f"a{i}") for i in range(13)]
            await asyncio.sleep(0.25)
            tasks += [h.post(f"b{i}") for i in range(3)]  # arrive mid-drain
            assert await _until(lambda: h.front.awake == "D", timeout=20)
            for t in tasks:
                await t
            first_flip = next(i for i, e in enumerate(h.p.timeline)
                              if e.startswith("rpc:release_memory_occupation"))
            leg1_marks = [e for e in h.p.timeline[:first_flip] if e.startswith("gen:")]
            assert len(leg1_marks) == 16, h.p.timeline[:40]
            assert h.p.peak_in_flight <= 4, h.p.peak_in_flight
            assert not h.front.queue
            assert not h.front._ready_for_d

    asyncio.run(body())


# ------------------------------------------------- T2, R-6 (no HTTP derivation)
def test_t2_the_front_is_told_the_two_bs_numbers_and_never_asks():
    """T2: both groups answer 500 on /get_server_info and the front still
    has the right bs values -- because the launcher wrote them (C2/R-6)."""

    async def body():
        async with Harness(awake="D", p_concurrency=4, d_bs=6,
                           tp_prefill_max_tokens=10) as h:
            h.p.info_status = 500
            h.d.info_status = 500
            await asyncio.sleep(0.3)
            assert h.front.p_concurrency == 4
            assert h.front.d_bs == 6
            assert h.p.info_hits == 0 and h.d.info_hits == 0

    asyncio.run(body())


# ------------------------------------------------------------- T3, law 2
def test_t3_d_admits_oldest_first_with_its_own_bs_and_refills_a_freed_seat():
    """T3: 10 queued, --d-bs 6 -> exactly 6 leg-2 POSTs in flight, in
    t_arrive order; request 7 posts only after one of the first 6 ends."""

    async def body():
        async with Harness(awake="P", p_concurrency=8, d_bs=6,
                           tp_prefill_max_tokens=10, idle_layout="D") as h:
            h.d.hold = {}
            # P is deliberately slow so all ten arrive INSIDE one P drain --
            # law 1 says that phase ends on an empty queue, so a fast P
            # would flip after the first arrival and the rest would wait for
            # a second P phase that D's held decodes never allow.
            h.p.delay = 0.35
            tasks = []
            for i in range(10):
                tasks.append(h.post(f"r{i:02d}"))
                await asyncio.sleep(0.03)  # a definite arrival order
            assert await _until(lambda: len(h.d.gen_marks) >= 6, timeout=25)
            await asyncio.sleep(0.5)
            assert len(h.d.gen_marks) == 6, h.d.gen_marks
            assert h.d.gen_marks == sorted(h.d.gen_marks), h.d.gen_marks
            assert h.d.gen_marks == [f"r{i:02d}" for i in range(6)]
            assert h.front.seats_free() == 0
            h.d.release("r00")
            assert await _until(lambda: len(h.d.gen_marks) >= 7, timeout=15)
            assert h.d.gen_marks[6] == "r06"
            h.d.release_all()
            for t in tasks:
                await t

    asyncio.run(body())


# ------------------------------------------------ T4, law 2 at the launcher
def test_t4_the_two_bs_values_are_per_group_argv_and_not_in_common_flags():
    """T4: --p-bs 4 --d-bs 6 -> one --max-running-requests per group with
    the right value, NONE in common_flags."""
    common = launcher_mod.common_flags("/m", 8, 512, 16.0, 262144)
    assert "--max-running-requests" not in common
    ap = launcher_mod.argv_p("py", "/m", [1, 2, 3], 8, 512, 16.0, [], 4, 262144)
    ad = launcher_mod.argv_d("py", "/m", [1, 2, 3], 8, 512, 16.0, [], 6, 262144, 22000)
    assert ap.count("--max-running-requests") == 1
    assert ad.count("--max-running-requests") == 1
    assert ap[ap.index("--max-running-requests") + 1] == "4"
    assert ad[ad.index("--max-running-requests") + 1] == "6"
    assert ad[ad.index("--tp-prefill-max-tokens") + 1] == "22000"
    assert "--tp-prefill-max-tokens" not in ap
    assert common.count("--max-kv-per-request") == 1


# ------------------------------------------------------------- T5, R-15
def test_t5_a_dead_client_expires_the_barrier_and_does_not_stall_the_queue():
    """T5: the resolved request's client never posts -> the barrier expires
    by POST_BARRIER_S, W36 is counted, the seat is released and the rest of
    the queue is served."""

    async def body():
        old = front_mod.POST_BARRIER_S
        front_mod.POST_BARRIER_S = 0.2
        try:
            async with Harness(awake="D", p_concurrency=8, d_bs=2,
                               tp_prefill_max_tokens=10) as h:
                loop = asyncio.get_event_loop()
                dead = Pending("dead", "/generate", {}, "x", time.time(), loop.create_future())
                h.front._ready_for_d.append(dead)
                h.front._sync_batch_gate()
                live = []
                for i in range(3):
                    p = Pending(f"live{i}", "/generate", {}, "x", time.time(), loop.create_future())
                    h.front._ready_for_d.append(p)
                    live.append(p)
                h.front._sync_batch_gate()
                # every live request answers its hand-off at once
                async def responder(p: Pending):
                    await p.fut
                    front_mod.Front._mark_posted(p)
                    if p.seat is not None:
                        p.seat.release("test_done")
                waiters = [asyncio.create_task(responder(p)) for p in live]
                assert await _until(
                    lambda: h.front.counters.get("W36_Weg2AdmitterBarrierExpired", 0) >= 1,
                    timeout=10)
                await asyncio.wait_for(asyncio.gather(*waiters), timeout=10)
                assert h.front.seats_free() == h.front.d_bs
        finally:
            front_mod.POST_BARRIER_S = old

    asyncio.run(body())


# ------------------------------------------------------------- T6, R-16
def test_t6_a_short_arrival_does_not_take_a_seat_ahead_of_queued_batch_work():
    """T6: with _ready_for_d non-empty a SHORT arrival waits on the explicit
    gate -- a bare Semaphore is FIFO among waiters and would let it in."""

    async def body():
        async with Harness(awake="D", p_concurrency=8, d_bs=4,
                           tp_prefill_max_tokens=10) as h:
            loop = asyncio.get_event_loop()
            for i in range(2):
                h.front._ready_for_d.append(
                    Pending(f"b{i}", "/generate", {}, "x", time.time(), loop.create_future()))
            h.front._sync_batch_gate()
            assert not h.front._batch_gate.is_set()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(h.front._acquire_short_seat("short-1"), 0.3)
            h.front._ready_for_d.clear()
            h.front._sync_batch_gate()
            seat = await asyncio.wait_for(h.front._acquire_short_seat("short-2"), 2.0)
            assert seat is not None and seat.held
            seat.release("test")

    asyncio.run(body())


# --------------------------------------------------------------- T7, R-2
def test_t7_idle_layout_pp_does_not_lose_the_requests_p_just_prefilled():
    """T7: --idle-layout pp, drain 5, queue empty -> all 5 futures resolve
    and _ready_for_d is empty at rest.  This is the LOST-REQUEST class the
    design's own law-5 guard introduced."""

    async def body():
        async with Harness(awake="P", p_concurrency=8, d_bs=8,
                           tp_prefill_max_tokens=10, idle_layout="P") as h:
            tasks = [h.post(f"z{i}") for i in range(5)]
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
            assert [s for s, _ in results] == [200] * 5
            assert not h.front._ready_for_d
            assert h.front._batch_gate.is_set()

    asyncio.run(body())


# --------------------------------------------------------------- T8, law 5
def test_t8_idle_layout_decides_all_three_rest_cases():
    """T8: tp + empty queue + P awake -> flips to D; pp + empty queue + D
    awake + not outstanding + dwell expired -> flips to P; pp + P awake +
    empty queue -> stays on P."""

    async def case_tp():
        async with Harness(awake="P", p_concurrency=8, d_bs=8,
                           tp_prefill_max_tokens=10, idle_layout="D") as h:
            assert await _until(lambda: h.front.awake == "D", timeout=10)

    async def case_pp_mirror():
        async with Harness(awake="D", p_concurrency=8, d_bs=8,
                           tp_prefill_max_tokens=10, idle_layout="P",
                           min_dwell_ms=0.0) as h:
            assert await _until(lambda: h.front.awake == "P", timeout=10)

    async def case_pp_rest():
        async with Harness(awake="P", p_concurrency=8, d_bs=8,
                           tp_prefill_max_tokens=10, idle_layout="P") as h:
            await asyncio.sleep(1.0)
            assert h.front.awake == "P"

    asyncio.run(case_tp())
    asyncio.run(case_pp_mirror())
    asyncio.run(case_pp_rest())


# --------------------------------------------------------------- T9, law 4
def _stub_scheduler(x: int, host_carry: int, tp_size: int = 1):
    from sglang.srt.managers.scheduler import Scheduler

    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        ps=SimpleNamespace(tp_size=tp_size),
        tree_cache=SimpleNamespace(
            cache_controller=SimpleNamespace(mem_pool_host=SimpleNamespace(size=host_carry))
        ),
    )
    # bind the REAL method bodies back onto the stub, so the arithmetic and
    # the host-carry read under test are the shipped ones, not doubles.
    stub.weg2_uncached_extent = lambda req, head=None: Scheduler.weg2_uncached_extent(
        stub, req, head
    )
    stub._weg2_host_carry_tokens = lambda: Scheduler._weg2_host_carry_tokens(stub)
    return Scheduler, stub


def _stub_req(prompt_tokens: int, prefix: int = 0, host_hit: int = 0, rid: str = "r"):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(prompt_tokens)),
        prefix_indices=list(range(prefix)),
        host_hit_length=host_hit,
    )


def test_t9a_the_d_side_x_gate_binds_on_the_real_extent_at_the_boundary():
    """T9: X=10,000 -> uncached 9,999 admits, 10,001 raises W31.  And the
    measured weg2zr2 shape: a prompt the front priced SHORT whose realised
    uncached extent is 19,401 is REFUSED, not served through a grant."""
    Scheduler, stub = _stub_scheduler(10000, host_carry=30518)
    assert Scheduler._weg2_x_refuses(stub, _stub_req(9999)) is False
    assert Scheduler._weg2_x_refuses(stub, _stub_req(10001)) is True
    # the front's estimate said "remainder 0"; the realised extent is 19,401
    assert Scheduler._weg2_x_refuses(stub, _stub_req(19401, rid="weg2-4-8")) is True
    # a prefix already in the store is NOT uncached: the flip's own request
    # must stay admissible or the round trip can never complete.
    assert Scheduler._weg2_x_refuses(
        stub, _stub_req(19401, prefix=1000, host_hit=18000)) is False
    assert Scheduler.weg2_uncached_extent(stub, _stub_req(19401, prefix=1000, host_hit=18000)) == 401


def test_t9b_x_zero_is_off_and_the_carrier_wall_is_exempt():
    """The default path is untouched (X=0 = off), and a prompt no store read
    could ever cover is exempt rather than bounced around the wall (R-3)."""
    Scheduler, off = _stub_scheduler(0, host_carry=30518)
    assert Scheduler._weg2_x_refuses(off, _stub_req(84027)) is False
    Scheduler, on = _stub_scheduler(10000, host_carry=30518)
    assert Scheduler._weg2_x_refuses(on, _stub_req(84027)) is False  # > host carry
    assert Scheduler._weg2_x_refuses(on, _stub_req(30000)) is True   # under it: refused


def test_t9c_the_front_requeues_a_w31_once_and_then_raises_w35():
    """T9 front half: a W31 body re-joins route BATCH exactly once; a second
    W31 on the same rid is W35 by name, never a third pass."""

    async def body():
        async with Harness(awake="D", p_concurrency=8, d_bs=4,
                           tp_prefill_max_tokens=10 ** 9, flip_min_work_tokens=1,
                           idle_layout="D") as h:
            h.d.refuse_x_for["q1"] = 1  # refuse once, then serve
            status, _ = await asyncio.wait_for(h.post("q1"), timeout=30)
            assert status == 200
            assert h.front.counters["W50_Weg2TpPrefillExceeded"] == 1
            assert h.front.counters.get("W35_Weg2XReQueueLoop", 0) == 0
            assert h.p.gen_marks == ["q1"], h.p.gen_marks  # P prefilled it

            h.d.refuse_x_for["q2"] = 5  # refuse for ever
            status, text = await asyncio.wait_for(h.post("q2"), timeout=30)
            assert status == 503
            assert "W35 Weg2XReQueueLoop" in text
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 1
            assert h.d.x_refusals.count("q2") == 2  # never a third pass

    asyncio.run(body())


def test_t9d_all_four_front_sites_read_one_x():
    """C9: the front prices and routes against the SAME number."""
    assert double_prefill_verdict(20000, 0, 0, 22000) == "serve"
    assert double_prefill_verdict(30000, 0, 0, 22000) == "reroute"
    assert double_prefill_verdict(30000, 0, 1, 22000) == "W16"
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=22000)
    assert f._leg2_verdict(pt=20000, ct=0, priced=True, pending=None,
                           single_prefill=False, stream=False, rid="r") == "serve"
    assert f._leg2_verdict(pt=30000, ct=0, priced=True, pending=None,
                           single_prefill=False, stream=False, rid="r") == "short_mispriced"
    assert is_x_refusal(503, "W50 Weg2TpPrefillExceeded: ...") is True
    assert is_x_refusal(503, "W28 Weg2Leg2Unpriced") is False
    assert is_x_refusal(200, "W50 Weg2TpPrefillExceeded") is False


# -------------------------------------------------------------- T10, R-5
def test_t10_a_trickle_below_the_threshold_does_not_buy_a_round_trip():
    """T10, ADAPTED TO #1271 RULE (c): THE BACKLOG SUM IS OVER UNCACHED TOKENS.

    The behaviour asserted is unchanged -- queued work below
    --flip-min-work-tokens does not flip, crossing it or the fairness bound
    does, and the weg2zr2 4.3 s P phase must not recur. What changed is the
    QUANTITY the threshold is applied to: `_flip_economics_ok` summed
    `est_prompt`, which counts the cached head P never recomputes, so a backlog
    could flip on prefixes already resident. It now sums `est_uncached`.

    The old fixture set only `est_prompt`, leaving `est_uncached` at its default
    0, so under the new rule the backlog read 0 and the test went red. It is
    adapted rather than deleted, and STRENGTHENED to pin the rule it now
    depends on: the cached-head case below (huge est_prompt, tiny uncached) is
    the one #1271 exists to hold, and would flip under the old sum.
    """
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=22000)
    loop = asyncio.new_event_loop()
    try:
        mk = lambda est, unc=None: Pending(
            "r", "/generate", {}, "x", time.time(), loop.create_future(),
            est_prompt=est, est_uncached=est if unc is None else unc)
        f.queue.append(mk(5000))
        assert f._flip_economics_ok(fairness_fired=False) is False
        f.queue.append(mk(18000))
        assert f._flip_economics_ok(fairness_fired=False) is True
        f.queue.clear()
        f.queue.append(mk(10))
        assert f._flip_economics_ok(fairness_fired=False) is False
        assert f._flip_economics_ok(fairness_fired=True) is True

        # RULE (c) ITSELF: a backlog that is huge on est_prompt and tiny on
        # uncached must HOLD. On the pre-#1271 sum this flips.
        f.queue.clear()
        f.queue.append(mk(40000, unc=100))
        assert f._flip_economics_ok(fairness_fired=False) is False, \
            "a backlog of resident prefixes must not buy a round trip"
        # and the same est_prompt with the work actually uncached must flip
        f.queue.clear()
        f.queue.append(mk(40000, unc=40000))
        assert f._flip_economics_ok(fairness_fired=False) is True
    finally:
        loop.close()


def test_t10b_min_dwell_is_derived_from_the_last_flip_and_names_its_overrides():
    """C8/K7: dwell derived from the last completed flip in THAT direction,
    0 before the first, and two named overrides."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    assert f._derived_min_dwell_ms("D", "P") == (0.0, "none-first-flip")
    f.flip_log.append({"sleep": "D", "wake": "P", "flip_ms": 14000})
    assert f._derived_min_dwell_ms("D", "P") == (14000.0, "last-flip-D->P")
    f.t_awake = time.time()  # just woke: dwell not served
    assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=False,
                       oldest_wait_s=0.0) is False
    assert f._dwell_ok("D", "P", fairness_fired=True, work_exhausted=False,
                       oldest_wait_s=0.0) is True
    assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=True,
                       oldest_wait_s=99.0) is True
    g = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, min_dwell_ms=0.0)
    assert g._derived_min_dwell_ms("D", "P") == (0.0, "flag")


def test_a1_1_the_fairness_switch_is_named_and_zero_disables_it():
    """A1-1: the ONE sanctioned pre-emption, a named switch, 0 = off."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)
    assert f._fairness_switch(time.time() - 100.0, "batch") is True
    assert f.admit_d is False
    assert f.counters["fairness_bound_hits"] == 1
    off = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 0.0)
    assert off._fairness_switch(time.time() - 10_000.0, "batch") is False
    assert off.admit_d is True
    assert off.counters.get("fairness_bound_hits", 0) == 0


# ------------------------------------------------------- C10, X provenance
def test_c10_x_is_derived_with_a_provenance_line_naming_its_three_inputs():
    assert launcher_mod.derive_x_star(13.247, 690.0, 3640.0, 4096) == 22556
    assert launcher_mod.derive_x_star(13.247, 690.0, 3640.0, 30000) == 30000  # floor
    with pytest.raises(ValueError):
        launcher_mod.derive_x_star(13.247, 4000.0, 3640.0, 4096)  # no break-even
    x, prov = launcher_mod.resolve_x(None, "/nonexistent-evidence-dir", 4096)
    assert x == 22556
    assert "source=recorded PRE-BARLINK" in prov
    for term in ("flip_s=", "r_D=", "r_P=", "floor="):
        assert term in prov
    x2, prov2 = launcher_mod.resolve_x(9999, "/nonexistent-evidence-dir", 4096)
    assert x2 == 9999 and "source=flag" in prov2


def test_c10_x_prefers_this_rigs_own_measured_lines(tmp_path):
    """C10, ADAPTED TO #1271 RULE (a): r_P IS A GROUP THROUGHPUT, NOT A LATENCY.

    The behaviour asserted is unchanged -- X prefers this rig's own measured
    front log over the recorded PRE-BARLINK pair, and names it. What changed is
    what r_P MEANS: the leg-1 `wall` is one request's latency and the front
    drains P at p_concurrency=8, so `uncached/wall` charged a leg for the time
    it sat queued behind its peers. r_P now accumulates a drain's uncached
    tokens and divides by that drain's OWN `drain_s`, read from the new
    `WEG2 P-DRAIN` line -- the same shape as r_D.

    The old fixture carried no P-DRAIN line at all, so the log no longer
    carried "all three instruments" and resolve_x correctly fell through to the
    PRE-BARLINK fallback -- which is why this went red. The fixture is brought
    up to the contract, and the two leg-1 walls are deliberately DIFFERENT from
    drain_s so that the aggregate and the old per-request reading cannot
    coincide: this asserts the drain denominator, not just that a number came
    back.
    """
    log = tmp_path / "boot_weg2_x_0907_000000.front.log"
    log.write_text(
        "WEG2-FLIP done epoch=1 slept=D woke=P ... flip_total=13247 ms weights_tags=8 dc={}\n"
        "WEG2-FLIP done epoch=2 slept=P woke=D ... flip_total=13247 ms weights_tags=8 dc={}\n"
        "WEG2-SERVED group=P leg=1 rid=a prompt_tokens=30100 cached_tokens=0 wall=8.27s epoch=1\n"
        "WEG2-SERVED group=P leg=1 rid=c prompt_tokens=30100 cached_tokens=100 wall=9.00s epoch=1\n"
        "WEG2 P-DRAIN epoch=1 prefilled=2 drain_s=12.000\n"
        "WEG2-SERVED group=D leg=2 rid=b status=200 prompt_tokens=30100 cached_tokens=0 "
        "completion_tokens=5 uncached=30100 verdict=single_prefill wall=43.62s epoch=2\n"
    )
    x, prov = launcher_mod.resolve_x(None, str(tmp_path), 4096)
    assert "source=boot:" in prov and log.name in prov
    # r_P is the DRAIN aggregate: (30100 + 30000) uncached over drain_s=12.0,
    # NOT the median of 30100/8.27 and 30000/9.00.
    r_p_drain = (30100 + 30000) / 12.000
    assert x == launcher_mod.derive_x_star(13.247, 30100 / 43.62, r_p_drain, 4096)
    per_request = launcher_mod.derive_x_star(
        13.247, 30100 / 43.62, launcher_mod._median([30100 / 8.27, 30000 / 9.00]), 4096)
    assert x != per_request, "the drain aggregate must not coincide with the old latency reading"


# =====================================================================
# ROUND-1 FIXER -- the four findings, each pinned at its own seam.
# =====================================================================

def _bare_pending(rid: str) -> Pending:
    return Pending(rid, "/generate", {}, "x", time.time(),
                   asyncio.get_event_loop().create_future())


def test_f1a_the_batch_admitter_rechecks_the_phase_after_the_seat_acquire():
    """FINDING 1: `await self._d_seat.acquire()` blocks for the whole
    lifetime of a running decode, so the guard at the TOP of the admitter
    loop is arbitrarily stale by the time the seat is granted.  A request
    resolved there POSTs its leg 2 into a group that is flipping or asleep,
    and `leg2` re-registers it in `D.outstanding` in the middle of
    `drain(D)` -- W1, three times W2 STOP.

    RED at 5cde96fc7d: `AssertionError: BUG: d_admitter resolved a request
    into group D while state='flipping' awake='P'`.
    """

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=1)
        p1, p2 = _bare_pending("r1"), _bare_pending("r2")
        f._ready_for_d.extend([p1, p2])
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        assert await _until(lambda: p1.fut.done(), 5.0), "p1 never admitted"
        p1.posted_evt.set()
        # The admitter is now queued on the seat for p2.
        await asyncio.sleep(0.3)
        assert not p2.fut.done()
        # THE FLIP: state='flipping' (front.py:1097) and then awake='P'.
        f.state, f.awake = "flipping", "P"
        p1.seat.release("leg2_finished")
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not p2.fut.done(), (
            "d_admitter resolved a request into D while state=%r awake=%r"
            % (f.state, f.awake))
        assert f.counters["d_admit_phase_moved"] >= 1, "the re-check must be COUNTED"
        # and the request is still answerable and still oldest-first
        assert list(f._ready_for_d) == [p2]
        assert not f._batch_gate.is_set(), (
            "popping the last entry would open the batch gate and let a SHORT "
            "arrival take the seat the admitter is queued for (R-16 inverted)")

    asyncio.run(body())


def test_f1b_the_short_path_rechecks_the_phase_after_its_own_acquire():
    """FINDING 1, the other face: mutant M5 (deleting the SHORT path's
    post-acquire re-check at front.py:729-733) survived the whole suite, so
    the property was unpinned on BOTH paths.  Pinned here."""

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=1)
        held = await f._acquire_short_seat("busy")
        assert held is not None
        task = asyncio.create_task(f._acquire_short_seat("racer"))
        await asyncio.sleep(0.2)
        assert not task.done(), "the racer must be queued on the seat"
        f.state, f.awake = "flipping", "P"
        held.release("leg2_finished")
        seat = await asyncio.wait_for(task, 5.0)
        assert seat is None, "a SHORT seat must not be granted into a flipping group"
        assert f.seats_free() == f.d_bs, "the seat must be given back, not leaked"

    asyncio.run(body())


def test_f1c_a_stop_during_the_acquire_still_reaches_the_waiting_request():
    """FINDING 1, third face: `do_stop` answers `self.queue +
    self._ready_for_d` (front.py:538).  A request the admitter had already
    popped was in NEITHER, so a STOP raised while it waited for a seat never
    reached it -- its coroutine stayed on `await fut` behind a one-hour
    client timeout."""

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=1)
        p1, p2 = _bare_pending("r1"), _bare_pending("r2")
        f._ready_for_d.extend([p1, p2])
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        assert await _until(lambda: p1.fut.done(), 5.0)
        p1.posted_evt.set()
        await asyncio.sleep(0.3)
        f.do_stop("W99 TestStop", "a stop while the admitter waits for a seat")
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert p2.fut.done() and isinstance(p2.fut.exception(), front_mod.Weg2Stop)

    asyncio.run(body())


def test_f2a_a_streamed_leg2_is_requeued_on_the_503_w31_shape():
    """FINDING 2: `is_x_refusal` was consulted 27 lines BELOW the
    `if stream:` branch's own `return resp`, so law 4's re-route -- the half
    that makes W31 a policy rather than a kill -- existed only on the
    non-streamed leg.  The launcher passes `--tp-prefill-max-tokens` to
    argv_d on every boot and OpenAI chat completions under load are
    streamed, so this was the normal shape."""

    async def body():
        async with Harness(awake="D", p_concurrency=2, d_bs=4,
                           tp_prefill_max_tokens=10, idle_layout="D") as h:
            h.d.refuse_x_for["S1"] = 1
            t = h.post("S1", chars=40, stream=True)
            status, _text = await asyncio.wait_for(t, 20.0)
            assert status == 200, "the client must be SERVED, through P"
            assert h.front.counters["W50_Weg2TpPrefillExceeded"] == 1
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 0
            assert "S1" in h.p.gen_marks, "P must have prefilled the re-queued request"

    asyncio.run(body())


def test_f2b_a_streamed_leg2_is_requeued_on_the_inband_w31_shape():
    """FINDING 2, the second wire shape and the one the producer actually
    takes for a stream: `tokenizer_manager.py:1518-1537` raises
    HTTPException(503) only `if not is_stream`; otherwise the same abort is
    yielded as an IN-BAND chunk on a 200.  The refusal is read BEFORE
    `resp.prepare()`, where nothing is committed yet, so both shapes
    re-route identically."""

    async def body():
        async with Harness(awake="D", p_concurrency=2, d_bs=4,
                           tp_prefill_max_tokens=10, idle_layout="D") as h:
            h.d.refuse_x_inband_for["S2"] = 1
            t = h.post("S2", chars=40, stream=True)
            status, text = await asyncio.wait_for(t, 20.0)
            assert status == 200
            assert front_mod.X_REFUSAL_NAME not in text, (
                "the client must get the served answer, not D's refusal")
            assert h.front.counters["W50_stream_inband_requeued"] == 1
            assert h.front.counters["W50_Weg2TpPrefillExceeded"] == 1
            assert h.front.counters["W28_Weg2Leg2Unpriced_stream_served"] == 0, (
                "a refusal with a name must never land in W28")
            assert "S2" in h.p.gen_marks

    asyncio.run(body())


def test_f2c_a_late_inband_w31_is_counted_by_name_not_as_w28():
    """The W16 precedent's counterpart: after the first byte the re-route is
    impossible, so the refusal is COUNTED by name rather than absorbed as an
    unpriced stream.  W16 had `W16_stream_served`; W31 had none."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=10)
    p = _pending_for_verdict()
    assert f._leg2_verdict(0, 0, False, p, False, True, "r", x_inband=True) == "W50_stream_served"
    assert f.counters["W28_Weg2Leg2Unpriced_stream_served"] == 0
    # unchanged without the marker
    assert f._leg2_verdict(0, 0, False, p, False, True, "r") == "unpriced"
    assert f.counters["W28_Weg2Leg2Unpriced_stream_served"] == 1


def _pending_for_verdict() -> Pending:
    fut = asyncio.new_event_loop().create_future()
    return Pending("r", "/generate", {}, "x", time.time(), fut)


def test_f3_the_x_verdict_is_the_groups_or_it_is_not_taken():
    """FINDING 3: `len(req.prefix_indices)` is a match against THIS rank's
    radix tree, whose evictions differ per rank under D's uneven DCP.  The
    verdict then REPLACED that rank's waiting_queue and sent an abort that
    is a no-op off rank 0 -- a permanent, silent rank split.

    The extent now carries #823's MIN-reduced per-rid group match (no new
    collective: the third consumer of the reduce
    `_update_uniform_pool_budget` already runs pre-branch), and below
    `tp_size > 1` a rid the group has no opinion on is ABSTAINED on."""
    from sglang.srt.managers import tp_head_congruence as thc

    Scheduler, solo = _stub_scheduler(10000, host_carry=30518, tp_size=1)
    req = _stub_req(12000, prefix=4000, host_hit=0, rid="a")
    # tp_size == 1: nothing to reduce, today's arithmetic, unchanged.
    assert Scheduler.weg2_uncached_extent(solo, req) == 8000
    assert Scheduler._weg2_x_refuses(solo, req) is False

    Scheduler, grp = _stub_scheduler(10000, host_carry=30518, tp_size=3)
    # no group opinion -> ABSTAIN, never a rank-local refusal
    assert Scheduler._weg2_x_refuses(grp, req) is False
    assert grp._weg2_x_abstained == 1
    # the group's MIN says a peer matched only 500 of this rank's 4,000, so
    # the group's uncached extent is 11,500 and the verdict is W31 for ALL.
    head = thc.build_uniform_head_inputs(["a"], [500], None, True)
    assert Scheduler.weg2_uncached_extent(grp, req, head) == 11500
    assert Scheduler._weg2_x_refuses(grp, req, head) is True
    # agreement is byte-identical to the pre-fix arithmetic
    agree = thc.build_uniform_head_inputs(["a"], [4000], None, True)
    assert Scheduler.weg2_uncached_extent(grp, req, agree) == 8000
    assert Scheduler._weg2_x_refuses(grp, req, agree) is False
    # a rid some rank does not hold MIN-reduces to absent -> no group opinion
    absent = thc.build_uniform_head_inputs(["a"], [-1], None, True)
    assert thc.group_match_for(absent, "a") is None
    assert Scheduler._weg2_x_refuses(grp, req, absent) is False


def test_f3b_the_queue_filter_is_what_makes_a_split_permanent():
    """M6 (deleting the waiting_queue filter) survived 17/17.  The filter is
    load-bearing -- without it the refused request is re-offered every pass
    and never served -- and that is exactly why the verdict driving it has
    to be the group's."""
    from sglang.srt.managers.scheduler import Scheduler

    sent = []
    kept = [_stub_req(10, rid="keep"), _stub_req(10, rid="drop")]
    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=10),
        ps=SimpleNamespace(tp_size=3),
        waiting_queue=list(kept),
        enable_hicache_storage=False,
        enable_hierarchical_cache=False,
        tree_cache=SimpleNamespace(
            cache_controller=SimpleNamespace(mem_pool_host=SimpleNamespace(size=30518))
        ),
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=lambda a, b: sent.append(a))
        ),
    )
    stub.weg2_uncached_extent = lambda req, head=None: Scheduler.weg2_uncached_extent(
        stub, req, head)
    kept[1].time_stats = SimpleNamespace(
        trace_ctx=SimpleNamespace(abort=lambda abort_info=None: None))
    Scheduler._weg2_answer_x_refusals(stub, [kept[1]])
    assert [r.rid for r in stub.waiting_queue] == ["keep"]
    assert len(sent) == 1, "the refusal is answered BY NAME, never a silent skip"


def test_f4a_the_d_admitter_bounds_the_aggregate_host_staging_tokens():
    """FINDING 4, LINK 1 (boot weg2sc1's origin): C4 opens --d-bs seats at
    once and nothing coupled that count to D's host staging pool, which is a
    TOKEN budget.  Measured: three requests staged (occupied=25100 vs
    limit=27466), the fourth met `#915 PREFETCH REFUSED
    reason=vote_negative`; its prefix WAS in the store, so its whole prompt
    then priced as uncached, C11 refused it correctly for the wrong reason,
    and C12 re-queued a store-resident rid to P -- which is what W27 killed
    the group over."""

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                  d_bs=6, carrier_max_tokens=27466)
        # ROUND 4: the bound is no longer a front-side derivation at all --
        # `--d-admit-max-tokens` is an operator CEILING (unset here) and the
        # budget is group D's own #915 reading, stubbed below at the pool
        # size this boot measured.
        assert f.d_admit_max_tokens is None, "no front-side proxy for D's pool"
        t0 = time.time()

        async def _reading():
            return {"available": 27466, "occupied": 0, "limit": 27466,
                    "size": 30518, "threshold": 256, "pool_id": 1,
                    "phase": "TP", "generation": 1, "t": t0}

        f._d_pool_reading = _reading
        ps = []
        for i in range(4):
            p = _bare_pending(f"r{i}")
            p.est_prompt = 8400
            ps.append(p)
        f._ready_for_d.extend(ps)
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        for p in ps[:3]:
            # R-15's barrier: the admitter resolves one future, then waits for
            # that request's POST before the next.
            assert await _until(lambda p=p: p.fut.done(), 5.0), f"{p.rid} not admitted"
            p.posted_evt.set()
        await asyncio.sleep(0.4)
        assert not ps[3].fut.done(), (
            "3 x 8400 = 25200 of a 27466-token budget; the fourth does not fit "
            "and there are 3 free SEATS -- the count alone would admit it")
        assert f.seats_free() == 3
        assert f.counters["d_admit_token_held"] >= 1
        # a decode finishes -> the tokens come back with the seat
        ps[0].seat.release("leg2_finished")
        assert await _until(lambda: ps[3].fut.done(), 5.0), (
            "a freed seat's tokens must be refunded, or the bound wedges")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


def test_f4b_a_sole_request_is_never_starved_by_the_token_budget():
    """The bound only ever DELAYS an admission.  A request alone in flight
    is admitted whatever it costs -- the truly-oversized case belongs to
    CARRIER-EXCEEDS (R-3), not here."""

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                  d_bs=6, carrier_max_tokens=1000)
        t0 = time.time()

        async def _reading():
            # A pool with ten rows left: the request cannot possibly fit, and
            # it is admitted anyway BECAUSE NO SEAT IS IN USE -- with nothing
            # running on D nothing would ever free a row, so a refusal here
            # would wedge the queue rather than delay it.
            return {"available": 10, "occupied": 0, "limit": 1000,
                    "size": 1000, "threshold": 256, "pool_id": 1,
                    "phase": "TP", "generation": 1, "t": t0}

        f._d_pool_reading = _reading
        p = _bare_pending("huge")
        p.est_prompt = 900000
        f._ready_for_d.append(p)
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        assert await _until(lambda: p.fut.done(), 5.0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


def test_f4c_zero_disables_the_token_budget_and_it_is_a_flag():
    """MUST NOT 5/10: a knob, a derived default, a provenance line -- no env
    gate and no hand number.  0 restores the count-only admitter."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
              d_bs=6, carrier_max_tokens=27466, d_admit_max_tokens=0)
    assert f.d_admit_max_tokens == 0
    Seat(f, "running", "batch", tokens=10 ** 9)
    empty = {"available": 0, "occupied": 27466, "limit": 27466, "size": 30518,
             "threshold": 256, "pool_id": 1, "phase": "TP", "generation": 1,
             "t": time.time()}
    assert f._d_token_budget_blocks("r", 10 ** 9, empty) is False
    # and the launcher forwards it only when the operator set it, so the
    # derivation stays with the one place that knows D's pool size.
    src = launcher_mod.front_argv_for.__doc__ or ""
    assert "TOLD" in src
    argv = launcher_mod.front_argv_for(
        "py", "/tmp/store", 1, 2, {}, [],
        SimpleNamespace(tag="t", fairness_w_s=45.0, min_dwell_ms=None,
                        drain_deadline_s=120.0, d_admit_max_tokens=None),
        0, 27466, 4, 6, 10000, 10000, "D")
    assert "--d-admit-max-tokens" not in argv
    argv = launcher_mod.front_argv_for(
        "py", "/tmp/store", 1, 2, {}, [],
        SimpleNamespace(tag="t", fairness_w_s=45.0, min_dwell_ms=None,
                        drain_deadline_s=120.0, d_admit_max_tokens=0),
        0, 27466, 4, 6, 10000, 10000, "D")
    assert argv[argv.index("--d-admit-max-tokens") + 1] == "0"


# ------------------------------------------------------- ROUND 2, FINDING 1
def _head_stub(reqs, can_fast: bool = False, has_match_prefix: bool = True):
    """A Scheduler stand-in for the head vote, with the REAL method bound."""
    tree_fields = {"supports_fast_match_prefix": lambda: can_fast}
    if has_match_prefix:
        tree_fields["match_prefix"] = lambda *a, **k: None
    return SimpleNamespace(waiting_queue=list(reqs),
                           tree_cache=SimpleNamespace(**tree_fields))


def _head_inputs_from_vote(canonical, matches):
    from sglang.srt.managers import tp_head_congruence as thc

    return thc.build_uniform_head_inputs(
        canonical, thc.build_head_order_payload(canonical, matches), None, True
    )


def _gate_stub(x: int = 22000, host_carry: int = 30518, tp_size: int = 3):
    _, stub = _stub_scheduler(x, host_carry=host_carry, tp_size=tp_size)
    return stub


def _flip_served_req(rid: str = "flip-1"):
    """The flip's OWN request: 30,000 tokens, 29,000 of them a store hit that
    the prefetch has just published (1,000 device + 28,000 host)."""
    return SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(30000)),
                           prefix_indices=list(range(1000)), host_hit_length=28000)


def test_f5a_the_head_vote_is_a_measurement_not_a_default(monkeypatch):
    """FINDING 1, the producer.  The vote's measurement was gated on
    `tree.supports_fast_match_prefix()`, which `BasePrefixCache` returns
    False for and NOTHING in this tree overrides -- so the branch never ran
    and the vote published `num_matched_prefix_tokens`, a field initialised
    to 0 and written only by the call the guard was suppressing.  A vote is
    a measurement or an abstention; it is never a default."""
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

    # the fact the fix rests on, asserted rather than assumed
    assert BasePrefixCache.supports_fast_match_prefix(None) is False

    def measured(tree, r, include_req=True):
        r.num_matched_prefix_tokens = 29000

    monkeypatch.setattr(sp, "match_prefix_for_req", measured)
    req = SimpleNamespace(rid="flip-1", num_matched_prefix_tokens=0)
    canonical, matches = Scheduler._local_head_prefix_matches(_head_stub([req]))
    assert canonical == ["flip-1"]
    assert matches == {"flip-1": 29000}, matches


def test_f5b_a_store_resident_request_is_not_refused_by_its_own_group(monkeypatch):
    """FINDING 1, the consequence, end to end: the request the flip exists to
    serve.  With the vote a constant 0, `group_match_for` returns 0 -- an
    OPINION, not an abstention (None only at <= -1) -- and C11's delta
    prices the WHOLE loaded prefix as uncached: extent 30,000 against
    X=22,000, W31, re-route, P re-prefills a store-resident rid, flip, W31
    again.  That is the R-3 livelock with no exit."""
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers.scheduler import Scheduler

    def measured(tree, r, include_req=True):
        r.num_matched_prefix_tokens = 29000

    monkeypatch.setattr(sp, "match_prefix_for_req", measured)
    vote_req = SimpleNamespace(rid="flip-1", num_matched_prefix_tokens=0)
    head = _head_inputs_from_vote(*Scheduler._local_head_prefix_matches(_head_stub([vote_req])))
    gate, req = _gate_stub(), _flip_served_req()
    assert Scheduler.weg2_uncached_extent(gate, req, head) == 1000
    assert Scheduler._weg2_x_refuses(gate, req, head) is False


def test_f5c_a_rid_this_rank_cannot_price_is_absent_never_zero(monkeypatch):
    """FINDING 1, the abstain direction, per rid.  A rank that cannot price a
    rid must contribute ABSENT -- the gate then abstains, which admits --
    and it must still price the rest of the head."""
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers import tp_head_congruence as thc
    from sglang.srt.managers.scheduler import Scheduler

    def measured_except_flip1(tree, r, include_req=True):
        if r.rid == "flip-1":
            raise RuntimeError("no match on this rank")
        r.num_matched_prefix_tokens = 4000

    monkeypatch.setattr(sp, "match_prefix_for_req", measured_except_flip1)
    reqs = [SimpleNamespace(rid="flip-1", num_matched_prefix_tokens=0),
            SimpleNamespace(rid="other", num_matched_prefix_tokens=0)]
    canonical, matches = Scheduler._local_head_prefix_matches(_head_stub(reqs))
    assert "flip-1" not in matches, matches
    assert matches["other"] == 4000, "one unpriceable rid may not void the head"
    head = _head_inputs_from_vote(canonical, matches)
    assert thc.group_match_for(head, "flip-1") is None
    gate, req = _gate_stub(), _flip_served_req()
    assert Scheduler._weg2_x_refuses(gate, req, head) is False
    assert gate._weg2_x_abstained == 1

    # ... and a tree with no matching at all abstains for the whole head,
    # rather than publishing a head of zeros.
    _, none_matches = Scheduler._local_head_prefix_matches(
        _head_stub(reqs, has_match_prefix=False))
    assert none_matches == {}


def _shipped_head_vote_order() -> str:
    """WHICH ORDER THE SHIPPED REDUCE TAKES ITS TWO STEPS IN.  Read off the
    shipped function so the driver below exercises the real ordering rather
    than a hard-coded one."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._update_uniform_pool_budget)
    vote = src.index("self._local_head_prefix_matches()")
    drain = src.index("_ballot_verdicts = self._drain_prefetch_progress()")
    return "vote_first" if vote < drain else "drain_first"


def test_f5d_the_head_vote_is_measured_below_the_prefetch_drain(monkeypatch):
    """FINDING 1, the ordering.  `_drain_prefetch_progress` is the call that
    publishes a completed store read into the host tier, and the pass on
    which a request first becomes eligible IS the pass its prefetch
    completes.  Taken above the drain the vote is a PRE-load snapshot of a
    tree the gate then reads POST-load, so the group's match is 0 while the
    rank's own is the whole loaded prefix and the delta prices the
    difference as uncached work.  Driven in the order the shipped reduce
    actually uses."""
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers.scheduler import Scheduler

    published = {"tokens": 0}  # what the host tier can hand back right now

    def measured(tree, r, include_req=True):
        r.num_matched_prefix_tokens = published["tokens"]

    monkeypatch.setattr(sp, "match_prefix_for_req", measured)
    vote_req = SimpleNamespace(rid="flip-1", num_matched_prefix_tokens=0)
    # `can_fast=True` so this test is about ORDER alone: the capability
    # guard f5a covers cannot also be the reason it fails.
    stub = _head_stub([vote_req], can_fast=True)

    def vote():
        return Scheduler._local_head_prefix_matches(stub)

    def drain():
        published["tokens"] = 29000  # _insert_helper_host publishes the span

    order = _shipped_head_vote_order()
    if order == "vote_first":
        canonical, matches = vote()
        drain()
    else:
        drain()
        canonical, matches = vote()
    assert order == "drain_first", (
        "the head vote is taken above the prefetch drain, so it is a pre-load "
        "snapshot of the tree the X gate reads post-load"
    )
    head = _head_inputs_from_vote(canonical, matches)
    gate, req = _gate_stub(), _flip_served_req()
    assert Scheduler._weg2_x_refuses(gate, req, head) is False


def test_f5e_the_gate_still_refuses_a_genuinely_cold_prompt(monkeypatch):
    """The fix may not disarm law 4.  A cold 30,000-token prompt the group
    agrees on is still W31, and a group MIN below this rank's match still
    prices the group's extent, not this rank's."""
    from sglang.srt.managers import schedule_policy as sp
    from sglang.srt.managers.scheduler import Scheduler

    def cold(tree, r, include_req=True):
        r.num_matched_prefix_tokens = 0

    monkeypatch.setattr(sp, "match_prefix_for_req", cold)
    vote_req = SimpleNamespace(rid="cold", num_matched_prefix_tokens=0)
    head = _head_inputs_from_vote(*Scheduler._local_head_prefix_matches(_head_stub([vote_req])))
    gate = _gate_stub()
    cold_req = SimpleNamespace(rid="cold", full_untruncated_fill_ids=list(range(30000)),
                               prefix_indices=[], host_hit_length=0)
    assert Scheduler.weg2_uncached_extent(gate, cold_req, head) == 30000
    assert Scheduler._weg2_x_refuses(gate, cold_req, head) is True


# ------------------------------------------------------- ROUND 2, FINDING 2
def _handoff_front(**kw) -> Front:
    kwargs = dict(d_bs=2, p_concurrency=2, idle_layout="P", min_dwell_ms=0.0,
                  tp_prefill_max_tokens=0, flip_min_work_tokens=1)
    kwargs.update(kw)
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, **kwargs)


async def _drive_handoff(f: Front, queue_marks: List[str], after_handoff=None):
    """Hand ONE request to D and stop there: the client never resumes, so
    the rid never reaches `leg2` and never enters `D.outstanding`.  Returns
    the flips the controller took while the request was in that window."""
    flips: List[tuple] = []

    async def fake_flip(src, dst):
        flips.append((src, dst))

    f.flip = fake_flip
    for mark in queue_marks:
        f.queue.append(Pending(mark, "/generate", {}, "x", time.time(),
                               asyncio.get_event_loop().create_future(), est_prompt=100))
    fut = asyncio.get_event_loop().create_future()
    p = Pending("h1", "/generate", {}, "x", time.time(), fut, est_prompt=10)
    f._ready_for_d.append(p)
    f._sync_batch_gate()
    tasks = [asyncio.create_task(f.d_admitter()), asyncio.create_task(f.controller())]
    try:
        assert await _until(lambda: fut.done(), timeout=5), "the seat was never granted"
        assert not f._ready_for_d and not f.groups["D"].outstanding
        # THE WINDOW, stated in the state that exists at the parent commit:
        # a seat is held, and the rid is in neither set the controller reads.
        assert f._seats_in_use() == 1
        if after_handoff is not None:
            after_handoff(f)
        await asyncio.sleep(1.0)  # five controller ticks in the window
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return flips


def test_f6a_the_idle_mirror_does_not_sleep_d_under_a_handed_off_request():
    """FINDING 2: between the admitter's popleft + `fut.set_result(True)`
    and `leg2`'s `D.outstanding[rid] = ...` the request is in NEITHER set
    the C6 idle mirror reads, so the mirror could flip D->P on a request it
    had already handed to D -- and `flip`'s own `drain(D)` cannot protect
    it, because `D.outstanding` is empty by construction there.  The client
    then POSTs into a sleeping group: R-2's LOST-REQUEST class, D->P."""

    async def body():
        f = _handoff_front()
        assert await _drive_handoff(f, []) == []

    asyncio.run(body())


def test_f6b_the_work_arm_does_not_sleep_d_under_a_handed_off_request():
    """The same window, the other guard: `not D.outstanding or not
    self.admit_d` at :1514.  Queue non-empty, so this is the work arm and
    not the idle mirror -- and the `admit_d` half needs the term too, since
    it flips deliberately."""

    async def body():
        f = _handoff_front(idle_layout="D")
        assert await _drive_handoff(f, ["q1"]) == []
        # the `admit_d` half: the fairness switch closes D down DURING the
        # window, which is exactly when it is closed in production.
        f2 = _handoff_front(idle_layout="D")

        def close_d(front):
            front.admit_d = False

        assert await _drive_handoff(f2, ["q2"], after_handoff=close_d) == []

    asyncio.run(body())


def test_f6c_the_flip_still_happens_once_the_request_has_arrived():
    """The term may only DELAY: with D holding the request the drain can see
    it, the work arm flips exactly as before."""

    async def body():
        f = _handoff_front(idle_layout="D")
        flips: List[tuple] = []

        async def fake_flip(src, dst):
            flips.append((src, dst))

        f.flip = fake_flip
        f.queue.append(Pending("q1", "/generate", {}, "x", time.time(),
                               asyncio.get_event_loop().create_future(), est_prompt=100))
        f.admit_d = False  # the front is closing D down; D holds nothing
        assert f._seats_in_use() == 0
        task = asyncio.create_task(f.controller())
        try:
            assert await _until(lambda: flips == [("D", "P")], timeout=5), flips
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


def test_f6d_the_handoff_term_is_derived_from_the_seat_not_recorded():
    """No second ledger: the term is `seats - outstanding`, so it has the
    seat's own lifecycle and cannot be left behind by a missing deleter."""

    async def body():
        f = _handoff_front()
        assert f._handoff_in_flight() == 0
        await f._d_seat.acquire()  # the admitter's own acquire
        seat = front_mod.Seat(f, "h1", "batch", tokens=10)
        assert f._seats_in_use() == 1 and f._handoff_in_flight() == 1
        f.groups["D"].outstanding["h1"] = time.time()  # leg2's first line
        assert f._handoff_in_flight() == 0, "a request D holds is not a hand-off"
        del f.groups["D"].outstanding["h1"]
        seat.release("test")
        assert f._seats_in_use() == 0 and f._handoff_in_flight() == 0

    asyncio.run(body())


# ======================================================= ROUND 6 (FIX 6), MF-1
# Law 5's idle-rest decision, keyed on the CONFIGURED layout in BOTH
# directions.  Boot weg2sc2 could not observe it at all: `--idle-layout tp`
# makes the launcher emit front `--idle-layout D`, and BOTH old emit sites
# were gated on the OTHER layout, so `WEG2 IDLE-REST` was unreachable under
# the default -- a code fact, not a missed measurement.  These drive the REAL
# controller, so they answer "does it run", not "is it written".
def _idle_lines(caplog) -> List[str]:
    return [ln for ln in caplog.text.splitlines() if "WEG2 IDLE-REST" in ln]


def test_g1a_idle_tp_rests_on_d_and_says_so(caplog):
    """MF-1, the direction that could not print: D awake, empty backlog,
    `--idle-layout tp` (front `D`).  The front must REST -- and say which
    layout it is resting in and why -- rather than fall silently through the
    controller pass, which is what weg2sc2's zero IDLE-REST lines were."""

    async def body():
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            async with Harness(awake="D", p_concurrency=4, d_bs=6,
                               tp_prefill_max_tokens=10, idle_layout="D") as h:
                await asyncio.sleep(1.0)  # five controller ticks
                assert h.front.awake == "D", "resting means NOT flipping"
                assert h.front.epoch == 0
                lines = _idle_lines(caplog)
                assert len(lines) == 1, (
                    f"exactly one line per rest spell (the controller ticks at "
                    f"0.2 s; a rest without an edge trigger would print five a "
                    f"second), got {len(lines)}")
                assert "layout=D" in lines[0] and "configured=D" in lines[0]
                assert "reason=" in lines[0] and "held_s=" in lines[0]

    asyncio.run(body())


def test_g1b_idle_pp_rests_on_p_and_says_so(caplog):
    """The mirror direction, which round 3 could already print -- kept as a
    test so ONE decision serving both cannot regress on this side while it is
    being fixed on the other."""

    async def body():
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            async with Harness(awake="P", p_concurrency=4, d_bs=6,
                               tp_prefill_max_tokens=10, idle_layout="P") as h:
                await asyncio.sleep(1.0)
                assert h.front.awake == "P" and h.front.epoch == 0
                lines = _idle_lines(caplog)
                assert len(lines) == 1, f"one line, not one per tick: {len(lines)}"
                assert "layout=P" in lines[0] and "configured=P" in lines[0]

    asyncio.run(body())


def test_g1c_a_front_awake_in_the_wrong_layout_flips_once_then_rests(caplog):
    """The two CROSS cases, both halves of law 5's mirror: when the awake
    group is not the configured idle layout, the front flips to it ONCE and
    then rests there -- it does not flip back and it does not keep flipping."""

    async def d_awake_idle_pp():
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            async with Harness(awake="D", p_concurrency=4, d_bs=6,
                               tp_prefill_max_tokens=10, idle_layout="P",
                               min_dwell_ms=0.0) as h:
                assert await _until(lambda: h.front.awake == "P", timeout=10)
                await asyncio.sleep(0.8)
                assert h.front.awake == "P", "and it stays there"
                assert h.front.epoch == 1, "exactly one flip, not a ping-pong"
                lines = _idle_lines(caplog)
                assert len(lines) == 1 and "layout=P" in lines[0], lines

    async def p_awake_idle_tp():
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            async with Harness(awake="P", p_concurrency=4, d_bs=6,
                               tp_prefill_max_tokens=10, idle_layout="D") as h:
                assert await _until(lambda: h.front.awake == "D", timeout=10)
                await asyncio.sleep(0.8)
                assert h.front.awake == "D" and h.front.epoch == 1
                lines = _idle_lines(caplog)
                assert len(lines) == 1 and "layout=D" in lines[0], lines

    asyncio.run(d_awake_idle_pp())
    caplog.clear()
    asyncio.run(p_awake_idle_tp())


# ======================================================= ROUND 6 (FIX 6), MF-3
def test_g2_every_drain_prices_what_the_disarmed_store_read_cost_p(caplog):
    """MF-3 (a): the W38 cost is a number in the log, per drain epoch.

    W38 refuses every storage read on group P, so a follow-up whose prefix
    has left P's device tier is re-prefilled whole.  The arithmetic is
    asserted in ``test_weg2_sched_fix3_0907.py::test_a14``; what this proves
    is REACH -- that a real drain emits the line at all, beside the P-DRAIN
    line it belongs to, with the epoch's own denominators."""

    async def body():
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            async with Harness(awake="P", p_concurrency=4, d_bs=6,
                               tp_prefill_max_tokens=10, idle_layout="D") as h:
                for i in range(3):
                    h.post(f"pr{i}")
                assert await _until(lambda: h.front.awake == "D", timeout=15)
                await asyncio.sleep(0.3)
        lines = [ln for ln in caplog.text.splitlines() if "WEG2 P-PREFIX-REUSE" in ln]
        assert len(lines) == 1, f"one line per drain that prefilled anything: {lines}"
        assert "requests=3" in lines[0], lines[0]
        # the fake group answers every leg 1 with cached_tokens=100, so the
        # reused term is a MEASUREMENT that moves -- not a hardcoded zero.
        assert "prefix_tokens_reused=300" in lines[0], lines[0]
        assert "prefix_tokens_available_in_store=" in lines[0]
        assert "forgone_tokens=" in lines[0]
        drain = [ln for ln in caplog.text.splitlines() if "WEG2 P-DRAIN" in ln]
        assert len(drain) == 1, "and it is priced per drain epoch, beside it"

    asyncio.run(body())


def test_g3_the_launcher_states_the_carrierless_pp_arm_at_launch_from_the_argv_it_runs():
    """MF-3 (b): the cost is stated ONCE AT LAUNCH, so it is never silent --
    and it is read off the P argv this launcher is about to run rather than
    asserted from memory, so a group P that stopped being a PP group would
    change the line instead of leaving it lying."""
    argv = launcher_mod.argv_p("py", "/m", [1, 2, 3], 8, 512, 1.0, [], 4, 30000)
    line = launcher_mod.w38_armed_line(argv)
    # RECONCILED on the 0908 train: W38's own gate is deleted and the banner
    # states the ONE surviving arm, #1245's undistributable drop.
    assert line.startswith("WEG2 #1245 ARMED"), (
        "group P ships --pp-size 3, so the undistributable load-back IS dropped on it")
    assert "W38 RETIRED INTO IT" in line, "and it says which arm was retired into which"
    assert "#968" in line and "PP0-authoritative" in line, "name the remedy"
    assert "P-PREFIX-REUSE" in line, "and where the interim cost is measured"
    # the other polarity is not hypothetical: it is what a carrier or a TP-only
    # group P would produce, and the line must then stop claiming the cost.
    assert launcher_mod.w38_armed_line(["--pp-size", "1"]).startswith("WEG2 #1245 NOT ARMED")
    # WIRED, not merely written (desk-written-never-executed): main logs it.
    main_src = inspect.getsource(launcher_mod.main)
    assert "log(w38_armed_line(spec_p.argv))" in main_src
