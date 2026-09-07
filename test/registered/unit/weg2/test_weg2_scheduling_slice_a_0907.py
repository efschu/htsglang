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
import re
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from aiohttp import ClientSession, ClientTimeout, web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import launcher as launcher_mod
from sglang.srt.weg2.front import Front, Pending, double_prefill_verdict, is_x_refusal


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
                    {"error": f"W31 Weg2TpPrefillExceeded rid=? uncached=99999"}, status=503
                )
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

    def post(self, mark: str, chars: int = 3000) -> asyncio.Task:
        body = {"prompt": f"MARK{mark} " + f"{mark}" * chars}
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
def _stub_scheduler(x: int, host_carry: int):
    from sglang.srt.managers.scheduler import Scheduler

    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        tree_cache=SimpleNamespace(
            cache_controller=SimpleNamespace(mem_pool_host=SimpleNamespace(size=host_carry))
        ),
    )
    # bind the REAL method bodies back onto the stub, so the arithmetic and
    # the host-carry read under test are the shipped ones, not doubles.
    stub.weg2_uncached_extent = lambda req: Scheduler.weg2_uncached_extent(stub, req)
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
            assert h.front.counters["W31_Weg2TpPrefillExceeded"] == 1
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
    assert is_x_refusal(503, "W31 Weg2TpPrefillExceeded: ...") is True
    assert is_x_refusal(503, "W28 Weg2Leg2Unpriced") is False
    assert is_x_refusal(200, "W31 Weg2TpPrefillExceeded") is False


# -------------------------------------------------------------- T10, R-5
def test_t10_a_trickle_below_the_threshold_does_not_buy_a_round_trip():
    """T10: queued work below --flip-min-work-tokens does not flip; crossing
    it, or the fairness bound firing, does.  The weg2zr2 4.3 s P phase (29.4
    s round trip to prefill 5 requests) must not recur."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=22000)
    loop = asyncio.new_event_loop()
    try:
        mk = lambda est: Pending("r", "/generate", {}, "x", time.time(),
                                 loop.create_future(), est_prompt=est)
        f.queue.append(mk(5000))
        assert f._flip_economics_ok(fairness_fired=False) is False
        f.queue.append(mk(18000))
        assert f._flip_economics_ok(fairness_fired=False) is True
        f.queue.clear()
        f.queue.append(mk(10))
        assert f._flip_economics_ok(fairness_fired=False) is False
        assert f._flip_economics_ok(fairness_fired=True) is True
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
    log = tmp_path / "boot_weg2_x_0907_000000.front.log"
    log.write_text(
        "WEG2-FLIP done epoch=1 slept=D woke=P ... flip_total=13247 ms weights_tags=8 dc={}\n"
        "WEG2-FLIP done epoch=2 slept=P woke=D ... flip_total=13247 ms weights_tags=8 dc={}\n"
        "WEG2-SERVED group=P leg=1 rid=a prompt_tokens=30100 cached_tokens=0 wall=8.27s epoch=1\n"
        "WEG2-SERVED group=D leg=2 rid=b status=200 prompt_tokens=30100 cached_tokens=0 "
        "completion_tokens=5 uncached=30100 verdict=single_prefill wall=43.62s epoch=2\n"
    )
    x, prov = launcher_mod.resolve_x(None, str(tmp_path), 4096)
    assert "source=boot:" in prov and log.name in prov
    assert x == launcher_mod.derive_x_star(13.247, 30100 / 43.62, 30100 / 8.27, 4096)
