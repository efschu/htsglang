"""H91 part C: the Weg-2 front's phase policy (user design 25.09.2026).

One test per behaviour, each red on 0d5a086e59 and green with the fix:

* rule 1 -- a P phase takes at most ``--p-phase-max-requests`` (6) and
  flips; the overlap is planned against ``--p-pool-tokens``;
* rule 2 -- D decodes everything it was handed before the flip back
  (a prefilled request still waiting for a D seat holds the D phase), and
  the wake message carries ``handoff_n``;
* rule 3 -- the wait bound parks D's running decodes via
  ``POST /weg2/park_running`` and flips to P; the parked request keeps its
  client stream and finishes in the next D phase; a D without the endpoint
  (404) falls back to the pre-H91 path, named;
* rule 4 -- an X-route request D serves itself never counts as waiting.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU. P and D are aiohttp
test servers; every async body runs under ``asyncio.run`` (no
pytest-asyncio in this venv).
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import ClientSession, ClientTimeout, web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front import Front, Pending  # noqa: E402


#: the agreed path (part B); a literal so this module imports on the base too,
#: where rule 2's test is red on behaviour rather than on an import.
PARK_PATH = "/weg2/park_running"


# ------------------------------------------------------------------ fakes
class FakeGroup:
    """P or D. Records the order of what the front asks, the resume bodies,
    the park calls, and which requests it is running right now."""

    def __init__(self, name: str, delay: float = 0.02, park: Optional[str] = "ok"):
        self.name = name
        self.delay = delay
        self.park = park  # "ok" = endpoint present, None = absent (404)
        self.timeline: List[str] = []
        self.gen_marks: List[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.hold: Optional[Dict[str, asyncio.Event]] = None
        self.running: Dict[str, str] = {}  # rid -> mark, while held
        self.resume_bodies: List[dict] = []
        self.park_bodies: List[dict] = []
        self.server: Optional[TestServer] = None
        self.url = ""

    @staticmethod
    def _mark(payload: dict) -> str:
        m = re.search(r"MARK(\S+)", front_mod.request_text(payload))
        return m.group(1) if m else "?"

    async def _generate(self, request: web.Request) -> web.Response:
        payload = await request.json()
        mark = self._mark(payload)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self.timeline.append(f"gen:{mark}")
        self.gen_marks.append(mark)
        try:
            if self.hold is not None:
                self.running[payload.get("rid") or mark] = mark
                try:
                    await self.hold.setdefault(mark, asyncio.Event()).wait()
                finally:
                    self.running.pop(payload.get("rid") or mark, None)
            else:
                await asyncio.sleep(self.delay)
            return web.json_response({
                "choices": [{"text": "x"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": 100}},
            })
        finally:
            self.in_flight -= 1
            self.timeline.append(f"done:{mark}")

    async def _ok(self, request: web.Request) -> web.Response:
        self.timeline.append(f"rpc:{request.path.strip('/')}")
        return web.json_response({"ok": True})

    async def _resume(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.resume_bodies.append(body)
        self.timeline.append("rpc:resume_memory_occupation")
        return web.json_response({"ok": True})

    async def _park(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.park_bodies.append(body)
        self.timeline.append("rpc:weg2/park_running")
        return web.json_response({"parked": sorted(self.running)})

    async def _info(self, request: web.Request) -> web.Response:
        return web.json_response({"x": 1})

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/generate", self._generate)
        app.router.add_post("/flush_cache", self._ok)
        app.router.add_post("/release_memory_occupation", self._ok)
        app.router.add_post("/resume_memory_occupation", self._resume)
        app.router.add_post("/abort_request", self._ok)
        app.router.add_get("/get_server_info", self._info)
        if self.park is not None:
            app.router.add_post(PARK_PATH, self._park)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.close()

    def release(self, mark: str) -> None:
        assert self.hold is not None
        self.hold.setdefault(mark, asyncio.Event()).set()

    def release_all(self) -> None:
        for ev in (self.hold or {}).values():
            ev.set()
        self.hold = None


class Harness:
    def __init__(self, admitter: bool = True, d_park: Optional[str] = "ok", **front_kwargs):
        self.front_kwargs = front_kwargs
        self.admitter = admitter
        self.p = FakeGroup("P")
        self.d = FakeGroup("D", park=d_park)
        self.front: Optional[Front] = None
        self.tasks: List[asyncio.Task] = []
        self.posts: List[asyncio.Task] = []
        self.server: Optional[TestServer] = None
        self.client: Optional[ClientSession] = None

    async def __aenter__(self) -> "Harness":
        await self.p.start()
        await self.d.start()
        kw = dict(awake="D", tag="t", store_dir="", prefill_sid=0, decode_sid=0,
                  dc_reserve={}, w_s=0.0, tp_prefill_max_tokens=10, min_dwell_ms=0.0,
                  idle_layout="D")
        kw.update(self.front_kwargs)
        self.front = Front(self.p.url, self.d.url, **kw)
        self.front.session = ClientSession(timeout=ClientTimeout(total=60))
        app = web.Application(client_max_size=1024 ** 3)
        app.router.add_post("/generate", self.front.handle_generate)
        self.server = TestServer(app)
        await self.server.start_server()
        self.client = ClientSession(timeout=ClientTimeout(total=60))
        self.tasks = [asyncio.create_task(self.front.controller())]
        if self.admitter:
            self.tasks.append(asyncio.create_task(self.front.d_admitter()))
        return self

    async def __aexit__(self, *exc) -> None:
        for t in self.posts:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self.posts, return_exceptions=True)
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.client.close()
        await self.front.session.close()
        await self.server.close()
        await self.p.stop()
        await self.d.stop()

    def post(self, mark: str, chars: int = 3000) -> asyncio.Task:
        body = {"prompt": f"MARK{mark} " + mark * chars}
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
    return pred()


def _first(timeline: List[str], entry: str) -> int:
    return next(i for i, e in enumerate(timeline) if e == entry)


# ------------------------------------------------------------- pure half
def test_pure_phase_policy_terms():
    from sglang.srt.weg2 import phase_policy as pp

    assert pp.PARK_PATH == PARK_PATH
    assert pp.P_PHASE_MAX_REQUESTS_DEFAULT == 6
    assert pp.P_POOL_TOKENS_DEFAULT == 262144
    assert pp.D_WAIT_BOUND_S_DEFAULT == 60.0
    # overlap plan: head first, stop at the first that does not fit, >= 1
    assert pp.plan_p_overlap([100_000, 100_000, 100_000], 262144, 6) == 2
    assert pp.plan_p_overlap([400_000, 10], 262144, 6) == 1
    assert pp.plan_p_overlap([10] * 9, 262144, 6) == 6
    assert pp.plan_p_overlap([10] * 9, 0, 0) == 9
    assert pp.p_overlap_admits(0, 0, 10 ** 9, 262144) is True
    assert pp.p_request_cost(2000, 0, skip_leg1=True) == 0
    assert pp.p_request_cost(2000, 2500) == 2500
    # the wait is counted from max(arrival, D phase start)
    assert pp.d_phase_wait_s(t_arrive=10.0, t_phase_start=50.0, now=70.0) == 20.0
    assert pp.d_phase_wait_s(t_arrive=60.0, t_phase_start=50.0, now=70.0) == 10.0
    assert pp.wait_bound_fired(60.0, 60.0) and not pp.wait_bound_fired(59.9, 60.0)
    assert not pp.wait_bound_fired(1e9, 0.0) and not pp.wait_bound_fired(None, 60.0)
    assert pp.park_body(7, 60.0) == {"epoch": 7, "reason": "wait-bound-60s"}
    assert pp.park_verdict(200, json.dumps({"parked": ["a", "b"]})) == ("parked", ["a", "b"], "")
    assert pp.park_verdict(404, "")[0] == "unsupported"
    assert pp.park_verdict(501, "")[0] == "unsupported"
    assert pp.park_verdict(500, "x")[0] == "failed"
    assert pp.park_verdict(200, "{}")[0] == "failed"        # malformed is never "nothing parked"
    assert pp.park_verdict(200, "not json")[0] == "failed"
    assert pp.park_verdict(409, "not group D")[0] == "failed"   # part B: server is not D
    # part B's answer carries more fields; only parked is read
    assert pp.park_verdict(200, json.dumps({"success": True, "parked": ["r"], "held": [],
                                            "epoch": 3, "message": "ok"}))[1] == ["r"]
    # part B re-queues a parked request no sleep followed after 30 s
    assert pp.PARK_REQUEUE_S == 30.0
    assert not pp.parked_lapsed(100.0, 129.9) and pp.parked_lapsed(100.0, 130.0)
    # the leg-1 stall bound: old enough AND no P work for as long
    assert pp.leg1_stalled(now=200.0, t_dispatch=0.0, t_last_evidence=0.0, bound_s=180.0)
    assert not pp.leg1_stalled(now=200.0, t_dispatch=0.0, t_last_evidence=30.0, bound_s=180.0)
    assert not pp.leg1_stalled(now=170.0, t_dispatch=0.0, t_last_evidence=0.0, bound_s=180.0)
    assert not pp.leg1_stalled(now=1e9, t_dispatch=0.0, t_last_evidence=0.0, bound_s=0.0)
    assert pp.P_LEG1_STALL_S_DEFAULT == 180.0


def test_main_flags_default_to_the_user_design():
    """Unified tree (operator 26.09.): the flags parse to None and
    phase_policy.resolve_front_defaults gives the user design under the NF
    standard form (profile nextflash), 0 under qwen27b -- pinned in
    test_unify_standard_form_profile.py."""
    import inspect
    import types

    from sglang.srt.weg2 import phase_policy

    src = inspect.getsource(front_mod.main)
    for flag in ("--p-phase-max-requests", "--p-pool-tokens", "--d-wait-bound-s",
                 "--p-leg1-stall-s"):
        i = src.index(f'"{flag}"')
        assert "default=None" in src[i:i + 80], flag
    assert "phase_policy.resolve_front_defaults(args, bool(envs.SGLANG_WEG2_STANDARD_FORM.get()))" in src
    assert "d_wait_bound_s=args.d_wait_bound_s" in src
    assert "p_phase_max_requests=args.p_phase_max_requests" in src
    ns = types.SimpleNamespace(p_phase_max_requests=None, p_pool_tokens=None,
                               d_wait_bound_s=None, p_leg1_stall_s=None)
    phase_policy.resolve_front_defaults(ns, True)
    assert (ns.p_phase_max_requests, ns.p_pool_tokens, ns.d_wait_bound_s, ns.p_leg1_stall_s) == (
        phase_policy.P_PHASE_MAX_REQUESTS_DEFAULT, phase_policy.P_POOL_TOKENS_DEFAULT,
        phase_policy.D_WAIT_BOUND_S_DEFAULT, phase_policy.P_LEG1_STALL_S_DEFAULT)


# ------------------------------------------------------ rule 1: the cap
def test_rule1_a_p_phase_takes_at_most_six_then_flips_with_handoff_n():
    async def body():
        async with Harness(awake="P", p_concurrency=8, d_bs=8,
                           p_phase_max_requests=6) as h:
            h.p.delay = 0.3
            tasks = [h.post(f"c{i}") for i in range(9)]
            assert await _until(lambda: "rpc:release_memory_occupation" in h.p.timeline, 20)
            first = _first(h.p.timeline, "rpc:release_memory_occupation")
            before = [e for e in h.p.timeline[:first] if e.startswith("gen:")]
            assert len(before) == 6, h.p.timeline[:first]
            results = await asyncio.wait_for(asyncio.gather(*tasks), 40)
            assert [s for s, _ in results] == [200] * 9
            # rule 2: the wake of D carried the hand-off count on its kv resume
            kv = [b for b in h.d.resume_bodies if b.get("tags") == ["kv_cache"]]
            assert kv and kv[0].get("handoff_n") == 6 and kv[0].get("parked_n") == 0, kv
            # the second P phase took the remaining three
            assert h.p.gen_marks.count("c8") == 1 and len(h.p.gen_marks) == 9

    asyncio.run(body())


def test_rule1_the_overlap_is_planned_against_p_pool_tokens():
    async def body():
        # each prompt ~2000 est tokens ("MARKo0 " + 6000 chars at 3 chars/token)
        async with Harness(awake="P", p_concurrency=8, d_bs=8,
                           p_phase_max_requests=6, p_pool_tokens=4500) as h:
            h.p.delay = 0.25
            tasks = [h.post(f"o{i}") for i in range(4)]
            results = await asyncio.wait_for(asyncio.gather(*tasks), 40)
            assert [s for s, _ in results] == [200] * 4
            assert h.p.peak_in_flight == 2, h.p.peak_in_flight
            assert len(h.p.gen_marks) == 4

    asyncio.run(body())


# ------------------------------------------------ rule 2: D decodes all
def test_rule2_d_does_not_flip_back_while_a_prefilled_request_waits_for_a_seat():
    """Base kwargs only: on 0d5a086e59 the front flipped D->P the moment D's
    ledger was empty, with the P phase's batch still in _ready_for_d."""

    async def body():
        async with Harness(admitter=False, awake="D", d_bs=1) as h:
            loop = asyncio.get_running_loop()
            ready = Pending("weg2-1-1", "/generate", {}, "x", time.time(), loop.create_future(),
                            est_prompt=5000, est_uncached=5000)
            h.front._ready_for_d.append(ready)
            h.front._sync_batch_gate()
            h.front.queue.append(Pending("weg2-1-2", "/generate", {}, "y", time.time(),
                                         loop.create_future(), est_prompt=5000, est_uncached=5000))
            await asyncio.sleep(0.8)
            assert h.front.awake == "D", "flipped away from a prefilled request D never ran"
            assert "rpc:release_memory_occupation" not in h.d.timeline
            # D ran it (the admitter's pop) -- now the flip back may come
            h.front._ready_for_d.popleft()
            ready.fut.set_result(True)
            h.front._sync_batch_gate()
            assert await _until(lambda: h.front.awake == "P", 10)

    asyncio.run(body())


# ------------------------------------------ rule 3: wait bound -> park
def test_rule3_the_wait_bound_parks_the_running_decode_and_flips_to_p():
    async def body():
        async with Harness(awake="P", p_concurrency=4, d_bs=2,
                           d_wait_bound_s=0.6, p_phase_max_requests=6) as h:
            h.d.hold = {}
            t0 = h.post("r0")
            assert await _until(lambda: h.d.running, 20)          # D decodes r0
            rid0 = next(iter(h.d.running))
            epoch_d = h.front.epoch
            t1 = h.post("r1")                                     # waits for P
            assert await _until(lambda: "gen:r1" in h.p.timeline, 20)
            # park first, then D's sleep; the body is the agreed shape
            assert h.d.park_bodies == [{"epoch": epoch_d, "reason": "wait-bound-0.6s"}]
            assert _first(h.d.timeline, "rpc:weg2/park_running") < \
                _first(h.d.timeline, "rpc:release_memory_occupation")
            # r0 stays in flight for the front: stream open, no requeue, no 2nd leg 1
            assert not t0.done()
            assert h.p.gen_marks.count("r0") == 1
            assert h.front.counters["wait_bound_fired"] == 1 and h.front.counters["d_parked"] == 1
            # D wakes again: r0 is resumed (parked_n=1), r1 is the new hand-off
            assert await _until(lambda: h.front.awake == "D" and not h.front._d_parked, 20)
            kv = [b for b in h.d.resume_bodies if b.get("tags") == ["kv_cache"]]
            assert kv[-1].get("parked_n") == 1 and kv[-1].get("handoff_n") == 1, kv
            assert rid0 in h.front.groups["D"].outstanding
            h.d.release_all()
            (s0, _), (s1, _) = await asyncio.wait_for(asyncio.gather(t0, t1), 20)
            assert (s0, s1) == (200, 200)
            assert h.d.gen_marks.count("r0") == 1                 # never re-posted to D

    asyncio.run(body())


def test_rule3_a_d_without_the_park_endpoint_falls_back_and_says_so(caplog):
    import logging

    async def body():
        async with Harness(awake="P", p_concurrency=4, d_bs=2, d_park=None,
                           d_wait_bound_s=0.4, p_phase_max_requests=6) as h:
            h.d.hold = {}
            t0 = h.post("r0")
            assert await _until(lambda: h.d.running, 20)
            t1 = h.post("r1")
            assert await _until(lambda: h.front.counters["park_unsupported"] == 1, 10)
            assert h.front._park_unsupported is True and not h.front._d_parked
            # fallback = the pre-H91 path: no flip while r0 still decodes
            await asyncio.sleep(0.6)
            assert h.front.awake == "D" and "rpc:release_memory_occupation" not in h.d.timeline
            assert not h.front.admit_d                            # admission closed
            h.d.release("r0")
            (s0, _) = await asyncio.wait_for(t0, 10)
            assert s0 == 200
            assert await _until(lambda: "gen:r1" in h.p.timeline, 20)   # then the flip
            h.d.release_all()
            (s1, _) = await asyncio.wait_for(t1, 20)
            assert s1 == 200

    with caplog.at_level(logging.WARNING, logger="weg2.front"):
        asyncio.run(body())
    assert any("WEG2 PARK-UNSUPPORTED" in r.message and "status=404" in r.message
               for r in caplog.records)


# ---------------------------------------------- rule 4: X route untouched
def test_rule4_an_x_route_request_d_serves_is_never_waiting():
    async def body():
        async with Harness(awake="D", d_bs=2, tp_prefill_max_tokens=100_000,
                           d_wait_bound_s=0.3, p_phase_max_requests=6) as h:
            h.d.hold = {}
            t = h.post("s0")
            assert await _until(lambda: h.d.running, 10)          # SHORT -> D directly
            await asyncio.sleep(1.0)                              # > 3x the bound
            assert h.front.counters["wait_bound_fired"] == 0
            assert h.d.park_bodies == [] and h.front.awake == "D"
            assert h.front.counters["route_short"] == 1 and not h.p.gen_marks
            h.d.release_all()
            (s, _) = await asyncio.wait_for(t, 10)
            assert s == 200

    asyncio.run(body())


# ------------------------------- rule 3: W1b never aborts a parked request
def test_rule3_w1b_aborts_only_the_stuck_requests_and_spares_the_parked_ones():
    """W1b's abort_all would take the wait-bound-parked requests too; with
    any of them on D the stuck ones are aborted per rid. Without parked
    requests W1b is byte-identical (abort_all)."""
    import collections
    import types

    def mk(outstanding, parked):
        log = []
        f = types.SimpleNamespace(counters=collections.Counter(), drain_deadline_s=120.0,
                                  _d_parked=dict(parked))

        async def rpc(g, path, body, timeout):
            log.append((g.name, path, body))
            for r in list(g.outstanding):
                if body.get("abort_all") or body.get("rid") == r:
                    g.outstanding.pop(r, None)
            return 200, "{}"

        f.rpc = rpc
        f._abort_parked_on_drain = Front._abort_parked_on_drain.__get__(f)
        g = types.SimpleNamespace(name="D", url="http://d", outstanding=dict(outstanding))
        return f, g, log

    f, g, log = mk({"a": 1.0, "b": 1.0}, {})
    assert asyncio.run(f._abort_parked_on_drain(g, "D")) is True
    assert log == [("D", "/abort_request", {"rid": "", "abort_all": True})]
    f, g, log = mk({"a": 1.0, "p": 1.0}, {"p": time.time()})
    assert asyncio.run(f._abort_parked_on_drain(g, "D")) is True
    assert log == [("D", "/abort_request", {"rid": "a"})], log
    assert list(g.outstanding) == ["p"] and f.counters["W1b_parked_aborted"] == 1
    f, g, log = mk({"p": 1.0}, {"p": time.time()})
    assert asyncio.run(f._abort_parked_on_drain(g, "D")) is True and log == []
    # a park D has already re-queued (older than 30 s) is running again: W1b as before
    f, g, log = mk({"a": 1.0, "p": 1.0}, {"p": time.time() - 31.0})
    assert asyncio.run(f._abort_parked_on_drain(g, "D")) is True
    assert log == [("D", "/abort_request", {"rid": "", "abort_all": True})]


def test_rule3_a_lapsed_park_counts_as_running_again():
    """Part B re-queues a parked request after 30 s without a sleep; the
    front's drain/W3 ledger must count it from then on, and the controller
    drops it from the parked set by name."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 0.0, d_wait_bound_s=60.0)
    D = f.groups["D"]
    D.outstanding.update({"a": time.time(), "b": time.time()})
    f._d_parked.update({"a": time.time(), "b": time.time() - 31.0})
    assert f._flip_ledger(D) == ["b"]
    f._drop_lapsed_parks()
    assert list(f._d_parked) == ["a"] and f.counters["d_parked_lapsed"] == 1
    assert f._flip_ledger(f.groups["P"]) == []


# ------------------------------------ part A companion: the leg-1 stall bound
def test_leg1_that_never_starts_is_requeued_as_an_intake_stall_and_the_flip_follows():
    async def body():
        async with Harness(awake="P", p_concurrency=2, d_bs=2, p_phase_max_requests=6,
                           p_leg1_stall_s=1.0) as h:
            h.p.hold = {}                                         # P queues it, never starts
            t = h.post("st")
            assert await _until(lambda: h.front.counters["p_leg1_stall"] >= 1, 15)
            assert await _until(lambda: "rpc:release_memory_occupation" in h.p.timeline, 15)
            # aborted on P by rid before P was put to sleep, requeued, not failed
            assert _first(h.p.timeline, "rpc:abort_request") < \
                _first(h.p.timeline, "rpc:release_memory_occupation")
            assert h.front.counters["p_intake_stalls"] >= 1
            assert h.front.counters["leg1_failures"] == 0 and not t.done()
            h.p.release_all()                                     # P works again
            (s, _) = await asyncio.wait_for(t, 30)
            assert s == 200
            assert h.p.gen_marks.count("st") >= 2                 # prefilled in a later P phase

    asyncio.run(body())
