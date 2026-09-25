# SPDX-License-Identifier: Apache-2.0
"""RC7-X (27B line, user decision 2026-09-25 ~06:43Z: "x entscheidung mit in den release").

PART 1 -- r_D, the D-direct prefill rate the live X* is solved from, came from
the WHOLE leg-2 wall (prefill + the decode of every completion token). Boot
weg2rc4 (``boot_weg2_weg2rc4_9738626129_0925_040419.front.log``) sampled
``uncached=2 wall=6.85s`` and held ``X=4096 <- X_prev=4096 r_D=0 r_P=7035``
for 51 re-solves. Now: r_D = uncached / D's OWN prefill clock
(``prefill_finished_time - forward_entry_time``, weg2/prefill_clock.py) of a
prefill D ran alone, uncached >= 2048, on both wire shapes; no clock, no
sample -- never the wall. D's W50 riegel stands at ``--x-ceiling-tokens`` and
the front clamps the live X to it, so a live X above the launch X is not a
W50 detour.

PART 2 -- X splits by D's load: X_idle (the live X) only while D holds
nothing, behind a 250 ms singleton window (a burst goes to P together);
X_busy (``--x-busy-tokens``, one chunk) while D decodes others. Design point
(a): a request deferred by the split is re-granted on D once D is idle and
quiet -- it neither starves behind FLIP-ECONOMICS nor buys a flip.

Every novelty is tested in BOTH directions; the launcher->front seam is tested
at the ARGV, writer (``front_argv_for``) AND reader (the front's own ``main``).
Hermetic: no GPU, no boot, no /dev/shm; the fake D is an aiohttp TestServer.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import json
import os
import sys
import time
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import ClientSession, ClientTimeout, web
from aiohttp.test_utils import TestServer

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import prefill_clock
from sglang.srt.weg2.front import (
    X_RD_MIN_UNCACHED,
    X_SOLO_WINDOW_S,
    Front,
    Pending,
    r_d_sample,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

MIB = 1024 * 1024
X_IDLE = 10000       # a live X* above the floor (the RC4 arithmetic gives ~10.2k)
CEIL = 16384


def _front(**kw):
    """A REAL Front (constructor included), no server, fake flip RPCs."""
    kw.setdefault("tp_prefill_max_tokens", X_IDLE)
    kw.setdefault("x_ceiling_tokens", CEIL)
    kw.setdefault("d_admit_max_tokens", 0)
    f = front_mod.Front(
        prefill="http://127.0.0.1:9", decode="http://127.0.0.1:9", awake="D", tag="rc7x",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, **kw,
    )
    f.stops = []

    async def rpc(g, path, body, timeout):
        if path == "/flush_cache":
            return 200, "{}"
        await asyncio.sleep(0.001)
        tags = tuple((body or {}).get("tags", ()))
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})

    f.rpc = rpc
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    return f


def _pending(rid, tokens, *, eligible=True, deferred=True, t_arrive=None):
    fut = asyncio.get_running_loop().create_future()
    return Pending(rid, "/generate", {}, "x", t_arrive or time.time(), fut,
                   est_prompt=tokens, est_uncached=tokens, d_eligible=eligible,
                   x_deferred=deferred)


async def _cancel(task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


async def _arrive(f, rid, uncached, delay=0.0):
    """handle_generate's two arrival stamps, then the real grant decision."""
    await asyncio.sleep(delay)
    f._rid += 1
    seq = f._rid
    f._x_last_arrival = time.time()
    refused, deferred = [], []
    seat = await f._x_short_grant(rid, uncached, uncached, seq, refused, deferred)
    return seat, deferred


# ===========================================================================
# PART 1 -- the r_D instrument
# ===========================================================================


class RdSampleArithmetic(CustomTestCase):
    def test_the_prefill_clock_is_the_denominator(self):
        rate, why = r_d_sample(8000, 3.9, True)
        self.assertEqual(why, "d_prefill_clock")
        self.assertAlmostEqual(rate, 8000 / 3.9)

    def test_there_is_no_wall_argument_at_all(self):
        """The danger direction: a wall parameter is how the decode got back in."""
        params = inspect.signature(r_d_sample).parameters
        self.assertFalse([p for p in params if "wall" in p], params)

    def test_no_clock_is_no_sample_never_the_wall(self):
        self.assertEqual(r_d_sample(8000, None, True), (None, "no_prefill_clock"))
        self.assertEqual(r_d_sample(8000, 0.0, True), (None, "bad_prefill_clock"))

    def test_the_rc4_sample_is_refused(self):
        # weg2rc4 weg2-0-1: prompt 4316, cached 4314 -> uncached 2, wall 6.85 s
        self.assertEqual(r_d_sample(2, 0.004, True), (None, "below_min"))
        self.assertEqual(r_d_sample(0, 1.0, True), (None, "below_min"))

    def test_the_minimum_is_2048_both_sides(self):
        self.assertEqual(X_RD_MIN_UNCACHED, 2048)
        self.assertIsNone(r_d_sample(2047, 1.0, True)[0])
        self.assertIsNotNone(r_d_sample(2048, 1.0, True)[0])

    def test_a_concurrent_leg_is_refused(self):
        self.assertEqual(r_d_sample(8000, 3.9, False), (None, "concurrent"))

    def test_mutant_leg2_no_longer_divides_by_its_wall(self):
        src = inspect.getsource(Front.leg2)
        self.assertNotIn("_unc / _w", src, "the whole-wall r_D is back")
        self.assertIn("r_d_sample(_unc, _ps, _solo)", src)
        # both wire shapes sample
        self.assertEqual(src.count("_sample_r_d(pt, ct, comp, verdict, dterms)"), 2)


class PrefillClockDSide(CustomTestCase):
    def setUp(self):
        prefill_clock._reset_for_tests()

    tearDown = setUp

    @staticmethod
    def _req(rid, fe, pf, prompt=8200, cached=200):
        return types.SimpleNamespace(
            rid=rid, origin_input_ids=list(range(prompt)), cached_tokens=cached,
            time_stats=types.SimpleNamespace(forward_entry_time=fe, prefill_finished_time=pf))

    D = types.SimpleNamespace(tp_prefill_max_tokens=CEIL)

    def test_armed_only_on_the_weg2_d_group(self):
        self.assertTrue(prefill_clock.armed(self.D))
        self.assertFalse(prefill_clock.armed(types.SimpleNamespace(tp_prefill_max_tokens=0)))
        self.assertFalse(prefill_clock.armed(types.SimpleNamespace()))

    def test_the_clock_is_prefill_finished_minus_forward_entry(self):
        prefill_clock.note_prefill_finished(self._req("r1", 100.0, 103.9), self.D)
        snap = prefill_clock.snapshot()
        self.assertAlmostEqual(snap["r1"]["s"], 3.9)
        self.assertEqual((snap["r1"]["prompt"], snap["r1"]["cached"]), (8200, 200))

    def test_nothing_off_the_d_group_or_without_both_stamps(self):
        prefill_clock.note_prefill_finished(self._req("p", 1.0, 2.0),
                                            types.SimpleNamespace(tp_prefill_max_tokens=0))
        prefill_clock.note_prefill_finished(self._req("u", 0.0, 2.0), self.D)
        prefill_clock.note_prefill_finished(self._req("i", 5.0, 2.0), self.D)
        self.assertEqual(prefill_clock.snapshot(), {})

    def test_the_ring_is_bounded(self):
        for i in range(prefill_clock.RING_MAX + 7):
            prefill_clock.note_prefill_finished(self._req(f"r{i}", 1.0, 2.0), self.D)
        snap = prefill_clock.snapshot()
        self.assertEqual(len(snap), prefill_clock.RING_MAX)
        self.assertNotIn("r0", snap)

    def test_lookup_by_rid_then_by_shape(self):
        blk = {"a": {"s": 1.0, "prompt": 8200, "cached": 200},
               "d-own": {"s": 2.0, "prompt": 9000, "cached": 0}}
        self.assertEqual(prefill_clock.lookup(blk, "a", 1, 1), 1.0)
        # /v1/messages: D ran it under its own rid -- matched by its usage
        self.assertEqual(prefill_clock.lookup(blk, "weg2-3-9", 9000, 0), 2.0)
        self.assertIsNone(prefill_clock.lookup(blk, "weg2-3-9", 9000, 1))
        self.assertIsNone(prefill_clock.lookup({}, "a", 8200, 200))
        self.assertIsNone(prefill_clock.lookup(None, "a", 8200, 200))

    def test_writer_sites_in_the_scheduler(self):
        from sglang.srt.managers import scheduler as sched_mod
        from sglang.srt.managers.scheduler_components import batch_result_processor as brp

        src = inspect.getsource(brp.SchedulerBatchResultProcessor.process_batch_result_prefill)
        i = src.index("req.time_stats.set_prefill_finished_time()")
        self.assertIn("_weg2_prefill_clock.note_prefill_finished(req, self.server_args)",
                      src[i:i + 500])
        gis = inspect.getsource(sched_mod.Scheduler.get_internal_state)
        self.assertIn("ret[_weg2_prefill_clock.INTERNAL_STATE_KEY] = _weg2_prefill_clock.snapshot()", gis)
        self.assertIn("_weg2_prefill_clock.armed(self.server_args)", gis)

    def test_the_front_reads_what_d_publishes(self):
        """Writer (snapshot) -> the internal_states[0] body -> reader (_draft_terms)."""
        prefill_clock.note_prefill_finished(self._req("weg2-1-1", 10.0, 12.5), self.D)
        body = {"internal_states": [{prefill_clock.INTERNAL_STATE_KEY: prefill_clock.snapshot()}]}

        class _R:
            status = 200

            async def json(self):
                return body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        me = types.SimpleNamespace(session=types.SimpleNamespace(get=lambda url: _R()))
        g = types.SimpleNamespace(url="http://d")
        out = asyncio.run(Front._draft_terms(me, g, None, rid="weg2-1-1", pt=8200, ct=200))
        self.assertAlmostEqual(out["prefill_s"], 2.5)
        out2 = asyncio.run(Front._draft_terms(me, g, None, rid="other", pt=1, ct=0))
        self.assertIsNone(out2["prefill_s"])


# --------------------------------------------------------------------------
# the probe end to end: a real front, a fake D that decodes long after it
# prefilled, both wire shapes
# --------------------------------------------------------------------------


class FakeD:
    PROMPT, CACHED, COMP = 8200, 200, 300

    def __init__(self, *, prefill_s=0.2, wall_s=0.6, publish="rid"):
        self.prefill_s, self.wall_s, self.publish = prefill_s, wall_s, publish
        self.block = {}
        self.server = None
        self.url = ""

    async def _chat(self, request):
        body = await request.json()
        key = body.get("rid") if self.publish == "rid" else "d-own-rid"
        if self.publish != "none":
            self.block[key] = {"s": self.prefill_s, "prompt": self.PROMPT, "cached": self.CACHED}
        await asyncio.sleep(self.wall_s)  # prefill AND the decode of COMP tokens
        usage = {"prompt_tokens": self.PROMPT, "completion_tokens": self.COMP,
                 "prompt_tokens_details": {"cached_tokens": self.CACHED}}
        if body.get("stream"):
            resp = web.StreamResponse()
            resp.content_type = "text/event-stream"
            await resp.prepare(request)
            await resp.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
            await resp.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response({"choices": [{"message": {"content": "hi"}}], "usage": usage})

    async def _info(self, request):
        return web.json_response(
            {"internal_states": [{prefill_clock.INTERNAL_STATE_KEY: dict(self.block)}]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/get_server_info", self._info)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")


async def _probe(stream, publish="rid", n=1):
    d = FakeD(publish=publish)
    await d.start()
    f = front_mod.Front(
        prefill="http://127.0.0.1:9", decode=d.url, awake="D", tag="rc7x", store_dir="/tmp",
        prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        tp_prefill_max_tokens=X_IDLE, x_ceiling_tokens=CEIL, d_admit_max_tokens=0)
    f.session = ClientSession(timeout=ClientTimeout(total=30))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", f.handle_generate)
    srv = TestServer(app)
    await srv.start_server()
    client = ClientSession(timeout=ClientTimeout(total=30))
    try:
        async def one(i):
            body = {"model": "m", "stream": stream,
                    "messages": [{"role": "user", "content": f"q{i} " + "w" * 3000}]}
            async with client.post(str(srv.make_url("/v1/chat/completions")), json=body) as r:
                return r.status, await r.text()
        res = await asyncio.gather(*(one(i) for i in range(n)))
    finally:
        await client.close()
        await f.session.close()
        await srv.close()
        await d.server.close()
    return f, res


class TheProbeEndToEnd(CustomTestCase):
    UNC = FakeD.PROMPT - FakeD.CACHED

    def _check_sampled(self, f, res):
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(list(f._x_samples["r_d"]), [self.UNC / 0.2],
                         "r_D must be uncached / D's prefill clock (0.2 s)")
        self.assertNotAlmostEqual(f._x_samples["r_d"][0], self.UNC / 0.6, delta=1000,
                                  msg="the leg wall (prefill + decode) is the weg2rc4 defect")
        self.assertEqual(f.counters["r_d_sampled"], 1)
        self.assertTrue(f._x_r_d_src.startswith("d_prefill_clock solo"), f._x_r_d_src)

    def test_non_streamed_leg_samples_the_prefill_clock(self):
        self._check_sampled(*asyncio.run(_probe(stream=False)))

    def test_streamed_leg_samples_too(self):
        # it used to sample on the non-streamed branch only
        self._check_sampled(*asyncio.run(_probe(stream=True)))

    def test_messages_shape_matches_by_usage_when_d_dropped_the_rid(self):
        self._check_sampled(*asyncio.run(_probe(stream=False, publish="shape")))

    def test_no_clock_means_no_sample_never_the_wall(self):
        f, res = asyncio.run(_probe(stream=False, publish="none"))
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(len(f._x_samples["r_d"]), 0)
        self.assertEqual(f.counters["r_d_skipped_no_prefill_clock"], 1)

    def test_two_overlapping_legs_are_not_samples(self):
        f, res = asyncio.run(_probe(stream=False, n=2))
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(len(f._x_samples["r_d"]), 0)
        self.assertGreaterEqual(f.counters["r_d_skipped_concurrent"], 1)


class LiveXAndTheCeiling(CustomTestCase):
    """RC4 arithmetic (computed, not measured): r_D ~2050, r_P ~7500, flip_s ~1.8
    -> X* ~10.2k. The re-solve leaves the 4096 floor -- up to the ceiling."""

    def _solve(self, **kw):
        async def body():
            f = _front(tp_prefill_max_tokens=4096, **kw)
            f.note_x_sample("flip_s", 1.8)
            f.note_x_sample("r_p", 7500.0)
            f.note_x_sample("r_d", 2050.0)  # completes the triple -> re-solve
            return f
        return asyncio.run(body())

    def test_x_rises_above_the_floor_to_x_star(self):
        from sglang.srt.weg2.launcher import derive_x_star

        f = self._solve(x_ceiling_tokens=CEIL)
        want = derive_x_star(1.8, 2050.0, 7500.0, 4096)
        self.assertGreater(want, 9500)
        self.assertEqual(f.tp_prefill_max_tokens, want)
        self.assertEqual(f.flip_min_work_tokens, want, "C7 follows X")
        self.assertEqual(f._x_busy_in_force(), 4096)

    def test_the_ceiling_clamps_it(self):
        f = self._solve(x_ceiling_tokens=8192)
        self.assertEqual(f.tp_prefill_max_tokens, 8192)
        self.assertEqual(f.counters["x_ceiling_clamped"], 1)

    def test_unset_ceiling_is_the_launch_x_so_x_cannot_rise(self):
        f = self._solve(x_ceiling_tokens=0)
        self.assertEqual(f.x_ceiling_tokens, 4096)
        self.assertEqual(f.tp_prefill_max_tokens, 4096)

    def test_units_refusal_stays_sharp(self):
        from sglang.srt.weg2.launcher import MixedRateUnits, derive_x_star

        with self.assertRaises(MixedRateUnits):
            derive_x_star(1.8, 2050.0, 7500.0, 4096, unit_d="request_latency")
        src = inspect.getsource(Front.resolve_x_live)
        self.assertIn("unit_d=RATE_UNIT_GROUP_THROUGHPUT", src)
        self.assertIn("unit_p=RATE_UNIT_GROUP_THROUGHPUT", src)
        self.assertIn("except MixedRateUnits", src)

    def test_the_lines_carry_x_busy_and_the_r_d_source(self):
        f = self._solve(x_ceiling_tokens=CEIL)
        line = f.idle_policy_line()
        for want in ("X_busy=4096", f"X_ceiling={CEIL}", "r_d_src=d_prefill_clock",
                     "r_d_min_uncached=2048", "x_solo_window_ms=250"):
            self.assertIn(want, line)
        src = inspect.getsource(Front.resolve_x_live)
        self.assertIn("X*=%d X_ceiling=%s X_busy=%d", src)


# ===========================================================================
# PART 2 -- the busy/idle split
# ===========================================================================


class GrantDecision(CustomTestCase):
    def test_idle_d_takes_an_x_idle_prompt_after_a_quiet_window(self):
        async def body():
            f = _front()
            t0 = time.time()
            seat, deferred = await _arrive(f, "lone", 8000)
            return f, seat, deferred, time.time() - t0

        f, seat, deferred, dt = asyncio.run(body())
        self.assertIsNotNone(seat, "a lone 8k prompt on an idle D stays on D")
        self.assertEqual(deferred, [])
        self.assertGreaterEqual(dt, X_SOLO_WINDOW_S * 0.9)
        self.assertEqual(f.counters["x_idle_granted"], 1)

    def test_busy_d_defers_the_same_prompt(self):
        async def body():
            f = _front()
            f.groups["D"].outstanding["decoding"] = time.time()
            seat, deferred = await _arrive(f, "big", 8000)
            return f, seat, deferred

        f, seat, deferred = asyncio.run(body())
        self.assertIsNone(seat, "while D decodes, 8k > X_busy must not go to D")
        self.assertEqual(deferred, ["busy"])
        self.assertEqual(f._seats_in_use(), 0)

    def test_busy_d_still_takes_a_prompt_within_x_busy(self):
        async def body():
            f = _front()
            f.groups["D"].outstanding["decoding"] = time.time()
            return await _arrive(f, "small", 3000)

        seat, deferred = asyncio.run(body())
        self.assertIsNotNone(seat)
        self.assertEqual(deferred, [])

    def test_x_busy_zero_means_no_d_prefill_while_d_decodes(self):
        async def body():
            f = _front(x_busy_tokens=0)
            f.groups["D"].outstanding["decoding"] = time.time()
            busy = await _arrive(f, "one", 1)
            f.groups["D"].outstanding.clear()
            idle = await _arrive(f, "two", 1)
            return busy, idle

        busy, idle = asyncio.run(body())
        self.assertIsNone(busy[0])
        self.assertIsNotNone(idle[0])

    def test_x_busy_never_exceeds_x_in_force(self):
        async def body():
            return _front(tp_prefill_max_tokens=4096, x_busy_tokens=9000)._x_busy_in_force()

        self.assertEqual(asyncio.run(body()), 4096)

    def test_a_burst_of_four_8k_all_defers_for_one_flip(self):
        async def body():
            f = _front()
            res = await asyncio.gather(*(_arrive(f, f"b{i}", 8000, 0.01 * i) for i in range(4)))
            return f, res

        f, res = asyncio.run(body())
        self.assertTrue(all(seat is None for seat, _ in res), res)
        self.assertTrue(all(d for _, d in res), res)
        self.assertEqual(f._seats_in_use(), 0, "every window seat went back")

    def test_one_decision_line_per_short_decision(self):
        async def body():
            f = _front()
            with self.assertLogs("weg2.front", level="INFO") as cm:
                await _arrive(f, "lone", 8000)
                f.groups["D"].outstanding["decoding"] = time.time()
                await _arrive(f, "big", 8000)
            return [ln for ln in cm.output if "WEG2 SHORT-DECISION" in ln]

        lines = asyncio.run(body())
        self.assertEqual(len(lines), 2, lines)
        self.assertIn("d_state=idle X_applied=10000 uncached=8000 d_running=0 verdict=grant", lines[0])
        self.assertIn("d_state=busy X_applied=4096 uncached=8000 d_running=1 verdict=defer", lines[1])

    def test_handle_generate_marks_the_deferred_pending(self):
        src = inspect.getsource(Front.handle_generate)
        self.assertIn("seat = await self._x_short_grant(rid, remainder, est_prompt, _x_seq,", src)
        self.assertIn("x_deferred=bool(x_deferred) and not short_refused,", src)
        self.assertIn("d_eligible=short_ok and not short_refused)", src)


# --------------------------------------------------------------------------
# design point (a): the deferred request, when D empties
# --------------------------------------------------------------------------


async def _run_controller(f, until, seconds, events=()):
    """Real controller loop; flips are recorded, never executed. ``events``:
    [(t_offset, fn)] run at those offsets."""
    seen = {}

    async def flip(src, dst):
        seen.setdefault("flip", (src, dst, time.time()))
        f.awake = dst

    async def leg1(p):
        p.leg1_prompt_tokens = 64

    f.flip = flip
    f.leg1 = leg1
    t0 = time.time()
    pending = sorted(events)
    task = asyncio.create_task(f.controller())
    while time.time() - t0 < seconds and not until(seen):
        while pending and time.time() - t0 >= pending[0][0]:
            pending.pop(0)[1]()
        await asyncio.sleep(0.01)
    await _cancel(task)
    return seen


class CaseA(CustomTestCase):
    def _scenario(self, *, flip_min_work=None, queue=((8000, True, True),), quiet_at_idle=False,
                  seconds=1.6):
        async def body():
            kw = {} if flip_min_work is None else {"flip_min_work_tokens": flip_min_work}
            f = _front(**kw)
            D = f.groups["D"]
            D.outstanding["decoding"] = time.time()
            for i, (tok, elig, dfr) in enumerate(queue):
                f.queue.append(_pending(f"q{i}", tok, eligible=elig, deferred=dfr))
            snap = {}

            def d_empties():
                D.outstanding.clear()
                if quiet_at_idle:
                    f._x_last_arrival = time.time()  # an arrival right now: not quiet yet

            def look():
                snap["at_0.55"] = (len(f.queue), len(f._ready_for_d))

            seen = await _run_controller(f, lambda s: "flip" in s, seconds,
                                         events=[(0.4, d_empties), (0.55, look)])
            return f, seen, snap

        return asyncio.run(body())

    def test_served_on_d_without_a_flip_when_min_work_follows_x(self):
        # min_work follows X (10000) > 8000: economics alone would HOLD -> starvation
        f, seen, _ = self._scenario()
        self.assertNotIn("flip", seen)
        self.assertEqual(len(f.queue), 0)
        self.assertEqual([p.rid for p in f._ready_for_d], ["q0"])
        self.assertTrue(f._ready_for_d[0].d_direct)
        self.assertEqual(f.counters["x_idle_regrant_requests"], 1)

    def test_no_needless_flip_even_when_economics_would_flip(self):
        f, seen, _ = self._scenario(flip_min_work=1000)
        self.assertNotIn("flip", seen, "a flip for work D takes itself (X* says it does not pay)")
        self.assertEqual([p.rid for p in f._ready_for_d], ["q0"])

    def test_it_waits_for_the_quiet_window_and_still_does_not_flip(self):
        f, seen, snap = self._scenario(flip_min_work=1000, quiet_at_idle=True)
        self.assertEqual(snap["at_0.55"], (1, 0), "inside the window: neither moved nor flipped")
        self.assertNotIn("flip", seen)
        self.assertEqual([p.rid for p in f._ready_for_d], ["q0"])

    def test_nothing_moves_while_d_decodes(self):
        async def body():
            f = _front(flip_min_work_tokens=1000)
            f.groups["D"].outstanding["decoding"] = time.time()
            f.queue.append(_pending("q0", 8000))
            seen = await _run_controller(f, lambda s: "flip" in s, 0.6)
            return f, seen

        f, seen = asyncio.run(body())
        self.assertNotIn("flip", seen, "drain-and-flip: a decode is never cut")
        self.assertEqual((len(f.queue), len(f._ready_for_d)), (1, 0))

    def test_law_1_a_long_request_keeps_everything_for_p(self):
        f, seen, _ = self._scenario(queue=((8000, True, True), (30000, False, False)))
        self.assertIn("flip", seen)
        self.assertEqual(seen["flip"][:2], ("D", "P"))
        self.assertEqual(len(f._ready_for_d), 0)

    def test_law_4_a_backlog_above_x_idle_flips(self):
        # the 4 x 8k burst after it deferred: 32k > X_idle -> one flip, P prefills all
        f, seen, _ = self._scenario(queue=tuple((8000, True, True) for _ in range(4)))
        self.assertIn("flip", seen)
        self.assertEqual(len(f._ready_for_d), 0)

    def test_fairness_wins(self):
        async def body():
            f = _front()
            f.admit_d = False
            f.queue.append(_pending("q0", 8000))
            return f._x_idle_regrant(time.time())

        self.assertEqual(asyncio.run(body()), "none")

    def test_short_drain_b_is_bounded_by_x_busy_while_d_decodes(self):
        async def body():
            f = _front(d_short_drain_tokens=CEIL)
            f.groups["D"].outstanding["decoding"] = time.time()
            f.queue.append(_pending("s0", 6000, deferred=False))
            busy = f._d_short_drain(time.time())
            f.groups["D"].outstanding.clear()
            idle = f._d_short_drain(time.time())
            return busy, idle

        self.assertEqual(asyncio.run(body()), (0, 1))

    def test_the_controller_order(self):
        src = inspect.getsource(Front.controller)
        d_arm = src[src.index('if self.awake == "D":'):src.index("t_drain0 = time.time()")]
        self.assertLess(d_arm.index("self._d_short_drain("), d_arm.index("self._x_idle_regrant("))
        self.assertLess(d_arm.index("self._x_idle_regrant("), d_arm.index("self._fairness_switch("))
        econ = d_arm[d_arm.index("if (d_work_exhausted or not self.admit_d) and not handing_off:"):]
        self.assertLess(econ.index('_x_rg == "wait" and self.admit_d and not fairness_fired'),
                        econ.index("self._flip_economics_ok(fairness_fired)"))


# ===========================================================================
# the launcher -> front seam, at the argv: writer AND reader
# ===========================================================================


class LauncherFrontSeam(CustomTestCase):
    def _ns(self, *extra):
        from sglang.srt.weg2 import launcher as L

        return L, L.build_parser().parse_args(["--tree", "/x", "--tag", "t", *extra])

    def _argv(self, *extra):
        L, ns = self._ns(*extra)
        return L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, 4096, 4096, "D")

    def test_writer_unset_ships_the_argv_as_before(self):
        argv = self._argv()
        self.assertNotIn("--x-ceiling-tokens", argv)
        self.assertNotIn("--x-busy-tokens", argv)

    def test_writer_set_reaches_the_front(self):
        argv = self._argv("--x-ceiling-tokens", str(CEIL), "--x-busy-tokens", "2048")
        self.assertEqual(argv[argv.index("--x-ceiling-tokens") + 1], str(CEIL))
        self.assertEqual(argv[argv.index("--x-busy-tokens") + 1], "2048")

    def _read(self, argv):
        """The front's OWN main() parses the argv the launcher wrote."""
        got = {}

        class Stop(Exception):
            pass

        def capture(*a, **kw):
            got.update(kw)
            raise Stop()

        with mock.patch.object(front_mod, "Front", capture), \
                mock.patch.object(sys, "argv", ["front"] + argv[3:]):
            with self.assertRaises(Stop):
                front_mod.main()
        return got

    def test_reader_parses_what_the_writer_wrote(self):
        got = self._read(self._argv("--x-ceiling-tokens", str(CEIL), "--x-busy-tokens", "2048"))
        self.assertEqual((got["x_ceiling_tokens"], got["x_busy_tokens"]), (CEIL, 2048))
        self.assertEqual(got["tp_prefill_max_tokens"], 4096)

    def test_reader_defaults_when_the_writer_wrote_nothing(self):
        got = self._read(self._argv())
        self.assertEqual((got["x_ceiling_tokens"], got["x_busy_tokens"]),
                         (0, front_mod.X_BUSY_DEFAULT_TOKENS))

    def test_d_riegel_is_the_ceiling(self):
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(L.resolve_x_ceiling(None, 4096, None)[0], 4096)
        self.assertEqual(L.resolve_x_ceiling(CEIL, 4096, 2048)[0], CEIL)
        src = inspect.getsource(L.main)
        self.assertEqual(src.count("max_kv_per_request, x_d_riegel, ns.num_continuous_decode_steps"), 2,
                         "both argv_d calls (dry and real) arm D's W50 riegel at the ceiling")
        self.assertIn("--tp-prefill-max-tokens {x_d_riegel}", src)

    def test_argv_d_carries_the_riegel(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.argv_d)
        self.assertIn('"--tp-prefill-max-tokens", str(x_tokens)', src)

    def test_refusals(self):
        from sglang.srt.weg2 import launcher as L

        with self.assertRaises(SystemExit) as cm:
            L.resolve_x_ceiling(2048, 4096, None)
        self.assertIn("W154", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            L.resolve_x_ceiling(CEIL, 4096, -1)
        self.assertIn("W155", str(cm.exception))
        self.assertEqual(L.resolve_x_ceiling(CEIL, 4096, 0)[0], CEIL, "0 is legal")


if __name__ == "__main__":
    unittest.main()
