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
    r_d_probe,
    x_rd_min_uncached,
    x_solo_window_s,
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


M = X_RD_MIN_UNCACHED


class RdSampleArithmetic(CustomTestCase):
    """r_d_probe: the NF line's H84 contract and name (one function, both lines)."""

    def test_the_prefill_time_is_the_denominator(self):
        rate, why = r_d_probe(8000, 3.9, M)
        self.assertEqual(why, "sample")
        self.assertAlmostEqual(rate, 8000 / 3.9)

    def test_there_is_no_wall_argument_at_all(self):
        """The danger direction: a wall parameter is how the decode got back in."""
        params = inspect.signature(r_d_probe).parameters
        self.assertFalse([p for p in params if "wall" in p], params)

    def test_no_prefill_time_is_no_sample_never_the_wall(self):
        self.assertEqual(r_d_probe(8000, None, M), (None, "no_prefill_time"))
        self.assertEqual(r_d_probe(8000, 0.0, M), (None, "no_prefill_time"))

    def test_the_rc4_sample_is_refused(self):
        # weg2rc4 weg2-0-1: prompt 4316, cached 4314 -> uncached 2, wall 6.85 s
        self.assertEqual(r_d_probe(2, 0.004, M), (None, "short"))
        self.assertEqual(r_d_probe(0, 1.0, M), (None, "short"))

    def test_the_minimum_is_2048_both_sides_and_an_env_knob(self):
        self.assertEqual((X_RD_MIN_UNCACHED, x_rd_min_uncached()), (2048, 2048))
        self.assertIsNone(r_d_probe(2047, 1.0, M)[0])
        self.assertIsNotNone(r_d_probe(2048, 1.0, M)[0])
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_X_RD_MIN_UNCACHED": "1000",
                                          "SGLANG_WEG2_X_SOLO_WINDOW_MS": "100"}):
            self.assertEqual(x_rd_min_uncached(), 1000)
            self.assertAlmostEqual(x_solo_window_s(), 0.1)
        self.assertAlmostEqual(x_solo_window_s(), X_SOLO_WINDOW_S)

    def test_a_concurrent_leg_is_refused_before_the_probe(self):
        src = inspect.getsource(Front.leg2)
        i = src.index("if not _solo:")
        self.assertIn('self.counters["r_d_skipped_concurrent"] += 1', src[i:i + 200])
        self.assertLess(i, src.index("r_d_probe(_unc, _ps, x_rd_min_uncached())"))

    def test_mutant_leg2_no_longer_divides_by_its_wall(self):
        src = inspect.getsource(Front.leg2)
        self.assertNotIn("_unc / _w", src, "the whole-wall r_D is back")
        self.assertIn("r_d_probe(_unc, _ps, x_rd_min_uncached())", src)
        # both wire shapes sample
        self.assertEqual(src.count("_sample_r_d(pt, ct, verdict, dterms)"), 2)


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
        rec = snap["recent"][-1]
        self.assertAlmostEqual(rec["s"], 3.9)
        self.assertEqual((rec["rid"], rec["prompt"], rec["cached"], rec["seq"]), ("r1", 8200, 200, 1))
        self.assertEqual((snap["boot"], snap["seq"]), (prefill_clock.BOOT, 1))

    def test_nothing_off_the_d_group_unstamped_takes_a_seq_only(self):
        prefill_clock.note_prefill_finished(self._req("p", 1.0, 2.0),
                                            types.SimpleNamespace(tp_prefill_max_tokens=0))
        self.assertEqual(prefill_clock.snapshot()["seq"], 0, "not a Weg-2 D: nothing at all")
        prefill_clock.note_prefill_finished(self._req("u", 0.0, 2.0), self.D)
        prefill_clock.note_prefill_finished(self._req("i", 5.0, 2.0), self.D)
        snap = prefill_clock.snapshot()
        self.assertEqual((snap["seq"], snap["recent"]), (2, []),
                         "a prefill without both stamps is visible as a seq, never as a value")

    def test_the_ring_is_bounded(self):
        for i in range(prefill_clock.RING_MAX + 7):
            prefill_clock.note_prefill_finished(self._req(f"r{i}", 1.0, 2.0), self.D)
        snap = prefill_clock.snapshot()
        self.assertEqual(len(snap["recent"]), prefill_clock.RING_MAX)
        self.assertEqual(snap["seq"], prefill_clock.RING_MAX + 7)
        self.assertNotIn("r0", [r["rid"] for r in snap["recent"]])

    def test_attribution_needs_a_record_newer_than_the_mark(self):
        """The NF line's H85 rule (Operator 25.09.: the shape fallback may hit an
        OLD record of the same shape): newer than the leg's entry mark AND (its
        rid OR the only new prefill) AND D's prompt - cached == the leg's
        uncached; else no sample, the reason in d_prefill_attr."""
        B = "b"

        def blk(seq, *recs):
            return {"boot": B, "seq": seq, "recent": [
                {"seq": q, "rid": r, "s": s, "prompt": p, "cached": c} for q, r, s, p, c in recs]}

        att = prefill_clock.attribute
        old_same_shape = (1, "old", 9.0, 8200, 200)
        mine = (2, "weg2-3-9", 2.0, 8200, 200)
        d_own = (2, "d-own", 2.0, 8200, 200)
        self.assertEqual(att(blk(2, old_same_shape, mine), "weg2-3-9", 8000, (B, 1)), (2.0, "rid"))
        self.assertEqual(att(blk(2, old_same_shape, d_own), "weg2-3-9", 8000, (B, 1)),
                         (2.0, "sole_new"), "/v1/messages before the rid fix")
        self.assertEqual(att(blk(1, old_same_shape), "weg2-3-9", 8000, (B, 1)), (None, "absent"),
                         "an OLD record of the same shape is never this leg's")
        self.assertEqual(att(blk(2, old_same_shape, mine), "weg2-3-9", 8000, None),
                         (None, "no_mark"), "the first leg of a boot has no mark")
        self.assertEqual(att(blk(3, old_same_shape, d_own, (3, "hc", 0.1, 5, 0)), "weg2-3-9", 8000,
                             (B, 1)), (None, "ambiguous(new=2)"))
        self.assertEqual(att(blk(2, mine), "weg2-3-9", 7999, (B, 1))[0], None, "extent mismatch")
        self.assertEqual(att(blk(2, mine), "weg2-3-9", 8000, ("other-boot", 1)),
                         (None, "d_restarted"))

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

    def test_the_body_field_wins_over_server_info(self):
        """NF H84's carrier (meta_info / sglext weg2_prefill_s) first, the 27B's
        internal_states[0] block second -- same name, same quantity."""
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
        mark = (prefill_clock.BOOT, 0)
        out = asyncio.run(Front._draft_terms(me, g, {"sglext": {"weg2_prefill_s": 1.25}},
                                             rid="weg2-1-1", uncached=8000, mark=mark))
        self.assertEqual((out["prefill_s"], out["prefill_src"], out["prefill_attr"]),
                         (1.25, "body", "body"))
        out = asyncio.run(Front._draft_terms(me, g, {"usage": {}}, rid="weg2-1-1",
                                             uncached=8000, mark=mark))
        self.assertEqual((out["prefill_s"], out["prefill_src"], out["prefill_attr"]),
                         (2.5, "server_info", "rid"))

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
        out = asyncio.run(Front._draft_terms(me, g, None, rid="weg2-1-1", uncached=8000,
                                             mark=(prefill_clock.BOOT, 0)))
        self.assertAlmostEqual(out["prefill_s"], 2.5)
        self.assertEqual(me._d_prefill_mark, (prefill_clock.BOOT, 1), "every read moves the mark")
        out2 = asyncio.run(Front._draft_terms(me, g, None, rid="weg2-1-1", uncached=8000,
                                              mark=me._d_prefill_mark))
        self.assertEqual((out2["prefill_s"], out2["prefill_attr"]), (None, "absent"),
                         "the same record read again is not newer than the mark")


# --------------------------------------------------------------------------
# the probe end to end: a real front, a fake D that decodes long after it
# prefilled, both wire shapes
# --------------------------------------------------------------------------


class FakeD:
    PROMPT, CACHED, COMP = 8200, 200, 300

    BOOT = "fake-d-boot"

    def __init__(self, *, prefill_s=0.2, wall_s=0.6, publish="rid"):
        self.prefill_s, self.wall_s, self.publish = prefill_s, wall_s, publish
        self.seq = 0
        self.recent = []
        self.server = None
        self.url = ""

    async def _chat(self, request):
        body = await request.json()
        key = body.get("rid") if self.publish == "rid" else "d-own-rid"
        self.seq += 1  # every finished prefill takes a seq (prefill_clock)
        if self.publish not in ("none", "body"):
            self.recent.append({"seq": self.seq, "rid": key, "s": self.prefill_s,
                                "prompt": self.PROMPT, "cached": self.CACHED})
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
        out = {"choices": [{"message": {"content": "hi"}}], "usage": usage}
        if self.publish == "body":  # the NF H84 carrier: sglext on the OpenAI wire
            out["sglext"] = {"weg2_prefill_s": self.prefill_s}
        return web.json_response(out)

    async def _info(self, request):
        return web.json_response({"internal_states": [{prefill_clock.INTERNAL_STATE_KEY: {
            "boot": self.BOOT, "seq": self.seq, "recent": list(self.recent)}}]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/get_server_info", self._info)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")


async def _probe(stream, publish="rid", n=1, busy=False, prompt=None, cached=None, mark=True):
    d = FakeD(publish=publish)
    if prompt is not None:
        d.PROMPT, d.CACHED = prompt, cached
    await d.start()
    f = front_mod.Front(
        prefill="http://127.0.0.1:9", decode=d.url, awake="D", tag="rc7x", store_dir="/tmp",
        prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        tp_prefill_max_tokens=X_IDLE, x_ceiling_tokens=CEIL, d_admit_max_tokens=0)
    f.session = ClientSession(timeout=ClientTimeout(total=30))
    if mark:  # a previous /get_server_info read of this D (not the boot's first leg)
        f._d_prefill_mark = (FakeD.BOOT, d.seq)
    if busy:  # D decodes another request (Review V (3))
        f.groups["D"].outstanding["decoding-other"] = time.time()
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
        self.assertEqual(f.counters["r_d_sampled_server_info"] + f.counters["r_d_sampled_body"], 1)
        self.assertTrue(f._x_r_d_src.startswith("d_prefill_s verdict="), f._x_r_d_src)

    def test_non_streamed_leg_samples_the_prefill_clock(self):
        self._check_sampled(*asyncio.run(_probe(stream=False)))

    def test_streamed_leg_samples_too(self):
        # it used to sample on the non-streamed branch only
        self._check_sampled(*asyncio.run(_probe(stream=True)))

    def test_the_nf_body_carrier_alone_suffices(self):
        self._check_sampled(*asyncio.run(_probe(stream=False, publish="body")))

    def test_d_own_rid_is_attributed_as_the_sole_new_prefill(self):
        f, res = asyncio.run(_probe(stream=False, publish="shape"))
        self._check_sampled(f, res)
        self.assertEqual(f.counters["r_d_sampled_server_info"], 1)

    def test_the_first_leg_of_a_boot_has_no_mark_and_no_sample(self):
        f, res = asyncio.run(_probe(stream=True, mark=False))
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(len(f._x_samples["r_d"]), 0)
        self.assertEqual(f.counters["r_d_skipped_no_prefill_time"], 1)
        self.assertEqual(f._d_prefill_mark, (FakeD.BOOT, 1), "its read sets the mark")

    def test_no_clock_means_no_sample_never_the_wall(self):
        f, res = asyncio.run(_probe(stream=False, publish="none"))
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual(len(f._x_samples["r_d"]), 0)
        self.assertEqual(f.counters["r_d_skipped_no_prefill_time"], 1)

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

    def test_the_line_names_which_bound_acted(self):
        with self.assertLogs("weg2.front", level="INFO") as cm:
            self._solve(x_ceiling_tokens=8192)
        line = [ln for ln in cm.output if "WEG2 X RE-SOLVE n=" in ln][-1]
        self.assertIn("X=8192 <- X_prev=4096", line)
        self.assertIn("clamp=ceiling floor=4096 ceiling=8192", line)
        self.assertIn("r_d source=live n=1", line)
        self.assertIn("X_busy=4096", line)

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
        for want in ("X_busy=4096", f"X_ceiling={CEIL}", "r_d_src=d_prefill_s",
                     "r_d_min_uncached=2048", "x_solo_window_ms=250"):
            self.assertIn(want, line)
        src = inspect.getsource(Front.resolve_x_live)
        self.assertIn('"WEG2 X RE-SOLVE n=%d X=%d <- X_prev=%d X*=%d clamp=%s floor=%d "', src)
        self.assertIn('" X_busy=%d (27B', src)
        st = f.state_dict()
        self.assertEqual((st["x_busy_tokens"], st["x_ceiling_tokens"]), (4096, CEIL))
        self.assertGreater(st["x_tokens"], 4096)


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
        self.assertEqual(f.counters["x_solo_d"], 1)

    def test_busy_d_defers_the_same_prompt(self):
        async def body():
            f = _front()
            f.groups["D"].outstanding["decoding"] = time.time()
            seat, deferred = await _arrive(f, "big", 8000)
            return f, seat, deferred

        f, seat, deferred = asyncio.run(body())
        self.assertIsNone(seat, "while D decodes, 8k > X_busy must not go to D")
        self.assertEqual(deferred, ["d_outstanding=1"])
        self.assertEqual(f.counters["x_solo_p"], 1)
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
                await _arrive(f, "small", 3000)
            return [ln for ln in cm.output if "WEG2 X-SOLO rid=" in ln]

        lines = asyncio.run(body())
        self.assertEqual(len(lines), 3, lines)
        # the NF line's X-SOLO shape + the 27B fields (busy, x_applied, d_running)
        self.assertIn("rid=lone uncached=8000 X_live=10000 verdict=d reason=solo busy=0 "
                      "x_applied=10000 d_running=0", lines[0])
        self.assertIn("rid=big uncached=8000 X_live=10000 verdict=p reason=d_outstanding=1 busy=1 "
                      "x_applied=4096 d_running=1", lines[1])
        self.assertIn("rid=small uncached=3000 X_live=10000 verdict=d reason=within_x_busy busy=1 "
                      "x_applied=4096", lines[2])

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
        self.assertEqual(L.resolve_x_ceiling(0, 4096, None)[0], 4096, "0 = off (NF H84)")
        c, line = L.resolve_x_ceiling(CEIL, 4096, 2048)
        self.assertEqual(c, CEIL)
        self.assertTrue(line.startswith(f"X CEILING: --x-ceiling-tokens {CEIL} -- group D "
                                        f"--tp-prefill-max-tokens {CEIL}"), line)
        self.assertTrue(L.resolve_x_ceiling(0, 4096, None)[1].startswith("X CEILING: off"))
        src = inspect.getsource(L.main)
        self.assertEqual(src.count("max_kv_per_request, x_d_riegel, ns.num_continuous_decode_steps"), 2,
                         "both argv_d calls (dry and real) arm D's W50 riegel at the ceiling")
        self.assertIn("--tp-prefill-max-tokens {x_d_riegel}", src)

    def test_argv_d_carries_the_riegel(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.argv_d)
        self.assertIn('"--tp-prefill-max-tokens", str(x_tokens)', src)

    def test_below_the_start_x_is_lifted_and_a_negative_x_busy_refused(self):
        from sglang.srt.weg2 import launcher as L

        c, line = L.resolve_x_ceiling(2048, 4096, None)
        self.assertEqual(c, 4096)
        self.assertIn("(asked 2048 < start X 4096: lifted to the start X)", line)
        argv = self._argv("--x-ceiling-tokens", "2048")
        self.assertEqual(argv[argv.index("--x-ceiling-tokens") + 1], "4096",
                         "the front is told the SAME number D's riegel got")
        with self.assertRaises(SystemExit) as cm:
            L.resolve_x_ceiling(CEIL, 4096, -1)
        self.assertIn("W155", str(cm.exception))
        self.assertEqual(L.resolve_x_ceiling(CEIL, 4096, 0)[0], CEIL, "x_busy 0 is legal")
        src = inspect.getsource(L.main)
        self.assertIn("log(x_ceiling_provenance)", src)


# ===========================================================================
# Review V (2026-09-25 ~08:20Z): A1, A2, (3)
# ===========================================================================


class ReviewA1TheWindowNeverLeaksItsSeat(CustomTestCase):
    """A1: the singleton window holds a D seat across an await. Cancelled there
    (client gone), the seat stayed taken -- `_handoff_in_flight()` >= 1 for
    good, D never at rest, no D->P flip again (V-1 measured (1, 1))."""

    def test_a_cancel_inside_the_window_gives_the_seat_back(self):
        async def body():
            f = _front()
            task = asyncio.create_task(_arrive(f, "lone", 8000))
            for _ in range(200):
                if f._x_windows:
                    break
                await asyncio.sleep(0.005)
            opened = (f._x_windows, f._seats_in_use())
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return opened, (f._seats_in_use(), f._handoff_in_flight(), f._x_windows)

        opened, after = asyncio.run(body())
        self.assertEqual(opened, (1, 1), "the window holds its seat while open")
        self.assertEqual(after, (0, 0, 0), "V-1: (seats_in_use, handoff) stayed (1, 1)")

    def test_the_normal_exit_keeps_the_seat_for_the_grant(self):
        async def body():
            f = _front()
            seat, _ = await _arrive(f, "lone", 8000)
            return seat.held, f._seats_in_use()

        self.assertEqual(asyncio.run(body()), (True, 1))

    def test_after_the_cancel_the_idle_d_flips_again(self):
        async def body():
            f = _front(idle_layout="P")
            task = asyncio.create_task(_arrive(f, "lone", 8000))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return await _run_controller(f, lambda s: "flip" in s, 1.0)

        seen = asyncio.run(body())
        self.assertIn("flip", seen, "a leaked seat would hold D awake forever")
        self.assertEqual(seen["flip"][:2], ("D", "P"))


class ReviewA2RequeuePricedAtDsExtent(CustomTestCase):
    """A2: a SHORT D refused (W50) is re-queued for P priced at D's MEASURED
    extent, not at the char estimate that let it through (V-5: est 9000 < X =
    min_work 10000 -> held on an idle D until the fairness bound)."""

    W50 = (b'{"error": {"message": "W50 Weg2TpPrefillExceeded: this group may prefill at '
           b'most 12288 uncached tokens itself (--tp-prefill-max-tokens); this request\'s '
           b'extent after prefix matching is 13000. Refused by name so the caller re-routes '
           b'it through the prefill group -- never prefilled here silently."}}')

    def _requeue(self, body):
        async def run():
            f = _front()  # X = 10000, flip_min_work_tokens follows X
            req = types.SimpleNamespace(path="/v1/chat/completions")
            text = "w" * 27000  # char estimate 27000 // 3 + 1 = 9001 < X
            task = asyncio.create_task(
                f._requeue_after_x_refusal(req, "weg2-1-1", {}, text, False, None, None, body))
            for _ in range(200):
                if f.queue:
                    break
                await asyncio.sleep(0.005)
            est = f.queue[0].est_uncached
            seen = await _run_controller(f, lambda s: "flip" in s, 1.0)
            await _cancel(task)
            return est, seen

        return asyncio.run(run())

    def test_with_ds_extent_the_backlog_is_worth_its_flip(self):
        est, seen = self._requeue(self.W50)
        self.assertEqual(est, 13000)
        self.assertIn("flip", seen, "13000 >= min_work 10000 on an idle D: P prefills it now")
        self.assertEqual(seen["flip"][:2], ("D", "P"))

    def test_without_a_parsed_extent_the_estimate_stands(self):
        est, seen = self._requeue(b'{"error": "W50 Weg2TpPrefillExceeded"}')
        self.assertEqual(est, 9001, "no measurement, no invented number")
        self.assertNotIn("flip", seen)

    def test_a_measurement_never_lowers_the_price(self):
        body = self.W50.replace(b"is 13000", b"is 5000")
        est, _ = self._requeue(body)
        self.assertEqual(est, 9001)


class ReviewX3BusyOverrunIsCounted(CustomTestCase):
    """(3): a SHORT granted within X_busy while D decoded others (busy=1), whose
    REALISED uncached exceeded X_busy -> counter x_busy_overrun + one line."""

    def _run(self, **kw):
        async def body():
            with self.assertLogs("weg2.front", level="INFO") as cm:
                f, res = await _probe(**kw)
            return f, res, [ln for ln in cm.output if "WEG2 X-BUSY-OVERRUN" in ln]

        return asyncio.run(body())

    def test_busy_grant_that_prefilled_more_than_x_busy_is_counted_both_wire_shapes(self):
        for stream in (False, True):
            f, res, lines = self._run(stream=stream, busy=True)  # est ~1k, realised 8000
            self.assertTrue(all(s == 200 for s, _ in res), res)
            self.assertEqual(f.counters["x_busy_overrun"], 1, stream)
            self.assertEqual(len(lines), 1, lines)
            self.assertIn("uncached=8000 x_applied=4096 busy=1 over=3904", lines[0])
            self.assertEqual(f._x_grants, {}, "the grant record is consumed by its leg 2")

    def test_busy_grant_within_x_busy_is_not(self):
        f, res, lines = self._run(stream=False, busy=True, prompt=3000, cached=0)
        self.assertEqual((f.counters["x_busy_overrun"], lines), (0, []))

    def test_idle_grant_is_not_an_x_busy_overrun(self):
        # realised 12000 > X_busy AND > X_idle, but D decoded nobody: no stall
        f, res, lines = self._run(stream=False, busy=False, prompt=12000, cached=0)
        self.assertTrue(all(s == 200 for s, _ in res), res)
        self.assertEqual((f.counters["x_busy_overrun"], lines), (0, []))


class StoreShortTailStaysAtTheLaunchX(CustomTestCase):
    """D's riegel rises to the ceiling; the #1324/#1471 store-short tail (a
    RECOVERY prefill on D, which halts every running decode like a SHORT grant)
    keeps pricing against the launch X. Writer = launcher env for D, reader =
    D's scheduler."""

    def test_writer(self):
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(L.store_short_tail_env(4096, 4096), {}, "unset ceiling: D env as before")
        self.assertEqual(L.store_short_tail_env(4096, CEIL), {L.STORE_SHORT_TAIL_X_ENV: "4096"})
        src = inspect.getsource(L.main)
        self.assertEqual(src.count("env_d.update(store_short_tail_env(x_tokens, x_d_riegel))"), 2)

    def test_reader_both_directions(self):
        from sglang.srt.managers import scheduler as S
        from sglang.srt.weg2 import launcher as L

        self.assertEqual(S.STORE_SHORT_TAIL_X_ENV, L.STORE_SHORT_TAIL_X_ENV, "one name, both ends")
        sched = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=CEIL))
        req = types.SimpleNamespace(_weg2_store_delivered=4000,
                                    full_untruncated_fill_ids=list(range(10000)))  # remainder 6000
        env = {k: v for k, v in os.environ.items() if k != S.STORE_SHORT_TAIL_X_ENV}
        env["SGLANG_WEG2_STORE_SHORT_TAIL"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(S._weg2_store_short_tail_x(sched), CEIL)
            self.assertTrue(S._weg2_store_tail_settles(sched, req), "no cap: priced at the riegel")
        env[S.STORE_SHORT_TAIL_X_ENV] = "4096"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(S._weg2_store_short_tail_x(sched), 4096)
            self.assertFalse(S._weg2_store_tail_settles(sched, req), "capped at the launch X")


if __name__ == "__main__":
    unittest.main()
