# SPDX-License-Identifier: Apache-2.0
"""27B idle policy (user order 2026-09-24, desk/27b-up-idle-layout-0924).

The user asked for three settings on the flip controller:

(a) the resting ("Warte-") layout, D or P -- EXISTS: launcher --idle-layout
    {tp,pp} (K8, default tp) -> front --idle-layout {D,P}; unchanged here.
(b) after the decode, pending prefill work up to N tokens is computed in the
    D layout instead of flipping -- NEW: --d-short-drain-tokens N. What
    existed is per-request (the SHORT route at arrival, X = 4096); a SHORT
    request that was QUEUED (it arrived while P was awake or a flip ran) sat
    behind FLIP-ECONOMICS "hold" while D was awake, released only by the
    fairness bound (45 s) or the next LONG arrival.
(c) when nothing large is pending (or nothing at all), D stays T seconds after
    its work ended in case a short request comes; after T it flips to the
    resting layout -- NEW: --d-hold-s T. What existed is min-dwell, measured
    since the WAKE (a D that decoded for a minute left at once), and for a
    small backlog D cannot serve only the fairness bound -- measured on
    weg2xsn438: rid weg2-8-5 (an image request, 35 est. tokens) waited
    52.0 s behind 227 FLIP-ECONOMICS verdict=hold lines.

DANGER DIRECTIONS guarded here:
* both switches off (the defaults) must keep today's decisions;
* law 4 -- D never prefills above X: a LONG, vision (route forced long),
  carrier or re-queued request is never drained to D, and one of them in the
  queue keeps the whole backlog for P (law 1: P drains everything);
* a hold never starts while D holds work and never spans a flip;
* the launcher ships the front argv byte for byte as before when unset.

The decision methods take the clock as an argument, so the timelines below
are simulated; the controller tests run the real loop with short holds.
Hermetic: no GPU, no boot, no HTTP.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MIB = 1024 * 1024
X = 4096


def _front(**kw):
    kw.setdefault("flip_min_work_tokens", X)
    f = front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="idlepol",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, tp_prefill_max_tokens=X, **kw,
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


def _pending(rid, tokens, *, eligible=True, t_arrive=None):
    fut = asyncio.get_running_loop().create_future()
    return front_mod.Pending(rid, "/generate", {}, "x", t_arrive or time.time(), fut,
                             est_prompt=tokens, est_uncached=tokens, d_eligible=eligible)


async def _cancel(task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# (c) the hold, on a simulated time axis
# ---------------------------------------------------------------------------


class HoldTimeline(CustomTestCase):
    """_note_d_free + _d_hold_expired over explicit clock values."""

    def _walk(self, f, kind, events):
        """events: [(t, free_or_None, expect)] -- free None = no observation."""
        got = []
        for t, free, _ in events:
            if free is not None:
                f._note_d_free(free, t)
            got.append(f._d_hold_expired(kind, t))
        return got

    def test_idle_hold_restarts_when_d_works_again(self):
        f = _front(d_hold_s=5.0)
        ev = [(100.0, True, False),   # D's work ended: the hold starts
              (102.0, True, False),
              (104.9, True, False),
              (105.0, True, True),    # held 5.0 s -> may leave
              (105.5, False, False),  # a SHORT arrival: D works again
              (108.0, True, False),   # work ended again: a NEW hold from 108
              (112.9, True, False),
              (113.0, True, True)]
        self.assertEqual(self._walk(f, "idle", ev), [e[2] for e in ev])

    def test_backlog_hold_runs_on_the_same_clock(self):
        f = _front(d_hold_s=3.0)
        ev = [(10.0, True, False), (12.9, True, False), (13.0, True, True)]
        self.assertEqual(self._walk(f, "backlog", ev), [e[2] for e in ev])

    def test_off_is_todays_decision_for_both_kinds(self):
        f = _front()  # --d-hold-s unset
        for t, free in ((0.0, True), (0.1, False), (50.0, True)):
            f._note_d_free(free, t)
            self.assertTrue(f._d_hold_expired("idle", t), "idle: only min-dwell gates, as today")
            self.assertFalse(f._d_hold_expired("backlog", t), "backlog: only fairness releases, as today")

    def test_zero_hold_releases_at_once_but_only_on_a_free_d(self):
        f = _front(d_hold_s=0.0)
        self.assertFalse(f._d_hold_expired("idle", 1.0), "no free observation yet -> no hold started")
        f._note_d_free(True, 1.0)
        self.assertTrue(f._d_hold_expired("idle", 1.0))
        f._note_d_free(False, 1.5)
        self.assertFalse(f._d_hold_expired("backlog", 1.5), "D holds work -> never leaves on a hold")

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            _front(d_hold_s=1.0)._d_hold_expired("typo", 0.0)

    def test_a_hold_never_spans_a_flip(self):
        async def body():
            f = _front(d_hold_s=5.0)
            f._note_d_free(True, time.time() - 60.0)
            await f.flip("D", "P")
            return f

        f = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertIsNone(f._d_free_since)


# ---------------------------------------------------------------------------
# (b) the SHORT drain, decision only
# ---------------------------------------------------------------------------


class ShortDrainDecision(CustomTestCase):
    def _run(self, n, queue_spec, **state):
        async def body():
            f = _front(d_short_drain_tokens=n)
            for k, v in state.items():
                setattr(f, k, v)
            for i, (tok, elig) in enumerate(queue_spec):
                f.queue.append(_pending(f"r{i}", tok, eligible=elig))
            moved = f._d_short_drain(time.time())
            return f, moved

        return asyncio.run(body())

    def test_a_short_only_backlog_within_n_goes_to_d_oldest_first(self):
        f, moved = self._run(X, [(1000, True), (2000, True)])
        self.assertEqual(moved, 2)
        self.assertEqual(len(f.queue), 0)
        self.assertEqual([p.rid for p in f._ready_for_d], ["r0", "r1"])
        self.assertTrue(all(p.d_direct for p in f._ready_for_d))
        self.assertFalse(f._batch_gate.is_set(), "SHORT arrivals must queue behind the drained backlog")
        self.assertEqual(f.counters["d_short_drain_tokens"], 3000)

    def test_off_by_default(self):
        f, moved = self._run(0, [(1000, True)])
        self.assertEqual((moved, len(f.queue)), (0, 1))

    def test_a_backlog_above_n_stays_for_the_flip(self):
        f, moved = self._run(X, [(3000, True), (2000, True)])
        self.assertEqual((moved, len(f.queue)), (0, 2))

    def test_law_4_one_request_above_x_keeps_everything_for_p(self):
        # even with a generous N: a request above X is P's, and law 1 says P
        # then prefills the WHOLE backlog -- nothing is split off to D.
        f, moved = self._run(10 * X, [(1000, True), (X + 1, True)])
        self.assertEqual((moved, len(f.queue)), (0, 2))

    def test_a_vision_or_long_routed_request_is_never_drained(self):
        f, moved = self._run(X, [(35, False)])  # route forced long (W102) -> not eligible
        self.assertEqual((moved, len(f.queue)), (0, 1))

    def test_a_requeued_request_is_never_drained(self):
        async def body():
            f = _front(d_short_drain_tokens=X)
            p = _pending("rq", 100)
            p.x_requeues = 1
            f.queue.append(p)
            return f._d_short_drain(time.time()), len(f.queue)

        self.assertEqual(asyncio.run(body()), (0, 1))

    def test_only_an_awake_serving_admitting_d_drains(self):
        for state in ({"awake": "P"}, {"state": "flipping"}, {"admit_d": False}):
            f, moved = self._run(X, [(1000, True)], **state)
            self.assertEqual((moved, len(f.queue)), (0, 1), state)


# ---------------------------------------------------------------------------
# the controller, real loop, short holds
# ---------------------------------------------------------------------------


async def _run_controller(f, until, seconds):
    seen = {}
    orig_flip = f.flip

    async def flip(src, dst):
        seen.setdefault("flip", (src, dst, time.time()))
        return await orig_flip(src, dst)

    async def leg1(p):
        p.leg1_prompt_tokens = 64

    f.flip = flip
    f.leg1 = leg1
    t0 = time.time()
    task = asyncio.create_task(f.controller())
    while time.time() - t0 < seconds and not until(seen):
        await asyncio.sleep(0.01)
    await _cancel(task)
    return seen, t0


class ControllerShortDrain(CustomTestCase):
    def _backlog(self, n):
        async def body():
            f = _front(d_short_drain_tokens=n)
            for i, tok in enumerate((1000, 2000)):
                f.queue.append(_pending(f"s{i}", tok))
            seen, _ = await _run_controller(f, lambda s: False, 0.7)
            return f, seen

        return asyncio.run(body())

    def test_switch_on_serves_the_short_backlog_on_d_without_a_flip(self):
        f, seen = self._backlog(X)
        self.assertNotIn("flip", seen)
        self.assertEqual(len(f.queue), 0)
        self.assertEqual([p.rid for p in f._ready_for_d], ["s0", "s1"])

    def test_default_keeps_todays_hold(self):
        # 3000 queued tokens < --flip-min-work-tokens 4096: today the backlog
        # just waits on an awake D (released by fairness or a LONG arrival).
        f, seen = self._backlog(0)
        self.assertNotIn("flip", seen)
        self.assertEqual(len(f.queue), 2)
        self.assertEqual(len(f._ready_for_d), 0)


class ControllerIdleHold(CustomTestCase):
    def _gap(self, **kw):
        async def body():
            f = _front(idle_layout="P", **kw)
            seen, t0 = await _run_controller(f, lambda s: "flip" in s, 3.0)
            return seen, t0, f.stops

        seen, t0, stops = asyncio.run(body())
        self.assertEqual(stops, [])
        self.assertIn("flip", seen)
        self.assertEqual(seen["flip"][:2], ("D", "P"))
        return seen["flip"][2] - t0

    def test_default_idle_flip_to_p_is_not_held(self):
        # first tick at 0.2 s, no flip yet -> min-dwell 0: flips at once.
        self.assertLess(self._gap(), 0.5)

    def test_switch_on_d_rests_the_hold_first(self):
        gap = self._gap(d_hold_s=0.8)
        self.assertGreater(gap, 0.9, f"idle flip after {gap:.2f} s -- the hold did not hold")
        self.assertLess(gap, 1.6)

    def test_idle_layout_d_never_flips_on_a_hold(self):
        async def body():
            f = _front(idle_layout="D", d_hold_s=0.2)
            seen, _ = await _run_controller(f, lambda s: "flip" in s, 0.8)
            return seen

        self.assertNotIn("flip", asyncio.run(body()))


class ControllerBacklogHold(CustomTestCase):
    """A small backlog D cannot serve (the weg2xsn438 image request)."""

    def _flip_after(self, seconds, **kw):
        async def body():
            f = _front(**kw)
            f.queue.append(_pending("img", 35, eligible=False))
            seen, t0 = await _run_controller(f, lambda s: "flip" in s, seconds)
            return seen, t0

        seen, t0 = asyncio.run(body())
        return None if "flip" not in seen else seen["flip"][2] - t0

    def test_default_holds_it_like_today(self):
        self.assertIsNone(self._flip_after(1.2), "today: only the fairness bound (45 s) releases it")

    def test_switch_on_flips_after_the_hold_instead_of_the_fairness_bound(self):
        gap = self._flip_after(3.0, d_hold_s=0.6)
        self.assertIsNotNone(gap)
        self.assertGreater(gap, 0.7)
        self.assertLess(gap, 1.4)

    def test_the_drain_does_not_take_what_d_cannot_serve(self):
        gap = self._flip_after(3.0, d_hold_s=0.3, d_short_drain_tokens=X)
        self.assertIsNotNone(gap, "the image request must still reach P")


class ControllerHoldRestartsOnWork(CustomTestCase):
    def test_a_short_served_inside_the_hold_restarts_it(self):
        async def body():
            f = _front(idle_layout="P", d_hold_s=0.8)
            D = f.groups["D"]
            seen = {}
            orig_flip = f.flip

            async def flip(src, dst):
                seen.setdefault("flip", time.time())
                return await orig_flip(src, dst)

            f.flip = flip
            task = asyncio.create_task(f.controller())
            await asyncio.sleep(0.5)            # inside the hold
            D.outstanding["short"] = time.time()  # a SHORT arrival is served on D
            await asyncio.sleep(0.5)
            t_end = time.time()
            D.outstanding.pop("short", None)    # its leg 2 ended: D is free again
            while "flip" not in seen and time.time() - t_end < 3.0:
                await asyncio.sleep(0.01)
            await _cancel(task)
            return seen, t_end

        seen, t_end = asyncio.run(body())
        self.assertIn("flip", seen)
        self.assertGreater(seen["flip"] - t_end, 0.7, "the hold must count from the END of D's work")


# ---------------------------------------------------------------------------
# wiring: handle_generate, re-queues, controller order, launcher
# ---------------------------------------------------------------------------


class Wiring(CustomTestCase):
    def test_the_batch_pending_carries_its_own_route_verdict(self):
        src = inspect.getsource(front_mod.Front.handle_generate)
        # RC2 review: minus the SHORT that D's own #915 budget just refused
        # (test_weg2_idle_drain_budget_rc2_0924.py).
        self.assertIn("d_eligible=short_ok and not short_refused)", src)
        tail = src[src.rfind("await fut  # leg 1 done and D awake"):]
        self.assertIn("if p.d_direct:", tail)
        i = tail.index("if p.d_direct:")
        self.assertIn("pending=None, seat=p.seat)", tail[i:i + 400])

    def test_a_requeued_request_is_p_s_again(self):
        for name in ("leg2", "_requeue_after_x_refusal"):
            src = inspect.getsource(getattr(front_mod.Front, name))
            self.assertRegex(src, r"\.d_eligible = \w+\.d_direct = False", name)

    def test_the_controller_order(self):
        src = inspect.getsource(front_mod.Front.controller)
        d_arm = src[src.index('if self.awake == "D":'):src.index("t_drain0 = time.time()")]
        self.assertLess(d_arm.index("self._d_short_drain("), d_arm.index("self._fairness_switch("),
                        "(b) runs before the D arm's other decisions")
        idle = d_arm[d_arm.index("if not self.queue:"):]
        self.assertIn('self._d_hold_expired("idle", _now)', idle[:idle.index("continue")])
        econ = d_arm[d_arm.index("if not self._flip_economics_ok(fairness_fired):"):]
        self.assertIn('self._d_hold_expired("backlog", _now)', econ[:400])

    def test_the_front_cli_and_the_startup_line(self):
        src = inspect.getsource(front_mod.main)
        self.assertIn('"--d-short-drain-tokens", type=int, default=0', src)
        self.assertIn('"--d-hold-s", type=float, default=None', src)
        self.assertIn("d_short_drain_tokens=args.d_short_drain_tokens", src)
        self.assertIn("d_hold_s=args.d_hold_s", src)
        line = _front(d_short_drain_tokens=X, d_hold_s=10.0, idle_layout="P").idle_policy_line()
        for want in ("idle_layout=P", f"d_short_drain_tokens={X}", "d_hold_s=10.0", f"X={X}"):
            self.assertIn(want, line)
        self.assertIn("d_hold_s=off", _front().idle_policy_line())


class LauncherFlags(CustomTestCase):
    def _ns(self, *extra):
        from sglang.srt.weg2 import launcher as L

        return L, L.build_parser().parse_args(["--tree", "/x", "--tag", "t", *extra])

    def _argv(self, *extra):
        L, ns = self._ns(*extra)
        return L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, X, X, "D")

    def test_defaults_are_off(self):
        _, ns = self._ns()
        self.assertEqual((ns.idle_layout, ns.d_short_drain_tokens, ns.d_hold_s), ("tp", 0, None))

    def test_unset_ships_the_front_argv_unchanged(self):
        argv = self._argv()
        self.assertNotIn("--d-short-drain-tokens", argv)
        self.assertNotIn("--d-hold-s", argv)

    def test_set_reaches_the_front(self):
        argv = self._argv("--idle-layout", "pp", "--d-short-drain-tokens", str(X), "--d-hold-s", "10")
        self.assertEqual(argv[argv.index("--d-short-drain-tokens") + 1], str(X))
        self.assertEqual(argv[argv.index("--d-hold-s") + 1], "10.0")

    def test_main_logs_the_policy_and_refuses_negatives(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.main)
        self.assertIn("IDLE POLICY (27B, user order 2026-09-24)", src)
        self.assertIn("both must be >= 0", src)


if __name__ == "__main__":
    unittest.main()
