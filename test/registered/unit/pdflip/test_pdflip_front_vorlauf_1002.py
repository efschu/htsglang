# SPDX-License-Identifier: Apache-2.0
"""FRONT-VORLAUF 1002: two front-side costs of the D>P Vorlauf, each behind its switch.

Boot N5q (...10021513_9126170083_1002_151409.front.log), epoch 4, Vorlauf 501 ms:
the LONG pdflip-3-5 was queued (BATCH, 15:17:13.541) while a manual P>D flip ran.

    13.541  BATCH queued; the arrival kick wakes the controller -- state=flipping, continue
    13.854  P>D done (manual flip, the controller is not in it)
    ~13.94  the controller's next 0.2 s tick ...
    14.037  ... and inside that pass 'PDFLIP-FRONT GC-PAUSE generation=2 ms=92 collected=0'
    14.038  FLIP-ECONOMICS verdict=flip, MIN-DWELL overridden_by=d_idle
    14.042  PDFLIP-FLIP begin epoch=4 sleep=D wake=P

* GC-GUARD ``FLLIPER_PDFLIP_FRONT_GC_GUARD`` (default on, pdflip/front_gc_guard.py):
  CPython never starts a generation-2 pass on its own; the due pass runs when no
  flip is open and no queued verdict waits (bounded by
  ``FLLIPER_PDFLIP_FRONT_GC_MAX_DEFER_S``, never inside a flip), the warm-up end
  freezes the heap again, a slow pass refreezes its survivors (capped).
* DONE-KICK ``FLLIPER_PDFLIP_CTL_KICK_DONE_QUEUED`` (default on): a P>D flip that
  closes with P-bound work queued wakes the controller at once; the decision
  (economics, MIN-DWELL, fairness, hand-off window) is the same, only not one
  tick later. Marker ``PDFLIP-FLIP DONE-KICK``.
* fixed beside it: MANUAL-FLIP RETURN-SKIP kicked an unknown reason
  ("manual_return_skip") -- a ValueError after the flip, the POST answered 500.

DANGER DIRECTIONS: a full pass inside a flip; a due pass deferred forever
(memory); a decision the tick would not have taken (economics/dwell must still
hold); a kick racing d_admitter for _ready_for_d; switch off = as before.
Hermetic: no GPU, no boot, no HTTP (the group RPCs are stubbed).
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as front_mod
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

MIB = 1024 * 1024
ENV_ARRIVAL = "FLLIPER_PDFLIP_CTL_KICK_ARRIVAL"
ENV_AFTER_FLIP = "FLLIPER_PDFLIP_CTL_KICK_AFTER_FLIP"
ENV_DONE = "FLLIPER_PDFLIP_CTL_KICK_DONE_QUEUED"
ENV_GUARD = "FLLIPER_PDFLIP_FRONT_GC_GUARD"
ALL_ENVS = (ENV_ARRIVAL, ENV_AFTER_FLIP, ENV_DONE, "FLLIPER_PDFLIP_DC_OFF_PATH")


class _Env:
    def __init__(self, **on):
        self.on, self.saved = on, {}

    def __enter__(self):
        for k in ALL_ENVS:
            self.saved[k] = os.environ.pop(k, None)
        os.environ.update(self.on)
        return self

    def __exit__(self, *exc):
        for k in ALL_ENVS:
            os.environ.pop(k, None)
            if self.saved.get(k) is not None:
                os.environ[k] = self.saved[k]
        return False


def _front(**env):
    with _Env(**env):
        f = front_mod.Front(
            prefill="http://p", decode="http://d", awake="D", tag="warmup",
            store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
            weight_chunks=2, flip_min_work_tokens=1,
        )
    f.stops = []
    f.inject = None  # called once per D kv resume (the return half's last RPC)

    async def rpc(g, path, body, timeout):
        if path == "/flush_cache":
            return 200, "{}"
        tags = tuple((body or {}).get("tags", ()))
        if (path == "/resume_memory_occupation" and g is f.groups["D"] and front_mod.KV_TAG in tags
                and f.inject is not None):
            inject, f.inject = f.inject, None
            inject()
        await asyncio.sleep(0.001)
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})

    async def leg1(p):
        p.leg1_prompt_tokens = 64

    f.rpc = rpc
    f.leg1 = leg1
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    f._manual_flip_refusal = lambda: None
    return f


def _pending(rid="pdflip-3-5", tokens=98798, uncached=25070):
    fut = asyncio.get_running_loop().create_future()
    return front_mod.Pending(rid, "/generate", {}, "x", time.time(), fut,
                             est_prompt=tokens, est_uncached=uncached)


async def _cancel(task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# DONE-KICK -- N5q epoch 4 replayed: a manual round trip D->P->D, the LONG
# arrives during the return half; how long from the P->D done to the D->P begin?
# ---------------------------------------------------------------------------


async def _done_to_begin(f, *, d_running=False, ready_for_d=False, wait_s=1.0):
    begins = []
    orig_flip = f.flip

    async def flip(src, dst):
        begins.append((src, dst, time.time()))
        return await orig_flip(src, dst)

    f.flip = flip

    def arrive():  # what handle_generate does at 13.541: append + arrival kick
        f.queue.append(_pending())
        f._kick_controller("arrival")
        if d_running:
            f.groups["D"].outstanding["decoding"] = time.time()
        if ready_for_d:
            f._ready_for_d.append(_pending(rid="prefilled", tokens=64, uncached=64))

    ctl = asyncio.create_task(f.controller())
    await asyncio.sleep(0.05)
    n0 = len(f.flip_log)

    async def manual():
        await f.flip("D", "P")         # the manual first half (handle_manual_flip's)
        f.inject = arrive              # the LONG arrives during the return half
        await f.flip("P", "D")

    await manual()
    assert len(f.flip_log) == n0 + 2 and f.awake == "D", (f.flip_log, f.stops)
    t_done = f.t_awake
    deadline = time.time() + wait_s
    while time.time() < deadline and len(begins) < 3:
        await asyncio.sleep(0.002)
    await _cancel(ctl)
    third = begins[2] if len(begins) >= 3 else None
    return (None if third is None else (third[0], third[1], third[2] - t_done)), f


class DoneKick(CustomTestCase):
    ON = {ENV_ARRIVAL: "1", ENV_AFTER_FLIP: "1"}

    def test_red_the_queued_long_flips_back_at_the_done_not_at_the_next_tick(self):
        async def body():
            f = _front(**self.ON)
            with self.assertLogs(front_mod.logger, level=logging.INFO) as cm:
                got, f = await _done_to_begin(f)
            return got, f, cm.output

        got, f, out = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertIsNotNone(got, "the controller never flipped D->P for the queued LONG")
        src, dst, gap = got
        self.assertEqual((src, dst), ("D", "P"))
        self.assertLess(gap, 0.06, f"done -> begin {gap * 1000:.0f} ms: the controller waited for its tick")
        self.assertTrue(any("PDFLIP-FLIP DONE-KICK epoch=" in l and "held_ready_for_d=0" in l for l in out), out)
        self.assertTrue(any("PDFLIP-FLIP DONE-KICK begin epoch=" in l and "done_to_begin_ms=" in l
                            for l in out), out)
        self.assertEqual(f.counters.get("ctl_kick_done_queued"), 1)

    def test_switch_off_keeps_the_tick(self):
        async def body():
            return await _done_to_begin(_front(**dict(self.ON, **{ENV_DONE: "0"})))

        got, f = asyncio.run(body())
        self.assertIsNotNone(got)
        self.assertGreater(got[2], 0.12, f"switch off: done -> begin {got[2] * 1000:.0f} ms, the tick is gone")
        self.assertFalse(f.counters.get("ctl_kick_done_queued"))

    def test_only_the_when_d_with_a_running_decode_still_holds(self):
        # D's work is not exhausted: the D-branch holds exactly as on a tick.
        async def body():
            return await _done_to_begin(_front(**self.ON), d_running=True, wait_s=0.5)

        got, f = asyncio.run(body())
        self.assertIsNone(got, f"flipped D->P over a running decode: {got}")
        self.assertEqual(f.counters.get("ctl_kick_done_queued"), 1, "the kick itself fired")

    def test_held_while_prefilled_requests_wait_for_d(self):
        async def body():
            f = _front(**self.ON)
            with self.assertLogs(front_mod.logger, level=logging.INFO) as cm:
                got, f = await _done_to_begin(f, ready_for_d=True, wait_s=0.4)
            return got, f, cm.output

        got, f, out = asyncio.run(body())
        self.assertFalse(f.counters.get("ctl_kick_done_queued"))
        self.assertGreaterEqual(f.counters.get("ctl_kick_held_ready_for_d", 0), 1)
        self.assertTrue(any("PDFLIP-FLIP DONE-KICK" in l and "held_ready_for_d=1" in l for l in out), out)

    def test_an_empty_queue_gives_no_kick(self):
        async def body():
            f = _front(**self.ON)
            await f.flip("D", "P")
            await f.flip("P", "D")
            return f

        f = asyncio.run(body())
        self.assertFalse(f.counters.get("ctl_kick_done_queued"))

    def test_the_switch_is_announced_and_defaults_on(self):
        f = _front()
        self.assertTrue(f._kick_done_queued)
        self.assertIn("kick_done_queued=on", f.flipfast_line())
        self.assertIn("kick_done_queued=off", _front(**{ENV_DONE: "0"}).flipfast_line())


class ManualReturnSkipKick(CustomTestCase):
    def test_red_the_return_skip_kick_is_a_known_reason(self):
        # 446d67df99 kicked "manual_return_skip": _kick_controller raised ValueError
        # after the flip (its own test stubs _kick_controller, so nothing saw it).
        async def body():
            f = _front(**{ENV_AFTER_FLIP: "1"})
            orig_flip = f.flip

            async def flip(src, dst):
                await orig_flip(src, dst)
                if (src, dst) == ("D", "P"):
                    f.queue.append(_pending(uncached=133401))

            f.flip = flip
            return await f.handle_manual_flip(None), f

        r, f = asyncio.run(body())
        self.assertEqual(r.status, 200)
        self.assertEqual(f.awake, "P")
        self.assertEqual(f.counters.get("manual_flip_return_skip"), 1)


# ---------------------------------------------------------------------------
# GC-GUARD
# ---------------------------------------------------------------------------


def _churn(n=400_000):
    """Long-lived container objects: enough generation-1 passes for CPython to
    start a generation-2 pass on its own (count[2] > threshold2, and the
    pending long-lived objects well past a quarter of generation 2)."""
    keep = []
    for i in range(n):
        keep.append([i])
    return keep


class _Gen2Watch:
    def __init__(self):
        self.starts = 0

    def __call__(self, phase, info):
        if phase == "start" and info.get("generation") == 2:
            self.starts += 1

    def __enter__(self):
        gc.callbacks.append(self)
        return self

    def __exit__(self, *exc):
        gc.callbacks.remove(self)
        return False


class _GuardCase(CustomTestCase):
    def setUp(self):
        from flliper.srt.pdflip import front_gc_guard as gcg

        self.gcg = gcg
        self.saved = gc.get_threshold()
        gc.collect()

    def tearDown(self):
        gc.set_threshold(*self.saved)


class GcGuardSchedule(_GuardCase):
    def test_control_cpython_starts_full_passes_under_this_churn(self):
        # the premise: without the guard the same churn DOES trigger generation 2
        with _Gen2Watch() as w:
            keep = _churn()
        del keep
        self.assertGreater(w.starts, 0)

    def test_red_no_automatic_full_pass_while_a_flip_is_open(self):
        f = front_mod.Front.__new__(front_mod.Front)
        f.state, f.queue = "flipping", []
        g = self.gcg.GcGuard(min_interval_s=0.0)
        self.assertTrue(g.arm())
        try:
            with _Gen2Watch() as w:
                keep = _churn()
                critical, flipping = front_mod.Front._gc_critical(f)
                self.assertEqual((critical, flipping), (True, True))
                self.assertTrue(g.due(), "the churn did not make a pass due")
                self.assertIsNone(g.step(critical, flipping))
            self.assertEqual(w.starts, 0, "a generation-2 pass ran inside the flip")
            # the flip closed, nothing queued: the due pass runs now, by the guard
            f.state = "serving"
            with _Gen2Watch() as w2, self.assertLogs(front_mod.logger, level=logging.INFO) as cm:
                self.assertEqual(g.step(*front_mod.Front._gc_critical(f)), "idle")
            self.assertEqual(w2.starts, 1)
            self.assertFalse(g.due())
            self.assertTrue(any("PDFLIP-FRONT GC-GUARD full reason=idle" in l for l in cm.output), cm.output)
            del keep
        finally:
            g.disarm()

    def test_a_queued_verdict_defers_until_the_bound_but_a_flip_always_defers(self):
        clock = [100.0]
        g = self.gcg.GcGuard(max_defer_s=30.0, min_interval_s=0.0, clock=lambda: clock[0])
        g.arm()
        try:
            keep = _churn()
            self.assertTrue(g.due())
            self.assertIsNone(g.step(critical=True, flipping=False))   # verdict waits: deferred
            clock[0] += 29.0
            self.assertIsNone(g.step(critical=True, flipping=False))
            clock[0] += 2.0
            self.assertIsNone(g.step(critical=True, flipping=True))    # past the bound, but a flip
            with _Gen2Watch() as w:
                self.assertEqual(g.step(critical=True, flipping=False), "max-defer")
            self.assertEqual(w.starts, 1)
            self.assertEqual(g.counters["full_max_defer"], 1)
            del keep
        finally:
            g.disarm()

    def test_a_slow_pass_refreezes_its_survivors_capped(self):
        g = self.gcg.GcGuard(min_interval_s=0.0, refreeze_ms=0.0, refreeze_max=1)
        g.arm()
        try:
            with mock.patch.object(self.gcg.gc, "freeze") as freeze, \
                    self.assertLogs(front_mod.logger, level=logging.INFO) as cm:
                keep = _churn()
                self.assertEqual(g.step(False, False), "idle")
                keep2 = _churn()
                self.assertEqual(g.step(False, False), "idle")
            self.assertEqual(freeze.call_count, 1, "the refreeze cap did not hold")
            self.assertTrue(any("PDFLIP-FRONT GC-GUARD refreeze n=1/1" in l for l in cm.output), cm.output)
            del keep, keep2
        finally:
            g.disarm()

    def test_disarm_restores_cpythons_thresholds(self):
        g = self.gcg.GcGuard()
        g.arm()
        self.assertEqual(gc.get_threshold()[2], self.gcg.GEN2_OFF)
        self.assertEqual(gc.get_threshold()[:2], self.saved[:2])
        g.disarm()
        self.assertEqual(gc.get_threshold(), self.saved)

    def test_a_paused_collector_gets_no_pass_of_ours(self):
        g = self.gcg.GcGuard(min_interval_s=0.0)
        g.arm()
        try:
            keep = _churn()
            gc.disable()
            try:
                self.assertIsNone(g.step(False, False))
            finally:
                gc.enable()
            del keep
        finally:
            g.disarm()

    def test_the_probe_names_the_trigger(self):
        probe = front_mod.install_gc_pause_probe(threshold_ms=0.0)
        g = self.gcg.GcGuard(min_interval_s=0.0)
        try:
            with self.assertLogs(front_mod.logger, level="WARNING") as cm:
                gc.collect()
                g._collect()
            lines = [m for m in cm.output if "PDFLIP-FRONT GC-PAUSE" in m]
            self.assertTrue(any("trigger=auto" in l for l in lines), lines)
            self.assertTrue(any("trigger=guard" in l for l in lines), lines)
        finally:
            gc.callbacks.remove(probe)


class GcGuardFrontWiring(_GuardCase):
    def test_front_critical_is_a_flip_or_a_queued_verdict(self):
        f = front_mod.Front.__new__(front_mod.Front)
        f.state, f.queue = "serving", []
        self.assertEqual(front_mod.Front._gc_critical(f), (False, False))
        f.queue = [object()]
        self.assertEqual(front_mod.Front._gc_critical(f), (True, False))
        f.state = "flipping"
        self.assertEqual(front_mod.Front._gc_critical(f), (True, True))

    def test_the_warm_freeze_waits_for_every_warmup_task(self):
        async def body():
            loop = asyncio.get_running_loop()
            f = front_mod.Front.__new__(front_mod.Front)
            g = self.gcg.GcGuard()
            f.__dict__["_gc_guard_obj"] = g
            futs = {k: loop.create_future() for k in front_mod.Front.GC_WARM_TASKS}
            with mock.patch.object(self.gcg.gc, "freeze") as freeze, \
                    self.assertLogs(front_mod.logger, level=logging.INFO) as cm:
                t = asyncio.ensure_future(f._gc_warm_freeze(dict(futs)))
                for k in list(futs)[:-1]:
                    futs[k].set_result(None)
                await asyncio.sleep(0.02)
                before = freeze.call_count
                futs["x_exact"].set_result(None)
                await asyncio.wait_for(t, 1.0)
            return before, freeze.call_count, cm.output

        before, after, out = asyncio.run(body())
        self.assertEqual((before, after), (0, 1))
        self.assertTrue(any("PDFLIP-FRONT GC-GUARD warm-freeze after=launcher_prewarm+flip_imports_prewarm"
                            "+sidecar_prewarm+x_exact" in l for l in out), out)

    def test_startup_arms_it_behind_its_switch_and_cleanup_disarms(self):
        import inspect

        from flliper.srt.environ import envs

        src = inspect.getsource(front_mod.Front.startup)
        self.assertIn("envs.FLLIPER_PDFLIP_FRONT_GC_GUARD.get()", src)
        self.assertIn('app["gc_guard"] = asyncio.create_task(self.gc_guard_sampler())', src)
        self.assertIn('app["gc_warm_freeze"] = asyncio.create_task(self._gc_warm_freeze(app))', src)
        self.assertIn("_g.disarm()", inspect.getsource(front_mod.Front.cleanup))
        self.assertTrue(envs.FLLIPER_PDFLIP_FRONT_GC_GUARD.get(), "the guard must default on")
        with envs.FLLIPER_PDFLIP_FRONT_GC_GUARD.override(False):
            self.assertFalse(envs.FLLIPER_PDFLIP_FRONT_GC_GUARD.get())

    def test_the_sampler_drives_the_guard_with_the_fronts_reading(self):
        async def body():
            f = front_mod.Front.__new__(front_mod.Front)
            f.state, f.queue = "flipping", []
            seen = []

            class G:
                def step(self, critical, flipping):
                    seen.append((critical, flipping))

            f.__dict__["_gc_guard_obj"] = G()
            with mock.patch.object(front_mod._gcg, "POLL_S", 0.005):
                t = asyncio.ensure_future(f.gc_guard_sampler())
                await asyncio.sleep(0.03)
                f.state = "serving"
                await asyncio.sleep(0.03)
                await _cancel(t)
            return seen

        seen = asyncio.run(body())
        self.assertIn((True, True), seen)
        self.assertEqual(seen[-1], (False, False))


if __name__ == "__main__":
    unittest.main()
