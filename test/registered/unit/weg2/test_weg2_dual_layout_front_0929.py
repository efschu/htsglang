# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 (F26): the front's dual-layout mode.

Both groups stay awake and the front never flips: a queued request's leg 1
goes to P while ``awake`` stays "D", the prefilled request lands in
``_ready_for_d`` for the admitter, and no flip is ever taken.

DANGER DIRECTIONS guarded here:
* dual on must never call ``flip`` (a flip would sleep a group that the dual
  budgets assume awake -- the co-residence plan has no room for a wake);
* dual off must keep today's order: leg 1 only after a D->P flip;
* the P pass is the SAME code as the phase router's (moved, not copied): the
  pump calls the one nested ``_p_drain_pass`` and nothing else;
* a failed pass does not kill the pump (it backs off and runs the next pass).

Hermetic: no GPU, no boot, no HTTP (the group RPCs are stubbed).
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

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1024 * 1024


def _front(dual: bool):
    f = front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="dual",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1, dual_layout=dual,
    )
    f.stops = []

    async def rpc(g, path, body, timeout):
        return 200, json.dumps({"per_tag": {}, "critical_path": ""})

    f.rpc = rpc
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    return f


def _pending(rid="dual-1", tokens=65536):
    fut = asyncio.get_running_loop().create_future()
    return front_mod.Pending(rid, "/generate", {}, "x", time.time(), fut,
                             est_prompt=tokens, est_uncached=tokens)


def _run(coro):
    """asyncio.run leaves NO current loop behind, and later tests in the same
    worker that call asyncio.get_event_loop() then fail (seen with
    test_weg2_parked_abort_w1b_1401 under xdist). Leave a fresh loop set."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


async def _cancel(task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


async def _drive(f, n=2, fail_first=False, wait_s=3.0):
    seen = {"leg1": [], "flips": [], "awake_at_leg1": []}

    async def leg1(p):
        if fail_first and not seen["leg1"]:
            seen["leg1"].append(("fail", p.rid))
            raise RuntimeError("stub leg-1 failure")
        seen["leg1"].append(("ok", p.rid))
        seen["awake_at_leg1"].append(f.awake)
        p.leg1_prompt_tokens = 64

    async def flip(src, dst):
        seen["flips"].append((src, dst))
        f.awake = dst
        f.t_awake = time.time()

    f.leg1 = leg1
    f.flip = flip
    for i in range(n):
        f.queue.append(_pending(rid=f"dual-{i}"))
    task = asyncio.create_task(f.controller())
    deadline = time.time() + wait_s
    while time.time() < deadline:
        ok = [r for s, r in seen["leg1"] if s == "ok"]
        if len(ok) >= n and not f.queue:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)  # a few more ticks: any flip would show now
    await _cancel(task)
    if f._dual_task is not None:
        await _cancel(f._dual_task)
    return seen


class DualLayoutFront(CustomTestCase):
    def test_dual_prefills_on_p_without_ever_flipping(self):
        async def run():
            f = _front(True)
            seen = await _drive(f, n=3)
            return f, seen

        f, seen = _run(run())
        self.assertEqual(seen["flips"], [], "dual layout took a flip")
        self.assertEqual([r for s, r in seen["leg1"]], ["dual-0", "dual-1", "dual-2"])
        self.assertTrue(all(a == "D" for a in seen["awake_at_leg1"]), seen)
        self.assertEqual([p.rid for p in f._ready_for_d], ["dual-0", "dual-1", "dual-2"])
        self.assertGreaterEqual(f.counters["dual_passes"], 1)

    def test_default_still_flips_before_leg1(self):
        async def run():
            f = _front(False)
            return await _drive(f, n=1)

        seen = _run(run())
        self.assertTrue(seen["flips"], "the default front stopped flipping")
        self.assertEqual(seen["flips"][0], ("D", "P"))
        self.assertTrue(all(a == "P" for a in seen["awake_at_leg1"]), seen)

    def test_failed_leg_does_not_kill_the_pump(self):
        async def run():
            f = _front(True)
            seen = await _drive(f, n=2, fail_first=True, wait_s=4.0)
            return f, seen

        f, seen = _run(run())
        self.assertEqual(seen["flips"], [])
        self.assertIn(("ok", "dual-1"), seen["leg1"])

    def test_pump_runs_the_one_moved_pass(self):
        src = inspect.getsource(front_mod.Front.controller)
        self.assertEqual(src.count("async def _p_drain_pass"), 1)
        self.assertIn("self._dual_pump(_p_drain_pass)", src)
        self.assertIn("await _p_drain_pass()", src)
        # the drain pool is called exactly once in the controller: the pass.
        self.assertEqual(src.count("await _p_drain_pool("), 1)


if __name__ == "__main__":
    unittest.main()
