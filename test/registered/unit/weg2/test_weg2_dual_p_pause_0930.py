# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 unified KV (A): the front pauses P's leg 1 when D is short of
KV on a card, and resumes it later (user order 30.09. 07:25Z: "wird vram kv
knapp, pausiert P und gibt den context frei und das erarbeitete in den L2 zur
späteren weiterverwendung wenn vram kv wieder frei wird").

DANGER DIRECTIONS guarded here:
* pressure from any card ledger -> every in-flight leg 1 is marked paused and
  aborted on P ONCE; no new P pass starts while the pressure holds;
* the paused request is requeued at the HEAD, still in leg 1 -- its future is
  NOT failed (no W50, no 503);
* no ledgers configured -> the pump is byte-identical (no ledger read);
* a pause that arrives after P finished the leg changes nothing.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MIB = 1 << 20


def _front():
    f = front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="dual",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1, dual_layout=True,
    )
    f.rpcs = []

    async def rpc(g, path, body, timeout):
        f.rpcs.append((g.name, path, body.get("rid")))
        return 200, "{}"

    f.rpc = rpc
    return f


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _pending(rid):
    fut = asyncio.get_running_loop().create_future()
    return front_mod.Pending(rid, "/generate", {}, "x", time.time(), fut,
                             est_prompt=40000, est_uncached=40000)


class DualPPause(CustomTestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(prefix="wkvf"), "card")
        self.d = K.CardKvLedger(self.path, "D")
        self.p = K.CardKvLedger(self.path, "P")
        self.d.join(1000 * MIB)
        self.p.join(1000 * MIB)

    def test_pressure_pauses_inflight_and_holds_new_passes(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = [self.path]
            p = _pending("r1")
            f._dual_inflight["r1"] = p
            self.p.request(800 * MIB)
            self.d.request(500 * MIB)                 # D short -> pressure on P
            started = []

            async def pass_fn():
                started.append(1)
            f.queue.append(_pending("r2"))
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.05)
            f._dual_pump(pass_fn)                     # a second tick: no double abort
            await asyncio.sleep(0.05)
            return f, p, started

        f, p, started = _run(run())
        self.assertTrue(p.dual_pause)
        self.assertEqual(f.rpcs, [("P", "/abort_request", "r1")])
        self.assertEqual(started, [], "a P pass started while D was short")
        self.assertEqual(f.counters["dual_p_pauses"], 1)

    def test_paused_leg_requeues_at_head_without_failing(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = [self.path]
            p = _pending("r1")
            p.dual_pause = True
            f.queue.append(_pending("r0"))
            f._dual_requeue_paused(p)
            return f, p

        f, p = _run(run())
        self.assertIs(f.queue[0], p)
        self.assertFalse(p.fut.done())
        self.assertFalse(p.leg1_done)
        self.assertEqual(p.dual_paused_n, 1)
        self.assertFalse(p.dual_pause)

    def test_no_ledgers_no_reads(self):
        called = []
        orig = front_mod._dual_p_pressure
        front_mod._dual_p_pressure = lambda paths: called.append(paths) or 0

        async def run():
            f = _front()
            f.queue.append(_pending("r2"))
            f._dual_pump(lambda: asyncio.sleep(0))
            await asyncio.sleep(0.01)
            return f
        try:
            _run(run())
        finally:
            front_mod._dual_p_pressure = orig
        self.assertEqual(called, [])

    def test_resume_when_pressure_gone(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = [self.path]
            self.p.request(800 * MIB)
            self.d.request(500 * MIB)
            started = []

            async def pass_fn():
                started.append(1)
            f.queue.append(_pending("r1"))
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.02)
            before = list(started)
            self.p.release(800 * MIB)                  # P freed its context: pressure answered
            # D's next tick takes the 300 MiB it was short of (ledger demand[D] -> 0). Since the
            # dual pressure fix (02.10.) the stages read D's shortfall too: P does not resume
            # into rows D still waits for
            self.assertEqual(self.d.request(300 * MIB), (300 * MIB, 0))
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.02)
            return before, started

        before, after = _run(run())
        self.assertEqual(before, [])
        self.assertEqual(after, [1])

    def test_wiring_in_the_drain(self):
        import inspect

        src = inspect.getsource(front_mod.Front.controller)
        self.assertIn("self._dual_requeue_paused(p)", src)
        self.assertIn("self._dual_inflight.pop(p.rid, None)", src)

    def test_launcher_hands_the_ledgers_to_the_front(self):
        import types

        from sglang.srt.weg2 import launcher as L

        cards = [types.SimpleNamespace(uuid="GPU-a", nvml_index=0), types.SimpleNamespace(uuid="GPU-b", nvml_index=1)]
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-share"])
        L.resolve_dual_layout(ns)
        paths = L.dual_kv_ledger_paths(ns, cards)
        self.assertEqual(paths, [K.ledger_path("t", "GPU-a"), K.ledger_path("t", "GPU-b")])
        off = L.build_parser().parse_args(["--tree", "/x", "--tag", "t"])
        self.assertEqual(L.dual_kv_ledger_paths(off, cards), [])

    def test_ranks_and_front_share_the_ledger_namespace(self):
        from sglang.srt.weg2 import launcher as L

        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "boot7", "--dual-share"])
        L.resolve_dual_layout(ns)
        for g in ("P", "D"):
            self.assertEqual(L.dual_share_env(ns, g)["SGLANG_WEG2_DUAL_KV_TAG"], "boot7")

