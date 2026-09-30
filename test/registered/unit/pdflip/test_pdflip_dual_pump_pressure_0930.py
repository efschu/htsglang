# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 metal replay dual17 (wjt89v, ...09301301 @9024c1f473).

From 13:08:30 D TP0 (5090) logged GROUP-WAIT want=344064 need=603979776
granted=310378496 pressure_on_P=293601280, up to 360710144 at 13:09:01. TP1
and TP2 showed pressure_on_P=0 (granted=need). P held B, whose prefill had
started at 13:05:20. The front never paused: its log carries no DUAL line
after the start.

_dual_pump read the ledgers only AFTER "a pass is still running -> return",
and a leg 1 in flight IS a running pass. So the reader never ran while there
was anything to pause.

DANGER DIRECTIONS guarded here:
* pressure on ONE card (TP0 short, the other two not) pauses the leg in
  flight while its pass is still running;
* the pressure reader leaves an instrument line on every change and at most
  every DUAL_PRESSURE_LOG_S while pressure holds; an absent ledger reads -1.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import front as front_mod
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

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


def _cards():
    root = tempfile.mkdtemp(prefix="wkvpump")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for i, pth in enumerate(paths):
        d, p = K.CardKvLedger(pth, "D"), K.CardKvLedger(pth, "P")
        d.contribute(1000 * MIB, committed=1000 * MIB)
        p.contribute(1000 * MIB)                     # P sized while D held its pool: budget 2000
        d.release(900 * MIB)                         # D keeps 100
        p.request(900 * MIB)                         # P holds B's grant on every card
        if i:
            d.request(200 * MIB)                     # TP1/TP2: granted = need, no pressure
    return paths


def _press_tp0(paths):
    K.CardKvLedger(paths[0], "D").request(1500 * MIB)   # TP0 (5090): D short -> pressure on P


class PumpReadsPressureWhileAPassRuns(CustomTestCase):
    def test_dual17_tp0_short_pauses_the_leg_in_flight(self):
        paths = _cards()
        self.assertEqual(K.peek(paths[0]).pressure["P"], 0)   # the pass starts unpressed (13:05:20)

        async def run():
            f = _front()
            f.dual_kv_ledgers = paths
            gate = asyncio.Event()

            async def pass_fn():                     # the P pass that carries B's leg 1
                await gate.wait()

            fut = asyncio.get_running_loop().create_future()
            b = front_mod.Pending("pdflip-0-4", "/generate", {}, "x", time.time(), fut,
                                  est_prompt=129188, est_uncached=129188)
            f.queue.append(b)
            f._dual_pump(pass_fn)                    # starts the pass (nothing in flight yet)
            f._dual_inflight["pdflip-0-4"] = b         # the pass took B
            await asyncio.sleep(0.02)
            _press_tp0(paths)                        # 13:08:30: D TP0 short while B prefills
            assert K.peek(paths[0]).pressure["P"] > 0 and K.peek(paths[1]).pressure["P"] == 0
            for _ in range(3):                       # controller ticks while the pass runs
                await asyncio.sleep(0.02)
                f._dual_pump(pass_fn)
            await asyncio.sleep(0.05)
            gate.set()
            return f, b

        f, b = _run(run())
        self.assertTrue(b.dual_pause, "the front never paused B while its pass ran (dual17)")
        self.assertEqual(f.rpcs.count(("P", "/abort_request", "pdflip-0-4")), 1)

    def test_pressure_reader_instrument_line(self):
        paths = _cards()
        _press_tp0(paths)
        paths = paths + [os.path.join(tempfile.mkdtemp(prefix="wkvabs"), "absent")]
        f = _front()
        f.dual_kv_ledgers = paths
        lines = []
        clock = [1000.0]
        with mock.patch.object(front_mod.logger, "info", lambda m, *a: lines.append(m % a)), \
                mock.patch.object(front_mod.time, "time", lambda: clock[0]):
            for i in range(100):                     # 100 ticks over 50 s
                clock[0] = 1000.0 + i * 0.5
                self.assertGreater(f._dual_pressure_tick(), 0)
        got = [l for l in lines if "PDFLIP DUAL PRESSURE-READ" in l]
        self.assertTrue(got)
        self.assertLessEqual(len(got), 2)                # at the change + once after 30 s
        self.assertIn("per_card=[", got[0])
        self.assertIn("-1]", got[0])                     # the absent ledger is named


if __name__ == "__main__":
    import unittest

    unittest.main()
