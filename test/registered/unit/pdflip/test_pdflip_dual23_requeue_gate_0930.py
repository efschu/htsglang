# SPDX-License-Identifier: Apache-2.0
"""Metal replay dual23 (hceth5, ...09301558 @e61d82fdd2).

- 16:07:02,473 P-PAUSE pdflip-0-8 (B had not started: prompt_tokens=0).
- 16:07:04,552 P-PAUSED requeue (pauses=1). 16:07:04,620 D-ADMIT pdflip-0-8:
  the pool's on_done handed the requeued B to D anyway.
- 16:07:04,686 a SECOND P-PAUSE at unchanged pressure 360710144: the running
  pass took the requeued head again. The pump's gates only run before a NEW
  pass, so there were 0 RESUME-WAIT lines.
- 16:07:09 D refused B (uncached 129186 > X) -> W50 re-route through P.

DANGER DIRECTIONS guarded here:
* a requeued pause never reaches D from the pass it was paused in;
* no leg 1 leaves the pool while D presses P, and a paused head only
  resumes by _dual_resume_held -- also INSIDE a running pass;
* so two pauses at unchanged pressure are impossible.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import front as F
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20


def _front():
    f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="dual", store_dir="/tmp",
                prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=True)
    f.aborts = []

    async def rpc(g, path, body, timeout):
        f.aborts.append(body.get("rid"))
        return 200, "{}"

    f.rpc = rpc
    return f


def _cards():
    root = tempfile.mkdtemp(prefix="wkv23")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for pth in paths:
        d, p = K.CardKvLedger(pth, "D"), K.CardKvLedger(pth, "P")
        d.contribute(2000 * MIB, committed=2000 * MIB)
        p.contribute(1000 * MIB)
        d.release(1000 * MIB)
    return paths


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


class Dual23(CustomTestCase):
    def test_requeued_b_is_not_redispatched_nor_handed_to_d_at_unchanged_pressure(self):
        paths = _cards()
        K.CardKvLedger(paths[0], "P").request(700 * MIB)          # P holds B's grant on the 5090
        pressed = K.CardKvLedger(paths[0], "D")

        async def run():
            f = _front()
            f.dual_kv_ledgers = paths
            fut = asyncio.get_running_loop().create_future()
            b = F.Pending("pdflip-0-8", "/generate", {}, "x", time.time(), fut, est_prompt=129186,
                          est_uncached=129186)
            f.queue = collections.deque([b])
            to_d, legs = [], []

            async def one(p):                                     # the pass's leg 1 on P
                legs.append(p.rid)
                f._dual_inflight[p.rid] = p                        # as the real one() does
                if len(legs) == 1:
                    pressed.request(5000 * MIB)                   # 16:07:02 D short -> the pump pauses it
                    f._dual_pump(lambda: asyncio.sleep(0))
                    await asyncio.sleep(0.01)
                    p.leg1_aborted = True                          # P answers 200, prompt_tokens=0
                f._dual_inflight.pop(p.rid, None)
                if getattr(p, "dual_pause", False) and getattr(p, "leg1_aborted", False):
                    p.leg1_aborted = False
                    f._dual_requeue_paused(p)
                    return p
                p.leg1_done = True
                return p

            def on_done(p):                                        # the pass's _on_leg1_done
                if f._dual_skip_after_requeue(p):
                    return
                to_d.append(p.rid)

            await asyncio.wait_for(F._p_drain_pool(
                f.queue, 1, one, on_done,
                lambda: not (f.dual_kv_ledgers and f._dual_dispatch_held())), timeout=5)
            return f, b, to_d, legs

        f, b, to_d, legs = _run(run())
        self.assertEqual(to_d, [], "the requeued, never-started B was handed to D (dual23 W50)")
        self.assertEqual(legs, ["pdflip-0-8"], "the running pass took B again at unchanged pressure (pause 2)")
        self.assertEqual(b.dual_paused_n, 1)
        self.assertIs(f.queue[0], b)

    def test_the_real_drain_uses_both_gates(self):
        src = inspect.getsource(F.Front.controller)
        self.assertIn("if self._dual_skip_after_requeue(p):", src)
        self.assertIn("self._dual_dispatch_held()", src)


class FlipFormIsUntouched(CustomTestCase):
    """z30y7 (27b-row-authority, flip form): a LONG request after the flip kick
    must dispatch -- the dual gates act only under --dual-layout."""

    def test_flip_front_dispatches_and_hands_to_d(self):
        async def run():
            f = F.Front(prefill="http://p", decode="http://d", awake="P", tag="flip", store_dir="/tmp",
                        prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                        flip_min_work_tokens=1, dual_layout=False)
            f.dual_kv_ledgers = []
            fut = asyncio.get_running_loop().create_future()
            p = F.Pending("pdflip-31-31", "/generate", {}, "x", time.time(), fut, est_prompt=6275,
                          est_uncached=6275)
            p.dual_requeued = True          # even a stray mark must not skip it in the flip form
            f.queue = collections.deque([p])
            legs, to_d = [], []

            async def one(q):
                legs.append(q.rid)
                q.leg1_done = True
                return q

            def on_done(q):
                if f._dual_skip_after_requeue(q):
                    return
                to_d.append(q.rid)

            await asyncio.wait_for(F._p_drain_pool(
                f.queue, 1, one, on_done,
                lambda: not (f.dual_layout and f.dual_kv_ledgers and f._dual_dispatch_held())), timeout=5)
            return legs, to_d

        legs, to_d = _run(run())
        self.assertEqual(legs, ["pdflip-31-31"])
        self.assertEqual(to_d, ["pdflip-31-31"])
