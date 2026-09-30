# SPDX-License-Identifier: Apache-2.0
"""Metal replay dual22 (f5jsd5, ...09301524 @29a9e2aa63).

The pause worked: P-PAUSE, P-PAUSED requeue, RELEASE on all three stages,
pressure 0. Then:
- 15:32:13,278 requeue; at 15:32:13,434 the front sent B (same rid
  weg2-0-8) back to P, 0.16 s later.
- 15:32:13 PP2 applied its recorded abort ("PP0 idle since 1.0 s") while
  PP0's frame fwd 340 for the OLD B (tokens 2048-3072) still sat in its
  inbox: "ROW-PROBE DEFER ... names weg2-0-8 not locatable here".
- Result: PpRowDeferCapExceeded #1180 on PP2 -> DEBUG-HOLD -> W17 -> B 503.
  pauses=2: the resume went straight into D still growing.

DANGER DIRECTIONS guarded here:
* a follower applies its recorded abort only once it has executed every
  pass PP0 launched (PP0's idle stamp carries its forward count). PP0 idle
  alone is not ordered with the ring;
* a PAUSED request goes back to P only when every card shows pressure 0,
  P committed 0 (every stage released) and D demand 0 (D stopped growing);
* D's cleared pressure clears its demand too.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import types
import unittest.mock as mock
import uuid

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.srt.weg2 import front as F
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20


class FollowerAbortFollowsTheRing(CustomTestCase):
    def test_dual22_a_frame_still_in_the_inbox_holds_the_abort(self):
        tag = "t-%s" % uuid.uuid4().hex[:8]
        env = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_KV_TAG": tag}
        applied = []

        def process():
            applied.append(True)
            f._pending_chunked_abort_req = None

        b = types.SimpleNamespace(rid="weg2-0-8")
        f = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=2), _pending_chunked_abort_req=b,
                                  process_pending_chunked_abort=process, forward_ct=338,
                                  _pp_microbatches_drained=lambda: True)
        pp0 = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0), forward_ct=341)  # launched 339, 340
        try:
            with mock.patch.dict(os.environ, env):
                self.assertFalse(S.follower_release_aborted_chunk(f, now=100.0))   # PP2 sees the abort
                S.mark_pp0_idle(pp0, now=101.0)                                    # PP0 idle, 1.0 s later
                self.assertFalse(S.follower_release_aborted_chunk(f, now=101.5),
                                 "applied while PP0's fwd 339/340 had not run here (dual22 #1180)")
                f.forward_ct = 341                                                 # both frames executed
                self.assertTrue(S.follower_release_aborted_chunk(f, now=102.0))
            self.assertEqual(applied, [True])
        finally:
            try:
                os.unlink(S._idle_marker(tag))
            except OSError:
                pass


def _front():
    f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="dual", store_dir="/tmp",
                prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=True)

    async def rpc(g, path, body, timeout):
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


class ResumeOnlyWhenEveryStageReleased(CustomTestCase):
    def _cards(self):
        root = tempfile.mkdtemp(prefix="wkv22")
        paths = [os.path.join(root, "card%d" % i) for i in range(3)]
        for pth in paths:
            d, p = K.CardKvLedger(pth, "D"), K.CardKvLedger(pth, "P")
            d.contribute(2000 * MIB, committed=2000 * MIB)
            p.contribute(1000 * MIB)
            d.release(1000 * MIB)
        return paths

    def test_dual22_the_paused_head_waits_for_the_followers_and_for_d(self):
        paths = self._cards()
        K.CardKvLedger(paths[2], "P").request(512 * MIB)          # PP2 still holds the old B
        K.CardKvLedger(paths[0], "D").request(5000 * MIB)         # D still growing on the 5090
        K.CardKvLedger(paths[0], "D").release(K.peek(paths[0]).committed["D"] - 1000 * MIB)

        async def run():
            f = _front()
            f.dual_kv_ledgers = paths
            started = []

            async def pass_fn():
                started.append(1)

            fut = asyncio.get_running_loop().create_future()
            b = F.Pending("weg2-0-8", "/generate", {}, "x", time.time(), fut, est_prompt=129185,
                          est_uncached=129185)
            b.dual_paused_n = 1                                    # requeued at the head (P-PAUSED)
            f.queue.append(b)
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.02)
            n_held = len(started)
            K.CardKvLedger(paths[2], "P").release(512 * MIB)       # FOLLOWER-ABORT-APPLIED, RELEASE
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.02)
            n_d = len(started)
            K.CardKvLedger(paths[0], "D").clear_pressure()         # D's demand fits: it stops asking
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.02)
            return n_held, n_d, len(started)

        n_held, n_d, n_go = _run(run())
        self.assertEqual(n_held, 0, "the paused B went back to P while PP2 still held it (dual22)")
        self.assertEqual(n_d, 0, "the paused B went back while D still grew (dual22 pause 2)")
        self.assertEqual(n_go, 1)

    def test_an_ordinary_head_is_not_held(self):
        paths = self._cards()
        K.CardKvLedger(paths[2], "P").request(512 * MIB)

        async def run():
            f = _front()
            f.dual_kv_ledgers = paths
            started = []

            async def pass_fn():
                started.append(1)

            fut = asyncio.get_running_loop().create_future()
            f.queue.append(F.Pending("weg2-0-9", "/generate", {}, "x", time.time(), fut,
                                     est_prompt=6221, est_uncached=6221))
            f._dual_pump(pass_fn)
            await asyncio.sleep(0.02)
            return len(started)

        self.assertEqual(_run(run()), 1)

    def test_cleared_pressure_clears_d_demand(self):
        paths = self._cards()
        d = K.CardKvLedger(paths[1], "D")
        d.request(5000 * MIB)
        self.assertGreater(K.peek(paths[1]).demand["D"], 0)
        d.clear_pressure()
        self.assertEqual(K.peek(paths[1]).demand["D"], 0)
        self.assertTrue(K.p_resume_ready([paths[1]])[0])          # no pressure, P 0, D demand 0
