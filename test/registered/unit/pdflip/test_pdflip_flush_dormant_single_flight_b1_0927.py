# SPDX-License-Identifier: Apache-2.0
"""27B rc12o27 b1 (dkr27breleasedraftbar1w109271302, 7cac9f5372), group P dead 13:11:55Z.

front.log 1326 13:11:50,170 PDFLIP-FLIP begin epoch=5 (POST /pdflip/flip, the probe, 1367) while the
controller's P-drain ran; 1331/1334 13:11:53,435 P-DRAIN done -> a SECOND PDFLIP-FLIP begin epoch=5.
Two quiesce loops polled /flush_cache (pairs 1335/1336, 1339/1340); one got 200 and released P
(1342/1346); the other's flush (1348, 13:11:53,581) queued behind the release. P log 40188-40204:
PP0 refused it (vote pending), PP1/PP2 ran it on their own verdict: RESET JOIN, HOST-POOL CLEAR,
Reset HybridReqToTokenPool -> ReqToTokenPool.clear() req_to_token.zero_() on the unmapped kv_cache
region -> illegal memory access at MambaPool.reset_state's first _sync_device (40258/40312).

FD: a /flush_cache on a dormant (released) group is refused by name before any pool is touched.
FS: one flip at a time; a flip while another is open (or of a group not awake) is skipped by name.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import flush_wrapper as FW  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.managers.io_struct import FlushCacheReqInput  # noqa: E402
from flliper.srt.pdflip import front as front_mod  # noqa: E402


class _Chan:
    def __init__(self):
        self.sent = []
        self.send_to_tokenizer = self

    def send_output(self, out, req):
        self.sent.append((out, req))


def _wrapper(dormant):
    calls = []

    def flush(**kw):
        calls.append(kw)
        return True

    chan = _Chan()
    w = FW.SchedulerFlushWrapper(flush_cache=flush, is_fully_idle=lambda: True,
                                 ipc_channels=chan, is_dormant=lambda: dormant["v"])
    return w, calls, chan


class TheFlushOnAReleasedGroupIsRefused(unittest.TestCase):
    def test_b1_flush_after_release_touches_nothing(self):
        d = {"v": True}
        w, calls, _ = _wrapper(d)
        with mock.patch.dict(os.environ, {FW.ENV_DORMANT_REFUSE: "1"}):
            out = w.handle(FlushCacheReqInput())
        self.assertFalse(out.success)
        self.assertIn("W25 PdFlipDormantRefused", out.message)
        self.assertEqual(calls, [], "flush_cache must not run on released pools")

    def test_awake_group_flushes_as_before(self):
        w, calls, _ = _wrapper({"v": False})
        with mock.patch.dict(os.environ, {FW.ENV_DORMANT_REFUSE: "1"}):
            out = w.handle(FlushCacheReqInput())
        self.assertTrue(out.success)
        self.assertEqual(calls, [{"tp_group_verdict": True}])

    def test_switch_off_is_the_metal_behaviour(self):
        w, calls, _ = _wrapper({"v": True})
        with mock.patch.dict(os.environ, {FW.ENV_DORMANT_REFUSE: "0"}):
            self.assertTrue(w.handle(FlushCacheReqInput()).success)
        self.assertEqual(len(calls), 1)

    def test_a_deferred_flush_that_meets_a_release_is_refused(self):
        d = {"v": False}
        w, calls, chan = _wrapper(d)
        w._is_fully_idle = lambda: False
        req = FlushCacheReqInput(timeout_s=30.0)
        with mock.patch.dict(os.environ, {FW.ENV_DORMANT_REFUSE: "1"}):
            self.assertIsNone(w.handle(req))
            d["v"] = True
            w._is_fully_idle = lambda: True
            w.check_pending()
        self.assertEqual(calls, [])
        self.assertFalse(chan.sent[0][0].success)

    def test_no_dormant_reader_keeps_the_old_wrapper(self):
        calls = []
        w = FW.SchedulerFlushWrapper(flush_cache=lambda **kw: calls.append(kw) or True,
                                     is_fully_idle=lambda: True, ipc_channels=_Chan())
        self.assertTrue(w.handle(FlushCacheReqInput()).success)

    def test_the_scheduler_wires_its_dormant_flag(self):
        from flliper.srt.managers import scheduler as S

        src = inspect.getsource(S)
        i = src.index("self.flush_wrapper = SchedulerFlushWrapper(")
        self.assertIn('is_dormant=lambda: bool(getattr(self, "pdflip_dormant", False))', src[i:i + 500])

    def test_the_sleep_legs_own_flush_does_not_pass_the_wrapper(self):
        # the release's flush runs BEFORE the pause and before pdflip_dormant is set; the wake's
        # restore flush runs after the resume, before it is cleared -- both call flush_cache directly
        src = inspect.getsource(_wu)
        i = src.index("self.flush_cache(zero_kv=False)")
        self.assertLess(i, src.index("scheduler.pdflip_dormant = True", i))
        self.assertLess(i, src.index("self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)", i))


def _front(awake="P"):
    f = object.__new__(front_mod.Front)
    f.awake, f.epoch, f.state = awake, 5, "serving"
    f.counters = collections.Counter()
    return f


class OneFlipAtATime(unittest.TestCase):
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_b1_the_controllers_flip_during_the_probes_flip_is_skipped(self):
        f = _front()
        entered = []

        async def body(self, src, dst):
            entered.append((src, dst))
            await asyncio.sleep(0.05)
            self.awake = dst

        wrapped = front_mod._flip_single_flight(body)

        async def both():
            await asyncio.gather(wrapped(f, "P", "D"), wrapped(f, "P", "D"))

        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_FLIP_SINGLE_FLIGHT": "1"}), \
             self.assertLogs(front_mod.logger, level="WARNING") as cap:
            self._run(both())
        self.assertEqual(entered, [("P", "D")], "exactly one flip body ran")
        self.assertEqual(f.counters["flip_skipped_concurrent"], 1)
        self.assertIn("PDFLIP-FLIP SKIPPED epoch=5 sleep=P wake=D: another flip is open", cap.output[0])
        self.assertFalse(f._flip_open)

    def test_a_flip_of_a_sleeping_group_is_skipped(self):
        f = _front(awake="D")
        ran = []

        async def body(self, src, dst):
            ran.append(1)

        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_FLIP_SINGLE_FLIGHT": "1"}):
            self._run(front_mod._flip_single_flight(body)(f, "P", "D"))
        self.assertEqual(ran, [])

    def test_the_guard_is_released_on_an_exception(self):
        f = _front()

        async def body(self, src, dst):
            raise RuntimeError("leg failed")

        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_FLIP_SINGLE_FLIGHT": "1"}):
            with self.assertRaises(RuntimeError):
                self._run(front_mod._flip_single_flight(body)(f, "P", "D"))
        self.assertFalse(f._flip_open)

    def test_sequential_flips_both_run(self):
        f = _front()
        ran = []

        async def body(self, src, dst):
            ran.append((src, dst))
            self.awake = dst

        w = front_mod._flip_single_flight(body)

        async def seq():
            await w(f, "P", "D")
            await w(f, "D", "P")

        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_FLIP_SINGLE_FLIGHT": "1"}):
            self._run(seq())
        self.assertEqual(ran, [("P", "D"), ("D", "P")])

    def test_switch_off_is_the_metal_behaviour(self):
        f = _front()
        entered = []

        async def body(self, src, dst):
            entered.append(1)
            await asyncio.sleep(0.02)

        w = front_mod._flip_single_flight(body)

        async def both():
            await asyncio.gather(w(f, "P", "D"), w(f, "P", "D"))

        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_FLIP_SINGLE_FLIGHT": "0"}):
            self._run(both())
        self.assertEqual(len(entered), 2)

    def test_front_flip_is_guarded_and_keeps_its_source(self):
        self.assertTrue(hasattr(front_mod.Front.flip, "__wrapped__"))
        src = inspect.getsource(front_mod.Front.flip)
        self.assertIn("PDFLIP-FLIP begin epoch=", src)


if __name__ == "__main__":
    unittest.main()
