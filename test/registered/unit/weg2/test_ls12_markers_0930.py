# SPDX-License-Identifier: Apache-2.0
"""LS12 (30.09.): metal markers for the 27B Leistungsschalter that had none.

The ls12 proof boot must show per switch that it ACTS. Four switches of the twelve left no trace
in any log (checked against the green row-authority boot dkr27browauthoritybar1fs09301128):
SGLANG_DFLASH_VERIFY_VOCAB_ARGMAX, SGLANG_VRAM_PEAK_FAST_READ, SGLANG_HICACHE_LOAD_ASYNC_INDEX and
the threshold of SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS; the controller kicks
(SGLANG_WEG2_CTL_KICK_ARRIVAL / _AFTER_FLIP) counted only in the live /weg2/state.

Pinned here, per marker: switch ON -> the line appears (with its bounded cadence); switch OFF ->
no line at all (the default log stays byte-identical); a doubling cadence gives at most
log2(N)+1 lines for N events.
"""

import asyncio
import logging
import math
import os
import threading
import types
import unittest
from unittest import mock

import torch


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _capture(name):
    lg = logging.getLogger(name)
    h = _Capture()
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.DEBUG)
    return lg, h, old


class TestVramPeakFastRead(unittest.TestCase):
    def setUp(self):
        from sglang.srt.model_executor import vram_family_census as V

        self.V = V
        V._FAST_READ_SEEN.clear()
        self.lg, self.h, self.old = _capture(V.__name__)

    def tearDown(self):
        self.lg.removeHandler(self.h)
        self.lg.setLevel(self.old)
        self.V._FAST_READ_SEEN.clear()

    def _read(self, env, stats=None, boom=False):
        fake = {"allocated_bytes": {"all": {"peak": 3 << 30}}}

        def ms(_dev):
            if boom:
                raise RuntimeError("no stats")
            return stats or fake

        with mock.patch.dict(os.environ, env), \
             mock.patch.object(torch._C, "_cuda_memoryStats", ms, create=True), \
             mock.patch.object(torch.cuda, "current_device", lambda: 0), \
             mock.patch.object(torch.cuda, "max_memory_allocated", lambda *a, **k: 7):
            return [self.V._max_allocated_bytes(torch.cuda) for _ in range(3)]

    def test_on_armed_once(self):
        vals = self._read({"SGLANG_VRAM_PEAK_FAST_READ": "1"})
        self.assertEqual(vals, [3 << 30] * 3)
        lines = [l for l in self.h.lines if "VRAM-PEAK-FAST-READ" in l]
        self.assertEqual(len(lines), 1)
        self.assertIn("armed: first read 3072 MiB", lines[0])

    def test_fallback_is_named_once(self):
        vals = self._read({"SGLANG_VRAM_PEAK_FAST_READ": "1"}, boom=True)
        self.assertEqual(vals, [7, 7, 7])
        lines = [l for l in self.h.lines if "VRAM-PEAK-FAST-READ" in l]
        self.assertEqual(len(lines), 1)
        self.assertIn("fallback", lines[0])

    def test_off_no_line(self):
        vals = self._read({"SGLANG_VRAM_PEAK_FAST_READ": "0"})
        self.assertEqual(vals, [7, 7, 7])
        self.assertEqual([l for l in self.h.lines if "VRAM-PEAK-FAST-READ" in l], [])


class TestWedgeRecoveryThreshold(unittest.TestCase):
    def _run(self, env):
        from sglang.srt.managers.scheduler_components import invariant_checker as IC

        lg, h, old = _capture(IC.__name__)
        stop = threading.Event()
        stop.set()
        try:
            with mock.patch.dict(os.environ, env), \
                 mock.patch.object(IC, "make_admission_wedge_poller", lambda s: (lambda: None)):
                if "SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS" not in env:
                    os.environ.pop("SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS", None)
                t = IC.create_admission_wedge_watchdog(object(), poll_interval=0.001, stop=stop)
                t.join(2)
        finally:
            lg.removeHandler(h)
            lg.setLevel(old)
        return [l for l in h.lines if "recovery armed after" in l]

    def test_override_names_the_threshold(self):
        lines = self._run({"SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS": "2"})
        self.assertEqual(len(lines), 1)
        self.assertIn("ADMISSION-WEDGE recovery armed after 2.0s", lines[0])

    def test_unset_no_line(self):
        self.assertEqual(self._run({}), [])


class TestLoadAsyncIndex(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache.pool_host import arena_pool as A

        self.A = A
        A._LOAD_ASYNC_INDEX_SEEN.clear()
        self.lg, self.h, self.old = _capture(A.__name__)

    def tearDown(self):
        self.lg.removeHandler(self.h)
        self.lg.setLevel(self.old)
        self.A._LOAD_ASYNC_INDEX_SEEN.clear()

    def test_on_once_and_off_silent(self):
        with mock.patch.dict(os.environ, {"SGLANG_HICACHE_LOAD_ASYNC_INDEX": "0"}):
            self.assertFalse(self.A.load_index_async())
        self.assertEqual([l for l in self.h.lines if "LOAD-ASYNC-INDEX" in l], [])
        with mock.patch.dict(os.environ, {"SGLANG_HICACHE_LOAD_ASYNC_INDEX": "1"}):
            self.assertTrue(all(self.A.load_index_async() for _ in range(5)))
        lines = [l for l in self.h.lines if "LOAD-ASYNC-INDEX" in l]
        self.assertEqual(len(lines), 1)
        self.assertIn("HICACHE-LOAD-ASYNC-INDEX armed", lines[0])


class TestVerifyVocabArgmax(unittest.TestCase):
    def test_processor_note_once_per_verdict(self):
        from sglang.srt.layers import logits_processor as LP

        LP._VERIFY_LOCAL_VOCAB_SEEN.clear()
        lg, h, old = _capture(LP.__name__)
        try:
            for _ in range(3):
                LP._verify_local_vocab_note(True, {"tp_gather": True})
            LP._verify_local_vocab_note(False, {"tp_gather": False})
        finally:
            lg.removeHandler(h)
            lg.setLevel(old)
            LP._VERIFY_LOCAL_VOCAB_SEEN.clear()
        lines = [l for l in h.lines if "DFLASH-VERIFY-VOCAB-ARGMAX" in l]
        self.assertEqual(len(lines), 2)
        self.assertIn("armed", lines[0])
        self.assertIn("INERT", lines[1])
        self.assertIn("tp_gather=False", lines[1])

    def test_note_only_when_requested(self):
        """The processor calls the note only under the env -- the call site is guarded."""
        import inspect

        from sglang.srt.layers import logits_processor as LP

        src = inspect.getsource(LP.LogitsProcessor.__init__)
        i = src.index("_verify_local_vocab_note(")
        self.assertIn("if verify_local_vocab_requested():", src[max(0, i - 300):i])

    def test_round_cadence_log2(self):
        from sglang.srt.speculative import dflash_worker_v2 as W

        lg, h, old = _capture(W.__name__)
        w = types.SimpleNamespace()
        try:
            for i in range(1000):
                W._verify_vocab_round_note(w, "argmax" if i % 10 else "gather")
        finally:
            lg.removeHandler(h)
            lg.setLevel(old)
        lines = [l for l in h.lines if "DFLASH-VERIFY-VOCAB-ARGMAX rounds=" in l]
        self.assertEqual(len(lines), math.floor(math.log2(1000)) + 1)
        self.assertIn("rounds=512 argmax=460 gather=52", lines[-1])


class TestFrontKickMarker(unittest.TestCase):
    def _kicks(self, env, n):
        from sglang.srt.weg2 import front as F

        lg, h, old = _capture(F.logger.name)
        try:
            with mock.patch.dict(os.environ, env):
                f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="ls12",
                            store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={},
                            w_s=45.0, weight_chunks=2, flip_min_work_tokens=1)

            async def body():
                for _ in range(n):
                    f._kick_controller("after_flip")
                    f._kick_controller("arrival")

            asyncio.run(body())
        finally:
            lg.removeHandler(h)
            lg.setLevel(old)
        return [l for l in h.lines if "WEG2-FLIPFAST kick why=" in l]

    def test_on_doubling_per_reason(self):
        lines = self._kicks({"SGLANG_WEG2_CTL_KICK_ARRIVAL": "1", "SGLANG_WEG2_CTL_KICK_AFTER_FLIP": "1"}, 100)
        per = {w: [l for l in lines if "why=%s " % w in l] for w in ("arrival", "after_flip")}
        for w, ls in per.items():
            self.assertEqual(len(ls), math.floor(math.log2(100)) + 1, w)
        self.assertIn("why=arrival n=64", per["arrival"][-1])

    def test_off_no_line(self):
        env = {"SGLANG_WEG2_CTL_KICK_ARRIVAL": "0", "SGLANG_WEG2_CTL_KICK_AFTER_FLIP": "0"}
        self.assertEqual(self._kicks(env, 10), [])


if __name__ == "__main__":
    unittest.main()
