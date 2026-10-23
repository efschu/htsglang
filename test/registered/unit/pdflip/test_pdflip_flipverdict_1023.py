# SPDX-License-Identifier: Apache-2.0
"""27b-flipverdict-1023: two server-side halves of the 27B duo findings of 23.10.

A (boot 16, 12:56:13Z / 12:59:04Z, group P): ADMISSION-WEDGE "1 queued, 0 running, NO first token for 103.6s" five seconds
after the request arrived. The verdict aged it from P's LAST first token (12:56:23Z, before P slept and rested in the idle
layout), so the first poll that saw the arrival alarmed, dumped WEDGE-STACK on PP0/PP1 and posted a corridor-relief
request. The fix for exactly this shape exists since deskq 1981 (queue clock: age = min(token age, age of the non-empty
queue)), but it was default OFF and no 27B profile sets it. User order 14.10.: new functions default ON -> on by default
(still bound to the dual layout, as 1981 built it).

B (boot 15, first request 139 s): NOT a JIT build -- the hash extension loaded from the cache volume in 0.02-0.14 s and
every triton cold load took 0.0 s. Still, the hash extension was first loaded by the first request (prefetch thread of
the scheduler): after a source change of hash_binding.cpp that load is a 4-7 s ninja build (y6o 01.10., queue_ms=6338).
It is now loaded at hicache init, before READY, on both census sites (init_hicache, hiradix_init).

DANGER DIRECTIONS: the queue clock never alarms earlier than before (min of the two ages), a real stand still alarms;
outside the dual layout the verdict is byte for byte the old one; the prewarm never raises (the first hash retries).
"""
from __future__ import annotations

import ast
import logging
import os
import pathlib
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import invariant_checker as IC  # noqa: E402
from flliper.srt.mem_cache.cpp_utils import native_hash as NH  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

QC = "FLLIPER_ADMISSION_WEDGE_QUEUE_CLOCK"
DUAL = "FLLIPER_PDFLIP_DUAL_LAYOUT"
SRC = pathlib.Path(NH.__file__).resolve().parents[2]   # .../flliper/srt


def _unset_qc(dual: bool = True):
    e = {k: v for k, v in os.environ.items() if k != QC}
    e[DUAL] = "1" if dual else "0"
    return mock.patch.dict(os.environ, e, clear=True)


class _Sched:
    def __init__(self, last_token, queued=0, running=0):
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = types.SimpleNamespace(reqs=[object()] * running)
        self.last_first_token_progress_time = last_token
        self.last_prefill_progress_time = None
        self.forward_ct = 0


class QueueClockDefaultOn1023(CustomTestCase):
    def test_unset_env_is_armed_in_the_dual_layout(self):
        with _unset_qc():
            self.assertTrue(IC._wedge_queue_clock_armed())

    def test_boot16_p_arrival_after_sleep_and_rest_does_not_alarm(self):
        """P's last first token 12:56:23Z (t=0), P sleeps 12:56:23-12:56:56 and rests empty until the arrival 12:58:59Z
        (t=155.6); the server polled 12:59:04Z (t=160.6) and alarmed with 'NO first token for 103.6s'."""
        with _unset_qc():
            s = _Sched(last_token=0.0, queued=0)
            for now in (40.0, 80.0, 120.0, 150.0):          # empty queue while P rests: never an alarm, no stamp
                alarm, _ = IC.check_admission_wedge_once(s, now=now)
                self.assertFalse(alarm)
            s.waiting_queue = [object()]                     # pdflip-5-20 arrives
            alarm, detail = IC.check_admission_wedge_once(s, now=160.6)
            self.assertFalse(alarm, detail)                  # the false alarm of 12:59:04Z
            alarm, detail = IC.check_admission_wedge_once(s, now=181.0)
            self.assertTrue(alarm, detail)                   # queued 20.4 s, 0 running, no token: a real stand still alarms

    def test_outside_the_dual_layout_the_old_verdict_stays(self):
        with _unset_qc(dual=False):
            s = _Sched(last_token=0.0, queued=1)
            alarm, detail = IC.check_admission_wedge_once(s, now=100.0)
            self.assertTrue(alarm, detail)
            self.assertFalse(hasattr(s, "_wedge_queue_since"))


class HashPrewarm1023(CustomTestCase):
    def test_prewarm_loads_once_and_logs(self):
        mod = types.SimpleNamespace(get_hash=lambda *a, **k: None)
        with mock.patch.object(NH, "_MODULE", None), mock.patch.object(NH, "_load_via_torch", return_value=mod) as ld, \
                self.assertLogs(NH.logger, level=logging.INFO) as cm:
            self.assertTrue(NH.prewarm_native_hash("init_hicache"))
            self.assertTrue(NH.prewarm_native_hash("init_hicache"))
            self.assertIs(NH._MODULE, mod)
        self.assertEqual(ld.call_count, 1)
        self.assertTrue(any("HICACHE-HASH-PREWARM site=init_hicache" in r for r in cm.output), cm.output)

    def test_prewarm_failure_is_logged_never_raised(self):
        with mock.patch.object(NH, "_MODULE", None), \
                mock.patch.object(NH, "_load_via_torch", side_effect=RuntimeError("no compiler")), \
                self.assertLogs(NH.logger, level=logging.WARNING) as cm:
            self.assertFalse(NH.prewarm_native_hash("hiradix_init"))
            self.assertIsNone(NH._MODULE)
        self.assertTrue(any("FAILED" in r for r in cm.output), cm.output)


def _calls_in(fn: ast.AST) -> list:
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
    return out


class HashPrewarmSites1023(CustomTestCase):
    """Both census sites (pdflip/carrier_census.CENSUS_SITES) load the hash extension at hicache init, before READY."""

    def _fn(self, rel: str, name: str, cls: str | None = None) -> ast.AST:
        tree = ast.parse((SRC / rel).read_text())
        scope = tree
        if cls:
            scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_unified_factory_prewarms_after_init_hicache(self):
        fn = self._fn("mem_cache/registry.py", "_create_unified_radix_cache")
        calls = _calls_in(fn)
        self.assertIn("init_hicache", calls)
        self.assertIn("prewarm_native_hash", calls)

    def test_hiradix_init_prewarms(self):
        fn = self._fn("mem_cache/hiradix_cache.py", "__init__", cls="HiRadixCache")
        self.assertIn("prewarm_native_hash", _calls_in(fn))
