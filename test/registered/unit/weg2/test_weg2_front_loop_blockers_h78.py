# SPDX-License-Identifier: Apache-2.0
"""H78 (fnFL2): no stage of the front's flip / leg path holds the event loop.

MEASURED on the metal front logs of x172-x175 (a stretch between two lines the
SAME coroutine emits with nothing awaited in between is time the loop could
not leave):

* ``WEG2-FLIP-ORDER`` -> the gathered sleep leg ``ISSUED``: 18-47 ms on EVERY
  flip (median ~22 ms) -- ``_sleep_leg_gate`` parsing the 3.2 MB append-only
  measured-record sidecar;
* each group's first sleep, kv-wake ``RETURNED`` -> ``WEG2-DC``: 132-155 ms --
  ``ps`` + ``nvidia-smi`` + the /proc dormant sample (51-63 ms) and the sidecar
  append (79-94 ms), on the loop AND before ``state = serving``;
* ``WEG2-FLIP-RATCHET post`` -> ``WEG2-FLIP done epoch=2``: 83-96 ms -- the
  ratchet's sidecar append;
* x174 leg 1 of the 97k needle: ``Server disconnected`` 3 ms after the flip,
  on a pooled P connection idle 7.66 s against P's 5 s uvicorn keep-alive.

Pinned here, each red on the tree before H78:

1. the gate parses the sidecar once per FILE VERSION, and the flip prepares
   that parse in a worker thread (loop ticker);
2. with SGLANG_WEG2_DC_OFF_PATH=1 the first-sleep reading runs in a worker the
   flip awaits (W19 keeps its place), and both sidecar appends leave the loop
   AND the flip -- FIFO, landed before the next gate and before cleanup;
3. the ratchet's post reading is still taken at ``done epoch=2``, on the loop;
4. the flip path's lazy imports are prewarmed off the loop at startup, and the
   H75 launcher prewarm imports with the cyclic GC paused and freezes the heap
   (a REAL import in a worker thread stalled the loop 6x > 40 ms, max 201 ms:
   full GC passes, 219-225 ms each in the front's heap, 0.0 ms once frozen);
5. the front's pooled connector drops an idle connection BEFORE the groups'
   keep-alive does, so a frozen loop cannot write into a closed socket.

Hermetic: no GPU (NVML stubbed), no boot; the group RPCs are stubbed, the
keep-alive case uses a real loopback socket served from its own thread.
"""
from __future__ import annotations

import asyncio
import gc
import inspect
import json
import logging
import os
import re
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout

from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import host_ledger as hl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

MIB = 1024 * 1024
ENV_DC = "SGLANG_WEG2_DC_OFF_PATH"
CARD = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
#: A loop stall the ticker must NOT see (the injected costs below are 0.6 s).
MAX_GAP_S = 0.3


def _write_sidecar(path: str, n: int = 400, tag: str = "older-boot") -> None:
    """An append-only sidecar in the real shape (image entries + FLIP rows)."""
    samples = []
    for i in range(n):
        g = ("P", "D", "FLIP")[i % 3]
        e = {"group": g, "boot_tag": f"{tag}-{i // 3}", "at": f"2026-09-{1 + i % 24:02d}T00:00:{i % 60:02d}Z",
             "commit": "c0ffee", "form_key": "wtags=17", "pids": list(range(40))}
        if g == "FLIP":
            e["flip_ratchet_gib"] = 5.0
        else:
            e.update(rss_shmem_gib=60.0, shmem_delta_gib=0.37, extra_gib=31.8)
        samples.append(e)
    with open(path, "w") as f:
        json.dump({"samples": samples}, f, indent=1)


class _Env:
    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.saved = os.environ.pop(ENV_DC, None)
        if self.on:
            os.environ[ENV_DC] = "1"
        return self

    def __exit__(self, *exc):
        os.environ.pop(ENV_DC, None)
        if self.saved is not None:
            os.environ[ENV_DC] = self.saved
        return False


def _front(measured_record: str, dc_off_path: bool, prefill_sid=0, decode_sid=0, dc_reserve=None):
    """A real Front with the group RPCs stubbed (the flipfast suite's harness)."""
    with _Env(dc_off_path):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="h78test",
                    store_dir="/tmp", prefill_sid=prefill_sid, decode_sid=decode_sid,
                    dc_reserve=dict(dc_reserve or {}), w_s=45.0, weight_chunks=2,
                    flip_min_work_tokens=1, measured_record=measured_record, commit="h78")
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
    # The X re-solve imports the launcher (seconds) on its first call: not what
    # this suite measures, and H75 already moved it off the loop.
    f.resolve_x_live = lambda: None
    return f


class _Ticker:
    """Ticks every 20 ms on the loop; ``max_gap`` is the longest the loop was held."""

    def __init__(self):
        self.ticks = []
        self.task = None

    async def __aenter__(self):
        async def run():
            while True:
                self.ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        self.task = asyncio.ensure_future(run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        return False

    @property
    def max_gap(self) -> float:
        return max((b - a for a, b in zip(self.ticks, self.ticks[1:])), default=0.0)


class _NoNvml(CustomTestCase):
    """Every flip here reads no card: `_nvml_free` is stubbed to "no reading"."""

    def setUp(self):
        self._nvml = F._nvml_free
        F._nvml_free = lambda: []
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "weg2_measured_record.json")

    def tearDown(self):
        F._nvml_free = self._nvml
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# 1. the sleep-leg gate's sidecar parse
# ---------------------------------------------------------------------------
class TheGateParsesOncePerFileVersion(_NoNvml):
    def _stub(self):
        f = F.Front.__new__(F.Front)
        f.measured_record = self.path
        return f

    def test_repeated_gates_parse_the_unchanged_sidecar_once(self):
        """RED before H78: every gate call parsed the whole file again."""
        _write_sidecar(self.path)
        f = self._stub()
        real = hl.read_measured_record
        calls = []

        def counting(path, *a, **k):
            calls.append(path)
            return real(path, *a, **k)

        with mock.patch.object(F.host_ledger, "read_measured_record", side_effect=counting), \
                mock.patch.object(hl, "read_cgroup_pressure",
                                  return_value={"file_gib": 70.0, "shmem_gib": 60.0}):
            for _ in range(3):
                self.assertIsNone(f._sleep_leg_gate("D"))
        self.assertEqual(len(calls), 1, calls)

    def test_a_replaced_sidecar_is_read_again_and_the_gate_follows_it(self):
        """The cache is never older than the file: an append (tmp + replace)
        changes the key, and the gate then grades the NEW newest entry."""
        _write_sidecar(self.path)
        f = self._stub()
        with mock.patch.object(hl, "read_cgroup_pressure",
                               return_value={"file_gib": 62.0, "shmem_gib": 60.0}):
            self.assertIsNone(f._sleep_leg_gate("D"))          # 0.37 GiB need, 2.0 cushion
            hl.append_measured_record(self.path, {
                "group": "D", "boot_tag": "newer", "at": "2026-09-30T00:00:00Z",
                "rss_shmem_gib": 60.0, "shmem_delta_gib": 4.32})
            with self.assertRaises(hl.Weg2SleepLegCushionDeficit):
                f._sleep_leg_gate("D")                           # 4.32 GiB need > 2.0

    def test_the_view_answers_exactly_what_the_ledger_reader_answers(self):
        _write_sidecar(self.path, n=1200)
        view = F.SidecarView()
        self.assertEqual(view.read(self.path), hl.read_measured_record(self.path))
        self.assertTrue(view.fresh(self.path))
        missing = os.path.join(self._tmp.name, "nope.json")
        self.assertEqual(view.read(missing), {})
        self.assertFalse(view.fresh(missing))

    def test_the_flip_prepares_the_parse_off_the_loop(self):
        """RED before H78: a slow parse (stand-in 0.6 s) held the loop inside the flip."""
        _write_sidecar(self.path)
        real = hl.read_measured_record
        threads = []

        def slow(path, *a, **k):
            threads.append(threading.get_ident())
            time.sleep(0.6)
            return real(path, *a, **k)

        async def body():
            f = _front(self.path, dc_off_path=False)
            async with _Ticker() as tk:
                with mock.patch.object(F.host_ledger, "read_measured_record", side_effect=slow):
                    await f.flip("D", "P")
            return f, tk, threading.get_ident()

        f, tk, loop_tid = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertTrue(threads, "the gate never read the sidecar")
        self.assertTrue(all(t != loop_tid for t in threads), "the sidecar was parsed on the loop")
        self.assertLess(tk.max_gap, MAX_GAP_S, f"the flip held the loop {tk.max_gap:.3f} s")

    def test_startup_prewarms_the_view(self):
        src = inspect.getsource(F.Front.startup)
        self.assertIn('app["sidecar_prewarm"] = asyncio.create_task(self._sidecar_ready())', src)
        flip = inspect.getsource(F.Front.flip)
        ready, gate = flip.index("await self._sidecar_ready()"), flip.index("self._sleep_leg_gate(S)")
        self.assertLess(ready, gate, "the view must be prepared BEFORE the gate reads it")


# ---------------------------------------------------------------------------
# 2./3. the first-sleep reading and the two sidecar appends
# ---------------------------------------------------------------------------
READ_DELAY = 0.2     # per ps / nvidia-smi stand-in; a first-sleep reading calls three of them
APPEND_DELAY = 0.6   # per sidecar append stand-in


class _Slow:
    """Slow stand-ins for ps / nvidia-smi / the sidecar append, recording threads."""

    def __init__(self, reading):
        self.reading = dict(reading)
        self.calls = []
        self._real_append = hl.append_measured_record

    def session_pids(self, sid):
        self.calls.append(("pids", threading.get_ident()))
        time.sleep(READ_DELAY)
        return {101, 102}

    def nvml_process_mib(self, pids):
        self.calls.append(("nvml", threading.get_ident()))
        time.sleep(READ_DELAY)
        return dict(self.reading)

    def append(self, path, rec):
        self.calls.append(("append:" + str(rec.get("group")), threading.get_ident()))
        time.sleep(APPEND_DELAY)
        self._real_append(path, rec)


class TheFirstSleepLeavesTheLoop(_NoNvml):
    def setUp(self):
        super().setUp()
        self._saved = (F._session_pids, F._nvml_process_mib)
        _write_sidecar(self.path, n=30)

    def tearDown(self):
        F._session_pids, F._nvml_process_mib = self._saved
        super().tearDown()

    def _install(self, reading):
        s = _Slow(reading)
        F._session_pids = s.session_pids
        F._nvml_process_mib = s.nvml_process_mib
        return s

    def _groups_in_file(self):
        with open(self.path) as fh:
            return [e.get("group") for e in json.load(fh)["samples"]]

    def test_switch_on_the_loop_stays_free_and_the_flip_skips_the_appends(self):
        """RED before H78: 0.6 s of reading + 0.6 s per append on the loop and on the flip."""
        s = self._install({CARD: 1700})
        post_threads = []
        real_post = hl.read_flip_currency_gib

        def post_reader(*a, **k):
            post_threads.append(threading.get_ident())
            return real_post(*a, **k)

        async def body():
            f = _front(self.path, dc_off_path=True, prefill_sid=11, decode_sid=22,
                       dc_reserve={CARD: 2000})
            walls = []
            with mock.patch.object(F.host_ledger, "append_measured_record", side_effect=s.append), \
                    mock.patch.object(F.host_ledger, "read_flip_currency_gib", side_effect=post_reader):
                async with _Ticker() as tk:
                    t0 = time.monotonic()
                    await f.flip("D", "P")
                    walls.append(time.monotonic() - t0)
                    await asyncio.sleep(APPEND_DELAY + 0.4)   # a P phase: D's append lands in it
                    t0 = time.monotonic()
                    await f.flip("P", "D")                      # P's first sleep AND done epoch=2
                    walls.append(time.monotonic() - t0)
                    groups_at_flip_end = self._groups_in_file()
                    settle = getattr(f, "_sidecar_writes_settled", None)  # absent before H78
                    if settle is not None:
                        await settle()
            return f, walls, tk, threading.get_ident(), groups_at_flip_end

        f, walls, tk, loop_tid, at_end = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertLess(tk.max_gap, MAX_GAP_S, f"the flip held the loop {tk.max_gap:.3f} s")
        # the reading still GATES the flip (three 0.2 s stand-ins)...
        for w in walls:
            self.assertGreater(w, 3 * READ_DELAY - 0.05, walls)
        # ...but no append is on it any more (one would add 0.6 s)
        for w in walls:
            self.assertLess(w, 3 * READ_DELAY + APPEND_DELAY - 0.2, walls)
        self.assertTrue(all(tid != loop_tid for _, tid in s.calls), s.calls)
        # the post reading is still taken at done epoch=2, on the loop
        self.assertTrue(post_threads and all(t == loop_tid for t in post_threads), post_threads)
        # the records all land, in flip order, after the flip returned
        self.assertNotEqual(at_end[-2:], ["P", "FLIP"], "the appends were still on the flip")
        self.assertEqual(self._groups_in_file()[-3:], ["D", "P", "FLIP"])
        back = hl.read_measured_record(self.path)
        self.assertEqual(back["D"]["boot_tag"], "h78test")
        self.assertEqual(back["D"]["vram_residue_mib"], {CARD: 1700})
        self.assertEqual(back["FLIP"]["boot_tag"], "h78test")
        self.assertEqual(f.dc_measured_d, {CARD: 1700})

    def test_w19_still_stops_the_first_d_sleep_before_it_closes(self):
        self._install({CARD: 2600})

        async def body():
            f = _front(self.path, dc_off_path=True, prefill_sid=11, decode_sid=22,
                       dc_reserve={CARD: 2000})
            await f.flip("D", "P")
            return f

        f = asyncio.run(body())
        self.assertEqual([n for n, _ in f.stops], ["W19 DormantResidueRefused"], f.stops)
        self.assertEqual(f.awake, "D")

    def test_the_next_gate_waits_for_an_append_still_in_the_writer(self):
        """Back to back (the adopt pair): the P->D gate must read the sidecar
        the synchronous append would have left -- D's record included."""
        s = self._install({CARD: 1700})
        seen = []

        async def body():
            f = _front(self.path, dc_off_path=True, prefill_sid=11, decode_sid=22,
                       dc_reserve={CARD: 2000})
            real_gate = f._sleep_leg_gate

            def gate(S):
                seen.append(hl.read_measured_record(self.path).get("D", {}).get("boot_tag"))
                return real_gate(S)

            f._sleep_leg_gate = gate
            with mock.patch.object(F.host_ledger, "append_measured_record", side_effect=s.append):
                await f.flip("D", "P")
                await f.flip("P", "D")
                await f._sidecar_writes_settled()
            return f

        f = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertEqual(len(seen), 2, seen)
        self.assertEqual(seen[1], "h78test", "the P->D gate read the sidecar before D's append landed")

    def test_switch_off_keeps_everything_on_the_flip(self):
        """Default path: reading AND appends on the loop, as before H78."""
        s = self._install({CARD: 1700})

        async def body():
            f = _front(self.path, dc_off_path=False, prefill_sid=11, decode_sid=22,
                       dc_reserve={CARD: 2000})
            with mock.patch.object(F.host_ledger, "append_measured_record", side_effect=s.append):
                await f.flip("D", "P")
                at_end = self._groups_in_file()
            return f, at_end, threading.get_ident()

        f, at_end, loop_tid = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertEqual(at_end[-1], "D", "the default path must append before the flip returns")
        self.assertTrue(all(tid == loop_tid for _, tid in s.calls), s.calls)

    def test_cleanup_waits_for_the_writer(self):
        s = self._install({})

        async def body():
            f = _front(self.path, dc_off_path=True)
            with mock.patch.object(F.host_ledger, "append_measured_record", side_effect=s.append):
                f._sidecar_submit(f._persist_dormant_image, {
                    "group": "D", "boot_tag": "at-cleanup", "at": "2026-09-30T00:00:00Z",
                    "rss_shmem_gib": 1.0})
                await f.cleanup({})
            return f

        asyncio.run(body())
        self.assertEqual(hl.read_measured_record(self.path)["D"]["boot_tag"], "at-cleanup")


class TheRatchetTakesItsReadingAtDone(CustomTestCase):
    class _Stub:
        def __init__(self, path):
            self.tag = "h78ratchet"
            self.commit = "h78"
            self.weights_tags = ["a"] * 17
            self.measured_record = path
            self._flip_ratchet_pre_gib = 80.0
            self._flip_ratchet_pre_at = "2026-09-25T00:00:00Z"
            self._flip_ratchet_written = True   # the caller claimed the latch
            self.ledger_arm = {}

    def test_a_taken_reading_is_written_as_given(self):
        """RED before H78: `_write_flip_ratchet` took no reading from its caller."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weg2_measured_record.json")
            stub = self._Stub(path)
            F.Front._write_flip_ratchet(stub, 2, (85.25, "2026-09-25T00:43:09Z"))
            rec = hl.read_measured_record(path)["FLIP"]
        self.assertAlmostEqual(rec["flip_ratchet_gib"], 5.25, places=3)
        self.assertAlmostEqual(rec["post_second_leg_nonreclaim_gib"], 85.25, places=3)
        self.assertEqual(rec["post_second_leg_at"], "2026-09-25T00:43:09Z")
        self.assertEqual(rec["at"], "2026-09-25T00:43:09Z")

    def test_without_a_reading_the_latch_still_rules(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weg2_measured_record.json")
            stub = self._Stub(path)
            F.Front._write_flip_ratchet(stub, 2)
            self.assertFalse(os.path.exists(path), "a claimed latch must not write twice")


# ---------------------------------------------------------------------------
# 4. the flip path's lazy imports
# ---------------------------------------------------------------------------
class TheFlipPathImportsArePrewarmed(CustomTestCase):
    def test_the_imports_run_in_a_worker_while_the_loop_ticks(self):
        """RED before H78: there was no such prewarm."""
        seen = []

        def slow_import(name):
            seen.append((name, threading.get_ident()))
            time.sleep(0.15)

        async def run():
            async with _Ticker() as tk:
                with mock.patch.object(F.importlib, "import_module", side_effect=slow_import):
                    await F.Front._prewarm_flip_path_imports(F.Front.__new__(F.Front))
            return tk, threading.get_ident()

        tk, loop_tid = asyncio.run(run())
        self.assertEqual([n for n, _ in seen], list(F.Front.FLIP_PATH_IMPORTS))
        self.assertTrue(all(t != loop_tid for _, t in seen))
        self.assertLess(tk.max_gap, MAX_GAP_S)

    def test_the_real_modules_import_from_a_worker(self):
        with self.assertNoLogs(F.logger, level="WARNING"):
            asyncio.run(F.Front._prewarm_flip_path_imports(F.Front.__new__(F.Front)))
        import sys

        for name in F.Front.FLIP_PATH_IMPORTS:
            self.assertIn(name, sys.modules)

    def test_they_are_the_modules_the_flip_path_imports_lazily(self):
        src = inspect.getsource(F)
        for name in F.Front.FLIP_PATH_IMPORTS:
            pkg, mod = name.rsplit(".", 1)
            self.assertTrue(re.search(rf"from {re.escape(pkg)}(\.{mod} import| import {mod} )", src),
                            f"{name} is no longer imported lazily in front.py")

    def test_startup_schedules_it(self):
        src = inspect.getsource(F.Front.startup)
        self.assertIn('app["flip_imports_prewarm"] = asyncio.create_task(self._prewarm_flip_path_imports())', src)


# ---------------------------------------------------------------------------
# 4b. the cyclic GC during the launcher prewarm
# ---------------------------------------------------------------------------
class TheLauncherPrewarmRunsWithoutFullGcPasses(CustomTestCase):
    def test_the_import_runs_with_the_collector_paused_then_freezes(self):
        """RED before H78: the import ran with the collector on (full passes
        of 50-200 ms, GIL held) and nothing was frozen afterwards."""
        seen = {}

        def fake_import(name):
            seen["name"] = name
            seen["gc_enabled_during"] = gc.isenabled()
            seen["thread"] = threading.get_ident()

        was = gc.isenabled()
        with mock.patch.object(F.importlib, "import_module", side_effect=fake_import), \
                mock.patch.object(F.gc, "freeze") as freeze:
            loop_tid = asyncio.run(self._prewarm())
        self.assertEqual(seen["name"], "sglang.srt.weg2.launcher")
        self.assertNotEqual(seen["thread"], loop_tid)
        self.assertFalse(seen["gc_enabled_during"], "the import ran with the collector on")
        self.assertEqual(freeze.call_count, 1, "the imported heap was not frozen")
        self.assertEqual(gc.isenabled(), was, "the collector's state was not restored")

    @staticmethod
    async def _prewarm():
        await F.Front._prewarm_launcher_import(F.Front.__new__(F.Front))
        return threading.get_ident()

    def test_a_failing_import_still_restores_the_collector(self):
        was = gc.isenabled()
        with mock.patch.object(F.importlib, "import_module", side_effect=ValueError("boom")), \
                mock.patch.object(F.gc, "freeze"), self.assertLogs(F.logger, level="WARNING"):
            asyncio.run(self._prewarm())
        self.assertEqual(gc.isenabled(), was)

    def test_a_real_prewarm_holds_the_loop_under_120_ms(self):
        """The metal shape: a fresh interpreter, the front imported, the REAL
        launcher import in the prewarm, a 2 ms ticker on the loop. Before H78:
        full GC passes of 101-201 ms inside the import (desk, 2026-09-25)."""
        code = (
            "import asyncio, time\n"
            "import sglang.srt.weg2.front as F\n"
            "async def main():\n"
            "    ticks = []\n"
            "    async def tick():\n"
            "        while True:\n"
            "            ticks.append(time.monotonic()); await asyncio.sleep(0.002)\n"
            "    t = asyncio.ensure_future(tick()); await asyncio.sleep(0.02)\n"
            "    await F.Front._prewarm_launcher_import(F.Front.__new__(F.Front))\n"
            "    t.cancel()\n"
            "    gaps = [b - a for a, b in zip(ticks, ticks[1:])]\n"
            "    print('PREWARM-GAP %.3f %d' % (max(gaps), len(ticks)))\n"
            "asyncio.run(main())\n"
        )
        import subprocess
        import sys

        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.startswith("PREWARM-GAP")]
        self.assertEqual(len(lines), 1, r.stderr[-2000:])
        max_gap, n = float(lines[0].split()[1]), int(lines[0].split()[2])
        self.assertGreater(n, 100, "the ticker never ran")
        self.assertLess(max_gap, 0.12, f"the prewarm held the loop {max_gap * 1000:.0f} ms")

    def test_the_probe_names_a_slow_full_pass_and_only_a_full_one(self):
        probe = F.install_gc_pause_probe(threshold_ms=0.0)
        try:
            with self.assertLogs(F.logger, level="WARNING") as cm:
                gc.collect(0)
                gc.collect(1)
                gc.collect()
            lines = [m for m in cm.output if "WEG2-FRONT GC-PAUSE" in m]
            self.assertEqual(len(lines), 1, cm.output)
            self.assertIn("generation=2", lines[0])
        finally:
            gc.callbacks.remove(probe)

    def test_startup_installs_it_and_cleanup_removes_it(self):
        self.assertIn('app["gc_pause_probe"] = install_gc_pause_probe()', inspect.getsource(F.Front.startup))
        self.assertIn("gc.callbacks.remove(probe)", inspect.getsource(F.Front.cleanup))
        self.assertIn("_import_without_full_gc", inspect.getsource(F.Front._prewarm_launcher_import))


# ---------------------------------------------------------------------------
# 5. the pooled connection vs the groups' keep-alive
# ---------------------------------------------------------------------------
class _KeepAliveServer:
    """Raw HTTP/1.1 on its OWN loop and thread: answers 200, and closes a
    connection that stayed idle ``keepalive_s`` after a response -- uvicorn's
    ``timeout_keep_alive``. Its own thread, so it closes while the client's
    loop is frozen, which is the x174 order of events."""

    def __init__(self, keepalive_s: float):
        self.keepalive_s = keepalive_s
        self.conns = 0
        self.requests = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(10)
        return False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/x"

    def _run(self):
        self.loop = asyncio.new_event_loop()
        srv = self.loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
        self.port = srv.sockets[0].getsockname()[1]
        self._ready.set()
        self.loop.run_forever()
        srv.close()
        self.loop.close()

    async def _handle(self, reader, writer):
        self.conns += 1
        try:
            while True:
                try:
                    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), self.keepalive_s)
                except asyncio.TimeoutError:
                    return
                m = re.search(rb"Content-Length: (\d+)", head, re.I)
                if m:
                    await reader.readexactly(int(m.group(1)))
                self.requests += 1
                body = b'{"ok":true}'
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()


SERVER_KEEPALIVE_S = 0.5
LOOP_FREEZE_S = 1.2


async def _two_requests_across_a_freeze(url: str, connector) -> object:
    async with ClientSession(connector=connector, timeout=ClientTimeout(total=10)) as s:
        async with s.post(url, json={"leg": 1}) as r:
            await r.read()
        time.sleep(LOOP_FREEZE_S)   # the loop is held (x174: 7.5 s); the server closes the idle socket
        try:
            async with s.post(url, json={"leg": 2}) as r:
                await r.read()
                return r.status
        except ClientConnectionError as e:   # ServerDisconnectedError / ClientOSError
            return type(e).__name__


class ThePoolDropsWhatTheServerAlreadyClosed(CustomTestCase):
    def test_the_client_bound_sits_below_the_server(self):
        self.assertEqual(F.rpc_keepalive(5.0)[:2], (4.0, 5.0))
        self.assertEqual(F.rpc_keepalive(0.6)[:2], (0.3, 0.6))
        self.assertEqual(F.rpc_keepalive(0.0)[:2], (None, 0.0))
        client, server, source = F.rpc_keepalive()
        self.assertIn("SGLANG_TIMEOUT_KEEP_ALIVE", source)
        self.assertLess(client, server)

    def test_the_pre_h78_connector_writes_into_the_closed_socket(self):
        """The x174 mechanism, reproduced: aiohttp's 15 s default reuses a
        connection the peer closed while the loop was frozen."""
        with _KeepAliveServer(SERVER_KEEPALIVE_S) as srv:
            async def body():
                return await _two_requests_across_a_freeze(srv.url, F.SportTCPConnector())

            out = asyncio.run(body())
        self.assertNotEqual(out, 200, "the stale connection was not reused -- the x174 mechanism "
                                      "did not reproduce, so this suite proves nothing")

    def test_the_front_connector_opens_a_fresh_one_instead(self):
        """RED before H78: there was no bounded connector (and startup used the default)."""
        with _KeepAliveServer(SERVER_KEEPALIVE_S) as srv:
            async def body():
                return await _two_requests_across_a_freeze(
                    srv.url, F.rpc_connector(SERVER_KEEPALIVE_S))

            out = asyncio.run(body())
            conns, reqs = srv.conns, srv.requests
        self.assertEqual(out, 200)
        self.assertEqual((conns, reqs), (2, 2), "leg 2 must go out on a NEW connection, once")

    def test_a_server_without_keepalive_margin_gets_force_close(self):
        async def body():
            c = F.rpc_connector(0.0)
            try:
                return c.force_close
            finally:
                await c.close()

        self.assertTrue(asyncio.run(body()))

    def test_startup_builds_the_session_on_it(self):
        src = inspect.getsource(F.Front.startup)
        self.assertIn("connector=rpc_connector(),", src)
        self.assertNotIn("connector=SportTCPConnector(),", src)
        self.assertIn("WEG2-FRONT RPC-KEEPALIVE", src)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
