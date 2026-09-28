# SPDX-License-Identifier: Apache-2.0
"""#55 F2: graphics-clock lock after 1 s of idle in the default layout.

User order 28.09.: "warum erst nach 60s ? warum nicht nach 1s leerlauf im
default layout, das 'hochtakten' wird doch nur ms brauchen?? <<< umsetzen".

DANGER DIRECTIONS guarded here:
* a request or a flip must find the clocks UNLOCKED before it proceeds
  (unlock is synchronous, at the request's first byte / the flip's first
  statement);
* no lock while a flip is open, a request is in flight, or work is queued --
  the signal is the front's own state, never GPU load;
* fail-open: no daemon, a refusing daemon or a dead connection leave the
  clocks untouched, and a front that dies releases the lock (the connection
  is the lease);
* the switch is OFF by default.

Hermetic: no GPU, no NVML, no boot. The daemon runs against a fake NVML on a
unix socket in a temp dir.
"""
from __future__ import annotations

import asyncio
import collections
import os
import tempfile
import threading
import unittest

from sglang.srt.weg2 import idle_clock
from sglang.srt.weg2 import idle_clock_daemon as dmn


class FakeClient:
    def __init__(self, fail_lock=None, fail_unlock=None, refuse=False, unlock_s=0.0):
        self.calls = []
        self.closed = 0
        self.fail_lock, self.fail_unlock, self.refuse = fail_lock, fail_unlock, refuse
        self.unlock_s = unlock_s
        self.addr = "fake:0"

    def call(self, op, **extra):
        if op == "unlock" and self.unlock_s:
            import time as _t
            _t.sleep(self.unlock_s)   # the ~30 ms of a -lmc unlock, stretched
        self.calls.append(op)
        if op == "lock" and self.fail_lock:
            raise self.fail_lock
        if op == "unlock" and self.fail_unlock:
            raise self.fail_unlock
        if op == "lock" and self.refuse:
            return {"ok": False, "err": "no permission"}
        return {"ok": True, "ms": 0.4, "cards": [0, 1, 2], "mhz": [210, 210, 210]}

    def close(self):
        self.closed += 1


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _ic(client=None, idle_s=1.0):
    clk = Clock()
    return idle_clock.IdleClock(client or FakeClient(), idle_s=idle_s, clock=clk), clk


class TestLockAfterOneSecond(unittest.TestCase):
    def test_locks_only_after_idle_s_at_rest(self):
        ic, clk = _ic()
        ic.note_rest()                  # rest begins
        clk.t += 0.6
        ic.note_rest()
        self.assertEqual(ic.client.calls, [])
        clk.t += 0.4                    # 1.0 s at rest
        ic.note_rest()
        self.assertEqual(ic.client.calls, ["lock"])
        self.assertTrue(ic.locked)
        clk.t += 5
        ic.note_rest()                  # stays locked, no second lock
        self.assertEqual(ic.client.calls, ["lock"])

    def test_default_idle_s_is_one_second(self):
        self.assertEqual(idle_clock.DEFAULT_IDLE_S, 1.0)

    def test_busy_restarts_the_second(self):
        ic, clk = _ic()
        ic.note_rest()
        clk.t += 0.9
        ic.note_busy("controller-busy")
        ic.note_rest()
        clk.t += 0.9
        ic.note_rest()
        self.assertEqual(ic.client.calls, [])


class TestUnlockBeforeWork(unittest.TestCase):
    def _locked(self, client=None):
        ic, clk = _ic(client)
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        self.assertTrue(ic.locked)
        return ic, clk

    def test_request_entry_unlocks_synchronously(self):
        ic, _ = self._locked()
        ic.enter()
        self.assertEqual(ic.client.calls, ["lock", "unlock"])
        self.assertFalse(ic.locked)
        self.assertEqual(ic.inflight, 1)

    def test_no_lock_while_a_request_is_in_flight(self):
        ic, clk = _ic()
        ic.enter()
        for _ in range(5):
            ic.note_rest()
            clk.t += 1.0
        self.assertEqual(ic.client.calls, [])
        ic.leave()
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        self.assertEqual(ic.client.calls, ["lock"])

    def test_no_lock_while_a_flip_is_open_or_work_is_queued(self):
        ic, clk = _ic()
        for kw in ({"serving": False}, {"queued": 1}):
            for _ in range(4):
                ic.note_rest(**kw)
                clk.t += 1.0
        self.assertEqual(ic.client.calls, [])

    def test_request_entry_starts_the_unlock_without_waiting_and_ready_waits(self):
        ic, _ = self._locked(FakeClient(unlock_s=0.2))

        async def go():
            import time as _t
            t0 = _t.perf_counter()
            ic.enter()                                   # returns at once; unlock runs in a thread
            entered_ms = (_t.perf_counter() - t0) * 1e3
            self.assertTrue(ic._unlock_pending())
            ic.note_rest()                               # never re-locks while the unlock is pending
            await ic.ready()                             # the first group request waits here
            return entered_ms

        entered_ms = asyncio.run(go())
        self.assertLess(entered_ms, 100.0)
        self.assertEqual(ic.client.calls, ["lock", "unlock"])
        self.assertFalse(ic.locked)
        self.assertEqual(ic.counters["gate_waits"], 1)

    def test_second_request_does_not_start_a_second_unlock(self):
        ic, _ = self._locked(FakeClient(unlock_s=0.1))

        async def go():
            ic.enter()
            ic.enter()
            await ic.ready()

        asyncio.run(go())
        self.assertEqual(ic.client.calls, ["lock", "unlock"])

    def test_flip_waits_for_a_pending_request_unlock(self):
        ic, _ = self._locked(FakeClient(unlock_s=0.1))

        async def go():
            ic.enter()
            await ic.before_flip()
            return ic.locked

        self.assertFalse(asyncio.run(go()))
        self.assertEqual(ic.client.calls, ["lock", "unlock"])

    def test_flip_unlocks_before_its_first_statement(self):
        from sglang.srt.weg2 import front as front_mod
        f = front_mod.Front.__new__(front_mod.Front)
        ic, _ = self._locked()
        f._idle_clock = ic
        f._flip_open = False
        f.awake = "D"
        f.counters = collections.Counter()
        f.groups = {}  # the flip body fails at its first group lookup -- after the unlock
        with self.assertRaises(KeyError):
            asyncio.run(f.flip("D", "P"))
        self.assertEqual(ic.client.calls, ["lock", "unlock"])
        self.assertFalse(ic.locked)


class TestFrontIdleDisposition(unittest.TestCase):
    def _front(self, ic, awake_layout="D", state="serving"):
        from sglang.srt.weg2 import front as front_mod
        f = front_mod.Front.__new__(front_mod.Front)
        f._idle_clock = ic
        f.idle_layout = awake_layout
        f._idle_rest_shown = True  # skip the edge-triggered log line
        f.queue = collections.deque()
        f._ready_for_d = collections.deque()
        f.state = state
        return f

    def test_rest_in_the_configured_layout_locks_after_one_second(self):
        ic, clk = _ic()
        f = self._front(ic)
        self.assertEqual(f._idle_disposition("D", True), "rest")
        clk.t += 1.0
        f._idle_disposition("D", True)
        self.assertEqual(ic.client.calls, ["lock"])
        self.assertEqual(f._idle_disposition("D", False), "busy")
        self.assertEqual(ic.client.calls, ["lock", "unlock"])

    def test_not_the_configured_layout_never_locks(self):
        ic, clk = _ic()
        f = self._front(ic, awake_layout="P")
        for _ in range(4):
            self.assertEqual(f._idle_disposition("D", True), "flip")
            clk.t += 1.0
        self.assertEqual(ic.client.calls, [])

    def test_flipping_state_never_locks(self):
        ic, clk = _ic()
        f = self._front(ic, state="flipping")
        for _ in range(4):
            f._idle_disposition("D", True)
            clk.t += 1.0
        self.assertEqual(ic.client.calls, [])

    def test_switch_off_leaves_the_disposition_untouched(self):
        f = self._front(None)
        self.assertEqual(f._idle_disposition("D", True), "rest")
        self.assertEqual(f._idle_disposition("D", False), "busy")


class TestFailOpen(unittest.TestCase):
    def test_no_daemon_means_no_lock_and_a_retry_later(self):
        ic, clk = _ic(FakeClient(fail_lock=ConnectionRefusedError("refused")))
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        self.assertFalse(ic.locked)
        self.assertEqual(ic.counters["unavailable"], 1)
        clk.t += 5.0
        ic.note_rest()                              # inside retry_s: no new attempt
        self.assertEqual(ic.client.calls, ["lock"])
        clk.t += idle_clock.RETRY_S
        ic.note_rest()
        self.assertEqual(ic.client.calls, ["lock", "lock"])

    def test_refusing_daemon_is_not_a_lock(self):
        ic, clk = _ic(FakeClient(refuse=True))
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        self.assertFalse(ic.locked)
        self.assertEqual(ic.client.closed, 1)

    def test_failed_unlock_releases_by_closing_the_lease(self):
        ic, clk = _ic(FakeClient(fail_unlock=TimeoutError("slow")))
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        ic.enter()
        self.assertFalse(ic.locked)
        self.assertEqual(ic.client.closed, 1)
        self.assertEqual(ic.counters["unlock_via_close"], 1)

    def test_switch_is_off_by_default(self):
        self.assertIsNone(idle_clock.from_env({}))
        self.assertIsNone(idle_clock.from_env({"SGLANG_WEG2_IDLE_CLOCK": "0"}))
        self.assertIsNotNone(idle_clock.from_env({"SGLANG_WEG2_IDLE_CLOCK": "1"}))
        self.assertEqual(idle_clock.middlewares(None), [])


class FakeNvml:
    """gfx lock always works (unless fail_index); memory modes may be refused like on a GeForce."""

    def __init__(self, fail_index=None, mem_rc=None, app_rc=None, mem_fail_index=None):
        self.locked, self.mem, self.app = {}, {}, {}
        self.ops = []
        self.fail_index, self.mem_rc, self.app_rc, self.mem_fail_index = fail_index, mem_rc, app_rc, mem_fail_index

    def lock(self, h, lo, hi):
        if h == self.fail_index:
            raise dmn.NvmlError("NVML rc=4 Insufficient Permissions", 4)
        self.locked[h] = (lo, hi)
        self.ops.append(("lock", h))

    def reset(self, h):
        self.locked.pop(h, None)
        self.ops.append(("reset", h))

    def lock_mem(self, h, lo, hi):
        if self.mem_rc is not None or h == self.mem_fail_index:
            raise dmn.NvmlError("NVML rc=3 Not Supported", self.mem_rc or 3)
        self.mem[h] = (lo, hi)

    def reset_mem(self, h):
        if self.mem_rc is not None and h not in self.mem:
            raise dmn.NvmlError("NVML rc=3 Not Supported", self.mem_rc)
        self.mem.pop(h, None)

    def set_app(self, h, mem, gfx):
        if self.app_rc is not None:
            raise dmn.NvmlError("NVML rc=3 Not Supported", self.app_rc)
        self.app[h] = (mem, gfx)

    def reset_app(self, h):
        if self.app_rc is not None:
            raise dmn.NvmlError("NVML rc=3 Not Supported", self.app_rc)
        self.app.pop(h, None)


def _cards(n=3):
    return [dmn.Card(i, f"GPU-{i}", i, 210, mem_min=405, gfx_at_mem_min=210) for i in range(n)]


class TestDaemonState(unittest.TestCase):
    def test_lock_unlock_and_holders(self):
        nv = FakeNvml()
        st = dmn.ClockState(nv, _cards())
        a, b = object(), object()
        self.assertTrue(st.lock(a)["locked"])
        self.assertEqual(nv.locked, {0: (210, 210), 1: (210, 210), 2: (210, 210)})
        st.lock(b)
        self.assertTrue(st.release(a)["locked"])    # b still holds it
        self.assertFalse(st.release(b)["locked"])
        self.assertEqual(nv.locked, {})

    def test_partial_lock_is_undone(self):
        nv = FakeNvml(fail_index=2)
        st = dmn.ClockState(nv, _cards())
        r = st.lock(object())
        self.assertFalse(r["ok"])
        self.assertFalse(r["locked"])
        self.assertEqual(nv.locked, {})
        self.assertEqual(st.holders, set())


class TestDaemonMemoryModes(unittest.TestCase):
    """28.09. 3080 test: the SM lock alone saved 12 W (still P2); the memory clock is the lever."""

    def test_default_is_graphics_only(self):
        nv = FakeNvml()
        r = dmn.ClockState(nv, _cards()).lock(object())
        self.assertTrue(r["locked"])
        self.assertEqual((r["mem_mode"], r["mem_locked"], nv.mem), ("off", False, {}))

    def test_mem_lock_to_the_lowest_supported_clock_and_back(self):
        nv = FakeNvml()
        st = dmn.ClockState(nv, _cards(), mem="lock")
        a = object()
        r = st.lock(a)
        self.assertEqual((r["mem_mode"], r["mem_locked"], r["mem_mhz"]), ("lock", True, [405, 405, 405]))
        self.assertEqual(nv.mem, {0: (405, 405), 1: (405, 405), 2: (405, 405)})
        st.release(a)
        self.assertEqual((nv.mem, nv.locked), ({}, {}))

    def test_request_overrides_the_default_mode(self):
        nv = FakeNvml()
        st = dmn.ClockState(nv, _cards(), mem="off")
        r = st.lock(object(), "app")
        self.assertEqual((r["mem_mode"], r["mem_locked"]), ("app", True))
        self.assertEqual(nv.app[0], (405, 210))
        self.assertFalse(st.lock(object(), "bogus")["ok"])

    def test_refused_memory_step_keeps_the_graphics_lock_and_names_it(self):
        nv = FakeNvml(mem_rc=3)
        st = dmn.ClockState(nv, _cards(), mem="lock")
        a = object()
        r = st.lock(a)
        self.assertTrue(r["ok"] and r["locked"])
        self.assertFalse(r["mem_locked"])
        self.assertIn("Not Supported", r["mem_err"])
        self.assertEqual(len(nv.locked), 3)
        rel = st.release(a)
        self.assertTrue(rel["ok"])            # the unsupported mem reset is not an error
        self.assertEqual(nv.locked, {})

    def test_partial_memory_lock_is_undone(self):
        nv = FakeNvml(mem_fail_index=2)
        st = dmn.ClockState(nv, _cards(), mem="lock")
        r = st.lock(object())
        self.assertFalse(r["mem_locked"])
        self.assertEqual(nv.mem, {})
        self.assertEqual(len(nv.locked), 3)


class TestDaemonLeaseEndToEnd(unittest.TestCase):
    """A real daemon loop on a unix socket, the real DaemonClient on the other end."""

    def test_close_of_the_connection_releases_the_lock(self):
        nv = FakeNvml()
        st = dmn.ClockState(nv, _cards())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ic.sock")
            loop = asyncio.new_event_loop()
            started = threading.Event()

            async def run():
                srv = await asyncio.start_unix_server(lambda r, w: dmn.serve_conn(st, r, w), path=path)
                started.set()
                async with srv:
                    await srv.serve_forever()

            task_holder = {}

            def runner():
                asyncio.set_event_loop(loop)
                task_holder["t"] = loop.create_task(run())
                try:
                    loop.run_until_complete(task_holder["t"])
                except asyncio.CancelledError:
                    pass

            th = threading.Thread(target=runner, daemon=True)
            th.start()
            self.assertTrue(started.wait(5))
            try:
                c = idle_clock.DaemonClient("unix:" + path, 2.0)
                r = c.call("lock")
                self.assertTrue(r["ok"] and r["locked"])
                self.assertEqual(len(nv.locked), 3)
                self.assertTrue(c.call("unlock")["ok"])
                self.assertEqual(nv.locked, {})
                r = c.call("lock", mem="lock")
                self.assertTrue(r["mem_locked"])
                self.assertEqual(len(nv.locked), 3)
                self.assertEqual(len(nv.mem), 3)
                c.close()                       # the front dies / gives up: lease gone
                for _ in range(200):
                    if not nv.locked and not nv.mem:
                        break
                    threading.Event().wait(0.01)
                self.assertEqual((nv.locked, nv.mem), ({}, {}))
            finally:
                loop.call_soon_threadsafe(task_holder["t"].cancel)
                th.join(5)


class TestMiddleware(unittest.TestCase):
    def test_post_counts_in_flight_and_unlocks_first(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        ic, clk = _ic()
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        seen = {}

        async def handler(request):
            seen["inflight"] = ic.inflight
            await ic.ready()
            seen["locked"] = ic.locked
            return web.json_response({"ok": True})

        async def go():
            app = web.Application(middlewares=idle_clock.middlewares(ic))
            app.router.add_post("/generate", handler)
            app.router.add_get("/health", handler)
            async with TestClient(TestServer(app)) as cl:
                await cl.get("/health")
                self.assertEqual(seen, {"inflight": 0, "locked": True})   # GET does not count
                await cl.post("/generate", json={})
            return seen

        asyncio.run(go())
        self.assertEqual(seen, {"inflight": 1, "locked": False})
        self.assertEqual(ic.inflight, 0)


class TestGroupRequestGate(unittest.TestCase):
    """The front's group session carries the gate: a request to a group waits for the pending unlock."""

    def test_group_request_starts_only_after_the_unlock_landed(self):
        from aiohttp import ClientSession, web
        from aiohttp.test_utils import TestServer

        ic, clk = _ic(FakeClient(unlock_s=0.15))
        ic.note_rest()
        clk.t += 1.0
        ic.note_rest()
        seen = {}

        async def group(request):          # stands in for the P/D group server
            seen["calls_at_group"] = list(ic.client.calls)
            return web.json_response({"ok": True})

        async def go():
            app = web.Application()
            app.router.add_post("/generate", group)
            async with TestServer(app) as srv:
                async with ClientSession(trace_configs=idle_clock.trace_configs(ic)) as s:
                    ic.enter()                  # client request's first byte at the front
                    async with s.post(srv.make_url("/generate"), json={}) as r:
                        await r.read()

        asyncio.run(go())
        self.assertEqual(seen["calls_at_group"], ["lock", "unlock"])
        self.assertEqual(idle_clock.trace_configs(None), [])


if __name__ == "__main__":
    unittest.main()
