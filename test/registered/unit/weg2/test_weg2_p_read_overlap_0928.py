"""RO (weg2.p_read_overlap): P computes other queued work while a store read
runs -- "Read läuft, zweite Anfrage wartet".

Before: the front's drain pool counted a leg whose request P only holds for
its store read (held, twin-deferred or paced) like a computing leg, so with
``p_concurrency + ahead`` such legs in flight the next request waited in the
front and P idled (27B parkdraft 14:23:14-46, weg2-0-1 reading + weg2-0-2
TWIN-DEFER). After: PP0 names those rids in its read-state file and every
such leg frees one extra dispatch slot (capped), so the second request is
prefilled while the first one still reads.

The fake P below keeps PP0's real bookkeeping (``_weg2_store_held``,
``_weg2_told_pacing``, ``waiting_queue``) and writes it with the real
``export``; the front side is the real ``_p_drain_pool`` with the real
``ReadState`` reader.
"""

import asyncio
import collections
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.weg2 import front
from sglang.srt.weg2 import p_read_overlap as ro

PORT = 31999


class _FakeP:
    """PP0's bookkeeping, exported after every change like the pass does."""

    def __init__(self):
        self.s = SimpleNamespace(
            server_args=SimpleNamespace(port=PORT),
            _weg2_store_held={}, _weg2_told_pacing={}, waiting_queue=[])
        self.events = []

    def _export(self):
        ro.export(self.s)

    def queue(self, rid):
        self.s.waiting_queue.append(SimpleNamespace(rid=rid))

    def hold(self, rid):
        self.s._weg2_store_held[rid] = object()
        self._export()

    def pace(self, rid):
        self.s._weg2_store_held.pop(rid, None)
        self.s._weg2_told_pacing[rid] = object()
        self._export()

    def admit(self, rid):
        self.s._weg2_store_held.pop(rid, None)
        self.s._weg2_told_pacing.pop(rid, None)
        self.s.waiting_queue = [r for r in self.s.waiting_queue if r.rid != rid]
        self._export()


def _run(items, limit, overlap, reads, *, cap=2, paced=(), budget=0, cost=None,
         max_dispatch=0, idle_s=0.4):
    """Drain ``items`` (rids). A rid in ``reads`` is held by P for its store
    read until the test sets its event (``idle_s`` after the start); paced
    ones then sit in the pacing window for 0.1 s. Returns the event log."""

    async def main():
        p = _FakeP()
        log = []
        read_done = {r: asyncio.Event() for r in reads}
        loop = asyncio.get_running_loop()
        t0 = loop.time()

        def note(what, rid):
            log.append((round(loop.time() - t0, 2), what, rid))

        async def one(rid):
            note("dispatch", rid)
            p.queue(rid)
            if rid in reads:
                p.hold(rid)
                await read_done[rid].wait()
                note("read_done", rid)
                if rid in paced:
                    p.pace(rid)
                    await asyncio.sleep(0.1)
            p.admit(rid)
            note("prefill", rid)
            await asyncio.sleep(0.05)
            return rid

        async def release_reads():
            await asyncio.sleep(idle_s)
            for ev in read_done.values():
                ev.set()

        state = ro.ReadState(PORT) if overlap else None
        extra = None if state is None else (
            lambda ps: ro.extra_slots(ps, state.rids(), cap))
        rel = asyncio.ensure_future(release_reads())
        await front._p_drain_pool(
            collections.deque(items), limit, one, lambda _r: None, lambda: True,
            max_dispatch=max_dispatch, cost=cost, budget=budget,
            extra=extra, poll_s=0.02)
        await rel
        return log

    return asyncio.run(main())


def _t(log, what, rid):
    return next(t for t, w, r in log if w == what and r == rid)


class _Dir(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {ro.ENV_DIR: self._td.name})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._td.cleanup()


class ReadRunsSecondRequestWaits(_Dir):
    """The named test: one leg-1 slot, A reads, B waits."""

    def test_before_p_idles_while_a_reads(self):
        log = _run(["A", "B"], 1, overlap=False, reads={"A"})
        # B left the front only after A's read ended and A was admitted: the
        # whole read was idle on P
        self.assertGreaterEqual(_t(log, "dispatch", "B"), _t(log, "read_done", "A"))

    def test_after_b_is_prefilled_while_a_reads(self):
        log = _run(["A", "B"], 1, overlap=True, reads={"A"})
        self.assertLess(_t(log, "prefill", "B"), _t(log, "read_done", "A"))
        self.assertLess(_t(log, "dispatch", "B"), 0.2)


class TwinAndPacing(_Dir):
    def test_reading_leg_and_deferred_twin_free_two_slots(self):
        # p_concurrency 1 + ahead 1: A reads, B is its deferred twin -- both
        # held; C (unrelated, no read) is prefilled meanwhile
        log = _run(["A", "B", "C"], 2, overlap=True, reads={"A", "B"})
        self.assertLess(_t(log, "prefill", "C"), _t(log, "read_done", "A"))

    def test_pacing_window_counts_as_held(self):
        log = _run(["A", "B"], 1, overlap=True, reads={"A"}, paced={"A"})
        # B dispatched during the read already; nothing waits on the window
        self.assertLess(_t(log, "prefill", "B"), _t(log, "prefill", "A"))

    def test_paced_rid_alone_frees_a_slot(self):
        s = SimpleNamespace(server_args=SimpleNamespace(port=PORT), _weg2_store_held={},
                            _weg2_told_pacing={"A": 1}, waiting_queue=[SimpleNamespace(rid="A")])
        self.assertEqual(ro.reading_rids(s), {"A"})


class TheCapStaysACap(_Dir):
    def test_a_computing_leg_frees_nothing(self):
        # A needs no read: it computes; B waits exactly as before
        log = _run(["A", "B"], 1, overlap=True, reads=set())
        self.assertGreaterEqual(_t(log, "dispatch", "B"), _t(log, "prefill", "A"))

    def test_extra_is_capped(self):
        log = _run(["A", "B", "C", "D", "E"], 1, overlap=True, reads={"A", "B", "C", "D", "E"}, cap=2)
        early = [r for t, w, r in log if w == "dispatch" and t < 0.3]
        self.assertEqual(early, ["A", "B", "C"])

    def test_queue_order_kept(self):
        log = _run(["A", "B", "C", "D"], 1, overlap=True, reads={"A", "B"})
        self.assertEqual([r for _t2, w, r in log if w == "dispatch"], ["A", "B", "C", "D"])

    def test_phase_cap_still_binds(self):
        log = _run(["A", "B", "C"], 1, overlap=True, reads={"A"}, max_dispatch=1)
        self.assertEqual([r for _t2, w, r in log if w == "dispatch"], ["A"])

    def test_pool_budget_still_binds(self):
        # the overlap dispatch is checked against P's pool like any other
        log = _run(["A", "B"], 1, overlap=True, reads={"A"}, budget=100, cost=lambda _r: 80)
        self.assertGreaterEqual(_t(log, "dispatch", "B"), _t(log, "read_done", "A"))


class TheStateFile(_Dir):
    def _s(self, held=(), pacing=(), queued=()):
        return SimpleNamespace(server_args=SimpleNamespace(port=PORT),
                               _weg2_store_held={r: 1 for r in held},
                               _weg2_told_pacing={r: 1 for r in pacing},
                               waiting_queue=[SimpleNamespace(rid=r) for r in queued])

    def test_written_only_on_change(self):
        s = self._s(held=["A"], queued=["A"])
        self.assertTrue(ro.export(s))
        self.assertFalse(ro.export(s))
        with open(ro.state_path(PORT)) as fh:
            self.assertEqual(json.load(fh)["rids"], ["A"])
        s._weg2_store_held.clear()
        self.assertTrue(ro.export(s))

    def test_parked_rid_is_not_reading(self):
        # held for the dormant hold / settle, not in the waiting queue
        self.assertEqual(ro.reading_rids(self._s(held=["A", "B"], queued=["B"])), {"B"})

    def test_switch_off_writes_nothing(self):
        with mock.patch.dict(os.environ, {ro.ENV: "0"}):
            self.assertFalse(ro.export(self._s(held=["A"], queued=["A"])))
        self.assertFalse(os.path.exists(ro.state_path(PORT)))

    def test_previous_boot_file_is_removed(self):
        with open(ro.state_path(PORT), "w") as fh:
            json.dump({"rids": ["weg2-0-1"]}, fh)
        self.assertEqual(ro.ReadState(PORT).rids(), set())

    def test_broken_file_reads_as_nothing(self):
        st = ro.ReadState(PORT)
        with open(ro.state_path(PORT), "w") as fh:
            fh.write("{\"rids\": [")
        self.assertEqual(st.rids(), set())

    def test_export_never_raises(self):
        with mock.patch.dict(os.environ, {ro.ENV_DIR: "/nonexistent/dir"}):
            self.assertFalse(ro.export(self._s(held=["A"], queued=["A"])))

    def test_env(self):
        self.assertTrue(ro.enabled({}))
        self.assertFalse(ro.enabled({ro.ENV: "0"}))
        self.assertEqual(ro.max_extra({}), 2)
        self.assertEqual(ro.max_extra({ro.ENV_MAX: "0"}), 0)
        self.assertEqual(ro.max_extra({ro.ENV_MAX: "x"}), 2)


if __name__ == "__main__":
    unittest.main()
