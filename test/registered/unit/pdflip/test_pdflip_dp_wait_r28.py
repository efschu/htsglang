# SPDX-License-Identifier: Apache-2.0
"""R28 (release table row 28): one ``PDFLIP DP-WAIT`` line per D->P waiter.

THE GAP. Under agent load a request that must go over P and arrives while D
is awake waits for a D->P flip. The front log carried the pieces of that wait
on five line shapes and no line per request, so "why did pdflip-4-5 wait 103 s"
took a join over ``BATCH queued (awake=D``, ``PDFLIP-FAIRNESS``, ``PDFLIP-FLIP
begin``, ``PDFLIP-FLIP-TIMELINE`` and ``PDFLIP-FLIP done``. Measured, rc9i
(dkrnfbar1agent09252237, front.log):

* :536 22:47:14.655 pdflip-4-5 queued (awake=D), D decoding pdflip-1-4;
* :560 22:47:58.428 the admitter hands pdflip-0-2 to D (1.3 s before the bound);
* :562 22:47:59.745 PDFLIP-FAIRNESS fired (45.1 s), :565 PDFLIP-FLIP begin
  outstanding=1;
* :613 quiesce@55882 ms -- the flip's drain waited for pdflip-0-2 (57.2 s decode,
  :580);
* :614 22:48:58.132 PDFLIP-FLIP done flip_total=58387 ms -> P awake after 103.5 s.

These tests pin the decomposition on exactly those numbers, the hooks that
feed it, and that it is an instrument only. Hermetic, CPU.
"""
from __future__ import annotations

import collections
import inspect
import logging
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from flliper.srt.pdflip import dp_wait as DW  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# rc9i front.log wall clock, seconds after 22:47:00
T_Q_45 = 14.655      # :536 pdflip-4-5 BATCH queued (awake=D)
T_Q_46 = 61.146      # :570 pdflip-4-6 queued inside the open flip (22:48:01.146)
T_BEGIN = 59.745     # :565 PDFLIP-FLIP begin epoch=4 sleep=D wake=P outstanding=1
T_DRAIN = T_BEGIN + 55.882   # :613 quiesce@55882
T_DONE = 118.132     # :614 PDFLIP-FLIP done (22:48:58.132)


def _counters(**kw):
    c = collections.Counter()
    c.update(kw)
    return c


class TheDecomposition(CustomTestCase):

    def test_rc9i_pdflip_4_5_fairness_then_drain(self):
        a = DW.take(T_Q_45, "long", False, d_outstanding=1, handoff=0, ready_for_d=1,
                    counters=_counters(fairness_bound_hits=0, d_admits=5))
        dec = DW.decompose(a, T_BEGIN, T_DRAIN, T_DONE)
        self.assertAlmostEqual(dec["wait_s"], 103.477, places=3)
        self.assertAlmostEqual(dec["hold_s"], 45.090, places=3)
        self.assertAlmostEqual(dec["drain_s"], 55.882, places=3)
        self.assertAlmostEqual(dec["flip_s"], 2.505, places=3)
        self.assertAlmostEqual(dec["hold_s"] + dec["drain_s"] + dec["flip_s"], dec["wait_s"], places=9)
        self.assertEqual(DW.dominant(dec), "drain")
        by = DW.hold_by(a, _counters(fairness_bound_hits=1, d_admits=6))
        self.assertEqual(by, "d-work+fairness")

    def test_rc9i_pdflip_4_6_arrived_inside_the_open_flip(self):
        a = DW.take(T_Q_46, "long", True, 1, 0, 0, _counters())
        dec = DW.decompose(a, T_BEGIN, T_DRAIN, T_DONE)
        self.assertEqual(dec["hold_s"], 0.0)
        self.assertAlmostEqual(dec["drain_s"], T_DRAIN - T_Q_46, places=6)   # 54.481
        self.assertAlmostEqual(dec["flip_s"], T_DONE - T_DRAIN, places=6)    # 2.505
        self.assertAlmostEqual(dec["wait_s"], 56.986, places=3)
        self.assertEqual(DW.hold_by(a, _counters(fairness_bound_hits=9)), "flip-open")

    def test_arrival_after_the_drain_has_only_flip_time(self):
        a = DW.take(T_DRAIN + 0.5, "long", True, 0, 0, 0, _counters())
        dec = DW.decompose(a, T_BEGIN, T_DRAIN, T_DONE)
        self.assertEqual((dec["hold_s"], dec["drain_s"]), (0.0, 0.0))
        self.assertAlmostEqual(dec["flip_s"], T_DONE - T_DRAIN - 0.5, places=6)

    def test_immediate_flip_on_an_idle_d(self):
        a = DW.take(10.0, "long", False, 0, 0, 0, _counters())
        dec = DW.decompose(a, 10.0, 10.1, 12.5)
        self.assertEqual(dec["hold_s"], 0.0)
        self.assertEqual(DW.dominant(dec), "flip")
        self.assertEqual(DW.hold_by(a, _counters()), "none")

    def test_missing_drain_mark_books_no_drain(self):
        a = DW.take(10.0, "long", False, 0, 0, 0, _counters())
        dec = DW.decompose(a, 11.0, None, 14.0)
        self.assertEqual(dec["drain_s"], 0.0)
        self.assertAlmostEqual(dec["hold_s"] + dec["flip_s"], 4.0, places=9)

    def test_hold_by_names_each_component_from_counter_rises(self):
        a = DW.take(0.0, "batch", False, 0, 0, 0,
                    _counters(min_dwell_holds=3, pdflip_drain_waiting=1, W1_PdFlipDrainRefused=0))
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=5, pdflip_drain_waiting=1)),
                         "min-dwell")
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=3, pdflip_drain_waiting=2)),
                         "flip-returned")
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=3, pdflip_drain_waiting=1,
                                                 W1_PdFlipDrainRefused=1)),
                         "flip-returned")
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=3, pdflip_drain_waiting=1, d_admits=1)),
                         "d-work")

    def test_the_line_carries_every_term(self):
        a = DW.take(T_Q_45, "long", False, 1, 0, 1, _counters(d_admits=5))
        dec = DW.decompose(a, T_BEGIN, T_DRAIN, T_DONE)
        s = DW.line("pdflip-4-5", 5, a, dec, "d-work+fairness", _counters(d_admits=6),
                    238321, 238321, 0, False)
        for frag in ("PDFLIP DP-WAIT rid=pdflip-4-5 epoch=5 wait_s=103.5 hold_s=45.1 drain_s=55.9 "
                     "flip_s=2.5 dominant=drain hold_by=d-work+fairness origin=long",
                     "d_outstanding=1 handoff=0 ready_for_d=1", "d_admitted_during_wait=1",
                     "est_prompt=238321 uncached=238321 presence_span=0 span_known=False"):
            self.assertIn(frag, s)


def _fake_front(awake="D", state="serving", outstanding=1, ready=0, handoff=0):
    fake = SimpleNamespace(
        awake=awake, state=state, counters=collections.Counter(),
        groups={"D": SimpleNamespace(outstanding={f"r{i}": None for i in range(outstanding)})},
        _ready_for_d=collections.deque(range(ready)), queue=collections.deque(),
        t_awake=0.0, epoch=5)
    fake._handoff_in_flight = lambda: handoff
    return fake


def _pending(rid="pdflip-4-5"):
    return F.Pending(rid, "/v1/messages", {}, "", 0.0, None, est_prompt=238321,
                     est_uncached=238321)


class TheFrontHooks(CustomTestCase):

    def test_mark_only_while_d_is_awake(self):
        fr, p = _fake_front(awake="P"), _pending()
        p.dp_arrival = DW.take(0.0, "long", False, 0, 0, 0, {})
        F.Front._dp_mark(fr, p, "long")
        self.assertIsNone(p.dp_arrival, "a request queued while P is awake waits for no D->P flip")
        fr = _fake_front(awake="D", state="flipping", outstanding=1, ready=2, handoff=1)
        F.Front._dp_mark(fr, p, "long")
        a = p.dp_arrival
        self.assertEqual((a.flipping, a.d_outstanding, a.ready_for_d, a.handoff, a.origin),
                         (True, 1, 2, 1, "long"))

    def test_mark_never_raises(self):
        fr, p = _fake_front(), _pending()
        fr.groups = {}  # broken front state
        F.Front._dp_mark(fr, p, "long")
        self.assertIsNone(p.dp_arrival)

    def test_report_prints_once_and_counts(self):
        fr = _fake_front()
        p, q = _pending("pdflip-4-5"), _pending("pdflip-4-9")
        p.dp_arrival = DW.take(T_Q_45, "long", False, 1, 0, 1, fr.counters)
        fr.queue.extend([p, q])          # q: queued while P was awake, no snapshot
        fr.t_awake = T_DONE
        with self.assertLogs("pdflip.front", level=logging.INFO) as cm:
            F.Front._dp_report(fr, T_BEGIN, T_DRAIN)
        lines = [r.getMessage() for r in cm.records if "PDFLIP DP-WAIT" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        self.assertIn("rid=pdflip-4-5", lines[0])
        self.assertIsNone(p.dp_arrival, "reported once, then cleared")
        self.assertEqual(fr.counters["dp_wait_reported"], 1)
        self.assertEqual(fr.counters["dp_wait_max_ms"], 103477)
        self.assertEqual((fr.counters["dp_wait_ge10s"], fr.counters["dp_wait_ge30s"],
                          fr.counters["dp_wait_ge60s"]), (1, 1, 1))
        F.Front._dp_report(fr, T_BEGIN, T_DRAIN)   # second flip: nothing left to report
        self.assertEqual(fr.counters["dp_wait_reported"], 1)

    def test_report_never_raises(self):
        fr, p = _fake_front(), _pending()
        p.dp_arrival = "not a snapshot"
        fr.queue.append(p)
        F.Front._dp_report(fr, T_BEGIN, T_DRAIN)
        self.assertEqual(fr.counters["dp_wait_reported"], 0)


class TheWiring(CustomTestCase):
    """Source pins: where the snapshot is taken and where the line is printed."""

    def test_every_queue_entry_on_the_d_side_takes_a_snapshot(self):
        for meth in (F.Front.handle_generate, F.Front.leg2, F.Front._requeue_after_x_refusal):
            lines = inspect.getsource(meth).split("\n")
            idx = [i for i, ln in enumerate(lines) if ".queue.append(" in ln]
            self.assertTrue(idx, meth.__name__)
            for i in idx:
                self.assertIn("self._dp_mark(", lines[i + 1],
                              f"{meth.__name__}: queue.append without a DP-WAIT snapshot")

    def test_the_intake_stall_requeue_takes_none(self):
        # P is awake there; a snapshot would report a P-phase wait as D->P.
        self.assertNotIn("_dp_mark", inspect.getsource(F.Front._requeue_intake_stalled))

    def test_flip_reports_after_done_and_only_d_to_p(self):
        src = inspect.getsource(F.Front.flip)
        i_done = src.index('logger.info("PDFLIP-FLIP done epoch=')
        i_rep = src.index("self._dp_report(t_flip0, _dp_drain_end)")
        self.assertLess(i_done, i_rep)
        self.assertIn('if src == "D" and dst == "P":\n            self._dp_report(', src)
        self.assertLess(src.index('_dp_drain_end = self._flip_marks.get("quiesce")'),
                        src.index("self._flip_marks = {}"))
        # the quiesce mark is the moment drain(S) returned
        i_drain = src.index("if not await self.drain(S):")
        self.assertLess(i_drain, src.index('self._flip_marks["quiesce"] = time.time()'))

    def test_counters_the_hold_reads_are_written(self):
        self.assertIn('self.counters["d_admits"] += 1', inspect.getsource(F.Front._log_admit))
        self.assertIn('self.counters["min_dwell_holds"] += 1', inspect.getsource(F.Front._dwell_ok))
        flip = inspect.getsource(F.Front.flip)
        for key in ("pdflip_drain_waiting", "W1_PdFlipDrainRefused"):
            self.assertIn(f'self.counters["{key}"] += 1', flip)
            self.assertIn(key, DW.SNAP_KEYS)

    def test_instrument_only_nothing_decides_on_it(self):
        """``dp_arrival`` is read and written in the two R28 methods and nowhere
        else in the front: no route, admission or flip verdict can depend on it."""
        src = inspect.getsource(F)
        users = set()
        for name, fn in inspect.getmembers(F.Front, inspect.isfunction):
            try:
                body = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            if "dp_arrival" in body:
                users.add(name)
        self.assertEqual(users, {"_dp_mark", "_dp_report"})
        self.assertIn("dp_arrival: Optional[_dp_wait.DpArrival] = None", src)


if __name__ == "__main__":
    unittest.main()


class TheParkDwellNamesItsHold(CustomTestCase):
    """#1416i, NF z30e (ca2a9706ec) 21:31:33.522: pdflip-32-64 queued while
    ``PDFLIP PARK-IMMEDIATE-DWELL epoch=32 ... awake_ms=3502 min_dwell_ms=6050``
    held the immediate park (one line per D phase, logged before or at the
    arrival: no counter rises for it); the flip began 3.1 s later and the
    DP-WAIT line said ``hold_by=d-work``."""

    T_ARR, T_DWELL_LAST, T_FLIP0 = 33.522, 36.600, 36.636

    def test_a_dwell_hold_inside_the_hold_names_min_dwell(self):
        a = DW.take(self.T_ARR, "long", False, 5, 0, 0, _counters())
        by = DW.hold_by(a, _counters(), t_flip0=self.T_FLIP0, dwell_held_t=self.T_DWELL_LAST)
        self.assertEqual(by, "d-work+min-dwell")

    def test_a_dwell_before_the_arrival_or_none_changes_nothing(self):
        a = DW.take(self.T_ARR, "long", False, 5, 0, 0, _counters())
        self.assertEqual(DW.hold_by(a, _counters(), t_flip0=self.T_FLIP0, dwell_held_t=30.0), "d-work")
        self.assertEqual(DW.hold_by(a, _counters()), "d-work")

    def test_the_report_passes_the_fronts_last_dwell_hold(self):
        fr = _fake_front()
        p = _pending("pdflip-32-64")
        p.dp_arrival = DW.take(self.T_ARR, "long", False, 5, 0, 0, fr.counters)
        fr.queue.append(p)
        fr.t_awake = self.T_FLIP0 + 2.6
        fr._park_dwell_held_t = self.T_DWELL_LAST
        with self.assertLogs("pdflip.front", level=logging.INFO) as cm:
            F.Front._dp_report(fr, self.T_FLIP0, None)
        line = [r.getMessage() for r in cm.records if "PDFLIP DP-WAIT" in r.getMessage()][0]
        self.assertIn("hold_by=d-work+min-dwell", line)

    def test_the_dwell_hold_stamps_its_wall_time(self):
        src = inspect.getsource(F.Front)
        i = src.index("if not dwell_ok:")
        self.assertIn("self._park_dwell_held_t = time.time()", src[i:i + 400])
