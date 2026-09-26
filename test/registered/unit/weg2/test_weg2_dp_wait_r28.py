# SPDX-License-Identifier: Apache-2.0
"""R28 (release table row 28): one ``WEG2 DP-WAIT`` line per D->P waiter.

THE GAP. Under agent load a request that must go over P and arrives while D
is awake waits for a D->P flip. The front log carried the pieces of that wait
on five line shapes and no line per request, so "why did weg2-4-5 wait 103 s"
took a join over ``BATCH queued (awake=D``, ``WEG2-FAIRNESS``, ``WEG2-FLIP
begin``, ``WEG2-FLIP-TIMELINE`` and ``WEG2-FLIP done``. Measured, rc9i
(dkrnfbar1agent09252237, front.log):

* :536 22:47:14.655 weg2-4-5 queued (awake=D), D decoding weg2-1-4;
* :560 22:47:58.428 the admitter hands weg2-0-2 to D (1.3 s before the bound);
* :562 22:47:59.745 WEG2-FAIRNESS fired (45.1 s), :565 WEG2-FLIP begin
  outstanding=1;
* :613 quiesce@55882 ms -- the flip's drain waited for weg2-0-2 (57.2 s decode,
  :580);
* :614 22:48:58.132 WEG2-FLIP done flip_total=58387 ms -> P awake after 103.5 s.

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

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.weg2 import dp_wait as DW  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# rc9i front.log wall clock, seconds after 22:47:00
T_Q_45 = 14.655      # :536 weg2-4-5 BATCH queued (awake=D)
T_Q_46 = 61.146      # :570 weg2-4-6 queued inside the open flip (22:48:01.146)
T_BEGIN = 59.745     # :565 WEG2-FLIP begin epoch=4 sleep=D wake=P outstanding=1
T_DRAIN = T_BEGIN + 55.882   # :613 quiesce@55882
T_DONE = 118.132     # :614 WEG2-FLIP done (22:48:58.132)


def _counters(**kw):
    c = collections.Counter()
    c.update(kw)
    return c


class TheDecomposition(CustomTestCase):

    def test_rc9i_weg2_4_5_fairness_then_drain(self):
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

    def test_rc9i_weg2_4_6_arrived_inside_the_open_flip(self):
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
                    _counters(min_dwell_holds=3, weg2_drain_waiting=1, W1_Weg2DrainRefused=0))
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=5, weg2_drain_waiting=1)),
                         "min-dwell")
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=3, weg2_drain_waiting=2)),
                         "flip-returned")
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=3, weg2_drain_waiting=1,
                                                 W1_Weg2DrainRefused=1)),
                         "flip-returned")
        self.assertEqual(DW.hold_by(a, _counters(min_dwell_holds=3, weg2_drain_waiting=1, d_admits=1)),
                         "d-work")

    def test_the_line_carries_every_term(self):
        a = DW.take(T_Q_45, "long", False, 1, 0, 1, _counters(d_admits=5))
        dec = DW.decompose(a, T_BEGIN, T_DRAIN, T_DONE)
        s = DW.line("weg2-4-5", 5, a, dec, "d-work+fairness", _counters(d_admits=6),
                    238321, 238321, 0, False)
        for frag in ("WEG2 DP-WAIT rid=weg2-4-5 epoch=5 wait_s=103.5 hold_s=45.1 drain_s=55.9 "
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


def _pending(rid="weg2-4-5"):
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
        p, q = _pending("weg2-4-5"), _pending("weg2-4-9")
        p.dp_arrival = DW.take(T_Q_45, "long", False, 1, 0, 1, fr.counters)
        fr.queue.extend([p, q])          # q: queued while P was awake, no snapshot
        fr.t_awake = T_DONE
        with self.assertLogs("weg2.front", level=logging.INFO) as cm:
            F.Front._dp_report(fr, T_BEGIN, T_DRAIN)
        lines = [r.getMessage() for r in cm.records if "WEG2 DP-WAIT" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        self.assertIn("rid=weg2-4-5", lines[0])
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
        i_done = src.index('logger.info("WEG2-FLIP done epoch=')
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
        for key in ("weg2_drain_waiting", "W1_Weg2DrainRefused"):
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
