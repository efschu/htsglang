# SPDX-License-Identifier: Apache-2.0
"""#243 front seam (NF rc12r dkrnfh91dprbar1dauer09271632, weg2-12-39 / weg2-13-44).

P prefilled weg2-12-39 whole (76602 tokens) and published its hand-off; the rid then waited 255 s for
a D seat, a flip's P reset gave the pages back, and at the D admission the fetch lost 1196 pages and the
X gate refused it mid-stream. NF's handoff_pending names such a loss (status(rid) state=lost,
first_lost_page, page_size); the front reads it for every rid waiting for a D seat (controller sweep,
admitter before and after the seat) and sends a loss over X back to P at once
(``WEG2 HANDOFF-LOST-REROUTE ... path=fresh-P``), and drops the rid's marks at every rid end.
"""

import asyncio
import collections
import os
import sys
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2 import handoff_seam as HS  # noqa: E402

_FAKE = "_fake_handoff_pending_243"


class _Fake:
    """Installs a fake NF module under a private name and points the seam at it."""

    def __init__(self, status=None, raise_status=False, absent=False):
        self.calls, self.drops = [], []
        self._status = status or {}
        self._raise = raise_status
        self._absent = absent

    def __enter__(self):
        self._old = HS.MODULE
        HS.MODULE = _FAKE if not self._absent else "_no_such_module_243_xyz"
        HS.reset_module_cache()
        if not self._absent:
            m = types.ModuleType(_FAKE)

            def status(rid):
                self.calls.append(rid)
                if self._raise:
                    raise RuntimeError("boom")
                return self._status.get(rid, {"state": "none"})

            def drop(rid, reason):
                self.drops.append((rid, reason))

            m.status, m.drop = status, drop
            sys.modules[_FAKE] = m
        return self

    def __exit__(self, *a):
        sys.modules.pop(_FAKE, None)
        HS.MODULE = self._old
        HS.reset_module_cache()


def _front():
    f = object.__new__(F.Front)
    f.tp_prefill_max_tokens = 12288
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f.kicks, f.marks, f.syncs = [], [], []
    f._kick_controller = lambda why: f.kicks.append(why)
    f._dp_mark = lambda p, kind: f.marks.append((p.rid, kind))
    f._sync_batch_gate = lambda: f.syncs.append(1)
    return f


_LOOP = asyncio.new_event_loop()


def _p(rid="weg2-12-39", prompt=76602, **kw):
    p = F.Pending(rid, "/v1/messages", {}, "text", 0.0, _LOOP.create_future(),
                  est_prompt=prompt, est_uncached=186, span_known=True, **kw)
    p.leg1_done = True
    p.leg1_prompt_tokens = prompt
    return p


LOST_OVER = {"state": "lost", "first_lost_page": 10, "pages": 1197, "page_size": 64}   # credit 640
LOST_UNDER = {"state": "lost", "first_lost_page": 1100, "pages": 1197, "page_size": 64}  # credit 70400


class Check(unittest.TestCase):
    def test_pending_is_left_alone(self):
        f, p = _front(), _p()
        with _Fake({"weg2-12-39": {"state": "pending", "pages": 1197, "page_size": 64}}) as fk:
            self.assertFalse(f._hl_check(p, "sweep"))
        self.assertEqual((fk.calls, fk.drops, list(f.queue), p.leg1_done), (["weg2-12-39"], [], [], True))

    def test_lost_over_x_goes_fresh_to_p(self):
        f, p = _front(), _p()
        f.queue.append(_p("weg2-99-1"))
        with _Fake({"weg2-12-39": LOST_OVER}) as fk, self.assertLogs(F.logger, level="WARNING") as cap:
            self.assertTrue(f._hl_check(p, "admit"))
        self.assertEqual(fk.drops, [("weg2-12-39", "reroute_fresh")])
        self.assertIs(f.queue[0], p)  # the oldest: back at the head
        self.assertEqual((p.leg1_done, p.d_direct, p.d_eligible, p.seat, p.handoff_lost_reroutes,
                          p.est_uncached), (False, False, False, None, 1, 76602 - 640))
        self.assertIn("WEG2 HANDOFF-LOST-REROUTE rid=weg2-12-39 first_lost_page=10 credit=640 path=fresh-P",
                      cap.output[0])
        self.assertEqual((f.marks, f.kicks, f.counters["handoff_lost_reroutes"]),
                         ([("weg2-12-39", "reroute")], ["arrival"], 1))

    def test_lost_under_x_stays_for_d_and_is_named_once(self):
        f, p = _front(), _p()
        with _Fake({"weg2-12-39": LOST_UNDER}) as fk, self.assertLogs(F.logger, level="WARNING") as cap:
            self.assertFalse(f._hl_check(p, "sweep"))
            self.assertFalse(f._hl_check(p, "admit"))
        self.assertEqual((fk.drops, list(f.queue), p.leg1_done), ([], [], True))
        self.assertEqual(len([l for l in cap.output if "HANDOFF-LOST-KEPT" in l]), 1)
        self.assertIn("uncached=6202 X=12288", cap.output[0])

    def test_second_loss_over_x_ends_named_never_sent_to_d(self):
        f, p = _front(), _p()
        p.handoff_lost_reroutes = 1
        with _Fake({"weg2-12-39": LOST_OVER}) as fk, self.assertLogs(F.logger, level="ERROR") as cap:
            self.assertTrue(f._hl_check(p, "admit"))  # the caller takes it out of D's line
        self.assertEqual((fk.drops, list(f.queue)), ([("weg2-12-39", "terminal_w35")], []))
        self.assertIsInstance(p.fut.exception(), HS.Weg2HandoffLost)
        self.assertIn("W35 Weg2XReQueueLoop rid=weg2-12-39", str(p.fut.exception()))
        self.assertIn("WEG2 HANDOFF-LOST-TERMINAL rid=weg2-12-39", cap.output[0])
        self.assertEqual((f.counters["W35_Weg2XReQueueLoop"], f.counters["handoff_lost_terminal"]), (1, 1))

    def test_second_loss_under_x_stays_kept(self):
        f, p = _front(), _p()
        p.handoff_lost_reroutes = 1
        with _Fake({"weg2-12-39": LOST_UNDER}) as fk:
            self.assertFalse(f._hl_check(p, "admit"))
        self.assertEqual((fk.drops, list(f.queue), p.fut.done()), ([], [], False))

    def test_sweep_takes_a_terminal_rid_out_of_ds_line(self):
        f = _front()
        a, b = _p("weg2-1-1"), _p("weg2-12-39")
        b.handoff_lost_reroutes = 1
        f._ready_for_d.extend([a, b])
        with _Fake({"weg2-12-39": LOST_OVER}):
            self.assertEqual(f._hl_sweep(), 1)
        self.assertEqual((list(f._ready_for_d), list(f.queue)), ([a], []))
        self.assertTrue(b.fut.done())

    def test_handle_generate_answers_a_failed_future_with_503(self):
        src = open(F.__file__).read()
        i = src.index("            await fut  # leg 1 done and D awake")
        blk = src[i:i + 400]
        self.assertIn("except Exception as e:", blk)
        self.assertIn("status=503", blk)

    def test_missing_module_reads_none(self):
        f, p = _front(), _p()
        with _Fake(absent=True):
            self.assertEqual(HS.status("weg2-12-39")["state"], "none")
            self.assertFalse(f._hl_check(p, "admit"))
            HS.drop("weg2-12-39", "served")  # silent
        self.assertEqual(list(f.queue), [])

    def test_raising_status_reads_none(self):
        f, p = _front(), _p()
        with _Fake(raise_status=True):
            self.assertFalse(f._hl_check(p, "admit"))
        self.assertEqual(list(f.queue), [])

    def test_switch_off_reads_nothing(self):
        f, p = _front(), _p()
        with _Fake({"weg2-12-39": LOST_OVER}) as fk:
            os.environ["SGLANG_WEG2_HANDOFF_LOST_REROUTE"] = "0"
            try:
                self.assertFalse(f._hl_check(p, "admit"))
                self.assertEqual(f._hl_sweep(), 0)
            finally:
                os.environ.pop("SGLANG_WEG2_HANDOFF_LOST_REROUTE")
        self.assertEqual((fk.calls, list(f.queue)), ([], []))

    def test_no_leg1_no_check(self):
        f = _front()
        cold, kept, direct = _p("a"), _p("b", skip_leg1=True), _p("c")
        cold.leg1_done = False
        direct.d_direct = True
        with _Fake({r: LOST_OVER for r in "abc"}) as fk:
            for p in (cold, kept, direct):
                self.assertFalse(f._hl_check(p, "sweep"))
        self.assertEqual(fk.calls, [])

    def test_no_page_size_credits_nothing(self):
        self.assertEqual(HS.lost_terms({"state": "lost", "first_lost_page": 10, "page_size": 0}, 5000, 4096),
                         (0, 5000, True))
        self.assertIsNone(HS.lost_terms({"state": "pending"}, 5000, 4096))


class Sweep(unittest.TestCase):
    def test_sweep_moves_only_the_lost_one_and_keeps_the_order(self):
        f = _front()
        a, b, c = _p("weg2-1-1"), _p("weg2-12-39"), _p("weg2-1-3")
        f._ready_for_d.extend([a, b, c])
        with _Fake({"weg2-12-39": LOST_OVER, "weg2-1-1": {"state": "pending"}}):
            self.assertEqual(f._hl_sweep(), 1)
        self.assertEqual(list(f._ready_for_d), [a, c])
        self.assertEqual(list(f.queue), [b])
        self.assertEqual(f.syncs, [1])


class RidEnd(unittest.TestCase):
    def _run(self, handler, rid="weg2-12-39"):
        req = {}
        if rid:
            HS.note_request_rid(req, rid)
        return _LOOP.run_until_complete(HS.wrap_handler(handler)(req))

    def test_drop_at_every_end(self):
        async def ok(r):
            return types.SimpleNamespace(status=200)

        async def w35(r):
            return types.SimpleNamespace(status=503)

        async def boom(r):
            raise ValueError("x")

        async def gone(r):
            raise asyncio.CancelledError()

        with _Fake() as fk:
            self._run(ok)
            self._run(w35)
            with self.assertRaises(ValueError):
                self._run(boom)
            with self.assertRaises(asyncio.CancelledError):
                self._run(gone)
            self._run(ok, rid=None)
        self.assertEqual(fk.drops, [("weg2-12-39", "served"), ("weg2-12-39", "status_503"),
                                    ("weg2-12-39", "abort:ValueError"), ("weg2-12-39", "disconnect")])


class Wiring(unittest.TestCase):
    def setUp(self):
        self.src = open(F.__file__).read()

    def test_switch(self):
        self.assertTrue(HS.enabled({}))
        self.assertFalse(HS.enabled({"SGLANG_WEG2_HANDOFF_LOST_REROUTE": "0"}))

    def test_admitter_checks_before_and_after_the_seat(self):
        i = self.src.index("    async def d_admitter(self)")
        blk = self.src[i:i + 9000]
        acq = blk.index("await self._d_seat.acquire()")
        self.assertLess(blk.index('if self._hl_check(p, "admit"):'), acq)
        after = blk.index('if self._hl_check(p, "seat"):')
        self.assertLess(acq, after)
        self.assertLess(after, blk.index("p.fut.set_result(True)"))
        self.assertIn("self._d_seat.release()", blk[after:after + 400])

    def test_controller_sweeps_every_pass(self):
        # the loop head, not a fixed prefix: RO (p_read_overlap) added its
        # setup above `while True:` and pushed the pins past 1500 chars
        i = self.src.index("    async def controller(self)")
        j = self.src.index("        while True:", i)
        blk = self.src[j:j + 1500]
        self.assertLess(blk.index("self._hl_sweep()"), blk.index("self._rvp_take()"))

    def test_the_rid_end_wrapper_is_registered_and_the_rid_noted(self):
        i = self.src.index("front.handle_generate = _hs.wrap_handler(front.handle_generate)")
        self.assertLess(i, self.src.index("app.router.add_post(path, front.handle_generate)"))
        i = self.src.index('rid = f"weg2-{self.epoch}-{self._rid}"')
        self.assertIn("_hs.note_request_rid(request, rid)", self.src[i:i + 200])


if __name__ == "__main__":
    unittest.main()
