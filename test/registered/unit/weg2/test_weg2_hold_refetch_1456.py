# SPDX-License-Identifier: Apache-2.0
"""#1456: the dormant hold finishes and tops up its reads while the group
sleeps, and the orphan collector leaves held requests alone.

Boot weg2xsn211 (#1455): the reads issued at the hold terminated short
(69631 of 99570 pages -- P's write-through had not landed), the drain's
orphan collector then closed them, and the missing 30 % loaded only after
the wake: 5 s of the 8 s from prefill end to the first joint decode.
Hermetic: fake scheduler, bound methods.
"""
import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _fake(hold, progress, shortfall):
    f = SimpleNamespace(weg2_dormant=True, weg2_dormant_hold=hold, issued=[], cleared=[])
    f.tree_cache = SimpleNamespace(check_prefetch_progress=lambda rid: progress.get(rid, True))
    f._weg2_note_store_shortfall = lambda req: shortfall.get(req.rid)
    f._prefetch_kvcache = lambda req, rematch=True, limit_tokens=None: (f.issued.append(req.rid), "issued")[1]
    f._clear_prefetch_deferral_fields = lambda req: f.cleared.append(req.rid)
    return f


class HoldRefetch(CustomTestCase):
    def test_short_reads_are_reissued_complete_ones_not(self):
        a, b, c = SimpleNamespace(rid="a"), SimpleNamespace(rid="b"), SimpleNamespace(rid="c")
        f = _fake([a, b, c], progress={"c": False}, shortfall={"a": "store_prefix_short", "b": None})
        self.assertEqual(Scheduler._weg2_hold_refetch(f), 1)
        self.assertEqual(f.issued, ["a"])          # b complete, c still reading
        self.assertEqual(a._1456_n, 1)
        # rate limit: the same request is not re-issued within 2 s
        self.assertEqual(Scheduler._weg2_hold_refetch(f), 0)
        a._1456_last = 0.0
        self.assertEqual(Scheduler._weg2_hold_refetch(f), 1)
        self.assertEqual(a._1456_n, 2)

    def test_not_dormant_or_empty_hold_does_nothing(self):
        f = _fake([SimpleNamespace(rid="a")], {}, {"a": "store_prefix_short"})
        f.weg2_dormant = False
        self.assertEqual(Scheduler._weg2_hold_refetch(f), 0)
        f = _fake([], {}, {})
        f.weg2_dormant = True
        self.assertEqual(Scheduler._weg2_hold_refetch(f), 0)

    def test_a_failing_top_up_never_raises(self):
        a = SimpleNamespace(rid="a")
        f = _fake([a], {}, {"a": "store_prefix_short"})
        f._prefetch_kvcache = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
        self.assertEqual(Scheduler._weg2_hold_refetch(f), 0)


class Wiring(CustomTestCase):
    def test_hold_time_claims_adopt_the_group_min_1461(self):
        """boot weg2xsn216: TP2 probed 94207 while TP0/TP1 read 97870 (P still
        writing the tail) -> DRAFT-DISAGREE STOP.  For rids in the hold the
        controller adopts the MIN; outside the hold the law stays STOP."""
        from sglang.srt.managers import cache_controller as cc
        src = inspect.getsource(cc)
        self.assertIn('operation.request_id in getattr(self, "weg2_hold_rids", ())', src)
        self.assertIn("#1461 DRAFT-CLAIM MIN-ADOPTED", src)
        self.assertIn("assert_draft_claims_agree(_mn, _mx, operation.request_id)", src)
        with self.assertRaises(cc.Weg2DraftDisagree):
            cc.assert_draft_claims_agree(94207, 97870, "x")
        sched = inspect.getsource(Scheduler._add_request_to_queue)
        self.assertIn("_cc.weg2_hold_rids.add(req.rid)", sched)
        self.assertIn(".discard(_r.rid)", inspect.getsource(Scheduler._weg2_release_dormant_hold))
        self.assertIn(".discard(_r.rid)  # #1461", inspect.getsource(Scheduler._weg2_abort_dormant_hold))

    def test_orphan_collector_spares_the_hold_and_idle_calls_the_top_up(self):
        src = inspect.getsource(Scheduler)
        self.assertIn("if r not in verdicts and str(r) not in _held", src)
        idle = inspect.getsource(Scheduler.on_idle)
        self.assertIn("self._weg2_hold_refetch()", idle)


if __name__ == "__main__":
    unittest.main()
