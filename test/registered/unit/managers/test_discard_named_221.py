"""#221 Fix A "benannt verwerfen" (hardening, error class: silent loss of #1068
anchor / re-issue marks), analysis qwen-221-analyse-1024.md, boot 1ecda02a58.

Three places dropped a #1068 A12.2 deferral mark -- or the read that guarded
it -- without a line and without a census exit, so the #915 identity
``deferred == landed + exits`` ran 5 marks short in the 24.10. boots:

  site 1  reset_join          HiCacheController.reset(): the RESET JOIN counts what the
                              stop observed; UnifiedRadixCache._reset_full drops the
                              ongoing_prefetch records unseen.
  site 2  hold_refetch_clear  Scheduler._pdflip_refetch_one / _pdflip_refetch_budget
                              cleared the mark with no census exit.
  site 3  sleep_reset         flush_cache -> tree_cache.reset() ran no clearer over the
                              held population.

What is asserted, per site: a discarded mark gives ONE ``#221 DISCARD-NAMED site=..
rid=.. marks=.. tokens=..`` line AND exactly one census exit. THE DANGER DIRECTION
(own tests): a census exit never closes a LIVE mark -- a clear that does not take, a
landed-hold without a mark, a request that is not in the held population and a
re-issue that is not allowed all count nothing and leave the mark standing.

Drives the shipped ``Scheduler`` methods and ``HiCacheController.reset`` on
stand-ins; no GPU, no network.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m pytest \\
        test/registered/unit/managers/test_discard_named_221.py -q
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers import cache_controller as cc_module  # noqa: E402
from flliper.srt.managers import discard_named as dn  # noqa: E402
from flliper.srt.managers import scheduler as sch  # noqa: E402
from flliper.srt.managers.cache_controller import (  # noqa: E402
    HiCacheController,
    PrefetchOperation,
    StorageStopResult,
)
from flliper.srt.mem_cache import match_refusal_census as census  # noqa: E402

LOG = logging.getLogger("test_discard_named_221")


def _delta(before):
    after = census.gate_snapshot()
    keys = set(after) | set(before)
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys if after.get(k, 0) != before.get(k, 0)}


def _marked(rid="pdflip-12-170", reason="store_prefix_short", span=121593):
    return SimpleNamespace(
        rid=rid,
        prefetch_deferred=reason,
        prefetch_defer_since=time.monotonic(),
        prefetch_defer_attempts=1,
        prefetch_defer_passes=2,
        _prefetch_landed_hold_once=False,
        _prefetch_span_tokens=span,
    )


def _real_clear(obj):
    return sch.Scheduler._clear_prefetch_deferral_fields.__get__(obj)


def _lines(logs, site):
    return [m for m in logs.output if dn.TAG + " site=" + site + " rid=" in m]


def _totals(logs, site):
    return [m for m in logs.output if dn.TOTAL_TAG + " site=" + site + " " in m]


# ------------------------------------------------------------------ the module


class TestDiscardMark(unittest.TestCase):
    def test_a_standing_mark_is_named_and_counted_once(self):
        req = _marked()
        before = census.gate_snapshot()
        with self.assertLogs(LOG, level="INFO") as logs:
            ok = dn.discard_mark(req, dn.SITE_HOLD_REFETCH_CLEAR, LOG, _real_clear(SimpleNamespace()))
        self.assertTrue(ok)
        self.assertIsNone(req.prefetch_deferred)  # the mark is gone
        self.assertEqual(_delta(before), {"defer_cleared_refetch": 1})
        (line,) = _lines(logs, "hold_refetch_clear")
        self.assertIn("rid=pdflip-12-170 marks=1 tokens=121593", line)
        self.assertIn("reason=store_prefix_short attempts=1 passes=2", line)

    def test_no_mark_no_line_no_exit_but_the_fields_are_cleared_as_before(self):
        req = SimpleNamespace(rid="x", prefetch_deferred=None, prefetch_defer_since=None,
                              prefetch_defer_attempts=None, prefetch_defer_passes=None,
                              _prefetch_landed_hold_once=True)
        before = census.gate_snapshot()
        with mock.patch.object(LOG, "info") as info:
            ok = dn.discard_mark(req, dn.SITE_HOLD_REFETCH_CLEAR, LOG, _real_clear(SimpleNamespace()))
        self.assertFalse(ok)
        self.assertEqual(_delta(before), {})
        info.assert_not_called()
        self.assertFalse(req._prefetch_landed_hold_once)  # pre-#221 behaviour kept

    def test_DANGER_a_clear_that_does_not_take_closes_no_live_mark(self):
        req = _marked()
        before = census.gate_snapshot()
        with mock.patch.object(LOG, "info") as info:
            self.assertFalse(dn.discard_mark(req, dn.SITE_SLEEP_RESET, LOG, None))  # no clearer on the object
            self.assertFalse(dn.discard_mark(req, dn.SITE_SLEEP_RESET, LOG, lambda r: None))  # clearer is a no-op
        self.assertEqual(req.prefetch_deferred, "store_prefix_short")  # still live
        self.assertEqual(_delta(before), {})
        info.assert_not_called()

    def test_DANGER_a_landed_hold_alone_is_not_a_mark_exit(self):
        # LANDED already counted its exit when it set the hold: counting again is the duo0700 double exit
        req = SimpleNamespace(rid="x", prefetch_deferred=None, prefetch_defer_since=None,
                              prefetch_defer_attempts=None, prefetch_defer_passes=None,
                              _prefetch_landed_hold_once=True)
        before = census.gate_snapshot()
        rids, marks, _ = dn.discard_marks([req], dn.SITE_SLEEP_RESET, LOG, _real_clear(SimpleNamespace()))
        self.assertEqual((rids, marks), ([], 0))
        self.assertEqual(_delta(before), {})
        self.assertTrue(req._prefetch_landed_hold_once)  # not even touched at the sleep reset

    def test_switch_off_is_the_old_behaviour(self):
        req = _marked()
        before = census.gate_snapshot()
        with mock.patch.dict(os.environ, {dn.ENV: "0"}), mock.patch.object(LOG, "info") as info:
            self.assertFalse(dn.discard_mark(req, dn.SITE_HOLD_REFETCH_CLEAR, LOG, _real_clear(SimpleNamespace())))
            self.assertEqual(dn.discard_marks([_marked("y")], dn.SITE_SLEEP_RESET, LOG,
                                              _real_clear(SimpleNamespace())), ([], 0, 0))
        self.assertIsNone(req.prefetch_deferred)  # site 2 cleared as it always did
        self.assertEqual(_delta(before), {})
        info.assert_not_called()

    def test_default_is_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dn.ENV, None)
            self.assertTrue(dn.enabled())


# ------------------------------------------------------------------ site 2


def _refetch_obj(issued):
    obj = sch.Scheduler.__new__(sch.Scheduler)
    obj.tree_cache = SimpleNamespace(check_prefetch_progress=lambda rid: True,
                                     prefetch_loaded_tokens_by_reqid={}, ongoing_prefetch={})
    obj._pdflip_note_store_shortfall = lambda req: "record-short"
    obj._prefetch_kvcache = lambda req: issued.append((req.rid, req.prefetch_deferred)) or "issued"
    return obj


class TestSite2HoldRefetchClear(unittest.TestCase):
    def _req(self):
        req = _marked(rid="pdflip-12-170")
        req._1456_last = 0.0
        return req

    def test_the_hold_refetch_names_the_mark_it_clears_and_counts_the_exit(self):
        issued = []
        obj = _refetch_obj(issued)
        req = self._req()
        before = census.gate_snapshot()
        with self.assertLogs(sch.logger, level="INFO") as logs:
            verdict = sch.Scheduler._pdflip_refetch_one(obj, req, time.monotonic(), allow_reissue=True)
        self.assertEqual(verdict, "reissued")
        self.assertEqual(issued, [("pdflip-12-170", None)])  # cleared BEFORE the re-issue, as before
        self.assertEqual(_delta(before), {"defer_cleared_refetch": 1})
        (line,) = _lines(logs, "hold_refetch_clear")
        self.assertIn("rid=pdflip-12-170 marks=1 tokens=121593", line)
        self.assertIn("via=refetch", line)

    def test_a_request_without_a_mark_counts_no_exit(self):
        issued = []
        obj = _refetch_obj(issued)
        req = self._req()
        req.prefetch_deferred = None
        before = census.gate_snapshot()
        with self.assertLogs(sch.logger, level="INFO") as logs:
            sch.Scheduler._pdflip_refetch_one(obj, req, time.monotonic(), allow_reissue=True)
        self.assertEqual(_delta(before), {})
        self.assertEqual(_lines(logs, "hold_refetch_clear"), [])

    def test_DANGER_due_without_reissue_leaves_the_mark_standing_and_counts_nothing(self):
        issued = []
        obj = _refetch_obj(issued)
        req = self._req()
        before = census.gate_snapshot()
        verdict = sch.Scheduler._pdflip_refetch_one(obj, req, time.monotonic(), allow_reissue=False)
        self.assertEqual(verdict, "due")
        self.assertEqual(req.prefetch_deferred, "store_prefix_short")
        self.assertEqual(_delta(before), {})
        self.assertEqual(issued, [])

    def test_the_budget_reread_names_its_mark_too(self):
        issued = []
        obj = _refetch_obj(issued)
        obj.tree_cache.cache_controller = SimpleNamespace(prefetch_rate_limited=lambda: False)
        req = self._req()
        req._1471b_budget = True  # the last read was refused by the host budget
        req._1471b_t = 0.0
        before = census.gate_snapshot()
        with self.assertLogs(sch.logger, level="INFO") as logs:
            verdict = sch.Scheduler._pdflip_refetch_budget(obj, req, time.monotonic(), True)
        self.assertEqual(verdict, "reissued")
        self.assertEqual(_delta(before), {"defer_cleared_refetch": 1})
        self.assertIn("via=budget", _lines(logs, "hold_refetch_clear")[0])


# ------------------------------------------------------------------ site 3


class TestSite3SleepReset(unittest.TestCase):
    def _sched(self, parked=(), hold=(), settle=(), queue=()):
        s = sch.Scheduler.__new__(sch.Scheduler)
        s.pdflip_d_parked = list(parked)
        s.pdflip_dormant_hold = list(hold)
        s.pdflip_post_wake_settle = list(settle)
        s.waiting_queue = list(queue)
        return s

    def test_the_held_population_is_named_and_closed_each_with_one_exit(self):
        a, b, c = _marked("pdflip-12-168", span=100), _marked("pdflip-12-170", span=200), _marked("pdflip-8-51", span=50)
        s = self._sched(parked=[a], hold=[b], settle=[c, a])  # `a` twice: counted once
        before = census.gate_snapshot()
        with self.assertLogs(sch.logger, level="INFO") as logs:
            rids, marks, tokens = sch._discard_named_sleep_reset(s)
        self.assertEqual((sorted(rids), marks, tokens), (["pdflip-12-168", "pdflip-12-170", "pdflip-8-51"], 3, 350))
        self.assertEqual(_delta(before), {"defer_lost_at_reset": 3})
        self.assertEqual(len(_lines(logs, "sleep_reset")), 3)
        for r in (a, b, c):
            self.assertIsNone(r.prefetch_deferred)
        (total,) = _totals(logs, "sleep_reset")
        self.assertIn("n=3 marks=3 tokens=350", total)

    def test_DANGER_a_queued_request_keeps_its_live_mark(self):
        queued = _marked("pdflip-queued")
        held = _marked("pdflip-held")
        s = self._sched(parked=[held], queue=[queued])
        before = census.gate_snapshot()
        with self.assertLogs(sch.logger, level="INFO") as logs:
            rids, marks, _ = sch._discard_named_sleep_reset(s)
        self.assertEqual((rids, marks), (["pdflip-held"], 1))
        self.assertEqual(queued.prefetch_deferred, "store_prefix_short")  # retried in-pass, never closed
        self.assertEqual(_delta(before), {"defer_lost_at_reset": 1})
        self.assertEqual(len(_lines(logs, "sleep_reset")), 1)

    def test_an_empty_population_still_prints_the_aggregate_line(self):
        s = self._sched()
        with self.assertLogs(sch.logger, level="INFO") as logs:
            self.assertEqual(sch._discard_named_sleep_reset(s), ([], 0, 0))
        (total,) = _totals(logs, "sleep_reset")
        self.assertIn("n=0 marks=0 tokens=0 rids=-", total)

    def test_the_call_sits_right_after_the_tree_reset_in_flush_cache(self):
        src = open(sch.__file__).read()
        m = re.search(r"self\.tree_cache\.reset\(\)\n\s+_discard_named_sleep_reset\(self\)", src)
        self.assertIsNotNone(m, "the named clearer must follow tree_cache.reset() in flush_cache")
        i = m.start()
        j = src.index("_pdflip_flush_nonblock.sweep_memo_clear(self)", i)
        self.assertLess(i, j)
        k = src.index("def flush_cache(")
        self.assertLess(k, i)  # inside flush_cache
        # and nothing else in flush_cache resets the tree after it
        self.assertEqual(src[k:i].count("self.tree_cache.reset()"), 0)

    def test_the_hold_refetch_sites_go_through_the_named_clearer(self):
        src = open(sch.__file__).read()
        i = src.index("def _pdflip_refetch_one")
        j = src.index("def _pdflip_group_min_ints")
        self.assertEqual(src[i:j].count("_discard_named.discard_mark("), 2)
        self.assertNotIn("clear(req)  # the shortfall mark", src[i:j])


# ------------------------------------------------------------------ site 1


def _controller():
    cc = HiCacheController.__new__(HiCacheController)
    cc.storage_stop_event = threading.Event()
    cc.enable_storage = False  # no restart half
    cc.write_queue, cc.load_queue, cc.ack_write_queue, cc.ack_load_queue = [], [], [], []
    cc.prefetch_queue = Queue()
    cc.prefetch_buffer = Queue()
    cc._prefetch_current = None
    cc._prefetch_io_current = None
    cc._prefetch_io_inflight = {}
    cc.pdflip_reset_reaper_alive = lambda: False
    cc._pdflip_park_read_detach_rids = lambda: None
    cc._stop_storage_threads = lambda: StorageStopResult(["prefetch", "backup"], 0, 0, 0.0)
    return cc


def _op(rid, n):
    return PrefetchOperation(rid, torch.arange(n, dtype=torch.int64), list(range(n)))


def _record(op):
    return SimpleNamespace(operation=op, host_indices=op.host_indices)


class TestSite1ResetJoin(unittest.TestCase):
    def test_observed_and_unobserved_reads_are_named_one_line_per_rid(self):
        cc = _controller()
        in_queue, in_buffer = _op("pdflip-12-168", 4096), _op("pdflip-12-170", 2048)
        cc.prefetch_queue.put(in_queue)
        cc.prefetch_buffer.put(in_buffer)
        done = _op("pdflip-8-51", 1024)  # probed and loaded, no loop holds it any more: the join cannot see it
        # what UnifiedRadixCache._reset_full hands over: the three records at the moment of the clear
        cc._discard_named_records = dn.snapshot_records(
            {"pdflip-12-168": _record(in_queue), "pdflip-8-51": _record(done)})
        before = census.gate_snapshot()
        with self.assertLogs(cc_module.logger, level="INFO") as logs:
            cc.reset()
        l168, l170, l851 = (
            [m for m in logs.output if dn.TAG + " site=reset_join rid=" + rid + " " in m] for rid in
            ("pdflip-12-168", "pdflip-12-170", "pdflip-8-51"))
        self.assertEqual((len(l168), len(l170), len(l851)), (1, 1, 1))
        self.assertIn("marks=1 tokens=4096 observed=1", l168[0])  # record + op are ONE unit
        self.assertIn("marks=1 tokens=2048 observed=1", l170[0])
        self.assertIn("marks=1 tokens=1024 observed=0", l851[0])  # the unobserved one, now named
        (total,) = _totals(logs, "reset_join")
        self.assertIn("n=3 marks=3 tokens=7168", total)
        self.assertEqual(_delta(before), {"discard_reset_join": 3, "discard_reset_join_tokens": 7168})
        self.assertFalse(hasattr(cc, "_discard_named_records"))  # consumed, never stale

    def test_DANGER_the_controller_never_closes_a_mark(self):
        # the controller does not see the Req: only the read-unit key moves, never a term of the identity
        cc = _controller()
        op = _op("pdflip-12-170", 512)
        cc.prefetch_queue.put(op)
        before = census.gate_snapshot()
        with self.assertLogs(cc_module.logger, level="INFO"):
            cc.reset()
        moved = _delta(before)
        self.assertEqual(set(moved), {"discard_reset_join", "discard_reset_join_tokens"})
        self.assertFalse([k for k in moved if k.startswith("defer_") or k == "landed"])

    def test_nothing_discarded_still_prints_the_aggregate_line(self):
        cc = _controller()
        before = census.gate_snapshot()
        with self.assertLogs(cc_module.logger, level="INFO") as logs:
            cc.reset()
        (total,) = _totals(logs, "reset_join")
        self.assertIn("n=0 marks=0 tokens=0 rids=-", total)
        self.assertEqual(_delta(before), {})

    def test_the_merged_reset_names_the_records_it_drops_too(self):
        cc = _controller()
        cc.pdflip_reset_reaper_alive = lambda: True
        cc._discard_named_records = {id(object()): ("pdflip-6-6", 1337)}
        before = census.gate_snapshot()
        with self.assertLogs(cc_module.logger, level="INFO") as logs:
            cc.reset()
        self.assertIn("marks=1 tokens=1337 observed=0", _lines(logs, "reset_join")[0])
        self.assertEqual(_delta(before), {"discard_reset_join": 1, "discard_reset_join_tokens": 1337})

    def test_reset_full_hands_the_records_over_before_it_clears_them(self):
        from flliper.srt.mem_cache import unified_radix_cache as urc

        src = open(urc.__file__).read()
        i = src.index("self.cache_controller._discard_named_records = _discard_named.snapshot_records(")
        j = src.index("self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}")
        k = src.index("self.cache_controller.reset()", j)
        self.assertLess(src.index("def _reset_full"), i)
        self.assertLess(i, j)
        self.assertLess(j, k)

    def test_snapshot_records_keys_by_operation_so_op_and_record_are_one_unit(self):
        op = _op("r", 10)
        snap = dn.snapshot_records({"r": _record(op)})
        self.assertEqual(snap, {id(op): ("r", 10)})
        self.assertEqual(dn.snapshot_records({}), {})
        with mock.patch.dict(os.environ, {dn.ENV: "off"}):
            self.assertEqual(dn.snapshot_records({"r": _record(op)}), {})


if __name__ == "__main__":
    unittest.main()
