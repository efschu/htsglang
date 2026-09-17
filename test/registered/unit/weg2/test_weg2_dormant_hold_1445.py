# SPDX-License-Identifier: Apache-2.0
"""#1445: the #1443 dormant hold is a QUEUE, so an abort reaches it, and a
HEALTH probe is never held.

Boot weg2xsn205: the tokenizer's /health_generate probe (rid HEALTH_CHECK)
arrived while group P was dormant, was HELD by #1443, its sender's abort
found nothing (the abort only scanned ``waiting_queue``), the wake released
the orphan into P's waiting queue where it sat unscheduled, P never went
idle (flush_cache 400 x 180 over 90 s) and flip 2 died with W3
Weg2DrainWitnessDisagreement.

Two fixes, two danger directions:
* an abort landing while the request is held drops it from the hold, tells
  the tokenizer (AbortReq) and releases the hicache reservation -- prefix
  match and abort_all exactly as for ``waiting_queue``; other held requests
  stay;
* a health probe on a dormant group takes the pre-#1443 W25 refusal path
  (streamed straight out) instead of the hold -- pinned at the source, and
  the predicate is upstream's own ``is_health_check_generate_req``.

Hermetic: bound methods on a fake scheduler, no GPU.
"""
import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Tok:
    def __init__(self):
        self.sent = []

    def send_output(self, msg, req):
        self.sent.append((msg, req))


class _Tree:
    def __init__(self):
        self.released = []

    def release_aborted_request(self, rid):
        self.released.append(rid)


def _fake(hold, hicache=True):
    f = SimpleNamespace(
        weg2_dormant_hold=hold,
        enable_hicache_storage=hicache,
        tree_cache=_Tree(),
        ipc_channels=SimpleNamespace(send_to_tokenizer=_Tok()),
        waiting_queue=[],
        prefetched=[],
    )
    # #1448: the release re-enters intake; the fake's intake records the
    # prefetch and queues, like the real one does once the flag is cleared.
    def _intake(req):
        f.prefetched.append(req.rid)
        f.waiting_queue.append(req)
    f._add_request_to_queue = _intake
    return f


def _req(rid):
    return SimpleNamespace(rid=rid)


class AbortReachesTheHold(CustomTestCase):
    def test_prefix_match_drops_only_the_matching_request(self):
        hold = [_req("HEALTH_CHECK-1"), _req("weg2-0-1"), _req("weg2-0-2")]
        f = _fake(hold)
        n = Scheduler._weg2_abort_dormant_hold(f, AbortReq(rid="HEALTH_CHECK"))
        self.assertEqual(n, 1)
        self.assertEqual([r.rid for r in f.weg2_dormant_hold], ["weg2-0-1", "weg2-0-2"])
        self.assertEqual(f.tree_cache.released, ["HEALTH_CHECK-1"])
        (msg, req), = f.ipc_channels.send_to_tokenizer.sent
        self.assertIsInstance(msg, AbortReq)
        self.assertEqual(msg.rid, "HEALTH_CHECK-1")
        # and the release afterwards no longer re-queues the aborted one
        Scheduler._weg2_release_dormant_hold(f)
        self.assertEqual([r.rid for r in f.waiting_queue], ["weg2-0-1", "weg2-0-2"])
        self.assertEqual(f.prefetched, [])  # #1455: the prefetch ran at the hold, not at the wake
        self.assertEqual(f.weg2_dormant_hold, [])

    def test_abort_all_empties_the_hold(self):
        f = _fake([_req("a"), _req("b")])
        n = Scheduler._weg2_abort_dormant_hold(f, AbortReq(rid="", abort_all=True))
        self.assertEqual(n, 2)
        self.assertEqual(f.weg2_dormant_hold, [])
        self.assertEqual(len(f.ipc_channels.send_to_tokenizer.sent), 2)

    def test_no_match_touches_nothing(self):
        f = _fake([_req("a")])
        self.assertEqual(Scheduler._weg2_abort_dormant_hold(f, AbortReq(rid="zzz")), 0)
        self.assertEqual(len(f.weg2_dormant_hold), 1)
        self.assertEqual(f.ipc_channels.send_to_tokenizer.sent, [])
        self.assertEqual(f.tree_cache.released, [])

    def test_empty_or_absent_hold(self):
        self.assertEqual(Scheduler._weg2_abort_dormant_hold(_fake([]), AbortReq(rid="a")), 0)
        self.assertEqual(Scheduler._weg2_abort_dormant_hold(SimpleNamespace(), AbortReq(rid="a")), 0)

    def test_hicache_off_skips_release(self):
        f = _fake([_req("a")], hicache=False)
        Scheduler._weg2_abort_dormant_hold(f, AbortReq(rid="a"))
        self.assertEqual(f.tree_cache.released, [])
        self.assertEqual(len(f.ipc_channels.send_to_tokenizer.sent), 1)


class Wiring(CustomTestCase):
    def test_abort_path_calls_the_hold_drain(self):
        src = inspect.getsource(Scheduler._abort_request_now)
        self.assertIn("self._weg2_abort_dormant_hold(recv_req)", src)
        # after the waiting-queue loop, before the grammar queue
        self.assertLess(src.index("Abort queued request"), src.index("_weg2_abort_dormant_hold"))
        self.assertLess(src.index("_weg2_abort_dormant_hold"), src.index("grammar_manager.abort_requests"))

    def test_hold_follows_the_prefetch_and_the_wake_does_not_flush_1455(self):
        """#1455: the prefetch is issued at the hold (it runs in the executor
        during the flip) and the wake keeps it -- the wake-side flush is off
        by default, the sleep-side flush stays."""
        src = inspect.getsource(Scheduler._add_request_to_queue)
        self.assertGreater(src.index("weg2_dormant_hold"), src.index("_prefetch_kvcache(req)"))
        rel = inspect.getsource(Scheduler._weg2_release_dormant_hold)
        self.assertIn("waiting_queue.extend(released)", rel)
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        wsrc = inspect.getsource(wu)
        self.assertIn('os.environ.get("SGLANG_WEG2_WAKE_FLUSH", "0") == "1"', wsrc)
        self.assertIn("flushed = self._weg2_wake_restore_pools()", wsrc)
        body = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_wake_restore_pools)
        self.assertNotIn("sched.tree_cache.reset", body)  # only the docstring names it
        self.assertIn("req_to_token_pool.clear()", body)

    def test_health_probe_takes_w25_not_the_hold(self):
        src = inspect.getsource(Scheduler.handle_generate_request)
        cond = ("if getattr(self, \"weg2_dormant\", False) and (\n"
                "            not _weg2_dormant_admit_armed() or is_health_check_generate_req(recv_req)\n"
                "        ):\n"
                "            self._weg2_refuse_dormant(recv_req, context=\"generate\")")
        self.assertIn(cond, src)
        # upstream's own predicate, no fork-local flag
        self.assertTrue(sched_mod.is_health_check_generate_req(SimpleNamespace(rid="HEALTH_CHECK-7")))
        self.assertFalse(sched_mod.is_health_check_generate_req(SimpleNamespace(rid="weg2-0-1")))


if __name__ == "__main__":
    unittest.main()
