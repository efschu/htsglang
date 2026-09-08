# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T14): the D-side claim.

The presence probe answers KV pages ``k`` and draft pages ``d``. The claim
is ``d`` (trim, at most one HiCache chunk of re-prefill) when ``k - d`` fits
one chunk, else ``k`` with the request marked draft-cold BY NAME over
``[d, k)`` -- the #993 zero fill is the deterministic fill behind the name,
never silent. ``_draft_page_get`` stops discarding the hit count, and a rank
whose claim differs from the group's is a STOP, never a compensation.
"""

import types
import unittest

import pytest

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    Weg2DraftDisagree,
    assert_draft_claims_agree,
    resolve_draft_claim,
)
from sglang.srt.managers.phase_flip_draft_bootstrap import draft_cold_reason
from sglang.test.test_utils import CustomTestCase

CHUNK = 4096


class Stub:
    def __init__(self, **kw):
        self.draft_page_get_func = None
        self._draft_l3_hits = 0
        self._draft_l3_misses = 0
        self._draft_l3_logged_at = -1
        self._draft_get_warned_at = 0
        self.__dict__.update(kw)

    _draft_page_get = HiCacheController._draft_page_get
    _draft_page_get_flags = HiCacheController._draft_page_get_flags
    _log_draft_l3_progress = HiCacheController._log_draft_l3_progress


class TestClaimPolicy(CustomTestCase):
    def test_t14_draft_page_get_returns_the_hit_count(self):
        self.assertEqual(Stub()._draft_page_get(["a", "b"], None), -1)
        s = Stub(draft_page_get_func=lambda h, i: [True, False, True])
        self.assertEqual(s._draft_page_get(["a", "b", "c"], None), 2)

        def boom(h, i):
            raise OSError("store gone")

        self.assertEqual(Stub(draft_page_get_func=boom)._draft_page_get(["a"], None), 0)

    def test_t14_claim_full_trim_cold(self):
        no_reprobe = lambda d: d  # noqa: E731 - the anchor sits at d
        self.assertEqual(resolve_draft_claim(10, 10, CHUNK, no_reprobe), (10, 10, "full", None))
        self.assertEqual(resolve_draft_claim(10, 12, CHUNK, no_reprobe), (10, 10, "full", None))
        self.assertEqual(resolve_draft_claim(5000, 4000, CHUNK, no_reprobe), (4000, 4000, "trim", None))
        self.assertEqual(
            resolve_draft_claim(9000, 4000, CHUNK, no_reprobe), (9000, 4000, "cold", (4000, 9000))
        )
        # the trim lands on the nearest anchor at or below d; if that pushes the
        # re-prefill past one chunk the request goes cold by name instead
        self.assertEqual(resolve_draft_claim(5000, 4000, CHUNK, lambda d: 3500), (3500, 3500, "trim", None))
        self.assertEqual(
            resolve_draft_claim(5000, 4000, CHUNK, lambda d: 800), (5000, 4000, "cold", (4000, 5000))
        )
        self.assertEqual(resolve_draft_claim(10, 0, CHUNK, no_reprobe), (0, 0, "trim", None))
        self.assertEqual(resolve_draft_claim(0, 0, CHUNK, no_reprobe), (0, 0, "full", None))

    def test_t14_cold_reason_third_trigger(self):
        controller = types.SimpleNamespace(draft_cold_spans={"r1": (4000, 9000)})
        sched = types.SimpleNamespace(
            tree_cache=types.SimpleNamespace(cache_controller=controller)
        )
        req = types.SimpleNamespace(rid="r1", prefix_indices=list(range(9000)))
        reason = draft_cold_reason(sched, req, True)
        self.assertIsNotNone(reason)
        self.assertIn("5000 of 9000", reason)
        self.assertIn("[4000, 9000)", reason)
        self.assertNotIn("r1", controller.draft_cold_spans)  # consumed once
        warm = types.SimpleNamespace(rid="r2", prefix_indices=list(range(9000)))
        self.assertIsNone(draft_cold_reason(sched, warm, True))

    def test_t14_rank_disagreement_is_a_stop(self):
        assert_draft_claims_agree(7, 7, "rid")
        with pytest.raises(Weg2DraftDisagree, match="WEG2 DRAFT-DISAGREE STOP"):
            assert_draft_claims_agree(7, 9, "rid")


class TestDisagreeIsAGroupStop(CustomTestCase):
    """S5: a rank disagreement inside the prefetch DAEMON thread must end
    the group, not the thread. At 35bdc9e310 `Weg2DraftDisagree` escaped
    `prefetch_thread_func` (no except, only finally), the thread died and
    the process kept serving with a dead prefetch path -- a wedge dressed
    as a STOP. The fork's crash path is SIGQUIT to the parent (the
    scheduler's own `run_scheduler_process` except -> kill_process_tree),
    then the signal on this process."""

    def test_fix2_disagreement_in_the_prefetch_thread_reaches_the_kill(self):
        import os
        import signal
        import threading
        from queue import Queue
        from unittest import mock

        from sglang.srt.managers import cache_controller as cc

        class Op:
            request_id = "rid-1"
            host_indices = []

            def mark_terminate(self):
                pass

        stop = threading.Event()
        q = Queue()
        q.put(Op())
        stop.set()

        def all_reduce(t, op):
            # the MIN over [claim, -claim]: rank 0 claimed 5, another rank 7
            t[0] = 5
            t[1] = -7

        stub = Stub(
            prefetch_queue=q,
            storage_stop_event=stop,
            prefetch_io_aux_func=lambda: None,
            _storage_hit_query=lambda op: (["h"] * 5, 5),
            draft_tier_armed=lambda scope: True,
            _all_reduce_prefetch_groups=all_reduce,
            draft_cold_spans={},
            _prefetch_drained_after_stop=0,
            prefetch_threshold=1,
            page_size=1,
            prefetch_revoke_queue=Queue(),
            append_host_mem_release=lambda *a, **k: None,
        )
        stub.prefetch_thread_func = HiCacheController.prefetch_thread_func.__get__(stub)
        stub._stop_group_from_thread = getattr(HiCacheController, "_stop_group_from_thread", None)
        if stub._stop_group_from_thread is not None:
            stub._stop_group_from_thread = stub._stop_group_from_thread.__get__(stub)
        parent = mock.MagicMock()

        class FakeProc:  # MagicMock reserves the `parent` kwarg for itself
            def parent(self):
                return parent

        with mock.patch.object(cc.os, "kill") as kill, \
                mock.patch.object(cc.psutil, "Process", return_value=FakeProc()):
            with self.assertLogs(cc.logger, level="ERROR") as logs:
                stub.prefetch_thread_func()  # returns; never raises out of the thread
        self.assertTrue(any("WEG2 DRAFT-DISAGREE STOP" in line for line in logs.output), logs.output)
        parent.send_signal.assert_called_once_with(signal.SIGQUIT)
        kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)


class TestAdmissionLines(CustomTestCase):
    """C14 L6/L7: one line per request at admission, with the span the
    zeros fill, or the warm page count -- the aggregate ADMISSION line
    (one 'First reason' for N requests) stays as the denominator."""

    def _scheduler(self, spans):
        import torch

        return types.SimpleNamespace(
            draft_worker=object(),
            tree_cache=types.SimpleNamespace(
                cache_controller=types.SimpleNamespace(
                    draft_tier_armed=lambda scope: True, draft_cold_spans=spans
                )
            ),
            req_to_token_pool=types.SimpleNamespace(req_to_token=torch.zeros(4, 16, dtype=torch.int64)),
        )

    def test_fix2_l6_l7_one_line_per_request_once(self):
        from unittest import mock

        from sglang.srt.managers import phase_flip_draft_bootstrap as pfdb

        sched = self._scheduler({"cold": (4000, 9000)})
        cold = types.SimpleNamespace(rid="cold", prefix_indices=list(range(9000)), req_pool_idx=0)
        warm = types.SimpleNamespace(rid="warm", prefix_indices=list(range(9000)), req_pool_idx=1)
        fresh = types.SimpleNamespace(rid="fresh", prefix_indices=[], req_pool_idx=2)
        batch = types.SimpleNamespace(reqs=[cold, warm, fresh])
        with mock.patch.object(pfdb, "draft_kv_pool", lambda dw: object()):
            with self.assertLogs(pfdb.logger, level="INFO") as logs:
                out = pfdb.arm_draft_cold_for_admission(sched, batch)
            self.assertEqual(out["cold"], 1)
            text = "\n".join(logs.output)
            self.assertIn(
                "WEG2 DRAFT-COLD rid=cold span=[4000,9000) of 9000 prefix pages -- cold by name, "
                "zeros are the fill (#993), rounds_owed=1",
                text,
            )
            self.assertIn("WEG2 DRAFT-WARM rid=warm pages=9000 miss=0", text)
            self.assertNotIn("rid=fresh", text)
            self.assertIn("ADMISSION draft-cold: 1 request(s) marked", text)  # the denominator stays
            # a chunked prefill's later visit prints neither line again
            with self.assertLogs(pfdb.logger, level="DEBUG") as logs2:
                pfdb.logger.debug("visit 2")
                pfdb.arm_draft_cold_for_admission(sched, batch)
            text2 = "\n".join(logs2.output)
            self.assertNotIn("DRAFT-COLD rid=", text2)
            self.assertNotIn("DRAFT-WARM rid=", text2)

    def test_fix2_l6_whole_prefix_span_for_the_disarmed_and_seam_reasons(self):
        from unittest import mock

        from sglang.srt.managers import phase_flip_draft_bootstrap as pfdb

        sched = self._scheduler({})
        sched.tree_cache.cache_controller.draft_tier_armed = lambda scope: False
        req = types.SimpleNamespace(rid="r", prefix_indices=list(range(12)), req_pool_idx=0)
        batch = types.SimpleNamespace(reqs=[req])
        with mock.patch.object(pfdb, "draft_kv_pool", lambda dw: object()), \
                mock.patch.object(pfdb, "scrub_draft_kv", lambda pool, rows: (12, [0])):
            with self.assertLogs(pfdb.logger, level="INFO") as logs:
                pfdb.arm_draft_cold_for_admission(sched, batch)
        self.assertIn("WEG2 DRAFT-COLD rid=r span=[0,12) of 12 prefix pages", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
