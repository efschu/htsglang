"""#1479 (weg2xsn241): a hold re-read revoked below the prefetch threshold
left nothing in flight and a short record; the refetch state machine must
re-issue it, not report "reading" until the settle bound lapses -- and the
hold loop must drain the revoke queue so the revoked read leaves
``ongoing_prefetch`` in the sleep."""
import functools
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler import Scheduler


def _holder(ongoing, record, shortfall=None):
    h = types.SimpleNamespace()
    h.tree_cache = types.SimpleNamespace(
        ongoing_prefetch=ongoing,
        prefetch_loaded_tokens_by_reqid=record,
        check_prefetch_progress=lambda rid: rid not in ongoing,
    )
    h._weg2_note_store_shortfall = lambda req: shortfall
    h.issued = []
    h._prefetch_kvcache = lambda req: h.issued.append(req.rid) or "issued"
    h._clear_prefetch_deferral_fields = lambda req: None
    return h


def _req(rid="a", span=97871, short=True):
    return types.SimpleNamespace(rid=rid, _prefetch_span_tokens=span, _1471_short=short, _1456_last=0.0)


class Test1479(unittest.TestCase):
    def test_revoked_short_record_reissues(self):
        h = _holder(ongoing={}, record={"a": 4095})
        r = _req()
        self.assertEqual(Scheduler._weg2_refetch_one(h, r, 100.0), "reissued")
        self.assertEqual(h.issued, ["a"])
        self.assertEqual(r._1456_n, 1)
        self.assertTrue(r._1471_short)

    def test_in_flight_short_record_still_reading(self):
        h = _holder(ongoing={"a": object()}, record={"a": 4095})
        h.tree_cache.check_prefetch_progress = lambda rid: True  # terminated but not popped yet
        self.assertEqual(Scheduler._weg2_refetch_one(h, _req(), 100.0), "reading")
        self.assertEqual(h.issued, [])

    def test_record_covering_span_completes(self):
        h = _holder(ongoing={}, record={"a": 97871})
        r = _req()
        self.assertEqual(Scheduler._weg2_refetch_one(h, r, 100.0), "complete")
        self.assertFalse(r._1471_short)

    def test_reissue_respects_two_second_wait(self):
        h = _holder(ongoing={}, record={"a": 4095})
        r = _req(); r._1456_last = 99.5
        self.assertEqual(Scheduler._weg2_refetch_one(h, r, 100.0), "wait")
        self.assertEqual(h.issued, [])

    def test_hold_loop_drains_group_min_revokes(self):
        # #1479b: the count drained is the group MIN of the local queue sizes,
        # never an unbounded rank-local pop.
        calls = []
        h = _holder(ongoing={}, record={"a": 4095})
        h.tree_cache._drain_storage_control_queues_impl = lambda **kw: calls.append(kw)
        h.tree_cache.cache_controller = types.SimpleNamespace(prefetch_revoke_queue=types.SimpleNamespace(qsize=lambda: 3))
        h._weg2_group_min_ints = lambda vals: [min(v, 2) for v in vals]  # a peer holds only 2
        h.weg2_dormant_hold = [_req()]
        h.weg2_dormant = True
        self.assertEqual(Scheduler._weg2_hold_refetch(h), 1)
        self.assertEqual(calls, [dict(n_revoke=2, n_backup=0, n_release=0, extra_release_counts=None, log_metrics=False)])

    def test_hold_loop_skips_drain_when_group_min_is_zero(self):
        calls = []
        h = _holder(ongoing={}, record={"a": 4095})
        h.tree_cache._drain_storage_control_queues_impl = lambda **kw: calls.append(kw)
        h.tree_cache.cache_controller = types.SimpleNamespace(prefetch_revoke_queue=types.SimpleNamespace(qsize=lambda: 1))
        h._weg2_group_min_ints = lambda vals: [0 for _ in vals]  # a peer has nothing yet
        h.weg2_dormant_hold = [_req()]
        h.weg2_dormant = True
        Scheduler._weg2_hold_refetch(h)
        self.assertEqual(calls, [])

    def test_group_min_ints_local_without_group(self):
        h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=1), tp_cpu_group=None)
        self.assertEqual(Scheduler._weg2_group_min_ints(h, [3, 0]), [3, 0])

    def test_zero_answer_first_read_reissues_1478(self):
        # #1478: a first read that materialized nothing (deliverable=0 -> no
        # #1324 shortfall, _1471_short never set) must be re-issued, not
        # reported complete with nothing on the host.
        class _Rec(int):
            materialized = 0
        h = _holder(ongoing={}, record={"a": _Rec(0)})
        r = _req(short=False)
        self.assertEqual(Scheduler._weg2_refetch_one(h, r, 100.0), "reissued")
        self.assertEqual(h.issued, ["a"])
        self.assertTrue(r._1471_short)

    def test_zero_answer_in_flight_still_reading(self):
        class _Rec(int):
            materialized = 0
        h = _holder(ongoing={"a": object()}, record={"a": _Rec(0)})
        h.tree_cache.check_prefetch_progress = lambda rid: True
        self.assertEqual(Scheduler._weg2_refetch_one(h, _req(short=False), 100.0), "reading")

    def test_drain_is_fail_soft_without_impl(self):
        h = _holder(ongoing={}, record={})
        Scheduler._weg2_drain_prefetch_revokes(h)  # no impl attribute -> no error


if __name__ == "__main__":
    unittest.main()
