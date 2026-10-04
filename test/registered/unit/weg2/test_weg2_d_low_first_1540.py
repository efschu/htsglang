# SPDX-License-Identifier: Apache-2.0
"""#1540 D-LOW-FIRST placement + D-LIVE-FLOOR instrument (dual layout, group D).

Wedge analysis deskq/done/1540: D gives P's card its bytes back only through the D-KV shrink, and the
shrink stops at the highest id any request holds (``max_live_id``). D's allocator hands ids out in
free-list order (token allocator: freed ids go to the TAIL; paged: to the head, unsorted), so one
running request can sit near the top of the mapped span for its whole life (b9i weg2-0-50 page 229371
of 229376: 1.4 GB held for 62 s).

DANGER DIRECTIONS guarded here:
* default (SGLANG_WEG2_DUAL_D_LOW_FIRST=0): allocation is byte-for-byte what it was (no attribute, no sort);
* on: the next allocation after frees takes the LOWEST free ids, so ``max_live_id`` falls;
* rank agreement: the trigger is the count of allocation calls (wall time is never read) and the sort
  is a pure function of the list -- two allocators fed the same calls hold the same list;
* the owner bias / weighted placement own the order when set: low-first stays out of their way;
* only the token and paged allocator are armed; any other class is refused loudly;
* the instrument never raises into the D tick and never changes the verdict.
"""
from __future__ import annotations

import os
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.weg2 import dual_d_kv_stage as D
from sglang.srt.weg2 import dual_p_kv_stage as P
from sglang.srt.weg2 import dual_pkvwait_instr as PI
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _tok(size=64, need_sort=False):
    return TokenToKVPoolAllocator(size, torch.int64, "cpu", None, need_sort)


def _paged(size=64, page=4, need_sort=False):
    return PagedTokenToKVPoolAllocator(size, page, torch.int64, "cpu", None, need_sort)


def _fragment(a):
    """4 requests of 8 ids each, the LOW ones finish: ids 1..16 are freed, 17..32 stay live."""
    reqs = [a.alloc(8) for _ in range(4)]
    a.free(reqs[0])
    a.free(reqs[1])
    return reqs


class TestTokenAllocatorLowFirst(CustomTestCase):
    def test_default_off_is_untouched(self):
        a, b = _tok(), _tok()
        self.assertFalse(hasattr(a, "_weg2_low_first_every"))
        ra, rb = _fragment(a), _fragment(b)
        # off: the freed low ids went to the TAIL, the next allocation takes the pristine HEAD (high ids)
        got = a.alloc(8)
        self.assertEqual(got.tolist(), list(range(33, 41)))
        self.assertEqual(b.alloc(8).tolist(), got.tolist())
        self.assertEqual(a.free_pages.tolist(), b.free_pages.tolist())
        self.assertFalse(hasattr(a, "_weg2_low_first_sorts"))

    def test_on_new_work_lands_low(self):
        a = _tok()
        a.set_low_first(1)
        _fragment(a)
        got = a.alloc(8)
        self.assertEqual(got.tolist(), list(range(1, 9)))          # the freed low ids, not 33..40
        self.assertEqual(a._weg2_low_first_sorts, 5)               # N=1: one sort per allocation call (4 + this one)

    def test_max_live_id_falls(self):
        off, on = _tok(), _tok()
        on.set_low_first(1)
        for a in (off, on):
            _fragment(a)
            a.alloc(8)
        self.assertEqual(P.max_live_id(off, 1), 40)
        self.assertEqual(P.max_live_id(on, 1), 32)                 # new work reused the freed low ids

    def test_need_sort_merges_released_ids(self):
        a = _tok(need_sort=True)
        a.set_low_first(1)
        _fragment(a)
        self.assertGreater(len(a.release_pages), 0)
        got = a.alloc(8)
        self.assertEqual(got.tolist(), list(range(1, 9)))
        self.assertEqual(len(a.release_pages), 0)

    def test_every_n_counts_calls_not_time(self):
        a = _tok()
        a.set_low_first(3)
        _fragment(a)                                                # 4 allocs already counted
        calls0 = a._weg2_low_first_calls
        with mock.patch("time.time", side_effect=AssertionError("wall time read")), \
                mock.patch("time.monotonic", side_effect=AssertionError("wall time read")):
            for _ in range(5):
                a.alloc(1)
        self.assertEqual(a._weg2_low_first_calls, calls0 + 5)
        self.assertGreaterEqual(a._weg2_low_first_sorts, 1)

    def test_two_ranks_same_calls_same_list(self):
        r0, r1 = _tok(), _tok()
        for a in (r0, r1):
            a.set_low_first(2)
        outs = []
        for a in (r0, r1):
            seq = []
            for i in range(30):
                x = a.alloc(3)
                seq.append(x.tolist())
                if i % 3 == 1:
                    a.free(x)
            outs.append((seq, a.free_pages.tolist(), a._weg2_low_first_sorts))
        self.assertEqual(outs[0], outs[1])

    def test_owner_bias_owns_the_order(self):
        a = _tok()
        a.set_low_first(1)
        a._owner_bias = (2, 0, 1)
        _fragment(a)
        before = a.free_pages.tolist()
        a.alloc(1)
        self.assertEqual(a._weg2_low_first_sorts, 0)
        self.assertEqual(a.free_pages.tolist(), before[1:])

    def test_free_group_in_flight_is_not_reordered(self):
        a = _tok()
        a.set_low_first(1)
        a.is_not_in_free_group = False
        a.alloc(1)
        self.assertEqual(a._weg2_low_first_sorts, 0)


class TestPagedAllocatorLowFirst(CustomTestCase):
    def test_default_off_head_is_what_was_freed(self):
        a = _paged()
        x = a.alloc(8)                 # pages 1,2
        y = a.alloc(8)                 # pages 3,4
        a.free(x)
        nxt = a.alloc(8)
        self.assertEqual(nxt.tolist()[0] // 4, 1)    # paged off: the freed page is reused first (as before)
        self.assertFalse(hasattr(a, "_weg2_low_first_sorts"))
        del y

    def test_on_sorts_and_places_low(self):
        a = _paged()
        a.set_low_first(1)
        xs = [a.alloc(4) for _ in range(6)]          # pages 1..6
        a.free(xs[4])                                # page 5 first, then 1
        a.free(xs[0])
        self.assertEqual(a.free_pages.tolist()[:2], [1, 5])   # paged free puts the freed ids at the head, unsorted
        a.free_pages = torch.cat((a.free_pages[1:], a.free_pages[:1]))   # a scrambled head: 5, 7, ..., 1
        got = a.alloc(4)
        self.assertEqual(got.tolist()[0] // 4, 1)
        self.assertEqual(a._weg2_low_first_sorts, 7)               # N=1: six allocs before + this one


class _FakeAlloc:
    free_pages = None
    release_pages = None


class TestArm(CustomTestCase):
    def test_off_by_default(self):
        a = _tok()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_WEG2_DUAL_D_LOW_FIRST", None)
            self.assertEqual(D._arm_low_first(a), 0)
        self.assertFalse(hasattr(a, "_weg2_low_first_every"))

    def test_on_arms_token_and_paged(self):
        for a in (_tok(), _paged()):
            with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_D_LOW_FIRST": "8"}):
                self.assertEqual(D._arm_low_first(a), 8)
            self.assertEqual(a._weg2_low_first_every, 8)

    def test_other_class_is_refused_loudly(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_D_LOW_FIRST": "8"}), \
                self.assertLogs(D.logger, level="WARNING") as cm:
            self.assertEqual(D._arm_low_first(_FakeAlloc()), 0)
        self.assertTrue(any("NOT armed" in m for m in cm.output))

    def test_env_garbage_does_not_arm(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_D_LOW_FIRST": "abc"}):
            self.assertEqual(D.low_first_every(), 0)


class _Req:
    def __init__(self, rid, idx, n):
        self.rid, self.req_pool_idx = rid, idx
        self.origin_input_ids = list(range(n))
        self.output_ids = []


class _Pool:
    def __init__(self, rows):
        self.req_to_token = rows


class _Sched:
    def __init__(self, reqs, rows):
        self.running_batch = type("B", (), {"reqs": reqs})()
        self.chunked_req = None
        self.waiting_queue = []
        self.weg2_d_parked = []
        self.req_to_token_pool = _Pool(rows)


class TestLiveFloorCensus(CustomTestCase):
    def setUp(self):
        PI._reset_for_tests()

    def test_names_the_holders_over_the_need_and_the_free_room_below(self):
        a = _tok()
        a.set_low_first(1)
        _fragment(a)                                   # live 17..32, free 1..16 (+33..64)
        rows = torch.zeros((4, 16), dtype=torch.int64)
        rows[0, :8] = torch.arange(17, 25)
        rows[1, :8] = torch.arange(25, 33)
        sched = _Sched([_Req("weg2-0-low", 0, 8), _Req("weg2-0-high", 1, 8)], rows)
        out = PI.live_floor_census(sched, a, need_tokens=26, page=1)
        self.assertEqual(out["reqs_over_need"], 1)
        self.assertTrue(out["h1"].startswith("running:weg2-0-high:row32"))
        self.assertEqual(out["free_below_need"], 16 + 0)     # ids 1..16 are below 26; the pristine tail starts at 33
        self.assertEqual(out["low_first"], 1)

    def test_garbage_never_raises(self):
        for sched, alloc in ((None, None), (object(), object()), (_Sched([_Req("x", 9, 4)], torch.zeros((1, 2), dtype=torch.int64)), _FakeAlloc())):
            out = PI.live_floor_census(sched, alloc, 10, 1)
            self.assertIsInstance(out, dict)

    def test_instrument_only_for_live_floor_and_never_raises(self):
        actor = mock.Mock(mapped_tokens=1024, page=1, allocator=_tok())
        with mock.patch.object(PI, "begin", return_value=0) as begin, \
                mock.patch.object(PI, "live_floor_census", side_effect=RuntimeError("boom")):
            D._instr_live_floor(object(), actor, "none", 8, 16, 32, 5.0)
            begin.assert_not_called()                  # other reasons: not even the rate slot is taken
            D._instr_live_floor(object(), actor, "live_floor", 8, 16, 32, 5.0)   # census raises: swallowed
            begin.assert_called_once()

    def test_instrument_line_is_logged_once_per_gap(self):
        actor = mock.Mock(mapped_tokens=1024, page=1, allocator=_tok())
        with mock.patch.object(PI, "enabled", return_value=True), \
                self.assertLogs(D.logger, level="INFO") as cm:
            D._instr_live_floor(object(), actor, "live_floor", 8, 16, 32, 5.0)
            D._instr_live_floor(object(), actor, "live_floor", 8, 16, 32, 5.0)   # within MIN_GAP_S: suppressed
        lines = [m for m in cm.output if "#1540 D-LIVE-FLOOR" in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("mapped=1024", lines[0])
        self.assertIn("reqs_over_need=", lines[0])


if __name__ == "__main__":
    import unittest

    unittest.main()
