"""H106: the two D deaths of 28.09. afternoon, one admission family.

(1) rc12z23 e7b7c22459, boot dkrnfh91dprsavisnoadoptstbar1dauer09281515, D
    15:28:40-15:36:12 (deadman BUSY-STARVED). Under H105b only the attention
    host runs the load-back; the workers read its verdict first. The host's
    load-back was refused by the group floor and drained ITS evictable leaves
    (xsn285 WEG2-LOADBACK-EVICT: 116416 at 15:28:40, 97152 at 15:28:52) -- the
    workers never did. ``#1045 FLOOR PUBLISHED floor=30336 max=243904
    pools_equal=False``: max - floor = 213568 = 116416 + 97152; the workers'
    H105 local_budget 243904 = 30336 free + 213568 evictable; the host's
    budget 48576 = floor 30336 + own evictable 18240 < price 79176 -> the head
    weg2-14-33 was refused 8920 times with 0 running, nothing ever drained the
    workers, the floor never rose.

    Fix: the host's drain rides the H105 verdict it sends anyway; every worker
    drains its own evictable leaves alike. A head the group can no longer
    fund with nothing running stops by name (FormAAdmissionDeadlock).

(2) rc12z22-dwell30 7d507357b9, boot ...dwell30bar1dauer09281540, D 15:51:41,
    all three ranks: ``SEAT-AGE DISPLACE rid_out=weg2-6-33 ... trigger=kv``
    requeued the youngest running request AFTER the pass's #580 prefetch
    drain; the admission loop reached it and ``_prefetch_done_for`` raised
    ('the queue was mutated in between'). Fix: the displaced rid is excluded
    from this pass's admission by name (a not-done verdict, every rank alike).

Driven through the REAL ``PrefillAdder.add_one_req``, the REAL
``Scheduler._form_a_admission_follow_fn`` / ``_prefetch_done_for`` and the
REAL ``d_park_runtime.displace_for_age`` (stand-in scheduler; the TP
broadcast is an ordered stream the host fills first).
"""

from __future__ import annotations

import os
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.managers import schedule_policy as sp
from sglang.srt.managers import tp_match_floor as m
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

#: weg2-14-33 geometry (D log 15:28:40-15:36:12).
RID = "weg2-14-33"
DEVICE_PREFIX = 18240
HOST_EXTENT = 75008 - DEVICE_PREFIX
FILL = 75016
HOST_DRAINED = 116416  # WEG2-LOADBACK-EVICT 15:28:40 (TP0 only)
WORKER_EVICTABLE = 116416  # the same leaves, still on the workers
AVAILABLE = 200_000  # the pre-lock gate passes; the load-back floor refuses


class _Channel:
    """The TP broadcast as it really is: one ordered stream from the host."""

    def __init__(self):
        self.posts = []
        self.read = {1: 0, 2: 0}

    def exchange(self, tp_rank):
        def _ex(site, payload):
            if tp_rank == 0:
                self.posts.append((site, payload))
                return payload
            i = self.read[tp_rank]
            self.read[tp_rank] += 1
            return self.posts[i][1]

        return _ex


def _scheduler(ch, tp_rank, tree=None, running_empty=False):
    s = SimpleNamespace(
        ps=SimpleNamespace(tp_size=3, pp_size=1),
        tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]),
        tp_cpu_group=None,
        tree_cache=tree,
        running_batch=SimpleNamespace(is_empty=lambda: running_empty),
        chunked_req=None,
    )
    s._form_a_tp_exchange = ch.exchange(tp_rank)
    for name in ("_form_a_is_host", "_form_a_admission_follow_fn"):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s


def _tree_cache(*, adopts: bool, drain: int = 0, evictable: int = 0):
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = evictable
    tc.full_evictable_size.return_value = evictable
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._weg2_loadback_no_room = 0
    tc._weg2_loadback_drained_total = 0
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    tc.evict.return_value = SimpleNamespace(num_tokens_evicted=evictable)

    def _init_load_back(params):
        # TP0: the group floor refuses, xsn285 drains THIS rank's leaves and
        # counts them; the mamba restore adopted the anchor -> the WAIT branch.
        if adopts:
            tc._weg2_loadback_drained_total += drain
            params.req.mamba_loadback_anchor_adopted = True
        return torch.arange(0, dtype=torch.int64), params.req.last_node

    tc.init_load_back.side_effect = _init_load_back
    return tc


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _adder(tree_cache, available=AVAILABLE):
    rb = MagicMock()
    rb.reqs = []
    return PrefillAdder(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=_allocator(available),
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=4096,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _req():
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = torch.arange(DEVICE_PREFIX, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(FILL))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.best_match_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


class HostDrainRidesTheVerdictTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _rank(self, ch, tp_rank, tree, *, extent):
        adder = _adder(tree)
        sched = _scheduler(ch, tp_rank, tree)
        with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
            m, "this_rank_follows", return_value=tp_rank != 0
        ):
            adder.form_a_admission_follow = sched._form_a_admission_follow_fn()
        with patch.object(sp, "_pp_load_back_extent", return_value=extent), patch(
            "sglang.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            return adder.add_one_req(_req(), truncation_align_size=None)

    def _run(self, host_drain):
        ch = _Channel()
        trees = {0: _tree_cache(adopts=True, drain=host_drain, evictable=0)}
        out = {0: self._rank(ch, 0, trees[0], extent=HOST_EXTENT)}
        for r in (1, 2):
            trees[r] = _tree_cache(adopts=False, evictable=WORKER_EVICTABLE)
            out[r] = self._rank(ch, r, trees[r], extent=HOST_EXTENT)
        return out, trees

    def test_rc12z23_workers_drain_what_the_host_drained(self):
        """THE DEATH: TP0 alone drained, the workers kept the leaves, the
        group floor stayed at their availability for 7 min."""
        out, trees = self._run(HOST_DRAINED)
        for r in (0, 1, 2):
            self.assertEqual(out[r], AddReqResult.NO_TOKEN)
        for r in (1, 2):
            trees[r].evict.assert_called_once()
            (params,), _ = trees[r].evict.call_args
            self.assertEqual(
                params.num_tokens,
                WORKER_EVICTABLE,
                f"worker {r} kept its evictable leaves while the host drained "
                f"{HOST_DRAINED} -- the rc12z23 floor split",
            )

    def test_no_host_drain_no_worker_drain(self):
        out, trees = self._run(0)
        for r in (1, 2):
            self.assertEqual(out[r], AddReqResult.NO_TOKEN)
            trees[r].evict.assert_not_called()


class NamedDeadlockStopTest(unittest.TestCase):
    def test_watch_names_a_head_that_stands_still_with_nothing_running(self):
        w = m.FormAAdmissionWedgeWatch(120.0)
        self.assertEqual(w.observe(RID, "NO_TOKEN", 79176, 48576, running_empty=True, now=0.0), "")
        self.assertEqual(w.observe(RID, "NO_TOKEN", 79176, 48576, running_empty=True, now=119.0), "")
        msg = w.observe(RID, "NO_TOKEN", 79176, 48576, running_empty=True, now=121.0)
        self.assertIn("H106 FORM-A ADMISSION DEADLOCK", msg)
        self.assertIn("budget=48576", msg)

    def test_watch_restarts_on_a_run_a_moved_budget_or_an_admit(self):
        w = m.FormAAdmissionWedgeWatch(120.0)
        w.observe(RID, "NO_TOKEN", 79176, 48576, running_empty=True, now=0.0)
        self.assertEqual(w.observe(RID, "NO_TOKEN", 79176, 48576, running_empty=False, now=200.0), "")
        w.observe(RID, "NO_TOKEN", 79176, 48576, running_empty=True, now=200.0)
        self.assertEqual(w.observe(RID, "NO_TOKEN", 79176, 60000, running_empty=True, now=330.0), "")
        self.assertEqual(w.observe(RID, "ADMIT", 79176, 90000, running_empty=True, now=500.0), "")
        self.assertEqual(m.FormAAdmissionWedgeWatch(0).observe(
            RID, "NO_TOKEN", 1, 0, running_empty=True, now=1e9), "")

    def test_every_rank_stops_in_the_same_gate_call(self):
        ch = _Channel()
        with patch.object(m, "form_a_follow_active", return_value=True):
            scheds = {}
            for r in (0, 1, 2):
                with patch.object(m, "this_rank_follows", return_value=r != 0):
                    s = _scheduler(ch, r, tree=None, running_empty=True)
                    s._h106_wedge_watch = m.FormAAdmissionWedgeWatch(120.0)
                    scheds[r] = s._form_a_admission_follow_fn()
        req = SimpleNamespace(rid=RID)
        with patch("time.monotonic", return_value=0.0):
            scheds[0](req, AddReqResult.NO_TOKEN, 79176, 48576)
            for r in (1, 2):
                scheds[r](req, None, 60936, 243904)
        with patch("time.monotonic", return_value=461.3):
            for r in (0, 1, 2):
                with self.assertRaises(m.FormAAdmissionDeadlock) as cm:
                    scheds[r](req, AddReqResult.NO_TOKEN if r == 0 else None, 79176, 48576)
                self.assertIn("rid=" + RID, str(cm.exception))


class DisplacedAfterTheDrainTest(unittest.TestCase):
    """rc12z22-dwell30 15:51:41: SEAT-AGE DISPLACE weg2-6-33 after the drain."""

    def _sched(self, victim, older):
        s = SimpleNamespace(
            ps=SimpleNamespace(tp_size=3, pp_size=1),
            waiting_queue=[older],
            server_args=SimpleNamespace(max_running_requests=6),
        )
        s._add_request_to_queue = lambda req, is_retracted=False: s.waiting_queue.append(req)
        s._prefetch_done_for = types.MethodType(Scheduler._prefetch_done_for, s)
        return s

    def _displace(self, s, victim):
        from sglang.srt.weg2 import d_park_runtime as dpr

        rb = MagicMock()
        rb.reqs = [victim]
        rb.spec_algorithm = None
        with patch("sglang.srt.weg2.seat_age.enabled", return_value=True), patch.object(
            dpr.d_seats, "d_flip_park_active", return_value=True
        ), patch.object(dpr, "seat_cap", return_value=1), patch(
            "sglang.srt.weg2.seat_age.displace_victim",
            return_value=("weg2-6-31", "weg2-6-33"),
        ), patch.object(dpr.d_seats, "mark_parked"), patch.object(
            dpr.d_seats, "park_site", return_value=None
        ), patch.object(dpr, "_partial_keep", return_value="0"):
            return dpr.displace_for_age(s, rb)

    def test_dwell30_the_displaced_rid_sits_out_this_pass(self):
        older = SimpleNamespace(rid="weg2-6-31")
        victim = SimpleNamespace(rid="weg2-6-33")
        s = self._sched(victim, older)
        verdicts = {"weg2-6-31": True}  # the pass's #580 drain, before SA
        self.assertEqual(self._displace(s, victim), "weg2-6-33")
        self.assertIn(victim, s.waiting_queue)
        from sglang.srt.weg2 import d_park_runtime as dpr

        self.assertEqual(dpr.exclude_displaced(s, verdicts), "weg2-6-33")
        # the admission loop reads a not-done verdict instead of raising
        self.assertFalse(s._prefetch_done_for(victim, verdicts))
        self.assertTrue(s._prefetch_done_for(older, verdicts))
        # one pass only
        self.assertIsNone(dpr.exclude_displaced(s, {}))


if __name__ == "__main__":
    unittest.main()
