"""H105b: on a Form A D group the attention host's LOAD-BACK WAIT is the
group's admission verdict too.

THE DEATH (rc12z20 3a86888ba5, boot
dkrnfh91dprsavisnoadoptstbar1dauer09281220, D log ~252090-252147, 12:55:34).
pdflip-36-144 (host depth 84480) was admitted on every rank; in the same pass
pdflip-36-145 came to the gate with a host-backed hit (TP0 depth 62016, device
prefix 3712):

* every rank passed the H105 gate and the host's verdict (ADMIT) went out --
  BEFORE the load-back;
* TP0: ``PDFLIP-LOADBACK-WAIT rid=pdflip-36-145 extent=62016 applied=0: no device
  room for the host hit yet ... the anchor is given back and the request
  waits`` -> NO_TOKEN;
* TP1/TP2 (expert workers, no recurrent state, never adopt an anchor):
  ``#1048 EXTENT STALE ... asked for 58304 ... served 0; taking the served
  amount`` and ``#988 LOADBACK rid=pdflip-36-145 prefix moved to 3712`` -> the
  rid admitted at its device prefix;
* the post-loop riegel: ``FormAAdmissionSplit: H105 RU FORM-A EXTEND-SET SPLIT
  host=[('pdflip-36-144', 84480, 84656)] local=[..., ('pdflip-36-145', 3712,
  7552)]`` -- D dead.

Driven through the REAL ``PrefillAdder.add_one_req`` and the REAL
``Scheduler._form_a_admission_follow_fn`` / ``_form_a_extend_set_riegel`` (a
stand-in scheduler; the TP broadcast is an ordered stream the host fills
first, as ``broadcast_pyobj`` does). RED on cb7f2cdc35: the workers admit
pdflip-36-145, the riegel stops. GREEN with H105b.
"""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from flliper.srt.managers import schedule_policy as sp
from flliper.srt.managers import tp_match_floor as m
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

#: pdflip-36-145 geometry (D log 12:55:34).
RID = "pdflip-36-145"
DEVICE_PREFIX = 3712
HOST_EXTENT = 62016 - DEVICE_PREFIX  # TP0's own extent above its device rows
WORKER_EXTENT = 58304  # the workers' (stale) stamp
FILL = 62016 + 1629
AVAILABLE = 200_000  # every rank passes the H105 budget gate


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


def _scheduler(ch, tp_rank):
    s = SimpleNamespace(
        ps=SimpleNamespace(tp_size=3, pp_size=1),
        tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]),
        tp_cpu_group=None,
    )
    s._form_a_tp_exchange = ch.exchange(tp_rank)
    for name in (
        "_form_a_is_host",
        "_form_a_admission_follow_fn",
        "_form_a_extend_set_riegel",
    ):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s


def _tree_cache(*, adopts: bool, served: int):
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._pdflip_loadback_no_room = 0
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()

    def _init_load_back(params):
        # TP0: no device room -> 0 rows, but the mamba restore adopted the
        # anchor (the WAIT branch). A worker: 0 rows, no anchor ever.
        if adopts:
            params.req.mamba_loadback_anchor_adopted = True
        return torch.arange(served, dtype=torch.int64), params.req.last_node

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


def _install(adder, sched, tp_rank, *, form_a=True):
    with patch.object(m, "form_a_follow_active", return_value=form_a), patch.object(
        m, "this_rank_follows", return_value=form_a and tp_rank != 0
    ):
        adder.form_a_admission_follow = sched._form_a_admission_follow_fn()


class LoadBackWaitIsTheGroupsTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _rank(self, ch, tp_rank, *, adopts, served, extent):
        adder = _adder(_tree_cache(adopts=adopts, served=served))
        sched = _scheduler(ch, tp_rank)
        _install(adder, sched, tp_rank)
        req = _req()
        with patch.object(sp, "_pp_load_back_extent", return_value=extent), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            res = adder.add_one_req(req, truncation_align_size=None)
        return res, adder, sched, req

    def _run(self):
        ch = _Channel()
        out = {0: self._rank(ch, 0, adopts=True, served=0, extent=HOST_EXTENT)}
        for r in (1, 2):
            out[r] = self._rank(ch, r, adopts=False, served=0, extent=WORKER_EXTENT)
        return ch, out

    def test_rc12z20_every_rank_waits_with_the_host(self):
        """THE DEATH: TP0 WAITs in its load-back, the workers must not admit."""
        _, out = self._run()
        self.assertEqual(out[0][0], AddReqResult.NO_TOKEN)
        for r in (1, 2):
            self.assertEqual(
                out[r][0],
                AddReqResult.NO_TOKEN,
                f"worker {r} admitted {RID} at its device prefix ({out[r][0]}) while "
                "the attention host waited in its load-back -- the rc12z20 split",
            )
            self.assertEqual(out[r][1].can_run_list, [])
            # the worker never ran a load-back of its own: the host's wait
            # arrived before it
            out[r][1].tree_cache.init_load_back.assert_not_called()
            self.assertEqual(len(out[r][3].prefix_indices), DEVICE_PREFIX)

    def test_rc12z20_riegel_agrees_after_the_loop(self):
        """The post-loop riegel sees one extend set on every rank."""
        ch, out = self._run()
        with patch.object(m, "form_a_follow_active", return_value=True):
            with patch.object(m, "this_rank_follows", return_value=False):
                out[0][2]._form_a_extend_set_riegel(out[0][1].can_run_list)
            for r in (1, 2):
                with patch.object(m, "this_rank_follows", return_value=True):
                    out[r][2]._form_a_extend_set_riegel(out[r][1].can_run_list)

    def test_one_broadcast_per_gate_call(self):
        """No new collective: the host posts exactly one verdict for the rid
        (after its load-back), each worker reads exactly one."""
        ch, _ = self._run()
        verdicts = [p for s, p in ch.posts if s == "form-a-admission/tp<-verdict"]
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0][0], RID)
        self.assertEqual(verdicts[0][1], "NO_TOKEN")
        self.assertEqual(ch.read, {1: 1, 2: 1})

    def test_host_loads_then_admits_workers_follow(self):
        """The other branch: the host's load-back serves its extent, the ADMIT
        goes out after it, the workers load back on it."""
        ch = _Channel()
        h = self._rank(ch, 0, adopts=False, served=HOST_EXTENT, extent=HOST_EXTENT)
        self.assertNotEqual(h[0], AddReqResult.NO_TOKEN)
        self.assertEqual(len(h[1].can_run_list), 1)
        h[1].tree_cache.init_load_back.assert_called_once()
        verdicts = [p for s, p in ch.posts if s == "form-a-admission/tp<-verdict"]
        self.assertEqual([v[1] for v in verdicts], ["ADMIT"])
        w = self._rank(ch, 1, adopts=False, served=HOST_EXTENT, extent=HOST_EXTENT)
        self.assertNotEqual(w[0], AddReqResult.NO_TOKEN)
        w[1].tree_cache.init_load_back.assert_called_once()
        self.assertEqual(ch.read[1], 1)

    def test_host_gate_refusal_goes_out_before_any_load_back(self):
        """A budget refusal on the host is sent at once; nobody loads back."""
        ch = _Channel()
        adder = _adder(_tree_cache(adopts=True, served=0), available=1000)
        _install(adder, _scheduler(ch, 0), 0)
        req = _req()
        with patch.object(sp, "_pp_load_back_extent", return_value=HOST_EXTENT):
            res = adder.add_one_req(req, truncation_align_size=None)
        self.assertEqual(res, AddReqResult.NO_TOKEN)
        adder.tree_cache.init_load_back.assert_not_called()
        self.assertEqual([p[1] for s, p in ch.posts], ["NO_TOKEN"])


class OffFormAIsUnchangedTest(unittest.TestCase):
    """27B / classic boots: no follow callable, no broadcast; the load-back
    WAIT and the #1048 arm behave exactly as on the base."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _single(self, *, adopts, served, extent):
        ch = _Channel()
        adder = _adder(_tree_cache(adopts=adopts, served=served))
        sched = _scheduler(ch, 0)
        _install(adder, sched, 0, form_a=False)
        self.assertIsNone(adder.form_a_admission_follow)
        req = _req()
        with patch.object(sp, "_pp_load_back_extent", return_value=extent), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            res = adder.add_one_req(req, truncation_align_size=None)
        self.assertEqual(ch.posts, [])
        return res, adder, req

    def test_wait_stays_a_local_no_token(self):
        res, adder, _ = self._single(adopts=True, served=0, extent=HOST_EXTENT)
        self.assertEqual(res, AddReqResult.NO_TOKEN)
        adder.tree_cache.init_load_back.assert_called_once()

    def test_stale_extent_still_takes_the_served_amount(self):
        res, adder, req = self._single(adopts=False, served=0, extent=HOST_EXTENT)
        self.assertNotEqual(res, AddReqResult.NO_TOKEN)
        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(len(req.prefix_indices), DEVICE_PREFIX)

    def test_full_load_back_admits(self):
        res, adder, req = self._single(adopts=False, served=HOST_EXTENT, extent=HOST_EXTENT)
        self.assertNotEqual(res, AddReqResult.NO_TOKEN)
        self.assertEqual(len(req.prefix_indices), DEVICE_PREFIX + HOST_EXTENT)


if __name__ == "__main__":
    unittest.main()
