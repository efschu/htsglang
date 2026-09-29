"""H105c: a Form A worker that took the group's ADMIT follows it through its
own load-back instead of waiting alone.

THE DEATH (rc12z30g 7bd3541c4f, -st-vsync, no cut, Form A host,worker,worker,
1h07 under qwen load; D log boot_weg2_dkrnfh91dprsavisnoadoptstvsyncbar1dauer
09282210, ~745380-745460, 23:23:28). One pass after pdflip-180-303:

* pdflip-180-304 came to the gate host-backed; every rank passed it and the
  host's ADMIT went out (TP0 ``#988 LOADBACK rid=pdflip-180-304 prefix moved to
  71168 ... extent=71168``);
* TP1/TP2 read that ADMIT, then ran their own load-back:
  ``PDFLIP-LOADBACK-WAIT rid=pdflip-180-304 extent=63296 applied=0: no device room
  for the host hit yet (rem_total_tokens=157504.0)`` -> a rank-local NO_TOKEN,
  the loop ended;
* TP0 went on to pdflip-180-305 and posted its gate NO_TOKEN (price 98650,
  budget 89792);
* the workers' post-loop riegel read that post as the extend set:
  ``FormAAdmissionSplit: H105 RU FORM-A EXTEND-SET MALFORMED got=('pdflip-180-305',
  'NO_TOKEN', 98650, 89792, 0, '') local=[...]`` -- D dead, TP0 died on gloo.

Driven through the REAL ``PrefillAdder.add_one_req`` and the REAL
``Scheduler._form_a_admission_follow_fn`` / ``_form_a_extend_set_riegel`` with
the H105b harness (the TP broadcast as one ordered stream the host fills
first). RED on 7bd3541c4f: the worker returns NO_TOKEN for pdflip-180-304 and its
riegel reads the pdflip-180-305 verdict. GREEN with H105c.
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
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

#: pdflip-180-304 (D log 23:23:28): the group depth 71168; TP0 loads all of it,
#: the workers hold 7872 rows on device and load 63296 (their '#988' extent).
RID_304, RID_305 = "pdflip-180-304", "pdflip-180-305"
DEPTH_304 = 71168
WORKER_DEVICE = DEPTH_304 - 63296
WORKER_EXTENT = 63296
UNCACHED_304 = 221  # 'PDFLIP X-GATE rid=pdflip-180-304 uncached=221'
AVAILABLE = 157504  # 'rem_total_tokens=157504.0' on TP1/TP2
FILL_305 = 200_000  # the host's gate refuses pdflip-180-305 (metal: 98650 > 89792)


class _Channel:
    """The TP broadcast as it is: one ordered stream from the host."""

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
    for name in ("_form_a_is_host", "_form_a_admission_follow_fn", "_form_a_extend_set_riegel"):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s


def _tree_cache(*, worker: bool):
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._pdflip_loadback_no_room = 0
    tc._h105c_follow_room = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    tc.follow_room_seen = []

    def _init_load_back(params):
        req = params.req
        req.mamba_loadback_anchor_adopted = True  # the GDN anchor is adopted
        if not worker:
            return torch.arange(DEPTH_304, dtype=torch.int64), req.last_node
        tc.follow_room_seen.append(bool(tc._h105c_follow_room))
        if tc._h105c_follow_room:
            # its own room made: the load-back serves the worker's extent
            return torch.arange(WORKER_EXTENT, dtype=torch.int64), req.last_node
        return torch.arange(0, dtype=torch.int64), req.last_node  # the metal WAIT

    tc.init_load_back.side_effect = _init_load_back
    return tc


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _adder(tree_cache):
    rb = MagicMock()
    rb.reqs = []
    return PrefillAdder(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=_allocator(AVAILABLE),
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=4096,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _req(rid, device, fill):
    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = torch.arange(device, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(fill))
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
    req.extend_range = None
    req.set_extend_range.side_effect = lambda a, b: setattr(
        req, "extend_range", SimpleNamespace(start=a, end=b)
    )
    return req


def _install(adder, sched, tp_rank):
    with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
        m, "this_rank_follows", return_value=tp_rank != 0
    ):
        adder.form_a_admission_follow = sched._form_a_admission_follow_fn()


class FollowLoadBackTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _loop(self, ch, tp_rank):
        """One admission loop as the scheduler runs it: stop at the first
        result that is not CONTINUE."""
        worker = tp_rank != 0
        adder = _adder(_tree_cache(worker=worker))
        sched = _scheduler(ch, tp_rank)
        _install(adder, sched, tp_rank)
        dev = WORKER_DEVICE if worker else 0
        reqs = [
            _req(RID_304, dev, DEPTH_304 + UNCACHED_304),
            _req(RID_305, dev, FILL_305),
        ]
        extents = {RID_304: WORKER_EXTENT if worker else DEPTH_304, RID_305: None}
        results = []
        with patch.object(
            sp, "_pp_load_back_extent", side_effect=lambda r: extents[r.rid]
        ), patch("flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"):
            for req in reqs:
                res = adder.add_one_req(req, truncation_align_size=None)
                results.append((req.rid, res))
                if res != AddReqResult.CONTINUE:
                    break
        return adder, sched, results, reqs

    def _riegel(self, sched, adder, tp_rank):
        with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
            m, "this_rank_follows", return_value=tp_rank != 0
        ):
            sched._form_a_extend_set_riegel(adder.can_run_list)

    def test_rc12z30g_workers_follow_the_admit_and_the_riegel_agrees(self):
        ch = _Channel()
        host = self._loop(ch, 0)
        self.assertEqual([r for r, _ in host[2]], [RID_304, RID_305])
        self.assertEqual(host[2][1][1], AddReqResult.NO_TOKEN)  # the gate refusal
        self._riegel(host[1], host[0], 0)
        for r in (1, 2):
            w = self._loop(ch, r)
            self.assertEqual(
                [x for x, _ in w[2]], [RID_304, RID_305],
                f"worker {r} ended its loop at {w[2]} -- one gate call short of the "
                "host (the rc12z30g EXTEND-SET MALFORMED)",
            )
            self.assertEqual(w[2][1][1], AddReqResult.NO_TOKEN)  # followed the host
            self.assertEqual([q.rid for q in w[0].can_run_list], [RID_304])
            self.assertEqual(len(w[3][0].prefix_indices), DEPTH_304)
            # the retry made its own room; the first attempt did not
            self.assertEqual(w[0].tree_cache.follow_room_seen, [False, True])
            self._riegel(w[1], w[0], r)  # base: FormAAdmissionSplit MALFORMED
        self.assertEqual(ch.read, {1: 3, 2: 3})

    def test_unservable_follow_is_a_named_stop_not_a_local_wait(self):
        """Still no room after its own eviction: stop by name at the site."""
        ch = _Channel()
        self._loop(ch, 0)
        adder = _adder(_tree_cache(worker=True))
        adder.tree_cache.init_load_back.side_effect = lambda p: (
            setattr(p.req, "mamba_loadback_anchor_adopted", True)
            or (torch.arange(0, dtype=torch.int64), p.req.last_node)
        )
        sched = _scheduler(ch, 1)
        _install(adder, sched, 1)
        req = _req(RID_304, WORKER_DEVICE, DEPTH_304 + UNCACHED_304)
        with patch.object(sp, "_pp_load_back_extent", return_value=WORKER_EXTENT), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            with self.assertRaisesRegex(m.FormAAdmissionSplit, "H105c FORM-A FOLLOW LOAD-BACK UNSERVABLE"):
                adder.add_one_req(req, truncation_align_size=None)

    def test_host_first_wait_is_unchanged(self):
        """The host's own WAIT (H105b) still goes out as the group's NO_TOKEN;
        the host never retries."""
        ch = _Channel()
        tc = _tree_cache(worker=True)  # 0 rows + anchor on the first call
        adder = _adder(tc)
        _install(adder, _scheduler(ch, 0), 0)
        req = _req(RID_304, WORKER_DEVICE, DEPTH_304 + UNCACHED_304)
        with patch.object(sp, "_pp_load_back_extent", return_value=WORKER_EXTENT), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            res = adder.add_one_req(req, truncation_align_size=None)
        self.assertEqual(res, AddReqResult.NO_TOKEN)
        self.assertEqual(tc.follow_room_seen, [False])
        self.assertEqual([p[1] for _, p in ch.posts], ["NO_TOKEN"])


class _Tree:
    def __init__(self, *, available, evictable=0, floor=157504, admitted=0):
        self.token_to_kv_pool_allocator = _allocator(available)
        self._ev = evictable
        self.uniform_avail_floor = floor
        self.uniform_admitted_since_floor = admitted
        self.evicted = []

    def evictable_size(self):
        return self._ev

    def evict(self, params):
        self.evicted.append(params.num_tokens)
        self.token_to_kv_pool_allocator.available_size.return_value += params.num_tokens
        return SimpleNamespace(num_tokens_evicted=params.num_tokens)


class LoadBackFloorTest(unittest.TestCase):
    def test_a_pass_charged_floor_on_form_a(self):
        # 303's load-back took rows this pass: the second load-back sees them
        t = _Tree(available=157504, floor=157504, admitted=95142)
        with patch.object(m, "form_a_follow_active", return_value=True):
            self.assertEqual(urc._form_a_load_back_floor(t, 157504, 71168), 157504 - 95142)

    def test_off_form_a_the_published_floor(self):
        t = _Tree(available=1, floor=157504, admitted=95142)
        with patch.object(m, "form_a_follow_active", return_value=False):
            self.assertEqual(urc._form_a_load_back_floor(t, 157504, 71168), 157504)

    def test_follow_room_evicts_the_shortfall_only(self):
        t = _Tree(available=40000, evictable=90000)
        t._h105c_follow_room = True
        got = urc._form_a_load_back_floor(t, 157504, WORKER_EXTENT, RID_304)
        self.assertEqual(t.evicted, [WORKER_EXTENT - 40000])
        self.assertEqual(got, WORKER_EXTENT)


if __name__ == "__main__":
    unittest.main()
