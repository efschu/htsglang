"""H105e: under the #239 token cut a KV-only rank (no GDN anchor adopted) whose
load-back the group floor refused AFTER it took the gathered ADMIT follows the
ADMIT through its own load-back -- exactly as an anchor-adopting rank does
(H105c) -- instead of silently re-prefilling from its device prefix.

THE DEATH (NF xc boot dkrnfint4h6ablxcbar1dauer10061548, tree 220e3f8b70 +
integ fixes, profile nf-int4-h6-abl-xc; D log
boot_weg2_dkrnfint4h6ablxcbar1dauer10061548_220e3f8b70_1006_154845.D.log
227758-227879, 16:52:03-04Z, Form A x #239 token cut):

* TP1/TP2 ``[#904 match-census] verdict=refused ... why=MambaComponent:absent``
  -- the workers' trees carry no mamba state on the path; TP0
  ``MAMBA-HOST-RESUME ... anchor accepted at depth=16640 on a HOST-backed
  state``;
* all three ``H98x X-FLOOR-CREDIT rid=weg2-72-531 head=0 store=0 floor=16640
  total=18729 uncached=2089`` and ``WEG2 X-GATE ... verdict=admit``; the
  gathered H105 verdict ADMIT (no H105d refusal: every rank had the room);
* TP1/TP2: ``#1048 EXTENT STALE rid=weg2-72-531: this rank's stamp asked for
  16640 token(s) and its own load-back served 0`` -> ``#988 LOADBACK ...
  prefix moved to 0 ... kv_only=2`` -- no anchor adopted, so H105c's follow
  did not apply and the #1048 arm took 0;
* TP0: ``H105c FORM-A FOLLOW-ROOM rid=weg2-72-531 kv_tokens=16640 evicted=0
  available=65536`` -> ``H105c FORM-A FOLLOW LOAD-BACK ... applied=16640`` --
  its first load-back had also served 0 (the group floor refused it, the live
  pool had the room) and it followed;
* TP1/TP2 riegel: ``FormAAdmissionSplit: H105 RU FORM-A EXTEND-SET SPLIT
  host=[('weg2-72-531', 16640, 18729)] local=[('weg2-72-531', 0, 2688)]``.

Driven through the REAL ``PrefillAdder.add_one_req``, the REAL
``Scheduler._form_a_admission_follow_fn`` with the token cut's gather (three
ranks in threads) and the REAL ``form_a_extend_set_check``. RED on 18a31940b3:
the workers build (0, 2688) against the host's (16640, 18729). GREEN with
H105e.
"""

from __future__ import annotations

import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import schedule_policy as sp
from sglang.srt.managers import tp_match_floor as m
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

RID = "weg2-72-531"
EXTENT = 16640  # '#1042 EXTENT ... extent=16640' on TP0/1/2
TOTAL = 18729  # 'X-FLOOR-CREDIT ... total=18729'
CHUNK = 2688  # the workers' chunk on the metal: local=[('weg2-72-531', 0, 2688)]
AVAILABLE = {0: 65536, 1: 40000, 2: 40000}  # TP0: 'FOLLOW-ROOM ... available=65536'
TOKEN_CUT = "sglang.srt.rank_role.form_a_token_cut_active"


class _Gather:
    def __init__(self, n=3):
        self.n = n
        self.slots = [None] * n
        self.barrier = threading.Barrier(n, timeout=20)

    def fn(self, tp_rank):
        def _g(site, payload):
            self.slots[tp_rank] = payload
            self.barrier.wait()
            out = list(self.slots)
            self.barrier.wait()
            return out

        return _g


def _scheduler(gather, tp_rank):
    s = SimpleNamespace(
        ps=SimpleNamespace(tp_size=3, pp_size=1),
        tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]),
        tp_cpu_group=None,
    )
    s._form_a_tp_gather = gather.fn(tp_rank)
    s._form_a_tp_exchange = lambda site, payload: (_ for _ in ()).throw(
        AssertionError("the token cut gathers; it never broadcasts the verdict")
    )
    for name in ("_form_a_is_host", "_form_a_admission_follow_fn"):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _tree_cache(alloc, *, adopts_anchor):
    """The rank's tree: the first load-back is refused by the group floor
    (0 rows -- ``unified_radix_cache.load_back`` ``floor < kv_tokens``), the
    follow retry decides from the live pool and serves the extent. Only the
    host adopts a GDN anchor; a worker's walk is MambaComponent:absent."""
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._weg2_loadback_no_room = 0
    tc._h105c_follow_room = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    tc.token_to_kv_pool_allocator = alloc
    tc.load_backs = []

    def _init_load_back(params):
        req = params.req
        follow = bool(tc._h105c_follow_room)
        tc.load_backs.append(follow)
        # what the real load_back marks at its entry and at the floor refusal
        req._h105e_floor_refused = False
        if adopts_anchor:
            req.mamba_loadback_anchor_adopted = True
        if follow and alloc.available_size.return_value >= EXTENT:
            return torch.arange(EXTENT, dtype=torch.int64), req.last_node
        req._h105e_floor_refused = True
        return torch.arange(0, dtype=torch.int64), req.last_node

    tc.init_load_back.side_effect = _init_load_back
    return tc


def _best_match_node():
    device_parent = SimpleNamespace(evicted=False, parent=None, component_data={})
    return SimpleNamespace(
        evicted=True,
        parent=device_parent,
        component_data={
            BASE_COMPONENT_TYPE: SimpleNamespace(
                host_value=torch.arange(EXTENT, dtype=torch.int64)
            )
        },
    )


def _req():
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = torch.arange(0, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(TOTAL))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.best_match_node = _best_match_node()
    req.born_spilled = False
    req.born_spilled_deep = False
    req.extend_range = None
    req.set_extend_range.side_effect = lambda a, b: setattr(
        req, "extend_range", SimpleNamespace(start=a, end=b)
    )
    return req


def _adder(tp_rank, gather):
    alloc = _allocator(AVAILABLE[tp_rank])
    rb = MagicMock()
    rb.reqs = []
    adder = PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(alloc, adopts_anchor=tp_rank == 0),
        token_to_kv_pool_allocator=alloc,
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=CHUNK,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
        m, "this_rank_follows", return_value=tp_rank != 0
    ):
        adder.form_a_admission_follow = _scheduler(gather, tp_rank)._form_a_admission_follow_fn()
    return adder


def _group_gate(*, token_cut=True):
    gather = _Gather()
    out, adders, reqs = {}, {}, {}
    with patch(TOKEN_CUT, return_value=token_cut):
        for r in range(3):
            adders[r] = _adder(r, gather)
            reqs[r] = _req()

        def _run(r):
            try:
                out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
            except BaseException as e:  # noqa: BLE001 -- the result under test
                out[r] = e

        with patch.object(sp, "_pp_load_back_extent", return_value=EXTENT), patch(
            "sglang.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ):
            threads = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
    return out, adders, reqs


class KvOnlyFollowTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_xc_1652_kv_only_workers_follow_and_the_riegel_agrees(self):
        # H98e: switched on (default) the first load-back past the gathered
        # ADMIT already decides from the live pool; off, it is the floor's and
        # the H105e follow retries -- both must end in the host's extend set.
        from sglang.srt.environ import envs

        for room_first, seen in ((False, [False, True]), (True, [True])):
            with self.subTest(room_first=room_first), \
                    envs.SGLANG_WEG2_ENABLE_FORM_A_ADMIT_ROOM_FIRST.override(room_first):
                self._xc_1652(seen)

    def _xc_1652(self, seen):
        out, adders, reqs = _group_gate()
        host_set = m.form_a_extend_set(adders[0].can_run_list)
        self.assertEqual(host_set, [(RID, EXTENT, TOTAL)])
        for r in (1, 2):
            self.assertNotIsInstance(out[r], BaseException, f"TP{r}: {out[r]}")
            local = m.form_a_extend_set(adders[r].can_run_list)
            # base: FormAAdmissionSplit H105 RU FORM-A EXTEND-SET SPLIT
            # host=[(rid, 16640, 18729)] local=[(rid, 0, 2688)] -- the metal line
            m.form_a_extend_set_check(local, is_host=False, exchange=lambda p: host_set)
            # off: the retry made its own room; the first attempt was the floor's
            self.assertEqual(adders[r].tree_cache.load_backs, seen, f"TP{r}")
            self.assertEqual(len(reqs[r].prefix_indices), EXTENT, f"TP{r}")
        self.assertEqual(adders[0].tree_cache.load_backs, seen)

    def test_off_the_token_cut_a_kv_only_rank_keeps_the_1048_arm(self):
        """No cut (H105b form): untouched -- a KV-only worker that served 0
        takes the #1048 arm as before, no follow retry."""
        rb = MagicMock()
        rb.reqs = []
        alloc = _allocator(AVAILABLE[1])
        tc = _tree_cache(alloc, adopts_anchor=False)
        adder = PrefillAdder(
            page_size=1, tree_cache=tc, token_to_kv_pool_allocator=alloc,
            running_batch=rb, new_token_ratio=1.0, rem_input_tokens=10**9,
            rem_chunk_tokens=CHUNK, num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        follow = SimpleNamespace(host_decides_load_back=False)
        req = _req()
        with patch(TOKEN_CUT, return_value=False):
            got = sp._h105c_follow(adder, req, EXTENT, follow, tc.init_load_back(
                SimpleNamespace(req=req, best_match_node=req.best_match_node)
            )[0])
        self.assertEqual(int(got.numel()), 0)
        self.assertEqual(tc.load_backs, [False])

    def test_a_non_floor_zero_is_not_followed(self):
        """Served 0 for another reason (stale stamp, unservable component):
        no marker, no retry -- the #1048 arm as before, even on the cut."""
        req = _req()
        req._h105e_floor_refused = False
        follow = SimpleNamespace(host_decides_load_back=False)
        adder = SimpleNamespace(tree_cache=MagicMock())
        with patch(TOKEN_CUT, return_value=True):
            got = sp._h105c_follow(
                adder, req, EXTENT, follow, torch.arange(0, dtype=torch.int64)
            )
        self.assertEqual(int(got.numel()), 0)
        adder.tree_cache.init_load_back.assert_not_called()


class LoadBackFloorMarkTest(unittest.TestCase):
    """The marker is written by the REAL ``UnifiedRadixCache.load_back``: False
    at its entry, True only at the group-floor refusal."""

    def _tree(self, floor):
        t = SimpleNamespace()
        kv_xfer = SimpleNamespace(host_indices=torch.arange(EXTENT), nodes_to_load=())
        base = MagicMock()
        base.build_hicache_transfers.return_value = [kv_xfer]
        t.cache_controller = MagicMock()
        t.cache_controller.load.return_value = torch.arange(EXTENT)
        t.components = {BASE_COMPONENT_TYPE: base}
        t._components_tuple = ()
        lock = SimpleNamespace(to_dec_params=lambda: None, delta=0)
        t.inc_host_lock_ref = lambda node: lock
        t.inc_lock_ref = lambda node: lock
        t.dec_lock_ref = lambda *a: None
        t.dec_host_lock_ref = lambda *a: None
        t._1424_verify_load_chain = lambda *a, **k: None
        t._build_sidecar_transfers = lambda *a: []
        t.load_back_threshold = 10
        t.uniform_avail_floor = floor
        t.evictable_size = lambda: 0
        t.token_to_kv_pool_allocator = _allocator(65536)
        t._record_store_event = lambda *a, **k: None
        t._update_evictable_leaf_sets = lambda *a: None
        t.ongoing_load_back = {}
        t.metrics_collector = None
        return t

    def _call(self, t, req):
        with patch.object(urc, "_form_a_load_back_floor", side_effect=lambda tree, f, k, rid=None: f), \
                patch.object(urc, "_form_a_note_loaded"), \
                patch("sglang.srt.weg2.pp_slot_fidelity.local_pp_room", return_value=None), \
                patch("sglang.srt.weg2.pp_slot_fidelity.note_loaded"):
            return urc.UnifiedRadixCache.load_back(t, MagicMock(), None, req=req)

    def test_floor_refusal_marks_the_request(self):
        req = SimpleNamespace(rid=RID)
        self.assertFalse(self._call(self._tree(floor=1000), req))
        self.assertIs(req._h105e_floor_refused, True)

    def test_a_load_back_that_fits_clears_the_mark(self):
        req = SimpleNamespace(rid=RID, _h105e_floor_refused=True)
        self.assertTrue(self._call(self._tree(floor=65536), req))
        self.assertIs(req._h105e_floor_refused, False)


if __name__ == "__main__":
    unittest.main()
