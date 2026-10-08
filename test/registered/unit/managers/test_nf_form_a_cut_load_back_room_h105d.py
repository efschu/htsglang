"""H105d: under the #239 token cut a rank's gate knows whether it can pay its
own load-back rows BEFORE the group's MIN is gathered.

THE DEATH (NF y9nf6, image rc12z30y9nf6, tree 191228abc5; D log
boot_weg2_dkrnfint4h6ablbar1dauer10040608_191228abc5_1004_060857.D.log
~158218-158410, 06:32:52Z, Form A x #239 token cut [0, 7, 1]):

* ``WEG2 X-GATE rid=weg2-32-93 uncached=359 ... verdict=admit`` on TP0/1/2 --
  the gathered H105 verdict ADMIT, priced against ``rem_total_tokens=318912``
  (available 85568 + reported evictable 237568 - this pass's offset);
* the load-back then needed 121856 rows per rank; the peel delivered 7616
  (``WEG2-LOADBACK-EVICT ... floor=85568 ... requested=237568 evicted=7616``,
  ``EVICT-FRONTIER-CENSUS ... on_frontier=29312 aux_locked={}
  behind_device_child=200640``: write_back leaves whose backup the full host
  arena refused, ``R12 SHADOW-SHORT why=arena_claim``);
* ``H105c FORM-A FOLLOW-ROOM ... kv_tokens=121856 evicted=0 available=93184``
  -> ``FormAAdmissionSplit: H105c FORM-A FOLLOW LOAD-BACK UNSERVABLE
  rid=weg2-32-93`` on all three ranks -> RANK-DEATH, D group gone.

Driven through the REAL ``PrefillAdder.add_one_req`` and the REAL
``Scheduler._form_a_admission_follow_fn`` with the token cut's gather (three
ranks in threads, one all-gather per gate call). RED on 191228abc5: every rank
raises H105c. GREEN with H105d: every rank votes NO_TOKEN, the request stays
queued, nobody loads back.
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
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

RID = "weg2-32-93"
UNCACHED = 359  # 'WEG2 X-GATE rid=weg2-32-93 uncached=359'
KV_ROWS = 121856  # 'kv_tokens=121856' on TP0/TP1/TP2
EXTENT = {0: 134144, 1: 121856, 2: 121856}  # 'H105c ... extent=' per rank
AVAILABLE = 85568  # 'WEG2-LOADBACK-EVICT ... floor=85568'
REPORTED_EVICTABLE = 237568  # 'requested=237568'
DELIVERED = 7616  # 'evicted=7616' -- what the peel could pay
TOKEN_CUT = "sglang.srt.rank_role.form_a_token_cut_active"


class _Gather:
    """The token cut's all-gather over the TP cpu group: every rank's payload,
    in TP order, once all three arrived."""

    def __init__(self, n=3):
        self.n = n
        self.slots = [None] * n
        self.barrier = threading.Barrier(n, timeout=20)
        self.calls = {r: 0 for r in range(n)}

    def fn(self, tp_rank):
        def _g(site, payload):
            self.calls[tp_rank] += 1
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


def _tree_cache(alloc, *, deliverable):
    """The rank's tree: REPORTS ``REPORTED_EVICTABLE``, its peel PAYS only
    ``deliverable`` (the rest sits behind un-backable write_back leaves)."""
    tc = MagicMock()
    st = {"reported": REPORTED_EVICTABLE, "deliverable": deliverable}
    tc.supports_mamba.return_value = False
    tc.evictable_size.side_effect = lambda: st["reported"]
    tc.full_evictable_size.side_effect = lambda: st["reported"]
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._weg2_loadback_no_room = 0
    tc._h105c_follow_room = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    tc.token_to_kv_pool_allocator = alloc
    tc.load_backs = []
    tc.evicts = []

    def _evict(params):
        got = min(int(params.num_tokens), st["deliverable"])
        st["deliverable"] -= got
        st["reported"] -= got
        alloc.available_size.return_value += got
        tc.evicts.append((int(params.num_tokens), got))
        return SimpleNamespace(num_tokens_evicted=got)

    tc.evict.side_effect = _evict

    def _init_load_back(params):
        req = params.req
        follow = bool(tc._h105c_follow_room)
        tc.load_backs.append(follow)
        if follow:  # unified_radix_cache._form_a_load_back_floor, follow-room
            avail = alloc.available_size.return_value
            if avail < KV_ROWS:
                _evict(SimpleNamespace(num_tokens=min(st["reported"], KV_ROWS - avail)))
            if alloc.available_size.return_value >= KV_ROWS:
                alloc.available_size.return_value -= KV_ROWS
                return torch.arange(KV_ROWS, dtype=torch.int64), req.last_node
        # the group floor refused (85568 < 121856): 0 rows, GDN anchor adopted
        req.mamba_loadback_anchor_adopted = True
        return torch.arange(0, dtype=torch.int64), req.last_node

    tc.init_load_back.side_effect = _init_load_back
    return tc


def _best_match_node():
    """A host-only run of KV_ROWS under a device-resident parent."""
    device_parent = SimpleNamespace(evicted=False, parent=None, component_data={})
    return SimpleNamespace(
        evicted=True,
        parent=device_parent,
        component_data={
            BASE_COMPONENT_TYPE: SimpleNamespace(
                host_value=torch.arange(KV_ROWS, dtype=torch.int64)
            )
        },
    )


def _req(tp_rank):
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = torch.arange(0, dtype=torch.int64)  # device_len=0
    req.full_untruncated_fill_ids = list(range(EXTENT[0] + UNCACHED))
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
    req._h105d_extent = EXTENT[tp_rank]
    return req


def _adder(tp_rank, gather, *, deliverable):
    alloc = _allocator(AVAILABLE)
    rb = MagicMock()
    rb.reqs = []
    adder = PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(alloc, deliverable=deliverable),
        token_to_kv_pool_allocator=alloc,
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=16384,  # the 12647-token rest after the load-back fits one chunk
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
        m, "this_rank_follows", return_value=tp_rank != 0
    ):
        adder.form_a_admission_follow = _scheduler(gather, tp_rank)._form_a_admission_follow_fn()
    return adder


def _group_gate(*, deliverable):
    """One gate call for weg2-32-93 on TP0/1/2 at once; per rank the result
    or the exception it raised."""
    gather = _Gather()
    out, adders, reqs = {}, {}, {}
    with patch(TOKEN_CUT, return_value=True):
        for r in range(3):
            adders[r] = _adder(r, gather, deliverable=deliverable)
            reqs[r] = _req(r)
        # the gate passed on the metal: the price fits the REPORTED budget
        assert adders[0].rem_total_tokens > EXTENT[0] + UNCACHED + 64

        def _run(r):
            try:
                out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
            except BaseException as e:  # noqa: BLE001 -- the result under test
                out[r] = e

        with patch.object(
            sp, "_pp_load_back_extent", side_effect=lambda q: q._h105d_extent
        ), patch("sglang.srt.mem_cache.common.release_admission_acquired_mamba_slot"):
            threads = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
    return out, adders, reqs, gather


def _refused():
    """The fix's own counter; None on a base without it (the RED run fails on
    the behaviour, not on a missing name)."""
    return getattr(sp, "_H105D_REFUSED", {"n": None})["n"]


class TokenCutLoadBackRoomTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        if hasattr(sp, "_H105D_REFUSED"):
            sp._H105D_REFUSED["n"] = 0

    def test_y9nf6_undeliverable_evictable_keeps_the_rid_queued(self):
        """06:32:52Z: the peel pays 7616 of 237568 -- no rank can hold 121856
        rows; the group must WAIT, not die."""
        out, adders, _, gather = _group_gate(deliverable=DELIVERED)
        for r in range(3):
            self.assertNotIsInstance(
                out[r], m.FormAAdmissionSplit,
                f"TP{r} died: {out[r]} (the y9nf6 RANK-DEATH)",
            )
            self.assertEqual(out[r], AddReqResult.NO_TOKEN, f"TP{r}: {out[r]}")
            self.assertEqual(adders[r].can_run_list, [])
            # nobody loaded back: the vote came first and was NO_TOKEN
            self.assertEqual(adders[r].tree_cache.load_backs, [])
            # the room attempt ran before the vote: shortfall asked, 7616 paid
            self.assertEqual(adders[r].tree_cache.evicts, [(KV_ROWS - AVAILABLE, DELIVERED)])
        self.assertEqual(gather.calls, {0: 1, 1: 1, 2: 1})
        self.assertEqual(_refused(), 3)

    def test_deliverable_room_admits_and_the_follow_load_back_serves(self):
        """The peel CAN pay the shortfall on every rank: ADMIT as before, and
        H105c's follow load-back finds the room the vote already made (H98e
        on, the default: on its first attempt; off: on the retry after the
        floor's refusal)."""
        from sglang.srt.environ import envs

        for room_first, seen in ((False, [False, True]), (True, [True])):
            with self.subTest(room_first=room_first), \
                    envs.SGLANG_WEG2_ENABLE_FORM_A_ADMIT_ROOM_FIRST.override(room_first):
                if hasattr(sp, "_H105D_REFUSED"):
                    sp._H105D_REFUSED["n"] = 0
                out, adders, reqs, gather = _group_gate(deliverable=REPORTED_EVICTABLE)
                for r in range(3):
                    self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]}")
                    self.assertEqual([q.rid for q in adders[r].can_run_list], [RID])
                    self.assertEqual(adders[r].tree_cache.load_backs, seen)
                    self.assertEqual(len(reqs[r].prefix_indices), KV_ROWS)
                self.assertEqual(gather.calls, {0: 1, 1: 1, 2: 1})
                self.assertIn(_refused(), (0, None))  # no refusal (None: a base without the counter)

    def test_one_tight_rank_refuses_for_the_group(self):
        """Only TP2's peel is short: its NO_TOKEN is the group's MIN, every rank
        keeps the rid queued and none loads back."""
        gather = _Gather()
        out, adders, reqs = {}, {}, {}
        with patch(TOKEN_CUT, return_value=True):
            for r in range(3):
                adders[r] = _adder(
                    r, gather, deliverable=DELIVERED if r == 2 else REPORTED_EVICTABLE
                )
                reqs[r] = _req(r)

            def _run(r):
                try:
                    out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
                except BaseException as e:  # noqa: BLE001
                    out[r] = e

            with patch.object(
                sp, "_pp_load_back_extent", side_effect=lambda q: q._h105d_extent
            ), patch("sglang.srt.mem_cache.common.release_admission_acquired_mamba_slot"):
                ts = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join(30)
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.NO_TOKEN, f"TP{r}: {out[r]}")
            self.assertEqual(adders[r].tree_cache.load_backs, [])
        self.assertEqual(_refused(), 1)


class LoadBackRowsTest(unittest.TestCase):
    def test_rows_are_the_evicted_run_only(self):
        dev = SimpleNamespace(evicted=False, parent=None, component_data={})
        mid = SimpleNamespace(
            evicted=True, parent=dev,
            component_data={BASE_COMPONENT_TYPE: SimpleNamespace(host_value=torch.arange(12288))},
        )
        leaf = SimpleNamespace(
            evicted=True, parent=mid,
            component_data={BASE_COMPONENT_TYPE: SimpleNamespace(host_value=torch.arange(109568))},
        )
        self.assertEqual(sp._h105d_load_back_kv_rows(leaf), KV_ROWS)
        self.assertEqual(sp._h105d_load_back_kv_rows(dev), 0)

    def test_off_the_token_cut_untouched(self):
        """No cut: the probe never evicts and never refuses (H105b/H106 decide)."""
        tc = MagicMock()
        adder = SimpleNamespace(tree_cache=tc, token_to_kv_pool_allocator=_allocator(0))
        req = SimpleNamespace(rid=RID, best_match_node=_best_match_node())
        with patch(TOKEN_CUT, return_value=False), patch.object(
            sp, "_pp_load_back_extent", return_value=KV_ROWS
        ):
            self.assertTrue(sp._h105d_cut_load_back_room(adder, req))
        tc.evict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
