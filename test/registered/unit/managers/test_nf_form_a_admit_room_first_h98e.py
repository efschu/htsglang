"""H98e: the cause behind H98d -- a Form A load-back whose group ADMIT is
already taken decides its room from the LIVE pool on its first attempt, so it
no longer drains the device tail of a prefix the same pass has voted for.

THE DEATH (NF xc D, c5da548b7c, boot dkrnfint4h6ablxcbar1dauer10071818,
D log 841600-843160, 07.10.2026 21:37:20-21:37:29Z), one pass, in order:

1. the usable vote plants pdflip-130-2092 at 175104 (``MATCH-CENSUS-DEEP
   rid=pdflip-130-2092 reached=175104 accepted=175104``, its anchor on TP0's
   device);
2. pdflip-130-2093 is admitted first (``X-GATE rid=pdflip-130-2093 uncached=12
   ... verdict=admit``). Under the #239 token cut the H105 verdict is gathered
   BEFORE the load-back, and H105d let every rank vote ADMIT only because its
   LIVE pool held its own load-back rows. Its first load-back nevertheless
   went through the pass-published, ledger-charged floor, which refused, and
   the xsn285 branch drained EVERY evictable leaf (``EVICT-FRONTIER-CENSUS
   request=168832 delivered_before=53824`` on all three ranks). Right after
   it, H105c's retry decided from the live pool and needed nothing:
   ``H105c FORM-A FOLLOW-ROOM rid=pdflip-130-2093 kv_tokens=49280 evicted=0
   available=142464`` (142464 - 53824 = 88640 rows were free before the
   drain -- more than the 49280 it loads);
3. the drain took pdflip-130-2092's device tail; TP0's mamba arena was full
   (``ARENA-DROP ... slot_bytes=58834944 ... freed=0``), so the nodes went to
   the host KV-only (``P-FUND EVICT KV-ONLY n=19..22``, 960+320+256 = 1536)
   and the anchor at 175104 was gone;
4. pdflip-130-2092's admission on TP0 ended at 173568 < 175104 -> H98
   HOST-BELOW-GROUP (H98d now defers it; this test checks it never goes stale).

Driven through the REAL ``PrefillAdder.add_one_req`` (three ranks, the token
cut's gather), the REAL ``Scheduler._form_a_admission_follow_fn`` and the REAL
``UnifiedRadixCache.load_back`` (floor, follow-room and xsn285 branch); the
vote and pdflip-130-2092's admission use the H98 tree model and helpers of
``test_nf_form_a_stale_vote_h98d``. The published floor's value at 21:37:29 is
not in the log (``#1045 FLOOR PUBLISHED`` is sampled); any value below 49280
refuses -- STALE_FLOOR is that assumption.

RED on 6ac60fd1be (H98d present): every rank drains 168832, TP0 admits
pdflip-130-2092 at 173568 and records the stale pair. GREEN with H98e: no rank
evicts, every rank admits 175104.
"""

from __future__ import annotations

import importlib.util
import os
import threading
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
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
)
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _load_h98d():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "h98d_model", os.path.join(here, "test_nf_form_a_stale_vote_h98d.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H98 = _load_h98d()

ROOM_SWITCH = "FLLIPER_PDFLIP_ENABLE_FORM_A_ADMIT_ROOM_FIRST"
TOKEN_CUT = "flliper.srt.rank_role.form_a_token_cut_active"

#: pdflip-130-2093 (D log 21:37:29): 49280 KV rows to load on every rank
#: ('FOLLOW-ROOM ... kv_tokens=49280'); TP0's extent 67392, the workers'
#: 49280 ('H105c FORM-A FOLLOW LOAD-BACK ... extent=67392/49280').
RID_93 = "pdflip-130-2093"
KV_ROWS = 49280
EXTENT = {0: 67392, 1: 49280, 2: 49280}
UNCACHED_93 = 12  # 'X-GATE rid=pdflip-130-2093 uncached=12'
EVICTABLE = 168832  # 'EVICT-FRONTIER-CENSUS request=168832' = evictable_size()
DELIVERABLE = 53824  # '... delivered_before=53824': what the peel could pay
LIVE = 142464 - DELIVERABLE  # 88640 free before the drain ('available=142464' after)
STALE_FLOOR = 40000  # assumption: any published floor below KV_ROWS refuses


class _Gather:
    """The token cut's H105 gather: every rank's payload, in rank order."""

    def __init__(self, n=3):
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
    for name in ("_form_a_is_host", "_form_a_admission_follow_fn"):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


class _Rank:
    """One rank's cache for ``UnifiedRadixCache.load_back`` (the real method
    runs on this namespace) plus the rank's peel: it pays at most DELIVERABLE
    tokens, and a peel that pays all of it has taken pdflip-130-2092's device
    tail -- on TP0 its anchor at 175104 leaves with it (P-FUND KV-ONLY, the
    mamba arena full). A shorter peel is not modelled (no order is claimed)."""

    def __init__(self, tp_rank, h98_tree, *, live):
        self.tp_rank = tp_rank
        self.h98_tree = h98_tree
        self.evicts = []
        self.follow_room = []
        self.left = DELIVERABLE
        t = SimpleNamespace()
        self.t = t
        t.token_to_kv_pool_allocator = _allocator(live)
        t.uniform_avail_floor = STALE_FLOOR
        t.uniform_admitted_since_floor = 0
        t._h105c_follow_room = False
        t.evictable_size = lambda: EVICTABLE - (DELIVERABLE - self.left)
        t.evict = self._evict
        kv_xfer = SimpleNamespace(host_indices=torch.arange(KV_ROWS), nodes_to_load=())
        base = MagicMock()
        base.build_hicache_transfers.return_value = [kv_xfer]
        t.cache_controller = MagicMock()
        t.cache_controller.load.side_effect = self._load
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
        t._record_store_event = lambda *a, **k: None
        t._update_evictable_leaf_sets = lambda *a: None
        t.ongoing_load_back = {}
        t.metrics_collector = None

    def _evict(self, params):
        got = min(int(params.num_tokens), self.left)
        self.left -= got
        self.evicts.append(int(params.num_tokens))
        self.t.token_to_kv_pool_allocator.available_size.return_value += got
        if self.left == 0 and self.tp_rank == 0:
            self.h98_tree.anchors = sorted(H98.HOST_AFTER_DRAIN)
        return SimpleNamespace(num_tokens_evicted=got)

    def _load(self, host_indices, node_id, extra_pools=None):
        alloc = self.t.token_to_kv_pool_allocator
        n = len(host_indices)
        if alloc.available_size.return_value < n:
            return None
        alloc.available_size.return_value -= n
        return torch.arange(n)


def _tree_cache(rank: _Rank):
    """The adder's tree: ``init_load_back`` runs the REAL ``load_back`` on the
    rank's namespace under the adder's follow-room flag; TP0 adopts the GDN
    anchor (the workers' walks are KV-only)."""
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    # one tree per rank: H105d's own room attempt peels the same leaves
    tc.evictable_size.side_effect = rank.t.evictable_size
    tc.full_evictable_size.side_effect = rank.t.evictable_size
    tc.evict.side_effect = rank.t.evict
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._pdflip_loadback_no_room = 0
    tc._h105c_follow_room = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()

    def _init_load_back(params):
        req = params.req
        rank.t._h105c_follow_room = bool(tc._h105c_follow_room)
        rank.follow_room.append(rank.t._h105c_follow_room)
        req.mamba_loadback_anchor_adopted = rank.tp_rank == 0
        ok = urc.UnifiedRadixCache.load_back(rank.t, params.best_match_node, None, req=req)
        n = EXTENT[rank.tp_rank] if ok else 0
        return torch.arange(n, dtype=torch.int64), req.last_node

    tc.init_load_back.side_effect = _init_load_back
    return tc


def _best_match_node():
    device_parent = SimpleNamespace(evicted=False, parent=None, component_data={})
    return SimpleNamespace(
        id=2669,
        evicted=True,
        parent=device_parent,
        component_data={
            BASE_COMPONENT_TYPE: SimpleNamespace(
                host_value=torch.arange(KV_ROWS, dtype=torch.int64)
            )
        },
    )


def _req_93(tp_rank):
    req = MagicMock(spec=Req)
    req.rid = RID_93
    req.priority = 0
    # the workers hold 67392 - 49280 rows on device ('#988 LOADBACK ... prefix
    # moved to 67392 ... extent=49280' on TP1/TP2), TP0 none ('device_len=0')
    req.prefix_indices = torch.arange(EXTENT[0] - EXTENT[tp_rank], dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(EXTENT[0] + UNCACHED_93))
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
    req._h98e_extent = EXTENT[tp_rank]
    return req


def _adder(rank: _Rank, gather):
    alloc = rank.t.token_to_kv_pool_allocator
    rb = MagicMock()
    rb.reqs = []
    adder = PrefillAdder(
        page_size=1,
        tree_cache=_tree_cache(rank),
        token_to_kv_pool_allocator=alloc,
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=16384,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
        m, "this_rank_follows", return_value=rank.tp_rank != 0
    ):
        adder.form_a_admission_follow = _scheduler(gather, rank.tp_rank)._form_a_admission_follow_fn()
    return adder


def _pass(*, room_first=True, live=LIVE):
    """One pass: the vote, pdflip-130-2093's gate on all three ranks at once,
    then pdflip-130-2092's admission match on every rank."""
    import contextlib

    from flliper.srt.environ import envs

    # read by NAME so this file runs on 6ac60fd1be too (no switch = the drain)
    field = getattr(envs, ROOM_SWITCH, None)
    room = field.override(room_first) if field is not None else contextlib.nullcontext()
    trees = H98._trees_at_vote()
    with H98._switches(defer=True):
        planted = H98._plant(trees)
    assert planted == {H98.RID: H98.GROUP}, planted  # the vote is right when taken
    ranks = {r: _Rank(r, trees[r], live=live) for r in range(3)}
    gather = _Gather()
    out, adders, reqs = {}, {}, {}
    with room, patch(
        TOKEN_CUT, return_value=True
    ), patch.object(m, "form_a_follow_active", return_value=True):
        for r in range(3):
            adders[r] = _adder(ranks[r], gather)
            reqs[r] = _req_93(r)

        def _run(r):
            try:
                out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
            except BaseException as e:  # noqa: BLE001 -- the result under test
                out[r] = e

        # patched ONCE around the threads (a patch inside a thread races its
        # peers' and can leave a mock behind)
        with patch.object(
            sp, "_pp_load_back_extent", side_effect=lambda q: q._h98e_extent
        ), patch(
            "flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"
        ), patch.object(urc, "_form_a_note_loaded"), patch(
            "flliper.srt.pdflip.pp_slot_fidelity.local_pp_room", return_value=None
        ), patch("flliper.srt.pdflip.pp_slot_fidelity.note_loaded"):
            threads = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
    with H98._switches(defer=True):
        geometry = {r: H98._admit(t, r, planted) for r, t in trees.items()}
    return out, adders, reqs, ranks, trees, geometry


class AdmitRoomFirstTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        H98._reset_module_state()

    def test_xc_2137_the_admitted_load_back_leaves_the_voted_prefix_alone(self):
        out, adders, reqs, ranks, trees, geometry = _pass()
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]}")
            self.assertEqual([q.rid for q in adders[r].can_run_list], [RID_93])
            self.assertEqual(len(reqs[r].prefix_indices), EXTENT[0], f"TP{r}")
            # base: [168832] -- the xsn285 drain of every evictable leaf
            self.assertEqual(ranks[r].evicts, [], f"TP{r} evicted for room it had")
        # base: {0: 173568, ...} and the H98d stale record (FormAHostBelowGroup
        # with the deferral off -- the 21:37:29 death)
        self.assertEqual(geometry, {0: H98.GROUP, 1: H98.GROUP, 2: H98.GROUP})
        self.assertIsNone(m.form_a_host_vote_stale(trees[0], H98._req()))

    def test_a_real_shortfall_costs_the_shortfall_not_the_drain(self):
        """The live pool is short: it is paid ONCE, by H105d's room attempt
        before the gather, and the load-back finds it (base: H105d pays the
        shortfall, then the floor refuses and drains the remaining ~150k).
        With H110 the room priced is the load-back rows PLUS the first chunk
        behind them (what is left of the extend after the rank's own
        load-back, cut to rem_chunk_tokens 16384, plus one page of 1): TP0
        (extent 67392) still has a full 16384 chunk behind its 49280-row
        load-back, TP1/TP2 (extent 49280) only 12 rows. The single eviction is
        therefore 35665 on TP0 and 19293 on TP1/TP2 -- one payment each, far
        below the EVICTABLE drain."""
        out, adders, reqs, ranks, _, _ = _pass(live=30000)
        h110_chunk_rows = {0: 16384 + 1, 1: 12 + 1, 2: 12 + 1}
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]}")
            self.assertEqual(
                ranks[r].evicts, [KV_ROWS + h110_chunk_rows[r] - 30000], f"TP{r}"
            )
            self.assertEqual(ranks[r].follow_room, [True], f"TP{r}")

    def test_switch_off_is_the_pre_h98e_drain(self):
        """Off: the first attempt is the floor's, the drain runs, H105c's retry
        loads -- and pdflip-130-2092's vote goes stale on TP0 (H98d defers it)."""
        out, adders, _, ranks, trees, geometry = _pass(room_first=False)
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]}")
            self.assertEqual(ranks[r].evicts, [EVICTABLE], f"TP{r}")
            self.assertEqual(ranks[r].follow_room, [False, True], f"TP{r}")
        self.assertEqual(geometry[0], H98.LOCAL)
        self.assertEqual(
            m.form_a_host_vote_stale(trees[0], H98._req()), (H98.LOCAL, H98.GROUP)
        )


if __name__ == "__main__":
    unittest.main()
