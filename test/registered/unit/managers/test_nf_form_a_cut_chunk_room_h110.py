"""H110: under the #239 token cut a rank's gate prices the rows THIS PASS
allocates for the request -- load-back + first extend chunk + the chunks the
adder already promised -- not the load-back alone.

THE DEATH (NF int15, boot commit 9341551eae; D log
boot_weg2_dkrnfint4h6ablxcbar1dauer10080122_9341551eae_1008_012219.D.log
437580-437746, 05:53:12Z, D awake since the P>D flip of 05:39:39Z, no flip
running):

* ``PDFLIP X-GATE rid=pdflip-44-438 uncached=4714 ... verdict=admit`` on TP0/1/2,
  ``#794 GROUP-NARROWED this prefill chunk from 4096 to 2368``;
* ``#988 LOADBACK rid=pdflip-44-438 prefix moved to 102144``,
  ``PDFLIP-START-LOADING tokens=86464`` on every rank -- H105d let it through:
  86464 load-back rows fit the ~87488 available (1024 left afterwards);
* the 2368-token chunk: ``Available full tokens: 73600 (full_available_size=1024
  + full_evictable_size_=72576)``, ``EVICTION UNDER-DELIVERED: asked for 1408
  tokens, the pool received 0`` -- the one frontier leaf (72576 tokens) is an
  un-backed write_back node whose backup the full host arena refused
  (``#1421 BACKUP-REFUSED why=arena_claim node=1130 tokens=72576``), and the UD
  drop is local-PP only;
* ``RuntimeError: Prefill out of memory`` on TP0/TP1/TP2 -> RANK-DEATH,
  front W17 PdFlipGroupDead 40 s later, deadman, container gone.

Driven through the REAL ``PrefillAdder.add_one_req`` and the REAL
``Scheduler._form_a_admission_follow_fn`` with the token cut's gather (three
ranks in threads). RED on 9341551eae: every rank ADMITs and loads back, leaving
1024 rows for a 2368-row chunk the peel cannot fund. GREEN with H110: every
rank votes NO_TOKEN, nobody loads back, the request stays queued.
"""

from __future__ import annotations

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
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
)
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

RID = "pdflip-44-438"
PROMPT = 106858  # 'PDFLIP D-IDS rid=pdflip-44-438 n=106858'
DEPTH = 102144  # '#988 LOADBACK ... prefix moved to 102144'
KV_ROWS = 86464  # 'PDFLIP-START-LOADING tokens=86464' on TP0/TP1/TP2
DEVICE_PREFIX = DEPTH - KV_ROWS  # 15680 device rows matched above the host run
CHUNK = 2368  # '#794 GROUP-NARROWED ... from 4096 to 2368'
PAGE = 64  # 'asked for 1408' = 2368 - 1024 + 64
LEFT_AFTER_LOAD = 1024  # 'full_available_size=1024'
AVAILABLE = KV_ROWS + LEFT_AFTER_LOAD  # 87488 before the load-back (derived)
REPORTED_EVICTABLE = 72576  # 'full_evictable_size_=72576'
TOKEN_CUT = "flliper.srt.rank_role.form_a_token_cut_active"


class _Gather:
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
    """REPORTS 72576 evictable; the peel pays only ``deliverable`` (05:53:12Z:
    0 -- the leaf's backup is refused for arena room)."""
    tc = MagicMock()
    st = {"reported": REPORTED_EVICTABLE, "deliverable": deliverable}
    tc.supports_mamba.return_value = False
    tc.evictable_size.side_effect = lambda: st["reported"]
    tc.full_evictable_size.side_effect = lambda: st["reported"]
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc._pdflip_loadback_no_room = 0
    tc._h105c_follow_room = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    tc.token_to_kv_pool_allocator = alloc
    tc.load_backs = []
    tc.evicts = []
    tc.st = st

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
        tc.load_backs.append(bool(tc._h105c_follow_room))
        avail = alloc.available_size.return_value
        if avail >= KV_ROWS:  # the load-back takes its rows at admission
            alloc.available_size.return_value -= KV_ROWS
            return (
                torch.arange(DEVICE_PREFIX + KV_ROWS, dtype=torch.int64),
                req.last_node,
            )
        req.mamba_loadback_anchor_adopted = True
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
                host_value=torch.arange(KV_ROWS, dtype=torch.int64)
            )
        },
    )


def _req(*, load_back=True):
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = torch.arange(
        DEVICE_PREFIX if load_back else DEPTH, dtype=torch.int64
    )
    req.full_untruncated_fill_ids = list(range(PROMPT))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.best_match_node = _best_match_node() if load_back else SimpleNamespace(
        evicted=False, parent=None, component_data={}
    )
    req.born_spilled = False
    req.born_spilled_deep = False
    req.extend_range = None
    req.set_extend_range.side_effect = lambda a, b: setattr(
        req, "extend_range", SimpleNamespace(start=a, end=b)
    )
    req._h110_extent = DEPTH if load_back else None
    return req


def _adder(tp_rank, gather, *, available, deliverable):
    alloc = _allocator(available)
    rb = MagicMock()
    rb.reqs = []
    adder = PrefillAdder(
        page_size=PAGE,
        tree_cache=_tree_cache(alloc, deliverable=deliverable),
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
        adder.form_a_admission_follow = _scheduler(
            gather, tp_rank
        )._form_a_admission_follow_fn()
    return adder


def _group_gate(*, available=AVAILABLE, deliverable=0, load_back=True):
    gather = _Gather()
    out, adders, reqs = {}, {}, {}
    with patch(TOKEN_CUT, return_value=True):
        for r in range(3):
            adders[r] = _adder(r, gather, available=available, deliverable=deliverable)
            reqs[r] = _req(load_back=load_back)
        # the metal's lifetime gate passed: the price fits the REPORTED budget
        assert adders[0].rem_total_tokens > KV_ROWS + CHUNK + 64 + PAGE

        def _run(r):
            try:
                out[r] = adders[r].add_one_req(reqs[r], truncation_align_size=None)
            except BaseException as e:  # noqa: BLE001 -- the result under test
                out[r] = e

        with patch.object(
            sp, "_pp_load_back_extent", side_effect=lambda q: q._h110_extent
        ), patch("flliper.srt.mem_cache.common.release_admission_acquired_mamba_slot"):
            threads = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
    return out, adders, reqs, gather


def _fundable_after(adder) -> int:
    """What ``alloc_for_extend`` can get for the chunk after the admission:
    the live pool plus what the peel can still pay."""
    tc = adder.tree_cache
    return int(adder.token_to_kv_pool_allocator.available_size()) + tc.st["deliverable"]


class ChunkRoomAfterLoadBackTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        sp._H105D_REFUSED["n"] = 0

    def test_0553_load_back_fits_but_chunk_does_not_keeps_the_rid_queued(self):
        """05:53:12Z: 86464 rows fit 87488, the 2368-row chunk behind them
        does not and the peel pays 0 -- the group must WAIT, not admit into
        'Prefill out of memory'."""
        out, adders, _, gather = _group_gate()
        for r in range(3):
            admitted = out[r] == AddReqResult.CONTINUE
            if admitted:  # the base's outcome, stated as the OOM it leads to
                self.assertGreaterEqual(
                    _fundable_after(adders[r]), CHUNK,
                    f"TP{r} admitted {RID} with {_fundable_after(adders[r])} fundable "
                    f"rows for a {CHUNK}-row chunk: 'Prefill out of memory' (05:53:12Z)",
                )
            self.assertEqual(out[r], AddReqResult.NO_TOKEN, f"TP{r}: {out[r]}")
            self.assertEqual(adders[r].can_run_list, [])
            self.assertEqual(adders[r].tree_cache.load_backs, [])
            # the room attempt now asks, BEFORE the vote, for the very 1408 the
            # chunk's allocation asked for after it (86464 + 2368 + 64 - 87488)
            self.assertEqual(
                adders[r].tree_cache.evicts, [(KV_ROWS + CHUNK + PAGE - AVAILABLE, 0)]
            )
            self.assertEqual(KV_ROWS + CHUNK + PAGE - AVAILABLE, 1408)
        self.assertEqual(gather.calls, {0: 1, 1: 1, 2: 1})
        self.assertEqual(sp._H105D_REFUSED["n"], 3)

    def test_room_for_load_back_and_chunk_admits(self):
        """Negative branch: the pool holds load-back + chunk + page -> ADMIT
        without any eviction, the chunk is fundable after the load-back."""
        out, adders, reqs, _ = _group_gate(available=KV_ROWS + CHUNK + PAGE)
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]}")
            self.assertEqual(adders[r].tree_cache.evicts, [])
            self.assertGreaterEqual(_fundable_after(adders[r]), CHUNK)
        self.assertEqual(sp._H105D_REFUSED["n"], 0)

    def test_deliverable_peel_admits(self):
        """The peel CAN pay the chunk's shortfall: the gate evicts it now and
        admits; the chunk is fundable after the load-back."""
        out, adders, _, _ = _group_gate(deliverable=REPORTED_EVICTABLE)
        for r in range(3):
            self.assertEqual(out[r], AddReqResult.CONTINUE, f"TP{r}: {out[r]}")
            self.assertGreaterEqual(_fundable_after(adders[r]), CHUNK)
        self.assertEqual(sp._H105D_REFUSED["n"], 0)


class ChunkRowsTest(unittest.TestCase):
    def test_rows_are_rest_after_load_back_cut_to_width_plus_page(self):
        adder = SimpleNamespace(rem_chunk_tokens=CHUNK, page_size=PAGE)
        req = SimpleNamespace(
            full_untruncated_fill_ids=list(range(PROMPT)),
            prefix_indices=torch.arange(DEVICE_PREFIX),
        )
        # rest 4714 cut to the narrowed width 2368, plus one page
        self.assertEqual(sp._h110_chunk_rows(adder, req, KV_ROWS), CHUNK + PAGE)
        # a short rest is taken whole
        adder.rem_chunk_tokens = 8192
        self.assertEqual(sp._h110_chunk_rows(adder, req, KV_ROWS), PROMPT - DEPTH + PAGE)
        # no chunking: the whole rest
        adder.rem_chunk_tokens = None
        self.assertEqual(sp._h110_chunk_rows(adder, req, KV_ROWS), PROMPT - DEPTH + PAGE)

    def test_promised_rows_count_the_pass_admissions(self):
        """Two requests admitted earlier in the pass hold 2368 + 512 rows that
        ``available_size()`` does not show yet; a desk double's non-int field
        counts nothing."""
        adder = SimpleNamespace(
            page_size=PAGE,
            can_run_list=[
                SimpleNamespace(extend_input_len=2368),
                SimpleNamespace(extend_input_len=512),
                SimpleNamespace(extend_input_len=MagicMock()),
            ],
        )
        self.assertEqual(sp._h110_promised_rows(adder), 2368 + 512 + 3 * PAGE)

    def test_promised_chunk_refuses_the_second_request(self):
        """No load-back, no evictable the peel can pay: the second request of
        the pass does not fit beside the first one's promised chunk."""
        alloc = _allocator(3000)
        tc = MagicMock()
        tc.evictable_size.return_value = REPORTED_EVICTABLE
        tc.evict.return_value = SimpleNamespace(num_tokens_evicted=0)
        adder = SimpleNamespace(
            tree_cache=tc, token_to_kv_pool_allocator=alloc, rem_chunk_tokens=CHUNK,
            page_size=PAGE, rem_total_tokens=0,
            can_run_list=[SimpleNamespace(extend_input_len=2000)],
        )
        req = SimpleNamespace(
            rid=RID, full_untruncated_fill_ids=list(range(1000)),
            prefix_indices=torch.arange(0),
        )
        sp._H105D_REFUSED["n"] = 0
        with patch(TOKEN_CUT, return_value=True), patch.object(
            sp, "_pp_load_back_extent", return_value=None
        ):
            self.assertFalse(sp._h105d_cut_load_back_room(adder, req))
            adder.can_run_list = []
            self.assertTrue(sp._h105d_cut_load_back_room(adder, req))


if __name__ == "__main__":
    unittest.main()
