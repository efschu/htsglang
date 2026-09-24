"""fnFL2 H42 (Task #118): several END-ANCHOR tails per P forward.

WHAT IS PINNED. With ``SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS`` (and the
END-ANCHOR group P arms), a burst of whole-fit prompts runs as ONE body
forward (each body ``[0, N-1)``) and ONE tail forward (each tail
``[N-1, N)``) instead of one request reaching its end per forward. Unarmed,
the stock path -- the body minted as THE chunked request, the second whole-fit
prompt refused (#967) -- is reproduced exactly, against a trace recorded on the
pre-H42 tree.

HARNESS. The REAL ``PrefillAdder`` (a private copy of ``schedule_policy``
evaluated with ``SGLANG_WEG2_END_ANCHOR=1``, as test_weg2_zero_remainder_1233
does, so no other file sees a reloaded module), driven by a fake PP3 group:
PP0 decides with ``add_one_req``; PP1/PP2 execute PP0's forwarded decision
per rid (``scheduled_extents`` + the carried last-chunk verdict, #791/#996)
through ``_add_scheduled_req``. Every rank holds its OWN Req objects (separate
processes on the rig); the scheduler coordination is the one
``managers/anchor_tails.py`` provides and ``scheduler.py`` calls. Results of a
pass are processed two passes later (PP3 lap), so every tail is re-added
while its body's result is still owed -- the realistic inflight order.
"""

import importlib.util
import os
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.srt.environ import envs
from sglang.srt.managers import anchor_tails as at
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

N_TOK = 4096
CHUNK = 16384
PP = 3
LAP = 2  # results of pass k are processed at pass k + LAP

_SP = None


def _sp():
    """A PRIVATE schedule_policy with the END-ANCHOR armed (see 1233's note on
    why the shared module is never reloaded)."""
    global _SP
    if _SP is None:
        import sglang.srt.managers.schedule_policy as sp

        old = os.environ.get("SGLANG_WEG2_END_ANCHOR")
        os.environ["SGLANG_WEG2_END_ANCHOR"] = "1"
        try:
            spec = importlib.util.spec_from_file_location(
                "sglang.srt.managers._schedule_policy_probe_h42", sp.__file__
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            if old is None:
                os.environ.pop("SGLANG_WEG2_END_ANCHOR", None)
            else:
                os.environ["SGLANG_WEG2_END_ANCHOR"] = old
        _SP = mod
    return _SP


def _tree_cache():
    tc = MagicMock()
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.evictable_size.return_value = 0
    tc.disable = False
    tc.supports_mamba.return_value = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(avail):
    al = MagicMock()
    al.full_available_size.return_value = avail
    al.swa_available_size.return_value = avail
    al.available_size.return_value = avail
    return al


def _running_batch():
    b = MagicMock()
    b.reqs = []
    b.batch_size.return_value = 0
    return b


def _req(rid, n):
    r = MagicMock(spec=Req)
    r.rid = rid
    r.priority = 0
    r.prefix_indices = []
    r.full_untruncated_fill_ids = list(range(n))
    r.origin_input_ids = list(range(n))
    r.output_ids = []
    r.host_hit_length = 0
    r.swa_host_hit_length = 0
    r.sampling_params = SimpleNamespace(max_new_tokens=1, ignore_eos=False)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.born_spilled = False
    r.born_spilled_deep = False
    r.last_node = None
    r.mamba_pool_idx = None
    r.inflight_middle_chunks = 0
    r.extend_range = None
    r.to_finish = None
    r.finished_reason = None
    r.finished.side_effect = lambda: r.finished_reason is not None
    r.needs_host_load_back.return_value = False
    r.set_extend_range = MagicMock(
        side_effect=lambda s, e: setattr(r, "extend_range", Range(s, e))
    )
    r.truncate_prefix_to = MagicMock(
        side_effect=lambda n: setattr(r, "prefix_indices", r.prefix_indices[:n])
    )
    return r


class FakeRank:
    """One P stage: the scheduler's prefill coordination around the real adder."""

    def __init__(self, sp, pp_rank, specs, *, use_tails=True):
        self.sp = sp
        self.pp_rank = pp_rank
        self.reqs = OrderedDict((rid, _req(rid, n)) for rid, n in specs)
        self.waiting = list(self.reqs.values())
        self.tails = []
        self.chunked = None
        self.use_tails = use_tails
        self.anchor_at = {}  # rid -> [stash ends]: where this rank's anchors land
        self.outputs = {}  # rid -> number of output tokens (must end at 1)
        self.batches = []
        #: H42b: PP0-side store-told readiness and the burst-assembly gate
        self.ready_fn = None
        self.hold_fn = None

    def _stash_one(self, r):
        self.anchor_at.setdefault(r.rid, []).append(r.extend_range.end)
        r.prefix_indices = list(range(r.extend_range.end))

    def run_pass(self, decision=None):
        # get_next_batch_to_run: stash the chunked request and every tail
        if self.chunked is not None and self.chunked.extend_range.end > len(
            self.chunked.prefix_indices
        ):
            self._stash_one(self.chunked)
        if self.use_tails:
            for t in at.stash_due(self.tails):
                self._stash_one(t)
        follower = decision is not None
        adder = self.sp.PrefillAdder(
            page_size=1,
            tree_cache=_tree_cache(),
            token_to_kv_pool_allocator=_allocator(10**7),
            running_batch=_running_batch(),
            new_token_ratio=1.0,
            rem_input_tokens=10**7,
            rem_chunk_tokens=CHUNK,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
            scheduled_extents=(
                {rid: (p, e) for rid, (p, e, _l) in decision.items()} if follower else None
            ),
            scheduled_last_chunk=(
                {rid: l for rid, (_p, _e, l) in decision.items()} if follower else None
            ),
        )
        incoming = {rid: p for rid, (p, _e, _l) in decision.items()} if follower else None
        readd = None
        if self.use_tails and self.tails:
            readd = at.readd_anchor_tails(self.tails, adder, incoming=incoming)
            self.tails = readd.kept
        if self.chunked is not None:
            self.chunked.init_next_round_input()
            if incoming is None or incoming.get(self.chunked.rid) is not None:
                self.chunked = adder.add_chunked_req(self.chunked)
        adder.chunked_req_outstanding = self.chunked is not None
        admit_from = list(self.waiting)
        if not follower and self.hold_fn is not None and self.hold_fn(self, adder):
            admit_from = []
        for r in admit_from:
            if follower and r.rid not in decision:
                continue
            if not follower and self.ready_fn is not None and not self.ready_fn(r):
                continue
            if any(x is r for x in adder.can_run_list):
                continue
            res = adder.add_one_req(r, truncation_align_size=None)
            if res.name != "CONTINUE":
                break
        can = list(adder.can_run_list)
        if follower:
            order = list(decision.keys())
            assert sorted(x.rid for x in can) == sorted(order), (
                "a follower must execute exactly the named rids",
                [x.rid for x in can],
                order,
            )
            can.sort(key=lambda x: order.index(x.rid))
        if adder.new_chunked_req is not None:
            assert self.chunked is None
            self.chunked = adder.new_chunked_req
        if self.chunked is not None:
            self.chunked.inflight_middle_chunks += 1
        bodies = ()
        if self.use_tails and (adder.new_anchor_tails or readd is not None):
            self.tails = at.adopt_anchor_tails(self.tails, adder.new_anchor_tails, can)
            bodies = at.bodies_in_batch(self.tails, can)
        self.waiting = [r for r in self.waiting if not any(r is x for x in can)]
        batch = SimpleNamespace(
            reqs=can,
            extents=[(x.rid, x.extend_range.start, x.extend_range.end) for x in can],
            contains_last=at.contains_last_prefill_chunk(can, self.chunked, bodies),
            bodies=tuple(x.rid for x in bodies),
        )
        if can:  # an empty pass builds no batch (scheduler: `return None`)
            self.batches.append(batch)
        out = OrderedDict()
        for x in can:
            s, e = x.extend_range.start, x.extend_range.end
            out[x.rid] = (s, e - s, e >= len(x.full_untruncated_fill_ids))
        return batch, out

    def process(self, batch):
        """process_batch_result_prefill's inflight branch, per request."""
        for r in batch.reqs:
            if r.finished() and r.inflight_middle_chunks <= 0:
                continue
            if r.inflight_middle_chunks <= 0:
                self.outputs[r.rid] = self.outputs.get(r.rid, 0) + 1
                if r.to_finish is not None:
                    r.finished_reason = r.to_finish
            else:
                r.inflight_middle_chunks -= 1


def run_group(specs, *, passes=12, use_tails=True, abort_after=None):
    sp = _sp()
    ranks = [FakeRank(sp, k, specs, use_tails=use_tails) for k in range(PP)]
    pending = []
    trace = []
    for k in range(passes):
        if abort_after is not None and k == abort_after[0]:
            for rk in ranks:
                for t in at.abort_targets(rk.tails, rid=abort_after[1], abort_all=False):
                    t.to_finish = "ABORT"
        while pending and pending[0][0] <= k:
            _, per_rank = pending.pop(0)
            for rk, b in zip(ranks, per_rank):
                rk.process(b)
        b0, decision = ranks[0].run_pass()
        if not b0.reqs:
            continue
        per_rank = [b0]
        for rk in ranks[1:]:
            b, _ = rk.run_pass(dict(decision))
            per_rank.append(b)
        trace.append(tuple(b0.extents))
        pending.append((k + LAP, per_rank))
    for _, per_rank in pending:
        for rk, b in zip(ranks, per_rank):
            rk.process(b)
    return ranks, trace


@pytest.fixture
def armed():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(True):
        yield


@pytest.fixture
def unarmed():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(False):
        yield


FOUR = [("A", N_TOK), ("B", N_TOK), ("C", N_TOK), ("D", N_TOK)]
EIGHT = FOUR + [("E", N_TOK), ("F", N_TOK), ("G", N_TOK), ("H", N_TOK)]


# ------------------------------------------------------------ the burst
def test_four_4k_prompts_share_one_16k_forward_and_their_tails_the_next(armed):
    ranks, trace = run_group(FOUR)
    assert trace[0] == tuple((rid, 0, N_TOK - 1) for rid in "ABCD"), trace
    assert trace[1] == tuple((rid, N_TOK - 1, N_TOK) for rid in "ABCD"), trace
    assert len(trace) == 2, "4 requests = 1 body forward + 1 tail forward"
    for rk in ranks:
        assert rk.chunked is None and rk.tails == []
        assert rk.batches[0].bodies == ("A", "B", "C", "D")
        assert rk.batches[1].bodies == ()


def test_followers_execute_the_same_extents_per_rid(armed):
    ranks, _ = run_group(FOUR)
    ref = [b.extents for b in ranks[0].batches]
    for rk in ranks[1:]:
        assert [b.extents for b in rk.batches] == ref
        assert [b.bodies for b in rk.batches] == [b.bodies for b in ranks[0].batches]
        assert [b.contains_last for b in rk.batches] == [
            b.contains_last for b in ranks[0].batches
        ]


def test_every_request_keeps_its_own_anchor_at_n_minus_1_and_ends_once(armed):
    ranks, _ = run_group(FOUR)
    for rk in ranks:
        assert rk.anchor_at == {rid: [N_TOK - 1] for rid in "ABCD"}, rk.anchor_at
        assert rk.outputs == {rid: 1 for rid in "ABCD"}, (rk.pp_rank, rk.outputs)
        for r in rk.reqs.values():
            assert r.inflight_middle_chunks == 0


def test_no_second_continuation_refusal_in_the_burst(armed):
    sp = _sp()
    before = dict(sp._SECOND_CONTINUATION_REFUSALS)
    run_group(FOUR)
    assert dict(sp._SECOND_CONTINUATION_REFUSALS) == before


def test_eight_prompts_mix_tails_bodies_and_one_chunked_continuation(armed):
    ranks, trace = run_group(EIGHT)
    for rk in ranks:
        assert rk.outputs == {rid: 1 for rid, _ in EIGHT}, (rk.pp_rank, rk.outputs)
        for rid, _ in EIGHT:
            assert rk.anchor_at[rid][-1] == N_TOK - 1, (rid, rk.anchor_at[rid])
        assert [b.extents for b in rk.batches] == [b.extents for b in ranks[0].batches]
    assert sum(len(t) for t in trace) >= 16
    assert len(trace) <= 4, trace
    for step in trace:
        assert sum(e - s for _, s, e in step) <= CHUNK


def test_abort_of_one_request_mid_burst(armed):
    ranks, trace = run_group(FOUR, abort_after=(1, "B"))
    assert trace[1] == tuple((rid, N_TOK - 1, N_TOK) for rid in "ABCD")
    for rk in ranks:
        assert rk.reqs["B"].finished_reason == "ABORT"
        for rid in "ACD":
            assert rk.reqs[rid].finished_reason is None
        assert rk.outputs == {rid: 1 for rid in "ABCD"}
        assert rk.tails == []


# ------------------------------------------------------------ unarmed = stock
#: Recorded on the pre-H42 tree (H37 applied: 1d95257feb + cba020cfd2) with
#: the same harness minus the tail calls: the body is THE chunked request,
#: B-D are refused (#967) one pass at a time.
STOCK_FOUR = [
    (("A", 0, 4095),),
    (("A", 4095, 4096), ("B", 0, 4095)),
    (("B", 4095, 4096), ("C", 0, 4095)),
    (("C", 4095, 4096), ("D", 0, 4095)),
    (("D", 4095, 4096),),
]


STOCK_EIGHT = [
    (("A", 0, 4095),),
    (("A", 4095, 4096), ("B", 0, 4095)),
    (("B", 4095, 4096), ("C", 0, 4095)),
    (("C", 4095, 4096), ("D", 0, 4095)),
    (("D", 4095, 4096), ("E", 0, 4095)),
    (("E", 4095, 4096), ("F", 0, 4095)),
    (("F", 4095, 4096), ("G", 0, 4095)),
    (("G", 4095, 4096), ("H", 0, 4095)),
    (("H", 4095, 4096),),
]


def test_unarmed_eight_is_the_recorded_stock_trace(unarmed):
    _, trace = run_group(EIGHT, passes=20)
    assert trace == STOCK_EIGHT


def test_unarmed_is_the_stock_path(unarmed):
    sp = _sp()
    before = sum(sp._SECOND_CONTINUATION_REFUSALS.values())
    ranks, trace = run_group(FOUR)
    assert trace == STOCK_FOUR
    for rk in ranks:
        assert rk.tails == [] and all(b.bodies == () for b in rk.batches)
    assert sum(sp._SECOND_CONTINUATION_REFUSALS.values()) > before


def test_unarmed_without_the_tail_calls_is_identical(unarmed):
    _, with_calls = run_group(FOUR, use_tails=True)
    _, without = run_group(FOUR, use_tails=False)
    assert with_calls == without == STOCK_FOUR


def test_end_anchor_off_disarms_the_switch():
    with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(True):
        assert at.multi_anchor_tails_armed(False) is False
        assert at.multi_anchor_tails_armed(True) is True
    with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(False):
        assert at.multi_anchor_tails_armed(True) is False


# ------------------------------------------------------------ pure rules
def test_anchor_body_rule():
    assert at.is_anchor_body(fill_len=4096, start=0, end=4095, grain=1)
    assert not at.is_anchor_body(fill_len=4096, start=0, end=4096, grain=1)
    assert not at.is_anchor_body(fill_len=4096, start=0, end=4094, grain=1)
    assert at.is_anchor_body(fill_len=4096, start=0, end=4032, grain=64)
    assert not at.is_anchor_body(fill_len=4096, start=4032, end=4032, grain=64)
    assert not at.is_anchor_body(fill_len=1, start=0, end=0, grain=1)


def test_contains_last_is_stock_without_tails():
    a, b, c = object(), object(), object()
    for can in ([a], [a, b], [b], [b, c]):
        for chunked in (None, a):
            stock = chunked is None or len(can) != 1
            assert at.contains_last_prefill_chunk(can, chunked, ()) == stock
    assert at.contains_last_prefill_chunk([b], None, (b,)) is False
    assert at.contains_last_prefill_chunk([b, c], None, (b, c)) is True


# ------------------------------------------------------------ void per tail
def _holder(tails_before, current):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    pool = SimpleNamespace(req_to_token=MagicMock())
    alloc = MagicMock()
    h = SimpleNamespace(
        pp_loop_size=PP,
        chunked_req=None,
        anchor_tails=list(tails_before),
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=alloc,
    )
    SchedulerPPMixin._pp_note_chunked_req_before_admission(h, 1)
    h.anchor_tails = list(current)
    return h


def test_void_restores_every_tail_and_drops_the_voided_mints():
    from sglang.srt.managers.scheduler_pp_mixin import (
        pp_restore_anchor_tails_after_void,
        pp_void_keeps_request,
    )

    tails = [_req(rid, N_TOK) for rid in "ABCD"]
    for t in tails:  # re-added final by the voided pass; body result still owed
        t.prefix_indices = list(range(N_TOK - 1))
        t.req_pool_idx = 0
        t.extend_range = Range(N_TOK - 1, N_TOK)
        t.inflight_middle_chunks = 1
    minted = _req("E", N_TOK)
    minted.extend_range = Range(0, N_TOK - 1)
    minted.inflight_middle_chunks = 1
    h = _holder(tails, [minted])
    before = pp_restore_anchor_tails_after_void(h, 1)
    assert list(before) == tails and h.anchor_tails == tails
    for t in tails:
        assert t.extend_range == Range(N_TOK - 1, N_TOK - 1)  # parked shape
        assert t.inflight_middle_chunks == 1  # the body's increment stays owed
        assert pp_void_keeps_request(t, set(), None, before)
    assert not pp_void_keeps_request(minted, set(), None, before)
    assert h.token_to_kv_pool_allocator.free.call_count == 4


def test_void_gives_back_a_truncated_readds_increment():
    from sglang.srt.managers.scheduler_pp_mixin import pp_restore_anchor_tails_after_void

    t = _req("A", N_TOK)
    t.prefix_indices = list(range(N_TOK - 1))
    t.req_pool_idx = 0
    t.extend_range = Range(N_TOK - 1, N_TOK)
    t.inflight_middle_chunks = 2  # body + this pass's (truncated) re-add
    h = _holder([t], [t])
    pp_restore_anchor_tails_after_void(h, 1)
    assert t.inflight_middle_chunks == 1


def test_void_without_tails_is_a_no_op():
    from sglang.srt.managers.scheduler_pp_mixin import pp_restore_anchor_tails_after_void

    h = _holder([], [])
    assert pp_restore_anchor_tails_after_void(h, 1) == ()
    assert h.anchor_tails == []
    assert h.token_to_kv_pool_allocator.free.call_count == 0


def test_tails_are_live_for_the_flip_authority_and_locations():
    from sglang.srt.managers.phase_flip_runtime import _live_reqs
    from sglang.srt.managers.scheduler_pp_mixin import pp_request_locations

    tails = [_req(rid, N_TOK) for rid in "AB"]
    h = SimpleNamespace(running_mbs=[], anchor_tails=tails, chunked_req=None, waiting_queue=[])
    assert all(any(t is x for x in _live_reqs(h)) for t in tails)
    assert set(pp_request_locations(h)) == {"A", "B"}


# ------------------------------------------------------------ launcher reading
def test_launch_line_names_the_tail_mode():
    import json

    from sglang.srt.weg2 import launcher as L

    def _argv():
        return L.argv_p(py="/nonexistent/python", model="/nonexistent/model",
                        budgets=[28208, 17840, 17168], s_gb=1, m_mib=600,
                        store_cfg=json.dumps({"max_size": "1"}), extra=[],
                        p_bs=4, p_max_total_tokens=1277631, draft_kv_on_p=False)

    with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(True):
        argv = _argv()
        with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(True):
            on = L.p_micro_batch_line(argv)
        with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(False):
            off = L.p_micro_batch_line(argv)
    assert "mode=undivided width=4 " in on and "anchor tail of its own request" in on
    assert "at most one request REACHES ITS END" in off


def test_capture_bound_follows_the_switch():
    from sglang.srt.weg2 import tail_handoff as th

    with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(False):
        assert th.capture_keep() == th.KEEP_RIDS
    with envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.override(True):
        # one capture / part-file set per request P holds until the flip
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", max_running_requests=6)
        )
        assert th.capture_keep() == 6
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        assert th.capture_keep() == th.KEEP_CAPTURES_MULTI_TAIL == 4


# ============================================================ H42b
# x153b (13:54:40-13:55:00, epoch 9): PP0 admitted rid 10 ALONE (fwd13, 7848
# tokens, 2.4 s) because rid 11's #1400 store verdict was published one pass
# later and rids 14/15 reached P's loop only after that forward; and rid 16
# (queued since 13:54:43) sat out fwd17 because the lone re-added tail of rid
# 15 was counted against the one free seat.


def test_count_arm_uses_the_fresh_admission_count():
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
    # the carried continuations (re-added tails, the chunked request) hold
    # their seats already: x153b fwd17 counted rid 15's lone tail against the
    # one seat rid 10 had freed, and rid 16 waited a pass
    assert ">= self._uniform_allocatable_reqs(running_bs, _head_inputs) + _carried_n:" in src
    assert "_carried_n = len(adder.can_run_list) if adder.multi_anchor_tails else 0" in src
    assert "_burst_hold = self._weg2_burst_assembly_hold(adder, running_batch)" in src
    loop = src.index("for req in self.waiting_queue:")
    assert src.index("if _burst_hold is not None:", loop) - loop < 80


def _v(**kw):
    base = dict(window_ms=300, now=10.0, carried=0, ready_arrivals=[9.95], ready_tokens=7849,
                pending=1, last_arrival=9.99, budget_tokens=CHUNK, seat_cap=4)
    base.update(kw)
    return at.burst_hold_verdict(**base)


def test_burst_verdict_rules():
    assert _v().hold and _v().reason == "assembling"  # the x153b fwd13 state
    assert _v(window_ms=0).reason == "off"
    assert _v(carried=2).reason == "carried"
    assert _v(ready_arrivals=[], ready_tokens=0).reason == "nothing-ready"
    assert _v(budget_tokens=None).reason == "no-chunk-budget"
    assert _v(ready_tokens=CHUNK).reason == "budget-full"
    assert _v(ready_arrivals=[9.95] * 4).reason == "seats-full"
    assert _v(ready_arrivals=[9.6]).reason == "window"
    assert _v(pending=0, last_arrival=9.9).reason == "quiet"  # 100 ms >= 75 ms
    assert _v(pending=0, last_arrival=9.99).hold  # a rid arrived 10 ms ago
    assert _v(pending=0, last_arrival=None).reason == "quiet"
    assert _v(pending=1, last_arrival=9.5).hold  # quiet, but a #1400 verdict is outstanding


def _holder_sched(queue, told, held, seats=4):
    return SimpleNamespace(
        waiting_queue=queue,
        _weg2_store_told_armed=True,
        _weg2_store_told=dict(told),
        _weg2_store_held=dict(held),
        get_num_allocatable_reqs=lambda running_bs: seats,
    )


def test_the_scheduler_method_holds_and_releases(monkeypatch):
    import time as _time

    from sglang.srt.managers.scheduler import Scheduler

    clock = [100.0]
    monkeypatch.setattr(_time, "monotonic", lambda: clock[0])
    a, b = _req("A", N_TOK), _req("B", N_TOK)
    h = _holder_sched([a, b], told={"A": 0}, held={"B": b})
    adder = SimpleNamespace(can_run_list=[], rem_chunk_tokens=CHUNK)
    rb = SimpleNamespace(reqs=[])
    with envs.SGLANG_WEG2_P_BURST_ASSEMBLY_MS.override(300):
        assert Scheduler._weg2_burst_assembly_hold(h, adder, rb) == "assembling"
        clock[0] += 0.05
        h._weg2_store_told["B"] = 0
        h._weg2_store_held.pop("B")
        assert Scheduler._weg2_burst_assembly_hold(h, adder, rb) == "assembling"  # not quiet yet
        clock[0] += 0.08
        assert Scheduler._weg2_burst_assembly_hold(h, adder, rb) is None  # quiet
        adder.can_run_list = [a]
        assert Scheduler._weg2_burst_assembly_hold(h, adder, rb) is None  # carried
    with envs.SGLANG_WEG2_P_BURST_ASSEMBLY_MS.override(0):
        fresh = SimpleNamespace(can_run_list=[], rem_chunk_tokens=CHUNK)
        assert Scheduler._weg2_burst_assembly_hold(h, fresh, rb) is None


def _burst_run(window_ms):
    """PP0 + followers, x153b's arrival shape: A and B reach P in pass 0 (A's
    store verdict at pass 1, B's at pass 2), C and D in pass 2 (verdicts at
    pass 3). An idle pass costs 10 ms, a forward 2.4 s."""
    sp = _sp()
    specs = [("A", N_TOK), ("B", N_TOK), ("C", N_TOK), ("D", N_TOK)]
    arrive = {"A": 0, "B": 0, "C": 2, "D": 2}
    told_at = {"A": 1, "B": 2, "C": 3, "D": 3}
    ranks = [FakeRank(sp, k, specs) for k in range(PP)]
    for rk in ranks:
        rk.waiting = []
    clock = [0.0]
    seen = {}
    k_now = [0]

    def ready(r):
        return told_at[r.rid] <= k_now[0]

    def hold(rk, adder):
        queued = list(rk.waiting)
        for r in queued:
            seen.setdefault(r.rid, clock[0])
        rdy = [r for r in queued if ready(r)]
        v = at.burst_hold_verdict(
            window_ms=window_ms, now=clock[0], carried=len(adder.can_run_list),
            ready_arrivals=[seen[r.rid] for r in rdy],
            ready_tokens=sum(len(r.full_untruncated_fill_ids) for r in rdy),
            pending=sum(1 for r in queued if not ready(r)),
            last_arrival=max(seen.values()) if seen else None,
            budget_tokens=CHUNK, seat_cap=4)
        return v.hold

    ranks[0].ready_fn = ready
    ranks[0].hold_fn = hold
    trace, pending = [], []
    for k in range(12):
        k_now[0] = k
        for rk in ranks:
            rk.waiting += [rk.reqs[rid] for rid, a in arrive.items() if a == k]
        while pending and pending[0][0] <= k:
            _, per_rank = pending.pop(0)
            for rk, b in zip(ranks, per_rank):
                rk.process(b)
        b0, decision = ranks[0].run_pass()
        if not b0.reqs:
            clock[0] += 0.01
            continue
        per_rank = [b0] + [rk.run_pass(dict(decision))[0] for rk in ranks[1:]]
        trace.append(tuple(b0.extents))
        pending.append((k + LAP, per_rank))
        clock[0] += 2.4
    for _, per_rank in pending:
        for rk, b in zip(ranks, per_rank):
            rk.process(b)
    return ranks, trace


def test_without_assembly_the_first_arrival_runs_alone(armed):
    _, trace = _burst_run(0)
    assert trace[0] == (("A", 0, N_TOK - 1),)  # the x153b fwd13 shape
    assert len(trace) >= 4


def test_assembly_runs_the_burst_as_one_body_forward(armed):
    ranks, trace = _burst_run(300)
    assert trace[0] == tuple((rid, 0, N_TOK - 1) for rid in "ABCD"), trace  # '#new-seq: 4'
    assert trace[1] == tuple((rid, N_TOK - 1, N_TOK) for rid in "ABCD"), trace
    assert len(trace) == 2
    for rk in ranks:
        assert rk.outputs == {rid: 1 for rid in "ABCD"}
        assert rk.anchor_at == {rid: [N_TOK - 1] for rid in "ABCD"}
