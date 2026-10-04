"""Q-710 / Auftrag 1090: 'Prefill out of memory' in prepare_for_extend hands the batch back.

Specimen NF y9nf boot ...10040027, P PP0 00:42:31Z: the pass admitted three requests, their
admission spent rows (load-backs), locks and slots, ``alloc_for_extend`` then found 2816 rows
for 13738 tokens -> RuntimeError -> RANK-DEATH. Q-700 prices that specimen at admission; this
pins the second line of defence with FAULT INJECTION: a batch whose ``prepare_for_extend``
runs the real shape (request rows + mamba slots drawn first, then the token allocation fails)
and the real ``Scheduler._weg2_prepare_for_extend_or_hand_back`` + ``oom_rollback`` around it.

Pinned: nothing leaks (rows, mamba slots incl. the COW slot, admission locks), the resident
chunked request keeps its row and its place, the loads have landed before their pins go, the
requests are back at the head of the queue in admission order, the PP wire of the dead pass is
gone, every refusal re-raises the ORIGINAL error (the old rank death), and the dual / flip
paths never see any of it.
"""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache import common as cm  # noqa: E402
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    HybridReqToTokenPool,
    ReqToTokenPool,
)
from sglang.srt.weg2 import oom_rollback as rb  # noqa: E402


class _Mamba:
    def __init__(self, n):
        self.free_ids = list(range(1, n + 1))
        self.freed = []

    def take(self):
        return torch.tensor([self.free_ids.pop(0)], dtype=torch.int64)

    def free(self, t):
        for v in t.reshape(-1).tolist():
            assert v not in self.freed or v in self.free_ids, f"double free of mamba slot {v}"
            self.freed.append(v)
            self.free_ids.append(v)


class _Pool(ReqToTokenPool):
    """The real row pool + the real hybrid ``_rollback_alloc`` (what alloc() itself undoes with)."""

    _rollback_alloc = HybridReqToTokenPool._rollback_alloc

    def __init__(self, rows, mamba):
        super().__init__(rows, 8, "cpu", False)
        self.mamba_allocator = mamba


class _Node:
    def __init__(self, name):
        self.name = name
        self.locks = 0


class _Ctl:
    def __init__(self, log):
        self.ack_load_queue = []
        self.log = log


class _Event:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def synchronize(self):
        self.log.append(f"sync:{self.name}")


class _Tree:
    """The unified-tree surface the rollback touches."""

    def __init__(self, log, pool=None):
        self.log = log
        self.req_to_token_pool = pool
        self.cache_controller = _Ctl(log)
        self.skips = []

    def _anchor_dec_skip(self, req, params):
        self.skips.append(req.rid)

    def dec_lock_ref(self, node, params=None, skip_swa=False):
        assert node.locks > 0, f"unlock of an unlocked node {node.name}"
        node.locks -= 1
        self.log.append(f"unlock:{node.name}")

    def loading_check(self):
        self.log.append("loading_check")
        self.cache_controller.ack_load_queue.clear()


class _Req:
    def __init__(self, rid, *, row=None, cow=False, node=None):
        self.rid = rid
        self.req_pool_idx = row
        self.mamba_pool_idx = None
        self.mamba_slot_acquired_this_admission = False
        self.mamba_ping_pong_track_buffer = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.mamba_loadback_anchor_adopted = False
        self.swa_uuid_for_lock = None
        self.session = None
        self.inflight_middle_chunks = 0
        self.kv_committed_len = 0
        self.last_node = node or _Node(rid)
        self.cow = cow

    def __repr__(self):
        return f"<{self.rid}>"


class _Batch:
    """prepare_for_extend: alloc_req_slots first (rows, mamba), then the token allocation."""

    def __init__(self, sched, reqs, oom):
        self.s, self.reqs, self.oom = sched, reqs, oom
        self.prepared = False

    def prepare_for_extend(self):
        pool, mamba = self.s.req_to_token_pool, self.s.req_to_token_pool.mamba_allocator
        rows = pool.alloc(self.reqs)
        assert rows is not None
        for r in self.reqs:
            if r.mamba_pool_idx is None:
                r.mamba_pool_idx = mamba.take()
            else:  # the COW slot the match drew: alloc() resets the stamp and carries it
                r.mamba_slot_acquired_this_admission = False
        if self.oom:
            raise cm.PrefillOutOfMemory(
                "Prefill out of memory. Try to lower your batch size.\n"
                "Try to allocate 13738 tokens.\nAvailable full tokens: 2816"
            )
        self.prepared = True


def _sched(*, tp=1, pp=3, pp_rank=0, chunked=None, queue=()):
    log = []
    mamba = _Mamba(16)
    pool = _Pool(8, mamba)
    s = types.SimpleNamespace(
        ps=types.SimpleNamespace(tp_size=tp, pp_size=pp, pp_rank=pp_rank),
        req_to_token_pool=pool,
        tree_cache=_Tree(log, pool),
        chunked_req=chunked,
        waiting_queue=list(queue),
        anchor_tails=None,
        _pp_admission_last_built_decision="DEAD-PASS",
        _pp_load_back_wire={"dead": 1},
        log=log,
    )
    s.run = types.MethodType(Scheduler._weg2_prepare_for_extend_or_hand_back, s)
    return s


def _admit(sched, names, *, cow=()):
    """What add_one_req leaves on a fresh request: lock on last_node, a COW slot for some."""
    reqs = []
    for n in names:
        r = _Req(n)
        r.last_node.locks = 1
        if n in cow:
            r.mamba_pool_idx = sched.req_to_token_pool.mamba_allocator.take()
            r.mamba_slot_acquired_this_admission = True
            r.mamba_cow_src_index = 5
            r.mamba_needs_clear = True
        reqs.append(r)
    return reqs


@pytest.fixture(autouse=True)
def _group_p(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_WEG2_OOM_ROLLBACK", "1")
    yield


def _call(sched, reqs, oom=True, chunked_before=None, adder=None):
    adder = adder or types.SimpleNamespace(new_anchor_tails=(), weg2_skip_extend_taken=False)
    batch = _Batch(sched, reqs, oom)
    return sched.run(batch, reqs, adder, chunked_before), batch


# ---- the injection: the specimen pass -----------------------------------------------------


def test_the_oom_hands_the_whole_fresh_batch_back_and_leaks_nothing():
    s = _sched(queue=["later-1"])
    rows0 = s.req_to_token_pool.available_size()
    reqs = _admit(s, ["weg2-10-83", "weg2-11-84", "weg2-12-86"], cow=("weg2-11-84",))
    s.waiting_queue = ["later-1"]
    s.tree_cache.cache_controller.ack_load_queue.append(
        (None, _Event(s.log, "load-88768"), [1])
    )
    ok, batch = _call(s, reqs)

    assert ok is False and not batch.prepared
    assert s.req_to_token_pool.available_size() == rows0          # every row came back
    assert all(r.req_pool_idx is None for r in reqs)
    m = s.req_to_token_pool.mamba_allocator
    assert sorted(m.free_ids) == list(range(1, 17))                # every mamba slot, COW incl.
    assert len(m.freed) == len(set(m.freed)) == 3                  # ... exactly once each
    assert all(r.last_node.locks == 0 for r in reqs)               # admission locks released
    assert all(r.mamba_pool_idx is None and not r.mamba_needs_clear for r in reqs)
    assert all(r.mamba_cow_src_index is None for r in reqs)
    assert s.waiting_queue == reqs + ["later-1"]                   # head, admission order
    assert s.chunked_req is None
    assert s._pp_admission_last_built_decision is None and s._pp_load_back_wire is None
    # the load landed BEFORE any pin went: no retry may read rows the copy has not filled
    assert s.log.index("sync:load-88768") < s.log.index("loading_check") < s.log.index(
        "unlock:weg2-12-86"
    )
    assert s._weg2_oom_rollbacks == {r.rid: 1 for r in reqs}


def test_the_retry_after_the_rollback_can_take_the_same_rows():
    s = _sched()
    reqs = _admit(s, ["a", "b"])
    _call(s, reqs)
    for r in reqs:
        r.last_node.locks = 1                                      # re-admitted by the next pass
    ok, batch = _call(s, reqs, oom=False)
    assert ok is True and batch.prepared


def test_healthy_pass_is_untouched():
    s = _sched()
    reqs = _admit(s, ["a"])
    ok, batch = _call(s, reqs, oom=False)
    assert ok is True and batch.prepared
    assert reqs[0].last_node.locks == 1 and reqs[0].req_pool_idx is not None
    assert not hasattr(s, "_weg2_oom_rollbacks")


# ---- the resident chunked request ------------------------------------------------------------


def test_the_resident_continuation_keeps_row_place_and_chunk_count():
    s = _sched()
    cont = _Req("weg2-12-86", row=None)
    cont.last_node.locks = 1
    cont.req_pool_idx = s.req_to_token_pool.alloc([cont])[0]
    cont.mamba_pool_idx = s.req_to_token_pool.mamba_allocator.take()
    cont.inflight_middle_chunks = 1 + 1                            # the pass's own increment
    s.chunked_req = cont                                           # truncated: stays chunked
    fresh = _admit(s, ["weg2-10-83"])
    row = cont.req_pool_idx
    ok, _ = _call(s, [cont] + fresh, chunked_before=cont)

    assert ok is False
    assert cont.req_pool_idx == row and cont.mamba_pool_idx is not None   # NOT freed (#616)
    assert cont.last_node.locks == 1
    assert s.chunked_req is cont and cont.inflight_middle_chunks == 1
    assert s.waiting_queue == fresh
    assert s._weg2_oom_rollbacks == {"weg2-10-83": 1}              # the innocent one is not counted


def test_a_final_chunk_continuation_that_the_adder_had_cleared_is_restored():
    s = _sched()
    cont = _Req("weg2-12-86")
    cont.kv_committed_len = 100
    cont.req_pool_idx = s.req_to_token_pool.alloc([cont])[0]
    cont.mamba_pool_idx = s.req_to_token_pool.mamba_allocator.take()
    s.chunked_req = None                                           # add_chunked_req returned None
    ok, _ = _call(s, [cont], chunked_before=cont)
    assert ok is False and s.chunked_req is cont and cont.req_pool_idx is not None
    assert s.waiting_queue == []


def test_a_new_chunked_request_is_unminted():
    s = _sched()
    fresh = _admit(s, ["weg2-11-84"])
    fresh[0].inflight_middle_chunks = 1
    s.chunked_req = fresh[0]                                       # adder.new_chunked_req adopted
    ok, _ = _call(s, fresh, chunked_before=None)
    assert ok is False and s.chunked_req is None
    assert fresh[0].inflight_middle_chunks == 0 and s.waiting_queue == fresh


# ---- every refusal is the old rank death, state untouched ------------------------------------


@pytest.mark.parametrize(
    "kw,why",
    [
        (dict(tp=3), "tp>1"),
        (dict(pp_rank=1), "PP follower"),
    ],
)
def test_refusals_reraise_the_original_oom_and_touch_nothing(kw, why):
    s = _sched(**kw)
    reqs = _admit(s, ["a", "b"])
    with pytest.raises(cm.PrefillOutOfMemory):
        _call(s, reqs)
    assert all(r.last_node.locks == 1 for r in reqs)               # the dying rank is left alone
    assert s.waiting_queue == [] and not hasattr(s, "_weg2_oom_rollbacks")


def test_anchor_tails_a_second_row_owner_and_sessions_refuse():
    s = _sched()
    reqs = _admit(s, ["a"])
    adder = types.SimpleNamespace(new_anchor_tails=("t",), weg2_skip_extend_taken=False)
    with pytest.raises(cm.PrefillOutOfMemory):
        _call(s, reqs, adder=adder)

    s = _sched()
    stray = _Req("stray")
    stray.kv_committed_len = 100
    stray.req_pool_idx = s.req_to_token_pool.alloc([stray])[0]     # owns a row, is not chunked_req
    with pytest.raises(cm.PrefillOutOfMemory):
        _call(s, [stray] + _admit(s, ["a"]), chunked_before=None)

    s = _sched()
    reqs = _admit(s, ["a"])
    reqs[0].session = object()
    with pytest.raises(cm.PrefillOutOfMemory):
        _call(s, reqs)


def test_a_tree_that_is_not_the_unified_tree_refuses():
    s = _sched()
    s.tree_cache = types.SimpleNamespace(cache_controller=None)
    with pytest.raises(cm.PrefillOutOfMemory):
        _call(s, _admit(s, ["a"]))


def test_a_rid_that_keeps_coming_back_ends_in_the_named_stop():
    s = _sched()
    cap = envs.SGLANG_WEG2_OOM_ROLLBACK_MAX.get()
    reqs = _admit(s, ["weg2-11-84"])
    for _ in range(cap):
        ok, _b = _call(s, reqs)
        assert ok is False
        reqs[0].last_node.locks = 1
    with pytest.raises(cm.PrefillOutOfMemory):
        _call(s, reqs)                                             # EXHAUSTED: old behaviour


def test_not_group_p_or_knob_off_is_the_old_rank_death(monkeypatch):
    for env, val in (("SGLANG_WEG2_GROUP", "D"), ("SGLANG_WEG2_GROUP", ""),
                     ("SGLANG_WEG2_OOM_ROLLBACK", "0")):
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
        monkeypatch.setenv("SGLANG_WEG2_OOM_ROLLBACK", "1")
        monkeypatch.setenv(env, val)
        s = _sched()
        reqs = _admit(s, ["a"])
        with pytest.raises(cm.PrefillOutOfMemory):
            _call(s, reqs)
        assert reqs[0].last_node.locks == 1 and s.waiting_queue == []


def test_a_non_oom_error_is_never_taken(monkeypatch):
    s = _sched()
    reqs = _admit(s, ["a"])
    b = _Batch(s, reqs, False)
    b.prepare_for_extend = lambda: (_ for _ in ()).throw(RuntimeError("something else"))
    with pytest.raises(RuntimeError, match="something else"):
        s.run(b, reqs, types.SimpleNamespace(new_anchor_tails=(), weg2_skip_extend_taken=False), None)


# ---- the named errors and the wiring ----------------------------------------------------------


def test_alloc_for_extend_names_the_oom_on_both_allocator_shapes(monkeypatch):
    batch = types.SimpleNamespace(
        maybe_evict_swa=lambda: None,
        reqs=[],
        prefix_lens=[],
        extend_lens=[],
        device="cpu",
        req_to_token_pool=None,
        tree_cache=None,
        extend_num_tokens=7,
    )
    monkeypatch.setattr(cm, "alloc_req_slots", lambda *a, **k: [])
    monkeypatch.setattr(cm, "_alloc_page_size", lambda b: 1)

    def _boom(*a, **k):
        raise cm.TokenSlotsExhausted("Out of memory. Try to allocate 7 tokens.")

    monkeypatch.setattr(cm, "alloc_token_slots", _boom)
    with pytest.raises(cm.PrefillOutOfMemory):
        cm.alloc_for_extend(batch)
    assert issubclass(cm.PrefillOutOfMemory, RuntimeError)
    assert issubclass(cm.TokenSlotsExhausted, RuntimeError)
    assert not issubclass(cm.TokenSlotsExhausted, cm.PrefillOutOfMemory)  # decode OOM stays itself


def test_the_scheduler_uses_the_guarded_call_and_snapshots_the_chunked_req():
    import inspect

    src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
    assert "new_batch.prepare_for_extend()" not in src          # no bare call left on this path
    assert "_weg2_prepare_for_extend_or_hand_back(" in src
    assert src.index("_oom_chunked_before = self.chunked_req") < src.index(
        "adder.add_chunked_req(self.chunked_req)"
    )
    # the dual lane has its own prepare_for_extend calls and is not routed through the hook
    lane = open(
        os.path.join(os.path.dirname(cm.__file__), "..", "model_executor", "dual_group_lane.py")
    ).read()
    assert "oom_rollback" not in lane and "_weg2_prepare_for_extend_or_hand_back" not in lane
