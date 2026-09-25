"""H91a (25.09.2026): group P admits its queued backlog in order.

cu130 acceptance dkrnfbar1agent09252021 (1a4e6ff19a), P log 20:40:49-20:42:10:
rid 53572be2 (need 42,442) sat 61 s in P's queue with nothing running,
pool_avail=220480 and 41,664 rows evictable (0 locked), and was answered 503
WEG2-INTAKE-STALL by the admission-wedge path -- one extra flip.

Two behaviours, each red on 0d5a086e59:

1. #1400's told verdict survives an admission visit that did not seat the
   request (it was consumed there; every later visit skipped the rid as
   ``weg2_store_told_pending`` for ever).
2. A refused request that fits free + evictable rows is NOT an intake stall
   (no 503); the stall stays for what no row of the phase can fund.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from http import HTTPStatus
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_store_told as told_mod  # noqa: E402
from sglang.srt.weg2 import intake_stall as st  # noqa: E402

# The scheduler module is NOT imported: without a GPU its import dies in the
# Triton device probe (torchao -> torch.utils._triton, IndexError). The shipped
# source is read from disk instead, and the one method under test is compiled
# out of it -- the base and the fix are measured through the same door.
_SCHED_PY = os.path.join(os.path.dirname(told_mod.__file__), "scheduler.py")


def _scheduler_src() -> str:
    with open(_SCHED_PY) as f:
        return f.read()


class _AbortReq:
    def __init__(self, finished_reason=None, rid=None, **kw):
        self.finished_reason = finished_reason
        self.rid = rid


def _scheduler_method(name):
    src = _scheduler_src()
    for cls in ast.parse(src).body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    g = {"AbortReq": _AbortReq, "HTTPStatus": HTTPStatus,
                         "logger": logging.getLogger("h91a-probe"), "__name__": "h91a_probe"}
                    try:
                        from sglang.srt.weg2 import p_intake
                        g["_p_intake"] = p_intake
                    except ImportError:  # the base has no such module
                        pass
                    exec(compile(ast.Module(body=[fn], type_ignores=[]), _SCHED_PY, "exec"), g)
                    return g[name]
    raise AssertionError(f"Scheduler.{name} not found")


# --------------------------------------------------------------------------
# 1. the told verdict is kept until the request leaves the queue
# --------------------------------------------------------------------------


class _Tree:
    """A PP0 tree double whose store read terminated with ``completed``."""

    def __init__(self):
        self.prefetch_loaded_tokens_by_reqid = {}
        self._completed = {}

    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return self._completed.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, pp_rank=0):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = _Tree()
        self.waiting_queue = []
        self.pp_flip_counters = None

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        return "issued"


def _scheduler_told_gate():
    """The callable the scheduler's admission loop applies at the #1400 told
    site, resolved from the shipped source (so the base and the fix are
    measured through the same door)."""
    src = _scheduler_src()
    i = src.index("if self.enable_hicache_storage and _pp_group and _told_armed:")
    blk = src[i:i + 900]
    m = re.search(r"_told_loaded = ([\w\.]+)\(", blk)
    assert m, blk
    name = m.group(1)
    if name == "weg2_store_told.admission":
        return told_mod.admission
    assert name == "_p_intake.told_admission", name
    from sglang.srt.weg2 import p_intake

    return lambda s, r, note: p_intake.told_admission(s, r, note, told_mod.admission)


def _published(sched, rid, completed):
    req = SimpleNamespace(rid=rid, prefetch_deferred=None, origin_input_ids=None)
    sched.waiting_queue.append(req)
    assert told_mod.armed(sched)
    told_mod.intake(sched, req, lambda k: None)
    sched.tree_cache._completed[rid] = completed
    wire = told_mod.pp0_publish(sched, [])
    assert [(w.rid, w.told) for w in wire] == [(rid, completed)]
    return req


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(told_mod.ENV_ARMED, raising=False)
    yield


def test_the_told_verdict_survives_a_visit_the_adder_could_not_seat():
    """53572be2: told published at 20:40:58, the visit in the same pass ran
    beside d762c570's chunk and was not seated; with the verdict consumed
    there the rid skipped as weg2_store_told_pending until the 503."""
    gate = _scheduler_told_gate()
    s = _Sched(0)
    req = _published(s, "53572be2", completed=0)  # #1416 anchor clamp -> 0
    skips = []
    note = lambda kind, rid: skips.append(kind)  # noqa: E731
    assert gate(s, req, note) == 0            # visit 1: verdict taken ...
    assert req._weg2_prefix_cap == 0
    # ... the adder did not seat it (chunk budget spent): it stays queued
    for _ in range(3):                         # every later visit admits
        assert gate(s, req, note) == 0
    assert told_mod.SKIP_TOLD_PENDING not in skips
    assert req._weg2_prefix_cap == 0


def test_a_kept_verdict_goes_with_the_request_and_a_fresh_told_replaces_it():
    from sglang.srt.weg2 import p_intake

    gate = lambda s, r, note: p_intake.told_admission(s, r, note, told_mod.admission)  # noqa: E731
    s = _Sched(0)
    req = _published(s, "aaaa0001", completed=4096)
    s.tree_cache.prefetch_loaded_tokens_by_reqid["aaaa0001"] = 4096
    assert gate(s, req, lambda k, r: None) == 4096
    assert gate(s, req, lambda k, r: None) == 4096      # kept, credit kept
    # the pass seats it: the queue loses it, settle drops the verdict
    s.waiting_queue.remove(req)
    assert p_intake.settle_told(s, s.waiting_queue) == 1
    assert getattr(s, p_intake.KEPT_ATTR) == {}
    # a NEW request object under the rid (re-intake) never inherits it
    skips = []
    other = SimpleNamespace(rid="aaaa0001", prefetch_deferred=None)
    assert gate(s, other, lambda k, r: skips.append(k)) is None
    assert skips == [told_mod.SKIP_TOLD_PENDING]
    # a fresh told for a still-queued request replaces the kept one
    s2 = _Sched(0)
    r2 = _published(s2, "bbbb0001", completed=0)
    assert gate(s2, r2, lambda k, r: None) == 0
    s2.tree_cache._completed["bbbb0001"] = 8192
    s2._weg2_store_told["bbbb0001"] = 8192               # re-published
    assert gate(s2, r2, lambda k, r: None) == 0          # credit popped fresh
    assert r2._weg2_prefix_cap == 8192
    assert getattr(s2, p_intake.KEPT_ATTR)["bbbb0001"].told == 8192


def test_the_scheduler_settles_kept_verdicts_where_the_queue_is_committed():
    src = _scheduler_src()
    i = src.index("self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]")
    assert "_p_intake.settle_told(self, self.waiting_queue)" in src[i:i + 300]


# --------------------------------------------------------------------------
# 2. the intake stall is only what the phase cannot fund
# --------------------------------------------------------------------------


class _Alloc:
    def __init__(self, avail):
        self._a = avail

    def available_size(self):
        return self._a


class _PoolTree:
    def __init__(self, evictable, wt=None):
        self._e = evictable
        self.ongoing_write_through = dict(wt or {})
        self.ongoing_load_back = {}

    def full_evictable_size(self):
        return self._e

    def evictable_size(self):
        return self._e


def _p0(monkeypatch, *, need, avail, evictable, pool=262144, wt=None):
    from sglang.srt.managers.corridor_guard import GROUP_ENV

    monkeypatch.setenv(GROUP_ENV, "P")
    obj = SimpleNamespace()
    obj.ps = SimpleNamespace(pp_rank=0)
    obj._weg2_intake_watch = st.IntakeStallWatch(hold_s=1.0)
    obj.max_total_num_tokens = pool
    obj.token_to_kv_pool_allocator = _Alloc(avail)
    obj.tree_cache = _PoolTree(evictable, wt)
    req = SimpleNamespace(rid="53572be2148b48c8ba8272f8c5dbe370",
                          full_untruncated_fill_ids=[0] * need, prefix_indices=[])
    obj.waiting_queue = [req]
    obj.enable_hicache_storage = False
    obj.enable_hierarchical_cache = False
    sent = []
    obj.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda a, r: sent.append(a)))
    return _scheduler_method("_weg2_intake_stall_observe"), obj, req, sent


def test_a_request_that_fits_free_plus_evictable_is_not_answered_503(monkeypatch):
    """The cu130 numbers: need 42,442 against free 220,480 + evictable 41,664,
    nothing running, the admission-wedge hand-off (immediate)."""
    observe, obj, req, sent = _p0(monkeypatch, need=42442, avail=220480, evictable=41664)
    observe(
        obj, req, None, note="gate=admission-wedge waiting=1 reason=phase-flip-off",
        immediate=True)
    assert sent == [] and obj.waiting_queue == [req]
    assert obj._weg2_intake_watch.stalls == 0
    # the backlog case: free rows short, the finished prefills' rows evictable
    observe, obj, req, sent = _p0(monkeypatch, need=42442, avail=10000, evictable=240000)
    observe(obj, req, None)
    observe(obj, req, None)
    assert sent == [] and obj.waiting_queue == [req]


def test_the_stall_stays_for_what_the_phase_cannot_fund(monkeypatch):
    # larger than the whole pool: refused at once, named too large
    observe, obj, req, sent = _p0(monkeypatch, need=300000, avail=262144, evictable=0)
    observe(obj, req, None)
    assert len(sent) == 1 and st.is_too_large(sent[0].finished_reason["message"])
    # fits the pool, but free + evictable cannot give it and nothing is in
    # flight: rows are held by something only the flip releases -> the stall
    observe, obj, req, sent = _p0(monkeypatch, need=200000, avail=50000, evictable=0)
    observe(obj, req, None, immediate=True)
    assert len(sent) == 1
    assert sent[0].finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
    assert st.is_intake_stall(sent[0].finished_reason["message"])
    assert obj.waiting_queue == []
    # the seat gate is not a token question: no free request slot with
    # nothing running stays the stall even when the rows would fit
    observe, obj, req, sent = _p0(monkeypatch, need=42442, avail=220480, evictable=0)
    observe(obj, req, None, note="gate=seats allocatable_reqs=0 req_slots_free=0 waiting=1",
            immediate=True)
    assert len(sent) == 1 and "gate=seats" in sent[0].finished_reason["message"]
    # same shortfall with a write-through in flight: its ack frees rows, wait
    observe, obj, req, sent = _p0(monkeypatch, need=200000, avail=50000, evictable=0, wt={7: 1})
    observe(obj, req, None, immediate=True)
    assert sent == [] and obj.waiting_queue == [req]


def test_intake_verdict_terms():
    v = st.intake_verdict
    assert v(need_tokens=42442, pool_tokens=262144, free_tokens=220480,
             evictable_tokens=41664, inflight=False) == st.INTAKE_FITS
    assert v(need_tokens=262144, pool_tokens=262144, free_tokens=0,
             evictable_tokens=262144, inflight=False) == st.INTAKE_FITS
    assert v(need_tokens=262145, pool_tokens=262144, free_tokens=262144,
             evictable_tokens=0, inflight=True) == st.INTAKE_IMPOSSIBLE
    assert v(need_tokens=100, pool_tokens=262144, free_tokens=50,
             evictable_tokens=0, inflight=True) == st.INTAKE_WAITS
    assert v(need_tokens=100, pool_tokens=262144, free_tokens=50,
             evictable_tokens=0, inflight=False) == st.INTAKE_IMPOSSIBLE
