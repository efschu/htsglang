# SPDX-License-Identifier: Apache-2.0
"""NF design defect 1 (15.10.2026): P-INTAKE-STALL "need_tokens > rem_total_tokens" while the pipeline still computes.

THE SPECIMENS (h16r 39fa0976ce 110016 P log, h16p 226aad7105 133855 P log; every stall is named by PP0 in the FIRST pass
after a finish, ``held_s`` = the wall time between TWO passes of PP0, never a second of idle):

    11:47:19  weg2-74-213 need=163712 rem_total=104512 held_s=2.8   (pool_avail=104512, tracked_evictable=157632)
    11:52:21  weg2-86-234 need=171968 rem_total=63296  held_s=2.6   (pool_avail=63296,  tracked_evictable=198848)
    11:09:22  weg2-5-36   need=118272 rem_total=109632 held_s=1.1   (pool_avail=109760, tracked_evictable=152384)
    13:58:24  weg2-18-102 need=77120  rem_total=72704  held_s=5.6   (pool_avail=72704,  protected=173760: locked, in flight)
    16:18:59  weg2-160-1221 need=98304 rem_total=127936 held_s=1.4  (protected=86528: locked, in flight)

THE DEVIATION. ``IntakeStallWatch`` is the answer to "nothing runs, so nothing will ever free a token" (xsn272). The scheduler
fed it ``running_empty=True`` whenever ``running_batch`` was empty and ``chunked_req``/``anchor_tails`` unset -- but on a PP ring the
LAST chunk of the finishing request is a launched pass whose result PP0 has not consumed (``_pp_launched_pending``): its rows are
LOCKED until it returns (13:58, 16:18: protected rows, nothing evictable) or its follower stages have not republished their room
facts yet (11:47, 11:52: the PR cap still read the pre-finish room). PP0 passes only when the ring delivers, so the first refusal
inside that window and the first pass after it are seconds apart and the 1 s hold is over at once. The stall named a request that
the finishing neighbour's rows would have paid one pass later.

THE FIX (one place, every caller): the premise is read where the verdict is taken. ``_weg2_intake_stall_observe`` asks
``intake_stall.work_in_flight`` -- running batch, chunked continuation, anchor tails, launched-and-unreturned PP slots -- and while
anything is in flight the refusal is a WAIT (``watch.progress()``: the hold restarts at the first idle pass, the wedge hand-off
included). What is left at idle is named with its cause (``p_intake.stall_cause``).
"""
from __future__ import annotations

import ast
import logging
import os
from http import HTTPStatus
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import intake_stall as st  # noqa: E402

try:
    from sglang.srt.weg2 import p_intake  # noqa: E402
except ImportError:  # pragma: no cover
    p_intake = None

_SCHED_PY = os.path.join(os.path.dirname(os.path.dirname(st.__file__)), "managers", "scheduler.py")


class _AbortReq:
    def __init__(self, finished_reason=None, rid=None):
        self.finished_reason = finished_reason
        self.rid = rid


def _observe_fn():
    """Scheduler._weg2_intake_stall_observe compiled out of the shipped source (the module itself does not import without a GPU)."""
    src = open(_SCHED_PY).read()
    for cls in ast.parse(src).body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_weg2_intake_stall_observe":
                    g = {"AbortReq": _AbortReq, "HTTPStatus": HTTPStatus,
                         "logger": logging.getLogger("intake-inflight-probe"), "__name__": "intake_inflight_probe"}
                    if p_intake is not None:
                        g["_p_intake"] = p_intake
                    exec(compile(ast.Module(body=[fn], type_ignores=[]), _SCHED_PY, "exec"), g)
                    return g["_weg2_intake_stall_observe"]
    raise AssertionError("Scheduler._weg2_intake_stall_observe not found")


class _Alloc:
    def __init__(self, avail):
        self._a = avail

    def available_size(self):
        return self._a


class _Tree:
    """The terms the verdict and the cause read."""

    def __init__(self, evictable, *, protected=0, deliverable=None, room_cap=None, backups=0):
        self._e = evictable
        self._p = protected
        self._d = evictable if deliverable is None else deliverable
        self.weg2_pp_room_cap = room_cap
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_backup = {i: object() for i in range(backups)}

    def full_evictable_size(self):
        return self._e

    def evictable_size(self):
        return self._e

    def protected_size(self):
        return self._p

    def deliverable_evictable_size(self):
        return self._d


class _Clock:
    def __init__(self, monkeypatch):
        import time

        self.t = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: self.t)


def _p0(monkeypatch, *, need, avail, evictable, protected=0, deliverable=None, room_cap=None, backups=0,
        pool=262144, in_flight=()):
    from sglang.srt.managers.corridor_guard import GROUP_ENV

    monkeypatch.setenv(GROUP_ENV, "P")
    obj = SimpleNamespace()
    obj.ps = SimpleNamespace(pp_rank=0)
    obj._weg2_intake_watch = st.IntakeStallWatch(hold_s=1.0)
    obj.max_total_num_tokens = pool
    obj.token_to_kv_pool_allocator = _Alloc(avail)
    obj.tree_cache = _Tree(evictable, protected=protected, deliverable=deliverable, room_cap=room_cap, backups=backups)
    obj.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
    obj.chunked_req = None
    obj.anchor_tails = None
    obj._pp_launched_pending = set(in_flight)
    req = SimpleNamespace(rid="weg2-74-213", full_untruncated_fill_ids=[0] * need, prefix_indices=[])
    obj.waiting_queue = [req]
    obj.enable_hicache_storage = False
    obj.enable_hierarchical_cache = False
    sent = []
    obj.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda a, r: sent.append(a)))
    return _observe_fn(), obj, req, sent


def _stalled(sent):
    return [a for a in sent if st.is_intake_stall(a.finished_reason["message"])]


# ---------------------------------------------------------------------------
# 1. the five specimens: a finishing pipeline pass is not "nothing runs"
# ---------------------------------------------------------------------------

#: (label, need, avail, evictable, protected, room_cap, seconds between the refusal inside the window and the first idle pass)
_SPECIMENS = [
    ("11:47 weg2-74-213", 163712, 104512, 157632, 0, 104512, 2.8),
    ("11:52 weg2-86-234", 171968, 63296, 198848, 0, 63296, 2.6),
    ("11:09 weg2-5-36", 118272, 109760, 152384, 0, 109632, 1.1),
    ("13:58 weg2-18-102 (rows locked by the in-flight pass)", 77120, 72704, 0, 173760, None, 5.6),
    # 16:18: the refusal there was the adder's lifetime gate (total_tokens >= rem_total=127936 although need 98304 < it); the
    # verdict term that refused is not in the log, so the room cap below is MODELLED (an estimator reading under the need)
    ("16:18 weg2-160-1221 (rows locked by the in-flight pass; cap modelled)", 98304, 127936, 44160, 86528, 90000, 1.4),
]


@pytest.mark.parametrize("label,need,avail,evictable,protected,room_cap,gap", _SPECIMENS)
def test_a_refusal_inside_the_in_flight_window_does_not_age_into_a_stall(
        monkeypatch, label, need, avail, evictable, protected, room_cap, gap):
    """Pass 1 refuses while the last chunk is in flight; pass 2 (the ring delivered, ``gap`` s later) is the FIRST idle pass.
    RED on 3b1e746e0e: pass 2 names the stall (held_s=gap >= 1 s). GREEN: the hold starts at pass 2."""
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=need, avail=avail, evictable=evictable, protected=protected,
                                  room_cap=room_cap, in_flight={2})
    observe(obj, req, None)                       # pass 1: the finishing pass is still on the ring
    assert _stalled(sent) == [], label
    clock.t += gap
    obj._pp_launched_pending.clear()              # the ring delivered: PASS-TAIL finished=1
    observe(obj, req, None)                       # pass 2: first idle pass
    assert _stalled(sent) == [] and obj.waiting_queue == [req], label
    assert obj._weg2_intake_watch.stalls == 0, label


def test_the_follow_up_pass_with_fresh_room_facts_admits_instead_of_naming(monkeypatch):
    """11:47 end to end: after the window the followers' facts are fresh (cap gone), the verdict is FITS, the watch stands down."""
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=157632, room_cap=104512,
                                  in_flight={2})
    observe(obj, req, None)
    clock.t += 2.8
    obj._pp_launched_pending.clear()
    observe(obj, req, None)
    obj.tree_cache.weg2_pp_room_cap = None        # the followers republished (the cap is dropped / lifted)
    clock.t += 0.005
    observe(obj, req, None)
    clock.t += 2.0
    observe(obj, req, None)
    assert _stalled(sent) == [] and obj._weg2_intake_watch.stalls == 0


def test_the_wedge_hand_off_is_not_a_stall_while_a_pass_is_in_flight(monkeypatch):
    """``immediate=True`` (the admission-wedge hand-off) used to skip the hold altogether."""
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=157632, room_cap=104512,
                                  in_flight={1})
    observe(obj, req, None, note="gate=admission-wedge waiting=1 reason=x", immediate=True)
    assert sent == [] and obj.waiting_queue == [req]


# ---------------------------------------------------------------------------
# 2. what runs counts: every term of the predicate
# ---------------------------------------------------------------------------


def test_work_in_flight_names_every_term():
    f = st.work_in_flight
    base = dict(running_batch=SimpleNamespace(is_empty=lambda: True), chunked_req=None, anchor_tails=None,
                _pp_launched_pending=set())
    assert f(SimpleNamespace(**base)) == ""
    assert f(SimpleNamespace()) == ""             # a stand-in without any of the terms: nothing runs (xsn272 doubles)
    assert "running_batch" in f(SimpleNamespace(**{**base, "running_batch": SimpleNamespace(is_empty=lambda: False)}))
    assert "chunked_req" in f(SimpleNamespace(**{**base, "chunked_req": object()}))
    assert "anchor_tails" in f(SimpleNamespace(**{**base, "anchor_tails": [object()]}))
    assert "pp_launched_pending=[0, 2]" in f(SimpleNamespace(**{**base, "_pp_launched_pending": {2, 0}}))


def test_each_term_alone_holds_the_stall_back(monkeypatch):
    for attr, val in (("running_batch", SimpleNamespace(is_empty=lambda: False, reqs=[1])),
                      ("chunked_req", SimpleNamespace(rid="c")),
                      ("anchor_tails", [object()]),
                      ("_pp_launched_pending", {1})):
        observe, obj, req, sent = _p0(monkeypatch, need=200000, avail=50000, evictable=0)
        setattr(obj, attr, val)
        observe(obj, req, None, immediate=True)
        assert sent == [], attr


# ---------------------------------------------------------------------------
# 3. what is left at idle is a stall, and it is named with its cause
# ---------------------------------------------------------------------------


def test_a_real_idle_refusal_still_names_the_stall_after_the_hold(monkeypatch):
    """xsn272 stays: nothing in flight, the same refusal for the hold -> the stall (503, requeue, flip)."""
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=200000, avail=50000, evictable=0)
    observe(obj, req, None)
    clock.t += 0.5
    observe(obj, req, None)
    assert sent == []
    clock.t += 0.6
    observe(obj, req, None)
    assert len(_stalled(sent)) == 1 and obj.waiting_queue == []
    assert sent[0].finished_reason["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE


def test_the_hold_counts_idle_time_only(monkeypatch):
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=200000, avail=50000, evictable=0)
    observe(obj, req, None)                       # idle: hold starts
    clock.t += 0.9
    obj._pp_launched_pending.add(3)               # a pass is launched: the hold is gone
    observe(obj, req, None)
    clock.t += 0.9
    obj._pp_launched_pending.clear()
    observe(obj, req, None)                       # idle again: restarts, 0.9 s + 0.9 s of wall must not count
    assert sent == []
    clock.t += 1.05
    observe(obj, req, None)
    assert len(_stalled(sent)) == 1


@pytest.mark.skipif(p_intake is None, reason="no p_intake")
def test_the_stall_names_its_cause_estimator(monkeypatch):
    """Rows exist and are unlocked (free + reported >= need) but the payability estimators refuse them: THE deviation, named."""
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=157632, deliverable=0,
                                  room_cap=104512)
    observe(obj, req, None)
    clock.t += 1.1
    observe(obj, req, None)
    msg = sent[0].finished_reason["message"]
    assert st.is_intake_stall(msg) and "cause=estimator" in msg
    assert "free=104512" in msg and "evict=157632" in msg and "deliv=0" in msg and "cap=104512" in msg


@pytest.mark.skipif(p_intake is None, reason="no p_intake")
def test_cause_pool_and_cause_locked():
    c = p_intake.stall_cause
    mk = lambda **k: SimpleNamespace(tree_cache=_Tree(**k), token_to_kv_pool_allocator=_Alloc(k.pop("avail", 0)))  # noqa: E731
    s = SimpleNamespace(tree_cache=_Tree(0), token_to_kv_pool_allocator=_Alloc(50000))
    assert c(s, 200000, 262144).startswith("cause=pool")
    s = SimpleNamespace(tree_cache=_Tree(0, protected=173760), token_to_kv_pool_allocator=_Alloc(72704))
    assert c(s, 200000, 262144).startswith("cause=locked")
    s = SimpleNamespace(tree_cache=_Tree(157632, deliverable=0), token_to_kv_pool_allocator=_Alloc(104512))
    assert c(s, 163712, 262144).startswith("cause=estimator")
    assert c(SimpleNamespace(), 10, 100).startswith("cause=unread")      # a stand-in answers, never raises
    del mk


# ---------------------------------------------------------------------------
# 4. K9b stays: an open backup of an IDLE P is still a wait (the backup window can outlast the hold)
# ---------------------------------------------------------------------------


def test_k9b_still_waits_on_open_backups_at_idle(monkeypatch):
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=0, room_cap=104512, backups=3)
    for _ in range(4):
        observe(obj, req, None)
        clock.t += 1.5
    assert sent == [] and obj._weg2_intake_watch.stalls == 0
    obj.tree_cache.ongoing_backup.clear()          # the backups are done and the rows are still not payable
    observe(obj, req, None)
    clock.t += 1.1
    observe(obj, req, None)
    assert len(_stalled(sent)) == 1


# ---------------------------------------------------------------------------
# 5. mutants: the tests above must notice
# ---------------------------------------------------------------------------


def test_mutant_work_in_flight_blind_is_caught(monkeypatch):
    """In-process mutant: the predicate says '' always (= the base). The 11:47 specimen then names the stall."""
    monkeypatch.setattr(st, "work_in_flight", lambda scheduler: "")
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=157632, room_cap=104512,
                                  in_flight={2})
    observe(obj, req, None)
    clock.t += 2.8
    obj._pp_launched_pending.clear()
    observe(obj, req, None)
    assert len(_stalled(sent)) == 1, "the mutant must reproduce the defect"


def test_mutant_launched_pending_ignored_is_caught(monkeypatch):
    real = st.work_in_flight

    def blind(scheduler):
        return real(SimpleNamespace(running_batch=getattr(scheduler, "running_batch", None),
                                    chunked_req=getattr(scheduler, "chunked_req", None),
                                    anchor_tails=getattr(scheduler, "anchor_tails", None)))

    monkeypatch.setattr(st, "work_in_flight", blind)
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=157632, room_cap=104512,
                                  in_flight={2})
    observe(obj, req, None)
    clock.t += 2.8
    obj._pp_launched_pending.clear()
    observe(obj, req, None)
    assert len(_stalled(sent)) == 1


def test_mutant_busy_does_not_restart_the_hold_is_caught(monkeypatch):
    """In-process mutant: the predicate fires but the hold is NOT restarted (``progress`` a no-op): the refusal inside the
    window still ages and the first idle pass names the stall -- the 11:47 specimen must go red."""
    monkeypatch.setattr(st.IntakeStallWatch, "progress", lambda self: None)
    clock = _Clock(monkeypatch)
    observe, obj, req, sent = _p0(monkeypatch, need=163712, avail=104512, evictable=157632, room_cap=104512,
                                  in_flight={2})
    # the mutant keeps a hold that began inside the window: seed it the way ``observe`` would have before the gate existed
    obj._weg2_intake_watch._rid = req.rid
    obj._weg2_intake_watch._since = clock.t
    observe(obj, req, None)
    clock.t += 2.8
    obj._pp_launched_pending.clear()
    observe(obj, req, None)
    assert len(_stalled(sent)) == 1, "the mutant must reproduce the defect"


def test_every_caller_goes_through_the_one_observe():
    """The premise lives in ``_weg2_intake_stall_observe`` itself: the three scheduler callers and the wedge hand-off pass through it."""
    src = open(_SCHED_PY).read()
    i = src.index("def _weg2_intake_stall_observe")
    body = src[i:src.index("def _abort_request_now", i)]
    assert "work_in_flight(self)" in body
    assert body.index("work_in_flight(self)") < body.index("intake_phase_verdict")
    assert body.index("work_in_flight(self)") < body.index("watch.observe(")
