"""H91 Teil B: the scheduler side of the D park (weg2/d_park_runtime.py).

* ``POST /weg2/park_running`` parks every running D request before D's sleep:
  retracted RETAINING the span (KV, GDN anchor, draft rows) with a forced host
  write-through, streams open, oldest first;
* the sleep's dormant point hands them to the #1443 hold FIRST;
* a decode-pressure retraction on D parks the YOUNGEST at the queue head;
* D's admission puts parked requests first and holds newcomers back;
* an abort reaches the parked list.
The collaborator runs against a stand-in scheduler; ``scheduler.py`` itself is
READ, not imported, by the wiring ratchets (its import pulls transformers ->
torchao -> the Triton device list, which a GPU-less desk does not have)."""
from __future__ import annotations

import os
import types
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.base_prefix_cache import FORCE_HOST_WRITE_THROUGH_ATTR  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)


def _req(rid, seq):
    return types.SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 10, output_ids=[0] * 3,
        is_fast_lane=False, spill_class=None,
    )


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.retract_calls = []

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        self.retract_calls.append((offload_kv, retain))
        out, self.reqs = self.reqs, []
        return out


class _Sched:
    def __init__(self, running=(), waiting=()):
        self.running_batch = _Batch(running)
        self.waiting_queue = list(waiting)
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace()
        self.weg2_dormant = False
        self.noted = []
        self.sent = []
        self.enable_hicache_storage = False
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: self.sent.append(o))
        )

    def _969ad_note_retract(self, req, site):
        self.noted.append((req.rid, site))

    def _add_request_to_queue(self, req, is_retracted=False):
        if self.weg2_dormant:  # the #1443 hold, as the real intake does it
            hold = getattr(self, "weg2_dormant_hold", None)
            if hold is None:
                hold = self.weg2_dormant_hold = []
            hold.append(req)
        else:
            self.waiting_queue.append(req)

    def _weg2_group_min_flags(self, flags):
        return [1 if f else 0 for f in flags]

    def uniform_min_avail(self):
        return 0


def _park(s, epoch=5):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason="d-to-p"))


def test_park_running_retracts_retaining_and_answers_the_rids_oldest_first():
    old, young, new = _req("old", 1), _req("young", 2), _req("new", 3)
    s = _Sched(running=[young, old], waiting=[new])
    out = _park(s)
    assert out.success and out.parked == ["old", "young"] and out.held == ["new"]
    assert out.epoch == 5
    assert s.running_batch.retract_calls == [(False, True)]  # retain=True, no D2H copy
    assert all(getattr(r, FORCE_HOST_WRITE_THROUGH_ATTR) for r in (old, young))
    assert ds.park_site(old) == ds.SITE_FLIP and getattr(old, ds.EPOCH_ATTR) == 5
    assert ds.park_site(new) is None  # never started: waits behind, not a park
    assert s.waiting_queue == []  # the sleep asserts an idle group
    assert [r.rid for r in s.weg2_d_parked] == ["old", "young", "new"]
    assert ("old", "weg2_park_running") in s.noted
    assert s.running_batch.batch_is_full is False


def test_park_running_refuses_off_group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    s = _Sched(running=[_req("a", 1)])
    out = _park(s)
    assert not out.success and out.parked == []
    assert s.running_batch.retract_calls == []


def test_park_running_on_a_dormant_group_lists_and_touches_nothing():
    s = _Sched(running=[])
    s.weg2_dormant = True
    s.weg2_d_parked = [_req("p", 1)]
    out = _park(s)
    assert out.success and out.parked == ["p"] and s.running_batch.retract_calls == []


def test_the_sleep_holds_the_parked_requests_first():
    old, young, new = _req("old", 1), _req("young", 2), _req("new", 3)
    s = _Sched(running=[young, old], waiting=[new])
    _park(s)
    s.weg2_dormant = True
    s.weg2_dormant_hold = [_req("earlier-held", 0)]
    assert rt.hold_parked(s, hold_armed=True) == 3
    assert [r.rid for r in s.weg2_dormant_hold] == ["old", "young", "new", "earlier-held"]
    assert s.weg2_d_parked == []


def test_an_unarmed_hold_keeps_them_parked_and_the_wake_requeues_them_first():
    a = _req("a", 1)
    s = _Sched(running=[a])
    _park(s)
    s.weg2_dormant = True
    assert rt.hold_parked(s, hold_armed=False) == 0
    assert [r.rid for r in s.weg2_d_parked] == ["a"]
    assert rt.park_tick(s) == 0  # still dormant: nothing moves
    s.weg2_dormant = False
    s.waiting_queue = [_req("woke-later", 9)]
    assert rt.park_tick(s) == 1
    assert [r.rid for r in s.waiting_queue] == ["a", "woke-later"]


def test_a_park_whose_sleep_never_came_rejoins_the_queue_head(monkeypatch):
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")
    a = _req("a", 1)
    s = _Sched(running=[a])
    _park(s)
    s.waiting_queue = [_req("later", 9)]
    assert rt.park_tick(s) == 0  # just parked: the front's sleep is coming
    setattr(a, ds.SINCE_ATTR, 0.0)  # ... it never came
    assert rt.park_tick(s) == 1
    assert [r.rid for r in s.waiting_queue] == ["a", "later"]


def test_a_decode_pressure_retraction_parks_the_youngest_at_the_head():
    young = _req("young", 2)
    other = _req("other", 7)
    s = _Sched(running=[_req("old", 1)], waiting=[other, young])
    assert rt.note_retracted(s, [young]) == 1
    assert ds.park_site(young) == ds.SITE_PRESSURE
    assert [r.rid for r in s.waiting_queue] == ["young", "other"]


def test_admission_puts_parked_first_and_holds_newcomers_back():
    old = _req("old", 1)
    young = _req("young", 2)
    ds.mark_parked(young, ds.SITE_PRESSURE)
    new = _req("new", 3)
    s = _Sched(running=[old], waiting=[new, young])
    gate = rt.admission(s, s.running_batch)
    assert [r.rid for r in s.waiting_queue] == ["young", "new"]
    assert gate.skip(young) == "weg2_d_park_older_live"
    assert gate.skip(new) == "weg2_d_park_first"
    s2 = _Sched(running=[], waiting=[_req("n", 4)])
    assert rt.admission(s2, s2.running_batch) is None  # nothing parked: the stock loop


def test_an_abort_reaches_the_parked_list():
    a, b = _req("weg2-1-1", 1), _req("weg2-1-2", 2)
    s = _Sched(running=[a, b])
    _park(s)
    n = rt.park_abort(s, types.SimpleNamespace(rid="weg2-1-2", abort_all=False))
    assert n == 1 and [r.rid for r in s.weg2_d_parked] == ["weg2-1-1"]
    assert len(s.sent) == 1


def test_the_decode_retraction_order_parks_the_youngest_on_group_d():
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    sa = types.SimpleNamespace(retraction_policy="fcfs", schedule_low_priority_values_first=False)
    # the stock key retracts the fewest-outputs request ('old' here); D retracts 'young'
    old = types.SimpleNamespace(rid="old", kv_arrival_seq=1, output_ids=[0], origin_input_ids=[0] * 5,
                                is_fast_lane=False, spill_class=None, priority=None)
    young = types.SimpleNamespace(rid="young", kv_arrival_seq=2, output_ids=[0] * 50,
                                  origin_input_ids=[0] * 5, is_fast_lane=False, spill_class=None,
                                  priority=None)
    order = ScheduleBatch._get_decode_retraction_order([young, old], sa, allow_policy_sort=True)
    assert [young, old][order[-1]] is young


def test_the_batch_retract_all_can_retain():
    import inspect

    from sglang.srt.managers.schedule_batch import ScheduleBatch

    assert "retain" in inspect.signature(ScheduleBatch.retract_all).parameters


# ---- wiring ratchets (the sites a desk test cannot drive) -------------------

def _read(*parts):
    return open(os.path.join(_SRT, *parts)).read()


def test_wiring_park_running_route_and_rpc():
    assert '"/weg2/park_running"' in _read("entrypoints", "http_server.py")
    assert '("weg2_park_running", Weg2ParkRunningReqOutput)' in _read(
        "managers", "tokenizer_control_mixin.py")
    sch = _read("managers", "scheduler.py")
    assert "(Weg2ParkRunningReqInput, self.handle_weg2_park_running)" in sch
    # H91c3-2: the handler also passes #1443's dormant admit (the late hold)
    assert "return d_park_runtime.park_running(\n" in sch
    assert "self, recv_req, late_hold_armed=_weg2_dormant_admit_armed()" in sch


def test_wiring_sleep_retract_admission_abort_tick():
    wu = _read("managers", "scheduler_components", "weight_updater.py")
    i = wu.index("scheduler.weg2_dormant = True")
    assert "weg2_d_hold_parked" in wu[i:i + 1500]
    sch = _read("managers", "scheduler.py")
    assert ("self._add_request_to_queue(req, is_retracted=True)\n"
            "        self._weg2_d_park_note_retracted(retracted_reqs)") in sch
    assert "_d_park_gate = self._weg2_d_park_admission(running_batch)" in sch
    assert "_d_skip = _d_park_gate.skip(req)" in sch
    assert "self._weg2_d_park_abort(recv_req)" in sch
    assert "self._weg2_d_park_tick()" in sch
    assert "hold_armed=_weg2_dormant_admit_armed()" in sch
