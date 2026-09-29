"""ARRIVAL-SEAT rule (user 29.09. ~19:40Z, weg2/arrival_seat_rule.py).

"wenn im aktuellen decode noch platz frei wäre für einen weiteren parallelen
decode mit ebendiesem kv bedarf des ankommenden requests, dann soll das decode
direkt pausiert werden ... und prefillt werden (entweder in D größe oder mit
flip in P)". Finding behind it (NF1e): weg2-40-121, 385 tokens, went to P over
H91c3 SEATS-FULL and paid two flips, 40 s.

The 27B-mandated cases: switch off = today's path; on + free seat + uncached
<= X = D prefill; on + free seat + uncached > X = the immediate flip; no seat =
no flip request; the wait bound = the youngest decode parks. Both geometries
(NF: d_wait_bound 60 s; 27B: bound 0 -> the fairness 45 s)."""
from __future__ import annotations

import asyncio
import collections
import json
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import arrival_seat_rule as asr  # noqa: E402
from sglang.srt.weg2 import front as front_mod  # noqa: E402

X = 4096


# ---------------------------------------------------------------- pure

def test_verdict_is_the_rule():
    assert asr.verdict(True, True, 385, X) == asr.D_PREFILL
    assert asr.verdict(True, True, 16448, X) == asr.FLIP_NOW
    assert asr.verdict(False, True, 385, X) == asr.WAIT_SEAT
    assert asr.verdict(False, True, 16448, X) == asr.WAIT_SEAT
    assert asr.verdict(True, False, 16448, X) == asr.WAIT_SEAT


def test_kv_need_is_prompt_plus_reserve_and_reads_d():
    assert asr.decode_reserve(None, 2048) == 2048
    assert asr.decode_reserve(256, 2048) == 256
    assert asr.max_tokens_of({"max_completion_tokens": 700}) == 700
    assert asr.max_tokens_of({"sampling_params": {"max_new_tokens": 9}}) == 9
    assert asr.kv_need(6511, 2048) == 8559
    assert asr.kv_fits(8559, {"available": 5000, "evictable": 4000}) == (True, "kv need=8559 free=9000")
    assert asr.kv_fits(9001, {"available": 5000, "evictable": 4000})[0] is False
    assert asr.kv_fits(10 ** 9, None) == (True, "kv=unread")


def test_youngest_is_the_last_admitted_and_unstamped_is_never_chosen():
    admit = {"a": 1.0, "b": 3.0, "c": 2.0}
    assert asr.youngest_running(admit, ["a", "b", "c", "old"]) == "b"
    assert asr.youngest_running(admit, ["old"]) is None
    assert asr.oldest_wait_s([100.0, 50.0], 80.0, 130.0) == 50.0   # D phase start, not arrival
    assert asr.bound_fired(60.0, 60.0) and not asr.bound_fired(59.9, 60.0) and not asr.bound_fired(99, 0)


# ---------------------------------------------------------------- front harness

class _Fut:
    def __init__(self, done=False):
        self._d = done

    def done(self):
        return self._d


def _pending(rid, uncached, t_arrive, est_prompt=None, payload=None):
    return types.SimpleNamespace(rid=rid, est_uncached=uncached, est_prompt=est_prompt or uncached,
                                 t_arrive=t_arrive, fut=_Fut(), payload=payload or {},
                                 skip_leg1=False, leg1_done=False, p_only=False, x_requeues=0,
                                 x_deferred=False)


def _front(running=(), n=6, bound=60.0, w_s=45.0, kv=None, admit_t=None):
    f = front_mod.Front.__new__(front_mod.Front)
    f.counters = collections.Counter()
    f.awake, f.admit_d, f.state, f.epoch = "D", True, "serving", 7
    f.d_bs, f._d_phase_n, f.d_wait_bound_s, f.w_s = 6, n, bound, w_s
    f.t_awake = time.time() - 5.0
    f.tp_prefill_max_tokens = X
    f.queue, f._ready_for_d, f._d_parked = [], [], {}
    D = types.SimpleNamespace(name="D", outstanding={r: 1.0 for r in running}, url="http://d")
    f.groups = {"D": D}
    f._flip_ledger = lambda g: [r for r in g.outstanding if r not in f._d_parked]
    # the front's hand-off term: seats held for rids D does not run yet
    f._handoff_in_flight = lambda: len([s for s in f._d_seats_live if s.rid not in D.outstanding])
    f._d_seat = asyncio.Semaphore(6)
    f._d_seats_live = set()
    f._batch_gate = asyncio.Event()
    f._batch_gate.set()
    f.drain_deadline_s = 5.0
    f._d_token_budget_blocks = lambda *a, **k: False

    async def _no_reading():
        return None

    async def _kv():
        return kv

    f._d_reading_if_armed = _no_reading
    f._arrival_seat_kv_reading = _kv
    f._d_admit_t = dict(admit_t or {})
    f.rpc_calls = []

    async def rpc(g, path, body, timeout):
        f.rpc_calls.append((path, dict(body)))
        return 200, json.dumps({"parked": [body["youngest"]] if body.get("youngest") else [], "held": []})

    f.rpc = rpc
    return f


def _on(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")


def test_off_is_todays_path_seats_full_goes_to_p(monkeypatch):
    """Switch off: H91c3-3 exactly as before (SEATS-FULL -> BATCH), the rule's
    state is never created."""
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", raising=False)
    f = _front(running=["a"], n=1)
    assert asyncio.run(f._acquire_short_seat("new", 385)) is None
    assert f.counters["short_phase_seats_full"] == 1
    assert "_asr_state" not in f.__dict__
    assert not any(k.startswith("arrival_seat") for k in f.counters)


def test_on_free_seat_short_is_prefilled_on_d_at_once(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a", "b"], n=6, kv={"available": 50000, "evictable": 0})
    seat = asyncio.run(f._acquire_short_seat("new", 385, [], max_tokens=512))
    assert seat is not None and seat.rid == "new"
    assert f.counters["arrival_seat_d_prefill"] == 1
    assert f.counters["short_phase_seats_full"] == 0


def test_on_free_seat_over_x_flips_now(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 16448, time.time())
    f.queue = [p]
    wait_fired, fairness_fired, immediate = asyncio.run(f._arrival_seat_step(f.groups["D"], time.time()))
    assert (wait_fired, fairness_fired, immediate) == (True, True, p)
    assert f.counters["arrival_seat_flip_now"] == 1
    # the collect window / window gate were never consulted: nothing of theirs moved
    assert not any(k.startswith("park_collect") or k.startswith("park_window") for k in f.counters)


def test_on_no_seat_requests_no_flip_and_the_short_waits_on_d(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=1, kv={"available": 400000, "evictable": 0})
    f.queue = [_pending("big", 16448, time.time())]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (False, False, None)
    assert f.counters["arrival_seat_wait_seat"] == 1

    async def body():
        t = asyncio.ensure_future(f._acquire_short_seat("short", 385, []))
        await asyncio.sleep(0.2)
        assert not t.done()                       # waits -- no BATCH fall-through
        assert f.counters["short_phase_seats_full"] == 0
        f.groups["D"].outstanding.pop("a")        # the running decode ends
        seat = await asyncio.wait_for(t, 2.0)
        assert seat is not None and seat.rid == "short"

    asyncio.run(body())


def test_on_kv_short_waits_too(monkeypatch):
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 1000, "evictable": 0})
    f.queue = [_pending("big", 16448, time.time())]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (False, False, None)
    assert f.counters["arrival_seat_wait_kv"] == 1


def test_on_arrival_order_the_older_waiter_takes_the_freed_seat(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=1)

    async def body():
        t1 = asyncio.ensure_future(f._acquire_short_seat("first", 100, []))
        await asyncio.sleep(0.05)
        t2 = asyncio.ensure_future(f._acquire_short_seat("second", 100, []))
        await asyncio.sleep(0.1)
        f.groups["D"].outstanding.pop("a")
        s1 = await asyncio.wait_for(t1, 2.0)
        await asyncio.sleep(0.2)                      # its seat is held (hand-off)
        assert not t2.done()
        f.groups["D"].outstanding["first"] = 1.0      # it now runs on D
        await asyncio.sleep(0.2)
        assert not t2.done()                          # n=1: the second waits for the next seat
        f.groups["D"].outstanding.pop("first")
        s1.release("test")
        s2 = await asyncio.wait_for(t2, 2.0)
        return s1.rid, s2.rid

    assert asyncio.run(body()) == ("first", "second")


@pytest.mark.parametrize("bound,w_s", [(60.0, 45.0), (0.0, 45.0)], ids=["nf-bound60", "27b-fairness45"])
def test_on_wait_bound_parks_the_youngest_and_never_flips(monkeypatch, bound, w_s):
    _on(monkeypatch)
    f = _front(running=["a", "b", "c"], n=3, bound=bound, w_s=w_s,
               admit_t={"a": 1.0, "b": 3.0, "c": 2.0})
    victim_seat = front_mod.Seat(f, "b", "short")
    now = time.time()
    f.t_awake = now - 200.0
    f.queue = [_pending("big", 16448, now - 120.0)]           # waited past the bound
    wait_fired, fairness_fired, immediate = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert f.rpc_calls and f.rpc_calls[0][1]["youngest"] == "b"
    assert f.rpc_calls[0][1]["reason"] == asr.REASON_YOUNGEST
    assert f.counters["arrival_seat_youngest_park"] == 1
    assert victim_seat.held is False                           # its front seat went free
    # the freed seat is the arrival rule's: it flips now for the waiting > X request
    assert (wait_fired, fairness_fired, immediate) == (True, True, f.queue[0])
    # once per bound interval: the next tick does not park a second one
    asyncio.run(f._arrival_seat_step(f.groups["D"], now + 1.0))
    assert len(f.rpc_calls) == 1


def test_on_wait_bound_refused_is_asked_again(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a", "b"], n=2, admit_t={"a": 1.0, "b": 2.0})

    async def rpc(g, path, body, timeout):
        f.rpc_calls.append(body)
        return 200, json.dumps({"parked": [], "held": []})

    f.rpc = rpc
    now = time.time()
    f.t_awake = now - 200.0
    f.queue = [_pending("big", 16448, now - 100.0)]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["arrival_seat_youngest_park_refused"] == 1
    asyncio.run(f._arrival_seat_step(f.groups["D"], now + 1.0))   # past the 0.5 s retry
    assert len(f.rpc_calls) == 2


# ---------------------------------------------------------------- D side

class _Req:
    def __init__(self, rid):
        self.rid = rid


class _Batch:
    def __init__(self, rids, spec=False):
        self.reqs = [_Req(r) for r in rids]
        self.spec_algorithm = types.SimpleNamespace(is_none=lambda: not spec)
        self.released = []

    def release_req(self, idx, last, server_args, retain=True):
        self.released.append((idx, retain))

    def filter_batch(self, keep_indices=None, **_):
        if keep_indices is not None:
            self.reqs = [self.reqs[i] for i in keep_indices]

    def is_empty(self):
        return not self.reqs


def _sched(batch):
    s = types.SimpleNamespace(enable_overlap=False, last_batch=None, result_queue=None,
                              running_batch=batch, waiting_queue=[], server_args=None,
                              weg2_dormant=False)
    s._add_request_to_queue = lambda req, is_retracted=False: s.waiting_queue.append(req)
    return s


def test_d_parks_only_the_named_youngest(monkeypatch):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput
    from sglang.srt.weg2 import d_park_runtime as dpr

    monkeypatch.setattr(dpr.d_seats, "d_flip_park_active", lambda: True)
    monkeypatch.setattr(dpr.d_seats, "mark_parked", lambda req, site, now=None: setattr(req, "_site", site))
    b = _Batch(["weg2-1-1", "weg2-1-2", "weg2-1-3"])
    s = _sched(b)
    out = dpr.park_running(s, Weg2ParkRunningReqInput(epoch=3, reason=asr.REASON_YOUNGEST, youngest="weg2-1-3"))
    assert out.success and list(out.parked) == ["weg2-1-3"]
    assert [r.rid for r in b.reqs] == ["weg2-1-1", "weg2-1-2"]
    assert [r.rid for r in s.waiting_queue] == ["weg2-1-3"] and b.released == [(2, True)]
    assert s.waiting_queue[0]._site == dpr.d_seats.SITE_PRESSURE


def test_d_refuses_a_non_back_youngest_under_spec(monkeypatch):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput
    from sglang.srt.weg2 import d_park_runtime as dpr

    monkeypatch.setattr(dpr.d_seats, "d_flip_park_active", lambda: True)
    b = _Batch(["weg2-1-1", "weg2-1-2"], spec=True)
    out = dpr.park_running(_sched(b), Weg2ParkRunningReqInput(epoch=3, reason="x", youngest="weg2-1-1"))
    assert not out.success and list(out.parked) == [] and len(b.reqs) == 2
