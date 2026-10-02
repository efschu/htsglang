"""NF-STAU (y3m boot ...dauer09292136, 29.09. 21:57-22:05): D decoded weg2-6-19
alone (bs1) for minutes while requests waited.

Metal, front log:
* 21:56:56 weg2-50-66 (66563 tokens, uncached 3267 > X) ``ARRIVAL-SEAT
  verdict=wait_seat ... no seat`` -- then seats freed (D-REFILL seats_free
  1..6 until 21:57:53) and nothing moved for 5 min: the head waited on the KV
  test (need 130563 = prompt + max_tokens against the MAPPED stage), the one
  log line still said "no seat". weg2-50-67..73 (2911 tokens, need 3167) sat
  behind it (no KV backfill).
* the rule's bound (c) never parked: weg2-6-19 was youngest-parked at
  21:48:39, D ran it again (PARK-RESUME 21:56:50), but it stayed in the rule's
  parked set -- counted as no seat (22:02:16 ``taken=0``) and never a victim.
* 22:03:44 weg2-50-74 ``kv need=130563 > free=54016`` while D's KV ladder
  reached 524288 with ~77k used.

Fixes pinned here: the KV fit counts against D's ladder ceiling (IPC,
``weg2_kv.ladder_*``), KV backfill past a head that does not fit (arrival
order among the fitting ones, none past the bound), the parked set follows
D (lapse, flip back), the wait line names a changed reason, and the TTFT
clocks (arrival -> verdict, arrival -> first token) are counted for IPC."""
from __future__ import annotations

import asyncio
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import X, _front, _on, _pending  # noqa: E402

from sglang.srt.weg2 import arrival_seat_rule as asr  # noqa: E402
from sglang.srt.weg2 import phase_policy  # noqa: E402


@pytest.fixture(autouse=True)
def _pbound_flip_now_off(monkeypatch):
    """These tests pin the seat/KV verdict and the dwell holds for requests that
    need P -- the rule PBOUND-FLIP-NOW (default on, 02.10.) replaces; they keep
    covering it as the switch-off path."""
    monkeypatch.setenv("SGLANG_WEG2_PBOUND_FLIP_NOW", "0")


# y3m 22:03:44: stage free 54016, ladder 524288 with ~77k used
LADDER_READING = {"available": 30000, "evictable": 24016, "capacity": 524288,
                  "ladder_ceiling": 524288, "ladder_stage": 131072, "ladder_used": 77000}


# ---------------------------------------------------------------- pure

def test_the_fit_counts_against_the_ladder_not_the_mapped_stage():
    stage_only = {"available": 30000, "evictable": 24016}
    assert asr.kv_fits(130563, stage_only)[0] is False            # the metal verdict
    fits, why = asr.kv_fits(130563, LADDER_READING)
    assert fits and "ladder 524288-77000=447288" in why and "stage free 54016" in why
    # the ladder really full: the stage reading still decides the larger free
    full = dict(LADDER_READING, ladder_used=520000)
    assert asr.kv_fits(130563, full)[0] is False
    assert asr.ladder_free(None) is None and asr.ladder_free(stage_only) is None
    assert asr.ladder_free({"ladder_ceiling": 0, "ladder_used": 1}) is None


def test_backfill_stops_at_the_bound_and_clocks_count():
    assert asr.backfill_allowed(None, 60.0) and asr.backfill_allowed(59.0, 60.0)
    assert not asr.backfill_allowed(60.0, 60.0)
    c = __import__("collections").Counter()
    asr.note_ms(c, "ttft", 1200.4)
    asr.note_ms(c, "ttft", 800.0)
    blk = asr.state_block(c, {}, True)
    assert (blk["ttft_n"], blk["ttft_ms_sum"], blk["ttft_ms_max"]) == (2, 2000, 1200)
    assert blk["verdict_n"] == 0 and blk["backfill"] == 0


# ---------------------------------------------------------------- front: over-X queue

def _big_head_small_tail(now):
    head = _pending("weg2-50-66", 3267, now - 5.0, est_prompt=66563, payload={"max_tokens": 64000})
    tail = [_pending("weg2-50-%d" % i, 863, now - 4.0 + 0.01 * i, est_prompt=2911,
                     payload={"max_tokens": 256}) for i in (67, 68)]
    return head, tail


def test_a_head_whose_kv_does_not_fit_lets_a_fitting_later_one_flip(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["weg2-6-19"], n=6, kv={"available": 30000, "evictable": 24016})
    now = time.time()
    f.t_awake = now - 30.0
    head, tail = _big_head_small_tail(now)
    for p in [head] + tail:
        p.est_uncached = max(p.est_uncached, X + 1)
    f.queue = [head] + tail
    wait_fired, fairness_fired, immediate = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert (wait_fired, fairness_fired) == (True, True)
    assert immediate is tail[0]                               # arrival order among the fitting
    assert f.counters["arrival_seat_backfill"] == 1
    assert f.counters["arrival_seat_verdict_n"] == 1


def test_with_the_ladder_the_head_itself_flips(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["weg2-6-19"], n=6, kv=dict(LADDER_READING))
    now = time.time()
    head, tail = _big_head_small_tail(now)
    head.est_uncached = X + 1
    f.queue = [head] + tail
    _, _, immediate = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert immediate is head and f.counters["arrival_seat_backfill"] == 0


def test_no_backfill_once_the_head_waited_the_bound(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 30000, "evictable": 24016},
               admit_t={"a": 1.0})
    now = time.time()
    f.t_awake = now - 300.0
    head, tail = _big_head_small_tail(now)
    head.t_arrive = now - 120.0
    for p in [head] + tail:
        p.est_uncached = X + 1
    f.queue = [head] + tail
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert f.counters["arrival_seat_backfill"] == 0
    # (c) displaced for the head instead: the youngest decode parked
    assert f.rpc_calls and f.rpc_calls[0][1]["youngest"] == "a"
    assert res[2] is not tail[0]


def test_the_wait_line_names_a_changed_reason(monkeypatch, caplog):
    _on(monkeypatch)
    f = _front(running=["a"], n=1, kv={"available": 1000, "evictable": 0})
    now = time.time()
    p = _pending("big", X + 10, now, est_prompt=66563)
    f.queue = [p]
    with caplog.at_level("INFO", logger="weg2.front"):
        asyncio.run(f._arrival_seat_step(f.groups["D"], now))          # no seat
        asyncio.run(f._arrival_seat_step(f.groups["D"], now + 0.2))    # same: no second line
        f.groups["D"].outstanding.pop("a")
        asyncio.run(f._arrival_seat_step(f.groups["D"], now + 0.4))    # seat free, KV short
    lines = [r.getMessage() for r in caplog.records if "verdict=wait_seat" in r.getMessage()]
    assert len(lines) == 2 and "why=seat" in lines[0] and "why=kv" in lines[1]
    assert f.counters["arrival_seat_wait_seat"] == 1 and f.counters["arrival_seat_wait_kv"] == 1


# ---------------------------------------------------------------- front: the parked set follows D

def test_a_lapsed_youngest_park_counts_as_running_again(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["weg2-6-19"], n=1, kv=dict(LADDER_READING), admit_t={"weg2-6-19": 1.0})
    now = time.time()
    st = f._asr_st()
    st["parked"]["weg2-6-19"] = now - phase_policy.PARK_REQUEUE_S - 1.0   # parked 21:48:39
    f.queue = [_pending("big", X + 10, now, est_prompt=2911)]
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert "weg2-6-19" not in st["parked"]
    assert f._arrival_seat_taken() == (1, 1)                  # it holds its seat again
    assert res == (False, False, None)                        # n=1: no seat, no flip


def test_a_flip_back_to_d_clears_the_rule_parks(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6)
    f._asr_st()["parked"]["a"] = time.time()
    f._asr_clear_parked("flip-to-D")
    assert f._asr_st()["parked"] == {}
    g = _front()
    g._asr_clear_parked("flip-to-D")                           # no rule state: a no-op
    assert "_asr_state" not in g.__dict__


# ---------------------------------------------------------------- front: D-prefill waiters

def test_a_short_waiter_backfills_past_an_older_one_whose_kv_does_not_fit(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 5000, "evictable": 0})

    async def body():
        t_big = asyncio.ensure_future(f._acquire_short_seat("big", 60000, [], max_tokens=64000))
        await asyncio.sleep(0.1)
        t_small = asyncio.ensure_future(f._acquire_short_seat("small", 385, [], max_tokens=256))
        seat = await asyncio.wait_for(t_small, 2.0)
        assert seat is not None and seat.rid == "small"
        assert not t_big.done()                               # the head still waits for its KV
        t_big.cancel()
        return f.counters["arrival_seat_backfill"]

    assert asyncio.run(body()) == 1


def test_arrival_stamps_are_bounded_and_popped(monkeypatch):
    _on(monkeypatch)
    f = _front()
    monkeypatch.setattr(type(f), "ASR_ARRIVALS_KEPT", 3)
    for i in range(5):
        f._asr_arrival_note("r%d" % i)
    assert list(f.__dict__["_asr_arrive"]) == ["r2", "r3", "r4"]
    assert f._asr_arrival_of("r3", pop=True) is not None and f._asr_arrival_of("r3") is None
    assert f._asr_verdict_ms("r4", None) >= 0 and f.counters["arrival_seat_verdict_n"] == 1


# ---------------------------------------------------------------- D side: the ladder on /server_info

def test_d_publishes_its_ladder_ceiling_and_used(monkeypatch):
    from sglang.srt.weg2 import d_seat_vram as dsv
    from sglang.srt.weg2.d_mem_sched import MemSched

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    ladder = (32768, 65536, 131072, 262144, 393216, 524288)
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", ",".join(str(t) for t in ladder))
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_BY_DEMAND", "1")
    req = types.SimpleNamespace(rid="weg2-6-19", origin_input_ids=[0] * 76000, output_ids=[0] * 1000)
    sched = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096),
        running_batch=types.SimpleNamespace(reqs=[req]), waiting_queue=[], chunked_req=None,
        last_batch=None)
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e", n=1, cap=6, done=True, stage=2,
                                                  stage_tokens=131072))
    # before the machine's first tick: the form's top usable stage
    assert dsv.kv_ladder_reading(sched) == {"ladder_ceiling": 524288, "ladder_stage": 131072,
                                            "ladder_used": 77000}
    setattr(sched, dsv.MEM_SCHED_ATTR, MemSched(stage_tokens=ladder[:5], air_tokens=0, stage=3))
    assert dsv.kv_ladder_reading(sched) == {"ladder_ceiling": 393216, "ladder_stage": 262144,
                                            "ladder_used": 77000}
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert dsv.kv_ladder_reading(sched) is None


@pytest.mark.parametrize("path", ["python/sglang/srt/managers/scheduler.py"])
def test_the_scheduler_route_merges_the_ladder(path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(asr.__file__))))))
    src = open(os.path.join(root, path)).read()
    i = src.index('ret["weg2_kv"] = {')
    j = src.index("except Exception:", i)
    assert "kv_ladder_reading(self)" in src[i:j] and 'ret["weg2_kv"].update(_lad)' in src[i:j]
