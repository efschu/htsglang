"""ARRIVAL-SEAT MIN-DWELL + non-stream progress (NF-Operator 30.09., y5c).

y5c front.log (boot ...dauer09301914_9a1b0a2c91_0930_191424): weg2-0-2
(113k, NON-stream) was parked 6x by ``ARRIVAL-SEAT verdict=flip_now`` ->
``PARK-RUNNING reason=immediate-over-x``, each time for a younger arrival;
park 4 (19:19:52.199, for weg2-8-14) came 1.1 s after its PARK-RESUME
(19:19:50.430). 91 s decoded, 83 s parked, WEG2-CLIENT-GONE at 235 s, 1650
tokens lost. Operator's rule (ski rental): a flip_now to P waits until the
decodes resumed in this D phase have decoded, since their resume, one
measured flip round trip -- X-COST-LINE's price, never a constant.

Second: the front stamps progress per streamed chunk only; for a non-stream
rid ``outstanding_stalest.no_token_s`` and ``d_park_stuck`` read a
blindness as a stall. No per-rid IPC source exists -> the rows say so.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import arrival_seat_rule as asr  # noqa: E402
from sglang.srt.weg2.front_state_ipc import OutstandingBook  # noqa: E402
from sglang.srt.weg2.park_stuck import ParkStuck  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pbound_flip_now_off(monkeypatch):
    """These tests pin the seat/KV verdict and the dwell holds for requests that
    need P -- the rule PBOUND-FLIP-NOW (default on, 02.10.) replaces; they keep
    covering it as the switch-off path."""
    monkeypatch.setenv("SGLANG_WEG2_PBOUND_FLIP_NOW", "0")


# y5c, the six flip_now verdicts that parked weg2-0-2:
# (flip_now rid, epoch, PARK-RESUME of the phase or None, flip_now time, price_s
#  = the PARK-ROUND-TRIP / X COST-LINE price in force at the verdict)
def _t(h, m, s):  # seconds of the day (UTC)
    return h * 3600.0 + m * 60.0 + s


PARKS = [
    ("weg2-2-9", 2, None, _t(19, 18, 46.822), 3.97),                 # first D phase: nothing resumed
    ("weg2-4-10", 4, _t(19, 19, 1.351), _t(19, 19, 6.773), 4.50),   # dwell 5.42
    ("weg2-6-12", 6, _t(19, 19, 18.059), _t(19, 19, 30.625), 4.89),  # dwell 12.57
    ("weg2-8-14", 8, _t(19, 19, 50.430), _t(19, 19, 51.527), 4.99),  # dwell 1.10 -> HOLD
    ("weg2-10-15", 10, _t(19, 19, 58.652), _t(19, 20, 15.188), 4.93),  # dwell 16.54
    ("weg2-12-19", 12, _t(19, 20, 30.535), _t(19, 20, 53.805), 4.69),  # dwell 23.27
]


def test_replay_the_six_parks_only_park_4_is_held():
    held = []
    for rid, _ep, t_res, t_flip, price in PARKS:
        resumed = {} if t_res is None else {"weg2-0-2": t_res}
        h = asr.min_dwell_hold(resumed, ["weg2-0-2"], t_flip, price)
        held.append(None if h is None else (rid, round(h[1], 2)))
    assert held == [None, None, None, ("weg2-8-14", 1.1), None, None]
    # held until 19:19:55.42 (resume + 4.99), then the flip may come
    assert asr.min_dwell_hold({"weg2-0-2": _t(19, 19, 50.430)}, ["weg2-0-2"],
                              _t(19, 19, 55.43), 4.99) is None


def test_unmeasured_price_never_holds():
    assert asr.min_dwell_hold({"a": 0.0}, ["a"], 0.5, None) is None
    assert asr.min_dwell_hold({"a": 0.0}, ["a"], 0.5, 0.0) is None


def test_the_shortest_resumed_dwell_decides():
    assert asr.min_dwell_hold({"a": 0.0, "b": 3.0}, ["a", "b", "c"], 4.0, 2.0) == ("b", 1.0)
    # a resumed rid that no longer runs does not hold
    assert asr.min_dwell_hold({"b": 3.0}, ["a"], 4.0, 2.0) is None


def _park4_front(monkeypatch, dwell_s, age_plan=True, switch=None):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "1" if age_plan else "0")
    if switch is None:
        # ARRIVAL-MIN-DWELL-OFF (02.10.): the hold is opt-in now -- these tests pin it on
        monkeypatch.setenv("SGLANG_WEG2_ARRIVAL_MIN_DWELL", "1")
    elif switch == "default":
        monkeypatch.delenv("SGLANG_WEG2_ARRIVAL_MIN_DWELL", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_ARRIVAL_MIN_DWELL", switch)
    running = ["weg2-0-2", "weg2-6-12", "weg2-6-13"]
    # y5c 19:19:51.527: taken=3 n=6, kv need 112885, free 236016
    f = _front(running=running, n=6, kv={"available": 236016, "evictable": 0, "decode_clip": 4096},
               admit_t={r: 1.0 + i for i, r in enumerate(running)})
    now = time.time()
    f.epoch = 8
    f.t_awake = now - dwell_s                       # PARK-RESUME epoch=8 = the wake's end
    f._seat_resumed_epoch = {r: 8 for r in running}
    f._flip_round_trip_price = lambda: (4.99, "ski-live:y5c-epoch8")
    f._park_dwell_held_t = None
    p = _pending("weg2-8-14", 108789, now - 0.05, payload={"max_tokens": 4096})
    f.queue = [p]
    return f, p, now


def test_park_4_is_held_then_flips_age_plan(monkeypatch):
    f, p, now = _park4_front(monkeypatch, 1.097)
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert res == (False, False, None), res
    assert f.counters["arrival_seat_min_dwell_hold"] == 1
    assert f.counters["arrival_seat_flip_now"] == 0
    assert not f.rpc_calls, "nobody parked while the resumed decodes dwell"
    assert f._park_dwell_held_t is not None, "DP-WAIT names min-dwell"
    # asked again within the hold: one line/counter per rid and phase
    asyncio.run(f._arrival_seat_step(f.groups["D"], time.time()))
    assert f.counters["arrival_seat_min_dwell_hold"] == 1
    # one round trip decoded: the flip comes
    f.t_awake = time.time() - 5.0
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], time.time()))
    assert res == (True, True, p)


def test_park_4_is_held_without_the_age_plan(monkeypatch):
    f, p, now = _park4_front(monkeypatch, 1.097, age_plan=False)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["arrival_seat_min_dwell_hold"] == 1


def test_parks_past_one_round_trip_flip_as_before(monkeypatch):
    for dwell in (5.422, 12.566, 16.536, 23.270):
        f, p, now = _park4_front(monkeypatch, dwell)
        assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, p), dwell
        assert f.counters["arrival_seat_min_dwell_hold"] == 0


def test_red_the_default_is_off_the_arrival_flips_at_once(monkeypatch):
    """User rule 02.10.: an arriving request is prefilled AT ONCE -- NF found this
    hold (DP-WAIT hold_s=1.6-1.9, hold_by=d-work+min-dwell); default off now."""
    f, p, now = _park4_front(monkeypatch, 1.097, switch="default")
    assert asr.min_dwell_enabled() is False
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, p)
    assert f.counters["arrival_seat_min_dwell_hold"] == 0


def test_switch_off_is_todays_flip(monkeypatch):
    f, p, now = _park4_front(monkeypatch, 1.097, switch="0")
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, p)


def test_a_phase_that_resumed_nothing_is_not_held(monkeypatch):
    f, p, now = _park4_front(monkeypatch, 0.2)
    f._seat_resumed_epoch = {r: 6 for r in f._seat_resumed_epoch}   # resumed in an older phase
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, p)


def test_no_measured_round_trip_no_hold(monkeypatch):
    f, p, now = _park4_front(monkeypatch, 0.2)
    f._flip_round_trip_price = lambda: (None, "none:no-warm-legs")
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, p)


# ------------------------------------------------ non-stream progress (IPC)

def test_nonstream_row_is_marked_not_stalled():
    b = OutstandingBook()
    b.arrive("weg2-0-2", 0.0)
    b.stream("weg2-0-2", False)
    b.arrive("s", 0.0)
    b.stream("s", True)
    blk = b.block(200.0, [], {}, {"weg2-0-2": 0.0, "s": 0.0}, ["weg2-0-2"], False)
    rows = {r["rid"]: r for r in blk["outstanding_stalest"]}
    assert rows["weg2-0-2"]["stream"] == 0 and rows["weg2-0-2"]["no_token_s"] is None
    assert rows["weg2-0-2"]["where"] == "parked"
    assert rows["s"]["stream"] == 1 and rows["s"]["no_token_s"] == 200.0
    assert blk["outstanding_nonstream_n"] == 1
    assert blk["outstanding_stalest"][0]["rid"] == "s", "a visible stall sorts first"
    b.end("weg2-0-2")
    assert "weg2-0-2" not in b.nonstream


def test_nonstream_parks_are_blind_not_stuck():
    ps = ParkStuck()
    ps.note_stream("weg2-0-2", False)
    for _ in range(6):                       # y5c: six parks, no chunk the front could see
        ps.note_park(["weg2-0-2"])
    ps.note_stream("s", True)
    for _ in range(3):
        ps.note_park(["s"])
    blk = ps.block(3)
    assert blk["stuck"] == 1 and blk["rids"] == {"s": 3}
    assert blk["blind_nonstream"] == {"weg2-0-2": 6}
    assert blk["max_streak"] == 3 and blk["max_streak_boot"] == 3
    ps.done("weg2-0-2")
    assert ps.block(3)["blind_nonstream"] == {}
