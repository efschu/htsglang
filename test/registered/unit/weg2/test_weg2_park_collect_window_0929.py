"""PARK-COLLECT-WINDOW (29.09.): the immediate park's successor, for qwen27b AND
nextflash (user: "dort sind es ja die selben Fehler").

NF z30w-park (boot ...z30wparkbar1dauer09290827, under agent load plus the
user's OpenWebUI stream: "es flattert, meine Anfrage kommt nur sehr abgewürgt
durch"): 46 PARK-IMMEDIATE FIRED in 32 min, park -> resume median 9.5 s, p90
22.3 s. 27B row-authority (boot ...dkr27browauthoritybar1w109290020, 00:21-00:54Z,
agent load): 74 immediate parks, park -> resume median 8.2 s.

User decision 29.09. ~09:15Z ("vorschlag akzeptiert"): D keeps decoding; once
the pending P work passes the bound it collects -- SKI RENTAL -- until waiting
has cost what the flip costs (one measured round trip), D empty flips at once,
hard caps stay, a fixed x is an override only. Pinned here: the policy, the
switch (off = the immediate park byte for byte), the round-trip record per
checkpoint x form (never shared between the models), and a replay of BOTH
boots' measured over-X arrivals and D streams, old against new."""
from __future__ import annotations

import collections
import json
import os
import pathlib
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "park_collect_0929"
BOOTS = ("nextflash_z30w_park.json", "qwen27b_row_authority.json")
WAIT_BOUND_S = 60.0   # H91 standard form: d_wait_bound 60 s
CAP = 6               # p_phase_max_requests


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def _replay(fx, ski: bool, dt: float = 0.1):
    """The measured over-X arrivals and D streams of one boot under today's park
    (PARK-DECODE-DWELL: D decodes one cycle, then any over-X arrival parks) or
    under the ski window (price = the boot's measured round trip per running
    stream). A park stalls every running stream for D->P + P's work + P->D +
    resume. Returns parks, per-stream interruptions and stall, and per-request
    hold (arrival -> park) and TTFT (hold + the cycle that serves it)."""
    dp, pd, res = fx["dp_ms_median"] / 1e3, fx["pd_ms_median"] / 1e3, fx["resume_ms_median"] / 1e3
    arrivals, streams = fx["long_arrivals"], fx["d_streams"]
    dwell = dp + pd + res
    rem = [d for _, d in streams]
    cuts, stall = [0] * len(streams), [0.0] * len(streams)
    holds, ttft = [], []
    t, wake, asleep_until, i, parks, queue = 0.0, 0.0, 0.0, 0, 0, []
    while t < arrivals[-1][0] + 600.0:
        while i < len(arrivals) and arrivals[i][0] <= t:
            queue.append(tuple(arrivals[i]))
            i += 1
        awake = t >= asleep_until
        active = [j for j, (a, _) in enumerate(streams) if a <= t and rem[j] > 0]
        if awake:
            for j in active:
                rem[j] -= dt
        fire = awake and bool(queue) and (not active or t - wake >= dwell)
        if fire and active and ski:
            fire = pp.park_collect_window(queue, t, wake, len(active), fx["round_trip_s"] * len(active),
                                          fx["x_tokens"], max_requests=CAP, wait_bound_s=WAIT_BOUND_S)[0]
        if fire:
            parks += 1
            cycle = dp + sum(u for _, u in queue) / fx["p_tok_s"] + pd + res
            for j in active:
                cuts[j] += 1
                stall[j] += cycle
            holds += [t - a for a, _ in queue]
            ttft += [t - a + cycle for a, _ in queue]
            queue, asleep_until = [], t + cycle
            wake = asleep_until
        if i >= len(arrivals) and not queue:
            break
        t += dt
    return types.SimpleNamespace(parks=parks, cuts=sum(cuts), max_cuts=max(cuts), stall=sum(stall),
                                 holds=holds, ttft=sorted(ttft), dwell=dwell)


def test_policy_timer_override_opens_at_the_crossing_and_closes_after_x():
    # one request barely over X (NF 08:33:14 epoch 6: 6396 uncached, 3 running)
    kw = dict(timer=True)
    assert pp.park_collect_window([(100.0, 6396)], 100.2, 90.0, 3, 20.0, 4096, **kw)[:2] == (False, "collect")
    fire, why, t0, left = pp.park_collect_window([(100.0, 6396)], 120.1, 90.0, 3, 20.0, 4096, **kw)
    assert (fire, why, t0) == (True, "timer", 100.0) and left == 0.0
    # never before this D phase woke: a request queued during the P phase opens it at the wake
    assert pp.park_collect_window([(80.0, 6396)], 95.0, 90.0, 3, 20.0, 4096, **kw)[1:3] == ("collect", 90.0)
    # a threshold above X: the window opens once the SUM passes it
    _, why, t0, _ = pp.park_collect_window([(100.0, 5000), (104.0, 5000)], 105.0, 90.0, 3, 20.0, 8192, **kw)
    assert (why, t0) == ("collect", 104.0)
    assert pp.park_collect_window([(100.0, 5000)], 200.0, 90.0, 3, 20.0, 8192)[1] == "below-threshold"
    assert pp.park_collect_window([], 1.0, 0.0, 3, 20.0, 4096)[1] == "empty"


def test_policy_ski_rent_is_the_summed_wait_against_the_price():
    # one request, price one round trip: fires after exactly that wait (the user's rule)
    assert pp.park_collect_window([(100.0, 6396)], 107.0, 90.0, 1, 7.87, 4096)[1] == "collect"
    assert pp.park_collect_window([(100.0, 6396)], 107.9, 90.0, 1, 7.87, 4096)[1] == "rent"
    # three running streams: the caller's price is three round trips
    fire, why, _, left = pp.park_collect_window([(100.0, 6396)], 108.0, 90.0, 3, 3 * 7.87, 4096)
    assert (fire, why) == (False, "collect") and left == pytest.approx(3 * 7.87 - 8.0)
    # a second waiting request pays rent too: two requests reach the price twice as fast
    two = [(100.0, 6396), (100.0, 5000)]
    assert pp.park_collect_window(two, 111.9, 90.0, 3, 3 * 7.87, 4096)[1] == "rent"
    # the rent never counts time before the window opened (the D wake)
    assert pp.park_collect_window([(50.0, 6396)], 97.0, 90.0, 1, 7.87, 4096)[1] == "collect"


def test_policy_flips_early_on_idle_d_and_hard_caps():
    q = [(100.0, 6396)]
    assert pp.park_collect_window(q, 100.1, 90.0, 0, 20.0, 4096)[1] == "d-idle"
    six = [(100.0 + k, 5000) for k in range(6)]
    assert pp.park_collect_window(six, 106.0, 90.0, 3, 99.0, 4096, max_requests=6)[1] == "cap-requests"
    assert pp.park_collect_window([(100.0, 300000)], 100.1, 90.0, 3, 99.0, 4096,
                                  pool_tokens=262144)[1] == "cap-pool"
    assert pp.park_collect_window(q, 110.0, 90.0, 3, 99.0, 4096, wait_bound_s=10.0)[1] == "wait-bound"


@pytest.mark.parametrize("boot", BOOTS)
def test_replay_measured_arrivals_fewer_flips_and_stream_stalls_bounded_ttft(boot):
    fx = _fixture(boot)
    old, new = _replay(fx, ski=False), _replay(fx, ski=True)
    n = len(fx["long_arrivals"])
    assert len(old.holds) == len(new.holds) == n                 # every arrival is served
    assert new.parks < old.parks
    assert new.cuts <= 0.6 * old.cuts                            # stream interruptions
    assert new.stall <= 0.7 * old.stall                          # stalled stream-seconds
    assert new.max_cuts < old.max_cuts                           # the worst-hit stream
    # never an unbounded wait (27B: the ski bound holds, not a drain): an arrival
    # waits at most the wait bound past the decode dwell of the phase it met
    assert max(new.holds) <= WAIT_BOUND_S + new.dwell + 1.0
    # the TTFT price of collecting, bounded by a third of the wait bound at the
    # median and the p90 (measured: NF +12.0 / +19.0 s, 27B +9.8 / +4.4 s)
    med = lambda v: v[len(v) // 2]  # noqa: E731
    p90 = lambda v: v[int(0.9 * len(v))]  # noqa: E731
    assert med(new.ttft) - med(old.ttft) <= WAIT_BOUND_S / 3
    assert p90(new.ttft) - p90(old.ttft) <= WAIT_BOUND_S / 3


def test_round_trip_record_is_per_checkpoint_and_form(tmp_path):
    nf = "Qwen3.8-Flash-Next|arch=moe,experts=offload,draft=mtp,kv=full,flip=on"
    q27 = "Qwen3.8-27B|arch=dense,experts=none,draft=dflash,kv=full,flip=on"
    rec = lambda key, rt_ms, at: pp.park_round_trip_record(  # noqa: E731
        form_key=key, dp_ms=rt_ms, pd_ms=2000.0, resume_ms=1000.0, boot_tag="b", commit=None, at=at)
    path = tmp_path / "weg2_measured_record.json"
    path.write_text(json.dumps({"samples": [
        {"group": "P", "rss_shmem_gib": 1.0, "at": "2026-09-29 09:00:00,000"},
        rec(nf, 2525.0, "2026-09-29 08:40:00,000"), rec(nf, 2600.0, "2026-09-29 08:50:00,000"),
        rec(q27, 2167.0, "2026-09-29 09:10:00,000")]}))
    got = pp.read_park_round_trip(str(path), nf)
    assert got["round_trip_s"] == pytest.approx(5.6) and got["form_key"] == nf   # newest of NF only
    assert pp.read_park_round_trip(str(path), q27)["round_trip_s"] == pytest.approx(5.167)
    assert pp.read_park_round_trip(str(path), "other|x") is None
    assert pp.read_park_round_trip(str(tmp_path / "missing.json"), nf) is None
    assert pp.read_park_round_trip(str(path), "") is None                        # no form, no seed
    assert pp.park_round_trip_s(2525.0, 0.0, 1000.0) is None                     # K7 before its flip


def _flip_log(dp_ms=2525, pd_ms=2218, warm_pairs=3, cold_ms=24600):
    """The boot's cold first flip (H34b: lane registration) and warm pairs."""
    log = [{"sleep": "D", "wake": "P", "flip_ms": cold_ms}, {"sleep": "P", "wake": "D", "flip_ms": pd_ms}]
    for _ in range(warm_pairs):
        log += [{"sleep": "D", "wake": "P", "flip_ms": dp_ms}, {"sleep": "P", "wake": "D", "flip_ms": pd_ms}]
    return log


# NF z30w-park medians: first resume cold, then warm ones around 3123 ms
WARM_RESUMES = [11800.0, 3100.0, 3123.0, 3150.0]


def _front(running, waits_s, awake_s=30.0, uncached=6396, **attrs):
    from sglang.srt.weg2 import front as F

    now = time.time()
    t_awake = now - awake_s
    ns = types.SimpleNamespace(
        epoch=6, _park_attempt_epoch=-1, _park_resume_epoch=-1, _park_immediate_dwell_epoch=-1,
        _park_collect_epoch=-1, _resume_ms_log=[], flip_log=[], _park_rt_seed=None,
        _park_form_key="Qwen3.8-Flash-Next|arch=moe", t_awake=t_awake, _d_decode_epoch=6,
        _d_decode_t0=t_awake + 1.0,
        queue=[types.SimpleNamespace(rid=f"weg2-6-{18 + j}", est_uncached=uncached, t_arrive=now - w,
                                     p_only=False, x_requeues=0, leg1_done=False, skip_leg1=False)
               for j, w in enumerate(waits_s)],
        tp_prefill_max_tokens=4096, counters=collections.Counter(),
        p_phase_max_requests=6, p_pool_tokens=262144, d_wait_bound_s=0.0,
        _flip_ledger=lambda g: [f"weg2-5-{j}" for j in range(running)],
        _derived_min_dwell_ms=lambda s, d: ((2525, "warm-D->P") if (s, d) == ("D", "P")
                                            else (2218, "warm-P->D")),
    )
    ns._park_collect_window_s = lambda: F.Front._park_collect_window_s(ns)
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    for k, v in attrs.items():
        setattr(ns, k, v)
    return F.Front._immediate_park_due, ns


def test_front_switch_off_is_the_immediate_park():
    fn, ns = _front(running=3, waits_s=[0.2])
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(False):
        assert fn(ns, None, time.time()).rid == "weg2-6-18"
    assert ns.counters["park_collect_holds"] == 0


def test_front_ski_default_prices_live_then_record_then_unmeasured(caplog):
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True):
        # this boot measured a round trip: 2525 + 2218 + 3123 ms = 7.87 s, 3 streams -> 23.6 s
        fn, ns = _front(running=3, waits_s=[10.0], _resume_ms_log=list(WARM_RESUMES), flip_log=_flip_log())
        with caplog.at_level("INFO"):
            assert fn(ns, None, time.time()) is None
        assert ns.counters["park_collect_holds"] == 1
        assert any("PARK-COLLECT-WINDOW HOLD" in m and "ski-live" in m and "price_s=23.6" in m
                   for m in caplog.messages)
        fn, ns = _front(running=3, waits_s=[24.0], _resume_ms_log=list(WARM_RESUMES), flip_log=_flip_log())
        assert fn(ns, None, time.time()).rid == "weg2-6-18"
        assert ns.counters["park_collect_rent"] == 1
        # nothing measured yet this boot: the record of this checkpoint x form seeds it
        seed = {"round_trip_s": 6.94, "boot_tag": "dkr27brow", "at": "2026-09-29 00:40:00,000"}
        fn, ns = _front(running=1, waits_s=[5.0], _park_rt_seed=seed)
        assert fn(ns, None, time.time()) is None
        fn, ns = _front(running=1, waits_s=[7.5], _park_rt_seed=seed)
        assert fn(ns, None, time.time()) is not None
        # neither measured nor recorded: price 0 -- the immediate park, named
        caplog.clear()
        fn, ns = _front(running=3, waits_s=[0.2])
        with caplog.at_level("INFO"):
            assert fn(ns, None, time.time()) is not None
        assert any("PARK-COLLECT-WINDOW FIRE" in m and "why=rent" in m for m in caplog.messages)


def test_front_timer_override_and_caps():
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True), \
            envs.SGLANG_WEG2_PARK_COLLECT_WINDOW_S.override(20.0):
        fn, ns = _front(running=3, waits_s=[0.2], _resume_ms_log=list(WARM_RESUMES), flip_log=_flip_log())
        assert fn(ns, None, time.time()) is None
        fn, ns = _front(running=3, waits_s=[20.5, 3.0])
        assert fn(ns, None, time.time()).rid == "weg2-6-18"
        assert ns.counters["park_collect_timer"] == 1
        # D's decodes ended: the immediate park is not asked at all (it only runs
        # while D runs something) -- the idle D flips on its own path, at once
        fn, ns = _front(running=0, waits_s=[0.2])
        assert fn(ns, None, time.time()) is None and ns.counters["park_collect_holds"] == 0
        # the P phase cap is full: flip at once
        fn, ns = _front(running=3, waits_s=[1, 1, 1, 1, 1, 1])
        assert fn(ns, None, time.time()) is not None
        assert ns.counters["park_collect_cap_requests"] == 1
        # the threshold flag: 20000 pending tokens needed before the window opens
        with envs.SGLANG_WEG2_PARK_COLLECT_THRESHOLD_TOKENS.override(20000):
            fn, ns = _front(running=3, waits_s=[25.0])
            assert fn(ns, None, time.time()) is None


def test_front_collect_window_replaces_the_decode_dwell(caplog):
    """27B review 29.09.: the ski price charges the round trip per running
    stream; PARK-DECODE-DWELL charging it again would cost the same wait twice.
    3 s awake: switch off, the decode dwell (D->P + P->D + resume) holds; switch
    on, only K7's D->P (2.5 s) and the fairness floor remain, and the window
    decides."""
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(False):
        fn, ns = _front(running=3, waits_s=[30.0], awake_s=3.0)
        with caplog.at_level("INFO"):
            assert fn(ns, None, time.time()) is None
        assert ns.counters["park_immediate_dwell_holds"] == 1
    caplog.clear()
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True), \
            envs.SGLANG_WEG2_PARK_COLLECT_WINDOW_S.override(20.0):
        fn, ns = _front(running=3, waits_s=[30.0], awake_s=3.0)
        with caplog.at_level("INFO"):
            fn(ns, None, time.time())
        assert ns.counters["park_immediate_dwell_holds"] == 0
        assert not any("PARK-IMMEDIATE-DWELL" in m for m in caplog.messages)
        assert any("PARK-COLLECT-WINDOW" in m for m in caplog.messages)


def test_warm_resume_median_ignores_the_cold_first_and_one_outlier():
    # the first resume (first park, JIT, pinning) is never a sample
    assert pp.warm_resume_ms([11800.0]) is None
    assert pp.warm_resume_ms([11800.0, 3100.0]) == 3100.0
    # one 10 s outlier among warm resumes does not move the median
    assert pp.warm_resume_ms([11800.0, 3100.0, 3123.0, 10000.0, 3150.0]) == pytest.approx(3136.5)
    # only the last `window` warm samples count
    assert pp.warm_resume_ms([9e3, 1e4, 1e4, 1e4, 3000.0, 3000.0, 3000.0], window=3) == 3000.0


def test_front_one_slow_resume_does_not_blow_up_the_window():
    """bs6 x one cold 10 s resume would be a 60 s window (27B review): the
    live price is the warm median, 7.87 s per stream, not the last sample."""
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True):
        log = list(WARM_RESUMES) + [10000.0]                # the last D phase resumed slowly
        fn, ns = _front(running=6, waits_s=[48.0], awake_s=60.0, _resume_ms_log=log,
                        flip_log=_flip_log())
        rt_s, src = ns._park_collect_window_s()
        assert rt_s == pytest.approx((2525 + 2218 + 3136.5) / 1000.0) and src.startswith("ski-live:")
        assert fn(ns, None, time.time()) is not None       # 48 s >= 6 x 7.88 s: fires
        assert ns.counters["park_collect_rent"] == 1
        # before any warm sample: the cold first flip/resume price nothing -> seed/unmeasured
        fn, ns = _front(running=6, waits_s=[1.0], _resume_ms_log=[11800.0],
                        flip_log=_flip_log(warm_pairs=0)[:1])
        assert ns._park_collect_window_s()[1].startswith("ski-unmeasured")


def test_front_writes_its_first_round_trips_per_form(tmp_path):
    from sglang.srt.weg2 import front as F

    path = tmp_path / "weg2_measured_record.json"
    submitted = []
    ns = types.SimpleNamespace(
        _park_form_key="Qwen3.8-27B|arch=dense", measured_record=str(path), _park_rt_written=0,
        _resume_ms_log=[9800.0], tag="dkr27b", commit="bb82fbcb68", epoch=4,
        flip_log=_flip_log(dp_ms=2167, pd_ms=2421, warm_pairs=0)[:1],
        _sidecar_submit=lambda fn, *a: submitted.append(fn(*a)))
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True):
        # the cold first park (first flip 24.6 s, first resume 9.8 s) is no record
        F.Front._note_park_round_trip(ns)
        assert submitted == [] and not path.exists()
        # warm flips and resumes: the warm medians are written, the cold ones never
        ns.flip_log = _flip_log(dp_ms=2167, pd_ms=2421, warm_pairs=2)
        ns._resume_ms_log = [9800.0, 2351.0, 2360.0, 2340.0]
        for _ in range(pp.PARK_ROUND_TRIP_RECORDS + 2):
            F.Front._note_park_round_trip(ns)
        assert len(submitted) == pp.PARK_ROUND_TRIP_RECORDS            # the first ones only
        got = pp.read_park_round_trip(str(path), "Qwen3.8-27B|arch=dense")
        assert got["round_trip_s"] == pytest.approx(6.939) and got["boot_tag"] == "dkr27b"
        assert got["resume_ms"] == 2351.0 and got["dp_ms"] == 2167.0          # no cold sample in it
        ns._park_form_key, ns._park_rt_written = "", 0                   # no form: nothing written
        F.Front._note_park_round_trip(ns)
        assert len(submitted) == pp.PARK_ROUND_TRIP_RECORDS
