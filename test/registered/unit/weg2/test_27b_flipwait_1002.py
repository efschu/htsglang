"""FLIPWAIT (02.10.2026) -- the 27B flip layout's two user-visible waits before P
may prefill, from boot N3o (desk/27b-n3o-1002 @ 1adfe578de, profile
27b-row-authority-cut43, front log ...dkr27browauthoritycut43bar1fs10020544
_1adfe578de_1002_054455.front.log; 41 DP-WAIT lines, p50 10.0 s, p90 59.5 s,
max 106.4 s; NF nf-int4 on the same day: p50 1.9-2.5 s, max 3.5-7.1 s).

(1) DRAIN 58-105 s. 05:49:00 the acceptance flip probe (docker/probes.py p_flip,
    probes_*.jsonl "flip" wall_s=110.74) POSTed /weg2/flip. handle_manual_flip
    called flip(D, P) past PK2: no MIN-DWELL, no FLIP-ECONOMICS, no PARK-RUNNING;
    "WEG2-FLIP begin epoch=4 sleep=D wake=P outstanding=5 queue=0" and the drain
    waited 106.7 s (FLIP-TIMELINE epoch=5 quiesce@106697) for five agent decodes
    (weg2-2-11: 10550 tokens, 135 s). weg2-4-18..21 arrived meanwhile: DP-WAIT
    drain_s 104.6 / 97.0 / 73.0 / 58.2. Fix: under the immediate park the
    manual flip parks D's running decodes first (PK2's own rule, "a D->P flip
    NEVER drains running decodes -- whatever decided it").
(2) HOLD 20-46 s. PARK-COLLECT-WINDOW HOLD with running=1..4 of d_bs=6 (epochs
    12, 14, 16, 18, 20, 24, 28): the P-bound arrival collected rent or waited
    for the 45 s fairness bound beside free D seats. NF prefills after ~2 s by
    its ARRIVAL-SEAT rule (user 29.09. ~19:40Z: a free seat -> pause and
    prefill now, no collect window), which the 27B line does not carry. Fix
    (switch SGLANG_WEG2_ENABLE_PARK_SEAT_FREE): case (a) of that rule, minimal
    -- a free seat parks at once; NF's MIN-DWELL (y5c) keeps a decode D resumed
    in this phase decoding one measured round trip first.

Hermetic, CPU only. Each test named red_* is red on desk/27b-n3q-1002 @
a8d5c6ef0f.
"""
from __future__ import annotations

import asyncio
import collections
import importlib.util
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import form as FM  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# N3o's measured warm legs (FLIP done epochs 4/6, PARK-ROUND-TRIP 11.6-12.7 s)
FLIP_LOG = ([{"sleep": "D", "wake": "P", "flip_ms": 8496}, {"sleep": "P", "wake": "D", "flip_ms": 2743}]
            + [{"sleep": "D", "wake": "P", "flip_ms": 7597}, {"sleep": "P", "wake": "D", "flip_ms": 2241}] * 3)
RESUMES = [663.0, 1759.0, 2855.0, 2049.0]


def _front(running, waits_s, d_bs=6, awake_s=30.0, uncached=6472, ready=0, handoff=0,
           resumed=(), **attrs):
    """A namespace front for :meth:`Front._immediate_park_due` (the park
    collect window's own harness shape, test_weg2_park_collect_window_0929)."""
    from sglang.srt.weg2 import front as F

    now = time.time()
    t_awake = now - awake_s
    run = [f"weg2-15-{j}" for j in range(running)]
    ns = types.SimpleNamespace(
        epoch=16, _park_attempt_epoch=-1, _park_resume_epoch=-1, _park_immediate_dwell_epoch=-1,
        _park_collect_epoch=-1, _resume_ms_log=list(RESUMES), flip_log=list(FLIP_LOG),
        _park_rt_seed=None, _park_form_key="Qwen3.8-27B-INT8|arch=dense", t_awake=t_awake,
        _d_decode_epoch=16, _d_decode_t0=t_awake + 1.0,
        queue=[types.SimpleNamespace(rid=f"weg2-16-{39 + j}", est_uncached=uncached, t_arrive=now - w,
                                     p_only=False, x_requeues=0, leg1_done=False, skip_leg1=False,
                                     x_deferred=False)
               for j, w in enumerate(waits_s)],
        tp_prefill_max_tokens=3797, counters=collections.Counter(), d_bs=d_bs,
        p_phase_max_requests=6, p_pool_tokens=262144, d_wait_bound_s=0.0,
        _ready_for_d=[types.SimpleNamespace(rid=f"weg2-14-{j}", fut=None) for j in range(ready)],
        _handoff_in_flight=lambda: handoff,
        _seat_resumed_epoch={r: 16 for r in resumed},
        _flip_ledger=lambda g: list(run),
        _derived_min_dwell_ms=lambda s, d: ((2000, "warm-D->P") if (s, d) == ("D", "P")
                                            else (2241, "warm-P->D")),
    )
    ns._park_collect_window_s = lambda: F.Front._park_collect_window_s(ns)
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    for k, v in attrs.items():
        setattr(ns, k, v)
    return F.Front._immediate_park_due, ns


def _on():
    return (envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True),
            envs.SGLANG_WEG2_ENABLE_PARK_SEAT_FREE.override(True))


# ------------------------------------------------------------ (2) PARK-SEAT-FREE
def test_red_a_free_seat_parks_now_instead_of_collecting(caplog):
    """N3o epoch 16 05:53:39: weg2-16-39 (uncached 6472 > X 3797), running=2 of 6,
    round trip 11.37 s -> base: HOLD (price 22.7 s, fired 20.2 s later)."""
    a, b = _on()
    with a, b:
        fn, ns = _front(running=2, waits_s=[0.5])
        with caplog.at_level("INFO"):
            got = fn(ns, None, time.time())
        assert got is not None and got.rid == "weg2-16-39"
        assert ns.counters["park_seat_free_fired"] == 1
        assert ns.counters["park_collect_holds"] == 0
        assert any("WEG2 PARK-SEAT-FREE FIRE" in m and "taken=2 n=6" in m for m in caplog.messages)


def test_no_free_seat_keeps_the_collect_window(caplog):
    """N3o epoch 2 05:48:15: running=6 of 6 -> the collect window decides (HOLD)."""
    a, b = _on()
    with a, b:
        fn, ns = _front(running=6, waits_s=[0.5])
        with caplog.at_level("INFO"):
            assert fn(ns, None, time.time()) is None
        assert ns.counters["park_seat_free_fired"] == 0 and ns.counters["park_collect_holds"] == 1
        assert any("PARK-COLLECT-WINDOW HOLD" in m for m in caplog.messages)
        # hand-offs in flight and prefilled requests waiting for D take seats too
        fn, ns = _front(running=3, waits_s=[0.5], ready=2, handoff=1)
        assert fn(ns, None, time.time()) is None and ns.counters["park_collect_holds"] == 1
        fn, ns = _front(running=3, waits_s=[0.5], ready=1, handoff=1)
        assert fn(ns, None, time.time()) is not None and ns.counters["park_seat_free_fired"] == 1


def test_switch_off_is_the_collect_window_byte_for_byte():
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(True), \
            envs.SGLANG_WEG2_ENABLE_PARK_SEAT_FREE.override(False):
        fn, ns = _front(running=2, waits_s=[0.5])
        assert fn(ns, None, time.time()) is None
        assert ns.counters["park_collect_holds"] == 1 and ns.counters["park_seat_free_fired"] == 0
    # collect window off: the switch reads nothing -- the immediate park as before
    with envs.SGLANG_WEG2_ENABLE_PARK_COLLECT_WINDOW.override(False), \
            envs.SGLANG_WEG2_ENABLE_PARK_SEAT_FREE.override(True):
        fn, ns = _front(running=6, waits_s=[0.5])
        assert fn(ns, None, time.time()) is not None and ns.counters["park_seat_free_fired"] == 0


def test_the_immediate_park_dwell_still_comes_first():
    """D woke 1 s ago (floor 2 s): no park yet, free seat or not."""
    a, b = _on()
    with a, b:
        fn, ns = _front(running=2, waits_s=[0.5], awake_s=1.0)
        assert fn(ns, None, time.time()) is None
        assert ns.counters["park_immediate_dwell_holds"] == 1 and ns.counters["park_seat_free_fired"] == 0


def test_danger_a_resumed_decode_is_not_parked_again_before_one_round_trip(caplog):
    """DANGER DIRECTION (NF y5c 30.09.: weg2-0-2 parked 6x, client gone at 235 s):
    a decode D resumed in this phase must decode one measured round trip
    (here 7.6 + 2.2 + 1.9 s = 11.7 s) before a seat-free park stops it again."""
    a, b = _on()
    with a, b:
        fn, ns = _front(running=2, waits_s=[0.5], awake_s=5.0, resumed=("weg2-15-0",))
        with caplog.at_level("INFO"):
            assert fn(ns, None, time.time()) is None
        assert ns.counters["park_seat_free_min_dwell_hold"] == 1
        assert ns._park_dwell_held_t > 0
        assert any("PARK-SEAT-FREE MIN-DWELL hold" in m and "resumed_rid=weg2-15-0" in m
                   for m in caplog.messages)
        # past one round trip: the park comes
        fn, ns = _front(running=2, waits_s=[0.5], awake_s=12.5, resumed=("weg2-15-0",))
        assert fn(ns, None, time.time()) is not None and ns.counters["park_seat_free_fired"] == 1
        # the resumed decode has ended: nothing to protect, the park comes at once
        fn, ns = _front(running=2, waits_s=[0.5], awake_s=5.0, resumed=("weg2-9-9",))
        assert fn(ns, None, time.time()) is not None
        # nothing measured and no record: never a constant, no hold
        fn, ns = _front(running=2, waits_s=[0.5], awake_s=5.0, resumed=("weg2-15-0",),
                        flip_log=[], _resume_ms_log=[])
        assert fn(ns, None, time.time()) is not None


def test_pure_terms():
    assert pp.seat_free(5, 6) and not pp.seat_free(6, 6) and not pp.seat_free(0, 0)
    assert pp.resumed_min_dwell_hold({"a": 100.0}, ["a", "b"], 105.0, 11.7) == ("a", 5.0)
    assert pp.resumed_min_dwell_hold({"a": 100.0}, ["a"], 112.0, 11.7) is None
    assert pp.resumed_min_dwell_hold({"a": 100.0}, ["b"], 101.0, 11.7) is None
    assert pp.resumed_min_dwell_hold({"a": 100.0}, ["a"], 101.0, None) is None
    assert pp.resumed_min_dwell_hold({"a": 100.0, "c": 104.0}, ["a", "c"], 105.0, 11.7) == ("c", 1.0)


# ------------------------------------------------------------ (1) manual flip
def _h91c():
    spec = importlib.util.spec_from_file_location(
        "_flipwait_h91c", os.path.join(HERE, "test_weg2_phase_policy_h91c.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _form_env():
    return FM.Weg2Form(arch="dense", experts="none", draft="dflash", p_draft="none", kv="paged_dcp",
                       flip="family", vision="off", profile="qwen27b", model="m").env_value()


@pytest.fixture
def q27(monkeypatch):
    for k in ("SGLANG_WEG2_D_PARK_IMMEDIATE", "SGLANG_WEG2_STANDARD_FORM", "SGLANG_WEG2_D_PARK",
              "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(FM.FORM_ENV, _form_env())
    return monkeypatch


def test_red_a_manual_flip_under_the_immediate_park_parks_instead_of_draining(q27):
    """05:49:00 shape: r0 decodes on D (held, like weg2-2-11's 135 s), the queue
    is empty, POST /weg2/flip. Base: flip(D, P) drains r0 -- the handler never
    returns while r0 decodes. Fix: PARK-MANUAL-FLIP, then both flips; r0
    resumes after the flip back and is served once."""
    H = _h91c()

    async def body():
        async with H.Harness(awake="P", p_concurrency=4, d_bs=2, d_park_immediate=True) as h:
            h.d.hold = {}
            t0 = h.post("r0")
            assert await H._until(lambda: h.d.running and h.front.awake == "D", 20)
            assert not h.front.queue
            flip = asyncio.create_task(h.front.handle_manual_flip(None))
            done, _ = await asyncio.wait({flip}, timeout=15)
            assert flip in done, h.d.timeline              # base: still draining r0
            assert flip.result().status == 200
            assert [b["reason"] for b in h.d.park_bodies] == [pp.PARK_REASON_MANUAL]
            tl = h.d.timeline
            i_park = H._first(tl, "rpc:weg2/park_running")
            assert [i for i, e in enumerate(tl) if e == "rpc:release_memory_occupation" and i > i_park]
            assert h.front.counters["park_manual_flip"] == 1
            assert not t0.done()                            # in flight, parked, never cut
            assert await H._until(lambda: h.front.awake == "D" and not h.front._d_parked, 20)
            h.d.release_all()
            s0, _ = await asyncio.wait_for(t0, 20)
            assert s0 == 200 and h.d.gen_marks.count("r0") == 1   # no second leg, no re-prefill

    asyncio.run(body())


def test_manual_flip_without_the_immediate_park_drains_as_before(q27):
    """Switch off (NF x177 / any non-park form): the handler is unchanged."""
    H = _h91c()

    async def body():
        async with H.Harness(awake="P", p_concurrency=4, d_bs=2) as h:
            h.d.hold = {}
            t0 = h.post("r0")
            assert await H._until(lambda: h.d.running and h.front.awake == "D", 20)
            flip = asyncio.create_task(h.front.handle_manual_flip(None))
            await asyncio.sleep(2.0)
            assert not flip.done() and h.d.park_bodies == []
            h.d.release_all()
            await asyncio.wait_for(flip, 20)
            s0, _ = await asyncio.wait_for(t0, 20)
            assert s0 == 200

    asyncio.run(body())
