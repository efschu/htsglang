"""27B Dual round 2 (boot dkr27bnvfp4dualbar1drei10220138, image 31c8d07d72): P asleep, LONG waits, 9 min nothing.

front.log 01:42:32,818 ``PDFLIP DUAL-KV-PRESSURE d_signal id_space=0.99 -- D is short beyond the ledger bytes`` (D maps
8192 of its 1M-row id space; the table fill counts the rows D never mapped) -> 01:42:32,822 ``stage=1`` -> 01:42:33,029
``stage=1-lend`` -> 01:42:33,649 ``stage=2 ... p_state=sleeping`` (P SLEEP-LEND 6312427520 / 5131730944 / 4846518272 B).
From 01:42:43 every request is ``LONG -> P leg 1 / BATCH queued``; D gets no seat, so ``d_seat_done`` never moves, and
the sleeping wake waits for a seat end that cannot come. 0 Prefill / 0 Decode batches until the stop at 01:52Z.

31c8d07d72 made the front SEE D's signal file (own-tag reading); the companions built for exactly that (#1540
D-SIGNAL-SEATS, #1976 D-SIGNAL-LOCKED, B9e / f11) were default off. Fix: both default on, and the wake from sleep
counts an idle D (no live seat, nothing waiting for one) like a finished seat.
"""

import asyncio
import collections
import os
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_d_priority as DP
from flliper.srt.pdflip import dual_p_kv_stage as PK
from flliper.srt.pdflip import front as FR
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
TAG = "dkr27bnvfp4dualbar1drei10220138"
SLEEP_LOANS = (6312427520, 5131730944, 4846518272)   # P.log 01:42:36 SLEEP-LEND freed=... (PP0 / PP1 / PP2)
STEP_B = 64 << 20
LONG = 6                                             # front.log: BATCH queued, queue 6-8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_DUAL_KV_TAG", "FLLIPER_PDFLIP_TAG", "FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP",
              "FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", "FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED",
              "FLLIPER_PDFLIP_DUAL_D_AIR_TOKENS", "FLLIPER_PDFLIP_DUAL_WAKE_SEES_LOAN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(DP.P_SLEEP_ENV, "1")         # the dual arm: PDFLIP-DUAL-P-SLEEP armed
    tmp = tempfile.mkdtemp(prefix="wkv1022b")
    monkeypatch.setattr(PK, "stage_file", lambda tag, r, root="": os.path.join(tmp, "%s-pp%d.json" % (tag, int(r))))
    monkeypatch.setattr(DP, "d_signal_file", lambda tag, root="": os.path.join(tmp, "wkvd-%s.json" % tag))
    return tmp


def _cards(asleep: bool):
    paths = []
    for r in range(3):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvc1022b"), "card")
        d = K.CardKvLedger(path, "D")
        d.contribute(4 * GiB, committed=64 << 20)      # D idle: boot stage mapped, nothing grown
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        actor = types.SimpleNamespace(ledger=p, mapped_tokens=0, step=4096, top=266240,
                                      table=lambda: [0, STEP_B], weights_bytes=SLEEP_LOANS[r])
        if asleep:
            p.lend(SLEEP_LOANS[r])
            actor._sleep_lent = SLEEP_LOANS[r]
        PK.publish_stage(actor, TAG, r)
        paths.append(path)
    return paths


def _publish_d(id_frac=0.99, locked_frac=0.0001):
    DP.publish_d_signal(DP.d_signal_file(TAG), id_frac=id_frac, arena_complete=0, arena_slots=8, now=time.time(),
                        locked_frac=locked_frac, arena_pinned=0)


def _front(paths, p_state: str):
    calls = []
    f = types.SimpleNamespace(dual_kv_ledgers=list(paths), tag=TAG, counters=collections.Counter(),
                              queue=collections.deque(range(LONG)), _dual_inflight=[], _d_seats_live=set(),
                              _ready_for_d=collections.deque(), DUAL_LEND_GATE_LOG_S=FR.Front.DUAL_LEND_GATE_LOG_S)
    for name in ("_dual_stages", "_dual_stage_tick", "_dual_p_stage_reading", "_dual_lend_gate_lent",
                 "_dual_wake_room_stages"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    f._dual_p_sleep_probe_tick = lambda pressure: False

    def _rec(name):
        async def _a(*_a, **_k):
            calls.append(name)
        return _a

    f._dual_p_sleep, f._dual_p_wake, f._dual_p_lend = _rec("sleep"), _rec("wake"), _rec("lend")
    st = f._dual_stages()
    st.p_state = p_state
    if p_state == "sleeping":
        st._seat_mark = 0
    return f, st, calls


def _run(f, n=12):
    async def _go():
        for _ in range(n):
            f._dual_stage_tick(0)
            await asyncio.sleep(0)
    asyncio.run(_go())


def test_boot_standing_id_space_without_d_work_does_not_put_p_to_sleep():
    """01:42:32: no D seat, no request waiting for one, ledger without D demand -- the standing table fill is not
    D's pressure: no stage 1, no lend, no sleep."""
    paths = _cards(asleep=False)
    _publish_d(id_frac=0.99)
    f, st, calls = _front(paths, "serving")
    _run(f)
    assert st.p_state == "serving" and calls == []
    assert f.counters["dual_kv_pressure_stage1"] == 0


@pytest.mark.parametrize("signal", ["standing_0.99", "none"])
def test_p_asleep_long_waits_d_idle_wakes_p(signal):
    """The case of round 2: P asleep (loans on every card), LONG requests wait for leg 1, D idle. The wake must not
    wait for a D seat end that cannot come."""
    paths = _cards(asleep=True)
    if signal != "none":
        _publish_d(id_frac=0.99)
    f, st, calls = _front(paths, "sleeping")
    _run(f)
    assert calls == ["wake"], "P wakes once (front: stage=resume ... from=sleep)"
    assert st.p_state == "serving"


def test_p_asleep_with_a_live_d_seat_still_waits_for_its_end():
    """Unchanged where it holds: a live D seat keeps P asleep until a seat ends (D gives its rows back)."""
    paths = _cards(asleep=True)
    f, st, calls = _front(paths, "sleeping")
    f._d_seats_live = {"seat"}
    _run(f)
    assert calls == [] and st.p_state == "sleeping"
    f.counters["d_seat_done"] += 1
    _run(f, 1)
    assert calls == ["wake"]


def test_pure_wake_rule_d_idle():
    st = DP.PressureStages(sleep_capable=True)
    st.p_state, st._seat_mark = "sleeping", 0
    kw = dict(pressure=0, p_committed=0, free_min=20 * GiB, p_grant_bytes=STEP_B, d_air_bytes=STEP_B,
              seats_done=0, weights_bytes=6 * GiB)
    assert st.tick(**kw) == (None, None), "default d_idle=False: the old rule"
    action, line = st.tick(d_idle=True, **kw)
    assert action == "wake" and "from=sleep" in line
    st.p_state = "sleeping"
    assert st.tick(d_idle=True, **dict(kw, pressure=1)) == (None, None), "pressure still holds P asleep"
