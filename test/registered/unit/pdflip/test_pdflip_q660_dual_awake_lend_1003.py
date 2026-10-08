# SPDX-License-Identifier: Apache-2.0
"""Q-660 DUAL-AWAKE-LEND: in the 27B NVFP4 dual (P and D active at the same time,
no flip) P gives its KV to D when D runs short -- AWAKE, not only asleep (user
rule 03.10. ~15:35Z: "wenn auf D kv knapp wird, gibt P seinen kv auf").

Metal y8v (fs10031504, 15:20:53): D at "full token usage 0.96", the Mamba arena
full (complete=112 of 112), the card ledgers with free bytes and no D demand ->
no stage ever fired; and even a fired stage 1 lent P's bytes only at the sleep.

Order under test: stop (P pauses at its chunk boundary, releases its KV) ->
lend (awake, P's freed device bytes join the card pool, D grows into them) ->
sleep (only if the pressure still holds) ; return: reclaim after a calm hold
or a D seat that ended, P prefills again only once the loan is back. D never
retracts.

DANGER DIRECTION: D short while P sits awake on free bytes (the y8v stall), or
P taking the loan back under D's pressure, or P prefilling before its loan is
back.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import json
import os
import tempfile
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_d_kv_stage as DK
from flliper.srt.pdflip import dual_d_priority as DP
from flliper.srt.pdflip import dual_p_kv_stage as PK
from flliper.srt.pdflip import front as FR
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
STEP_B = 256 << 20         # one P grant step
AIR_B = 128 << 20          # D's look-ahead
W_B = 10 * GiB             # P's weights image


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP", "FLLIPER_PDFLIP_DUAL_P_SLEEP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_KV_TAG", "q660-test-%d" % os.getpid())


def _ticks(st, readings):
    out = []
    for r in readings:
        base = dict(pressure=0, p_committed=0, free_min=0, p_grant_bytes=STEP_B, d_air_bytes=AIR_B,
                    seats_done=0, weights_bytes=W_B, host_ok=True)
        base.update(r)
        out.append(st.tick(**base))
    return out


# -- the stage machine: stop -> lend -> sleep ; reclaim -> resume ----------------

def test_the_order_is_stop_then_lend_awake_then_sleep():
    st = DP.PressureStages(sleep_capable=True, sleep_after=2)
    out = _ticks(st, [dict(pressure=5, p_committed=900),     # D short -> stage 1: P stops
                      dict(pressure=5, p_committed=900),     # P still releasing
                      dict(pressure=5, p_committed=0),       # released -> lend at once, awake
                      dict(pressure=5),                      # the loan did not cover D: held 1
                      dict(pressure=5)])                     # held 2 -> only now stage 2
    assert [a for a, _ in out] == ["stop", None, "lend", None, "sleep"]
    assert "stage=1-lend" in out[2][1] and "p_state=lent" in out[2][1]
    assert st.counts["lend"] == 1 and st.counts["stage2"] == 1


def test_the_loan_that_covers_d_keeps_p_awake():
    st = DP.PressureStages(sleep_capable=True, sleep_after=2, reclaim_after=50)
    out = _ticks(st, [dict(pressure=5, p_committed=1), dict(pressure=5),
                      dict(pressure=0), dict(pressure=0), dict(pressure=0)])   # D fits after the loan
    assert [a for a, _ in out] == ["stop", "lend", None, None, None]
    assert st.p_state == "lent" and st.counts["stage2"] == 0, "no sleep when the awake loan was enough"


def test_without_pressure_after_the_hold_p_reclaims_and_resumes_only_with_the_loan_back():
    st = DP.PressureStages(sleep_capable=True, sleep_after=9, reclaim_after=3)
    room = STEP_B + AIR_B
    out = _ticks(st, [dict(pressure=5, p_committed=1), dict(pressure=5),           # stop, lend
                      dict(free_min=room), dict(free_min=room),                      # calm 1, 2
                      dict(free_min=room, p_lent=GiB),                               # calm 3 -> reclaim
                      dict(free_min=room, p_lent=GiB),                               # loan not back yet
                      dict(free_min=room, p_lent=0)])                                # back -> resume
    assert [a for a, _ in out] == ["stop", "lend", None, None, "reclaim", None, "resume"]
    assert "stage=1-reclaim" in out[4][1] and "from=lend" in out[6][1] and st.p_state == "serving"


def test_a_d_seat_that_ended_reclaims_before_the_hold():
    st = DP.PressureStages(sleep_capable=True, sleep_after=9, reclaim_after=50)
    room = STEP_B + AIR_B
    out = _ticks(st, [dict(pressure=5, p_committed=1, seats_done=4), dict(pressure=5, seats_done=4),
                      dict(free_min=room, seats_done=4), dict(free_min=room, seats_done=5)])
    assert [a for a, _ in out] == ["stop", "lend", None, "reclaim"]


def test_pressure_during_the_reclaim_leaves_the_loan_with_d():
    st = DP.PressureStages(sleep_capable=True, sleep_after=9, reclaim_after=1)
    room = STEP_B + AIR_B
    out = _ticks(st, [dict(pressure=5, p_committed=1), dict(pressure=5),
                      dict(free_min=room, p_lent=GiB),                               # -> reclaim
                      dict(pressure=5, p_lent=GiB)])                                 # D presses again
    assert [a for a, _ in out] == ["stop", "lend", "reclaim", None]
    assert st.p_state == "lent", "the loan stays with D, P does not take it back under pressure"


def test_no_reclaim_without_room_on_every_card():
    st = DP.PressureStages(sleep_capable=True, sleep_after=9, reclaim_after=1)
    out = _ticks(st, [dict(pressure=5, p_committed=1), dict(pressure=5),
                      dict(free_min=10 * GiB, card_room=[(10 * GiB, GiB), (GiB - 1, GiB)])])
    assert [a for a, _ in out] == ["stop", "lend", None]


def _stage_mutant(fixed, back):
    src = textwrap.dedent(inspect.getsource(DP.PressureStages))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant: " + fixed
    ns = dict(vars(DP))
    exec(compile(src.replace(fixed, back), DP.__file__, "exec"), ns)
    return ns["PressureStages"]


@pytest.mark.parametrize("fixed,back,test", [
    ("if int(p_lent) <= 0:", "if True:",
     test_without_pressure_after_the_hold_p_reclaims_and_resumes_only_with_the_loan_back),
    ('self.p_state = "lent"\n                self._held = 0', 'self.p_state = "sleeping"\n                self._held = 0',
     test_the_order_is_stop_then_lend_awake_then_sleep),
], ids=["resume-before-loan-back", "lend-skipped"])
def test_each_q660_guard_has_a_red_mutant(monkeypatch, fixed, back, test):
    monkeypatch.setattr(DP, "PressureStages", _stage_mutant(fixed, back))
    with pytest.raises(AssertionError):
        test()


# -- the ledger: P awake lends, D grows; the loan comes back only when free ------

def _ledgers(budget=1 * GiB):
    path = os.path.join(tempfile.mkdtemp(prefix="wkvq660"), "card")
    d = K.CardKvLedger(path, "D")
    d.contribute(budget, committed=0)
    p = K.CardKvLedger(path, "P")
    p.contribute(0)
    return path, d, p


def _p_rank(monkeypatch, ledger, mapped=0):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    actor = types.SimpleNamespace(ledger=ledger, mapped_tokens=mapped)
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
        **{PK.ACTOR_ATTR: actor})))
    published = []
    monkeypatch.setattr(PK, "_republish_stage", lambda s, a: published.append(PK.lent_bytes(a)))
    return sched, actor, published


def test_d_short_and_p_awake_p_lends_and_d_grows(monkeypatch):
    path, d, p = _ledgers()
    sched, actor, published = _p_rank(monkeypatch, p)
    budget = K.peek(path).free
    assert d.request(budget)[0] == budget                       # D's pool is the whole card
    granted, _ = d.request(2 * GiB)                             # D short: nothing left
    assert granted == 0 and K.peek(path).demand["D"] == 2 * GiB
    phys = [64 << 20]

    def _empty():
        phys[0] += 2 * GiB                                      # P's released KV + allocator cache

    lent = PK.awake_lend(sched, "test", phys=lambda: phys[0], empty_cache=_empty)
    assert lent == 2 * GiB and PK.lent_bytes(actor) == 2 * GiB and published == [2 * GiB]
    granted, _ = d.request(2 * GiB)                             # D grows into the awake loan
    assert granted == 2 * GiB and K.peek(path).demand["D"] == 0


def test_the_awake_loan_comes_back_only_when_d_left_it(monkeypatch):
    path, d, p = _ledgers()
    sched, actor, _ = _p_rank(monkeypatch, p)
    phys = [0]
    PK.awake_lend(sched, phys=lambda: phys[0], empty_cache=lambda: phys.__setitem__(0, GiB))
    d.request(K.peek(path).free)                                # D holds the loan
    assert PK.awake_reclaim(sched, "calm") == 0 and PK.lent_bytes(actor) == GiB, "never under D"
    d.release(GiB)
    assert PK.awake_reclaim(sched, "calm") == GiB and PK.lent_bytes(actor) == 0


def test_p_with_live_kv_does_not_lend(monkeypatch):
    _path, _d, p = _ledgers()
    sched, actor, published = _p_rank(monkeypatch, p, mapped=4096)
    assert PK.awake_lend(sched, phys=lambda: 0, empty_cache=lambda: None) == 0 and published == []


def test_the_wake_takes_back_both_loans(monkeypatch):
    path, d, p = _ledgers()
    sched, actor, _ = _p_rank(monkeypatch, p)
    phys = [0]
    PK.awake_lend(sched, phys=lambda: phys[0], empty_cache=lambda: phys.__setitem__(0, GiB))
    monkeypatch.setattr(PK, "phys_free_bytes", lambda: phys[0])
    before = PK.sleep_phys_before(sched)
    phys[0] += W_B
    PK.sleep_lend(sched, before)
    assert PK.lent_bytes(actor) == GiB + W_B
    assert PK.wake_reclaim(sched) == GiB + W_B and PK.lent_bytes(actor) == 0


def test_d_never_retracts_the_ladder_answers_with_the_loan(monkeypatch):
    """D short: the stages stop P, P lends awake and D's request is granted -- the
    D side has nothing to retract; the retract guard on a dual D rank still stops
    named (it was never loosened)."""
    path, d, p = _ledgers()
    sched, actor, _ = _p_rank(monkeypatch, p)
    p_held = 512 << 20
    d.request(K.peek(path).free - p_held)
    assert p.request(p_held)[0] == p_held                      # P holds its share
    short, _ = d.request(GiB)
    st = DP.PressureStages(sleep_capable=False)
    phys = [0]
    acts = []
    for _ in range(6):
        demand = int(K.peek(path).demand["D"])
        act, _line = st.tick(pressure=demand, p_committed=int(K.peek(path).committed["P"]), free_min=0,
                             p_grant_bytes=STEP_B, d_air_bytes=AIR_B, seats_done=0)
        acts.append(act)
        if act == "stop":
            p.release(p_held)                                  # stage 1: P releases its KV ...
            phys[0] += p_held
        elif act == "lend":                                    # ... and lends the freed bytes awake
            PK.awake_lend(sched, phys=lambda: phys[0], empty_cache=lambda: phys.__setitem__(0, phys[0] + GiB))
            d.request(GiB)
    assert acts[:2] == ["stop", "lend"] and "retract" not in acts
    assert K.peek(path).demand["D"] == 0, "D fits after the awake loan"
    batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="pdflip-0-1")])
    with pytest.raises(DP.PdFlipDualDRetract):
        DP.refuse_d_retract(batch, kv_full=True, reason="test",
                            env={"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"})


# -- D's own shortage beyond the ledger bytes (the y8v case) ----------------------

def test_d_signal_counts_id_space_and_arena_only_when_fresh():
    now = 1000.0
    sig = dict(ts=now, id_frac=0.96, arena_complete=112, arena_slots=112)
    short, why = DP.d_signal_short(sig, now=now, unit=STEP_B)
    assert short == STEP_B and "id_space=0.96" in why and "arena=112/112" in why
    assert DP.d_signal_short(dict(sig, id_frac=0.5, arena_complete=100), now=now, unit=STEP_B) == (0, "")
    assert DP.d_signal_short(dict(sig, id_frac=0.5, arena_complete=110), now=now, unit=STEP_B)[0] == STEP_B
    assert DP.d_signal_short(sig, now=now + DP.D_SIGNAL_MAX_AGE_S + 1, unit=STEP_B) == (0, ""), "stale"
    assert DP.d_signal_short(None, now=now, unit=STEP_B) == (0, "")


def test_d_publishes_its_signal_and_the_front_reads_it_as_d_short(monkeypatch):
    tag = os.environ["FLLIPER_PDFLIP_DUAL_KV_TAG"]
    sig_path = DP.d_signal_file(tag)
    monkeypatch.setattr(DK, "_arena_census", lambda: (112, 112))
    actor = types.SimpleNamespace(allocator=types.SimpleNamespace(size=1000, available_size=lambda: 30))
    sched = types.SimpleNamespace(tp_rank=0, tree_cache=types.SimpleNamespace(evictable_size=lambda: 10))
    try:
        DK.publish_d_signal(sched, actor)
        with open(sig_path) as f:
            sig = json.load(f)
        assert abs(sig["id_frac"] - 0.96) < 1e-9 and sig["arena_complete"] == 112
        path, _d, _p = _ledgers()                              # free ledger bytes, no D demand (y8v)
        f = types.SimpleNamespace(dual_kv_ledgers=[path])
        rd = FR.Front._dual_p_stage_reading(f)
        assert rd["d_short"] > 0, "D's id space and arena press P although the ledger shows no demand"
        assert rd["p_lent"] == 0
    finally:
        for x in (sig_path,):
            try:
                os.unlink(x)
            except OSError:
                pass


def test_d_signal_only_from_tp0(monkeypatch):
    monkeypatch.setattr(DP, "publish_d_signal", lambda *a, **k: pytest.fail("a worker rank published"))
    DK.publish_d_signal(types.SimpleNamespace(tp_rank=1), types.SimpleNamespace())


# -- the front: lend/reclaim go to P as /pdflip/dual_p_lend ---------------------------

def _front(readings):
    calls = []

    async def leg_rpc(g, path, body, timeout):
        calls.append((g, path, body))
        return 200, "{}"

    it = iter(readings)
    f = types.SimpleNamespace(groups={"P": "P-group"}, counters=collections.Counter(), leg_rpc=leg_rpc)
    f._dual_stages_obj = DP.PressureStages(sleep_capable=False, reclaim_after=1)
    f._dual_p_sleep_probe_tick = lambda pressure: False
    f._dual_p_stage_reading = lambda: dict(next(it))
    for name in ("_dual_stages", "_dual_stage_tick", "_dual_p_lend"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    return f, calls


def test_the_front_sends_lend_and_reclaim_to_p_and_counts_them():
    r = dict(p_committed=0, free_min=GiB, p_grant_bytes=STEP_B, d_air_bytes=AIR_B, weights_bytes=W_B,
             card_room=None, p_lent=0)
    f, calls = _front([dict(r, p_committed=1, d_short=0), dict(r, d_short=STEP_B),
                       dict(r, d_short=0, p_lent=GiB)])

    async def _run():
        eff = [f._dual_stage_tick(5), f._dual_stage_tick(0), f._dual_stage_tick(0)]
        await asyncio.sleep(0)
        return eff

    eff = asyncio.run(_run())
    assert eff == [5, STEP_B, 0], "the tick hands the pump D's shortage for the pause of leg 1"
    assert [(g, p, b["action"]) for g, p, b in calls] == [("P-group", "/pdflip/dual_p_lend", "lend"),
                                                         ("P-group", "/pdflip/dual_p_lend", "reclaim")]
    assert f.counters["dual_kv_pressure_stage1"] == 1 and f.counters["dual_kv_pressure_lend"] == 1
    assert f.counters["dual_kv_pressure_reclaim"] == 1


def test_the_p_ranks_route_the_order_to_the_stage(monkeypatch):
    from flliper.srt.managers import io_struct as IO
    from flliper.srt.managers import scheduler as SC

    seen = []
    monkeypatch.setattr(PK, "awake_lend", lambda s, why="": seen.append(("lend", why)) or 0)
    monkeypatch.setattr(PK, "awake_reclaim", lambda s, why="": seen.append(("reclaim", why)) or 0)
    sc = types.SimpleNamespace()
    SC.Scheduler.handle_pdflip_dual_p_lend(sc, IO.PdFlipDualPLendReqInput(action="lend", why="w"))
    SC.Scheduler.handle_pdflip_dual_p_lend(sc, IO.PdFlipDualPLendReqInput(action="reclaim"))
    assert seen == [("lend", "w"), ("reclaim", "front")]
    src = inspect.getsource(SC.Scheduler.init_request_dispatcher)
    assert "(PdFlipDualPLendReqInput, self.handle_pdflip_dual_p_lend)" in src
    with open(os.path.join(os.path.dirname(SC.__file__), "..", "entrypoints", "http_server.py")) as fh:
        assert '"/pdflip/dual_p_lend"' in fh.read()
