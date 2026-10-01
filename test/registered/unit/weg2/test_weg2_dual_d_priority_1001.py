# SPDX-License-Identifier: Apache-2.0
"""DUAL D PRIORITY UNDER KV PRESSURE (user decision 01.10.): decode is never
interrupted in the dual layout.

Metal gmps7 (dkr27bnvfp4dual1mbar1fs10011748, D 17:53:15): "KV cache pool is full.
Retract requests. #retracted_reqs: 4" -> four W50 re-routes -> PP0's grant for four
contexts -> cuMemCreate OOM, PP0 dead.

DANGER DIRECTIONS guarded here, MUTANT per guard (asserted in-suite):
* (i) group D of the dual layout never retracts: _retract_decode_and_requeue
  raises W-DUAL-D-RETRACT BEFORE the batch is touched; off the dual layout (and on
  group P) the retract runs as before; the two callers that swallow exceptions
  re-raise the named stop.
"""
from __future__ import annotations

import inspect
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import scheduler as SC
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}


class _Touched(Exception):
    """the batch's retract_decode ran (the decode was interrupted)"""


def _retract(monkeypatch, env, fn=None):
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    def _retract_decode(_args):
        raise _Touched()

    batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="weg2-0-1")], retract_decode=_retract_decode)
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 0),
        new_token_ratio_tracker=types.SimpleNamespace(current=0.3),
        tree_cache=types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace()),
        _weg2_d_park_draft_snapshot=lambda b: None,
        server_args=None,
    )
    meth = types.MethodType(fn or SC.Scheduler._retract_decode_and_requeue, sched)
    return meth(batch, kv_full_retract_flag=True)


def test_dual_d_never_retracts_a_decode(monkeypatch):
    with pytest.raises(DP.Weg2DualDRetract, match="W-DUAL-D-RETRACT"):
        _retract(monkeypatch, DUAL_D)


@pytest.mark.parametrize("env", [{}, {"SGLANG_WEG2_GROUP": "D"},
                                 {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}],
                         ids=["no-weg2", "flip-D", "dual-P"])
def test_off_the_dual_d_the_retract_runs_as_before(monkeypatch, env):
    with pytest.raises(_Touched):
        _retract(monkeypatch, env)


def test_the_swallowing_callers_reraise_the_named_stop():
    for name in ("_weg2_d_retract_guard_sites",):
        assert not hasattr(SC.Scheduler, name)  # no second implementation of the guard
    src = inspect.getsource(SC.Scheduler)
    for marker in ('logger.warning("#888b carrier yield failed: %s", e)',
                   'logger.warning("%s rung 3 (retract) failed: %s", self._LADDER_PREFIX, e)'):
        i = src.index(marker)
        head = src[max(0, i - 400):i]
        assert "isinstance(e, Weg2DualDRetract)" in head and "raise" in head, marker


def test_the_guard_removed_mutant_turns_the_retract_test_red(monkeypatch):
    fn = inspect.unwrap(SC.Scheduler._retract_decode_and_requeue)
    src = textwrap.dedent(inspect.getsource(fn))
    guard = "refuse_d_retract(batch, kv_full=kv_full_retract_flag, reason=reason)"
    assert src.count(guard) == 1, "the guard moved -- re-aim the mutant"
    ns = dict(vars(SC))
    exec(compile(src.replace(guard, "pass"), SC.__file__, "exec"), ns)
    with pytest.raises(_Touched):                      # the decode WAS interrupted
        _retract(monkeypatch, DUAL_D, fn=ns["_retract_decode_and_requeue"])


# -- (ii) D's level counts its locked rows (a hold the bookkeeping misses) -------

from sglang.srt.weg2 import dual_d_kv_stage as DK  # noqa: E402


def _d(mapped, free, evictable):
    actor = types.SimpleNamespace(mapped_tokens=mapped,
                                  allocator=types.SimpleNamespace(available_size=lambda: free))
    sched = types.SimpleNamespace(tree_cache=types.SimpleNamespace(evictable_size=lambda: evictable))
    return sched, actor


def test_d_locked_rows_are_mapped_minus_free_minus_evictable():
    sched, actor = _d(mapped=221184, free=4000, evictable=10000)
    assert DK.d_locked_rows(sched, actor) == 221184 - 4000 - 10000


def test_a_hold_the_bookkeeping_misses_keeps_d_from_shrinking_under_it():
    # metal gmps7 17:53:05: 3 running (~156k incl. the queue) + weg2-0-13's #243 hold 61625
    demand, held, air, step = 156232, 61625, 1600 + 6 * 4, 4096
    sched, actor = _d(mapped=221184, free=221184 - demand - held, evictable=0)
    want = DK.want_local_tokens(demand, DK.d_locked_rows(sched, actor), air, step)
    assert want >= demand + held, "the level covers the held rows"
    verdict, _level = DK.decide(221184, want, p_waiting=True, below_rounds=0, step=step)
    assert verdict != "shrink", "D must not give back rows a hold still occupies"


def test_the_bookkeeping_only_mutant_turns_the_hold_test_red(monkeypatch):
    monkeypatch.setattr(DK, "want_local_tokens", lambda demand, locked, air, step:
                        DK.want_tokens(demand, 0, air, step))
    with pytest.raises(AssertionError):
        test_a_hold_the_bookkeeping_misses_keeps_d_from_shrinking_under_it()


def test_the_tick_uses_the_locked_level():
    assert "want_local_tokens(demand_local, d_locked_rows(sched, actor)" in inspect.getsource(DK.tick)


# -- the P stages: order, capability, host floor, hysteresis --------------------

GiB = 1 << 30
STEP_B = 64 << 20          # one P grant step on the tightest card
AIR_B = 32 << 20           # D's look-ahead
W_B = 10 * GiB             # P's weights image (27B NVFP4 dual: PP0 5.93 + PP1 2.20 + PP2 2.44 GB)


def _ticks(st, readings):
    out = []
    for r in readings:
        base = dict(pressure=0, p_committed=0, free_min=0, p_grant_bytes=STEP_B, d_air_bytes=AIR_B,
                    seats_done=0, weights_bytes=W_B, host_ok=True)
        base.update(r)
        out.append(st.tick(**base))
    return out


def test_stage_1_comes_first_and_stage_2_only_after_p_released_everything():
    st = DP.PressureStages(sleep_capable=True, sleep_after=2)
    out = _ticks(st, [dict(pressure=5, p_committed=900),     # D short, P holds KV -> stage 1
                      dict(pressure=5, p_committed=900),     # P still releasing: no sleep
                      dict(pressure=5, p_committed=900),
                      dict(pressure=5, p_committed=0),       # released, held 1
                      dict(pressure=5, p_committed=0)])      # held 2 -> stage 2
    acts = [a for a, _ in out]
    assert acts == ["stop", None, None, None, "sleep"], acts
    assert "stage=1" in out[0][1] and "stage=2" in out[4][1] and "p_state=sleeping" in out[4][1]


def test_without_the_capability_stage_2_says_unavailable_once():
    st = DP.PressureStages(sleep_capable=False, sleep_after=1)
    out = _ticks(st, [dict(pressure=5, p_committed=1), dict(pressure=5), dict(pressure=5)])
    assert [a for a, _ in out] == ["stop", None, None]
    assert out[1][1].endswith("stage=2 d_need=5 freed=0 p_state=stopped unavailable (weights resident)")
    assert out[2][1] is None


def test_the_host_floor_refuses_stage_2_by_name():
    assert DP.host_allows_sleep(17 * GiB, W_B) and not DP.host_allows_sleep(15 * GiB, W_B)
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    out = _ticks(st, [dict(pressure=5, p_committed=1), dict(pressure=5, host_ok=False)])
    assert out[1][0] is None and "refused host_ram" in out[1][1]
    assert st.p_state == "stopped"


def test_stopped_returns_only_when_a_grant_plus_the_look_ahead_is_free():
    st = DP.PressureStages(sleep_capable=True, sleep_after=9)
    out = _ticks(st, [dict(pressure=5, p_committed=1),
                      dict(free_min=STEP_B + AIR_B - 1),       # gap too small: stays stopped
                      dict(free_min=STEP_B + AIR_B)])
    assert [a for a, _ in out] == ["stop", None, "resume"]
    assert "stage=resume" in out[2][1]


def test_asleep_p_returns_only_after_a_seat_ended_and_the_weights_fit():
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    big = W_B + STEP_B + AIR_B
    out = _ticks(st, [dict(pressure=5, p_committed=1, seats_done=3), dict(pressure=5, seats_done=3),
                      dict(free_min=big, seats_done=3),         # no seat ended since the sleep
                      dict(free_min=big - 1, seats_done=4),     # a seat ended, the weights do not fit
                      dict(free_min=big, seats_done=4)])
    assert [a for a, _ in out] == ["stop", "sleep", None, None, "wake"]
    assert "from=sleep" in out[4][1] and st.p_state == "serving"


def _stage_mutant(fixed, back):
    src = textwrap.dedent(inspect.getsource(DP.PressureStages))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant: " + fixed
    ns = dict(vars(DP))
    exec(compile(src.replace(fixed, back), DP.__file__, "exec"), ns)
    return ns["PressureStages"]


@pytest.mark.parametrize("fixed,back,test", [
    ("if p_committed > 0:", "if False:", test_stage_1_comes_first_and_stage_2_only_after_p_released_everything),
    ("if free_min >= int(p_grant_bytes) + int(d_air_bytes):", "if free_min >= 0:",
     test_stopped_returns_only_when_a_grant_plus_the_look_ahead_is_free),
    ("int(seats_done) > self._seat_mark", "True", test_asleep_p_returns_only_after_a_seat_ended_and_the_weights_fit),
    ("if not host_ok:", "if False:", test_the_host_floor_refuses_stage_2_by_name),
], ids=["order", "hysteresis-stopped", "seat-ended", "host-floor"])
def test_each_stage_guard_has_a_red_mutant(monkeypatch, fixed, back, test):
    monkeypatch.setattr(DP, "PressureStages", _stage_mutant(fixed, back))
    with pytest.raises(AssertionError):
        test()


# -- the card loan: P's freed weights bytes while it sleeps -----------------------

import tempfile  # noqa: E402

from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as PK  # noqa: E402


def _ledgers(budget=1 * GiB):
    path = os.path.join(tempfile.mkdtemp(prefix="wkvloan"), "card")
    d = K.CardKvLedger(path, "D")
    d.contribute(budget, committed=0)
    p = K.CardKvLedger(path, "P")
    p.contribute(0)
    return path, d, p


def test_the_loan_grows_the_pool_and_comes_back_only_when_free():
    path, d, p = _ledgers()
    free0 = K.peek(path).free
    p.lend(W_B)
    assert K.peek(path).free == free0 + W_B
    d.request(free0 + 1)                      # D grows into the loan
    assert p.reclaim(W_B) is False, "D committed into the loan: the wake must wait"
    d.release(free0 + 1)
    assert p.reclaim(W_B) is True and K.peek(path).free == free0


def test_sleep_lends_and_wake_reclaims_on_dual_p(monkeypatch):
    path, d, p = _ledgers()
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    actor = types.SimpleNamespace(ledger=p)
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
        **{PK.ACTOR_ATTR: actor})))
    phys = [1 * GiB]
    monkeypatch.setattr(PK, "phys_free_bytes", lambda: phys[0])
    before = PK.sleep_phys_before(sched)
    phys[0] = 1 * GiB + W_B                   # the release freed the weights
    assert PK.sleep_lend(sched, before) == W_B
    free_sleep = K.peek(path).free
    d.request(free_sleep)                     # D takes everything, the loan included
    with pytest.raises(PK.Weg2DualPWakeShort, match="W-DUAL-P-WAKE-SHORT"):
        PK.wake_reclaim(sched)
    d.release(free_sleep)
    assert PK.wake_reclaim(sched) == W_B


def test_off_dual_p_the_sleep_hooks_do_nothing(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    sched = types.SimpleNamespace()
    assert PK.sleep_phys_before(sched) is None and PK.sleep_lend(sched, 5) == 0 and PK.wake_reclaim(sched) == 0


def test_the_unchecked_reclaim_mutant_turns_the_loan_test_red(monkeypatch):
    src = textwrap.dedent(inspect.getsource(K.CardKvLedger.reclaim))
    fixed = "if st.free < n:"
    assert src.count(fixed) == 1
    ns = dict(vars(K))
    exec(compile(src.replace(fixed, "if False:"), K.__file__, "exec"), ns)
    monkeypatch.setattr(K.CardKvLedger, "reclaim", ns["reclaim"])
    with pytest.raises(AssertionError):
        test_the_loan_grows_the_pool_and_comes_back_only_when_free()


# -- the front's P sleep: its own epoch sequence, the flip untouched --------------

import asyncio  # noqa: E402
import collections  # noqa: E402

from sglang.srt.weg2 import front as FR  # noqa: E402


def _front():
    calls = []

    async def leg_rpc(g, path, body, timeout):
        calls.append((g, path, body))
        return 200, "{}"

    async def quiesce(g):
        calls.append(("QUIESCE", g))
        return True, ""

    f = types.SimpleNamespace(groups={"P": "P-group", "D": "D-group"}, boot_epoch="b42", epoch=7,
                              weight_chunks=2, counters=collections.Counter(), leg_rpc=leg_rpc,
                              do_stop=lambda *a: calls.append(("STOP",) + a), quiesce=quiesce)
    for name in ("_dual_p_sleep", "_dual_p_wake", "_dual_p_weights_tags"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    return f, calls


def test_p_sleep_and_wake_use_their_own_epochs_and_leave_the_flip_alone():
    f, calls = _front()
    asyncio.run(f._dual_p_sleep())
    asyncio.run(f._dual_p_wake())
    assert calls[0] == ("QUIESCE", "P-group"), "the group-idle witness comes before the sleep leg"
    (g1, p1, b1), (g2, p2, b2) = calls[1:]
    assert (g1, p1) == ("P-group", "/release_memory_occupation") and (g2, p2) == ("P-group", "/resume_memory_occupation")
    assert b1["tags"][0] == FR.KV_TAG and b2["tags"][-1] == FR.KV_TAG     # KV first to sleep, last to wake
    assert set(b1["tags"][1:]) == set(FR.weights_family_tags(2)) == set(b2["tags"][:-1])
    assert b1["epoch"] == FR.credit_epoch("b42", "dps1") and b2["epoch"] == FR.credit_epoch("b42", "dpw1")
    assert f.epoch == 7, "the flip epoch is not advanced"
    assert not [k for k in f.counters if k.startswith("flip")], "no flip counter moved"


def test_the_launcher_arms_stage_2_only_in_the_dual_unified_form(monkeypatch):
    from sglang.srt.weg2 import launcher as L

    # gmps9: default OFF, and 'on' under --dual-share is a named refusal (P's weights in the private
    # memory-saver pool keep the union bind's freed copies reserved -> P-PP0 KV budget refused)
    assert L.build_parser().parse_args(["--tree", "/t", "--tag", "t"]).dual_p_sleep == "off"
    share_on = types.SimpleNamespace(dual_layout=True, dual_share=True, dual_unified_kv="on", dual_p_sleep="on",
                                     tag="t")
    monkeypatch.setattr(L, "dual_p_sleep_share_supported", lambda: False)   # a tree without steps 1-3
    with pytest.raises(L.Weg2DualPSleepShareRefused, match="W-DUAL-P-SLEEP-SHARE"):
        L.dual_p_sleep_armed(share_on)
    monkeypatch.undo()
    share_off = types.SimpleNamespace(**{**vars(share_on), "dual_p_sleep": "off"})
    assert not L.dual_p_sleep_armed(share_off) and L.dual_p_sleep_argv(share_off, ["--x"]) == ["--x"]
    assert "SGLANG_WEG2_WEIGHTS_RESIDENT" not in L.dual_share_env(share_off, "P")
    assert L.dual_p_sleep_front_env(share_off, 4120) == {}
    on = types.SimpleNamespace(dual_layout=True, dual_share=False, dual_unified_kv="on", dual_p_sleep="on", tag="t")
    assert L.dual_p_sleep_armed(on)
    assert "--enable-weights-cpu-backup" in L.dual_p_sleep_argv(on, ["--x"])
    assert L.dual_p_sleep_front_env(on, 4120) == {DP.P_SLEEP_ENV: "1", "SGLANG_WEG2_DUAL_D_AIR_TOKENS": "4120"}
    assert L.dual_d_air_tokens(6) == 4096 + 6 * 4


def test_apply_dual_p_sleep_pops_the_host_ring_and_adds_the_backup():
    from sglang.srt.weg2 import launcher as L

    on = types.SimpleNamespace(dual_layout=True, dual_share=False, dual_unified_kv="on", dual_p_sleep="on")
    spec = types.SimpleNamespace(argv=["--a"], env={"TMS_HOST_RING_DIR": "/r", "TMS_HOST_RING_MAP": "m", "X": "1"})
    logs = []
    assert L.apply_dual_p_sleep(on, spec, logs.append)
    assert spec.argv == ["--a", "--enable-weights-cpu-backup"] and "TMS_HOST_RING_DIR" not in spec.env
    assert "TMS_HOST_RING_MAP" not in spec.env and spec.env["X"] == "1" and logs
    assert "apply_dual_p_sleep(ns, spec_p, log)" in inspect.getsource(L.main)
