# SPDX-License-Identifier: Apache-2.0
"""#1956 WAKE-SEES-LOAN: the dual front wakes P from sleep only when every card has
P's loan free again (per-card card_room), never on the blind total fallback.

Metal f9 (rc12z30y9f9 = 94f94b1809, boot ...top112bar1fs10050459, 05.10.):
05:17:49 every P rank SLEEP-LENDs (PP1 3235905536 B on card GPU-5c648f96); D's group
grant (want=434176, TP1 need=939524096, granted only 760741888 before the loan) then
takes 939524096 B on PP1's card -- 178782208 B of them out of the loan (TP1 committed
5033164800 -> 5972688896, ledger_free 3057123328 at 05:17:59). 05:18:00.555 a D seat
ends, 05:18:00.612 the front sends `stage=resume ... freed=3057123328 from=sleep`:
the front env has no SGLANG_WEG2_DUAL_KV_TAG (#1495 off), it read no stage file,
card_room was None and the fallback compared free_min against weights+grant+air =
0+0+0 (front.log `#1480 ... legacy_p_lent=0` while the rank files said 1946157056).
PP1's wake_reclaim: W-DUAL-P-WAKE-SHORT -> W29 -> W17.

The front here has NO env tag (the f9 condition); only ``front.tag`` knows the boot.
DANGER DIRECTION: P wakes while a card's free is below that card's loan. Default OFF =
the old wake byte for byte (asserted as the f9 reading).
"""
from __future__ import annotations

import os
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.srt.weg2 import dual_p_kv_stage as PK
from sglang.srt.weg2 import front as FR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TAG = "dkr27bnvfp4dual1mpsleepsharegreentop112bar1fs10050459"
LOANS = (8017412096, 3235905536, 3531603968)           # f9 P.log:76398-76414 SLEEP-LEND
D_BUDGET = (5186977792, 5793906688, 5829558272)        # f9 D.log 05:17:49 LEDGER-PHYS budget
D_FREE = (2972385280, 760741888, 1199046656)           # ... ledger_free before the loan
D_GROW = (402653184, 939524096, 872415232)             # f9 D.log 05:17:49 GROUP-WAIT need
STEP_B = 64 << 20


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("SGLANG_WEG2_DUAL_KV_TAG", "SGLANG_WEG2_TAG", "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP",
              "SGLANG_WEG2_DUAL_D_AIR_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    tmp = tempfile.mkdtemp(prefix="wkvs1956")
    monkeypatch.setattr(PK, "stage_file",
                        lambda tag, r, root="": os.path.join(tmp, "%s-pp%d.json" % (tag, int(r))))
    return tmp


class _Rig:
    """Three cards, three P ranks asleep with the f9 loans, D grown into them."""

    def __init__(self, monkeypatch):
        self.mp = monkeypatch
        self.paths, self.d, self.actors, self.sched = [], [], [], []
        for r in range(3):
            path = os.path.join(tempfile.mkdtemp(prefix="wkvc1956"), "card")
            d = K.CardKvLedger(path, "D")
            d.contribute(D_BUDGET[r], committed=D_BUDGET[r] - D_FREE[r])
            p = K.CardKvLedger(path, "P")
            p.contribute(0)
            actor = types.SimpleNamespace(ledger=p, mapped_tokens=0, step=4096, top=196608,
                                          table=lambda: [0, STEP_B], weights_bytes=0)
            self.paths.append(path)
            self.d.append(d)
            self.actors.append(actor)
            self.sched.append(types.SimpleNamespace(tp_worker=types.SimpleNamespace(
                model_runner=types.SimpleNamespace(pp_rank=r, **{PK.ACTOR_ATTR: actor}))))
        # the ranks publish under THEIR tag (rank env = launcher dual_share_env)
        monkeypatch.setattr(PK, "_republish_stage", lambda s, a: PK.publish_stage(
            a, TAG, s.tp_worker.model_runner.pp_rank))
        for r in range(3):
            PK.publish_stage(self.actors[r], TAG, r)

    def rank(self, r, fn, *a):
        self.mp.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
        self.mp.setenv("SGLANG_WEG2_GROUP", "P")
        try:
            return fn(self.sched[r], *a)
        finally:
            self.mp.delenv("SGLANG_WEG2_DUAL_LAYOUT")
            self.mp.delenv("SGLANG_WEG2_GROUP")

    def sleep_and_d_grows(self):
        for r in range(3):
            self.mp.setattr(PK, "phys_free_bytes", lambda r=r: LOANS[r])
            assert self.rank(r, PK.sleep_lend, 0) == LOANS[r]
        for r in range(3):
            got, _ = self.d[r].request(D_GROW[r])
            assert got == D_GROW[r]
        assert self.d[1].state().free == 3057123328     # f9 front `freed=3057123328`, P exc `has 3057123328 B free`

    def front(self, sleep_leg_done=True):
        f = types.SimpleNamespace(dual_kv_ledgers=list(self.paths), tag=TAG,
                                  DUAL_LEND_GATE_LOG_S=FR.Front.DUAL_LEND_GATE_LOG_S)
        for name in ("_dual_p_stage_reading", "_dual_lend_gate_lent", "_dual_wake_room_stages"):
            setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
        st = DP.PressureStages(sleep_capable=True)
        st.wake_needs_room = bool(envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.get())   # as Front._dual_stages
        st.p_state = "sleeping"
        st._seat_mark = 0
        st.sleep_leg_done = sleep_leg_done
        return f, st

    @staticmethod
    def tick(f, st, seats_done=1):
        rd = f._dual_p_stage_reading()
        rd.pop("d_short", None)
        return st.tick(pressure=0, seats_done=seats_done, host_ok=True, **rd)


def test_default_off_is_the_f9_wake_into_the_short():
    """Default OFF reproduces f9: blind reading, wake at free 3057123328 < loan 3235905536,
    PP1's wake_reclaim raises W-DUAL-P-WAKE-SHORT (PP0/PP2 get theirs)."""
    assert envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.get() is False
    mp = pytest.MonkeyPatch()
    try:
        rig = _Rig(mp)
        rig.sleep_and_d_grows()
        f, st = rig.front()
        rd = f._dual_p_stage_reading()
        assert rd["card_room"] is None and rd["weights_bytes"] == 0 and rd["p_grant_bytes"] == 0
        action, line = rig.tick(f, st)
        assert action == "wake" and "freed=3057123328" in line and "from=sleep" in line
        assert rig.rank(0, PK.wake_reclaim) == LOANS[0]
        with pytest.raises(PK.Weg2DualPWakeShort, match="3235905536 B .* 3057123328 B free"):
            rig.rank(1, PK.wake_reclaim)
    finally:
        mp.undo()


def test_on_holds_until_the_card_has_its_loan_back(monkeypatch):
    with envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.override(True):
        rig = _Rig(monkeypatch)
        rig.sleep_and_d_grows()
        f, st = rig.front()
        rd = f._dual_p_stage_reading()
        assert rd["card_room"] is not None and len(rd["card_room"]) == 3
        assert rd["card_room"][1] == (3057123328, LOANS[1] + STEP_B)
        assert rig.tick(f, st) == (None, None)            # f9's moment: P stays asleep
        assert st.p_state == "sleeping"
        # D's seat end shrinks D's KV (f9 D.log 05:18:03 SHRINK 434176 -> 335872)
        rig.d[1].release(D_GROW[1])
        action, line = rig.tick(f, st)
        assert action == "wake" and "from=sleep" in line
        for r in range(3):                                 # every rank gets its loan back
            assert rig.rank(r, PK.wake_reclaim) == LOANS[r]


def test_on_blind_holds_with_one_named_line(monkeypatch, _env):
    with envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.override(True):
        rig = _Rig(monkeypatch)
        rig.sleep_and_d_grows()
        for r in range(3):
            rig.d[r].release(D_GROW[r])                    # room everywhere -- only the reading is missing
        os.unlink(PK.stage_file(TAG, 2))
        f, st = rig.front()
        action, line = rig.tick(f, st)
        assert action is None and "held=blind" in line and "#1956" in line
        assert rig.tick(f, st) == (None, None)             # said once
        assert st.p_state == "sleeping"


def test_on_wake_waits_for_the_sleep_leg_answer(monkeypatch):
    """The ladder enters 'sleeping' when it ISSUES the sleep; the loan is published at
    the leg's end (f9 release 35.6 s). No wake decision before the leg answered."""
    with envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.override(True):
        st = DP.PressureStages(sleep_capable=True, sleep_after=1)
        st.wake_needs_room = True
        st.p_state = "lent"
        action, _ = st.tick(pressure=1, p_committed=0, free_min=0, p_grant_bytes=0, d_air_bytes=0,
                            seats_done=0, host_ok=True)
        assert action == "sleep" and st.sleep_leg_done is False
        room = [(10 << 30, 1 << 30)] * 3
        assert st.tick(pressure=0, p_committed=0, free_min=10 << 30, p_grant_bytes=0, d_air_bytes=0,
                       seats_done=1, card_room=room) == (None, None)
        st.sleep_leg_done = True                           # Front._dual_p_sleep after the 200
        action, _ = st.tick(pressure=0, p_committed=0, free_min=10 << 30, p_grant_bytes=0, d_air_bytes=0,
                            seats_done=1, card_room=room)
        assert action == "wake"


def test_off_keeps_the_old_reading_and_ladder(monkeypatch):
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    assert st.wake_needs_room is False and st.sleep_leg_done is True
    st.p_state = "lent"
    st.tick(pressure=1, p_committed=0, free_min=0, p_grant_bytes=0, d_air_bytes=0, seats_done=0, host_ok=True)
    assert st.p_state == "sleeping" and st.sleep_leg_done is True
    action, _ = st.tick(pressure=0, p_committed=0, free_min=0, p_grant_bytes=0, d_air_bytes=0, seats_done=1)
    assert action == "wake"                                # the old trivial fallback, unchanged
    src = open(FR.__file__).read()
    assert "card_stages = self._dual_wake_room_stages(stages)" in src
    assert "if envs.SGLANG_WEG2_DUAL_WAKE_SEES_LOAN.get():" in src
