# SPDX-License-Identifier: Apache-2.0
"""DUAL PRESSURE FIX (27B dual, 02.10.): the D-priority ladder actually runs.

User rule (dual layout): D NEVER retracts a running decode. Under KV pressure
(1) P stops prefilling and releases its KV, (2) if that is not enough P sleeps
and parks its weights in host RAM, (3) when a seat finishes P comes back.

Metal gmps12 (boot_weg2_dkr27bnvfp4dual1mpsleepbar1fs10020008_9d50005b75, D TP0):
00:16:45 SHRINK 167936 -> 69632 (a P prompt waited), 00:16:50 SHRINK 106496 -> 69632
(weg2-0-52 parked by W50-midstream since 00:16:47), 00:16:51 hand-back weg2-0-53
N=35736, 00:16:52 "full token usage 0.97", 00:16:53 W-DUAL-D-RETRACT (kv_full) --
while the card ledger had ledger_free=2410348544. The stage-2 sleep itself was
proven only by the probe (SLEEP-LEND 7958691840 / 3219128320 / 3508535296 B).

DANGER DIRECTIONS, one test + one mutant each (asserted in-suite):
* (A) the wake from sleep is judged PER CARD (loan + grant step + look-ahead free
  on every card); the old total "weights sum vs the tightest card" never holds on a
  3080 -- P would sleep forever after real pressure. The loan is republished into
  the stage file at the sleep and at the wake.
* (B1) a full decode round on dual D grows (group-uniform) or HOLDS the batch for
  the iteration (D-HOLD-FOR-GROW), never retracts; only a hold past
  SGLANG_WEG2_DUAL_D_HOLD_MAX_S ends in W-DUAL-D-RETRACT. The front's stages read
  D's SHORTFALL (ledger demand[D]): the ledger pressure on P is capped at P's
  committed bytes and reads 0 once stage 1 is done, which made stage 2 unreachable.
* (B2) D's level follows the TIGHTEST rank's free rows (MIN over the ranks).
* (B3) no shrink below the look-ahead room while D runs work; no P-waiting shrink
  while a hold (parked / W50 midstream / hold episode) exists.
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

from sglang.srt.managers import scheduler as SC
from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_kv_stage as DK
from sglang.srt.weg2 import dual_d_priority as DP
from sglang.srt.weg2 import dual_p_kv_stage as PK
from sglang.srt.weg2 import front as FR
from sglang.srt.weg2.d_seat_vram import AllocInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

GiB = 1 << 30
DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}

# gmps12, P.log 00:11:54: the three stages' loans (PP0 / PP1 / PP2)
LOANS = (7958691840, 3219128320, 3508535296)
# the card budgets the handover names (4.25 / 6.41 / 6.12 GB), paired in order
BUDGETS = (4247453696, 6410000000, 6120000000)
STEP = 4096
STEP_B = 64 << 20          # one P grant step


def _ledger_set(tmp, d_committed=(0, 0, 0)):
    """Three cards, D boot-contributed its budget (committing ``d_committed``), P joined."""
    out = []
    for i, (budget, dc) in enumerate(zip(BUDGETS, d_committed)):
        path = os.path.join(tmp, "card%d" % i)
        d = K.CardKvLedger(path, "D")
        d.contribute(budget, committed=dc)
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        out.append((path, d, p))
    return out


def _stage_actor(p_ledger, lent=0, weights=0):
    return types.SimpleNamespace(ledger=p_ledger, step=STEP, top=196608, table=lambda: [0, STEP_B],
                                 weights_bytes=weights, _sleep_lent=lent)


@pytest.fixture()
def stage_dir(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="wkvs1002")
    monkeypatch.setattr(PK, "stage_file", lambda tag, r, root="": os.path.join(tmp, "st-pp%d.json" % int(r)))
    monkeypatch.setenv("SGLANG_WEG2_DUAL_KV_TAG", "t1002")
    monkeypatch.setenv("SGLANG_WEG2_DUAL_D_AIR_TOKENS", "1624")
    return tmp


def _reading_front(paths):
    f = types.SimpleNamespace(dual_kv_ledgers=list(paths))
    f._dual_p_stage_reading = types.MethodType(FR.Front._dual_p_stage_reading, f)
    return f


def _asleep_after_gmps12(stage_dir, d_into_loan=(0, 0, 0)):
    """P slept (loans published per stage), D then committed ``d_into_loan`` more."""
    cards = _ledger_set(stage_dir)
    for r, ((path, d, p), loan) in enumerate(zip(cards, LOANS)):
        p.lend(loan)
        PK.publish_stage(_stage_actor(p, lent=loan, weights=loan), "t1002", r)
        if d_into_loan[r]:
            got, _ = d.request(d_into_loan[r])
            assert got == d_into_loan[r]
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    st.p_state, st._seat_mark = "sleeping", 3
    return cards, st


# -- (A) the wake from sleep, per card ------------------------------------------------

def test_a_asleep_p_wakes_when_every_card_has_its_loan_free(stage_dir):
    cards, st = _asleep_after_gmps12(stage_dir)
    rd = _reading_front([c[0] for c in cards])._dual_p_stage_reading()
    rd.pop("d_short", None)
    # the old total: P's weights against the tightest card -- never true on these cards
    assert rd["free_min"] < rd["weights_bytes"] == sum(LOANS)
    action, line = st.tick(pressure=0, seats_done=4, host_ok=True, **rd)
    assert action == "wake" and "from=sleep" in line
    assert rd["card_room"][0][1] > LOANS[0], "the need carries the loan + grant + air"


def test_a_one_card_with_d_inside_its_loan_keeps_p_asleep(stage_dir):
    # D grew into PP1's loan on its 3080 (6.41 GB budget + 3.22 GB loan): that card cannot
    # give the loan back, the wake would die in W-DUAL-P-WAKE-SHORT
    cards, st = _asleep_after_gmps12(stage_dir, d_into_loan=(0, BUDGETS[1] + 1 * GiB, 0))
    rd = _reading_front([c[0] for c in cards])._dual_p_stage_reading()
    rd.pop("d_short", None)
    assert st.tick(pressure=0, seats_done=4, host_ok=True, **rd) == (None, None)
    assert st.p_state == "sleeping"


def test_a_the_loan_is_republished_at_sleep_and_wake(monkeypatch, stage_dir):
    path, d, p = _ledger_set(stage_dir)[1]
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    actor = _stage_actor(p)
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
        pp_rank=1, **{PK.ACTOR_ATTR: actor})))
    phys = [1 * GiB]
    monkeypatch.setattr(PK, "phys_free_bytes", lambda: phys[0])
    before = PK.sleep_phys_before(sched)
    phys[0] += LOANS[1]
    assert PK.sleep_lend(sched, before) == LOANS[1]
    with open(PK.stage_file("t1002", 1)) as f:
        assert json.load(f)["lent"] == LOANS[1], "the front reads the loan per card"
    assert PK.wake_reclaim(sched) == LOANS[1]
    with open(PK.stage_file("t1002", 1)) as f:
        assert json.load(f)["lent"] == 0


def _stage_mutant(fixed, back):
    src = textwrap.dedent(inspect.getsource(DP.PressureStages))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant: " + fixed
    ns = dict(vars(DP))
    exec(compile(src.replace(fixed, back), DP.__file__, "exec"), ns)
    return ns["PressureStages"]


@pytest.mark.parametrize("fixed,back,test", [
    ("if card_room is not None:", "if False:", test_a_asleep_p_wakes_when_every_card_has_its_loan_free),
    ("all(int(f) >= int(n) for f, n in card_room)", "any(int(f) >= int(n) for f, n in card_room)",
     test_a_one_card_with_d_inside_its_loan_keeps_p_asleep),
], ids=["no-card-room", "any-card"])
def test_a_each_wake_guard_has_a_red_mutant(monkeypatch, stage_dir, fixed, back, test):
    monkeypatch.setattr(DP, "PressureStages", _stage_mutant(fixed, back))
    with pytest.raises(AssertionError):
        test(stage_dir)


def test_a_the_front_without_card_room_mutant_is_red(monkeypatch, stage_dir):
    src = textwrap.dedent(inspect.getsource(FR.Front._dual_p_stage_reading))
    fixed = '"card_room": card_room if stages else None'
    assert src.count(fixed) == 1
    ns = dict(vars(FR))
    exec(compile(src.replace(fixed, '"card_room": None'), FR.__file__, "exec"), ns)
    monkeypatch.setattr(FR.Front, "_dual_p_stage_reading", ns["_dual_p_stage_reading"])
    with pytest.raises(AssertionError):
        test_a_asleep_p_wakes_when_every_card_has_its_loan_free(stage_dir)


def test_a_no_republish_mutant_is_red(monkeypatch, stage_dir):
    src = textwrap.dedent(inspect.getsource(PK.sleep_lend))
    fixed = "    _republish_stage(sched, actor)\n"
    assert src.count(fixed) == 1
    ns = dict(vars(PK))
    exec(compile(src.replace(fixed, ""), PK.__file__, "exec"), ns)
    monkeypatch.setattr(PK, "sleep_lend", ns["sleep_lend"])
    with pytest.raises((AssertionError, OSError)):
        test_a_the_loan_is_republished_at_sleep_and_wake(monkeypatch, stage_dir)


