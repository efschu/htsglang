"""27B Dual-NVFP4 21.10. (boot dkr27bnvfp4dualbar1drei10212251): D-TP0 (RTX 5090) driver_free 2.7 MiB.

Chain on metal (D.log / P.log / front.log of that boot):

* 22:57:14 PP0/PP1/PP2 lend their freed allocator cache to the card pools (Q-660 DUAL-P-LEND
  939524096 / 738197504 / 836763648 B); D grows into PP0's loan.
* 22:57:17 PP0's reclaim is REFUSED (free=671088640 B), PP1/PP2 reclaim; front.log 22:57:17,222
  ``stage=resume ... from=lend`` 200 ms later: the front read the stage files under 'pdflip' (its env
  carries no FLLIPER_PDFLIP_DUAL_KV_TAG, #1495/#1480 both default off and unset) and saw p_lent 0.
* The 5090's card budget keeps the 939524096 B loan for the rest of the boot (5771362304 B from 22:57:20)
  while P regrows its cache (PP0 reserved 16442 -> 17068 MiB): phys - ledger_free 2587 MiB -> 0.
* 23:10:19 D-KV GROW 122880 -> 270336 (+1879048192 B) on that loan with P holding 2290089984 B:
  ``#1028c BOUND driver_free`` 1794.7 -> 2.7 MiB (TP1/TP2: their loans came back, no dip).

Fix: the front keys the stage files by its own --tag (front.dual_kv_tag), #1495 and #1480 default ON.
"""

import os
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_d_priority as DP
from flliper.srt.pdflip import dual_p_kv_stage as PK
from flliper.srt.pdflip import front as FR
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
TAG = "dkr27bnvfp4dualbar1drei10212251"
LENDS = (939524096, 738197504, 836763648)          # P.log 22:57:14 Q-660 DUAL-P-LEND PP0 / PP1 / PP2
RECLAIM_FREE = 671088640                            # P.log 22:57:17 PP0 refused: free=671088640
STEP_B = 64 << 20
# 23:10:19 card 0 (the 5090), D.log LEDGER-PHYS / GROUP-WAIT / GROW
BUDGET = 5715460096
P_HELD = 2290089984                                 # PP0 GRANT pdflip-0-51, 159744 tokens
D_HELD = 1543503872                                 # D mapped 122880 tokens
D_NEED = 1879048192                                 # GROW 122880 -> 270336
DRV_BEFORE_MIB = 1794.7                             # #1028c BOUND driver_free before the grow
DRV_LOW_MIB = 2.7                                   # ... after it
D_LAW_MIB = 767                                     # PDFLIP-LAUNCH CORRIDOR-FLOOR card=...e773fd938f6d group=D floor=767


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # the FRONT's env of that boot: no tag
    for k in ("FLLIPER_PDFLIP_DUAL_KV_TAG", "FLLIPER_PDFLIP_TAG", "FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP"):
        monkeypatch.delenv(k, raising=False)
    tmp = tempfile.mkdtemp(prefix="wkvs1022")
    monkeypatch.setattr(PK, "stage_file", lambda tag, r, root="": os.path.join(tmp, "%s-pp%d.json" % (tag, int(r))))
    return tmp


def _as_rank(monkeypatch, fn):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    try:
        return fn()
    finally:
        monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT")
        monkeypatch.delenv("FLLIPER_PDFLIP_GROUP")


def _metal_2257(monkeypatch):
    """Three cards, the real awake_lend / awake_reclaim and stage files: all lend, D takes part of PP0's
    loan so its reclaim is refused at free=671088640, PP1/PP2 get theirs back."""
    paths, scheds, dleds = [], [], []
    for r in range(3):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvc1022"), "card")
        d = K.CardKvLedger(path, "D")
        d.contribute(4 << 30, committed=0)
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        actor = types.SimpleNamespace(ledger=p, mapped_tokens=0, step=4096, top=266240,
                                      table=lambda: [0, STEP_B], weights_bytes=0)
        scheds.append(types.SimpleNamespace(tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(pp_rank=r, **{PK.ACTOR_ATTR: actor}))))
        paths.append(path)
        dleds.append(d)
        _as_rank(monkeypatch, lambda a=actor, r=r: PK.publish_stage(a, TAG, r))
    # the ranks publish under THEIR tag (launcher dual_share_env: FLLIPER_PDFLIP_DUAL_KV_TAG = --tag)
    monkeypatch.setattr(PK, "_republish_stage",
                        lambda s, a: PK.publish_stage(a, TAG, s.tp_worker.model_runner.pp_rank))
    phys = [0, 0, 0]
    for r in range(3):
        got = _as_rank(monkeypatch, lambda r=r: PK.awake_lend(
            scheds[r], "stage=1-lend", phys=lambda: phys[r], empty_cache=lambda: phys.__setitem__(r, LENDS[r])))
        assert got == LENDS[r]
    st0 = K.peek(paths[0])
    take = int(st0.free) - RECLAIM_FREE                  # D grows into PP0's loan
    assert dleds[0].request(take)[0] == take
    assert K.peek(paths[0]).free == RECLAIM_FREE
    assert _as_rank(monkeypatch, lambda: PK.awake_reclaim(scheds[0], "stage=1-reclaim")) == 0
    assert _as_rank(monkeypatch, lambda: PK.awake_reclaim(scheds[1], "stage=1-reclaim")) == LENDS[1]
    assert _as_rank(monkeypatch, lambda: PK.awake_reclaim(scheds[2], "stage=1-reclaim")) == LENDS[2]
    return paths, scheds, dleds


def _front(paths, tag=TAG):
    f = types.SimpleNamespace(dual_kv_ledgers=list(paths), tag=tag, DUAL_LEND_GATE_LOG_S=FR.Front.DUAL_LEND_GATE_LOG_S)
    st = DP.PressureStages(sleep_capable=False, reclaim_after=10)
    st.p_state = "reclaiming"                           # front.log 22:57:17,023 stage=1-reclaim sent to P
    f._dual_stages_obj = st
    for name in ("_dual_p_stage_reading", "_dual_lend_gate_lent"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    return f, st


def _tick(f, st):
    rd = f._dual_p_stage_reading()
    rd.pop("d_short", None)
    return st.tick(pressure=0, seats_done=0, host_ok=True, **rd), rd


@pytest.mark.parametrize("gate", [False, True])
def test_front_sees_pp0s_refused_loan_and_does_not_resume(monkeypatch, gate):
    """Metal 22:57:17,222: resume 200 ms after PP0's refused reclaim. The front now reads the stage files
    under its own --tag; with or without the #1480 gate, P stays 'reclaiming' while PP0's loan is with D."""
    paths, _, _ = _metal_2257(monkeypatch)
    with envs.FLLIPER_PDFLIP_DUAL_LEND_RESUME_GATE.override(gate):
        f, st = _front(paths)
        (action, line), rd = _tick(f, st)
        assert rd["p_lent"] == LENDS[0], "only PP0 still lends (PP1/PP2 reclaimed)"
        assert rd["card_room"] is not None and len(rd["card_room"]) == 3, "the per-card return check sees the cards"
        assert action is None and st.p_state == "reclaiming", "no resume while the 5090's loan is with D"


@pytest.mark.parametrize("gate", [False, True])
def test_front_retries_the_reclaim_and_resumes_once_the_loan_is_back(monkeypatch, gate):
    paths, scheds, dleds = _metal_2257(monkeypatch)
    with envs.FLLIPER_PDFLIP_DUAL_LEND_RESUME_GATE.override(gate):
        f, st = _front(paths)
        actions = [_tick(f, st)[0][0] for _ in range(10)]
        assert actions[:9] == [None] * 9 and actions[9] == "reclaim", "the ladder asks PP0 again (reclaim_after)"
        dleds[0].release(int(K.peek(paths[0]).committed["D"]))   # D's seat ends, the loan is free
        assert _as_rank(monkeypatch, lambda: PK.awake_reclaim(scheds[0], "retry")) == LENDS[0]
        (action, line), rd = _tick(f, st)
        assert rd["p_lent"] == 0 and action == "resume" and "from=lend" in line and st.p_state == "serving"


def test_operator_env_tag_still_wins(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_KV_TAG", "op-tag")
    assert FR.dual_kv_tag(types.SimpleNamespace(tag=TAG)) == "op-tag"
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_KV_TAG")
    assert FR.dual_kv_tag(types.SimpleNamespace(tag=TAG)) == TAG
    assert FR.dual_kv_tag(types.SimpleNamespace(tag="")) == "pdflip"


def test_switches_default_on():
    """User order 14.10. (new features on): #1480 and #1495 were built default off on 04.10. and fell out of
    the dual container arm; the boot of 21.10. ran blind."""
    from flliper.srt.pdflip.launcher import dual_front_kv_tag_env

    assert envs.FLLIPER_PDFLIP_DUAL_LEND_RESUME_GATE.get() is True
    assert envs.FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX.get() is True
    ns = types.SimpleNamespace(dual_layout=True, tag=TAG)
    assert dual_front_kv_tag_env(ns, {}) == {"FLLIPER_PDFLIP_DUAL_KV_TAG": TAG}
    assert dual_front_kv_tag_env(ns, {"FLLIPER_PDFLIP_DUAL_FRONT_KV_TAG_FIX": "0"}) == {}


def _card0(loan: bool):
    path = os.path.join(tempfile.mkdtemp(prefix="wkv2310"), "card")
    d = K.CardKvLedger(path, "D")
    base = BUDGET - LENDS[0]                         # the boot budget; the loan is added by lend()
    d.contribute(base, committed=D_HELD)
    p = K.CardKvLedger(path, "P")
    p.contribute(0)
    if loan:
        p.lend(LENDS[0])
    assert K.peek(path).budget == (BUDGET if loan else BUDGET - LENDS[0])
    assert p.request(P_HELD)[0] == P_HELD
    return d


def test_metal_2310_grow_lands_on_the_loan_and_the_low_without_it():
    """23:10:19 on card 0. With the loan in the budget the ledger grants D's whole grow and the card falls
    by the grant (1794.7 -> 2.7 MiB). With the loan back (P reclaimed it before it prefilled again) the same
    grow is short: D presses P instead of mapping, and the card keeps the loan's bytes free."""
    d = _card0(loan=True)
    granted, pressure = d.request(D_NEED)
    assert granted == D_NEED and pressure == 0
    assert abs((DRV_BEFORE_MIB - DRV_LOW_MIB) - D_NEED / MIB) < 1.0, "the dip IS the grow (1792 MiB)"

    d = _card0(loan=False)
    granted, pressure = d.request(D_NEED)
    assert granted < D_NEED, "short: DKvStage.group_grow gives the grant back (MIN verdict 0)"
    assert pressure == D_NEED - granted, "D presses P (stage 1 pause), it does not map past the card"
    low_mib = DRV_LOW_MIB + LENDS[0] / MIB
    assert low_mib >= D_LAW_MIB, "expected TP0 low ~%.0f MiB >= the measured D law %d MiB" % (low_mib, D_LAW_MIB)
