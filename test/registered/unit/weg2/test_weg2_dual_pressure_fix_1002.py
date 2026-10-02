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


# -- D's actor on a gmps12-sized card -------------------------------------------------

TOP = 262144
TOK_B = 6144               # ~ D's bytes per token on TP0 (gmps12: -603979776 B for 98304 tokens)
G = 2 << 20


class _Spans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=(TOP + 64) * TOK_B, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


class _Alloc:
    """available = mapped - used (the cap follows the mapping)."""

    size = 0

    def __init__(self, used):
        self.used = int(used)
        self.actor = None

    def available_size(self):
        return max(0, int(self.actor.mapped_tokens) - self.used)


def _d_actor(path, mapped, used, gmin=None):
    t = types.SimpleNamespace(shape=(TOP + 64,), numel=lambda: (TOP + 64) * (TOK_B // 2), element_size=lambda: 2)
    geom = PK._geom_for(t, TOP, 64, "k", (TOP + 64) * TOK_B)
    led = K.CardKvLedger(path, "D")
    alloc = _Alloc(used)
    a = DK.DKvStage([(1, geom)], led, allocator=alloc, pools=[], page_size=64, granule=G, top_tokens=TOP,
                    spans=_Spans(), step=STEP, engage_cap=lambda *x: None, gmin=gmin or (lambda v: list(v)))
    alloc.actor = a
    return a, led


def _card(tmp, mapped, used, p_committed, free_extra, gmin=None):
    """gmps12 TP0 card after the shrink to ``mapped``: D committed its level, P
    ``p_committed`` (00:16:44: 830472192), ``free_extra`` bytes free in the ledger."""
    path = os.path.join(tmp, "card")
    a, d_led = _d_actor(path, mapped, used, gmin)
    b = a.bytes_for(mapped) - a.bytes_for(0)
    d_led.contribute(b + p_committed + free_extra, committed=b)
    a.mapped_tokens, a._committed = mapped, b
    p = K.CardKvLedger(path, "P")
    p.contribute(0)
    if p_committed:
        got, _ = p.request(p_committed)
        assert got == p_committed
    return a, path, p


class _Batch:
    def __init__(self, reqs, need):
        self.reqs, self._need, self.prepared = list(reqs), int(need), 0

    def filter_batch(self):
        pass

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def new_tokens_required_next_decode(self):
        return self._need

    def prepare_for_decode(self):
        self.prepared += 1

    def retract_decode(self, _args):
        raise AssertionError("the decode was interrupted")


def _sched(actor, pre_avail):
    s = types.SimpleNamespace(
        admission_limiter=types.SimpleNamespace(auto=False),
        kv_session_offload=None, tree_cache=types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace()),
        forward_ct=1, _cross_schedule_mode=False,
        new_token_ratio_tracker=types.SimpleNamespace(current=0.3, decay_step=lambda: None),
        token_to_kv_pool_allocator=actor.allocator, server_args=None,
        tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(**{DK.ACTOR_ATTR: actor})),
        _weg2_group_min_ints=lambda v: list(v), _weg2_d_park_draft_snapshot=lambda b: None)
    s.uniform_min_avail = lambda: int(pre_avail)
    s._retract_decode_and_requeue = types.MethodType(SC.Scheduler._retract_decode_and_requeue, s)
    return s


def _round(monkeypatch, sched, batch, fn=None):
    """One update_running_batch on a dual D rank (the eviction is a no-op here)."""
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(SC, "evict_from_tree_cache", lambda *a, **k: None)
    meth = types.MethodType(fn or inspect.unwrap(SC.Scheduler.update_running_batch), sched)
    out = meth(batch)
    return out, DP.take_hold_for_grow(out)


def _gmps12_full_round(tmp, p_committed, free_extra):
    # 69632 mapped after the 00:16:50 shrink, weg2-0-53's 35736 rows + the parked span: 6 rows left,
    # a DFlash round of one seat needs 8 -> kv_full (uniform_min_avail 6 < 8)
    a, path, p = _card(tmp, 69632, 69626, p_committed, free_extra)
    batch = _Batch([types.SimpleNamespace(rid="weg2-0-53")], need=8)
    return a, path, p, batch, _sched(a, pre_avail=6)


# -- (B1) emergency growth / hold instead of the retract ------------------------------

def test_b1_kv_full_with_ledger_room_grows_and_decodes(monkeypatch):
    # gmps12 exactly: ledger_free=2410348544 on TP0 when the retract came
    a, path, p, batch, sched = _gmps12_full_round(tempfile.mkdtemp(prefix="wb1"), 830472192, 2410348544)
    out, held = _round(monkeypatch, sched, batch)
    assert not held and out.prepared == 1, "the decode runs this iteration"
    assert a.mapped_tokens > 69632 and a.allocator.available_size() >= 8


def test_b1_short_card_holds_presses_p_then_decodes_once_p_released(monkeypatch, caplog):
    a, path, p, batch, sched = _gmps12_full_round(tempfile.mkdtemp(prefix="wb1s"), 830472192, 0)
    caplog.set_level("WARNING")
    for _ in range(3):
        out, held = _round(monkeypatch, sched, batch)
        assert held and out.prepared == 0, "held: no decode, nothing retracted"
    st = K.peek(path)
    assert st.pressure["P"] > 0 and st.demand["D"] > 0, "D's request pressed P (stage 1)"
    assert sched._weg2_d_hold["n"] == 3
    assert any("D-HOLD-FOR-GROW n=1 ms=" in r.getMessage() for r in caplog.records)
    p.release(830472192)                                   # stage 1 done: P gave its KV back
    out, held = _round(monkeypatch, sched, batch)
    assert not held and out.prepared == 1 and a.mapped_tokens > 69632
    assert any("D-HOLD-FOR-GROW n=3" in r.getMessage() and "released=grown" in r.getMessage()
               for r in caplog.records)
    assert getattr(sched, "_weg2_d_hold", None) is None


def test_b1_only_a_hold_past_the_max_is_the_named_stop(monkeypatch):
    a, path, p, batch, sched = _gmps12_full_round(tempfile.mkdtemp(prefix="wb1x"), 830472192, 0)
    monkeypatch.setenv(DP.HOLD_MAX_ENV, "5")
    clock = [1000.0]
    real = DP.grow_or_hold
    monkeypatch.setattr(DP, "grow_or_hold", lambda *a, **k: real(*a, now=lambda: clock[0], **k))
    _round(monkeypatch, sched, batch)                      # t=0: hold
    clock[0] += 4.9
    assert _round(monkeypatch, sched, batch)[1]            # 4.9 s: still a hold
    clock[0] += 0.2
    with pytest.raises(DP.Weg2DualDRetract, match="W-DUAL-D-RETRACT"):
        _round(monkeypatch, sched, batch)                  # 5.1 s: the named stop, never a retract


def test_b1_the_default_hold_max_is_120_s():
    assert DP.hold_max_s({}) == 120.0 and DP.hold_max_s({DP.HOLD_MAX_ENV: "30"}) == 30.0
    assert DP.hold_max_s({DP.HOLD_MAX_ENV: "x"}) == 120.0


def test_b1_off_the_dual_d_nothing_holds():
    assert DP.grow_or_hold(types.SimpleNamespace(), None, 8, env={}) == "retract"
    assert DP.grow_or_hold(types.SimpleNamespace(), None, 8,
                           env={"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}) == "retract"


def test_b1_both_callers_skip_the_decode_of_a_held_batch():
    src = inspect.getsource(SC.Scheduler.get_next_batch_to_run)
    assert src.count("running_batch = self.update_running_batch(running_batch)") == 2
    assert src.count("if _weg2_dual_d_priority.take_hold_for_grow(running_batch):") == 2


def test_b1_without_the_hold_mutant_is_the_gmps12_retract(monkeypatch):
    fn = inspect.unwrap(SC.Scheduler.update_running_batch)
    src = textwrap.dedent(inspect.getsource(fn))
    fixed = "if kv_full_retract_flag and _weg2_dual_d_priority.d_retract_forbidden():"
    assert src.count(fixed) == 1, "the hook moved -- re-aim the mutant"
    ns = dict(vars(SC))
    ns["evict_from_tree_cache"] = lambda *a, **k: None
    exec(compile(src.replace(fixed, "if False:"), SC.__file__, "exec"), ns)
    a, path, p, batch, sched = _gmps12_full_round(tempfile.mkdtemp(prefix="wb1m"), 830472192, 2410348544)
    with pytest.raises(DP.Weg2DualDRetract, match="W-DUAL-D-RETRACT"):
        _round(monkeypatch, sched, batch, fn=ns["update_running_batch"])


# -- (B1, front) the stages read D's shortfall: stage 2 is reachable -------------------

def _stage_front(paths):
    f = types.SimpleNamespace(dual_kv_ledgers=list(paths), counters=collections.Counter(), queue=[],
                              _dual_inflight=[])
    calls = []

    async def _sleep():
        calls.append("sleep")

    async def _wake():
        calls.append("wake")

    f._dual_p_sleep, f._dual_p_wake = _sleep, _wake
    f._dual_p_sleep_probe_tick = lambda pressure: False
    for name in ("_dual_stages", "_dual_p_stage_reading", "_dual_stage_tick"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    return f, calls


def test_b1_with_p_fully_released_d_still_short_p_sleeps(monkeypatch, stage_dir):
    # stage 1 complete (P commits 0 everywhere), D's grow still short on PP1's card: the
    # ledger pressure on P reads 0 (arbitrate caps it at P's committed bytes) -- the ladder
    # must still go to stage 2
    monkeypatch.setenv(DP.P_SLEEP_ENV, "1")
    monkeypatch.setenv(DP.SLEEP_AFTER_TICKS_ENV, "2")
    monkeypatch.setattr(FR, "_mem_available_bytes", lambda: 200 * GiB)
    cards = _ledger_set(stage_dir, d_committed=(0, BUDGETS[1], 0))
    for r, (path, d, p) in enumerate(cards):
        PK.publish_stage(_stage_actor(p), "t1002", r)
    got, pressure = cards[1][1].request(512 << 20)
    assert got == 0 and pressure == 0 and K.peek(cards[1][0]).demand["D"] == 512 << 20
    f, calls = _stage_front([c[0] for c in cards])

    async def ticks():
        for _ in range(4):
            f._dual_stage_tick(0)          # the pump's ledger pressure on P: 0
            await asyncio.sleep(0)

    asyncio.run(ticks())
    assert f._dual_stages_obj.p_state == "sleeping" and calls == ["sleep"], (f._dual_stages_obj.p_state, calls)
    assert f.counters["dual_kv_pressure_stage1"] == 1 and f.counters["dual_kv_pressure_stage2"] == 1


def test_b1_the_front_without_d_short_mutant_never_sleeps(monkeypatch, stage_dir):
    src = textwrap.dedent(inspect.getsource(FR.Front._dual_stage_tick))
    fixed = "pressure = max(int(pressure), d_short)"
    assert src.count(fixed) == 1
    ns = dict(vars(FR))
    exec(compile(src.replace(fixed, "pressure = int(pressure)"), FR.__file__, "exec"), ns)
    monkeypatch.setattr(FR.Front, "_dual_stage_tick", ns["_dual_stage_tick"])
    with pytest.raises(AssertionError):
        test_b1_with_p_fully_released_d_still_short_p_sleeps(monkeypatch, stage_dir)


# -- (B2) D's level follows the tightest rank -----------------------------------------

class _Tree:
    def __init__(self, ev):
        self.ev = int(ev)

    def evictable_size(self):
        return self.ev


def _tick_sched(actor, running, ev, gmin, parked=()):
    req = [types.SimpleNamespace(rid=r, origin_input_ids=[0] * n, output_ids=[]) for r, n in running]
    return types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=req), chunked_req=None, waiting_queue=[],
        server_args=types.SimpleNamespace(chunked_prefill_size=1600, speculative_num_draft_tokens=4,
                                          max_running_requests=6),
        tree_cache=_Tree(ev), weg2_d_parked=list(parked),
        tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(**{DK.ACTOR_ATTR: actor})),
        _weg2_group_min_ints=gmin)


def _other_rank(avail):
    """The tightest rank reports ``avail`` free rows (slot 5 of the tick's vector)."""
    def g(vals):
        vals = list(vals)
        if len(vals) > 5:
            vals[5] = min(vals[5], int(avail))
        return vals
    return g


def test_b2_the_tightest_rank_grows_d_before_the_round_runs_full(monkeypatch):
    # after the shrink to 69632 and the 35736 hand-back: THIS rank's bookkeeping fits (its tree
    # holds 30000 evictable rows), the tightest rank has 2000 rows left -- under one chunk +
    # a decode round of 6 seats + 2 lattice steps
    a, path, p = _card(tempfile.mkdtemp(prefix="wb2"), 69632, 35736, 0, 4 * GiB)
    monkeypatch.setattr(DK._pk, "max_live_id", lambda *x: 0)
    sched = _tick_sched(a, [("weg2-0-53", 35736)], ev=30000, gmin=_other_rank(2000))
    assert DK.tick(sched) == "grow"
    assert a.mapped_tokens >= 69632 + STEP


def test_b2_without_the_tightest_rank_mutant_does_not_grow(monkeypatch):
    monkeypatch.setattr(DK, "floor_want", lambda want, *a, **k: int(want))
    with pytest.raises(AssertionError):
        test_b2_the_tightest_rank_grows_d_before_the_round_runs_full(monkeypatch)


def test_b2_an_idle_d_never_grows_for_its_cache(monkeypatch):
    assert DK.floor_want(8192, 69632, 10, 1624, STEP, demand=0) == 8192
    assert DK.floor_want(8192, 69632, DK.NO_AVAIL, 1624, STEP, demand=5) == 8192
    assert DK.floor_want(8192, 69632, 1624 + 2 * STEP, 1624, STEP, demand=5) == 8192
    assert DK.floor_want(8192, 69632, 2000, 1624, STEP, demand=5) == 69632 + 2 * STEP


# -- (B3) no shrink under holds, keep the look-ahead room ------------------------------

def test_b3_a_shrink_keeps_the_look_ahead_room():
    # 00:16:45: 167936 mapped, the two seats' level 69632, P waits; the group has 106000 rows free
    air = 1624
    v, lvl = DK.decide(167936, 69632, True, 0, STEP, avail_min=106000, air=air)
    assert v == "shrink" and 106000 - (167936 - lvl) >= air + 2 * STEP
    assert lvl > 69632, "not down to the bookkeeping: the room stays"
    # nearly no free rows: no shrink at all
    assert DK.decide(167936, 69632, True, 0, STEP, avail_min=air + 2 * STEP + 100, air=air) == ("hold", 167936)
    # no reading: the old rule
    assert DK.decide(167936, 69632, True, 0, STEP) == ("shrink", 69632)


def test_b3_a_parked_request_stops_the_p_waiting_shrink(monkeypatch):
    # 00:16:50: weg2-0-52 parked by W50-midstream (00:16:47), P waits -> SHRINK 106496 -> 69632
    a, path, p = _card(tempfile.mkdtemp(prefix="wb3"), 106496, 40000, 0, 4 * GiB)
    got, _ = p.request(8 << 30)                          # P's grant is short -> demand[P] > 0
    assert K.peek(path).demand["P"] > 0
    monkeypatch.setattr(DK._pk, "max_live_id", lambda *x: 0)
    sched = _tick_sched(a, [("weg2-0-48", 34000)], ev=0, gmin=lambda v: list(v),
                        parked=[types.SimpleNamespace(rid="weg2-0-52")])
    for _ in range(8):
        DK.tick(sched)
    assert a.mapped_tokens == 106496, "D shrank under the parked request's context"
    sched.weg2_d_parked = []                             # the hold is gone: P's wait shrinks again
    for _ in range(2):
        DK.tick(sched)
    assert a.mapped_tokens < 106496


def test_b3_the_holds_ignored_mutant_turns_the_park_test_red(monkeypatch):
    monkeypatch.setattr(DK, "d_holds", lambda sched: 0)
    with pytest.raises(AssertionError):
        test_b3_a_parked_request_stops_the_p_waiting_shrink(monkeypatch)


def test_b3_the_room_ignored_mutant_turns_the_room_test_red(monkeypatch):
    src = textwrap.dedent(inspect.getsource(DK.decide))
    fixed = "if avail_min is not None:"
    assert src.count(fixed) == 1
    ns = dict(vars(DK))
    exec(compile(src.replace(fixed, "if False:"), DK.__file__, "exec"), ns)
    monkeypatch.setattr(DK, "decide", ns["decide"])
    with pytest.raises(AssertionError):
        test_b3_a_shrink_keeps_the_look_ahead_room()
