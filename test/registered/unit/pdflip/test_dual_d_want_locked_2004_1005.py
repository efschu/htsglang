# SPDX-License-Identifier: Apache-2.0
"""#2004 D-WANT-LOCKED + d_want instrument (dual D, FLLIPER_PDFLIP_DUAL_D_WANT_LOCKED, default off).

deskq/done/2000 (Karte-0-Wand 262k), pt4 fs10051332 D TP0 PKVWAIT-INSTR: mapped 308634, mapped - avail_min
149246 occupied rows, need == mapped -- ``d_demand`` adds the FULL token count of every request, so the six
seats that share one 36864-token prefix were counted six times and D's level (and with it D's rows on
card 0) stood ~160k tokens above what it holds.

DANGER DIRECTION guarded here: D UNDER-SUPPLY (retract, kv_full, OOM). So:
* the switch off is the old code (same want, same trajectory);
* on, the level is never below locked + incoming + air, never above the old rule's;
* a queued request is counted whole, the chunked one by its rest;
* no reading of the locked rows = the old want;
* grow stays immediate and the floor_want / live floor stay behind the new want (D never waits);
* flip form / P: nothing is armed.
"""
from __future__ import annotations

import logging
import os
import tempfile
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import card_kv_ledger as K  # noqa: E402
from flliper.srt.pdflip import dual_d_kv_stage as D  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as P  # noqa: E402
from flliper.srt.pdflip import dual_pkvwait_instr as PI  # noqa: E402
from flliper.srt.pdflip.d_seat_vram import AllocInfo  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

ON = "FLLIPER_PDFLIP_DUAL_D_WANT_LOCKED"
DUAL_D = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D", D.MAX_TOKENS_ENV: "131072"}
KEYS = ("FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP", D.MAX_TOKENS_ENV, ON)
STEP = 4096
G = 2 << 20
ALLOC = (131072 + 64) * 2048
PREFIX, TAIL, N = 36864, 2560, 6        # six seats on one shared prefix (the pt4 holders)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    PI._reset_for_tests()


def _req(rid, n, out=0, prefix=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(n)), output_ids=[0] * out,
                                 prefix_indices=list(range(prefix)))


# -- pure ---------------------------------------------------------------------------------------------

def test_six_holders_on_one_prefix_want_follows_the_locked_rows():
    demand = N * (PREFIX + TAIL)                      # 6 x 39424 = 236544, what d_demand sums
    locked = PREFIX + N * TAIL                        # the prefix once + six tails = 52224 rows held
    air = 1600 + 6 * 4
    old = D.want_local_tokens(demand, locked, air, STEP)
    new = D.want_locked_local_tokens(demand, locked, 0, air, STEP)
    assert old == D.want_tokens(demand, 0, air, STEP)               # the overcount is what the old rule asks
    assert new == D.want_tokens(locked, 0, air, STEP)               # locked + air (the growth reserve), on the lattice
    assert new < old and old - new >= 180000
    assert new >= locked + air, "never below what D holds plus the decode/verify air"


def test_never_above_the_old_rule_and_never_below_locked_incoming_air():
    air = 1624
    for demand, locked, incoming in [(236544, 52224, 0), (156232, 217857, 0), (40000, 61625, 40000),
                                     (0, 0, 0), (50000, 50000, 5000), (300000, 20000, 90000)]:
        old = D.want_local_tokens(demand, locked, air, STEP)
        new = D.want_locked_local_tokens(demand, locked, incoming, air, STEP)
        assert new <= old, (demand, locked, incoming)
        assert new >= min(old, D.want_tokens(locked, incoming, air, STEP)), (demand, locked, incoming)


def test_a_hold_the_bookkeeping_misses_still_holds_the_level():
    # gmps7 17:53:05 (test_pdflip_dual_d_priority_1001): locked above the bookkeeping -> the old level, unchanged
    demand, held, air = 156232, 61625, 1624
    locked = demand + held
    assert D.want_locked_local_tokens(demand, locked, 0, air, STEP) == D.want_local_tokens(demand, locked, air, STEP)


def test_queue_counted_whole_chunked_by_its_rest():
    sched = types.SimpleNamespace(
        waiting_queue=[_req("w1", 30000), _req("w2", 4000, prefix=3000)],
        chunked_req=_req("c", 40000, prefix=24000))
    assert D.d_incoming_tokens(sched) == 30000 + 4000 + (40000 - 24000)
    assert D.d_incoming_tokens(types.SimpleNamespace(waiting_queue=[], chunked_req=None)) == 0


def test_queue_request_lifts_the_level_the_ignore_incoming_mutant_turns_red(monkeypatch):
    def check():
        w = D.want_locked_local_tokens(236544 + 30000, 52224, 30000, 1624, STEP)
        assert w >= 52224 + 30000, "a waiting request's rows are asked for before it is admitted"

    check()
    real = D.want_locked_local_tokens
    monkeypatch.setattr(D, "want_locked_local_tokens", lambda d, lk, inc, air, st: real(d, lk, 0, air, st))
    with pytest.raises(AssertionError):
        check()


def test_the_cap_removed_mutant_turns_the_never_above_test_red(monkeypatch):
    def check():
        # a hold the bookkeeping misses (61625) + a queued 40000: the old rule asks locked + air only
        old = D.want_local_tokens(40000, 61625, 1624, STEP)
        assert D.want_locked_local_tokens(40000, 61625, 40000, 1624, STEP) <= old

    check()
    monkeypatch.setattr(D, "want_locked_local_tokens",
                        lambda d, lk, inc, air, st: D.want_tokens(lk, inc, air, st))
    with pytest.raises(AssertionError):
        check()


def test_gate_and_switch():
    for env in [{}, {ON: "1"}, {"FLLIPER_PDFLIP_GROUP": "D", ON: "1"},
                {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P", D.MAX_TOKENS_ENV: "131072", ON: "1"},
                {**DUAL_D, ON: "0"}, dict(DUAL_D)]:
        assert D.want_locked_armed(env) is False, env
    assert D.want_locked_armed({**DUAL_D, ON: "1"}) is True


def test_no_reading_means_the_old_want(monkeypatch):
    monkeypatch.setenv(ON, "1")
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    actor = types.SimpleNamespace(mapped_tokens=126976, step=STEP, allocator=object())   # no available_size
    sched = types.SimpleNamespace(waiting_queue=[], chunked_req=None, tree_cache=None)
    want, info = D._want_locked_step(sched, actor, 236544, 241664, 1624)
    assert want == 241664 and info["mode"] == "noreading"


# -- the tick -----------------------------------------------------------------------------------------

class _Alloc:
    def __init__(self, free):
        self.free = free

    def available_size(self):
        return self.free


class _Spans:
    available = True

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        return 0


def _tick_world(mapped, locked, running, waiting=(), chunked=None, evictable=0):
    path = os.path.join(tempfile.mkdtemp(prefix="wdl"), "card")
    led = K.CardKvLedger(path, "D")
    geom = P._geom_for(torch.zeros(131072 + 64, 512), 131072, 64, "k", ALLOC)
    a = D.DKvStage([(1, geom)], led, allocator=_Alloc(mapped - locked - evictable), pools=[], page_size=64,
                   granule=G, top_tokens=131072, spans=_Spans(), step=STEP, engage_cap=lambda *x: None,
                   gmin=lambda v: v)
    b = a.bytes_for(mapped) - a.bytes_for(0)
    led.contribute(b, committed=b)
    led_p = K.CardKvLedger(path, "P")                # P has joined the card (else D keeps its boot pool) ...
    led_p.contribute(a.bytes_for(131072) - a.bytes_for(0))   # ... and brought free bytes: a grow is granted
    a.mapped_tokens, a._committed = mapped, b
    sched = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=list(running)), chunked_req=chunked,
        waiting_queue=list(waiting), server_args=types.SimpleNamespace(chunked_prefill_size=4096,
                                                                       speculative_num_draft_tokens=8),
        tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=a)),
        tree_cache=types.SimpleNamespace(evictable_size=lambda: evictable),
        _pdflip_group_min_ints=lambda v: v)
    return a, sched


def _holders():
    return [_req("pdflip-0-%d" % i, PREFIX + TAIL) for i in range(N)]


def _run(a, sched, n, floor=0):
    verdicts = []
    with mock.patch.object(D._pk, "max_live_id", lambda *x: floor // 64):
        for _ in range(n):
            verdicts.append(D.tick(sched))
    return verdicts


def test_switch_off_is_the_old_trajectory(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    mapped, locked = 241664, PREFIX + N * TAIL
    a, sched = _tick_world(mapped, locked, _holders())
    ref_a, ref_sched = _tick_world(mapped, locked, _holders())
    # reference: the instrument and the step replaced by the identity -- the code before #2004
    ref_sched.tp_worker.model_runner.dual_d_kv = ref_a
    with mock.patch.object(D, "_want_locked_step", lambda s, ac, dem, w, air: (w, {})), \
            mock.patch.object(D, "_instr_d_want", lambda *x, **k: None):
        ref = _run(ref_a, ref_sched, 100, floor=locked)
    got = _run(a, sched, 100, floor=locked)
    assert got == ref and a.mapped_tokens == ref_a.mapped_tokens == mapped
    assert set(got) == {"hold"}


def test_switch_on_shrinks_the_overcount_but_holds_a_floor(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(ON, "1")
    mapped, locked = 241664, PREFIX + N * TAIL
    floor = 60000                                    # the highest live row of the group
    a, sched = _tick_world(mapped, locked, _holders())
    _run(a, sched, 100, floor=floor)
    assert a.mapped_tokens < mapped, "the overcounted level was given back"
    assert a.mapped_tokens >= floor, "never below a live row"
    assert a.mapped_tokens >= locked + 1624, "never below what D holds"


def test_switch_on_a_waiting_request_grows_at_once(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(ON, "1")
    locked = PREFIX + N * TAIL
    mapped = 61440
    a, sched = _tick_world(mapped, locked, _holders(), waiting=[_req("w", 40000)])
    verdicts = _run(a, sched, 1, floor=locked)
    assert verdicts == ["grow"], "D asks at once for the queued request's rows (never waits for the hold rounds)"
    assert a.mapped_tokens >= locked + 40000


def test_switch_on_tight_pool_floor_want_still_grows(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(ON, "1")
    mapped = 53248
    locked = mapped - 800                            # 800 free rows: under air + two steps
    a, sched = _tick_world(mapped, locked, _holders())
    verdicts = _run(a, sched, 1, floor=locked)
    assert verdicts == ["grow"] and a.mapped_tokens >= mapped + STEP


# -- the instrument -------------------------------------------------------------------------------------

def test_d_want_line_names_demand_locked_and_the_levels(monkeypatch, caplog):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    mapped, locked = 241664, PREFIX + N * TAIL
    a, sched = _tick_world(mapped, locked, _holders())
    with caplog.at_level(logging.INFO):
        _run(a, sched, 3, floor=locked)
    lines = [r.getMessage() for r in caplog.records if "marker=d_want" in r.getMessage()]
    assert len(lines) == 1, lines                    # rate limited: one per 5 s
    ln = lines[0]
    for key in ("mode=off", "demand=%d" % (N * (PREFIX + TAIL)), "locked=%d" % locked, "shared=%d" % (N * (PREFIX + TAIL) - locked),
                "want_old=", "want_new=", " want=", "mapped=%d" % mapped, "avail_min=", "floor=%d" % locked, "incoming=0"):
        assert key in ln, (key, ln)
    # the thesis: the line carries demand - locked >= 20k (here 184320)
    assert N * (PREFIX + TAIL) - locked >= 20000


def test_d_want_line_is_off_the_dual_gate_silent(monkeypatch, caplog):
    for k in DUAL_D:
        monkeypatch.delenv(k, raising=False)
    a, sched = _tick_world(241664, 52224, _holders())
    with caplog.at_level(logging.INFO), mock.patch.object(D._pk, "max_live_id", lambda *x: 0):
        _instr = D._instr_d_want
        _instr(sched, a, {"mode": "off"}, want=1, floor=0, avail_min=1, verdict="grow", level=1,
               p_waiting=False, group_demand=0)
    assert not [r for r in caplog.records if "marker=d_want" in r.getMessage()]


def test_a_failing_reading_never_breaks_the_tick(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(ON, "1")
    a, sched = _tick_world(241664, 52224, _holders())
    with mock.patch.object(D, "d_incoming_tokens", side_effect=RuntimeError("boom")):
        verdicts = _run(a, sched, 3, floor=52224)
    assert set(verdicts) == {"hold"}                 # the old want
