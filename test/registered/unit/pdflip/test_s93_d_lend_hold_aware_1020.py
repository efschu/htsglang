"""S93: the D expert-row lend stays while D's waiting queue holds only requests D's admission will not take.

Metal (S91, done/nf-s91-decode-regression-1020.md): rc20 D ``boot_weg2_..._b40f47c91c_1020_081757.D.log``, TP0:
the waiting queue was non-empty in 99 % of the decode rounds (best boot 1017_183228: 0 %), its members held out by
the admission's own verdicts (``gate=older_live`` 175, ``held_for_older`` 66, ``pdflip_d_head_hold`` 33,
NO_TOKEN / D-WALL-HEAD). ``extend_pending()`` read any waiting request as "an extend is next": every lend went back
0-2 s later (``lend rows_on 4->19`` / ``return rows_on 19->4 (extend pending)``), decode ran on 4-7 expert rows
(rc19 same commit: rows <= 6 bs1 38.4 gpu-ms, rows > 20 bs1 23.5).

What must hold:
(1) a queue held by the park gate (older_live / held_for_older, ``AdmissionGate.skip`` -- the admission's own
    verdict) keeps the lend at round start, and a decode round with it lends after the settle rounds;
(2) a TAKEABLE waiter returns the lend at once, and the line names it;
(3) DQH armed on the unchanged fingerprint holds the queue; a changed fingerprint does not;
(4) the gate counts only on the running set it was taken on (a change: the lend goes back);
(5) the switch off = the old predicate (any waiting request is pending);
(6) rank unity: the verdict reads replicated inputs only -- two ranks with the same queue / running / gate
    answer alike, and no group collective is entered on the lend path;
(7) the held census line is written once per change.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_head_hold  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import d_seats  # noqa: E402
from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_live_wipe_s93", os.path.join(os.path.dirname(__file__), "test_pdflip_d_seat_live_wipe_s1_0928.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

ROW = 4096
STEP_BYTES = 4 * ROW * 2
MIB = 1 << 20


def _req(rid):
    return types.SimpleNamespace(rid=rid, output_ids=[], origin_input_ids=[0] * 8)


def _lent(tms):
    r = h._rank(tms)
    h._pause_all(tms)
    dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
    for p in list(tms.allocs):
        tms.resume(p)
    ctl = r.sched._pdflip_d_seat_vram
    assert ctl.lend(10 * STEP_BYTES) == 8 and ctl.rows_on == 40 and ctl.lent_from == 32
    r.sched.chunked_req = None
    return r, ctl


def _held_queue(sched):
    """rc20's shape: the older pdflip-1-10 decodes; the younger pdflip-1-20 is pressure-parked behind it
    (older_live) -- the gate is the admission's own (d_seats.admission_gate), handed over like the scheduler does."""
    older, parked = _req("pdflip-1-10"), _req("pdflip-1-20")
    d_seats.mark_parked(parked, d_seats.SITE_PRESSURE, now=1.0)
    sched.running_batch = types.SimpleNamespace(reqs=[older])
    sched.waiting_queue = [parked]
    gate = d_seats.admission_gate(sched.waiting_queue, running=list(sched.running_batch.reqs), seats=6)
    assert gate.barrier and "pdflip-1-20" in gate.blocked
    dtl.note_gate(sched, gate, sched.running_batch)
    return older, parked, gate


class _Ctx:
    def __init__(self, tms, free=10 * STEP_BYTES, hold=True):
        self.cms = [h._armed(tms), envs.FLLIPER_PDFLIP_D_LEND_FLOOR_MIB.override("0"),
                    envs.FLLIPER_PDFLIP_D_LEND_KEEP_SHORT.override(True),
                    envs.FLLIPER_PDFLIP_D_LEND_HOLD_AWARE.override(hold),
                    mock.patch.object(dtl, "_card_free_bytes", lambda: free)]

    def __enter__(self):
        for c in self.cms:
            c.__enter__()

    def __exit__(self, *a):
        for c in reversed(self.cms):
            c.__exit__(*a)


def _decode_batch():
    mode = types.SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False, is_idle=lambda: False,
                                 is_extend=lambda: False)
    return types.SimpleNamespace(forward_mode=mode)


# (1) ---------------------------------------------------------------------------------------------------------
def test_round_start_keeps_the_lend_while_the_queue_is_held_by_the_park_gate():
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        assert dtl.extend_pending(r.sched) is False
        assert dtl.round_start(r.sched) is None
        assert ctl.rows_on == 40 and ctl.lent_from == 32, "the lend went back for a request D will not take"


def test_a_decode_round_with_a_held_queue_lends_after_the_settle_rounds():
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        ctl.unlend()
        assert ctl.lent_from is None
        _held_queue(r.sched)
        settle = int(envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.get())
        got = [dtl.on_batch(r.sched, _decode_batch()) for _ in range(settle)]
        assert got[-1] == "lend" and ctl.lent_from is not None, got


# (2) ---------------------------------------------------------------------------------------------------------
def test_a_takeable_waiter_returns_the_lend_at_once_and_is_named(caplog):
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        # an arrival OLDER than the parked one passes the barrier (SA): the admission takes it
        newcomer = _req("pdflip-1-15")
        r.sched.waiting_queue.append(newcomer)
        assert dtl.held_reason(r.sched, newcomer) is None
        caplog.set_level(logging.INFO, logger=dtl.__name__)
        assert dtl.round_start(r.sched) == "return"
        assert ctl.lent_from is None
        assert any("return rows_on 40->32 (extend pending rid=pdflip-1-15)" in m for m in caplog.messages)


def test_a_younger_newcomer_behind_the_barrier_is_held_too():
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        young = _req("pdflip-1-30")
        r.sched.waiting_queue.append(young)
        assert dtl.held_reason(r.sched, young) == "pdflip_d_park_first"
        assert dtl.round_start(r.sched) is None and ctl.lent_from == 32


# (3) ---------------------------------------------------------------------------------------------------------
def _dqh_sched(sched, fp_now):
    dqh = d_head_hold.DHeadHold(group_d=True, token_cut=lambda: True)
    dqh.note_pass(fp=("fp", 1), built_nothing=True)
    sched.pdflip_d_head_hold = dqh
    sched._pdflip_d_head_fingerprint = lambda rb, pv: fp_now
    return dqh


def test_dqh_on_the_unchanged_fingerprint_holds_the_queue():
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        r.sched.running_batch = types.SimpleNamespace(reqs=[_req("pdflip-1-10")])
        r.sched.waiting_queue = [_req("pdflip-1-40")]       # the refused head (NO_TOKEN / wall)
        dqh = _dqh_sched(r.sched, ("fp", 1))
        tick = dqh.state.tick
        assert dtl.held_reason(r.sched, r.sched.waiting_queue[0]) == "pdflip_d_head_hold"
        assert dtl.round_start(r.sched) is None and ctl.lent_from == 32
        assert dqh.state.tick == tick, "the lend read must not tick DQH (should_hold is the scheduler's call)"


def test_dqh_with_a_changed_fingerprint_does_not_hold():
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        r.sched.running_batch = types.SimpleNamespace(reqs=[_req("pdflip-1-10")])
        r.sched.waiting_queue = [_req("pdflip-1-40")]
        _dqh_sched(r.sched, ("fp", 2))
        assert dtl.round_start(r.sched) == "return" and ctl.lent_from is None


# (4) ---------------------------------------------------------------------------------------------------------
def test_the_gate_is_void_once_the_running_set_changed():
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        r.sched.running_batch = types.SimpleNamespace(reqs=[])   # the older one finished: it may resume now
        assert dtl.round_start(r.sched) == "return" and ctl.lent_from is None


# (5) ---------------------------------------------------------------------------------------------------------
def test_switch_off_is_the_old_predicate():
    tms = h.CoreTms()
    with _Ctx(tms, hold=False):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        assert dtl.extend_pending(r.sched) is True
        assert dtl.round_start(r.sched) == "return" and ctl.lent_from is None


# (6) ---------------------------------------------------------------------------------------------------------
def test_two_ranks_with_the_same_replicated_inputs_answer_alike_and_enter_no_collective():
    answers = []
    for _rank in range(2):
        tms = h.CoreTms()
        with _Ctx(tms):
            r, ctl = _lent(tms)
            r.sched._pdflip_group_min_ints = mock.Mock(side_effect=AssertionError("a collective on the lend path"))
            _held_queue(r.sched)
            r.sched.waiting_queue.append(_req("pdflip-1-30"))
            answers.append((dtl.pending_trigger(r.sched),
                            [dtl.held_reason(r.sched, q) for q in r.sched.waiting_queue],
                            dtl.round_start(r.sched), ctl.lent_from))
            r.sched._pdflip_group_min_ints.assert_not_called()
    assert answers[0] == answers[1] == (None, ["pdflip_d_park_older_live", "pdflip_d_park_first"], None, 32)


# (7) ---------------------------------------------------------------------------------------------------------
def test_the_hold_line_is_written_once_per_change(caplog):
    tms = h.CoreTms()
    with _Ctx(tms):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        caplog.set_level(logging.INFO, logger=dtl.__name__)
        for _ in range(5):
            dtl.round_start(r.sched)
        lines = [m for m in caplog.messages if "hold-aware" in m]
        assert len(lines) == 1, lines
        assert "waiting=1 held=1 reasons={'pdflip_d_park_older_live': 1} first_held=pdflip-1-20" in lines[0]
        r.sched.waiting_queue.append(_req("pdflip-1-30"))
        dtl.round_start(r.sched)
        assert len([m for m in caplog.messages if "hold-aware" in m]) == 2


# (8) ---------------------------------------------------------------------------------------------------------
def test_a_held_member_does_not_turn_a_short_pending_extend_into_the_full_return(caplog):
    """KEEP-SHORT prices only the takeable pending work: the held (unpriced, big) member builds no extend this
    pass, so it may not force the full return of every lent row."""
    tms = h.CoreTms()
    with _Ctx(tms, free=400 * MIB), envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override("0.5"):
        r, ctl = _lent(tms)
        _held_queue(r.sched)
        short = types.SimpleNamespace(rid="pdflip-1-15", output_ids=[], weg2_front_x=6400, weg2_front_priced=95,
                                      weg2_front_halfx=3200)
        r.sched.waiting_queue.append(short)
        assert dtl.short_pending_rows(r.sched) is not None
        caplog.set_level(logging.INFO, logger=dtl.__name__)
        got = dtl.round_start(r.sched)
        assert got == "keep", got
        assert ctl.lent_from == 32 and ctl.rows_on == 40, "the held member forced the full return of the lend"
        assert any("keep rows_on 40 (short extend pending rid=pdflip-1-15" in m for m in caplog.messages)


# EXTEND-KORRIDOR (NF rc33 22.10.): the width vote, the short-extend price and the HALT/after-extend trim keep the card's
# corridor law (extend_trim.corridor_line_mib); this module pins the arithmetic it was written for at the old 300 MiB
# halt line -- the law cases are pinned in test_nf_rc33_extendkorridor_1022.
import pytest as _pytest_line300  # noqa: E402


@_pytest_line300.fixture(autouse=True)
def _extend_line_300_for_this_module(monkeypatch):
    from flliper.srt.pdflip import extend_trim as _et

    monkeypatch.setenv("FLLIPER_CORRIDOR_LAW_FLOOR_MIB", "300")
    _et._CACHE["line"] = getattr(_et, "_UNSET", None)
    yield
    _et._CACHE["line"] = getattr(_et, "_UNSET", None)
