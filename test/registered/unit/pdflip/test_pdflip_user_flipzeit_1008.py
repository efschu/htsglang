"""USER-FLIPZEIT (08.10., NF dauer10081045 int17 49873f9af8, D->P flip epoch 5).

The user's flip time (rule 07.10. ~04:20Z, "IDLE ZEIT IST NICHT FLIPZEIT"):
D>P = the LATER of (last D token, arrival of the waiter) -> the begin of P's
first prefill chunk on PP0; P>D = the end of P's last prefill chunk -> D's
first decode token. Idle never counts.

Metal epoch 5: the front's ``PDFLIP-FLIPCYCLE stage=total`` said 30401 ms
(quiesce 28180 ms) and ``flip_user_time.flip_user_ms`` 30974 ms with
``start_source=park_rpc_sent`` (10:53:16.622). But D kept producing tokens
after the park RPC -- its #248h capacity re-queue decoded pdflip-0-9/-0-12 to
10:53:45.2 (PDFLIP-SERVED 1791456825.214) -- and P's PP0 began its first chunk
at 1791456827.596. The clock took the park RPC's send for "decode end"; no
field carried the user's number (~2.4 s). Now every D>P ``flip_user_time``
and every P>D ``flip_first_work`` carries ``user_flipzeit_ms`` with its
support values; the begin -> done total stays as ``flip_total_ms`` with a
label that says it includes quiesce/idle. A missing support value gives
``user_flipzeit_ms`` None and the reason, never a substitute.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import front_state_ipc as fsi  # noqa: E402

# metal, front clock (time.time) -- events.jsonl of nfint4h6ablxcdauer-boot-20261008T104503Z-8390
PARK_SENT = 1791456796.622      # flip_user_time.start_ts (park_rpc_sent)
FLIP_BEGIN = 1791456797.171     # PDFLIP-FLIP begin epoch=4
WAITER = 1791456701.677         # pdflip-2-39: DP-WAIT wait_s=125.9 at 10:53:47.577
LAST_D = 1791456825.214         # PDFLIP-SERVED group=D rid=pdflip-0-9 10:53:45.214
DONE = 1791456827.571           # PDFLIP-FLIP done epoch=5
PP0 = 1791456827.596            # PDFLIP-FLIP-PPFWD stage=first prefill_start_ts


def _epoch5_clock():
    c = fsi.DpFlipClock()
    c.note_d_served(1791456795.0)                  # an earlier served leg 2
    c.note_park(4, PARK_SENT, 546.0)
    c.begin(4, FLIP_BEGIN, oldest_waiter_ts=WAITER)
    for t in (1791456797.6, 1791456810.0, LAST_D - 0.02):   # D streams on during the quiesce
        c.note_d_token(t)
    c.note_d_served(LAST_D)                         # pdflip-0-9's leg 2 served
    c.done(DONE)
    return c


def test_epoch5_the_user_flip_time_starts_at_ds_last_token_not_at_the_park_rpc():
    """Red on 49873f9af8: no ``user_flipzeit_ms`` (and no ``note_d_token``):
    the only "user" field said 30974 ms from the park RPC's send while D was
    still producing tokens for 28.6 s."""
    ev = _epoch5_clock().first_prefill("pdflip-2-39", FLIP_BEGIN, {"ts": PP0, "pid": 775, "ct": 32, "pp_rank": 0},
                                       rid_arrival_ts=WAITER)
    assert ev["user_flipzeit_ms"] == round((PP0 - LAST_D) * 1000.0) == 2382
    assert ev["user_flipzeit_start_source"] == "d_leg2_served"
    assert (ev["last_d_token_ts"], ev["last_d_token_source"]) == (round(LAST_D, 3), "d_leg2_served")
    assert (ev["waiter_arrival_ts"], ev["waiter_arrival_source"]) == (round(WAITER, 3), "first_leg1_rid_arrival")
    assert ev["first_p_chunk_ts"] == round(PP0, 3) and ev["user_flipzeit_missing"] is None
    # the instrument keeps its value, labelled; the old fields stay (dashboard)
    assert ev["flip_total_ms"] == ev["parts"]["legs_ms"] == round((DONE - FLIP_BEGIN) * 1000.0)
    assert "quiesce" in ev["flip_total_note"] and "NOT the user flip time" in ev["flip_total_note"]
    assert ev["start_source"] == "park_rpc_sent" and ev["flip_user_ms"] == round((PP0 - PARK_SENT) * 1000.0)


def test_a_waiter_that_arrives_after_ds_last_token_starts_the_span():
    """Idle before the arrival is not flip time: start = the arrival."""
    c = fsi.DpFlipClock()
    c.note_d_token(50.0)
    c.begin(3, 60.0, oldest_waiter_ts=58.0)
    c.done(62.0)
    ev = c.first_prefill("r", 62.0, {"ts": 62.5}, rid_arrival_ts=58.0)
    assert (ev["user_flipzeit_start_source"], ev["user_flipzeit_ms"]) == ("waiter_arrival", 4500)
    # a flip with no waiter at its begin: the first leg 1's own arrival is the waiter
    c.begin(5, 100.0, oldest_waiter_ts=None)
    c.done(102.0)
    ev = c.first_prefill("r2", 140.0, {"ts": 140.4}, rid_arrival_ts=139.9)
    assert (ev["waiter_arrival_source"], ev["user_flipzeit_ms"]) == ("first_leg1_rid_arrival", 500)


def test_dp_without_a_proven_waiter_arrival_has_no_start_and_pre_wait_is_only_diagnosis():
    """NF int18 epoch 11 (planner decision 08.10.): D>P flip time = first P chunk - max(last D token,
    waiter arrival). A D token alone is a substitute for an unproven arrival -> null, not a number.
    The idle gap between the last token and a LATER arrival is ``pre_wait_ms``, never flip time."""
    c = fsi.DpFlipClock()
    c.note_d_token(50.0)
    c.begin(3, 60.0, oldest_waiter_ts=None)
    c.done(62.0)
    ev = c.first_prefill("r", 62.0, {"ts": 62.5}, rid_arrival_ts=None)   # no arrival proven anywhere
    assert ev["user_flipzeit_ms"] is None and ev["user_flipzeit_start_ts"] is None
    assert ev["user_flipzeit_missing"] == "start_missing:waiter_arrival_unproven"
    c.note_d_token(70.0)
    c.begin(5, 80.0, oldest_waiter_ts=75.0)                              # arrived 5 s after the last token
    c.done(82.0)
    ev = c.first_prefill("r2", 82.0, {"ts": 82.5}, rid_arrival_ts=75.0)
    assert (ev["user_flipzeit_ms"], ev["pre_wait_ms"]) == (7500, 5000)
    c.note_d_token(90.0)
    c.begin(7, 100.0, oldest_waiter_ts=85.0)                             # waiter held since before the last token
    c.done(102.0)
    ev = c.first_prefill("r3", 102.0, {"ts": 102.5}, rid_arrival_ts=85.0)
    assert (ev["user_flipzeit_ms"], ev["pre_wait_ms"]) == (12500, 0)


def test_a_missing_support_value_is_null_with_its_reason_never_a_substitute():
    c = _epoch5_clock()
    ev = c.first_prefill("pdflip-2-39", FLIP_BEGIN, {"missing": "no_rise_by_leg1_end"})
    assert ev["user_flipzeit_ms"] is None
    assert ev["user_flipzeit_missing"] == "end_missing:no_rise_by_leg1_end"
    c = fsi.DpFlipClock()                             # no D token, no waiter, nothing served
    c.begin(1, 10.0, oldest_waiter_ts=None)
    c.done(12.0)
    ev = c.first_prefill("r", 12.0, {"ts": 12.2})
    assert ev["user_flipzeit_ms"] is None and ev["user_flipzeit_missing"].startswith("start_missing")


def test_pd_ends_of_ps_last_chunk_from_the_beacons_to_ds_first_token():
    """P>D: P's last finished forward (its ranks' beacons) -> D's first decode
    token; without a beacon the front's receipt of P's last leg 1; a forward
    still running or ending after the token is not P's last chunk end."""
    ns = 1_000_000_000
    beacons = {775: (40, 101 * ns, int(102.5 * ns)), 776: (40, 101 * ns, int(102.9 * ns)),
               777: (41, int(103.5 * ns), int(103.0 * ns))}       # 777: inside a forward
    assert fsi.p_last_forward_done(beacons) == 102.9
    assert fsi.p_last_forward_done(beacons, not_after=102.6) == 102.5
    assert fsi.p_last_forward_done({}) is None
    ev = {"epoch": 6, "dir": "P>D", "flip_begin_ts": 103.0, "first_work_ts": 106.5, "done_ts": 105.9,
          "p_end_ts": 103.0, "flip_user_ms": 3500}
    out = fsi.FirstWorkClock.pd_user(ev, 102.9, None, "d_work_waiting_at_flip_begin:handoff")
    assert (out["user_flipzeit_ms"], out["last_p_chunk_end_source"], out["first_d_token_ts"]) == \
        (3600, "p_beacon_last_forward_done", 106.5)
    assert out["flip_total_ms"] == 2900 and out["flip_user_ms"] == 3500 and out["p_leg1_end_ts"] == 103.0
    out = fsi.FirstWorkClock.pd_user(ev, None, None, None)
    assert (out["user_flipzeit_ms"], out["last_p_chunk_end_source"]) == (3500, "p_leg1_end")


def _pd_front(reason, rows):
    book = SimpleNamespace(rows=rows)
    return SimpleNamespace(_user_flip_reason={6: reason}, _req_book_obj=book, groups={}, tag="")


def test_the_front_starts_an_idle_pd_flip_at_the_first_tokens_arrival(monkeypatch):
    """An idle P>D flip (nothing waited for D at its begin): the request whose
    token ends it is the waiter -- P's idle end -> its arrival is not flip
    time. Without a rid the waiter is unknown: null, not P's end."""
    monkeypatch.setattr(front_mod.Front, "_group_beacons", lambda self, g: {})
    ev = {"epoch": 6, "dir": "P>D", "flip_begin_ts": 103.0, "first_work_ts": 160.4, "done_ts": 105.9,
          "p_end_ts": 103.0, "rid": "pdflip-6-9"}
    f = _pd_front("idle", {"pdflip-6-9": {"arrival_ts": 160.0}})
    out = front_mod.Front._pd_user_flipzeit(f, ev)
    assert (out["user_flipzeit_ms"], out["user_flipzeit_start_source"]) == (400, "waiter_arrival")
    assert f._flip_user_last["P>D"]["user_flipzeit_ms"] == 400
    out = front_mod.Front._pd_user_flipzeit(_pd_front("idle", {}), dict(ev, rid=None))
    assert out["user_flipzeit_ms"] is None and out["user_flipzeit_missing"] == "idle_flip_waiter_unknown"
    out = front_mod.Front._pd_user_flipzeit(_pd_front("handoff", {}), dict(ev, first_work_ts=106.5))
    assert (out["user_flipzeit_ms"], out["waiter_arrival_source"]) == \
        (3500, "d_work_waiting_at_flip_begin:handoff")


def test_the_front_feeds_the_clock_from_ds_stream_and_labels_the_total():
    """Wiring: D's stream content feeds ``note_d_token`` (leg 2), the first
    leg 1 after a D->P flip passes its arrival, both directions are noted,
    state.json carries front.flip_user, the stage=total line its label."""
    leg2 = inspect.getsource(front_mod.Front.leg2)
    assert "note_d_token(" in leg2
    leg1 = inspect.getsource(front_mod.Front.leg1)
    assert "rid_arrival_ts=p.t_arrive" in leg1 and "_user_flipzeit_note(self, _dp)" in leg1
    assert "_pd_user_flipzeit(self, ev)" in inspect.getsource(front_mod.Front._ipc_first_work_seen)
    assert "_pd_user_flipzeit(self, ev)" in inspect.getsource(front_mod.Front._ipc_first_work_at)
    src = inspect.getsource(front_mod.Front)
    assert 'out["flip_user"]' in src
    assert '"PDFLIP-FLIPCYCLE stage=total dir=%s>%s epoch=%d ms=%.0f floor_ms=%d"\n' in src
    assert '" flip_total_ms=%.0f (%s)",' in src and "FLIP_TOTAL_NOTE_SHORT)" in src
