"""S56 (NF rc15 19.10.): R6 counted a running D prefill with the FRONT's arrival price instead of D's real rest, and
did not move it to P.

User 19.10.: "um 9:20:26 wird nur EIN request prefillt und zwar 4492 neue token". SOLL v2 R6 (with the user decision
18.10. "Nur echt laufend zählt") + R7 + R11.

Belegfall (boot ..._489909d138_1019_071429, front + D log):
  07:20:17,610 front SET-VERDICT pdflip-6-11:3513 halb COLLECT        -- front price 3513 (credit 93952)
  07:20:22,168 ARRIVAL-SEAT pdflip-6-11 d_prefill (R3 D_AFTER_TIMER)
  07:20:22 D   MATCH-CENSUS-DEEP pdflip-6-11 accepted=98240; X-GATE uncached=977; #969 EXTENT (98240, 99217, ..., 977)
  07:20:22,831 front SET-VERDICT pdflip-6-12:4492 halb COLLECT
  07:20:23,036 front SET-VERDICT S=8005 X=6216 verdict=REROUTE_P why=R6 ... D prefill of 3513 running
  07:20:24,705 PARK-RUNNING rids=['pdflip-6-11']                     -- 6-11 was NOT moved to P
  07:20:30     P prefilled ONE request: 4492 new tokens               -- a flip for a halb request (R2/R3/R7)
D's real rest was 977: S = 4492 + 977 = 5469 < X = 6216 -> no flip (R7); the set collects on (R3).
"""
from __future__ import annotations

import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402  (T_halb 4.5 s, F_halb 0.5: SOLL section 5)

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

X_RC15 = 6216
FRONT_PRICE_611 = 3513   # X-EXACT-PRICE pdflip-6-11 pending=3513
D_REST_611 = 977         # D: 99217 - 98240 (X-GATE uncached=977)
U_612 = 4492             # pdflip-6-12, credit 0


def _fx(now):
    f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})   # D decodes (R3/R4 wait)
    f.tp_prefill_max_tokens = X_RC15
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    return f


def _grant_611(f, now, *, rest, redirectable=True):
    """pdflip-6-11: granted to D at its FRONT price, posted, D began the extend and reports ``rest``."""
    f._d_pf_book().grant(rid="pdflip-6-11", tokens=FRONT_PRICE_611, now=now - 0.9)
    f._d_pf_book().enter_leg2(rid="pdflip-6-11")
    f.groups["D"].outstanding["pdflip-6-11"] = now - 0.9
    f._rvp_state()
    if redirectable:
        f.__dict__.setdefault("_d_prefill_legs", {})["pdflip-6-11"] = now - 0.9
        f._leg2_lookahead.add("pdflip-6-11")
    f._d_ext_reading = lambda: {dxb.rid_key("pdflip-6-11"): (int((now - 0.8) * 1e9), int(rest))}


def _halb_612(f, now):
    f._set_st()["shorts"]["pdflip-6-12"] = (now, U_612, X_RC15)


def _verdicts(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP SET-VERDICT" in r.getMessage()]


def test_rc15_scenario_no_flip_for_a_halb_request_d_rest_977(monkeypatch, caplog):
    """THE REGRESSION: S = 4492 + D's real rest 977 = 5469 < X -> COLLECT, no REROUTE_P (R6 counts D's number, R7)."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_611(f, now, rest=D_REST_611)
    _halb_612(f, now)
    assert f._dc_carried_running() == D_REST_611, "R6's running term is D's rest, not the front's 3513"
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    (line,) = _verdicts(caplog)
    assert "verdict=COLLECT" in line and "S=5469" in line and "X=6216" in line, line
    assert f.counters["set_reroute_p"] == 0 and f.counters["r6_redirected"] == 0
    assert "pdflip-6-11" not in f._rvp_requeue, "nothing is moved without a flip"


def test_rc15_scenario_after_t_half_the_set_goes_to_d_r3(monkeypatch, caplog):
    """R3: T_halb runs out with S < X -> the set goes to D, still no flip."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_611(f, now, rest=D_REST_611)
    _halb_612(f, now - 4.6)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "D"
    assert any("verdict=D_AFTER_TIMER" in m for m in _verdicts(caplog))
    assert f.counters["set_reroute_p"] == 0


def test_mutant_guard_a_real_big_d_rest_still_reroutes_and_moves_the_prefill(monkeypatch, caplog):
    """Danger direction 1: R6 must keep firing. D's real rest 3000 + 4492 = 7492 > X -> REROUTE_P, and the running D
    prefill is MOVED to P (PARK-HANDBACK record), not left running on D."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_611(f, now, rest=3000)
    _halb_612(f, now)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "P"
    assert any("verdict=REROUTE_P" in m and "S=7492" in m for m in _verdicts(caplog))
    assert f.counters["r6_redirected"] == 1
    assert f._rvp_requeue["pdflip-6-11"]["reason"] == "park_handback"
    assert any("PDFLIP R6-REDIRECT" in r.getMessage() and "pdflip-6-11:3000" in r.getMessage()
               for r in caplog.records)


def test_a_rest_that_cannot_move_to_p_does_not_count(monkeypatch, caplog):
    """User finding (b): R6 may count the running rest only if it is really redirected. A committed stream (not in
    the lookahead) runs on D to its end -> 0 in S, named, no flip for 4492."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_611(f, now, rest=5000, redirectable=False)
    _halb_612(f, now)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._decode_collect(now) == "hold"
    assert f.counters["set_reroute_p"] == 0
    assert any("R6-REST-STAYS-ON-D rid=pdflip-6-11 d_rest=5000" in r.getMessage() for r in caplog.records)


def test_no_reported_rest_counts_nothing_never_the_front_price(monkeypatch):
    """No estimate: a started prefill without D's number adds 0 (named), never the granted 3513."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_611(f, now, rest=D_REST_611)
    f._d_ext_reading = lambda: {dxb.rid_key("pdflip-6-11"): int((now - 0.8) * 1e9)}   # start only, no rest
    assert f._dc_carried_running() == 0
    assert f.counters["r6_rest_unreported"] >= 1
    assert f._d_carried_tokens() == FRONT_PRICE_611, "the kurz path's X-SUM book is unchanged"


def test_mutant_guard_r7_no_flip_below_x_for_every_rest_under_the_gap(monkeypatch):
    """Danger direction 2 (R7): for every D rest with 4492 + rest < X the verdict is never a flip."""
    _on(monkeypatch)
    for rest in (0, 1, 977, X_RC15 - U_612 - 1):
        now = time.time()
        f = _fx(now)
        _grant_611(f, now, rest=rest)
        _halb_612(f, now)
        assert f._decode_collect(now) == "hold", rest
        assert f.counters["set_reroute_p"] == 0 and f.counters["set_flip_p"] == 0, rest


# ------------------------------------------------------------------------- the channel: D's real rest, rank-agreed

def _req(n_prompt, n_prefix, n_out=0):
    return types.SimpleNamespace(origin_input_ids=[0] * n_prompt, output_ids=[0] * n_out,
                                 prefix_indices=[0] * n_prefix)


def test_req_rest_is_prompt_plus_output_minus_the_extended_prefix():
    assert dxb.req_rest(_req(99217, 98240)) == D_REST_611          # rc15 pdflip-6-11 (#969 EXTENT)
    assert dxb.req_rest(_req(4492, 0)) == 4492
    assert dxb.req_rest(_req(21601, 21568, n_out=160)) == 193      # a park resume: output counts as prompt
    assert dxb.req_rest(object()) == -1


def test_writer_reports_the_rest_of_every_chunk_under_the_first_start(tmp_path):
    t = [2_000_000_000_000]
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), slots=8, clock_ns=lambda: t[0])
    sid = 11
    read = lambda: dxb.read_group_full(str(tmp_path), "D", sid, session_of=lambda pid: sid)  # noqa: E731
    w.note_recv(rid="a")
    w.note_extend(rids=["a"], rests=[12000])
    assert read() == {dxb.rid_key("a"): (t[0], 12000)}
    t0 = t[0]
    t[0] += 10
    w.note_extend(rids=["a"], rests=[5784])                        # second chunk
    assert read() == {dxb.rid_key("a"): (t0, 5784)}
    assert dxb.rest_of(read(), "a", t0 / 1e9) == 5784
    assert dxb.rest_of(read(), "a", t0 / 1e9 + 1.0) is None         # an extend before the grant: not this grant's
    assert dxb.started(read(), "a", t0 / 1e9)


def test_rank_agreement_the_reading_does_not_depend_on_the_file_order(tmp_path):
    """R11 / Rang-Einigkeit: every rank of D writes its record; the front's reduction is deterministic (newest per
    rank, max over ranks) -- the same answer for any order of the rank files."""
    sid = 12
    ws = [dxb.DExtendBeacon(str(tmp_path / f"D-pid{i}.bin"), clock_ns=lambda: 3_000_000_000_000) for i in (1, 2, 3)]
    for w in ws:
        w.note_recv(rid="a")
    for w in reversed(ws):
        w.note_extend(rids=["a"], rests=[977])
    one = dxb.read_group_full(str(tmp_path), "D", sid, session_of=lambda pid: sid)
    assert one == {dxb.rid_key("a"): (3_000_000_000_000, 977)}
    ws[1].note_extend(rids=["a"], rests=[978])                     # one rank disagrees: the max, by name of the rule
    two = dxb.read_group_full(str(tmp_path), "D", sid, session_of=lambda pid: sid)
    assert two[dxb.rid_key("a")] == (3_000_000_000_000, 978)


def test_v1_files_are_read_as_starts_without_a_rest(tmp_path):
    import struct
    path = tmp_path / "D-pid9.bin"
    raw = bytearray(struct.pack(dxb._HDR_FMT, dxb._MAGIC, 1, 2, 1, 9) + b"\0" * 32)
    struct.pack_into("<Qq", raw, dxb._HDR_SIZE, dxb.rid_key("a"), 5)
    path.write_bytes(bytes(raw))
    got = dxb.read_group_full(str(tmp_path), "D", 4, session_of=lambda pid: 4)
    assert got == {dxb.rid_key("a"): (5, -1)}
    assert dxb.rest_of(got, "a", 0.0) is None and dxb.started(got, "a", 0.0)


def test_rc15_red_on_base_the_front_price_made_s_8005(monkeypatch, caplog):
    """The rc15 reading shape (D's start only, as before S56): the base counted the FRONT price 3513 -> S=8005 >
    X=6216 -> REROUTE_P (the metal line 07:20:23,036). Now no D rest -> 0, and the halb set collects (R3/R7)."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_611(f, now, rest=D_REST_611)
    f._d_ext_reading = lambda: {dxb.rid_key("pdflip-6-11"): int((now - 0.8) * 1e9)}
    _halb_612(f, now)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    assert not any("verdict=REROUTE_P" in m for m in _verdicts(caplog)), _verdicts(caplog)
