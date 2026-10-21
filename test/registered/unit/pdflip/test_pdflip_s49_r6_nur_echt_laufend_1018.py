"""S49 (NF 18.10.): SOLL v2 R6 after the user's decision 18.10. ~21:15Z -- verbatim "Nur echt laufend zählt
(Recommended)" (SCHEDULER-SOLL.md, R6): only a D prefill that already computes counts into the set's S; granted but
not started does not, and a big request waits its T_halb (R4) to the end.

Belegfall (S46 report done/nf-s46-sitzwarten-1018.md (3).4, NF rc13 boot ..._1018_200305, front + D log):
  20:33:48,456 front SET-VERDICT pdflip-22-249 kurz 1426 D_NOW, D-ADMIT
  20:33:48,552 D recv (D-LEAD-MS recv_ms)            -- D holds the POST
  20:33:48,592 front SET-VERDICT pdflip-20-238 p_only 8793 X=7543 waited 1,4 s verdict=REROUTE_P why=R6 S=10219
  20:33:48,633 D first extend of 22-249 (D-LEAD-MS extend_ms)
The R6 flip came 41 ms BEFORE D began the extend, 3,1 s before 238's T_halb; 20-244 was parked before its first token.

The front edge: the book row is counted for R6 only once D's D-EXTEND-BEACON recorded the rid's extend at or after
its grant (pdflip/d_extend_beacon.py); ``_d_carried_tokens`` (X-SUM price) keeps every granted row.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front, _pending  # noqa: E402
from test_pdflip_set_scheduler_1017 import _on  # noqa: E402  (T_halb 4.5 s, F_halb 0.5: SOLL section 5)

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_extend_beacon as dxb  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

X_RC13 = 7543


def _fx(now):
    f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})   # one decode: D decodes
    f.tp_prefill_max_tokens = X_RC13
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    return f


def _big(now, waited=1.4):
    g = _pending("g", 8793, now - waited)
    g.p_only = True
    return g


def _grant_in_leg2(f, rid, tokens, t_grant):
    f._d_pf_book().grant(rid=rid, tokens=tokens, now=t_grant)
    f._d_pf_book().enter_leg2(rid=rid)
    f.groups["D"].outstanding[rid] = t_grant


def _redirectable(f, rid):
    """S56: a D-prefill leg whose stream is still in the lookahead -- the front can move it to P (R6 counts it)."""
    f._rvp_state()
    f.__dict__.setdefault("_d_prefill_legs", {})[rid] = time.time()
    f._leg2_lookahead.add(rid)


def _verdicts(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP SET-VERDICT" in r.getMessage()]


def test_granted_not_started_is_no_reroute_and_the_big_member_waits_its_t_half(monkeypatch, caplog):
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    f.queue = [_big(now)]
    _grant_in_leg2(f, "k", 1426, now - 0.136)
    f._d_ext_reading = lambda: {}                       # D has the POST, no extend recorded
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    assert f.counters["set_reroute_p"] == 0
    (line,) = _verdicts(caplog)
    assert "verdict=COLLECT" in line and "R4 big member waits" in line and "S=8793" in line
    assert any("R6-NOT-STARTED granted=1426 running=0" in r.getMessage() for r in caplog.records)
    # R4 to the end: the T_halb runs out -> the normal R4 flip, not an R6 reroute
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now + 3.2) == "P"
    (line,) = _verdicts(caplog)
    assert "verdict=FLIP_P" in line and "R4 T_halb over" in line
    assert f.counters["set_reroute_p"] == 0


def test_started_is_rerouted_at_once(monkeypatch, caplog):
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    f.queue = [_big(now)]
    _grant_in_leg2(f, "k", 1426, now - 0.136)
    _redirectable(f, "k")
    # D's first extend after the grant, with D's REAL rest (S56: D-EXTEND-BEACON v2)
    f._d_ext_reading = lambda: {dxb.rid_key("k"): (int((now - 0.05) * 1e9), 1426)}
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "P"
    assert f.counters["set_reroute_p"] == 1
    assert any("verdict=REROUTE_P" in m and "S=10219" in m for m in _verdicts(caplog))
    assert f.counters["r6_redirected"] == 1 and "k" in f._rvp_requeue, "S56: the counted D prefill MOVES to P"


def test_an_extend_before_the_grant_is_not_this_grant_started(monkeypatch):
    """A PARK-HANDBACK re-grant of the same rid: D's record of the earlier leg is older than the new grant."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_in_leg2(f, "k", 1426, now - 0.1)
    f._d_ext_reading = lambda: {dxb.rid_key("k"): int((now - 30.0) * 1e9)}
    assert f._dc_carried_running() == 0
    assert f._d_carried_tokens() == 1426


def test_no_reading_counts_nothing_and_is_named(monkeypatch, caplog):
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_in_leg2(f, "k", 1426, now - 0.1)
    f._d_ext_reading = lambda: None
    with caplog.at_level(logging.WARNING, logger="pdflip.front"):
        assert f._dc_carried_running() == 0
        assert f._dc_carried_running() == 0
    assert f.counters["r6_extend_unread"] >= 1
    assert sum("R6-EXTEND-UNREAD" in r.getMessage() for r in caplog.records) == 1, "warned once"


def test_no_rows_reads_nothing(monkeypatch):
    _on(monkeypatch)
    f = _fx(time.time())
    f._d_pf_book()
    calls = []
    f._d_ext_reading = lambda: calls.append(1) or {}
    assert f._dc_carried_running() == 0 and calls == []


def test_switch_off_is_the_book_alone_as_before(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv(dxb.ENV, "0")
    now = time.time()
    f = _fx(now)
    f.queue = [_big(now)]
    _grant_in_leg2(f, "k", 1426, now - 0.136)
    f._d_ext_reading = lambda: {}
    assert f._dc_carried_running() == 1426
    assert f._decode_collect(now) == "P" and f.counters["set_reroute_p"] == 1


def test_default_is_on():
    from flliper.srt.environ import envs
    assert envs.FLLIPER_PDFLIP_D_EXTEND_BEACON.default is True


def test_kurz_path_price_is_unchanged_by_the_beacon(monkeypatch):
    """R2 / X-SUM-PRICE: what D has to prefill (granted, not finished) is still the kurz path's carried."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    _grant_in_leg2(f, "k", 1426, now - 0.1)
    f._d_ext_reading = lambda: {}
    assert f._d_carried_tokens() == 1426 and f._dc_carried() == 1426 and f._dc_d_busy() is True


# ------------------------------------------------------------------------------------- the channel itself (D -> front)

def test_writer_records_the_first_extend_after_the_arrival_only(tmp_path):
    t = [1_000_000_000_000]
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"), slots=4, clock_ns=lambda: t[0])
    sid = os.getsid(0)
    read = lambda: dxb.read_group(str(tmp_path), "D", sid, session_of=lambda pid: sid)  # noqa: E731
    w.note_extend(rids=["a"])                       # never received: nothing
    assert read() == {}
    w.note_recv(rid="a")
    assert read() == {}                             # received, not extended: not started
    w.note_extend(rids=["a", "dec"])
    assert read() == {dxb.rid_key("a"): t[0]}
    t[0] += 5
    w.note_extend(rids=["a"])                       # the next chunk: no new record
    assert read()[dxb.rid_key("a")] == t[0] - 5
    w.note_recv(rid="a")                            # a new leg of the same rid
    w.note_extend(rids=["a"])
    assert read()[dxb.rid_key("a")] == t[0]
    assert dxb.started(read(), "a", t[0] / 1e9) and not dxb.started(read(), "a", t[0] / 1e9 + 1.0)
    for i in range(6):                              # the ring wraps; the newest stay
        w.note_recv(rid=f"r{i}")
        w.note_extend(rids=[f"r{i}"])
    got = read()
    assert dxb.rid_key("r5") in got and dxb.rid_key("r1") not in got


def test_reader_skips_foreign_sessions_and_empty_dirs(tmp_path):
    w = dxb.DExtendBeacon(str(tmp_path / "D-pid1.bin"))
    w.note_recv(rid="a")
    w.note_extend(rids=["a"])
    assert dxb.read_group(str(tmp_path), "D", 7, session_of=lambda pid: 8) is None
    assert dxb.read_group(str(tmp_path / "none"), "D", 7, session_of=lambda pid: 7) is None
    assert dxb.read_group("", "D", 7) is None


def test_front_reads_d_files_from_the_arena(tmp_path, monkeypatch):
    """End to end: a D rank's writer under FLLIPER_HICACHE_ARENA_DIR, the front's reader on the same dir, this
    process's session as group D's session."""
    _on(monkeypatch)
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    w = dxb.build(group="D")
    assert w is not None and dxb.build(group="P") is None
    now = time.time()
    f = _fx(now)
    f.groups["D"].sid = os.getsid(0)
    _grant_in_leg2(f, "k", 1426, now - 0.1)
    w.note_recv(rid="k")
    _redirectable(f, "k")
    assert f._dc_carried_running() == 0
    w.note_extend(rids=["k"], rests=[1426])
    assert f._dc_carried_running() == 1426
    w.note_extend(rids=["k"], rests=[400])          # S56: the next chunk -- D's rest shrinks, the front follows
    assert f._dc_carried_running() == 400


def test_build_off_without_switch_or_arena(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv(dxb.ENV, "0")
    assert dxb.build(group="D") is None
    monkeypatch.delenv(dxb.ENV)
    monkeypatch.delenv("FLLIPER_HICACHE_ARENA_DIR")
    assert dxb.build(group="D") is None


def test_r6_arrival_seat_step_edge_no_flip_now_for_a_granted_not_started_short(monkeypatch, caplog):
    """The controller's arrival-seat step (where rc13's PBOUND-FLIP-NOW came from) sees the same R6 S."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    f.queue = [_big(now)]
    _grant_in_leg2(f, "k", 1426, now - 0.136)
    f._d_ext_reading = lambda: {}
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        wf, _fair, imm = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert imm is None or imm.rid != "g"
    assert not any("verdict=REROUTE_P" in m for m in _verdicts(caplog))


# ------------------------------------------------------------- X-SUM-PRICE (user 05.10.) next to R6's "really running"

def _halb_set(f, rid, u, t):
    f._set_st()["shorts"][rid] = (t, u, int(f.tp_prefill_max_tokens))


def test_xsum_a_set_is_not_handed_to_d_on_top_of_an_unstarted_grant_and_no_flip(monkeypatch, caplog):
    """NF int23xsum 09.10. (X=6008): s0 (5200, halb) is granted, s1 (5200, halb) arrives. The grant is no flip reason (not
    begun, R6 user 18.10.) -- and, while D DECODES and s1's T_halb is over (R3 timer end), s1 is not put on D on top of it
    (5200 + 5200 > X, X-SUM-PRICE, NS:S49): the set WAITS. D begins s0 -> R6 (really running) reroutes.
    rc30 R8 fix (21.10.): on an IDLE D this hold no longer exists -- R8 decides at once (test_nf_rc30_r8_1021)."""
    _on(monkeypatch)
    now = time.time()
    f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})   # D decodes (R3 timer end, not R8)
    f.tp_prefill_max_tokens = 6008
    f.t_awake = now - 60.0
    f._sync_batch_gate = lambda: None
    f._d_pf_book().grant(rid="s0", tokens=5200, now=now - 0.05)
    _halb_set(f, "s1", 5200, now - 5.0)
    f._d_ext_reading = lambda: {}
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(now) == "hold"
    (line,) = _verdicts(caplog)
    assert "verdict=COLLECT" in line and "X-SUM" in line and "not started" in line
    assert f.counters["set_reroute_p"] == 0 and f.counters["set_xsum_hold_unstarted"] == 1
    _redirectable(f, "s0")
    f._d_ext_reading = lambda: {dxb.rid_key("s0"): (time.time_ns(), 5200)}   # D began s0: really running, rest 5200
    assert f._decode_collect(time.time()) == "P" and f.counters["set_reroute_p"] == 1


def test_xsum_hold_ends_on_d_when_the_grant_is_done(monkeypatch):
    _on(monkeypatch)
    now = time.time()
    f = _front(running=["dec"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008
    f.t_awake = now - 60.0
    f._d_pf_book().grant(rid="s0", tokens=5200, now=now - 0.05)
    _halb_set(f, "s1", 5200, now - 5.0)
    f._d_ext_reading = lambda: {}
    assert f._decode_collect(now) == "hold"
    f._d_pf_book().done(rid="s0")                                             # first content: D's prefill over
    assert f._decode_collect(time.time()) == "D"
    assert f._set_st()["released"].get("s1") == "D"


def test_xsum_hold_never_holds_a_flip_verdict(monkeypatch):
    """R4 at T_halb end (or company) stays a flip; the hold only concerns a D verdict."""
    _on(monkeypatch)
    now = time.time()
    f = _fx(now)
    f.queue = [_big(now, waited=5.0)]
    _grant_in_leg2(f, "k", 1426, now - 0.1)
    f._d_ext_reading = lambda: {}
    assert f._decode_collect(now) == "P" and f.counters["set_flip_p"] == 1
    assert f.counters["set_xsum_hold_unstarted"] == 0
