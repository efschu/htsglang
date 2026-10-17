# SPDX-License-Identifier: Apache-2.0
"""set_x_wache (h16u 1017, SCHEDULER-SOLL v2 R2/R3/R6/R7): measures each SET-VERDICT line, both danger directions
at the boundary (S == X, X-1), the throttle, the hour balance, the old-boot line, the boot switch."""
import importlib.util
import os
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
WACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "devtools", "set_x_wache.py"))
DEPLOYED = "/spinning/gpu-arb/deskq/wacht/set_x_wache.py"


@pytest.fixture()
def W(tmp_path, monkeypatch):
    monkeypatch.setenv("SX_OUT", str(tmp_path))
    monkeypatch.setenv("SX_EVIDENCE", str(tmp_path / "ev"))
    assert os.path.exists(WACHE), "devtools/set_x_wache.py missing"
    spec = importlib.util.spec_from_file_location("set_x_wache_t", WACHE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vline(ts="2026-10-17 10:00:01,123", *, n=2, members="a:3500,b:3200", S=6700, X=6000, halfX=3000,
          classes="halb,halb", dd=1, seats=4, th="4.5", w="2.0", verdict="FLIP_P", why="R3 S>=X"):
    return ("[%s] INFO weg2.front: WEG2 SET-VERDICT epoch=7 set=3 n=%d members=%s S=%d X=%d halfX=%d classes=%s "
            "d_decoding=%d seats_free=%d t_half_s=%s waited_s=%s verdict=%s why=%s") % (
        ts, n, members, S, X, halfX, classes, dd, seats, th, w, verdict, why)


def feed(W, st, lines):
    out = []
    for ln in lines:
        st.feed(ln, out.append)
    return out


def art_of(W, **kw):
    v = W.parse_verdict(vline(**kw))
    assert isinstance(v, dict), v
    return W.check(v)


# --- the measurement, each rule, both directions ------------------------------------------------------------------


def test_flip_under_x_is_r7_and_at_x_is_not(W):
    assert art_of(W, S=5999, verdict="FLIP_P") == "R7"
    assert art_of(W, S=6000, verdict="FLIP_P") is None
    assert art_of(W, S=6700, verdict="FLIP_P") is None


def test_p_only_flip_under_x_is_no_violation(W):
    assert art_of(W, S=500, n=1, members="v:500", classes="p_only", verdict="FLIP_P") is None
    assert art_of(W, S=500, n=1, members="v:500", classes="halb,p_only", verdict="FLIP_P") is None


def test_reroute_under_x_is_r7(W):
    assert art_of(W, S=5000, verdict="REROUTE_P") == "R7"
    assert art_of(W, S=9000, verdict="REROUTE_P") is None


def test_d_after_timer_at_or_over_x_is_r3(W):
    assert art_of(W, S=6000, verdict="D_AFTER_TIMER") == "R3"
    assert art_of(W, S=7000, verdict="D_AFTER_TIMER") == "R3"
    assert art_of(W, S=5999, verdict="D_AFTER_TIMER") is None


def test_d_prefill_over_x_is_r6(W):
    assert art_of(W, S=6001, n=1, members="a:300", classes="kurz", w="0.0", verdict="D_NOW", why="R2") == "R6"
    assert art_of(W, S=300, n=1, members="a:300", classes="kurz", w="0.0", verdict="D_NOW", why="R2") is None


def test_short_waiting_is_r2(W):
    kw = dict(S=300, n=1, members="a:300", classes="kurz", verdict="D_NOW", why="R2")
    assert art_of(W, w="0.0", **kw) is None
    assert art_of(W, w="0.5", **kw) is None           # the limit is > 0.5
    assert art_of(W, w="0.6", **kw) == "R2"
    assert art_of(W, w="0.0", **{**kw, "verdict": "FLIP_P"}) == "R2"      # a short must never flip
    assert art_of(W, w="0.0", **{**kw, "verdict": "COLLECT"}) == "R2"     # nor collect


def test_not_d_and_seat_wait_are_allowed_reasons_but_an_unexplained_wait_is_r2(W):
    """R5 (corrected): NOT_D with a reason and D_NOW why=seat_wait are no R2 findings; waiting without a reason is."""
    kw = dict(S=300, n=1, members="a:300", classes="kurz")
    assert art_of(W, w="0.0", verdict="NOT_D", why="d_budget", **kw) is None
    assert art_of(W, w="2.0", verdict="NOT_D", why="yield_or_gate", **kw) is None
    assert art_of(W, w="0.0", verdict="NOT_D", why="mystery", **kw) == "R2"
    assert art_of(W, w="3.0", verdict="D_NOW", why="seat_wait", **kw) is None
    assert art_of(W, w="3.0", verdict="D_NOW", why="R2", **kw) == "R2"


def test_collect_is_no_decision(W):
    assert art_of(W, S=3500, n=1, members="a:3500", classes="halb", w="0.0", verdict="COLLECT") is None
    assert art_of(W, S=6000, n=1, members="a:6000", classes="gross", w="1.0", verdict="COLLECT") is None


def test_format_mismatch_is_flagged_not_dropped(W, tmp_path):
    bad = ["[2026-10-17 10:00:00,000] INFO x: WEG2 SET-VERDICT epoch=7 set=3 S=6700 verdict=FLIP_P",
           vline().replace("verdict=FLIP_P", "verdict=SPRING"),
           vline().replace("classes=halb,halb", "classes=riesig")]
    for b in bad:
        assert W.parse_verdict(b) == "FORMAT", b
    st = W.State(clock=lambda: 1000.0, log=str(tmp_path / "x.log"), stunde=str(tmp_path / "s.txt"))
    out = feed(W, st, bad[:1])
    assert out and "FORMAT" in out[0]
    assert W.parse_verdict("some other line") is None


# --- the wiring: feed -> check -> emit / log / hour balance -------------------------------------------------------------


def test_stdout_only_violations_and_throttle(W, tmp_path):
    now = [1000.0]
    st = W.State(clock=lambda: now[0], log=str(tmp_path / "set_x.log"), stunde=str(tmp_path / "set_x_stunde.txt"))
    good = vline(S=6700, verdict="FLIP_P")
    bad = vline(S=4000, verdict="FLIP_P")
    out = feed(W, st, [good, good])
    assert out == []                                  # a conforming set is silent
    out = feed(W, st, [bad])
    assert len(out) == 1 and "R7" in out[0]
    out = feed(W, st, [bad, bad])
    assert out == []                                  # throttled: 1 line per art per 120 s
    now[0] += 121.0
    out = feed(W, st, [bad])
    assert len(out) == 1 and "n=3 seit letzter Meldung" in out[0]     # the suppressed ones are counted
    other = feed(W, st, [vline(S=7000, verdict="D_AFTER_TIMER")])       # another art is not throttled by R7
    assert len(other) == 1 and "R3" in other[0]
    log = open(tmp_path / "set_x.log").read().splitlines()
    assert len(log) == 3 and all(len(x) <= 200 for x in log)


def test_hour_balance_counts_and_old_boot_line(W, tmp_path):
    stunde = tmp_path / "set_x_stunde.txt"
    st = W.State(clock=lambda: 5000.0, log=str(tmp_path / "set_x.log"), stunde=str(stunde))
    feed(W, st, [vline("2026-10-17 10:00:01,000", S=6700, verdict="FLIP_P"),
                 vline("2026-10-17 10:10:01,000", S=4000, verdict="FLIP_P"),
                 vline("2026-10-17 10:20:01,000", S=3500, n=1, members="a:3500", classes="halb", w="0.0", verdict="COLLECT"),
                 vline("2026-10-17 11:00:01,000", S=300, n=1, members="k:300", classes="kurz", w="0.0", verdict="D_NOW", why="R2")])
    rows = stunde.read_text().splitlines()
    assert len(rows) == 1 and rows[0].startswith("2026-10-17 10Z Sets 2 (FLIP_P 2,")
    assert "Verstoesse R7 1, R3 0, R6 0, R2 0, FORMAT 0" in rows[0] and "COLLECT-Zeilen 1" in rows[0]
    # old boot: no SET-VERDICT at all -> one line per hour
    old = tmp_path / "old.txt"
    st2 = W.State(clock=lambda: 5000.0, log=str(tmp_path / "o.log"), stunde=str(old))
    feed(W, st2, ["[2026-10-17 10:00:01,000] INFO weg2.front: WEG2-FLIP begin epoch=0 sleep=D wake=P",
                  "[2026-10-17 11:00:01,000] INFO weg2.front: WEG2-FLIP begin epoch=1 sleep=P wake=D",
                  "[2026-10-17 12:00:01,000] INFO weg2.front: WEG2-FLIP begin epoch=2 sleep=D wake=P"])
    assert old.read_text().splitlines() == ["2026-10-17 10Z SET-VERDICT fehlt (alter Stand)",
                                            "2026-10-17 11Z SET-VERDICT fehlt (alter Stand)"]


def test_newest_front_switches_on_boot_change(W, tmp_path):
    ev = tmp_path / "ev"
    ev.mkdir()
    a, b = ev / "boot_weg2_A_1.front.log", ev / "boot_weg2_B_2.front.log"
    a.write_text("x\n")
    time.sleep(0.01)
    b.write_text("y\n")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    assert W.newest_front() == str(b)
    os.utime(a, (3000, 3000))
    assert W.newest_front() == str(a)


def test_replay_never_writes_live_files_and_shows_every_case(W, tmp_path, capsys):
    f = tmp_path / "t.front.log"
    f.write_text("\n".join([vline("2026-10-17 10:00:01,000", S=4000, verdict="FLIP_P"),
                            vline("2026-10-17 10:00:02,000", S=4100, verdict="FLIP_P")]) + "\n")
    W.replay(str(f))
    out = capsys.readouterr().out
    assert out.count("R7") >= 2 or "n=2" in out                    # replay shows each violation (no throttle)
    assert not os.path.exists(tmp_path / "set_x.log") and not os.path.exists(tmp_path / "set_x_stunde.txt")


def test_repo_copy_equals_the_deployed_copy():
    if not os.path.exists(DEPLOYED):
        pytest.skip("deployed copy absent on this host")
    assert open(WACHE).read() == open(DEPLOYED).read()
