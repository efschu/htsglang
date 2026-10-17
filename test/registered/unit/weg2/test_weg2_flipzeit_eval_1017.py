# SPDX-License-Identifier: Apache-2.0
"""flipzeit_eval (h16u 1017, SOLL R10): p50/p90/max per direction from WEG2-FLIP-USERZEIT, hold share from DP-WAIT,
flips per hour, double flips (same direction < 30 s), the T_halb recommendation and its 30 % deviation edge."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "devtools", "flipzeit_eval.py"))
REAL = ("/spinning/docker-acceptance/nf/evidence/boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10161721_"
        "b8cf6c1fa7_1016_172204.front.log")


@pytest.fixture(scope="module")
def E():
    assert os.path.exists(EVAL), "devtools/flipzeit_eval.py missing"
    spec = importlib.util.spec_from_file_location("flipzeit_eval_t", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def uz(hms, d, ms, date="2026-10-16"):
    return "[%s %s,000] INFO weg2.front: WEG2-FLIP-USERZEIT dir=%s epoch=1 user_flipzeit_ms=%s missing=None start=x" % (date, hms, d, ms)


def test_percentile_linear(E):
    assert E.percentile([], 50) is None
    assert E.percentile([4.0], 90) == 4.0
    assert E.percentile([1, 2, 3, 4, 10], 50) == 3.0
    assert E.percentile([1, 2, 3, 4, 10], 90) == pytest.approx(7.6)
    assert E.percentile([10, 1, 3, 2, 4], 100) == 10.0


def test_per_direction_stats_and_flipcycle_citation_not_counted(E):
    lines = [uz("10:00:00", "D>P", 1000), uz("10:00:10", "P>D", 2000), uz("10:01:00", "D>P", 3000),
             "[2026-10-16 10:01:01,000] INFO weg2.front: WEG2-FLIPCYCLE stage=total dir=D>P epoch=3 ms=9 "
             "(NOT the user flip time, see WEG2-FLIP-USERZEIT user_flipzeit_ms)"]
    r = E.evaluate(E.parse(lines))
    assert r["n_flips"] == 2 + 1
    assert r["dir"]["D>P"] == {"n": 2, "n_missing": 0, "p50_s": 2.0, "p90_s": pytest.approx(2.8), "max_s": 3.0}
    assert r["dir"]["P>D"]["p90_s"] == 2.0


def test_missing_value_counted_not_averaged(E):
    r = E.evaluate(E.parse([uz("10:00:00", "D>P", "None"), uz("10:05:00", "D>P", 4000)]))
    assert r["dir"]["D>P"]["n"] == 1 and r["dir"]["D>P"]["n_missing"] == 1 and r["dir"]["D>P"]["max_s"] == 4.0


def test_hold_share_from_dp_wait(E):
    lines = [uz("10:00:00", "D>P", 1000),
             "[2026-10-16 10:00:01,000] INFO weg2.front: WEG2 DP-WAIT rid=a epoch=1 wait_s=10.0 hold_s=6.0 drain_s=0.0 flip_s=4.0 dominant=hold",
             "[2026-10-16 10:00:01,000] INFO weg2.front: WEG2 DP-WAIT rid=b epoch=1 wait_s=5.0 hold_s=1.0 drain_s=0.0 flip_s=4.0 dominant=flip"]
    r = E.evaluate(E.parse(lines))
    assert r["dp_wait"]["n"] == 2 and r["dp_wait"]["hold_share"] == pytest.approx(7.0 / 15.0)
    assert E.evaluate(E.parse([uz("10:00:00", "D>P", 1000)]))["dp_wait"]["hold_share"] is None


def test_double_flip_window_edge(E):
    a = E.double_flips([(0.0, "D>P", 1.0), (29.9, "D>P", 1.0)])
    b = E.double_flips([(0.0, "D>P", 1.0), (30.0, "D>P", 1.0)])
    c = E.double_flips([(0.0, "D>P", 1.0), (10.0, "P>D", 1.0)])         # opposite directions: not a double flip
    assert len(a) == 1 and b == [] and c == []


def test_flips_per_hour_buckets(E):
    lines = [uz("10:00:00", "D>P", 1), uz("10:59:59", "P>D", 1), uz("11:00:00", "D>P", 1)]
    r = E.evaluate(E.parse(lines))
    assert r["per_hour"] == {"2026-10-16 10": 2, "2026-10-16 11": 1}
    assert r["flips_per_h_mean"] == pytest.approx(3.0)      # span 10:00:00 -> 11:00:00 = 1 h


@pytest.mark.parametrize("flag,verdict", [(4.8, "ok"), (6.24, "ok"), (6.25, "abweichung"), (3.36, "ok"), (3.35, "abweichung")])
def test_t_half_deviation_edge_is_30_percent_of_the_recommendation(E, flag, verdict):
    # p90 D>P = 6 (single value), p90 P>D = 3.6 -> recommendation 4.8 s; 30 % band = [3.36, 6.24]
    r = E.evaluate(E.parse([uz("10:00:00", "D>P", 6000), uz("10:00:50", "P>D", 3600)]), t_half_s=flag)
    assert r["t_half"]["recommend_s"] == pytest.approx(4.8)
    assert r["t_half"]["verdict"] == verdict


def test_no_data_is_no_verdict(E):
    r = E.evaluate(E.parse([]))
    assert r["t_half"]["verdict"] == "keine-daten" and r["flips_per_h_mean"] is None


def test_selftest_runs_green(E, capsys):
    assert E.main(["--selftest"]) == 0
    assert "SELFTEST GRUEN" in capsys.readouterr().out


@pytest.mark.skipif(not os.path.exists(REAL), reason="evidence log absent")
def test_real_log_of_16_10_golden(E):
    with open(REAL, errors="replace") as f:
        r = E.evaluate(E.parse(f))
    assert r["dir"]["D>P"]["n"] == 90 and r["dir"]["P>D"]["n"] == 90
    assert r["t_half"]["recommend_s"] == pytest.approx(4.47, abs=0.02)
    assert r["dir"]["D>P"]["p90_s"] == pytest.approx(5.61, abs=0.02)
    assert r["dir"]["P>D"]["p90_s"] == pytest.approx(3.32, abs=0.02)
