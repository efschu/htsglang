# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H34b -- die Mindest-Verweildauer (K7) nicht aus dem ersten Boot-Flip.

DER BEFUND (fnFL2x148, 1d95257feb, 24.09. 11:51Z): der erste D->P-Flip nach
dem Boot dauerte 24,6 s -- er registriert als einziger die Host-Staging-Puffer
der On-card-Lanes (``WEG2-SEQ persist ... new/grow ... register_ms=7324 /
7820 / 7867 / 5918``); alle folgenden Flips 1,8-2,3 s. Die Front leitete den
Dwell aus diesem Flip ab (``derived_from_flip_ms=24598 provenance=last-flip-
D->P verdict=hold``, 87 Zeilen) und hielt den naechsten D->P-Flip 17,5 s
(11:58:03,815 -> 11:58:21,321), obwohl D schon 7,1 s wach war.

``fixtures/min_dwell_h34b/fnFL2x148.front.lines``: die WEG2-FLIP-done- und
MIN-DWELL-Zeilen des Front-Logs, woertlich, hinter ``flip_total`` gekuerzt.
"""

import os
import re
import time

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, warm_min_dwell_ms
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "min_dwell_h34b", "fnFL2x148.front.lines")
_RX_DONE = re.compile(r"\[([\d-]+ [\d:,]+)\] INFO weg2\.front: WEG2-FLIP done epoch=(\d+) "
                      r"slept=([DP]) woke=([DP]) flip_total=(\d+) ms")
_RX_DWELL = re.compile(r"\[([\d-]+ [\d:,]+)\] INFO weg2\.front: WEG2 MIN-DWELL src=([DP]) dst=([DP]) "
                       r"awake_ms=(\d+) derived_from_flip_ms=(\d+) overridden_by=\S+ "
                       r"provenance=(\S+) verdict=(\w+)")


def _x148():
    with open(FIX) as fh:
        text = fh.read()
    flips = [{"epoch": int(m.group(2)), "sleep": m.group(3), "wake": m.group(4),
              "flip_ms": int(m.group(5)), "ts": m.group(1)} for m in _RX_DONE.finditer(text)]
    dwells = [m.groups() for m in _RX_DWELL.finditer(text)]
    return flips, dwells


def _front():
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)


def _log_before(flips, ts):
    return [f for f in flips if f["ts"] < ts]


def test_the_fixture_is_the_x148_finding():
    flips, dwells = _x148()
    assert [(f["sleep"], f["wake"], f["flip_ms"]) for f in flips] == [
        ("D", "P", 24598), ("P", "D", 2296), ("D", "P", 1865), ("P", "D", 1835),
        ("D", "P", 1964), ("P", "D", 1804), ("D", "P", 1950), ("P", "D", 1819)]
    hold = [d for d in dwells if d[6] == "hold"]
    assert hold and hold[0][0] == "2024-09-24 11:58:03,815".replace("2024", "2026")
    assert hold[0][3:6] == ("7108", "24598", "last-flip-D->P")


def test_the_held_decision_flips_with_the_warm_price():
    """11:58:03,815: D 7,1 s wach, im Log der erste D->P (24,6 s, kalt) und
    ein P->D (2,3 s). Neu: Dwell 2296 ms aus dem warmen Flip -> flip."""
    flips, dwells = _x148()
    ts, _src, _dst, awake, *_ = [d for d in dwells if d[6] == "hold"][0]
    f = _front()
    f.flip_log.extend(_log_before(flips, ts))
    need, prov = f._derived_min_dwell_ms("D", "P")
    assert need == 2296.0
    assert prov.startswith("median-warm-any-direction:n=1") and " " not in prov
    assert "first-flip-D->P-24598ms-excluded" in prov
    f.t_awake = time.time() - int(awake) / 1000.0
    assert f._dwell_ok("D", "P", fairness_fired=False, work_exhausted=False,
                       oldest_wait_s=0.0) is True


def test_later_decisions_take_the_median_of_the_warm_same_direction_flips():
    flips, _ = _x148()
    f = _front()
    f.flip_log.extend(flips)
    need, prov = f._derived_min_dwell_ms("D", "P")
    assert need == 1950.0  # median(1865, 1964, 1950)
    assert prov.startswith("median-warm-D->P:n=3")
    need, prov = f._derived_min_dwell_ms("P", "D")
    assert need == 1827.0  # median(2296, 1835, 1804, 1819)
    assert prov.startswith("median-warm-P->D:n=4")
    # never the cold flip, whatever the window
    assert max(warm_min_dwell_ms(flips, "D", "P", window=50)[0],
               warm_min_dwell_ms(flips, "P", "D", window=50)[0]) < 3000


def test_the_window_lets_old_flips_roll_out():
    log = [{"sleep": "D", "wake": "P", "flip_ms": 24598}]
    log += [{"sleep": "D", "wake": "P", "flip_ms": ms} for ms in (9000, 9000, 1800, 1900, 2000)]
    assert warm_min_dwell_ms(log, "D", "P", window=3)[0] == 1900.0
    assert warm_min_dwell_ms(log, "D", "P", window=5)[0] == 2000.0


def test_before_any_later_flip_the_dwell_is_zero_as_before_the_first():
    assert warm_min_dwell_ms([], "D", "P") == (0.0, "none-first-flip")
    need, prov = warm_min_dwell_ms([{"sleep": "D", "wake": "P", "flip_ms": 24598}], "P", "D")
    assert need == 0.0 and prov.startswith("none-after-first-flip")
    f = _front()
    assert f._derived_min_dwell_ms("D", "P") == (0.0, "none-first-flip")


def _mutant_last_same_direction(log, src, dst):
    for rec in reversed(log):
        if rec["sleep"] == src and rec["wake"] == dst:
            return float(rec["flip_ms"])
    return 0.0


def _mutant_median_with_first(log, src, dst):
    same = [r["flip_ms"] for r in log if r["sleep"] == src and r["wake"] == dst][-5:]
    same.sort()
    return float(same[len(same) // 2]) if same else 0.0


@pytest.mark.parametrize("mutant", [_mutant_last_same_direction, _mutant_median_with_first])
def test_a_rule_that_keeps_the_first_flip_holds_x148_again(mutant):
    """Der Mutant (die alte Regel / Median MIT dem Erst-Flip) haelt die
    Entscheidung um 11:58:03 wieder 24,6 s -- die neue Regel nicht."""
    flips, dwells = _x148()
    ts, _s, _d, awake, *_ = [d for d in dwells if d[6] == "hold"][0]
    log = _log_before(flips, ts)
    assert mutant(log, "D", "P") > int(awake)          # mutant: hold
    assert warm_min_dwell_ms(log, "D", "P")[0] < int(awake)   # fix: flip


def test_the_switch_restores_the_old_rule():
    flips, _ = _x148()
    f = _front()
    f.flip_log.extend(flips[:2])
    with envs.SGLANG_WEG2_ENABLE_WARM_MIN_DWELL.override(False):
        assert f._derived_min_dwell_ms("D", "P") == (24598.0, "last-flip-D->P")
    assert f._derived_min_dwell_ms("D", "P")[0] == 2296.0
    g = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, min_dwell_ms=500.0)
    g.flip_log.extend(flips)
    assert g._derived_min_dwell_ms("D", "P") == (500.0, "flag")


def test_the_levers_are_registered():
    assert envs.SGLANG_WEG2_ENABLE_WARM_MIN_DWELL.get() is True
    assert envs.SGLANG_WEG2_MIN_DWELL_WINDOW.get() == 5
    assert front_mod.warm_min_dwell_ms is warm_min_dwell_ms
