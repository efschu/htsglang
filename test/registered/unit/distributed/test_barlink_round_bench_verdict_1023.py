"""barlink-switch-1023: a contended or non-linear bench run is not a
calibration.

REPRO: the 3-card runs of 23.10. (09:14Z host detour, 09:21Z switch-direct)
were taken with card 2 under the user's desktop load and printed the line
that hands their fit (1320 us / 9.8 GB/s, 1816 us / 172 GB/s) to
``barlink_bar1.DEFAULT_*``. The rows below are those artifacts' rows,
verbatim; the switch-direct 2-card run of 09:10Z is the accepted control.
CPU-only: nvidia-smi is faked.
"""

import importlib.util
import os
import unittest

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_HERE, "..", "..", "..", "..", "scripts", "pdflip",
                      "barlink_round_bench.py")


def _bench():
    spec = importlib.util.spec_from_file_location("barlink_round_bench_t", _BENCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(cells):
    return [{"window_mib": w, "nbytes": n, "rounds": r, "wire_bytes": wb,
             "bar1_ms": ms, "correct": True} for w, n, r, wb, ms in cells]


#: p2p3sw_rebar_1023_0921.B_switch_direct.json (3 cards, switch-direct)
THREE_CARD_SWITCH = _rows([
    (16, 25165824, 7, 33554432, 11.968755),
    (16, 100663296, 25, 134217728, 47.485168),
    (40, 25165824, 3, 33554432, 3.809215),
    (40, 100663296, 10, 134217728, 15.932719),
    (128, 25165824, 1, 33554432, 2.233924),
    (128, 100663296, 4, 134217728, 10.383668),
])
#: p2p3_rebar_1023_0914.A_host_detour.json (3 cards, host detour)
THREE_CARD_HOST = _rows([
    (16, 25165824, 7, 33554432, 12.055185),
    (16, 100663296, 25, 134217728, 47.481315),
    (40, 25165824, 3, 33554432, 6.656308),
    (40, 100663296, 10, 134217728, 24.788736),
    (128, 25165824, 1, 33554432, 4.807893),
    (128, 100663296, 4, 134217728, 20.478814),
])
#: p2p_rebar_1023_0909.B_switch_direct.json (2 cards, the shipped one)
TWO_CARD_SWITCH = _rows([
    (16, 25165824, 5, 25165824, 2.0489715),
    (16, 100663296, 19, 100663296, 7.853802),
    (40, 25165824, 2, 25165824, 1.605772),
    (40, 100663296, 8, 100663296, 6.1887635),
    (128, 25165824, 1, 25165824, 1.370169),
    (128, 100663296, 3, 100663296, 5.365819),
])


class _Run:
    def __init__(self, lines):
        self.lines, self.calls = list(lines), 0

    def __call__(self, *a, **k):
        txt = self.lines[min(self.calls, len(self.lines) - 1)]
        self.calls += 1
        return type("R", (), {"stdout": txt})()


class TestFitQuality(CustomTestCase):
    def setUp(self):
        self.b = _bench()

    def test_repro_both_three_card_runs_are_refused_as_nonlinear(self):
        for rows in (THREE_CARD_SWITCH, THREE_CARD_HOST):
            fit = self.b._least_squares(rows)
            why = self.b.calibration_verdict(rows, fit, [])
            self.assertTrue(why.startswith("NONLINEAR"), why)

    def test_the_two_card_switch_run_is_accepted_and_is_the_shipped_one(self):
        fit = self.b._least_squares(TWO_CARD_SWITCH)
        self.assertEqual(self.b.calibration_verdict(TWO_CARD_SWITCH, fit, []), "")
        self.assertAlmostEqual(fit[0], 155.5, delta=0.1)
        self.assertAlmostEqual(fit[1], 20.45, delta=0.01)

    def test_contended_beats_a_clean_fit(self):
        fit = self.b._least_squares(TWO_CARD_SWITCH)
        why = self.b.calibration_verdict(TWO_CARD_SWITCH, fit, ["card 2: 37 %"])
        self.assertTrue(why.startswith("CONTENDED"))

    def test_a_wrong_cell_is_no_calibration(self):
        rows = [dict(r) for r in TWO_CARD_SWITCH]
        rows[1]["correct"] = False
        fit = self.b._least_squares(rows)
        self.assertIn("WRONG", self.b.calibration_verdict(rows, fit, []))


class TestContention(CustomTestCase):
    def setUp(self):
        self.b = _bench()

    def test_repro_card2_under_the_desktop_load_is_contended(self):
        # 23.10. 09:2xZ: cards 0/1 idle, card 2 37 % / 228 W / 1779 MiB
        run = _Run(["0, 0, 18.4, 18\n1, 0, 17.5, 1\n2, 37, 228.1, 1779\n"])
        snap = self.b.contention_sample([0, 1, 2], samples=2, gap_s=0.0, run=run)
        why = self.b.contention_verdict(snap)
        self.assertEqual(len(why), 1)
        self.assertIn("card 2", why[0])
        self.assertIn("37 %", why[0])

    def test_burst_in_one_sample_counts(self):
        run = _Run(["0, 0, 20, 18\n", "0, 41, 95, 18\n", "0, 0, 20, 18\n"])
        snap = self.b.contention_sample([0], samples=3, gap_s=0.0, run=run)
        self.assertEqual(snap[0]["util_max"], 41.0)
        self.assertTrue(self.b.contention_verdict(snap))

    def test_idle_cards_are_clean_and_nvidia_smi_failure_is_not_fatal(self):
        run = _Run(["0, 0, 18, 18\n1, 3, 17, 1\n"])
        snap = self.b.contention_sample([0, 1], samples=1, run=run)
        self.assertEqual(self.b.contention_verdict(snap), [])

        def boom(*a, **k):
            raise OSError("no nvidia-smi")
        snap = self.b.contention_sample([0], samples=2, gap_s=0.0, run=boom)
        self.assertIn("error", snap[0])
        self.assertEqual(self.b.contention_verdict(snap), [])

    def test_json_roundtrip_keys_are_strings(self):
        """The pre-snapshot crosses into the ranks as JSON: int keys come
        back as strings and must still be graded."""
        import json
        snap = {2: {"util_max": 37.0, "power_max": 228.0, "mem_used_mib": 1779}}
        back = json.loads(json.dumps(snap))
        self.assertTrue(self.b.contention_verdict(back))


if __name__ == "__main__":
    unittest.main()
