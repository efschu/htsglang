# SPDX-License-Identifier: Apache-2.0
"""scripts/dual_layout/eval_dual_boot.py overlap: D rounds are placed by their
``t:`` field, not by the log line's stamp.

DANGER DIRECTION guarded here: the D log flushes 'Decode rank batch' lines in
bursts (y6e boot dual1i ...10010932: 6453 rounds over 457 s of ``t:`` landed in
109 log seconds, lag p50 1.4 s / p99 19 s), so bucketing by the line stamp
reported "5 s with BOTH" for ~55 s of real overlap and filed the slow rounds
under "P idle". P activity comes from the per-rank chunk windows
(PDFLIP-VRAM-PEAK t_unix_ms minus 'Prefill rank batch' gpu-ms).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "..", "..", "..", "scripts", "dual_layout", "eval_dual_boot.py")


def _load():
    spec = importlib.util.spec_from_file_location("eval_dual_boot", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T0 = 1790847400.0  # epoch seconds


def _stamp(t):
    import datetime

    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _write_logs(d):
    """P: 10 chunks of 1024 tokens, each 0.5 s on every rank, back to back from
    T0+10 to T0+15. D: one round every 0.1 s from T0 to T0+30 (bs1); rounds
    inside the P window take 200 ms, outside 30 ms. Every D line is STAMPED
    with the same late second (T0+40) -- the bundled flush."""
    p = os.path.join(d, "boot_weg2_x.P.log")
    dl = os.path.join(d, "boot_weg2_x.D.log")
    with open(p, "w") as f:
        for i in range(10):
            end = T0 + 10 + 0.5 * (i + 1)
            for r in range(3):
                f.write(f"[{_stamp(end)} PP{r}] Prefill rank batch, #new-token: 1024, #cached-token: 0, "
                        f"#chunks: 1, gpu-ms: 500.0\n")
                f.write(f"[{_stamp(end)} PP{r}] PDFLIP-VRAM-PEAK rank={r} phase=chunk rows=1024 n=1 "
                        f"t0_unix_ms=na t_unix_ms={int(end * 1000)} window_ms=na\n")
            f.write(f"[{_stamp(end)} PP0] Prefill batch, #new-seq: 1, #new-token: 1024, #cached-token: 0, "
                    f"#pending-token: 0\n")
    with open(dl, "w") as f:
        for k in range(300):
            t = T0 + 0.1 * k
            g = 200.0 if T0 + 10 <= t <= T0 + 15 else 30.0
            f.write(f"[{_stamp(T0 + 40)} TP0] Decode rank batch, rank: 0, #round: {k}, t: {t:.3f}, bs: 1, "
                    f"#rows: 8, #fwd: 2, gpu-ms: {g} (split unavailable)\n")
    open(os.path.join(d, "boot_weg2_x.front.log"), "w").close()
    return p, dl


class EvalDualBootT(CustomTestCase):
    def test_bundled_d_lines_are_placed_by_t(self):
        m = _load()
        with tempfile.TemporaryDirectory() as d:
            p, dl = _write_logs(d)
            o = m.overlap(p, dl)
        # ~5 s of P activity, all of it covered by D rounds
        self.assertGreaterEqual(o["both_s"], 4.5)
        self.assertLessEqual(o["both_s"], 5.6)
        self.assertEqual(o["d_source"], "t-field")
        self.assertEqual(o["p_source"], "chunk-windows")
        # the slow rounds are the ones during P, the fast ones are P idle
        during, idle = sorted(o["during"]), sorted(o["idle"])
        self.assertGreaterEqual(len(during), 45)
        self.assertEqual(during[len(during) // 2], 200.0)
        self.assertEqual(idle[len(idle) // 2], 30.0)
        self.assertNotIn(200.0, idle)

    def test_p_falls_back_to_batch_lines_without_chunk_windows(self):
        m = _load()
        with tempfile.TemporaryDirectory() as d:
            p, dl = _write_logs(d)
            with open(p) as f:
                keep = [x for x in f if "VRAM-PEAK" not in x]
            with open(p, "w") as f:
                f.writelines(keep)
            o = m.overlap(p, dl)
        self.assertEqual(o["p_source"], "batch-line-seconds")
        self.assertGreater(o["both_s"], 0)


class EvalDualBootPairByTime(CustomTestCase):
    """ITEM 200 part 2: VRAM-PEAK ends and 'Prefill rank batch' lines are paired
    by time (last unused batch with stamp <= peak end, within a max gap), not by
    zip(); unmatched peaks are dropped and counted."""

    def test_pairing_function(self):
        m = _load()
        pairs, dropped = m.pair_peaks_with_batches(
            [10.5, 20.5, 30.5], [(10.0, 1.0), (20.0, 2.0), (30.0, 3.0)])
        self.assertEqual(pairs, [(10.5, 1.0), (20.5, 2.0), (30.5, 3.0)])
        self.assertEqual(dropped, 0)

    def test_missing_batch_line_does_not_shift_later_pairs(self):
        m = _load()
        # the batch line of the 2nd chunk is missing: zip() would pair peak 2
        # with batch 3 and peak 3 with nothing
        pairs, dropped = m.pair_peaks_with_batches(
            [10.5, 40.5, 70.5], [(10.0, 1.0), (70.0, 3.0)], max_gap=5.0)
        self.assertEqual(pairs, [(10.5, 1.0), (70.5, 3.0)])
        self.assertEqual(dropped, 1)

    def test_batch_is_used_once_and_stamp_must_not_be_after_the_peak(self):
        m = _load()
        pairs, dropped = m.pair_peaks_with_batches([10.5, 11.5], [(10.0, 1.0)])
        self.assertEqual(pairs, [(10.5, 1.0)])
        self.assertEqual(dropped, 1)
        pairs, dropped = m.pair_peaks_with_batches([10.5], [(12.0, 9.0)])  # stamped after the peak
        self.assertEqual((pairs, dropped), ([], 1))

    def test_last_unused_batch_wins(self):
        m = _load()
        pairs, dropped = m.pair_peaks_with_batches([10.5], [(8.0, 1.0), (10.0, 2.0)])
        self.assertEqual(pairs, [(10.5, 2.0)])
        self.assertEqual(dropped, 0)

    def test_max_gap(self):
        m = _load()
        self.assertEqual(m.pair_peaks_with_batches([100.0], [(10.0, 1.0)], max_gap=10.0), ([], 1))
        self.assertEqual(m.pair_peaks_with_batches([100.0], [(95.0, 1.0)], max_gap=10.0), ([(100.0, 1.0)], 0))

    def test_overlap_drops_and_reports_unmatched_peaks(self):
        m = _load()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "boot_weg2_x.P.log")
            dl = os.path.join(d, "boot_weg2_x.D.log")
            open(dl, "w").close()
            with open(p, "w") as f:
                for i in range(5):  # one 1.0 s chunk every 3 s, PP0 only
                    end = T0 + 10 + 3 * i + 0.5
                    f.write(f"[{_stamp(end)} PP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=1024 n=1 "
                            f"t0_unix_ms=na t_unix_ms={int(end * 1000)} window_ms=na\n")
                    if i != 2:  # chunk 3's batch line is missing
                        f.write(f"[{_stamp(end)} PP0] Prefill rank batch, #new-token: 1024, #cached-token: 0, "
                                f"#chunks: 1, gpu-ms: 1000.0\n")
            o = m.overlap(p, dl)
        self.assertEqual((o["peaks_paired"], o["peaks_dropped"]), (4, 1))
        # 4 one-second windows; zip() would have paired peak 3 with batch 4
        # and dropped the last peak's pairing silently (4 windows, wrong ends)
        self.assertAlmostEqual(o["p_s"], 4.0, delta=0.01)
