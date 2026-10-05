"""Nutzer 02.10.: "zwischen D decode ende 9:16:32 und D prefill start 9:16:37 steht das rig 5 s still".

Measured (NF y6x, D TP0, 07:16Z): two prefill chunks between two rankstats samples -- 2944 + 475 tokens,
gpu_ms 4766 + 1961, compute only 1035 + 289 (the rest is expert H2D streaming and DCP collectives).
The D prefill span must reach back over the summed gpu_ms, not only the summed compute.
"""
import unittest

from rigdash import activity, ipcboot


def rec(ts, new, chunks, comp, gpu, last_t, last_gpu):
    return ipcboot.compact({"ts": ts, "prefill": {"new_tokens": new, "chunks": chunks, "compute_ms": comp,
                                                  "gpu_ms": gpu, "last": {"t": last_t, "gpu_ms": last_gpu}}})


class DPrefillSpan(unittest.TestCase):
    def test_two_chunks_span_their_gpu_time(self):
        a = rec(100.0, 0, 0, 0.0, 0.0, None, None)
        b = rec(107.0, 2944 + 475, 2, 1035.0 + 289.0, 4766.0 + 1961.0, 106.8, 1961.0)
        ring = [{"t": 100.0, "r": {"D.tp0pp0": a}}, {"t": 107.0, "r": {"D.tp0pp0": b}}]
        cs = activity.chunks(ring, ["D.tp0pp0"], "D")
        self.assertEqual(len(cs), 1)
        start = cs[0]["s"]
        # 106.8 - 6.727 s = 100.07: the prefill started right after the decode ended, no 5-s hole
        self.assertAlmostEqual(start, 106.8 - (4766.0 + 1961.0) / 1000.0, places=3)

    def test_pipeline_stage_keeps_compute_rule(self):
        # P (several stages): a chunk's wait is queueing behind the previous one -- unchanged
        a = rec(100.0, 0, 0, 0.0, 0.0, None, None)
        b = rec(107.0, 32768, 2, 4000.0, 9000.0, 106.8, 2000.0)
        ring = [{"t": 100.0, "r": {"P.tp0pp0": a, "P.tp0pp2": a}},
                {"t": 107.0, "r": {"P.tp0pp0": b, "P.tp0pp2": b}}]
        cs = activity.chunks(ring, ["P.tp0pp0", "P.tp0pp2"], "P")
        self.assertAlmostEqual(cs[0]["s"], 106.8 - 4.0, places=3)


if __name__ == "__main__":
    unittest.main()
