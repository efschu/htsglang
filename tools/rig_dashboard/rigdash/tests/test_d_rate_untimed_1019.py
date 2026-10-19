"""19.10. (rc18 D TP0, boot ..._7737b78948_1019_200330): the phase bar showed "D prefill/extend 2,544 tok/s" for the
D prefill at 20:49:17Z.  The log says 5420 tokens in 8011 ms (D-DECODE-GAP 20:49:21 d_prefill_ms=8011 extends=2
prefill_tokens=5420): 2688 tokens gpu-ms 3728.3 (line at 20:49:21) and 2732 tokens gpu-ms 3891.9 + a 109-token
MIXED-D extend (lines at 20:49:23).

Root cause: the rank's RankPrefillLog counts a MIXED-D forward (and a refused #691 pairing) in chunks / new_tokens
but neither in gpu_ms nor in ``last`` -- prefill.last stays the previous TIMED chunk.  activity.chunks read every
such sample off that record: the 15 MIXED-D extends of 20:49:23-20:50:59 (9760 tokens, all >= 64 = "wide") landed
on the interval of the 2732-token chunk, and the burst was 15180 tokens / 5.97 s = 2544 tok/s.

Second term: prefill.last.t is the moment the line was flushed (time.time()), 4.2 s / 2.2 s after the two
forwards -- the wall clock from those stamps is 5.97 s for 8.0 s of prefill.  One stage (D) is therefore rated at
compute time: tokens / the summed own durations of the timed chunks (5529 / 7.62 s = 726 tok/s).
RED on 19393ee2cb: the burst and the segment rate are 2544 tok/s and the extends are "wide".
"""
import unittest

from rigdash import activity, ipcboot

K = "D.tp0pp0"


def rec(ts, new, chunks, comp, gpu, last):
    return ipcboot.compact({"ts": ts, "prefill": {"new_tokens": new, "chunks": chunks, "compute_ms": comp,
                                                  "gpu_ms": gpu, "last": last}})


def metal_ring():
    """Counters as the rank wrote them (cumulative), last = the newest TIMED chunk only."""
    rows = []
    new, ch, comp, gpu = 1000.0, 10.0, 5000.0, 9000.0
    last = {"t": 90.0, "gpu_ms": 1526.7, "own_ms": 1526.7, "new": 811}
    rows.append(rec(100.0, new, ch, comp, gpu, last))
    # 20:49:21 -- 2688 tokens, gpu-ms 3728.3 (compute 983.6), line flushed at 121.30
    new, ch, comp, gpu = new + 2688, ch + 1, comp + 983.6, gpu + 3728.3
    last = {"t": 121.30, "gpu_ms": 3728.3, "own_ms": 3728.3, "new": 2688}
    rows.append(rec(121.5, new, ch, comp, gpu, last))
    # 20:49:23 -- 2732 tokens gpu-ms 3891.9 (compute 1058.3) + the 109-token MIXED-D extend (untimed), same sample
    new, ch, comp, gpu = new + 2732 + 109, ch + 2, comp + 1058.3, gpu + 3891.9
    last = {"t": 123.54, "gpu_ms": 3891.9, "own_ms": 3891.9, "new": 2732}
    rows.append(rec(123.8, new, ch, comp, gpu, last))
    # the MIXED-D extends after it: counted, never timed, prefill.last unchanged
    for ts, toks in ((132.5, (405, 561)), (138.5, (931,)), (159.5, (242,)), (162.5, (193,)), (169.5, (2470,)),
                     (170.5, (197,)), (186.5, (209,)), (199.5, (1338, 2139)), (204.5, (192,)), (210.5, (427,)),
                     (217.5, (131,)), (219.5, (216,))):
        new, ch = new + sum(toks), ch + len(toks)
        rows.append(rec(ts, new, ch, comp, gpu, last))
    return [{"t": r["ts"], "r": {K: r}} for r in rows]


class UntimedChunks(unittest.TestCase):
    def test_metal_untimed_extends_add_9651_tokens(self):
        # 9760 tokens of 15 MIXED-D extends; the 109 of 20:49:23 ride in the timed sample of the 2732-token chunk
        ring = metal_ring()
        un = [c for c in activity.chunks(ring, [K], "D") if not c.get("timed", True)]
        self.assertEqual(sum(c["tok"] for c in un), 9760 - 109)

    def test_untimed_chunks_sit_at_their_own_rank_clocks_not_on_the_old_chunk(self):
        ring = metal_ring()
        un = [c for c in activity.chunks(ring, [K], "D") if not c.get("timed", True)]
        first = min(un, key=lambda c: c["s"])
        self.assertEqual((first["s"], first["e0"]), (123.8, 132.5))     # between the two rank clocks
        self.assertTrue(all(c["s"] >= 123.8 for c in un))               # never on [119.65, 123.54]

    def test_untimed_extends_are_not_wide_and_count_as_admit_extends(self):
        ring = metal_ring()
        wide = activity.chunks(ring, [K], "D", "wide")
        self.assertEqual(sum(c["tok"] for c in wide), 2688 + 2732 + 109)
        admit = activity.chunks(ring, [K], "D", "admit")
        self.assertEqual(sum(c["tok"] for c in admit), 9760 - 109)

    def test_metal_d_burst_rate_is_the_compute_time_rate(self):
        ring = metal_ring()
        bs = activity.bursts(activity.chunks(ring, [K], "D", "wide"))
        b = max(bs, key=lambda x: x["tok"])
        self.assertEqual(b["tok_timed"], 5529)
        self.assertAlmostEqual(b["rate"], 5529 / (3.7283 + 3.8919), places=1)     # 725.6, not 2544
        self.assertLess(b["rate"], 1000.0)

    def test_the_whole_d_burst_with_untimed_never_rates_them(self):
        ring = metal_ring()
        bs = activity.bursts(activity.chunks(ring, [K], "D"))
        for b in bs:
            if b["rate"] is not None:
                self.assertLess(b["rate"], 1000.0)
        self.assertTrue(all(b["rate"] is None for b in bs if all(not c.get("timed", True) for c in b["cs"])))

    def test_segment_rate_over_the_burst(self):
        ring = metal_ring()
        wide = activity.chunks(ring, [K], "D", "wide")
        r = activity.compute_rate(wide, 117.4, 123.6)
        self.assertAlmostEqual(r, 5529 / (3.7283 + 3.8919), places=1)

    def test_rank_gpu_rate_skips_untimed_pairs(self):
        ring = metal_ring()
        rates = ipcboot._rank_gpu_rates(ring, [K], "D", 100.0, 230.0, True)
        # Δnew / Δcompute over the timed pairs only: (2688 + 2841) / 2.0419 s; before, the untimed 9760 tokens
        # entered with 0 ms
        self.assertAlmostEqual(rates["tp0pp0"]["tps"], (2688 + 2841) / ((983.6 + 1058.3) / 1000.0), places=0)

    def test_is_untimed(self):
        a = rec(1.0, 10, 1, 5.0, 9.0, {"t": 0.5, "gpu_ms": 9.0})
        self.assertTrue(activity.is_untimed(a, rec(2.0, 300, 2, 5.0, 9.0, {"t": 0.5, "gpu_ms": 9.0})))
        self.assertFalse(activity.is_untimed(a, rec(2.0, 300, 2, 8.0, 19.0, {"t": 1.9, "gpu_ms": 10.0})))
        # older rankstats without gpu_ms decide nothing
        old_a = ipcboot.compact({"ts": 1.0, "prefill": {"new_tokens": 10, "chunks": 1, "last": {"t": 0.5}}})
        old_b = ipcboot.compact({"ts": 2.0, "prefill": {"new_tokens": 300, "chunks": 2, "last": {"t": 0.5}}})
        self.assertFalse(activity.is_untimed(old_a, old_b))


if __name__ == "__main__":
    unittest.main()
