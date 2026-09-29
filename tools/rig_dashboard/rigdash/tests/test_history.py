"""Persistent history (DASHBOARD-GRAFIKEN): compaction 1 s -> 10 s -> 60 s, retention, the size
cap, queries across tiers, late backfill, the decode spread and the card roles."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import history  # noqa: E402

T0 = 1790600000        # a multiple of 60? no: use aligned values below


def aligned(t, s=60):
    return (t // s) * s


class TestCompaction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = history.HistoryDB(os.path.join(self.tmp.name, "h.sqlite"))
        self.t0 = aligned(T0)

    def tearDown(self):
        self.db.db.close()
        self.tmp.cleanup()

    def test_means_per_tier(self):
        # 1-s samples 0..119 over two minutes: value = second index
        self.db.put([("x", self.t0 + i, float(i)) for i in range(120)])
        now = self.t0 + 120 + history.MODEL_LAG_S + 2 * history.MODEL_BUCKET_S + 10 + 60 + 1
        self.db.compact(now)
        p1 = dict(self.db.db.execute("SELECT ts, v FROM p1").fetchall())
        self.assertEqual(p1[self.t0], 4.5)                  # mean of 0..9
        self.assertEqual(p1[self.t0 + 110], 114.5)
        p2 = dict(self.db.db.execute("SELECT ts, v FROM p2").fetchall())
        self.assertEqual(p2[self.t0], 29.5)                 # mean of 0..59 (= mean of the six 10-s means)
        self.assertEqual(p2[self.t0 + 60], 89.5)
        # a second compaction does not double anything
        self.db.compact(now + 5)
        self.assertEqual(self.db.db.execute("SELECT COUNT(*) FROM p2").fetchone()[0], 2)

    def test_retention_and_query_across_tiers(self):
        self.db.put([("x", self.t0 + i, 1.0) for i in range(600)])
        now = self.t0 + 600 + 200
        self.db.compact(now)
        # p0 expires; the query must still answer from p1 / p2
        later = self.t0 + 3 * 3600 + 900
        self.db.compact(later)
        self.assertEqual(self.db.db.execute("SELECT COUNT(*) FROM p0").fetchone()[0], 0)
        got = self.db.query(["x"], self.t0, self.t0 + 600, 60, now=later)["x"]
        self.assertEqual(len(got), 10)
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in got.values()))
        got5 = self.db.query(["x"], self.t0, self.t0 + 600, 5, now=later)["x"]   # p0 gone -> p1 (10 s)
        self.assertEqual(len(got5), 60)

    def test_fresh_part_from_the_finer_tier(self):
        self.db.put([("x", self.t0 + i, 2.0) for i in range(300)])
        self.db.compact(self.t0 + 150)       # p1 cursor somewhere inside
        got = self.db.query(["x"], self.t0, self.t0 + 300, 10, now=self.t0 + 300)["x"]
        self.assertEqual(len(got), 30)       # compacted part from p1, the rest from p0

    def test_late_backfill_rebucket(self):
        self.db.put([("gpu", self.t0 + i, 5.0) for i in range(600)])
        self.db.compact(self.t0 + 600 + 200)
        # a boot's backfill arrives after the tiers passed it
        self.db.put([("m", self.t0 + 5 * i, 3.0) for i in range(60)])
        self.db.rebucket(["m"], self.t0, self.t0 + 300)
        got = self.db.query(["m", "gpu"], self.t0, self.t0 + 300, 60, now=self.t0 + 900)
        self.assertEqual(sorted(got["m"].values()), [3.0] * 5)
        self.assertEqual(sorted(got["gpu"].values()), [5.0] * 5)

    def test_size_cap(self):
        db = history.HistoryDB(os.path.join(self.tmp.name, "cap.sqlite"), max_mb=0)
        db.db.executemany("INSERT INTO p2(sid, ts, v) VALUES (1, ?, 1.0)",
                          [(self.t0 + 60 * i,) for i in range(3000)])
        db.compact(self.t0 + 3000 * 60)
        n = db.db.execute("SELECT COUNT(*) FROM p2").fetchone()[0]
        self.assertLess(n, 3000)            # the oldest days went
        db.db.close()

    def test_marks(self):
        self.db.mark(self.t0 + 1, "27B", "flip", "P>D", 2588)
        self.db.mark(self.t0 + 1, "27B", "flip", "P>D", 2588)       # idempotent
        self.db.mark(self.t0 + 2, "NF", "boot", "x")
        self.assertEqual(len(self.db.marks("27B", self.t0, self.t0 + 10)), 1)


class TestBuckets(unittest.TestCase):
    def test_decode_spread_time_weighted(self):
        lines = [{"t": 100.0, "gen_tps": None, "running": 2},
                 {"t": 110.0, "gen_tps": 100.0, "running": 2},
                 {"t": 200.0, "gen_tps": 50.0, "running": 1}]   # gap 90 s > DEC_GAP_MAX_S: not spread
        tok, cov, stream = history.decode_buckets(lines, 100.0, 4, 5.0)
        self.assertEqual(tok[:2], [500.0, 500.0])
        self.assertEqual(cov[:2], [5.0, 5.0])
        self.assertEqual(stream[:2], [50.0, 50.0])
        self.assertEqual(tok[2:], [0.0, 0.0])
        self.assertIsNone(stream[2])

    def test_decode_artefact_not_spread(self):
        lines = [{"t": 100.0}, {"t": 105.0, "gen_tps": 999.0, "gen_art": True, "running": 1}]
        tok, _, _ = history.decode_buckets(lines, 100.0, 1, 5.0)
        self.assertEqual(tok, [0.0])

    def test_prefill_first_rank_only(self):
        ev = [{"t": 101.0, "rank": 0, "new_tok": 4096}, {"t": 101.0, "rank": 1, "new_tok": 4096},
              {"t": 106.0, "new_tok": 100}]
        self.assertEqual(history.prefill_tokens(ev, 100.0, 2, 5.0), [4096.0, 100.0])

    def test_card_roles_from_launch(self):
        cards = [{"index": 0, "uuid": "GPU-a3080"}, {"index": 1, "uuid": "GPU-b5090"}, {"index": 2, "uuid": "GPU-c3080"}]
        vis = "GPU-b5090,GPU-a3080,GPU-c3080"
        launch = {"P": {"argv": ["--tp-size", "1", "--pp-size", "3", "--rank-gpu-id", "0,1,2"],
                        "env": {"CUDA_VISIBLE_DEVICES": vis}},
                  "D": {"argv": ["--tp-size", "3", "--pp-size", "1", "--rank-gpu-id", "0,1,2"],
                        "env": {"CUDA_VISIBLE_DEVICES": vis}}}
        r = history.card_roles(launch, cards)
        self.assertEqual(r["GPU-b5090"], ["D-TP0", "P-PP0"])
        self.assertEqual(r["GPU-a3080"], ["D-TP1", "P-PP1"])

    def test_parse_host(self):
        txt = ("cpu  100 0 100 700 100 0 0 0 0 0\nMemTotal:       131801264 kB\n"
               "MemAvailable:   103070916 kB\ncg 1073741824\ncg 1073741824\n")
        v = history.parse_host(txt)
        self.assertEqual(v["cpu"], (1000, 800))
        self.assertEqual(v["cg"], 2 * 1073741824)
        self.assertEqual(v["MemTotal"], 131801264)


if __name__ == "__main__":
    unittest.main()
