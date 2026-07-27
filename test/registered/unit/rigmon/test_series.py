"""CPU unit tests for the rigmon time-series store.

No GPU, no NVML, no server: the store is pure Python over injected timestamps.
"""

import unittest

from sglang.srt.rigmon.series import (
    DEFAULT_TIERS,
    Aggregate,
    Point,
    TierSpec,
    TimeSeries,
    parse_duration,
    parse_tier_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


TIERS = (
    TierSpec("live", 1.0, 10.0),
    TierSpec("mid", 5.0, 60.0),
    TierSpec("slow", 20.0, 400.0),
)


class TestDurationParsing(CustomTestCase):
    def test_units(self):
        self.assertEqual(parse_duration("30"), 30.0)
        self.assertEqual(parse_duration("30s"), 30.0)
        self.assertEqual(parse_duration("15m"), 900.0)
        self.assertEqual(parse_duration("6h"), 21600.0)
        self.assertEqual(parse_duration("7d"), 604800.0)

    def test_tier_spec(self):
        t = parse_tier_spec("live:1s:10m")
        self.assertEqual((t.name, t.period_s, t.retain_s), ("live", 1.0, 600.0))
        self.assertEqual(t.capacity, 600)

    def test_tier_spec_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_tier_spec("live:1s")
        with self.assertRaises(ValueError):
            parse_tier_spec(":1s:10m")

    def test_retain_shorter_than_period_is_rejected(self):
        with self.assertRaises(ValueError):
            TierSpec("bad", 10.0, 1.0)


class TestAggregate(CustomTestCase):
    def test_min_mean_max_preserved(self):
        a = Aggregate.of(10.0)
        a.add(30.0)
        a.add(20.0)
        self.assertEqual((a.min, a.max, a.n), (10.0, 30.0, 3))
        self.assertAlmostEqual(a.mean, 20.0)

    def test_roundtrip_json(self):
        a = Aggregate.of(1.0)
        a.add(3.0)
        b = Aggregate.from_json(a.to_json())
        self.assertEqual((b.min, b.max, b.n), (1.0, 3.0, 2))
        self.assertAlmostEqual(b.mean, 2.0)


class TestTimeSeries(CustomTestCase):
    def test_ordering_of_tiers_enforced(self):
        with self.assertRaises(ValueError):
            TimeSeries([TierSpec("a", 10.0, 100.0), TierSpec("b", 1.0, 100.0)])

    def test_raw_writes_land_in_finest_tier(self):
        ts = TimeSeries(TIERS)
        for i in range(5):
            ts.add(1000.0 + i, {"x": float(i)})
        got = ts.query(resolution="live")
        self.assertEqual(got["resolution"], "live")
        self.assertFalse(got["aggregated"])
        self.assertEqual(len(got["points"]), 5)

    def test_cascade_downsamples(self):
        ts = TimeSeries(TIERS)
        for i in range(20):
            ts.add(1000.0 + i, {"x": float(i)})
        live = ts.query(resolution="live")["points"]
        mid = ts.query(resolution="mid")["points"]
        # live retains 10 s, so only the last ~10 buckets survive
        self.assertLessEqual(len(live), 10)
        # 20 samples at 1 s fold into 5 s buckets
        self.assertGreaterEqual(len(mid), 3)
        self.assertTrue(ts.query(resolution="mid")["aggregated"])

    def test_downsampling_preserves_peaks(self):
        """A spike must survive aggregation: a mean-only tier would hide
        exactly the thermal excursions this dashboard exists to show."""
        ts = TimeSeries(TIERS)
        for i in range(10):
            ts.add(1000.0 + i, {"temp": 60.0 if i != 3 else 88.0})
        ts.add(1010.0, {"temp": 60.0})  # close the buckets
        pts = [Point.from_json(p) for p in ts.query(resolution="mid")["points"]]
        peak = max(p.values["temp"].max for p in pts if "temp" in p.values)
        self.assertEqual(peak, 88.0)

    def test_capacity_is_enforced(self):
        ts = TimeSeries([TierSpec("live", 1.0, 5.0)])
        for i in range(50):
            ts.add(1000.0 + i, {"x": float(i)})
        self.assertLessEqual(len(ts.tiers[0]), 5)

    def test_none_and_non_numeric_ignored(self):
        ts = TimeSeries(TIERS)
        ts.add(1000.0, {"x": 1.0, "y": None, "z": "text", "b": True})
        pt = Point.from_json(ts.query(resolution="live")["points"][0])
        self.assertEqual(set(pt.values), {"x"})

    def test_latest_is_the_raw_sample(self):
        ts = TimeSeries(TIERS)
        ts.add(1000.0, {"x": 1.0})
        ts.add(1000.4, {"x": 9.0})
        got = ts.latest()
        self.assertIsNotNone(got)
        self.assertEqual(got[1]["x"], 9.0)

    def test_backwards_clock_does_not_corrupt_order(self):
        ts = TimeSeries(TIERS)
        for i in range(6):
            ts.add(1000.0 + i, {"x": 1.0})
        ts.add(1002.0, {"x": 5.0})  # a step backwards into an existing bucket
        pts = [Point.from_json(p) for p in ts.query(resolution="live")["points"]]
        self.assertEqual([p.ts for p in pts], sorted(p.ts for p in pts))
        bucket = [p for p in pts if p.ts == 1002.0][0]
        self.assertEqual(bucket.values["x"].n, 2)

    def test_pick_resolution(self):
        ts = TimeSeries(TIERS)
        self.assertEqual(ts.pick_resolution(5.0, max_points=600), "live")
        self.assertEqual(ts.pick_resolution(3000.0, max_points=100), "slow")
        # Beyond every tier: coarsest, never an exception.
        self.assertEqual(ts.pick_resolution(10**9, max_points=10), "slow")

    def test_query_unknown_resolution_names_the_configured_ones(self):
        ts = TimeSeries(TIERS)
        with self.assertRaises(KeyError) as cm:
            ts.query(resolution="nope")
        self.assertIn("live", str(cm.exception))

    def test_window_query_selects_resolution(self):
        ts = TimeSeries(TIERS)
        for i in range(40):
            ts.add(1000.0 + i, {"x": float(i)})
        got = ts.query(window_s=6.0, now=1040.0)
        self.assertEqual(got["resolution"], "live")

    def test_key_filter(self):
        ts = TimeSeries(TIERS)
        ts.add(1000.0, {"x": 1.0, "y": 2.0})
        got = ts.query(resolution="live", keys=["y"])
        self.assertEqual(list(got["points"][0]["v"]), ["y"])

    def test_configurable_resolutions_are_reported(self):
        ts = TimeSeries(TIERS)
        names = [r["name"] for r in ts.resolutions()]
        self.assertEqual(names, ["live", "mid", "slow"])
        self.assertEqual(ts.resolutions()[0]["period_s"], 1.0)

    def test_default_tiers_are_ordered(self):
        periods = [t.period_s for t in DEFAULT_TIERS]
        self.assertEqual(periods, sorted(periods))


class TestExportIngest(CustomTestCase):
    """The push protocol: incremental export, back-fill after an outage."""

    def test_export_is_incremental(self):
        ts = TimeSeries(TIERS)
        for i in range(6):
            ts.add(1000.0 + i, {"x": float(i)})
        first, cursors = ts.export_since(None)
        self.assertIn("live", first)
        again, cursors2 = ts.export_since(cursors)
        self.assertNotIn("live", again)
        ts.add(1010.0, {"x": 42.0})
        third, _ = ts.export_since(cursors2)
        self.assertIn("live", third)

    def test_ingest_reconstructs_on_the_far_side(self):
        src = TimeSeries(TIERS)
        for i in range(12):
            src.add(1000.0 + i, {"x": float(i)})
        dst = TimeSeries(TIERS)
        payload, _ = src.export_since(None)
        for tier, pts in payload.items():
            dst.ingest_points(tier, [Point.from_json(p) for p in pts])
        a = src.query(resolution="mid")["points"]
        b = dst.query(resolution="mid")["points"]
        self.assertEqual([p["t"] for p in a], [p["t"] for p in b])

    def test_ingest_unknown_tier_raises_named_error(self):
        dst = TimeSeries(TIERS)
        with self.assertRaises(KeyError):
            dst.ingest_points("weekly", [])

    def test_backfill_after_outage_does_not_duplicate(self):
        src = TimeSeries(TIERS)
        dst = TimeSeries(TIERS)
        for i in range(4):
            src.add(1000.0 + i, {"x": 1.0})
        payload, cursors = src.export_since(None)
        for tier, pts in payload.items():
            dst.ingest_points(tier, [Point.from_json(p) for p in pts])
        # Aggregator was down: cursors are NOT advanced, so the same range is
        # resent and must fold in, not stack up.
        for tier, pts in payload.items():
            dst.ingest_points(tier, [Point.from_json(p) for p in pts])
        self.assertEqual(len(dst.query(resolution="live")["points"]), 4)


if __name__ == "__main__":
    unittest.main()
