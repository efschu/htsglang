"""CPU unit tests for the collector loop and the multi-node aggregator.

No NVML, no engine, no socket: the device backend and the engine scraper are
injected, and the aggregator is exercised through its object API.
"""

import json
import unittest

from sglang.srt.rigmon.aggregator import Aggregator
from sglang.srt.rigmon.collector import Collector, PushClient, flatten_sample
from sglang.srt.rigmon.config import AggregatorConfig, CollectorConfig
from sglang.srt.rigmon.series import TierSpec
from sglang.srt.rigmon.sources import (
    CardSample,
    DeviceBackend,
    EngineSample,
    FieldStatus,
    GpuSampler,
    NullBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TIERS = (TierSpec("live", 1.0, 60.0), TierSpec("mid", 10.0, 600.0))


class FakeBackend(DeviceBackend):
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.profiling_calls = 0

    def fields(self):
        return [FieldStatus("power_w", "power", "W", True, "fake")]

    def sample(self, with_profiling=False):
        self.calls += 1
        if with_profiling:
            self.profiling_calls += 1
        return [
            CardSample(
                index=0, name="RTX 5090", uuid="GPU-5090",
                mem_total_mib=32607, mem_used_mib=20000,
                temp_c=61.0, power_w=310.0, sm_clock_mhz=2400,
                sm_clock_max_mhz=3090, util_gpu_pct=88.0, util_mem_pct=70.0,
                sm_active=0.88, dram_active=0.70,
                activity_source="nvml-utilization (coarse fallback)",
                energy_mj=1000 * self.calls,
            )
        ]


class FakeScraper:
    def __init__(self, tokens_start=0.0, up=True):
        self.n = 0
        self.up = up
        self.tokens = tokens_start

    def scrape(self):
        if not self.up:
            return EngineSample(up=False, reason="connection refused")
        self.n += 1
        self.tokens += 50.0
        return EngineSample(
            up=True,
            metrics={
                "generation_tokens_total": self.tokens,
                "num_running_reqs": 1.0,
                "gen_throughput": 47.0,
            },
            info={"tp_size": 1, "rank_gpu_id": [0], "dcp_size": 1},
        )


class FakeClock:
    def __init__(self, start=1000.0, step=1.0):
        self.t = start
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class TestFlatten(CustomTestCase):
    def test_keys_are_by_physical_card_not_by_rank(self):
        cards = FakeBackend().sample()
        vals = flatten_sample(cards, EngineSample(up=True, metrics={"token_usage": 0.5}))
        self.assertIn("gpu.0.power_w", vals)
        self.assertIn("engine.token_usage", vals)
        self.assertEqual(vals["engine.up"], 1.0)

    def test_throttle_is_recorded_as_a_series(self):
        c = CardSample(index=2, name="RTX 3080", throttle=["sw_thermal_slowdown"])
        vals = flatten_sample([c], EngineSample(up=False))
        self.assertEqual(vals["gpu.2.throttled"], 1.0)

    def test_idle_is_not_recorded_as_throttled(self):
        c = CardSample(index=0, name="RTX 5090", throttle=["gpu_idle"])
        vals = flatten_sample([c], EngineSample(up=False))
        self.assertEqual(vals["gpu.0.throttled"], 0.0)

    def test_engine_down_still_produces_a_sample(self):
        vals = flatten_sample([], EngineSample(up=False, reason="refused"))
        self.assertEqual(vals["engine.up"], 0.0)


class TestCollector(CustomTestCase):
    def _collector(self, **kw):
        cfg = CollectorConfig(node_id="node-a", tiers=TIERS, engine_url="", **kw)
        return Collector(
            config=cfg,
            sampler=GpuSampler(FakeBackend(), counters_every=3),
            scraper=FakeScraper(),
            clock=FakeClock(),
        )

    def test_tick_writes_to_the_series(self):
        c = self._collector()
        c.tick()
        c.tick()
        got = c.series.query(resolution="live")
        self.assertGreaterEqual(len(got["points"]), 1)
        self.assertIn("gpu.0.power_w", got["points"][0]["v"])

    def test_group_throughput_from_counter_delta(self):
        c = self._collector()
        c.tick()
        c.tick()
        snap = c.snapshot()
        self.assertAlmostEqual(snap["throughput"]["gen_tok_s"], 50.0)
        self.assertIn("counter delta", snap["throughput"]["source"])

    def test_profiling_cadence_is_decoupled_from_the_base_cadence(self):
        backend = FakeBackend()
        c = Collector(
            config=CollectorConfig(node_id="n", tiers=TIERS, engine_url=""),
            sampler=GpuSampler(backend, counters_every=3),
            scraper=FakeScraper(),
            clock=FakeClock(),
        )
        for _ in range(9):
            c.tick()
        self.assertEqual(backend.calls, 9)
        self.assertEqual(backend.profiling_calls, 3)

    def test_profiling_can_be_switched_off_entirely(self):
        backend = FakeBackend()
        s = GpuSampler(backend, counters_every=0)
        for _ in range(5):
            s.sample()
        self.assertEqual(backend.profiling_calls, 0)

    def test_legacy_profile_every_keyword_still_maps(self):
        # The pre-rename spelling; the planner CLI still passes it.
        s = GpuSampler(FakeBackend(), profile_every=4)
        self.assertEqual(s.counters_every, 4)

    def test_snapshot_reports_field_availability_and_backend(self):
        c = self._collector()
        c.tick()
        snap = c.snapshot()
        self.assertEqual(snap["device_backend"], "fake")
        self.assertTrue(snap["fields"])
        self.assertIn("resolutions", snap)
        self.assertEqual(
            [r["name"] for r in snap["resolutions"]], ["live", "mid"]
        )

    def test_snapshot_is_json_serialisable(self):
        c = self._collector()
        c.tick()
        json.dumps(c.snapshot())

    def test_engine_down_does_not_break_the_loop(self):
        c = Collector(
            config=CollectorConfig(node_id="n", tiers=TIERS, engine_url=""),
            sampler=GpuSampler(FakeBackend(), counters_every=0),
            scraper=FakeScraper(up=False),
            clock=FakeClock(),
        )
        c.tick()
        snap = c.snapshot()
        self.assertFalse(snap["engine"]["up"])
        self.assertIn("refused", snap["engine"]["reason"])

    def test_null_backend_host_still_collects_engine_metrics(self):
        c = Collector(
            config=CollectorConfig(node_id="n", tiers=TIERS, engine_url=""),
            sampler=GpuSampler(NullBackend("no GPU on this host"), counters_every=0),
            scraper=FakeScraper(),
            clock=FakeClock(),
        )
        c.tick()
        snap = c.snapshot()
        self.assertEqual(snap["cards"], [])
        self.assertEqual(snap["fields"][0]["reason"], "no GPU on this host")


class TestPushIngest(CustomTestCase):
    def _pair(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS, token="secret"))
        sent = []

        def opener(url, body, headers):
            sent.append((url, headers))
            payload = json.loads(body.decode())
            if not agg.check_token(payload["node_id"], headers.get("X-Rigmon-Token", "")):
                raise PermissionError("401")
            agg.ingest(payload, address="10.0.0.2")
            return "{}"

        client = PushClient("http://agg:8770", "secret", "node-b", opener=opener)
        col = Collector(
            config=CollectorConfig(node_id="node-b", tiers=TIERS, engine_url=""),
            sampler=GpuSampler(FakeBackend(), counters_every=0),
            scraper=FakeScraper(),
            push=client,
            clock=FakeClock(),
        )
        return agg, col, client, sent

    def test_remote_node_appears_after_a_push(self):
        agg, col, client, _ = self._pair()
        for _ in range(5):
            col.tick()
        self.assertTrue(col.push_now())
        ids = [n["node_id"] for n in agg.nodes()]
        self.assertIn("node-b", ids)
        node = [n for n in agg.nodes() if n["node_id"] == "node-b"][0]
        self.assertTrue(node["remote"])
        self.assertEqual(node["address"], "10.0.0.2")
        self.assertGreater(node["points_received"], 0)

    def test_series_survives_the_hop(self):
        agg, col, client, _ = self._pair()
        for _ in range(5):
            col.tick()
        col.push_now()
        got = agg.query(node_id="node-b", resolution="live")
        pts = got["nodes"]["node-b"]["points"]
        self.assertGreaterEqual(len(pts), 1)
        self.assertIn("gpu.0.power_w", pts[0]["v"])

    def test_push_is_incremental(self):
        agg, col, client, sent = self._pair()
        for _ in range(4):
            col.tick()
        col.push_now()
        first = agg.nodes()[0]["points_received"]
        col.push_now()  # nothing new
        self.assertEqual(agg.nodes()[0]["points_received"], first)
        col.tick()
        col.tick()
        col.push_now()
        self.assertGreater(agg.nodes()[0]["points_received"], first)

    def test_failed_push_is_retried_not_lost(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS, token="secret"))
        state = {"fail": True}

        def opener(url, body, headers):
            if state["fail"]:
                raise OSError("connection refused")
            agg.ingest(json.loads(body.decode()))
            return "{}"

        client = PushClient("http://agg", "secret", "node-b", opener=opener)
        col = Collector(
            config=CollectorConfig(node_id="node-b", tiers=TIERS, engine_url=""),
            sampler=GpuSampler(FakeBackend(), counters_every=0),
            scraper=FakeScraper(),
            push=client,
            clock=FakeClock(),
        )
        for _ in range(4):
            col.tick()
        self.assertFalse(col.push_now())
        self.assertIn("connection refused", client.last_error)
        self.assertEqual(client.cursors, {})
        state["fail"] = False
        self.assertTrue(col.push_now())
        self.assertGreater(agg.nodes()[0]["points_received"], 0)
        self.assertIsNone(client.last_error)

    def test_wrong_token_is_rejected(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS, token="secret"))
        self.assertFalse(agg.check_token("node-b", "wrong"))
        self.assertTrue(agg.check_token("node-b", "secret"))

    def test_mismatched_resolutions_are_named_not_silently_dropped(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        res = agg.ingest(
            {
                "node_id": "node-c",
                "points": {"live": [{"t": 1000.0, "v": {"x": 1.0}}],
                           "weekly": [{"t": 1000.0, "v": {"x": 1.0}}]},
            }
        )
        self.assertEqual(res["accepted"], 1)
        self.assertEqual(res["skipped_tiers"], ["weekly"])
        self.assertIn("--resolution", res["warning"])

    def test_push_without_node_id_is_refused(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        with self.assertRaises(ValueError):
            agg.ingest({"points": {}})


class TestJoinHandshake(CustomTestCase):
    def test_pairing_token_is_single_use(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        tok = agg.mint_join_token()["join_token"]
        out = agg.redeem_join_token(tok, "node-b")
        self.assertTrue(out["push_token"])
        with self.assertRaises(PermissionError):
            agg.redeem_join_token(tok, "node-b")

    def test_pairing_token_expires(self):
        clock = [1000.0]
        agg = Aggregator(
            AggregatorConfig(tiers=TIERS, join_token_ttl_s=60.0),
            clock=lambda: clock[0],
        )
        tok = agg.mint_join_token()["join_token"]
        clock[0] += 61.0
        with self.assertRaises(PermissionError):
            agg.redeem_join_token(tok, "node-b")

    def test_issued_token_authenticates_that_node_only(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS, token="fallback"))
        tok = agg.mint_join_token()["join_token"]
        node_token = agg.redeem_join_token(tok, "node-b")["push_token"]
        self.assertTrue(agg.check_token("node-b", node_token))
        self.assertFalse(agg.check_token("node-b", "fallback"))


class TestAggregatorReads(CustomTestCase):
    def test_local_node_needs_no_push(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        col = Collector(
            config=CollectorConfig(node_id="node-a", tiers=TIERS, engine_url=""),
            sampler=GpuSampler(FakeBackend(), counters_every=0),
            scraper=FakeScraper(),
            clock=FakeClock(),
        )
        agg.attach_local(col)
        col.tick()
        col.tick()
        nodes = agg.nodes()
        self.assertEqual(nodes[0]["node_id"], "node-a")
        self.assertFalse(nodes[0]["remote"])
        pts = agg.query(node_id="node-a", resolution="live")["nodes"]["node-a"]
        self.assertGreaterEqual(len(pts["points"]), 1)

    def test_snapshot_never_leaks_node_tokens(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        tok = agg.mint_join_token()["join_token"]
        secret = agg.redeem_join_token(tok, "node-b")["push_token"]
        agg.ingest({"node_id": "node-b", "meta": {"x": 1}, "points": {}})
        blob = json.dumps(agg.snapshot())
        self.assertNotIn(secret, blob)

    def test_stale_nodes_are_marked_not_removed(self):
        clock = [1000.0]
        agg = Aggregator(
            AggregatorConfig(tiers=TIERS, node_stale_s=30.0), clock=lambda: clock[0]
        )
        agg.ingest({"node_id": "node-b", "points": {}})
        self.assertFalse(agg.nodes()[0]["stale"])
        clock[0] += 120.0
        self.assertTrue(agg.nodes()[0]["stale"])
        self.assertEqual(len(agg.nodes()), 1)

    def test_unknown_node_query_raises(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        with self.assertRaises(KeyError):
            agg.query(node_id="nope")

    def test_resolution_selection_by_window(self):
        agg = Aggregator(AggregatorConfig(tiers=TIERS))
        pts = [{"t": 1000.0 + i, "v": {"x": [1.0, 1.0, 1.0, 1]}} for i in range(40)]
        agg.ingest({"node_id": "n", "points": {"live": pts}})
        got = agg.query(node_id="n", window_s=500.0, max_points=10)
        self.assertEqual(got["nodes"]["n"]["resolution"], "mid")
        self.assertTrue(got["nodes"]["n"]["aggregated"])


class TestAggregatorConfig(CustomTestCase):
    def test_non_loopback_without_token_is_refused(self):
        errs = AggregatorConfig(host="0.0.0.0", token="").validate()
        self.assertTrue(errs)
        self.assertIn("--token", errs[0])

    def test_loopback_without_token_is_fine(self):
        self.assertEqual(AggregatorConfig(host="127.0.0.1", token="").validate(), [])

    def test_bad_port(self):
        self.assertTrue(AggregatorConfig(port=99999).validate())


if __name__ == "__main__":
    unittest.main()
