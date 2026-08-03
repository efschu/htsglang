# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for live_metrics.py -- the landing-page live monitor snapshot.

No GPU, no server boot: /metrics is a mocked Prometheus text blob and NVML is a
fake object. Covers the design contract:

  * non-cached prefill tok/s SUBTRACTS cache-served tokens from the prompt delta,
  * per-tier cache-hit rates parse out of the multi-label cached_tokens_total,
  * delta math returns None on the first call and re-baselines safely on a
    counter reset (server restart),
  * NVML per-card fields map by UUID with used/total memory + utilization,
  * missing spec / hicache metrics degrade gracefully to None.
"""

import unittest

from sglang.srt.planner.live_metrics import (
    GpuLive,
    read_gpu_live,
    snapshot,
)


# ---------------------------------------------------------------------------
# Fake NVML (per-card live telemetry).
# ---------------------------------------------------------------------------
class _FakeCard:
    def __init__(self, name, uuid, util, mem_util, sm, mem_clk, watts, limit,
                 temp, mem_used_mib, mem_total_mib, bus_id=None):
        self.name = name
        self.uuid = uuid
        #: PCI BDF -- the only thing the #331 identity map bridges the CUDA
        #: ordinal over (#397). A card without one cannot be placed.
        self.bus_id = bus_id
        self.util = util
        self.mem_util = mem_util
        self.sm = sm
        self.mem_clk = mem_clk
        self.watts = watts
        self.limit = limit
        self.temp = temp
        self.mem_used = mem_used_mib * 2**20
        self.mem_total = mem_total_mib * 2**20


class _Util:
    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory


class _Mem:
    def __init__(self, used, total):
        self.used = used
        self.total = total
        self.free = total - used


class _FakeNvml:
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, cards):
        self.cards = cards
        self.shutdown_calls = 0

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i):
        return self.cards[i]

    def nvmlDeviceGetName(self, h):
        return h.name

    def nvmlDeviceGetUUID(self, h):
        return h.uuid

    def nvmlDeviceGetPciInfo(self, h):
        if h.bus_id is None:
            raise RuntimeError("no pci info")

        class _Pci:
            busId = h.bus_id

        return _Pci()

    def nvmlDeviceGetUtilizationRates(self, h):
        return _Util(h.util, h.mem_util)

    def nvmlDeviceGetMemoryInfo(self, h):
        return _Mem(h.mem_used, h.mem_total)

    def nvmlDeviceGetClockInfo(self, h, which):
        return h.sm if which == self.NVML_CLOCK_SM else h.mem_clk

    def nvmlDeviceGetTemperature(self, h, sensor):
        return h.temp

    def nvmlDeviceGetPowerUsage(self, h):
        return h.watts * 1000.0

    def nvmlDeviceGetPowerManagementLimit(self, h):
        return h.limit * 1000.0


BDF_3080 = "00000000:01:00.0"
BDF_5090 = "00000000:2D:00.0"


def _rig(bus_ids=True):
    # NVML/PCI order: 3080 at index 0, 5090 at index 1 (torch order differs) --
    # UUID is what the client maps on.
    return _FakeNvml([
        _FakeCard("RTX 3080", "GPU-aaaa", util=40, mem_util=25, sm=1900,
                  mem_clk=1180, watts=280, limit=320, temp=70,
                  mem_used_mib=8000, mem_total_mib=20480,
                  bus_id=BDF_3080 if bus_ids else None),
        _FakeCard("RTX 5090", "GPU-bbbb", util=88, mem_util=60, sm=2800,
                  mem_clk=1500, watts=480, limit=575, temp=62,
                  mem_used_mib=24000, mem_total_mib=32768,
                  bus_id=BDF_5090 if bus_ids else None),
    ])


def _cuda_fastest_first():
    """torch's view of that rig: the 5090 is cuda:0, the 3080 cuda:1."""
    return {BDF_5090: 0, BDF_3080: 1}


def _inject_cuda_order(mapping):
    from unittest import mock

    from sglang.srt.registry import nvml as registry_nvml

    return mock.patch.object(
        registry_nvml,
        "_cuda_ordinals_by_bus",
        lambda allow_cuda_init=False: dict(mapping),
    )


# ---------------------------------------------------------------------------
# Mocked /metrics blobs.
# ---------------------------------------------------------------------------
def _metrics(prompt, gen, device=0, host=0, storage=0, *, spec=True,
             hicache=True):
    lines = [
        "# HELP sglang:prompt_tokens_total total prompt tokens",
        "# TYPE sglang:prompt_tokens_total counter",
        f'sglang:prompt_tokens_total{{model="m"}} {prompt}',
        f'sglang:generation_tokens_total{{model="m"}} {gen}',
        f'sglang:gen_throughput{{model="m"}} 42.0',
    ]
    if device:
        lines.append(
            f'sglang:cached_tokens_total{{model="m",cache_source="device"}} {device}')
    if host:
        lines.append(
            f'sglang:cached_tokens_total{{model="m",cache_source="host"}} {host}')
    if storage:
        lines.append(
            'sglang:cached_tokens_total{model="m",cache_source="storage_file"} '
            f'{storage}')
    if spec:
        lines += [
            'sglang:spec_accept_rate{model="m"} 0.72',
            'sglang:spec_num_steps{model="m"} 3',
            'sglang:spec_ema_accept_len{model="m"} 2.4',
        ]
    if hicache:
        lines += [
            'sglang:hicache_host_used_tokens{model="m"} 120000',
            'sglang:hicache_host_total_tokens{model="m"} 400000',
        ]
    return "\n".join(lines) + "\n"


class TestNvmlPerCard(unittest.TestCase):
    def test_fields_and_uuid_mapping(self):
        cards = read_gpu_live(nvml=_rig())
        self.assertEqual(len(cards), 2)
        self.assertIsInstance(cards[0], GpuLive)
        # index 0 is the 3080 in NVML order; identity is via uuid, not name/order
        c0 = cards[0]
        self.assertEqual(c0.uuid, "aaaa")  # normalized (lowercase hex only)
        self.assertEqual(c0.name, "RTX 3080")
        self.assertEqual(c0.utilization_pct, 40)
        self.assertEqual(c0.mem_utilization_pct, 25)
        self.assertEqual(c0.sm_clock_mhz, 1900)
        self.assertEqual(c0.mem_clock_mhz, 1180)
        self.assertAlmostEqual(c0.power_watts, 280.0)
        self.assertAlmostEqual(c0.power_limit_w, 320.0)
        self.assertEqual(c0.temperature_c, 70)
        self.assertEqual(c0.mem_used_mib, 8000)
        self.assertEqual(c0.mem_total_mib, 20480)
        self.assertAlmostEqual(c0.mem_used_frac, 8000 / 20480)
        # the 5090 lives at NVML index 1, mapped by its own uuid
        c1 = cards[1]
        self.assertEqual(c1.uuid, "bbbb")
        self.assertEqual(c1.mem_total_mib, 32768)
        self.assertEqual(c1.utilization_pct, 88)

    def test_injected_nvml_not_shutdown(self):
        rig = _rig()
        read_gpu_live(nvml=rig)
        self.assertEqual(rig.shutdown_calls, 0)  # caller owns injected nvml

    def test_cuda_index_bridged_and_marked(self):
        # Every card carries its CUDA-order index (the --rank-gpu-id /
        # --base-gpu-id space) next to the NVML index, resolved against the
        # #331 identity map over the PCI BDF: the 5090 (nvml:1) is cuda:0 and
        # the 3080 (nvml:0) is cuda:1, i.e. the two orders diverge here and
        # the annotation has to cross them.
        with _inject_cuda_order(_cuda_fastest_first()):
            cards = read_gpu_live(nvml=_rig())
        by_name = {c.name: c for c in cards}
        self.assertEqual(by_name["RTX 5090"].cuda_index, 0)
        self.assertEqual(by_name["RTX 3080"].cuda_index, 1)
        for c in cards:
            self.assertEqual(c.cuda_index_source, "identity-map")
            j = c.to_json()
            self.assertIn("cuda_index", j)
            self.assertIn("cuda_index_source", j)

    def test_an_unresolvable_order_leaves_the_cards_unbridged(self):
        # #397: no emulated order. A card whose CUDA ordinal cannot be
        # resolved carries None, and the UI shows only the NVML index --
        # rather than a plausible number that names the wrong card.
        with _inject_cuda_order({}):
            cards = read_gpu_live(nvml=_rig())
        for c in cards:
            self.assertIsNone(c.cuda_index)
            self.assertIsNone(c.cuda_index_source)

    def test_cards_without_a_pci_bdf_stay_unbridged(self):
        with _inject_cuda_order(_cuda_fastest_first()):
            cards = read_gpu_live(nvml=_rig(bus_ids=False))
        for c in cards:
            self.assertIsNone(c.cuda_index)


class TestPrefillSubtractsCache(unittest.TestCase):
    def test_noncached_prefill_excludes_cache_served(self):
        # window: +1000 prompt tokens, of which 600 came from caches -> only 400
        # are real (non-cached) prefill work. +2000 generation tokens.
        m0 = _metrics(prompt=1000, gen=5000, device=200, host=300, storage=100)
        m1 = _metrics(prompt=2000, gen=7000, device=400, host=400, storage=100)
        snap0, st0 = snapshot("http://x", None, nvml=_rig(),
                              metrics_text=m0, now=100.0)
        self.assertIsNone(snap0["rates"])  # first call: no deltas yet
        snap1, st1 = snapshot("http://x", st0, nvml=_rig(),
                              metrics_text=m1, now=110.0)  # dt = 10s
        r = snap1["rates"]
        # gross prompt delta = 1000; cached delta = (600-... ) let's compute:
        # cached_total m0 = 600, m1 = 900 -> cached delta 300. non-cached =
        # 1000 - 300 = 700 over 10s = 70 tok/s.
        self.assertAlmostEqual(r["prefill_tok_s"], 70.0)
        self.assertAlmostEqual(r["prefill_tok_s_gross"], 100.0)
        self.assertAlmostEqual(r["cached_tok_s"], 30.0)
        self.assertAlmostEqual(r["decode_tok_s"], 200.0)

    def test_cached_exceeding_prompt_clamps_to_zero(self):
        m0 = _metrics(prompt=1000, gen=1000, device=100)
        # prompt +50 but cached +500 (cross-boundary) -> non-cached clamps to 0
        m1 = _metrics(prompt=1050, gen=1000, device=600)
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=m0, now=0.0)
        snap1, _ = snapshot("x", st0, nvml=_rig(), metrics_text=m1, now=1.0)
        self.assertAlmostEqual(snap1["rates"]["prefill_tok_s"], 0.0)


class TestPerTierHitRates(unittest.TestCase):
    def test_hit_rate_per_tier(self):
        m0 = _metrics(prompt=0, gen=0)
        m1 = _metrics(prompt=1000, gen=100, device=500, host=200, storage=100)
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=m0, now=0.0)
        snap1, _ = snapshot("x", st0, nvml=_rig(), metrics_text=m1, now=1.0)
        hr = snap1["cache_hit_rates"]
        self.assertAlmostEqual(hr["device"], 0.5)          # 500/1000
        self.assertAlmostEqual(hr["host"], 0.2)            # 200/1000
        self.assertAlmostEqual(hr["storage_file"], 0.1)    # 100/1000
        self.assertAlmostEqual(hr["overall"], 0.8)         # 800/1000

    def test_hit_rate_none_when_no_prompt_tokens(self):
        m0 = _metrics(prompt=1000, gen=0, device=100)
        m1 = _metrics(prompt=1000, gen=500, device=100)  # no new prompt tokens
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=m0, now=0.0)
        snap1, _ = snapshot("x", st0, nvml=_rig(), metrics_text=m1, now=1.0)
        self.assertIsNone(snap1["cache_hit_rates"])
        # decode still measured
        self.assertAlmostEqual(snap1["rates"]["decode_tok_s"], 500.0)


class TestResetAndFirstCall(unittest.TestCase):
    def test_first_call_rates_none(self):
        snap, st = snapshot("x", None, nvml=_rig(),
                            metrics_text=_metrics(500, 500), now=0.0)
        self.assertTrue(snap["ok"])
        self.assertIsNone(snap["rates"])
        self.assertIsNone(snap["cache_hit_rates"])
        self.assertIn("counters", st)

    def test_counter_reset_is_safe(self):
        # server restarts: counters drop from high back to small -> no negative
        # rate, clamps to 0 and re-baselines.
        m0 = _metrics(prompt=1_000_000, gen=1_000_000, device=500_000)
        m1 = _metrics(prompt=100, gen=100, device=10)   # fresh process
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=m0, now=0.0)
        snap1, st1 = snapshot("x", st0, nvml=_rig(), metrics_text=m1, now=5.0)
        r = snap1["rates"]
        self.assertGreaterEqual(r["decode_tok_s"], 0.0)
        self.assertGreaterEqual(r["prefill_tok_s"], 0.0)
        self.assertEqual(r["decode_tok_s"], 0.0)
        self.assertEqual(r["prefill_tok_s"], 0.0)

    def test_nonpositive_dt_yields_zero(self):
        m = _metrics(1000, 1000, device=100)
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=m, now=10.0)
        snap1, _ = snapshot("x", st0, nvml=_rig(),
                            metrics_text=_metrics(2000, 3000, device=200),
                            now=10.0)  # same timestamp -> dt = 0
        self.assertEqual(snap1["rates"]["decode_tok_s"], 0.0)
        self.assertEqual(snap1["rates"]["prefill_tok_s"], 0.0)


class TestGracefulDegradation(unittest.TestCase):
    def test_spec_absent_is_none(self):
        m = _metrics(500, 500, spec=False)
        snap, _ = snapshot("x", None, nvml=_rig(), metrics_text=m, now=0.0)
        self.assertIsNone(snap["spec"])

    def test_spec_present_parsed(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=_metrics(500, 500), now=0.0)
        s = snap["spec"]
        self.assertAlmostEqual(s["accept_rate"], 0.72)
        self.assertAlmostEqual(s["adaptive_k"], 3.0)
        self.assertAlmostEqual(s["ema_accept_len"], 2.4)

    def test_hicache_absent_is_none(self):
        m = _metrics(500, 500, hicache=False)
        snap, _ = snapshot("x", None, nvml=_rig(), metrics_text=m, now=0.0)
        self.assertIsNone(snap["hicache"])

    def test_hicache_present_parsed(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=_metrics(500, 500), now=0.0)
        h = snap["hicache"]
        self.assertAlmostEqual(h["host_used_tokens"], 120000.0)
        self.assertAlmostEqual(h["host_total_tokens"], 400000.0)
        self.assertAlmostEqual(h["host_used_frac"], 0.3)

    def test_nvml_failure_degrades_to_empty_gpus(self):
        class _Boom:
            def nvmlInit(self):
                pass

            def nvmlDeviceGetCount(self):
                raise RuntimeError("no NVML on this box")

            def nvmlShutdown(self):
                pass

        snap, _ = snapshot("x", None, nvml=_Boom(),
                           metrics_text=_metrics(500, 500), now=0.0)
        self.assertTrue(snap["ok"])   # scrape still succeeded
        self.assertEqual(snap["gpus"], [])
        self.assertIsNotNone(snap["nvml_error"])


class TestEndpointAndConfig(unittest.TestCase):
    def test_supervisor_launch_config(self):
        class _Settings:
            host = "127.0.0.1"
            port = 30000

            def launch_command(self):
                return ["python", "-m", "sglang.launch_server", "--tp", "2"]

        # dataclasses.asdict fails on a non-dataclass -> launch_config falls back
        # to None but base_url is still derived; server_info fetch is skipped
        # here because we inject metrics_text (no network).
        class _Sup:
            settings = _Settings()

        snap, _ = snapshot(_Sup(), None, nvml=_rig(),
                           metrics_text=_metrics(500, 500), now=0.0)
        self.assertEqual(snap["endpoint"], "http://127.0.0.1:30000")

    def test_supervisor_dataclass_launch_config(self):
        import dataclasses

        @dataclasses.dataclass
        class _DCSettings:
            host: str = "127.0.0.1"
            port: int = 8000
            tp_size: int = 2

            def launch_command(self):
                return ["python", "server", "--tp-size", "2"]

        class _Sup:
            settings = _DCSettings()

        snap, _ = snapshot(_Sup(), None, nvml=_rig(),
                           metrics_text=_metrics(500, 500), now=0.0)
        cfg = snap["launch_config"]
        self.assertEqual(cfg["tp_size"], 2)
        self.assertEqual(cfg["launch_argv"], ["python", "server", "--tp-size", "2"])

    def test_string_endpoint_normalized(self):
        snap, _ = snapshot("localhost:30000", None, nvml=_rig(),
                           metrics_text=_metrics(500, 500), now=0.0)
        self.assertEqual(snap["endpoint"], "http://localhost:30000")

    def test_no_endpoint_and_no_metrics_is_error(self):
        snap, state = snapshot("", None, nvml=_rig(), now=0.0)
        self.assertFalse(snap["ok"])
        self.assertIn("no endpoint", snap["error"])


if __name__ == "__main__":
    unittest.main()


class TestConcurrencyGauges(unittest.TestCase):
    """Per-session rates need the concurrency gauges: a server-wide token rate
    says nothing about what one request experiences."""

    def _with_reqs(self, running=None, queued=None):
        m = _metrics(prompt=100, gen=100)
        extra = []
        if running is not None:
            extra.append('sglang:num_running_reqs{model="m"} %s' % running)
        if queued is not None:
            extra.append('sglang:num_queue_reqs{model="m"} %s' % queued)
        return m + ("\n".join(extra) + "\n" if extra else "")

    def test_gauges_are_carried_into_the_snapshot(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=self._with_reqs(running=4, queued=2),
                           now=0.0)
        self.assertEqual(snap["num_running_reqs"], 4.0)
        self.assertEqual(snap["num_queue_reqs"], 2.0)

    def test_absent_gauge_is_none_not_zero(self):
        # "no concurrency metric" and "nothing running" must stay tellable
        # apart -- the UI shows n/a for one and 0 for the other.
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=self._with_reqs(), now=0.0)
        self.assertIsNone(snap["num_running_reqs"])
        self.assertIsNone(snap["num_queue_reqs"])

    def test_zero_running_is_reported_as_zero(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=self._with_reqs(running=0), now=0.0)
        self.assertEqual(snap["num_running_reqs"], 0.0)


# ---------------------------------------------------------------------------
# #522: rolling medians, the four-state diagnosis, and the tier view, all
# carried by the same snapshot/state contract.
# ---------------------------------------------------------------------------
class TestRateMediansInSnapshot(unittest.TestCase):
    """The badge behind an idle rate tile. What must hold end-to-end: idle
    polls do not enter the window, so an idle server still shows the median of
    what it did while it was working."""

    def test_first_call_has_no_medians(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=_metrics(prompt=100, gen=100), now=0.0)
        self.assertEqual(snap["rate_medians"], {})

    def test_window_survives_in_the_returned_state(self):
        m0 = _metrics(prompt=100, gen=100)
        m1 = _metrics(prompt=100, gen=200)
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=m0, now=0.0)
        snap1, st1 = snapshot("x", st0, nvml=_rig(), metrics_text=m1, now=1.0)
        self.assertEqual(st1["rate_windows"]["decode_tok_s"], [100.0])
        self.assertEqual(snap1["rate_medians"]["decode_tok_s"]["median"], 100.0)
        self.assertEqual(snap1["rate_medians"]["decode_tok_s"]["n"], 1)

    def test_idle_polls_keep_the_median_and_do_not_dilute_it(self):
        texts = [(_metrics(prompt=100, gen=100), 0.0),
                 (_metrics(prompt=100, gen=200), 1.0),
                 (_metrics(prompt=100, gen=260), 2.0)]
        state = None
        for text, t in texts:
            snap, state = snapshot("x", state, nvml=_rig(), metrics_text=text,
                                   now=t)
        # now 10 idle polls: counters frozen, decode reads 0
        idle = _metrics(prompt=100, gen=260)
        for i in range(10):
            snap, state = snapshot("x", state, nvml=_rig(), metrics_text=idle,
                                   now=3.0 + i)
        self.assertEqual(snap["rates"]["decode_tok_s"], 0.0)
        med = snap["rate_medians"]["decode_tok_s"]
        self.assertEqual(med["n"], 2)          # only the two working windows
        self.assertEqual(med["median"], 80.0)  # median(100, 60)

    def test_spec_median_needs_decode_activity(self):
        idle = _metrics(prompt=100, gen=100)
        _, st0 = snapshot("x", None, nvml=_rig(), metrics_text=idle, now=0.0)
        snap, _ = snapshot("x", st0, nvml=_rig(), metrics_text=idle, now=1.0)
        self.assertNotIn("spec_accept_rate", snap["rate_medians"])


class TestServerStateInSnapshot(unittest.TestCase):
    """The four-state diagnosis. The falsifier against the defect: a snapshot
    whose scrape failed must NOT claim the metrics-flag state unless the API
    probe answered."""

    def test_healthy_scrape_is_state_four(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=_metrics(prompt=1, gen=1), now=0.0)
        self.assertEqual(snap["server_state"]["state"], "running_with_metrics")
        self.assertTrue(snap["server_state"]["running"])

    def test_refused_scrape_without_api_is_not_running(self):
        from sglang.srt.planner import server_state as ss

        snap, _ = snapshot(
            "http://127.0.0.1:1", None, nvml=_rig(), metrics_text="",
            metrics_probe=ss.Probe(ok=False, path="/metrics",
                                   error="ConnectionRefusedError: refused"),
            probe_opener=lambda url, timeout=None: (_ for _ in ()).throw(
                ConnectionRefusedError("refused")),
            tcp_probe=lambda h, p: False, now=0.0)
        self.assertEqual(snap["server_state"]["state"], "not_running")
        self.assertNotIn("enable-metrics", snap["server_state"]["headline"])

    def test_api_up_and_metrics_404_is_the_flag_diagnosis(self):
        from sglang.srt.planner import server_state as ss

        class _R:
            def getcode(self):
                return 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        snap, _ = snapshot(
            "http://127.0.0.1:1", None, nvml=_rig(), metrics_text="",
            metrics_probe=ss.Probe(ok=False, path="/metrics", status=404,
                                   error="HTTP 404"),
            probe_opener=lambda url, timeout=None: _R(), now=0.0)
        self.assertEqual(snap["server_state"]["state"], "running_no_metrics")
        self.assertIn("--enable-metrics", snap["server_state"]["headline"])

    def test_managed_boot_is_starting_not_a_flag_claim(self):
        from sglang.srt.planner import server_state as ss

        snap, _ = snapshot(
            "http://127.0.0.1:1", None, nvml=_rig(), metrics_text="",
            metrics_probe=ss.Probe(ok=False, path="/metrics", error="refused"),
            probe_opener=lambda url, timeout=None: (_ for _ in ()).throw(
                ConnectionRefusedError("refused")),
            managed_state="booting", now=0.0)
        self.assertEqual(snap["server_state"]["state"], "starting")
        self.assertNotIn("enable-metrics", snap["server_state"]["headline"])


class TestSpillTiersInSnapshot(unittest.TestCase):
    def test_tier_rows_are_always_present(self):
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=_metrics(prompt=1, gen=1), now=0.0)
        tiers = snap["spill_tiers"]
        self.assertTrue(tiers["rows"])
        self.assertTrue(tiers["absent_tiers"])
        # HiCache is exported by _metrics(), so its host tier is measured.
        self.assertIn("hicache_host_ram", tiers["measured_tiers"])

    def test_a_measured_tier_shows_up_from_the_scrape(self):
        extra = ('sglang:spill_tier_used_bytes{model="m",'
                 'spill_tier="expert_host_ram"} 1024\n')
        snap, _ = snapshot("x", None, nvml=_rig(),
                           metrics_text=_metrics(prompt=1, gen=1) + extra,
                           now=0.0)
        rows = {r["id"]: r for r in snap["spill_tiers"]["rows"]}
        self.assertEqual(rows["expert_host_ram"]["provenance"], "measured")
        self.assertEqual(rows["expert_host_ram"]["used"], 1024.0)
