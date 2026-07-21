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
                 temp, mem_used_mib, mem_total_mib):
        self.name = name
        self.uuid = uuid
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


def _rig():
    # NVML/PCI order: 3080 at index 0, 5090 at index 1 (torch order differs) --
    # UUID is what the client maps on.
    return _FakeNvml([
        _FakeCard("RTX 3080", "GPU-aaaa", util=40, mem_util=25, sm=1900,
                  mem_clk=1180, watts=280, limit=320, temp=70,
                  mem_used_mib=8000, mem_total_mib=20480),
        _FakeCard("RTX 5090", "GPU-bbbb", util=88, mem_util=60, sm=2800,
                  mem_clk=1500, watts=480, limit=575, temp=62,
                  mem_used_mib=24000, mem_total_mib=32768),
    ])


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
