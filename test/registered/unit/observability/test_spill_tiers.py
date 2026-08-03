# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for observability/spill_tiers.py (#522).

Hermetic: every source is a duck-typed fake. No scheduler, no GPU, no tensors
beyond an object with ``numel``/``element_size``.

The rules under test are the ones that decide whether the dashboard tells the
truth:
  * an INACTIVE consumer contributes no key at all -- never a placeholder 0,
    because absence in the scrape is what the panel renders as "absent";
  * a CONFIGURED-but-empty tier contributes 0 -- that is a real reading;
  * cumulative ledgers are not used as occupancy (the pinned pool is summed
    from the live per-layer tensors, not from StreamingStagingLedger);
  * a raising source drops its tier instead of taking the scheduler down.
"""

import unittest

from sglang.srt.observability import spill_tiers as st
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _T:
    def __init__(self, n, es=2):
        self._n, self._es = n, es

    def numel(self):
        return self._n

    def element_size(self):
        return self._es


class _Mod:
    def __init__(self, pinned=None):
        if pinned is not None:
            self._expert_offload = type("C", (), {"_pinned": pinned})()


class _Model:
    def __init__(self, mods):
        self._mods = mods

    def modules(self):
        return list(self._mods)


class _Runner:
    def __init__(self, model):
        self.model = model


class _HostPool:
    def __init__(self, per_token=1024):
        self._pt = per_token

    def get_size_per_token(self):
        return self._pt


class _Kvso:
    def __init__(self, spills=0, max_spills=4, region_tokens=100,
                 per_token=1024, dest=None):
        self.host_pool = _HostPool(per_token)
        self.spills = {i: object() for i in range(spills)}
        self.max_spills = max_spills
        self.region_tokens = region_tokens
        self._dest = dest


class _Tier:
    def __init__(self, name):
        self.name = name


class _Parked:
    def __init__(self, tier_index, rows):
        self.tier_index, self.rows = tier_index, rows


class _Ctl:
    def __init__(self, tiers, parked=None):
        self.tiers = tiers
        self.parked = parked or {}


class _Evictor:
    def __init__(self, enabled, total):
        self._eviction_enabled, self._total_bytes = enabled, total


class _TreeCache:
    def __init__(self, evictor):
        self.cache_controller = type(
            "C", (), {"storage_backend": type("B", (), {"_evictor": evictor})()})()


class _Sched:
    def __init__(self, runner=None, kvso=None, tree_cache=None):
        self.tp_worker = type("W", (), {"model_runner": runner})()
        self.kv_session_offload = kvso
        self.tree_cache = tree_cache


class TestExpertHostBytes(CustomTestCase):
    def test_sums_the_live_pinned_tensors(self):
        model = _Model([_Mod({"w1": _T(1000), "w2": _T(500)}), _Mod()])
        self.assertEqual(st.expert_host_bytes(_Runner(model)), 3000)

    def test_no_installed_cache_is_none_not_zero(self):
        self.assertIsNone(st.expert_host_bytes(_Runner(_Model([_Mod()]))))

    def test_installed_but_empty_cache_is_none(self):
        """No pinned tensors means the pool does not exist yet -- the tier is
        absent, not an empty 0-byte pool."""
        self.assertIsNone(st.expert_host_bytes(_Runner(_Model([_Mod({})]))))

    def test_missing_runner_is_none(self):
        self.assertIsNone(st.expert_host_bytes(None))

    def test_a_non_tensor_slot_is_skipped_not_counted(self):
        model = _Model([_Mod({"w1": _T(10), "bad": object()})])
        self.assertEqual(st.expert_host_bytes(_Runner(model)), 20)


class TestKvSessionBytes(CustomTestCase):
    def test_used_and_total_from_regions(self):
        used, total = st.kv_session_host_bytes(
            _Kvso(spills=2, max_spills=4, region_tokens=100, per_token=1024))
        self.assertEqual(used, 2 * 100 * 1024)
        self.assertEqual(total, 4 * 100 * 1024)

    def test_disabled_manager_is_none(self):
        self.assertIsNone(st.kv_session_host_bytes(None))

    def test_no_host_pool_is_none(self):
        self.assertIsNone(st.kv_session_host_bytes(type("M", (), {})()))

    def test_enabled_and_empty_is_a_real_zero(self):
        used, total = st.kv_session_host_bytes(_Kvso(spills=0))
        self.assertEqual(used, 0)
        self.assertGreater(total, 0)


class TestParkBytes(CustomTestCase):
    def test_configured_tiers_appear_even_while_empty(self):
        ctl = _Ctl([_Tier("file"), _Tier("mooncake")])
        got = st.park_bytes_by_tier(_Kvso(dest=ctl))
        self.assertEqual(got, {"park:file": 0, "park:mooncake": 0})

    def test_parked_rows_convert_with_the_pool_token_size(self):
        ctl = _Ctl([_Tier("file"), _Tier("mooncake")],
                   {"a": _Parked(1, 50), "b": _Parked(1, 25), "c": _Parked(0, 10)})
        got = st.park_bytes_by_tier(_Kvso(per_token=2048, dest=ctl))
        self.assertEqual(got["park:mooncake"], 75 * 2048)
        self.assertEqual(got["park:file"], 10 * 2048)

    def test_no_destinations_configured_is_an_empty_mapping(self):
        self.assertEqual(st.park_bytes_by_tier(_Kvso(dest=None)), {})
        self.assertEqual(st.park_bytes_by_tier(None), {})

    def test_cumulative_transfer_counters_are_not_used(self):
        """park_bytes_out is bytes EVER moved; using it as occupancy would
        grow without bound. The controller's counters must not leak in."""
        ctl = _Ctl([_Tier("mooncake")])
        ctl.counters = {"park_bytes_out": 10 ** 12, "unpark_bytes_in": 5}
        self.assertEqual(st.park_bytes_by_tier(_Kvso(dest=ctl)),
                         {"park:mooncake": 0})


class TestHicacheFileBytes(CustomTestCase):
    def test_tracked_total_is_reported(self):
        self.assertEqual(
            st.hicache_file_bytes(_TreeCache(_Evictor(True, 4096))), 4096)

    def test_untracked_evictor_is_none_not_zero(self):
        """Without eviction configured the evictor never counts, so its 0 is
        wrong rather than empty -- the tier must read absent."""
        self.assertIsNone(st.hicache_file_bytes(_TreeCache(_Evictor(False, 0))))

    def test_no_backend_is_none(self):
        self.assertIsNone(st.hicache_file_bytes(None))


class TestCollect(CustomTestCase):
    def test_inactive_consumers_contribute_no_keys(self):
        used, total = st.collect_spill_tiers(_Sched())
        self.assertEqual(used, {})
        self.assertEqual(total, {})

    def test_active_consumers_are_all_present(self):
        sched = _Sched(
            runner=_Runner(_Model([_Mod({"w": _T(100)})])),
            kvso=_Kvso(spills=1, dest=_Ctl([_Tier("mooncake")],
                                           {"a": _Parked(0, 7)})),
            tree_cache=_TreeCache(_Evictor(True, 999)))
        used, total = st.collect_spill_tiers(sched)
        self.assertEqual(set(used), {st.TIER_EXPERT_HOST, st.TIER_KV_SESSION_HOST,
                                     st.TIER_HICACHE_FILE, "park:mooncake"})
        self.assertEqual(set(total), {st.TIER_KV_SESSION_HOST})

    def test_a_raising_source_drops_its_tier_and_keeps_the_rest(self):
        class _Boom:
            @property
            def host_pool(self):
                raise RuntimeError("boom")

        sched = _Sched(runner=_Runner(_Model([_Mod({"w": _T(100)})])),
                       kvso=_Boom())
        used, _ = st.collect_spill_tiers(sched)
        self.assertIn(st.TIER_EXPERT_HOST, used)
        self.assertNotIn(st.TIER_KV_SESSION_HOST, used)

    def test_tier_ids_match_the_dashboard_catalogue(self):
        from sglang.srt.planner import tier_occupancy

        rows = {r.id for r in tier_occupancy.tier_rows("")}
        self.assertIn(st.TIER_EXPERT_HOST, rows)
        self.assertIn(st.TIER_KV_SESSION_HOST, rows)
        self.assertIn(st.TIER_HICACHE_FILE, rows)
        self.assertEqual(st.TIER_PARK_PREFIX, "park:")


if __name__ == "__main__":
    unittest.main()


class TestPrometheusRoundTrip(CustomTestCase):
    """The gauges are not asserted to work -- they are RUN.

    A dict handed to SchedulerStats has to come back out of a real
    ``generate_latest()`` scrape and parse into the dashboard's tier map. This
    runs in a subprocess because the Prometheus default registry tolerates
    exactly one SchedulerMetricsCollector per process; no GPU, no server, no
    network.
    """

    _PROG = r'''
import types, json
from prometheus_client import REGISTRY, generate_latest
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector, SchedulerStats)
from sglang.srt.planner import tier_occupancy as to

sa = types.SimpleNamespace(
    enable_metrics=True, enable_metrics_for_all_schedulers=False,
    kv_events_config=None, prefill_delayer_forward_passes_buckets=None,
    prefill_delayer_max_delay_passes=0,
    prefill_delayer_wait_seconds_buckets=None)
labels = {"model_name": "m", "moe_ep_rank": 0, "engine_type": "e",
          "tp_rank": 0, "pp_rank": 0, "dp_rank": 0}
c = SchedulerMetricsCollector(labels=labels, server_args=sa)
s = SchedulerStats()
s.spill_tier_used_bytes = {"expert_host_ram": 4096, "park:mooncake": 512}
s.spill_tier_total_bytes = {"kv_session_host_ram": 8192}
c.log_stats(s)
text = generate_latest(REGISTRY).decode()
used, total = to.spill_tier_bytes(text)
rows = {r.id: [r.provenance, r.used] for r in to.tier_rows(text)}
print("@@" + json.dumps({"used": used, "total": total, "rows": rows}))
'''

    def _run(self):
        import json
        import subprocess
        import sys

        out = subprocess.run([sys.executable, "-c", self._PROG],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        line = [x for x in out.stdout.splitlines() if x.startswith("@@")]
        self.assertTrue(line, out.stdout[-2000:] + out.stderr[-2000:])
        return json.loads(line[0][2:])

    def test_dict_survives_the_scrape_and_lands_on_the_right_rows(self):
        got = self._run()
        self.assertEqual(got["used"], {"expert_host_ram": 4096.0,
                                       "park:mooncake": 512.0})
        self.assertEqual(got["total"], {"kv_session_host_ram": 8192.0})
        self.assertEqual(got["rows"]["expert_host_ram"], ["measured", 4096.0])
        self.assertEqual(got["rows"]["park:mooncake"], ["measured", 512.0])
        # A tier that reported nothing stays absent, with no number attached.
        self.assertEqual(got["rows"]["hicache_file_disk"][0], "absent")
        self.assertIsNone(got["rows"]["hicache_file_disk"][1])
