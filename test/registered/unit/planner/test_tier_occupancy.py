# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for tier_occupancy.py -- the spill/offload tier panel (#522).

Hermetic: /metrics is a text blob, /proc/meminfo is injected, no server.

The properties under test are the honesty rules, not the arithmetic:
  * a tier with no live source is DRAWN as absent with a reason, never hidden
    and never zeroed;
  * a failed scrape makes every tier UNKNOWN, and the reason says so instead
    of claiming the tiers are empty;
  * the host-RAM sum adds only byte-valued local host-RAM rows -- the HiCache
    row speaks tokens and is excluded by name.
"""

import unittest

from sglang.srt.planner import tier_occupancy as to
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_LBL = 'model_name="m",instance="i"'
_MEMINFO = "MemTotal:      131072000 kB\nMemFree: 1 kB\n"


def _scrape(**tiers):
    lines = ["# HELP sglang:spill_tier_used_bytes x", "# TYPE ... gauge"]
    for tier, val in tiers.items():
        tier = tier.replace("__", ":")
        lines.append(
            f'sglang:spill_tier_used_bytes{{{_LBL},spill_tier="{tier}"}} {val}')
    return "\n".join(lines) + "\n"


def _by_id(rows):
    return {r["id"]: r for r in rows}


class TestScrapeParsing(CustomTestCase):
    def test_labels_are_kept_apart(self):
        used, total = to.spill_tier_bytes(
            _scrape(expert_host_ram=100, kv_session_host_ram=200))
        self.assertEqual(used, {"expert_host_ram": 100.0,
                                "kv_session_host_ram": 200.0})
        self.assertEqual(total, {})

    def test_totals_parse_into_their_own_map(self):
        text = (f'sglang:spill_tier_total_bytes{{{_LBL},'
                'spill_tier="kv_session_host_ram"} 8192\n')
        used, total = to.spill_tier_bytes(text)
        self.assertEqual(used, {})
        self.assertEqual(total, {"kv_session_host_ram": 8192.0})

    def test_comments_and_unlabelled_lines_are_ignored(self):
        used, _ = to.spill_tier_bytes(
            "# sglang:spill_tier_used_bytes fake\n"
            "sglang:spill_tier_used_bytes 55\n")
        self.assertEqual(used, {})

    def test_empty_scrape_is_empty_not_an_error(self):
        self.assertEqual(to.spill_tier_bytes(""), ({}, {}))
        self.assertEqual(to.spill_tier_bytes(None), ({}, {}))


class TestHostMemTotal(CustomTestCase):
    def test_kb_to_bytes(self):
        self.assertEqual(to.host_mem_total_bytes(_MEMINFO), 131072000 * 1024)

    def test_missing_line_is_none(self):
        self.assertIsNone(to.host_mem_total_bytes("MemFree: 12 kB\n"))


class TestAbsentHonesty(CustomTestCase):
    def test_every_catalogued_tier_is_drawn_even_with_no_data(self):
        rows = to.tier_rows("", metrics_available=True)
        ids = {r.id for r in rows}
        for expected in ("vram_short_term_register", "vram_peer",
                         "expert_host_ram", "kv_session_host_ram",
                         "hicache_host_ram", "hicache_file_disk",
                         "disk_nvme_experts", "disk_hibernate",
                         "remote_rig_ram", "remote_rig_vram",
                         "remote_rig_disk"):
            self.assertIn(expected, ids)

    def test_absent_rows_carry_a_reason_and_no_number(self):
        for row in to.tier_rows("", metrics_available=True):
            if row.provenance != to.ABSENT:
                continue
            self.assertIsNone(row.used, row.id)
            self.assertIsNone(row.total, row.id)
            self.assertTrue(row.missing_reason.strip(), row.id)

    def test_absent_is_never_rendered_as_zero(self):
        rows = to.tier_rows("", metrics_available=True)
        self.assertNotIn(0, [r.used for r in rows])
        self.assertNotIn(0.0, [r.used for r in rows])

    def test_unwired_registries_are_named_as_the_reason(self):
        rows = _by_id([r.to_json() for r in to.tier_rows("")])
        self.assertIn("#286", rows["vram_short_term_register"]["label"])
        self.assertIn("no production path",
                      rows["vram_short_term_register"]["missing_reason"])
        self.assertIn("design only", rows["disk_nvme_experts"]["missing_reason"])

    def test_no_metrics_means_unknown_not_empty(self):
        rows = _by_id([r.to_json() for r in to.tier_rows(
            "", metrics_available=False)])
        for tier in ("expert_host_ram", "kv_session_host_ram",
                     "hicache_host_ram", "hicache_file_disk"):
            self.assertEqual(rows[tier]["provenance"], to.ABSENT)
            self.assertIn("unknown, not empty", rows[tier]["missing_reason"])

    def test_provenance_vocabulary_has_no_probably_tier(self):
        seen = {r.provenance for r in to.tier_rows(
            _scrape(expert_host_ram=1), metrics_available=True)}
        self.assertTrue(seen <= {to.MEASURED, to.ABSENT}, seen)


class TestMeasuredRows(CustomTestCase):
    def test_expert_pool_is_measured_with_its_source_named(self):
        row = _by_id([r.to_json() for r in to.tier_rows(
            _scrape(expert_host_ram=10 * 2 ** 30), host_total=2 ** 37)])
        r = row["expert_host_ram"]
        self.assertEqual(r["provenance"], to.MEASURED)
        self.assertEqual(r["used"], float(10 * 2 ** 30))
        self.assertIn("pinned host tensors", r["source"])
        self.assertIn("/proc/meminfo", r["total_scope"])

    def test_kvso_capacity_comes_from_the_scrape_not_from_proc(self):
        text = _scrape(kv_session_host_ram=2 ** 31) + (
            f'sglang:spill_tier_total_bytes{{{_LBL},'
            'spill_tier="kv_session_host_ram"} 8589934592\n')
        r = _by_id([x.to_json() for x in to.tier_rows(
            text, host_total=2 ** 37)])["kv_session_host_ram"]
        self.assertEqual(r["total"], 8589934592.0)
        self.assertEqual(r["total_scope"], "")
        self.assertAlmostEqual(r["used_frac"], 0.25)

    def test_park_backend_decides_kind_and_location(self):
        rows = _by_id([r.to_json() for r in to.tier_rows(
            _scrape(**{"park__file": 100, "park__mooncake": 200}))])
        self.assertEqual((rows["park:file"]["kind"],
                          rows["park:file"]["location"]), ("disk", "local"))
        self.assertEqual((rows["park:mooncake"]["kind"],
                          rows["park:mooncake"]["location"]),
                         ("host_ram", "remote"))
        # A configured remote target replaces the "no remote target" row.
        self.assertNotIn("remote_rig_ram", rows)

    def test_no_remote_destination_yields_a_visible_absent_remote_row(self):
        rows = _by_id([r.to_json() for r in to.tier_rows(
            _scrape(**{"park__file": 100}))])
        self.assertEqual(rows["remote_rig_ram"]["provenance"], to.ABSENT)
        self.assertIn("no remote park destination",
                      rows["remote_rig_ram"]["missing_reason"])

    def test_hicache_row_keeps_tokens_as_its_unit(self):
        r = _by_id([x.to_json() for x in to.tier_rows(
            "", hicache={"host_used_tokens": 1200,
                         "host_total_tokens": 8000})])["hicache_host_ram"]
        self.assertEqual(r["unit"], "tokens")
        self.assertEqual(r["used"], 1200)
        self.assertIn("no byte gauge exists", r["total_scope"])

    def test_a_configured_but_empty_tier_is_measured_zero_not_absent(self):
        r = _by_id([x.to_json() for x in to.tier_rows(
            _scrape(**{"park__file": 0}))])["park:file"]
        self.assertEqual(r["provenance"], to.MEASURED)
        self.assertEqual(r["used"], 0.0)


class TestTierView(CustomTestCase):
    def _view(self, **kw):
        return to.tier_view(
            _scrape(expert_host_ram=10 * 2 ** 30,
                    kv_session_host_ram=2 * 2 ** 30,
                    **{"park__mooncake": 2 ** 29}),
            hicache={"host_used_tokens": 1200, "host_total_tokens": 8000},
            meminfo_text=_MEMINFO, **kw)

    def test_sum_covers_local_byte_valued_host_ram_only(self):
        v = self._view()
        self.assertEqual(v["host_ram_used_bytes"], float(12 * 2 ** 30))
        self.assertEqual(sorted(v["host_ram_counted"]),
                         ["expert_host_ram", "kv_session_host_ram"])

    def test_token_valued_row_is_excluded_and_says_so(self):
        v = self._view()
        self.assertEqual(v["host_ram_excluded_non_byte"], ["hicache_host_ram"])
        self.assertNotIn("hicache_host_ram", v["host_ram_counted"])

    def test_remote_tier_is_not_folded_into_the_host_sum(self):
        v = self._view()
        self.assertNotIn("park:mooncake", v["host_ram_counted"])

    def test_total_ram_is_labelled_as_the_dashboard_host(self):
        v = self._view()
        self.assertEqual(v["host_ram_total_bytes"], 131072000 * 1024)
        self.assertIn("dashboard host", v["host_ram_total_scope"])

    def test_absent_and_measured_lists_partition_the_rows(self):
        v = self._view()
        self.assertEqual(
            len(v["measured_tiers"]) + len(v["absent_tiers"]), len(v["rows"]))

    def test_nothing_measured_reports_no_sum_rather_than_zero(self):
        v = to.tier_view("", meminfo_text=_MEMINFO, metrics_available=False)
        self.assertIsNone(v["host_ram_used_bytes"])
        self.assertEqual(v["measured_tiers"], [])


if __name__ == "__main__":
    unittest.main()
