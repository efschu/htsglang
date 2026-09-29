# SPDX-License-Identifier: Apache-2.0
"""Host ledger in the Docker form (boot z30w-park, 2026-09-29): mark, arena, store.

MEASURED (z30w-park, creep_0929.jsonl 09:12:00Z, memory.max 84.00 GiB):
  nonreclaim 79.46 GiB = anon 25.14 + shmem 53.36 + kernel 0.95 (- file cache);
  /dev/shm held 7.9 of the shmem (arena dir 6.27 + weg2-seq 1.50 + l3idx 0.13),
  the expert store tmpfs /mnt/nf-experts the map's 'Store ~39.2 GiB'.
  The ledger predicted 47.72 GiB against a 95.90 mark:
    * the mark: a finite memory.max reaps first -- 84, not 95.90;
    * the arena: priced from the launcher env (ARENA_GIB 22, MAMBA 140) and a
      per-TOKEN page, while the ranks ran the group env (4 GiB, 32 slots) in
      786432-B pages -- 23.75 GiB priced, 4.03 + 1.75 on /dev/shm;
    * the store: 39.2 GiB of shmem priced nowhere.

Guarded here: reap_mark_gib (CT999 unchanged, Docker 84), the ledger's
WATERMARK/BOOT bound under memory.max, the cold_tier_shm_gib post in the sum
and on the ARM line, cold_tier_shm_post (tmpfs only, from the map, resident
bytes credited), the arena term through the group env, and the z30w
composition -- what the corrections book and what stays unbooked.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GIB = int(host_ledger.GIB)
MIB = 1 << 20
CONST = host_ledger.OBSERVED_REAP_NONRECLAIM_BYTES / host_ledger.GIB

NF_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
Z30W_FRONT = ("/spinning/docker-acceptance/nf/evidence/boot_weg2_dkrnfh91dprsavisadoptstcutvsyncod"
              "x2bswre2cutz30wparkbar1dauer09290827_50ae2014b0_0929_082743.front.log")
#: the z30w group env, the arena lines (both groups carry the same values)
Z30W_ARENA_ENV = {
    "SGLANG_HICACHE_ARENA_GIB": "4",
    "SGLANG_HICACHE_ARENA_MAMBA_SLOTS": "32",
    "SGLANG_HICACHE_ARENA_KV_PAGE_BYTES": "786432",
    "SGLANG_HICACHE_ARENA_STAGING_GB": "0.05",
}
#: z30w PLATZTAUSCH-KARTE / DRAFT-SWAP: 346 slots x 48 layers x 2.417 MiB
Z30W_STORE_SLOTS = 346
Z30W_ROW_MIB = 40142.0 / (346 * 48)

#: a small ring, so an arm funds under the 84 GiB mark and choose() returns its lines
RING_KW = dict(ring_bytes=4096 * MIB, ring_span1_bytes=4096 * MIB)


def _live(max_gib=None):
    return {"current_gib": 1.09, "nonreclaim_gib": 0.79, "file_reclaimable_gib": 0.30,
            "source": "test", "max_gib": max_gib}


class TestReapMark(CustomTestCase):
    def test_ct999_keeps_the_recorded_mark(self):
        self.assertAlmostEqual(host_ledger.reap_mark_gib(None), CONST, places=6)
        self.assertAlmostEqual(host_ledger.reap_mark_gib(
            int(118.05 * GIB), "FALLBACK lxcfs MemTotal (memory.max is 'max')"), CONST, places=6)

    def test_docker_memory_max_is_the_mark(self):
        self.assertAlmostEqual(host_ledger.reap_mark_gib(84 * GIB, "cgroup memory.max"), 84.0, places=6)
        # the front reads memory.max directly (no label): same rule
        self.assertAlmostEqual(host_ledger.reap_mark_gib(84 * GIB), 84.0, places=6)

    def test_a_ceiling_above_the_host_mark_does_not_raise_it(self):
        self.assertAlmostEqual(host_ledger.reap_mark_gib(200 * GIB, "cgroup memory.max"), CONST, places=6)


class TestChooseUnderMemoryMax(CustomTestCase):
    def _lines(self, ceiling, source):
        with mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=_live()):
            _arm, _h, lines = host_ledger.choose(
                int(125.70 * GIB), int(101.96 * GIB), **RING_KW, ring_provenance="test",
                cg_current_bytes=int(1.09 * GIB), cg_ceiling_bytes=ceiling,
                cg_ceiling_source=source, cg_oom_kill=0)
        return lines

    def _mark_and_bound(self, lines):
        w = [ln for ln in lines if ln.startswith("WEG2-HOST WATERMARK=")]
        self.assertEqual(len(w), 1, lines)
        m = re.search(r"WATERMARK=([0-9.]+) GiB .*BOOT bound = ([0-9.]+) GiB \(margin ([0-9.]+)", w[0])
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    def test_docker_grades_against_84(self):
        mark, bound, margin = self._mark_and_bound(self._lines(84 * GIB, "cgroup memory.max"))
        self.assertAlmostEqual(mark, 84.0, places=2)
        self.assertAlmostEqual(bound, 84.0 - margin, places=2)

    def test_ct999_grades_against_95_90(self):
        mark, _b, _m = self._mark_and_bound(
            self._lines(int(118.05 * GIB), "FALLBACK lxcfs MemTotal (memory.max is 'max')"))
        self.assertAlmostEqual(mark, round(CONST, 2), places=2)

    def test_the_front_line_reads_memory_max_itself(self):
        with mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=_live(84.0)):
            ln = host_ledger.watermark_provenance()
        self.assertIn("WATERMARK=84.00", ln)
        with mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=_live(None)):
            ln = host_ledger.watermark_provenance()
        self.assertIn(f"WATERMARK={CONST:.2f}", ln)


class TestColdTierTerm(CustomTestCase):
    def _arm(self, gib):
        return host_ledger.price(int(125.70 * GIB), int(101.96 * GIB), 1, 600, **RING_KW,
                                 cg_current_bytes=int(1.09 * GIB), cg_ceiling_bytes=84 * GIB,
                                 cold_tier_shm_gib=gib)

    def test_the_post_rides_the_run_peak_and_the_arm_line(self):
        a0, a1 = self._arm(0.0), self._arm(39.20)
        self.assertAlmostEqual(a1.predicted_run_peak_gib() - a0.predicted_run_peak_gib(), 39.20, places=2)
        self.assertAlmostEqual(a1.terms["cold_tier_shm_gib"], 39.20, places=2)
        self.assertIn("cold_tier_shm=39.20", host_ledger.arm_terms_line(a1))
        self.assertNotIn("cold_tier_shm=", host_ledger.arm_terms_line(a0))

    def test_every_pricing_path_takes_the_keyword(self):
        for fn in (host_ledger.charge_terms, host_ledger.price, host_ledger.choose,
                   launcher.choose_host_ledger):
            self.assertIn("cold_tier_shm_gib", inspect.signature(fn).parameters, fn.__name__)
        self.assertIn("group_envs", inspect.signature(launcher.choose_host_ledger).parameters)


class TestColdTierPost(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = os.path.join(self.tmp.name, "nf-experts", "fnFL2")
        os.makedirs(self.store)
        self.karte = os.path.join(self.tmp.name, "expert_map.json")
        with open(self.karte, "w") as fh:
            json.dump({"slots": Z30W_STORE_SLOTS, "p_layer_stage": [0] * 29 + [1] * 11 + [2] * 8}, fh)
        self.ns = SimpleNamespace(env_d=f"SGLANG_MOE_EXPERT_STORE_DIR={self.store}", env_p="",
                                  _expert_row_mib=Z30W_ROW_MIB)

    def tearDown(self):
        self.tmp.cleanup()

    def _mounts(self, fs):
        return (f"rootfs / overlay rw 0 0\n"
                f"tmpfs {os.path.join(self.tmp.name, 'nf-experts')} {fs} rw 0 0\n")

    def test_a_tmpfs_store_is_the_map_size(self):
        gib, line = launcher.cold_tier_shm_post(self.ns, "unused", karte_path=self.karte,
                                                mounts_text=self._mounts("tmpfs"))
        self.assertAlmostEqual(gib, 40142.0 / 1024.0, places=2)   # 39.20 GiB, the z30w map
        self.assertIn(launcher.COLD_TIER_MARKER, line)
        self.assertIn("slots 346 x 48 layers", line)

    def test_a_disk_store_is_no_post(self):
        gib, line = launcher.cold_tier_shm_post(self.ns, "unused", karte_path=self.karte,
                                                mounts_text=self._mounts("xfs"))
        self.assertEqual(gib, 0.0)
        self.assertIn("reclaimable", line)

    def test_bytes_already_resident_are_credited(self):
        with open(os.path.join(self.store, "rows.bin"), "wb") as fh:
            fh.write(b"\1" * (8 * MIB))
        gib, _line = launcher.cold_tier_shm_post(self.ns, "unused", karte_path=self.karte,
                                                 mounts_text=self._mounts("tmpfs"))
        self.assertAlmostEqual(gib, (40142.0 - 8.0) / 1024.0, delta=0.01)

    def test_the_group_env_names_the_store_not_the_launcher_env(self):
        with mock.patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STORE_DIR": "/elsewhere"}):
            self.assertEqual(launcher.expert_store_dir_of(self.ns), self.store)
        ns = SimpleNamespace(env_d="", env_p="")
        with mock.patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STORE_DIR": ""}):
            gib, line = launcher.cold_tier_shm_post(ns, "unused")
        self.assertEqual(gib, 0.0)
        self.assertIn("no expert store", line)


class TestArenaFromTheGroupEnv(CustomTestCase):
    def test_group_env_wins_and_the_larger_group_prices(self):
        with mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_GIB": "22"}):
            self.assertEqual(launcher._group_env_value(
                "SGLANG_HICACHE_ARENA_GIB", [{"SGLANG_HICACHE_ARENA_GIB": "4"}, {}], "22"), "4")
            self.assertEqual(launcher._group_env_value(
                "SGLANG_HICACHE_ARENA_GIB", [{"SGLANG_HICACHE_ARENA_GIB": "4"},
                                             {"SGLANG_HICACHE_ARENA_GIB": "6"}], "22"), "6")
            self.assertEqual(launcher._group_env_value("SGLANG_HICACHE_ARENA_GIB", [{}, {}], "8"), "22")

    @unittest.skipUnless(os.path.exists(os.path.join(NF_MODEL, "config.json")), "NF checkpoint absent")
    def test_z30w_arena_prices_what_dev_shm_held(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_HICACHE_DRAFT_TIER": "off",
                                          "SGLANG_WEG2_DRAFT_ON_P": "0"}):
            t = launcher._weg2_arena_ledger_terms(NF_MODEL, [dict(Z30W_ARENA_ENV), dict(Z30W_ARENA_ENV)])
        # KV 5461 x 786432 B = 4.00 GiB (arena-786432.bin 4.03 on /dev/shm) + mamba 32 slots
        # (arena-58834944.bin 1.75): 5.75 GiB, against 23.75 priced at z30w
        self.assertAlmostEqual(t["arena_gib"], 5.75, delta=0.05)


class TestZ30wComposition(CustomTestCase):
    """What the corrections book on the z30w reading, and what stays open."""

    @unittest.skipUnless(os.path.exists(Z30W_FRONT), "z30w evidence absent")
    def test_the_reading(self):
        text = open(Z30W_FRONT, errors="replace").read()
        pred = float(re.search(r"RUN-PEAK ADVISORY: this arm predicts non-reclaimable "
                               r"memory.current=([0-9.]+) GiB", text).group(1))
        arena_old = float(re.search(r"WEG2-HOST-LEDGER ARM S=1 M=600 .*? arena=([0-9.]+)", text).group(1))
        self.assertAlmostEqual(pred, 47.72, places=2)
        self.assertAlmostEqual(arena_old, 23.75, places=2)
        store = Z30W_STORE_SLOTS * 48 * Z30W_ROW_MIB / 1024.0
        corrected = pred - arena_old + 5.75 + store
        measured = 79.46   # creep_0929.jsonl 09:12:00Z: memory.current - file cache
        # the booked corrections: +21.2 GiB (68.9 against 47.72)
        self.assertAlmostEqual(corrected, 68.92, delta=0.1)
        # NOT closed: ~10.5 GiB stay unbooked -- non-rank anon (python 5.23 +
        # detokenizer 2.18 GiB pss_anon) and shmem outside store/arena
        # (weg2-seq 1.50, QSA sidecar 0.26, handoff 0.23, ~3 GiB unattributed).
        self.assertAlmostEqual(measured - corrected, 10.5, delta=0.2)
        # the verdict the mark changes: 84 - margin, not 95.90 - margin
        self.assertLess(host_ledger.reap_mark_gib(84 * GIB, "cgroup memory.max"), CONST - 11.8)


if __name__ == "__main__":
    unittest.main()
