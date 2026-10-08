# SPDX-License-Identifier: Apache-2.0
"""z30y W87/W20 and 27B z30y W97 (2026-09-29 13:01/13:02Z): two follow-ups of fe3cdc442d.

1. THE TMPFS EXPERT STORE COUNTS ONCE. fe3cdc442d booked it (COLD_TIER_SHM,
   38.97 GiB) in ``_boot_charges_gib`` -- the run peak against memory.max AND
   the two moment leftovers above FLOOR (16) and the CLI reserve (10). z30y:
   run peak 77.98 vs hard bound 82.53 (funded), launch -16.61 / run -19.98
   (refused): the same bytes bound twice. MEASURED z30x2 (mem csv 12:48-12:57Z,
   memory.max 84): peak non-reclaimable 79.54 GiB (shmem 53.29, anon 25.45) --
   the form ran. The store stays in the run peak (where its pages land) and
   leaves the moments, as #1236 took the disk store out of them. 27B: post 0,
   every number byte-identical.

2. THE LATCH IS DERIVED FROM THE MARK. Since the mark follows memory.max, the
   27B profile's fixed --host-riegel-gib 93.0 (written for the 95.90 CT999
   mark) sits above the 74.53 GiB bound of a 76 GiB container and W97 refused
   the boot. The latch is now mark minus the measured margin; a flag may only
   tighten it, a flag above the bound is replaced and named in one line.

3. THE ARENA TERM AND THE RANKS COUNT FROM ONE SOURCE (W3 arena8 probe
   09291303 on z30x2 424346f693: ledger arena=23.75 from the launcher default
   22, ranks got FLLIPER_HICACHE_ARENA_GIB=8 per group). fe3cdc442d already
   reads the group env (z30y ARM arena=5.75 at 4 GiB); the slot rule is now
   one function for both, and a ledger-derived arena size (#1453) is written
   into the group env that names the arena, where build_env gives it to the
   ranks -- before, the re-price read the old group value.
"""
from __future__ import annotations

import csv
import json
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import launcher
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = int(host_ledger.GIB)
MIB = 1 << 20
RING_KW = dict(ring_bytes=4096 * MIB, ring_span1_bytes=4096 * MIB)
Z30Y_COLD = 38.97
Z30X2_PEAK, Z30X2_SHMEM = 79.54, 53.3  # [M] z30x2 kvdemand dauer 09291223, mem csv


def _arm(cold, ceiling=84 * GIB):
    return host_ledger.price(int(125.70 * GIB), int(100.64 * GIB), 1, 600, **RING_KW,
                             cg_current_bytes=int(1.18 * GIB), cg_ceiling_bytes=ceiling,
                             cold_tier_shm_gib=cold)


class TestStoreCountsOnce(CustomTestCase):
    def test_the_moments_do_not_charge_the_store(self):
        a0, a1 = _arm(0.0), _arm(Z30Y_COLD)
        self.assertAlmostEqual(a1.launch_leftover_gib, a0.launch_leftover_gib, places=6)
        self.assertAlmostEqual(a1.run_leftover_gib, a0.run_leftover_gib, places=6)

    def test_the_run_peak_still_does(self):
        a0, a1 = _arm(0.0), _arm(Z30Y_COLD)
        self.assertAlmostEqual(a1.predicted_run_peak_gib() - a0.predicted_run_peak_gib(),
                               Z30Y_COLD, places=2)
        self.assertIn("cold_tier_shm=38.97", host_ledger.arm_terms_line(a1))

    def test_without_a_store_nothing_moves(self):
        # 27B: no expert store, post 0 -- the arm is the arm it was
        a = _arm(0.0)
        self.assertEqual(a.terms["cold_tier_shm_gib"], 0.0)

    def test_z30x2_metal_bounds_the_post(self):
        """Post + base <= measured: the run peak z30y predicts for the z30x2 form
        (77.98, printed by the refused launch) is not above what z30x2 measured.
        [M] pinned (3f9ca8bf39): peak non-reclaimable 79.54 GiB at shmem 53.3 --
        the csv itself was rewritten by the 15:34Z boot of the same profile."""
        peak, shmem = Z30X2_PEAK, Z30X2_SHMEM
        self.assertGreater(shmem, Z30Y_COLD)  # the store's pages are in the reading
        self.assertLessEqual(77.98, peak + 0.01)
        self.assertLess(peak, 84.0)  # and the form ran under memory.max


class TestRiegelDerived(CustomTestCase):
    def test_a_flag_above_the_bound_takes_the_derivation(self):
        v, line = host_ledger.effective_riegel_gib(93.0, 74.53)
        self.assertAlmostEqual(v, 74.53, places=6)
        self.assertIn(host_ledger.RIEGEL_MARKER, line)
        self.assertIn("93.00", line)

    def test_a_flag_below_only_tightens(self):
        self.assertEqual(host_ledger.effective_riegel_gib(73.0, 82.53), (73.0, ""))
        self.assertEqual(host_ledger.effective_riegel_gib(None, 82.53), (None, ""))

    def _choose(self, max_gib, riegel):
        live = {"current_gib": 1.18, "nonreclaim_gib": 0.79, "file_reclaimable_gib": 0.39,
                "source": "test", "max_gib": max_gib}
        with mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=live):
            return host_ledger.choose(
                int(125.70 * GIB), int(100.64 * GIB), **RING_KW, ring_provenance="test",
                cg_current_bytes=int(1.18 * GIB), cg_ceiling_bytes=int(max_gib * GIB),
                cg_ceiling_source="cgroup memory.max", cg_oom_kill=0,
                deviation_reason="27B z30y fixed latch", guard_gib=riegel)

    def test_27b_container_76_takes_the_derivation_instead_of_w97(self):
        arm, _h, lines = self._choose(76.0, 93.0)
        rl = [ln for ln in lines if ln.startswith(host_ledger.RIEGEL_MARKER)]
        self.assertEqual(len(rl), 1, lines)
        self.assertLess(arm.riegel_effective, 76.0)
        self.assertLessEqual(arm.riegel_effective, 76.0 - 1.0)  # mark minus a margin

    def test_nf_container_84_same_rule(self):
        arm, _h, _lines = self._choose(84.0, 93.0)
        self.assertLess(arm.riegel_effective, 84.0)

    def test_the_launcher_feeds_the_arena_the_derived_latch(self):
        import inspect

        self.assertIn('getattr(arm, "riegel_effective"', inspect.getsource(launcher))


NF_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


class TestArenaOneSource(CustomTestCase):
    def test_ranks_and_ledger_share_the_slot_rule(self):
        from flliper.srt.mem_cache.pool_host import arena_pool

        page = 786432
        with mock.patch.dict(os.environ, {"FLLIPER_HICACHE_ARENA_GIB": "8"}, clear=False):
            os.environ.pop(arena_pool.ENV_ARENA_KV_PAGE_BYTES, None)
            self.assertEqual(arena_pool.planned_arena_slots(page), arena_pool.arena_slots_for(8.0, page))
        self.assertEqual(arena_pool.arena_slots_for(8.0, page), 10922)
        self.assertEqual(arena_pool.arena_slots_for(0.001, page), 1024)
        import inspect

        self.assertIn("arena_slots_for(kv_gib", inspect.getsource(launcher._pdflip_arena_ledger_terms))

    def test_a_ledger_arena_size_reaches_the_group_that_names_it(self):
        ns = type("NS", (), {})()
        ns.env_d = "FLLIPER_HICACHE_ARENA_MAMBA_SLOTS=32;FLLIPER_HICACHE_ARENA_GIB=8"
        ns.env_p = "FLLIPER_MOE_SCRATCH_SLOTS=74,48,48"
        groups = [launcher.parse_group_env(ns.env_d), launcher.parse_group_env(ns.env_p)]
        done = launcher.arena_gib_into_groups(ns, groups, 6.5)
        self.assertEqual(done, ["env_d"])
        self.assertEqual(launcher.parse_group_env(ns.env_d)["FLLIPER_HICACHE_ARENA_GIB"], "6.5")
        self.assertEqual(launcher.parse_group_env(ns.env_d)["FLLIPER_HICACHE_ARENA_MAMBA_SLOTS"], "32")
        self.assertEqual(ns.env_p, "FLLIPER_MOE_SCRATCH_SLOTS=74,48,48")
        self.assertEqual(launcher._group_env_value("FLLIPER_HICACHE_ARENA_GIB", groups, "22"), "6.5")

    def test_the_launcher_writes_the_resize_into_the_groups(self):
        import inspect

        self.assertIn("arena_gib_into_groups(ns, _ledger_group_envs, _arena_new)", inspect.getsource(launcher))

    @unittest.skipUnless(os.path.exists(os.path.join(NF_MODEL, "config.json")), "NF checkpoint absent")
    def test_arena8_group_env_prices_8_not_22(self):
        env = {"FLLIPER_HICACHE_ARENA_GIB": "8", "FLLIPER_HICACHE_ARENA_MAMBA_SLOTS": "32"}
        t8 = launcher._pdflip_arena_ledger_terms(NF_MODEL, [dict(env), dict(env)])
        t22 = launcher._pdflip_arena_ledger_terms(
            NF_MODEL, [{"FLLIPER_HICACHE_ARENA_GIB": "22", "FLLIPER_HICACHE_ARENA_MAMBA_SLOTS": "32"}])
        self.assertLess(t8["arena_gib"], 12.0)
        # 14 GiB of KV slots apart (plus their draft cells when the draft tier is on)
        self.assertGreaterEqual(t22["arena_gib"] - t8["arena_gib"], 14.0 - 0.01)
        self.assertLess(t22["arena_gib"] - t8["arena_gib"], 14.0 * 1.25)


R27B = "/spinning/docker-acceptance/27b/evidence/weg2_measured_record.json"
CENSUS27B = "/spinning/docker-acceptance/27b/evidence/host_census_record.json"
# [M] mem_rowauthorityz30ybar1fs.csv, boot 09291331: max(cg_current - file) =
# 62.29 GiB at 13:36:49Z. The mem_<profile>.csv is rewritten by the next boot
# of the same profile (z30x2's was, 15:34Z), so the reading is pinned here.
Z30Y_27B_PEAK = 62.29
CENSUS_27B = {"nonrank_anon_gib": 7.20, "seq_ring_gib": 3.00,   # record 13:37:58Z
              "arena_sidecar_gib": 0.0, "arena_handoff_gib": 0.0}


class TestCensusCountsOnce(CustomTestCase):
    """27B d2 13:52Z (z30y): W97 run peak 86.19 vs 65.53 at 13:23Z; the boot
    measured at most 60.85 non-reclaimable. The 13:31 boot's own run residual
    (10.53 GiB, sampled with census posts 0) already holds front/detokenizer/
    compile-worker anon and the lane ring -- and d2 charged them again."""

    def _entry(self, resid, census=None):
        e = {"run_residual_gib": resid, "sampled_at_flip_epoch": 0, "boot_tag": "b",
             "pids": [1], "at": "t"}
        if census is not None:
            e["residual_census_gib"] = census
        return {"P": e}

    def test_a_residual_sampled_without_the_census_gives_it_back(self):
        o, src = host_ledger.run_origin_gib(0.77, self._entry(10.53), census_now_gib=10.20)
        self.assertAlmostEqual(o, 0.77, places=2)  # 10.53 - 10.20 = 0.33 < launch 0.77
        o2, _ = host_ledger.run_origin_gib(0.77, self._entry(15.0), census_now_gib=10.20)
        self.assertAlmostEqual(o2, 4.80, places=2)

    def test_a_residual_that_already_subtracted_them_is_left_alone(self):
        o, _ = host_ledger.run_origin_gib(0.77, self._entry(10.53, census=10.20), census_now_gib=10.20)
        self.assertAlmostEqual(o, 10.53, places=2)

    def test_no_census_no_change(self):
        o, _ = host_ledger.run_origin_gib(0.77, self._entry(10.53))
        self.assertAlmostEqual(o, 10.53, places=2)

    def test_the_sampler_stores_what_it_subtracted(self):
        import inspect

        self.assertIn('"residual_census_gib": _census_gib', inspect.getsource(host_ledger))

    @unittest.skipUnless(os.path.exists(R27B) and os.path.exists(CENSUS27B), "27B z30y evidence absent")
    def test_27b_d2_replay_brackets_its_own_measured_peak(self):
        """Operator 29.09., two-sided: measured peak <= prediction(record
        09291331) <= measured peak + a small, justified margin. ONE currency:
        memory.current - inactive_file - active_file (the level the reap mark
        and the launch origin 0.77 are read in); the front's 60.85 is the
        post-flip anon+shmem+slab_unreclaimable POINT reading (ratchet
        instrument), not the peak. Upper margin 1.0 = the L3 index priced 0.82
        against 0.125 measured at 13:37:58Z (priced for its growth) + one 2-s
        step of the csv (0.25). Red on 1b8d1f25fd (75.80, upper) and on
        b03e724bda (60.05, lower: arena file 31.41 vs 30.18 priced, 1.56 GiB
        other_tmpfs+anon_shared no post booked). The d2 arm replayed term by term (TERMS line 13:52:11Z:
        heaps 17.23, anchors 0.40, rings 0.22, arena 30.18, l3 0.82, overhead
        0.02, draft pools 0.17, ratchet 0.03, priced bounce 15.75) with the REAL
        census record (13:37:58Z) and the REAL residual record. Measured: peak
        non-reclaimable 62.29 GiB at 13:36:49Z (cg_current - file, mem csv).
        Margin 0.25 GiB = one 2-s sample step of the csv at the peak slope
        (13:36:43 -> :49 +0.23). Red on 1b8d1f25fd (75.8: the 15.75 bounce
        region that no rank maps -- census xchg 0.0006 -- beside the census
        posts that measure the bytes it was a stand-in for)."""
        from flliper.srt.pdflip import host_census

        entry = next(iter(json.load(open(CENSUS27B)).values()))
        census = host_census.ledger_terms(entry)
        images = host_ledger.ImageTerms(0.0, 0.0, "t", "t", True, True, 0.0, 0.0)
        t = host_ledger.charge_terms(1, 600, 3, images, s_gb_d=4, staging_gb=0.0494, anchor_mib=101,
                                     xchg_bounce_host_bytes=int(15.75 * GIB),
                                     flip_ratchet_gib=0.03, arena_gib=30.18,
                                     l3_index_gib=0.82, census=census)
        rec = host_ledger.read_measured_record(R27B, boot_tag="dkr27browauthorityz30ybar1fs09291331")
        origin, _src = host_ledger.run_origin_gib(0.77, rec, census_now_gib=host_ledger.census_posts_gib(t))
        predicted = origin + host_ledger._run_moment_charges_gib(t)
        self.assertAlmostEqual(t["anchors_gib"], 0.40, places=2)  # the TERMS line, field by field
        self.assertAlmostEqual(t["rings_gib"], 0.22, places=2)
        self.assertAlmostEqual(t["heaps_gib"], 17.23, places=1)
        peak = Z30Y_27B_PEAK
        self.assertGreaterEqual(predicted, peak, (predicted, t.get("arena_census_excess_gib"),
                                                  t.get("unposted_shm_gib")))
        self.assertLessEqual(predicted, peak + 1.0, (predicted, t["xchg_bounce_gib"]))

    def test_measured_shm_posts_count_once(self):
        terms = {"arena_gib": 30.18, "anchors_gib": 0.40, "rings_gib": 0.22, "overhead_gib": 0.025,
                 "draft_host_p_gib": 0.116, "draft_host_d_gib": 0.058}
        p = host_ledger.census_shm_posts(terms, {"arena_measured_gib": 31.41, "other_tmpfs_gib": 0.89,
                                                 "unbooked_shm_gib": 1.49})
        self.assertAlmostEqual(p["arena_census_excess_gib"], 1.23, places=2)
        self.assertAlmostEqual(p["unposted_shm_gib"], 0.89 + 1.49 - 0.819, places=2)
        # a smaller measured arena is another arm, never a refund; no record -> 0
        self.assertEqual(host_ledger.census_shm_posts(terms, {"arena_measured_gib": 5.0})["arena_census_excess_gib"], 0.0)
        self.assertEqual(host_ledger.census_shm_posts(terms, None), {"arena_census_excess_gib": 0.0,
                                                                   "unposted_shm_gib": 0.0})

    def test_the_priced_bounce_stands_without_a_census(self):
        self.assertAlmostEqual(host_ledger.xchg_carrier_gib(int(15.75 * GIB), None), 15.75, places=6)
        self.assertAlmostEqual(host_ledger.xchg_carrier_gib(int(15.75 * GIB), {"xchg_measured_gib": None}), 15.75, places=6)
        self.assertAlmostEqual(host_ledger.xchg_carrier_gib(int(15.75 * GIB), {"xchg_measured_gib": 0.0006}), 0.0006, places=6)
        self.assertEqual(host_ledger.xchg_carrier_gib(0, {"xchg_measured_gib": 0.2}), 0.0)  # NF: no bounce arm

    @unittest.skipUnless(os.path.exists(R27B), "27B measured record absent")
    def test_27b_z30y_replay_prediction_below_measured_plus_margin(self):
        """The real record: the census taken out of the floor brings the d2
        origin back to the launch reading; predicted = 65.53 (13:31) + census
        10.20 + l3 +0.09 = 75.82 against the refused 86.19, and the census posts
        (not the floor) now carry those bytes. Measured peak 60.85 + the posts
        the 13:31 boot had not yet charged is the bound."""
        rec = host_ledger.read_measured_record(R27B, boot_tag="dkr27browauthorityz30ybar1fs09291331")
        o_old, _ = host_ledger.run_origin_gib(0.77, rec)
        o_new, src = host_ledger.run_origin_gib(0.77, rec, census_now_gib=sum(CENSUS_27B.values()))
        self.assertGreater(o_old, 10.0)       # the double count: residual holds the census
        self.assertLess(o_new, 1.0, src)      # once: the posts carry it, the floor does not
        # the prediction bound itself is test_27b_d2_replay_is_below_its_own_measured_peak


if __name__ == "__main__":
    unittest.main()
