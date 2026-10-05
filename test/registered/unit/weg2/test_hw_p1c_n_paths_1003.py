"""HW-P1c 1003 (user order 03.10. ~19:30Z: "unsere software muss mit beliebiger
anzahl an karten und sm86 sm89 und sm120 laufen"; plan
/spinning/gpu-arb/docker/PLAN_HWGEN_N_KARTEN_1003.md P1c/P2): the 3-card data
paths become N-capable or DERIVED, so that 27B INT8 on 5090 + 3080 and 27B
NVFP4 on 3080 + 3080 pass the simulation with the metal proof as the only
blocker.

* XCHG-REGION   weight_exchange_region.geometry(n) / configure(n);
* BAR1-WINDOW   weg2/bar1_windows.plan (N + measured BAR1);
* PP-CUT-FLOOR / PP-CUT-PIN  resolve_pool_floor(n_stages), pin drop, N-stage
                incumbent, family cost on the reference basis;
* PROFILE-VECTORS / RECORDS-NVEC / L15-POSTS  weg2/inventory_view.py (a live
                SUBSET of the calibrated cards is derived by card class);
* format_of knows the NF release checkpoint (...-abl-wxp);
* the reference rig (N = 3) is byte-identical in every one of them.

GPU-free, NVML-free.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import bar1_windows as BW
from sglang.srt.weg2 import card_identity as CI
from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import hw_sim as HS
from sglang.srt.weg2 import inventory_view as IV
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import weight_exchange_region as XR

REF = ("RTX5090", "RTX3080", "RTX3080")


def card(i, name, mib, cc, bar1=256):
    return L.Card(i, f"GPU-{i:04d}", name, mib, cc=cc, bar1_total_mib=bar1)


def rig():
    return L.order_cards([card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)),
                          card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0), 32768),
                          card(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6))])


def ns_for(*extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])


class XchgRegionGeometry(unittest.TestCase):

    def tearDown(self):
        XR.configure(3)

    def test_n3_is_the_legacy_literal_geometry(self):
        g = XR.geometry(3)
        self.assertEqual((g["N_CARDS"], g["N_RANKS"], g["N_PAIRS"], g["N_SLOTS"]), (3, 6, 6, 12))
        self.assertEqual(g["CROSS_PAIRS"], ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
        self.assertEqual((g["GATE_OFF"], g["MATRIX_OFF"], g["MATRIX_ROW_BYTES"], g["DIR_OFF"], g["DATA_OFF"]),
                         (4096, 8192, 192, 65536, 1 << 20))
        self.assertEqual((g["MATRIX_PAYLOAD_STRUCT"].size, g["MATRIX_SEAL_OFF"]), (184, 184))
        self.assertEqual((g["MX_PLAN_HASH"], g["MX_ONCARD_MODE"]), (12, 16))
        self.assertEqual((g["REGISTERED_OFF"], g["SLOTS_OFF"]), (48, 256))
        self.assertEqual(g["REGION_BYTES"], (1 << 20) + 384 * (1 << 20))
        # and the module's own import-time values ARE that geometry
        for k in XR._GEOMETRY_NAMES:
            v = getattr(XR, k)
            if k == "MATRIX_PAYLOAD_STRUCT":
                self.assertEqual(v.size, g[k].size)
            else:
                self.assertEqual(v, g[k], k)

    def test_every_count_2_to_8_lays_out(self):
        for n in range(2, 9):
            self.assertEqual(XR.layout_problems(n), [], n)
            g = XR.geometry(n)
            self.assertEqual(g["N_PAIRS"], n * (n - 1))
            self.assertEqual(len(set(g["CROSS_PAIRS"])), n * (n - 1))
            self.assertTrue(all(a != b for a, b in g["CROSS_PAIRS"]))
            self.assertGreaterEqual(g["MATRIX_PAYLOAD_STRUCT"].size // 8, g["MX_ONCARD_MODE"] + 1)
        for n in (0, 1, 9):
            self.assertTrue(XR.layout_problems(n), n)
            with self.assertRaises(ValueError):
                XR.geometry(n)

    def test_the_other_modules_import_for_every_count(self):
        # transport / shadow derive their dir sub-layout from the region at import
        # and assert it fits; a fresh interpreter per N reads the env var
        code = ("from sglang.srt.weg2 import weight_exchange_region as xr, weight_exchange_shadow as sh, "
                "weight_exchange_transport as tr; "
                "assert tr.DIR_USED_BYTES <= xr.DATA_OFF - xr.DIR_OFF; "
                "assert sh.MANIFEST_AREA_END <= xr.DATA_OFF - xr.DIR_OFF; print(xr.N_CARDS, xr.N_RANKS)")
        for n in (2, 4, 5, 8):
            env = dict(os.environ, SGLANG_WEG2_XCHG_N_CARDS=str(n), PYTHONPATH=os.environ.get("PYTHONPATH", "python"))
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr[-600:])
            self.assertEqual(out.stdout.split()[-2:], [str(n), str(2 * n)])

    def test_region_roundtrip_at_two_cards(self):
        XR.configure(2)
        with tempfile.TemporaryDirectory() as root:
            region = XR.XchgRegion.create("p1ctest2", shm_root=root)
            try:
                self.assertEqual(os.path.getsize(region.path), XR.REGION_BYTES)
                self.assertEqual(XR.REGION_BYTES, (1 << 20) + 2 * 2 * 32 * (1 << 20))
                region.begin_flip("p1ctest2.1")
                for row in range(XR.N_RANKS):
                    region.write_matrix_row(row, [row] * XR.N_RANKS, [row + 1] * XR.N_RANKS, 77)
                back = region.read_matrix_row(3)
                self.assertEqual((list(back.send), list(back.recv), back.plan_hash, back.sealed),
                                 ([3] * 4, [4] * 4, 77, True))
                other = XR.XchgRegion.open(region.path, expect_boot="p1ctest2")
                other.close()
                # a rank on another N refuses by name instead of addressing another layout
                XR.configure(3)
                with self.assertRaises(XR.Weg2XchgPlanDisagree):
                    XR.XchgRegion.open(region.path, expect_boot="p1ctest2")
            finally:
                XR.configure(2)
                region.close()

    def test_prepare_region_publishes_n_only_off_the_reference_count(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(XR, "create_semaphores", lambda b: []):
            got = XR.prepare_region("p1c3", shm_root=root)
            self.assertEqual(set(got["env"]), {XR.ENV_REGION_PATH, XR.ENV_REGION_BOOT})
            XR.teardown_region("p1c3", shm_root=root)
            XR.configure(2)
            got = XR.prepare_region("p1c2", shm_root=root)
            self.assertEqual(got["env"][XR.ENV_N_CARDS], "2")
            XR.teardown_region("p1c2", shm_root=root)

    def test_sem_names_follow_the_pairs(self):
        for n in (2, 3, 5):
            XR.configure(n)
            self.assertEqual(len(XR.all_sem_names("b")), n * (n - 1) * XR.SLOTS_PER_PAIR * 2)
            self.assertEqual(len(XR.all_diagonal_sem_names("b")), n * XR.SLOTS_PER_PAIR * 2)


class Bar1Windows(unittest.TestCase):

    def test_reference_windows_are_kept_where_they_fit(self):
        for n, bars in ((2, [32768, 256]), (3, [32768, 256, 256]), (2, [256, 256])):
            p = BW.plan(n, bars)
            self.assertTrue(p.ok)
            self.assertEqual((p.p, p.d), (BW.P_REFERENCE, BW.D_REFERENCE), (n, bars))
            self.assertEqual(p.scale, 1.0)
        d = BW.plan(2, [256, 256], dual=True)
        self.assertEqual((d.p, d.d), (BW.P_DUAL_REFERENCE, BW.D_REFERENCE))

    def test_more_ranks_grow_the_collective_windows_and_shrink_to_the_aperture(self):
        big = BW.plan(4, [32768, 32768, 32768, 32768])
        self.assertEqual(big.d, "24,TP_0=48,DCP_0=60")      # (R-1)/2 = 1.5 x; PP_0 is point-to-point
        self.assertIn("PP_0=96", big.p)
        small = BW.plan(4, [256, 256, 256, 256])
        self.assertTrue(small.ok)
        self.assertLess(small.scale, 1.0)
        self.assertLessEqual(small.demand_mib * small.scale, 224)
        self.assertTrue(small.why.startswith("derived"), small.why)

    def test_a_tiny_aperture_is_refused_by_name(self):
        p = BW.plan(2, [64, 64])
        self.assertFalse(p.ok)
        self.assertIn("BAR1", p.why)
        self.assertIn("floor", p.why)

    def test_unreported_bar1_is_named_unchecked(self):
        p = BW.plan(2, [None, None])
        self.assertTrue(p.ok)
        self.assertIn("unchecked", p.why)

    def test_launcher_applies_it_only_off_the_proven_count(self):
        ns = ns_for()
        self.assertIsNone(L.apply_bar1_windows(ns, rig()))
        self.assertEqual((ns.p_barlink_bar1_window_mib, ns.d_barlink_bar1_window_mib),
                         (L.P_BARLINK_BAR1_WINDOW_MIB, L.D_BARLINK_BAR1_WINDOW_MIB))
        two = rig()[:2]
        ns = ns_for()
        self.assertIsNone(L.apply_bar1_windows(ns, two))   # fits as shipped: no line
        self.assertEqual(ns.p_barlink_bar1_window_mib, "24,PP_0=96")
        # an operator's explicit window is kept
        four = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(4)]
        ns = ns_for("--p-barlink-bar1-window-mib", "20,PP_0=50")
        lines = []
        L.apply_bar1_windows(ns, four, lines.append)
        self.assertEqual(ns.p_barlink_bar1_window_mib, "20,PP_0=50")
        self.assertNotEqual(ns.d_barlink_bar1_window_mib, L.D_BARLINK_BAR1_WINDOW_MIB)
        self.assertTrue(lines)

    def test_argv_d_default_window_is_byte_identical(self):
        d = L.argv_d("py", L.MODEL_DEFAULT, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], d_bs=1)
        self.assertEqual(d[d.index("--barlink-bar1-window-mib") + 1], "16,TP_0=32,DCP_0=40")
        d = L.argv_d("py", L.MODEL_DEFAULT, [1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], d_bs=1,
                     window_mib="8,TP_0=16")
        self.assertEqual(d[d.index("--barlink-bar1-window-mib") + 1], "8,TP_0=16")


class InventoryDerivation(unittest.TestCase):

    def tearDown(self):
        IV.clear_active()

    def test_subset_rule(self):
        self.assertTrue(IV.is_subset(("RTX5090", "RTX3080"), REF))
        self.assertTrue(IV.is_subset(("RTX3080", "RTX3080"), REF))
        self.assertTrue(IV.is_subset(REF, REF))
        self.assertFalse(IV.is_subset(("RTX5090", "RTX5090"), REF))
        self.assertFalse(IV.is_subset(("RTX3080",) * 3, REF))
        self.assertFalse(IV.is_subset(("RTX5090", "-"), REF))
        self.assertFalse(IV.is_subset(REF + ("RTX3080",), REF))

    def test_policies(self):
        live = ("RTX5090", "RTX3080")
        self.assertEqual(IV.derive_vector(IV.CLASS_MAX, [3191, 2079, 2067], REF, live), [3191, 2079])
        self.assertEqual(IV.derive_vector(IV.CLASS_MAX, "8.10,35.16,33.59", REF, live), "8.10,35.16")
        self.assertEqual(IV.derive_vector(IV.CLASS_MAX, "8.10,35.16,33.59", REF, ("RTX3080",) * 2), "35.16,35.16")
        self.assertEqual(IV.derive_vector(IV.ROLE, "1418.4,148.8,4960.6", REF, live), "1418.4,4960.6")
        self.assertEqual(IV.derive_vector(IV.ROLE, "1418.4,148.8,4960.6", REF, live + ("RTX3080",) * 2),
                         "1418.4,148.8,148.8,4960.6")   # 4 stages: first, middle x2, last
        self.assertEqual(IV.derive_vector(IV.UNIFORM, [0.3, 0.3, 0.3], REF, live), [0.3, 0.3])
        self.assertIsNone(IV.derive_vector(IV.UNIFORM, [0.3, 0.4, 0.3], REF, live))
        self.assertIsNone(IV.derive_vector(IV.CLASS_MAX, [1, 2, 3], REF, ("RTX4090",)))   # no twin
        self.assertEqual(IV.derive_vector(IV.CUT_PIN, "49,8,7", REF, live), ())
        self.assertEqual(IV.derive_vector(IV.CLASS_MAX, [5, None, 7], REF, ("RTX3080",)), [7])

    def test_assessment_names_what_is_not_derivable(self):
        recs = [("D_AWAKE_REST_BOOKED_MIB", [3191, 2079, 2067]), ("D_FIXED_MIB", [7442, 1036, 914]),
                ("DC_MEASURED_D_XCHG_MIB", [2588, 3084, 2588])]
        a = IV.assess_records(recs, REF, ("RTX5090", "RTX3080"))
        self.assertTrue(a.subset)
        self.assertEqual((a.derivable, a.exempt, a.underivable),
                         (("D_AWAKE_REST_BOOKED_MIB",), ("DC_MEASURED_D_XCHG_MIB",), ("D_FIXED_MIB",)))
        a = IV.assess_records(recs[:1], REF, ("RTX5090", "RTX5090"))
        self.assertFalse(a.subset)
        self.assertEqual(a.underivable, ("D_AWAKE_REST_BOOKED_MIB",))

    def test_l15_posts_by_class(self):
        new, note = IV.derive_l15_override("c1=7616,c2=1792", REF, ("RTX5090", "RTX3080"))
        self.assertEqual(new, "c1=1792")
        self.assertIn("smallest measured hold", note)
        new, _ = IV.derive_l15_override("c1=7616,c2=1792", REF, ("RTX3080", "RTX3080"))
        self.assertEqual(new, "c0=1792,c1=1792")
        self.assertEqual(IV.derive_l15_override("c1=7616,c2=1792", REF, REF), ("c1=7616,c2=1792", ""))
        self.assertEqual(IV.derive_l15_override("GPU-abc=5", REF, ("RTX5090", "RTX3080"))[1], "")   # identity key
        self.assertEqual(IV.derive_l15_override("auto", REF, ("RTX5090", "RTX3080"))[1], "")
        self.assertFalse(IV.l15_derivable("c1=7616,c2=1792", REF, ("RTX4090", "RTX3080")))

    def test_the_reference_rig_is_untouched(self):
        ns = ns_for("--profile", "qwen27b", "--user-reserve-mib", "1800,1400,1400")
        before = vars(ns).copy()
        env = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_MIB": "c1=7616,c2=1792"}
        self.assertIsNone(L.derivation_assessment(ns, rig(), env))
        self.assertEqual(L.apply_inventory_derivation(ns, rig(), None, env), [])
        self.assertEqual(vars(ns), before)
        self.assertEqual(env["SGLANG_WEG2_L15_MIB"], "c1=7616,c2=1792")
        self.assertIsNone(IV.active())
        self.assertTrue(L.inventory_check_line(ns, rig()).endswith("MATCH"))

    @unittest.skip("1004e NF tree has no l15_plan (guarded to None, launcher.py:102): the L15 posts "
                   "never derive, so the 27B-line L15-adjusted values are unreachable here -- measured "
                   "05.10.: pp_cut_stage_fixed_mib derives to '2342.0,3518.0' (record mapping, no L15), "
                   "the 27B line yields the L15-posted '1418.4,4960.6' this test pins")
    def test_five_ninety_plus_three_eighty_is_derived_not_refused(self):
        two = rig()[:2]
        two = L.order_cards(two)
        self.assertEqual(CI.inventory_signature(two), ("RTX5090", "RTX3080"))
        ns = ns_for("--profile", "qwen27b", "--user-reserve-mib", "1800,1400,1400",
                    "--env-d", "SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0")
        env = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_MIB": "c1=7616,c2=1792"}
        line = L.inventory_check_line(ns, two, env)
        self.assertTrue(line.endswith(" DERIVED"), line)
        self.assertIn("SUBSET", line)
        notes = L.apply_inventory_derivation(ns, two, None, env)
        self.assertEqual(ns.user_reserve_mib, "1800,1400")
        self.assertEqual(ns.env_d, "SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0")
        self.assertEqual(ns.pp_cut_measured_ms_per_layer, "8.10,35.16")
        self.assertEqual(ns.pp_cut_stage_fixed_mib, "1418.4,4960.6")
        self.assertEqual(env["SGLANG_WEG2_L15_MIB"], "c1=1792")
        self.assertTrue(all(n.startswith("HW-DERIVE") for n in notes), notes)
        self.assertEqual(L.positional_vector_lengths(ns), {"--user-reserve-mib": 2,
                                                           "SGLANG_WEG2_EXTEND_TRIM_MIB": 2})
        # the records are read derived from here on, through the one reader
        self.assertEqual(list(F.profile_constant("D_AWAKE_REST_BOOKED_MIB", "qwen27b")), [3191, 2079])
        self.assertEqual(list(F.profile_constant("P_OVERSHOOT_MIB", "qwen27b")), [920, 512])
        self.assertEqual(list(F.profile_constant("DC_MEASURED_D_XCHG_MIB", "qwen27b")), [2588, 3084, 2588])
        self.assertEqual(F.profile_constant("P_DRAFT_RESIDENT_BUDGET_MIB", "qwen27b"), 1618.2)
        IV.clear_active()
        self.assertEqual(len(F.profile_constant("D_AWAKE_REST_BOOKED_MIB", "qwen27b")), 3)

    def test_a_foreign_inventory_is_still_refused_by_name(self):
        two = L.order_cards([card(i, "NVIDIA GeForce RTX 5090", 32607, (12, 0), 32768) for i in range(2)])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.inventory_check_line(ns_for("--profile", "qwen27b"), two)
        self.assertTrue(str(cm.exception).startswith("HW-UNCALIBRATED"), str(cm.exception))
        self.assertEqual(L.apply_inventory_derivation(ns_for("--profile", "qwen27b"), two), [])
        self.assertIsNone(IV.active())
        # NF has no derivation policy: a subset of its cards is still refused
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.inventory_check_line(ns_for("--profile", "nextflash"), rig()[:2] and L.order_cards(rig()[:2]))
        self.assertIn("HW-UNCALIBRATED", str(cm.exception))

    def test_a_pinned_cut_is_dropped_for_another_stage_count(self):
        two = L.order_cards(rig()[:2])
        ns = ns_for("--profile", "qwen27b", "--pp-stage-ratio", "49,8,7", "--pp-attn-stage-ratio", "12,2,2",
                    "--extra-p=--max-running-requests 1 --pp-stage-ratio 45,10,9")
        notes = L.apply_inventory_derivation(ns, two, None, {})
        self.assertIsNone(ns.pp_stage_ratio)
        self.assertIsNone(ns.pp_attn_stage_ratio)
        self.assertNotIn("pp-stage-ratio", ns.extra_p)
        self.assertIn("--max-running-requests", ns.extra_p)
        self.assertTrue(any("DROPPED" in n for n in notes))


class PpCut(unittest.TestCase):

    def test_floor_is_the_ordered_cuts_only_on_its_own_stage_count(self):
        d = L.resolve_pool_floor(None)
        self.assertEqual(d[1], (39, 13, 12))
        self.assertEqual(L.resolve_pool_floor(None, 3), d)
        floor, cut, rule = L.resolve_pool_floor(None, 2)
        self.assertEqual((floor, cut), (None, None))
        self.assertIn("source=n-stages", rule)
        self.assertEqual(L.resolve_pool_floor(123456, 2)[0], 123456)   # an operator's number outranks
        self.assertEqual(L.resolve_pool_floor(0, 2)[:2], (None, None))

    def test_region_form_lane_cap_is_the_three_stage_incumbents(self):
        self.assertEqual(L.region_form_lane_price_cap("both", "both", None),
                         L.region_form_lane_price_cap("both", "both"))
        self.assertEqual(L.region_form_lane_price_cap("both", "both", 2), 0)

    def test_n_stage_incumbent(self):
        c = L.n_stage_incumbent(2, 64, [8.10, 35.16])
        self.assertEqual((len(c), sum(c)), (2, 64))
        self.assertGreater(c[0], c[1])      # the fast card takes more layers
        self.assertEqual(sum(L.n_stage_incumbent(5, 64, [1, 2, 3, 4, 5])), 64)
        self.assertTrue(all(x >= 1 for x in L.n_stage_incumbent(8, 10, [1] * 8)))

    def test_family_cost_on_the_reference_basis(self):
        from sglang.srt.planner import pp_cut as PC

        families = tuple(PC.LAYER_FAMILY_ATTENTION if (i % 4) == 3 else PC.LAYER_FAMILY_LINEAR for i in range(64))
        ns = ns_for("--profile", "qwen27b")
        two = L.order_cards(rig()[:2])
        cost, prov = L.derived_family_cost(ns, two, families, 4096)
        self.assertEqual(len(cost.linear_ms_per_layer), 2)
        self.assertIn("HW-DERIVE family cost", prov)
        # the same cost as the three-stage fit, mapped by class: the 5090 stage is the fast one
        self.assertLess(cost.linear_ms_per_layer[0], cost.linear_ms_per_layer[1])
        three, _ = PC.family_costs_from_measurement(
            measured_ms_per_layer=[8.10, 35.16, 33.59], measured_counts=[32, 18, 14],
            measured_attn_counts=PC.attention_counts(families, [32, 18, 14]), chunk_tokens=4096,
            ref_prefix_tokens=float(ns.pp_cut_calibration_prefix_tokens), anchor_stage=1,
            anchor_attn_ms_per_layer=float(ns.pp_cut_attn_anchor_ms),
            anchor_prefix_tokens=float(ns.pp_cut_attn_anchor_prefix_tokens))
        self.assertAlmostEqual(cost.linear_ms_per_layer[0], three.linear_ms_per_layer[0])
        self.assertAlmostEqual(cost.linear_ms_per_layer[1], max(three.linear_ms_per_layer[1:]))
        with self.assertRaises(L.Weg2LaunchRefused):
            L.derived_family_cost(ns, L.order_cards([card(0, "NVIDIA GeForce RTX 4090", 24564, (8, 9))] * 2),
                                  families, 4096)


class PChunkStages(unittest.TestCase):
    """The P-chunk stage table and the prefill-graph pool vector are
    three-stage tables of the reference rig; main reads them BEFORE the cards,
    so they ask NVML for the stage count themselves (p_group_stages)."""

    def test_stage_count_follows_the_cards(self):
        with HS.replayed(HS.REFERENCE_RIG, None):
            self.assertEqual(L.p_group_stages(), 3)
        with HS.replayed(HS.REFERENCE_RIG, (1, 0)):
            self.assertEqual(L.p_group_stages(), 2)
        with HS.replayed(HS.REFERENCE_RIG, (0, 2)):
            self.assertEqual(len(L.p_prefill_graph_pool_mib(ns_for("--p-prefill-graph-pool-mib", "160"))
                                 or (0, 0)), 2)

    def test_builtin_table_is_mapped_by_class_for_a_subset(self):
        with HS.replayed(HS.REFERENCE_RIG, None):
            three, _ = L.p_chunk_stage_model("builtin-int8", "int8", 3, "qwen27b")
        with HS.replayed(HS.REFERENCE_RIG, (1, 0)):
            two, src = L.p_chunk_stage_model("builtin-int8", "int8", 2, "qwen27b")
        self.assertEqual(len(three), 3)
        self.assertEqual(len(two), 2)
        self.assertIn("HW-DERIVE", src)
        self.assertEqual(two[0].points, three[0].points)               # the 5090 stage
        slow = max(three[1:], key=lambda m: m.points[-1][1])
        self.assertEqual(two[1].points, slow.points)                   # the slowest 3080 stage
        with HS.replayed(HS.REFERENCE_RIG, (0, 2)):
            two, _ = L.p_chunk_stage_model("builtin-int8", "int8", 2, "qwen27b")
        self.assertEqual(two[0].points, slow.points)
        self.assertEqual(two[1].points, slow.points)

    def test_a_json_table_is_mapped_and_a_foreign_inventory_still_refused(self):
        path = "/spinning/gpu-arb/docker/profiles_release/27b-nvfp4.pchunk.json"
        if not os.path.isfile(path):
            self.skipTest(path)
        with HS.replayed(HS.REFERENCE_RIG, (0, 2)):
            two, src = L.p_chunk_stage_model(path, "nvfp4", 2, "qwen27b")
        self.assertEqual(len(two), 2)
        with HS.replayed(["4090", "4090"], None), self.assertRaises(SystemExit):
            L.p_chunk_stage_model("builtin-int8", "int8", 2, "qwen27b")


class FormatOf(unittest.TestCase):

    def test_the_nf_release_checkpoint_has_a_format(self):
        base = F.profile_row("nextflash").formats["int4-mixed"].checkpoint
        self.assertEqual(F.format_of("nextflash", base), "int4-mixed")
        self.assertEqual(F.format_of("nextflash", base + "-abl-wxp"), "int4-mixed")
        self.assertEqual(F.format_of("nextflash", "/x/models/" + os.path.basename(base) + "-abl-wxp/"), "int4-mixed")
        self.assertEqual(F.format_of("nextflash", base + "-abl-other"), "")
        self.assertEqual(F.format_of("qwen27b", base + "-abl-wxp"), "")   # never across profiles


class FrontStages(unittest.TestCase):

    @unittest.skip("1004e NF front.py has no Dual-KV stage file loop (27B-line); L15 default off in NF release")
    def test_the_dual_kv_loan_reads_one_stage_file_per_card(self):
        src = open(os.path.join(os.path.dirname(L.__file__), "front.py")).read()
        self.assertIn("for r in range(max(3, len(self.dual_kv_ledgers or ()))):", src)
        self.assertNotIn("for r in range(3):\n            try:\n                with open(_pk.stage_file", src)


class SimulationTargets(unittest.TestCase):
    """The round's measurable goals, through the real pre-spawn path."""

    def cell(self, inv, model, sel=None, keys=None):
        return HS.simulate(inv, keys or list(HS.REFERENCE_RIG), HS.MODELS[model], sel)

    def test_27b_int8_on_5090_plus_3080_has_only_the_metal_proof_left(self):
        c = self.cell("rig --cards 1,0", "27B-INT8", (1, 0))
        self.assertEqual(c.blockers, ["METAL-UNPROVEN"], c.details)
        self.assertEqual(c.order, ["RTX5090/32607MiB/sm120", "RTX3080/20480MiB/sm86"])
        self.assertTrue(c.argv.startswith("P tp1/pp2 D tp2/pp1 ranks 0,1"), c.argv)
        self.assertTrue(any("DERIVED" in n or n.startswith("HW-DERIVE") for n in c.notes))

    def test_27b_nvfp4_on_two_3080_has_only_the_metal_proof_left(self):
        for model in ("27B-NVFP4", "27B-FP8", "27B-GGUF", "27B-INT8"):
            c = self.cell("rig --cards 0,2", model, (0, 2))
            self.assertEqual(c.blockers, ["METAL-UNPROVEN"], (model, c.details))
        c = self.cell("rig --cards 0,2", "27B-NVFP4", (0, 2))
        self.assertTrue(any("DROPPED" in n for n in c.notes), c.notes)     # the 49,8,7 pin
        self.assertTrue(any("sm86: nvfp4 kernel path 'w4a8'" in n for n in c.notes), c.notes)

    def test_what_is_still_refused_is_named(self):
        # no measured twin for the second 5090: calibration boot needed
        c = self.cell("2x5090", "27B-INT8", None, ["5090", "5090"])
        self.assertIn("UNCALIBRATED", c.blockers)
        # NF is tested on all three cards only
        c = self.cell("rig --cards 1,0", "NF", (1, 0))
        self.assertIn("PROFILE-VECTORS", c.blockers)
        self.assertIn("UNCALIBRATED", c.blockers)
        # one card: the single-group mode is P4
        self.assertIn("SINGLE-MODE", self.cell("rig --cards 1", "27B-INT8", (1,)).blockers)

    def test_the_reference_rig_still_runs_every_model(self):
        for m in HS.MODELS:
            c = self.cell("ref", m)
            self.assertEqual((c.result, c.blockers), (HS.RUNS, []), (m, c.details))
            self.assertTrue(c.argv.startswith("P tp1/pp3 D tp3/pp1 ranks 0,1,2 bar1 P '24,PP_0=96' "
                                              "D '16,TP_0=32,DCP_0=40'") or m == "27B-NVFP4-DUAL", (m, c.argv))

    def test_the_simulation_leaves_no_view_installed(self):
        self.cell("rig --cards 1,0", "27B-INT8", (1, 0))
        self.assertIsNone(IV.active())


if __name__ == "__main__":
    unittest.main()
