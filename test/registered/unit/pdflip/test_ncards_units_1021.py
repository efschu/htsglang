"""NCARDS 1021: fLLiper's planning core and launcher for N identical cards (2 x / 3 x / 4 x RTX 3090) -- the unit level.

The inventory of every N = 3 assumption is in ``/spinning/gpu-arb/deskq/done/planer-ncards-1021.md``; most of them were made N-capable
or DERIVED by HW-P1c 1003 (``test_hw_p1c_n_paths_1003``).  What this round closed, and what this file pins:

* ``launcher.p_group_stages``      answered 3 on a FOREIGN inventory (N x RTX 3090), so a forced launch priced an N-stage cut with
  three-entry tables (``prefill_graph_pool_mib has 3 entries for 4 stages`` -> W40 "not one cut ... is priceable at all"): it follows the
  live card count now;
* ``launcher._p_chunk_by_class``   mapped only a SUBSET of the calibrated cards; a foreign inventory takes the SLOWEST calibrated stage of
  the card's arch twin class (sm86 -> RTX3080), as the records' forced borrow does -- no twin (sm89) is still refused;
* ``card_identity.arch_twin_of_label``  the class-LABEL form of ``arch_twin_class``;
* ``launcher.dual_alloc_cache_book_env``  read a positional 3-vector straight against N cards.

The reference rig (RTX 5090 + 2 x RTX 3080, N = 3) is byte-identical in all of it: ``test_reference_rig_*``.  GPU-free, NVML-free.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_identity as CI
from flliper.srt.pdflip import hw_sim as HS
from flliper.srt.pdflip import launcher as L
from flliper.srt.pdflip import topology as T
from flliper.srt.pdflip import weight_exchange_region as XR


class ArchTwinOfLabel(unittest.TestCase):
    def test_sm86_is_the_3080_and_sm120_the_5090(self):
        self.assertEqual(CI.arch_twin_of_label("RTX3090/24576MiB/sm86"), "RTX3080")
        self.assertEqual(CI.arch_twin_of_label("RTX3080/20480MiB/sm86"), "RTX3080")
        self.assertEqual(CI.arch_twin_of_label("RTX5090/32607MiB/sm120"), "RTX5090")

    def test_no_calibrated_twin_is_none(self):
        self.assertIsNone(CI.arch_twin_of_label("RTX4090/24564MiB/sm89"))      # no calibrated sm89 class
        self.assertIsNone(CI.arch_twin_of_label("RTX3090"))
        self.assertIsNone(CI.arch_twin_of_label(""))
        self.assertIsNone(CI.arch_twin_of_label(None))

    def test_the_card_form_and_the_label_form_agree(self):
        for key in ("3090", "3080-20G", "5090", "4090"):
            row = HS.replay_rows([key])[0]
            card = L.Card(0, row["uuid"], row["name"], row["total_bytes"] // (1 << 20), cc=(row["cc_major"], row["cc_minor"]))
            label = "%s/%dMiB/sm%d%d" % (CI.class_label(card), card.total_mib, *card.cc)
            self.assertEqual(CI.arch_twin_of_label(label), CI.arch_twin_class(card), key)


class PGroupStagesFollowTheCards(unittest.TestCase):
    """The P-chunk stage table and the prefill-graph pool vector ask NVML for the stage count (main reads them before the cards)."""

    def test_foreign_n_identical_cards_give_n_stages(self):
        for n in (2, 3, 4):
            with HS.replayed(["3090"] * n, None):
                self.assertEqual(L.p_group_stages(), n)

    def test_reference_rig_stays_three_and_a_subset_two(self):
        with HS.replayed(HS.REFERENCE_RIG, None):
            self.assertEqual(L.p_group_stages(), 3)
        with HS.replayed(HS.REFERENCE_RIG, (1, 0)):
            self.assertEqual(L.p_group_stages(), 2)

    def test_no_nvml_is_three(self):
        from unittest import mock
        with mock.patch.object(L, "_live_inventory_or_none", return_value=None):
            self.assertEqual(L.p_group_stages(), L.P_PREFILL_GRAPH_STAGES)


class PChunkTablesOnForeignCards(unittest.TestCase):
    def _three(self):
        with HS.replayed(HS.REFERENCE_RIG, None):
            return L.p_chunk_stage_model("builtin-int8", "int8", 3, "qwen27b")[0]

    def test_every_3090_stage_takes_the_slowest_3080_stage(self):
        three = self._three()
        slow = max(three[1:], key=lambda m: m.points[-1][1])
        for n in (2, 3, 4):
            with HS.replayed(["3090"] * n, None):
                tab, src = L.p_chunk_stage_model("builtin-int8", "int8", n, "qwen27b")
            self.assertEqual(len(tab), n, n)
            for m in tab:
                self.assertEqual(m.points, slow.points)

    def test_no_arch_twin_is_still_refused(self):
        with HS.replayed(["4090", "4090"], None), self.assertRaises(SystemExit):
            L.p_chunk_stage_model("builtin-int8", "int8", 2, "qwen27b")

    def test_reference_rig_table_is_untouched(self):
        three = self._three()
        self.assertEqual(len(three), 3)
        with HS.replayed(HS.REFERENCE_RIG, None):
            self.assertIsNone(L._p_chunk_by_class(three, lambda m: m.points[-1][1], 3, "qwen27b"))

    def test_prefill_graph_pool_has_n_entries_on_n_cards(self):
        for n in (2, 4):
            from unittest import mock
            with HS.replayed(["3090"] * n, None), mock.patch.object(L, "p_prefill_graph_buckets", return_value=[512]):
                ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
                pool = L.p_prefill_graph_pool_mib(ns)
                self.assertEqual(len(pool), n, pool)
                one = L.p_prefill_graph_pool_mib(L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--p-prefill-graph-pool-mib", "160"]))
                self.assertEqual(one, (160.0,) * n)


class TopologyOfN(unittest.TestCase):
    """What blocks an N-card launch is named and data-shaped; N = 2 and 4 carry no CODE blocker any more (HW-P1c) -- only the measured
    records of another inventory (forceable) and the metal proof."""

    def test_n2_n4_blockers_are_records_and_metal_only(self):
        for n in (2, 4):
            for dual in (False, True):
                ctx = T.TopologyContext(profile="qwen27b", weight_format="nvfp4", dual=dual, weight_source="exchange", bar1_mib=(256,) * n)
                self.assertEqual({b.code for b in T.blockers(n, ctx)} - {"RECORDS-NVEC", "METAL-UNPROVEN"}, set(), (n, dual))
                self.assertIn("METAL-UNPROVEN", {b.code for b in T.blockers(n, ctx)})

    def test_release_topology_is_pp_n_over_tp_n(self):
        for n in (2, 3, 4):
            t = T.release_topology(n)
            self.assertEqual((t.p_pp, t.d_tp, t.n_cards), (n, n, n))
            self.assertEqual(t.proven, n in T.PROVEN_CARD_COUNTS)

    def test_reference_count_is_proven_and_blocker_free(self):
        self.assertEqual(T.PROVEN_CARD_COUNTS, (3,))
        self.assertEqual(T.blockers(3, T.TopologyContext(profile="qwen27b", weight_format="int8")), ())

    def test_xchg_region_geometry_follows_n(self):
        try:
            for n in (2, 4):
                XR.configure(n)
                self.assertEqual(XR.N_CARDS, n)
                self.assertEqual(XR.N_PAIRS, n * (n - 1))
        finally:
            XR.configure(3)
        self.assertEqual((XR.N_CARDS, XR.N_PAIRS), (3, 6))


if __name__ == "__main__":
    unittest.main()
