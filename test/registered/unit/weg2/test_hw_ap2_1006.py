"""AP2 1006 (nf-hwgen-ap2): the reference-bound records that stopped a --force dry run of a foreign card count.

Planner finding (fable-planner, AP-C review, ZUSTAENDIG 16:51Z): a dry run with complete N-vectors did NOT run, not
even with --force:

  (a) NF N = 2 (5090 + 3080)       W167 -- the built-in P-card reference P_CARD_REFERENCE_FNFL2 is a 3-stage
                                          measurement; ``recut_check`` refused "Stufenzahl 2 gegen 3"; behind it the
                                          3-vector transient support (activation_reserve_mib_by_stage) and the
                                          3-vector records the forced boot kept (W40 "--pp-cut-stage-fixed-mib has 3
                                          entries", then "not one cut ... priceable");
  (b) N = 4 (2 x 5090 + 2 x 3080)  W71 UUID census (AP4, already forced past on this line), then AP1 (c)'s named
                                          refusal "awake-overshoot vector ... 3 entries for 4 cards", then every cut
                                          UNPRICED for a PCIe width outside the measured table;
  (c) 27B N = 2 at KV 262144       W40 "no SERVABLE cut" -- the profile's rig-bound --user-reserve-mib
                                          1800,1400,1400 carried to the foreign inventory (a FLAG: with 0,0 the dry
                                          run runs; nothing to change in the launcher, see the report).

NOW, with --force (Q-710: value refusals HW-UNCALIBRATED, named UNMEASURED, never silent):
  * ``inventory_view.borrow_vector`` + ``apply_inventory_derivation``: every positional record is BORROWED for the
    live cards (class-max by class or AP1 arch twin, else stage/rank ROLE);
  * ``p_card_chunk.restage_reference`` / ``launcher._restage_p_card_reference``: the P-card reference is re-staged by
    ROLE onto the live stage count (K0 shifted by layers as in a recut) -- DERIVED when every stage keeps its class,
    a forced borrow otherwise;
  * ``launcher._live_stage_support``: the builtin transient support follows the live stage count;
  * ``launcher.forced_lane_borrow``: a PCIe width without a measured crossing price is priced at the nearest measured
    width not faster than its own;
  * ``launcher.derived_family_cost``: a stage class without a measured twin borrows its arch twin's cost (AP1 rule).
Without --force everything refuses as before; the reference rig (N = 3) is untouched.

Honesty: every borrowed figure is UNMEASURED on the card it is applied to; green here is the desk gate, not a boot.
GPU-free, NVML-free.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import p_card_chunk as PC
from sglang.srt.weg2 import card_identity as CI
from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import inventory_view as IV
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import refusals as R
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-weg2-unit")

REF = ("RTX5090", "RTX3080", "RTX3080")


def card(i, name, mib, cc):
    return L.Card(i, f"GPU-ap2-{i:02d}", name, mib, reserved_mib=0, cc=cc, bar1_total_mib=256)


C3080 = lambda i: card(i, "NVIDIA GeForce RTX 3080", 20480, (8, 6))        # noqa: E731
C3090 = lambda i: card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6))        # noqa: E731
C5090 = lambda i: card(i, "NVIDIA GeForce RTX 5090", 32607, (12, 0))       # noqa: E731
C4090 = lambda i: card(i, "NVIDIA GeForce RTX 4090", 24564, (8, 9))        # noqa: E731


def ns_for(*extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])


def rig():
    return L.order_cards([C3080(0), C5090(1), C3080(2)])


#: NF's layer kinds (48 layers, full attention every 4th: 12 attention layers -- 9,3 at the 38,10 cut)
NF_KINDS = tuple("full_attention" if i % 4 == 3 else "linear_attention" for i in range(48))


class _Armed(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        R.arm(False)
        os.environ.pop(R.ENV_FORCED_BOOT, None)
        self.addCleanup(R.arm, False)
        IV.clear_active()
        self.addCleanup(IV.clear_active)

    def force(self):
        R.arm(True)

    def forced(self, code="HW-UNCALIBRATED"):
        return [d["text"] for d in R.forced_list() if d["code"] == code]


class TestBorrowVector(unittest.TestCase):
    def test_a_subset_keeps_its_derivation(self):
        self.assertEqual(IV.borrow_vector(IV.CLASS_MAX, (452, 397, 0), REF, ("RTX5090", "RTX3080")),
                         IV.derive_vector(IV.CLASS_MAX, (452, 397, 0), REF, ("RTX5090", "RTX3080")))

    def test_class_max_with_a_repeated_class(self):
        live = ("RTX5090", "RTX5090", "RTX3080", "RTX3080")
        self.assertIsNone(IV.derive_vector(IV.CLASS_MAX, (452, 397, 0), REF, live))   # not a subset
        self.assertEqual(IV.borrow_vector(IV.CLASS_MAX, (452, 397, 0), REF, live), (452, 452, 397, 397))
        self.assertEqual(IV.borrow_vector(IV.CLASS_MAX, "8.10,35.16,33.59", REF, live), "8.10,8.10,35.16,35.16")

    def test_a_foreign_class_takes_its_twin_or_nothing(self):
        live = ("RTX3090/24576MiB/sm86",) * 2
        self.assertIsNone(IV.borrow_vector(IV.CLASS_MAX, (452, 397, 0), REF, live))
        self.assertEqual(IV.borrow_vector(IV.CLASS_MAX, (452, 397, 0), REF, live,
                                          {"RTX3090/24576MiB/sm86": "RTX3080"}), (397, 397))

    def test_role_for_records_without_a_policy(self):
        self.assertEqual(IV.borrow_vector(None, (7442, 1036, 914), REF, ("RTX5090", "RTX3080")), (7442, 914))
        self.assertEqual(IV.borrow_vector(None, (2672, None, None), REF, ("A", "B", "C", "D")),
                         (2672, None, None, None))
        self.assertEqual(IV.borrow_vector(IV.ROLE, "1584.0,527.1,1065.4", REF, ("A", "B", "C", "D")),
                         "1584.0,527.1,527.1,1065.4")
        self.assertIsNone(IV.borrow_vector(None, (1, 2, 3), REF, ("A",)))            # one card: no role to place
        self.assertIsNone(IV.borrow_vector(None, (1, 2), REF, ("A", "B")))            # not a positional 3-vector

    def test_exempt_policies_are_untouched(self):
        self.assertEqual(IV.borrow_vector(IV.CONSUMER, (2588, 3084, 2588), REF, ("A", "B")), (2588, 3084, 2588))

    def test_apply_active_reads_only_the_named_records_borrowed(self):
        IV.set_active("p", REF, ("RTX5090", "RTX3080"), borrow=["D_FIXED_MIB"], twins={})
        try:
            self.assertEqual(IV.apply_active("D_FIXED_MIB", "p", (7442, 1036, 914)), (7442, 914))
            self.assertEqual(IV.apply_active("OTHER", "p", (1, 2, 3)), (1, 2, 3))
            self.assertEqual(IV.apply_active("D_FIXED_MIB", "q", (7442, 1036, 914)), (7442, 1036, 914))
        finally:
            IV.clear_active()
        self.assertEqual(IV.apply_active("D_FIXED_MIB", "p", (7442, 1036, 914)), (7442, 1036, 914))


class TestForcedRecordBorrow(_Armed):
    def two(self):
        return L.order_cards([C5090(1), C3080(0)])

    def test_without_force_nothing_is_installed(self):
        ns = ns_for("--profile", "nextflash")
        self.assertEqual(L.apply_inventory_derivation(ns, self.two(), None, {}), [])
        self.assertIsNone(IV.active())
        self.assertEqual(len(F.profile_constant("D_FIXED_MIB", "nextflash")), 3)

    def test_force_borrows_every_positional_record_and_names_it(self):
        self.force()
        ns = ns_for("--profile", "nextflash")
        notes = L.apply_inventory_derivation(ns, self.two(), None, {})
        self.assertTrue(any("record D_FIXED_MIB" in n and "FORCED BORROW" in n for n in notes), notes)
        self.assertEqual(list(F.profile_constant("D_FIXED_MIB", "nextflash")), [7442, 914])
        self.assertEqual(list(F.profile_constant("P_OVERSHOOT_MIB", "nextflash")), [452, 397])
        self.assertEqual(len(F.profile_constant("P_PP_STAGE_FIXED_MIB", "nextflash").split(",")), 2)
        # the consumer-indexed triple is not touched
        self.assertEqual(len(F.profile_constant("DC_MEASURED_D_XCHG_MIB", "nextflash")), 3)
        txt = self.forced()
        self.assertTrue(any("BORROWED for [RTX5090, RTX3080]" in t and "D_FIXED_MIB" in t for t in txt), txt)

    def test_force_at_four_cards_gives_four_entries(self):
        self.force()
        four = L.order_cards([C5090(0), C5090(1), C3080(2), C3080(3)])
        L.apply_inventory_derivation(ns_for("--profile", "nextflash"), four, None, {})
        self.assertEqual(list(F.profile_constant("P_OVERSHOOT_MIB", "nextflash")), [452, 452, 397, 397])
        self.assertEqual(list(F.profile_constant("D_EXTEND_GROWTH_MIB", "nextflash")), [2434, 2176, 2176, 1728])

    def test_the_reference_rig_is_untouched_under_force(self):
        self.force()
        ns = ns_for("--profile", "nextflash")
        before = vars(ns).copy()
        self.assertEqual(L.apply_inventory_derivation(ns, rig(), None, {}), [])
        self.assertEqual(vars(ns), before)
        self.assertIsNone(IV.active())
        self.assertEqual(R.forced_list(), [])


class TestRestage(unittest.TestCase):
    REF_P = PC.P_CARD_REFERENCE_FNFL2
    CO = PC.P_CARD_CO_TENANT_FNFL2

    def sup(self):
        return L.P_PREFILL_TRANSIENT_SUPPORT

    def test_role_stage_map(self):
        self.assertEqual(PC.role_stage_map(2, 3), (0, 2))
        self.assertEqual(PC.role_stage_map(3, 3), (0, 1, 2))
        self.assertEqual(PC.role_stage_map(4, 3), (0, 1, 1, 2))
        self.assertIsNone(PC.role_stage_map(1, 3))

    def test_identity_map_equals_the_recut(self):
        split = (27, 12, 9)
        kw = dict(layer_kinds=NF_KINDS, dense_mib_per_layer=62.0, mamba_mib_per_linear_layer_per_slot=1.5588)
        a = PC.recut_reference(self.REF_P, split, **kw)
        b, sup, co = PC.restage_reference(self.REF_P, self.sup(), self.CO, split, (0, 1, 2), **kw)
        self.assertEqual(a.headroom0_mib, b.headroom0_mib)
        self.assertEqual(a.row_card_mib, b.row_card_mib)
        self.assertEqual(sup.points, self.sup().points)
        self.assertEqual(co.max_mib, self.CO.max_mib)

    def test_two_stages_take_first_and_last_and_shift_k0_by_layers(self):
        kw = dict(layer_kinds=NF_KINDS, dense_mib_per_layer=62.0, mamba_mib_per_linear_layer_per_slot=1.5588)
        r, sup, co = PC.restage_reference(self.REF_P, self.sup(), self.CO, (38, 10), (0, 2), **kw)
        _a0, lin0 = PC.stage_kind_counts(NF_KINDS, self.REF_P.stage_layers)
        _a1, lin1 = PC.stage_kind_counts(NF_KINDS, (38, 10))
        mam = self.REF_P.mamba_slots * 1.5588
        want = (round(self.REF_P.headroom0_mib[0] - 9 * 62.0 - (lin1[0] - lin0[0]) * mam, 1),
                round(self.REF_P.headroom0_mib[2] - 2 * 62.0 - (lin1[1] - lin0[2]) * mam, 1))
        self.assertEqual(r.headroom0_mib, want)
        self.assertEqual(r.stage_layers, (38, 10))
        self.assertEqual(r.cap_mib, (self.REF_P.cap_mib[0], self.REF_P.cap_mib[2]))
        self.assertEqual(co.max_mib, (self.CO.max_mib[0], self.CO.max_mib[2]))
        self.assertEqual(sup.n_stages, 2)
        self.assertEqual(sup.points[0].mib, (self.sup().points[0].mib[0], self.sup().points[0].mib[2]))
        self.assertIn("RESTAGE", r.source)

    def test_a_support_already_staged_for_the_live_count_is_kept(self):
        kw = dict(layer_kinds=NF_KINDS, dense_mib_per_layer=62.0, mamba_mib_per_linear_layer_per_slot=1.5588)
        two = PC.TransientSupport(model="m", points=tuple(
            PC.TransientPoint(chunk=p.chunk, mib=(p.mib[0], p.mib[2]), source=p.source) for p in self.sup().points))
        _r, sup, _co = PC.restage_reference(self.REF_P, two, self.CO, (38, 10), (0, 2), **kw)
        self.assertIs(sup, two)

    def test_what_cannot_be_restaged_stays_w167(self):
        kw = dict(layer_kinds=NF_KINDS, dense_mib_per_layer=62.0, mamba_mib_per_linear_layer_per_slot=1.5588)
        for split, smap, kinds in (((38, 9), (0, 2), NF_KINDS), ((38, 10), (0, 2), None),
                                   ((38, 10), (0, 5), NF_KINDS)):
            with self.assertRaises(PC.PCutRecutRefused) as cm:
                PC.restage_reference(self.REF_P, self.sup(), self.CO, split, smap,
                                     **dict(kw, layer_kinds=kinds))
            self.assertIn("W167", str(cm.exception))


class TestRestageInTheLauncher(_Armed):
    KW = dict(layer_kinds=NF_KINDS, dense_mib_per_layer=62.0, mamba_mib_per_linear_layer_per_slot=1.5588)

    def run_restage(self, cards, split):
        lines = []
        ns = ns_for("--profile", "nextflash")
        out = L._restage_p_card_reference(ns, cards, PC.P_CARD_REFERENCE_FNFL2, L.P_PREFILL_TRANSIENT_SUPPORT,
                                          PC.P_CARD_CO_TENANT_FNFL2, split, lines.append, **self.KW)
        return out, lines

    def test_same_classes_are_derived_without_a_refusal(self):
        (r, _s, _c), lines = self.run_restage(L.order_cards([C5090(1), C3080(0)]), (38, 10))
        self.assertEqual(r.stage_layers, (38, 10))
        self.assertTrue(any("RESTAGE" in ln and "DERIVED" in ln for ln in lines), lines)
        self.assertEqual(R.forced_list(), [])

    def test_a_stage_on_another_class_refuses_without_force(self):
        four = L.order_cards([C5090(0), C5090(1), C3080(2), C3080(3)])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            self.run_restage(four, (19, 19, 5, 5))
        self.assertIn("W167", str(cm.exception))
        self.assertIn("HW-UNCALIBRATED", str(cm.exception))
        self.assertIn("stage(s) 1", str(cm.exception))

    def test_force_borrows_the_row_and_names_it(self):
        self.force()
        four = L.order_cards([C5090(0), C5090(1), C3080(2), C3080(3)])
        (r, sup, co), _ = self.run_restage(four, (19, 19, 5, 5))
        self.assertEqual(len(r.headroom0_mib), 4)
        self.assertEqual(sup.n_stages, 4)
        self.assertEqual(len(co.max_mib), 4)
        self.assertTrue(any("RESTAGE" in t and "BORROWS" in t for t in self.forced()), self.forced())


class TestLiveStageSupport(_Armed):
    def test_no_view_no_change(self):
        self.assertIs(L._live_stage_support(L.P_PREFILL_TRANSIENT_SUPPORT), L.P_PREFILL_TRANSIENT_SUPPORT)

    def test_a_view_of_two_cards_gives_two_stages(self):
        IV.set_active("nextflash", REF, ("RTX5090", "RTX3080"))
        sup = L._live_stage_support(L.P_PREFILL_TRANSIENT_SUPPORT)
        self.assertEqual(sup.n_stages, 2)
        base = L.P_PREFILL_TRANSIENT_SUPPORT.points[0].mib
        self.assertEqual(sup.points[0].mib, (base[0], base[2]))
        self.assertEqual(len(L.p_prefill_transient_vector_mib(16384, "nextflash")), 2)

    def test_a_view_of_three_cards_keeps_the_support(self):
        IV.set_active("nextflash", REF, ("RTX5090", "RTX3080", "RTX3080/x"))
        self.assertIs(L._live_stage_support(L.P_PREFILL_TRANSIENT_SUPPORT), L.P_PREFILL_TRANSIENT_SUPPORT)


class TestLaneBorrow(_Armed):
    def cards(self, n):
        return L.order_cards([C5090(i) for i in range(n)])

    def test_measured_widths_are_returned_as_given(self):
        lanes = [4, 8]
        self.assertEqual(L.forced_lane_borrow(self.cards(2), lanes), lanes)
        self.assertEqual(R.forced_list(), [])

    def test_an_unmeasured_width_refuses_without_force(self):
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.forced_lane_borrow(self.cards(2), [16, 8])
        self.assertIn("x16 -> x8", str(cm.exception))

    def test_force_prices_at_the_nearest_measured_width_not_faster(self):
        self.force()
        self.assertEqual(L.forced_lane_borrow(self.cards(4), [16, None, 8, 2]), [8, 4, 8, 2])
        txt = self.forced()
        self.assertTrue(any("x16 -> x8" in t and "x? -> x4" in t for t in txt), txt)
        self.assertFalse(any("x2 ->" in t for t in txt), txt)       # below every measured width: no borrow


class TestFamilyCostTwin(_Armed):
    def test_sm86_borrows_the_3080_cost_under_force_only(self):
        ns = ns_for("--profile", "nextflash")
        fams = list(NF_KINDS)          # pp_cut.LAYER_FAMILY_* are the checkpoint's layer_types strings
        four = L.order_cards([C3090(i) for i in range(4)])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.derived_family_cost(ns, four, fams, 4096)
        self.assertIn("no measured twin", str(cm.exception))
        self.force()
        cost, prov = L.derived_family_cost(ns, four, fams, 4096)
        ref, _ = L.derived_family_cost(ns, L.order_cards([C3080(0), C3080(1)]), fams, 4096)
        self.assertEqual(cost.linear_ms_per_layer, (ref.linear_ms_per_layer[0],) * 4)
        self.assertTrue(any("BORROWS the per-layer cost of class RTX3080" in t for t in self.forced()))

    def test_sm89_stays_refused_with_force(self):
        self.force()
        fams = list(NF_KINDS)
        with self.assertRaises(L.Weg2LaunchRefused):
            L.derived_family_cost(ns_for("--profile", "nextflash"), L.order_cards([C4090(0), C4090(1)]), fams, 4096)


class TestCensusCarriesItsClass(unittest.TestCase):
    """The census producer writes the class each row was measured on (AP4's optional field): a foreign inventory
    then borrows the row of its own class instead of the heaviest (the 5090 image on a 3080: 24330 MiB of 20480)."""

    def test_rows_name_their_class_and_a_foreign_3080_borrows_a_3080_row(self):
        import test_weg2_xchg_census_1273 as T
        from sglang.srt.weg2 import xchg_residency as XR

        build = T._build()
        self.assertEqual(build.blob["cards"][T.BIG]["card_class"], "RTX5090")
        self.assertEqual(build.blob["cards"][T.SM1]["card_class"], "RTX3080")
        census = XR.XchgCensus(
            cards={u: XR.CardCensus(uuid=u, tags=e["tags"], dormant_proc_used_mib=e["dormant_proc_used_mib"],
                                    card_class=e["card_class"]) for u, e in build.blob["cards"].items()},
            waves=(T.FAMILY,))
        _c, borrows, _n = XR.resolve_census(census, [C3080(7)])
        self.assertEqual([(b.kind, b.donor_class) for b in borrows], [("class", "RTX3080")])


if __name__ == "__main__":
    unittest.main()
