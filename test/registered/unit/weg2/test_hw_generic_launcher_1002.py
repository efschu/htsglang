"""HW-GENERIC 1002 (user 02.10.: "die software soll fuer jede hardware laufen
... aktuell halt 'nur' sm86 und sm120"): the weg2 launcher on ANY sm_86 /
sm_120 card set.

Two halves:

1. THE REGRESSION GATE. On the reference rig (nvml0 RTX 3080 20480 MiB sm86,
   nvml1 RTX 5090 32607 MiB sm120, nvml2 RTX 3080 20480 MiB sm86) every
   hardware-dependent launcher decision is IDENTICAL to the base tree
   3fe878018d: the plan fingerprint
   (``hw_generic_rig_plan_fingerprint_1002.py``) written on the base tree is
   the golden file this tree must reproduce byte for byte -- card order, CVD,
   W19 residue (both weight sources), DC expectations, the P-cut attention
   anchor stage, the xchg census constants, the builtin chunk model's power
   classes, the D budgets and budget lines of BOTH profile rows (nextflash =
   nf-int4; qwen27b = 27b / 27b-row-authority-cut43 / 27b-nvfp4-dual1i), the
   L1.5 posts, the PP-COST stage-model choice and the planner presets.

2. SYNTHETIC INVENTORIES. 3x RTX 3090, 3x RTX 5090, 2x RTX 5090 + RTX 3080,
   RTX PRO 6000 Blackwell + 2x RTX A6000, a stock 10 GB RTX 3080, and a mix
   with an sm_89 card: each reaches a coherent order and then either the
   named HW-UNCALIBRATED path (positional measurements of the reference rig
   are never borrowed) or the named HW-ARCH / HW-COUNT refusal -- never a
   KeyError, never a silent 3080/5090 number.

GPU-free, NVML-free (replay seam / hand-built cards).
"""

import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import card_identity as CI
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import xchg_census as XC

HERE = os.path.dirname(os.path.abspath(__file__))
# NF y7 line (desk/nf-y7j-hwgen-1002): the golden is re-derived on THIS line's base
# e7a70285bd (y7h + H4 fix + big_cards fix), the tree without HW-GENERIC; the
# 3fe878018d golden stays beside it for the release-tree line.
GOLDEN = os.path.join(HERE, "fixtures", "hw_generic_1002", "rig_plan_fingerprint_base_e7a70285bd.json")
MIB = 1 << 20


def card(i, name, mib, cc, uuid=None, reserved=0):
    return L.Card(i, uuid or f"GPU-{i:04d}", name, mib, reserved_mib=reserved, cc=cc)


def rig():
    return [card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6), "GPU-5c648f96", 425),
            card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0), "GPU-31d7ef41", 518),
            card(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6), "GPU-62dbbae1", 425)]


SYNTHETIC = {
    "3x3090": [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(3)],
    "3x5090": [card(i, "NVIDIA GeForce RTX 5090", 32607, (12, 0)) for i in range(3)],
    "2x5090+3080": [card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)),
                    card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
                    card(2, "NVIDIA GeForce RTX 5090", 32607, (12, 0))],
    "pro6000+2xa6000": [card(0, "NVIDIA RTX A6000", 49140, (8, 6)),
                        card(1, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 97887, (12, 0)),
                        card(2, "NVIDIA RTX A6000", 49140, (8, 6))],
    "5090+2x3080-10GB": [card(0, "NVIDIA GeForce RTX 3080", 10240, (8, 6)),
                         card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
                         card(2, "NVIDIA GeForce RTX 3080", 10240, (8, 6))],
}


def ns_for(profile, *extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--profile", profile, *extra])


class RegressionGateReferenceRig(unittest.TestCase):
    """The reference rig's plan is the base tree's, byte for byte."""

    def test_plan_fingerprint_is_identical_to_the_base_tree(self):
        sys.path.insert(0, HERE)
        try:
            import hw_generic_rig_plan_fingerprint_1002 as FP
        finally:
            sys.path.remove(HERE)
        with open(GOLDEN) as fh:
            golden = json.load(fh)
        now = json.loads(json.dumps(FP.fingerprint(), sort_keys=True, default=str))
        self.assertEqual(sorted(now), sorted(golden))
        for key in sorted(golden):
            self.assertEqual(now[key], golden[key], f"plan fingerprint {key!r} differs from base e7a70285bd")

    def test_order_is_the_old_5090_first_then_3080s_by_nvml_index(self):
        o = L.order_cards(rig())
        self.assertEqual([c.nvml_index for c in o], [1, 0, 2])
        self.assertEqual(CI.inventory_signature(o), CI.REFERENCE_INVENTORY)

    def test_inventory_check_passes_on_every_release_profile_row(self):
        o = L.order_cards(rig())
        for prof, extra in (("nextflash", ["--d-foreign-context-mib", "1446,896,894"]),
                            ("qwen27b", ["--pp-stage-ratio", "43,11,10"])):
            line = L.inventory_check_line(ns_for(prof, *extra), o)
            self.assertIn("MATCH", line)
            self.assertIn("ordinal 0: nvml1", line)

    def test_carve_scope_on_the_reference_classes_is_unchanged(self):
        o = L.order_cards(rig())
        # qwen27b: >= 32000 MiB -> only the 5090; nextflash: every card
        self.assertEqual([L.driver_carve_charged(c, 32000) for c in o], [True, False, False])
        self.assertEqual([L.driver_carve_charged(c, 0) for c in o], [True, True, True])


class SyntheticInventories(unittest.TestCase):
    """Foreign sm_86/sm_120 sets: coherent order, then a NAMED path."""

    def test_orders_are_by_memory_then_bandwidth_then_nvml_index(self):
        want = {
            "3x3090": [0, 1, 2],
            "3x5090": [0, 1, 2],
            "2x5090+3080": [1, 2, 0],
            "pro6000+2xa6000": [1, 0, 2],
            "5090+2x3080-10GB": [1, 0, 2],
        }
        for name, cards in SYNTHETIC.items():
            self.assertEqual([c.nvml_index for c in L.order_cards(list(cards))], want[name], name)

    def test_bandwidth_breaks_a_memory_tie(self):
        a = L.Card(0, "GPU-a", "NVIDIA X", 24576, cc=(8, 6), mem_bus_width_bits=256, mem_clock_max_mhz=9501)
        b = L.Card(1, "GPU-b", "NVIDIA Y", 24576, cc=(8, 6), mem_bus_width_bits=384, mem_clock_max_mhz=9751)
        c = L.Card(2, "GPU-c", "NVIDIA Z", 24576, cc=(8, 6))
        self.assertEqual([x.nvml_index for x in L.order_cards([a, b, c])], [1, 0, 2])

    def test_every_foreign_inventory_is_refused_uncalibrated_by_name(self):
        for name, cards in SYNTHETIC.items():
            o = L.order_cards(list(cards))
            for prof in ("nextflash", "qwen27b"):
                with self.assertRaises(L.Weg2LaunchRefused, msg=name) as cm:
                    L.inventory_check_line(ns_for(prof, "--d-nontorch-mib", "1981,528,524"), o)
                msg = str(cm.exception)
                self.assertTrue(msg.startswith(CI.CODE_UNCALIBRATED), msg)
                self.assertIn("NOT borrowed", msg)
                self.assertIn("--d-nontorch-mib", msg)
                self.assertIn("card_rate_pass --run", msg)

    def test_a_profile_naming_a_foreign_inventory_does_not_unlock_reference_records(self):
        """A profile written for 3x3090 (``--profile-inventory``) passes the
        VECTOR check -- and is still refused, by name, because this tree's
        measured records were taken on the reference rig: naming an
        inventory never makes another inventory's measurements hold."""
        o = L.order_cards(list(SYNTHETIC["3x3090"]))
        sig = ",".join(CI.inventory_signature(o))
        self.assertEqual(sig, "RTX3090/24576MiB/sm86,RTX3090/24576MiB/sm86,RTX3090/24576MiB/sm86")
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.inventory_check_line(ns_for("qwen27b", "--profile-inventory", sig), o)
        msg = str(cm.exception)
        self.assertIn("measured records", msg)
        self.assertNotIn("vectors (--profile-inventory)", msg)  # the vector check passed
        self.assertIn("D_OVERSHOOT_MIB", msg)

    def test_reference_rig_with_explicit_profile_inventory_passes_both_checks(self):
        o = L.order_cards(rig())
        line = L.inventory_check_line(ns_for("nextflash", "--profile-inventory", "RTX5090,RTX3080,RTX3080"), o)
        self.assertIn("MATCH", line)
        self.assertIn("vectors: --profile-inventory", line)

    def test_w19_residue_is_never_borrowed(self):
        for name in ("3x3090", "3x5090", "pro6000+2xa6000", "5090+2x3080-10GB"):
            o = L.order_cards(list(SYNTHETIC[name]))
            for c in o:
                cls = CI.calibration_class(c)
                if cls is not None:
                    continue  # a real 5090 / 20 GB 3080 keeps its measured class
                for ws in ("exchange", "serving"):
                    with self.assertRaises(L.Weg2LaunchRefused, msg=name) as cm:
                        L.dc_measured_d_mib(c, ws)
                    self.assertIn("HW-UNCALIBRATED", str(cm.exception))
                    self.assertIn(CI.card_key(c), str(cm.exception))
                self.assertIsNone(L.dc_expect_mib(c))
                with self.assertRaises(XC.Weg2XchgResidencyUnarmable):
                    XC.dormant_for_card(c, {}, "stem")

    def test_a_10gb_3080_is_not_the_20gb_class(self):
        c = card(0, "NVIDIA GeForce RTX 3080", 10240, (8, 6))
        self.assertIsNone(CI.calibration_class(c))
        self.assertEqual(CI.card_key(c), "RTX3080/10240MiB/sm86")
        with self.assertRaises(L.Weg2LaunchRefused):
            L.dc_measured_d_mib(c, "serving")

    def test_attn_anchor_stage_needs_its_measured_class(self):
        self.assertEqual(L.attn_anchor_stage(L.order_cards(list(SYNTHETIC["2x5090+3080"]))), 2)
        for name in ("3x3090", "3x5090", "pro6000+2xa6000"):
            with self.assertRaises(L.Weg2LaunchRefused) as cm:
                L.attn_anchor_stage(L.order_cards(list(SYNTHETIC[name])))
            self.assertIn("HW-UNCALIBRATED", str(cm.exception))

    def test_uncalibrated_cards_book_their_nvml_carve(self):
        for c in SYNTHETIC["3x3090"]:
            self.assertTrue(L.driver_carve_charged(c, 32000))

    @unittest.skipUnless(hasattr(L, "resolve_pp_cut_stage_model"),
                         "PP-COST stage model (27B release 01.10.) is not on the NF y7 line")
    def test_stage_model_is_not_picked_for_foreign_cards(self):
        from sglang.srt.weg2 import form as F

        ckpt = F.profile_row("qwen27b").formats["int8"].checkpoint
        for name, cards in SYNTHETIC.items():
            inv = CI.inventory_signature(L.order_cards(list(cards)))
            m, why = L.resolve_pp_cut_stage_model("auto", "qwen27b", ckpt, inventory=inv)
            self.assertIsNone(m, name)
            self.assertIn("previous pricing", why)

    def test_p_stage_class_mismatch_is_named(self):
        live = CI.inventory_signature(L.order_cards(list(SYNTHETIC["2x5090+3080"])))
        lines = L.p_stage_class_mismatch_lines(3, live)
        self.assertEqual(len(lines), 1)
        self.assertIn("PP1: live RTX5090 vs calibration RTX3080", lines[0])
        self.assertEqual(L.p_stage_class_mismatch_lines(3, CI.REFERENCE_INVENTORY), [])


class ArchAndCountGates(unittest.TestCase):
    """resolve_cards (the one card-list producer) gates the arch; order_cards
    the count -- both by name, through the NVML replay seam."""

    def _replay(self, rows):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(rows, fh)
        self.addCleanup(os.unlink, path)
        old = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
        os.environ[nvml_registry.ENV_NVML_REPLAY] = path

        def _restore():
            if old is None:
                os.environ.pop(nvml_registry.ENV_NVML_REPLAY, None)
            else:
                os.environ[nvml_registry.ENV_NVML_REPLAY] = old
        self.addCleanup(_restore)

    @staticmethod
    def _row(i, name, mib, cc):
        r = {"index": i, "uuid": f"GPU-{i:04d}", "name": name, "total_bytes": mib * MIB,
             "reserved_bytes": 0, "pci_bus_id": f"0000:0{i + 1}:00.0"}
        if cc is not None:
            r["cc_major"], r["cc_minor"] = cc
        return r

    def test_reference_rig_resolves_with_its_properties(self):
        self._replay([self._row(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)),
                      self._row(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
                      self._row(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6))])
        o = L.order_cards(L.resolve_cards())
        self.assertEqual([c.nvml_index for c in o], [1, 0, 2])
        self.assertEqual([c.cc for c in o], [(12, 0), (8, 6), (8, 6)])

    def test_an_sm89_card_in_the_mix_is_refused_by_name(self):
        self._replay([self._row(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)),
                      self._row(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
                      self._row(2, "NVIDIA GeForce RTX 4090", 24564, (8, 9))])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.resolve_cards()
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("HW-ARCH"), msg)
        self.assertIn("nvml2 'NVIDIA GeForce RTX 4090': compute capability 8.9 (sm89)", msg)
        self.assertNotIn("nvml0", msg)

    def test_an_unreported_cc_is_refused_not_guessed(self):
        self._replay([self._row(0, "NVIDIA GeForce RTX 3080", 20480, None),
                      self._row(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
                      self._row(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6))])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.resolve_cards()
        self.assertIn("compute capability not reported", str(cm.exception))

    def test_sm80_and_sm90_are_refused(self):
        for cc, nm in (((8, 0), "NVIDIA A100-PCIE-40GB"), ((9, 0), "NVIDIA H100 PCIe")):
            self._replay([self._row(i, nm, 40960, cc) for i in range(3)])
            with self.assertRaises(L.Weg2LaunchRefused):
                L.resolve_cards()

    def test_card_count_other_than_three_is_refused_by_name(self):
        for n in (2, 4):
            cards = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(n)]
            with self.assertRaises(L.Weg2LaunchRefused) as cm:
                L.order_cards(cards)
            self.assertTrue(str(cm.exception).startswith("HW-COUNT"), str(cm.exception))


class Stage2TopologyEnabling(unittest.TestCase):
    """S2 (enabling refactor): the argv sizes and the rank map come from the
    topology of the card count; N = 3 is byte-identical, N != 3 is refused
    BY NAME with what still assumes three ranks."""

    def test_n3_is_the_release_argv(self):
        from sglang.srt.weg2 import topology as T

        t = T.plan_topology(3)
        self.assertEqual((t.p_tp, t.p_pp, t.d_tp, t.d_pp, t.host_ordinal), (1, 3, 3, 1, 0))
        self.assertEqual(T.rank_gpu_id_csv(L.WEG2_CARD_COUNT), "0,1,2")
        src = open(L.__file__).read()
        self.assertNotIn('"--pp-size", "3"', src)
        self.assertNotIn('"--tp-size", "3"', src)
        self.assertNotIn('"--rank-gpu-id", "0,1,2"', src)

    def test_unproven_counts_are_refused_with_the_blockers(self):
        from sglang.srt.weg2 import topology as T

        for n in (2, 4, 8):
            t = T.release_topology(n)
            self.assertEqual((t.p_pp, t.d_tp), (n, n))
            self.assertFalse(t.proven)
            with self.assertRaises(T.TopologyRefused) as cm:
                T.plan_topology(n)
            self.assertIn("BAR1 group windows", str(cm.exception))
        for n in (1, 9):
            with self.assertRaises(T.TopologyRefused):
                T.release_topology(n)


if __name__ == "__main__":
    unittest.main()
