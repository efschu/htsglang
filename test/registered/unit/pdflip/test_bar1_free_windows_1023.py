# SPDX-License-Identifier: Apache-2.0
"""B6 (23.10.): the BAR1 windows of an N-card pdflip launch are derived from the
BAR1 that is FREE before the launch, not from the BAR1 total.

Evidence: boot dkr27bnvfp4duobar1duo10230534 -- 5090 with a 256 MiB BAR1 and
15 MiB baseline use; D TP0 died "group 'dcp:0' was explicitly given 40 MiB ...
but only 27 MiB of BAR1 is available" (Bar1WindowRefused) instead of the launch
being scaled/refused before the start.  The constraint a card has to meet:

    sum(windows) + OVERHEAD_MIB * (k - 1) <= free_before_launch - RESERVE_MIB
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import bar1_windows as BW
from flliper.srt.pdflip import launcher as L
from flliper.srt.pdflip import propose_oracle as O
from flliper.srt.registry import nvml

MIB = 1 << 20


def _sum(p, d):
    return sum(BW.parse_windows(p)[1].values()) + BW.parse_windows(p)[0] \
        + sum(BW.parse_windows(d)[1].values()) + BW.parse_windows(d)[0]


def _count(p, d):
    return sum(len(BW.parse_windows(x)[1]) + (1 if BW.parse_windows(x)[0] else 0) for x in (p, d))


def _constraint_holds(pl, free):
    k = _count(pl.p, pl.d)
    return _sum(pl.p, pl.d) + BW.OVERHEAD_MIB * (k - 1) <= free - BW.RESERVE_MIB


def card(i, name, mib, cc, bar1=256, free=None):
    return L.Card(i, f"GPU-{i:04d}", name, mib, cc=cc, bar1_total_mib=bar1, bar1_free_mib=free)


def ns_for(*extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])


def rig3(free_a=None, free_b=None, free_5090=None):
    return L.order_cards([
        card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6), free=free_a),
        card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0), 32768, free=free_5090),
        card(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6), free=free_b)])


class PlanFreeBar1(unittest.TestCase):

    def test_constant_is_the_measured_one(self):
        self.assertEqual(BW.OVERHEAD_MIB, 3.5)
        self.assertEqual(BW.RESERVE_MIB, 32)

    def test_reference_rig_unchanged_free_unknown_and_255(self):
        for free in (None, [None] * 3, [32768, 255, 255], [None, 255, 255], [255, 255, 255]):
            p = BW.plan(3, [32768, 256, 256], bar1_free_mib=free)
            self.assertTrue(p.ok)
            self.assertEqual((p.p, p.d), (BW.P_REFERENCE, BW.D_REFERENCE), free)
            self.assertEqual(p.scale, 1.0)
            self.assertEqual(p.why, "")
        # 2 cards as shipped
        p = BW.plan(2, [32768, 256], bar1_free_mib=[32768, 255])
        self.assertEqual((p.p, p.d), (BW.P_REFERENCE, BW.D_REFERENCE))

    def test_5090_256_with_15_used_is_scaled_before_the_start(self):
        # 5090 with a 256 MiB BAR1 and 15 MiB baseline -> free 241; the 3080 is clean (255)
        p = BW.plan(2, [256, 256], bar1_free_mib=[241, 255])
        self.assertTrue(p.ok)
        # usable 241 - 32 = 209, overhead 3.5 * 4 = 14 -> scale (209 - 14) / 208
        self.assertAlmostEqual(p.scale, 195 / 208.0)
        self.assertEqual((p.p, p.d), ("22,PP_0=90", "15,TP_0=30,DCP_0=37"))
        self.assertTrue(_constraint_holds(p, 241))
        self.assertEqual(p.free_min_mib, 241)
        self.assertEqual(p.overhead_mib, 14.0)
        self.assertIn("free BAR1 241", p.line())
        self.assertIn("baseline use 15 of BAR1 256", p.line())
        # the total alone would have passed it -- that was the defect
        q = BW.plan(2, [256, 256])
        self.assertEqual((q.p, q.d), (BW.P_REFERENCE, BW.D_REFERENCE))

    def test_todays_three_card_rig_scales_to_the_display_seat(self):
        # 5090 (BAR1 32768) + 3080 free 255 + 3080 (display seat, Used 67) free 189
        p = BW.plan(3, [32768, 256, 256], bar1_free_mib=[32768, 255, 189])
        self.assertTrue(p.ok)
        self.assertEqual(p.usable_mib, 189 - 32)
        self.assertAlmostEqual(p.scale, (157 - 14) / 208.0)
        self.assertEqual((p.p, p.d), ("16,PP_0=66", "11,TP_0=22,DCP_0=27"))
        for spec in (p.p, p.d):
            dflt, named = BW.parse_windows(spec)
            for w in ([dflt] if dflt else []) + list(named.values()):
                self.assertGreaterEqual(w, BW.MIN_WINDOW_MIB)
        self.assertTrue(_constraint_holds(p, 189))
        self.assertEqual(p.free_min_mib, 189)

    def test_two_card_5090_plus_clean_3080_keeps_the_shipped_windows(self):
        p = BW.plan(2, [32768, 256], bar1_free_mib=[32768, 255])
        self.assertTrue(p.ok)
        self.assertEqual((p.p, p.d), (BW.P_REFERENCE, BW.D_REFERENCE))
        self.assertEqual(p.scale, 1.0)

    def test_free_that_leaves_no_room_is_refused_by_name(self):
        p = BW.plan(2, [256, 256], bar1_free_mib=[80, 255])
        self.assertFalse(p.ok)
        self.assertIn("free BAR1 80", p.why)
        self.assertIn("baseline use 176", p.why)
        self.assertIn("floor", p.why)

    def test_total_stands_in_for_an_unreported_free(self):
        p = BW.plan(2, [256, 64], bar1_free_mib=[255, None])
        self.assertFalse(p.ok)       # 64 total -> 32 usable, as before
        self.assertIn("BAR1 64", p.why)


class ApplyBar1WindowsFree(unittest.TestCase):

    def test_proven_n3_with_a_card_holding_bar1_is_planned(self):
        ns = ns_for()
        lines = []
        out = L.apply_bar1_windows(ns, rig3(free_a=255, free_b=189, free_5090=32768), lines.append)
        self.assertIsNotNone(out)
        self.assertEqual(lines, [out])
        self.assertTrue(out.startswith("BAR1-WINDOW "))
        self.assertEqual(ns.p_barlink_bar1_window_mib, "16,PP_0=66")
        self.assertEqual(ns.d_barlink_bar1_window_mib, "11,TP_0=22,DCP_0=27")

    def test_proven_n3_free_unknown_is_byte_identical(self):
        ns = ns_for()
        lines = []
        self.assertIsNone(L.apply_bar1_windows(ns, rig3(), lines.append))
        self.assertEqual(lines, [])
        self.assertEqual((ns.p_barlink_bar1_window_mib, ns.d_barlink_bar1_window_mib),
                         (L.P_BARLINK_BAR1_WINDOW_MIB, L.D_BARLINK_BAR1_WINDOW_MIB))

    def test_proven_n3_free_255_everywhere_is_byte_identical(self):
        ns = ns_for()
        lines = []
        self.assertIsNone(L.apply_bar1_windows(ns, rig3(255, 255, 32768), lines.append))
        self.assertEqual(lines, [])
        self.assertEqual((ns.p_barlink_bar1_window_mib, ns.d_barlink_bar1_window_mib),
                         (L.P_BARLINK_BAR1_WINDOW_MIB, L.D_BARLINK_BAR1_WINDOW_MIB))

    def test_a_hopeless_free_bar1_is_refused_before_the_start(self):
        ns = ns_for()
        with self.assertRaises(L.PdFlipLaunchRefused) as cm:
            L.apply_bar1_windows(ns, rig3(free_a=255, free_b=70, free_5090=32768))
        self.assertIn("BAR1-WINDOW", str(cm.exception))
        self.assertIn("free BAR1 70", str(cm.exception))

    def test_non_proven_n2_5090_with_baseline_is_scaled(self):
        two = L.order_cards([card(0, "NVIDIA GeForce RTX 5090", 32607, (12, 0), 256, free=241),
                             card(1, "NVIDIA GeForce RTX 3080", 20480, (8, 6), free=255)])
        ns = ns_for()
        lines = []
        L.apply_bar1_windows(ns, two, lines.append)
        self.assertEqual((ns.p_barlink_bar1_window_mib, ns.d_barlink_bar1_window_mib),
                         ("22,PP_0=90", "15,TP_0=30,DCP_0=37"))
        self.assertEqual(len(lines), 1)

    def test_card_dataclass_default_is_unknown(self):
        self.assertIsNone(L.Card(0, "u", "n", 1).bar1_free_mib)


class ReplayRoundTrip(unittest.TestCase):

    def test_bar1_free_survives_replay_row_and_the_registry_reader(self):
        rows = [O.replay_row(0, uuid="GPU-a", name="NVIDIA GeForce RTX 5090", total_mib=32607, cc=(12, 0),
                             bar1_total_mib=256, bar1_free_mib=241),
                O.replay_row(1, uuid="GPU-b", name="NVIDIA GeForce RTX 3080", total_mib=20480, cc=(8, 6),
                             bar1_total_mib=256)]
        self.assertEqual(rows[0]["bar1_free_bytes"], 241 * MIB)
        self.assertNotIn("bar1_free_bytes", rows[1])
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(rows, fh)
        saved = os.environ.get(nvml.ENV_NVML_REPLAY)
        os.environ[nvml.ENV_NVML_REPLAY] = path
        try:
            devs = nvml.list_devices()
            self.assertEqual(devs[0].bar1_free_bytes, 241 * MIB)
            self.assertEqual(devs[0].bar1_total_bytes, 256 * MIB)
            self.assertIsNone(devs[1].bar1_free_bytes)
            cards = L.selected_nvml_cards()
            self.assertEqual([c.bar1_free_mib for c in cards], [241, None])
        finally:
            if saved is None:
                os.environ.pop(nvml.ENV_NVML_REPLAY, None)
            else:
                os.environ[nvml.ENV_NVML_REPLAY] = saved
            os.unlink(path)

    def test_free_is_not_an_identity_field(self):
        self.assertNotIn("bar1_free_bytes", nvml.IDENTITY_FIELDS)

    def test_replay_from_hardware_profile_passes_the_free_node(self):
        prof = {"schema": "flliper.hardware/1", "cards": [{
            "nvml_index": 0, "uuid": "GPU-a", "name": "NVIDIA GeForce RTX 5090", "cc": [12, 0],
            "vram_total_mib": {"v": 32607}, "bar1_total_mib": {"v": 256}, "bar1_free_mib": {"v": 241},
            "pcie": {}}, {
            "nvml_index": 1, "uuid": "GPU-b", "name": "NVIDIA GeForce RTX 3080", "cc": [8, 6],
            "vram_total_mib": {"v": 20480}, "bar1_total_mib": {"v": 256},
            "bar1_free_mib": {"v": None}, "pcie": {}}]}
        rows = O.replay_from_hardware_profile(prof)
        self.assertEqual(rows[0]["bar1_free_bytes"], 241 * MIB)
        self.assertNotIn("bar1_free_bytes", rows[1])


if __name__ == "__main__":
    unittest.main()
