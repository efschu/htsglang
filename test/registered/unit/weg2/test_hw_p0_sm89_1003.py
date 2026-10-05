"""HW-P0 1003 (PLAN_HWGEN_N_KARTEN_1003 P0): sm_89 in the arch gate with a
NAMED calibration fallback, and a graph-mem anchor key that names the cards.

User order 03.10. ~19:30Z: the software must run on sm86, sm89 and sm120 with
any card count. P0 makes sm_89 a gate-passing arch on both lines:

* ``card_identity.SUPPORTED_ARCHS`` holds (8, 9); an sm_89 card is never
  refused HW-ARCH. It has no calibration class, so it takes the named
  fallback ``CALIBRATION_FALLBACK`` (record label = its card key) and every
  positional value stops at HW-UNCALIBRATED, which now also NAMES the arch.
* ``UNCALIBRATED_ARCHS`` is derived (supported archs without a calibrated
  class), never a hand list -- calibrating an sm_89 class removes it.
* ``graphmem.anchor_key`` (K8, v3) carries the sorted card-class labels; a
  calibrated label implies the arch (exact cc match), an uncalibrated one
  spells ``sm<cc>``. A rig with other cards -- other arch, other tier, other
  count -- never inherits the reference rig's anchor as "measured".

Reference rig (RTX5090 sm120 + 2x RTX3080 sm86) unchanged: no message, no
arch note, same labels; the golden plan fingerprint is pinned by
``test_hw_generic_launcher_1002``.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from sglang.srt.planner import graphmem
from sglang.srt.weg2 import card_identity as ci
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _c(i, name, mib, cc):
    return ci.CardProps(nvml_index=i, uuid=f"GPU-p0-{i}", name=name, total_mib=mib, cc=cc)


R5090 = lambda i: _c(i, "NVIDIA GeForce RTX 5090", 32607, (12, 0))  # noqa: E731
R3080 = lambda i: _c(i, "NVIDIA GeForce RTX 3080", 20480, (8, 6))  # noqa: E731
R4090 = lambda i: _c(i, "NVIDIA GeForce RTX 4090", 24564, (8, 9))  # noqa: E731
R3090 = lambda i: _c(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6))  # noqa: E731


class TestArchGateSm89(unittest.TestCase):
    def test_supported_archs_are_sm86_sm89_sm120(self):
        self.assertEqual(ci.SUPPORTED_ARCHS, ((8, 6), (8, 9), (12, 0)))

    def test_uncalibrated_archs_are_derived_not_listed(self):
        calibrated = {c.cc for c in ci.CALIBRATED_CLASSES}
        self.assertEqual(ci.UNCALIBRATED_ARCHS,
                         tuple(a for a in ci.SUPPORTED_ARCHS if a not in calibrated))
        self.assertEqual(ci.UNCALIBRATED_ARCHS, ((8, 9),))

    def test_sm89_passes_the_gate_sm80_sm90_do_not(self):
        ci.arch_gate([R4090(0), R4090(1)])  # no raise: never HW-ARCH
        for cc in ((7, 0), (7, 5), (8, 0), (9, 0), (10, 0), (12, 1), None):  # None = cc not reported
            with self.assertRaises(ci.CardInventoryRefused) as cm:
                ci.arch_gate([_c(0, "NVIDIA X", 24000, cc)])
            self.assertTrue(str(cm.exception).startswith(ci.CODE_ARCH))
            self.assertIn("sm_86, sm_89 and sm_120", str(cm.exception))

    def test_sm89_takes_the_named_fallback(self):
        c = R4090(0)
        self.assertIsNone(ci.calibration_class(c))
        self.assertTrue(ci.arch_uncalibrated(c))
        self.assertEqual(ci.class_label(c), "RTX4090/24564MiB/sm89")
        # the reference classes and an uncalibrated sm86 board are NOT arch-uncalibrated
        for c in (R5090(0), R3080(1), R3090(2)):
            self.assertFalse(ci.arch_uncalibrated(c))
        self.assertFalse(ci.arch_uncalibrated(_c(0, "NVIDIA GeForce RTX 3080", 20480, None)))

    def test_uncalibrated_message_names_the_arch_and_the_fallback(self):
        o = ci.order_cards([R3080(0), R5090(1), R4090(2)], expect_count=3)
        msg = ci.uncalibrated_message(o, ci.REFERENCE_INVENTORY, ("X",), "the profile")
        self.assertTrue(msg.startswith(ci.CODE_UNCALIBRATED), msg)
        self.assertIn("ordinal 1: live RTX4090/24564MiB/sm89 vs calibrated RTX3080", msg)
        self.assertIn("sm89 has no calibration class in this release", msg)
        self.assertIn(ci.CALIBRATION_FALLBACK, msg)
        self.assertIn("card_rate_pass --run", msg)

    def test_a_foreign_sm86_inventory_has_no_arch_note(self):
        o = ci.order_cards([R3090(0), R3090(1), R3090(2)], expect_count=3)
        msg = ci.uncalibrated_message(o, ci.REFERENCE_INVENTORY, ("X",), "the profile")
        self.assertTrue(msg.startswith(ci.CODE_UNCALIBRATED), msg)
        self.assertNotIn("no calibration class in this release", msg)
        # without an uncalibrated arch the message text is the pre-P0 text
        self.assertIn("). Positional measurements of the other inventory are NOT", msg)

    def test_reference_rig_is_calibrated(self):
        o = ci.order_cards([R3080(0), R5090(1), R3080(2)], expect_count=3)
        self.assertEqual(ci.inventory_signature(o), ci.REFERENCE_INVENTORY)
        self.assertIsNone(ci.uncalibrated_message(o, ci.REFERENCE_INVENTORY, ("X",), "p"))


def _meta(cards):
    return {
        "model_path": "/models/Qwen3.6-27B-INT8",
        "tp_size": 3,
        "kv_cache_dtype": "auto",
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
        "speculative_adaptive": True,
        "attention_backend": "flashinfer",
        "page_size": 1,
        "decode_bs": [1, 2],
        "card_classes": cards,
    }


_SUMMARY = {"per_rank_mib": {"0": 300.0, "1": 300.0, "2": 300.0},
            "items": [{"label": "target decode", "total_mib": 900.0}]}


class TestAnchorKeyNamesTheInventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = graphmem.AnchorStore(os.path.join(self.tmp.name, "a.json"))
        self.rig = tuple(ci.class_label(c) for c in (R5090(1), R3080(0), R3080(2)))
        self.store.record(_meta(self.rig), _SUMMARY, source="test")

    def test_reference_rig_still_measured(self):
        hit = self.store.lookup(_meta(self.rig))
        self.assertIsNotNone(hit)
        self.assertEqual(hit["provenance"], "measured")

    def test_no_foreign_inventory_inherits_the_rig_anchor(self):
        foreign = {
            "sm89 mixed in": (R5090(0), R3080(1), R4090(2)),
            "pure sm89": (R4090(0), R4090(1), R4090(2)),
            "other sm86 tier": (R5090(0), R3090(1), R3090(2)),
            "subset (2 of 3)": (R5090(0), R3080(1)),
            "superset (4)": (R5090(0), R3080(1), R3080(2), R3080(3)),
        }
        for what, cards in foreign.items():
            labels = tuple(ci.class_label(c) for c in cards)
            self.assertIsNone(self.store.lookup(_meta(labels)), what)

    def test_the_key_spells_the_uncalibrated_arch(self):
        key = graphmem.anchor_key(_meta(tuple(ci.class_label(c) for c in (R4090(0), R3080(1)))))
        self.assertTrue(key.startswith("v3|"), key)
        self.assertIn("cards:RTX3080+RTX4090/24564MiB/sm89", key)


if __name__ == "__main__":
    unittest.main()
