# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Ledger registration AND a real park/restore. CPU, hermetic.

The standing rule this file answers to: registration that is only asserted is
decoration. So the tests move actual bytes -- park frees the tensors and the
module really becomes unusable, restore brings it back and the weights are
bit-identical. A park that quietly kept the tensors alive would pass a
"registered: true" assertion and fail the only thing the ledger is for.

CPU throughout: the park path is device-agnostic (`.to("meta")` /
`to_empty()`), so the mechanism is exercised without holding a card, which
matters because the GPU window is held by another agent.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_ledger.py -v
"""

import unittest

import torch
from torch import nn

from sglang.srt.translator.ledger import (
    OFFLOAD_CLASS,
    AudioAssetLedger,
    ParkError,
)


def tiny_module(in_features=64, out_features=128):
    torch.manual_seed(466)
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.Linear(out_features, in_features),
    )


class TestAssetClassIsDeclared(unittest.TestCase):
    def test_the_class_exists_in_the_runtime_register(self):
        from sglang.srt.model_executor.offload_register import OFFLOAD_CLASSES
        from sglang.srt.model_executor.short_term_offload_register import (
            ASSET_CLASSES,
            LadderRank,
        )

        self.assertIn(OFFLOAD_CLASS, OFFLOAD_CLASSES)
        descriptor = ASSET_CLASSES[OFFLOAD_CLASS]
        self.assertEqual(descriptor.ladder_rank, LadderRank.COLD_SECOND_MODEL)
        self.assertTrue(descriptor.wired)
        self.assertFalse(descriptor.va_stable_required)
        self.assertEqual(descriptor.dimension_presets, ("module",))

    def test_it_participates_in_the_global_eviction_order(self):
        # The point of registering rather than tracking privately: the
        # translator's assets must be comparable against everything else on
        # the ONE ladder, not on a private victim list.
        from sglang.srt.model_executor.short_term_offload_register import (
            ASSET_CLASSES,
            LadderRank,
        )

        ours = ASSET_CLASSES[OFFLOAD_CLASS].ladder_rank
        self.assertLess(
            ours,
            LadderRank.IDLE_SESSIONS,
            "an idle translator must be given up BEFORE a user's idle session",
        )
        self.assertLess(ours, LadderRank.ACTIVE_WORK)


class TestRegistration(unittest.TestCase):
    def test_registering_reports_a_real_size(self):
        ledger = AudioAssetLedger()
        module = tiny_module()
        asset = ledger.register("trunk", module)
        expected = sum(t.nbytes for t in module.state_dict().values())
        self.assertEqual(asset.size_bytes(), expected)
        self.assertGreater(asset.size_bytes(), 0)

    def test_double_registration_is_a_hard_error(self):
        ledger = AudioAssetLedger()
        ledger.register("trunk", tiny_module())
        with self.assertRaises(ParkError) as ctx:
            ledger.register("trunk", tiny_module())
        self.assertIn("lifecycle bug", str(ctx.exception))

    def test_an_unknown_asset_is_named_in_the_error(self):
        with self.assertRaises(ParkError) as ctx:
            AudioAssetLedger().park("nope")
        self.assertIn("nope", str(ctx.exception))

    def test_the_report_shows_residency(self):
        ledger = AudioAssetLedger()
        ledger.register_all([("trunk", tiny_module()), ("codec", tiny_module())])
        report = ledger.to_json()
        self.assertEqual(report["offload_class"], OFFLOAD_CLASS)
        self.assertEqual(len(report["assets"]), 2)
        self.assertGreater(report["resident_mib"], 0.0)
        self.assertTrue(all(not a["parked"] for a in report["assets"]))


class TestParkRestoreActuallyMovesBytes(unittest.TestCase):
    """The smoke the standing rule demands: not 'registered', but 'moved'."""

    def setUp(self):
        self.ledger = AudioAssetLedger()
        self.module = tiny_module()
        self.reference = {
            k: v.detach().clone() for k, v in self.module.state_dict().items()
        }
        self.probe = torch.randn(4, 64)
        with torch.inference_mode():
            self.expected = self.module(self.probe).clone()
        self.ledger.register("trunk", self.module)

    def test_park_frees_the_tensors_for_real(self):
        freed = self.ledger.park("trunk")
        self.assertGreater(freed, 0)
        self.assertTrue(self.ledger.get("trunk").parked)
        # THE proof: the parameters are gone, not copied. A park that kept the
        # tensors alive would free nothing and pass a weaker assertion.
        for tensor in self.module.state_dict().values():
            self.assertEqual(
                tensor.device.type, "meta",
                "park left a live tensor behind; no VRAM was actually freed",
            )

    def test_a_parked_module_cannot_be_used(self):
        self.ledger.park("trunk")
        with self.assertRaises(Exception):
            with torch.inference_mode():
                self.module(self.probe)

    def test_restore_brings_back_bit_identical_weights(self):
        self.ledger.park("trunk")
        elapsed = self.ledger.restore("trunk")
        self.assertGreater(elapsed, 0.0)
        self.assertFalse(self.ledger.get("trunk").parked)
        restored = self.module.state_dict()
        self.assertEqual(set(restored), set(self.reference))
        for key, original in self.reference.items():
            self.assertTrue(
                torch.equal(restored[key].cpu(), original),
                f"{key} changed across a park/restore cycle",
            )

    def test_the_module_computes_identically_after_a_round_trip(self):
        self.ledger.park("trunk")
        self.ledger.restore("trunk")
        with torch.inference_mode():
            after = self.module(self.probe)
        # Bit-identical, not merely close: nothing in a park is numeric.
        self.assertTrue(torch.equal(after, self.expected))

    def test_the_measured_restore_cost_replaces_the_estimate(self):
        self.assertIsNone(self.ledger.get("trunk").measured_restore_ms)
        self.ledger.park("trunk")
        self.ledger.restore("trunk")
        measured = self.ledger.get("trunk").measured_restore_ms
        self.assertIsNotNone(measured)
        self.assertGreater(measured, 0.0)
        self.assertEqual(self.ledger.to_json()["assets"][0]["restore_ms"], round(measured, 1))

    def test_repeated_cycles_are_stable(self):
        for _ in range(3):
            self.ledger.park("trunk")
            self.ledger.restore("trunk")
        with torch.inference_mode():
            after = self.module(self.probe)
        self.assertTrue(torch.equal(after, self.expected))

    def test_parking_twice_is_a_no_op_not_an_error(self):
        first = self.ledger.park("trunk")
        second = self.ledger.park("trunk")
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_restoring_a_resident_asset_is_a_no_op(self):
        self.assertEqual(self.ledger.restore("trunk"), 0.0)


class TestMultiModulePolicy(unittest.TestCase):
    def test_modules_park_independently_at_module_grain(self):
        ledger = AudioAssetLedger()
        modules = {name: tiny_module() for name in ("talker", "predictor", "codec")}
        ledger.register_all(modules.items())
        before = ledger.to_json()["resident_mib"]

        ledger.park("codec")

        report = ledger.to_json()
        self.assertLess(report["resident_mib"], before)
        parked = {a["name"]: a["parked"] for a in report["assets"]}
        self.assertTrue(parked["codec"])
        self.assertFalse(parked["talker"])
        self.assertFalse(parked["predictor"])

    def test_ensure_resident_restores_only_what_is_parked(self):
        ledger = AudioAssetLedger()
        ledger.register_all(
            [("talker", tiny_module()), ("codec", tiny_module())]
        )
        ledger.park("codec")
        restored = ledger.ensure_resident()
        self.assertEqual(list(restored), ["codec"])
        self.assertEqual(ledger.to_json()["assets"][0]["parked"], False)

    def test_park_all_then_restore_all(self):
        ledger = AudioAssetLedger()
        ledger.register_all(
            [("talker", tiny_module()), ("codec", tiny_module())]
        )
        freed = ledger.park_all()
        self.assertGreater(freed, 0)
        self.assertEqual(ledger.to_json()["resident_mib"], 0.0)
        ledger.ensure_resident()
        self.assertGreater(ledger.to_json()["resident_mib"], 0.0)


if __name__ == "__main__":
    unittest.main()
