# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#589: a pinned rank must not read its CUDA_VISIBLE_DEVICES index as an NVML one.

The #392/#397/#406 device-order family, reappearing in the one place that had
looked safe: ``current_device_uuid``. A worker pinned with
``CUDA_VISIBLE_DEVICES=<n>`` used to resolve ``n`` against NVML's device list.
But ``n`` indexes CUDA's enumeration of the UNMASKED rig -- ``FASTEST_FIRST``
by default -- and NVML enumerates in PCI bus order. On the reference rig those
disagree by construction: the RTX 5090 is CUDA ordinal 0 and NVML index 1.

The 2026-08-05 window 5 is the field case reproduced here. Three pinned ranks
self-reported their cards and got:

    rank 0  CVD=0  really the 5090      reported a 3080   (wrong)
    rank 1  CVD=1  really a 3080        reported the 5090 (wrong, the swap)
    rank 2  CVD=2  really the other 3080  reported it     (right, by luck)

Rank 0 held 27762 MiB, which only the 32 GB card can hold, so the misreport is
independently provable from the window artifacts. The consequence is
OOM-shaped rather than cosmetic: the mem_ledger would have charged the 5090's
4195 MiB activation bound against a 3080 measured at 1766.

Rank 2 is why this file tests all three ranks. A suite that checked one rank
had a one-in-three chance of picking the coincidence and calling the bug fixed.
The control rig where the two orders agree is here for the same reason: a
falsifier built only on an agreeing rig passes with the defect fully in place.

No driver is touched. The NVML device list and the CUDA-ordinal bridge are
both injected, so the whole file runs at the desk under CUDA_VISIBLE_DEVICES=99.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import (
    DeviceInfo,
    DeviceNotFoundError,
    DeviceOrderUnresolvedError,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024**2

UUID_3080_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_B = "GPU-cccccccc-0000-0000-0000-000000000003"

BDF_3080_A = "00000000:01:00.0"
BDF_5090 = "00000000:2D:00.0"
BDF_3080_B = "00000000:41:00.0"

TOTAL_3080_MIB = 20480
TOTAL_5090_MIB = 32768


def _nvml_pci_bus_order():
    """NVML's view: PCI bus order, so the 5090 sits at index 1."""
    return [
        DeviceInfo(
            0, UUID_3080_A, "NVIDIA GeForce RTX 3080", TOTAL_3080_MIB * MIB, BDF_3080_A
        ),
        DeviceInfo(
            1, UUID_5090, "NVIDIA GeForce RTX 5090", TOTAL_5090_MIB * MIB, BDF_5090
        ),
        DeviceInfo(
            2, UUID_3080_B, "NVIDIA GeForce RTX 3080", TOTAL_3080_MIB * MIB, BDF_3080_B
        ),
    ]


#: CUDA's FASTEST_FIRST view of the same rig, as the unmasked launcher sees it.
#: This is what a CUDA_VISIBLE_DEVICES value actually indexes.
CUDA_ORDER_FASTEST_FIRST = {0: BDF_5090, 1: BDF_3080_A, 2: BDF_3080_B}

#: The control rig: both enumerations name the same card by the same number.
CUDA_ORDER_AGREEING = {0: BDF_3080_A, 1: BDF_5090, 2: BDF_3080_B}

#: Ground truth for the divergent rig: which card each CVD pin really binds.
TRUTH_DIVERGENT = {0: UUID_5090, 1: UUID_3080_A, 2: UUID_3080_B}

#: What the unfixed code returned: NVML index == CVD value. Rank 2 agrees with
#: the truth above, which is the coincidence this file refuses to rely on.
UNFIXED_ANSWER = {0: UUID_3080_A, 1: UUID_5090, 2: UUID_3080_B}


def _pinned_bridge(cuda_order, pin):
    """The CUDA-ordinal bridge as seen INSIDE a process pinned to ``pin``.

    The mask leaves exactly one card visible, and torch renumbers it to
    ordinal 0 -- that renumbering is the whole reason the pin value cannot be
    recovered from inside the process, and the reason resolving ordinal 0 by
    bus id is the only honest answer.
    """
    return {cuda_order[pin]: 0}


class TestPinnedCardIdentity589(CustomTestCase):
    def _resolve(self, pin, cuda_order=None, bridge=None):
        devices = _nvml_pci_bus_order()
        if bridge is None:
            bridge = _pinned_bridge(cuda_order or CUDA_ORDER_FASTEST_FIRST, pin)
        with patch.object(registry_nvml, "list_devices", return_value=devices), patch.object(
            registry_nvml, "_cuda_ordinals_by_bus", return_value=bridge
        ), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(pin)}, clear=False):
            os.environ.pop("CUDA_DEVICE_ORDER", None)
            return registry_nvml.current_device_uuid()

    def test_every_rank_of_the_window_resolves_its_real_card(self):
        """The falsifier. Unfixed, ranks 0 and 1 return each other's card."""
        for pin, expected in TRUTH_DIVERGENT.items():
            with self.subTest(rank=pin):
                self.assertEqual(self._resolve(pin), expected)

    def test_the_unfixed_answer_is_actually_different(self):
        """Proves the test above can fail: on this rig the NVML-index reading
        and the truth disagree for ranks 0 and 1. Without this, a suite could
        pass because both readings happen to coincide."""
        disagreeing = [
            pin for pin in TRUTH_DIVERGENT if UNFIXED_ANSWER[pin] != TRUTH_DIVERGENT[pin]
        ]
        self.assertEqual(disagreeing, [0, 1])
        self.assertEqual(UNFIXED_ANSWER[2], TRUTH_DIVERGENT[2])

    def test_rank_zero_does_not_report_a_3080(self):
        """The window's own evidence: rank 0 held 27762 MiB, which does not fit
        in a 20480 MiB card, so any answer naming a 3080 is refutable from the
        artifact alone."""
        uuid = self._resolve(0)
        card = {d.uuid: d for d in _nvml_pci_bus_order()}[uuid]
        self.assertGreaterEqual(card.total_mib, 27762)
        self.assertIn("5090", card.name)

    def test_the_control_rig_resolves_the_same_way(self):
        """Where the orders agree, index-reading and identity-reading coincide;
        the fix must not have quietly broken that case."""
        for pin, expected in ((0, UUID_3080_A), (1, UUID_5090), (2, UUID_3080_B)):
            with self.subTest(pin=pin):
                self.assertEqual(
                    self._resolve(pin, cuda_order=CUDA_ORDER_AGREEING), expected
                )

    def test_a_uuid_pin_answers_itself(self):
        devices = _nvml_pci_bus_order()
        with patch.object(registry_nvml, "list_devices", return_value=devices):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": UUID_5090}):
                self.assertEqual(registry_nvml.current_device_uuid(), UUID_5090)

    def test_no_bridge_and_no_declared_order_is_refused_not_guessed(self):
        """The #397 rule: with no way to know, say so. Returning the NVML card
        of the same number is exactly the defect."""
        with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
            self._resolve(0, bridge={})
        message = str(ctx.exception)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", message)
        self.assertIn("PCI_BUS_ID", message)
        # The refusal must not leak the wrong card as if it were the answer.
        self.assertNotIn("is GPU-aaaaaaaa", message)

    def test_a_declared_pci_bus_order_makes_the_index_readable(self):
        """CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA order == NVML order, so the
        literal reading is true and is allowed -- the launch path for GGUF
        recipes depends on it."""
        devices = _nvml_pci_bus_order()
        with patch.object(registry_nvml, "list_devices", return_value=devices), patch.object(
            registry_nvml, "_cuda_ordinals_by_bus", return_value={}
        ), patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        ):
            self.assertEqual(registry_nvml.current_device_uuid(), UUID_5090)

    def test_an_out_of_range_declared_index_is_named(self):
        devices = _nvml_pci_bus_order()
        with patch.object(registry_nvml, "list_devices", return_value=devices), patch.object(
            registry_nvml, "_cuda_ordinals_by_bus", return_value={}
        ), patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "9", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        ):
            with self.assertRaises(DeviceNotFoundError) as ctx:
                registry_nvml.current_device_uuid()
            self.assertIn("out of range", str(ctx.exception))

    def test_the_bridge_is_the_identity_map_not_a_second_one(self):
        """#397 consolidation: the answer must come from the shared map, so
        patching the map's only CUDA input is enough to steer it."""
        devices = _nvml_pci_bus_order()
        with patch.object(registry_nvml, "list_devices", return_value=devices), patch.object(
            registry_nvml, "_cuda_ordinals_by_bus", return_value={BDF_3080_B: 0}
        ), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            os.environ.pop("CUDA_DEVICE_ORDER", None)
            self.assertEqual(registry_nvml.current_device_uuid(), UUID_3080_B)


if __name__ == "__main__":
    unittest.main()
