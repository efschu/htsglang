# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#406: ``CUDA_VISIBLE_DEVICES`` entries are CUDA positions, not NVML indices.

``min_visible_cuda_capability_no_init`` answers the once-per-process questions
-- a ``tl.constexpr`` baked into a Triton kernel at import, a JIT backend armed
for the whole process. While torch.cuda is uninitialized it reads NVML, and it
used to resolve each ``CUDA_VISIBLE_DEVICES`` entry by indexing the NVML-
ordered device list with it. NVML enumerates in PCI bus order; CUDA enumerates
FASTEST_FIRST by default. On this rig (RTX 5090 at NVML index 1, CUDA ordinal
0) the two disagree, so ``CUDA_VISIBLE_DEVICES=0`` -- the 5090 -- resolved to a
3080, and ``CUDA_VISIBLE_DEVICES=1`` -- a 3080 -- resolved to the 5090. The
second direction is the dangerous one: sm120 kernel variants armed for a
process that runs on sm86.

``_nvml_cuda_device0``, twenty lines below in the same file, already emulated
the enumeration order correctly. The fix shares that emulation
(``_nvml_devices_in_cuda_order``) instead of keeping a second, wrong answer to
the same question.

The #392 falsifier pattern: the rigs here have CUDA and NVML disagreeing in
the reference shape, plus a homogeneous control rig where the order cannot
matter. Each divergence case pins the exact pre-fix answer, so reverting the
fix turns those red and leaves the control green.

No GPU and no driver: NVML is an injected fake and torch.cuda is reported
uninitialized throughout.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.utils.common as common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SM86 = (8, 6)
SM120 = (12, 0)

UUID_3080_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_B = "GPU-cccccccc-0000-0000-0000-000000000003"

#: NVML order is PCI bus order, so the 5090 sits at index 1 -- the real shape
#: of this rig, and the whole reason the two enumerations diverge.
MIXED_RIG = [
    (8, 6, "NVIDIA GeForce RTX 3080", UUID_3080_A, "0000:01:00.0"),
    (12, 0, "NVIDIA GeForce RTX 5090", UUID_5090, "0000:2d:00.0"),
    (8, 6, "NVIDIA GeForce RTX 3080", UUID_3080_B, "0000:41:00.0"),
]

#: The same three cards in the order CUDA hands them out under the default
#: FASTEST_FIRST, written by hand rather than derived from the code under
#: test: position 0 is the 5090, the two 3080s follow.
CUDA_ORDER = [MIXED_RIG[1], MIXED_RIG[0], MIXED_RIG[2]]

#: The control: three identical cards. FASTEST_FIRST cannot reorder what has
#: one capability, so every answer here must be the pre-fix answer.
UNIFORM_RIG = [
    (8, 6, "NVIDIA GeForce RTX 3080", UUID_3080_A, "0000:01:00.0"),
    (8, 6, "NVIDIA GeForce RTX 3080", UUID_3080_B, "0000:41:00.0"),
]


def make_fake_pynvml(devices):
    """devices: list of ``(major, minor, name, uuid, bus_id)``."""
    return SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(devices),
        nvmlDeviceGetHandleByIndex=lambda i: devices[i],
        nvmlDeviceGetCudaComputeCapability=lambda d: (d[0], d[1]),
        nvmlDeviceGetName=lambda d: d[2],
        nvmlDeviceGetUUID=lambda d: d[3],
        nvmlDeviceGetPciInfo=lambda d: SimpleNamespace(busId=d[4]),
    )


class _CapabilityCase(CustomTestCase):
    """Ask the floor question against an injected rig and a chosen env."""

    def setUp(self):
        self._clear()
        self.addCleanup(self._clear)

    def _clear(self):
        common.min_visible_cuda_capability_no_init.cache_clear()
        common._nvml_cuda_device0.cache_clear()
        common._nvml_all_devices.cache_clear()

    def floor(self, devices=None, env=None):
        devices = MIXED_RIG if devices is None else devices
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        }
        clean_env.update(env or {})
        with mock.patch.dict(sys.modules, {"pynvml": make_fake_pynvml(devices)}):
            with mock.patch.dict(os.environ, clean_env, clear=True):
                with mock.patch.object(common, "is_cuda", lambda: True):
                    with mock.patch.object(
                        common.torch.cuda, "is_initialized", lambda: False
                    ):
                        self._clear()
                        try:
                            return common.min_visible_cuda_capability_no_init()
                        finally:
                            self._clear()


# ===========================================================================
# Guard rail: if the fixture stops biting, nothing below means anything.
# ===========================================================================
class FixtureDivergesTest(_CapabilityCase):
    def test_nvml_index_zero_is_a_3080_and_cuda_ordinal_zero_is_the_5090(self):
        self.assertEqual(MIXED_RIG[0][:2], SM86)
        self.assertEqual(MIXED_RIG[1][:2], SM120)
        with mock.patch.dict(sys.modules, {"pynvml": make_fake_pynvml(MIXED_RIG)}):
            self._clear()
            ordered, pci_order = common._nvml_devices_in_cuda_order()
        self.assertFalse(pci_order)
        self.assertEqual(ordered[0][2], UUID_5090)
        self.assertEqual(ordered[0][0], SM120)


# ===========================================================================
# The contract: an entry names the card CUDA would give position N to.
# ===========================================================================
class VisibleSetFollowsCudaOrderTest(_CapabilityCase):
    def test_position_zero_is_the_fastest_card(self):
        """``CUDA_VISIBLE_DEVICES=0`` pins the 5090 on this rig. Pre-fix the
        floor was NVML index 0's sm86 -- an sm86 kernel variant for a process
        that only ever runs on the 5090."""
        self.assertEqual(self.floor(env={"CUDA_VISIBLE_DEVICES": "0"}), SM120)

    def test_position_one_is_not_the_5090(self):
        """The dangerous direction, pinned. ``CUDA_VISIBLE_DEVICES=1`` is a
        3080; pre-fix it resolved to NVML index 1, the 5090, and the process
        armed sm120 variants its card cannot launch."""
        answer = self.floor(env={"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(answer, SM86)
        self.assertNotEqual(answer, SM120)

    def test_a_uuid_entry_is_order_free(self):
        self.assertEqual(
            self.floor(env={"CUDA_VISIBLE_DEVICES": f"GPU-{UUID_5090[4:8]}"}),
            SM120,
        )

    def test_pci_bus_order_makes_every_position_exact(self):
        """With CUDA_DEVICE_ORDER=PCI_BUS_ID the orders agree by definition,
        so position 1 is the 5090 and no conservative fallback is needed."""
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        self.assertEqual(self.floor(env={**env, "CUDA_VISIBLE_DEVICES": "1"}), SM120)
        self.assertEqual(self.floor(env={**env, "CUDA_VISIBLE_DEVICES": "0"}), SM86)


# ===========================================================================
# The unresolvable case: a floor is still answerable, and it is the low one.
# ===========================================================================
class UnresolvablePositionTest(_CapabilityCase):
    def test_a_mixed_rig_past_position_zero_falls_to_the_global_floor(self):
        """FASTEST_FIRST specifies only position 0. For a MINIMUM the honest
        answer is the floor over every card NVML reports: it can only be lower
        than the true floor, and low is the safe direction -- it forgoes an
        optimization instead of arming a variant a card cannot run.

        ``_nvml_cuda_device0`` refuses the same input, because it is asked for
        the IDENTITY of a card rather than for a bound over them; the two
        answers differ on purpose.
        """
        self.assertEqual(self.floor(env={"CUDA_VISIBLE_DEVICES": "1"}), SM86)
        self.assertEqual(self.floor(env={"CUDA_VISIBLE_DEVICES": "0,2"}), SM86)

    def test_the_floor_never_exceeds_any_visible_card(self):
        """The safety property behind both branches: whatever the entries can
        be resolved to, the answer is at most the floor of the cards CUDA
        actually exposes. Violating it upward is the failure that arms a
        kernel variant a visible card cannot launch."""
        for entry in ("0", "1", "2", "0,1", "1,2", "0,1,2"):
            with self.subTest(cvd=entry):
                answer = self.floor(env={"CUDA_VISIBLE_DEVICES": entry})
                visible = [CUDA_ORDER[int(i)][:2] for i in entry.split(",")]
                self.assertLessEqual(answer, min(visible))

    def test_an_unparsable_entry_still_stops_the_scan(self):
        self.assertIsNone(self.floor(env={"CUDA_VISIBLE_DEVICES": "x"}))
        self.assertIsNone(self.floor(env={"CUDA_VISIBLE_DEVICES": ""}))
        self.assertIsNone(self.floor(env={"CUDA_VISIBLE_DEVICES": "99"}))


# ===========================================================================
# Control: behaviour-neutral where the orders cannot diverge.
# ===========================================================================
class OrdersCannotDivergeTest(_CapabilityCase):
    def test_unset_is_the_floor_over_the_whole_box(self):
        self.assertEqual(self.floor(), SM86)

    def test_uniform_rig_is_unchanged_at_every_position(self):
        for entry in ("0", "1", "0,1"):
            with self.subTest(cvd=entry):
                self.assertEqual(
                    self.floor(
                        devices=UNIFORM_RIG, env={"CUDA_VISIBLE_DEVICES": entry}
                    ),
                    SM86,
                )

    def test_no_nvml_device_is_none_not_a_guess(self):
        self.assertIsNone(self.floor(devices=[]))


# ===========================================================================
# The adjacent implementation this fix reuses must keep its own answers.
# ===========================================================================
class Device0ResolutionUnchangedTest(_CapabilityCase):
    def _device0(self, devices, env):
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        }
        clean_env.update(env)
        with mock.patch.dict(sys.modules, {"pynvml": make_fake_pynvml(devices)}):
            with mock.patch.dict(os.environ, clean_env, clear=True):
                self._clear()
                try:
                    return common._nvml_cuda_device0()
                finally:
                    self._clear()

    def test_fastest_first_still_picks_the_5090(self):
        info = self._device0(MIXED_RIG, {})
        self.assertEqual(info.capability, SM120)
        self.assertIn("5090", info.name)

    def test_pci_order_still_uses_the_bus_id(self):
        info = self._device0(MIXED_RIG, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        self.assertEqual(info.capability, SM86)

    def test_a_mixed_rig_past_position_zero_is_still_refused(self):
        self.assertIsNone(self._device0(MIXED_RIG, {"CUDA_VISIBLE_DEVICES": "1"}))


if __name__ == "__main__":
    unittest.main()
