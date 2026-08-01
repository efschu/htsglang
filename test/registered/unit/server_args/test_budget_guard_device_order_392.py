# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#392: the budget guard and the runtime must agree on WHICH card a rank is.

``--rank-gpu-id`` names CUDA ordinals -- the launcher writes each value into
the child process's ``CUDA_VISIBLE_DEVICES``, which CUDA resolves in its own
enumeration order. The per-card budget guard used to read its totals out of
NVML positionally, i.e. in PCI bus order. On a rig where the two enumerations
diverge that is a different card, so the guard checked one card and the rank
spent its budget on another.

#349 sweep-3 arm L is the field case reproduced here: the budget vector
19000,15000,15000 under a 2,1,1 ratio was ACCEPTED, because NVML index 0 is a
20480 MiB 3080 -- and then rank 0 bound the 5090 (CUDA ordinal 0), was handed
the fraction 19000/20480 = 0.928 of a 32768 MiB card = ~30.4 GiB, and OOMed.

Every rig in this file therefore has CUDA and NVML disagreeing in exactly that
shape, plus one control rig where they agree. A suite built on a rig where the
orders agree would pass with the bug still in place.

No driver is touched: the NVML device list, the CUDA-ordinal bridge and the
per-card memory read are all injected.
"""

import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.planner.feasibility import validate_plan_inputs
from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceInfo, MemoryInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.uneven_perf import PlanInputs
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

#: The arm-L vector: 19000 for the double shard of a 2,1,1 ratio, 15000 for
#: each single shard.
ARM_L_BUDGETS = [19000, 15000, 15000]
ARM_L_RATIO = [2, 1, 1]


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


def _cuda_fastest_first():
    """torch's view: FASTEST_FIRST, so the 5090 is ordinal 0."""
    return {BDF_5090: 0, BDF_3080_A: 1, BDF_3080_B: 2}


def _cuda_agreeing_with_nvml():
    """The control rig: both enumerations name the same card by the same
    number. The pre-#392 guard is correct here, which is the point."""
    return {BDF_3080_A: 0, BDF_5090: 1, BDF_3080_B: 2}


_FREE_MIB = {
    UUID_3080_A: 19800,
    UUID_5090: 32000,
    UUID_3080_B: 19800,
}


def _fake_memory_info(uuid):
    total = {
        UUID_3080_A: TOTAL_3080_MIB,
        UUID_5090: TOTAL_5090_MIB,
        UUID_3080_B: TOTAL_3080_MIB,
    }[uuid]
    return MemoryInfo(
        total_bytes=total * MIB,
        free_bytes=_FREE_MIB[uuid] * MIB,
        used_bytes=(total - _FREE_MIB[uuid]) * MIB,
    )


def rig(cuda_bridge=None, devices=None):
    """Patch the three driver-facing entry points the resolver uses."""
    return (
        patch.object(
            registry_nvml,
            "list_devices",
            lambda: devices() if devices else _nvml_pci_bus_order(),
        ),
        patch.object(
            registry_nvml,
            "_cuda_ordinals_by_bus",
            lambda allow_cuda_init=False: (
                cuda_bridge() if cuda_bridge else _cuda_fastest_first()
            ),
        ),
        patch.object(registry_nvml, "memory_info_for_uuid", _fake_memory_info),
    )


class _RigCase(CustomTestCase):
    cuda_bridge = staticmethod(_cuda_fastest_first)

    def setUp(self):
        self._patches = rig(cuda_bridge=type(self).cuda_bridge)
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for p in self._patches:
            p.stop()

    def resolve(self, tp_size=3, rank_gpu_id=(0, 1, 2), budgets=None, ratio=None):
        args = ServerArgs(
            model_path="dummy",
            tp_size=tp_size,
            rank_gpu_id=list(rank_gpu_id),
            rank_gpu_memory_mib=list(budgets) if budgets else None,
            rank_tp_ratio=list(ratio) if ratio else None,
        )
        args._handle_uneven_tp()
        return args


# ===========================================================================
# The contract: rank r is checked against the card rank r binds.
# ===========================================================================
class RankCardIdentityTest(_RigCase):
    def test_the_two_orders_disagree_on_this_rig(self):
        """Guard rail for the fixtures themselves.

        If this ever passes trivially, every other test in the file has
        stopped biting.
        """
        imap = registry_nvml.identity_map(allow_cuda_init=True)
        self.assertEqual(imap.by_nvml_index(0).uuid, UUID_3080_A)
        self.assertEqual(imap.by_cuda_ordinal(0).uuid, UUID_5090)
        self.assertNotEqual(imap.by_nvml_index(0).uuid, imap.by_cuda_ordinal(0).uuid)

    def test_totals_follow_the_cuda_ordinal_not_the_nvml_index(self):
        """The invariant, stated directly: for every rank r the total the
        guard reads is the total of the card r binds."""
        totals = {
            gid: card.total_mib
            for gid, card in server_args_module._resolve_rank_gpu_cards(
                [0, 1, 2]
            ).items()
        }
        self.assertEqual(
            totals, {0: TOTAL_5090_MIB, 1: TOTAL_3080_MIB, 2: TOTAL_3080_MIB}
        )
        uuids = {
            gid: card.uuid
            for gid, card in server_args_module._resolve_rank_gpu_cards(
                [0, 1, 2]
            ).items()
        }
        self.assertEqual(uuids, {0: UUID_5090, 1: UUID_3080_A, 2: UUID_3080_B})

    def test_arm_L_vector_converts_against_the_5090(self):
        """The sweep-3 arm L regression.

        Pre-fix the guard read NVML index 0 (a 3080) for rank 0 and derived
        the fraction 19000/20480; applied on the 5090 the rank then asked for
        ~30.4 GiB of a 32 GiB card and OOMed. The MiB value the operator
        wrote must survive the conversion.
        """
        args = self.resolve(budgets=ARM_L_BUDGETS, ratio=ARM_L_RATIO)
        fractions = args._rank_mem_fraction_static
        self.assertAlmostEqual(fractions[0], 19000 / TOTAL_5090_MIB, places=9)
        self.assertAlmostEqual(fractions[1], 15000 / TOTAL_3080_MIB, places=9)
        self.assertAlmostEqual(fractions[2], 15000 / TOTAL_3080_MIB, places=9)
        # Restated as the thing that actually matters: the fraction buys back
        # the MiB that were asked for, on the card the rank runs on.
        self.assertAlmostEqual(fractions[0] * TOTAL_5090_MIB, 19000, places=6)

    def test_budget_fitting_only_the_nvml_neighbour_is_refused(self):
        """25000 MiB on rank 1: fits NVML index 1 (the 5090), does not fit
        CUDA ordinal 1 (a 3080). The runtime binds the 3080, so the guard
        must refuse -- and name the card it refused for."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve(budgets=[19000, 25000, 15000], ratio=ARM_L_RATIO)
        message = str(ctx.exception)
        self.assertIn("Physical impossibility", message)
        self.assertIn("GPU 1", message)  # the rank's ordinal
        self.assertIn(UUID_3080_A, message)  # the physical card
        self.assertIn(BDF_3080_A, message)
        self.assertIn(str(TOTAL_3080_MIB), message)  # actual total
        self.assertIn("25000", message)  # requested
        self.assertIn("[1]", message)  # the rank

    def test_colocated_sum_is_checked_against_the_bound_card(self):
        """Two ranks on CUDA ordinal 1 (a 20480 MiB 3080): 2 x 15000 = 30000
        cannot fit, even though NVML index 1 (the 5090) would take it."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve(
                tp_size=3,
                rank_gpu_id=(0, 1, 1),
                budgets=[19000, 15000, 15000],
                ratio=[2, 1, 1],
            )
        message = str(ctx.exception)
        self.assertIn("Physical impossibility", message)
        self.assertIn(UUID_3080_A, message)
        self.assertIn("30000", message)

    def test_unresolvable_ordinal_is_refused_not_guessed(self):
        """An ordinal past the CUDA bridge is an error. Silently reading the
        NVML device of the same number is the #392 defect in its purest
        form."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve(
                tp_size=4,
                rank_gpu_id=(0, 1, 2, 3),
                budgets=[19000, 15000, 15000, 15000],
                ratio=[2, 1, 1, 1],
            )
        message = str(ctx.exception)
        self.assertIn("CUDA enumeration order", message)
        self.assertIn("[3]", message)


class NoCudaBridgeTest(_RigCase):
    cuda_bridge = staticmethod(dict)

    def test_missing_bridge_fails_fast_instead_of_falling_back(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve(budgets=ARM_L_BUDGETS, ratio=ARM_L_RATIO)
        self.assertIn("CUDA enumeration order", str(ctx.exception))


# ===========================================================================
# Control: the same vector on a rig whose orders agree.
# ===========================================================================
class AgreeingOrdersTest(_RigCase):
    cuda_bridge = staticmethod(_cuda_agreeing_with_nvml)

    def test_same_vector_is_accepted_and_maps_straight_through(self):
        args = self.resolve(budgets=ARM_L_BUDGETS, ratio=ARM_L_RATIO)
        fractions = args._rank_mem_fraction_static
        # Ordinal 0 is the 3080 here, and 19000 of 20480 is legal.
        self.assertAlmostEqual(fractions[0], 19000 / TOTAL_3080_MIB, places=9)
        self.assertAlmostEqual(fractions[1], 15000 / TOTAL_5090_MIB, places=9)
        self.assertAlmostEqual(fractions[2], 15000 / TOTAL_3080_MIB, places=9)

    def test_impossible_vector_is_still_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve(budgets=[26000, 15000, 15000], ratio=ARM_L_RATIO)
        self.assertIn(UUID_3080_A, str(ctx.exception))


# ===========================================================================
# The planner's mirror of the same check (HardwareSpec.gpu).
# ===========================================================================
def _planner_hardware():
    """The same rig as a live NVML spec: ``index`` is NVML, ``cuda_index`` is
    the space --rank-gpu-id is written in."""
    return HardwareSpec(
        gpus=(
            GpuDescriptor(
                index=0,
                name="NVIDIA GeForce RTX 3080",
                total_mib=TOTAL_3080_MIB,
                uuid=UUID_3080_A,
                cuda_index=1,
            ),
            GpuDescriptor(
                index=1,
                name="NVIDIA GeForce RTX 5090",
                total_mib=TOTAL_5090_MIB,
                uuid=UUID_5090,
                cuda_index=0,
            ),
            GpuDescriptor(
                index=2,
                name="NVIDIA GeForce RTX 3080",
                total_mib=TOTAL_3080_MIB,
                uuid=UUID_3080_B,
                cuda_index=2,
            ),
        ),
        source="nvml",
        cuda_index_source="torch",
    )


def _plan_inputs(budgets):
    return PlanInputs(
        tp_size=3,
        model_path="dummy",
        rank_gpu_id=[0, 1, 2],
        effective_vram_mib=list(budgets),
        rank_tp_ratio=list(ARM_L_RATIO),
    )


class PlannerFeasibilityTest(CustomTestCase):
    def test_lookup_resolves_the_cuda_ordinal(self):
        hw = _planner_hardware()
        self.assertEqual(hw.gpu(0).uuid, UUID_5090)
        self.assertEqual(hw.gpu(1).uuid, UUID_3080_A)
        self.assertEqual(hw.rank_gpu_id_of(hw.gpus[1]), 0)

    def test_arm_L_vector_is_feasible_for_the_planner_too(self):
        self.assertEqual(
            validate_plan_inputs(_plan_inputs(ARM_L_BUDGETS), _planner_hardware()),
            [],
        )

    def test_budget_fitting_only_the_nvml_neighbour_is_refused(self):
        errors = validate_plan_inputs(
            _plan_inputs([19000, 25000, 15000]), _planner_hardware()
        )
        self.assertTrue(errors)
        self.assertIn("Physical impossibility", errors[0])
        self.assertIn("GPU 1", errors[0])
        self.assertIn(str(TOTAL_3080_MIB), errors[0])

    def test_manual_spec_without_a_bridge_keeps_list_position(self):
        """Manual specs declare no CUDA order; the list position stays the
        meaning of the value, as GpuDescriptor documents."""
        hw = HardwareSpec(
            gpus=(
                GpuDescriptor(index=0, name="RTX 5090", total_mib=TOTAL_5090_MIB),
                GpuDescriptor(index=1, name="RTX 3080", total_mib=TOTAL_3080_MIB),
                GpuDescriptor(index=2, name="RTX 3080", total_mib=TOTAL_3080_MIB),
            ),
            source="manual",
        )
        self.assertEqual(hw.gpu(0).total_mib, TOTAL_5090_MIB)
        self.assertEqual(hw.rank_gpu_id_of(hw.gpus[2]), 2)


if __name__ == "__main__":
    unittest.main()
