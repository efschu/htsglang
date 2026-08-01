# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#400: the dual-group lane's own pool must be inside the per-card ledger.

#349 sweep-3 / followup arm L is the field case. ``--rank-gpu-memory-mib
19000,15000,15000`` under a ``2,1,1`` ratio with ``--dual-group-lane`` was
ACCEPTED by every guard, rank 0 verifiably bound the 32 GiB card (#392 holds),
and the process then died with

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB.
    GPU 0 has a total capacity of 31.34 GiB of which 63.69 MiB is free.
    ... this process has 31.14 GiB memory in use

*while loading the lane's complement weight shard*
(``build_dual_group_lanes -> ModelRunner -> build_lane_model ->
_load_lane_part``). 31.14 GiB in use against a 19000 MiB budget is the whole
statement: the lane allocates OUTSIDE the ledger the budget guard checks.

The lane's items are not unknowable. Its KV/state pool is the mandatory
``--dual-group-lane-budget-mib``, and its complement shard is a fixed unit
fraction of the model that the nesting plan spells out (BIG ``2,1,1`` ->
FAST ``2,2``: the host card additionally materializes 2 of 4 units). Both are
config-only, so both belong in the guard, not in a post-hoc boot log.

Contract pinned here, not arithmetic:

* an over-committed lane card is REFUSED at argument time, naming the card,
  every charged item and the sum (pre-fix: accepted, boots, OOMs);
* a lane that genuinely fits still boots;
* an unsizeable lane is refused as "cannot bound" rather than waved through;
* no ``--dual-group-lane`` means no new refusal at all.

No driver and no checkpoint are touched: the NVML device list, the CUDA-ordinal
bridge, the per-card memory read and the model's weight footprint are injected.
"""

import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceInfo, MemoryInfo
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024**2

UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_3080_B = "GPU-cccccccc-0000-0000-0000-000000000003"

BDF_5090 = "00000000:2D:00.0"
BDF_3080_A = "00000000:01:00.0"
BDF_3080_B = "00000000:41:00.0"

TOTAL_5090_MIB = 32100  # the 31.34 GiB the arm-L log reports
TOTAL_3080_MIB = 20480

#: The arm-L configuration, verbatim from boot_matrix/arms.py.
ARM_L_RATIO = [2, 1, 1]
ARM_L_BUDGETS = [19000, 15000, 15000]
ARM_L_LANE_POOL_MIB = 2048

#: The vehicle's block-quantized FP8 weight footprint. Named in
#: ``hull_needs_real_storage``: "a real hull of the entire model (28.75 GiB at
#: 27B)". Injected rather than read so the test needs no checkpoint.
MODEL_WEIGHT_MIB = 29440  # 28.75 GiB


def _nvml_devices():
    """PCI bus order -- the 5090 sits at NVML index 1, as on the rig."""
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
    """torch's view: the 5090 is CUDA ordinal 0, which is what rank 0 binds."""
    return {BDF_5090: 0, BDF_3080_A: 1, BDF_3080_B: 2}


_FREE_MIB = {UUID_5090: TOTAL_5090_MIB, UUID_3080_A: 19800, UUID_3080_B: 19800}


def _fake_memory_info(uuid):
    total = {
        UUID_5090: TOTAL_5090_MIB,
        UUID_3080_A: TOTAL_3080_MIB,
        UUID_3080_B: TOTAL_3080_MIB,
    }[uuid]
    return MemoryInfo(
        total_bytes=total * MIB,
        free_bytes=_FREE_MIB[uuid] * MIB,
        used_bytes=(total - _FREE_MIB[uuid]) * MIB,
    )


class _LaneRigCase(CustomTestCase):
    """The reference rig with the lane's weight footprint injected."""

    weight_mib = MODEL_WEIGHT_MIB

    def setUp(self):
        self._patches = [
            patch.object(registry_nvml, "list_devices", _nvml_devices),
            patch.object(
                registry_nvml,
                "_cuda_ordinals_by_bus",
                lambda allow_cuda_init=False: _cuda_fastest_first(),
            ),
            patch.object(registry_nvml, "memory_info_for_uuid", _fake_memory_info),
            patch.object(
                server_args_module,
                "_model_total_weight_mib",
                staticmethod(lambda server_args: type(self).weight_mib),
            ),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for p in self._patches:
            p.stop()

    def resolve(
        self,
        budgets=None,
        ratio=None,
        lane=True,
        lane_pool_mib=ARM_L_LANE_POOL_MIB,
        rank_gpu_id=(0, 1, 2),
        part_gpu_id=None,
        speed_dial=None,
    ):
        args = ServerArgs(
            model_path="dummy",
            tp_size=len(rank_gpu_id),
            rank_gpu_id=list(rank_gpu_id),
            rank_gpu_memory_mib=list(budgets or ARM_L_BUDGETS),
            rank_tp_ratio=list(ratio or ARM_L_RATIO),
            dual_group_lane=lane,
            dual_group_lane_budget_mib=lane_pool_mib if lane else None,
            dual_group_lane_part_gpu_id=list(part_gpu_id) if part_gpu_id else None,
            dual_group_lane_speed_dial=speed_dial,
        )
        args._handle_uneven_tp()
        return args


# ===========================================================================
# The defect: arm L, exactly as the boot matrix launches it.
# ===========================================================================
class ArmLIsRefusedTest(_LaneRigCase):
    def test_the_rig_binds_rank_0_to_the_5090(self):
        """Guard rail for the fixture. If rank 0 stopped landing on the big
        card this file would be testing #392, not #400."""
        cards = server_args_module._resolve_rank_gpu_cards([0, 1, 2])
        self.assertEqual(cards[0].uuid, UUID_5090)
        self.assertEqual(cards[0].total_mib, TOTAL_5090_MIB)

    def test_arm_L_is_refused_at_argument_time(self):
        """19000 (budget) + ~14720 (2/4 units of complement) + 2048 (lane
        pool) = ~35768 MiB against a 32100 MiB card. Cannot run.

        Pre-fix this configuration resolves without complaint and dies later
        in ``_load_lane_part``; that is the can-fail proof for this test.
        """
        with self.assertRaises(ValueError) as ctx:
            self.resolve()
        self.assertIn("dual-group lane", str(ctx.exception))

    def test_the_refusal_names_the_card_it_refused_for(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve()
        message = str(ctx.exception)
        self.assertIn("GPU 0", message)
        self.assertIn(UUID_5090, message)
        self.assertIn(BDF_5090, message)
        self.assertIn(str(TOTAL_5090_MIB), message)

    def test_the_refusal_itemizes_every_charged_post(self):
        """A bare "does not fit" would send the operator back to guessing.
        Every item that was charged has to be readable off the message."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve()
        message = str(ctx.exception)
        self.assertIn("--rank-gpu-memory-mib", message)
        self.assertIn("19000", message)
        self.assertIn("--dual-group-lane-budget-mib", message)
        self.assertIn(str(ARM_L_LANE_POOL_MIB), message)
        self.assertIn("complement", message)
        self.assertIn("2/4 units", message)

    def test_the_refusal_says_the_estimate_is_a_floor(self):
        """The charged posts are the ones that can be sized config-only. The
        message must not imply the rest were checked and found to fit."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve()
        self.assertIn("floor", str(ctx.exception).lower())
        self.assertIn("not priced here", str(ctx.exception))


# ===========================================================================
# The green direction: a lane that genuinely fits still boots.
# ===========================================================================
class FittingLaneTest(_LaneRigCase):
    def test_a_lane_that_fits_is_accepted(self):
        """Same rig, same ratio, a budget sized for the lane: 13000 + 14720 +
        2048 = 29768 MiB under the 32100 MiB card."""
        args = self.resolve(budgets=[13000, 15000, 15000])
        self.assertAlmostEqual(
            args._rank_mem_fraction_static[0], 13000 / TOTAL_5090_MIB, places=9
        )

    def test_the_boundary_is_the_card_total_not_a_margin(self):
        """No safety factor is invented on top of the physical limit: a
        configuration summing to exactly the card total is legal."""
        budget = TOTAL_5090_MIB - ARM_L_LANE_POOL_MIB - (MODEL_WEIGHT_MIB // 2)
        args = self.resolve(budgets=[budget, 15000, 15000])
        self.assertEqual(args.rank_gpu_memory_mib[0], budget)
        with self.assertRaises(ValueError):
            self.resolve(budgets=[budget + 1, 15000, 15000])

    def test_the_charge_is_the_dialled_pool_not_the_written_one(self):
        """--dual-group-lane-speed-dial only ever REDUCES the pool. Charging
        the un-dialled number would refuse configurations that fit, so the
        guard has to read the same resolver the lane reads."""
        budget = TOTAL_5090_MIB - (MODEL_WEIGHT_MIB // 2) - 300
        with self.assertRaises(ValueError):
            self.resolve(budgets=[budget, 15000, 15000], lane_pool_mib=2048)
        # Dial 1.0 is the minimum end of the scale: one eighth of the pool.
        self.resolve(budgets=[budget, 15000, 15000], lane_pool_mib=2048, speed_dial=1.0)

    def test_the_two_card_lane_charges_the_foreign_card(self):
        """--dual-group-lane-part-gpu-id moves the complement to another
        card; the flag's own help says that card's budget has to leave room
        for it. 15000 (rank 1's budget) + 14720 (the part) exceeds a 20480
        MiB 3080, so the placement is refused for THAT card."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve(budgets=[13000, 15000, 15000], part_gpu_id=[0, 1])
        message = str(ctx.exception)
        self.assertIn("GPU 1", message)
        self.assertIn(UUID_3080_A, message)
        self.assertIn("--dual-group-lane-part-gpu-id", message)


# ===========================================================================
# Unknowable is refused, not waved through.
# ===========================================================================
class UnboundableLaneTest(_LaneRigCase):
    weight_mib = None

    def test_an_unsizeable_lane_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve()
        message = str(ctx.exception)
        self.assertIn("cannot bound", message)
        self.assertIn("complement", message)

    def test_the_operator_has_a_named_door(self):
        """ "Cannot bound" must not be a dead end: the message names the
        override, and the override actually works."""
        with self.assertRaises(ValueError) as ctx:
            self.resolve()
        self.assertIn("SGLANG_DUAL_GROUP_LANE_SKIP_BUDGET_CHECK", str(ctx.exception))
        with patch.dict(
            "os.environ", {"SGLANG_DUAL_GROUP_LANE_SKIP_BUDGET_CHECK": "1"}
        ):
            self.resolve()


# ===========================================================================
# Backward compatibility: no lane, no new refusal.
# ===========================================================================
class NoLaneIsUnchangedTest(_LaneRigCase):
    def test_the_same_budgets_without_a_lane_still_resolve(self):
        """The arm-L budget vector on its own is legal (#392 pinned that) and
        must stay legal -- the lane is what over-commits the card, and this
        guard may only speak when a lane is configured."""
        args = self.resolve(lane=False)
        self.assertAlmostEqual(
            args._rank_mem_fraction_static[0], 19000 / TOTAL_5090_MIB, places=9
        )

    def test_the_plain_physical_impossibility_check_still_fires(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve(lane=False, budgets=[19000, 25000, 15000])
        self.assertIn("Physical impossibility", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
