# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#505-A2-04: the PD topology may not invent a CUDA<->NVML order.

``disaggregation.topology.nvml_card_totals_mib`` reads per-card VRAM totals
out of NVML (PCI bus order) and has to re-key them to the CUDA ordinals the
placement flags speak (``--rank-gpu-id``, ``--disaggregation-prefill-gpus``).
When the CUDA side of the #331/#397 identity map could not be built it used to
log "assuming identical enumeration orders" and fall back to the IDENTITY map
-- on the reference rig (RTX 5090 at CUDA ordinal 0 / NVML index 1) that
attributes every card's capacity to a different card, and the boot-time
feasibility check then passes a plan that cannot fit.

This file is the #392 falsifier pattern applied to that caller: the NVML
device list and the CUDA-ordinal bridge are injected so that the two
enumerations disagree in the reference shape, and the bridge is then removed.
No driver is touched.

Three arms:

  * NO BRIDGE. The totals must come back as UNKNOWN (``None``), never as the
    NVML order relabelled -- and the plan that the fabricated identity made
    look feasible must be reported as not verifiable instead of passing.
  * BRIDGE PRESENT. The same rig with a working bridge still resolves exactly
    and still REJECTS the infeasible plan, so the refusal above is a refusal
    to guess and not a blanket loss of the check.
  * SINGLE CARD. A host with exactly one physical GPU needs no bridge; cuda:0
    can only be that card, so it stays answerable.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.topology import (
    TOPOLOGY_COLOCATED_CONGRUENT,
    apply_pd_topology,
    nvml_card_totals_mib,
    reindex_totals_cuda_order,
)
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceInfo
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

#: What the identity map answers on this rig: CUDA ordinal -> total MiB.
TRUE_CUDA_TOTALS = {
    0: TOTAL_5090_MIB,
    1: TOTAL_3080_MIB,
    2: TOTAL_3080_MIB,
}

#: What the identity fallback used to answer: the NVML order, relabelled.
#: Every entry names a different physical card than the one above.
FABRICATED_CUDA_TOTALS = {
    0: TOTAL_3080_MIB,
    1: TOTAL_5090_MIB,
    2: TOTAL_3080_MIB,
}

#: A decode rank pinned to CUDA ordinal 1 with a budget that fits the 5090
#: but not a 3080. Under the fabricated identity cuda:1 "is" the 5090, so the
#: plan boots; on the real rig cuda:1 is a 3080 and the plan cannot fit.
RANK_BUDGET_MIB = 28000
PREFILL_BUDGET_MIB = 2000


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


def _single_card():
    return [
        DeviceInfo(
            0, UUID_5090, "NVIDIA GeForce RTX 5090", TOTAL_5090_MIB * MIB, BDF_5090
        )
    ]


def _cuda_fastest_first():
    """torch's view: FASTEST_FIRST, so the 5090 is ordinal 0."""
    return {BDF_5090: 0, BDF_3080_A: 1, BDF_3080_B: 2}


class _FakePynvml:
    """Enough pynvml for a caller that reads handles positionally.

    Handles ARE NVML indices here, so a caller that reuses a CUDA ordinal as
    an NVML index reads another card's numbers -- the defect, made
    observable.
    """

    NVMLError = Exception

    def __init__(self, devices):
        self._devices = list(devices)

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        return None

    def nvmlDeviceGetCount(self):
        return len(self._devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        return self._devices[index]

    def nvmlDeviceGetName(self, handle):
        return handle.name

    def nvmlDeviceGetUUID(self, handle):
        return handle.uuid

    def nvmlDeviceGetMemoryInfo(self, handle):
        class _Mem:
            total = handle.total_bytes
            free = handle.total_bytes - 512 * MIB
            used = 512 * MIB

        return _Mem()

    def nvmlDeviceGetPciInfo(self, handle):
        class _Pci:
            busId = handle.pci_bus_id

        return _Pci()


def _congruent_args(gpu: int):
    """colocated-congruent on ONE card, budget pinned in absolute MiB."""
    return SimpleNamespace(
        disaggregation_topology=TOPOLOGY_COLOCATED_CONGRUENT,
        disaggregation_prefill_gpus=[gpu],
        disaggregation_prefill_layer_split=None,
        disaggregation_prefill_budget_mib=PREFILL_BUDGET_MIB,
        disaggregation_mode="null",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        rank_gpu_id=[gpu],
        base_gpu_id=0,
        gpu_id_step=1,
        rank_gpu_memory_mib=[RANK_BUDGET_MIB],
        mem_fraction_static=None,
        model_path="",
        enable_mixed_chunk=False,
        disaggregation_prefill_lane_interval=1,
    )


class _RigCase(CustomTestCase):
    """A rig whose CUDA and NVML orders disagree, injected at every seam."""

    devices = staticmethod(_nvml_pci_bus_order)
    cuda_bridge = staticmethod(_cuda_fastest_first)

    def setUp(self):
        devices = type(self).devices()
        self._patches = [
            patch.object(registry_nvml, "list_devices", lambda: list(devices)),
            patch.object(
                registry_nvml,
                "_cuda_ordinals_by_bus",
                lambda allow_cuda_init=False: type(self).cuda_bridge(),
            ),
            patch.dict(sys.modules, {"pynvml": _FakePynvml(devices)}),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for p in self._patches:
            p.stop()


class FixtureDivergesTest(_RigCase):
    """If the fixture stops biting, nothing below means anything."""

    def test_the_two_orders_disagree_on_this_rig(self):
        imap = registry_nvml.identity_map(allow_cuda_init=True)
        self.assertEqual(imap.by_nvml_index(0).uuid, UUID_3080_A)
        self.assertEqual(imap.by_cuda_ordinal(0).uuid, UUID_5090)
        self.assertNotEqual(TRUE_CUDA_TOTALS, FABRICATED_CUDA_TOTALS)


class NoCudaBridgeTest(_RigCase):
    """The bridge cannot be built. Nothing may be invented from that."""

    cuda_bridge = staticmethod(dict)

    def test_totals_are_unknown_not_the_nvml_order_relabelled(self):
        totals = nvml_card_totals_mib()
        self.assertNotEqual(
            totals,
            FABRICATED_CUDA_TOTALS,
            "the CUDA->NVML bridge is unavailable, so these totals name "
            "whichever card NVML happened to enumerate at that index",
        )
        self.assertIsNone(totals)

    def test_pure_reindex_refuses_the_identity_assumption(self):
        nvml_totals = {0: TOTAL_3080_MIB, 1: TOTAL_5090_MIB, 2: TOTAL_3080_MIB}
        self.assertIsNone(reindex_totals_cuda_order(nvml_totals, {}))

    def test_an_infeasible_plan_is_not_reported_as_feasible(self):
        """The damage: 28000 + 2000 MiB on cuda:1, a 20480 MiB 3080.

        Under the fabricated identity cuda:1 carried the 5090's 32768 MiB and
        the boot-time check passed the plan without a word.
        """
        args = _congruent_args(1)
        with self.assertLogs("sglang.srt.disaggregation.topology", "WARNING") as logs:
            plan = apply_pd_topology(args, setenv={})
        self.assertIsNotNone(plan)
        self.assertIsNone(plan.card(1).total_mib)
        text = "\n".join(logs.output)
        self.assertIn("GPU 1", text)
        self.assertIn("not computable", text)
        self.assertIn("UNVERIFIED", text)


class PartialCudaBridgeTest(_RigCase):
    """Only some cards are placeable (the CUDA_VISIBLE_DEVICES-masked case).

    The placed card keeps its real capacity; the rest stay unknown. A card
    the bridge could not place must not pick up the total of whichever card
    NVML enumerated at the same index.
    """

    cuda_bridge = staticmethod(lambda: {BDF_5090: 0})

    def test_unplaced_cards_get_no_capacity_at_all(self):
        totals = nvml_card_totals_mib()
        self.assertEqual(totals, {0: TOTAL_5090_MIB})
        self.assertNotIn(1, totals)
        self.assertNotIn(2, totals)

    def test_a_plan_on_an_unplaced_card_is_unverified(self):
        with self.assertLogs("sglang.srt.disaggregation.topology", "WARNING") as logs:
            plan = apply_pd_topology(_congruent_args(1), setenv={})
        self.assertIsNone(plan.card(1).total_mib)
        self.assertIn("UNVERIFIED", "\n".join(logs.output))


class CudaBridgePresentTest(_RigCase):
    """The refusal above is a refusal to GUESS, not a lost check."""

    def test_totals_resolve_to_the_real_cards(self):
        self.assertEqual(nvml_card_totals_mib(), TRUE_CUDA_TOTALS)

    def test_the_same_infeasible_plan_is_rejected(self):
        args = _congruent_args(1)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            apply_pd_topology(args, setenv={})

    def test_the_plan_the_fabrication_confused_it_with_still_passes(self):
        """cuda:0 IS the 5090, so the identical budget fits there."""
        plan = apply_pd_topology(_congruent_args(0), setenv={})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.card(0).total_mib, TOTAL_5090_MIB)


class SingleCardHostTest(_RigCase):
    """One physical GPU: cuda:0 can only be that card, bridge or not."""

    devices = staticmethod(_single_card)
    cuda_bridge = staticmethod(dict)

    def test_totals_stay_answerable_without_a_bridge(self):
        self.assertEqual(nvml_card_totals_mib(), {0: TOTAL_5090_MIB})

    def test_pure_reindex_agrees(self):
        self.assertEqual(
            reindex_totals_cuda_order({0: TOTAL_5090_MIB}, {}),
            {0: TOTAL_5090_MIB},
        )


if __name__ == "__main__":
    unittest.main()
