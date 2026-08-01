# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#397: one card-identity map, and no silent ordering when it cannot answer.

Three separately-maintained bridges used to answer "which physical card is
index i": the #331 ``registry.nvml.IdentityMap``, ``server_args``'s own
PCI-bus mapping, and ``planner.device_map``'s resolver with its FASTEST_FIRST
emulation. Three answers to one question is three chances to disagree, and the
device-order family has already bitten four recorded times (the torch-vs-NVML
memory read, runbook 6.1, the #331 audit, #349 sweep-3 arm L / #392).

This file is the #392 falsifier pattern applied to the consolidation. Every
rig here has CUDA and NVML disagreeing in the reference shape -- the RTX 5090
at CUDA ordinal 0 and NVML index 1 -- plus one control rig where they agree.
A suite built on an agreeing rig would pass with every bug in place.

Two things are pinned:

  * AGREEMENT. Each migrated caller is asked which physical card an index
    names, and its answer must be the identity map's answer. Pinned per
    caller, not once: the defect is precisely that two callers answer
    differently, so a shared helper asserted once would not catch it.
  * REFUSAL. With no CUDA bridge, no caller may fall back to the index of the
    same number or to an emulated order. The heuristic's silent mis-order is
    reproduced here against the ordering rule it used, and the same rig now
    produces a NAMED error instead.

No driver is touched: the NVML device list, the CUDA-ordinal bridge and the
per-card memory reads are injected.
"""

import sys
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.planner import device_map as device_map_module
from sglang.srt.planner import flags as flags_module
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceInfo, DeviceOrderUnresolvedError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1024**2

UUID_3080_A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
UUID_5090 = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
UUID_3080_B = "GPU-cccccccc-0000-0000-0000-000000000003"

BDF_3080_A = "00000000:01:00.0"
BDF_5090 = "00000000:2D:00.0"
BDF_3080_B = "00000000:41:00.0"

TOTAL_3080_MIB = 20480
TOTAL_5090_MIB = 32768

#: The one true answer for every rig below: CUDA ordinal -> card uuid.
EXPECTED_CUDA_TO_UUID = {0: UUID_5090, 1: UUID_3080_A, 2: UUID_3080_B}
EXPECTED_CUDA_TO_NVML = {0: 1, 1: 0, 2: 2}
EXPECTED_CUDA_TO_TOTAL = {
    0: TOTAL_5090_MIB,
    1: TOTAL_3080_MIB,
    2: TOTAL_3080_MIB,
}


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
    number. Every migrated caller was already right here, which is why a
    suite built only on this rig proves nothing."""
    return {BDF_3080_A: 0, BDF_5090: 1, BDF_3080_B: 2}


class _FakePynvml:
    """Enough pynvml for the callers that still read a handle directly.

    Handles ARE NVML indices, so a caller that reuses a CUDA ordinal as an
    NVML index reads the wrong card's numbers out of this object -- which is
    the defect, made observable.
    """

    NVMLError = Exception

    def __init__(self, devices):
        self._devices = list(devices)

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        return None

    def nvmlSystemGetDriverVersion(self):
        return "580.00"

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


class _RigCase(CustomTestCase):
    """A rig whose CUDA and NVML orders disagree, injected at every seam."""

    cuda_bridge = staticmethod(_cuda_fastest_first)
    devices = staticmethod(_nvml_pci_bus_order)

    def setUp(self):
        devices = type(self).devices()
        self._pynvml = _FakePynvml(devices)
        self._patches = [
            patch.object(registry_nvml, "list_devices", lambda: list(devices)),
            patch.object(
                registry_nvml,
                "_cuda_ordinals_by_bus",
                lambda allow_cuda_init=False: type(self).cuda_bridge(),
            ),
            patch.dict(sys.modules, {"pynvml": self._pynvml}),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)
        # The planner adapter caches; every test builds its own view of the
        # injected rig and hands the cache back empty afterwards.
        device_map_module._CACHE = None
        self.addCleanup(setattr, device_map_module, "_CACHE", None)

    def _stop(self):
        for p in self._patches:
            p.stop()

    def imap(self):
        return registry_nvml.identity_map(allow_cuda_init=True)


# ===========================================================================
# Guard rail: if the fixtures stop biting, nothing below means anything.
# ===========================================================================
class FixtureDivergesTest(_RigCase):
    def test_the_two_orders_disagree_on_this_rig(self):
        imap = self.imap()
        self.assertEqual(imap.by_nvml_index(0).uuid, UUID_3080_A)
        self.assertEqual(imap.by_cuda_ordinal(0).uuid, UUID_5090)
        self.assertNotEqual(imap.by_nvml_index(0).uuid, imap.by_cuda_ordinal(0).uuid)

    def test_the_identity_map_is_the_reference_answer(self):
        imap = self.imap()
        self.assertEqual(
            {o: imap.by_cuda_ordinal(o).uuid for o in (0, 1, 2)},
            EXPECTED_CUDA_TO_UUID,
        )
        self.assertEqual(imap.cuda_to_nvml("test"), EXPECTED_CUDA_TO_NVML)


# ===========================================================================
# Per-caller agreement: every migrated caller resolves the same card.
# ===========================================================================
class MigratedCallersAgreeTest(_RigCase):
    def test_server_args_bridge_shell_matches(self):
        """``_torch_to_nvml_gpu_index_mapping`` is a delegating shell now."""
        self.assertEqual(
            server_args_module._torch_to_nvml_gpu_index_mapping(),
            EXPECTED_CUDA_TO_NVML,
        )

    def test_query_gpu_total_mib_matches(self):
        """#336's caller: the total of the card the PROCESS runs on."""
        self.assertEqual(
            server_args_module._query_gpu_total_mib([0, 1, 2], "--test-flag"),
            EXPECTED_CUDA_TO_TOTAL,
        )
        # Restated as the field case: CUDA 0 is a 32 GiB card here, and the
        # 20 GiB card sitting at NVML 0 is a different one.
        self.assertNotEqual(
            server_args_module._query_gpu_total_mib([0], "--test-flag")[0],
            TOTAL_3080_MIB,
        )

    def test_rank_gpu_card_resolution_matches(self):
        """#392's caller, unchanged by #397 -- pinned so the consolidation
        cannot drift it."""
        cards = server_args_module._resolve_rank_gpu_cards([0, 1, 2])
        self.assertEqual(
            {o: c.uuid for o, c in cards.items()}, EXPECTED_CUDA_TO_UUID
        )

    def test_uneven_perf_inventory_matches(self):
        """The hardware micro-probe keys the profile CACHE by these uuids, so
        a misattributed card persists a profile of another machine."""
        from sglang.srt import uneven_perf

        with patch("torch.cuda.device_count", return_value=3):
            gpus, driver = uneven_perf._nvml_gpu_inventory()
        self.assertEqual(driver, "580.00")
        self.assertEqual(
            {g["cuda_index"]: g["uuid"] for g in gpus}, EXPECTED_CUDA_TO_UUID
        )
        self.assertEqual(
            {g["cuda_index"]: g["total_mib"] for g in gpus},
            EXPECTED_CUDA_TO_TOTAL,
        )

    def test_planner_device_map_matches(self):
        dm = device_map_module.build_device_map()
        self.assertEqual(dm.source, device_map_module.IDENTITY_MAP_SOURCE)
        self.assertEqual(dm.cuda_to_nvml(), EXPECTED_CUDA_TO_NVML)
        self.assertEqual(dm.cuda_for_uuid(UUID_5090), 0)
        self.assertEqual(
            {e.cuda_index: e.total_mib for e in dm.entries},
            EXPECTED_CUDA_TO_TOTAL,
        )

    def test_planner_device_map_from_an_injected_handle_matches(self):
        """``live_metrics`` owns its NVML handle and injects it; the answer
        must not depend on who opened the driver."""
        dm = device_map_module.build_device_map(nvml=self._pynvml)
        self.assertEqual(dm.cuda_to_nvml(), EXPECTED_CUDA_TO_NVML)

    def test_flag_planner_cuda_index_matches(self):
        """The values that get WRITTEN into --base-gpu-id / --rank-gpu-id."""
        device_map_module.device_map(refresh=True)
        inventory = [
            {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB, "uuid": UUID_3080_A},
            {"name": "RTX 5090", "total_mib": TOTAL_5090_MIB, "uuid": UUID_5090},
            {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB, "uuid": UUID_3080_B},
        ]
        resolved = [flags_module._cuda_index_at(inventory, p) for p in range(3)]
        self.assertEqual([r[0] for r in resolved], [1, 0, 2])
        self.assertEqual({r[1] for r in resolved}, {"bridged"})
        # The single-gpu preset pins the 5090 by its CUDA index, not by its
        # NVML index / list position 1.
        pos, cuda_idx, why = flags_module._pick_single_gpu(inventory)
        self.assertEqual(pos, 1)
        self.assertEqual(cuda_idx, 0)
        self.assertIn("cuda:0", why)

    def test_energy_compute_share_bridge_matches(self):
        """The fourth local copy of the bridge, found by the straggler audit
        and migrated with the rest."""
        from sglang.srt.planner import energy

        self.assertEqual(energy._torch_to_nvml_index(), EXPECTED_CUDA_TO_NVML)

    def test_pd_topology_totals_match(self):
        """``disaggregation.topology`` still reaches the map through the
        deprecated shell; the answer has to be the same one."""
        from sglang.srt.disaggregation import topology

        self.assertEqual(topology.nvml_card_totals_mib(), EXPECTED_CUDA_TO_TOTAL)


# ===========================================================================
# Refusal: no bridge means no answer, never the index of the same number.
# ===========================================================================
class NoCudaBridgeTest(_RigCase):
    cuda_bridge = staticmethod(dict)

    def test_identity_map_names_what_it_could_not_place(self):
        with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
            self.imap().require_cuda_ordinals("a caller")
        message = str(ctx.exception)
        self.assertIn("a caller", message)
        self.assertIn("3 of 3", message)
        for uuid in (UUID_3080_A, UUID_5090, UUID_3080_B):
            self.assertIn(uuid, message)  # WHAT
        self.assertIn("Reason:", message)  # WHY

    def test_query_gpu_total_mib_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            server_args_module._query_gpu_total_mib([0], "--test-flag")
        self.assertIn("Refusing to guess", str(ctx.exception))

    def test_uneven_perf_inventory_refuses(self):
        from sglang.srt import uneven_perf

        with patch("torch.cuda.device_count", return_value=3):
            with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
                uneven_perf._nvml_gpu_inventory()
        self.assertIn("profile cache", str(ctx.exception))

    def test_planner_device_map_raises_and_the_cached_view_goes_empty(self):
        with self.assertRaises(DeviceOrderUnresolvedError):
            device_map_module.build_device_map()
        # The UI path must not crash -- but it must not invent an order
        # either. "No bridge" is the honest cached answer.
        dm = device_map_module.device_map(refresh=True)
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)
        self.assertEqual(dm.nvml_to_cuda(), {})

    def test_flag_planner_refuses_to_write_a_pin(self):
        device_map_module.device_map(refresh=True)
        inventory = [
            {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB, "uuid": UUID_3080_A},
            {"name": "RTX 5090", "total_mib": TOTAL_5090_MIB, "uuid": UUID_5090},
        ]
        with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
            flags_module._cuda_index_at(inventory, 1)
        message = str(ctx.exception)
        self.assertIn("RTX 5090", message)
        self.assertIn(UUID_5090, message)
        self.assertIn("Refusing to guess", message)
        # The preset degrades instead of emitting a guessed --base-gpu-id.
        pos, cuda_idx, why = flags_module._pick_single_gpu(inventory)
        self.assertEqual(pos, 1)
        self.assertIsNone(cuda_idx)
        self.assertIn("NOT pinned", why)
        self.assertIsNone(flags_module._pick_stock_subset(inventory, [1, 2], 2))

    def test_energy_leaves_an_unresolvable_rank_unattributed(self):
        """It used to add the rank's share to the NVML card of the same
        number, i.e. to a different physical card."""
        from sglang.srt.planner import energy

        notes = []
        cfg = type("_Cfg", (), {"rank_tp_ratio": [2, 1, 1], "rank_gpu_id": [0, 1, 2]})()
        share = energy._compute_share_by_nvml(cfg, ["a", "b", "c"], notes)
        self.assertEqual(share, [0.0, 0.0, 0.0])
        self.assertEqual(len(notes), 3)
        self.assertIn("could not be resolved", notes[0])

    def test_the_bridge_shell_stays_empty_rather_than_partial(self):
        """The shell's remaining callers read ``{}`` as 'no bridge'. A
        PARTIAL map would be read as 'these are the only cards'."""
        self.assertEqual(
            server_args_module._torch_to_nvml_gpu_index_mapping(), {}
        )


# ===========================================================================
# The heuristic's silent mis-order, reproduced against the rule it used.
# ===========================================================================
def _unknown_card_devices():
    """Two cards no SEED_CARDS entry knows, wired so CUDA reverses NVML."""
    return [
        DeviceInfo(0, "GPU-dddd-1", "Acme Accelerator X", 20480 * MIB, BDF_3080_A),
        DeviceInfo(1, "GPU-eeee-2", "Acme Accelerator Y", 32768 * MIB, BDF_5090),
    ]


class HeuristicMisorderTest(_RigCase):
    devices = staticmethod(_unknown_card_devices)
    cuda_bridge = staticmethod(lambda: {BDF_5090: 0, BDF_3080_A: 1})

    @staticmethod
    def _emulate_fastest_first(names, peak):
        """The deleted ``device_map.emulate_cuda_order``, verbatim in rule:
        stable sort by SEED_CARDS fp16 peak descending, unknown names rank
        0.0. Kept HERE, in the test, so the defect it produced stays
        reproducible after the implementation is gone."""
        order = sorted(range(len(names)), key=lambda i: (-peak(names[i]), i))
        out = [0] * len(names)
        for cuda_idx, pos in enumerate(order):
            out[pos] = cuda_idx
        return out

    def test_the_emulation_is_wrong_on_this_rig_and_says_nothing(self):
        """Pre-fix behaviour, reproduced.

        Both cards are unknown to SEED_CARDS, so both rank 0.0, so the stable
        sort keeps NVML order and the emulation answers 'cuda:0 is nvml:0'.
        The real CUDA order is the reverse. Nothing in the result says so --
        the caller got two plausible integers and a "heuristic" label that
        does not distinguish 'emulated and right' from 'emulated and wrong'.
        """
        names = ["Acme Accelerator X", "Acme Accelerator Y"]
        emulated = self._emulate_fastest_first(names, lambda _n: 0.0)
        self.assertEqual(emulated, [0, 1])  # what the old path returned
        truth = self.imap().cuda_to_nvml("truth")
        self.assertEqual(truth, {1: 0, 0: 1})  # what the cards actually are
        # nvml:0 would have been written into a flag as cuda:0; it is cuda:1.
        self.assertNotEqual(emulated[0], self.imap().by_nvml_index(0).cuda_ordinal)

    def test_post_fix_the_same_rig_resolves_exactly(self):
        dm = device_map_module.build_device_map()
        self.assertEqual(dm.cuda_to_nvml(), {0: 1, 1: 0})
        self.assertEqual(dm.source, device_map_module.IDENTITY_MAP_SOURCE)

    def test_post_fix_an_unresolvable_rig_is_named_not_emulated(self):
        with patch.object(
            registry_nvml, "_cuda_ordinals_by_bus", lambda allow_cuda_init=False: {}
        ):
            with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
                device_map_module.build_device_map()
        message = str(ctx.exception)
        self.assertIn("Acme Accelerator X", message)
        self.assertIn("Acme Accelerator Y", message)
        self.assertIn("Reason:", message)


# ===========================================================================
# Behaviour-neutral control: orders agree, every caller unchanged.
# ===========================================================================
class AgreeingOrdersTest(_RigCase):
    cuda_bridge = staticmethod(_cuda_agreeing_with_nvml)

    #: On this rig cuda == nvml for every card, so every caller must return
    #: the identity mapping -- and the numbers below are the ones the
    #: pre-#397 code produced too. This is the "nothing moved" pin.
    IDENTITY = {0: 0, 1: 1, 2: 2}

    def test_every_migrated_caller_is_unchanged(self):
        from sglang.srt import uneven_perf
        from sglang.srt.disaggregation import topology

        self.assertEqual(
            server_args_module._torch_to_nvml_gpu_index_mapping(), self.IDENTITY
        )
        self.assertEqual(
            server_args_module._query_gpu_total_mib([0, 1, 2], "--test-flag"),
            {0: TOTAL_3080_MIB, 1: TOTAL_5090_MIB, 2: TOTAL_3080_MIB},
        )
        self.assertEqual(
            topology.nvml_card_totals_mib(),
            {0: TOTAL_3080_MIB, 1: TOTAL_5090_MIB, 2: TOTAL_3080_MIB},
        )
        dm = device_map_module.build_device_map()
        self.assertEqual(dm.cuda_to_nvml(), self.IDENTITY)
        self.assertEqual(dm.cuda_for_uuid(UUID_5090), 1)
        with patch("torch.cuda.device_count", return_value=3):
            gpus, _ = uneven_perf._nvml_gpu_inventory()
        self.assertEqual(
            {g["cuda_index"]: g["uuid"] for g in gpus},
            {0: UUID_3080_A, 1: UUID_5090, 2: UUID_3080_B},
        )

    def test_flag_planner_is_unchanged(self):
        device_map_module.device_map(refresh=True)
        inventory = [
            {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB, "uuid": UUID_3080_A},
            {"name": "RTX 5090", "total_mib": TOTAL_5090_MIB, "uuid": UUID_5090},
            {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB, "uuid": UUID_3080_B},
        ]
        self.assertEqual(
            [flags_module._cuda_index_at(inventory, p)[0] for p in range(3)],
            [0, 1, 2],
        )


# ===========================================================================
# The offline planner: no card identity at all, so nothing to resolve.
# ===========================================================================
class OfflineInventoryTest(CustomTestCase):
    """``--gpu NAME:MIB`` specs declare no CUDA order.

    #392 already fixed the convention for manual ``HardwareSpec``s: the list
    position IS the meaning of the value. #397 keeps that and only renames it
    from the silent "identity" last resort to a "declared" source the preset
    text surfaces -- there is no live order here to get wrong.
    """

    INVENTORY = [
        {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB},
        {"name": "RTX 5090", "total_mib": TOTAL_5090_MIB},
    ]

    def test_declared_order_is_the_list_order(self):
        self.assertEqual(
            [flags_module._cuda_index_at(self.INVENTORY, p) for p in range(2)],
            [(0, "declared"), (1, "declared")],
        )

    def test_the_preset_says_the_order_is_declared(self):
        pos, cuda_idx, why = flags_module._pick_single_gpu(self.INVENTORY)
        self.assertEqual((pos, cuda_idx), (1, 1))
        self.assertIn("DECLARED", why)

    def test_an_explicit_cuda_index_still_wins(self):
        rig = [
            {"name": "RTX 3080", "total_mib": TOTAL_3080_MIB, "cuda_index": 1},
            {"name": "RTX 5090", "total_mib": TOTAL_5090_MIB, "cuda_index": 0},
        ]
        pos, cuda_idx, why = flags_module._pick_single_gpu(rig)
        self.assertEqual((pos, cuda_idx), (1, 0))
        self.assertNotIn("DECLARED", why)


if __name__ == "__main__":
    unittest.main()
