# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#406: the BAR1 window is sized from the card that will host it.

``bar1_free`` asks NVML how much of a card's BAR1 aperture is still free, and
``window_for`` turns that number into the region size every rank of a group
maps. The NVML handle used to be fetched with ``nvmlDeviceGetHandleByIndex(
_ordinal(device))`` -- a CUDA ordinal passed into an NVML index. The two
enumerations are different orderings of the same cards, so on this rig (RTX
5090 at CUDA ordinal 0, NVML index 1) the window for the 5090 was sized from a
3080's free bytes. Transport-load-bearing, not display: too large and the
holder answers ENOMEM, too small and every message above the clipped size
silently drops to the gloo layer.

The #392 falsifier pattern: every rig here has the two orders disagreeing in
the reference shape, plus one control rig where they agree. The exact pre-fix
answer is pinned, so reverting the fix turns the divergence cases red and
leaves the control green.

No driver is touched: the NVML device list, the CUDA-ordinal bridge, pynvml
itself and the sysfs aperture read are injected.
"""

import os
import sys
import unittest
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import barlink_bar1 as bar1_module
from sglang.srt.distributed.device_communicators import barlink_matrix as matrix_module
from sglang.srt.distributed.device_communicators import (
    barlink_matrix_transport as transport,
)
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceInfo, DeviceOrderUnresolvedError
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

#: BAR1 aperture, gross, per card. Same size everywhere -- the point is the
#: FREE part, which differs.
BAR1_GROSS_MIB = 256

#: Free BAR1 per card, deliberately far apart so a wrong card is visible in
#: the returned number rather than only in a log line. The card the ranks
#: actually bind (the 5090) is the TIGHT one here, so reading its 3080
#: neighbour instead hands out a window that does not exist.
BAR1_FREE_MIB = {
    UUID_3080_A: 240,
    UUID_5090: 100,
    UUID_3080_B: 50,
}

#: What ``window_for`` asks for, and what it keeps back. Pinned in the
#: environment so the arithmetic below is exact.
REQUEST_MIB = 96
RESERVE_MIB = 32

#: 100 free - 32 reserve. The window the 5090 can actually carry.
CORRECT_WINDOW_MIB = 68

#: 240 free - 32 reserve = 208 >= 96, so the pre-fix code handed out the full
#: request. This is the wrong answer, named: a 96 MiB window on a card with
#: 100 MiB of free BAR1 minus the reserve nobody may take.
WRONG_WINDOW_MIB = REQUEST_MIB


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
    number. The pre-fix read was correct here, which is why a suite built
    only on this rig proves nothing."""
    return {BDF_3080_A: 0, BDF_5090: 1, BDF_3080_B: 2}


class _FakePynvml:
    """Enough pynvml for the BAR1 read. Handles ARE NVML indices, so a caller
    that passes a CUDA ordinal reads the neighbouring card out of this."""

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

    def nvmlDeviceGetBAR1MemoryInfo(self, handle):
        class _Bar1:
            bar1Total = BAR1_GROSS_MIB * MIB
            bar1Free = BAR1_FREE_MIB[handle.uuid] * MIB
            bar1Used = (BAR1_GROSS_MIB - BAR1_FREE_MIB[handle.uuid]) * MIB

        return _Bar1()


class _FakeDevice:
    """A torch device as far as this module cares: it carries an ordinal."""

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return f"cuda:{self.index}"


class _FakeWindow:
    def __init__(self, size):
        self.size = size


class _RigCase(CustomTestCase):
    """CUDA and NVML disagreeing, injected at every seam."""

    cuda_bridge = staticmethod(_cuda_fastest_first)
    #: ``{cuda ordinal: BDF}`` as ``bdf_of_card`` would answer, or None to
    #: make the PCI address unresolvable and force the ordinal route.
    bdf_by_ordinal = {0: BDF_5090, 1: BDF_3080_A, 2: BDF_3080_B}

    def setUp(self):
        devices = _nvml_pci_bus_order()
        self.pynvml = _FakePynvml(devices)
        table = type(self).bdf_by_ordinal

        def _bdf_of_card(device):
            if table is None:
                raise RuntimeError("no PCI address on this build")
            return table[device.index]

        self._patches = [
            patch.object(registry_nvml, "list_devices", lambda: list(devices)),
            patch.object(
                registry_nvml,
                "_cuda_ordinals_by_bus",
                lambda allow_cuda_init=False: type(self).cuda_bridge(),
            ),
            patch.dict(sys.modules, {"pynvml": self.pynvml}),
            patch.object(matrix_module, "bdf_of_card", _bdf_of_card),
            patch.object(
                bar1_module,
                "bar1_window",
                lambda bdf: _FakeWindow(BAR1_GROSS_MIB * MIB),
            ),
            patch.dict(
                os.environ,
                {
                    "SGLANG_BARLINK_BAR1_WINDOW_MIB": str(REQUEST_MIB),
                    "SGLANG_BARLINK_BAR1_RESERVE_MIB": str(RESERVE_MIB),
                },
            ),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)
        transport._LEDGER.clear()
        self.addCleanup(transport._LEDGER.clear)

    def _stop(self):
        for p in self._patches:
            p.stop()


# ===========================================================================
# Guard rail: if the fixture stops biting, nothing below means anything.
# ===========================================================================
class FixtureDivergesTest(_RigCase):
    def test_the_two_orders_disagree_on_this_rig(self):
        imap = registry_nvml.identity_map(allow_cuda_init=True)
        self.assertEqual(imap.by_nvml_index(0).uuid, UUID_3080_A)
        self.assertEqual(imap.by_cuda_ordinal(0).uuid, UUID_5090)

    def test_the_handle_of_the_ordinal_is_the_wrong_card(self):
        """The defect, made observable: NVML handle 0 is a 3080 with 240 MiB
        of free BAR1, while CUDA ordinal 0 is the 5090 with 100 MiB."""
        by_ordinal = self.pynvml.nvmlDeviceGetHandleByIndex(0)
        self.assertEqual(by_ordinal.uuid, UUID_3080_A)
        self.assertEqual(
            self.pynvml.nvmlDeviceGetBAR1MemoryInfo(by_ordinal).bar1Free,
            BAR1_FREE_MIB[UUID_3080_A] * MIB,
        )


# ===========================================================================
# The contract: the free bytes belong to the card the device binds.
# ===========================================================================
class Bar1FreeFollowsTheCardTest(_RigCase):
    def test_free_is_read_over_the_pci_address(self):
        free, gross, source = transport.bar1_free(_FakeDevice(0))
        self.assertEqual(source, "nvml")
        self.assertEqual(free, BAR1_FREE_MIB[UUID_5090] * MIB)
        self.assertEqual(gross, BAR1_GROSS_MIB * MIB)
        # The pre-fix answer, named.
        self.assertNotEqual(free, BAR1_FREE_MIB[UUID_3080_A] * MIB)

    def test_every_ordinal_lands_on_its_own_card(self):
        self.assertEqual(
            {
                ordinal: transport.bar1_free(_FakeDevice(ordinal))[0]
                for ordinal in (0, 1, 2)
            },
            {
                0: BAR1_FREE_MIB[UUID_5090] * MIB,
                1: BAR1_FREE_MIB[UUID_3080_A] * MIB,
                2: BAR1_FREE_MIB[UUID_3080_B] * MIB,
            },
        )

    def test_the_card_resolver_names_the_physical_card(self):
        card = transport.nvml_card_for_device(_FakeDevice(0), bdf=BDF_5090)
        self.assertEqual(card.uuid, UUID_5090)
        self.assertEqual(card.nvml_index, 1)

    def test_window_for_is_clipped_to_the_hosting_card(self):
        """End to end: 100 MiB free on the 5090 minus a 32 MiB reserve is a
        68 MiB window. Pre-fix the 3080's 240 MiB free let the full 96 MiB
        request through -- a window the hosting card cannot hold."""
        self.assertEqual(
            transport.window_for("tp", _FakeDevice(0)), CORRECT_WINDOW_MIB * MIB
        )
        self.assertNotEqual(
            transport.window_for("tp", _FakeDevice(0)), WRONG_WINDOW_MIB * MIB
        )


class OrdinalRouteTest(_RigCase):
    """No PCI address available: the map's CUDA-ordinal side answers, and it
    answers the same card. The ordinal is never used as an NVML index."""

    bdf_by_ordinal = None

    def test_free_still_follows_the_cuda_ordinal_bridge(self):
        free, gross, source = transport.bar1_free(_FakeDevice(0))
        self.assertEqual(source, "nvml")
        self.assertEqual(free, BAR1_FREE_MIB[UUID_5090] * MIB)
        self.assertNotEqual(free, BAR1_FREE_MIB[UUID_3080_A] * MIB)
        # No sysfs read without an address: gross comes from NVML's total.
        self.assertEqual(gross, BAR1_GROSS_MIB * MIB)


# ===========================================================================
# Refusal: no card, no read.
# ===========================================================================
class UnresolvableCardTest(_RigCase):
    cuda_bridge = staticmethod(dict)
    bdf_by_ordinal = None

    def test_the_resolver_raises_and_names_what_it_could_not_place(self):
        with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
            transport.nvml_card_for_device(_FakeDevice(0))
        message = str(ctx.exception)
        self.assertIn("cuda:0", message)
        self.assertIn(UUID_5090, message)
        self.assertIn(BDF_5090, message)

    def test_bar1_free_degrades_to_unknown_instead_of_another_card(self):
        with self.assertLogs(transport.logger, level="WARNING") as logs:
            free, _, source = transport.bar1_free(_FakeDevice(0))
        self.assertIsNone(free)
        self.assertEqual(source, "sysfs-gross")
        self.assertIn("could not be told which card", "\n".join(logs.output))

    def test_no_free_value_of_any_card_is_returned(self):
        free, _, _ = transport.bar1_free(_FakeDevice(0))
        self.assertNotIn(free, [mib * MIB for mib in BAR1_FREE_MIB.values()])


# ===========================================================================
# Control: the same rig with the orders agreeing.
# ===========================================================================
class AgreeingOrdersTest(_RigCase):
    cuda_bridge = staticmethod(_cuda_agreeing_with_nvml)
    bdf_by_ordinal = {0: BDF_3080_A, 1: BDF_5090, 2: BDF_3080_B}

    def test_ordinal_zero_is_nvml_zero_here(self):
        free, _, source = transport.bar1_free(_FakeDevice(0))
        self.assertEqual(source, "nvml")
        self.assertEqual(free, BAR1_FREE_MIB[UUID_3080_A] * MIB)

    def test_window_for_is_unchanged(self):
        self.assertEqual(transport.window_for("tp", _FakeDevice(0)), REQUEST_MIB * MIB)


if __name__ == "__main__":
    unittest.main()
