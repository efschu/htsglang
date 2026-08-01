# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""CPU tests for planner.device_map -- the planner's view of the ONE map.

Since #397 this module is a shell over ``registry.nvml.IdentityMap``: it has
no resolver of its own and no fallback. What is left to test here is the
adapter's own behaviour -- the injected-NVML path ``live_metrics`` uses, the
DeviceMap lookups, and the never-raises contract of the cached accessor.

The consolidated resolution itself, and the refusal that replaced the
FASTEST_FIRST emulation, are pinned in ``test_device_order_bridges_397.py``
against a rig whose CUDA and NVML orders deliberately disagree. No GPU
required anywhere: NVML is a fake object and the CUDA bridge is injected.
"""

import unittest
from unittest import mock

from sglang.srt.planner import device_map
from sglang.srt.registry import nvml as registry_nvml
from sglang.srt.registry.nvml import DeviceOrderUnresolvedError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 2**20

BDF_3080_A = "00000000:01:00.0"
BDF_5090 = "00000000:2D:00.0"
BDF_3080_B = "00000000:41:00.0"


class _FakeCard:
    def __init__(self, name, uuid, total_mib, bus_id):
        self.name = name
        self.uuid = uuid
        self.total = total_mib * MIB
        self.bus_id = bus_id


class _Mem:
    def __init__(self, total):
        self.total = total


class _Pci:
    def __init__(self, bus_id):
        self.busId = bus_id


class _FakeNvml:
    def __init__(self, cards):
        self.cards = cards

    def nvmlDeviceGetCount(self):
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i):
        return self.cards[i]

    def nvmlDeviceGetName(self, h):
        return h.name

    def nvmlDeviceGetUUID(self, h):
        return h.uuid

    def nvmlDeviceGetMemoryInfo(self, h):
        return _Mem(h.total)

    def nvmlDeviceGetPciInfo(self, h):
        return _Pci(h.bus_id)


def _box_rig():
    """THE reference box in NVML/PCI order: 3080, 5090, 3080."""
    return _FakeNvml([
        _FakeCard("NVIDIA GeForce RTX 3080", "GPU-aaaa", 20480, BDF_3080_A),
        _FakeCard("NVIDIA GeForce RTX 5090", "GPU-bbbb", 32607, BDF_5090),
        _FakeCard("NVIDIA GeForce RTX 3080", "GPU-cccc", 20480, BDF_3080_B),
    ])


def _box_cuda_order():
    """torch's FASTEST_FIRST view of that rig: the 5090 is cuda:0."""
    return {BDF_5090: 0, BDF_3080_A: 1, BDF_3080_B: 2}


def _bridge(mapping):
    return mock.patch.object(
        registry_nvml,
        "_cuda_ordinals_by_bus",
        lambda allow_cuda_init=False: dict(mapping),
    )


class TestBuildDeviceMap(CustomTestCase):
    def setUp(self):
        device_map._CACHE = None
        self.addCleanup(setattr, device_map, "_CACHE", None)

    def test_injected_rig_resolves_through_the_identity_map(self):
        with _bridge(_box_cuda_order()):
            dm = device_map.build_device_map(nvml=_box_rig())
        self.assertEqual(dm.source, device_map.IDENTITY_MAP_SOURCE)
        self.assertEqual(dm.nvml_to_cuda(), {0: 1, 1: 0, 2: 2})
        self.assertEqual(dm.cuda_to_nvml(), {1: 0, 0: 1, 2: 2})
        # UUID lookup (normalized: 'GPU-bbbb' == 'bbbb').
        self.assertEqual(dm.cuda_for_uuid("GPU-bbbb"), 0)
        self.assertEqual(dm.cuda_for_uuid("bbbb"), 0)
        self.assertIsNone(dm.cuda_for_uuid("ffff"))
        self.assertEqual(dm.cuda_for_nvml(1), 0)
        e = dm.entries[1]
        self.assertEqual(e.nvml_index, 1)
        self.assertEqual(e.cuda_index, 0)
        self.assertEqual(e.total_mib, 32607)
        self.assertIn("5090", e.name)
        # JSON payload carries the source marker for the UI.
        self.assertEqual(dm.to_json()["source"], device_map.IDENTITY_MAP_SOURCE)

    def test_no_cuda_bridge_is_a_named_error_not_an_emulated_order(self):
        """Where the FASTEST_FIRST emulation used to answer (#397)."""
        with _bridge({}):
            with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
                device_map.build_device_map(nvml=_box_rig())
        message = str(ctx.exception)
        self.assertIn("the planner device map", message)
        self.assertIn("5090", message)
        self.assertIn("Reason:", message)

    def test_a_partially_bridged_rig_is_refused_whole(self):
        """A partial bridge is worse than none: the caller fills the gap with
        the NVML index and never learns that it guessed."""
        with _bridge({BDF_5090: 0}):
            with self.assertRaises(DeviceOrderUnresolvedError) as ctx:
                device_map.build_device_map(nvml=_box_rig())
        self.assertIn("2 of 3", str(ctx.exception))

    def test_no_torch_yields_the_empty_map(self):
        dm = device_map.build_device_map(nvml=_box_rig(), allow_torch=False)
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)

    def test_gpu_less_host_never_crashes(self):
        class _Broken:
            def nvmlDeviceGetCount(self):
                raise RuntimeError("no NVML")

        dm = device_map.build_device_map(nvml=_Broken())
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)
        self.assertEqual(dm.nvml_to_cuda(), {})

    def test_cards_without_a_pci_bdf_cannot_be_placed(self):
        """No BDF means no bridge for that card, and the all-or-nothing rule
        turns it into a named error rather than a plausible index."""
        rig = _FakeNvml([_FakeCard("Mystery Card", "GPU-dddd", 8192, None)])
        rig.nvmlDeviceGetPciInfo = mock.Mock(side_effect=RuntimeError("no pci"))
        with _bridge(_box_cuda_order()):
            with self.assertRaises(DeviceOrderUnresolvedError):
                device_map.build_device_map(nvml=rig)


class TestCachedAccessor(CustomTestCase):
    def setUp(self):
        device_map._CACHE = None
        self.addCleanup(setattr, device_map, "_CACHE", None)

    def test_cached_device_map_never_raises(self):
        with mock.patch.object(
            device_map, "build_device_map", side_effect=RuntimeError("boom")
        ):
            dm = device_map.device_map(refresh=True)
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)

    def test_an_unresolvable_order_caches_no_bridge_not_a_guess(self):
        with mock.patch.object(
            device_map,
            "build_device_map",
            side_effect=DeviceOrderUnresolvedError("nope"),
        ):
            dm = device_map.device_map(refresh=True)
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)
        self.assertEqual(dm.cuda_to_nvml(), {})


class TestNormUuid(CustomTestCase):
    def test_spellings_compare_equal(self):
        self.assertEqual(
            device_map.norm_uuid("GPU-AAAA-bbbb"), device_map.norm_uuid(b"aaaabbbb")
        )


if __name__ == "__main__":
    unittest.main()
