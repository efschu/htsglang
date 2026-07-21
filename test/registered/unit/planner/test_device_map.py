# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""CPU tests for planner.device_map -- the CUDA-order <-> NVML-order bridge.

No GPU required: NVML is a fake object and the torch enumeration is mocked.
Covers the reference-box divergence (cuda:0 = 5090 = nvml:1), the documented
FASTEST_FIRST emulation fallback (marked "heuristic"), the exact torch/UUID
path (marked "torch"), the CUDA_VISIBLE_DEVICES guard, and the never-crash
contract on a GPU-less host.
"""

import os
import unittest
from unittest import mock

from sglang.srt.planner import device_map
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeCard:
    def __init__(self, name, uuid, total_mib):
        self.name = name
        self.uuid = uuid
        self.total = total_mib * 2**20


class _Mem:
    def __init__(self, total):
        self.total = total


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


def _box_rig():
    """THE reference box in NVML/PCI order: 3080, 5090, 3080."""
    return _FakeNvml([
        _FakeCard("NVIDIA GeForce RTX 3080", "GPU-aaaa", 20480),
        _FakeCard("NVIDIA GeForce RTX 5090", "GPU-bbbb", 32607),
        _FakeCard("NVIDIA GeForce RTX 3080", "GPU-cccc", 20480),
    ])


class TestEmulateCudaOrder(CustomTestCase):
    def test_box_shape_fastest_first(self):
        # NVML order [3080, 5090, 3080] -> the 5090 is the fastest card, so
        # FASTEST_FIRST gives it cuda:0; the 3080s keep PCI order (stable
        # tie-break) as cuda:1 / cuda:2. This is the measured real mapping
        # of the reference box (verified via torch + nvidia-smi).
        self.assertEqual(
            device_map.emulate_cuda_order(
                ["RTX 3080", "RTX 5090", "RTX 3080"]
            ),
            [1, 0, 2],
        )

    def test_unknown_names_keep_order(self):
        # Unknown cards all rank 0.0 -> the stable sort keeps NVML order
        # (identity), never crashes.
        self.assertEqual(
            device_map.emulate_cuda_order(["FooCard", "BarCard"]), [0, 1]
        )

    def test_empty(self):
        self.assertEqual(device_map.emulate_cuda_order([]), [])


class TestBuildDeviceMap(CustomTestCase):
    def test_fake_rig_falls_back_to_heuristic(self):
        # Fake UUIDs can never match the real torch enumeration -> the
        # documented FASTEST_FIRST emulation, marked as such.
        dm = device_map.build_device_map(nvml=_box_rig())
        self.assertEqual(dm.source, "heuristic")
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
        self.assertEqual(dm.to_json()["source"], "heuristic")

    def test_torch_uuid_bridge_is_exact(self):
        # torch's CUDA order (mocked): 5090 first, then the two 3080s in PCI
        # order -- matched by UUID, source "torch" (exact, not heuristic).
        with mock.patch.object(
            device_map,
            "_torch_cuda_uuids",
            return_value=["bbbb", "aaaa", "cccc"],
        ):
            dm = device_map.build_device_map(nvml=_box_rig())
        self.assertEqual(dm.source, "torch")
        self.assertEqual(dm.nvml_to_cuda(), {0: 1, 1: 0, 2: 2})

    def test_torch_count_mismatch_falls_back(self):
        # A torch view of a DIFFERENT device set (e.g. filtered) must not be
        # trusted: heuristic fallback.
        with mock.patch.object(
            device_map, "_torch_cuda_uuids", return_value=["bbbb"]
        ):
            dm = device_map.build_device_map(nvml=_box_rig())
        self.assertEqual(dm.source, "heuristic")

    def test_no_torch_disallowed(self):
        dm = device_map.build_device_map(nvml=_box_rig(), allow_torch=False)
        self.assertEqual(dm.source, "heuristic")

    def test_cvd_filtered_process_distrusts_torch(self):
        # CUDA_VISIBLE_DEVICES filters THIS process -> torch enumerates a
        # remapped subset, not the bare CUDA order the engine flags use.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertIsNone(device_map._torch_cuda_uuids())

    def test_gpu_less_host_never_crashes(self):
        class _Broken:
            def nvmlDeviceGetCount(self):
                raise RuntimeError("no NVML")

        dm = device_map.build_device_map(nvml=_Broken())
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)
        self.assertEqual(dm.nvml_to_cuda(), {})

    def test_cached_device_map_never_raises(self):
        with mock.patch.object(
            device_map, "build_device_map", side_effect=RuntimeError("boom")
        ):
            dm = device_map.device_map(refresh=True)
        self.assertEqual(dm.entries, ())
        self.assertIsNone(dm.source)
        # rebuild the real cache for other tests on this host.
        device_map.device_map(refresh=True)


if __name__ == "__main__":
    unittest.main()
