# SPDX-License-Identifier: Apache-2.0
"""#1377 step 4b -- ONE replay seam in list_devices(), not one per caller.

Walls 6 (W48 bootstrap predicate), 7 (widest_layer_terms signature, TypeError
inside choose_host_ledger) and 9 (same_form_candidates) all lived in the
LAUNCHER, before the first rank. A lane probe cannot see them structurally, and
those three cost three windows. Running the real launcher at the desk needs
recorded cards, and that is the only seam it was missing.

WHY THE SEAM IS HERE AND NOT IN `resolve_cards()`. I first reported
`launcher.py:1593 resolve_cards()` as "the only consumer" of `list_devices`.
That was unchecked and wrong: measured on this tree, 45 files import
`registry.nvml` and there are 23 `list_devices()` call sites outside
video_enhance. Five sit on the boot path -- `rig_fingerprint`,
`card_totals_from_nvml`, `_device_infos`, `_real_cards`,
`_cards_from_homogeneous_node`. A replay point in `resolve_cards` alone would
give the launcher recorded cards while those five kept reading the RUNNING
machine: one boot, two machines, and the seam invisible until metal --
DRY-RUN-MISST-LIVE-BOX exactly.

`test_a_second_boot_path_reader_sees_the_same_cards` is that argument as a
measurement rather than a claim.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml
from sglang.test.test_utils import CustomTestCase

CARDS = [
    {"index": 0, "uuid": "GPU-aaa", "name": "NVIDIA GeForce RTX 5090",
     "total_bytes": 34191769600, "reserved_bytes": 0, "pci_bus_id": "0:1:0"},
    {"index": 1, "uuid": "GPU-bbb", "name": "NVIDIA GeForce RTX 3080",
     "total_bytes": 21474836480, "reserved_bytes": 0, "pci_bus_id": "0:2:0"},
    {"index": 2, "uuid": "GPU-ccc", "name": "NVIDIA GeForce RTX 3080",
     "total_bytes": 21474836480, "reserved_bytes": 0, "pci_bus_id": "0:3:0"},
]


class _Armed:
    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(CARDS, fh)
        self.saved = os.environ.get(nvml.ENV_NVML_REPLAY)
        os.environ[nvml.ENV_NVML_REPLAY] = self.path
        return self

    def __exit__(self, *a):
        if self.saved is None:
            os.environ.pop(nvml.ENV_NVML_REPLAY, None)
        else:
            os.environ[nvml.ENV_NVML_REPLAY] = self.saved
        os.unlink(self.path)


class TheSeamIsOneSeamForEveryReader(CustomTestCase):
    def test_unset_means_the_live_path_and_not_an_empty_list(self):
        """None, never []. An empty list is a legitimate answer on a GPU-less
        box, so conflating the two would make a desk run look like a machine
        without cards instead of an unconfigured one."""
        saved = os.environ.pop(nvml.ENV_NVML_REPLAY, None)
        try:
            self.assertIsNone(nvml._replay_devices())
        finally:
            if saved is not None:
                os.environ[nvml.ENV_NVML_REPLAY] = saved

    def test_the_recorded_rows_become_the_SAME_DeviceInfo(self):
        """Not a second implementation of the device list: one type, one field
        set, so no consumer can tell the paths apart."""
        with _Armed():
            got = nvml.list_devices()
        self.assertEqual([c.uuid for c in got], ["GPU-aaa", "GPU-bbb", "GPU-ccc"])
        self.assertEqual(got[0].total_bytes, 34191769600)
        self.assertTrue(all(isinstance(c, nvml.DeviceInfo) for c in got))

    def test_the_launcher_resolves_cards_through_it(self):
        from sglang.srt.weg2 import launcher as lc

        with _Armed():
            cards = lc.order_cards(lc.resolve_cards())
        self.assertEqual(len(cards), 3)
        self.assertEqual({c.uuid for c in cards},
                         {"GPU-aaa", "GPU-bbb", "GPU-ccc"})

    def test_a_second_boot_path_reader_sees_the_same_cards(self):
        """THE PLACEMENT ARGUMENT, MEASURED. `card_totals_from_nvml` is one of
        the five boot-path readers that a seam in `resolve_cards` would have
        left on the live box. It must come through the same seam."""
        from sglang.srt.registry import arbiter

        with _Armed():
            totals = arbiter.card_totals_from_nvml()
        self.assertEqual(len(totals), 3,
                         "a second reader did not honour the seam -- the "
                         "replay would hand one boot two machines")


if __name__ == "__main__":
    unittest.main()
