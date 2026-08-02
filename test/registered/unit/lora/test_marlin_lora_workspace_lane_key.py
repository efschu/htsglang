# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#274 §13.11 item 3: the Marlin LoRA workspace through the lane-keyed
accessor.

The workspace was the one ``resources.buffers`` entry managed past
``get_buffer`` with a raw string key. That is exactly the shape of the D3
workspace family (one name, two lanes, both writing a lock buffer), kept
harmless only by "no lane path reaches it today". These tests pin the fixed
acquisition and keep the pre-fix defect reproducible:

* per-lane isolation: two lane scopes get two tensors, the serving group
  (no scope) gets its own, and repeated acquisition inside one scope is the
  same object (keyed-lazy);
* device stays in the key: a second device makes a second entry instead of
  evicting the first (the old code's re-create-on-device-change, expressed
  as naming);
* the can-fail arm: the verbatim PRE-FIX access pattern -- a raw name key on
  ``resources.buffers`` -- hands every lane the same object under the very
  scopes the fixed path separates, so the isolation assertion is proven to
  discriminate, not to pass vacuously.
"""

import unittest

import torch

from sglang.srt.lora import lora_moe_runner_marlin as mod
from sglang.srt.runtime_context import get_context, lane_scope, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _install_fake_factory(created):
    """A CPU stand-in for ``marlin_make_workspace`` (CUDA-only in prod)."""

    def fake(device, max_blocks_per_sm=4):
        ws = torch.zeros(4, dtype=torch.int32)
        created.append((str(device), ws))
        return ws

    mod.marlin_make_workspace = fake


class TestLaneKeyedMarlinWorkspace(CustomTestCase):
    def setUp(self):
        reset_context()
        self.created = []
        self._had = hasattr(mod, "marlin_make_workspace")
        self._old = getattr(mod, "marlin_make_workspace", None)
        _install_fake_factory(self.created)

    def tearDown(self):
        if self._had:
            mod.marlin_make_workspace = self._old
        else:
            del mod.marlin_make_workspace
        reset_context()

    def test_lanes_get_their_own_workspace(self):
        serving = mod._acquire_workspace("cpu")
        with lane_scope(0):
            lane0 = mod._acquire_workspace("cpu")
        with lane_scope(1):
            lane1 = mod._acquire_workspace("cpu")
        self.assertIsNot(serving, lane0)
        self.assertIsNot(serving, lane1)
        self.assertIsNot(lane0, lane1)
        self.assertEqual(len(self.created), 3)

    def test_keyed_lazy_within_a_scope(self):
        a = mod._acquire_workspace("cpu")
        b = mod._acquire_workspace("cpu")
        self.assertIs(a, b)
        with lane_scope(0):
            c = mod._acquire_workspace("cpu")
            d = mod._acquire_workspace("cpu")
        self.assertIs(c, d)
        self.assertEqual(len(self.created), 2)

    def test_device_is_part_of_the_key(self):
        a = mod._acquire_workspace("cpu")
        b = mod._acquire_workspace("meta")
        self.assertIsNot(a, b)
        self.assertEqual([d for d, _ in self.created], ["cpu", "meta"])
        # The first device's entry survives the second's creation -- one
        # entry per device, not re-create-and-evict.
        self.assertIs(mod._acquire_workspace("cpu"), a)
        self.assertEqual(len(self.created), 2)

    def test_prefix_pattern_shares_across_lanes_can_fail_arm(self):
        """The pre-fix raw-key pattern under the same scopes: every lane the
        same tensor. This is what the fixed path is measured against."""

        def prefix_acquire(device):
            buffers = get_context().resources.buffers
            ws = buffers.get("marlin_lora_workspace")
            if ws is None:
                ws = mod.marlin_make_workspace(device, max_blocks_per_sm=4)
                buffers["marlin_lora_workspace"] = ws
            return ws

        serving = prefix_acquire("cpu")
        with lane_scope(0):
            lane0 = prefix_acquire("cpu")
        with lane_scope(1):
            lane1 = prefix_acquire("cpu")
        self.assertIs(serving, lane0)
        self.assertIs(serving, lane1)
        self.assertEqual(len(self.created), 1)


if __name__ == "__main__":
    unittest.main()
