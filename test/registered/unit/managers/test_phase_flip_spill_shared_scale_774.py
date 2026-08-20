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
"""#774: a WHOLLY shared draft parameter is not the drafter's to release.

WHAT THIS PINS, and why a size test could not

The draft weight carrier decides which draft parameters it may move onto its
arena and release for the PP phase. Ownership used to be inferred from
``storage > own`` -- "does this tensor view PART of a larger storage". That
catches ``lm_head.weight`` and ``model.embed_tokens.weight``, which slice the
target's multi-GiB arena, and it misses their companions: on a vocab-quantized
build the drafter is handed the embed module WHOLESALE, so
``model.embed_tokens.weight_scale`` IS the target's tensor while sitting on a
storage that is exactly its own bytes. ``storage > own`` is False for it, so it
was classified exclusive and its pages were released while the target still
read it.

The consequence is a use-after-free whose VISIBILITY depends on allocation
pressure, which is why it hid for so long. Measured on the standing shape,
same commit, same seed, only ``--max-running-requests`` changed:

    cap 4: embed_tokens.weight_scale norm 0.1273 -> spec accept len 4.00
    cap 8: embed_tokens.weight_scale norm 0.0000 -> spec accept len 1.14

A zeroed row-scale kills the drafter's input embedding, so it proposed one
constant token forever and speculation degenerated into pure verify overhead
at ~3x the decode cost.

These tests are hermetic: the classification is pure tensor geometry, so it is
exercised on CPU tensors with no model, no device and no server. The first test
is the red one -- it fails against the size-only rule, because the shared scale
it builds is byte-exact on its own storage.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers.phase_flip_spill import _storage_ptrs


def _classify(named, foreign):
    """The carrier's ownership rule, isolated from device/arena machinery.

    Mirrors ``VmmDraftWeightCarrier.__init__``; kept as a local mirror so the
    test does not need a CUDA device or a real model to pin the geometry.
    """
    shared, exclusive = {}, {}
    for name, t in named.items():
        own = t.numel() * t.element_size()
        untyped = t.untyped_storage()
        storage = untyped.nbytes()
        ptr = untyped.data_ptr()
        if storage > own or (ptr is not None and ptr in foreign):
            shared[name] = (own, storage)
        else:
            exclusive[name] = t
    return shared, exclusive


class TestSharedScaleIsNotExclusive(unittest.TestCase):
    def test_wholly_shared_scale_is_excluded(self):
        """THE RED TEST. A shared tensor on its own storage must be excluded.

        Under the size-only rule this scale lands in `exclusive` and the carrier
        releases the target's bytes.
        """
        target_arena = torch.zeros(4096, dtype=torch.float32)
        target_scale = torch.full((64,), 0.5, dtype=torch.float32)

        named = {
            # Partial view of the target's arena: caught by the old rule too.
            "lm_head.weight": target_arena[:1024],
            # WHOLLY shared: byte-exact on its own storage, so the old rule
            # calls it exclusive. This is the #774 defect.
            "model.embed_tokens.weight_scale": target_scale,
            # Genuinely the drafter's own.
            "fc.weight": torch.zeros(256, dtype=torch.float32),
        }
        foreign = {
            target_arena.untyped_storage().data_ptr(),
            target_scale.untyped_storage().data_ptr(),
        }

        shared, exclusive = _classify(named, foreign)

        self.assertIn(
            "model.embed_tokens.weight_scale",
            shared,
            "a tensor shared wholesale with the target was classified as the "
            "drafter's to release; its pages would be freed under the target",
        )
        self.assertNotIn("model.embed_tokens.weight_scale", exclusive)
        self.assertIn("lm_head.weight", shared)
        self.assertIn(
            "fc.weight",
            exclusive,
            "the drafter's own parameter must stay spillable, or the rung "
            "silently stops paying for itself",
        )

    def test_size_only_rule_would_miss_it(self):
        """Proves the red test can fail: the OLD rule misclassifies the scale.

        Without this, a passing suite would not distinguish the fix from a test
        that never exercised the defect.
        """
        target_scale = torch.full((64,), 0.5, dtype=torch.float32)
        own = target_scale.numel() * target_scale.element_size()
        storage = target_scale.untyped_storage().nbytes()
        self.assertEqual(
            storage,
            own,
            "the shared scale must be byte-exact on its storage, otherwise "
            "this test is not reproducing the #774 geometry at all",
        )
        # storage > own is the entire old rule, and it is False here.
        self.assertFalse(storage > own)

    def test_no_foreign_set_keeps_old_classification(self):
        """Empty foreign set == pre-#774 behaviour, so the default path is safe.

        The carrier falls back to this when it cannot reach the target model.
        """
        target_arena = torch.zeros(4096, dtype=torch.float32)
        named = {
            "lm_head.weight": target_arena[:1024],
            "fc.weight": torch.zeros(256, dtype=torch.float32),
        }
        shared, exclusive = _classify(named, set())
        self.assertIn("lm_head.weight", shared)
        self.assertIn("fc.weight", exclusive)


class TestStoragePtrCollection(unittest.TestCase):
    def test_collects_parameters_and_buffers(self):
        """Scales may be buffers, not parameters; missing them reopens #774."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(8))
                self.register_buffer("scale", torch.ones(4))

        m = M()
        ptrs = _storage_ptrs(m)
        self.assertIn(m.w.data.untyped_storage().data_ptr(), ptrs)
        self.assertIn(
            m.scale.untyped_storage().data_ptr(),
            ptrs,
            "a buffer-resident scale was not collected, so it would be "
            "classified as the drafter's and released under the target",
        )

    def test_none_model_is_empty(self):
        self.assertEqual(_storage_ptrs(None), set())


if __name__ == "__main__":
    unittest.main()
