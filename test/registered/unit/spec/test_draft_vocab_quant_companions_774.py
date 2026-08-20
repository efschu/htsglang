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
"""#774: a quantized vocab is its weight AND its scale.

WHAT THIS PINS

``set_embed_and_head`` hands the draft the target's embedding ``.weight`` and
nothing else (``qwen3_5_mtp.py``: it assigns ``.weight`` and returns). On a
vocab-quantized checkpoint those rows are int8 and mean nothing without their
per-row scale -- and the draft's own embed module is NEVER LOADED, which the
sibling ``set_embed_and_head_modules`` states outright. The drafter therefore
dequantized the target's rows with whatever happened to sit in its own
uninitialized scale tensor.

That is not a latent hazard, it is what the rig was doing. Read at the share
point, before any decode, same commit, same seed, only --max-running-requests
changed:

    cap 4: draft scale norm 0.127293 -> spec accept len 4.00
    cap 8: draft scale norm 0.000000 -> spec accept len 1.14

The cap-4 value was stale bytes the allocation happened to land on; the cap-8
value was a zeroed block. A zeroed row-scale kills the drafter's input
embedding, so it proposed one constant token forever and speculation became
pure verify overhead. The healthy regime was an ACCIDENT of allocation.

These tests are hermetic: the sharing is plain attribute plumbing over
``torch.nn.Module``s, so it runs on CPU with no model, device or server.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class _Embed(torch.nn.Module):
    """Stand-in for a VocabParallelEmbedding with int8 rows."""

    def __init__(self, scale_value: float, rows: int = 8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(rows, 4))
        self.weight_scale = torch.nn.Parameter(torch.full((rows, 1), scale_value))


class _Inner(torch.nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed_tokens = embed


class _Model(torch.nn.Module):
    def __init__(self, scale_value: float):
        super().__init__()
        self.model = _Inner(_Embed(scale_value))
        self.lm_head = _Embed(scale_value)


class _Runner:
    def __init__(self, model):
        self.model = model


class _Worker:
    """Minimal stand-in carrying only what the method touches."""

    _VOCAB_QUANT_COMPANIONS = EagleDraftWorker._VOCAB_QUANT_COMPANIONS
    _share_vocab_quant_companions = (
        EagleDraftWorker._share_vocab_quant_companions
    )

    def __init__(self, draft_model):
        self.draft_runner = _Runner(draft_model)


class TestVocabQuantCompanionSharing(unittest.TestCase):
    def test_draft_scale_becomes_the_targets(self):
        """THE RED TEST. Without the fix the draft keeps its unloaded scale."""
        target = _Model(0.125)
        # The draft's own module was never loaded: zeros stand in for
        # "whatever this uninitialized block happens to contain".
        draft = _Model(0.0)
        worker = _Worker(draft)

        # Pre-condition: this is the #774 geometry, not something else.
        self.assertEqual(draft.model.embed_tokens.weight_scale.norm().item(), 0.0)

        worker._share_vocab_quant_companions(target)

        self.assertIs(
            draft.model.embed_tokens.weight_scale,
            target.model.embed_tokens.weight_scale,
            "the draft's embedding row-scale is not the target's; the draft "
            "would dequantize the target's rows with an unloaded scale",
        )
        self.assertIs(draft.lm_head.weight_scale, target.lm_head.weight_scale)
        self.assertGreater(
            draft.model.embed_tokens.weight_scale.norm().item(),
            0.0,
        )

    def test_unquantized_vocab_is_untouched(self):
        """No companions == byte-identical default path."""

        class _Plain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(8, 4))

        class _PlainModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner(_Plain())
                self.lm_head = _Plain()

        target, draft = _PlainModel(), _PlainModel()
        worker = _Worker(draft)
        worker._share_vocab_quant_companions(target)
        self.assertFalse(hasattr(draft.model.embed_tokens, "weight_scale"))

    def test_already_shared_module_is_left_alone(self):
        """The GGUF path shares modules wholesale; nothing to re-point."""
        target = _Model(0.125)
        draft = _Model(0.125)
        # Emulate wholesale module sharing.
        draft.model.embed_tokens = target.model.embed_tokens
        draft.lm_head = target.lm_head
        worker = _Worker(draft)
        worker._share_vocab_quant_companions(target)
        self.assertIs(
            draft.model.embed_tokens.weight_scale,
            target.model.embed_tokens.weight_scale,
        )

    def test_missing_models_are_a_noop(self):
        worker = _Worker(_Model(0.125))
        worker._share_vocab_quant_companions(None)  # must not raise

    def test_every_named_companion_is_carried(self):
        """A companion that is named but not carried reopens the defect."""
        target = _Model(0.125)
        draft = _Model(0.0)
        extra = {}
        for name in EagleDraftWorker._VOCAB_QUANT_COMPANIONS:
            # Plain tensors on purpose: a companion is a Parameter on one
            # quantization and a bare buffer on the next, and the sharing must
            # carry either. Clear any existing registration the same way the
            # implementation does, so the setup itself is not the thing under
            # test.
            t = torch.full((8, 1), 0.25)
            target.model.embed_tokens._parameters.pop(name, None)
            target.model.embed_tokens._buffers.pop(name, None)
            setattr(target.model.embed_tokens, name, t)
            extra[name] = t
        worker = _Worker(draft)
        worker._share_vocab_quant_companions(target)
        for name, t in extra.items():
            self.assertIs(
                getattr(draft.model.embed_tokens, name),
                t,
                f"companion {name!r} was named but not shared",
            )


if __name__ == "__main__":
    unittest.main()
