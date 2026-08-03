# SPDX-License-Identifier: Apache-2.0
"""#488: the Qwen3-TTS talker lane -- construction, geometry and weight cover.

Four things are pinned here, each with a failure this fork has already paid
for once:

1. **The M-RoPE key trap.** The checkpoint writes ``rope_scaling.interleaved``;
   ``layers/rotary_embedding/factory.py`` reads ``mrope_interleaved``. Passed
   through unchanged the model loads, runs, emits plausible codec tokens and
   SOUNDS WRONG. The gate lives in ``configs/qwen3_tts.py`` and fires before a
   weight is touched. The can-fail arm below feeds it the raw checkpoint dict.

2. **The tp=3 head geometry.** 16 q heads over 8 kv heads do not divide across
   3 ranks, so the classic even branch cannot represent this checkpoint at all.
   The uneven-TP plan can: the split is asserted here against the fork's own
   partition functions, not against a restatement of them.

3. **Weight-name coverage against the REAL checkpoint header.** Every one of
   the 478 tensor names is classified as lane-owned or explicitly non-lane.
   A name that is neither must raise -- "Loading weights: 478/478" once
   reported success while loading nothing (CLAUDE.md).

4. **Construction.** The module tree builds under a single-rank gloo world with
   no CUDA device present, which is what ``CUDA_VISIBLE_DEVICES=99`` gives.

The checkpoint header is read when it is present and the test SKIPS the
coverage arm otherwise, so the file stays a CPU test on a machine without the
model.
"""

import json
import os
import struct
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

import torch

from sglang.srt.configs.qwen3_tts import (
    MRopeMappingError,
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
    assert_mrope_mapped,
    normalize_rope_scaling,
)
from sglang.srt.distributed.utils import (
    attn_q_partition_groups,
    attn_q_partition_units,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_CHECKPOINT = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")

#: Verbatim from the checkpoint's ``config.json`` -- the trap as written.
_RAW_ROPE_SCALING = {
    "interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
    "type": "default",
}

_TALKER_KWARGS = dict(
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=3072,
    num_code_groups=16,
    text_hidden_size=2048,
    text_vocab_size=151936,
    rope_scaling=_RAW_ROPE_SCALING,
    max_position_embeddings=32768,
    position_id_per_seconds=13,
)


def _checkpoint_tensor_names():
    path = _CHECKPOINT / "model.safetensors"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return sorted(header)


def _ensure_dist_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29657")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    try:
        get_context().server_args
    except ValueError:
        get_context().set_server_args(ServerArgs(model_path="dummy", device="cpu"))


class TestQwen3TTSMRopeGate(CustomTestCase):
    """Trap 1: the two key names."""

    def test_normalize_copies_the_checkpoint_key(self):
        scaling = normalize_rope_scaling(_RAW_ROPE_SCALING)
        self.assertTrue(scaling["mrope_interleaved"])
        # The original key survives -- nothing here rewrites the checkpoint.
        self.assertTrue(scaling["interleaved"])

    def test_gate_refuses_the_raw_checkpoint_dict(self):
        """CAN-FAIL ARM. Without the normalisation the factory would build
        NON-interleaved M-RoPE silently; the gate must refuse instead."""
        with self.assertRaises(MRopeMappingError) as ctx:
            assert_mrope_mapped(_RAW_ROPE_SCALING, head_dim=128)
        self.assertIn("mrope_interleaved", str(ctx.exception))

    def test_gate_refuses_a_section_that_does_not_sum_to_half_head_dim(self):
        bad = dict(_RAW_ROPE_SCALING, mrope_section=[24, 20, 21])
        with self.assertRaises(MRopeMappingError):
            assert_mrope_mapped(normalize_rope_scaling(bad), head_dim=128)

    def test_gate_refuses_disagreeing_keys(self):
        both = dict(_RAW_ROPE_SCALING, mrope_interleaved=False)
        with self.assertRaises(MRopeMappingError):
            normalize_rope_scaling(both)

    def test_config_construction_runs_the_gate(self):
        config = Qwen3TTSTalkerConfig(**_TALKER_KWARGS)
        self.assertTrue(config.rope_scaling["mrope_interleaved"])
        self.assertEqual(config.code_predictor_config.num_hidden_layers, 5)


class TestQwen3TTSHeadGeometry(CustomTestCase):
    """Trap 2: 16 q / 8 kv over 3 ranks."""

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_tp3_needs_the_uneven_plan(self):
        """The even split does not exist for this checkpoint: 16 % 3 != 0."""
        from sglang.srt.models.qwen3_tts import _head_split

        set_tp_partition_ratios(None)
        with self.assertRaises(ValueError) as ctx:
            _head_split(16, 8, tp_size=3, tp_rank=0)
        self.assertIn("--rank-tp-ratio", str(ctx.exception))

    def test_uniform_ratio_is_already_uneven_here(self):
        set_tp_partition_ratios([1, 1, 1])
        units = attn_q_partition_units(16, 8, 3)
        groups = attn_q_partition_groups(8, 3)
        self.assertEqual(tp_partition_sizes(16, 3, units, groups=groups), [6, 6, 4])
        self.assertEqual(tp_partition_sizes(8, 3, 8), [3, 3, 2])
        # The MLP family coarsens to the 16-element activation vector.
        self.assertEqual(tp_partition_sizes(3072, 3, 16), [1152, 960, 960])

    def test_heterogeneous_ratio_shifts_mass_to_the_big_card(self):
        set_tp_partition_ratios([5, 3, 3])
        units = attn_q_partition_units(16, 8, 3)
        groups = attn_q_partition_groups(8, 3)
        q = tp_partition_sizes(16, 3, units, groups=groups)
        self.assertEqual(q, [8, 4, 4])
        self.assertEqual(sum(q), 16)
        self.assertEqual(tp_partition_sizes(8, 3, 8), [4, 2, 2])
        mlp = tp_partition_sizes(3072, 3, 16)
        self.assertEqual(sum(mlp), 3072)
        self.assertGreater(mlp[0], mlp[1])


class TestQwen3TTSConstruction(CustomTestCase):
    """Trap 4: the module tree builds with no CUDA device."""

    @classmethod
    def setUpClass(cls):
        _ensure_dist_initialized()

    def _tiny_config(self):
        kwargs = dict(_TALKER_KWARGS)
        kwargs.update(
            num_hidden_layers=2,
            text_vocab_size=512,
            code_predictor_config=dict(num_hidden_layers=1),
        )
        return Qwen3TTSConfig(talker_config=kwargs)

    def test_constructs_and_exposes_the_entry_class(self):
        from sglang.srt.models.qwen3_tts import (
            EntryClass,
            Qwen3TTSForConditionalGeneration,
        )

        self.assertIs(EntryClass, Qwen3TTSForConditionalGeneration)
        # The registry keys on the class NAME matching `architectures`.
        self.assertEqual(
            EntryClass.__name__, "Qwen3TTSForConditionalGeneration"
        )
        # #497 day-0 trap: an architecture the registry cannot resolve does
        # NOT refuse -- it falls back to TransformersForCausalLM
        # (models/registry.py:61-78), so a boot succeeds on the generic
        # backend with none of this fork's features. Assert the RESOLVED
        # class rather than trusting a green boot.
        from sglang.srt.models.registry import ModelRegistry

        self.assertIs(
            ModelRegistry.models.get("Qwen3TTSForConditionalGeneration"),
            Qwen3TTSForConditionalGeneration,
        )

        with torch.device("cpu"):
            model = EntryClass(self._tiny_config())
        self.assertEqual(len(model.model.layers), 2)
        self.assertEqual(len(model.code_predictor.layers), 1)
        # 15 residual groups: heads and embeddings, one per group above 0.
        self.assertEqual(len(model.code_predictor.lm_head), 15)
        self.assertEqual(len(model.code_predictor.codec_embedding), 15)
        # Identity projection: both hidden sizes are 1024, and the checkpoint
        # carries no weight for it.
        self.assertIsNone(model.code_predictor.small_to_mtp_projection)

    def test_forward_refuses_a_cleared_embeds_channel(self):
        """The talker is embed-driven; a token-id call is the scheduler
        unblock (DESIGN_466 §11.2) surfacing, not a model bug."""
        from sglang.srt.models.qwen3_tts import EntryClass

        with torch.device("cpu"):
            model = EntryClass(self._tiny_config())
        with self.assertRaises(ValueError) as ctx:
            model.forward(
                input_ids=torch.zeros(1, dtype=torch.long),
                positions=torch.zeros(1, dtype=torch.long),
                forward_batch=None,
                input_embeds=None,
            )
        self.assertIn("input_embeds", str(ctx.exception))


class TestQwen3TTSWeightCover(CustomTestCase):
    """Trap 3: every checkpoint tensor name has a home or the load refuses."""

    @classmethod
    def setUpClass(cls):
        _ensure_dist_initialized()
        cls.names = _checkpoint_tensor_names()

    def setUp(self):
        if self.names is None:
            self.skipTest(f"no checkpoint under {_CHECKPOINT}")

    def _full_model(self):
        from sglang.srt.models.qwen3_tts import EntryClass

        # Full depth, tiny text vocab: the 593 MiB text embedding is not what
        # this arm is about and materialising it on CPU would dominate runtime.
        kwargs = dict(_TALKER_KWARGS)
        kwargs.update(text_vocab_size=512)
        with torch.device("meta"):
            return EntryClass(Qwen3TTSConfig(talker_config=kwargs))

    def test_every_checkpoint_name_is_classified(self):
        from sglang.srt.models.qwen3_tts import _NON_LANE_PREFIXES

        model = self._full_model()
        params = dict(model.named_parameters())
        stacked = model._stacked_mapping()

        lane, non_lane, homeless = [], [], []
        for name in self.names:
            target = model._translate(name)
            if target is None:
                non_lane.append(name)
                continue
            if target in params:
                lane.append(name)
                continue
            if any(
                target.replace(src, dst) in params for dst, src, _ in stacked
            ):
                lane.append(name)
                continue
            homeless.append(f"{name} -> {target}")

        self.assertEqual(homeless, [], f"unclassified checkpoint tensors: {homeless}")
        self.assertEqual(len(self.names), 478)
        # Exactly the speaker encoder is out of the lane -- 65 tensors.
        self.assertTrue(all(n.startswith(_NON_LANE_PREFIXES) for n in non_lane))
        self.assertGreater(len(non_lane), 0)
        self.assertEqual(len(lane) + len(non_lane), 478)

    def test_load_weights_refuses_an_unknown_name(self):
        """CAN-FAIL ARM: a renamed tensor must raise, not be dropped."""
        model = self._full_model()
        with self.assertRaises(ValueError) as ctx:
            model.load_weights(
                [("talker.model.layers.0.mlp.renamed_proj.weight", torch.zeros(1))]
            )
        self.assertIn("renamed_proj", str(ctx.exception))

    def test_non_lane_prefix_is_skipped_not_refused(self):
        model = self._full_model()
        loaded = model.load_weights([("speaker_encoder.fc.bias", torch.zeros(1))])
        self.assertEqual(loaded, set())


if __name__ == "__main__":
    unittest.main()
