# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Falsifier for the M-RoPE key trap, plus talker geometry validation.

The trap: the checkpoint writes ``rope_scaling.interleaved`` and the rotary
factory reads ``rope_scaling.mrope_interleaved``. Unmapped, the factory builds
NON-interleaved M-RoPE and the model loads, runs, emits plausible codec tokens
and sounds wrong.

The requirement on this file is the CAN-FAIL PROOF: a gate nobody has watched
fail is not a gate. Every assert here has a paired negative test that shows the
check firing on the exact input it exists to reject, and
:func:`factory_would_interleave` mirrors the factory's read verbatim so the
falsifier tests the real predicate rather than a restatement of the fix.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_talker_config.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.translator.talker_config import (
    CHECKPOINT_INTERLEAVED_KEY,
    FACTORY_INTERLEAVED_KEY,
    MRopeMappingError,
    TalkerGeometry,
    assert_mrope_mapped,
    factory_would_interleave,
    normalize_rope_scaling,
    read_talker_geometry,
)

REAL_CHECKPOINT = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")

# Exactly what the real checkpoint carries.
CHECKPOINT_SCALING = {
    "interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
    "type": "default",
}
HEAD_DIM = 128


class TestTheTrapItself(unittest.TestCase):
    """The can-fail proof. These are the tests that justify the module."""

    def test_the_raw_checkpoint_dict_would_build_the_WRONG_rotary(self):
        # THE BUG, demonstrated. The checkpoint says interleaved: true, and the
        # factory's own read returns False on it. If this test ever starts
        # failing, the factory changed and the mapping may be unnecessary.
        self.assertTrue(CHECKPOINT_SCALING[CHECKPOINT_INTERLEAVED_KEY])
        self.assertFalse(
            factory_would_interleave(CHECKPOINT_SCALING),
            "the factory now reads the checkpoint key directly; re-check "
            "whether normalize_rope_scaling is still needed",
        )

    def test_normalisation_fixes_it(self):
        normalized = normalize_rope_scaling(CHECKPOINT_SCALING)
        self.assertTrue(factory_would_interleave(normalized))
        # And the original key survives, so nothing downstream that reads it
        # starts disagreeing with the factory.
        self.assertTrue(normalized[CHECKPOINT_INTERLEAVED_KEY])

    def test_the_assert_FIRES_on_the_unmapped_dict(self):
        # The can-fail proof: feed the gate exactly the input it exists to
        # reject and watch it raise.
        with self.assertRaises(MRopeMappingError) as ctx:
            assert_mrope_mapped(CHECKPOINT_SCALING, HEAD_DIM)
        message = str(ctx.exception)
        self.assertIn(FACTORY_INTERLEAVED_KEY, message)
        self.assertIn("sounds wrong", message)

    def test_the_assert_PASSES_on_the_mapped_dict(self):
        assert_mrope_mapped(
            normalize_rope_scaling(CHECKPOINT_SCALING),
            HEAD_DIM,
            source=CHECKPOINT_SCALING,
        )

    def test_a_checkpoint_that_does_not_want_interleaving_is_left_alone(self):
        scaling = dict(CHECKPOINT_SCALING, interleaved=False)
        normalized = normalize_rope_scaling(scaling)
        self.assertFalse(factory_would_interleave(normalized))
        assert_mrope_mapped(normalized, HEAD_DIM, source=scaling)

    def test_contradictory_keys_are_refused_rather_than_resolved(self):
        scaling = dict(CHECKPOINT_SCALING)
        scaling[FACTORY_INTERLEAVED_KEY] = False  # disagrees with interleaved=True
        with self.assertRaises(MRopeMappingError) as ctx:
            normalize_rope_scaling(scaling)
        self.assertIn("refusing to guess", str(ctx.exception))


class TestGeometryValidation(unittest.TestCase):
    def test_a_missing_mrope_section_is_refused(self):
        scaling = {"interleaved": True, "rope_type": "default"}
        with self.assertRaises(MRopeMappingError) as ctx:
            assert_mrope_mapped(normalize_rope_scaling(scaling), HEAD_DIM)
        self.assertIn("collapse", str(ctx.exception))

    def test_a_section_that_does_not_sum_to_half_head_dim_is_refused(self):
        # The factory silently auto-corrects this, after which the positions
        # no longer mean what the checkpoint was trained with.
        scaling = dict(CHECKPOINT_SCALING, mrope_section=[24, 20, 21])
        with self.assertRaises(MRopeMappingError) as ctx:
            assert_mrope_mapped(normalize_rope_scaling(scaling), HEAD_DIM)
        self.assertIn("auto-correct", str(ctx.exception))

    def test_the_real_section_sums_correctly(self):
        self.assertEqual(sum(CHECKPOINT_SCALING["mrope_section"]), HEAD_DIM // 2)

    def test_none_rope_scaling_is_refused(self):
        with self.assertRaises(MRopeMappingError):
            normalize_rope_scaling(None)


class TestAgainstTheRealCheckpoint(unittest.TestCase):
    @unittest.skipUnless(REAL_CHECKPOINT.exists(), "checkpoint not downloaded")
    def test_the_real_config_loads_and_is_validated(self):
        geometry = read_talker_geometry(REAL_CHECKPOINT)
        self.assertIsInstance(geometry, TalkerGeometry)
        # Measured from the weights in DESIGN §1.1.1.
        self.assertEqual(geometry.num_hidden_layers, 28)
        self.assertEqual(geometry.hidden_size, 1024)
        self.assertEqual(geometry.num_attention_heads, 16)
        self.assertEqual(geometry.num_key_value_heads, 8)
        self.assertEqual(geometry.head_dim, 128)
        self.assertEqual(geometry.num_code_groups, 16)
        self.assertEqual(geometry.text_hidden_size, 2048)
        self.assertEqual(geometry.code_predictor_layers, 5)
        self.assertEqual(geometry.code_predictor_vocab_size, 2048)

    @unittest.skipUnless(REAL_CHECKPOINT.exists(), "checkpoint not downloaded")
    def test_the_real_geometry_carries_the_mapped_key(self):
        geometry = read_talker_geometry(REAL_CHECKPOINT)
        self.assertTrue(
            factory_would_interleave(geometry.rope_scaling),
            "the geometry read from the real checkpoint would build "
            "non-interleaved M-RoPE",
        )

    @unittest.skipUnless(REAL_CHECKPOINT.exists(), "checkpoint not downloaded")
    def test_frame_arithmetic_matches_the_design(self):
        geometry = read_talker_geometry(REAL_CHECKPOINT)
        # 12.5 Hz frames x 16 codebooks = 200 codes per second of audio.
        self.assertEqual(geometry.position_id_per_seconds, 13)
        self.assertEqual(geometry.num_code_groups, 16)
        self.assertGreater(geometry.codes_per_second(), 100)

    def test_a_config_without_a_talker_section_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "llama"}))
            with self.assertRaises(MRopeMappingError) as ctx:
                read_talker_geometry(Path(tmp))
            self.assertIn("not a Qwen3-TTS checkpoint", str(ctx.exception))

    def test_a_synthetic_checkpoint_with_the_trap_is_refused_at_read_time(self):
        # There must be NO path that yields a geometry object carrying the
        # trap -- including a hand-built checkpoint that names the factory key
        # with the wrong value.
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "talker_config": {
                    "hidden_size": 1024, "num_hidden_layers": 28,
                    "num_attention_heads": 16, "num_key_value_heads": 8,
                    "head_dim": 128, "vocab_size": 3072,
                    "intermediate_size": 3072, "rms_norm_eps": 1e-6,
                    "rope_theta": 1e6, "max_position_embeddings": 32768,
                    "num_code_groups": 16, "text_hidden_size": 2048,
                    "text_vocab_size": 151936, "position_id_per_seconds": 13,
                    "codec_bos_id": 2149, "codec_eos_token_id": 2150,
                    "codec_pad_id": 2148,
                    "rope_scaling": {
                        "interleaved": True,
                        "mrope_interleaved": False,
                        "mrope_section": [24, 20, 20],
                        "rope_type": "default",
                    },
                }
            }
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            with self.assertRaises(MRopeMappingError):
                read_talker_geometry(Path(tmp))


if __name__ == "__main__":
    unittest.main()
