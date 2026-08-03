# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Shape contracts for the talker's prompt blocks and decode positions.

These are the tests that would have caught the decode-step rotary bug in one
run instead of an afternoon, and they cover the whole class rather than the
one instance.

The class: a tensor that is the right SIZE but the wrong SHAPE travels a long
way before anything complains. Flattening a prompt block from
``(1, T, 1024)`` to ``(1, T*1024)`` preserves the element count, so the
concatenation that consumes it still succeeds. Handing full-length positions
to M-RoPE on a one-token decode step does not raise either -- the query simply
BROADCASTS across the cache. Both surface far away, in an unrelated matmul.

**Prefill hides both**, because there query length equals cache length. That
asymmetry is exactly why every prefill-shaped check passed while decode was
broken, and it is why every negative test below is written at decode shape.

Each contract has a can-fail proof: it is shown raising on the precise input
it exists to reject.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_shape_contracts.py -v
"""

import unittest

import torch

from sglang.srt.translator.talker_config import (
    ShapeContractError,
    assert_position_contract,
    assert_prompt_block,
    assert_rotary_contract,
)

HIDDEN = 1024
HEAD_DIM = 128
AXES = 3          # M-RoPE sections for this checkpoint
PREFILL_LEN = 10
CACHE_LEN = 11    # after one decode step


class TestPromptBlockContract(unittest.TestCase):
    """One contract per prompt building block: role, text, speaker, codec, BOS/EOS."""

    def test_every_well_formed_block_passes(self):
        blocks = {
            "role": torch.zeros(1, 3, HIDDEN),
            "text": torch.zeros(1, 18, HIDDEN),
            "speaker": torch.zeros(1, 1, HIDDEN),
            "codec": torch.zeros(1, 7, HIDDEN),
            "bos_eos": torch.zeros(1, 2, HIDDEN),
        }
        for name, tensor in blocks.items():
            with self.subTest(block=name):
                shape = assert_prompt_block(name, tensor, HIDDEN)
                self.assertEqual(len(shape), 3)

    def test_a_FLATTENED_block_is_rejected(self):
        # The failure mode, demonstrated: same element count, wrong shape.
        good = torch.zeros(1, 18, HIDDEN)
        flat = good.reshape(1, 18 * HIDDEN)
        self.assertEqual(good.numel(), flat.numel())
        with self.assertRaises(ShapeContractError) as ctx:
            assert_prompt_block("text", flat, HIDDEN)
        self.assertIn("flattening failure mode", str(ctx.exception))

    def test_a_block_missing_its_batch_axis_is_rejected(self):
        with self.assertRaises(ShapeContractError):
            assert_prompt_block("text", torch.zeros(18, HIDDEN), HIDDEN)

    def test_a_wrong_hidden_size_is_rejected(self):
        # e.g. the pre-projection text width (2048) reaching a post-projection
        # slot, which is the other direction of the same seam.
        with self.assertRaises(ShapeContractError) as ctx:
            assert_prompt_block("text", torch.zeros(1, 18, 2048), HIDDEN)
        self.assertIn("hidden size", str(ctx.exception))

    def test_an_empty_block_is_rejected(self):
        with self.assertRaises(ShapeContractError):
            assert_prompt_block("codec", torch.zeros(1, 0, HIDDEN), HIDDEN)

    def test_a_wrong_batch_is_rejected(self):
        with self.assertRaises(ShapeContractError):
            assert_prompt_block("text", torch.zeros(2, 4, HIDDEN), HIDDEN, batch=1)

    def test_a_non_tensor_is_rejected(self):
        with self.assertRaises(ShapeContractError):
            assert_prompt_block("text", [[0.0] * HIDDEN], HIDDEN)


class TestDecodePositionContract(unittest.TestCase):
    """The invariant: len(position_ids) == query_len, never the cache length."""

    def test_prefill_passes(self):
        positions = torch.arange(PREFILL_LEN).view(1, 1, -1).repeat(AXES, 1, 1)
        assert_position_contract(
            positions, query_length=PREFILL_LEN, cache_length=PREFILL_LEN
        )

    def test_a_correct_decode_step_passes(self):
        # One token, one position -- the absolute position is 10, the LENGTH
        # is 1. Length is what the contract is about.
        positions = torch.tensor([[[10]]]).repeat(AXES, 1, 1)
        assert_position_contract(positions, query_length=1, cache_length=CACHE_LEN)

    def test_FULL_positions_on_a_decode_step_are_rejected(self):
        # THE BUG. Eleven positions for a one-token query: M-RoPE returns
        # cos/sin of length 11 and the query broadcasts across the cache
        # instead of raising.
        positions = torch.arange(CACHE_LEN).view(1, 1, -1).repeat(AXES, 1, 1)
        with self.assertRaises(ShapeContractError) as ctx:
            assert_position_contract(
                positions, query_length=1, cache_length=CACHE_LEN
            )
        message = str(ctx.exception)
        self.assertIn("CACHE length", message)
        self.assertIn("broadcast", message)

    def test_the_cache_length_hint_is_only_given_when_it_matches(self):
        # A length that is simply wrong should not be blamed on the cache.
        positions = torch.arange(5).view(1, 1, -1).repeat(AXES, 1, 1)
        with self.assertRaises(ShapeContractError) as ctx:
            assert_position_contract(
                positions, query_length=1, cache_length=CACHE_LEN
            )
        self.assertNotIn("CACHE length", str(ctx.exception))

    def test_a_wrong_mrope_axis_count_is_rejected(self):
        positions = torch.arange(4).view(1, 1, -1).repeat(2, 1, 1)
        with self.assertRaises(ShapeContractError) as ctx:
            assert_position_contract(positions, query_length=4, axes=AXES)
        self.assertIn("M-RoPE axes", str(ctx.exception))

    def test_a_two_dimensional_position_tensor_is_accepted(self):
        # Some call sites carry (batch, length) before the M-RoPE expansion.
        assert_position_contract(torch.tensor([[10]]), query_length=1)


class TestRotaryContract(unittest.TestCase):
    """The same invariant one step later, on cos/sin themselves."""

    def test_matching_lengths_pass(self):
        cos = torch.zeros(1, 1, HEAD_DIM)
        assert_rotary_contract(cos, cos, query_length=1, cache_length=CACHE_LEN)

    def test_FULL_LENGTH_cos_sin_on_a_decode_step_are_rejected(self):
        cos = torch.zeros(1, CACHE_LEN, HEAD_DIM)
        with self.assertRaises(ShapeContractError) as ctx:
            assert_rotary_contract(cos, cos, query_length=1, cache_length=CACHE_LEN)
        self.assertIn("CACHE length", str(ctx.exception))

    def test_prefill_cos_sin_pass(self):
        cos = torch.zeros(1, PREFILL_LEN, HEAD_DIM)
        assert_rotary_contract(
            cos, cos, query_length=PREFILL_LEN, cache_length=PREFILL_LEN
        )

    def test_both_cos_and_sin_are_checked(self):
        good = torch.zeros(1, 1, HEAD_DIM)
        bad = torch.zeros(1, CACHE_LEN, HEAD_DIM)
        with self.assertRaises(ShapeContractError) as ctx:
            assert_rotary_contract(good, bad, query_length=1)
        self.assertIn("sin", str(ctx.exception))


class TestTheAsymmetryThatHidTheBug(unittest.TestCase):
    def test_the_buggy_input_passes_at_PREFILL_and_fails_at_DECODE(self):
        """Why prefill-shaped tests could never have caught this.

        The same 'use the full sequence positions' behaviour is CORRECT during
        prefill and wrong during decode, because prefill's query length equals
        the cache length. Any test written only at prefill shape passes.
        """
        full_positions = torch.arange(PREFILL_LEN).view(1, 1, -1).repeat(AXES, 1, 1)

        # Prefill: query_len == cache_len == 10. Full positions are correct.
        assert_position_contract(
            full_positions, query_length=PREFILL_LEN, cache_length=PREFILL_LEN
        )

        # Decode: the identical construction is now wrong.
        with self.assertRaises(ShapeContractError):
            assert_position_contract(
                full_positions, query_length=1, cache_length=PREFILL_LEN
            )


if __name__ == "__main__":
    unittest.main()
