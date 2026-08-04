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

import numpy as np
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


class _FakeCache:
    def __init__(self, seen: int) -> None:
        self._seen = seen

    def get_seq_length(self) -> int:
        return self._seen


class _FakeGenerativeModel:
    """The parts of a transformers model the restoration actually touches."""

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, custom_kwarg=None, **kwargs
    ):
        return {"input_ids": input_ids, "past_key_values": past_key_values}


class TestCachePositionRestoration(unittest.TestCase):
    """The decode-seam fix, without the checkpoint.

    transformers 5.x stopped creating ``cache_position`` for installed
    (non-remote-code) models. The talker branches on it to tell prefill from
    decode, so its absence pinned every step to the prefill branch and handed
    M-RoPE cache-length positions for a one-token query -- the bug the
    contracts above describe. Restoring the input is the fix; these are its
    falsifiers.
    """

    def setUp(self):
        from sglang.srt.translator.qwen3_tts_compat import restore_cache_position

        self.restore = restore_cache_position

        class Model(_FakeGenerativeModel):
            pass

        self.model_class = Model

    def test_prefill_positions_start_at_zero(self):
        self.assertTrue(self.restore(self.model_class))
        inputs = self.model_class().prepare_inputs_for_generation(
            torch.zeros(1, PREFILL_LEN, dtype=torch.long)
        )
        self.assertEqual(
            inputs["cache_position"].tolist(), list(range(PREFILL_LEN))
        )

    def test_a_decode_step_gets_ONE_position_at_the_cache_offset(self):
        # THE FIX. One query token, one position, offset by what is cached --
        # which is exactly what makes the talker take its per-step branch
        # instead of rebuilding the whole sequence's positions.
        self.restore(self.model_class)
        inputs = self.model_class().prepare_inputs_for_generation(
            torch.zeros(1, 1, dtype=torch.long),
            past_key_values=_FakeCache(PREFILL_LEN),
        )
        self.assertEqual(inputs["cache_position"].tolist(), [PREFILL_LEN])
        # And the invariant the contract states, on the real object.
        assert_position_contract(
            inputs["cache_position"].view(1, -1),
            query_length=1,
            cache_length=CACHE_LEN,
        )

    def test_an_existing_cache_position_is_not_overwritten(self):
        class Model(_FakeGenerativeModel):
            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {"input_ids": input_ids, "cache_position": torch.tensor([7])}

        self.restore(Model)
        inputs = Model().prepare_inputs_for_generation(
            torch.zeros(1, 1, dtype=torch.long)
        )
        self.assertEqual(inputs["cache_position"].tolist(), [7])

    def test_it_is_idempotent(self):
        self.assertTrue(self.restore(self.model_class))
        self.assertFalse(self.restore(self.model_class))

    def test_the_SIGNATURE_survives_the_wrapper(self):
        """The functools.wraps lesson, pinned.

        ``generate()`` validates its ``model_kwargs`` by inspecting the
        signatures of ``forward`` and ``prepare_inputs_for_generation``. A
        wrapper written as ``(*args, **kwargs)`` erases the parameter names,
        and every legitimate extra kwarg the talker needs is then reported as
        unused -- which is a hard error, not a warning. This cost an afternoon
        once already, in a different wrapper around the same model.
        """
        import inspect

        before = set(
            inspect.signature(
                self.model_class.prepare_inputs_for_generation
            ).parameters
        )
        self.restore(self.model_class)
        after = set(
            inspect.signature(
                self.model_class.prepare_inputs_for_generation
            ).parameters
        )
        self.assertEqual(before, after)
        self.assertIn("custom_kwarg", after)


class TestResampleShim(unittest.TestCase):
    """``librosa.resample``, the second stub the audio path turned out to need.

    Not a reimplementation -- it delegates to ``scipy.signal.resample_poly`` --
    so what is worth pinning is that it is wired to the right rates and does
    not alias, both of which would silently degrade the x-vector the whole
    cloning path conditions on.
    """

    def setUp(self):
        from sglang.srt.translator.qwen3_tts_compat import librosa_resample

        self.resample = librosa_resample

    def test_the_length_follows_the_rate_ratio(self):
        signal = np.zeros(24000, dtype=np.float32)
        out = self.resample(signal, 24000, 16000)
        self.assertEqual(len(out), 16000)

    def test_a_tone_keeps_its_frequency(self):
        rate, target, hz = 24000, 16000, 440.0
        t = np.arange(rate, dtype=np.float32) / rate
        tone = np.sin(2 * np.pi * hz * t).astype(np.float32)
        out = self.resample(tone, rate, target)
        spectrum = np.abs(np.fft.rfft(out))
        peak = np.fft.rfftfreq(len(out), 1.0 / target)[int(np.argmax(spectrum))]
        self.assertAlmostEqual(peak, hz, delta=2.0)

    def test_downsampling_does_not_ALIAS_a_high_tone_into_the_band(self):
        # 10 kHz is representable at 24 kHz but above the 8 kHz Nyquist of the
        # 16 kHz output. Naive decimation would fold it back INTO the speech
        # band and nothing would raise; the anti-alias filter must remove it.
        rate, target, hz = 24000, 16000, 10000.0
        t = np.arange(rate, dtype=np.float32) / rate
        tone = np.sin(2 * np.pi * hz * t).astype(np.float32)
        out = self.resample(tone, rate, target)
        self.assertLess(float(np.abs(out).max()), 0.2)

    def test_an_identical_rate_is_a_pass_through(self):
        signal = np.linspace(-1, 1, 100).astype(np.float32)
        np.testing.assert_allclose(self.resample(signal, 16000, 16000), signal)


if __name__ == "__main__":
    unittest.main()
