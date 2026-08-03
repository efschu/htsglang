# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The weights actually reached the model -- checked against the file, not a twin.

**This exists because it happened.** Under transformers 5.12,
``from_pretrained`` printed ``Loading weights: 478/478`` for the Qwen3-TTS
checkpoint and loaded **none** of them: every talker tensor was still at
transformers' random initialisation (``std 0.02``, biases exactly zero) while
the file held trained values. ``talker.text_projection.linear_fc2.bias`` was
``[0, 0, 0, ...]`` in the model against ``[0.00227, -0.00110, 0.00616, ...]``
in the file.

**Nothing cheap could see it.** A randomly initialised talker in front of a
correctly loaded vocoder does not produce silence or noise -- it produces
fluent, speech-shaped babble, because the vocoder turns any code sequence into
speech. The only external symptom was that generation never emitted the
end-of-utterance code and ran to ``max_new_tokens``: 40.9 s of audio for a
nine-word sentence, against 3.85 s from the reference implementation on the
same checkpoint. That looks exactly like a sampling or prompt-conditioning
bug, and two rounds of investigation went there first.

It also produced a **spuriously excellent number**: 0.986 cosine similarity
between the reference clip's x-vector and the output's -- both extracted by
the same randomly initialised speaker encoder. Two garbage vectors from one
garbage encoder agree perfectly. That is the "reference twin that agrees with
the thing it validates" family, and it is precisely why the check below
compares against the checkpoint BYTES and never against another copy of the
model.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_weight_loading.py -v
"""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.srt.translator.qwen3_tts_compat import (
    CompatError,
    verify_and_load_weights,
)


class _TinyModel(torch.nn.Module):
    """Two tensors is enough: one weight, one bias, both easy to zero out."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width)


def _checkpoint(directory: Path, state, name: str = "model.safetensors") -> Path:
    path = directory / name
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return path


class TestWeightVerification(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        torch.manual_seed(1466)
        self.trained = {
            "proj.weight": torch.randn(8, 8),
            "proj.bias": torch.randn(8),
        }
        _checkpoint(self.root, self.trained)

    def test_a_model_that_did_NOT_load_is_detected_and_repaired(self):
        """THE FALSIFIER. Random init in, trained weights out."""
        model = _TinyModel()
        torch.nn.init.normal_(model.proj.weight, std=0.02)
        torch.nn.init.zeros_(model.proj.bias)
        self.assertFalse(
            torch.allclose(model.proj.bias, self.trained["proj.bias"], atol=1e-3)
        )

        report = verify_and_load_weights(model, self.root)

        self.assertGreater(report["checked"], 0)
        self.assertEqual(report["repaired"], 2)
        self.assertEqual(report["mismatched"], 0)
        torch.testing.assert_close(model.proj.bias, self.trained["proj.bias"])
        torch.testing.assert_close(model.proj.weight, self.trained["proj.weight"])

    def test_an_already_loaded_model_is_left_alone(self):
        model = _TinyModel()
        model.load_state_dict(self.trained)
        report = verify_and_load_weights(model, self.root)
        self.assertGreater(report["checked"], 0)
        self.assertEqual(report["repaired"], 0)

    def test_a_zeroed_bias_alone_is_enough_to_trip_it(self):
        # The exact shape of the real failure: the big tensors look plausible
        # (they ARE the right distribution) and only the bias is obviously
        # wrong. The check must not need the whole model to be broken.
        model = _TinyModel()
        model.load_state_dict(self.trained)
        with torch.no_grad():
            model.proj.bias.zero_()
        report = verify_and_load_weights(model, self.root)
        self.assertEqual(report["repaired"], 2)
        torch.testing.assert_close(model.proj.bias, self.trained["proj.bias"])

    def test_a_missing_checkpoint_is_a_LOUD_failure(self):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        with self.assertRaises(CompatError) as ctx:
            verify_and_load_weights(_TinyModel(), empty)
        self.assertIn("cannot be verified", str(ctx.exception))

    def test_a_checkpoint_that_does_not_fit_is_a_LOUD_failure(self):
        # Never a silent partial load: a talker that cannot be loaded must
        # fail rather than sound plausible.
        other = Path(tempfile.mkdtemp())
        _checkpoint(
            other,
            {"proj.weight": torch.randn(4, 4), "proj.bias": torch.randn(4)},
        )
        with self.assertRaises(CompatError) as ctx:
            verify_and_load_weights(_TinyModel(), other)
        self.assertIn("does not fit", str(ctx.exception))

    def test_dtype_is_taken_from_the_MODEL_not_the_file(self):
        # The real checkpoint is bf16 and the desk runs fp32; loading the
        # file's dtype would silently downcast the whole talker.
        half = Path(tempfile.mkdtemp())
        _checkpoint(half, {k: v.to(torch.bfloat16) for k, v in self.trained.items()})
        model = _TinyModel()
        torch.nn.init.zeros_(model.proj.bias)
        verify_and_load_weights(model, half)
        self.assertEqual(model.proj.bias.dtype, torch.float32)
        torch.testing.assert_close(
            model.proj.bias, self.trained["proj.bias"], atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    unittest.main()


class TestTalkerEmbedderShape(unittest.TestCase):
    """The talker-backed embedder satisfies the diarization contract.

    Hermetic on purpose: instantiating it needs the checkpoint, but the thing
    most likely to silently break is the CONTRACT -- a backend that does not
    satisfy the protocol is only discovered when a real turn arrives, which
    on this project means during a conversation in Spain.
    """

    def test_it_satisfies_the_SpeakerEmbedder_contract(self):
        import inspect

        from sglang.srt.translator.inprocess_tts import TalkerSpeakerEmbedder

        # `issubclass` is unavailable here: the protocol carries non-method
        # members (`name`, `min_seconds`), which is deliberate -- they are
        # part of the contract the session reads. So the check is structural.
        self.assertTrue(hasattr(TalkerSpeakerEmbedder, "min_seconds"))
        self.assertTrue(inspect.iscoroutinefunction(TalkerSpeakerEmbedder.embed))
        signature = inspect.signature(TalkerSpeakerEmbedder.embed)
        self.assertEqual(list(signature.parameters), ["self", "audio"])

    def test_it_declares_a_minimum_segment_length(self):
        from sglang.srt.translator.inprocess_tts import TalkerSpeakerEmbedder

        # A vector from 300 ms of speech is noise with a norm; the registry
        # relies on this bound to refuse embedding such a segment at all.
        self.assertGreater(TalkerSpeakerEmbedder.min_seconds, 0.0)
