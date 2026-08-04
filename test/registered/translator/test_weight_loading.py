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
    retarget_wrapper_device,
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


class _DetachedHolder:
    """``Qwen3TTSTokenizer``'s shape: a PLAIN object owning an ``nn.Module``.

    Not an ``nn.Module`` itself, so assigning it to a model does not register a
    submodule and ``model.to(device)`` never reaches the weights inside. It
    caches its own device at construction and places its inputs on that cache.
    That is the whole bug, in five lines.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.device = next(model.parameters()).device


class _HostModel(torch.nn.Module):
    """A model that owns a real submodule AND a detached holder."""

    def __init__(self) -> None:
        super().__init__()
        self.talker = torch.nn.Linear(8, 8)
        self.speech_tokenizer = _DetachedHolder(_TinyModel())


class _StaleWrapper:
    """The outer wrapper's device handling, reduced to its two lines.

    ``Qwen3TTSModel`` snapshots ``self.device`` in ``__init__`` and its
    tokenizer moves every prompt onto that snapshot. Both are reproduced
    verbatim in shape, because the bug lives entirely in the gap between them.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.device = getattr(model, "device", None)
        if self.device is None:
            self.device = next(model.parameters()).device

    def tokenize(self):
        return torch.zeros(3, dtype=torch.long).to(self.device)


class TestWrapperDeviceRetarget(unittest.TestCase):
    """``meta`` stands in for ``cuda`` so the falsifiers run without a card.

    What matters is only that the model is moved to a device OTHER than the one
    the wrapper snapshotted; ``meta`` is such a device and needs no hardware.
    """

    def test_a_moved_model_leaves_the_snapshot_stale(self):
        """FALSIFIER 1, negative half: the shipped device error, reproduced."""
        model = _TinyModel()
        wrapper = _StaleWrapper(model)
        model.to("meta")
        self.assertEqual(wrapper.device.type, "cpu")
        self.assertEqual(wrapper.tokenize().device.type, "cpu")
        self.assertEqual(next(model.parameters()).device.type, "meta")

    def test_retargeting_puts_the_prompt_where_the_weights_are(self):
        """FALSIFIER 1, positive half: same setup, one call, prompt follows."""
        model = _TinyModel()
        wrapper = _StaleWrapper(model)
        model.to("meta")
        report = retarget_wrapper_device(wrapper)
        self.assertEqual(report["wrapper"], "meta")
        self.assertEqual(wrapper.device.type, "meta")
        self.assertEqual(wrapper.tokenize().device.type, "meta")

    def test_a_detached_holder_is_NOT_moved_by_the_models_own_to(self):
        """FALSIFIER 2, negative half: the RTF-24 bug, reproduced.

        This is the one that cost 90% of every call: no exception, no warning,
        just the codec left behind on the CPU.
        """
        host = _HostModel()
        host.to("meta")
        self.assertEqual(next(host.talker.parameters()).device.type, "meta")
        self.assertNotIn("speech_tokenizer", host._modules)
        held = host.speech_tokenizer
        self.assertEqual(next(held.model.parameters()).device.type, "cpu")
        self.assertEqual(held.device.type, "cpu")

    def test_retargeting_takes_the_detached_holder_with_it(self):
        """FALSIFIER 2, positive half."""
        host = _HostModel()
        wrapper = _StaleWrapper(host)
        host.to("meta")
        report = retarget_wrapper_device(wrapper)
        self.assertEqual(report["speech_tokenizer"], "meta")
        held = host.speech_tokenizer
        self.assertEqual(next(held.model.parameters()).device.type, "meta")
        self.assertEqual(held.device.type, "meta")

    def test_an_explicit_device_is_held_against_the_model_not_assumed(self):
        """The caller names where it moved the model; the probe checks it.

        Passing a device the model is not actually on is the caller and the
        weights disagreeing, which is the failure this whole function exists to
        make impossible -- so it is refused rather than recorded.
        """
        host = _HostModel()
        wrapper = _StaleWrapper(host)
        with self.assertRaises(CompatError) as caught:
            retarget_wrapper_device(wrapper, "meta")
        self.assertIn("talker.weight@cpu", str(caught.exception))

    def test_a_stranded_tensor_after_the_move_is_a_LOUD_failure(self):
        """The independent state probe: a `.to()` that did nothing is caught.

        A holder whose `.to()` is a no-op is exactly what a silent split
        placement looks like from the outside, and it must not pass.
        """

        class _DeafHolder(_DetachedHolder):
            def __init__(self, model):
                super().__init__(model)
                self.model.to = lambda *a, **k: self.model

        host = _HostModel()
        host.speech_tokenizer = _DeafHolder(_TinyModel())
        wrapper = _StaleWrapper(host)
        with self.assertRaises(CompatError) as caught:
            retarget_wrapper_device(wrapper, "meta")
        self.assertIn("still not on meta", str(caught.exception))

    def test_it_is_idempotent_and_safe_on_an_unmoved_model(self):
        model = _TinyModel()
        wrapper = _StaleWrapper(model)
        for _ in range(3):
            self.assertEqual(retarget_wrapper_device(wrapper)["wrapper"], "cpu")
        self.assertEqual(wrapper.tokenize().device.type, "cpu")

    def test_a_wrapper_without_a_model_is_a_LOUD_failure(self):
        """A stale snapshot left in place would be a silent wrong-device bug."""

        class _Empty:
            model = None

        with self.assertRaises(CompatError):
            retarget_wrapper_device(_Empty())

    def test_a_model_without_parameters_is_a_LOUD_failure(self):
        class _NoParams(torch.nn.Module):
            pass

        class _Wrapper:
            def __init__(self):
                self.model = _NoParams()
                self.device = torch.device("cpu")

        wrapper = _Wrapper()
        # transformers' `.device` property is what a real model would expose;
        # a bare nn.Module has none, so this exercises the parameter fallback.
        self.assertIsNone(getattr(wrapper.model, "device", None))
        with self.assertRaises(CompatError):
            retarget_wrapper_device(wrapper)


if __name__ == "__main__":
    unittest.main()



if __name__ == "__main__":
    unittest.main()
