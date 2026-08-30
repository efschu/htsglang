"""CPU acceptance test for the ADOPTED Qwen4-Exp Gated Residual layer.

Guards ``python/sglang/srt/layers/hyperconnection.py`` (adopted verbatim from
upstream ``qwen4-main-squashed`` 99c9362e66, plus the fork-local
``_hc_param_device`` fallback) against the geometry MEASURED from the real
checkpoint, cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4.

Measured shapes, read out of the safetensors header of shard
``model-00001-of-00038.safetensors`` (all BF16, per layer, two instances per
layer -- ``attn_hyper_connection`` and ``mlp_hyper_connection``):

    input_mix_weight_down.weight    (320, 10240)
    input_mix_weight_up.weight      (10240, 320)
    block_inject_weight.weight      (4, 10240)
    hc_norm.weight                  (10240,)

with hidden_size 2560, hc_count 4, hc_lowrank 320, so 4 * 2560 = 10240. The
model-level ``hyper_connection_mixer`` (and the one under ``mtp.``) carries
``input_mix_weight_down`` / ``input_mix_weight_up`` / ``hc_norm`` but NO
``block_inject_weight`` -- established by filtering ``weight_map`` in
``model.safetensors.index.json``: 398 hyper-connection tensors = 48 layers x 2
instances x 4 + 3 + 3 + 1 mtp layer x 2 x 4.

Two defects this file exists to catch, one loud and one silent:

* LOUD -- ``hc_per_branch_norm`` must be True. Upstream's own wiring sets it
  (``models/qwen4_exp.py:1263``), but the flag DEFAULTS TO FALSE in
  ``HyperConnectionConfig``. At False the norm is sized ``hidden_size`` and
  ``hc_norm.weight`` becomes (2560,) against the checkpoint's measured
  (10240,); ``GroupedGemmaRMSNorm._weight_loader`` asserts
  ``param.size() == loaded_weight.size()``, so this dies at load.
* SILENT -- ``hc_norm`` is a ``GroupedGemmaRMSNorm``: weight ZERO-init, and
  ``out = x_norm * (1.0 + w)``. This tree's ``layers.layernorm.RMSNorm`` is
  ONES-init with ``out = x_norm * w``. Swapping one for the other never raises
  and never changes a shape; it silently rescales every block input. The norm
  semantics are therefore asserted directly, with a weight chosen so that the
  two conventions cannot agree.
"""

import unittest

import torch

from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)

HIDDEN = 2560
HC_COUNT = 4
HC_LOWRANK = 320
FLAT = HIDDEN * HC_COUNT  # 10240
EPS = 1e-6
TOKENS = 7

MEASURED_SHAPES = {
    "input_mix_weight_down.weight": (HC_LOWRANK, FLAT),
    "input_mix_weight_up.weight": (FLAT, HC_LOWRANK),
    "block_inject_weight.weight": (HC_COUNT, FLAT),
    "hc_norm.weight": (FLAT,),
}


def _config(**overrides):
    """The config upstream's own model file builds (qwen4_exp.py:1257-1263)."""
    kwargs = dict(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN,
        params_dtype=torch.float32,
        hc_lowrank=HC_LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    kwargs.update(overrides)
    return HyperConnectionConfig(**kwargs)


class TestQwen4ExpHyperConnectionShapes(unittest.TestCase):
    """Parameter names and shapes against the measured checkpoint."""

    def test_per_layer_instance_matches_measured_shapes(self):
        gr = GatedResidual(_config(), use_mix=True, use_combine=True)
        got = {name: tuple(p.shape) for name, p in gr.named_parameters()}
        self.assertEqual(set(got), set(MEASURED_SHAPES))
        for name, shape in MEASURED_SHAPES.items():
            self.assertEqual(got[name], shape, f"{name} shape drifted")

    def test_model_level_mixer_has_no_block_inject_weight(self):
        # The model-level hyper_connection_mixer wraps no block, so upstream
        # builds it with use_combine=False. The checkpoint agrees: it has no
        # block_inject_weight tensor.
        mixer = GatedResidual(_config(), use_mix=True, use_combine=False)
        names = {name for name, _ in mixer.named_parameters()}
        self.assertNotIn("block_inject_weight.weight", names)
        expected = {k for k in MEASURED_SHAPES if not k.startswith("block_inject")}
        self.assertEqual(names, expected)

    def test_hc_per_branch_norm_false_would_not_load_the_checkpoint(self):
        # Regression guard for the LOUD defect: the flag defaults to False and
        # then sizes hc_norm at hidden_size, not hc_count * hidden_size.
        wrong = GatedResidual(_config(hc_per_branch_norm=False), use_mix=True,
                              use_combine=True)
        wrong_shape = dict(wrong.named_parameters())["hc_norm.weight"].shape
        self.assertEqual(tuple(wrong_shape), (HIDDEN,))
        self.assertNotEqual(
            tuple(wrong_shape),
            MEASURED_SHAPES["hc_norm.weight"],
            "hc_per_branch_norm=False must NOT match the measured (10240,) "
            "tensor; if it does, this guard has stopped guarding anything",
        )
        # And prove the mismatch is what upstream's loader rejects.
        loaded = torch.zeros(FLAT)
        with self.assertRaises(AssertionError):
            wrong.hc_norm._weight_loader(wrong.hc_norm.weight, loaded)

    def test_lowrank_default_is_not_our_rank(self):
        # hc_lowrank defaults to 16, and a wrong rank still satisfies every
        # fused-path gate, so it produces a wrong-rank mixer with no error.
        # The config must therefore always be passed explicitly.
        self.assertEqual(HyperConnectionConfig().hc_lowrank, 16)
        self.assertNotEqual(HyperConnectionConfig().hc_lowrank, HC_LOWRANK)


class TestGroupedGemmaRMSNormSemantics(unittest.TestCase):
    """The silent defect: Gemma (1.0 + w) semantics and per-branch grouping."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(TOKENS, FLAT, dtype=torch.float32)
        self.norm = GroupedGemmaRMSNorm(FLAT, eps=EPS, group_size=HIDDEN)

    def _reference_x_norm(self, x):
        xg = x.float().reshape(TOKENS, HC_COUNT, HIDDEN)
        return (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + EPS)).flatten(-2)

    def test_weight_is_zero_initialised(self):
        # Gemma-style norms are trained around 0; a ones-init tensor is the
        # tell-tale of this tree's plain RMSNorm having been substituted.
        self.assertTrue(torch.all(self.norm.weight == 0))

    def test_gemma_one_plus_w_semantics(self):
        # w = 0.5 separates the two conventions: Gemma gives 1.5 * x_norm,
        # plain RMSNorm (x_norm * w) would give 0.5 * x_norm.
        with torch.no_grad():
            self.norm.weight.fill_(0.5)
        out = self.norm(self.x)
        x_norm = self._reference_x_norm(self.x)
        torch.testing.assert_close(out, x_norm * 1.5, rtol=1e-5, atol=1e-5)
        # Explicitly refute the plain-RMSNorm result.
        self.assertFalse(
            torch.allclose(out, x_norm * 0.5, rtol=1e-3, atol=1e-3),
            "output matches x_norm * w -- the tree's RMSNorm semantics have "
            "been substituted for Gemma's x_norm * (1 + w)",
        )

    def test_zero_weight_is_the_identity_on_x_norm(self):
        out = self.norm(self.x)
        torch.testing.assert_close(
            out, self._reference_x_norm(self.x), rtol=1e-5, atol=1e-5
        )

    def test_normalisation_is_per_branch_not_over_the_whole_row(self):
        # Rescaling ONE branch must leave the other branches' normed output
        # untouched. A single 10240-wide RMS would couple all four.
        scaled = self.x.clone().reshape(TOKENS, HC_COUNT, HIDDEN)
        scaled[:, 0, :] *= 1000.0
        scaled = scaled.flatten(-2)

        base = self.norm(self.x).reshape(TOKENS, HC_COUNT, HIDDEN)
        after = self.norm(scaled).reshape(TOKENS, HC_COUNT, HIDDEN)

        torch.testing.assert_close(
            base[:, 1:, :], after[:, 1:, :], rtol=1e-5, atol=1e-5
        )
        # Branch 0 is scale-invariant under RMS norm, so it should come back to
        # the same place too -- which is what makes the check above meaningful
        # rather than an artefact of the scale.
        torch.testing.assert_close(
            base[:, 0, :], after[:, 0, :], rtol=1e-3, atol=1e-3
        )


class TestGatedResidualAlgebra(unittest.TestCase):
    """mix/combine shape algebra and finiteness on deterministic weights."""

    def setUp(self):
        torch.manual_seed(0)
        self.gr = GatedResidual(_config(), use_mix=True, use_combine=True)
        # Deterministic, small weights. nn.Linear already initialises, but the
        # finiteness assertions below are only meaningful against known-finite
        # parameters -- asserting on default-initialised memory is how a test
        # ends up reporting NaNs that came from the allocator, not the algebra.
        with torch.no_grad():
            for name, p in self.gr.named_parameters():
                if name == "hc_norm.weight":
                    p.zero_()
                else:
                    p.normal_(0.0, 0.02)
        self.streams = torch.randn(TOKENS, FLAT, dtype=torch.float32)

    def test_mix_collapses_streams_to_one_hidden_vector(self):
        mixed, residuals = self.gr.mix(self.streams)
        self.assertEqual(tuple(mixed.shape), (TOKENS, HIDDEN))
        self.assertTrue(torch.isfinite(mixed).all())
        # mix hands combine BOTH the raw and the normed streams; combine needs
        # the normed one for its gate and the raw one as the residual.
        raw, normed = residuals
        self.assertEqual(tuple(raw.shape), (TOKENS, FLAT))
        self.assertEqual(tuple(normed.shape), (TOKENS, FLAT))
        self.assertTrue(torch.equal(raw, self.streams))

    def test_combine_returns_flat_streams_and_is_a_gated_add(self):
        mixed, residuals = self.gr.mix(self.streams)
        out = self.gr.combine(mixed, residuals)
        self.assertEqual(tuple(out.shape), (TOKENS, FLAT))
        self.assertTrue(torch.isfinite(out).all())

        # Width-connections are absent from this checkpoint (no n x n tensor
        # exists), so combine must be residual + gate * block_output with a
        # per-branch scalar gate -- never a lateral mix between branches.
        # Recover the gate per branch and check it reproduces the output.
        delta = (out - residuals[0]).reshape(TOKENS, HC_COUNT, HIDDEN)
        block = mixed.reshape(TOKENS, 1, HIDDEN)
        gate = delta[..., :1] / block[..., :1]
        torch.testing.assert_close(
            delta, gate * block, rtol=1e-4, atol=1e-4,
            msg="combine is not a per-branch scalar-gated add",
        )
        # 2 * sigmoid(.) is bounded in (0, 2) and centred on 1.0, so an
        # unmodulated write is an ordinary residual add.
        self.assertTrue(torch.all(gate > 0.0))
        self.assertTrue(torch.all(gate < 2.0))

    def test_empty_token_batch_is_handled(self):
        empty = torch.zeros(0, FLAT, dtype=torch.float32)
        mixed, residuals = self.gr.mix(empty)
        self.assertEqual(tuple(mixed.shape), (0, HIDDEN))
        out = self.gr.combine(mixed, residuals)
        self.assertEqual(tuple(out.shape), (0, FLAT))

    def test_mix_rejects_a_wrong_width(self):
        with self.assertRaises(AssertionError):
            self.gr.mix(torch.randn(TOKENS, HIDDEN, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
