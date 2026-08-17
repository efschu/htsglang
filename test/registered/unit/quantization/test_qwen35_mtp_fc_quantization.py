"""The MTP head's ``fc`` projection must be built through the quant config.

``Qwen3_5ForCausalLMMTP`` fuses the token embedding and the carried hidden
state with a single ``2*hidden -> hidden`` projection, stored in the
checkpoint as ``mtp.fc.weight``. Every Qwen3.5/3.6-era checkpoint on this box
left that projection dense -- the NVFP4 list names ``mtp.fc`` outright, the
INT8-W8A8 list carries ``re:.*mtp.*``, the AWQ list carries ``mtp`` -- so a
plain ``nn.Linear`` was indistinguishable from a correct one and the wrapper
never consulted ``quant_config`` for this one layer.

Qwen3.8-27B-INT8 is the first checkpoint that quantises it. Its ``ignore``
list is six regex entries, none of which match ``mtp.fc``, and the safetensors
index carries ``mtp.fc.weight_scale`` alongside ``mtp.fc.weight``.

Against a plain ``nn.Linear`` that combination is silent, which is what makes
it worth a test rather than a fix alone:

* ``mtp.fc.weight`` maps to ``fc.weight``, which IS in ``params_dict`` and has
  the same ``[out, in]`` shape an int-quantised tensor keeps, so
  ``default_weight_loader``'s size assert passes and the int8 payload is
  copied verbatim into a bf16 parameter -- no dequantisation, no error.
* ``mtp.fc.weight_scale`` maps to ``fc.weight_scale``, which is NOT in
  ``params_dict``, so it is dropped by the ``ignore_suffixes`` skip and never
  even reaches the unmatched-name warning.

The result is a drafter whose fusion projection is numeric garbage while the
load reports success -- the #318 / #290 signature again, with neither an
unloaded-parameter guard nor a warning to catch it.

The load-bearing detail is the NAME the layer is resolved under. It has to be
the checkpoint's own ``mtp.fc``, because that is the string every existing
exclusion list was written against; resolving the layer as bare ``fc`` would
miss ``re:.*mtp.*`` and build a quantised skeleton against the dense bf16
tensors of the checkpoint production currently runs.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.qwen3_5_mtp import build_mtp_fc
from sglang.test.test_utils import CustomTestCase

#: lokeshe09/Qwen3.8-27B-INT8, ``quantization_config.ignore`` verbatim.
Q38_IGNORE = [
    "re:.*(vision|visual).*",
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*norm.*",
    "re:.*conv1d.*",
    "re:.*linear_attn.*",
]

#: Qwen3.6-27B-INT8-W8A8, the two non-``linear_attn`` tail entries of the
#: 208-entry ``ignore`` list. The draft is excluded by one coarse regex.
Q36_IGNORE_TAIL = ["lm_head", "re:.*mtp.*"]

#: ocicek/Qwen3.6-27B-NVFP4 names the layer outright.
NVFP4_IGNORE_EXCERPT = ["lm_head", "mtp.fc", "mtp.layers.0.self_attn.q_proj"]

FC_LAYER_NAME = "mtp.fc"


class _SpyQuantConfig:
    """Records the layer names it is asked about, quantises nothing.

    Standing in for a real quant config keeps this test hermetic and CPU-only:
    a genuine compressed-tensors INT8 scheme resolves far enough to call
    ``torch.cuda.get_device_capability()``, which is exactly the kind of GPU
    dependency a name-resolution test should not carry.
    """

    def __init__(self):
        self.asked = []

    def get_quant_method(self, layer, prefix):
        self.asked.append(prefix)
        # What a real config returns for an EXCLUDED layer. Returning None
        # instead would trip the linear layer's own assert and prove nothing.
        return UnquantizedLinearMethod()


class TestMtpFcConsultsTheQuantConfig(CustomTestCase):
    def test_fc_is_resolved_through_the_quant_config(self):
        spy = _SpyQuantConfig()
        build_mtp_fc(hidden_size=64, quant_config=spy, prefix="")
        self.assertEqual(
            spy.asked,
            [FC_LAYER_NAME],
            "the MTP fusion projection must be resolved through the quant "
            "config; a bare nn.Linear silently mis-loads an int8 mtp.fc.weight "
            "and drops mtp.fc.weight_scale",
        )

    def test_fc_is_resolved_under_the_checkpoints_own_name(self):
        """Not bare ``fc``: every exclusion list was written as ``mtp.*``."""
        spy = _SpyQuantConfig()
        build_mtp_fc(hidden_size=64, quant_config=spy, prefix="")
        (asked,) = spy.asked
        self.assertTrue(
            should_ignore_layer(asked, ignore=Q36_IGNORE_TAIL, fused_mapping={}),
            f"{asked!r} must match Qwen3.6-INT8's 're:.*mtp.*' exclusion",
        )
        self.assertTrue(
            should_ignore_layer(asked, ignore=NVFP4_IGNORE_EXCERPT, fused_mapping={}),
            f"{asked!r} must match NVFP4's explicit 'mtp.fc' exclusion",
        )
        self.assertFalse(
            should_ignore_layer(asked, ignore=Q38_IGNORE, fused_mapping={}),
            f"{asked!r} must NOT be excluded by Qwen3.8-INT8, which quantises it",
        )

    def test_unquantised_build_still_exposes_a_plain_weight(self):
        """The dense checkpoints must keep loading exactly as before."""
        fc = build_mtp_fc(hidden_size=64, quant_config=None, prefix="")
        names = sorted(n for n, _ in fc.named_parameters())
        self.assertEqual(names, ["weight"])
        self.assertEqual(tuple(fc.weight.shape), (64, 128))

    def test_fc_returns_the_linear_output_tuple(self):
        """Callers must unpack; a bare tensor return would mask a regression."""
        import torch

        fc = build_mtp_fc(hidden_size=64, quant_config=None, prefix="")
        out = fc(torch.zeros(2, 128, dtype=fc.weight.dtype))
        self.assertIsInstance(out, tuple)
        self.assertEqual(tuple(out[0].shape), (2, 64))


if __name__ == "__main__":
    unittest.main()
