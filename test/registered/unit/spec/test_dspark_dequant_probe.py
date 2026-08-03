"""Hermetic tests for the DSpark stacked-projection support probes.

The probes decide whether the fused KV-projection stacking can read a
draft linear's weight; their contract is to *answer*, never to raise --
an unsupported scheme must route to the per-linear torch fallback in
``CommitKvProj.execute``. Packed quant methods (int4 marlin / AWQ /
GPTQ) expose ``qweight`` and have no ``weight`` attribute, so the
probe's own attribute access used to raise ``AttributeError`` on the
first speculative CUDA forward -- inside the very branch whose job is to
decline. The narrower twin: an fp8 linear whose ``weight_scale_inv`` is
absent must answer False rather than raise.

Pure CPU, no weights, no GPU.
"""

import unittest

import torch

from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
    _dequant_supported,
    _fused_commit_kv_proj_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeLinear(torch.nn.Module):
    """Bare module: an attribute exists only if it was handed in -- like a
    quantized ReplicatedLinear, where reading ``weight`` raises."""

    def __init__(self, **attrs):
        super().__init__()
        self.quant_method = object()  # no block_quant attribute
        for key, value in attrs.items():
            setattr(self, key, value)


class TestDequantProbe(CustomTestCase):
    def test_packed_qweight_only_answers_false(self):
        linear = _FakeLinear(qweight=torch.zeros(8, 8, dtype=torch.int32))
        self.assertFalse(_dequant_supported(linear))

    def test_plain_bf16_weight_answers_true(self):
        linear = _FakeLinear(weight=torch.zeros(8, 8, dtype=torch.bfloat16))
        self.assertTrue(_dequant_supported(linear))

    def test_fp8_with_matching_scale_answers_true(self):
        linear = _FakeLinear(
            weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(2, 2),
        )
        self.assertTrue(_dequant_supported(linear))

    def test_fp8_missing_scale_answers_false(self):
        linear = _FakeLinear(weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn))
        self.assertFalse(_dequant_supported(linear))

    def test_fp8_wrong_scale_shape_answers_false(self):
        linear = _FakeLinear(
            weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(3, 3),
        )
        self.assertFalse(_dequant_supported(linear))

    def test_fused_probe_routes_packed_to_fallback_without_raising(self):
        wkv_linears = [
            _FakeLinear(qweight=torch.zeros(8, 8, dtype=torch.int32)) for _ in range(3)
        ]
        self.assertFalse(_fused_commit_kv_proj_supported(wkv_linears=wkv_linears))


if __name__ == "__main__":
    unittest.main()
