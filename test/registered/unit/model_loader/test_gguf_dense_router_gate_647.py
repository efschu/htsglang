# SPDX-License-Identifier: Apache-2.0
"""A BF16 router gate must arrive as ``.weight``, not ``.qweight`` (#647).

``gguf_quant_weights_iterator`` decides quantized-vs-dense with a single
string comparison against one type name (``weight_utils.py:1448`` for the type
marker, ``:1517`` for the payload)::

    if weight_type.name != "F32":
        name = gguf_quantized_name(name, "qweight")

Every non-F32 tensor is therefore renamed, including tensors that are not
quantized at all. ``GGMLQuantizationType`` distinguishes F32, F16 and BF16 as
three separate unquantized types -- the layer's own notion of unquantized is
``UNQUANTIZED_TYPES = {F32, F16, BF16}`` (``layers/quantization/gguf.py:199``)
-- and this test is narrower than that set by exactly F16 and BF16.

For a module that HAS a GGUF quant method the rename is correct and
deliberate: the quant layer dequantizes dense F16/BF16 shards that arrive as
``.qweight`` (``gguf.py:1250-1256``), and the F32 carve-out in
``gguf_adapter_base.py:239-291`` keeps them there on purpose (task #64).

The defect is the modules that have NO quant method. An MoE router gate is
built ``ReplicatedLinear(..., quant_config=None)``
(``models/qwen3_moe.py:313-317``), so its only parameter is
``...mlp.gate.weight``; no ``.qweight`` and no ``.qweight_type`` exist on it.
The renamed tensors are then dropped without a word by the loader's
``if name not in params_dict: continue`` (``qwen3_moe.py:1204-1205``,
``:1246-1247``), the gate keeps whatever it was initialised with, and the MoE
routes on garbage. Wrong expert selection is not a crash -- it is fluent wrong
output, the #212 shape.

The carve-out at ``gguf_adapter_base.py:239-291`` cannot catch this: it skips
any module whose name lacks ``"proj"`` (``:269-271``), and a router gate is
``mlp.gate``.

``HazardTest`` demonstrates the misrouting from a real synthetic GGUF before
any fix, per #642's rule that a guard's test must show the hazard
independently. ``FixTest`` then pins the corrected dispatch.
"""

import pathlib
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# A router gate as it appears in a real MoE GGUF: BF16, dense, one row per
# expert. Small enough to keep the whole file a few KiB.
_NUM_EXPERTS = 4
_HIDDEN = 8


def _write_gguf(directory: pathlib.Path) -> str:
    """A tiny GGUF holding a BF16 router gate and an F32 contrast tensor."""
    import gguf
    import numpy as np

    path = directory / "tiny-moe.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen3moe")
    # BF16 router gate. numpy has no bfloat16, so hand gguf the raw payload
    # under an explicit BF16 type marker -- which is exactly how a real
    # export stores it and how gguf-py hands it back. Distinct ascending
    # values so the test can check the VALUES survive, not just the shape:
    # bf16 keeps the high two bytes of each float32, and small integers are
    # exactly representable, so the round-trip is lossless here.
    gate = np.arange(_NUM_EXPERTS * _HIDDEN, dtype=np.float32).reshape(
        _NUM_EXPERTS, _HIDDEN
    )
    gate_bf16_bytes = (
        gate.view(np.uint8)
        .reshape(_NUM_EXPERTS, _HIDDEN, 4)[:, :, 2:]
        .copy()
        .reshape(_NUM_EXPERTS, _HIDDEN * 2)
    )
    # raw_shape is the BYTE shape for BF16 -- gguf-py returns uint8 with the
    # last dimension doubled, and declares the logical shape from it.
    writer.add_tensor(
        "blk.0.ffn_gate_inp.weight",
        gate_bf16_bytes,
        raw_dtype=gguf.GGMLQuantizationType.BF16,
        raw_shape=(_NUM_EXPERTS, _HIDDEN * 2),
    )
    # F32 contrast: must keep ".weight" both before and after the fix.
    writer.add_tensor("blk.0.attn_norm.weight", np.ones((_HIDDEN,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


_NAME_MAP = {
    "blk.0.ffn_gate_inp.weight": "model.layers.0.mlp.gate.weight",
    "blk.0.attn_norm.weight": "model.layers.0.input_layernorm.weight",
}


def _emitted(path):
    """{name: tensor} for everything the iterator yields."""
    from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

    return dict(gguf_quant_weights_iterator(path, dict(_NAME_MAP)))


class HazardTest(CustomTestCase):
    """What the router gate collides with, shown without relying on the fix."""

    def test_router_gate_has_no_qweight_parameter_to_land_in(self):
        """The model side offers only ``.weight`` for a dense gate.

        ``ReplicatedLinear(..., quant_config=None)`` registers ``weight`` and
        nothing else, so a ``.qweight`` emitted for it matches no parameter
        and is dropped by the loader's ``name not in params_dict`` guard.
        This is the half of the hazard that lives in the model, and it is
        true regardless of what the iterator does.
        """
        import inspect

        from sglang.srt.models import qwen3_moe

        src = inspect.getsource(qwen3_moe.Qwen3MoeSparseMoeBlock.__init__)
        self.assertIn(
            "quant_config=None",
            src,
            "precondition: the router gate is built without a quant method, "
            "so it has no .qweight parameter",
        )

    def test_unquantized_types_are_wider_than_the_dispatch_test(self):
        """The two notions of 'unquantized' disagree on exactly F16/BF16.

        The iterator asks ``!= "F32"``; the layer knows F32, F16 and BF16 are
        all unquantized. That gap is the defect in one line.
        """
        import gguf

        from sglang.srt.layers.quantization.gguf import UNQUANTIZED_TYPES

        layer_view = {t.name for t in UNQUANTIZED_TYPES}
        self.assertEqual(layer_view, {"F32", "F16", "BF16"})
        self.assertIn(
            gguf.GGMLQuantizationType.BF16.name,
            layer_view,
            "BF16 is unquantized to the layer, yet the iterator's != 'F32' "
            "test routes it onto the quantized path",
        )


class FixTest(CustomTestCase):
    """The dispatch after #647."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _write_gguf(pathlib.Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_bf16_router_gate_keeps_the_weight_name(self):
        out = _emitted(self.path)
        self.assertIn(
            "model.layers.0.mlp.gate.weight",
            out,
            "a dense BF16 router gate must arrive under the name the model "
            "actually has",
        )
        self.assertNotIn(
            "model.layers.0.mlp.gate.qweight",
            out,
            "the gate must NOT be routed onto the quantized path",
        )

    def test_bf16_router_gate_emits_no_qweight_type_marker(self):
        """The type marker is dropped too, or it lands nowhere in its turn."""
        out = _emitted(self.path)
        self.assertNotIn("model.layers.0.mlp.gate.qweight_type", out)

    def test_bf16_router_gate_arrives_as_real_bf16_values(self):
        """Not the raw uint8 payload with a doubled last dimension.

        gguf-py returns BF16 as uint8 of width 2N; a gate handed over in that
        form would fail to load into an [experts, hidden] float parameter even
        once the name is right.
        """
        import numpy as np
        import torch

        out = _emitted(self.path)
        gate = out["model.layers.0.mlp.gate.weight"]
        self.assertEqual(gate.dtype, torch.bfloat16)
        self.assertEqual(tuple(gate.shape), (_NUM_EXPERTS, _HIDDEN))
        # The values must be the ones that were written, not a reinterpreted
        # byte pattern that merely happens to have the right shape.
        expected = np.arange(_NUM_EXPERTS * _HIDDEN, dtype=np.float32).reshape(
            _NUM_EXPERTS, _HIDDEN
        )
        np.testing.assert_allclose(gate.float().numpy(), expected)

    def test_f32_tensor_is_unaffected(self):
        """Backward compatibility: F32 never reached the rename anyway."""
        out = _emitted(self.path)
        self.assertIn("model.layers.0.input_layernorm.weight", out)


if __name__ == "__main__":
    unittest.main()
