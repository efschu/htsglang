"""Falsification + inertness proof for the Qwen3.5/3.6 GGUF MoE router-gate
dense-name restore (task #651, defect #647).

Root cause: the shared GGUF weight iterator
(``weight_utils.gguf_quant_weights_iterator``) renames EVERY non-F32 tensor's
``.weight`` leaf to ``.qweight``, i.e. it equates "the tensor is not F32" with
"the destination module is quantized". Those are different statements. A MoE
router gate is never quantized: ``Qwen2MoeSparseMoeBlock`` builds both
``mlp.gate`` and ``mlp.shared_expert_gate`` with ``quant_config=None``, so each
owns a dense ``.weight`` and no ``.qweight`` at all. The renamed tensor
therefore lands on a parameter that does not exist and is dropped with a single
``logger.warning`` (``qwen3_5.py``), leaving the gate at whatever
``torch.empty`` returned. A garbage router still routes every token to SOME
expert, so the model stays fluent and is quietly wrong.

Two things kept this invisible until now:
  * published GGUFs almost always store router gates F32, and F32 is the one
    type the iterator never renames;
  * BF16 is handed over by gguf-py as RAW uint8 with the last dimension
    doubled, which also fails ``Tensor.is_floating_point()`` -- so the
    dense-shard rescue in ``gguf.py:_cast_dense_qweight`` skips it as well.

Qwen3.6-35B-A3B-UD-Q4_K_XL is the checkpoint that exposes it: its MTP block
(``blk.40``) stores ``ffn_gate_inp`` and ``ffn_gate_inp_shexp`` as BF16, and
those two are the ONLY non-F32 dense tensors among its 753 tensors. On that
file the NEXTN draft's router never loads.

Fix (``gguf_qwen35.py`` ``transform_stream``): restore the dense ``.weight``
name for the two gate spellings, re-view the BF16 bytes, drop the stray
``.qweight_type``, and fall through to the existing value transforms so the
1-D ``shared_expert_gate`` still gets its unsqueeze.

These tests drive the REAL ``Qwen35GGUFAdapter.transform_stream`` on CPU
tensors (no GPU, no GGUF file, no server), like the #113 expert-name test.
"""

import unittest

import torch

from sglang.srt.model_loader.gguf_qwen35 import Qwen35GGUFAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# ggml type ids, as gguf-py numbers them.
_F32 = 0
_F16 = 1
_BF16 = 30
_Q4_K = 12

HIDDEN = 2048
NUM_EXPERTS = 256


class _Cfg:
    torch_dtype = torch.bfloat16


def _make_adapter(is_draft: bool = True, num_layers: int = 40) -> Qwen35GGUFAdapter:
    a = Qwen35GGUFAdapter.__new__(Qwen35GGUFAdapter)
    a.is_draft = is_draft
    a.num_layers = num_layers
    a.config = _Cfg()
    return a


def _qtype(value: int) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int32)


def _as_gguf_bf16_bytes(t: torch.Tensor) -> torch.Tensor:
    """Exactly what gguf-py hands back for a BF16 tensor: raw uint8 with the
    last dimension doubled."""
    return t.to(torch.bfloat16).contiguous().view(torch.uint8)


def _run(adapter, stream):
    return list(adapter.transform_stream(stream))


class TestQwen35RouterGateDtype(CustomTestCase):
    # ---------------------------------------------------------------- BF16

    def test_bf16_router_gate_is_restored_dense(self):
        """The load-bearing case: a BF16 ffn_gate_inp must arrive as a dense
        `.weight` holding the ORIGINAL values, not as a dropped `.qweight`."""
        values = torch.randn(NUM_EXPERTS, HIDDEN).to(torch.bfloat16)
        out = _run(
            _make_adapter(),
            [
                ("mtp.layers.0.mlp.gate.qweight_type", _qtype(_BF16)),
                ("mtp.layers.0.mlp.gate.qweight", _as_gguf_bf16_bytes(values)),
            ],
        )
        names = [n for n, _ in out]
        # The stray type param is dropped (the dense module has no such param).
        self.assertEqual(names, ["mtp.layers.0.mlp.gate.weight"])
        tensor = out[0][1]
        # Not raw bytes any more, and the right shape for
        # ReplicatedLinear(hidden_size, num_experts).
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(tuple(tensor.shape), (NUM_EXPERTS, HIDDEN))
        # Values survive the re-view bit-exactly -- a re-view is not a cast.
        self.assertTrue(torch.equal(tensor, values))

    def test_bug_repro_bf16_payload_is_bytes_and_fails_the_dense_rescue(self):
        """Documents why BF16 specifically breaks where F16 survives: the
        payload is uint8, so `_cast_dense_qweight`'s is_floating_point() gate
        skips it and it would reach the matmul as raw bytes of twice the
        width."""
        values = torch.randn(NUM_EXPERTS, HIDDEN)
        raw = _as_gguf_bf16_bytes(values)
        self.assertEqual(raw.dtype, torch.uint8)
        self.assertFalse(raw.is_floating_point())
        self.assertEqual(tuple(raw.shape), (NUM_EXPERTS, HIDDEN * 2))

    def test_bf16_shared_expert_gate_is_restored_and_unsqueezed(self):
        """GGUF stores ffn_gate_inp_shexp as a 1-D [hidden] vector; the module
        is nn.Linear(hidden, 1) with weight [1, hidden]. Under the `.qweight`
        rename the existing unsqueeze branch was bypassed too, so the fix has
        to reach it -- this is why the gate branch reassigns and falls through
        instead of yielding."""
        values = torch.randn(HIDDEN).to(torch.bfloat16)
        out = _run(
            _make_adapter(),
            [
                ("mtp.layers.0.mlp.shared_expert_gate.qweight_type", _qtype(_BF16)),
                (
                    "mtp.layers.0.mlp.shared_expert_gate.qweight",
                    _as_gguf_bf16_bytes(values),
                ),
            ],
        )
        names = [n for n, _ in out]
        self.assertEqual(names, ["mtp.layers.0.mlp.shared_expert_gate.weight"])
        tensor = out[0][1]
        self.assertEqual(tuple(tensor.shape), (1, HIDDEN))
        self.assertTrue(torch.equal(tensor[0], values))

    # ----------------------------------------------------------- other dtypes

    def test_f16_router_gate_is_restored_dense(self):
        """F16 is renamed by the iterator just like BF16; it happens to survive
        downstream, but it must still arrive under the dense name."""
        values = torch.randn(NUM_EXPERTS, HIDDEN).to(torch.float16)
        out = _run(
            _make_adapter(),
            [
                ("mtp.layers.0.mlp.gate.qweight_type", _qtype(_F16)),
                ("mtp.layers.0.mlp.gate.qweight", values),
            ],
        )
        self.assertEqual([n for n, _ in out], ["mtp.layers.0.mlp.gate.weight"])
        self.assertEqual(out[0][1].dtype, torch.bfloat16)

    def test_f32_router_gate_is_untouched(self):
        """The common case: F32 is never renamed by the iterator, so it arrives
        as `.weight` already and must pass through byte-identically. This is
        the inertness proof for every published GGUF that stores gates F32 --
        i.e. all 40 base-layer gates of the A3B checkpoint."""
        values = torch.randn(NUM_EXPERTS, HIDDEN)
        out = _run(
            _make_adapter(is_draft=False),
            [("model.layers.7.mlp.gate.weight", values)],
        )
        self.assertEqual([n for n, _ in out], ["model.layers.7.mlp.gate.weight"])
        self.assertTrue(torch.equal(out[0][1], values))

    # ------------------------------------------------------------- no capture

    def test_quantized_gate_proj_is_not_captured(self):
        """`mlp.gate_proj` is a genuinely quantized expert projection. Matching
        on a bare "gate" substring would divert it to the dense path and
        destroy it, so the suffixes are anchored on `.mlp.gate.` /
        `.shared_expert_gate.`."""
        payload = torch.zeros(8, dtype=torch.uint8)
        stream = [
            ("model.layers.7.mlp.experts.3.gate_proj.qweight_type", _qtype(_Q4_K)),
            ("model.layers.7.mlp.experts.3.gate_proj.qweight", payload),
        ]
        out = _run(_make_adapter(is_draft=False), stream)
        self.assertEqual([n for n, _ in out], [n for n, _ in stream])
        self.assertEqual(out[1][1].dtype, torch.uint8)

    def test_target_model_stream_is_unaffected(self):
        """Inertness for the main model: an ordinary target-side stream carrying
        no non-F32 gate passes through unchanged."""
        stream = [
            ("model.layers.3.self_attn.q_proj.qweight_type", _qtype(_Q4_K)),
            ("model.layers.3.self_attn.q_proj.qweight", torch.zeros(4, dtype=torch.uint8)),
            ("model.layers.3.mlp.gate.weight", torch.randn(NUM_EXPERTS, HIDDEN)),
        ]
        out = _run(_make_adapter(is_draft=False), stream)
        self.assertEqual([n for n, _ in out], [n for n, _ in stream])


if __name__ == "__main__":
    unittest.main()
