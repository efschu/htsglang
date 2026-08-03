# SPDX-License-Identifier: Apache-2.0
"""#479: what the ACTIVE DeepSeek-V4-Flash driver does with its two MXFP4 tensors.

The UD-IQ3_XXS export this fork serves carries exactly two ggml-type-39 tensors,
``blk.26.ffn_down_exps.weight`` and ``blk.42.ffn_down_exps.weight`` -- the MoE
DOWN projections of those layers, which every routed token reaches. Their
sibling ``ffn_gate_exps`` / ``ffn_up_exps`` tensors are NOT type 39 (IQ3_S on
layer 26, IQ3_XXS on layer 42), so those two layers are the only place in the
checkpoint where ``fused_moe_gguf`` is handed a MIXED type pair -- one type for
``w13``, a different one for ``w2``. That pair is what this file pins; the
single-type case was already covered by ``test_gguf_mxfp4_native.py``.

Three properties, all hermetic (``CUDA_VISIBLE_DEVICES=99``, no kernel runs):

1. **The pair is real.** ``TestActiveDriverTypePairs`` reads the shipped GGUF
   headers instead of trusting the constants below -- a state probe, not a
   restated claim. It skips when the export is not on this machine.
2. **Which branch the pair takes.** On a #398 wheel the pair (IQ, MXFP4) lands
   on the MoE MMVQ kernel with ggml type 39 passed through unchanged; with the
   kernels absent the loader has already turned the down tensor into Q5_0 and
   the same branch is taken with type 6. Neither arm reaches the MMQ branch
   (the IQ side has no MMQ kernel) and neither reaches the slow fallback loop.
3. **There is no silent third option.** The only way an unrepacked type 39 can
   reach a non-native build is with both levers off, and that combination
   RAISES out of the per-expert fallback rather than computing something. The
   load path never materialises these tensors as floats: both arms hand the
   consumer uint8 block bytes, 17 or 22 per 32 values.

Trace this pins, source of every step:
  ``layers/quantization/gguf.py:269``   ``SGLANG_GGUF_MXFP4_NATIVE`` kill switch
  ``layers/quantization/gguf.py:276``   wheel marker ``ggml_mxfp4_native``
  ``layers/quantization/gguf.py:281``   ``MXFP4_NATIVE``, evaluated once at import
  ``layers/quantization/gguf.py:282``   type 39 joins DEQUANT/MMVQ/MMQ
  ``model_loader/weight_utils.py:1419`` load pass 1: the type marker
  ``model_loader/weight_utils.py:1503`` load pass 2: the per-expert payload
  ``model_loader/gguf_mxfp4_repack.py:113`` native -> empty map -> identity
  ``layers/quantization/gguf.py:1519``  ``GGUFMoEMethod.apply`` -> fused_moe_gguf
  ``layers/quantization/gguf.py:1093``  the MMVQ branch both arms take
  ``layers/quantization/gguf.py:963``   the loud refusal, the only other exit
"""

from __future__ import annotations

import glob
import importlib
import os
import unittest

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as GGMLType

from test_gguf_mxfp4_native import _FakeNativeOp, _reload_gguf, synthetic_blocks

#: The active driver, as read off disk by ``TestActiveDriverTypePairs``.
DRIVER_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS"
)

#: ``layer -> (w13 ggml type, w2 ggml type)`` for the two MXFP4 layers.
#: IQ3_S = 21, IQ3_XXS = 18, MXFP4 = 39.
DRIVER_MXFP4_LAYERS = {26: (21, 39), 42: (18, 39)}

#: What the loader turns the w2 side into when the kernels are absent.
Q5_0 = int(GGMLType.Q5_0)


def _driver_present() -> bool:
    return bool(glob.glob(os.path.join(DRIVER_DIR, "*.gguf")))


class _KernelSpy:
    """Replaces the GGUF MoE/linear kernels with recorders.

    ``fused_moe_gguf`` reaches exactly three device entry points -- the MMQ MoE
    kernel, the MMVQ MoE kernel and (in the fallback loop) the dense linear
    dispatcher. Recording all three is what makes "which branch was taken" an
    observation rather than an inference from the source.
    """

    #: Sentinel for "this module global did not exist before the spy". On a
    #: host without CUDA the kernel names are never bound at all
    #: (``gguf.py:77-79``), so the spy must be able to remove them again
    #: instead of restoring a ``None`` the module never had.
    _ABSENT = object()

    def __init__(self, module):
        self.module = module
        self.calls: list[tuple[str, int]] = []
        self._saved: dict[str, object] = {}

    def __enter__(self):
        g = self.module

        def moe_vec(x, w, topk_ids, top_k, qtype, n, num_tokens):
            self.calls.append(("moe_a8_vec", int(qtype)))
            return torch.zeros(num_tokens * top_k, n, dtype=x.dtype)

        def moe_mmq(*args, **kwargs):
            self.calls.append(("moe_a8", -1))
            raise AssertionError("the MMQ MoE branch must not be reached here")

        def act(x):
            return x[..., : x.shape[-1] // 2]

        def sum_(out, dst):
            self.calls.append(("moe_sum", -1))
            dst.zero_()

        def mul_mat_vec(qweight, x, qtype, rows):
            self.calls.append(("mul_mat_vec_a8", int(qtype)))
            return torch.zeros(x.shape[0], rows, dtype=x.dtype)

        def mul_mat(qweight, x, qtype, rows):
            self.calls.append(("mul_mat_a8", int(qtype)))
            return torch.zeros(x.shape[0], rows, dtype=x.dtype)

        for name, fn in (
            ("ggml_moe_a8_vec", moe_vec),
            ("ggml_moe_a8", moe_mmq),
            ("ggml_mul_mat_vec_a8", mul_mat_vec),
            ("ggml_mul_mat_a8", mul_mat),
            ("silu_and_mul", act),
            ("moe_sum", sum_),
        ):
            self._saved[name] = getattr(g, name, self._ABSENT)
            setattr(g, name, fn)
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            if fn is self._ABSENT:
                delattr(self.module, name)
            else:
                setattr(self.module, name, fn)
        return False


def _drive_moe(module, qtype13: int, qtype2: int, num_tokens: int = 1):
    """Run ``fused_moe_gguf`` on a mixed pair and report the kernels it hit."""
    hidden, inter, experts, top_k = 64, 32, 4, 2
    x = torch.zeros(num_tokens, hidden, dtype=torch.float16)
    # Shapes only have to be self-consistent: every kernel is a recorder.
    w1 = torch.zeros(experts, 2 * inter, hidden, dtype=torch.uint8)
    w2 = torch.zeros(experts, hidden, inter, dtype=torch.uint8)
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float16)
    with _KernelSpy(module) as spy:
        module.fused_moe_gguf(
            x=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=qtype13,
            qweight_type2=qtype2,
            activation="silu",
        )
    return spy.calls


class TestActiveDriverTypePairs(unittest.TestCase):
    """The premise, read off the shipped file rather than asserted."""

    @unittest.skipUnless(_driver_present(), f"driver export not present: {DRIVER_DIR}")
    def test_the_export_carries_exactly_the_two_mixed_mxfp4_layers(self):
        import gguf

        types: dict[str, int] = {}
        for path in sorted(glob.glob(os.path.join(DRIVER_DIR, "*.gguf"))):
            for tensor in gguf.GGUFReader(path, "r").tensors:
                types[tensor.name] = int(tensor.tensor_type)

        mxfp4 = sorted(n for n, t in types.items() if t == int(GGMLType.MXFP4))
        self.assertEqual(
            mxfp4,
            ["blk.26.ffn_down_exps.weight", "blk.42.ffn_down_exps.weight"],
        )
        for layer, (w13_type, w2_type) in DRIVER_MXFP4_LAYERS.items():
            self.assertEqual(types[f"blk.{layer}.ffn_down_exps.weight"], w2_type)
            # gate and up are stacked into one w13 parameter, so they must
            # agree with each other as well as with the pinned constant.
            for leaf in ("ffn_gate_exps", "ffn_up_exps"):
                self.assertEqual(types[f"blk.{layer}.{leaf}.weight"], w13_type)


class TestMixedPairDispatch(unittest.TestCase):
    """Which kernel the (IQ, MXFP4) pair actually reaches, in both arms."""

    def setUp(self):
        self._env = os.environ.get("SGLANG_GGUF_MXFP4_NATIVE")

    def tearDown(self):
        if self._env is None:
            os.environ.pop("SGLANG_GGUF_MXFP4_NATIVE", None)
        else:
            os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = self._env
        _reload_gguf()

    def test_native_arm_passes_type_39_straight_into_the_moe_mmvq_kernel(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "1"
        with _FakeNativeOp():
            g = _reload_gguf()
            self.assertTrue(g.MXFP4_NATIVE)
            for layer, (w13_type, w2_type) in DRIVER_MXFP4_LAYERS.items():
                calls = _drive_moe(g, w13_type, w2_type)
                kernels = [name for name, _ in calls]
                self.assertEqual(
                    kernels,
                    ["moe_a8_vec", "moe_a8_vec", "moe_sum"],
                    f"layer {layer}",
                )
                self.assertEqual(
                    [t for name, t in calls if name == "moe_a8_vec"],
                    [w13_type, w2_type],
                    f"layer {layer}: the down side must reach the kernel as 39",
                )

    def test_repack_arm_reaches_the_same_branch_with_q5_0(self):
        """Kernels absent: the loader already replaced 39 with Q5_0."""
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "0"
        g = _reload_gguf()
        self.assertFalse(g.MXFP4_NATIVE)
        for w13_type, _ in DRIVER_MXFP4_LAYERS.values():
            calls = _drive_moe(g, w13_type, Q5_0)
            self.assertEqual(
                [name for name, _ in calls], ["moe_a8_vec", "moe_a8_vec", "moe_sum"]
            )
            self.assertEqual(
                [t for name, t in calls if name == "moe_a8_vec"], [w13_type, Q5_0]
            )

    def test_the_mmq_branch_is_unreachable_for_this_pair_at_any_batch_size(self):
        """The MMQ MoE branch needs BOTH sides in MMQ_QUANT_TYPES.

        Layer 26/42's w13 is an imatrix type, which has no MMQ kernel, so the
        ``x.shape[0] > 64`` condition can never carry the pair into the MMQ
        branch -- not even at prefill batch sizes. The spy raises if it does.
        """
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "1"
        with _FakeNativeOp():
            g = _reload_gguf()
            for w13_type, w2_type in DRIVER_MXFP4_LAYERS.values():
                self.assertNotIn(w13_type, {int(t) for t in g.MMQ_QUANT_TYPES})
                calls = _drive_moe(g, w13_type, w2_type, num_tokens=128)
                self.assertEqual(
                    [name for name, _ in calls],
                    ["moe_a8_vec", "moe_a8_vec", "moe_sum"],
                )

    def test_an_unrepacked_type_39_on_a_non_native_build_raises(self):
        """The only remaining combination, and it is loud.

        ``SGLANG_GGUF_MXFP4_NATIVE=0`` plus a payload that was never repacked
        falls out of both fast branches into the per-expert loop, and that loop
        refuses the type by name instead of producing numbers.
        """
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "0"
        g = _reload_gguf()
        for w13_type in (t for t, _ in DRIVER_MXFP4_LAYERS.values()):
            with self.assertRaises(NotImplementedError):
                _drive_moe(g, w13_type, int(GGMLType.MXFP4))


class TestLoadPathNeverMaterialisesFloats(unittest.TestCase):
    """#479's other half: what the LOAD does to a type-39 tensor.

    Neither arm dequantizes on the host. The native arm hands the payload
    through untouched (17 bytes per 32 values); the repack arm rewrites it into
    Q5_0 blocks (22 bytes per 32 values). Both stay uint8 -- there is no
    float16/float32 materialisation anywhere on the load path, so the resident
    cost of these two tensors is exactly their block bytes.
    """

    def setUp(self):
        self._env = os.environ.get("SGLANG_GGUF_MXFP4_NATIVE")

    def tearDown(self):
        if self._env is None:
            os.environ.pop("SGLANG_GGUF_MXFP4_NATIVE", None)
        else:
            os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = self._env
        _reload_gguf()
        from sglang.srt.model_loader import gguf_mxfp4_repack as r

        importlib.reload(r)

    def _repack_module(self):
        from sglang.srt.model_loader import gguf_mxfp4_repack as r

        return importlib.reload(r)

    def test_native_arm_is_byte_identity_and_stays_uint8(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "1"
        with _FakeNativeOp():
            _reload_gguf()
            r = self._repack_module()
            blocks = synthetic_blocks(64).reshape(1, -1)
            out = r.repacked_gguf_bytes(GGMLType.MXFP4, blocks, "blk.26.ffn_down_exps")
            np.testing.assert_array_equal(out, blocks)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(out.shape[-1] // 64, 17)
            self.assertEqual(
                r.repacked_gguf_type(GGMLType.MXFP4, "blk.26.ffn_down_exps"),
                GGMLType.MXFP4,
            )

    def test_repack_arm_grows_to_22_bytes_and_stays_uint8(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "0"
        _reload_gguf()
        r = self._repack_module()
        blocks = synthetic_blocks(64).reshape(1, -1)
        out = r.repacked_gguf_bytes(GGMLType.MXFP4, blocks, "blk.26.ffn_down_exps")
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape[-1] // 64, 22)
        self.assertEqual(
            r.repacked_gguf_type(GGMLType.MXFP4, "blk.26.ffn_down_exps"),
            GGMLType.Q5_0,
        )

    def test_the_two_driver_tensors_cost_exactly_their_block_bytes(self):
        """2 x [256, 4096, 2048] at 17 B / 32 values = 2.125 GiB, repack 2.750."""
        elements = 2 * 256 * 4096 * 2048
        native_bytes = elements // 32 * 17
        self.assertEqual(native_bytes / (1 << 30), 2.125)
        self.assertAlmostEqual(
            elements // 32 * 22 / (1 << 30) - native_bytes / (1 << 30),
            0.625,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
