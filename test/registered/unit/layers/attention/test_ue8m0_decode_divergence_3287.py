# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""#3287: the flash_mla_sm120 ue8m0 decode pair, pinned at desk.

THE PAIR. The SM120 sparse-decode path has two independent ue8m0 scale
decoders that a GPU test compares against each other:

  * torch    -- ``flash_mla_sm120.py:115``,
                ``scale_bytes.view(torch.float8_e8m0fnu)``, consumed
                ``.float()`` at ``:119-126``
  * triton   -- ``flash_mla_sm120_triton.py:172``,
                ``tl.math.exp2(scale_raw.to(tl.float32) - 127.0)``

They agree on every encoding except ``0xFF``, where torch gives NaN
(spec-correct: OCP MX v1.0 §5.4.1 makes 255 the sole NaN encoding) and the
Triton expression gives ``exp2(128) == inf``. The triton side is the
non-conformant one.

WHAT DISSOLVED UNDER SCRUTINY, recorded because the bundled appearance split
into separate facts:

1. The comparison originally cited as poisoned -- ``assert_close`` in
   ``TestGatherAndDequant`` -- is torch-against-torch: ``_build_kvcache``
   computes its reference with the SAME
   ``.view(torch.float8_e8m0fnu).float()`` the code under test uses, so no
   Triton decoder participates and byte 255 could not poison it. The genuinely
   exposed comparison is ``TestSparseDecodeTritonVsTorch._run``.

2. The "two decode sites do not even share a lowering" asymmetry
   (``tl.exp2`` for dsv4 vs ``tl.math.exp2`` here) is not a thing on the
   installed Triton: they are the SAME function object, defined once in
   ``triton/language/math.py`` and re-exported. Pinned below, so that a future
   Triton which splits them re-raises the question instead of hiding it.

3. Byte 255 is UNREACHABLE through the fork's own k-cache quantizer. It derives
   the byte as ``ceil(log2(max(max_abs, EPS) / FP8_MAX)) + 127`` with
   ``EPS=1e-8`` and ``FP8_MAX=448`` (``dsv4/quant_k_cache.py:54-59,89-90``,
   deliberately unclamped, reasoned at ``:80-88``), which for any finite bf16
   input lands in ``[92, 247]``. So this is a conformance gap in a decoder, not
   a live wrong answer -- which is why it is pinned here rather than
   "fixed" by changing kernel numerics on an unreachable input.

WHY THIS IS DECIDABLE AT DESK, unlike the #418/1d ftz question. That one turns
on whether ``ex2.approx`` is emitted in its ``.ftz`` form, which only affects
SUBNORMAL results and which no CPU evaluation can settle. ``exp2(128)`` is
``inf``, not a subnormal, and NaN-vs-inf is exact under every lowering. The
whole file is therefore hermetic, whereas
``test/registered/kernels/test_flash_mla_backends.py`` is SM120-gated
(``skipUnless(_IS_SM120)`` on every class) and cannot run here at all.

RECIPE, for any GPU window (that file, not this one): its cache generator draws
``torch.randint(120, 131, ...)`` -- 11 of the 156 encodings the quantizer can
actually emit, with the narrowness incidental ("keep exponents in a sane
range") rather than defended. Widening it to the reachable ``[92, 248)``
multiplies scale coverage of the SM120 triton-vs-torch comparison by 14. The
decode side of that widening is already proven safe below
(``test_the_two_decoders_agree_over_every_reachable_scale_byte``); what a card
would add is the gather/kernel machinery around it. Not applied here: it is a
change to a test that cannot be executed at desk, and shipping an unexecuted
edit is what this bundle's discipline forbids.

Hermetic: no CUDA, no Triton launch, no model.
"""

import math
import unittest

import torch

# The fork's k-cache quantizer constants (dsv4/quant_k_cache.py:140-142).
_FP8_MAX = 448.0
_EPS = 1e-8

# Every ue8m0 encoding except the NaN one.
_FINITE_BYTES = list(range(0, 255))


def _torch_decode(byte_values) -> torch.Tensor:
    """The flash_mla_sm120.py decoder: reinterpret as e8m0, widen to fp32."""
    return (
        torch.tensor(byte_values, dtype=torch.uint8).view(torch.float8_e8m0fnu).float()
    )


def _exp2_decode(byte_values) -> torch.Tensor:
    """The flash_mla_sm120_triton.py decoder: exp2(byte - 127) in fp32."""
    return torch.exp2(torch.tensor(byte_values, dtype=torch.float32) - 127.0)


def _quantizer_scale_byte(max_abs: float) -> int:
    """The byte quant_k_cache.py would emit for a tile with this max |value|."""
    scale = max(max_abs, _EPS) / _FP8_MAX
    return int(math.ceil(math.log2(scale))) + 127


class TestUe8m0DecodePair(unittest.TestCase):
    def test_the_two_decoders_agree_over_every_reachable_scale_byte(self):
        """Bytes 0..254 decode identically under both expressions.

        This is the desk half of the coverage the SM120 test's ``[120, 131)``
        generator leaves out. Asserted bit-exactly, not with a tolerance: both
        sides produce exact powers of two, so any difference is a decode
        defect and not rounding.
        """
        torch_side = _torch_decode(_FINITE_BYTES)
        exp2_side = _exp2_decode(_FINITE_BYTES)
        mismatch = [
            (b, torch_side[i].item(), exp2_side[i].item())
            for i, b in enumerate(_FINITE_BYTES)
            if torch_side[i].item() != exp2_side[i].item()
        ]
        self.assertEqual(mismatch, [], f"ue8m0 decoders disagree at {mismatch}")

    def test_byte_255_is_the_one_divergence_and_the_triton_side_is_wrong(self):
        """The finding, pinned as a known non-conformance.

        Spec: 255 is the sole NaN encoding. torch honours it; ``exp2(128)``
        saturates to inf. Pinned rather than repaired because the fork's
        quantizer cannot emit this byte (see the reachability test below), so
        changing kernel numerics for it would be an unvalidatable edit on an
        unreachable input. If a future writer CAN emit 255, this test is where
        the consequence is already written down.
        """
        torch_side = _torch_decode([255])[0].item()
        exp2_side = _exp2_decode([255])[0].item()
        self.assertTrue(
            math.isnan(torch_side),
            f"torch e8m0 decode of 0xFF should be NaN, got {torch_side}",
        )
        self.assertEqual(
            exp2_side,
            math.inf,
            f"exp2(128) should saturate to inf, got {exp2_side}",
        )
        self.assertNotEqual(
            math.isnan(torch_side),
            math.isnan(exp2_side),
            "the divergence this test exists for has disappeared -- if the "
            "triton side now yields NaN, delete the pin and the recipe with it",
        )

    def test_the_fork_quantizer_cannot_emit_the_divergent_byte(self):
        """Reachability: for any finite bf16 input the byte lands in [92, 247].

        This is what downgrades the divergence from a live wrong answer to a
        conformance gap. It is computed from the quantizer's own formula rather
        than asserted in prose, so a change to EPS, FP8_MAX or the clamping
        decision re-opens the question here.
        """
        finfo = torch.finfo(torch.bfloat16)
        probes = [
            0.0,
            _EPS,
            float(finfo.tiny),
            float(finfo.smallest_normal),
            1.0,
            float(finfo.max),
        ]
        bytes_emitted = [_quantizer_scale_byte(p) for p in probes]
        self.assertEqual(min(bytes_emitted), 92, f"got {bytes_emitted}")
        self.assertEqual(max(bytes_emitted), 247, f"got {bytes_emitted}")
        self.assertNotIn(255, bytes_emitted)
        for b in bytes_emitted:
            self.assertTrue(1 <= b <= 254, f"byte {b} outside the finite range")

    def test_the_sm120_generator_covers_a_fraction_of_the_reachable_range(self):
        """Why the recipe exists, stated as a number rather than an opinion.

        Kept as an arithmetic pin on the two ranges so the claim in the recipe
        above cannot quietly rot if either end moves.
        """
        generator_span = len(range(120, 131))
        reachable_span = len(range(92, 248))
        self.assertEqual(generator_span, 11)
        self.assertEqual(reachable_span, 156)
        self.assertLess(generator_span * 10, reachable_span)


class TestExp2LoweringIsShared(unittest.TestCase):
    """#3287 (was reported as part of the #418/1d ftz question): the fork's two
    ue8m0 decode sites were said to use different lowerings. They do not."""

    def test_tl_exp2_and_tl_math_exp2_are_the_same_function(self):
        """``dsv4/dequant_k_cache.py:139`` uses ``tl.exp2``;
        ``flash_mla_sm120_triton.py:172`` uses ``tl.math.exp2``. If those are
        one object, the ftz question is ONE question and a single PTX dump
        settles both sites. If a future Triton splits them, this fails and the
        two sites must be re-examined separately."""
        import triton.language as tl

        self.assertIs(
            tl.exp2,
            tl.math.exp2,
            "tl.exp2 and tl.math.exp2 are no longer the same function: the two "
            "ue8m0 decode sites may now lower differently, and the #418 ftz "
            "recipe must be run for each of them",
        )


if __name__ == "__main__":
    unittest.main()
