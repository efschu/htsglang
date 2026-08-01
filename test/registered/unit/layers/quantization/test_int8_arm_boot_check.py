"""The INT8-W8A8 arm is checked at BOOT, not inside the cold-build window (#384).

THE DEFECT THIS PINS. The production INT8-W8A8 default did not boot on the
standard CT999 venv because the stock pypi sgl-kernel ships without
``int8_scaled_mm``. A loud error did exist -- ``CompressedTensorsW8A8Int8``
raises when the arm is missing -- but it raises during LAYER CONSTRUCTION,
inside the JIT cold-build window, so the operator saw ``ColdBuildWindowError``
and advice to lower ``--mem-fraction-static``: neither the cause nor a fix.
Its text also blames the sm75 gencode floor, which is the right diagnosis for
a 2080 Ti and the wrong one for a modern card on a wheel that simply lacks the
arm.

Both facts -- the quantization and whether the symbol imports -- are known at
argument resolution, so the refusal belongs there.
"""

import unittest

from sglang.srt.layers.quantization.w8a8_int8 import (
    INT8_ARM_RUNBOOK_SECTION,
    int8_arm_available,
    require_int8_arm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRefusal(CustomTestCase):
    def test_int8_without_the_arm_is_refused_by_name(self):
        with self.assertRaises(RuntimeError) as cm:
            require_int8_arm("w8a8_int8", available=False)
        msg = str(cm.exception)
        self.assertIn("int8_scaled_mm", msg)
        self.assertIn("fork wheel", msg)
        self.assertIn(INT8_ARM_RUNBOOK_SECTION, msg)

    def test_the_message_names_the_mask_it_replaces(self):
        """So a reader who has SEEN the ColdBuildWindowError connects them."""
        with self.assertRaises(RuntimeError) as cm:
            require_int8_arm("compressed-tensors", available=False)
        msg = str(cm.exception)
        self.assertIn("ColdBuildWindowError", msg)
        self.assertIn("mem-fraction-static", msg)

    def test_compressed_tensors_spellings_both_covered(self):
        for q in ("compressed-tensors", "compressed_tensors", "W8A8_INT8"):
            with self.subTest(q=q):
                with self.assertRaises(RuntimeError):
                    require_int8_arm(q, available=False)


class TestPassThrough(CustomTestCase):
    def test_with_the_arm_present_it_is_silent(self):
        require_int8_arm("w8a8_int8", available=True)

    def test_unrelated_quantizations_are_untouched(self):
        for q in ("fp8", "awq", "gptq", "modelopt_fp4", None, ""):
            with self.subTest(q=q):
                require_int8_arm(q, available=False)

    def test_a_non_cuda_group_is_untouched(self):
        """CPU/NPU groups do not use this arm at all."""
        require_int8_arm("w8a8_int8", is_cuda_group=False, available=False)

    def test_the_probe_answers_for_this_interpreter(self):
        self.assertIsInstance(int8_arm_available(), bool)

    def test_the_default_path_consults_the_real_probe(self):
        """available=None must not silently mean 'fine'."""
        import unittest.mock as m

        with m.patch(
            "sglang.srt.layers.quantization.w8a8_int8.int8_arm_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                require_int8_arm("w8a8_int8")


if __name__ == "__main__":
    unittest.main()
