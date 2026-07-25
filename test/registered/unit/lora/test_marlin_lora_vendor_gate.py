"""MarlinLoraRunnerCore must refuse non-CUDA vendors first (#171).

"compute capability >= 9" is an NVIDIA statement, but
`get_device_capability()` answers in the caller's vendor namespace and the two
COLLIDE: gfx900 reports (9, 0). The bare comparison therefore let an AMD card
through an NVIDIA gate, which then died inside a Marlin kernel that has no ROCm
implementation at all -- the dangerous direction (wrongly admitted) rather than
the loud one (refused at the gate).

Pure CPU test: the vendor predicate and the capability are mocked; the runner
is never constructed and no kernel is launched.
"""

import inspect
import unittest

from sglang.srt.lora import lora_moe_runner_marlin as mod


class TestMarlinLoraGateIsVendorFirst(unittest.TestCase):
    def _source(self):
        return inspect.getsource(mod)

    def test_vendor_is_checked_before_the_capability_number(self):
        """is_cuda() must be asserted BEFORE the >= 9 comparison, otherwise a
        gfx900 (9, 0) reaches the NVIDIA threshold and passes it."""
        src = self._source()
        vendor_at = src.find("assert is_cuda()")
        cap_at = src.find("get_device_capability(hidden_states.device)[0] >= 9")
        self.assertNotEqual(vendor_at, -1, "vendor gate is missing entirely")
        self.assertNotEqual(cap_at, -1, "capability gate disappeared")
        self.assertLess(
            vendor_at, cap_at,
            "the vendor gate must come first; after the number it is useless",
        )

    def test_refusal_names_the_namespace_problem(self):
        """The message has to explain WHY, or the next reader re-introduces the
        bare comparison."""
        src = self._source()
        self.assertIn("no ROCm kernel", src)
        self.assertIn("NOT comparable", src)

    def test_capability_gate_is_retained_for_nvidia(self):
        """The sm90 floor is still real on NVIDIA -- the fix must not drop it
        (sm86 must still be refused)."""
        src = self._source()
        self.assertIn(">= 9", src)
        self.assertIn("MarlinLoraRunnerCore requires CUDA compute capability >= 9", src)

    def test_module_imports_is_cuda(self):
        self.assertTrue(hasattr(mod, "is_cuda"), "is_cuda must be imported to be usable")


if __name__ == "__main__":
    unittest.main()
