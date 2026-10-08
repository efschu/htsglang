"""1570: the 1330 W2c fla-GDN case must call chunk_gated_delta_rule with the real extend contract.

04.10. 12:28Z the case passed no initial_state_indices; the kernel loads it unconditionally, so the reference
call raised a CompilationError and W2c printed SKIP -- 'bad=0' hid that fla-GDN was never checked per green stage.
CPU-only: reads the script source (no GPU, no triton)."""
import ast
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[4] / "scripts/dual_group/green_ladder_metal_1330.py"


def _calls(src, name):
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == name]


class GdnContract(unittest.TestCase):
    def setUp(self):
        self.src = SCRIPT.read_text()

    def test_extend_contract_arguments(self):
        calls = _calls(self.src, "chunk_gated_delta_rule")
        self.assertEqual(len(calls), 1)
        kw = {k.arg for k in calls[0].keywords}
        for need in ("initial_state", "initial_state_indices", "cu_seqlens", "use_qk_l2norm_in_kernel"):
            self.assertIn(need, kw)

    def test_indices_and_cu_seqlens_are_int32(self):
        self.assertIn("gidx = torch.zeros(1, dtype=torch.int32", self.src)
        self.assertIn("gcu = torch.tensor([0, T_g], dtype=torch.int32", self.src)

    def test_state_pool_is_fresh_per_call(self):
        self.assertIn("initial_state=gpool0.clone()", self.src)   # the kernel updates the pool in place

    def test_gqa_branch_covered(self):
        self.assertIn("T_g, Hg_g, H_g, K_g = 1024, 4, 8, 128", self.src)

    def test_required_kernels_skip_is_bad(self):
        self.assertIn("--require-kernels", self.src)
        self.assertIn("FAIL-NOT-RUN", self.src)
        self.assertIn("a SKIP is not a pass", self.src)

    def test_w2c_only_exit_code(self):
        self.assertIn("--w2c-only", self.src)
        self.assertIn("return 1 if w2c_bad else 0", self.src)


if __name__ == "__main__":
    unittest.main()
