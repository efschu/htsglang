"""#651: the gfx1103 MES-wedge policy.

The wedge itself is an amdgpu firmware hang and cannot be fixed here, so the
policy makes its triggering regime unreachable: a prefill-chunk cap (the GGUF
large-batch path runs one bf16 GEMM with M = the prefill chunk) and a
free-memory floor (the wedge was reproduced at ~3% free).

These tests pin the parts that would otherwise rot silently:

* that unaffected hardware is NOT capped -- a throughput cap applied to every
  ROCm box "just in case" is a silent regression, and is the most likely way
  this policy would be got wrong;
* that a CPU rank (arch None) is never blocked, since a mixed-device pipeline
  has one;
* that the gfx1100 case WARNS, because `HSA_OVERRIDE_GFX_VERSION=11.0.0` makes
  gfx1103 report itself as gfx1100 and the policy would otherwise pass a
  genuinely affected machine in silence.

Run: PYTHONPATH=<repo>/python python -m pytest -q \
    test/registered/unit/distributed/test_wedge_policy_651.py
"""

import importlib.util
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_POLICY = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "docs", "dev", "651", "wedge_policy.py")
)
_spec = importlib.util.spec_from_file_location("wedge_policy_651", _POLICY)
wedge_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wedge_policy)

check = wedge_policy.check_wedge_policy
MAX_CP = wedge_policy.MAX_CHUNKED_PREFILL
MIN_FREE = wedge_policy.MIN_FREE_MIB_FOR_LARGE_GEMM


class TestWedgePolicy(unittest.TestCase):
    def test_affected_arch_refuses_oversized_prefill_chunk(self):
        res = check("gfx1103", chunked_prefill_size=1024)
        self.assertFalse(res.ok)
        self.assertTrue(any("chunked-prefill-size" in e for e in res.errors))
        # The message must carry the measurement, not just a rule.
        self.assertTrue(any("M=1024 wedges" in e for e in res.errors))

    def test_affected_arch_accepts_the_cap(self):
        res = check("gfx1103", chunked_prefill_size=MAX_CP)
        self.assertTrue(res.ok, res.errors)
        # Accepting is not the same as claiming a fix.
        self.assertTrue(any("MITIGATIONS, not a fix" in w for w in res.warnings))

    def test_unaffected_arch_is_never_capped(self):
        """Can-fail guard: widening WEDGE_ARCHS to all ROCm breaks this.

        A datacentre card must keep its 1024 chunk; capping it would be a
        silent throughput regression with no measurement behind it.
        """
        for arch in ("gfx942", "gfx90a", "gfx1030"):
            res = check(arch, chunked_prefill_size=4096)
            self.assertTrue(res.ok, f"{arch} must not be capped: {res.errors}")
            self.assertEqual(res.warnings, [], f"{arch} must not warn")

    def test_cpu_rank_is_not_blocked(self):
        """arch None is a rank with no accelerator (#651 mixed-device PP)."""
        res = check(None, chunked_prefill_size=8192)
        self.assertTrue(res.ok)
        self.assertEqual(res.warnings, [])

    def test_overridden_gfx1100_warns_rather_than_passing_silently(self):
        res = check("gfx1100", chunked_prefill_size=1024)
        # Not an error: a real gfx1100 is not known to wedge.
        self.assertTrue(res.ok)
        self.assertTrue(any("HSA_OVERRIDE_GFX_VERSION" in w for w in res.warnings))

    def test_memory_floor_refuses_the_reproduced_pressure(self):
        """736 MiB free is the level at which the wedge was reproduced."""
        res = check("gfx1103", chunked_prefill_size=MAX_CP, free_mib=736)
        self.assertFalse(res.ok)
        self.assertTrue(any("headroom" in e for e in res.errors))

    def test_memory_floor_passes_with_headroom(self):
        res = check("gfx1103", chunked_prefill_size=MAX_CP, free_mib=MIN_FREE + 1)
        self.assertTrue(res.ok, res.errors)


if __name__ == "__main__":
    unittest.main()
