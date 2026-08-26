"""#784: the ceiling ladder's margin note grades against the BINDING LEVEL.

RED FIRST. Every case below fails on the pre-fix line (`vram_after < 1024`),
and the two marked CAN-FAIL are the ones that prove the fix is not cosmetic:
the verdict has to CHANGE ITS MIND in both directions, or the line is the same
line with a longer comment.

  #602 named the 1024 as overstating the gap on flip boots -- the binding level
  there was the arming floor at 1728/1825/2467 MiB. A warning that fires
  systematically wrongly devalues the warnings that are right.
"""

import sys
import types
import unittest


def _install_fake_corridor(band_floor, arming, phase):
    """Stand in for corridor_guard/phase_flip_presence without importing the
    runtime: this test must not need CUDA, NVML or a live boot."""
    cg = types.ModuleType("sglang.srt.managers.corridor_guard")
    cg.corridor_band_floor_mib = lambda: band_floor
    cg.arming_floor_mib = lambda *a, **k: arming
    cg.committed_arming_mib = lambda a: max(0, int(a) - band_floor)
    cg.net_free_mib = lambda free, a: int(free) - cg.committed_arming_mib(a)
    pf = types.ModuleType("sglang.srt.managers.phase_flip_presence")
    pf.read_active_phase = lambda *a, **k: phase
    sys.modules["sglang.srt.managers.corridor_guard"] = cg
    sys.modules["sglang.srt.managers.phase_flip_presence"] = pf


class CeilingMarginLevel(unittest.TestCase):
    BAND_FLOOR = 819
    ARMING = 1331  # 819 band floor + 512 shipped seam-entry reserve

    def _level(self, free, phase="pp", arming=None):
        _install_fake_corridor(self.BAND_FLOOR, arming or self.ARMING, phase)
        for m in [k for k in sys.modules if k.endswith("planner.bench_suite")]:
            del sys.modules[m]
        from sglang.srt.planner.bench_suite import _ceiling_margin_level

        return _ceiling_margin_level(free)

    def test_names_phase_and_applied_level(self):
        """Rule 5: a verdict line without phase AND level is incomplete."""
        phase, name, level, net, floor = self._level(1100)
        self.assertEqual(phase, "pp")
        self.assertEqual(name, "arming_floor")
        self.assertEqual(level, self.ARMING)
        self.assertEqual(floor, self.BAND_FLOOR)

    def test_net_is_free_minus_committed_arming(self):
        """The comparison is corridor_guard's own, not a second derivation."""
        _, _, _, net, _ = self._level(1100)
        self.assertEqual(net, 1100 - (self.ARMING - self.BAND_FLOOR))

    def test_canfail_warns_where_the_old_rule_was_silent(self):
        """CAN-FAIL A. 1100 MiB free on a flip boot.

        Old rule: 1100 >= 1024, silent. Binding level: net 588 < 819 -- thin
        against the floor that actually binds. The line must now speak."""
        _, _, _, net, floor = self._level(1100)
        self.assertLess(net, floor, "must warn: 1100 is thin against the arming floor")
        self.assertGreaterEqual(1100, 1024, "old rule was silent here")

    def test_canfail_silent_where_the_old_rule_warned(self):
        """CAN-FAIL B. 1000 MiB free on a boot with NO flip.

        Old rule: 1000 < 1024, warns. No phase marker means no seam to enter,
        so the band floor is the level: net 1000 >= 819, nothing to say. This
        is the spurious warning #602 stumbled over."""
        _, name, level, net, floor = self._level(1000, phase=None)
        self.assertEqual(name, "band_floor")
        self.assertEqual(level, self.BAND_FLOOR)
        self.assertGreaterEqual(net, floor, "must NOT warn: 1000 clears the band floor")
        self.assertLess(1000, 1024, "old rule warned here")

    def test_no_flip_boot_uses_band_floor_not_arming(self):
        phase, name, level, _, _ = self._level(900, phase=None)
        self.assertEqual(phase, "unknown")
        self.assertEqual(name, "band_floor")
        self.assertEqual(level, self.BAND_FLOOR)

    def test_thin_stays_thin(self):
        """A genuinely thin reading still warns -- the fix must not go blind."""
        _, _, _, net, floor = self._level(900)
        self.assertLess(net, floor)

    def test_fallback_says_so_instead_of_reinstating_1024(self):
        """corridor_guard unavailable: grade on the band and NAME the fallback."""
        for k in (
            "sglang.srt.managers.corridor_guard",
            "sglang.srt.managers.phase_flip_presence",
        ):
            sys.modules[k] = None
        for m in [k for k in sys.modules if k.endswith("planner.bench_suite")]:
            del sys.modules[m]
        from sglang.srt.planner.bench_suite import _ceiling_margin_level

        _, name, level, net, floor = _ceiling_margin_level(1000)
        self.assertIn("fallback", name)
        self.assertEqual(level, 819)
        self.assertEqual(net, 1000)


if __name__ == "__main__":
    unittest.main()
