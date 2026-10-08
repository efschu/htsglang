"""Auftrag 1006 (Befund 06.10. ~14:30Z): ``nvfp4_w4a8_sm86.cuh`` setzte das dynamische Shared-Memory-Attribut hinter einem
``static bool attr_set`` -- je PROZESS, ``cudaFuncSetAttribute`` wirkt aber je GERÄT (aktuelles Gerät).  Karte 1 setzte es, Karte 2
im selben Messprozess übersprang es: "invalid argument" beim Start auf der zweiten 3080.

Das ist eine QUELLTEXT-RATSCHE: sie verhindert, dass das Muster zurückkommt.  Dass die Wirkung stimmt (zwei Geräte im selben
Prozess starten beide), lässt sich nur am Metall beweisen (Messlauf mit zwei sm_86-Karten in EINEM Prozess); hier ist nichts davon
simuliert, nur die Host-Logik als Quelltext geprüft.  Dazu ein Modell der Entscheidungslogik mit zwei Geräte-IDs.
"""

import os
import re
import unittest

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "python", "flliper", "jit_kernel", "csrc", "gemm")
FILES = ("nvfp4_w4a8_sm86.cuh", "nvfp4_w4a8_decode_sm86.cuh")


def _src(name):
    with open(os.path.join(CSRC, name), encoding="utf-8") as fh:
        return fh.read()


class TestAttrIsPerDevice(CustomTestCase):
    def test_no_process_wide_flag_guards_the_attribute(self):
        for name in FILES:
            s = _src(name)
            # `static bool x = false;` (scalar, no index) next to a cudaFuncSetAttribute is the defect
            for m in re.finditer(r"static\s+bool\s+(\w+)\s*=\s*false\s*;", s):
                window = s[m.end(): m.end() + 400]
                self.assertNotIn("cudaFuncSetAttribute", window, f"{name}: process-wide flag {m.group(1)} guards cudaFuncSetAttribute")

    def test_the_launcher_tracks_the_attribute_per_device_index(self):
        s = _src("nvfp4_w4a8_sm86.cuh")
        i = s.index("inline void launch_cfg")
        body = s[i: s.index("// Tile variants", i)]
        self.assertRegex(body, r"static\s+bool\s+attr_set\s*\[\s*\w+\s*\]")
        self.assertIn("cudaGetDevice", body)
        self.assertLess(body.index("RuntimeDeviceCheck(cudaGetDevice"), body.index("cudaFuncSetAttribute(kernel"))
        self.assertRegex(body, r"attr_set\s*\[\s*cur_dev\s*\]\s*=\s*true")
        self.assertRegex(body, r"!attr_set\s*\[\s*cur_dev\s*\]")

    def test_decision_logic_model_with_two_devices(self):
        """The same decision as the launcher, with two device ids: the old per-process flag skips the second card."""
        def old(calls, dev, state):
            if not state.get("flag"):
                calls.append(dev)
                state["flag"] = True

        def new(calls, dev, state, kmax=64):
            tracked = 0 <= dev < kmax
            arr = state.setdefault("arr", [False] * kmax)
            if not tracked or not arr[dev]:
                calls.append(dev)
                if tracked:
                    arr[dev] = True

        for fn, want in ((old, [0]), (new, [0, 1])):
            calls, st = [], {}
            for dev in (0, 1, 0, 1):
                fn(calls, dev, st)
            self.assertEqual(calls, want, fn.__name__)


if __name__ == "__main__":
    unittest.main()
