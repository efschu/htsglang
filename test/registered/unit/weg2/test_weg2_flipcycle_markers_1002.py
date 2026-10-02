"""FLIPCYCLE markers (02.10.): every serial flip stage with its physics floor
(front), and the park's own phases on D (H3) -- the lines the next boot is
read by (FLIPZYKLUS-PHYSIK-NF-1002.md section 5)."""

import inspect
import unittest

from sglang.srt.environ import envs
from sglang.srt.weg2 import d_park_runtime, front


class TheMarkers(unittest.TestCase):
    def test_front_prints_one_line_per_stage_with_floor(self):
        src = inspect.getsource(front.Front)
        self.assertIn('"WEG2-FLIPCYCLE stage=%s dir=%s>%s epoch=%d ms=%.0f floor_ms=%d"', src)
        self.assertIn('"WEG2-FLIPCYCLE stage=total dir=%s>%s epoch=%d ms=%.0f floor_ms=%d"', src)
        self.assertEqual(envs.SGLANG_WEG2_FLIPCYCLE_LEGS_FLOOR_MS.get(), 1000)

    def test_park_prints_its_phases(self):
        src = inspect.getsource(d_park_runtime.park_running)
        self.assertIn("WEG2-FLIPCYCLE stage=park", src)
        order = ["result", "filter", "draft", "end", "retract", "mark", "yield", "l3mark",
                 "depth", "clamp"]
        pos = [src.index('_ph("%s")' % n) for n in order]
        self.assertEqual(pos, sorted(pos))


if __name__ == "__main__":
    unittest.main()
