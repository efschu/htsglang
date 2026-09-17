"""#1467: the mamba slot allocator never synchronises the device in the
scheduler thread -- the #924D trail is opt-in and the double-free refusal is
evaluated behind a CUDA event (immediately on a CPU allocator)."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.mem_cache.allocator import mamba as m
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


class Test1467(unittest.TestCase):
    def setUp(self):
        self.a = m.MambaSlotAllocator(8, device="cpu")

    def test_alloc_free_roundtrip_without_trail(self):
        self.assertFalse(m._SLOT_TRAIL)
        s = self.a.alloc(3)
        self.assertEqual(s.tolist(), [1, 2, 3])
        self.assertEqual(len(self.a.free_slots), 5)
        self.a.free(s)
        self.assertEqual(len(self.a.free_slots), 8)
        self.assertFalse(getattr(self.a, "_slot_provenance", None))    # trail off: no book entries
        self.assertFalse(bool(self.a.slot_used[1:4].any()))

    def test_double_free_is_still_refused_with_the_releasers_stack(self):
        s = self.a.alloc(2)
        self.a.free(s)
        with self.assertRaises(m.MambaSlotDoubleFree):
            self.a.free(s)

    def test_sentinel_minus_one_is_routed_to_the_dummy_row_and_not_refused(self):
        s = self.a.alloc(1)
        idx = torch.cat([s, torch.tensor([-1], dtype=s.dtype)])
        self.a.free(idx)                       # must not raise, must not sync
        self.assertFalse(bool(self.a.slot_used[s].any()))
        self.assertEqual(self.a._1467_pending, [])

    def test_note_924d_is_silent_unless_armed(self):
        with self.assertLogs(m.logger, level="INFO") as cm:
            m.logger.info("probe")
            m.note_924d("alloc_component", rid="r1", slot=torch.tensor([1]))
        self.assertEqual([r.getMessage() for r in cm.records], ["probe"])


if __name__ == "__main__":
    unittest.main()
