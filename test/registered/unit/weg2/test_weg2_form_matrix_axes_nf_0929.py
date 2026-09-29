"""nextflash's FORM_MATRIX_AXES record (fmt int4): the D forms of the NF line, as agreed with
the architecture seat 29.09. (VRAM-VERTRAG-0929.md §3.3 'NF-Formen D'). Without the record
``matrix_spec('nextflash', 'int4')`` is refused -- and NF must never read the 27B matrix."""

import unittest

from sglang.srt.weg2 import form_measures as fm
from sglang.srt.weg2 import profile_records as pr
from sglang.test.test_utils import CustomTestCase


class NextflashMatrix(CustomTestCase):
    def setUp(self):
        self.spec = fm.matrix_spec("nextflash", "int4")

    def test_forms_and_startability(self):
        forms = {f.name: f for f in self.spec.forms}
        self.assertEqual(set(forms), {"A", "A_T3232", "A_T0", "A_ST", "B"})
        self.assertEqual({n for n, f in forms.items() if f.startable}, {"A", "A_T3232", "A_T0"})
        self.assertIn("M1 D-solo", forms["A_ST"].why_not)
        self.assertIn("29.09. 00:05Z", forms["B"].why_not)
        # the running form: owned cut with the round rule, token vector from the planner
        self.assertEqual(forms["A"].axes["owned_cut"], "round")
        self.assertNotIn("tokens", forms["A"].axes)
        self.assertEqual(forms["A_T3232"].axes["tokens"], "0,32,32")
        # every form is its own cell key
        self.assertEqual(len({fm.axes_key(f.axes) for f in forms.values()}), len(forms))

    def test_axes(self):
        self.assertEqual(self.spec.bs, (1, 2, 3, 4, 5, 6))
        self.assertEqual(self.spec.depths, ("2k", "10k", "32k", "97k", "240k", "257k"))
        self.assertEqual(self.spec.texts, ("code", "prose", "thinking"))
        self.assertEqual(self.spec.temps, ("warm",))

    def test_capacity_is_the_262144_pin(self):
        forms = {f.name: f for f in self.spec.forms}
        pts = dict(fm.DEPTH_POINTS)
        for name in ("A", "A_T3232", "A_T0"):
            for depth, cap in self.spec.capacity[name].items():
                self.assertLessEqual(cap * pts[depth], 262144, (name, depth))
                self.assertGreater((cap + 1) * pts[depth] if cap < 6 else 262145, 262144, (name, depth))
        st, why = fm.structural_state(self.spec, forms["A"], {"bs": 3, "depth": "97k"})
        self.assertEqual(st, fm.STATE_IMPOSSIBLE_CAPACITY)
        st, _ = fm.structural_state(self.spec, forms["A"], {"bs": 2, "depth": "97k"})
        self.assertIsNone(st)
        st, why = fm.structural_state(self.spec, forms["B"], {"bs": 1, "depth": "2k"})
        self.assertEqual(st, fm.STATE_IMPOSSIBLE_START)
        self.assertIn("RECHNUNG", self.spec.provenance)

    def test_own_record_not_a_constant_not_borrowed(self):
        rec = [r for r in pr.records("nextflash") if r.name == fm.MATRIX_RECORD]
        self.assertEqual([(r.fmt, r.measured_on) for r in rec], [("int4", "nextflash")])
        self.assertNotIn(fm.MATRIX_RECORD, pr.constants_of("nextflash"))
        with self.assertRaises(fm.FormMeasuresError):
            fm.matrix_spec("nextflash", "nvfp4")


if __name__ == "__main__":
    unittest.main()
