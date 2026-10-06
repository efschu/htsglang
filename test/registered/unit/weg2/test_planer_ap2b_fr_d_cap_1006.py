"""AP2b 1006 (nf-hwgen-ap2b, planner side): FR_D of the proposal is capped at the largest fraction WITH a Platztausch buffer.

AP2 oracle (deskq/done/nf-hwgen-ap2-1006-bericht.md): at N = 4 (2x RTX 5090 + 2x RTX 3080, 4x RTX 3090) the Form A solve
gives every D rank its whole expert share resident -- the proposal said ``--rank-moe-resident-fraction 1.000`` on all four
ranks, and the launcher refused it rightly (W120 Weg2PlatztauschBufferUnbuilt: "groesste Fraction mit Puffer 0.978"; a
rank without two scratch rows builds no buffer, the flip join finds no counterpart).  The ceiling belongs to the proposal,
not to the refusal (the refusal stays as the guard): ``propose_rules.fr_d_with_buffer`` caps each rank at
``planner.expert_residency.d_rank_fraction_caps`` (owned + pad local rows, the same count the launcher's map grades) and
names the cut in the reason text.  No reserve: the two rows ARE the buffer.

The NF-line launcher half (W71 census donor) and the oracle runs: nf-hwgen-ap2b-1006 (test_hw_ap2b_1006.py).
GPU-free, NVML-free.
"""

from __future__ import annotations

import os
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.planner import expert_residency as ER
    from sglang.srt.weg2 import propose as P
    from sglang.srt.weg2 import propose_rules as R
    import test_planer_apc_propose_1006 as T
except Exception as exc:  # pragma: no cover - no weg2 planner in this build
    pytest.skip(f"weg2 planner unavailable: {exc}", allow_module_level=True)


def setUpModule():
    T.setUpModule()


def tearDownModule():
    T.tearDownModule()


class TestRule(unittest.TestCase):
    def test_full_residency_is_cut_to_the_buffer_fraction(self):
        fr, caps, capped = R.fr_d_with_buffer([1.0] * 4, [94, 190, 114, 114], 1)
        self.assertEqual(fr, [0.978, 0.989, 0.982, 0.982])
        self.assertEqual(caps, ER.d_rank_fraction_caps([94, 190, 114, 114], 1))
        self.assertEqual(capped, [0, 1, 2, 3])

    def test_a_fraction_below_the_cap_is_untouched(self):
        fr, _caps, capped = R.fr_d_with_buffer([0.33, 0.7015, 0.9999], [147, 75, 290], 1)
        self.assertEqual(fr, [0.33, 0.701, 0.993])
        self.assertEqual(capped, [2])

    def test_note_names_rank_value_and_cap(self):
        fa = {"fr_capped": [1], "fr_raw": [0.5, 1.0], "fr": [0.5, 0.989]}
        self.assertIn("Rang 1 1.000 -> 0.989", R.fr_d_cap_note(fa))
        self.assertEqual(R.fr_d_cap_note({"fr_capped": []}), "")


class TestProposalAtFourCards(unittest.TestCase):
    def _fr_d(self, v):
        la = P.LaunchArgv(v["argv"], v["env"])
        fr = [float(x) for x in la.extra_get("d", "--rank-moe-resident-fraction").split(",")]
        owned = [int(float(x)) for x in la.extra_get("d", "--rank-moe-ratio").split(",")]
        return fr, owned, {w["key"]: w for w in v["werte"]}

    def test_every_d_rank_keeps_two_scratch_rows(self):
        for inv in ("n4_mixed", "n4_3090"):
            v = T._propose("nf", inv)
            fr, owned, w = self._fr_d(v)
            self.assertEqual(len(fr), 4)
            caps = ER.d_rank_fraction_caps(owned, 1)
            for f, o, c in zip(fr, owned, caps):
                self.assertLessEqual(f, c, (inv, fr, caps))
                self.assertGreaterEqual(o + 1 - ER.resident_rows(o + 1, f), 2, (inv, fr, owned))
            self.assertTrue(any(f == c for f, c in zip(fr, caps)), (inv, fr, caps))   # the cap was the binding term
            self.assertIn("Platztausch-Puffer", w["--extra-d --rank-moe-resident-fraction"]["grund"])

    def test_the_reference_rig_proposal_is_unchanged(self):
        v = T._propose("nf", "ref3")
        b = T._profile("nf")
        self.assertEqual(P.LaunchArgv(v["argv"], v["env"]).extra_get("d", "--rank-moe-resident-fraction"),
                         P.LaunchArgv(b.argv, b.env).extra_get("d", "--rank-moe-resident-fraction"))


if __name__ == "__main__":
    unittest.main()
