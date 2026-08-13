# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656 C22: a frame ballot that says WHICH term diverged, not merely that one did.

R1's ballot fingerprints the live slot set, the wave partition, the vector and
the direction into ONE number, so a divergence is detectable and unattributable.
Its message hedges accordingly -- "the ranks do not agree on the live slot set,
the wave partition or the vector" -- and then names the pool census as the
instrument, which only helps when the pool is what differs.

On boot_v2, 2026-08-13 16:00:42Z, it wasn't. The KV cap agreement had just
levelled the group and the three ranks' POOL CENSUS lines were IDENTICAL in
every field::

    PP0/PP1/PP2  size=579870 free=278572 cached=300034 unaccounted=1264

and the frames diverged anyway: PP1 framed 250257408, PP0/PP2 1658515222. So
there is a SECOND divergence source, of a different nature from the 40404-row
one, and the single digest cannot say which term carries it. Six rounds of it
followed; the purity valve kept the instance serving (that half worked), but a
successor would have had nothing to hunt with.

So the digest is now computed in THREE PARTS -- slots, waves, geometry -- and
each rides the reduction as its own ``[x, -x]`` MIN pair. The combined digest
is unchanged and still decides; the parts only ATTRIBUTE. That is six more
integers in a payload the round already reduces: no new collective, and the
collective COUNT invariant is untouched.

Hermetic: a stub runtime, no CUDA, no distributed.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime


def _runtime(*, vec=(30, 16, 18), n_layers=64):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._vec = tuple(vec)
    r._n_layers = n_layers
    return r


SLOTS = torch.tensor([1, 2, 3, 7, 11], dtype=torch.int64)
WAVES = ((0, 1), (2, 3))


class TheCombinedDigestIsUnchangedTest(unittest.TestCase):
    """The parts must not move the number the ballot actually votes on."""

    def test_the_parts_compose_to_the_digest_the_ballot_uses(self):
        r = _runtime()
        whole = r._frame_digest(SLOTS, "pp_to_tp", WAVES)
        parts = r._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        self.assertEqual(parts["frame"], whole)

    def test_every_part_is_positive_and_negatable(self):
        """The MIN pair trick needs a value whose negation is unambiguous."""
        r = _runtime()
        parts = r._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        for name, value in parts.items():
            self.assertGreaterEqual(value, 0, name)
            self.assertLess(value, 1 << 62, name)


class EachPartMovesOnlyForItsOwnTermTest(unittest.TestCase):
    """The attribution is only worth having if it is actually specific."""

    def test_a_live_set_difference_moves_only_the_slots_part(self):
        r = _runtime()
        a = r._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        other = torch.tensor([1, 2, 3, 7, 12], dtype=torch.int64)
        b = r._frame_digest_parts(other, "pp_to_tp", WAVES)
        self.assertNotEqual(a["slots"], b["slots"])
        self.assertEqual(a["waves"], b["waves"])
        self.assertEqual(a["geometry"], b["geometry"])

    def test_a_wave_partition_difference_moves_only_the_waves_part(self):
        r = _runtime()
        a = r._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        b = r._frame_digest_parts(SLOTS, "pp_to_tp", ((0,), (1,), (2, 3)))
        self.assertNotEqual(a["waves"], b["waves"])
        self.assertEqual(a["slots"], b["slots"])
        self.assertEqual(a["geometry"], b["geometry"])

    def test_a_vector_difference_moves_only_the_geometry_part(self):
        a = _runtime(vec=(30, 16, 18))._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        b = _runtime(vec=(30, 17, 17))._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        self.assertNotEqual(a["geometry"], b["geometry"])
        self.assertEqual(a["slots"], b["slots"])
        self.assertEqual(a["waves"], b["waves"])

    def test_the_direction_is_part_of_the_geometry(self):
        r = _runtime()
        a = r._frame_digest_parts(SLOTS, "pp_to_tp", WAVES)
        b = r._frame_digest_parts(SLOTS, "tp_to_pp", WAVES)
        self.assertNotEqual(a["geometry"], b["geometry"])


class TheReportNamesTheDivergingTermTest(unittest.TestCase):
    """What a successor actually reads at 03:00."""

    def _named(self, mine, group_lo, group_hi):
        return PhaseFlipRuntime._name_frame_divergence(mine, group_lo, group_hi)

    def test_it_names_the_live_slot_set_when_only_slots_differ(self):
        mine = {"slots": 11, "waves": 22, "geometry": 33}
        said = self._named(
            mine, {"slots": 10, "waves": 22, "geometry": 33},
            {"slots": 11, "waves": 22, "geometry": 33},
        )
        self.assertIn("live slot set", said)
        self.assertNotIn("wave partition", said)

    def test_it_names_the_wave_partition_when_only_waves_differ(self):
        mine = {"slots": 11, "waves": 22, "geometry": 33}
        said = self._named(
            mine, {"slots": 11, "waves": 21, "geometry": 33},
            {"slots": 11, "waves": 22, "geometry": 33},
        )
        self.assertIn("wave partition", said)
        self.assertNotIn("live slot set", said)

    def test_it_names_more_than_one_when_more_than_one_differs(self):
        mine = {"slots": 11, "waves": 22, "geometry": 33}
        said = self._named(
            mine, {"slots": 10, "waves": 21, "geometry": 33},
            {"slots": 11, "waves": 22, "geometry": 33},
        )
        self.assertIn("live slot set", said)
        self.assertIn("wave partition", said)

    def test_it_says_so_when_no_part_can_explain_it(self):
        """An unattributable divergence must READ as unattributable.

        Silence here would be the worst outcome: a successor would take the
        absence of a named term as evidence about the terms, when it is only
        evidence that the parts are not fine-grained enough yet.
        """
        mine = {"slots": 11, "waves": 22, "geometry": 33}
        said = self._named(mine, mine, mine)
        self.assertIn("no single term", said)


if __name__ == "__main__":
    unittest.main()
