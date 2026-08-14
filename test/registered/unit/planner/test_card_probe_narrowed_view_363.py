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
"""#363 defect 3: a scheduler rank must be able to find the rig's card probe.

THE DEFECT, as measured on metal in the ACT window (2026-08-14).

``#513`` replaced a newest-by-mtime probe lookup with a KEYED one, so a probe
measured over a different card set could not become this rig's profile. The key
is the SET of visible card UUIDs, compared for EQUALITY. That is the right
comparison for the process that writes the probe -- and the wrong one for every
process that reads it, because a scheduler rank runs under a NARROWED
``CUDA_VISIBLE_DEVICES`` and therefore sees one card, not three:

    CUDA_VISIBLE_DEVICES=0,1,2 -> FOUND
    CUDA_VISIBLE_DEVICES=1     -> NONE
    CUDA_VISIBLE_DEVICES=0     -> NONE

So on any real multi-rank boot NO rank can match the probe, the planner feed
raises ``PlannerFeedUnavailable('no card probe on disk')``, and the stage table
comes out ``1 stage(s), ... 0 flip target(s)``. That -- not the missing
per-stage measurements -- is why #363's flip targets are zero.

THE FIX, and the invariant it must not break.

Equality is replaced by CONTAINMENT: a probe matches a view when it DESCRIBES
every card that view can see (``visible <= probe``), under the same driver.
This keeps #513's protection pointing the way #513 aimed it -- a probe taken
while the arbiter had handed out only two of three cards still cannot serve a
three-card view, because two cards do not describe three -- while making the
narrowed view, which is the only view a rank ever has, resolvable.

Matching is on UUID throughout, never on a device index: a UUID is the one
name for a card that survives ``CUDA_VISIBLE_DEVICES`` narrowing unchanged,
and an index is exactly what narrowing renumbers.

The whole probe is returned, not the visible slice of it. That is deliberate
and load-bearing: ``key_solver.rates_from_probe`` indexes cards by
``cuda_index`` in the ``--rank-gpu-id`` space (the full-inventory space), so a
rank solving a three-rank layout needs all three cards' rates even though it
can only see its own card.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

A = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
B = "GPU-bbbbbbbb-0000-0000-0000-000000000002"
C = "GPU-cccccccc-0000-0000-0000-000000000003"
D = "GPU-dddddddd-0000-0000-0000-000000000004"
DRIVER = "595.58.03"


class _ProbeCache(unittest.TestCase):
    """A cache directory plus the two injection points the readers expose."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cardprobe363-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, uuids, driver=DRIVER, *, mtime=1000, tag="probe"):
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            CARD_PROBE_VERSION,
            card_probe_cache_path,
        )

        path = os.path.join(
            self.dir, os.path.basename(card_probe_cache_path(uuids, driver))
        )
        with open(path, "w") as f:
            json.dump(
                {
                    "version": CARD_PROBE_VERSION,
                    "driver": driver,
                    "tag": tag,
                    "cards": [
                        {"uuid": u, "name": "card", "cuda_index": i}
                        for i, u in enumerate(uuids)
                    ],
                    "pairs": [],
                },
                f,
            )
        os.utime(path, (mtime, mtime))
        return path

    def look(self, visible, driver=DRIVER):
        """The lookup as a rank performs it: only the cards IT can see."""
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            matching_cached_probe_json,
        )

        return matching_cached_probe_json(
            cache_dir=self.dir, inventory=(list(visible), driver)
        )


# ---------------------------------------------------------------------------
# 1. The falsifier: the metal reproduction, as a test
# ---------------------------------------------------------------------------


class TestNarrowedRankCanFindTheProbe(_ProbeCache):
    """Reproduces the three commands recorded in TICKET_363_ACT_VERDICT §6.1."""

    def test_the_three_commands_from_metal(self):
        """CVD=0,1,2 FOUND; CVD=1 NONE; CVD=0 NONE -- all three must be FOUND."""
        self.write([A, B, C], tag="rig")
        self.assertIsNotNone(self.look([A, B, C]), "full view (this one passed)")
        # These two are the defect. Red before the fix.
        self.assertIsNotNone(self.look([B]), "CUDA_VISIBLE_DEVICES=1")
        self.assertIsNotNone(self.look([A]), "CUDA_VISIBLE_DEVICES=0")

    def test_a_narrowed_rank_gets_the_WHOLE_probe_not_its_own_slice(self):
        """rates_from_probe indexes by cuda_index over the full rig, so a rank
        that can see one card still needs all three cards' rates."""
        self.write([A, B, C], tag="rig")
        got = self.look([B])
        self.assertIsNotNone(got)
        self.assertEqual(
            sorted(c["uuid"] for c in got["cards"]),
            sorted([A, B, C]),
            "a narrowed rank must still receive every card's rates",
        )

    def test_every_rank_of_a_tp3_boot_agrees_on_one_probe(self):
        """Three ranks, three narrowed views, one rig: same probe on each.

        A per-rank divergence here would be worse than the miss it replaces --
        ranks would solve different layouts from different rates."""
        self.write([A, B, C], tag="rig")
        seen = [self.look([u]) for u in (A, B, C)]
        self.assertTrue(all(p is not None for p in seen))
        self.assertEqual({p["tag"] for p in seen}, {"rig"})

    def test_the_readers_both_carry_the_fix(self):
        """The two audit sites of #513, exercised as themselves."""
        from sglang.srt.planner.rig_profile_source import (  # noqa: PLC0415
            _latest_card_probe,
        )
        from sglang.srt.planner.solver_api import cached_card_probe  # noqa: PLC0415

        self.write([A, B, C], tag="rig")
        inv = ([B], DRIVER)
        self.assertIsNotNone(cached_card_probe(cache_dir=self.dir, inventory=inv))
        self.assertIsNotNone(_latest_card_probe(cache_dir=self.dir, inventory=inv))


# ---------------------------------------------------------------------------
# 2. #513's protection, still pointing the way #513 aimed it
# ---------------------------------------------------------------------------


class TestContainmentDoesNotUndo513(_ProbeCache):
    def test_a_two_card_probe_still_cannot_serve_a_three_card_view(self):
        """Audit #506's exact shape: the newer probe was taken while the
        arbiter had handed out two of three cards. Containment is directional,
        so it is still a miss -- and the older, complete probe still wins."""
        self.write([A, B, C], mtime=1000, tag="full")
        self.write([A, B], mtime=2000, tag="handed-out-two")
        got = self.look([A, B, C])
        self.assertIsNotNone(got)
        self.assertEqual(got["tag"], "full")

    def test_a_probe_that_does_not_describe_my_card_is_a_miss(self):
        """The rank can see D. No probe knows D. A rate invented for D would
        be worse than no rate at all."""
        self.write([A, B, C], tag="rig")
        self.assertIsNone(self.look([D]))

    def test_a_probe_from_another_driver_is_still_a_miss(self):
        self.write([A, B, C], driver="580.01", tag="old")
        self.assertIsNone(self.look([B], driver=DRIVER))

    def test_an_unresolvable_inventory_is_still_a_miss(self):
        """No NVML means no attribution. Containment must not degrade into
        'anything contains nothing, so take the newest file'."""
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            matching_cached_probe_json,
        )

        self.write([A, B, C], tag="rig")
        self.assertIsNone(
            matching_cached_probe_json(cache_dir=self.dir, inventory=(None, None))
        )
        self.assertIsNone(
            matching_cached_probe_json(cache_dir=self.dir, inventory=([], DRIVER))
        )

    def test_a_torn_file_is_skipped_not_raised_on(self):
        self.write([A, B, C], tag="rig")
        with open(os.path.join(self.dir, "card_probe-torn.json"), "w") as f:
            f.write("{not json")
        self.assertIsNotNone(self.look([B]))


# ---------------------------------------------------------------------------
# 3. Which probe wins, when more than one describes the view
# ---------------------------------------------------------------------------


class TestPreferenceOrder(_ProbeCache):
    def test_an_exact_match_beats_a_superset_even_when_older(self):
        """A probe of exactly this view measured those cards WITH that view's
        contention; a bigger probe is a fallback, not an equal."""
        self.write([A, B, C], mtime=2000, tag="superset")
        self.write([B], mtime=1000, tag="exact")
        self.assertEqual(self.look([B])["tag"], "exact")

    def test_the_tighter_superset_wins_over_the_looser_one(self):
        self.write([A, B, C, D], mtime=2000, tag="four")
        self.write([A, B, C], mtime=1000, tag="three")
        self.assertEqual(self.look([B])["tag"], "three")

    def test_newest_breaks_a_tie_between_equally_tight_probes(self):
        self.write([A, B, C], mtime=1000, tag="older")
        # Same card set, same driver, different file name convention.
        p = self.write([A, B, C], mtime=3000, tag="newer")
        os.rename(p, os.path.join(self.dir, "card_probe-othername.json"))
        os.utime(
            os.path.join(self.dir, "card_probe-othername.json"), (3000, 3000)
        )
        self.assertEqual(self.look([B])["tag"], "newer")


if __name__ == "__main__":
    unittest.main()
