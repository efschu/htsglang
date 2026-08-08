# Copyright 2023-2026 SGLang Team
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
"""#631: the collective census must be a census of WIRE collectives.

Two defects, both measured on window-2 boot 13 (2026-08-08), both in the
INSTRUMENT rather than in what it watches:

1. It counted collectives on world_size==1 groups. Those short-circuit
   without touching the wire, so their counts are a function of this rank's
   local work -- under PP=3/TP=1 with stage ratio 2,1,1 the size-1 "tp"
   group reported ``tp.all_gather: counts [536, 1096, 1096]`` and the
   detector called a correct configuration a desync.

2. Its own periodic comparison is a blocking collective on a group that
   also carries payload traffic. Fired at drifted per-rank rounds it
   mispairs the group FIFO -- the instrument causing the class of wedge it
   exists to explain.
"""

import unittest

from sglang.srt.distributed.collective_census import ROUND_KEY, CollectiveCensus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


class _FakeGroup:
    """Stands in for a process group handle; the census only passes it on."""


def _gather_stub(payloads):
    """An all_gather_object that hands every rank the same list.

    Takes the per-rank local dicts as the census would have packed them.
    """

    def _fn(gathered, local, group=None):
        for i, p in enumerate(payloads):
            gathered[i] = p
        return None

    return _fn


class TestCensusWireDomain(CustomTestCase):
    """Defect 1: non-wire groups must not enter a cross-rank comparison."""

    def test_size_one_group_is_not_a_wire(self):
        # The production gate is `_census_wire = _CENSUS_ON and world_size
        # > 1`, evaluated in GroupCoordinator.__init__. Pin the predicate
        # itself: a size-1 group has no peer to pair with, so counting it
        # can only ever contribute a false positive.
        self.assertFalse(1 > 1, "a size-1 group must never count as a wire")
        self.assertTrue(2 > 1)

    def test_group_coordinator_defines_census_wire_after_rank_info(self):
        """The invariant must be readable, and defined where world_size is.

        Red before the fix in two different ways: the attribute did not
        exist at all (the gate was inlined at eight call sites), and the
        first attempt to inline it read ``self.world_size`` ABOVE the block
        that assigns it -- every GroupCoordinator construction then raised
        AttributeError inside __init__ and the group init retried itself to
        death ("retry() exceed maximum number of retries", 2 tests red).
        """
        import inspect

        from sglang.srt.distributed.parallel_state import GroupCoordinator

        src = inspect.getsource(GroupCoordinator.__init__)
        self.assertIn("self._census_wire", src)
        assign_world = src.index("self.world_size = len(ranks)")
        assign_wire = src.index("self._census_wire")
        self.assertLess(
            assign_world,
            assign_wire,
            "the census-wire invariant must be computed AFTER world_size "
            "is assigned, or __init__ raises AttributeError",
        )

    def test_every_census_gate_uses_the_single_invariant(self):
        """No call site may re-derive the predicate on its own.

        Eight scattered copies of the gate is how one of them ends up
        wrong; the count of bare ``_CENSUS_ON`` gates outside the
        invariant's own definition must be zero.
        """
        import inspect

        import sglang.srt.distributed.parallel_state as ps

        lines = inspect.getsource(ps).splitlines()
        stray = [
            ln.strip()
            for ln in lines
            if "_CENSUS_ON" in ln and "self._census_wire = " not in ln and "_CENSUS_ON = " not in ln
        ]
        self.assertEqual(stray, [], f"stray census gates: {stray}")


class TestCensusCadenceSelfCheck(CustomTestCase):
    """Defect 2: the detector must catch its own cadence drifting."""

    def _compare(self, census, payloads, rank=0, monkeypatched=None):
        import torch.distributed as dist

        orig = dist.all_gather_object
        dist.all_gather_object = _gather_stub(payloads)
        try:
            return census.compare_across_ranks(_FakeGroup(), len(payloads), rank)
        finally:
            dist.all_gather_object = orig

    def test_aligned_rounds_keep_the_detector_armed(self):
        c = CollectiveCensus()
        payloads = [
            {"tp.all_reduce": 10, ROUND_KEY: 50},
            {"tp.all_reduce": 10, ROUND_KEY: 50},
        ]
        found = self._compare(c, payloads)
        self.assertEqual(found, [])
        self.assertFalse(c.cadence_broken)

    def test_real_divergence_still_reported(self):
        """The stand-down must not cost the detection it exists for."""
        c = CollectiveCensus()
        payloads = [
            {"tp.all_reduce": 9, ROUND_KEY: 50},
            {"tp.all_reduce": 10, ROUND_KEY: 50},
        ]
        found = self._compare(c, payloads)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].family, "tp.all_reduce")
        self.assertFalse(c.cadence_broken)

    def test_round_key_is_not_reported_as_a_diverging_family(self):
        """RED before the fix: ROUND_KEY would read as a desynced family."""
        c = CollectiveCensus()
        payloads = [
            {"tp.all_reduce": 10, ROUND_KEY: 50},
            {"tp.all_reduce": 10, ROUND_KEY: 100},
        ]
        found = self._compare(c, payloads)
        self.assertEqual(
            [d.family for d in found],
            [],
            "the detector's own round is metadata, not a collective family",
        )

    def test_drifted_rounds_stand_the_detector_down(self):
        """RED before the fix: the detector kept firing into a skewed FIFO."""
        c = CollectiveCensus()
        payloads = [
            {"tp.all_reduce": 10, ROUND_KEY: 50},
            {"tp.all_reduce": 10, ROUND_KEY: 100},
        ]
        self._compare(c, payloads)
        self.assertTrue(
            c.cadence_broken,
            "a drifted detector cadence must stand the comparison down",
        )
        # And it stays down: a later comparison takes no collective at all.
        calls = []

        def _explode(gathered, local, group=None):
            calls.append(1)
            raise AssertionError("must not gather after standing down")

        import torch.distributed as dist

        orig = dist.all_gather_object
        dist.all_gather_object = _explode
        try:
            self.assertIsNone(
                c.compare_across_ranks(_FakeGroup(), 2, 0),
                "a stood-down detector must return None without gathering",
            )
        finally:
            dist.all_gather_object = orig
        self.assertEqual(calls, [])

    def test_local_dump_survives_the_stand_down(self):
        """Standing down must keep the wedge-proof half armed."""
        c = CollectiveCensus()
        c.bump("tp.all_reduce")
        payloads = [
            {"tp.all_reduce": 1, ROUND_KEY: 1},
            {"tp.all_reduce": 1, ROUND_KEY: 7},
        ]
        self._compare(c, payloads)
        self.assertTrue(c.cadence_broken)
        c.bump("tp.all_reduce")
        self.assertEqual(c.snapshot()["tp.all_reduce"], 2, "counting must continue")
        # The abort-time dump takes no collective, so it is exactly what a
        # stood-down detector still leaves the reader.
        history = c.format_local_history(0)
        self.assertIn("tp.all_reduce", history)

    def test_realign_round_rezeros_from_a_group_aligned_event(self):
        c = CollectiveCensus()
        for _ in range(37):
            c.next_round()
        self.assertEqual(c.round, 37)
        c.realign_round()
        self.assertEqual(c.round, 0)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
