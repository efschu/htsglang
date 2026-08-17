"""The crossing schedule for non-contiguous stages, and the levers it exposes.

A contiguous pipeline crosses `pp_size - 1` times per forward. The family full
plan -- 48 linear-attention layers on the 5090, the 16 interleaved
full-attention layers on the two 3080s -- crosses **31** times per token, and
this pins that number against the actual map rather than against a memo.

Two things the schedule makes checkable that prose could not:

* **the terminal-layer lever.** The final layer has no return crossing (its
  output goes to the head), so whichever stage owns it is owed one fewer
  crossing. Placing layer 63 on the SLOWEST link is therefore free, and the
  test below measures that rather than asserting it.
* **per-link cost.** The rig's edges are not alike, so a schedule's cost
  depends on which pairs it uses and not only on how many crossings it makes.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.distributed.pp_crossing_schedule import (
    Crossing,
    CrossingScheduleError,
    LoopbackLink,
    crossing_schedule,
    schedule_cost,
    stage_of_layer,
)
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 64
FA = [i for i in range(NUM_LAYERS) if i % 4 == 3]
GDN = [i for i in range(NUM_LAYERS) if i % 4 != 3]


def plan(fa_first_half, fa_second_half):
    return [frozenset(GDN), frozenset(fa_first_half), frozenset(fa_second_half)]


#: stage 0 = 5090, stage 1 = 3080 on x8, stage 2 = 3080 on x4.
SPLIT_8_8 = plan(FA[:8], FA[8:])


class TestTheFullPlanCrossingCount(CustomTestCase):
    def test_it_is_31_not_32(self):
        """16 out and 15 back: layer 63 is terminal, so it has no return."""
        self.assertEqual(len(crossing_schedule(SPLIT_8_8, NUM_LAYERS)), 31)

    def test_sixteen_leave_the_gdn_card_and_fifteen_return(self):
        sched = crossing_schedule(SPLIT_8_8, NUM_LAYERS)
        out = [c for c in sched if c.src == 0]
        back = [c for c in sched if c.dst == 0]
        self.assertEqual((len(out), len(back)), (16, 15))

    def test_every_crossing_is_adjacent_to_a_full_attention_layer(self):
        """The schedule must follow the MAP, not a guess about it."""
        for c in crossing_schedule(SPLIT_8_8, NUM_LAYERS):
            with self.subTest(after=c.after_layer):
                self.assertTrue(c.after_layer in FA or c.after_layer + 1 in FA)

    def test_a_contiguous_map_still_crosses_only_pp_size_minus_one(self):
        """The falsifier: 31 must come from non-contiguity, not from the
        function counting something else."""
        contiguous = [
            frozenset(range(0, 22)),
            frozenset(range(22, 43)),
            frozenset(range(43, 64)),
        ]
        self.assertEqual(len(crossing_schedule(contiguous, NUM_LAYERS)), 2)

    def test_slots_are_per_pair_and_start_at_zero(self):
        sched = crossing_schedule(SPLIT_8_8, NUM_LAYERS)
        for pair in {c.pair for c in sched}:
            slots = [c.slot for c in sched if c.pair == pair]
            self.assertEqual(slots, list(range(len(slots))))


class TestTheTerminalLayerLever(CustomTestCase):
    """Layer 63 is free to place on the slowest link. Measured, not asserted."""

    def _crossings_touching(self, sched, stage):
        return sum(1 for c in sched if c.src == stage or c.dst == stage)

    def test_owning_the_terminal_layer_costs_one_fewer_crossing(self):
        with_terminal = crossing_schedule(plan(FA[:8], FA[8:]), NUM_LAYERS)
        without_terminal = crossing_schedule(plan(FA[8:], FA[:8]), NUM_LAYERS)
        # stage 2 holds FA[8:] (which contains layer 63) in the first map.
        self.assertIn(63, FA[8:])
        self.assertEqual(self._crossings_touching(with_terminal, 2), 15)
        self.assertEqual(self._crossings_touching(without_terminal, 2), 16)

    def test_so_the_terminal_half_belongs_on_the_slowest_link(self):
        """The lever, priced. Stage 2 is the x4 edge here, so giving it the
        half that contains the terminal layer removes one crossing from the
        slowest pair -- for free, since the split is 8/8 either way."""
        x4_expensive = {(0, 2): 12.0, (2, 0): 12.0}
        cheap = 5.0
        terminal_on_x4 = schedule_cost(
            crossing_schedule(plan(FA[:8], FA[8:]), NUM_LAYERS), x4_expensive, cheap
        )
        terminal_on_x8 = schedule_cost(
            crossing_schedule(plan(FA[8:], FA[:8]), NUM_LAYERS), x4_expensive, cheap
        )
        self.assertLess(terminal_on_x4, terminal_on_x8)


class TestPerLinkCosting(CustomTestCase):
    """Slot-3's correction: price per LINK, because the edges differ. The
    absolute ms/pass figures are that survey's; what is checked here is that
    the model responds to WHICH pairs a schedule uses."""

    def test_moving_crossings_to_the_cheap_edge_lowers_the_cost(self):
        sched = crossing_schedule(SPLIT_8_8, NUM_LAYERS)
        expensive_x4 = schedule_cost(sched, {(0, 2): 12.0, (2, 0): 12.0}, 5.0)
        all_cheap = schedule_cost(sched, {}, 5.0)
        self.assertGreater(expensive_x4, all_cheap)

    def test_an_unpriced_pair_falls_back_rather_than_costing_zero(self):
        """A modelling gap must show up as a number, not vanish."""
        sched = crossing_schedule(SPLIT_8_8, NUM_LAYERS)
        self.assertEqual(schedule_cost(sched, {}, 1.0), float(len(sched)))

    def test_a_12_4_split_uses_the_x4_edge_less_than_8_8(self):
        """Why 8/8 is not free in time even though it is free in COUNT: the
        count is split-invariant, the per-link cost is not."""
        s88 = crossing_schedule(plan(FA[:8], FA[8:]), NUM_LAYERS)
        s124 = crossing_schedule(plan(FA[:12], FA[12:]), NUM_LAYERS)
        x4 = {(0, 2): 12.0, (2, 0): 12.0}
        self.assertEqual(len(s88), len(s124))  # count invariant
        self.assertGreater(schedule_cost(s88, x4, 5.0), schedule_cost(s124, x4, 5.0))


class TestTheMapIsValidated(CustomTestCase):
    def test_an_unowned_layer_is_refused_by_name(self):
        bad = [frozenset(GDN), frozenset(FA[:8]), frozenset(FA[8:-1])]
        with self.assertRaises(CrossingScheduleError) as cm:
            crossing_schedule(bad, NUM_LAYERS)
        self.assertIn("63", str(cm.exception))

    def test_a_doubly_owned_layer_is_refused_by_name(self):
        bad = [frozenset(GDN), frozenset(FA), frozenset([FA[0]])]
        with self.assertRaises(CrossingScheduleError) as cm:
            crossing_schedule(bad, NUM_LAYERS)
        self.assertIn("exactly one owner", str(cm.exception))

    def test_stage_of_layer_maps_every_layer(self):
        owner = stage_of_layer(SPLIT_8_8, NUM_LAYERS)
        self.assertEqual(len(owner), NUM_LAYERS)
        self.assertEqual(owner[0], 0)
        self.assertEqual(owner[3], 1)
        self.assertEqual(owner[63], 2)


class TestTheLoopbackDouble(CustomTestCase):
    """The double must be no more permissive than the transport it stands for,
    or driving a schedule through it proves nothing."""

    def test_a_whole_schedule_drives_end_to_end(self):
        sched = crossing_schedule(SPLIT_8_8, NUM_LAYERS)
        link = LoopbackLink()
        got = link.run(sched, payload_of=lambda c: c.after_layer)
        self.assertEqual(got, [c.after_layer for c in sched])
        self.assertEqual(len(link.sent), 31)
        self.assertEqual(len(link.received), 31)

    def test_a_missing_sender_refuses_rather_than_blocking(self):
        link = LoopbackLink()
        link._src, link._dst = 0, 1
        with self.assertRaises(CrossingScheduleError) as cm:
            link.recv(0, 0, timeout_s=0.01)
        self.assertIn("never arrived", str(cm.exception))

    def test_consuming_a_slot_twice_refuses(self):
        link = LoopbackLink()
        link._src, link._dst = 0, 1
        link.send(1, 0, "x", 1.0)
        link.recv(0, 0, 1.0)
        with self.assertRaises(CrossingScheduleError):
            link.recv(0, 0, 1.0)

    def test_reusing_a_slot_within_one_pass_refuses(self):
        link = LoopbackLink()
        link._src, link._dst = 0, 1
        link.send(1, 0, "x", 1.0)
        with self.assertRaises(CrossingScheduleError) as cm:
            link.send(1, 0, "y", 1.0)
        self.assertIn("already occupied", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
