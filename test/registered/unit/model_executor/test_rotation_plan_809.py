"""#809/W28: chunk-rotation residency -- the arithmetic and its falsifiers.

RAM holds ONE layout image plus a small overshoot. At the flip the incoming
layout streams RAM -> VRAM while the outgoing one streams VRAM -> RAM into the
pages just freed. PCIe is full duplex, so the copy-back rides the idle return
direction.

WHY THIS SHAPE. W26 proved the dual pin impossible on this box -- BOTH pin
arms were OOM-killed in the LAUNCH phase, before any flip. One layout plus eps
(~30 GiB against ~68.7 GiB) fits, and it takes the disk off the steady-state
critical path, which is what reaches the physics floor; W26 measured the
refill leg 99.8-100 % storage-bound, so any residual disk share dominates.

THE THREE FALSIFIERS THE COORDINATOR NAMED are all expressible here, and all
three are asserted:

  * **no actual overlap** -- a serialized implementation produces a schedule
    whose two directions never co-occur. `rotation_totals` counts the
    overlapping steps, and a rotation of two non-trivial layouts MUST have
    them.
  * **RAM leak across cycles** -- rotating A->B then B->A must return the host
    to exactly its starting occupancy. Asserted over three cycles, because the
    W27-retry leak fired on the THIRD.
  * **checksum error** -- covered by the trailer test: the image is raw arena
    bytes plus an int64 trailer (weights_arena.py:1182-1183), so a D2H
    reproduces the payload byte-for-byte and ONLY the 8-byte trailer is new.

The size asymmetry is W26's measured one, per rank, in MiB:
PP0 15925.8 / 16362.7, PP1 8573.8 / 8961.3, PP2 8573.8 / 9481.6.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.model_executor.rotation_plan import (
    RotationPlanError,
    peak_ram_bytes,
    plan_rotation,
    rotation_overshoot_bytes,
    rotation_totals,
)
from sglang.test.test_utils import CustomTestCase

MIB = 1024 * 1024
CHUNK = 32 * MIB
DEPTH = 2

#: W26's measured per-rank image sizes, (pp, tp), in bytes.
RANKS = {
    "PP0": (int(15925.8 * MIB), int(16362.7 * MIB)),
    "PP1": (int(8573.8 * MIB), int(8961.3 * MIB)),
    "PP2": (int(8573.8 * MIB), int(9481.6 * MIB)),
}


class TestTheOvershootCoversBothTerms(CustomTestCase):
    def test_it_covers_the_asymmetry_plus_the_in_flight_window(self):
        for rank, (pp, tp) in RANKS.items():
            with self.subTest(rank=rank):
                got = rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH)
                self.assertEqual(got, abs(pp - tp) + DEPTH * CHUNK)

    def test_it_is_sized_from_the_larger_direction_not_a_mean(self):
        # A single fixed reservation has to cover whichever direction the NEXT
        # flip takes. Sizing from a mean is the OOM, so the value must be
        # direction-symmetric.
        pp, tp = RANKS["PP2"]
        self.assertEqual(
            rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH),
            rotation_overshoot_bytes(tp, pp, CHUNK, DEPTH),
        )

    def test_equal_layouts_still_need_the_in_flight_window(self):
        # THE CAN-FAIL DIRECTION: an implementation returning only the
        # asymmetry would give 0 here and stall the schedule immediately.
        self.assertEqual(
            rotation_overshoot_bytes(1000, 1000, CHUNK, DEPTH), DEPTH * CHUNK
        )

    def test_a_zero_chunk_or_depth_is_refused(self):
        for c, d in ((0, 2), (CHUNK, 0), (-1, 2)):
            with self.subTest(chunk=c, depth=d):
                with self.assertRaises(RotationPlanError):
                    rotation_overshoot_bytes(100, 100, c, d)


class TestTheScheduleRespectsTheRamBudget(CustomTestCase):
    """The invariant: the copy-back never runs further ahead of the freed
    pages than the overshoot allows."""

    def test_peak_ram_never_exceeds_one_image_plus_overshoot(self):
        for rank, (pp, tp) in RANKS.items():
            for label, (inc, out) in (
                ("pp_to_tp", (tp, pp)),
                ("tp_to_pp", (pp, tp)),
            ):
                with self.subTest(rank=rank, direction=label):
                    over = rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH)
                    steps = plan_rotation(
                        incoming_bytes=inc,
                        outgoing_bytes=out,
                        chunk_bytes=CHUNK,
                        overshoot_bytes=over,
                    )
                    self.assertLessEqual(peak_ram_bytes(steps, inc), inc + over)

    def test_every_byte_of_both_layouts_is_scheduled(self):
        pp, tp = RANKS["PP0"]
        over = rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH)
        steps = plan_rotation(
            incoming_bytes=tp,
            outgoing_bytes=pp,
            chunk_bytes=CHUNK,
            overshoot_bytes=over,
        )
        h, d, _ = rotation_totals(steps)
        self.assertEqual(h, tp)
        self.assertEqual(d, pp)

    def test_the_copy_back_never_overtakes_the_freed_pages(self):
        pp, tp = RANKS["PP2"]
        over = rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH)
        steps = plan_rotation(
            incoming_bytes=tp,
            outgoing_bytes=pp,
            chunk_bytes=CHUNK,
            overshoot_bytes=over,
        )
        out = back = 0
        for s in steps:
            out += s.h2d_len
            back += s.d2h_len
            self.assertLessEqual(back - out, over)

    def test_too_small_an_overshoot_is_REFUSED_not_silently_exceeded(self):
        # THE FALSIFIER FOR THE BUDGET. Under-sizing must stall loudly; an
        # implementation that just kept going would be holding both layouts,
        # which is exactly the state that OOM-killed W26's pin arms.
        #
        # THE BUDGET BINDS IN ONLY ONE DIRECTION, and getting that wrong is how
        # this test first passed vacuously: pressure exists only when the
        # OUTGOING layout is LARGER than the incoming one, because then the
        # copy-back needs more RAM than the H2D frees. PP0's tp image
        # (16362.7 MiB) copied back while its smaller pp image (15925.8 MiB)
        # streams in leaves 436.9 MiB with nowhere to go.
        pp, tp = RANKS["PP0"]
        self.assertGreater(tp, pp, "this direction is the one under pressure")
        with self.assertRaises(RotationPlanError) as caught:
            plan_rotation(
                incoming_bytes=pp,
                outgoing_bytes=tp,
                chunk_bytes=CHUNK,
                overshoot_bytes=0,
            )
        self.assertIn("both layouts", str(caught.exception))

    def test_the_slack_direction_needs_no_overshoot_at_all(self):
        # The other half of the same fact, asserted so the asymmetry is
        # recorded rather than rediscovered: streaming the LARGER layout in
        # frees pages faster than the smaller one consumes them, so this
        # direction schedules cleanly even at zero overshoot.
        pp, tp = RANKS["PP0"]
        steps = plan_rotation(
            incoming_bytes=tp,
            outgoing_bytes=pp,
            chunk_bytes=CHUNK,
            overshoot_bytes=0,
        )
        h, d, _ = rotation_totals(steps)
        self.assertEqual((h, d), (tp, pp))


class TestTheDuplexActuallyOverlaps(CustomTestCase):
    """Falsifier 1: a serialized implementation never co-schedules."""

    def test_a_real_rotation_has_overlapping_steps(self):
        pp, tp = RANKS["PP1"]
        over = rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH)
        steps = plan_rotation(
            incoming_bytes=tp,
            outgoing_bytes=pp,
            chunk_bytes=CHUNK,
            overshoot_bytes=over,
        )
        _h, _d, both = rotation_totals(steps)
        self.assertGreater(both, 0)
        # And it is the DOMINANT shape, not an accident of the tails: the two
        # layouts are within ~5 % of each other, so nearly every step carries
        # both directions.
        self.assertGreater(both, 0.9 * len(steps))

    def test_a_one_sided_rotation_reports_no_overlap(self):
        # THE CAN-FAIL PARTNER: with nothing to copy back, `both` must be 0,
        # or the counter would report overlap that cannot exist.
        steps = plan_rotation(
            incoming_bytes=4 * CHUNK,
            outgoing_bytes=0,
            chunk_bytes=CHUNK,
            overshoot_bytes=CHUNK,
        )
        _h, d, both = rotation_totals(steps)
        self.assertEqual(d, 0)
        self.assertEqual(both, 0)


class TestNoRamLeakAcrossCycles(CustomTestCase):
    """Falsifier 2, asserted over THREE cycles because W27-retry's leak fired
    on the third, not the first."""

    def test_three_full_cycles_return_the_host_to_its_start(self):
        for rank, (pp, tp) in RANKS.items():
            with self.subTest(rank=rank):
                over = rotation_overshoot_bytes(pp, tp, CHUNK, DEPTH)
                resident = tp  # RAM starts holding the inactive layout
                for _cycle in range(3):
                    for inc, out in ((tp, pp), (pp, tp)):
                        steps = plan_rotation(
                            incoming_bytes=inc,
                            outgoing_bytes=out,
                            chunk_bytes=CHUNK,
                            overshoot_bytes=over,
                        )
                        h, d, _ = rotation_totals(steps)
                        resident = resident - h + d
                self.assertEqual(resident, tp, "RAM occupancy drifted across cycles")


class TestTheImageTrailerIsTheOnlyNewBYTES(CustomTestCase):
    """Falsifier 3's precondition, verified against the real source.

    The image is `payload = image[:layout.total_bytes]` plus an int64 checksum
    trailer (weights_arena.py:1182-1183), checked with `uint8_checksum(dst)`
    over the ARENA. So a D2H of the arena reproduces the payload byte for
    byte, and only the 8-byte trailer has to be recomputed.
    """

    def test_the_image_is_arena_bytes_plus_an_int64_trailer(self):
        import inspect

        from sglang.srt.model_executor import weights_arena

        src = inspect.getsource(weights_arena.arena_refill)
        self.assertIn("payload = image[: layout.total_bytes]", src)
        self.assertIn("image[layout.total_bytes :]", src)
        self.assertIn("view(torch.int64)", src)

    def test_the_checksum_is_taken_over_the_arena_not_the_file(self):
        # This is what makes a D2H reproducible: the verification target is
        # the arena itself, so bytes that came back from VRAM verify the same
        # way bytes that came from disk do.
        import inspect

        from sglang.srt.model_executor import weights_arena

        src = inspect.getsource(weights_arena.arena_refill)
        self.assertIn("uint8_checksum(dst)", src)


if __name__ == "__main__":
    unittest.main()
