"""#656: the weights arena must be backed for the layout it is ABOUT to hold.

THE BUG THIS PINS, measured on metal 2026-08-11. Rung 3 was written when "PP
is the larger layout on every rank of this rig" held, so the tail was
committed on tp->pp and released on pp->tp. ``--pp-stage-ratio 15,9,8``
derives 32,16,16 layers over 64, which puts the middle rank's PP layout
(6690 MiB) BELOW its TP layout (7924 MiB). The tp->pp leg then decommitted to
the smaller PP layout, and the next pp->tp refill copied the larger TP image
into the released tail:

    torch.AcceleratorError: CUDA error: invalid argument
      weights_arena.py:386 in arena_refill -> dst.copy_(payload)

inside the flip's no-return region, killing all three ranks at the first
flip after a tp->pp.

RED-FIRST: :func:`test_pp_to_tp_commits_before_it_copies` and
:func:`test_the_metal_sequence_that_faulted` both fail on the pre-fix tree --
the first because nothing committed before the copy, the second because the
recorded call order shows the copy hitting a 6690 MiB commitment with a 7924
MiB payload. :func:`test_the_gate_prices_the_growing_leg` fails because
``_arena_tail_bytes`` returned 0 for pp->tp by construction.
"""

import types
import unittest

from sglang.srt.managers.phase_flip_boot import PhaseFlipStacks
from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP

MIB = 1024 * 1024


class _Layout:
    def __init__(self, mib):
        self.total_bytes = int(mib * MIB)


class _Carrier:
    """A carrier that FAULTS if written beyond its committed prefix."""

    def __init__(self, nbytes, committed):
        self._nbytes = int(nbytes)
        self.committed = int(committed)
        self.calls = []

    def set_active_prefix(self, active_bytes):
        want = max(0, min(int(active_bytes), self._nbytes))
        self.calls.append(("prefix", want))
        if want > self.committed:
            self.committed = want
            return 0.0
        released = (self.committed - want) / MIB
        self.committed = want
        return released

    def pending_tail_bytes(self, active_bytes):
        return max(0, min(int(active_bytes), self._nbytes) - self.committed)


def _stacks(pp_mib, tp_mib, committed_mib, recorder, holds="tp"):
    """A PhaseFlipStacks whose refill is real and whose copy is instrumented.

    #809/W28: ONE host image. ``holds`` must name the layout the FIRST refill
    of a test streams in, because the rotation alternates it from there.
    """
    hi = max(pp_mib, tp_mib)
    carrier = _Carrier(hi * MIB, committed_mib * MIB)
    st = PhaseFlipStacks.__new__(PhaseFlipStacks)
    st.layout_pp = _Layout(pp_mib)
    st.layout_tp = _Layout(tp_mib)
    st.rotation_image = object()
    st.image_holds = holds
    st.arena = types.SimpleNamespace(numel=lambda: hi * MIB)
    st.arena_carrier = carrier
    return st, carrier


class _Recorder:
    def __init__(self):
        self.copies = []
        #: #856: the timing record the caller handed in on each copy, so the
        #: instrument's WIRING is pinned here and not merely tolerated.
        self.timings = []


def _patched_refill(st, carrier, rec, monkey):
    """Run PhaseFlipStacks.refill with arena_refill replaced by a recorder
    that FAULTS exactly as the driver does: writing past the committed span."""

    def fake_rotate_arena(
        *,
        arena,
        host_image,
        incoming_bytes,
        outgoing_bytes,
        chunk_bytes,
        depth,
        ring,
        ops=None,
        timing=None,
        priming=False,
        verify_incoming=True,
    ):
        # #856 CONTRACT CHANGE, TAKEN DELIBERATELY RATHER THAN WIDENED AWAY.
        # `arena_refill` gained an optional `timing` record so the refill leg
        # -- 91% of a tp_to_pp flip -- can say whether it was storage-bound or
        # link-bound instead of reporting one aggregate rate. It is an
        # INSTRUMENT: default None, no behaviour change, so this stub accepting
        # the kwarg is the whole of the adaptation. What must NOT be silently
        # accepted is the instrument coming unwired, so the record is captured
        # and asserted by `test_the_refill_leg_is_instrumented` below.
        rec.timings.append(timing)
        # #809/W28: the leg streams `incoming` IN and reads `outgoing` OUT of
        # the same arena, so BOTH layouts must be backed. The high-water is the
        # max of the two exactly as it was under the old `restore=` arm -- now
        # structurally rather than as a recovery path.
        need = max(int(incoming_bytes), int(outgoing_bytes))
        rec.copies.append((need, carrier.committed))
        if need > carrier.committed:
            raise RuntimeError(
                f"CUDA error: invalid argument -- wrote {need} bytes into an "
                f"arena committed to {carrier.committed}"
            )

    monkey(fake_rotate_arena)


class TestArenaHighWater(unittest.TestCase):
    def setUp(self):
        # #809/W28: the copy the leg makes is now the ROTATION, and
        # `_timed_arena_refill` imports it from this module at call time, so
        # this is the seam to instrument.
        import sglang.srt.model_executor.rotation_executor as rx

        self.rx = rx
        self._orig = rx.rotate_arena
        self.rec = _Recorder()

    def tearDown(self):
        self.rx.rotate_arena = self._orig

    def _install(self, st, carrier):
        _patched_refill(
            st,
            carrier,
            self.rec,
            lambda fn: setattr(self.rx, "rotate_arena", fn),
        )

    # -- the rank where TP is the larger layout (the regression) ----------

    def test_pp_to_tp_commits_before_it_copies(self):
        # PP 6690, TP 7924, arena committed down to PP by an earlier tp->pp.
        st, carrier = _stacks(6690, 7924, 6690, self.rec)
        self._install(st, carrier)
        st.refill(PP_TO_TP)
        need, committed_at_copy = self.rec.copies[0]
        self.assertEqual(need, 7924 * MIB)
        self.assertGreaterEqual(committed_at_copy, need)

    def test_the_metal_sequence_that_faulted(self):
        # tp->pp then pp->tp, which is exactly the order the instance died in.
        st, carrier = _stacks(6690, 7924, 7924, self.rec, holds="pp")
        self._install(st, carrier)
        st.refill(TP_TO_PP)
        st.refill(PP_TO_TP)
        for need, committed in self.rec.copies:
            self.assertGreaterEqual(committed, need)

    def test_the_tail_is_still_released_after_the_copy(self):
        # The fix must not buy safety by keeping the arena fully committed --
        # that would give away rung 3's entire purpose.
        st, carrier = _stacks(6690, 7924, 7924, self.rec, holds="pp")
        self._install(st, carrier)
        st.refill(TP_TO_PP)
        self.assertEqual(carrier.committed, 6690 * MIB)

    # -- the rank where PP is the larger layout (must not regress) --------

    def test_pp_larger_still_releases_on_pp_to_tp(self):
        st, carrier = _stacks(9115, 7924, 9115, self.rec)
        self._install(st, carrier)
        st.refill(PP_TO_TP)
        self.assertEqual(carrier.committed, 7924 * MIB)

    def test_pp_larger_recommits_on_tp_to_pp(self):
        st, carrier = _stacks(9115, 7924, 7924, self.rec, holds="pp")
        self._install(st, carrier)
        st.refill(TP_TO_PP)
        need, committed_at_copy = self.rec.copies[0]
        self.assertGreaterEqual(committed_at_copy, need)
        self.assertEqual(carrier.committed, 9115 * MIB)

    def test_high_water_is_the_max_of_both_layouts(self):
        st, _ = _stacks(6690, 7924, 6690, self.rec)
        self.assertEqual(st.refill_high_water_bytes(), 7924 * MIB)
        st2, _ = _stacks(9115, 7924, 9115, self.rec)
        self.assertEqual(st2.refill_high_water_bytes(), 9115 * MIB)

    def test_the_refill_leg_is_instrumented(self):
        # #856: the seam census puts `refill_highwater->weights_refill` at 91%
        # of a tp_to_pp flip, and it reported ONE aggregate MiB/s -- which
        # cannot separate a storage-bound leg from a link-bound one. The split
        # only exists if the caller actually hands the record down, so that
        # wiring is pinned HERE, at the call site this file already owns.
        # Without this, updating the stub's signature would have silently
        # tolerated the instrument being removed again.
        from sglang.srt.model_executor.weights_arena import RefillLegTiming

        st, carrier = _stacks(6690, 7924, 6690, self.rec)
        self._install(st, carrier)
        st.refill(PP_TO_TP)
        self.assertTrue(self.rec.timings, "no refill was recorded at all")
        for got in self.rec.timings:
            self.assertIsInstance(
                got,
                RefillLegTiming,
                "the refill leg must be handed a timing record; a None here "
                "means the bound attribution is dead and the leg is back to "
                "one aggregate rate",
            )

    def test_a_carrierless_stack_does_not_raise(self):
        st, _ = _stacks(6690, 7924, 6690, self.rec)
        st.arena_carrier = None
        self._install(st, _Carrier(7924 * MIB, 7924 * MIB))
        st.refill(PP_TO_TP)  # rung 3 off: nothing to commit, nothing to release


class TestGatePricesBothLegs(unittest.TestCase):
    """The commit is an allocation in the no-return region; it must be priced
    on whichever leg has to grow, not on a leg chosen at authoring time."""

    def _runtime(self, pp_mib, tp_mib, committed_mib):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        hi = max(pp_mib, tp_mib)
        carrier = _Carrier(hi * MIB, committed_mib * MIB)
        stacks = types.SimpleNamespace(
            arena_carrier=carrier,
            layout_pp=_Layout(pp_mib),
            layout_tp=_Layout(tp_mib),
            refill_high_water_bytes=lambda: hi * MIB,
        )
        rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        rt._census_scheduler = types.SimpleNamespace(phase_flip_stacks=stacks)
        return rt

    def test_prices_the_pp_to_tp_leg_when_tp_is_larger(self):
        rt = self._runtime(6690, 7924, 6690)
        self.assertEqual(rt._arena_tail_bytes(PP_TO_TP), (7924 - 6690) * MIB)

    def test_prices_the_tp_to_pp_leg_when_pp_is_larger(self):
        rt = self._runtime(9115, 7924, 7924)
        self.assertEqual(rt._arena_tail_bytes(TP_TO_PP), (9115 - 7924) * MIB)

    def test_zero_when_already_backed(self):
        rt = self._runtime(6690, 7924, 7924)
        self.assertEqual(rt._arena_tail_bytes(PP_TO_TP), 0)

    def test_no_carrier_is_zero_not_a_raise(self):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        rt._census_scheduler = types.SimpleNamespace(phase_flip_stacks=None)
        self.assertEqual(rt._arena_tail_bytes(PP_TO_TP), 0)


if __name__ == "__main__":
    unittest.main()
