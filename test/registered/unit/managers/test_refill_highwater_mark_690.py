"""#690: the census boundary between the arena commit and the H2D refill.

WHY THIS MARK EXISTS. The retained #631 census lines from 14 pp_to_tp
flips put the worst segment on every rank in ``gdn_state->weights_refill``
-- and because a mark is taken AFTER each pre-cutover mover runs, that
segment is the weights refill leg itself, not the GDN exchange. Across
ranks the leg spans 1057 / 1327 / 1903 ms while every rank refills the
SAME 9614.9 MiB TP image, and the ordering tracks the cards' PCIe link
width (x8 / x8 / x4) rather than any per-rank work. That reading says
"bandwidth-bound copy" -- but the leg also contains an arena commit
(``_commit_refill_high_water``), and an allocation that stalls on the
driver would look identical in a single combined bar.

So the leg needs exactly one interior boundary to separate the two, and
these pins hold it in place. The nesting pin is the load-bearing one: a
mark that fires but lands on the wrong side of the copy would attribute
the transfer to the commit and send the fix to the wrong module, which is
the failure mode #631's docstring already names.

The no-return-path contract from #631 applies unchanged here: this mark
sits inside the cutover's no-return region, so it may cost a log line
when it goes wrong, never a flip.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers import phase_flip_seam_census as census
from sglang.srt.managers.phase_flip_boot import PhaseFlipStacks

MIB = 1024 * 1024


def _flat_probe():
    """A probe that never moves. These pins are about ORDER, not arithmetic.

    Feeding a changing probe here would make the assertions depend on the
    trough logic that #631 already pins, and a failure would no longer say
    which of the two instruments broke.
    """
    return (8000 * MIB, 1000 * MIB, 900 * MIB)


def _stub_stacks(carrier):
    """A PhaseFlipStacks shell carrying only what the commit path reads.

    __init__ builds a whole flip's worth of layouts and images; the method
    under test reads exactly two attributes, so binding it to a shell keeps
    the pin on the boundary instead of on the constructor. The stub-drift
    risk this normally carries (#624) is bounded by
    test_the_commit_path_still_reads_only_what_this_stub_provides below.
    """
    stacks = SimpleNamespace(
        arena_carrier=carrier,
        refill_high_water_bytes=lambda: 9614 * MIB,
    )
    stacks._commit_refill_high_water = (
        PhaseFlipStacks._commit_refill_high_water.__get__(stacks, SimpleNamespace)
    )
    return stacks


def _labels(record):
    return [stage[0] for stage in record.stages]


class TestTheMarkEmits(unittest.TestCase):
    def setUp(self):
        census.reset()
        self.addCleanup(census.reset)

    def test_the_commit_emits_a_refill_highwater_boundary(self):
        census.begin("pp_to_tp", 2, probe=_flat_probe)
        _stub_stacks(None)._commit_refill_high_water()
        record = census.end()

        self.assertIn(
            "refill_highwater",
            _labels(record),
            "the refill leg has no interior boundary: a commit stall and a "
            "bandwidth-bound copy stay indistinguishable in one bar",
        )

    def test_a_rank_without_a_carrier_still_reports_the_boundary(self):
        """The no-carrier commit is a no-op, and that IS the measurement.

        Marking inside the carrier guard would drop the boundary on exactly
        the ranks where the commit costs nothing, so their remaining bar
        would silently mean 'commit + copy' while their peers' meant 'copy'.
        """
        census.begin("pp_to_tp", 0, probe=_flat_probe)
        _stub_stacks(None)._commit_refill_high_water()
        record = census.end()

        self.assertIn("refill_highwater", _labels(record))

    def test_the_commit_still_happens_before_the_mark(self):
        """Order within the method: grow the arena, THEN close the segment.

        Marking first would put the commit's own cost into the copy's bar --
        the precise misattribution this instrument was added to prevent.
        """
        events = []

        class _Carrier:
            def set_active_prefix(self, nbytes):
                events.append(("commit", nbytes))
                return 0

        real_mark = census.mark

        def _recording_mark(label):
            events.append(("mark", label))
            real_mark(label)

        census.mark = _recording_mark
        self.addCleanup(setattr, census, "mark", real_mark)

        census.begin("pp_to_tp", 1, probe=_flat_probe)
        _stub_stacks(_Carrier())._commit_refill_high_water()
        census.end()

        self.assertEqual(
            [kind for kind, _ in events],
            ["commit", "mark"],
            "the mark must close the commit segment, not open it",
        )
        self.assertEqual(events[0][1], 9614 * MIB)


class TestItNestsInTheCensusFormat(unittest.TestCase):
    """The boundary has to land BETWEEN the movers, not beside them."""

    def setUp(self):
        census.reset()
        self.addCleanup(census.reset)

    def test_the_boundary_falls_inside_the_gdn_to_weights_segment(self):
        """Replays the pre-cutover sequence of phase_flip_runtime.

        Marks there are taken after each mover runs, so the real order is
        gdn_state (GDN leg done) -> [commit -> copy] -> weights_refill. The
        new boundary must split that middle span rather than appear before
        gdn_state or after weights_refill.
        """
        census.begin("pp_to_tp", 2, probe=_flat_probe)
        census.mark("gdn_state")
        _stub_stacks(None)._commit_refill_high_water()  # commit half ends here
        census.mark("weights_refill")  # copy half ends here
        record = census.end()

        labels = _labels(record)
        self.assertIn("refill_highwater", labels)
        self.assertLess(
            labels.index("gdn_state"),
            labels.index("refill_highwater"),
            "boundary landed before the GDN leg closed",
        )
        self.assertLess(
            labels.index("refill_highwater"),
            labels.index("weights_refill"),
            "boundary landed after the copy: the copy's time would be "
            "charged to the commit",
        )

    def test_it_is_a_single_boundary_per_flip(self):
        """One commit per flip, so one mark -- a duplicate would halve the
        copy's bar into two unnamed pieces and break the segment count the
        #631 line reports."""
        census.begin("pp_to_tp", 2, probe=_flat_probe)
        _stub_stacks(None)._commit_refill_high_water()
        record = census.end()

        self.assertEqual(_labels(record).count("refill_highwater"), 1)


class TestTheInstrumentCannotKillAFlip(unittest.TestCase):
    """#631's contract, re-pinned at the new call site.

    This mark is the first census call on the phase_flip_boot side, so the
    guarantee has to be shown here and not merely inherited by argument.
    """

    def setUp(self):
        census.reset()
        self.addCleanup(census.reset)

    def test_the_commit_path_runs_with_no_census_open(self):
        _stub_stacks(None)._commit_refill_high_water()

    def test_a_probe_that_raises_does_not_escape_the_commit(self):
        def _angry_probe():
            raise RuntimeError("NVML is having a day")

        census.begin("pp_to_tp", 2, probe=_angry_probe)
        _stub_stacks(None)._commit_refill_high_water()
        census.end()


class TestTheStubTracksProduction(unittest.TestCase):
    """#624 guard: the shell above must not drift behind the real method."""

    def test_the_commit_path_still_reads_only_what_this_stub_provides(self):
        import inspect

        source = inspect.getsource(PhaseFlipStacks._commit_refill_high_water)
        reads = {
            name
            for name in ("arena_carrier", "refill_high_water_bytes")
            if f"self.{name}" in source
        }
        self.assertEqual(
            reads,
            {"arena_carrier", "refill_high_water_bytes"},
            "the commit path changed its attribute reads; the SimpleNamespace "
            "shell in this file no longer stands in for the real object",
        )


if __name__ == "__main__":
    unittest.main()
