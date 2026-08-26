"""#873: a dominant census segment is a BOUNDARY, and reporting it as a number invites a hand-fit.

WHAT HAPPENED, and it is the reason this file exists rather than a perf patch.
An operator asked why a flip takes ~6.4 s when the data would move in ~1 s, read
the seam census line, and hand-solved a two-point linear fit across three ranks:

  rank | MiB   | refill_highwater->weights_refill | copy @ 8843 MiB/s | RESIDUAL
  PP0  | 15926 | 4819.9 ms                        | 1801 ms           | 3019 ms
  PP1  |  8574 | 3988.6 ms                        |  969 ms           | 3020 ms
  PP2  |  8574 | 3979.0 ms                        |  969 ms           | 3010 ms

...and found a byte-independent, rank-independent, link-independent 3.0 s
constant worth 47 % of the flip. The fit is arithmetically correct. The constant
is an ARTEFACT: the segment brackets four mechanisms with four different cost
drivers, so no one-rate-plus-intercept model of it can be right, and its
intercept absorbs whatever the model omitted. (The same boot's own
`rotation_phase_report` lines decompose that segment as
`save 4.342 + checksum 0.319 + wait 0.084 + d2h-issue 0.026 + h2d-issue 0.020 +
ring 0.001 + plan 0.001` on PP0 -- host-side staging memcpy, not transfer, with
`gpu-span d2h 0.000s / h2d 0.000s`.)

THE CLASS. A segment whose two marks bracket several mechanisms reports a number
that cannot be attributed, and an unattributable number in the DOMINANT position
is worse than no number: it is the one a reader will model. This is the #851
indicator family -- `weights_refill` NAMES a copy and its mass is a host memcpy
-- and it is the shape `RotationPhases` (#809/W28) already names one level down
("the phrase named a bound nobody had observed").

WHY A LINK AND NOT A NEW MARK. The decomposition ALREADY EXISTS. It is emitted
by `rotation_phase_report` as a separate, unreferenced log line, and the census's
own worst-segment line does not mention that it exists. Two instruments, one
segment, no link -- so the reader of the dominant term cannot find the
decomposition and fits a model instead. Adding a third instrument would leave
that unchanged.

WHAT THIS FILE PINS:
  * a segment can be handed a decomposition by whoever ran the work;
  * the worst-segment line REPORTS that decomposition, with the unexplained
    remainder named rather than distributed (the #846 rule the RotationPhases
    docstring already states for its own residual);
  * a worst segment with NO decomposition is ANNOUNCED as unattributable, so
    "do not fit a model to this" is in the log instead of in a reviewer's head;
  * registering a decomposition may never raise on the flip's no-return path,
    which is the contract every instrument on this path has carried since #631.

Hermetic: pure formatting and bookkeeping over a stubbed probe, no CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_flip_seam_census import SeamCensus
from sglang.test.test_utils import CustomTestCase

# The real PP0 pp_to_tp numbers from boot_w40_857strict_0826_0516.log, in
# seconds, so the assertions are about the specimen and not about a toy.
PP0_ROTATION_TERMS = (
    ("save", 4.342),
    ("checksum", 0.319),
    ("wait", 0.084),
    ("d2h-issue", 0.026),
    ("h2d-issue", 0.020),
    ("ring", 0.001),
    ("plan", 0.001),
)


def _census(marks):
    """A census whose walk is driven by a fake clock, so segment widths are
    exact and the assertions are about the FORMATTER rather than about timing.

    The probe is stubbed to None: this file is about the time axis, and a
    memory probe that fails is already covered by the #631 census tests.
    """
    census = SeamCensus("pp_to_tp", 0, probe=lambda: None)
    census.times = [(label, t) for label, t in marks]
    census.stages = [(label, -1, -1, -1) for label, _ in marks]
    return census


def _pp0_walk():
    """entry .. done with PP0's real segment widths, in walk order."""
    widths = [
        ("entry", 0.0),
        ("flip_writeback", 1.6),
        ("hicache_quiesce", 117.3),
        ("resident_release", 589.3),
        ("plan", 0.7),
        ("backing_release", 72.5),
        ("allocator_cache_release", 0.5),
        ("backing_restore", 83.7),
        ("wave_loop_skipped", 0.2),
        ("gdn_state", 424.1),
        ("refill_highwater", 0.2),
        ("weights_refill", 4819.9),
        ("cutover", 113.5),
        ("done", 171.9),
    ]
    t = 0.0
    marks = []
    for label, ms in widths:
        t += ms / 1000.0
        marks.append((label, t))
    return marks


class TestTheWalkFixtureIsFaithful(CustomTestCase):
    """CONTROL. Passes with and without the fix. If the fixture does not
    reproduce the specimen's shape, every red below is about something else."""

    def test_the_worst_segment_is_the_refill_one(self):
        line = _census(_pp0_walk()).format_timing_line()
        self.assertIn("worst 'refill_highwater->weights_refill'", line)
        self.assertIn("4819.9", line)

    def test_the_commit_segment_is_present_and_tiny(self):
        """The answer to "was it the page commit?" was in every census line of
        that boot, at 0.2 ms, sorted to the far end where it reads as noise.
        Pinned as a fact about the walk, not as a claim about the fix."""
        line = _census(_pp0_walk()).format_timing_line()
        self.assertIn("gdn_state->refill_highwater 0.2", line)


class TestASegmentCanBeExplained(CustomTestCase):
    def test_a_decomposition_can_be_registered_for_a_segment(self):
        """RED. Without this the census has nowhere to put a decomposition that
        already exists elsewhere in the same process."""
        census = _census(_pp0_walk())
        census.explain("weights_refill", PP0_ROTATION_TERMS)
        line = census.format_timing_line()
        self.assertIn("save 4342", line.replace(".0", ""))

    def test_the_worst_segment_line_names_every_term(self):
        census = _census(_pp0_walk())
        census.explain("weights_refill", PP0_ROTATION_TERMS)
        line = census.format_timing_line()
        for name, _ in PP0_ROTATION_TERMS:
            self.assertIn(name, line)

    def test_the_unexplained_remainder_is_NAMED_not_distributed(self):
        """#846's rule, inherited: parts that do not add up to the whole must
        say so. 4819.9 ms of segment against 4793.0 ms of terms leaves 26.9 ms
        -- the rung-3 arena release and the mark itself -- and that has to be
        visible rather than folded into `save`."""
        census = _census(_pp0_walk())
        census.explain("weights_refill", PP0_ROTATION_TERMS)
        line = census.format_timing_line()
        self.assertIn("UNEXPLAINED", line)

    def test_a_decomposition_that_overruns_its_segment_still_reports(self):
        """A decomposition whose terms EXCEED the segment is a real condition
        (two timers, two clocks, a mark stamped late). It must be reported, not
        clamped to zero and not raised -- a clamp would read as perfect
        reconciliation, which is the failure this instrument exists to refuse."""
        census = _census(_pp0_walk())
        census.explain("weights_refill", (("save", 9.999),))
        line = census.format_timing_line()
        self.assertIn("OVERRUN", line)


class TestAnUnexplainedDominantSegmentIsAnnounced(CustomTestCase):
    """The half that would have stopped the hand-fit."""

    def test_an_undecomposed_worst_segment_says_it_cannot_be_attributed(self):
        line = _census(_pp0_walk()).format_timing_line()
        self.assertIn("NOT DECOMPOSED", line)

    def test_the_notice_warns_against_fitting_a_model_to_it(self):
        """Named explicitly, because the failure was not that someone lacked
        the number -- it was that the number invited a model."""
        line = _census(_pp0_walk()).format_timing_line()
        self.assertIn("do not fit", line.lower())

    def test_a_decomposed_worst_segment_does_NOT_carry_the_notice(self):
        """A warning that fires on every line is a warning nobody reads."""
        census = _census(_pp0_walk())
        census.explain("weights_refill", PP0_ROTATION_TERMS)
        self.assertNotIn("NOT DECOMPOSED", census.format_timing_line())

    def test_a_decomposition_on_a_NON_dominant_segment_does_not_silence_it(self):
        """The notice is about the segment a reader will act on. Explaining a
        0.2 ms segment must not make the 4819.9 ms one look attributed."""
        census = _census(_pp0_walk())
        census.explain("refill_highwater", (("commit", 0.0002),))
        self.assertIn("NOT DECOMPOSED", census.format_timing_line())


class TestTheInstrumentCannotKillAFlip(CustomTestCase):
    """The no-return-path contract this module has carried since #631: an
    instrument may never be the reason a flip dies. Every one of these is a
    shape a caller can actually produce."""

    def test_explaining_an_unknown_label_does_not_raise(self):
        census = _census(_pp0_walk())
        census.explain("no_such_mark", (("x", 1.0),))
        self.assertIsInstance(census.format_timing_line(), str)

    def test_a_garbage_decomposition_does_not_raise(self):
        census = _census(_pp0_walk())
        census.explain("weights_refill", "not a sequence of pairs")
        self.assertIsInstance(census.format_timing_line(), str)

    def test_a_non_numeric_term_does_not_raise(self):
        census = _census(_pp0_walk())
        census.explain("weights_refill", (("save", None), ("checksum", 0.3)))
        self.assertIsInstance(census.format_timing_line(), str)

    def test_the_module_level_explain_is_a_no_op_with_no_census_open(self):
        """`mark()` already has this property and `explain()` is called from the
        same places, so it needs it for the same reason."""
        from sglang.srt.managers import phase_flip_seam_census as sc

        sc.explain("weights_refill", (("save", 1.0),))

    def test_a_short_walk_still_formats(self):
        census = _census([("entry", 0.0)])
        census.explain("entry", (("x", 1.0),))
        self.assertIsInstance(census.format_timing_line(), str)


class TestTheRefillWiringExists(CustomTestCase):
    """The static half, and deliberately a SMALL one.

    The runtime half above is the real check for this class: whichever segment
    dominates a given flip is not knowable at the desk, so the enforceable rule
    is that the DOMINANT one is attributed or announced -- which the formatter
    now guarantees for every walk, on every rank, without anyone maintaining a
    list.

    What a static test CAN add is that the one decomposition the tree already
    knows how to produce stays WIRED. Delete the `explain` call and the log
    stays honest -- it reverts to "NOT DECOMPOSED" -- but the information that
    exists in the process is lost again, which is the exact state #873 found.

    NOT A PER-MARK CLASSIFICATION AUDIT, and that is a judgement rather than an
    omission. The walk has ~20 possible marks across two files and two
    registration mechanisms (`mark("x")` and a mover's `census_label`), and an
    allowlist asserting what each one brackets would be twenty claims this
    change did not verify. An allowlist of unverified reasons is the rubber
    stamp the #861c audit's dead-entry rule exists to prevent; it is filed
    rather than smuggled in here.
    """

    def test_the_refill_leg_registers_its_rotation_phases(self):
        import ast
        from pathlib import Path

        import sglang.srt.managers.phase_flip_boot as boot

        tree = ast.parse(Path(boot.__file__).read_text())
        labels = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "explain"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                labels.add(node.args[0].value)
        self.assertIn(
            "weights_refill",
            labels,
            "the refill leg no longer registers its #809/W28 phase breakdown "
            "with the seam census. The census line will fall back to 'NOT "
            "DECOMPOSED' -- honest, but the decomposition exists in this very "
            "process and would again be emitted only to an unreferenced line",
        )

    def test_the_registered_terms_cover_every_rotation_phase(self):
        """A decomposition that silently drops a term reconciles worse and
        blames the remainder. `RotationPhases.accounted_s` is the tree's own
        statement of which terms make up the leg; the registration must carry
        all of them."""
        import ast
        from pathlib import Path

        import sglang.srt.managers.phase_flip_boot as boot
        from sglang.srt.model_executor.rotation_executor import RotationPhases

        src = Path(boot.__file__).read_text()
        tree = ast.parse(src)
        registered = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "explain"
            ):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr.endswith("_s"):
                        registered.add(sub.attr)
        expected = {
            f.name
            for f in RotationPhases.__dataclass_fields__.values()
            if f.name.endswith("_s")
            and f.name not in ("total_s", "gpu_d2h_s", "gpu_h2d_s")
        }
        self.assertEqual(
            expected,
            registered & expected,
            "RotationPhases grew a term the census registration does not carry; "
            "it would land in UNEXPLAINED and read as instrument error rather "
            "than as the named phase it is. Missing: "
            + ", ".join(sorted(expected - registered)),
        )


if __name__ == "__main__":
    unittest.main()
