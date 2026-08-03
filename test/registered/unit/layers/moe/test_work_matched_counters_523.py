# SPDX-License-Identifier: Apache-2.0
"""The work-matched counter rule, as code (#523; the rule was written in #482).

#482 canonised the rule in prose: *comparisons count only equal work done, and
the final work-matched dump revision replaces the pre-teardown snapshot as the
`read_arm` rule.* It changed no code. What the code did was print each arm's
work point and ask the reader to hold two printouts side by side -- which is the
check the #439 windows skipped twice, publishing 1.5028x and 1.496x for a point
that is 1.4304x. This file pins the rule where it now lives: in the only path
that produces a cross-arm number.

Hermetic, CPU-only, no card. The inputs are the #439 green-corridor window's own
dumps; see PROVENANCE.md next to the fixtures for what is verbatim and what is
derived.

Three things are pinned:

1. **The gate.** A comparison of two arms that did not do the same work is
   REFUSED by name, with a non-zero exit and no number printed. Every refusal
   reason has its own case, because they are different defects.
2. **The falsifier.** The same mismatched input, with the gate disarmed,
   yields a number -- and that number is the withdrawn 1.5028x, to four
   decimals. That is the defect this ticket removes, reproduced on real bytes.
3. **The off-by-one on the expert extents** (115/71/70 against 116/72/71),
   which ARM3_COMPUTE.md left unreconciled. It is neither a warmup discard nor
   a teardown boundary: it is the #82 zero padding expert. Both readings are
   pinned so that neither can be "corrected" into the other.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
from contextlib import redirect_stdout

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..")
)
ARM_TOOLS = os.path.join(REPO_ROOT, "scripts", "dev", "394_s2_proof")
FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "work_matched_523"
)

sys.path.insert(0, ARM_TOOLS)

import read_arm  # noqa: E402

# The links the #439 preflight measured on this rig, per rank. There is no
# default for these anywhere in the tool: a link vector is a measurement.
LINKS = [14.42, 6.45, 13.41]


def fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    assert os.path.isdir(path), f"missing fixture {name}"
    return path


# ---------------------------------------------------------------------------
# 1. the work point of one arm
# ---------------------------------------------------------------------------


class TestTheWorkPointOfOneArm:
    def test_the_final_revision_has_one_work_point_for_all_ranks(self):
        entries = read_arm.load(_p(fixture("green_final")), "equal")
        work = read_arm.final_work_point("equal", entries)
        assert work == {
            "tokens": 163486,
            "forwards": 155359,
            "activations": 980916,
        }

    def test_the_pre_teardown_revision_announces_itself_through_its_ranks(self):
        """The 45 s timer skew is visible WITHIN one arm, before any A/B."""
        entries = read_arm.load(_p(fixture("green_preteardown")), "equal")
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.final_work_point("equal", entries)
        assert excinfo.value.reason == "non-final-revision"
        assert "45 s timer" in excinfo.value.message

    def test_a_dump_without_the_counters_is_refused_rather_than_assumed(self):
        entries = read_arm.load(_p(fixture("legacy_nowork")), "equal")
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.final_work_point("equal", entries)
        assert excinfo.value.reason == "missing-counter"

    def test_the_single_arm_readout_still_prints_the_work_line(self):
        """The single-arm output format is what every read_*.txt artifact is."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            read_arm.read_single(_p(fixture("green_final")), "equal")
        out = buffer.getvalue()
        assert (
            "work point of this arm: tokens=163486 forwards=155359 "
            "activations=980916" in out
        )
        assert "work=tokens=163486/forwards=155359/activations=980916" in out

    def test_the_single_arm_readout_never_produces_a_cross_arm_number(self):
        """A ratio is not reachable from the ungated path, by construction."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            read_arm.read_single(_p(fixture("green_final")), "compute")
        out = buffer.getvalue()
        assert "speedup" not in out
        assert "MATCHED" not in out
        # No ratio in any shape: the ungated path has nothing to divide by.
        assert re.search(r"\d+\.\d+\s*x", out) is None
        assert "--against" in out


# ---------------------------------------------------------------------------
# 2. the gate
# ---------------------------------------------------------------------------


class TestTheComparisonGate:
    def test_the_work_matched_pair_passes_and_reproduces_the_published_point(self):
        result = read_arm.compare(
            _p(fixture("green_final")), "equal", "compute", LINKS
        )
        # RESULTS.md, "Same table on the work-matched final revision".
        assert [round(v, 1) for v in result["h2d_delta_pct"]] == [-0.4, -41.0, 32.1]
        assert round(result["group_h2d_delta_pct"], 1) == -2.3
        assert result["clock_rank_a"] == 1  # the x4 3080
        assert result["clock_rank_b"] == 0  # the 5090: the clock moved
        assert round(result["clock_s_a"], 1) == 199.3
        assert round(result["clock_s_b"], 1) == 139.3
        # 1.4307x as published is that pair of ONE-DECIMAL seconds divided by
        # each other (199.3 / 139.3). From the full-precision bytes the same
        # dumps give 1.4304x. The 0.02 % between the two readings is a quoting
        # artifact, 20x under the window's own 0.424 % A-vs-A floor -- but it
        # is a second reading, so it is pinned rather than rounded away.
        assert round(result["speedup"], 4) == 1.4304
        assert round(result["clock_s_a"], 1) / round(result["clock_s_b"], 1) == pytest.approx(
            1.4307, abs=0.0001
        )

    def test_the_work_gap_between_two_arms_is_refused_by_name(self):
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.compare(_p(fixture("window_gap")), "equal", "compute", LINKS)
        refusal = excinfo.value
        assert refusal.reason == "work-mismatch"
        # The refusal NAMES the counters and the numbers, per CLAUDE.md's
        # fail-fast rule: a reader must not have to guess what was unequal.
        assert "tokens: equal=158254 compute=150323" in refusal.message
        assert "5.140 %" in refusal.message
        assert "tolerance 0.500 %" in refusal.message

    def test_the_pre_teardown_pair_is_refused_before_the_cross_arm_check(self):
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.compare(
                _p(fixture("green_preteardown")), "equal", "compute", LINKS
            )
        assert excinfo.value.reason == "non-final-revision"

    def test_a_legacy_dump_pair_is_refused(self):
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.compare(_p(fixture("legacy_nowork")), "equal", "compute", LINKS)
        assert excinfo.value.reason == "missing-counter"

    def test_a_rank_count_mismatch_is_refused(self, tmp_path):
        src = fixture("green_final")
        for name in os.listdir(src):
            if name.endswith(".json"):
                shutil.copy(os.path.join(src, name), tmp_path / name)
        os.remove(tmp_path / "expert_stats_compute.tp2ep0.json")
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.compare(tmp_path, "equal", "compute", LINKS)
        assert excinfo.value.reason == "rank-count-mismatch"

    def test_a_link_vector_that_misses_a_rank_is_refused(self):
        with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
            read_arm.compare(
                _p(fixture("green_final")), "equal", "compute", [14.42, 6.45]
            )
        assert excinfo.value.reason == "link-count-mismatch"

    def test_the_verdict_does_not_depend_on_which_arm_is_named_first(self):
        """A gate whose answer depends on argument order is not a gate."""
        forward = read_arm.compare(_p(fixture("green_final")), "equal", "compute")
        backward = read_arm.compare(_p(fixture("green_final")), "compute", "equal")
        assert forward["work_mismatch_pct"] == backward["work_mismatch_pct"]
        for path in ("window_gap", "green_preteardown", "legacy_nowork"):
            reasons = set()
            for a, b in (("equal", "compute"), ("compute", "equal")):
                with pytest.raises(read_arm.WorkMatchRefused) as excinfo:
                    read_arm.compare(_p(fixture(path)), a, b)
                reasons.add(excinfo.value.reason)
            assert len(reasons) == 1, f"{path} refuses differently by order"

    def test_no_number_survives_a_refusal(self):
        """Not a partial result: the refusal path returns nothing at all."""
        result = None
        try:
            result = read_arm.compare(
                _p(fixture("window_gap")), "equal", "compute", LINKS
            )
        except read_arm.WorkMatchRefused:
            pass
        assert result is None


# ---------------------------------------------------------------------------
# 3. the CLI, which is what a window actually runs
# ---------------------------------------------------------------------------


class TestTheCommandLine:
    def test_a_matched_window_exits_zero_and_prints_the_speedup(self, capsys):
        code = read_arm.main(
            [
                fixture("green_final"),
                "equal",
                "--against",
                "compute",
                "--links",
                "14.42,6.45,13.41",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "speedup 1.4304x" in out
        assert "-> MATCHED" in out

    def test_a_mismatched_window_exits_nonzero_and_prints_no_number(self, capsys):
        code = read_arm.main(
            [
                fixture("window_gap"),
                "equal",
                "--against",
                "compute",
                "--links",
                "14.42,6.45,13.41",
            ]
        )
        captured = capsys.readouterr()
        assert code == read_arm.EXIT_REFUSED != 0
        assert captured.out == ""
        assert "REFUSED (work-mismatch)" in captured.err
        assert "speedup" not in captured.err

    def test_the_single_arm_invocation_is_unchanged(self, capsys):
        code = read_arm.main([fixture("green_final"), "equal"])
        out = capsys.readouterr().out
        assert code == 0
        assert out.startswith("== arm 'equal' in ")

    def test_a_comparison_without_links_still_gates_but_reports_no_clock(
        self, capsys
    ):
        """The per-rank H2D delta accumulates too, so the gate is not optional."""
        assert read_arm.main([fixture("window_gap"), "equal", "--against", "compute"]) == (
            read_arm.EXIT_REFUSED
        )
        assert "REFUSED" in capsys.readouterr().err
        assert read_arm.main([fixture("green_final"), "equal", "--against", "compute"]) == 0
        out = capsys.readouterr().out
        assert "no --links given" in out
        assert "speedup" not in out


# ---------------------------------------------------------------------------
# 4. THE FALSIFIER: disarm the gate and the defect comes back, by the number
# ---------------------------------------------------------------------------


class TestTheFalsifier:
    """Feed a work-UNMATCHED comparison in; unfixed it passes, fixed it refuses.

    The "unfixed" arm is not a hypothetical: before #523 there was no cross-arm
    path in this tool at all, so the same two dumps were divided by hand and the
    result was published. Disarming the tolerance reproduces that hand exactly.
    """

    def test_the_disarmed_gate_reproduces_the_withdrawn_number(self):
        result = read_arm.compare(
            _p(fixture("window_gap")),
            "equal",
            "compute",
            LINKS,
            tolerance_pct=100.0,
        )
        # The number ARM3_COMPUTE.md withdrew, to four decimals, off real bytes.
        assert round(result["speedup"], 4) == 1.5028
        # ... and the group total the same revision reported.
        assert round(result["group_h2d_delta_pct"], 1) == -7.2

    def test_the_armed_gate_refuses_the_same_input(self):
        with pytest.raises(read_arm.WorkMatchRefused):
            read_arm.compare(_p(fixture("window_gap")), "equal", "compute", LINKS)

    def test_the_inflation_is_the_work_gap(self):
        """~5 % of unequal work bought ~5 % of speedup. That is the mechanism."""
        matched = read_arm.compare(
            _p(fixture("green_final")), "equal", "compute", LINKS
        )
        inflated = read_arm.compare(
            _p(fixture("window_gap")),
            "equal",
            "compute",
            LINKS,
            tolerance_pct=100.0,
        )
        work_gap = read_arm.work_mismatch_pct(
            read_arm.final_work_point(
                "equal", read_arm.load(_p(fixture("window_gap")), "equal")
            ),
            read_arm.final_work_point(
                "compute", read_arm.load(_p(fixture("window_gap")), "compute")
            ),
        )["tokens"]
        ratio_inflation = (inflated["speedup"] / matched["speedup"] - 1.0) * 100.0
        assert 4.5 < work_gap < 5.5
        assert 4.5 < ratio_inflation < 5.5

    def test_the_default_tolerance_binds_at_the_measured_geometry(self):
        """The default is not decoration: it decides both real cases (#493).

        Below it the real work-matched pair (0.053 %) would still pass and the
        real mismatched pair (5.14 %) would still refuse only if the threshold
        sits between them. Pin both ends so a future edit that raises it to
        "1 %" or drops it to "0.01 %" fails here rather than in a window.
        """
        matched_gap = max(
            read_arm.work_mismatch_pct(
                read_arm.final_work_point(
                    "equal", read_arm.load(_p(fixture("green_final")), "equal")
                ),
                read_arm.final_work_point(
                    "compute", read_arm.load(_p(fixture("green_final")), "compute")
                ),
            ).values()
        )
        mismatched_gap = max(
            read_arm.work_mismatch_pct(
                read_arm.final_work_point(
                    "equal", read_arm.load(_p(fixture("window_gap")), "equal")
                ),
                read_arm.final_work_point(
                    "compute", read_arm.load(_p(fixture("window_gap")), "compute")
                ),
            ).values()
        )
        assert matched_gap < read_arm.DEFAULT_WORK_TOLERANCE_PCT < mismatched_gap
        # and with an order of magnitude to spare on both sides
        assert matched_gap * 5 < read_arm.DEFAULT_WORK_TOLERANCE_PCT
        assert read_arm.DEFAULT_WORK_TOLERANCE_PCT * 5 < mismatched_gap


# ---------------------------------------------------------------------------
# 5. the off-by-one on the expert extents, EXPLAINED rather than overwritten
# ---------------------------------------------------------------------------


class TestTheExpertExtentOffByOne:
    """115/71/70 against 116/72/71 -- two readings of one partition.

    ARM3_COMPUTE.md's repaired-recipe table predicted base extents 115/71/70
    and installed extents 115/56/85; the green window's Gate 3 measured
    116/72/71 and 116/57/86 and the file recorded the delta as "not
    reconciled". It is not a measurement error, not a warmup discard and not a
    teardown boundary. It is the #82 zero padding expert: foreign top-k ids
    remap onto it, it is resident on every rank, and the LOGGED extent counts
    it (`expert_compute_placement.py:708-709`, `partition_units(...)[rank] + 1`,
    documented at :681-682) while the predicted table counted real experts
    only. Hence 256 in one reading and 259 = 256 + 3 in the other.

    Both readings are pinned. Either one alone is a trap: correcting the
    prediction to 116/72/71 would make the extents stop summing to
    `num_experts`, and correcting the measurement to 115/71/70 would make the
    resident-mass arithmetic that consumes it drop the pad slot.
    """

    BASE_PLAN = [30407, 18680, 18680]
    INSTALLED = [213, 104, 157]
    NUM_EXPERTS = 256

    def test_the_predicted_extents_are_the_real_experts(self):
        from sglang.srt.distributed.utils import partition_units

        base = partition_units(self.NUM_EXPERTS, self.BASE_PLAN)
        installed = partition_units(self.NUM_EXPERTS, self.INSTALLED)
        assert base == [115, 71, 70]
        assert installed == [115, 56, 85]
        assert sum(base) == sum(installed) == self.NUM_EXPERTS

    def test_the_measured_extents_are_the_same_partition_plus_the_pad(self):
        from sglang.srt.distributed.utils import partition_units

        base = [n + 1 for n in partition_units(self.NUM_EXPERTS, self.BASE_PLAN)]
        installed = [n + 1 for n in partition_units(self.NUM_EXPERTS, self.INSTALLED)]
        # Gate 3 of 2026-08-03_439_green/RESULTS.md, verbatim.
        assert base == [116, 72, 71]
        assert installed == [116, 57, 86]
        assert sum(base) == sum(installed) == self.NUM_EXPERTS + 3

    def test_the_delta_is_exactly_one_pad_slot_per_rank(self):
        from sglang.srt.distributed.utils import partition_units

        for plan in (self.BASE_PLAN, self.INSTALLED):
            raw = partition_units(self.NUM_EXPERTS, plan)
            padded = [n + 1 for n in raw]
            assert [p - r for p, r in zip(padded, raw)] == [1, 1, 1]

    def test_the_residency_correction_reads_the_padded_extent(self):
        """The +1 is load-bearing, not cosmetic: it sizes the resident mass.

        `resident_fraction_held_at_base_plan` builds `e_base` / `e_local` from
        `partition_units(...) + 1` because the pad slot is resident on every
        rank too. This pins the comment at :681-682 as the testable claim it
        is (CLAUDE.md: an invariant asserted in a comment is a test or a bug
        report).
        """
        import inspect

        from sglang.srt.layers.moe import expert_compute_placement as ecp

        source = inspect.getsource(ecp.resident_fraction_held_at_base_plan)
        assert "partition_units(int(num_experts), list(base))[rank] + 1" in source
        assert "partition_units(int(num_experts), list(installed))[rank] + 1" in source


# ---------------------------------------------------------------------------
# 6. the OTHER place a non-work-matched comparison used to pass silently
# ---------------------------------------------------------------------------


class TestTheS12WindowBasis:
    """s12 puts two arms side by side; what may differ silently is the window.

    The #482 rule binds on ACCUMULATING counters. s12 publishes medians and
    self-normalised shares, which are intensive, so an arm with more batches
    does not bias them -- that part of the inventory is a NEGATIVE finding and
    is stated as such in ``fenster_basis_pruefen``'s docstring rather than
    fixed. What is NOT covered is the window basis: ``punkt_fenster`` bounds
    the point by a request count from punkte.jsonl and falls through to 0 --
    "every large batch in the log, warmup included" -- when the point is
    missing. Two arms then aggregate different phases and nothing says so.
    """

    def test_equal_bases_are_silent(self):
        s12 = _s12()
        assert s12.fenster_basis_pruefen({"bar1:1": 12, "grundlinie:1": 12}) == []

    def test_a_differing_basis_is_named(self):
        s12 = _s12()
        warnungen = s12.fenster_basis_pruefen({"bar1:8": 96, "grundlinie:8": 64})
        assert len(warnungen) == 1
        assert "sessions=8" in warnungen[0]
        assert "bar1=96" in warnungen[0] and "grundlinie=64" in warnungen[0]
        assert "DIFFERENT window bases" in warnungen[0]

    def test_the_silent_fallback_is_the_case_that_is_named_loudest(self):
        """requests=0 keeps the warmup; that is the defect, not a default."""
        s12 = _s12()
        mixed = s12.fenster_basis_pruefen({"bar1:1": 0, "grundlinie:1": 12})
        assert "whole log incl. warmup" in mixed[0]
        both = s12.fenster_basis_pruefen({"bar1:1": 0, "grundlinie:1": 0})
        assert "NO window basis for any arm" in both[0]

    def test_a_single_arm_point_is_not_a_comparison(self):
        s12 = _s12()
        assert s12.fenster_basis_pruefen({"bar1:1": 0}) == []

    def test_punkt_fenster_really_keeps_everything_at_zero(self):
        """The claim the warning rests on, pinned rather than assumed."""
        s12 = _s12()
        rows = [{"rang": 0, "new_token": 2048} for _ in range(9)]
        assert len(s12.punkt_fenster(rows, 0)[0]) == 9
        assert len(s12.punkt_fenster(rows, 4)[0]) == 4

    def test_the_payload_carries_the_basis_and_the_warnings(self):
        s12 = _s12()
        payload = s12.auswerten([], {}, 5120, 3)
        assert payload["fenster_basis"] == {}
        assert payload["fenster_basis_warnungen"] == []


def _s12():
    battery = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
    if battery not in sys.path:
        sys.path.insert(0, battery)
    import s12_log_analyse

    return s12_log_analyse


def _p(path):
    import pathlib

    return pathlib.Path(path)


def test_the_fixtures_carry_their_provenance():
    """A fixture whose derivation is not written down is not evidence."""
    doc = os.path.join(FIXTURES, "PROVENANCE.md")
    assert os.path.isfile(doc)
    text = open(doc).read()
    assert "2026-08-03_439_green" in text
    for name in ("green_final", "green_preteardown", "window_gap", "legacy_nowork"):
        assert name in text
        assert os.path.isdir(os.path.join(FIXTURES, name))


def test_green_final_is_byte_identical_to_the_window_totals():
    """The unmodified fixture must stay unmodified: it is the published point."""
    path = os.path.join(FIXTURES, "green_final", "expert_stats_equal.tp0ep0.json")
    totals = json.load(open(path))["totals"]
    assert totals["h2d_bytes"] == 2017174290432
    assert totals["tokens"] == 163486
    assert totals["moe_compute_policy"] == "base-plan"
    assert totals["moe_compute_vector"] == "30407,18680,18680"
