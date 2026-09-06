"""#1207 -- the desk gate could not deliver a verdict, and this pins why.

THE GATE'S OWN RULE (`gate_partition_lib`'s module docstring, and the build
spec's §5.3): *the number of names pulled out of a log MUST equal the number
the log's own summary reports.*  Measured on the two captured runs of
2026-09-04 (`/tmp/gate_branch_weg1`, `/tmp/gate_base_1116175f6d`) that rule
did not hold, in both directions at once:

    wide lane   names failed=111 vs summary failed=113
                names error=5    vs summary error=11

so every run of the gate ended `VERDICT: INCONCLUSIVE` and no slice of WEG-1
could be measured against it.  Three separate defects produced those two
numbers, and each has its own test below:

  * ``SUBFAILED(abandon="'no_quorum'")`` -- pytest-subtests puts the subtest
    DESCRIPTION straight after the keyword with no space
    (``_pytest/subtests.py:397``, ``f"SUBFAILED{description}"``), so a
    ``SUBFAILED\\s`` pattern misses the line entirely.  Two such lines, and
    113 - 111 = 2.
  * the PRODUCT's own logger writes ``ERROR    sglang.srt...`` in column 0,
    which ``^ERROR\\s+(\\S+)`` reads as a test name.  Seven such lines in the
    captured log.
  * eight collection errors for ONE module (one per xdist worker) collapse
    into one entry under set-dedup, while pytest counts eight.  A tally that
    compares SET SIZES against OCCURRENCE COUNTS is comparing two different
    quantities; 4 real names + 1 invented one = 5 against 11.

The fourth defect has no arithmetic: the serial lane hit ONE uncollectable
module, pytest printed ``Interrupted: 1 error during collection``, and NOT
ONE test of that lane ran -- while the tally passed (0 names failed vs 0
summary failed, 1 name error vs 1 summary error) and the gate reported a
one-element failure set as if the lane had answered.

The fixtures are CAPTURED, not written: ``fixtures/gate_1207_wide_names.txt``
is every name-shaped line of that run's `wide.log` plus its verbatim summary
line, and ``fixtures/gate_1207_serial_collect_error.txt`` is that run's whole
`serial.log` with the ANSI escapes stripped.  ``.txt``, not ``.log``, because
``.gitignore:62`` is ``*.log`` and a captured fixture that cannot be committed
is not evidence -- the same reason
``scripts/fixtures/d2_injector_pingpong_excerpt.txt`` carries that suffix.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gate_partition_lib as lib  # noqa: E402
import gate_tier2_partitioned as runner  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WIDE = FIXTURES / "gate_1207_wide_names.txt"
SERIAL = FIXTURES / "gate_1207_serial_collect_error.txt"

# The captured wide lane's own summary line, verbatim from the fixture's last
# line. These are the numbers every assertion below is measured against.
WIDE_SUMMARY_FAILED = 113
WIDE_SUMMARY_ERRORS = 11
# Seven column-0 product logger lines; they are name-SHAPED and carry no node
# id, which is exactly how they are told apart from a test name.
WIDE_LOGGER_LINES = 7


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "lane.log"
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# G0-T-1  the subtest form the extractor never saw
# ---------------------------------------------------------------------------
def test_g0_t1_subfailed_with_a_parenthesised_description_is_extracted(tmp_path):
    """RED at the parent. Verbatim from `wide.log:6881`."""
    log = _write(
        tmp_path,
        'SUBFAILED(abandon="\'no_quorum\'") '
        "test/registered/unit/managers/test_flip_arm_snapshot_746.py"
        "::TestTheSnapshotCannotOutliveItsFlip::test_abandon_paths_clear_behaviorally\n"
        "1 failed, 2 passed in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.failures == {
        "test/registered/unit/managers/test_flip_arm_snapshot_746.py"
        "::TestTheSnapshotCannotOutliveItsFlip::test_abandon_paths_clear_behaviorally"
    }
    assert res.tally_ok, res.tally_note


# ---------------------------------------------------------------------------
# G0-T-7  the second subtest shape #1207 names
# ---------------------------------------------------------------------------
def test_g0_t7_subfailed_with_a_numeric_parameter_is_extracted(tmp_path):
    """RED at the parent. `SUBFAILED(term=173.6)`, the kwargs form of
    `_pytest/subtests.py:88-97`'s `_sub_test_description`."""
    log = _write(
        tmp_path,
        "SUBFAILED(term=173.6) test/registered/unit/mem_cache/test_x.py::T::t\n"
        "1 failed in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.failures == {"test/registered/unit/mem_cache/test_x.py::T::t"}
    assert res.tally_ok, res.tally_note


# ---------------------------------------------------------------------------
# G0-T-2  the product's logger is not a test name
# ---------------------------------------------------------------------------
def test_g0_t2_a_product_logger_line_is_not_taken_as_a_test_name(tmp_path):
    """RED at the parent. The first 120 characters are verbatim from
    `wide.log:4182`; a logging formatter pads the level to 8 columns, so the
    line begins in column 0 exactly like pytest's own summary."""
    log = _write(
        tmp_path,
        "ERROR    sglang.srt.managers.phase_flip_runtime:phase_flip_runtime.py:8289 "
        "PHASE-FLIP SEAM UNFUNDABLE -- PHASE FLIP STOOD DOWN (pp_to_tp).\n"
        "ERROR test/registered/unit/managers/test_pp_proxy_stamp_631.py - AttributeErr...\n"
        "1 error in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.errors == {
        "test/registered/unit/managers/test_pp_proxy_stamp_631.py"
    }
    assert res.unnamed_lines == 1
    assert res.tally_ok, res.tally_note


# ---------------------------------------------------------------------------
# G0-T-3  the tally counts OCCURRENCES, the verdict keeps a SET
# ---------------------------------------------------------------------------
def test_g0_t3_repeated_name_lines_tally_by_occurrence(tmp_path):
    """RED at the parent. One module, one nodeid, eight collection errors --
    one per xdist worker -- is what `wide.log:89-138` holds, and pytest counts
    eight. Set-dedup counts one, and 1 != 8 is not a broken RUN."""
    name = "test/registered/unit/managers/test_pp_proxy_stamp_631.py"
    log = _write(
        tmp_path,
        "".join(f"ERROR {name} - AttributeErr...\n" for _ in range(8))
        + "8 errors in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.error_lines == 8
    assert res.errors == {name}, "the verdict's arithmetic stays a SET"
    assert res.tally_ok, res.tally_note


# ---------------------------------------------------------------------------
# G0-T-4  the captured run, end to end
# ---------------------------------------------------------------------------
def test_g0_t4_the_captured_wide_lane_tallies(tmp_path):
    """RED at the parent: 111 vs 113 and 5 vs 11. This is the whole reason
    the gate could not answer, measured on the log it could not answer for."""
    res = lib.parse_log(WIDE)
    assert res.counts["failed"] == WIDE_SUMMARY_FAILED
    assert res.counts["error"] == WIDE_SUMMARY_ERRORS
    assert res.failure_lines == WIDE_SUMMARY_FAILED
    assert res.error_lines == WIDE_SUMMARY_ERRORS
    assert res.unnamed_lines == WIDE_LOGGER_LINES
    assert res.tally_ok, res.tally_note


# ---------------------------------------------------------------------------
# G0-T-8  a lane that was cut short during collection says so
# ---------------------------------------------------------------------------
def test_g0_t8_an_interrupted_collection_is_flagged_and_named(tmp_path):
    """RED at the parent. The captured serial lane: ONE module could not be
    imported, pytest printed `Interrupted: 1 error during collection`, and
    every other module of that lane went unrun."""
    res = lib.parse_log(SERIAL)
    assert res.interrupted is True
    assert res.collection_errors == {
        "registered/unit/managers/test_quiescence_no_carry_858.py"
    }


def test_g0_t8b_a_complete_run_is_not_flagged_as_interrupted(tmp_path):
    """The control that makes G0-T-8 mean something: the captured WIDE lane
    carries eleven collection errors and was NOT interrupted -- xdist keeps
    going -- so a flag that fired on any collection error would fire here and
    would gate nothing."""
    res = lib.parse_log(WIDE)
    assert res.interrupted is False


# ---------------------------------------------------------------------------
# G0-T-10  the runner refuses an interrupted lane
# ---------------------------------------------------------------------------
def test_g0_t10_the_runner_calls_an_interrupted_lane_inconclusive(tmp_path):
    """RED at the parent (`lane_verdict` does not exist there).

    THE DANGER DIRECTION, and the reason this test exists: the captured serial
    lane's tally is PERFECT -- 0 names failed against 0 summary failed, 1 name
    error against 1 summary error -- so every arithmetic check the gate had
    passed, and it reported a one-element failure set for a lane in which not
    one test ran."""
    res = lib.parse_log(SERIAL)
    assert res.tally_ok, "the tally alone cannot see this"
    ok, note = runner.lane_verdict(res, n_modules=120)
    assert ok is False
    assert "collection" in note


def test_g0_t10b_a_healthy_lane_still_passes_the_runner_check(tmp_path):
    """The arm that must NOT fire, or the check gates nothing."""
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t\n"
        "1 failed, 3 passed in 1.00s\n",
    )
    ok, note = runner.lane_verdict(lib.parse_log(log), n_modules=2)
    assert ok is True, note


def test_g0_t10c_a_lane_that_collected_nothing_is_still_refused(tmp_path):
    """GREEN PIN on the check that already existed (`A5` of
    `scripts/mutants_895_gate_exits.py`): a lane handed modules whose log
    reports no outcome at all must stay BROKEN."""
    log = _write(tmp_path, "no tests ran in 0.01s\n")
    ok, note = runner.lane_verdict(lib.parse_log(log), n_modules=3)
    assert ok is False
    assert "no test outcome" in note


# ---------------------------------------------------------------------------
# GREEN PINS -- what must not change
# ---------------------------------------------------------------------------
def test_g0_t5_a_plain_failed_line_still_yields_the_node_id(tmp_path):
    """GREEN BEFORE AND AFTER. Includes the ` - message` suffix, so a rule
    that took the LAST node-id-shaped token instead of the first would fail
    here."""
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t - "
        "AssertionError: expected other.py to be read\n"
        "1 failed in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.failures == {"test/registered/unit/managers/test_x.py::T::t"}
    assert res.tally_ok, res.tally_note


def test_g0_t6_the_name_sets_still_dedup(tmp_path):
    """GREEN BEFORE AND AFTER. The gate's verdict is set arithmetic --
    `union - recorded`, `recorded - union` -- and duplicates must not enter
    it. Only the TALLY counts occurrences."""
    name = "test/registered/unit/managers/test_x.py::T::t"
    log = _write(tmp_path, f"FAILED {name}\nFAILED {name}\n2 failed in 1.00s\n")
    res = lib.parse_log(log)
    assert res.failures == {name}
    assert res.all_names == {name}


def test_g0_t9_ansi_in_front_of_the_keyword_is_still_stripped(tmp_path):
    """GREEN BEFORE AND AFTER -- the first trap the module's own docstring
    names."""
    log = _write(
        tmp_path,
        "\x1b[31mFAILED\x1b[0m test/registered/unit/managers/test_x.py::T::t\n"
        "1 failed in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.failures == {"test/registered/unit/managers/test_x.py::T::t"}
    assert res.tally_ok, res.tally_note


# ---------------------------------------------------------------------------
# G0-T-12..14  the gate's SCOPE -- `unit/mem_cache` was outside it
# ---------------------------------------------------------------------------
def test_g0_t12_the_default_scope_covers_managers_and_mem_cache():
    """RED at the parent, where `--gate-path` defaults to `unit/managers`
    alone. Each path is paired with its OWN table: one table per path is what
    `gate_partition_build.py --gate-path/--out` produces, and a merged table
    would give a module of one path a verdict proved in another."""
    scope = runner.resolve_scope(None, None)
    assert [p for p, _ in scope] == [
        "test/registered/unit/managers",
        "test/registered/unit/mem_cache",
    ]
    assert [t.name for _, t in scope] == [
        "gate_partition.tsv",
        "gate_partition_mem_cache.tsv",
    ]
    for _, tbl in scope:
        assert tbl.is_file(), tbl


def test_g0_t13_a_path_without_its_table_is_refused():
    """A `--gate-path` with no `--table` beside it used to silently borrow the
    managers table, which would have handed every mem_cache module the verdict
    `unclassified` from the wrong document. Refused by name instead."""
    for paths, tables in (
        (["a/b"], None),
        (None, ["scripts/gate_partition.tsv"]),
        (["a/b", "c/d"], ["scripts/gate_partition.tsv"]),
    ):
        try:
            runner.resolve_scope(paths, tables)
        except runner.ScopeRefused as exc:
            assert "--gate-path" in str(exc) and "--table" in str(exc)
        else:
            raise AssertionError(f"not refused: {paths!r} {tables!r}")


def test_g0_t13b_a_gate_path_that_holds_no_test_module_is_refused(tmp_path):
    """RED at the parent (`merge_tables` does not exist there).

    THE DANGER DIRECTION FOR THE SCOPE CHANGE: `DEFAULT_SCOPE` hard-codes two
    directory names, and `Path.glob` on a directory that was renamed returns
    an empty list without complaint. The gate would then quietly go back to
    gating ONE path -- re-creating, silently, the exact defect this slice
    repairs. The refusal names the path and the count."""
    empty = tmp_path / "no_tests_here"
    empty.mkdir()
    table = tmp_path / "t.tsv"
    table.write_text("#module\tverdict\treason\tsha256\tref_failures\n")
    try:
        runner.merge_tables([(str(empty), table)])
    except runner.ScopeRefused as exc:
        assert str(empty) in str(exc)
        assert "0 " in str(exc)
    else:
        raise AssertionError("an empty gate path was not refused")


def test_g0_t13c_the_default_scope_paths_all_hold_modules():
    """The arm that must NOT fire: both hard-coded directories exist and hold
    test modules in this tree, so the refusal above gates something real."""
    table, present = runner.merge_tables(runner.resolve_scope(None, None))
    assert len(present) > 500, len(present)
    assert any(p.startswith("test/registered/unit/mem_cache/") for p in present)
    assert any(p.startswith("test/registered/unit/managers/") for p in present)


def test_g0_t14_the_1204_cut_is_inside_the_default_scope():
    """RED at the parent. `test_seam_abstain_into_reduction_1204.py` lives in
    `unit/mem_cache`, which no gate run reached, so the #1204 cut and every
    new WEG-1 test under that directory were invisible to the gate the build
    spec's §5.3 signs against."""
    root = Path(runner.ROOT)
    module = root / "test/registered/unit/mem_cache/test_seam_abstain_into_reduction_1204.py"
    assert module.is_file(), module
    covered = [
        gpath
        for gpath, _ in runner.resolve_scope(None, None)
        if module.is_relative_to(root / gpath)
    ]
    assert covered == ["test/registered/unit/mem_cache"]


def test_g0_t11_a_broken_extraction_is_still_refused(tmp_path):
    """GREEN BEFORE AND AFTER, and it is the gate itself: a log whose names
    do not add up to its summary must stay NOT reportable. A repair that made
    the tally pass unconditionally would delete the rule it was repairing."""
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t\n"
        "3 failed in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.tally_ok is False
    assert "names failed=1" in res.tally_note
    assert "summary failed=3" in res.tally_note
