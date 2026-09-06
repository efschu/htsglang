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

# The depth counts from THIS file's home, so it is re-derived whenever the
# module moves: test/registered/unit/managers -> the tree root is four up.
_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
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
    reports eleven ERRORs in its own summary and was NOT interrupted -- xdist
    keeps going -- so a flag that fired on any ERROR line would fire here and
    would gate nothing.

    THE BOUND OF THIS ARM, named because it was once claimed wider than it is:
    the fixture holds ELEVEN ERROR short-summary lines and ZERO ``ERROR
    collecting`` banners, so ``collection_errors`` is empty here and this arm
    cannot see a flag derived from the BANNERS. That direction is a separate
    arm, G0-T-8c below, and it is the one an implementation is likeliest to
    get wrong."""
    res = lib.parse_log(WIDE)
    assert res.interrupted is False
    assert res.error_lines == WIDE_SUMMARY_ERRORS
    assert res.collection_errors == set(), (
        "this arm does not gate the banner direction; G0-T-8c does"
    )


def test_g0_t8c_a_collection_error_without_the_marker_is_not_interrupted(tmp_path):
    """THE DANGER DIRECTION: ``interrupted`` must come from pytest's OWN
    ``Interrupted:`` marker, never from the presence of a collection error.

    THE HAZARD, measured: xdist collects around a bad module and keeps going,
    and the branch's own full gate run of 2026-09-06 held FOUR uncollectable
    modules in the wide lane (the ``*_631`` family, named under DID NOT RUN).
    An ``interrupted`` derived from ``collection_errors`` therefore turns that
    healthy 415-module lane into ``BROKEN ... INTERRUPTED during collection``
    and puts the gate back to ``VERDICT: INCONCLUSIVE`` -- #1207's symptom,
    restored on the very run cited as proof of its repair, with a reason that
    is false. Both terms of the distinction are asserted here, because a flag
    read off the banner is invisible to an arm that only checks the flag."""
    log = _write(
        tmp_path,
        "__ ERROR collecting registered/unit/managers/test_a.py ___\n"
        "ERROR test/registered/unit/managers/test_a.py - ImportError\n"
        "1 error in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.interrupted is False
    assert res.collection_errors == {"registered/unit/managers/test_a.py"}
    assert res.tally_ok, res.tally_note


def test_g0_t8d_a_lane_that_only_quotes_the_marker_is_not_interrupted(tmp_path):
    """THE FLAG'S OTHER DANGER DIRECTION -- its SHAPE, which nothing armed.

    ``interrupted`` is read off pytest's own banner, and the only thing that
    tells that banner from the words ``Interrupted:`` inside a failure MESSAGE
    is the anchoring: bangs on both sides, at the START of the line.

    THE HAZARD, and it is #1207's symptom restored with a reason that is false
    -- the same hazard G0-T-8c names for the flag's OTHER derivation: a
    de-anchored pattern reads this fully healthy 40-module lane as INTERRUPTED,
    so ``lane_verdict`` votes BROKEN, ``main()`` prints ``VERDICT:
    INCONCLUSIVE -- a lane's log is not an answer.`` and the gate exits 3 on a
    run in which everything collected ran and the tally is perfect. G0-T-8c
    guards the derivation (a collection error must not raise the flag); this
    arm guards the shape.

    The message quotes the banner IN FULL, bangs included, because a message
    quoting only the words is invisible to a widening that keeps the trailing
    ``!+$``."""
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t - AssertionError: "
        "the log must quote the banner "
        "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection "
        "!!!!!!!!!!!!!!!!!!!!\n"
        "1 failed, 400 passed in 60.00s\n",
    )
    res = lib.parse_log(log)
    assert res.interrupted is False
    assert res.collection_errors == set()
    assert res.failures == {"test/registered/unit/managers/test_x.py::T::t"}
    assert res.tally_ok is True, res.tally_note
    ok, note = runner.lane_verdict(res, n_modules=40)
    assert ok is True, f"a lane that merely quotes the marker answered: {note}"


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


def test_g0_t10d_an_uncollectable_module_with_no_error_name_is_refused(tmp_path):
    """THE SECOND REFUSAL BRANCH of `lane_verdict`, which had no arm at all.

    THE HAZARD: the banner and the short summary are two different parts of
    the log, so a run can name a module it could not import and still extract
    no ERROR name for it -- and the TALLY cannot catch that, because 0 names
    against a summary that counts 0 errors is a perfect tally. Asserted here
    (`tally_ok is True`) so the arm is known to gate something the arithmetic
    does not, which is the same false-pass class as the interrupted lane."""
    log = _write(
        tmp_path,
        "____ ERROR collecting registered/unit/mem_cache/test_x.py ____\n"
        "ImportError: no module named nope\n"
        "3 passed in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.tally_ok is True, "the tally alone cannot see this"
    assert res.interrupted is False
    ok, note = runner.lane_verdict(res, n_modules=4)
    assert ok is False
    assert "could not be collected" in note
    assert "registered/unit/mem_cache/test_x.py" in note


def test_g0_t10e_a_short_failed_extraction_is_refused_by_the_lane_check(tmp_path):
    """THE DANGER DIRECTION, and the site that actually votes SS5.3 rule 1.

    THE HAZARD: this slice MOVED the tally refusal out of `main()` into
    `lane_verdict`, and no arm followed it. G0-T-11 and G0-T-11b assert
    `parse_log(...).tally_ok is False` and stop there, so a `lane_verdict` that
    drops the term returns `(True, "extraction mismatch: ...")` -- the runner
    prints `wide   : OK ... names failed=1 vs summary failed=113` and hands the
    SHORT failure set on as the verdict. That is #1207's own headline defect
    (its captured case was `names failed=111 vs summary failed=113`) reported
    as a pass, so the refusal is asserted THROUGH `lane_verdict`, not only
    through the extractor. The must-not-fire arm is G0-T-10b."""
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t\n"
        "3 failed, 4000 passed in 700.00s\n",
    )
    res = lib.parse_log(log)
    assert res.tally_ok is False, "the extractor half of the rule"
    assert res.interrupted is False
    assert res.collection_errors == set()
    ok, note = runner.lane_verdict(res, n_modules=2)
    assert ok is False, f"a lane whose extraction is short is not reportable: {note}"
    assert "names failed=1" in note
    assert "summary failed=3" in note


def test_g0_t10f_a_short_error_extraction_alone_is_refused_by_the_lane_check(tmp_path):
    """THE ERROR HALF of the same site -- the half #1207 actually broke
    (`names error=5 vs summary error=11`), and the half G0-T-10e cannot reach.

    THE HAZARD this arm and no other catches: a `lane_verdict` that re-derives
    the rule from the FAILED terms only (`res.failure_lines != summary failed`)
    passes G0-T-10e and every other arm, while a lane whose ERROR extraction is
    short -- the xdist collection-error shape, four names against eight
    reported -- votes OK. The FAILED terms agree here on purpose, so only the
    ERROR half can produce the refusal."""
    name = "test/registered/unit/managers/test_pp_proxy_stamp_631.py"
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t\n"
        + "".join(f"ERROR {name} - AttributeErr...\n" for _ in range(4))
        + "1 failed, 8 errors in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.failure_lines == res.counts["failed"], "the FAILED half agrees"
    assert res.tally_ok is False
    ok, note = runner.lane_verdict(res, n_modules=2)
    assert ok is False, f"the ERROR half of the rule votes too: {note}"
    assert "names error=4" in note
    assert "summary error=8" in note


def test_g0_t10g_an_interruption_with_no_collect_banner_is_still_refused(tmp_path):
    """THE INTERRUPTED TERM ON ITS OWN -- refusal 4 of the runner's docstring,
    in the shape that carries no collection error at all.

    THE HAZARD: every other arm for this branch stages a lane pytest stopped
    DURING collection, so each one carries ``interrupted`` AND a non-empty
    ``collection_errors`` -- G0-T-10 uses the captured serial fixture, and
    `scripts/mutants_895_gate_exits.py`'s A12 stages an uncollectable module.
    A `lane_verdict` narrowed to ``res.interrupted and res.collection_errors``
    satisfies all of them. pytest ALSO stops early for ``--maxfail``/``-x``,
    which reach this runner through ``args.extra`` at `run_lane`'s
    ``cmd += extra``: the log then holds the marker, no banner, and a PERFECT
    tally -- 1 name failed against 1 summary failed -- so no arithmetic in the
    gate can see that 108 of this lane's 120 modules never started. Under the
    narrowed term the lane votes OK and `main()` folds a green, which is
    *"report a lane that never ran as a lane that answered"* going silent.

    THE BOUND OF THIS ARM, so it is not read as more than it is: it asserts the
    VERDICT, not the note. The note hard-codes *"INTERRUPTED during
    collection"* and the count ``n_modules - len(collection_errors)`` for every
    interruption, and neither is true of this log; pinning that sentence here
    would pin a false one."""
    log = _write(
        tmp_path,
        "FAILED test/registered/unit/managers/test_x.py::T::t\n"
        "!!!!!!!!!!!!!!!!!!!!!! Interrupted: stopping after 1 failures "
        "!!!!!!!!!!!!!!!!!!!!!!\n"
        "1 failed, 12 passed in 3.00s\n",
    )
    res = lib.parse_log(log)
    assert res.interrupted is True
    assert res.collection_errors == set(), "this shape carries no banner"
    assert res.tally_ok is True, "the tally alone cannot see this"
    ok, note = runner.lane_verdict(res, n_modules=120)
    assert ok is False, f"an interrupted lane is not an answer: {note}"


# ---------------------------------------------------------------------------
# G0-T-15  the DID NOT RUN census -- a module that never started is named
# ---------------------------------------------------------------------------
def test_g0_t15_the_uncollectable_census_names_every_lane_and_ignores_the_verdict(
    tmp_path,
):
    """THE DANGER DIRECTION for the runner's fourth numbered refusal, which
    promises the modules that could not be collected are *"named on every run
    whether the lane was interrupted or not -- their absence from a failure set
    is not a pass"* (`gate_tier2_partitioned.py`'s docstring, rule 4).

    THE HAZARD: the census is the only thing keeping that promise, and it is a
    print. Deleting it leaves every suite green. Both terms it must not be
    reduced to are asserted here: it spans ALL lanes, not the interrupted one,
    and it reports even when that lane's own `lane_verdict` is True -- a census
    gated on the verdict would go silent on exactly the healthy-lane case, the
    branch's own full gate run of 2026-09-06 (four uncollectable `*_631`
    modules in a wide lane that tallied OK).

    The must-not-fire arm is G0-T-15b."""
    wide = tmp_path / "wide.log"
    wide.write_text(
        "____ ERROR collecting registered/unit/managers/test_wide_bad.py ____\n"
        "ERROR test/registered/unit/managers/test_wide_bad.py - ImportError\n"
        "1 error, 12 passed in 3.00s\n"
    )
    res = {"wide": lib.parse_log(wide), "serial": lib.parse_log(SERIAL)}

    assert res["wide"].interrupted is False
    ok, _note = runner.lane_verdict(res["wide"], n_modules=8)
    assert ok is True, "the census must not need a BROKEN lane to speak"
    assert res["serial"].interrupted is True

    named = runner.uncollectable_modules(res)
    assert named == [
        "registered/unit/managers/test_quiescence_no_carry_858.py",
        "registered/unit/managers/test_wide_bad.py",
    ], named


def test_g0_t15b_a_run_with_nothing_uncollectable_names_nothing(tmp_path):
    """The arm that must NOT fire: a census that reported on every run would
    print a DID NOT RUN block for a clean gate and name nothing real."""
    clean = tmp_path / "clean.log"
    clean.write_text(
        "FAILED test/registered/unit/managers/test_x.py::T::t\n"
        "1 failed, 3 passed in 1.00s\n"
    )
    res = {"wide": lib.parse_log(clean), "serial": lib.parse_log(WIDE)}
    assert res["serial"].error_lines == WIDE_SUMMARY_ERRORS, (
        "the captured fixture's ERRORs are short-summary lines, not banners"
    )
    assert runner.uncollectable_modules(res) == []


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


def test_g0_t13d_one_module_in_two_tables_is_refused(tmp_path):
    """`merge_tables`' OTHER refusal, which had no arm at all.

    THE HAZARD: two tables carrying a row for the same module are two records
    of one fact with no rule for which wins, and without the refusal the second
    silently overwrites the first -- the module is then handed a verdict proved
    in ANOTHER gate path's run, which is the wrong-document failure `--table`
    pairing exists to prevent. Dict order decides, so nothing in the report
    says which proof was used. Both table paths and the module must be named,
    or the reader cannot tell which two documents disagree.

    The arm that must NOT fire is G0-T-13c."""
    # The two REAL gate paths, so the module-count refusal beside this one
    # cannot be what fires; only the tables are synthetic.
    a, b = (p for p, _ in runner.DEFAULT_SCOPE)
    t1 = tmp_path / "t1.tsv"
    t2 = tmp_path / "t2.tsv"
    shared = f"{a}/test_m.py"
    t1.write_text(f"{shared}\tSERIAL\tproved on path A\tdeadbeefdeadbeef\t\n")
    t2.write_text(f"{shared}\tPARALLEL\tproved on path B\tdeadbeefdeadbeef\t\n")
    try:
        runner.merge_tables([(a, t1), (b, t2)])
    except runner.ScopeRefused as exc:
        message = str(exc)
        assert shared in message
        assert str(t1) in message, message
        assert str(t2) in message, message
    else:
        raise AssertionError("one module in two tables was not refused")


def test_g0_t13e_gate_paths_and_tables_pair_in_the_order_given():
    """`resolve_scope`'s SUCCESS shape, which no arm exercised: G0-T-12 takes
    the both-None default and returns one line earlier, and G0-T-13 provokes
    only the refusals.

    THE HAZARD, with the bound it actually has at this head: a reversed pairing
    survives every other arm, because `merge_tables` unions the tables into one
    flat dict keyed by tree-relative module path and the paths into one flat
    `present` list -- so classification, lane assignment and exit code are
    byte-identical either way. What it changes is the two PROVENANCE lines the
    gate prints, ``# verify {table} against {gate_path}`` and
    ``# scope {gate_path}  <-  {table}``, which are the header of the artifact
    a window is signed against. A header naming the wrong proof for a path is
    the same wrong-document failure the refusals above exist for, one level
    out."""
    scope = runner.resolve_scope(["a/b", "c/d"], ["t1.tsv", "t2.tsv"])
    assert scope == [("a/b", Path("t1.tsv")), ("c/d", Path("t2.tsv"))]


def test_g0_t13c_the_default_scope_paths_all_hold_modules():
    """The arm that must NOT fire, for BOTH of `merge_tables`' refusals: both
    hard-coded directories exist and hold test modules in this tree, and the
    two real tables share no module, so neither refusal above fires on the
    scope the gate actually runs. Without this a refusal that fired on
    everything would pass G0-T-13b and G0-T-13d and gate nothing."""
    table, present = runner.merge_tables(runner.resolve_scope(None, None))
    assert len(present) > 500, len(present)
    assert any(p.startswith("test/registered/unit/mem_cache/") for p in present)
    assert any(p.startswith("test/registered/unit/managers/") for p in present)
    assert table, "the merged table is the thing the refusals guard"


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


def test_g0_t11b_a_broken_error_extraction_is_also_refused(tmp_path):
    """THE ERROR HALF of §5.3 rule 1, which is the half #1207 actually broke
    (`names error=5 vs summary error=11`) and the half G0-T-11 does not reach:
    it asserts only the FAILED terms, so an ERROR comparison replaced by a
    self-comparison (`want_e = res.error_lines`) passes every other arm.

    THE HAZARD that shape carries: both terms would come from the SAME object,
    a guard that cannot fire, and it produces a FALSE PASS on exactly the
    xdist collection-error shape below -- four extracted names against a
    summary counting eight reports, one per worker."""
    name = "test/registered/unit/managers/test_pp_proxy_stamp_631.py"
    log = _write(
        tmp_path,
        "".join(f"ERROR {name} - AttributeErr...\n" for _ in range(4))
        + "8 errors in 1.00s\n",
    )
    res = lib.parse_log(log)
    assert res.tally_ok is False
    assert "names error=4" in res.tally_note
    assert "summary error=8" in res.tally_note
