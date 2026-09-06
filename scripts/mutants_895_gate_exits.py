#!/usr/bin/env python3
"""#895 -- can-fail proof for the partitioned gate's exits, BOTH directions.

The gate is test infrastructure, so its own guards need the treatment it
demands of everything else: each one is provoked and shown to fire, and the
arms that must NOT fire are run too, because a guard that fires on everything
gates nothing either.

Every arm runs the REAL runner (`scripts/gate_tier2_partitioned.py`) against a
throwaway gate path of four-line probe modules and a table written to match.
Nothing in `test/` or `python/` is touched, and the probe path is removed
whether the arms pass or fail.

    +----+-----------------------------------------------+--------------------+
    | A0 | everything green                              | rc 0, no #895 block|
    | A1 | unrecorded failure, parallel lane, real       | rc 1, GENUINE      |
    | A2 | unrecorded failure, parallel lane, xdist-only | rc 6, NOT REPROD.  |
    | A3 | recorded failure that does not appear         | rc 4, MISSING      |
    | A4 | A1 and A2 together                            | rc 1 -- red wins   |
    | A5 | a lane holding modules that collect nothing   | rc 3, BROKEN       |
    | A6 | a verdict the runner does not know            | --verify rc 1      |
    | A7 | serial lane: fails in company, passes alone   | rc 6, NOT REPROD.  |
    | A8 | serial lane: an ordinary red                  | rc 1, GENUINE      |
    | A9 | the solo re-run itself cannot answer          | rc 3, INCONCLUSIVE |
    |A10 | the solo re-run collects nothing but parses   | rc 3, INCONCLUSIVE |
    |A11 | TWO modules that cannot be imported at all    | both UNCOLLECTABLE |
    |A12 | the same census, on an INTERRUPTED lane       | lane-side rc 3     |
    |A13 | --gate-path with no --table beside it        | rc 2, REFUSED      |
    |A14 | a gate path that holds no test module        | rc 2, REFUSED      |
    |A15 | a broken lane BEFORE a healthy one            | rc 3, still refuses|
    +----+-----------------------------------------------+--------------------+

A2 is the arm #895 exists for. Its probe module fails only when it runs under
an xdist worker, which is the machine-checkable form of "the module's
independence proof does not survive crowding": the gate sees a failure, the
solo re-run does not, and the runner must say so and STILL not report green.

A4 is the arm that keeps A2 honest. A real red and a crowding artefact in one
run must exit with the ordinary red code, or a load-sensitive regression could
be filed under the flake's name.

A11 is the arm for the runner's fourth numbered refusal (#1207), and it is
END TO END on purpose: the DID NOT RUN census is a print inside ``main()``, so
a unit arm on ``uncollectable_modules`` proves the list and not the report, and
deleting the block that prints it would still leave that arm green.  It stages
TWO uncollectable modules because a one-module census cannot distinguish a
counted denominator from a constant ``1``, nor a loop over the list from a
print of its first name -- and the real gate run this refusal was written for
had four such modules, not one.

A15 is the arm for the FOLD those lane verdicts feed, which is a separate term
from any of them.  Every other arm here stages its broken lane last among the
non-empty lanes, so all of them pass equally on a fold that keeps only the last
lane's answer -- and under such a fold a broken wide lane followed by a healthy
serial one exits GREEN, which is this slice's headline refusal going silent.
A15 breaks the FIRST lane and leaves the LAST healthy, so carrying and
forgetting are measurably different runs.

A12 is A11's other half, and the difference is the LANE, not the assertion.
The census earns its keep by standing BEFORE the LANE-SIDE ``VERDICT:
INCONCLUSIVE -- a lane's log is not an answer.`` return, which is the only
path an interrupted lane takes -- and A11's probe modules are PARALLEL, so its
lane runs under xdist workers, is never interrupted, and prints the census on
either side of a move. A12 stages the same two modules SERIAL: workers=0, one
process, pytest stops at ``Interrupted: 1 error during collection``, and the
run reaches that return. That is the run in which the reader's ONLY signal
that a module never started is this banner, so it is the run the arm asserts
on.

A12 is ALSO the only arm at any level that sees ``main()`` still CONSUME
``lane_verdict``: every unit arm calls the function directly and so proves the
function, not the call. It can only carry that weight if it names WHICH of the
runner's two ``VERDICT: INCONCLUSIVE`` returns it reached -- the lane-side one
above, or the #895 one a solo re-run that could not answer prints on a run
whose lanes all voted OK. It names the lane-side sentence in full, plus the
interrupted branch's own note, for that reason.

A13 and A14 are the arm for ``main()``'s ``except ScopeRefused``, the single
consumer of all three of the runner's scope refusals. The hazard is the one
``resolve_scope``'s docstring names: a ``--gate-path`` without its ``--table``
answered from the DEFAULT table is a real verdict, from the wrong document,
about modules that were never measured -- and it exits 1 like any other verify,
so nothing downstream can tell it from an answer. No other arm here reaches
exit code 2. Both arms pass ``--verify`` so that a refusal turned back into a
guess terminates instead of gating the whole default scope.

Run:  CUDA_VISIBLE_DEVICES="" <venv>/bin/python3 scripts/mutants_895_gate_exits.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "gate_tier2_partitioned.py"
PROBE_DIR = ROOT / ".gate895_probe"
PROBE_REL = PROBE_DIR.relative_to(ROOT).as_posix()
PY = os.environ.get("GATE_PY", "/spinning/htsglang-gpu/.venv/bin/python3")

MODULES = {
    # a module that simply passes -- present in every arm so that no arm is
    # decided by an empty run
    "test_p_ok.py": """
import unittest


class Ok(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(True)
""",
    # fails everywhere, including alone: the GENUINE case
    "test_p_real_red.py": """
import unittest


class RealRed(unittest.TestCase):
    def test_fails_everywhere(self):
        self.assertEqual(1, 2)
""",
    # fails ONLY under an xdist worker: the crowding case, made deterministic
    "test_p_crowd_only.py": """
import os
import unittest


class CrowdOnly(unittest.TestCase):
    def test_fails_only_in_a_worker(self):
        if os.environ.get("PYTEST_XDIST_WORKER"):
            self.fail("stand-in for a deadline missed under crowding")
""",
    # passes; the table will claim it fails, which is the false-green check
    "test_p_recorded.py": """
import unittest


class Recorded(unittest.TestCase):
    def test_the_table_says_this_one_fails(self):
        self.assertTrue(True)
""",
    # a test module with no tests in it
    "test_p_no_tests.py": """
# deliberately empty: a lane that collects nothing must not read as green
""",
    # fails in the lane, and its re-run dies before pytest can write a summary
    "test_p_dies_alone.py": """
import os
import unittest

if not os.environ.get("PYTEST_XDIST_WORKER"):
    # Stand-in for a re-run that does not survive to a summary line: a hard
    # crash, an OOM kill, a C-level abort. The absence of the failing name in
    # such a log must never be read as "it passed alone".
    os._exit(0)


class DiesAlone(unittest.TestCase):
    def test_fails_in_the_lane(self):
        self.fail("stand-in for a lane failure whose re-run cannot answer")
""",
    # fails in the lane, and cannot even be COLLECTED alone: the re-run's log
    # parses cleanly and says "1 error", which is an answer to a different
    # question than the one asked
    "test_p_uncollectable_alone.py": """
import os
import unittest

if not os.environ.get("PYTEST_XDIST_WORKER"):
    raise ImportError("stand-in for a re-run that cannot collect the module")


class UncollectableAlone(unittest.TestCase):
    def test_fails_in_the_lane(self):
        self.fail("stand-in for a lane failure whose re-run never runs it")
""",
    # cannot be imported in ANY process: the lane names it under DID NOT RUN
    # and not one of its tests ever runs, in either the lane or the re-run
    "test_p_uncollectable.py": """
raise ImportError("stand-in for a module the gate path holds but pytest "
                  "cannot import")
""",
    # A SECOND module uncollectable in ANY process, and the reason it is a
    # separate module rather than a second turn for `test_p_uncollectable_alone.py`:
    # that one raises only OUTSIDE an xdist worker, so in a PARALLEL lane it
    # imports, runs and merely FAILS -- measured, it leaves the census at 1.
    # The hazard a second genuinely-uncollectable module covers is that a
    # census of exactly one cannot tell a counted denominator from a pinned
    # `1`, nor a loop over the whole list from one that prints its first entry.
    "test_p_uncollectable_two.py": """
raise ImportError("stand-in for a SECOND module the gate path holds but "
                  "pytest cannot import")
""",
    # The serial lane's shape, in two modules. The marker is PROCESS state,
    # not a file: the serial lane runs one process in one order, so the second
    # module sees it, and a solo re-run of the second module alone -- a fresh
    # process, the first module never imported -- does not. A file on disk
    # would survive into the re-run and prove nothing.
    #
    # Sorted order is the run order, so the leaver must sort before the needer:
    # "leaves" < "needs".
    "test_p_leaves_state.py": """
import os
import unittest


class LeavesState(unittest.TestCase):
    def test_leaves_state_behind(self):
        os.environ["P895_NEIGHBOUR_RAN"] = "1"
""",
    "test_p_needs_state.py": """
import os
import unittest


class NeedsState(unittest.TestCase):
    def test_fails_only_after_the_neighbour(self):
        if os.environ.get("P895_NEIGHBOUR_RAN"):
            self.fail("stand-in for a failure that needs a neighbour's state")
""",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def stage(names: list[str]) -> None:
    """Put EXACTLY these probe modules in the gate path, and nothing else.

    The runner gates every ``test_*.py`` it finds in the path, whether the
    table lists it or not -- an unlisted module is demoted to the serial lane
    and run there. So an arm that leaves a stray red module lying about is
    measuring the stray, not the arm.
    """
    for p in PROBE_DIR.glob("test_*.py"):
        p.unlink()
    for name in names:
        (PROBE_DIR / name).write_text(MODULES[name].lstrip())


def write_table(path: Path, rows: list[tuple[str, str, str, list[str]]]) -> None:
    with path.open("w") as f:
        f.write("# probe table, #895 can-fail arms\n")
        f.write("#module\tverdict\treason\tsha256\tref_failures\n")
        for mod, verdict, reason, ref in rows:
            h = sha(ROOT / mod)
            f.write(f"{mod}\t{verdict}\t{reason}\t{h}\t{','.join(ref)}\n")


def run_gate(table: Path, extra: list[str] | None = None) -> tuple[int, str]:
    cmd = [PY, str(RUNNER), "--table", str(table), "--gate-path", PROBE_REL,
           "-n", "2", "--narrow", "2", "--outdir", str(PROBE_DIR / "out")]
    cmd += extra or []
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def run_raw(argv: list[str]) -> tuple[int, str]:
    """The runner with EXACTLY these arguments, plus an outdir.

    ``run_gate`` always supplies a matched ``--gate-path``/``--table`` pair,
    which is precisely the shape the scope refusals cannot fire on, so the arms
    that provoke them build the command line themselves.
    """
    cmd = [PY, str(RUNNER), *argv, "--outdir", str(PROBE_DIR / "out")]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def mod(name: str) -> str:
    return f"{PROBE_REL}/{name}"


def check(arm: str, want_rc: int, got_rc: int, out: str,
          must: list[str], must_not: list[str]) -> bool:
    ok = got_rc == want_rc
    why = [] if ok else [f"rc {got_rc}, wanted {want_rc}"]
    for s in must:
        if s not in out:
            ok = False
            why.append(f"missing from output: {s!r}")
    for s in must_not:
        if s in out:
            ok = False
            why.append(f"present in output but must not be: {s!r}")
    print(f"  {'PASS' if ok else 'FAIL'}  {arm}  (rc {got_rc})")
    for w in why:
        print(f"        {w}")
    if not ok:
        print("        ---- runner output ----")
        for line in out.splitlines():
            print(f"        | {line}")
    return ok


def main() -> int:
    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR)
    PROBE_DIR.mkdir()
    try:
        table = PROBE_DIR / "table.tsv"
        ok = True

        # A0 -- the arm that must NOT fire. Without it, an always-red guard
        # would pass every arm below.
        stage(["test_p_ok.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A0 all green", 0, rc, out, ["(0 failing test(s))"],
                    ["CLASSIFIED BY A SOLO RE-RUN", "MISSING FAILURE"])

        # A1 -- unrecorded failure in the wide lane that reproduces alone.
        stage(["test_p_ok.py", "test_p_real_red.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_real_red.py"), "PARALLEL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A1 unrecorded + real", 1, rc, out,
                    ["CLASSIFIED BY A SOLO RE-RUN", "GENUINE",
                     "test_p_real_red.py::RealRed::test_fails_everywhere"],
                    ["NOT REPRODUCED", "exit 6"])

        # A2 -- THE #895 ARM: fails in the lane, passes alone.
        stage(["test_p_ok.py", "test_p_crowd_only.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_crowd_only.py"), "PARALLEL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A2 unrecorded + crowding-only", 6, rc, out,
                    ["CLASSIFIED BY A SOLO RE-RUN", "NOT REPRODUCED",
                     "test_p_crowd_only.py::CrowdOnly::test_fails_only_in_a_worker",
                     "1 not reproduced"],
                    ["GENUINE"])

        # A3 -- the direction that already existed, still fires.
        recorded = (f"{PROBE_REL}/test_p_recorded.py::Recorded::"
                    f"test_the_table_says_this_one_fails")
        stage(["test_p_ok.py", "test_p_recorded.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_recorded.py"), "PARALLEL", "probe",
                             [recorded])])
        rc, out = run_gate(table)
        ok &= check("A3 recorded failure vanished", 4, rc, out,
                    ["PARTITION VIOLATION", f"MISSING FAILURE {recorded}"],
                    ["CLASSIFIED BY A SOLO RE-RUN"])

        # A4 -- both at once: the ordinary red must win the exit code.
        stage(["test_p_ok.py", "test_p_real_red.py", "test_p_crowd_only.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_real_red.py"), "PARALLEL", "probe", []),
                            (mod("test_p_crowd_only.py"), "PARALLEL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A4 real red outranks crowding", 1, rc, out,
                    ["GENUINE", "NOT REPRODUCED", "1 genuine, 1 not reproduced"],
                    ["VERDICT: exit 6"])

        # A5 -- a lane that was handed a module and collected nothing.
        stage(["test_p_no_tests.py"])
        write_table(table, [(mod("test_p_no_tests.py"), "PARALLEL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A5 lane collected nothing", 3, rc, out,
                    ["BROKEN", "no test outcome at all", "INCONCLUSIVE"], [])

        # A7 -- the same classification in the SERIAL lane. A module the table
        # sends to the serial lane fails there and passes alone: the gate must
        # say which, exactly as it does for a lane with workers. The stand-in
        # for "needs a neighbour" is the sibling module that leaves a marker
        # file behind, since the serial lane runs one process in one order.
        stage(["test_p_leaves_state.py", "test_p_needs_state.py"])
        write_table(table, [(mod("test_p_leaves_state.py"), "SERIAL", "probe", []),
                            (mod("test_p_needs_state.py"), "SERIAL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A7 serial lane, fails only in company", 6, rc, out,
                    ["CLASSIFIED BY A SOLO RE-RUN", "(lane serial)",
                     "NOT REPRODUCED", "1 not reproduced"],
                    ["GENUINE"])

        # A8 -- and the serial lane's ordinary red still reads as one.
        stage(["test_p_real_red.py"])
        write_table(table, [(mod("test_p_real_red.py"), "SERIAL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A8 serial lane, real red", 1, rc, out,
                    ["GENUINE"], ["NOT REPRODUCED", "VERDICT: exit 6"])

        # A9 -- the re-run that cannot answer. Reading its silence as "passes
        # alone" would turn a dead re-run into an alibi.
        stage(["test_p_ok.py", "test_p_dies_alone.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_dies_alone.py"), "PARALLEL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A9 re-run cannot answer", 3, rc, out,
                    ["RERUN INCONCLUSIVE", "1 inconclusive",
                     "VERDICT: INCONCLUSIVE"],
                    ["NOT REPRODUCED", "GENUINE"])

        # A10 -- the re-run whose log is perfectly well-formed and answers a
        # different question: pytest could not collect the module, said so, and
        # exited 2. The failing name is absent from that log for a reason that
        # has nothing to do with passing.
        stage(["test_p_ok.py", "test_p_uncollectable_alone.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_uncollectable_alone.py"), "PARALLEL",
                             "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A10 re-run could not collect", 3, rc, out,
                    ["RERUN INCONCLUSIVE", "pytest rc 2", "1 inconclusive"],
                    ["NOT REPRODUCED", "GENUINE"])

        # A11 -- the DID NOT RUN census. A module that cannot be imported ran
        # NOTHING, and the lane it sat in still tallies and still votes OK, so
        # nothing else in this harness or in the gate's own suite can see the
        # census disappear. The assertion is on the REPORT, because the report
        # is the whole mechanism: the reader's only signal that a module of the
        # gate's scope never started is its name under this banner. `must_not`
        # keeps it honest in the other direction -- the module must not be
        # classified as a passing-alone flake, since it never ran alone either.
        #
        # TWO uncollectable modules, not one, and the hazard is the census's
        # own arithmetic: on a one-module census the printed count and a
        # hard-coded `1` are the same string, and a loop over the list and a
        # print of its first entry are the same output -- so a census pinned to
        # one module would report the branch's own full gate run of 2026-09-06,
        # whose wide lane held FOUR banner modules, as a single name. Both the
        # denominator and every name are asserted for that reason.
        stage(["test_p_ok.py", "test_p_uncollectable.py",
               "test_p_uncollectable_two.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", []),
                            (mod("test_p_uncollectable.py"), "PARALLEL",
                             "probe", []),
                            (mod("test_p_uncollectable_two.py"), "PARALLEL",
                             "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A11 uncollectable modules named, all of them", 3, rc, out,
                    ["=== DID NOT RUN: 2 module(s) could not be collected ===",
                     f"UNCOLLECTABLE {PROBE_REL}/test_p_uncollectable.py",
                     f"UNCOLLECTABLE {PROBE_REL}/test_p_uncollectable_two.py",
                     "RERUN INCONCLUSIVE"],
                    ["NOT REPRODUCED", "GENUINE"])

        # A12 -- the census on the run that needs it: an INTERRUPTED lane,
        # AND the only arm that sees `main()` consume `lane_verdict`. Both
        # modules go SERIAL, so the lane runs with workers=0 in one process and
        # pytest stops at `Interrupted: 1 error during collection` -- the exact
        # #1207 shape, where every count the runner has still tallies and the
        # name of the module that never started is the only thing left to say.
        #
        # THE HAZARD THE EXPECTED STRINGS ARE SHAPED AGAINST: the runner prints
        # `VERDICT: INCONCLUSIVE` from TWO returns, and only one of them is
        # #1207's. The lane-side return is reached ONLY through `lane_verdict`;
        # the #895 return is reached on a run whose lanes all voted OK, because
        # the solo re-run of an uncollectable module cannot answer either. So
        # this arm names the lane-side sentence in full and the interrupted
        # branch's own note WITH its arithmetic (2 modules staged, 1
        # uncollectable, so 1 never ran). Asserting the bare substring instead
        # was measured to be satisfied by the #895 return, which leaves
        # `main()` free to go back to the pre-#1207 inline tally -- deleting
        # both refusals this slice added -- with every arm still green.
        stage(["test_p_ok.py", "test_p_uncollectable.py"])
        write_table(table, [(mod("test_p_ok.py"), "SERIAL", "probe", []),
                            (mod("test_p_uncollectable.py"), "SERIAL",
                             "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A12 census speaks on an INTERRUPTED run", 3, rc, out,
                    ["=== DID NOT RUN: 1 module(s) could not be collected ===",
                     f"UNCOLLECTABLE {PROBE_REL}/test_p_uncollectable.py",
                     "pytest was INTERRUPTED during collection, so the 1 other "
                     "module(s) this lane holds never ran",
                     "VERDICT: INCONCLUSIVE -- a lane's log is not an answer."],
                    ["NOT REPRODUCED", "GENUINE"])

        # A15 -- the ACCUMULATION of the lane verdicts, which is a different
        # term from the verdict itself. `main()` folds one `ok` per lane into
        # a single `broken`, and every arm above stages its broken lane LAST
        # among the non-empty lanes, so all of them are equally satisfied by a
        # fold that simply keeps the last lane's answer. The hazard that leaves
        # open is the whole slice's headline refusal going silent: a wide lane
        # whose log is not an answer, followed by any lane that voted OK, would
        # be reported with the OK lane's verdict and the run would exit green.
        #
        # So this arm breaks the FIRST lane and leaves the LAST one healthy:
        # the empty module goes PARALLEL (wide, no test outcome at all) and the
        # green one goes SERIAL. The lanes are visited wide, narrow, serial, so
        # a fold that forgets is measurably not the same as one that carries.
        stage(["test_p_no_tests.py", "test_p_ok.py"])
        write_table(table, [(mod("test_p_no_tests.py"), "PARALLEL", "probe", []),
                            (mod("test_p_ok.py"), "SERIAL", "probe", [])])
        rc, out = run_gate(table)
        ok &= check("A15 a broken lane before a healthy one still refuses",
                    3, rc, out,
                    ["no test outcome at all",
                     "VERDICT: INCONCLUSIVE -- a lane's log is not an answer."],
                    ["(0 failing test(s))"])

        # A13 -- a --gate-path with no --table. Answering it from the default
        # table would hand every module of the named path a verdict measured on
        # another path, and exit 1 like any ordinary verify. Refused by name.
        stage(["test_p_ok.py"])
        write_table(table, [(mod("test_p_ok.py"), "PARALLEL", "probe", [])])
        rc, out = run_raw(["--gate-path", PROBE_REL, "--verify"])
        ok &= check("A13 gate path without its table refused", 2, rc, out,
                    ["REFUSED:", "--gate-path given 1 time(s) and --table "
                     "given 0 time(s)"],
                    ["# verify"])

        # A14 -- the same consumer, reached by the other raise: a gate path
        # that holds no test module at all. A renamed directory glob-matches
        # nothing and raises nothing on its own, so the gate would silently
        # narrow to the paths that still exist and report that as the full run.
        empty = PROBE_DIR / "empty"
        empty.mkdir(exist_ok=True)
        rc, out = run_raw(["--gate-path", empty.relative_to(ROOT).as_posix(),
                           "--table", str(table), "--verify"])
        ok &= check("A14 gate path that gates nothing refused", 2, rc, out,
                    ["REFUSED:", "holds 0 test_*.py module(s)"],
                    ["# verify"])

        # A6 -- a verdict the runner does not know is named, not swallowed.
        stage(["test_p_ok.py"])
        write_table(table, [(mod("test_p_ok.py"), "FAST", "invented verdict", [])])
        rc, out = run_gate(table, ["--verify"])
        ok &= check("A6 unknown verdict named", 1, rc, out,
                    ["unknown verdict 'FAST'", "VERIFY FAILED"], [])

        print("\nALL ARMS PASS" if ok else "\nARMS FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(PROBE_DIR, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
