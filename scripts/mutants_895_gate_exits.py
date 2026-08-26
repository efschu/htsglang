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
    +----+-----------------------------------------------+--------------------+

A2 is the arm #895 exists for. Its probe module fails only when it runs under
an xdist worker, which is the machine-checkable form of "the module's
independence proof does not survive crowding": the gate sees a failure, the
solo re-run does not, and the runner must say so and STILL not report green.

A4 is the arm that keeps A2 honest. A real red and a crowding artefact in one
run must exit with the ordinary red code, or a load-sensitive regression could
be filed under the flake's name.

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
