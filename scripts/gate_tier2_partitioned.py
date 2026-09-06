#!/usr/bin/env python3
"""#868 -- the partitioned tier-2 gate.

THREE LANES, RUN ONE AFTER ANOTHER, ONE VERDICT:

  wide    (PARALLEL)  modules that PROVED they are process-independent, run
                      with many workers;
  narrow  (RANKS)     the same proof, but the module spawns real ranks, so it
                      gets a bounded worker count -- enough parallelism to pay,
                      few enough processes that no rank is starved past the
                      timeout the test is actually measuring;
  serial  (SERIAL)    everything whose proof did not hold, in one process, in
                      one order.

The lanes are SEQUENTIAL on purpose: a wide lane running alongside the narrow
one would re-create exactly the CPU pressure the narrow lane exists to avoid.

``--dist loadfile``, NEVER ``--dist loadscope``
-----------------------------------------------
xdist's ``loadscope`` groups by MODULE for plain test functions but by CLASS
for test methods.  This suite is unittest/TestCase based throughout, so
``loadscope`` splits a single FILE across workers, and a class that inherits
process state from a sibling class in the same file then fails.  That is
measurable on one module in isolation:

    test_chunked_commitment_701.py alone, serial            -> 17 passed
    test_chunked_commitment_701.py alone, -n 2 loadscope    ->  4 failed
    test_chunked_commitment_701.py alone, -n 2 loadfile     -> 17 passed

One module cannot depend on another module that is not there, so those four
failures are the engine's grouping unit and nothing else.  ``loadfile`` groups
by file, which is the unit the solo proof was taken in.  The runner refuses
``loadscope`` rather than trusting the caller to remember.

WHAT THIS RUNNER REFUSES TO DO SILENTLY
---------------------------------------
1. **Run a narrower gate than it claims.**  Every run prints what it left out
   and why, before it prints the result.

2. **Promote an unproven module.**  Not in the table, or bytes no longer
   matching the sha256 its proof was taken against -> serial lane, named in the
   report.  The default for the unknown is the slow, correct lane.

3. **Lose a known failure.**  Each row records the failures that module
   produces in the serial reference.  A recorded failure that does NOT appear
   in this run is a PARTITION VIOLATION: separating a module from the process
   state its failure depended on has turned a real red into a green.  That is
   the one direction repeated parallel runs cannot see, so it is checked
   arithmetically on every run.

4. **Report a lane that never ran as a lane that answered** (#1207).  A single
   unimportable module makes pytest print ``Interrupted: N error(s) during
   collection`` and stop, and the log it leaves behind TALLIES: zero names
   failed against zero summary failed, one name error against one summary
   error.  Measured 2026-09-04: the serial lane died at
   ``test_quiescence_no_carry_858.py`` and the gate printed a one-element
   failure set for a lane in which not one test ran.  Every check the runner
   had was arithmetic, and no arithmetic can see this.  An interrupted lane is
   INCONCLUSIVE, and the modules that could not be collected are named on
   every run whether the lane was interrupted or not -- their absence from a
   failure set is not a pass.

5. **Hand an UNRECORDED failure to the reader unclassified** (#895).  Check 3
   is one-way: it asks only whether a recorded failure came back.  On this tree
   every recorded failure belongs to an EXCLUDED module, so check 3 can never
   fire at the desk and EVERY desk failure is unrecorded.  Such a failure has
   two possible causes, and they are not the same finding:

     * the product is broken -- the ordinary red, and the reason for the gate;
     * the failure needs company.  In a lane with workers that is crowding --
       the admission proof is `solo failure set == serial failure set`, taken
       on a quiet box, and NOTE #868 §2.5 names the class it cannot settle: an
       assert about where concurrent ranks stand at a deadline, whose
       independent variable is the load on the box.  In the serial lane it is a
       neighbour's state, or the same load.

   Guessing between them was the reader's job until #895; now the runner
   re-runs the affected module ALONE, in a fresh process, in the same gate run,
   and prints the verdict.  Solo reproduces -> GENUINE, the gate is red as it
   always was.  Solo does not reproduce -> the failure did not survive
   isolation; that is its OWN exit code (6), never a green.  A gate that
   quietly forgave what it could not explain would forgive a real,
   load-sensitive regression with it.

``--prove`` adds the OTHER direction for acceptance: the union must EQUAL the
reference set, not merely contain it.  It is a separate flag because in normal
gate use an added failure is the gate doing its job, while during acceptance an
added failure means the partition changed the answer.

SCOPE: MORE THAN ONE GATE PATH, EACH WITH ITS OWN TABLE
-------------------------------------------------------
``--gate-path`` and ``--table`` are repeatable and PAIRED in order, and the
default scope is both directories the WEG-1 work lands in::

    test/registered/unit/managers   scripts/gate_partition.tsv
    test/registered/unit/mem_cache  scripts/gate_partition_mem_cache.tsv

``unit/mem_cache`` was outside every gate run until #1207, so the #1204 cut
and every test written under that directory were invisible to a gate the
build spec signs against.  The tables are NOT merged into one file: a table is
the record of ONE ``gate_partition_build.py --gate-path`` measurement, and a
module must never be handed a verdict proved in another path's run.  A
``--gate-path`` without the ``--table`` beside it is refused rather than
silently given the managers table.

EXIT CODES
----------
    0  green            1  failing test(s)
    2  refused: --dist loadscope, or a --gate-path/--table set that does not pair
    3  inconclusive: an extraction, a lane cut short during collection, or a
       solo re-run, could not answer
    4  PARTITION VIOLATION -- a recorded failure vanished (false green)
    5  --prove: the partitioned answer differs from the serial reference
    6  every failure in the run is unrecorded AND did NOT reproduce when its
       module was re-run alone (crowding, coupling, or the box).  6 is
       reserved for the case where that is the ONLY thing wrong: one genuine
       failure anywhere in the run and the ordinary red code wins instead.

CUDA_VISIBLE_DEVICES is forced empty and not overridable from the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition_lib import module_of, parse_log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ.get("GATE_PY", "/spinning/htsglang-gpu/.venv/bin/python3")

#: (gate path, its table). One table per path, in the order they run.
DEFAULT_SCOPE = (
    ("test/registered/unit/managers", "scripts/gate_partition.tsv"),
    ("test/registered/unit/mem_cache", "scripts/gate_partition_mem_cache.tsv"),
)


class ScopeRefused(ValueError):
    """The gate paths and the tables do not pair, or two tables claim one module."""


def resolve_scope(gate_paths, tables) -> list[tuple[str, Path]]:
    """Pair each gate path with the table its verdicts were measured against.

    THE HAZARD: a ``--gate-path`` with no ``--table`` beside it used to fall
    back to the managers table, which would hand every module of the new path
    the verdict ``unclassified: not in the partition table`` -- a real answer,
    from the wrong document, about modules that were never measured. Refused
    by name instead of guessed.

    WHAT THE ORDER CARRIES, and the bound on it, so the refusal's own sentence
    ("a table proved on one path says nothing about a module in another") is
    not read as resting on the pairing: it does not. ``merge_tables`` keys the
    merged verdicts by TREE-RELATIVE module path and unions both paths into one
    flat ``present`` list, so the classification, the lane assignment and the
    exit code are byte-identical under a reversed pairing. What the order
    decides is the two PROVENANCE lines -- ``# verify {table} against
    {gate_path}`` and ``# scope {gate_path}  <-  {table}`` -- which are the
    header of the artifact a window is signed against. A header that names the
    wrong proof for a path is the same wrong-document failure as above, one
    level out.
    """
    if gate_paths is None and tables is None:
        return [(p, ROOT / t) for p, t in DEFAULT_SCOPE]
    gate_paths = list(gate_paths or [])
    tables = list(tables or [])
    if not gate_paths or len(gate_paths) != len(tables):
        raise ScopeRefused(
            f"--gate-path given {len(gate_paths)} time(s) and --table given "
            f"{len(tables)} time(s). Each --gate-path needs the --table its "
            f"verdicts were proved against, in the same order; a table proved "
            f"on one path says nothing about a module in another. Give both, "
            f"or neither for the default scope "
            f"({', '.join(p for p, _ in DEFAULT_SCOPE)})."
        )
    return [(p, Path(t)) for p, t in zip(gate_paths, tables)]


def merge_tables(scope: list[tuple[str, Path]]) -> tuple[dict[str, dict], list[str]]:
    """The merged verdict table and the modules present, over the whole scope.

    Module keys are tree-relative paths, so two paths cannot collide by
    construction -- but two tables naming the same module WOULD be two records
    of one fact with no rule for which wins, so it is refused rather than
    resolved by dict order.

    A path that yields NO module is refused too, and that guard is aimed at
    ``DEFAULT_SCOPE``: it hard-codes two directory names, and ``glob`` on a
    directory that was renamed returns an empty list without complaint. The
    gate would go back to gating one path silently -- the very defect #1207
    repairs, re-created by a rename.
    """
    table: dict[str, dict] = {}
    owner: dict[str, Path] = {}
    present: list[str] = []
    for gate_path, table_path in scope:
        for mod, row in load_table(table_path).items():
            if mod in table:
                raise ScopeRefused(
                    f"{mod} carries a verdict in two tables ({owner[mod]} and "
                    f"{table_path}). One module, one proof, one table."
                )
            table[mod] = row
            owner[mod] = table_path
        here = [p.relative_to(ROOT).as_posix()
                for p in (ROOT / gate_path).glob("test_*.py")]
        if not here:
            raise ScopeRefused(
                f"gate path {gate_path} holds 0 test_*.py module(s) under "
                f"{ROOT}. A gate path that gates nothing is a narrower gate "
                f"reported as the full one; name a path that exists."
            )
        present += here
    return table, sorted(set(present))


def lane_verdict(res, n_modules: int) -> tuple[bool, str]:
    """Is this lane's log an ANSWER at all?  ``(ok, note)``.

    Four ways it is not, and only the last two are arithmetic:

    * pytest was INTERRUPTED during collection. Nothing it had collected ran,
      and the log still tallies -- that is #1207's serial lane, whose perfect
      0-vs-0 and 1-vs-1 covered a lane of 120 modules that never started.
    * a module could not be collected and the extraction found no ERROR name
      for it. The two terms come from different parts of the log (the
      ``ERROR collecting`` banner and the short summary), so this can fire.
    * the lane was handed modules and the log reports no test outcome at all.
      "no tests ran" and "everything passed" are the same log to a count, so
      emptiness is asked about separately.
    * names and summary disagree (#868's own rule, repaired by #1207).
    """
    if res.interrupted:
        named = ", ".join(sorted(res.collection_errors)) or "not named in the log"
        return False, (
            f"pytest was INTERRUPTED during collection, so the "
            f"{n_modules - len(res.collection_errors)} other module(s) this "
            f"lane holds never ran; uncollectable: {named}"
        )
    if res.collection_errors and res.error_lines == 0:
        return False, (
            f"{len(res.collection_errors)} module(s) could not be collected "
            f"({', '.join(sorted(res.collection_errors))}) but the extraction "
            f"found 0 ERROR name(s) in {n_modules} module(s) of lane log"
        )
    ran_something = sum(res.counts.get(k, 0) for k in
                        ("failed", "passed", "skipped", "error", "xfailed",
                         "xpassed", "subtests")) > 0
    if res.collected_nothing or not ran_something:
        return False, (
            f"lane holds {n_modules} module(s) but the log reports no test "
            f"outcome at all -- collected nothing, or the summary was not "
            f"understood"
        )
    if not res.tally_ok:
        return False, res.tally_note
    return True, res.tally_note


def uncollectable_modules(res: dict) -> list[str]:
    """Every module a lane's log says pytest could not import, sorted.

    THE HAZARD this census answers, and why it is INDEPENDENT of the lane
    verdict: a module that could not be imported ran NOTHING, and a reader
    scanning the failure set of an otherwise healthy lane has no way to tell
    "this module passed" from "this module never started". ``lane_verdict``
    cannot stand in for it -- its collection-error branch reads
    ``res.error_lines``, which counts the WHOLE lane's short-summary ERROR
    lines, so it says something about the lane and nothing about the module.
    The census is therefore taken on every run, whatever any lane voted.

    A REPORT, NOT A SECOND VOTE. The vote on these modules already exists and
    is the ERROR name the extraction pulled from the same log: measured on the
    branch's own full gate run of 2026-09-06 (``wide.log``), four banner
    modules against four distinct ERROR names in the failure set, so the gate
    was red for every one of them. Naming them a second time in a verdict
    would be a second record of a fact that already votes -- and the two
    records would not even share a name space, since a banner path is
    rootdir-relative and a node id is tree-relative.
    """
    return sorted({m for r in res.values() for m in r.collection_errors})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_table(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = (line.split("\t") + ["", "", "", "", ""])[:5]
        mod, verdict, reason, h, ref = parts
        rows[mod] = {"verdict": verdict, "reason": reason, "sha": h,
                     "ref": {x for x in ref.split(",") if x}}
    return rows


def run_lane(modules: list[str], out: Path, workers: int, extra: list[str]) -> tuple[int, float]:
    if not modules:
        out.write_text("no tests ran\n")
        return 0, 0.0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""            # forced, not overridable
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "python")    # derived from this file, never typed
    cmd = [PY, "-m", "pytest", *modules, "-q", "-p", "no:randomly",
           "-p", "no:cacheprovider", "--color=no", "-rfE"]
    if workers:
        cmd += ["-n", str(workers), "--dist", "loadfile"]
    cmd += extra
    t0 = time.time()
    with out.open("w") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return rc, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--wide", type=int, default=8, help="workers for the wide lane")
    ap.add_argument("--narrow", type=int, default=4, help="workers for the rank-spawning lane")
    ap.add_argument("--table", action="append", default=None,
                    help="partition table; repeatable, paired IN ORDER with "
                         "--gate-path. Omit both for the default scope.")
    ap.add_argument("--gate-path", action="append", default=None,
                    help="directory of test_*.py to gate; repeatable, and each "
                         "one needs its own --table")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--serial-only", action="store_true",
                    help="canary form: every lane collapsed into one process, one order")
    ap.add_argument("--verify", action="store_true",
                    help="check the table against the tree and exit; runs no tests")
    ap.add_argument("--prove", metavar="REF_LOG", default=None,
                    help="acceptance: require the union to EQUAL this serial "
                         "reference's failure set, in both directions")
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    # The lane WIDTH is a knob; the distribution MODE is not. `loadscope` groups
    # test methods by CLASS, so it splits single files across workers and breaks
    # a class that inherits process state from a sibling class in the same file
    # -- reproducible on ONE module with nothing else in the run. Refused here
    # rather than left to the caller to remember.
    if any("loadscope" in a for a in args.extra):
        print("REFUSED: --dist loadscope splits FILES across workers in this suite "
              "(xdist groups test methods by class, not by module). It produces "
              "failures that no module dependency explains -- reproducible on "
              "test_chunked_commitment_701.py alone. This gate uses --dist "
              "loadfile, which is the unit the solo proof was taken in.",
              file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    outdir = Path(args.outdir or f"/tmp/gate868_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        scope = resolve_scope(args.gate_path, args.table)
        table, present = merge_tables(scope)
    except ScopeRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    wide: list[str] = []
    narrow: list[str] = []
    serial: list[str] = []
    excluded: list[tuple[str, str]] = []
    demoted: list[tuple[str, str]] = []
    expected: dict[str, set[str]] = {}

    for mod in present:
        row = table.get(mod)
        if row is None:
            serial.append(mod)
            demoted.append((mod, "unclassified: not in the partition table"))
            continue
        expected[mod] = row["ref"]
        v = row["verdict"]
        if v == "EXCLUDED":
            excluded.append((mod, row["reason"]))
            continue
        if v in ("PARALLEL", "RANKS") and sha(ROOT / mod) != row["sha"]:
            serial.append(mod)
            demoted.append((mod, "proof stale: module bytes changed since the solo proof"))
            continue
        if v == "PARALLEL":
            wide.append(mod)
        elif v == "RANKS":
            narrow.append(mod)
        else:
            serial.append(mod)
            # SERIAL is a verdict; anything else in that column is a typo or a
            # verdict this runner does not know. Both land in the slow lane --
            # that part was already right -- but silently, which is not.
            if v != "SERIAL":
                demoted.append((mod, f"unknown verdict {v!r}: not one of "
                                     f"PARALLEL/RANKS/SERIAL/EXCLUDED"))

    stale_gone = [m for m in table if m not in present]

    if args.verify:
        for gate_path, table_path in scope:
            print(f"# verify {table_path} against {gate_path}")
        problems = 0
        for mod, reason in demoted:
            print(f"  UNPROVEN  {mod}\n            {reason}")
            problems += 1
        for mod in stale_gone:
            print(f"  MISSING   {mod}  (in table, gone from the tree)")
            problems += 1
        counts: dict[str, int] = {}
        for mod in present:
            k = table[mod]["verdict"] if mod in table else "UNCLASSIFIED"
            counts[k] = counts.get(k, 0) + 1
        print(f"  totals    {counts}  over {len(present)} module(s) in the tree")
        if problems:
            print(f"\nVERIFY FAILED: {problems} module(s) without a valid proof. Re-run the "
                  f"solo probe for them and rebuild with scripts/gate_partition_build.py.")
            return 1
        print("\nVERIFY OK: every module in the gate path carries a verdict, and every "
              "fast-lane verdict matches the bytes it was proved against.")
        return 0

    if args.serial_only:
        serial = sorted(wide + narrow + serial)
        wide, narrow = [], []

    print(f"# gate_tier2_partitioned {stamp}")
    print(f"# tree     {ROOT}")
    print(f"# commit   {subprocess.getoutput('git -C %s rev-parse --short HEAD' % ROOT)}")
    for gate_path, table_path in scope:
        print(f"# scope    {gate_path}  <-  {table_path}")
    print('# hermetic CUDA_VISIBLE_DEVICES="" forced on every lane')
    print(f"# lanes    wide={len(wide)} (-n {args.wide} --dist loadfile) | "
          f"narrow={len(narrow)} (-n {args.narrow} --dist loadfile) | "
          f"serial={len(serial)}")

    print("\n=== NOT IN THIS GATE (report, not a footnote) ===")
    if not excluded:
        print("  (nothing excluded)")
    for mod, reason in excluded:
        print(f"  EXCLUDED  {mod}\n            {reason}")
    if demoted:
        print("--- demoted to the serial lane at run time ---")
        for mod, reason in demoted:
            print(f"  DEMOTED   {mod}\n            {reason}")
    if stale_gone:
        print("--- in the table but no longer in the tree ---")
        for mod in stale_gone:
            print(f"  MISSING   {mod}")
    comp: dict[str, int] = {}
    for mod in serial:
        row = table.get(mod)
        k = row["reason"].split(":", 1)[0] if row else "unclassified"
        comp[k] = comp.get(k, 0) + 1
    print(f"--- serial lane composition: {comp or '{}'}")

    logs = {"wide": outdir / "wide.log", "narrow": outdir / "narrow.log",
            "serial": outdir / "serial.log"}
    rc_w, t_w = run_lane(wide, logs["wide"], args.wide, args.extra)
    rc_n, t_n = run_lane(narrow, logs["narrow"], args.narrow, args.extra)
    rc_s, t_s = run_lane(serial, logs["serial"], 0, args.extra)

    res = {k: parse_log(v) for k, v in logs.items()}

    print("\n=== tally gate (extraction must agree with the summary) ===")
    broken = False
    for name, mods in (("wide", wide), ("narrow", narrow), ("serial", serial)):
        if not mods:
            print(f"  {name:7s}: empty lane, skipped")
            continue
        r = res[name]
        ok, note = lane_verdict(r, len(mods))
        print(f"  {name:7s}: {'OK' if ok else 'BROKEN'}  counts={r.counts} "
              f"name lines={r.failure_lines + r.error_lines} "
              f"distinct={len(r.all_names)} unnamed={r.unnamed_lines} {note}")
        broken = broken or not ok

    # A module that could not be imported ran NOTHING, whatever the lane did
    # afterwards. It is named on every run, before any verdict, so its absence
    # from the failure set below is never read as a pass.
    uncollectable = uncollectable_modules(res)
    if uncollectable:
        print(f"\n=== DID NOT RUN: {len(uncollectable)} module(s) could not be "
              f"collected ===")
        print("  Paths are as pytest's own banner gives them (rootdir-relative).")
        for mod in uncollectable:
            print(f"  UNCOLLECTABLE {mod}")

    if broken:
        print("\nVERDICT: INCONCLUSIVE -- a lane's log is not an answer.")
        return 3

    union = set().union(*(r.all_names for r in res.values()))
    ran = set(wide) | set(narrow) | set(serial)
    recorded = set().union(*[expected[m] for m in ran if m in expected]) if ran else set()

    print("\n=== failure set (the verdict) ===")
    for name in sorted(union):
        print(f"  {name}")
    print(f"  ({len(union)} failing test(s))")

    print("\n=== timing ===")
    print(f"  wide   lane {t_w:8.2f}s  ({len(wide)} modules, -n {args.wide})")
    print(f"  narrow lane {t_n:8.2f}s  ({len(narrow)} modules, -n {args.narrow})")
    print(f"  serial lane {t_s:8.2f}s  ({len(serial)} modules)")
    print(f"  gate total  {t_w + t_n + t_s:8.2f}s")

    rc = 4 if (vanished := sorted(recorded - union)) else 0
    if vanished:
        print("\n=== PARTITION VIOLATION (false green) ===")
        print("  Recorded in the table, absent from this run. A failure that")
        print("  disappears when a module is separated from its neighbours is a")
        print("  FALSE GREEN, and no number of parallel repeats would show it.")
        for name in vanished:
            print(f"  MISSING FAILURE {name}")

    # ---- #895: the other direction of check 3, on every run ----------------
    # Every unrecorded failure, in every lane, is re-run ALONE and classified.
    #
    # The first cut of this only re-ran the lanes that had WORKERS, on the
    # argument that a serial-lane failure has one reading. The gate's own next
    # run falsified that: `test_pp_admission_wraparound_never_blocks.py`
    # failed in the SERIAL lane of two consecutive full runs -- once at load
    # 150 from foreign gates, once on a quiet box -- and passed 3/3 alone, and
    # passed again with the whole serial lane run standalone. So the serial
    # lane hands the reader exactly the same unclassified red, and the reader
    # had exactly the same guess to make.
    #
    # The re-run is a MEASUREMENT in every lane, not a retry, because it always
    # changes the same independent variable: neighbours. Wide/narrow, the
    # neighbours are the other workers; serial, they are the other modules in
    # the one process. Alone in a fresh process is the shape every row's proof
    # was taken in -- comparing against it is #868's admission criterion, run
    # on the failure instead of on the table.
    #
    # NOT REPRODUCED never means forgiven and never means "not real": a defect
    # that needs a neighbour's leaked state is a defect, and the serial lane
    # exists to keep exactly those visible. It means the failure did not
    # survive isolation, which is a different finding from the product being
    # broken, and the gate stays non-green for it either way (exit 6).
    lane_of = {}
    for lane_name, mods_ in (("wide", wide), ("narrow", narrow), ("serial", serial)):
        for m in mods_:
            lane_of[m] = lane_name
    unrecorded = sorted(union - recorded)
    not_reproduced: list[str] = []
    if unrecorded:
        print("\n=== UNRECORDED FAILURE, CLASSIFIED BY A SOLO RE-RUN (#895) ===")
        print("  Not recorded in the table. Either the product is broken, or the")
        print("  failure needs company -- crowding in a lane with workers, a")
        print("  neighbour's state in the serial lane, or the load on the box.")
        print("  Separated by measurement, here, now: each affected module is")
        print("  re-run ALONE in a fresh process, which is the shape its row in")
        print("  the table was proved in. The whole module is re-run, not the")
        print("  single test, because the proof's unit is the module.")
        verdicts: dict[str, str] = {}
        for mod in sorted({module_of(n) for n in unrecorded}):
            log = outdir / f"solo_{Path(mod).stem}.log"
            solo_rc, solo_t = run_lane([mod], log, 0, args.extra)
            solo = parse_log(log)
            names = {n for n in unrecorded if module_of(n) == mod}
            lane = lane_of.get(mod, "?")
            print(f"\n  MODULE {mod}  (lane {lane})")
            print(f"    solo re-run: rc={solo_rc} {solo_t:.2f}s counts={solo.counts} "
                  f"-> {log}")
            # A re-run only answers the question if it RAN. pytest past 1 means
            # it did not (2 interrupted, 3 internal, 4 usage, 5 nothing
            # collected), and a log with no summary or nothing collected means
            # the same. Without this the absence of the failing name would read
            # as "passes alone" -- the most expensive possible misreading, and
            # the one a module that dies during its re-run produces.
            if solo_rc not in (0, 1) or not solo.tally_ok or solo.collected_nothing:
                print(f"    RERUN INCONCLUSIVE -- "
                      f"{solo.tally_note or 'collected nothing'}"
                      f"{'' if solo_rc in (0, 1) else f' (pytest rc {solo_rc})'}")
                for n in sorted(names):
                    verdicts[n] = "INCONCLUSIVE"
                continue
            if lane in ("wide", "narrow"):
                why = ("the table admits this module to a lane with workers, and "
                       "this run disagrees:")
                what = "independence proof stale, or a deadline missed under crowding."
            else:
                why = ("it needs its neighbours or the box it ran on. That is a "
                       "finding about")
                what = ("coupling or load, not about the product -- and not a "
                        "dismissal either.")
            for n in sorted(names):
                if n in solo.all_names:
                    verdicts[n] = "GENUINE"
                    print(f"    GENUINE          {n}")
                    print("                     reproduces alone. This is a real "
                          "failure and the gate is red for it.")
                else:
                    verdicts[n] = "NOT REPRODUCED"
                    print(f"    NOT REPRODUCED   {n}")
                    print(f"                     passes alone -- {why}")
                    print(f"                     {what}")
                    print("                     NOT forgiven -- see exit code 6.")
        not_reproduced = sorted(n for n, v in verdicts.items() if v == "NOT REPRODUCED")
        inconclusive = sorted(n for n, v in verdicts.items() if v == "INCONCLUSIVE")
        genuine = sorted(n for n, v in verdicts.items() if v == "GENUINE")
        print(f"\n  summary: {len(genuine)} genuine, {len(not_reproduced)} not "
              f"reproduced, {len(inconclusive)} inconclusive")
        if inconclusive and rc == 0:
            rc = 3
            print("  VERDICT: INCONCLUSIVE -- a solo re-run could not answer.")
        # 6 only when non-reproduction is the ONLY thing wrong. One genuine
        # failure anywhere in the run and the ordinary red code wins, so a real
        # regression is never filed under a flake's name.
        #
        # KNOWN LIMIT, measured 2026-08-26: the re-run happens on this box,
        # seconds after the lane it is classifying, so it inherits whatever the
        # lane left behind -- warm CPUs, sockets in TIME_WAIT, ranks still
        # exiting. `test_pp_admission_wraparound_never_blocks` is 5/5 green
        # alone on a settled box and failed the gate's own re-run immediately
        # after the serial lane. That error runs toward GENUINE, i.e. toward
        # red, which is the safe direction: the re-run can call a load artefact
        # real, never the reverse. A settling delay or a quarantine re-run is
        # owed, not built.
        elif rc == 0 and not_reproduced and not (union - set(not_reproduced)):
            rc = 6
            print("  VERDICT: exit 6 -- every failure in this run is unrecorded and")
            print("  did not reproduce when its module was re-run alone. No failure")
            print("  in this run survived isolation, so the product is not implicated")
            print("  by any of them. This is NOT a green: a load-sensitive regression")
            print("  presents exactly like this, and what separates the two is a")
            print("  measurement nobody has taken -- the table row, or the box.")

    if args.prove:
        ref = parse_log(args.prove)
        if not ref.tally_ok:
            print(f"\nPROOF INCONCLUSIVE: reference log tally broken: {ref.tally_note}")
            return 3
        # Restricted to the modules this gate actually RAN: the reference
        # includes the device-requiring modules, which no desk run can execute.
        # Comparing against them would report a fixed, meaningless delta.
        refnames = {n for n in ref.all_names if module_of(n) in ran}
        added, lost = sorted(union - refnames), sorted(refnames - union)
        print("\n=== equivalence proof against the serial reference ===")
        print(f"  reference {args.prove}: {len(refnames)} failure(s) over the modules this "
              f"gate actually ran")
        print(f"  this gate: {len(union)} failure(s)")
        if not added and not lost:
            print("  EQUAL -- the partitioned gate answers the same question as the "
                  "serial one, in both directions.")
        else:
            for n in added:
                print(f"  ADDED   {n}   (parallel-only; crowding or a real dependency)")
            for n in lost:
                print(f"  LOST    {n}   (false green)")
            print("  NOT EQUAL -- the partition changed the answer.")
            rc = max(rc, 5)

    print(f"\n# logs {' '.join(str(v) for v in logs.values())}")
    # pytest's own exit codes: 0 ok, 1 tests failed, 2 interrupted, 3 internal
    # error, 4 usage error, 5 nothing collected. Anything past 1 says the RUN
    # did not happen as asked, which no failure set can express -- so it is
    # named whatever else the verdict is, not only when the gate is otherwise
    # green.
    for lane_name, lane_rc in (("wide", rc_w), ("narrow", rc_n), ("serial", rc_s)):
        if lane_rc not in (0, 1):
            print(f"# LANE RC {lane_name} lane exited {lane_rc} -- pytest did not "
                  f"complete the run it was asked for")
    if rc == 0 and union:
        rc = 1
    if rc == 0 and max(rc_w, rc_n, rc_s) not in (0, 1):
        rc = max(rc_w, rc_n, rc_s)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
