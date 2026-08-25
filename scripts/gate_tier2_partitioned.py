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

``--prove`` adds the OTHER direction for acceptance: the union must EQUAL the
reference set, not merely contain it.  It is a separate flag because in normal
gate use an added failure is the gate doing its job, while during acceptance an
added failure means the partition changed the answer.

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
DEFAULT_TABLE = ROOT / "scripts" / "gate_partition.tsv"
PY = os.environ.get("GATE_PY", "/spinning/htsglang-gpu/.venv/bin/python3")


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
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--gate-path", default="test/registered/unit/managers")
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

    table = load_table(Path(args.table))
    present = sorted(p.relative_to(ROOT).as_posix()
                     for p in (ROOT / args.gate_path).glob("test_*.py"))

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
        (wide if v == "PARALLEL" else narrow if v == "RANKS" else serial).append(mod)

    stale_gone = [m for m in table if m not in present]

    if args.verify:
        print(f"# verify {args.table} against {args.gate_path}")
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
    print(f"# table    {args.table}")
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
        print(f"  {name:7s}: {'OK' if r.tally_ok else 'BROKEN'}  counts={r.counts} "
              f"names={len(r.all_names)} {r.tally_note}")
        broken = broken or not r.tally_ok
    if broken:
        print("\nVERDICT: INCONCLUSIVE -- the extraction is broken, not the run.")
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
    if rc == 0 and union:
        rc = 1
    if rc == 0 and max(rc_w, rc_n, rc_s) not in (0, 1):
        rc = max(rc_w, rc_n, rc_s)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
