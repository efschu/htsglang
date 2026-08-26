#!/usr/bin/env python3
"""#868 -- build the tier-2 partition table from measured evidence.

Inputs
  --serial   log of the FULL serial gate run (the authority)
  --solo-dir directory of per-module solo logs (one process per module)

Output: a TSV data file, one row per test module in the gate path:

    module <TAB> verdict <TAB> reason <TAB> sha256 <TAB> ref_failures

verdict PARALLEL means the module PROVED, by measurement, that it produces the
same failure set alone in a fresh process as it does inside the full serial
run.  Everything else is SERIAL, with the reason recorded rather than
remembered.

The table is DATA, not code, for two reasons: the runner must be able to say
at every run exactly what it excluded and why, and a reviewer must be able to
re-derive the table from the two logs without reading the runner.

``sha256`` is the proof's expiry date.  A module whose bytes no longer match
the hash recorded here has an INVALID proof, and the runner demotes it to the
serial track.  That is what stops a new or edited module from drifting into
the parallel track without a solo measurement behind it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition_lib import by_module, parse_log  # noqa: E402

# ---------------------------------------------------------------------------
# BY-NAME exclusions. These are NOT solo-provable and must never be decided by
# the solo measurement, because the solo measurement would happily admit them.
# ---------------------------------------------------------------------------

# NOTE #860 §0.7, measured: three modules require a real accelerator and FAIL
# under CUDA_VISIBLE_DEVICES="". They are excluded from the desk gate
# altogether -- from BOTH tracks -- and the exclusion is reported on every run
# so nobody reads the narrowed gate as the full one.
NEEDS_DEVICE = {
    "test/registered/unit/managers/test_arena_high_water_631.py",
    "test/registered/unit/managers/test_phase_flip_rotation_wiring_809.py",
    "test/registered/unit/managers/test_restore_never_rebuild_677.py",
}

# NOTE #860 §1 divergence 2, measured: the multi-rank family spawns real ranks
# and waits out real timeouts, so it goes flaky-RED under CPU pressure -- a
# different member each run. That is crowding, not a dependency: the solo proof
# passes and would admit them into the wide lane.
#
# They are not refused, they are given their OWN lane with a bounded worker
# count, and the bound has a physical basis rather than a feeling: a module in
# this family occupies its controller plus its ranks, so the lane's peak
# process count is roughly `workers x (1 + ranks)`. Kept below the core count,
# no rank is starved and the timeout these tests wait on means what it meant
# serially. The lanes run one after another, so the wide lane never crowds this
# one.
#
# Membership is read from the SOURCE, never from a hand-kept name list: a
# module that starts real workers imports one of these. A name list goes stale
# the moment somebody adds a module; a source probe does not.
RANK_SPAWN_MARKERS = (
    "multiprocessing",
    "mp.Process",
    "spawn_rank",
    "_regime_shutdown_child",
    "torch.distributed",
    "init_process_group",
)


def is_rank_spawner(module: str, source: str) -> str | None:
    hit = [m for m in RANK_SPAWN_MARKERS if m in source]
    if hit:
        return "spawns_real_ranks:" + ",".join(sorted(hit))
    return None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--gate-path", default="test/registered/unit/managers")
    ap.add_argument("--serial", required=True)
    ap.add_argument("--solo-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="?")
    args = ap.parse_args()

    root = Path(args.root)
    gate_dir = root / args.gate_path

    serial = parse_log(args.serial)
    if not serial.tally_ok:
        print(f"REFUSING: serial reference log failed the tally gate: {serial.tally_note}",
              file=sys.stderr)
        return 2
    serial_by_mod = by_module(serial.all_names)

    modules = sorted(p.relative_to(root).as_posix() for p in gate_dir.glob("test_*.py"))

    rows = []
    stats = {"PARALLEL": 0, "RANKS": 0, "SERIAL": 0, "EXCLUDED": 0}
    for mod in modules:
        src = (root / mod).read_text(errors="replace")
        ref = sorted(serial_by_mod.get(mod, set()))
        h = sha(root / mod)

        if mod in NEEDS_DEVICE:
            rows.append((mod, "EXCLUDED", "needs_device:CVD_empty_fails_NOTE860_0.7", h, ref))
            stats["EXCLUDED"] += 1
            continue

        solo_log = Path(args.solo_dir) / (Path(mod).stem + ".log")
        if not solo_log.exists():
            rows.append((mod, "SERIAL", "unproven:no_solo_log", h, ref))
            stats["SERIAL"] += 1
            continue

        solo = parse_log(solo_log)
        if not solo.tally_ok:
            rows.append((mod, "SERIAL", f"unproven:tally_broken({solo.tally_note})", h, ref))
            stats["SERIAL"] += 1
            continue
        if solo.rc not in (0, 1):
            rows.append((mod, "SERIAL", f"unproven:solo_rc={solo.rc}", h, ref))
            stats["SERIAL"] += 1
            continue

        solo_names = sorted(solo.all_names)
        if solo_names == ref:
            # Proof holds. Which LANE it goes to is a crowding question, not a
            # correctness one, and it is answered from the source.
            spawns = is_rank_spawner(mod, src)
            if spawns:
                rows.append((mod, "RANKS", f"solo_equals_serial;{spawns}", h, ref))
                stats["RANKS"] += 1
            else:
                rows.append((mod, "PARALLEL", "solo_equals_serial", h, ref))
                stats["PARALLEL"] += 1
        else:
            only_solo = sorted(set(solo_names) - set(ref))
            only_ser = sorted(set(ref) - set(solo_names))
            bits = []
            if only_solo:
                bits.append("fails_only_solo=" + ";".join(x.split("::", 1)[-1] for x in only_solo))
            if only_ser:
                bits.append("fails_only_serial=" + ";".join(x.split("::", 1)[-1] for x in only_ser))
            rows.append((mod, "SERIAL", "solo_differs:" + "|".join(bits), h, ref))
            stats["SERIAL"] += 1

    out = Path(args.out)
    with out.open("w") as f:
        f.write("# tier-2 gate partition table -- #868\n")
        f.write("# GENERATED by scripts/gate_partition_build.py from measured logs.\n")
        f.write("# Do not hand-edit a verdict: the runner re-checks each module's sha256\n")
        f.write("# and demotes any module whose bytes no longer match its proof.\n")
        f.write(f"# commit      {args.commit}\n")
        f.write(f"# gate path   {args.gate_path}\n")
        f.write(f"# serial ref  {args.serial}\n")
        f.write(f"#             {serial.counts} wall={serial.wall}s\n")
        f.write(f"# solo dir    {args.solo_dir}\n")
        f.write("# hermetic    every run behind this table used CUDA_VISIBLE_DEVICES=\"\"\n")
        f.write("#\n")
        f.write("# verdicts\n")
        f.write("#   PARALLEL  solo failure set == serial failure set -> wide lane\n")
        f.write("#   RANKS     same proof, but the module spawns real ranks -> narrow lane,\n")
        f.write("#             bounded workers so no rank is starved past its timeout\n")
        f.write("#   SERIAL    proof failed -> one process, one order (reason below)\n")
        f.write("#   EXCLUDED  cannot run at the desk at all (needs a card) -> neither track,\n")
        f.write("#             reported on every run so the gate is never read wider than it is\n")
        f.write(f"# totals      {stats}\n")
        f.write("#\n")
        f.write("#module\tverdict\treason\tsha256\tref_failures\n")
        for mod, verdict, reason, h, ref in rows:
            f.write(f"{mod}\t{verdict}\t{reason}\t{h}\t{','.join(ref)}\n")

    print(f"wrote {out}  {stats}")
    print(f"serial reference: {serial.counts} wall={serial.wall}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
