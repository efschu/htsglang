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
import re
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
# #895, measured 2026-08-26 on a SHARED box: the narrow lane's worker bound is
# a mitigation against the lane's OWN crowding, and nothing else. Two members
# of the pp_proxy family went red in two consecutive gate runs, a different
# member each time, while the box carried 14-28 of external load from other
# strands:
#
#   run 1  test_pp_proxy_cross_epoch_mispair_795   gloo connectFullMesh failed
#                                                  "Connection closed by peer"
#   run 2  test_pp_proxy_readiness_contract_789    "PROXY READINESS TIMEOUT 0.4s"
#
# Both passed 4/0 alone, on the base commit AND on the fix under test, so
# neither is a product failure. This is NOTE #868 §2.5's SIMULTANEITY class --
# an assert about where concurrent real ranks stand at a deadline -- and §2.5
# says in as many words that a single solo measurement cannot settle it,
# because its independent variable is the load on the box. The solo proof
# therefore ADMITS these modules, exactly as it would admit a device-requiring
# one; both need a refusal that does not come from the measurement.
#
# BY NAME, not by source probe, and the reason is worth recording: a probe for
# short wall-clock margins finds the 0.4 s budget in ...readiness_contract_789
# but finds NOTHING in ...cross_epoch_mispair_795, whose failure was in the
# gloo rendezvous itself, before any assert of its own. The marker owed in
# NOTE #868 §6 would have caught one of the two.
#
# SCOPE, honestly: this is a rate reduction, not the class fix. Every module in
# the RANKS lane shares the hazard in principle; demoting all 35 would cost the
# gate its reason to exist (766 s -> 345 s is the whole point). The CLASS is
# handled in the runner instead -- #895 gives an unrecorded parallel-lane
# failure its own named exit and an automatic solo re-run, so any other member
# that goes this way is classified by machine on the run it happens. Demoted
# here are the members with demonstrated recurrence.
SIMULTANEITY_REASON = "simultaneity:NOTE868_2.5_not_solo_provable;895_observed_2026-08-26"

# #899, 2026-08-26: the wraparound module's load sensitivity was ROOTED and
# HARDENED rather than left demoted, so its row no longer stands on the same
# ground as the four above and must not claim to. Measured, both directions,
# on this box:
#
#   before, 96 busy-loop processes on 32 cores (load 93)  2 of 3 FAILED --
#     'wraparound-check mode=blocking' not found in '<no progress recorded>'
#     and stuck_ranks [0,1,2] != [] with all three progress files absent.
#     NOT ONE rank had reached its first marker inside the old 12 s / 30 s
#     wall constants: three `spawn` interpreters had not finished
#     `import torch` and the gloo rendezvous.
#   after, same generator, load 158-211 (harsher)          3 passed, 341 s.
#   quiet box, before and after                            3 passed, ~32 s.
#
# The repair is in the module: setup is WAITED OUT under its own named budget
# and MEASURED, and the observation window is `max(floor, measured_setup)`.
# No assertion's subject, rank or marker changed. That is why #898 §6's
# refusal does not reach it -- that refusal was of ONE SHARED deadline
# multiplier across this family, on the evidence that two of the five members
# carry no deadline literal at all and that scaling `789`'s 0.4 s budget would
# disarm the specimen it exists to fire on. Neither objection applies to
# repairing one module's own clock in place.
#
# STILL SERIAL, and this is a measurement gap rather than a doubt: promotion
# to the RANKS lane needs `solo failure set == serial failure set` taken
# against the new bytes, and the logs this table was built from
# (/tmp/868_serial_ref.log, /tmp/868_solo) no longer exist. Naming the ground
# correctly is what lets that promotion be a measurement next time instead of
# a re-litigation.
HARDENED_REASON = "simultaneity:hardened_899_2026-08-26;serial_pending_solo_reproof"

# Module -> the reason its row carries. Membership is the refusal; the value
# is the ground that refusal stands on, which is not the same for every member
# and should not be reported as if it were.
NOT_CROWDING_PROVABLE = {
    "test/registered/unit/managers/test_pp_proxy_cross_epoch_mispair_795.py": SIMULTANEITY_REASON,
    "test/registered/unit/managers/test_pp_proxy_readiness_contract_789.py": SIMULTANEITY_REASON,
    # Same harness, same 3-rank gloo rendezvous, same shortened readiness
    # budget: the two above are the members that were observed, not the only
    # members that share the shape. The family is the unit.
    "test/registered/unit/managers/test_pp_proxy_readiness_rendezvous_789.py": SIMULTANEITY_REASON,
    "test/registered/unit/managers/test_pp_proxy_retracted_pass_mispair_791c.py": SIMULTANEITY_REASON,
    # Same class, found by the #895 gate run rather than looked for. #868
    # classified this one SERIAL with the reason
    # `solo_differs:fails_only_solo=PPAdmissionWraparoundBlocks::
    # test_blocking_wraparound_wedges_the_ring` -- it failed ALONE and passed
    # in the full serial run. On 2026-08-26 it did the exact opposite: it
    # failed in the SERIAL lane while two foreign gate runs put the box at load
    # 150, and passed 3/3 alone 4 minutes later at load 10. Its RED case asserts
    # which of three real ranks has reached which progress marker at a deadline;
    # under that load two of the three had recorded no progress at all. A
    # verdict that inverts with the box's load is not "differs solo" -- the
    # lane was never the variable. The lane it lands in does not change (SERIAL
    # either way), the NAME of the reason does, and the name is what a reader
    # acts on. #899 then rooted it -- see HARDENED_REASON above.
    "test/registered/unit/managers/test_pp_admission_wraparound_never_blocks.py": HARDENED_REASON,
}

# #898, measured 2026-08-26: the SECOND exclusion class, and it is not the
# NEEDS_DEVICE one. A module that calls `popen_launch_server` does not merely
# want a card -- it starts a REAL server process out of process, loads a model
# and talks HTTP to it. At the desk that is refused twice over: the hermetic
# run has no card, and the `sglang` console script is not on the gate's PATH,
# so the module dies in setUpClass with
# `FileNotFoundError: No such file or directory: 'sglang'` before a single
# assertion of its own runs.
#
# WHY IT MUST BE ITS OWN CLASS RATHER THAN FOLDED INTO NEEDS_DEVICE. The two
# behave differently in the one direction that matters. A NEEDS_DEVICE module
# fails a test; a launcher module fails at SETUP, and a setup failure is an
# ERROR, not a FAILURE -- pytest counts it in a different column, and the
# summary line reads `46 failed ... 25 errors`. Every extraction that greps
# `^FAILED` therefore reports 46 where the truth is 71. The 25 that class
# exactly are the ones this rule excludes, so naming the class also repairs the
# count (Extraktions-Zaehlprobe).
#
# BY SOURCE PROBE, NOT BY NAME, for the reason the RANKS lane already gives:
# a hand-kept name list goes stale the moment somebody adds a module. Measured
# on this tree: the marker fires on 7 of 34 modules in test/registered/scheduler
# and on 0 of 337 in test/registered/unit/{managers,planner,server_args,
# mem_cache}, so adding it CANNOT move a row in the existing managers table.
SERVER_LAUNCH_MARKERS = ("popen_launch_server",)


def needs_live_server(source: str) -> str | None:
    hit = [m for m in SERVER_LAUNCH_MARKERS if m in source]
    if hit:
        return "needs_server:launches_real_server:" + ",".join(sorted(hit))
    return None


# #862: the source probe reads ONE module's own bytes, and that is one hop too
# few. `test/registered/radix_cache/unified_radix_tree` holds 8 modules; 7 carry
# the marker and the 8th,
# `test_unified_radix_cache_kl_dsv4_pp.py`, is four lines of
#
#     import test_unified_radix_cache_kl_dsv4 as dsv4_kl
#     class TestUnifiedDeepSeekV4FlashHiCachePP4TP2(dsv4_kl.TestUnified...):
#
# It inherits setUpClass from a sibling that launches a real server, so it
# launches one too -- and the own-bytes probe would have ADMITTED it to a lane,
# where it dies in setUpClass with the same `FileNotFoundError: 'sglang'` as the
# 7 it sits next to. A test-class inheritance edge crossing a module boundary is
# invisible to the marker; the import that carries it is not.
#
# Deliberately narrow: SIBLING modules only (the bare-name import that works
# because pytest puts the test's own directory on sys.path), and only from a
# module already refused by the own-bytes probe. It is a one-hop closure of an
# existing verdict, not an import graph walk.
#
# PRECISION, measured 2026-08-26 before the rule was written, the way #898
# measured its own: it fires on 1 of 710 modules across
# test/registered/unit/{managers,planner,server_args,mem_cache},
# test/registered/scheduler and test/registered/radix_cache/unified_radix_tree
# -- that one module -- and on 0 in every path that already has a table, so it
# CANNOT move an existing row.
def needs_live_server_via_sibling(source: str, launcher_stems: set[str]) -> str | None:
    for stem in sorted(launcher_stems):
        name = re.escape(stem)
        pattern = rf"^\s*(?:import\s+{name}\b|from\s+{name}\s+import)"
        if re.search(pattern, source, re.M):
            return f"needs_server:inherits_from_launcher_sibling:{stem}"
    return None


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

    # #862: the sibling-inheritance closure needs to know which modules the
    # own-bytes probe already refuses, so it is computed once over the whole
    # gate path before any row is decided.
    launcher_stems = {
        Path(m).stem
        for m in modules
        if needs_live_server((root / m).read_text(errors="replace"))
    }

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

        # #898: refused BEFORE the solo comparison, and before the module is
        # ever handed to a lane. Unlike the sets above this one is decided from
        # the SOURCE, so it needs no measurement and cannot go stale.
        srv = needs_live_server(src) or needs_live_server_via_sibling(
            src, launcher_stems - {Path(mod).stem}
        )
        if srv:
            rows.append((mod, "EXCLUDED", srv, h, ref))
            stats["EXCLUDED"] += 1
            continue

        # Refused BEFORE the solo comparison, for the same reason NEEDS_DEVICE
        # is: the solo measurement would admit it. See the set's own comment.
        if mod in NOT_CROWDING_PROVABLE:
            rows.append((mod, "SERIAL", NOT_CROWDING_PROVABLE[mod], h, ref))
            stats["SERIAL"] += 1
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
