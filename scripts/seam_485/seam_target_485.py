#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#485 -- WHICH seam term must fall? Per-rank, per-direction, absolute MiB.

m584's p2_seam_lever scaled every SEAM_* term of every rank by one factor and
reported the threshold as if the whole population had to move together. A
mechanism does not work that way: a mechanism attacks a named term on a named
leg. This sets each SEAM_* entry to an ABSOLUTE MiB value independently, so a
verdict can be attributed to the term a mechanism would actually touch.

Usage:
  seam_target_485.py --one <census> <pool> <budget>      (worker)
  seam_target_485.py probe <src_census> <out_root>       (the sweep)
"""
import glob
import io
import json
import os
import shutil
import subprocess
import sys

REPO = "/spinning/wt-desk-seam-485"
sys.path.insert(0, os.path.join(REPO, "python"))

ARGV_SRC = "/spinning/evidence-631/m485/ship_argv_live.txt"
DROP_FLAGS_WITH_VALUE = {
    "--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-layer-ratio",
}
BASE_BUDGET = (31400, 19300, 19300)
POOL = 280000


def build_argv(census_dir, pool, budget):
    raw = [a for a in open(ARGV_SRC).read().split("\n") if a]
    argv = raw[3:]
    out, i = [], 0
    while i < len(argv):
        if argv[i] in DROP_FLAGS_WITH_VALUE:
            i += 2
            continue
        out.append(argv[i])
        i += 1

    def put(flag, value):
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out.extend([flag, value])

    put("--device", "cuda")
    put("--pp-solve-cut", census_dir)
    put("--max-total-tokens", str(pool))
    put("--rank-gpu-memory-mib", ",".join(str(b) for b in budget))
    put("--rank-gpu-id", "0,1,2")
    put("--port", "30041")
    return out


def solve_inproc(census_dir, pool, budget):
    from sglang.srt.server_args import prepare_server_args
    buf = io.StringIO()
    try:
        with __import__("contextlib").redirect_stdout(buf), \
             __import__("contextlib").redirect_stderr(buf):
            sa = prepare_server_args(build_argv(census_dir, pool, budget))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"pp_layer_ratio={sa.pp_layer_ratio}"


def write_census(src, dst, overrides):
    """overrides: {(rank, state): absolute_mib}. Absent -> unchanged."""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for path in sorted(glob.glob(os.path.join(dst, "transient_pp*.json"))):
        d = json.load(open(path))
        rank = int(d["pp_rank"])
        tr = d.get("transient_mib_by_load_state") or {}
        for state in list(tr):
            if state.startswith("SEAM_") and (rank, state) in overrides:
                tr[state] = float(overrides[(rank, state)])
        if tr:
            worst = max(tr, key=lambda k: tr[k])
            d["worst_load_state"] = worst
            d["worst_transient_mib"] = tr[worst]
        d["transient_mib_by_load_state"] = tr
        json.dump(d, open(path, "w"), indent=1)


def solve(census):
    p = subprocess.run(
        [sys.executable, __file__, "--one", census, str(POOL),
         ",".join(str(b) for b in BASE_BUDGET)],
        capture_output=True, text=True, timeout=900,
    )
    for line in p.stdout.splitlines():
        if line.startswith("VERDICT "):
            _, verdict, detail = line.split(" ", 2)
            return verdict == "ADMIT", detail
    return False, f"no verdict (rc={p.returncode}) {p.stderr[-200:]}"


def main():
    if sys.argv[1] == "--one":
        census, pool = sys.argv[2], int(sys.argv[3])
        budget = tuple(int(x) for x in sys.argv[4].split(","))
        ok, detail = solve_inproc(census, pool, budget)
        print(f"VERDICT {'ADMIT' if ok else 'REFUSE'} {detail}".replace("\n", " | "))
        return 0

    src, out_root = sys.argv[2], sys.argv[3]
    os.makedirs(out_root, exist_ok=True)

    base = {}
    for path in sorted(glob.glob(os.path.join(src, "transient_pp*.json"))):
        d = json.load(open(path))
        r = int(d["pp_rank"])
        for k, v in (d.get("transient_mib_by_load_state") or {}).items():
            if k.startswith("SEAM_"):
                base[(r, k)] = v
    print("measured SEAM_* terms on this census:")
    for k in sorted(base):
        print(f"   rank {k[0]}  {k[1]:<16} {base[k]:8.1f} MiB")
    print()

    cases = []
    # A: baseline
    cases.append(("baseline (unchanged)", {}))
    # B: rank 0 only, both legs, down to a target
    for tgt in (900, 700, 600, 515, 460, 300, 0):
        cases.append((f"rank0 BOTH legs -> {tgt}",
                      {(0, "SEAM_TP_TO_PP"): tgt, (0, "SEAM_PP_TO_TP"): min(tgt, base[(0, "SEAM_PP_TO_TP")])}))
    # C: rank 0 tp_to_pp leg ONLY (what an arena-side mechanism touches)
    for tgt in (515, 300, 0):
        cases.append((f"rank0 TP_TO_PP only -> {tgt} (pp_to_tp left at "
                      f"{base[(0,'SEAM_PP_TO_TP')]:.0f})",
                      {(0, "SEAM_TP_TO_PP"): tgt}))
    # D: every rank, both legs (the m584 uniform sweep, as a control)
    for tgt in (515, 460):
        cases.append((f"ALL ranks both legs -> min(measured,{tgt})",
                      {k: min(v, tgt) for k, v in base.items()}))
    # E: rank 0 both legs to 515, ranks 1/2 untouched vs also capped
    cases.append(("rank0 both -> 515, ranks1/2 UNTOUCHED",
                  {(0, "SEAM_TP_TO_PP"): 515, (0, "SEAM_PP_TO_TP"): 515}))

    for name, ov in cases:
        dst = os.path.join(out_root, "c_" + str(abs(hash(name)) % 10**8))
        write_census(src, dst, ov)
        ok, detail = solve(dst)
        print(f"  {'ADMIT ' if ok else 'refuse'}  {name:<62} {detail.split('|')[0][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
