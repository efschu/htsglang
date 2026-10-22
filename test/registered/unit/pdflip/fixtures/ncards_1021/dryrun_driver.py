"""NCARDS 1021: one launcher dry run of a RELEASE PROFILE FILE on N cards of one catalog class, printed as a compact JSON document.

Run as a SUBPROCESS by ``test_ncards_dryruns_1021.py`` (a dry run is ~1.8 GB of resident memory; one process per run keeps the
serial test under the 2 GB scope cap, and the parent test imports nothing heavy).

    python dryrun_driver.py <profile.env> <catalog-card-id> <n> [--noforce] [--rig]

``--rig``: the reference rig NVML recording (5090 + 2 x 3080) instead of ``n`` synthetic cards.
``--arg=<launcher token>`` (repeatable): appended to the launcher argv (argparse: the last occurrence wins), e.g. ``--arg=--max-kv-per-request=262144``.
CPU only; no GPU, no container, no boot.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = pathlib.Path(__file__).resolve().parent
TREE = pathlib.Path(os.environ.get("NCARDS_TREE") or HERE.parents[5])
CKPT = HERE.parent / "planer_1006" / "checkpoints"
REPLAY_REF = HERE.parent / "xchg_launch_replay_0911" / "nvml_devices_1378.json"


def main(argv):
    from flliper.srt.pdflip import propose_oracle as O

    env_path, card, n = argv[0], argv[1], int(argv[2])
    force = "--noforce" not in argv
    spec = importlib.util.spec_from_file_location("ncards_kc", str(TREE / "tools" / "rig_dashboard" / "rigdash" / "kartenplan_catalog.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cat = {e["id"]: e for e in mod.CATALOG}
    rows = O.read_replay(str(REPLAY_REF)) if "--rig" in argv else O.replay_from_catalog([cat[card]] * n)
    snaps = {O.read_snapshot_manifest(str(d))["name"]: str(d) for d in sorted(CKPT.iterdir()) if (d / "manifest.json").is_file()}
    extra = [a[len("--arg="):] for a in argv if a.startswith("--arg=")]
    run = O.run_profile(env_path, rows, tree=str(TREE), force=force, snapshots=snaps, extra_args=extra)
    res = run.result
    doc = {"exc_type": res.exc_type, "exc_head": (res.exc_msg or "")[:160] if res.exc_type else "",
           "forced": sorted({f["code"] for f in res.forced})}
    if not res.exc_type:
        plan = O.parse_plan_dump(res.text)
        pc = plan.get("pp_cut") or {}
        doc["pp_cut"] = {k: pc.get(k) for k in ("layers", "attn", "chosen_pool", "pool_floor", "pool_floor_source")}
        doc["stages"] = len(str(pc.get("layers", "")).split(",")) if pc.get("layers") else 0
        doc["budgets"] = [[b.get("pass"), b.get("ordinal"), b.get("mib")] for b in plan.get("budgets", [])]
        # review 1021 A1: the launcher's D residency check (FRACTION-SOLVE D(Karte, Erwartung)) is part of the document -- it must RUN
        # (no ENTFAELLT) and every rank must PASS; "borrowed" = the 3-rank reference was taken to this rank count by role.
        d_lines = [l for l in res.text.splitlines() if "FRACTION-SOLVE D(Karte, Erwartung)" in l]
        if d_lines:
            last = {}
            for l in d_lines:
                m = re.search(r"rang(\d+): Ratio .*? Scratch (\d+) .*? f ([0-9.]+) -> R (\d+), .*? nach KV -?\d+ MiB .*?-> (PASST|[A-Z]\S*)", l)
                if m:
                    last[int(m.group(1))] = {"scratch": int(m.group(2)), "R": int(m.group(4)), "verdict": m.group(5)}
            doc["d_check"] = {"entfaellt": any("ENTFAELLT" in l for l in d_lines), "borrowed_by_role": any("BORROWED BY RANK ROLE" in l for l in d_lines),
                              "ranks": [last[k] for k in sorted(last)]}
        gf = plan.get("group_flags", {})
        doc["group_flags"] = {g: {k: gf.get(g, {}).get(k) for k in ("--rank-tp-ratio", "--rank-gpu-memory-mib", "--max-total-tokens", "--tp-size", "--pp-size")}
                              for g in ("P", "D")}
    print("NCARDS-JSON " + json.dumps(doc, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
