#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and print the L1.5 pool record P_AWAKE_PEAK_MIB from P group logs.

    python3 scripts/l15_pool_peak_record.py /spinning/docker-acceptance/27b/evidence \
        --contains dkr27browauthority --newest 8 [--write FILE] [--with-d-pool]

Reads ``WEG2-VRAM-PEAK`` lines (phase chunk/round/idle by default) of every
``*.P.log`` it is given (files, or a directory), keeps per card (P rank) the
MAXIMUM of ``peak_reserved_mib`` over all boots and lines (never a mean) and
prints it with the boot / time / line it came from.  Read-only on the logs.
``--write FILE`` stores the JSON the planner reads behind
SGLANG_WEG2_L15_POOL_PEAK_RECORD (default file: weg2/profile_records_data/
l15_pool_peak_<profile>.json, or point SGLANG_WEG2_L15_POOL_PEAK_RECORD_FILE at it).
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys


def _load_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "python", "sglang", "srt", "weg2", "l15_pool_peak.py")
    spec = importlib.util.spec_from_file_location("l15_pool_peak", os.path.normpath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["l15_pool_peak"] = mod
    spec.loader.exec_module(mod)
    return mod


def _boot_name(path):
    base = os.path.basename(path)
    return re.sub(r"\.P\.log$", "", base)


def _collect(inputs, pattern, contains, newest):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, pattern))
        else:
            files.append(p)
    if contains:
        files = [f for f in files if contains in os.path.basename(f)]
    files = sorted(set(files), key=lambda f: (os.path.getmtime(f), f))
    if newest:
        files = files[-newest:]
    return files


def _d_pool(p_log):
    d_log = re.sub(r"\.P\.log$", ".D.log", p_log)
    rx = re.compile(r"TP(\d+)\] KV pool sizing: available_bytes=(\d+)")
    seen = {}
    try:
        with open(d_log, "r", errors="replace") as fh:
            for line in fh:
                m = rx.search(line)
                if m:
                    seen[int(m.group(1))] = int(m.group(2)) // (1 << 20)
    except OSError:
        return None
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="P.log files or evidence directories")
    ap.add_argument("--pattern", default="*.P.log")
    ap.add_argument("--contains", default="", help="keep files whose name contains this")
    ap.add_argument("--newest", type=int, default=0, help="only the N newest files (by mtime)")
    ap.add_argument("--profile", default="qwen27b")
    ap.add_argument("--phases", default="chunk,round,idle")
    ap.add_argument("--basis", default="peak_reserved_mib")
    ap.add_argument("--write", default="", help="store the record JSON here")
    ap.add_argument("--json", action="store_true", help="print the JSON instead of the table")
    ap.add_argument("--with-d-pool", action="store_true",
                    help="print the D KV pool (MiB per TP rank) of the newest boot's D.log beside the peaks")
    a = ap.parse_args(argv)
    mod = _load_module()
    files = _collect(a.inputs, a.pattern, a.contains, a.newest)
    if not files:
        print("no P.log matched", file=sys.stderr)
        return 2

    def sources():
        for f in files:
            with open(f, "r", errors="replace") as fh:
                yield _boot_name(f), fh

    rec = mod.build_record(sources(), profile=a.profile,
                           phases=tuple(x for x in a.phases.split(",") if x), basis=a.basis)
    if not rec["cards"]:
        print(f"no WEG2-VRAM-PEAK record line in {len(files)} file(s): no record", file=sys.stderr)
        return 3
    print(json.dumps(rec, indent=1, sort_keys=True) if a.json else mod.format_record(rec))
    if a.with_d_pool:
        pool = _d_pool(files[-1])
        print(f"D KV pool (MiB per TP rank, {_boot_name(files[-1])}.D.log): {pool}")
    if a.write:
        with open(a.write, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"written {a.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
