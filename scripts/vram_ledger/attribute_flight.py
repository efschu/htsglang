#!/usr/bin/env python3
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read what the VRAM flight recorder wrote.

    # source 1 + 3: what each boot phase cost, per rank
    python scripts/vram_ledger/attribute_flight.py phases /tmp/flight

    # source 2: which line is holding the resident bytes
    python scripts/vram_ledger/attribute_flight.py snapshot /tmp/flight/*.pickle

Reads only. Touches no GPU, imports no torch, and writes nothing back into the
ledger -- turning these measurements into terms is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.mem_ledger.flight_recorder import (  # noqa: E402
    MIB,
    churn_attribution,
    list_boots,
    phase_deltas,
    read_marks,
    resident_attribution,
)


def _boots(args) -> int:
    import datetime

    boots = list_boots(args.directory)
    if not boots:
        print(f"No flight marks under {args.directory}.")
        return 1
    print(f"{'boot id':<24} {'first mark':<20} {'marks':>6}")
    for boot, wall, count in boots:
        stamp = datetime.datetime.fromtimestamp(wall).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{boot:<24} {stamp:<20} {count:>6}")
    print(f"\nlatest is {boots[-1][0]}; 'phases' shows it unless --boot says otherwise")
    return 0


def _phases(args) -> int:
    by_rank = read_marks(args.directory, boot=getattr(args, "boot", None))
    if not by_rank:
        print(f"No flight marks under {args.directory}.")
        print("Boot with SGLANG_VRAM_FLIGHT_DIR=<dir> to produce them.")
        return 1
    for rank in sorted(by_rank):
        marks = by_rank[rank]
        first = marks[0]
        print(
            f"\n=== rank {rank} (pid {first.get('pid')}, "
            f"boot {first.get('boot_id')}) ==="
        )
        card = first.get("card_uuid") or marks[-1].get("card_uuid") or "unresolved"
        print(f"card {card}")
        print(
            f"{'phase':<20} {'torch resv':>11} {'torch alloc':>12} "
            f"{'non-torch':>10} {'NVML self':>10} {'NVML free':>10}"
        )
        for m in marks:
            print(
                f"{str(m.get('phase')):<20} "
                f"{m.get('reserved_bytes', 0) // MIB:>7} MiB "
                f"{m.get('allocated_bytes', 0) // MIB:>8} MiB "
                f"{m.get('non_torch_bytes', 0) // MIB:>6} MiB "
                f"{m.get('nvml_self_bytes', 0) // MIB:>6} MiB "
                f"{m.get('nvml_free_bytes', 0) // MIB:>6} MiB"
            )
        print("\nposts (the difference between two marks IS the post):")
        for d in phase_deltas(marks):
            print(f"  {d.row()}")
        last = marks[-1]
        if last.get("nvml_carve_out_bytes"):
            print(
                f"\ncarve-out on this card: "
                f"{last['nvml_carve_out_bytes'] // MIB} MiB (REPORTED by NVML, "
                "never allocatable)"
            )
        procs = last.get("nvml_processes") or {}
        if procs:
            print("processes NVML sees on this card (the direct check, not a gap):")
            for pid, byte_count in sorted(procs.items(), key=lambda kv: -kv[1]):
                tag = " <- this rank" if int(pid) == int(last.get("pid", -1)) else ""
                print(f"  pid {pid:>8}  {byte_count // MIB:>6} MiB{tag}")
    return 0


def _snapshot(args) -> int:
    with open(args.path, "rb") as f:
        snapshot = pickle.load(f)

    resident, coverage = resident_attribution(snapshot)
    print(f"=== resident attribution ({args.path}) ===")
    print(coverage.verdict())
    for footprint in resident[: args.top]:
        print(f"  {footprint.mib:>7} MiB  x{footprint.count:<5} {footprint.site}")
        if args.stacks:
            for frame in footprint.stack[1:]:
                print(f"              <- {frame}")

    churn, churn_coverage, stats = churn_attribution(snapshot)
    print("\n=== churn attribution (device_traces window) ===")
    print(
        f"window {stats['window_seconds']:.1f} s, {stats['entries']} entries, "
        f"{stats['alloc_bytes'] // MIB} MiB allocated, peak outstanding "
        f"{stats['peak_outstanding_bytes'] // MIB} MiB"
    )
    print(churn_coverage.verdict())
    for footprint in churn[: args.top]:
        print(f"  {footprint.mib:>7} MiB  x{footprint.count:<5} {footprint.site}")
    return 0 if coverage.complete else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("phases", help="per-phase costs from the mark log")
    p.add_argument("directory")
    p.add_argument(
        "--boot",
        default=None,
        help="which boot to show (default: the latest; 'all' ignores the seams)",
    )
    p.set_defaults(func=_phases)

    p = sub.add_parser("boots", help="which boots this directory holds")
    p.add_argument("directory")
    p.set_defaults(func=_boots)

    p = sub.add_parser("snapshot", help="per-callsite bytes from a trace dump")
    p.add_argument("path")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--stacks", action="store_true", help="print the python stack")
    p.set_defaults(func=_snapshot)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
