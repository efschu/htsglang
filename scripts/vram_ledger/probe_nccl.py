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
"""Fold measured NCCL communicator buffers into the ledger's cache.

    # 1. boot the target recipe with the instrumentation armed
    SGLANG_NCCL_BUFFER_DUMP=/spinning/nccl_dumps <the usual launch>
    # 2. once the ranks are up (the buffers are allocated at communicator
    #    init, so no traffic is needed), fold the dumps in
    python scripts/vram_ledger/probe_nccl.py ingest --dump-dir /spinning/nccl_dumps

The measurement is valid for one (rig fingerprint, communicator set) pair and
the cache is keyed on both. Change the TP width, or hand the TP group to
barlink, and the signature moves and this measurement stops being used --
which is the point: an NCCL figure that outlived its communicator set is an
under- or over-charge wearing a measurement's clothes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
)
if REPO_PYTHON not in sys.path:
    sys.path.insert(0, REPO_PYTHON)

from sglang.srt.mem_ledger.nccl_probe import (  # noqa: E402
    ingest_dumps,
    load_nccl_buffers,
    nccl_cache_path,
)


def cmd_ingest(args) -> int:
    path, notes = ingest_dumps(args.dump_dir, cache_dir=args.cache_dir)
    for n in notes:
        print(" ", n)
    if path is None:
        print("Nothing cached.")
        return 1
    print("Cached:", path)
    with open(path) as f:
        d = json.load(f)
    print("  fingerprint:", d["hw_fingerprint"])
    print("  signature  :", d["nccl_signature"])
    for uuid, mib in sorted(d["per_uuid_mib"].items()):
        print(f"  {uuid}  {mib:8.1f} MiB")
    return 0


def cmd_show(args) -> int:
    if args.signature:
        per = load_nccl_buffers(
            args.fingerprint, args.signature, cache_dir=args.cache_dir
        )
        if per is None:
            print(
                "No NCCL measurement for fingerprint "
                f"{args.fingerprint} / signature {args.signature}. Expected at "
                f"{nccl_cache_path(args.fingerprint, args.signature, args.cache_dir)}"
            )
            return 1
        for uuid, mib in sorted(per.items()):
            print(f"  {uuid}  {mib:8.1f} MiB")
        return 0

    from sglang.srt.rigmon.card_probe import CACHE_DIR

    pattern = os.path.join(args.cache_dir or CACHE_DIR, "nccl_buffers-*.json")
    found = sorted(glob.glob(pattern))
    if not found:
        print("No NCCL measurements cached.")
        return 1
    for path in found:
        with open(path) as f:
            d = json.load(f)
        print(f"{os.path.basename(path)}")
        print("  fingerprint:", d.get("hw_fingerprint"))
        print("  signature  :", d.get("nccl_signature"))
        for uuid, mib in sorted((d.get("per_uuid_mib") or {}).items()):
            print(f"  {uuid}  {float(mib):8.1f} MiB")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="probe_nccl.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="fold per-card dumps into the cache")
    p_ing.add_argument("--dump-dir", required=True)
    p_ing.add_argument("--cache-dir", default=None)
    p_ing.set_defaults(fn=cmd_ingest)

    p_show = sub.add_parser("show", help="print cached measurements")
    p_show.add_argument("--fingerprint", default="")
    p_show.add_argument("--signature", default="")
    p_show.add_argument("--cache-dir", default=None)
    p_show.set_defaults(fn=cmd_show)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
