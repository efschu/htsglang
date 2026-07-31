#!/usr/bin/env python3
"""s06 check -- schema validation of the NCCL reference against the envelope
the consumer defines, plus the two coverage rules the #278 wrap-up bought
with a wasted measurement.

Schema (barlink_path_rates.NCCL_REFERENCE_*):
  * kind == "nccl_reference", schema_version == 1,
  * every row carries ALL ten mandatory fields; a row missing one is dropped
    by the loader, so a file full of nine-field rows loads as an empty file,
  * p50_us and p99_us are real numbers and p99 >= p50. A p99 below its p50 is
    not a tail, it is a bookkeeping error.

Coverage, which the schema alone does not enforce:
  * BOTH arms present -- idle AND a named load arm -- over the SAME (op, pair,
    size) keys. The #278 load axis was taken p50 on one side and p99 on the
    other, which made it uncomparable and unusable; the lesson is that a load
    arm is only worth taking symmetrically.
  * Both DIRECTIONS present for send_recv on every pair. The rig is asymmetric
    by construction and a one-directional table hides exactly that.

And the real gate: load_nccl_reference() must return measured profiles with
zero errors -- the file has to be usable by the loader that will use it, not
merely by this check.

NOT judged: the numbers. Whether P2P is faster than SHM here is what the rows
are for; the executor does not get an opinion.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    add_repo_to_path,
    is_number,
    load_json,
    missing_fields,
    require_envelope,
    run_check,
)

STEP = "s06_nccl_reference"

MANDATORY = (
    "op",
    "transport",
    "world",
    "src_pci",
    "dst_pci",
    "size_bytes",
    "iters",
    "p50_us",
    "p99_us",
    "load",
)


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "nccl_reference.json")
    payload = load_json(path, "nccl_reference.json")
    require_envelope(payload, "nccl_reference", "nccl_reference.json", 1)

    # The pair status comes first: an aborted pair explains the missing rows,
    # and "no rows" would report the consequence instead of the cause. The
    # producer writes its partial result after every pair, so a step that ran
    # out of budget leaves exactly this file.
    for status in payload.get("pairs_status") or []:
        if status.get("status") != "ok":
            detail = status.get("detail")
            raise CheckFail(
                f"pair {status.get('pci_pair')}: status {status.get('status')!r}"
                + (f" ({detail})" if detail else "")
                + " -- an aborted measurement is not a reference"
            )

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CheckFail("nccl_reference.json has no rows")

    arms = defaultdict(set)
    directions = defaultdict(set)
    for i, row in enumerate(rows):
        absent = missing_fields(row, MANDATORY)
        if absent:
            raise CheckFail(
                f"rows[{i}]: mandatory fields missing {absent} -- the loader drops "
                "rows like that, so the file then loads as empty"
            )
        for field in ("p50_us", "p99_us"):
            if not is_number(row[field]):
                raise CheckFail(f"rows[{i}]: {field} is {row[field]!r}, not a number")
        if row["p99_us"] < row["p50_us"]:
            raise CheckFail(
                f"rows[{i}] ({row['op']} {row['size_bytes']}B {row['load']}): "
                f"p99 {row['p99_us']} < p50 {row['p50_us']}"
            )
        if not is_number(row["iters"]) or row["iters"] < 1:
            raise CheckFail(f"rows[{i}]: iters is {row['iters']!r}")
        if row["transport"] in (None, ""):
            raise CheckFail(
                f"rows[{i}]: transport is empty -- without the NCCL_DEBUG finding "
                "the row cannot be attributed to anything"
            )

        key = (
            row["op"],
            tuple(sorted((row["src_pci"], row["dst_pci"]))),
            row["size_bytes"],
        )
        arms[key].add(row["load"])
        if row["op"] == "send_recv":
            pair = tuple(sorted((row["src_pci"], row["dst_pci"])))
            directions[pair].add((row["src_pci"], row["dst_pci"]))

    load_arms = {arm for arms_set in arms.values() for arm in arms_set}
    if "idle" not in load_arms:
        raise CheckFail("no idle arm -- without one there is no cost model")
    named_load = load_arms - {"idle"}
    if not named_load:
        raise CheckFail(
            "no named Last-Arm -- the load axis is mandatory in the schema and the "
            "whole reason the format was pinned down in the first place"
        )

    incomplete = [k for k, v in arms.items() if not ({"idle"} | named_load) <= v]
    if incomplete:
        key = incomplete[0]
        raise CheckFail(
            f"{len(incomplete)} key(s) only partly armed, e.g. "
            f"{key[0]} {key[1]} {key[2]}B -- the load axis has to lie symmetrisch "
            "over the same keys"
        )

    if not directions:
        raise CheckFail(
            "no send_recv rows -- symmetric collectives alone average away the "
            "rig's asymmetry"
        )
    for pair, seen in directions.items():
        if len(seen) < 2:
            raise CheckFail(
                f"send_recv for {pair} measured in one Richtung only ({sorted(seen)})"
            )

    _check_loadable(payload)


def _check_loadable(payload: dict) -> None:
    add_repo_to_path()
    try:
        from sglang.srt.distributed.device_communicators.barlink_path_rates import (
            load_nccl_reference,
        )
    except Exception as exc:
        raise CheckStop(f"barlink_path_rates not importable: {exc}") from exc

    res = load_nccl_reference(payload)
    if res.errors:
        raise CheckFail(f"rejected by the #279 loader: {res.errors[0]}")
    if not res.profiles:
        raise CheckFail("the #279 loader builds ZERO profiles from the file")
    measured = [p for p in res.profiles if p.provenance == "measured"]
    if not measured:
        raise CheckFail("not a single measured profile out of the reference")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
