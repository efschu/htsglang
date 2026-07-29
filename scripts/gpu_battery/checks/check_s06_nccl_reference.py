#!/usr/bin/env python3
"""s06 check -- schema validation of the NCCL reference against the envelope
the consumer defines, plus the two coverage rules the #278 wrap-up bought
with a wasted measurement.

Schema (htccl_path_rates.NCCL_REFERENCE_*):
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

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CheckFail("nccl_reference.json hat keine rows")

    for status in payload.get("pairs_status") or []:
        if status.get("status") != "ok":
            raise CheckFail(
                f"Paar {status.get('pci_pair')}: status {status.get('status')!r} -- "
                "eine abgebrochene Messung ist keine Referenz"
            )

    arms = defaultdict(set)
    directions = defaultdict(set)
    for i, row in enumerate(rows):
        absent = missing_fields(row, MANDATORY)
        if absent:
            raise CheckFail(
                f"rows[{i}]: Pflichtfelder fehlen {absent} -- der Lader verwirft "
                "solche Zeilen, die Datei laedt dann als leer"
            )
        for field in ("p50_us", "p99_us"):
            if not is_number(row[field]):
                raise CheckFail(f"rows[{i}]: {field} ist {row[field]!r}, keine Zahl")
        if row["p99_us"] < row["p50_us"]:
            raise CheckFail(
                f"rows[{i}] ({row['op']} {row['size_bytes']}B {row['load']}): "
                f"p99 {row['p99_us']} < p50 {row['p50_us']}"
            )
        if not is_number(row["iters"]) or row["iters"] < 1:
            raise CheckFail(f"rows[{i}]: iters ist {row['iters']!r}")
        if row["transport"] in (None, ""):
            raise CheckFail(
                f"rows[{i}]: transport ist leer -- ohne den NCCL_DEBUG-Befund ist "
                "die Zeile nicht zuordenbar"
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
        raise CheckFail("kein idle-Arm -- ohne ihn gibt es kein Kostenmodell")
    named_load = load_arms - {"idle"}
    if not named_load:
        raise CheckFail(
            "kein benannter Last-Arm -- die Last-Achse ist Pflicht im Schema und "
            "der Grund, warum das Format ueberhaupt festgelegt wurde"
        )

    incomplete = [k for k, v in arms.items() if not ({"idle"} | named_load) <= v]
    if incomplete:
        key = incomplete[0]
        raise CheckFail(
            f"{len(incomplete)} Schluessel nur teilweise bearmt, z.B. "
            f"{key[0]} {key[1]} {key[2]}B -- die Last-Achse muss symmetrisch "
            "ueber dieselben Schluessel liegen"
        )

    if not directions:
        raise CheckFail(
            "keine send_recv-Zeilen -- nur symmetrische Kollektive "
            "mitteln die Asymmetrie des Rigs weg"
        )
    for pair, seen in directions.items():
        if len(seen) < 2:
            raise CheckFail(
                f"send_recv fuer {pair} nur in einer Richtung gemessen ({sorted(seen)})"
            )

    _check_loadable(payload)


def _check_loadable(payload: dict) -> None:
    add_repo_to_path()
    try:
        from sglang.srt.distributed.device_communicators.htccl_path_rates import (
            load_nccl_reference,
        )
    except Exception as exc:
        raise CheckStop(f"htccl_path_rates nicht importierbar: {exc}") from exc

    res = load_nccl_reference(payload)
    if res.errors:
        raise CheckFail(f"vom #279-Lader abgelehnt: {res.errors[0]}")
    if not res.profiles:
        raise CheckFail("der #279-Lader baut NULL Profile aus der Datei")
    measured = [p for p in res.profiles if p.provenance == "measured"]
    if not measured:
        raise CheckFail("kein einziges measured-Profil aus der Referenz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
