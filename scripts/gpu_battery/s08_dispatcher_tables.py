#!/usr/bin/env python3
"""Feed the measured rate tables into the #279 dispatcher and re-check that
placeholder neutrality still holds.

This step is CPU-only. It needs no card -- it needs the ARTIFACTS of s01 and
s06, which is why it can be resumed on its own long after the boots are done.

TWO QUESTIONS, and the second is the important one:

1. Do the freshly measured sources actually load? load_rate_tables consumes
   only effective/measured values and is written to survive missing sources by
   degrading to placeholders. That is the right behaviour at runtime and the
   wrong outcome here: a run where all three sources silently failed to load
   looks identical to a run with no cards at all, except that hard rule 1
   keeps everything on the status quo and nobody notices.

2. Does hard rule 1 still hold once real profiles exist? Placeholder
   neutrality has only ever been tested with placeholders. The first time a
   class has measured candidates is the first time the rule can be violated,
   and the first time the sensor and the latency term are consulted at all.
   So this run keeps a deliberately placeholder-contaminated class alongside
   the measured one and asserts:

     * the contaminated class still decides STATUS_QUO,
     * the saturation sensor and the offload latency term are NEVER consulted
       for it -- the hooks installed here RAISE if touched, which is the only
       way to prove "not consulted" rather than "consulted and ignored",
     * a fully measured class does decide a real path, otherwise the tables
       were loaded for nothing.

Usage:
    python s08_dispatcher_tables.py --p2p-dir <s01 results> \\
        --nccl <s06 nccl_reference.json> --out <dir>/dispatcher_tables.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

MIB = 1024 * 1024
SIZES = (64 * 1024, 1 * MIB, 16 * MIB, 256 * MIB)

MEASURED_CLASS = "battery_measured"
CONTAMINATED_CLASS = "battery_contaminated"


class ConsultedError(AssertionError):
    """Raised by the trap hooks. Being raised at all is the finding."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p2p-dir", required=True)
    ap.add_argument("--nccl", required=True)
    ap.add_argument("--gdr-tsv", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo_python = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
    )
    sys.path.insert(0, repo_python)

    from sglang.srt.distributed.device_communicators.barlink_path_dispatcher import (
        PROVENANCE_MEASURED,
        PROVENANCE_PLACEHOLDER,
        STATUS_QUO,
        DispatchRequest,
        PathDispatcher,
        PathProfile,
    )
    from sglang.srt.distributed.device_communicators.barlink_path_rates import (
        capability_matrix_rows,
        load_rate_tables,
    )

    cap = os.path.join(args.p2p_dir, "capability_matrix.json")
    d2d = os.path.join(args.p2p_dir, "d2d_bench.json")

    # How many directed pairs can do peer access at all. Without a single one
    # there are legitimately no apertures, and demanding some would turn "this
    # rig has no P2P" -- a fully recorded outcome -- into a failure. The row key
    # comes from the loader so this counter cannot drift away from the producer
    # and disarm the aperture gate in check_s08 by always counting zero.
    peer_pairs = 0
    if os.path.exists(cap):
        try:
            with open(cap) as f:
                _, cap_rows = capability_matrix_rows(json.load(f))
            peer_pairs = sum(1 for row in cap_rows if row.get("can_access_peer"))
        except (OSError, json.JSONDecodeError):
            peer_pairs = 0

    result = load_rate_tables(
        p2p_capability_json=cap,
        p2p_d2d_json=d2d,
        gdr_tsv=args.gdr_tsv,
        nccl_reference_json=args.nccl,
    )

    measured = [p for p in result.profiles if p.provenance == PROVENANCE_MEASURED]
    placeholders = [
        p for p in result.profiles if p.provenance == PROVENANCE_PLACEHOLDER
    ]

    payload = {
        "kind": "dispatcher_tables",
        "schema_version": 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "sources": {
            "capability_matrix": {"path": cap, "exists": os.path.exists(cap)},
            "d2d_bench": {"path": d2d, "exists": os.path.exists(d2d)},
            "nccl_reference": {"path": args.nccl, "exists": os.path.exists(args.nccl)},
            "gdr_matrix": {
                "path": args.gdr_tsv,
                "exists": bool(args.gdr_tsv and os.path.exists(args.gdr_tsv)),
            },
        },
        "profiles": {
            "measured": len(measured),
            "placeholder": len(placeholders),
            "measured_names": sorted(p.name for p in measured)[:80],
        },
        "peer_capable_pairs": peer_pairs,
        "apertures": {f"{s}->{d}": v for (s, d), v in sorted(result.apertures.items())},
        "errors": list(result.errors),
        "skipped": list(result.skipped)[:40],
    }

    dispatcher = PathDispatcher()
    for profile in measured:
        dispatcher.register_path(profile, message_classes=(MEASURED_CLASS,))
    # The contaminated class gets the same measured profiles PLUS one
    # placeholder. Hard rule 1 says one placeholder candidate is enough to keep
    # the whole class on the status quo, and this is where that is retested now
    # that measured candidates exist to be tempted by.
    for profile in measured[:3]:
        dispatcher.register_path(profile, message_classes=(CONTAMINATED_CLASS,))
    dispatcher.register_path(
        PathProfile(
            name="battery:deliberate_placeholder",
            provenance=PROVENANCE_PLACEHOLDER,
            source="gpu battery neutrality probe",
        ),
        message_classes=(CONTAMINATED_CLASS,),
    )

    trap = {"sensor": False, "latency": False}

    def sensor(path_name):
        trap["sensor"] = True
        raise ConsultedError(f"saturation sensor consulted for {path_name}")

    def latency_term(offload_class):
        trap["latency"] = True
        raise ConsultedError(f"latency term consulted for {offload_class}")

    decisions = []
    violations = []

    # Contaminated class first, with the traps armed: if either hook fires, the
    # rule was broken before any cost was even compared.
    dispatcher.set_saturation_sensor(sensor)
    dispatcher.set_offload_latency_term(latency_term)
    for nbytes in SIZES:
        for protected in (False, True):
            dispatcher.round_boundary()
            try:
                decision = dispatcher.decide(
                    DispatchRequest(
                        message_class=CONTAMINATED_CLASS,
                        nbytes=nbytes,
                        lane="battery",
                        protected=protected,
                    )
                )
            except ConsultedError as exc:
                violations.append(
                    {
                        "message_class": CONTAMINATED_CLASS,
                        "nbytes": nbytes,
                        "protected": protected,
                        "violation": str(exc),
                    }
                )
                continue
            row = {
                "message_class": CONTAMINATED_CLASS,
                "nbytes": nbytes,
                "protected": protected,
                "path": decision.path,
                "status_quo": decision.status_quo,
                "overflowed": decision.overflowed,
                "reason": decision.reason,
            }
            decisions.append(row)
            if decision.path != STATUS_QUO or not decision.status_quo:
                violations.append(
                    {
                        **row,
                        "violation": "class with a Platzhalter candidate did "
                        "NOT decide the status quo",
                    }
                )

    # Measured class: hooks disarmed, because here consulting them is correct.
    dispatcher.set_saturation_sensor(lambda name: 0.0)
    dispatcher.set_offload_latency_term(lambda cls: 0.0)
    measured_paths = set()
    for nbytes in SIZES:
        for protected in (False, True):
            dispatcher.round_boundary()
            decision = dispatcher.decide(
                DispatchRequest(
                    message_class=MEASURED_CLASS,
                    nbytes=nbytes,
                    lane="battery",
                    protected=protected,
                )
            )
            decisions.append(
                {
                    "message_class": MEASURED_CLASS,
                    "nbytes": nbytes,
                    "protected": protected,
                    "path": decision.path,
                    "status_quo": decision.status_quo,
                    "overflowed": decision.overflowed,
                    "reason": decision.reason,
                }
            )
            if not decision.status_quo:
                measured_paths.add(decision.path)

    payload["decisions"] = decisions
    payload["neutrality_violations"] = violations
    payload["sensor_consulted_under_placeholder"] = trap["sensor"]
    payload["latency_consulted_under_placeholder"] = trap["latency"]
    payload["measured_class_decided_paths"] = sorted(measured_paths)

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"profiles: {len(measured)} measured, {len(placeholders)} placeholder")
    print(f"apertures: {len(result.apertures)}")
    print(f"loader errors: {len(result.errors)}")
    print(f"neutrality violations: {len(violations)}")
    print(f"measured class decided paths: {sorted(measured_paths) or 'none'}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
