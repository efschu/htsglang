#!/usr/bin/env python3
"""s08 check -- did the measured tables reach the dispatcher, and does hard
rule 1 still hold now that measured candidates exist?

The first assertion is against the failure mode this whole step exists for:
load_rate_tables degrades missing sources to placeholders LOUDLY but without
error, and hard rule 1 then keeps every class on the status quo. That is
correct at runtime and indistinguishable from success here. So:

  * all three sources must exist and have been consumed,
  * loader errors must be empty -- a partially usable artifact is fine at
    runtime and not fine as a measurement result,
  * at least one measured profile, and at least one measured APERTURE. Zero
    apertures means the capability matrix contributed nothing and every direct
    path is unbounded, which is the silent version of the bug this rule guards.

The second assertion is the one that could not be made before this run:

  * the deliberately placeholder-contaminated class decided STATUS_QUO
    everywhere, including for protected requests,
  * neither the saturation sensor nor the offload latency term was consulted
    for it. The probe hooks RAISE when touched, so "not consulted" is proven
    rather than inferred,
  * the fully measured class decided a real path at least once -- otherwise
    the tables were loaded and then ignored, and the step proved nothing in
    the other direction.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import CheckFail, CheckStop, load_json, require_envelope, run_check  # noqa: E402

STEP = "s08_dispatcher_tables"
REQUIRED_SOURCES = ("capability_matrix", "d2d_bench", "nccl_reference")


def check(step_dir: str) -> None:
    payload = load_json(
        os.path.join(step_dir, "dispatcher_tables.json"), "dispatcher_tables.json"
    )
    require_envelope(payload, "dispatcher_tables", "dispatcher_tables.json", 1)

    sources = payload.get("sources") or {}
    for name in REQUIRED_SOURCES:
        entry = sources.get(name) or {}
        if not entry.get("exists"):
            raise CheckStop(
                f"Quelle {name} fehlt ({entry.get('path')}) -- ohne sie ist der "
                "Lauf nicht von einem Lauf ganz ohne Karten unterscheidbar"
            )

    errors = payload.get("errors") or []
    if errors:
        raise CheckFail(f"{len(errors)} Lader-Fehler, erster: {errors[0]}")

    profiles = payload.get("profiles") or {}
    if not profiles.get("measured"):
        raise CheckFail(
            "null measured-Profile -- die frischen Quellen sind still zu "
            "Platzhaltern degradiert"
        )
    # Apertures exist only where peer access does. On a rig without P2P their
    # absence is the correct result, not a gap.
    if payload.get("peer_capable_pairs") and not payload.get("apertures"):
        raise CheckFail(
            "null effektive Aperturen trotz peer-faehiger Paare -- die capability "
            "matrix hat nichts beigetragen und jeder direkte Pfad ist unbegrenzt"
        )

    violations = payload.get("neutrality_violations") or []
    if violations:
        first = violations[0]
        raise CheckFail(
            f"{len(violations)} Verstoss/Verstoesse gegen die Platzhalter-"
            f"Neutralitaet, erster: {first.get('violation')}"
        )
    if payload.get("sensor_consulted_under_placeholder"):
        raise CheckFail(
            "der Saettigungs-Sensor wurde unter einem Platzhalter-Kandidaten "
            "konsultiert -- harte Regel 1 verlangt, dass er davor gar nicht "
            "befragt wird"
        )
    if payload.get("latency_consulted_under_placeholder"):
        raise CheckFail(
            "der Offload-Latenz-Term wurde unter einem Platzhalter-Kandidaten "
            "konsultiert"
        )

    decisions = payload.get("decisions") or []
    if not decisions:
        raise CheckFail("keine Entscheide protokolliert")
    contaminated = [
        d for d in decisions if d.get("message_class") == "battery_contaminated"
    ]
    if not contaminated:
        raise CheckFail("die kontaminierte Klasse wurde gar nicht befragt")
    for d in contaminated:
        if not d.get("status_quo"):
            raise CheckFail(
                f"kontaminierte Klasse entschied {d.get('path')!r} bei "
                f"{d.get('nbytes')} B (protected={d.get('protected')})"
            )

    if not payload.get("measured_class_decided_paths"):
        raise CheckFail(
            "die vollstaendig gemessene Klasse entschied nirgends einen echten "
            "Pfad -- die Tabellen wurden geladen und dann ignoriert"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
