#!/usr/bin/env python3
"""s09 check -- ladder flags on a live boot, and a sensor fed with this rig's
own occupancy.

Verified:
  * all four ladder flags are echoed back by the running server. Argument-time
    validation is already CPU-tested; what is new here is that the values
    survived into the scheduler rather than being parsed and dropped,
  * generation happened and both greedy runs produced IDENTICAL output. The
    ladders are supposed to be inert until something moves; two identical
    greedy generations that differ would say they are not,
  * a real occupancy series was collected -- enough samples for a trend, and a
    maximum above zero. A flat zero series would mean the load never reached
    the pool and the sensor was fed nothing,
  * the sensor produced a reading with a verdict, a finite occupancy and a
    trend, and produced the SAME reading twice from the same series. The trend
    is a deterministic least-squares fit; a wobbling verdict could not be
    wired to a flip decision,
  * no OOM / NCCL / traceback in the server log.

NOT tested, and deliberately not claimed: the wiring of the sensor to the
scheduler's occupancy counting, and any actual movement of state sets or KV.
Neither exists yet -- they are the open items this step's numbers are meant to
inform.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    classify_missing_result,
    is_number,
    load_json,
    require_envelope,
    run_check,
    scan_log_for_fatals,
)

STEP = "s09_sensor_smoke"
MIN_SAMPLES = 20
REQUIRED_FLAGS = (
    "gdn_state_set_ladder",
    "gdn_state_set_ladder_hysteresis",
    "kv_pressure_ladder",
    "kv_pressure_pre_stage",
)


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "sensor_smoke.json")
    classify_missing_result(step_dir, "sensor_smoke", path, "sensor_smoke.json")
    payload = load_json(path, "sensor_smoke.json")
    require_envelope(payload, "sensor_smoke", "sensor_smoke.json", 1)

    if payload.get("error"):
        raise CheckFail(f"Sonde meldet: {payload['error']}")

    flags = payload.get("flags") or {}
    for flag in REQUIRED_FLAGS:
        if flag not in flags:
            raise CheckFail(f"get_server_info kennt {flag} nicht")
        if flags[flag] in (None, "", False) and flag != "kv_pressure_pre_stage":
            raise CheckFail(
                f"{flag} kam als {flags[flag]!r} zurueck -- der Wert hat den "
                "Scheduler nicht erreicht"
            )
    if flags.get("kv_pressure_pre_stage") is not True:
        raise CheckFail(
            f"kv_pressure_pre_stage ist {flags.get('kv_pressure_pre_stage')!r}, "
            "obwohl das Flag gesetzt wurde"
        )

    if not payload.get("generation_nonempty"):
        raise CheckFail("mindestens eine Generierung war leer")
    if not payload.get("generation_identical"):
        raise CheckFail(
            "zwei identische Greedy-Generierungen unterscheiden sich -- die "
            "Leiter-Flags sind nicht inert"
        )

    samples = payload.get("samples") or 0
    if samples < MIN_SAMPLES:
        raise CheckFail(
            f"nur {samples} Belegungs-Samples (mindestens {MIN_SAMPLES} noetig, "
            "sonst gibt es keinen Trend zu fitten)"
        )
    occ_max = payload.get("occupancy_max")
    if not is_number(occ_max) or occ_max <= 0:
        raise CheckFail(
            f"maximale Belegung {occ_max!r} -- die Last hat den Pool nie erreicht, "
            "der Sensor bekam eine Nulllinie"
        )

    reading = payload.get("reading")
    if not isinstance(reading, dict):
        raise CheckFail("der Sensor hat keine Lesung geliefert")
    if not reading.get("verdict"):
        raise CheckFail("die Lesung hat kein Verdikt")
    if not is_number(reading.get("occupancy")):
        raise CheckFail(f"Lesung ohne Belegung: {reading.get('occupancy')!r}")
    if reading.get("trend_tokens_per_round") is None:
        raise CheckFail(
            "kein Trend in der Lesung -- die projizierte Erschoepfung ist der "
            "eigentliche Sensor, der Momentwert nicht"
        )
    if payload.get("reading_deterministic") is not True:
        raise CheckFail(
            "dieselbe Reihe ergibt zweimal verschiedene Lesungen -- der "
            "Kleinste-Quadrate-Trend ist nicht deterministisch"
        )

    fatal = scan_log_for_fatals(
        os.path.join(step_dir, "server.log"), "sensor_smoke: server.log"
    )
    if fatal:
        raise CheckFail(f"Fatal im Serverlog -- {fatal}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
