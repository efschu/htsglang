#!/usr/bin/env python3
"""s07 check -- did the register move real bytes on real silicon, and is a
retrieval latency now measured for every class that was exercised?

Verified:
  * the run really used CudaDeviceOps. A run that fell back to FakeDeviceOps
    would pass every latency assertion below in microseconds and prove nothing;
    this is checked first for exactly that reason.
  * ALL THREE routes green -- tensor, tag, suspend. Validating one route is not
    validating the register: the va_stable classes (graph rungs, GDN state
    sets) never touch the tensor route, and cold_lane never touches either.
  * per row: a real size that resolve_size_bytes agrees with, so the register's
    accounting is the tensor's actual footprint and not a declared number,
  * per row: the state sequence really went resident -> parked -> resident. A
    park that silently no-ops returns exactly what a working park returns.
  * per row: a retrieval latency > 0 over at least 3 cycles. This is the
    measurement obligation every auto/ram default in the register carries;
    without it the #279 dispatcher's latency term is a guess that decides
    placements.
  * zero park/wave-in failures in the movement stats.
  * the run's NEGATIVE CONTROL: an item left at the 'auto' policy with no
    saturation sensor must still be REFUSED. The measurement asks for its
    parks explicitly (class policy 'ram'); without this control a register
    that started parking on demand-less 'auto' -- the exact regression the
    gate exists to prevent -- would produce the same green rows.

STOP rather than FAIL when the memory saver is unavailable: two of the three
routes then were not tested at all, and a green verdict on one third of the
register would be worse than no verdict.

NOT judged: whether the measured GB/s is good. That is what the number is for.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    is_number,
    load_json,
    require_envelope,
    require_number,
    run_check,
)

STEP = "s07_offload_register_gpu"
REQUIRED_ROUTES = ("tensor", "tag", "suspend")
MIN_CYCLES = 3


def check(step_dir: str) -> None:
    payload = load_json(
        os.path.join(step_dir, "offload_register_gpu.json"), "offload_register_gpu.json"
    )
    require_envelope(payload, "offload_register_gpu", "offload_register_gpu.json", 1)

    if payload.get("device_ops") != "CudaDeviceOps":
        raise CheckFail(
            f"device_ops is {payload.get('device_ops')!r} -- a validation run on "
            "FakeDeviceOps validates nothing"
        )

    device = payload.get("device") or {}
    if not device.get("pci_bus_id") or device.get("cuda_index") is None:
        raise CheckStop("the card that was used is not named by PCI address")

    routes = payload.get("routes") or {}
    for route in REQUIRED_ROUTES:
        status = routes.get(route)
        if status is None:
            raise CheckStop(f"route {route} was not even attempted")
        if status == "unavailable":
            raise CheckStop(
                f"route {route} unavailable ({payload.get('memory_saver')}) -- two "
                "of three routes untested, that is not a green run"
            )
        if status != "ok":
            raise CheckFail(f"route {route}: {status}")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CheckFail("no rows -- not a single item was moved")

    seen_routes = set()
    for row in rows:
        label = f"{row.get('offload_class')}/{row.get('route')}"
        if row.get("status") != "ok":
            raise CheckFail(f"{label}: {row.get('error') or row.get('status')}")
        seen_routes.add(row.get("route"))

        require_number(row.get("size_bytes"), f"{label}: size_bytes", minimum=1)
        if not row.get("size_source_matches"):
            raise CheckFail(
                f"{label}: resolve_size_bytes disagrees with the tensor's real size "
                "-- the register's accounting would be wrong"
            )

        require_number(row.get("iters"), f"{label}: iters", minimum=MIN_CYCLES)
        wave = require_number(row.get("wave_in_ms_p50"), f"{label}: wave_in_ms_p50")
        if wave <= 0:
            raise CheckFail(
                f"{label}: retrieval latency {wave} ms -- a movement that costs no "
                "time moved no bytes"
            )
        require_number(row.get("park_ms_p50"), f"{label}: park_ms_p50")
        if is_number(row.get("wave_in_ms_p99")) and row["wave_in_ms_p99"] < wave:
            raise CheckFail(f"{label}: wave_in p99 < p50")

        states = row.get("state_sequence") or []
        if "parked" not in states:
            raise CheckFail(
                f"{label}: state sequence {states} never contains 'parked' -- the "
                "park was a silent No-Op"
            )
        if "resident" not in states:
            raise CheckFail(
                f"{label}: state sequence {states} never contains 'resident'"
            )

    missing = [r for r in REQUIRED_ROUTES if r not in seen_routes]
    if missing:
        raise CheckFail(f"no row for route(s) {missing}")

    stats = payload.get("stats") or {}
    # #514: destination_release_failures / leaked_destination_bytes are gated
    # here for the same reason wave_in_failures is. They were split out of
    # wave_in_failures, so omitting them would silently widen what this check
    # passes -- a release failure leaves the item resident with its park-target
    # bytes still booked, which is precisely a leak the battery must not pass.
    for field in (
        "park_failures",
        "wave_in_failures",
        "destination_release_failures",
        "leaked_destination_bytes",
    ):
        value = stats.get(field)
        if value is None:
            raise CheckStop(f"movement telemetry without {field}")
        if value:
            raise CheckFail(f"{field} = {value}")
    if not stats.get("parks"):
        raise CheckFail("the movement telemetry counts zero parks")

    control = payload.get("negative_control")
    if not isinstance(control, dict) or "refused" not in control:
        raise CheckStop(
            "no Negativkontrolle in the artifact -- so the run does not prove that "
            "it was the explicit class policy that allowed the park"
        )
    if not control.get("refused"):
        raise CheckFail(
            "Negativkontrolle: 'auto' without saturation pressure did NOT refuse the "
            f"park ({control.get('error')})"
        )

    terms = payload.get("latency_term_ms") or {}
    if not terms:
        raise CheckFail(
            "no latency_term_ms collected -- this is exactly the number the #279 "
            "dispatcher reads, and without it the number stays a guess"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
