#!/usr/bin/env python3
"""gdn / KV-pressure ladder smoke against a live server.

SCOPE, stated up front so the result is not over-read: the pressure sensor is
NOT yet wired to the scheduler's occupancy counting -- that is an open item on
the Erg.-9 GPU restlist. This step therefore does not test the wiring, because
the wiring does not exist. It tests the three things that ARE testable today
and that every later wiring depends on:

  1. the ladder flags survive a real boot. Both ladders validate their
     arguments at argument time and both construct objects at startup; a rig
     where --gdn-state-set-ladder or --kv-pressure-ladder kills the server is
     a rig where no wiring work can even begin.
  2. the server still generates, and generates DETERMINISTICALLY. The flags
     are supposed to be inert until something moves; two identical greedy
     generations that differ would mean they are not inert.
  3. the sensor consumes a REAL occupancy series from this rig -- scraped from
     sglang:token_usage against the real token capacity -- and produces a
     reading with a verdict. Feeding it invented numbers proves nothing about
     the units, the denominator or the trend projection; feeding it the rig's
     own numbers proves all three.

Usage:
    python s09_sensor_smoke.py --port 30099 --out <dir>/sensor_smoke.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.request

TOKEN_USAGE_RE = re.compile(r"^sglang:token_usage\{[^}]*\}\s+([0-9.eE+-]+)\s*$", re.M)


def http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def http_post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=30099)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-seconds", type=int, default=45)
    ap.add_argument("--sample-interval", type=float, default=0.5)
    args = ap.parse_args()

    repo_python = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
    )
    sys.path.insert(0, repo_python)

    from sglang.srt.model_executor.kv_pressure_ladder import (
        KvPressureSensor,
        OccupancySample,
    )

    base = f"http://{args.host}:{args.port}"
    payload = {
        "kind": "sensor_smoke",
        "schema_version": 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "port": args.port,
    }

    # --- 1. the flags reached the scheduler ---------------------------------
    info = json.loads(http_get(f"{base}/get_server_info", timeout=30))
    flags = {
        key: info.get(key)
        for key in (
            "gdn_state_set_ladder",
            "gdn_state_set_ladder_hysteresis",
            "kv_pressure_ladder",
            "kv_pressure_pre_stage",
        )
    }
    payload["flags"] = flags
    memory_usage = info.get("memory_usage") or {}
    capacity = memory_usage.get("token_capacity")
    payload["token_capacity"] = capacity
    payload["memory_usage"] = memory_usage
    payload["offload_register_env"] = os.environ.get("SGLANG_OFFLOAD_REGISTER")

    if not capacity:
        payload["error"] = "get_server_info meldet keine token_capacity"
        _write(args.out, payload)
        return 1

    # --- 2. deterministic generation ----------------------------------------
    prompt = (
        "Count from one to twenty in words, then name the first five prime "
        "numbers in ascending order."
    )
    gens = []
    for _ in range(2):
        out = http_post(
            f"{base}/generate",
            {
                "text": prompt,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 96},
            },
            timeout=300,
        )
        gens.append(out.get("text", ""))
    payload["generation_lengths"] = [len(g) for g in gens]
    payload["generation_identical"] = gens[0] == gens[1]
    payload["generation_nonempty"] = all(bool(g.strip()) for g in gens)

    # --- 3. a REAL occupancy series into the sensor --------------------------
    # Load has to be generated, or the series is a flat line at zero and the
    # trend projection is untested. Several long generations in parallel is the
    # cheapest way to move the water level on a small model.
    stop = {"stop": False}

    def load_worker(idx: int):
        while not stop["stop"]:
            try:
                http_post(
                    f"{base}/generate",
                    {
                        "text": f"Write a long detailed technical description, part {idx}.",
                        "sampling_params": {"temperature": 0.0, "max_new_tokens": 512},
                    },
                    timeout=120,
                )
            except Exception:
                return

    workers = [
        threading.Thread(target=load_worker, args=(i,), daemon=True) for i in range(4)
    ]
    for w in workers:
        w.start()

    samples = []
    raw = []
    t0 = time.time()
    round_index = 0
    while time.time() - t0 < args.sample_seconds:
        try:
            metrics = http_get(f"{base}/metrics", timeout=5)
        except Exception as exc:
            payload.setdefault("metrics_errors", []).append(repr(exc))
            time.sleep(args.sample_interval)
            continue
        match = TOKEN_USAGE_RE.search(metrics)
        if match:
            usage = float(match.group(1))
            raw.append(usage)
            samples.append(
                OccupancySample(
                    round_index=round_index,
                    used_tokens=int(usage * capacity),
                    total_tokens=int(capacity),
                )
            )
            round_index += 1
        time.sleep(args.sample_interval)

    stop["stop"] = True
    for w in workers:
        w.join(timeout=10)

    payload["samples"] = len(samples)
    payload["occupancy_raw"] = [round(v, 5) for v in raw]
    payload["occupancy_max"] = round(max(raw), 5) if raw else None

    if samples:
        sensor = KvPressureSensor()
        sensor.observe_series(samples)
        reading = sensor.reading()
        payload["reading"] = {
            "samples": reading.samples,
            "occupancy": reading.occupancy,
            "trend_tokens_per_round": reading.trend_tokens_per_round,
            "rounds_to_exhaustion": reading.rounds_to_exhaustion,
            "verdict": reading.verdict,
            "stage_verdict": reading.stage_verdict,
            "reason": reading.reason,
        }
        # Same series twice must give the same reading: the trend is a
        # deterministic least-squares fit, and a sensor whose verdict wobbles
        # on identical input cannot be wired to anything.
        sensor2 = KvPressureSensor()
        sensor2.observe_series(samples)
        r2 = sensor2.reading()
        payload["reading_deterministic"] = (
            r2.verdict == reading.verdict
            and r2.stage_verdict == reading.stage_verdict
            and r2.occupancy == reading.occupancy
            and r2.trend_tokens_per_round == reading.trend_tokens_per_round
        )

    _write(args.out, payload)
    print(f"Flags: {flags}")
    print(
        f"Kapazitaet: {capacity} Token; {len(samples)} Belegungs-Samples, "
        f"max {payload['occupancy_max']}"
    )
    print(f"Generierung identisch: {payload['generation_identical']}")
    print(f"Sensor-Verdikt: {(payload.get('reading') or {}).get('verdict')}")
    print(f"geschrieben: {args.out}")
    return 0


def _write(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
