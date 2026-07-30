#!/usr/bin/env python3
"""s11 -- compose bar1_e2e.json from what the host run left behind.

The step script talks to the host; this reads the files it brought back and
turns them into ONE artifact with a decidable content. Nothing here touches a
card, a socket or ssh, which is what makes it testable against fixtures.

Three extractions carry the step:

  * the graph gate. bar1_graph_check.py prints one BESTANDEN/GEFALLEN line per
    case with a [Gate]/[Info] marker and exits 0 only when every gate case
    passed. Both are recorded: the exit code alone would hide WHICH case fell.
  * per-group attainment. parallel_state logs "HTCCL enabled for group '<x>':
    angefordert=<a>, ERREICHT=<e>" on success and the same pair as a WARNING on
    fallback. The requested name is worthless -- it says bar1 either way. Only
    ERREICHT counts, and it counts PER GROUP: with SGLANG_UNEVEN_DCP=1 there
    are two (tp:0, dcp:0), and a run where one of them fell back to gloo is a
    mixed measurement, not a bar1 measurement.
  * the coverage bolt. htccl._select raises rather than falling back to the
    host-staged gloo level during a graph capture. It is extracted as its own
    field, with op and size, because it is the expected failure of the current
    integration and needs to be distinguishable from any other crash.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

KIND = "bar1_e2e"
SCHEMA_VERSION = 1

RE_GROUP = re.compile(
    r"group '(?P<gruppe>[^']+)': angefordert=(?P<angefordert>[^,\s]+),\s*"
    r"ERREICHT=(?P<erreicht>[A-Za-z0-9_\-]+)"
)
RE_KASSE = re.compile(r"BAR1-Kasse dieser Karte nach Gruppe '(?P<gruppe>[^']+)'")
RE_AUFBAU = re.compile(r"HTCCL-BAR1: Aufbau in\s+(?P<ms>[0-9.]+)\s*ms")
RE_RIEGEL = re.compile(
    r"HTCCL: '(?P<op>[A-Za-z0-9_]+)' mit (?P<bytes>\d+) Byte waehrend einer "
    r"CUDA-Graph-Aufzeichnung"
)
RE_GATE_CASE = re.compile(
    r"^\s*(?P<marke>BESTANDEN|GEFALLEN)\s*\[(?P<art>Gate|Info)\]\s*(?P<name>\S+)"
)

FATAL_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "NCCL error",
    "Watchdog caught collective operation timeout",
    # A boot killed by a plain traceback -- no OOM, no NCCL error -- is just as
    # dead. The shell already harvests this marker (s11_bar1_e2e.sh) and
    # check_common.FATAL_MARKERS carries it for every other step.
    "Traceback (most recent call last)",
)


def read_lines(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def parse_graph_check(step_dir: str) -> dict:
    lines = read_lines(os.path.join(step_dir, "graph_check.txt"))
    rc_raw = read_lines(os.path.join(step_dir, "graph_check_rc.txt"))
    try:
        rc = int(rc_raw[0].strip()) if rc_raw else None
    except ValueError:
        rc = None
    cases = []
    for line in lines:
        m = RE_GATE_CASE.match(line)
        if m:
            cases.append(
                {
                    "name": m.group("name"),
                    "gate": m.group("art") == "Gate",
                    "ok": m.group("marke") == "BESTANDEN",
                }
            )
    gates = [c for c in cases if c["gate"]]
    fallen = [c["name"] for c in gates if not c["ok"]]
    return {
        "rc": rc,
        "cases": len(cases),
        "gate_cases": len(gates),
        "gefallen": fallen,
        "alle_bestanden": bool(gates) and not fallen and rc == 0,
        "zusammenfassung_vorhanden": any("Zusammenfassung" in line for line in lines),
    }


def parse_log_evidence(step_dir: str) -> dict:
    lines = read_lines(os.path.join(step_dir, "htccl_lines.txt"))
    gruppen: dict = {}
    kasse_gruppen = []
    aufbau = []
    riegel = None
    fatal = None
    for line in lines:
        m = RE_GROUP.search(line)
        if m:
            gruppen[m.group("gruppe")] = {
                "gruppe": m.group("gruppe"),
                "angefordert": m.group("angefordert"),
                "erreicht": m.group("erreicht"),
            }
        m = RE_KASSE.search(line)
        if m and m.group("gruppe") not in kasse_gruppen:
            kasse_gruppen.append(m.group("gruppe"))
        m = RE_AUFBAU.search(line)
        if m:
            aufbau.append(float(m.group("ms")))
        m = RE_RIEGEL.search(line)
        if m and riegel is None:
            riegel = {
                "op": m.group("op"),
                "bytes": int(m.group("bytes")),
                "zeile": " ".join(line.split())[:300],
            }
        if fatal is None:
            for marker in FATAL_MARKERS:
                if marker in line:
                    fatal = " ".join(line.split())[:300]
                    break
    return {
        "gruppen": sorted(gruppen.values(), key=lambda g: g["gruppe"]),
        "aufbau_gruppen": kasse_gruppen,
        "aufbau_lines": len(aufbau),
        "aufbau_ms": aufbau,
        "riegel": riegel,
        "fatal": fatal,
    }


def parse_smoke(step_dir: str) -> dict:
    """Coherence, mechanically. The prompt asks for 1..20; how many of those
    numbers appear IN ORDER is a number, not an opinion."""
    path = os.path.join(step_dir, "smoke.json")
    out: dict = {
        "vorhanden": os.path.exists(path),
        "content_prefix": None,
        "spec_accept_length": None,
        "zahlen_in_folge": 0,
        "kohaerent": False,
        "error": None,
    }
    if not out["vorhanden"]:
        return out
    try:
        with open(path, errors="replace") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        out["error"] = f"smoke.json nicht lesbar: {exc}"
        return out
    if not isinstance(payload, dict):
        out["error"] = "smoke.json ist kein JSON-Objekt"
        return out
    if payload.get("error") or payload.get("object") == "error":
        out["error"] = " ".join(str(payload.get("error") or payload).split())[:200]
        return out
    try:
        choice = payload["choices"][0]
        text = choice["message"]["content"] or ""
        meta = choice.get("meta_info") or {}
    except (KeyError, IndexError, TypeError) as exc:
        out["error"] = f"Antwort ohne choices[0].message.content ({exc})"
        return out
    out["content_prefix"] = text[:300]
    accept = meta.get("spec_accept_length")
    if accept is None:
        accept = (payload.get("usage") or {}).get("spec_accept_length")
    out["spec_accept_length"] = accept

    pos = 0
    hits = 0
    for n in range(1, 21):
        found = text.find(str(n), pos)
        if found < 0:
            continue
        hits += 1
        pos = found + len(str(n))
    out["zahlen_in_folge"] = hits
    out["kohaerent"] = hits >= 15
    return out


def compose(step_dir: str, port: int, host_log: str) -> dict:
    unreachable = os.path.exists(os.path.join(step_dir, "host_unreachable.txt"))
    missing_integration = os.path.exists(
        os.path.join(step_dir, "integration_missing.txt")
    )
    blocked_lines = read_lines(os.path.join(step_dir, "blocked.txt"))
    payload: dict = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "host": os.environ.get("BAR1_HOST", ""),
        "reachable": not unreachable,
        "integration_present": not missing_integration,
        # A step that never got the cards must not be diagnosed through the
        # empty artifacts it left behind.
        "blocked": " ".join(" ".join(blocked_lines).split())[:200] or None,
        "port": port,
        "server_log_remote": host_log,
        "transport_angefordert": "bar1",
        "graph_freigabe": True,
        "uneven_dcp": True,
        "graph_check": parse_graph_check(step_dir),
        "smoke": parse_smoke(step_dir),
    }
    payload.update(parse_log_evidence(step_dir))
    payload["gruppen_bar1"] = sorted(
        g["gruppe"] for g in payload["gruppen"] if g["erreicht"] == "bar1"
    )
    payload["gruppen_ausgewichen"] = sorted(
        g["gruppe"] for g in payload["gruppen"] if g["erreicht"] != g["angefordert"]
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--host-log", default="")
    args = ap.parse_args()

    payload = compose(args.step_dir, args.port, args.host_log)
    out = os.path.join(args.step_dir, "bar1_e2e.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"bar1_e2e.json geschrieben: Gruppen bar1={payload['gruppen_bar1']}, "
        f"ausgewichen={payload['gruppen_ausgewichen']}, "
        f"Aufbau-Zeilen={payload['aufbau_lines']}, "
        f"Riegel={'ja' if payload['riegel'] else 'nein'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
