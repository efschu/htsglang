"""Kartenplaner (Item 510): Brücke zum Planer.

* ``gate``       im Server-Prozess, reine Funktionen aus card_identity/topology (kartenplan_gate).
* ``rederive``   Nachrechnen der Budgets mit ``launcher.budgets_from_dc`` in einem KINDPROZESS mit der
                 sglang-Umgebung (der Import zieht torch, ~1 GB und ~10 s; der Dashboard-Dienst hat
                 MemoryMax=1G und läuft ohne torch).  Gebraucht beim Bau der Records und im Test, nicht
                 bei jeder Seitenabfrage: das Ergebnis steht dann als ``planer_nachrechnung`` im Record.

Nie wird der Launcher als Prozess gestartet und nie ``launcher.main`` gerufen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

from rigdash import kartenplan_catalog as CAT

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
DEFAULT_PYTHON = "/spinning/htsglang-gpu/.venv/bin/python"



def _cc_of(name: str):
    for e in CAT.CATALOG:
        if e["nvml_name"] == name or name.endswith(e["name"]):
            return list(e["cc"])
    return None


def _cards_of(rec: dict) -> List[dict]:
    plan = rec.get("vram_plan")
    out = []
    if plan:
        by_uuid = {}
        for ln in rec.get("budget_lines", []):
            by_uuid[ln["ordinal"]] = ln["name"]
        for i, c in enumerate(plan["cards"]):
            name = c["class"]
            out.append({"nvml_index": c["nvml"], "uuid": c["uuid"], "name": name, "total_mib": c["total_mib"],
                        "reserved_mib": c["driver_reserved_mib"], "cc": _cc_of(name)})
    elif rec.get("boot_json", {}).get("cards"):
        for c in rec["boot_json"]["cards"]:
            out.append({"nvml_index": c["nvml_index"], "uuid": c["uuid"], "name": c["name"], "total_mib": c["total_mib"],
                        "reserved_mib": c["reserved_mib"], "cc": _cc_of(c["name"])})
    return out


def _p_rows(rec: dict, cards: List[dict]) -> List[dict]:
    """Die P-Budgetzeilen der LETZTEN Rechnung (Front-Log, beim Bau in Zahlen zerlegt: ``parsed``)."""
    lines = [ln for ln in rec.get("budget_lines", []) if ln["group"] == "P" and ln["label"] == "P"]
    if len(lines) < len(cards):
        return []
    last = lines[-len(cards):]
    if [ln["ordinal"] for ln in last] != list(range(len(cards))):
        return []
    P = [ln["parsed"] for ln in last]
    row = {"id": "%s:P" % rec["id"], "label": "P", "cards": cards, "dormant_other": [p["dormant"] for p in P],
           "overshoot": [p["over"] or 0 for p in P], "overshoot_prov": next((p["over_prov"] for p in P if p["over_prov"]), ""),
           "user_reserve": [p["reserve"] or 0 for p in P], "expected_mib": [ln["budget_mib"] for ln in last]}
    if any(p["l15"] is not None for p in P):
        row["l15"] = [p["l15"] or 0 for p in P]
    return [row]


def _d_rows(rec: dict, cards: List[dict]) -> List[dict]:
    plan = rec.get("vram_plan")
    if not plan:
        lines = [ln for ln in rec.get("budget_lines", []) if ln["group"] == "D" and ln["label"] == "D"]
        if len(lines) < len(cards):
            return []
        last = lines[-len(cards):]
        P = [ln["parsed"] for ln in last]
        row = {"id": "%s:D" % rec["id"], "label": "D", "cards": cards, "dormant_other": [p["dormant"] for p in P],
               "charge_carve": any(p["carve"] for p in P), "expected_mib": [ln["budget_mib"] for ln in last],
               "growth": [p["growth"] or 0 for p in P], "user_reserve": [p["reserve"] or 0 for p in P]}
        if any(p["booked"] is not None for p in P):
            row["booked_rest"] = [p["booked"] for p in P]
            row["booked_rest_prov"] = P[0]["booked_prov"] or ""
        else:
            row["overshoot"] = [p["over"] or 0 for p in P]
            if any(p["awake_rest"] is not None for p in P):
                row["awake_rest"] = [p["awake_rest"] for p in P]
        return [row]
    terms = plan["budget_terms"].get("D") or next((v for k, v in plan["budget_terms"].items() if k.startswith("D(")), None)
    if not terms:
        return []
    order = [c["uuid"] for c in plan["cards"]]
    t = [terms[u] for u in order]
    growth = [int(x.get("growth") or 0) for x in t]
    dorm = [int(x["dormant"]) - g for x, g in zip(t, growth)]
    row = {"id": "%s:D" % rec["id"], "label": "D", "cards": cards, "dormant_other": dorm, "growth": growth,
           "growth_prov": "WEG2-DORMANT-SERVED record", "charge_carve": any(int(x["carve"]) > 0 for x in t),
           "expected_mib": [int(x["budget"]) for x in t]}
    src = [str(x.get("awake_source", "")) for x in t]
    if all(s.startswith("RECORD") for s in src):
        row["booked_rest"] = [int(x["awake"]) for x in t]
        row["booked_rest_prov"] = src[0].replace("RECORD D_AWAKE_REST_BOOKED_MIB", "").strip()
    else:
        rest, over = [], []
        for x, s in zip(t, src):
            if s.startswith("D_AWAKE_REST_MIB"):
                rest.append(int(x["awake"]))
                over.append(0)
            else:
                rest.append(None)
                over.append(max(0, int(x["awake"]) - 404))
        row["awake_rest"] = rest
        row["awake_rest_prov"] = next((s.replace("D_AWAKE_REST_MIB", "").strip() for s in src if s.startswith("D_AWAKE_REST_MIB")), "")
        row["overshoot"] = over
    ur = next((o["value"] for o in plan.get("overrides", []) if o["key"] == "--user-reserve-mib"), None)
    if ur:
        row["user_reserve"] = [int(v) for v in str(ur).split(",")]
    return [row]


def rederive_rows(rec: dict) -> List[dict]:
    cards = _cards_of(rec)
    if not cards:
        return []
    return _d_rows(rec, cards) + _p_rows(rec, cards)


def call(op: str, payload: dict, *, tree_python: str, python: str = DEFAULT_PYTHON, timeout: int = 120) -> dict:
    """Kindprozess mit der sglang-Umgebung.  Kein GPU-Zugriff (CUDA_VISIBLE_DEVICES leer)."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": tree_python, "CUDA_VISIBLE_DEVICES": "",
           "HOME": os.environ.get("KARTENPLAN_RIG_HOME", "/root"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONWARNINGS": "ignore"}
    req = dict(payload, op=op)
    p = subprocess.run([python, RUNNER], input=json.dumps(req), capture_output=True, text=True, env=env, timeout=timeout)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        return {"ok": False, "error": "Planer-Kindprozess ohne gültige Antwort (rc %s): %s" % (p.returncode, p.stderr[-400:])}
    return out


def verify_record(rec: dict, tree_python: str, python: str = DEFAULT_PYTHON) -> dict:
    """Records gegen den Planer nachrechnen: Ergebnis je Zeile {got, expected, match}."""
    rows = rederive_rows(rec)
    if not rows:
        return {"ok": False, "why": "Record enthält keine Budgeteingaben (kein vram_plan, keine Front-Log-Zeilen)"}
    out = call("rederive", {"rows": rows}, tree_python=tree_python, python=python)
    if not out.get("ok"):
        return out
    res = []
    for r in out["result"]:
        got, exp = r["got_mib"], r["expected_mib"]
        res.append({"id": r["id"], "label": r["label"], "got_mib": got, "expected_mib": exp, "match": got == exp,
                    "error": r["error"], "lines": r["lines"]})
    return {"ok": True, "tree_python": tree_python, "rows": res, "all_match": all(x["match"] for x in res)}


def main(argv=None) -> int:
    """``python -m rigdash.kartenplan_bridge --trees-root /tmp/kp_trees`` : jeden Record gegen den Planer SEINER Revision
    nachrechnen und das Ergebnis als ``planer_nachrechnung`` in den Record schreiben (Schreibtisch, kein Dienst)."""
    import argparse
    import time

    from . import records as R

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--trees-root", required=True, help="Ordner mit <rev>/python/sglang (git archive der Boot-Revisionen)")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("ids", nargs="*")
    ns = ap.parse_args(argv)
    bad = 0
    for spec in R.SPECS:
        if ns.ids and spec["id"] not in ns.ids:
            continue
        rec = R.load(spec["id"])
        tree = os.path.join(ns.trees_root, spec["rev"], "python")
        res = verify_record(rec, tree, ns.python)
        res["checked_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        res["rev"] = spec["rev"]
        res.pop("tree_python", None)
        rec["planer_nachrechnung"] = res
        R.save(rec)
        print(spec["id"], "rev", spec["rev"], "->", "ALLE GLEICH" if res.get("all_match") else "ABWEICHUNG/FEHLER: %s" % (res.get("error") or res.get("why") or res.get("rows")))
        for r in res.get("rows", []):
            print("   ", r["id"], "Planer", r["got_mib"], "Boot", r["expected_mib"], "OK" if r["match"] else "ABWEICHUNG", r["error"] or "")
        bad += 0 if res.get("all_match") else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
