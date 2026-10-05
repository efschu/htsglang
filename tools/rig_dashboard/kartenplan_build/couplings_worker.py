"""Profil-Editor S4b (Auftrag 1432): langlebiger Worker für die Kopplungs-Engine.

Läuft als KINDPROZESS des Dashboards mit dem Python der sglang-Umgebung (``PYTHONPATH=<Baum>/python``, ``CUDA_VISIBLE_DEVICES=``).
Eine JSON-Zeile pro Anfrage auf stdin, eine JSON-Zeile Antwort auf stdout (``{"id": .., "ok": .., "result" | "error": ..}``), bis stdin
schließt.  Warum langlebig: der Import von ``sglang.srt.planner`` kostet Sekunden, und der Editor ruft nach jeder Eingabe (entprellt ~300 ms) neu
an; ein Kindprozess je Aufruf (wie ``runner.py``) würde jede Eingabe mit dem Importstart bezahlen.

Der Worker berührt keine GPU, keinen Launcher, kein Netz und keine Datei außer dem Import."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def topology(req: dict) -> dict:
    """Auftrag 1984 (C): ``{"what": "topology", "n": N}`` -> ``plan_topology(N)`` in der sglang-Umgebung dieses Kindprozesses.

    ``topology.plan_topology`` importiert fuer N != 3 ``weg2.weight_exchange_region`` (sglang): im Dashboard-Prozess (ohne sglang) geht das nicht,
    hier schon.  Antwort: ``{"ok": True, "refused": None}`` (durchgelassen) oder ``{"ok": True, "refused": "<Text der TopologyRefused>"}``;
    ein Import- oder Rechenfehler ist ``{"ok": False, "error": ...}`` (der Aufrufer nennt ihn in der Notiz, es ist keine Ablehnung)."""
    try:
        from sglang.srt.weg2 import topology as TP
        TP.plan_topology(int(req["n"]))
    except Exception as exc:  # noqa: BLE001 -- TopologyRefused ist die einzige Ablehnung, alles andere ein benannter Fehler
        if type(exc).__name__ == "TopologyRefused":
            return {"ok": True, "refused": str(exc)}
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"ok": True, "refused": None}


def main() -> int:
    try:
        from sglang.srt.planner import profile_couplings as PC
        boot_error = None
    except ImportError as exc:
        PC = None
        boot_error = "profile_couplings.py fehlt im Planer-Baum (%s)" % exc
    out = sys.stdout
    # bereit ist der Worker auch ohne profile_couplings: die Topologie braucht es nicht, und ein fehlendes Modul meldet jede Kopplungsanfrage selbst
    out.write(json.dumps({"id": 0, "ok": True, "ready": True, "error": boot_error}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.pop("id", None)
            if req.get("what") == "topology":
                res = topology(req)
            else:
                res = PC.run(req) if PC is not None else {"ok": False, "error": boot_error}
        except Exception as exc:  # noqa: BLE001 -- ein Fehler beendet den Worker nicht
            res = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        res = dict(res, id=rid)
        out.write(json.dumps(res, default=str) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
