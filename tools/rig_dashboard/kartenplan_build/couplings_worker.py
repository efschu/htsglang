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


def main() -> int:
    try:
        from sglang.srt.planner import profile_couplings as PC
        boot_error = None
    except ImportError as exc:
        PC = None
        boot_error = "profile_couplings.py fehlt im Planer-Baum (%s)" % exc
    out = sys.stdout
    out.write(json.dumps({"id": 0, "ok": PC is not None, "ready": True, "error": boot_error}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.pop("id", None)
            res = PC.run(req) if PC is not None else {"ok": False, "error": boot_error}
        except Exception as exc:  # noqa: BLE001 -- ein Fehler beendet den Worker nicht
            res = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        res = dict(res, id=rid)
        out.write(json.dumps(res, default=str) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
