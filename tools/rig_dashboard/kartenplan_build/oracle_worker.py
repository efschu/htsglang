"""AP-D (Plan Profil-Planer 06.10., Stufe B/C): langlebiger Worker fuer das Orakel (Launcher-Trockenlauf) und ``propose()``.

Laeuft als KINDPROZESS des Dashboards mit dem Python der sglang-Umgebung (``PYTHONPATH=<Baum>/python``, ``CUDA_VISIBLE_DEVICES=``), wie
``couplings_worker.py``: eine JSON-Zeile je Anfrage auf stdin, eine JSON-Zeile Antwort auf stdout (``{"id", "ok", ...}``), bis stdin schliesst.
Eigener Prozess statt des Kopplungs-Workers, weil ein Trockenlauf Sekunden dauert, den Launcher importiert und Prozesszustand anfasst
(``propose_oracle._module_state_guard`` stellt ihn wieder her): ein Absturz oder eine Zeitueberschreitung hier darf die Balken des Editors nicht
treffen.

Anfragen (``what``):

* ``verdikt``  ein Profil (``basis`` {env_path | env_text}) auf einem Inventar (``inventar`` {hardware | devices | cards}): das Dokument
  ``flliper.verdikt/1`` (``weg2/propose_verdict.run_verdikt``);
* ``propose``  Vorschlag (``weg2/propose.py``) + Orakel + Verdikte je Wert (``propose_verdict.run_propose``).

Der Worker beruehrt keine GPU, kein Netz, keinen echten Launcher-Start (nur ``launcher.main(--dry-run)`` auf einem NVML-Replay) und schreibt nur in
ein privates Temp-Verzeichnis.  Jeder Fehler kommt als ``{"ok": False, "error": ...}``, nie als Abbruch des Prozesses."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

#: the protocol channel: a PRIVATE copy of the real stdout, taken before anything is imported.  The import of the launcher (sglang, torch) and the
#: dry run itself print to stdout (logger handlers, warnings); fd 1 is pointed at stderr (the parent sends that to /dev/null), so the one JSON
#: line per answer is the only thing on the pipe the parent reads ("Kopplungs-Worker meldet sich ohne JSON" was the symptom, 06.10.)
_PROTO = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)


def tree_of_sglang() -> str:
    """Wurzel des Baums, aus dem ``sglang`` importiert wurde (``<Baum>/python/sglang/...`` -> ``<Baum>``): der ``--tree`` des Trockenlaufs."""
    import sglang

    return os.path.abspath(os.path.join(os.path.dirname(sglang.__file__), "..", ".."))


def handle(req: dict, tree: str) -> dict:
    from sglang.srt.weg2 import propose_verdict as PV

    what = req.get("what")
    if what == "verdikt":
        return PV.run_verdikt(req, tree=tree)
    if what == "propose":
        return PV.run_propose(req, tree=tree)
    if what == "ping":
        return {"ok": True, "tree": tree}
    return {"ok": False, "error": "unbekannte Anfrage %r (verdikt | propose | ping)" % (what,)}


def main() -> int:
    out = _PROTO
    boot_error = None
    tree = ""
    try:
        tree = tree_of_sglang()
        from sglang.srt.weg2 import launcher, propose_oracle, propose_verdict  # noqa: F401 -- der Import ist der teure Teil: vor dem ersten Hallo
    except Exception as exc:  # noqa: BLE001 -- bereit auch ohne Orakel: jede Anfrage nennt den Grund selbst
        boot_error = "Orakel nicht ladbar: %s: %s" % (type(exc).__name__, exc)
    out.write(json.dumps({"id": 0, "ok": True, "ready": True, "error": boot_error, "tree": tree}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.pop("id", None)
            res = handle(req, tree) if boot_error is None else {"ok": False, "error": boot_error}
        except Exception as exc:  # noqa: BLE001 -- ein Fehler beendet den Worker nicht
            res = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        res = dict(res, id=rid)
        out.write(json.dumps(res, default=str) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
