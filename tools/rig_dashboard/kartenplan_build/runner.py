"""Kartenplaner (Item 510): Planer-Funktionen als reine Funktionsaufrufe, OHNE GPU, OHNE Launcher-Prozess.

Läuft als KINDPROZESS des Dashboards mit dem Python der sglang-Umgebung
(``PYTHONPATH=<Baum>/python``, ``CUDA_VISIBLE_DEVICES=``), liest eine JSON-Anfrage von stdin und schreibt
eine JSON-Antwort nach stdout.  Warum ein Kindprozess und nicht im Dashboard-Prozess: der Dashboard-Dienst
läuft mit dem System-Python ohne torch (MemoryMax=1G); der sglang-Import zieht torch.  Der Aufruf selbst ist
derselbe wie in test/registered/unit/weg2/hw_generic_rig_plan_fingerprint_1002.py: Module importieren,
Funktionen mit einem synthetischen Karteninventar rufen.

Dieses Skript startet NIE den Launcher (``python -m sglang.srt.weg2.launcher``) und ruft nie ``launcher.main``.
Es berührt keinen Shared-Memory-Ordner, keinen Docker, keinen Port.

Operationen
  version    Welcher Baum, welche Funktionen sind da.
  gate       Karteninventar gegen das Gate des Planers (card_identity, topology): Arch, Zahl, Reihenfolge,
             Kalibrierklasse, Topologie.  Antwort: die ORIGINAL-Meldungen der Planer-Funktionen.
  rederive   D-/P-Budgets mit ``launcher.budgets_from_dc`` aus den aufgezeichneten Eingaben neu rechnen.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _cards(L, rows):
    out = []
    for r in rows:
        kw = dict(nvml_index=int(r["nvml_index"]), uuid=r["uuid"], name=r["name"], total_mib=int(r["total_mib"]),
                  reserved_mib=int(r.get("reserved_mib") or 0))
        if r.get("cc"):
            kw["cc"] = tuple(r["cc"])
        try:
            out.append(L.Card(**kw))
        except TypeError:          # Baum ohne cc-Feld (27B-Linie vor Item 260)
            kw.pop("cc", None)
            out.append(L.Card(**kw))
    return out


def op_version(req):
    from sglang.srt.weg2 import launcher as L
    info = {"launcher": os.path.relpath(L.__file__, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(L.__file__)))))
            if False else L.__file__, "has_budgets_from_dc": hasattr(L, "budgets_from_dc")}
    try:
        from sglang.srt.weg2 import card_identity  # noqa: F401
        info["card_identity"] = True
    except ImportError:
        info["card_identity"] = False
    try:
        from sglang.srt.weg2 import topology  # noqa: F401
        info["topology"] = True
    except ImportError:
        info["topology"] = False
    return info


def op_gate(req):
    """Gate des Planers (gleiche Funktion wie im Dashboard-Prozess, hier mit dem echten Paketimport)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rigdash"))
    import kartenplan_gate as G
    from sglang.srt.weg2 import card_identity as CI
    from sglang.srt.weg2 import topology as TP

    return G.gate(CI, TP, req["cards"])


def op_rederive(req):
    """``rows``: je Budgetzeile {label, ordinal, card, inputs{...}, expected_mib}.  Gleiche Funktion wie der Launcher."""
    from sglang.srt.weg2 import launcher as L

    out = []
    for row in req["rows"]:
        cards = _cards(L, row["cards"])
        dc = {c.uuid: int(v) for c, v in zip(cards, row["dormant_other"])}
        lines = []
        kw = dict(overshoot_mib=row.get("overshoot"), overshoot_provenance=row.get("overshoot_prov", ""),
                  user_reserve_by_card={c.uuid: int(v) for c, v in zip(cards, row.get("user_reserve") or [0] * len(cards))},
                  dormant_growth_mib=row.get("growth"), dormant_growth_provenance=row.get("growth_prov", ""),
                  charge_driver_carve=bool(row.get("charge_carve")), driver_carve_min_total_mib=int(row.get("carve_min_total") or 0),
                  awake_rest_mib=row.get("awake_rest"), awake_rest_provenance=row.get("awake_rest_prov", ""),
                  booked_rest_mib=row.get("booked_rest"), booked_rest_provenance=row.get("booked_rest_prov", ""),
                  l15_mib=row.get("l15"))
        import inspect
        sig = inspect.signature(L.budgets_from_dc).parameters
        dropped = []
        for k in list(kw):
            if k not in sig:                      # Revision älter als der Parameter: nur wenn der Wert nichts bewirkt, weglassen
                v = kw.pop(k)
                if v not in (None, "", False, 0) and not (isinstance(v, (list, tuple)) and not any(v)) \
                        and not (isinstance(v, dict) and not any(v.values())):
                    dropped.append(k)
        try:
            if dropped:
                raise TypeError("Revision kennt Parameter %s nicht (Eingabe wäre wirksam)" % ",".join(dropped))
            got = L.budgets_from_dc(cards, dc, lines.append, row["label"], **kw)
            err = None
        except Exception as exc:  # noqa: BLE001 - Planer-Verweigerung wird benannt weitergereicht
            got, err = None, "%s: %s" % (type(exc).__name__, exc)
        out.append({"id": row.get("id"), "label": row["label"], "got_mib": got, "expected_mib": row.get("expected_mib"),
                    "error": err, "lines": [ln for ln in lines if ln.startswith("budget ")]})
    return out


OPS = {"version": op_version, "gate": op_gate, "rederive": op_rederive}


def main() -> int:
    req = json.load(sys.stdin)
    fn = OPS.get(req.get("op"))
    if fn is None:
        json.dump({"ok": False, "error": "unbekannte Operation %r" % req.get("op")}, sys.stdout)
        return 2
    try:
        res = fn(req)
    except Exception as exc:  # noqa: BLE001
        import traceback
        json.dump({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "trace": traceback.format_exc()[-1500:]}, sys.stdout)
        return 1
    json.dump({"ok": True, "result": res}, sys.stdout, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
