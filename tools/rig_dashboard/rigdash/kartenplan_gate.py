"""Kartenplaner (Item 510): das Gate des Planers, als reine Funktion im Server-Prozess.

``flliper/srt/pdflip/card_identity.py`` und ``topology.py`` sind laut ihren Köpfen "PURE: stdlib only".  Sie
werden hier per Dateipfad aus dem ausgelieferten Planer-Baum geladen (nicht über ``import flliper``, das
torch zieht), und dann genauso gerufen wie der Launcher sie ruft: ``arch_gate``, ``order_cards``,
``uncalibrated_message``, ``plan_topology``.  Die Meldungen sind die ORIGINALTEXTE des Planers.

Kein Launcher-Aufruf, keine GPU, kein NVML: die Karten sind synthetische Dicts.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, List, Optional


class GateUnavailable(RuntimeError):
    """Der Planer-Baum liefert card_identity/topology nicht (zu alte Linie): benannt, nie geraten."""


def _load(name: str, path: str):
    if not os.path.isfile(path):
        raise GateUnavailable("%s is missing in the planner tree (%s)" % (os.path.basename(path), path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # @dataclass löst Typen über sys.modules[__module__] auf
    spec.loader.exec_module(mod)
    return mod


def load_modules(tree_python: str):
    base = os.path.join(tree_python, "flliper", "srt", "pdflip")
    return _load("kp_card_identity", os.path.join(base, "card_identity.py")), _load("kp_topology", os.path.join(base, "topology.py"))


def _row(r: dict) -> dict:
    d = dict(r)
    d["cc"] = tuple(r["cc"]) if r.get("cc") else None
    return d


def gate(ci, tp, rows: List[dict]) -> dict:
    """``rows``: [{nvml_index, uuid, name, total_mib, cc:[maj,min], bar1_total_mib?, pcie_max_gen?, pcie_max_width?}].

    Antwort mit dem Urteil je Prüfung (Arch je Karte, Kartenzahl, Kalibrierklasse, Topologie)."""
    res: dict = {"per_card": [], "count": None, "order": None, "calibration": None, "topology": None,
                 "reference_inventory": list(ci.REFERENCE_INVENTORY),
                 "supported_archs": [list(a) for a in ci.SUPPORTED_ARCHS],
                 "tree_card_identity": getattr(ci, "__file__", "")}
    for r in rows:
        c = _row(r)
        entry = {"nvml_index": r["nvml_index"], "name": r["name"], "card_key": ci.card_key(c),
                 "calibration_class": ci.calibration_class(c), "arch": {"ok": True, "message": None}}
        try:
            ci.arch_gate([c])
        except ci.CardInventoryRefused as exc:
            entry["arch"] = {"ok": False, "message": str(exc), "code": ci.CODE_ARCH}
        res["per_card"].append(entry)
    cards = [_row(r) for r in rows]
    try:
        ordered = ci.order_cards(cards, expect_count=3, gate=False)
        res["count"] = {"ok": True, "message": None}
    except ci.CardInventoryRefused as exc:
        res["count"] = {"ok": False, "code": ci.CODE_COUNT, "message": str(exc)}
        ordered = sorted(cards, key=ci.order_key)
    res["order"] = [{"nvml_index": ci.props_of(o).nvml_index, "class": ci.class_label(o)} for o in ordered]
    msg = ci.uncalibrated_message(
        ordered, list(ci.REFERENCE_INVENTORY),
        ["D_FIXED_MIB / D_AWAKE_REST / P_OVERSHOOT / P_ACTIVATION (positional rank records)", "Dormant-Residue W19",
         "P chunk model", "PP cut stage model"], "the release record set")
    res["calibration"] = {"ok": msg is None, "code": ci.CODE_UNCALIBRATED, "message": msg}
    try:
        t = tp.plan_topology(len(rows))
        res["topology"] = {"ok": True, "p": "PP%d" % t.p_pp, "d": "TP%d" % t.d_tp, "host_ordinal": t.host_ordinal,
                           "proven": t.proven, "message": None}
    except tp.TopologyRefused as exc:
        res["topology"] = {"ok": False, "code": "HW-TOPOLOGY", "message": str(exc),
                           "proven_card_counts": list(tp.PROVEN_CARD_COUNTS), "blockers": list(tp.N_NOT_3_BLOCKERS)}
    res["ok"] = bool(all(p["arch"]["ok"] for p in res["per_card"]) and res["count"]["ok"]
                     and res["calibration"]["ok"] and res["topology"]["ok"])
    return res
