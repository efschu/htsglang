"""KV-Köpfe je Rang: reine ANZEIGE (Nutzerentscheid 05.10., Stufe 1a), kein Einstellwert.

Es gibt kein Flag für die Kopfverteilung; sie folgt aus ``--rank-tp-ratio`` (Basis-Gewichtsliste) und der Kopfzahl des Modells.
Gerechnet wird NUR, was im Planer-Baum belegt ist, mit denselben Regeln wie die Laufzeit (``distributed/utils.py`` zieht torch und
ist im Dashboard nicht ladbar; die Regeln sind hier nachgebildet, ein Test vergleicht sie gegen die echte Funktion, wo sie ladbar ist):

* kein Plan (``--rank-tp-ratio`` fehlt oder Länge != tp): Standardpfad, gleichmäßig, wenn teilbar (utils.py ``tp_partition_sizes``, Z.1721-1723).
* Plan und KV-Köpfe < Ränge: REPLIZIERT, jeder Rang alle KV-Köpfe (utils.py ``attn_kv_replicated``, Z.1888-1915; ``kv == tp`` ist ausgenommen).
* Plan und KV-Köpfe >= Ränge: KV-Köpfe als unteilbare Einheiten, Largest-Remainder, jeder Rang >= 1 (``_partition_units_raw`` Z.1338-1364);
  bei Gewicht 0 (Form A) bekommt der Rang keine Köpfe (``_partition_units_with_empty_ranks`` Z.1367-1409).
* Q-Köpfe: nur der eindeutige Fall (genau ein Rang mit Gewicht > 0 trägt alle); sonst "nicht gerechnet" (kv-aligned Split, nicht nachgebildet).

Nichts wird geraten: fehlt die Kopfzahl (Modell nicht lesbar) oder ist die Gewichtsliste "auto", steht das so da.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

SRC = "distributed/utils.py"


def parse_ratios(text) -> Optional[object]:
    """``"1,0,0"`` -> [1, 0, 0]; ``"auto"``/``"auto-performance"`` -> der Text; sonst None (nicht lesbar, z. B. Shell-Variable)."""
    t = str(text or "").strip()
    if not t:
        return None
    if t.startswith("auto"):
        return t
    try:
        return [int(x) for x in t.replace(" ", "").split(",") if x != ""]
    except ValueError:
        return None


def _raw(units: int, weights: Sequence[int]) -> List[int]:
    """Largest-Remainder wie ``_partition_units_raw`` (utils.py:1338-1364): jeder Rang >= 1, Gleichstand zum niedrigeren Rang."""
    n = len(weights)
    if units < n:
        raise ValueError("%d Einheiten reichen nicht für %d Ränge (jeder Rang braucht mindestens eine)" % (units, n))
    total = sum(weights)
    quotas = [units * w / total for w in weights]
    sizes = [max(int(q), 1) for q in quotas]
    remaining = units - sum(sizes)
    if remaining < 0:
        for _ in range(-remaining):
            i = max(range(n), key=lambda r: (sizes[r], -r))
            sizes[i] -= 1
        remaining = 0
    order = sorted(range(n), key=lambda r: (quotas[r] - int(quotas[r]), -r), reverse=True)
    for k in range(remaining):
        sizes[order[k % n]] += 1
    return sizes


def split_units(units: int, weights: Sequence[int]) -> List[int]:
    """Wie ``partition_units(units, weights, groups=None, allow_zero=any(w == 0))`` (utils.py:1601-1629, 1367-1409)."""
    if any(w < 0 for w in weights):
        raise ValueError("negatives Gewicht in %s" % list(weights))
    if not any(w > 0 for w in weights):
        raise ValueError("alle Gewichte sind 0 (%s): kein Rang würde Köpfe tragen" % list(weights))
    if any(w == 0 for w in weights):
        kept = [r for r, w in enumerate(weights) if w > 0]
        sub = _raw(units, [weights[k] for k in kept])
        sizes = [0] * len(weights)
        for i, r in enumerate(kept):
            sizes[r] = sub[i]
        return sizes
    return _raw(units, weights)


def compute(*, tp_size: int, ratios, kv_heads: Optional[int], q_heads: Optional[int] = None) -> Dict[str, object]:
    """Anzeige-Ergebnis. ``status``: ``ok`` | ``unbekannt`` | ``fehler``; ``regime``: ``gleichmaessig`` | ``verteilt`` | ``repliziert`` |
    ``keins``; ``kv`` / ``q``: Köpfe je Rang oder None (nicht gerechnet); ``belege``: Fundstellen im Planer-Baum."""
    out: Dict[str, object] = {"status": "unbekannt", "regime": "keins", "kv": None, "q": None, "tp": tp_size, "satz": "", "notiz": [], "belege": []}
    if not kv_heads:
        out["satz"] = "Kopfzahl unbekannt (Modell im Container nicht lesbar): nicht gerechnet."
        return out
    if isinstance(ratios, str):
        out["satz"] = "Gewichte stehen auf \"%s\": der Planer löst sie, die Verteilung hängt vom Ergebnis ab: nicht gerechnet." % ratios
        return out
    plan = bool(ratios) and len(ratios) == tp_size
    if not plan:
        if kv_heads >= tp_size and kv_heads % tp_size == 0:
            out.update(status="ok", regime="gleichmaessig", kv=[kv_heads // tp_size] * tp_size,
                       satz="Kein Rang-Plan: gleichmäßig, %d KV-Köpfe je Rang." % (kv_heads // tp_size),
                       belege=["%s:1721-1723 (tp_partition_sizes ohne Plan)" % SRC])
        else:
            out["satz"] = "Kein Rang-Plan und %d KV-Köpfe auf %d Ränge: Standardpfad der Laufzeit, nicht gerechnet." % (kv_heads, tp_size)
        return out
    if kv_heads < tp_size:
        out.update(status="ok", regime="repliziert", kv=[kv_heads] * tp_size,
                   satz="%d KV-Köpfe < %d Ränge: REPLIZIERT, jeder Rang hält alle %d KV-Köpfe (die Token-Achse teilt uneven DCP)."
                        % (kv_heads, tp_size, kv_heads),
                   belege=["%s:1888-1915 (attn_kv_replicated)" % SRC])
        nz = [r for r, w in enumerate(ratios) if w > 0]
        if q_heads and len(nz) == 1 and q_heads % kv_heads == 0:
            q = [0] * tp_size
            q[nz[0]] = q_heads
            out["q"] = q
            out["notiz"].append("Q-Köpfe: nur Rang %d hat Gewicht, er trägt alle %d (belegt für NF im Boot-Log: [24, 0, 0])." % (nz[0], q_heads))
        elif q_heads:
            out["notiz"].append("Q-Köpfe: nicht gerechnet (kv-aligned Split in Einheiten von %d)." % kv_heads)
        return out
    try:
        kv = split_units(kv_heads, list(ratios))
    except ValueError as exc:
        out.update(status="fehler", satz="Verteilung nicht möglich: %s." % exc,
                   belege=["%s:1338-1364 (_partition_units_raw)" % SRC])
        return out
    out.update(status="ok", regime="verteilt", kv=kv,
               satz="%d KV-Köpfe auf %d Ränge nach Gewichten %s: %s (Largest-Remainder, jeder Rang mit Gewicht >= 1 Kopf)."
                    % (kv_heads, tp_size, ",".join(str(w) for w in ratios), ",".join(str(k) for k in kv)),
               belege=["%s:1338-1364, 1367-1409 (partition_units)" % SRC])
    if kv_heads == tp_size:
        out["notiz"].append("KV-Köpfe = Ränge: nicht repliziert (ausdrücklich ausgenommen, utils.py:1895-1914).")
    if q_heads:
        out["notiz"].append("Q-Köpfe: nicht gerechnet.")
    return out


def model_heads(model_dir: str) -> Optional[Dict[str, int]]:
    """Kopfzahlen aus ``config.json`` (``text_config`` bevorzugt) oder None, wenn nicht lesbar."""
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    t = cfg.get("text_config") or cfg
    q, kv = t.get("num_attention_heads"), t.get("num_key_value_heads")
    if not isinstance(q, int) or not isinstance(kv, int):
        return None
    return {"q": q, "kv": kv, "head_dim": t.get("head_dim"), "layers": t.get("num_hidden_layers")}


def _planner_rows(planner_only: Sequence[dict], have: Sequence[str]) -> List[dict]:
    """Vom Planer gerechnete ``--rank-tp-ratio`` (``profile_json.view``: ``planner_only``, Schlüssel ``extra:D:--rank-tp-ratio``) für Gruppen,
    in denen das Profil keinen Wert setzt."""
    out = []
    for p in planner_only:
        parts = str(p.get("key", "")).split(":")
        if len(parts) == 3 and parts[0] == "extra" and parts[2] == "--rank-tp-ratio" and parts[1] not in have:
            out.append({"name": "--rank-tp-ratio", "scope": parts[1], "value": "", "planner_value": p.get("value")})
    return out


def view(rows: Sequence[dict], model_dir: str = "", planner_only: Sequence[dict] = ()) -> List[Dict[str, object]]:
    """Je Gruppe mit ``--rank-tp-ratio`` eine Anzeige. ``rows`` sind die Zeilen von ``profile_json.view`` (name, scope, value, planner_value),
    ``planner_only`` die Werte, die der Planer rechnet und das Profil nicht setzt."""
    heads = model_heads(model_dir) if model_dir else None
    have = [r.get("scope") for r in rows if r.get("name") == "--rank-tp-ratio"]
    out: List[Dict[str, object]] = []
    for r in list(rows) + _planner_rows(planner_only, have):
        if r.get("name") != "--rank-tp-ratio":
            continue
        raw = r.get("value") or r.get("planner_value") or ""
        ratios = parse_ratios(raw)
        tp = len(ratios) if isinstance(ratios, list) else 0
        res = compute(tp_size=tp, ratios=ratios, kv_heads=(heads or {}).get("kv"), q_heads=(heads or {}).get("q")) if tp or isinstance(ratios, str) \
            else {"status": "unbekannt", "regime": "keins", "kv": None, "q": None, "tp": 0, "notiz": [], "belege": [],
                  "satz": "Gewichte \"%s\" nicht lesbar (Shell-Variable?): nicht gerechnet." % raw}
        res.update(group=r.get("scope"), ratios=raw, heads=heads, quelle="Profil" if r.get("value") else "Planer")
        out.append(res)
    return out
