"""DIE EXPERTEN-KARTE -- wer haelt welchen Experten, und wo liegt der Rest.

Nutzer-Gesetz 22.09.: *"alles was geshardet wird braucht ne karte"*. Die
Dense-Gewichte haben eine (``tp_widths``/``src_widths``), der Draft hat
eine (``region_tag=weights_draft``); die Experten hatten keine. Ihre
Aufteilung wurde an VIER Stellen unabhaengig aus denselben zwei Vektoren
abgeleitet -- ``slot_base_for_rank``, ``global_rows``,
``_hotset_global_ids`` und ``shared_geometry`` -- und genau dieses
"zweimal dasselbe ausrechnen" hat am 22.09. acht Boots gekostet:

    w42-44  P rechnete 451 Store-Plaetze, D 512
    w45     beide 512 -> 59 GB tmpfs -> oom_kill 27->28, P tot
    w46-49  #97: die globale Residenzmenge enthielt Ids, die P nie haelt

DIE KARTE IST DER TAUSCH, NICHT DIE VEREINIGUNG (Nutzer 22.09. 07:08Z):

    "falls das andere layout einen/mehrere andere experten im vram
     braucht wie das zum schluss geladene, muessen die vram layer zurueck
     in den moe cache und die benoetigten experten in den vram. kein
     zusaetzlicher systemram dafuer notwendig"

Der Store haelt also zu JEDEM ZEITPUNKT ``total - resident`` Zeilen, nicht
die Vereinigung aller je kalten Ids. Beide Phasen halten hier 188 von 512
resident, also hat der Store 324 Plaetze -- exakt die am Metall gemessene
Zahl aus fnFL2w24 und w30 (506,25 MiB je Tensor, 36,71 statt 58,01 GiB).
Beim Flip wandern nur die Ids, die die Phasen NICHT teilen: die
Schnittmenge bleibt auf den Karten liegen, der Rest tauscht Platz gegen
Platz. Die Belegung bleibt dabei konstant.

SKALIERTE GRENZEN, NICHT DIE ROHEN RATIOS. ``--rank-moe-ratio
183,137,168`` summiert 488, nicht 512; der Server skaliert auf
``[192,144,176]`` mit den Grenzen ``[0,192,336]``. fnFL2w49 starb an genau
funf Ids (183..187), weil ich die rohen Zahlen als Bereichsgrenzen nahm.
Die Skalierung gehoert deshalb HIERHIN -- einmal, an der Quelle der Karte
-- und nicht in jeden Leser.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

#: Die Karte, die der Launcher publiziert; beide Gruppen lesen dieselbe.
MAP_ENV = "SGLANG_MOE_EXPERT_MAP"

PHASE_PP = "P"
PHASE_TP = "D"


def scaled_spans(ratios: Sequence[int], total: int) -> List[int]:
    """Die Experten JE RANG, auf ``total`` skaliert -- wie der Server rechnet.

    ``--rank-moe-ratio`` ist ein VERHAELTNIS, keine Stueckzahl: 183,137,168
    summiert 488. Der letzte Rang bekommt den Rest, damit die Summe exakt
    ``total`` ist und keine Id zwischen zwei Baendern verschwindet.
    """
    r = [max(0, int(x)) for x in ratios]
    s = sum(r)
    if s <= 0 or total <= 0:
        return [0] * len(r)
    out = [round(x * total / s) for x in r[:-1]]
    out.append(total - sum(out))
    return out


def bounds(spans: Sequence[int]) -> List[int]:
    """Der erste globale Id je Rang (die ``lo`` aus den STORE-SLOTS-Zeilen)."""
    lo, acc = [], 0
    for span in spans:
        lo.append(acc)
        acc += int(span)
    return lo


def resident_sharded(ratios: Sequence[int], fractions: Sequence[float],
                     total: int) -> List[List[int]]:
    """Je Rang die GLOBALEN Ids, die diese Gruppe auf ihren Karten haelt.

    Die Auswahl ist "die ersten ``round(span * fraction)`` des eigenen
    Bandes" -- das ist, was der Offload-Plan ohne Hotset tut, und fnFL2w48
    hat am Metall gezeigt, dass ein Hotset daran nichts aendert (P folgte
    ihm nicht). Die Karte schreibt also auf, was WIRKLICH passiert, nicht
    was passieren sollte.
    """
    spans = scaled_spans(ratios, total)
    lo = bounds(spans)
    out: List[List[int]] = []
    for i, span in enumerate(spans):
        try:
            f = float(fractions[i])
        except (IndexError, TypeError, ValueError):
            f = 0.0
        f = min(max(f, 0.0), 1.0)
        out.append(list(range(lo[i], lo[i] + round(span * f))))
    return out


def resident_unsharded(fraction: float, total: int) -> List[List[int]]:
    """Die PP-Gruppe als EIN Rang: sie haelt jeden Experten und residiert
    die ersten ``round(total * fraction)``."""
    try:
        f = min(max(float(fraction), 0.0), 1.0)
    except (TypeError, ValueError):
        f = 0.0
    return [list(range(round(total * f)))]


def build(total: int,
          ratios: Sequence[int],
          fr_pp: float,
          fr_tp: Sequence[float]) -> dict:
    """Die Karte beider Phasen, mit EINER Platzzahl fuer beide.

    ``slots`` ist das Maximum der beiden kalten Mengen, nicht ihre
    Vereinigung: mehr Plaetze braucht der Tausch nie, weniger wuerde ihn
    unmoeglich machen. ``moves`` sagt, wieviele Zeilen ein Flip bewegt --
    die Schnittmenge bleibt liegen, und genau das ist die Ersparnis, die
    kein zusaetzliches Systemram kostet.
    """
    res_p = resident_unsharded(fr_pp, total)
    res_d = resident_sharded(ratios, fr_tp, total)
    flat_p = {g for ids in res_p for g in ids}
    flat_d = {g for ids in res_d for g in ids}
    kalt_p = [g for g in range(total) if g not in flat_p]
    kalt_d = [g for g in range(total) if g not in flat_d]
    slots = max(len(kalt_p), len(kalt_d))
    return {
        "version": 1,
        "total": int(total),
        "slots": int(slots),
        "spans": scaled_spans(ratios, total),
        "bounds": bounds(scaled_spans(ratios, total)),
        "phases": {
            PHASE_PP: {"resident": [list(x) for x in res_p],
                       "slot_of": {str(g): i for i, g in enumerate(kalt_p)}},
            PHASE_TP: {"resident": [list(x) for x in res_d],
                       "slot_of": {str(g): i for i, g in enumerate(kalt_d)}},
        },
        #: Wieviele Zeilen der Flip bewegt (beide Richtungen zusammen), und
        #: wieviele auf den Karten liegen bleiben.
        "moves": len(flat_p - flat_d) + len(flat_d - flat_p),
        "shared_resident": len(flat_p & flat_d),
    }


def phase_of(group: str) -> str:
    """``P`` fuer die PP-Gruppe, ``D`` sonst -- die Karte kennt nur zwei."""
    return PHASE_PP if str(group).upper().startswith("P") else PHASE_TP


def slot_of(karte: dict, phase: str, global_id: int) -> Optional[int]:
    """Der Platz dieser Id in DIESER Phase, oder ``None`` wenn sie resident
    ist (dann gehoert sie auf eine Karte, nicht in den Store)."""
    try:
        return karte["phases"][phase]["slot_of"].get(str(int(global_id)))
    except (KeyError, TypeError, ValueError):
        return None


def resident_of(karte: dict, phase: str, rank: int) -> List[int]:
    """Die globalen Ids, die dieser Rang in dieser Phase resident haelt."""
    try:
        res = karte["phases"][phase]["resident"]
        return list(res[int(rank)]) if int(rank) < len(res) else list(res[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return []


def refuse_if_inconsistent(karte: dict) -> Optional[str]:
    """Die EINE Stelle, an der die Karte gegen sich selbst geprueft wird.

    Nach dem Umbau kann ``#97`` nicht mehr aus zwei auseinanderlaufenden
    Ableitungen entstehen -- nur noch daraus, dass die Karte selbst falsch
    ist. Dann sagt sie es hier, einmal, mit Zahl.
    """
    total = int(karte.get("total", 0))
    if total <= 0:
        return "total <= 0"
    for phase, p in karte.get("phases", {}).items():
        res = {g for ids in p.get("resident", []) for g in ids}
        slot_of_ = p.get("slot_of", {})
        kalt = {int(k) for k in slot_of_}
        if res & kalt:
            gemein = sorted(res & kalt)[:4]
            return (f"Phase {phase}: {len(res & kalt)} Ids sind resident UND "
                    f"im Store (erste {gemein}) -- eine Id kann nur eines sein")
        if len(res) + len(kalt) != total:
            return (f"Phase {phase}: resident {len(res)} + Store {len(kalt)} "
                    f"= {len(res) + len(kalt)}, erwartet {total}")
        if slot_of_ and max(int(v) for v in slot_of_.values()) >= int(karte["slots"]):
            return (f"Phase {phase}: ein Platz liegt hinter dem Ende der "
                    f"Datei ({karte['slots']} Plaetze)")
    return None
