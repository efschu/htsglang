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


#: #134: mehr Baender als das je Layer-Chunk zu Tags macht, will der Plan
#: nicht -- 16 Chunks x 32 Baender sind schon 512 Tags. Die Zahl ist eine
#: PLAN-Groesse, keine Physik; sie steht hier, damit sie EINE Stelle hat.
MAX_BANDS = 32


def band_geometry(ratios: Sequence[int], total: int) -> tuple:
    """Die Breite eines EXPERTEN-BANDES und ihre Anzahl (#134).

    Die Breite ist der GROESSTE GEMEINSAME TEILER der D-Spans, und das ist
    keine Geschmacksfrage: ein Band ist die Einheit, die der Flip taggt und
    am Stueck bewegt. Schneidet es eine Besitzgrenze von D, gehoert eine
    Haelfte Rang r und die andere Rang r+1 -- dann gibt es fuer das Band
    keinen EINEN Halter, und genau daran ist heute frueh die Karte
    gescheitert ("298 Ids sind resident UND im Store").

    Mit ``ratios=[183,137,168]`` und ``total=512`` sind die Spans 192/144/176
    (``scaled_spans``); ggT = 16, also 32 Baender, und jede D-Grenze
    (0/192/336/512) faellt auf eine Bandgrenze. Die Breite wird damit AUS DER
    GEOMETRIE gerechnet statt im Arm gepinnt -- andere Ratios, andere
    Breite, ohne dass jemand daran denken muss
    (Memory ``arm-defaults-die-den-boot-toeten``).

    Gibt ``(0, 0)`` zurueck, wenn die Ratios keine brauchbare Teilung
    ergeben -- dann ist die Bandteilung AUS, und das ist die LANGSAME
    Ausweichrichtung, nie die falsche: der Flip transportiert dann wie
    vorher je Layer-Chunk. Zwei Faelle fuehren dahin, und beide lieber als
    ein Band mit zwei Besitzern:
      * der ggT teilt ``total`` nicht (eine Id fiele aus jedem Band),
      * der ggT ist so klein, dass mehr als ``MAX_BANDS`` Baender
        entstuenden -- bei 512 Experten und ggT 1 waeren das 512 Tags JE
        CHUNK, also 8192 im Plan, und ein Band von 1,5 MiB liegt unter der
        Groesse, ab der der Transport ueberhaupt asynchron laeuft
        (``piece_histogram``: >= 2 MiB).
    """
    from math import gcd

    spans = [int(x) for x in scaled_spans(ratios, total) if int(x) > 0]
    if not spans or int(total) <= 0:
        return 0, 0
    size = spans[0]
    for x in spans[1:]:
        size = gcd(size, x)
    if size <= 0 or int(total) % size != 0:
        return 0, 0
    count = int(total) // size
    if count > MAX_BANDS:
        return 0, 0
    return size, count


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


def resident_unsharded(fraction, total: int) -> List[List[int]]:
    """Je PP-STUFE die globalen Ids, die sie auf ihrer Karte haelt (#132).

    PP teilt nach LAYERN: jede Stufe haelt alle ``total`` Experten IHRER
    Layer und residiert davon die ersten ``round(total * f_i)``. Die
    Stufen haben verschiedene Karten und verschiedene Fractions -- genau
    dafuer gibt es zwei Layouts, und eine Zahl fuer alle drei wirft die
    Optimierung weg.

    ``fraction`` darf darum ein VEKTOR sein (eine Fraktion je Stufe); ein
    Skalar bleibt die alte Form, eine Liste mit einem Eintrag auch.
    """
    try:
        fs = [float(x) for x in fraction]          # Vektor
    except TypeError:
        try:
            fs = [float(fraction)]                 # Skalar
        except (TypeError, ValueError):
            fs = [0.0]
    except ValueError:
        fs = [0.0]
    if not fs:
        fs = [0.0]
    return [list(range(round(total * min(max(f, 0.0), 1.0)))) for f in fs]


def mirror_tp_selection(res_d: Sequence[Sequence[int]],
                        fr_pp, total: int) -> List[List[int]]:
    """#159: die PP-Stufen halten DIESELBEN Ids wie die TP-Gruppe.

    Der Austausch ist ein Besitzerwechsel -- er kann nur Bytes umhaengen,
    die BEIDE Seiten als denselben Tensor beschreiben. `resident_unsharded`
    nimmt fuer P die ersten ``round(total*f)`` Ids von 0..total-1,
    `resident_sharded` fuer D je Rang die ersten seines BANDES. Zwei
    verschiedene Praefixe ueber zwei verschiedenen Grundmengen, und deshalb
    ist die Schnittmenge winzig: gemessen fnFL2w132 zwei Ids von 193, und
    selbst bei gleicher ANZAHL (158 gegen 158) genau eine.

    Diese Funktion setzt Ps Auswahl auf die Vereinigung der D-Baender. Die
    MENGE bestimmt damit D (ueber --rank-moe-ratio und
    --rank-moe-resident-fraction der D-Gruppe); ``fr_pp`` bleibt als
    OBERGRENZE je Stufe: haelt eine Stufe die Menge nicht, wird sie hier
    gekuerzt und der Aufrufer sieht es am Laengenunterschied.

    Der Preis ist ehrlich zu nennen: die Menge ist fuer ALLE Layer
    dieselbe, weil die TP-Gruppe alle Layer traegt. Eine Stufe auf der
    grossen Karte kann also nicht mehr halten als eine auf der kleinen --
    nicht weil die Software es verbietet, sondern weil ein Byte, das die
    andere Seite nicht hat, nicht den Besitzer wechseln kann.
    """
    gemeinsam = sorted({int(g) for ids in res_d for g in ids})
    try:
        fs = [float(x) for x in fr_pp]
    except TypeError:
        fs = [float(fr_pp)]
    if not fs:
        fs = [1.0]
    out: List[List[int]] = []
    for f in fs:
        cap = round(int(total) * min(max(float(f), 0.0), 1.0))
        out.append(list(gemeinsam[:cap]) if cap < len(gemeinsam)
                   else list(gemeinsam))
    return out


def build(total: int,
          ratios: Sequence[int],
          fr_pp: float,
          fr_tp: Sequence[float],
          mirror: bool = False) -> dict:
    """Die Karte beider Phasen, mit EINER Platzzahl fuer beide.

    ``slots`` ist das Maximum der beiden kalten Mengen, nicht ihre
    Vereinigung: mehr Plaetze braucht der Tausch nie, weniger wuerde ihn
    unmoeglich machen. ``moves`` sagt, wieviele Zeilen ein Flip bewegt --
    die Schnittmenge bleibt liegen, und genau das ist die Ersparnis, die
    kein zusaetzliches Systemram kostet.
    """
    res_d = resident_sharded(ratios, fr_tp, total)
    # #159: mit `mirror` haelt P dieselben Ids wie D -- die einzige Form, die
    # der Austausch verbinden kann. Ohne `mirror` bleibt die alte Auswahl
    # byte-identisch, damit bestehende Rechnungen sich nicht still aendern.
    res_p = (mirror_tp_selection(res_d, fr_pp, total) if mirror
             else resident_unsharded(fr_pp, total))
    # #132 KALT IST, WAS MINDESTENS EINE STUFE NICHT HAELT.
    #
    # Nicht "was keine Stufe haelt" (die Vereinigung): der Store muss die
    # SCHLECHTEST versorgte Stufe bedienen koennen. Bei 0.367/0.75/0.95
    # haelt Stufe 0 nur 188 von 512 -- sie braucht 324 Zeilen, auch wenn
    # Stufe 2 fast alles auf der Karte hat. Die Vereinigung haette 26
    # gesagt und die Laufzeit waere mit "95 eigene kalte Experten haben in
    # der KARTE keinen Platz" gestorben (fnFL2w64).
    _sets_p = [set(ids) for ids in res_p] or [set()]
    # `flat_p` bleibt die Vereinigung -- sie beantwortet "wer liegt
    # IRGENDWO in P auf einer Karte" und traegt `moves`/`shared_resident`.
    # `kalt_p` beantwortet die ANDERE Frage und darf nicht dasselbe sein.
    flat_p = {g for ids in res_p for g in ids}
    flat_d = {g for ids in res_d for g in ids}
    kalt_p = [g for g in range(total)
              if any(g not in s_ for s_ in _sets_p)]
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


def join_verdict(karte: dict) -> List[str]:
    """#159: kann der FLIP die beiden Layouts ueberhaupt verbinden?

    `refuse_if_inconsistent` prueft die Karte gegen SICH SELBST. Diese
    Funktion prueft die beiden Phasen GEGENEINANDER -- und das ist die
    Frage, an der fnFL2w130/w131/w132 gestorben sind, jedes Mal erst beim
    Wake, 40 Minuten nach dem Start:

        W68 Weg2XchgPlanDisagree: ...experts.weg2_experts_w13_weight_packed:
        die PP-Seite haelt (36160, 2560), die TP-Reihen [(9600,2560),
        (20480,2560),(20480,2560)] -- weder Zeilenschnitt noch Spaltenschnitt
        noch Replikat noch PADDED cut, also koennen die zwei Gruppen nicht
        denselben Tensor beschreiben.

    Der Austausch vergleicht JE LAYER. Ein Layer gehoert genau einer
    PP-Stufe; D traegt alle Layer, haelt also fuer jeden dieselbe Menge.
    Beide Seiten muessen fuer diesen Layer DIESELBEN globalen Ids resident
    haben -- nicht dieselbe Anzahl, dieselben Ids.

    Gemessen an der Form von w132: P-Stufe 0 haelt 193 Experten, D haelt
    158, und die Schnittmenge ist ZWEI. Beide sagen "die ersten N", meinen
    aber verschiedene Grundmengen: P die ersten N von 0..511, D je Rang die
    ersten N SEINES Bandes.

    Rueckgabe: eine Zeile je Stufe, die nicht passt; leere Liste = joinbar.
    KEINE Ausnahme -- der Aufrufer entscheidet, ob er refused oder nur warnt.
    """
    try:
        res_p = karte["phases"][PHASE_PP]["resident"]
        res_d = karte["phases"][PHASE_TP]["resident"]
    except (KeyError, TypeError):
        return ["Karte ohne phases -- join nicht pruefbar"]
    d_ids = set()
    for ids in res_d:
        d_ids |= set(int(x) for x in ids)
    out: List[str] = []
    for i, ids in enumerate(res_p):
        p_ids = set(int(x) for x in ids)
        if p_ids == d_ids:
            continue
        out.append(
            f"Stufe {i}: P haelt {len(p_ids)} Experten, D haelt {len(d_ids)}, "
            f"gemeinsam {len(p_ids & d_ids)} -- P-only {len(p_ids - d_ids)}, "
            f"D-only {len(d_ids - p_ids)}. Der Austausch vergleicht je Layer "
            f"und braucht DIESELBEN Ids auf beiden Seiten."
        )
    return out


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
        _listen = [set(ids) for ids in p.get("resident", [])] or [set()]
        # #132, ZWEITE SEITE DERSELBEN NAHT: `build` bildet `kalt` fuer die
        # PP-Phase als "was MINDESTENS EINE Stufe nicht haelt" -- der Store
        # muss die schlechtest versorgte Stufe bedienen. Dieser Pruefer las
        # dagegen die VEREINIGUNG und meldete darum jede Id als "resident UND
        # im Store", die eine Stufe haelt und eine andere nicht (gemessen bei
        # FR_P 0.377/0.700/0.442: 165 Ids, die Karte wurde VERWORFEN und der
        # Lauf fiel auf die globale Menge zurueck).
        #
        # "Resident" im Sinne des Stores heisst deshalb: von JEDER Stufe
        # gehalten -- die Schnittmenge. Fuer die TP-Phase sind die Listen
        # disjunkte Baender, dort ist Schnitt = Vereinigung, sobald mehr als
        # ein Rang existiert; der Sonderfall EIN Rang faellt mit beidem
        # zusammen. Eine Mengenoperation fuer beide Phasen, und sie ist
        # dieselbe, die `build` benutzt.
        if len(_listen) > 1 and any(a & b for i, a in enumerate(_listen)
                                    for b in _listen[i + 1:]):
            res = set.intersection(*_listen)      # PP: ueberlappende Listen
        else:
            res = set().union(*_listen)           # TP: disjunkte Baender
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
