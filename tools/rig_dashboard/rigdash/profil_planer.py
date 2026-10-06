"""AP-H1 (Plan Profil-Planer 06.10., Zeile AP-H Teil 1): die DATEN der einen Seite "Profil-Planer".

Das Dashboard rechnet nichts (R1): diese Datei hält nur, was die Seite anzeigt und nicht aus dem Profil selbst kommt -- die vier Betriebsformen mit
je einem erklärenden Satz, die Abschnitte A (Aufteilung), B (KV), C (Experten) mit den Namen ihrer Werte (Wireframe ``dash-wireframe-eine-seite-1005``
Abschnitt 5), die Dual-ENV-Tabelle (Plan 4c Nachträge 12:20Z und 13:20Z) mit Standardwerten und die Grenzen der Regler.  ``ProfilEditor.list()``
liefert das als ``planer``; ``static/profil_planer.js`` zeichnet es.

Jeder Satz der Betriebsformen und der Dual-Tabelle steht in einer Quelle (Plan R2/R3, Katalogtext von ``--d-only`` und ``--dual-layout``,
``weg2/dual_green.py``, ``weg2/dual_share.py``); die Standardwerte der Dual-Tabelle prüft ``test_profil_planer_aph1_1006`` gegen den Quelltext.
Was nicht belegt ist, steht nicht hier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "flliper.planer-ui/1"

#: Betriebsform der Seite -> Name der Form in ``ProfilEditor.propose`` / ``propose_oracle`` (``single`` und ``dual`` sind AP-F und AP-E)
FORM_BACKEND = {"einzel": "single", "tp": "tp", "flip": "flip", "dual": "dual"}

#: die vier Formen (Plan R2); ``n_min``/``n_max`` = zulässige Kartenzahl (R2: Einzelkarte N=1, die übrigen N>=2; topology.py MIN_CARDS)
FORMEN: List[Dict[str, Any]] = [
    {"id": "einzel", "name": "Einzelkarte", "n_min": 1, "n_max": 1,
     "satz": "Eine Karte, ein Rang: der normale Server ohne den weg2-Launcher, zum Beispiel ein Laptop. Der Vorschlag ist dort eine Rechnung des Planers, kein Launcher-Lauf.",
     "quelle": "PLAN-PROFIL-PLANER-1006 R2 und R5b (Nutzer 06.10.)"},
    {"id": "tp", "name": "Nur TP", "n_min": 2, "n_max": None,
     "satz": "Nur die Decode-Gruppe D: alle Karten arbeiten als eine Tensor-Parallel-Gruppe, ohne Prefill-Gruppe, ohne Wechsel und ohne Front (Flag --d-only, ab zwei Karten).",
     "quelle": "Katalogtext von --d-only (Nutzer 25.09.)"},
    {"id": "flip", "name": "Flip PP/TP", "n_min": 2, "n_max": None,
     "satz": "Die Standardform: dieselben Karten arbeiten abwechselnd als Prefill-Gruppe P (Pipeline, liest den langen Eingabetext) und als Decode-Gruppe D (Tensor-parallel, rechnet die Antwort); der Server wechselt zwischen beiden (ab zwei Karten).",
     "quelle": "PLAN-PROFIL-PLANER-1006 R2; Glossar des Katalogs (P, D, Flip)"},
    {"id": "dual", "name": "Dual PP/TP", "n_min": 2, "n_max": None,
     "satz": "P (Pipeline) und D (Tensor-parallel) sind gleichzeitig wach, auf denselben Karten; die Front wechselt nie (Flag --dual-layout, ab zwei Karten, nur im 27B-Baum).",
     "quelle": "PLAN-PROFIL-PLANER-1006 R2 und R3 (Nutzer 06.10.); Katalogtext von --dual-layout"},
]

#: die Abschnitte der Seite und die Namen ihrer Werte (Wireframe Abschnitt 5; jeder Name steht im Katalog, ``test_profil_planer_aph1_1006``)
ABSCHNITTE: List[Dict[str, Any]] = [
    {"id": "A", "titel": "A  Aufteilung auf die Karten",
     "satz": "Wie das Modell auf die Karten verteilt wird: Layer je Karte (Prefill), Gewichte je Karte (Decode), Speicherbudget und Speicherposten (Fremdkontext, Nicht-Torch, Reserven, L15, Extend-Trim) je Karte.",
     "namen": ["--pp-size", "--tp-size", "--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-layer-ratio", "--pp-layer-set", "--pp-solve-objective",
               "--pp-solve-pool-floor", "--pp-solve-cut", "--pp-cut-expert-device-fraction", "--pp-cut-expert-lru-rows", "--p-layer-split",
               "--p-attn-head-split", "--rank-gpu-id", "--rank-gpu-memory-mib", "--rank-role", "--rank-tp-ratio", "--rank-mlp-ratio",
               "--rank-moe-ratio", "--rank-vocab-ratio", "--rank-user-reserve-mib", "--rank-auto-reserve-mib", "--user-reserve-mib",
               "--d-tp-objective",
               # Positionale Je-Karte-Vektoren des Launchers (POSITIONAL_VECTOR_FLAGS/-TOKENS, launcher.py): Speicherposten je Karte, Release-Profile nf*/27b* setzen sie
               "--d-foreign-context-mib", "--d-nontorch-mib", "--d-reserve-mib", "--pp-cut-reserve-mib",
               "SGLANG_WEG2_L15_MIB", "SGLANG_WEG2_EXTEND_TRIM_MIB"]},
    {"id": "B", "titel": "B  KV: Köpfe, Token-Anteile, DCP",
     "satz": "Wo der KV-Cache liegt: wie viele Token je Karte, ob ungleiches DCP gilt. Die KV-Köpfe je Rang sind abgeleitet und nur eine Anzeige.",
     "namen": ["--dcp-size", "--uneven-dcp", "--uneven-dcp-weighted", "--rank-kv-ratio", "SGLANG_UNEVEN_TOKEN_VECTOR", "--uneven-token-vector",
               "--d-uneven-token-vector", "--d-kv-token-cut", "--d-token-placement", "--kv-reshard-vectors"]},
    {"id": "C", "titel": "C  Experten (MoE)",
     "satz": "Wie viele Experten je Karte im VRAM liegen und wie die Experten verteilt sind (nur bei MoE-Modellen wirksam).",
     "namen": ["--rank-moe-resident-fraction", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "SGLANG_UNEVEN_MOE_EXPERT_SHARD", "SGLANG_UNEVEN_MOE_VECTOR",
               "SGLANG_MOE_SCRATCH_SLOTS", "--expert-placement-override"]},
]

#: die Dual-ENV-Tabelle (Plan 4c).  Namen aus ``weg2/dual_green.py`` (``GreenConfig.from_env``: ``ENV_PREFIX + "GREEN_" + "TABLE"``), ``weg2/dual_share.py``
#: (``STARVE_AGE_S``, ``STARVE_MAX_RUNG``) und ``environ.py`` (``SGLANG_WEG2_DUAL_GRANT_RETRY_MS``).  Standardwerte aus dem Quelltext, Beleg je Zeile.
DUAL_ENV = {
    "table": "SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE",
    "starve_age": "SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S",
    "starve_max": "SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG",
    "retry": "SGLANG_WEG2_DUAL_GRANT_RETRY_MS",
    "rungs": "SGLANG_WEG2_DUAL_SHARE_RUNGS",
}
DUAL = {
    "env": DUAL_ENV,
    #: ShareConfig.rungs (dual_share.py:139): Stufe k = Ps Anteil rungs[k]
    "rungs_default": [1.0, 0.75, 0.5, 0.25],
    #: GreenConfig.table (dual_green.py:1071), als Text im Format von ``from_env``: "bs<=:Stufe tau niedrig:Stufe tau hoch;..."
    "table_default": "2:1:0;4:2:1;1000000000:3:2",
    #: eine Schwelle ab dieser Größe zeigt die Seite als "alle größeren" (der Code nennt eine Schwelle ab 10**8 "inf", dual_green.py:1221; sein Standard und die Release-Profile nehmen 10**9)
    "table_unbegrenzt": 100000000,
    "starve_age_default": 60.0,         # dual_share.py:157
    "starve_max_default": 1,            # dual_share.py:158
    "retry_default": 0,                 # environ.py:684 (0 = aus)
    "quelle": {"table": "dual_green.py:1071 (Standard), :1097 (Format), :1204-1224 (Eintrittsstufe), :1255-1256 (D leer -> Stufe 0)",
               "starve": "dual_share.py:157-158 (Standard), :414-417 (Klemme); dual_green.py:1319-1322",
               "retry": "environ.py:679-684; dual_p_kv_stage.py:668-681, :893 (nur Dual-Layout, 0 = aus)"},
    "texte": {
        "stufen": "Stufe 0, 1, 2, 3 = Ps Anteil an den Recheneinheiten der Karte: 100, 75, 50, 25 Prozent (Standard der Stufen, SGLANG_WEG2_DUAL_SHARE_RUNGS).",
        "tabelle": "Wenn D nach Leerlauf wieder decodet und keine Mindestrate gesetzt ist, wählt die erste Zeile, deren Schwelle die D-Sitze erreicht oder übersteigt, "
                   "die Eintrittsstufe von P: die Spalte \"tau niedrig\" oder \"tau hoch\" (tau = wartende P-Arbeit in Sekunden; hoch = über der oberen Kante, Standard 10 s). "
                   "Ist D leer (0 Sitze), steht P immer bei 100 Prozent. Danach regelt die Front auf die gemessene D-Rundenzeit nach.",
        "klemme": "Aushungerungs-Klemme: wartet die älteste P-Anfrage länger als die eingestellten Sekunden, wird Ps Stufe auf höchstens die Klemmenstufe begrenzt, "
                  "egal was die Tabelle sagt. Die tiefste Stufe der Leiter (3) schaltet die Klemme aus.",
        "retry": "Wartende P-Anfragen fragen sonst bei jeder Scheduler-Runde nach KV-Platz (Code-Standard 0 = Dauerschleife, Ticket #1530). "
                 "Mit N größer 0 fragen sie höchstens alle N Millisekunden, sofort bei einer Änderung im Karten-Ledger.",
        "nur_dual": "Alle vier Werte wirken nur im Dual-Layout. Die Tabelle wirkt nur mit --dual-priority dynamic, --dual-share-actuators green und --dual-green-ladder on.",
    },
}

ZIELE = {"seats": [1, 256], "kv_tokens": [1024, 8 << 20], "kontext_presets": [32768, 65536, 131072, 262144]}


#: Positionale Vektoren des Launchers (launcher.py:5964-5973 POSITIONAL_VECTOR_FLAGS = Flag-dest-Namen, POSITIONAL_VECTOR_TOKENS = --extra-/--env-Tokens;
#: ein Test liest beide Tupel aus dem Quelltext und vergleicht).  Je Eintrag ein Rang; der Launcher verweigert so einen Vektor, wenn die Zahl nicht zum Inventar passt.
POSITIONAL_FLAGS = ["--d-foreign-context-mib", "--d-nontorch-mib", "--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-cut-expert-device-fraction",
                    "--pp-cut-expert-lru-rows", "--user-reserve-mib", "--d-reserve-mib", "--pp-cut-reserve-mib", "--d-reshard-presets", "--p-barlink-bar1-window-mib"]
POSITIONAL_TOKENS = ["--rank-role", "--rank-tp-ratio", "--rank-moe-ratio", "--rank-moe-resident-fraction", "--rank-user-reserve-mib", "--rank-gpu-memory-mib",
                     "--pp-stage-ratio", "--pp-attn-stage-ratio", "SGLANG_MOE_SCRATCH_SLOTS", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "SGLANG_WEG2_L15_MIB",
                     "SGLANG_WEG2_EXTEND_TRIM_MIB"]
#: Je-Karte-Vektoren aus den Abschnitten A-C, die der Launcher nicht positional fuehrt (je Eintrag ein Rang bzw. eine Stufe laut Katalogtext).
#: Bewusst NICHT dabei: Kommalisten, die keine Rang-Vektoren sind (--pp-layer-set, --p-layer-split, --p-attn-head-split, --kv-reshard-vectors,
#: --expert-placement-override, --d-kv-token-cut, --d-token-placement) und die Dual-/Planer-Listen (--dual-share-actuators, --cuda-graph-bs ...): die bleiben ein Textfeld.
SECTION_VECTORS = ["--pp-layer-ratio", "--rank-gpu-id", "--rank-mlp-ratio", "--rank-vocab-ratio", "--rank-auto-reserve-mib", "--rank-kv-ratio",
                   "SGLANG_UNEVEN_TOKEN_VECTOR", "--uneven-token-vector", "--d-uneven-token-vector", "SGLANG_UNEVEN_MOE_VECTOR"]


def vector_names() -> List[str]:
    """Die Namen, die als ein Feld je Rang gezeichnet werden (jeder genau einmal, Reihenfolge stabil)."""
    out: List[str] = []
    for n in POSITIONAL_FLAGS + [t.rstrip("=") for t in POSITIONAL_TOKENS] + SECTION_VECTORS:
        if n not in out:
            out.append(n)
    return out


def ui_info(vorschlag_formen: Any = (), entries: Optional[Mapping[str, Mapping[str, Any]]] = None, oracle: bool = False) -> Dict[str, Any]:
    """Das ``planer``-Objekt der ``list``-Antwort.

    ``vorschlag_formen``: die Formen, die ``ProfilEditor.propose`` kann (``flip``, ``tp``); eine Seitenform mit anderem Namen bekommt
    ``vorschlag: False`` und den Hinweis, dass ihr Vorschlag ein späteres Arbeitspaket ist.  ``entries``: der Katalog (für Text und Kanten der
    Dual-ENV-Werte, auch wenn das Profil sie nicht setzt).  ``oracle``: ob das Orakel konfiguriert ist (ohne es gibt es keinen Vorschlag)."""
    can = set(vorschlag_formen or ())
    formen = []
    for f in FORMEN:
        back = FORM_BACKEND[f["id"]]
        d = dict(f, backend=back, vorschlag=bool(oracle and back in can))
        if not d["vorschlag"]:
            if not oracle:
                d["hinweis"] = "Der Vorschlag braucht das Orakel (Launcher-Trockenlauf), das in diesem Dashboard nicht eingerichtet ist; die Werte lassen sich von Hand setzen."
            elif back == "single":
                d["hinweis"] = "Der Vorschlag für die Einzelkarte ist ein eigenes Arbeitspaket (AP-F) und in diesem Stand nicht enthalten; die Werte lassen sich von Hand setzen."
            else:
                d["hinweis"] = "Der Vorschlag für Dual ist ein eigenes Arbeitspaket (AP-E) und in diesem Stand nicht enthalten; die Werte lassen sich von Hand setzen."
        formen.append(d)
    ents = entries or {}
    dual = dict(DUAL)
    dual["werte"] = {}
    for key, name in DUAL_ENV.items():
        e = ents.get(name) or {}
        dual["werte"][name] = {"name": name, "rolle": key, "text": e.get("text") or "", "help": e.get("help") or "", "depends": [dict(d) for d in e.get("depends") or []],
                               "level": e.get("level"), "gain": e.get("gain") or "", "cost": e.get("cost") or ""}
    return {"schema": SCHEMA, "formen": formen, "abschnitte": [dict(a) for a in ABSCHNITTE], "vektoren": vector_names(),
            "positional": POSITIONAL_FLAGS + [t.rstrip("=") for t in POSITIONAL_TOKENS], "dual": dual, "ziele": dict(ZIELE), "oracle": bool(oracle)}


def all_section_names() -> List[str]:
    """Alle Namen der Abschnitte A-C (jeder genau einmal)."""
    return [n for a in ABSCHNITTE for n in a["namen"]]
