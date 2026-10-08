"""AP-H1 (Plan Profil-Planer 06.10., Zeile AP-H Teil 1): die DATEN der einen Seite "Profil-Planer".

Das Dashboard rechnet nichts (R1): diese Datei hält nur, was die Seite anzeigt und nicht aus dem Profil selbst kommt -- die vier Betriebsformen mit
je einem erklärenden Satz, die Abschnitte A (Aufteilung), B (KV), C (Experten) mit den Namen ihrer Werte (Wireframe ``dash-wireframe-eine-seite-1005``
Abschnitt 5), die Dual-ENV-Tabelle (Plan 4c Nachträge 12:20Z und 13:20Z) mit Standardwerten und die Grenzen der Regler.  ``ProfilEditor.list()``
liefert das als ``planer``; ``static/profil_planer.js`` zeichnet es.

Jeder Satz der Betriebsformen und der Dual-Tabelle steht in einer Quelle (Plan R2/R3, Katalogtext von ``--d-only`` und ``--dual-layout``,
``pdflip/dual_green.py``, ``pdflip/dual_share.py``); die Standardwerte der Dual-Tabelle prüft ``test_profil_planer_aph1_1006`` gegen den Quelltext.
Was nicht belegt ist, steht nicht hier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "flliper.planer-ui/1"

#: Betriebsform der Seite -> Name der Form in ``ProfilEditor.propose`` (AP-D: ``flip``/``tp`` Orakel, ``dual`` AP-E Orakel + Dual-Passung, ``single`` AP-F Planer-Rechnung)
FORM_BACKEND = {"einzel": "single", "tp": "tp", "flip": "flip", "dual": "dual"}

#: die vier Formen (Plan R2); ``n_min``/``n_max`` = zulässige Kartenzahl (R2: Einzelkarte N=1, die übrigen N>=2; topology.py MIN_CARDS)
FORMEN: List[Dict[str, Any]] = [
    {"id": "einzel", "name": "Single card", "n_min": 1, "n_max": 1,
     "satz": "One card, one rank: the normal server without the pdflip launcher, for example a laptop. The proposal there is a planner calculation, not a launcher run.",
     "quelle": "PLAN-PROFIL-PLANER-1006 R2 and R5b (user 06.10.)"},
    {"id": "tp", "name": "TP only", "n_min": 2, "n_max": None,
     "satz": "Only the decode group D: all cards work as one tensor-parallel group, without a prefill group, without switching and without a front (flag --d-only, from two cards).",
     "quelle": "Catalog text of --d-only (user 25.09.)"},
    {"id": "flip", "name": "Flip PP/TP", "n_min": 2, "n_max": None,
     "satz": "The standard form: the same cards work alternately as prefill group P (pipeline, reads the long input text) and as decode group D (tensor-parallel, computes the answer); the server switches between the two (from two cards).",
     "quelle": "PLAN-PROFIL-PLANER-1006 R2; glossary of the catalog (P, D, flip)"},
    {"id": "dual", "name": "Dual PP/TP", "n_min": 2, "n_max": None,
     "satz": "P (pipeline) and D (tensor-parallel) are awake at the same time, on the same cards; the front never switches (flag --dual-layout, from two cards, only in the 27B tree).",
     "quelle": "PLAN-PROFIL-PLANER-1006 R2 and R3 (user 06.10.); catalog text of --dual-layout"},
]

#: die Abschnitte der Seite und die Namen ihrer Werte (Wireframe Abschnitt 5; jeder Name steht im Katalog, ``test_profil_planer_aph1_1006``)
ABSCHNITTE: List[Dict[str, Any]] = [
    {"id": "A", "titel": "A  Split across the cards",
     "satz": "How the model is distributed across the cards: layers per card (prefill), weights per card (decode), memory budget and memory items (foreign context, non-torch, reserves, L15, extend trim) per card.",
     "namen": ["--pp-size", "--tp-size", "--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-layer-ratio", "--pp-layer-set", "--pp-solve-objective",
               "--pp-solve-pool-floor", "--pp-solve-cut", "--pp-cut-expert-device-fraction", "--pp-cut-expert-lru-rows", "--p-layer-split",
               "--p-attn-head-split", "--rank-gpu-id", "--rank-gpu-memory-mib", "--rank-role", "--rank-tp-ratio", "--rank-mlp-ratio",
               "--rank-moe-ratio", "--rank-vocab-ratio", "--rank-user-reserve-mib", "--rank-auto-reserve-mib", "--user-reserve-mib",
               "--d-tp-objective",
               # Positionale Je-Karte-Vektoren des Launchers (POSITIONAL_VECTOR_FLAGS/-TOKENS, launcher.py): Speicherposten je Karte, Release-Profile nf*/27b* setzen sie
               "--d-foreign-context-mib", "--d-nontorch-mib", "--d-reserve-mib", "--pp-cut-reserve-mib",
               "FLLIPER_PDFLIP_L15_MIB", "FLLIPER_PDFLIP_EXTEND_TRIM_MIB"]},
    {"id": "B", "titel": "B  KV: heads, token shares, DCP",
     "satz": "Where the KV cache lives: how many tokens per card, whether uneven DCP applies. The KV heads per rank are derived and only a display.",
     "namen": ["--dcp-size", "--uneven-dcp", "--uneven-dcp-weighted", "--rank-kv-ratio", "FLLIPER_UNEVEN_TOKEN_VECTOR", "--uneven-token-vector",
               "--d-uneven-token-vector", "--d-kv-token-cut", "--d-token-placement", "--kv-reshard-vectors"]},
    {"id": "C", "titel": "C  Experts (MoE)",
     "satz": "How many experts per card are in VRAM and how the experts are distributed (effective only for MoE models).",
     "namen": ["--rank-moe-resident-fraction", "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION", "FLLIPER_UNEVEN_MOE_EXPERT_SHARD", "FLLIPER_UNEVEN_MOE_VECTOR",
               "FLLIPER_MOE_SCRATCH_SLOTS", "--expert-placement-override"]},
]

#: die Dual-ENV-Tabelle (Plan 4c).  Namen aus ``pdflip/dual_green.py`` (``GreenConfig.from_env``: ``ENV_PREFIX + "GREEN_" + "TABLE"``), ``pdflip/dual_share.py``
#: (``STARVE_AGE_S``, ``STARVE_MAX_RUNG``) und ``environ.py`` (``FLLIPER_PDFLIP_DUAL_GRANT_RETRY_MS``).  Standardwerte aus dem Quelltext, Beleg je Zeile.
DUAL_ENV = {
    "table": "FLLIPER_PDFLIP_DUAL_SHARE_GREEN_TABLE",
    "starve_age": "FLLIPER_PDFLIP_DUAL_SHARE_STARVE_AGE_S",
    "starve_max": "FLLIPER_PDFLIP_DUAL_SHARE_STARVE_MAX_RUNG",
    "retry": "FLLIPER_PDFLIP_DUAL_GRANT_RETRY_MS",
    "rungs": "FLLIPER_PDFLIP_DUAL_SHARE_RUNGS",
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
    "quelle": {"table": "dual_green.py:1071 (default), :1097 (format), :1204-1224 (entry rung), :1255-1256 (D empty -> rung 0)",
               "starve": "dual_share.py:157-158 (default), :414-417 (clamp); dual_green.py:1319-1322",
               "retry": "environ.py:679-684; dual_p_kv_stage.py:668-681, :893 (dual layout only, 0 = off)"},
    "texte": {
        "stufen": "Rung 0, 1, 2, 3 = P's share of the compute units of the card: 100, 75, 50, 25 percent (default of the rungs, FLLIPER_PDFLIP_DUAL_SHARE_RUNGS).",
        "tabelle": "When D decodes again after idling and no minimum rate is set, the first row whose threshold the D seats reach or exceed selects the entry rung of P: the column \"tau low\" or \"tau high\" (tau = waiting P work in seconds; high = above the upper edge, default 10 s). If D is empty (0 seats), P always stands at 100 percent. Afterwards the front readjusts to the measured D round time.",
        "klemme": "Starvation clamp: if the oldest P request waits longer than the set seconds, P's rung is limited to at most the clamp rung, whatever the table says. The lowest rung of the ladder (3) switches the clamp off.",
        "retry": "Waiting P requests otherwise ask for KV space in every scheduler round (code default 0 = busy loop, ticket #1530). With N greater than 0 they ask at most every N milliseconds, immediately on a change in the card ledger.",
        "nur_dual": "All four values act only in the dual layout. The table acts only with --dual-priority dynamic, --dual-share-actuators green and --dual-green-ladder on.",
    },
}

ZIELE = {"seats": [1, 256], "kv_tokens": [1024, 8 << 20], "kontext_presets": [32768, 65536, 131072, 262144]}


#: Positionale Vektoren des Launchers (launcher.py:5964-5973 POSITIONAL_VECTOR_FLAGS = Flag-dest-Namen, POSITIONAL_VECTOR_TOKENS = --extra-/--env-Tokens;
#: ein Test liest beide Tupel aus dem Quelltext und vergleicht).  Je Eintrag ein Rang; der Launcher verweigert so einen Vektor, wenn die Zahl nicht zum Inventar passt.
#: Das ist die VOLLE Liste des Launchers; gezeichnet wird sie abzueglich LAUNCHER_NICHT_JE_KARTE (siehe unten).
POSITIONAL_FLAGS_ALL = ["--d-foreign-context-mib", "--d-nontorch-mib", "--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-cut-expert-device-fraction",
                        "--pp-cut-expert-lru-rows", "--user-reserve-mib", "--d-reserve-mib", "--pp-cut-reserve-mib", "--d-reshard-presets", "--p-barlink-bar1-window-mib"]
POSITIONAL_TOKENS_ALL = ["--rank-role", "--rank-tp-ratio", "--rank-moe-ratio", "--rank-moe-resident-fraction", "--rank-user-reserve-mib", "--rank-gpu-memory-mib",
                         "--pp-stage-ratio", "--pp-attn-stage-ratio", "FLLIPER_MOE_SCRATCH_SLOTS", "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION", "FLLIPER_PDFLIP_L15_MIB",
                         "FLLIPER_PDFLIP_EXTEND_TRIM_MIB"]
#: Aus der POSITIONAL-Liste des Launchers, die seine Topologie-Probe NICHT als Vektor je Karte zaehlt (launcher.py:6319-6328 _TOPOLOGY_VECTOR_FLAGS/-TOKENS):
#: das BAR1-Fenster ("24,PP_0=96", Code BAR1-WINDOW), die d_reshard-Presets (nicht je Karte) und L1.5 (eigene Probe).  Sie bleiben Textfelder mit Katalogtext.
#: Ein Test bildet _TOPOLOGY_VECTOR_FLAGS/-TOKENS aus dem Launcher-Quelltext nach und pinnt vector_names()/POSITIONAL_* dagegen.
LAUNCHER_NICHT_JE_KARTE = ["--p-barlink-bar1-window-mib", "--d-reshard-presets", "FLLIPER_PDFLIP_L15_MIB"]
POSITIONAL_FLAGS = [f for f in POSITIONAL_FLAGS_ALL if f not in LAUNCHER_NICHT_JE_KARTE]
POSITIONAL_TOKENS = [t for t in POSITIONAL_TOKENS_ALL if t not in LAUNCHER_NICHT_JE_KARTE]
#: Je-Karte-Vektoren aus den Abschnitten A-C, die der Launcher nicht positional fuehrt (je Eintrag ein Rang bzw. eine Stufe laut Katalogtext).
#: Bewusst NICHT dabei: Kommalisten, die keine Rang-Vektoren sind (--pp-layer-set, --p-layer-split, --p-attn-head-split, --kv-reshard-vectors,
#: --expert-placement-override, --d-kv-token-cut, --d-token-placement) und die Dual-/Planer-Listen (--dual-share-actuators, --cuda-graph-bs ...): die bleiben ein Textfeld.
SECTION_VECTORS = ["--pp-layer-ratio", "--rank-gpu-id", "--rank-mlp-ratio", "--rank-vocab-ratio", "--rank-auto-reserve-mib", "--rank-kv-ratio",
                   "FLLIPER_UNEVEN_TOKEN_VECTOR", "--uneven-token-vector", "--d-uneven-token-vector", "FLLIPER_UNEVEN_MOE_VECTOR"]


#: Je RANG (nicht je Karte): ``--rank-gpu-id`` nennt je Tensor-Parallel-Rang die physische Karte; Duplikate legen mehrere Raenge auf eine Karte
#: (catalog.json), die Eintragszahl ist also die Rangzahl und darf von der Kartenzahl abweichen.  Die Seite beschriftet nur "Rang i", ohne Kartenname, ohne Summe.
RANK_VECTORS = ["--rank-gpu-id"]


def vector_names() -> List[str]:
    """Die Namen, die als ein Feld je Rang gezeichnet werden (jeder genau einmal, Reihenfolge stabil)."""
    out: List[str] = []
    for n in POSITIONAL_FLAGS + [t.rstrip("=") for t in POSITIONAL_TOKENS] + SECTION_VECTORS:
        if n not in out:
            out.append(n)
    return out


def ui_info(vorschlag_formen: Any = (), entries: Optional[Mapping[str, Mapping[str, Any]]] = None, oracle: bool = False, dual: bool = True) -> Dict[str, Any]:
    """Das ``planer``-Objekt der ``list``-Antwort.

    ``vorschlag_formen``: die Formen, die ``ProfilEditor.propose`` kann (``ProfilEditor.FORMS``); eine Seitenform mit anderem Namen bekommt
    ``vorschlag: False`` und einen Hinweis (heute bedient ``propose`` alle vier Formen: ``flip``, ``tp``, ``dual``, ``single``).  ``entries``: der Katalog (für Text und Kanten der
    Dual-ENV-Werte, auch wenn das Profil sie nicht setzt).  ``oracle``: ob das Orakel konfiguriert ist (ohne es gibt es keinen Vorschlag).
    ``dual``: ob der Planer-Baum die Dual-Form traegt (``profil.dual_line_probe``); ohne sie (NF-Linie, Nutzerentscheid 07.10.) steht Dual weder in ``formen`` noch
    als Dual-ENV-Tabelle (kein Schluessel ``dual``) in der Antwort, ``dual_verfuegbar`` sagt es."""
    can = set(vorschlag_formen or ())
    formen = []
    for f in FORMEN:
        if f["id"] == "dual" and not dual:
            continue
        back = FORM_BACKEND[f["id"]]
        d = dict(f, backend=back, vorschlag=bool(oracle and back in can))
        if not d["vorschlag"]:
            if not oracle:
                d["hinweis"] = "The proposal needs the oracle (launcher dry run), which is not set up in this dashboard; the values can be set by hand."
            else:
                d["hinweis"] = "There is no proposal for this form in this version; the values can be set by hand."
        formen.append(d)
    ents = entries or {}
    out = {"schema": SCHEMA, "formen": formen, "abschnitte": [dict(a) for a in ABSCHNITTE], "vektoren": vector_names(),
           "positional": POSITIONAL_FLAGS + [t.rstrip("=") for t in POSITIONAL_TOKENS], "je_rang": list(RANK_VECTORS), "ziele": dict(ZIELE), "oracle": bool(oracle),
           "dual_verfuegbar": bool(dual)}
    if dual:
        tab = dict(DUAL)
        tab["werte"] = {}
        for key, name in DUAL_ENV.items():
            e = ents.get(name) or {}
            tab["werte"][name] = {"name": name, "rolle": key, "text": e.get("text") or "", "help": e.get("help") or "", "depends": [dict(d) for d in e.get("depends") or []],
                                  "level": e.get("level"), "gain": e.get("gain") or "", "cost": e.get("cost") or ""}
        out["dual"] = tab
    return out


def all_section_names() -> List[str]:
    """Alle Namen der Abschnitte A-C (jeder genau einmal)."""
    return [n for a in ABSCHNITTE for n in a["namen"]]
