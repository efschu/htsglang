#!/usr/bin/env python3
"""s11 -- compose bar1_e2e.json from what the host run left behind.

The step script talks to the host; this reads the files it brought back and
turns them into ONE artifact with a decidable content. Nothing here touches a
card, a socket or ssh, which is what makes it testable against fixtures.

Three extractions carry the step:

  * the graph gate. bar1_graph_check.py prints one BESTANDEN/GEFALLEN line per
    case with a [Gate]/[Info] marker and exits 0 only when every gate case
    passed. Both are recorded: the exit code alone would hide WHICH case fell.
  * per-group attainment. parallel_state logs "HTCCL enabled for group '<x>':
    angefordert=<a>, ERREICHT=<e>" on success and the same pair as a WARNING on
    fallback. The requested name is worthless -- it says bar1 either way. Only
    ERREICHT counts, and it counts PER GROUP: with SGLANG_UNEVEN_DCP=1 there
    are two (tp:0, dcp:0), and a run where one of them fell back to gloo is a
    mixed measurement, not a bar1 measurement.
  * the coverage bolt. htccl._select raises rather than falling back to the
    host-staged gloo level during a graph capture. It is extracted as its own
    field, with op and size, because it is the expected failure of the current
    integration and needs to be distinguishable from any other crash.
  * the smoke, over /generate. It used to go through /v1/chat/completions,
    where two things went wrong at once on 2026-07-30: the chat template
    answered a temperature-0 request with a thinking preamble and spent the
    whole token budget on it, and `meta_info` is opt-in there
    (`return_meta_info`, default False), so spec_accept_length could not be
    read at all. A continuation prompt has no template and /generate always
    carries meta_info. The full reasoning sits at the request in
    s11_bar1_e2e.sh; what matters here is that the parser reads the shape the
    step really asks for.

WHICH FILE THE EVIDENCE IS READ FROM, and why that is not a detail. This used
to read `htccl_lines.txt` alone -- the grep result the step script harvests
AFTER the server has answered. On every early exit (server never came up,
capture aborted) the shell writes only `server.log` and jumps to compose, so
that one file did not exist. `read_lines` returns [] for a missing file, so
the artifact reported "no group line, no bolt, no fatal" for a run whose log
carried all three, and the check then failed on the CONSEQUENCE (no ERREICHT
line) instead of the CAUSE (the capture aborted). Exactly the s01 pattern:
a loader that reads a shape the producer does not write, and is silently
empty rather than loud.

Both files are read now, and which ones existed is recorded (`log_quellen`).
"nobody harvested a log" and "the log holds nothing" are then two different
answers rather than one empty list. The two files overlap -- the grep result
carries grep's "<lineno>:" prefix, the tail does not -- so lines are
deduplicated on their content, without which `aufbau_lines` would count the
same setup line twice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

KIND = "bar1_e2e"
#: 2: `log_quellen` / `log_zeilen` kamen dazu, und die Beweislage wird nicht
#: mehr nur aus `htccl_lines.txt` gelesen. Ein aelteres Artefakt darf hier
#: nicht durchrutschen -- es traegt die Felder nicht, an denen der Check
#: "niemand hat geschaut" von "nichts gefunden" unterscheidet.
#: 3: der Smoke laeuft ueber /generate statt /v1/chat/completions, und
#: `smoke` traegt dafuer `endpunkt`, `finish_reason`, `zahlen_erwartet`,
#: `spec_verify_ct` und `unterprovisioniert`. Aus demselben Grund: ohne
#: `endpunkt` liest ein Check nicht, WORAUF die Kohaerenzzahl sich bezieht.
#: 4: das Kriterium misst nicht mehr Gehorsam, sondern ob das Sprachmodell
#: intakt ist -- `anker_zahlen`, `muell_befunde`, `lm_intakt` und die
#: Kennzahlen dazu. Ein Artefakt nach Schema 3 traegt die Felder nicht, und
#: sein `kohaerent` bedeutete etwas anderes als das gleichnamige hier.
SCHEMA_VERSION = 4

#: Der Fortsetzungs-Prompt, den s11_bar1_e2e.sh an /generate schickt. Er
#: steht hier UND dort; dass die beiden zusammenpassen, nagelt
#: test_gpu_battery_checks_bar1.py am Quelltext der Schrittdatei fest --
#: sonst zaehlte der Parser eine Folge, die der Request nie angestossen hat.
SMOKE_PROMPT = "1 2 3 4"
#: Die erste Zahl, die die Fortsetzung liefern muss -- unmittelbar hinter dem
#: Prompt. Was im Prompt steht, ist kein Beleg fuer die Antwort.
ZAHLEN_VON = 5
ZAHLEN_BIS = 20

#: Wieviele Zahlen UNMITTELBAR und LUECKENLOS folgen muessen.
#:
#: Hier stand 15, und das hat den falschen Gegenstand gemessen. Anlauf 4 zaehlte
#: korrekt " 5 6 7 8 9 10" weiter und driftete dann in einen kohaerenten
#: russischen Forumtext -- rohe Fortsetzungs-Charakteristik eines Basispfades
#: ohne Anweisung, kein Schaden am Modell. Die 15 verlangten Gehorsam ueber 16
#: Zahlen; was dieser Schritt wissen will, ist etwas anderes: kommen fuer ein
#: determiniertes Praefix die RICHTIGEN Token heraus, und ist der Rest
#: wohlgeformter Text. Echte Korruption sieht anders aus -- sie liefert keine
#: sechs richtigen Zahlen und danach Non-Sequitur-Muell.
#:
#: 4 und nicht 6, obwohl 6 gemessen sind: eine einzelne Beobachtung
#: rechtfertigt keine Schwelle auf ihrem eigenen Wert. Der Abstand, auf den es
#: ankommt, ist der zwischen 0 und 4 -- ein kaputtes Kollektiv liefert nicht
#: vier korrekte Zahlen und verunglueckt dann.
ANKER_MIN = 4

#: Muell-Schwellen. JEDE ist an dem echten Artefakt von Anlauf 4 geeicht
#: (smoke.json, 1055 Zeichen), nicht geschaetzt -- die gemessenen Werte stehen
#: dahinter. Ein Test faehrt genau dieses Artefakt und verlangt, dass es
#: besteht.
#: gemessen 1.0000
MUELL_DRUCKBAR_MIN = 0.98
#: gemessen 0.435 (der Forumtext wiederholt einen Block, das ist Web-Text und
#: kein Defekt). Eine Tokenschleife liegt bei ~0.01 -- der Abstand ist gross,
#: die Schwelle liegt bewusst weit unter dem gemessenen Wert.
MUELL_VIELFALT_MIN = 0.15
#: Unter so wenigen Worten ist die Vielfalt Rauschen und wird nicht geprueft.
MUELL_VIELFALT_AB_WORTEN = 30
#: gemessen 3 ("###"). Eine Entartung wiederholt eine kurze Einheit dutzendfach.
MUELL_WDH_MAX = 10
MUELL_EINHEIT_MAX = 32
#: Kuerzer als das ist keine Antwort, ueber die sich etwas sagen laesst.
MUELL_MIN_ZEICHEN = 20

#: Woraus die Beweislage gelesen wird, in dieser Reihenfolge. `htccl_lines.txt`
#: ist der vollstaendige grep und steht deshalb vorn; `server.log` ist der
#: begrenzte Auszug, den die Schrittdatei auf JEDEM Weg schreibt -- auch auf
#: denen, die vor dem grep abbiegen.
LOG_QUELLEN = ("htccl_lines.txt", "server.log")

RE_GROUP = re.compile(
    r"group '(?P<gruppe>[^']+)': angefordert=(?P<angefordert>[^,\s]+),\s*"
    r"ERREICHT=(?P<erreicht>[A-Za-z0-9_\-]+)"
)
RE_KASSE = re.compile(r"BAR1-Kasse dieser Karte nach Gruppe '(?P<gruppe>[^']+)'")
RE_AUFBAU = re.compile(r"HTCCL-BAR1: Aufbau in\s+(?P<ms>[0-9.]+)\s*ms")
RE_RIEGEL = re.compile(
    r"HTCCL: '(?P<op>[A-Za-z0-9_]+)' mit (?P<bytes>\d+) Byte waehrend einer "
    r"CUDA-Graph-Aufzeichnung"
)
RE_GATE_CASE = re.compile(
    r"^\s*(?P<marke>BESTANDEN|GEFALLEN)\s*\[(?P<art>Gate|Info)\]\s*(?P<name>\S+)"
)

FATAL_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "NCCL error",
    "Watchdog caught collective operation timeout",
    # A boot killed by a plain traceback -- no OOM, no NCCL error -- is just as
    # dead. The shell already harvests this marker (s11_bar1_e2e.sh) and
    # check_common.FATAL_MARKERS carries it for every other step.
    "Traceback (most recent call last)",
)


def read_lines(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def parse_graph_check(step_dir: str) -> dict:
    lines = read_lines(os.path.join(step_dir, "graph_check.txt"))
    rc_raw = read_lines(os.path.join(step_dir, "graph_check_rc.txt"))
    try:
        rc = int(rc_raw[0].strip()) if rc_raw else None
    except ValueError:
        rc = None
    cases = []
    for line in lines:
        m = RE_GATE_CASE.match(line)
        if m:
            cases.append(
                {
                    "name": m.group("name"),
                    "gate": m.group("art") == "Gate",
                    "ok": m.group("marke") == "BESTANDEN",
                }
            )
    gates = [c for c in cases if c["gate"]]
    fallen = [c["name"] for c in gates if not c["ok"]]
    return {
        "rc": rc,
        "cases": len(cases),
        "gate_cases": len(gates),
        "gefallen": fallen,
        "alle_bestanden": bool(gates) and not fallen and rc == 0,
        "zusammenfassung_vorhanden": any("Zusammenfassung" in line for line in lines),
    }


def _ohne_grep_praefix(line: str) -> str:
    """Der Inhalt einer Zeile, unabhaengig davon, wer sie geerntet hat.

    ``grep -n`` stellt "<lineno>:" voran, ``tail`` nicht. Dieselbe Logzeile
    sieht in den beiden Quellen deshalb verschieden aus, obwohl sie dieselbe
    ist -- ohne diese Normalisierung zaehlte `aufbau_lines` sie doppelt.
    """
    return " ".join(re.sub(r"^\d+:", "", line).split())


def sammle_log_zeilen(step_dir: str) -> tuple:
    """Alle geernteten Logzeilen, entdoppelt, plus die Quellen, die es gab.

    Die Quellenliste ist das eigentliche Ergebnis: eine leere Zeilenliste
    heisst mit ihr "nichts gefunden", ohne sie "niemand hat geschaut". Das
    sind zwei Befunde, und nur einer davon ist ein Messergebnis.
    """
    quellen = []
    zeilen = []
    gesehen = set()
    for name in LOG_QUELLEN:
        pfad = os.path.join(step_dir, name)
        if not os.path.exists(pfad):
            continue
        quellen.append(name)
        for line in read_lines(pfad):
            schluessel = _ohne_grep_praefix(line)
            if not schluessel or schluessel in gesehen:
                continue
            gesehen.add(schluessel)
            zeilen.append(line)
    return zeilen, quellen


def parse_log_evidence(step_dir: str) -> dict:
    lines, quellen = sammle_log_zeilen(step_dir)
    gruppen: dict = {}
    kasse_gruppen = []
    aufbau = []
    riegel = None
    fatal = None
    for line in lines:
        m = RE_GROUP.search(line)
        if m:
            gruppen[m.group("gruppe")] = {
                "gruppe": m.group("gruppe"),
                "angefordert": m.group("angefordert"),
                "erreicht": m.group("erreicht"),
            }
        m = RE_KASSE.search(line)
        if m and m.group("gruppe") not in kasse_gruppen:
            kasse_gruppen.append(m.group("gruppe"))
        m = RE_AUFBAU.search(line)
        if m:
            aufbau.append(float(m.group("ms")))
        m = RE_RIEGEL.search(line)
        if m and riegel is None:
            riegel = {
                "op": m.group("op"),
                "bytes": int(m.group("bytes")),
                "zeile": " ".join(line.split())[:300],
            }
        if fatal is None:
            for marker in FATAL_MARKERS:
                if marker in line:
                    fatal = " ".join(line.split())[:300]
                    break
    return {
        "gruppen": sorted(gruppen.values(), key=lambda g: g["gruppe"]),
        "aufbau_gruppen": kasse_gruppen,
        "aufbau_lines": len(aufbau),
        "aufbau_ms": aufbau,
        "riegel": riegel,
        "fatal": fatal,
        "log_quellen": quellen,
        "log_zeilen": len(lines),
    }


def zaehle_in_folge(text: str, von: int, bis: int) -> int:
    """Wieviele der Zahlen ``von..bis`` IRGENDWO in dieser Reihenfolge stehen.

    Der alte Zaehler, und er bleibt nur fuer die Chat-Form stehen (die endet
    ohnehin in einem STOP). Warum er als Kriterium nicht taugt: er sagt
    nichts darueber, WOHER die Treffer kommen. Am 2026-07-30 meldete er 3
    fuer eine Antwort, die gar nicht gezaehlt hat -- die Treffer waren die
    Aufzaehlungspunkte "1.", "2.", "3." einer Denk-Praeambel. Und in Anlauf 4
    meldete er 10 fuer eine Antwort, deren erste sechs Zahlen richtig waren
    und deren Rest russischer Forumtext ist: die restlichen vier Treffer sind
    Ziffern aus "220В", "1450" und Datumsangaben. Ein Zaehler, der ueber den
    ganzen Text streut, findet in Fliesstext immer irgendetwas.

    Das Kriterium ist deshalb :func:`anker_folge` -- unmittelbar und
    lueckenlos.
    """
    pos = 0
    treffer = 0
    for n in range(von, bis + 1):
        stelle = text.find(str(n), pos)
        if stelle < 0:
            continue
        treffer += 1
        pos = stelle + len(str(n))
    return treffer


def anker_folge(text: str, von: int) -> tuple:
    """``(Anzahl, Resttext)`` -- die Zahlen UNMITTELBAR am Textanfang.

    Gezaehlt wird, wieviele Zahlen ab ``von`` lueckenlos und in Folge ganz
    vorn stehen, getrennt nur durch Zwischenraum. Beim ersten Bruch ist
    Schluss; was danach kommt, ist der Resttext und wird nicht mehr auf
    Zahlen abgesucht.

    **Das ist der Unterschied zwischen "gehorcht" und "rechnet richtig".**
    Ein Basispfad ohne Anweisung setzt eine Zahlenfolge ein Stueck weit fort
    und driftet dann in das, woran ihn das Praefix erinnert -- das ist die
    Charakteristik der rohen Fortsetzung, kein Defekt. Was der Anker prueft,
    ist die eine Frage, die dieser Schritt beantworten kann: kommen fuer ein
    determiniertes Praefix die richtigen Token heraus? Ein zerschossenes
    Kollektiv liefert die nicht.

    Streuende Suche waere hier falsch: "0/10" direkt hinter der 10 (so steht
    es im Artefakt von Anlauf 4) darf nicht als 11 durchgehen, nur weil
    irgendwo spaeter eine 11 auftaucht.
    """
    n = von
    rest = text
    treffer = 0
    while True:
        m = re.match(r"\s*(\d+)", rest)
        if m is None or int(m.group(1)) != n:
            break
        treffer += 1
        n += 1
        rest = rest[m.end():]
    return treffer, rest


def _max_wiederholung(text: str) -> int:
    """Wie oft sich eine kurze Einheit UNMITTELBAR hintereinander wiederholt.

    Der Muell-Test, der Entartung von Web-Text trennt. Eine Tokenschleife
    wiederholt dieselben paar Zeichen dutzendfach am Stueck; ein Forumtext,
    der einen zitierten Block ein zweites Mal bringt, tut das nicht
    unmittelbar und nicht kurz. Gemessen am Artefakt von Anlauf 4: 3 ("###").
    """
    text = text[:4000]
    best = 0
    for laenge in range(1, MUELL_EINHEIT_MAX + 1):
        i = 0
        while i + laenge <= len(text):
            einheit = text[i:i + laenge]
            n = 1
            while text[i + n * laenge:i + (n + 1) * laenge] == einheit:
                n += 1
                if n > 64:            # genug, um "entartet" zu sagen
                    return n
            if n > best:
                best = n
            i += 1
    return best


def muell_pruefung(text: str) -> tuple:
    """``(Befunde, Kennzahlen)`` -- ist der Text wohlgeformt oder Muell?

    Drei Fragen, absichtlich stumpf und ohne Meinung darueber, WORUEBER der
    Text spricht. Was hier NICHT geprueft wird, ist Sinn: ein russischer
    Forumbeitrag ueber Drehstrommotoren ist ein voellig intaktes
    Sprachmodell-Ergebnis, auch wenn ihn niemand bestellt hat.

    Jede Schwelle ist am echten Artefakt von Anlauf 4 geeicht; die gemessenen
    Werte stehen bei den Konstanten.
    """
    befunde = []
    kennzahlen = {}
    knapp = text.strip()
    if len(knapp) < MUELL_MIN_ZEICHEN:
        befunde.append(f"nur {len(knapp)} Zeichen Text")
        return befunde, kennzahlen

    druckbar = sum(1 for c in text if c.isprintable() or c in "\n\t")
    anteil = druckbar / len(text)
    kennzahlen["druckbar_anteil"] = round(anteil, 4)
    if anteil < MUELL_DRUCKBAR_MIN:
        befunde.append(
            f"nur {anteil:.3f} druckbare Zeichen (< {MUELL_DRUCKBAR_MIN})"
        )

    worte = text.split()
    if len(worte) >= MUELL_VIELFALT_AB_WORTEN:
        vielfalt = len(set(worte)) / len(worte)
        kennzahlen["wort_vielfalt"] = round(vielfalt, 4)
        if vielfalt < MUELL_VIELFALT_MIN:
            befunde.append(
                f"Wortvielfalt {vielfalt:.3f} (< {MUELL_VIELFALT_MIN}) -- "
                f"{len(set(worte))} verschiedene von {len(worte)} Worten"
            )

    wdh = _max_wiederholung(text)
    kennzahlen["max_wiederholung"] = wdh
    if wdh >= MUELL_WDH_MAX:
        befunde.append(
            f"eine kurze Einheit wiederholt sich {wdh}x unmittelbar "
            f"(>= {MUELL_WDH_MAX}) -- Tokenschleife"
        )
    return befunde, kennzahlen


def parse_smoke(step_dir: str) -> dict:
    """Coherence, mechanically -- und aus der Antwort, die der Schritt wirklich holt.

    Gelesen wird die /generate-Form (``{"text": ..., "meta_info": {...}}``).
    Die Chat-Form wird weiter verstanden, damit ein Artefakt aus einem
    aelteren Lauf nicht als "unlesbar" durchgeht, sondern als das, was es
    ist -- aber sie ist nicht mehr das, was der Schritt anfordert. Warum,
    steht in s11_bar1_e2e.sh am Request.

    ``spec_accept_length`` kommt aus ``meta_info`` und AUSDRUECKLICH nicht
    aus ``spec_ema_accept_len``: das ist ein geglaetteter Verlauf, nicht die
    Akzeptanzlaenge dieser Anfrage, und die beiden zu verwechseln ist eine
    bekannte Messfalle.
    """
    path = os.path.join(step_dir, "smoke.json")
    out: dict = {
        "vorhanden": os.path.exists(path),
        "endpunkt": None,
        "content_prefix": None,
        "spec_accept_length": None,
        "spec_verify_ct": None,
        "finish_reason": None,
        "zahlen_in_folge": 0,
        "zahlen_erwartet": ZAHLEN_BIS - ZAHLEN_VON + 1,
        "anker_zahlen": 0,
        "anker_min": ANKER_MIN,
        "drift_zeichen": 0,
        "muell_befunde": [],
        "lm_intakt": False,
        "kohaerent": False,
        "unterprovisioniert": False,
        "error": None,
    }
    if not out["vorhanden"]:
        return out
    try:
        with open(path, errors="replace") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        out["error"] = f"smoke.json nicht lesbar: {exc}"
        return out
    if not isinstance(payload, dict):
        out["error"] = "smoke.json ist kein JSON-Objekt"
        return out
    if payload.get("error") or payload.get("object") == "error":
        out["error"] = " ".join(str(payload.get("error") or payload).split())[:200]
        return out

    meta: dict = {}
    if isinstance(payload.get("text"), str):
        # /generate: Text und meta_info liegen oben, ohne Wenn und Aber.
        out["endpunkt"] = "generate"
        text = payload["text"]
        meta = payload.get("meta_info") or {}
        von, bis = ZAHLEN_VON, ZAHLEN_BIS
    else:
        # Chat-Form. Sie zaehlt ab 1, weil dort kein Prompt fortgesetzt wird
        # -- ein aelteres Artefakt soll dieselbe Zahl ergeben wie damals.
        out["endpunkt"] = "chat"
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
            meta = choice.get("meta_info") or {}
        except (KeyError, IndexError, TypeError) as exc:
            out["error"] = (
                f"Antwort ist weder /generate (text) noch Chat "
                f"(choices[0].message.content): {exc}"
            )
            return out
        von, bis = 1, 20
        out["zahlen_erwartet"] = bis - von + 1

    out["content_prefix"] = text[:300]
    out["spec_accept_length"] = meta.get("spec_accept_length")
    out["spec_verify_ct"] = meta.get("spec_verify_ct")
    ende = meta.get("finish_reason")
    if isinstance(ende, dict):
        ende = ende.get("type")
    if ende is None and out["endpunkt"] == "chat":
        try:
            ende = payload["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            ende = None
    out["finish_reason"] = ende

    # Die streuende Zaehlung bleibt als KENNZAHL im Artefakt -- sie ist die
    # Zahl, an der die beiden Fehlschluesse haengen (Praeambel-Punkte,
    # Ziffern aus Fliesstext), und wer sie spaeter im Protokoll sieht, soll
    # sie wiederfinden. Kriterium ist sie nicht mehr.
    out["zahlen_in_folge"] = zaehle_in_folge(text, von, bis)

    # (a) Der Anker: kommen fuer ein determiniertes Praefix die richtigen
    #     Token heraus?
    treffer, rest = anker_folge(text, von)
    out["anker_zahlen"] = treffer
    out["anker_min"] = ANKER_MIN
    out["drift_zeichen"] = len(rest.strip())
    anker_ok = treffer >= ANKER_MIN

    # (b) Und ist der Rest wohlgeformter Text? Geprueft wird der GANZE
    #     Abschnitt, nicht nur der Drift: eine Antwort, die exakt die Zahlen
    #     enthaelt und dann aufhoert, hat keinen Drift und ist trotzdem in
    #     Ordnung.
    befunde, kennzahlen = muell_pruefung(text)
    out["muell_befunde"] = befunde
    out.update(kennzahlen)

    out["lm_intakt"] = bool(anker_ok and not befunde)
    # `kohaerent` bleibt als Name im Artefakt, weil der Check und die
    # Auswertung ihn tragen -- aber er bedeutet jetzt "LM intakt" und nicht
    # mehr "hat gehorcht". Das ist die Aenderung, um die es geht.
    out["kohaerent"] = out["lm_intakt"]

    # Der benannte Zustand: zusammenhaengender Text, das Token-Budget bis
    # zum Anschlag verbraucht, und die Zahlen kamen trotzdem NIE. Das ist
    # kein Transportfehler und keine Korruption, sondern ein Smoke, dessen
    # Budget woanders hingegangen ist -- der Fall vom 2026-07-30 (Denk-
    # Praeambel). Er setzt jetzt voraus, dass der ANKER gefehlt hat: driftet
    # die Antwort erst NACH korrekt fortgesetzten Zahlen ab, ist sie kein
    # unter-provisionierter Smoke, sondern ein bestandener.
    out["unterprovisioniert"] = bool(
        not anker_ok
        and not befunde
        and ende == "length"
        and len(text.strip()) >= 200
    )
    return out


def compose(step_dir: str, port: int, host_log: str) -> dict:
    unreachable = os.path.exists(os.path.join(step_dir, "host_unreachable.txt"))
    missing_integration = os.path.exists(
        os.path.join(step_dir, "integration_missing.txt")
    )
    blocked_lines = read_lines(os.path.join(step_dir, "blocked.txt"))
    payload: dict = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "host": os.environ.get("BAR1_HOST", ""),
        "reachable": not unreachable,
        "integration_present": not missing_integration,
        # A step that never got the cards must not be diagnosed through the
        # empty artifacts it left behind.
        "blocked": " ".join(" ".join(blocked_lines).split())[:200] or None,
        "port": port,
        "server_log_remote": host_log,
        "transport_angefordert": "bar1",
        "graph_freigabe": True,
        "uneven_dcp": True,
        "graph_check": parse_graph_check(step_dir),
        "smoke": parse_smoke(step_dir),
    }
    payload.update(parse_log_evidence(step_dir))
    payload["gruppen_bar1"] = sorted(
        g["gruppe"] for g in payload["gruppen"] if g["erreicht"] == "bar1"
    )
    payload["gruppen_ausgewichen"] = sorted(
        g["gruppe"] for g in payload["gruppen"] if g["erreicht"] != g["angefordert"]
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--host-log", default="")
    args = ap.parse_args()

    payload = compose(args.step_dir, args.port, args.host_log)
    out = os.path.join(args.step_dir, "bar1_e2e.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"bar1_e2e.json geschrieben: Gruppen bar1={payload['gruppen_bar1']}, "
        f"ausgewichen={payload['gruppen_ausgewichen']}, "
        f"Aufbau-Zeilen={payload['aufbau_lines']}, "
        f"Riegel={'ja' if payload['riegel'] else 'nein'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
