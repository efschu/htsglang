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
SCHEMA_VERSION = 3

#: Der Fortsetzungs-Prompt, den s11_bar1_e2e.sh an /generate schickt. Er
#: steht hier UND dort; dass die beiden zusammenpassen, nagelt
#: test_gpu_battery_checks_bar1.py am Quelltext der Schrittdatei fest --
#: sonst zaehlte der Parser eine Folge, die der Request nie angestossen hat.
SMOKE_PROMPT = "1 2 3 4"
#: Die Zahlen, die die Fortsetzung liefern muss. Beginnt bei 5, also hinter
#: dem Prompt: was im Prompt steht, ist kein Beleg fuer die Antwort.
ZAHLEN_VON = 5
ZAHLEN_BIS = 20
#: Wieviele davon in Folge dastehen muessen. Unveraendert 15 -- geaendert hat
#: sich der Nenner (16 statt 20), weil die ersten vier Zahlen jetzt im Prompt
#: stehen und nicht mehr mitzaehlen duerfen.
MIN_ZAHLEN_IN_FOLGE = 15

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
    """Wieviele der Zahlen ``von..bis`` in DIESER Reihenfolge im Text stehen.

    Eine Zahl, keine Meinung -- das war schon die Absicht und bleibt sie. Was
    dazugelernt wurde: der Zaehler sagt nichts darueber, WOHER die Treffer
    kommen. Im Lauf vom 2026-07-30 meldete er 3 fuer eine Antwort, die gar
    nicht gezaehlt hat -- die Treffer waren die Aufzaehlungspunkte "1.",
    "2.", "3." einer Denk-Praeambel. Deshalb faengt der Bereich jetzt bei 5
    an (die Fortsetzung von "1 2 3 4"): die kleinen Ziffern, die in jedem
    beliebigen Fliesstext vorkommen, zaehlen nicht mehr mit.
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

    treffer = zaehle_in_folge(text, von, bis)
    out["zahlen_in_folge"] = treffer
    out["kohaerent"] = treffer >= MIN_ZAHLEN_IN_FOLGE
    # Der benannte Zustand: zusammenhaengender Text, das Token-Budget bis
    # zum Anschlag verbraucht, und die Zahlen kamen trotzdem nicht. Das ist
    # kein Transportfehler und keine Korruption, sondern ein Smoke, dessen
    # Budget woanders hingegangen ist -- der Fall vom 2026-07-30. Er
    # bekommt einen eigenen Namen, damit ihn niemand als bar1-Befund liest.
    out["unterprovisioniert"] = bool(
        not out["kohaerent"] and ende == "length" and len(text.strip()) >= 200
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
