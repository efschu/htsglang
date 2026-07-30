#!/usr/bin/env python3
"""s11 check -- did the standard run really go over BAR1, and did it answer?

Order matters here, because the first thing that is wrong is the thing worth
reporting:

  1. host reachable, BAR1 code in the worktree under test (STOP -- nothing was
     measured),
  2. the graph gate. bar1_graph_check.py must have run and every GATE case must
     have passed. SGLANG_HTCCL_GRAPH_FREIGABE=1 without that evidence produces
     numbers from an operating point nobody can defend,
  3. was a log harvested at ALL. An empty evidence list means one of two very
     different things -- nobody looked, or nothing was there -- and only the
     second is a measurement. `log_quellen` names the files that existed,
  4. THE BOLT. htccl._select raises instead of quietly dropping to the
     host-staged gloo level during a graph capture. If it fired, that is the
     coverage gap of the current integration -- reported with op and size, in
     its own wording, so nobody has to open a log to tell it from a crash,
  5. a fatal in the log. This sits HERE and not at the end, deliberately: a
     boot that died takes the smoke request, the setup lines and everything
     else down with it, and reporting one of those instead means reporting the
     consequence while the cause is two fields away. That is exactly how a run
     whose capture aborted got reported as "no ERREICHT line in the log",
  6. per-group attainment. EVERY communicator group must report ERREICHT=bar1.
     The requested transport name says bar1 either way; it is not evidence. One
     group on bar1 and one on gloo is a MIXED run: correct-looking, and its
     numbers may not be reported as bar1 numbers,
  7. a BAR1 setup line per group. The ERREICHT line is written at group init;
     the setup line is written when the aperture actually handed over the
     space,
  8. the smoke request over /generate: coherent output (the continuation of
     "1 2 3 4", counted, not judged) and spec_accept_length present as a
     number. Three outcomes, not two -- "coherent text that never got to the
     numbers because the token budget went elsewhere" is its own verdict
     (UNTER-PROVISIONIERT) and says nothing about the transport.

NOT judged: the HEIGHT of spec_accept_length, the setup duration, and any
throughput. s11 answers "does the direct path carry a real run"; s12 answers
"what does it cost".
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    classify_missing_result,
    is_number,
    load_json,
    require_envelope,
    run_check,
)

STEP = "s11_bar1_e2e"
KIND = "bar1_e2e"

# With SGLANG_UNEVEN_DCP=1 the standard run builds two communicator groups. The
# names are prefixes because the counter suffix (tp:0, dcp:0) is an
# implementation detail of _get_unique_name.
REQUIRED_GROUP_PREFIXES = ("tp", "dcp")
MIN_ORDERED_NUMBERS = 15


def _group_for(prefix: str, gruppen: list) -> list:
    return [g for g in gruppen if str(g.get("gruppe", "")).startswith(prefix)]


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "bar1_e2e.json")
    classify_missing_result(step_dir, "bar1_e2e", path, "bar1_e2e.json")
    payload = load_json(path, "bar1_e2e.json")
    require_envelope(payload, KIND, "bar1_e2e.json", 3)

    if not payload.get("reachable"):
        raise CheckStop(
            f"Host {payload.get('host') or '?'} nicht erreichbar -- der Lauf hat "
            "nicht stattgefunden"
        )
    if not payload.get("integration_present"):
        raise CheckStop(
            "die BAR1-Integration liegt nicht im Arbeitsbaum unter Test "
            "(htccl_bar1.py / benchmark/bar1_graph_check.py) -- BAR1_HOST_WT setzen"
        )
    if payload.get("blocked"):
        raise CheckStop(f"Schritt blockiert: {payload['blocked']}")

    gate = payload.get("graph_check") or {}
    if not gate.get("gate_cases"):
        raise CheckStop(
            "bar1_graph_check hat keinen Gate-Fall gemeldet (rc="
            f"{gate.get('rc')!r}) -- das Tor ist nicht gelaufen, also ist "
            "SGLANG_HTCCL_GRAPH_FREIGABE=1 unbelegt"
        )
    if not gate.get("alle_bestanden"):
        raise CheckFail(
            f"Graph-Tor gefallen: {','.join(gate.get('gefallen') or []) or '?'} "
            f"(rc={gate.get('rc')!r}) -- unter CUDA-Graphen ist der Direktpfad "
            "damit nicht freigegeben"
        )

    # Erst die Frage, OB jemand geschaut hat. Eine leere Beweislage aus einer
    # Schrittablage ohne Logauszug ist kein Befund ueber den Lauf, sondern
    # einer ueber die Ernte -- und die beiden auseinanderzuhalten ist der
    # Unterschied zwischen "der Lauf hat nichts gemeldet" und "wir haben
    # nicht nachgesehen".
    if "log_quellen" not in payload:
        raise CheckStop(
            "bar1_e2e.json nennt kein 'log_quellen' -- das Artefakt stammt von "
            "einem aelteren Erzeuger, der die Beweislage nur aus htccl_lines.txt "
            "gelesen hat und auf jedem Abbruchweg leer ausging"
        )
    if not payload["log_quellen"]:
        raise CheckStop(
            "kein Logauszug in der Schrittablage (weder htccl_lines.txt noch "
            "server.log) -- niemand hat geschaut, also ist hier nichts zu "
            "entscheiden"
        )

    riegel = payload.get("riegel")
    if riegel:
        raise CheckFail(
            f"RIEGEL: htccl._select hat {riegel.get('op')!r} mit "
            f"{riegel.get('bytes')} Byte waehrend einer CUDA-Graph-Aufzeichnung "
            "abgebrochen, weil bar1 die Operation in dieser Groesse nicht fuehrt "
            "-- Deckungsluecke im BAR1-Transport, nicht ein Absturz. Genau dieses "
            "Szenario nimmt die BAR1-Integration ab."
        )

    # Und danach sofort der Absturz. Ein toter Boot reisst Smoke-Request,
    # Aufbauzeilen und ERREICHT-Zeilen mit -- wer eine von denen meldet,
    # meldet die Folge und laesst die Ursache zwei Felder weiter stehen.
    fatal = payload.get("fatal")
    if fatal:
        raise CheckFail(f"Fatal im Serverlog -- {fatal}")

    gruppen = payload.get("gruppen") or []
    if not gruppen:
        raise CheckFail(
            "keine einzige 'ERREICHT='-Zeile im Log (Quellen: "
            f"{payload['log_quellen']}, {payload.get('log_zeilen')} Zeilen) -- "
            "ohne sie ist der Arm des Messwerts unbelegt (der angeforderte "
            "Transportname steht auch bei Ausfall auf bar1)"
        )
    for prefix in REQUIRED_GROUP_PREFIXES:
        found = _group_for(prefix, gruppen)
        if not found:
            raise CheckFail(
                f"keine Gruppe {prefix!r} im Log (gemeldet: "
                f"{[g.get('gruppe') for g in gruppen]}) -- bei "
                "SGLANG_UNEVEN_DCP=1 muessen tp und dcp beide auftauchen"
            )
        nicht_bar1 = [g for g in found if g.get("erreicht") != "bar1"]
        if nicht_bar1:
            g = nicht_bar1[0]
            raise CheckFail(
                f"Gruppe {g.get('gruppe')!r} faehrt ERREICHT={g.get('erreicht')!r} "
                f"(angefordert {g.get('angefordert')!r}), waehrend "
                f"{payload.get('gruppen_bar1')} auf bar1 laufen -- ein gemischter "
                "Lauf, dessen Zahlen nicht als bar1-Zahlen gelten"
            )

    kasse = payload.get("aufbau_gruppen") or []
    if not payload.get("aufbau_lines"):
        raise CheckFail(
            "keine 'HTCCL-BAR1: Aufbau in'-Zeile -- kein Rang hat eine "
            "BAR1-Region wirklich aufgebaut"
        )
    for prefix in REQUIRED_GROUP_PREFIXES:
        if not any(str(name).startswith(prefix) for name in kasse):
            raise CheckFail(
                f"kein BAR1-Aufbau fuer Gruppe {prefix!r} belegt (aufgebaut: "
                f"{kasse}) -- die ERREICHT-Zeile allein steht schon vor der "
                "Apertur-Zusage"
            )

    smoke = payload.get("smoke") or {}
    if not smoke.get("vorhanden"):
        raise CheckFail("kein Smoke-Request abgesetzt (smoke.json fehlt)")
    if smoke.get("error"):
        raise CheckFail(f"Smoke-Request meldet: {smoke['error']}")
    if smoke.get("endpunkt") == "chat":
        raise CheckStop(
            "der Smoke kam ueber /v1/chat/completions -- dort haengt die "
            "Ausgabe am Chat-Template (Denk-Praeambel) und meta_info ist "
            "opt-in, also ist weder die Kohaerenz noch spec_accept_length "
            "aussagekraeftig. Das Artefakt stammt aus einem Lauf vor der "
            "Umstellung auf /generate; er ist zu wiederholen, nicht zu "
            "bewerten"
        )
    # DER BENANNTE ZUSTAND. Zusammenhaengender Text, Token-Budget bis zum
    # Anschlag verbraucht, Zahlen nie drangekommen. Am 2026-07-30 war das
    # eine Denk-Praeambel bei Temperatur 0 -- ein unter-provisionierter
    # Smoke, keine Transportstoerung und keine Korruption. Eigene Meldung,
    # damit er nie als bar1-Befund gelesen wird.
    if smoke.get("unterprovisioniert"):
        raise CheckFail(
            f"Smoke UNTER-PROVISIONIERT: die Antwort ist zusammenhaengend, hat "
            f"aber das Token-Budget aufgebraucht (finish_reason="
            f"{smoke.get('finish_reason')!r}), bevor die Zahlen drankamen -- "
            f"{smoke.get('zahlen_in_folge')} von {smoke.get('zahlen_erwartet')}. "
            f"Das ist ein Befund ueber den Smoke, NICHT ueber den Transport: "
            f"die Kollektive sind byte-belegt, und ein Trajektorienwechsel bei "
            f"Temperatur 0 liegt in der erwarteten Numerik-Klasse. Anfang: "
            f"{str(smoke.get('content_prefix'))[:80]!r}"
        )
    if not smoke.get("kohaerent"):
        raise CheckFail(
            f"Smoke-Ausgabe inkohaerent: nur {smoke.get('zahlen_in_folge')} von "
            f"{smoke.get('zahlen_erwartet')} Zahlen in Folge (mindestens "
            f"{MIN_ORDERED_NUMBERS}); Anfang: "
            f"{str(smoke.get('content_prefix'))[:80]!r}"
        )
    if not is_number(smoke.get("spec_accept_length")):
        raise CheckFail(
            f"spec_accept_length ist {smoke.get('spec_accept_length')!r} -- der "
            "Spec-Pfad laeuft nicht oder die Antwort traegt kein meta_info"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
