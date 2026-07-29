#!/usr/bin/env python3
"""s11 check -- did the standard run really go over BAR1, and did it answer?

Order matters here, because the first thing that is wrong is the thing worth
reporting:

  1. host reachable, BAR1 code in the worktree under test (STOP -- nothing was
     measured),
  2. the graph gate. bar1_graph_check.py must have run and every GATE case must
     have passed. SGLANG_HTCCL_GRAPH_FREIGABE=1 without that evidence produces
     numbers from an operating point nobody can defend,
  3. THE BOLT. htccl._select raises instead of quietly dropping to the
     host-staged gloo level during a graph capture. If it fired, that is the
     coverage gap of the current integration -- reported with op and size, in
     its own wording, so nobody has to open a log to tell it from a crash,
  4. per-group attainment. EVERY communicator group must report ERREICHT=bar1.
     The requested transport name says bar1 either way; it is not evidence. One
     group on bar1 and one on gloo is a MIXED run: correct-looking, and its
     numbers may not be reported as bar1 numbers,
  5. a BAR1 setup line per group. The ERREICHT line is written at group init;
     the setup line is written when the aperture actually handed over the
     space,
  6. the smoke request: coherent output (the numbers 1..20 in order, counted,
     not judged) and spec_accept_length present as a number,
  7. no OOM / NCCL / watchdog marker in the harvested log lines.

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
    require_envelope(payload, KIND, "bar1_e2e.json", 1)

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

    riegel = payload.get("riegel")
    if riegel:
        raise CheckFail(
            f"RIEGEL: htccl._select hat {riegel.get('op')!r} mit "
            f"{riegel.get('bytes')} Byte waehrend einer CUDA-Graph-Aufzeichnung "
            "abgebrochen, weil bar1 die Operation in dieser Groesse nicht fuehrt "
            "-- Deckungsluecke im BAR1-Transport, nicht ein Absturz. Genau dieses "
            "Szenario nimmt die BAR1-Integration ab."
        )

    gruppen = payload.get("gruppen") or []
    if not gruppen:
        raise CheckFail(
            "keine einzige 'ERREICHT='-Zeile im Log -- ohne sie ist der Arm des "
            "Messwerts unbelegt (der angeforderte Transportname steht auch bei "
            "Ausfall auf bar1)"
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
    if not smoke.get("kohaerent"):
        raise CheckFail(
            f"Smoke-Ausgabe inkohaerent: nur {smoke.get('zahlen_in_folge')} von 20 "
            f"Zahlen in Folge (mindestens {MIN_ORDERED_NUMBERS}); Anfang: "
            f"{str(smoke.get('content_prefix'))[:80]!r}"
        )
    if not is_number(smoke.get("spec_accept_length")):
        raise CheckFail(
            f"spec_accept_length ist {smoke.get('spec_accept_length')!r} -- der "
            "Spec-Pfad laeuft nicht oder die Antwort traegt kein meta_info"
        )

    fatal = payload.get("fatal")
    if fatal:
        raise CheckFail(f"Fatal im Serverlog -- {fatal}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
