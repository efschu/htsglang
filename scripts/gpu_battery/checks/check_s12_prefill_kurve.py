#!/usr/bin/env python3
"""s12 check -- is the prefill curve a measurement, not just a pile of numbers?

Verified:

  * every PLANNED session count has a number in BOTH arms. Half a pair is not a
    comparison, and an arm that broke off mid-run leaves exactly that,
  * the arms really alternated. A,B,A,B is the difference between a comparison
    and two afternoons; blockwise measurement is the one methodological error
    that cannot be repaired afterwards,
  * each point's boot ran the arm it claims. For bar1: every communicator group
    reports ERREICHT=bar1 -- the requested name says bar1 even when the group
    fell back to gloo, and one mixed group makes the point a mixed point. For
    the baseline: NO HTCCL group at all,
  * a decode point at bs=1 and one at bs=16 per arm,
  * an output sample per arm is persisted. A fast garbage run looks good in a
    throughput table (Messregel 5),
  * the baseline reproduces the known host-path numbers within the tolerance.
    It does not: STOP -- this environment is not the one those numbers came
    from, and comparing against them would be comparing two rigs.

Deliberately NOT judged -- and this is the point of the whole step: whether the
bar1 curve is flat or rising, and by how much it beats the baseline. Flat is a
finding, rising is a finding. A check that graded it would be deciding the
question the measurement exists to answer.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    is_number,
    load_json,
    require_envelope,
    run_check,
)

STEP = "s12_prefill_kurve"
KIND = "bar1_prefill_kurve"
ARME = ("bar1", "grundlinie")
DECODE_BATCHES = (1, 16)
REQUIRED_GROUP_PREFIXES = ("tp", "dcp")
MIN_SAMPLE_CHARS = 50


def _rate(payload: dict, arm: str, sessions: int):
    return (payload.get("kurven") or {}).get(arm, {}).get(str(sessions))


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "prefill_kurve.json")
    if not os.path.exists(path):
        raise CheckStop(
            f"prefill_kurve.json fehlt ({path}) -- der Schritt ist nicht gelaufen"
        )
    payload = load_json(path, "prefill_kurve.json")
    require_envelope(payload, KIND, "prefill_kurve.json", 1)

    if not payload.get("host_erreichbar"):
        raise CheckStop("Host nicht erreichbar -- es wurde nichts gemessen")
    if not payload.get("integration_vorhanden"):
        raise CheckStop(
            "die BAR1-Integration liegt nicht im Arbeitsbaum unter Test -- "
            "BAR1_HOST_WT setzen"
        )
    if payload.get("blockiert"):
        raise CheckStop(f"Schritt blockiert: {payload['blockiert']}")

    plan = payload.get("sessions_geplant") or []
    if not plan:
        raise CheckStop("kein Sessionplan im Ergebnis -- nichts zu pruefen")

    # --- completeness --------------------------------------------------------
    abbruch = payload.get("abbruch")
    for sessions in plan:
        for arm in ARME:
            rate = _rate(payload, arm, sessions)
            if not is_number(rate) or rate <= 0:
                raise CheckFail(
                    f"kein Durchsatz fuer Arm {arm!r} bei {sessions} Session(s) "
                    f"(Wert {rate!r})"
                    + (f"; Lauf abgebrochen: {abbruch}" if abbruch else "")
                )

    # --- interleaved, not blockwise -----------------------------------------
    reihenfolge = payload.get("reihenfolge") or []
    if len(reihenfolge) < 2 * len(plan):
        raise CheckFail(
            f"nur {len(reihenfolge)} Messpunkte fuer {2 * len(plan)} erwartete "
            "(je Sessionzahl zwei Arme)"
        )
    for a, b in zip(reihenfolge, reihenfolge[1:]):
        if a.get("arm") == b.get("arm"):
            raise CheckFail(
                f"zwei gleiche Arme hintereinander ({a.get('arm')} bei "
                f"{a.get('sessions')} und {b.get('sessions')} Sessions) -- "
                "blockweise gemessen, damit ist der Vergleich nicht verschraenkt"
            )

    # --- did each boot run the arm it claims? -------------------------------
    for eintrag in reihenfolge:
        arm = eintrag.get("arm")
        if not eintrag.get("beleg_vorhanden"):
            raise CheckFail(
                f"kein Transport-Beleg fuer {arm}/{eintrag.get('sessions')} -- "
                "ohne ihn ist der Arm des Messwerts unbelegt"
            )
        gruppen = eintrag.get("gruppen") or []
        if arm == "grundlinie":
            if gruppen:
                raise CheckFail(
                    f"Grundlinien-Boot bei {eintrag.get('sessions')} Sessions "
                    f"meldet HTCCL-Gruppen {[g.get('gruppe') for g in gruppen]} -- "
                    "die Grundlinie darf keine SGLANG_HTCCL*-Variable sehen"
                )
            continue
        if not gruppen:
            raise CheckFail(
                f"bar1-Boot bei {eintrag.get('sessions')} Sessions meldet keine "
                "einzige ERREICHT-Zeile -- HTCCL war nicht an"
            )
        for prefix in REQUIRED_GROUP_PREFIXES:
            treffer = [
                g for g in gruppen if str(g.get("gruppe", "")).startswith(prefix)
            ]
            if not treffer:
                raise CheckFail(
                    f"bar1-Boot bei {eintrag.get('sessions')} Sessions ohne Gruppe "
                    f"{prefix!r} (gemeldet: {[g.get('gruppe') for g in gruppen]})"
                )
            fremd = [g for g in treffer if g.get("erreicht") != "bar1"]
            if fremd:
                raise CheckFail(
                    f"bar1-Boot bei {eintrag.get('sessions')} Sessions: Gruppe "
                    f"{fremd[0].get('gruppe')!r} faehrt "
                    f"ERREICHT={fremd[0].get('erreicht')!r} -- gemischter Punkt, "
                    "kein bar1-Punkt"
                )

    # --- decode points -------------------------------------------------------
    decode = payload.get("decode") or []
    for arm in ARME:
        for batch in DECODE_BATCHES:
            treffer = [
                d
                for d in decode
                if d.get("arm") == arm
                and d.get("batch") == batch
                and is_number(d.get("decode_tok_s"))
            ]
            if not treffer:
                raise CheckFail(f"kein Decode-Punkt fuer Arm {arm!r} bei bs={batch}")

    # --- output samples ------------------------------------------------------
    samples = payload.get("output_samples") or {}
    for arm in ARME:
        sample = samples.get(arm)
        if not sample or len(str(sample)) < MIN_SAMPLE_CHARS:
            raise CheckFail(
                f"kein Ausgabe-Sample fuer Arm {arm!r} persistiert "
                f"({str(sample)[:40]!r}) -- ein schneller Muell-Lauf sieht in "
                "einer Durchsatztabelle gut aus"
            )

    # --- does the baseline reproduce? ---------------------------------------
    tol = payload.get("toleranz_pct")
    if not is_number(tol):
        tol = 5.0
    abw = payload.get("grundlinie_abweichung_pct") or {}
    bekannt = payload.get("grundlinie_bekannt") or {}
    schlimmster = None
    for sessions in plan:
        key = str(sessions)
        if key not in bekannt:
            continue
        if key not in abw:
            raise CheckStop(
                f"keine Abweichung gegen den bekannten Grundlinienwert bei "
                f"{sessions} Session(s) berechnet"
            )
        if schlimmster is None or abs(abw[key]) > abs(abw[schlimmster]):
            schlimmster = key
    if schlimmster is not None and abs(abw[schlimmster]) > tol:
        raise CheckStop(
            f"Grundlinie reproduziert nicht: bei {schlimmster} Session(s) "
            f"{_fmt(_rate(payload, 'grundlinie', int(schlimmster)))} statt "
            f"{bekannt[schlimmster]} tok/s ({abw[schlimmster]:+.1f} %, erlaubt "
            f"+-{tol} %) -- diese Umgebung ist nicht die, aus der die bekannten "
            f"Zahlen stammen ({payload.get('grundlinie_quelle')})"
        )


def _fmt(value) -> str:
    return f"{value:.1f}" if is_number(value) else str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
