"""PROFIL-EDITOR S1: the CURATED core of the value catalog -- the values a person is meant to touch.

Each entry: one plain sentence (``text``), what it buys and what it costs (``gain`` / ``cost``, the style of
``planner/tooltips.py``), the dependencies on other values (``depends``) and whether the planner derives it
(``planner_derived``).  Everything else in the catalog is harvested from the code (``profile_catalog.py``).

Rules for this file (the plan's "nie aus Vermutung"):

* A sentence states what the SOURCE says (the argparse ``help=``, the comment in ``environ.py``, the entrypoint's
  own check); where a number or a consequence is not in a source it is not here.
* A dependency edge ``{"to", "rel", "effect", "calc"}``: ``rel`` is one of
    ``tauscht``       moving it here takes from / gives to the other value (a zero-sum trade),
    ``braucht``       does nothing / is refused without the other,
    ``schliesst_aus`` cannot be combined with the other,
    ``abgeleitet_von`` the planner computes it from the other,
    ``skaliert_mit``  grows or shrinks together with the other.
  ``calc`` is ``"text"`` (this edge is stated in words) or ``"S4"`` (the live consequence in MiB / tokens / ms is
  computed from hardware + model profile in stage S4; until then the edge is words only).
* ``test_profile_catalog_1003`` pins every flag named here (and every ``to`` that starts with ``--``) against
  the launcher / ServerArgs, so an edge cannot point at a flag that does not exist.
"""

from __future__ import annotations

from typing import Dict


def _d(to, rel, effect, calc="text"):
    return {"to": to, "rel": rel, "effect": effect, "calc": calc}


CURATED: Dict[str, Dict[str, object]] = {
    # ------------------------------------------------------------------ Aufteilung auf die Karten
    "--pp-stage-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "einfach", "planner_derived": True,
        "text": "Wie viele Layer jede Karte in der Prefill-Gruppe (P, Pipeline) trägt, z. B. 42,11,11. Ohne Angabe rechnet der Planer den Schnitt.",
        "gain": "Mehr Layer auf die schnellere Karte verkürzt die langsamste Stufe und damit die Prefill-Zeit.",
        "cost": "Jeder Layer auf einer Karte belegt dort Gewichte; es bleibt weniger Platz für KV (Kontext). Die andere Karte gewinnt genau diesen Platz.",
        "depends": [
            _d("--pp-attn-stage-ratio", "braucht", "Beide gehören zusammen: der Launcher verweigert das eine ohne das andere (W40).", "text"),
            _d("--rank-gpu-memory-mib", "tauscht", "Layer von Karte A nach Karte B: A verliert Gewichte-Platz im Budget und damit KV-Kontext, B gewinnt ihn.", "S4"),
            _d("--max-kv-per-request", "skaliert_mit", "Der erreichbare Kontext ist das Minimum über die Stufen; jede Verschiebung verändert, welche Karte ihn begrenzt.", "S4"),
            _d("--pp-solve-pool-floor", "skaliert_mit", "Der Solver sucht den schnellsten Schnitt, dessen KV-Pool über diesem Boden bleibt.", "text"),
        ]},
    "--pp-attn-stage-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": True,
        "text": "Wie viele der vollen Attention-Layer jede Prefill-Stufe trägt (z. B. 10,3,3). Attention-Layer haben KV; die anderen Layer nicht.",
        "gain": "Legt fest, wo KV überhaupt entsteht: mehr Attention-Layer auf einer Karte heißt mehr KV-Bytes je Token dort.",
        "cost": "Muss zum Layer-Schnitt passen; ohne --pp-stage-ratio verweigert der Launcher.",
        "depends": [_d("--pp-stage-ratio", "braucht", "Zusammen anzugeben (W40).", "text"),
                    _d("--rank-gpu-memory-mib", "tauscht", "Mehr Attention-Layer auf einer Karte kosten dort KV-Platz je Token.", "S4")]},
    "--rank-gpu-memory-mib": {
        "kind": "flag", "group": "Speicher", "level": "einfach", "planner_derived": True, "scope": "server",
        "text": "Das gesamte Speicherbudget jeder Karte in MiB für diese Gruppe (Liste, Karte 1 zuerst). Der Planer rechnet es aus Kartengröße minus Treiber-Carve, Schlafrest und Reserve.",
        "gain": "Ein größeres Budget gibt Gewichten, Experten und KV mehr Platz.",
        "cost": "Der Wert ist das ganze Budget, kein Sicherheitsabschlag kommt darauf: zu hoch gewählt endet der Start im OOM.",
        "depends": [_d("--user-reserve-mib", "abgeleitet_von", "Der Planer zieht die Reserve für Ihre anderen Prozesse vom Budget ab.", "text"),
                    _d("--pp-stage-ratio", "tauscht", "Das Budget wird zwischen Gewichten (Layer-Schnitt), Experten und KV geteilt.", "S4"),
                    _d("--rank-moe-resident-fraction", "tauscht", "Mehr residente Experten füllen den freien Platz, der sonst KV wäre.", "S4")]},
    "--rank-tp-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": True, "scope": "server",
        "text": "Ungleiche Tensor-Parallelität: Gewichte je Karte für jede geteilte Dimension (z. B. 2,1,1 gibt Karte 1 die Hälfte). 'auto' leitet sie aus den Budgets ab.",
        "gain": "Eine große und zwei kleine Karten werden gleich voll; der KV-Pool (Minimum über die Karten) wächst.",
        "cost": "Die Summe der Gewichte muss jede geteilte Dimension teilen.",
        "depends": [_d("--rank-gpu-memory-mib", "abgeleitet_von", "'auto' rechnet die Gewichte aus der Budgetliste.", "text"),
                    _d("--rank-mlp-ratio", "braucht", "MLP- und MoE-Anteile werden getrennt nachgestellt, Attention/KV folgen diesem Vektor.", "text")]},
    "--rank-mlp-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": True, "scope": "server",
        "text": "Nur für die dichten MLP-Gewichte: wie viele Einheiten jede Karte trägt. Attention/KV folgen weiter --rank-tp-ratio.",
        "gain": "Verschiebt Gewichts-Bytes zwischen Karten, so wächst der KV-Pool der engsten Karte.",
        "cost": "Mehr MLP auf einer Karte belegt dort Platz, den sie sonst für Kontext hätte.",
        "depends": [_d("--rank-tp-ratio", "braucht", "Setzt einen aktiven ungleichen Plan voraus.", "text"),
                    _d("--rank-gpu-memory-mib", "tauscht", "Gewichte gegen KV je Karte.", "S4")]},
    "--rank-moe-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": True, "scope": "server",
        "text": "Nur für die MoE-Experten-Gewichte: wie viele Einheiten jede Karte trägt (z. B. 183,137,168). Entscheidet auch, welche Karte welchen Experten rechnet.",
        "gain": "Hauptschalter für den KV-Pool bei MoE-Modellen: Experten sind der größte verschiebbare Block.",
        "cost": "Mehr Experten auf einer Karte heißt dort weniger KV und mehr Last in der Runde dieser Karte.",
        "depends": [_d("--rank-moe-resident-fraction", "skaliert_mit", "Der Anteil bestimmt, wie viel des eigenen Expertenanteils auf der Karte residient liegt.", "text"),
                    _d("--rank-gpu-memory-mib", "tauscht", "Experten gegen KV je Karte.", "S4"),
                    _d("SGLANG_UNEVEN_MOE_EXPERT_SHARD", "braucht", "Die Experten werden nach Index auf die Karten verteilt.", "text")]},
    "--rank-moe-resident-fraction": {
        "kind": "flag", "group": "Speicher", "level": "einfach", "planner_derived": True, "scope": "server",
        "text": "Welcher Anteil der Experten einer Karte dauerhaft im VRAM liegt (Liste je Karte); der Rest wartet im pinned Host-Speicher und wird bei Bedarf geholt.",
        "gain": "Mehr residente Experten: weniger PCIe-Verkehr, schnelleres Decode.",
        "cost": "Jeder residente Experte belegt VRAM, der sonst KV (Kontext) wäre. Weniger Anteil spart VRAM, kostet Host-Zugriffe.",
        "depends": [_d("--rank-gpu-memory-mib", "tauscht", "Residente Experten gegen KV-Platz auf derselben Karte.", "S4"),
                    _d("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "schliesst_aus", "Dieselbe Größe als Env; bei Widerspruch verweigert der Start.", "text"),
                    _d("--pp-cut-expert-device-fraction", "skaliert_mit", "Der Prefill-Schnitt-Solver preist Experten-Bytes mit diesem Anteil.", "text")]},
    "--pp-cut-expert-device-fraction": {
        "kind": "flag", "group": "Speicher", "level": "experte", "planner_derived": True,
        "text": "Residenter Experten-Anteil je Prefill-Stufe (Vektor). Pflicht bei einem Modell mit Experten; der Schnitt-Solver preist die Expertenbytes mit diesem Anteil.",
        "gain": "Bestimmt, wie viel der Prefill-Gruppe pro Karte für Experten reserviert wird.",
        "cost": "Mehr Anteil: weniger Platz für KV in der Prefill-Gruppe.",
        "depends": [_d("--pp-cut-expert-lru-rows", "skaliert_mit", "Beide zusammen bestimmen die Expertenbytes je Stufe.", "text"),
                    _d("--pp-stage-ratio", "skaliert_mit", "Der Schnitt wird mit diesen Expertenbytes gerechnet.", "S4")]},
    "--pp-cut-expert-lru-rows": {
        "kind": "flag", "group": "Speicher", "level": "experte", "planner_derived": True,
        "text": "LRU- und Staging-Zeilen des Experten-Pools je Layer und Stufe (Vektor, Standard 0). Entspricht SGLANG_MOE_SCRATCH_SLOTS der P-Gruppe.",
        "gain": "Mehr Zeilen halten häufig gebrauchte Experten bereit, ohne sie dauerhaft zu fixieren.",
        "cost": "Jede Zeile belegt VRAM je Layer.",
        "depends": [_d("--pp-cut-expert-device-fraction", "skaliert_mit", "Beide zusammen bestimmen die Expertenbytes.", "text"),
                    _d("SGLANG_MOE_SCRATCH_SLOTS", "abgeleitet_von", "Dieselbe Größe als Env der P-Gruppe.", "text")]},
    "--pp-solve-pool-floor": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False,
        "text": "Harte Untergrenze (in Welt-KV-Token) für den KV-Pool, den der Schnitt-Solver wählen darf. 0 schaltet den Boden ab.",
        "gain": "Hält den Kontext, den ein schnellerer Schnitt sonst opfern würde.",
        "cost": "Ein hoher Boden schließt schnelle, aber KV-arme Schnitte aus.",
        "depends": [_d("--pp-stage-ratio", "skaliert_mit", "Begrenzt, welche Schnitte der Solver wählen darf.", "text")]},
    # ------------------------------------------------------------------ Kontext und Prefill
    "--max-kv-per-request": {
        "kind": "flag", "group": "Kontext", "level": "einfach", "planner_derived": False,
        "text": "KV-Obergrenze je Anfrage für beide Gruppen. Ohne Angabe gilt der Kontext des Modells.",
        "gain": "Ein kleineres Ziel gibt anderen Anfragen mehr Pool.",
        "cost": "Ein größeres Ziel braucht mehr KV-Bytes (Token x KV-Zelle des Modells) und kann nicht über dem Pool der engsten Karte liegen.",
        "depends": [_d("--rank-gpu-memory-mib", "skaliert_mit", "Der Pool der engsten Karte setzt die Grenze.", "S4"),
                    _d("--pp-stage-ratio", "skaliert_mit", "Der Layer-Schnitt bestimmt, welche Karte den Kontext begrenzt.", "S4")]},
    "--tp-prefill-max-tokens": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "X: wie viele neue Token die Decode-Gruppe D selbst prefillen darf. Größere Anfragen gehen an P. 0 lässt den Planer ableiten.",
        "gain": "Kurze Anfragen laufen sofort auf D ohne Flip.",
        "cost": "D rechnet Prefill langsamer als P und hält dabei seine Decode-Sitze.",
        "depends": [_d("--x-ceiling-tokens", "braucht", "Die Obergrenze, bis zu der X live steigen darf.", "text"),
                    _d("--d-short-drain-tokens", "skaliert_mit", "Der Drain gilt nur für Anfragen, die selbst unter X liegen.", "text")]},
    "--x-ceiling-tokens": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "Obergrenze, bis zu der das X (D-Prefill-Schwelle) live steigen darf. 0 = aus, D und Front behalten das Start-X.",
        "gain": "X darf sich der Lage anpassen.", "cost": "Eine Anfrage zwischen Start-X und Live-X geht nur an D, wenn sonst nichts läuft.",
        "depends": [_d("--tp-prefill-max-tokens", "braucht", "Das Start-X, ab dem die Obergrenze wirkt.", "text")]},
    "--p-chunk-policy": {
        "kind": "flag", "group": "Prefill", "level": "einfach", "planner_derived": False,
        "text": "Wie breit die Prefill-Chunks sind: 'fixed' (jeder Forward nimmt --chunked-prefill-size) oder 'dynamic' (ein Plan je Anfrage mit Anlauf- und Auslauf-Rampe).",
        "gain": "dynamic kann den Prefill-Durchsatz bei langen Prompts erhöhen.",
        "cost": "Größere Chunks brauchen mehr Aktivierungs-Speicher je Karte und lassen weniger Platz für KV.",
        "depends": [_d("--p-chunk-max", "braucht", "dynamic: die breiteste Chunk-Größe.", "text"),
                    _d("--p-chunk-model", "braucht", "dynamic: das Zeitmodell je Stufe.", "text"),
                    _d("--chunked-prefill-size", "schliesst_aus", "Bei dynamic wird --p-chunk-max zur Chunk-Größe der P-Gruppe.", "text")]},
    "--p-chunk-max": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "Nur mit --p-chunk-policy dynamic: der breiteste Chunk; er wird zur Chunk-Größe der P-Gruppe.",
        "gain": "Größerer Chunk: höherer Prefill-Durchsatz.", "cost": "Aktivierungs-Spitze wächst mit der Chunk-Breite (weniger KV-Platz).",
        "depends": [_d("--p-chunk-policy", "braucht", "Wirkt nur bei dynamic.", "text"),
                    _d("--rank-gpu-memory-mib", "tauscht", "Aktivierung gegen KV auf jeder P-Karte.", "S4")]},
    "--p-chunk-model": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "Nur mit dynamic: das Zeitmodell je Stufe ('builtin-int8', 'fit:<P.log>' oder eine JSON-Datei).",
        "gain": "Ein auf die eigene Hardware gefittetes Modell plant bessere Chunks.", "cost": "Das eingebaute Modell ist auf dem Referenz-Rig gemessen.",
        "depends": [_d("--p-chunk-policy", "braucht", "Wirkt nur bei dynamic.", "text")]},
    "--p-chunk-mscale": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "Nur mit dynamic und eingebautem/gefittetem Modell: die Kostenkurve über der Chunk-Breite je Stufe.",
        "gain": "Berücksichtigt, dass breite Chunks auf der 5090 billiger je Token sind.", "cost": "Hochrechnung bis zur Messung (laut Hilfetext).",
        "depends": [_d("--p-chunk-policy", "braucht", "Wirkt nur bei dynamic.", "text")]},
    "--chunked-prefill-size": {
        "kind": "flag", "group": "Prefill", "level": "einfach", "planner_derived": False, "scope": "server",
        "text": "Wie viele Token ein Prefill-Forward höchstens nimmt.",
        "gain": "Größer: weniger Forwards je Prompt.", "cost": "Größer: höhere Aktivierungs-Spitze auf jeder Karte, weniger Platz für KV.",
        "depends": [_d("--rank-gpu-memory-mib", "tauscht", "Aktivierung gegen KV.", "S4"),
                    _d("--tp-prefill-max-tokens", "skaliert_mit", "X liegt nie unter dieser Größe.", "text")]},
    "--p-bs": {
        "kind": "flag", "group": "Prefill", "level": "einfach", "planner_derived": False,
        "text": "Gleichzeitige Anfragen der Prefill-Gruppe (und Nebenläufigkeit der Front, Leg 1).",
        "gain": "Mehr Anfragen gleichzeitig im Prefill.", "cost": "Dimensioniert auch den Request-Pool der Gruppe P (vor der Budgetrechnung gelöst).",
        "depends": [_d("--rank-gpu-memory-mib", "tauscht", "Der Request-Pool kostet Speicher je Karte.", "S4")]},
    "--d-bs": {
        "kind": "flag", "group": "Decode", "level": "einfach", "planner_derived": False,
        "text": "Gleichzeitige Anfragen der Decode-Gruppe und Zahl der Front-Sitze.",
        "gain": "Mehr Sitze: mehr Nutzer gleichzeitig.", "cost": "Jeder Sitz braucht Mamba-/State-Speicher und verlängert die Decode-Runde.",
        "depends": [_d("--rank-gpu-memory-mib", "tauscht", "Sitze (State-Pools) gegen KV.", "S4")]},
    "--max-running-requests": {
        "kind": "flag", "group": "Decode", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Wie viele Anfragen eine Gruppe gleichzeitig laufen lässt (der Launcher setzt es aus --p-bs / --d-bs).",
        "gain": "Mehr gleichzeitig.", "cost": "Größere Request-Pools.",
        "depends": [_d("--d-bs", "abgeleitet_von", "Bei D gleich der Sitzzahl.", "text"), _d("--p-bs", "abgeleitet_von", "Bei P gleich --p-bs.", "text")]},
    # ------------------------------------------------------------------ Speicher / Host
    "--user-reserve-mib": {
        "kind": "flag", "group": "Speicher", "level": "einfach", "planner_derived": False,
        "text": "VRAM in MiB je Karte, der für Ihre eigenen anderen Prozesse frei bleibt (Skalar oder Liste in Karten-Ordinal-Reihenfolge, 5090 zuerst).",
        "gain": "Hält Platz für andere Programme auf den Karten.", "cost": "Reduziert direkt das Budget und damit KV/Experten. Standard 0.",
        "depends": [_d("--rank-gpu-memory-mib", "abgeleitet_von", "Wird vom Budget abgezogen.", "text")]},
    "--store-max-gb": {
        "kind": "flag", "group": "Cache", "level": "einfach", "planner_derived": False,
        "text": "Größe des Stores (Verzeichnis auf der Platte) in GB, absolut. 0 = aus dem Pool der P-Gruppe ableiten.",
        "gain": "Mehr Store hält mehr Seiten außerhalb des VRAM.", "cost": "Platz im Store-Verzeichnis; ein Wert unter dem P-Pool wird mit Namen verweigert (W57), nie still gekürzt.",
        "depends": [_d("--pp-stage-ratio", "skaliert_mit", "Der P-Pool (Grundlage der Ableitung) hängt vom Layer-Schnitt ab.", "text")]},
    "--pin-ledger-arm-m": {
        "kind": "flag", "group": "Host", "level": "experte", "planner_derived": False,
        "text": "Fixiert die Größe M (MiB) des Mamba-Host-Pools der Host-Ledger-Armierung; 0 = der Planer wählt.",
        "gain": "Mehr Host-Anker-Slots für lange Prompts.", "cost": "Host-Speicher.",
        "depends": [_d("PROFILE_MEMAVAIL_MIN_GIB", "skaliert_mit", "Mehr Pin heißt weniger freier Host-Speicher.", "text")]},
    "--host-riegel-gib": {
        "kind": "flag", "group": "Host", "level": "experte", "planner_derived": False,
        "text": "Laufzeit-Riegel (GiB nicht rückholbar), unter dem eine abgelehnte Host-Ledger-Abweichung akzeptiert wird; muss unter der harten Grenze liegen.",
        "gain": "Erlaubt einen Start trotz abgelehnter Host-Rechnung.", "cost": "Der Riegel fängt nur ab, was er sieht.",
        "depends": [_d("--host-ledger-deviation", "braucht", "Gehört zu einer benannten Abweichung.", "text")]},
    "--host-ledger-deviation": {
        "kind": "flag", "group": "Host", "level": "experte", "planner_derived": False,
        "text": "Akzeptiert ein abgelehntes Host-Ledger-Urteil unter einem benannten Grund; verlangt --host-riegel-gib.",
        "gain": "Start trotz Ablehnung der Host-Rechnung.", "cost": "Kein Term wird neu gepreist.",
        "depends": [_d("--host-riegel-gib", "braucht", "Verlangt den Laufzeit-Riegel.", "text")]},
    "--weg2-weight-source": {
        "kind": "flag", "group": "Speicher", "level": "experte", "planner_derived": False,
        "text": "Woher eine aufwachende Gruppe ihre Gewichte-Bytes holt: 'ring' (Standard, aus dem Host-Ring) oder 'exchange' (Karte zu Karte).",
        "gain": "exchange spart Host-Speicher.", "cost": "exchange braucht VRAM-Spitze, solange beide Gruppen auf einer Karte liegen.",
        "depends": [_d("--weg2-xchg-census", "braucht", "exchange verlangt die gemessene Census-Datei, sonst verweigert der Start (W71).", "text")]},
    "--weg2-xchg-census": {
        "kind": "flag", "group": "Speicher", "level": "experte", "planner_derived": False,
        "text": "JSON-Datei mit der gemessenen Tabelle je Karte/Tag/Gruppe, aus der W71 die VRAM-Spitze des Austauschs preist.",
        "gain": "Macht den Kartenaustausch preisbar.", "cost": "Muss für genau dieses Modell gemessen sein.",
        "depends": [_d("--weg2-weight-source", "braucht", "Nur mit exchange verlangt.", "text")]},
    "--weg2-xchg-census-map": {
        "kind": "flag", "group": "Speicher", "level": "experte", "planner_derived": False,
        "text": "Benennt die Census-Zeile, aus der eine Karte gepreist wird, die nicht per UUID in der Census-Datei steht: Liste <live>=<Census-UUID>, <live> = Karten-UUID, nvml<N> oder Klasse (RTX3080). Ohne Eintrag leiht der Start die schwerste Zeile der gleichen Klasse (sonst der ganzen Census) und verweigert das als HW-BORROWED, bis Force gesetzt ist.",
        "gain": "Fremde Karten, andere Kartenzahl oder anderes Rig starten mit dem Austausch, ohne dass die Census-Datei neu gemessen wird.",
        "cost": "Die Bytes sind die einer anderen Karte; der Peak wird weiter gegen die echte VRAM-Summe der Karte geprueft.",
        "depends": [_d("--weg2-xchg-census", "braucht", "Benennt eine Zeile dieser Census-Datei.", "text")]},
    "--weg2-vision": {
        "kind": "flag", "group": "Speicher", "level": "einfach", "planner_derived": False,
        "text": "'off' (Standard) startet beide Gruppen ohne Vision-Tower; 'resident' hält ihn geladen.",
        "gain": "off spart je Prefill-Karte VRAM und Host-Ring.", "cost": "Ohne Tower keine Bildeingabe. off ändert den Form-Schlüssel der P-Gruppe (erster Start rechnet seinen Ring neu).",
        "depends": []},
    "--flip-weights": {
        "kind": "flag", "group": "Speicher", "level": "experte", "planner_derived": False,
        "text": "'family' (Standard) verschiebt die Gewichte beim Wechsel P/D; 'resident' hält beide Gruppen im VRAM und wechselt nur den KV-Cache.",
        "gain": "resident: kein Gewichtstransport im Flip.", "cost": "resident braucht VRAM für beide Gewichtssätze.",
        "depends": [_d("--rank-gpu-memory-mib", "tauscht", "resident: zweiter Gewichtssatz gegen KV.", "S4")]},
    # ------------------------------------------------------------------ Decode / Spekulation
    "--spec-form": {
        "kind": "flag", "group": "Decode", "level": "einfach", "planner_derived": False,
        "text": "Die spekulative Dekodierung beider Gruppen: NEXTN (Standard, mtp-Kopf des Checkpoints) oder DFLASH (externer Draft).",
        "gain": "Spekulation erhöht die Token je Runde, wenn der Draft gut trifft.", "cost": "Der Draft belegt VRAM auf jeder Karte der Gruppe; ohne ihn bleibt mehr KV.",
        "depends": [_d("--draft-kv-on-p", "skaliert_mit", "Ob P die Draft-KV erzeugt.", "text"),
                    _d("--rank-gpu-memory-mib", "tauscht", "Draft-Gewichte gegen KV.", "S4")]},
    "--draft-kv-on-p": {
        "kind": "flag", "group": "Decode", "level": "experte", "planner_derived": False,
        "text": "Ob Gruppe P auch die Draft-KV erzeugt (on/off).", "gain": "D muss die Draft-KV nicht selbst aufbauen.", "cost": "Mehr Arbeit und Speicher auf P.",
        "depends": [_d("--spec-form", "braucht", "Nur mit Draft sinnvoll.", "text")]},
    "--d-tp-objective": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": True,
        "text": "Wofür der Gewichtsvektor der Decode-Gruppe gelöst wird: der Betriebspunkt bs6 (Standard) oder 'maxkv' (größter KV-Pool zuerst).",
        "gain": "bs6: schnellere Runde im Betrieb.", "cost": "maxkv gibt Leistung für Kontext ab.",
        "depends": [_d("--rank-tp-ratio", "abgeleitet_von", "Der Planer löst den Vektor für dieses Ziel.", "text")]},
    "--d-reshard": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False,
        "text": "Die Gewichtsaufteilung der Decode-Gruppe pro Ladeklasse (off/wake/wake-seg/live).",
        "gain": "wake wählt die MLP-Verteilung bei jedem P→D-Wechsel.", "cost": "Mehr als ein Preset verweigert den Boot, solange der Ausführer nicht verdrahtet ist.",
        "depends": [_d("--rank-mlp-ratio", "abgeleitet_von", "Mit einem Preset gleich der statischen --rank-mlp-ratio.", "text")]},
    "--d-kv-token-cut": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": True,
        "text": "Schneidet den Attention-KV der Decode-Gruppe nach Token über die Karten (off/maxmin/Vektor/joint/owned).",
        "gain": "Der KV-Pool wächst, weil jede Karte nach ihrem Platz Token trägt.", "cost": "Ein Flip-Boot braucht dann die Host-Stufe und genau eine D-KV-Stufe.",
        "depends": [_d("--rank-moe-ratio", "skaliert_mit", "owned löst Anteile, FR_D und Experten-Besitz zusammen.", "text")]},
    "--idle-layout": {
        "kind": "flag", "group": "Decode", "level": "einfach", "planner_derived": False,
        "text": "Welche Gruppe im Ruhezustand wach ist: tp (Decode-Gruppe D) oder pp (Prefill-Gruppe P).",
        "gain": "Ruhe auf P bedient die erste Anfrage ohne Flip.", "cost": "Das andere Layout muss beim Wechsel aufwachen.",
        "depends": [_d("--d-hold-s", "skaliert_mit", "Haltezeit von D vor dem Flip im Ruhezustand pp.", "text")]},
    "--d-hold-s": {
        "kind": "flag", "group": "Decode", "level": "experte", "planner_derived": False,
        "text": "D bleibt so viele Sekunden nach seiner Arbeit wach, bevor es flippt (0/leer = aus).",
        "gain": "Kurze Anfragen im Fenster laufen ohne Flip.", "cost": "D belegt Karten und blockiert P währenddessen.",
        "depends": [_d("--idle-layout", "skaliert_mit", "Wirkt im Ruhezustand pp.", "text")]},
    "--d-short-drain-tokens": {
        "kind": "flag", "group": "Decode", "level": "experte", "planner_derived": False,
        "text": "Solange D wach ist, wird ein Rückstau nur aus kurzen Anfragen (je höchstens X) auf D bedient, wenn ihre Tokens N nicht übersteigen. 0 = aus.",
        "gain": "Kein Flip für kleine Rückstaus.", "cost": "N größer als X verweigert (W153).",
        "depends": [_d("--tp-prefill-max-tokens", "skaliert_mit", "N darf X nicht übersteigen.", "text")]},
    # ------------------------------------------------------------------ Start / Identität
    "--profile": {
        "kind": "flag", "group": "Start", "level": "einfach", "planner_derived": False,
        "text": "Das Registry-Profil der Checkpoints beider Gruppen (qwen27b, nextflash).",
        "gain": "Zieht Architektur, Schalter-Defaults und Records des Modells.", "cost": "Ein anderes Modell braucht eine Registry-Zeile.",
        "depends": [_d("PROFILE_LINE", "skaliert_mit", "Die Linie 27b/nf des Profils.", "text")]},
    "--model": {"kind": "flag", "group": "Start", "level": "einfach", "planner_derived": False,
                "text": "Pfad des Modell-Checkpoints (oder der GGUF-Datei).", "gain": "", "cost": "Der Entrypoint prüft, dass der Pfad existiert.",
                "depends": [_d("PROFILE_MODEL", "abgeleitet_von", "Dasselbe als Profil-Variable.", "text")]},
    "--p-hostgap": {"kind": "flag", "group": "Fehlersuche", "level": "experte", "planner_derived": False,
                    "text": "Instrument: eine Zeile je Forward mit dem Leerlauf der Karte davor (P).", "gain": "Messdaten für P-Host-Overlap.", "cost": "Kleiner Mehraufwand; Instrument, ändert keine Entscheidung.", "depends": []},
    "--extra-p": {"kind": "flag", "group": "Start", "level": "experte", "planner_derived": False,
                  "text": "Weitere Server-Flags für Gruppe P, als eine Zeichenkette (Shell-Trennung).", "gain": "", "cost": "", "depends": []},
    "--extra-d": {"kind": "flag", "group": "Start", "level": "experte", "planner_derived": False,
                  "text": "Weitere Server-Flags für Gruppe D, als eine Zeichenkette (Shell-Trennung).", "gain": "", "cost": "", "depends": []},
    "--env-p": {"kind": "flag", "group": "Start", "level": "experte", "planner_derived": False,
                "text": "Umgebungsvariablen nur für Gruppe P: KEY=VAL;KEY=VAL (Werte dürfen Kommas tragen, wirken zuletzt).", "gain": "", "cost": "", "depends": []},
    "--env-d": {"kind": "flag", "group": "Start", "level": "experte", "planner_derived": False,
                "text": "Umgebungsvariablen nur für Gruppe D: KEY=VAL;KEY=VAL (Werte dürfen Kommas tragen, wirken zuletzt).", "gain": "", "cost": "", "depends": []},
    # ------------------------------------------------------------------ Umgebungsvariablen (Auswahl)
    "SGLANG_MOE_RESIDENT_EXPERT_FRACTION": {
        "kind": "env", "group": "Speicher", "level": "experte", "planner_derived": True,
        "text": "Anteil der Experten je Layer, der im VRAM bleibt (1.0 = kein Offload); darunter holt ein Pinned-Host-Pool mit LRU die kalten Experten.",
        "gain": "Weniger als 1.0 spart VRAM für KV.", "cost": "Kalte Experten werden per PCIe geholt (Decode langsamer).",
        "depends": [_d("--rank-moe-resident-fraction", "schliesst_aus", "Dieselbe Größe als Flag; bei Widerspruch verweigert der Start.", "text"),
                    _d("--rank-gpu-memory-mib", "tauscht", "Residente Experten gegen KV.", "S4")]},
    "SGLANG_MOE_SCRATCH_SLOTS": {
        "kind": "env", "group": "Speicher", "level": "experte", "planner_derived": True,
        "text": "LRU- und Staging-Zeilen des Experten-Pools je Layer (Liste je Karte).",
        "gain": "Hält gefragte Experten bereit.", "cost": "VRAM je Zeile und Layer.",
        "depends": [_d("--pp-cut-expert-lru-rows", "abgeleitet_von", "Die P-Gruppe liest es aus diesem Flag.", "text")]},
    "SGLANG_UNEVEN_MOE_EXPERT_SHARD": {
        "kind": "env", "group": "Aufteilung", "level": "experte", "planner_derived": False,
        "text": "Verteilt MoE-Experten nach Index (ganze Experten je Karte) unter einem ungleichen Plan.",
        "gain": "Nötig, damit --rank-moe-ratio Experten ungleich verteilen kann.", "cost": "Das nextflash-Profil schaltet es je Gruppe an; global bleibt es aus.",
        "depends": [_d("--rank-moe-ratio", "braucht", "Die Verteilung nach Index.", "text")]},
    "SGLANG_WEG2_OWNED_BASE": {
        "kind": "env", "group": "Aufteilung", "level": "experte", "planner_derived": False,
        "text": "Basis der Experten-Besitz-Lösung: 'derive' (Planer leitet ab) oder 'stated' (der Vektor des Profils ist die Basis).",
        "gain": "stated hält die gemessene Hand-Basis.", "cost": "derive braucht Hitze-Records, sonst keine Handbasis.",
        "depends": [_d("--rank-moe-ratio", "abgeleitet_von", "Bei stated ist der Profil-Vektor die Basis, bei derive nur der Startwert.", "text")]},
    "SGLANG_WEG2_D_SEAT_REWAKE": {
        "kind": "env", "group": "Decode", "level": "experte", "planner_derived": False,
        "text": "Die Sitzzahl von D wird live an einer Rundengrenze angepasst (wächst bei wartenden Anfragen, schrumpft bei freien Sitzen).",
        "gain": "Sitze folgen der Last ohne Gewichtstransport.", "cost": "Experten-Zeilen der neuen Sitze gehen ab.",
        "depends": [_d("--d-bs", "skaliert_mit", "Obere Grenze der Sitze.", "text")]},
    "SGLANG_WEG2_STORE_MLOCK": {
        "kind": "env", "group": "Host", "level": "experte", "planner_derived": False,
        "text": "Sperrt jeden Experten-Store vor der Registrierung im RAM (mlock).",
        "gain": "Vermeidet Stalls beim Registrieren hinter der Host-Kompaktierung.", "cost": "Wirksam nur mit dem Host-Sysctl vm.compact_unevictable_allowed=0.",
        "depends": []},
    "SGLANG_WEG2_TAG_STALL_SENTINEL_S": {
        "kind": "env", "group": "Fehlersuche", "level": "experte", "planner_derived": False,
        "text": "Stack-Dumps, wenn ein Schlaf-Tag länger als diese Sekunden dauert (0 = aus).",
        "gain": "Diagnose von Schlaf-Stalls.", "cost": "Dateien im Evidence-Verzeichnis; Diagnose-Instrument.", "depends": []},
    # ------------------------------------------------------------------ Profil-Variablen (Entrypoint)
    "PROFILE_NAME": {"kind": "var", "group": "Profil", "level": "einfach", "planner_derived": False,
                     "text": "Der Name des Profils.", "gain": "", "cost": "", "depends": []},
    "PROFILE_LINE": {"kind": "var", "group": "Profil", "level": "einfach", "planner_derived": False,
                     "text": "Die Linie des Profils (27b oder nf). Ein Image trägt genau eine Linie; eine abweichende Profil-Linie verweigert der Entrypoint.",
                     "gain": "", "cost": "", "depends": []},
    "PROFILE_STATUS": {"kind": "var", "group": "Profil", "level": "einfach", "planner_derived": False,
                       "text": "Wie weit das Profil bewiesen ist: abgenommen (läuft), experimentell/formnachweis (nur mit Freigabe), vorbereitet/geplant (verweigert).",
                       "gain": "", "cost": "", "depends": [_d("PROFIL-STATUS", "braucht", "Ablehnung 'Profil ist nicht abgenommen'; mit Force übergehbar.", "text")]},
    "PROFILE_FORMAT": {"kind": "var", "group": "Profil", "level": "einfach", "planner_derived": False,
                       "text": "Das Gewichtsformat, das das Profil erwartet; der Entrypoint vergleicht es mit dem erkannten Format des Checkpoints.",
                       "gain": "", "cost": "", "depends": []},
    "PROFILE_MODEL": {"kind": "var", "group": "Profil", "level": "einfach", "planner_derived": False,
                      "text": "Pfad des Modells; der Entrypoint verlangt, dass er existiert.", "gain": "", "cost": "", "depends": []},
    "PROFILE_DRAFT": {"kind": "var", "group": "Profil", "level": "einfach", "planner_derived": False,
                      "text": "Pfad des Draft-Modells (leer = keiner); der Entrypoint verlangt, dass er existiert.", "gain": "", "cost": "",
                      "depends": [_d("--spec-form", "braucht", "Der Draft wird nur mit Spekulation genutzt.", "text")]},
    "PROFILE_CARD_COUNT": {"kind": "var", "group": "Hardware", "level": "einfach", "planner_derived": False,
                           "text": "Wie viele Karten das Profil erwartet (Topologie P = PPn, D = TPn).", "gain": "", "cost": "Abweichende Kartenzahl: HW-COUNT.",
                           "depends": [_d("PROFILE_INVENTORY", "skaliert_mit", "Das Inventar hat so viele Einträge.", "text"),
                                       _d("--pp-stage-ratio", "skaliert_mit", "Jeder Positions-Vektor hat so viele Einträge.", "text")]},
    "PROFILE_INVENTORY": {"kind": "var", "group": "Hardware", "level": "einfach", "planner_derived": False,
                          "text": "Das Karteninventar in Kalibrier-Klassen und Kartenreihenfolge (z. B. RTX5090,RTX3080,RTX3080), auf dem die Positions-Messwerte des Profils gemessen wurden.",
                          "gain": "", "cost": "Ein anderes echtes Inventar: HW-UNCALIBRATED (die Zahlen sind nicht für diese Karten gemessen).",
                          "depends": [_d("PROFILE_CARD_COUNT", "skaliert_mit", "Zahl der Einträge.", "text")]},
    "PROFILE_SHM_MIN_GIB": {"kind": "var", "group": "Host", "level": "einfach", "planner_derived": False,
                            "text": "Mindestgröße von /dev/shm in GiB; der Entrypoint prüft sie (docker run --shm-size).", "gain": "", "cost": "Zu klein: SHM-Ablehnung.", "depends": []},
    "PROFILE_MEMAVAIL_MIN_GIB": {"kind": "var", "group": "Host", "level": "einfach", "planner_derived": False,
                                 "text": "Mindestens so viel MemAvailable (GiB) muss der Host vor dem Start frei haben.", "gain": "", "cost": "Zu wenig: MEMAVAIL-Ablehnung.", "depends": []},
    "PROFILE_TMPFS_STORE": {"kind": "var", "group": "Host", "level": "experte", "planner_derived": False,
                            "text": "Pfad des tmpfs-Mounts für den Experten-Store.", "gain": "", "cost": "Kein tmpfs: STORE-Ablehnung (cudaHostRegister auf ZFS-mmap scheitert).",
                            "depends": [_d("PROFILE_TMPFS_GIB", "skaliert_mit", "Die Größe des Mounts.", "text")]},
    "PROFILE_TMPFS_GIB": {"kind": "var", "group": "Host", "level": "experte", "planner_derived": False,
                          "text": "Mindestgröße (GiB) des tmpfs-Mounts für den Experten-Store.", "gain": "", "cost": "Zu klein: STORE-Ablehnung.",
                          "depends": [_d("PROFILE_TMPFS_STORE", "braucht", "Gilt für diesen Mount.", "text")]},
    "PROFILE_REQUIRED_PATHS": {"kind": "var", "group": "Profil", "level": "experte", "planner_derived": False,
                               "text": "Weitere Pfade, die der Entrypoint zusätzlich zu Modell und Draft prüft.", "gain": "", "cost": "Fehlt einer: Ablehnung (Modell fehlt).", "depends": []},
    "PROFILE_SERVE_PORT": {"kind": "var", "group": "Start", "level": "einfach", "planner_derived": False,
                           "text": "Der Port, auf dem der Server antwortet (Front 30030, D-only 30032).", "gain": "", "cost": "", "depends": []},
    "PROFILE_D_ONLY": {"kind": "var", "group": "Start", "level": "einfach", "planner_derived": False,
                       "text": "1 = nur Gruppe D auf allen Karten, keine Front (der Entrypoint hängt --d-only an).", "gain": "", "cost": "Kein Prefill/Decode-Wechsel.",
                       "depends": [_d("PROFILE_SERVE_PORT", "skaliert_mit", "Clients sprechen D direkt an.", "text")]},
}
