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
        # AP-G 06.10.: Text aus dem argparse-help= (#1264) und der Launcher-Zeile draft_on_p_line; die Kanten (Draft-Platzierung, Host-Budget) stehen im Kantenkatalog.
        "text": "Ob Gruppe P der reine Draft-KV-ERZEUGER ist (#1264). 'on' (Standard; stehender Nutzerauftrag vom 2026-09-07: Draft-KV über den Flip hinweg) legt den mtp.*-Kopf des Checkpoints auf Ps letzte Stufe, damit D nach einem Flip mit warmer Draft-KV wieder aufnimmt. 'off' bootet die rg6-bewiesene Form: P trägt KEIN Speculative-Flag und keinen MTP-Kopf (der Launcher sagt es in einer Zeile; die Serving-Basis / der A-B-Arm, nie still). Gruppe D bleibt bei beiden Werten unverändert: sie behält ihren eigenen NEXTN-Kopf, denn 'off' nimmt den Erzeuger weg, nicht die spekulative Dekodierung.",
        "gain": "'on': D nimmt nach einem Flip mit warmer Draft-KV wieder auf.",
        "cost": "'on': der MTP-Kopf liegt auf Ps letzter Stufe. 'off': P bootet ohne Draft; D parkt seine Draft im pinned System-RAM, solange P läuft (WEG2-DRAFT-PARK/UNPARK), und startet jede geflippte Anfrage DRAFT-COLD.",
        "satz_quelle": "launcher.py --draft-kv-on-p (help=) und launcher.draft_on_p_line",
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
    # ------------------------------------------------------------------ AP-G (Planer-Workflow 06.10.): Form A, ungleiches DCP, Draft, Betriebsform, Dual
    # Jeder Satz unten ist die Uebersetzung des argparse-``help=`` bzw. des environ.py-Kommentars (Quelle je Eintrag im ``quelle``-Feld);
    # was die Quelle nicht sagt, steht NICHT hier (gain/cost bleiben leer). Die Kanten dieser Werte stehen NUR im Kantenkatalog
    # (kantenkatalog_1004.json, K62ff) mit Beleg; ``depends`` bleibt hier leer, damit keine unbelegte kuratierte Kante entsteht.
    "--rank-role": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Form A (Attention-Host-Layout): je Rang eine Rolle, 'host' oder 'worker' (z. B. host,worker,worker für eine 5090 und zwei 3080). Genau ein Rang darf Host sein. Der Host rechnet alle dichten Teile des Modells, hält den ganzen KV-Cache für den vollen Kontext, die GDN-Zustände, den spekulativen Draft und die CUDA-Graphen sowie seinen Anteil an den MoE-Experten; ein Worker hält nur seine eigenen Experten und sonst nichts: er rechnet sie für die Zeilen, die der Host verteilt, und schickt die Teilsummen zurück. Erst dieses Flag macht eine Null in --rank-tp-ratio als Layout lesbar: eine Null ist nur für einen Rang erlaubt, den dieses Flag Worker nennt, und jeder Worker braucht eine.",
        "gain": "", "cost": "Braucht ein ausdrückliches --rank-tp-ratio; nur reine Tensor-Parallelität auf einem Knoten.",
        "satz_quelle": "server_args.py rank_role (Arg help=)", "depends": []},
    "--rank-kv-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Wem die KV-Token gehören (ungleiches DCP), entkoppelt von der Gewichtsaufteilung: Wo ein Kontext-Token liegt, verschiebt nur, wo seine Attention-Rechnung läuft. 'coupled' (Standard): das bisherige Verhalten. 'capacity' (Alias 'auto'): Besitz proportional zur tatsächlich freien Token-Kapazität jedes Rangs nach dem Laden der Gewichte. 'speed': vom Kapazitätsanteil in Richtung des Speicherbandbreiten-Anteils verschoben, soweit --rank-perf-loose-ctx-percent es erlaubt. 'corridor' (#602): 'capacity' plus eine harte Freiraum-Untergrenze je Karte. Eine Liste positiver ganzer Zahlen (ein Eintrag je Rang) legt den Besitzvektor fest. Werte ungleich 'coupled' schalten den gewichteten-DCP-Weg ein. Die Env SGLANG_UNEVEN_TOKEN_VECTOR (expliziter Vektor) hat Vorrang vor diesem Flag; die Gewichtsaufteilung (--rank-tp-ratio, --rank-mlp-ratio, ...) bleibt unberührt.",
        "gain": "'capacity' maximiert max_total_num_tokens (Konvergenz in einem Boot). Gemessen (#210, 27B FP8 TP=3, ungleiches DCP, 120k Token im Speicher, bs=1, ohne Spec): Vektor [2,3,3] auf [2,1,1] senkte den kontextabhängigen Teil des Decode-Schritts um 24,5 % (2,296 auf 1,732 ms; Rauschen von Boot zu Boot 1,07 %), das sind -2,5 % Schrittzeit insgesamt, mit dem Kontext linear wachsend.",
        "cost": "'capacity' verlagert bei tiefem Kontext Attention-Arbeit auf die Karten, die mehr Token halten. 'speed' braucht die Bandbreitenwerte je Rang (--rank-tp-ratio auto-performance), sonst fällt es auf 'capacity' zurück und sagt es. Werte ungleich 'coupled' verlangen --rank-gpu-id mit einem ungleichen --rank-tp-ratio-Plan. 'corridor': eine Karte, die Reserve plus noch nicht angefallenen Bedarf nicht tragen kann, bricht den Boot mit den Zahlen je Karte ab.",
        "satz_quelle": "server_args.py rank_kv_ratio (Arg help=)", "depends": []},
    "--dcp-size": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Die Größe der Decode-Context-Parallelität (Standard 1; Alias --decode-context-parallel-size).",
        "gain": "", "cost": "", "satz_quelle": "server_args.py dcp_size (Arg help=)", "depends": []},
    "--uneven-dcp": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Schaltet ungleiches DCP ein. Aus der Env SGLANG_UNEVEN_DCP (#781) zum Flag befördert; in den meisten Fällen durch --rank-kv-ratio abgelöst.",
        "gain": "", "cost": "", "satz_quelle": "server_args.py uneven_dcp (Arg help=)", "depends": []},
    "--uneven-dcp-weighted": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Schaltet gewichtetes ungleiches DCP ein. Aus der Env SGLANG_UNEVEN_DCP_WEIGHTED (#781) zum Flag befördert.",
        "gain": "", "cost": "", "satz_quelle": "server_args.py uneven_dcp_weighted (Arg help=)", "depends": []},
    "SGLANG_UNEVEN_TOKEN_VECTOR": {
        "kind": "env", "group": "Aufteilung", "level": "experte", "planner_derived": False,
        "text": "Aufteilungsvektor der Token-Achse bei ungleichem DCP (\"a,b,c\", je DCP-Rang eine positive ganze Zahl). Er überstimmt den aus dem Budget geschätzten Vektor, den resolve_cp_token_ratios sonst ableiten würde. Die KV-Pool-Selbstkalibrierung gibt ihn als Neustart-Hinweis aus (gemessenes Optimum aus der tatsächlich profilierten Token-Kapazität je Rang). Modelltyp-unabhängig: er richtet sich nach der gemessenen Kapazität, die dtype-unabhängig ist.",
        "gain": "Beim nächsten Boot zurückgespeist, konvergieren die KV-Pools je Rang auf das profilierte Optimum.", "cost": "",
        "satz_quelle": "environ.py SGLANG_UNEVEN_TOKEN_VECTOR (Kommentar)", "depends": []},
    "--rank-vocab-ratio": {
        "kind": "flag", "group": "Aufteilung", "level": "experte", "planner_derived": False, "scope": "server",
        "text": "Gewichtete Aufteilung der gekoppelten Vokabular-Schichten (VocabParallelEmbedding / ParallelLMHead, auch von NEXTN/EAGLE-Drafts geteilt): 'auto' oder je Rang eine positive ganze Zahl (Länge gleich --tp-size). Der lm_head-Matvec liest bei jedem Decode-Schritt den ganzen Gewichtsanteil; bei ungleichen Karten wird die gleichmäßige Aufteilung (Standard, bleibt auch unter --rank-tp-ratio gleichmäßig) von der langsamsten Karte begrenzt. 'auto' leitet die Gewichte aus den Speicherbandbreiten-Werten des gecachten auto-performance-Hardwareprofils ab, sonst aus dem aufgelösten --rank-tp-ratio-Vektor. Die Env SGLANG_UNEVEN_VOCAB_VECTOR (expliziter Vektor) hat Vorrang vor diesem Flag.",
        "gain": "Gewichten nach Speicherbandbreite gleicht die Lesezeit des lm_head aus statt der Anteilsbreite.",
        "cost": "Verlangt einen aktiven --rank-tp-ratio-Plan. Standard AUS: ohne das Flag bleibt die Vokabular-Aufteilung gleichmäßig und das Verhalten unverändert.",
        "satz_quelle": "server_args.py rank_vocab_ratio (Arg help=)", "depends": []},
    "--speculative-draft-placement": {
        "kind": "flag", "group": "Decode", "level": "einfach", "planner_derived": False, "scope": "server",
        "text": "Wo das spekulative Draft-Modell läuft. 'split' (Standard): der Draft ist wie bisher über alle TP-Ränge tensor-parallel geteilt (byte-identisch, ob das Flag fehlt oder 'split' heißt). 'solo': der Draft läuft UNGETEILT auf EINEM festgelegten Rang (siehe --speculative-draft-gpu) und sendet seine k Draft-Token-IDs einmal je Runde an die übrigen Ränge; diese bauen das Draft-Modell auf dem meta-Gerät (keine Draft-Gewichte, kein Draft-KV-Pool, keine Draft-CUDA-Graphen) und überspringen den Draft-Forward.",
        "gain": "Auf Rigs ohne P2P ersetzt 'solo' die k host-gestagten Draft-All-Reduces je Runde durch einen kleinen Broadcast.",
        "cost": "'solo' (v1): nur Familie EAGLE/EAGLE3/NEXTN, topk == 1, kein Rejection Sampling, reine Single-Node-TP.",
        "satz_quelle": "server_args.py speculative_draft_placement (Arg help=)", "depends": []},
    "--d-only": {
        "kind": "flag", "group": "Start", "level": "einfach", "planner_derived": False,
        "text": "Nur Gruppe D (TP3 auf allen Karten), kein P, kein Flip, keine Front (Nutzer 25.09.). D wird mit derselben Env und demselben argv gebaut wie im Flip-Boot (Budgets aus der Erwartung wie im Dry-Run), nimmt aber jede ungecachte Länge selbst an (--tp-prefill-max-tokens = --max-kv-per-request) und wird direkt auf dem Port angesprochen.",
        "gain": "Für Hosts, deren RAM den Flip nicht trägt.", "cost": "Kein P, kein Flip, keine Front.",
        "satz_quelle": "launcher.py --d-only (help=)", "depends": []},
    "--profile-inventory": {
        "kind": "flag", "group": "Hardware", "level": "experte", "planner_derived": False,
        "text": "HW-GENERIC 1002: das Karteninventar (Kalibrierklassen-Labels in Kartenreihenfolge, weg2/card_identity.py, z. B. 'RTX5090,RTX3080,RTX3080'), auf dem die positionalen Vektoren des Profils gemessen wurden. Standard: das Inventar, das die gemessenen Records des Profils angeben (profile_records_data/<Profil>.json, Feld 'inventory'). Ein abweichendes Live-Inventar wird beim Namen verweigert (HW-UNCALIBRATED); positionale Messungen anderer Karten werden nie geliehen.",
        "gain": "", "cost": "Abweichendes Inventar: HW-UNCALIBRATED.",
        "satz_quelle": "launcher.py --profile-inventory (help=)", "depends": []},
    # ---- Dual (launcher.py resolve_dual_layout, argparse :22699-22798)
    "--dual-layout": {
        "kind": "flag", "group": "Dual", "level": "einfach", "planner_derived": False,
        "text": "DUAL-TP3PP3 (F26, Nutzer 29.09.): BEIDE Gruppen bleiben den ganzen Boot wach, die Front flippt nie. Impliziert --flip-weights resident; P wird nach READY nicht schlafen gelegt, statt dessen wird sein WACH-Fußabdruck gemessen und D danach bemessen (dieselbe PID-Messung wie beim Schlafrest). Das Budget von P selbst senkt man mit --extra-p '--rank-gpu-memory-mib ...' (W100 erlaubt das Senken). Verweigert --weg2-d-adopt on und --idle-layout pp (help=; der Code stellt pp ohne Verweigerung auf tp um, launcher.py:14722). Standard aus: der Launcher ist byte-identisch.",
        "gain": "", "cost": "Kein Flip; verweigert --weg2-d-adopt on; ein gesetztes --idle-layout pp wird auf tp umgestellt (kein Verweigern).",
        "satz_quelle": "launcher.py --dual-layout (help=)", "depends": []},
    "--dual-share": {
        "kind": "flag", "group": "Dual", "level": "einfach", "planner_derived": False,
        "text": "DUAL-TP3PP3 Stufe 1b (impliziert --dual-layout): Ps Stufe rechnet auf Ds TP-Shards (Hüllen über Ds drei Shards ihrer Layer) und bindet den Shard des D-Rangs auf ihrer Karte über das Union-Image an Ds Bytes. D bootet direkt nach P als Union-OWNER, bemessen aus Ps GEPLANTEM Budget (+ --dual-p-overhead-mib); P wartet auf Ds Image, bevor es etwas lädt.",
        "gain": "", "cost": "",
        "satz_quelle": "launcher.py --dual-share (help=)", "depends": []},
    "--dual-p-overhead-mib": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3 --dual-share (Standard 1500): was P auf einer Karte außerhalb seines --rank-gpu-memory-mib-Budgets hält (CUDA-Kontext, Graphen, Aktivierungen); wird angerechnet, wenn D aus Ps Plan statt aus Ps Messung bemessen wird.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-p-overhead-mib (help=)", "depends": []},
    "--dual-d-prefill-tokens": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3 D-Prefill-Zulassung, Standard 0 (Nutzer 01.10.: im Dual-Layout läuft JEDER Prefill auf P, D decodet nur). Ds X (sein W31-Riegel und die Routing-Grenze der Front) wird zu 1 + diesem Wert: die 1 ist die N-1-Anker-Konvention (P hält das letzte Prompt-Token zurück, Ds erster Schritt rechnet es); alles Größere geht an P, und ein größerer Rest an D wird beim Namen verweigert (W31 -> P), nie still neu gerechnet. Ohne --dual-layout ignoriert; ersetzt dort --tp-prefill-max-tokens / --x-ceiling-tokens / --d-short-drain-tokens.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-d-prefill-tokens (help=)", "depends": []},
    "--dual-p-duty": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3 Pausenzulassung (Standard 1.0 = keine Pause): P und D laufen gleichzeitig und teilen jede Karte, nur die Physik begrenzt sie (Nutzer 01.10.). Unter 1.0 DARF P pausieren, während D Decodes hält: der Anteil der Wanduhrzeit, in dem Ps erste Stufe rechnen darf (weg2/dual_duty.py). Einfache Form: PP0 wartet nach jedem Forward t_fwd*(1-duty)/duty. Mit der Env SGLANG_WEG2_DUAL_P_GANG_CHUNKS=K (Gang-Fenster): PP0 rechnet K Chunks, wartet, bis die Pipeline leer ist, und hält dann, damit D allein auf allen Karten ist.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-p-duty (help=)", "depends": []},
    "--dual-p-sm-pct": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3 mit --dual-mps on (Standard 100): CUDA_MPS_ACTIVE_THREAD_PERCENTAGE für Gruppe P, also der Anteil der SMs, den Ps Kernels belegen dürfen, während D decodet. 100 = keine Grenze.",
        "gain": "", "cost": "Gemessen am 29.09. (risk-1 bench, 5090): unbegrenztes P nimmt ~90 % der Karte und Ds Schritt läuft ~7x langsamer; 50 teilt ~50/50; die Summe beider Anteile bleibt in beiden Fällen ~1,0.",
        "satz_quelle": "launcher.py --dual-p-sm-pct (help=)", "depends": []},
    "--dual-priority": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE Stufe 2: der P/D-Anteilsregler der Front schreibt eine Stufe (Ps Anteil 1.0/0.75/0.5/0.25, Profil-Env SGLANG_WEG2_DUAL_SHARE_RUNGS) nach <busy>.ctl; P wirkt über --dual-share-actuators. p = P voll (heute), balanced/d = eine feste Stufe, solange D decodet, dynamic = Matrix aus tau (wartende Prefill-Token / Ps volle Rate) x D-bs. Zur Laufzeit umschaltbar: POST /weg2/dual-priority an der Front. Ungesetzt = aus.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-priority (help=)", "depends": []},
    "--dual-d-min-rate-tps": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE: Ds Mindest-Decode-Rate je Anfrage (Token/s, gemessen am Leg-2-Token-Strom); darunter geht die Stufe in Richtung D. 0 = aus.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-d-min-rate-tps (help=)", "depends": []},
    "--dual-p-min-share": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE: Untergrenze von Ps Anteil (Stufen darunter werden nie benutzt; Standard 0.25).",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-p-min-share (help=)", "depends": []},
    "--dual-share-actuators": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE: Komma-Liste der P-Stellglieder (Standard 'chunk'): chunk (PP0s Chunk-Obergrenze), duty (P-Duty-Drossel beim Anteil der Stufe), green (Stufe 3, NICHT gebaut: benannter Rückfall auf chunk).",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-share-actuators (help=)", "depends": []},
    "--dual-green-ladder": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE Stufe 3 (weg2/dual_green.py): Ps SM-Anteil als Green-Context-LEITER (100/75/50/25 % je Forward, dynamisch, hoch UND runter; PP0 stempelt die Stufe auf den Request-Draht, damit alle drei Stufen dieselbe fahren). 'on' = Leiter + Halte-BEOBACHTER (die P-STUFE-Zeile trägt would_hold), 'hold' = zusätzlich hält PP0 wirklich (0 %), solange die Arena voll ist. Standard off = argv/env/Startpfad byte-identisch.",
        "gain": "", "cost": "Braucht --dual-priority, 'green' in --dual-share-actuators, --dual-mps on und kein --dual-p-sm-pct.",
        "satz_quelle": "launcher.py --dual-green-ladder (help=)", "depends": []},
    "SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE": {
        "kind": "env", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "Eintrittsstufe von Ps SM-Anteil, wenn D nach Leerlauf wieder decodet und keine Mindestrate (--dual-d-min-rate-tps) gesetzt ist: Zeilen \"bs-Schwelle:Stufe tau niedrig:Stufe tau hoch\", getrennt durch Semikolon (Beispiel im Code: \"2:1:0;4:2:1;99:3:2\"). Es gilt die erste Zeile, deren Schwelle D-bs erreicht oder übersteigt; tau ist die wartende P-Arbeit in Sekunden (hoch = über der oberen tau-Kante, Standard 2 und 10 s). Stufe 0/1/2/3 = Ps Anteil 100/75/50/25 % (Standard der Stufen SGLANG_WEG2_DUAL_SHARE_RUNGS = 1,0.75,0.5,0.25). Standard im Code: 2:1:0;4:2:1;1000000000:3:2. Ist D leer (bs 0), steht P sofort auf 100 %. Danach regelt die Front auf die gemessene D-Rundenzeit nach; die Aushungerungs-Klemme gilt zusätzlich. Nur im Dual-Layout (27B-Linie).",
        "gain": "", "cost": "", "satz_quelle": "dual_green.py GreenConfig.table (Zeile 1071), from_env (Zeile 1097), start_stage (Zeilen 1204-1224); dual_share.py ShareConfig (Zeilen 139-141)", "depends": []},
    "SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S": {
        "kind": "env", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "Aushungerungs-Klemme, Teil 1 (Dual-Regler der Front): wartet die älteste P-Anfrage länger als diese Sekunden, wird Ps Stufe auf höchstens SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG begrenzt, egal was Matrix oder Tabelle sagen. Standard 60. Im Modus p (P voll) ist die Klemme aus.",
        "gain": "", "cost": "", "satz_quelle": "dual_share.py ShareConfig.starve_age_s (Zeile 157) und Klemme (Zeilen 414-417); dual_green.py (Zeilen 1319-1322)", "depends": []},
    "SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG": {
        "kind": "env", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "Aushungerungs-Klemme, Teil 2: die tiefste Stufe, auf die Ps Anteil begrenzt wird, wenn die älteste P-Anfrage länger als SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S wartet (0/1/2/3 = 100/75/50/25 %). Standard 1, also P bekommt dann mindestens 75 %. Die tiefste Stufe der Leiter (3 bei vier Stufen) schaltet die Klemme aus, weil sie nie senkt; weicher wird sie mit höherem SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S oder Stufe 2 (Kommentar der Release-Datei 27b-nvfp4-dual.env).",
        "gain": "", "cost": "", "satz_quelle": "dual_share.py ShareConfig.starve_max_rung (Zeile 158) und Klemme (Zeilen 414-417); Profilkommentar 27b-nvfp4-dual.env", "depends": []},
    "SGLANG_WEG2_DUAL_GRANT_RETRY_MS": {
        "kind": "env", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "Drosselung der KV-Vergabe-Wiederholung (#1530, nur Dual-Layout, nur PP0; 0 = AUS, dann probiert jede Scheduler-Runde es erneut). Mit N > 0 probiert eine wartende Anfrage höchstens alle N ms erneut, außer ein Karten-Ledger-Eintrag hat sich seit ihrem letzten Versuch geändert; Stufen-Tabellen werden nach mtime zwischengespeichert. Anlass im Code-Kommentar: Boot B9g 5 (04.10.), PP0 wiederholte neun wartende Beine in JEDER Runde (~2000 Vergaben/s), CPU 101 % unter der GIL, Forward und Publish verhungerten.",
        "gain": "Entlastet PP0 (CPU unter der GIL), solange P-Anfragen auf KV-Platz warten.", "cost": "",
        "satz_quelle": "environ.py SGLANG_WEG2_DUAL_GRANT_RETRY_MS (Kommentar, Zeilen 679-684); dual_p_kv_stage.py (Zeilen 668-681, 893)", "depends": []},
    "--dual-d-capture-prio": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE Stufe 1a: D nimmt seine CUDA-Graphen auf dem Stream höchster Priorität des Geräts auf (Graph-Knoten behalten die Priorität des Capture-Streams). Nur Gruppe D des Dual-Layouts.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-d-capture-prio (help=)", "depends": []},
    "--dual-p-mps-low-prio": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-SHARE Stufe 1b: P startet mit CUDA_MPS_CLIENT_PRIORITY=1 (unter normal; beim Verbinden gelesen, nur beim Boot).",
        "gain": "", "cost": "Braucht --dual-mps on, sonst steht eine benannte W-DUAL-SHARE-FALLBACK-Zeile im Log.",
        "satz_quelle": "launcher.py --dual-p-mps-low-prio (help=)", "depends": []},
    "--dual-p-sleep": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "Standard: AN mit --dual-share, sonst AUS. Mit --dual-unified-kv on: D-PRIORITÄT Stufe 2 (Nutzer 01.10.): ist D noch knapp, nachdem P gestoppt und sein KV freigegeben hat, schläft P und parkt seine Gewichte im Host-RAM (P bootet mit --enable-weights-cpu-backup und NICHT resident; das Host-Image wird bei der Pause angelegt und nach der Wiederherstellung freigegeben, kein ruhendes Image, KEIN-DAUER-HOSTRAM). off = nur Stufe 1; die Front druckt 'stage=2 unavailable (weights resident)'.",
        "gain": "", "cost": "Der Hilfetext nennt außerdem die Verweigerung W-DUAL-P-SLEEP-SHARE für 'on' zusammen mit --dual-share, solange der Baum Ps union-gebundenen Teil nicht außerhalb des Pools lädt (launcher.dual_p_sleep_armed prüft das am Baum).",
        "satz_quelle": "launcher.py --dual-p-sleep (help=; Verlaufsteil des Hilfetexts nicht übernommen)", "depends": []},
    "--dual-unified-kv": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3: ein KV-Pool je Karte, den P und D zur Laufzeit teilen (Nutzer-Order 30.09. 07:10Z/07:25Z; weg2/card_kv_ledger.py). P bildet KV nur ab, solange es prefillt, und pausiert, wenn D knapp ist. Braucht --dual-share.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-unified-kv (help=)", "depends": []},
    "--dual-p-kv-max-tokens": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3 --dual-unified-kv (Standard 196608): Ps KV-Pool-Zeilen (virtuell; Seiten kommen aus dem Karten-Pool). Jeder K/V-Puffer wird in dieser Größe angelegt und sofort gekürzt, die Boot-Spitze ist also ein Puffer (Token x Bytes je Token je Layer).",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-p-kv-max-tokens (help=)", "depends": []},
    "--dual-d-kv-max-tokens": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3 --dual-unified-kv (Standard 1048576): Ds KV-Pool in GLOBALEN Token (virtuell; D behält seinen Boot-Stand gemappt und wächst aus dem Karten-Pool). Muss Ds Boot-Kontext übersteigen; jeder Puffer wird in seinem Owner-Anteil davon angelegt und sofort gekürzt.",
        "gain": "", "cost": "", "satz_quelle": "launcher.py --dual-d-kv-max-tokens (help=)", "depends": []},
    "--dual-mps": {
        "kind": "flag", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "DUAL-TP3PP3: startet vor den Gruppen einen privaten MPS-Control-Daemon (Pipe-Verzeichnis unter dem Run-Verzeichnis des Boots), damit P- und D-Kernels auf einer Karte gleichzeitig laufen statt zeitgeteilt. Nur mit --dual-layout.",
        "gain": "", "cost": "VERWEIGERT, solange SGLANG_WEG2_DUAL_MPS_OPT_IN=1 fehlt: gemessen, beide Gruppen hängen unter Extend-großen Collectives (Repro v2 scjhru S1, Boots kw6pft/ndktv4). Das Dual-Layout läuft ohne MPS; Latenzwächter ist --dual-p-duty.",
        "satz_quelle": "launcher.py --dual-mps (help=)", "depends": []},
    # ---- Waechter-Schalter 27B (06.10.): Texte NUR aus der Meldeliste des 27B-Sitzes, je Satz die Zeile der Meldeliste in satz_quelle
    # (WML = /spinning/gpu-arb/deskq/done/waechter-schalter-27b-meldeliste-1006.md; Z. = Zeile dieser Datei). Nichts ergaenzt.
    "SGLANG_ENABLE_SCHEDULER_WATCHDOG_KILL": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Hart-Watchdog des Schedulers: Steht der Forward-Zähler still, wird nach dem Dump nach 5 s SIGQUIT an den Eltern-Prozess geschickt (die Gruppe stirbt). Standard 1. Bei 0 bleiben Dump und Logzeile, es wird KEIN Signal geschickt. Die Schwelle bleibt --watchdog-timeout (300 s); der weiche Watchdog bleibt unverändert.",
        "gain": "", "cost": "Bei 0 beendet nichts einen echten Scheduler-Hänger.",
        "satz_quelle": "WML Z. 12 (Abschnitt A)", "depends": []},
    "SGLANG_ENABLE_SUBPROCESS_WATCHDOG_KILL": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Ein Scheduler- oder Detokenizer-Kindprozess mit exit != 0 lässt den Server per SIGQUIT sterben. Standard 1. Bei 0 wird der Tod gemeldet, es wird aber kein Signal geschickt. Das Abfrage-Intervall bleibt 1 s.",
        "gain": "", "cost": "Bei 0 bleibt der Server als Hülle um den toten Rang stehen (Zombie-Gefahr).",
        "satz_quelle": "WML Z. 13 (Abschnitt A)", "depends": []},
    "SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "W17 Weg2GroupDead: Antwortet eine Gruppe nicht auf /health (und ihr Prozess ist tot) oder ist ein Rang gehalten, stoppt die Front. Standard 1. Bei 0 steht die Zeile „W17 ... suppressed“ im Log und es wird nicht gestoppt; die WEG2-HEALTH-Zeilen und die /health-Fakten bleiben unverändert, die Front-/health meldet weiter 503 (Tatsache). Hängt an SGLANG_WEG2_GROUP_DEAD_STREAK.",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 14 (Abschnitt A)", "depends": []},
    "SGLANG_WEG2_GROUP_DEAD_STREAK": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Wie viele aufeinanderfolgende /health-Fehlschläge (5-s-Takt) W17 braucht (ganze Zahl ≥ 1, Standard 2). Ein gehaltener Rang stoppt weiterhin sofort. Für den Stopp nur wirksam, wenn W17 an ist (SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP). Nur im 27B-Baum gilt die Zahl auch für das 503 der Front-/health (dann immer, damit beide dieselbe Zahl lesen); im NF-Baum nur für das W17-Gate.",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 15 (Abschnitt A); Baum-Unterschied WML Z. 39 (Abschnitt D, Punkt 2)", "depends": []},
    "SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "W2 Weg2DrainStuck: So viele W1 DrainRefused in Folge stoppen die Front. Standard 1. Bei 0 verweigert jedes W1 weiter den Flip und W2 wird nur als ERROR geloggt. Hängt an SGLANG_WEG2_DRAIN_STUCK_REFUSALS.",
        "gain": "", "cost": "Bei 0 wiederholt ein hängender Drain W1 endlos.",
        "satz_quelle": "WML Z. 16 (Abschnitt A)", "depends": []},
    "SGLANG_WEG2_DRAIN_STUCK_REFUSALS": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Die „in Folge“-Zahl für W2 (ganze Zahl ≥ 1, Standard 3). Nur wirksam, wenn W2 an ist (SGLANG_WEG2_ENABLE_DRAIN_STUCK_STOP).",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 17 (Abschnitt A)", "depends": []},
    "SGLANG_WEG2_ENABLE_CONTROLLER_DEAD_STOP": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "CONTROLLER-DEAD (#1264): Eine Ausnahme im offenen Flip stoppt die Front mit Namen. Standard 1. Bei 0 bleiben Zeile und Traceback, es wird nicht gestoppt.",
        "gain": "", "cost": "Bei 0 bleibt die Front in „flipping“ (der Controller überspringt jede weitere Runde): sie bedient nichts, ist aber „am Leben“. Der gefährlichste der Schalter; Dashboard-Warnung empfohlen.",
        "satz_quelle": "WML Z. 18 (Abschnitt A)", "depends": []},
    "SGLANG_WEG2_HOST_GUARD_W22": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Nur im 27B-Baum. W22 HostWatermarkBreached (Host-RAM-Pegel) stoppt die Front. Text-Wert on/off (on/1/true/yes, off/0/false/no), Standard on. Bei off wird das Verdikt geloggt („HOST GUARD W22 AUS: NOT stopping“, gedrosselt), es wird NICHT gestoppt; beim Front-Start steht eine laute WARNUNG „HOST GUARD W22 AUS: Host-RAM ist nicht mehr geschützt“. Ein Tippfehler gilt als on. Die Schwelle (--host-riegel-gib, host_ledger-Marke) bleibt unverändert.",
        "gain": "", "cost": "Bei off ist der Host-RAM nicht mehr geschützt (Wortlaut der Start-Warnung).",
        "satz_quelle": "WML Z. 19 (Abschnitt A); 27B-eigen WML Z. 39 (Abschnitt D, Punkt 1)", "depends": []},
    "SGLANG_WEG2_HOST_GUARD_W98": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Nur im 27B-Baum. W98 HostRateLatched (Rate-Latch), getrennt von W22 schaltbar; sonst wie W22 (Text-Wert on/off, Standard on; bei off WARNUNG „HOST GUARD W98 AUS: Host-RAM ist nicht mehr geschützt“). Bei off läuft der W22-Pegeltest im selben Tick weiter.",
        "gain": "", "cost": "Bei off ist der Host-RAM nicht mehr geschützt (Wortlaut der Start-Warnung).",
        "satz_quelle": "WML Z. 20 (Abschnitt A); 27B-eigen WML Z. 39 (Abschnitt D, Punkt 1)", "depends": []},
    "SGLANG_ADMISSION_WEDGE_SECONDS": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "ADMISSION-WEDGE-Alarm: Sekunden ohne erstes Token bei queued>0 und running=0 (speist den Wedge-Status, die Recovery und die Übergabe bei Intake-Stall). Standard 20,0 s; nicht-positiv = 20. Ungesetzt bleibt der Poll 10 s; gesetzt folgt er im 27B-Baum der halben Schwelle (der NF-Baum lässt ihn bei 10 s). Die Recovery-Schwelle ist ein eigener Wert (SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS; Form nextflash/qwen27b 2,0 s, sonst 60 s); nur im 27B-Baum warnt beim Start eine Zeile, wenn Recovery > Alarm (nicht beim Default-Paar 20/60).",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 26 (Abschnitt B); Baum-Unterschiede WML Z. 39 (Abschnitt D, Punkte 1 und 3)", "depends": []},
    "SGLANG_ADMISSION_WEDGE_MODE": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Nur im 27B-Baum. Modus des ADMISSION-WEDGE-Alarms: act (Standard) = Alarm + Status-Datei + Recovery-Treiber (heute). log = Alarm + Status, KEIN Recovery-Versuch. off = der Wedge-Watchdog-Thread startet nicht (kein Alarm, kein Status, keine Recovery; wedge_status liest „keine Messung“). Ein unbekanntes Wort (auch „stop“) gilt als act. SGLANG_WEDGE_STATUS_DISABLE bleibt unabhängig.",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 27 (Abschnitt B); 27B-eigen WML Z. 39 (Abschnitt D, Punkt 1)", "depends": []},
    "SGLANG_PREFILL_LIVELOCK_SECONDS": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "PREFILL-LIVELOCK: Sekunden ohne Decode-Runde bei running>0 und queued>0. Standard 20,0 s; nicht-positiv = Standard. Unabhängig von der Wedge-Schwelle.",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 28 (Abschnitt B); Regel „nicht-positiv = Default“ WML Z. 6", "depends": []},
    "SGLANG_PREFILL_LIVELOCK_MODE": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "Nur im 27B-Baum. Modus des PREFILL-LIVELOCK-Alarms: log (Standard) = Urteil + ERROR-Zeile, nie Recovery. off = kein Livelock-Urteil. Ein unbekanntes Wort gilt als log. Es gibt keinen stop-Modus für reine Alarme.",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 29 (Abschnitt B); Überschrift Abschnitt B WML Z. 22; 27B-eigen WML Z. 39 (Abschnitt D, Punkt 1)", "depends": []},
    "SGLANG_WEG2_FLIP_STALL_SLACK": {
        "kind": "env", "group": "Wächter", "level": "experte", "planner_derived": False,
        "text": "WEG2-FLIP STALL (#1262, Tier-3-Signal des Deadman): Faktor auf den zuletzt gemessenen Flip. Standard 4,0; nicht-positiv = 4,0. Vor dem ersten Flip gilt weiter --drain-deadline-s. Der Detektor stoppt selbst nie.",
        "gain": "", "cost": "", "satz_quelle": "WML Z. 30 (Abschnitt B)", "depends": []},
    # ---- Dual: D-COMPACT und ARENA-AUX-SPILL (Quellen je Eintrag)
    "SGLANG_WEG2_DUAL_D_COMPACT": {
        "kind": "env", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "D-COMPACT: D verschiebt lebende KV-Zeilen laufender Sitze in freie niedrige Zeilen, damit sein KV-Regal schrumpfen kann, wenn P auf eine Karte wartet. Standard an, 0 = aus. Wirkt nur im Dual-Layout (SGLANG_WEG2_DUAL_LAYOUT=1) in Gruppe D mit SGLANG_WEG2_DUAL_D_KV_MAX_TOKENS > 0; Flip, INT8, NF und Gruppe P erreichen das Modul nie. Greift erst, wenn P länger als SGLANG_WEG2_DUAL_D_LIVE_YIELD_WAIT_S (Standard 4 s) gewartet hat. Scheitert die Verschiebung, steht eine benannte Zeile „D-COMPACT REFUSED reason=...“ im D-Log. Unbelegt: die Wirkung am Metall (laut Bericht nur CPU-getestet, kein Metall-Boot).",
        "gain": "", "cost": "", "satz_quelle": "dual-dcompact-vorschlag-1006.md Z. 4 (Meldeliste) und Z. 10 (nicht bewiesen); environ.py-Kommentar zu SGLANG_WEG2_DUAL_D_COMPACT; dual_d_compact.py Modulkopf (Gate: dual_d_kv_stage.armed() AND Schalter)", "depends": []},
    "SGLANG_WEG2_DUAL_ARENA_AUX_SPILL_S": {
        "kind": "env", "group": "Dual", "level": "experte", "planner_derived": False,
        "text": "Q-1190b ARENA-AUX-SPILL (nur Dual-Layout, beide Gruppen): Steht die Wand der gemeinsamen KV-Arena so viele Sekunden, nehmen das Abgeben von D (D-ARENA-YIELD) und das Trimmen von P (ARENA-TRIM) zusätzlich Blätter, deren einzige Verweigerung der nur im Host liegende Zusatzzustand (aux) ist: nach jedem gewöhnlichen Blatt, weiterhin zuerst die L3-Kopie jeder KV-Seite, nie ein gesperrter, laufender, von einem END-Anker (V1) gehaltener oder beanspruchter Knoten. Das Blatt geht ganz, sein Elternknoten wird Blatt und folgt. Standard 30 s; 0 = aus = die alte Verweigerung. Eine nicht lesbare Zahl gilt als 30 s, eine negative als 0. Roher os.environ-Schalter ohne Eintrag in environ.py. Benannte Zeilen: „Q-1190b DUAL ARENA-AUX-SPILL ON/OFF“ und „aux=1 aux_leaves=N“ auf der D-ARENA-YIELD-/ARENA-TRIM-Zeile.",
        "gain": "", "cost": "", "satz_quelle": "dual_arena_spill.py Kommentar „THE FIX“ (Zeilen 348-359) und aux_after_s (Zeilen 376-383); Commit 6e3060ec02 (Nachricht)", "depends": []},
    # ---- X-CURVES (AP-K, 06.10.): argparse help= des Launchers; Refusals W190-W197 aus x_curves.py
    "--x-mode": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "X-CURVES (Nutzer 06.10.: nur fixed und curve): wie die Front X setzt. fixed = --tp-prefill-max-tokens für den ganzen Boot, keine Live-Neuberechnung. curve = X PRO ANFRAGE aus --x-curves (die D-/P-Prefill-Kurven und der Flip-Preis dieses Modells × dieser Form × dieser Hardware); --x-ceiling-tokens daneben begrenzt X von oben, Ds W50-Riegel ist der NIEDRIGERE Wert aus der Hüllkurve der Kurven und dieser Obergrenze. Ungesetzt = die Front genau wie vor dem Flag, argv byte-identisch. Im Dual-Layout verweigert (W197: kein X-Umbau für Dual). Ein unbekanntes Wort verweigert W196; curve ohne --x-curves verweigert W190.",
        "gain": "", "cost": "Im Dual-Layout verweigert (W197).",
        "satz_quelle": "launcher.py --x-mode (help=, Zeilen 23013-23020); Refusals W190, W196, W197: x_curves.py (Zeilen 88-104, _check_mode_words)", "depends": []},
    "--x-curves": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "X-CURVES: die Kurvendatei (weg2-x-curves/1, tools/build_x_curves.py). Nur mit --x-mode curve (sonst W194); fehlt oder ist die Datei nicht lesbar W190, ist sie fehlerhaft W191, stammt sie von einem anderen Modell, einer anderen Form oder anderer Hardware W192: beim Start verweigert. Im Dual-Layout verweigert (W197).",
        "gain": "", "cost": "",
        "satz_quelle": "launcher.py --x-curves (help=, Zeilen 23021-23024); W197: x_curves.py refuse_in_dual", "depends": []},
    "--x-curves-beyond": {
        "kind": "flag", "group": "Prefill", "level": "experte", "planner_derived": False,
        "text": "X-CURVES: eine Anfrage, die tiefer liegt, als die Kurven reichen: clamp (Standard, zum Preis der tiefsten Zeile berechnet und benannt) oder refuse (W193 an der Front). Nur mit --x-mode curve (sonst W194); ein unbekannter Wert verweigert W196; im Dual-Layout verweigert (W197).",
        "gain": "", "cost": "refuse weist solche Anfragen an der Front ab (W193).",
        "satz_quelle": "launcher.py --x-curves-beyond (help=, Zeilen 23025-23027); W194/W196/W197: x_curves.py (Zeilen 94-104, _check_mode_words, refuse_in_dual)", "depends": []},
}
