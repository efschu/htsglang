# Wie starte ich den htsglang-Standardlauf (Uebergabe-Paket)

Stand: 2026-07-29, Repo-Stand integration/r3-probe-next2 @ a9fd89aab6.
Dieses Paket macht einen fremden Agenten startfaehig fuer UNSEREN
Standardlauf, ohne dass er raten muss. Es beschreibt NUR Funktionen,
die validiert laufen.

## Was der Standardlauf ist

Qwen3.6-27B-FP8 auf Rig 1 (1x RTX 5090 32 GB + 2x RTX 3080 20 GB) mit:
- **uneven TP=3** (`--rank-tp-ratio auto-performance`: die 5090 traegt
  den groesseren Shard, Verhaeltnis wird automatisch gewaehlt)
- **uneven DCP** (gewichteter Token-Split des KV ueber die Karten —
  grosse Karte haelt mehr Kontext)
- **NEXTN-Spekulation, fest k=3** (topk=1, 4 Draft-Tokens)
- **FP8-KV-Cache** (`fp8_e4m3`)
- **volle CUDA-Graphen** (Decode full-graph; NIE eager validieren)

Erwartungswerte gesund: Decode (bs=1, mit Spec) grob 100-120 tok/s,
Prefill (TP=3, 2048er-Chunk) grob 1000-1300 tok/s, Accept-Laenge je nach
Inhalt ~2,7-3,3 (Code hoeher als Prosa).

## Reihenfolge fuer einen neuen Agenten

1. `README.md` (diese Datei) lesen.
2. `01_standardlauf.md` — Umgebung, exaktes Startkommando, Ready-Check,
   Smoke-Request, sauberes Beenden.
3. `02_regeln_und_fallen.md` — Arbitrierung, Fallen, Verbote. PFLICHT.
4. `03_was_funktioniert.md` — welche Features tragen, welche gesperrt sind.
5. `04_messen.md` — nur noetig, wenn gemessen werden soll.

## Die zwei wichtigsten Wahrheitsquellen im Repo

- `docs/rig-runbook.md` — die MASSGEBLICHE Betriebsanleitung. Weicht
  dieses Paket vom Runbook ab, gilt das Runbook.
- `FEATURES_VS_UPSTREAM.md` (Repo-Wurzel) — ehrlicher Feature-Status.

## Harte Grundregeln (Kurzfassung, Details in 02)

- Karten NUR mit Lock nehmen (`/tmp/gpu-card-N.lock` = VERZEICHNIS mit
  Info-Datei) und nach dem Lauf freigeben + melden.
- Aktuell (2026-07-29) laeuft ein GPU-Treiber-Update; Karten koennen vom
  Nutzer belegt sein. Im Zweifel NICHT booten, sondern zurueckmelden.
- Nie Serverlogs in den Agenten-Kontext ziehen. Nie unbegrenzt warten.
- `py-spy dump` vor jedem Kill. Nur eigene PIDs beenden, nie breites pkill.
