# 03 — Was funktioniert (und was du NICHT anfassen sollst)

Massgeblich und aktueller als diese Liste: `FEATURES_VS_UPSTREAM.md` in der
Repo-Wurzel. Hier die Kurzfassung fuer den Standardlauf-Kontext.

## Validiert und im Standardlauf aktiv

| Feature | Status |
|---|---|
| uneven TP (`--rank-tp-ratio auto-performance`) | validiert, Standard |
| uneven DCP gewichtet (`SGLANG_UNEVEN_DCP=1` + `_WEIGHTED=1`) | validiert, Standard |
| NEXTN-Spekulation k=3, topk=1 | validiert, Standard |
| FP8-KV (`fp8_e4m3`) | validiert (auf sm86 bit-exakt gegengeprueft) |
| Volle CUDA-Graphen (Decode full-graph, Prefill breakable-graph) | Standard — per Default AN, nichts tun |
| Prefix-Caching, chunked prefill | Standard |
| Metrics-Endpoint (`--enable-metrics`) | Pflicht in jedem Boot |

## Validiert, aber NICHT Teil des Standardlaufs (nur auf Auftrag)

- GGUF-Modelle (Q3/Q4/Q6/Q8) inkl. MoE, uneven TP=3 — eigene Rezepte im Runbook.
- DFLASH (Upstream seit sglang PR #31840, 2026-07-27); unser Fork-Delta: GGUF lm_head für Drafters, Solo-Draft-KV-Pool, Platzierungsflag. Solo-Platzierung auf der 5090 via `--speculative-draft-placement solo` (eigener Pfad; Multiturn schwächer als NEXTN auf strukturiertem Output, siehe docs/dev/TASK_285_DFLASH_STRUCTURED.md).
- Adaptive draft-Länge (Upstream seit sglang, basiert auf AdaptiveController); unser Fork-Delta: Erweiterung auf mehrere Achsen (Algorithmus NEXTN↔DFLASH, topk, Geometrie), Cross-Algo-Leiter #156.
- Experten-Offload (MoE-Modelle), KV-Spill ins RAM, Suspend/Hibernate (#89).
- PD-Disaggregation, Cross-Rig-PP (braucht PVE-Host, nicht Container).
- Dual-Lane/Mehrfach-Gruppen (Forschungsstand — NICHT ohne Operator-Briefing).

## Gesperrt / verworfen — NICHT versuchen, auch nicht "zum Testen"

- **topk>1 (Tree-Spec) unter uneven-DCP**: still-falsch + perf-negativ, Guard
  ist absichtlich drin. Nicht umgehen.
- Reserve 2200 auf 3080ern, Vocab-Ratio 7,3,3, Bidir-Ring, Prefill-Rebalance
  (mehr MLP auf die 5090): alles gemessen verworfen. Nicht neu probieren.
- `--max-running-requests` unbounded bei Hybrid-Modellen: Auto-Mamba-Cap beachten.

## Wichtig zu wissen, bevor du etwas "reparierst"

- Eager-Modus ist KEIN gueltiger Gegencheck (versteckt Graph-Replay-Bugs).
- fp8@sm8x erzeugt bei identischen Prompts nicht zwingend identische Tokens
  (bekannt, kein Bug). Byte-Vergleiche nur mit den dafuer gebauten Harnessen.
- GDN-Prefill ab ~109 Token ist upstream nicht reproduzierbar (bekannt).
- Wenn etwas nicht laeuft: Befund sichern (Log-Pfad + grep-Zeilen, py-spy bei
  Haenger), Karten freigeben, ZURUECKMELDEN. Nicht selbst im Fork debuggen.
