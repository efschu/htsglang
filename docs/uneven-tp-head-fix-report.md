# Uneven-TP Head-Geometrie-Fix — Untersuchungs- und Verifikationsbericht

**Fork:** htsglang (sglang-Fork für heterogenes uneven Tensor Parallelism)
**Branch/Commit:** `feature/uneven-tp` @ `f7ff51435`
**Hardware:** 1× RTX 5090 (32 GB) + 2× RTX 3080 (20 GB), TP=3 uneven
**Modell:** Qwen3.6-27B GDN-Hybrid (MoE + Gated-Delta-Net + wenige MHA-Layer), FP8, MTP
**Datum:** 2026-07-14

---

## 1. Zusammenfassung

Bei langem Kontext (ab ~7k Tokens) degenerierte die Ausgabe unter uneven TP
gelegentlich zu Wiederholschleifen mit Buchstaben-Verstümmelung („Rmisches"
statt „Römisches"). Die Wurzel: Der **flashinfer-Attention-Backend berechnete
die Query-Head-Zahl pro Rank falsch** — als gleichmäßiges
`num_attention_heads // tp_size` statt der tatsächlichen uneven Aufteilung, die
das Modell vornimmt. Der Fix (`_local_attn_head_counts()`) übernimmt die
per-Rank-Head-Geometrie vom Modell. Verifiziert über alle Feature-Kombinationen
(uneven TP, self-calibration, MTP/NEXTN, HiCache-GDN). Der Default-Pfad (even TP)
ist byte-identisch.

---

## 2. Der Bug

### 2.1 Mechanismus

Unter `--rank-tp-ratio auto` teilt das **Modell** (`qwen3_next.py`) die
Attention-Heads UNGLEICH über die Ranks auf — ganze GQA-Gruppen pro Rank, via
`tp_partition_size()` + `local_num_*_heads`. Der **flashinfer-Backend** aber
berechnete die Head-Zahl an drei Stellen (`should_use_tensor_core`, decode- und
prefill-Indices-Updater) *selbst neu* als gleichmäßiges
`num_attention_heads // attn_tp_size`. Damit bekam `plan()`/`run()` auf jedem
Rank die falsche `num_qo_heads` und damit eine falsche `gqa_group_size`.

### 2.2 Konkrete Zahlen (Qwen3.6-27B: 24 Q-Heads / 4 KV-Heads, TP=3, ratio [2,1,1])

| Rank | GPU | REAL (Modell) q/kv → gqa | PRE-FIX (flashinfer) q/kv → gqa |
|------|------|--------------------------|-------------------------------|
| 0 | 5090 | 12 / 2 → **6** | 8 / 2 → **4** ❌ |
| 1 | 3080 | 6 / 1 → **6** | 8 / 1 → **8** ❌ |
| 2 | 3080 | 6 / 1 → **6** | 8 / 1 → **8** ❌ |

`num_kv_heads` war bereits per-Rank-korrekt (auto-resolved: 2/1/1) — **nur
`num_qo_heads` war falsch**. Dadurch stimmte die geplante `gqa_group_size` auf
*keinem* Rank mit der echten Q/K/V-Tensor-Geometrie überein.

### 2.3 Warum nur bei langem Kontext

Split-KV-Attention (das Zerlegen langer KV-Sequenzen in Chunks + Merge) wird
erst bei langem KV aktiviert. Bei kurzen Prompts (ein einziger Tile, kein Split)
ist die falsche Head-Zahl folgenlos. Bei langem Kontext wird die Split-Partition
falsch dimensioniert → per-Rank-korrupte Attention → der `o_proj`-all-reduce
kombiniert divergierende Partials → Degeneration.

---

## 3. Der Fix

`_local_attn_head_counts(model_runner)` in `flashinfer_backend.py` leitet die
per-Rank `(num_qo_heads, num_kv_heads)` aus derselben Logik ab, die das Modell
selbst nutzt (`tp_partition_size(units=total_num_kv_heads)` +
`get_num_kv_heads(rank=...)`), und wird an den drei Stellen eingesetzt.

- **Default-Pfad byte-identisch:** degradiert bei even TP zu `total // tp_size`.
- **Diff:** +61 / −18 Zeilen, isoliert auf `flashinfer_backend.py`.
- **Review bestätigt:** `gqa_group_size` (Zeile 2293) ergibt nach dem Fix
  uniform 6 auf allen Rängen; für fp8-KV ohnehin toter Code (Tensor-Core-Pfad
  greift vorher).

---

## 4. Verifikations-Ergebnisse

Alle Tests: uneven TP=3 (5090 + 2×3080), fp8-KV, `--rank-tp-ratio auto`,
`SGLANG_UNEVEN_MLP_VECTOR=63,32,41`, temperature=0 (Greedy).

### 4.1 Methodik-Kalibrierung

Bei kurzem Kontext ist das Modell vollständig deterministisch — Referenzwert
für „normale" Konsistenz:

| Test | bit-identisch | Ähnlichkeit |
|------|---------------|-------------|
| Kurzprompt ×4 | 4/4 (1,00) | 1,00 |

**Wichtige Messlehre:** Bei langem Kontext fällt die Roh-Text-Ähnlichkeit auf
0,40–0,76 — aber das misst nur die **Formulierungs-/Reasoning-Variation**, nicht
die Aufgaben-Korrektheit. Der aussagekräftige Test ist **Needle-in-Haystack mit
Ground-Truth** (Retrieval-Korrektheit + Konsistenz), nicht bit-/Text-Ähnlichkeit.

### 4.2 Needle-in-Haystack (Retrieval-Korrektheit) — MIT Fix

Eindeutiger Code im langen Kontext, erzwungene Kurzantwort, je 4 Läufe:

| Kontext | Nadel-Position | Code korrekt | Konsistent |
|---------|----------------|--------------|------------|
| 1,5k tok | 50 % | 4/4 | ✅ |
| 8k tok | 50 % | 4/4 | ✅ |
| 18k tok | 50 % | 4/4 | ✅ |
| 18k tok | 10 % | 4/4 | ✅ |
| 18k tok | 90 % | 4/4 | ✅ |

→ **100 % korrekt + konsistent** bei allen Längen und Positionen.

### 4.3 Baseline OHNE Fix (Notwendigkeits-Nachweis)

| Test | Baseline (ohne Fix) | Head-Fix |
|------|---------------------|----------|
| needle 8k | 4/4 | 4/4 |
| needle 18k | **3/4** (1× verfehlt, inkonsistent) | 4/4 |
| gesund-5k (lange Generierung) | 3/4 kohärent (1× Schleife) | 6/6 kohärent |
| Buchstaben | „Rmisches", „Quntenmechanik" | korrekt |

**Einordnung:** Die Degeneration ist *probabilistisch* — bei komfortablem VRAM
(reserve 8192) tritt sie selten auf (in einer 16er-Stichprobe 0×). Der Fix ist
daher **durch den mathematisch nachweisbaren Geometrie-Fehler** gerechtfertigt
(gqa 4 statt 6 auf der 5090), nicht durch eine Degenerationsrate. Die frühen,
drastischen Degenerationen liefen zusätzlich unter VRAM-Druck (reserve 2048,
teils 40 MB frei) — ein möglicher Mit-Verstärker.

### 4.4 MTP / NEXTN (volle Zielconfig)

Config: head-fix + `--speculative-algorithm NEXTN` (steps 3, topk 1, draft 4).

| Metrik | Wert |
|--------|------|
| needle 8k / 18k | 3/3 korrekt + konsistent |
| Accept-Length | 2,58 (accept rate 0,53) |
| Decode-Durchsatz | 85–89 t/s (CUDA-Graph aktiv) |
| Decode ohne NEXTN (Vergleich) | ~40 t/s |

→ MTP verdoppelt den Decode-Durchsatz und liefert korrektes Retrieval.

### 4.5 HiCache-GDN (hierarchischer KV+Mamba-Cache mit Host-Offload)

Config: head-fix + `--enable-hierarchical-cache --hicache-mem-layout page_first_direct`.

**Host-Pool-Allokation (folgt dem uneven Split):**

| Rank | GPU | Host-KV | Host-Mamba |
|------|------|---------|-----------|
| 0 | 5090 | 14,97 GB | 13,32 GB |
| 1 | 3080 | 7,48 GB | 6,66 GB |
| 2 | 3080 | 7,48 GB | 6,66 GB |

**Host-Loadback nach erzwungener Eviction:**

| Schritt | Ergebnis |
|---------|----------|
| Referenz-Nadel kalt | Code korrekt (99137) |
| 48 Füll-Prompts (GPU-Cache überschrieben) | — |
| Re-Query aus Host-Pool | **3/3 korrekt + konsistent** |
| Host-Loadback-Evidenz | `#cached-token: 22528` bei niedrigem GPU-usage |

→ Der Mamba+KV-Host-Roundtrip ist korrekt. Der frühere Divergenzbefund
(807/999/472 Zeichen beim Host-Resume) war ein **Symptom des head-fix-Bugs**,
kein HiCache-Roundtrip-Fehler.

### 4.6 HiCache + MTP (volle Zielkonfiguration, Kurztest)

Config: head-fix + HiCache + `--speculative-algorithm NEXTN` — alle Features zusammen.

| Metrik | Wert |
|--------|------|
| Boot | ok (Host-Pools uneven: 5090 10,78 GB, 3080er je 5,39 GB) |
| needle 8k / 18k | 3/3 korrekt + konsistent |
| Accept-Length | 2,55–2,65 |
| Decode-Durchsatz | 82–89 t/s (CUDA-Graph) |

→ Die komplette Zielkonfiguration (uneven TP + self-calibration + per-GPU-Reserve
+ MTP + HiCache-GDN) läuft zusammen, mit korrektem Retrieval und aktivem MTP.

---

## 4.7 Grenze: TP > KV-Heads

Der uneven-Split verteilt KV-Heads als **ganze, unteilbare Units** (jeder Rank
≥ 1 Unit). `partition_units()` lehnt `tp_size > num_kv_heads` mit einem klaren
Fehler ab (`"Cannot give each of N ranks at least one of M units"`) — **fail-fast,
keine stille Korruption**.

- Qwen3.6-27B hat 4 KV-Heads → **TP=2/3/4 werden unterstützt**, TP≥5 abgelehnt.
- Fehlend wäre **KV-Head-Replikation** (mehrere Ranks teilen einen KV-Head), wie
  Standard-GQA-TP sie bei `tp_size > num_kv_heads` macht. In Kombination mit
  uneven Ratios nicht-trivial und im Fork nicht implementiert → möglicher
  Follow-up (eigenes Feature, kein Bug).

---

## 5. Kern-Feature-Nachweise (uneven TP mit heterogenen GPUs)

### 5.1 Self-calibration (`--rank-tp-ratio auto`)

Aus den NVML-Totalen abgeleitete Memory-Budgets (reserve 4096/GPU):

```
derived memory budgets [28511, 16384, 16384] MiB   (5090, 3080, 3080)
```

### 5.2 Uneven KV-Split (DCP nach Kapazität)

KV-Cache 343.105 Tokens, aber pro Rank ungleich groß — die stärkere Karte trägt
den größeren Anteil:

| Rank | GPU | K-Cache pro Rank |
|------|------|------------------|
| 0 | 5090 | 2,62 GB |
| 1/2 | 3080 | 1,31 GB (je) |

### 5.3 Self-calibrating token vector

Der Server berechnet den optimalen Kalibrierungs-Vektor selbst und empfiehlt ihn:

```
uneven TP: restart with SGLANG_UNEVEN_MLP_VECTOR=68,35,33
to raise the KV pool from 343105 to ~452988 tokens   (+32 %)
```

### 5.4 KV-Pool und VRAM-Ausnutzung nach Reserve

Der per-GPU-Reserve ist der **Haupthebel** für die KV-Pool-Größe. Er wird von
jedem Rank-Budget abgezogen; da der Token-Pool von der schwächsten Karte (3080)
begrenzt wird, wirkt sich Reserve dort überproportional aus:

| reserve/GPU | Budgets (5090/3080/3080) MiB | KV-Pool (Tokens) | VRAM-Ausnutzung |
|-------------|------------------------------|------------------|-----------------|
| **2048** | 30559 / 18432 / 18432 | **637.985** | 5090 **97 %**, 3080er **92 %** |
| 4096 | 28511 / 16384 / 16384 | 343.105 | 5090 84 %, 3080er 74–80 % |
| 8192 | — | — | 5090 ~67 %, 3080er ~58–65 % |

→ `reserve 2048` verdoppelt den KV-Pool gegenüber `4096` und füllt jede Karte
nahezu voll (92–97 %). **Trade-off:** niedriger Reserve = mehr KV, aber weniger
Headroom gegen Runtime-OOM (CUDA-Context, Aktivierungs-Spitzen, Graph-Capture).
Die Wahl ist bewusst dem Nutzer überlassen (Kern-Feature, per GPU einstellbar).

Der self-calibrated Vektor ist reserve-abhängig: bei `4096` empfiehlt der Server
`68,35,33` (+33 % KV); bei `2048` ist `63,32,41` bereits optimal (kein Vorschlag).

### 5.4b Uneven KV-Split in Bytes (max-KV-Boot, reserve 2048)

Gleicher Token-Pool auf allen Rängen, aber die Bytes folgen dem uneven
Head-Share — jede Karte trägt proportional zu ihrem Budget:

| Rank | GPU | K-Cache | Token-Pool |
|------|------|---------|-----------|
| 0 | 5090 | 4,87 GB | 637.985 (uniform) |
| 1/2 | 3080 | 2,43 GB (je) | 637.985 (uniform) |

### 5.5 Durchsatz (Referenz)

| Phase | Wert |
|-------|------|
| Prefill (Median über 388 Chunks) | 1407 t/s (max 1564) |
| Decode mit MTP | 85–89 t/s |
| Decode ohne MTP | ~40 t/s |

(Der einzelne 180-t/s-Ausreißer war der erste Prefill-Chunk eines frischen
Requests — Mamba-State-Setup-Fixkosten, amortisiert sich ab dem 2. Chunk.)

### 5.6 KV-Kapazität im Kontext (FP8 vs INT4/AWQ)

Der FP8-Max von ~638k Tokens (reserve 2048) ist strukturell durch die
Weights-Größe begrenzt, nicht durch das uneven-TP-Feature:

| Modell | Weights (TP=3) | frei für KV | erreichbarer Kontext |
|--------|----------------|-------------|----------------------|
| FP8 (dieser Test) | ~8,81 GB/Rank (~26 GB) | begrenzt | ~638k Tokens |
| AWQ/INT4 (vLLM-Referenz) | ~halb (~13 GB) | ~13 GB mehr | ~1M Tokens |

Der Unterschied zu den vLLM-1M-Werten ist der **Quantisierungs-Hebel**: INT4-
Weights sind ~halb so groß wie FP8, was ~13 GB mehr KV-Platz pro Setup frei
macht. Bei gleicher Quantisierung sind die KV-Größen vergleichbar; für 1M+ ist
das INT4/AWQ-Modell nötig, kein sglang-Limit.

---

## 6. Methodik-Lehren

1. **Bit-/Text-Ähnlichkeit misst bei Reasoning-Modellen das falsche.** Die
   Formulierung divergiert bei langem Kontext (Split-KV-Jitter), die *Antwort*
   bleibt korrekt. Aussagekräftig: Needle-Retrieval mit Ground-Truth
   (Korrektheit + Konsistenz), kalibriert gegen Kurzprompt-Referenz.
2. **Degeneration ist probabilistisch** und wird durch VRAM-Druck mit-verstärkt
   — ein Config-Faktor als möglicher Confounder. Der Fix stützt sich daher auf
   den mathematischen Geometrie-Beweis, nicht auf Degenerationsraten.
3. **VRAM-Headroom** großzügig setzen (reserve) für Tests, sonst OOM beim
   CUDA-Graph-Capture auf den 20-GB-Karten — verfälscht Läufe als scheinbare
   Degeneration/Hänger.

---

## 7. Offene Follow-ups (niedrige Priorität)

| Punkt | Beschreibung | Nutzen |
|-------|--------------|--------|
| **Fail-fast-Guard** | Dieselbe `//tp_size`-Bug-Klasse steckt in ~10 anderen Backends (triton, MLA-Varianten, wave). NICHT blind mitfixen (MLA nutzt `kv_lora_rank`, kein GQA → Helper wäre falsch). Stattdessen Start-Guard „uneven TP nur auf flashinfer validiert". | Wandelt stille Korruption in klaren Fehler. Klein, sicher. |
| **deterministic-inference-Deadlock** | `--enable-deterministic-inference` deadlockt auf uneven TP (collective-desync: Rank 0/5090 im Kernel, 3080er warten im Request-Broadcast). | Nur für Bit-Reproduzierbarkeit; funktional nicht nötig. Tiefer Fix. |
| **MLA-Workspaces / KDA-conv[0]** | Reste aus #50; MLA = DeepSeek-artig, KDA = Multi-Conv. | Für Qwen3.6-GDN irrelevant. |
