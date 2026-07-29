# EVAL — P2P-Aussicht: Solo-5090-Prefill gegen verteilten Decode (Rig 1)

Ueberschlagsanalyse auf Nutzerfrage 2026-07-29. **Kein Neu-Messen, kein Boot,
keine GPU.** Alles unten ist entweder ein bereits gemessener Anker (mit Quelle)
oder eine Roofline-/Additionsrechnung darauf. Basis fuer jeden Vergleich ist die
**heutige TP=3-uneven-DCP-Konfiguration**.

Hardware: 5090 32607 MiB / ~1,79 TB/s / sm120; 2x 3080 20480 MiB / ~0,76 TB/s /
sm86; alle PHB, GPU0 auf x4, heute kein P2P. Angenommene Aenderung: Treiber-Update
bringt P2P, **3080er nur ueber das kleine 256-MiB-BAR-Fenster, 5090 volles
32-GiB-BAR** (DESIGN_201, P2P-PRAEZISIERUNG).

---

## 0. Kurzfassung in fuenf Zeilen

1. **Ein Solo-5090-Prefill enthaelt null kartenuebergreifenden Verkehr. P2P wirkt
   darauf exakt 0 %.** Der Solo-Pfad ist heute schon ~3,1x schneller als der
   TP=3-Prefill (3510 gegen 1114 Tok/s je 2048er-Chunk) — und zwar genau, weil
   er den 68-%-Kollektivboden nicht bezahlt.
2. **Die Obergrenze, die P2P dem TP=3-Prefill geben kann, ist die Solo-Zahl
   selbst.** Bei Kollektivboden = 0 bleibt das Fenster = max(Rang-Compute) =
   586,5 ms je 2048 (die 3080er takten) = 3492 Tok/s ~ 3510 Tok/s solo. Die
   ganze TP-Parallelitaet des Prefills geht heute fuer die Kompensation der
   langsamen Karten drauf.
3. **Das 256-MiB-Fenster spielt beim Aktivierungs-Handover keine Rolle** (10-20
   KiB je Layer-Grenze bei bs=1, 5,3 KiB/Token KV je 3080 unter 4:1:1 — das
   Fenster bindet erst ab ~49k Token je Uebergabe). Es ist ausschliesslich eine
   Chunking-, keine Bandbreitenfrage, und nur fuer GB-Klasse-KV-Bewegung.
4. **Die Praemisse "Decode ist rechnergebunden" traegt nicht.** bs=1-Decode ist
   bandbreiten-/latenzgebunden; ein 4:1:1-Split ist das prefill-optimale
   Compute-Verhaeltnis, auf die falsche Phase angewandt. Und ein *Layer*-Split
   ist bei bs=1 strukturell das falsche Werkzeug: er addiert Stufenzeiten,
   waehrend TP Bandbreiten addiert.
5. **Der einzige Ort, an dem P2P dieses Rig strukturell veraendert**, ist der
   Kollektivboden im TP-Prefill (+20 bis +69 % geschaetzt, harte Decke +213 %)
   und die neue Tier-Sprosse *peer-VRAM* fuer KALTE KV-Posten. Fuer heisse
   KV-Seiten bleibt Peer-VRAM unbrauchbar (Rechnung in §5.3).

---

## 1. Die Anker (alles gemessen, Quelle in Klammern)

| Groesse | Wert | Quelle |
|---|---:|---|
| TP=3-Prefill-Fenster, FP8, 2048er-Chunk (kalt 21765) | **1837,6 ms = 1114 Tok/s** | #252 CollectiveClock |
| davon 5090: compute / wait | 196,6 / 1641,1 ms | #252 |
| davon 3080er: compute / wait | 586,5 / 558,5 · 1251,0 / 1279,3 ms | #252 |
| Wartezeit-Anteil je Rang | **~68 % des Fensters** | #252 |
| Shard-Rebalance 6,1,1 (A/B) | Fenster −7,6 %, Boden unbewegt (1188,0→1190,7), e2e **+8,2 %**, Decode −13,7 %, Kontext −47,9 % | #264 (verworfen) |
| Solo-5090-Prefill, Q3-GGUF, graph, TP=1-Lane | **283 ms/1k = 3534 Tok/s**; 2048er 583,1-583,7 ms = **3510 Tok/s** | #274 Slice B/C |
| Solo-5090-Decode, Q3-GGUF, graph, no-spec | **16,10-16,23 ms/Schritt = 61,6-62,1 Tok/s** | Lane-Spec R6/R7a |
| TP=3-Decode FP8+MTP, bs=1 | 91,92 Tok/s; Verify-Boden 33,65 ms; accept 3,13 | Baseline / Slice C |
| Decode-Leiter Q6-GGUF+MTP (code/prosa) | TP=2 88,38 / 72,22 → TP=3 **118,01 / 98,62** (+33,5 % / +36,6 %) | #73 K-Quant |
| Kollektiv-Budget im Decode (kritische GPU-Zeit) | **15,8 % (single) / 23,6 % (dual)** | #199-Verdikt |
| PP-Hop je Layer-Grenze @bs=1 | 142 us Nutzlast + **249 us Metadaten** (ungecachtes gloo-Pickle) | DESIGN_201 §1.3/§3.2 |
| PP=2 44/20, Q3, 9313er-Kontext | 44,2 Tok/s | #201 Slice-1-Smoke |
| Nesting-Kopplung (zwei Lanes, byte-geteilte Gewichte) | 4,2 % Decode / 4,3 % Prefill / 0 % KV | EVAL_272 |
| Nebenlaeufigkeit Verband+Lane, Kartenaequivalent E | Prefill-Lane 0,974 → **1,130**; Decode-Lane 0,914 → **1,440** | Slice C3 |
| Lane-Spec gefangen: Verify-Kosten | **~12,5 ms fix + 3,7 ms je Zeile**; K=1-Runde 24,77 ms | Runde 6/R7a |
| RDMA-Aperturbefund (naechster Verwandter) | BAR-Groesse bewegt **nichts**: rc5090 (32 GiB BAR) vs rc3080 (256 MiB BAR) ≤0,02 us Delta auf jeder Stufe | #277 W1 / #278 |

Modellgeometrie (aus `config.json`, nicht geschaetzt): 64 Layer, `full_attention_
interval 4` → **16 Full-Attention-Layer**, `num_key_value_heads 4`, `head_dim 256`
→ **KV = 32,0 KiB/Token bei fp8** (64,0 KiB bei bf16). Gegenprobe: gemessener
TP=3-Pool 883.584 Token × 32 KiB = 27,0 GiB — passt zu den protokollierten
8,4-10,5 GB freiem VRAM. `hidden_size 5120` → Aktivierung je Layer-Grenze
**10,0 KiB/Token** (bf16), mit Residual 20,0 KiB.

---

## 2. Szenario 1 — kurzer Prefill SOLO auf der 5090, Decode-Layer 4:1:1

### 2.1 Der Prefill selbst

| | heute | mit P2P | tragende Annahmen | was die Probe entscheidet |
|---|---:|---:|---|---|
| Solo-5090-Prefill 1k (Q3) | **3534 Tok/s** | **3534 Tok/s (±0)** | Ein TP=1-Prefill fuehrt kein einziges Kollektiv und keinen einzigen Karten-Transfer aus | nichts — es gibt keinen Pfad, auf den P2P wirken koennte |
| Solo-5090-Prefill 2048 (Q3) | **3510 Tok/s** | 3510 Tok/s (±0) | dito | — |
| Solo-5090-Prefill 2048 (Q4) | 3200-3500 Tok/s (geschaetzt) | ±0 | Prefill ist compute-gebunden; Q4_K vs Q3_K aendert nur die Dequant-Last | Q4-K-Quant-Kernel-Effizienz auf sm120 |
| **Referenz: TP=3-Prefill 2048** | **1114 Tok/s** | 1343-1884 Tok/s | §3 | Kollektiv-Transportpfad |

**Der Solo-Prefill gewinnt heute schon 3,1x gegen TP=3 — ohne P2P.** Der Grund
steht in #252: 68 % des TP-Fensters sind Kollektivkosten, und die restlichen 32 %
werden von den 3080ern getaktet (586,5 ms Compute gegen 196,6 ms auf der 5090).
Rechnet die 5090 dasselbe Fenster allein, braucht sie ~3 × 196,6 = 590 ms —
gemessen 583 ms. Die Rechnung schliesst sich auf 1,2 %.

### 2.2 Der Handover an die 3080-Decode-Layer — hier koennte P2P wirken

Nach einem Solo-Prefill liegt der gesamte KV auf der 5090; die Layer, die im
Decode auf den 3080ern liegen, brauchen ihren KV-Anteil dort. Unter 4:1:1 haelt
jede 3080 **1/6 der Layer → 5,33 KiB/Token** (fp8).

| Prefill-Laenge | KV je 3080 | heute (Host-Bounce, 6,5/13,4 GB/s D2H+H2D) | mit P2P (1 Hop) | Anteil am Prefill-Fenster |
|---:|---:|---:|---:|---:|
| 1024 Tok | 5,3 MiB | ~1,6-2,6 ms | ~0,8-1,3 ms | **0,3-0,5 %** |
| 2048 Tok | 10,7 MiB | ~3,2-5,2 ms | ~1,6-2,6 ms | 0,3-0,4 % |
| 32768 Tok | 171 MiB | ~51-84 ms | ~26-42 ms | 0,3-0,5 % |

**Verdikt: P2P aendert am kurzen Prefill inklusive Handover nichts Messbares
(<0,5 %).** Die Uebergabe skaliert mit derselben Konstanten wie der Prefill
selbst; der Anteil bleibt konstant klein.

**Aperturfrage, direkt beantwortet:** 5,3-171 MiB je Uebergabe liegen unter der
nominellen 256-MiB-Apertur. Das Fenster bindet erstmals bei
`256 MiB / 5,33 KiB = 49.152 Token` je einzelner Uebergabe — und selbst dann ist
es eine **Chunking**-, keine Bandbreitenfrage. Fuer die **Aktivierungen** an den
Layer-Grenzen (10-20 KiB bei bs=1, ~20 MiB bei 2048er-Prefill-Chunk) ist das
Fenster um 4-5 Groessenordnungen zu gross, um je eine Rolle zu spielen.
Relevant wird die Apertur ausschliesslich bei **KV-Migration und Gewichts-/
Spill-Bewegung in GB-Klasse** (§5).

### 2.3 Der 4:1:1-Decode-Layer-Split

Ein Layer-Split ist bei bs=1 **sequentiell**: `T = Σ_i f_i/BW_i + Hops`. Mit
`r = BW_5090/BW_3080 = 2,35`:

| Anordnung | Modell T/T_solo | Q3-GGUF Decode | Kommentar |
|---|---:|---:|---|
| solo 5090 | 1,000 | **61,9 Tok/s** (gemessen) | Referenz |
| 4:1:1 Layer-Split, 2 Hops | 0,667 + 0,333·2,35 + 2·0,391 ms/16,15 ms = **1,448** | ~42,8 Tok/s | −31 % gegen solo |
| 4:1:1 mit P2P | **1,436** | ~43,1 Tok/s | +0,8 % |
| 4:1:1 mit P2P + Formen-Cache | 1,404 | ~44,1 Tok/s | +3,0 % |

Tragende Annahmen: (a) Hop = 142 us Nutzlast + 249 us Metadaten (gemessen);
(b) P2P entfernt 60-80 % der Nutzlast-Etappe (Host-Staging + Sync), die reine
Drahtzeit fuer 10-20 KiB liegt bei 1-3 us — der Rest ist Fixkosten;
(c) **die 249 us Metadaten sind gepickelte gloo-Nachrichten ueber den Host und
werden von P2P ueberhaupt nicht beruehrt**. Der benannte, nicht gebaute
Formen-Cache ist damit der **groessere** Hebel als P2P.

**Kernaussage: der Preis des Layer-Splits ist der sequentielle Bandbreitenterm
(+45 %), nicht der Draht (+5 %). P2P adressiert die falschen 5 %.**

---

## 3. Szenario 2 — der AWQ-BF16-INT4-Referenzcheckpoint

### 3.1 Fit-Rechnung auf 32 GB (5090, 32607 MiB total)

Feste Posten (gemessene Anker): CUDA-/Prozess-Sockel ~1536 MiB · Graphen +
Aktivierungen ~1500 MiB · flashinfer-Workspace ~412 MiB · Prefill-Scratch
~700 MiB · GDN-State ~69 MiB je Session (3×23 MiB, Sizer-Anker aus dem
Baseline-Boot) · MTP/NEXTN-Kopf 1500-2684 MiB (der gemessene 2684-MiB-Komplement
ist vom `lm_head` 248320×5120 dominiert).
→ **Fixblock ~5,4 GiB mit MTP, ~3,9 GiB ohne.**

| Checkpoint | Gewichte | Fixblock | KV-Rest | Token @fp8-KV (32 KiB) | Verdikt |
|---|---:|---:|---:|---:|---|
| **Q3_K_M GGUF** | 12,87 GiB | 5,4 | **13,6 GiB** | **~445.000** | passt bequem |
| **Q4_K_M GGUF** | 15,66 GiB | 5,4 | **10,8 GiB** | **~354.000** | passt bequem |
| Q6_K GGUF | 20,98 GiB | 5,4 | 5,5 GiB | ~180.000 | passt noch |
| Qwen3.6-27B-FP8 | 23,3 GiB | 5,4 | 3,1 GiB | ~102.000 | gemessen **OOM** im Praxisboot (Slice B: 25 GiB + 4,7 GiB Rang-0-Floor + 3 GiB Lane-Posten > 32,6 GiB) |
| **AWQ-BF16-INT4** | **26,37 GiB** | 5,4 | **0,04 GiB** | **~1.300** | **passt NICHT** |
| AWQ ohne MTP | 26,37 GiB | 3,9 | 1,55 GiB | ~50.000 | physisch am Rand, kein brauchbarer Arbeitspunkt |

**Der Referenz-Checkpoint blockiert den Solo-Prefill-Pfad.** Das ist keine
Feinjustierung: 26,37 GiB Gewichte lassen auf 31,8 GiB nutzbarem VRAM keinen
Raum fuer Kopf + Graphen + KV. Fuer den Solo-Prefill ist **Q4_K_M die
naheliegende Wahl** (10,8 GiB KV-Rest = 354k Token auf einer Karte, mehr als das
halbe heutige Rig-Pool von 883k).

### 3.2 Prefill-tok/s fuer AWQ, kurzer Content

| Anordnung | heute | mit P2P | tragende Annahmen | was die Probe entscheidet |
|---|---:|---:|---|---|
| AWQ TP=3 uneven DCP (der reale Pfad) | **1050-1300 Tok/s** | **1250-1550 Tok/s** | Anker FP8 1155,9 Tok/s @1172 Tok; AWQ-marlin dequantisiert nach bf16 und umgeht den Triton-FP8-Pfad der 5090 (der 8 % der Prefill-Kernelzeit frisst) → eher am oberen Rand | ob P2P den Kollektivboden ueberhaupt anfasst |
| AWQ solo 5090 | **nicht lauffaehig** (§3.1) | nicht lauffaehig | Gewichte allein 26,37 GiB | — (Fit ist Arithmetik, keine Messfrage) |
| *hypothetisch*, wenn AWQ solo passte | 3500-5000 Tok/s | ±0 | Untergrenze = gemessene Q3-GGUF-Solozahl 3510; Obergrenze = 5090-Compute-Hochrechnung 196,6 ms / Shard-Anteil ~0,5 → 393 ms/2048 = 5210 Tok/s | Kernel-Effizienz marlin-Prefill auf sm120 |
| **Q4_K_M solo 5090 (der realisierbare Ersatz)** | **3200-3500 Tok/s** | ±0 | §2.1 | Q4-Dequant-Kosten |

**Verdikt fuer Frage 2: die belastbare Zahl fuer AWQ ist 1050-1300 Tok/s
(heute) bzw. 1250-1550 Tok/s (mit P2P) — der 3x-Sprung ist beim AWQ-Checkpoint
nicht erreichbar, sondern nur ueber einen Quant-Wechsel auf Q4-GGUF.**

### 3.3 Was P2P dem TP=3-Prefill tut — die eine Stelle, an der es wirklich zaehlt

Das Fenster ist `max(Rang-Compute) + Kollektivboden = 586,5 + 1251 = 1837,6 ms`.
Der Boden ist nach #264 gegen Shard-Umverteilung **immun** (1188,0 → 1190,7 ms
bei −7,6 % Fenster, e2e trotzdem +8,2 % — verworfen). Er ist damit die einzige
verbleibende Angriffsflaeche, und P2P ist der erste Kandidat dafuer.

| Annahme ueber den Boden | Fenster | Tok/s | Delta zu heute |
|---|---:|---:|---:|
| heute (Host-gestagte Kollektive) | 1837,6 ms | 1114 | — |
| P2P entfernt 25 % des Bodens | 1524,8 ms | 1343 | **+20,6 %** |
| P2P entfernt 60 % des Bodens | 1087,1 ms | 1884 | **+69,1 %** |
| Boden = 0 (physische Decke) | 586,5 ms | 3492 | +213 % |

Die 25-60-%-Spanne stuetzt sich auf den GDR-Befund, dass ohne Direktpfad
**~86 % der Kollektivkosten Host-Staging sind** (#266, cross-rig gemessen —
intra-rig ist der Anteil unbekannt und genau das die Probe-Frage). Die Decke
ist bemerkenswert: **perfektes P2P bringt den TP=3-Prefill exakt auf die
Solo-5090-Zahl**, weil dann die 3080er den Takt geben. Mehr ist mit dieser
Kartenmischung strukturell nicht drin.

---

## 4. Ehrliche Pruefung der Praemisse "rechnergebunden"

| Phase | Bindung | Beleg | Konsequenz fuer das Split-Verhaeltnis |
|---|---|---|---|
| Prefill | **compute-gebunden — Praemisse traegt** | 5090 196,6 ms vs 3080 586,5 ms Compute fuer aehnliche Shards; SM-saettigend (C3: zwei SM-saettigende Lasten koennen nicht beide voll laufen) | 4:1:1 ist ungefaehr das richtige Verhaeltnis — **fuer den Prefill** |
| Decode bs=1 | **bandbreiten-/latenzgebunden — Praemisse traegt NICHT** | TP=2→TP=3 misst +33,5 %, Bandbreitenaddition sagt +29,8 % voraus (4 pp); GDN-Kernel = 1,49-1,84 % der Decode-Wall; Kollektive 15,8 % | Verhaeltnis muss ∝ Bandbreite (2,35:1:1), nicht ∝ Compute (4:1:1) sein — **und der Mechanismus muss TP sein, nicht Layer-Split** |

Die 4:1:1-Zahl des Nutzers ist also das **prefill-optimale Verhaeltnis, auf die
Decode-Phase angewandt**. Unter einem Layer-Split ist sie doppelt ungeeignet:
falsches Verhaeltnis, und ein Mechanismus, der Stufenzeiten addiert statt
Bandbreiten (DESIGN_201 §3.3: "PP ist per Konstruktion exakt S-mal schlechter
als ideales TP"). Der Layer-Split verdient sich seinen Platz nur ueber
**Kapazitaet**, nie ueber Tempo.

---

## 5. Szenario 3 — Ausbreitung ueber die KV-Druck-Treppe (DESIGN_201 Erg. 9)

### 5.1 Die Treppe, mit Zahlen an jeder Sprosse

| Sprosse | Prefill | Decode | KV-Kapazitaet | Wirkung von P2P |
|---|---:|---:|---:|---|
| **0 — TP=1-Lane solo 5090** (Q3/Q4) | **3510 Tok/s** (gemessen) | **61,9 Tok/s** (gemessen, no-spec) | 445k (Q3) / 354k (Q4) Token | **0 %** — kein karten­uebergreifender Verkehr |
| **1 — TP=2** (5090 + x8-3080 = cuda:2) | ~1370-1580 Tok/s (Modell) | ~88 Tok/s (Modell ×1,42; gemessen Q6+MTP 88,38) | ~2/3 des TP=3-Pools | Boden kleiner (2 Raenge) → +15-45 % Prefill |
| **2 — TP=3 uneven DCP** | **1114 Tok/s** (gemessen) | ~115 Tok/s Modell; gemessen Q6+MTP **118,01**, FP8+MTP **91,92** | **883k Token** (gemessen) | **+20-69 % Prefill**, **+5-12 % Decode** |

Modellfaktoren fuer den Decode (Bandbreitenaddition, ideal): solo 1,00 / TP=2
1,42 / TP=3 1,85. Die gemessene Stufe TP=2→TP=3 bestaetigt das Modell auf 4
Prozentpunkte — **TP skaliert auf diesem Rig fast bandbreitenproportional, der
Kollektivboden frisst den Decode also NICHT**. Genau deshalb ist die
Decode-Aussicht von P2P klein: die Obergrenze ist das gemessene Kollektivbudget
von **15,8 % (bs=1) / 23,6 % (dual)** der kritischen GPU-Zeit, und P2P entfernt
davon nur den Host-Staging-Anteil → realistisch **+5-12 %**.

**Die Treppe wird mit P2P also unsymmetrisch:** der Aufstieg kostet weniger
Prefill-Tempo als heute (die Prefill-Strafe je Sprosse schrumpft), waehrend der
Decode-Gewinn je Sprosse praktisch unveraendert bleibt. Das macht das Aufsteigen
unter KV-Druck **billiger** und verschiebt die Hysterese-Schwelle aus Erg. 9
nach unten — ein echter, aber zweitrangiger Effekt.

### 5.2 "Was passiert, wenn allein die 5090 decoded"

| | Prefill | Decode bs=1 | KV-Pool |
|---|---:|---:|---:|
| TP=3 uneven DCP (heute) | 1114 Tok/s | 91,9-118 Tok/s | 883k Token |
| solo 5090 (Q3/Q4) | **3510 Tok/s (+215 %)** | **61,9 Tok/s (−34 bis −48 %)** | 445k / 354k Token |

**Solo ist eine Prefill-/TTFT-Maschine, der Verband eine Decode-Maschine.** Wer
allein die 5090 decoden laesst, kauft 3,1x Prefill fuer ~40 % Decode und die
halbe Kapazitaet. P2P verschiebt an diesem Tausch **nichts** (§2.1, §5.1
Sprosse 0).

### 5.3 Peer-VRAM als neue Tier-Sprosse — und ihre harte Grenze

P2P eroeffnet in der Tier-Leiter aus Erg. 8 (eigener VRAM → **peer-VRAM** →
Host-RAM → Remote-RAM) eine neue Sprosse. Zwei Rechnungen dazu:

**Wo sie zahlt.** Gemessen ist, dass der Verbands-KV vom knappsten Rang
dimensioniert wird, und das sind die 3080er — 1444 MiB, die der Speed-Dial auf
der 5090 freigibt, aendern `max_total_num_tokens` nicht (Slice C). Koennten die
3080-Raenge kalte KV-Seiten auf der 5090-Reserve parken (~1,4-3,3 GiB gemessen
frei), waeren das `2,8 GiB / 32 KiB ≈ +90k Token`, also **rund +10 % Rig-Pool**
— und die gute Richtung: geschrieben wird in die **5090 mit vollem 32-GiB-BAR**,
nicht durch das 256-MiB-Fenster.

**Wo sie NICHT zahlt.** Fuer *heisse* KV-Seiten ist Peer-VRAM unbrauchbar:
Attention liest je Schritt die ganze Historie. Bei 32k Kontext = 1 GiB KV; laegen
davon 10 % remote, kostet das je Decode-Schritt `100 MiB / 6-13 GB/s = 8-17 ms`
gegen einen 16-ms-Schritt. **P2P macht das Verschieben ~2x billiger, macht
entfernten KV aber nicht heiss lesbar.** Peer-VRAM bleibt damit strikt eine
Spill-/Session-Offload-Klasse (#134/#236), genau wie Host-RAM — nur ein Hop
naeher und ohne Host-RAM-Bandbreite zu verbrauchen (die #125-Experten-Streaming
und Spill ohnehin brauchen).

### 5.4 Richtungswahl — die eine konkrete Konsequenz der Fenster-Asymmetrie

Weil die 5090 volles BAR hat und die 3080er nur 256 MiB, ist jede
GB-Klasse-Bewegung als **"die kleine Karte zieht/schreibt in die grosse"** zu
formulieren, nicht umgekehrt. Der verwandte gemessene Befund stuetzt die
Richtungsfrage: beim RDMA-Pfad kostet der Switch-Pfad bei posted writes nur
+2,8 %, bei non-posted reads **+53 %** (#277) — Richtung ist auf diesem Rig
teurer als Groesse. Die Probe muss beide Richtungen je Paar getrennt vermessen.

---

## 6. "Optimal fuer jede Anwendung verschoben" — was das konkret heisst

Es gibt **keine** einzelne Aufteilung, die beide Phasen optimiert; die Optima
zeigen in entgegengesetzte Richtungen:

| Phase | Optimum | Warum | gemessener Beleg |
|---|---|---|---|
| Prefill (compute-, SM-gebunden) | **Konzentration auf die 5090, TP=1** | jeder zusaetzliche Rang bringt den 3080-Takt (586,5 ms) und den Kollektivboden (68 %) mit | 3510 gegen 1114 Tok/s |
| Decode bs=1 (bandbreitengebunden) | **TP ueber ALLE Karten**, Shard ∝ Bandbreite; nie Layer-Split | TP addiert Bandbreiten, PP addiert Zeiten | Leiter 62 / 88 / 118 Tok/s |

Die Konfiguration, die beides gleichzeitig liefert, ist damit **nicht ein
besseres Verhaeltnis, sondern zwei gleichzeitige Geometrien auf denselben Bytes**
— genau die Mehrfach-Gruppen-Runtime (#274/#272): eine TP=1-Prefill-/PD-Lane auf
der 5090 neben dem TP=3-Decode-Verband, Gewichte byte-geteilt ueber das Nesting.
Die Kosten sind beziffert und klein:

* Nesting-Kopplung: **4,2 % Decode / 4,3 % Prefill / 0 % KV** (EVAL_272);
* Nebenlaeufigkeitsgewinn: Kartenaequivalent **E 1,130 Prefill-Lane / 1,440
  Decode-Lane** gegen 0,974/0,914 seriell (Slice C3);
* Preis der geschuetzten Klasse unter echter Gleichzeitigkeit: **+9,7 %**,
  ursaechlich als SM-Konkurrenz belegt (`prefill_wait_ms` = 0,01 ms).

**P2P veraendert diese Empfehlung nicht.** Sie steht heute schon, ohne P2P, und
die Zahlen dafuer sind gemessen, nicht geschaetzt. P2P verbessert daran nur den
Verbands-Prefill (§3.3) — also genau den Pfad, den diese Anordnung ohnehin
umgeht.

---

## 7. Was ausschliesslich der Probe-Lauf entscheidet

1. **Laufen Allreduce/Broadcast unter diesem P2P ueberhaupt geraetedirekt?**
   Davon haengt der gesamte §3.3-Gewinn (+20 bis +69 % TP-Prefill) ab. NCCL hat
   auf diesem Rig das BAR nie genutzt; ob es das mit P2P tut, ist offen.
2. **Effektiv nutzbare Apertur je gerichtetem Paar** (nominal 256 MiB ist eine
   Obergrenze, nicht ein Versprechen). Fuer alles in dieser Analyse ausser
   GB-Klasse-KV-Migration ist die Antwort irrelevant — das ist selbst ein
   Ergebnis: **die Apertur darf die Bauentscheidungen nicht dominieren.**
3. **Kosten eines Fenster-Remaps** (us oder ms?). Entscheidet, ob
   Apertur-Chunking gegenueber Ablehnung>Apertur die richtige Default-Politik
   ist. Bei 40 Chunks je 10-GiB-KV-Migration ist selbst ein ms-Remap
   vernachlaessigbar.
4. **Wieviel der 142-us-Nutzlast-Etappe je Hop faellt wirklich weg?** Die
   249 us Metadaten fallen definitiv nicht weg — der Formen-Cache bleibt der
   groessere, unabhaengige Hebel.
5. **Richtungsasymmetrie je Paar** (§5.4), getrennt fuer lesend und schreibend,
   Root-Port- gegen Switch-Karte.

Nicht zu messen, weil arithmetisch entschieden: die Fit-Tabelle in §3.1 und die
Feststellung, dass ein Solo-Prefill keinen Transferpfad hat, auf den P2P wirken
koennte.

---

## Geltungsbereich

Gilt fuer **dieses Rig** (5090 + 2x 3080, PHB, GPU0 auf x4) und **dieses Modell**
(Qwen3.6-27B, 64 Layer, 16 Full-Attention-Layer, hidden 5120). Alle
P2P-Zahlen tragen den Vorbehalt aus DESIGN_201 P2P-PRAEZISIERUNG: Apertur und
Transportpfad sind bis zur Probe unbekannt, und nichts hier ist als Erwartung
formuliert, die die Probe bestaetigen soll. Die Nicht-P2P-Zahlen sind gemessen
und stehen unabhaengig davon. [[rig-ist-untergrenze]]: ein Rig mit NVLink oder
vollen Lanes verschiebt jede Kollektiv-Aussage — kein Urteil ueber P2P als
Konzept.
