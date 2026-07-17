# Qwen3.6-27B - Engine-Vergleichsmatrix (Final)

Stand: 2026-07-17. Alle Zahlen sind gemessen (Phase 3a-3c, M27a-M27e); keine
geschaetzten oder interpolierten Werte. Fehlende Zellen sind explizit als
INFEASIBLE (mit Kurzgrund) oder n/a markiert.

## 1. Kopf: Hardware, Modell, Engines, gemeinsame Settings

### Hardware (dieses System)
- 1x NVIDIA RTX 5090, 32 GB VRAM (NVML 32607 MiB)
- 2x NVIDIA RTX 3080, 20 GB VRAM je (NVML 20480 MiB)
- Eine der 3080 haengt an PCIe Gen4 x4 (schmalere Anbindung). Kein NVLink,
  kein GPU-P2P (GeForce, nvidia-smi -p2p zeigt NS/CNS fuer alle Paare -> PHB-only).
- NVML/PCI-Enumeration schwankt zwischen Boots/Treiberzustaenden; physische
  GPU-IDs werden zur Laufzeit ueber NVML aufgeloest (5090 nicht hart verdrahtet).
  Container-CUDA-Ordnung ist FASTEST_FIRST -> cuda:0 = 5090.

### Modell
Qwen3.6-27B-Familie (hybrid GDN / Linear-Attention + eingebetteter MTP/NEXTN-Draft,
blk.64). Quant-Varianten:
- Original Qwen3.6-27B-FP8 (layers-*.safetensors + mtp.safetensors)
- Qwen3.6-27B-AWQ-BF16-INT4 (compressed-tensors, ~14 GB)
- unsloth UD-GGUF Q6_K_XL (26.0 GB, + mmproj-BF16 Vision-Tower)
- unsloth UD-GGUF Q8_K_XL (35.8 GB) -- auf beiden Forks INFEASIBLE (siehe Fussnote 8)

### Engines (exakte Versionen / Images / Commits)
- llama.cpp: `ghcr.io/ggml-org/llama.cpp:server-cuda`, Upstream b10015 (id 297b3e6a71e1),
  Modell unsloth Qwen3.6-27B-MTP-GGUF (eingebetteter MTP-Draft).
- shvllm (vLLM-Fork, uneven-TP / rank-gpu-id): Image `shvllm-qwen35-gguf:cu129-uneven`
  (id 6ee0897f1157), Repo-HEAD f78ea433f, pip NCCL 2.30.7.
- htsglang (sglang-Fork, GGUF-Plugin + uneven-TP): Repo-HEAD 3e76cbbf1 (UNMODIFIED).
  TP=2/TP=3 als VM-lokale venv (torch 2.11+cu130, NCCL 2.28.9); TP=4 co-located als
  Docker-Image `htsglang-qwen35-gguf:cu130-3e76cbbf1` (NCCL 2.30.7 gebacken -- co-located
  TP=4 braucht NCCL >= 2.30, daher docker-only).

### Gemeinsame Settings (alle Zellen)
- MTP / Spekulatives Decoding UEBERALL AKTIV:
  - vLLM/shvllm: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  - sglang/htsglang: `--speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
  - llama.cpp: `--spec-type draft-mtp --spec-draft-n-max 3`
- KV-Cache-dtype fp8 (shvllm `--kv-cache-dtype fp8`, htsglang `fp8_e4m3`);
  llama.cpp hat KEIN fp8-KV -> naechster Analog q8_0 Block-Quant (Fussnote 1).
- UNCACHED per Konstruktion: unique-Nonce token-id-Prompts, cached_tokens == 0 in
  jeder Anfrage jeder Zelle (bei llama.cpp benigne Feld-Semantik, siehe JSON-Caveat).
- Thermal-Gate <= 80 C vor jedem Boot (cooldown.sh, feuerte bei jedem Boot).
- Ein aufgewaermter Battery-Durchlauf pro Boot (kein Median-of-3); maxKV bei
  llama.cpp per Bisektion, bei den Forks aus der Boot-Zeile.

### Metrik-Definitionen (Spalten)
- **Prefill20k** (P1): 1 Request, 20000 exakte unique input_ids, 1 Output-Token -> tok/s.
- **Dec1k code / prose** (D1): 1 Request, ~1000-Token-Decode (ignore_eos) -> tok/s.
- **Par8-Prefill** (P8): 8 parallele Requests x 6000 input_ids -> aggregierte tok/s.
- **Par8-Dec code / prose** (D8): 8 parallele Requests x ~400-Token-Decode -> aggregierte tok/s.
- **maxKV**: groesster bootbarer+servender KV-Pool an der Zell-Config (config-gebunden,
  siehe Kontext-Sektion 3 zur Abgrenzung gegen den kalibrierten Maximalkontext).
- **Accept**: MTP/NEXTN spec_accept_length (D1 code / D1 prose). Nur sglang/htsglang
  exponiert das; vLLM- und llama.cpp-Oberflaechen nicht -> dort n/a. Der MTP-Speedup
  steckt bei allen Engines bereits in den tok/s.

---

## 2. Haupttabellen pro Szenario

Alle tok/s in Tokens/Sekunde, maxKV in Tokens.

### Szenario S1 -- Layer-Split (3 GPUs, 5090 + 2x 3080, 72 GB)

Nur llama.cpp: echte TP kann die 3 heterogenen Karten nicht spannen, daher
`-sm layer` (Pipeline ueber alle 3 GPUs). Die Forks fahren dieses Szenario nicht.

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| llama.cpp Q8 (UD-Q8_K_XL) | 1835.4 | 57.9 | 45.0 | 1914.4 | 150.1 | 156.8 | 355568 | n/a |
| llama.cpp Q6 (UD-Q6_K_XL) | 1688.9 | 74.5 | 50.8 | 1741.5 | 176.3 | 154.6 | 518096 | n/a |

### Szenario S2 -- TP=2 (2x RTX 3080, 5090 nicht beteiligt)

Echte Tensor-Parallelitaet ueber die beiden 3080. htsglang TP=2 wurde bei
PHYSISCH ENTFERNTER 5090 gemessen (reine 2x3080), shvllm mit CUDA_VISIBLE_DEVICES=0,2,
llama.cpp mit `-sm tensor` (das geplante `-sm row` ist hier INFEASIBLE, Fussnote 2).

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| llama.cpp Q6 (-sm tensor) | 1089.8 | 79.6 | 63.5 | 991.2 | 149.5 | 130.8 | 266856 | n/a |
| shvllm Q6 (GGUF) | 1128.1 | 44.6 | 36.3 | 1092.5 | 70.7 | 65.0 | 25600 | n/a |
| shvllm AWQ-INT4 | 1153.6 | 75.5 | 62.4 | 1121.9 | 195.0 | 187.8 | 94400 | n/a |
| htsglang Q6 (GGUF) | 1126.1 | 54.5 | 48.7 | 1098.7 | 65.5 | 49.8 | 32768 | 3.36 / 3.04 |
| htsglang AWQ-INT4 | 1169.2 | 86.5 | 63.9 | 1157.2 | 136.5 | 119.7 | 148864 | 3.22 / 2.38 |
| Q8 (jede Engine) | INFEASIBLE: passt nicht auf 2x20 GB (Q8=35.8 GB, per Plan ausgeschlossen); zusaetzlich GGUF-Q8-Loaderbug (Fussnote 8) | | | | | | | |

Hinweis: Bei htsglang TP=2 ist die Nebenlaeufigkeit hart auf 2 gedeckelt (Mamba-State-Cache
ist der Engpass, nicht KV-Tokens) -> die Par8-Spalten sind Durchsatz bei 2-at-a-time
gequeuten Requests, KEIN 8-Wege-Batching (Fussnote 4). shvllm laesst echtes 8-Wege-Batching
zu (D8 ~2.6x D1 bei AWQ). maxKV-Effizienz Q6: shvllm serviert 25600 @ GMU 0.88 unaided,
htsglang braucht manuelles KV-Cap 32768 (Fussnote 7 / #63).

### Szenario S4 -- TP=3 uneven-auto (5090 + 2x 3080, gewichtetes uneven-DCP)

VRAM-gewichtete uneven-TP-Aufteilung, rank0 = 5090. htsglang mit BEIDEN Varianten:
- **V1 = max-KV** (`--rank-tp-ratio auto`, kalibrierter Token-Vektor, reiner Pool-Maximum).
- **V2 = max-perf @ >= 100k KV** (`auto-performance` mit MLP-Konzentration auf die 5090;
  GGUF nutzt eine PINNED-MLP-Approximation, da auto-performance auf GGUF INFEASIBLE ist, Fussnote 5).

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| shvllm Q6 (GGUF) | 1239.8 | 61.7 | 46.2 | 1215.2 | 180.3 | 155.3 | 1139318 | n/a |
| shvllm AWQ-INT4 | 1228.3 | 83.7 | 62.7 | 1211.6 | 288.1 | 266.2 | 1146573 | n/a |
| shvllm FP8 | 1218.7 | 73.9 | 60.7 | 1223.5 | 287.9 | 273.7 | 1046126 | n/a |
| shvllm Q8 (GGUF) | INFEASIBLE: UD-Q8_K_XL mixed-precision fused qkvz, GGUF-Plugin lehnt fruh ab (Fussnote 8) | | | | | | | |
| htsglang FP8 -- V1 max-KV | 1123.7 | 98.4 | 80.8 | 1125.7 | 354.9 | 268.9 | 530944 | 3.28 / 2.69 |
| htsglang FP8 -- V2 max-perf | 1202.7 | 84.4 | 61.3 | 1207.8 | 343.2 | 260.2 | 299968 | 3.12 / 2.25 |
| htsglang AWQ -- V1 max-KV | 1123.9 | 103.2 | 93.1 | 1135.2 | 346.0 | 299.2 | 566912 | 3.14 / 2.86 |
| htsglang AWQ -- V2 max-perf | 1247.2 | 115.7 | 94.8 | 1261.6 | 345.3 | 305.2 | 441536 | 3.40 / 2.79 |
| htsglang Q6 -- V1 max-KV | 1102.7 | 65.9 | 54.6 | 1111.5 | 184.0 | 157.0 | 546560 | 3.19 / 2.66 |
| htsglang Q6 -- V2 max-perf (pinned-MLP) | 1207.5 | 63.2 | 54.2 | 1213.4 | 221.7 | 164.1 | 241216 | 2.99 / 2.58 |
| htsglang Q8 -- V1/V2 | INFEASIBLE: laedt weiter als shvllm, crasht dann an mixed-dtype padding (Fussnote 8) | | | | | | | |

Hinweis: Hier ist die Nebenlaeufigkeit NICHT mamba-gedeckelt (5090 im Mix, ~100 Mamba-Slots)
-> die htsglang-Par8-Spalten sind echtes 8-Wege-Batching (D8 >> D1). Die V2-Wirkung
haengt am Quant (Fussnote 5): AWQ V2 = echter Doppel-Gewinn (Prefill +11% UND Single-Decode
+12%), FP8 V2 = Prefill-Gewinn aber Single-Decode-Einbruch (Decode-Knee), GGUF V2 = nur
Pinned-MLP-Approximation.

### Szenario S3 -- TP=4 co-located (5090 x2 + 2x 3080, via MPS)

4 gleich grosse Ranks auf 3 physischen GPUs; die beiden Ranks 0+1 teilen sich die 5090
ueber NVIDIA MPS. Absolutes Per-Rank-Budget (kein uneven-Ratio, kein DCP -> V1/V2 n/a).
shvllm @ 14500 MiB/Rank, htsglang @ 13500 MiB/Rank (Engine-Differenz, Fussnote 3).

| Engine x Quant | Prefill20k | Dec1k code | Dec1k prose | Par8-Prefill | Par8-Dec code | Par8-Dec prose | maxKV | Accept c/p |
|---|---|---|---|---|---|---|---|---|
| shvllm FP8 (@14500) | 1411.9 | 107.3 | 82.2 | 1442.9 | 323.4 | 308.0 | 429096 | n/a |
| shvllm AWQ-INT4 (@14500) | 1417.5 | 103.2 | 72.8 | 1443.7 | 306.9 | 296.9 | 514036 | n/a |
| shvllm Q6 (GGUF, @14500) | 1384.0 | 62.6 | 52.2 | 1424.4 | 220.7 | 175.0 | 443740 | n/a |
| shvllm Q8 (GGUF) | INFEASIBLE: identisch zu shvllm-Q8-TP=3 (mixed-precision fused qkvz, Fussnote 8) | | | | | | | |
| htsglang FP8 (@13500) | 1326.3 | 92.1 | 91.5 | 1341.4 | 244.6 | 244.6 | 330818 | 2.79 / 2.75 |
| htsglang AWQ-INT4 (@13500) | 1330.7 | 115.3 | 103.5 | 1351.9 | 259.0 | 216.2 | 345064 | 3.18 / 2.84 |
| htsglang Q6 (GGUF, @13500) | 1282.3 | 76.4 | 65.6 | 1311.9 | 166.0 | 164.3 | 322174 | 3.45 / 2.98 |
| htsglang Q8 (GGUF) | INFEASIBLE: kein Boot versucht, bekannter Loaderbug (Fussnote 8) | | | | | | | |

Anomalie (offen, an Main verwiesen): Bei shvllm TP=4 schlaegt FP8 die AWQ-INT4 im
Single-Decode (107.3 vs 103.2), bei htsglang ist es umgekehrt (AWQ 115.3 vs FP8 92.1) --
Engine-abhaengige INT4-Dequant-/fp8-Pfad-Kosten.

---

## 3. Kontext-Sektion: config-gebundene maxKV vs kalibrierte Maximalkontexte

Die maxKV-Werte der Tabellen oben sind CONFIG-GEBUNDENE Pool-Groessen an der jeweiligen
Benchmark-Config (Kontextlaenge, Reserve, MTP aktiv). Sie duerfen NICHT direkt als
"maximaler Kontext der Engine" gelesen werden -- und schon gar nicht 1:1 zwischen den
Engines verglichen werden, weil sie aus unterschiedlichen Boot-Zeilen mit unterschiedlicher
Semantik stammen:

- **shvllm-maxKV** = vLLM-Boot-Zeile "GPU KV cache size: N tokens".
- **htsglang-maxKV** = sglang-Boot-Zeile "max_total_num_tokens".
- **llama.cpp-maxKV** = groesstes servendes `-c` via Bisektion.

Das ist der Grund, warum shvllm TP=3 (>1M) und htsglang TP=3 (~530k) trotz gleicher
MTP-Aktivierung so weit auseinanderliegen: verschiedene Engines zaehlen den DCP-Pool
unterschiedlich, KEIN Aequivalenz-Beweis (Aepfel/Birnen).

### Kalibrierte Maximalkontexte (aus HANDOFF, was jeweils gemessen wurde)

**htsglang uneven-DCP-Kalibrierung (single-node, VRAM-gewichtet, konvergierte Token-Vektoren):**
Das sind die tatsaechlichen KV-Ceilings des Forks pro Quant, per Selbst-Kalibrierung
ermittelt -- getrennt nach MTP aus/an:

| Kontext-Klasse | Wert (Tokens) | Was gemessen |
|---|---|---|
| AWQ, no-MTP | 886080 | konvergierter uneven-DCP-Pool ohne Draft-KV (Vektor [31,15,18]) |
| GGUF-Q6, no-MTP | 840896 | konvergierter uneven-DCP-Pool ohne Draft-KV |
| FP8, no-MTP | 804416 | konvergierter uneven-DCP-Pool ohne Draft-KV (Vektor [32,15,17]) |
| FP8, MTP aktiv | 530944 | mit Draft-KV; stabil servend (RESERVE 3000,2200,2200 + TOKVEC 33,13,18) |
| GGUF-Q6, MTP aktiv | 532224 | mit Draft-KV; stabil servend (RESERVE 3000,2200,2200) |

Der Sprung no-MTP -> MTP (z.B. FP8 804416 -> 530944, -34%) ist FUNDAMENTAL, kein Bug:
der eingebettete Draft belegt eigenes KV-/Mamba-State-Budget. Die MTP-Zeilen der TP=3-V1-Tabelle
(FP8 530944, Q6 546560) reproduzieren genau diese MTP-aktive Klasse.

**shvllm TP=3 (aus den Boot-Logs, MTP aktiv):** >1M-Klasse -- FP8 1046126, Q6 1139318,
AWQ 1146573. Das ist der maximierte DCP-Pool, den vLLM in seiner Boot-Zeile ausweist;
er ist wegen der abweichenden Zaehl-Semantik NICHT direkt gegen die htsglang-max_total_num_tokens
aufzurechnen.

Fazit der Sektion: Innerhalb einer Engine sind maxKV-Werte vergleichbar; ueber Engines
hinweg nur qualitativ. Die kalibrierten htsglang-Zahlen (886080 / 840896 / 804416) zeigen
die reine Fork-Kapazitaet OHNE MTP; die Matrix-Zellen zeigen den servenden Pool MIT MTP.

---

## 4. Fussnoten / Caveats (M27a-M27e)

1. **llama.cpp q8_0-KV statt fp8** (M27a): llama.cpp hat kein fp8-KV. Alle Zellen nutzen
   8-bit BLOCK-Quant KV (`-ctk/-ctv q8_0`, + Draft `-ctkd/-ctvd q8_0`, `-fa on`) als naechsten
   Analog zu `--kv-cache-dtype fp8` der anderen Engines. q8_0 (Block-Quant) != fp8 (per-Tensor).

2. **llama.cpp -sm tensor CPU-Sampling + -sm row INFEASIBLE** (M27a): Das urspruenglich geplante
   `-sm row` scheitert am Modell-Load ("device CUDA0 does not support split buffers" -- braucht
   P2P/Split-Buffer, auf diesen GeForce-Karten nicht vorhanden). Ersetzt durch `-sm tensor`
   (echte TP). Dabei loggt llama.cpp "backend sampling not supported with SPLIT_MODE_TENSOR;
   using CPU" -> Sampling laeuft auf der CPU (Limitierung, kein Fehler). NCCL braucht
   `--ipc=host --shm-size + NCCL_P2P_DISABLE=1 + NCCL_NVLS_ENABLE=0`.

3. **RANK_MIB 13500 statt 14500 (htsglang TP=4)** (M27e F1): 14500 MiB/Rank OOMt bei der
   Draft-CUDA-Graph-Capture auf der 5090 (beide co-located Ranks erreichen ~15.5 GiB = Budget
   + ~1 GiB non-KV-Overhead). 13500 bootet sauber. shvllm passt bei 14500 (vLLMs MiB-Budget
   deckt Gesamt-Prozessnutzung inkl. Graphs) -- Engine-Differenz, relevant fuer jeden
   Cross-Engine-maxKV-Vergleich in S3.

4. **htsglang TP=2 Mamba-Concurrency-Deckel = 2 + manuelles Q6-Tuning** (M27c): Auf 2x20 GB
   ist der Mamba/Linear-Attn-State-Cache (73-75 MiB/Slot/Rank) der Engpass, nicht KV -> sglang
   reduziert max_running_requests automatisch 16 -> 2. Alle TP=2-Par8-Werte sind daher 2-at-a-time
   gequeuter Durchsatz, kein 8-Wege-Batching. Beide Quants brauchten `--mem-fraction-static 0.90`
   (Default 0.749 -> Mamba-Pool 0 Slots -> Boot-RuntimeError). Q6 zusaetzlich manuelles KV-Cap
   32768 (bei 0.90 uncapped bootet 102328, OOMt aber beim ersten Forward im GGUF-Dequant-Scratch).

5. **V2-Quant-Abhaengigkeit** (M27d): AWQ V2 = echter STRICT WIN (Prefill +11%, Single-Decode +12%,
   maxKV 441536) -- AWQ hat mehr MLP-Units (544), die 5090 absorbiert die Konzentration ohne
   Decode-Knee. FP8 V2 = Prefill +7% ABER Single-Decode-Drop (D1c 84.4 vs V1 98.4, Decode-Knee),
   maxKV 299968. GGUF V2 = auto-performance INFEASIBLE (uneven_perf.py:531 open(model_path/'config.json')
   -> NotADirectoryError, weil model_path bei GGUF die .gguf-DATEI ist); daher Pinned-MLP-Approximation
   (`--rank-mlp-ratio 5,1,1`), kein Decode-Knee-Guard.

6. **MPS-Neustart-Lektion** (M27b/M27e): Der dokumentierte Check `ls /tmp/nvidia-mps/` ist ein
   FALSE POSITIVE -- veraltete Sockets ueberleben den Daemon-Tod. Der MPS-Daemon war tatsaechlich
   TOT; co-located Ranks time-slicen dann nur. Liveness NUR ueber die Control-Daemon-Antwort
   pruefen (`echo get_default_active_thread_percentage | nvidia-cuda-mps-control` -> 100.0),
   nie ueber ls. Eine erste shvllm-tp4-Battery ohne live MPS wurde verworfen und neu gefahren.

7. **Q6-TP=2-Speichereffizienz-Frage (#63)**: shvllm serviert q6_tp2 mit maxKV 25600 @ GMU 0.88
   ohne Sondertuning; htsglang braucht bei gleicher HW manuelles KV-Cap (32768) + mem-fraction 0.90,
   und selbst der natuerliche Pool (102328) servt nicht. Ursachenverdacht: GGUF-On-the-fly-Dequant-Scratch,
   Mamba-Slots, mmproj-Last. Offenes Untersuchungs-Item an Main.

8. **Q8 auf beiden Forks INFEASIBLE (unterschiedliche Fehlertiefen)**: Gemeinsame Root-Cause --
   unsloth UD-Q8_K_XL speichert den fused GDN `in_proj_qkvz` in MIXED precision (fp16 + uint8).
   - shvllm lehnt FRUEH ab: `ValueError: Detected some but not all shards of ...in_proj_qkvz are
     quantized` (is_layer_skipped_gguf fused-shard-Check), identisch bei TP=3 und TP=4.
   - htsglang laedt WEITER (past shard-check), crasht dann bei model init:
     `AssertionError: Data container has mixed dtypes: {torch.float16, torch.uint8}`
     (gguf.py:475 _create_padded_weight_param). Q6_K_XL laedt in beiden Plugins sauber ->
     Q8_K_XL-spezifisches Layout. Fix (mixed-dtype GGUF-Layer) an Main als TODO (#64), ausserhalb
     Messumfang. Q8 wird daher nur von llama.cpp getragen (S1 Layer-Split).

---

## 5. Fazit (ehrlich, pro Disziplin)

**Single-Stream-Decode (D1):** htsglang gewinnt klar, getrieben von MTP/NEXTN + immer-an
Decode-Graphs. Spitzenwerte: AWQ V2 TP=3 (D1c 115.7) und AWQ TP=4 co-located (D1c 115.3);
AWQ V1 TP=3 (103.2). Die vLLM-Seite (shvllm) toppt nur im TP=4-FP8-Fall (107.3). llama.cpp
liegt strukturell darunter (bestes Q6 TP=2 -sm tensor: 79.6). -> **htsglang holt die
Single-Decode-Krone via MTP + Graphs.**

**Par8 / 8-Wege-Durchsatz (D8):** shvllm dominiert dort, wo es echtes 8-Wege-Batching zulaesst:
TP=3 AWQ 288.1 und FP8 287.9 code. htsglang TP=3 ist gleichauf im code-Peak (FP8 V1 354.9 --
hier sogar hoeher) aber inhomogener; entscheidend ist, dass htsglang TP=2 hart mamba-gedeckelt
ist (D8c nur 65.5-136.5). -> **Bei genuinem 8-Wege-Batching liefern shvllm-TP=3 und htsglang-TP=3
die hoechsten Aggregate; htsglang-TP=2 faellt wegen Mamba-Cap zurueck.**

**Prefill (P1/P8):** TP=4 co-located gewinnt eindeutig -- shvllm FP8/AWQ liegen bei 1411-1443 tok/s
(P1/P8), htsglang TP=4 bei 1282-1352. Zwei Ranks auf der 5090 + je einer auf den 3080 maximieren
den Prefill-Compute. Layer-Split (llama.cpp S1) ist im Single-Prefill nominal hoch (Q8 1835,
Q6 1689), aber das ist server-native Timing (ohne HTTP) und nicht 1:1 mit den wall-clock-Zahlen
der Forks vergleichbar. -> **TP=4 co-located = Prefill-Krone (unter den vergleichbaren
wall-clock-Messungen).**

**Max Kontext (config-maxKV):** shvllm haelt bei TP=3 uneven-auto die groessten servenden Pools
(>1M: AWQ 1146573, Q6 1139318, FP8 1046126). htsglang-TP=3 liegt per Boot-Zeile bei ~530-567k
(MTP-aktive Klasse), was aber wegen abweichender Zaehl-Semantik NICHT als "halb so viel Kontext"
gelesen werden darf. -> **shvllm haelt die groessten config-maxKV bei TP=3.**

### Die 2-3 klarsten Gesamtaussagen
1. **htsglang gewinnt Single-Stream-Decode** (MTP/NEXTN + Decode-Graphs), am staerksten mit AWQ.
2. **shvllm haelt die groessten config-maxKV bei TP=3 uneven-auto** (>1M-Klasse) und liefert
   das robusteste echte 8-Wege-Batching (TP=3 AWQ/FP8 ~288 D8c).
3. **TP=4 co-located ist die Prefill-Krone** (shvllm FP8/AWQ ~1410-1443 tok/s), erkauft mit
   MPS-Aufwand und knapperem KV-Budget (13500/14500 MiB/Rank).
4. **llama.cpp** ist der pragmatische 3-Karten-Allrounder (einzige Q8-faehige Engine hier, hoher
   Layer-Split-Prefill), verliert aber im Decode und muss q8_0-KV statt fp8 nutzen; echtes TP
   ist auf diesen GeForce-Karten nur ueber `-sm tensor` (mit CPU-Sampling) moeglich, nicht `-sm row`.

---

*Verifikation: Alle uebertragenen Zahlen wurden gegen matrix_results/*.json (llamacpp,
shvllm, htsglang_tp2/tp3/tp4) rueckgeprueft; kalibrierte Kontextwerte gegen HANDOFF.md
(Zeilen 620/623/1213/1242/1304). Stichproben bestaetigt u.a.: shvllm awq_tp3 D8c 288.1,
htsglang awq_tp3_V2 D1c 115.7, htsglang fp8_tp4 maxKV 330818, llama.cpp q6_layersplit
maxKV 518096, shvllm fp8_tp4 D1c 107.3.*
