# EVAL: FP8-Quantisierung des DFLASH-Drafters (Qwen3.6-27B)

Schreibtisch-Recherche, keine GPU, kein Boot. Stand 2026-07-29.
Suchtiefe: 6 Web-Suchen + 5 Seiten-Fetches (HF-Repos, arXiv, vLLM-Blog),
plus lokale Checkpoint-Header/Configs und Fork-Code (`dflash.py`,
`fp8_dequant_gemv.py`, `server_args.py`), plus R7b-Messtabelle in
`docs/dev/INTEGRATION_R3_VALIDATION.md`.

---

## 1. Existiert ein FP8-DFLASH bereits?

**Nein — kein oeffentlicher FP8/compressed-tensors-DFLASH-Drafter gefunden.**
Gesucht wurde nach `Qwen3.6-27B-DFlash` + FP8 / W8A8 / compressed-tensors /
llm-compressor / speculators. Treffer sind ausschliesslich:

* das BF16-Original [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash),
* FP8-Varianten des **Ziels** (`Qwen/Qwen3.6-27B-FP8`, blockweise fp8 128,
  `vrfai/Qwen3.6-27B-FP8` per llm-compressor) — nicht des Drafters,
* GGUF-Quantisierungen des Drafters (siehe unten).

Der vLLM-Blog zu DFlash + llm-compressor
([Laguna XS.2, 2026-05-28](https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor))
quantisiert ebenfalls nur das Ziel (FP8/NVFP4/INT4/INT8), der 0.6B-DFlash-Kopf
bleibt dort unquantisiert. Es gibt also weder Checkpoint noch publizierte
Rezeptur fuer einen FP8-DFLASH.

### Die GGUF-Variante, die der Nutzer meint

Mehrere Community-Repos stagen llama.cpp-Quants des z-lab-Drafters fuer die
`dflash-draft`-Architektur:

| Quant | Groesse | Community-Notiz |
|---|---|---|
| F16 | 3,47 GB | Baseline |
| Q8_0 | 1,75-1,85 GB | "matches F16 acceptance", Accept 36-45 %, minimal schneller als F16 |
| Q6_K | 1,43 GB | "der Vollstaendigkeit halber" |
| Q5_K_M | 1,23 GB | — |
| Q4_K_M | 1,03 GB | Accept 28-29 %, **-17 Punkte gegen F16** |
| IQ4_NL | 0,99 GB | — |
| IQ4_XS | 0,93 GB | von einem Repo als bester Durchsatz-Kompromiss empfohlen (+12,8 %) |

Quellen: [spiritbuun](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF),
[Ardenzard](https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF),
[Anbeeld](https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF),
[Lucebox](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF),
[Alittlehammmer](https://huggingface.co/Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp).

**Der zentrale Community-Befund:** der 3.6er Drafter hat kausale SWA-Schichten
(`layer_types` `[S,S,S,S,F]`, Fenster 2048 — im lokalen `config.json`
bestaetigt), und genau die sind Q4-fragil. Q8_0 ist der kleinste Quant, der
F16-Qualitaet haelt.

### Lokaler Bestand

`/spinning/llm_stuff/club-3090/models-cache/`:

| Verzeichnis | `model.safetensors` | dtype |
|---|---|---|
| `qwen3.6-27b-dflash` | 3.460.432.504 B = 3300,3 MiB | `bfloat16` |
| `qwen3.6-35b-a3b-dflash` | 771.819.674 B = 736,1 MiB | bf16 |
| `qwen3.5-122b-a10b-dflash` | 1.547.794.655 B = 1476,1 MiB | bf16 |
| `gemma-4-31b-it-dflash` | 3.071.941.240 B = 2929,6 MiB | bf16 |
| `gemma-4-26b-a4b-it-dflash` | 859.384.328 B = 819,6 MiB | bf16 |

Alle fuenf sind BF16, keine Quant-Metadaten, kein `quantization_config` in
irgendeiner `config.json`. **Lokal existiert kein quantisierter DFLASH — auch
kein GGUF.** Die GGUF-Variante, von der der Nutzer spricht, ist ein
HF-Community-Artefakt, das hier noch nicht liegt.

---

## 2. Quantisierungsweg BF16 -> FP8

### Format: per-channel dynamic FP8 (W8A8-dynamic, `strategy: channel`)

Nicht blockweise 128x128. Begruendung aus unserem eigenen Kernel-Bestand:

* Die **per-channel**-Variante ist die, die #189/#192 auf Karten ohne natives
  fp8-GEMM bedient (`fp8_dequant_gemv.py`, per-channel-Branch: Skala haengt nur
  von `n` ab, verlaesst die k-Schleife, keine ragged decline). Der
  Block-Branch (#179) ist die schwaechere Seite auf sm86.
* Der Routing-Pfad auf sm86 ist gemessen dokumentiert: ein w8a8-Config, dessen
  `min_capability` (89) nicht erfuellt ist, geht ueber `_get_scheme` nach
  `CompressedTensorsW8A16Fp8` und dort in den Dequant-Branch — genau die
  Stelle, an der unser Kernel haengt.
* Die #255-Tunings sind Triton-**Block**-GEMM-Configs fuer die 27B-MLP-Shapes
  bei mlp-ratio [2,1,1]. Sie decken die DFLASH-Kopf-Shapes nicht ab; per-channel
  umgeht diese Luecke.

Werkzeug: `llm-compressor` `model_free_ptq` bzw. das Standard-FP8-W8A8-Rezept,
Ausgabe in `compressed-tensors`
([Doku](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/)).
Dynamic-Activation heisst: keine Kalibrierdaten noetig, ein reiner
Gewichts-Pass. Der Drafter hat weder `embed_tokens` noch `lm_head` im
Checkpoint (leiht beides vom Ziel), also gibt es dort nichts auszunehmen.

### Zwei Code-Vorbedingungen im Fork (beide klein, beide notwendig)

1. **`fc` ist ein blankes `nn.Linear`.** `python/sglang/srt/models/dflash.py:367`:
   `self.fc = nn.Linear(num_context_features * hidden_size, hidden_size, bias=False)`
   — ohne `quant_config`. Alle anderen Linears (`qkv_proj`, `o_proj`,
   `gate_up_proj`, `down_proj`, Laguna-`g_proj`) sind quant-config-faehig.
   Folge: entweder `fc` im Rezept auf die `ignore`-Liste (bleibt BF16, 250 MiB)
   oder `fc` auf `ReplicatedLinear` umstellen (3 Zeilen, spart weitere 125 MiB).
   Ohne eine der beiden Massnahmen scheitert das Laden, weil
   `default_weight_loader` einen fp8-Tensor in einen bf16-Parameter schreiben
   soll.
2. **Der Schalter existiert bereits**: `speculative_draft_model_quantization`
   (`server_args.py:1971`) — Default ist die Quantisierung des Ziels, `unquant`
   schaltet ab. Kein neuer Flag noetig.

### Kostenlose Alternative, die zuerst zu pruefen ist

Ein **Q8_0-GGUF-Drafter** ist 1,75 GB — praktisch dieselbe Groesse, die FP8
erreicht — existiert bereits, ist von der Community gegen F16 gemessen, und
kostet keinen Quantisierungslauf. Das R7b-Verdikt haelt ausdruecklich fest,
dass die alte Drafter-Lane-GGUF-Sperre fuer DFLASH **nicht** gilt. Risiko:
llama.cpp-Tensornamen der `dflash-draft`-Arch muessen auf unseren Loader
abgebildet werden; das ist der eigentliche Integrationsaufwand und kann groesser
sein als der FP8-Lauf.

---

## 3. Erwarteter Accept-Verlust

### Externe Evidenz

| Quelle | Aussage |
|---|---|
| [Speculative Decoding Meets Quantization (arXiv 2505.22179)](https://arxiv.org/pdf/2505.22179) | W8A8 und W4A16 zeigen **nahezu keine** Degradation der mittleren Accept-Laenge gegen FP16; erst W4A4 bricht ein |
| [EAGLE-3 on AMD Instinct (vLLM-Blog 2026-07-13)](https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct) | Kimi-K2.5: BF16-Draft 1,69-1,90x, **FP8-Draft (Quark, row-wise scaled FP8 GEMM) 1,76-2,00x** — der FP8-Drafter ist netto schneller, der Accept-Verlust also kleiner als der Rechengewinn |
| [Quantize the Target, Quantize the Drafter (arXiv 2607.04244)](https://arxiv.org/abs/2607.04244) | Wettbewerbsbericht Qwen3.5-4B auf A10G: Block-Diffusion-Drafter + Quantisierung + SWA, 6,978x Gesamt-Speedup; FP8 als der ausgewogene Punkt, aggressives int4 degradiert messbar. Konkrete Per-Format-Tabellen aus dem PDF nicht extrahierbar |
| GGUF-Community fuer **genau diesen** Drafter | Q8_0 haelt F16-Accept (36-45 %); Q4_K_M faellt auf 28-29 % |

### Lokale Evidenz

* **R7b-Q6_K-Gegenboot**: der NEXTN-Kopf komplett in Q6_K statt Q3_K/Q4_K ergab
  dasselbe Accept-Band — Kopf-Quantisierung war messbar nicht der Engpass. Der
  Engpass lag auf Position 0 (24-45 %) und die R7b-Bilanz schliesst
  Kopf-Quantisierung explizit als eine von vier gepruueften Ursachen aus.
* **Grenze dieser Evidenz** (Stichprobenbreite): der NEXTN-Kopf ist **eine**
  Schicht ohne SWA. Der DFLASH-Kopf sind **fuenf** Schichten, vier davon
  `sliding_attention` — und genau diese SWA-Schichten sind der Ort, an dem die
  GGUF-Community den Q4-Einbruch verortet. Der Q6_K-Beleg traegt also die These
  "Drafter-Koepfe sind quantisierungsrobust" nur fuer 8-bit-Klasse und nur fuer
  Nicht-SWA-Koepfe; er ist kein Freibrief fuer den DFLASH-Kopf.

### Ehrliche Spanne

**0 bis -3 Accept-Punkte** (relativ 0-7 %) fuer FP8-e4m3 per-channel W8, mit
dem wahrscheinlichen Wert bei **-1 Punkt oder im Rauschen**. Begruendung: FP8-e4m3
liegt praezisionsseitig in der Q8_0-Klasse (8 bit), fuer die die Community
"matches F16" misst; die per-channel-Skala ist gegen Q8_0s per-32-Block-Skala
etwas grober, was die Spanne nach unten offen haelt. Wo ein Verlust zuerst
sichtbar wird: an den **Block-Tail-Positionen** (13-15), nicht auf Position 0 —
publizierte DFlash-Positionskurven liegen bei 85-90 % auf Position 0 und
12-20 % auf Position 14.

**Staerkste Gegen-Evidenz** (bewusst benannt, nicht weggeraeumt):

1. Der Q4-Einbruch bei genau diesem Drafter zeigt, dass die SWA-Schichten
   quantisierungs-**empfindlicher** sind als ein normaler Draft-Kopf. 8 bit ist
   laut derselben Quelle sicher, aber die Empfindlichkeitsachse existiert
   nachweislich.
2. Die beiden GGUF-Repos widersprechen sich: eines misst Accept und verwirft
   Q4_K_M, das andere misst Durchsatz auf einer VRAM-engen 3090 und empfiehlt
   IQ4_XS (+12,8 %), weil Q8_0 dort **5,8 % langsamer** war. Das ist kein
   Widerspruch in der Physik, sondern zwei verschiedene Zielfunktionen —
   und eine Warnung, dass "kleiner" auf einer engen Karte gewinnen kann,
   obwohl der Accept faellt.
3. Eine Blog-Quelle behauptet fuer INT4/INT8-Gewichte nur 35 % Top-1-Accept
   gegen ~80 % bei voller Praezision. Das widerspricht 2505.22179 direkt und
   ist nicht per-format aufgeschluesselt; niedrig gewichtet, aber notiert.

---

## 4. Perf-Einschaetzung je Zielkarte

Ausgangspunkt: Kopf-Forward **2,58 ms captured** (BF16, 5090, R7a/R7b).

### Ist der Kopf bandbreiten- oder rechengebunden?

3300 MiB = 3,46 GB Gewichte je Forward / 2,58 ms = **1,34 TB/s effektiv**. Die
5090 hat ~1,79 TB/s Spitzenbandbreite — der Kopf laeuft also bei ~75 % des
Bandbreiten-Dachs und ist eindeutig **bandbreitengebunden**, nicht
rechengebunden. Das ist die entscheidende Groesse fuer die Prognose: FP8 halbiert
genau die Achse, die limitiert.

### 5090 (sm120)

Bytes je Forward FP8: 1,86 GB (1776 MiB, `fc` bf16) bzw. 1,73 GB (`fc`
quantisiert). Bei gleicher Bandbreiteneffizienz: **1,3-1,5 ms**, konservativ
**1,4-1,8 ms** — also **-30 bis -45 %** gegen 2,58 ms. Natives fp8-GEMM ist auf
sm120 vorhanden; das #255-Tuning belegt, dass Triton-fp8-Shapes auf dieser Karte
noch 18-46 % Luft hatten, bevor sie getunt wurden — die DFLASH-Kopf-Shapes sind
dort **nicht** enthalten, ein Tuner-Lauf ueber die vier Kopf-Shapes gehoert also
zum Vorhaben. Untere Erwartung ohne Tuning: Parity bis -20 %.

### 3080 (sm86)

Kein natives fp8-GEMM. Drei Pfade, alle mit Haken:

1. **Fused per-channel Dequant-GEMV (#189/#192)** — die schnelle Option (+35 %
   Decode gegen den Materialize-Pfad, gemessen am 27B FP8 TP=3), aber
   `FUSED_GEMV_MAX_ROWS = 8` (`fp8_dequant_gemv.py:78`). **DFLASH draftet einen
   Block von 16 Token gleichzeitig** (`block_size: 16` in der Drafter-Config),
   also M=16 > 8 — **der Kernel feuert fuer den DFLASH-Kopf heute nicht**. Die
   Konstante auf 16 zu heben ist eine Zeile plus Microbench; ohne das faellt der
   Kopf auf Pfad 2 oder 3.
2. **`CompressedTensorsW8A16Fp8`-Dequant (materialize + `F.linear`)** —
   materialisiert die BF16-Kopie, liest und schreibt also **mehr** Bytes als
   BF16 direkt. FP8 waere dort ein reiner VRAM-Gewinn und ein **Perf-Verlust**.
   Mittlerer Relativfehler 0,0133 gegen 0,0014 des fused Kernels.
3. **`gptq_marlin_gemm` W8A16** — das einzige fp8-GEMM, das sm86 hat, und
   run-to-run nichtdeterministisch oberhalb ~109 Prompt-Token (0 von 1200
   Mismatches bis M=109, erster Mismatch bei M=128; Fix nicht gemergt).

Groessenordnung: die 3080 20 GB hat ~760 GB/s. BF16-Kopf-Forward hat dort ein
Bandbreiten-Dach von **>=4,6 ms**; FP8 ueber einen echten fused Pfad **>=2,4 ms**.
Ueber den Materialize-Pfad eher **6-8 ms**, also langsamer als BF16.

**Wichtiger Nebenbefund zur sm86-Nichtdeterminismus-Sperre:** sie gatet das
**Ziel**, nicht einen **Drafter**. Unter Greedy-Verify ist die Ausgabe die
argmax-Folge des Ziels, unabhaengig davon, was der Drafter vorgeschlagen hat —
ein nichtdeterministischer Drafter veraendert nur die Accept-Varianz und damit
die Geschwindigkeit, nicht die Token. Die fp8@3080-Karenz ist damit kein
Hindernis fuer eine DFLASH-Lane auf einer 3080, solange der Verify-Pfad exakt
bleibt.

---

## 5. Fit-Tabelle

### Posten (aus dem Checkpoint gelesen, R7b-Tabelle)

| Posten | BF16 MiB | FP8 MiB | Anmerkung |
|---|---|---|---|
| MLP `gate/up/down` x5 | 2550,0 | 1275,0 | quant-config-faehig |
| `self_attn q,o` x5 | 400,0 | 200,0 | quant-config-faehig |
| `self_attn k,v` x5 | 100,0 | 50,0 | quant-config-faehig |
| `fc [5120, 25600]` | 250,0 | **250,0** / 125,0 | blankes `nn.Linear`; 125 nur nach Code-Aenderung |
| Normen | 0,1 | 0,1 | |
| per-channel-Skalen | — | ~1,0 | ~256k fp32 |
| **Summe Gewichte** | **3300,1** | **1776,1** / **1651,1** | Ersparnis 1524-1649 MiB |
| KV, naiver Pool (5 KV-Schichten, 20480 B/Token, 19200 Token) | 2000 | **2000** | DFlash ist laut [vLLM #41559](https://github.com/vllm-project/vllm/issues/41559) auf bf16-KV festgenagelt (nicht-kausale Attention) — **fp8-KV ist hier kein Hebel** |
| KV, SWA-bewusster Pool (4 von 5 Schichten Fenster 2048) | ~110 | ~110 | |
| Graphen | ~50 | ~50 | |

### Fit gegen die gemessenen freien MiB

**5090, R7b Boot 1: 1710 MiB frei** (nach voller Lane-Bringup, NEXTN-Lane steht):

| Variante | Bedarf | gegen 1710 | Verdikt |
|---|---|---|---|
| BF16 + naiver KV-Pool | 5350 | -3640 | passt nicht |
| BF16 + SWA-Pool | 3460 | -1750 | passt nicht |
| FP8 (`fc` bf16) + naiver KV-Pool | 3826 | -2116 | passt nicht |
| FP8 (`fc` bf16) + SWA-Pool | 1936 | **-226** | knapp daneben |
| FP8 (`fc` quantisiert) + SWA-Pool | 1811 | **-101** | knapp daneben |

**Ausweg 1 aus R7b — NEXTN-Lane durch DFLASH ERSETZEN** statt beide fahren:
frei werden 2684 MiB Komplement + ~400 MiB Kopf-Pool, also ~3794 MiB
zusaetzlich, insgesamt **~4790 MiB verfuegbar**:

| Variante | Bedarf | gegen ~4790 | Verdikt |
|---|---|---|---|
| BF16 + naiver KV-Pool | 5350 | -560 | passt nicht |
| BF16 + SWA-Pool | 3460 | +1330 | passt (R7b: "traegt, und dann knapp") |
| FP8 + naiver KV-Pool | 3826 | **+964** | **passt** — das ist neu |
| FP8 + SWA-Pool | 1936 | **+2854** | **passt bequem** |

**3080 (20480 MiB total), neben dem Verband-Shard:** die freien MiB je 3080 sind
fuer die R7b-Konfiguration **nicht gemessen** — R7b fuehrt genau diese
Freiraum-Messung als offenen Posten fuer R7c. Kein Verdikt ohne Messung. Was
sich sagen laesst: FP8 senkt den Bedarf einer 3080-Platzierung auf
**1936-3826 MiB** statt 3460-5350 MiB, und mit dem SWA-Pool auf unter 2 GB —
eine Groesse, die auf einer 20-GB-Karte mit einem Drittel-Shard des 27B
plausibel ist, aber eben unbelegt.

### Was der groessere Hebel ist

Der **SWA-bewusste KV-Pool spart 1890 MiB**, die **FP8-Quantisierung 1524-1649
MiB**. Beide sind noetig, keiner allein loest den 5090-Fall bei beibehaltener
NEXTN-Lane. Wer nur eines bauen kann, baut den SWA-Pool zuerst: er ist
verlustfrei, aendert keinen Zahlenwert im Modell, und wirkt auch fuer die
BF16-Variante.

---

## Empfehlung fuer R7c/R8

1. **Zuerst den SWA-bewussten KV-Pool** fuer die DFLASH-Lane (1890 MiB,
   verlustfrei, quantisierungsunabhaengig). Ohne ihn passt keine Variante.
2. **Dann Q8_0-GGUF-Drafter probieren, bevor FP8 gebaut wird** — gleiche
   Groessenklasse (1,75 GB gegen 1,78 GB FP8), existiert bereits, ist gegen F16
   gemessen, kostet keinen Quantisierungslauf. Gate: laedt unser
   GGUF-Pfad die `dflash-draft`-Tensornamen? Das ist eine CPU-Frage, in einer
   Stunde beantwortbar.
3. **FP8 nur, wenn 2 scheitert**: llm-compressor W8A8-dynamic `strategy: channel`,
   `fc` auf `ignore` (oder `ReplicatedLinear`), `speculative_draft_model_quantization`
   setzen. Kein Blockwise.
4. **Nicht auf einer 3080 messen, bevor `FUSED_GEMV_MAX_ROWS` auf >= 16 steht** —
   sonst misst man den Materialize-Pfad und schreibt "FP8 ist auf sm86 langsamer"
   ins Protokoll, obwohl nur der Dispatch-Gate zu eng war.
5. **Accept-Gate fuer den Vergleich**: Positionskurve, nicht mittlere
   Accept-Laenge. Ein 8-bit-Verlust zeigt sich an den Block-Tail-Positionen
   13-15, und genau dort ist die Kurve ohnehin flach — die mittlere Laenge
   wuerde ihn verschlucken.
