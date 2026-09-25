# NVFP4 natives Byte-Layout: Vertrag (Backlog #38, N4B, 25.09.2026)

Baum: `desk/27b-nvfp4-native-0925` auf `bb086e1120` (RC7b). Alle Zeilenangaben gelten für diesen Baum.
Zweck: EIN Layout, das die 5090 (sm_120, FP4-Tensorkerne) direkt rechnet und das der 3080-Kernel W4A8 (N4A)
direkt liest. Am Flip wird nichts umgeformt.

Belegwerkzeuge: devindex `where weight_scale_interleaved` (einziger Leser: `ModelOptFp4LinearMethod.apply`,
modelopt_quant.py:1785 am Pin), `symbol_profile alias_or_bind_derived_param` (4 Aufrufer, alle modelopt).
Checkpoint-Werte kommen aus den Safetensors-Headern (Skript im Bericht).

## 0. Was im RadixArk-Checkpoint überhaupt NVFP4 ist

`hf_quant_config.json`: `quant_algo=MIXED_PRECISION`. Klasse `ModelOptMixedPrecisionConfig`,
Auswahl je Präfix in modelopt_quant.py:962-982.

| Linear (je Schicht) | Algo | Anzahl | Methode |
|---|---|---|---|
| mlp.gate_proj / up_proj / down_proj | NVFP4 g16 | 64 x 3 | `ModelOptFp4LinearMethod` |
| lm_head | NVFP4 g16 | 1 | `ModelOptFp4LinearMethod` (ParallelLMHead, qwen3_vl.py:1349) |
| self_attn q/k/v/o (16 Schichten), linear_attn in_proj_qkv / in_proj_z / out_proj (48) | FP8 per-tensor W8A8 statisch | 208 | `ModelOptFp8LinearMethod` |
| mtp.* | BF16 (exclude) | – | – |

**Folge:** NVFP4 deckt nur das MLP und den lm_head ab, etwa 70 % der Linear-FLOPs ohne lm_head
(MLP 17,1 G Parameter gegen 7,2 G FP8). Die FP8-Anteile laufen heute unter `--fp8-uniform-marlin` auf allen Karten
als Marlin-W8A16. Sie gehören NICHT zu diesem Vertrag (siehe §7, offener Punkt).

Checkpoint-Tensoren je NVFP4-Linear (Header, geprüft an layers.0/3 und lm_head):

| Name | dtype | Shape | Bedeutung |
|---|---|---|---|
| `weight` | U8 | [N, K/2] | E2M1, 2 Werte je Byte |
| `weight_scale` | F8_E4M3 | [N, K/16] | Blockskala je 16 K-Elemente, zeilenweise, NICHT geswizzelt |
| `weight_scale_2` | F32 | [] | globale Gewichtsskala |
| `input_scale` | F32 | [] | globale statische Aktivierungsskala |

Geprüft an allen 64 Schichten: `gate_proj` und `up_proj` haben dieselben Werte für `weight_scale_2` und `input_scale`.
Damit ist die `max()` in modelopt_quant.py:1673-1674 für das fusionierte `gate_up_proj` exakt und keine Näherung.

## 1. Tensoren nach `create_weights` (modelopt_quant.py:1595-1670)

Je Rang gilt `N = output_size_per_partition = sum(output_partition_sizes)` und `K = input_size_per_partition`
(Prüfung K % 16 == 0 in :1620).

| Parameter | Zeile | dtype | Shape | Loader-Attribute |
|---|---|---|---|---|
| `weight` | :1633-1643 | uint8 | [N, K/2] | ModelWeightParameter, output_dim=0, input_dim=1 |
| `input_scale` | :1645-1650 | fp32 | [n_parts] | PerTensorScaleParameter, needs_scalar_to_array |
| `weight_scale_2` | :1652-1657 | fp32 | [n_parts] | PerTensorScaleParameter, needs_scalar_to_array |
| `weight_scale` | :1659-1670 | float8_e4m3fn | [N, K/16] | ModelWeightParameter, output_dim=0, input_dim=1 |

`n_parts = len(output_partition_sizes)`: 2 für gate_up, 1 für down und lm_head.

## 2. Tensoren nach `process_weights_after_loading`, Pfad sm_120 nativ (Backend `cutlass`)

Ablauf in modelopt_quant.py:1672-1860. `auto` löst auf sm_120 `cutlass` auf (das fork-eigene CUTLASS-sm_120a,
fp4_utils.py:199-222). Auf sm_86 löst `auto` `marlin` auf (fp4_utils.py:223-229). Der 27B-Launcher erzwingt heute
`--fp4-gemm-backend marlin` für alle Ränge (weg2/launcher.py:9247-9261).

1. **Skalare** (:1673-1685):
   - `alpha = max(input_scale) * max(weight_scale_2)`: fp32, **0-dim**, neuer `Parameter` (copy_or_rebind_param, layers/utils/common.py:58-71).
   - `input_scale_inv = 1 / max(input_scale)`: fp32, 0-dim, ebenfalls neuer `Parameter`.
   - `input_scale` [n_parts] und `weight_scale_2` [n_parts] bleiben als Parameter liegen, unverändert.
2. **Gewicht** (:1775-1777, `pad_nvfp4_weight` :238-285):
   - N wird auf ein Vielfaches von 32 gepolstert, K auf ein Vielfaches von 32 Elementen (= 16 Byte).
   - `weights_padding_cols` (Byte) wird Layer-Attribut.
   - `output_size_per_partition` wird VOR dem Polstern auf `weight.shape[0]` gesetzt (:1687), `apply` schneidet die Ausgabe wieder ab (:1922).
   - Ist nichts zu polstern, behält `copy_or_rebind_param` die Parameter-Identität, und alle Loader-Attribute bleiben erhalten.
3. **Skalen-Swizzle** (:1779-1813):
   - `weight_scale` [N, K/16] wird auf [M_pad = ceil128(N), K_pad = ceil4(K/16)] mit Nullen gepolstert (:1786-1790).
   - Danach folgt `reshape(1, M_pad/128, 4, 32, K_pad/4, 4).permute(0,1,4,3,2,5)` (:1802-1803). Das ist das
     CUTLASS-/TRT-LLM-Layout „128x4“. Kernel-Gegenstück: `cvt_quant_to_fp4_get_sf_out_offset`,
     jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh:107-146.
   - Die Byte-Adresse der Skala für Zeile m und Skalenspalte kb (kb = k/16) ist:

     ```
     off(m, kb) = (m // 128) * (K_pad * 128)          # Zeilen-Kachel, zusammenhaengend
                + (kb // 4)  * 512                    # K-Kachel (4 Skalen = 64 K-Elemente)
                + (m % 32)   * 16
                + ((m % 128) // 32) * 4
                + (kb % 4)
     ```
   - Bindung (:1810): `alias_or_bind_derived_param(layer, "weight_scale", "weight_scale_interleaved", ...)`,
     layers/utils/common.py:74-106.
     - **Ohne Polster** (Shape und dtype gleich) werden die geswizzelten Bytes IN den Speicher von `weight_scale`
       geschrieben. `weight_scale_interleaved` ist dann dasselbe Parameter-Objekt.
       `named_parameters()` dedupliziert und liefert es EINMAL unter dem Namen **`weight_scale`**.
       Der Name `weight_scale` trägt also die geswizzelten Bytes.
     - **Mit Polster** wird ein eigener Parameter `weight_scale_interleaved` [M_pad, K_pad] angelegt.
       Der rohe `weight_scale` bleibt als totes Byte-Paket liegen: kein Kernel liest ihn, aber der Tausch sieht ihn.
4. **Swiglu-Nebenkanal** (:1814-1858): nur bei `_interleave_for_swiglu_fusion`. Für Qwen3.5 nicht gesetzt,
   daher nicht Teil des Vertrags.

**Endzustand, nativer Pfad** (Parameter in `named_parameters`-Reihenfolge):
`weight` u8 [N_p32, K_p32/2], `input_scale` f32 [n_parts], `weight_scale_2` f32 [n_parts],
`weight_scale` (= interleaved) e4m3 [ceil128(N), ceil4(K/16)] geswizzelt, `alpha` f32 [], `input_scale_inv` f32 [].
Dazu die Nicht-Tensor-Attribute `weights_padding_cols` und `output_size_per_partition`.

**Für ALLE 27B-Shapes gilt: kein Polster.** Alle Shards sind Vielfache von 128, weil der uneven-TP-Block
[128, 128] ist (siehe §4).

| Tensor | P (5090, ganze Schicht) | D-Shard (Beispiel 73/32/31 Einheiten à 128) |
|---|---|---|
| gate_up.weight | u8 [34816, 2560] | u8 [2·9344 / 2·4096 / 2·3968, 2560] |
| gate_up.weight_scale | e4m3 [34816, 320] swz | [18688 / 8192 / 7936, 320] swz |
| down.weight | u8 [5120, 8704] | u8 [5120, 4672 / 2048 / 1984] |
| down.weight_scale | e4m3 [5120, 1088] swz | [5120, 584 / 256 / 248] swz |
| lm_head.weight | u8 [248320, 2560] | u8 [82816, 2560] je Rang (Vokabular gerade verteilt, 128 Pad-Zeilen am letzten Rang) |
| lm_head.weight_scale | e4m3 [248320, 320] swz | [82816, 320] swz (82816 = 647·128) |

## 3. Rechenvertrag (was jeder Kernel aus diesen Bytes rechnen muss)

- **E2M1-Nibble-Reihenfolge:** Byte j einer Zeile trägt Element 2j im UNTEREN Nibble und 2j+1 im OBEREN.
  Beleg: `cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1` mit %1 = array[0], nvfp4_quant.cuh:41. PTX legt den ersten
  Quelloperanden ins obere Nibble.
  Kodierung: Bit 3 ist das Vorzeichen, Bits 0-2 ergeben {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Mal 2 ergibt das
  {0, 1, 2, 3, 4, 6, 8, 12}, exakt in INT8 (Nutzerformel).
- **Gewicht:** `W[n,k] = e2m1(n,k) * e4m3(ws[n, k//16]) * weight_scale_2`.
- **5090, W4A4** (`apply`, :1885-1926):
  - Quantisierung: `x_fp4, x_sf = fp4_quantize(x, input_scale_inv)` (Blockskala je 16, 128x4-geswizzelt).
  - GEMM: `out = alpha * sum_k e2m1(x)*sf_x * e2m1(w)*sf_w`.
  - Kernel: `fp4_gemm` → `cutlass_scaled_fp4_mm(x_fp4, weight, x_sf, weight_scale_interleaved, alpha)`, modelopt_quant.py:144-160.
- **3080, W4A8 (N4A):**
  - `out[m,n] = s_x[m] * weight_scale_2 * sum_b ( e4m3(ws[n,b]) * sum_{k in b} (2·e2m1(w)) * q8(x) ) / 2`.
  - Benötigt: `weight`, `weight_scale` (geswizzelt, Adressformel oben) und `weight_scale_2` (Skalar, `max()` über n_parts).
  - NICHT benötigt: `alpha` und `input_scale_inv`, denn die gehören zur FP4-Aktivierung.
  - `weight_scale_2` liegt als [n_parts]-Parameter vor. Der Kernel nimmt `weight_scale_2.max()`; für gate_up ist das exakt, siehe §0.

## 4. Aufteilungsvertrag (uneven TP, PP)

- **Block.** `ModelOptMixedPrecisionConfig.weight_block_size` ergibt für RadixArk **[128, 128]**
  (modelopt_quant.py:741-770 → `modelopt_fp4_uneven_tp_block(16)` = lcm(32, 128, 16), :185-231).
  `Qwen2MoeMLP` vergröbert die MLP-Einheiten einmal und gibt dieselben Einheiten an gate_up und down (qwen2_moe.py:177-275).
  Daraus folgt: intermediate 17408 = 136 Einheiten à 128, und jede Rang-Grenze liegt auf 128 Elementen.
- **Zeilenschnitt** (column-parallel gate_up, lm_head-Vokabular):
  - Rang- und Komponentengrenzen sind Vielfache von 128, also ganze Skalen-Zeilenkacheln.
  - Die Kachel t liegt in BEIDEN Layouts (roh und geswizzelt) auf den Bytes [t·128·K_pad, (t+1)·128·K_pad).
  - Ein Zeilenschnitt auf 128er-Grenzen ist darum **layout-neutral**: dieselben Bytes wie heute beim INT8-Zeilenschnitt.
  - Die fusionierten Komponenten [gate_r | up_r] sind je 128-ausgerichtet, also keine Kachel über die gate/up-Grenze.
- **Spaltenschnitt** (row-parallel down, K-Achse):
  - Rohes Gewicht [N, K/2]: gewöhnlicher Spaltenschnitt, K-Grenze auf 128 Elementen = 64 Byte.
  - Geswizzelte Skala: **KEIN gewöhnlicher Spaltenschnitt.** In der **Kachelsicht** `view(N/128, K_pad·128)` (Bytes)
    IST er einer: Rang r bekommt die Spalten [kb0·128, (kb0+K_pad_r)·128).
  - Bedingung: kb0 und K_pad_r sind Vielfache von 4, also K-Grenzen auf 64 Elementen. Mit Einheiten à 128 gilt immer
    K_pad_r = 8·Einheiten.
  - Der Shard eines Rangs ist dann byte-gleich zum nativen Swizzle seines eigenen [N, K_r/16]-Shards. Herleitung:
    Die Formel in §2 ist innerhalb der Kachel lokal. Der Zeilenkachel-Stride K_pad·128 ist die einzige Größe, die vom
    vollen K abhängt.
- **PP (P-Schnitt 42/11/11):** P hält ganze Schichten, kein Schnitt, keine Bedingung.
- **Skalare** (`input_scale`, `weight_scale_2`, `alpha`, `input_scale_inv`): REPLIZIERT, auf allen Rängen einer Schicht identisch.
- **Polster verboten** im nativen Tauschmodus. Jeder Rang-Shard muss N % 128 == 0 und (K/16) % 4 == 0 erfüllen,
  sonst entstehen die Doppel-Skala (§2.3) und abweichende Shapes. Für 27B ist das durch den Block erfüllt.
  Zum Desk-Check: siehe `native_mixed.check_shard_alignment` im Skelett.

## 5. Was der Marlin-Pfad aus denselben Bytes macht (warum er am Flip nicht mitspielt)

`prepare_nvfp4_layer_for_marlin` (marlin_utils_fp4.py:127-221) formt um:
- `weight` → int32 [K/16, 2N] (gptq_marlin_repack);
- `weight_scale` → Marlin-permutiert, fp8 [K/16, N];
- `weight_global_scale` (bf16, verarbeitet) und `workspace` kommen neu dazu.

Das ist ein anderes Byte-Layout. Ein Rang auf Marlin kann native Bytes aus dem Tausch nicht direkt lesen: Rückfall
Marlin = Umformung am Flip (Repack je Schicht nach Ankunft plus Transient). Das bestätigt die Profilzeile in
docker/profiles/27b-nvfp4.env („die Layouts passen im Tausch nicht zusammen“).

## 6. Der nativ-gemischte Modus (Skelett in diesem Baum)

`--fp4-gemm-backend native-mixed` (Default AUS):
- Jeder Rang behält das native Layout aus §2.
- Die Kernel-Wahl hängt an der Compute Capability des Rangs:
  - sm_12x → `cutlass` (W4A4);
  - sm_8x → `w4a8_int8`, der N4A-Kernel über eine Registry-Naht;
  - ohne registrierten W4A8-Kernel → harter Fehler. Marlin nur mit ausdrücklicher Erlaubnis; es ist dann eine
    abgeleitete Kopie mit Repack am Flip, siehe §5.

## 7. Offen

- FP8-Linears (30 % der Linear-FLOPs): nativ FP8 auf der 5090 hätte dasselbe Tauschproblem ([N,K] e4m3 + Skalar
  gegen den Marlin-FP8-Repack). Nicht Teil dieser Runde.
- NVFP4-DFlash2-Draft: geht durch dieselbe Methode, also derselbe Vertrag. Shapes nicht geprüft.
