# NVFP4 natives Byte-Layout: Vertrag (Backlog #38, N4B, 25.09.2026)

Baum: `desk/27b-nvfp4-native-0925`, Basis `bb086e1120` (RC7b). Zeilenangaben in §0-§5 gelten für die Basis bb086e1120 (vor den Skelett-Commits, die modelopt_quant.py ab :1687 um ~30 Zeilen verschieben).
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

`--fp4-gemm-backend native-mixed` ist per Default AUS. Mit AUS sind `auto` und `marlin` byte-gleich, das belegt der
Test `test_default_paths_unchanged`.

- **Auswahl je Rang** (fp4_utils.initialize_fp4_gemm_config → nvfp4_native_mixed.resolve_this_rank):
  - sm_12x → `cutlass`;
  - sm_10x → `flashinfer_cutedsl`;
  - sm_8x → `w4a8_int8`, wenn der N4A-Kern registriert ist (Registry-Naht `register_w4a8_kernel`; Autoload des Moduls
    `sglang.srt.layers.quantization.nvfp4_w4a8_int8`, falls vorhanden);
  - sonst: harter, benannter Fehler.
  - Marlin gibt es nur mit `SGLANG_FP4_NATIVE_MIXED_ALLOW_MARLIN=1`. Der Rang ist dann als „verlässt das gemeinsame
    Layout“ markiert (`is_fp4_native_mixed_shared_layout() == False`).
- **Laden:**
  - Ein Shard, den der native Pfad polstern müsste, wird verweigert (`check_shard_alignment`, auch für fusionierte
    Komponenten).
  - `weight_global_scale` (fp32, 0-dim) wird auf JEDEM Rang gebunden, damit die Parametermenge auf allen Rängen gleich ist.
  - Die geswizzelte Skala bekommt einen Stempel: `nvfp4_sf_layout="128x4"`. Bei row-parallel-Schichten kommt
    `nvfp4_sf_tile_view=True` dazu.
  - Der Blackwell-Zwang entfällt nur für `w4a8_int8`.
- **apply:** `w4a8_int8` → `nvfp4_native_mixed.apply_w4a8` → der registrierte Kernel.
  Die Signatur steht im Modulkopf: (x, weight, weight_scale_swizzled, weight_global_scale, out_features).

## 7. Tausch-Deskriptoren: Lückenliste (Stand fee61349c3 + Tile-View-Commit)

Der 27B-Flip läuft über `--weg2-weight-source exchange` (docker/profiles/27b.env:64).

| # | Stelle | Was bricht | Vorschlag | Aufwand | Stand |
|---|---|---|---|---|---|
| L1 | weg2/xchg_manifest.py:665-843 `_axis_of` + weg2/weight_exchange.py `StorageGeom.of` | down_proj.weight_scale (geswizzelt) wird im P→D-Join als gewöhnlicher COLS-Schnitt gelesen. Das ergibt **falsche Bytes ohne Fehler**, am Test gezeigt: `test_plain_column_slice_is_wrong`. | Kachelsicht (N/128, K_pad·128) für gestempelte row-parallel-Skalen in `StorageGeom.of`. Alle vier Aufrufer laufen durch diese eine Stelle: ParamGeom.of, Drift-Check xchg_manifest.py:1279, flat-tables weight_exchange.py:1882/2155. | S | **gebaut**, gegatet durch den Stempel |
| L2 | weg2/weight_exchange_shadow.py:2950 `_qkv_component_rows` | Deklarierte Komponenten einer Kachelsicht-Skala müssen in Kacheln gezählt werden. | `_in_nvfp4_sf_tiles`: Division durch 128; eine nicht ganze Kachel ergibt `()`, der Join verweigert dann. | S | **gebaut** (betrifft heute keine Klasse, weil row-parallel nicht fusioniert ist) |
| L3 | Zeilenschnitte gate_up / lm_head | kein Bruch: 128er-Grenzen sind layout-neutral. lm_head D: 82816 = 647 Kacheln, das Vokabular-Pad (128 Zeilen) ist genau 1 Kachel und wird ZEROFILL (Skala 0). | Element-Sicht beibehalten, damit die Vokabular-Pad-Arithmetik (`_padded`, :800-812) weiter in Zeilen zählt. | – | Test `test_fused_row_shard_is_layout_neutral` |
| L4 | weight_exchange_shadow.py:755-775 `tensor_class` | `weight_scale_2`, `input_scale_inv`, `alpha`, `weight_global_scale`, `weight_scale_interleaved` fehlen in den `leafs` und werden als eigene Klasse gezählt. Rotation und Manifest-Klassennamen sind dadurch falsch gruppiert, die Bytes aber korrekt. | Die Namen (plus `weight_global_scale_w4a16`, §13) als Blätter, NUR unter dem boot-einheitlichen Env `SGLANG_FP4_NATIVE_MIXED_BOOT=1` (Launcher setzt es für sich und beide Gruppen). Das heutige Marlin-Profil behält seine Klassennamen. | S | **gebaut** (N4C), Test `TestExchangeClassesL4` |
| L5 | modelopt_quant.py:1775-1813 Polster | Ein N- oder K/16-Polster erzeugt zwei Skalen-Tensoren (roh + interleaved) mit abweichender Shape. | Im Modus verweigert (`check_shard_alignment`). Für 27B nie nötig (Block [128,128]). | S | **gebaut** |
| L6 | weg2/launcher.py:9247-9296 `uniform_marlin_argv` | `--fp8-uniform-marlin` + modelopt erzwingt `--fp4-gemm-backend marlin` für beide Gruppen. Es fehlt ein Launcher-Weg zu `native-mixed`. | Launcher-Flag (z. B. `--fp4-native-mixed`), das statt `marlin` die Form `native-mixed` ausgibt, FP8 bleibt vorerst Marlin. Dazu argv_gate-Zählung P/D nachziehen. | S | **gebaut**: Launcher-Flag `--fp4-native-mixed` (nur mit `--fp8-uniform-marlin` auf ModelOpt, sonst W160), Default byte-gleich |
| L7 | uneven_perf.py:736-760, 2005-2030 | Die Planer-Lanes kennen `nvfp4_native` (5090) und `nvfp4_marlin`, aber keine Lane `nvfp4_w4a8` für sm_86. P-Schnitt und D-Ratio würden mit Marlin-Raten der 3080 geplant. | N4C: Weg (b), also keine W4A8-Lane. Die Lanes `nvfp4_native` (5090) / `nvfp4_marlin` (3080) bekommen unter native-mixed Messwerte aus dem Evidenz-Register `planner/nvfp4_lane_record.py` (zcx7pv, M=512, mit Power-Limit), nur wo das Rig-Profil selbst nichts gemessen hat. Default-Boot liest das Profil unverändert. | M | **gebaut** (N4C), Test `TestPlannerLanesL7` |
| L8 | Marlin-Rückfall (marlin_utils_fp4.py:127-221) | Ein anderes Layout auf demselben Parameternamen: P-Ganzes und D-Stück haben andere Shapes, int32 [K/16, 2N] gegen u8 [N, K/2], der Join antwortet mit W68. | N4C: KEINE Kopie. Die Parameter behalten die native Form; nur der INHALT wird in place und blockweise (N-Bänder) permutiert, vor dem Einschlafen zurück nach native, nach dem Aufwachen nach Marlin (§13). | M | **gebaut** (N4C), Tests `test_nvfp4_marlin_inplace.py` |
| L9 | Skalare `alpha`, `input_scale_inv`, `weight_global_scale` (0-dim), `input_scale`/`weight_scale_2` [n_parts] | kein Bruch: `StorageGeom.of` beschreibt 0-dim als 1×1 (weight_exchange.py, Zweig `len(shape)==0`), der Join liest REPLICATED. | – | – | – |
| L10 | Parameter-Aliase | `weight_scale_interleaved` IST `weight_scale`; `named_parameters` sieht es einmal. Gestempelt wird das Objekt, der Stempel gilt für beide Namen. | Kein Code nötig. Nicht über `.data` gehen: das verliert den Stempel. | – | – |
| L11 | Ladeweg nach Flip-Eingang | Der Tausch schreibt Bytes in bestehende Parameter-Storages; `process_weights_after_loading` läuft nicht erneut. | Kein Bruch: Swizzle und Skalare sind Zustand der Bytes, nicht des Ladens. | – | – |
| L12 | Aufteilung P 42/11/11 | kein Schnitt in P (ganze Schichten) | – | – | – |
| L13 | Aufteilung D 58/25/25 | Das MLP läuft in Einheiten à 128 (136 Einheiten). Nur Rang-Grenzen auf 128 sind zulässig, sonst verweigert L5. Heute erzwingt das der Block. | Planer-Rundung auf 128er-Einheiten (besteht). | – | – |

## 8. FP8-Anteil (Attention/GDN, ~30 % der Linear-FLOPs), Operator-Auftrag 25.09.

- **Heute:** ModelOptFp8LinearMethod mit `SGLANG_FORCE_FP8_MARLIN` auf allen Rängen (launcher.py:9247-9255):
  W8A16 Marlin, Layout Marlin-FP8 (repacked).
- **5090 nativ:**
  - W8A8-FP8 statisch per-tensor. Die `input_scale` des Checkpoints ist die kalibrierte Aktivierungsskala, das ist der
    vom Exporteur vorgesehene Modus.
  - Pfad: `apply_fp8_linear` (fp8_utils.py:2203) → CUTLASS/`_scaled_mm` auf den FP8-Kernen.
  - Das Layout nach dem Laden ist `weight` e4m3 [K, N] als `.t()`-View auf [N, K]-Storage; die per-channel-Skala kommt
    aus `convert_to_channelwise` (modelopt_quant.py:661-668).
  - Frühere Messung (INTEGRATION_R3_VALIDATION.md:15960-15970, 31.07., M=2048):
    - fp8_native 529-568 TFLOPS gegen fp8_marlin 204-216;
    - nvfp4_native 1147-1386 gegen nvfp4_marlin 223-233.
- **3080** (kein FP8-Tensorpfad):
  - (a) Marlin W8A16 ist heute das Maß: 54-61 TFLOPS bei M=2048 (gleiche Quelle).
  - (b) Dequant nach FP16 plus FP16-MMA mit FP16-Akku: rechnerisch 119 TF, aber FP16-Akkumulation über K=5120/6144
    kostet Präzision. Nicht ohne Qualitätsmessung.
  - (c) INT8: E4M3 → INT8 ist NICHT verlustfrei (3 Mantissenbits bei 4 Exponentenbits Dynamik). Nach der Nutzerorder
    vom 08.09. nur dort, wo es nachweislich keine Intelligenz kostet. Stumm nie.
  - **Physikalisch schnellster verlustfreier Weg:** W8A16 (bf16-MMA mit FP32-Akku, 59,5 TF Spitze). Er ist bei großem M
    rechengebunden und bei Decode bandbreitengebunden, der Kernel ist dafür zweitrangig.
- **EIN Layout für FP8:**
  - Das native [N, K]-e4m3-Storage plus Skalar-Skala ist trivial schneidbar (Zeilen- und Spaltenschnitte ohne Swizzle).
  - Die 3080 bräuchte dann einen W8A16-Kernel, der [N, K] K-kontigu liest (CUTLASS-2.x-mixed-input oder
    Triton-Dequant-GEMM), statt des Marlin-Repacks.
  - Der Decode-GEMV auf [N, K]-Zeilen ist einfach und bandbreitengerecht.
  - Aufwand: M bis L (neuer sm_86-Kern + Tausch wie INT8 heute).
  - **Hebel P:** nur die 5090-Stufe gewinnt (FP8-Anteil 0,54 → 0,20 ms je Schicht und 512er-Chunk). Die 3080-Stufe
    bleibt beim FP8-Anteil gleich.

## 9. Das „other“ der P-Stufen (Operator-Auftrag 25.09., Desk aus #PGAP)

**Messung.** Quelle: #PGAP `gpu_fwd_ms` je 512er-Chunk, je Rang linear über die Chunk-Startposition p (k Token,
p ≤ 64k) gefittet. Boot weg2rc4 (INT8), Schnitt 42/11/11, Attention 10/3/3. Parser:
benchmark/nvfp4_native/pgap_fit.py (die Zeilenform steht in logindex nur als raw_shape, darum eigener Parser).

| Rang | Karte | Schichten (Attn/GDN) | gpu_fwd(p) INT8 | gpu_fwd(p) NVFP4-Marlin (weg2rc4n4) |
|---|---|---|---|---|
| PP0 | 5090 | 42 (10/32) | 48,6 + 0,894·p ms | 94,6 + 1,133·p ms |
| PP1 | 3080 | 11 (3/8) | 38,1 + 1,083·p ms | 93,8 + 1,113·p ms |
| PP2 | 3080 | 11 (3/8) | 42,1 + 1,111·p ms | 94,3 + 1,116·p ms |

Die Backlog-Werte 1,8 / 6,1 ms je Schicht sind **Mediane über den ganzen Boot** (Kontexte bis 148k). Für 8k gilt ein
mittleres p von 3,75k.

**Zerlegung pro Schicht und 512er-Chunk.** Zeilen 1 und 2 sind gemessen, Zeilen 3 bis 5 abgeleitet bzw. geschätzt,
wie in der Spalte „Quelle“ angegeben.

| Anteil | 5090 | 3080 | Quelle |
|---|---|---|---|
| Attention, kontextabhängig | 0,089 ms je Attn-Schicht je 1k Kontext (~144 TFLOPS, 69 % der BF16/FP32-Akku-Spitze) | 0,361 ms (~35,7 TFLOPS, 60 % von 59,5) | Steigung der Fits |
| GDN (linear attention, fla/triton) | ~0,11 ms je GDN-Schicht | ~0,31–0,39 ms | NF-Boots fnFL2x16x–x178, fnNV4f*, `ATTN-TIMING-PREFILL`, gleiche GDN-Dims (16/48 × 128), 8k/16k-Chunks. Bei 512 eher mehr. |
| Nicht-GEMM-Konstante gesamt | 0,58 ms je Schicht | 1,30 ms (PP1), 1,67 ms (PP2, letzte Stufe) | Fit(p=0) minus INT8-GEMM zur Lane-Rate bei M=2048; **Obergrenze** |
| davon Norm/Aktivierung/Quant/Residual | ~0,12 ms (Bandbreite ~1,5 TB/s) | ~0,28 ms (~180 MB bei ~650 GB/s) | Rechnung |
| unerklärter Rest | ~0,3 ms | ~0,6 ms | vermutlich INT8-GEMM bei M=512 langsamer als bei M=2048; Messung im Fenster zcx7pv |

**Modellprobe:** (16 + 2) × max Stufe bei p = 3,75k. INT8 8760 tok/s gegen gemessen 8905, NVFP4-Marlin 4468 gegen 4246.
Die 3080-Marlin-Rate bei M=512 ergibt sich aus dem NVFP4-Fit zu 53,8 TFLOPS (Lane-Wert bei M=2048: 63/60).

**Physik (Hochrechnung):**
- Attention auf sm_86 läuft schon bei ~60 % der FP16-MMA-Spitze (FP32-Akku). Der einzige große Hebel ist INT8-QK nach
  Sage-Art (INT8 238 TOPS). Der ist qualitätspflichtig (Nutzerorder 08.09.).
- GDN ist bei T=512 latenzgebunden: viele kleine Triton-Kerne. Hebel wäre ein fusionierter Chunk-Kern, Umfang unbekannt.
- Das P-Layout 42/11/11 legt 10 von 16 Attention-Schichten auf die 5090 (dort 4× schneller je FLOP). Die 3080-Stufen
  tragen je 3 Attention-Schichten: das ist der Kontext-Term 1,08 ms/k.

## 10. Hochrechnung P8k und D (benchmark/nvfp4_native/projection.py)

**HOCHRECHNUNG, keine Messung.** Kontextbewusstes Modell aus §9, kalibriert auf beide Boots.

| 3080 W4A8 (TOPS bei M=512) | 5090 FP4 + FP8 nativ | 5090 FP4 nativ, FP8 Marlin |
|---|---|---|
| 100 | Schnitt 49/8/7, **8,2–8,8k tok/s** | 45/10/9, 6,7–7,2k |
| 120 | 48/8/8, **8,2–8,8k** | 45/10/9, 6,8–7,3k |
| 155 | 47/9/8, **8,5–9,1k** | 43/11/10, 7,0–7,5k |

Heute gemessen: NVFP4-Marlin 4246, INT8 8905 tok/s.
- Neu bindet die **5090-Stufe** mit ihrer Nicht-GEMM-Konstante (0,58 ms je Schicht bei 47–49 Schichten).
- 10k tok/s bräuchte weniger „other“ auf der 5090 oder die Attention-/GDN-Hebel aus §9. GEMMs allein reichen nicht.
- **Offen:** Das 5090-VRAM für 47–49 P-Schichten. Im Vergleich zu 42 kommen ~1,6 GB Gewichte dazu, plus der P-KV bzw.
  Mamba-Zustand der zusätzlichen Schichten. Der Planer muss neu passen (P-Schnitt ist Flag + Messung).

## 12. Messung Fenster zcx7pv (25.09. ~12:40Z, Karten 1+2, Rohdaten /spinning/evidence-665-f1/n4b_bench_0925_1239/)

**MESSUNG.** µs je Aufruf, CUDA-Graph, Gewichte über >L2 rotiert. 5090 bei 400 W. `apply_nat` enthält die
FP4-Aktivierungs-Quantisierung.

| 5090, M=512 | NVFP4 nativ (apply_nat) | INT8 W8A8 (q+mm) | NVFP4 Marlin | BF16 |
|---|---|---|---|---|
| gate_up 34816×5120 | 220,6 (828 TF) | 391 (470 T) | 874 (209 TF) | 970 |
| down 5120×17408 | 119,4 (764 TF) | 192 (500 T) | 415 (220 TF) | 498 |
| M=4096 gate_up | 1575 (927 TF; mm allein 1344 = 1086 TF) | 2739 (535 T) | 7546 (194 TF) | 7714 |

| 5090, M=512, FP8-Projektionen | FP8 nativ cuBLASLt (`_scaled_mm` + Quant) | FP8 Marlin | INT8 q+mm |
|---|---|---|---|
| qkvz 16384×5120 | 260,6 (330 TF) | 417,2 | 218,8 |
| qkv 14336×5120 | 207,6 (362 TF) | 366,4 | 165,0 |
| o 5120×6144 | 107,2 (301 TF) | 161,6 | 77,7 |

Das `sgl_kernel`-FP8-CUTLASS auf sm_120 bricht ab: „Arch conditional MMA instruction used without targeting appropriate
compute capability“, der Kern ist nicht für sm_120a gebaut. flashinfer-b12x verlangt CUDA 13, am Rig ist nvcc 12.9.

**Decode, kleines M, 5090:**
- gate_up M=1: nativ 96 µs (1135 GB/s), Marlin **64 µs (1565 GB/s)**.
- M=16: 91 gegen 68 µs.
- M=48: 87 gegen 95 µs.
- **Marlin ist bis M≈32 schneller.**

**Rest-Kerne, T=512:**

| Kern | 5090 | 3080 |
|---|---|---|
| GDN-Chunk | 70 µs | 272 µs |
| Attention (SDPA) Präfix 4k | 332 µs (155 TF) | 1056 µs (49 TF) |
| Attention Diagonale | 42 µs | 115 µs |
| RMSNorm | 3,8 µs | 15,9 µs |
| silu_mul | 13,5 µs | 79,7 µs |
| Quant int8 (h/i) | 2,8 / 10,1 µs | 12,5 / 44,4 µs |

**Schichtsumme** (Messung je Kern, Summe gerechnet) gegen #PGAP bei 8k:
- **5090:** INT8-Kerne 1,025 ms gegen 1,237 ms gemessen, unerklärt 0,21 ms (17 %); GEMM-Anteil 0,87 ms = 70 %.
  NVFP4 nativ + FP8 nativ: 0,853 + 0,21 = **1,06 ms**. Marlin heute 2,01 + 0,21 = 2,22 ms. Nativ ist also **2,1×
  schneller als heute**, aber nur 1,17× schneller als INT8.
- **3080:** INT8-Kerne 3,59 ms gegen 3,83 ms gemessen, unerklärt 0,24 ms (6 %); GEMM-Anteil 2,99 ms = 78 %;
  Nicht-GEMM gemessen 0,61 ms je Schicht.
  - GDN: 0,27 ms × 8/11 Schichten.
  - Attention: 3 Schichten × (0,97 + 0,11) ms, auf 11 Schichten verteilt.
  - Norm und silu: 0,11 ms.

**INT4 (Nutzerfrage):**
- 3080: `mma.m16n8k64.s4` 471 TOPS = 2,00× INT8 (235). Nativ, SASS IMMA.16864.S4.S4.
- 5090: s4 wird emuliert (CALL, ~180 ALU-Ops + 2× IMMA.16832.S8). Gemessen 92–101 TOPS, also 0,1× INT8
  (1007–1010 TOPS mma.sync).

**HOCHRECHNUNG** (projection.py `--window`: Kernzeiten gemessen, Zusammensetzung, Schnitt und W4A8-Rate
angenommen, dazu der gemessene unerklärte Rest je Karte):

| 3080 W4A8 | 5090 NVFP4 nativ + FP8 nativ | 5090 NVFP4 nativ + FP8 Marlin |
|---|---|---|
| ≤117 TOPS (N4A, synthetische Obergrenze) | Schnitt 47/9/8, P8k **8,9k** (kalibriert 7,9–9,1k) | 45/10/9, 7,9k (7,0–8,0k) |
| 150 TOPS | 47/9/8, **9,1k** (8,0–9,3k) | 43/11/10, 8,1k (7,1–8,2k) |

- Es bindet die 5090 (47 Schichten zu ~1,06 ms). Darum hilft W4A8 117 → 150 kaum.
- 5090-VRAM in P bei 47 Schichten: Gewichte 14,5 GiB, Mamba 1,32, Graph/Reserve 1,2. Für KV bleiben ~13,5 GiB
  (~443k Token, Zelle 32 KiB), 262k Kontext passt.
- **D (Hochrechnung aus gemessenen Kernen), GEMM-Anteil des 5090-Rangs je Verify-Runde:**
  - M ≤ 16: Marlin 6,3–6,6 ms gegen nativ 8,3–9,0 ms, also **+2 ms bei bs1**.
  - M = 48: 9,8 gegen 8,5 ms.
  - Auf der 3080 bei M=48 ist Marlin rechengebunden (47 TF). W4A8 spart dort ~4–5 ms je Runde.
  - Folge: bs6 unter Last ~+15 %, bs1 ohne ein FP4-GEMV für kleines M auf dem nativen Layout eher −10 %.

## 11. Offen

- NVFP4-DFlash2-Draft: geht durch dieselbe Methode, also derselbe Vertrag. Shapes nicht geprüft.
- Mikrobench-Zahlen dieses Sitzes: siehe Bericht (GPU-Fenster nach der NF-Nachabnahme).

## 13. N4C (25.09.): 3080 = Marlin W4A16 auf dem gemeinsamen Layout (Weg b)

**Entscheidung.** FlashInfer `mm_bf16_fp4(backend="cute-dsl-native")` (PR #5242, Squash 2f3bc5a, 23.09.) liest das
native Layout direkt, ist aber auf SM120/121 gegatet (`@supported_compute_capability([120, 121])`). Der Dequant ist
Inline-PTX `cvt.rn.bf16x2.e2m1x2` / `cvt.rn.bf16x2.e4m3x2` (nur Blackwell bzw. sm_90+). Auf sm_86 gibt es diese
Instruktionen nicht; das ist eine Architekturgrenze, kein Upgrade-Risiko. Das Release 0.7.0 (22.09.) enthält ihn nicht.
Das `cute-dsl`-Backend in 0.6.14 ist sm_100+ (TMA, Cluster) und packt um. Also Weg (b).
Für die **5090 im Decode** ist cute-dsl-native dagegen genau richtig (W4A16 nativ, Upstream: down M=1 1,25× gegen
Marlin). Das hängt am FlashInfer-Upgrade (Strang F); Fs Kernelwahl ist eingehängt (unten).

**Dispatch** (`nvfp4_native_mixed.resolve_rank_backend`), Default der Option bleibt AUS:
- sm_12x → `cutlass` (W4A4). Für kleines M Strang Fs `nvfp4_sm12x_w4a16.maybe_apply_sm12x_w4a16` (FlashInfer
  cute-dsl-native W4A16 auf denselben nativen Bytes, `SGLANG_FP4_SM12X_W4A16_MAX_M`, Default AUS), EIN Aufruf in
  `ModelOptFp4LinearMethod.apply` nach den sm_8x-Zweigen; `None` → W4A4. alpha = `weight_global_scale`, auf jedem
  native-mixed-Rang gebunden. Braucht FlashInfer nach 0.7.0 (2f3bc5a) plus CuTe DSL 4.7.1 (Fs Zweig).
- sm_8x → `marlin_native_inplace` (Default) oder `w4a8_int8` mit `SGLANG_FP4_NATIVE_MIXED_SM8X=w4a8` (N4A-Naht bleibt).
- Die frühere Env `SGLANG_FP4_NATIVE_MIXED_ALLOW_MARLIN` (Marlin mit eigenem Layout) entfällt.

**In-place-Umformung** (`nvfp4_marlin_inplace.py`):
- Gleiche Bytezahl: `weight` u8 [N,K/2] ↔ int32 [K/16,2R] je Band; `weight_scale` e4m3 [N,K/16] (128x4) ↔ fp8 [K/16,R].
  Parameter-Objekte, Shapes, dtypes und Storages bleiben; Tausch, Join, CUDA-Graphen und Coverage sehen die native Menge.
- N-Bänder: R Vielfaches von 128, Band ≤ 16 MiB (`SGLANG_FP4_NATIVE_MIXED_BAND_MIB`), K-Schritte ≤ 2 MiB
  (`..._CHUNK_MIB`). Arbeitsmenge je Schritt ≤ Band + 6 Schritte = 28 MiB. KEINE Reserve (keine-korridor-reserve-nie).
- GEMM je Band (`apply_fp4_marlin_linear`), Ausgaben entlang N verkettet (exakt, keine Reduktion über N).
- Skalen: `nvfp4_marlin_process_scales` ist für E4M3 ≥ 0 verlustfrei und umkehrbar; negative Skalen werden beim Laden
  verweigert.
- Globale Marlin-Skala: Parameter `weight_global_scale_w4a16` (bf16 [1]) auf JEDEM native-mixed-Rang (replizierter
  Wert, einheitliche Parametermenge). Workspace = lokales Scratch (wird beim Aufwachen genullt, wie heute).
- Stempel `nvfp4_content` je Parameter-Objekt macht jede Umformung idempotent.
- Hooks (`weight_updater`): Schlafseite `_weg2_nvfp4_marlin_to_native` VOR dem Seam-Digest und dem ersten Deposit;
  Wachseite `_weg2_nvfp4_marlin_after_wake` NACH dem Seam-Digest-After, vor jedem Forward. Hat der Tausch getragen
  (`CARRIER_EXCHANGE`), gilt der Inhalt als nativ, egal was der Stempel sagt.
- Log-Zeilen: `NVFP4-MARLIN-INPLACE to_native layers=.. bytes=..MiB ms=..` und `... to_marlin ... delivered_native=..`.

**Kosten (RECHNUNG, nicht gemessen).** Je 3080 D ~2,5 GB (64 × 35 MB MLP-Stück + lm_head-Stück 238 MB), P 1,35–1,9 GB (8–9 Schichten à 150 MB, letzte Stufe + lm_head 716 MB) NVFP4; Verkehr ~8–10× bei ~600 GB/s plus
~10k Kernelstarts: ~40–80 ms je Richtung und Karte, beide 3080 parallel, also ~+2–4 % auf 3,1–4,0 s Flip.
Band-Verkettung im P-Forward: ~0,14 ms je 3080-Schicht (gate_up [512,34816] bf16 lesen+schreiben).

**Hochrechnung P8k** (`projection.py --window`, Kerne gemessen, Zusammensetzung gerechnet; 3080 = gemessenes Marlin):

| 3080 | 5090 | Schnitt | P8k (kalibriert) | 5090-VRAM P (fix / KV 262k / frei für KV) |
|---|---|---|---|---|
| Marlin W4A16 | NVFP4 nativ + FP8 Marlin (heute) | **50/7/7** | 7,1k (6,3–7,2k) | 17,8 / 8,5 / 12,7 GiB = 391k Token |
| Marlin W4A16 | NVFP4 nativ + FP8 nativ (sgl-kernel 120a offen) | **52/6/6** | 8,2k (7,2–8,3k) | 18,4 / 9,0 / 12,2 GiB = 354k Token |
| W4A8 117 TOPS (zum Vergleich) | + FP8 nativ | 47/9/8 | 8,9k (7,9–9,1k) | 17,0 / 8,0 / 13,5 GiB |

KV-Zelle mit 5 Draft-Schichten gerechnet (N4Bs Modell). Ohne Draft-Seiten auf P (`--dflash-produce-on-p off`) ist sie
kleiner, 262k passt dann erst recht. Der Unterschied W4A16 gegen W4A8 in P ist ~9–10 %, nicht „kaum“.

**FP8-Anteil.** 5090 nativ FP8 wartet auf den sgl-kernel-Neubau `86;120a` (CUTLASS FP8 bricht heute auf sm_120 ab).
Bis dahin bleibt FP8 auf allen Karten Marlin W8A16 (`--fp8-uniform-marlin`): **OFFEN**. Wird die 5090 FP8-nativ,
braucht die 3080 vermutlich dieselbe In-place-Behandlung für ihre FP8-Marlin-Gewichte (FP8-Marlin dürfte ebenfalls eine
Permutation gleicher Bytezahl sein; NICHT geprüft, eigene Aufgabe).
