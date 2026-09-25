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
| L4 | weight_exchange_shadow.py:755-775 `tensor_class` | `weight_scale_2`, `input_scale_inv`, `alpha`, `weight_global_scale`, `weight_scale_interleaved` fehlen in den `leafs` und werden als eigene Klasse gezählt. Rotation und Manifest-Klassennamen sind dadurch falsch gruppiert, die Bytes aber korrekt. | Die fünf Namen in `leafs` aufnehmen. Das ändert die Klassennamen im Manifest AUCH für das heutige Marlin-NVFP4-Profil (`weight_global_scale`), darum nicht in dieser Runde. | S | Plan |
| L5 | modelopt_quant.py:1775-1813 Polster | Ein N- oder K/16-Polster erzeugt zwei Skalen-Tensoren (roh + interleaved) mit abweichender Shape. | Im Modus verweigert (`check_shard_alignment`). Für 27B nie nötig (Block [128,128]). | S | **gebaut** |
| L6 | weg2/launcher.py:9247-9296 `uniform_marlin_argv` | `--fp8-uniform-marlin` + modelopt erzwingt `--fp4-gemm-backend marlin` für beide Gruppen. Es fehlt ein Launcher-Weg zu `native-mixed`. | Launcher-Flag (z. B. `--fp4-native-mixed`), das statt `marlin` die Form `native-mixed` ausgibt, FP8 bleibt vorerst Marlin. Dazu argv_gate-Zählung P/D nachziehen. | S | **gebaut**: Launcher-Flag `--fp4-native-mixed` (nur mit `--fp8-uniform-marlin` auf ModelOpt, sonst W160), Default byte-gleich |
| L7 | uneven_perf.py:736-760, 2005-2030 | Die Planer-Lanes kennen `nvfp4_native` (5090) und `nvfp4_marlin`, aber keine Lane `nvfp4_w4a8` für sm_86. P-Schnitt und D-Ratio würden mit Marlin-Raten der 3080 geplant. | Lane `nvfp4_w4a8_int8` mit N4As Mikrobench-Werten; Familienauflösung `mlp`/`vocab` für native-mixed. | M | Plan |
| L8 | Marlin-Rückfall (marlin_utils_fp4.py:127-221) | Ein anderes Layout auf demselben Parameternamen: P-Ganzes und D-Stück haben andere Shapes, int32 [K/16, 2N] gegen u8 [N, K/2], der Join antwortet mit W68. | Im Modus nur mit ausdrücklicher Env-Freigabe. Richtig wäre abgeleitete Marlin-Kopie + Repack nach jedem Flip-Eingang (Zeit + Transient). | M | Refusal gebaut, Repack-Pfad Plan |
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

## 9. Offen

- NVFP4-DFlash2-Draft: geht durch dieselbe Methode, also derselbe Vertrag. Shapes nicht geprüft.
- Mikrobench-Zahlen dieses Sitzes: siehe Bericht (GPU-Fenster nach der NF-Nachabnahme).
