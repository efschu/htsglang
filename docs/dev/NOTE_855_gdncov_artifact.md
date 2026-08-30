# NOTE #855 — `Qwen3.8-27B-INT8-gdncov`: the GDN-covered W8A8 artifact

Step 3 of `ANALYSE_854_w8a16_vs_w8a8.md` §8 ("requant the incumbent ourselves,
with GDN coverage, staying W8A8"), executed at the desk on 2026-08-24.
**CPU/disk only — no GPU was claimed, nothing was booted, serving stayed down.**

**Artifact:** `/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov`
**Size:** 30,819,147,232 B = 29,391.4 MiB = **28.70 GiB** (incumbent: 33.86 GiB)
**Build tool:** `tools/requant_gdn_int8_855.py` (36/36 hermetic self-tests)
**Dispatch test:** `test/registered/unit/quantization/test_gdn_int8_dispatch_855.py` (12/12)

**Headline:** the 48 GDN layers' 144 dense projections are now int8 per-channel
symmetric with dynamic per-token activations — the identical scheme, the
identical kernel, and the identical `config_groups` block as the rest of the
model. The artifact frees **5,278.0 MiB**, i.e. **+168,897 KV tokens**, which is
**more than either figure previously published** (§4 reconciles why).

**Quality risk: no red flag found.** The GDN weights are statistically
indistinguishable from the MLP weights the incumbent's producer already
quantized (§3.2). The data-free RTN route is **not** disqualified, and the
calibrated alternative (AutoRound/SmoothQuant on GDN only) is **not** needed as
a follow-up on present evidence. What remains open is the boot A/B (§6) — this
note supplies structure and arithmetic, not behaviour.

---

## 1. What was built, and what was deliberately not

Quantized — 3 families x 48 layers = **144 projections**:

| tensor | shape | BF16 | int8 |
|---|---|---:|---:|
| `linear_attn.in_proj_qkv.weight` | (10240, 5120) | 4,800.0 MiB | 2,400.0 MiB |
| `linear_attn.in_proj_z.weight` | (6144, 5120) | 2,880.0 MiB | 1,440.0 MiB |
| `linear_attn.out_proj.weight` | (5120, 6144) | 2,880.0 MiB | 1,440.0 MiB |
| **total** | | **10,560.0 MiB** | **5,280.0 MiB** |

That total reproduces ANALYSE_854 §3.2's "GDN dense projections | 10,560.0
*(BF16)*" row exactly, which confirms the analysis's component attribution
against the real tensors rather than against its model of them.

Left BF16 and still in the ignore list: `in_proj_a`, `in_proj_b` (48x5120 gates,
22.5 MiB each), `conv1d`, `norm`, `A_log`, `dt_bias`, plus `embed_tokens`,
`lm_head`, and the vision tower. The embed/lm_head axis is #727's and is
deliberately untouched here — #763 is the reason two new quantized components
do not get introduced in one artifact.

### 1.1 The 144/240 boundary is forced by the runtime, not chosen

A GDN layer has five projection-shaped tensors, not three. The other two
(`in_proj_a`, `in_proj_b`) are excluded, and the cut line is not a taste call:

`qwen3_5.py:1283-1288` packs `in_proj_qkv + in_proj_z -> in_proj_qkvz` and
`in_proj_b + in_proj_a -> in_proj_ba`. `should_ignore_layer`
(`compressed_tensors/utils.py:62-79`) resolves a packed module by checking every
constituent shard and **raises `ValueError` if they disagree**:

```python
elif should_ignore_shard != should_ignore_layer:
    raise ValueError(
        f"Found different quantization schemes for "
        f"{shard_proj_names} in {layer_name}. SGLang "
        "requires all to use the same scheme.")
```

So the only legal cut lines are "all of qkvz" and "all of ba". Quantizing
`in_proj_qkv` without `in_proj_z` would not be a quality tradeoff, it would be a
load-time crash. The chosen split sits exactly on that boundary, and
`test_packed_module_shards_agree` pins it.

### 1.2 Ignore-list surgery

Dropping `re:.*linear_attn.*` outright would also expose the two gates, whose
int8 weights this artifact does not contain — the loader would then hunt for
`in_proj_ba` scales that do not exist. One entry therefore becomes two:

```
- "re:.*linear_attn.*"
+ "re:.*linear_attn\\.in_proj_a.*"
+ "re:.*linear_attn\\.in_proj_b.*"
```

`conv1d` and `norm` need no replacement — the checkpoint's own
`re:.*conv1d.*` / `re:.*norm.*` entries already cover them. Everything else in
the ignore list is carried over untouched, and `config_groups` is byte-identical
to the incumbent's.

## 2. Method: data-free RTN, because that is the incumbent's own method

The incumbent's `weights` group reads `strategy: channel, symmetric: true,
num_bits: 8, dynamic: false, observer: memoryless_minmax`. `memoryless_minmax`
per output channel **is** round-to-nearest on `amax/127`; no calibration data
enters it. Requantizing the skipped tensors the same way is therefore not a
cheaper approximation of the producer's method — it is the same method applied
to the tensors the producer skipped. Verified from the incumbent's own
`config.json`, as required, rather than assumed.

### 2.1 One correction the checkpoint's own layout forced

Scales are stored **BF16** (`[out, 1]`, matching the existing companions), so
BF16 is what the runtime dequantizes with. Quantizing against the fp32 scale and
only then rounding the scale down to BF16 leaves an error the ideal RTN bound
does not cover: BF16 carries 8 mantissa bits, so a scale off by up to `2^-9`
relative multiplies through a code of up to 127 (~0.25 extra steps), and a
stored scale that lands *below* the true one additionally clips the amax
element. The first self-test run caught exactly this (`err within RTN bound`
FAILED at 0.75 steps).

The tool now quantizes against the BF16 scale itself, and nudges that scale up
to the next BF16 value whenever rounding put it below `amax/127`. This
guarantees `|w/s| <= 127`, making the clamp dead code rather than a silent error
source, and restores the exact half-step bound. **All 144 tensors measure
`max_err/step = 0.5000` exactly** — the bound is tight and met, on every tensor.

### 2.2 Cost discipline (the #727 pattern, reused)

Per-output-channel scales make row blocking **exact**, not approximate — each
output row is independent, so a blocked result is bit-identical to a
whole-tensor one. The self-test proves this against a deliberately unblocked
reference rather than asserting it. Only the 16 shards carrying a target were
rewritten; the other 2 are hardlinked (`same-inode` verified), as are all
non-safetensors files. The `quantize_per_channel_symmetric` body is carried over
from `tools/requant_vocab_int8.py` — same scheme, same code.

Peak RSS stayed at roughly one shard (~2.6 GiB); there is no 8-GiB upcast spike.

## 3. Verification

### 3.1 Structural — 16/16 checks, 0 failures

- 144 targets identified; all BF16 -> I8; **shapes preserved**; each has a BF16
  `weight_scale` of shape `[out, 1]`.
- All **1,319** non-target tensors unchanged in dtype and shape; a sampled MLP
  tensor is **bit-identical** on disk; `in_proj_a` is still BF16, identical, and
  has **no** scale companion.
- The only new tensors are exactly the 144 scale companions; no tensor lost.
- `model.safetensors.index.json`: weight_map matches the shard headers exactly,
  every entry resides in its named shard, each scale sits in the same shard as
  its weight, and `metadata.total_size` equals the recomputed byte count
  (30,819,147,232 both ways).
- Size delta matches the analytic prediction to within 0.01 MiB.
- `quant_method`, `format`, top-level config keys preserved; `config_groups`
  byte-identical.

**Checksums.** `crc32.txt` covers only the 8 tokenizer/template text files, and
nothing in `model_loader/` or `configs/` reads it — it is provenance metadata,
not a loader input. All 8 files are hardlinked byte-identical into the artifact.
Noted honestly: 3 of those 8 (`chat_template.jinja`, `generation_config.json`,
`tokenizer_config.json`) **already fail the incumbent's own crc32.txt**,
identically, because they were edited after it was written (mtimes confirm).
gdncov inherits that pre-existing staleness byte-for-byte; it neither introduces
nor repairs it.

### 3.2 Numerical — the quality question, and the control that answers it

Round-tripped **from the bytes on disk** (not from in-memory state):

| family | n | rel-Frobenius err: min / median / max | min SNR | max err/step |
|---|---:|---:|---:|---:|
| `in_proj_qkv` | 48 | 0.934 % / 1.025 % / 1.155 % | 38.8 dB | 0.5000 |
| `in_proj_z` | 48 | 0.926 % / 0.984 % / 1.146 % | 38.8 dB | 0.5000 |
| `out_proj` | 48 | 0.982 % / 1.054 % / **1.447 %** | 36.8 dB | 0.5000 |
| **all 144** | 144 | 0.926 % / 1.018 % / **1.447 %** | **36.8 dB** | 0.5000 |

Worst tensor: `layers.57.linear_attn.out_proj.weight`, 1.447 %, SNR 36.8 dB.
~1 % / ~40 dB is the textbook result for per-channel int8 RTN on
Gaussian-shaped weights, and no tensor is an outlier against its own family.

**The control that matters.** The brief asked whether a pathological per-channel
range is the reason the original quantizer skipped these layers. It is not.
Measuring the same difficulty statistic — crest factor `amax/rms` per output
channel — on GDN (BF16 originals) and on the MLP tensors the producer **did**
quantize (dequantized incumbent):

| tensors | median crest | max crest | max amax-ratio |
|---|---:|---:|---:|
| GDN (all 144, quantized here) | ~4.0-4.3 | 53.8 | 23.6 |
| MLP `down/gate/up_proj` (already int8 in the incumbent) | ~3.8-4.5 | 32.7 | 22.3 |

The two populations are the same population. GDN's peak crest is somewhat
higher (53.8 vs 32.7) but its median is if anything slightly *lower*, and its
amax-ratio range is indistinguishable. **There is no outlier-channel pathology
in the GDN projections**, and nothing here disqualifies the data-free route.

The exclusion is therefore best read as a recipe default — llmcompressor
configs routinely exclude custom/non-standard attention modules by name — not as
a quality finding the producer made and we are overriding. That reading is a
hypothesis about someone else's intent; the measurement above is not.

### 3.3 Dispatch — hermetic, `CUDA_VISIBLE_DEVICES=""`

`test_gdn_int8_dispatch_855.py`, 12/12 passed, no GPU, no checkpoint bytes.
The GDN projections resolve to **`CompressedTensorsW8A8Int8`** with
`strategy="channel"` and `is_static_input_scheme=False` (dynamic per-token
activations), via:

- routing: `get_scheme_dict` -> `should_ignore_layer`,
  `compressed_tensors.py:1060-1063`
- dispatch: `_is_dynamic_token_w8a8` at `compressed_tensors.py:562-577`, taken at
  **`compressed_tensors.py:803-809`**:

```python
if self._is_dynamic_token_w8a8(weight_quant, input_quant):
    if not _is_npu:
        return CompressedTensorsW8A8Int8(
            strategy=weight_quant.strategy,
            is_static_input_scheme=False,
            input_symmetric=input_quant.symmetric,
        )
```

`test_gdn_scheme_identical_to_mlp_scheme` asserts the GDN tuple equals the
`mlp.down_proj` tuple — the point being that GDN *joins* the existing lane
rather than acquiring one of its own. `test_incumbent_ignores_all_gdn` runs the
incumbent's ignore list through the same matcher and requires the opposite
answer, so the suite cannot pass vacuously.

### 3.4 The #763-class hazard: checked in code, then confirmed live

`out_proj` is RowParallel, so its **input** dim (6144) is sharded while its
per-channel scales are indexed by the **output** dim (5120) and must be held
whole on every rank. If they were narrowed with the K shard, every rank would
dequantize with wrong scales — wrong numbers, no error raised. That is the #763
shape exactly.

It cannot happen here, by type: `ChannelQuantScaleParameter`
(`parameter.py:407-413`) is `_ColumnvLLMParameter` only, is **not** a
`RowvLLMParameter`, and so inherits `BasevLLMParameter.load_row_parallel_weight`
— the full-copy path, not a narrowing one. `RowParallelLinear` passes the full
`output_size` (`linear.py:2083`) regardless of TP, so the allocation is
TP-invariant. Contrast `in_proj_qkvz` (ColumnParallel): output rows *are* split,
so its scales *are* narrowed, correctly.

Nothing in the tree asserted this. Added: `TestGdnOutProjScaleNotSharded`
(3 tests), which fails loudly if `ChannelQuantScaleParameter` ever acquires a
row/input-dim loader.

**Uneven-TP unit family.** The residual worry was that `in_proj_ba` (still
ignored -> `UnquantizedLinearMethod` -> keeps raw units, `linear.py:630-639`)
could diverge from the now-quantized `in_proj_qkvz`/`out_proj`, silently
desynchronising the head split. Checked live against the built artifact's own
config:

```
weight_block_size derived from gdncov config: [16, 16]
  gdn_tp_units basis   with-int8=16  no-quant=16  identical=True
  in_proj_qkvz         with-int8=16  no-quant=16  identical=True
  out_proj             with-int8=16  no-quant=16  identical=True
```

The int8 block is `lcm(8,16) = 16` (`w8a8_int8.py:169-195`), the GDN unit
elements-per-unit are 384 and 128 — both already multiples of 16 — so the
coarsening pass is a no-op and both branches land on the same 16. Additionally
`gdn_tp_units` is computed **once** per module (`qwen3_5.py:209-216`) and passed
by reference to conv1d, `in_proj_qkvz`, `in_proj_ba`, `out_proj`, `A_log` and
`dt_bias`, so the coupling is structural rather than re-derived. No divergence.

Loud guards that would catch a mistake at load time (none silent):
`utils.py:62-79` (packed-shard disagreement), `w8a8_int8.py:198-241`
(`verify_int8_scaled_mm_supports_shape`, per-rank N%8 / K%16),
`distributed/utils.py:1064-1068` (unalignable unit family),
`compressed_tensors_w8a8_int8.py:56-64` (sm<80), `:164` (unknown strategy).

**Not covered by this note:** `test_int8_w8a8_uneven_tp_align.py` could not be
run — it fails at pytest **collection** on `ModuleNotFoundError: No module named
'datasets'`, a pre-existing environment gap via the conftest import chain. The
file is untouched by this work and the incumbent tree fails it identically. The
three assertions it makes about the GDN unit family were instead reproduced
directly, live, against the artifact config (output above).

## 4. The KV-number reconciliation

The brief asked which of two published figures is right. **Both are — for
different comparisons — and neither is the number for this artifact.**

| comparison | freed | KV tokens | % of 457k pool | source |
|---|---:|---:|---:|---|
| incumbent -> lued W8A16 | 4,518.1 MiB | +144,579 | +31.6 % | ANALYSE_854 §3.3(b) |
| incumbent -> SmoothQuant W8A8 | 4,872.6 MiB | +155,923 | +34.1 % | ANALYSE_854 §3.3(c) |
| **incumbent -> gdncov (measured)** | **5,278.0 MiB** | **+168,897** | **+37.0 %** | this artifact |

ANALYSE_854 states both of its numbers correctly and attributes each to the
right checkpoint. The imprecision is in **NOTE_855 §3.4 and §3.5**, which
attach the ~156k figure to *"the GDN exclusion"* itself:

> "on top of the 4.87 GiB / +156k-token KV win the same exclusion also costs us"

That conflates two terms. 4,872.6 MiB is the delta to **SmoothQuant**, and
SmoothQuant reverts the MTP head to BF16 (810.0 MiB vs our INT8 405.1 MiB). So
**404.9 MiB of that 4,872.6 is an MTP regression, not GDN coverage** — a price
SmoothQuant pays and gdncov does not, because gdncov leaves the incumbent's INT8
MTP head untouched. Adding it back: 4,872.6 + 404.9 = 5,277.5 MiB, which matches
the measured 5,278.0 MiB to 0.5 MiB.

**So the pure GDN-coverage term is 5,278.0 MiB / +168,897 tokens / +37.0 %.**
NOTE_855's "+156k" understates what the exclusion actually costs by **12,974
tokens**. Where a single number is wanted for the GDN axis, use **+168,897**,
and quote +144,579 and +155,923 only with their checkpoints named.

Arithmetic: KV is 32,768 B/token (16 full-attention layers x 2 x 4 heads x 256
dim x 1 B fp8, ANALYSE_854 §3.1), so 1 MiB = 32 tokens exactly. Measured delta
36,353,564,128 - 30,819,147,232 = 5,534,416,896 B; / 32,768 = 168,897.

Per card at TP=3: ~1,759 MiB freed.

## 5. Red flags

1. **None on the quantization itself.** No outlier pathology (§3.2), error
   bound tight and met on all 144 tensors, no structural defect (§3.1), no
   silent-wrongness path (§3.4).
2. **The activation axis is untested and is the real quality risk.** NOTE_855
   §3.5 argued that in W8A8-vs-W8A16 comparisons the *activation* side is
   usually the dominant error term. This artifact puts 144 previously-BF16
   activation paths onto `per_token_quant_int8`
   (`compressed_tensors_w8a8_int8.py:213`). The weight error measured here
   (~1 %) does **not** bound that. The A/B must settle it.
3. **The GDN recurrent path is not a plain MLP.** `out_proj` consumes the
   gated-delta-net core output, whose dynamic range across a long sequence is
   less well-behaved than an MLP's. Per-token activation quant there is the
   single most plausible place for a quality regression to appear, and it is
   why §6 arm 3 asks for coherence across a **flip**, not just short prompts.
4. **Pre-existing:** the incumbent fails its own `crc32.txt` on 3 tokenizer/
   template files; gdncov inherits this unchanged (§3.1). Inert for the loader,
   but it means `crc32.txt` is not a usable integrity oracle for either
   checkpoint.
5. **Not measured here:** anything requiring a forward pass. Perplexity, KLD,
   accept length, tok/s — all of it is §6.

## 6. A/B window ticket — ready to run

Not scheduled by this note. Desk-prepared per ANALYSE_854 §9, same-boot floor
(#375), full feature set (graphs + spec + uneven-DCP/TP), never eager.

**Arms** — one boot each, same commit (Patchstand vor Last):

- **A** — `Qwen3.8-27B-INT8` (incumbent, GDN in BF16)
- **B** — `Qwen3.8-27B-INT8-gdncov` (this artifact)
- **C** — *optional, only if the quality axis is contested*
  `lued/Qwen3.8-27B-INT8-W8A16-MTP`. Vetoed on speed by NOTE_855 (2.05-2.27x vs
  A, 2.84-3.31x vs B), so it is a quality reference, not a candidate.

An **A-vs-A noise floor first** (Benchmark-Harness-Pflichten).

**Measurements**

1. **VRAM ledger / KV pool.** The direct falsifier of §4: expect
   **B = A + 168,897 KV tokens (+37.0 %)**, ~1,759 MiB/card at TP=3. If the
   measured delta disagrees, the footprint model is wrong and §4 is suspect.
   Verify the 819-1229 MiB NVML-free corridor under load (VRAM-Korridor-Regel);
   free = NVML-free, never total-used.
2. **Quality.** club-3090 quality suite + per-token KLD vs the BF16 teacher
   **through the serving path**, with A as an arm. This is the number that
   decides whether GDN coverage is free or paid for. Gate: B must not regress
   measurably against A.
3. **Spec acceptance from `meta_info`** — never `spec_ema_accept_len`
   (Spec-Acceptance-Messfalle). Reference band is **#774**: the defect drove
   accept length to **1.02** tokens/verify, healthy is **3.625-4.0**
   (`4e1b940900`). There is **no** "#779 gate threshold 2.0" — #779 adds a
   can-fail proof, not a numeric threshold (ANALYSE_854 §9.5). A and B carry the
   same INT8 MTP head, so a drop here means the GDN change disturbed drafting,
   which is a genuine finding rather than a head difference.
4. **Prefill tok/s**, bs=1 and bs=8, cold prefix (prefix caching off for this
   measurement only). NOTE_855 §3.4 predicts B is **1.39x (sm120) / 1.46x
   (sm86)** faster than A on linear-layer time.
5. **Decode tok/s**, bs=1 and bs=8, runs >= 10 s, ms/round per worker with
   COMPUTE vs WAIT split (ms-pro-Runde-als-Messlatte).
6. **#763-class coherence — the acceptance gate, both layouts.**
   (a) plain **TP=3 with uneven-DCP/uneven-TP active** (never disabled —
   Uneven-Verteilung-nie-abgeschaltet), and (b) **PP/flip layouts**, with at
   least **one full flip cycle under load** before any "stable" claim
   (WEDGE-RECOVERY abnahme-lehre; 8 short requests prove nothing). §3.4 shows
   the scale-sharding and unit-family hazards are closed in code and live at the
   config level, so this is the behavioural confirmation of a desk result, not a
   fishing expedition. Load via HiCache/radix session load, not manual fill
   batteries (Lastprobe-via-Session-Load).
7. **HiCache phase-uniformity**: no cache miss between tokens of the same model
   across the flip (HiCache-Phasen-Uniform-Pflicht).

**Decision rule.** If (2) shows no measurable quality regression and (6) is
clean in both layouts, gdncov replaces `Qwen3.8-27B-INT8` as the standard
checkpoint: it is strictly smaller (-5.16 GiB), strictly faster on prefill
(1.39-1.46x on linear layers), and gives back +168,897 KV tokens. If (2) does
show a regression, the follow-up is **calibrated int8 on the GDN projections
only** (AutoRound, or SmoothQuant restricted to `linear_attn` so the incumbent's
MLP/FA/MTP tensors are not disturbed) — *not* a return to BF16 GDN, whose price
NOTE_855 already measured. On present desk evidence that follow-up is not
indicated (§3.2).

## 7. Reproducing

```
CUDA_VISIBLE_DEVICES="" python3 tools/requant_gdn_int8_855.py --self-test
CUDA_VISIBLE_DEVICES="" python3 tools/requant_gdn_int8_855.py \
    --src /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8 \
    --dst /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov \
    --stats-out build_855_stats.json
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/python python3 -m pytest \
    test/registered/unit/quantization/test_gdn_int8_dispatch_855.py -q
```

`--dry-run` prints the plan (144 targets, 16 shards rewritten, 2 hardlinked, and
the ignore-list rewrite) without writing anything. Runtime was a few minutes on
CPU; peak RSS ~one shard.
