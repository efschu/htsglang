# Task 319 -- INT8 W8A8 as a prefill lever on the mixed rig (recon + design, no GPU)

Recon-only task, CPU-side (`CUDA_VISIBLE_DEVICES=99`), no card window spent, no
model downloaded. Written so a future GPU window can go straight to a falsifier
without re-deriving any of this.

## 0. The thesis under test

3080s (sm_86) carry native INT8 tensor cores at roughly 2x their FP16 peak
(GA102 datasheet: ~119 TFLOPS FP16 dense, ~238 TOPS INT8 dense) but no FP8
tensor path at all (`compute capability 8.6 has no fp8 tensor path (needs
8.9+)`, #213/#296). This fork's FP8 serving path on sm_86 therefore runs
through `fp8_marlin` (dequant-to-bf16-then-matmul on the Marlin GEMM) or the
plain `fp8_w8a16` dequant fallback -- both are BF16-tensor-core compute
underneath, not INT8. #252 measured the 3080s pacing prefill on a TP=3 FP8
boot (5090 computes ~3x shorter, waits ~390 ms longer). The question: would
switching the checkpoint format to INT8 W8A8 let the 3080s compute their share
of the prefill MLP on their strongest arithmetic path, and does that move the
needle against the measured collective-wait floor.

KV-int8 is out of scope per the brief (same byte width as fp8-KV, no
independent win, already closed).

## 1. Checkpoint landscape

**Verdict: checkpoints exist, no self-quantization needed.** HF Hub search
(`huggingface_hub.HfApi.list_models`, `/spinning/shvllm/.venv/bin/python`,
queries `Qwen3.6-27B`, `Qwen3.6 27B w8a8`, `Qwen3.6-27B int8`, `Qwen3.6-27B
smoothquant` (zero hits), `Qwen3.6-27B GPTQ-Int8` (zero hits), `AEON Qwen3.6`)
surfaced two checkpoints whose `config.json` `quantization_config` is a
genuine **dynamic per-token activation + per-channel weight INT8** scheme --
`format: "int-quantized"`, `input_activations.dynamic: true`,
`input_activations.strategy: "token"`, `weights.strategy: "channel"` -- which
is exactly the shape `CompressedTensorsW8A8Int8`
(`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`)
expects:

| repo | base model | shards | total size | tags | notes |
|---|---|---|---|---|---|
| `Avesed/Qwen3.6-27B-INT8-W8A8` | `Qwen/Qwen3.6-27B` (stock, not a finetune) | 2 files, `model.safetensors` + `mtp.safetensors` | 29.10 GiB | `compressed-tensors`, `int8`, `w8a8`, `vllm` | MTP draft weights cleanly separated into their own shard; best pick for a clean baseline against the existing FP8 benchmark corpus |
| `groxaxo/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-W8A8` | `llmfan46/...-Native-MTP-Preserved` (finetune) | 11 shards | 29.08 GiB | `auto-round`, `compressed-tensors`, `int8`, `w8a8`, `vllm`, `mtp` | Most-downloaded (28.5k) of the set, MTP explicitly preserved through quantization; quantized with the AutoRound *algorithm* but serialized in compressed-tensors *format*, so it loads through the same scheme code |

Both carry a `vllm` tag, i.e. someone has already round-tripped them through a
vLLM-family loader -- a real (if weak) signal the safetensors layout matches
what `CompressedTensorsW8A8Int8.create_weights` expects (int8 weight,
per-channel `weight_scale`, dynamic per-token activation, no `input_scale`
tensor needed).

Rejected near-misses, for the record (same search noise, different scheme):

* `TheHouseOfTheDude/Qwen3.6-27B-INT8`, `havenoammo/Qwen3.6-27B-INT8-MTP`,
  `lued/...-INT8-W8A16-MTP` -- `format: "pack-quantized"`, no
  `input_activations` block at all. These are **weight-only INT8 (W8A16)**:
  activations stay bf16/fp16, weight is dequantized before the matmul. They
  do not exercise `sgl_kernel.int8_scaled_mm` and would not test this
  thesis -- they are architecturally the INT8 analogue of `fp8_w8a16`, not
  of `fp8_native`.
* `Minachist/Qwen3.6-27B-INT8-Autoround-V2` -- AutoRound's own serialization
  (`autoround_version` key, mixed 4/8/16-bit per-module map, e.g.
  `linear_attn.in_proj_a/b` pinned to fp16), not compressed-tensors. Would
  need its own loader path; not evaluated further.
* `nameistoken/Qwen3.6-27B-Quark-W8A8-INT8` -- AMD Quark's own
  `quantization_config` shape (`algo_config`, no `config_groups`), not
  compressed-tensors either.

**Self-quantization recipe, kept as a fallback only** (not needed given the
above, sketched per the brief in case the two candidates fail correctness
checks on the card): `llm-compressor` `SmoothQuantModifier` +
`GPTQModifier(scheme="W8A8")` against `Qwen/Qwen3.6-27B` (bf16 source, ~54 GiB
on disk), calibration on ~512 sequences of a general-domain set (e.g.
`neuralmagic/LLM_compression_calibration` or the project's own prompt corpus
for content-axis parity with existing benchmark data, per the
Benchmark-Harness-Pflichten memory note). Expected wall time on this rig:
GPTQ's Hessian pass is the expensive part, single-GPU, no TP -- realistically
a multi-hour CPU-adjacent job best run on the 5090 alone; RAM: the bf16 source
checkpoint plus activations cache is the binding constraint (fits inside this
host's headroom per prior GGUF conversions of the same model family, but
budget it explicitly before starting). Not worth doing before the microbench
in Section 4 clears its own bar.

## 2. Fork-stand matrix

### 2a. Quantization classes present

`w8a8_int8` is registered both as a standalone quant method
(`python/sglang/srt/layers/quantization/w8a8_int8.py`, `W8A8Int8Config`,
`get_name() == "w8a8_int8"`) and as a compressed-tensors scheme
(`compressed_tensors_w8a8_int8.py`, `CompressedTensorsW8A8Int8`) reached via
`--quantization compressed-tensors` autodetection off `config.json`
(`python/sglang/srt/layers/quantization/__init__.py:91,134`, entry
`"compressed-tensors": CompressedTensorsConfig`). Both HF candidates in
Section 1 autodetect through the compressed-tensors path, so the reference
launch command's `--quantization compressed-tensors` flag needs no change to
target them.

### 2b. Kernel, per architecture

Single kernel for the dense linear path: `sgl_kernel.int8_scaled_mm`
(CUTLASS 2.x/3.x, `sgl-kernel/csrc/gemm/int8_gemm_kernel.cu`), called from
both `W8A8Int8LinearMethod.apply` and `CompressedTensorsW8A8Int8.apply_weights`
after a Triton dynamic per-token activation quant
(`per_token_quant_int8`, `sglang/srt/layers/quantization/int8_kernel.py` --
arch-general, no gap here).

> **Resolved (task #327 pre-stage, commit `7da6f0cb2f`).** The sm120 arm now
> exists; see `TASK_327_INT8_SM120_WHEEL.md`. The section below describes the
> state before that commit and stays as the record of how the gap was found.

**`int8_scaled_mm`'s own SM dispatch (`int8_gemm_kernel.cu:699-744`) is
explicit and closed-ended:**

```
sm_version = getSMVersion();               // major*10 + minor, e.g. 86, 120
if (75 <= sm < 80)      sm75_dispatch_shape<...>(...);
else if (80 <= sm < 90) sm89_dispatch_shape<...>(...);  // sm86/89 branch
else if (sm == 90)      sm90_dispatch_shape<...>(...);  // Hopper, cutlass 3.x
else                    TORCH_CHECK_NOT_IMPLEMENTED(false,
                            "No implemented int8_scaled_mm for current compute capability.");
```

| card | this rig's SM | dispatch branch | result |
|---|---|---|---|
| RTX 3080 x2 | 86 | `sm_version >= 80 && sm_version < 90`, `sm_version == 86` sub-branch (`sm89_dispatch_shape`, small-shmem Ampere/Ada tuning) | **native INT8 tensor-core GEMM, this is the thesis's target lane** |
| RTX 5090 | 120 | falls through every branch (`sm == 90` is an exact match, not `>=`; there is no `sm >= 100` or `sm >= 120` arm) | **`TORCH_CHECK_NOT_IMPLEMENTED` at the first forward call.** No Blackwell int8 kernel exists anywhere in the tree (`grep -rn sm100\|sm120 sgl-kernel/csrc/gemm/*int8*` -- zero hits; there is exactly one int8 gemm source file, no per-arch sibling like the fp8 lanes have) |

This is a **hard architectural blocker, not a performance question**: under
pure TP every rank loads a shard of the SAME int8-dtype weight tensor and must
run it through SOME int8-capable kernel. On this rig's 5090 rank there
currently is none -- neither a native path (missing dispatch branch) nor a
fallback (see 2c). A W8A8 INT8 checkpoint would crash the 5090 rank's first
forward, not merely run it slower.

Compounding this: `CompressedTensorsW8A8Int8.get_min_capability()` (line 122
of the scheme file) returns **80** ("ampere and up") -- sm_120 satisfies
`120 >= 80` and is admitted by whatever capability gate reads this method, so
nothing at plan time flags the 5090 as unsupported. The gate is honest about
the *scheme's* origin (Ampere+ compressed-tensors int8 shipped upstream before
Blackwell existed) but wrong about *this fork's kernel's* actual coverage.
The failure surfaces late -- after weight load, at the first real forward --
exactly the failure mode the `_FORMAT_LANES` mechanism (2c) and the lane
probe's "ask functionally, don't infer from a capability integer" principle
(`uneven_perf.py:775-778`, `_bench_gemm_fp8_native_tflops` docstring) was
built to convert into a loud, pre-boot fact instead of a late crash.

### 2c. No dequant fallback lane exists for INT8 (unlike FP8's three-lane ladder)

> **Superseded for the 5090 (task #327 pre-stage).** This section's cost estimate
> was wrong in one direction: the native branch was not "a materially bigger and
> higher-risk undertaking" but one forwarding template plus an `else if`, because
> the INT8 SASS for sm_120 was already being emitted from the sm86 arm. The
> dequant fallback lane is still the answer for cards with no IMMA path at all
> (sm100/sm103); it is no longer needed to make this rig bootable on INT8.
> See `TASK_327_INT8_SM120_WHEEL.md`.

FP8 has `LANE_FP8_W8A16`
(`_bench_gemm_w8a16_tflops`, `uneven_perf.py:874-913`, using
`fp8_utils.dequant_fp8_block_weight`) -- a card with no fp8 tensor path at all
still runs the checkpoint, dequantizing fp8 weights to bf16 per forward and
using the card's dense bf16 units. This is what keeps the 3080s bootable on
an FP8 checkpoint today.

**No equivalent exists for int8.** `grep -rn "dequant.*int8\|int8.*dequant"
python/sglang/srt/layers/quantization/*.py` is empty. If the 5090's missing
`int8_scaled_mm` branch (2b) is to be worked around cheaply rather than by
writing a new CUTLASS Blackwell kernel, someone has to write an
`dequant_int8_channel_weight`-style helper (per-channel int8 weight -> bf16,
mirroring `dequant_fp8_block_weight` but with a per-output-channel scale
instead of a 2-D block grid -- simpler, actually, since INT8 W8A8 here is
per-channel not per-block) and wire it into a new `LANE_INT8_W8A16`-equivalent
fallback. This is the **minimum viable fix to make the checkpoint bootable at
all** on this rig, independent of whether the native lane's speedup ends up
mattering (Section 4). Rough size: mirrors an existing ~40-line function plus
its lane-probe wrapper and `_FORMAT_LANES` entry -- half a day, not a new
kernel-authoring project. The alternative (writing a real CUTLASS sm100/sm120
int8 dispatch branch, matching the effort that produced the existing sm90
branch) is a materially bigger and higher-risk undertaking and is not
justified before Section 4's verdict.

### 2d. Alignment family: where uneven TP would break INT8, and why it currently would

`int8_scaled_mm`'s own `TORCH_CHECK`s (`int8_gemm_kernel.cu:677-679`) are:

```
mat_a.size(1) % 16 == 0   // K, both operands
mat_b.size(0) % 16 == 0   // K
mat_b.size(1) % 8  == 0   // N (output_size_per_partition)
```

This is a genuinely **coarser** (easier to satisfy) constraint than the
existing alignment families this fork has already hit and fixed -- AWQ/GPTQ
Marlin's `min_thread_n = 64` / `min_thread_k = 128` (#289, #300), FP8's
`weight_block_size = [128, 128]` block-scale grid. 16/8 divides into all of
those. The problem is not that the requirement is strict; it is that **the
uneven-TP shard-coarsening machinery that exists specifically to satisfy this
class of constraint does not fire for this quant config at all today:**

`CompressedTensorsConfig.weight_block_size`
(`compressed_tensors.py:277-291`) -- the property `_quant_block_aligned_units`
(`layers/linear.py:156-188`) reads to decide how much to coarsen a
`--rank-mlp-ratio`/`--rank-tp-ratio` shard -- returns a value only when the
scheme has a `block_structure` (FP8 block quant) or a `group_size`
(AWQ/GPTQ INT4, via the `_group_size_block` fallback, `compressed_tensors.py:
133-199`). Both HF candidates in Section 1 use **`strategy: "channel"`** with
neither `block_structure` nor `group_size` -- `weight_block_size` returns
`None` for them, `_quant_block_aligned_units` short-circuits at
`if not block: return units` (`linear.py:184-185`), and **no coarsening
happens.** Per-rank `input_size_per_partition` / `output_size_per_partition`
land wherever the base (fine-grained) unit split puts them, with no guarantee
of a multiple of 16 or 8.

This is the exact shape of the bug class `_group_size_block`'s own docstring
names as precedent: *"Without it, an uneven `--rank-tp-ratio` split produces
`input_size_per_partition % group_size != 0`... this is what blocked AWQ
under uneven TP; even TP happened to divide cleanly."* The standalone
`W8A8Int8Config` (`w8a8_int8.py`) has the identical gap -- it defines no
`weight_block_size` attribute at all, so `getattr(quant_config,
"weight_block_size", None)` returns `None` unconditionally, same
no-coarsening outcome.

**Fix shape** (design only, not implemented here): extend
`CompressedTensorsConfig.weight_block_size` (and give `W8A8Int8Config` the
same attribute) to recognize a CHANNEL-strategy int8 scheme and return an
alignment block representing the *kernel's* requirement rather than a
quantization-semantic block -- `[8, 16]` (block_idx 0 = output/N mult-8,
block_idx 1 = input/K mult-16), reusing the exact same `block_aligned_units`
plumbing `_group_size_block` already established for the Marlin family. Note
for whoever implements this: unlike FP8's block or AWQ's group, this block
carries no quantization meaning at all (channel-strategy scales are already
per-output-row, insensitive to how the row is split) -- it exists purely to
keep the CUTLASS kernel's own alignment check satisfied under an uneven
split. Worth a one-line comment distinguishing the two motivations so a
future reader does not go looking for a scale-grid reason that is not there.

Until this exists, `--rank-tp-ratio auto-performance` / `--rank-mlp-ratio`
on an INT8 W8A8 checkpoint will boot correctly by luck on splits that happen
to land on 16/8-aligned boundaries and abort inside the CUTLASS kernel (late,
mid-first-forward, the same failure class as 2b) on splits that do not --
which on Qwen3.6-27B's dimensions is not guaranteed for an arbitrary
MLP-unit vector.

### 2e. Summary table

| concern | fp8 (today) | int8 w8a8 (proposed) |
|---|---|---|
| native lane on 3080 (sm_86) | none (`needs 8.9+`) | **yes** -- `sm89_dispatch_shape` branch, this is the whole thesis |
| native lane on 5090 (sm_120) | yes, `torch._scaled_mm` | **no dispatch branch at all -- hard crash, not a slow path** |
| universal fallback lane | yes, `fp8_w8a16` dequant | **does not exist yet** -- must be written before the checkpoint is bootable on this rig at all |
| capability gate honesty | n/a (functional probe, no static gate) | `get_min_capability()==80` wrongly admits sm_120 |
| uneven-TP shard alignment | handled (`weight_block_size=[128,128]`) | **not wired** -- `weight_block_size` returns `None` for channel-strategy int8, no coarsening fires |

## 3. Lane design: `int8_native` in `uneven_perf.py`

### 3a. Format-key detection

`checkpoint_compute_format` (`uneven_perf.py:1733-1765`) needs a new
predicate parallel to `_is_fp8_like` (line 1723), keyed off the same
`(quant_method, format)` pair read from `config.json`'s
`quantization_config`:

```python
def _is_int8_w8a8_like(method: str, fmt: str, qc: dict) -> bool:
    """True for a genuine dynamic-activation INT8 scheme (int8_scaled_mm's
    target), false for weight-only INT8/W8A16 (dequant-and-matmul, no
    tensor-core benefit from this lane table).

    compressed-tensors' own `format` field already distinguishes the two on
    every checkpoint surveyed for #319: "int-quantized" (unpacked int8
    bytes, paired with a real `input_activations` block, dynamic per-token)
    vs "pack-quantized" (weight-only, no `input_activations` at all). Trust
    the input_activations block over the format string -- it is what
    actually determines which kernel runs -- and require dynamic + 8-bit so
    a static-scale or int4 variant does not silently take this lane."""
    if method not in ("compressed-tensors", "compressed_tensors", "w8a8_int8"):
        return False
    groups = (qc.get("config_groups") or {}).values()
    for g in groups:
        ia = g.get("input_activations") or {}
        if ia.get("dynamic") and int(ia.get("num_bits", 0)) == 8:
            return True
    return method == "w8a8_int8"  # standalone scheme has no config_groups;
                                   # its existence in server_args IS the fact
```

`_FORMAT_LANES["int8"] = (LANE_INT8_NATIVE, LANE_INT8_W8A16)` -- native first
(mirrors the fp8 dispatch order and the serving path's own preference), the
w8a16 fallback second, once 2c exists. Until 2c exists, ship
`_FORMAT_LANES["int8"] = (LANE_INT8_NATIVE,)` alone: on the 5090 today no
lane is available at all, and `rank_gemm_scores`'s existing "no lane measured
on this card" branch (`uneven_perf.py:1816-1829`) already produces the
correct LOUD bf16-fallback-with-warning behavior for that rank -- it costs
nothing extra to wire up, and it is honest about the real gap rather than
inventing a lane the serving path cannot use yet either.

### 3b. What the probe measures

A new `_bench_gemm_int8_native_tflops(dev)`, same shape and signature
contract as its fp8 siblings (`_PROBE_GEMM_M/K/N = 2048, 5120, 17408`, same
warmup/iteration counts, `(tflops_or_None, note)` return): build int8 `mat_a`
(`M, K`) and `mat_b` (`N, K` transposed to `K, N` column-major, matching
`int8_scaled_mm`'s layout requirement at line 674-675 of the kernel), unit
`float32` `scales_a`/`scales_b`, call `sgl_kernel.int8_scaled_mm` directly --
same "ask functionally, don't infer from a capability integer" principle the
fp8 lanes already follow, so the sm_120 gap in 2b surfaces as a probe NOTE
(`"int8 GEMM did not run: RuntimeError: ... No implemented int8_scaled_mm
for current compute capability."`) rather than a silent number, exactly the
failure-to-fact conversion this table exists for. That note is then exactly
what feeds `rank_gemm_scores`'s per-card lane selection: **on this rig the
3080s dispatch `int8_native`, the 5090 dispatches nothing until 2c ships and
falls back to bf16-with-warning** -- the planner does not need a separate
"choose fp8-marlin vs int8-native" decision procedure; it is the same
first-available-lane-per-card mechanism already built for fp8, driven purely
by `checkpoint_compute_format` returning `"int8"` instead of `"fp8"` for
these checkpoints.

### 3c. `PROFILE_VERSION`: no bump, and that is the deliberate choice

The new lane keys (`int8_native`, `int8_w8a16`) slot into the **already
version-3, already-declared** `gemm_lanes` / `gemm_lane_notes` fields
(`_PROFILE_VERSION_FIELDS[3]`, `uneven_perf.py:135`) as new dict entries, not
new top-level fields. `migrate_profile` (`uneven_perf.py:611-629`) checks
field-name presence (`f not in entry`), not sub-key completeness, so this
requires **no `PROFILE_VERSION` bump**.

This is a deliberate reading of the #303 lesson, not an oversight: bumping
`PROFILE_VERSION` changes the cache key itself
(`profile_cache_path`, `uuids/driver/PROFILE_VERSION` hashed into the file
name, `uneven_perf.py:583-587`), which invalidates **every** cached profile
on every rig and forces a fresh full probe on the next
`--rank-tp-ratio auto-performance` boot -- including the pairwise NCCL link
matrix, the slowest and most failure-prone phase (`_link_worker` /
`_create_c10d_store`). That is exactly the mechanism that cost 600 s per boot
in the Welle-2 window when `PROFILE_VERSION` went 2->3 for the fp8 lanes
themselves (`docs/dev/INTEGRATION_R3_VALIDATION.md:12033-12045`). Adding the
int8 lanes as new keys within the *existing* v3 field names sidesteps that
entirely: the cache path does not change, nothing re-probes the link matrix.

Cost of not bumping: a rig with an already-cached v3 profile (written before
this feature existed) will not automatically pick up the int8 lane -- it
already "has" the `gemm_lanes` field, so `migrate_profile` reports no gap for
it. This degrades exactly the same way the existing marlin-throws-on-sm86
case does today: `rank_gemm_scores` reports "not probed" per lane in its
warning and falls back to bf16 for that rank, pointing at
`SGLANG_PERF_REPROBE=1` -- the established, already-shipped UX for this
situation, not a new one. Given the choice between a known 600 s tax on every
rig's next boot and a one-time `SGLANG_PERF_REPROBE=1` for whoever wants the
new lane measured, the second is the correct default.

## 4. Expectation calculation and verdict

**Anchor data.** #252 (`docs/dev/INTEGRATION_R3_VALIDATION.md:4720-4735`),
CollectiveClock split, TP=3 FP8 boot, cold 21765-token prefill, steady chunk:

| rank | card | compute (ms) | wait (ms) | window (ms) | wait share |
|---|---|---:|---:|---:|---:|
| TP0 | 5090 | 196.6 | 1641.1 | 1837.7 | 89.3 % |
| TP1 | 3080 | 586.5 | 1251.0 | 1837.5 | 68.1 % |
| TP2 | 3080 | 558.5 | 1279.3 | 1837.8 | 69.6 % |

The task brief's "68 % collective floor" is the 3080s' wait share specifically
(TP0's is 89.3 %, not part of that floor -- it is idle-waiting on the 3080s
and the collective, not itself collective-bound in the same sense). #252's own
reading: *"wait is ~68% of the window on EVERY rank [that matters here] --
collective cost, not skew... only the ~390 ms imbalance [TP0's excess wait
over TP1/TP2's] is recoverable by a shard rebalance."* That sentence already
draws the line this section needs: the ~390 ms gap is a WORK-SPLIT lever
(`--rank-mlp-ratio`, already exploited by #296's `10,1,1` prefill optimum,
+13.6-14.2 % measured), separate from and additive to whatever a
compute-throughput lever like INT8 could do to the 3080s' own 586.5/558.5 ms
compute bucket.

**Upper-bound arithmetic.** Take the optimistic case: 100 % of the 3080's
586.5 ms "compute" bucket is int8-eligible GEMM (in reality it also includes
attention, RMSNorm/rope, and any bf16-resident sub-layers -- an over-estimate
in this fork's favor), and the achieved INT8 lane hits the full nominal 2x
of the card's own achieved bf16 rate (62-63 TFLOPS measured, #298b table)
rather than a partial fraction of it. That halves the 3080 compute bucket to
~293 ms. If the collective's wait portion is genuinely fixed in absolute
terms (bandwidth-bound transport on a no-P2P/PHB rig, "collective cost, not
skew" -- not merely waiting on the 3080 to finish, since TP0 already waits
89.3% while computing almost nothing), the window shrinks by the compute
delta: 1837.6 -> ~1544 ms, an **upper bound of ~16 %** on this specific
prefill chunk. The wait *fraction* of the new, smaller window would then rise
to ~81 % -- consistent with #252's own framing that the collective floor does
not move, it just eats a larger share of whatever compute time remains.

**Why the realistic number is well under that upper bound:**

1. **The lane comparison point is not `bf16`, it is the checkpoint's own
   current lane.** #298b measured the 3080's fp8 dispatch lane
   (`fp8_w8a16`, dequant-then-bf16-matmul) at 53.5-53.6 TFLOPS -- *slower*
   than the card's own plain dense bf16 GEMM (62.2-63.2 TFLOPS), because it
   pays block-scale dequant overhead on top of the same underlying bf16
   matmul. The uneven_perf module comment records the marlin-vs-dequant gap
   directly: marlin is measured "-8.1 % [behind marlin]" at prefill for the
   dequant lane (`uneven_perf.py:884`), i.e. marlin ~1.09x over w8a16 -- the
   brief's "~1.11x" is this same figure, not a separately measured number
   (fp8-marlin itself never successfully measured on this rig; see 2c/#298b,
   it threw on a foreign-worktree JIT cache, a build artifact not an
   architecture fact). So the honest ceiling for "what INT8 must beat" is
   ~58-63 TFLOPS (whichever fp8 lane a correctly-built rig actually lands
   on), not some slower number -- and INT8's 2x-of-bf16 nominal ceiling
   (~120-126 TFLOPS-equivalent) is a ~2x win over THAT baseline, matching the
   upper-bound arithmetic above, not exceeding it.
2. **Not all "compute" is MLP GEMM.** W8A8 int8 only covers `Linear` targets
   (`compressed_tensors_w8a8_int8.py` scheme registration, `targets:
   ["Linear"]`) -- attention QKV/O and MLP gate/up/down, yes; GDN linear-attn
   in/out projections, yes (per the Minachist config's per-module bit map,
   these are exactly the layers third-party quantizers also target at 8
   bits); but softmax, RMSNorm, RoPE, and the GDN scan itself stay in their
   existing dtype regardless of weight format. A realistic GEMM share of the
   586.5 ms compute bucket is well below 100 %.
3. **Real CUTLASS int8 GEMM efficiency rarely hits the full 2x-of-FP16
   nominal ratio at a fixed shape** -- per-token/per-channel scale epilogue
   overhead, and a probe shape (2048x5120x17408) chosen to represent the MLP
   FFN specifically, not the smaller attention-projection GEMMs that make up
   part of the same compute bucket at a less favorable arithmetic intensity.
4. **Two hard prerequisites gate ANY of this being measurable at all**: the
   5090 needs a working lane (2c, ~half a day) before the checkpoint even
   boots on this rig's TP=3 layout, and the alignment gap (2d) needs closing
   before an uneven `--rank-mlp-ratio` split is safe to use with this format
   -- both are prerequisite engineering, not part of the payoff.

**Verdict: plausible but modest, and gated behind cheap prerequisites --
falsify with a microbenchmark before investing further.** The mechanism is
real (3080 does have native int8 tensor cores the current fp8 path cannot
reach) and the direction is right, but the realistic gain against the
measured #252 window is closer to high-single-digit to low-double-digit
percent of *prefill* time, not the 2x the raw hardware ratio suggests, and it
does nothing for the 68 % collective floor itself or for decode (W8A8 int8's
activation-quant overhead is a pure prefill/large-batch win; at bs=1 decode
the per-token dynamic quant is dead weight with no compensating GEMM
efficiency gain, likely a small regression, same shape as fp8_w8a16's
-69.9 % decode/-8.1 % prefill asymmetry noted in `uneven_perf.py:884`).
Given the effort floor (5090 dequant-fallback lane + alignment-block fix, both
required just to reach a bootable, uneven-TP-safe state, BEFORE any of the
above payoff is even realizable), the correct next step is the microbenchmark
in Section 5(a) -- an isolated `int8_scaled_mm` vs `int8_w8a16`-equivalent
dequant timing at the probe shape on an actual 3080, cheap and fast (~seconds
of card time, same cost class as the existing #298b lane probe) -- BEFORE
writing the 5090 fallback lane, the alignment fix, or touching a real
checkpoint. If the microbench does not clear a solid margin over the existing
fp8 dispatch lane's TFLOPS number, stop there; the checkpoint bring-up and
full-model A/B are not worth the engineering floor above.

## 5. Measurement plan (design only, not run)

### 5(a) -- Cheapest possible falsifier: isolated lane microbench (no checkpoint needed)

Runs entirely inside the existing lane-probe machinery, on a real card, in
seconds -- no model load, no checkpoint download. This is the correct FIRST
card-window item, before anything else in this document:

1. Land the `_bench_gemm_int8_native_tflops` probe function (3b) as a
   standalone script mirroring `lane_probe_only.py` (the tool #298b already
   used to sidestep the hanging link-matrix phase, per
   `docs/dev/INTEGRATION_R3_VALIDATION.md:12094-12098`).
2. Run it on the 3080s and the 5090 (resolve physical indices via NVML at
   run time per this fork's standing hardware-inventory rule, never by a
   fixed assumption).
3. Compare against the already-cached #298b table (bf16 62.2/63.2/233.3,
   fp8_w8a16 53.5/53.6/178.4, fp8_marlin throws on 3080 / 216.6 on 5090) at
   the SAME probe shape, same warmup/iteration counts -- a same-run, same-rig
   ratio, not a cross-session comparison.
4. Decision gate: if 3080 `int8_native` TFLOPS clears the existing fp8
   dispatch lane (53.5-53.6) by a wide, noise-floor-clearing margin (expect
   it trivially will, per Section 4's arithmetic -- the interesting number is
   HOW wide, i.e. whether it is closer to 1.5x or 2x the current lane, since
   that number is what feeds the #252 window arithmetic), proceed to 5b's
   checkpoint bring-up. If it does not run at all (unexpected -- sm_86 sits
   inside the kernel's own explicit `sm89_dispatch_shape` branch, but the
   marlin lane's foreign-JIT-cache failure in #298b is a live reminder to
   check the error is architectural and not another build/cache artifact
   before concluding anything), stop and diagnose the build, not the
   architecture.

### 5(b) -- If 5(a) clears the bar: full-checkpoint boot comparison

Prerequisite work (Section 2c dequant fallback for the 5090, Section 2d
alignment-block fix) must land first, or this arm is limited to TP configs
that happen to keep every rank's shard 16/8-aligned by luck, and the 5090
rank simply cannot boot at all without 2c.

* Vehicle: `Avesed/Qwen3.6-27B-INT8-W8A8` (Section 1), same reference launch
  command as the CLAUDE.md baseline but `--quantization compressed-tensors`
  against this checkpoint path instead of the AWQ one, TP=3.
* Arm A (even-ish baseline): `--rank-tp-ratio auto-performance` (VRAM-auto
  split, mirrors the #296 anchor).
* Arm B (genuine multi-rank-per-GPU, per this fork's standing hardware rule):
  TP=4 with two ranks pinned to the 5090 (resolve its physical index via
  NVML at run time, never assumed) and one rank each on the two 3080s --
  this ALSO forces the 2c fallback lane to actually run under real
  co-residence pressure, not just solo.
* Measure: per-rank CollectiveClock compute/wait split (#252 instrumentation,
  already wired into the "Prefill rank batch" log line), same cold-prefill
  chunk size as the #252 anchor for a like-for-like comparison, plus TTFT and
  prefill tok/s at s=1 and s=8 (same shape as #296's table).
* Correctness gate before any perf number counts: short-generation coherence
  + accept-length sanity across 5 prompts (the same battery #289 ran), since
  this is a genuinely new dequant/kernel combination on this fork, not a
  config flip.

### 5(c) -- Separate posten: spec-to-decode aggregate at bs=8, bar1 operating point

Independent of (a)/(b) -- this is the standing decode-aggregate measurement
the brief asks to keep current, not an int8-specific probe. Recipe: the
`2026-07-30_phasen_optima` anchor boot (`--rank-tp-ratio auto-performance`
base split, `--rank-mlp-ratio` / `--rank-kv-ratio 7,3,3` pinned, BAR1
transport, reserve `4500,4200,4200`, `--decode-log-interval 1`), bs=8,
measuring `ms/Verify` (accept_len * 1000 / gen_tok_s, per-request) and
aggregate tok/s (Tick), using the `s14_decode_punkt.py` prefilled-not-grown
window methodology (2048-token warmed prefix, `ignore_eos` +
context-derived `max_new_tokens` to hold `#running-req` fixed, middle-cut
window). Reuse the established noise floors verbatim rather than
re-deriving them: **ms/Verify 2.72 %, tok/s (Tick) 7.53 %**
(`docs/dev/INTEGRATION_R3_VALIDATION.md:11794-11816`, six-sample A-vs-A at
bs=16, carried forward per the brief). Report nothing under those floors.

## 6. Bottom line

* **Checkpoints**: solved, no self-quant needed --
  `Avesed/Qwen3.6-27B-INT8-W8A8` (stock base model, MTP-separated) is the
  pick, `groxaxo/...-Native-MTP-Preserved-W8A8` a viable second (finetune,
  most-downloaded, MTP explicitly preserved).
* **Fork stand**: the compute kernel exists and is correctly the 3080's
  strong lane (`int8_scaled_mm`'s `sm89_dispatch_shape` branch), but **the
  checkpoint would not boot on the 5090 rank of THIS rig today** (no sm_120
  dispatch branch, no fallback lane) and **uneven-TP shard alignment is not
  wired for this quant strategy** (`weight_block_size` returns `None` for
  channel-strategy int8, unlike FP8/AWQ). Both are concrete, small, located
  fixes (Sections 2c/2d), not open research. `get_min_capability()==80` is a
  latent trap that would let a boot attempt get past plan-time checks before
  crashing at first forward (Sections 2a-2e).
* **Lane design**: a clean extension of the existing `_FORMAT_LANES`
  machinery, no `PROFILE_VERSION` bump needed -- new dict keys inside
  already-declared v3 fields, deliberately avoiding a repeat of #303's 600 s
  link-matrix tax.
* **Expectation**: upper bound ~16 % of one measured prefill window's wall
  time, realistically less once GEMM-share-of-compute and achieved (not
  nominal) INT8/BF16 efficiency ratios are accounted for; the 68 %
  collective floor itself is untouched by any compute-throughput lever, per
  #252's own reading. Real, right direction, modest.
* **Recommended next step**: the Section 5(a) microbenchmark ONLY -- cheap,
  fast, falsifies or confirms the core throughput number before either
  prerequisite fix (2c, 2d) or a full checkpoint bring-up is attempted.
