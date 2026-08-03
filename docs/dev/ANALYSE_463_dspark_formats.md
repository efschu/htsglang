<!-- Adopted into the tree by task #470 (2026-08-03) as the authority for the
DSpark format/placement question; written under task #463. Body verbatim.
Recommendation R1 (§5) is the ticket #470 executes; the R4 note in §5 is the
standing "banked, negative yield on this rig" entry for lossless MXFP4->FP8. -->

# Task #463 — a DSpark draft head for DeepSeek-V4-Flash in a format this rig can run

Desk + network research, 2026-08-03. **No GPU touched** (every python invocation
under `CUDA_VISIBLE_DEVICES=99`; no `/spinning/gpu-arb/` window taken). No repo
commits; no worktree created.

Catalog sections read (`/spinning/wt-merge-ops/docs/dev/FEATURE_CATALOG.md`):
**§3** memory tiers / offload / spill, **§4** speculative decoding, **§8** GGUF
stack, **§9** quant lanes, **§17** META combination matrix. Also read:
`ANALYSE_447_llamacpp_dsv4_harvest.md` (full), `ANALYSE_456_dsv4f_matrix_sweep.md`
§6-§7, `ROADMAP_456_matrix_execution.md`, `docs/rig-runbook.md` §4.5.4 / §4.5.4b,
`/spinning/gpu-battery-results/2026-08-03_447_dspark/prompts.json`,
`python/sglang/srt/planner/rejected.py`.

Trees cited: **ours** = `/spinning/wt-merge-ops` (integration tip, `661f1f5c78`).
Local artifacts = `/spinning/llm_stuff/club-3090/models-cache/`.

---

## 0 — The short answer to the user's question

> *"Isn't there another format besides MXFP4? It should exist."*

**Yes — four of them exist, and one is bit-exact.** But none of them is the
reason the #447 arm did not boot on this rig, and only one of the four actually
helps here. The honest ordering of the finding:

1. **MXFP4 is the source of truth, not a repack.** DeepSeek ships the DSpark
   routed experts as MXFP4 *inside the original checkpoint*. There is no BF16
   or FP8 master anywhere public. Everything else in the world is derived from
   these bytes.
2. **A bit-exact FP8 representation exists and our own tree can produce it**
   (`cast_e2m1fn_to_e4m3fn`, verified below at max-rel-err 0 on real -0731
   data). It is **1.88x larger**, so it makes the VRAM problem worse.
3. **Q2_K GGUF requants of the -0731 head exist** (6.49 GiB, −3.66 GiB) but
   under **four mutually incompatible community arch strings**, three of which
   have no consumer at all.
4. **The 5090 already has a native MXFP4 kernel path.** `Mxfp4MarlinMoEMethod`
   accepts SM120 (`mxfp4_marlin_moe.py:116-117`, `utils/common.py:642`). For a
   draft that lives only on the 5090, **the format question does not bite** —
   what blocks that arm is a placement refusal and a VRAM budget, not a dtype.

So the correct reframing: the format hunt succeeded, and it is **not** the
cheapest path. Section 5 ranks them.

---

## 1 — What the local -0731 head actually contains (measured, not inferred)

`/spinning/llm_stuff/club-3090/models-cache/DeepSeek-V4-Flash-0731-dspark-head-filtered/`
(symlinks into `.../DeepSeek-V4-Flash-0731-dspark-head/`, shards 46-48 of 48 +
a filtered `model.safetensors.index.json`). Safetensors headers read directly:

| dtype | tensors | bytes |
|---|---:|---:|
| `I8` (MXFP4 nibble pairs) | 2 304 | **9.000 GiB** |
| `F8_E8M0` (block scales) | 2 329 | 0.563 GiB |
| `F8_E4M3` | 25 | 0.416 GiB |
| `BF16` | 20 | 0.129 GiB |
| `F32` | 27 | 0.009 GiB |
| **total** | **4 705** | **10.117 GiB** |

Routed experts = `mtp.{0,1,2}.ffn.experts.{0..255}.w{1,2,3}`: `I8 [2048, 2048]`
(w1/w3) and `I8 [4096, 1024]` (w2) with `F8_E8M0 [·, ·/32]` scales — i.e. 4-bit
E2M1 values, one E8M0 exponent per 32 elements. That is MXFP4, 9.5625 GiB of
the 10.117.

> ### CONTRADICTION — flag against `ANALYSE_447_llamacpp_dsv4_harvest.md` §1.5
> That section states the three `mtp.` shards are *"10.12 GiB, fp8 (`.weight` +
> `.scale` pairs, `quantization_config.fmt = e4m3`, `scale_fmt = ue8m0`, block
> `[128, 128]`)"*. **The size is right; the dtype claim is wrong for the routed
> experts.** It was inferred from the top-level `quantization_config` in
> `config.json` (which describes the *target's* non-`mtp` tensors) rather than
> read per tensor. Only the 25 `F8_E4M3` tensors (attention, shared experts,
> `main_proj`) are fp8 block-`[128,128]`; the 2 304 expert tensors are MXFP4.
> §1.5's conclusion *"Effort to use it: no loader change"* rests on that wrong
> dtype and is the root of the #447 NO-GO. **Suggested correction to
> `ANALYSE_447` §1.5 and to adoption candidate A in §4.**
>
> Corollary that survives: am17an's GGUF *is* a bit-exact repack — its 9 MXFP4
> tensors total exactly 9.562 GiB, matching the checkpoint's 9.000 + 0.563.

Also measured (36 sampled expert scale tensors, all three stages): the **E8M0
exponent span within a tensor is 2-5** (typically exactly 3 distinct scale
values, exponents −6..−4). Load-bearing for §3.3: an E4M3 scale field
(2^−9..2^8) covers that range with room to spare, so an MXFP4→NVFP4 re-encode
is exactly lossless on this checkpoint, not merely approximately.

---

## 2 — Route (a): existing artifacts

### 2.1 Official DeepSeek / NVIDIA releases — nothing non-MXFP4

| repo | shape | verdict |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `mtp.*` in shards 46-48, **MXFP4 experts** | the source; what we have locally |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` (0704) | same shape | older revision, same format |
| `nvidia/DeepSeek-V4-Flash-NVFP4` | 46 shards, `moe_quant_algo: NVFP4`, group 16 | **`hf_quant_config.ignore` contains `"mtp.*"`** — NVIDIA deliberately leaves the DSpark head untouched. There is no NVFP4 DSpark head. |
| `nvidia/DeepSeek-V4-Pro-NVFP4` | same policy | — |

The NVIDIA exclusion list is the strongest external evidence that **no vendor
has produced a non-MXFP4 DSpark expert stack.** Upstream sglang PR
[#33276](https://github.com/sgl-project/sglang/pull/33276) exists precisely to
make that hybrid load: it adds `stages.*` to the hybrid-DSV4-NVFP4 exclusions
so `mtp.0.ffn.experts.198.w1.scale → stages.0.mlp.experts.198.gate_proj.weight_scale_inv`
stops erroring, and the resulting path is *MXFP4 experts on a FlashInfer TRT-LLM
MxFP4 kernel* (SM100). It does **not** add a new format.

### 2.2 Community artifacts — one lossless FP8, one Q2_K, four arch strings

| repo | file | GiB | arch string | expert encoding |
|---|---|---:|---|---|
| `AtlasCloud/DeepSeek-V4-Flash-DSpark-FP8-dspark_only` | 3 safetensors shards | **18.56** | safetensors, `mtp.*` | **E2M1 → E4M3, lossless**, via SGLang's own `cast_e2m1fn_to_e4m3fn` |
| `am17an/DeepseekV4-Flash-20260731-DSpark` | 1 GGUF | 10.148 | `dflash` | MXFP4, bit-exact repack |
| `alessandrobologna/…-0731-DSpark-Drafter-GGUF` | MXFP4-Q8_0 | 10.149 | `deepseek_v4_flash_dspark_draft` | MXFP4 |
| ” | **Q2_K-Q8_0** | **6.492** | ” | **Q2_K** (max block-rel err 0.4907) |
| `bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF` | Q2K-Q8, `-0731` + base | 6.492 | `deepseek4-dspark` | Q2_K |
| `Lucebox/…-DSpark-Drafter-GGUF` | Q4RMFP4-denseF16 | 10.528 | `deepseek4-dflash-draft` | mixed |
| `YanissAmz/DeepSeek-V4-Flash-DSpark-draft-GGUF` | "bf16" | 10.148 | `dflash` | **filename is a misnomer** — 10.148 GiB is the MXFP4 size; 19.85 G params in bf16 would be ~37 GiB |
| `anemll/DSv4-Flash-DSpark-draft` | 3 `.bin` | 10.12 | CoreML manifest | Apple Neural Engine, irrelevant here |
| `fraserprice/DeepSeek-V4-Flash-DSpark` | `dspark-mtp-*` split out | 10.12 | safetensors | same MXFP4 bytes, just re-sharded |

Two things to take away:

* **The arch-string space is fragmented four ways** and only `dflash` has a
  real consumer (llama.cpp master, PRs #25173 + #25784). alessandrobologna's
  README says it outright: *"No compatibility with llama.cpp, Ollama, LM
  Studio, vLLM, or other runtimes is implied."* A GGUF route is therefore not
  "download and load"; it is "pick one of four incompatible containers and
  write the reader".
* **AtlasCloud's FP8 is the interesting one** and it is reproducible in-tree —
  see §3.1. It is also for the **0704** base, not `-0731`.

### 2.3 llama.cpp's converter — exact command and output map

`conversion/deepseek.py` (fetched from `ggml-org/llama.cpp` master this
session, 1 014 lines) registers `DeepseekV4DSparkModel` → `MODEL_ARCH.DFLASH`
at `:915-917`. Relevant mechanics:

* `filter_tensors` at `:967-971` keeps only `mtp.*`; `_rekey_mtp_tensor_name`
  at `:974-998` maps `mtp.<stage>.<rest> → layers.<stage>.<rest>`, with
  `main_proj/main_norm/markov_head.markov_w{1,2}/confidence_head.proj` and the
  `hc_head_*` / `norm.weight` singletons lifted to root.
* `set_vocab` at `:999-1011` **requires `--target-model-dir`** (the head has no
  tokenizer).
* `prepare_tensors` at `:909-912` **hard-sets
  `ftype = LlamaFileType.MOSTLY_MXFP4_MOE`** — the converter cannot emit any
  other expert encoding. `--outtype` only accepts
  `f32|f16|bf16|q8_0|tq1_0|tq2_0|auto` and does not reach the routed experts.

Conversion command (derived from the above; **not executed here**):

```bash
python3 convert_hf_to_gguf.py \
  --mtp-only \
  --target-model-dir "$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL" \
  --outfile "$OUT/dspark-0731-dflash-MXFP4.gguf" \
  "$MODEL_ROOT/DeepSeek-V4-Flash-0731-dspark-head-filtered"
```

Output map — read from the local am17an file with `GGUFReader`, 81 tensors,
which is exactly what the command above reproduces:

```
blk.{0,1,2}.ffn_{gate,down,up}_exps.weight   MXFP4  [4096,2048,256]/[2048,4096,256]   9.562 GiB
blk.{0,1,2}.attn_{q_a,q_b,kv,output_a,output_b}.weight,
blk.{0,1,2}.ffn_{gate,down,up}_shexp.weight,
fc.weight [12288,4096]                       Q8_0                                     0.442 GiB
markov_w{1,2}.weight [256,129280],
conf_proj.weight [4352,1],
blk.N.ffn_gate_inp.weight [4096,256]         BF16                                     0.129 GiB
blk.N.{attn_norm,ffn_norm,attn_sinks,attn_{q_a,kv_a}_norm,exp_probs_b.bias,
       hc_{attn,ffn}_{fn,base,scale}},
enc.output_norm, output_norm, output_hc_*    F32                                      0.009 GiB
```

`token_embd` / `output` are absent by design (borrowed from the target),
matching `deepseek_v4_dspark.py:632-647`.

**Can our #113 GGUF draft path load it directly? No.** `ANALYSE_447` §1.4's
four blockers all still hold and I re-verified the two decisive ones:
`gguf_registry.py:90-108` accepts `("qwen35","qwen35moe","gemma4","deepseek4",
"dflash-draft")` — `"dflash"` is not in the set — and
`deepseek_v4_dspark.py:861-864` accepts only `mtp.<stage>.*`, returning `None`
(silent `continue` at `:786-788`) for `blk.N.*`, so a GGUF head would load
**zero** tensors and boot into the #290 accept-collapse mode rather than
erroring. The #113 work solved the draft-MTP *expert namespace* for the
`dflash-draft` Qwen3 family; it does not carry over to `dflash`.

---

## 3 — Route (b): repack, with honest arithmetic

### 3.1 The dequant smoke — RAN, on CPU

`/root/.claude/jobs/1481bb40/tmp/smoke_463.py`, on the real local shard 46,
tensor `mtp.0.ffn.experts.0.w1` (`I8 [2048,2048]` + `F8_E8M0 [2048,128]`):

```
E8M0 exponent range: -6 .. -4  (3 distinct)
dequant fp32: absmax 0.375  rms 0.0660352  zeros 10.73%  distinct |v| 12
bf16 round-trip exact: True
cast_e2m1fn_to_e4m3fn round-trip EXACT: True   max rel err 0
fp8 bytes 8388608 + scale 512  vs  mxfp4 4194304 + 262144  -> 1.882x
```

Three results:

* **MXFP4 → bf16 is exactly representable** (12 distinct magnitudes = E2M1's 8
  levels × 3 scales). No dequant loss anywhere on this path.
* **The in-tree lossless cast is genuinely lossless** — the `MAX_OFFSET_BITS=6`
  trick at `fp8.py:198-236` reproduces the MXFP4 values with **max relative
  error 0** on real -0731 data. AtlasCloud's "losslessly represented" claim is
  confirmed against our own implementation, not taken on trust.
* It costs **1.882x the bytes**.

Requant error, same tensor, via `gguf.quants` (only Q8_0 is implemented in the
Python reference — the K-quants raise `NotImplementedError`, so a real
Q2/Q3 error number needs `llama-quantize`, not gguf-py):

```
type        bits/w        bytes   rel-RMSE   cos-sim
MXFP4(src)   4.250      4456448    0.00000  1.000000
Q8_0         8.500      8912896    0.00567  0.999487
```

### 3.2 Size table — 19.327 G routed-expert weights + 0.554 GiB non-expert

| format | bits/w | experts GiB | total GiB | vs MXFP4 |
|---|---:|---:|---:|---:|
| **MXFP4 (source)** | 4.250 | 9.562 | **10.117** | +0.000 |
| GPTQ INT4 g128 | 4.156 | 9.352 | 9.906 | −0.211 |
| NVFP4 g16 / GGUF Q4_K | 4.500 | 10.125 | 10.680 | +0.562 |
| GGUF IQ4_XS | 4.250 | 9.562 | 10.117 | +0.000 |
| GGUF Q3_K / IQ3_S | 3.438 | 7.734 | 8.289 | −1.828 |
| GGUF IQ3_XXS | 3.062 | 6.891 | 7.445 | −2.672 |
| **GGUF Q2_K** | 2.625 | 5.906 | **6.461** | **−3.656** |
| GGUF IQ2_XS | 2.312 | 5.203 | 5.758 | −4.359 |
| GGUF IQ2_XXS | 2.062 | 4.641 | 5.195 | −4.922 |
| FP8 e4m3 (lossless cast) | 8.031 | 18.070 | **18.625** | **+8.508** |
| GGUF Q8_0 | 8.500 | 19.125 | 19.680 | +9.562 |

(The community Q2_K-Q8_0 file at 6.492 GiB matches the Q2_K row; their
non-expert tensors are Q8_0/F32 rather than the source's fp8/bf16.)

**The brief's arithmetic constraint is confirmed and sharpened.** A 4-bit
requant closes nothing — GPTQ-INT4 saves 0.2 GiB, NVFP4 and Q4_K *cost* 0.6.
Against a stated ~4 GiB gap, only **Q2-class** clears it: Q2_K is −3.66 (still
~0.35 GiB short), IQ2_XS −4.36 clears, IQ2_XXS −4.92 clears comfortably. Q3_K
at −1.83 does **not** close a 4 GiB gap on its own.

### 3.3 Which requant target is actually runnable on which card

This is the constraint that decides the route, and it is per-architecture:

| encoding | 5090 (sm120) | 3080 (sm86) | mechanism in our tree |
|---|---|---|---|
| MXFP4 | **YES** | no | `Mxfp4MarlinMoEMethod`, `mxfp4_marlin_moe.py:116-117` — `is_sm90_supported() or is_sm120_supported()`; `is_sm120_supported = _device_version_gate(..., [12], (12,8))` at `utils/common.py:642`, and torch here is `2.11.0+cu128` |
| MXFP4 → fp8 (DEQUANT=1) | YES | **no** — block-fp8 triton dies with `type fp8e4nv not supported in this architecture` (`PLAN_417_dsv4_arch_paths.md`) | `fp8.py:1793-1811` |
| NVFP4 g16 | YES (sm120 native) | no | catalog §9 "NVFP4 (V4 class usable via dequant fallback for unpackable layers)" |
| **GPTQ/AWQ INT4 marlin** | **YES** | **YES** | `marlin_utils.py:99` refuses only below capability 80; the fork's MoE-marlin offload lane (#77/#120) already runs GPTQ/AWQ experts on this rig |
| GGUF K-quants | YES | **YES** | catalog §8, MMQ/dequant kernels are sm86-native |
| BF16 | 36 GiB — does not fit anywhere | — | `--speculative-draft-model-quantization unquant` exists as the escape hatch and is useless at this size |

That table is the real finding of route (b): **the only two encodings that run
on all three cards are GGUF K-quants and GPTQ-INT4 marlin.** Everything else
forces the draft onto the 5090 alone.

And the E8M0 span measured in §1 (≤ 2^5 within a tensor) means MXFP4 → NVFP4 is
an *exact* re-encode (duplicate each 32-block scale to two 16-blocks; every
power-of-two scale lands exactly in E4M3). NVFP4 buys nothing over MXFP4 here
though — same card, +0.56 GiB.

---

## 4 — Route (c): placement

### 4.1 What the solo refusal actually protects — and why DSPARK is close to free

`server_args.py` (`_handle_speculative_draft_placement`), read in full:

```
# v2 scope: the EagleDraftWorker family (EAGLE / EAGLE3 / NEXTN) and
# DFLASH. DFLASH goes solo cleanly because its draft is a self-drafting
# block model built weight-TP=1 on the host (heads%tp never binds at
# tp=1) whose per-round output is a fixed block of token ids -- exactly
# the one-broadcast-per-round contract. STANDALONE / NGRAM / DSPARK are
# still unsupported.
if not (algo.is_eagle() or algo.is_dflash()):
    raise ValueError(...)
```

The invariant being protected is the **one-broadcast-per-round contract**: the
solo rank must be able to hand the verifying ranks a payload that is *only*
token ids. The neighbouring refusals make the boundary explicit — `topk > 1`
is refused because a tree needs scores and parents; rejection sampling is
refused because it needs per-step draft probabilities; `FROZEN_KV_MTP` is
refused for a real architectural reason (its draft reads the target KV in
place, which no single rank holds under TP/DCP).

**DSpark fails none of those tests.** It is the same shape as DFLASH — a
self-drafting semi-autoregressive block model, `dspark_block_size = 5`, whose
round output is a block of token ids, with the Markov head chaining positions
*inside* the draft graph. The one delta versus DFLASH: the confidence head
truncates the block early (`p_min`), so the block length is variable and the
broadcast must carry **one extra integer** (the accepted block length) next to
the ids. Its target-hidden-state input (`main_proj` over layers 40/41/42) is
post-all-reduce and therefore replicated on every rank, so the solo rank has it.

So `DSPARK` in that refusal list reads as **unreviewed scope, not an
architectural exclusion** — unlike `FROZEN_KV_MTP` two branches above, which
carries its own physical reason. This is the single cheapest code change on the
board, and it is the one that unlocks the format-free arm.

Caveat to state in the ticket: DSpark's confidence-threshold truncation is
verify-gated, so output identity holds at temperature 0. At temperature > 0
with threshold acceptance (rather than rejection sampling) the arm is not
distribution-preserving — that is a property of DSpark, not of solo placement,
but it should be named in the same review.

### 4.2 Non-solo (`split`) — why it fails today, and what would fix it

Under the default `split` placement `dspark_worker_v2` builds a `ModelRunner`
per rank (`gpu_id`/`tp_rank` threaded through at `:65-103`), so the draft's 256
experts shard three ways: ~3.2 GiB of expert weight per rank, and **the VRAM
gap disappears by construction**. That is the structurally cheapest answer to
the memory problem.

It fails on kernels, not memory: ranks 1 and 2 are sm86, and MXFP4-marlin
raises `RuntimeError("MXFP4 Marlin requires SM90 or SM120.")` there, while the
DEQUANT=1 path lands on block-fp8 triton which sm86 does not have either.

There is a **per-draft** backend knob — `--speculative-moe-runner-backend`
(`server_args.py:3327`, resolved at `moe/utils.py:282-283, 319, 546`) — so the
draft can already use a different MoE runner than the target. What does **not**
exist is a **per-rank** one: the selection is process-uniform, and a 3-rank
group with sm120+sm86+sm86 needs two different answers for the same model.

Hence: **`split` + a requant to an encoding all three cards can run** (GGUF
K-quant, or GPTQ-INT4 marlin) is the only route that needs neither a placement
change nor a VRAM trade. It costs the most format work (§5).

### 4.3 Draft-expert offload — priced, and not recommended

The offload machinery (`layers/moe/expert_offload.py`, `resident_fraction.py`)
has **no draft-vs-target distinction** — `_moe_offload_active()` (`fp8.py:1177-1186`)
is deliberately group-wide, and a grep for `draft`/`spec` across
`expert_offload.py` + `resident_fraction.py` returns nothing relevant. So
"offload the draft's experts" is not currently selectable at all; it is a new
seam, not a flag.

The economics are also thin. One draft forward processes all 5 block positions,
routing `top_k = 6` of 256 per position per stage → up to 5 × 6 × 3 = 90
expert-fetches, 12.6 MB per expert at MXFP4 (w1+w2+w3) ⇒ **0.23-1.13 GiB per
draft forward** depending on how much the 5 positions' routes overlap. Compare
the measured target figure in catalog §3: **0.366-0.535 GiB/token** eager. So a
cold draft block costs roughly 0.5-2 target-tokens of PCIe traffic, on the same
links that are already the decode bottleneck (and #439 measured rank 0 as the
clock rank; per `rig-interconnect-p2p` GPU0 sits on ×4). At accept 0.49-0.77 —
i.e. ~2.5-3.9 of 5 tokens landing — that is plausibly break-even at best, and
it converts a latency-critical inner loop into a PCIe-bound one.

Verdict: **do not build this for #463.** If draft experts must spill, the
right shape is a *pinned-resident* draft (residency 1.0, never evicted), which
is the opposite of what the offload lane does.

### 4.4 The VRAM gap is a budget allocation, not a wall

> **Flag, against the brief's established facts.** I could not locate any
> persisted evidence for the "~4 GiB VRAM short on the 5090 (32607 MiB)"
> figure. `/spinning/gpu-battery-results/2026-08-03_447_dspark/` contains
> `prompts.json` and nothing else; `/spinning/wt-447-dspark` has no boot log;
> no server log under `/spinning/gpu-battery-results/` from 2026-08-02/03
> mentions dspark. I treat the number as operator-reported and
> **UNVERIFIED-BY-ME**, and give the arithmetic that makes it checkable rather
> than restating it.

A solo draft on the 5090 needs ≈ **10.5-11 GiB**: 10.117 GiB of weights, a
negligible draft KV pool (3 layers, `head_count_kv = 1`, `key_length = 512`,
SWA 128 → single-digit MiB at the recipe's 8192 context), plus activations.

Against the runbook's rank-0 budget of **30 407 MiB** (§4.5.4, repaired reserve
`2200,1800,1800`, base plan `30407,18680,18680` of a 32 607 MiB card). The
target is a 119.4 GiB UD-Q3_K_XL against 66.2 GiB of total budget across three
cards, so it necessarily runs with expert offload — meaning **rank 0's budget
is filled by a *tunable* resident expert set**, not by a fixed weight load.

Freeing ~11 GiB on rank 0 therefore means cutting its resident expert count. At
Q3_K, one DSV4-Flash expert is 3 × (2048 × 4096) × 3.44 bits ≈ **10.8 MiB**, so
11 GiB ≈ **1 040 experts** ≈ 25 per layer across the 41 MoE layers, out of rank
0's `119`-of-256 shard (`--rank-tp-ratio auto` → `MoE [119, 69, 68]`). That is
roughly a **21 % residency cut on the clock rank**, paid in extra PCIe misses
on exactly the rank #439 identified as the clock.

**That trade — not a dtype — is the real price of the solo arm**, and it is
measurable in one window: boot the target alone at the reduced rank-0 residency
and read the decode delta against the A-vs-A floor, before any draft exists.
Q2_K would shrink the ask from ~11 GiB to ~6.9 GiB (a ~13 % residency cut
instead of 21 %), which is the *only* thing a Q2 requant buys on this route.

### 4.5 Why aggressive draft quantization is legitimate here

Worth stating explicitly because it cuts against the standing quality-last
rule: **the drafter is lossless by construction** — the target's verify forward
is the sole token source, so quantizing the *draft* costs acceptance rate, never
output quality. The `#126` lossy-bucket gate in `ROADMAP_456` is about *target*
cold experts and does not apply. External corroboration: bleysg's README
reports Q2_K routed experts *"measured equal to Q4_K in acceptance and
throughput, 4.2 GiB smaller"* (their measurement, ds4 engine, DGX Spark — not
ours, and not on our target).

---

## 5 — Recommendation, Aufwand/Ertrag

Ordered by effort, cheapest first. Gains and efforts both stated; the ratio is
the reader's call.

| # | route | what it needs | gain | effort |
|---|---|---|---|---|
| **R1** | **Lift the solo refusal for DSPARK + free ~11 GiB on rank 0** | one branch in `_handle_speculative_draft_placement` (+ block-length in the broadcast payload); a residency cut on rank 0 | **first DSpark boot on this fork, no format work at all** — MXFP4-marlin is already native on sm120 | **small.** One code branch + one flag change. The residency cut is measurable independently. |
| R2 | R1, but with the Q2_K GGUF head to cut the residency ask 21 %→13 % | R1 **plus** a `dflash`-family GGUF reader (arch string, name map, MXFP4/Q2_K expert path) | −3.66 GiB, softer residency hit | **medium-large.** `ANALYSE_447` candidate E, and now worse: four incompatible arch strings, only `dflash` has a consumer, and the `-0731` Q2_K file uses `deepseek_v4_flash_dspark_draft` instead. |
| R3 | `split` placement + GPTQ-INT4 requant of the experts | a CPU requant pipeline (MXFP4→bf16→GPTQ g128, ~9.4 GiB out) + GPTQ name/shape handling in `deepseek_v4_dspark.py` (whose `.scale → .weight_scale_inv` rewrite at `:889` is fp8-specific) | **VRAM gap vanishes** (3.2 GiB/rank), all three cards get a marlin path, no placement change | **large.** New requant pipeline + a second loader shape. Structurally the *best* answer; not the cheapest first boot. |
| R4 | Lossless MXFP4→FP8 via `cast_e2m1fn_to_e4m3fn` (or AtlasCloud's file) | nothing new — the cast is in-tree and verified | none here; **+8.5 GiB** and dead on sm86 | small effort, **negative yield on this rig.** Bank it as the answer for a future all-sm90+ box. |
| R5 | Draft-expert offload | a new draft-aware seam in the offload lane | ~break-even at best, §4.3 | large. **Do not build.** |
| R6 | Per-rank MoE runner backend (sm120 marlin on rank 0, K-quant on 1-2) | a genuinely new capability | would make heterogeneous `split` general | large; a real feature, not a #463 step. Register it, don't start it. |

**Recommendation: R1.** It is the only route whose first boot needs no new
format, no new loader and no download — the bytes are already on disk at
`/spinning/llm_stuff/club-3090/models-cache/DeepSeek-V4-Flash-0731-dspark-head-filtered/`
and the 5090 already has the kernel. R3 is the right *destination* if DSpark
proves its 1.5-1.8x on this stack; R2 is dominated by R3 (same effort class,
strictly less general) and should not be started before R1 has produced an
accept number.

### Ticket text for the operator

> **#463-A — DSpark solo arm on the 5090: lift the placement refusal, price
> the residency trade**
>
> *Desk slice (no card):*
> 1. Correct `ANALYSE_447_llamacpp_dsv4_harvest.md` §1.5 + §4 candidate A: the
>    `mtp.*` routed experts are **MXFP4**, not fp8. Measured: 2 304 × `I8` =
>    9.000 GiB + 2 329 × `F8_E8M0` = 0.563 GiB; only 25 tensors are `F8_E4M3`.
>    Candidate A's "no loader change" conclusion depended on the wrong dtype.
> 2. Extend `_handle_speculative_draft_placement`'s v2 scope from
>    `is_eagle() or is_dflash()` to include `is_dspark()`, with the falsifier
>    first: a test that the refusal fires today and does not after. Carry the
>    variable block length in the solo broadcast payload next to the token ids
>    (DSpark's confidence head truncates the block; DFLASH's is fixed). Name
>    the temperature>0 threshold-acceptance caveat in the same review.
> 3. Set `--speculative-moe-runner-backend marlin` so the draft's MXFP4 experts
>    take `Mxfp4MarlinMoEMethod` (sm120) while the GGUF target keeps its own
>    runner. Verify `SGLANG_DSV4_FP4_DEQUANT` stays **0** — DEQUANT=1 both
>    trips `assert get_moe_runner_backend().is_auto()` (`fp8.py:426-430`) and
>    would inflate the head to 18.6 GiB.
>
> *Card slice, two boots in one window:*
> 4. **Boot A (no draft):** target only, rank-0 resident expert set cut by
>    ~11 GiB (≈ 1 040 Q3_K experts, ≈ 25/layer of rank 0's 119-of-256 shard).
>    Read the decode delta against a same-boot A-vs-A floor. **This prices the
>    solo arm independently of whether DSpark works** and is the number that
>    decides whether R1 is worth keeping.
> 5. **Boot B (draft on):** same residency, add
>    `--speculative-algorithm DSPARK --speculative-draft-model-path
>    $MODEL_ROOT/DeepSeek-V4-Flash-0731-dspark-head-filtered
>    --speculative-draft-placement solo --speculative-draft-gpu <5090 rank>
>    --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
>    --speculative-num-steps 1 --speculative-eagle-topk 1`
>    (the last three are pinned by `speculative_hook.py:367-430`, not free).
>    Read `meta_info.spec_accept_length`, `spec_verify_ct`, decode seconds —
>    **not** `spec_ema_accept_len`. Prompt set:
>    `/spinning/gpu-battery-results/2026-08-03_447_dspark/prompts.json`.
>    Reference band 0.49-0.77 (llama.cpp PR #25784, *their* domains, order of
>    magnitude only).
> 6. Answer `ANALYSE_447` §2.4 inside the same window: are the CSA/HCA/LID
>    compressor writes idempotent under a re-run at the same positions? If not,
>    rejected draft tokens corrupt compressor state and the accept number is
>    meaningless.
>
> *Gate:* if Boot A's residency cut costs more than DSpark's measured
> multiplier returns, R1 is refuted and the next step is **R3** (GPTQ-INT4
> requant + `split`), not R2.

---

## 6 — Provenance

* Local artifacts read: the three `-0731` DSpark safetensors shards (headers +
  one full expert tensor), `config.json`, and
  `DeepseekV4-Flash-20260731-DSpark.gguf` (81 tensors via `GGUFReader`).
* Scripts written and executed on CPU under `CUDA_VISIBLE_DEVICES=99`:
  `/root/.claude/jobs/1481bb40/tmp/smoke_463.py` (dequant + lossless-cast
  verification + Q8_0 requant error),
  `/root/.claude/jobs/1481bb40/tmp/smoke_463b.py` (scale dynamic range over 36
  sampled expert tensors + the size table).
* HF API reads: `am17an/…`, `AtlasCloud/…-DSpark-FP8-dspark_only`,
  `AtlasCloud/…-0731-FP8-DSpark`, `alessandrobologna/…` (both revisions),
  `bleysg/…`, `Lucebox/…`, `YanissAmz/…`, `fraserprice/…`, `anemll/…`,
  `nvidia/DeepSeek-V4-Flash-NVFP4` (`config.json` `hf_quant_config`),
  plus three `search=` sweeps of the model index.
* Upstream fetched: `ggml-org/llama.cpp` master `conversion/deepseek.py` and
  `convert_hf_to_gguf.py` (raw, to
  `/root/.claude/jobs/1481bb40/tmp/`); sgl-project/sglang PR #33276 page.
* Fork files read (all `/spinning/wt-merge-ops`):
  `layers/quantization/fp8.py:150-236, 405-445, 1177-1186, 1770-1830`,
  `layers/quantization/mxfp4_marlin_moe.py:12, 116-117`,
  `layers/quantization/marlin_utils.py:87-170`,
  `layers/quantization/marlin_utils_fp4.py:95-135`,
  `utils/common.py:642`,
  `server_args.py:3320-3350, 3661, 7075-7175`,
  `layers/moe/utils.py:282-283, 319, 546`,
  `speculative/dspark_components/dspark_worker_v2.py`,
  `models/deepseek_v4_dspark.py` (quant/scale/offload grep),
  `layers/moe/resident_fraction.py`, `layers/moe/expert_offload.py`,
  `docs/rig-runbook.md` §4.5.4 / §4.5.4b.
* No GPU used. No `/spinning/gpu-arb/` window taken. No repo commits.

---

## 7 — Adoption record (task #470, desk slice, 2026-08-03)

Added when this analysis was adopted into the tree. Everything below is
**DESK-WRITTEN** — no GPU window was taken; the boots are specified in
`TICKET_470_dspark_boots.md`.

### 7.1 What R1 turned into, in code

| §5 item | shipped as |
|---|---|
| lift the solo refusal for DSPARK | `server_args._handle_speculative_draft_placement`: the v2 scope test `algo.is_eagle() or algo.is_dflash()` became `algo.is_dflash_family()`, with the admission reasoning written in the same style as DFLASH's entry. `FROZEN_KV_MTP` is untouched and pinned by a test. |
| block length in the broadcast | `speculative/dspark_components/dspark_solo.py`. The payload is ONE packed int64 tensor `[graph_num_tokens, flags, bs, ids…, verify_lens…]`, so the round still costs exactly one broadcast; §4.1's "one extra integer" is one integer *per request*, because the truncation is per request (`RaggedVerifyLayout.verify_lens`), not per round. |
| `--speculative-moe-runner-backend marlin` | the flag already existed (`server_args.py:3327`); the gap was that `build_draft_tp_worker` never entered `speculative_moe_backend_context()`, so DFLASH/DSPARK drafts inherited the TARGET's runner. Extended, not duplicated. |
| DEQUANT stays 0 | unchanged; re-verified at `fp8.py:426-430`. |

### 7.2 Limits found while wiring it, which §4.1 did not name

* **Greedy acceptance only.** Non-greedy DSpark acceptance reads
  `draft_block.corrected_logits` — `[bs, gamma, vocab]` — on every verifying
  rank. That is not a payload. Refused by name per round
  (`refuse_solo_nongreedy_round`); it is the DSpark analogue of the existing
  rejection-sampling refusal and it belongs to solo, not to DSpark.
* **`SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` is default-ON and solo-fatal.** It
  shards `markov_w2` over the lm_head TP group, SKIPS the single vocab
  all_gather and gathers per markov step instead. Under solo the markov head
  exists only on the host, so the shadows have no shard for the per-step
  gather and the skipped single gather is the one they are sitting in.
  Disabled under solo with a logged reason rather than refused — the user did
  not ask for a default-on optimization by name.
* **The graph-folded greedy proposal is off under solo.** Its
  `compute_base_logits` call carries the vocab all_gather into the capture,
  and the shadows capture no draft graphs. Same trade DFLASH already makes.
* **The draft's embedding had to be hoisted.** `forward_embed` calls the
  TARGET's vocab-parallel embedding from INSIDE the draft forward. Under solo
  that all_reduce would sit inside a graph the shadows never run. The solo
  path computes the embedding eagerly and hands it in as `input_embeds`.

### 7.3 R4, banked

Lossless MXFP4 → FP8 via `cast_e2m1fn_to_e4m3fn` is **not rejected as wrong —
it is rejected as negative-yield on this rig**: verified bit-exact (§3.1, max
rel err 0), costs +8.5 GiB, and the fp8 block path is dead on sm86
(`PLAN_417`). It is the correct answer for an all-sm90+ box and should be
reached for there without re-deriving it. Do not re-open it for this rig
without a new fact (a card change, or an fp8 path that runs on sm86).

R2 (Q2_K GGUF head) stays dominated by R3 and must not be started before R1
has produced an accept number. R5 (draft-expert offload) and R6 (per-rank MoE
runner backend) are registered, not started.

### 7.4 Upstream deltas that landed after this analysis was written

* **sgl-project/sglang #33344** publishes a third-party EAGLE3 draft head for
  exactly this checkpoint (`AQ-MedAI/DeepSeek-V4-Flash-0731-eagle3`, marlin MoE
  runner) with an **upstream-reported** accept of 2.62-3.20 across six
  benchmarks — an order above the DSpark ladder's 0.49-0.77. Not our
  measurement. It does not change R1's code (the solo unlock is head-agnostic:
  EAGLE was already admitted, and the block-length payload is DSpark-only), but
  it changes which head Boot B should carry. See `TICKET_470_dspark_boots.md`
  §Boot C.
* **#33298 (merged upstream)**: in-graph philox sampling for the DSPARK
  graph-folded proposal, moving the sampler into a new `dspark_draft_sampler.py`.
  Our solo path deliberately does not fold the sampler, so the behaviours do not
  conflict — but `dspark_draft.py` will move under our feet on the next rebase.
* **#33312**: DSpark shared-expert loading is broken on upstream main
  (ServerArgs-burndown fallout). We are not on upstream main; it is a rebase
  gate, not a live exposure.
