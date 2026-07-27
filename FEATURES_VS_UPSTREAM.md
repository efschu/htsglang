# htsglang Fork Features

Comparison as of 2026-07-25 (status vocabulary updated 2026-07-26, see Changelog), checked directly
(not against memory or task lists) against two
branches: `integration/r3-probe` (repo `wt-merge-probe`, includes at least `4c90038a78`) and
`feat/htccl-gfx900` (repo `wt-htccl`, tip `3cc2fc9da5`, 9 commits, of which 2 — `fa5c507476` and
`3cc2fc9da5` — are **not yet merged** into `integration/r3-probe`; noted per row where it applies).
`integration/r2` is superseded by `integration/r3-probe`. `/spinning/htsglang` is a stale checkout
and was not used as a source.

This document lists the features carried by the htsglang fork and whether an equivalent
capability is present in upstream SGLang, upstream vLLM, `llama.cpp`, or `ik_llama.cpp`.

## Status legend

**Status**
- `Built` — merged; covered by our own tests only.
- `Boot-checked` — executed on hardware with a real model; coherent output.
- `Cross-checked` — validated against an independent reference (another
  backend, a solo/TP=1 run as oracle, `torch`/`torch.distributed`, or a
  byte-/token-identity that must hold for structural reasons). Reference and
  scope are stated per row.

All entries are work in progress. No entry implies external review or upstream-mergeable maturity.
Reference rig: 1x RTX 5090 + 2x RTX 3080 for most rows, plus 1x RTX 2080 Ti (sm75) + 1x Radeon RX
Vega 64 (gfx900) for the cross-vendor rows.

`falsifikator-geprueft` marks a row whose own test is on record as red before its fix and green
after.

**fp8 on RTX 3080, since #190:** `gptq_marlin_gemm`, the only fp8 GEMM sm86 has, is measured
run-to-run nondeterministic above ~109 prompt tokens (0/1200 mismatches through M=109, first
mismatch at M=128; `fix/gdn-prefill-determinism`, not yet merged). Byte-/token-identity claims
above that length on a 3080 are flagged per row rather than counted as `Cross-checked`; the RTX
5090 (sm120, a different fp8 GEMM path) is unaffected at any length.

## Core concepts

Defined once here; detail sections below reference these by name rather than re-explaining them.
- **Asymmetric / uneven TP** (`--rank-tp-ratio`, row 1): unequal per-rank attention-head/weight
  shard sizes within one tensor-parallel group, for mismatched GPUs.
- **Asymmetric / uneven DCP** (`--rank-kv-ratio`, row 2): capacity-weighted KV-cache ownership
  across ranks during decode (decode context parallelism).

## Document structure

To stay readable on GitHub without horizontal scrolling: the **overview matrix** below gives one
narrow row per feature with verdict tokens only (`yes` / `partial` / `no` / `n/a` / `unverified`),
linked by anchor to its **detail section**, which carries the fork status, key measured numbers,
and only the upstream distinctions not already implied by the matrix token.

**Column definitions**
- **Fork** — `Built` / `Boot-checked` / `Cross-checked` (see Status legend above), `WIP` (present
  but not complete/validated), `Exp` (highly experimental, not production-ready). A trailing `*`
  means the capability lives only on a not-yet-merged branch, named in the detail section.
- **SGLang / vLLM / llama.cpp / ik_llama.cpp** — `yes` / `partial` / `no` / `n/a` / `unverified`.
  `partial` always names the mechanism difference in the detail section, never left implicit.
  `unverified` means the check could not be completed with the sources available in this pass.
  `n/a` means the row's specific comparison point doesn't apply to that engine (e.g. a base
  capability like GGUF or PD-disaggregation exists there, but not the fork's delta on top of it).

## Overview matrix

| # | Feature | Fork | SGLang | vLLM | llama.cpp | ik_llama.cpp |
|---|---|---|---|---|---|---|
| [1](#f1) | Asymmetric tensor parallelism | Boot-checked | no | no | partial | partial |
| [2](#f2) | Asymmetric decode context parallelism | Cross-checked | partial | partial | no | no |
| [3](#f3) | Rank-to-GPU mapping and co-location | Boot-checked | no | no | no | no |
| [4](#f4) | Solo drafter placement | Built | no | yes | partial | partial |
| [5](#f5) | Cross-algorithm drafter routing | WIP | no | no | no | no |
| [6](#f6) | CUDA graph memory aliasing for spec branches | Boot-checked | partial | partial | no | no |
| [7](#f7) | MoE expert offload + asymmetric TP/DCP | Boot-checked | partial | partial | partial | partial |
| [8a](#f8a) | Bespoke GGUF adapter framework | Boot-checked | no | no | n/a | n/a |
| [8b](#f8b) | Qwen3.5/3.6 GGUF | Boot-checked | no | no | yes | yes |
| [8c](#f8c) | Gemma-4 GGUF | Boot-checked | no | no | yes | yes |
| [8d](#f8d) | GGUF K-quant compute kernels | Boot-checked | partial | partial | yes | yes |
| [8e](#f8e) | Asymmetric-TP x GGUF correctness | Boot-checked | no | no | n/a | n/a |
| [8f](#f8f) | Multimodal and dynamic-quant GGUF | Boot-checked | partial | partial | yes | partial |
| [9](#f9) | Hibernate checkpoint/restore | Boot-checked | no | partial | partial | partial |
| [10](#f10) | Measured VRAM budget | Boot-checked | partial | partial | partial | partial |
| [11](#f11) | Cross-architecture speculative determinism | Boot-checked | partial | partial | no | no |
| [12](#f12) | Weightless-KV lane | Cross-checked | no | no | no | no |
| [13](#f13) | Rig dashboard / planner UI | Exp | n/a | n/a | n/a | n/a |
| [14](#f14) | Single-node PD disaggregation | Boot-checked | yes (base) | yes (base) | no | no |
| [15](#f15) | Asymmetric-TP quantization correctness | Boot-checked | partial | partial | n/a | n/a |
| [16](#f16) | Fast-lane priority scheduling | Built | partial | partial | no | no |
| [17](#f17) | HiCache under asymmetric-TP/DCP | Boot-checked | yes (base) | n/a | partial | partial |
| [18](#f18) | TP greater than num_kv_heads | Boot-checked | partial | partial | partial | partial |
| [19](#f19) | Broad model bring-up under asymmetric-TP | Boot-checked | n/a | n/a | n/a | n/a |
| [20](#f20) | Session KV spill | Exp | partial | partial | partial | partial |
| [21](#f21) | HTCCL cross-vendor collectives | Cross-checked | no | no | partial | partial |
| [22](#f22) | fp8 dequant fallback (W8A16) | Cross-checked* | no | no | partial | unverified |
| [23](#f23) | Turing/gfx900 without sgl-kernel | Cross-checked | no | no | partial | partial |

---

## Model coverage, tested combinations, and measured numbers

The matrix above says which capabilities exist. This section says what was actually run on
hardware and what came out of it. Evidence tiers are the ones from the Status legend
(`Built` / `Boot-checked` / `Cross-checked`); a configuration that exists only as code is named as
`Built` and carries no numbers.

**Measurement environment, stated once.** Main rig: 1x RTX 5090 (sm120) + 2x RTX 3080 (sm86), no
NVLink and no CUDA P2P (GeForce, PHB topology — all cross-GPU traffic is host-staged), one 3080 on
PCIe Gen4 x4 (~6.5 GB/s host-staged DMA against ~13-14 GB/s for the other two), and during the
#203 window one 3080 in software thermal slowdown at 85-87 C (1719-1840 MHz against 1920 MHz on
the identical card). Clock pinning is refused by the driver on these cards. Cross-rig rows add a
second host with an RTX 2080 Ti (sm75) and a Radeon RX Vega 64 (gfx900) over 40G RoCE. The numbers
below are what this hardware yields; they are a lower bound for the configurations, not a
projection of them.

### Model families and quantization formats

| Family | Architecture class | Formats loaded on hardware | Tier |
|---|---|---|---|
| Qwen3.6-27B | dense, GDN/attention hybrid (64 layers, 16 full-attention, 4 kv-heads x 256, embedded NEXTN/MTP draft) | FP8 (native, + `mtp.safetensors`; KV `fp8_e5m2`); AWQ-BF16-INT4 compressed-tensors; GGUF K-quant `Q3_K_M`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, Q8-class; unsloth dynamic `UD-Q4_K_XL`, `UD-Q6_K_XL` (+ `mmproj` vision tower), `UD-Q8_K_XL` | Boot-checked |
| Qwen3.6-35B-A3B | MoE + GDN hybrid (40 layers, 10 full-attention, 2 kv-heads x 256, 256 experts / 8 active, `nextn=1`) | FP8 (e4m3 dynamic, + `mtp.safetensors`); AWQ-4bit (AutoAWQ, g32) | Boot-checked |
| Qwen3.5-122B-A10B | MoE + GDN hybrid (48 layers, 256 experts top-8, 3 linear : 1 full) | GPTQ-Int4 (g128) | Boot-checked |
| Gemma-4-31B-it | dense, SWA hybrid (10 global : 50 sliding) | int4-AutoRound (g128 sym); GGUF `Q4_K_M` | Boot-checked |
| Gemma-4-26B-A4B-it | MoE, SWA hybrid (30 layers, 128 experts, global kv=2) | compressed-tensors pack-quantized W4A16 int4 (g32) | Boot-checked |
| Llama-3.1-8B-Instruct | dense GQA | bf16, unquantized | Boot-checked |
| Qwen3.5-4B | dense, GDN hybrid | fp16; FP8-dynamic (compressed-tensors) | Cross-checked |
| Qwen2.5-1.5B, Qwen3-0.6B, Qwen3.5-2B, Qwen-0.5B | small dense and hybrid, incl. replicated-KV geometries | bf16 / fp16 | Boot-checked |

The small models are diagnostic vehicles — solo oracles and falsifiers for the DCP, co-location
and `kv == tp` work — not serving targets. Format coverage is not uniform across families, and the
reasons differ per gap: Gemma-4 GGUF is verified for `Q4_K_M` only, and its MoE, MTP and vision
paths fail fast in the adapter by design (row 8c); Qwen3.6-35B-A3B GGUF does not load, because the
`qwen35` adapter maps none of its MoE expert tensors; FP8 on the 122B has no placement here, since
its pinned host pool would need ~116 GiB against 108 GiB of RAM. `UD-Q8_K_XL` needed mixed-dtype
handling for the fused GDN `in_proj_qkvz` before it would load at all; records that mark it
infeasible predate that fix.

### Tested combinations — family x format x feature

| Configuration | Features exercised together | Tier | Evidence |
|---|---|---|---|
| Qwen3.6-27B FP8, TP=3 (5090 + 2x 3080) | uneven TP, uneven DCP, NEXTN/MTP k=3, flashinfer, CUDA graphs | Boot-checked | rows 1/2; #210, #217, #103 |
| Qwen3.6-27B FP8, TP=3, greedy, no spec | uneven DCP on Triton vs a DCP-off ground truth | Cross-checked (byte-identical on `short_code`) | row 2, #173 G4 |
| Qwen3.6-27B FP8, TP=3, MTP, CUDA graphs | uneven DCP + chain speculative verify, Triton vs flashinfer | Cross-checked (token ids identical on the 3 short prompts) | row 2, #180 V4 |
| Qwen3.6-27B FP8, TP=3, NEXTN k=3 | uneven TP + session KV spill, 2 co-resident sessions | Boot-checked | row 20, #217 |
| Qwen3.6-27B AWQ-BF16-INT4, TP=3 | uneven TP/DCP + HiCache (host-RAM L2, file L3), 8 concurrent requests | Boot-checked (restore deterministic, 8/8 hit) | row 17 |
| Qwen3.6-27B GGUF `UD-Q6_K_XL` (+ `mmproj`), TP=3 | uneven TP + GGUF + NEXTN/MTP + CUDA graphs | Boot-checked | rows 8b/8f |
| Qwen3.6-27B Q8-class GGUF, TP=3, CUDA graphs | uneven TP + GGUF + MMQ decode threshold | Boot-checked | row 8d, #163 |
| Qwen3.6-27B GGUF `Q3_K_M`, TP=3 | uneven TP + GGUF + hibernate across process exit | Boot-checked | row 9, #89 |
| Qwen3.6-27B GGUF `Q6_K` / `Q4_K_M`…`Q8_0`, TP=3 and TP=2 | uneven TP + GGUF, greedy-deterministic | Boot-checked | row 8b |
| Qwen3.6-27B AWQ-BF16-INT4, TP=3 | uneven TP + INT4 group alignment | Boot-checked | row 15 |
| Qwen3.6-27B GGUF `UD-Q6_K_XL`, TP=4 co-located on 3 cards | rank co-location (two ranks on one physical GPU) | Boot-checked | row 3 |
| Qwen3.6-35B-A3B FP8, TP=3 with 2 kv-heads | TP above the model's kv-head count: replicated KV on all 10 global layers + token-sharded DCP, with MTP and CUDA graphs | Boot-checked | row 18 |
| Qwen3.6-35B-A3B FP8, TP=3 | uneven TP + `--rank-kv-ratio capacity`; + HiCache file L3 | Boot-checked; HiCache restore Cross-checked (greedy ids identical cold vs restored) | `docs/advanced_features/uneven_kv_token_ratio.md` §9 |
| Qwen3.6-35B-A3B FP8, TP=3 mixed-arch | fp8 fused MoE with a Marlin W8A16 fallback on the sm86 ranks while sm120 runs the native path | Boot-checked (microbench cos >= 0.99998) | row 15 |
| Qwen3.6-35B-A3B AWQ-4bit, single node | solo prefill TP=1 on the 5090 + decode at uneven TP=3 `1,3,3` with weighted DCP, GDN state handoff, ctx 131072 | Boot-checked (disagg-off byte-identical) | row 14, `docs/advanced_features/pd_disagg_single_node.md` |
| Qwen3.6-35B-A3B AWQ-4bit, TP=3 | uneven TP + uneven DCP + MoE expert offload | Cross-checked (32/32 tokens identical to a TP=1 run) | row 7 |
| Qwen3.5-122B-A10B GPTQ-Int4, TP=3 + 108 GB host RAM | uneven TP + uneven DCP + MoE expert offload | Boot-checked (self-deterministic 5/5; not bit-identical to the no-offload case — Marlin-Int4 tiling) | row 7, #77 |
| Gemma-4-31B-it int4-AutoRound, TP=1 and TP=3 | uneven TP on an SWA hybrid | Boot-checked | row 19 |
| Gemma-4-31B-it int4-AutoRound, TP=3 | uneven TP + EAGLE3 speculation | Cross-checked (4/4 temp-0 probes byte-identical to the no-spec oracle) | row 19, #101 |
| Gemma-4-31B-it int4-AutoRound, TP=3 | SWA-DCP Stage B + `--swa-pool-sizing cap` + CUDA graphs; needle planted ~3k tokens past the 1024-token window | Cross-checked (byte-identical to a TP=1 solo-5090 oracle) | SWA-DCP Stage B (#96-H5) |
| Gemma-4-31B-it GGUF `Q4_K_M`, TP=1 on the 5090 and TP=3 | GGUF adapter framework + uneven TP | Boot-checked | row 8c |
| Gemma-4-26B-A4B-it W4A16, TP=1 | MoE SWA hybrid bring-up (vision-ignore + gated-GeLU Marlin fix) | Boot-checked (3.6k needle hit) | row 19 |
| Llama-3.1-8B bf16, TP=2 (5090 head + 3080 worker), CUDA graphs | weightless-KV lane + EAGLE3 chain spec at topk 1 + solo draft placement | Boot-checked; the lane by itself Cross-checked against a TP=1 solo oracle (#124) | rows 12/4, #143 |
| Llama-3.1-8B bf16, TP=5 over two hosts (5 cards, 4 architecture classes), ratio 4,3,3,2,1 | cross-rig TP + uneven TP + EAGLE3 split and solo arms | Boot-checked (prose byte-identical to solo; code and mixed diverge for documented reduction-order reasons) | Nordstern L0 |
| Qwen3.6-27B FP8, TP=4 cross-rig over RDMA, eager | uneven TP 6,4,4,2 + uneven DCP + NEXTN 3 + solo draft over HTCCL `ucx` | Boot-checked | row 21 (#198), #204 |
| Qwen3.5-4B fp16, mixed-vendor TP=2 (2080 Ti + Vega 64), Triton, eager | cross-vendor collectives (`gloo`/`device`), even 2/2 and uneven 3,1 | Cross-checked (byte-exact vs `torch.distributed`; model-scale byte-identical to `gloo`, 4/4) | rows 21/23 |
| Qwen3.5-4B-FP8-dynamic, solo Vega 64 / solo 2080 Ti / mixed TP=2 | fp8 W8A16 dequant fallback + cross-vendor TP | Cross-checked (byte-identical, solo runs as oracle) | row 22 |
| Qwen2.5-1.5B / Qwen3-0.6B / Qwen3.5-2B | replicated-KV geometry, DCP and co-location falsifiers across Triton/flashinfer and NCCL/gloo | Boot-checked | rows 2/18 |

**Combinations with no hardware boot on record.** Fast-lane priority scheduling (row 16) is
`Built` only. Solo draft placement (row 4) has no dedicated boot of its own; it is exercised
inside the #143 lane arm, the #198 cross-rig arm and the Nordstern S4 arm above. Cross-vendor
CUDA-graph capture is not demonstrated (row 21). MoE-model hibernation is not built (row 9).
Session KV spill lists spec-in-tick spill coincidence and 3-session co-residency as not validated
(row 20); the 3-session lifecycle benchmark was run and discarded, because the third session was
never admitted and a feature-off control hit the same ceiling — a scheduler admission limit, not a
spill defect, and no number from it is carried. Tree speculation at `--speculative-eagle-topk > 1`
under weighted DCP is gated off (Guarded/descoped). The `fp8.py` `Fp8Config` family is not wired
to the capability probe (row 22). Gemma-4 under uneven DCP is blocked (`SWAKVPool.set_kv_buffer()`
has no `dcp_kv_mask`), and Gemma-4 rejects the flashinfer backend outright — those rows are Triton
with bf16 KV.

### Measured numbers

#### Throughput and per-round cost

| Comparison | Configuration | Measured | Source |
|---|---|---|---|
| Chain speculation on the weightless-KV lane, vs the same lane without spec | Llama-3.1-8B bf16, TP=2, EAGLE3 topk 1 / 3 steps / 4 draft tokens, solo draft, CUDA graphs, 256 tokens per run, 2 cold boots per arm | 71.67 -> 80.67 (**+12.6%** one_token), 69.52 -> 120.05 (**+72.7%** code), 71.80 -> 87.20 (**+21.5%** prose), 71.53 -> 86.18 (**+20.5%** mixed). Content-robust form: a verify round costs 1.22-1.27 plain decode steps and returns 1.38-2.12 tokens. Every class clears the noise floor by at least 4.8x | #143 Gate 4 |
| Weightless lane decode graph-captured vs eager | same lane | 13.1 -> **63.5 tok/s** | #214 |
| MoE expert offload vs the largest solo placement | Qwen3.5-122B-A10B GPTQ-Int4, TP=3 + 108 GB host RAM, 64 resident + 16 scratch of 256 experts/layer | 6.97 tok/s eager vs 4.8 tok/s solo 5090 (**+45%**); later graph paths 10.61 (graph-static) and 16.34 (graph + hot-set), from a separate recording | #77 |
| `--rank-kv-ratio speed` (`[2,1,1]`) vs `capacity` (`[2,3,3]`), content pinned by construction (spec off, `ignore_eos`, dense model) | Qwen3.6-27B FP8, TP=3 uneven DCP, flashinfer, bs=1, ctx 131072, 120,420 resident tokens, 6 interleaved cold boots | depth term 2.2955 -> 1.7324 ms/step = **-24.5%** (t = -17.85) against a 1.07% A-vs-A noise floor; step time 28.530 -> 27.813 ms (-2.51%). Spec-on cross-check at identical accept (3.871) and verify count (155): depth term -26.7%, round time 43.307 -> 41.523 ms (-4.12%, i.e. +4.3% tok/s) | #210 |
| MTP + adaptive draft vs no speculation | Qwen3.6-27B FP8, TP=3 uneven DCP, fp8 KV, CUDA graphs, bs=1, accept 3.32 | code 40.3 -> **90.7** tok/s (2.25x), prose 40.2 -> **116.3** tok/s (2.89x), at lower board power (640 vs 729 W) | #146 |
| GGUF MMQ decode threshold ON vs OFF | Qwen3.6-27B Q8-class GGUF, TP=3 uneven, M=8, 30 s window, greedy | code 201.60 -> 222.93 (**+10.6%**), prose 201.87 -> 221.46 (+9.7%), mixed 201.33 -> 221.33 (+9.9%). The whole rate distribution moves (p50 25.2 -> 27.9, p95 25.5 -> 28.3), so it is not a tail artifact. Only the sm120 rank reroutes (11320 MMQ / 0 MMVQ on TP0 against 0 / 11320 on TP1/TP2). Not byte-identical when ON | row 8d, #163 |
| Tuned K-quant kernels vs the legacy dispatch (kill-switch) | Qwen3.6-27B `Q6_K_XL` GGUF + MTP, TP=2 on 2x 3080 | 67.86 / 54.17 -> **88.38 / 72.22** tok/s code/prose (+30% / +33%); TP=3 uneven tuned 118.01 / 98.62. Argmax-identical 100/100, not bit-parity | #73, `11fb6e88cd` |
| HTCCL `device` vs `gloo` transport, cross-vendor | Qwen3.5-4B, mixed TP=2 (2080 Ti + Vega 64), Triton, eager, slope method | decode **+37%** even 2/2 (10.28 -> 14.07) and **+48%** uneven 3,1 (11.13 -> 16.51); prefill +45% (540.5 -> 786.2) / +62% (604.7 -> 982.0); byte-identical 4/4 between the transports | row 21 |
| Cross-rig transport L1 -> L2 (pipelining, progress interleaving, short small-message path) | Qwen3.6-27B FP8, TP=4 cross-rig, uneven TP 6,4,4,2, NEXTN 3 + solo draft, Triton, eager, ctx 8192, bs=1, slope method, 3 reps | slope tok/s code 16.38 -> 17.31 (+5.7%), prose 15.67 -> 17.57 (+12.1%), mixed 16.54 -> 18.36 (+11.0%); `spec_accept_length` identical per content class across arms (3.08 / 3.03 / 2.91) | #198 |
| RDMA vs 1 GbE on the same cross-rig TP=4 arm, decode by streaming | Qwen3.6-27B FP8, TP=4 | 28.4 / 28.6 / 27.1 vs 7.8 / 7.7 / 7.5 tok/s = **3.6-3.7x**. The win is latency, not bandwidth: 14x at the barrier on the same wire, 1.2x at 4 MiB | #204 |
| TP=5 across two hosts, against a solo reference | Llama-3.1-8B bf16, ratio 4,3,3,2,1, ctx 4096, eager, HTCCL `gloo`, no DCP or spec | code 4.32 / prose 4.73 / mixed 4.82 tok/s against ~76 tok/s solo on the 5090 — ~5.7-6.3% of solo. This is a capacity and correctness result, not a throughput one; the EAGLE3 split and solo arms (4.39-5.39) show no measurable winner at accept 1.36 | Nordstern L0 |
| Draft placement solo vs split, controlled same-tree A/B | Qwen3.6-27B FP8, TP=3, DFLASH, same prompts, 450-token clean decode, 4 reps, 16/16 valid | split 74.88 vs solo **82.71** (code, -9.5%), 64.66 vs **67.58** (prose, -4.3%) — solo kept. The accept-length gap that earlier favoured split was a cross-commit artefact | #160 |
| DFLASH vs NEXTN across context length | Qwen3.6-27B FP8, greedy, 1024 decode tokens, mean of 2 runs | ctx 4096: 125.7 vs 118.8 (**+6%**); ctx 49152: 98.6 vs 95.3 (+3.5%, **within spread** — parity). The two arms also differ in placement (NEXTN at uneven TP=3, DFLASH at TP=4 co-located with MPS), so this is algorithm plus topology, not algorithm alone. In DFLASH's weak multiturn regime it runs 18.9-20.9% behind NEXTN | #157 |
| Cross-algorithm bandit vs the static winner of the same cell | Qwen3.6-27B FP8, one regime cell | 75.52 vs 89.22 tok/s — the bandit loses the cell; switch cost ~2.5 ms. Row 5 stays WIP | #156 |
| fp8 W8A16 dequant fallback vs fp16, same TP config | Qwen3.5-4B-FP8-dynamic, mixed TP=2 | 12.67 vs 16.51 tok/s = **-23%** decode, prefill roughly neutral (982 vs 966). A placement enabler, not a speed lever — on this pair solo fp8 on the 2080 Ti (15.23) is faster than the mixed pair | row 22 |
| Hibernate restore vs cold start | Qwen3.6-27B GGUF `Q3_K_M` class, TP=3 uneven | ~50 s -> 8-14 s to ready. This is a documented summary range, not a single raw A/B run — a narrower measurement of the skippable transform alone (~44 s -> seconds) supports the order of magnitude. The FP8 path has nothing skippable and shows no meaningful benefit | row 9, #89 |
| `--rank-vocab-ratio 7,3,3` vs default | Qwen3.6-27B FP8, TP=3 uneven DCP, MTP, baseline n=19 / arm n=8 | single code 90.30 -> 95.19 tok/s (+5.41%), but dual +0.45 / -0.11 / +0.76% round rate — **within spread** on the axis that decided it. Rejected; recorded as a lead only | #103, #199 |
| Speculative k=4 vs k=3 | same arm | accept +7.2-12.2%, round rate -6.6 to -7.7% single and -13.3% dual, net single +0.2-3.5% and dual -6.00%. Rejected; the working-arm recipe is unchanged | #103 |
| KV-token split perf vs capacity, content **not** pinned | Qwen3.6-27B FP8, TP=3, six points | round rate mean +0.67% (sd 1.74%, range -1.62 to +2.80), tok/s mean +3.65% (sd 7.17%) — **within spread**. All six points produced different output text, which is why #210 re-ran the question with content pinned | #203 |
| Cross-rig async collective overlap (`SGLANG_HTCCL_UCX_OVERLAP=1`) vs the same transport without it | Qwen3.6-27B FP8, TP=4 cross-rig | code +3.5%, prose -2.6%, mixed -2.9% — **within spread**; a measured negative result, matching the dependency analysis that predicted it | #198 |
| Intra-rig DCP collective overlap and DCP comm fusion | Llama-3.1-8B TP=2 lane arm; Qwen3.6-27B TP=3 | -0.17 to +3.12% across four classes, and ~80.5 / 80.6 / 80.85 tok/s for the fusion variants — **within spread** on both | #128, DCP fusion |

#### Capacity

| Comparison | Configuration | Measured | Source |
|---|---|---|---|
| `--rank-kv-ratio capacity` vs `coupled` | Qwen3.6-27B FP8, TP=3 | `max_total_num_tokens` 443,904 ([30,17,17]) -> **563,456** ([33,13,18]), **+26.9%**; pool-end free VRAM 5.21/2.33/3.58 GB (stranded on the 3080s) -> 2.71/2.46/2.33 GB (balanced). Decode cost within +-1% at shallow, 8k and 24k depth | `docs/advanced_features/uneven_kv_token_ratio.md` §9 |
| same | Qwen3.6-35B-A3B FP8, TP=3 | 1,911,488 -> **2,187,648** pre-cap, **+14.4%**; the #79 mamba ceiling binds first at these settings, so the pool gain becomes usable only with more requests or longer context | same |
| Demand-driven `[auto-mamba]` sizer vs the stock fixed mamba-slot sizer — one binary, one model, stock flags on one side | Qwen3.6-27B FP8, 3 cards, context uncapped at the model max 262,144 | pool **883,584** tokens with 7 demand-driven mamba slots, against 146,024 (stock flags PP=3 even, 9 fixed slots) and 176,066 (stock flags PP=3 uneven, 14 fixed slots); free VRAM left over 8.40-10.50 vs 5.19-20.06 GB. This is a sizing-formula difference, not a parallelism one: VRAM, `context_len` and `max_running_requests` were each eliminated as the binder by measurement | Baseline block; `INTEGRATION_R3_VALIDATION.md` |
| Capacity cost of the `speed` KV split | Qwen3.6-27B FP8, TP=3 uneven DCP | `max_total_num_tokens` 393,228 in **both** arms at the deep no-spec point — the hybrid mamba cap binds first, so the bandwidth shift is free there (raw KV headroom even rises 2.3%). It is not free in general: on the spec-on ctx-32768 reference the same shift costs 50% of raw headroom (842,856 -> 422,480), which that arm's mamba cap simply hides | #210 |
| Where the `speed` split puts the context | Qwen3.6-27B FP8, TP=3, `--rank-perf-tune dec` | KV rows 52,258 / 23,055 / 23,055 against the capacity boot's 24,584 / 36,876 / 36,876 — the 5090's share of the context goes from 25% to 53% at an unchanged `max_total_num_tokens` of 98,328 | #210 |
| Speculative CUDA-graph branch aliasing | cross-algorithm lazy capture, arm C | 542.0 MiB released under CUDA graphs — the only GPU number recorded against this path | row 6, #156-4 |
| MoE expert offload | Qwen3.5-122B-A10B GPTQ-Int4 on 3 cards (72 GB aggregate) | 64 resident + 16 scratch experts per layer on GPU, **176 of 256 offloaded** to 108 GiB of host RAM. Upstream has no host-spill load path for this model at all. The host-RAM wall is real: an FP8 122B would need ~116 GiB pinned and stays out of reach | row 7, #77 |
| Weight footprint, GGUF vs FP8 | Qwen3.6-27B, TP=3, same pool (81,960 tokens) | FP8 29.7 GB of weights (35.8 with the draft) against GGUF Q4 **18.3 GB** (18.8); GGUF Q8 35.1 GB | #157 |
| fp8 W8A16 dequant fallback | Qwen3.5-4B-FP8-dynamic on a Vega 64 (8.0 GB) | fp16 weights are 8.8 GB and do not fit; fp8 fits at 6.27 GB with 1.07 GB free | row 22 |
| Weightless-KV lane, where the memory goes | Qwen3.6-27B GGUF, TP=3, lane on | `max_total_num_tokens` 67,000 with ownership pinned `[6,5,5]`; the weight-holding head has 4.03 GB free against 14.59 GB on the weightless workers | row 12 |
| Draft-KV placement trade | Qwen3.6-27B FP8, TP=3, DFLASH split vs solo | split spreads draft-KV over 10.9 / 22.9 / 10.1 GB across the three cards instead of concentrating it on rank 0 — a capacity trade for the 4-10% throughput solo wins, selectable per deployment | #160 |
| Cross-rig weight split monotonicity | Llama-3.1-8B bf16, TP=5, ratio 4,3,3,2,1 | 4.41 / 3.55 / 3.55 / 2.46 / 1.66 GB across 5090 / 3080 / 3080 / 2080 Ti / Vega 64 — monotone with the ratio. The context ceiling on that rig is set by the drafter's `max_position_embeddings`, not by memory | Nordstern L0 |
| KV re-scatter cost of the single-node PD split | Qwen3.6-35B-A3B AWQ, fp8 KV | 10.1 KiB/token = 1.2 GB for a whole 120k prompt, about 0.2 s of PCIe time against the prefill it saves | row 14, PD doc |

#### Concurrent sessions, prefill, and restore

| Comparison | Configuration | Measured | Source |
|---|---|---|---|
| Single-node PD disaggregation, time to first token | Qwen3.6-35B-A3B AWQ, fp8 KV, ctx 131,072 both sides, no CUDA graph, prefill chunked 2048 solo on the 5090, decode at uneven TP=3 `1,3,3` with weighted DCP, 2-3 runs each, spread < 3% | prompt 2,048: 0.49 -> **0.15 s (3.3x)**; 8,192: 2.27 -> **0.46 s (4.9x)**; 32,768: 9.73 -> **3.44 s (2.8x)**; 122,880: 42.2 -> **18.9 s (2.2x)**. Proxy and handoff overhead is 40-60 ms. Decode pays for it: -13% at 2k context, -2% at 32k | row 14, PD doc (#99 M3) |
| PD disaggregation under SM contention | 32k prefill burst during a 512-token decode | decode 29.27 -> 31.25 s (+6.7%), the prefill burst itself 3.44 -> 3.54 s. No mitigation knob was needed | same |
| Session KV spill, 2 co-resident sessions, victim time per verify round | Qwen3.6-27B FP8, TP=3 uneven, NEXTN k=3, pool 4200, prompt 1200, holder 1000 / victim 1400, restore margin 1024, hysteresis 40; mean of 3 boots per arm, boot-to-boot sd <= 1.5% | pre-spill on device **41.7 ms** (76.5 tok/s, both arms); during spill at the host floor 131.5 -> **113.5 ms** (7.6 -> 8.8 tok/s); restore transient including the MTP backfill **37.9 ms** (69.8 tok/s); settled and alone 91.5 -> **37.4 ms** (10.9 -> 75.6 tok/s). The backfill costs nothing measurable (37.9 vs 37.4 ms against a 0.15-0.70% noise floor) | #217 |
| same, restore reliability | same | **0 of 9** boots restored the victim before the readiness fix, **3 of 3** after. The gate counted only the free list, not the radix-evictable tokens of a neighbour session that had finished | #217 |
| same, end to end | victim request, 1400 tokens | 71.2 s -> **28.6 s** (2.49x). Re-cut on the same raw window (truncated at the next spill instead of at request end), the older run gives victim **76.98 tok/s** post-restore and holder **76.06** — the previously published 15.97 / 63.52 averaged a window that was 90% a second, never-repaired spill | #217; KV-offload block |
| Admission headroom rule | 2-session spill choreography, `H = P - 2p` | `holder_new + victim_new <= H/0.73`; bounded empirically between 2400 (admitted) and 2500 (refused) against H = 1800 | KV-offload block |
| Per-session throughput during a real overlap: one spilled session against a device-resident one | Qwen3.6-27B FP8, TP=3 uneven DCP, NEXTN topk 1, three sessions with the FCFS-newest victim spilled, anti-starvation floor 8 | spilled session **2.75-2.77 tok/s**, concurrent device session **53.8-57.9 tok/s** against a ~85 tok/s solo baseline — the device session keeps 63-68% of solo. Victim inter-token gap never exceeds 0.41 s, exactly the 8-iteration cadence. Overlap actually occurred in 7 of 12 and 10 of 13 runs; the light regime never spills | Per-session QoS measurement |
| same, regulator vs a naive static tick | same choreography | floor 8: spilled 2.75 / device **57.9**. Static tick=1: spilled 7.50 / device **23.4** — the regulator buys about 2.5x device throughput at the cost of the spilled session's latency | same |
| GGUF vs FP8 under concurrency, aggregate tok/s | Qwen3.6-27B, TP=3 uneven auto-performance, ctx 8192, CUDA graphs, spec off, one fixed 8-prompt set (one content class only, see the note below) | 1 / 2 / 4 / 8 sessions — FP8 37.8 / 81.7 / 156.7 / 270.7; GGUF Q8 50.2 / 91.3 / 157.3 / 203.3; GGUF Q4 65.2 / 106.8 / 153.0 / 189.3. Scaling from 1 to 8 sessions: FP8 **7.2x**, GGUF Q8 4.1x, GGUF Q4 2.9x. With spec on, GGUF drafts better (accept 2.70-3.18 against 2.37-2.63) and is still 30-42% slower at 4-8 sessions, so the cause is the verify/decode kernel, not draft quality | #157 |
| same, per-session tok/s under load | same, spec on | FP8 74.8 -> 62.5 -> 53.9 -> 41.7 across 1 / 2 / 4 / 8 sessions; GGUF Q8 87.3 -> 58.3 -> 35.5 -> 29.3; GGUF Q4 87.8 -> 54.7 -> 31.3 -> 26.8 | #157 |
| Single vs dual session on the KV-split arms | Qwen3.6-27B FP8, TP=3 | aggregate tok/s 78.27 -> 149.28 (code), 69.01 -> 121.78 (prose), 73.26 -> 130.44 (mixed) | #203 |
| Batch scaling | Qwen3.6-27B FP8, TP=3 | decode 40 tok/s at bs=1 up to 427 tok/s at bs=16 | #146 |
| Concurrent prefill, full fork program vs stock flags on the same binary | Qwen3.6-27B FP8, 8 simultaneous 1172-token prompts, `cached_tokens=0` on every request | aggregate prefill 1221.6 tok/s against 3000.8 for stock flags at PP=3 uneven. The fork's prefill is already saturated at M=1 (1155.9 -> 1221.6, +5.7%) while the stock pipeline split goes 1495.0 -> 3000.8 (+101%). Decode on the same pair runs the other way, 91.92 vs 28.28 tok/s — but with both sides shackled identically (no MTP, no overlap scheduler) the parallelism axis alone is 33.46 vs 35.73, i.e. 6.8% in the stock split's favour | Baseline block |
| Collective share of the decode span | Qwen3.6-27B FP8, TP=3 uneven DCP, MTP, flashinfer, CUDA graphs | 252.2 of 1600 ms single session (15.8%), 415.9 of 1760.9 ms dual (23.6%) — the comm fraction grows with batch because collectives scale with tokens while the weight-bound GEMMs at bs=1..8 do not. NCCL is already at its floor (27.7 us per bf16 all-reduce against a 31-37 us microbenchmark floor), and at these batch sizes there is no independent compute in the layer to hide it behind | #199 |
| Prefill ceiling on gfx900 after the triton-gcn5 port | Vega 64, real server run | a 10,189-token prompt in 5.4 s (1879 tok/s), where the previous path ran out of memory from about 4k | Cross-vendor block |

#### How these numbers were taken

Raw tok/s follows the output content on this rig (r = 0.90) and carries a 2.6-4.2% boot-to-boot
spread — 2.60% is the worst directly measured excursion — against 0.09-0.85% for ms per verify
round, so anything needing finer resolution than ~3.5% between two arms is stated on the
round-time axis, and each campaign measured its own detection limit A-vs-A before judging any arm.
Decode is taken by slope over two generation lengths at one prompt, or by streaming where a
repeated prompt would hit the radix prefix cache; end-to-end slope figures taken against a cached
prefill were retracted rather than corrected (#204, Nordstern L0). Rows marked *within spread* are
leads, not gains. Two limits apply to specific rows: the concurrency sweep predates the mandatory
code/prose/mixed content axis and measured one content class, so only its arm-vs-arm statement is
load-bearing; and the energy figures come from commit-recorded single runs with GPU board power
only, not from medians with a stated spread.

---

## Detail sections

<a id="f1"></a>
### 1. Asymmetric tensor parallelism

**Feature:** (`--rank-tp-ratio auto`) unequal per-rank shard sizes within one TP group — see Core
concepts.

**Fork status:** Boot-checked — validated TP=3 on 1x RTX 5090 + 2x RTX 3080 (Qwen3.6-27B FP8);
greedy decode is self-deterministic (byte-identical run-to-run/cold-vs-warm on the same GPUs, not
a cross-hardware claim — see row 11). Includes 3 correctness fixes: an `o_proj`-vs-head-split
reject-guard for 3 architectures whose attention silently used the wrong head split (`dd68fad951`);
DFLASH per-rank attention/MLP shards (`5af72c7a60`/`734f77e313`), validated green (arm I, MLP
units `[68,34,34]`); a `kv == tp` replicated-KV widening that was implemented, GPU-measured, and
**reverted** (dies on first forward, see Guarded/descoped below).

**Upstream:** replaces sglang's requirement of equal, head-divisible shards.

<a id="f2"></a>
### 2. Asymmetric decode context parallelism

**Feature:** (`--rank-kv-ratio`) capacity-weighted per-rank KV ownership during decode — see Core
concepts.

**Fork status:** Cross-checked — token-split variant validated. The arg-gate dependency on a
non-uniform `--rank-tp-ratio` was audited and confirmed genuine, not arbitrary. A silent-ignore
defect in `resolve_cp_token_ratios` (an explicit token vector with no plan booted green but did
nothing) now hard-rejects instead (`4c90038a78`, falsifikator-geprueft: booted green while silently
doing nothing). Separately found and
guarded: stock (non-fork) `--dcp-size N` under the Triton backend silently corrupts output when KV
heads aren't replicated across the DCP group — the fork's own uneven-DCP geometry is exempt.

Two independent-reference checks, both on the main rig (27B FP8, TP=3, 5090 + 2x 3080):
- **#173 G4** (Triton uneven-DCP vs. a DCP-off ground truth, greedy, no spec): `short_code`
  byte-identical arm-for-arm. The same run's `chunked` (11650-token) prompt also matched, but sits
  past the ~109-token fp8@3080 boundary (see legend) and is excluded from the tier.
- **#180 V4** (Triton vs. flashinfer, chain speculative verify under uneven DCP, 27B FP8 TP=3, MTP,
  greedy, CUDA graphs on, 4 prompts): token ids identical arm-for-arm on the 3 short prompts;
  `meta_info.spec_accept_length` in the same band. The 4th prompt (11650 tokens) is separately on
  record as cache-state-sensitive on the Triton lane (not attributed to #180) and, combined with
  the fp8@3080 boundary, is excluded.

**Upstream:** replaces sglang's DCP, which only splits KV evenly across ranks.

<a id="f3"></a>
### 3. Rank-to-GPU mapping and co-location

**Feature:** (`--rank-gpu-id`, `--rank-gpu-memory-mib`) assigns each rank to an NVML-resolved
physical GPU; duplicates co-locate multiple ranks on one GPU.

**Fork status:** Boot-checked — co-location itself is exercised on real hardware via row 18's
TP=5-on-3-cards boot (#62), which depends on this mapping. Co-location requires NCCL >= 2.30
(shipped in the fork's Docker image). `--rank-tp-ratio`/`--rank-kv-ratio` no longer require
`--rank-gpu-id` to be set
(`c51dd9c371`): sharding-ratio validity and physical placement are independent concerns, and
coupling them blocked the cross-vendor case (row 21), where NVML cannot name an AMD rank.

**Upstream:** sglang places ranks via `CUDA_VISIBLE_DEVICES` only; this adds an explicit per-rank
physical-GPU mapping.

<a id="f4"></a>
### 4. Solo drafter placement

**Feature:** (`--speculative-draft-placement solo`) runs the draft model unsharded on one GPU,
broadcasting its output instead of all-reducing.

**Fork status:** Built — registered unit tests (solo placement, weight/KV planning, vocab
broadcast); no hardware boot recorded for this row.

**Upstream:** no equivalent flag in sglang.

<a id="f5"></a>
### 5. Cross-algorithm drafter routing

**Feature:** (`--speculative-cross-algorithm*`) NEXTN/MTP and DFLASH resident simultaneously,
switched per batch by a bandit controller (accept-tokens/round), rank-0 decision + TP broadcast.

**Fork status:** Work in progress — dual residence, per-batch switching, and the bandit controller
are implemented (registered bandit test); the context-length gate from the drafter training config
is not yet implemented. Lazy single-graph capture + DFLASH context-retirement (#156-4, `f2c96f31b3`)
is merged and validated green under CUDA graphs (arm C: 542.0 MiB released; arm G: full stack
green).

**Upstream:** no equivalent in sglang, which adapts or selects a single drafter's parameters.

<a id="f6"></a>
### 6. CUDA graph memory aliasing for spec branches

**Feature:** (#93/#102) inactive speculative-depth CUDA-graph branches hold no physical VRAM via
cuMem tag aliasing.

**Fork status:** Boot-checked — `kv_vmm_backing` / adaptive runtime state; the only GPU number
recorded against this aliasing path is row 5's #156-4 arm-C boot (`f2c96f31b3`, 542.0 MiB released
under CUDA graphs).

**Upstream:** sglang has related VMM/cuMem machinery (a multi-spec-graph roadmap item) not yet
applied to speculative CUDA-graph branches.

<a id="f7"></a>
### 7. MoE expert offload + asymmetric TP/DCP

**Feature:** MoE expert offloading to host RAM combined with asymmetric TP and DCP (GPTQ/AWQ/FP8).

**Fork status:** Boot-checked — validated on a 122B-A10B MoE across three mismatched GPUs.

**Upstream:** sglang offloads weights layer-granularly (`--cpu-offload-gb`), not expert-granularly,
and not combined with asymmetric TP/DCP.

<a id="f8a"></a>
### 8a. Bespoke GGUF adapter framework

**Feature:** (#129) `gguf_registry` + `GGUFAdapterBase`: per-model-family GGUF loaders (name maps +
inverse weight transforms) on top of the generic GGUF path, plus sibling-file config/tokenizer
loading for archs the generic metadata reader can't parse.

**Fork status:** Boot-checked — registry with two families; unit tests (header, sizing). Boot
evidence comes from rows 8b-8f, which load through this registry on real hardware.

**Upstream:** sglang's generic GGUF path can't load these arches.

<a id="f8b"></a>
### 8b. Qwen3.5/3.6 GGUF

**Feature:** GGUF arch `qwen35`/`qwen35moe`: GDN/RMSNorm/`out_proj` inverse transforms, plus
NEXTN/MTP draft (including MoE draft) loaded from the same file.

**Fork status:** Boot-checked — dense + MoE + NEXTN/MTP; K-quants Q4_K_M...Q8_0 coherent and
greedy-deterministic; validated Q6_K at asymmetric TP=3 (5090 + 2x 3080).

**Upstream:** unsupported in sglang.

<a id="f8c"></a>
### 8c. Gemma-4 GGUF

**Feature:** GGUF arch `gemma4`, dense: inverse transforms distinct from Qwen (dequantized
`token_embd`, identity norm handling, tied `lm_head`, `k==v` shard duplication).

**Fork status:** Boot-checked — Gemma-4-31B-it Q4_K_M validated (TP=1 on RTX 5090, ~61 tok/s,
coherent + self-deterministic; asymmetric TP=3 green). MoE/MTP/vision Gemma-4 GGUF deferred
(fail-fast); only Q4_K_M empirically verified.

**Upstream:** unsupported in sglang.

<a id="f8d"></a>
### 8d. GGUF K-quant compute kernels

**Feature:** (`sgl-kernel` MMQ/MMVQ) tuned K-quant kernels: per-device MMVQ<->MMQ crossover,
prefill-oriented MMQ cap, batched MMVQ, quantized vocab/embedding, I-Matrix quant.

**Fork status:** Boot-checked — merged from `feat/kquant-kernel`, kernel tests. Opt-in
`--gguf-mmq-decode-threshold` (#163, default OFF): measured **+9.7-10.6%** end-to-end tok/s
(Qwen3.6-27B UD-Q8_K_XL, TP=3, CUDA graphs) but only the sm120 rank reroutes (confirmed by per-rank
kernel-call counts: 11320 MMQ / 0 MMVQ on TP0, 0 MMQ / 11320 MMVQ on TP1/TP2 — 2 of 3 ranks
unaffected). **Not byte-identical when ON** (MMQ/MMVQ reduce in a different order); flag OFF
reproduces the prior dispatch exactly.

**Upstream:** sglang has the base MMQ/MMVQ kernels; the crossover/cap/quantized-vocab tuning is
fork-only.

<a id="f8e"></a>
### 8e. Asymmetric-TP x GGUF correctness

**Feature:** composes GGUF with row 1: K-quant superblock alignment, GDN/MoE per-rank block
coarsening, GGUF-MoE out-of-bounds expert-id fixes, per-rank local-expert-count guard; same
alignment applied to compressed-tensors AWQ/GPTQ INT4.

**Fork status:** Boot-checked — a series of merged bugfixes (#80, #81, #82, #109) with registered
tests. The #82/#109 class (out-of-bounds expert ids, K-quant superblock alignment under uneven
sharding) was found via real GPU crashes/reads (falsifikator-geprueft — each guard test corresponds
to a reproduced hardware fault).

**Upstream:** n/a — asymmetric TP is absent from sglang, so this bugfix class doesn't apply there.

<a id="f8f"></a>
### 8f. Multimodal and dynamic-quant GGUF

**Feature:** load a vision tower from a companion `mmproj` GGUF; load unsloth "UD" dynamic-quant
GGUFs (mixed precision).

**Fork status:** Boot-checked — UD Q6_K_XL (+ mmproj) validated in the benchmark matrix; UD
Q8_K_XL infeasible on the reference rig (size + a known Q8 loader limitation).

**Upstream:** sglang's generic path doesn't load these variants for the affected arches.

<a id="f9"></a>
### 9. Hibernate checkpoint/restore

**Feature:** (#89) persists warm server state to disk so it survives process exit and reloads
without full re-initialization.

**Fork status:** Boot-checked, validated for dense GGUF (load 50s -> 8-14s under asymmetric
TP=3, survives process exit). The FP8 path is functional with negligible load-time benefit;
MoE-model hibernation deferred.

**Upstream:** sglang has diffusion-server offload/wake-up only, no full LLM-server snapshot.

<a id="f10"></a>
### 10. Measured VRAM budget

**Feature:** (`--rank-gpu-memory-mib`, component registry) per-rank absolute MiB budget derived
from measured component usage rather than a global fraction.

**Fork status:** Boot-checked — per-rank absolute MiB budget plus a self-calibrating KV split (boot
logs a vector hint fed back on restart).

**Upstream:** sglang uses a fraction-based global setting (`mem-fraction-static`), no per-rank
absolute budget.

<a id="f11"></a>
### 11. Cross-architecture speculative determinism

**Feature:** verify-sync and CUDA-graph padding across sm86 + sm120; sampling broadcast from rank
0.

**Fork status:** Boot-checked — three divergence root causes resolved; the emitted greedy token
sequence is reproducible across the mixed-architecture TP group (not bit-identical activations:
sm86/sm120 reduce in a different order). Agreement is enforced by the rank-0 sampling broadcast,
not an independent per-architecture comparison.

**Upstream:** sglang has single-architecture determinism modes; mixed-GPU-architecture TP groups
aren't addressed.

<a id="f12"></a>
### 12. Weightless-KV lane

**Feature:** (`--weightless-kv-fastlane`; unrelated to row 16's fast-lane scheduling despite the
shared name) a meta-device worker holds only KV cache and attention while a separate head holds
the weights.

**Fork status:** Cross-checked — chunked prefill and graph-decode paths in place. The lane's
determinism harness (#124) checks output against a TP=1 solo run as reference oracle. Includes
per-ROLE KV storage precision (`--weightless-kv-worker-cache-dtype`, opt-in, default off): the
weightless workers can hold their KV token-shard in fp8 while the head keeps its own format, since
KV bytes cross the role boundary only in the model compute dtype. Whether that buys capacity
depends on which rank binds the min-reduced token budget — the boot log names it (see
`docs_new/weightless_kv_role_precision.md`). Chain speculation
(EAGLE/EAGLE3/NEXTN at `--speculative-eagle-topk 1`) composes with the lane as of #143 via
`--speculative-draft-placement solo` hosted on the lane's head rank; tree verify, adaptive
draft length and the block-decode/host-spill tier stay hard-rejected alongside it.
Design: `docs_new/weightless_chain_spec.md`.

**Upstream:** no equivalent in sglang.

<a id="f13"></a>
### 13. Rig dashboard / planner UI

**Feature:** capacity-planning tool reporting work-normalized J/token under asymmetric DCP.

**Fork status:** Highly experimental — functional but under active development, not
production-ready (`tools/rig_dashboard`).

**Upstream:** n/a — external tooling, not a comparable upstream capability.

<a id="f14"></a>
### 14. Single-node PD disaggregation

**Feature:** single-node heterogeneous prefill/decode split: prefill solo on the fastest card
(TP=1), decode distributed under asymmetric-TP/DCP, with GDN/Mamba state handoff.

**Fork status:** Boot-checked — single-node PD pair green (#99 M1/M2), token-vector KV re-scatter,
crash-robust.

**Upstream:** sglang provides base PD-disaggregation; the single-node solo-prefill +
asymmetric-TP/DCP decode + GDN handoff is the fork's own delta on top of it.

<a id="f15"></a>
### 15. Asymmetric-TP quantization correctness

**Feature:** asymmetric-TP quant correctness + upstream quant bugfixes: GPTQ-MoE `w2_scales`
TP>1 fix, AWQ marlin zero-point staging fix, `moe_wna16` K-mask fix, compressed-tensors/AutoRound-
int4 group alignment.

**Fork status:** Boot-checked — bugfixes #83, #85, GPTQ `w2_scales` (symmetric + asymmetric), the
latter found during row 7's real 122B-A10B MoE boot campaign (falsifikator-geprueft — the stock
load defect reproduced on hardware before the fix).

**Upstream:** sglang has the underlying quant methods but a genuine stock GPTQ-MoE TP>1 load defect
(fork-fixed here) and no asymmetric-TP alignment.

<a id="f16"></a>
### 16. Fast-lane priority scheduling

**Feature:** (`--enable-fast-lane`) opt-in latency-priority class that preempts a tagged request
into the running batch, with a reserved-heavy-slots floor + heavy-aging; default off.

**Fork status:** Built — Variant C Stage 0 (`--enable-fast-lane`, `--fast-lane-priority`,
`--fast-lane-reserved-heavy-slots`, `--fast-lane-heavy-aging-ms`); no hardware boot recorded for
this row.

**Upstream:** sglang already has priority scheduling/preemption; this reserved-floor fast-lane
class is the fork's addition on top.

<a id="f17"></a>
### 17. HiCache under asymmetric-TP/DCP

**Feature:** makes sglang's tiered KV cache (host-RAM L2 + file L3) correct under non-uniform
per-rank layouts: global-to-owned-compact index translation, an NCCL-deadlock fix, a hybrid-SWA
host-pool fix.

**Fork status:** Boot-checked — DCP index translation + prefetch-deadlock + host-pool fixes; the
deadlock was reproduced live before the fix (falsifikator-geprueft).

**Upstream:** HiCache itself is upstream sglang; correctness under the fork's non-uniform layouts
is the delta.

<a id="f18"></a>
### 18. TP greater than num_kv_heads

**Feature:** replicated-KV + token-sharding: lets TP degree exceed the model's KV-head count —
and, via co-location, the physical GPU count — including GQA re-grouping to single-head
geometries.

**Fork status:** Boot-checked — validated TP=5 on 3 cards via co-location (#62).

**Upstream:** sglang already replicates KV under GQA when `tp > kv_heads`, but not combined with
asymmetric-TP/token-sharded DCP.

<a id="f19"></a>
### 19. Broad model bring-up under asymmetric-TP

**Feature:** Qwen3.6-27B (GDN) and 35B-A3B (MoE) at asymmetric TP=3; Gemma-4 31B dense and
26B-A4B MoE SWA-hybrid; small/replicated-KV models.

**Fork status:** Boot-checked — per-model; Gemma-4 EAGLE3 head fix (#101), 26B-A4B boot fix,
`--swa-pool-sizing`.

**Upstream:** n/a — model-support work specific to the fork's own asymmetric-TP/speculative code,
not a general capability comparison.

<a id="f20"></a>
### 20. Session KV spill

**Feature:** (`--enable-kv-session-offload` + tick/margin/hysteresis flags) on VRAM overflow, the
newest in-flight request's KV shard offloads to host RAM while it keeps decoding; FCFS-by-arrival
victim order.

**Fork status:** Experimental — S1 scope only (single spilled request, eager path); overlap /
multi-request planned, not built. RAM-budget-sized host pool (1.00 GB/rank of 24 GiB), prefill-side
spill, and a configurable wave-back threshold are all green in the boot matrix; a sentinel-row
radix-tree leak was found and fixed (`c49472949a`). Two scenarios explicitly **not validated**:
spec-in-tick spill coincidence, 3-session co-residency.

**Upstream:** sglang retracts (frees + re-prefills) on exhaustion rather than keep-decoding-while-
spilled.

<a id="f21"></a>
### 21. HTCCL cross-vendor collectives

**Feature:** (`SGLANG_HTCCL_TRANSPORT` = `gloo`/`shm`/`device`/`ucx`) vendor-neutral TP collectives
that never call NCCL/RCCL, so one TP group can mix an NVIDIA and an AMD GPU; the `device` transport
reduces on-GPU over host-mapped memory and is CUDA-graph capturable. The `ucx` transport is the
cross-**host** data plane (Nordstern L1): same host-staged semantics as `gloo`, RDMA instead of
TCP.

**Fork status:** Cross-checked — merged into `integration/r3-probe` (`73679d6b47`,
`9a10846a82`, plus `feat/htccl-gfx900`'s `aec1308973`). Correctness: known-answer tests per
collective/dtype/world-size/transport vs `torch.distributed`; 2 real bugs found and fixed
pre-cross-vendor (an output-buffer aliasing defect; a `reduce_scatter` wrong-axis defect for
`dim >= 2`, `8acd4221a3` — falsifikator-geprueft: RED on the old axis, GREEN after the fix on all
three transports). **Cross-vendor (2080 Ti sm75 + Vega 64 gfx900), eager:** byte-exact vs
`torch.distributed`; model-scale byte-identical to `gloo` (Qwen3.5-4B, even 2/2 and uneven 3,1);
`device` transport **+37%/+48% decode, +45%/+62% prefill vs `gloo`** (16.51 vs 11.13 tok/s
uneven-decode). **Cross-vendor with CUDA graphs: in reach, not demonstrated** — 4 CUDA-only
assumptions were found and fixed on gfx900 in sequence; the last (`fa5c507476`, a device-side
`assert()` that fails kernel launch entirely on gfx900) is on `feat/htccl-gfx900` (tip
`3cc2fc9da5`), **not yet merged**; symmetric decode capture and a separate NVIDIA-side
prefill-capture assertion remain open.

**`ucx` transport (Nordstern L1) — cross-host RDMA data plane.** One registry entry
(`TRANSPORT_REGISTRY["ucx"]`) plus `htccl_ucx.py` / `htccl_ucx_bindings.py`; no dispatch site
changed. Serves `all_reduce`, `all_gather`, `broadcast`, `reduce_scatter`, plus an internal
`barrier`. Host-staged like `gloo` (GPU -> pinned host -> UCX -> pinned host -> GPU); there is no
GPUDirect on this hardware, so that is the only path, not a simplification.

*Binding decision — ctypes over `libucp`, not ucx-py/Cython.* The deciding constraint is that
version parity requires loading a **specific** library path. `ucx-py` bundles or links its own UCX
(RAPIDS `libucx-*` wheels), which hard-codes one side of the mismatch instead of resolving it, and
is asyncio-shaped — the wrong control flow for a synchronous collective on the bs=1 decode path.
Cython needs a compiler and UCX **headers** per host; the second rig has the runtime libraries but
not the `-dev` package. A subprocess bridge adds an IPC hop to a ~1.5 us path. ctypes needs no
build step, no headers, and takes the library from `SGLANG_HTCCL_UCX_LIB`. Struct layouts are
mask-driven and over-allocated, and the transcribed offsets are asserted at import.

*Version parity is enforced, not hoped for.* Mixed UCX releases do not degrade — endpoint creation
fails with the useless `invalid bandwidth 0.00`. The rendezvous gathers each rank's version over
the existing gloo `cpu_group` and refuses **before any endpoint exists**, naming every rank's
version and library path and the `SGLANG_HTCCL_UCX_LIB` remedy.

*Latency shaping.* Default is a single-step full exchange (all receives and sends posted together,
one round trip at any world size); `all_reduce` switches to a ring above
`SGLANG_HTCCL_UCX_RING_MIB`. Endpoints are persistent and wired up during construction, so no
decode step pays a UCX handshake. `handles()` is deliberately **size-independent** (unlike `shm`'s
slot ceiling) so two ranks can never disagree about whether a collective goes over UCX or gloo.

**Fork status (`ucx`):** Implemented, validated CPU-only on real RDMA — not yet exercised with a
GPU or a model. Local (single host, UCX `self`/`sm`/`tcp` loopback): 16 tests green across
world 2/3/4, every collective vs a computed reference, incl. the buffer-aliasing and
`reduce_scatter dim>=2` traps, forced multi-chunk transfers, and idempotent teardown; the 47
pre-existing HTCCL tests still pass. **Cross-rig over 40G RoCE** (main rig 10.10.10.1 <-> second
rig 10.10.10.2, `UCX_TLS=rc` with no TCP fallback, so a pass proves RDMA carried it): all
collectives green on both ranks; rendezvous + wireup 0.11 s. Throughput, median, per direction
while the reverse direction runs simultaneously — 8 KiB 35 us / 1.8 Gbit/s, 64 KiB 75 us /
7.0 Gbit/s, 512 KiB 355 us / 11.8 Gbit/s, **4 MiB 1.61 ms / 20.8 Gbit/s (peak)**, 32 MiB 23.6 ms /
11.4 Gbit/s; barrier median 12 us, min 9.9 us. Raw link the same day, `ucx_perftest tag_bw`
unidirectional: 3413 MB/s (~27.3 Gbit/s), so the 4 MiB peak is **~76% of the unidirectional
budget while moving traffic both ways**. **Two known gaps, both diagnosed:** (a) small-message
cost is software, not wire — a 12 us barrier against a ~1.5 us link is Python/ctypes per-call
overhead, already cut roughly in half by memoising staging views and reusing the ctypes request
struct, and the floor for bs=1 decode until the call path is shortened further; (b) the 32 MiB
regression vs 4 MiB is CPU-memory-bound, not link-bound — V1 makes four passes over the payload
(stage in, wire, accumulate, stage out), ~4x32 MiB of host traffic that is not overlapped with the
wire. Chunked pipelining of D2H against the transfer is the fix and belongs to L2.

**L2 (task #198) — transport-level speed-up. Both L1 gaps closed; measured, not projected.**

*Where the 12 us barrier actually was.* Profiled on the real link (phase timestamps inside the
barrier, 2000 samples), and cross-checked against a same-host `self/sm/tcp` run whose barrier was
**10.9 us** — i.e. the wire contributed essentially nothing and the number was ~100% software.
The 12.37 us split: setup 1.43 (a `fill_(0)` on a byte nobody reads, plus a peer dict rebuilt with
f-string keys per call), posting 4.29 (two ctypes crossings wrapped in eagerly formatted error
strings, per-post `data_ptr`/`numel`/`element_size`, a `c_void_p` object per post), waiting 4.13
(median 5-6 spin passes, each rebuilding a list and reading the clock). So ~7 us of the 12 was
bookkeeping around two library calls.

*Where the 32 MiB regression actually was.* Measured directly: the four host passes over 32 MiB
cost **9.4 ms single-threaded**, and a 32 MiB buffer runs at ~13 GiB/s versus ~34 GiB/s for a
4 MiB one because it no longer fits in L3. Against a ~13 ms wire budget that fully accounts for
the 23.6 ms observed. So the fix is worth about as much for *cache locality* as for overlap.

*What changed.* (a) **Pipelined `all_reduce`** — the payload is processed chunk by chunk out of
two rotating slot sets, scheduled `stage-in(k+1) -> wait(k) -> post(k+1) -> finish(k)`, so the
stage-in sits under chunk k's transfer and the accumulate + copy-out sit under chunk k+1's.
(b) **Progress interleaving** — the staged copies call `ucp_worker_progress` every
`SGLANG_HTCCL_UCX_PROGRESS_KIB` (default 256). This is load-bearing, not defensive: without it the
rendezvous handshake for the next chunk cannot start while the CPU is inside a several-hundred-
microsecond memcpy, and the measured 32 MiB figure falls from 17.4 to **13.8 Gbit/s**.
(c) **Short small-message path** — precomputed barrier slots and staging records
(`view, ptr, nbytes` memoised together), error strings built only on failure, bound ctypes function
objects, a single-request fast path in `wait`, and a dedicated single-chunk branch in `all_reduce`
so a decode-sized collective never builds the pipeline's closures. `SGLANG_HTCCL_UCX_PIPELINE=0`
restores the old path as an A/B control.

*Cross-rig result*, 40G RoCE, `UCX_TLS=rc`, world 2, median of 5 reps of per-cell medians, both
directions live:

| cell | L1 (#117) | L2 (#198) | |
|---|---|---|---|
| 8 KiB | 37.0 us / 1.77 Gbit/s | **26.6 us / 2.47 Gbit/s** | -28% latency |
| 64 KiB | 60.2 us / 8.71 | **50.4 us / 10.39** | -16% |
| 512 KiB | 232.9 us / 18.01 | **223.3 us / 18.78** | -4% |
| 4 MiB | 1625.7 us / 20.64 | **1583.3 us / 21.19** | -3% (was already at the wire) |
| 32 MiB | 24.30 ms / 11.05 | **15.42 ms / 17.41** | **-37% / +58%** |
| barrier | 12 us (min 9.9) | **5.5 us** | -54% |

The 32 MiB point moved from 53% of the 4 MiB peak to **82%** of it. The barrier's remaining 5.5 us
is now 0.20 setup / 2.25 posting / 3.08 waiting, i.e. mostly the ctypes marshalling floor for two
5-argument calls plus 5 poll passes over a ~1.5 us link — the next lever there is fewer crossings,
not cheaper ones.

*Sweep, for whoever tunes further:* at 32 MiB, `CHUNK_MIB=2` gave 17.72 and `CHUNK_MIB=8` gave
17.34 Gbit/s against 17.41 at the default 4 — within ~1.5x the run-to-run spread, so the default
was left alone rather than tuned to this one link.

*Deliberately NOT pipelined:* the ring `all_reduce` (world > 2 only, and this fleet has two rigs,
so it cannot be measured here) and `broadcast`. They inherit the cheaper post/wait path but still
make unoverlapped host passes.

**L2 block 2 — `all_gather` pipelined + single-chunk fast path; measured honestly, including the
part that did not move.** `all_gather` now has (a) the same two-parity chunk pipeline as
`all_reduce` (multi-chunk payloads; own rank's slice is copied device-locally inside finish(k) and
never crosses the wire — on a GPU that is a D2D copy) and (b) a single-chunk fast path (the
all_gather twin of `all_reduce`'s flat branch: precomputed slots, no staging dict, no per-call key
formatting; the own slice copies straight from the input). `SGLANG_HTCCL_UCX_PIPELINE=0` keeps the
pre-pipelining path verbatim as the A/B control. Correctness: all references EXACT (atol 0.0 —
all_gather is pure data movement), ramp payloads, chunk-boundary/ragged/2d-axis cases, pipelined ==
unpipelined and fastpath == generic bit-for-bit, in the registered unit test, the local selftest
(world 2+3) and cross-rig over RDMA; a deliberate parity-bug mutation flips 8+ tests red.
Cross-rig numbers (reps 5): 32 MiB 26.4 -> 24.9 ms (**only ~+6%**), 512 KiB 268 -> 242 us, 8 KiB
43.2 -> 41.2 us; all_reduce and barrier unchanged (8 KiB 27.4 us, barrier 5.2 us — the standing
rule that every change re-measures the decode path held). *Why 32 MiB barely moved, profiled not
guessed:* phase timers on the real link show total 22.5 ms = stage 1.4 + wait 8.1 + **finish 12.6**;
the finish pass writes 64 MiB into a FRESHLY allocated output (mandatory — returning a reused
buffer is the `_get_out_buf` aliasing bug), and a fresh 64 MiB CPU tensor costs ~4 ms in pure
mmap/page-fault zero-fill (measured: 6.6 ms fresh+2 copies vs 2.7 ms into a reused buffer) plus
DRAM bandwidth the NIC's RDMA DMA is competing for. The CPU-tensor bench is therefore the WORST
case for all_gather (its host passes are 2x all_reduce's per byte); on the GPU path the finish
copies are H2D/D2D DMAs and the fault storm does not exist. The structure is proven by the
all_reduce numbers; the honest end-to-end number for all_gather comes from the model run, not from
this cell.

**L2 block 3 — small-message path, second pass.** Profiled the 8 KiB all_reduce on the real link
(phase timestamps, 48 samples): stage 4.5 / post 8.1 / wait 8.3 / finish 4.2 us + ~2 us residual
(lock, seq, `empty_like`). Post and wait are the ctypes + UCX-internal + RTT floor; stage/finish
are torch dispatch. What changed: (a) the single-chunk all_gather builds its output FLAT
(`view(world, n)` rows, `select+copy` per rank instead of `select+reshape+copy`, and dim==0 — the
common gather axis — returns a single `view`, no movedim/reshape), with the own-slice copy placed
BETWEEN the posts and the wait so it runs while the wire is in flight (progress-interleaved above
one progress block, so a large single-chunk copy cannot starve its own RNDV handshake); (b) `wait`
gained an n==2 fast path (one recv + one non-inline send, the world-2 exchange shape) holding two
locals instead of rebuilding a list per poll pass. Cross-rig (reps 5): **all_gather 8 KiB
43.2 -> 27.0 us (-37%), 64 KiB 67 -> 49.0 us, 512 KiB 268 -> 219 us** — the small all_gather now
costs the same as the small all_reduce; all_reduce 8 KiB 27.4 -> 26.5-27.1 us; barrier steady at
5.2 us. *Deliberately rejected:* posting the send directly from a CPU input and fusing the last
accumulate into the output (`torch.add(out=)`) — both only fire for fp32 CPU tensors, i.e. they
would have tuned the harness, not the bf16/GPU model path. *C/Cython extension: skipped against
the >2x bar* — it could recover roughly the ~5 us of Python around the two posts and the polls, on
a 27 us collective; the honest ceiling is ~1.3x, and it would add a build step on every host,
which the ctypes design exists to avoid.

**L2 block 4 — async collectives (`all_reduce_async` / `all_gather_async` -> handle,
`wait_async(handle)` -> out).** The three contracts from the design sketch are implemented and
falsified-first: (i) *ownership* — async staging comes from a power-of-two-size-class free-list
pool, acquired at issue, owned by the handle, released only inside `wait_async`; the caller's
input is free the moment issue returns (staged before the first post; all_gather's own slice is
later copied out of the STAGING slot, never re-read from the input); results are freshly
allocated, never pool views; double-wait raises. A deliberate release-at-issue mutation flips 10+
lifetime tests red. (ii) *progress* — issue ends with one progress pass (pushes eager sends onto
the wire); completion happens under the progress loop in `wait_async`; no progress thread (worker
is THREAD_MODE_SINGLE under the transport lock). Eager-sized payloads then move in hardware while
the caller computes; RNDV-sized ones degrade toward sync cost but stay correct. (iii) *order* —
`_next_seq` counts issues under the lock, handles carry their seq, tags match exactly, so waits
may be out of issue order (tested, including a sync collective between issue and wait, and mixed
ar+ag outstanding). *Honest transport-level overlap measurement, CPU tensors over the real link:*
`sync(coll+busy)` vs `issue+busy+wait` shows **nothing hidden** (-6 to +4 us) — as it must: at
8 KiB ~22 of 27 us are LOCAL software on the same CPU the busy loop runs on, only the ~5 us of
wire is hideable, and the handle/pool overhead (~4 us) eats it. The async API's value case is the
GPU consumer, where the model computes on the GPU while CPU+wire run the collective — that is the
87%-idle lock-step stall this API exists to attack, and it is a model-runner measurement, not a
transport one.

**L2 block 5 — consumer-side overlap, `SGLANG_HTCCL_UCX_OVERLAP=1` (default OFF).** The MLP
all-reduce rides the existing `fuse_mlp_allreduce` seam, which is the one legal deferral window in
a dense decoder chain (layer N's down_proj AR is mathematically movable into layer N+1's
`prepare_attn`, where AR + residual + layernorm happen together; everything else in the chain is a
strict dependency). Three touch points, all no-ops with the flag off: (a)
`should_fuse_mlp_allreduce_with_next_layer` gains a group-uniform gate (env flag + transport class
only — no rank-local state, per the rank-local-test-before-collective rule; the structural guards
moe_cp / dp+eagle / input_scattered / SCATTERED / is_last_layer still veto); (b)
`RowParallelLinear.forward` issues `all_reduce_async` at the skip point and attaches
`(comm, handle)` to the tensor exactly like the existing `_sglang_needs_allreduce_fusion` tag;
(c) `prepare_attn`'s fusion branch checks for the handle FIRST (a kernel fusion on unreduced data
with a handle in flight would double-reduce and orphan the requests) and falls back to the
unchanged sync AR when no handle was attached — so an issue-side refusal degrades cleanly.
Communicator plumbing: `HTCCLCommunicator.supports_async/all_reduce_async/wait_async`, with
`supports_async` shaped like `handles()` (payload-independent, group-uniform); covered by seam
unit tests (56 registered HTCCL tests green). *Honest expectation, stated before the E2E run:* the
overlap window is the host-side layer-boundary work (tag, loop, prepare_attn entry — tens of us in
eager mode), so this hides at most the wire+remote share of one AR per layer; the dependency
analysis says the big idle sits elsewhere (straggler lock-step, draft-solo phases). The A/B/C
end-to-end run is the arbiter, not this paragraph.

*Overlap design sketch (analysis) — now implemented through blocks 4-5 above.*

**L2 end-to-end proof (task #198, 2026-07-26).** TP=4 cross-rig working arm (Qwen3.6-27B-FP8,
`--rank-tp-ratio 6,4,4,2 --rank-kv-ratio capacity`, NEXTN 3 + solo draft, triton backend, eager,
ctx 8192, bs=1), slope method, 3 content classes, reps 3, same boot recipe for every arm; only the
8 HTCCL-touched files (and one env flag) differ. Arms: **A** = pre-L2 transport (684ef3dd13
content), **B** = L2 transport (blocks 1-3), **C** = B + `SGLANG_HTCCL_UCX_OVERLAP=1` (activation
proven by the once-per-rank "overlap ACTIVE" log on all 4 ranks), **D** = B with
`--rank-kv-ratio coupled` (perf-oriented KV split, 2080 Ti share 26.6% -> 12.5%).

| slope tok/s | A | B | C | D | B/A | C/B |
|---|---|---|---|---|---|---|
| code  | 16.38 | 17.31 | 17.92 | 18.11 | +5.7% | +3.5% |
| prose | 15.67 | 17.57 | 17.12 | 16.84 | +12.1% | -2.6% |
| mixed | 16.54 | 18.36 | 17.84 | 18.15 | +11.0% | -2.9% |

`spec_accept_length` identical across arms per content (3.08 / 3.03 / 2.91) — the arms generated
the same tokens; zero degeneration anywhere. **Verdict: the transport work IS the end-to-end win
(~+10% mean B/A); the async overlap (C/B) is neutral within run-to-run spread** — exactly what the
dependency analysis predicted (the deferral window is only the host-side layer boundary), and now
it is a measured negative result, not a guess. D (fewer tokens on the slowest card) is also
within spread at THIS benchmark's short sequences (~0.2-2k tokens live context) — attention is a
small compute share there; the KV-split lever belongs to long-context runs (input for #103).
Per-rank GPU utilization, sampled during the measure window of every arm: main rig ~10-13% per
GPU, 2080 Ti ~50-55% — stable across A/B/C/D. The slowest card is >4x as utilized as any other
rank and paces the lock-step group; neither the faster transport nor the overlap moved that
ratio, which makes straggler compute (not communication) the next order of magnitude
(#103 k-matrix / split balance, #200 autotune). Raw results: `/root/crossrig/res_arm_{a,b,c,d}*.json`,
GPU samples `gpu_{a,b,c,d}_{main,rig2}.log` (CT999). The transport is now within ~20% of the
wire at large sizes, which makes the next order of magnitude a *scheduling* problem, not a
transport one: in the TP=4 cross-rig L0 run the main rig was **87% idle**, waiting in lock step.
Every collective here is synchronous — the caller blocks from post to completion — so the model's
compute cannot cover the link. The shape of the fix, mirroring the DCP collective-overlap work
(#128): split each collective into `all_reduce_async(inp) -> handle` and `wait(handle) -> out`,
where the async half stages, posts, and returns immediately, and the wait half progresses the
worker and finishes the accumulate + copy-out. The transport already has the pieces — the request
list *is* the handle, `_pipelined_all_reduce`'s staged slot dict is already a per-collective
carrier, and the pipeline loop already proves the CPU work can be deferred past the post. What it
does not have, and what the follow-up task must add: (i) slot ownership that survives past the
call, so an outstanding handle's staging buffers cannot be reused by the next collective — the
two-parity rotation must become an allocator with a free list, and this is exactly the
shared-buffer trap this file has been bitten by before; (ii) a rule for who progresses the worker
while the caller is off computing, since UCX makes no progress unattended for handshakes;
(iii) an ordering contract, because `_next_seq` assumes collectives complete in issue order.
The consumer side (issuing the all-reduce of layer N and waiting for it after layer N+1's GEMM)
is a model-runner change and belongs with #128's machinery, not here.

*GPU staging end-to-end (TP=2 cross-rig with a real model) is written up but deliberately NOT run
yet* — see the recipe below.

**Recipe — cross-rig GPU/model bring-up (L1 -> L2, not yet executed).**
1. *Preconditions.* Same UCX release on both rigs (main rig: `SGLANG_HTCCL_UCX_LIB=/opt/ucx116/lib/libucp.so.0`
   against the second rig's system 1.16.0); `/dev/infiniband` present in whatever namespace the
   ranks run in; RoCE port ACTIVE. Note the container CT999 used for development has **no**
   `/dev/infiniband` and no 10.10.10.x address — GPU ranks must run where both the cards and the
   NIC are visible, which today means the PVE host or a container configured with both.
2. *Flags.* `SGLANG_HTCCL=1 SGLANG_HTCCL_TRANSPORT=ucx`, `--nnodes 2 --node-rank {0,1}`,
   `--dist-init-addr <LAN ip>:<port>` (control plane stays on the 1 GbE LAN; only UCX rides the
   RoCE link). Per rank: `UCX_TLS=rc,self,sm`, `UCX_IB_GID_INDEX=3`, `UCX_NET_DEVICES=rocep4s0f1:1`
   (main) / `rocep1s0f1:1` (second).
3. *`--enforce-eager` is required.* Like `gloo` and `shm`, this transport synchronises with the
   host inside every collective, so it cannot be inside a CUDA-graph capture. Only the `device`
   transport is capturable.
4. *Model geometry.* Same constraint the L0 run hit: `tp_size <= q/kv` units. Qwen3.5-4B cannot do
   TP=5; for a TP=2 cross-rig smoke test any model that fits one card per rig is fine.
5. *Expected first failures*, in likelihood order: pinned-buffer registration cost on first
   collective (the staging buffers are `pin_memory=True` only when the rank's device is CUDA);
   a rank-uniformity break if any code path issues a collective on one rig and not the other; and
   the ~12 us/collective software floor showing up as poor decode tok/s — which is a known L2
   item, not a bug.
6. *Validation bar.* Byte-identical output vs a solo run on one rig, the same bar the cross-vendor
   `device` transport had to clear.

**Upstream:** sglang's distributed backend is NCCL/RCCL only, never bridged.

<a id="f22"></a>
### 22. fp8 dequant fallback (W8A16)

**Feature:** serves fp8 checkpoints on GPUs without a native fp8 GEMM via a dequant W8A16 path
(compressed-tensors `CompressedTensorsW8A16Fp8`), gated by a functional capability probe rather
than a capability-number comparison (`torch.cuda.get_device_capability()` reports `(9,0)` for both
Hopper and gfx900).

**Fork status:** Cross-checked, GPU-validated cross-vendor — on `feat/htccl-gfx900` (`3cc2fc9da5`),
not yet merged into `integration/r3-probe`. CUDA path unchanged by construction. Correctness:
Qwen3.5-4B-FP8-dynamic, solo Vega 64 vs. solo 2080 Ti, vs. mixed TP=2 uneven 3,1, vs. mixed TP=2
even 2/2 — all **byte-identical** (solo runs as oracle; neither card is in the sm80-88 range the
fp8@3080 caveat covers). Model fits solo on Vega 64 in fp8 (6.27 GB weights, 1.07 GB free) where
fp16 doesn't fit at all. Costs **~23% of decode** vs fp16 at the same TP config (12.67 vs 16.51
tok/s); on this pair the mixed configuration is pointless since the model fits solo on the 2080 Ti
alone (15.23 tok/s). Open: the non-compressed-tensors `fp8.py` `Fp8Config` family (the user's own
27B/35B checkpoints) is not wired to the probe. A separate fused dequant-GEMV kernel for the same
lane (design B, #189) decodes raw fp8-e4m3 bytes **bit-exact against `torch.to(float32)`** (max
diff 0.0), with a tighter fp32 error band than the materialize-then-`F.linear` path it would
replace (mean relative error 0.0014 vs 0.0133). Not yet merged or wired into a model boot; only a
pre-merge semantic desk-check against #192 has been done.

**Upstream:** sglang requires a native GEMM or Marlin (sm80+); no dequant fallback.

<a id="f23"></a>
### 23. Turing/gfx900 without sgl-kernel

**Feature:** lets the server start on GPUs `sgl-kernel`'s cubin-only wheel doesn't cover (floor
sm80) via a two-level capability predicate (`sgl_kernel_importable()`/`sgl_kernel_runnable()`)
instead of platform checks, with real fallbacks (`forward_native`, torch-native sampler backend).

**Fork status:** Cross-checked, GPU-validated on both vendors — Turing support (`0eb7e68880`),
rope/clamp_position routing (`3f0a93ac1c`), a 4th platform-vs-availability bug, this time on the
NVIDIA rank (`621311aa24`, falsifikator-geprueft — reproduced on hardware before the fix). Verified
end-to-end on a real RTX 2080 Ti with `sgl_kernel`
absent: all 11 core modules import, server starts, coherent generation, 608 unit tests pass.
`forward_native` measured **byte-identical between sm75 and gfx900** (not vs. the kernel path,
which differs by a ~4.8e-07-class reduction-order difference). Mixed-vendor TP=2 (Triton,
HTCCL/gloo) reproduced the same token ids as both solo runs — solo runs as the independent oracle
on each vendor. **Scope note:** gfx900 Triton support itself depends on the external
`Said-Akbar/triton-gcn5` fork, not fork code.

**Upstream:** no capability-fallback path in sglang for `sgl-kernel`-class dependencies.

---

### SWA-DCP Stage B (#96)

Status: Cross-checked. Merged into `integration/r3-probe` in the Window-4 merge stack
(2026-07-27) together with `fix/gemma4-textonly-mask` (#186), which is its required partner —
either branch alone leaves H4 red. The ~10 global
full-attention layers of an SWA-hybrid (Gemma-4 class) are token-sharded by the weighted owner
rule of #173; the ~50 sliding-window layers keep their unsharded local path, so no
`(owner slice ∩ window)` masking arises at all (the window-sharding alternative was measured
against and rejected in #91 §4). Requires `--swa-pool-sizing cap` (Stage A, row 19) — in ratio
mode the unsharded SWA pool would be scaled by the *global* context budget. Refused: HiCache,
speculative decoding, MLA, weightless-KV, pure-SWA models. Evidence (Window 3, on the then-
unmerged branch pair): Stage B boots (H4) and a needle planted ~3k tokens beyond the 1024-token
sliding window retrieves byte-identical to a TP=1 solo-5090 oracle (#96-H5); H6/H7 also green.
The **~+6-10% figure remains an ex-ante design estimate, not a measurement** — no throughput
number has been taken; the recipe is `docs_new/swa_dcp_stage_b_triton.md` §8.
Carried along, because Stage B is where it bites: `_plan_aware_dcp_group_q_head_counts` took
`max()` over a hybrid model's two kv-head bases, which is right for a workspace size and wrong
for a collective's per-rank counts — for 32 q heads over bases {16, 8} and ratios [5,3,2] the max
is `[16,10,8]`, sum 34 against a total of 32. Collectives now use the full-attention base with an
exhaustiveness check; single-base models are byte-identical.

---

## Guarded / descoped (implemented in code, deliberately gated off)

Built and evaluated, then gated off — listed for completeness, not shipped as usable capabilities.
No llama.cpp/ik_llama.cpp comparison here: these are internal fork decisions about the fork's own
uneven-TP/DCP machinery, which has no upstream analog (see rows 1/2/18 for that comparison).

- **Tree speculative decoding with `--speculative-eagle-topk > 1` under asymmetric-weighted DCP
  (#76)** — Built and GPU-tested; found silently non-greedy under weighted DCP and
  perf-negative on this rig; restored as a hard fail-fast guard with a CPU test
  (falsifikator-geprueft — reproduced on hardware before the guard).
- **Replicated-KV eligibility widened to `kv == tp` (the `<` -> `<=` flip, row 1)** —
  Built, red/green-tested on CPU, and GPU-measured (falsifikator-geprueft — the CPU test was
  written red-then-green against the flip); the GPU measurement **refuted** it: at
  `kv == tp` the alignment repair that makes uneven splits work at `kv < tp` has no room to
  operate, so it dies on the first forward. Existing `<` semantics kept, with the measured
  rationale pinned in a test. A genuinely uneven `kv == tp` would need a ragged kernel supporting
  per-rank non-uniform GQA mapping, not a threshold change.

## Cross-vendor bring-up: additional bugfixes with upstream relevance, and non-defects

Found during the HTCCL/cross-vendor campaign (rows 21-23); each is a genuine, independently
triggerable defect class. These are SGLang/Triton-backend-internal defects — llama.cpp and
ik_llama.cpp use an entirely different compute stack (ggml), so no comparison column applies.

- **Even-DCP under the Triton backend silently corrupts output** when KV heads aren't replicated
  across the DCP group (only correct when `tp_size // total_kv_heads >= dcp_size`, the geometry of
  upstream `sglang` [#25090](https://github.com/sgl-project/sglang/issues/25090)) — now rejected at
  backend construction; the fork's own uneven-DCP geometry is exempt. Left open: the uneven-DCP +
  dense-model-class combination (also produces mojibake), and flashinfer's silent no-op on plain
  `--dcp-size N`.
- **`o_proj` reject-guard for uneven-TP-unaware attention classes** (`dd68fad951`, folded into row
  1) — avoids trading a shape error for silent wrong numerics on 3 model classes.
- **`--rank-kv-ratio` arg gate silently ignoring an unusable token vector, now a hard reject**
  (`4c90038a78`, folded into row 2).
- **`GraphSharedOutput` — investigated as a suspected shared-buffer defect, confirmed NOT a
  defect.** A falsifier (an unshared, correctly-sized variant) reproduces the shared run
  bit-for-bit; the buffer is obtained once per runner in `__init__`, not per call, so nothing could
  hand it to a live consumer. Verified for the overlap scheduler ON (the shipped configuration);
  `return_logprob`'s separate read path was not exercised.
- **Cold JIT builds collide with the device-collective deadline** (`fix/jit-coldbuild-robustness`).
  `jit_kernel` modules build on FIRST CALL, and several first calls land in the pre-capture warmup
  forward, so ranks reach that forward minutes apart on an empty cache. `HTCCLDeviceTransport`'s
  wait kernels compare `clock64()` against `_TIMEOUT_CYCLES = 60e9` (~23 s at 2.6 GHz) and trap on
  expiry, poisoning the CUDA context; the `cudaErrorLaunchFailure` then surfaces on the NEXT,
  unrelated kernel. Measured: 6/6 boots RED on a cold cache, 1/1 GREEN with the same tree once
  warm, stall 23-30 s. Fixed by a cold-build *window* around the warmup forwards
  (`srt/utils/jit_cold_build.py`) rather than by raising the constant — the recorded pass stays
  outside it, so the deadline baked into the captured graph is unchanged. Opened rank-uniformly and
  unconditionally; a rank-local predicate in front of a group collective is the hang family that
  already produced the pynccl and CustomAllreduce defects. Falsifikator-geprueft: measured 6/6
  boots RED on a cold JIT cache, 1/1 GREEN with the identical tree once warm.
- **The JIT kernel cache does not self-heal** (`fix/jit-coldbuild-robustness`). A build killed
  mid-flight leaves `build.ninja` + `cuda.cu` + `cuda_0.o.d` and no `.so`; every later process then
  dies with `Check failed: (lib_handle_ != nullptr)`. Four such directories accumulated on the r3
  host and had to be removed by hand — one interrupted boot turns into a permanent failure.
  `jit_kernel/cache_health.py` classifies entries (complete = a `.so` exists, and is never touched)
  and discards poison, with a host+pid build marker so a co-located rank's in-flight directory is
  not mistaken for wreckage.
- **Validator hygiene:** the campaign's own output-corruption validator mis-scored a healthy,
  math-heavy sample as `CORRUPT`; the faulty letter-fraction rule was removed rather than tuned.

## Scope note

This matrix lists only capabilities with landed code (verified directly against
`integration/r3-probe` and, where noted, the not-yet-merged `feat/htccl-gfx900`). Planned or only
partially prototyped items — a host-RAM tiered-KV fabric for the weightless lane, a draft-KV-pool
DCP layout, symmetric cross-vendor CUDA-graph capture (row 21), and the `fp8.py` `Fp8Config` family
on non-CUDA-native hardware (row 22) — are intentionally excluded until they land.

## Sources

**Speculative decoding (upstream and research)**
[vLLM DynamicProposer](https://github.com/vllm-project/vllm/pull/26504) · [DSL RFC](https://github.com/vllm-project/vllm/issues/36657)
· [Automate Spec RFC](https://github.com/vllm-project/vllm/issues/4565) · [Disaggregated Standalone Draft RFC](https://github.com/vllm-project/vllm/issues/42109)
· [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
· [SGLang roadmap #23705](https://github.com/sgl-project/sglang/issues/23705) · [AutoSpec RFC](https://github.com/sgl-project/sglang/issues/15319)
· [Dynamic SPD](https://github.com/sgl-project/sglang/issues/9319) · [SGLang adaptive spec docs](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding)
· [SGLang speculative decoding docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding) · [DSpark blog](https://lmsys.org/blog/2026-07-06-dspark-sglang)
· [BanditSpec (2025)](https://arxiv.org/abs/2505.15141) · [OnlineSpec (2026)](https://arxiv.org/abs/2603.12617)
· [LongSpec (2025)](https://arxiv.org/abs/2502.17421) · [EasySpec (2025)](https://arxiv.org/pdf/2502.02493)
· [llama.cpp `--spec-draft-*` device flags](https://github.com/ggml-org/llama.cpp)

**Heterogeneous parallelism and context parallelism**
[HexGen](https://arxiv.org/abs/2311.11514) · [Hetis](https://dl.acm.org/doi/10.1145/3712285.3759784) · [Tangram](https://arxiv.org/pdf/2606.16907)
· [Cronus](https://arxiv.org/pdf/2509.17357) · [Helix](https://arxiv.org/pdf/2507.07120) · [Medha](https://arxiv.org/pdf/2409.17264)
· [Tessera](https://arxiv.org/pdf/2604.10180) · [Context Parallelism for Million-Token Inference](https://arxiv.org/pdf/2411.01783)
· [vLLM parallelism scaling docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling) · [SGLang DCP issue](https://github.com/sgl-project/sglang/issues/12196)
· [SGLang DCP roadmap](https://github.com/sgl-project/sglang/issues/21788)
· [vLLM context parallel deployment](https://docs.vllm.ai/en/main/serving/context_parallel_deployment.html)
· [llama.cpp `--tensor-split`/`--split-mode`](https://github.com/ggml-org/llama.cpp)
· [ik_llama.cpp parameters](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md)

**MoE expert offloading**
[KTransformers](https://github.com/kvcache-ai/ktransformers) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
· [MoE-Infinity](https://arxiv.org/html/2401.14361) · [SGLang offload PR](https://github.com/sgl-project/sglang/pull/3675)
· [SGLang expert-granular request](https://github.com/sgl-project/sglang/issues/14233) · [vLLM offload config](https://docs.vllm.ai/en/latest/api/vllm/config/offload/)

**KV/attention disaggregation and PD**
[Adrenaline](https://arxiv.org/pdf/2503.20552) · [Mooncake](https://www.usenix.org/system/files/fast25-qin.pdf) · [CrossPool](https://arxiv.org/pdf/2606.24506)
· [SGLang PD disaggregation docs](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

**Checkpoint/restore and memory**
[ServerlessLLM](https://arxiv.org/pdf/2401.14351) · [Tangram](https://arxiv.org/pdf/2512.01357) · [PipeBoost](https://arxiv.org/pdf/2503.17707)
· [vLLM CUDA checkpoint/restore RFC](https://github.com/vllm-project/vllm/issues/34303) · [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
· [vLLM CuMemAllocator](https://docs.vllm.ai/en/latest/api/vllm/device_allocator/cumem/) · [SGLang offload/wake-up PR](https://github.com/sgl-project/sglang/pull/19152)
· [vLLM optimization config](https://docs.vllm.ai/en/stable/configuration/optimization/) · [CRIU](https://criu.org)
· [llama.cpp `--prompt-cache`](https://github.com/ggml-org/llama.cpp) · [llama.cpp server slot save/restore](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
· [llama.cpp `-fit`/`--fit-params-target`](https://github.com/ggml-org/llama.cpp)
· [ik_llama.cpp key features (DeepWiki)](https://deepwiki.com/ikawrakow/ik_llama.cpp/1.1-key-features-and-performance-improvements)

**GGUF**
[vLLM GGUF docs](https://docs.vllm.ai/en/stable/features/quantization/gguf/) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
· [GGUF format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · [Qwen3.5 GGUF evals](https://kaitchup.substack.com/p/more-qwen35-gguf-evals-and-speculative)
· [llama.cpp `src/models/qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
· [llama.cpp `src/models/gemma4.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp)
· [Gemma docs](https://ai.google.dev/gemma) · [llama.cpp MMQ/MMVQ](https://github.com/ggml-org/llama.cpp)
· [ik_llama.cpp key features (DeepWiki)](https://deepwiki.com/ikawrakow/ik_llama.cpp/1.1-key-features-and-performance-improvements)
· [unsloth dynamic GGUF quants](https://huggingface.co/unsloth) · [llama.cpp/ggml K-quant block layout](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

**Determinism**
[SGLang deterministic inference docs](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/deterministic_inference.md)
· [SGLang determinism issue](https://github.com/sgl-project/sglang/issues/10278) · [SGLang determinism blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic)
· [Defeating Nondeterminism in LLM Inference (Thinking Machines)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
· [Batch Speculative Decoding Done Right](https://arxiv.org/pdf/2510.22876) · [Bit-Exact AI Inference Verification](https://arxiv.org/pdf/2606.00279)

**Quantization and scheduling**
[SGLang quantization docs](https://docs.sglang.io/docs/advanced_features/quantization) · [Marlin](https://github.com/IST-DASLab/marlin)
· [SGLang server arguments](https://docs.sglang.io) · [llama.cpp `--prio`/`--prio-batch`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
· [SGLang HiCache docs](https://docs.sglang.io/docs/advanced_features/hicache)
· [vLLM preemption & swap docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
· [SGLang retraction / conservativeness docs](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
· [llama.cpp `--no-kv-offload`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

**Device identity and telemetry**
[NVIDIA MPS](https://docs.nvidia.com/deploy/mps) · [CUDA_VISIBLE_DEVICES](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
· [NVML API](https://docs.nvidia.com/deploy/nvml-api) · [NVIDIA DCGM](https://developer.nvidia.com/dcgm) · [Grafana](https://grafana.com)
· [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/) · [SGLang metrics docs](https://docs.sglang.io)
· [llama.cpp server `--metrics`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
· [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)

**Cross-vendor collectives and low-end/AMD GPU support**
[NCCL](https://docs.nvidia.com/deeplearning/nccl) · [RCCL](https://github.com/ROCm/rccl) · [Gloo](https://github.com/pytorch/gloo)
· [PyTorch distributed backends](https://pytorch.org/docs/stable/distributed.html) · [UCX](https://openucx.org)
· [NVIDIA Turing architecture](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth) · [AMD gfx900 (Vega)](https://gpuopen.com)
· [Triton](https://github.com/triton-lang/triton) · [triton-gcn5 (external)](https://github.com/Said-Akbar/triton-gcn5)
· [Marlin](https://github.com/IST-DASLab/marlin) · [torch `_scaled_mm`](https://pytorch.org/docs/stable/generated/torch.Tensor._scaled_mm.html)
· [compressed-tensors fp8 overview](https://docs.vllm.ai/en/latest/features/quantization/int8.html)
· [llama.cpp `convert_hf_to_gguf.py --fp8-as-q8`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
· [llama.cpp RPC backend README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)

---

## Changelog

Prior passes (2026-07-22 through 2026-07-25) added the llama.cpp/ik_llama.cpp comparison columns
across every row, upgraded HTCCL (row 21) from "planned" to "implemented and validated," added the
fp8 W8A16 dequant fallback (row 22) and Turing/gfx900 base-operation (row 23) rows, restructured
the file from one wide table into a compact matrix + detail-section layout, and replaced
unverified/hedge-worded claims with either measured facts or explicitly flagged open questions.
**This pass (2026-07-25):** condensed every detail section to feature/fork-status/upstream-delta,
removed the reference-list duplication between detail sections and the Sources section below (same
links, one copy), and compressed the guarded/descoped and cross-vendor-bugfix lists — no status
level, measured number, or `unverified`/`not yet merged` marker was removed or weakened in the
process. Sources for the llama.cpp/ik_llama.cpp columns: a local `llama.cpp` checkout
(`/spinning/llm_stuff/llama.cpp-master`, commit `0c4fa7a989`) read directly for CLI flags, model
architecture files, and conversion-script behavior; GitHub API/raw-file fetches against
`ikawrakow/ik_llama.cpp` `main` (no local checkout exists); WebSearch/WebFetch for project docs and
discussion threads where neither repo answered directly.
**This pass (2026-07-26):** replaced the single `Implemented` status token with a three-tier
evidence classification (`Built` / `Boot-checked` / `Cross-checked`, see Status legend), applied
per row to the existing evidence. Added a `falsifikator-geprueft` marker where a row's own test
was red before its fix and green after. Folded in the #190 finding
(`fix/gdn-prefill-determinism`, not yet merged): `gptq_marlin_gemm`, the only fp8 GEMM the RTX 3080
has, is nondeterministic above roughly 109 prompt tokens; flagged the two cross-checks that
included a long fp8@3080 prompt (row 2's #173 G4 chunked prompt, #180 V4's 4th prompt) as excluded
past that boundary. Added three cross-checks: row 2 (#180 V4, Triton vs. flashinfer chain-verify
parity under uneven DCP), row 12 (#124's TP=1-solo-oracle regression harness for the weightless-KV
lane), row 22 (#189's fp8-e4m3 raw-byte decode, bit-exact against `torch`, not yet merged/wired).
Updated the guarded/descoped SWA-DCP Stage B entry with the 2026-07-26 Window 3 finding (H4-H7
green on an unmerged branch pair, #96-H5 needle retrieval Cross-checked against a TP=1 solo
oracle); Stage B stays out of the main matrix since neither branch is merged.
**This pass (2026-07-26, tone):** the overview matrix keeps its SGLang/vLLM/llama.cpp/ik_llama.cpp
columns; every detail section's `Upstream:` line was cut down to a brief note against upstream
sglang only (the fork's actual base), since that is where "what changed" is unambiguous. vLLM,
llama.cpp, and ik_llama.cpp comparisons in detail sections — engine-by-engine capability lists,
mechanism-difference asides, "no equivalent"/"ahead of" framing — are removed; those engines are
compared only in the matrix now. No status level, measured number, or `unverified`/`not yet merged`
marker was touched.
**This pass (2026-07-27, placement):** SWA-DCP Stage B (#96) moved out of "Guarded / descoped"
into its own detail section. Its text already described the Window-4 merge; the surrounding
heading still declared it gated off, so the document contradicted itself. Wording adjusted to
state the merge plainly instead of as a correction to the old status. No status level, evidence
tier, or measured number was changed — the `~+6-10%` figure remains an ex-ante design estimate,
and the ex-post note in the 2026-07-26 entry above ("stays out of the main matrix since neither
branch is merged") is left standing as the record of what was true then. An overview-matrix row
for Stage B is still open.
**This pass (2026-07-27, model coverage and measured numbers):** added the section "Model
coverage, tested combinations, and measured numbers" directly below the overview matrix. It has
four parts: the model families and the quantization formats actually loaded for each; the
family x format x feature combinations that were run, at the document's existing evidence tiers,
with the combinations that have no hardware boot named separately rather than omitted; the
measured numbers on three axes (throughput and per-round cost, capacity, concurrent sessions and
time to first token); and a short note on how the numbers were taken. No existing row, status
level or number was changed. Two figures are carried in their corrected form: the KV-spill
victim's post-restore rate is 76.98 tok/s and the holder's 76.06 (the earlier 15.97 / 63.52
averaged a window that was 90% a second, never-repaired spill), and the #143 Gate 4 result is
+12.6 / +72.7 / +21.5 / +20.5%. Results that fell inside the run-to-run spread — the
`--rank-vocab-ratio` lead, k=4, the content-unpinned KV-split A/B (#203), the cross-rig async
overlap arm, the intra-rig DCP overlap and fusion arms, and DFLASH-vs-NEXTN at long context — are
marked as such and are not presented as gains. Comparisons against vLLM, llama.cpp and
ik_llama.cpp stay confined to the overview matrix; the new section compares only against upstream
sglang and against the fork's own prior arms. Sources: the local measurement package (which stays
out of this repository), the task records for #77, #89, #103, #143, #146, #156, #157, #160, #163,
#173, #180, #198, #199, #203, #204, #210 and #217, and `INTEGRATION_R3_VALIDATION.md`,
`docs/advanced_features/uneven_kv_token_ratio.md` and `docs/advanced_features/pd_disagg_single_node.md`
in this repository.
